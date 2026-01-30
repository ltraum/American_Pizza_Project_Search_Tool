"""
Semantic search implementation using sentence transformers and ChromaDB
Uses locally hostable models compatible with Windows
Now supports overlapping two-sentence window chunking for improved relevance
"""
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import torch

from config import (
    VECTOR_DB_DIR,
    SEMANTIC_MODEL_NAME,
    SEMANTIC_SEARCH_TOP_K,
    WINDOWS_USE_CPU,
    EMBEDDINGS_CACHE_DIR,
    EMBED_BATCH_SIZE,
    ENABLE_QUERY_EXPANSION,
    QUERY_EXPANSION_MODEL,
    MULTI_QUERY_RRF,
    MULTI_QUERY_RRF_K,
    MULTI_QUERY_MAX_PHRASES,
    USE_QUERY_PASSAGE_PREFIX,
    QUERY_PREFIX,
    PASSAGE_PREFIX,
    CHUNK_WINDOW_SIZE,
    CHUNK_OVERLAP,
)
from sentence_chunker import chunk_document_into_sentence_pairs, create_multi_level_chunks
from query_expander import QueryExpander

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _apply_query_passage_prefix(text: str, is_query: bool) -> str:
    """Apply E5/BGE-style prefix when USE_QUERY_PASSAGE_PREFIX is enabled."""
    if not USE_QUERY_PASSAGE_PREFIX or not text:
        return text
    return (QUERY_PREFIX + text) if is_query else (PASSAGE_PREFIX + text)


def _reciprocal_rank_fusion(
    list_of_results: List[Dict[str, Any]],
    k: int = 60,
) -> List[tuple]:
    """
    Fuse multiple query result lists using Reciprocal Rank Fusion (RRF).
    Each result is (id, document, metadata, distance). Returns list of
    (id, document, metadata, best_similarity) sorted by RRF score descending.
    """
    # id -> (rrf_score, best_similarity, doc, metadata)
    fused: Dict[str, tuple] = {}
    for res in list_of_results:
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for rank_1based, (cid, doc, meta, dist) in enumerate(
            zip(ids, docs, metas, dists), start=1
        ):
            sim = 1.0 - dist
            rrf_inc = 1.0 / (k + rank_1based)
            if cid not in fused:
                fused[cid] = (rrf_inc, sim, doc, meta)
            else:
                old_rrf, old_sim, old_doc, old_meta = fused[cid]
                fused[cid] = (old_rrf + rrf_inc, max(old_sim, sim), old_doc, old_meta)
    # Sort by RRF score desc, return (id, doc, metadata, best_similarity)
    ordered = sorted(
        [(cid, doc, meta, sim) for cid, (rrf, sim, doc, meta) in fused.items()],
        key=lambda x: fused[x[0]][0],
        reverse=True,
    )
    return [(cid, doc, meta, sim) for cid, doc, meta, sim in ordered]


class SemanticSearch:
    """Semantic search using embeddings and vector similarity"""
    
    def __init__(
        self, 
        model_name: str = None,
        vector_db_dir: Path = None,
        use_cpu: bool = None,
        enable_query_expansion: bool = None
    ):
        self.model_name = model_name or SEMANTIC_MODEL_NAME
        self.vector_db_dir = vector_db_dir or VECTOR_DB_DIR
        self.use_cpu = use_cpu if use_cpu is not None else WINDOWS_USE_CPU
        self.enable_query_expansion = enable_query_expansion if enable_query_expansion is not None else ENABLE_QUERY_EXPANSION
        
        # Initialize model
        self.model = None
        self.device = "cpu" if self.use_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize vector database
        self.client = None
        self.collection = None
        self.collection_name = "pizza_interviews"
        
        # Initialize query expander if enabled (uses AI by default if API key available)
        self.query_expander = None
        if self.enable_query_expansion:
            self.query_expander = QueryExpander(
                use_llm=True,  # Use AI-based expansion
                model=QUERY_EXPANSION_MODEL
            )
        
    def load_model(self):
        """Load the embedding model"""
        if self.model is not None:
            return
        
        logger.info(f"Loading semantic model: {self.model_name}")
        logger.info(f"Using device: {self.device}")
        
        try:
            # Load model with cache directory
            cache_dir = str(EMBEDDINGS_CACHE_DIR)
            self.model = SentenceTransformer(
                self.model_name,
                cache_folder=cache_dir,
                device=self.device
            )
            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def initialize_vector_db(self):
        """Initialize ChromaDB vector database"""
        if self.client is not None:
            return
        
        # Create vector DB directory
        self.vector_db_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client (persistent, local)
        self.client = chromadb.PersistentClient(
            path=str(self.vector_db_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        logger.info(f"Initialized vector database at {self.vector_db_dir}")
    
    def get_or_create_collection(self, recreate: bool = False):
        """Get or create the collection for storing embeddings"""
        if self.client is None:
            self.initialize_vector_db()
        
        # Delete existing collection if recreating
        if recreate:
            try:
                self.client.delete_collection(self.collection_name)
                logger.info(f"Deleted existing collection: {self.collection_name}")
            except Exception:
                pass
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
        except Exception:
            # Use cosine distance so similarity = 1 - distance is correct.
            # Chroma defaults to L2; for normalized embeddings cosine is appropriate.
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={
                    "description": "Pizza interview and survey data",
                    "hnsw:space": "cosine",
                }
            )
            logger.info(f"Created new collection: {self.collection_name} (cosine)")
    
    def build_index(
        self, 
        documents: List[Dict[str, Any]], 
        recreate: bool = True,
        use_sentence_chunks: bool = True,
        window_size: int = None,
        overlap: int = None
    ):
        """
        Build semantic search index by generating embeddings.
        
        Chunking uses config CHUNK_WINDOW_SIZE / CHUNK_OVERLAP by default. Smaller windows
        (e.g. window_size=1) improve recall for short queries like "family" or "spicy".
        
        Args:
            documents: List of document dictionaries
            recreate: Whether to recreate the index
            use_sentence_chunks: If True, use sentence-window chunking (default: True)
            window_size: Sentences per chunk (default: from config, typically 1 for single-sentence)
            overlap: Sentences to overlap between chunks (default: from config)
        """
        if window_size is None:
            window_size = CHUNK_WINDOW_SIZE
        if overlap is None:
            overlap = CHUNK_OVERLAP
        if self.model is None:
            self.load_model()
        
        if self.collection is None:
            self.get_or_create_collection(recreate=recreate)
        
        if recreate:
            self.get_or_create_collection(recreate=True)
        else:
            # Use existing index when corpus hasn't changed (persistent ChromaDB)
            existing_count = self.collection.count()
            if existing_count > 0:
                logger.info(
                    f"Using existing vector index ({existing_count} items). "
                    "Skip embedding generation. To rebuild, use rebuild_index=True."
                )
                return
        
        if use_sentence_chunks:
            logger.info(f"Building index with sentence-pair chunking (window={window_size}, overlap={overlap})...")
            all_chunks = []
            all_ids = []
            all_metadatas = []
            
            for doc in documents:
                doc_id = doc['id']
                doc_text = doc['text']
                doc_metadata = doc.get('metadata', {})
                row_index = doc.get('row_index', -1)
                
                # Create sentence-pair chunks
                chunks = chunk_document_into_sentence_pairs(
                    doc_text,
                    window_size=window_size,
                    overlap=overlap
                )
                
                # If no chunks created (empty/short text), use full document
                if not chunks:
                    chunks = [{'text': doc_text, 'start_sentence_idx': 0, 'end_sentence_idx': 0, 'sentence_count': 1}]
                
                # Create a chunk entry for each sentence pair
                for chunk_idx, chunk in enumerate(chunks):
                    chunk_id = f"{doc_id}_chunk_{chunk_idx}"
                    all_chunks.append(chunk['text'])
                    all_ids.append(chunk_id)
                    
                    # Store metadata linking back to original document
                    chunk_metadata = {
                        'row_index': row_index,
                        'doc_id': doc_id,
                        'chunk_index': chunk_idx,
                        'start_sentence_idx': chunk['start_sentence_idx'],
                        'end_sentence_idx': chunk['end_sentence_idx'],
                        'sentence_count': chunk['sentence_count'],
                        'is_chunk': True
                    }
                    
                    # Add original document metadata
                    for key, value in doc_metadata.items():
                        if isinstance(value, (str, int, float, bool)):
                            chunk_metadata[key] = value
                        else:
                            chunk_metadata[key] = str(value)
                    
                    all_metadatas.append(chunk_metadata)
            
            logger.info(f"Created {len(all_chunks)} sentence-pair chunks from {len(documents)} documents")
            texts = all_chunks
            ids = all_ids
            metadatas = all_metadatas
        else:
            # Original behavior: document-level indexing
            logger.info(f"Generating embeddings for {len(documents)} documents (document-level)...")
            texts = [doc['text'] for doc in documents]
            ids = [doc['id'] for doc in documents]
            
            metadatas = []
            for doc in documents:
                metadata = {
                    'row_index': doc.get('row_index', -1),
                    'is_chunk': False
                }
                if 'metadata' in doc:
                    for key, value in doc['metadata'].items():
                        if isinstance(value, (str, int, float, bool)):
                            metadata[key] = value
                        else:
                            metadata[key] = str(value)
                metadatas.append(metadata)
        
        # Generate embeddings in batches for efficiency
        # When using E5/BGE-style models, prepend passage prefix to each text
        batch_size = EMBED_BATCH_SIZE
        all_embeddings = []
        texts_to_encode = [
            _apply_query_passage_prefix(t, is_query=False) for t in texts
        ] if USE_QUERY_PASSAGE_PREFIX else texts
        
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        for i in range(0, len(texts_to_encode), batch_size):
            batch_texts = texts_to_encode[i:i+batch_size]
            batch_embeddings = self.model.encode(
                batch_texts,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            all_embeddings.extend(batch_embeddings.tolist())
        
        # Add to collection
        logger.info("Adding embeddings to vector database...")
        self.collection.add(
            embeddings=all_embeddings,
            documents=texts,
            ids=ids,
            metadatas=metadatas
        )
        
        logger.info(f"Indexed {len(texts)} items in vector database")
    
    def search(
        self, 
        query: str, 
        limit: int = None,
        score_threshold: float = None,
        aggregate_chunks: str = "none"
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search with improved relevance using sentence-pair chunks.
        
        When using sentence-pair chunking, this method searches for matching sentence pairs
        and can aggregate results by document using different strategies.
        
        Args:
            query: Search query string
            limit: Maximum number of results (default: SEMANTIC_SEARCH_TOP_K)
            score_threshold: Minimum similarity score (0-1)
            aggregate_chunks: Aggregation strategy:
                - "none": Return all matching chunks (no aggregation) - most transparent
                - "best": Return best matching chunk per document (simple max)
                - "max": Same as "best" - uses max score per document
                - "avg_top3": Return document with average of top 3 chunk scores
                - "sum": Return document with sum of all chunk scores (boost for multiple matches)
        
        Returns:
            List of matching chunks/documents with similarity scores
        """
        if self.model is None:
            self.load_model()
        
        if self.collection is None:
            self.get_or_create_collection()
        
        # If limit is None, use a very large number to get all results
        # Otherwise use the provided limit or default to SEMANTIC_SEARCH_TOP_K
        if limit is None:
            search_limit = 10000  # Very large number to get all results
            result_limit = None  # No limit on final results
        else:
            limit = limit or SEMANTIC_SEARCH_TOP_K
            # Search for more results if we're aggregating (to account for multiple chunks per doc)
            # Use a larger multiplier to ensure we get good coverage
            search_limit = limit * 5 if aggregate_chunks != "none" else limit
            result_limit = limit
        
        # Expand query and decide single- vs multi-query retrieval
        original_query = query
        self._expansion_fallback_used = False  # set True when AI expansion failed and rule-based was used
        self._expansion_success = False       # set True when AI expansion was used successfully
        use_multi_query_rrf = (
            self.enable_query_expansion
            and self.query_expander
            and MULTI_QUERY_RRF
        )
        
        if use_multi_query_rrf:
            # Multi-query + RRF: embed each phrase separately, search each, fuse by RRF
            phrases, used_rule_fallback = self.query_expander.get_expansion_phrases(
                query, max_phrases=MULTI_QUERY_MAX_PHRASES
            )
            self._expansion_fallback_used = used_rule_fallback
            self._expansion_success = len(phrases) > 1 and not used_rule_fallback
            # Troubleshooting: print expanded phrases to terminal
            print(f"[Query expansion] original: {original_query!r}")
            print(f"[Query expansion] {len(phrases)} phrase(s):")
            for i, p in enumerate(phrases, 1):
                print(f"  {i}. {p!r}")
            if len(phrases) > 1:
                logger.info(f"Multi-query RRF: {len(phrases)} phrases for '{original_query[:50]}...'")
            list_of_raw = []
            for phrase in phrases:
                qtext = _apply_query_passage_prefix(phrase, is_query=True)
                emb = self.model.encode(qtext, convert_to_numpy=True).tolist()
                raw = self.collection.query(
                    query_embeddings=[emb],
                    n_results=search_limit,
                    include=['documents', 'metadatas', 'distances']
                )
                list_of_raw.append(raw)
            fused_tuples = _reciprocal_rank_fusion(list_of_raw, k=MULTI_QUERY_RRF_K)
            # Build chunk_results from fused list (same shape as single-query path)
            chunk_results = []
            for i, (chunk_id, chunk_text, metadata, similarity) in enumerate(fused_tuples):
                if score_threshold and similarity < score_threshold:
                    continue
                is_chunk = metadata.get('is_chunk', False)
                doc_id = metadata.get('doc_id', chunk_id) if is_chunk else chunk_id
                chunk_results.append({
                    'id': chunk_id,
                    'doc_id': doc_id,
                    'text': chunk_text,
                    'score': similarity,
                    'rank': i + 1,
                    'metadata': {k: v for k, v in metadata.items()
                                 if k not in ['row_index', 'doc_id', 'chunk_index', 'is_chunk']},
                    'row_index': metadata.get('row_index', -1),
                    'is_chunk': is_chunk,
                    'chunk_index': metadata.get('chunk_index', None),
                    'start_sentence_idx': metadata.get('start_sentence_idx', None),
                    'end_sentence_idx': metadata.get('end_sentence_idx', None),
                    'sentence_count': metadata.get('sentence_count', None)
                })
            results_have_chunks = True
        else:
            # Single-query path (existing behavior)
            if self.enable_query_expansion and self.query_expander:
                query = self.query_expander.expand_query(query, max_expansions=10)
                self._expansion_success = query.strip() != original_query.strip()
                if query != original_query:
                    print(f"[Query expansion] original: {original_query!r}")
                    print(f"[Query expansion] expanded: {query!r}")
                    logger.info(f"Query expanded from '{original_query}' to '{query[:200]}...'")
            qtext = _apply_query_passage_prefix(query, is_query=True)
            query_embedding = self.model.encode(qtext, convert_to_numpy=True).tolist()
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=search_limit,
                include=['documents', 'metadatas', 'distances']
            )
            results_have_chunks = results['ids'] and len(results['ids'][0]) > 0
            chunk_results = []
            if results_have_chunks:
                for i, (chunk_id, chunk_text, metadata, distance) in enumerate(zip(
                    results['ids'][0],
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    similarity = 1 - distance
                    if score_threshold and similarity < score_threshold:
                        continue
                    is_chunk = metadata.get('is_chunk', False)
                    doc_id = metadata.get('doc_id', chunk_id) if is_chunk else chunk_id
                    chunk_result = {
                        'id': chunk_id,
                        'doc_id': doc_id,
                        'text': chunk_text,
                        'score': similarity,
                        'rank': i + 1,
                        'metadata': {k: v for k, v in metadata.items()
                                     if k not in ['row_index', 'doc_id', 'chunk_index', 'is_chunk']},
                        'row_index': metadata.get('row_index', -1),
                        'is_chunk': is_chunk,
                        'chunk_index': metadata.get('chunk_index', None),
                        'start_sentence_idx': metadata.get('start_sentence_idx', None),
                        'end_sentence_idx': metadata.get('end_sentence_idx', None),
                        'sentence_count': metadata.get('sentence_count', None)
                    }
                    chunk_results.append(chunk_result)
        
        # Shared: build formatted_results from chunk_results (both single- and multi-query paths)
        formatted_results = []
        if chunk_results:
            if aggregate_chunks == "none":
                formatted_results = chunk_results if result_limit is None else chunk_results[:result_limit]
            else:
                doc_groups = {}
                for chunk in chunk_results:
                    doc_id = chunk['doc_id']
                    if doc_id not in doc_groups:
                        doc_groups[doc_id] = []
                    doc_groups[doc_id].append(chunk)
                aggregated_results = []
                for doc_id, chunks in doc_groups.items():
                    chunks_sorted = sorted(chunks, key=lambda x: x['score'], reverse=True)
                    if aggregate_chunks in ["best", "max"]:
                        best_chunk = chunks_sorted[0]
                        if len(chunks) > 1:
                            best_chunk['matching_chunks_count'] = len(chunks)
                            best_chunk['all_chunk_scores'] = [c['score'] for c in chunks_sorted]
                        aggregated_results.append(best_chunk)
                    elif aggregate_chunks == "avg_top3":
                        top_chunks = chunks_sorted[:3]
                        avg_score = sum(c['score'] for c in top_chunks) / len(top_chunks)
                        best_chunk = top_chunks[0].copy()
                        best_chunk['score'] = avg_score
                        best_chunk['aggregation_method'] = 'avg_top3'
                        best_chunk['matching_chunks_count'] = len(chunks)
                        best_chunk['top_chunk_scores'] = [c['score'] for c in top_chunks]
                        aggregated_results.append(best_chunk)
                    elif aggregate_chunks == "sum":
                        total_score = sum(c['score'] for c in chunks)
                        best_chunk = chunks_sorted[0].copy()
                        best_chunk['score'] = total_score
                        best_chunk['aggregation_method'] = 'sum'
                        best_chunk['matching_chunks_count'] = len(chunks)
                        best_chunk['all_chunk_scores'] = [c['score'] for c in chunks_sorted]
                        aggregated_results.append(best_chunk)
                aggregated_results.sort(key=lambda x: x['score'], reverse=True)
                formatted_results = aggregated_results if result_limit is None else aggregated_results[:result_limit]
                for i, result in enumerate(formatted_results):
                    result['rank'] = i + 1
        
        logger.info(f"Found {len(formatted_results)} semantic results for query: {original_query} (aggregation: {aggregate_chunks})")
        return formatted_results
    
    def hybrid_search(
        self,
        query: str,
        fulltext_results: List[Dict[str, Any]],
        limit: int = None
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search: combine fulltext and semantic results
        Re-ranks fulltext results using semantic similarity
        
        Now uses sentence-pair chunking for more accurate semantic scoring:
        - Finds the best matching sentence pair within each fulltext result
        - Uses that pair's similarity score for re-ranking
        """
        if self.model is None:
            self.load_model()
        
        limit = limit or SEMANTIC_SEARCH_TOP_K
        
        # Expand query if enabled
        original_query = query
        if self.enable_query_expansion and self.query_expander:
            query = self.query_expander.expand_query(query, max_expansions=10)
            if query != original_query:
                logger.info(f"Query expanded from '{original_query}' to '{query[:200]}...'")
        
        qtext = _apply_query_passage_prefix(query, is_query=True)
        query_embedding = self.model.encode(qtext, convert_to_numpy=True)
        
        # Re-rank fulltext results by semantic similarity
        for result in fulltext_results:
            result_text = result['text']
            chunks = chunk_document_into_sentence_pairs(
                result_text, window_size=CHUNK_WINDOW_SIZE, overlap=CHUNK_OVERLAP
            )
            if not chunks:
                chunks = [{'text': result_text}]
            best_similarity = 0.0
            best_chunk_text = result_text
            for chunk in chunks:
                chunk_text = chunk['text']
                ptext = _apply_query_passage_prefix(chunk_text, is_query=False)
                chunk_embedding = self.model.encode(ptext, convert_to_numpy=True)
                similarity = float(torch.nn.functional.cosine_similarity(
                    torch.tensor(query_embedding).unsqueeze(0),
                    torch.tensor(chunk_embedding).unsqueeze(0)
                ))
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_chunk_text = chunk_text
            
            result['semantic_score'] = best_similarity
            result['best_matching_chunk'] = best_chunk_text
            # Combined score (weighted average)
            # Normalize fulltext score to 0-1 range if needed
            fulltext_score = result.get('score', 0)
            if fulltext_score > 1:
                # Whoosh scores can be > 1, normalize roughly
                fulltext_score = min(fulltext_score / 10.0, 1.0)
            result['combined_score'] = 0.5 * fulltext_score + 0.5 * best_similarity
        
        # Sort by combined score
        fulltext_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return fulltext_results[:limit]
