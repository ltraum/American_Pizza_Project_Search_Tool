"""
Full-text search implementation using Whoosh
Pure Python, Windows compatible
"""
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from whoosh import index
from whoosh.fields import Schema, TEXT, ID, STORED
from whoosh.analysis import StemmingAnalyzer
from whoosh.qparser import QueryParser, MultifieldParser
from whoosh.query import Or, And
import logging

from config import SEARCH_INDEX_DIR, FULLTEXT_INDEX_NAME, FULLTEXT_SEARCH_TOP_K
from data_loader import PizzaDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FullTextSearch:
    """Full-text search using Whoosh index"""
    
    def __init__(self, index_dir: Path = None):
        self.index_dir = index_dir or SEARCH_INDEX_DIR
        self.index_name = FULLTEXT_INDEX_NAME
        self.index_path = self.index_dir / self.index_name
        self.schema = None
        self.ix = None
        
    def create_schema(self, sample_doc: Dict[str, Any]) -> Schema:
        """Create Whoosh schema based on document structure"""
        # Base schema with text and metadata
        schema_fields = {
            'id': ID(stored=True, unique=True),
            # Stemming makes simple morphology match (mushroom vs mushrooms),
            # without needing fuzzy query expansion.
            'text': TEXT(stored=True, analyzer=StemmingAnalyzer()),  # Main searchable text
        }
        
        # Add metadata fields as stored (not searchable, but retrievable)
        if 'metadata' in sample_doc:
            for key in sample_doc['metadata'].keys():
                # Store metadata but don't make all fields searchable
                # to keep index size manageable
                schema_fields[f'meta_{key}'] = STORED
        
        return Schema(**schema_fields)
    
    def build_index(self, documents: List[Dict[str, Any]], recreate: bool = True):
        """Build or update the search index"""
        if not documents:
            raise ValueError("No documents provided for indexing")
        
        # Create schema from first document
        if self.schema is None:
            self.schema = self.create_schema(documents[0])
        
        # Remove existing index directory first when recreating, then (re)create it.
        # Doing this in the right order avoids transient \"missing _MAIN_0.toc\" errors
        # from Whoosh when the directory is removed while an index is being created.
        if recreate and self.index_path.exists():
            import shutil
            shutil.rmtree(self.index_path)
            logger.info(f"Removed existing index at {self.index_path}")
        
        # (Re)create index directory
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # Create index
        index_already_existed = index.exists_in(str(self.index_path))
        if not index_already_existed:
            logger.info(f"Creating new index at {self.index_path}")
            self.ix = index.create_in(str(self.index_path), self.schema)
        else:
            logger.info(f"Opening existing index at {self.index_path}")
            self.ix = index.open_dir(str(self.index_path))
        
        # Skip re-indexing when using a pre-built index (recreate=False and index existed)
        if index_already_existed and not recreate:
            logger.info("Using existing Whoosh index, skipping document indexing")
            return
        
        # Index documents
        writer = self.ix.writer()
        indexed_count = 0
        
        for doc in documents:
            # Prepare document for indexing
            index_doc = {
                'id': doc['id'],
                'text': doc['text'],
            }
            
            # Add metadata fields
            if 'metadata' in doc:
                for key, value in doc['metadata'].items():
                    index_doc[f'meta_{key}'] = str(value) if value is not None else ""
            
            try:
                writer.add_document(**index_doc)
                indexed_count += 1
            except Exception as e:
                logger.warning(f"Error indexing document {doc['id']}: {e}")
        
        writer.commit()
        logger.info(f"Indexed {indexed_count} documents")
    
    def load_index(self):
        """Load existing index"""
        if not index.exists_in(str(self.index_path)):
            raise FileNotFoundError(f"Index not found at {self.index_path}. Run build_index() first.")
        
        if self.schema is None:
            # Try to infer schema from existing index
            self.ix = index.open_dir(str(self.index_path))
        else:
            self.ix = index.open_dir(str(self.index_path))
        
        logger.info(f"Loaded index from {self.index_path}")
    
    def search(
        self,
        query: str,
        limit: int = None,
        fields: List[str] = None,
        fallback_to_fuzzy: bool = True,
        fuzzy_maxdist: int = 2,
        min_results_for_fallback: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Perform full-text search
        
        Args:
            query: Search query string
            limit: Maximum number of results (default: FULLTEXT_SEARCH_TOP_K)
            fields: Fields to search in (default: ['text'])
        
        Returns:
            List of matching documents with scores
        """
        if self.ix is None:
            self.load_index()
        
        # If limit is None, use a very large number to get all results
        if limit is None:
            limit = 10000  # Very large number to get all results
        else:
            limit = limit or FULLTEXT_SEARCH_TOP_K
        fields = fields or ['text']
        
        # Create query parser
        if len(fields) == 1:
            parser = QueryParser(fields[0], schema=self.ix.schema)
        else:
            parser = MultifieldParser(fields, schema=self.ix.schema)
        
        # Parse query (exact / stemmed matching)
        try:
            q = parser.parse(query)
        except Exception as e:
            logger.warning(f"Query parsing error: {e}. Using simple query.")
            q = parser.parse(query.replace('"', ''))
        
        def _format(whoosh_results, match_type: str) -> List[Dict[str, Any]]:
            formatted = []
            seen_ids = set()
            for result in whoosh_results:
                doc_id = result["id"]
                if doc_id in seen_ids:
                    continue
                seen_ids.add(doc_id)
                doc = {
                    "id": doc_id,
                    "text": result.get("text", ""),
                    "score": float(result.score),
                    "rank": int(result.rank) + 1,
                    "match_type": match_type,  # "exact" | "fuzzy"
                }
                metadata = {}
                for key in result.keys():
                    if key.startswith("meta_"):
                        orig_key = key.replace("meta_", "")
                        metadata[orig_key] = result[key]
                doc["metadata"] = metadata
                formatted.append(doc)
            return formatted

        # Search (exact/stemmed), with optional fuzzy fallback for typos.
        formatted_results: List[Dict[str, Any]] = []
        with self.ix.searcher() as searcher:
            exact = searcher.search(q, limit=limit)
            formatted_results = _format(exact, match_type="exact")

            if (
                fallback_to_fuzzy
                and query
                and '"' not in query  # avoid breaking phrase/advanced syntax
                and len(formatted_results) < max(0, int(min_results_for_fallback))
            ):
                # Fuzzy fallback: allows small letter differences (edit distance).
                # Stemming already handles pluralization; fuzzy is mainly for typos.
                def _token_dist(token: str) -> int:
                    # Keep fuzzy conservative for short tokens (reduces false positives).
                    # Longer tokens can tolerate 2 edits (e.g. "pinapples" -> "pineapple").
                    t = (token or "").strip()
                    if len(t) <= 4:
                        return 0
                    if len(t) <= 6:
                        return min(int(fuzzy_maxdist), 1)
                    return int(fuzzy_maxdist)

                fuzzy_parts = []
                for word in query.split():
                    w = word.strip()
                    if not w:
                        continue
                    d = _token_dist(w)
                    fuzzy_parts.append(f"{w}~{d}" if d > 0 else w)
                fuzzy_query = " ".join(fuzzy_parts)
                try:
                    fq = parser.parse(fuzzy_query)
                    fuzzy = searcher.search(fq, limit=limit)
                    fuzzy_formatted = _format(fuzzy, match_type="fuzzy")
                    # Merge: keep highest score per doc_id.
                    by_id = {r["id"]: r for r in formatted_results}
                    for r in fuzzy_formatted:
                        if r["id"] not in by_id or r["score"] > by_id[r["id"]]["score"]:
                            by_id[r["id"]] = r
                    formatted_results = sorted(by_id.values(), key=lambda x: x.get("score", 0), reverse=True)
                except Exception as e:
                    logger.warning(f"Fuzzy fallback query parsing/search error: {e}")
        
        logger.info(f"Found {len(formatted_results)} unique results for query: {query}")
        return formatted_results
    
    def search_phrase(self, phrase: str, limit: int = None) -> List[Dict[str, Any]]:
        """Search for exact phrase"""
        return self.search(f'"{phrase}"', limit=limit, fallback_to_fuzzy=False)
    
    def search_fuzzy(self, query: str, limit: int = None, maxdist: int = 1) -> List[Dict[str, Any]]:
        """Fuzzy search (handles typos)"""
        fuzzy_query = " ".join([f"{word}~{int(maxdist)}" for word in query.split()])
        return self.search(fuzzy_query, limit=limit, fallback_to_fuzzy=False)
