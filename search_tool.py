"""
Main search tool combining full-text and semantic search
Provides unified interface for searching pizza interview data
"""
import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import logging
import pandas as pd

from config import EXCEL_FILE, CHUNK_WINDOW_SIZE, CHUNK_OVERLAP
from data_loader import PizzaDataLoader
from fulltext_search import FullTextSearch
from semantic_search import SemanticSearch
from search_result_utils import filter_metadata_prefixed_results

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PizzaSearchTool:
    """Unified search tool for pizza interview data"""
    
    def __init__(
        self,
        excel_file: Path = None,
        rebuild_index: bool = False,
        skip_indexing: bool = False
    ):
        self.excel_file = excel_file or EXCEL_FILE
        self.data_loader = PizzaDataLoader(self.excel_file)
        self.fulltext_search = FullTextSearch()
        self.semantic_search = SemanticSearch()
        self.documents = None
        
        # Load and index data (unless skipping for info-only mode)
        if not skip_indexing:
            self._initialize(rebuild_index)
        else:
            # Just load data for info display (no preprocessing needed)
            self.data_loader.load_data()
    
    def _initialize(self, rebuild_index: bool = False):
        """Initialize search indices"""
        logger.info("Initializing search tool...")
        
        # Load data
        self.data_loader.load_data()
        self.documents = self.data_loader.preprocess_for_search()
        
        # Build indices
        logger.info("Building full-text search index...")
        self.fulltext_search.build_index(self.documents, recreate=rebuild_index)
        
        logger.info("Building semantic search index...")
        self.semantic_search.build_index(
            self.documents,
            recreate=rebuild_index,
            window_size=CHUNK_WINDOW_SIZE,
            overlap=CHUNK_OVERLAP,
        )
        
        logger.info("Search tool initialized successfully!")
    
    def get_full_text_by_doc_id(self, doc_id: Any) -> str:
        """Return full narrative text for a document by its id (row index string)."""
        if self.documents is None:
            return ""
        doc_id = str(doc_id) if doc_id is not None else ""
        for doc in self.documents:
            if str(doc.get("id")) == doc_id:
                return (doc.get("text") or "").strip()
        return ""
    
    def get_full_text_by_respondent_id(self, respondent_id: Any) -> str:
        """Return full narrative text for a document by respondent/participant id from metadata."""
        if self.documents is None or respondent_id is None:
            return ""
        keys = ("participant_id", "participant", "id", "participant_number")
        for doc in self.documents:
            meta = doc.get("metadata") or {}
            for k in keys:
                if meta.get(k) is not None and str(meta.get(k)) == str(respondent_id):
                    return (doc.get("text") or "").strip()
        return ""
    
    def _enrich_results_with_full_text(self, results_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Set full_text from original data for results that have doc_id or respondent id but no full_text."""
        for r in results_list:
            if r.get("full_text"):
                continue
            doc_id = r.get("doc_id") or r.get("id")
            full = self.get_full_text_by_doc_id(doc_id) if doc_id is not None else ""
            if not full:
                meta = r.get("metadata") or {}
                rid = meta.get("participant_id") or meta.get("participant") or meta.get("id") or meta.get("participant_number")
                full = self.get_full_text_by_respondent_id(rid) if rid is not None else ""
            if full:
                r["full_text"] = full
        return results_list
    
    def _apply_metadata_filters(
        self,
        results: List[Dict[str, Any]],
        filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Apply metadata filters to search results
        
        Args:
            results: List of search results
            filters: Dictionary of filter criteria
                Format: {
                    'field_name': {
                        'operator': 'equals' | 'contains' | 'greater_than' | 'less_than' | 'between' | 'in',
                        'value': value or [min, max] for between
                    }
                }
        
        Returns:
            Filtered list of results
        """
        if not filters:
            return results
        
        filtered_results = []
        
        for result in results:
            metadata = result.get('metadata', {})
            match = True
            
            for field_name, filter_spec in filters.items():
                if not isinstance(filter_spec, dict):
                    # Simple value match (backward compatibility)
                    filter_spec = {'operator': 'equals', 'value': filter_spec}
                
                operator = filter_spec.get('operator', 'equals')
                value = filter_spec.get('value')
                
                # Skip if value is empty or None
                if value is None or value == '' or (isinstance(value, list) and len(value) == 0):
                    continue
                
                # Get field value from metadata
                field_value = metadata.get(field_name)
                
                # Skip if field doesn't exist and operator requires a value
                if field_value is None or pd.isna(field_value):
                    if operator in ['equals', 'contains', 'greater_than', 'less_than', 'in']:
                        match = False
                        break
                    continue
                
                # Apply filter based on operator
                if operator == 'equals':
                    if str(field_value).lower() != str(value).lower():
                        match = False
                        break
                
                elif operator == 'contains':
                    if str(value).lower() not in str(field_value).lower():
                        match = False
                        break
                
                elif operator == 'greater_than':
                    try:
                        if float(field_value) <= float(value):
                            match = False
                            break
                    except (ValueError, TypeError):
                        match = False
                        break
                
                elif operator == 'less_than':
                    try:
                        if float(field_value) >= float(value):
                            match = False
                            break
                    except (ValueError, TypeError):
                        match = False
                        break
                
                elif operator == 'between':
                    try:
                        min_val, max_val = value
                        field_float = float(field_value)
                        if not (min_val <= field_float <= max_val):
                            match = False
                            break
                    except (ValueError, TypeError):
                        match = False
                        break
                
                elif operator == 'in' or operator == 'includes':
                    # For 'includes', check if field value matches any of the selected values
                    # Support both single value (backward compatibility) and list of values
                    if isinstance(value, list):
                        # Multi-select: field value must match one of the selected values
                        if not any(str(field_value).lower() == str(v).lower() for v in value):
                            match = False
                            break
                    else:
                        # Single value (backward compatibility)
                        if str(field_value).lower() != str(value).lower():
                            match = False
                            break
                
                elif operator == 'not_includes':
                    # For 'not_includes', exclude if field value matches any of the selected values
                    # Support both single value (backward compatibility) and list of values
                    if isinstance(value, list):
                        # Multi-select: exclude if field value matches any of the selected values
                        if any(str(field_value).lower() == str(v).lower() for v in value):
                            match = False
                            break
                    else:
                        # Single value (backward compatibility)
                        if str(field_value).lower() == str(value).lower():
                            match = False
                            break
                
                elif operator == 'not_equals':
                    if str(field_value).lower() == str(value).lower():
                        match = False
                        break
            
            if match:
                filtered_results.append(result)
        
        return filtered_results
    
    def get_available_metadata_fields(self) -> List[str]:
        """Get list of available metadata fields from the data"""
        if self.documents is None:
            self.data_loader.load_data()
            self.documents = self.data_loader.preprocess_for_search()
        
        if not self.documents:
            return []
        
        # Get all unique metadata keys from documents
        all_keys = set()
        for doc in self.documents:
            if 'metadata' in doc:
                all_keys.update(doc['metadata'].keys())
        
        return sorted(list(all_keys))
    
    def get_metadata_field_values(self, field_name: str, limit: int = 100) -> List[Any]:
        """Get unique values for a metadata field (for dropdowns)"""
        if self.documents is None:
            self.data_loader.load_data()
            self.documents = self.data_loader.preprocess_for_search()
        
        if not self.documents:
            return []
        
        values = set()
        for doc in self.documents:
            if 'metadata' in doc:
                value = doc['metadata'].get(field_name)
                if value is not None and not pd.isna(value):
                    values.add(value)
        
        # Convert to list and sort
        sorted_values = sorted(list(values), key=lambda x: str(x))
        return sorted_values[:limit]
    
    def search(
        self,
        query: str,
        search_type: str = "hybrid",
        limit: int = 10,
        semantic_threshold: float = None,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform search
        
        Args:
            query: Search query
            search_type: "fulltext", "semantic", or "hybrid"
            limit: Maximum number of results
            semantic_threshold: Minimum similarity score for semantic search
            metadata_filters: Optional dictionary of metadata filters
                Format: {
                    'field_name': {
                        'operator': 'equals' | 'contains' | 'greater_than' | 'less_than' | 'between' | 'in',
                        'value': value or [min, max] for between
                    }
                }
        
        Returns:
            Dictionary with search results and metadata
        """
        results = {
            "query": query,
            "search_type": search_type,
            "results": [],
            "fulltext_results": [],
            "semantic_results": [],
            "expansion_fallback_used": False,
            "search_status": {
                "query_expansion": "skipped",  # "success" | "fallback" | "skipped"
                "fulltext": "ok",               # "ok" | "error"
                "fulltext_error": None,
                "semantic": "ok",               # "ok" | "error"
                "semantic_error": None,
            },
        }
        
        # Handle limit=None (return all results)
        search_limit = limit * 2 if limit is not None else None
        fulltext_results = []
        semantic_results = []
        
        if search_type in ["fulltext", "hybrid"]:
            logger.info(f"Performing full-text search: {query}")
            try:
                fulltext_results = self.fulltext_search.search(query, limit=search_limit)
                results["fulltext_results"] = fulltext_results
            except Exception as e:
                logger.exception("Full-text search failed")
                results["search_status"]["fulltext"] = "error"
                results["search_status"]["fulltext_error"] = str(e)
        
        if search_type in ["semantic", "hybrid"]:
            logger.info(f"Performing semantic search: {query}")
            try:
                # Default to "none" aggregation (return all chunks) for transparency
                semantic_results = self.semantic_search.search(
                    query, 
                    limit=search_limit,
                    score_threshold=semantic_threshold,
                    aggregate_chunks="none"  # Return all matching chunks (most transparent)
                )
                results["semantic_results"] = semantic_results
                results["expansion_fallback_used"] = getattr(
                    self.semantic_search, "_expansion_fallback_used", False
                )
                # Query expansion status: success (AI expanded), fallback (rules used), skipped
                if getattr(self.semantic_search, "_expansion_success", False):
                    results["search_status"]["query_expansion"] = "success"
                elif results["expansion_fallback_used"]:
                    results["search_status"]["query_expansion"] = "fallback"
                else:
                    results["search_status"]["query_expansion"] = "skipped"
            except Exception as e:
                logger.exception("Semantic search failed")
                results["search_status"]["semantic"] = "error"
                results["search_status"]["semantic_error"] = str(e)
        
        # Combine results based on search type
        if search_type == "fulltext":
            # Deduplicate fulltext results by document ID
            seen_ids = set()
            deduplicated = []
            for result in fulltext_results:
                result_id = result['id']
                # Also check doc_id if present to avoid duplicates
                doc_id = result.get('doc_id', result_id)
                if result_id not in seen_ids and doc_id not in seen_ids:
                    result['search_type'] = 'fulltext'
                    deduplicated.append(result)
                    seen_ids.add(result_id)
                    if doc_id != result_id:
                        seen_ids.add(doc_id)
            # Sort by score (descending) to maintain relevance order
            deduplicated.sort(key=lambda x: x.get('score', 0), reverse=True)
            results["results"] = self._enrich_results_with_full_text(
                deduplicated if limit is None else deduplicated[:limit]
            )
        elif search_type == "semantic":
            # Label semantic results
            for result in semantic_results:
                result['search_type'] = 'semantic'
            results["results"] = self._enrich_results_with_full_text(
                semantic_results if limit is None else semantic_results[:limit]
            )
        else:  # hybrid
            # Merge and deduplicate results
            # For chunk-based results, we need to be careful about deduplication
            # If semantic results are chunks (not aggregated), we dedupe by chunk_id
            # If semantic results are aggregated by doc, we dedupe by doc_id
            seen_ids = set()
            seen_doc_ids = set()
            combined = []
            # For hybrid, avoid letting a semantic "near miss" hide an exact keyword hit.
            # Only treat a doc as "covered" by semantic chunks if the chunk includes
            # at least one whole-word query term.
            query_terms = [
                t for t in re.findall(r"\b\w+\b", (query or "").lower())
                if len(t) >= 3
            ]

            def _variants(t: str) -> List[str]:
                t = (t or "").lower().strip()
                if not t:
                    return []
                out = {t}
                if len(t) > 4 and t.endswith("ies"):
                    out.add(t[:-3] + "y")
                if len(t) > 4 and t.endswith("es"):
                    out.add(t[:-2])
                if len(t) > 3 and t.endswith("s"):
                    out.add(t[:-1])
                if len(t) > 2 and not t.endswith("s"):
                    out.add(t + "s")
                if len(t) > 2 and t.endswith("y"):
                    out.add(t[:-1] + "ies")
                return sorted({x for x in out if x and len(x) >= 3})

            def _has_whole_word_term(text: str) -> bool:
                if not text or not query_terms:
                    return False
                tl = text.lower()
                for t in query_terms:
                    vs = _variants(t) or [t]
                    if any(re.search(rf"\b{re.escape(v)}\b", tl) for v in vs):
                        return True
                return False
            
            # Add semantic results first (typically higher quality)
            for result in semantic_results:
                # For chunks: use chunk id; for aggregated: use doc_id
                if result.get('is_chunk'):
                    # Individual chunk - dedupe by chunk id
                    if result['id'] not in seen_ids:
                        result['search_type'] = 'semantic'
                        combined.append(result)
                        seen_ids.add(result['id'])
                        # Only block fulltext duplicates when this semantic chunk actually
                        # contains a whole-word query term; otherwise let fulltext surface.
                        if 'doc_id' in result and (not query_terms or _has_whole_word_term(result.get("text") or "")):
                            seen_doc_ids.add(result['doc_id'])
                else:
                    # Aggregated result - dedupe by doc_id or id
                    result_id = result.get('doc_id') or result['id']
                    if result_id not in seen_ids and result_id not in seen_doc_ids:
                        result['search_type'] = 'semantic'
                        combined.append(result)
                        seen_ids.add(result['id'])
                        if 'doc_id' in result and (not query_terms or _has_whole_word_term(result.get("text") or "")):
                            seen_doc_ids.add(result['doc_id'])
            
            # Add fulltext results that weren't already included
            for result in fulltext_results:
                result_id = result['id']
                # Check both id and doc_id to avoid duplicates
                if result_id not in seen_ids and result_id not in seen_doc_ids:
                    result['search_type'] = 'fulltext'
                    combined.append(result)
                    seen_ids.add(result_id)
            
            # Sort by: exact whole-word match first, then fulltext over semantic, then score
            def _sort_key(r: Dict[str, Any]):
                txt = (r.get("text") or "").lower()
                exact = 1 if (query_terms and _has_whole_word_term(txt)) else 0
                st = r.get("search_type")
                st_bias = 1 if st == "fulltext" else 0
                return (exact, st_bias, r.get("score", 0))

            combined.sort(key=_sort_key, reverse=True)
            results["results"] = self._enrich_results_with_full_text(
                combined if limit is None else combined[:limit]
            )
        
        # Apply metadata filters if provided
        if metadata_filters:
            logger.info(f"Applying metadata filters: {metadata_filters}")
            original_count = len(results["results"])
            results["results"] = self._apply_metadata_filters(results["results"], metadata_filters)
            new_count = len(results["results"])
            logger.info(f"Filtered results: {original_count} -> {new_count}")
            if original_count > 0 and new_count == 0:
                logger.warning("All results were removed by metadata filters. Try clearing filters or relaxing filter values.")
        
        # Drop results whose excerpt is metadata + narrative (so only narrative excerpts surface)
        before = len(results["results"])
        results["results"] = filter_metadata_prefixed_results(results["results"])
        if before > len(results["results"]):
            logger.info(f"Filtered out {before - len(results['results'])} metadata-prefixed excerpts")

        # Apply a global minimum score threshold in semantic+hybrid modes.
        #
        # Semantic results: `score` is cosine similarity in [0, 1].
        # Full-text results: Whoosh scores are unbounded, so we normalize to [0, 1]
        # within the current result set by dividing by max full-text score.
        #
        # Keep a result if EITHER semantic similarity >= threshold OR normalized
        # full-text relevance >= threshold. Drop only when BOTH are below.
        if semantic_threshold is not None:
            try:
                thr = float(semantic_threshold)
            except (TypeError, ValueError):
                thr = None
            if thr is not None and thr > 0:
                ft_scores = [
                    float(r.get("score", 0) or 0)
                    for r in (results.get("results") or [])
                    if r.get("search_type") == "fulltext"
                ]
                max_ft = max(ft_scores) if ft_scores else 0.0

                def _ft_norm(score: float) -> float:
                    if max_ft and max_ft > 0:
                        return max(0.0, min(1.0, float(score) / float(max_ft)))
                    return 0.0

                kept = []
                removed = 0
                for r in results.get("results") or []:
                    st = r.get("search_type")
                    sem_sim = None
                    ft_norm = None
                    if st == "semantic":
                        try:
                            sem_sim = float(r.get("score", 0) or 0.0)
                        except (TypeError, ValueError):
                            sem_sim = 0.0
                        sem_sim = max(0.0, min(1.0, sem_sim))
                        r["semantic_similarity"] = sem_sim
                    elif st == "fulltext":
                        try:
                            ft_norm = _ft_norm(float(r.get("score", 0) or 0.0))
                        except (TypeError, ValueError):
                            ft_norm = 0.0
                        r["fulltext_score_normalized"] = ft_norm

                    sem_ok = (sem_sim is not None) and (sem_sim >= thr)
                    ft_ok = (ft_norm is not None) and (ft_norm >= thr)
                    if sem_ok or ft_ok:
                        kept.append(r)
                    else:
                        removed += 1

                if removed:
                    logger.info(
                        f"Applied minimum score threshold {thr:.3f}: "
                        f"removed {removed} results (kept {len(kept)})"
                    )
                results["results"] = kept
        
        results["total_results"] = len(results["results"])
        return results
    
    def search_fulltext(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Full-text search only"""
        return self.search(query, search_type="fulltext", limit=limit)["results"]
    
    def search_semantic(self, query: str, limit: int = 10, threshold: float = None) -> List[Dict[str, Any]]:
        """Semantic search only"""
        return self.search(query, search_type="semantic", limit=limit, semantic_threshold=threshold)["results"]
    
    def search_hybrid(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Hybrid search (combines both methods)"""
        return self.search(query, search_type="hybrid", limit=limit)["results"]
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get information about the loaded data"""
        return self.data_loader.get_data_info()
    
    def print_results(self, results: Dict[str, Any], detailed: bool = False):
        """Print search results in a readable format"""
        print(f"\n{'='*80}")
        print(f"Search Query: {results['query']}")
        print(f"Search Type: {results['search_type']}")
        print(f"Total Results: {results['total_results']}")
        print(f"{'='*80}\n")
        
        for i, result in enumerate(results['results'], 1):
            print(f"Result {i} (Rank {result.get('rank', i)}, Score: {result.get('score', 0):.4f})")
            print(f"ID: {result['id']}")
            if result.get('doc_id') and result['doc_id'] != result['id']:
                print(f"Document ID: {result['doc_id']}")
            if result.get('is_chunk'):
                print(f"Type: Sentence-pair chunk (sentences {result.get('start_sentence_idx', '?')}-{result.get('end_sentence_idx', '?')})")
                if result.get('matching_chunks_count', 0) > 1:
                    print(f"  → {result['matching_chunks_count']} matching chunks found in this document")
            if detailed:
                text_preview = result['text'][:500]
                if result.get('best_matching_chunk'):
                    print(f"Best matching chunk: {result['best_matching_chunk'][:300]}...")
                print(f"Text: {text_preview}...")
                if result.get('metadata'):
                    print(f"Metadata: {json.dumps(result['metadata'], indent=2)}")
            else:
                print(f"Text: {result['text'][:200]}...")
            print("-" * 80)


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(description="Pizza Interview Search Tool")
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument(
        "--type",
        choices=["fulltext", "semantic", "hybrid"],
        default="hybrid",
        help="Search type (default: hybrid)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of results (default: 10)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Minimum similarity score for semantic search"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild search indices"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show data information"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed results"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    
    args = parser.parse_args()
    
    # Show data info if requested (skip indexing for faster info display)
    if args.info:
        try:
            search_tool = PizzaSearchTool(skip_indexing=True)
            info = search_tool.get_data_info()
            print(json.dumps(info, indent=2, default=str))
            return 0
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return 1
    
    # Initialize search tool (with indexing)
    try:
        search_tool = PizzaSearchTool(rebuild_index=args.rebuild)
    except Exception as e:
        logger.error(f"Error initializing search tool: {e}")
        return 1
    
    # Perform search if query provided
    if args.query:
        try:
            results = search_tool.search(
                query=args.query,
                search_type=args.type,
                limit=args.limit,
                semantic_threshold=args.threshold
            )
            
            if args.json:
                print(json.dumps(results, indent=2, default=str))
            else:
                search_tool.print_results(results, detailed=args.detailed)
        except Exception as e:
            logger.error(f"Error performing search: {e}")
            return 1
    else:
        parser.print_help()
    
    return 0


if __name__ == "__main__":
    exit(main())
