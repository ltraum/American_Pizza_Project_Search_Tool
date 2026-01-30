"""
FastAPI backend for Pizza Interview Search Tool.
Exposes search, metadata, and data-info as REST APIs so the UI only hits the server
when searching or loading filters—no full-page reruns on every interaction.
"""
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from search_result_utils import enrich_fulltext_results as _enrich_fulltext_results

import os
import secrets

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field

# Lazy singleton – import and create only when first request arrives
_search_tool = None


def get_search_tool():
    global _search_tool
    if _search_tool is None:
        from search_tool import PizzaSearchTool
        tool = PizzaSearchTool(skip_indexing=False)
        tool.semantic_search.load_model()
        _search_tool = tool
    return _search_tool


# ---- Distribution histogram (mirrors app.py logic) ----
def _get_participant_id(metadata: Dict, result_id: Any) -> Any:
    if not metadata:
        return result_id
    for key in ["participant_id", "participant", "id", "participant_number"]:
        if key in metadata and metadata.get(key) is not None:
            return metadata[key]
    return result_id


def compute_distribution_data(results: Dict[str, Any], query: str = "") -> Optional[Dict[str, Any]]:
    """Compute histogram bins/counts for prevalence chart. Expects results to already have normalized_score."""
    if not results or not results.get("results"):
        return None
    all_results = results["results"]

    participant_max: Dict[Any, float] = {}
    for r in all_results:
        mid = _get_participant_id(r.get("metadata") or {}, r.get("id"))
        participant_max[mid] = max(participant_max.get(mid, 0), r.get("normalized_score", 0))

    max_scores = list(participant_max.values())
    if not max_scores:
        return None

    nbins = 30
    bins = [i / nbins for i in range(nbins + 1)]
    counts = [0] * nbins
    for s in max_scores:
        idx = min(int(s * nbins), nbins - 1) if s < 1.0 else nbins - 1
        counts[idx] += 1

    return {
        "bins": bins,
        "counts": counts,
        "title": f'Prevalence of "{query}" across interviews.' if query else "Prevalence across interviews.",
    }


# ---- Pydantic models ----
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    search_type: str = Field(default="hybrid", pattern="^(hybrid|fulltext|semantic)$")
    semantic_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    metadata_filters: Optional[Dict[str, Any]] = None


class MetadataFilterValue(BaseModel):
    operator: str
    value: Any


# ---- Lifespan: optional preload ----
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Could preload get_search_tool() here to avoid first-request delay
    yield


app = FastAPI(title="Pizza Interview Search API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Optional simple auth gate (shared-link deployments) ----
_basic = HTTPBasic(auto_error=False)
_AUTH_USERNAME = os.getenv("UI_AUTH_USERNAME", "").strip()
_AUTH_PASSWORD = os.getenv("UI_AUTH_PASSWORD", "").strip()


def _auth_enabled() -> bool:
    return bool(_AUTH_USERNAME and _AUTH_PASSWORD)


def require_basic_auth(credentials: Optional[HTTPBasicCredentials] = Depends(_basic)) -> None:
    """
    If UI_AUTH_USERNAME and UI_AUTH_PASSWORD are set, require HTTP Basic auth.
    Applied per-route to cover both the UI (/) and APIs (/api/*).
    """
    if not _auth_enabled():
        return

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Basic"},
        )

    ok_user = secrets.compare_digest(credentials.username or "", _AUTH_USERNAME)
    ok_pass = secrets.compare_digest(credentials.password or "", _AUTH_PASSWORD)
    if not (ok_user and ok_pass):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )


# Frontend: index.html in project root
INDEX_HTML = Path(__file__).parent / "index.html"


@app.get("/")
async def root(_: None = Depends(require_basic_auth)):
    """Serve the frontend."""
    if not INDEX_HTML.exists():
        raise HTTPException(status_code=404, detail="Frontend not found (missing index.html)")
    return FileResponse(INDEX_HTML)


_STOPWORDS = {
    # Small stopword list: enough to avoid boosting on "the", etc.
    "a", "an", "and", "are", "as", "at", "be", "but", "by",
    "for", "from", "has", "have", "he", "her", "hers", "him", "his",
    "i", "if", "in", "into", "is", "it", "its",
    "me", "my", "of", "on", "or", "our", "ours", "she", "so",
    "that", "the", "their", "them", "then", "there", "these", "they", "this", "those", "to",
    "was", "we", "were", "what", "when", "where", "which", "who", "why", "with", "you", "your",
}


def _extract_query_terms(query: str) -> List[str]:
    terms = re.findall(r"\b\w+\b", (query or "").lower())
    # Keep content-ish terms only; avoid over-penalizing short/common words.
    return [t for t in terms if len(t) >= 3 and t not in _STOPWORDS]


def _extract_quoted_phrases(query: str) -> List[str]:
    phrases = [p.strip() for p in re.findall(r'"([^"]+)"', query or "") if p.strip()]
    # Cap to avoid pathological inputs
    return phrases[:5]


def _term_variants(term: str) -> List[str]:
    """
    Very small morphology helper for UI-facing matching/boosting/highlighting.
    This is NOT semantic stemming; it just helps plural/singular queries line up
    with the underlying text (e.g. "pineapples" vs "pineapple").
    """
    t = (term or "").lower().strip()
    if not t:
        return []
    out = {t}
    # singularize common plural forms
    if len(t) > 4 and t.endswith("ies"):
        out.add(t[:-3] + "y")
    if len(t) > 4 and t.endswith("es"):
        out.add(t[:-2])
    if len(t) > 3 and t.endswith("s"):
        out.add(t[:-1])
    # pluralize simple singular forms
    if len(t) > 2 and not t.endswith("s"):
        out.add(t + "s")
    if len(t) > 2 and t.endswith("y"):
        out.add(t[:-1] + "ies")
    # keep only reasonable tokens
    out = {x for x in out if x and len(x) >= 3 and x not in _STOPWORDS}
    return sorted(out)


def _whole_word_match_count(text: str, terms: List[str]) -> int:
    if not text or not terms:
        return 0
    tl = text.lower()
    c = 0
    for t in terms:
        # Whole-word match: "pineapple" should not match "pineapplesauce"
        variants = _term_variants(t) or [t]
        if any(re.search(rf"\b{re.escape(v)}\b", tl) for v in variants):
            c += 1
    return c


def _phrase_match(text: str, phrases: List[str]) -> bool:
    if not text or not phrases:
        return False
    tl = text.lower()
    for p in phrases:
        pl = p.lower()
        # Whole phrase boundary match at edges; simple but effective.
        if re.search(rf"\b{re.escape(pl)}\b", tl):
            return True
    return False


def _add_normalized_scores(results: Dict[str, Any], query: str) -> None:
    """
    Mutate results['results'] in place, adding normalized_score (0–1) to each item.

    In hybrid mode, we want:
    - exact (whole-word) matches for the original query to rank first
    - high-scoring full-text hits to rank above semantic-only "near misses"
    """
    arr = results.get("results") or []
    fulltext_scores = [r.get("score", 0) for r in arr if r.get("search_type") == "fulltext"]
    max_ft = max(fulltext_scores) if fulltext_scores else 1.0

    terms = _extract_query_terms(query)
    phrases = _extract_quoted_phrases(query)
    is_single_term_query = len(terms) == 1 and not phrases

    for r in arr:
        st = r.get("search_type", "hybrid")
        s = r.get("score", 0) or 0.0

        # Base normalization: keep legacy behavior but clamp to [0, 1].
        if st == "semantic":
            base = float(s)
        elif st == "fulltext":
            base = float(s) / float(max_ft) if max_ft and max_ft > 0 else float(s) / 10.0
        else:
            sem = r.get("semantic_score")
            base = float(sem) if sem is not None else (float(s) / float(max_ft) if max_ft and max_ft > 0 else float(s) / 10.0)
        base = min(1.0, max(0.0, base))

        # Compute exact-match signals (prefer whole-word matches).
        text_for_match = (r.get("text") or r.get("full_text") or "").strip()
        phrase_hit = _phrase_match(text_for_match, phrases)
        term_hits = _whole_word_match_count(text_for_match, terms)
        has_any_term = term_hits > 0
        has_all_terms = bool(terms) and term_hits == len(terms)

        # Boost exact matches; gently bias toward full-text when scores are close.
        boost = 0.0
        if phrase_hit:
            boost += 0.25
        elif has_all_terms:
            boost += 0.18
        elif has_any_term:
            boost += 0.10

        if st == "fulltext":
            boost += 0.03

        # Penalize semantic "false positives" for keyword-ish queries (like "pineapple").
        # Keep them, but push them down below literal matches.
        if st == "semantic" and terms and not has_any_term and not phrase_hit:
            base *= 0.55 if is_single_term_query else 0.75

        # Penalize fulltext hits that don't contain any term variant; this avoids
        # misleading 1.000 badges for documents that don't actually mention the term.
        if st == "fulltext" and terms and not has_any_term and not phrase_hit:
            base *= 0.45 if is_single_term_query else 0.7

        r["normalized_score"] = min(1.0, max(0.0, base + boost))


@app.post("/api/search")
async def search_api(body: SearchRequest, _: None = Depends(require_basic_auth)):
    """Run search and return results plus distribution data for the chart."""
    tool = get_search_tool()
    results = tool.search(
        query=body.query,
        search_type=body.search_type,
        limit=None,
        semantic_threshold=body.semantic_threshold if (body.semantic_threshold or 0) > 0 else None,
        metadata_filters=body.metadata_filters,
    )
    _add_normalized_scores(results, body.query)
    _enrich_fulltext_results(results, body.query)
    # Ensure UI shows best (exact-match boosted) results first even when not grouped.
    results["results"] = sorted(
        results.get("results") or [],
        key=lambda r: (r.get("normalized_score") or 0),
        reverse=True,
    )
    distribution = compute_distribution_data(results, query=body.query)
    return {"results": results, "distribution": distribution}


@app.get("/api/metadata/fields")
async def metadata_fields(_: None = Depends(require_basic_auth)):
    """List available metadata field names (excluding q1–q5 response columns)."""
    tool = get_search_tool()
    all_fields = tool.get_available_metadata_fields()
    filtered = [f for f in all_fields if not re.match(r"q[1-5]_response", f, re.IGNORECASE)]
    return {"fields": filtered}


@app.get("/api/metadata/values")
async def metadata_values(field: str, limit: int = 100, _: None = Depends(require_basic_auth)):
    """Get unique values for a metadata field (for dropdowns)."""
    tool = get_search_tool()
    values = tool.get_metadata_field_values(field, limit=limit)
    return {"field": field, "values": values}


@app.get("/api/data-info")
async def data_info(_: None = Depends(require_basic_auth)):
    """Return data shape, columns, sample for the Data Info view."""
    tool = get_search_tool()
    info = tool.get_data_info()
    # Ensure sample is JSON-serializable
    import pandas as pd
    sample = info.get("sample", [])
    if sample:
        info["sample"] = [{k: (v if pd.notna(v) else None) for k, v in r.items()} for r in sample]
    return info


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False)
