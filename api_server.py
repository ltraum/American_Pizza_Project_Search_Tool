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

# Load .env so OPENAI_API_KEY and other env vars are available (api_server does not import config)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import secrets
import json
import time
import csv

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


class ThemeRequest(BaseModel):
    seed: str = Field(..., min_length=1)
    # Base filters (same shape as search filters)
    metadata_filters: Optional[Dict[str, Any]] = None
    # Which part of the interview to use as the theme-mode text
    text_source: str = Field(default="all", pattern="^(all|q1|q2|q3|q4|q5)$")
    # Theme-mode knobs
    max_docs: Optional[int] = Field(default=None, ge=1)
    n_quotes: int = Field(default=3, ge=1, le=10)
    n_bullets: int = Field(default=3, ge=1, le=10)
    bullet_words: str = Field(default="10-18")
    max_themes: int = Field(default=12, ge=1, le=50)
    # LLooM-style graded scoring
    include_bullet_theme_scores: bool = Field(default=True)
    score_batch_size: int = Field(default=6, ge=1, le=25)
    score_get_highlights: bool = Field(default=True)
    max_points_for_scoring: Optional[int] = Field(default=None, ge=1)
    include_theme_summaries: bool = Field(default=True)


class ThemeExploreRequest(ThemeRequest):
    # Exploration-only: how many candidate labels to generate per cluster.
    # (Initial implementation returns 1 candidate/cluster; later steps expand this.)
    candidates_per_cluster: int = Field(default=3, ge=1, le=10)


class ThemeScoreRequest(BaseModel):
    explore_id: str = Field(..., min_length=1)
    # Either provide IDs (from explore.candidate_themes[].theme_id) OR full theme dicts.
    selected_theme_ids: Optional[List[str]] = None
    selected_themes: Optional[List[Dict[str, Any]]] = None
    # Scoring knobs (subset of ThemeRequest)
    score_batch_size: int = Field(default=6, ge=1, le=25)
    max_points_for_scoring: Optional[int] = Field(default=None, ge=1)
    include_theme_summaries: bool = Field(default=True)


class ThemeRunSaveRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    payload: Dict[str, Any] = Field(...)


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
THEME_RUNS_DIR = Path(__file__).parent / "theme_runs"

# In-memory cache: key -> (timestamp, payload)
_THEME_CACHE: Dict[str, tuple] = {}
_THEME_CACHE_TTL_S = int(os.getenv("THEME_CACHE_TTL_S", "3600"))

# Exploration cache:
# - by stable request key (dedupe repeated explore requests)
# - by explore_id (required for subsequent scoring/rescore)
_THEME_EXPLORE_CACHE_BY_KEY: Dict[str, tuple] = {}  # key -> (timestamp, explore_id)
_THEME_EXPLORE_CACHE_BY_ID: Dict[str, tuple] = {}  # explore_id -> (timestamp, payload)


@app.get("/")
async def root(_: None = Depends(require_basic_auth)):
    """Serve the frontend."""
    if not INDEX_HTML.exists():
        raise HTTPException(status_code=404, detail="Frontend not found (missing index.html)")
    return FileResponse(INDEX_HTML)


def _require_openai_key_or_400() -> None:
    if not (os.getenv("OPENAI_API_KEY") or "").strip():
        raise HTTPException(
            status_code=400,
            detail="Theme mode requires OPENAI_API_KEY. Set it in your environment or .env and restart the server.",
        )


def _theme_cache_key(body: ThemeRequest) -> str:
    # Stable JSON key (avoid dict ordering issues)
    def _stable(obj: Any) -> str:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)

    return _stable(
        {
            "seed": body.seed.strip(),
            "metadata_filters": body.metadata_filters,
            "text_source": body.text_source,
            "max_docs": body.max_docs,
            "n_quotes": body.n_quotes,
            "n_bullets": body.n_bullets,
            "bullet_words": body.bullet_words,
            "max_themes": body.max_themes,
            "include_bullet_theme_scores": body.include_bullet_theme_scores,
            "score_batch_size": body.score_batch_size,
            "score_get_highlights": body.score_get_highlights,
            "max_points_for_scoring": body.max_points_for_scoring,
            "include_theme_summaries": body.include_theme_summaries,
        }
    )


def _theme_explore_cache_key(body: ThemeExploreRequest) -> str:
    # Stable JSON key (avoid dict ordering issues)
    def _stable(obj: Any) -> str:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)

    return _stable(
        {
            "seed": body.seed.strip(),
            "metadata_filters": body.metadata_filters,
            "text_source": body.text_source,
            "max_docs": body.max_docs,
            "n_quotes": body.n_quotes,
            "n_bullets": body.n_bullets,
            "bullet_words": body.bullet_words,
            "max_themes": body.max_themes,
            "candidates_per_cluster": body.candidates_per_cluster,
        }
    )


def _cache_get(key: str) -> Optional[Dict[str, Any]]:
    item = _THEME_CACHE.get(key)
    if not item:
        return None
    ts, payload = item
    if (time.time() - ts) > _THEME_CACHE_TTL_S:
        _THEME_CACHE.pop(key, None)
        return None
    return payload


def _cache_set(key: str, payload: Dict[str, Any]) -> None:
    _THEME_CACHE[key] = (time.time(), payload)


def _explore_cache_get_by_id(explore_id: str) -> Optional[Dict[str, Any]]:
    item = _THEME_EXPLORE_CACHE_BY_ID.get(explore_id)
    if not item:
        return None
    ts, payload = item
    if (time.time() - ts) > _THEME_CACHE_TTL_S:
        _THEME_EXPLORE_CACHE_BY_ID.pop(explore_id, None)
        return None
    return payload


def _explore_cache_get_by_key(key: str) -> Optional[Dict[str, Any]]:
    item = _THEME_EXPLORE_CACHE_BY_KEY.get(key)
    if not item:
        return None
    ts, explore_id = item
    if (time.time() - ts) > _THEME_CACHE_TTL_S:
        _THEME_EXPLORE_CACHE_BY_KEY.pop(key, None)
        return None
    payload = _explore_cache_get_by_id(explore_id)
    if payload is None:
        _THEME_EXPLORE_CACHE_BY_KEY.pop(key, None)
    return payload


def _explore_cache_set(key: str, explore_id: str, payload: Dict[str, Any]) -> None:
    now = time.time()
    _THEME_EXPLORE_CACHE_BY_KEY[key] = (now, explore_id)
    _THEME_EXPLORE_CACHE_BY_ID[explore_id] = (now, payload)


def _safe_filename(name: str) -> str:
    # Very small sanitizer for run names/ids.
    keep = []
    for ch in (name or "").strip():
        if ch.isalnum() or ch in ("-", "_", ".", " "):
            keep.append(ch)
    out = "".join(keep).strip().replace(" ", "_")
    return out[:120] if out else "run"


def _run_path(run_id: str) -> Path:
    return THEME_RUNS_DIR / f"{_safe_filename(run_id)}.json"

def _run_artifact_path(run_id: str, suffix: str) -> Path:
    # suffix examples: "points.csv", "themes.csv", "scores.csv"
    return THEME_RUNS_DIR / f"{_safe_filename(run_id)}_{suffix}"


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


@app.post("/api/theme")
async def theme_api(body: ThemeRequest, _: None = Depends(require_basic_auth)):
    _require_openai_key_or_400()
    tool = get_search_tool()
    if tool.documents is None:
        # Defensive: tool should have loaded documents in __init__
        tool.data_loader.load_data()
        tool.documents = tool.data_loader.preprocess_for_search()

    cache_key = _theme_cache_key(body)
    cached = _cache_get(cache_key)
    if cached is not None:
        return {"theme": cached, "cached": True}

    try:
        from theme_mode import ThemeModeParams, run_theme_mode
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Theme mode import failed: {e}")

    # Reuse the existing sentence-transformers model already loaded for semantic search
    try:
        embedder = tool.semantic_search.model
        if embedder is None:
            tool.semantic_search.load_model()
            embedder = tool.semantic_search.model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load embedding model: {e}")

    params = ThemeModeParams(
        seed=body.seed,
        n_quotes=body.n_quotes,
        n_bullets=body.n_bullets,
        bullet_words=body.bullet_words,
        max_docs=body.max_docs,
        max_themes=body.max_themes,
        include_bullet_theme_scores=body.include_bullet_theme_scores,
        score_batch_size=body.score_batch_size,
        score_get_highlights=body.score_get_highlights,
        max_points_for_scoring=body.max_points_for_scoring,
        include_theme_summaries=body.include_theme_summaries,
    )

    try:
        payload = await run_theme_mode(
            documents=tool.documents,
            metadata_filters=body.metadata_filters,
            text_source=body.text_source,
            embedder=embedder,
            params=params,
        )
    except RuntimeError as e:
        # e.g. OPENAI_API_KEY missing
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Theme mode failed: {e}")

    _cache_set(cache_key, payload)
    return {"theme": payload, "cached": False}


@app.post("/api/theme/explore")
async def theme_explore_api(body: ThemeExploreRequest, _: None = Depends(require_basic_auth)):
    """
    Exploration stage for Theme mode:
    - distill -> bullets -> cluster -> label clusters (no graded scoring)
    - return a bank of candidate themes + points for interactive UI
    """
    _require_openai_key_or_400()
    tool = get_search_tool()
    if tool.documents is None:
        # Defensive: tool should have loaded documents in __init__
        tool.data_loader.load_data()
        tool.documents = tool.data_loader.preprocess_for_search()

    cache_key = _theme_explore_cache_key(body)
    cached = _explore_cache_get_by_key(cache_key)
    if cached is not None:
        return {"explore": cached, "cached": True}

    try:
        from theme_mode import ThemeModeParams, run_theme_exploration
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Theme exploration import failed: {e}")

    # Reuse the existing sentence-transformers model already loaded for semantic search
    try:
        embedder = tool.semantic_search.model
        if embedder is None:
            tool.semantic_search.load_model()
            embedder = tool.semantic_search.model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load embedding model: {e}")

    # Exploration explicitly disables graded scoring and LLM review to keep it cheaper.
    params = ThemeModeParams(
        seed=body.seed,
        n_quotes=body.n_quotes,
        n_bullets=body.n_bullets,
        bullet_words=body.bullet_words,
        max_docs=body.max_docs,
        max_themes=body.max_themes,
        auto_review=False,
        include_doc_theme_scores=False,
        include_bullet_theme_scores=False,
        include_theme_summaries=False,
    )

    try:
        raw = await run_theme_exploration(
            documents=tool.documents,
            metadata_filters=body.metadata_filters,
            text_source=body.text_source,
            embedder=embedder,
            params=params,
            candidates_per_cluster=body.candidates_per_cluster,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Theme explore failed: {e}")

    payload = raw if isinstance(raw, dict) else {}
    explore_id = str(payload.get("explore_id") or f"explore_{int(time.time())}")
    payload["explore_id"] = explore_id

    _explore_cache_set(cache_key, explore_id, payload)
    return {"explore": payload, "cached": False}


@app.post("/api/theme/score")
async def theme_score_api(body: ThemeScoreRequest, _: None = Depends(require_basic_auth)):
    """
    Scoring stage for Theme mode:
    - reuse a prior exploration run (points + candidate theme bank)
    - run graded LLM scoring ONLY for selected themes
    """
    _require_openai_key_or_400()

    explore = _explore_cache_get_by_id(body.explore_id)
    if explore is None:
        raise HTTPException(status_code=404, detail="Explore run not found or expired. Re-run exploration.")

    points = explore.get("points") or []
    candidate_themes = explore.get("candidate_themes") or []
    seed = (explore.get("seed") or "").strip()
    if not seed:
        seed = "theme"

    selected: List[Dict[str, Any]] = []
    if isinstance(body.selected_themes, list) and body.selected_themes:
        selected = [t for t in body.selected_themes if isinstance(t, dict)]
    elif isinstance(body.selected_theme_ids, list) and body.selected_theme_ids:
        want = {str(x) for x in body.selected_theme_ids if str(x).strip()}
        for t in candidate_themes:
            if not isinstance(t, dict):
                continue
            tid = str(t.get("theme_id") or "").strip()
            if tid and tid in want:
                selected.append(t)
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide selected_theme_ids or selected_themes.",
        )

    if not selected:
        raise HTTPException(status_code=400, detail="No valid themes selected for scoring.")

    try:
        from theme_mode import ThemeModeParams, run_theme_scoring
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Theme scoring import failed: {e}")

    # Scoring params: keep it focused on scoring knobs.
    params = ThemeModeParams(
        seed=seed,
        include_doc_theme_scores=False,
        include_bullet_theme_scores=True,
        include_theme_summaries=bool(body.include_theme_summaries),
        score_batch_size=body.score_batch_size,
        max_points_for_scoring=body.max_points_for_scoring,
    )

    t0 = time.time()
    try:
        scores = await run_theme_scoring(
            points=points,
            selected_themes=selected,
            seed=seed,
            params=params,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Theme scoring failed: {e}")
    elapsed_s = time.time() - t0

    payload = {
        "run_id": f"score_{int(time.time())}",
        "explore_id": body.explore_id,
        "seed": seed,
        "metadata_filters": explore.get("metadata_filters"),
        "text_source": explore.get("text_source"),
        "counts": {
            **(explore.get("counts") or {}),
            "themes_selected": len(selected),
        },
        "themes": selected,
        "points": points,
        "scores": scores,
        "timing": {"elapsed_seconds": elapsed_s},
    }
    return {"theme": payload}


@app.get("/api/theme/runs")
async def list_theme_runs(_: None = Depends(require_basic_auth)):
    THEME_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    runs = []
    for p in sorted(THEME_RUNS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            runs.append(
                {
                    "run_id": data.get("run_id") or p.stem,
                    "name": data.get("name") or p.stem,
                    "created_at": data.get("created_at"),
                    "seed": (data.get("payload") or {}).get("seed"),
                    "metadata_filters": (data.get("payload") or {}).get("metadata_filters"),
                }
            )
        except Exception:
            continue
    return {"runs": runs}


@app.get("/api/theme/runs/{run_id}")
async def load_theme_run(run_id: str, _: None = Depends(require_basic_auth)):
    p = _run_path(run_id)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Theme run not found")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read run: {e}")


@app.post("/api/theme/runs")
async def save_theme_run(body: ThemeRunSaveRequest, _: None = Depends(require_basic_auth)):
    THEME_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = (body.payload.get("run_id") or "") if isinstance(body.payload, dict) else ""
    if not run_id:
        run_id = f"theme_{int(time.time())}"
    record = {
        "run_id": run_id,
        "name": body.name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "payload": body.payload,
    }
    p = _run_path(run_id)
    try:
        p.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save run: {e}")

    # Also write CSV artifacts for analysis in spreadsheets.
    # These are derived from the saved payload and placed alongside the JSON.
    try:
        payload_root = body.payload if isinstance(body.payload, dict) else {}
        # Support saved bundles that wrap exploration + scoring:
        # { explore: {...}, selection: {...}, scored: {...} }
        payload = payload_root
        if isinstance(payload_root.get("scored"), dict):
            payload = payload_root["scored"]
        points = payload.get("points") or []
        themes = payload.get("themes") or []
        seed = (payload.get("seed") or payload_root.get("seed") or "")

        # Points CSV
        points_path = _run_artifact_path(run_id, "points.csv")
        with points_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "bullet_id",
                    "doc_id",
                    "bullet",
                    "source_text",
                    "start_sentence_idx",
                    "end_sentence_idx",
                    "sentence_count",
                    "cluster_id",
                    "x",
                    "y",
                    "region_of_residence",
                    "income",
                    "pizza_consumption",
                ],
            )
            w.writeheader()
            for pt in points:
                md = (pt.get("metadata") or {}) if isinstance(pt, dict) else {}
                w.writerow(
                    {
                        "bullet_id": pt.get("bullet_id"),
                        "doc_id": pt.get("doc_id"),
                        "bullet": pt.get("bullet"),
                        "source_text": pt.get("source_text") or pt.get("quote") or "",
                        "start_sentence_idx": pt.get("start_sentence_idx"),
                        "end_sentence_idx": pt.get("end_sentence_idx"),
                        "sentence_count": pt.get("sentence_count"),
                        "cluster_id": pt.get("cluster_id"),
                        "x": pt.get("x"),
                        "y": pt.get("y"),
                        "region_of_residence": md.get("region_of_residence"),
                        "income": md.get("income"),
                        "pizza_consumption": md.get("pizza_consumption"),
                    }
                )

        # Themes CSV
        themes_path = _run_artifact_path(run_id, "themes.csv")
        with themes_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "theme_id",
                    "cluster_id",
                    "name",
                    "prompt",
                    "n_points",
                    "representative_bullets",
                    "representative_quotes",
                ],
            )
            w.writeheader()
            for th in themes:
                w.writerow(
                    {
                        "theme_id": th.get("theme_id"),
                        "cluster_id": th.get("cluster_id"),
                        "name": th.get("name"),
                        "prompt": th.get("prompt"),
                        "n_points": th.get("n_points"),
                        "representative_bullets": " | ".join(th.get("representative_bullets") or []),
                        "representative_quotes": " | ".join(th.get("representative_quotes") or []),
                    }
                )

        # Quote/Bullet × Theme score CSV
        # Prefer graded LLM scoring rows if present; fall back to cluster membership (0/1).
        score_rows = None
        if isinstance(payload.get("scores"), dict) and isinstance(payload["scores"].get("bullet_theme_scores"), list):
            score_rows = payload["scores"]["bullet_theme_scores"]

        scores_path = _run_artifact_path(run_id, "scores.csv")
        with scores_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "doc_id",
                    "text",
                    "bullet_id",
                    "bullet",
                    # Match the sample spreadsheet column naming.
                    "concept_id",
                    "concept_name",
                    "concept_prompt",
                    "score",
                    "rationale",
                    "highlight",
                    "concept_seed",
                ],
            )
            w.writeheader()
            if score_rows is not None:
                for r in score_rows:
                    if not isinstance(r, dict):
                        continue
                    w.writerow(
                        {
                            "doc_id": r.get("doc_id"),
                            "text": r.get("text"),
                            "bullet_id": r.get("bullet_id"),
                            "bullet": r.get("bullet"),
                            "concept_id": r.get("concept_id"),
                            "concept_name": r.get("concept_name"),
                            "concept_prompt": r.get("concept_prompt"),
                            "score": r.get("score"),
                            "rationale": r.get("rationale"),
                            "highlight": r.get("highlight"),
                            "concept_seed": r.get("concept_seed") or seed,
                        }
                    )
            else:
                for pt in points:
                    if not isinstance(pt, dict):
                        continue
                    src = (pt.get("source_text") or pt.get("quote") or "").strip()
                    highlight = ""
                    if src:
                        toks = src.split()
                        highlight = " ".join(toks[:18]) + ("…" if len(toks) > 18 else "")
                    for th in themes:
                        if not isinstance(th, dict):
                            continue
                        score = 1.0 if str(pt.get("cluster_id")) == str(th.get("cluster_id")) else 0.0
                        w.writerow(
                            {
                                "doc_id": pt.get("doc_id"),
                                "text": src,
                                "bullet_id": pt.get("bullet_id"),
                                "bullet": pt.get("bullet") or "",
                                "concept_id": th.get("cluster_id"),
                                "concept_name": th.get("name") or "",
                                "concept_prompt": th.get("prompt") or "",
                                "score": score,
                                "rationale": "",
                                "highlight": highlight,
                                "concept_seed": seed,
                            }
                        )
    except Exception:
        # Artifacts are best-effort; do not fail the save if CSV generation fails.
        pass

    return {"ok": True, "run_id": run_id}


@app.delete("/api/theme/runs/{run_id}")
async def delete_theme_run(run_id: str, _: None = Depends(require_basic_auth)):
    p = _run_path(run_id)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Theme run not found")
    try:
        p.unlink()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete run: {e}")
    return {"ok": True}


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
