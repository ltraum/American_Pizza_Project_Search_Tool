"""
Theme mode (LLooM-style) induction pipeline.

This module is intentionally self-contained (no imports from lloom_source_material)
so it can run in the FastAPI app without Jupyter/anywidget dependencies.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ---- Prompts (adapted from LLooM) ----

FILTER_PROMPT_TEMPLATE = """I have the following TEXT EXAMPLE:
{ex}

Please extract {n_quotes} QUOTES exactly copied from this EXAMPLE that are related to {seed_upper}. Please respond ONLY with a valid JSON in the following format:
{{
  "relevant_quotes": ["<QUOTE_1>", "<QUOTE_2>"]
}}
"""

SUMMARIZE_PROMPT_TEMPLATE = """I have the following TEXT EXAMPLE:
{ex}

TASK:
Summarize the main point(s) of this EXAMPLE related to {seed_upper} into {n_bullets} bullet points.

STYLE (important):
- Each bullet should be a *thick-description code* (not a generic topic label).
- Be concrete and socially situated: include at least one of (A) setting/context, (B) social unit/roles, (C) occasion/ritual/timing, (D) meaning/norm/constraint.
- Avoid vague umbrella words like "experience", "community", "culture", "important" unless you anchor them to specifics.
- It is OK if a bullet is narrow; we prefer specificity over broadness.
- Keep each bullet to about {n_words} words.

Please respond ONLY with a valid JSON in the following format:
{{
  "bullets": ["<BULLET_1>", "<BULLET_2>"]
}}
"""

SUMMARIZE_CHUNK_PROMPT_TEMPLATE = """I have the following TEXT EXCERPT (2-3 sentences):
{ex}

TASK:
Write ONE bullet point that captures the main point(s) of this EXCERPT related to {seed_upper}.

STYLE (important):
- The bullet should be a *thick-description code* (not a generic topic label).
- Be concrete and socially situated: include at least one of (A) setting/context, (B) social unit/roles, (C) occasion/ritual/timing, (D) meaning/norm/constraint.
- Avoid vague umbrella words like "experience", "community", "culture", "important" unless you anchor them to specifics.
- Be faithful to the excerpt: do NOT invent details not present in the text.
- Keep the bullet to about {n_words} words.

Please respond ONLY with a valid JSON in the following format:
{{
  "bullet": "<BULLET>"
}}
"""

SYNTHESIZE_PROMPT_TEMPLATE = """I have this set of bullet point summaries of text examples:
{examples_json}

TASK:
Write ONE specific, distinct THEME that unifies these examples.
The theme MUST BE RELATED TO {seed_upper}.

GUIDELINES (important):
- Prefer fine-grained, socially situated categories over broad umbrella topics.
- It is OK if the theme applies to only a subset of examples; do NOT create a theme that would apply to nearly all examples.
- The THEME NAME should be concrete (not just a synonym of {seed_upper}); include at least one of: setting/context, social unit, occasion, or meaning/norm.
- Avoid near-duplicate themes (synonyms with slightly different wording).

For the theme, write:
- a 3-7 word NAME
- an associated 1-sentence PROMPT that could take in a new text example and determine whether the theme applies (criteria should be specific enough to exclude many non-matching cases)
- 2-4 example_ids for items that BEST exemplify the theme

Please respond ONLY with a valid JSON in the following format:
{{
  "theme": {{
    "name": "<THEME_NAME>",
    "prompt": "<THEME_PROMPT>",
    "example_ids": ["<EXAMPLE_ID_1>", "<EXAMPLE_ID_2>"]
  }}
}}
"""

SYNTHESIZE_MULTI_PROMPT_TEMPLATE = """I have this set of bullet point summaries of text examples:
{examples_json}

TASK:
Generate {k} specific, distinct THEME CANDIDATES that could plausibly unify these examples.
Each theme MUST BE RELATED TO {seed_upper}.

GUIDELINES (important):
- Prefer fine-grained, socially situated categories over broad umbrella topics.
- Each candidate should be meaningfully different (not synonyms / paraphrases).
- It is OK if a theme applies to only a subset of examples; do NOT create themes that would apply to nearly all examples.
- The THEME NAME should be concrete (not just a synonym of {seed_upper}); include at least one of: setting/context, social unit, occasion, or meaning/norm.

For EACH theme candidate, write:
- a 3-7 word NAME
- an associated 1-sentence PROMPT that could take in a new text example and determine whether the theme applies
- 2-4 example_ids for items that BEST exemplify the theme (choose from the example_id values provided)

Please respond ONLY with a valid JSON in the following format:
{{
  "themes": [
    {{
      "name": "<THEME_NAME>",
      "prompt": "<THEME_PROMPT>",
      "example_ids": ["<EXAMPLE_ID_1>", "<EXAMPLE_ID_2>"]
    }}
  ]
}}
"""

REVIEW_REMOVE_PROMPT_TEMPLATE = """I have this set of candidate themes generated from clustered text examples:
{themes_list}

TASK:
Identify any themes that should be REMOVED because they are:
- too generic/broad (would apply to most examples), OR
- too vague (no concrete social context), OR
- near-duplicates of another theme (synonyms / paraphrases).

NOTES:
- It is OK for a theme to apply to only a subset of examples.
- Prefer retaining fine-grained, socially situated themes.

Please respond ONLY with valid JSON in the following format:
{{
  "remove_cluster_ids": [1, 2]
}}
"""

REVIEW_SELECT_PROMPT_TEMPLATE = """I have this set of candidate themes generated from clustered text examples:
{themes_list}

TASK:
Select AT MOST {max_themes} themes to include in the final set.

SELECTION CRITERIA:
- Not too generic/vague (should not describe most examples)
- Not overlapping / redundant with each other (avoid synonyms)
- Together, cover a range of distinct, socially situated patterns

Please respond ONLY with valid JSON in the following format:
{{
  "selected_cluster_ids": [3, 4, 5]
}}
"""

SCORE_HIGHLIGHT_PROMPT_TEMPLATE = """CONTEXT:
I have the following text examples in a JSON:
{examples_json}

I also have a pattern named {concept_name} with the following PROMPT:
{concept_prompt}

TASK:
For each example, please evaluate the PROMPT by generating a 1-sentence RATIONALE of your thought process and providing a resulting ANSWER of ONE of the following multiple-choice options, including just the letter:
- A: Strongly agree
- B: Agree
- C: Neither agree nor disagree
- D: Disagree
- E: Strongly disagree

Please also include one 1-sentence QUOTE exactly copied from the example that illustrates this pattern.

Respond with ONLY a JSON with the following format, escaping any quotes within strings with a backslash:
{{
  "pattern_results": [
    {{
      "example_id": "<example_id>",
      "rationale": "<rationale>",
      "answer": "<answer>",
      "quote": "<quote>"
    }}
  ]
}}
"""

SUMMARIZE_CONCEPT_PROMPT_TEMPLATE = """Please write a BRIEF {summary_length} executive summary of the theme "{concept_name}" as it appears in the following examples.
{examples}

DO NOT write the summary as a third party using terms like "the text examples" or "they discuss", but write the summary from the perspective of the text authors making the points directly.
Please respond ONLY with a valid JSON in the following format:
{{
  "summary": "<SUMMARY>"
}}
"""


# ---- Data models ----

@dataclass(frozen=True)
class ThemeModeParams:
    seed: str
    n_quotes: int = 3
    n_bullets: int = 3
    bullet_words: str = "10-18"
    max_docs: Optional[int] = None
    max_themes: int = 12
    # Theme quality controls (LLooM-style review)
    auto_review: bool = True
    max_candidate_clusters: int = 30
    # Scoring / quantification
    include_doc_theme_scores: bool = True
    include_bullet_theme_scores: bool = True
    score_batch_size: int = 6
    score_get_highlights: bool = True
    max_points_for_scoring: Optional[int] = None
    include_theme_summaries: bool = True
    summary_length: str = "15-20 word"
    # Clustering knobs
    umap_neighbors: int = 10
    umap_min_dist: float = 0.1
    random_state: int = 7
    # Concurrency / API
    llm_concurrency: int = 6
    llm_timeout_s: int = 45
    # Limits for hover payload size
    representative_bullets_per_theme: int = 6
    representative_quotes_per_theme: int = 4
    sample_bullets_per_cluster_for_label: int = 35


def require_openai_api_key() -> str:
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required for Theme mode.")
    return key


def _safe_json_load(s: Any) -> Optional[dict]:
    """Best-effort JSON parsing from OpenAI responses."""
    if s is None:
        return None
    if isinstance(s, dict):
        return s
    if not isinstance(s, str):
        return None
    txt = s.strip()
    if not txt:
        return None
    # try direct JSON
    try:
        return json.loads(txt)
    except Exception:
        pass
    # try substring from first '{' to last '}'
    try:
        i = txt.find("{")
        j = txt.rfind("}")
        if i != -1 and j != -1 and j > i:
            return json.loads(txt[i : j + 1])
    except Exception:
        return None
    return None


def _chunk_list(xs: Sequence[Any], n: int) -> List[List[Any]]:
    if n <= 0:
        return [list(xs)]
    return [list(xs[i : i + n]) for i in range(0, len(xs), n)]


def _parse_bucketed_answer(x: Any) -> float:
    """
    LLooM-style bucketed answer → numeric score.
    A: 1.0, B: 0.75, C: 0.5, D: 0.25, E: 0.0
    """
    answer_scores = {"A": 1.0, "B": 0.75, "C": 0.5, "D": 0.25, "E": 0.0}
    if x is None:
        return 0.0
    s = str(x).strip().upper()
    if not s:
        return 0.0
    k = s[0]
    return float(answer_scores.get(k, 0.0))


def _split_into_sentences(text: str) -> List[str]:
    """
    Lightweight sentence splitter (matches the one used elsewhere in this repo).
    Keeps punctuation at the end of each sentence when present.
    """
    if not text or not str(text).strip():
        return []
    t = str(text).strip()
    # Normalize whitespace but keep sentence punctuation.
    t = re.sub(r"\s+", " ", t)
    sents = re.split(r"(?<=[.!?])\s+", t)
    return [s.strip() for s in sents if s and s.strip()]


def _chunk_sentences_2_3(sentences: List[str]) -> List[Dict[str, Any]]:
    """
    Chunk a sentence list into contiguous chunks of 2–3 sentences when possible.
    For very short inputs, falls back to a single chunk of 1 sentence.

    Returns list of:
      { start_sentence_idx, end_sentence_idx, sentence_count, text, sentences }
    Indices are within the provided `sentences` list.
    """
    sents = [s for s in (sentences or []) if s and str(s).strip()]
    n = len(sents)
    if n == 0:
        return []
    if n == 1:
        return [
            {
                "start_sentence_idx": 0,
                "end_sentence_idx": 0,
                "sentence_count": 1,
                "sentences": [sents[0]],
                "text": sents[0],
            }
        ]

    out: List[Dict[str, Any]] = []
    i = 0
    while i < n:
        rem = n - i
        # Avoid a trailing 1-sentence chunk:
        # - rem == 4 -> do 2 + 2
        # - rem == 2/3 -> take all
        if rem == 4:
            size = 2
        elif rem in (2, 3):
            size = rem
        else:
            size = 3

        j = min(n, i + size)
        chunk_sents = sents[i:j]
        if not chunk_sents:
            break
        out.append(
            {
                "start_sentence_idx": i,
                "end_sentence_idx": j - 1,
                "sentence_count": len(chunk_sents),
                "sentences": chunk_sents,
                "text": " ".join(chunk_sents).strip(),
            }
        )
        i = j
    return out


def _meta_get_ci(meta: Dict[str, Any], key: str) -> Any:
    """Case-insensitive metadata key access."""
    if not meta or not key:
        return None
    kl = key.lower()
    for k, v in meta.items():
        if str(k).lower() == kl:
            return v
    return None


def apply_metadata_filters_to_docs(
    docs: List[Dict[str, Any]], filters: Optional[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Apply the same metadata filter semantics as `PizzaSearchTool._apply_metadata_filters`,
    but to raw documents: each doc is {id, text, metadata}.
    """
    if not filters:
        return docs

    import pandas as pd  # local import (already a project dependency)

    out: List[Dict[str, Any]] = []
    for d in docs:
        meta = d.get("metadata") or {}
        ok = True
        for field_name, filter_spec in (filters or {}).items():
            if not isinstance(filter_spec, dict):
                filter_spec = {"operator": "equals", "value": filter_spec}
            op = filter_spec.get("operator", "equals")
            value = filter_spec.get("value")
            if value is None or value == "" or (isinstance(value, list) and len(value) == 0):
                continue

            field_value = _meta_get_ci(meta, field_name)
            if field_value is None or pd.isna(field_value):
                if op in ["equals", "contains", "greater_than", "less_than", "in", "includes"]:
                    ok = False
                    break
                continue

            if op == "equals":
                if str(field_value).lower() != str(value).lower():
                    ok = False
                    break
            elif op == "not_equals":
                if str(field_value).lower() == str(value).lower():
                    ok = False
                    break
            elif op == "contains":
                if str(value).lower() not in str(field_value).lower():
                    ok = False
                    break
            elif op == "greater_than":
                try:
                    if float(field_value) <= float(value):
                        ok = False
                        break
                except (TypeError, ValueError):
                    ok = False
                    break
            elif op == "less_than":
                try:
                    if float(field_value) >= float(value):
                        ok = False
                        break
                except (TypeError, ValueError):
                    ok = False
                    break
            elif op == "between":
                try:
                    lo, hi = value
                    fv = float(field_value)
                    if not (float(lo) <= fv <= float(hi)):
                        ok = False
                        break
                except Exception:
                    ok = False
                    break
            elif op in ["in", "includes"]:
                if isinstance(value, list):
                    if not any(str(field_value).lower() == str(v).lower() for v in value):
                        ok = False
                        break
                else:
                    if str(field_value).lower() != str(value).lower():
                        ok = False
                        break
            elif op == "not_includes":
                if isinstance(value, list):
                    if any(str(field_value).lower() == str(v).lower() for v in value):
                        ok = False
                        break
                else:
                    if str(field_value).lower() == str(value).lower():
                        ok = False
                        break
            else:
                # Unknown operator: fail closed.
                ok = False
                break

        if ok:
            out.append(d)
    return out


def _select_doc_text(doc: Dict[str, Any], text_source: str) -> str:
    """
    Select which interview text to feed into theme-mode.

    - text_source="all": use doc["text"] (q1..q5 concatenated by the loader)
    - text_source="q5": use metadata["q5_response"] (case-insensitive)
    """
    ts = (text_source or "all").strip().lower()
    if ts in {"q1", "q2", "q3", "q4", "q5"}:
        meta = doc.get("metadata") or {}
        v = _meta_get_ci(meta, f"{ts}_response")
        return (str(v).strip() if v is not None else "")
    return (doc.get("text") or "").strip()


async def _openai_json(
    client: Any,
    model: str,
    system: str,
    user: str,
    timeout_s: int,
    semaphore: asyncio.Semaphore,
) -> Optional[dict]:
    async with semaphore:
        # OpenAI python SDK supports async via `AsyncOpenAI`, but this repo
        # currently uses sync client elsewhere. We keep it simple by running
        # sync calls in a thread.
        def _call():
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                response_format={"type": "json_object"},
                temperature=0.4,
                timeout=timeout_s,
            )
            return resp.choices[0].message.content

        try:
            content = await asyncio.wait_for(asyncio.to_thread(_call), timeout=timeout_s + 5)
        except Exception:
            return None
        return _safe_json_load(content)


async def distill_filter_quotes(
    docs: List[Dict[str, Any]],
    *,
    params: ThemeModeParams,
    llm_model: str = "gpt-4o-mini",
    _client: Any = None,
    _semaphore: asyncio.Semaphore | None = None,
) -> Dict[str, List[str]]:
    """
    Returns: doc_id -> list of quotes
    """
    from openai import OpenAI

    client = _client or OpenAI(api_key=require_openai_api_key())
    sem = _semaphore or asyncio.Semaphore(max(1, int(params.llm_concurrency)))
    seed_upper = (params.seed or "").strip().upper()
    system = "You extract verbatim quotes from text. Return only valid JSON."

    async def one(d: Dict[str, Any]) -> Tuple[str, List[str]]:
        doc_id = str(d.get("id"))
        text = (d.get("text") or "").strip()
        if not text:
            return doc_id, []
        prompt = FILTER_PROMPT_TEMPLATE.format(ex=text, n_quotes=params.n_quotes, seed_upper=seed_upper)
        j = await _openai_json(client, llm_model, system, prompt, params.llm_timeout_s, sem)
        quotes = []
        if isinstance(j, dict):
            q = j.get("relevant_quotes")
            if isinstance(q, list):
                quotes = [str(x).strip() for x in q if str(x).strip()]
        return doc_id, quotes

    pairs = await asyncio.gather(*[one(d) for d in docs])
    return {doc_id: quotes for doc_id, quotes in pairs}


async def distill_summarize_bullets(
    docs: List[Dict[str, Any]],
    doc_quotes: Dict[str, List[str]],
    *,
    params: ThemeModeParams,
    llm_model: str = "gpt-4o-mini",
    _client: Any = None,
    _semaphore: asyncio.Semaphore | None = None,
) -> List[Dict[str, Any]]:
    """
    Returns list of bullet objects:
      {
        bullet_id, doc_id, bullet,
        source_text, source_sentences[],
        start_sentence_idx, end_sentence_idx, sentence_count,
        quote (back-compat for UI),
        metadata{...}
      }
    """
    from openai import OpenAI

    client = _client or OpenAI(api_key=require_openai_api_key())
    sem = _semaphore or asyncio.Semaphore(max(1, int(params.llm_concurrency)))
    seed_upper = (params.seed or "").strip().upper()
    system = "You are a careful qualitative analyst. Return only valid JSON."

    # Map doc_id -> metadata subset (for dot filtering/coloring)
    doc_meta_subset: Dict[str, Dict[str, Any]] = {}
    for d in docs:
        doc_id = str(d.get("id"))
        meta = d.get("metadata") or {}
        doc_meta_subset[doc_id] = {
            "participant_id": _meta_get_ci(meta, "participant_id"),
            "age": _meta_get_ci(meta, "age"),
            "city_of_residence": _meta_get_ci(meta, "city_of_residence"),
            "state_of_residence": _meta_get_ci(meta, "state_of_residence"),
            "region_of_residence": _meta_get_ci(meta, "region_of_residence"),
            "income": _meta_get_ci(meta, "income"),
            "pizza_consumption": _meta_get_ci(meta, "pizza_consumption"),
            "food_restrictions": _meta_get_ci(meta, "food_restrictions"),
        }

    async def one(doc_id: str) -> List[Dict[str, Any]]:
        quotes = doc_quotes.get(doc_id) or []
        if not quotes:
            return []
        # Create 2–3 sentence chunks from the relevant (seed-filtered) excerpt text.
        relevant_text = " ".join([str(q).strip() for q in quotes if str(q).strip()]).strip()
        sentences = _split_into_sentences(relevant_text)
        chunks = _chunk_sentences_2_3(sentences)
        if not chunks:
            return []

        # Safety cap: prevent pathological runs from creating hundreds of bullets per doc.
        # (Still respects the "2–3 sentences per bullet" rule.)
        chunks = chunks[:50]

        async def one_chunk(ch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            src_text = (ch.get("text") or "").strip()
            if not src_text:
                return None
            prompt = SUMMARIZE_CHUNK_PROMPT_TEMPLATE.format(
                ex=src_text,
                seed_upper=seed_upper,
                n_words=params.bullet_words,
            )
            j = await _openai_json(client, llm_model, system, prompt, params.llm_timeout_s, sem)
            bullet = ""
            if isinstance(j, dict) and isinstance(j.get("bullet"), str):
                bullet = j["bullet"].strip()
            # Fallback: if model fails, use the excerpt itself (truncated) as a "bullet".
            if not bullet:
                bullet = src_text
                if len(bullet) > 220:
                    bullet = bullet[:217].rstrip() + "..."

            return {
                "bullet_id": str(uuid.uuid4()),
                "doc_id": doc_id,
                "bullet": bullet,
                # Stable linkage: the exact excerpt (2–3 sentences) used for the bullet.
                "source_text": src_text,
                "source_sentences": list(ch.get("sentences") or []),
                "start_sentence_idx": int(ch.get("start_sentence_idx", 0)),
                "end_sentence_idx": int(ch.get("end_sentence_idx", 0)),
                "sentence_count": int(ch.get("sentence_count", 0)),
                # Back-compat for existing UI/tooltips.
                "quote": src_text,
                "metadata": doc_meta_subset.get(doc_id) or {},
            }

        rows = await asyncio.gather(*[one_chunk(ch) for ch in chunks])
        return [r for r in rows if isinstance(r, dict)]

    doc_ids = [str(d.get("id")) for d in docs]
    rows_nested = await asyncio.gather(*[one(doc_id) for doc_id in doc_ids])
    rows: List[Dict[str, Any]] = []
    for chunk in rows_nested:
        rows.extend(chunk)
    return rows


async def score_bullets_against_themes(
    points: List[Dict[str, Any]],
    themes: List[Dict[str, Any]],
    *,
    seed: str,
    params: ThemeModeParams,
    llm_model: str,
    _client: Any,
    _semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    """
    LLooM-style graded scoring:
    - For each theme (concept), score each bullet's source_text with A–E buckets,
      mapped to numeric scores in [0, 1], plus rationale and highlight quote.

    Returns:
      {
        "bullet_theme_scores": [ {doc_id,text,bullet_id,bullet,concept_id,concept_name,concept_prompt,score,rationale,highlight,concept_seed}... ],
        "concept_summaries": { concept_id: summary, ... }
      }
    """
    pts = [p for p in (points or []) if isinstance(p, dict)]
    ths = [t for t in (themes or []) if isinstance(t, dict)]
    if not pts or not ths:
        return {"bullet_theme_scores": [], "concept_summaries": {}}

    # Optional cap (prevents runaway scoring on huge runs).
    if params.max_points_for_scoring is not None and int(params.max_points_for_scoring) > 0:
        pts = pts[: int(params.max_points_for_scoring)]

    # Batch examples (LLooM does per-concept batching).
    batch_size = max(1, int(params.score_batch_size))
    system = "You are a careful evaluator. Return only valid JSON."

    # Map bullet_id -> point for fast join.
    by_bullet_id: Dict[str, Dict[str, Any]] = {}
    for p in pts:
        bid = str(p.get("bullet_id") or "").strip()
        if bid:
            by_bullet_id[bid] = p

    bullet_ids = [str(p.get("bullet_id")) for p in pts if p.get("bullet_id") is not None]

    async def score_one_theme(theme: Dict[str, Any]) -> List[Dict[str, Any]]:
        cid = theme.get("cluster_id")
        theme_id = theme.get("theme_id")
        cname = (theme.get("name") or "").strip()
        cprompt = (theme.get("prompt") or "").strip()
        concept_id = str(theme_id) if theme_id is not None else str(cid)

        rows: List[Dict[str, Any]] = []
        for batch in _chunk_list(bullet_ids, batch_size):
            examples = []
            for bid in batch:
                p = by_bullet_id.get(str(bid))
                if not p:
                    continue
                txt = (p.get("source_text") or p.get("quote") or "").strip()
                examples.append({"example_id": str(bid), "example": txt})
            examples_json = json.dumps(examples, ensure_ascii=False)

            prompt = SCORE_HIGHLIGHT_PROMPT_TEMPLATE.format(
                examples_json=examples_json,
                concept_name=cname or f"Theme {concept_id}",
                concept_prompt=cprompt,
            )
            j = await _openai_json(_client, llm_model, system, prompt, params.llm_timeout_s, _semaphore)
            results = []
            if isinstance(j, dict) and isinstance(j.get("pattern_results"), list):
                results = [x for x in j["pattern_results"] if isinstance(x, dict)]

            # Index results by example_id
            got: Dict[str, Dict[str, Any]] = {}
            for r in results:
                exid = str(r.get("example_id") or "").strip()
                if exid:
                    got[exid] = r

            for bid in batch:
                p = by_bullet_id.get(str(bid))
                if not p:
                    continue
                src = (p.get("source_text") or p.get("quote") or "").strip()
                bullet = (p.get("bullet") or "").strip()
                r = got.get(str(bid)) or {}
                score = _parse_bucketed_answer(r.get("answer"))
                rationale = (r.get("rationale") or "").strip() if isinstance(r.get("rationale"), str) else ""
                highlight = (r.get("quote") or "").strip() if isinstance(r.get("quote"), str) else ""

                rows.append(
                    {
                        "doc_id": str(p.get("doc_id")),
                        "text": src,
                        "bullet_id": str(bid),
                        "bullet": bullet,
                        "concept_id": concept_id,
                        "concept_name": cname,
                        "concept_prompt": cprompt,
                        "score": float(score),
                        "rationale": rationale,
                        "highlight": highlight,
                        "concept_seed": seed,
                    }
                )
        return rows

    # Score all themes concurrently (bounded by semaphore inside _openai_json).
    score_rows_nested = await asyncio.gather(*[score_one_theme(t) for t in ths])
    score_rows: List[Dict[str, Any]] = []
    for part in score_rows_nested:
        score_rows.extend(part)

    # Optional concept summaries (LLooM-style): summarize top matches per concept.
    summaries: Dict[str, str] = {}
    if params.include_theme_summaries:
        # Group by concept_id and collect representative highlights/texts.
        by_c: Dict[str, List[Dict[str, Any]]] = {}
        for r in score_rows:
            by_c.setdefault(str(r.get("concept_id")), []).append(r)
        for cid, rows in by_c.items():
            # Take top scored rows (prefer highlight, fall back to text)
            rows_sorted = sorted(rows, key=lambda x: float(x.get("score") or 0.0), reverse=True)
            top = [x for x in rows_sorted if float(x.get("score") or 0.0) >= 0.75][:8]
            if not top:
                continue
            ex_lines: List[str] = []
            for x in top:
                ex = (x.get("highlight") or x.get("text") or "").strip()
                if ex:
                    ex_lines.append(f"- {ex}")
            if not ex_lines:
                continue
            examples_block = "\n".join(ex_lines)

            # Find concept name
            cname = ""
            for t in ths:
                tid = str(t.get("theme_id") or t.get("cluster_id"))
                if tid == str(cid):
                    cname = (t.get("name") or "").strip()
                    break

            prompt = SUMMARIZE_CONCEPT_PROMPT_TEMPLATE.format(
                summary_length=params.summary_length,
                concept_name=cname or f"Theme {cid}",
                examples=examples_block,
            )
            j = await _openai_json(_client, llm_model, system, prompt, params.llm_timeout_s, _semaphore)
            if isinstance(j, dict) and isinstance(j.get("summary"), str) and j["summary"].strip():
                summaries[str(cid)] = j["summary"].strip()

    return {"bullet_theme_scores": score_rows, "concept_summaries": summaries}


def _choose_cluster_params(n_points: int) -> int:
    # A simple heuristic: aim for finer clusters by default.
    if n_points <= 0:
        return 3
    return max(3, min(20, int(math.ceil(n_points / 30))))


def _clamp_int(x: Any, lo: int, hi: int) -> int:
    try:
        v = int(x)
    except Exception:
        return lo
    return max(lo, min(hi, v))


def _themes_list_for_review(themes: List[Dict[str, Any]]) -> str:
    """
    Render a stable, human-readable theme list for LLM review.
    We identify themes by cluster_id to avoid name collisions.
    """
    lines: List[str] = []
    for t in themes:
        cid = t.get("cluster_id")
        name = (t.get("name") or "").strip()
        prompt = (t.get("prompt") or "").strip()
        n = t.get("n_points")
        lines.append(f"- cluster_id: {cid} | n_points: {n} | name: {name} | prompt: {prompt}")
    return "\n".join(lines)


async def _review_filter_themes(
    themes: List[Dict[str, Any]],
    *,
    params: ThemeModeParams,
    llm_model: str,
    _client: Any,
    _semaphore: asyncio.Semaphore,
) -> List[Dict[str, Any]]:
    """
    LLooM-style: remove generic/vague/duplicate themes, then select a diverse set.
    Returns filtered theme list (order preserved as much as possible).
    """
    if not params.auto_review:
        return themes
    if not themes:
        return themes

    # Safety cap: avoid very long prompts.
    max_cands = _clamp_int(params.max_candidate_clusters, 3, 80)
    cand = list(themes)[:max_cands]

    system = "You are a careful qualitative analyst. Return only valid JSON."
    themes_list = _themes_list_for_review(cand)

    # 1) Remove low-quality themes
    remove_prompt = REVIEW_REMOVE_PROMPT_TEMPLATE.format(themes_list=themes_list)
    j_remove = await _openai_json(_client, llm_model, system, remove_prompt, params.llm_timeout_s, _semaphore)
    remove_ids: set[int] = set()
    if isinstance(j_remove, dict):
        xs = j_remove.get("remove_cluster_ids")
        if isinstance(xs, list):
            for x in xs:
                try:
                    remove_ids.add(int(x))
                except Exception:
                    continue

    kept = [t for t in cand if int(t.get("cluster_id", -1)) not in remove_ids]
    if not kept:
        # Fall back: keep originals if reviewer removed everything.
        kept = cand

    # 2) Select a diverse, non-overlapping subset
    select_prompt = REVIEW_SELECT_PROMPT_TEMPLATE.format(
        themes_list=_themes_list_for_review(kept),
        max_themes=max(1, int(params.max_themes)),
    )
    j_sel = await _openai_json(_client, llm_model, system, select_prompt, params.llm_timeout_s, _semaphore)
    sel_ids: List[int] = []
    if isinstance(j_sel, dict) and isinstance(j_sel.get("selected_cluster_ids"), list):
        for x in j_sel["selected_cluster_ids"]:
            try:
                sel_ids.append(int(x))
            except Exception:
                continue

    if not sel_ids:
        # If reviewer fails, keep top-N by cluster size (already ordered upstream).
        return kept[: max(1, int(params.max_themes))]

    sel_set = set(sel_ids)
    out = [t for t in kept if int(t.get("cluster_id", -1)) in sel_set]
    if not out:
        return kept[: max(1, int(params.max_themes))]
    return out[: max(1, int(params.max_themes))]


def _compute_doc_theme_scores(points: List[Dict[str, Any]], themes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Quantify per-document theme prevalence using cluster membership of bullets.

    Score definition (0..1):
      score(doc, theme) = (# bullets from doc assigned to theme cluster) / (total non-noise bullets for doc)

    This allows "no theme applies" (all zeros) for a document.
    """
    labeled_cids = {int(t.get("cluster_id")) for t in themes if t.get("cluster_id") is not None}

    # Count total non-noise bullets per doc, and per (doc, cluster) within labeled clusters.
    doc_total: Dict[str, int] = {}
    doc_cluster: Dict[Tuple[str, int], int] = {}

    for p in points:
        doc_id = str(p.get("doc_id"))
        cid = int(p.get("cluster_id", -1))
        if cid == -1:
            continue
        doc_total[doc_id] = doc_total.get(doc_id, 0) + 1
        if cid in labeled_cids:
            key = (doc_id, cid)
            doc_cluster[key] = doc_cluster.get(key, 0) + 1

    # Build doc-theme score rows
    doc_scores: List[Dict[str, Any]] = []
    for doc_id, total in doc_total.items():
        denom = max(1, int(total))
        for cid in labeled_cids:
            n_in = int(doc_cluster.get((doc_id, cid), 0))
            score = float(n_in) / float(denom)
            doc_scores.append(
                {
                    "doc_id": doc_id,
                    "cluster_id": int(cid),
                    "score": score,
                    "n_bullets_in_theme": n_in,
                    "n_bullets_total": int(total),
                }
            )

    # Theme prevalence summary
    by_cid: Dict[int, List[float]] = {int(cid): [] for cid in labeled_cids}
    for row in doc_scores:
        by_cid[int(row["cluster_id"])].append(float(row["score"]))

    theme_prevalence: List[Dict[str, Any]] = []
    for cid, scores in by_cid.items():
        if not scores:
            theme_prevalence.append(
                {"cluster_id": int(cid), "docs": 0, "mean_score": 0.0, "docs_with_any": 0}
            )
            continue
        docs = len(scores)
        mean_score = float(sum(scores) / max(1, docs))
        docs_with_any = int(sum(1 for s in scores if s > 0))
        theme_prevalence.append(
            {"cluster_id": int(cid), "docs": docs, "mean_score": mean_score, "docs_with_any": docs_with_any}
        )

    return {"doc_scores": doc_scores, "theme_prevalence": theme_prevalence}


def cluster_points(
    bullets: List[Dict[str, Any]],
    *,
    embedder: Any,
    params: ThemeModeParams,
) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    """
    Adds to each bullet dict:
      - x, y (UMAP 2D)
      - cluster_id (HDBSCAN label)
    Returns:
      (bullets_with_coords, cluster_stats)
    """
    texts = [b["bullet"] for b in bullets]
    if not texts:
        return bullets, {}

    import numpy as np

    # UMAP + HDBSCAN are optional deps; import here to keep module importable.
    import umap
    from hdbscan import HDBSCAN

    embs = embedder.encode(texts, convert_to_numpy=True)
    if isinstance(embs, list):
        embs = np.array(embs)

    # 5D for clustering (LLooM-like)
    umap_5 = umap.UMAP(
        n_neighbors=params.umap_neighbors,
        n_components=5,
        min_dist=params.umap_min_dist,
        metric="cosine",
        random_state=params.random_state,
    )
    embs_5 = umap_5.fit_transform(embs)

    # 2D for plotting
    umap_2 = umap.UMAP(
        n_neighbors=params.umap_neighbors,
        n_components=2,
        min_dist=params.umap_min_dist,
        metric="cosine",
        random_state=params.random_state,
    )
    coords = umap_2.fit_transform(embs)

    min_cluster_size = _choose_cluster_params(len(bullets))
    hdb = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="leaf",
    )
    labels = hdb.fit_predict(embs_5)

    cluster_stats: Dict[int, Dict[str, Any]] = {}
    for i, b in enumerate(bullets):
        b["x"] = float(coords[i][0])
        b["y"] = float(coords[i][1])
        b["cluster_id"] = int(labels[i])
        cid = int(labels[i])
        if cid not in cluster_stats:
            cluster_stats[cid] = {"n": 0}
        cluster_stats[cid]["n"] += 1

    return bullets, cluster_stats


def _cluster_centroids(points: List[Dict[str, Any]]) -> Dict[int, Tuple[float, float]]:
    import numpy as np

    by: Dict[int, List[Tuple[float, float]]] = {}
    for p in points:
        cid = int(p.get("cluster_id", -1))
        if "x" not in p or "y" not in p:
            continue
        by.setdefault(cid, []).append((float(p["x"]), float(p["y"])))
    out: Dict[int, Tuple[float, float]] = {}
    for cid, xy in by.items():
        arr = np.array(xy, dtype=float)
        if len(arr) == 0:
            continue
        out[cid] = (float(arr[:, 0].mean()), float(arr[:, 1].mean()))
    return out


async def synthesize_theme_candidates_for_cluster(
    cluster_id: int,
    cluster_points: List[Dict[str, Any]],
    doc_quotes: Dict[str, List[str]],
    *,
    params: ThemeModeParams,
    k: int = 3,
    llm_model: str = "gpt-4o",
    _client: Any = None,
    _semaphore: asyncio.Semaphore | None = None,
) -> List[Dict[str, Any]]:
    """
    Generate K candidate labels (name + criteria prompt) for a single cluster in ONE LLM call.
    Returns a list of theme dicts, each with a unique `theme_id` but sharing the same `cluster_id`.
    """
    from openai import OpenAI

    kk = max(1, int(k or 1))
    client = _client or OpenAI(api_key=require_openai_api_key())
    sem = _semaphore or asyncio.Semaphore(max(1, int(params.llm_concurrency)))
    seed_upper = (params.seed or "").strip().upper()
    system = (
        "You are a sociologist doing qualitative coding. "
        "Label clusters with specific, concrete themes. Return only valid JSON."
    )

    sampled = list(cluster_points)[: max(1, int(params.sample_bullets_per_cluster_for_label))]
    examples = [{"example_id": str(p.get("doc_id")), "example": str(p.get("bullet") or "")} for p in sampled]
    examples_json = json.dumps(examples, ensure_ascii=False)

    prompt = SYNTHESIZE_MULTI_PROMPT_TEMPLATE.format(
        examples_json=examples_json,
        seed_upper=seed_upper,
        k=kk,
    )
    j = await _openai_json(client, llm_model, system, prompt, params.llm_timeout_s, sem)

    specs: List[Dict[str, Any]] = []
    if isinstance(j, dict) and isinstance(j.get("themes"), list):
        for t in j["themes"]:
            if not isinstance(t, dict):
                continue
            name = (t.get("name") or "").strip()
            tprompt = (t.get("prompt") or "").strip()
            ex = t.get("example_ids")
            example_ids: List[str] = []
            if isinstance(ex, list):
                example_ids = [str(x).strip() for x in ex if str(x).strip()]
            if name:
                specs.append({"name": name, "prompt": tprompt, "example_ids": example_ids})

    # De-duplicate by (name,prompt) to avoid LLM paraphrase collisions
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for s in specs:
        key = (str(s.get("name") or "").lower(), str(s.get("prompt") or "").lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(s)
    specs = deduped[:kk]

    # Fallback if model fails
    if not specs:
        specs = [{"name": f"Cluster {cluster_id}", "prompt": "", "example_ids": []}]

    # Build a quick lookup doc_id -> bullets within the cluster
    bullets_by_doc: Dict[str, List[str]] = {}
    for p in cluster_points:
        did = str(p.get("doc_id") or "").strip()
        b = (p.get("bullet") or "").strip()
        if not did or not b:
            continue
        bullets_by_doc.setdefault(did, []).append(b)

    out: List[Dict[str, Any]] = []
    for s in specs:
        example_doc_ids = [x for x in (s.get("example_ids") or []) if x]

        rep_bullets: List[str] = []
        if example_doc_ids:
            for did in example_doc_ids:
                for b in bullets_by_doc.get(str(did), []):
                    rep_bullets.append(b)
                    if len(rep_bullets) >= int(params.representative_bullets_per_theme):
                        break
                if len(rep_bullets) >= int(params.representative_bullets_per_theme):
                    break
        if not rep_bullets:
            rep_bullets = [p.get("bullet") for p in cluster_points[: int(params.representative_bullets_per_theme)] if p.get("bullet")]

        rep_quotes: List[str] = []
        cand_doc_ids = example_doc_ids[:] if example_doc_ids else [str(p.get("doc_id")) for p in cluster_points]
        cand_doc_ids = [d for d in cand_doc_ids if d]
        for did in cand_doc_ids:
            qs = doc_quotes.get(str(did)) or []
            for q in qs:
                rep_quotes.append(q)
                if len(rep_quotes) >= int(params.representative_quotes_per_theme):
                    break
            if len(rep_quotes) >= int(params.representative_quotes_per_theme):
                break

        out.append(
            {
                "cluster_id": int(cluster_id),
                "theme_id": str(uuid.uuid4()),
                "name": (s.get("name") or f"Cluster {cluster_id}").strip(),
                "prompt": (s.get("prompt") or "").strip(),
                "n_points": len(cluster_points),
                "representative_bullets": rep_bullets,
                "representative_quotes": rep_quotes,
                "example_doc_ids": example_doc_ids,
            }
        )

    return out


async def synthesize_theme_for_cluster(
    cluster_id: int,
    cluster_points: List[Dict[str, Any]],
    doc_quotes: Dict[str, List[str]],
    *,
    params: ThemeModeParams,
    llm_model: str = "gpt-4o",
    _client: Any = None,
    _semaphore: asyncio.Semaphore | None = None,
) -> Dict[str, Any]:
    from openai import OpenAI

    client = _client or OpenAI(api_key=require_openai_api_key())
    sem = _semaphore or asyncio.Semaphore(max(1, int(params.llm_concurrency)))
    seed_upper = (params.seed or "").strip().upper()
    system = (
        "You are a sociologist doing qualitative coding. "
        "Label clusters with specific, concrete themes. Return only valid JSON."
    )

    # sample bullets for prompt
    sampled = list(cluster_points)
    # deterministic-ish sampling: cap and keep order stable
    sampled = sampled[: max(1, int(params.sample_bullets_per_cluster_for_label))]
    examples = [{"example_id": p["doc_id"], "example": p["bullet"]} for p in sampled]
    examples_json = json.dumps(examples, ensure_ascii=False)

    prompt = SYNTHESIZE_PROMPT_TEMPLATE.format(examples_json=examples_json, seed_upper=seed_upper)
    j = await _openai_json(client, llm_model, system, prompt, params.llm_timeout_s, sem)

    name = f"Cluster {cluster_id}"
    theme_prompt = ""
    example_ids: List[str] = []
    if isinstance(j, dict) and isinstance(j.get("theme"), dict):
        t = j["theme"]
        if isinstance(t.get("name"), str) and t["name"].strip():
            name = t["name"].strip()
        if isinstance(t.get("prompt"), str):
            theme_prompt = t["prompt"].strip()
        ex = t.get("example_ids")
        if isinstance(ex, list):
            example_ids = [str(x).strip() for x in ex if str(x).strip()]

    # representative bullets: just the first N points (stable); UI tooltip uses these.
    rep_bullets = [p["bullet"] for p in cluster_points[: params.representative_bullets_per_theme]]

    # representative quotes from example_ids, falling back to any doc_ids in cluster
    rep_quotes: List[str] = []
    doc_ids_in_cluster = [str(p["doc_id"]) for p in cluster_points]
    candidate_doc_ids = []
    for exid in example_ids:
        if exid in doc_quotes:
            candidate_doc_ids.append(exid)
    if not candidate_doc_ids:
        candidate_doc_ids = doc_ids_in_cluster[: params.representative_quotes_per_theme]
    for did in candidate_doc_ids:
        qs = doc_quotes.get(did) or []
        for q in qs:
            rep_quotes.append(q)
            if len(rep_quotes) >= params.representative_quotes_per_theme:
                break
        if len(rep_quotes) >= params.representative_quotes_per_theme:
            break

    return {
        "cluster_id": int(cluster_id),
        "name": name,
        "prompt": theme_prompt,
        "n_points": len(cluster_points),
        "representative_bullets": rep_bullets,
        "representative_quotes": rep_quotes,
        "example_doc_ids": example_ids,
    }


async def run_theme_exploration(
    *,
    documents: List[Dict[str, Any]],
    metadata_filters: Optional[Dict[str, Any]],
    text_source: str = "all",
    embedder: Any,
    params: ThemeModeParams,
    candidates_per_cluster: int = 3,
    llm_distill_model: str = "gpt-4o-mini",
    llm_synth_model: str = "gpt-4o",
) -> Dict[str, Any]:
    """
    Exploration stage:
    - distill -> bullets -> cluster
    - for top clusters, generate K candidate theme labels per cluster
    - no review filtering; no scoring
    """
    t0 = time.time()
    seed = (params.seed or "").strip()
    if not seed:
        raise ValueError("seed must be a non-empty string")

    from openai import OpenAI

    openai_client = OpenAI(api_key=require_openai_api_key())
    llm_sem = asyncio.Semaphore(max(1, int(params.llm_concurrency)))

    docs_meta = apply_metadata_filters_to_docs(documents, metadata_filters)
    total_docs_after_meta = len(docs_meta)

    ts = (text_source or "all").strip().lower()
    docs_text: List[Dict[str, Any]] = []
    for d in docs_meta:
        txt = _select_doc_text(d, ts)
        if not txt:
            continue
        d2 = dict(d)
        d2["text"] = txt
        docs_text.append(d2)
    total_docs_after_text_source = len(docs_text)

    docs = docs_text
    if params.max_docs is not None and params.max_docs > 0:
        docs = docs[: int(params.max_docs)]
    used_docs = len(docs)

    doc_quotes = await distill_filter_quotes(
        docs, params=params, llm_model=llm_distill_model, _client=openai_client, _semaphore=llm_sem
    )
    bullets = await distill_summarize_bullets(
        docs, doc_quotes, params=params, llm_model=llm_distill_model, _client=openai_client, _semaphore=llm_sem
    )

    bullets, cluster_stats = cluster_points(bullets, embedder=embedder, params=params)
    centroids = _cluster_centroids(bullets)

    cluster_ids_all = sorted(
        [cid for cid in cluster_stats.keys() if cid != -1],
        key=lambda cid: cluster_stats[cid]["n"],
        reverse=True,
    )
    n_clusters = max(1, int(params.max_themes or 12))
    cluster_ids = cluster_ids_all[: min(len(cluster_ids_all), n_clusters)]

    by_cluster: Dict[int, List[Dict[str, Any]]] = {}
    for p in bullets:
        cid = int(p.get("cluster_id", -1))
        if cid in cluster_ids:
            by_cluster.setdefault(cid, []).append(p)

    kk = max(1, int(candidates_per_cluster or 1))
    nested = await asyncio.gather(
        *[
            synthesize_theme_candidates_for_cluster(
                cid,
                by_cluster.get(cid, []),
                doc_quotes,
                params=params,
                k=kk,
                llm_model=llm_synth_model,
                _client=openai_client,
                _semaphore=llm_sem,
            )
            for cid in cluster_ids
        ]
    )
    candidate_themes: List[Dict[str, Any]] = []
    for part in nested:
        if isinstance(part, list):
            candidate_themes.extend([x for x in part if isinstance(x, dict)])

    for th in candidate_themes:
        cid = int(th.get("cluster_id", -1))
        if cid in centroids:
            th["centroid"] = {"x": centroids[cid][0], "y": centroids[cid][1]}

    elapsed_s = time.time() - t0
    return {
        "explore_id": str(uuid.uuid4()),
        "seed": seed,
        "metadata_filters": metadata_filters,
        "text_source": ts,
        "params": {
            "n_quotes": params.n_quotes,
            "n_bullets": params.n_bullets,
            "bullet_words": params.bullet_words,
            "max_docs": params.max_docs,
            "max_themes": params.max_themes,
            "candidates_per_cluster": kk,
        },
        "counts": {
            "docs_total_after_meta_filters": total_docs_after_meta,
            "docs_total_after_text_source": total_docs_after_text_source,
            "docs_used": used_docs,
            "bullets": len(bullets),
            "clusters_labeled": len(cluster_ids),
            "theme_candidates": len(candidate_themes),
        },
        "points": bullets,
        "candidate_themes": candidate_themes,
        "timing": {"elapsed_seconds": elapsed_s},
    }


async def run_theme_scoring(
    *,
    points: List[Dict[str, Any]],
    selected_themes: List[Dict[str, Any]],
    seed: str,
    params: ThemeModeParams,
    llm_score_model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """
    Scoring stage:
    - run LLooM-style graded scoring only for selected themes
    - returns a dict with bullet_theme_scores + concept_summaries
    """
    from openai import OpenAI

    openai_client = OpenAI(api_key=require_openai_api_key())
    llm_sem = asyncio.Semaphore(max(1, int(params.llm_concurrency)))

    scored = await score_bullets_against_themes(
        points,
        selected_themes,
        seed=(seed or params.seed or "").strip(),
        params=params,
        llm_model=llm_score_model,
        _client=openai_client,
        _semaphore=llm_sem,
    )
    return scored if isinstance(scored, dict) else {"bullet_theme_scores": [], "concept_summaries": {}}


async def run_theme_mode(
    *,
    documents: List[Dict[str, Any]],
    metadata_filters: Optional[Dict[str, Any]],
    text_source: str = "all",
    embedder: Any,
    params: ThemeModeParams,
    llm_distill_model: str = "gpt-4o-mini",
    llm_synth_model: str = "gpt-4o",
) -> Dict[str, Any]:
    """
    Orchestrate the full pipeline and return JSON-serializable output:
      { run_id, seed, params, counts, themes[], points[] }
    """
    t0 = time.time()
    seed = (params.seed or "").strip()
    if not seed:
        raise ValueError("seed must be a non-empty string")

    # Shared OpenAI client + concurrency control for the whole run
    from openai import OpenAI

    openai_client = OpenAI(api_key=require_openai_api_key())
    llm_sem = asyncio.Semaphore(max(1, int(params.llm_concurrency)))

    # 1) Pre-cluster metadata filtering
    docs_meta = apply_metadata_filters_to_docs(documents, metadata_filters)
    total_docs_after_meta = len(docs_meta)

    # 2) Select which question text to use (e.g., q5 only), and drop empties
    ts = (text_source or "all").strip().lower()
    docs_text: List[Dict[str, Any]] = []
    for d in docs_meta:
        txt = _select_doc_text(d, ts)
        if not txt:
            continue
        # Copy to avoid mutating the shared `documents` list used elsewhere.
        d2 = dict(d)
        d2["text"] = txt
        docs_text.append(d2)

    total_docs_after_text_source = len(docs_text)

    # 3) Safety cap (after filtering + text selection)
    docs = docs_text
    if params.max_docs is not None and params.max_docs > 0:
        docs = docs[: int(params.max_docs)]
    used_docs = len(docs)

    # Distill quotes and bullets
    doc_quotes = await distill_filter_quotes(
        docs, params=params, llm_model=llm_distill_model, _client=openai_client, _semaphore=llm_sem
    )
    bullets = await distill_summarize_bullets(
        docs, doc_quotes, params=params, llm_model=llm_distill_model, _client=openai_client, _semaphore=llm_sem
    )

    # Cluster
    bullets, cluster_stats = cluster_points(bullets, embedder=embedder, params=params)
    centroids = _cluster_centroids(bullets)

    # Choose candidate clusters to label
    cluster_ids_all = sorted(
        [cid for cid in cluster_stats.keys() if cid != -1],
        key=lambda cid: cluster_stats[cid]["n"],
        reverse=True,
    )
    # Label more candidates than we ultimately show; review will select a diverse subset.
    k = max(1, int(params.max_themes)) if (params.max_themes is not None and params.max_themes > 0) else 12
    cand_k = min(len(cluster_ids_all), max(k, min(int(params.max_candidate_clusters), k * 3)))
    cluster_ids = cluster_ids_all[:cand_k]

    # Build per-cluster point lists
    by_cluster: Dict[int, List[Dict[str, Any]]] = {}
    for p in bullets:
        cid = int(p.get("cluster_id", -1))
        if cid in cluster_ids:
            by_cluster.setdefault(cid, []).append(p)

    # Synthesize theme label per cluster (async)
    themes = await asyncio.gather(
        *[
            synthesize_theme_for_cluster(
                cid,
                by_cluster.get(cid, []),
                doc_quotes,
                params=params,
                llm_model=llm_synth_model,
                _client=openai_client,
                _semaphore=llm_sem,
            )
            for cid in cluster_ids
        ]
    )

    # LLooM-style review: remove generic/duplicate themes, then select diverse set.
    themes = await _review_filter_themes(
        list(themes),
        params=params,
        llm_model=llm_synth_model,
        _client=openai_client,
        _semaphore=llm_sem,
    )

    # Attach centroid for label placement
    for th in themes:
        cid = int(th["cluster_id"])
        if cid in centroids:
            th["centroid"] = {"x": centroids[cid][0], "y": centroids[cid][1]}

    # Quantify prevalence per document (LLooM-like "theme scores")
    scoring: Dict[str, Any] = {}
    if params.include_doc_theme_scores:
        scoring = _compute_doc_theme_scores(bullets, list(themes))

    # LLooM-style graded scoring: bullet/excerpt × theme with rationale/highlight
    if params.include_bullet_theme_scores:
        scored = await score_bullets_against_themes(
            bullets,
            list(themes),
            seed=seed,
            params=params,
            llm_model=llm_distill_model,  # LLooM uses gpt-4o-mini for scoring by default
            _client=openai_client,
            _semaphore=llm_sem,
        )
        if isinstance(scored, dict):
            scoring = dict(scoring or {})
            scoring.update(scored)

    elapsed_s = time.time() - t0
    return {
        "run_id": str(uuid.uuid4()),
        "seed": seed,
        "metadata_filters": metadata_filters,
        "text_source": ts,
        "params": {
            "n_quotes": params.n_quotes,
            "n_bullets": params.n_bullets,
            "bullet_words": params.bullet_words,
            "max_docs": params.max_docs,
            "max_themes": params.max_themes,
        },
        "counts": {
            "docs_total_after_meta_filters": total_docs_after_meta,
            "docs_total_after_text_source": total_docs_after_text_source,
            "docs_used": used_docs,
            "bullets": len(bullets),
            "clusters_labeled": len(themes),
        },
        "themes": list(themes),
        "points": bullets,
        "scores": scoring,
        "timing": {"elapsed_seconds": elapsed_s},
    }

