"""
Shared helpers for search result improvement: fulltext snippet highlighting,
longform_response inference, and full_text for "View full interview".
"""
import html
import re
from typing import Any, Dict, List, Optional, Tuple

# Heuristic: excerpt looks like "metadata dump + narrative" (e.g. "46 38 San Diego ... $89k Weekly No food restrictions San Diego's pizza...")
METADATA_PREFIX_LOOKAHEAD = 140  # chars to check for metadata patterns


def looks_like_metadata_prefixed_excerpt(text: str) -> bool:
    """
    Return True if the result text appears to start with concatenated metadata
    (e.g. participant id, age, city, income, frequency) rather than narrative.
    Such results are filtered out so only narrative excerpts surface.
    """
    if not text or not text.strip():
        return False
    prefix = text[:METADATA_PREFIX_LOOKAHEAD]
    # Starts with digits (e.g. participant id, age) and contains metadata-like tokens
    if not re.match(r"^\d+\s+\d+", prefix):
        return False
    metadata_indicators = (
        re.search(r"\$\d+k?", prefix)  # income like $89k
        or "Weekly" in prefix
        or "Monthly" in prefix
        or "No food restrictions" in prefix
        or re.search(r"\b(California|West)\b", prefix)
    )
    return bool(metadata_indicators)


def infer_longform_response(
    metadata: Dict[str, Any], quote_text: str, matching_sentences: Optional[List[str]] = None
) -> Optional[int]:
    """Return 1–5 if quote_text (or any of matching_sentences) appears in qN_response, else None."""
    for n in range(1, 6):
        key = f"q{n}_response"
        val = metadata.get(key)
        if val is None:
            continue
        txt = str(val).strip()
        if not txt:
            continue
        if quote_text.strip() in txt:
            return n
        if matching_sentences:
            for s in matching_sentences:
                if s.strip() in txt:
                    return n
    return None


def sentence_matching_snippet_and_longform(
    full_text: str, query: str, metadata: Dict[str, Any]
) -> Tuple[str, str, Optional[int]]:
    """
    For fulltext results: return (plain_snippet, highlighted_html, longform_response).
    - plain_snippet: sentence(s) containing any query term
    - highlighted_html: same with <mark> around matched terms (HTML-safe)
    - longform_response: 1–5 if the snippet comes from q1_response..q5_response, else None
    """
    if not full_text:
        return "", "", None
    base_words = re.findall(r"\b\w+\b", (query or "").lower())
    # Add simple singular/plural variants so "pineapples" highlights "pineapple" and vice versa.
    query_words = set()
    for w in base_words:
        w = w.strip().lower()
        if not w:
            continue
        query_words.add(w)
        if len(w) > 4 and w.endswith("ies"):
            query_words.add(w[:-3] + "y")
        if len(w) > 4 and w.endswith("es"):
            query_words.add(w[:-2])
        if len(w) > 3 and w.endswith("s"):
            query_words.add(w[:-1])
        if len(w) > 2 and not w.endswith("s"):
            query_words.add(w + "s")
        if len(w) > 2 and w.endswith("y"):
            query_words.add(w[:-1] + "ies")
    query_words = [w for w in sorted(query_words) if len(w) >= 3]
    sentences = re.split(r"(?<=[.!?])\s+", full_text)
    if not query_words:
        snippet = " ".join(sentences[:2]) if len(sentences) >= 2 else (full_text[:400] + "..." if len(full_text) > 400 else full_text)
        return snippet, html.escape(snippet), None
    matching: List[str] = [
        s for s in sentences
        if any(re.search(rf"\b{re.escape(w)}\b", s.lower()) for w in query_words)
    ]
    if not matching:
        snippet = " ".join(sentences[:2]) if len(sentences) >= 2 else (full_text[:400] + "..." if len(full_text) > 400 else full_text)
    else:
        snippet = " ".join(matching[:5])  # up to 5 matching sentences
    # Build highlighted_html: escape snippet, then wrap query-term spans
    escaped = html.escape(snippet)
    spans: List[Tuple[int, int]] = []
    for w in query_words:
        for m in re.finditer(rf"\b{re.escape(w)}\b", escaped, re.IGNORECASE):
            spans.append((m.start(), m.end()))
    spans.sort(key=lambda x: (x[0], -x[1]))
    merged: List[Tuple[int, int]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    out: List[str] = []
    prev = 0
    for s, e in merged:
        out.append(escaped[prev:s])
        out.append('<mark class="search-hit">')
        out.append(escaped[s:e])
        out.append("</mark>")
        prev = e
    out.append(escaped[prev:])
    highlighted_html = "".join(out)
    longform = infer_longform_response(metadata, snippet, matching_sentences=matching)
    return snippet, highlighted_html, longform


def enrich_fulltext_results(results: Dict[str, Any], query: str) -> None:
    """
    For each fulltext result, set snippet, highlighted_html, full_text, longform_response.
    For other results (e.g. semantic / hybrid), set full_text / snippet when possible
    and set longform_response when inferable from metadata.
    Mutates results['results'] in place.
    """
    arr = results.get("results") or []
    for r in arr:
        if r.get("search_type") != "fulltext":
            continue
        text = r.get("text") or ""
        meta = r.get("metadata") or {}
        snippet, highlighted_html, longform = sentence_matching_snippet_and_longform(text, query, meta)
        r["full_text"] = text
        r["text"] = snippet
        r["highlighted_html"] = highlighted_html
        if longform is not None:
            r["longform_response"] = longform

    # For semantic / hybrid results, many items only have a single "text" field.
    # To support contextual snippets (and the "full transcript" view), we treat that
    # text as the full response and derive a shorter snippet from it.
    for r in arr:
        if r.get("search_type") != "semantic":
            continue
        if r.get("full_text"):
            continue
        text = r.get("text") or ""
        if not text:
            continue
        meta = r.get("metadata") or {}
        snippet, highlighted_html, longform = sentence_matching_snippet_and_longform(
            text, query, meta
        )
        r["full_text"] = text
        r["text"] = snippet
        if "highlighted_html" not in r:
            r["highlighted_html"] = highlighted_html
        if longform is not None and "longform_response" not in r:
            r["longform_response"] = longform

    for r in arr:
        if "longform_response" in r:
            continue
        meta = r.get("metadata") or {}
        quote = r.get("text") or ""
        inferred = infer_longform_response(meta, quote, matching_sentences=None)
        if inferred is not None:
            r["longform_response"] = inferred


def filter_metadata_prefixed_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove results whose excerpt text starts with concatenated metadata
    (e.g. "46 38 San Diego California West $89k Weekly No food restrictions...")
    so only narrative-only excerpts surface.
    """
    return [r for r in results if not looks_like_metadata_prefixed_excerpt(r.get("text") or "")]
