"""Keyword page search over the text layer ("which page has X").

This is the fast, exact signal. The RAG facility's semantic retrieval will be
added as a second signal and the two scores blended; this module stays as the
instant first answer.
"""

import math
import re
from functools import lru_cache

from ..storage import storage

WORD = re.compile(r"[a-z0-9][a-z0-9.%-]*")


def build_index(page_texts: list[str]) -> list[str]:
    return [" ".join(t.split()) for t in page_texts]  # collapse whitespace


@lru_cache(maxsize=64)
def _pages(doc_id: str) -> tuple[str, ...]:
    return tuple(storage.read_json(doc_id, "pages.json") or [])


def search(doc_id: str, query: str, limit: int = 10) -> list[dict]:
    pages = _pages(doc_id)
    terms = list(dict.fromkeys(WORD.findall(query.lower())))
    if not terms or not pages:
        return []
    lowered = [p.lower() for p in pages]
    phrase = " ".join(terms)

    # IDF so that "revenue" on every page counts less than "segment".
    df = {t: sum(1 for p in lowered if t in p) for t in terms}
    idf = {t: math.log(1 + len(pages) / (1 + df[t])) for t in terms}

    hits = []
    for i, text in enumerate(lowered):
        matched = [t for t in terms if t in text]
        if not matched:
            continue
        score = sum(idf[t] * (1 + math.log(text.count(t))) for t in matched)
        score *= len(matched) / len(terms)  # pages matching every term rank first
        if len(terms) > 1 and phrase in text:
            score *= 2
        hits.append((score, i))

    hits.sort(reverse=True)
    return [
        {"page": i + 1, "score": round(s, 3), "snippet": _snippet(pages[i], terms)}
        for s, i in hits[:limit]
    ]


def _snippet(text: str, terms: list[str], width: int = 160) -> str:
    lowered = text.lower()
    pos = min((lowered.find(t) for t in terms if t in lowered), default=0)
    start = max(0, pos - width // 3)
    snippet = text[start : start + width]
    return ("..." if start else "") + snippet + ("..." if start + width < len(text) else "")
