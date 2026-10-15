"""Anchoring: find a template field again in a different PDF.

A template field is stored as a *fingerprint* of what it is, not where it was:

    kind      table / figure / text
    anchor    its own title, else the heading above it        "Segment Revenue"
    context   text just above it (heading + caption)          "Revenue by business segment for..."
    header    column names, with periods generalised          "segment | <month> <year> | qtd | ytd"
    labels    row labels (tables), legend/axis words (figures), opening words (text)
    position  page position in the document and box on the page (a weak hint)

To locate it in another PDF: shortlist pages by text, detect regions on just
those pages, score every same-kind region on every signal, and return the best
with its score and its margin over the runner-up. A thin margin is reported
as "ambiguous" for review rather than guessed - two tables with identical
columns (revenue vs costs) differ only in their headings, and a template must
not silently pick the wrong one.
"""

import re
from collections.abc import Callable
from difflib import SequenceMatcher

KIND = {"table": "table", "chart_vector": "figure", "image": "figure", "text": "text", "heading": "text"}

MONTH = r"jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|june?|july?|aug(ust)?|sep(t(ember)?)?|oct(ober)?|nov(ember)?|dec(ember)?"
GENERALISE = [
    (re.compile(rf"\b({MONTH})\b\.?", re.I), "<month>"),
    (re.compile(r"\b(fy|cy|h[12]|q[1-4])\s*'?\d{2,4}\b", re.I), "<period>"),
    (re.compile(r"\bq[1-4]\b", re.I), "<quarter>"),
    (re.compile(r"\b(19|20)\d\d\b"), "<year>"),
    (re.compile(r"[-−(]?[$€£₹]?\d[\d,]*\.?\d*%?\)?"), "<n>"),
]
STOP = {"the", "of", "and", "for", "by", "in", "to", "a", "an", "on", "as", "at", "is", "wise"}

WEIGHTS = {
    "table": {"anchor": 0.35, "header": 0.20, "labels": 0.30, "position": 0.15},
    "figure": {"anchor": 0.55, "labels": 0.25, "position": 0.20},
    "text": {"anchor": 0.55, "labels": 0.25, "position": 0.20},
}
MATCH_SCORE = 0.55
MIN_MARGIN = 0.08
CANDIDATE_PAGES = 4


def generalise(text: str) -> str:
    text = (text or "").lower()
    for pattern, token in GENERALISE:
        text = pattern.sub(token, text)
    return " ".join(text.split())


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"<\w+>|[a-z]{2,}", generalise(text)) if t not in STOP}


def similarity(a: str, b: str, weight: dict[str, float] | None = None) -> float:
    """Blend of word overlap (order-free) and character similarity (typos).

    With `weight`, overlap counts each word by how distinctive it is among the
    candidates: "segment" in both "Segment Revenue" and "Segment Costs" says
    little; "revenue" vs "costs" decides it.
    """
    if not a or not b:
        return 0.0
    ta, tb = _tokens(a), _tokens(b)
    if weight is None:
        overlap = len(ta & tb) / len(ta | tb) if ta | tb else 0.0
    else:
        w = lambda t: weight.get(t, max(weight.values(), default=1.0))  # noqa: E731
        union = sum(w(t) for t in ta | tb)
        overlap = sum(w(t) for t in ta & tb) / union if union else 0.0
    ratio = SequenceMatcher(None, generalise(a), generalise(b)).ratio()
    return 0.7 * overlap + 0.3 * ratio


def set_similarity(a: list[str], b: list[str]) -> float | None:
    """Fuzzy overlap of two label sets ("lowa" still matches "Iowa")."""
    a = [_unconfuse(generalise(x)) for x in a if x and x.strip()]
    b = [_unconfuse(generalise(x)) for x in b if x and x.strip()]
    if not a and not b:
        return None  # nothing to compare: neutral, not a mismatch
    if not a or not b:
        return 0.0
    matched = sum(1 for x in a if any(x == y or SequenceMatcher(None, x, y).ratio() > 0.85 for y in b))
    return matched / max(len(a), len(b))


_CONFUSABLE = str.maketrans({"l": "i", "1": "i", "|": "i", "0": "o"})


def _unconfuse(text: str) -> str:
    """OCR's usual swaps (I/l/1, O/0) compared as equal: "lowa" is "Iowa"."""
    return text.translate(_CONFUSABLE)


# ---------------------------------------------------------------- fingerprint

def context_above(regions: list[dict], bbox: list[float], reach: float = 0.12) -> str:
    """Heading and caption text sitting just above a box, nearest first."""
    x0, top, x1, _ = bbox
    near = [
        r for r in regions
        if r["kind"] in ("heading", "text") and top - reach <= r["bbox"][3] <= top + 0.01
        and r["bbox"][0] < x1 and r["bbox"][2] > x0
    ]
    near.sort(key=lambda r: -r["bbox"][3])
    # Nearest first, up to and including the field's own heading: text above
    # that heading belongs to the previous section.
    picked = []
    for r in near[:3]:
        picked.append(r["label"])
        if r["kind"] == "heading":
            break
    return " ".join(picked)


def nearest_heading(regions: list[dict], bbox: list[float]) -> str | None:
    x0, top, x1, _ = bbox
    above = [r for r in regions if r["kind"] == "heading" and r["bbox"][3] <= top + 0.01
             and r["bbox"][0] < x1 and r["bbox"][2] > x0 - 0.05]
    return max(above, key=lambda r: r["bbox"][3])["label"] if above else None


def fingerprint(region: dict, page_regions: list[dict], extraction: dict | None,
                page_no: int, page_count: int) -> dict:
    kind = KIND.get(region["kind"], "text")
    extraction = extraction or {}
    title = extraction.get("title") or region.get("title")
    anchor = title or nearest_heading(page_regions, region["bbox"]) or ""
    if kind == "table":
        labels = [row[0] for row in extraction.get("data") or [] if row and row[0]]
        header = [generalise(c) for c in extraction.get("columns") or []]
    elif kind == "figure":
        labels, header = list(extraction.get("labels") or [])[:40], []
    else:
        labels, header = (extraction.get("text") or region.get("label", "")).split()[:30], []
    return {
        "kind": kind,
        "anchor": anchor,
        "context": context_above(page_regions, region["bbox"]),
        "header": header,
        "labels": labels,
        "page_ratio": (page_no - 1) / max(1, page_count - 1),
        "bbox": region["bbox"],
        "shape": [len(extraction.get("data") or []), len(extraction.get("columns") or [])],
    }


# ---------------------------------------------------------------- locate

def shortlist_pages(fp: dict, page_texts: list[str]) -> list[int]:
    """Pages whose text shares the most distinctive words with the fingerprint,
    plus the page at the same relative position (layouts drift, rarely jump)."""
    import math

    pages = [_tokens(t) for t in page_texts]
    n = len(pages) or 1
    # Words on most pages ("performance", "month") say little about where the
    # field is; rare ones ("revenue", "agriculture") say a lot.
    rarity = lambda t: math.log(1 + n / (1 + sum(1 for p in pages if t in p)))  # noqa: E731
    names = " ".join([fp["anchor"], *fp.get("anchor_aliases", [])])
    want = {t: 2 * rarity(t) for t in _tokens(names)}
    for t in _tokens(" ".join([fp["context"], " ".join(fp["labels"][:20])])):
        want.setdefault(t, rarity(t))
    total = sum(want.values()) or 1
    scores = []
    for i, have in enumerate(pages):
        scores.append((sum(w for t, w in want.items() if t in have) / total, i + 1))
    scores.sort(reverse=True)
    shortlist = [p for s, p in scores[:CANDIDATE_PAGES] if s > 0]
    expected = 1 + round(fp["page_ratio"] * (len(page_texts) - 1))
    for p in (expected, expected - 1, expected + 1):
        if 1 <= p <= len(page_texts) and p not in shortlist:
            shortlist.append(p)
    return shortlist


def distinctiveness(cand_fps: list[dict]) -> dict[str, float]:
    """IDF-style weight of each word across the candidates' anchors/contexts."""
    import math

    docs = [_tokens(f"{c['anchor']} {c['context']}") for c in cand_fps]
    n = len(docs) or 1
    vocab = set().union(*docs) if docs else set()
    return {t: math.log(1 + n / sum(1 for d in docs if t in d)) for t in vocab}


def _anchor_match(names: list[str], context: str, cand_fp: dict, weight) -> float:
    best = 0.0
    for name in names:
        best = max(best,
                   similarity(name, cand_fp["anchor"], weight),
                   0.8 * similarity(name, cand_fp["context"], weight))
    return max(best, 0.6 * similarity(context, cand_fp["context"], weight))


def score_candidate(fp: dict, cand: dict, cand_fp: dict, weight: dict | None = None) -> tuple[float, dict]:
    kind = fp["kind"]
    parts: dict[str, float] = {}
    names = [fp["anchor"], *fp.get("anchor_aliases", [])]
    parts["anchor"] = _anchor_match(names, fp["context"], cand_fp, weight)
    if kind == "table":
        parts["header"] = set_similarity(fp["header"], cand_fp["header"]) or 0.0
    labels = set_similarity(fp["labels"], cand_fp["labels"])
    parts["labels"] = 0.5 if labels is None else labels
    page_shift = abs(fp["page_ratio"] - cand_fp["page_ratio"])
    fx, fy = (fp["bbox"][0] + fp["bbox"][2]) / 2, (fp["bbox"][1] + fp["bbox"][3]) / 2
    cx, cy = (cand["bbox"][0] + cand["bbox"][2]) / 2, (cand["bbox"][1] + cand["bbox"][3]) / 2
    parts["position"] = max(0.0, 1 - 1.5 * page_shift - abs(fx - cx) - abs(fy - cy))
    weights = WEIGHTS[kind]
    score = sum(weights[k] * parts[k] for k in weights)
    # Known look-alikes: a candidate resembling one of them more than the field
    # itself is that look-alike ("Segment Costs" when we want "Segment Revenue").
    for decoy in fp.get("lookalikes", []):
        resemblance = similarity(decoy, cand_fp["anchor"], weight)
        if resemblance > parts["anchor"]:
            parts["lookalike"] = round(resemblance, 2)
            score -= 0.5 * (resemblance - parts["anchor"])
    return score, parts


def locate(fp: dict, page_texts: list[str],
           detect: Callable[[int], dict],
           extract: Callable[[int, dict], dict | None]) -> dict:
    """Best match for a fingerprint in another document.

    detect(page)          -> that page's region result (cached Auto detection)
    extract(page, region) -> text-layer reading of a candidate (None to skip)
    """
    found = []
    page_count = len(page_texts)
    for page in shortlist_pages(fp, page_texts):
        regions = detect(page)["regions"]
        for r in regions:
            if KIND.get(r["kind"]) != fp["kind"]:
                continue
            if fp["kind"] == "text" and r["kind"] == "heading":
                continue
            found.append((page, r, fingerprint(r, regions, extract(page, r), page, page_count)))
    weight = distinctiveness([c for _, _, c in found])
    candidates = []
    for page, r, cand_fp in found:
        score, parts = score_candidate(fp, r, cand_fp, weight)
        candidates.append({"page": page, "bbox": r["bbox"], "kind": r["kind"], "score": round(score, 3),
                           "signals": {k: round(v, 2) for k, v in parts.items()},
                           "anchor": cand_fp["anchor"]})
    candidates.sort(key=lambda c: -c["score"])
    best = candidates[0] if candidates else None
    margin = best["score"] - candidates[1]["score"] if len(candidates) > 1 else (best["score"] if best else 0)
    if best is None or best["score"] < 0.4:
        status = "not_found"
    elif best["score"] < MATCH_SCORE or margin < MIN_MARGIN:
        status = "ambiguous"
    else:
        status = "matched"
    return {"status": status, "best": best, "margin": round(margin, 3), "candidates": candidates[:5]}


# ---------------------------------------------------------------- template learning

def with_lookalikes(fp: dict, page_texts: list[str], detect, extract, own_page: int) -> dict:
    """Record the field's look-alikes in its own sample document.

    Run when the field is created: whatever else in the sample scores close to
    the field (same columns, same rows, different heading) is stored as a
    negative, so later documents can't confuse the two.
    """
    res = locate(fp, page_texts, detect, extract)
    lookalikes = []
    for c in res["candidates"]:
        same = c["page"] == own_page and _iou(c["bbox"], fp["bbox"]) > 0.5
        if not same and c["score"] >= 0.4 and c["anchor"] and similarity(c["anchor"], fp["anchor"]) < 0.9:
            lookalikes.append(c["anchor"])
    return {**fp, "lookalikes": sorted(set(lookalikes))}


def learn(fp: dict, result: dict, confirmed: bool = False) -> dict:
    """Fold a match back into the fingerprint: its heading becomes an alias.

    Only from a confident match or one a person confirmed - learning from a
    wrong match would teach the template the wrong thing.
    """
    best = result.get("best")
    if not best or not best["anchor"]:
        return fp
    if not confirmed and not (result["status"] == "matched" and result["margin"] >= 0.15):
        return fp
    aliases = list(fp.get("anchor_aliases", []))
    known = [fp["anchor"], *aliases]
    if all(generalise(best["anchor"]) != generalise(k) for k in known):
        aliases.append(best["anchor"])
    return {**fp, "anchor_aliases": aliases[-12:]}


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0
