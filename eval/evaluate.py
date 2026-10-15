"""Score layout detectors against eval/ground_truth.json.

    cd backend && .venv-ml/bin/python ../eval/evaluate.py [structure docling paddle auto] [--misses]

Metrics (data regions = the tables and figures an analyst would extract):
  table / figure recall  found with IoU >= 0.5 and the right kind
  wrong kind             found the box but called a table a figure or vice versa
  bad boxes              boxes that cut or merge data regions (overlap one without matching)
  text kept              share of ordinary text (outside data regions) that ends up in a
                         text or heading region, i.e. was not swallowed by a figure box
ML detector outputs are cached in eval/.cache so rule changes re-score in seconds.
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import pdfplumber  # noqa: E402

from app.services.detectors import DETECTORS, is_available  # noqa: E402

CACHE = ROOT / "eval" / ".cache"
COARSE = {"table": "table", "chart_vector": "figure", "image": "figure"}


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def covered(a, b):
    """Share of `a` lying inside `b`."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1])
    return ix * iy / area if area else 0.0


def run(strategy: str, doc: str, page: int) -> tuple[list[dict], float]:
    path = ROOT / doc
    cacheable = strategy in ("docling", "paddle")
    cache_file = CACHE / f"{strategy}-{Path(doc).stem}-{page}.json"
    if cacheable and cache_file.exists():
        out = json.loads(cache_file.read_text())
    else:
        out = DETECTORS[strategy]["run"](f"eval-{strategy}-{doc}", path, page)
        if cacheable:
            CACHE.mkdir(exist_ok=True)
            cache_file.write_text(json.dumps(out))
    return out["regions"], out["ms"]


def words_of(doc: str, page: int) -> list[list[float]]:
    with pdfplumber.open(ROOT / doc) as pdf:
        p = pdf.pages[page - 1]
        cx0, ctop, cx1, cbottom = p.cropbox
        w, h = cx1 - cx0, cbottom - ctop
        return [
            [(x["x0"] - cx0) / w, (x["top"] - ctop) / h, (x["x1"] - cx0) / w, (x["bottom"] - ctop) / h]
            for x in p.extract_words()
        ]


def score(strategy: str, gt: dict, show_misses: bool) -> dict:
    tot = {"table": [0, 0], "figure": [0, 0], "wrong_kind": 0, "bad": 0, "text": [0, 0], "ms": []}
    misses = []
    for entry in gt["pages"]:
        doc, page = entry["doc"], entry["page"]
        regions, ms = run(strategy, doc, page)
        tot["ms"].append(ms)
        preds = [r for r in regions if r["kind"] in COARSE]
        used = set()
        for g in entry["regions"]:
            gk = "figure" if g["kind"] == "any" else g["kind"]  # "any": either kind is correct
            tot[gk][1] += 1
            best = max(
                ((iou(g["bbox"], p["bbox"]), i) for i, p in enumerate(preds) if i not in used),
                default=(0, None),
            )
            if best[0] >= 0.5:
                used.add(best[1])
                if g["kind"] == "any" or COARSE[preds[best[1]]["kind"]] == g["kind"]:
                    tot[gk][0] += 1
                else:
                    tot["wrong_kind"] += 1
                    misses.append(f"{Path(doc).stem[:14]} p{page}: {g['note']} -> wrong kind ({preds[best[1]]['kind']})")
            else:
                misses.append(f"{Path(doc).stem[:14]} p{page}: {g['note']} -> missed (best IoU {best[0]:.2f})")
        for i, p in enumerate(preds):
            if i in used:
                continue
            if any(covered(g["bbox"], p["bbox"]) > 0.1 or covered(p["bbox"], g["bbox"]) > 0.5 for g in entry["regions"]):
                tot["bad"] += 1
                misses.append(f"{Path(doc).stem[:14]} p{page}: bad {p['kind']} box {[round(v, 2) for v in p['bbox']]}")
        # Ordinary text: words outside every data region must land in a text/heading region.
        texts = [r["bbox"] for r in regions if r["kind"] in ("text", "heading")]
        lost = []
        for wb in words_of(doc, page):
            if any(covered(wb, g["bbox"]) > 0.5 for g in entry["regions"]):
                continue
            tot["text"][1] += 1
            ok = any(covered(wb, t) > 0.5 for t in texts)
            tot["text"][0] += ok
            if not ok:
                lost.append(wb)
        if len(lost) > 5:
            hider = [p["kind"] for p in preds if any(covered(w, p["bbox"]) > 0.5 for w in lost)]
            misses.append(f"{Path(doc).stem[:14]} p{page}: {len(lost)} words of ordinary text not in a text region"
                          f" (under: {sorted(set(hider)) or 'nothing - not grouped'})")
    if show_misses:
        print(f"\n--- {strategy} misses ---")
        for m in misses:
            print("  ", m)
    return tot


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    strategies = [s for s in (args or ["structure", "docling", "paddle"]) if s in DETECTORS and is_available(s)]
    gt = json.loads((ROOT / "eval" / "ground_truth.json").read_text())
    rows = []
    for s in strategies:
        t0 = time.perf_counter()
        rows.append((s, score(s, gt, "--misses" in sys.argv), time.perf_counter() - t0))
    print(f"\n{'strategy':<11}{'tables':>9}{'figures':>10}{'wrong kind':>12}{'bad boxes':>11}{'text kept':>11}{'ms/page':>9}")
    for s, t, _ in rows:
        tr, fr = t["table"], t["figure"]
        kept = 100 * t["text"][0] / max(1, t["text"][1])
        print(
            f"{s:<11}{tr[0]:>4}/{tr[1]:<4}{fr[0]:>5}/{fr[1]:<4}{t['wrong_kind']:>12}{t['bad']:>11}{kept:>10.0f}%{sum(t['ms']) / len(t['ms']):>9.0f}"
        )


if __name__ == "__main__":
    main()
