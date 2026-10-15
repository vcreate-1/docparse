"""Compare layout detectors on a PDF: speed, what each finds, and agreement.

    cd backend && .venv-ml/bin/python ../scripts/compare_detectors.py ../some.pdf [--pages 1-10]

Two boxes "agree" when they have the same kind and IoU >= 0.5. Each detector
is compared against every other, so you can see e.g. tables one finds that
another misses.
"""

import argparse
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.services.detectors import DETECTORS, is_available  # noqa: E402

KINDS = ["table", "chart_vector", "image", "heading", "text", "scanned_page"]


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def matched(r, others):
    return any(o["kind"] == r["kind"] and iou(r["bbox"], o["bbox"]) >= 0.5 for o in others)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--pages", default=None, help="e.g. 1-10")
    args = ap.parse_args()

    import pypdfium2 as pdfium

    n = len(pdfium.PdfDocument(args.pdf))
    lo, hi = (map(int, args.pages.split("-")) if args.pages else (1, n))
    pages = range(lo, min(hi, n) + 1)

    # Work on a temp copy so the page cache keys don't collide with app data.
    tmp = Path(tempfile.mkdtemp()) / "doc.pdf"
    shutil.copy(args.pdf, tmp)
    names = [d for d in DETECTORS if is_available(d)]
    results: dict[str, dict[int, list]] = {d: {} for d in names}
    timing: dict[str, list[float]] = {d: [] for d in names}

    for d in names:
        DETECTORS[d]["run"]("cmp", tmp, pages[0])  # warm-up: model load isn't per-page cost
        for p in pages:
            out = DETECTORS[d]["run"]("cmp", tmp, p)
            results[d][p] = [r for r in out["regions"] if r["kind"] != "scanned_page"]
            timing[d].append(out["ms"])
        print(f"ran {d} on {len(pages)} pages", file=sys.stderr)

    print(f"\n{args.pdf}  pages {lo}-{pages[-1]}\n")
    print(f"{'detector':<12}{'ms/page':>9}  " + "  ".join(f"{k[:7]:>7}" for k in KINDS))
    for d in names:
        counts = Counter(r["kind"] for rs in results[d].values() for r in rs)
        avg = sum(timing[d]) / len(timing[d])
        print(f"{d:<12}{avg:>9.0f}  " + "  ".join(f"{counts.get(k, 0):>7}" for k in KINDS))

    print("\nAgreement: share of A's boxes that B also found (same kind, IoU >= 0.5)")
    for a in names:
        for b in names:
            if a == b:
                continue
            for kind in ("table", "chart_vector", "image", "heading"):
                mine = [(p, r) for p, rs in results[a].items() for r in rs if r["kind"] == kind]
                if not mine:
                    continue
                hit = sum(matched(r, results[b][p]) for p, r in mine)
                missed = [p for p, r in mine if not matched(r, results[b][p])]
                note = f"  missed on pages {sorted(set(missed))[:12]}" if missed else ""
                print(f"  {a:>9} -> {b:<9} {kind:<13} {hit}/{len(mine)}{note}")


if __name__ == "__main__":
    main()
