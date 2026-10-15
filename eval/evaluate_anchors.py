"""Score anchoring: build a template field from one PDF, find it in others.

    cd backend && .venv-ml/bin/python ../eval/evaluate_anchors.py [--detector auto|structure] [--show]

Cases:
  series  samples/series (make_report_series.py): template from January, located
          in Feb-Dec. Targets move pages, headings are reworded, a decoy table with
          identical columns appears, rows are added. manifest.json is the answer key.
  traps   real PDFs with near-duplicate tables in the same document (Apple's
          income statement vs comprehensive income; KBB brand vs model tables):
          the field must win clearly over its look-alikes.

Outcomes: correct (right page, IoU >= 0.5, status matched), flagged (status
ambiguous/not_found: a person reviews it - safe), silent error (matched but
wrong - the one that must not happen).
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import pdfplumber  # noqa: E402
import pypdfium2 as pdfium  # noqa: E402

from app.services.anchors import fingerprint, learn, locate, with_lookalikes  # noqa: E402
from app.services.detectors import DETECTORS  # noqa: E402
from app.services.extract import extract_region  # noqa: E402

DETECTOR = sys.argv[sys.argv.index("--detector") + 1] if "--detector" in sys.argv else "auto"
_cache: dict = {}
MARGINS: dict[str, list[float]] = {}


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


class Doc:
    def __init__(self, path: Path):
        self.path = path
        self.id = f"anchor-eval-{path.stem}-{path.parent.name}"
        d = pdfium.PdfDocument(str(path))
        self.texts = []
        for i in range(len(d)):
            tp = d[i].get_textpage()
            self.texts.append(tp.get_text_range())
        d.close()
        self.pdf = pdfplumber.open(path)

    def detect(self, page: int) -> dict:
        key = (self.id, page, DETECTOR)
        if key not in _cache:
            _cache[key] = DETECTORS[DETECTOR]["run"](self.id, self.path, page)
        return _cache[key]

    def extract(self, page: int, region: dict):
        kind = region["kind"] if region["kind"] != "heading" else "text"
        out = extract_region(self.pdf.pages[page - 1], region["bbox"], kind, ocr=False)
        return None if "needs_ocr" in out else out


LEARN = "--no-learn" not in sys.argv


def template_field(doc: Doc, page: int, bbox: list[float], kind: str) -> dict | None:
    fp = _template_field(doc, page, bbox, kind)
    if fp is not None and LEARN:
        fp = with_lookalikes(fp, doc.texts, doc.detect, doc.extract, page)
    return fp


def _template_field(doc: Doc, page: int, bbox: list[float], kind: str) -> dict | None:
    regions = doc.detect(page)["regions"]
    kinds = {"table": ("table",), "figure": ("chart_vector", "image"), "text": ("text",)}[kind]
    same = [r for r in regions if r["kind"] in kinds]
    if not same:
        return None
    region = max(same, key=lambda r: iou(r["bbox"], bbox))
    if iou(region["bbox"], bbox) < 0.3:
        return None
    return fingerprint(region, regions, doc.extract(page, region), page, len(doc.texts))


def contains(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / small if small else 0.0


def run_case(name, fp, target: Doc, page, bbox, tally, show):
    res = locate(fp, target.texts, target.detect, target.extract)
    if LEARN:
        fp.update(learn(fp, res))  # documents are processed in order, as a batch would
    best = res["best"]
    # Text blocks: the answer key's box is the text frame, the detected block is
    # the text itself - judge by containment rather than IoU.
    overlap = contains if fp["kind"] == "text" else iou
    right = bool(best) and best["page"] == page and overlap(best["bbox"], bbox) >= 0.5
    if res["status"] == "matched":
        outcome = "correct" if right else "SILENT ERROR"
    else:
        outcome = f"flagged ({res['status']}{', would have been right' if right else ''})"
    tally[outcome.split(" ")[0]] = tally.get(outcome.split(" ")[0], 0) + 1
    if best:
        MARGINS.setdefault(fp["kind"], []).append(res["margin"])
    if show or outcome != "correct":
        got = f"p{best['page']} score {best['score']:.2f} margin {res['margin']:.2f}" if best else "nothing"
        print(f"  {name:<44} expected p{page:<3} got {got:<32} {outcome}")
        if best and (show or outcome != "correct"):
            print(f"      signals {best['signals']} | anchor seen: {best['anchor'][:50]!r}")
            for c in res["candidates"][1:3]:
                print(f"      runner-up p{c['page']} {c['score']:.2f} {c['signals']} {c['anchor'][:40]!r}")


def main():
    show = "--show" in sys.argv
    tally: dict[str, int] = {}
    series = ROOT / "samples" / "series"
    manifest = json.loads((series / "manifest.json").read_text())
    files = sorted(manifest)
    if "--template" in sys.argv:  # build the template from another month
        t = files[int(sys.argv[sys.argv.index("--template") + 1]) - 1]
        files = [t] + [f for f in files if f != t]
    template_doc = Doc(series / files[0])
    print(f"series: template from {files[0]}, detector={DETECTOR}")
    for field, truth in manifest[files[0]].items():
        fp = template_field(template_doc, truth["page"], truth["bbox"], truth["kind"])
        if fp is None:
            print(f"  {field}: template region not detected in {files[0]}")
            continue
        for f in files[1:]:
            t = manifest[f][field]
            run_case(f"{field} -> {f}", fp, Doc(series / f), t["page"], t["bbox"], tally, show)

    print("traps: near-duplicate tables in one real document")
    traps = [
        ("e0dac020-713e-47ce-809a-f58dd78dfa91.pdf", 4, [0.03, 0.215, 0.969, 0.722], "income statement"),
        ("e0dac020-713e-47ce-809a-f58dd78dfa91.pdf", 5, [0.03, 0.12, 0.969, 0.36], "comprehensive income"),
        ("Q3-2026-Kelley-Blue-Book-EV-Sales-Report-updated-as-of-10-13-26.pdf", 2, [0.046, 0.068, 0.946, 0.589], "KBB brand table"),
        ("Q3-2026-Kelley-Blue-Book-EV-Sales-Report-updated-as-of-10-13-26.pdf", 3, [0.046, 0.068, 0.952, 0.92], "KBB model table"),
    ]
    for fname, page, bbox, name in traps:
        path = ROOT / fname
        if not path.exists():
            continue
        doc = Doc(path)
        fp = template_field(doc, page, bbox, "table")
        if fp is None:
            print(f"  {name}: template region not detected")
            continue
        run_case(name, fp, doc, page, bbox, tally, show)

    for kind, ms in MARGINS.items():
        print(f"  {kind:<6} margin over runner-up: min {min(ms):.2f}, median {sorted(ms)[len(ms) // 2]:.2f}")
    total = sum(tally.values())
    print(f"\n{total} lookups: " + ", ".join(f"{k} {v}" for k, v in sorted(tally.items())))


if __name__ == "__main__":
    main()
