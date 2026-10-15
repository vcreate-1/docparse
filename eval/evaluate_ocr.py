"""Score OCR extraction of picture-only regions against eval/ocr_truth.json.

    cd backend && .venv-ml/bin/python ../eval/evaluate_ocr.py [--show]

A pair counts when its label and value appear in the same output row (or the
same text line for text output). Reports per region: pairs found, the engine
used, confidence label and any issues raised - an issue on a correct reading
is a false alarm, no issue on a wrong one is a silent error.
"""

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.extract import run_ocr  # noqa: E402


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9%.\-]", "", s.lower().replace("−", "-").replace("–", "-"))


def main():
    truth = json.loads((ROOT / "eval" / "ocr_truth.json").read_text())
    total_found = total = 0
    for r in truth["regions"]:
        what = "table" if r["kind"] == "table" else "chart"
        t0 = time.perf_counter()
        out = run_ocr({"needs_ocr": what}, ROOT / r["doc"], r["page"], r["bbox"])
        secs = time.perf_counter() - t0
        rows = out.get("data") or []
        if out.get("columns"):
            rows = [out["columns"], *rows]
        texts = [" ".join(row) for row in rows] or (out.get("labels") or [])
        found, missing = 0, []
        for label, value in r["pairs"]:
            # labels compared without punctuation: "Land &. Shell" is the same label
            ok = any(norm(label).replace(".", "")[:12] in norm(t).replace(".", "") and norm(value) in norm(t) for t in texts)
            found += ok
            if not ok:
                missing.append(f"{label}={value}")
        total_found += found
        total += len(r["pairs"])
        print(f"{Path(r['doc']).stem[:16]:<17} p{r['page']:<3} {r['kind']:<6} {found:>3}/{len(r['pairs']):<3} "
              f"{out['method']:<14} {out['confidence']:<12} {secs:4.1f}s  issues: {out.get('issues') or 'none'}")
        if "--show" in sys.argv or missing:
            if missing:
                print("      missing:", missing[:8])
            if "--show" in sys.argv:
                for row in rows[:6]:
                    print("      ", row)
    print(f"\n{total_found}/{total} label/value pairs read correctly")


if __name__ == "__main__":
    main()
