"""Score picture-chart reading (vision model + OCR check) against eval/chart_truth.json.

    cd backend && DOCPARSE_LLM_BASE_URL=http://localhost:1234/v1 DOCPARSE_LLM_MODEL=qwen/qwen3.8-27b \\
        .venv-ml/bin/python ../eval/evaluate_charts.py
"""

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.services import llm  # noqa: E402
from app.services.extract import run_ocr  # noqa: E402


def key(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


def find(rows, series, x):
    """The row whose x label is closest to `x` (not the first partial match:
    "Grain" must not stand in for "Farm products excl. grain")."""
    from difflib import SequenceMatcher

    cands = [r for r in rows if series is None or key(series) in key(r[0])]
    if not cands:
        return None
    best = max(cands, key=lambda r: SequenceMatcher(None, key(x), key(r[1])).ratio())
    return best if SequenceMatcher(None, key(x), key(best[1])).ratio() >= 0.6 else None


def num(s):
    try:
        return float(str(s).replace(",", ""))
    except ValueError:
        return None


def main():
    if llm.vision_client() is None:
        sys.exit("no vision model configured (LM Studio with a 'vlm' model, or OpenAI)")
    truth = json.loads((ROOT / "eval" / "chart_truth.json").read_text())
    good = total = 0
    for ch in truth["charts"]:
        t0 = time.perf_counter()
        out = run_ocr({"needs_ocr": "chart"}, ROOT / ch["doc"], ch["page"], ch["bbox"])
        secs = time.perf_counter() - t0
        rows, status = out.get("data") or [], out.get("row_status") or []
        st = {id(r): status[i] if i < len(status) else "?" for i, r in enumerate(rows)}
        ok = n = 0
        misses = []
        for series, x, v in ch.get("printed", []):
            n += 1
            r = find(rows, series, x)
            hit = r is not None and (num(r[2]) == v or str(v) in r[3].replace(",", "")) and st[id(r)] == "exact"
            ok += hit
            if not hit:
                misses.append(f"{series or ''} {x}={v} got {r[2:5] + [st[id(r)]] if r else 'nothing'}")
        for series, x, v, tol in ch.get("estimated", []):
            n += 1
            r = find(rows, series, x)
            hit = r is not None and num(r[2]) is not None and abs(num(r[2]) - v) <= tol * v and st[id(r)] == "approximate"
            ok += hit
            if not hit:
                misses.append(f"estimate {series} {x}~{v} got {r[2:5] + [st[id(r)]] if r else 'nothing'}")
        for x, text in ch.get("printed_text", []):
            n += 1
            r = find(rows, None, x)
            hit = r is not None and key(text) in key(r[3]) and st[id(r)] == "exact"
            ok += hit
            if not hit:
                misses.append(f"{x}={text} got {r[2:5] + [st[id(r)]] if r else 'nothing'}")
        if ch.get("none_printed"):
            n += 1
            falsely = [r for r in rows if st[id(r)] != "approximate"]
            hit = not falsely
            ok += hit
            if not hit:
                misses.append(f"claimed printed values on an unlabelled chart: {[r[:3] for r in falsely][:4]}")
        good += ok
        total += n
        print(f"{Path(ch['doc']).stem[:16]:<17} p{ch['page']:<3} {ok:>3}/{n:<3} {out['method']:<14} {out['confidence']:<12} {secs:5.0f}s")
        for m in misses[:6]:
            print("      miss:", m)
    print(f"\n{good}/{total} chart values read and labelled correctly")


if __name__ == "__main__":
    main()
