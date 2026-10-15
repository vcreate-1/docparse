"""Score field extraction from text against eval/text_truth.json.

    cd backend && DOCPARSE_LLM_BASE_URL=http://localhost:1234/v1 DOCPARSE_LLM_MODEL=qwen/qwen3.8-27b \\
        .venv/bin/python ../eval/evaluate_text.py

Needs a configured model (facility, OpenAI or a local OpenAI-compatible server).
A field counts when its value is right AND it wasn't flagged unsupported; a right
value that was flagged is a false alarm, a wrong value that passed is a silent error.
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import pdfplumber  # noqa: E402

from app.services import llm  # noqa: E402
from app.services.extract import extract_region  # noqa: E402
from app.services.text_extract import extract_fields  # noqa: E402


def right(expected, value) -> bool:
    if value is None:
        return expected is None
    if isinstance(expected, (int, float)):
        try:
            return abs(float(value) - expected) < 1e-6
        except (TypeError, ValueError):
            return False
    return str(value).strip().lower() == str(expected).strip().lower()


def main():
    if llm.client() is None:
        sys.exit("no model configured (set DOCPARSE_LLM_BASE_URL + DOCPARSE_LLM_MODEL for LM Studio)")
    truth = json.loads((ROOT / "eval" / "text_truth.json").read_text())
    tally = {"correct": 0, "false alarm": 0, "silent error": 0, "wrong but flagged": 0}
    total, secs = 0, []
    for case in truth["cases"]:
        with pdfplumber.open(ROOT / case["doc"]) as pdf:
            text = extract_region(pdf.pages[case["page"] - 1], case["bbox"], "text", ocr=False)["text"]
        specs = [{k: v for k, v in f.items() if k != "expected"} for f in case["fields"]]
        t0 = time.perf_counter()
        r = extract_fields(text, specs)
        secs.append(time.perf_counter() - t0)
        got = {f["name"]: f for f in r["fields"]}
        for f in case["fields"]:
            g = got[f["name"]]
            ok, flagged = right(f["expected"], g["value"]), g["status"] == "unsupported"
            outcome = ("correct" if ok and not flagged else "false alarm" if ok else
                       "wrong but flagged" if flagged else "silent error")
            tally[outcome] += 1
            total += 1
            mark = "  " if outcome == "correct" else "!!"
            print(f"{mark} {Path(case['doc']).stem[:12]} p{case['page']:<3} {f['name']:<30} expected {f['expected']!s:<14} "
                  f"got {g['value']!s:<14} {g['status']:<11} {outcome if outcome != 'correct' else ''}")
    c = llm.client()
    print(f"\n{c.name} / {c.model}: {tally['correct']}/{total} correct | " +
          ", ".join(f"{k} {v}" for k, v in tally.items() if k != "correct") +
          f" | {sum(secs) / len(secs):.1f}s per passage (uncached)")


if __name__ == "__main__":
    main()
