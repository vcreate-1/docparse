"""OCR shaping and quality checks. Uses synthetic OCR output, so no OCR engine
is needed; the real-engine floor runs only where PaddleOCR and the PDFs exist."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from app.services.ocr import OcrLine, OcrResult, assess, chart_rows, paired_values, reconcile_totals


def _result(items, conf=0.97):
    return OcrResult("test", [OcrLine(t, conf, b) for t, b in items])


def test_values_pair_with_labels_far_away():
    # Power report p10: label left, value at the end of the bar; a two-line label
    # whose value sits between its lines; a value above a bar with the label below.
    r = _result([
        ("73%", (181, 9, 206, 23)),
        ("Mobile", (289, 19, 322, 32)), ("33%", (426, 24, 448, 37)), ("turbines", (289, 30, 329, 43)),
        ("Fuel cells", (289, 150, 332, 163)), ("47%", (461, 150, 482, 163)),
        ("Actively evaluating", (156, 174, 232, 186)),
    ])
    rows = {r[0]: r[1:] for r in paired_values(r)}
    assert rows["Mobile turbines"] == ["33%"]
    assert rows["Fuel cells"] == ["47%"]
    assert rows["Actively evaluating"] == ["73%"]


def test_side_by_side_lists_do_not_merge():
    r = _result([
        ("Texas", (0, 0, 40, 12)), ("142%", (100, 0, 130, 12)), ("California", (200, 0, 260, 12)), ("-50%", (300, 0, 330, 12)),
        ("Georgia", (0, 18, 50, 30)), ("75%", (100, 18, 125, 30)), ("Iowa", (200, 18, 230, 30)), ("-60%", (300, 18, 330, 30)),
    ])
    rows = {r[0]: r[1:] for r in paired_values(r)}
    assert rows == {"Texas": ["142%"], "California": ["-50%"], "Georgia": ["75%"], "Iowa": ["-60%"]}


def test_inline_labels_and_titles():
    r = _result([
        ("Carloads rose in September, with 17 of the 20 categories up.", (0, 0, 400, 12)),
        ("Grain: 5,867, 5.5%", (0, 20, 120, 32)),
        ("Coal: -7,274, -2.4%", (0, 36, 120, 48)),
    ])
    assert chart_rows(r) == [["Grain", "5,867", "5.5%"], ["Coal", "-7,274", "-2.4%"]]


def test_totals_must_reconcile():
    rows = [["Label", "Q3"], ["A", "1,000"], ["B", "2,500"], ["Total", "3,500"]]
    assert reconcile_totals(rows) == []
    rows[3][1] = "3,900"  # an OCR misread somewhere
    assert reconcile_totals(rows) and "3,500" in reconcile_totals(rows)[0]


def test_low_confidence_is_an_issue():
    assert assess(_result([("12,3B5", (0, 0, 10, 10))], conf=0.55))
    assert assess(_result([])) == ["no text read"]
    assert assess(_result([("12,385", (0, 0, 10, 10))])) == []


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(
    importlib.util.find_spec("paddleocr") is None
    or not all((ROOT / r["doc"]).exists() for r in json.loads((ROOT / "eval" / "ocr_truth.json").read_text())["regions"]),
    reason="PaddleOCR or eval PDFs not present",
)
def test_ocr_floor(capsys):
    sys.path.insert(0, str(ROOT / "eval"))
    import evaluate_ocr

    evaluate_ocr.main()
    out = capsys.readouterr().out
    found, total = map(int, out.strip().splitlines()[-1].split()[0].split("/"))
    assert found >= 91, out
