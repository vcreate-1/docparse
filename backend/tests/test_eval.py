"""Regression gate on the hand-labelled eval set (eval/ground_truth.json).

Runs only where the source PDFs are present (they aren't committed). Structure
is always scored; Auto only where a layout model is installed. Thresholds are
the scores reached when this was written - raise them as the detectors improve,
never lower them to make a change pass.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GT = json.loads((ROOT / "eval" / "ground_truth.json").read_text())
HAVE_PDFS = all((ROOT / p["doc"]).exists() for p in GT["pages"])
sys.path.insert(0, str(ROOT / "eval"))




def _score(strategy):
    import evaluate

    return evaluate.score(strategy, GT, show_misses=False)


@pytest.mark.skipif(not HAVE_PDFS, reason="eval PDFs not present")
def test_structure_floor():
    t = _score("structure")
    assert t["table"][0] >= 4
    assert t["figure"][0] >= 13
    assert t["text"][0] / t["text"][1] >= 0.98


@pytest.mark.skipif(not HAVE_PDFS or importlib.util.find_spec("docling") is None, reason="no PDFs or layout model")
def test_auto_floor():
    t = _score("auto")
    assert t["table"][0] == t["table"][1]
    assert t["figure"][0] >= t["figure"][1] - 1
    assert t["bad"] <= 1
    assert t["text"][0] / t["text"][1] >= 0.99


SERIES = ROOT / "samples" / "series" / "manifest.json"


@pytest.mark.skipif(not SERIES.exists(), reason="run scripts/make_report_series.py first")
def test_anchoring_floor(capsys, monkeypatch):
    # Template from November (the hardest month for wording), PDF rules only so
    # this runs without the ML models. Every lookup must be correct.
    monkeypatch.setattr(sys, "argv", ["x", "--template", "11", "--detector", "structure"])
    import importlib

    import evaluate_anchors

    importlib.reload(evaluate_anchors)
    evaluate_anchors.main()
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert "SILENT" not in last and "flagged" not in last, last
