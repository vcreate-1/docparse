"""Picture charts: the vision model's JSON is checked against OCR text."""

from app.services import chart_read, llm
from app.services.ocr import OcrLine, OcrResult, chart_layout_issues


class FakeVision:
    name, model = "fake", "fake-vlm"

    def __init__(self, answer):
        self.answer = answer

    def supports_vision(self):
        return True

    def complete_json(self, system, user, schema, **kw):
        assert kw.get("images"), "the chart image must be sent"
        return self.answer


def _ocr(*texts):
    return OcrResult("test", [OcrLine(t, 0.98, (0, i * 20, 100, i * 20 + 15)) for i, t in enumerate(texts)])


def _pt(x, value, label=None, printed=True):
    return {"x": x, "value": value, "label_text": label, "printed": printed}


def test_printed_values_confirmed_by_ocr_and_estimates_labelled(monkeypatch, tmp_path):
    monkeypatch.setattr(llm.settings, "data_dir", tmp_path)
    monkeypatch.setattr(llm, "vision_client", lambda: FakeVision({
        "title": "Global supply forecast by region (GW)", "unit": "GW",
        "series": [{"name": "AMER", "points": [_pt("2020", 24, printed=False), _pt("2025", 49, "49"), _pt("2030", 109, "109 GW")]},
                   {"name": "EMEA", "points": [_pt("2030", -7274, "-7,274")]}],
        "notes": [{"series": "AMER", "x": "2030", "text": "17% CAGR"}],
    }))
    out = chart_read.read_chart(b"png", _ocr("49", "AMER", "109 GW", "17% CAGR", "-7,274"))
    assert out["columns"] == ["Series", "X", "Value", "Printed as", "Source", "Note"]
    assert out["row_status"] == ["approximate", "exact", "exact", "exact"]  # signed number found too
    assert out["data"][2][5] == "17% CAGR"
    assert out["confidence"] == "approximate"  # an estimate is in the table


def test_invented_printed_value_is_flagged(monkeypatch, tmp_path):
    monkeypatch.setattr(llm.settings, "data_dir", tmp_path)
    monkeypatch.setattr(llm, "vision_client", lambda: FakeVision({
        "title": None, "unit": None, "notes": [],
        "series": [{"name": "S", "points": [_pt("2025", 51, "51")]}],  # OCR saw 49, not 51
    }))
    out = chart_read.read_chart(b"png", _ocr("49", "2025"))
    assert out["row_status"] == ["unsupported"] and out["issues"]


def test_no_vision_model_returns_none(monkeypatch):
    monkeypatch.setattr(llm, "vision_client", lambda: None)
    assert chart_read.read_chart(b"png", _ocr("49")) is None


def test_ocr_pairing_gate_catches_stacked_chart_garbage():
    # The exact output that slipped through on Sector p10 before the gate.
    garbage = [["AMER □APAC", "21", "32", "49", "2025"], ["AMER GW CAGR", "109", "17%", "2030"]]
    assert chart_layout_issues(garbage)
    assert chart_layout_issues([["Grain", "5,867", "5.5%"], ["Coal", "-7,274", "-2.4%"]]) == []
