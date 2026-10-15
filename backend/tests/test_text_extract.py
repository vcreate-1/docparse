"""Field extraction from text: JSON in, JSON out, table-shaped, verified.

A fake model stands in for the facility / OpenAI / LM Studio, so these tests
pin down the verification without any model installed."""

import pytest

from app.services import llm, text_extract

TEXT = ("U.S. railroads originated nearly 295,000 intermodal containers and trailers per week in "
        "September. U.S. GDP grew 2.2% in Q2 2026 over Q1 2026.")


class FakeModel:
    name, model = "fake", "fake-1"

    def __init__(self, answer):
        self.answer = answer
        self.calls = 0

    def complete_json(self, system, user, schema, **kw):
        self.calls += 1
        assert schema["additionalProperties"] is False  # strict schema is always sent
        return self.answer


@pytest.fixture
def model(monkeypatch, tmp_path):
    monkeypatch.setattr(llm.settings, "data_dir", tmp_path)  # isolated cache

    def use(answer):
        fake = FakeModel(answer)
        monkeypatch.setattr(llm, "client", lambda: fake)
        return fake

    return use


FIELDS = [{"name": "units_per_week", "type": "number"}, {"name": "gdp_growth", "type": "percent"},
          {"name": "coal_carloads", "type": "number"}]


def test_grounded_values_are_exact_and_table_shaped(model):
    model({"units_per_week": {"value": 295000, "evidence": "295,000 intermodal containers"},
           "gdp_growth": {"value": 2.2, "evidence": "GDP grew 2.2%"},
           "coal_carloads": {"value": None, "evidence": None}})
    r = text_extract.extract_fields(TEXT, FIELDS)
    assert r["confidence"] == "exact"  # "not stated" is a correct answer, not a downgrade
    assert r["columns"] == ["Field", "Value", "Status", "Evidence"]
    assert r["data"][0][:3] == ["units_per_week", "295000", "exact"]
    assert r["data"][2][2] == "not_found"


def test_invented_number_is_flagged(model):
    model({"units_per_week": {"value": 310000, "evidence": "295,000 intermodal containers"},
           "gdp_growth": {"value": 2.2, "evidence": "GDP grew 2.2%"},
           "coal_carloads": {"value": 1200, "evidence": "coal carloads were 1,200"}})
    r = text_extract.extract_fields(TEXT, FIELDS)
    assert r["confidence"] == "approximate"
    status = {row[0]: row[2] for row in r["data"]}
    assert status == {"units_per_week": "unsupported", "gdp_growth": "exact", "coal_carloads": "unsupported"}
    assert any("not in the text" in i for i in r["issues"])


def test_scaled_number_is_near_exact():
    assert text_extract.verify({"type": "number"}, 1_200_000, "revenue of $1.2 million", "revenue of $1.2 million")[0] == "near_exact"
    assert text_extract.verify({"type": "number"}, 12_00_00_000, "₹12 crore", "₹12 crore")[0] == "near_exact"


def test_rows_mode_returns_one_row_per_item(model):
    model({"rows": [
        {"indicator": {"value": "U.S. GDP", "evidence": "U.S. GDP"}, "value": {"value": 2.2, "evidence": "grew 2.2%"}},
        {"indicator": {"value": "Intermodal", "evidence": "intermodal"}, "value": {"value": 295000, "evidence": "295,000"}},
    ]})
    fields = [{"name": "indicator", "type": "text"}, {"name": "value", "type": "number"}]
    r = text_extract.extract_fields(TEXT, fields, rows=True)
    assert r["columns"] == ["indicator", "value"]
    assert r["data"] == [["U.S. GDP", "2.2"], ["Intermodal", "295000"]]


def test_same_request_is_cached(model):
    fake = model({"units_per_week": {"value": 295000, "evidence": "295,000"},
                  "gdp_growth": {"value": None, "evidence": None}, "coal_carloads": {"value": None, "evidence": None}})
    first = text_extract.extract_fields(TEXT, FIELDS)
    second = text_extract.extract_fields(TEXT, FIELDS)
    assert fake.calls == 1 and not first["cached"] and second["cached"]


def test_no_model_configured(monkeypatch):
    monkeypatch.setattr(llm, "client", lambda: None)
    r = text_extract.extract_fields(TEXT, FIELDS)
    assert r["method"] == "none" and "not configured" in r["note"] or "none is configured" in r["note"]
