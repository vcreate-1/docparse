"""Reading picture charts: a vision model for structure, OCR as the check.

OCR alone reads characters exactly but can't tell which number belongs to
which series - on a stacked area chart the nearest text to "21" is an axis
year or the legend. A vision-language model understands the chart (series,
x values, which band a label sits in) but can misread or invent digits. So:

  1. The vision model returns JSON: series -> points {x, value, label_text,
     printed}. printed = the number is written on the chart for that point;
     otherwise it was estimated from the axis.
  2. Every value claimed as printed must appear in the OCR text of the chart.
       printed + found by OCR  -> exact
       printed, not in OCR     -> unsupported (flagged)
       estimated               -> approximate (labelled, never passed as exact)
  3. Output is a table: Series | X | Value | Printed as | Source | Note.
"""

import re

from . import llm
from .ocr import OcrResult

PROMPT_VERSION = "chart-v1"

SYSTEM = """You read charts from business reports and return their data.
- Return every data series. For each x value (category, year, period) give that series' value.
- printed=true only if that number is written on the chart for that series and x; copy it
  exactly as printed into label_text (e.g. "109 GW", "15-20%", "5,904").
- Otherwise estimate the value from the axis, set printed=false and label_text=null.
- Stacked charts: give each series' own value, not the cumulative height.
- Pie charts: one series named after the chart; x = slice name.
- Bar charts with one value per bar: one series; x = bar label.
- Put annotations that are not data points (CAGR, notes) in notes, tied to series/x when clear.
- If the image is not a chart with data, return an empty series list.
/no_think"""

POINT = {
    "type": "object", "additionalProperties": False,
    "required": ["x", "value", "label_text", "printed"],
    "properties": {"x": {"type": "string"}, "value": {"type": ["number", "null"]},
                   "label_text": {"type": ["string", "null"]}, "printed": {"type": "boolean"}},
}
SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["title", "unit", "series", "notes"],
    "properties": {
        "title": {"type": ["string", "null"]},
        "unit": {"type": ["string", "null"]},
        "series": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["name", "points"],
            "properties": {"name": {"type": "string"}, "points": {"type": "array", "items": POINT}}}},
        "notes": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["series", "x", "text"],
            "properties": {"series": {"type": ["string", "null"]}, "x": {"type": ["string", "null"]},
                           "text": {"type": "string"}}}},
    },
}


def _norm(s: str) -> str:
    return re.sub(r"[\s,]", "", (s or "").lower()).replace("−", "-").replace("–", "-")


def read_chart(png: bytes, ocr: OcrResult) -> dict | None:
    """Chart as a verified table, or None when no vision model is available."""
    c = llm.vision_client()
    if c is None:
        return None
    raw, cached = llm.cached_call(c, PROMPT_VERSION, SYSTEM, "Read this chart.", SCHEMA, images=[png])
    ocr_text = " ".join(line.text for line in ocr.lines)
    ocr_norm = _norm(ocr_text)
    # Unsigned digits: OCR reads "-155" and "20-25%"; the model's label may carry
    # the sign elsewhere or split the range - compare the digit runs themselves.
    ocr_numbers = {_norm(n) for n in re.findall(r"\d[\d,]*\.?\d*", ocr_text)}

    notes = {}
    for n in raw.get("notes") or []:
        notes.setdefault(((n.get("series") or "").lower(), (n.get("x") or "").lower()), []).append(n["text"])

    rows, statuses = [], []
    for s in raw.get("series") or []:
        for p in s.get("points") or []:
            value, label = p.get("value"), p.get("label_text")
            if p.get("printed"):
                digits = re.findall(r"\d[\d,]*\.?\d*", label or "")
                in_ocr = bool(label) and (_norm(label) in ocr_norm or (digits and all(_norm(d) in ocr_numbers for d in digits)))
                status = "exact" if in_ocr else "unsupported"
                source = "printed" if in_ocr else "printed? (not found by OCR)"
            else:
                status, source = "approximate", "estimated from axis"
            statuses.append(status)
            note = "; ".join(notes.get((s["name"].lower(), str(p["x"]).lower()), []))
            rows.append([s["name"], str(p["x"]), "" if value is None else _num(value), label or "", source, note])

    if not rows:
        return {"type": "chart", "method": f"vision:{c.name}", "confidence": "approximate",
                "columns": [], "data": [], "issues": ["the vision model found no chart data"],
                "note": "No chart data found in this picture.", "model": c.model, "cached": cached}
    printed_ok = statuses.count("exact")
    estimated = statuses.count("approximate")
    flagged = statuses.count("unsupported")
    issues = []
    if flagged:
        issues.append(f"{flagged} value(s) the model says are printed were not found by OCR")
    confidence = "approximate" if (flagged or estimated) else "exact"
    parts = [f"{printed_ok} printed value(s) confirmed by OCR"]
    if estimated:
        parts.append(f"{estimated} estimated from the axis (approximate)")
    if flagged:
        parts.append(f"{flagged} not confirmed - check against the chart")
    return {
        "type": "chart", "method": f"vision:{c.name}", "confidence": confidence,
        "title": raw.get("title"), "unit": raw.get("unit"),
        "columns": ["Series", "X", "Value", "Printed as", "Source", "Note"], "data": rows,
        "row_status": statuses, "issues": issues, "model": c.model, "prompt_version": PROMPT_VERSION,
        "cached": cached, "note": "Chart read by a vision model, checked against OCR: " + ", ".join(parts) + ".",
    }


def _num(v) -> str:
    return str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)
