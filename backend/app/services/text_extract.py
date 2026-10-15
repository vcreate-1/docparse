"""Structured fields from prose, with every value checked against the text.

The text layer gives exact characters; turning a sentence into data
("nearly 295,000 per week in September" -> intermodal_weekly = 295000) needs a
language model. Models can produce values the text doesn't contain, so:

  1. The model sees only the selected text, at temperature 0, and must return
     each field as {value, evidence}: evidence is the exact sentence fragment
     the value comes from (null when the text doesn't state it).
  2. Code verifies every field:
       exact        evidence is verbatim in the text and contains the value as written
       near_exact   evidence is verbatim; the value is a reformatting of it
                    ("2.6%" -> 2.6, "1.2 million" -> 1200000)
       unsupported  evidence not in the text, or the value not in the evidence
       not_found    the model says the text doesn't state it
     Unsupported values are reported, never passed silently.
"""

import re

from . import llm

PROMPT_VERSION = "fields-v3"

SYSTEM = """You extract data from a passage of a business document.
Rules:
- Use only the passage. Do not use outside knowledge, do not infer, do not calculate.
- For every field return {"value": ..., "evidence": ...}.
- "evidence" must be copied character-for-character from the passage: the shortest
  fragment that states the value. Do not paraphrase it.
- If the passage does not state a field, return {"value": null, "evidence": null}.
- Numbers: return the number as stated (2.6 for "2.6%"), without thousands separators.
- Changes: a decline, decrease, drop or fall is a negative number ("declined 8%" -> -8).
- When asked for rows, return one row per item the passage lists, in passage order.
/no_think"""

JSON_TYPES = {
    "number": {"type": ["number", "null"]},
    "percent": {"type": ["number", "null"]},
    "text": {"type": ["string", "null"]},
    "date": {"type": ["string", "null"]},
    "boolean": {"type": ["boolean", "null"]},
    "list": {"type": ["array", "null"], "items": {"type": "string"}},
}
SCALE = {"thousand": 1e3, "k": 1e3, "million": 1e6, "mn": 1e6, "m": 1e6, "billion": 1e9, "bn": 1e9,
         "trillion": 1e12, "lakh": 1e5, "lakhs": 1e5, "crore": 1e7, "crores": 1e7}


def schema_for(fields: list[dict], rows: bool = False) -> dict:
    """Strict JSON schema: every field is an object {value, evidence}; with
    rows=True, an array of such records (one per item the passage lists)."""
    record = _record_schema(fields)
    if not rows:
        return record
    return {"type": "object", "additionalProperties": False, "required": ["rows"],
            "properties": {"rows": {"type": "array", "items": record}}}


def _record_schema(fields: list[dict]) -> dict:
    props = {}
    for f in fields:
        props[f["name"]] = {
            "type": "object",
            "additionalProperties": False,
            "required": ["value", "evidence"],
            "properties": {"value": JSON_TYPES.get(f.get("type", "text"), JSON_TYPES["text"]),
                           "evidence": {"type": ["string", "null"]}},
        }
    return {"type": "object", "additionalProperties": False, "required": [f["name"] for f in fields],
            "properties": props}


def prompt_for(text: str, fields: list[dict], instruction: str | None, context: str | None,
               rows: bool = False) -> str:
    lines = ["Return one row per item the passage lists, with these fields:" if rows else "Fields to extract:"]
    for f in fields:
        desc = f" - {f['description']}" if f.get("description") else ""
        lines.append(f"- {f['name']} ({f.get('type', 'text')}){desc}")
    if instruction:
        lines += ["", f"Instruction: {instruction}"]
    if context:
        lines += ["", f"Section heading (for orientation only, not evidence): {context}"]
    lines += ["", "Passage:", "<<<", text, ">>>"]
    return "\n".join(lines)


# ---------------------------------------------------------------- verification

def _norm(s: str) -> str:
    s = s.lower().replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = s.replace("–", "-").replace("—", "-").replace("−", "-")
    return " ".join(s.split())


NUM = re.compile(r"(?<![\w.])[-(]?\$?\s?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\)?\s*(%|percent)?\s*(thousand|million|billion|trillion|lakhs?|crores?|mn|bn|k|m)?\b", re.I)


WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty".split())}
WORDS.update({"thirty": 30, "forty": 40, "fifty": 50, "hundred": 100})
ORDINALS = {w: i + 1 for i, w in enumerate(
    "first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth".split())}
DECLINE = re.compile(r"\b(declin\w*|fell|fall\w*|drop\w*|down|decreas\w*|lower\w*|loss\w*|negative)\b", re.I)


def numbers_in(text: str) -> list[tuple[float, float]]:
    """(as written, with scale word applied) for each number in the text,
    including number words ("one percent") and ordinals ("ninth month")."""
    out = []
    for word in re.findall(r"[a-z]+", text.lower()):
        if word in WORDS:
            out.append((float(WORDS[word]), float(WORDS[word])))
        elif word in ORDINALS:
            out.append((float(ORDINALS[word]), float(ORDINALS[word])))
    for m in NUM.finditer(text):
        base = float((m.group(1) + (m.group(2) or "")).replace(",", ""))
        if m.group(0).strip().startswith(("-", "(")):
            base = -base
        scale = SCALE.get((m.group(4) or "").lower(), 1)
        out.append((base, base * scale))
    return out


def verify(field: dict, value, evidence, text: str) -> tuple[str, str | None]:
    """(status, issue) for one extracted field."""
    if value is None:
        return "not_found", None
    if not evidence:
        return "unsupported", "no evidence quoted"
    if _norm(evidence) not in _norm(text):
        return "unsupported", "evidence is not in the text"
    kind = field.get("type", "text")
    if kind in ("number", "percent"):
        try:
            v = float(value)
        except (TypeError, ValueError):
            return "unsupported", f"not a number: {value!r}"
        found = numbers_in(evidence)
        if any(abs(v - written) < 1e-9 for written, _ in found) and not _is_word_number(v, evidence):
            return "exact", None
        if any(abs(v - scaled) < 1e-6 * max(1, abs(scaled)) for _, scaled in found):
            return "near_exact", None
        # "dropped by nearly 8%" stored as -8: the sign comes from the wording.
        if v < 0 and DECLINE.search(evidence) and any(abs(-v - w) < 1e-9 for w, _ in found):
            return "near_exact", None
        return "unsupported", f"{value} does not appear in the evidence"
    if kind == "list":
        items = value if isinstance(value, list) else [value]
        missing = [i for i in items if _norm(str(i)) not in _norm(text)]
        return ("exact", None) if not missing else ("unsupported", f"not in the text: {missing[:3]}")
    if kind == "boolean":
        return "near_exact", None  # a yes/no is a judgement on the quoted evidence
    if _norm(str(value)) in _norm(evidence):
        return "exact", None
    return "near_exact", "value paraphrases the evidence"


def _is_word_number(v: float, evidence: str) -> bool:
    """True when the value only appears spelled out ("one percent"): a reading, not a copy."""
    digits = [w for w, _ in numbers_in(re.sub(r"[A-Za-z]+", " ", evidence))]
    return not any(abs(v - d) < 1e-9 for d in digits)


ORDER = {"exact": 0, "near_exact": 1, "not_found": 0, "unsupported": 2}  # "not stated" is a correct answer
CONFIDENCE = {0: "exact", 1: "near_exact", 2: "approximate"}


def extract_fields(text: str, fields: list[dict], instruction: str | None = None,
                   context: str | None = None, *, rows: bool = False, ref_id: str | None = None,
                   pages: list[int] | None = None) -> dict:
    """JSON in (passage + field schema), JSON out, always shaped as a table:
    columns + data (+ per-cell status/evidence), the same shape as table
    extraction, so the UI, templates and the batch loader treat them alike.

      rows=False  one record  -> columns Field | Value | Status | Evidence
      rows=True   many records -> one column per field, one row per item
    """
    c = llm.client()
    base = {"type": "fields", "text": text, "title": None}
    if c is None:
        return {**base, "method": "none", "confidence": "approximate", "columns": [], "data": [], "fields": [],
                "note": "Field extraction needs the RAG facility, an OpenAI API key or a local model "
                        "(LM Studio), and none is configured. The text itself is exact."}
    if not text.strip():
        return {**base, "method": f"llm:{c.name}", "confidence": "approximate", "columns": [], "data": [],
                "fields": [], "note": "No text in this region to extract from."}
    raw, cached = llm.cached_call(c, PROMPT_VERSION, SYSTEM, prompt_for(text, fields, instruction, context, rows),
                                  schema_for(fields, rows), ref_id=ref_id, pages=pages)
    records = raw.get("rows", []) if rows else [raw]
    checked, worst = [], 0
    for rec in records:
        row = []
        for f in fields:
            item = (rec or {}).get(f["name"]) or {}
            value, evidence = item.get("value"), item.get("evidence")
            status, issue = verify(f, value, evidence, text)
            worst = max(worst, ORDER[status])
            row.append({"name": f["name"], "type": f.get("type", "text"), "value": value,
                        "evidence": evidence, "status": status, "issue": issue})
        checked.append(row)
    flagged = [f"{x['name']}: {x['issue']}" for row in checked for x in row if x["status"] == "unsupported"]
    if rows:
        columns = [f["name"] for f in fields]
        data = [[_cell(x) for x in row] for row in checked]
    else:
        columns = ["Field", "Value", "Status", "Evidence"]
        data = [[x["name"], _cell(x), x["status"], x["evidence"] or ""] for x in (checked[0] if checked else [])]
    return {
        **base, "method": f"llm:{c.name}", "confidence": CONFIDENCE[worst],
        "columns": columns, "data": data, "fields": checked if rows else (checked[0] if checked else []),
        "model": c.model, "prompt_version": PROMPT_VERSION, "cached": cached, "issues": flagged,
        "note": ("Every value was found word-for-word in the text." if worst == 0 else
                 "Values reformatted from the quoted text; check before use." if worst == 1 else
                 "Some values could not be found in the text - not trusted: " + "; ".join(flagged)),
    }


def _cell(x: dict) -> str:
    v = x["value"]
    if v is None:
        return ""
    if isinstance(v, list):
        return "; ".join(map(str, v))
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)
