"""Turn raw extracted table rows into title + flattened columns + clean data.

Real reports have stacked headers:

    ELECTRIC VEHICLE (BRAND)                       <- title row (one cell)
           | Q3 Sales          | YTD Sales  ...    <- group row (spanned cells)
           | 2026 | 2025 | YOY | 2026 | ...        <- leaf row
    Audi   | 1,039| 18,071 | -94.3% | ...          <- first data row

This becomes title="ELECTRIC VEHICLE (BRAND)" and columns like
"Q3 Sales 2026", which are stable names a template can match next month.
"""

import re

NUMERIC = re.compile(r"^[\s(+\-−]*[$€£₹]?\s*[\d.,\s]+%?\)?$|^[-—–]$")
SPLIT_NUMBER = re.compile(r"(?<=\d) (?=[\d,])")
YEAR = re.compile(r"^(19|20)\d\d$")


def _is_numeric(cell: str) -> bool:
    return bool(cell) and bool(NUMERIC.match(cell))


def clean_cell(cell: str | None) -> str:
    text = " ".join((cell or "").split())
    # Small or tightly kerned fonts make "726,100" come out as "7 26,100".
    if _is_numeric(text):
        text = SPLIT_NUMBER.sub("", text)
    return text


def structure(rows: list[list[str | None]]) -> dict:
    rows = [[clean_cell(c) for c in row] for row in rows]
    rows = [r for r in rows if any(r)]
    if not rows:
        return {"title": None, "columns": [], "data": []}

    title = None
    filled = [c for c in rows[0] if c]
    if len(filled) == 1 and len(rows[0]) > 2 and not _is_numeric(filled[0]):
        title = filled[0]
        rows = rows[1:]
    # Further title lines: label-only rows before anything with values
    # ("# of AAR Traffic Categories*" / "With Year-Over-Year Gains" / ...).
    while title and len(rows) > 1 and rows[0][0] and not any(rows[0][1:]) and not rows[0][0].endswith(":") \
            and len(rows[0][0]) < 60:
        title = f"{title} {rows[0][0]}"
        rows = rows[1:]

    # Header rows = leading rows until one is mostly numbers.
    header_rows: list[list[str]] = []
    for row in rows[:8]:
        cells = [c for c in row[1:] if c]
        if row[0] and not cells and header_rows:
            break  # "Net sales:" - a section label opens the body, it isn't a header
        if cells and sum(_is_numeric(c) for c in cells) / len(cells) >= 0.5:
            # A leaf row of years ("2026 | 2025") is numeric but still a header,
            # recognisable because the label column is empty.
            years = all(YEAR.match(c) for c in cells)  # "Month | 2024 | 2025": a header, label or not
            if (row[0] or not header_rows) and not years:
                break
        header_rows.append(row)
    data = rows[len(header_rows) :]
    # A header row with text only in the label column is the tail of the
    # sentence introducing the table ("...as follows (in millions):") or a
    # caption - a title at most, never part of every column's name.
    captions = [hr for hr in header_rows if hr[0] and not any(hr[1:])]
    header_rows = [hr for hr in header_rows if hr not in captions]
    if captions and title is None:
        title = captions[-1][0]

    width = max(len(r) for r in rows)
    columns = []
    for col in range(width):
        parts = []
        for hr in header_rows:
            # Forward-fill spanned group cells: "Q3 Sales" covers the next empties.
            value = next((hr[c] for c in range(col, -1, -1) if c < len(hr) and hr[c]), "")
            if col < len(hr) and not hr[col] and hr is header_rows[-1]:
                value = ""  # leaf row never inherits
            if value and value not in parts:
                parts.append(value)
        columns.append(" ".join(parts) or ("Label" if col == 0 else f"Column {col + 1}"))

    return {"title": title, "columns": columns, "data": data}


# ---------------------------------------------------------------------------
# Borderless tables: columns from alignment instead of ruling lines.
#
# Financial statements converted from HTML (SEC filings) and most Word tables
# have no grid. Their figures are right-aligned, so the right edges of numeric
# tokens cluster into columns; the label column is everything left of them;
# header phrases sit above the column(s) they describe.

NUMBER_TOKEN = re.compile(r"^\(?[$€£₹]?\(?-?[\d.,]+%?\)?$|^[—–-]$")


def align_table(words: list[dict]) -> list[list[str]]:
    """pdfplumber words of one table region -> rows of cells (header rows first)."""
    words = [w for w in words if w["text"] not in ("$", "€", "£", "₹")]
    if not words:
        return []
    lines = _lines(words)
    left = min(w["x0"] for w in words)
    # Value columns come from real values only - not header dates ("27,",
    # "2026") and not numbers inside labels near the left edge.
    values = [w for w in words if _value_token(w["text"]) and w["x0"] - left >= 25]
    if not values:
        return [[" ".join(w["text"] for w in line)] for line in lines]

    # Column anchors: clusters of right edges (figures are right-aligned).
    anchors: list[list[float]] = []
    for x1 in sorted(w["x1"] for w in values):
        if anchors and x1 - anchors[-1][-1] <= 8:
            anchors[-1].append(x1)
        else:
            anchors.append([x1])
    anchors = [a for a in anchors if len(a) >= 2] or anchors  # one-off strays aren't columns
    cols = [(min(w["x0"] for w in values if abs(w["x1"] - max(a)) <= 8), max(a)) for a in anchors]
    label_edge = min(c[0] for c in cols) - 4

    rows: list[list[str]] = []
    for line in lines:
        cells = [""] * (len(cols) + 1)
        in_cols = [w for w in line if _is_value(w, cols)]
        # Headings start well right of the label column (dates and years are
        # centred over their column, so they can poke left of the figures).
        header = not in_cols and line[0]["x0"] - left > 0.6 * (label_edge - left)
        year_row = not in_cols and sum(1 for w in line if YEAR.match(w["text"])) >= 2
        if year_row:
            cells[0] = " ".join(w["text"] for w in line if not YEAR.match(w["text"]) and w["x1"] <= label_edge)
            for w in line:
                if YEAR.match(w["text"]):
                    k = min(range(len(cols)), key=lambda k: abs((cols[k][0] + cols[k][1]) / 2 - (w["x0"] + w["x1"]) / 2))
                    cells[k + 1] = w["text"]
            rows.append(cells)
            continue
        if header:
            # Column headings: each phrase goes over the leftmost column it spans.
            for x0, x1, text in _phrases(line):
                overlapped = [k for k, (c0, c1) in enumerate(cols) if x0 < c1 + 4 and x1 > c0 - 4]
                k = overlapped[0] if overlapped else min(range(len(cols)), key=lambda k: abs(cols[k][1] - x1))
                cells[k + 1] = (cells[k + 1] + " " + text).strip()
        else:
            # A data row (possibly indented) or a section label ("Net sales:").
            cells[0] = " ".join(w["text"] for w in line if w not in in_cols)
            for w in in_cols:
                i = min(range(len(cols)), key=lambda k: abs(cols[k][1] - w["x1"]))
                cells[i + 1] = (cells[i + 1] + " " + w["text"]).strip()
        rows.append(cells)
    return _merge_wrapped(rows)


def _value_token(text: str) -> bool:
    return bool(NUMBER_TOKEN.match(text)) and not text.endswith(",") and not YEAR.match(text)


def _is_value(w: dict, cols) -> bool:
    return _value_token(w["text"]) and w["x1"] >= cols[0][0] - 2


def _lines(words: list[dict]) -> list[list[dict]]:
    out: list[list[dict]] = []
    for w in sorted(words, key=lambda w: (w["top"], w["x0"])):
        if out and abs(w["top"] - out[-1][0]["top"]) <= 3:
            out[-1].append(w)
        else:
            out.append([w])
    return [sorted(line, key=lambda w: w["x0"]) for line in out]


def _phrases(words: list[dict]) -> list[tuple[float, float, str]]:
    out: list[list] = []
    for w in sorted(words, key=lambda w: w["x0"]):
        if out and w["x0"] - out[-1][1] <= 6:
            out[-1][1] = w["x1"]
            out[-1][2] += " " + w["text"]
        else:
            out.append([w["x0"], w["x1"], w["text"]])
    return [tuple(p) for p in out]


def _merge_wrapped(rows: list[list[str]]) -> list[list[str]]:
    """A label that wraps puts its numbers on the second line: join the two."""
    out: list[list[str]] = []
    for row in rows:
        prev = out[-1] if out else None
        if (
            prev is not None
            and prev[0] and not any(prev[1:]) and not prev[0].endswith(":")
            and row[0][:1].islower() and any(row[1:])
        ):
            prev[0] = f"{prev[0]} {row[0]}"
            prev[1:] = row[1:]
        else:
            out.append(row)
    return out


def table_quality(t: dict) -> float:
    """How clean an extracted table looks: share of filled value cells, with
    penalties for the tell-tale signs of a misread grid."""
    data = t.get("data") or []
    if not data:
        return 0.0
    values = [c for row in data for c in row[1:]]
    if not values:
        return 0.0
    filled = sum(1 for c in values if c) / len(values)
    junk = sum(1 for c in values if c in ("$", "%", "(", ")")) / len(values)
    clipped = sum(1 for row in data if row[0][:1].islower()) / len(data)  # "oducts" = a cut label
    named = sum(1 for c in t.get("columns", [])[1:] if not c.startswith("Column ")) / max(1, len(t.get("columns", [])) - 1)
    return filled - junk - 0.5 * clipped + 0.2 * named
