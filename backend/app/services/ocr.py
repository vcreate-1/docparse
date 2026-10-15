"""OCR for regions with no text layer: pictures of tables, raster charts, scans.

Engine order (decided with the user): PaddleOCR first; if its result has
issues and Snowflake Cortex is configured, retry with Cortex and keep the
better reading. Without Cortex, the issues are reported so the region goes
to human review instead of silently passing.

"Issues" are measurable, not a feeling:
  - nothing read
  - low confidence (mean < 0.85, or > 15% of lines below 0.6)
  - for tables: rows that don't line up into columns
  - totals that don't reconcile (a "Total" row vs the sum of the rows above)
"""

import importlib.util
import io
import os
import re
import threading
from dataclasses import dataclass, field

from PIL import Image

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")  # skips a slow network probe

MIN_MEAN_CONF = 0.85
LOW_LINE_CONF = 0.6
MAX_LOW_LINES = 0.15


@dataclass
class OcrLine:
    text: str
    conf: float
    box: tuple[float, float, float, float]  # x0, top, x1, bottom in image pixels
    words: list[tuple[str, tuple[float, float, float, float]]] = field(default_factory=list)


@dataclass
class OcrResult:
    engine: str
    lines: list[OcrLine]
    issues: list[str] = field(default_factory=list)

    @property
    def mean_conf(self) -> float:
        return sum(line.conf for line in self.lines) / len(self.lines) if self.lines else 0.0


# ---------------------------------------------------------------- engines

class PaddleEngine:
    name = "paddleocr"
    _model = None
    _lock = threading.Lock()

    @staticmethod
    def available() -> bool:
        return importlib.util.find_spec("paddleocr") is not None

    def read(self, image: Image.Image) -> OcrResult:
        import numpy as np

        with self._lock:
            if PaddleEngine._model is None:
                from paddleocr import PaddleOCR

                # Mobile models: ~3-5 s per region on CPU and, on the eval
                # set, every printed value read correctly.
                PaddleEngine._model = PaddleOCR(
                    text_detection_model_name="PP-OCRv5_mobile_det",
                    text_recognition_model_name="PP-OCRv5_mobile_rec",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                )
            res = PaddleEngine._model.predict(
                np.asarray(image.convert("RGB"))[:, :, ::-1], return_word_box=True
            )[0]
        word_texts = res.get("text_word") or [None] * len(res["rec_texts"])
        word_boxes = res.get("text_word_boxes") or [None] * len(res["rec_texts"])
        lines = []
        for text, conf, box, wt, wb in zip(
            res["rec_texts"], res["rec_scores"], res["rec_boxes"].tolist(), word_texts, word_boxes
        ):
            if not text.strip():
                continue
            words = []
            if wt is not None and wb is not None and len(wt) == len(wb):
                words = [(t, tuple(float(v) for v in b)) for t, b in zip(wt, wb) if t.strip()]
            lines.append(OcrLine(text, float(conf), tuple(float(v) for v in box), words))
        return OcrResult(self.name, lines)


class CortexEngine:
    """Snowflake Cortex AI_PARSE_DOCUMENT (OCR mode) on a cropped region.

    UNTESTED: written against Snowflake's documented API; no credentials were
    available when this was built. Enabled only when DOCPARSE_CORTEX_STAGE and
    SNOWFLAKE_* connection variables are set. Cortex returns text without
    positions, so lines get synthetic boxes in reading order - fine for text
    and labelled charts; table columns are rebuilt from the text.
    """

    name = "cortex"

    @staticmethod
    def available() -> bool:
        return bool(os.environ.get("DOCPARSE_CORTEX_STAGE")) and importlib.util.find_spec(
            "snowflake.connector"
        ) is not None

    def read(self, image: Image.Image) -> OcrResult:
        import json
        import tempfile
        import uuid

        import snowflake.connector

        stage = os.environ["DOCPARSE_CORTEX_STAGE"]  # e.g. @DOCPARSE.PUBLIC.OCR_STAGE
        conn = snowflake.connector.connect(
            account=os.environ["SNOWFLAKE_ACCOUNT"],
            user=os.environ["SNOWFLAKE_USER"],
            password=os.environ.get("SNOWFLAKE_PASSWORD"),
            authenticator=os.environ.get("SNOWFLAKE_AUTHENTICATOR", "snowflake"),
            warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE"),
            role=os.environ.get("SNOWFLAKE_ROLE"),
        )
        name = f"ocr-{uuid.uuid4().hex}.png"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, name)
                image.save(path)
                cur = conn.cursor()
                cur.execute(f"PUT 'file://{path}' {stage} AUTO_COMPRESS=FALSE OVERWRITE=TRUE")
                cur.execute(
                    f"SELECT AI_PARSE_DOCUMENT(TO_FILE('{stage}', '{name}'), {{'mode': 'OCR'}})"
                )
                content = json.loads(cur.fetchone()[0]).get("content", "")
                cur.execute(f"REMOVE {stage}/{name}")
        finally:
            conn.close()
        lines = [
            OcrLine(text.strip(), 0.95, (0.0, i * 20.0, 1000.0, i * 20.0 + 16))  # no positions from Cortex
            for i, text in enumerate(content.splitlines())
            if text.strip()
        ]
        return OcrResult(self.name, lines)


ENGINES = [PaddleEngine, CortexEngine]


def available_engines() -> list[str]:
    return [e.name for e in ENGINES if e.available()]


# ---------------------------------------------------------------- quality

NUMBER = re.compile(r"\(?-?[$€£₹]?\d[\d,]*\.?\d*(-\d+(\.\d+)?)?%?\)?")


def assess(result: OcrResult, rows: list[list[str]] | None = None) -> list[str]:
    """Reasons this OCR reading shouldn't be trusted as-is (empty = fine)."""
    issues = []
    if not result.lines:
        return ["no text read"]
    if result.mean_conf < MIN_MEAN_CONF:
        issues.append(f"low confidence ({result.mean_conf:.0%} mean)")
    low = sum(1 for line in result.lines if line.conf < LOW_LINE_CONF) / len(result.lines)
    if low > MAX_LOW_LINES:
        issues.append(f"{low:.0%} of lines below {LOW_LINE_CONF:.0%} confidence")
    if rows:
        issues += _table_issues(rows)
    return issues


def _table_issues(rows: list[list[str]]) -> list[str]:
    issues = []
    data = [r for r in rows if any(r[1:])]
    if not data:
        return ["no values in table columns"]
    widths = [sum(1 for c in r[1:] if c) for r in data]
    full = max(widths)
    if full > 1 and sum(1 for w in widths if w < full / 2) > len(data) / 3:
        issues.append("many rows don't fill the columns")
    issues += reconcile_totals(rows)
    return issues


def to_number(cell: str) -> float | None:
    c = cell.replace(",", "").replace("$", "").replace("%", "").strip()
    neg = c.startswith("(") and c.endswith(")") or c.startswith("-") or c.startswith("−")
    c = c.strip("()-−+ ")
    try:
        v = float(c)
    except ValueError:
        return None
    return -v if neg else v


def reconcile_totals(rows: list[list[str]]) -> list[str]:
    """A row labelled "Total" must equal the sum of the rows above it, per column.

    Skipped for share columns (100% totals of rounded percentages) and when
    the table has subtotals (nested sections make the simple sum invalid).
    """
    issues = []
    totals = [i for i, r in enumerate(rows) if r and r[0].strip().lower().startswith("total")]
    if len(totals) != 1:
        return issues
    t = totals[0]
    body = [r for r in rows[:t] if r and any(to_number(c) is not None for c in r[1:])]
    if len(body) < 2:
        return issues
    for col in range(1, len(rows[t])):
        cell = rows[t][col]
        if "%" in cell or to_number(cell) is None:
            continue
        nums = [to_number(r[col]) for r in body if col < len(r)]
        nums = [n for n in nums if n is not None]
        if len(nums) < 2:
            continue
        total, s = to_number(cell), sum(nums)
        if abs(total - s) > max(1.0, 0.005 * abs(total)):
            issues.append(f"column {col + 1}: rows sum to {s:,.0f} but total says {total:,.0f}")
    return issues


# ---------------------------------------------------------------- shaping

def _tokens(result: OcrResult, scale: float) -> list[dict]:
    """OCR output -> pdfplumber-like word dicts in PDF points (pixels / scale).

    Uses PaddleOCR's per-word boxes where it gives them; otherwise splits a
    line's box across its tokens in proportion to their length.
    """
    out = []
    for line in result.lines:
        if line.words and len(line.words) == len(line.text.split()):
            pieces = line.words
        else:
            x0, top, x1, bottom = line.box
            tokens = line.text.split()
            span, total = x1 - x0, max(1, sum(len(t) for t in tokens) + len(tokens) - 1)
            pieces, cursor = [], x0
            for tok in tokens:
                w = span * len(tok) / total
                pieces.append((tok, (cursor, top, cursor + w, bottom)))
                cursor += w + span / total
        for text, (x0, top, x1, bottom) in pieces:
            out.append({"text": text, "x0": x0 / scale, "x1": x1 / scale,
                        "top": top / scale, "bottom": bottom / scale, "conf": line.conf})
    return out


def as_words(result: OcrResult, scale: float = 1.0) -> list[dict]:
    return _tokens(result, scale)


VALUE = re.compile(r"^[-−(+]?[$€£₹]?\d[\d,]*(\.\d+)?(-\d+(\.\d+)?)?%?\)?,?$")


def paired_values(result: OcrResult, scale: float = 1.0) -> list[list[str]]:
    """Pair every printed value with the label nearest to it.

    Handles the layouts table alignment can't: labels and values far apart
    ("Fuel cells ........ 47%"), side-by-side key/value lists, values above
    bars with labels below, and pie labels split over two lines ("Electrical
    Systems," / "40-45%"). Same-row labels to the left win; otherwise the
    nearest label directly above or below.
    """
    # Titles and captions are sentences: their numbers ("17 of the 20
    # categories", "Sept. 2026") belong to the sentence, not to the chart.
    result = OcrResult(result.engine, [l for l in result.lines if not _is_prose(l.text)], result.issues)
    toks = _tokens(result, scale)
    if not toks:
        return []
    height = sorted(t["bottom"] - t["top"] for t in toks)[len(toks) // 2]
    values = [t for t in toks if VALUE.match(t["text"])]
    # Label words: anything that isn't a value - including "&" in "Land & Shell".
    texts = [t for t in toks if t not in values and not re.fullmatch(r"[\W_]*\d[\d\W_]*", t["text"])]

    # Words -> label phrases: same row and close together.
    phrases: list[dict] = []
    for t in sorted(texts, key=lambda t: (round(t["top"] / (height * 0.6)), t["x0"])):
        p = phrases[-1] if phrases else None
        if p and abs(t["top"] - p["top"]) < height * 0.6 and 0 <= t["x0"] - p["x1"] < height * 1.2:
            p["text"] += " " + t["text"]
            p["x1"], p["bottom"] = t["x1"], max(p["bottom"], t["bottom"])
        else:
            phrases.append(dict(t))
    # Two-line labels: a phrase directly under another with no value on the
    # upper one's row continues it ("Mobile" / "turbines").
    def overlaps_row(v, p):
        return min(v["bottom"], p["bottom"]) - max(v["top"], p["top"]) > 0.25 * height

    def continues(upper, lower) -> bool:
        """Is `lower` the second line of label `upper`? In a list each value
        lines up with one row; a two-line label's value straddles both."""
        if not (-0.2 * height <= lower["top"] - upper["bottom"] < height * 0.8):
            return False
        if abs(lower["x0"] - upper["x0"]) > height * 1.5:
            return False
        own = [v for v in values if overlaps_row(v, upper) and v["x0"] >= upper["x1"] - 2]
        return all(overlaps_row(v, lower) for v in own)

    merged: list[dict] = []
    for p in sorted(phrases, key=lambda p: (p["top"], p["x0"])):
        up = next((m for m in reversed(merged) if continues(m, p)), None)
        if up is not None:
            up["text"] += " " + p["text"]
            up["x1"], up["bottom"] = max(up["x1"], p["x1"]), p["bottom"]
        else:
            merged.append(p)

    rows: dict[int, list[str]] = {}
    order: list[int] = []
    for v in sorted(values, key=lambda v: (v["top"], v["x0"])):
        best, best_d = None, None
        for i, p in enumerate(merged):
            same_row = min(v["bottom"], p["bottom"]) - max(v["top"], p["top"]) > 0.3 * height
            if same_row and p["x1"] <= v["x0"] + 2:
                d = v["x0"] - p["x1"]
            elif min(v["x1"], p["x1"]) - max(v["x0"], p["x0"]) > -height:  # same column
                # value under its label (pie) or above its bar with the label below
                gap = p["top"] - v["bottom"] if p["top"] >= v["top"] else v["top"] - p["bottom"]
                d = 1.5 * max(0.0, gap)
                if d > height * 20:
                    continue
            else:
                continue
            if best_d is None or d < best_d:
                best, best_d = i, d
        if best is None or best_d > height * 20:
            continue
        if best not in rows:
            rows[best] = [merged[best]["text"].rstrip(":,")]
            order.append(best)
        rows[best].append(v["text"].rstrip(","))
    return [rows[i] for i in order]


LABELLED_VALUE = re.compile(r"^(?P<label>.*?[^\d\s,.%-])[:,]?\s+(?P<values>[-−(]?[$€£₹]?\d[\d,.%\s()+-]*)$")


def labelled_values(result: OcrResult) -> list[list[str]]:
    """Inline data labels like "Grain: 5,867, 5.5%" -> ["Grain", "5,867", "5.5%"]."""
    rows = []
    for line in sorted(result.lines, key=lambda l: (l.box[1], l.box[0])):
        m = LABELLED_VALUE.match(line.text.strip())
        if not m or len(m.group("label").strip(" :,")) < 2:
            continue
        values = [v.strip(" ,") for v in re.split(r",\s+|\s+", m.group("values")) if v.strip(" ,")]
        values = [v for v in values if NUMBER.fullmatch(v)]
        if values:
            rows.append([m.group("label").strip(" :,"), *values])
    return rows


def _is_prose(text: str) -> bool:
    return len(re.findall(r"[A-Za-z]{2,}", text)) >= 5


def chart_rows(result: OcrResult, scale: float = 1.0) -> list[list[str]]:
    """Inline "Label: value" lines first, then position-based pairing for the
    values left over (sidebars, values on bars, pie labels)."""
    inline = labelled_values(result)
    used = {id(l) for l in result.lines if LABELLED_VALUE.match(l.text.strip())} if inline else set()
    rest = OcrResult(result.engine, [l for l in result.lines if id(l) not in used], result.issues)
    return inline + paired_values(rest, scale)


LEGEND_GLYPHS = re.compile(r"[□■▪▫●○◆◇▲△►▶◼◻]")


def chart_layout_issues(rows: list[list[str]]) -> list[str]:
    """Signs that label pairing misread a chart's layout (stacked areas, legends,
    annotations): legend symbols in labels, axis years among the values, or
    rows of wildly different widths. Then the pairing must not be shown as data."""
    issues = []
    if not rows:
        return issues
    if any(LEGEND_GLYPHS.search(r[0] or "") for r in rows):
        issues.append("legend symbols mixed into labels")
    values = [v for r in rows for v in r[1:]]
    if sum(1 for v in values if re.fullmatch(r"(19|20)\d\d", v.strip())) >= 2:
        issues.append("axis years mixed into values")
    widths = {len(r) for r in rows}
    if len(rows) >= 2 and max(widths) - min(widths) >= 2:
        issues.append("rows have inconsistent numbers of values")
    return [f"chart layout not understood ({', '.join(issues)})"] if issues else []


def read_region(image: Image.Image, shape=None) -> tuple[OcrResult, list[list[str]] | None]:
    """Paddle first; Cortex when Paddle's reading has issues and Cortex is set up.

    `shape` turns a reading into table rows (or None for plain text); the
    quality checks - including totals reconciling - run on those rows, so a
    misread table triggers the fallback, not only low confidence.
    """
    def attempt(engine) -> tuple[OcrResult, list[list[str]] | None]:
        result = engine().read(image)
        rows = shape(result) if shape else None
        result.issues = assess(result, rows)
        return result, rows

    first = attempt(PaddleEngine) if PaddleEngine.available() else None
    if first is not None and (not first[0].issues or not CortexEngine.available()):
        return first
    if CortexEngine.available():
        try:
            second = attempt(CortexEngine)
        except Exception as e:  # untested path: never let it break the preview
            if first is not None:
                first[0].issues.append(f"Cortex fallback failed: {e}")
                return first
            raise
        if first is None or len(second[0].issues) < len(first[0].issues):
            return second
        return first
    return OcrResult("none", [], ["no OCR engine installed (requirements-ml.txt)"]), None


def png(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
