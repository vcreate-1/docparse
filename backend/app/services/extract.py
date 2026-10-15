"""Preview extraction for a selected region.

Shows the user what the batch run would produce for a box before they save it
into a template. Each result says how trustworthy it is:
  exact       - read straight from the text layer
  near_exact  - derived from vector geometry
  approximate - needs a vision model (estimated values)
"""

from pdfplumber.page import Page

from ..pdf import denormalise
from .tables import align_table, structure, table_quality


def extract_region(page: Page, bbox: list[float], kind: str, ocr: bool = True) -> dict:
    """Read a region from the text layer. Where there is no text layer, return
    {"needs_ocr": ...} instead (when `ocr`), so the caller can run the slow OCR
    step *after* releasing the document lock - see run_ocr()."""
    region = page.crop(_snap_to_words(page, denormalise(page, bbox)))
    words = [w["text"] for w in region.extract_words()]
    # No text layer under the box: the content is pixels, so it needs OCR.
    pixels_only = len(words) < 3 and ocr

    if kind == "table":
        result = _table(region)
        if result["method"] == "none" and pixels_only:
            return {"needs_ocr": "table"}
        return result
    if kind in ("heading", "text", "custom"):
        if pixels_only:
            return {"needs_ocr": "text"}
        return {
            "type": "text",
            "confidence": "exact",
            "method": "text_layer",
            "text": region.extract_text().strip(),
        }
    if kind == "chart_vector":
        numbers = [w for w in words if _is_number(w)]
        return {
            "type": "chart",
            "method": "vector_geometry",
            "confidence": "exact" if len(numbers) >= 3 else "near_exact",
            "labels": words[:80],
            "numeric_labels": numbers[:40],
            "drawing_count": len(region.rects) + len(region.lines) + len(region.curves),
            "note": (
                "Data labels found in the text layer - values can be read exactly."
                if len(numbers) >= 3
                else "No data labels - values will be computed from bar/line geometry "
                "calibrated against the axis ticks."
            ),
        }
    if kind == "image":
        if ocr:
            return {"needs_ocr": "chart", "words": words}
        return {
            "type": "chart",
            "method": "vision_llm",
            "confidence": "approximate",
            "labels": words[:80],
            "note": "Raster image; values need OCR or a vision model.",
        }
    if kind == "scanned_page":
        if ocr:
            return {"needs_ocr": "page"}
        return {"type": "text", "method": "ocr", "confidence": "approximate", "text": "",
                "note": "No text layer; needs OCR."}
    raise ValueError(f"unknown kind: {kind}")


# ---------------------------------------------------------------- OCR paths

OCR_SCALE = 4  # ~288 dpi: small chart labels need it


def run_ocr(pending: dict, pdf_path, page_no: int, bbox: list[float]) -> dict:
    """Second half of extract_region for pixel-only regions; no pdfplumber needed."""
    what = pending["needs_ocr"]
    if what == "table":
        return _ocr_table(pdf_path, page_no, bbox)
    if what == "chart":
        return _ocr_chart(pdf_path, page_no, bbox, pending.get("words", []))
    return _ocr_text(pdf_path, page_no, [0, 0, 1, 1] if what == "page" else bbox)


def _crop_image(pdf_path, page_no: int, bbox: list[float]):
    import pypdfium2 as pdfium

    from ..pdf import PDFIUM_LOCK

    with PDFIUM_LOCK:
        doc = pdfium.PdfDocument(pdf_path)
        try:
            img = doc[page_no - 1].render(scale=OCR_SCALE).to_pil().convert("RGB")
        finally:
            doc.close()
    w, h = img.size  # rendered with rotation + cropbox: same frame as bbox
    return img.crop((int(bbox[0] * w), int(bbox[1] * h), int(bbox[2] * w), int(bbox[3] * h)))


def _ocr_meta(result, extra_issues=()) -> dict:
    issues = list(result.issues) + list(extra_issues)
    return {
        "method": f"ocr:{result.engine}",
        "confidence": "approximate" if issues else "near_exact",
        "ocr_confidence": round(result.mean_conf, 3),
        "issues": issues,
    }


def _ocr_table(pdf_path, page_no: int, bbox: list[float]) -> dict:
    """OCR a picture of a table. Two readings, keep the cleaner: column
    alignment (real grids with headers) or label/value pairing (key/value
    lists laid out side by side)."""
    from . import ocr

    def shape(r):
        aligned = structure(align_table(ocr.as_words(r, OCR_SCALE)))
        paired = {"title": None, "columns": [], "data": ocr.paired_values(r, OCR_SCALE)}
        best = max((aligned, paired), key=_pair_quality)
        shape.table = best
        return [best["columns"], *best["data"]] if best["columns"] else best["data"]

    result, _ = ocr.read_region(_crop_image(pdf_path, page_no, bbox), shape=shape)
    table = getattr(shape, "table", {"title": None, "columns": [], "data": []})
    if not table["columns"] and table["data"]:
        width = max(len(r) for r in table["data"])
        table = {**table, "columns": ["Label"] + [f"Value {i}" for i in range(1, width)],
                 "data": [r + [""] * (width - len(r)) for r in table["data"]]}
    return {"type": "table", **_ocr_meta(result), **table, "note": _ocr_note(result, "table")}


def _pair_quality(t: dict) -> float:
    """Share of rows that are a short label plus at least one value."""
    data = t.get("data") or []
    if not data:
        return 0.0
    good = sum(1 for r in data if r and r[0] and len(r[0].split()) <= 5 and any(c for c in r[1:]))
    return good / len(data) + 0.05 * len(t.get("columns") or [])


def _ocr_text(pdf_path, page_no: int, bbox: list[float]) -> dict:
    from . import ocr

    result, _ = ocr.read_region(_crop_image(pdf_path, page_no, bbox))
    text = "\n".join(line.text for line in sorted(result.lines, key=lambda l: (round(l.box[1] / 12), l.box[0])))
    return {"type": "text", **_ocr_meta(result), "text": text, "note": _ocr_note(result, "text")}


def _ocr_chart(pdf_path, page_no: int, bbox: list[float], words: list[str]) -> dict:
    """Picture charts: vision model for structure + OCR as the check when a
    vision model is available; otherwise OCR label pairing, which only works
    for simple labelled charts and says so when the layout defeats it."""
    from . import chart_read, ocr

    image = _crop_image(pdf_path, page_no, bbox)
    result, rows = ocr.read_region(image, shape=lambda r: ocr.chart_rows(r, OCR_SCALE))
    try:
        vision = chart_read.read_chart(ocr.png(image), result)
    except Exception as e:  # a model hiccup must not lose the OCR reading
        vision = None
        result.issues.append(f"vision model failed: {e}")
    if vision is not None:
        return {**vision, "ocr_confidence": round(result.mean_conf, 3),
                "labels": [line.text for line in result.lines][:80]}

    lines = [line.text for line in result.lines]
    numbers = [t for t in lines if _is_number(t)]
    layout = ocr.chart_layout_issues(rows or [])
    if rows and len(rows) >= 2 and not layout:
        width = max(len(r) for r in rows)
        return {
            "type": "chart", **_ocr_meta(result),
            "columns": ["Label"] + [f"Value {i}" for i in range(1, width)],
            "data": [r + [""] * (width - len(r)) for r in rows],
            "labels": lines[:80], "numeric_labels": numbers[:40],
            "note": _ocr_note(result, "chart labels"),
        }
    why = layout or ["no printed values: chart values need a vision model (estimate)"]
    meta = _ocr_meta(result, why)
    return {"type": "chart", **meta, "confidence": "approximate",
            "labels": lines[:80] or words[:80], "numeric_labels": numbers[:40],
            "note": _ocr_note(result, "axis and title text") + " A vision model is needed to read this chart's data."}


def _ocr_note(result, what: str) -> str:
    engine = {"paddleocr": "PaddleOCR", "cortex": "Snowflake Cortex"}.get(result.engine, result.engine)
    if not result.lines:
        return f"OCR ({engine}) found no text. " + "; ".join(result.issues)
    base = f"{what.capitalize()} read by OCR ({engine}, {result.mean_conf:.0%} mean confidence)."
    if result.issues:
        return base + " Check before use: " + "; ".join(result.issues) + "."
    return base + " Spot-check a value before relying on it."


def _snap_to_words(page: Page, box: tuple) -> tuple:
    """Grow the box to whole words: a box drawn (or detected) a hair too
    tight would otherwise clip "233,285" to "233,28" or "Products" to "oducts"."""
    x0, top, x1, bottom = box
    for w in page.extract_words():
        ix = min(x1, w["x1"]) - max(x0, w["x0"])
        iy = min(bottom, w["bottom"]) - max(top, w["top"])
        if ix > 0 and iy > 0 and ix * iy > 0.5 * (w["x1"] - w["x0"]) * (w["bottom"] - w["top"]):
            x0, top, x1, bottom = min(x0, w["x0"]), min(top, w["top"]), max(x1, w["x1"]), max(bottom, w["bottom"])
    # A few points of margin: a box drawn exactly on a table's border line
    # would otherwise crop the line away, and the grid loses its header row.
    pad = 3
    px0, ptop, px1, pbottom = page.bbox
    return (max(px0, x0 - pad), max(ptop, top - pad), min(px1, x1 + pad), min(pbottom, bottom + pad))


def _table(region: Page) -> dict:
    """Read a table two ways and keep the cleaner result.

    Ruled grids (KBB) read best from their lines; borderless tables (SEC
    filings, Word) read best from text alignment. Scoring both avoids having
    to know in advance which kind of table this is.
    """
    candidates = []
    try:
        ruled = region.extract_tables()
    except Exception:
        ruled = []
    if ruled:
        candidates.append(("pdfplumber:lines", structure(max(ruled, key=len))))
    aligned = align_table(region.extract_words(keep_blank_chars=False))
    if aligned:
        candidates.append(("alignment", structure(aligned)))
    if candidates:
        method, best = max(candidates, key=lambda c: table_quality(c[1]))
        if table_quality(best) > 0.2:
            return {"type": "table", "method": method, "confidence": "exact", **best}
    return {
        "type": "table",
        "method": "none",
        "confidence": "approximate",
        "title": None,
        "columns": [],
        "data": [],
        "note": "No table structure found here. If this is a picture of a table it needs OCR.",
    }


def _is_number(word: str) -> bool:
    cleaned = word.strip("()%$€£₹,").replace(",", "")
    try:
        float(cleaned)
        return True
    except ValueError:
        return False
