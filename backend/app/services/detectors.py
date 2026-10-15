"""Pluggable layout detectors, so their overlays can be compared in the UI.

  structure - reads the PDF's own objects with pdfplumber (detect.py). Fast,
              no model, but only finds what the PDF explicitly encodes.
  docling   - IBM Docling layout model on the rendered page image.
  paddle    - PaddleOCR PP-DocLayout on the rendered page image.

ML detectors only propose *boxes*. Extraction always reads the exact text
layer inside whichever box the user picks, so a better detector improves
selection, never the numbers themselves.

ML models are optional (requirements-ml.txt); a missing one is reported as
unavailable rather than breaking the app.
"""

import ctypes
import importlib.util
import os
import threading
import time
from collections.abc import Callable

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_raw
from pdfplumber.page import Page
from PIL import Image

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")  # skips a slow network probe

from ..pdf import PDFIUM_LOCK, normalise, open_page
from .detect import METHODS, detect_page

RENDER_SCALE = 2.0  # 144 dpi: what both layout models are trained around

# Model label -> our region kind. Figures are split into image / vector chart
# afterwards by looking at what the PDF actually has inside the box.
LABEL_MAP = {
    "table": "table",
    "picture": "figure",
    "image": "figure",
    "figure": "figure",
    "chart": "figure",
    "title": "heading",
    "doc_title": "heading",
    "section_header": "heading",
    "paragraph_title": "heading",
    "figure_title": "text",
    "chart_title": "text",
    "table_title": "text",
    "caption": "text",
    "text": "text",
    "paragraph": "text",
    "content": "text",
    "abstract": "text",
    "list_item": "text",
    "footnote": "text",
    "reference": "text",
    "page_header": "text",
    "page_footer": "text",
    "header": "text",
    "footer": "text",
    "number": "text",
    "aside_text": "text",
    "formula": "text",
    "key_value_region": "text",
    "form": "text",
}


def _render(pdf_path, page_no: int) -> Image.Image:
    # pdfium applies rotation and the cropbox, so pixel coords map straight
    # onto the normalised display frame PDF.js uses.
    with PDFIUM_LOCK:
        doc = pdfium.PdfDocument(pdf_path)
        try:
            return doc[page_no - 1].render(scale=RENDER_SCALE).to_pil().convert("RGB")
        finally:
            doc.close()


def _to_regions(page: Page, boxes: list[tuple[str, float, list[float]]], pictures) -> list[dict]:
    """(model_label, score, normalised box) -> Region dicts."""
    img_boxes = [normalise(page, *b) for b in pictures]
    regions = []
    for label, score, bbox in boxes:
        kind = LABEL_MAP.get(label.lower(), "text")
        bbox = [round(min(1.0, max(0.0, v)), 5) for v in bbox]
        if kind == "figure":
            raster = sum(_overlap(bbox, b) for b in img_boxes) > 0.5
            kind = "image" if raster else "chart_vector"
        regions.append(
            {
                "id": f"p{page.page_number}-{len(regions)}",
                "kind": kind,
                "bbox": bbox,
                "method": METHODS[kind],
                "label": f"{label} ({score:.0%})",
                "model_label": label,
                "score": round(score, 3),
            }
        )
    return regions


def _ml_regions(boxes: list[tuple[str, float, list[float]]]) -> list[dict]:
    """Model output in the coarse vocabulary detect_page merges: table / figure / heading / text."""
    return [
        {
            "kind": LABEL_MAP.get(label.lower(), "text"),
            "bbox": [min(1.0, max(0.0, v)) for v in bbox],
            "label": f"{label} ({score:.0%})",
            "model_label": label,
            "score": round(score, 3),
        }
        for label, score, bbox in boxes
    ]


def visible_pictures(pdf_path, page: Page) -> list[tuple[float, float, float, float]] | None:
    """Each embedded image's *visible* box, in pdfplumber coordinates.

    Layout tools place a full photo and crop it with a clipping mask; pdfplumber
    reports the uncropped size, which then covers whatever sits below the
    visible part (Kyndryl p27: a banner box hid a pie chart). pdfium exposes the
    clip path, so intersect the two. Rotated pages fall back to pdfplumber.
    """
    if page.rotation:
        return None
    pl_x0, pl_top = page.mediabox[0], page.mediabox[1]
    out = []
    with PDFIUM_LOCK:
        doc = pdfium.PdfDocument(pdf_path)
        try:
            pg = doc[page.page_number - 1]
            mb_l, _, _, mb_t = pg.get_mediabox()
            for obj in pg.get_objects(filter=(pdfium_raw.FPDF_PAGEOBJ_IMAGE,), max_depth=3):
                l, b, r, t = obj.get_bounds()
                clip = _clip_bounds(obj)
                if clip:
                    l, b, r, t = max(l, clip[0]), max(b, clip[1]), min(r, clip[2]), min(t, clip[3])
                if r - l < 1 or t - b < 1:
                    continue  # fully clipped away
                out.append((l - mb_l + pl_x0, mb_t - t + pl_top, r - mb_l + pl_x0, mb_t - b + pl_top))
        finally:
            doc.close()
    return out


def _clip_bounds(obj) -> tuple[float, float, float, float] | None:
    clip = pdfium_raw.FPDFPageObj_GetClipPath(obj.raw)
    if not clip:
        return None
    xs, ys = [], []
    x, y = ctypes.c_float(), ctypes.c_float()
    for i in range(pdfium_raw.FPDFClipPath_CountPaths(clip)):
        for j in range(pdfium_raw.FPDFClipPath_CountPathSegments(clip, i)):
            seg = pdfium_raw.FPDFClipPath_GetPathSegment(clip, i, j)
            if pdfium_raw.FPDFPathSegment_GetPoint(seg, x, y):
                xs.append(x.value)
                ys.append(y.value)
    return (min(xs), min(ys), max(xs), max(ys)) if xs else None


def _overlap(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1])
    return ix * iy / area if area else 0.0


# ---------------------------------------------------------------- models
# Loaded lazily on first use and kept for the life of the process. Each model
# has a lock: neither library documents thread-safe inference.
_models: dict[str, object] = {}
_model_locks = {"docling": threading.Lock(), "paddle": threading.Lock()}


def _docling_boxes(pdf_path, page_no: int):
    """Docling's layout stage only (no OCR, no table structure), via its public
    converter API - its internal model classes change between releases."""
    with _model_locks["docling"]:
        conv = _models.get("docling")
        if conv is None:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption

            opts = PdfPipelineOptions(do_ocr=False, do_table_structure=False)
            conv = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
            )
            _models["docling"] = conv
        page = conv.convert(str(pdf_path), page_range=(page_no, page_no)).pages[0]
    w, h = page.size.width, page.size.height
    return [
        (
            c.label.value if hasattr(c.label, "value") else str(c.label),
            float(c.confidence),
            [c.bbox.l / w, c.bbox.t / h, c.bbox.r / w, c.bbox.b / h],  # TOPLEFT origin
        )
        for c in page.predictions.layout.clusters
    ]


def _paddle_boxes(pdf_path, page_no: int):
    import numpy as np

    img = _render(pdf_path, page_no)
    with _model_locks["paddle"]:
        model = _models.get("paddle")
        if model is None:
            from paddleocr import LayoutDetection

            model = LayoutDetection(model_name="PP-DocLayout_plus-L")
            _models["paddle"] = model
        result = model.predict(np.asarray(img)[:, :, ::-1], batch_size=1)  # BGR, like OpenCV
    return [
        (b["label"], float(b["score"]), [float(v) / d for v, d in zip(b["coordinate"], (img.width, img.height) * 2)])
        for res in result
        for b in res["boxes"]
    ]


def _ml_detector(boxes_fn: Callable) -> Callable:
    def run(doc_id: str, pdf_path, page_no: int) -> dict:
        started = time.perf_counter()
        # Inference happens outside the pdfplumber lock: it takes ~0.5 s and
        # must not hold up fast structure requests on the same document.
        boxes = boxes_fn(pdf_path, page_no)
        with open_page(doc_id, pdf_path, page_no) as page:
            pictures = visible_pictures(pdf_path, page) or [
                (i["x0"], i["top"], i["x1"], i["bottom"]) for i in page.images
            ]
            regions = _to_regions(page, boxes, pictures)
            x0, top, x1, bottom = page.cropbox
        return {
            "page": page_no,
            "width": x1 - x0,
            "height": bottom - top,
            "scanned": False,
            "regions": regions,
            "ms": round((time.perf_counter() - started) * 1000, 1),
        }

    return run


def _structure(doc_id: str, pdf_path, page_no: int) -> dict:
    with open_page(doc_id, pdf_path, page_no) as page:
        return detect_page(page, pictures=visible_pictures(pdf_path, page))


def needs_layout_model(page: Page, pictures) -> bool:
    """Is this page graphic enough that a layout model should place the figures?

    Text- and table-only pages skip the model and stay at ~20-50 ms. Measured
    on the eval set: every page whose figures the rules got wrong trips one of these.
    """
    area = page.width * page.height
    if any((b[2] - b[0]) * (b[3] - b[1]) / area > 0.03 for b in (pictures or [])):
        return True
    return len(page.curves) >= 10 or len(page.rects) + len(page.lines) >= 15


def _combine(docling: list[dict], paddle: list[dict]) -> list[dict]:
    """Docling's regions, with a coarse figure replaced by the finer PP-DocLayout
    chart(s) inside it. Docling sometimes boxes a whole infographic panel where
    PP-DocLayout isolates the chart; otherwise Docling is the better of the two."""
    charts = [p for p in paddle if p["kind"] == "figure" and p["model_label"] == "chart"]
    out = []
    for d in docling:
        if d["kind"] == "figure":
            inner = [
                c for c in charts
                if _overlap(c["bbox"], d["bbox"]) > 0.9 and _iou_n(c["bbox"], d["bbox"]) < 0.5
                and _area_n(c["bbox"]) > 0.2 * _area_n(d["bbox"])
            ]
            if inner:
                out.extend(inner)
                continue
        out.append(d)
    return out


def _area_n(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _iou_n(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = _area_n(a) + _area_n(b) - ix * iy
    return ix * iy / union if union else 0.0


def _auto_models() -> list[str]:
    """Which layout model(s) Auto uses. DOCPARSE_AUTO_MODEL=docling|paddle|both
    overrides; the default is decided by eval/evaluate.py results."""
    wanted = os.environ.get("DOCPARSE_AUTO_MODEL", "both")
    models = ["docling", "paddle"] if wanted == "both" else [wanted]
    available = [m for m in models if is_available(m)]
    if available:
        return available
    return [m for m in ("docling", "paddle") if is_available(m)][:1]


def _auto(doc_id: str, pdf_path, page_no: int) -> dict:
    """Structure for tables and text; a layout model for figures, only where needed."""
    started = time.perf_counter()
    models = _auto_models()
    with open_page(doc_id, pdf_path, page_no) as page:
        pictures = visible_pictures(pdf_path, page)
        if not models or not needs_layout_model(page, pictures):
            result = detect_page(page, pictures=pictures)
            return {**result, "model": None}
    # Inference outside the pdfplumber lock (see _ml_detector).
    found = {
        m: _ml_regions((_docling_boxes if m == "docling" else _paddle_boxes)(pdf_path, page_no))
        for m in models
    }
    ml = _combine(found["docling"], found["paddle"]) if len(found) == 2 else next(iter(found.values()))
    with open_page(doc_id, pdf_path, page_no) as page:
        result = detect_page(page, pictures=pictures, ml_regions=ml)
    return {**result, "model": "+".join(models), "ms": round((time.perf_counter() - started) * 1000, 1)}


DETECTORS: dict[str, dict] = {
    "auto": {
        "label": "Auto",
        "description": "PDF structure for tables and text; a layout model places figures on graphic pages.",
        "requires": [],
        "run": _auto,
    },
    "structure": {
        "label": "PDF structure",
        "description": "Reads tables, images and drawings encoded in the PDF. No model.",
        "requires": [],
        "run": _structure,
    },
    "docling": {
        "label": "Docling layout",
        "description": "IBM Docling layout model (CPU), layout stage only.",
        "requires": ["docling"],
        "run": _ml_detector(_docling_boxes),
    },
    "paddle": {
        "label": "PP-DocLayout",
        "description": "PaddleOCR PP-DocLayout_plus-L on the page image (CPU).",
        "requires": ["paddleocr", "paddle"],
        "run": _ml_detector(_paddle_boxes),
    },
}


def is_available(name: str) -> bool:
    return all(importlib.util.find_spec(mod) is not None for mod in DETECTORS[name]["requires"])


def list_detectors() -> list[dict]:
    return [
        {"id": k, "label": v["label"], "description": v["description"], "available": is_available(k)}
        for k, v in DETECTORS.items()
    ]


def warm_up() -> None:
    """Load the layout and OCR models once at startup (background thread).

    Docling alone takes up to a minute to load its weights; without this the
    first user action after a deploy - opening a graphic page, saving a
    template - would wait for it.
    """
    import io
    import logging

    log = logging.getLogger("docparse.warmup")
    blank = pdfium.PdfDocument.new()
    blank.new_page(612, 792)
    buf = io.BytesIO()
    blank.save(buf)
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(buf.getvalue())
        path = f.name
    for name, fn in (("docling", _docling_boxes), ("paddle", _paddle_boxes)):
        if is_available(name):
            started = time.perf_counter()
            try:
                fn(path, 1)
                log.warning("warm-up: %s ready in %.1fs", name, time.perf_counter() - started)
            except Exception as e:  # a failed warm-up must never stop the server
                log.warning("warm-up: %s failed: %s", name, e)
    try:
        from . import ocr

        if ocr.PaddleEngine.available():
            from PIL import Image

            ocr.PaddleEngine().read(Image.new("RGB", (64, 32), "white"))
            log.warning("warm-up: OCR ready")
    except Exception as e:
        log.warning("warm-up: OCR failed: %s", e)
