"""Template fields at runtime: fingerprint on save, locate in other documents,
learn from confirmations. Glue between storage, detection, extraction and
anchors.py - the batch runner will call the same functions."""

from ..pdf import open_page
from ..storage import storage
from . import anchors, llm, text_extract
from .extract import extract_region, run_ocr

DETECTOR = "auto"


def _detect(doc_id: str):
    from ..routers.documents import _regions  # cached per page

    return lambda page: _regions(doc_id, page, DETECTOR)


def _extract_text_layer(doc_id: str):
    path = storage.pdf_path(doc_id)

    def extract(page: int, region: dict):
        kind = "text" if region["kind"] == "heading" else region["kind"]
        with open_page(doc_id, path, page) as p:
            out = extract_region(p, region["bbox"], kind, ocr=False)
        return None if "needs_ocr" in out else out

    return extract


def _page_texts(doc_id: str) -> list[str]:
    return storage.read_json(doc_id, "pages.json") or []


def _region_for(doc_id: str, page: int, bbox: list[float], kind: str) -> tuple[dict, list[dict]]:
    """The detected region behind a template target, or the drawn box itself."""
    regions = _detect(doc_id)(page)["regions"]
    best, best_iou = None, 0.0
    for r in regions:
        overlap = anchors._iou(r["bbox"], bbox)
        if overlap > best_iou:
            best, best_iou = r, overlap
    if best is not None and best_iou >= 0.5:
        return best, regions
    drawn_kind = {"custom": "text"}.get(kind, kind)
    return {"kind": drawn_kind, "bbox": bbox, "label": ""}, regions


def fingerprint_target(doc_id: str, target: dict, previous: dict | None = None) -> dict:
    page, bbox = target["page"], target["bbox"]
    region, regions = _region_for(doc_id, page, bbox, target["kind"])
    extraction = _extract_text_layer(doc_id)(page, region)
    texts = _page_texts(doc_id)
    fp = anchors.fingerprint(region, regions, extraction, page, len(texts))
    fp = anchors.with_lookalikes(fp, texts, _detect(doc_id), _extract_text_layer(doc_id), page)
    if previous:  # keep what earlier documents taught this field
        fp["anchor_aliases"] = previous.get("anchor_aliases", [])
    return fp


def locate_target(fp: dict, doc_id: str, with_values: bool = True,
                  field_specs: list[dict] | None = None, instruction: str | None = None,
                  rows: bool = False) -> dict:
    result = anchors.locate(fp, _page_texts(doc_id), _detect(doc_id), _extract_text_layer(doc_id))
    best = result["best"]
    if best and with_values:
        path = storage.pdf_path(doc_id)
        kind = "text" if best["kind"] == "heading" else best["kind"]
        with open_page(doc_id, path, best["page"]) as p:
            values = extract_region(p, best["bbox"], kind)
        if "needs_ocr" in values:
            values = run_ocr(values, path, best["page"], best["bbox"])
        if field_specs and values.get("type") == "text":
            values = {**values, **extract_text_fields(doc_id, best["page"], values.get("text", ""),
                                                      field_specs, instruction, best["anchor"], rows)}
        result["values"] = values
    return result


def confirm_target(fp: dict, doc_id: str, page: int, bbox: list[float]) -> dict:
    """A person confirmed (or corrected) where the field is in this document."""
    region, regions = _region_for(doc_id, page, bbox, fp["kind"])
    anchor = (region.get("title") or anchors.nearest_heading(regions, region["bbox"]) or "").strip()
    fake = {"status": "matched", "margin": 1.0, "best": {"anchor": anchor}}
    return anchors.learn(fp, fake, confirmed=True)


def facility_ref_id(doc_id: str) -> str | None:
    """Upload the document to the RAG facility once; reuse its ref id for every prompt."""
    if not llm.FacilityClient.available():
        return None
    meta = storage.read_json(doc_id, "meta.json") or {}
    if not meta.get("facility_ref_id"):
        meta["facility_ref_id"] = llm.FacilityClient().upload(storage.pdf_path(doc_id), meta.get("file_name", "doc.pdf"))
        storage.write_json(doc_id, "meta.json", meta)
    return meta["facility_ref_id"]


def extract_text_fields(doc_id: str, page: int, text: str, field_specs: list[dict],
                        instruction: str | None, heading: str | None, rows: bool = False) -> dict:
    return text_extract.extract_fields(
        text, field_specs, instruction, heading, rows=rows, ref_id=facility_ref_id(doc_id), pages=[page])
