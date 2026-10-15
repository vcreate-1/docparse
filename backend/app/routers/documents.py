from datetime import UTC, datetime
from typing import Literal

import pypdfium2 as pdfium
from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..config import settings
from ..pdf import PDFIUM_LOCK, open_page
from ..services import search as search_service
from ..services.detect import suggest_anchor
from ..services.detectors import DETECTORS, is_available
from ..services import fields as fields_service
from ..services.extract import extract_region, run_ocr
from ..storage import storage

router = APIRouter(prefix="/api/documents", tags=["documents"])

# Handlers are plain `def`: PDF work is CPU-bound, and FastAPI runs sync
# handlers in its thread pool so one slow page never blocks the event loop.


def _meta(doc_id: str) -> dict:
    try:
        meta = storage.read_json(doc_id, "meta.json")
    except KeyError:
        meta = None
    if meta is None:
        raise HTTPException(404, "Document not found")
    return meta


@router.post("")
def upload(file: UploadFile) -> dict:
    data = file.file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"PDF larger than {settings.max_upload_mb} MB")

    doc_id = storage.sha256(data)
    if storage.exists(doc_id):  # same bytes uploaded before: reuse everything
        return {**_meta(doc_id), "deduplicated": True}

    try:
        page_texts, page_sizes, title = _scan(data)
    except pdfium.PdfiumError as e:
        if "password" in str(e).lower():
            raise HTTPException(422, "Password-protected PDFs are not supported yet")
        raise HTTPException(422, "Not a readable PDF")
    page_count = len(page_texts)

    storage.put_pdf(doc_id, data)
    storage.write_json(doc_id, "pages.json", search_service.build_index(page_texts))
    meta = {
        "id": doc_id,
        "file_name": file.filename or "document.pdf",
        "title": title,
        "page_count": page_count,
        "page_sizes": page_sizes,
        "size_bytes": len(data),
        "text_pages": sum(1 for t in page_texts if len(t.strip()) >= 20),
        "uploaded_at": datetime.now(UTC).isoformat(),
        "facility_ref_id": None,  # filled once the RAG facility upload is wired in
    }
    storage.write_json(doc_id, "meta.json", meta)
    return {**meta, "deduplicated": False}


def _scan(data: bytes) -> tuple[list[str], list[list[float]], str]:
    """One fast pdfium pass: validates the file, page text for search, page sizes.

    Page sizes are cropbox with rotation applied - what PDF.js displays - so
    the viewer can lay out every page slot before PDF.js has loaded anything.
    """
    with PDFIUM_LOCK:
        return _scan_locked(data)


def _scan_locked(data: bytes) -> tuple[list[str], list[list[float]], str]:
    pdf = pdfium.PdfDocument(data)
    try:
        texts, sizes = [], []
        for i in range(len(pdf)):
            page = pdf[i]
            textpage = page.get_textpage()
            texts.append(textpage.get_text_range())
            w, h = page.get_size()
            sizes.append([round(w, 2), round(h, 2)])
            textpage.close()
            page.close()
        title = pdf.get_metadata_dict().get("Title", "") or ""
        return texts, sizes, title
    finally:
        pdf.close()


@router.get("")
def list_documents() -> list[dict]:
    return storage.list_docs()


@router.get("/{doc_id}")
def get_document(doc_id: str) -> dict:
    return _meta(doc_id)


@router.get("/{doc_id}/file")
def get_file(doc_id: str) -> FileResponse:
    _meta(doc_id)
    # FileResponse honours Range requests, so PDF.js can fetch only the
    # pages it is about to render instead of the whole 400-page file.
    return FileResponse(
        storage.pdf_path(doc_id),
        media_type="application/pdf",
        headers={"Cache-Control": "private, max-age=86400, immutable"},
    )


def _regions(doc_id: str, page_no: int, detector: str = "structure") -> dict:
    meta = _meta(doc_id)
    if not 1 <= page_no <= meta["page_count"]:
        raise HTTPException(404, "Page out of range")
    if detector not in DETECTORS:
        raise HTTPException(404, f"Unknown detector '{detector}'")
    if not is_available(detector):
        raise HTTPException(409, f"Detector '{detector}' is not installed (requirements-ml.txt)")
    cache_name = f"regions/{detector}/{page_no}.json"
    cached = storage.read_json(doc_id, cache_name)
    if cached is not None:
        return cached
    result = DETECTORS[detector]["run"](doc_id, storage.pdf_path(doc_id), page_no)
    storage.write_json(doc_id, cache_name, result)
    return result


@router.get("/{doc_id}/pages/{page_no}/regions")
def page_regions(doc_id: str, page_no: int, detector: str = "structure") -> dict:
    return _regions(doc_id, page_no, detector)


@router.get("/{doc_id}/search")
def search(doc_id: str, q: str, limit: int = 10) -> list[dict]:
    _meta(doc_id)
    return search_service.search(doc_id, q, limit)


class FieldSpec(BaseModel):
    name: str = Field(min_length=1, max_length=60, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    type: Literal["text", "number", "percent", "date", "boolean", "list"] = "text"
    description: str | None = Field(default=None, max_length=300)


class ExtractRequest(BaseModel):
    page: int
    bbox: list[float] = Field(min_length=4, max_length=4)
    kind: Literal["table", "chart_vector", "image", "heading", "text", "custom", "scanned_page"]
    # Text regions: structured fields to pull out of the prose with the language model.
    fields: list[FieldSpec] = Field(default=[], max_length=30)
    instruction: str | None = Field(default=None, max_length=1000)
    rows: bool = False  # one row per item the text lists, instead of one record


@router.post("/{doc_id}/extract")
def extract(doc_id: str, req: ExtractRequest) -> dict:
    x0, y0, x1, y1 = req.bbox
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise HTTPException(422, "bbox must be normalised 0..1 with x0<x1, y0<y1")
    regions = _regions(doc_id, req.page)["regions"]
    with open_page(doc_id, storage.pdf_path(doc_id), req.page) as page:
        result = extract_region(page, req.bbox, req.kind)
    if "needs_ocr" in result:
        # Seconds of OCR run outside the document lock, so the viewer keeps
        # loading other pages of this document meanwhile.
        result = run_ocr(result, storage.pdf_path(doc_id), req.page, req.bbox)
    if req.fields and result.get("type") == "text":
        anchor = suggest_anchor(regions, req.bbox)
        result = {**result, **fields_service.extract_text_fields(
            doc_id, req.page, result.get("text", ""), [f.model_dump() for f in req.fields],
            req.instruction, anchor, rows=req.rows)}
    return {**result, "anchor_heading": suggest_anchor(regions, req.bbox)}
