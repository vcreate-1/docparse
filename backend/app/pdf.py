"""PDF engines and the coordinate convention they share.

Engines (all permissively licensed):
  - pypdfium2 (Apache/BSD): fast whole-document passes at upload (validation,
    page sizes, text index). ~20x faster than pdfplumber for full-text extraction.
  - pdfplumber (MIT): everything geometric (tables, words with fonts, images,
    vector drawings, cropping to a selected region).

Coordinates: bboxes exchanged with the frontend are normalised to 0..1 of the
page *as displayed* (cropbox, rotation applied), top-left origin. That is the
same frame PDF.js renders, so overlays line up at any zoom. pdfplumber already
reports objects in this rotated frame; we only have to subtract the cropbox.
"""

import threading
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pdfplumber
from pdfplumber.page import Page

MAX_OPEN_DOCS = 8

# PDFium is not thread-safe: two threads inside it at once can crash the whole
# process. Every pypdfium2 call in the app holds this lock. Docling uses
# pypdfium2 internally with its own global lock, so share that one when present.
try:
    from docling.utils.locks import pypdfium2_lock as PDFIUM_LOCK
except ImportError:
    PDFIUM_LOCK = threading.Lock()

_registry_lock = threading.Lock()
_open: OrderedDict[str, tuple[pdfplumber.PDF, threading.Lock]] = OrderedDict()


@contextmanager
def open_page(doc_id: str, path: Path, page_no: int) -> Iterator[Page]:
    """A pdfplumber page from a small cache of open documents.

    Opening parses the page tree (~45 ms on a 400-page file); reusing an open
    document makes a page ~3x cheaper. pdfminer objects aren't thread-safe, so
    each document has its own lock - different documents still run in parallel.
    """
    with _registry_lock:
        entry = _open.get(doc_id)
        if entry is None:
            entry = (pdfplumber.open(path), threading.Lock())
            _open[doc_id] = entry
            while len(_open) > MAX_OPEN_DOCS:
                _, (old_pdf, old_lock) = _open.popitem(last=False)
                with old_lock:  # wait for any in-flight use before closing
                    old_pdf.close()
        else:
            _open.move_to_end(doc_id)
    pdf, lock = entry
    with lock:
        page = pdf.pages[page_no - 1]
        try:
            yield page
        finally:
            page.close()  # drops parsed objects so memory doesn't grow per page


def normalise(page: Page, x0: float, top: float, x1: float, bottom: float) -> list[float]:
    cx0, ctop, cx1, cbottom = page.cropbox
    w, h = cx1 - cx0, cbottom - ctop
    return [
        round(min(1.0, max(0.0, (x0 - cx0) / w)), 5),
        round(min(1.0, max(0.0, (top - ctop) / h)), 5),
        round(min(1.0, max(0.0, (x1 - cx0) / w)), 5),
        round(min(1.0, max(0.0, (bottom - ctop) / h)), 5),
    ]


def denormalise(page: Page, bbox: list[float]) -> tuple[float, float, float, float]:
    """Normalised display bbox -> pdfplumber coordinates, clamped to the page."""
    cx0, ctop, cx1, cbottom = page.cropbox
    w, h = cx1 - cx0, cbottom - ctop
    px0, ptop, px1, pbottom = page.bbox
    return (
        max(px0, cx0 + bbox[0] * w),
        max(ptop, ctop + bbox[1] * h),
        min(px1, cx0 + bbox[2] * w),
        min(pbottom, ctop + bbox[3] * h),
    )
