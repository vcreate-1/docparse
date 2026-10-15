"""Overlay boxes must land on the same pixels the viewer draws.

Renders tricky pages with pdfium (which, like PDF.js, applies /Rotate and the
cropbox) and checks the detected text box covers the rendered ink.
"""

import pdfplumber
import pymupdf  # dev-only: used to build the test PDF
import pypdfium2 as pdfium
import pytest
from PIL import ImageOps

from app.services.detect import detect_page


def _make(path, rotation: int, cropbox: tuple | None, upright: bool):
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=800)
    page.set_rotation(rotation)
    # upright=True is the common real case: a landscape page whose content is
    # drawn rotated so it reads normally. upright=False shows text sideways.
    page.insert_text((200, 300), "Quarterly Heading", fontsize=28, rotate=rotation if upright else 0)
    if cropbox:
        page.set_cropbox(pymupdf.Rect(*cropbox))
    doc.save(path)


def _ink_bbox(path) -> list[float]:
    img = pdfium.PdfDocument(path)[0].render(scale=2).to_pil().convert("L")
    x0, y0, x1, y1 = ImageOps.invert(img).point(lambda v: 255 if v > 60 else 0).getbbox()
    return [x0 / img.width, y0 / img.height, x1 / img.width, y1 / img.height]


CASES = [
    (rot, crop, upright)
    for rot in (0, 90, 180, 270)
    for crop in (None, (50, 100, 550, 700))
    for upright in (True, False)
    if not (rot == 0 and not upright)
]


@pytest.mark.parametrize("rotation,cropbox,upright", CASES)
def test_text_box_matches_rendered_ink(tmp_path, rotation, cropbox, upright):
    path = tmp_path / "t.pdf"
    _make(path, rotation, cropbox, upright)
    with pdfplumber.open(path) as pdf:
        regions = detect_page(pdf.pages[0])["regions"]
    assert len(regions) == 1, regions  # one block, not one box per word or letter
    box, ink = regions[0]["bbox"], _ink_bbox(path)
    # Text boxes include the font's ascent/descent, so they're a bit larger
    # than the ink; they must contain it and stay close to it.
    tol = 0.03
    assert box[0] - tol <= ink[0] and box[1] - tol <= ink[1], (box, ink)
    assert box[2] + tol >= ink[2] and box[3] + tol >= ink[3], (box, ink)
    assert all(abs(a - b) < 0.06 for a, b in zip(box, ink)), (box, ink)
    if upright:
        assert regions[0]["label"] == "Quarterly Heading"


def test_white_shading_behind_paragraphs_is_not_a_chart(tmp_path):
    # Word draws an invisible white rectangle behind every shaded text line
    # (seen in the Parker Bay report); 25 of them used to read as a "chart"
    # and hide the paragraphs.
    path = tmp_path / "shaded.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=800)
    shape = page.new_shape()
    for i in range(20):
        y = 100 + i * 14
        shape.draw_rect(pymupdf.Rect(50, y, 550, y + 14))
        shape.finish(fill=(1, 1, 1), color=None)
    shape.commit()
    for i in range(20):
        page.insert_text((55, 111 + i * 14), "Mining truck shipments declined by more than five percent.", fontsize=10)
    doc.save(path)
    with pdfplumber.open(path) as pdf:
        kinds = {r["kind"] for r in detect_page(pdf.pages[0])["regions"]}
    assert "chart_vector" not in kinds
    assert "text" in kinds


def test_invisible_shape_colours():
    from app.services.detect import _invisible

    assert _invisible({"stroke": False, "non_stroking_color": 1.0})  # gray white
    assert _invisible({"stroke": False, "non_stroking_color": (1, 1, 1)})  # RGB white
    assert _invisible({"stroke": False, "non_stroking_color": (0, 0, 0, 0)})  # CMYK no ink
    assert not _invisible({"stroke": False, "non_stroking_color": (0.25, 0.45, 0.8)})  # a blue bar
    assert not _invisible({"stroke": True, "non_stroking_color": 1.0})  # white box with a border
    assert not _invisible({"stroke": False, "non_stroking_color": ("P1",)})  # pattern fill (Infographics PDF)
