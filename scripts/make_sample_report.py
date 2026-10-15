"""Generate synthetic monthly reports for development and tests.

Each report contains every content type the extractor routes differently:
headings + prose, a ruled table, a vector bar chart with data labels, a raster
chart image, and a scanned (image-only) page. The segment table drifts to a
later page month by month, the way real reports do over the years.

    python scripts/make_sample_report.py --out samples --months 3
    python scripts/make_sample_report.py --out samples --months 1 --filler 400
"""

import argparse
import random
from pathlib import Path

import pymupdf

W, H = 595, 842  # A4 in points
SEGMENTS = ["Retail", "Commercial", "Industrial", "Agriculture", "Public sector"]
BODY = (
    "Performance this month reflected seasonal demand and the continued shift towards "
    "commercial accounts. Management commentary on margins, collections and capital "
    "expenditure follows in the sections below. Figures are unaudited and in INR crore."
)


def heading(page, y, text, size=16):
    page.insert_text((50, y), text, fontsize=size, fontname="helv")


def para(page, y, text=BODY, height=80):
    page.insert_textbox(pymupdf.Rect(50, y, W - 50, y + height), text, fontsize=10, fontname="helv")


def ruled_table(page, top, rows):
    col_w = [150, 80, 80, 80, 80]
    row_h = 20
    x0 = 50
    for r, row in enumerate(rows):
        y = top + r * row_h
        x = x0
        for c, cell in enumerate(row):
            page.insert_text((x + 4, y + 14), str(cell), fontsize=9,
                             fontname="hebo" if r == 0 else "helv")
            x += col_w[c]
    width, height = sum(col_w), row_h * len(rows)
    shape = page.new_shape()
    for r in range(len(rows) + 1):
        shape.draw_line((x0, top + r * row_h), (x0 + width, top + r * row_h))
    x = x0
    for w in [0, *col_w]:
        x += w
        shape.draw_line((x, top), (x, top + height))
    shape.finish(color=(0, 0, 0), width=0.6)
    shape.commit()
    return top + height


def bar_chart(page, top, values, labels):
    left, bottom, height, bar_w, gap = 90, top + 200, 180, 50, 30
    shape = page.new_shape()
    shape.draw_line((left - 10, bottom), (left + len(values) * (bar_w + gap), bottom))
    shape.draw_line((left - 10, bottom), (left - 10, bottom - height))
    shape.finish(color=(0.2, 0.2, 0.2), width=0.8)
    vmax = max(values)
    for i, (v, lab) in enumerate(zip(values, labels)):
        x = left + i * (bar_w + gap)
        bh = height * v / vmax
        shape.draw_rect(pymupdf.Rect(x, bottom - bh, x + bar_w, bottom))
        shape.finish(fill=(0.25, 0.45, 0.8), color=None)
        page.insert_text((x + 8, bottom - bh - 4), f"{v:,.1f}", fontsize=8)
        page.insert_text((x + 2, bottom + 12), lab[:10], fontsize=7)
    shape.commit()


def raster_chart(page, rect, seed):
    """A line chart drawn as pixels, embedded as an image - no text layer inside."""
    rng = random.Random(seed)
    tmp = pymupdf.open()
    p = tmp.new_page(width=400, height=220)
    s = p.new_shape()
    pts = [(30 + i * 30, 180 - rng.uniform(20, 150)) for i in range(12)]
    s.draw_polyline(pts)
    s.finish(color=(0.85, 0.3, 0.2), width=2)
    s.draw_line((30, 190), (380, 190))
    s.draw_line((30, 190), (30, 20))
    s.finish(color=(0, 0, 0), width=1)
    s.commit()
    pix = p.get_pixmap(dpi=110)
    page.insert_image(rect, stream=pix.tobytes("png"))


def scanned_page(doc, seed):
    """Render a text page to an image and insert it as a page with no text layer."""
    tmp = pymupdf.open()
    p = tmp.new_page(width=W, height=H)
    heading(p, 80, "Annexure: Signed regulatory filing")
    para(p, 100, BODY * 3, height=300)
    pix = p.get_pixmap(dpi=100)
    page = doc.new_page(width=W, height=H)
    page.insert_image(page.rect, stream=pix.tobytes("png"))


def make_report(path: Path, year: int, month: int, filler: int, seed: int):
    rng = random.Random(seed)
    doc = pymupdf.open()
    month_name = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][month - 1]

    cover = doc.new_page(width=W, height=H)
    heading(cover, 300, f"Monthly Performance Report - {month_name} {year}", 22)
    para(cover, 330, "Prepared by Finance. Internal and confidential.", 40)

    # Layout drift: one extra page of commentary every few months.
    for i in range(1 + month // 4 + filler // 2):
        p = doc.new_page(width=W, height=H)
        heading(p, 70, f"Management commentary ({i + 1})")
        for k in range(6):
            para(p, 100 + k * 110)

    values = [round(rng.uniform(200, 900), 1) for _ in SEGMENTS]
    p = doc.new_page(width=W, height=H)
    heading(p, 70, "Segment Revenue")
    para(p, 85, "Revenue by business segment for the month, quarter to date and year to date.", 30)
    rows = [["Segment", "Month", "QTD", "YTD", "YoY %"]]
    for seg, v in zip(SEGMENTS, values):
        rows.append([seg, f"{v:,.1f}", f"{v * 2.9:,.1f}", f"{v * month:,.1f}",
                     f"{rng.uniform(-8, 15):.1f}"])
    rows.append(["Total", f"{sum(values):,.1f}", f"{sum(values) * 2.9:,.1f}",
                 f"{sum(values) * month:,.1f}", ""])
    end = ruled_table(p, 130, rows)
    heading(p, end + 40, "Revenue by segment (chart)", 13)
    bar_chart(p, end + 50, values, SEGMENTS)

    p = doc.new_page(width=W, height=H)
    heading(p, 70, "Collections trend")
    para(p, 85, "Daily collections efficiency over the last twelve months.", 30)
    raster_chart(p, pymupdf.Rect(60, 130, 520, 400), seed)
    heading(p, 450, "Outlook", 13)
    para(p, 465, height=200)

    for i in range(filler - filler // 2):
        p = doc.new_page(width=W, height=H)
        heading(p, 70, f"Appendix {i + 1}: Regional detail")
        for k in range(6):
            para(p, 100 + k * 110)

    scanned_page(doc, seed)
    doc.save(path, garbage=3, deflate=True)
    return doc.page_count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="samples")
    ap.add_argument("--months", type=int, default=3)
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--filler", type=int, default=0, help="extra pages, to test big PDFs")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for m in range(1, args.months + 1):
        path = out / f"report_{args.year}_{m:02d}{'_big' if args.filler else ''}.pdf"
        n = make_report(path, args.year, m, args.filler, seed=args.year * 100 + m)
        print(f"{path}  ({n} pages)")


if __name__ == "__main__":
    main()
