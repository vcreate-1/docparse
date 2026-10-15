"""Generate a monthly report series that is deliberately hard to anchor in.

Month to month, the things a template must find:
  - segment revenue table : moves page and position, heading reworded, a new
                            segment row appears mid-year, period labels change
  - collections chart     : raster chart, moves page, title reworded
  - outlook text          : section moves; wording of the body changes
Plus decoys that fool coordinate- or header-only matching:
  - "Segment Costs" table with *identical* columns, sometimes before the target
  - a second chart with a similar title in some months

Writes the PDFs and manifest.json with the true page + bbox of every target in
every file, which is the anchoring answer key.

    python scripts/make_report_series.py --out samples/series --months 12
"""

import argparse
import json
import random
from pathlib import Path

import pymupdf

W, H = 595, 842
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
REVENUE_HEADINGS = ["Segment Revenue", "Revenue by Segment", "Segment-wise Revenue", "Revenue by Business Segment"]
CHART_TITLES = ["Collections trend", "Collections efficiency trend", "Trend in collections"]
BODY = (
    "Performance this month reflected seasonal demand and the continued shift towards "
    "commercial accounts. Figures are unaudited and in INR crore."
)


def heading(page, y, text, size=16):
    page.insert_text((50, y), text, fontsize=size, fontname="helv")


def para(page, y, text=BODY, height=60):
    page.insert_textbox(pymupdf.Rect(50, y, W - 50, y + height), text, fontsize=10, fontname="helv")


def ruled_table(page, top, rows, col_w=(150, 80, 80, 80, 80)):
    row_h = 20
    x0 = 50
    for r, row in enumerate(rows):
        x = x0
        for c, cell in enumerate(row):
            page.insert_text((x + 4, top + r * row_h + 14), str(cell), fontsize=9,
                             fontname="hebo" if r == 0 else "helv")
            x += col_w[c]
    width, height = sum(col_w), row_h * len(rows)
    shape = page.new_shape()
    for r in range(len(rows) + 1):
        shape.draw_line((x0, top + r * row_h), (x0 + width, top + r * row_h))
    x = x0
    for w in (0, *col_w):
        x += w
        shape.draw_line((x, top), (x, top + height))
    shape.finish(color=(0, 0, 0), width=0.6)
    shape.commit()
    return pymupdf.Rect(x0, top, x0 + width, top + height)


def raster_chart(page, rect, seed):
    rng = random.Random(seed)
    tmp = pymupdf.open()
    p = tmp.new_page(width=400, height=220)
    s = p.new_shape()
    s.draw_polyline([(30 + i * 30, 180 - rng.uniform(20, 150)) for i in range(12)])
    s.finish(color=(0.85, 0.3, 0.2), width=2)
    s.draw_line((30, 190), (380, 190))
    s.draw_line((30, 190), (30, 20))
    s.finish(color=(0, 0, 0), width=1)
    s.commit()
    page.insert_image(rect, stream=p.get_pixmap(dpi=110).tobytes("png"))
    return rect


def norm(rect, page) -> list[float]:
    return [round(rect.x0 / W, 4), round(rect.y0 / H, 4), round(rect.x1 / W, 4), round(rect.y1 / H, 4)]


def table_rows(segments, values, month, label_period):
    rows = [["Segment", label_period, "QTD", "YTD", "YoY %"]]
    for seg in segments:
        v = values[seg]
        rows.append([seg, f"{v:,.1f}", f"{v * 2.9:,.1f}", f"{v * month:,.1f}", f"{(v % 17) - 5:.1f}"])
    total = sum(values[s] for s in segments)
    rows.append(["Total", f"{total:,.1f}", f"{total * 2.9:,.1f}", f"{total * month:,.1f}", ""])
    return rows


def make(path: Path, year: int, month: int) -> dict:
    rng = random.Random(year * 100 + month)
    doc = pymupdf.open()
    truth = {}
    segments = ["Retail", "Commercial", "Industrial", "Agriculture", "Public sector"]
    if month >= 7:
        segments.append("Exports")  # a new row mid-year
    revenue = {s: round(rng.uniform(200, 900), 1) for s in segments}
    costs = {s: round(v * rng.uniform(0.5, 0.8), 1) for s, v in revenue.items()}
    period = f"{MONTHS[month - 1]} {year}"

    cover = doc.new_page(width=W, height=H)
    heading(cover, 300, f"Monthly Performance Report - {MONTHS[month - 1]} {year}", 22)

    for i in range(1 + (month % 4)):  # 1-4 pages of commentary: everything after drifts
        p = doc.new_page(width=W, height=H)
        heading(p, 70, f"Management commentary ({i + 1})")
        for k in range(rng.randint(4, 7)):
            para(p, 100 + k * 100)

    # Decoy first in odd months: same columns, different meaning.
    def costs_page():
        p = doc.new_page(width=W, height=H)
        heading(p, 70, "Segment Costs")
        para(p, 85, "Operating costs by business segment for the month, quarter to date and year to date.", 30)
        ruled_table(p, 130, table_rows(segments, costs, month, period))

    if month % 2:
        costs_page()

    p = doc.new_page(width=W, height=H)
    top = 70 + rng.choice([0, 0, 60, 140])  # vertical drift on the page
    if top > 70:
        para(p, 70, height=top - 80)
    heading(p, top, REVENUE_HEADINGS[month % len(REVENUE_HEADINGS)])
    para(p, top + 15, "Revenue by business segment for the month, quarter to date and year to date.", 30)
    rect = ruled_table(p, top + 60, table_rows(segments, revenue, month, period))
    truth["segment_revenue"] = {"page": doc.page_count, "bbox": norm(rect, p), "kind": "table"}

    if not month % 2:
        costs_page()

    p = doc.new_page(width=W, height=H)
    if month % 3 == 0:  # a similar-looking decoy chart above the real one
        heading(p, 60, "Collections by region")
        raster_chart(p, pymupdf.Rect(60, 80, 520, 300), year * 1000 + month)
        y = 340
    else:
        y = 70
    heading(p, y, CHART_TITLES[month % len(CHART_TITLES)])
    para(p, y + 15, "Daily collections efficiency over the last twelve months.", 30)
    rect = raster_chart(p, pymupdf.Rect(60, y + 60, 520, y + 280), year * 100 + month)
    truth["collections_chart"] = {"page": doc.page_count, "bbox": norm(rect, p), "kind": "figure"}

    p = doc.new_page(width=W, height=H)
    heading(p, 70, "Outlook")
    rect = pymupdf.Rect(50, 85, W - 50, 85 + 120)
    p.insert_textbox(rect, f"For {MONTHS[(month) % 12]} we expect demand to remain "
                     f"{rng.choice(['firm', 'steady', 'resilient'])} across segments. " + BODY, fontsize=10, fontname="helv")
    truth["outlook"] = {"page": doc.page_count, "bbox": norm(pymupdf.Rect(50, 85, W - 50, 140), p), "kind": "text"}

    doc.save(path, garbage=3, deflate=True)
    return truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="samples/series")
    ap.add_argument("--months", type=int, default=12)
    ap.add_argument("--year", type=int, default=2025)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for m in range(1, args.months + 1):
        path = out / f"report_{args.year}_{m:02d}.pdf"
        manifest[path.name] = make(path, args.year, m)
        print(path, {k: v["page"] for k, v in manifest[path.name].items()})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
