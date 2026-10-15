"""Fast structural region detection for the viewer overlay.

Digital PDFs already describe most of their own layout, so this reads it
directly with pdfplumber instead of running an ML model:
  - tables          -> ruled-table finder
  - raster figures  -> embedded image placements
  - vector charts   -> clusters of drawing objects (rects, lines, curves)
  - headings / text -> words grouped into blocks; headings by font size / weight

Typically 10-100 ms per page. Model-based detectors (Docling, PP-DocLayout)
plug in alongside via `detectors.py`; the overlay and templates only depend on
the Region shape returned here.
"""

import re
import statistics
import time

from pdfplumber.page import Page

from ..pdf import denormalise, normalise
from .tables import structure

Box = tuple[float, float, float, float]  # x0, top, x1, bottom in pdfplumber space

# Size filters, as a fraction of page area.
MIN_FIGURE_AREA = 0.01
MAX_FIGURE_AREA = 0.9
CLUSTER_GAP = 3.0  # drawings closer than this (pt) belong to the same figure
MIN_CHART_SHAPES = 5  # bars + axes + ticks; one or two rects is a panel or border
MAX_FIGURE_TEXT = 0.35  # figures are mostly graphics; more text than this is a text box

# Which extractor each region kind is routed to by default (see design notes).
METHODS = {
    "table": "pdfplumber",
    "chart_vector": "vector_geometry",
    "image": "vision_llm",
    "heading": "text_layer",
    "text": "text_layer",
    "scanned_page": "ocr_cortex",
}

BOLD_HINTS = ("bold", "black", "heavy", "demi", "semibold")


def _area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _overlap(a: Box, b: Box) -> float:
    """Fraction of `a` covered by `b`."""
    inter = (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))
    return _area(inter) / _area(a) if _area(a) else 0.0


def _union(a: Box, b: Box) -> Box:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _near(a: Box, b: Box, gap: float) -> bool:
    return a[0] - gap <= b[2] and b[0] - gap <= a[2] and a[1] - gap <= b[3] and b[1] - gap <= a[3]


def _box(obj: dict) -> Box:
    return (obj["x0"], obj["top"], obj["x1"], obj["bottom"])


def detect_page(
    page: Page,
    pictures: list[Box] | None = None,
    ml_regions: list[dict] | None = None,
) -> dict:
    """Regions on one page.

    pictures    visible extent of each embedded image (pdfplumber space). PDF
                images are often cropped by a clipping mask (InDesign banners);
                pdfplumber only knows the uncropped size. Defaults to that.
    ml_regions  boxes from a layout model, normalised, each {kind, bbox, ...}
                with kind table / figure / heading / text. When given, the
                model decides where figures are; the PDF still supplies tables
                (real cells) and all text. See merge rules inline.
    """
    started = time.perf_counter()
    px0, ptop, px1, pbottom = page.cropbox
    page_area = (px1 - px0) * (pbottom - ptop)
    regions: list[dict] = []
    hides_text: list[Box] = []  # tables and vector figures: words inside are their labels/cells
    model_boxes: list[Box] = []  # the subset placed by a layout model (tight, trustworthy)

    def add(kind: str, box: Box, **extra) -> None:
        regions.append(
            {
                "id": f"p{page.page_number}-{len(regions)}",
                "kind": kind,
                "bbox": normalise(page, *box),
                "method": METHODS[kind],
                **extra,
            }
        )

    words = page.extract_words(extra_attrs=["fontname"])
    for w in words:
        # Font size from line thickness. pdfminer's per-char `size` follows the
        # glyph width on sideways text, which would split words into letters.
        w["size"] = (w["bottom"] - w["top"]) if w["upright"] else (w["x1"] - w["x0"])
    char_count = sum(len(w["text"]) for w in words)
    if pictures is None:
        pictures = [_box(i) for i in page.images]
    ml = [dict(r, box=denormalise(page, r["bbox"])) for r in (ml_regions or [])]

    # Scanned page: no text layer, one big image. Routed to OCR as a whole; a
    # layout model, if run, can still say where headings and figures are.
    if char_count < 20 and any(_area(b) / page_area > 0.6 for b in pictures):
        add("scanned_page", page.cropbox, label="Scanned page - no text layer")
        for r in ml:
            kind = {"figure": "image"}.get(r["kind"], r["kind"])
            add(kind, r["box"], label=r["label"], model_label=r.get("model_label"), score=r.get("score"))
        return _result(page, regions, started, scanned=True)

    ml_tables = [r for r in ml if r["kind"] == "table"]
    # Docling reports a picture *and* a table for the same picture-of-a-table; keep the table.
    ml_figures = [
        r for r in ml
        if r["kind"] == "figure" and not any(_iou(r["box"], t["box"]) > 0.8 for t in ml_tables)
    ]

    # Tables: ruled grids from the PDF have real cells, so they win.
    try:
        found = page.find_tables()
    except Exception:  # malformed drawing data shouldn't kill the whole page
        found = []
    table_boxes: list[Box] = []
    for t in found:
        rows, cols = len(t.rows), max((len(r.cells) for r in t.rows), default=0)
        # Filled bars in a vector chart look like a one-row grid; those are charts.
        if rows < 2 or cols < 2:
            continue
        cells = t.extract()
        if not _looks_like_table(cells):
            continue
        # A grid inside a model-detected figure that the model doesn't call a
        # table is the figure's own gridlines (scatter-plot quadrants).
        if any(_overlap(t.bbox, f["box"]) > 0.7 for f in ml_figures) and not any(
            _iou(t.bbox, m["box"]) >= 0.5 for m in ml_tables
        ):
            continue
        info = structure(cells)
        tbox = _extend_to_header(t.bbox, words)
        add(
            "table",
            tbox,
            label=f"{info['title'] or 'Table'} ({rows}x{cols})",
            title=info["title"],
            header=info["columns"][:24],
            rows=rows,
            cols=cols,
        )
        table_boxes.append(tbox)
    # Tables only the model sees: pictures of tables (OCR) or borderless tables.
    for m in ml_tables:
        if any(_iou(m["box"], b) >= 0.3 for b in table_boxes):
            continue
        raster = _raster_share(m["box"], pictures) > 0.5
        if not raster:
            m["box"] = _extend_to_header(m["box"], words)
        add("table", m["box"], label="Table (picture)" if raster else "Table (borderless)",
            model_label=m.get("model_label"), score=m.get("score"),
            **({"method": "ocr"} if raster else {}))
        table_boxes.append(m["box"])
    hides_text += table_boxes

    figure_boxes: list[Box] = []
    if ml_regions is not None:
        # Figures come from the model; what is inside each box decides how it's read.
        shapes = _visible_shapes(page, page_area)
        for f in ml_figures:
            if any(_iou(f["box"], b) >= 0.5 for b in table_boxes):
                continue
            if _is_page_decoration(f["box"], page, words):
                continue
            # A picture underneath (a map, a photo backdrop) doesn't make it a
            # raster chart if the bars and labels on top are vector + text.
            drawn = sum(1 for b in shapes if _overlap(b, f["box"]) > 0.8)
            raster = _raster_share(f["box"], pictures) > 0.5 and drawn < MIN_CHART_SHAPES
            box = f["box"] if raster else _fit_to_graphics(f["box"], shapes, pictures, words)
            add("image" if raster else "chart_vector", box, label=f["label"],
                model_label=f.get("model_label"), score=f.get("score"))
            figure_boxes.append(box)
            if not raster:
                hides_text.append(box)
                model_boxes.append(box)

    # Pictures the model didn't cover (photos, logos, or no model at all). A
    # picture with lots of real text over it is a text box or background.
    for b in pictures:
        frac = _area(b) / page_area
        if (
            MIN_FIGURE_AREA <= frac <= MAX_FIGURE_AREA
            and not _is_page_decoration(b, page, words)
            and not any(_overlap(b, o) > 0.5 for o in table_boxes + figure_boxes)
            and not any(_overlap(o, b) > 0.5 for o in figure_boxes)
            and _text_coverage(b, words) <= MAX_FIGURE_TEXT
        ):
            add("image", b, label="Image / raster chart")
            figure_boxes.append(b)
            # Text on a photo is real overlay text (banner headlines): never hidden.

    # Vector charts from drawing clusters - the only figure source without a
    # model, and a fallback for isolated drawings the model missed.
    for b, shape_count in _cluster_drawings(page, page_area, exclude=table_boxes + figure_boxes):
        if shape_count < MIN_CHART_SHAPES:
            continue
        frac = _area(b) / page_area
        if not (MIN_FIGURE_AREA <= frac <= MAX_FIGURE_AREA):
            continue
        if b[2] - b[0] < 20 or b[3] - b[1] < 20:  # rules, underlines, separators
            continue
        if _is_page_decoration(b, page, words):
            continue
        if any(_overlap(b, o) > 0.5 or _overlap(o, b) > 0.5 for o in table_boxes + figure_boxes):
            continue
        if ml_regions is not None and any(_overlap(b, m["box"]) > 0 for m in ml):
            continue  # the model has an opinion about this area; trust it
        if _text_coverage(b, words) > MAX_FIGURE_TEXT:  # shaded paragraphs, text boxes
            continue
        # The shapes alone miss data labels above bars, axis ticks and legends;
        # pull in words sitting just outside the drawing so the box covers them.
        near = (b[0] - 25, b[1] - 15, b[2] + 25, b[3] + 20)
        for w in words:
            if _near(_box(w), near, 0):
                b = _union(b, _box(w))
        add("chart_vector", b, label="Vector chart / drawing")
        figure_boxes.append(b)
        hides_text.append(b)

    # Headings and text blocks. Words inside tables are cells. Words inside a
    # vector figure are its labels - unless they form prose (a question, a
    # caption sentence), which layout models often box together with the chart.
    free_words = [w for w in words if not any(_overlap(_box(w), t) > 0.5 for t in table_boxes)]
    body_size = statistics.median(w["size"] for w in words) if words else 10.0
    for block in _blocks(free_words):
        text = " ".join(" ".join(w["text"] for w in line) for line in block)
        flat = [w for line in block for w in line]
        max_size = max(w["size"] for w in flat)
        all_bold = all(any(h in w["fontname"].lower() for h in BOLD_HINTS) for w in flat)
        is_heading = len(text) <= 120 and len(block) <= 3 and (
            max_size >= body_size * 1.15 or (all_bold and not text.endswith("."))
        )
        box = flat[0]["x0"], flat[0]["top"], flat[0]["x1"], flat[0]["bottom"]
        for w in flat:
            box = _union(box, _box(w))
        holder = next((o for o in hides_text if _overlap(box, o) > 0.5), None)
        if holder is not None and not _keeps_text(box, flat, block, is_heading, holder, holder in model_boxes):
            continue  # a label belonging to the figure or table
        add("heading" if is_heading else "text", box, label=text[:160])

    return _result(page, regions, started, scanned=False)


_FIGURE = re.compile(r"^\(?[$€£₹]?\(?-?[\d,]*\d[\d.,]*%?\)?$|^[—–]$")


def _extend_to_header(box: Box, words: list[dict]) -> Box:
    """Grow a table upward over its column-header rows.

    Layout models - and ruled-table finders on HTML-converted filings - often
    start the box at the first data row, leaving the period headers ("Three
    Months Ended" / "June 27," / "2026") outside; numbers without their column
    labels are useless. Header lines sit just above, inside the table's width,
    and start right of the label column. Short section labels ("Net sales:")
    between them and the box are stepped over.
    """
    x0, top, x1, bottom = box
    width = x1 - x0
    cursor, new_top = top, top
    for _ in range(8):
        above = [
            w for w in words
            if w["bottom"] <= cursor + 1 and cursor - w["bottom"] < 14
            and w["x0"] >= x0 - 2 and w["x1"] <= x1 + 2
        ]
        if not above:
            break
        line_top = max(w["top"] for w in above)
        line = sorted((w for w in above if abs(w["top"] - line_top) <= 3), key=lambda w: w["x0"])
        cursor = min(w["top"] for w in line)
        if line[0]["x0"] >= x0 + 0.25 * width:
            new_top = cursor  # a column-header line
            continue
        text = " ".join(w["text"] for w in line)
        if text.endswith(":") and len(text) <= 40:
            continue  # section label: keep looking above it
        if any(_FIGURE.match(w["text"]) for w in line if w["x0"] >= x0 + 0.4 * width):
            new_top = cursor  # a data row the box missed (label + figures)
            continue
        break  # a title or prose in the label column: the table starts below
    return (x0, new_top, x1, bottom)


def _is_page_decoration(box: Box, page: Page, words: list[dict]) -> bool:
    """Header/footer art: touches the top or bottom page edge, stays within the
    outer 15% band and carries no text (waves, swooshes, colour bars)."""
    x0, top, x1, bottom = page.cropbox
    h = bottom - top
    edge = 0.02 * h
    in_top = box[1] - top <= edge and box[3] - top <= 0.15 * h
    in_bottom = bottom - box[3] <= edge and bottom - box[1] <= 0.15 * h
    if not (in_top or in_bottom):
        return False
    return not any(_overlap(_box(w), box) > 0.5 for w in words)


def _keeps_text(
    box: Box, flat: list[dict], block: list, is_heading: bool, holder: Box, trusted: bool
) -> bool:
    """Should a text block inside a figure stay selectable text?

    Yes for prose (questions, caption sentences), headings and lines along the
    figure's top or bottom edge (titles, "Source: ..."): models box them with the
    chart, but they are what a template anchors on. Short labels inside the
    plot (axis ticks, data labels, legend entries) belong to the figure.
    """
    if len(flat) >= 8 and len(block) >= 2:
        return True
    height = holder[3] - holder[1]
    at_edge = box[1] - holder[1] < 0.15 * height or holder[3] - box[3] < 0.15 * height
    # Bold labels inside a diagram are part of it; a heading on the figure's
    # edge is its title. Heuristic drawing clusters can over-merge (panels),
    # so inside those a heading is always kept.
    if is_heading and (at_edge or not trusted):
        return True
    return at_edge and len(flat) >= 4


def _visible_shapes(page: Page, page_area: float) -> list[Box]:
    return [
        _box(o)
        for o in (*page.rects, *page.lines, *page.curves)
        if not _invisible(o) and _area(_box(o)) < 0.8 * page_area
    ]


def _fit_to_graphics(box: Box, shapes: list[Box], pictures: list[Box], words: list[dict]) -> Box:
    """Shrink a model's figure box to the graphics actually inside it.

    Layout models box generously - chart plus its question text, title and
    source line. Keep the drawn shapes and pictures, plus labels hugging them.
    """
    inside = [b for b in shapes + pictures if _overlap(b, box) > 0.8]
    if not inside:
        return box
    fit = inside[0]
    for b in inside[1:]:
        fit = _union(fit, b)
    if _area(fit) < 0.15 * _area(box):
        return box  # too little drawing to judge by; keep the model's box
    near = (fit[0] - 25, fit[1] - 15, fit[2] + 25, fit[3] + 20)
    for w in words:
        wb = _box(w)
        if _overlap(wb, near) > 0.5 and _overlap(wb, box) > 0.5:
            fit = _union(fit, wb)
    return (max(box[0], fit[0]), max(box[1], fit[1]), min(box[2], fit[2]), min(box[3], fit[3]))


def _iou(a: Box, b: Box) -> float:
    inter = _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))
    union = _area(a) + _area(b) - inter
    return inter / union if union else 0.0


def _raster_share(box: Box, pictures: list[Box]) -> float:
    """Share of `box` covered by visible pictures (capped at 1)."""
    return min(1.0, sum(_overlap(box, p) for p in pictures))


def _text_coverage(box: Box, words: list[dict]) -> float:
    """Share of `box` covered by words lying inside it."""
    inside = sum(_area(_box(w)) for w in words if _overlap(_box(w), box) > 0.5)
    return inside / _area(box) if _area(box) else 0.0


def _invisible(obj: dict) -> bool:
    """White-filled, unstroked shapes can't be seen on a white page.

    Word draws one behind every shaded or "highlighted" line of text; counting
    them as drawing made whole paragraphs look like a chart.
    """
    if obj.get("stroke"):
        return False
    color = obj.get("non_stroking_color")
    if isinstance(color, (int, float)):
        color = (color,)
    # Patterns and shadings come through as names or objects, not numbers;
    # treat anything that isn't a plain colour as visible.
    if not color or not all(isinstance(c, (int, float)) for c in color):
        return False
    if len(color) == 4:  # CMYK: white is no ink at all
        return all(c <= 0.02 for c in color)
    return all(c >= 0.98 for c in color)  # gray or RGB


def _looks_like_table(cells: list[list[str | None]]) -> bool:
    """Reject ruled boxes that aren't tables: title panels, page borders.

    pdfplumber turns any set of rectangle edges into a grid; real tables have
    text in a good share of their cells.
    """
    flat = [c for row in cells for c in row]
    filled = sum(1 for c in flat if c and c.strip())
    return filled >= 4 and filled >= 0.25 * len(flat)


def _cluster_drawings(
    page: Page, page_area: float, exclude: list[Box]
) -> list[tuple[Box, int]]:
    """Group nearby drawing objects into (box, number of shapes) clusters."""
    shapes = [
        _box(o)
        for o in (*page.rects, *page.lines, *page.curves)
        if not _invisible(o)
        and _area(_box(o)) < 0.8 * page_area  # page backgrounds would swallow everything
        and not any(_overlap(_box(o), e) > 0.5 for e in exclude)  # table grid lines
    ]
    clusters: list[tuple[Box, int]] = []
    for s in shapes:
        merged, count = s, 1
        keep = []
        for c, n in clusters:
            if _near(c, merged, CLUSTER_GAP):
                merged, count = _union(c, merged), count + n
            else:
                keep.append((c, n))
        keep.append((merged, count))
        clusters = keep
    # One more pass: unions can bring two earlier clusters within reach of each other.
    changed = True
    while changed:
        changed = False
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if _near(clusters[i][0], clusters[j][0], CLUSTER_GAP):
                    (a, n), (b, m) = clusters[i], clusters.pop(j)
                    clusters[i] = (_union(a, b), n + m)
                    changed = True
                    break
            if changed:
                break
    return clusters


def _blocks(words: list[dict]) -> list[list[list[dict]]]:
    """Words -> lines -> blocks (paragraph-like groups), returned as lists of lines.

    Lines split at wide horizontal gaps so side-by-side columns stay separate;
    lines join a block when they're close vertically, overlap horizontally and
    share a font size (so a heading doesn't merge into the paragraph below).
    """
    # Left to right, each word joins a line it overlaps vertically and sits just
    # after; vertical overlap (not equal tops) keeps raised digits like the "13"
    # in "10/13/26" on their line.
    lines: list[list[dict]] = []
    for w in sorted(words, key=lambda w: w["x0"]):
        for line in lines:
            last = line[-1]
            overlap = min(w["bottom"], last["bottom"]) - max(w["top"], last["top"])
            if (
                overlap >= 0.5 * min(w["bottom"] - w["top"], last["bottom"] - last["top"])
                and -0.5 <= w["x0"] - last["x1"] < 1.5 * w["size"]
            ):
                line.append(w)
                break
        else:
            lines.append([w])

    def line_box(line: list[dict]) -> Box:
        return (line[0]["x0"], min(w["top"] for w in line), line[-1]["x1"], max(w["bottom"] for w in line))

    blocks: list[list[list[dict]]] = []
    for line in sorted(lines, key=lambda ln: (line_box(ln)[1], line_box(ln)[0])):
        lb = line_box(line)
        size = statistics.median(w["size"] for w in line)
        for block in blocks:
            last = block[-1]
            bb = line_box(last)
            last_size = statistics.median(w["size"] for w in last)
            if (
                0 <= lb[1] - bb[3] < 0.8 * size
                and lb[0] < bb[2]
                and bb[0] < lb[2]
                and abs(size - last_size) <= 0.15 * last_size
            ):
                block.append(line)
                break
        else:
            blocks.append([line])
    return blocks


def _result(page: Page, regions: list[dict], started: float, scanned: bool) -> dict:
    x0, top, x1, bottom = page.cropbox
    return {
        "page": page.page_number,
        "width": x1 - x0,
        "height": bottom - top,
        "scanned": scanned,
        "regions": regions,
        "ms": round((time.perf_counter() - started) * 1000, 1),
    }


def suggest_anchor(regions: list[dict], bbox: list[float]) -> str | None:
    """Nearest heading above (or overlapping the top of) a selected box.

    This turns "a box at these coordinates" into "the thing under heading X",
    which is what lets a template find the same content in next month's report.
    """
    x0, y0, x1, _ = bbox
    best, best_y = None, -1.0
    for r in regions:
        if r["kind"] == "table" and r.get("title") and _contains(r["bbox"], bbox):
            return r["title"]  # a table's own title row is the strongest anchor
        if r["kind"] != "heading":
            continue
        rx0, _, rx1, ry1 = r["bbox"]
        horizontally_related = rx0 < x1 and rx1 > x0 - 0.05
        if ry1 <= y0 + 0.01 and horizontally_related and ry1 > best_y:
            best, best_y = r["label"], ry1
    return best


def _contains(outer: list[float], inner: list[float], slack: float = 0.02) -> bool:
    return (
        inner[0] >= outer[0] - slack
        and inner[1] >= outer[1] - slack
        and inner[2] <= outer[2] + slack
        and inner[3] <= outer[3] + slack
    )
