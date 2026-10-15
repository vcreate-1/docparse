from app.services.tables import clean_cell, structure

# Shape taken from the KBB EV sales report, page 2.
KBB_ROWS = [
    ["ELECTRIC VEHICLE (BRAND)", "", "", "", "", "", "", "", ""],
    ["", "Q3 Sales", "", "", "YTD Sales", "", "", "Segment Share", ""],
    ["", "2026", "2025", "YOY", "2026", "2025", "YOY", "Q3", "YTD"],
    ["Audi", "1,039", "18,071", "-94.3%", "2,736", "29,630", "-90.8%", "0.4%", "0.4%"],
    ["Total", "262,732", "434,392", "-39.5%", "7 26,100", "1,041,863", "-30.3%", "", ""],
]


def test_multi_row_header_is_flattened():
    s = structure(KBB_ROWS)
    assert s["title"] == "ELECTRIC VEHICLE (BRAND)"
    assert s["columns"] == [
        "Label", "Q3 Sales 2026", "Q3 Sales 2025", "Q3 Sales YOY",
        "YTD Sales 2026", "YTD Sales 2025", "YTD Sales YOY",
        "Segment Share Q3", "Segment Share YTD",
    ]
    assert [r[0] for r in s["data"]] == ["Audi", "Total"]


def test_split_numbers_are_rejoined():
    assert structure(KBB_ROWS)["data"][1][4] == "726,100"
    assert clean_cell("7 26,100") == "726,100"
    assert clean_cell("Q3 Sales") == "Q3 Sales"  # text keeps its spaces


def test_headerless_continuation_table():
    s = structure([["Toyota", "83", "111"], ["Hyundai", "-", "2"]])
    assert s["title"] is None
    assert len(s["data"]) == 2


def test_layout_boxes_are_not_tables():
    from app.services.detect import _looks_like_table

    # KBB cover: the blue title panel and photo border form a ruled "grid".
    cover = [[None, "Electric Vehicle\nSales Report\nQ3 2026", None], ["", "Estimates as of 10/13/26", ""], [None, "", None]]
    assert not _looks_like_table(cover)
    assert _looks_like_table(KBB_ROWS)


def _words(lines):
    """[(top, [(x0, text), ...]), ...] -> pdfplumber-like word dicts (6pt per char)."""
    out = []
    for top, items in lines:
        for x0, text in items:
            out.append({"text": text, "x0": x0, "x1": x0 + 6 * len(text), "top": top, "bottom": top + 9})
    return out


def test_borderless_financial_statement():
    # Shape of Apple's 10-Q income statement: stacked period headers, "$" signs,
    # a section label, right-aligned figures.
    from app.services.tables import align_table

    words = _words([
        (100, [(300, "Three"), (336, "Months"), (378, "Ended"), (460, "Nine"), (490, "Months"), (532, "Ended")]),
        (112, [(300, "June"), (330, "27,"), (380, "June"), (410, "28,"), (460, "June"), (490, "27,"), (540, "June"), (570, "28,")]),
        (124, [(306, "2026"), (386, "2025"), (466, "2026"), (546, "2025")]),
        (140, [(20, "Net"), (44, "sales:")]),
        (152, [(30, "Products"), (290, "$"), (306, "78,678"), (370, "$"), (386, "66,613"), (466, "272,629"), (546, "233,287")]),
        (164, [(30, "Services"), (306, "30,739"), (386, "27,423"), (472, "91,728"), (552, "80,408")]),
    ])
    s = structure(align_table(words))
    assert s["columns"][1:] == [
        "Three Months Ended June 27, 2026", "Three Months Ended June 28, 2025",
        "Nine Months Ended June 27, 2026", "Nine Months Ended June 28, 2025",
    ]
    assert s["data"][0] == ["Net sales:", "", "", "", ""]
    assert s["data"][1] == ["Products", "78,678", "66,613", "272,629", "233,287"]


def test_caption_row_is_title_not_header():
    s = structure([
        ["follows (dollars in millions):", "", ""],
        ["", "2026", "2025"],
        ["Net sales", "100", "90"],
    ])
    assert s["title"] == "follows (dollars in millions):"
    assert s["columns"] == ["Label", "2026", "2025"]
