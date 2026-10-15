from app.services.anchors import generalise, score_candidate, set_similarity, similarity


def test_periods_are_generalised():
    assert generalise("Jan 2025") == generalise("Feb 2026") == "<month> <year>"
    assert generalise("Q3 FY26") == generalise("Q4 FY27")


def test_distinctive_words_decide():
    # "segment" is shared by both headings; "revenue" vs "costs" must decide.
    weight = {"segment": 0.4, "revenue": 1.1, "costs": 1.1, "business": 0.4}
    template = "Revenue by Business Segment"
    assert similarity(template, "Segment Revenue", weight) > similarity(template, "Segment Costs", weight) + 0.25


def test_misread_labels_still_match():
    assert set_similarity(["Iowa", "Illinois", "Texas"], ["lowa", "llinois", "Texas"]) == 1.0


def _fp(anchor, lookalikes=()):
    return {"kind": "table", "anchor": anchor, "context": "", "header": ["segment", "qtd"],
            "labels": ["Retail", "Total"], "page_ratio": 0.5, "bbox": [0.1, 0.2, 0.9, 0.5],
            "lookalikes": list(lookalikes)}


def test_known_lookalike_is_penalised():
    cand = {"bbox": [0.1, 0.2, 0.9, 0.5]}
    decoy = {**_fp("Segment Costs"), "context": ""}
    plain, _ = score_candidate(_fp("Segment Revenue"), cand, decoy)
    guarded, parts = score_candidate(_fp("Segment Revenue", ["Segment Costs"]), cand, decoy)
    assert "lookalike" in parts and guarded < plain - 0.2
