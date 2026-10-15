import importlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
make_sample_report = importlib.import_module("make_sample_report")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    data = tmp_path_factory.mktemp("data")
    import app.config

    app.config.settings.data_dir = data
    import app.storage

    app.storage.storage.__init__(data)  # point the shared storage at the temp dir
    from app.main import app as fastapi_app

    return TestClient(fastapi_app)


@pytest.fixture(scope="module")
def report(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("pdf") / "report.pdf"
    make_sample_report.make_report(path, 2025, 1, filler=0, seed=1)
    return path


@pytest.fixture(scope="module")
def doc_id(client, report) -> str:
    res = client.post("/api/documents", files={"file": ("report.pdf", report.read_bytes())})
    assert res.status_code == 200
    return res.json()["id"]


def test_upload_dedupes_identical_files(client, report, doc_id):
    again = client.post("/api/documents", files={"file": ("copy.pdf", report.read_bytes())})
    assert again.json()["id"] == doc_id
    assert again.json()["deduplicated"] is True


def test_rejects_non_pdf(client):
    res = client.post("/api/documents", files={"file": ("x.pdf", b"not a pdf")})
    assert res.status_code == 422


def test_file_supports_range_requests(client, doc_id):
    res = client.get(f"/api/documents/{doc_id}/file", headers={"Range": "bytes=0-99"})
    assert res.status_code == 206
    assert len(res.content) == 100


def _kinds(client, doc_id, page):
    return [r["kind"] for r in client.get(f"/api/documents/{doc_id}/pages/{page}/regions").json()["regions"]]


def test_detects_each_content_type(client, doc_id):
    # cover, commentary, table + vector chart, raster chart, scanned page
    assert "table" in _kinds(client, doc_id, 3)
    assert "chart_vector" in _kinds(client, doc_id, 3)
    assert "image" in _kinds(client, doc_id, 4)
    assert _kinds(client, doc_id, 5) == ["scanned_page"]


def test_search_finds_table_page(client, doc_id):
    hits = client.get(f"/api/documents/{doc_id}/search", params={"q": "segment revenue"}).json()
    assert hits[0]["page"] == 3


def test_extract_table_with_anchor(client, doc_id):
    regions = client.get(f"/api/documents/{doc_id}/pages/3/regions").json()["regions"]
    table = next(r for r in regions if r["kind"] == "table")
    res = client.post(
        f"/api/documents/{doc_id}/extract",
        json={"page": 3, "bbox": table["bbox"], "kind": "table"},
    ).json()
    assert res["confidence"] == "exact"
    assert res["columns"][:2] == ["Segment", "Month"]
    assert res["data"][-1][0] == "Total"
    assert res["anchor_heading"] == "Segment Revenue"


def test_vector_chart_reads_data_labels(client, doc_id):
    regions = client.get(f"/api/documents/{doc_id}/pages/3/regions").json()["regions"]
    chart = next(r for r in regions if r["kind"] == "chart_vector")
    res = client.post(
        f"/api/documents/{doc_id}/extract",
        json={"page": 3, "bbox": chart["bbox"], "kind": "chart_vector"},
    ).json()
    assert res["confidence"] == "exact"
    assert len(res["numeric_labels"]) == 5


def test_template_versioning(client, doc_id):
    body = {
        "name": "Monthly report",
        "source_doc_id": doc_id,
        "targets": [
            {"name": "segment_revenue", "kind": "table", "method": "pdfplumber",
             "page": 3, "bbox": [0.1, 0.1, 0.9, 0.3], "anchor_heading": "Segment Revenue"}
        ],
    }
    created = client.post("/api/templates", json=body).json()
    assert created["version"] == 1
    updated = client.put(f"/api/templates/{created['id']}", json=body).json()
    assert updated["version"] == 2


def test_concurrent_pdfium_use_is_serialised(report):
    # PDFium crashes the whole process if two threads enter it at once; the
    # browser routinely requests several pages in parallel.
    from concurrent.futures import ThreadPoolExecutor

    from app.routers.documents import _scan

    data = report.read_bytes()
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda _: _scan(data), range(32)))
    assert all(len(texts) == 5 for texts, _, _ in results)


def test_template_locates_field_in_another_month(client, tmp_path_factory):
    # Template from January; November moves the table, rewords its heading and
    # carries an identical-columns "Segment Costs" decoy.
    series = importlib.import_module("make_report_series")
    d = tmp_path_factory.mktemp("series")
    truth = {m: series.make(d / f"r{m}.pdf", 2025, m) for m in (1, 11)}
    ids = {m: client.post("/api/documents", files={"file": (f"r{m}.pdf", (d / f"r{m}.pdf").read_bytes())}).json()["id"]
           for m in (1, 11)}
    t = truth[1]["segment_revenue"]
    created = client.post("/api/templates", json={
        "name": "Monthly", "source_doc_id": ids[1],
        "targets": [{"name": "segment_revenue", "kind": "table", "method": "pdfplumber", "page": t["page"], "bbox": t["bbox"]}],
    }).json()
    fp = created["targets"][0]["fingerprint"]
    assert fp["anchor"] == series.REVENUE_HEADINGS[1 % 4] and "Segment Costs" in fp["lookalikes"]

    res = client.post(f"/api/templates/{created['id']}/locate", json={"doc_id": ids[11]}).json()["fields"][0]
    assert res["status"] == "matched"
    assert res["best"]["page"] == truth[11]["segment_revenue"]["page"]
    assert res["values"]["columns"][1] == "Nov 2025"  # read with the new period's header
    assert [r[0] for r in res["values"]["data"]][-2:] == ["Exports", "Total"]  # the row added mid-year

    confirmed = client.post(f"/api/templates/{created['id']}/fields/0/confirm",
                            json={"doc_id": ids[11], "page": res["best"]["page"], "bbox": res["best"]["bbox"]}).json()
    assert confirmed["version"] == 2
    assert series.REVENUE_HEADINGS[11 % 4] in confirmed["targets"][0]["fingerprint"]["anchor_aliases"]
