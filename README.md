# docparse

Point-and-extract for PDFs: open a document, see what's extractable, select
tables / charts / text, and save the selections as a reusable template that a
batch run (Airflow -> Snowflake, next phase) applies to similar documents.

## Run locally

```bash
# backend (port 8000)
cd backend
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/uvicorn app.main:app --reload --port 8000

# frontend (port 5173, proxies /api to the backend)
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 and drop in a PDF.

Synthetic test reports (table, vector chart, raster chart, scanned page, with
the table drifting pages month to month):

```bash
backend/.venv/bin/python scripts/make_sample_report.py --out samples --months 3
backend/.venv/bin/python scripts/make_sample_report.py --out samples --months 1 --filler 400
```

Tests: `cd backend && .venv/bin/python -m pytest -q`

## Using it

- **Find** tab: keyword search for the page that has what you need (`/` focuses it).
- Outlined boxes are detected regions (toggle types under **Layers**). Click one,
  or Tab to it and press Enter, to preview its extraction. `Esc` clears.
- **Draw region** (`D`) to drag a box around anything; choose what to treat it as.
- **Add to template** (`⌘↵`), then name and **Save** the template. Each save is a new version.
- `+` / `-` zoom, `Esc` cancels drawing.

## How it's built

| Piece | Where | Notes |
|---|---|---|
| PDF engines | `backend/app/pdf.py` | pypdfium2 for whole-document passes at upload (fast); pdfplumber for all geometry. All MIT/Apache/BSD. A shared lock serialises PDFium, which is not thread-safe. |
| Region detection | `backend/app/services/detect.py` | Reads the PDF's own structure (tables, images, vector drawings, font sizes). ~10-90 ms/page, cached per page. |
| Detector registry | `backend/app/services/detectors.py` | `structure` (above), plus optional `docling` and `paddle` (PP-DocLayout) layout models for the compare overlay. |
| Table structuring | `backend/app/services/tables.py` | Title row, flattened multi-row headers ("Q3 Sales 2026"), repairs split numbers, rejects ruled boxes that aren't tables. |
| Extraction preview | `backend/app/services/extract.py` | Always reads the exact text layer inside the chosen box, whichever detector proposed it. |
| Page search | `backend/app/services/search.py` | Keyword + IDF over the text layer. |
| Storage | `backend/app/storage.py` | Local folder keyed by sha256, standing in for a Snowflake stage. |
| Viewer | `frontend/src/components/Viewer.tsx` | PDF.js with range loading, virtualised pages, SVG overlay, keyboard-reachable regions. |

## Detection strategy: Auto (default)

No single detector wins. Measured on a hand-labelled answer key
(`eval/ground_truth.json`: 26 pages from 8 real PDFs, 41 tables/figures):

| | tables | figures | bad boxes | text kept | speed |
|---|---|---|---|---|---|
| PDF structure (rules) | 6/8 | 18/33 | 6 | 99% | ~40 ms/page |
| Docling | 8/8 | 32/33 | 2 | 93% | ~340 ms/page |
| PP-DocLayout | 7/8 | 31/33 | 3 | 96% | ~670 ms/page |
| **Auto** | **8/8** | **33/33** | **0** | **100%** | ~1 s on graphic pages, ~20 ms otherwise |

Auto, per page ([detectors.py](backend/app/services/detectors.py) `_auto`, merge rules in
[detect.py](backend/app/services/detect.py) `detect_page`):

1. Text/table-only pages: rules only (fast, exact).
2. Graphic pages: Docling + PP-DocLayout place figures (Docling's regions; a coarse
   Docling panel is replaced by the finer PP-DocLayout chart inside it).
3. Tables always come from the PDF when it has real cells; model-only tables are
   pictures (OCR) or borderless.
4. Model figure boxes are shrunk to the graphics inside them; prose, titles and
   source lines inside stay selectable text; text on photos is never hidden.
5. What is inside each box decides how it's read (text layer, vector labels, OCR, vision).

Other fixes the answer key drove: visible extent of cropped (clip-masked) photos via
PDFium, invisible white shapes ignored, page-edge decoration dropped, scatter-plot
gridlines not read as tables, and a borderless-table reader (columns from number
alignment, stacked period headers) for SEC-style financial statements.

Scoring and regression:

```bash
cd backend
.venv-ml/bin/python ../eval/evaluate.py auto structure docling paddle --misses
.venv-ml/bin/python -m pytest      # includes eval floors when the PDFs are present
```

The ML models are optional (~2 GB) and live in their own venv:
`python3 -m venv .venv-ml && .venv-ml/bin/pip install -r requirements-dev.txt -r requirements-ml.txt`.
Without them Auto falls back to the rules. `DOCPARSE_AUTO_MODEL=docling|paddle|both` (default both).

## Reading pictures: OCR

Regions with no text layer (pictures of tables, raster charts, scanned pages)
go through OCR ([ocr.py](backend/app/services/ocr.py)): **PaddleOCR first**; if
its reading has issues and Snowflake Cortex is configured, the region is retried
with Cortex and the better reading kept. Without Cortex, issues are shown so a
person checks the region.

Issues are measured: nothing read; low confidence (mean < 85% or many lines < 60%);
rows that don't fill the columns; a "Total" row that doesn't equal the sum above it.

Readings are shaped three ways and the best kept: column alignment (real grids),
inline labels ("Grain: 5,867, 5.5%") and position-based pairing (label left, value
at the bar's end; values above bars; two-line pie labels).

Measured on `eval/ocr_truth.json` (93 label/value pairs in 5 picture regions):
**91/93 correct**; the 2 misses are character misreads ("lowa" for Iowa), which
anchoring tolerates when matching labels.

Cortex: set `DOCPARSE_CORTEX_STAGE` and `SNOWFLAKE_ACCOUNT/USER/PASSWORD/WAREHOUSE/ROLE`
and install `snowflake-connector-python`. **The Cortex path is untested** (no
credentials were available); it is off until configured and can't break previews.

```bash
.venv-ml/bin/python ../eval/evaluate_ocr.py --show
```

## Anchoring: finding a field in the next document

Saving a template stores a **fingerprint** per field ([anchors.py](backend/app/services/anchors.py)),
not coordinates: its title / heading, the caption above it, column headers with
periods generalised ("Jan 2025" = "Nov 2026"), row labels, kind, and rough position.
It also records the field's **look-alikes** in the sample (e.g. "Segment Costs" with
identical columns next to "Segment Revenue") as negatives.

Locating in another PDF: shortlist pages by rare words, detect regions on just those
pages, score each same-kind region (distinctive words weigh more than shared ones),
and return the best with its score and lead over the runner-up:

- **Matched**: confident and clearly ahead
- **Needs review**: close call; a person confirms in the UI, and the template learns
  that document's heading as an alias (new version)
- **Not found**

In the UI: Template tab → **Test on another document**. API:
`POST /api/templates/{id}/locate`, `POST /api/templates/{id}/fields/{i}/confirm`.

Measured on a deliberately hard 12-month series (`scripts/make_report_series.py`:
tables move page and position, headings reworded, a decoy table with identical
columns, rows added mid-year): templates built from **each** of the 12 months,
37 lookups each = **444/444 correct**, no silent errors; the target table leads its
decoy by ≥ 0.39. Also correct without the ML models. Real multi-document series
are the next test.

```bash
python scripts/make_report_series.py --out samples/series --months 12
cd backend && .venv-ml/bin/python ../eval/evaluate_anchors.py --template 11 --show
```

## Design

The UI was reviewed against Dieter Rams' principles, Jony Ive's stated
philosophy and Adham Dannaway's Practical UI rules. Resulting conventions:
one meaning per colour (green = saved field / exact; dashes = compare or
drawing), WCAG-checked colours in both themes, outlines-only overlay at rest,
a focus scrim on selection, confidence stated as ✓ / ≈ / ⚠ plus a sentence,
8pt spacing and a 5-step type scale, keyboard-reachable regions.

## Not wired yet

- RAG facility (upload ref id, semantic page search, prompt + JSON schema)
- Vision-model chart estimation; vector-chart value computation from geometry
- Snowflake stage storage; Cortex OCR fallback untested
- Airflow batch runs (reusing fields.locate_target), review queue across many documents
- Mobile bottom-sheet panel (the design review's suggestion; current mobile layout is usable but basic)
