# TrueEdit OCR

Edit text in a photographed or scanned document — and change nothing else.

Upload a photo or PDF → TrueEdit flattens and reads the page → click any text → type the
replacement → compare → export. Only the pixels of the text you edit change; logos, barcodes,
tables, borders and every untouched word stay bit-identical to the original.

## Quick start

Requires Python 3.10+ (Linux or macOS). Tesseract is optional (secondary OCR engine).

```bash
git clone https://github.com/Roberthani/llm
cd llm
git checkout claude/tender-hopper-hzl8uf
scripts/setup.sh      # one-time: virtualenv, dependencies, OCR models (~5 min)
scripts/run.sh        # then open http://127.0.0.1:8000   (HOST / PORT env vars to change)
```

Try it with the sample photos in `samples/` (a retail order and an invoice, photographed with
perspective, folds and shadows).

## Testing

```bash
scripts/setup.sh --dev                         # adds pytest + Playwright Chromium
.venv/bin/python scripts/smoke_test.py         # browser end-to-end check against a running scripts/run.sh
.venv/bin/python -m pytest                     # full suite (~15 min on 4 CPU cores)
.venv/bin/python -m pytest -m "not e2e"        # API / pipeline only
.venv/bin/python scripts/eval_ocr.py photo --overlay /tmp/ocr.jpg   # OCR accuracy vs ground truth
```

The smoke test uploads a sample, waits for OCR, checks text is editable and logo/barcode are
locked, edits one field and verifies no other pixel changed, exercises undo/redo, exports PDF
and PNG and reopens them. Test artefacts (diff images, overlays, exports, metrics) land in
`test-results/`.

## Deploy (get a shareable link)

The repo ships a production `Dockerfile` (Ubuntu 24.04, Python 3.12, Tesseract, OCR models
baked in; ~1.1 GB image) and a `railway.json`. Memory: ~0.5–0.7 GB steady, ~1 GB peak per
analysis — give it **2 GB RAM**. Set `TRUEEDIT_PASSWORD` so the link isn't open to anyone
(browser asks for it once; any username). `/api/health` stays public for health checks.

**Railway (recommended, ~5 minutes):**
1. railway.com → sign in with GitHub → **New Project → Deploy from GitHub repo** → `Roberthani/llm`.
2. Service → **Settings → Source**: branch `claude/tender-hopper-hzl8uf` (if not the default).
3. Service → **Variables**: add `TRUEEDIT_PASSWORD` = a password of your choice.
4. Service → **Settings → Networking → Generate Domain** → that `*.up.railway.app` URL is your link.

Railway reads `railway.json` (Dockerfile build, `/api/health` health check) and sets `PORT`.

**Any other Docker host** (Fly.io, Render, a VM):
```bash
docker build -t trueedit .
docker run -d -p 8000:8000 -e TRUEEDIT_PASSWORD=change-me --memory=2g trueedit
```
Uploaded documents live on the container's disk under `/data/projects` and are deleted after
`TRUEEDIT_RETENTION_DAYS` (2 by default in the image); mount a volume there to keep them.
Check a deployment end-to-end with `TRUEEDIT_PASSWORD=… .venv/bin/python scripts/smoke_test.py https://your-url`.

## Workflow

| Step | What happens |
|---|---|
| Upload | JPG, PNG, HEIC, WEBP, TIFF, BMP, PDF (multi-page, vector or scanned). Type is checked by content, not extension. |
| Analyze | Runs in a background worker with live progress. Original file and decoded original are never modified. |
| Review | Uncertain readings are listed instead of guessed: low confidence, unclear characters, engine disagreement, ambiguous numbers, overlapping regions, uncertain table cells. |
| Edit | Click a word (double-click/tap for the whole line) or draw a region. Replace, delete, restyle, nudge, resize/move the area, find & replace, undo/redo. |
| Compare | Side by side, swipe, overlay/blink, and a changed-pixel map with the exact count of pixels changed and how many lie outside the edited areas. |
| Export | PDF (original page size, lossless or compact, searchable text layer), PNG, JPG, or a `.trueedit` project file to keep editing later. |

## How it works

```
upload ─► ingest (EXIF, PDF raster @≤300 dpi) ─► preprocess ─► analyse ─► edit (patches) ─► export
                                                 │                │
               original.png (untouched) ◄────────┤                ├─ PP-OCRv5 det (2 scales) + rec
               canvas.png  = warp(original)  ◄───┤                ├─ PP-OCRv4 + Tesseract 2nd opinion
               ocr_input   = enhanced copy   ◄───┘                ├─ layout: rulings, tables, cells,
                             (OCR only)                           │   barcode, logo/graphics, whitespace
                                                                  └─ words, fonts, weight, colour, align
```

* **Preprocessing** (`trueedit/preprocess.py`): page boundary detection with sub-pixel corner
  fitting → perspective correction (snapped to Letter/A4/Legal when the aspect matches) →
  0/90/180/270° orientation from text-line geometry + the PP-OCR direction classifier → deskew.
  The editable canvas is only geometrically resampled. Shadow/illumination flattening, contrast
  normalisation, denoise and sharpening are applied to a *separate* OCR copy.
* **OCR** (`trueedit/ocr/`): PaddleOCR PP-OCRv5 detection at two scales (small text and isolated
  glyphs) and recognition with per-character confidence and position (from CTC time steps).
  Lines that are low-confidence, contain look-alike characters in numbers, or have an
  unexplained gap are re-read by PP-OCRv4 and Tesseract; agreement confirms, two engines
  against one corrects-and-flags, numeric context breaks ties, anything else is flagged.
  Word boxes come from ink segmentation aligned with recognised characters (tight boxes,
  missing spaces recovered).
* **Editing** (`trueedit/render.py`, `trueedit/editing.py`): the original text is re-rendered in
  candidate fonts over its own pixels; a least-squares model (darkness = ink contrast ×
  blurred glyph coverage) picks family, weight, size, width, blur and ink colour. The edit then
  removes only that word's ink (hysteresis mask, neighbours and rulings protected), inpaints
  the paper locally, matches sensor noise, and draws the replacement on the original baseline
  with the same alignment. If the new text is longer, the following words of the line are
  *moved* pixel-for-pixel (not re-drawn); if there is no room it condenses ≤12 % / shrinks
  ≤15 % and warns. Each edit returns a patch + exact changed-pixel mask; everything else is
  untouched by construction.
* **Export** (`trueedit/export.py`): image inputs → PDF with the edited page at its physical size
  (detected paper size, else DPI) plus an invisible OCR text layer. Vector PDF inputs → the
  *original PDF* is kept; old text under edits is redacted (removed, not just covered) and only
  the changed pixels are overlaid, so untouched content stays vector and selectable.

## API (summary)

`POST /api/projects` (multipart `file`) · `GET /api/projects/{id}` · `GET …/status` ·
`POST …/analyze {mode: full|fast|manual}` · `GET …/pages/{n}/analysis` ·
`GET …/pages/{n}/image/{canvas|original|ocr}` · `GET/PUT …/edits` ·
`POST …/pages/{n}/render {edits, regions}` · `POST …/pages/{n}/ocr-region {bbox}` ·
`GET …/pages/{n}/fidelity` · `GET …/pages/{n}/diff.png` · `POST …/export {format: pdf|png|jpg|project}` ·
`POST /api/projects/import` · `GET /api/health`.

Errors are JSON `{error: {code, message, hint}}`; analysis failures carry `recovery` actions
(`retry`, `retry_fast`, `manual`).

## Configuration

| Variable | Default | |
|---|---|---|
| `TRUEEDIT_DATA` | `./data/projects` | project storage |
| `TRUEEDIT_TIME_LIMIT` | `240` | analysis time limit per page (s) |
| `TRUEEDIT_WORKERS` | `1` | concurrent analysis jobs |
| `TRUEEDIT_OCR_THREADS` | CPU count | ONNX Runtime threads |
| `TRUEEDIT_RETENTION_DAYS` | `7` | delete projects untouched for this long (on start-up) |
| `TRUEEDIT_PASSWORD` | — | require this password (HTTP Basic, any username) |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | listen address (the Docker image uses `0.0.0.0`) |
| `TRUEEDIT_FAULT` | — | test-only fault injection: `ocr_crash`, `ocr_unavailable`, `ocr_slow` |

## Known limitations

* Validated on synthetic-but-realistic photographed documents (rendered, then perspective,
  fold, shadow, blur, noise and JPEG applied) with exact ground truth; not yet on a corpus of
  real phone photos.
* No true dewarping of curved/crumpled pages: folds are tolerated (line-local editing), but
  strongly curved text lines are not straightened.
* Replacement glyphs come from bundled fonts (Liberation / DejaVu, metric-compatible with
  Arial / Times / Courier). Documents set in other typefaces get the closest match; weight is
  ambiguous on very blurry thin text.
* Handwriting, vertical text, and non-Latin editing are not supported (PP-OCRv5 can read many
  scripts, but the replacement fonts are Latin-focused).
* Tables need visible rulings (full grid or horizontal rules); whitespace-only tables are read
  line by line.
* Single-process server with in-memory job status and one shared password (no per-user accounts);
  run a single instance.

Model weights: PaddleOCR PP-OCRv5/v4 (Apache-2.0) via the `onnxocr` and `rapidocr-onnxruntime`
PyPI packages. Fonts: Liberation (SIL OFL 1.1) and DejaVu (Bitstream Vera licence), see
`trueedit/fonts/`.
