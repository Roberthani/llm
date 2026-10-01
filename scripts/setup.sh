#!/usr/bin/env bash
# One-time setup: Python virtualenv, dependencies and OCR models.
#   scripts/setup.sh          runtime only (enough for scripts/run.sh)
#   scripts/setup.sh --dev    + test tools (pytest, Playwright browser)
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python3}
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
  echo "ERROR: Python 3.10+ is required (found: $("$PY" --version 2>&1)). Set PYTHON=/path/to/python3.11" >&2
  exit 1
fi
command -v tesseract >/dev/null || echo "NOTE: 'tesseract' not found — optional secondary OCR engine disabled. Install: sudo apt-get install tesseract-ocr  |  brew install tesseract"

"$PY" -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements.txt
.venv/bin/python -m pip install -q --no-deps -r requirements-models.txt
if [[ "${1:-}" == "--dev" ]]; then
  .venv/bin/python -m pip install -q -r requirements-dev.txt
  [[ -n "${PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD:-}" ]] || .venv/bin/python -m playwright install chromium
fi
.venv/bin/python -c "from trueedit.ocr.ppocr import load_models; m = load_models(); print('OCR models OK — PP-OCRv5 primary, PP-OCRv4 secondary:', m['rec4'] is not None)"
echo "Setup complete. Start the app with:  scripts/run.sh   then open http://127.0.0.1:8000"
