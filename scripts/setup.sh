#!/usr/bin/env bash
# One-time setup: virtualenv, Python deps, OCR models, test fixtures.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v tesseract >/dev/null || echo "NOTE: install the 'tesseract-ocr' system package for the secondary OCR engine (optional but recommended)."
python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements-dev.txt
.venv/bin/pip install -q --no-deps -r requirements-models.txt
.venv/bin/python -c "from trueedit.ocr.ppocr import load_models; m = load_models(); print('OCR models OK (secondary v4:', m['rec4'] is not None, ')')"
.venv/bin/python tests/fixtures/generate.py
echo "Setup complete. Run:  scripts/run.sh   then open http://127.0.0.1:8000"
