"""Background analysis jobs with progress, time limits and recoverable failures."""
from __future__ import annotations

import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import storage as S
from .ingest import load_document, InputError
from .ocr.pipeline import AnalysisTimeout, analyze_page
from .ocr.ppocr import OCRUnavailable, load_models
from .preprocess import enhance_for_ocr, prepare_page

_POOL = ThreadPoolExecutor(max_workers=int(os.environ.get("TRUEEDIT_WORKERS", "1")), thread_name_prefix="ocr")
_STATUS: dict[str, dict] = {}
_SL = threading.Lock()
TIME_LIMIT = float(os.environ.get("TRUEEDIT_TIME_LIMIT", "240"))  # seconds per page

RECOVERY = {
    "timeout": [
        {"action": "retry_fast", "label": "Retry in fast mode (skip second-opinion OCR)"},
        {"action": "manual", "label": "Continue without OCR (draw regions manually)"},
    ],
    "ocr_failed": [
        {"action": "retry", "label": "Retry analysis"},
        {"action": "manual", "label": "Continue without OCR (draw regions manually)"},
    ],
    "ocr_unavailable": [
        {"action": "retry", "label": "Retry analysis"},
        {"action": "manual", "label": "Continue without OCR (draw regions manually)"},
    ],
}


def status(pid: str) -> dict:
    with _SL:
        st = _STATUS.get(pid)
        if st:
            return dict(st)
    m = S.meta(pid)
    return {"status": m.get("status", "unknown"), "progress": 1.0 if m.get("status") == "ready" else 0.0,
            "stage": m.get("stage", ""), "error": m.get("error")}


def _set(pid, **kw):
    with _SL:
        st = _STATUS.setdefault(pid, {"status": "queued", "progress": 0.0, "stage": "Queued", "error": None})
        st.update(kw)
        st["updated"] = time.time()


def submit(pid: str, mode: str = "full"):
    _set(pid, status="queued", progress=0.0, stage="Queued", error=None, mode=mode)
    m = S.meta(pid)
    m.update(status="queued", error=None)
    S.save_meta(pid, m)
    return _POOL.submit(_run, pid, mode)


def _run(pid: str, mode: str):
    try:
        _analyze(pid, mode)
    except Exception as e:  # never leave a job silently stuck
        traceback.print_exc()
        code = "ocr_unavailable" if isinstance(e, OCRUnavailable) else \
            "timeout" if isinstance(e, AnalysisTimeout) else "ocr_failed"
        msg = {"timeout": "Analysis took too long and was stopped.",
               "ocr_unavailable": f"The OCR engine is not available: {e}",
               "ocr_failed": f"Text recognition failed: {e}"}[code]
        err = {"code": code, "message": msg, "recovery": RECOVERY[code]}
        _set(pid, status="error", stage="Failed", error=err)
        try:
            m = S.meta(pid)
            m.update(status="error", error=err)
            S.save_meta(pid, m)
        except Exception:
            pass


def _analyze(pid: str, mode: str):
    m = S.meta(pid)
    d = S.pdir(pid)
    data = (d / m["source_file"]).read_bytes()
    doc = load_document(data, m.get("filename", ""))
    n = len(doc.pages)
    models = None
    if mode != "manual":
        _set(pid, status="running", stage="Starting OCR engine", progress=0.01)
        models = load_models()
    fault = os.environ.get("TRUEEDIT_FAULT", "")
    pages_meta = []
    for i, pg in enumerate(doc.pages):
        base = i / n
        span = 1.0 / n

        def prog(f, msg, i=i, base=base, span=span):
            _set(pid, status="running", progress=round(base + span * (0.15 + 0.85 * f), 4),
                 stage=f"Page {i + 1}/{n}: {msg}")

        _set(pid, status="running", progress=round(base, 4), stage=f"Page {i + 1}/{n}: Correcting image")
        pdir = S.page_dir(pid, i, create=True)
        S.save_image(pdir / "original.png", pg.image)
        if models is not None or pg.is_photo:
            P = prepare_page(pg.image, models, pg.is_photo)
        else:
            P = prepare_page(pg.image, None, pg.is_photo)
        S.save_image(pdir / "canvas.png", P.canvas)
        S.save_image(pdir / "ocr_input.jpg", P.ocr_input, 90)
        if mode == "manual":
            H, W = P.canvas.shape[:2]
            A = {"size": [W, H], "regions": [], "tables": [], "barcodes": [], "graphics": [], "rulings": [],
                 "whitespace": [], "review": [], "text_height": 24.0,
                 "warnings": ["OCR skipped — draw regions to edit text manually."],
                 "stats": {"lines": 0, "words": 0, "mean_conf": 0, "review_items": 0, "manual": True}}
        else:
            if fault == "ocr_crash":
                raise RuntimeError("simulated OCR crash (fault injection)")
            if fault == "ocr_slow":
                time.sleep(3)
                raise AnalysisTimeout("simulated timeout (fault injection)")
            deadline = time.time() + TIME_LIMIT
            A = analyze_page(P.canvas, P.ocr_input, models, page=i, progress=prog, deadline=deadline,
                             second_opinion=(mode != "fast"))
        S.write_json(pdir / "analysis.json", A)
        H, W = P.canvas.shape[:2]
        paper = next((s.get("paper") for s in P.steps if s.get("step") == "perspective_correction"), None)
        pages_meta.append({
            "index": i, "width": W, "height": H, "orig_width": pg.image.shape[1], "orig_height": pg.image.shape[0],
            "is_photo": pg.is_photo, "dpi": pg.dpi, "pdf_size_pt": pg.pdf_size_pt, "pdf_rotation": pg.pdf_rotation,
            "has_text_layer": pg.has_text_layer, "transform": np.asarray(P.transform).tolist(), "steps": P.steps,
            "paper": paper, "warnings": P.warnings + A.get("warnings", []), "stats": A.get("stats", {}),
        })
        m = S.meta(pid)
        m["pages"] = pages_meta + m.get("pages", [])[len(pages_meta):]
        S.save_meta(pid, m)
    m = S.meta(pid)
    m.update(pages=pages_meta, status="ready", error=None, analysis_mode=mode, page_count=n,
             warnings=doc.warnings)
    S.save_meta(pid, m)
    _set(pid, status="ready", progress=1.0, stage="Done", error=None)
