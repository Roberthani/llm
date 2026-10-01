"""TrueEdit OCR — HTTP API + static web app."""
from __future__ import annotations

import base64
import os
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import export as X
from . import jobs
from . import storage as S
from .editing import EditError, render_page_edits
from .ingest import MAX_UPLOAD_BYTES, InputError, load_document

WEB = Path(__file__).resolve().parents[1] / "web"
VERSION = "1.0.0"

app = FastAPI(title="TrueEdit OCR", version=VERSION)


@app.on_event("startup")
def _startup():
    # drop abandoned projects (default: untouched for 7 days)
    S.cleanup(float(os.environ.get("TRUEEDIT_RETENTION_DAYS", "7")) * 86400)


def err(status: int, code: str, message: str, hint: str | None = None, **extra):
    return JSONResponse({"error": {"code": code, "message": message, "hint": hint, **extra}}, status_code=status)


@app.exception_handler(S.NotFound)
async def _nf(_req, exc):
    return err(404, "not_found", "Project or page not found.")


@app.exception_handler(InputError)
async def _ie(_req, exc: InputError):
    return err(400, exc.code, str(exc), exc.hint)


@app.exception_handler(EditError)
async def _ee(_req, exc: EditError):
    return err(422, "invalid_edit", str(exc))


@app.exception_handler(Exception)
async def _any(_req, exc: Exception):
    import traceback

    traceback.print_exc()
    return err(500, "server_error", "Something went wrong on the server.", "Try again. Your edits are saved.")


# ------------------------------------------------------------------ health

@app.get("/api/health")
def health():
    from .ocr.secondary import tesseract_available

    ocr = {"primary": "PP-OCRv5 (ONNX Runtime)", "available": False}
    try:
        from .ocr.ppocr import load_models

        mods = load_models()
        ocr.update(available=True, secondary=[n for n, v in (("PP-OCRv4", mods.get("rec4")),) if v] +
                   (["Tesseract 5"] if tesseract_available() else []))
    except Exception as e:
        ocr["error"] = str(e)
    return {"ok": True, "version": VERSION, "ocr": ocr, "max_upload_mb": MAX_UPLOAD_BYTES // 1_000_000}


# ------------------------------------------------------------------ projects

def _public_meta(pid: str) -> dict:
    m = S.meta(pid)
    st = jobs.status(pid)
    return {**{k: v for k, v in m.items() if k not in ("source_file",)}, "job": st}


@app.post("/api/projects", status_code=201)
async def upload(file: UploadFile = File(...)):
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    name = os.path.basename(file.filename or "document")
    # validate early so the user gets an immediate, specific error
    doc = load_document(data, name)
    pid = S.new_id()
    d = S.create(pid)
    ext = {"application/pdf": "pdf", "image/jpeg": "jpg", "image/png": "png", "image/heic": "heic",
           "image/webp": "webp", "image/tiff": "tif", "image/bmp": "bmp", "image/avif": "avif"}[doc.mime]
    src = f"source.{ext}"
    (d / src).write_bytes(data)
    S.write_json(d / "meta.json", {
        "id": pid, "filename": name, "mime": doc.mime, "kind": doc.kind, "source_file": src, "bytes": len(data),
        "page_count": len(doc.pages), "pages": [], "status": "queued", "created": time.time(),
        "warnings": doc.warnings})
    S.write_json(d / "edits.json", {"version": 0, "edits": [], "review": {}, "regions": []})
    jobs.submit(pid, "full")
    return _public_meta(pid)


@app.post("/api/projects/import", status_code=201)
async def import_project(file: UploadFile = File(...)):
    data = await file.read(MAX_UPLOAD_BYTES * 2)
    pid = X.import_project(data)
    return _public_meta(pid)


@app.get("/api/projects/{pid}")
def get_project(pid: str):
    return _public_meta(pid)


@app.get("/api/projects/{pid}/status")
def get_status(pid: str):
    S.pdir(pid)
    return jobs.status(pid)


class AnalyzeReq(BaseModel):
    mode: str = "full"


@app.post("/api/projects/{pid}/analyze")
def reanalyze(pid: str, req: AnalyzeReq):
    if req.mode not in ("full", "fast", "manual"):
        raise HTTPException(400, "mode must be full|fast|manual")
    S.pdir(pid)
    jobs.submit(pid, req.mode)
    return jobs.status(pid)


@app.delete("/api/projects/{pid}")
def delete_project(pid: str):
    S.pdir(pid)
    S.delete(pid)
    return {"deleted": pid}


@app.get("/api/projects/{pid}/pages/{n}/analysis")
def page_analysis(pid: str, n: int):
    a = S.read_json(S.page_dir(pid, n) / "analysis.json")
    if a is None:
        raise S.NotFound(pid)
    return a


@app.get("/api/projects/{pid}/pages/{n}/image/{variant}")
def page_image(pid: str, n: int, variant: str, request: Request, fmt: str = "jpg", max_side: int = 0):
    files = {"canvas": "canvas.png", "original": "original.png", "ocr": "ocr_input.jpg"}
    if variant not in files:
        raise HTTPException(404)
    p = S.page_dir(pid, n) / files[variant]
    if not p.exists():
        raise S.NotFound(pid)
    if fmt == "png" and variant != "ocr" and not max_side:
        return FileResponse(p, media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})
    img = S.load_image(p)
    if max_side and max(img.shape[:2]) > max_side:
        s = max_side / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    if fmt == "png":
        data, mt = cv2.imencode(".png", img)[1].tobytes(), "image/png"
    else:
        data, mt = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes(), "image/jpeg"
    return Response(data, media_type=mt, headers={"Cache-Control": "private, max-age=3600"})


# ------------------------------------------------------------------ editing

class EditsDoc(BaseModel):
    edits: list[dict] = []
    review: dict = {}
    regions: list[dict] = []


@app.get("/api/projects/{pid}/edits")
def get_edits(pid: str):
    return S.edits(pid)


@app.put("/api/projects/{pid}/edits")
def put_edits(pid: str, doc: EditsDoc):
    return S.save_edits(pid, doc.model_dump())


class RenderReq(BaseModel):
    edits: list[dict]
    regions: list[dict] = []


@app.post("/api/projects/{pid}/pages/{n}/render")
def render(pid: str, n: int, req: RenderReq):
    pdir = S.page_dir(pid, n)
    canvas = S.load_image(pdir / "canvas.png")
    A = S.read_json(pdir / "analysis.json", {})
    A = X.merge_manual_regions(A, {"regions": req.regions}, n)
    t = time.time()
    patches = render_page_edits(canvas, A, req.edits, cache_ns=f"{pid}:{n}")
    out = []
    for p in patches:
        cb = p.bbox
        item = {"edit_id": p.edit_id, "changed_bbox": cb, "info": _clean(p.info)}
        if cb:
            x0, y0, x1, y1 = cb
            lx, ly = x0 - p.x, y0 - p.y
            rgb = p.rgb[ly:ly + (y1 - y0), lx:lx + (x1 - x0)]
            m = p.mask[ly:ly + (y1 - y0), lx:lx + (x1 - x0)]
            rgba = np.dstack([rgb, (m * 255).astype(np.uint8)])
            item["png"] = base64.b64encode(cv2.imencode(".png", rgba)[1].tobytes()).decode()
            item["x"], item["y"], item["w"], item["h"] = x0, y0, x1 - x0, y1 - y0
        out.append(item)
    return {"patches": out, "ms": int((time.time() - t) * 1000)}


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


class RegionReq(BaseModel):
    bbox: list[float]


@app.post("/api/projects/{pid}/pages/{n}/ocr-region")
def ocr_region(pid: str, n: int, req: RegionReq):
    """Recognise text inside a user-drawn region (e.g. a word the automatic pass missed)."""
    from .ocr.ppocr import crop_quad, load_models
    from .ocr.secondary import tesseract_line

    pdir = S.page_dir(pid, n)
    canvas = S.load_image(pdir / "canvas.png")
    H, W = canvas.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in req.bbox]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 - x0 < 3 or y1 - y0 < 3:
        raise EditError("region too small")
    p = 6
    crop = canvas[max(0, y0 - p):min(H, y1 + p), max(0, x0 - p):min(W, x1 + p)]
    from .preprocess import flatten_illumination

    crop_e = flatten_illumination(crop, max(8, (y1 - y0) * 0.8))
    try:
        mods = load_models()
    except Exception as e:
        return {"text": "", "conf": 0.0, "engine": None, "message": f"OCR unavailable: {e}. Type the text manually."}
    boxes = mods["det"](crop_e, limit_side=1280)
    boxes.sort(key=lambda b: (round(b.quad[:, 1].mean() / max(1, (y1 - y0) / 3)), b.quad[:, 0].min()))
    crops = [crop_quad(crop_e, b.quad)[0] for b in boxes] or [crop_e]
    recs = mods["rec"](crops)
    import unicodedata

    text = " ".join(unicodedata.normalize("NFKC", r.text).strip() for r in recs if r.text.strip())
    conf = float(np.mean([r.conf for r in recs if r.text.strip()])) if any(r.text.strip() for r in recs) else 0.0
    tess = tesseract_line(crop_e)
    alts = []
    if tess and tess[0] and tess[0].replace(" ", "") != text.replace(" ", ""):
        alts.append({"engine": "tesseract", "text": tess[0], "conf": round(tess[1], 3)})
    return {"text": text, "conf": round(conf, 4), "engine": "ppocr_v5", "alternatives": alts,
            "needs_review": conf < 0.85 or bool(alts)}


@app.get("/api/projects/{pid}/pages/{n}/fidelity")
def page_fidelity(pid: str, n: int):
    return X.fidelity(pid, n)


@app.get("/api/projects/{pid}/pages/{n}/diff.png")
def page_diff(pid: str, n: int, v: str = ""):
    return Response(X.diff_image(pid, n), media_type="image/png", headers={"Cache-Control": "no-store"})


@app.get("/api/projects/{pid}/pages/{n}/edited")
def page_edited(pid: str, n: int, fmt: str = "jpg", max_side: int = 0, v: str = ""):
    _, edited, _ = X.edited_page(pid, n)
    img = edited
    if max_side and max(img.shape[:2]) > max_side:
        s = max_side / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    if fmt == "png":
        return Response(cv2.imencode(".png", img)[1].tobytes(), media_type="image/png")
    return Response(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes(), media_type="image/jpeg")


# ------------------------------------------------------------------ export

class ExportReq(BaseModel):
    format: str = "pdf"
    quality: str = "lossless"


@app.post("/api/projects/{pid}/export")
def export(pid: str, req: ExportReq):
    m = S.meta(pid)
    if m.get("status") != "ready":
        return err(409, "not_ready", "The document is still being analysed.")
    base = os.path.splitext(m.get("filename") or "document")[0] or "document"
    if req.format == "pdf":
        data, mt, ext = X.export_pdf(pid, req.quality), "application/pdf", "pdf"
    elif req.format in ("png", "jpg"):
        data, mt, ext = X.export_images(pid, req.format)
    elif req.format == "project":
        data, mt, ext = X.export_project(pid), "application/zip", "trueedit"
    else:
        return err(400, "bad_format", "format must be pdf, png, jpg or project")
    fname = f"{base}-edited.{ext}" if ext != "trueedit" else f"{base}.trueedit"
    return Response(data, media_type=mt, headers={"Content-Disposition": f'attachment; filename="{fname}"',
                                                   "X-Export-Bytes": str(len(data))})


# ------------------------------------------------------------------ static web app

@app.get("/")
def index():
    return FileResponse(WEB / "index.html", headers={"Cache-Control": "no-cache"})


app.mount("/static", StaticFiles(directory=WEB), name="static")


def main():
    import uvicorn

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("trueedit.app:app", host=host, port=port, workers=1)


if __name__ == "__main__":
    main()
