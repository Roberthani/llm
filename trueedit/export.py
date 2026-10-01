"""Export edited documents (PDF / PNG / JPG / project) and fidelity measurement."""
from __future__ import annotations

import io
import json
import zipfile

import cv2
import numpy as np

from . import storage as S
from .editing import render_page_edits
from .render import composite

PAPER_PT = {"letter": (612.0, 792.0), "a4": (595.28, 841.89), "legal": (612.0, 1008.0), "a5": (419.53, 595.28)}


def page_edits(doc: dict, page: int) -> list[dict]:
    return [e for e in doc.get("edits", []) if int(e.get("page", 0)) == page and not e.get("disabled")]


def edited_page(pid: str, page: int, doc: dict | None = None):
    """Returns (canvas, edited, patches)."""
    doc = doc if doc is not None else S.edits(pid)
    pdir = S.page_dir(pid, page)
    canvas = S.load_image(pdir / "canvas.png")
    edits = page_edits(doc, page)
    if not edits:
        return canvas, canvas, []
    A = S.read_json(pdir / "analysis.json", {})
    A = merge_manual_regions(A, doc, page)
    patches = render_page_edits(canvas, A, edits, cache_ns=f"{pid}:{page}")
    return canvas, composite(canvas, patches), patches


def merge_manual_regions(A: dict, doc: dict, page: int) -> dict:
    regs = [r for r in doc.get("regions", []) if int(r.get("page", -1)) == page]
    if not regs:
        return A
    A = dict(A)
    A["regions"] = list(A.get("regions", [])) + regs
    return A


def fidelity(pid: str, page: int, doc: dict | None = None) -> dict:
    canvas, edited, patches = edited_page(pid, page, doc)
    diff = np.any(canvas != edited, axis=2)
    H, W = diff.shape
    allowed = np.zeros_like(diff)
    per = []
    for p in patches:
        boxes = [p.info.get("effective_bbox"), p.info.get("new_text_bbox")]
        sig = (p.info.get("style") or {}).get("sigma", 1.0) or 1.0
        mg = int(3 + 2 * sig)
        for b in boxes:
            if b:
                allowed[max(0, b[1] - mg):b[3] + mg, max(0, b[0] - mg):b[2] + mg] = True
        cb = p.bbox
        n = int(p.mask.sum())
        per.append({"edit_id": p.edit_id, "changed_pixels": n, "changed_bbox": cb,
                    "moved": p.info.get("moved", []), "warnings": p.info.get("warnings", [])})
    # words moved by reflow: their old and new positions are part of the intended change
    A = S.read_json(S.page_dir(pid, page) / "analysis.json", {})
    words = {w["id"]: w for r in A.get("regions", []) for w in r["words"]}
    for p in patches:
        # a moved word carries its own blur halo, which extends ~2 sigma beyond its ink box
        sig = (p.info.get("style") or {}).get("sigma", 1.0) or 1.0
        mg = int(3 + 2 * sig)
        for mv in p.info.get("moved", []):
            w = words.get(mv["id"])
            if w:
                b = w["bbox"]
                for dx in (0, mv["dx"]):
                    allowed[max(0, b[1] - mg):b[3] + mg, max(0, b[0] + dx - mg):b[2] + dx + mg] = True
    outside = diff & ~allowed
    ys, xs = np.nonzero(diff)
    total = int(diff.sum())
    return {
        "page": page, "width": W, "height": H, "edits": len(patches),
        "changed_pixels": total, "changed_fraction": round(total / diff.size, 6),
        "changed_bbox": [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)] if total else None,
        "outside_intended_pixels": int(outside.sum()),
        "untouched_identical": int(outside.sum()) == 0,
        "per_edit": per,
    }


def diff_image(pid: str, page: int, max_side: int = 1600) -> bytes:
    canvas, edited, _ = edited_page(pid, page)
    d = np.any(canvas != edited, axis=2)
    base = cv2.cvtColor(cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    base = (base.astype(np.float32) * 0.35 + 255 * 0.65).astype(np.uint8)
    base[d] = (40, 40, 230)
    # make tiny changes visible when downscaled
    dd = cv2.dilate(d.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    ring = dd & ~d
    base[ring] = (120, 120, 255)
    s = min(1.0, max_side / max(base.shape[:2]))
    if s < 1:
        base = cv2.resize(base, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return cv2.imencode(".png", base)[1].tobytes()


def _png(img) -> bytes:
    return cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 6])[1].tobytes()


def _jpg(img, q=95) -> bytes:
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()


def page_size_pt(pm: dict) -> tuple[float, float, str]:
    """Physical size for an image page: detected paper > embedded DPI > 200 DPI assumption."""
    W, H = pm["width"], pm["height"]
    if pm.get("pdf_size_pt"):
        return pm["pdf_size_pt"][0], pm["pdf_size_pt"][1], "pdf"
    if pm.get("paper") in PAPER_PT:
        w, h = PAPER_PT[pm["paper"]]
        return (w, h, "paper:" + pm["paper"]) if H >= W else (h, w, "paper:" + pm["paper"])
    dpi = pm.get("dpi") or 0
    if dpi < 150 or pm.get("steps") and any(s.get("step") == "perspective_correction" and "skipped" not in s
                                              for s in pm["steps"]):
        dpi = 200.0  # camera files carry a meaningless 72 dpi; a rectified photo has no true dpi either
    return W * 72.0 / dpi, H * 72.0 / dpi, f"dpi:{dpi:g}"


def export_pdf(pid: str, quality: str = "lossless") -> bytes:
    import pymupdf

    m = S.meta(pid)
    doc = S.edits(pid)
    if m["kind"] == "pdf" and all(not p.get("is_photo") for p in m["pages"]):
        return _export_pdf_overlay(pid, m, doc)
    out = pymupdf.open()
    A_all = []
    for pm in m["pages"]:
        i = pm["index"]
        canvas, edited, patches = edited_page(pid, i, doc)
        w, h, _ = page_size_pt(pm)
        page = out.new_page(width=w, height=h)
        data = _png(edited) if quality == "lossless" else _jpg(edited, 92)
        page.insert_image(page.rect, stream=data)
        _text_layer(page, pid, i, doc, patches, edited.shape[1], edited.shape[0])
    out.set_metadata({"producer": "TrueEdit OCR", "creator": "TrueEdit OCR"})
    return out.tobytes(garbage=3, deflate=True)


def _text_layer(page, pid, i, doc, patches, W, H):
    """Invisible, searchable text matching the visible page (edited words use the new text)."""
    A = S.read_json(S.page_dir(pid, i) / "analysis.json", {})
    sx, sy = page.rect.width / W, page.rect.height / H
    replaced = set()
    edits = page_edits(doc, i)
    for e, p in zip(edits, patches):
        replaced.update(e.get("target_ids", []))
        if e.get("text", "").strip() and p.info.get("new_text_bbox"):
            _invisible(page, e["text"], p.info["new_text_bbox"], sx, sy)
    for r in A.get("regions", []):
        if r["id"] in replaced:
            continue
        for w in r["words"]:
            if w["id"] in replaced:
                continue
            _invisible(page, w["text"], w["bbox"], sx, sy)


def _invisible(page, text, b, sx, sy):
    import pymupdf

    try:
        fs = max(2.0, (b[3] - b[1]) * sy * 0.9)
        page.insert_text(pymupdf.Point(b[0] * sx, b[3] * sy - (b[3] - b[1]) * sy * 0.15), text, fontsize=fs,
                         render_mode=3)
    except Exception:
        pass


def _export_pdf_overlay(pid: str, m: dict, doc: dict) -> bytes:
    """Vector PDF input: keep the original PDF and only patch edited areas."""
    import pymupdf

    src = pymupdf.open(stream=(S.pdir(pid) / m["source_file"]).read_bytes(), filetype="pdf")
    for pm in m["pages"]:
        i = pm["index"]
        edits = page_edits(doc, i)
        if not edits:
            continue
        canvas, edited, patches = edited_page(pid, i, doc)
        page = src[i]
        W, H = canvas.shape[1], canvas.shape[0]
        sx, sy = page.rect.width / W, page.rect.height / H
        A = S.read_json(S.page_dir(pid, i) / "analysis.json", {})
        words = {w["id"]: w for r in A.get("regions", []) for w in r["words"]}
        lines = {r["id"]: r for r in A.get("regions", [])}
        dm = page.derotation_matrix
        # 1) remove the original text objects under edited/moved words (so old text is not left behind)
        for e, p in zip(edits, patches):
            ids = list(e.get("target_ids", [])) + [mv["id"] for mv in p.info.get("moved", [])]
            boxes = []
            for t in ids:
                if t in words:
                    boxes.append(words[t]["bbox"])
                elif t in lines:
                    boxes += [w["bbox"] for w in lines[t]["words"]]
            if not boxes and e.get("bbox"):
                boxes = [e["bbox"]]
            for b in boxes:
                # shrink a little so neighbouring glyphs are not caught
                r = pymupdf.Rect((b[0] + 2) * sx, (b[1] + 3) * sy, (b[2] - 2) * sx, (b[3] - 3) * sy)
                if r.is_empty:
                    continue
                page.add_redact_annot(r * dm)
        page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE, graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
                              text=pymupdf.PDF_REDACT_TEXT_REMOVE)
        # 2) overlay only the changed pixels (transparent elsewhere: vector content stays vector)
        for p in patches:
            cb = p.bbox
            if not cb:
                continue
            x0, y0, x1, y1 = cb
            rgb = edited[y0:y1, x0:x1]
            mask = np.any(canvas[y0:y1, x0:x1] != rgb, axis=2)
            mask = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            rgba = np.dstack([rgb, (mask * 255).astype(np.uint8)])
            png = cv2.imencode(".png", rgba)[1].tobytes()
            rect = pymupdf.Rect(x0 * sx, y0 * sy, x1 * sx, y1 * sy)
            page.insert_image(rect * dm, stream=png, rotate=-page.rotation if page.rotation else 0)
            for e in edits:
                if e.get("id") == p.edit_id and e.get("text", "").strip() and p.info.get("new_text_bbox"):
                    _invisible(page, e["text"], p.info["new_text_bbox"], sx, sy)
            # words that were moved along the line keep their (now invisible) text at the new spot
            for mv in p.info.get("moved", []):
                w = words.get(mv["id"])
                if w:
                    b = w["bbox"]
                    _invisible(page, w["text"], [b[0] + mv["dx"], b[1], b[2] + mv["dx"], b[3]], sx, sy)
    return src.tobytes(garbage=3, deflate=True)


def export_images(pid: str, fmt: str = "png") -> tuple[bytes, str, str]:
    m = S.meta(pid)
    doc = S.edits(pid)
    files = []
    for pm in m["pages"]:
        _, edited, _ = edited_page(pid, pm["index"], doc)
        files.append(_png(edited) if fmt == "png" else _jpg(edited, 95))
    ext = "png" if fmt == "png" else "jpg"
    if len(files) == 1:
        return files[0], f"image/{'png' if ext == 'png' else 'jpeg'}", ext
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for i, f in enumerate(files):
            z.writestr(f"page-{i + 1}.{ext}", f)
    return buf.getvalue(), "application/zip", "zip"


def export_project(pid: str) -> bytes:
    m = S.meta(pid)
    d = S.pdir(pid)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("trueedit.json", json.dumps({"format": "trueedit-project", "version": 1, "meta": m,
                                                "edits": S.edits(pid)}))
        z.write(d / m["source_file"], m["source_file"])
        for pm in m["pages"]:
            pd = d / "pages" / str(pm["index"])
            for f in ("canvas.png", "analysis.json", "ocr_input.jpg", "original.png"):
                if (pd / f).exists():
                    z.write(pd / f, f"pages/{pm['index']}/{f}")
    return buf.getvalue()


def import_project(data: bytes) -> str:
    from .ingest import InputError

    try:
        z = zipfile.ZipFile(io.BytesIO(data))
        man = json.loads(z.read("trueedit.json"))
    except Exception as e:
        raise InputError("Not a valid TrueEdit project file.", "invalid_project") from e
    if man.get("format") != "trueedit-project":
        raise InputError("Not a valid TrueEdit project file.", "invalid_project")
    pid = S.new_id()
    d = S.create(pid)
    m = man["meta"]
    m["id"] = pid
    m["imported"] = True
    allowed = {m["source_file"]} | {f"pages/{p['index']}/{f}" for p in m.get("pages", [])
                                    for f in ("canvas.png", "analysis.json", "ocr_input.jpg", "original.png")}
    for p in m.get("pages", []):
        (d / "pages" / str(p["index"])).mkdir(parents=True, exist_ok=True)
    for name in z.namelist():
        if name in allowed:  # never extract arbitrary paths
            target = d / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(name))
    S.write_json(d / "meta.json", m)
    S.write_json(d / "edits.json", man.get("edits", {"version": 0, "edits": []}))
    return pid
