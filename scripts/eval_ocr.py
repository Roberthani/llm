"""Measure OCR accuracy & box placement against fixture ground truth.

Ground-truth word boxes (from the browser DOM) are mapped clean -> photo (known simulation
transform) -> canvas (the app's own recorded transform), then matched to the app's word boxes.

usage: python scripts/eval_ocr.py [photo|clean|lowres|rot90|exif] [--json out.json] [--overlay out.jpg]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "fixtures"))

from generate import clean_to_photo  # noqa: E402
from trueedit.ingest import load_document  # noqa: E402
from trueedit.ocr.pipeline import analyze_page  # noqa: E402
from trueedit.ocr.ppocr import load_models  # noqa: E402
from trueedit.preprocess import prepare_page  # noqa: E402

FX = ROOT / "tests" / "fixtures" / "out"
CASES = {
    "photo": ("order_photo.jpg", "photo", None),
    "clean": ("order_clean.png", None, None),
    "lowres": ("order_lowres.jpg", "photo", 800 / 4032),
    "rot90": ("order_photo_rot90.jpg", "photo_rot90", None),
    "exif": ("order_photo_exif.jpg", "photo", None),
    "invoice": ("invoice_photo.jpg", "photo", None),
}
GT = {"invoice": ("invoice_clean.json", "invoice_photo.json")}


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def gt_to_canvas(box, mode, meta, T, shrink):
    x0, y0, x1, y1 = box
    pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float64)
    if mode in ("photo", "photo_rot90"):
        pts = clean_to_photo(pts, meta)
        if shrink:
            pts = pts * shrink
        if mode == "photo_rot90":
            h = meta["photo_size"][1]
            pts = np.stack([h - 1 - pts[:, 1], pts[:, 0]], 1)  # rotate 90 CW
    v = np.hstack([pts, np.ones((4, 1))]) @ np.array(T).T
    v = v[:, :2] / v[:, 2:3]
    return [v[:, 0].min(), v[:, 1].min(), v[:, 0].max(), v[:, 1].max()]


def run(case: str, overlay: str | None = None):
    fname, mode, shrink = CASES[case]
    gj, mj = GT.get(case, ("order_clean.json", "order_photo.json"))
    gt = json.loads((FX / gj).read_text())
    meta = json.loads((FX / mj).read_text())
    doc = load_document((FX / fname).read_bytes(), fname)
    pg = doc.pages[0]
    models = load_models()
    P = prepare_page(pg.image, models, is_photo=pg.is_photo)
    A = analyze_page(P.canvas, P.ocr_input, models)
    words = [w | {"role": r["role"], "line": r["id"]} for r in A["regions"] for w in r["words"]]
    res = []
    for g in gt["words"]:
        gb = gt_to_canvas(g["box"], mode, meta, P.transform, shrink)
        best, bi = None, 0.0
        for w in words:
            v = iou(gb, w["bbox"])
            if v > bi:
                bi, best = v, w
        # words may be merged (e.g. "N/A" next to punctuation): accept containment match
        if best is not None and bi <= 0.3:
            # tiny glyphs (e.g. a dash): accept a word whose centre lies inside the ground-truth box
            for w in words:
                cx, cy = (w["bbox"][0] + w["bbox"][2]) / 2, (w["bbox"][1] + w["bbox"][3]) / 2
                if gb[0] <= cx <= gb[2] and gb[1] <= cy <= gb[3] and w["text"] == g["text"]:
                    best, bi = w, max(bi, 0.31)
                    break
        hit = best is not None and bi > 0.3
        text_ok = hit and best["text"].strip(".,:;") == g["text"].strip(".,:;")
        res.append({"gt": g["text"], "field": g["field"], "iou": round(bi, 3),
                    "pred": best["text"] if best else None, "text_ok": bool(text_ok),
                    "pred_conf": best["conf"] if best else None, "gt_box_canvas": [round(v, 1) for v in gb],
                    "pred_box": best["bbox"] if best else None, "role": best["role"] if best else None,
                    "gt_bold": g["bold"], "pred_bold": (best["style"]["weight"] == "bold") if best else None,
                    "weight_score": best["style"].get("weight_score") if best else None})
    n = len(res)
    found = sum(r["iou"] > 0.3 for r in res)
    ok = sum(r["text_ok"] for r in res)
    ious = [r["iou"] for r in res if r["iou"] > 0.3]
    # centre error in px
    cerr = []
    for r in res:
        if r["iou"] > 0.3:
            a, b = r["gt_box_canvas"], r["pred_box"]
            cerr.append(np.hypot((a[0] + a[2] - b[0] - b[2]) / 2, (a[1] + a[3] - b[1] - b[3]) / 2))
    fields = {}
    for r in res:
        if r["field"]:
            f = fields.setdefault(r["field"], [0, 0])
            f[0] += 1
            f[1] += r["text_ok"]
    # silent errors: wrong text that is NOT flagged for review
    flagged_words = {wid for it in A["review"] for wid in it.get("word_ids", [])}
    flagged_lines = {it["region_id"] for it in A["review"]}
    silent = []
    for r in res:
        if r["iou"] > 0.3 and not r["text_ok"]:
            w = next(w for w in words if w["bbox"] == r["pred_box"])
            if w["id"] not in flagged_words and w["line"] not in flagged_lines and w["role"] == "text":
                silent.append({"gt": r["gt"], "pred": r["pred"], "conf": r["pred_conf"]})
    summary = {
        "case": case, "gt_words": n, "detected": found, "text_exact": ok,
        "word_accuracy": round(ok / n, 4), "detection_recall": round(found / n, 4),
        "mean_iou": round(float(np.mean(ious)), 3) if ious else 0, "median_center_err_px": round(float(np.median(cerr)), 2) if cerr else None,
        "p95_center_err_px": round(float(np.percentile(cerr, 95)), 2) if cerr else None,
        "review_items": len(A["review"]),
        "weight_accuracy": round(float(np.mean([r["gt_bold"] == r["pred_bold"] for r in res if r["text_ok"] and r["role"] == "text"])), 4),
        "weight_errors": [(r["gt"], r["gt_bold"], r["weight_score"]) for r in res if r["text_ok"] and r["role"] == "text" and r["gt_bold"] != r["pred_bold"]], "stats": A["stats"],
        "barcodes": len(A["barcodes"]), "graphics": [g["kind"] for g in A["graphics"]],
        "tables": [(t["kind"], t["rows"], t["cols"]) for t in A["tables"]],
        "fields_wrong": {k: v for k, v in fields.items() if v[1] < v[0]},
        "errors": [r for r in res if not r["text_ok"]],
        "silent_errors": silent,
        "steps": [s["step"] for s in P.steps],
    }
    if overlay:
        im = P.canvas.copy()
        for r in A["regions"]:
            for w in r["words"]:
                b = w["bbox"]
                col = (0, 160, 0) if w["conf"] >= 0.85 else (0, 0, 230)
                if r["role"] != "text":
                    col = (200, 120, 0)
                cv2.rectangle(im, (b[0], b[1]), (b[2], b[3]), col, 2)
        for r in res:
            b = [int(v) for v in r["gt_box_canvas"]]
            cv2.rectangle(im, (b[0], b[1]), (b[2], b[3]), (255, 0, 255), 1)
        for g in A["barcodes"] + A["graphics"]:
            b = g["box"]
            cv2.rectangle(im, (b[0], b[1]), (b[2], b[3]), (255, 0, 0), 3)
        for t in A["tables"]:
            for c in t["cells"]:
                b = c["box"]
                cv2.rectangle(im, (b[0], b[1]), (b[2], b[3]), (0, 200, 255), 1)
        cv2.imwrite(overlay, im)
    return summary, A


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("case", nargs="?", default="photo")
    ap.add_argument("--json")
    ap.add_argument("--overlay")
    a = ap.parse_args()
    s, _ = run(a.case, a.overlay)
    full = json.dumps(s, indent=1, default=str)
    if a.json:
        Path(a.json).write_text(full)
    brief = {k: v for k, v in s.items() if k not in ("errors",)}
    print(json.dumps(brief, indent=1, default=str))
    for e in s["errors"]:
        print(f"  MISS gt={e['gt']!r:30} pred={e['pred']!r:30} iou={e['iou']} conf={e['pred_conf']} role={e['role']}")
