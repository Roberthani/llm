"""Test helpers: locate OCR words by their text (never by hard-coded coordinates), re-read pixels."""
from __future__ import annotations

import unicodedata

import cv2
import numpy as np


def words(A):
    return [(r, w) for r in A["regions"] for w in r["words"]]


def find_word(A, text, line_contains=None, nth=0):
    hits = [(r, w) for r, w in words(A) if w["text"] == text and (line_contains is None or line_contains in r["text"])]
    assert len(hits) > nth, f"word {text!r} not found by OCR"
    return hits[nth]


def find_line(A, text):
    hits = [r for r in A["regions"] if r["text"] == text]
    assert hits, f"line {text!r} not found by OCR"
    return hits[0]


def read_text(img, box, pad=None):
    """OCR a canvas region with the production recogniser (detect lines, then read them)."""
    from trueedit.ocr.ppocr import crop_quad, load_models
    from trueedit.preprocess import flatten_illumination

    m = load_models()
    H, W = img.shape[:2]
    if pad is None:
        pad = int(max(8, 0.6 * (box[3] - box[1])))
    # isolate the region: real pixels inside the box (+2 px), plain paper colour around it
    bx0, by0, bx1, by1 = max(0, box[0] - 2), max(0, box[1] - 2), min(W, box[2] + 2), min(H, box[3] + 2)
    inner = img[by0:by1, bx0:bx1]
    paper = np.percentile(inner.reshape(-1, 3), 90, axis=0).astype(np.uint8)
    crop = np.empty((by1 - by0 + 2 * pad, bx1 - bx0 + 2 * pad, 3), np.uint8)
    crop[:] = paper
    crop[pad:pad + inner.shape[0], pad:pad + inner.shape[1]] = inner
    crop = flatten_illumination(crop, max(8, (box[3] - box[1]) * 0.8))
    boxes = m["det"](crop, limit_side=1600)
    if not boxes:
        return ""
    boxes.sort(key=lambda b: b.quad[:, 0].min())
    recs = m["rec"]([crop_quad(crop, b.quad)[0] for b in boxes])
    return " ".join(unicodedata.normalize("NFKC", r.text).strip() for r in recs if r.text.strip())


def grow(b, m, W=None, H=None):
    out = [b[0] - m, b[1] - m, b[2] + m, b[3] + m]
    if W is not None:
        out = [max(0, out[0]), max(0, out[1]), min(W, out[2]), min(H, out[3])]
    return [int(v) for v in out]


def changed_mask(a, b):
    return np.any(a != b, axis=2)


def save_diff(path, before, after, box=None, pad=60):
    d = changed_mask(before, after)
    vis = cv2.cvtColor(cv2.cvtColor(before, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    vis = (vis * 0.4 + 255 * 0.6).astype(np.uint8)
    vis[d] = (0, 0, 255)
    if box:
        H, W = d.shape
        x0, y0, x1, y1 = grow(box, pad, W, H)
        strip = np.vstack([before[y0:y1, x0:x1], after[y0:y1, x0:x1], vis[y0:y1, x0:x1]])
        cv2.imwrite(str(path), strip)
    else:
        cv2.imwrite(str(path), vis)


# shared paths (also importable from the e2e package, where `conftest` is ambiguous)
from pathlib import Path as _P
import json as _json

FX = _P(__file__).resolve().parent / "fixtures" / "out"
RESULTS = _P(__file__).resolve().parents[1] / "test-results"


def save_result(name, data):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / name).write_text(_json.dumps(data, indent=1, default=str))
