"""Document layout analysis: rulings/borders, tables & cells, barcodes, logos/graphics,
whitespace. All coordinates are canvas pixels, boxes are [x0, y0, x1, y1] (x1/y1 exclusive).
"""
from __future__ import annotations

import cv2
import numpy as np


def ink_mask(ocr_input: np.ndarray) -> np.ndarray:
    """Binary mask (uint8 0/255) of dark marks on the illumination-flattened page."""
    gray = cv2.cvtColor(ocr_input, cv2.COLOR_BGR2GRAY) if ocr_input.ndim == 3 else ocr_input
    t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = min(200.0, max(110.0, t))
    return np.where(gray < thr, 255, 0).astype(np.uint8)


def faint_ink_mask(ocr_input: np.ndarray) -> np.ndarray:
    """More sensitive mark mask (light grey hairlines), used only for finding rulings."""
    gray = cv2.cvtColor(ocr_input, cv2.COLOR_BGR2GRAY) if ocr_input.ndim == 3 else ocr_input
    paper = float(np.percentile(gray, 90))
    return np.where(gray < min(235.0, paper - 28.0), 255, 0).astype(np.uint8)


def detect_rulings(ink: np.ndarray, min_len: int | None = None):
    """Long horizontal / vertical strokes (table rules, borders, underlines)."""
    h, w = ink.shape
    L = min_len or max(40, min(h, w) // 25)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, L))
    hm = cv2.morphologyEx(ink, cv2.MORPH_OPEN, hk)
    vm = cv2.morphologyEx(ink, cv2.MORPH_OPEN, vk)
    # bridge tiny gaps from noise / JPEG
    hm = cv2.morphologyEx(hm, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1)))
    vm = cv2.morphologyEx(vm, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9)))
    segs = []
    for kind, m in (("h", hm), ("v", vm)):
        n, _, st, _ = cv2.connectedComponentsWithStats(m)
        for i in range(1, n):
            x, y, bw, bh, a = st[i]
            if (kind == "h" and bw >= L) or (kind == "v" and bh >= L):
                segs.append({"kind": kind, "box": [int(x), int(y), int(x + bw), int(y + bh)],
                             "thickness": int(bh if kind == "h" else bw)})
    return hm, vm, segs


def detect_tables(hm: np.ndarray, vm: np.ndarray, min_cell: int = 12):
    grid = cv2.bitwise_or(hm, vm)
    grid_d = cv2.dilate(grid, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(grid_d)
    tables = []
    for i in range(1, n):
        x, y, bw, bh, _ = st[i]
        if bw < 80 or bh < 40:
            continue
        sub_h = hm[y:y + bh, x:x + bw]
        sub_v = vm[y:y + bh, x:x + bw]
        # count distinct rules
        rows = _count_runs(sub_h.max(1) > 0)
        cols = _count_runs(sub_v.max(0) > 0)
        if rows < 2 or cols < 2:
            continue
        comp = (lab[y:y + bh, x:x + bw] == i).astype(np.uint8)
        inv = np.where(comp > 0, 0, 255).astype(np.uint8)
        cn, clab, cst, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
        cells = []
        for j in range(1, cn):
            cx, cy, cw, ch, ca = cst[j]
            if cw < min_cell or ch < min_cell:
                continue
            # cells touching the outer bbox edge are outside the table (unless the box is open)
            if cx == 0 or cy == 0 or cx + cw >= bw or cy + ch >= bh:
                continue
            if ca < 0.7 * cw * ch:
                continue
            cells.append([int(x + cx), int(y + cy), int(x + cx + cw), int(y + cy + ch)])
        if len(cells) < 2:
            # a single box (frame) — keep as border, not a table
            continue
        cells = _index_cells(cells)
        nrows = 1 + max(c["row"] for c in cells)
        ncols = 1 + max(c["col"] for c in cells)
        kind = "table" if (nrows >= 2 and ncols >= 2) else "frame"
        tables.append({"box": [int(x), int(y), int(x + bw), int(y + bh)], "rows": nrows, "cols": ncols,
                       "cells": cells, "kind": kind})
    for k, t in enumerate(tables):
        t["id"] = f"t{k}"
    return tables


def _count_runs(v: np.ndarray) -> int:
    v = v.astype(np.int8)
    return int(np.sum(np.diff(np.concatenate([[0], v])) == 1))


def _cluster(vals, tol):
    vals = sorted(vals)
    centers = []
    for v in vals:
        if centers and abs(v - centers[-1][-1]) <= tol:
            centers[-1].append(v)
        else:
            centers.append([v])
    return [float(np.mean(c)) for c in centers]


def _index_cells(cells):
    ys = _cluster([c[1] for c in cells], 6)
    xs = _cluster([c[0] for c in cells], 6)
    out = []
    for c in cells:
        r = int(np.argmin([abs(c[1] - y) for y in ys]))
        k = int(np.argmin([abs(c[0] - x) for x in xs]))
        out.append({"box": c, "row": r, "col": k})
    out.sort(key=lambda c: (c["row"], c["col"]))
    return out


def detect_barcodes(gray: np.ndarray, ink: np.ndarray):
    """1-D barcodes: dense runs of parallel vertical bars."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    g = cv2.convertScaleAbs(np.abs(gx) - np.abs(gy))
    g = cv2.blur(g, (9, 9))
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 9)))
    th = cv2.erode(th, None, iterations=3)
    th = cv2.dilate(th, None, iterations=3)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w < 60 or h < 20:
            continue
        sub = ink[y:y + h, x:x + w] > 0
        if sub.size == 0:
            continue
        # scanlines at 25/50/75% height must all cross many bars with consistent pattern
        rows = [sub[int(h * f)] for f in (0.25, 0.5, 0.75)]
        trans = [int(np.sum(np.abs(np.diff(r.astype(np.int8))))) for r in rows]
        if min(trans) < 30:
            continue
        # bars are vertical: column profile consistent from top to bottom
        top, bot = sub[: h // 3].mean(0), sub[-h // 3:].mean(0)
        if np.corrcoef(top, bot)[0, 1] < 0.8:
            continue
        # grow to the full bar extent using the ink mask columns
        cols = np.where(sub.mean(0) > 0.6)[0]
        if len(cols) == 0:
            continue
        x0, x1 = x + int(cols.min()), x + int(cols.max()) + 1
        colmask = ink[:, x0:x1] > 0
        prof = colmask.mean(1)
        y0, y1 = y, y + h
        while y0 > 0 and prof[y0 - 1] > 0.3:
            y0 -= 1
        while y1 < ink.shape[0] and prof[y1] > 0.3:
            y1 += 1
        out.append({"kind": "barcode", "box": [x0 - 2, y0 - 2, x1 + 2, y1 + 2], "bars": int(np.median(trans) // 2)})
    return out


def detect_graphics(canvas: np.ndarray, ink: np.ndarray, rule_mask: np.ndarray, text_h: float, barcodes,
                    word_boxes=None):
    """Non-text graphics (logos, emblems, photos, stamps): large dense components."""
    m = cv2.subtract(ink, rule_mask)
    # colour content counts as graphic too (logos are often coloured fills)
    hsv = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV)
    sat = ((hsv[..., 1] > 80) & (hsv[..., 2] < 235)).astype(np.uint8) * 255
    m = cv2.bitwise_or(m, sat)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, _, st, _ = cv2.connectedComponentsWithStats(m)
    out = []
    big = 2.2 * text_h
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if w < big or h < big:
            continue
        fill = a / float(w * h)
        if fill < 0.3:
            continue  # outlines / frames, handled as borders
        box = [int(x), int(y), int(x + w), int(y + h)]
        if any(_overlap(box, b["box"]) > 0.5 for b in barcodes):
            continue
        if word_boxes and any(_overlap(box, w) > 0.6 for w in word_boxes):
            continue  # a large glyph (e.g. a heading initial), not a picture
        out.append({"kind": "graphic", "box": box, "fill": round(fill, 3)})
    # drop graphics nested inside other graphics (parts of the same emblem)
    out.sort(key=lambda g: -(g["box"][2] - g["box"][0]) * (g["box"][3] - g["box"][1]))
    kept = []
    for g in out:
        if not any(_overlap(g["box"], k["box"]) > 0.8 for k in kept):
            kept.append(g)
    return kept


def _overlap(a, b) -> float:
    """Intersection area / area of a."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    aa = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    return ix * iy / aa


def whitespace_bands(ink: np.ndarray, text_h: float, rule_mask: np.ndarray):
    m = cv2.subtract(ink, rule_mask)
    rows = (m > 0).mean(1) > 0.0015
    out, start = [], None
    for y, r in enumerate(np.append(rows, True)):
        if not r and start is None:
            start = y
        elif r and start is not None:
            if y - start > 3 * text_h:
                out.append({"kind": "whitespace", "box": [0, int(start), int(ink.shape[1]), int(y)]})
            start = None
    return out


def detect_rule_tables(segs, word_boxes, existing, em: float):
    """Tables drawn with horizontal rules only: rows between rules, columns from text alignment."""
    hs = sorted([sg["box"] for sg in segs if sg["kind"] == "h"], key=lambda b: b[1])
    tables = []
    used = set()
    for i, a in enumerate(hs):
        if i in used:
            continue
        group = [a]
        for j in range(i + 1, len(hs)):
            b = hs[j]
            if abs(b[0] - a[0]) < 2 * em and abs(b[2] - a[2]) < 2 * em and b[1] - group[-1][3] < 4 * em:
                group.append(b)
                used.add(j)
        if len(group) < 3:
            continue
        box = [min(g[0] for g in group), group[0][1], max(g[2] for g in group), group[-1][3]]
        if any(_overlap(box, t["box"]) > 0.3 for t in existing):
            continue
        # header row: text just above the first rule belongs to the table too
        above = [w for w in word_boxes if w[3] <= box[1] + 2 and box[1] - w[3] < 1.6 * em and w[0] >= box[0] - em
                 and w[2] <= box[2] + em]
        top = min([w[1] for w in above], default=box[1]) - 4
        ys = [top] + [(g[1] + g[3]) // 2 for g in group]
        inside = [w for w in word_boxes if w[1] >= top - 2 and w[3] <= box[3] + 2 and w[0] >= box[0] - em and w[2] <= box[2] + em]
        if len(inside) < 4:
            continue
        # columns: x ranges covered by text, split at empty gaps >= 1 em
        cover = np.zeros(box[2] - box[0] + 2, bool)
        for w in inside:
            cover[max(0, w[0] - box[0]):max(0, w[2] - box[0])] = True
        cols, x = [], 0
        while x < len(cover):
            if cover[x]:
                x0 = x
                while x < len(cover) and (cover[x] or (x + int(em) < len(cover) and cover[x:x + int(em)].any())):
                    x += 1
                cols.append([box[0] + x0, box[0] + x])
            x += 1
        if len(cols) < 2:
            continue
        # cell borders halfway between columns
        xb = [box[0]] + [(cols[k][1] + cols[k + 1][0]) // 2 for k in range(len(cols) - 1)] + [box[2]]
        cells = []
        for r in range(len(ys) - 1):
            for c in range(len(xb) - 1):
                cells.append({"box": [int(xb[c]), int(ys[r]), int(xb[c + 1]), int(ys[r + 1])], "row": r, "col": c})
        tables.append({"box": [int(box[0]), int(top), int(box[2]), int(box[3])], "rows": len(ys) - 1, "cols": len(cols),
                       "cells": cells, "kind": "table", "ruling": "horizontal"})
    return tables
