"""Page analysis: layout + multi-engine OCR + reconciliation + review flags."""
from __future__ import annotations

import re
import time
import unicodedata
from typing import Callable

import cv2
import numpy as np

from .. import layout as L
from .ppocr import crop_quad
from .secondary import tesseract_line

LINE_LOW_CONF = 0.85
WORD_LOW_CONF = 0.80
CHAR_UNCLEAR = 0.60
SECOND_OPINION_CONF = 0.95
CONFUSABLE = set("OoQDIlLiSsZzBGgqT|")
NUMERIC_RE = re.compile(r"^[-+(]?[$€£#]?\d[\d,.\-/:%)]*$")


class AnalysisTimeout(RuntimeError):
    pass


def _nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s)


def _bbox(pts) -> list[int]:
    pts = np.asarray(pts)
    return [int(np.floor(pts[:, 0].min())), int(np.floor(pts[:, 1].min())),
            int(np.ceil(pts[:, 0].max())), int(np.ceil(pts[:, 1].max()))]


def _clip(b, W, H):
    return [max(0, b[0]), max(0, b[1]), min(W, b[2]), min(H, b[3])]


def _tight(ink: np.ndarray, b, pad=1):
    H, W = ink.shape
    b = _clip(b, W, H)
    sub = ink[b[1]:b[3], b[0]:b[2]]
    if sub.size == 0:
        return b, False
    ys, xs = np.nonzero(sub)
    if len(xs) < 3:
        return b, False
    return _clip([b[0] + int(xs.min()) - pad, b[1] + int(ys.min()) - pad,
                  b[0] + int(xs.max()) + 1 + pad, b[1] + int(ys.max()) + 1 + pad], W, H), True


def _hex(bgr) -> str:
    b, g, r = [int(round(v)) for v in bgr]
    return f"#{r:02x}{g:02x}{b:02x}"


def _stroke_width(gray: np.ndarray, sub: np.ndarray, b) -> float | None:
    """Blur-robust stroke width: total ink 'darkness mass' divided by skeleton length."""
    g = gray[b[1]:b[3], b[0]:b[2]].astype(np.float32)
    if g.shape != sub.shape or sub.sum() < 6:
        return None
    bg = float(np.percentile(g[~sub], 75)) if (~sub).any() else 255.0
    lo = float(np.percentile(g[sub], 8))
    if bg - lo < 25:
        return None
    dark = np.clip((bg - g) / (bg - lo), 0, 1)
    mass = float(dark[cv2.dilate(sub.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0].sum())
    try:
        skel = cv2.ximgproc.thinning(sub.astype(np.uint8) * 255)
    except Exception:
        return None
    n = float(np.count_nonzero(skel))
    return mass / n if n > 3 else None


def estimate_style(canvas: np.ndarray, ink: np.ndarray, b, text: str, gray: np.ndarray | None = None) -> dict:
    sub = ink[b[1]:b[3], b[0]:b[2]] > 0
    st = {"font_size_px": None, "baseline": None, "stroke_px": None, "stroke_ratio": None,
          "color": "#000000", "weight": "regular"}
    if sub.size == 0 or sub.sum() < 4:
        return st
    cols = np.where(sub.any(0))[0]
    tops = np.array([np.argmax(sub[:, c]) for c in cols])
    bots = np.array([sub.shape[0] - 1 - np.argmax(sub[::-1, c]) for c in cols])
    baseline = float(np.median(bots)) + 1
    asc_top = float(np.percentile(tops, 5))
    has_asc = any(ch.isupper() or ch.isdigit() or ch in "bdfhklt!?/()[]{}#$%&@|'\"" for ch in text)
    if has_asc:
        size = (baseline - asc_top) / 0.72
    else:
        size = (baseline - float(np.median(tops))) / 0.53
    stroke = _stroke_width(gray, sub, b) if gray is not None else None
    if stroke is None:
        dt = cv2.distanceTransform(sub.astype(np.uint8), cv2.DIST_L2, 3)
        ridge = (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8))) & (dt > 0)
        stroke = float(2 * np.median(dt[ridge]) - 0.5) if ridge.any() else 1.0
    core = sub
    if stroke >= 3:
        core = cv2.erode(sub.astype(np.uint8), np.ones((2, 2), np.uint8)) > 0
        if core.sum() < 4:
            core = sub
    px = canvas[b[1]:b[3], b[0]:b[2]][core]
    # darkest half of the core pixels = the true ink colour (edges are blended with paper)
    lum = px.astype(np.float32) @ np.array([0.114, 0.587, 0.299], np.float32)
    dark = px[lum <= np.median(lum)]
    color = np.median(dark if len(dark) else px, axis=0)
    st.update({"font_size_px": round(max(4.0, size), 2), "baseline": round(b[1] + baseline, 2),
               "stroke_px": round(stroke, 2), "stroke_ratio": round(stroke / max(4.0, size), 4),
               "color": _hex(color)})
    return st


def _is_numeric_like(t: str) -> bool:
    t = t.strip()
    if not t:
        return False
    digits = sum(ch.isdigit() for ch in t)
    return digits >= max(1, 0.5 * len(t.replace(" ", "")))


def _choose(cands: list[dict]) -> tuple[dict, list[dict], str]:
    """Reconcile engine outputs. Returns (chosen, alternatives, verdict)."""
    prim = cands[0]
    others = [c for c in cands[1:] if c["text"] is not None]
    norm = lambda s: re.sub(r"\s+", "", s)
    agree = [c for c in others if norm(c["text"]) == norm(prim["text"])]
    alts = [c for c in others if norm(c["text"]) != norm(prim["text"]) and c["text"].strip()]
    if agree:
        return prim, alts, "confirmed"
    if len(others) >= 2 and norm(others[0]["text"]) == norm(others[1]["text"]) and others[0]["text"].strip():
        # both secondary engines agree against the primary
        ch = dict(others[0])
        ch["conf"] = float(min(others[0]["conf"], others[1]["conf"]))
        return ch, [prim], "corrected"
    # contextual validation: numeric-looking candidates should parse as numbers
    if _is_numeric_like(prim["text"]):
        valid = [c for c in [prim] + others if NUMERIC_RE.match(c["text"].replace(" ", ""))]
        if valid and not NUMERIC_RE.match(prim["text"].replace(" ", "")):
            best = max(valid, key=lambda c: c["conf"])
            return best, [c for c in [prim] + others if c is not best], "numeric_context"
    return prim, alts, "disagreement" if alts else "single"


def analyze_page(canvas: np.ndarray, ocr_input: np.ndarray, models, page: int = 0,
                 progress: Callable[[float, str], None] | None = None, deadline: float | None = None,
                 second_opinion: bool = True) -> dict:
    t_start = time.time()
    prog = progress or (lambda f, m: None)

    def check_time():
        if deadline and time.time() > deadline:
            raise AnalysisTimeout("analysis exceeded its time limit")

    H, W = canvas.shape[:2]
    prog(0.02, "Finding layout")
    ink_all = L.ink_mask(ocr_input)
    hm, vm, segs = L.detect_rulings(ink_all)
    rule_mask = cv2.bitwise_or(hm, vm)
    rule_mask_d = cv2.dilate(rule_mask, np.ones((3, 3), np.uint8))
    ink = cv2.subtract(ink_all, rule_mask_d)
    gray = cv2.cvtColor(ocr_input, cv2.COLOR_BGR2GRAY)
    barcodes = L.detect_barcodes(gray, ink_all)
    tables = L.detect_tables(hm, vm)
    check_time()

    prog(0.08, "Detecting text")
    det = models["det"]
    boxes = _multiscale(det, ocr_input)
    scale = 1.0
    hts = [min(np.linalg.norm(b.quad[0] - b.quad[3]), np.linalg.norm(b.quad[1] - b.quad[2])) for b in boxes]
    med_h = float(np.median(hts)) if hts else 0.0
    det_img = ocr_input
    if boxes and med_h < 22 and max(H, W) * 2 <= 6000:
        # low-resolution page: upsample so small glyphs survive detection/recognition
        scale = 2.0
        det_img = cv2.resize(ocr_input, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        boxes = _multiscale(det, det_img, int(2400 * scale))
    check_time()

    # drop detections that are barcode bars
    kept, dropped = [], 0
    for b in boxes:
        bb = _bbox(b.quad / scale)
        if any(L._overlap(bb, bc["box"]) > 0.4 for bc in barcodes):
            dropped += 1
            continue
        kept.append(b)
    boxes = kept

    prog(0.25, f"Reading {len(boxes)} text lines")
    crops, Ms = [], []
    for b in boxes:
        c, M = crop_quad(det_img, b.quad)
        crops.append(c)
        Ms.append(M)
    recs = models["rec"](crops) if crops else []
    check_time()

    lines = []
    for i, (b, r, crop, M) in enumerate(zip(boxes, recs, crops, Ms)):
        chars = []
        for c in r.chars:
            cc = _nfkc(c["c"])
            chars.append({**c, "c": cc})
        while chars and chars[0]["c"].isspace():
            chars.pop(0)
        while chars and chars[-1]["c"].isspace():
            chars.pop()
        if not chars:
            continue
        lines.append({"det": b, "crop": crop, "M": M, "chars": chars,
                      "text": "".join(c["c"] for c in chars),
                      "conf": float(np.mean([c["conf"] for c in chars])), "engines": {
                          "ppocr_v5": {"text": "".join(c["c"] for c in chars),
                                       "conf": round(float(np.mean([c["conf"] for c in chars])), 4)}}})

    # ---- second opinions on weak / ambiguous lines
    prog(0.55, "Verifying uncertain text")
    rec4 = models.get("rec4")
    need = []
    for k, ln in enumerate(lines):
        minc = min(c["conf"] for c in ln["chars"])
        ambiguous = _is_numeric_like(ln["text"]) and any(ch in CONFUSABLE for ch in ln["text"])
        if ln["conf"] < SECOND_OPINION_CONF or minc < 0.85 or ambiguous:
            need.append(k)
    if second_opinion and need:
        v4 = rec4([lines[k]["crop"] for k in need]) if rec4 else [None] * len(need)
        for j, k in enumerate(need):
            check_time()
            ln = lines[k]
            cands = [{"engine": "ppocr_v5", "text": ln["text"], "conf": ln["conf"]}]
            if v4[j] is not None:
                t4 = _nfkc(v4[j].text).strip()
                ln["engines"]["ppocr_v4"] = {"text": t4, "conf": round(v4[j].conf, 4)}
                cands.append({"engine": "ppocr_v4", "text": t4, "conf": v4[j].conf})
            tr = tesseract_line(ln["crop"])
            if tr is not None:
                ln["engines"]["tesseract"] = {"text": tr[0], "conf": round(tr[1], 4)}
                cands.append({"engine": "tesseract", "text": tr[0], "conf": tr[1]})
            chosen, alts, verdict = _choose(cands)
            ln["verdict"] = verdict
            ln["alternatives"] = [{"engine": a["engine"], "text": a["text"], "conf": round(float(a["conf"]), 4)}
                                  for a in alts]
            if verdict == "confirmed":
                ln["conf"] = max(ln["conf"], min(0.97, max(c["conf"] for c in cands)))
                same = [c["text"] for c in cands if re.sub(r"\s+", "", c["text"]) == re.sub(r"\s+", "", ln["text"])]
                best = max(same, key=lambda t: t.count(" "))
                if best.count(" ") > ln["text"].count(" "):
                    ln["spaced_text"] = re.sub(r"\s+", " ", best.strip())
            elif chosen["engine"] != "ppocr_v5":
                ln["replaced_text"] = chosen["text"]
                ln["conf"] = float(chosen["conf"])

    prog(0.75, "Building regions")
    regions = []
    for k, ln in enumerate(lines):
        regions += _build_line_regions(ln, k, page, scale, canvas, ink, W, H, gray)
    check_time()

    text_h = float(np.median([r["style"]["font_size_px"] or 0 for r in regions])) if regions else 24.0
    graphics = L.detect_graphics(canvas, ink_all, rule_mask_d, max(8.0, text_h * 0.75), barcodes)
    whitespace = L.whitespace_bands(ink_all, max(8.0, text_h * 0.75), rule_mask_d)

    low_res = bool(regions) and text_h < 16
    if low_res:
        # tiny glyphs: confidence scores are not trustworthy — send nearly everything to review
        for r in regions:
            if r["conf"] < 0.985 and r["role"] == "text":
                r["flags"].append({"type": "low_resolution",
                                   "message": "Text is very small in this image; verify this reading"})
    _assign_tables(regions, tables)
    _classify_roles(regions, graphics, barcodes)
    _weights(regions)
    _alignment(regions, tables)
    _flags(regions)
    _blocks(regions)

    review = []
    for r in regions:
        for f in r["flags"]:
            review.append({"id": f"{r['id']}:{f['type']}", "region_id": r["id"], "page": page, "type": f["type"],
                           "message": f["message"], "text": r["text"], "bbox": r["bbox"],
                           "alternatives": f.get("alternatives", []), "word_ids": f.get("word_ids", [])})
        if r["flags"]:
            r["needs_review"] = True

    confs = [r["conf"] for r in regions if r["role"] == "text"]
    stats = {"lines": len(regions), "words": sum(len(r["words"]) for r in regions),
             "mean_conf": round(float(np.mean(confs)), 4) if confs else 0.0,
             "low_conf_lines": sum(1 for r in regions if r["conf"] < LINE_LOW_CONF),
             "review_items": len(review), "second_opinion_lines": len(need) if second_opinion else 0,
             "barcode_detections_dropped": dropped, "det_scale": scale,
             "seconds": round(time.time() - t_start, 2), "low_resolution": low_res}
    prog(1.0, "Done")
    warnings = []
    if low_res:
        warnings.append("Text in this image is very small (low resolution). OCR is unreliable — retake the photo closer "
                        "or at higher resolution, or review every flagged line.")
    if not regions:
        warnings.append("No text was found on this page. You can draw a region manually to edit it.")
    return {"warnings": warnings, "size": [W, H], "text_height": round(text_h, 2), "regions": regions, "tables": tables,
            "barcodes": barcodes, "graphics": graphics, "rulings": segs[:500], "whitespace": whitespace,
            "review": review, "stats": stats}


def _multiscale(det, img, limit=2400):
    """Union of detections at two scales: fine text at full size, isolated/large glyphs at reduced size."""
    a = det(img, limit_side=limit)
    small = int(min(max(img.shape[:2]), limit) * 0.62)
    b = det(img, limit_side=small)
    ab = [_bbox(x.quad) for x in a]
    for x in b:
        bb = _bbox(x.quad)
        if all(L._overlap(bb, o) < 0.25 and L._overlap(o, bb) < 0.25 for o in ab):
            a.append(x)
            ab.append(bb)
    return a


def _map_crop_x(M, x, h, scale):
    pts = np.float32([[x, 0], [x, h]]).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, M).reshape(-1, 2) / scale


def _char_positions(ln, scale):
    """Canvas x-centre of every recognised character (from CTC time steps)."""
    ch_h = ln["crop"].shape[0]
    out = []
    for c in ln["chars"]:
        xc = (c["x0"] + c["x1"]) / 2
        p = _map_crop_x(ln["M"], xc, ch_h, scale)
        out.append(float(p[:, 0].mean()))
    return out


def _aligned_chars(primary: str, prim_x: list[float], text: str) -> list[float]:
    """Positions for `text` characters by aligning it to the primary recognition."""
    import difflib

    xs = [None] * len(text)
    sm = difflib.SequenceMatcher(a=primary, b=text, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("equal", "replace"):
            for k in range(j2 - j1):
                src = i1 + min(k * max(1, i2 - i1) // max(1, j2 - j1), max(0, i2 - i1 - 1))
                if src < len(prim_x):
                    xs[j1 + k] = prim_x[src]
    known = [(i, x) for i, x in enumerate(xs) if x is not None]
    if not known:
        lo, hi = (min(prim_x), max(prim_x)) if prim_x else (0.0, 1.0)
        return [lo + (hi - lo) * (i + 0.5) / max(1, len(text)) for i in range(len(text))]
    for i in range(len(xs)):
        if xs[i] is None:
            prev = next(((j, x) for j, x in reversed(known) if j < i), None)
            nxt = next(((j, x) for j, x in known if j > i), None)
            if prev and nxt:
                xs[i] = prev[1] + (nxt[1] - prev[1]) * (i - prev[0]) / (nxt[0] - prev[0])
            else:
                xs[i] = (prev or nxt)[1]
    return xs  # type: ignore[return-value]


def _ink_segments(ink, quad, W, H):
    """Connected ink components belonging to a text line, grouped into words by gaps."""
    b = _clip(_bbox(quad), W, H)
    if b[2] - b[0] < 2 or b[3] - b[1] < 2:
        return None
    sub = ink[b[1]:b[3], b[0]:b[2]]
    poly = np.zeros_like(sub)
    cv2.fillPoly(poly, [np.round(quad - [b[0], b[1]]).astype(np.int32)], 255)
    sub = cv2.bitwise_and(sub, poly)
    n, lab, st, _ = cv2.connectedComponentsWithStats(sub, connectivity=8)
    if n <= 1:
        return None
    qh = max(4.0, (np.linalg.norm(quad[0] - quad[3]) + np.linalg.norm(quad[1] - quad[2])) / 2)
    # vertical centre of the line along x (quad may be slightly tilted)
    cy0 = (quad[0][1] + quad[3][1]) / 2 - b[1]
    cy1 = (quad[1][1] + quad[2][1]) / 2 - b[1]
    x0q, x1q = (quad[0][0] + quad[3][0]) / 2 - b[0], (quad[1][0] + quad[2][0]) / 2 - b[0]
    comps, small = [], []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 2:
            continue
        xm = x + w / 2
        t = 0.0 if x1q == x0q else min(1.0, max(0.0, (xm - x0q) / (x1q - x0q)))
        cy = cy0 + (cy1 - cy0) * t
        box = (int(x), int(y), int(x + w), int(y + h))
        if y + h < cy - 0.2 * qh or y > cy + 0.2 * qh:
            small.append(box)  # punctuation / neighbouring line: decided below
            continue
        comps.append(box)
    if not comps:
        return None
    # body band from the main glyphs; small marks (. , - ' ") inside it belong to this line
    top = float(np.percentile([c[1] for c in comps], 20))
    bot = float(np.percentile([c[3] for c in comps], 80))
    pad = 0.18 * (bot - top)
    for c in small:
        if c[1] >= top - pad and c[3] <= bot + pad:
            comps.append(c)
    hs = sorted(c[3] - c[1] for c in comps)
    em = max(4.0, float(np.percentile(hs, 90)) / 0.72)
    gap_thr = 1.0  # atoms: horizontally overlapping/touching components; words are formed later
    comps.sort()
    segs = []
    for c in comps:
        if segs and c[0] - segs[-1]["x1"] < gap_thr:
            s = segs[-1]
            s["x1"] = max(s["x1"], c[2])
            s["y0"] = min(s["y0"], c[1])
            s["y1"] = max(s["y1"], c[3])
            s["n"] += 1
        else:
            segs.append({"x0": c[0], "x1": c[2], "y0": c[1], "y1": c[3], "n": 1})
    for s in segs:
        s["x0"] += b[0]
        s["x1"] += b[0]
        s["y0"] += b[1]
        s["y1"] += b[1]
    return segs, em


def _build_line_regions(ln, k, page, scale, canvas, ink, W, H, gray=None):
    chars = ln["chars"]
    text = ln.get("replaced_text") or ln["text"]
    replaced = "replaced_text" in ln
    prim_x = _char_positions(ln, scale)
    if replaced:
        xs = _aligned_chars(ln["text"], prim_x, text)
        tchars = [{"c": c, "conf": ln["conf"], "x": x} for c, x in zip(text, xs)]
    elif ln.get("spaced_text"):
        # same characters as the primary read, plus word spaces another engine saw
        prim = [(c, x) for c, x in zip(chars, prim_x) if not c["c"].isspace()]
        tchars, k = [], 0
        for ch in ln["spaced_text"]:
            if ch == " ":
                if tchars and k < len(prim):
                    tchars.append({"c": " ", "conf": 1.0, "x": (tchars[-1]["x"] + prim[k][1]) / 2})
                continue
            if k < len(prim):
                c, x = prim[k]
                tchars.append({"c": c["c"], "conf": c["conf"], "x": x})
                k += 1
    else:
        tchars = [{"c": c["c"], "conf": c["conf"], "x": x} for c, x in zip(chars, prim_x)]
    quad = ln["det"].quad / scale
    seg = _ink_segments(ink, quad, W, H)
    words = []
    if seg:
        atoms, em = seg
        spaces = [c["x"] for c in tchars if c["c"].isspace()]
        gaps = [b["x0"] - a["x1"] for a, b in zip(atoms, atoms[1:])]
        med_gap = float(np.median(gaps)) if gaps else 0.0
        segs = [dict(atoms[0])]
        for a in atoms[1:]:
            s0 = segs[-1]
            gap = a["x0"] - s0["x1"]
            tol = 0.25 * em
            has_space = any(s0["x1"] - tol <= x <= a["x0"] + tol for x in spaces)
            split = (gap >= max(0.3 * em, 2.5 * med_gap)) or (has_space and gap >= max(0.17 * em, 1.5 * med_gap))
            if split:
                segs.append(dict(a))
            else:
                s0["x1"] = max(s0["x1"], a["x1"])
                s0["y0"] = min(s0["y0"], a["y0"])
                s0["y1"] = max(s0["y1"], a["y1"])
        # assign every non-space char to the nearest ink segment
        for c in tchars:
            if c["c"].isspace():
                continue
            d = [0 if s["x0"] <= c["x"] <= s["x1"] else min(abs(c["x"] - s["x0"]), abs(c["x"] - s["x1"])) for s in segs]
            segs[int(np.argmin(d))].setdefault("chars", []).append(c)
        # segments without characters are merged into the nearest neighbour (dropped punctuation etc.)
        filled = [s for s in segs if s.get("chars")]
        for s in segs:
            if s.get("chars") or not filled:
                continue
            nb = min(filled, key=lambda f: min(abs(f["x0"] - s["x1"]), abs(s["x0"] - f["x1"])))
            if min(abs(nb["x0"] - s["x1"]), abs(s["x0"] - nb["x1"])) < 0.6 * em:
                nb["x0"], nb["x1"] = min(nb["x0"], s["x0"]), max(nb["x1"], s["x1"])
                nb["y0"], nb["y1"] = min(nb["y0"], s["y0"]), max(nb["y1"], s["y1"])
        for s in filled:
            s["chars"].sort(key=lambda c: c["x"])
            wb = _clip([s["x0"] - 1, s["y0"] - 1, s["x1"] + 1, s["y1"] + 1], W, H)
            words.append({"text": "".join(c["c"] for c in s["chars"]), "bbox": wb,
                          "chars": [{"c": c["c"], "conf": round(c["conf"], 4)} for c in s["chars"]],
                          "conf": round(float(np.mean([c["conf"] for c in s["chars"]])), 4), "has_ink": True})
    if not words:
        b0 = _clip(_bbox(quad), W, H)
        b, has_ink = _tight(ink, b0)
        words.append({"text": text.strip(), "bbox": b, "has_ink": has_ink,
                      "chars": [{"c": c["c"], "conf": round(c["conf"], 4)} for c in tchars if not c["c"].isspace()],
                      "conf": round(ln["conf"], 4)})
    for wi, w in enumerate(words):
        w["id"] = f"p{page}-l{k}-w{wi}"
        if replaced:
            w["chars"] = []
            w["conf"] = round(ln["conf"], 4)
        w["style"] = estimate_style(canvas, ink, w["bbox"], w["text"], gray)
    lb = [min(w["bbox"][0] for w in words), min(w["bbox"][1] for w in words),
          max(w["bbox"][2] for w in words), max(w["bbox"][3] for w in words)]
    line_text = " ".join(w["text"] for w in words)
    style = estimate_style(canvas, ink, lb, line_text, gray)
    return [{"id": f"p{page}-l{k}", "page": page, "kind": "line", "text": line_text, "conf": round(ln["conf"], 4),
             "min_char_conf": round(min(c["conf"] for c in chars), 4), "bbox": lb, "quad": quad.round(1).tolist(),
             "words": words, "style": style, "engines": ln["engines"], "verdict": ln.get("verdict", "primary"),
             "alternatives": ln.get("alternatives", []), "role": "text", "locked": False, "table": None,
             "flags": [], "source": "ocr"}]


def _center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def _assign_tables(regions, tables):
    """Attach table/cell info; split lines that straddle several cells."""
    cells = [(t, c) for t in tables for c in t["cells"]]
    if not cells:
        return
    out = []
    for r in regions:
        by_cell: dict[int, list] = {}
        for w in r["words"]:
            cx, cy = _center(w["bbox"])
            hit = None
            for j, (t, c) in enumerate(cells):
                b = c["box"]
                if b[0] <= cx < b[2] and b[1] <= cy < b[3]:
                    hit = j
                    break
            by_cell.setdefault(hit, []).append(w)
        if len(by_cell) <= 1:
            j = next(iter(by_cell))
            if j is not None:
                t, c = cells[j]
                r["table"] = {"id": t["id"], "row": c["row"], "col": c["col"], "cell": c["box"]}
                for w in r["words"]:
                    if L._overlap(w["bbox"], c["box"]) < 0.85:
                        r["flags"].append({"type": "table_cell_uncertain", "word_ids": [w["id"]],
                                           "message": f"'{w['text']}' crosses a table cell border"})
            out.append(r)
            continue
        # split
        for n, (j, ws) in enumerate(sorted(by_cell.items(), key=lambda kv: min(w["bbox"][0] for w in kv[1]))):
            nr = dict(r)
            nr["id"] = f"{r['id']}s{n}"
            nr["words"] = ws
            for wi, w in enumerate(ws):
                w["id"] = f"{nr['id']}-w{wi}"
            nr["text"] = " ".join(w["text"] for w in ws)
            nr["bbox"] = [min(w["bbox"][0] for w in ws), min(w["bbox"][1] for w in ws),
                          max(w["bbox"][2] for w in ws), max(w["bbox"][3] for w in ws)]
            nr["conf"] = round(float(np.mean([w["conf"] for w in ws])), 4)
            nr["flags"] = list(r["flags"])
            nr["engines"] = {}
            nr["alternatives"] = []
            nr["verdict"] = "split"
            if j is not None:
                t, c = cells[j]
                nr["table"] = {"id": t["id"], "row": c["row"], "col": c["col"], "cell": c["box"]}
            else:
                nr["table"] = None
            out.append(nr)
    regions[:] = out


def _classify_roles(regions, graphics, barcodes):
    for g in graphics:
        gb = g["box"]
        gh = gb[3] - gb[1]
        group = list(gb)
        changed = True
        members = []
        while changed:
            changed = False
            for r in regions:
                if r["role"] != "text" or r in members:
                    continue
                b = r["bbox"]
                cy = (b[1] + b[3]) / 2
                inside_y = group[1] - 0.1 * gh <= cy <= group[3] + 0.1 * gh
                near_x = (b[0] - group[2] < 0.6 * gh) and (group[0] - b[2] < 0.6 * gh)
                if L._overlap(b, gb) > 0.3 or (inside_y and near_x):
                    members.append(r)
                    group = [min(group[0], b[0]), min(group[1], b[1]), max(group[2], b[2]), max(group[3], b[3])]
                    changed = True
        if members:
            g["kind"] = "logo"
            g["group_box"] = group
            for r in members:
                r["role"] = "logo_text"
                r["locked"] = True
    for bc in barcodes:
        bb = bc["box"]
        bh = bb[3] - bb[1]
        for r in regions:
            b = r["bbox"]
            horiz = min(b[2], bb[2]) - max(b[0], bb[0]) > 0.5 * (b[2] - b[0])
            below = 0 <= b[1] - bb[3] < 0.6 * bh
            above = 0 <= bb[1] - b[3] < 0.6 * bh
            if horiz and (below or above) and len(r["words"]) == 1:
                r["role"] = "barcode_text"
                r["locked"] = True
                bc["text_region"] = r["id"]


def _weights(regions):
    items = [(w["style"]["stroke_ratio"], w["style"]["font_size_px"]) for r in regions if r["role"] == "text"
             for w in r["words"] if w["style"]["stroke_ratio"] and len(w["text"]) >= 3]
    if not items:
        return
    rat = np.array([i[0] for i in items])
    sz = np.array([i[1] for i in items])
    glob = float(np.percentile(rat, 35))

    def base_for(size):
        sel = rat[np.abs(sz - size) <= 0.2 * size]
        return float(np.percentile(sel, 35)) if len(sel) >= 8 else glob

    for r in regions:
        for w in [r] + r["words"]:
            s = w["style"]
            if not s["stroke_ratio"]:
                continue
            base = base_for(s["font_size_px"] or 0)
            s["weight_score"] = round(s["stroke_ratio"] / base, 3)
            s["weight"] = "bold" if s["weight_score"] > 1.3 else "regular"
        for w in r["words"]:
            if len(w["text"]) <= 2 and r["style"].get("weight"):
                w["style"]["weight"] = r["style"]["weight"]


def _alignment(regions, tables):
    for r in regions:
        b = r["bbox"]
        align = "left"
        if r["table"]:
            c = r["table"]["cell"]
            lg, rg = b[0] - c[0], c[2] - b[2]
            if rg < 0.5 * lg:
                align = "right"
            elif abs(lg - rg) < 0.2 * max(1, lg + rg) and min(lg, rg) > 6:
                align = "center"
        else:
            th = max(4, (r["style"]["font_size_px"] or 20) * 0.4)
            same_right = [o for o in regions if o is not r and abs(o["bbox"][2] - b[2]) < th
                          and abs(o["bbox"][0] - b[0]) > th and abs(o["bbox"][1] - b[1]) < 6 * (b[3] - b[1])]
            same_left = [o for o in regions if o is not r and abs(o["bbox"][0] - b[0]) < th
                         and abs(o["bbox"][1] - b[1]) < 6 * (b[3] - b[1])]
            if len(same_right) >= 1 and len(same_right) > len(same_left):
                align = "right"
        r["style"]["align"] = align
        for w in r["words"]:
            w["style"]["align"] = align


def _flags(regions):
    for r in regions:
        if r["role"] != "text":
            continue
        if r["conf"] < LINE_LOW_CONF:
            r["flags"].append({"type": "low_confidence", "message": f"Low OCR confidence ({r['conf']:.0%})",
                               "alternatives": r.get("alternatives", [])})
        if r.get("verdict") == "disagreement":
            r["flags"].append({"type": "engine_disagreement", "message": "OCR engines disagree",
                               "alternatives": r["alternatives"]})
        elif r.get("verdict") in ("corrected", "numeric_context"):
            r["flags"].append({"type": "engine_disagreement",
                               "message": "Primary OCR was overruled by secondary engines — please confirm",
                               "alternatives": r["alternatives"]})
        for w in r["words"]:
            unclear = [i for i, c in enumerate(w["chars"]) if c["conf"] < CHAR_UNCLEAR]
            if unclear:
                r["flags"].append({"type": "unclear_characters", "word_ids": [w["id"]], "chars": unclear,
                                   "message": f"Unclear character(s) in '{w['text']}'"})
            elif w["conf"] < WORD_LOW_CONF and r["conf"] >= LINE_LOW_CONF:
                r["flags"].append({"type": "low_confidence", "word_ids": [w["id"]],
                                   "message": f"Low confidence word '{w['text']}' ({w['conf']:.0%})"})
            weak = [c for c in w["chars"] if c["c"] in CONFUSABLE and c["conf"] < 0.97]
            if _is_numeric_like(w["text"]) and (weak or (not w["chars"] and any(ch in CONFUSABLE for ch in w["text"]))):
                r["flags"].append({"type": "ambiguous_number", "word_ids": [w["id"]],
                                   "message": f"'{w['text']}' mixes digits with look-alike letters"})
        if r["verdict"] in ("disagreement", "corrected", "numeric_context") and _is_numeric_like(r["text"]):
            r["flags"].append({"type": "ambiguous_number", "message": "Engines read this number differently",
                               "alternatives": r["alternatives"]})
    # overlapping regions
    for i, a in enumerate(regions):
        for b in regions[i + 1:]:
            ov = max(L._overlap(a["bbox"], b["bbox"]), L._overlap(b["bbox"], a["bbox"]))
            if ov > 0.35:
                for r, o in ((a, b), (b, a)):
                    r["flags"].append({"type": "overlap", "message": f"Overlaps another text region ('{o['text']}')"})
    # de-duplicate flag types per region (keep first of each type/word combination)
    for r in regions:
        seen, fl = set(), []
        for f in r["flags"]:
            key = (f["type"], tuple(f.get("word_ids", [])))
            if key not in seen:
                seen.add(key)
                fl.append(f)
        r["flags"] = fl


def _blocks(regions):
    order = sorted(range(len(regions)), key=lambda i: (regions[i]["bbox"][1], regions[i]["bbox"][0]))
    block_of = {}
    nb = 0
    for i in order:
        r = regions[i]
        b = r["bbox"]
        h = b[3] - b[1]
        best = None
        for j in order:
            if j == i or j not in block_of:
                continue
            o = regions[j]["bbox"]
            if r["table"] or regions[j]["table"]:
                if not (r["table"] and regions[j]["table"] and r["table"]["cell"] == regions[j]["table"]["cell"]):
                    continue
            gap = b[1] - o[3]
            if -0.3 * h <= gap <= 0.9 * h and abs(o[0] - b[0]) < 1.5 * h:
                best = block_of[j]
                break
        if best is None:
            best = nb
            nb += 1
        block_of[i] = best
        r["block"] = best
