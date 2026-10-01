"""Region-based text editing on the page raster.

For every edit only the pixels of the selected text are removed (local inpainting) and the
replacement text is drawn into the same spot. Everything else on the page is left
bit-identical: each edit yields a patch plus a mask of exactly the pixels it changed.

Style matching is self-calibrated: the recognised original text is re-rendered in each
candidate font and compared with the real pixels (size, horizontal scale, weight, blur,
colour), so the replacement inherits the measured look of the original print/photo.
"""
from __future__ import annotations

import hashlib
import math
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_DIR = Path(__file__).parent / "fonts"
FAMILIES = {
    "sans": ("LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf"),
    "serif": ("LiberationSerif-Regular.ttf", "LiberationSerif-Bold.ttf"),
    "mono": ("LiberationMono-Regular.ttf", "LiberationMono-Bold.ttf"),
    "dejavu-sans": ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"),
    "dejavu-serif": ("DejaVuSerif.ttf", "DejaVuSerif-Bold.ttf"),
    "dejavu-mono": ("DejaVuSansMono.ttf", "DejaVuSansMono-Bold.ttf"),
}
SS = 4  # supersampling factor for glyph rendering
SIGMAS = (0.0, 0.4, 0.7, 1.0, 1.3, 1.7, 2.2)


@lru_cache(maxsize=256)
def _font(family: str, bold: bool, px: int) -> ImageFont.FreeTypeFont:
    reg, bld = FAMILIES[family]
    return ImageFont.truetype(str(FONT_DIR / (bld if bold else reg)), max(4, px))


def text_alpha(text: str, family: str, bold: bool, size_px: float, hscale: float = 1.0):
    """Coverage mask of `text` at real pixel scale.

    Returns (alpha float32, pen_x, baseline_y): pen origin and baseline inside the mask.
    """
    f = _font(family, bold, int(round(size_px * SS)))
    asc, desc = f.getmetrics()
    pad = 4 * SS
    w = int(math.ceil(f.getlength(text))) + 2 * pad if text else 2 * pad
    img = Image.new("L", (max(1, w), asc + desc + 2 * pad), 0)
    if text:
        ImageDraw.Draw(img).text((pad, pad), text, font=f, fill=255)
    a = np.asarray(img, np.float32) / 255.0
    ow = max(1, int(round(a.shape[1] * hscale / SS)))
    oh = max(1, int(round(a.shape[0] / SS)))
    a = cv2.resize(a, (ow, oh), interpolation=cv2.INTER_AREA)
    sx = ow / (img.width / 1.0)
    sy = oh / (img.height / 1.0)
    return a, pad * sx, (pad + asc) * sy


def _ink_geom(m: np.ndarray):
    """(x0, x1, top, baseline) of a binary mask using robust column statistics."""
    cols = np.where(m.any(0))[0]
    if len(cols) == 0:
        return None
    tops = np.array([np.argmax(m[:, c]) for c in cols])
    bots = np.array([m.shape[0] - 1 - np.argmax(m[::-1, c]) for c in cols])
    return float(cols.min()), float(cols.max() + 1), float(np.percentile(tops, 5)), float(np.median(bots) + 1)


def _shift(a: np.ndarray, dx: float, dy: float, shape) -> np.ndarray:
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(a, M, (shape[1], shape[0]), flags=cv2.INTER_LINEAR, borderValue=0)


def _blur(a, s):
    return cv2.GaussianBlur(a, (0, 0), s) if s > 0 else a


@dataclass
class StyleFit:
    family: str
    bold: bool
    size_px: float
    hscale: float
    sigma: float
    color: tuple  # BGR
    baseline: float  # canvas y
    score: float
    tracking: float = 0.0
    candidates: list = field(default_factory=list)

    def to_dict(self):
        return {"family": self.family, "bold": self.bold, "size_px": round(self.size_px, 2),
                "hscale": round(self.hscale, 4), "sigma": self.sigma,
                "color": "#%02x%02x%02x" % (int(self.color[2]), int(self.color[1]), int(self.color[0])),
                "baseline": round(self.baseline, 2), "score": round(self.score, 4)}


def _local_background(win: np.ndarray, removal: np.ndarray) -> np.ndarray:
    m = (removal > 0).astype(np.uint8) * 255
    if not m.any():
        return win.copy()
    return cv2.inpaint(win, m, 3, cv2.INPAINT_TELEA)


def _darkness(win: np.ndarray, bg: np.ndarray, mask: np.ndarray):
    g = cv2.cvtColor(win, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY).astype(np.float32)
    vals = g[mask > 0]
    lo = float(np.percentile(vals, 3)) if vals.size else float(g.min())
    contrast = np.maximum(b - lo, 8.0)
    return np.clip((b - g) / contrast, 0, 1), lo


def _rough_ink(win: np.ndarray, allowed: np.ndarray, size_hint: float) -> np.ndarray:
    """Pixels of dark marks inside `allowed`, including anti-aliased fringes."""
    g = cv2.cvtColor(win, cv2.COLOR_BGR2GRAY).astype(np.float32)
    k = int(max(5, size_hint * 0.6)) | 1
    bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.medianBlur(bg.astype(np.uint8), k if k <= 255 else 255).astype(np.float32)
    bg = np.maximum(bg, g)
    d = bg - g
    core = d[allowed > 0]
    if core.size == 0:
        return np.zeros(g.shape, np.uint8)
    peak = float(np.percentile(core, 99))
    if peak < 12:
        return np.zeros(g.shape, np.uint8)
    strong = (d > max(10.0, 0.45 * peak)) & (allowed > 0)
    weak = (d > max(5.0, 0.10 * peak)) & (allowed > 0)
    # hysteresis: keep weak pixels connected to strong ones (fringes), not stray noise
    n, lab = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
    keep = np.unique(lab[strong])
    keep = keep[keep > 0]
    m = np.isin(lab, keep)
    m = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)) & (allowed > 0)
    return (m > 0).astype(np.uint8)


def fit_style(canvas: np.ndarray, bbox, text: str, obstacles_mask: np.ndarray | None = None,
              hint: dict | None = None, families=None, parts: list | None = None) -> StyleFit | None:
    """Calibrate font/size/scale/blur/colour by re-rendering `text` over its own pixels."""
    hint = hint or {}
    text = text.strip()
    if not text:
        return None
    H, W = canvas.shape[:2]
    x0, y0, x1, y1 = bbox
    h = max(4, y1 - y0)
    pad = int(max(4, 0.5 * h))
    wx0, wy0, wx1, wy1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
    win = canvas[wy0:wy1, wx0:wx1]
    allowed = np.zeros(win.shape[:2], np.uint8)
    e = 3
    allowed[max(0, y0 - wy0 - e):y1 - wy0 + e, max(0, x0 - wx0 - e):x1 - wx0 + e] = 1
    if obstacles_mask is not None:
        allowed &= (obstacles_mask[wy0:wy1, wx0:wx1] == 0).astype(np.uint8)
    rough = _rough_ink(win, allowed, h)
    if rough.sum() < 6:
        return None
    bg = _local_background(win, rough)
    D, lo = _darkness(win, bg, rough)
    if parts:
        # normalise each word on its own: lines often mix a grey label with a black value
        for pb in parts:
            px0, py0 = max(0, pb[0] - wx0 - 2), max(0, pb[1] - wy0 - 2)
            px1, py1 = max(0, pb[2] - wx0 + 2), max(0, pb[3] - wy0 + 2)
            sl = (slice(py0, py1), slice(px0, px1))
            sub_mask = np.zeros_like(rough)
            sub_mask[sl] = rough[sl]
            if sub_mask.sum() >= 4:
                Dp, _ = _darkness(win, bg, sub_mask)
                D[sl] = Dp[sl]
    D = D * (allowed > 0)
    geo = _ink_geom(D > 0.5)
    if geo is None:
        return None
    gx0, gx1, gtop, gbase = geo
    # observed darkness in grey levels; the model is  obs = contrast * blur(glyph coverage)
    gwin = cv2.cvtColor(win, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gbg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY).astype(np.float32)
    obs = np.clip(gbg - gwin, 0, None) * (allowed > 0)
    part_sl = []
    for pb in (parts or [[x0, y0, x1, y1]]):
        py0, py1 = max(0, pb[1] - wy0 - 3), max(0, pb[3] - wy0 + 3)
        px0, px1 = max(0, pb[0] - wx0 - 3), max(0, pb[2] - wx0 + 3)
        if py1 > py0 and px1 > px0:
            part_sl.append((slice(py0, py1), slice(px0, px1)))
    tot = float((obs ** 2).sum()) or 1.0

    def model_err(Am):
        res = float((obs ** 2).sum())
        for sl in part_sl:
            o, a = obs[sl], Am[sl]
            aa = float((a * a).sum())
            if aa <= 1e-6:
                continue
            c = max(0.0, float((o * a).sum()) / aa)
            res += float(((o - c * a) ** 2).sum()) - float((o ** 2).sum())
        return res / tot

    has_asc = any(ch.isupper() or ch.isdigit() or ch in "bdfhklt!?/()[]{}#$%&@|'\"" for ch in text)
    fams = families or ([hint["family"]] if hint.get("family") in FAMILIES else list(FAMILIES))
    bolds = [hint["bold"]] if isinstance(hint.get("bold"), bool) else [False, True]
    size0 = float(hint.get("size_px") or h / 0.9)
    def geom(fam, bold, size, hs, sg):
        a, px, base = text_alpha(text, fam, bold, size, hs)
        ab = _blur(a, sg)
        mx = float(ab.max())
        return (a, px, base, _ink_geom(ab > 0.5 * mx) if mx > 0 else None)

    def evaluate(fam, bold, size, sg_geo):
        # size from blurred-rendering geometry (thresholds comparable with the blurred original)
        for _ in range(3):
            a, px, base, g2 = geom(fam, bold, size, 1.0, sg_geo)
            if g2 is None:
                return None
            rh = g2[3] - g2[2]
            oh = gbase - gtop
            if rh <= 0 or oh <= 0:
                return None
            nsize = size * oh / rh
            done = abs(nsize - size) < 0.05
            size = nsize
            if done:
                break
        a, px, base, g2 = geom(fam, bold, size, 1.0, sg_geo)
        if g2 is None:
            return None
        hs = (gx1 - gx0) / max(1.0, g2[1] - g2[0])
        if not 0.6 < hs < 1.6:
            return None
        a, px, base, g2 = geom(fam, bold, size, hs, sg_geo)
        dx = gx0 - g2[0]
        dy = gbase - g2[3]
        placed = _shift(a, dx, dy, D.shape)
        best = None
        for sg in SIGMAS:
            err = model_err(_blur(placed, sg))
            if best is None or err < best[0]:
                best = (err, sg)
        err, sg = best
        for ddx, ddy in ((0.5, 0), (-0.5, 0), (0, 0.5), (0, -0.5)):
            e2 = model_err(_blur(_shift(a, dx + ddx, dy + ddy, D.shape), sg))
            if e2 < err:
                err, dx, dy = e2, dx + ddx, dy + ddy
        return {"family": fam, "bold": bold, "size": size, "hscale": hs, "sigma": sg,
                "score": err + 0.15 * abs(math.log(hs)), "baseline": wy0 + dy + base, "pen_x": wx0 + dx + px}

    results = []
    for fam in fams:
        for bold in bolds:
            r0 = evaluate(fam, bold, size0, 0.0)
            if r0 is not None:
                results.append(r0)
    # second pass: re-measure the best candidates with their own blur applied to the rendering
    results.sort(key=lambda r: r["score"])
    refined = []
    pool = results[:6]
    top = results[0]["family"] if results else None
    for r in results[6:]:
        if r["family"] == top and r["bold"] != results[0]["bold"] and not any(
                p["family"] == top and p["bold"] == r["bold"] for p in pool):
            pool.append(r)
    for r0 in pool:
        if r0["sigma"] > 0.5:
            r1 = evaluate(r0["family"], r0["bold"], r0["size"], r0["sigma"])
            if r1 is not None and r1["score"] < r0["score"]:
                r0 = r1
        refined.append(r0)
    results = refined + [r for r in results[6:] if r not in pool]
    if not results:
        return None
    results.sort(key=lambda r: r["score"])
    r = results[0]
    a, px, base = text_alpha(text, r["family"], r["bold"], r["size"], r["hscale"])
    A = _blur(_shift(a, r["pen_x"] - wx0 - px, r["baseline"] - wy0 - base, D.shape), r["sigma"])
    best_sl, best_m = None, -1.0
    for sl in part_sl:
        m = float(A[sl].sum())
        if m > best_m:
            best_sl, best_m = sl, m
    sl = best_sl or (slice(None), slice(None))
    Aa = A[sl]
    aa = float((Aa * Aa).sum())
    if aa > 1e-6:
        color = []
        sel = Aa > 0.5
        for ch in range(3):
            d = (bg[sl][..., ch].astype(np.float32) - win[sl][..., ch].astype(np.float32)) * (allowed[sl] > 0)
            c = max(0.0, float((d * Aa).sum()) / aa)
            ref = float(np.median(bg[sl][..., ch][sel])) if sel.any() else float(np.median(bg[..., ch]))
            color.append(float(np.clip(ref - c, 0, 255)))
        color = tuple(color)
    else:
        color = (lo, lo, lo)
    return StyleFit(r["family"], r["bold"], r["size"], r["hscale"], r["sigma"], color, r["baseline"], r["score"],
                    candidates=[{k: (round(v, 3) if isinstance(v, float) else v) for k, v in c.items()}
                                for c in results[:4]])


# ---------------------------------------------------------------------------------- edits

@dataclass
class Patch:
    edit_id: str
    x: int
    y: int
    rgb: np.ndarray  # h,w,3 BGR
    mask: np.ndarray  # h,w bool — pixels changed by this edit
    info: dict

    @property
    def bbox(self):
        ys, xs = np.nonzero(self.mask)
        if len(xs) == 0:
            return None
        return [int(self.x + xs.min()), int(self.y + ys.min()), int(self.x + xs.max() + 1), int(self.y + ys.max() + 1)]


def _hex_to_bgr(s: str):
    s = s.lstrip("#")
    r, g, b = int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    return (float(b), float(g), float(r))


def _noise_sigma(win: np.ndarray, exclude: np.ndarray) -> float:
    g = cv2.cvtColor(win, cv2.COLOR_BGR2GRAY).astype(np.float32)
    res = g - cv2.GaussianBlur(g, (0, 0), 1.5)
    sel = res[(exclude == 0)]
    if sel.size < 50:
        return 0.0
    return float(np.median(np.abs(sel)) * 1.4826)


def render_edit(canvas: np.ndarray, edit: dict, ctx: dict) -> Patch:
    """Apply one edit to `canvas` (not modified) and return the patch.

    edit: {id, bbox:[x0,y0,x1,y1], text, original_text, style:{...overrides}}
    ctx:  {obstacles: [[x0,y0,x1,y1], ...]  boxes that must not be touched,
           rule_boxes: [...] rulings, fit: StyleFit|None (cached calibration from the parent line),
           limits: (left_x, right_x) horizontal room available}
    """
    H, W = canvas.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in edit["bbox"]]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    new_text = edit.get("text", "")
    ov = edit.get("style") or {}
    info: dict = {"warnings": []}
    fit: StyleFit | None = ctx.get("fit")
    h = max(4, y1 - y0)
    size_hint = fit.size_px if fit else h / 0.9

    # ---- working window: original box + generous room for longer replacement text
    room_l, room_r = ctx.get("limits") or (max(0, x0 - 4 * h), min(W, x1 + 12 * h))
    span_l, span_r = ctx.get("line_room") or (room_l, room_r)
    span_l, span_r = max(0, min(span_l, room_l)), min(W, max(span_r, room_r))
    for f in ctx.get("followers") or []:
        span_l, span_r = min(span_l, f["bbox"][0]), max(span_r, f["bbox"][2])
    mg = int(max(6, 0.5 * size_hint))
    wx0, wx1 = max(0, min(x0, int(span_l)) - mg), min(W, max(x1, int(span_r)) + mg)
    wy0, wy1 = max(0, y0 - 2 * mg), min(H, y1 + 2 * mg)
    win = canvas[wy0:wy1, wx0:wx1].copy()

    obst = np.zeros(win.shape[:2], np.uint8)
    for b in ctx.get("obstacles", []):
        bx0, by0, bx1, by1 = b[0] - wx0 - 1, b[1] - wy0 - 1, b[2] - wx0 + 1, b[3] - wy0 + 1
        if bx1 > 0 and by1 > 0 and bx0 < obst.shape[1] and by0 < obst.shape[0]:
            obst[max(0, by0):max(0, by1), max(0, bx0):max(0, bx1)] = 1
    for b in ctx.get("rule_boxes", []):
        bx0, by0, bx1, by1 = b[0] - wx0 - 1, b[1] - wy0 - 1, b[2] - wx0 + 1, b[3] - wy0 + 1
        if bx1 > 0 and by1 > 0 and bx0 < obst.shape[1] and by0 < obst.shape[0]:
            obst[max(0, by0):max(0, by1), max(0, bx0):max(0, bx1)] = 1

    # ---- 1. remove the original text pixels only
    e = int(max(2, round((fit.sigma if fit else 1.0) * 2 + 1)))
    allowed = np.zeros(win.shape[:2], np.uint8)
    allowed[max(0, y0 - wy0 - e):y1 - wy0 + e, max(0, x0 - wx0 - e):x1 - wx0 + e] = 1
    followers = ctx.get("followers") or []
    fol_obst = np.zeros(win.shape[:2], np.uint8)
    for f in followers:
        b = f["bbox"]
        fol_obst[max(0, b[1] - wy0 - 1):max(0, b[3] - wy0 + 1), max(0, b[0] - wx0 - 1):max(0, b[2] - wx0 + 1)] = 1
    allowed &= ((obst == 0) & (fol_obst == 0)).astype(np.uint8)
    removal = _rough_ink(win, allowed, size_hint)
    if edit.get("erase_box"):
        removal = allowed.copy()  # manual region with no clear ink: clear the whole box
    # followers' own ink (only needed if they have to move)
    fol_ink = []
    for f in followers:
        b = f["bbox"]
        fa = np.zeros(win.shape[:2], np.uint8)
        fa[max(0, b[1] - wy0 - e):max(0, b[3] - wy0 + e), max(0, b[0] - wx0 - e):max(0, b[2] - wx0 + e)] = 1
        fa &= (obst == 0).astype(np.uint8)
        fol_ink.append(_rough_ink(win, fa, size_hint))
    bg = _local_background(win, removal)
    sig_n = _noise_sigma(win, cv2.dilate(removal, np.ones((5, 5), np.uint8)))
    seed = int(hashlib.sha1((edit.get("id", "") + new_text).encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    out = bg.astype(np.float32)
    if removal.any() and sig_n > 0.3:
        noise = rng.normal(0, sig_n, out.shape[:2]).astype(np.float32)[..., None]
        out += noise * removal[..., None]

    # ---- 2. draw replacement text
    new_box = None
    shift = 0
    if new_text.strip():
        fam = ov.get("family") or (fit.family if fit else "sans")
        if fam not in FAMILIES:
            fam = "sans"
        bold = ov["bold"] if isinstance(ov.get("bold"), bool) else (fit.bold if fit else False)
        size = float(ov.get("size_px") or (fit.size_px if fit else h / 0.9))
        hs = float(ov.get("hscale") or (fit.hscale if fit else 1.0))
        sigma = float(ov["sigma"]) if ov.get("sigma") is not None else (fit.sigma if fit else 0.7)
        color = _hex_to_bgr(ov["color"]) if ov.get("color") else (fit.color if fit else (30.0, 30.0, 30.0))
        align = ov.get("align") or ctx.get("align") or "left"
        baseline = fit.baseline if fit else (y1 - 0.22 * (y1 - y0))
        # ink extents of the original word define the anchor
        anchor_l, anchor_r = x0, x1
        a, pen, base = text_alpha(new_text, fam, bold, size, hs)
        g = _ink_geom(a > 0.5)
        ink_w = (g[1] - g[0]) if g else a.shape[1]
        avail = (room_r - room_l) if ctx.get("limits") else None
        if avail and ink_w > avail and not ov.get("no_fit"):
            # condense first (up to 12%), then shrink the size (up to 15%)
            f = max(0.88, avail / ink_w)
            hs2 = hs * f
            a, pen, base = text_alpha(new_text, fam, bold, size, hs2)
            g = _ink_geom(a > 0.5)
            ink_w = (g[1] - g[0]) if g else a.shape[1]
            info["condensed"] = round(f, 3)
            if ink_w > avail:
                s2 = max(0.85, avail / ink_w)
                size *= s2
                a, pen, base = text_alpha(new_text, fam, bold, size, hs2)
                g = _ink_geom(a > 0.5)
                ink_w = (g[1] - g[0]) if g else a.shape[1]
                info["shrunk"] = round(s2, 3)
            if ink_w > avail + 1:
                info["warnings"].append("Replacement text is wider than the available space and may touch nearby content.")
        gx0 = g[0] if g else 0.0
        if align == "right":
            tx = anchor_r - (gx0 + ink_w)
        elif align == "center":
            tx = (anchor_l + anchor_r) / 2 - (gx0 + ink_w / 2)
        else:
            tx = anchor_l - gx0
        tx += float(ov.get("dx") or 0)
        new_l, new_r = tx + gx0, tx + gx0 + ink_w
        shift = _reflow_shift(ctx, followers, x0, x1, new_l, new_r, size)
        if shift:
            info["moved"] = [{"id": f["id"], "dx": shift} for f in followers]
        ty = baseline - base + float(ov.get("dy") or 0)
        A = _shift(a, tx - wx0, ty - wy0, win.shape[:2])
        A = _blur(A, sigma)
        if shift:
            out = _move_followers(win, out, bg, fol_ink, shift, rng, sig_n)
        A = np.clip(A, 0, 1)
        Cv = np.array(color, np.float32)[None, None, :]
        out = out * (1 - A[..., None]) + Cv * A[..., None]
        if sig_n > 0.3:
            tn = rng.normal(0, sig_n, out.shape[:2]).astype(np.float32)[..., None]
            out += tn * (A[..., None] > 0.02)
        ys, xs = np.nonzero(A > 0.02)
        if len(xs):
            new_box = [int(wx0 + xs.min()), int(wy0 + ys.min()), int(wx0 + xs.max() + 1), int(wy0 + ys.max() + 1)]
            fo = fol_obst
            if shift:
                fo = cv2.warpAffine(fol_obst, np.float32([[1, 0, shift], [0, 1, 0]]), (fol_obst.shape[1], fol_obst.shape[0]),
                                    flags=cv2.INTER_NEAREST, borderValue=0)
            if (obst[ys, xs] | fo[ys, xs]).any():
                info["warnings"].append("Replacement text overlaps neighbouring text or lines.")
        info["style"] = {"family": fam, "bold": bold, "size_px": round(size, 2), "hscale": round(hs, 4),
                         "sigma": sigma, "align": align,
                         "color": "#%02x%02x%02x" % (int(color[2]), int(color[1]), int(color[0]))}
    elif followers and ctx.get("reflow", True) and not ov.get("no_reflow"):
        # deletion: close the gap so the line reads naturally
        shift = _reflow_shift(ctx, followers, x0, x1, None, None, size_hint)
        if shift:
            out = _move_followers(win, out, bg, fol_ink, shift, rng, sig_n)
            info["moved"] = [{"id": f["id"], "dx": shift} for f in followers]
    out = np.clip(np.round(out), 0, 255).astype(np.uint8)
    changed = np.any(out != win, axis=2)
    info["new_text_bbox"] = new_box
    info["removed_pixels"] = int(removal.sum())
    p = Patch(edit.get("id", ""), wx0, wy0, out, changed, info)
    info["changed_bbox"] = p.bbox
    return p


def _reflow_shift(ctx, followers, x0, x1, new_l, new_r, size) -> int:
    """Integer x-shift for the words that follow the edited one on the same line (0 = none)."""
    if not followers or not ctx.get("reflow", True):
        return 0
    d = ctx.get("follow_dir", 1)
    if d > 0:
        gap = followers[0]["bbox"][0] - x1
        if new_r is None:  # deletion: next word takes the deleted word's place
            want = x0 - followers[0]["bbox"][0]
        else:
            want = (new_r + gap) - followers[0]["bbox"][0]
        last = max(f["bbox"][2] for f in followers)
        room = ctx.get("line_room", (0, 1e9))[1] - last
        want = min(want, room)
    else:
        gap = x0 - followers[0]["bbox"][2]
        if new_l is None:
            want = x1 - followers[0]["bbox"][2]
        else:
            want = (new_l - gap) - followers[0]["bbox"][2]
        first = min(f["bbox"][0] for f in followers)
        room = first - ctx.get("line_room", (0, 1e9))[0]
        want = max(want, -room)
    if abs(want) < max(1.0, 0.12 * size):
        return 0
    return int(round(want))


def _move_followers(win, out, bg, fol_ink, dx, rng, sig_n):
    """Translate whole following words by dx pixels, keeping their exact pixel appearance."""
    Hh, Ww = win.shape[:2]
    for ink in fol_ink:
        if not ink.any():
            continue
        m = cv2.dilate(ink, np.ones((3, 3), np.uint8)) > 0
        # erase at the old position
        bgl = _local_background(win, m.astype(np.uint8))
        fill = bgl.astype(np.float32)
        if sig_n > 0.3:
            fill += rng.normal(0, sig_n, fill.shape[:2]).astype(np.float32)[..., None]
        out[m] = fill[m]
        # paste the word's "darkness" (paper minus ink) at the new position
        dark = (bgl.astype(np.float32) - win.astype(np.float32)) * m[..., None]
        M = np.float32([[1, 0, dx], [0, 1, 0]])
        dark_s = cv2.warpAffine(dark, M, (Ww, Hh), flags=cv2.INTER_NEAREST, borderValue=0)
        m_s = cv2.warpAffine(m.astype(np.uint8), M, (Ww, Hh), flags=cv2.INTER_NEAREST, borderValue=0) > 0
        out[m_s] = out[m_s] - dark_s[m_s]
    return out


def composite(canvas: np.ndarray, patches: list[Patch]) -> np.ndarray:
    out = canvas.copy()
    for p in patches:
        h, w = p.mask.shape
        reg = out[p.y:p.y + h, p.x:p.x + w]
        reg[p.mask] = p.rgb[p.mask]
    return out


_FIT_CACHE: dict[str, StyleFit | None] = {}
_FIT_LOCK = threading.Lock()


def cached_fit(key: str, fn):
    with _FIT_LOCK:
        if key in _FIT_CACHE:
            return _FIT_CACHE[key]
    v = fn()
    with _FIT_LOCK:
        if len(_FIT_CACHE) > 4000:
            _FIT_CACHE.clear()
        _FIT_CACHE[key] = v
    return v
