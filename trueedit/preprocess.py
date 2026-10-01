"""Image preprocessing.

The original decoded image is never modified. Preprocessing produces:
  * canvas    - the colour working page (perspective-corrected, oriented, deskewed). Pixel
                values are only resampled, never re-toned, so the page keeps its true look.
  * ocr_input - an enhanced copy of the canvas (flattened illumination, shadow removal,
                contrast normalisation, light denoise/sharpen) used only for OCR/analysis.
  * transform - 3x3 matrix mapping original-image pixels to canvas pixels.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

STANDARD_ASPECTS = {"letter": 11 / 8.5, "a4": 297 / 210, "legal": 14 / 8.5, "a5": 210 / 148, "receipt-80mm": None}


@dataclass
class Prepared:
    canvas: np.ndarray
    ocr_input: np.ndarray
    transform: np.ndarray  # original -> canvas
    steps: list[dict] = field(default_factory=list)
    page_quad: list | None = None  # in original coords
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- page detection

def _refine_corners(mask: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Fit a robust line to the boundary pixels near each quad side and intersect them."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return quad
    pts = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    lines = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        ab = b - a
        L = np.linalg.norm(ab)
        if L < 1:
            return quad
        n = np.array([-ab[1], ab[0]]) / L
        d = np.abs((pts - a) @ n)
        t = ((pts - a) @ ab) / (L * L)
        sel = pts[(d < max(4.0, L * 0.01)) & (t > 0.08) & (t < 0.92)]
        if len(sel) < 10:
            return quad
        vx, vy, x0, y0 = cv2.fitLine(sel, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0]), np.array([vx, vy])))
    out = []
    for i in range(4):
        (p1, d1), (p2, d2) = lines[i - 1], lines[i]
        A = np.array([d1, -d2]).T
        if abs(np.linalg.det(A)) < 1e-6:
            return quad
        s = np.linalg.solve(A, p2 - p1)
        out.append(p1 + s[0] * d1)
    out = np.array(out, np.float32)
    if np.max(np.linalg.norm(out - quad, axis=1)) > 0.05 * max(mask.shape):
        return quad
    return out


def _order(pts):
    pts = np.asarray(pts, np.float32)
    c = pts.mean(0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]  # clockwise in image coords starting from ~ -pi (left)
    i = np.argmin(pts.sum(1))
    return np.roll(pts, -i, axis=0)


def detect_page(img: np.ndarray) -> tuple[np.ndarray | None, float]:
    """Find the document quadrilateral. Returns (quad in img coords, confidence)."""
    H, W = img.shape[:2]
    s = 1000.0 / max(H, W)
    small = cv2.resize(img, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    L = cv2.GaussianBlur(lab[..., 0], (0, 0), 2)
    sat = cv2.GaussianBlur(hsv[..., 1], (0, 0), 2)
    # paper: bright and weakly saturated relative to the scene
    score = L.astype(np.float32) - 0.8 * sat.astype(np.float32)
    score = cv2.normalize(score, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, th = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    n, lab_, stats, _ = cv2.connectedComponentsWithStats(th)
    if n <= 1:
        return None, 0.0
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = np.where(lab_ == k, 255, 0).astype(np.uint8)
    # fill holes (text, graphics inside the page)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros_like(mask)
    cv2.drawContours(mask, [c], -1, 255, -1)
    area_frac = cv2.contourArea(c) / float(mask.size)
    if area_frac < 0.15:
        return None, 0.0
    hull = cv2.convexHull(c)
    quad = None
    peri = cv2.arcLength(hull, True)
    for eps in (0.01, 0.02, 0.03, 0.05, 0.08):
        ap = cv2.approxPolyDP(hull, eps * peri, True)
        if len(ap) == 4:
            quad = ap.reshape(4, 2).astype(np.float32)
            break
    if quad is None:
        rect = cv2.minAreaRect(hull)
        quad = cv2.boxPoints(rect).astype(np.float32)
        conf = 0.4
    else:
        conf = 0.9
    quad = _order(quad)
    quad = _refine_corners(mask, quad)
    # does the region touch most of the image border? then it is a scan / tight crop
    margin = 0.012 * max(mask.shape)
    near = [(p[0] < margin or p[1] < margin or p[0] > mask.shape[1] - margin or p[1] > mask.shape[0] - margin) for p in quad]
    if sum(near) >= 3 or area_frac > 0.93:
        return None, 0.0
    # contour should be well explained by the quad
    qarea = cv2.contourArea(quad)
    if qarea <= 0 or abs(qarea - cv2.contourArea(c)) / qarea > 0.08:
        conf = min(conf, 0.5)
    return quad / s, conf


def _page_size(q: np.ndarray, max_side: int) -> tuple[int, int, str | None]:
    wt, wb = np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3])
    hl, hr = np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1])
    w, h = max(wt, wb), max(hl, hr)
    portrait = h >= w
    aspect = max(h, w) / min(h, w)
    snapped = None
    for name, a in STANDARD_ASPECTS.items():
        if a and abs(aspect - a) / a < 0.06:
            snapped = name
            long_side = max(h, w, min(h, w) * a)
            short_side = long_side / a
            h, w = (long_side, short_side) if portrait else (short_side, long_side)
            break
    sc = min(1.0, max_side / max(w, h))
    return int(round(w * sc)), int(round(h * sc)), snapped


# ---------------------------------------------------------------- orientation & skew

def _text_angles(models, img: np.ndarray):
    det = models["det"]
    boxes = det(img, limit_side=1280)
    angs, sizes = [], []
    for b in boxes:
        q = b.quad
        e1, e2 = q[1] - q[0], q[3] - q[0]
        l1, l2 = np.linalg.norm(e1), np.linalg.norm(e2)
        if max(l1, l2) < 2.0 * min(l1, l2):  # too square to say
            continue
        e = e1 if l1 >= l2 else e2
        a = np.degrees(np.arctan2(e[1], e[0]))
        angs.append(((a + 90) % 180) - 90)
        sizes.append(max(l1, l2))
    return boxes, np.array(angs), np.array(sizes)


def detect_orientation(models, img: np.ndarray) -> tuple[int, float, float]:
    """Return (rotation_deg in {0,90,180,270} clockwise to apply, confidence, skew_deg)."""
    from .ocr.ppocr import crop_quad

    boxes, angs, sizes = _text_angles(models, img)
    if len(angs) < 3:
        return 0, 0.0, 0.0
    vertical = np.abs(angs) > 45
    rot = 0
    if np.average(vertical, weights=sizes) > 0.5:
        rot = 90
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        boxes, angs, sizes = _text_angles(models, img)
    # 0 vs 180 using the text-line classifier on the longest lines
    order = np.argsort([-np.linalg.norm(b.quad[1] - b.quad[0]) for b in boxes])[:40]
    crops = [crop_quad(img, boxes[i].quad)[0] for i in order]
    votes = models["cls"](crops)
    flips = sum(c for a, c in votes if a == 180)
    keeps = sum(c for a, c in votes if a == 0)
    conf = abs(flips - keeps) / max(1e-6, flips + keeps)
    if flips > keeps:
        rot = (rot + 180) % 360
    horiz = angs[np.abs(angs) <= 20]
    skew = float(np.median(horiz)) if len(horiz) >= 3 else 0.0
    return rot, float(conf), skew


def _rot_matrix(rot: int, w: int, h: int) -> tuple[np.ndarray, tuple[int, int]]:
    if rot == 90:
        return np.array([[0, -1, h - 1], [1, 0, 0], [0, 0, 1]], np.float64), (h, w)
    if rot == 180:
        return np.array([[-1, 0, w - 1], [0, -1, h - 1], [0, 0, 1]], np.float64), (w, h)
    if rot == 270:
        return np.array([[0, 1, 0], [-1, 0, w - 1], [0, 0, 1]], np.float64), (h, w)
    return np.eye(3), (w, h)


# ---------------------------------------------------------------- enhancement

def estimate_noise(gray: np.ndarray) -> float:
    lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
    return float(np.median(np.abs(lap)) * 1.4826 / 2.0)


def flatten_illumination(img: np.ndarray, text_px: float = 24.0) -> np.ndarray:
    """Divide out the paper background (shadows, uneven lighting, colour cast)."""
    k = int(max(15, text_px * 2.2)) | 1
    small_s = 0.5
    sm = cv2.resize(img, None, fx=small_s, fy=small_s, interpolation=cv2.INTER_AREA)
    kk = max(7, int(k * small_s)) | 1
    bg = cv2.morphologyEx(sm, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk)))
    bg = cv2.medianBlur(bg, min(255, (kk * 2) | 1) if kk * 2 < 256 else 255)
    bg = cv2.GaussianBlur(bg, (0, 0), kk / 3)
    bg = cv2.resize(bg, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    out = img.astype(np.float32) / np.maximum(bg, 1.0) * 245.0
    return np.clip(out, 0, 255).astype(np.uint8)


def enhance_for_ocr(canvas: np.ndarray, text_px: float = 24.0) -> tuple[np.ndarray, list[dict]]:
    steps = []
    flat = flatten_illumination(canvas, text_px)
    steps.append({"step": "illumination_flattening", "kernel": int(text_px * 2.2)})
    gray = cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY)
    noise = estimate_noise(gray)
    if noise > 5.0:
        flat = cv2.fastNlMeansDenoisingColored(flat, None, min(10, noise), min(10, noise), 5, 15)
        steps.append({"step": "denoise", "sigma": round(noise, 2)})
    # contrast normalisation: stretch ink to black, paper to white
    lab = cv2.cvtColor(flat, cv2.COLOR_BGR2LAB)
    L = lab[..., 0].astype(np.float32)
    lo, hi = np.percentile(L, 0.5), np.percentile(L, 99.0)
    if hi - lo > 20:
        L = np.clip((L - lo) / (hi - lo) * 255.0, 0, 255)
        lab[..., 0] = L.astype(np.uint8)
        flat = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        steps.append({"step": "contrast_stretch", "lo": float(lo), "hi": float(hi)})
    # mild blur compensation
    sharp = cv2.Laplacian(cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    if sharp < 400:
        bl = cv2.GaussianBlur(flat, (0, 0), 1.2)
        flat = cv2.addWeighted(flat, 1.6, bl, -0.6, 0)
        steps.append({"step": "unsharp_mask", "laplacian_var": round(float(sharp), 1)})
    return flat, steps


# ---------------------------------------------------------------- main entry

def prepare_page(img: np.ndarray, models, is_photo: bool, max_side: int = 3600) -> Prepared:
    steps: list[dict] = []
    warnings: list[str] = []
    h0, w0 = img.shape[:2]
    T = np.eye(3)
    canvas = img
    quad = None
    if is_photo:
        quad, qconf = detect_page(img)
        if quad is not None and qconf >= 0.5:
            W, Hh, snapped = _page_size(quad, max_side)
            dst = np.float32([[0, 0], [W - 1, 0], [W - 1, Hh - 1], [0, Hh - 1]])
            P = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
            canvas = cv2.warpPerspective(img, P, (W, Hh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            T = P.astype(np.float64) @ T
            steps.append({"step": "perspective_correction", "quad": quad.round(1).tolist(), "size": [W, Hh],
                          "paper": snapped, "confidence": qconf})
        else:
            steps.append({"step": "perspective_correction", "skipped": "no page boundary found"})
    # cap working resolution (huge inputs)
    h, w = canvas.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        canvas = cv2.resize(canvas, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        T = np.diag([s, s, 1.0]) @ T
        steps.append({"step": "downscale", "factor": round(s, 4)})
        warnings.append(f"Large image reduced to {canvas.shape[1]}x{canvas.shape[0]} px for editing.")
    # orientation + skew from text lines
    if models is not None:
        probe = canvas
        ps = min(1.0, 1600 / max(canvas.shape[:2]))
        if ps < 1:
            probe = cv2.resize(canvas, None, fx=ps, fy=ps, interpolation=cv2.INTER_AREA)
        probe = flatten_illumination(probe, 16)
        rot, rconf, skew = detect_orientation(models, probe)
        if rot:
            h, w = canvas.shape[:2]
            R, (nw, nh) = _rot_matrix(rot, w, h)
            canvas = cv2.warpAffine(canvas, R[:2], (nw, nh), flags=cv2.INTER_NEAREST)
            T = R @ T
            steps.append({"step": "orientation", "rotate_cw": rot, "confidence": round(rconf, 3)})
        if abs(skew) > 0.25:
            h, w = canvas.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), skew, 1.0)
            canvas = cv2.warpAffine(canvas, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            T = np.vstack([M, [0, 0, 1]]) @ T
            steps.append({"step": "deskew", "degrees": round(skew, 3)})
    ocr_input, esteps = enhance_for_ocr(canvas)
    steps += esteps
    return Prepared(canvas=canvas, ocr_input=ocr_input, transform=T, steps=steps,
                    page_quad=None if quad is None else quad.tolist(), warnings=warnings)
