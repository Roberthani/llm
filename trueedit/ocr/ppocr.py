"""PaddleOCR (PP-OCR) inference on ONNX Runtime.

Primary engine: PP-OCRv5 detection + recognition (models shipped in the `onnxocr` wheel).
Secondary recogniser: PP-OCRv4 recognition (models shipped in `rapidocr_onnxruntime`).

Beyond plain text, recognition returns per-character confidences and horizontal positions
(decoded from CTC time steps) so lines can be split into word boxes and individual
uncertain characters can be flagged.
"""
from __future__ import annotations

import importlib.util
import math
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

_LOCK = threading.Lock()
_CACHE: dict[str, object] = {}


def _pkg_dir(name: str) -> Path | None:
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.origin:
        return None
    return Path(spec.origin).parent


def _session(path: Path):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = int(os.environ.get("TRUEEDIT_OCR_THREADS", max(1, (os.cpu_count() or 2))))
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


class OCRUnavailable(RuntimeError):
    pass


@dataclass
class DetBox:
    quad: np.ndarray  # (4,2) float32, clockwise from top-left
    score: float


@dataclass
class RecResult:
    text: str
    conf: float
    chars: list[dict] = field(default_factory=list)  # {c, conf, x0, x1} in crop px


# ---------------------------------------------------------------- detection

class Detector:
    def __init__(self, model: Path, limit_side: int = 2400, thresh=0.3, box_thresh=0.55, unclip=1.6):
        self.sess = _session(model)
        self.inp = self.sess.get_inputs()[0].name
        self.limit_side = limit_side
        self.thresh, self.box_thresh, self.unclip = thresh, box_thresh, unclip

    def __call__(self, img: np.ndarray, limit_side: int | None = None) -> list[DetBox]:
        h, w = img.shape[:2]
        lim = limit_side or self.limit_side
        r = min(1.0, lim / max(h, w))
        nh = max(32, int(round(h * r / 32)) * 32)
        nw = max(32, int(round(w * r / 32)) * 32)
        x = cv2.resize(img, (nw, nh)).astype(np.float32) / 255.0
        x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        x = x.transpose(2, 0, 1)[None]
        pred = self.sess.run(None, {self.inp: x})[0][0, 0]
        return self._boxes(pred, w / nw, h / nh)

    def _boxes(self, pred: np.ndarray, sx: float, sy: float) -> list[DetBox]:
        bitmap = (pred > self.thresh).astype(np.uint8)
        contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in contours:
            if len(c) < 4:
                continue
            (cx, cy), (rw, rh), ang = cv2.minAreaRect(c)
            if min(rw, rh) < 2:
                continue
            x0, y0, bw, bh = cv2.boundingRect(c)
            m = np.zeros((bh, bw), np.uint8)
            cv2.fillPoly(m, [c - [x0, y0]], 1)
            score = float(cv2.mean(pred[y0:y0 + bh, x0:x0 + bw], m)[0])
            if score < self.box_thresh:
                continue
            area = cv2.contourArea(c)
            peri = cv2.arcLength(c, True)
            if peri <= 0:
                continue
            d = area * self.unclip / peri
            rect = ((cx, cy), (rw + 2 * d, rh + 2 * d), ang)
            if min(rect[1]) < 5:
                continue
            q = cv2.boxPoints(rect)
            q[:, 0] *= sx
            q[:, 1] *= sy
            out.append(DetBox(order_quad(q), score))
        return out


def order_quad(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float32)
    s = pts.sum(1)
    d = np.diff(pts, axis=1).ravel()
    tl, br = pts[np.argmin(s)], pts[np.argmax(s)]
    tr, bl = pts[np.argmin(d)], pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], np.float32)


def crop_quad(img: np.ndarray, quad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rectify a quad to an upright crop. Returns (crop, M) where M maps crop->image."""
    q = order_quad(quad)
    w = int(round(max(np.linalg.norm(q[0] - q[1]), np.linalg.norm(q[3] - q[2]))))
    h = int(round(max(np.linalg.norm(q[0] - q[3]), np.linalg.norm(q[1] - q[2]))))
    w, h = max(w, 2), max(h, 2)
    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(q, dst)
    crop = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return crop, np.linalg.inv(M)


# ---------------------------------------------------------------- recognition

class Recognizer:
    def __init__(self, model: Path, charset: list[str], img_h: int = 48, batch: int = 8):
        self.sess = _session(model)
        self.inp = self.sess.get_inputs()[0].name
        self.chars = ["<blank>"] + charset
        self.img_h = img_h
        self.batch = batch

    def _prep(self, img: np.ndarray, max_ratio: float):
        h, w = img.shape[:2]
        tw = int(math.ceil(self.img_h * max_ratio))
        rw = min(tw, max(1, int(math.ceil(self.img_h * w / h))))
        x = cv2.resize(img, (rw, self.img_h)).astype(np.float32)
        x = (x / 255.0 - 0.5) / 0.5
        out = np.zeros((3, self.img_h, tw), np.float32)
        out[:, :, :rw] = x.transpose(2, 0, 1)
        return out, rw, tw

    def __call__(self, crops: list[np.ndarray]) -> list[RecResult]:
        res: list[RecResult | None] = [None] * len(crops)
        order = np.argsort([c.shape[1] / max(1, c.shape[0]) for c in crops])
        for b0 in range(0, len(crops), self.batch):
            idx = order[b0:b0 + self.batch]
            max_ratio = max(320 / 48, max(crops[i].shape[1] / max(1, crops[i].shape[0]) for i in idx))
            batch, meta = [], []
            for i in idx:
                x, rw, tw = self._prep(crops[i], max_ratio)
                batch.append(x)
                meta.append((rw, tw))
            probs = self.sess.run(None, {self.inp: np.stack(batch)})[0]
            for k, i in enumerate(idx):
                rw, tw = meta[k]
                res[i] = self._decode(probs[k], crops[i].shape[1] / rw, tw)
        return res  # type: ignore[return-value]

    def _decode(self, p: np.ndarray, sx: float, tw: int) -> RecResult:
        T = p.shape[0]
        step = tw / T
        am = p.argmax(1)
        mx = p.max(1)
        chars = []
        t = 0
        while t < T:
            k = am[t]
            t1 = t
            while t1 + 1 < T and am[t1 + 1] == k:
                t1 += 1
            if k != 0 and k < len(self.chars):
                conf = float(mx[t:t1 + 1].max())
                chars.append({"c": self.chars[k], "conf": conf,
                              "x0": t * step * sx, "x1": (t1 + 1) * step * sx})
            t = t1 + 1
        text = "".join(c["c"] for c in chars)
        conf = float(np.mean([c["conf"] for c in chars])) if chars else 0.0
        return RecResult(text, conf, chars)


class Classifier:
    """Text-line 0/180 degree classifier."""

    def __init__(self, model: Path):
        self.sess = _session(model)
        self.inp = self.sess.get_inputs()[0].name

    def __call__(self, crops: list[np.ndarray]) -> list[tuple[int, float]]:
        if not crops:
            return []
        xs = []
        for c in crops:
            h, w = c.shape[:2]
            rw = min(192, max(1, int(math.ceil(48 * w / h))))
            x = cv2.resize(c, (rw, 48)).astype(np.float32)
            x = (x / 255.0 - 0.5) / 0.5
            o = np.zeros((3, 48, 192), np.float32)
            o[:, :, :rw] = x.transpose(2, 0, 1)
            xs.append(o)
        out = []
        for b in range(0, len(xs), 16):
            p = self.sess.run(None, {self.inp: np.stack(xs[b:b + 16])})[0]
            out += [(0 if r[0] >= r[1] else 180, float(r.max())) for r in p]
        return out


# ---------------------------------------------------------------- model loading

def load_models():
    """Load and cache all models. Raises OCRUnavailable with a clear message on failure."""
    with _LOCK:
        if "models" in _CACHE:
            return _CACHE["models"]
        if os.environ.get("TRUEEDIT_FAULT") == "ocr_unavailable":
            raise OCRUnavailable("OCR engine failed to start (fault injected for testing)")
        v5 = _pkg_dir("onnxocr")
        if v5 is None:
            raise OCRUnavailable("PP-OCRv5 models not installed (pip install onnxocr)")
        base = v5 / "models" / "ppocrv5"
        try:
            charset = (base / "ppocrv5_dict.txt").read_text(encoding="utf-8").splitlines()
            charset = [c for c in charset] + [" "]
            det = Detector(base / "det" / "det.onnx")
            rec = Recognizer(base / "rec" / "rec.onnx", charset)
            cls = Classifier(base / "cls" / "cls.onnx")
        except Exception as e:  # pragma: no cover - depends on install
            raise OCRUnavailable(f"Could not load PP-OCRv5 models: {e}") from e
        rec4 = None
        v4 = _pkg_dir("rapidocr_onnxruntime")
        if v4 is not None:
            try:
                p = v4 / "models" / "ch_PP-OCRv4_rec_infer.onnx"
                sess = _session(p)
                meta = sess.get_modelmeta().custom_metadata_map
                cs = meta.get("character", "").splitlines()
                if cs:
                    rec4 = Recognizer(p, cs + [" "])
            except Exception:
                rec4 = None
        models = {"det": det, "rec": rec, "cls": cls, "rec4": rec4}
        _CACHE["models"] = models
        return models
