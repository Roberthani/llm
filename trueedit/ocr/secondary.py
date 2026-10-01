"""Secondary OCR opinions used on low-confidence or ambiguous lines.

* PP-OCRv4 recogniser (different weights from the primary PP-OCRv5)
* Tesseract 5 LSTM (independent architecture)
"""
from __future__ import annotations

import shutil
import unicodedata

import cv2
import numpy as np

_TESS = shutil.which("tesseract")


def tesseract_available() -> bool:
    return _TESS is not None


def tesseract_line(crop: np.ndarray, timeout: float = 8.0) -> tuple[str, float] | None:
    if not _TESS:
        return None
    import pytesseract

    h, w = crop.shape[:2]
    s = max(1.0, 48.0 / max(1, h))
    img = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC) if s > 1 else crop
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    pad = int(0.3 * gray.shape[0])
    gray = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    try:
        d = pytesseract.image_to_data(gray, config="--psm 7 --oem 1", output_type=pytesseract.Output.DICT,
                                      timeout=timeout)
    except Exception:
        return None
    words, confs = [], []
    for t, c in zip(d["text"], d["conf"]):
        t = (t or "").strip()
        try:
            c = float(c)
        except Exception:
            c = -1
        if t and c >= 0:
            words.append(t)
            confs.append(c / 100.0)
    if not words:
        return "", 0.0
    return unicodedata.normalize("NFKC", " ".join(words)), float(np.mean(confs))
