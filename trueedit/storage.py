"""On-disk project store. The uploaded file and the decoded original are never modified."""
from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

DATA_DIR = Path(os.environ.get("TRUEEDIT_DATA", Path(__file__).resolve().parents[1] / "data" / "projects"))
_ID = re.compile(r"^[a-f0-9]{12}$")
_LOCKS: dict[str, threading.RLock] = {}
_GL = threading.Lock()


def lock(pid: str) -> threading.RLock:
    with _GL:
        return _LOCKS.setdefault(pid, threading.RLock())


class NotFound(KeyError):
    pass


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def pdir(pid: str) -> Path:
    if not _ID.match(pid or ""):
        raise NotFound(pid)
    d = DATA_DIR / pid
    if not d.exists():
        raise NotFound(pid)
    return d


def create(pid: str) -> Path:
    d = DATA_DIR / pid
    (d / "pages").mkdir(parents=True, exist_ok=True)
    return d


def page_dir(pid: str, n: int) -> Path:
    d = pdir(pid) / "pages" / str(int(n))
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, separators=(",", ":"), default=_jdefault))
    os.replace(tmp, path)


def _jdefault(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def save_image(path: Path, img: np.ndarray, quality: int | None = None) -> None:
    ext = path.suffix.lower()
    params = [cv2.IMWRITE_PNG_COMPRESSION, 3] if ext == ".png" else [cv2.IMWRITE_JPEG_QUALITY, quality or 92]
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise IOError(f"could not encode {path.name}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(buf.tobytes())
    os.replace(tmp, path)


_IMG_CACHE: dict[str, tuple[float, np.ndarray]] = {}


def load_image(path: Path) -> np.ndarray:
    key = str(path)
    mt = path.stat().st_mtime
    c = _IMG_CACHE.get(key)
    if c and c[0] == mt:
        return c[1]
    img = cv2.imread(key, cv2.IMREAD_COLOR)
    if img is None:
        raise NotFound(key)
    if len(_IMG_CACHE) > 24:
        _IMG_CACHE.clear()
    _IMG_CACHE[key] = (mt, img)
    return img


def meta(pid: str) -> dict:
    m = read_json(pdir(pid) / "meta.json")
    if m is None:
        raise NotFound(pid)
    return m


def save_meta(pid: str, m: dict) -> None:
    m["updated"] = time.time()
    write_json(pdir(pid) / "meta.json", m)


def edits(pid: str) -> dict:
    return read_json(pdir(pid) / "edits.json", {"version": 0, "edits": [], "review": {}, "regions": []})


def save_edits(pid: str, doc: dict) -> dict:
    with lock(pid):
        cur = edits(pid)
        doc = {"version": int(cur.get("version", 0)) + 1, "edits": doc.get("edits", []),
               "review": doc.get("review", {}), "regions": doc.get("regions", []), "saved": time.time()}
        write_json(pdir(pid) / "edits.json", doc)
        return doc


def delete(pid: str) -> None:
    shutil.rmtree(pdir(pid), ignore_errors=True)


def cleanup(max_age_s: float = 7 * 86400) -> int:
    n = 0
    if not DATA_DIR.exists():
        return 0
    now = time.time()
    for d in DATA_DIR.iterdir():
        try:
            if d.is_dir() and now - d.stat().st_mtime > max_age_s:
                shutil.rmtree(d, ignore_errors=True)
                n += 1
        except OSError:
            pass
    return n
