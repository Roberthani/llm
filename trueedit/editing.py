"""Resolve user edits against the page analysis and render them as local patches."""
from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict

import numpy as np

from . import render as R


class EditError(ValueError):
    pass


def _ov(a, b) -> float:
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return iy / max(1, min(a[3] - a[1], b[3] - b[1]))


def _inter(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _contains(outer, inner, tol=2) -> bool:
    return (inner[0] >= outer[0] - tol and inner[1] >= outer[1] - tol and
            inner[2] <= outer[2] + tol and inner[3] <= outer[3] + tol)


class PageIndex:
    def __init__(self, analysis: dict):
        self.a = analysis
        self.lines = {r["id"]: r for r in analysis.get("regions", [])}
        self.words = {}
        self.word_line = {}
        for r in analysis.get("regions", []):
            for w in r["words"]:
                self.words[w["id"]] = w
                self.word_line[w["id"]] = r["id"]
        self.rule_boxes = [s["box"] for s in analysis.get("rulings", [])]
        self.protected = [g.get("box") for g in analysis.get("graphics", [])] + \
                         [b["box"] for b in analysis.get("barcodes", [])]
        W, H = analysis.get("size", [0, 0])
        self.W, self.H = W, H

    def target_words(self, edit) -> list[dict]:
        ids = edit.get("target_ids") or []
        out = []
        for i in ids:
            if i in self.words:
                out.append(self.words[i])
            elif i in self.lines:
                out += self.lines[i]["words"]
        return out

    def parent_line(self, edit):
        for i in edit.get("target_ids") or []:
            if i in self.lines:
                return self.lines[i]
            if i in self.word_line:
                return self.lines[self.word_line[i]]
        return None


def build_context(canvas: np.ndarray, idx: PageIndex, edit: dict, fit_key: str | None,
                  touched: dict | None = None) -> dict:
    """`canvas` is the untouched page (used for style fitting); `touched` maps region/word ids
    to their current bbox after earlier edits (None = deleted)."""
    touched = touched or {}
    bbox = [int(v) for v in edit["bbox"]]
    tw = idx.target_words(edit)
    tids = {w["id"] for w in tw}
    line = idx.parent_line(edit)
    h = max(4, bbox[3] - bbox[1])
    cur = lambda w: touched.get(w["id"], w["bbox"]) if w["id"] in touched else w["bbox"]
    line_ids = {w["id"] for w in line["words"]} if line else set()
    style_ov = edit.get("style") or {}
    align = style_ov.get("align") or (line["style"].get("align", "left") if line else "left")
    reflow = edit.get("source") != "manual" and not style_ov.get("no_reflow") and line is not None
    followers = []
    if reflow:
        others = [w for w in line["words"] if w["id"] not in tids and cur(w) is not None]
        if align == "right":
            followers = [w for w in others if cur(w)[2] <= bbox[0] + 1]
            followers.sort(key=lambda w: -cur(w)[2])
            follow_dir = -1
        else:
            followers = [w for w in others if cur(w)[0] >= bbox[2] - 1]
            followers.sort(key=lambda w: cur(w)[0])
            follow_dir = 1
    fids = {w["id"] for w in followers}
    obstacles, nonline = [], []
    for wid, w in idx.words.items():
        if wid in tids or wid in fids:
            continue
        b = cur(w)
        if b is None:
            continue
        if _contains(bbox, b, 0) and edit.get("source") == "manual":
            continue  # manual region deliberately covers this word
        obstacles.append(b)
        if wid not in line_ids:
            nonline.append(b)
    obstacles += [b for b in idx.protected if b]
    nonline += [b for b in idx.protected if b]
    vrules = [b for b in idx.rule_boxes if (b[3] - b[1]) > (b[2] - b[0]) and _ov(b, bbox) > 0.5]
    gap = 0.35 * h

    def room(obs):
        left, right = 0.0, float(idx.W or canvas.shape[1])
        for b in [b for b in obs if _ov(b, bbox) > 0.3] + vrules:
            if b[2] <= bbox[0] + 1:
                left = max(left, b[2] + gap)
            elif b[0] >= bbox[2] - 1:
                right = min(right, b[0] - gap)
        if line and line.get("table"):
            c = line["table"]["cell"]
            left, right = max(left, c[0] + 0.25 * h), min(right, c[2] - 0.25 * h)
        return min(left, bbox[0]), max(right, bbox[2])

    line_room = room(nonline + [b for w in line["words"] if w["id"] not in tids and w["id"] not in fids
                                for b in [cur(w)] if b is not None] if line else obstacles)
    limits = room(obstacles + [cur(w) for w in followers])
    if followers:
        fb = [cur(w) for w in followers]
        if follow_dir > 0:
            need = (max(b[2] for b in fb) - min(b[0] for b in fb)) + max(0, fb[0][0] - bbox[2])
            limits = (limits[0], max(bbox[2], line_room[1] - need))
        else:
            need = (max(b[2] for b in fb) - min(b[0] for b in fb)) + max(0, bbox[0] - fb[0][2])
            limits = (min(bbox[0], line_room[0] + need), limits[1])
    ctx = {"obstacles": obstacles, "rule_boxes": idx.rule_boxes, "limits": limits, "line_room": line_room,
           "align": align, "followers": [{"id": w["id"], "bbox": cur(w)} for w in followers],
           "follow_dir": follow_dir if followers else 1, "reflow": reflow}
    if line and edit.get("source") != "manual":
        key = fit_key + ":" + line["id"] if fit_key else None
        hint = {"size_px": line["style"].get("font_size_px")}

        def obst_mask(keep_ids):
            obst = np.zeros(canvas.shape[:2], np.uint8)
            for wid, w in idx.words.items():
                if wid not in keep_ids:
                    b = w["bbox"]
                    obst[b[1]:b[3], b[0]:b[2]] = 1
            for b in idx.rule_boxes:
                obst[b[1]:b[3], b[0]:b[2]] = 1
            return obst

        if len(line["text"].replace(" ", "")) < 4:
            # too few glyphs to identify a typeface: borrow it from nearby text of the same size
            fam = _reference_family(canvas, idx, line, fit_key, obst_mask)
            if fam:
                hint["family"] = fam

        def do_fit():
            return R.fit_style(canvas, line["bbox"], line["text"], obst_mask(line_ids), hint,
                               parts=[w["bbox"] for w in line["words"]])

        lf = R.cached_fit(key, do_fit) if key else do_fit()
        ctx["fit"] = lf
        # refine weight and colour on the target word(s) themselves (lines can mix styles)
        if lf is not None and tw and len(tw) < len(line["words"]):
            ttext = " ".join(w["text"] for w in tw)
            if len(ttext.strip()) >= 2:
                tb = [min(w["bbox"][0] for w in tw), min(w["bbox"][1] for w in tw),
                      max(w["bbox"][2] for w in tw), max(w["bbox"][3] for w in tw)]
                wkey = (key + ":" + ",".join(sorted(tids))) if key else None
                long_word = len(ttext.replace(" ", "")) >= 4
                hint_w = {"size_px": lf.size_px} if long_word else {"family": lf.family, "size_px": lf.size_px}
                wf_fn = lambda: R.fit_style(canvas, tb, ttext, obst_mask(tids), hint_w)
                wf = R.cached_fit(wkey, wf_fn) if wkey else wf_fn()
                if wf is not None:
                    # a word may be set in a different font than the rest of its line (labels vs values)
                    fam = wf.family if (long_word and wf.family != lf.family and wf.score < 0.85 * lf.score) else lf.family
                    own = fam != lf.family
                    ctx["fit"] = R.StyleFit(fam, wf.bold if long_word else lf.bold,
                                            wf.size_px if own else lf.size_px,
                                            wf.hscale if (len(ttext) >= 5 or own) else lf.hscale,
                                            wf.sigma if own else lf.sigma, wf.color, lf.baseline, wf.score)
    elif edit.get("original_text"):
        ctx["fit"] = R.fit_style(canvas, bbox, edit["original_text"])
    return ctx


def _reference_family(canvas, idx: PageIndex, line: dict, fit_key, obst_mask):
    size = line["style"].get("font_size_px") or 0
    cy = (line["bbox"][1] + line["bbox"][3]) / 2
    cx = (line["bbox"][0] + line["bbox"][2]) / 2
    cands = []
    for r in idx.lines.values():
        if r is line or r.get("role") != "text" or len(r["text"].replace(" ", "")) < 6:
            continue
        rs = r["style"].get("font_size_px") or 0
        if not size or abs(rs - size) > 0.15 * size:
            continue
        b = r["bbox"]
        d = abs((b[1] + b[3]) / 2 - cy) * 3 + abs((b[0] + b[2]) / 2 - cx)
        cands.append((d, r))
    cands.sort(key=lambda t: t[0])
    votes: dict[str, float] = {}
    for _, r in cands[:3]:
        ids = {w["id"] for w in r["words"]}
        key = (fit_key + ":" + r["id"]) if fit_key else None
        fn = lambda r=r, ids=ids: R.fit_style(canvas, r["bbox"], r["text"], obst_mask(ids),
                                              {"size_px": r["style"].get("font_size_px")},
                                              parts=[w["bbox"] for w in r["words"]])
        f = R.cached_fit(key, fn) if key else fn()
        if f is not None:
            votes[f.family] = votes.get(f.family, 0) + 1.0 / (0.02 + f.score)
    return max(votes, key=votes.get) if votes else None


_PATCH_CACHE: "OrderedDict[str, R.Patch]" = OrderedDict()
_PC_LOCK = threading.Lock()


def _edit_key(e: dict) -> str:
    keep = {k: e.get(k) for k in ("id", "bbox", "text", "style", "target_ids", "source", "original_text", "erase_box")}
    return hashlib.sha1(json.dumps(keep, sort_keys=True).encode()).hexdigest()


def validate_edit(e: dict, W: int, H: int):
    b = e.get("bbox")
    if not isinstance(b, (list, tuple)) or len(b) != 4:
        raise EditError("edit needs bbox [x0,y0,x1,y1]")
    x0, y0, x1, y1 = [float(v) for v in b]
    if not (x1 > x0 and y1 > y0):
        raise EditError("empty edit region")
    if x1 <= 0 or y1 <= 0 or x0 >= W or y0 >= H:
        raise EditError("edit region is outside the page")
    if len(e.get("text", "")) > 500:
        raise EditError("replacement text too long")


def render_page_edits(canvas: np.ndarray, analysis: dict, edits: list[dict], cache_ns: str = "") -> list[R.Patch]:
    """Render edits in order. Each edit sees the result of earlier edits (positions included)."""
    idx = PageIndex(analysis)
    H, W = canvas.shape[:2]
    cur = canvas
    patches: list[R.Patch] = []
    touched: dict[str, list | None] = {}
    chain: list[tuple[list, str]] = []  # (window bbox, key) of applied patches
    for e in edits:
        validate_edit(e, W, H)
        e = dict(e)
        e["bbox"] = [int(round(v)) for v in e["bbox"]]
        tw = idx.target_words(e)
        # targets moved / replaced by an earlier edit: follow them to where they are now
        if tw and any(w["id"] in touched for w in tw):
            orig = _union([w["bbox"] for w in tw])
            now = _union([touched.get(w["id"], w["bbox"]) if w["id"] in touched else w["bbox"] for w in tw])
            if now is not None:
                same = all(abs(a - b) <= 3 for a, b in zip(e["bbox"], orig))
                e["bbox"] = now if same else _union([e["bbox"], now])
        k = _edit_key(e)
        deps = [kk for (bb, kk) in chain if _inter(bb, _grow(e["bbox"], 3 * (e["bbox"][3] - e["bbox"][1])))]
        key = hashlib.sha1((cache_ns + k + "|".join(deps)).encode()).hexdigest()
        with _PC_LOCK:
            p = _PATCH_CACHE.get(key)
            if p is not None:
                _PATCH_CACHE.move_to_end(key)
        if p is None:
            ctx = build_context(canvas, idx, e, cache_ns, touched)  # style is fitted on the untouched page
            p = R.render_edit(cur, e, ctx)
            if ctx.get("fit") is not None:
                p.info["fit"] = ctx["fit"].to_dict()
            p.info["effective_bbox"] = e["bbox"]
            with _PC_LOCK:
                _PATCH_CACHE[key] = p
                # bound memory: ~150 MB of cached patches
                while len(_PATCH_CACHE) > 1 and sum(q.rgb.nbytes + q.mask.nbytes for q in _PATCH_CACHE.values()) > 150e6:
                    _PATCH_CACHE.popitem(last=False)
        patches.append(p)
        cur = R.composite(cur, [p])
        # update positions of edited / moved words
        nb = p.info.get("new_text_bbox")
        for i, w in enumerate(tw):
            touched[w["id"]] = (nb if i == 0 else None) if nb else None
        for mv in p.info.get("moved", []):
            w = idx.words.get(mv["id"])
            if w is None:
                continue
            b = touched.get(mv["id"], w["bbox"]) or w["bbox"]
            touched[mv["id"]] = [b[0] + mv["dx"], b[1], b[2] + mv["dx"], b[3]]
        chain.append(([p.x, p.y, p.x + p.mask.shape[1], p.y + p.mask.shape[0]], key))
    return patches


def _union(boxes):
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return None
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def _grow(b, m):
    return [b[0] - m, b[1] - m, b[2] + m, b[3] + m]
