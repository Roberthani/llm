"""Gates 4, 5, 6 — layout preservation, real edits, pixel-diff regression.

Each edit is rendered through the production API. Assertions:
  * the edit changed pixels, and the new text reads back correctly with the OCR engine
  * the old text is gone
  * pixels outside the intended region (edited text box + replacement box + words that the
    edit legitimately moved along its line) are bit-identical
  * logo, barcode, table rulings and every other word are bit-identical
"""
import base64

import cv2
import numpy as np
import pytest

from helpers import RESULTS, save_result
from helpers import changed_mask, find_line, find_word, grow, read_text, save_diff
from trueedit import storage as S
from trueedit.render import composite


def page_state(client, pid, n=0):
    A = client.get(f"/api/projects/{pid}/pages/{n}/analysis").json()
    canvas = S.load_image(S.page_dir(pid, n) / "canvas.png").copy()
    return A, canvas


def apply_via_api(client, pid, edits, n=0):
    r = client.post(f"/api/projects/{pid}/pages/{n}/render", json={"edits": edits})
    assert r.status_code == 200, r.text
    _, canvas = page_state(client, pid, n)
    out = canvas.copy()
    for p in r.json()["patches"]:
        if "png" not in p:
            continue
        rgba = cv2.imdecode(np.frombuffer(base64.b64decode(p["png"]), np.uint8), cv2.IMREAD_UNCHANGED)
        m = rgba[..., 3] > 0
        reg = out[p["y"]:p["y"] + p["h"], p["x"]:p["x"] + p["w"]]
        reg[m] = rgba[..., :3][m]
    return canvas, out, r.json()["patches"]


def allowed_mask(shape, A, edits, patches):
    H, W = shape[:2]
    allow = np.zeros((H, W), bool)
    words = {w["id"]: w for r in A["regions"] for w in r["words"]}
    for e, p in zip(edits, patches):
        sig = (p["info"].get("style") or {}).get("sigma", 1.0) or 1.0
        mg = int(3 + 2 * sig)
        for b in (p["info"].get("effective_bbox"), p["info"].get("new_text_bbox")):
            if b:
                x0, y0, x1, y1 = grow(b, mg, W, H)
                allow[y0:y1, x0:x1] = True
        for mv in p["info"].get("moved", []):
            b = words[mv["id"]]["bbox"]
            for dx in (0, mv["dx"]):
                x0, y0, x1, y1 = grow([b[0] + dx, b[1], b[2] + dx, b[3]], mg, W, H)
                allow[y0:y1, x0:x1] = True
    return allow


def protected_boxes(A, edits, patches):
    """Everything that must stay identical: graphics, barcode, rulings, and all untouched words."""
    touched = set()
    for e, p in zip(edits, patches):
        for t in e["target_ids"]:
            touched.add(t)
            for r in A["regions"]:
                if r["id"] == t:
                    touched |= {w["id"] for w in r["words"]}
        touched |= {m["id"] for m in p["info"].get("moved", [])}
    boxes = [("logo", g.get("group_box", g["box"])) for g in A["graphics"]] + [("barcode", b["box"]) for b in A["barcodes"]]
    boxes += [("word:" + w["text"], w["bbox"]) for r in A["regions"] for w in r["words"] if w["id"] not in touched
              and r["id"] not in touched]
    return boxes


CASES = {
    # name: (fixture, how to find the target, replacement)
    "A_first_name": ("order_photo.jpg", ("word", "Daniel", None), "Michael"),
    "B_last_name": ("order_photo.jpg", ("word", "Kowalski", None), "Johnson"),
    "C_item_description": ("order_photo.jpg", ("line", "Organic Potting Soil 50 L", None), "Premium Garden Compost 40 L"),
    "D_delete_value": ("order_photo.jpg", ("word", "N/A", "Instructions"), ""),
    "E_small_field_NA": ("order_photo.jpg", ("word", "N/A", "Email"), "None"),
    "A_first_name_invoice": ("invoice_photo.jpg", ("word", "Priya", None), "Margaret"),
    "B_last_name_invoice": ("invoice_photo.jpg", ("word", "Raman", None), "Okafor"),
    "C_item_invoice": ("invoice_photo.jpg", ("line", "Synthetic Oil Filter", None), "Cabin Air Filter"),
    "E_small_field_invoice": ("invoice_photo.jpg", ("word", "N/A", None), "90 days"),
}


def make_edit(A, how, text):
    kind, t, ctx = how
    if kind == "word":
        r, w = find_word(A, t, ctx)
        return {"id": "t-" + w["id"], "page": 0, "bbox": w["bbox"], "text": text, "target_ids": [w["id"]],
                "original_text": w["text"], "source": "ocr"}, w["text"]
    r = find_line(A, t)
    return {"id": "t-" + r["id"], "page": 0, "bbox": r["bbox"], "text": text, "target_ids": [r["id"]],
            "original_text": r["text"], "source": "ocr"}, r["text"]


RESULTS_ROWS = {}


@pytest.mark.parametrize("name", list(CASES))
def test_single_edit(client, project, name):
    fixture, how, new = CASES[name]
    pid = project(fixture)
    A, canvas = page_state(client, pid)
    edit, old = make_edit(A, how, new)
    before, after, patches = apply_via_api(client, pid, [edit])
    d = changed_mask(before, after)
    allow = allowed_mask(before.shape, A, [edit], patches)
    outside = int((d & ~allow).sum())
    info = patches[0]["info"]
    save_diff(RESULTS / f"edit_{name}.png", before, after, info.get("changed_bbox") or edit["bbox"])
    row = {"changed_pixels": int(d.sum()), "changed_fraction": float(d.mean()), "outside_intended": outside,
           "changed_bbox": info.get("changed_bbox"), "style": info.get("style"), "moved": info.get("moved", []),
           "warnings": info.get("warnings")}
    # 1) something changed, nothing outside the intended region
    assert d.sum() > 50
    assert outside == 0, f"{outside} pixels changed outside the edited region"
    # 2) protected content is bit-identical
    for label, b in protected_boxes(A, [edit], patches):
        x0, y0, x1, y1 = b
        assert not d[y0:y1, x0:x1].any(), f"{label} was modified"
    # 3) the result reads correctly; the old text is gone
    nb = info.get("new_text_bbox")
    region = nb or edit["bbox"]
    got = read_text(after, region)
    row["reread"] = got
    if new:
        assert got.replace(" ", "") == new.replace(" ", ""), f"re-read {got!r} != {new!r}"
    else:
        x0, y0, x1, y1 = edit["bbox"]
        patch = cv2.cvtColor(after[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(float)
        assert patch.std() < 8, "deleted area should be plain background"
        assert read_text(after, edit["bbox"], pad=2) == ""
    # moved words keep their exact appearance (only translated)
    for mv in info.get("moved", []):
        w = next(w for r in A["regions"] for w in r["words"] if w["id"] == mv["id"])
        x0, y0, x1, y1 = w["bbox"]
        a = before[y0:y1, x0:x1].astype(int)
        b = after[y0:y1, x0 + mv["dx"]:x1 + mv["dx"]].astype(int)
        assert np.abs(a - b).mean() < 6, "moved word changed appearance"
        assert read_text(after, [x0 + mv["dx"], y0, x1 + mv["dx"], y1]) == w["text"]
    save_result(f"edit_{name}.json", row)


def test_combined_edits_sequence(client, project):
    """First name + last name + item description + small field together (reflow interacts)."""
    pid = project("order_photo.jpg")
    A, canvas = page_state(client, pid)
    eds = [make_edit(A, CASES[n][1], CASES[n][2])[0] for n in
           ("A_first_name", "B_last_name", "C_item_description", "E_small_field_NA")]
    before, after, patches = apply_via_api(client, pid, eds)
    d = changed_mask(before, after)
    allow = allowed_mask(before.shape, A, eds, patches)
    assert int((d & ~allow).sum()) == 0
    name_box = [min(p["changed_bbox"][0] for p in patches[:2]), min(p["changed_bbox"][1] for p in patches[:2]),
                max(p["changed_bbox"][2] for p in patches[:2]), max(p["changed_bbox"][3] for p in patches[:2])]
    assert read_text(after, name_box).replace(" ", "") == "MichaelJohnson"
    # natural word spacing kept: the gap between first and last name equals the original gap
    _, wa = find_word(A, "Daniel")
    _, wb = find_word(A, "Kowalski")

    def word_gap(img):
        y0, y1 = wa["bbox"][1], wa["bbox"][3]
        x0, x1 = wa["bbox"][0] - 5, max(p["changed_bbox"][2] for p in patches[:2]) + 5
        g = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(int)
        ink = (g < np.percentile(g, 95) - 60).any(0)
        cols = np.nonzero(ink)[0]
        gaps = np.diff(cols)
        return int(gaps.max()) - 1

    g0, g1 = word_gap(before), word_gap(after)
    assert abs(g1 - g0) <= 3, (g0, g1)
    for label, b in protected_boxes(A, eds, patches):
        x0, y0, x1, y1 = b
        assert not d[y0:y1, x0:x1].any(), f"{label} was modified"
    save_diff(RESULTS / "edit_combined_full.png", before, after)
    save_result("edit_combined.json", {"changed_pixels": int(d.sum()), "changed_fraction": float(d.mean())})


def test_edit_longer_than_space_condenses_without_collision(client, project):
    pid = project("order_photo.jpg")
    A, _ = page_state(client, pid)
    # the quantity "6" (right-aligned in the narrow QTY column)
    qty_col = next(r["table"]["col"] for r in A["regions"] if r["text"] == "QTY")
    r, w = next((r, w) for r in A["regions"] for w in r["words"]
                if w["text"] == "6" and r["table"] and r["table"]["col"] == qty_col)
    e = {"id": "q", "page": 0, "bbox": w["bbox"], "text": "16", "target_ids": [w["id"]], "source": "ocr"}
    before, after, patches = apply_via_api(client, pid, [e])
    d = changed_mask(before, after)
    cell = r["table"]["cell"]
    ys, xs = np.nonzero(d)
    assert xs.min() >= cell[0] and xs.max() < cell[2], "edit must stay inside its table cell"
    assert read_text(after, patches[0]["info"]["new_text_bbox"]) == "16"


def test_no_edits_means_identical(client, project):
    """Gate 4: the page is never reconstructed — without edits the output is the canvas itself."""
    pid = project("order_photo.jpg")
    _, canvas = page_state(client, pid)
    r = client.get(f"/api/projects/{pid}/pages/0/edited?fmt=png")
    out = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert np.array_equal(out, canvas)
    f = client.get(f"/api/projects/{pid}/pages/0/fidelity").json()
    assert f["changed_pixels"] == 0


def test_canvas_is_only_geometrically_resampled(client, project):
    """Gate 4: the working page is the original photo, warped — never re-toned or redrawn."""
    pid = project("order_photo.jpg")
    m = client.get(f"/api/projects/{pid}").json()["pages"][0]
    orig = S.load_image(S.page_dir(pid, 0) / "original.png")
    canvas = S.load_image(S.page_dir(pid, 0) / "canvas.png")
    T = np.array(m["transform"])
    rewarp = cv2.warpPerspective(orig, T, (canvas.shape[1], canvas.shape[0]), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
    diff = np.abs(rewarp.astype(int) - canvas.astype(int))
    assert diff.mean() < 1.0 and np.percentile(diff, 99.5) <= 8


def test_fidelity_endpoint_reports_locality(client, project):
    pid = project("order_photo.jpg")
    A, _ = page_state(client, pid)
    e, _ = make_edit(A, CASES["A_first_name"][1], "Michael")
    client.put(f"/api/projects/{pid}/edits", json={"edits": [e]})
    f = client.get(f"/api/projects/{pid}/pages/0/fidelity").json()
    assert f["edits"] == 1 and f["changed_pixels"] > 0 and f["outside_intended_pixels"] == 0
    assert f["changed_fraction"] < 0.002
    r = client.get(f"/api/projects/{pid}/pages/0/diff.png")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    save_result("fidelity_first_name.json", f)
