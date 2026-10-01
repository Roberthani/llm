"""Browser end-to-end tests (Gates 1, 9, 10 and the editor features) — real server, real OCR, real Chromium."""
import json
import os
import re

import cv2
import numpy as np
import pymupdf
import pytest

from helpers import FX, RESULTS, read_text, save_result

pytestmark = pytest.mark.e2e


# ------------------------------------------------------------------ helpers

def open_and_upload(page, url, name, timeout=180_000):
    page.goto(url)
    page.wait_for_selector("#drop")
    page.set_input_files("#file", str(FX / name))
    page.wait_for_selector("#screen-editor:not([hidden])", timeout=timeout)
    page.wait_for_function("document.querySelectorAll('#region-layer .r').length > 0 || "
                           "(window.__trueedit.A[0] && window.__trueedit.A[0].regions.length === 0)")


def word_id(page, text, line_contains=None, nth=0):
    return page.evaluate("""([t, lc, nth]) => {
        const out = [];
        for (const r of __trueedit.A[__trueedit.page].regions) for (const w of r.words)
            if (w.text === t && (!lc || r.text.includes(lc))) out.push(w.id);
        return out[nth] || null; }""", [text, line_contains, nth])


def line_of(page, wid):
    return page.evaluate("(w) => __trueedit.A[__trueedit.page].regions.find(r => r.words.some(x => x.id === w)).id", wid)


def patches(page):
    return page.eval_on_selector_all("#patch-layer img", "e => e.length")


def wait_patches(page, n):
    page.wait_for_function(f"document.querySelectorAll('#patch-layer img').length === {n}", timeout=60_000)


def focus_word(page, wid):
    """Zoom the view onto a word (as a user would pinch/zoom) so it is comfortably tappable."""
    page.evaluate("""(w) => { const S = __trueedit; const r = S.A[S.page].regions.find(r => r.words.some(x => x.id === w));
        const b = (S.pos[S.page] && S.pos[S.page][w]) || r.words.find(x => x.id === w).bbox;
        const st = document.getElementById('stage').getBoundingClientRect();
        const z = Math.min(2.0, st.width / ((b[2] - b[0]) * 6));
        S.view = {z, x: st.width / 2 - (b[0] + b[2]) / 2 * z, y: st.height * 0.3 - (b[1] + b[3]) / 2 * z};
        document.getElementById('page').style.transform = `translate(${S.view.x}px, ${S.view.y}px) scale(${z})`;
        document.documentElement.style.setProperty('--inv', 1 / z); }""", wid)


def edit_word(page, wid, text, tap=False):
    focus_word(page, wid)
    sel = f'#region-layer .r[data-w="{wid}"]'
    (page.tap if tap else page.click)(sel)
    page.wait_for_selector("#pane-edit:not([hidden])")
    page.fill("#edit-input", text)
    n = patches(page)
    page.click("#btn-apply")
    return n


def export_download(page, fmt, out_path):
    page.click("#btn-export")
    page.wait_for_selector("#export:not([hidden])")
    page.check(f"input[name=fmt][value={fmt}]")
    with page.expect_download(timeout=120_000) as dl:
        page.click("#exp-go")
    dl.value.save_as(out_path)
    page.wait_for_function("document.getElementById('exp-status').textContent.startsWith('Downloaded')")
    page.click("#exp-close")
    return out_path


def server_canvas(pid):
    from trueedit import storage as S

    return S.load_image(S.page_dir(pid, 0) / "canvas.png"), json.loads((S.page_dir(pid, 0) / "analysis.json").read_text())


# ------------------------------------------------------------------ Gate 1

def test_app_health(desktop, live_server):
    page = desktop
    page.goto(live_server)
    page.wait_for_selector("#drop")
    page.wait_for_function("document.getElementById('engine-line').textContent.includes('PP-OCRv5')")
    assert page.is_visible("text=Drop a file here")
    open_and_upload(page, live_server, "order_clean.png")
    assert page.eval_on_selector_all("#region-layer .r", "e => e.length") > 150
    page.watch.assert_clean()


# ------------------------------------------------------------------ Gate 10 (desktop acceptance)

def test_end_to_end_acceptance(desktop, live_server, tmp_path):
    page = desktop
    page.goto(live_server)                                                         # 1 open app
    open_and_upload(page, live_server, "order_photo.jpg")                          # 2-3 upload + wait OCR
    pid = page.evaluate("__trueedit.pid")
    canvas, A = server_canvas(pid)
    # 4 verify OCR regions: every key field is a detected box at the right place
    for t in ("Daniel", "Kowalski", "Organic", "PO-55821-CA", "NW-2026-118734", "189.99", "663.99", "QTY"):
        assert word_id(page, t), f"{t} not detected"
    page.screenshot(path=str(RESULTS / "e2e_01_analyzed.png"))
    # 5 change first name
    edit_word(page, word_id(page, "Daniel"), "Michael")
    wait_patches(page, 1)
    # 6 change last name (its box moved right when the first name grew)
    edit_word(page, word_id(page, "Kowalski"), "Johnson")
    wait_patches(page, 2)
    page.screenshot(path=str(RESULTS / "e2e_02_names.png"))
    # 7 replace one item description (double-click selects the whole line)
    wid = word_id(page, "Organic")
    focus_word(page, wid)
    page.dblclick(f'#region-layer .r[data-w="{wid}"]')
    page.wait_for_function("document.getElementById('edit-title').textContent === 'Edit line'")
    assert page.input_value("#edit-input") == "Organic Potting Soil 50 L"
    page.fill("#edit-input", "Premium Garden Compost 40 L")
    page.click("#btn-apply")
    wait_patches(page, 3)
    page.click("#zoom-fit")
    page.screenshot(path=str(RESULTS / "e2e_03_edited.png"))
    # 8-12 nothing else is touched (checked on the export below) ; 13 compare
    page.click("#btn-compare")
    page.wait_for_function("document.getElementById('cmp-stats').textContent.includes('identical')", timeout=60_000)
    stats = page.text_content("#cmp-stats")
    assert "Every other pixel is identical" in stats, stats
    for mode in ("swipe", "overlay", "diff", "side"):
        page.click(f"#cmp-modes button[data-m={mode}]")
        page.wait_for_function("[...document.querySelectorAll('#cmp-body img')].every(i => i.complete && i.naturalWidth > 0)",
                               timeout=60_000)
        page.screenshot(path=str(RESULTS / f"e2e_04_compare_{mode}.png"))
    page.click("#cmp-close")
    # 14 export PDF ; 15 reopen
    pdf_path = export_download(page, "pdf", str(tmp_path / "out.pdf"))
    doc = pymupdf.open(pdf_path)
    assert doc.page_count == 1
    assert abs(doc[0].rect.width - 612) < 0.5 and abs(doc[0].rect.height - 792) < 0.5
    H, W = canvas.shape[:2]
    pix = doc[0].get_pixmap(matrix=pymupdf.Matrix(W / doc[0].rect.width, H / doc[0].rect.height), alpha=False)
    out = cv2.cvtColor(np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(RESULTS / "e2e_05_exported_page.png"), out)
    # 16 inspect: edits present, everything else the same
    words = {w["text"]: w for r in A["regions"] for w in r["words"]}
    lines = {r["text"]: r for r in A["regions"]}
    nb = words["Daniel"]["bbox"]
    assert read_text(out, [nb[0], nb[1], nb[2] + 220, nb[3]]).replace(" ", "") == "MichaelJohnson"
    assert read_text(out, lines["Organic Potting Soil 50 L"]["bbox"][:2] +
                     [lines["Organic Potting Soil 50 L"]["bbox"][2] + 120, lines["Organic Potting Soil 50 L"]["bbox"][3]]
                     ).replace(" ", "") == "PremiumGardenCompost40L"
    diff = np.abs(out.astype(int) - canvas.astype(int)).max(axis=2)
    edited = np.zeros(diff.shape, bool)
    info = page.evaluate("__trueedit.info[0]")
    for e in page.evaluate("__trueedit.doc.edits"):
        i = info[e["id"]]
        for b in [e["bbox"], i.get("new_text_bbox"), i.get("changed_bbox")]:
            if b:
                edited[max(0, b[1] - 8):b[3] + 8, max(0, b[0] - 8):b[2] + 8] = True
    outside = diff[~edited]
    assert outside.mean() < 0.8 and np.percentile(outside, 99.9) <= 12, (outside.mean(), np.percentile(outside, 99.9))
    for label in ("PO-55821-CA", "NW-2026-118734", "189.99", "379.98", "663.99", "463.99", "Subtotal"):
        b = words[label]["bbox"]
        assert read_text(out, b) == label                      # reference / prices / totals untouched
    for r in A["regions"]:
        if r["table"] and r["table"]["col"] == 3 and r["text"] != "QTY":
            b = r["bbox"]
            assert diff[b[1]:b[3], b[0]:b[2]].max() <= 12      # quantities untouched
    for g in A["graphics"] + A["barcodes"]:
        b = g.get("group_box", g["box"])
        assert diff[b[1]:b[3], b[0]:b[2]].mean() < 0.8         # logo & barcode untouched
    save_result("e2e_acceptance.json", {"outside_mean_abs": float(outside.mean()),
                                        "outside_p999": float(np.percentile(outside, 99.9)),
                                        "compare_stats": stats, "pdf_bytes": os.path.getsize(pdf_path)})
    page.watch.assert_clean()


# ------------------------------------------------------------------ editor features

def test_undo_redo_and_persistence(desktop, live_server):
    page = desktop
    open_and_upload(page, live_server, "order_clean.png")
    edit_word(page, word_id(page, "Daniel"), "Anna")
    wait_patches(page, 1)
    page.click("#btn-undo")
    wait_patches(page, 0)
    assert page.evaluate("__trueedit.doc.edits.length") == 0
    page.click("#btn-redo")
    wait_patches(page, 1)
    page.click("#zoom-fit")
    page.keyboard.press("Control+z")
    wait_patches(page, 0)
    page.keyboard.press("Control+Shift+z")
    wait_patches(page, 1)
    # edits survive a reload (autosaved session)
    page.wait_for_timeout(800)
    page.reload()
    page.wait_for_selector("#screen-editor:not([hidden])")
    wait_patches(page, 1)
    assert page.evaluate("__trueedit.doc.edits[0].text") == "Anna"
    # delete + restore original
    wid = word_id(page, "Kowalski")
    focus_word(page, wid)
    page.click(f'#region-layer .r[data-w="{wid}"]')
    page.click("#btn-delete")
    wait_patches(page, 2)
    page.click("#btn-revert")
    wait_patches(page, 1)
    page.watch.assert_clean()


def test_manual_region_and_resize(desktop, live_server):
    page = desktop
    open_and_upload(page, live_server, "order_clean.png")
    wid = word_id(page, "Tremblay")
    focus_word(page, wid)
    box = page.locator(f'#region-layer .r[data-w="{wid}"]').bounding_box()
    page.click("#mode-draw")
    page.mouse.move(box["x"] - 6, box["y"] - 6)
    page.mouse.down()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, steps=4)
    page.mouse.move(box["x"] + box["width"] + 6, box["y"] + box["height"] + 6, steps=4)
    page.mouse.up()
    page.wait_for_selector("#pane-edit:not([hidden])")
    page.wait_for_function("document.getElementById('edit-title').textContent === 'Edit region'")
    assert page.text_content("#edit-orig").strip() == "Tremblay"   # OCR of the drawn region
    page.fill("#edit-input", "Lavoie")
    page.click("#btn-apply")
    wait_patches(page, 1)
    # resize: select a word, enable "Adjust area", drag the bottom-right handle
    wid = word_id(page, "Curbside")
    focus_word(page, wid)
    page.click(f'#region-layer .r[data-w="{wid}"]')
    page.click("#btn-adjust")
    h = page.locator("#sel-layer .h.se").bounding_box()
    before = page.evaluate("__trueedit.sel.bbox")
    page.mouse.move(h["x"] + h["width"] / 2, h["y"] + h["height"] / 2)
    page.mouse.down()
    page.mouse.move(h["x"] + 40, h["y"] + 6, steps=5)
    page.mouse.up()
    after = page.evaluate("__trueedit.sel.bbox")
    assert after[2] > before[2]
    # move the area
    s = page.locator("#sel-layer .sel").bounding_box()
    page.mouse.move(s["x"] + s["width"] / 2, s["y"] + s["height"] / 2)
    page.mouse.down()
    page.mouse.move(s["x"] + s["width"] / 2 + 10, s["y"] + s["height"] / 2, steps=3)
    page.mouse.up()
    moved = page.evaluate("__trueedit.sel.bbox")
    assert moved[0] > after[0]
    page.fill("#edit-input", "Express")
    page.click("#btn-apply")
    wait_patches(page, 2)
    page.watch.assert_clean()


def test_find_replace_and_review(desktop, live_server):
    page = desktop
    open_and_upload(page, live_server, "order_clean.png")
    page.keyboard.press("Control+f")
    page.fill("#find-q", "N/A")
    page.wait_for_function("document.getElementById('find-status').textContent.includes('3 match')")
    page.click("#find-next")
    assert page.text_content("#find-status").startswith("1 of 3")
    page.fill("#find-r", "None")
    page.click("#find-all")
    wait_patches(page, 3)
    assert page.evaluate("__trueedit.doc.edits.map(e => e.text)") == ["None"] * 3
    page.click("#btn-changes")
    assert page.eval_on_selector_all("#change-list li", "e => e.length") == 3
    page.watch.assert_clean()


def test_review_queue(desktop, live_server):
    page = desktop
    open_and_upload(page, live_server, "order_lowres.jpg")
    page.click("#btn-review")
    n0 = int(page.text_content("#review-count"))
    assert n0 > 0
    page.locator("#review-list [data-ok]").first.click()
    page.wait_for_function(f"document.getElementById('review-count').textContent === '{n0 - 1}'")
    page.locator("#review-list [data-show]").nth(1).click()
    page.wait_for_selector("#pane-edit:not([hidden])")
    assert page.is_visible("#edit-warn")  # uncertain reading is called out before editing
    page.watch.assert_clean()


def test_failure_ui_messages(desktop, live_server):
    page = desktop
    page.goto(live_server)
    page.set_input_files("#file", str(FX / "notes.txt"))
    page.wait_for_selector("#up-error:not([hidden])")
    assert "Unsupported file type" in page.text_content("#up-error")
    page.set_input_files("#file", str(FX / "corrupt.jpg"))
    page.wait_for_function("document.getElementById('up-error').textContent.includes('corrupt')")
    # OCR failure -> recovery options -> continue without OCR
    os.environ["TRUEEDIT_FAULT"] = "ocr_crash"
    try:
        page.set_input_files("#file", str(FX / "order_clean.png"))
        page.wait_for_selector("#an-actions:not([hidden])", timeout=60_000)
        assert "Text recognition failed" in page.text_content("#an-error")
        page.click("text=Continue without OCR")
        page.wait_for_selector("#screen-editor:not([hidden])", timeout=60_000)
    finally:
        os.environ.pop("TRUEEDIT_FAULT", None)
    assert page.evaluate("__trueedit.A[0].regions.length") == 0
    page.wait_for_selector("#page-banner:not([hidden])")
    assert "OCR skipped" in page.text_content("#page-banner")


def test_multipage_navigation(desktop, live_server):
    page = desktop
    open_and_upload(page, live_server, "order_multipage.pdf")
    assert page.text_content("#pg-label") == "1 / 3"
    page.click("#pg-next")
    page.wait_for_function("document.getElementById('pg-label').textContent === '2 / 3'")
    page.wait_for_function("__trueedit.page === 1 && document.querySelectorAll('#region-layer .r').length > 100")
    edit_word(page, word_id(page, "Kowalski"), "Nguyen")
    wait_patches(page, 1)
    page.click("#pg-prev")
    page.wait_for_function("__trueedit.page === 0")
    wait_patches(page, 0)
    page.watch.assert_clean()


# ------------------------------------------------------------------ Gate 9 (mobile)

def pinch(page, cx, cy, d0, d1, steps=8):
    cdp = page.context.new_cdp_session(page)

    def pts(d):
        return [{"x": cx - d / 2, "y": cy, "id": 1}, {"x": cx + d / 2, "y": cy, "id": 2}]

    cdp.send("Input.dispatchTouchEvent", {"type": "touchStart", "touchPoints": pts(d0)})
    for i in range(1, steps + 1):
        cdp.send("Input.dispatchTouchEvent", {"type": "touchMove", "touchPoints": pts(d0 + (d1 - d0) * i / steps)})
    cdp.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})


def test_mobile_workflow(mobile, live_server, tmp_path):
    page = mobile
    page.goto(live_server)
    page.wait_for_selector("#drop")
    assert page.evaluate("document.scrollingElement.scrollWidth <= innerWidth")
    page.screenshot(path=str(RESULTS / "mobile_01_upload.png"))
    open_and_upload(page, live_server, "order_photo.jpg")
    page.screenshot(path=str(RESULTS / "mobile_02_editor.png"))
    assert page.evaluate("document.scrollingElement.scrollWidth <= innerWidth")
    # zoom: buttons and a real two-finger pinch
    z0 = page.evaluate("__trueedit.view.z")
    page.tap("#zoom-in")
    z1 = page.evaluate("__trueedit.view.z")
    assert z1 > z0
    st = page.locator("#stage").bounding_box()
    pinch(page, st["x"] + st["width"] / 2, st["y"] + st["height"] / 2, 80, 240)
    z2 = page.evaluate("__trueedit.view.z")
    assert z2 > z1 * 2, (z1, z2)
    # select text with a tap and edit in the bottom sheet
    edit_word(page, word_id(page, "Daniel"), "Michael", tap=True)
    wait_patches(page, 1)
    panel = page.locator("#panel").bounding_box()
    assert panel["y"] > 844 * 0.35 and panel["width"] >= 389    # bottom sheet, full width
    page.screenshot(path=str(RESULTS / "mobile_03_edit.png"))
    # pan with one finger (drag on the page)
    v0 = page.evaluate("__trueedit.view.x")
    cdp = page.context.new_cdp_session(page)
    y = st["y"] + 120
    cdp.send("Input.dispatchTouchEvent", {"type": "touchStart", "touchPoints": [{"x": 200, "y": y, "id": 1}]})
    for i in range(1, 6):
        cdp.send("Input.dispatchTouchEvent", {"type": "touchMove", "touchPoints": [{"x": 200 - 20 * i, "y": y, "id": 1}]})
    cdp.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})
    assert page.evaluate("__trueedit.view.x") < v0 - 50
    # compare & export on the phone
    # a human pauses after a pan before tapping; a tap during an active fling only stops the fling
    page.wait_for_timeout(700)
    page.tap("#btn-compare")
    page.wait_for_function("document.getElementById('cmp-stats').textContent.includes('identical')", timeout=60_000)
    page.screenshot(path=str(RESULTS / "mobile_04_compare.png"))
    page.tap("#cmp-close")
    path = export_download(page, "pdf", str(tmp_path / "m.pdf"))
    assert pymupdf.open(path).page_count == 1
    page.watch.assert_clean()
