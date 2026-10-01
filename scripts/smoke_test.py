"""End-to-end smoke test against a RUNNING TrueEdit server (real browser, real OCR).

    scripts/run.sh &                       # start the app
    .venv/bin/python scripts/smoke_test.py [http://127.0.0.1:8000] [samples/order_photo.jpg]

Needs the test tools:  scripts/setup.sh --dev
Prints PASS/FAIL per check; exit code 0 only if every check passes.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
URL = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
SAMPLE = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "samples" / "order_photo.jpg"
CHROME = os.environ.get("TRUEEDIT_CHROME", "/opt/pw-browsers/chromium-1194/chrome-linux/chrome")
OUT = Path(tempfile.mkdtemp(prefix="trueedit-smoke-"))

results: list[tuple[str, bool, str]] = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""), flush=True)
    return ok


def get(path, binary=False):
    with urllib.request.urlopen(URL + path, timeout=120) as r:
        data = r.read()
    return data if binary else json.loads(data)


def img(data):
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def main():
    from playwright.sync_api import sync_playwright

    # 1. server
    try:
        h = get("/api/health")
        check("server starts and API responds", h.get("ok"), f"version {h.get('version')}")
        check("OCR engine available", h["ocr"]["available"], h["ocr"].get("primary", h["ocr"].get("error")))
    except Exception as e:
        check("server starts and API responds", False, str(e))
        return

    with sync_playwright() as p:
        b = p.chromium.launch(**({"executable_path": CHROME} if os.path.exists(CHROME) else {}))
        page = b.new_context(viewport={"width": 1400, "height": 900}, accept_downloads=True).new_page()
        errors = []
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(str(e)))

        # 2. page loads
        page.goto(URL + "/")
        page.wait_for_selector("#drop", timeout=20_000)
        check(f"{URL} loads", page.is_visible("text=Drop a file here"))

        # 3-4. upload + OCR
        t = time.time()
        page.set_input_files("#file", str(SAMPLE))
        check("upload accepted", page.wait_for_selector("#screen-analyze:not([hidden]), #screen-editor:not([hidden])",
                                                         timeout=30_000) is not None, SAMPLE.name)
        page.wait_for_selector("#screen-editor:not([hidden])", timeout=300_000)
        pid = page.evaluate("__trueedit.pid")
        A = get(f"/api/projects/{pid}/pages/0/analysis")
        n_words = sum(len(r["words"]) for r in A["regions"])
        check("OCR completes", n_words > 50, f"{n_words} words, {len(A['regions'])} lines in {time.time() - t:.0f}s")

        def wid(text):
            return next((w["id"] for r in A["regions"] for w in r["words"] if w["text"] == text), None)

        def focus_and_click(w):
            page.evaluate("""(w) => { const S = __trueedit; const r = S.A[0].regions.find(r => r.words.some(x => x.id === w));
                const b = r.words.find(x => x.id === w).bbox; const st = document.getElementById('stage').getBoundingClientRect();
                const z = Math.min(2, st.width / ((b[2] - b[0]) * 6));
                S.view = {z, x: st.width / 2 - (b[0] + b[2]) / 2 * z, y: st.height * 0.3 - (b[1] + b[3]) / 2 * z};
                document.getElementById('page').style.transform = `translate(${S.view.x}px, ${S.view.y}px) scale(${z})`; }""", w)
            page.click(f'#region-layer .r[data-w="{w}"]')
            page.wait_for_selector("#pane-edit:not([hidden])")

        # 5. detected text is editable
        target = wid("Daniel")
        check("customer first name detected", target is not None)
        focus_and_click(target)
        check("detected text becomes editable",
              page.input_value("#edit-input") == "Daniel" and page.is_enabled("#edit-input") and page.is_enabled("#btn-apply"))

        # 6. logo / barcode locked
        labels = page.eval_on_selector_all("#region-layer .g span", "e => e.map(x => x.textContent)")
        check("logo and barcode marked protected", any("Logo" in l for l in labels) and any("Barcode" in l for l in labels),
              ", ".join(labels))
        roles = {r["role"] for r in A["regions"]}
        check("logo/barcode text excluded from editable text", {"logo_text", "barcode_text"} <= roles)
        for role in ("logo_text", "barcode_text"):
            r = next(r for r in A["regions"] if r["role"] == role)
            focus_and_click(r["words"][0]["id"])
            check(f"{role.replace('_', ' ')} is locked in the editor ('{r['text']}')",
                  page.is_visible("#edit-locked") and not page.is_enabled("#edit-input")
                  and not page.is_enabled("#btn-apply"))

        # 7. edit one field
        focus_and_click(target)
        page.fill("#edit-input", "Michael")
        page.click("#btn-apply")
        page.wait_for_function("document.querySelectorAll('#patch-layer img').length === 1", timeout=60_000)
        page.wait_for_timeout(800)  # autosave
        f = get(f"/api/projects/{pid}/pages/0/fidelity")
        check("edit applied", f["edits"] == 1 and f["changed_pixels"] > 0,
              f"{f['changed_pixels']} px changed ({f['changed_fraction'] * 100:.3f}% of page)")
        check("no pixel changed outside the edited field", f["outside_intended_pixels"] == 0)
        canvas = img(get(f"/api/projects/{pid}/pages/0/image/canvas?fmt=png", True))
        edited = img(get(f"/api/projects/{pid}/pages/0/edited?fmt=png", True))
        d = np.any(canvas != edited, axis=2)
        line = next(r for r in A["regions"] if any(w["id"] == target for w in r["words"]))
        touched = []
        for r in A["regions"]:
            if r["id"] == line["id"]:
                continue
            for w in r["words"]:
                x0, y0, x1, y1 = w["bbox"]
                if d[y0:y1, x0:x1].any():
                    touched.append(w["text"])
        for g in A["graphics"] + A["barcodes"]:
            x0, y0, x1, y1 = g.get("group_box", g["box"])
            if d[y0:y1, x0:x1].any():
                touched.append(g["kind"])
        check("unrelated fields, logo and barcode unchanged", not touched,
              f"{n_words - len(line['words'])} other words + logo + barcode compared" if not touched else str(touched))

        # 8. undo / redo
        page.click("#btn-undo")
        page.wait_for_function("document.querySelectorAll('#patch-layer img').length === 0", timeout=30_000)
        check("undo removes the edit", page.evaluate("__trueedit.doc.edits.length") == 0)
        page.click("#btn-redo")
        page.wait_for_function("document.querySelectorAll('#patch-layer img').length === 1", timeout=30_000)
        check("redo restores the edit", page.evaluate("__trueedit.doc.edits[0].text") == "Michael")
        page.wait_for_timeout(800)

        # 9. export
        files = {}
        for fmt in ("pdf", "png"):
            page.click("#btn-export")
            page.wait_for_selector("#export:not([hidden])")
            page.check(f"input[name=fmt][value={fmt}]")
            with page.expect_download(timeout=120_000) as dl:
                page.click("#exp-go")
            files[fmt] = OUT / dl.value.suggested_filename
            dl.value.save_as(files[fmt])
            page.click("#exp-close")
            check(f"export {fmt.upper()} downloads", files[fmt].stat().st_size > 10_000,
                  f"{files[fmt].name}, {files[fmt].stat().st_size / 1e6:.1f} MB")
        check("no browser console errors", not errors, "; ".join(errors[:3]))
        b.close()

    # 10. reopen exports
    import pymupdf

    png = cv2.imread(str(files["png"]))
    check("exported PNG reopens identical to the editor result", png is not None and np.array_equal(png, edited),
          f"{png.shape[1]}x{png.shape[0]}" if png is not None else "unreadable")
    doc = pymupdf.open(files["pdf"])
    pg = doc[0]
    check("exported PDF reopens", doc.page_count == 1, f"{doc.page_count} page, {pg.rect.width:.0f}x{pg.rect.height:.0f} pt")
    txt = pg.get_text()
    check("PDF text layer has the edit (Michael) and not the old name (Daniel)", "Michael" in txt and "Daniel" not in txt)
    H, W = edited.shape[:2]
    pix = pg.get_pixmap(matrix=pymupdf.Matrix(W / pg.rect.width, H / pg.rect.height), alpha=False)
    raster = cv2.cvtColor(np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3), cv2.COLOR_RGB2BGR)
    diff = np.abs(raster.astype(int) - edited.astype(int))
    check("PDF page matches the editor result", raster.shape == edited.shape and diff.mean() < 1.0,
          f"mean abs pixel diff {diff.mean():.3f}")
    from trueedit.ocr.ppocr import crop_quad, load_models

    m = load_models()
    x0, y0, x1, y1 = line["bbox"]
    crop = raster[max(0, y0 - 10):y1 + 10, max(0, x0 - 10):x1 + 60]
    boxes = sorted(m["det"](crop, limit_side=1600), key=lambda b: b.quad[:, 0].min())
    read = " ".join(r.text for r in m["rec"]([crop_quad(crop, b.quad)[0] for b in boxes]))
    check("edited name reads back from the exported PDF", read.replace(" ", "").startswith("Michael"), repr(read))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # any crash is a failure, never a silent pass
        import traceback

        traceback.print_exc()
        check("smoke test ran to completion", False, repr(e))
    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed. Exports in {OUT}")
    sys.exit(1 if failed or not results else 0)
