"""Gate 7 — export PDF / PNG / JPG / project, reopen and verify."""
import io
import zipfile

import cv2
import numpy as np
import pymupdf
import pytest

from helpers import FX, RESULTS, save_result
from helpers import changed_mask, find_line, find_word, read_text
from test_editing import CASES, make_edit, page_state
from trueedit import export as X


def setup_edits(client, pid, names=("A_first_name", "B_last_name", "C_item_description")):
    A, canvas = page_state(client, pid)
    eds = [make_edit(A, CASES[n][1], CASES[n][2])[0] for n in names]
    r = client.put(f"/api/projects/{pid}/edits", json={"edits": eds})
    assert r.status_code == 200
    _, edited, patches = X.edited_page(pid, 0)
    return A, canvas, edited, eds


def export(client, pid, fmt, quality="lossless"):
    r = client.post(f"/api/projects/{pid}/export", json={"format": fmt, "quality": quality})
    assert r.status_code == 200, r.text
    assert "attachment" in r.headers["content-disposition"]
    return r.content


def rasterize(pdf_bytes, n, size):
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    p = doc[n]
    W, H = size
    mat = pymupdf.Matrix(W / p.rect.width, H / p.rect.height)
    pix = p.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3]
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR), doc


def test_png_export_is_exact(client, project):
    pid = project("order_photo.jpg")
    A, canvas, edited, eds = setup_edits(client, pid)
    png = export(client, pid, "png")
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == canvas.shape
    assert np.array_equal(img, edited), "PNG must equal the editor result exactly"
    d = changed_mask(canvas, img)
    for g in A["graphics"] + A["barcodes"]:
        b = g.get("group_box", g["box"])
        assert not d[b[1]:b[3], b[0]:b[2]].any()
    cv2.imwrite(str(RESULTS / "export_photo.png"), img)


def test_jpg_export(client, project):
    pid = project("order_photo.jpg")
    _, canvas, edited, _ = setup_edits(client, pid)
    jpg = export(client, pid, "jpg")
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == canvas.shape
    assert np.abs(img.astype(int) - edited.astype(int)).mean() < 2.5  # only JPEG loss


@pytest.mark.parametrize("quality", ["lossless", "compact"])
def test_pdf_export_photo(client, project, quality):
    pid = project("order_photo.jpg")
    A, canvas, edited, eds = setup_edits(client, pid)
    pdf = export(client, pid, "pdf", quality)
    H, W = canvas.shape[:2]
    img, doc = rasterize(pdf, 0, (W, H))
    assert doc.page_count == 1
    assert abs(doc[0].rect.width - 612) < 0.5 and abs(doc[0].rect.height - 792) < 0.5  # detected US Letter
    diff = np.abs(img.astype(int) - edited.astype(int))
    tol = 1.0 if quality == "lossless" else 3.0
    assert diff.mean() < tol, diff.mean()
    # edits present, originals gone, untouched fields intact (read back from the exported PDF raster)
    r_name = find_word(A, "Daniel")[0]
    got = read_text(img, [r_name["bbox"][0], r_name["bbox"][1], r_name["bbox"][2] + 200, r_name["bbox"][3]])
    assert got.replace(" ", "") == "MichaelJohnson", got
    for t in ("PO-55821-CA", "NW-2026-118734", "663.99", "189.99"):
        r, w = find_word(A, t)
        assert read_text(img, w["bbox"]) == t
    # logo / barcode preserved
    for g in A["graphics"] + A["barcodes"]:
        b = g.get("group_box", g["box"])
        assert diff[b[1]:b[3], b[0]:b[2]].mean() < tol
    # searchable text layer reflects the edit
    txt = doc[0].get_text()
    assert "Michael" in txt and "Johnson" in txt and "Daniel" not in txt and "Kowalski" not in txt
    assert "PO-55821-CA" in txt
    if quality == "lossless":
        (RESULTS / "export_photo.pdf").write_bytes(pdf)
    save_result(f"export_pdf_{quality}.json", {"bytes": len(pdf), "mean_abs_diff": float(diff.mean()),
                                               "page_pt": [doc[0].rect.width, doc[0].rect.height]})


def test_pdf_export_vector_input_keeps_vectors(client, project):
    pid = project("order.pdf")
    A, canvas = page_state(client, pid)
    r, w = find_word(A, "Daniel")
    r2 = find_line(A, "Organic Potting Soil 50 L")
    eds = [{"id": "a", "page": 0, "bbox": w["bbox"], "text": "Michael", "target_ids": [w["id"]], "source": "ocr"},
           {"id": "c", "page": 0, "bbox": r2["bbox"], "text": "Premium Garden Compost 40 L", "target_ids": [r2["id"]], "source": "ocr"}]
    client.put(f"/api/projects/{pid}/edits", json={"edits": eds})
    _, edited, _ = X.edited_page(pid, 0)
    pdf = export(client, pid, "pdf")
    src = pymupdf.open(stream=(FX / "order.pdf").read_bytes(), filetype="pdf")
    img, doc = rasterize(pdf, 0, (canvas.shape[1], canvas.shape[0]))
    assert doc.page_count == 1 and doc[0].rect == src[0].rect
    txt = doc[0].get_text()
    assert "Daniel" not in txt and "Organic Potting" not in txt, "old text must be removed, not just covered"
    assert "Michael" in txt and "Kowalski" in txt and "PO-55821-CA" in txt and "Cedar Raised Garden Bed" in txt
    # untouched content is still real vector text (selectable), page drawings preserved
    assert len(doc[0].get_drawings()) >= len(src[0].get_drawings()) * 0.95
    diff = np.abs(img.astype(int) - edited.astype(int))
    assert diff.mean() < 1.0
    got = read_text(img, [w["bbox"][0], w["bbox"][1], w["bbox"][2] + 220, w["bbox"][3]])
    assert got.replace(" ", "") == "MichaelKowalski", got
    (RESULTS / "export_vector.pdf").write_bytes(pdf)


def test_pdf_export_multipage(client, project):
    pid = project("order_multipage.pdf")
    A1 = client.get(f"/api/projects/{pid}/pages/1/analysis").json()
    r, w = find_word(A1, "Kowalski")
    eds = [{"id": "p2", "page": 1, "bbox": w["bbox"], "text": "Nguyen", "target_ids": [w["id"]], "source": "ocr"}]
    client.put(f"/api/projects/{pid}/edits", json={"edits": eds})
    pdf = export(client, pid, "pdf")
    doc = pymupdf.open(stream=pdf, filetype="pdf")
    src = pymupdf.open(stream=(FX / "order_multipage.pdf").read_bytes(), filetype="pdf")
    assert doc.page_count == 3
    for i in range(3):
        assert doc[i].rect == src[i].rect
    assert "Kowalski" in doc[0].get_text() and "Kowalski" in doc[2].get_text()
    assert "Kowalski" not in doc[1].get_text() and "Nguyen" in doc[1].get_text()
    # pages without edits are byte-for-byte the same content streams
    assert doc[0].read_contents() == src[0].read_contents()
    imgs = client.post(f"/api/projects/{pid}/export", json={"format": "png"})
    z = zipfile.ZipFile(io.BytesIO(imgs.content))
    assert sorted(z.namelist()) == ["page-1.png", "page-2.png", "page-3.png"]


def test_project_roundtrip(client, project):
    pid = project("order_photo.jpg")
    _, canvas, edited, eds = setup_edits(client, pid)
    blob = export(client, pid, "project")
    r = client.post("/api/projects/import", files={"file": ("x.trueedit", blob)})
    assert r.status_code == 201
    pid2 = r.json()["id"]
    assert r.json()["status"] == "ready"
    assert client.get(f"/api/projects/{pid2}/edits").json()["edits"] == eds
    _, edited2, _ = X.edited_page(pid2, 0)
    assert np.array_equal(edited, edited2)


def test_import_rejects_garbage(client):
    r = client.post("/api/projects/import", files={"file": ("x.trueedit", b"PK\x03\x04garbage")})
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_project"
