"""Gate 2 — input pipeline: formats, page dimensions, rotation, preview."""
import cv2
import numpy as np
import pytest

from conftest import FX, upload
from trueedit.ingest import InputError, load_document


@pytest.mark.parametrize("name,kind,pages", [
    ("order_photo.jpg", "image", 1), ("order_clean.png", "image", 1), ("order.heic", "image", 1),
    ("order.pdf", "pdf", 1), ("order_multipage.pdf", "pdf", 3), ("order_scan.pdf", "pdf", 1),
])
def test_formats_decode(name, kind, pages):
    doc = load_document((FX / name).read_bytes(), name)
    assert doc.kind == kind and len(doc.pages) == pages
    for p in doc.pages:
        assert p.image.dtype == np.uint8 and p.image.ndim == 3 and min(p.image.shape[:2]) > 500


def test_image_dimensions_exact():
    src = cv2.imread(str(FX / "order_photo.jpg"))
    doc = load_document((FX / "order_photo.jpg").read_bytes())
    assert doc.pages[0].image.shape == src.shape
    assert np.abs(doc.pages[0].image.astype(int) - src.astype(int)).mean() < 1.0  # same decode


def test_pdf_page_size_and_raster():
    doc = load_document((FX / "order_multipage.pdf").read_bytes())
    for p in doc.pages:
        assert p.pdf_size_pt == (612.0, 792.0)
        h, w = p.image.shape[:2]
        assert abs(w / h - 612 / 792) < 0.002
        assert p.has_text_layer and not p.is_photo


def test_exif_orientation_applied():
    doc = load_document((FX / "order_photo_exif.jpg").read_bytes())
    h, w = doc.pages[0].image.shape[:2]
    assert h > w, "EXIF orientation 6 must produce a portrait image"
    assert doc.pages[0].exif_orientation == 6


@pytest.mark.parametrize("name", ["order_photo.jpg", "order_clean.png", "order.pdf", "order_multipage.pdf", "order.heic",
                                  "order_photo_rot90.jpg"])
def test_api_upload_preview(client, project, name):
    pid = project(name)
    m = client.get(f"/api/projects/{pid}").json()
    assert m["status"] == "ready" and len(m["pages"]) == m["page_count"]
    for pm in m["pages"]:
        r = client.get(f"/api/projects/{pid}/pages/{pm['index']}/image/canvas?fmt=png")
        assert r.status_code == 200
        img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
        assert img.shape[:2] == (pm["height"], pm["width"])
        assert pm["height"] > pm["width"], "pages must be upright portrait after preprocessing"
        r = client.get(f"/api/projects/{pid}/pages/{pm['index']}/image/original?fmt=jpg")
        assert r.status_code == 200
    if name.endswith(".pdf"):
        assert all(p["pdf_size_pt"] == [612.0, 792.0] for p in m["pages"])
    if name in ("order_photo.jpg", "order.heic", "order_photo_rot90.jpg"):
        assert m["pages"][0]["paper"] == "letter"
        assert abs(m["pages"][0]["height"] / m["pages"][0]["width"] - 11 / 8.5) < 0.01


def test_rotated_photo_is_turned_upright(project, client):
    pid = project("order_photo_rot90.jpg")
    pm = client.get(f"/api/projects/{pid}").json()["pages"][0]
    assert any(s["step"] == "orientation" for s in pm["steps"])
    A = client.get(f"/api/projects/{pid}/pages/0/analysis").json()
    assert any(w["text"] == "Kowalski" for r in A["regions"] for w in r["words"])


def test_original_is_preserved(project, client):
    pid = project("order_photo.jpg")
    from trueedit import storage as S

    d = S.pdir(pid)
    assert (d / "source.jpg").read_bytes() == (FX / "order_photo.jpg").read_bytes()
    orig = cv2.imread(str(d / "pages/0/original.png"))
    assert orig.shape == cv2.imread(str(FX / "order_photo.jpg")).shape


def test_page_navigation_multipage(project, client):
    pid = project("order_multipage.pdf")
    texts = []
    for n in range(3):
        A = client.get(f"/api/projects/{pid}/pages/{n}/analysis").json()
        texts.append(" ".join(r["text"] for r in A["regions"]))
    assert "Page 1 of 3" in texts[0] and "Page 2 of 3" in texts[1] and "Page 3 of 3" in texts[2]
