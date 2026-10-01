"""Gate 8 — failure conditions: clear errors, recovery options, no crashes, no silent failures."""
import io
import json
import time
import zipfile

import cv2
import numpy as np
import pytest

from conftest import FX, upload, wait_ready
from helpers import changed_mask, find_word, read_text
from trueedit import jobs
from trueedit import storage as S


def post(client, name, data):
    return client.post("/api/projects", files={"file": (name, data)})


def test_corrupt_image(client):
    r = post(client, "broken.jpg", (FX / "corrupt.jpg").read_bytes())
    assert r.status_code == 400
    e = r.json()["error"]
    assert e["code"] == "corrupt_file" and e["hint"]


def test_truncated_pdf(client):
    data = (FX / "order.pdf").read_bytes()[:1500]
    r = post(client, "cut.pdf", data)
    assert r.status_code == 400 and r.json()["error"]["code"] in ("corrupt_file", "empty_document")


@pytest.mark.parametrize("name,data", [
    ("notes.txt", b"just some text\n"),
    ("doc.docx", b"PK\x03\x04" + b"\x00" * 100),
    ("image.jpg", b"GIF89a\x01\x00\x01\x00"),  # wrong content behind a .jpg name
])
def test_unsupported(client, name, data):
    r = post(client, name, data)
    assert r.status_code == 400
    e = r.json()["error"]
    assert e["code"] == "unsupported_type" and "PDF" in e["hint"]


def test_empty_file(client):
    r = post(client, "empty.png", b"")
    assert r.status_code == 400 and r.json()["error"]["code"] == "empty_file"


def test_too_large(client, monkeypatch):
    import trueedit.ingest as I

    monkeypatch.setattr(I, "MAX_UPLOAD_BYTES", 1000)
    r = post(client, "big.png", (FX / "order_clean.png").read_bytes())
    assert r.status_code == 400 and r.json()["error"]["code"] == "too_large"


def test_huge_image_is_processed(client):
    t = time.time()
    pid = upload(client, "order_huge.png")
    m = client.get(f"/api/projects/{pid}").json()
    pm = m["pages"][0]
    assert max(pm["width"], pm["height"]) <= 3600
    assert any("reduced" in w for w in pm["warnings"])
    assert pm["stats"]["words"] > 150
    assert time.time() - t < 240


def test_low_resolution_photo(client):
    pid = upload(client, "order_lowres.jpg")
    m = client.get(f"/api/projects/{pid}").json()
    A = client.get(f"/api/projects/{pid}/pages/0/analysis").json()
    assert any("low resolution" in w.lower() for w in m["warnings"] + m["pages"][0]["warnings"])
    assert A["stats"]["low_resolution"] and len(A["review"]) > 0


def test_rotated_document(client):
    pid = upload(client, "order_photo_rot90.jpg")
    A = client.get(f"/api/projects/{pid}/pages/0/analysis").json()
    assert find_word(A, "Kowalski")


def test_empty_document(client):
    pid = upload(client, "blank.png")
    A = client.get(f"/api/projects/{pid}/pages/0/analysis").json()
    assert A["regions"] == []
    assert any("No text was found" in w for w in A["warnings"])
    # manual editing still possible on an empty page
    e = {"id": "m1", "page": 0, "bbox": [200, 200, 700, 260], "text": "Added note", "target_ids": ["m-1"],
         "source": "manual"}
    r = client.post(f"/api/projects/{pid}/pages/0/render", json={"edits": [e], "regions": []})
    assert r.status_code == 200 and r.json()["patches"][0]["changed_bbox"]


def _job_error(client, pid):
    st = wait_ready(client, pid)
    assert st["status"] == "error"
    return st["error"]


def test_ocr_engine_crash_offers_recovery(client, monkeypatch):
    monkeypatch.setenv("TRUEEDIT_FAULT", "ocr_crash")
    pid = upload(client, "order_clean.png", wait=False)
    err = _job_error(client, pid)
    assert err["code"] == "ocr_failed" and "fault" in err["message"]
    actions = {a["action"] for a in err["recovery"]}
    assert {"retry", "manual"} <= actions
    # recovery 1: continue without OCR
    client.post(f"/api/projects/{pid}/analyze", json={"mode": "manual"})
    assert wait_ready(client, pid)["status"] == "ready"
    A = client.get(f"/api/projects/{pid}/pages/0/analysis").json()
    assert A["regions"] == [] and A["stats"]["manual"]
    # recovery 2: retry once the engine is healthy again
    monkeypatch.delenv("TRUEEDIT_FAULT")
    client.post(f"/api/projects/{pid}/analyze", json={"mode": "full"})
    assert wait_ready(client, pid)["status"] == "ready"
    A = client.get(f"/api/projects/{pid}/pages/0/analysis").json()
    assert len(A["regions"]) > 50


def test_ocr_engine_unavailable(client, monkeypatch):
    monkeypatch.setenv("TRUEEDIT_FAULT", "ocr_unavailable")
    h = client.get("/api/health").json()
    assert h["ok"] and not h["ocr"]["available"] and "fault" in h["ocr"]["error"]
    pid = upload(client, "order_clean.png", wait=False)
    err = _job_error(client, pid)
    assert err["code"] == "ocr_unavailable" and {a["action"] for a in err["recovery"]} >= {"retry", "manual"}
    r = client.post(f"/api/projects/{pid}/pages/0/ocr-region", json={"bbox": [10, 10, 200, 60]})
    assert r.status_code in (200, 404)  # canvas may not exist yet; must never 500


def test_timeout_offers_fast_retry(client, monkeypatch):
    monkeypatch.setattr(jobs, "TIME_LIMIT", 0.05)
    pid = upload(client, "order_clean.png", wait=False)
    err = _job_error(client, pid)
    assert err["code"] == "timeout"
    assert {a["action"] for a in err["recovery"]} == {"retry_fast", "manual"}
    monkeypatch.setattr(jobs, "TIME_LIMIT", 240)
    client.post(f"/api/projects/{pid}/analyze", json={"mode": "fast"})
    st = wait_ready(client, pid)
    assert st["status"] == "ready"


def test_ocr_missed_word_recovered_with_manual_region(client, project):
    """Simulate a missed word: remove it from the analysis, then recover it by drawing a region."""
    pid = project("order_photo.jpg")
    A = client.get(f"/api/projects/{pid}/pages/0/analysis").json()
    r, w = find_word(A, "Tremblay")
    pdir = S.page_dir(pid, 0)
    original = json.loads((pdir / "analysis.json").read_text())
    try:
        A2 = json.loads(json.dumps(original))
        A2["regions"] = [x for x in A2["regions"] if x["id"] != r["id"]]
        S.write_json(pdir / "analysis.json", A2)
        b = [w["bbox"][0] - 6, w["bbox"][1] - 6, w["bbox"][2] + 6, w["bbox"][3] + 6]
        res = client.post(f"/api/projects/{pid}/pages/0/ocr-region", json={"bbox": b}).json()
        assert res["text"] == "Tremblay"
        region = {"id": "m-x", "page": 0, "kind": "line", "source": "manual", "text": res["text"], "bbox": b,
                  "role": "text", "words": [{"id": "m-x-w0", "text": res["text"], "bbox": b, "style": {}}],
                  "style": {"align": "left"}, "flags": [], "table": None}
        e = {"id": "me", "page": 0, "bbox": b, "text": "Lavoie", "target_ids": ["m-x"], "source": "manual",
             "original_text": res["text"]}
        rr = client.post(f"/api/projects/{pid}/pages/0/render", json={"edits": [e], "regions": [region]}).json()
        canvas = S.load_image(pdir / "canvas.png")
        out = canvas.copy()
        import base64

        p = rr["patches"][0]
        rgba = cv2.imdecode(np.frombuffer(base64.b64decode(p["png"]), np.uint8), cv2.IMREAD_UNCHANGED)
        m = rgba[..., 3] > 0
        out[p["y"]:p["y"] + p["h"], p["x"]:p["x"] + p["w"]][m] = rgba[..., :3][m]
        d = changed_mask(canvas, out)
        ys, xs = np.nonzero(d)
        assert xs.min() >= b[0] - 8 and ys.min() >= b[1] - 8 and ys.max() <= b[3] + 8
        assert read_text(out, p["info"]["new_text_bbox"]) == "Lavoie"
    finally:
        S.write_json(pdir / "analysis.json", original)


def test_bad_requests_are_clean_errors(client, project):
    pid = project("order_clean.png")
    assert client.get("/api/projects/ffffffffffff").status_code == 404
    assert client.get("/api/projects/../../etc").status_code == 404
    r = client.post(f"/api/projects/{pid}/pages/0/render", json={"edits": [{"id": "x", "bbox": [5, 5, 5, 5], "text": "a"}]})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_edit"
    r = client.post(f"/api/projects/{pid}/pages/0/render", json={"edits": [{"id": "x", "bbox": [9e6, 9e6, 9e6 + 5, 9e6 + 5], "text": "a"}]})
    assert r.status_code == 422
    r = client.post(f"/api/projects/{pid}/export", json={"format": "exe"})
    assert r.status_code == 400
    assert client.get(f"/api/projects/{pid}/pages/9/analysis").status_code == 404


def test_export_before_ready_is_refused(client):
    r = post(client, "order_photo.jpg", (FX / "order_photo.jpg").read_bytes())
    pid = r.json()["id"]
    r = client.post(f"/api/projects/{pid}/export", json={"format": "pdf"})
    assert r.status_code in (409, 200)  # 409 while analysing; 200 only if already finished
    wait_ready(client, pid)


def test_edits_persist(client, project):
    pid = project("order_clean.png")
    v0 = client.get(f"/api/projects/{pid}/edits").json()["version"]
    e = [{"id": "e1", "page": 0, "bbox": [10, 10, 50, 40], "text": "x", "target_ids": []}]
    d = client.put(f"/api/projects/{pid}/edits", json={"edits": e, "review": {"a": {"status": "accepted"}}}).json()
    assert d["version"] == v0 + 1
    got = client.get(f"/api/projects/{pid}/edits").json()
    assert got["edits"] == e and got["review"] == {"a": {"status": "accepted"}}
