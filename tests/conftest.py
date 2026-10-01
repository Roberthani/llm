"""Shared fixtures. Uses the real OCR engines — nothing in the analysis/edit path is mocked."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FX = ROOT / "tests" / "fixtures" / "out"
RESULTS = ROOT / "test-results"
os.environ.setdefault("TRUEEDIT_DATA", tempfile.mkdtemp(prefix="trueedit-test-"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests" / "fixtures"))

REQUIRED = ["order_photo.jpg", "invoice_photo.jpg", "order_multipage.pdf", "order.pdf", "order.heic"]


def pytest_sessionstart(session):
    if not all((FX / f).exists() for f in REQUIRED):
        subprocess.run([sys.executable, str(ROOT / "tests" / "fixtures" / "generate.py")], check=True)
    RESULTS.mkdir(exist_ok=True)


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from trueedit.app import app

    with TestClient(app) as c:
        yield c


def wait_ready(client, pid, timeout=300):
    t = time.time()
    while time.time() - t < timeout:
        st = client.get(f"/api/projects/{pid}/status").json()
        if st["status"] in ("ready", "error"):
            return st
        time.sleep(0.3)
    raise TimeoutError(pid)


_PROJECTS: dict[str, str] = {}


def upload(client, name, wait=True):
    r = client.post("/api/projects", files={"file": (name, (FX / name).read_bytes())})
    assert r.status_code == 201, r.text
    pid = r.json()["id"]
    if wait:
        st = wait_ready(client, pid)
        assert st["status"] == "ready", st
    return pid


@pytest.fixture(scope="session")
def project(client):
    """Session cache: fixture name -> analysed project id (fresh edits each time it is requested)."""

    def get(name):
        if name not in _PROJECTS:
            _PROJECTS[name] = upload(client, name)
        pid = _PROJECTS[name]
        client.put(f"/api/projects/{pid}/edits", json={"edits": [], "review": {}, "regions": []})
        return pid

    return get


def save_result(name: str, data):
    (RESULTS / name).write_text(json.dumps(data, indent=1, default=str))
