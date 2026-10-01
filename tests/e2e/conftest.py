"""Live server + real Chromium for end-to-end tests."""
import os
import socket
import threading
import time

import pytest

CHROME = os.environ.get("TRUEEDIT_CHROME", "/opt/pw-browsers/chromium-1194/chrome-linux/chrome")


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture(scope="session")
def live_server():
    import uvicorn

    from trueedit.app import app

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.1)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    th.join(timeout=5)


@pytest.fixture(scope="session")
def browser():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        kw = {"executable_path": CHROME} if os.path.exists(CHROME) else {}
        b = p.chromium.launch(**kw)
        yield b
        b.close()


class Watch:
    """Collects console errors, page errors and failed API calls."""

    def __init__(self, page):
        self.errors, self.bad = [], []
        page.on("console", lambda m: self.errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: self.errors.append(str(e)))
        page.on("response", lambda r: self.bad.append((r.status, r.url)) if r.status >= 500 else None)

    def assert_clean(self, allow_4xx=True):
        assert not self.errors, self.errors
        assert not self.bad, self.bad


@pytest.fixture
def desktop(browser):
    ctx = browser.new_context(viewport={"width": 1400, "height": 900}, accept_downloads=True)
    page = ctx.new_page()
    page.watch = Watch(page)
    yield page
    ctx.close()


@pytest.fixture
def mobile(browser):
    ctx = browser.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=3, is_mobile=True,
                              has_touch=True, accept_downloads=True,
                              user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 "
                                         "(KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1")
    page = ctx.new_page()
    page.watch = Watch(page)
    yield page
    ctx.close()
