"""Shared fetch helper: legislation.gov.uk and Find Case Law sit behind an
AWS WAF JS challenge that plain HTTP clients can't pass. A real (headless)
browser passes it after a few seconds, so we drive one Playwright browser
for every fetch in a session rather than spinning one up per request.
"""
from __future__ import annotations

import time
from pathlib import Path

from playwright.sync_api import sync_playwright

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
)


class WafBrowser:
    """Reusable headless-browser session for fetching WAF-protected pages."""

    def __init__(self) -> None:
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=True)
        self._page = self._browser.new_page(user_agent=USER_AGENT)

    def get(self, url: str, settle_seconds: float = 8.0, timeout_ms: int = 30000) -> str:
        """Fetch a URL, waiting out the WAF challenge, and return page content (text/XML)."""
        self._page.goto(url, wait_until="networkidle", timeout=timeout_ms)
        self._page.wait_for_timeout(int(settle_seconds * 1000))
        try:
            self._page.wait_for_load_state("networkidle", timeout=15000)
        except Exception:
            pass

        # XML documents get wrapped by Chrome's built-in XML viewer; pull the raw
        # source back out rather than returning the pretty-printed viewer HTML.
        raw = self._page.evaluate(
            """() => {
                const src = document.getElementById('webkit-xml-viewer-source-xml');
                if (src) return src.innerHTML;
                return document.body ? document.body.innerText : document.documentElement.outerHTML;
            }"""
        )
        return raw

    def get_via_inpage_fetch(self, warmup_url: str, xml_url: str, settle_seconds: float = 6.0) -> str:
        """Find Case Law serves data.xml as a browser download rather than a
        navigable page, so page.goto() can't retrieve it directly. Instead,
        load the human-readable case page first (passes the WAF challenge and
        sets its cookie), then fetch the XML via an in-page fetch() call that
        rides on the same session.
        """
        self._page.goto(warmup_url, wait_until="networkidle", timeout=30000)
        self._page.wait_for_timeout(int(settle_seconds * 1000))
        try:
            self._page.wait_for_load_state("networkidle", timeout=15000)
        except Exception:
            pass
        return self._page.evaluate(
            """async (url) => {
                const r = await fetch(url);
                return await r.text();
            }""",
            xml_url,
        )

    def close(self) -> None:
        self._browser.close()
        self._pw.stop()

    def __enter__(self) -> "WafBrowser":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def fetch_and_save(browser: WafBrowser, url: str, dest: Path, *, polite_delay: float = 2.0) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [skip, cached] {dest.name}")
        return
    print(f"  fetching {url}")
    content = browser.get(url)
    dest.write_text(content, encoding="utf-8")
    time.sleep(polite_delay)
