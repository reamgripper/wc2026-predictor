"""
Ingestion layer — scrapes a match preview URL and returns clean Markdown.

Strategy (tried in order):
  1. trafilatura  — fast, handles most news sites (ESPN, Sky Sports, etc.)
  2. requests + markdownify — full-page HTML-to-Markdown fallback
  3. crawl4ai  — optional JS-rendered scrape (requires `pip install crawl4ai`)

Features:
  - Per-domain rate limiting (RATE_LIMIT_SECONDS between requests to same host)
  - Randomised User-Agent rotation
  - Configurable HTTP timeout
"""
from __future__ import annotations

import asyncio
import random
import re
import time
from typing import Optional, Tuple
from urllib.parse import urlparse

import requests

# ── Constants ────────────────────────────────────────────────────────────────

RATE_LIMIT_SECONDS = 3.0

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4 Mobile/15E148 Safari/604.1",
]

_LAST_HIT: dict[str, float] = {}


# ── Public API ────────────────────────────────────────────────────────────────

class ScrapingError(Exception):
    """Raised when all extraction methods fail."""


def scrape_url(url: str, timeout: int = 30) -> Tuple[str, str]:
    """
    Fetch *url* and return ``(markdown_text, page_title)``.

    Raises ``ScrapingError`` if every method fails.
    """
    _rate_limit(urlparse(url).netloc)

    # 1 — trafilatura (best article-body extraction)
    md = _via_trafilatura(url, timeout)
    if _looks_good(md):
        return md, _extract_title(md)

    # 2 — plain requests + markdownify (whole-page fallback)
    html = _fetch_html(url, timeout)
    if html:
        md = _html_to_markdown(html)
        if _looks_good(md):
            return md, _extract_title(md)

    # 3 — crawl4ai (JS-rendered pages; optional dependency)
    md = _via_crawl4ai(url, timeout)
    if _looks_good(md):
        return md, _extract_title(md)

    # Final fallback: return whatever we have even if short
    for candidate in [md, _html_to_markdown(html or ""), ""]:
        if candidate and len(candidate.strip()) > 50:
            return candidate, _extract_title(candidate)

    raise ScrapingError(
        f"Could not extract meaningful text from {url}.\n"
        "Possible reasons: paywalled content, heavy JS rendering, or anti-bot protection.\n"
        "Try installing crawl4ai (`pip install crawl4ai && playwright install`) for JS sites."
    )


# ── Private helpers ───────────────────────────────────────────────────────────

def _rate_limit(domain: str) -> None:
    now = time.monotonic()
    wait = RATE_LIMIT_SECONDS - (now - _LAST_HIT.get(domain, 0))
    if wait > 0:
        time.sleep(wait)
    _LAST_HIT[domain] = time.monotonic()


def _default_headers() -> dict:
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }


def _fetch_html(url: str, timeout: int) -> Optional[str]:
    try:
        resp = requests.get(url, headers=_default_headers(), timeout=timeout, allow_redirects=True)
        resp.raise_for_status()
        return resp.text
    except Exception:
        return None


def _via_trafilatura(url: str, timeout: int) -> Optional[str]:
    try:
        import trafilatura
        html = trafilatura.fetch_url(url, no_ssl=False)
        if not html:
            return None
        result = trafilatura.extract(
            html,
            output_format="markdown",
            include_links=False,
            include_images=False,
            favor_recall=True,
            include_tables=True,
        )
        return result
    except Exception:
        return None


def _html_to_markdown(html: str) -> str:
    try:
        from markdownify import markdownify as md
        return md(
            html,
            heading_style="ATX",
            strip=["script", "style", "nav", "footer", "header", "aside", "iframe"],
        )
    except ImportError:
        pass
    # Bare strip fallback
    text = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", text).strip()


def _via_crawl4ai(url: str, timeout: int) -> Optional[str]:
    try:
        from crawl4ai import AsyncWebCrawler  # type: ignore

        async def _run() -> Optional[str]:
            async with AsyncWebCrawler(verbose=False) as crawler:
                result = await crawler.arun(
                    url=url,
                    word_count_threshold=100,
                    exclude_external_links=True,
                    remove_overlay_elements=True,
                    bypass_cache=True,
                )
                return result.markdown if result.success else None

        # Handle already-running event loop (Streamlit / Jupyter)
        try:
            loop = asyncio.get_running_loop()
            import nest_asyncio  # type: ignore
            nest_asyncio.apply(loop)
            return loop.run_until_complete(_run())
        except RuntimeError:
            return asyncio.run(_run())
    except Exception:
        return None


def _looks_good(text: Optional[str], min_chars: int = 300) -> bool:
    return bool(text) and len(text.strip()) >= min_chars


def _extract_title(md: str) -> str:
    for line in md.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
        if stripped.startswith("## "):
            return stripped[3:].strip()
    for line in md.splitlines():
        if line.strip():
            return line.strip()[:120]
    return "Untitled Preview"
