"""
SearXNG / Searx URL discovery for match previews.

Queries a SearXNG instance (local or public) via its JSON API and returns
structured search results that the user can pick from before scraping.

SearXNG JSON endpoint: GET {base_url}/search?q=...&format=json&categories=general
Docs: https://docs.searxng.org/dev/searxng_extra/index.html

Quick-start (local Docker):
  docker run -d -p 8080:8080 searxng/searxng
  → use  http://localhost:8080  as your instance URL

Public instances (rate-limited, use sparingly):
  https://search.sapti.me   https://searx.be   https://searxng.world
"""
from __future__ import annotations

import random
import re
import time
from typing import List, Optional
from urllib.parse import quote_plus, urljoin

import requests

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_INSTANCE = "https://searxng.world"

# Public instances known to allow automated JSON/HTML requests.
# sx.catgirl.cloud blocks headless requests with a bot-challenge page.
PUBLIC_INSTANCES = [
    "https://searxng.world",
    "https://search.sapti.me",
    "https://searx.be",
    "https://paulgo.io",
    "http://localhost:8080",   # local Docker
]

# Sites most likely to have pre-match analysis (used for query boosting)
PREVIEW_SITES = [
    "site:skysports.com",
    "site:bbc.co.uk/sport",
    "site:espn.com",
    "site:optaanalyst.com",
    "site:theathletic.com",
    "site:whoscored.com",
    "site:90min.com",
    "site:goal.com",
    "site:theguardian.com/football",
]

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
]

_LAST_HIT: dict[str, float] = {}
RATE_LIMIT = 1.5  # seconds between requests to the same SearXNG instance


# ── Public types ──────────────────────────────────────────────────────────────

class SearchResult:
    """A single result returned by SearXNG."""
    __slots__ = ("title", "url", "snippet", "engine", "score")

    def __init__(
        self,
        title: str,
        url: str,
        snippet: str = "",
        engine: str = "",
        score: float = 0.0,
    ):
        self.title   = title
        self.url     = url
        self.snippet = snippet
        self.engine  = engine
        self.score   = score

    def __repr__(self) -> str:
        return f"SearchResult(title={self.title!r}, url={self.url!r})"

    def as_dict(self) -> dict:
        return {
            "title":   self.title,
            "url":     self.url,
            "snippet": self.snippet,
            "engine":  self.engine,
            "score":   self.score,
        }


class SearchError(Exception):
    """Raised when the SearXNG instance is unreachable or returns an error."""


# ── Core search function ──────────────────────────────────────────────────────

def search_match_previews(
    home_team: str,
    away_team: str,
    *,
    extra_terms: str = "preview analysis",
    tournament: str = "",
    instance_url: str = DEFAULT_INSTANCE,
    max_results: int = 12,
    site_filter: Optional[str] = None,
    language: str = "en",
) -> List[SearchResult]:
    """
    Search *instance_url* for pre-match previews of *home_team* vs *away_team*.

    Parameters
    ----------
    home_team, away_team : str
        Team names used to build the query.
    extra_terms : str
        Appended to the query, e.g. ``"preview analysis"``.
    tournament : str
        Optional tournament name appended to focus results.
    instance_url : str
        SearXNG base URL.
    max_results : int
        Maximum number of results to return (SearXNG typically pages at 10).
    site_filter : str | None
        Restrict to one domain, e.g. ``"skysports.com"``.  When set the
        site: operator is prepended to the query and the broad site-list
        boosting is skipped.
    language : str
        SearXNG language parameter.

    Returns
    -------
    list[SearchResult]
        Deduplicated, sorted by relevance score.

    Raises
    ------
    SearchError
        If the instance is unreachable or returns a non-200 response.
    """
    query = _build_query(home_team, away_team, extra_terms, tournament, site_filter)
    results = _fetch_results(query, instance_url=instance_url, language=language)

    # Deduplicate by URL, preserve order
    seen: set[str] = set()
    unique: List[SearchResult] = []
    for r in results:
        if r.url not in seen:
            seen.add(r.url)
            unique.append(r)

    return unique[:max_results]


def check_instance(instance_url: str, timeout: int = 6) -> dict:
    """
    Probe *instance_url* and return a status dict::

        {"reachable": bool, "json_enabled": bool, "html_enabled": bool, "note": str}

    All callers should check ``reachable`` first; ``json_enabled`` is only
    meaningful when ``reachable`` is True.
    """
    base = instance_url.rstrip("/")
    headers = {"User-Agent": random.choice(USER_AGENTS)}
    result = {"reachable": False, "json_enabled": False, "html_enabled": False, "note": ""}

    # 1 — JSON endpoint
    try:
        r = requests.get(
            base + "/search",
            params={"q": "test", "format": "json"},
            headers=headers,
            timeout=timeout,
        )
        result["reachable"] = True
        if r.status_code == 200:
            try:
                r.json()
                result["json_enabled"] = True
            except Exception:
                pass
        elif r.status_code == 403:
            result["note"] = "JSON format disabled on this instance (403). HTML fallback will be used."
    except requests.ConnectionError:
        result["note"] = "Connection refused. Is the instance running?"
        return result
    except Exception as exc:
        result["note"] = str(exc)
        return result

    # 2 — HTML endpoint (always available)
    try:
        r2 = requests.get(
            base + "/search",
            params={"q": "test"},
            headers={**headers, "Accept": "text/html"},
            timeout=timeout,
        )
        result["html_enabled"] = r2.status_code == 200
    except Exception:
        pass

    if not result["note"]:
        if result["json_enabled"]:
            result["note"] = "JSON API available — fastest mode."
        elif result["html_enabled"]:
            result["note"] = "JSON disabled; using HTML fallback (slightly slower)."
        else:
            result["note"] = "Instance reachable but search returned an error."

    return result


def build_query_string(
    home_team: str,
    away_team: str,
    extra_terms: str = "preview analysis",
    tournament: str = "",
) -> str:
    """Return the search query string (visible in the UI so users can edit it)."""
    return _build_query(home_team, away_team, extra_terms, tournament, site_filter=None)


# ── Private helpers ───────────────────────────────────────────────────────────

def _build_query(
    home: str,
    away: str,
    extra: str,
    tournament: str,
    site_filter: Optional[str],
) -> str:
    parts = [f'"{home}" "{away}"']
    if tournament:
        parts.append(f'"{tournament}"')
    parts.append(extra)
    if site_filter:
        parts.insert(0, f"site:{site_filter}")
    return " ".join(parts)


def _fetch_results(
    query: str,
    *,
    instance_url: str,
    language: str,
    timeout: int = 20,
) -> List[SearchResult]:
    """
    Try JSON endpoint first; transparently fall back to HTML parsing if the
    instance has ``format=json`` disabled (common on public instances).
    """
    base = instance_url.rstrip("/")
    _rate_limit(base)

    common_params = {
        "q":          query,
        "language":   language,
        "categories": "general",
        "safesearch": "0",
    }
    headers_base = {"User-Agent": random.choice(USER_AGENTS)}

    # ── 1. Try JSON ────────────────────────────────────────────────────────────
    try:
        resp = requests.get(
            base + "/search",
            params={**common_params, "format": "json"},
            headers={**headers_base, "Accept": "application/json"},
            timeout=timeout,
        )
        if resp.status_code == 200:
            try:
                data = resp.json()
                results = _parse_json_results(data)
                if results:
                    return results
                # JSON worked but returned 0 results — still fall through to HTML
            except Exception:
                pass  # JSON parse failed → try HTML
        # 429 is fatal regardless of format
        if resp.status_code == 429:
            raise SearchError(
                "Rate-limited by the SearXNG instance. "
                "Wait a moment and try again, or switch to a local Docker instance."
            )
    except SearchError:
        raise
    except requests.ConnectionError:
        raise SearchError(
            f"Cannot reach SearXNG at {instance_url}.\n"
            "Start a local instance:  docker run -d -p 8080:8080 searxng/searxng\n"
            "Or use a public instance (e.g. https://searxng.world) in the settings."
        )
    except requests.Timeout:
        raise SearchError(f"SearXNG at {instance_url} timed out after {timeout}s.")
    except Exception as exc:
        raise SearchError(f"Unexpected error querying SearXNG: {exc}")

    # ── 2. HTML fallback ───────────────────────────────────────────────────────
    _rate_limit(base)
    try:
        resp_html = requests.get(
            base + "/search",
            params=common_params,
            headers={**headers_base, "Accept": "text/html,application/xhtml+xml"},
            timeout=timeout,
        )
        if resp_html.status_code == 429:
            raise SearchError(
                "Rate-limited by the SearXNG instance. "
                "Wait a moment and try again, or switch to a local Docker instance."
            )
        if resp_html.status_code != 200:
            raise SearchError(
                f"SearXNG returned HTTP {resp_html.status_code}. "
                "Check the instance URL or try a different one."
            )
        results = _parse_html_results(resp_html.text)
        return results
    except SearchError:
        raise
    except Exception as exc:
        raise SearchError(f"Both JSON and HTML search failed: {exc}")


def _parse_json_results(raw: dict) -> List[SearchResult]:
    results = []
    for item in raw.get("results", []):
        url = item.get("url", "")
        if not url:
            continue
        results.append(SearchResult(
            title=item.get("title", url),
            url=url,
            snippet=item.get("content", ""),
            engine=", ".join(item.get("engines", [])),
            score=float(item.get("score", 0.0)),
        ))
    results.sort(key=lambda r: r.score, reverse=True)
    return results


def _parse_html_results(html: str) -> List[SearchResult]:
    """
    Parse SearXNG HTML output.

    SearXNG renders results as ``<article class="result ...">`` blocks.
    Falls back to a simple ``<a href>`` heuristic if the article structure
    is absent (some themes differ).
    """
    results: List[SearchResult] = []

    # Try BeautifulSoup (installed via markdownify → html5lib transitive dep)
    try:
        from bs4 import BeautifulSoup  # type: ignore
        soup = BeautifulSoup(html, "html.parser")

        # SearXNG default theme: <article class="result ...">
        articles = soup.find_all("article", class_=re.compile(r"result"))
        for art in articles:
            a_tag = art.find("a", class_=re.compile(r"url_header|result-title")) or art.find("h3", class_=re.compile(r"title"))
            if a_tag:
                link = a_tag.find("a") if a_tag.name != "a" else a_tag
            else:
                link = art.find("a", href=True)
            if not link:
                continue
            url = link.get("href", "")
            if not url or not url.startswith("http"):
                continue
            title = link.get_text(strip=True) or url
            snippet_tag = art.find(class_=re.compile(r"content|snippet|description"))
            snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
            results.append(SearchResult(title=title, url=url, snippet=snippet))

        if results:
            return results

        # Fallback: any <h3><a href> that looks like an external result
        for h3 in soup.find_all("h3"):
            link = h3.find("a", href=True)
            if not link:
                continue
            url = link.get("href", "")
            if not url.startswith("http"):
                continue
            title = link.get_text(strip=True) or url
            results.append(SearchResult(title=title, url=url))

        return results

    except ImportError:
        pass

    # Pure-regex last resort (no bs4)
    for m in re.finditer(
        r'<h[23][^>]*>\s*<a[^>]+href=["\']([^"\']+)["\'][^>]*>([^<]+)</a>',
        html,
        re.IGNORECASE,
    ):
        url, title = m.group(1).strip(), m.group(2).strip()
        if url.startswith("http") and len(title) > 5:
            results.append(SearchResult(title=title, url=url))

    return results


def _rate_limit(domain: str) -> None:
    now = time.monotonic()
    wait = RATE_LIMIT - (now - _LAST_HIT.get(domain, 0.0))
    if wait > 0:
        time.sleep(wait)
    _LAST_HIT[domain] = time.monotonic()
