"""HTTP fetcher for VLR.gg with Cloudflare-resistant impersonation.

Mirrors the anti-bot strategy used by the main site's RefreshLiveData:
curl_cffi JA3 impersonation (newest Chrome first) -> older Chrome -> cloudscraper.
Polite delays + bounded retries with backoff. Pure fetch; no parsing.
"""
from __future__ import annotations

import time

BASE = "https://www.vlr.gg"

# Newest first; if VLR/Cloudflare rejects one JA3 we fall back to an older one.
_IMPERSONATIONS = ["chrome131", "chrome124", "chrome120", "chrome116"]

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


class FetchError(RuntimeError):
    pass


def _curl_cffi_get(url: str, timeout: int):
    from curl_cffi import requests as creq

    last = None
    for imp in _IMPERSONATIONS:
        try:
            r = creq.get(url, impersonate=imp, timeout=timeout)
            if r.status_code == 200:
                return r.text
            last = FetchError(f"HTTP {r.status_code} via {imp}")
        except Exception as e:  # noqa: BLE001 - want to try the next JA3
            last = e
    raise last if last else FetchError("curl_cffi: no impersonation worked")


def _cloudscraper_get(url: str, timeout: int):
    import cloudscraper

    scraper = cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "darwin", "mobile": False}
    )
    scraper.headers.update({"User-Agent": _UA})
    r = scraper.get(url, timeout=timeout)
    if r.status_code != 200:
        raise FetchError(f"cloudscraper HTTP {r.status_code}")
    return r.text


def fetch(url: str, *, timeout: int = 30, retries: int = 3, backoff: float = 4.0) -> str:
    """Fetch a URL's HTML, trying curl_cffi then cloudscraper, with retries."""
    attempt = 0
    last = None
    while attempt < retries:
        for getter in (_curl_cffi_get, _cloudscraper_get):
            try:
                return getter(url, timeout)
            except Exception as e:  # noqa: BLE001
                last = e
        attempt += 1
        if attempt < retries:
            time.sleep(backoff * attempt)
    raise FetchError(f"failed to fetch {url}: {last}")


def match_urls(match_id: str) -> dict[str, str]:
    """The three pages needed to fully enrich one match (series)."""
    mid = str(match_id)
    return {
        "overview": f"{BASE}/{mid}",
        "economy": f"{BASE}/{mid}/?game=all&tab=economy",
        "performance": f"{BASE}/{mid}/?game=all&tab=performance",
    }
