"""
RefreshLiveData.py — VCT live-data refresh pipeline.

Driven entirely by the live-event window declared in
MoreTestingMaybeFiles.live_events_today():

  1. Discover every event whose date window contains today (with a small lead/trail).
  2. For each region of each live event, scan VLR for completed matches and scrape
     new ones into data/maps/{event_id}.csv and data/series/{event_id}.csv.
  3. Scrape upcoming (un-played) matches for the next 7 days from every live event.
  4. Rebuild match_results.csv, fetch any new match dates, and rebuild the BenPom
     rating timeline.

Future-proofing: to onboard a new VCT event, add one entry to ALL_EVENTS with
start/end dates and the VLR stats URLs.  No other code change is required.

Writes live progress to /tmp/mhub_refresh_progress.json.  PID-locked so only one
instance runs at a time.

Exit code is always 0 on graceful failure; errors are surfaced via the progress
file so the UI can show what went wrong.
"""
import os, sys, json, time, re, subprocess, datetime, traceback, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from bs4 import BeautifulSoup

# Cloudflare on datacenter IPs (Render/Vercel/AWS) flags plain `requests`
# because Python's stdlib ssl has a distinctive JA3 fingerprint.  We try
# strategies in order of resilience and log which one finally succeeded
# so the progress file shows what's actually working on each host.
#
# Strategy 1: curl_cffi with chrome131 impersonation (most recent JA3,
#             usually enough for VLR).
# Strategy 2: curl_cffi with chrome120 (older JA3, sometimes evades when
#             newer fingerprints are pattern-matched).
# Strategy 3: cloudscraper — has a built-in solver for Cloudflare's
#             classic JS challenge, in case curl_cffi gets a v2 prompt.
# Strategy 4: plain requests — last-resort.
#
# Each module is imported in its own try/except so a missing optional
# dependency never breaks the others.
_curl_cffi_err = None
_cloudscraper_err = None
try:
    from curl_cffi import requests as cffi_requests  # type: ignore
    _CFFI_AVAILABLE = True
    try:
        import curl_cffi as _cc_mod  # type: ignore
        _CFFI_VERSION = getattr(_cc_mod, "__version__", "?")
    except Exception:
        _CFFI_VERSION = "?"
except Exception as _e:
    cffi_requests = None
    _CFFI_AVAILABLE = False
    _CFFI_VERSION = "n/a"
    _curl_cffi_err = f"{type(_e).__name__}: {_e}"

try:
    import cloudscraper  # type: ignore
    _CS_AVAILABLE = True
except Exception as _e:
    cloudscraper = None
    _CS_AVAILABLE = False
    _cloudscraper_err = f"{type(_e).__name__}: {_e}"

_cloudscraper_session = None  # lazy-init


def _get_cloudscraper():
    global _cloudscraper_session
    if not _CS_AVAILABLE:
        return None
    if _cloudscraper_session is None:
        _cloudscraper_session = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "darwin", "mobile": False}
        )
    return _cloudscraper_session

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from MoreTestingMaybeFiles import ALL_EVENTS, live_events_today, _parse_vlr_stats_url

PROGRESS_FILE = "/tmp/mhub_refresh_progress.json"
LOCK_FILE     = "/tmp/mhub_refresh.lock"

# Browser-shaped headers — Render/AWS IPs often fail Cloudflare's bot check
# without enough of the modern client hints.  Keep this in sync with whatever
# real browsers send.  A failed Cloudflare challenge will be detected and
# logged so the operator can flip to a proxy if VLR ever locks down harder.
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/130.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,"
              "image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Referer": "https://www.vlr.gg/",
    "Sec-Ch-Ua": '"Chromium";v="130", "Google Chrome";v="130", "Not?A_Brand";v="99"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"macOS"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}

# VLR team-name → org-code map.  Anything missing falls back to the literal name
# (operator can backfill VLR_NAME_TO_ORG as new teams appear).
VLR_NAME_TO_ORG = {
    '100 Thieves': '100T', 'BBL Esports': 'BBL', 'Cloud9': 'C9',
    'DetonatioN FocusMe': 'DFM', 'ENVY': 'ENVY', 'Eternal Fire': 'EF',
    'Evil Geniuses': 'EG', 'FNATIC': 'FNC', 'FULL SENSE': 'FS',
    'FURIA': 'FUR', 'FUT Esports': 'FUT', 'G2 Esports': 'G2',
    'GIANTX': 'GX', 'Gen.G': 'GEN', 'Gentle Mates': 'M8',
    'Global Esports': 'GE', 'KRÜ Esports': 'KRÜ', 'Karmine Corp': 'KC',
    'Kiwoom DRX': 'KRX', 'LEVIATÁN': 'LEV', 'LOUD': 'LOUD',
    'MIBR': 'MIBR', 'NRG': 'NRG', 'Natus Vincere': 'NAVI',
    'Nongshim RedForce': 'NS', 'PCIFIC Esports': 'PCF', 'Paper Rex': 'PRX',
    'Rex Regum Qeon': 'RRQ', 'Sentinels': 'SEN', 'T1': 'T1',
    'Team Heretics': 'TH', 'Team Liquid': 'TL', 'Team Secret': 'TS',
    'Team Vitality': 'VIT', 'VARREL': 'VL', 'ZETA DIVISION': 'ZETA',
}

# Cloudflare challenge fingerprints — if any of these show up in a response body
# we treat the page as unscrapeable and log loudly instead of silently parsing.
_CLOUDFLARE_FINGERPRINTS = (
    "Just a moment",
    "cf-challenge",
    "challenge-platform",
    "__cf_chl_",
    "Attention Required",
)

_log_entries = []
_error_entries = []


# ── Progress reporting ────────────────────────────────────────────────────────

def _write(phase, pct, message, extra_log=None, error=None):
    if extra_log:
        _log_entries.extend(extra_log if isinstance(extra_log, list) else [extra_log])
    if error:
        _error_entries.append(error)
    # Summarize bypass strategy outcomes so the progress file shows which
    # one actually worked on this host (critical for diagnosing Render).
    succ_counts = {}
    fail_counts = {}
    for a in _strategy_log["attempts"]:
        bucket = succ_counts if a["ok"] else fail_counts
        bucket[a["strategy"]] = bucket.get(a["strategy"], 0) + 1
    data = {
        "phase":   phase,
        "pct":     pct,
        "message": message,
        "log":     list(_log_entries[-30:]),
        "errors":  list(_error_entries[-10:]),
        "ts":      time.time(),
        "fetch":   {
            "success_by_strategy": succ_counts,
            "fail_by_strategy":    fail_counts,
            "first_success":       _strategy_log["first_success"],
            "total_attempts":      len(_strategy_log["attempts"]),
        },
    }
    try:
        with open(PROGRESS_FILE, "w") as f:
            json.dump(data, f)
    except Exception as e:
        print(f"  progress-file write failed: {e}", flush=True)
    print(f"  [{pct:3d}%] {message}", flush=True)
    if extra_log:
        for line in (extra_log if isinstance(extra_log, list) else [extra_log]):
            print(f"         {line}", flush=True)
    if error:
        print(f"         ERROR: {error}", flush=True)


# ── HTTP helper ───────────────────────────────────────────────────────────────

_strategy_log = {"available": None, "first_success": None, "attempts": []}


def _record_strategy(used, success, status=None, length=None, cf=False, err=None):
    _strategy_log["attempts"].append({
        "strategy": used, "ok": success,
        "status": status, "len": length, "cf": cf, "err": err,
    })
    if success and _strategy_log["first_success"] is None:
        _strategy_log["first_success"] = used


def _looks_like_cloudflare(text):
    return bool(text) and any(fp in text for fp in _CLOUDFLARE_FINGERPRINTS) and len(text) < 60000


def _try_strategy(strategy, url, timeout):
    """Returns (status, text, err).  Raises nothing — captures all failures."""
    try:
        if strategy == "curl_cffi:chrome131":
            r = cffi_requests.get(url, headers=HEADERS, timeout=timeout,
                                  impersonate="chrome131", allow_redirects=True)
            return r.status_code, r.text or "", None
        if strategy == "curl_cffi:chrome120":
            r = cffi_requests.get(url, headers=HEADERS, timeout=timeout,
                                  impersonate="chrome120", allow_redirects=True)
            return r.status_code, r.text or "", None
        if strategy == "curl_cffi:chrome":
            r = cffi_requests.get(url, headers=HEADERS, timeout=timeout,
                                  impersonate="chrome", allow_redirects=True)
            return r.status_code, r.text or "", None
        if strategy == "cloudscraper":
            sess = _get_cloudscraper()
            if sess is None:
                return None, "", "cloudscraper not available"
            r = sess.get(url, timeout=timeout)
            return r.status_code, r.text or "", None
        if strategy == "requests":
            r = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
            return r.status_code, r.text or "", None
        return None, "", f"unknown strategy {strategy}"
    except Exception as e:
        return None, "", f"{type(e).__name__}: {e}"


def _fetch(url, *, timeout=15, retries=None, backoff=None):
    """
    GET `url` and return BeautifulSoup or None.

    Tries multiple bypass strategies in order of expected resilience, stopping
    at the first that returns real (non-Cloudflare, 2xx) HTML.  Every attempt's
    outcome is recorded in `_strategy_log` so the progress file shows which
    strategy worked (or that all failed and why).

    `retries` / `backoff` are accepted but ignored — the strategy chain itself
    is the retry mechanism.  Kept in the signature so existing callers don't
    raise TypeError.
    """
    del retries, backoff  # silence linters; intentionally unused
    strategies = []
    if _CFFI_AVAILABLE:
        strategies += ["curl_cffi:chrome131", "curl_cffi:chrome120", "curl_cffi:chrome"]
    if _CS_AVAILABLE:
        strategies.append("cloudscraper")
    strategies.append("requests")

    last_err = None
    for strat in strategies:
        status, text, err = _try_strategy(strat, url, timeout)
        if err is not None:
            _record_strategy(strat, False, err=err)
            last_err = f"{strat} → {err}"
            continue
        if status in (403, 429) or (status is not None and status >= 500):
            _record_strategy(strat, False, status=status, length=len(text))
            last_err = f"{strat} → HTTP {status}"
            continue
        if _looks_like_cloudflare(text):
            _record_strategy(strat, False, status=status, length=len(text), cf=True)
            last_err = f"{strat} → Cloudflare challenge ({len(text)}B)"
            continue
        # Success.
        _record_strategy(strat, True, status=status, length=len(text))
        return BeautifulSoup(text, "html.parser")

    if last_err:
        _error_entries.append(f"{last_err} on {url}")
        print(f"  fetch failed (all strategies): {last_err} for {url}", flush=True)
    return None


# ── VLR helpers ───────────────────────────────────────────────────────────────

def _get_completed_urls(vlr_id, slug):
    url = f"https://www.vlr.gg/event/matches/{vlr_id}/{slug}/"
    soup = _fetch(url)
    if soup is None:
        return []
    out = []
    for a in soup.select("a.wf-module-item.match-item"):
        href = a.get("href", "")
        status_el = a.select_one(".ml-status")
        if not status_el or status_el.get_text(strip=True).lower() != "completed":
            continue
        if re.match(r"^/\d+/", href):
            full = "https://www.vlr.gg" + href
            if full not in out:
                out.append(full)
    return out


def _match_id_from_url(url):
    m = re.search(r"/(\d+)/", url)
    return m.group(1) if m else None


def _existing_match_ids(event_csv_id):
    import pandas as pd
    p = os.path.join(ROOT, "data", "maps", f"{event_csv_id}.csv")
    if not os.path.exists(p):
        return set()
    try:
        df = pd.read_csv(p, usecols=["MatchID"])
        return set(df["MatchID"].dropna().astype(str).tolist())
    except Exception:
        return set()


def _scrape_match_page(url, region_tag):
    """Returns (map_rows, series_rows, match_name_str)."""
    soup = _fetch(url)
    if soup is None:
        return [], [], "? (fetch failed)"

    fmt_el  = soup.select_one(".match-header-vs-note")
    fmt_raw = fmt_el.get_text(strip=True).lower() if fmt_el else ""
    series_fmt = "bo5" if ("bo5" in fmt_raw or "best of 5" in fmt_raw) else (
                  "bo1" if ("bo1" in fmt_raw or "best of 1" in fmt_raw) else "bo3")

    mid = _match_id_from_url(url) or ""

    teams_el = soup.select(".match-header-link-name .wf-title-med")
    team_a = teams_el[0].get_text(strip=True) if len(teams_el) > 0 else "?"
    team_b = teams_el[1].get_text(strip=True) if len(teams_el) > 1 else "?"
    scores_el = soup.select(".match-header-vs-score .js-spoiler")
    score_a = scores_el[0].get_text(strip=True) if len(scores_el) > 0 else "?"
    score_b = scores_el[1].get_text(strip=True) if len(scores_el) > 1 else "?"
    display = f"{team_a} {score_a}–{score_b} {team_b}"

    map_rows, series_rows = [], []

    for game_div in soup.select("div.vm-stats-game"):
        game_id = game_div.get("data-game-id", "")
        is_all  = (game_id == "all")
        map_name = ""
        if not is_all:
            hdr = game_div.select_one(".vm-stats-game-header .map")
            if hdr:
                fd = hdr.find("div")
                if fd:
                    map_name = fd.get_text(strip=True)

        for table in game_div.select("table.wf-table-inset.mod-overview"):
            tbody = table.find("tbody")
            if not tbody:
                continue
            for tr in tbody.find_all("tr"):
                tds = tr.find_all("td")
                if len(tds) < 10:
                    continue
                ptd    = tds[0]
                pname  = ptd.select_one(".text-of")
                porg   = ptd.select_one(".ge-text-light")
                pa     = ptd.find("a", href=True)
                player = pname.get_text(strip=True) if pname else ""
                org    = porg.get_text(strip=True)  if porg  else ""
                if not player:
                    continue

                def _s(td):
                    sp = td.find("span", class_=lambda c: c and "mod-both" in c.split())
                    return sp.get_text(strip=True) if sp else td.get_text(strip=True)

                row = {
                    "Player":       player,
                    "Org":          org,
                    "ProfileURL":   ("https://www.vlr.gg" + pa["href"]) if pa else "",
                    "Region":       region_tag,
                    "MatchID":      mid,
                    "MapNum":       game_id,
                    "MapName":      map_name,
                    "SeriesFormat": series_fmt,
                    "R2.0":   _s(tds[2]) if len(tds) > 2  else "",
                    "ACS":    _s(tds[3]) if len(tds) > 3  else "",
                    "K":      _s(tds[4]) if len(tds) > 4  else "",
                    "D":      _s(tds[5]) if len(tds) > 5  else "",
                    "A":      _s(tds[6]) if len(tds) > 6  else "",
                    "KAST":   _s(tds[8]) if len(tds) > 8  else "",
                    "ADR":    _s(tds[9]) if len(tds) > 9  else "",
                    "HS%":    _s(tds[10]) if len(tds) > 10 else "",
                    "FK":     _s(tds[11]) if len(tds) > 11 else "",
                    "FD":     _s(tds[12]) if len(tds) > 12 else "",
                }
                try:
                    k_i, d_i = int(row["K"]), int(row["D"])
                    row["K:D"] = round(k_i / d_i, 2) if d_i else float(k_i)
                except Exception:
                    row["K:D"] = ""

                (series_rows if is_all else map_rows).append(row)

    return map_rows, series_rows, display


def _scrape_date(mid):
    soup = _fetch(f"https://www.vlr.gg/{mid}/", retries=2)
    if soup is None:
        return None
    el = soup.find("div", class_="moment-tz-convert", attrs={"data-utc-ts": True})
    if el:
        return el["data-utc-ts"][:10]
    return None


# ── Upcoming match scraper ────────────────────────────────────────────────────

def _scrape_upcoming_for(vlr_id, slug, region, event_label):
    """Return upcoming-match dicts for one (event, region) within the next 7 days."""
    from datetime import datetime as _dt
    from bs4 import NavigableString
    today = datetime.date.today()
    cutoff = today + datetime.timedelta(days=7)

    url = f"https://www.vlr.gg/event/matches/{vlr_id}/{slug}/"
    soup = _fetch(url, retries=2)
    if soup is None:
        return []

    container = soup.select_one(".col.mod-1") or soup.body
    if container is None:
        return []

    out = []
    current_date = None
    for el in container.children:
        if isinstance(el, NavigableString):
            continue
        classes = el.get("class") or []

        if "wf-label" in classes and "mod-large" in classes:
            txt = re.sub(r"(Today|Yesterday)$", "", el.get_text(strip=True)).strip()
            try:
                current_date = _dt.strptime(txt, "%a, %B %d, %Y").date().isoformat()
            except ValueError:
                current_date = None
            continue

        if "wf-card" not in classes:
            continue

        for a in el.select("a.wf-module-item.match-item"):
            status_el = a.select_one(".ml-status")
            status = status_el.get_text(strip=True).lower() if status_el else ""
            if status in ("completed", "live"):
                continue

            ts_el = a.select_one(".moment-tz-convert")
            match_date = ts_el["data-utc-ts"][:10] if (ts_el and ts_el.get("data-utc-ts")) else current_date
            if not match_date:
                continue
            try:
                md = datetime.date.fromisoformat(match_date)
            except Exception:
                continue
            if md < today or md > cutoff:
                continue

            teams = a.select(".match-item-vs-team-name")
            if len(teams) < 2:
                continue
            team_a = teams[0].get_text(strip=True)
            team_b = teams[1].get_text(strip=True)
            if not team_a or not team_b or "TBD" in team_a or "TBD" in team_b:
                continue

            fmt_el = a.select_one(".match-item-event-series")
            fmt_raw = fmt_el.get_text(strip=True).lower() if fmt_el else ""
            fmt = "bo5" if ("bo5" in fmt_raw or "best of 5" in fmt_raw) else (
                  "bo1" if ("bo1" in fmt_raw or "best of 1" in fmt_raw) else "bo3")

            out.append({
                "team_a": team_a, "team_b": team_b,
                "org_a":  VLR_NAME_TO_ORG.get(team_a, team_a),
                "org_b":  VLR_NAME_TO_ORG.get(team_b, team_b),
                "date":   match_date,
                "region": region,
                "event":  f"{event_label} — {region}" if region != "International" else event_label,
                "format": fmt,
            })
    return out


# ── Main pipeline ──────────────────────────────────────────────────────────────

def _event_to_target(ev):
    regions = []
    for region, url in ev["regions"].items():
        vlr_id, slug = _parse_vlr_stats_url(url)
        if vlr_id and slug:
            regions.append((region, vlr_id, slug))
    if not regions:
        return None
    return {
        "event_csv_id": ev["id"],
        "label":        ev["label"],
        "regions":      regions,
    }


def _resolve_live_targets():
    """
    Returns a list of target dicts for every event we should poll right now.

    Primary source: every event whose declared date window contains today
    (with a small lead/trail) AND has at least one populated region URL.

    Fallback: if no live event has populated URLs yet (e.g. a future split is
    declared but VLR hasn't posted it), use the most-recent past event with
    populated URLs so the pipeline keeps refreshing the last completed split.
    """
    targets = []
    for ev in live_events_today():
        t = _event_to_target(ev)
        if t:
            targets.append(t)
    if targets:
        return targets

    # Fallback: walk ALL_EVENTS in reverse-chronological order and pick the
    # first one with populated URLs.
    dated = [(ev.get("end") or ev.get("start") or "", ev) for ev in ALL_EVENTS]
    for _, ev in sorted(dated, key=lambda x: x[0], reverse=True):
        t = _event_to_target(ev)
        if t:
            return [t]
    return []


def main():
    import pandas as pd

    today_str = datetime.date.today().isoformat()
    avail_msg = []
    if _CFFI_AVAILABLE:
        avail_msg.append(f"curl_cffi {_CFFI_VERSION}")
    else:
        avail_msg.append(f"curl_cffi ✗ ({_curl_cffi_err})")
    if _CS_AVAILABLE:
        avail_msg.append("cloudscraper ✓")
    else:
        avail_msg.append(f"cloudscraper ✗ ({_cloudscraper_err})")
    _write("checking", 2, f"Checking VCT data — today is {today_str}…",
           [f"Today: {today_str}",
            f"Bypass: {' | '.join(avail_msg)}"])

    targets = _resolve_live_targets()
    if not targets:
        _write("done", 100, "No live VCT events configured for today.",
               ["No event in ALL_EVENTS matched today's date window with a populated VLR URL.",
                "To onboard a new event, add an entry with start/end + region URLs to MoreTestingMaybeFiles.ALL_EVENTS."])
        print("\nNo live events. Done.", flush=True)
        return

    _write("checking", 4,
           f"Live events: {', '.join(t['label'] for t in targets)}",
           [f"Live events ({len(targets)}): " + ", ".join(t["label"] for t in targets)])

    tl_path = os.path.join(ROOT, "data", "rating_timeline.json")
    last_date = "unknown"
    try:
        with open(tl_path) as f:
            tl = json.load(f)
        cps = tl.get("checkpoints", [])
        if cps:
            last_date = cps[-1]["date"]
    except Exception:
        pass
    _write("checking", 6, f"Last ratings checkpoint: {last_date}",
           [f"Last checkpoint: {last_date}"])

    # ── Step 1: Scan VLR for completed matches across every live event ───────
    # All region scans are independent HTTP fetches → parallelize across them.
    all_new_urls = []   # list of (url, region, event_csv_id)
    total_completed = 0
    _scan_jobs = []  # (event_label, region, vlr_id, slug, event_csv_id, existing_set)
    for t in targets:
        existing = _existing_match_ids(t["event_csv_id"])
        _write("checking", 8,
               f"Scanning {t['label']} — {len(existing)} match(es) on disk…",
               [f"[{t['label']}] {len(existing)} match(es) already cached"])
        for region, vlr_id, slug in t["regions"]:
            _scan_jobs.append((t["label"], region, vlr_id, slug, t["event_csv_id"], existing))

    _scan_lock = threading.Lock()
    _scan_done = {"n": 0}
    def _scan_one(job):
        label, region, vlr_id, slug, ev_csv_id, existing = job
        urls = _get_completed_urls(vlr_id, slug)
        new = [u for u in urls if _match_id_from_url(u) not in existing]
        with _scan_lock:
            _scan_done["n"] += 1
            pct = 8 + int(_scan_done["n"] / max(len(_scan_jobs), 1) * 20)
            _write("checking", pct,
                   f"{label} / {region}: {len(urls)} completed, {len(new)} new",
                   [f"✓ {label} / {region}: {len(urls)} completed ({len(new)} new)"])
        return (label, region, ev_csv_id, urls, new)

    with ThreadPoolExecutor(max_workers=4) as _ex:
        for _label, _region, _ev_csv_id, _urls, _new in _ex.map(_scan_one, _scan_jobs):
            total_completed += len(_urls)
            for _u in _new:
                all_new_urls.append((_u, _region, _ev_csv_id))

    _write("checking", 30,
           f"Scan complete — {total_completed} completed across {len(targets)} live event(s), "
           f"{len(all_new_urls)} new to scrape",
           [f"Total completed across all live events: {total_completed}",
            f"New to scrape: {len(all_new_urls)}"])

    # ── Step 2: Scrape upcoming for every live event (always) — parallel ────
    _upc_jobs = []
    for t in targets:
        for region, vlr_id, slug in t["regions"]:
            _upc_jobs.append((vlr_id, slug, region, t["label"]))
    all_upcoming = []
    if _upc_jobs:
        with ThreadPoolExecutor(max_workers=4) as _ex:
            for _upc in _ex.map(lambda j: _scrape_upcoming_for(*j), _upc_jobs):
                all_upcoming.extend(_upc)

    seen = set()
    deduped = []
    for m in sorted(all_upcoming, key=lambda x: x["date"]):
        key = f"{m['team_a']}-{m['team_b']}-{m['date']}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(m)
    out_upc = os.path.join(ROOT, "data", "upcoming_matches.json")
    try:
        with open(out_upc, "w") as f:
            json.dump(deduped, f, indent=2)
        _write("checking", 34,
               f"Upcoming matches: {len(deduped)} in next 7 days",
               [f"Upcoming saved: {len(deduped)} match(es)"])
    except Exception as e:
        _write("checking", 34, "Upcoming write failed",
               error=f"upcoming write failed: {e}")

    if not all_new_urls:
        _write("done", 100,
               f"All match data current through {last_date}",
               [f"✓ No new completed matches — ratings up to date"])
        print("\nNo new completed matches. Done.", flush=True)
        return

    # ── Step 3: Scrape new matches in parallel ───────────────────────────────
    total_new = len(all_new_urls)
    by_event_maps   = {}   # event_csv_id → [row, ...]
    by_event_series = {}

    if total_new:
        _scrape_lock = threading.Lock()
        _scrape_done = {"n": 0}
        def _scrape_one(args):
            url, region, ev_id = args
            mr, sr, display = _scrape_match_page(url, region)
            return (ev_id, region, mr, sr, display)

        with ThreadPoolExecutor(max_workers=4) as _ex:
            _futures = [_ex.submit(_scrape_one, a) for a in all_new_urls]
            for _fut in as_completed(_futures):
                _ev_id, _region, _mr, _sr, _display = _fut.result()
                by_event_maps.setdefault(_ev_id, []).extend(_mr)
                by_event_series.setdefault(_ev_id, []).extend(_sr)
                with _scrape_lock:
                    _scrape_done["n"] += 1
                    pct = 36 + int(_scrape_done["n"] / total_new * 32)
                    _write("scraping", pct,
                           f"Scraping match {_scrape_done['n']}/{total_new}…",
                           [f"  [{_region}] {_display}"])

    # Persist per event
    for ev_id, rows in by_event_maps.items():
        if not rows:
            continue
        path = os.path.join(ROOT, "data", "maps", f"{ev_id}.csv")
        new_df = pd.DataFrame(rows)
        if os.path.exists(path):
            old_df = pd.read_csv(path)
            combined = pd.concat([old_df, new_df], ignore_index=True).drop_duplicates()
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            combined = new_df
        combined.to_csv(path, index=False)
    for ev_id, rows in by_event_series.items():
        if not rows:
            continue
        path = os.path.join(ROOT, "data", "series", f"{ev_id}.csv")
        new_df = pd.DataFrame(rows)
        if os.path.exists(path):
            old_df = pd.read_csv(path)
            combined = pd.concat([old_df, new_df], ignore_index=True).drop_duplicates()
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            combined = new_df
        combined.to_csv(path, index=False)

    # ── Step 4: Rebuild match_results.csv ────────────────────────────────────
    _write("building", 70, "Rebuilding match_results.csv…",
           ["Rebuilding match_results.csv…"])
    try:
        subprocess.run(
            [sys.executable, os.path.join(ROOT, "scrapers", "BuildMatchResults.py")],
            cwd=ROOT, check=True, capture_output=True, timeout=300,
        )
    except subprocess.CalledProcessError as e:
        _write("building", 70, "BuildMatchResults failed",
               error=f"BuildMatchResults: {e.stderr.decode('utf-8','ignore')[-400:] if e.stderr else e}")
    except Exception as e:
        _write("building", 70, "BuildMatchResults failed", error=str(e))

    # ── Step 5: Scrape dates for new match IDs ──────────────────────────────
    _write("scraping_dates", 75, "Fetching match dates from VLR…",
           ["Fetching exact match dates…"])
    try:
        mr_df    = pd.read_csv(os.path.join(ROOT, "data", "match_results.csv"))
        all_ids  = [str(int(m)) for m in mr_df["MatchID"].unique()]
    except Exception as e:
        _write("scraping_dates", 75, "Could not read match_results.csv", error=str(e))
        all_ids = []
    out_path = os.path.join(ROOT, "data", "match_dates.json")
    existing_dates = {}
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                existing_dates = json.load(f)
        except Exception:
            existing_dates = {}
    to_fetch = [m for m in all_ids if m not in existing_dates]
    print(f"  {len(to_fetch)} new match dates to fetch", flush=True)

    if to_fetch:
        _date_lock = threading.Lock()
        _date_done = {"n": 0}
        def _date_one(_mid):
            return (_mid, _scrape_date(_mid))

        with ThreadPoolExecutor(max_workers=4) as _ex:
            _futures = [_ex.submit(_date_one, m) for m in to_fetch]
            for _fut in as_completed(_futures):
                _mid, _d = _fut.result()
                if _d:
                    existing_dates[_mid] = _d
                with _date_lock:
                    _date_done["n"] += 1
                    pct = 75 + int(_date_done["n"] / max(len(to_fetch), 1) * 12)
                    _write("scraping_dates", min(pct, 87),
                           f"Fetching dates… ({_date_done['n']}/{len(to_fetch)})")
                    # Save progress every 10 to survive crashes mid-batch
                    if _date_done["n"] % 10 == 0:
                        try:
                            with open(out_path, "w") as f:
                                json.dump(existing_dates, f)
                        except Exception:
                            pass

    try:
        with open(out_path, "w") as f:
            json.dump(existing_dates, f, indent=2)
    except Exception as e:
        _write("scraping_dates", 87, "match_dates write failed", error=str(e))

    # ── Step 6: Rebuild rating timeline ─────────────────────────────────────
    _write("building_ratings", 90, "Rebuilding BenPom ratings…",
           ["Running BenPom model…"])
    try:
        subprocess.run(
            [sys.executable, os.path.join(ROOT, "scrapers", "BuildRatingTimeline.py")],
            cwd=ROOT, check=True, capture_output=True, timeout=600,
        )
    except subprocess.CalledProcessError as e:
        _write("building_ratings", 90, "BuildRatingTimeline failed",
               error=f"BuildRatingTimeline: {e.stderr.decode('utf-8','ignore')[-400:] if e.stderr else e}")
    except Exception as e:
        _write("building_ratings", 90, "BuildRatingTimeline failed", error=str(e))

    try:
        with open(tl_path) as f:
            tl2 = json.load(f)
        new_last = tl2["checkpoints"][-1]["date"] if tl2.get("checkpoints") else today_str
    except Exception:
        new_last = today_str

    _write("done", 100,
           f"Ratings updated through {new_last}",
           [f"✓ Scraped {total_new} new match(es) across {len(by_event_maps)} event(s)",
            f"✓ Ratings rebuilt through {new_last}"])
    print(f"\nDone! {total_new} new matches, ratings through {new_last}", flush=True)


if __name__ == "__main__":
    import fcntl
    lock_fd = open(LOCK_FILE, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Already running — exiting.", flush=True)
        sys.exit(0)
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        _write("error", 100, "Refresh pipeline crashed",
               error=tb.splitlines()[-1] if tb else "unknown")
        print(tb, flush=True)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()
