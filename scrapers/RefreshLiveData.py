"""
RefreshLiveData.py — VCT live-data refresh pipeline.

Driven entirely by the live-event window declared in
MoreTestingMaybeFiles.live_events_today():

  1. Discover every event whose date window contains today (with a small lead/trail).
  2. For each region of each live event, scan VLR for completed matches and scrape
     new ones into data/maps/{event_id}.csv and data/series/{event_id}.csv.
  3. Scrape upcoming (un-played) matches for the next ~month from every live event.
  4. Rebuild match_results.csv, fetch any new match dates, and rebuild the BenPom
     rating timeline.

Future-proofing: to onboard a new VCT event, add one entry to ALL_EVENTS with
start/end dates and its region KEYS.  The VLR stats URLs may be left blank —
when the event goes live they are auto-discovered from VLR's /vct-{year} season
page (see _resolve_event_url).  Supplying an explicit URL still works and skips
discovery.  No other code change is required.

Writes live progress to /tmp/mhub_refresh_progress.json.  PID-locked so only one
instance runs at a time.

Exit code is always 0 on graceful failure; errors are surfaced via the progress
file so the UI can show what went wrong.
"""
import os, sys, json, time, re, subprocess, datetime, traceback
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
    'Dragon Ranger Gaming': 'DRG', 'Xi Lai Gaming': 'XLG',
    'EDward Gaming': 'EDG', 'Bilibili Gaming': 'BLG', 'Trace Esports': 'TE',
    'FunPlus Phoenix': 'FPX', 'JDG Esports': 'JDG', 'TYLOO': 'TYL',
    'Nova Esports': 'NOVA', 'Titan Esports Club': 'TEC', 'All Gamers': 'AG',
    'Wolves Esports': 'WOL', 'Attacking Soul Esports': 'ASE',
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
# Case-insensitive index — VLR sometimes upper-cases names on the matches page
# (e.g. "KIWOOM DRX" vs the dict's "Kiwoom DRX"), which otherwise leaves the
# org as the full name and breaks the logo lookup.
_VLR_NAME_TO_ORG_CI = {k.lower(): v for k, v in VLR_NAME_TO_ORG.items()}

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


# Per-process flag — once we've seen Cloudflare block all strategies on a URL,
# subsequent fetches skip the slow cloudscraper JS-challenge solver (which can
# burn 20-30s on Render before timing out), going straight to curl_cffi +
# requests fallback.  This is what turns a 7+ minute hung scrape into a
# ~30 second fast-fail when prod's datacenter IP is being blocked.
_cf_globally_blocked = False


def _fetch(url, *, timeout=8, retries=None, backoff=None):
    """
    GET `url` and return BeautifulSoup or None.

    Tries multiple bypass strategies in order of expected resilience, stopping
    at the first that returns real (non-Cloudflare, 2xx) HTML.  Every attempt's
    outcome is recorded in `_strategy_log` so the progress file shows which
    strategy worked (or that all failed and why).

    Per-strategy timeout dropped to 8s (was 15s).  On Render's datacenter IP
    when Cloudflare is actively blocking, a 15s × 5-strategy wall meant each
    URL hung the script for over a minute before failing.

    `retries` / `backoff` are accepted but ignored — the strategy chain itself
    is the retry mechanism.  Kept in the signature so existing callers don't
    raise TypeError.
    """
    del retries, backoff
    global _cf_globally_blocked

    strategies = []
    if _CFFI_AVAILABLE:
        strategies += ["curl_cffi:chrome131", "curl_cffi:chrome120", "curl_cffi:chrome"]
    # cloudscraper executes the CF JS challenge and can take 20+s.  Once we've
    # been globally blocked at least once, skip it entirely — it's not going to
    # save us and just slows the whole pipeline down.
    if _CS_AVAILABLE and not _cf_globally_blocked:
        strategies.append("cloudscraper")
    strategies.append("requests")

    last_err = None
    cf_blocked_this_url = True  # flip False if any strategy returns real HTML
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
        cf_blocked_this_url = False
        _record_strategy(strat, True, status=status, length=len(text))
        return BeautifulSoup(text, "html.parser")

    if cf_blocked_this_url:
        _cf_globally_blocked = True

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
    """Returns (map_rows, series_rows, match_name_str).

    Auto-retries with backoff if the first fetch parses 0 stat rows from a
    valid-looking page. We've observed VLR occasionally serving a match page
    where `.match-header` is populated (so display works) but the
    `div.vm-stats-game` blocks are either missing or empty — likely a
    transient window right after the match flips to "completed" but before
    the per-map stat tables are published. Silently returning ([], [], …)
    in that case means the match never lands in the CSV and the user can't
    tell from the progress log that anything went wrong. So: retry up to
    twice with 5s/10s sleeps, and on the final failure dump the raw HTML
    to /tmp/scrape_empty_<mid>.html for forensic inspection.
    """
    return _scrape_match_page_with_retry(url, region_tag)


def _scrape_match_page_with_retry(url, region_tag, max_attempts=3):
    attempt_results = []
    last_soup_text = None
    for attempt in range(1, max_attempts + 1):
        soup = _fetch(url)
        if soup is None:
            attempt_results.append(f"attempt {attempt}: fetch failed")
            if attempt < max_attempts:
                time.sleep(5 * attempt)
                continue
            return [], [], "? (fetch failed)"

        map_rows, series_rows, display = _parse_match_html(soup, url, region_tag)
        last_soup_text = str(soup)
        if map_rows or series_rows:
            return map_rows, series_rows, display
        attempt_results.append(
            f"attempt {attempt}: 0 rows from valid HTML "
            f"(size={len(last_soup_text)}, display={display!r})"
        )
        if attempt < max_attempts:
            time.sleep(5 * attempt)

    # All attempts produced 0 rows. Dump HTML + log so we can see what VLR
    # actually served. The pipeline already retries this URL next refresh
    # (CSV-missing match → flagged "new" again).
    try:
        mid = _match_id_from_url(url) or "unknown"
        dump_path = f"/tmp/scrape_empty_{mid}.html"
        with open(dump_path, "w") as fh:
            fh.write(last_soup_text or "")
        _error_entries.append(
            f"0-row scrape after {max_attempts} attempts on {url}; "
            f"HTML dumped to {dump_path}. Attempts: {attempt_results}"
        )
        print(f"  EMPTY SCRAPE: {url} — HTML dumped to {dump_path}", flush=True)
    except Exception as _e:
        _error_entries.append(f"0-row scrape on {url}, also failed to dump HTML: {_e}")
    return [], [], display


def _parse_match_html(soup, url, region_tag):
    """Pure parsing — no fetch, no retry. Split out so retry can call it.

    VLR migrated the per-map stats from a `table.wf-table-inset.mod-overview`
    layout to a `div.ovw-*` grid (~mid-2026): each `div.vm-stats-game` now holds
    two `.ovw-table` blocks (team A then team B) of `.ovw-row`s; every stat sits
    in a `.ovw-cell`, and K/D/A collapsed into one `.ovw-cell.mod-kda` cell whose
    kills/deaths/assists live in `.ovw-kda-stat[data-col=…]`. Combined
    (attack+defense) values are still the `span.mod-both` inside each cell — but
    the class is now `side mod-both` (matched by token, so it still resolves).
    """
    # Series format: the score block adds a "final" note, so scan every
    # `.match-header-vs-note` for the Bo token rather than trusting the first.
    notes = " ".join(n.get_text(" ", strip=True).lower()
                     for n in soup.select(".match-header-vs-note"))
    series_fmt = "bo5" if ("bo5" in notes or "best of 5" in notes) else (
                  "bo1" if ("bo1" in notes or "best of 1" in notes) else "bo3")

    mid = _match_id_from_url(url) or ""

    teams_el = soup.select(".match-header-link-name .wf-title-med")
    team_a = teams_el[0].get_text(strip=True) if len(teams_el) > 0 else "?"
    team_b = teams_el[1].get_text(strip=True) if len(teams_el) > 1 else "?"

    def _both(el):
        """Combined (atk+def) value: prefer a `mod-both` span, else raw text."""
        if el is None:
            return ""
        sp = el.find("span", class_=lambda c: c and "mod-both" in c.split())
        return sp.get_text(strip=True) if sp else el.get_text(" ", strip=True)

    def _kda(kda_cell, col):
        if kda_cell is None:
            return ""
        return _both(kda_cell.select_one(f'.ovw-kda-stat[data-col="{col}"]'))

    map_rows, series_rows = [], []
    a_wins = b_wins = 0

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
            # Tally an oriented series score from each map's winner (the
            # `.score.mod-win` span) for the progress-log display string.
            gh = game_div.select_one(".vm-stats-game-header")
            scs = gh.select(".score") if gh else []
            if len(scs) >= 2:
                a_wins += 1 if "mod-win" in (scs[0].get("class") or []) else 0
                b_wins += 1 if "mod-win" in (scs[1].get("class") or []) else 0

        for prow in game_div.select(".ovw-row"):
            pcell = prow.select_one(".ovw-cell.mod-player") or prow.select_one(".mod-player")
            if pcell is None:
                continue  # header row / non-player row
            pname = pcell.select_one(".text-of")
            porg  = pcell.select_one(".ge-text-light")
            pa    = pcell.find("a", href=True)
            player = pname.get_text(strip=True) if pname else ""
            org    = porg.get_text(strip=True)  if porg  else ""
            if not player:
                continue

            cells = prow.select(".ovw-cell")
            # Anchor every column on the KDA cell so leading-column changes
            # (icons, +/- toggles) can't shift the stat mapping.
            kda_i = next((i for i, c in enumerate(cells)
                          if "mod-kda" in (c.get("class") or [])), 3)
            kda_cell = cells[kda_i] if kda_i < len(cells) else None

            def _at(idx):
                return _both(cells[idx]) if 0 <= idx < len(cells) else ""

            row = {
                "Player":       player,
                "Org":          org,
                "ProfileURL":   ("https://www.vlr.gg" + pa["href"]) if pa else "",
                "Region":       region_tag,
                "MatchID":      mid,
                "MapNum":       game_id,
                "MapName":      map_name,
                "SeriesFormat": series_fmt,
                "R2.0":   _at(kda_i - 2),   # rating
                "ACS":    _at(kda_i - 1),
                "K":      _kda(kda_cell, "kills"),
                "D":      _kda(kda_cell, "deaths"),
                "A":      _kda(kda_cell, "assists"),
                "KAST":   _at(kda_i + 2),   # skip the +/- cell at kda_i+1
                "ADR":    _at(kda_i + 3),
                "HS%":    _at(kda_i + 4),
                "FK":     _at(kda_i + 5),
                "FD":     _at(kda_i + 6),
            }
            try:
                k_i, d_i = int(row["K"]), int(row["D"])
                row["K:D"] = round(k_i / d_i, 2) if d_i else float(k_i)
            except Exception:
                row["K:D"] = ""

            (series_rows if is_all else map_rows).append(row)

    sa = str(a_wins) if (a_wins or b_wins) else "?"
    sb = str(b_wins) if (a_wins or b_wins) else "?"
    display = f"{team_a} {sa}–{sb} {team_b}"

    return map_rows, series_rows, display


def _et_walltime_to_utc(ts):
    """VLR's `data-utc-ts` attribute is mislabeled: its value is actually the
    match's US/Eastern WALL-CLOCK time (verified — its HH:MM always equals the
    page's own displayed "H:MM PM EDT/EST" text digit-for-digit; a true-UTC
    value 4-5h earlier would NOT match). Treating it as literal UTC (as an
    earlier version of this code did) double-shifts every displayed time by
    the ET offset. This converts the ET wall-clock string to the TRUE UTC
    instant (DST-aware via zoneinfo, correct for any date) so downstream
    'Z'-suffixed ISO consumers (the browser's local-time formatter) get a
    genuinely correct instant. Returns "YYYY-MM-DD HH:MM:SS" UTC, or the input
    unchanged if parsing fails."""
    try:
        from zoneinfo import ZoneInfo
        naive = datetime.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        et = naive.replace(tzinfo=ZoneInfo("America/New_York"))
        return et.astimezone(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ts


def _scrape_date(mid):
    """Return the match's raw VLR timestamp, "YYYY-MM-DD HH:MM:SS" — this is
    US/Eastern wall-clock time despite VLR's "data-utc-ts" name (see
    _et_walltime_to_utc). Callers slice [:10] for the date-only
    match_dates.json (ET calendar day — matches VLR's own day-bucketing, so
    left as-is) and must pass the FULL string through _et_walltime_to_utc
    before storing/displaying it as a time. None if not found."""
    soup = _fetch(f"https://www.vlr.gg/{mid}/", retries=2)
    if soup is None:
        return None
    el = soup.find("div", class_="moment-tz-convert", attrs={"data-utc-ts": True})
    if el:
        return el["data-utc-ts"]
    return None


# ── Upcoming match scraper ────────────────────────────────────────────────────

def _scrape_upcoming_for(vlr_id, slug, region, event_label):
    """Return upcoming-match dicts for one (event, region) within the next ~month."""
    from datetime import datetime as _dt
    from bs4 import NavigableString
    today = datetime.date.today()
    cutoff = today + datetime.timedelta(days=31)

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
            utc_ts = ts_el["data-utc-ts"] if (ts_el and ts_el.get("data-utc-ts")) else ""
            match_date = utc_ts[:10] if utc_ts else current_date
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

            # Round / stage label (e.g. "Playoffs: Grand Final"). Captured so
            # downstream code can detect grand finals and apply the bo5_gf
            # veto (upper-bracket team gets both bans + first pick).
            stage_el  = a.select_one(".match-item-event")
            stage_raw = stage_el.get_text(" ", strip=True) if stage_el else ""

            out.append({
                "match_id": _match_id_from_url(a.get("href", "")) or "",
                "team_a": team_a, "team_b": team_b,
                "org_a":  _VLR_NAME_TO_ORG_CI.get(team_a.lower(), team_a),
                "org_b":  _VLR_NAME_TO_ORG_CI.get(team_b.lower(), team_b),
                "date":   match_date,
                "datetime": (_et_walltime_to_utc(utc_ts) if utc_ts else ""),   # true UTC, for local-time display
                "region": region,
                "event":  f"{event_label} — {region}" if region != "International" else event_label,
                "format": fmt,
                "match_name": stage_raw,
            })
    return out


# ── VLR event-URL auto-discovery ────────────────────────────────────────────────
# Future-proofing: an ALL_EVENTS entry only needs id/label/year/start/end and its
# region KEYS — the region URLs may be left blank.  When such an event goes live,
# we resolve the missing VLR stats URLs from VLR's season page (/vct-{year}) by
# matching the slug on region + stage (domestic) or label tokens (international).
# Events that already carry an explicit URL are untouched (no extra fetches).

_season_cache = {}  # year -> [(vlr_id, slug, name_text), ...]

# Region key → token that appears in VLR domestic-event slugs (vct-{yr}-{tok}-…).
_REGION_SLUG_TOKEN = {
    "EMEA": "emea", "Americas": "americas", "Pacific": "pacific", "CN": "china",
}


def _vlr_season_events(year):
    """Return [(vlr_id, slug, name_text), …] from VLR's /vct-{year} page (cached)."""
    if year in _season_cache:
        return _season_cache[year]
    out = []
    soup = _fetch(f"https://www.vlr.gg/vct-{year}")
    if soup is not None:
        seen = set()
        for a in soup.find_all("a", href=True):
            m = re.match(r"^/event/(\d+)/([^/?#]+)", a["href"])
            if not m:
                continue
            key = (m.group(1), m.group(2))
            if key in seen:
                continue
            seen.add(key)
            out.append((m.group(1), m.group(2),
                        re.sub(r"\s+", " ", a.get_text(" ", strip=True))))
    _season_cache[year] = out
    return out


def _resolve_event_url(ev, region):
    """Auto-discover a VLR stats URL for (event, region) from the season page.
    Returns a stats URL string or None.  Lets future events onboard with no
    manual URL entry."""
    try:
        year = int(ev.get("year") or str(ev.get("start", ""))[:4])
    except (TypeError, ValueError):
        return None
    cands = _vlr_season_events(year)
    if not cands:
        return None

    eid   = (ev.get("id") or "").lower()
    label = (ev.get("label") or "").lower()

    if "kickoff" in eid:   stage_tokens = ("kickoff",)
    elif "stage2" in eid:  stage_tokens = ("stage-2", "stage2")
    elif "stage1" in eid:  stage_tokens = ("stage-1", "stage1")
    else:                  stage_tokens = ()

    def slug_ok(slug):
        s = slug.lower()
        if region in _REGION_SLUG_TOKEN:          # domestic / regional
            if _REGION_SLUG_TOKEN[region] not in s:
                return False
            return any(t in s for t in stage_tokens)
        # international — match the distinguishing label tokens against the slug
        if "champions" in label:
            return "champions" in s
        if "masters" in label:
            extras = [w for w in re.split(r"[^a-z0-9]+", label)
                      if w and w not in ("masters", str(year))]
            return "masters" in s and any(t in s for t in extras)
        toks = [w for w in re.split(r"[^a-z0-9]+", label) if len(w) > 2]
        return bool(toks) and all(t in s for t in toks)

    for vlr_id, slug, _txt in cands:
        if slug_ok(slug):
            return f"https://www.vlr.gg/event/stats/{vlr_id}/{slug}"
    return None


# ── Main pipeline ──────────────────────────────────────────────────────────────

def _event_to_target(ev):
    regions = []
    for region, url in ev["regions"].items():
        vlr_id, slug = _parse_vlr_stats_url(url)
        if not (vlr_id and slug):
            # Blank placeholder — auto-discover from VLR so future events
            # onboard without anyone hand-entering URLs.
            resolved = _resolve_event_url(ev, region)
            if resolved:
                vlr_id, slug = _parse_vlr_stats_url(resolved)
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


def _upcoming_lookahead_targets(live_targets):
    """Events that START within ~a month but aren't 'live' yet (i.e. beyond the
    live lead window), so their already-posted VLR schedule still surfaces as
    'upcoming' up to a month out — e.g. Pacific Stage 2 matches posted weeks
    before the split begins. URLs auto-resolve from VLR's season page when the
    ALL_EVENTS placeholder is blank. Deduped against the live targets."""
    import datetime as _dt
    today = _dt.date.today()
    horizon = today + _dt.timedelta(days=31)
    have = {t.get("label") for t in live_targets}
    out = []
    for ev in ALL_EVENTS:
        s = ev.get("start")
        if not s or ev.get("label") in have:
            continue
        try:
            sd = _dt.date.fromisoformat(s)
        except Exception:
            continue
        if today < sd <= horizon:
            t = _event_to_target(ev)
            if t:
                out.append(t)
    return out


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
               ["No event in ALL_EVENTS matched today's date window (URLs auto-resolve from VLR's season page when blank).",
                "To onboard a new event, add an entry with start/end + region keys to MoreTestingMaybeFiles.ALL_EVENTS."])
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
    all_new_urls = []   # list of (url, region, event_csv_id)
    total_completed = 0
    scan_steps = sum(len(t["regions"]) for t in targets) or 1
    step_idx = 0

    for t in targets:
        existing = _existing_match_ids(t["event_csv_id"])
        _write("checking", 8,
               f"Scanning {t['label']} — {len(existing)} match(es) on disk…",
               [f"[{t['label']}] {len(existing)} match(es) already cached"])

        for region, vlr_id, slug in t["regions"]:
            step_idx += 1
            pct = 8 + int(step_idx / scan_steps * 20)
            _write("checking", pct, f"Checking {t['label']} / {region}…")
            urls = _get_completed_urls(vlr_id, slug)
            total_completed += len(urls)
            new = [u for u in urls if _match_id_from_url(u) not in existing]
            for u in new:
                all_new_urls.append((u, region, t["event_csv_id"]))
            _write("checking", pct,
                   f"{t['label']} / {region}: {len(urls)} completed, {len(new)} new",
                   [f"✓ {t['label']} / {region}: {len(urls)} completed ({len(new)} new)"])
            time.sleep(0.2)  # was 0.6 — kept sequential (Cloudflare-safe) but shorter

    _write("checking", 30,
           f"Scan complete — {total_completed} completed across {len(targets)} live event(s), "
           f"{len(all_new_urls)} new to scrape",
           [f"Total completed across all live events: {total_completed}",
            f"New to scrape: {len(all_new_urls)}"])

    # ── Step 2: Scrape upcoming for live events + any starting within a month ─
    # Look-ahead lets a not-yet-live split (e.g. Pacific Stage 2, 3 weeks out)
    # surface its already-posted VLR schedule as "upcoming".
    upc_targets = list(targets) + _upcoming_lookahead_targets(targets)
    all_upcoming = []
    for t in upc_targets:
        for region, vlr_id, slug in t["regions"]:
            upc = _scrape_upcoming_for(vlr_id, slug, region, t["label"])
            all_upcoming.extend(upc)
            time.sleep(0.15)  # was 0.4

    seen = set()
    deduped = []
    for m in sorted(all_upcoming, key=lambda x: x["date"]):
        key = f"{m['team_a']}-{m['team_b']}-{m['date']}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(m)

    # Exact UTC start times for the SOONEST upcoming matches. The matches-list
    # page only shows a bare "5:00 PM" (no timezone), so pull the authoritative
    # data-utc-ts from each match's own page — bounded to the next ~10 days and
    # cached in match_times.json so each match is fetched at most once.
    _times_path = os.path.join(ROOT, "data", "match_times.json")
    try:
        with open(_times_path) as f:
            _upc_times = json.load(f)
    except Exception:
        _upc_times = {}
    _soon_cut = (datetime.date.today() + datetime.timedelta(days=10)).isoformat()
    _t_fetched = 0
    for _m in deduped:
        if _m.get("datetime") or _m.get("date", "") > _soon_cut:
            continue
        _mid = str(_m.get("match_id") or "")
        if not _mid:
            continue
        if _mid in _upc_times:
            _m["datetime"] = _upc_times[_mid]
        elif _t_fetched < 40:
            _ts = _scrape_date(_mid)   # raw ET wall-clock
            _t_fetched += 1
            if _ts:
                _utc = _et_walltime_to_utc(_ts)
                _m["datetime"] = _utc
                _upc_times[_mid] = _utc
            time.sleep(0.15)
    if _t_fetched:
        try:
            with open(_times_path, "w") as f:
                json.dump(_upc_times, f, indent=2)
        except Exception:
            pass

    out_upc = os.path.join(ROOT, "data", "upcoming_matches.json")
    try:
        with open(out_upc, "w") as f:
            json.dump(deduped, f, indent=2)
        _write("checking", 34,
               f"Upcoming matches: {len(deduped)} in next ~month",
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

    # ── Step 3: Scrape new matches into the right per-event CSVs ─────────────
    total_new = len(all_new_urls)
    by_event_maps   = {}   # event_csv_id → [row, ...]
    by_event_series = {}
    # VLR sometimes marks a match "completed" before the per-map stats tables
    # have populated. _scrape_match_page returns ([], [], display) in that
    # case and the match silently never lands in the CSV — so the next refresh
    # picks it up as "new" again. Track empties explicitly so they show up in
    # the progress log instead of being conflated with successful scrapes.
    empty_scrapes = []

    for i, (url, region, ev_id) in enumerate(all_new_urls, 1):
        pct = 36 + int(i / total_new * 32)
        _write("scraping", pct, f"Scraping match {i}/{total_new}…")
        mr, sr, display = _scrape_match_page(url, region)
        by_event_maps.setdefault(ev_id, []).extend(mr)
        by_event_series.setdefault(ev_id, []).extend(sr)
        empty = not mr and not sr
        if empty:
            empty_scrapes.append((url, region, display))
        suffix = "  ⚠ no stats yet — will retry next refresh" if empty else ""
        _write("scraping", pct, f"Scraping {i}/{total_new}…",
               [f"  [{region}] {display}{suffix}"])
        time.sleep(0.25)  # was 0.7

    if empty_scrapes:
        lines = [f"⚠ {len(empty_scrapes)} match(es) returned no stats (VLR hadn't published "
                 f"per-map data yet) — these will retry on the next refresh:"]
        for url, region, disp in empty_scrapes:
            lines.append(f"  [{region}] {disp}  ({url})")
        _write("scraping", 68,
               f"{len(all_new_urls) - len(empty_scrapes)}/{len(all_new_urls)} scraped, "
               f"{len(empty_scrapes)} pending stats",
               lines)

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
    out_path   = os.path.join(ROOT, "data", "match_dates.json")
    times_path = os.path.join(ROOT, "data", "match_times.json")

    def _load_json(p):
        if os.path.exists(p):
            try:
                with open(p) as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    existing_dates = _load_json(out_path)    # MatchID -> "YYYY-MM-DD" (unchanged format)
    existing_times = _load_json(times_path)  # MatchID -> "YYYY-MM-DD HH:MM:SS" (UTC)

    # New matches get date + time in one fetch (VLR serves both in data-utc-ts).
    to_fetch = [m for m in all_ids if m not in existing_dates]
    # Backfill exact UTC times for recent matches that predate this feature
    # (times weren't stored before). Bounded to the last ~45 days so the
    # page-load scrape stays fast; older matches keep date-only.
    _recent_cut = (datetime.date.today() - datetime.timedelta(days=45)).isoformat()
    to_fetch_times = [m for m in all_ids
                      if m not in to_fetch and m not in existing_times
                      and str(existing_dates.get(m, "")) >= _recent_cut]
    fetch_all = to_fetch + to_fetch_times
    print(f"  {len(to_fetch)} new match dates, {len(to_fetch_times)} recent times to fetch",
          flush=True)

    def _dump_dates_times():
        try:
            with open(out_path, "w") as f:
                json.dump(existing_dates, f, indent=2)
            with open(times_path, "w") as f:
                json.dump(existing_times, f, indent=2)
        except Exception as e:
            _write("scraping_dates", 87, "match_dates/times write failed", error=str(e))

    for i, mid in enumerate(fetch_all, 1):
        ts = _scrape_date(mid)   # raw ET wall-clock "YYYY-MM-DD HH:MM:SS" or None
        if ts:
            existing_dates[mid] = ts[:10]                       # ET calendar day (unchanged semantics)
            existing_times[mid] = _et_walltime_to_utc(ts)        # true UTC instant, for display
        pct = 75 + int(i / max(len(fetch_all), 1) * 12)
        _write("scraping_dates", min(pct, 87),
               f"Fetching dates/times… ({i}/{len(fetch_all)})")
        if i % 10 == 0:
            _dump_dates_times()
        time.sleep(0.15)  # was 0.45

    _dump_dates_times()

    # ── Step 5b: Catch up map veto sequences ────────────────────────────────
    # Veto scraper is incremental (skips already-scraped MatchIDs). The veto
    # data feeds the live map pool detector — we rely on the actual pick/ban
    # text from VLR rather than play-count heuristics.
    _write("scraping_vetos", 80, "Fetching map veto sequences…",
           ["Catching up new match vetos…"])
    try:
        subprocess.run(
            [sys.executable, os.path.join(ROOT, "scrapers", "ScrapeMapVetos.py")],
            cwd=ROOT, check=True, capture_output=True, timeout=900,
        )
    except subprocess.CalledProcessError as e:
        _write("scraping_vetos", 80, "ScrapeMapVetos failed",
               error=f"ScrapeMapVetos: {e.stderr.decode('utf-8','ignore')[-400:] if e.stderr else e}")
    except Exception as e:
        _write("scraping_vetos", 80, "ScrapeMapVetos failed", error=str(e))

    # ── Steps 5c / 6 / 7: Build veto model + rating timeline + map ratings ──
    # All three previously ran via subprocess.run, each paying a fresh
    # Python+pandas+numpy startup (~2-3 s on Render). We now import them and
    # call main() in-process — saves the cold-start cost on every refresh.
    _scripts_dir = os.path.join(ROOT, "scrapers")
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)

    def _run_inproc(label, module_name, pct, slug, argv_extra=()):
        _write(slug, pct, label, [label + "…"])
        saved_argv = sys.argv[:]
        sys.argv = [module_name + ".py", *argv_extra]
        try:
            mod = __import__(module_name)
            # If module already imported in a prior run, reload to pick up data changes
            import importlib
            importlib.reload(mod)
            mod.main()
        except SystemExit:
            pass
        except Exception as e:
            _write(slug, pct, f"{module_name} failed",
                   error=f"{module_name}: {str(e)[-400:]}")
        finally:
            sys.argv = saved_argv

    # Uses the now-fresh map_vetos.csv to compute ban/pick rates per team for
    # every active snapshot — including the current one — so the simulator's
    # "Predicted Veto" section has data for current rosters in current pools.
    _run_inproc("Rebuilding veto patterns", "BuildVetoModel", 83, "building_veto_model")

    # BenPom timeline is incremental — re-solves only checkpoints for match
    # days newer than the last one already in rating_timeline.json.
    _run_inproc("Rebuilding BenPom ratings", "BuildRatingTimeline", 85, "building_ratings")

    # --refresh: only rebuilds the current-year snapshots and reuses
    # historical ratings from the existing JSON. Historical data is immutable.
    _run_inproc("Rebuilding per-map ratings", "BuildMapRatings", 92, "building_map_ratings",
                argv_extra=("--refresh",))

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
