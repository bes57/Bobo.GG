import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import json
import os
from flask import Blueprint, render_template_string, request
from MoreTestingMaybeFiles import ALL_EVENTS

vct_bp = Blueprint('vct', __name__)

@vct_bp.app_template_filter('player_hue')
def player_hue(name):
    return sum(ord(c) for c in (name or '')) % 360

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

STAT_LABELS = {
    "R2.0": "VLR Rating",
    "K:D":    "Kill/Death Ratio",
    "KAST":   "KAST %",
    "ADR":    "Avg Damage / Round",
    "HS%":    "Headshot %",
    "CL%":    "Clutch %",
    "FKPR":   "First Kills Per Round",
    "FIPR":   "First Interactions / Round",
    "FIWR":   "First Interaction Win %",
    "KPR":    "Kills Per Round",
    "DPR":    "Deaths Per Round",
    "APR":    "Assists Per Round",
}

LIVE_EVENT_ID = "2026_stage2"   # Masters London completed 2026-06-21; now reads from data/2026_masters_london.csv like other past events
ALLTIME_ID = "all_time"
ALLTIME_INTL_ID = "all_time_intl"     # All-Time aggregate, international events only
ALLTIME_DOM_ID = "all_time_dom"       # All-Time aggregate, domestic/regional events only
ALLTIME_EVENT = {"id": ALLTIME_ID, "label": "All-Time", "year": 0, "regions": {"International": ""}}
ALLTIME_INTL_EVENT = {"id": ALLTIME_INTL_ID, "label": "All-Time (Internationals Only)", "year": 0, "regions": {"International": ""}}
ALLTIME_DOM_EVENT = {"id": ALLTIME_DOM_ID, "label": "All-Time (Domestic Only)", "year": 0, "regions": {"International": ""}}

# The set of synthetic All-Time aggregate IDs and the event dicts they resolve to.
ALLTIME_IDS = {ALLTIME_ID, ALLTIME_INTL_ID, ALLTIME_DOM_ID}
ALLTIME_EVENTS_BY_ID = {
    ALLTIME_ID:      ALLTIME_EVENT,
    ALLTIME_INTL_ID: ALLTIME_INTL_EVENT,
    ALLTIME_DOM_ID:  ALLTIME_DOM_EVENT,
}


def _is_international(event):
    """Source-of-truth rule: an event is international iff its regions dict
    contains the 'International' key; otherwise it is domestic/regional."""
    return "International" in event.get("regions", {})


def _alltime_event_filter(alltime_id):
    """Return a predicate selecting which ALL_EVENTS a given All-Time aggregate
    should include. CN-only events are always excluded (they feed BenPom, not
    the player leaderboards). The intl/dom variants further restrict to
    international or domestic events respectively."""
    def keep(e):
        if list(e["regions"].keys()) == ["CN"]:
            return False
        if alltime_id == ALLTIME_INTL_ID:
            return _is_international(e)
        if alltime_id == ALLTIME_DOM_ID:
            return not _is_international(e)
        return True   # plain all_time: every non-CN-only event
    return keep

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

_event_cache = {}       # event_id -> DataFrame
_headshot_cache = {}    # profile_url -> headshot_url or ""
_headshots_loaded = False

_HEADSHOTS_FILE = os.path.join(os.path.dirname(__file__), "data", "headshots.json")

def _ensure_headshots_loaded():
    global _headshots_loaded
    if not _headshots_loaded:
        if os.path.exists(_HEADSHOTS_FILE):
            with open(_HEADSHOTS_FILE) as f:
                _headshot_cache.update(json.load(f))
            print(f"Loaded {len(_headshot_cache)} headshots from {_HEADSHOTS_FILE}")
        _headshots_loaded = True

def get_events_by_year():
    by_year = {}
    for e in ALL_EVENTS:
        # CN-only events feed BenPom (team ratings) but are hidden from the
        # event-leaderboard dropdown — user wants CN scoped to team stats, not players.
        if list(e["regions"].keys()) == ["CN"]:
            continue
        # Hide events with no data yet (haven't started, or pre-scrape).
        csv_path = os.path.join(os.path.dirname(__file__), "data", f"{e['id']}.csv")
        if not os.path.exists(csv_path):
            continue
        by_year.setdefault(e["year"], []).append(e)
    # Sort within each year by start date, most-recent first, so events like
    # Masters Shanghai (Jun 2024) sit between Stage 2 and Stage 1 instead of
    # falling wherever ALL_EVENTS happens to list them.
    for year in by_year:
        by_year[year].sort(key=lambda e: e.get("start", ""), reverse=True)
    return sorted(by_year.items(), reverse=True)

def scrape_stats(region, url):
    print(f"Scraping {region} — {url}...")
    try:
        res = requests.get(url, headers=HEADERS, timeout=15)
    except Exception as e:
        print(f"  Request failed: {e}")
        return pd.DataFrame()
    soup = BeautifulSoup(res.text, "html.parser")
    table = soup.find("table")
    if not table:
        print(f"  No table found.")
        return pd.DataFrame()
    raw_headers = [th.get_text(strip=True) for th in table.find_all("th")]
    col_names = ["Player", "Org", "ProfileURL"] + raw_headers[1:]
    rows = []
    for tr in table.find("tbody").find_all("tr"):
        tds = tr.find_all("td")
        if not tds:
            continue
        row = []
        for i, td in enumerate(tds):
            if i == 0:
                lines = [l.strip() for l in td.get_text(separator="\n", strip=True).split("\n") if l.strip()]
                player_name = lines[0] if lines else ""
                org = lines[1] if len(lines) > 1 else ""
                a = td.find("a", href=True)
                profile_url = ("https://www.vlr.gg" + a["href"]) if a else ""
                row.extend([player_name, org, profile_url])
            else:
                row.append(td.get_text(strip=True))
        if row:
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=col_names[:len(rows[0])])
    df.insert(0, "Region", region)
    return df

def _scrape_event_live(event):
    """Scrape all regions for an event and return a cleaned DataFrame."""
    dfs = []
    for region_name, url in event["regions"].items():
        # Skip region slots with no URL yet (e.g. an upcoming event that's the
        # live target but hasn't been posted on VLR). Scraping "" just fails and
        # the per-region time.sleep(1) would otherwise burn ~1s each — costly
        # when the All-Time view concatenates every event, including this one.
        if not url:
            continue
        df = scrape_stats(region_name, url)
        if not df.empty:
            dfs.append(df)
        time.sleep(1)
    if not dfs:
        return pd.DataFrame()
    cache = pd.concat(dfs, ignore_index=True)
    cache["HeadshotURL"] = cache["ProfileURL"].map(lambda u: _headshot_cache.get(u, ""))
    if "R2.0" in cache.columns:
        r2 = pd.to_numeric(cache["R2.0"].astype(str).str.replace("%", ""), errors="coerce")
        cache = cache[r2.notna() & (r2 > 0)].reset_index(drop=True)
    if list(event["regions"].keys()) == ["International"] and "Org" in cache.columns:
        cache["Region"] = cache["Org"].map(lambda org: ORG_REGIONS.get(org, "International"))
    # Strip showmatch / non-franchised players
    if "Org" in cache.columns:
        cache = cache[cache["Org"].isin(ORG_REGIONS)].reset_index(drop=True)
    return cache


def _add_derived_stats(df):
    """Add FIPR (first interactions / round) and FIWR (first-blood win %)
    columns derived from FK / FD / Rnd. Safe no-op if columns missing.
    All first-interaction stats render to 2 decimal places for consistency."""
    if df.empty or not {"FK", "FD", "Rnd"}.issubset(df.columns):
        return df
    fk  = pd.to_numeric(df["FK"],  errors="coerce")
    fd  = pd.to_numeric(df["FD"],  errors="coerce")
    rnd = pd.to_numeric(df["Rnd"], errors="coerce")
    fi  = fk + fd
    df["FIPR"] = (fi / rnd).apply(lambda v: f"{v:.2f}" if pd.notna(v) else "")
    fiwr = (fk / fi * 100)
    df["FIWR"] = fiwr.apply(lambda v: f"{v:.2f}%" if pd.notna(v) else "")
    # Per-round KDA row: KPR/APR ship raw in the event CSVs; DPR we derive from
    # D / Rnd. All three render to 2 decimals like the other per-round stats.
    if "D" in df.columns:
        df["DPR"] = pd.to_numeric(df["D"], errors="coerce") / rnd
    # Reformat numeric stats to always 2 decimals (CSV drops trailing zero).
    for col in ("FKPR", "KPR", "DPR", "APR", "R2.0", "K:D"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").apply(
                lambda v: f"{v:.2f}" if pd.notna(v) else ""
            )
    # ADR uses 1 decimal place.
    if "ADR" in df.columns:
        df["ADR"] = pd.to_numeric(df["ADR"], errors="coerce").apply(
            lambda v: f"{v:.1f}" if pd.notna(v) else ""
        )
    return df


def load_event(event):
    event_id = event["id"]
    if event_id in _event_cache:
        return _event_cache[event_id]

    # All-Time (and its Internationals-Only / Domestic-Only variants):
    # concatenate the relevant subset of events' leaderboards, tagging each row
    # with the source event label so duplicates (same player, different event)
    # stay distinguishable in the giant ranking.
    if event_id in ALLTIME_IDS:
        print(f"{event['label']} — concatenating matching events...")
        keep = _alltime_event_filter(event_id)
        parts = []
        for e in ALL_EVENTS:
            if not keep(e):
                continue
            sub = load_event(e).copy()
            if not sub.empty:
                sub["Event"] = e["label"]
                parts.append(sub)
        cache = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        _event_cache[event_id] = cache
        return cache

    # Live event: always scrape fresh
    if event_id == LIVE_EVENT_ID:
        print(f"Live event — scraping {event_id}...")
        cache = _scrape_event_live(event)
    else:
        # Past event: load from pre-scraped CSV if available
        csv_path = os.path.join(DATA_DIR, f"{event_id}.csv")
        if os.path.exists(csv_path):
            print(f"Loading {event_id} from CSV...")
            cache = pd.read_csv(csv_path)
            cache["HeadshotURL"] = cache.get("ProfileURL", pd.Series()).map(
                lambda u: _headshot_cache.get(u, "")
            )
        else:
            print(f"No CSV for {event_id} — scraping live...")
            cache = _scrape_event_live(event)

    cache = _add_derived_stats(cache)
    _event_cache[event_id] = cache
    return cache

def get_all(df, col):
    if df.empty or col not in df.columns:
        return []
    keep = [c for c in ["Player", "Org", "ProfileURL", "HeadshotURL", "Region", "Rnd", "Event", "FK", "FD", col] if c in df.columns]
    tmp = df[keep].copy()
    # Sort numerically (strip % for ordering) but preserve the display string.
    sort_vals = pd.to_numeric(tmp[col].astype(str).str.replace("%", ""), errors="coerce")
    tmp = tmp.loc[sort_vals.notna()].assign(_sort=sort_vals[sort_vals.notna()])
    # kind="stable" is REQUIRED for consistency with the dashboard cards. The
    # dashboard sorts client-side with JS Array.sort (stable since ES2019), so
    # tied values keep their source-data order. pandas' default sort is
    # quicksort (unstable) — it would break ties in a different order, so the
    # same two tied players (e.g. FIPR 0.35) could swap places between the
    # dashboard top-5 and this full-ranking page. Stable sort makes both pages
    # fall back to the same source order, so ties are ordered identically.
    return tmp.sort_values("_sort", ascending=False, kind="stable").drop(columns="_sort").to_dict("records")

def build_data(cache, event):
    is_multi = len(event["regions"]) > 1
    is_international = not is_multi and list(event["regions"].keys()) == ["International"]
    stat_cols = list(STAT_LABELS.keys())

    def to_records(df):
        if df.empty:
            return []
        want = ["Player", "Org", "ProfileURL", "HeadshotURL", "Region", "Rnd", "Event", "FK", "FD"] + stat_cols
        cols = [c for c in want if c in df.columns]
        return df[cols].fillna("").to_dict("records")

    # Ship every row exactly once under "All"; the client derives each region's
    # subset by filtering on the row's own Region field. Building per-region
    # arrays here used to duplicate the whole dataset N times — on the All-Time
    # view that doubled a 1.6 MB payload, the dominant cost of that page's load.
    data = {"All": to_records(cache)}

    present = set(cache["Region"].unique()) if (not cache.empty and "Region" in cache.columns) else set()
    if is_multi:
        available_regions = ["All"] + [r for r in event["regions"] if r in present]
    elif is_international:
        available_regions = ["All"] + [r for r in ["EMEA", "Americas", "Pacific", "CN"] if r in present]
    else:
        available_regions = ["All"]
    return data, available_regions


# Maps the org tag shown on VLR.gg's stats table to the team's home region.
# Used to assign real regions (EMEA / Americas / Pacific) to players at
# international events instead of the uninformative "International" label.
# CN franchised teams are grouped under "Pacific" to match VCT's bracket structure.
ORG_REGIONS = {
    # EMEA
    "TL":   "EMEA",  "FNC":  "EMEA",  "NAVI": "EMEA",  "VIT":  "EMEA",
    "BBL":  "EMEA",  "GX":   "EMEA",  "KC":   "EMEA",  "TH":   "EMEA",
    "FUT":  "EMEA",  "GIA":  "EMEA",  "MKOI": "EMEA",
    "M8":   "EMEA",
    # Americas
    "SEN":  "Americas",  "G2":   "Americas",  "MIBR": "Americas",
    "NRG":  "Americas",  "100T": "Americas",  "C9":   "Americas",
    "EG":   "Americas",  "KRÜ":  "Americas",  "LEV":  "Americas",
    "FUR":  "Americas",  "LOUD": "Americas",
    # Pacific
    "PRX":  "Pacific",  "DRX":  "Pacific",  "T1":   "Pacific",
    "TLN":  "Pacific",  "GEN":  "Pacific",  "DFM":  "Pacific",
    "ZETA": "Pacific",  "RRQ":  "Pacific",  "TS":   "Pacific",
    "GE":   "Pacific",  "KRX":  "Pacific",  "NS":   "Pacific",
    "FS":   "Pacific",
    # CN
    "EDG":  "CN",  "BLG":  "CN",  "TE":   "CN",  "DRG":  "CN",
    "ASE":  "CN",  "AG":   "CN",  "XLG":  "CN",  "WOL":  "CN",
    "FPX":  "CN",  "JDG":  "CN",  "NOVA": "CN",  "TEC":  "CN",
    "TYL":  "CN",  "TYLOO":"CN",
}

# ── Player best-match lookup ──────────────────────────────────────────────────

_player_match_cache = {}  # (profile_url, event_id) -> result dict

SERIES_DIR = os.path.join(DATA_DIR, "series")

def _fetch_agents_from_match(match_id, profile_url):
    """Fetch one match page and return all unique agents the player used across maps."""
    from urllib.parse import urlparse
    profile_path = urlparse(profile_url).path
    try:
        res = requests.get(f"https://www.vlr.gg/{match_id}", headers=HEADERS, timeout=15)
        soup = BeautifulSoup(res.text, "html.parser")
    except Exception:
        return []

    seen = []
    # Iterate individual map divs (numeric data-game-id), skip "all"
    for game_div in soup.find_all("div", attrs={"data-game-id": True}):
        if game_div["data-game-id"] == "all":
            continue
        for tr in game_div.find_all("tr"):
            if tr.find("a", href=lambda h: h and profile_path in (h or "")):
                tds = tr.find_all("td")
                if len(tds) > 1:
                    for img in tds[1].find_all("img"):
                        name = img.get("alt", "").capitalize()
                        if name and name not in seen:
                            seen.append(name)
                break  # found the player in this map, move to next map
    return seen


def _orient_score(score, is_win):
    """match_results.csv stores series score from the winner's perspective
    (e.g. '2-0'). Flip it so it always reads from the player's perspective:
    a loss shows '0-2', a win shows '2-0'."""
    if not score or is_win is None:
        return score or None
    if is_win:
        return score
    parts = score.split("-", 1)
    if len(parts) == 2:
        return f"{parts[1].strip()}-{parts[0].strip()}"
    return score


_match_results_cache = None
_match_results_mtime = 0.0
def _get_match_results():
    """Cached read of match_results.csv keyed by MatchID for series-level rows
    (MapNum == 'all'). Returns {match_id: (winner_org, score, match_name)}."""
    global _match_results_cache, _match_results_mtime
    path = os.path.join(os.path.dirname(__file__), "data", "match_results.csv")
    if not os.path.exists(path):
        return {}
    mtime = os.path.getmtime(path)
    if _match_results_cache is not None and mtime <= _match_results_mtime:
        return _match_results_cache
    try:
        mr = pd.read_csv(path, dtype={"MatchID": str, "MapNum": str})
    except Exception:
        return _match_results_cache or {}
    series = mr[mr["MapNum"] == "all"]
    _match_results_cache = {
        str(r["MatchID"]): (str(r.get("WinnerOrg", "")), str(r.get("Score", "")), str(r.get("MatchName", "")))
        for _, r in series.iterrows()
    }
    _match_results_mtime = mtime
    return _match_results_cache


def _best_match_from_df(df, profile_url):
    """Return (best_row, opponent_org) for a player across all rows in df, or
    (None, None) if no qualifying rows exist."""
    df = df.copy()
    df["_url"] = df["ProfileURL"].astype(str).str.rstrip("/")
    player_rows = df[df["_url"] == profile_url.rstrip("/")].copy()
    if player_rows.empty:
        return None, None
    player_rows["R2.0"] = pd.to_numeric(player_rows["R2.0"], errors="coerce")
    if player_rows["R2.0"].dropna().empty:
        return None, None
    best = player_rows.loc[player_rows["R2.0"].idxmax()]
    match_id = str(best.get("MatchID", ""))
    player_org = str(best.get("Org", ""))
    match_rows = df[df["MatchID"].astype(str) == match_id]
    other_orgs = [o for o in match_rows["Org"].unique() if o != player_org]
    opponent_org = other_orgs[0] if other_orgs else "Unknown"
    return best, opponent_org


def get_player_best_match(profile_url, event_id):
    cache_key = (profile_url, event_id)
    if cache_key in _player_match_cache:
        return _player_match_cache[cache_key]

    # All-time (and intl-only / domestic-only variants): walk the matching
    # subset of events' series CSVs, find the player's single best match across
    # that subset, and report which event it was from.
    if event_id in ALLTIME_IDS:
        keep = _alltime_event_filter(event_id)
        best_row, best_opponent, best_event = None, None, None
        for e in ALL_EVENTS:
            if not keep(e):
                continue
            path = os.path.join(SERIES_DIR, f"{e['id']}.csv")
            if not os.path.exists(path):
                continue
            try:
                df_e = pd.read_csv(path)
            except Exception:
                continue
            row, opp = _best_match_from_df(df_e, profile_url)
            if row is None:
                continue
            if best_row is None or row["R2.0"] > best_row["R2.0"]:
                best_row, best_opponent, best_event = row, opp, e
        if best_row is None:
            result = {"error": "No match data found for this player"}
            _player_match_cache[cache_key] = result
            return result
        best, opponent_org = best_row, best_opponent
        match_id = str(best.get("MatchID", ""))
        rating, kills, deaths = best.get("R2.0"), best.get("K"), best.get("D")
        agents = _fetch_agents_from_match(match_id, profile_url) if match_id else []
        mr = _get_match_results().get(match_id, ("", "", ""))
        player_org_str = str(best_row.get("Org", ""))
        is_win = mr[0] == player_org_str if mr[0] else None
        score = _orient_score(mr[1], is_win)
        result = {
            "rating":       float(rating) if pd.notna(rating) else None,
            "kills":        int(kills)    if pd.notna(kills)   else None,
            "deaths":       int(deaths)   if pd.notna(deaths)  else None,
            "agents":       agents,
            "player_org":   player_org_str,
            "opponent":     opponent_org,
            "event_label":  best_event["label"],
            "match_id":     match_id,
            "series_score": score,
            "result":       ("W" if is_win else "L") if is_win is not None else None,
        }
        _player_match_cache[cache_key] = result
        return result

    series_path = os.path.join(SERIES_DIR, f"{event_id}.csv")
    if not os.path.exists(series_path):
        result = {"error": "Match data not available for this event"}
        _player_match_cache[cache_key] = result
        return result

    try:
        df = pd.read_csv(series_path)
    except Exception:
        result = {"error": "Could not read match data"}
        _player_match_cache[cache_key] = result
        return result

    best, opponent_org = _best_match_from_df(df, profile_url)
    if best is None:
        result = {"error": "No match data found for this player"}
        _player_match_cache[cache_key] = result
        return result

    match_id = str(best.get("MatchID", ""))
    rating   = best.get("R2.0")
    kills    = best.get("K")
    deaths   = best.get("D")

    # Single match-page fetch just for agents
    agents = _fetch_agents_from_match(match_id, profile_url) if match_id else []

    mr = _get_match_results().get(match_id, ("", "", ""))
    player_org_str = str(best.get("Org", ""))
    is_win = mr[0] == player_org_str if mr[0] else None
    score = _orient_score(mr[1], is_win)
    result = {
        "rating":       float(rating) if pd.notna(rating) else None,
        "kills":        int(kills)    if pd.notna(kills)   else None,
        "deaths":       int(deaths)   if pd.notna(deaths)  else None,
        "agents":       agents,
        "player_org":   player_org_str,
        "opponent":     opponent_org,
        "match_id":     match_id,
        "series_score": score,
        "result":       ("W" if is_win else "L") if is_win is not None else None,
    }
    _player_match_cache[cache_key] = result
    return result



MAIN_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=800">
<title>Event Leaderboards</title>
<link rel="icon" type="image/svg+xml" href="/favicon.svg">
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="/static/base.css">
<style>
  .page { position:relative; z-index:1; padding:40px 32px 60px; }
  .top-nav { display:flex; align-items:center; margin-bottom:32px; }
  .home-logo { height:80px; width:auto; display:block; opacity:.85; transition:opacity .2s; }
  .home-logo:hover { opacity:1; }
  header { text-align:center; margin-bottom:16px; }
  header h1 { font-family:'Plus Jakarta Sans',sans-serif; font-size:1rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; color:var(--soft); }
  .event-title { text-align:center; font-family:'Plus Jakarta Sans',sans-serif; font-size:clamp(1.85rem,4.4vw,3.1rem); font-weight:800; letter-spacing:-1px; margin-bottom:20px; }
  .event-selector-wrap { text-align:center; margin-bottom:24px; }
  .event-wrap { display:inline-block; position:relative; }
  .event-select { -webkit-appearance:none; appearance:none; padding:9px 38px 9px 20px; border-radius:99px; border:2px solid #f0ecf4; background:white; font-family:'DM Sans',sans-serif; font-size:.88rem; font-weight:500; color:var(--ink); cursor:pointer; box-shadow:0 2px 8px #0001; outline:none; transition:border-color .2s; min-width:220px; }
  .event-select:focus { border-color:var(--lavender); }
  .chevron { position:absolute; right:14px; top:50%; transform:translateY(-50%); pointer-events:none; color:var(--soft); font-size:.75rem; }
  .region-filter { display:flex; justify-content:center; gap:10px; margin-bottom:20px; flex-wrap:wrap; }
  .rounds-wrap { display:flex; align-items:center; justify-content:center; gap:14px; margin-top:18px; margin-bottom:36px; flex-wrap:wrap; }
  .rounds-label { font-size:.83rem; color:var(--soft); font-weight:500; }
  .rounds-val { font-family:'Plus Jakarta Sans',sans-serif; font-weight:700; color:var(--ink); min-width:40px; display:inline-block; }
  input[type=range].rounds-slider { -webkit-appearance:none; width:180px; height:4px; border-radius:99px; background:#f0ecf4; outline:none; cursor:pointer; vertical-align:middle; }
  input[type=range].rounds-slider::-webkit-slider-thumb { -webkit-appearance:none; width:18px; height:18px; border-radius:50%; background:var(--ink); cursor:pointer; }
  input[type=range].rounds-slider::-moz-range-thumb { width:18px; height:18px; border:none; border-radius:50%; background:var(--ink); cursor:pointer; }
  .filter-btn { padding:8px 22px; border-radius:99px; border:2px solid transparent; background:white; font-family:'DM Sans',sans-serif; font-size:.85rem; font-weight:500; cursor:pointer; transition:all .2s; box-shadow:0 2px 8px #0001; }
  .filter-btn:hover,.filter-btn.active { background:var(--ink); color:white; }
  .grid { display:grid; grid-template-columns:repeat(3,1fr); gap:20px; max-width:1200px; margin:0 auto; }
  @media(max-width:900px){.grid{grid-template-columns:repeat(2,1fr);}}
  @media(max-width:580px){.grid{grid-template-columns:1fr;}}
  .card { background:white; border-radius:20px; padding:22px; box-shadow:0 4px 24px #0000000a; transition:transform .2s,box-shadow .2s; cursor:pointer; }
  .card:hover { transform:translateY(-4px); box-shadow:0 12px 32px #00000014; }
  .card-header { display:flex; align-items:center; gap:10px; margin-bottom:18px; }
  .stat-pill { font-family:'Plus Jakarta Sans',sans-serif; font-size:.78rem; font-weight:700; letter-spacing:.08em; padding:4px 12px; border-radius:99px; text-transform:uppercase; }
  .card-title { font-family:'Plus Jakarta Sans',sans-serif; font-size:1.06rem; font-weight:700; }
  .pill-0{background:var(--rose);color:#8a3040} .pill-1{background:var(--sky);color:#1a4a7a}
  .pill-2{background:var(--mint);color:#1a6a4a} .pill-3{background:var(--peach);color:#8a4a1a}
  .pill-4{background:var(--lavender);color:#4a1a8a} .pill-5{background:var(--lemon);color:#6a5a1a}
  .player-row { display:flex; align-items:center; gap:12px; padding:9px 0; border-bottom:1px solid #f0ecf4; }
  .player-row:last-child { border-bottom:none; }
  .rank { font-family:'Plus Jakarta Sans',sans-serif; font-size:1.08rem; font-weight:800; color:#ccc; width:20px; text-align:center; flex-shrink:0; }
  .r1{color:#f0b429} .r2{color:#9eaab5} .r3{color:#c07c3a}
  .avatar-ph { border-radius:50%; flex-shrink:0; display:flex; align-items:center; justify-content:center; font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; color:white; }
  .player-info { flex:1; min-width:0; }
  .player-name { font-weight:500; font-size:.9rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .player-meta { font-size:.72rem; color:var(--soft); margin-top:1px; }
  .stat-val { font-family:'Plus Jakarta Sans',sans-serif; font-size:1.08rem; font-weight:700; flex-shrink:0; }
  .empty { color:var(--soft); font-size:.85rem; padding:12px 0; text-align:center; }
  .view-more { margin-top:12px; font-size:.75rem; color:#bbb; text-align:right; }
  @keyframes fadeDown{from{opacity:0;transform:translateY(-16px)}to{opacity:1;transform:translateY(0)}}
  @keyframes fadeUp{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}
  @keyframes modalIn{from{opacity:0;transform:scale(.96)}to{opacity:1;transform:scale(1)}}
  footer { text-align:center; margin-top:56px; color:var(--soft); font-size:.78rem; font-weight:300; }
  .player-row.clickable { cursor:pointer; border-radius:10px; transition:background .15s; }
  .player-row.clickable:hover { background:#f9f4fc; }
  /* ── Modal ── */
  .modal-backdrop { position:fixed; inset:0; background:#2a1f2daa; backdrop-filter:blur(4px); z-index:300; display:flex; align-items:center; justify-content:center; padding:20px; }
  .modal-box { background:white; border-radius:24px; padding:28px 32px 32px; max-width:580px; width:100%; max-height:90vh; overflow-y:auto; box-shadow:0 24px 60px #0003; position:relative; animation:modalIn .2s ease; }
  .modal-close { position:absolute; top:14px; right:18px; background:none; border:none; font-size:1.5rem; cursor:pointer; color:var(--soft); line-height:1; padding:4px; }
  .modal-close:hover { color:var(--ink); }
  .modal-player { display:flex; align-items:center; gap:18px; margin-bottom:22px; }
  .modal-player { flex-direction:column; align-items:center; text-align:center; }
  .modal-avatar { width:135px; height:135px; border-radius:50%; object-fit:cover; flex-shrink:0; }
  .modal-avatar-ph { width:135px; height:135px; border-radius:50%; flex-shrink:0; display:flex; align-items:center; justify-content:center; font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:40px; color:white; }
  .modal-name { font-family:'Plus Jakarta Sans',sans-serif; font-size:1.5rem; font-weight:800; line-height:1.1; }
  .modal-meta { color:var(--soft); font-size:.82rem; margin-top:4px; }
  .modal-stat-badge { display:inline-flex; align-items:center; gap:6px; background:#f0ecf4; border-radius:99px; padding:4px 12px; font-family:'Plus Jakarta Sans',sans-serif; font-size:.88rem; font-weight:700; margin-top:6px; }
  .modal-section-title { font-family:'Plus Jakarta Sans',sans-serif; font-size:.76rem; font-weight:700; letter-spacing:.09em; text-transform:uppercase; color:var(--soft); margin-bottom:10px; padding-bottom:8px; border-bottom:1px solid #f0ecf4; text-align:center; }
  .modal-section { margin-bottom:22px; }
  .best-match-card { background:#fdf6f0; border-radius:14px; padding:16px 18px; color:inherit; text-decoration:none; display:block; text-align:center; transition:background .15s; }
  a.best-match-card:hover { background:#f7ecdf; }
  /* Grid forces "vs" to sit on the card's central axis; team names balance
     around it regardless of length (NRG vs LOUD reads symmetric, not lopsided). */
  .bm-matchup { display:grid; grid-template-columns:1fr auto 1fr; align-items:center; column-gap:12px; font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.18rem; margin-bottom:6px; }
  .bm-side { display:flex; align-items:center; gap:8px; }
  .bm-side-left  { justify-self:end; }
  .bm-side-right { justify-self:start; }
  .bm-matchup .bm-vs { justify-self:center; color:var(--soft); font-weight:600; font-size:.85rem; }
  .bm-team-logo { height:26px; width:auto; object-fit:contain; }
  .best-match-vs { font-size:.78rem; color:var(--soft); margin-bottom:12px; }
  .bm-result { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; padding:2px 9px; border-radius:99px; font-size:.8rem; letter-spacing:.04em; }
  .bm-result-W { background:#d6f5e3; color:#1a7a3f; }
  .bm-result-L { background:#fbe0e0; color:#a51d1d; }
  /* 3-equal-column grid keeps Kills exactly under "vs" and centers each
     stat in its own column. max-width tightens the spacing so Rating and
     Deaths don't drift to the card edges. */
  .best-match-stats { display:grid; grid-template-columns:repeat(3, 1fr); align-items:center; max-width:220px; margin:0 auto; }
  .best-match-agents { display:flex; gap:6px; justify-content:center; margin-top:12px; flex-wrap:wrap; }
  .agent-chip { background:white; border-radius:8px; padding:3px 8px; font-size:.75rem; font-weight:500; color:var(--ink); border:1px solid #f0ecf4; }
  .best-match-stat { text-align:center; min-width:44px; }
  .best-match-stat-val { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.4rem; display:block; }
  .best-match-stat-lbl { font-size:.65rem; color:var(--soft); text-transform:uppercase; letter-spacing:.07em; }
  .modal-loading { color:var(--soft); font-size:.85rem; padding:16px 0; text-align:center; }
  .dist-wrap { position:relative; }
  .dist-wrap canvas { display:block; width:100%; cursor:crosshair; }
  .dist-caption { text-align:center; font-size:.78rem; color:var(--soft); margin-top:8px; }
  .dist-tooltip { display:none; position:absolute; background:white; border:1px solid #f0ecf4; border-radius:10px; padding:6px 11px; font-size:.76rem; pointer-events:none; box-shadow:0 4px 16px #0002; z-index:10; white-space:nowrap; line-height:1.5; }
</style>
</head>
<body>
<div class="page">
  <div class="top-nav">
    <a href="/"><img src="/logo.svg" alt="Home" class="home-logo"></a>
  </div>
  <header>
    <h1>Event Leaderboards</h1>
  </header>
  <div class="event-title">{{ event.label }}</div>

  <div style="text-align:center;margin-bottom:28px;line-height:1.7;">
    <span style="font-size:.72rem;color:var(--soft);font-style:italic;">Quick Notes:</span><br>
    <span style="font-size:.72rem;color:var(--soft);font-style:italic;">- 2024 Masters Shanghai has been omitted due to missing data from CN servers</span><br>
    <span style="font-size:.72rem;color:var(--soft);font-style:italic;">- 2023 Regular Season only has aggregate &ldquo;League&rdquo; data instead of being partitioned into Split 1 and Split 2</span>
  </div>
  <div class="event-selector-wrap">
    <div class="event-wrap">
      <select class="event-select" onchange="window.location='/vct/?event='+this.value">
        <optgroup label="Overall">
          <option value="all_time"{% if event_id == "all_time" %} selected{% endif %}>All-Time</option>
          <option value="all_time_intl"{% if event_id == "all_time_intl" %} selected{% endif %}>All-Time (Internationals Only)</option>
          <option value="all_time_dom"{% if event_id == "all_time_dom" %} selected{% endif %}>All-Time (Domestic Only)</option>
        </optgroup>
        {% for year, year_events in events_by_year %}
        <optgroup label="{{ year }}">
          {% for e in year_events %}
          <option value="{{ e.id }}"{% if e.id == event_id %} selected{% endif %}>{{ e.label }}</option>
          {% endfor %}
        </optgroup>
        {% endfor %}
      </select>
      <span class="chevron">&#9662;</span>
    </div>
  </div>

  {% if available_regions|length > 1 %}
  <div class="region-filter" id="region-filter">
    {% for region in available_regions %}
    <button class="filter-btn{% if loop.first %} active{% endif %}"
            onclick="switchRegion('{{ region }}',this)">
      {{ 'All Regions' if region == 'All' else region }}
    </button>
    {% endfor %}
  </div>
  {% endif %}

  <div class="rounds-wrap">
    <span class="rounds-label">Min rounds: <span class="rounds-val" id="rounds-val">50+</span></span>
    <input type="range" class="rounds-slider" id="rounds-slider" min="{{ 50 if is_alltime else 0 }}" max="300" step="10" value="50" oninput="updateMinRounds(this.value)">
  </div>

  <div class="grid" id="grid"></div>
  <footer>
    Data sourced from VLR.gg &mdash; stats load on first visit to each event
    <div style="margin-top:8px;">Like my work? Tips are appreciated! <a href="https://ko-fi.com/bobovct" target="_blank" rel="noopener" style="color:inherit;text-decoration:underline;">ko-fi.com/bobovct</a></div>
  </footer>
</div>

<div class="modal-backdrop" id="modal-backdrop" style="display:none" onclick="closeModal(event)">
  <div class="modal-box" id="modal-box">
    <button class="modal-close" onclick="closeModal()">&times;</button>
    <div class="modal-player" id="modal-player"></div>
    <div class="modal-section">
      <div class="modal-section-title">Best Match Performance {% if is_alltime %}of All-Time{% else %}of the Event{% endif %}</div>
      <div id="modal-match"><div class="modal-loading">Loading match data&hellip;</div></div>
    </div>
    <div class="modal-section dist-wrap">
      <div class="modal-section-title" id="modal-dist-title">Distribution</div>
      <canvas id="dist-canvas" height="180"></canvas>
      <div class="dist-caption" id="dist-caption"></div>
      <div class="dist-tooltip" id="dist-tooltip"></div>
    </div>
  </div>
</div>
<script>
const ROUNDS_FLOOR = {{ 50 if is_alltime else 0 }};
const DATA = {{ data_json | safe }};
const STAT_LABELS = {{ stat_labels_json | safe }};
const EVENT_ID = {{ event_id | tojson }};
const EVENT_LABEL = {{ event.label | tojson }};
const STATS = Object.keys(STAT_LABELS);
const PILL_CLASSES = ['pill-0','pill-1','pill-2','pill-3','pill-4','pill-5'];
let currentRegion = 'All';
let minRounds = (function(){
  var v = parseInt(localStorage.getItem('bobo_min_rounds'));
  if (isNaN(v)) v = 50;
  return Math.max(v, ROUNDS_FLOOR);
})();

function rankClass(i) { return i===0?'r1':i===1?'r2':i===2?'r3':''; }

function avatarColor(name) {
  const colors = ['#f4a0ae','#90b8e8','#90d4b4','#f4b878','#b498e8','#e8d478','#78c8e8','#e898c8'];
  let hash = 0;
  for (let i = 0; i < name.length; i++) hash = name.charCodeAt(i) + ((hash << 5) - hash);
  return colors[Math.abs(hash) % colors.length];
}

function showInitialsFallback(img) {
  const color = avatarColor(img.dataset.name || '');
  const s = parseInt(img.dataset.size);
  const div = document.createElement('div');
  div.className = 'avatar-ph';
  div.style.cssText = `width:${s}px;height:${s}px;font-size:${Math.round(s*0.32)}px;background:${color}`;
  div.textContent = (img.dataset.name||'').slice(0,2).toUpperCase();
  img.replaceWith(div);
}

function avatarHTML(name, size, headshot) {
  const s = size || 52;
  const color = avatarColor(name||'');
  if (headshot) {
    return `<img src="${headshot}" data-name="${name}" data-size="${s}" style="width:${s}px;height:${s}px;border-radius:50%;object-fit:cover;flex-shrink:0" onerror="showInitialsFallback(this)">`;
  }
  return `<div class="avatar-ph" style="width:${s}px;height:${s}px;font-size:${Math.round(s*0.32)}px;background:${color}">${(name||'').slice(0,2).toUpperCase()}</div>`;
}

function parseVal(v) {
  return parseFloat(String(v || '').replace('%', '')) || 0;
}

function esc(s) {
  return String(s||'').replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function getTopN(players, stat, n) {
  return (players || [])
    .filter(p => p[stat] !== undefined && p[stat] !== '')
    .filter(p => minRounds === 0 || (parseInt(p.Rnd) || 0) >= minRounds)
    .sort((a, b) => parseVal(b[stat]) - parseVal(a[stat]))
    .slice(0, n);
}

function renderCard(stat, players, idx) {
  const rows = players.length ? players.map((p, i) =>
    `<div class="player-row clickable"
       data-profile="${esc(p.ProfileURL||'')}"
       data-name="${esc(p.Player)}" data-headshot="${esc(p.HeadshotURL||'')}"
       data-org="${esc(p.Org||'')}" data-region="${esc(p.Region||'')}"
       data-fk="${esc(p.FK||'')}" data-fd="${esc(p.FD||'')}"
       data-statval="${esc(String(p[stat]||''))}" data-stat="${esc(stat)}"
       onclick="openPlayerModal(this,event)">
      <div class="rank ${rankClass(i)}">${i+1}</div>
      ${avatarHTML(p.Player, 52, p.HeadshotURL||'')}
      <div class="player-info">
        <div class="player-name">${p.Player}</div>
        <div class="player-meta">${p.Org||''} &middot; ${p.Region}${p.Event ? ' &middot; ' + p.Event : ''}</div>
      </div>
      <div class="stat-val">${p[stat]}</div>
    </div>`
  ).join('') : '<div class="empty">No data for this selection</div>';

  return `<div class="card"
    onclick="window.location='/vct/ranking/${encodeURIComponent(stat)}?event=${EVENT_ID}&region=${currentRegion}'">
    <div class="card-header">
      <div class="stat-pill ${PILL_CLASSES[idx % PILL_CLASSES.length]}">${stat}</div>
      <div class="card-title">${STAT_LABELS[stat]}</div>
    </div>
    ${rows}
    <div class="view-more">View full rankings &rarr;</div>
  </div>`;
}

// Region subsets are derived (and memoized) from the single "All" array the
// server now ships, rather than being sent as duplicate arrays per region.
const _regionPlayersCache = {};
function regionPlayers(region) {
  if (region === 'All') return DATA['All'] || [];
  if (!_regionPlayersCache[region]) {
    _regionPlayersCache[region] = (DATA['All'] || []).filter(p => p.Region === region);
  }
  return _regionPlayersCache[region];
}

function renderGrid(region) {
  const players = regionPlayers(region);
  document.getElementById('grid').innerHTML = STATS.map((stat, idx) =>
    renderCard(stat, getTopN(players, stat, 5), idx)
  ).join('');
}

function switchRegion(region, btn) {
  currentRegion = region;
  document.querySelectorAll('#region-filter .filter-btn').forEach(b => b.classList.remove('active'));
  if (btn) btn.classList.add('active');
  renderGrid(region);
}

function updateMinRounds(val) {
  minRounds = parseInt(val) || 0;
  localStorage.setItem('bobo_min_rounds', minRounds);
  document.getElementById('rounds-val').textContent = minRounds === 0 ? 'Any' : minRounds + '+';
  renderGrid(currentRegion);
}

(function syncRoundsSlider(){
  var sl = document.getElementById('rounds-slider');
  var lb = document.getElementById('rounds-val');
  if (sl) sl.value = minRounds;
  if (lb) lb.textContent = minRounds === 0 ? 'Any' : minRounds + '+';
})();

renderGrid('All');

// ── Player modal ──────────────────────────────────────────────────────────────

function openPlayerModal(el, e) {
  if (e) e.stopPropagation();
  const stat      = el.dataset.stat;
  const name      = el.dataset.name;
  const headshot  = el.dataset.headshot;
  const org       = el.dataset.org;
  const region    = el.dataset.region;
  const statVal   = el.dataset.statval;
  const profileUrl = el.dataset.profile;

  // Render player header
  const avatarEl = headshot
    ? `<img class="modal-avatar" src="${esc(headshot)}" onerror="this.replaceWith(Object.assign(document.createElement('div'),{className:'modal-avatar-ph',style:'background:'+avatarColor(${JSON.stringify(name)}),textContent:${JSON.stringify(name)}.slice(0,2).toUpperCase()}))">`
    : `<div class="modal-avatar-ph" style="background:${avatarColor(name)}">${name.slice(0,2).toUpperCase()}</div>`;
  const fk = el.dataset.fk, fd = el.dataset.fd;
  const fdInfo = (stat === 'FIWR' && fk !== '' && fd !== '')
    ? `<div class="modal-meta" style="margin-top:10px">First duels won: <b>${esc(fk)}</b> &middot; First duels lost: <b>${esc(fd)}</b></div>`
    : '';
  document.getElementById('modal-player').innerHTML = `
    ${avatarEl}
    <div>
      <div class="modal-name">${esc(name)}</div>
      <div class="modal-meta">${esc(org)} &middot; ${esc(region)}</div>
      <div class="modal-stat-badge">${esc(stat)} &nbsp; ${esc(statVal)}</div>
      ${fdInfo}
    </div>`;

  // Reset match section
  document.getElementById('modal-match').innerHTML = '<div class="modal-loading">Loading match data&hellip;</div>';

  // Draw distribution immediately from client-side data
  const allPlayers = DATA['All'] || [];
  const values = allPlayers
    .map(p => parseVal(p[stat]))
    .filter(v => v > 0);
  const statPlayers = allPlayers
    .filter(p => p[stat] !== undefined && p[stat] !== '')
    .map(p => ({name: p.Player, org: p.Org||'', event: p.Event||'', val: parseVal(p[stat])}))
    .filter(p => p.val > 0);
  document.getElementById('modal-dist-title').textContent = `${STAT_LABELS[stat]} Distribution — ${values.length} players`;
  drawDistribution(values, parseVal(statVal), stat, statPlayers);

  // Show modal
  document.getElementById('modal-backdrop').style.display = 'flex';
  document.body.style.overflow = 'hidden';

  // Fetch best match async
  if (profileUrl) {
    fetch(`/vct/api/player_best_match?url=${encodeURIComponent(profileUrl)}&event=${encodeURIComponent(EVENT_ID)}`)
      .then(r => r.json())
      .then(data => renderBestMatch(data, org))
      .catch(() => renderBestMatch({error: 'Request failed'}, org));
  } else {
    document.getElementById('modal-match').innerHTML = '<div class="modal-loading" style="color:#ccc">No profile link available</div>';
  }
}

function closeModal(e) {
  if (e && e.target !== document.getElementById('modal-backdrop')) return;
  document.getElementById('modal-backdrop').style.display = 'none';
  document.body.style.overflow = '';
}

function renderBestMatch(data, playerOrg) {
  const el = document.getElementById('modal-match');
  if (data.error) {
    el.innerHTML = `<div class="modal-loading" style="color:#ccc">${esc(data.error)}</div>`;
    return;
  }
  // Prefer the player's org AT THE TIME of the best match (from the API)
  // over the current org from the leaderboard row. For all-time, these can
  // differ — N4rrate's best was on SEN even though he's currently KC.
  const historicalOrg = data.player_org || playerOrg;
  const agentChips = (data.agents||[]).map(a =>
    `<span class="agent-chip">${esc(a)}</span>`
  ).join('');
  const teamLogo = (org) => org
    ? `<img src="/logos/${esc(org)}.png" alt="${esc(org)}" class="bm-team-logo" onerror="this.style.display='none'">`
    : '';
  const openTag = data.match_id ? `<a class="best-match-card" href="https://www.vlr.gg/${esc(data.match_id)}/" target="_blank" rel="noopener" title="Open match on VLR.gg">` : `<div class="best-match-card">`;
  const closeTag = data.match_id ? '</a>' : '</div>';
  el.innerHTML = `
    ${openTag}
      <div class="bm-matchup">
        <div class="bm-side bm-side-left">${teamLogo(historicalOrg)}<span>${esc(historicalOrg||'?')}</span></div>
        <span class="bm-vs">vs</span>
        <div class="bm-side bm-side-right"><span>${esc(data.opponent||'?')}</span>${teamLogo(data.opponent)}</div>
      </div>
      <div class="best-match-vs">${data.result ? `<span class="bm-result bm-result-${data.result}">${esc(data.result)}${data.series_score ? ' ' + esc(data.series_score) : ''}</span>` : ''}${data.result && data.event_label ? ' &middot; ' : ''}${data.event_label ? esc(data.event_label) : ''}</div>
      <div class="best-match-stats">
        <div class="best-match-stat">
          <span class="best-match-stat-val">${data.rating != null ? data.rating.toFixed(2) : '—'}</span>
          <span class="best-match-stat-lbl">Rating</span>
        </div>
        <div class="best-match-stat">
          <span class="best-match-stat-val">${data.kills ?? '—'}</span>
          <span class="best-match-stat-lbl">Kills</span>
        </div>
        <div class="best-match-stat">
          <span class="best-match-stat-val">${data.deaths ?? '—'}</span>
          <span class="best-match-stat-lbl">Deaths</span>
        </div>
      </div>
      <div class="best-match-agents">${agentChips||'<span class="agent-chip">—</span>'}</div>
    ${closeTag}`;
}

// ── Normal distribution canvas ────────────────────────────────────────────────

let distState = null;

function drawDistribution(values, playerVal, stat, statPlayers) {
  const canvas = document.getElementById('dist-canvas');
  const dpr    = window.devicePixelRatio || 1;
  const W      = canvas.parentElement.offsetWidth || 520;
  const H      = 180;
  canvas.style.width  = W + 'px';
  canvas.style.height = H + 'px';
  canvas.width  = W * dpr;
  canvas.height = H * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  if (!values.length) return;

  const mean = values.reduce((a,b)=>a+b,0) / values.length;
  const std  = Math.sqrt(values.reduce((a,b)=>a+(b-mean)**2,0) / values.length) || 0.001;

  const PAD  = { l:30, r:30, t:20, b:36 };
  const pw   = W - PAD.l - PAD.r;
  const ph   = H - PAD.t - PAD.b;
  const xMin = mean - 3.6*std;
  const xMax = mean + 3.6*std;

  const toX  = v => PAD.l + (v - xMin) / (xMax - xMin) * pw;
  const pdf  = v => Math.exp(-0.5*((v-mean)/std)**2);  // unnormalised
  const maxY = pdf(mean);
  const toY  = p => PAD.t + ph - (p / maxY) * ph;

  const N   = 400;
  const dx  = (xMax - xMin) / N;
  const pPx = toX(playerVal);

  // Shaded area to the right of player (highlight their side)
  ctx.beginPath();
  for (let i = 0; i <= N; i++) {
    const v = xMin + i*dx;
    if (v < playerVal) continue;
    i === 0 || v - dx < playerVal
      ? ctx.moveTo(toX(v), toY(pdf(v)))
      : ctx.lineTo(toX(v), toY(pdf(v)));
  }
  ctx.lineTo(toX(xMax), toY(0)); ctx.lineTo(pPx, toY(0)); ctx.closePath();
  ctx.fillStyle = '#d4b8f430'; ctx.fill();

  // Bell curve
  ctx.beginPath();
  for (let i = 0; i <= N; i++) {
    const v = xMin + i*dx;
    i === 0 ? ctx.moveTo(toX(v), toY(pdf(v))) : ctx.lineTo(toX(v), toY(pdf(v)));
  }
  ctx.strokeStyle = '#2a1f2d'; ctx.lineWidth = 2; ctx.stroke();

  // Mean line
  ctx.beginPath(); ctx.moveTo(toX(mean), toY(1)); ctx.lineTo(toX(mean), toY(0));
  ctx.strokeStyle = '#ddd'; ctx.lineWidth = 1; ctx.stroke();

  // Player line
  ctx.beginPath(); ctx.moveTo(pPx, toY(pdf(playerVal))); ctx.lineTo(pPx, toY(0));
  ctx.strokeStyle = '#7c3aed'; ctx.lineWidth = 2;
  ctx.setLineDash([5,4]); ctx.stroke(); ctx.setLineDash([]);

  // Player dot
  ctx.beginPath(); ctx.arc(pPx, toY(pdf(playerVal)), 5, 0, 2*Math.PI);
  ctx.fillStyle = '#7c3aed'; ctx.fill();

  // X axis
  ctx.beginPath(); ctx.moveTo(PAD.l, toY(0)+1); ctx.lineTo(W-PAD.r, toY(0)+1);
  ctx.strokeStyle = '#e0dce8'; ctx.lineWidth = 1; ctx.stroke();

  // Labels
  ctx.font = '11px "DM Sans",sans-serif'; ctx.fillStyle = '#9e96a8'; ctx.textAlign = 'center';
  [[xMin,''], [mean,'avg'], [xMax,'']].forEach(([v,lbl]) => {
    const label = lbl ? `${v.toFixed(2)} (${lbl})` : v.toFixed(2);
    ctx.fillText(label, toX(v), H - 8);
  });

  // Player label
  ctx.font = 'bold 12px "Plus Jakarta Sans",sans-serif';
  ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic';
  // Centered directly above the dot; white halo keeps it crisp over the curve.
  const valStr = String(playerVal);
  const lblY = Math.max(toY(pdf(playerVal)) - 13, 13);
  ctx.lineWidth = 4; ctx.lineJoin = 'round'; ctx.strokeStyle = '#fff';
  ctx.strokeText(valStr, pPx, lblY);
  ctx.fillStyle = '#7c3aed';
  ctx.fillText(valStr, pPx, lblY);

  // Percentile caption
  const below  = values.filter(v => v < playerVal).length;
  const topPct = Math.round((1 - below/values.length)*100);
  const scope  = EVENT_ID.indexOf('all_time') === 0 ? 'all-time' : `at ${EVENT_LABEL}`;
  const pctTxt = topPct <= 50
    ? `Top ${topPct}% — better than ${100-topPct}% of players ${scope}`
    : `Bottom ${100-topPct}% — better than ${100-topPct}% of players ${scope}`;
  document.getElementById('dist-caption').textContent = pctTxt;

  distState = {xMin, xMax, PAD, pw, statPlayers: statPlayers || []};
}

(function() {
  const canvas = document.getElementById('dist-canvas');
  const tooltip = document.getElementById('dist-tooltip');
  if (!canvas || !tooltip) return;
  canvas.addEventListener('mousemove', function(e) {
    if (!distState || !distState.statPlayers.length) { tooltip.style.display='none'; return; }
    const rect = canvas.getBoundingClientRect();
    const mouseX = (e.clientX - rect.left) * (canvas.width / rect.width) / (window.devicePixelRatio||1);
    const {xMin, xMax, PAD, pw, statPlayers} = distState;
    if (mouseX < PAD.l || mouseX > PAD.l + pw) { tooltip.style.display='none'; return; }
    const hoverVal = xMin + (mouseX - PAD.l) / pw * (xMax - xMin);
    const nearest = statPlayers.reduce((a,b) => Math.abs(a.val-hoverVal) < Math.abs(b.val-hoverVal) ? a : b);
    const below = statPlayers.filter(p => p.val < nearest.val).length;
    const pct = Math.round((1 - below/statPlayers.length) * 100);
    const pctLabel = pct <= 50 ? `Top ${pct}%` : `Bottom ${100-pct}%`;
    const eventLine = nearest.event ? `<br><span style="color:#9e96a8;font-size:.7rem">${esc(nearest.event)}</span>` : '';
    tooltip.innerHTML = `<strong style="font-family:'Plus Jakarta Sans',sans-serif">${esc(nearest.name)}</strong>${nearest.org ? ` <span style="color:#9e96a8;font-weight:400">${esc(nearest.org)}</span>` : ''}${eventLine}<br><span style="color:#7c3aed;font-weight:700">${nearest.val.toFixed(2)}</span> · <span style="color:#9e96a8">${pctLabel}</span>`;
    const wrapEl = canvas.parentElement;
    const wrapRect = wrapEl.getBoundingClientRect();
    const tipX = e.clientX - wrapRect.left;
    const tipY = e.clientY - wrapRect.top;
    tooltip.style.display = 'block';
    tooltip.style.left = Math.min(tipX + 14, wrapEl.offsetWidth - 190) + 'px';
    tooltip.style.top = Math.max(tipY - 78, 4) + 'px';
  });
  canvas.addEventListener('mouseleave', () => { tooltip.style.display='none'; });
})();
</script>
</body>
</html>
"""

RANKING_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=800">
<title>{{ stat }} Rankings - VCT Stats</title>
<link rel="icon" type="image/svg+xml" href="/favicon.svg">
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="/static/base.css">
<style>
  .page { position:relative; z-index:1; padding:40px 32px 60px; max-width:900px; margin:0 auto; }
  .top-nav { padding:32px 32px 0; }
  .home-logo { height:80px; width:auto; display:block; opacity:.85; transition:opacity .2s; }
  .home-logo:hover { opacity:1; }
  .back { display:inline-flex; align-items:center; gap:8px; text-decoration:none; color:var(--ink); font-family:'Plus Jakarta Sans',sans-serif; font-size:1.4rem; font-weight:700; transition:opacity .2s; margin-bottom:32px; }
  .back:hover { opacity:.7; }
  header { margin-bottom:32px; }
  header h1 { font-family:'Plus Jakarta Sans',sans-serif; font-size:2.4rem; font-weight:800; }
  header p { color:var(--soft); font-size:.9rem; margin-top:6px; }
  .region-filter { display:flex; gap:10px; margin-bottom:28px; flex-wrap:wrap; }
  .filter-btn { padding:7px 18px; border-radius:99px; border:2px solid transparent; background:white; font-family:'DM Sans',sans-serif; font-size:.82rem; font-weight:500; cursor:pointer; transition:all .2s; box-shadow:0 2px 8px #0001; }
  .filter-btn:hover,.filter-btn.active { background:var(--ink); color:white; }
  /* translateZ promotes the wrap to its own GPU layer — the big shadow is
     painted once and composited on scroll instead of re-painted every frame. */
  .table-wrap { background:white; border-radius:20px; overflow:hidden; box-shadow:0 4px 24px #0000000a; transform:translateZ(0); }
  /* table-layout:fixed stops the browser from recomputing column widths
     from every one of 2000+ rows on each layout pass. */
  table { width:100%; border-collapse:collapse; table-layout:fixed; }
  thead th { padding:14px 18px; text-align:left; font-family:'Plus Jakarta Sans',sans-serif; font-size:.8rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase; color:var(--soft); border-bottom:2px solid #f0ecf4; }
  thead th.num { text-align:right; }
  /* Explicit column widths for table-layout:fixed. Rank fits 4 digits comfortably. */
  thead th:nth-child(1) { width:120px; }
  thead th:nth-child(3) { width:90px; }
  thead th:nth-child(4) { width:130px; }
  thead th:nth-child(5) { width:110px; }
  /* No transition on every row — the 150ms hover transition fires on every
     row that scrolls past the cursor, generating dozens of concurrent
     animations during scroll. Hover becomes instant; perf jumps. */
  tbody tr:hover { background:#fdf6f0; }
  tbody td { padding:11px 18px; border-bottom:1px solid #f6f2fa; font-size:.88rem; vertical-align:middle; white-space:nowrap; }
  /* Numeric cells (rank, stat) should never ellipsis — they're always short. */
  tbody td.rank-cell, tbody td.num { overflow:visible; text-overflow:clip; }
  /* Player/Team/Region cells clip if absurdly long, no ellipsis spillover. */
  tbody td:nth-child(2), tbody td:nth-child(3), tbody td:nth-child(4) { overflow:hidden; text-overflow:ellipsis; }
  tbody td.num { text-align:right; font-family:'Plus Jakarta Sans',sans-serif; font-weight:700; font-size:1.06rem; }
  tbody tr:last-child td { border-bottom:none; }
  /* Skip rendering off-screen rows. */
  tbody tr { content-visibility:auto; contain-intrinsic-size:0 75px; }
  .rank-cell { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; color:#ccc; width:44px; }
  .r1{color:#f0b429} .r2{color:#9eaab5} .r3{color:#c07c3a}
  .player-cell { display:flex; align-items:center; gap:12px; }
  .avatar-ph { border-radius:50%; flex-shrink:0; display:flex; align-items:center; justify-content:center; font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; color:white; }
  .avatar-img { width:52px; height:52px; border-radius:50%; object-fit:cover; flex-shrink:0; }
  .badge { display:inline-block; padding:2px 8px; border-radius:99px; font-size:.7rem; font-weight:600; background:#f0ecf4; color:var(--soft); }
  .search-wrap { margin-bottom:14px; }
  .search-input { width:100%; padding:10px 18px; border-radius:99px; border:2px solid #f0ecf4; background:white; font-family:'DM Sans',sans-serif; font-size:.88rem; color:var(--ink); outline:none; box-shadow:0 2px 8px #0001; transition:border-color .2s; }
  .search-input:focus { border-color:var(--lavender); }
  .rounds-wrap { display:flex; align-items:center; gap:14px; margin-bottom:18px; flex-wrap:wrap; }
  .rounds-label { font-size:.83rem; color:var(--soft); font-weight:500; }
  .rounds-val { font-family:'Plus Jakarta Sans',sans-serif; font-weight:700; color:var(--ink); min-width:40px; display:inline-block; }
  input[type=range].rounds-slider { -webkit-appearance:none; width:180px; height:4px; border-radius:99px; background:#f0ecf4; outline:none; cursor:pointer; vertical-align:middle; }
  input[type=range].rounds-slider::-webkit-slider-thumb { -webkit-appearance:none; width:18px; height:18px; border-radius:50%; background:var(--ink); cursor:pointer; }
  input[type=range].rounds-slider::-moz-range-thumb { width:18px; height:18px; border:none; border-radius:50%; background:var(--ink); cursor:pointer; }
  .no-results { text-align:center; padding:24px; color:var(--soft); font-size:.88rem; }
  tbody tr { cursor:pointer; }
  @keyframes modalIn{from{opacity:0;transform:scale(.96)}to{opacity:1;transform:scale(1)}}
  .modal-backdrop { position:fixed; inset:0; background:#2a1f2daa; backdrop-filter:blur(4px); z-index:300; display:flex; align-items:center; justify-content:center; padding:20px; }
  .modal-box { background:white; border-radius:24px; padding:28px 32px 32px; max-width:580px; width:100%; max-height:90vh; overflow-y:auto; box-shadow:0 24px 60px #0003; position:relative; animation:modalIn .2s ease; }
  .modal-close { position:absolute; top:14px; right:18px; background:none; border:none; font-size:1.5rem; cursor:pointer; color:var(--soft); line-height:1; padding:4px; }
  .modal-close:hover { color:var(--ink); }
  .modal-player { display:flex; align-items:center; gap:18px; margin-bottom:22px; }
  .modal-player { flex-direction:column; align-items:center; text-align:center; }
  .modal-avatar { width:135px; height:135px; border-radius:50%; object-fit:cover; flex-shrink:0; }
  .modal-avatar-ph { width:135px; height:135px; border-radius:50%; flex-shrink:0; display:flex; align-items:center; justify-content:center; font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:40px; color:white; }
  .modal-name { font-family:'Plus Jakarta Sans',sans-serif; font-size:1.5rem; font-weight:800; line-height:1.1; }
  .modal-meta { color:var(--soft); font-size:.82rem; margin-top:4px; }
  .modal-stat-badge { display:inline-flex; align-items:center; gap:6px; background:#f0ecf4; border-radius:99px; padding:4px 12px; font-family:'Plus Jakarta Sans',sans-serif; font-size:.88rem; font-weight:700; margin-top:6px; }
  .modal-section-title { font-family:'Plus Jakarta Sans',sans-serif; font-size:.76rem; font-weight:700; letter-spacing:.09em; text-transform:uppercase; color:var(--soft); margin-bottom:10px; padding-bottom:8px; border-bottom:1px solid #f0ecf4; text-align:center; }
  .modal-section { margin-bottom:22px; }
  .best-match-card { background:#fdf6f0; border-radius:14px; padding:16px 18px; color:inherit; text-decoration:none; display:block; text-align:center; transition:background .15s; }
  a.best-match-card:hover { background:#f7ecdf; }
  /* Grid forces "vs" to sit on the card's central axis; team names balance
     around it regardless of length (NRG vs LOUD reads symmetric, not lopsided). */
  .bm-matchup { display:grid; grid-template-columns:1fr auto 1fr; align-items:center; column-gap:12px; font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.18rem; margin-bottom:6px; }
  .bm-side { display:flex; align-items:center; gap:8px; }
  .bm-side-left  { justify-self:end; }
  .bm-side-right { justify-self:start; }
  .bm-matchup .bm-vs { justify-self:center; color:var(--soft); font-weight:600; font-size:.85rem; }
  .bm-team-logo { height:26px; width:auto; object-fit:contain; }
  .best-match-vs { font-size:.78rem; color:var(--soft); margin-bottom:12px; }
  .bm-result { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; padding:2px 9px; border-radius:99px; font-size:.8rem; letter-spacing:.04em; }
  .bm-result-W { background:#d6f5e3; color:#1a7a3f; }
  .bm-result-L { background:#fbe0e0; color:#a51d1d; }
  /* 3-equal-column grid keeps Kills exactly under "vs" and centers each
     stat in its own column. max-width tightens the spacing so Rating and
     Deaths don't drift to the card edges. */
  .best-match-stats { display:grid; grid-template-columns:repeat(3, 1fr); align-items:center; max-width:220px; margin:0 auto; }
  .best-match-agents { display:flex; gap:6px; justify-content:center; margin-top:12px; flex-wrap:wrap; }
  .agent-chip { background:white; border-radius:8px; padding:3px 8px; font-size:.75rem; font-weight:500; color:var(--ink); border:1px solid #f0ecf4; }
  .best-match-stat { text-align:center; min-width:44px; }
  .best-match-stat-val { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.4rem; display:block; }
  .best-match-stat-lbl { font-size:.65rem; color:var(--soft); text-transform:uppercase; letter-spacing:.07em; }
  .modal-loading { color:var(--soft); font-size:.85rem; padding:16px 0; text-align:center; }
  .dist-wrap { position:relative; }
  .dist-wrap canvas { display:block; width:100%; cursor:crosshair; }
  .dist-caption { text-align:center; font-size:.78rem; color:var(--soft); margin-top:8px; }
  .dist-tooltip { display:none; position:absolute; background:white; border:1px solid #f0ecf4; border-radius:10px; padding:6px 11px; font-size:.76rem; pointer-events:none; box-shadow:0 4px 16px #0002; z-index:10; white-space:nowrap; line-height:1.5; }
</style>
</head>
<body>
<div class="top-nav">
  <a href="/"><img src="/logo.svg" alt="Home" class="home-logo"></a>
</div>
<div class="page">
  <a class="back" href="/vct/?event={{ event_id }}">&#8592; Back to dashboard</a>
  <header>
    <h1>{{ stat_label }}</h1>
    <p>{{ event.label }} &mdash; Full rankings</p>
  </header>

  {% if available_regions|length > 1 %}
  <div class="region-filter" id="region-filter">
    {% for region in available_regions %}
    <button class="filter-btn{% if region == active_region %} active{% endif %}"
            onclick="filterRegion('{{ region }}',this)">
      {{ 'All Regions' if region == 'All' else region }}
    </button>
    {% endfor %}
  </div>
  {% endif %}

  <div class="search-wrap">
    <input class="search-input" id="search" type="text" placeholder="Search player name..." oninput="applyFilters()" autocomplete="off">
  </div>
  <div class="rounds-wrap">
    <span class="rounds-label">Min rounds: <span class="rounds-val" id="rounds-val">50+</span></span>
    <input type="range" class="rounds-slider" id="rounds-slider" min="{{ 50 if is_alltime else 0 }}" max="300" step="10" value="50" oninput="updateMinRounds(this.value)">
  </div>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>Player</th>
          <th>Team</th>
          <th>Region</th>
          <th class="num">{{ stat }}</th>
        </tr>
      </thead>
      <tbody id="tbody">
        <tr id="top-spacer"><td colspan="5" style="padding:0;border:none"></td></tr>
        <tr id="bottom-spacer"><td colspan="5" style="padding:0;border:none"></td></tr>
        <tr id="no-results" style="display:none">
          <td colspan="5" class="no-results">No players match your search.</td>
        </tr>
      </tbody>
    </table>
  </div>
</div>

<div class="modal-backdrop" id="modal-backdrop" style="display:none" onclick="closeModal(event)">
  <div class="modal-box" id="modal-box">
    <button class="modal-close" onclick="closeModal()">&times;</button>
    <div class="modal-player" id="modal-player"></div>
    <div class="modal-section">
      <div class="modal-section-title">Best Match Performance {% if is_alltime %}of All-Time{% else %}of the Event{% endif %}</div>
      <div id="modal-match"><div class="modal-loading">Loading match data&hellip;</div></div>
    </div>
    <div class="modal-section dist-wrap">
      <div class="modal-section-title" id="modal-dist-title">Distribution</div>
      <canvas id="dist-canvas" height="180"></canvas>
      <div class="dist-caption" id="dist-caption"></div>
      <div class="dist-tooltip" id="dist-tooltip"></div>
    </div>
  </div>
</div>
<script>
const ROUNDS_FLOOR = {{ 50 if is_alltime else 0 }};
const STAT_VALUES = {{ stat_values_json | safe }};
const STAT_PLAYERS = {{ players_hover_json | safe }};
const EVENT_ID = {{ event_id | tojson }};
const EVENT_LABEL = {{ event.label | tojson }};
const STAT_LABELS = {{ stat_labels_json | safe }};
const CURRENT_STAT = {{ stat | tojson }};
const PLAYERS = {{ players_json | safe }};
// All-time view includes the Event column, so rows are 2 lines tall.
const HAS_EVENT_COL = PLAYERS.length > 0 && !!PLAYERS[0].Event;
const ROW_HEIGHT = HAS_EVENT_COL ? 75 : 55;
const ROW_BUFFER = 6;

function rankShowInitials(img) {
  const hue = img.dataset.hue;
  const name = img.dataset.name || '';
  const div = document.createElement('div');
  div.className = 'avatar-ph';
  div.style.cssText = `width:52px;height:52px;font-size:16px;background:hsl(${hue},55%,70%)`;
  div.textContent = name.slice(0,2).toUpperCase();
  img.replaceWith(div);
}

let activeRegion = '{{ active_region }}';
let minRounds = (function(){
  var v = parseInt(localStorage.getItem('bobo_min_rounds'));
  if (isNaN(v)) v = 50;
  return Math.max(v, ROUNDS_FLOOR);
})();

function filterRegion(region, btn) {
  activeRegion = region;
  document.querySelectorAll('#region-filter .filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  applyFilters();
}

function updateMinRounds(val) {
  minRounds = parseInt(val) || 0;
  localStorage.setItem('bobo_min_rounds', minRounds);
  document.getElementById('rounds-val').textContent = minRounds === 0 ? 'Any' : minRounds + '+';
  applyFilters();
}

(function syncRoundsSlider(){
  var sl = document.getElementById('rounds-slider');
  var lb = document.getElementById('rounds-val');
  if (sl) sl.value = minRounds;
  if (lb) lb.textContent = minRounds === 0 ? 'Any' : minRounds + '+';
})();

// ── Virtual scroller ─────────────────────────────────────────────────────────
// Only ~30 rows live in the DOM at a time. Filter operates on the data array,
// scroll updates the rendered window. Huge win on all-time (~2400 rows).

function htmlEsc(s) {
  return String(s == null ? '' : s).replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
function playerHue(name) {
  let s = 0; const n = (name || '');
  for (let i = 0; i < n.length; i++) s += n.charCodeAt(i);
  return s % 360;
}

let filteredPlayers = PLAYERS;
// The region + min-rounds pool the ranks are numbered over. `rankedPool` is the
// array (used to draw the modal distribution over the same population); its
// length is the "#rank of N" denominator shown in a player's modal.
let rankedPool = PLAYERS;
let rankedTotal = PLAYERS.length;

function rowHTML(p, rank) {
  const rc = rank === 1 ? 'r1' : rank === 2 ? 'r2' : rank === 3 ? 'r3' : '';
  const hue = playerHue(p.Player);
  const avatar = p.HeadshotURL
    ? `<img class="avatar-img" src="${htmlEsc(p.HeadshotURL)}" loading="lazy" decoding="async" data-name="${htmlEsc(p.Player)}" data-hue="${hue}" onerror="rankShowInitials(this)">`
    : `<div class="avatar-ph" style="width:52px;height:52px;font-size:16px;background:hsl(${hue},55%,70%)">${htmlEsc((p.Player||'').slice(0,2).toUpperCase())}</div>`;
  const eventTag = p.Event ? `<div class="player-event-tag">${htmlEsc(p.Event)}</div>` : '';
  return `<tr data-region="${htmlEsc(p.Region)}" data-player="${htmlEsc((p.Player||'').toLowerCase())}" data-rounds="${htmlEsc(p.Rnd)}" `
       + `data-profile="${htmlEsc(p.ProfileURL)}" data-headshot="${htmlEsc(p.HeadshotURL)}" `
       + `data-name="${htmlEsc(p.Player)}" data-org="${htmlEsc(p.Org)}" `
       + `data-fk="${htmlEsc(p.FK)}" data-fd="${htmlEsc(p.FD)}" `
       + `data-statval="${htmlEsc(p[CURRENT_STAT])}" data-stat="${htmlEsc(CURRENT_STAT)}" data-rank="${rank}" `
       + `onclick="openPlayerModal(this, event)">`
       + `<td class="rank-cell ${rc}">${rank}</td>`
       + `<td><div class="player-cell">${avatar}<div class="player-name-wrap"><div>${htmlEsc(p.Player)}</div>${eventTag}</div></div></td>`
       + `<td>${htmlEsc(p.Org)}</td>`
       + `<td><span class="badge">${htmlEsc(p.Region)}</span></td>`
       + `<td class="num">${htmlEsc(p[CURRENT_STAT])}</td>`
       + `</tr>`;
}

let _renderState = { firstIdx: -1, lastIdx: -1, totalLen: -1 };
function renderRows() {
  const tableWrap = document.querySelector('.table-wrap');
  if (!tableWrap) return;
  const wrapRect = tableWrap.getBoundingClientRect();
  const wrapTopAbs = wrapRect.top + window.scrollY;
  const headerH = 50;
  const viewportTop = window.scrollY;
  const viewportBottom = viewportTop + window.innerHeight;

  const total = filteredPlayers.length;
  let firstIdx = Math.floor((viewportTop - wrapTopAbs - headerH) / ROW_HEIGHT) - ROW_BUFFER;
  let lastIdx  = Math.ceil((viewportBottom - wrapTopAbs - headerH) / ROW_HEIGHT) + ROW_BUFFER;
  firstIdx = Math.max(0, firstIdx);
  lastIdx  = Math.min(total, lastIdx);
  if (firstIdx >= lastIdx) { firstIdx = 0; lastIdx = Math.min(total, 30); }

  // Skip if window unchanged and total length unchanged.
  if (firstIdx === _renderState.firstIdx && lastIdx === _renderState.lastIdx && total === _renderState.totalLen) return;
  _renderState = { firstIdx, lastIdx, totalLen: total };

  const topPx = firstIdx * ROW_HEIGHT;
  const botPx = Math.max(0, (total - lastIdx) * ROW_HEIGHT);

  // Build rows in slice.
  let html = '';
  for (let i = firstIdx; i < lastIdx; i++) html += rowHTML(filteredPlayers[i], filteredPlayers[i]._rank);

  const tbody = document.getElementById('tbody');
  tbody.innerHTML =
    `<tr id="top-spacer" style="height:${topPx}px"><td colspan="5" style="padding:0;border:none;height:${topPx}px"></td></tr>` +
    html +
    `<tr id="no-results" style="display:${total === 0 ? '' : 'none'}"><td colspan="5" class="no-results">No players match your search.</td></tr>` +
    `<tr id="bottom-spacer" style="height:${botPx}px"><td colspan="5" style="padding:0;border:none;height:${botPx}px"></td></tr>`;
}

function applyFilters() {
  const query = (document.getElementById('search').value || '').trim().toLowerCase();
  // Rank is assigned over the region + min-rounds pool ONLY — never the search.
  // That way a name search just hides non-matching rows; the rows that remain
  // keep their true standing (e.g. searching "n" still shows a player as #137,
  // not renumbered to #2).
  const ranked = PLAYERS.filter(p => {
    if (activeRegion !== 'All' && p.Region !== activeRegion) return false;
    if (minRounds > 0 && (parseInt(p.Rnd) || 0) < minRounds) return false;
    return true;
  });
  ranked.forEach((p, i) => { p._rank = i + 1; });
  rankedPool = ranked;
  rankedTotal = ranked.length;
  filteredPlayers = query
    ? ranked.filter(p => (p.Player || '').toLowerCase().includes(query))
    : ranked;
  // Force re-render on filter change.
  _renderState = { firstIdx: -1, lastIdx: -1, totalLen: -1 };
  renderRows();
}

let _scrollPending = false;
window.addEventListener('scroll', () => {
  if (_scrollPending) return;
  _scrollPending = true;
  requestAnimationFrame(() => { renderRows(); _scrollPending = false; });
}, { passive: true });
window.addEventListener('resize', () => {
  _renderState = { firstIdx: -1, lastIdx: -1, totalLen: -1 };
  renderRows();
});

applyFilters();

// ── Modal ─────────────────────────────────────────────────────────────────────

function esc(s) {
  return String(s||'').replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function avatarColor(name) {
  const colors = ['#f4a0ae','#90b8e8','#90d4b4','#f4b878','#b498e8','#e8d478','#78c8e8','#e898c8'];
  let hash = 0;
  for (let i = 0; i < name.length; i++) hash = name.charCodeAt(i) + ((hash << 5) - hash);
  return colors[Math.abs(hash) % colors.length];
}

function openPlayerModal(el, e) {
  if (e) e.stopPropagation();
  const stat       = el.dataset.stat;
  const name       = el.dataset.name;
  const headshot   = el.dataset.headshot;
  const org        = el.dataset.org;
  const region     = el.dataset.region;
  const statVal    = el.dataset.statval;
  const profileUrl = el.dataset.profile;

  const avatarEl = headshot
    ? `<img class="modal-avatar" src="${esc(headshot)}" onerror="this.replaceWith(Object.assign(document.createElement('div'),{className:'modal-avatar-ph',style:'background:'+avatarColor(${JSON.stringify(name)}),textContent:${JSON.stringify(name)}.slice(0,2).toUpperCase()}))">`
    : `<div class="modal-avatar-ph" style="background:${avatarColor(name)}">${name.slice(0,2).toUpperCase()}</div>`;
  const fk = el.dataset.fk, fd = el.dataset.fd;
  const fdInfo = (stat === 'FIWR' && fk !== '' && fd !== '')
    ? `<div class="modal-meta" style="margin-top:10px">First duels won: <b>${esc(fk)}</b> &middot; First duels lost: <b>${esc(fd)}</b></div>`
    : '';
  document.getElementById('modal-player').innerHTML = `
    ${avatarEl}
    <div>
      <div class="modal-name">${esc(name)}</div>
      <div class="modal-meta">${esc(org)} &middot; ${esc(region)}</div>
      <div class="modal-stat-badge">${esc(stat)} &nbsp; ${esc(statVal)}</div>
      ${fdInfo}
    </div>`;

  document.getElementById('modal-match').innerHTML = '<div class="modal-loading">Loading match data&hellip;</div>';

  const playerVal = parseFloat(String(statVal).replace('%','')) || 0;
  // Draw the distribution over the SAME pool the table ranks are numbered over
  // (region + min-rounds filtered) so the curve, percentile, count, and the
  // "#rank of N" line all agree. STAT_VALUES would be the unfiltered superset.
  const distVals = [], distPlayers = [];
  for (const p of rankedPool) {
    const v = parseFloat(String(p[stat] == null ? '' : p[stat]).replace('%',''));
    if (p[stat] !== '' && p[stat] != null && !isNaN(v)) {
      distVals.push(v);
      distPlayers.push({name: p.Player, org: p.Org || '', event: p.Event || '', val: v});
    }
  }
  document.getElementById('modal-dist-title').textContent = `${STAT_LABELS[stat]||stat} Distribution — ${distVals.length} players`;
  drawDistribution(distVals, playerVal, stat, distPlayers);
  // Lead the caption with the player's standing in this ranking, e.g.
  // "#2345 of 2370 · Bottom 1% — better than 1% of players all-time".
  if (el.dataset.rank) {
    const cap = document.getElementById('dist-caption');
    cap.textContent = `#${el.dataset.rank} of ${rankedTotal} · ${cap.textContent}`;
  }

  document.getElementById('modal-backdrop').style.display = 'flex';
  document.body.style.overflow = 'hidden';

  if (profileUrl) {
    fetch(`/vct/api/player_best_match?url=${encodeURIComponent(profileUrl)}&event=${encodeURIComponent(EVENT_ID)}`)
      .then(r => r.json())
      .then(data => renderBestMatch(data, org))
      .catch(() => renderBestMatch({error: 'Request failed'}, org));
  } else {
    document.getElementById('modal-match').innerHTML = '<div class="modal-loading" style="color:#ccc">No profile link available</div>';
  }
}

function closeModal(e) {
  if (e && e.target !== document.getElementById('modal-backdrop')) return;
  document.getElementById('modal-backdrop').style.display = 'none';
  document.body.style.overflow = '';
}

function renderBestMatch(data, playerOrg) {
  const el = document.getElementById('modal-match');
  if (data.error) {
    el.innerHTML = `<div class="modal-loading" style="color:#ccc">${esc(data.error)}</div>`;
    return;
  }
  // Prefer the player's org AT THE TIME of the best match (from the API)
  // over the current org from the leaderboard row.
  const historicalOrg = data.player_org || playerOrg;
  const agentChips = (data.agents||[]).map(a => `<span class="agent-chip">${esc(a)}</span>`).join('');
  const teamLogo = (org) => org
    ? `<img src="/logos/${esc(org)}.png" alt="${esc(org)}" class="bm-team-logo" onerror="this.style.display='none'">`
    : '';
  const openTag = data.match_id ? `<a class="best-match-card" href="https://www.vlr.gg/${esc(data.match_id)}/" target="_blank" rel="noopener" title="Open match on VLR.gg">` : `<div class="best-match-card">`;
  const closeTag = data.match_id ? '</a>' : '</div>';
  el.innerHTML = `
    ${openTag}
      <div class="bm-matchup">
        <div class="bm-side bm-side-left">${teamLogo(historicalOrg)}<span>${esc(historicalOrg||'?')}</span></div>
        <span class="bm-vs">vs</span>
        <div class="bm-side bm-side-right"><span>${esc(data.opponent||'?')}</span>${teamLogo(data.opponent)}</div>
      </div>
      <div class="best-match-vs">${data.result ? `<span class="bm-result bm-result-${data.result}">${esc(data.result)}${data.series_score ? ' ' + esc(data.series_score) : ''}</span>` : ''}${data.result && data.event_label ? ' &middot; ' : ''}${data.event_label ? esc(data.event_label) : ''}</div>
      <div class="best-match-stats">
        <div class="best-match-stat">
          <span class="best-match-stat-val">${data.rating != null ? data.rating.toFixed(2) : '—'}</span>
          <span class="best-match-stat-lbl">Rating</span>
        </div>
        <div class="best-match-stat">
          <span class="best-match-stat-val">${data.kills ?? '—'}</span>
          <span class="best-match-stat-lbl">Kills</span>
        </div>
        <div class="best-match-stat">
          <span class="best-match-stat-val">${data.deaths ?? '—'}</span>
          <span class="best-match-stat-lbl">Deaths</span>
        </div>
      </div>
      <div class="best-match-agents">${agentChips||'<span class="agent-chip">—</span>'}</div>
    ${closeTag}`;
}

let distState = null;

function drawDistribution(values, playerVal, stat, statPlayers) {
  const canvas = document.getElementById('dist-canvas');
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.parentElement.offsetWidth || 520;
  const H = 180;
  canvas.style.width = W + 'px';
  canvas.style.height = H + 'px';
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  if (!values.length) return;

  const mean = values.reduce((a,b)=>a+b,0) / values.length;
  const std  = Math.sqrt(values.reduce((a,b)=>a+(b-mean)**2,0) / values.length) || 0.001;
  const PAD  = {l:40,r:20,t:20,b:36};
  const pw   = W - PAD.l - PAD.r;
  const ph   = H - PAD.t - PAD.b;
  const xMin = mean - 3.6*std;
  const xMax = mean + 3.6*std;
  const toX  = v => PAD.l + (v - xMin) / (xMax - xMin) * pw;
  const pdf  = v => Math.exp(-0.5*((v-mean)/std)**2);
  const maxY = pdf(mean);
  const toY  = p => PAD.t + ph - (p / maxY) * ph;
  const N    = 400;
  const dx   = (xMax - xMin) / N;
  const pPx  = toX(playerVal);

  ctx.beginPath();
  for (let i = 0; i <= N; i++) {
    const v = xMin + i*dx;
    if (v < playerVal) continue;
    i === 0 || v - dx < playerVal
      ? ctx.moveTo(toX(v), toY(pdf(v)))
      : ctx.lineTo(toX(v), toY(pdf(v)));
  }
  ctx.lineTo(toX(xMax), toY(0)); ctx.lineTo(pPx, toY(0)); ctx.closePath();
  ctx.fillStyle = '#d4b8f430'; ctx.fill();

  ctx.beginPath();
  for (let i = 0; i <= N; i++) {
    const v = xMin + i*dx;
    i === 0 ? ctx.moveTo(toX(v), toY(pdf(v))) : ctx.lineTo(toX(v), toY(pdf(v)));
  }
  ctx.strokeStyle = '#2a1f2d'; ctx.lineWidth = 2; ctx.stroke();

  ctx.beginPath(); ctx.moveTo(toX(mean), toY(1)); ctx.lineTo(toX(mean), toY(0));
  ctx.strokeStyle = '#ddd'; ctx.lineWidth = 1; ctx.stroke();

  ctx.beginPath(); ctx.moveTo(pPx, toY(pdf(playerVal))); ctx.lineTo(pPx, toY(0));
  ctx.strokeStyle = '#7c3aed'; ctx.lineWidth = 2;
  ctx.setLineDash([5,4]); ctx.stroke(); ctx.setLineDash([]);

  ctx.beginPath(); ctx.arc(pPx, toY(pdf(playerVal)), 5, 0, 2*Math.PI);
  ctx.fillStyle = '#7c3aed'; ctx.fill();

  ctx.beginPath(); ctx.moveTo(PAD.l, toY(0)+1); ctx.lineTo(W-PAD.r, toY(0)+1);
  ctx.strokeStyle = '#e0dce8'; ctx.lineWidth = 1; ctx.stroke();

  ctx.font = '11px "DM Sans",sans-serif'; ctx.fillStyle = '#9e96a8'; ctx.textAlign = 'center';
  [[xMin,''], [mean,'avg'], [xMax,'']].forEach(([v,lbl]) => {
    const label = lbl ? `${v.toFixed(2)} (${lbl})` : v.toFixed(2);
    ctx.fillText(label, toX(v), H - 8);
  });

  ctx.font = 'bold 12px "Plus Jakarta Sans",sans-serif';
  ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic';
  // Centered directly above the dot; white halo keeps it crisp over the curve.
  const valStr = String(playerVal);
  const lblY = Math.max(toY(pdf(playerVal)) - 13, 13);
  ctx.lineWidth = 4; ctx.lineJoin = 'round'; ctx.strokeStyle = '#fff';
  ctx.strokeText(valStr, pPx, lblY);
  ctx.fillStyle = '#7c3aed';
  ctx.fillText(valStr, pPx, lblY);

  const below  = values.filter(v => v < playerVal).length;
  const topPct = Math.round((1 - below/values.length)*100);
  const scope  = EVENT_ID.indexOf('all_time') === 0 ? 'all-time' : `at ${EVENT_LABEL}`;
  const pctTxt = topPct <= 50
    ? `Top ${topPct}% — better than ${100-topPct}% of players ${scope}`
    : `Bottom ${100-topPct}% — better than ${100-topPct}% of players ${scope}`;
  document.getElementById('dist-caption').textContent = pctTxt;

  distState = {xMin, xMax, PAD, pw, statPlayers: statPlayers || []};
}

(function() {
  const canvas = document.getElementById('dist-canvas');
  const tooltip = document.getElementById('dist-tooltip');
  if (!canvas || !tooltip) return;
  canvas.addEventListener('mousemove', function(e) {
    if (!distState || !distState.statPlayers.length) { tooltip.style.display='none'; return; }
    const rect = canvas.getBoundingClientRect();
    const mouseX = (e.clientX - rect.left) * (canvas.width / rect.width) / (window.devicePixelRatio||1);
    const {xMin, xMax, PAD, pw, statPlayers} = distState;
    if (mouseX < PAD.l || mouseX > PAD.l + pw) { tooltip.style.display='none'; return; }
    const hoverVal = xMin + (mouseX - PAD.l) / pw * (xMax - xMin);
    const nearest = statPlayers.reduce((a,b) => Math.abs(a.val-hoverVal) < Math.abs(b.val-hoverVal) ? a : b);
    const below = statPlayers.filter(p => p.val < nearest.val).length;
    const pct = Math.round((1 - below/statPlayers.length) * 100);
    const pctLabel = pct <= 50 ? `Top ${pct}%` : `Bottom ${100-pct}%`;
    const eventLine = nearest.event ? `<br><span style="color:#9e96a8;font-size:.7rem">${esc(nearest.event)}</span>` : '';
    tooltip.innerHTML = `<strong style="font-family:'Plus Jakarta Sans',sans-serif">${esc(nearest.name)}</strong>${nearest.org ? ` <span style="color:#9e96a8;font-weight:400">${esc(nearest.org)}</span>` : ''}${eventLine}<br><span style="color:#7c3aed;font-weight:700">${nearest.val.toFixed(2)}</span> · <span style="color:#9e96a8">${pctLabel}</span>`;
    const wrapEl = canvas.parentElement;
    const wrapRect = wrapEl.getBoundingClientRect();
    const tipX = e.clientX - wrapRect.left;
    const tipY = e.clientY - wrapRect.top;
    tooltip.style.display = 'block';
    tooltip.style.left = Math.min(tipX + 14, wrapEl.offsetWidth - 190) + 'px';
    tooltip.style.top = Math.max(tipY - 78, 4) + 'px';
  });
  canvas.addEventListener('mouseleave', () => { tooltip.style.display='none'; });
})();
</script>
</body>
</html>
"""


def _most_recent_event_with_data():
    """Most recent non-CN-only event whose top-level CSV exists. ALL_EVENTS is
    ordered most-recent-first, so the first match wins."""
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    for e in ALL_EVENTS:
        if list(e["regions"].keys()) == ["CN"]:
            continue
        if os.path.exists(os.path.join(data_dir, f"{e['id']}.csv")):
            return e
    return ALL_EVENTS[0]


@vct_bp.route("/")
def index():
    _ensure_headshots_loaded()

    default_event = ALLTIME_EVENT
    event_id = request.args.get("event", ALLTIME_ID)
    event = ALLTIME_EVENTS_BY_ID.get(event_id) or next((e for e in ALL_EVENTS if e["id"] == event_id), default_event)

    cache = load_event(event)
    data, available_regions = build_data(cache, event)

    return render_template_string(
        MAIN_HTML,
        data_json=json.dumps(data),
        stat_labels_json=json.dumps(STAT_LABELS),
        event=event,
        event_id=event_id,
        is_alltime=event_id in ALLTIME_IDS,
        events_by_year=get_events_by_year(),
        available_regions=available_regions,
    )


@vct_bp.route("/ranking/<stat>")
def ranking(stat):
    _ensure_headshots_loaded()

    if stat not in STAT_LABELS:
        return "Unknown stat", 404

    default_id = "2026_stage1"
    event_id = request.args.get("event", default_id)
    event = ALLTIME_EVENTS_BY_ID.get(event_id) or next((e for e in ALL_EVENTS if e["id"] == event_id), ALL_EVENTS[0])
    active_region = request.args.get("region", "All")

    cache = _event_cache.get(event_id)
    if cache is None:
        cache = load_event(event)

    is_multi = len(event["regions"]) > 1
    is_international = not is_multi and list(event["regions"].keys()) == ["International"]
    if is_multi:
        available_regions = ["All"] + list(event["regions"].keys())
    elif is_international and not cache.empty and "Region" in cache.columns:
        actual = cache["Region"].unique().tolist()
        available_regions = ["All"] + [r for r in ["EMEA", "Americas", "Pacific", "CN"] if r in actual]
    else:
        available_regions = ["All"]

    players = get_all(cache, stat)

    def _num(v):
        try:
            return float(str(v).replace("%", ""))
        except (ValueError, TypeError):
            return None

    stat_values = [_num(p[stat]) for p in players if _num(p.get(stat)) is not None]
    players_hover = [{"name": p["Player"], "org": p.get("Org", ""), "event": p.get("Event", ""), "val": _num(p[stat])}
                     for p in players if _num(p.get(stat)) is not None]

    return render_template_string(
        RANKING_HTML,
        stat=stat,
        stat_label=STAT_LABELS[stat],
        players_json=json.dumps(players),
        active_region=active_region,
        event=event,
        event_id=event_id,
        is_alltime=event_id in ALLTIME_IDS,
        available_regions=available_regions,
        stat_values_json=json.dumps(stat_values),
        stat_labels_json=json.dumps(STAT_LABELS),
        players_hover_json=json.dumps(players_hover),
    )


@vct_bp.route("/api/player_best_match")
def player_best_match_api():
    profile_url = request.args.get("url", "")
    event_id    = request.args.get("event", "")
    if not profile_url or not event_id:
        return json.dumps({"error": "Missing parameters"}), 400
    result = get_player_best_match(profile_url, event_id)
    return json.dumps(result)


# ── Standalone player card (iframe-embeddable; opened by the /alpha home modal) ──
# Renders ONLY the player header + Best Match Performance + percentile
# normal-distribution chart — the same three pieces the /vct/ modal shows — on a
# clean white page that posts its height to the parent so the modal can size to
# fit. Reuses the modal CSS classes, drawDistribution, get_player_best_match, and
# the same per-event stat-value list (computed server-side, embedded as JSON).
PLAYER_CARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ player.Player }} — {{ stat_label }}</title>
<link rel="icon" type="image/svg+xml" href="/favicon.svg">
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="/static/base.css">
<style>
  :root { --ink:#2a1f2d; --soft:#9e96a8; }
  html { scrollbar-gutter:stable both-edges; }
  html,body { background:#fff; margin:0; }
  /* kill base.css's gradient washes so the card is solid white like the /vct/ modal */
  body::before, body::after { display:none !important; content:none !important; }
  body { font-family:'DM Sans',system-ui,sans-serif; color:var(--ink); }
  .pc-wrap { max-width:580px; margin:0 auto; padding:28px 32px 32px; box-sizing:border-box; }
  .modal-player { display:flex; flex-direction:column; align-items:center; text-align:center; gap:0; margin-bottom:22px; }
  .modal-avatar { width:135px; height:135px; border-radius:50%; object-fit:cover; flex-shrink:0; }
  .modal-avatar-ph { width:135px; height:135px; border-radius:50%; flex-shrink:0; display:flex; align-items:center; justify-content:center; font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:40px; color:white; }
  .modal-name { font-family:'Plus Jakarta Sans',sans-serif; font-size:1.5rem; font-weight:800; line-height:1.1; margin-top:14px; }
  .modal-meta { color:var(--soft); font-size:.82rem; margin-top:4px; }
  .modal-stat-badge { display:inline-flex; align-items:center; gap:6px; background:#f0ecf4; border-radius:99px; padding:4px 12px; font-family:'Plus Jakarta Sans',sans-serif; font-size:.88rem; font-weight:700; margin-top:6px; }
  .modal-section-title { font-family:'Plus Jakarta Sans',sans-serif; font-size:.76rem; font-weight:700; letter-spacing:.09em; text-transform:uppercase; color:var(--soft); margin-bottom:10px; padding-bottom:8px; border-bottom:1px solid #f0ecf4; text-align:center; }
  .modal-section { margin-bottom:22px; }
  .best-match-card { background:#fdf6f0; border-radius:14px; padding:16px 18px; color:inherit; text-decoration:none; display:block; text-align:center; transition:background .15s; }
  a.best-match-card:hover { background:#f7ecdf; }
  .bm-matchup { display:grid; grid-template-columns:1fr auto 1fr; align-items:center; column-gap:12px; font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.18rem; margin-bottom:6px; }
  .bm-side { display:flex; align-items:center; gap:8px; }
  .bm-side-left  { justify-self:end; }
  .bm-side-right { justify-self:start; }
  .bm-matchup .bm-vs { justify-self:center; color:var(--soft); font-weight:600; font-size:.85rem; }
  .bm-team-logo { height:26px; width:auto; object-fit:contain; }
  .best-match-vs { font-size:.78rem; color:var(--soft); margin-bottom:12px; }
  .bm-result { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; padding:2px 9px; border-radius:99px; font-size:.8rem; letter-spacing:.04em; }
  .bm-result-W { background:#d6f5e3; color:#1a7a3f; }
  .bm-result-L { background:#fbe0e0; color:#a51d1d; }
  .best-match-stats { display:grid; grid-template-columns:repeat(3, 1fr); align-items:center; max-width:220px; margin:0 auto; }
  .best-match-agents { display:flex; gap:6px; justify-content:center; margin-top:12px; flex-wrap:wrap; }
  .agent-chip { background:white; border-radius:8px; padding:3px 8px; font-size:.75rem; font-weight:500; color:var(--ink); border:1px solid #f0ecf4; }
  .best-match-stat { text-align:center; min-width:44px; }
  .best-match-stat-val { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.4rem; display:block; }
  .best-match-stat-lbl { font-size:.65rem; color:var(--soft); text-transform:uppercase; letter-spacing:.07em; }
  .modal-loading { color:var(--soft); font-size:.85rem; padding:16px 0; text-align:center; }
  .dist-wrap { position:relative; }
  .dist-wrap canvas { display:block; width:100%; cursor:crosshair; }
  .dist-caption { text-align:center; font-size:.78rem; color:var(--soft); margin-top:8px; }
  .dist-tooltip { display:none; position:absolute; background:white; border:1px solid #f0ecf4; border-radius:10px; padding:6px 11px; font-size:.76rem; pointer-events:none; box-shadow:0 4px 16px #0002; z-index:10; white-space:nowrap; line-height:1.5; }
</style>
</head>
<body>
<div class="pc-wrap">
  <div class="modal-player" id="modal-player"></div>
  <div class="modal-section">
    <div class="modal-section-title">Best Match Performance {% if is_alltime %}of All-Time{% else %}of the Event{% endif %}</div>
    <div id="modal-match"><div class="modal-loading">Loading match data&hellip;</div></div>
  </div>
  <div class="modal-section dist-wrap">
    <div class="modal-section-title" id="modal-dist-title">Distribution</div>
    <canvas id="dist-canvas" height="180"></canvas>
    <div class="dist-caption" id="dist-caption"></div>
    <div class="dist-tooltip" id="dist-tooltip"></div>
  </div>
</div>
<script>
const STAT       = {{ stat | tojson }};
const STAT_LABEL = {{ stat_label | tojson }};
const PLAYER     = {{ player_json | safe }};
const STAT_VALUES  = {{ stat_values_json | safe }};
const STAT_PLAYERS = {{ players_hover_json | safe }};
const BEST_MATCH = {{ best_match_json | safe }};
const EVENT_ID    = {{ event_id | tojson }};
const EVENT_LABEL = {{ event.label | tojson }};

function esc(s) {
  return String(s||'').replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
function parseVal(v) {
  return parseFloat(String(v || '').replace('%', '')) || 0;
}
function avatarColor(name) {
  const colors = ['#f4a0ae','#90b8e8','#90d4b4','#f4b878','#b498e8','#e8d478','#78c8e8','#e898c8'];
  let hash = 0;
  for (let i = 0; i < name.length; i++) hash = name.charCodeAt(i) + ((hash << 5) - hash);
  return colors[Math.abs(hash) % colors.length];
}

// ── Player header (mirrors openPlayerModal's header block) ────────────────────
(function renderHeader() {
  const name = PLAYER.name || '';
  const headshot = PLAYER.headshot || '';
  const avatarEl = headshot
    ? `<img class="modal-avatar" src="${esc(headshot)}" onerror="this.replaceWith(Object.assign(document.createElement('div'),{className:'modal-avatar-ph',style:'background:'+avatarColor(${JSON.stringify(name)}),textContent:${JSON.stringify(name)}.slice(0,2).toUpperCase()}))">`
    : `<div class="modal-avatar-ph" style="background:${avatarColor(name)}">${name.slice(0,2).toUpperCase()}</div>`;
  const fk = PLAYER.fk, fd = PLAYER.fd;
  const fdInfo = (STAT === 'FIWR' && fk !== '' && fd !== '')
    ? `<div class="modal-meta" style="margin-top:10px">First duels won: <b>${esc(fk)}</b> &middot; First duels lost: <b>${esc(fd)}</b></div>`
    : '';
  document.getElementById('modal-player').innerHTML = `
    ${avatarEl}
    <div>
      <div class="modal-name">${esc(name)}</div>
      <div class="modal-meta">${esc(PLAYER.org)} &middot; ${esc(PLAYER.region)}</div>
      <div class="modal-stat-badge">${esc(STAT)} &nbsp; ${esc(PLAYER.value)}</div>
      ${fdInfo}
    </div>`;
})();

// ── Best match (ported verbatim from the /vct/ modal's renderBestMatch) ───────
function renderBestMatch(data, playerOrg) {
  const el = document.getElementById('modal-match');
  if (!data || data.error) {
    el.innerHTML = `<div class="modal-loading" style="color:#ccc">${esc((data&&data.error)||'No match data found')}</div>`;
    return;
  }
  const historicalOrg = data.player_org || playerOrg;
  const agentChips = (data.agents||[]).map(a => `<span class="agent-chip">${esc(a)}</span>`).join('');
  const teamLogo = (org) => org
    ? `<img src="/logos/${esc(org)}.png" alt="${esc(org)}" class="bm-team-logo" onerror="this.style.display='none'">`
    : '';
  const openTag = data.match_id ? `<a class="best-match-card" href="https://www.vlr.gg/${esc(data.match_id)}/" target="_blank" rel="noopener" title="Open match on VLR.gg">` : `<div class="best-match-card">`;
  const closeTag = data.match_id ? '</a>' : '</div>';
  el.innerHTML = `
    ${openTag}
      <div class="bm-matchup">
        <div class="bm-side bm-side-left">${teamLogo(historicalOrg)}<span>${esc(historicalOrg||'?')}</span></div>
        <span class="bm-vs">vs</span>
        <div class="bm-side bm-side-right"><span>${esc(data.opponent||'?')}</span>${teamLogo(data.opponent)}</div>
      </div>
      <div class="best-match-vs">${data.result ? `<span class="bm-result bm-result-${data.result}">${esc(data.result)}${data.series_score ? ' ' + esc(data.series_score) : ''}</span>` : ''}${data.result && data.event_label ? ' &middot; ' : ''}${data.event_label ? esc(data.event_label) : ''}</div>
      <div class="best-match-stats">
        <div class="best-match-stat">
          <span class="best-match-stat-val">${data.rating != null ? data.rating.toFixed(2) : '—'}</span>
          <span class="best-match-stat-lbl">Rating</span>
        </div>
        <div class="best-match-stat">
          <span class="best-match-stat-val">${data.kills ?? '—'}</span>
          <span class="best-match-stat-lbl">Kills</span>
        </div>
        <div class="best-match-stat">
          <span class="best-match-stat-val">${data.deaths ?? '—'}</span>
          <span class="best-match-stat-lbl">Deaths</span>
        </div>
      </div>
      <div class="best-match-agents">${agentChips||'<span class="agent-chip">—</span>'}</div>
    ${closeTag}`;
}
renderBestMatch(BEST_MATCH, PLAYER.org);

// ── Normal distribution canvas (ported verbatim from the /vct/ modal) ─────────
let distState = null;
function drawDistribution(values, playerVal, stat, statPlayers) {
  const canvas = document.getElementById('dist-canvas');
  const dpr    = window.devicePixelRatio || 1;
  const H      = 180;
  // Let CSS size the display width to 100% of the box (so the canvas is always
  // full-width and centered), then read the realized width back for the backing
  // buffer. Setting a fixed px width from an early offsetWidth (in the iframe,
  // before layout settles) was locking it narrow → left-aligned in its box.
  canvas.style.width  = '100%';
  canvas.style.height = H + 'px';
  const W = canvas.clientWidth || canvas.parentElement.offsetWidth || 520;
  canvas.width  = W * dpr;
  canvas.height = H * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  if (!values.length) return;
  const mean = values.reduce((a,b)=>a+b,0) / values.length;
  const std  = Math.sqrt(values.reduce((a,b)=>a+(b-mean)**2,0) / values.length) || 0.001;
  const PAD  = { l:30, r:30, t:20, b:36 };
  const pw   = W - PAD.l - PAD.r;
  const ph   = H - PAD.t - PAD.b;
  const xMin = mean - 3.6*std;
  const xMax = mean + 3.6*std;
  const toX  = v => PAD.l + (v - xMin) / (xMax - xMin) * pw;
  const pdf  = v => Math.exp(-0.5*((v-mean)/std)**2);
  const maxY = pdf(mean);
  const toY  = p => PAD.t + ph - (p / maxY) * ph;
  const N   = 400;
  const dx  = (xMax - xMin) / N;
  const pPx = toX(playerVal);
  ctx.beginPath();
  for (let i = 0; i <= N; i++) {
    const v = xMin + i*dx;
    if (v < playerVal) continue;
    i === 0 || v - dx < playerVal ? ctx.moveTo(toX(v), toY(pdf(v))) : ctx.lineTo(toX(v), toY(pdf(v)));
  }
  ctx.lineTo(toX(xMax), toY(0)); ctx.lineTo(pPx, toY(0)); ctx.closePath();
  ctx.fillStyle = '#d4b8f430'; ctx.fill();
  ctx.beginPath();
  for (let i = 0; i <= N; i++) {
    const v = xMin + i*dx;
    i === 0 ? ctx.moveTo(toX(v), toY(pdf(v))) : ctx.lineTo(toX(v), toY(pdf(v)));
  }
  ctx.strokeStyle = '#2a1f2d'; ctx.lineWidth = 2; ctx.stroke();
  ctx.beginPath(); ctx.moveTo(toX(mean), toY(1)); ctx.lineTo(toX(mean), toY(0));
  ctx.strokeStyle = '#ddd'; ctx.lineWidth = 1; ctx.stroke();
  ctx.beginPath(); ctx.moveTo(pPx, toY(pdf(playerVal))); ctx.lineTo(pPx, toY(0));
  ctx.strokeStyle = '#7c3aed'; ctx.lineWidth = 2;
  ctx.setLineDash([5,4]); ctx.stroke(); ctx.setLineDash([]);
  ctx.beginPath(); ctx.arc(pPx, toY(pdf(playerVal)), 5, 0, 2*Math.PI);
  ctx.fillStyle = '#7c3aed'; ctx.fill();
  ctx.beginPath(); ctx.moveTo(PAD.l, toY(0)+1); ctx.lineTo(W-PAD.r, toY(0)+1);
  ctx.strokeStyle = '#e0dce8'; ctx.lineWidth = 1; ctx.stroke();
  ctx.font = '11px "DM Sans",sans-serif'; ctx.fillStyle = '#9e96a8'; ctx.textAlign = 'center';
  [[xMin,''], [mean,'avg'], [xMax,'']].forEach(([v,lbl]) => {
    const label = lbl ? `${v.toFixed(2)} (${lbl})` : v.toFixed(2);
    ctx.fillText(label, toX(v), H - 8);
  });
  ctx.font = 'bold 12px "Plus Jakarta Sans",sans-serif';
  ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic';
  // Centered directly above the dot; white halo keeps it crisp over the curve.
  const valStr = String(playerVal);
  const lblY = Math.max(toY(pdf(playerVal)) - 13, 13);
  ctx.lineWidth = 4; ctx.lineJoin = 'round'; ctx.strokeStyle = '#fff';
  ctx.strokeText(valStr, pPx, lblY);
  ctx.fillStyle = '#7c3aed';
  ctx.fillText(valStr, pPx, lblY);
  const below  = values.filter(v => v < playerVal).length;
  const topPct = Math.round((1 - below/values.length)*100);
  const scope  = EVENT_ID.indexOf('all_time') === 0 ? 'all-time' : `at ${EVENT_LABEL}`;
  const pctTxt = topPct <= 50
    ? `Top ${topPct}% — better than ${100-topPct}% of players ${scope}`
    : `Bottom ${100-topPct}% — better than ${100-topPct}% of players ${scope}`;
  document.getElementById('dist-caption').textContent = pctTxt;
  distState = {xMin, xMax, PAD, pw, statPlayers: statPlayers || []};
}

(function() {
  const canvas = document.getElementById('dist-canvas');
  const tooltip = document.getElementById('dist-tooltip');
  if (!canvas || !tooltip) return;
  canvas.addEventListener('mousemove', function(e) {
    if (!distState || !distState.statPlayers.length) { tooltip.style.display='none'; return; }
    const rect = canvas.getBoundingClientRect();
    const mouseX = (e.clientX - rect.left) * (canvas.width / rect.width) / (window.devicePixelRatio||1);
    const {xMin, xMax, PAD, pw, statPlayers} = distState;
    if (mouseX < PAD.l || mouseX > PAD.l + pw) { tooltip.style.display='none'; return; }
    const hoverVal = xMin + (mouseX - PAD.l) / pw * (xMax - xMin);
    const nearest = statPlayers.reduce((a,b) => Math.abs(a.val-hoverVal) < Math.abs(b.val-hoverVal) ? a : b);
    const below = statPlayers.filter(p => p.val < nearest.val).length;
    const pct = Math.round((1 - below/statPlayers.length) * 100);
    const pctLabel = pct <= 50 ? `Top ${pct}%` : `Bottom ${100-pct}%`;
    const eventLine = nearest.event ? `<br><span style="color:#9e96a8;font-size:.7rem">${esc(nearest.event)}</span>` : '';
    tooltip.innerHTML = `<strong style="font-family:'Plus Jakarta Sans',sans-serif">${esc(nearest.name)}</strong>${nearest.org ? ` <span style="color:#9e96a8;font-weight:400">${esc(nearest.org)}</span>` : ''}${eventLine}<br><span style="color:#7c3aed;font-weight:700">${nearest.val.toFixed(2)}</span> · <span style="color:#9e96a8">${pctLabel}</span>`;
    const wrapEl = canvas.parentElement;
    const tipX = e.clientX - wrapEl.getBoundingClientRect().left;
    const tipY = e.clientY - wrapEl.getBoundingClientRect().top;
    tooltip.style.display = 'block';
    tooltip.style.left = Math.min(tipX + 14, wrapEl.offsetWidth - 190) + 'px';
    tooltip.style.top = Math.max(tipY - 78, 4) + 'px';
  });
  canvas.addEventListener('mouseleave', () => { tooltip.style.display='none'; });
})();

function renderDist() {
  document.getElementById('modal-dist-title').textContent = `${STAT_LABEL} Distribution — ${STAT_VALUES.length} players`;
  drawDistribution(STAT_VALUES, parseVal(PLAYER.value), STAT, STAT_PLAYERS);
}
renderDist();
window.addEventListener('resize', renderDist);
window.addEventListener('load', renderDist);
// re-measure once the iframe layout has settled so the buffer matches the final width
setTimeout(renderDist, 120); setTimeout(renderDist, 400);

// ── Post height so the parent modal sizes to fit this card ────────────────────
(function(){
  function postH(){ try{ parent.postMessage({__teamH: Math.ceil(document.documentElement.scrollHeight)}, '*'); }catch(e){} }
  postH();
  window.addEventListener('load', postH);
  window.addEventListener('resize', postH);
  setTimeout(postH, 250); setTimeout(postH, 800); setTimeout(postH, 1500);
})();
</script>
</body>
</html>
"""


@vct_bp.route("/player")
def player_card():
    """Standalone, iframe-embeddable card for ONE player at ONE event: header +
    Best Match Performance + percentile distribution. Opened by the /alpha home
    page's player-leader rows (via the alpha-nav modal). Query params:
      profile — the player's VLR ProfileURL (required)
      stat    — stat key, e.g. R2.0 / KAST / HS% / FIWR (default R2.0)
      event   — event id (default: most recent event with data, same as /vct/)."""
    _ensure_headshots_loaded()

    profile_url = request.args.get("profile", "")
    stat = request.args.get("stat", "R2.0")
    if stat not in STAT_LABELS:
        stat = "R2.0"

    default_event = _most_recent_event_with_data()
    event_id = request.args.get("event", "") or default_event["id"]
    event = (ALLTIME_EVENTS_BY_ID.get(event_id)
             or next((e for e in ALL_EVENTS if e["id"] == event_id), default_event))
    event_id = event["id"]

    cache = _event_cache.get(event_id)
    if cache is None:
        cache = load_event(event)

    # Full sorted stat list for this event — the source for both the distribution
    # values and the lookup of this player's row (matched by ProfileURL).
    players = get_all(cache, stat)

    def _num(v):
        try:
            return float(str(v).replace("%", ""))
        except (ValueError, TypeError):
            return None

    stat_values = [_num(p[stat]) for p in players if _num(p.get(stat)) is not None]
    players_hover = [{"name": p["Player"], "org": p.get("Org", ""),
                      "event": p.get("Event", ""), "val": _num(p[stat])}
                     for p in players if _num(p.get(stat)) is not None]

    row = next((p for p in players if p.get("ProfileURL") == profile_url), None)
    if row is None:
        return "Player not found at this event", 404

    player = {
        "name":     row.get("Player", ""),
        "org":      row.get("Org", ""),
        "region":   row.get("Region", ""),
        "headshot": row.get("HeadshotURL", "") or "",
        "value":    str(row.get(stat, "")),
        "fk":       "" if row.get("FK", "") in (None, "") else str(row.get("FK", "")),
        "fd":       "" if row.get("FD", "") in (None, "") else str(row.get("FD", "")),
        "Player":   row.get("Player", ""),
    }

    best_match = get_player_best_match(profile_url, event_id) if profile_url else {"error": "No profile link available"}

    return render_template_string(
        PLAYER_CARD_HTML,
        stat=stat,
        stat_label=STAT_LABELS[stat],
        player=player,
        player_json=json.dumps(player),
        best_match_json=json.dumps(best_match),
        stat_values_json=json.dumps(stat_values),
        players_hover_json=json.dumps(players_hover),
        event=event,
        event_id=event_id,
        is_alltime=event_id in ALLTIME_IDS,
    )


