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
    "FKPR":   "First Kills Per Round",
}

LIVE_EVENT_ID = "2026_stage1"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

_event_cache = {}       # event_id -> DataFrame
_headshot_cache = {}    # profile_url -> headshot_url or ""
_headshots_loaded = False

_HEADSHOTS_FILE = os.path.join(os.path.dirname(__file__), "headshots.json")

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
        by_year.setdefault(e["year"], []).append(e)
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
    return cache


def load_event(event):
    event_id = event["id"]
    if event_id in _event_cache:
        return _event_cache[event_id]

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

    _event_cache[event_id] = cache
    return cache

def get_all(df, col):
    if df.empty or col not in df.columns:
        return []
    keep = [c for c in ["Player", "Org", "ProfileURL", "HeadshotURL", "Region", "Rnd", col] if c in df.columns]
    tmp = df[keep].copy()
    tmp[col] = pd.to_numeric(tmp[col].astype(str).str.replace("%", ""), errors="coerce")
    return tmp.dropna(subset=[col]).sort_values(col, ascending=False).to_dict("records")

def build_data(cache, event):
    is_multi = len(event["regions"]) > 1
    is_international = not is_multi and list(event["regions"].keys()) == ["International"]
    stat_cols = list(STAT_LABELS.keys())

    def to_records(df):
        if df.empty:
            return []
        want = ["Player", "Org", "ProfileURL", "HeadshotURL", "Region", "Rnd"] + stat_cols
        cols = [c for c in want if c in df.columns]
        return df[cols].fillna("").to_dict("records")

    data = {"All": to_records(cache)}
    if is_multi:
        for region_name in event["regions"]:
            df_r = cache[cache["Region"] == region_name] if not cache.empty else pd.DataFrame()
            data[region_name] = to_records(df_r)
    elif is_international:
        for region_name in ["EMEA", "Americas", "Pacific", "CN"]:
            df_r = cache[cache["Region"] == region_name] if not cache.empty else pd.DataFrame()
            if not df_r.empty:
                data[region_name] = to_records(df_r)

    if is_multi:
        available_regions = ["All"] + list(event["regions"].keys())
    elif is_international:
        available_regions = ["All"] + [r for r in ["EMEA", "Americas", "Pacific", "CN"] if r in data]
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
    "FUT":  "EMEA",  "GIA":  "EMEA",  "MKOI": "EMEA",  "WOL":  "EMEA",
    "M8":   "EMEA",  "FPX":  "EMEA",
    # Americas
    "SEN":  "Americas",  "G2":   "Americas",  "MIBR": "Americas",
    "NRG":  "Americas",  "100T": "Americas",  "C9":   "Americas",
    "EG":   "Americas",  "KRÜ":  "Americas",  "LEV":  "Americas",
    "FUR":  "Americas",  "LOUD": "Americas",
    # Pacific
    "PRX":  "Pacific",  "DRX":  "Pacific",  "T1":   "Pacific",
    "TLN":  "Pacific",  "GEN":  "Pacific",  "DFM":  "Pacific",
    "ZETA": "Pacific",  "RRQ":  "Pacific",  "TS":   "Pacific",
    "GE":   "Pacific",
    # CN
    "EDG":  "CN",  "BLG":  "CN",  "KRX":  "CN",  "TE":  "CN",
    "DRG":  "CN",  "ASE":  "CN",  "NS":   "CN",  "AG":  "CN",
    "XLG":  "CN",
}

# ── Player best-match scraping ────────────────────────────────────────────────

_player_match_cache = {}  # (profile_url, event_id) -> result dict

# Keywords that ALL must appear (case-sensitive) in the .m-item-event text on
# VLR.gg's player match-history page, verified by live inspection of the HTML.
EVENT_CARD_KEYWORDS = {
    "2026_stage1":           ["VCT 26:", "Stage 1"],
    "2026_masters_santiago": ["Santiago"],
    "2026_kickoff":          ["VCT 26:", "Kickoff"],
    "2025_champions":        ["Champions 2025"],
    "2025_stage2":           ["VCT 25:", "Stage 2"],
    "2025_masters_toronto":  ["Toronto"],
    "2025_stage1":           ["VCT 25:", "Stage 1"],
    "2025_masters_bangkok":  ["Bangkok"],
    "2025_kickoff":          ["VCT 25:", "Kickoff"],
    "2024_champions":        ["VCT 2024:", "Champions"],
    "2024_stage2":           ["VCT 24:", "Stage 2"],
    "2024_masters_madrid":   ["Masters Madrid"],
    "2024_stage1":           ["VCT 24:", "Stage 1"],
    "2024_kickoff":          ["VCT 24:", "Kickoff"],
    "2023_champions":        ["VCT 2023:", "Champions"],
    "2023_masters_tokyo":    ["Masters Tokyo"],
    "2023_league":           ["VCT 2023:", "Regular Season"],
    "2023_lock_in":          ["LOCK//IN"],
}

def parse_match_page_for_player(match_url, profile_path):
    """Return the player's highest-rated map entry from a match page."""
    try:
        res = requests.get(match_url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(res.text, "html.parser")
    except Exception:
        return None

    teams = [el.get_text(strip=True)
             for el in soup.select(".match-header-link-name .wf-title-med")]

    best = None
    for table_idx, table in enumerate(soup.find_all("table")):
        tbody = table.find("tbody")
        if not tbody:
            continue
        for row in tbody.find_all("tr"):
            if not row.find("a", href=profile_path):
                continue
            cells = row.find_all("td")
            if len(cells) < 6:
                continue

            agents = [img.get("alt", "").capitalize()
                      for img in cells[1].select("img") if img.get("alt")]

            def grab_f(cell):
                s = cell.find(class_=lambda c: c and "mod-both" in c)
                try: return float(s.get_text(strip=True)) if s else None
                except: return None

            def grab_i(cell):
                s = cell.find(class_=lambda c: c and "mod-both" in c)
                try: return int(s.get_text(strip=True)) if s else None
                except: return None

            rating = grab_f(cells[2])
            kills  = grab_i(cells[4])
            deaths = grab_i(cells[5])
            if rating is None:
                continue

            opponent = teams[1 - (table_idx % 2)] if len(teams) >= 2 else "Unknown"
            if best is None or rating > best["rating"]:
                best = {"rating": rating, "kills": kills or 0,
                        "deaths": deaths or 0, "agents": agents,
                        "opponent": opponent, "match_url": match_url}
    return best

def scrape_player_best_match(profile_url, event_id):
    cache_key = (profile_url, event_id)
    if cache_key in _player_match_cache:
        return _player_match_cache[cache_key]

    from urllib.parse import urlparse
    profile_path = urlparse(profile_url).path          # /player/729/zellsis
    parts = profile_path.strip("/").split("/")          # ["player","729","zellsis"]
    if len(parts) < 3:
        return {"error": "Invalid profile URL"}

    base_url = f"https://www.vlr.gg/player/matches/{parts[1]}/{parts[2]}"
    keywords = EVENT_CARD_KEYWORDS.get(event_id, [])

    # Paginate up to 4 pages: recent events land on page 1, but for 2024/2023
    # events a player's history may push those matches to page 2+.
    match_hrefs = []
    for page in range(1, 5):
        url = base_url if page == 1 else f"{base_url}?page={page}"
        try:
            res = requests.get(url, headers=HEADERS, timeout=15)
            soup = BeautifulSoup(res.text, "html.parser")
        except Exception as e:
            return {"error": f"Network error: {e}"}

        cards = soup.select("a.wf-card.m-item")
        if not cards:
            break  # no more pages

        for card in cards:
            event_div = card.select_one(".m-item-event")
            if not event_div:
                continue
            event_text = " ".join(event_div.get_text(separator=" ", strip=True).split())
            if keywords and all(kw in event_text for kw in keywords):
                href = card.get("href", "")
                if href:
                    match_hrefs.append(href)

        if match_hrefs:
            break  # found matches for this event, no need to go further
        time.sleep(0.5)

    if not match_hrefs:
        result = {"error": "No matches found for this player at this event"}
        _player_match_cache[cache_key] = result
        return result

    best = None
    for href in match_hrefs[:20]:
        data = parse_match_page_for_player("https://www.vlr.gg" + href, profile_path)
        if data and (best is None or data["rating"] > best["rating"]):
            best = data
        time.sleep(0.3)

    result = best or {"error": "Could not parse match data"}
    _player_match_cache[cache_key] = result
    return result



MAIN_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Event Leaderboards</title>
<link rel="icon" type="image/svg+xml" href="/favicon.svg">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --rose:#f4b8c1; --peach:#f9cba7; --mint:#b8e8d4;
    --sky:#b8d8f4; --lavender:#d4b8f4; --lemon:#f4edb8;
    --cream:#fdf6f0; --ink:#2a1f2d; --soft:#7a6e7e;
  }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--cream); font-family:'DM Sans',sans-serif; color:var(--ink); min-height:100vh; }
  body::before {
    content:''; position:fixed; inset:0; pointer-events:none; z-index:0;
    background:
      radial-gradient(ellipse 60% 50% at 10% 10%,#f4b8c155 0%,transparent 70%),
      radial-gradient(ellipse 50% 60% at 90% 20%,#b8d8f455 0%,transparent 70%),
      radial-gradient(ellipse 55% 45% at 15% 85%,#b8e8d455 0%,transparent 70%),
      radial-gradient(ellipse 60% 50% at 85% 80%,#d4b8f455 0%,transparent 70%);
  }
  .page { position:relative; z-index:1; padding:40px 32px 60px; }
  .top-nav { display:flex; align-items:center; margin-bottom:32px; animation:fadeDown .7s ease both; }
  .home-logo { height:80px; width:auto; display:block; opacity:.85; transition:opacity .2s; }
  .home-logo:hover { opacity:1; }
  header { text-align:center; margin-bottom:16px; animation:fadeDown .7s ease both; }
  header h1 { font-family:'Syne',sans-serif; font-size:1rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; color:var(--soft); }
  .event-title { text-align:center; font-family:'Syne',sans-serif; font-size:clamp(2rem,5vw,3.4rem); font-weight:700; letter-spacing:-1px; margin-bottom:20px; animation:fadeDown .7s .03s ease both; }
  .event-selector-wrap { text-align:center; margin-bottom:24px; animation:fadeDown .7s .05s ease both; }
  .event-wrap { display:inline-block; position:relative; }
  .event-select { -webkit-appearance:none; appearance:none; padding:9px 38px 9px 20px; border-radius:99px; border:2px solid #f0ecf4; background:white; font-family:'DM Sans',sans-serif; font-size:.88rem; font-weight:500; color:var(--ink); cursor:pointer; box-shadow:0 2px 8px #0001; outline:none; transition:border-color .2s; min-width:220px; }
  .event-select:focus { border-color:var(--lavender); }
  .chevron { position:absolute; right:14px; top:50%; transform:translateY(-50%); pointer-events:none; color:var(--soft); font-size:.75rem; }
  .region-filter { display:flex; justify-content:center; gap:10px; margin-bottom:20px; flex-wrap:wrap; animation:fadeDown .7s .1s ease both; }
  .rounds-wrap { display:flex; align-items:center; justify-content:center; gap:14px; margin-bottom:36px; flex-wrap:wrap; animation:fadeDown .7s .15s ease both; }
  .rounds-label { font-size:.83rem; color:var(--soft); font-weight:500; }
  .rounds-val { font-family:'Syne',sans-serif; font-weight:700; color:var(--ink); min-width:40px; display:inline-block; }
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
  .stat-pill { font-family:'Syne',sans-serif; font-size:.7rem; font-weight:700; letter-spacing:.08em; padding:4px 12px; border-radius:99px; text-transform:uppercase; }
  .card-title { font-family:'Syne',sans-serif; font-size:.95rem; font-weight:700; }
  .pill-0{background:var(--rose);color:#8a3040} .pill-1{background:var(--sky);color:#1a4a7a}
  .pill-2{background:var(--mint);color:#1a6a4a} .pill-3{background:var(--peach);color:#8a4a1a}
  .pill-4{background:var(--lavender);color:#4a1a8a} .pill-5{background:var(--lemon);color:#6a5a1a}
  .player-row { display:flex; align-items:center; gap:12px; padding:9px 0; border-bottom:1px solid #f0ecf4; }
  .player-row:last-child { border-bottom:none; }
  .rank { font-family:'Syne',sans-serif; font-size:1rem; font-weight:800; color:#ccc; width:20px; text-align:center; flex-shrink:0; }
  .r1{color:#f0b429} .r2{color:#9eaab5} .r3{color:#c07c3a}
  .avatar-ph { border-radius:50%; flex-shrink:0; display:flex; align-items:center; justify-content:center; font-family:'Syne',sans-serif; font-weight:800; color:white; }
  .player-info { flex:1; min-width:0; }
  .player-name { font-weight:500; font-size:.9rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .player-meta { font-size:.72rem; color:var(--soft); margin-top:1px; }
  .stat-val { font-family:'Syne',sans-serif; font-size:1rem; font-weight:700; flex-shrink:0; }
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
  .modal-avatar-ph { width:135px; height:135px; border-radius:50%; flex-shrink:0; display:flex; align-items:center; justify-content:center; font-family:'Syne',sans-serif; font-weight:800; font-size:40px; color:white; }
  .modal-name { font-family:'Syne',sans-serif; font-size:1.35rem; font-weight:800; line-height:1.1; }
  .modal-meta { color:var(--soft); font-size:.82rem; margin-top:4px; }
  .modal-stat-badge { display:inline-flex; align-items:center; gap:6px; background:#f0ecf4; border-radius:99px; padding:4px 12px; font-family:'Syne',sans-serif; font-size:.8rem; font-weight:700; margin-top:6px; }
  .modal-section-title { font-family:'Syne',sans-serif; font-size:.68rem; font-weight:700; letter-spacing:.09em; text-transform:uppercase; color:var(--soft); margin-bottom:10px; padding-bottom:8px; border-bottom:1px solid #f0ecf4; }
  .modal-section { margin-bottom:22px; }
  .best-match-card { background:#fdf6f0; border-radius:14px; padding:14px 18px; }
  .best-match-vs { font-size:.78rem; color:var(--soft); margin-bottom:10px; }
  .best-match-stats { display:flex; align-items:center; gap:20px; flex-wrap:wrap; }
  .best-match-agents { display:flex; gap:6px; margin-right:4px; }
  .agent-chip { background:white; border-radius:8px; padding:3px 8px; font-size:.75rem; font-weight:500; color:var(--ink); border:1px solid #f0ecf4; }
  .best-match-stat { text-align:center; min-width:44px; }
  .best-match-stat-val { font-family:'Syne',sans-serif; font-weight:800; font-size:1.25rem; display:block; }
  .best-match-stat-lbl { font-size:.65rem; color:var(--soft); text-transform:uppercase; letter-spacing:.07em; }
  .modal-loading { color:var(--soft); font-size:.85rem; padding:16px 0; }
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

  <div class="event-selector-wrap">
    <div class="event-wrap">
      <select class="event-select" onchange="window.location='/vct/?event='+this.value">
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
    <span class="rounds-label">Min rounds: <span class="rounds-val" id="rounds-val">Any</span></span>
    <input type="range" class="rounds-slider" id="rounds-slider" min="0" max="300" step="10" value="0" oninput="updateMinRounds(this.value)">
  </div>

  <div class="grid" id="grid"></div>
  <footer>Data sourced from VLR.gg &mdash; stats load on first visit to each event</footer>
</div>

<div class="modal-backdrop" id="modal-backdrop" style="display:none" onclick="closeModal(event)">
  <div class="modal-box" id="modal-box">
    <button class="modal-close" onclick="closeModal()">&times;</button>
    <div class="modal-player" id="modal-player"></div>
    <div class="modal-section">
      <div class="modal-section-title">Best Map Performance of the Event</div>
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
const DATA = {{ data_json | safe }};
const STAT_LABELS = {{ stat_labels_json | safe }};
const EVENT_ID = {{ event_id | tojson }};
const STATS = Object.keys(STAT_LABELS);
const PILL_CLASSES = ['pill-0','pill-1','pill-2','pill-3','pill-4','pill-5'];
let currentRegion = 'All';
let minRounds = 0;

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
       data-statval="${esc(String(p[stat]||''))}" data-stat="${esc(stat)}"
       onclick="openPlayerModal(this,event)">
      <div class="rank ${rankClass(i)}">${i+1}</div>
      ${avatarHTML(p.Player, 52, p.HeadshotURL||'')}
      <div class="player-info">
        <div class="player-name">${p.Player}</div>
        <div class="player-meta">${p.Org||''} &middot; ${p.Region}</div>
      </div>
      <div class="stat-val">${p[stat]}</div>
    </div>`
  ).join('') : '<div class="empty">No data for this selection</div>';

  return `<div class="card" style="animation:fadeUp .5s ${idx*0.06}s ease both"
    onclick="window.location='/vct/ranking/${encodeURIComponent(stat)}?event=${EVENT_ID}&region=${currentRegion}'">
    <div class="card-header">
      <div class="stat-pill ${PILL_CLASSES[idx]}">${stat}</div>
      <div class="card-title">${STAT_LABELS[stat]}</div>
    </div>
    ${rows}
    <div class="view-more">View full rankings &rarr;</div>
  </div>`;
}

function renderGrid(region) {
  const players = DATA[region] || DATA['All'];
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
  document.getElementById('rounds-val').textContent = minRounds === 0 ? 'Any' : minRounds + '+';
  renderGrid(currentRegion);
}

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
  document.getElementById('modal-player').innerHTML = `
    ${avatarEl}
    <div>
      <div class="modal-name">${esc(name)}</div>
      <div class="modal-meta">${esc(org)} &middot; ${esc(region)}</div>
      <div class="modal-stat-badge">${esc(stat)} &nbsp; ${esc(statVal)}</div>
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
    .map(p => ({name: p.Player, org: p.Org||'', val: parseVal(p[stat])}))
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
      .then(renderBestMatch)
      .catch(() => renderBestMatch({error: 'Request failed'}));
  } else {
    document.getElementById('modal-match').innerHTML = '<div class="modal-loading" style="color:#ccc">No profile link available</div>';
  }
}

function closeModal(e) {
  if (e && e.target !== document.getElementById('modal-backdrop')) return;
  document.getElementById('modal-backdrop').style.display = 'none';
  document.body.style.overflow = '';
}

function renderBestMatch(data) {
  const el = document.getElementById('modal-match');
  if (data.error) {
    el.innerHTML = `<div class="modal-loading" style="color:#ccc">${esc(data.error)}</div>`;
    return;
  }
  const agentChips = (data.agents||[]).map(a =>
    `<span class="agent-chip">${esc(a)}</span>`
  ).join('');
  el.innerHTML = `
    <div class="best-match-card">
      <div class="best-match-vs">vs ${esc(data.opponent||'?')}</div>
      <div class="best-match-stats">
        <div class="best-match-agents">${agentChips||'<span class="agent-chip">—</span>'}</div>
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
    </div>`;
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

  const PAD  = { l:40, r:20, t:20, b:36 };
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
  ctx.font = 'bold 12px "Syne",sans-serif'; ctx.fillStyle = '#7c3aed';
  ctx.textAlign = pPx > W*0.7 ? 'right' : 'left';
  ctx.fillText(String(playerVal), pPx + (pPx > W*0.7 ? -8 : 8), toY(pdf(playerVal)) - 8);

  // Percentile caption
  const below  = values.filter(v => v < playerVal).length;
  const topPct = Math.round((1 - below/values.length)*100);
  const pctTxt = topPct <= 50
    ? `Top ${topPct}% — better than ${100-topPct}% of players at this event`
    : `Bottom ${100-topPct}% — better than ${100-topPct}% of players at this event`;
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
    tooltip.innerHTML = `<strong style="font-family:'Syne',sans-serif">${esc(nearest.name)}</strong>${nearest.org ? ` <span style="color:#9e96a8;font-weight:400">${esc(nearest.org)}</span>` : ''}<br><span style="color:#7c3aed;font-weight:700">${nearest.val.toFixed(2)}</span> · <span style="color:#9e96a8">${pctLabel}</span>`;
    const wrapEl = canvas.parentElement;
    const wrapRect = wrapEl.getBoundingClientRect();
    const tipX = e.clientX - wrapRect.left;
    const tipY = e.clientY - wrapRect.top;
    tooltip.style.display = 'block';
    tooltip.style.left = Math.min(tipX + 14, wrapEl.offsetWidth - 190) + 'px';
    tooltip.style.top = Math.max(tipY - 58, 4) + 'px';
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
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ stat }} Rankings - VCT Stats</title>
<link rel="icon" type="image/svg+xml" href="/favicon.svg">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  :root { --cream:#fdf6f0; --ink:#2a1f2d; --soft:#7a6e7e; --lavender:#d4b8f4; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--cream); font-family:'DM Sans',sans-serif; color:var(--ink); min-height:100vh; }
  body::before {
    content:''; position:fixed; inset:0; pointer-events:none; z-index:0;
    background:
      radial-gradient(ellipse 60% 50% at 10% 10%,#f4b8c155 0%,transparent 70%),
      radial-gradient(ellipse 50% 60% at 90% 20%,#b8d8f455 0%,transparent 70%),
      radial-gradient(ellipse 55% 45% at 15% 85%,#b8e8d455 0%,transparent 70%),
      radial-gradient(ellipse 60% 50% at 85% 80%,#d4b8f455 0%,transparent 70%);
  }
  .page { position:relative; z-index:1; padding:40px 32px 60px; max-width:900px; margin:0 auto; }
  .top-nav { padding:32px 32px 0; }
  .home-logo { height:80px; width:auto; display:block; opacity:.85; transition:opacity .2s; }
  .home-logo:hover { opacity:1; }
  .back { display:inline-flex; align-items:center; gap:8px; text-decoration:none; color:var(--ink); font-family:'Syne',sans-serif; font-size:1.4rem; font-weight:700; transition:opacity .2s; margin-bottom:32px; }
  .back:hover { opacity:.7; }
  header { margin-bottom:32px; }
  header h1 { font-family:'Syne',sans-serif; font-size:2.4rem; font-weight:800; }
  header p { color:var(--soft); font-size:.9rem; margin-top:6px; }
  .region-filter { display:flex; gap:10px; margin-bottom:28px; flex-wrap:wrap; }
  .filter-btn { padding:7px 18px; border-radius:99px; border:2px solid transparent; background:white; font-family:'DM Sans',sans-serif; font-size:.82rem; font-weight:500; cursor:pointer; transition:all .2s; box-shadow:0 2px 8px #0001; }
  .filter-btn:hover,.filter-btn.active { background:var(--ink); color:white; }
  .table-wrap { background:white; border-radius:20px; overflow:hidden; box-shadow:0 4px 24px #0000000a; }
  table { width:100%; border-collapse:collapse; }
  thead th { padding:14px 18px; text-align:left; font-family:'Syne',sans-serif; font-size:.72rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase; color:var(--soft); border-bottom:2px solid #f0ecf4; }
  thead th.num { text-align:right; }
  tbody tr { transition:background .15s; }
  tbody tr:hover { background:#fdf6f0; }
  tbody td { padding:11px 18px; border-bottom:1px solid #f6f2fa; font-size:.88rem; vertical-align:middle; }
  tbody td.num { text-align:right; font-family:'Syne',sans-serif; font-weight:700; font-size:1rem; }
  tbody tr:last-child td { border-bottom:none; }
  .rank-cell { font-family:'Syne',sans-serif; font-weight:800; color:#ccc; width:44px; }
  .r1{color:#f0b429} .r2{color:#9eaab5} .r3{color:#c07c3a}
  .player-cell { display:flex; align-items:center; gap:12px; }
  .avatar-ph { border-radius:50%; flex-shrink:0; display:flex; align-items:center; justify-content:center; font-family:'Syne',sans-serif; font-weight:800; color:white; }
  .avatar-img { width:52px; height:52px; border-radius:50%; object-fit:cover; flex-shrink:0; }
  .badge { display:inline-block; padding:2px 8px; border-radius:99px; font-size:.7rem; font-weight:600; background:#f0ecf4; color:var(--soft); }
  .search-wrap { margin-bottom:14px; }
  .search-input { width:100%; padding:10px 18px; border-radius:99px; border:2px solid #f0ecf4; background:white; font-family:'DM Sans',sans-serif; font-size:.88rem; color:var(--ink); outline:none; box-shadow:0 2px 8px #0001; transition:border-color .2s; }
  .search-input:focus { border-color:var(--lavender); }
  .rounds-wrap { display:flex; align-items:center; gap:14px; margin-bottom:18px; flex-wrap:wrap; }
  .rounds-label { font-size:.83rem; color:var(--soft); font-weight:500; }
  .rounds-val { font-family:'Syne',sans-serif; font-weight:700; color:var(--ink); min-width:40px; display:inline-block; }
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
  .modal-avatar-ph { width:135px; height:135px; border-radius:50%; flex-shrink:0; display:flex; align-items:center; justify-content:center; font-family:'Syne',sans-serif; font-weight:800; font-size:40px; color:white; }
  .modal-name { font-family:'Syne',sans-serif; font-size:1.35rem; font-weight:800; line-height:1.1; }
  .modal-meta { color:var(--soft); font-size:.82rem; margin-top:4px; }
  .modal-stat-badge { display:inline-flex; align-items:center; gap:6px; background:#f0ecf4; border-radius:99px; padding:4px 12px; font-family:'Syne',sans-serif; font-size:.8rem; font-weight:700; margin-top:6px; }
  .modal-section-title { font-family:'Syne',sans-serif; font-size:.68rem; font-weight:700; letter-spacing:.09em; text-transform:uppercase; color:var(--soft); margin-bottom:10px; padding-bottom:8px; border-bottom:1px solid #f0ecf4; }
  .modal-section { margin-bottom:22px; }
  .best-match-card { background:#fdf6f0; border-radius:14px; padding:14px 18px; }
  .best-match-vs { font-size:.78rem; color:var(--soft); margin-bottom:10px; }
  .best-match-stats { display:flex; align-items:center; gap:20px; flex-wrap:wrap; }
  .best-match-agents { display:flex; gap:6px; margin-right:4px; }
  .agent-chip { background:white; border-radius:8px; padding:3px 8px; font-size:.75rem; font-weight:500; color:var(--ink); border:1px solid #f0ecf4; }
  .best-match-stat { text-align:center; min-width:44px; }
  .best-match-stat-val { font-family:'Syne',sans-serif; font-weight:800; font-size:1.25rem; display:block; }
  .best-match-stat-lbl { font-size:.65rem; color:var(--soft); text-transform:uppercase; letter-spacing:.07em; }
  .modal-loading { color:var(--soft); font-size:.85rem; padding:16px 0; }
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
    <span class="rounds-label">Min rounds: <span class="rounds-val" id="rounds-val">Any</span></span>
    <input type="range" class="rounds-slider" id="rounds-slider" min="0" max="300" step="10" value="0" oninput="updateMinRounds(this.value)">
  </div>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th style="width:44px">#</th>
          <th>Player</th>
          <th>Team</th>
          <th>Region</th>
          <th class="num">{{ stat }}</th>
        </tr>
      </thead>
      <tbody id="tbody">
        {% for p in players %}
        <tr data-region="{{ p.Region }}" data-player="{{ p.Player | lower }}" data-rounds="{{ p.get('Rnd', '') }}"
            data-profile="{{ p.get('ProfileURL','') }}" data-headshot="{{ p.get('HeadshotURL','') }}"
            data-name="{{ p.Player }}" data-org="{{ p.get('Org','') }}"
            data-statval="{{ p[stat] }}" data-stat="{{ stat }}"
            onclick="openPlayerModal(this, event)">
          <td class="rank-cell {% if loop.index == 1 %}r1{% elif loop.index == 2 %}r2{% elif loop.index == 3 %}r3{% endif %}">{{ loop.index }}</td>
          <td>
            <div class="player-cell">
              {% if p.get('HeadshotURL') %}
              <img class="avatar-img" src="{{ p.HeadshotURL }}" data-name="{{ p.Player }}" data-hue="{{ p.Player | player_hue }}" onerror="rankShowInitials(this)">
              {% else %}
              <div class="avatar-ph" style="width:52px;height:52px;font-size:16px;background:hsl({{ p.Player | player_hue }},55%,70%)">{{ p.Player[:2] | upper }}</div>
              {% endif %}
              {{ p.Player }}
            </div>
          </td>
          <td>{{ p.get('Org','') }}</td>
          <td><span class="badge">{{ p.Region }}</span></td>
          <td class="num">{{ p[stat] }}</td>
        </tr>
        {% endfor %}
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
      <div class="modal-section-title">Best Map Performance</div>
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
const STAT_VALUES = {{ stat_values_json | safe }};
const STAT_PLAYERS = {{ players_hover_json | safe }};
const EVENT_ID = {{ event_id | tojson }};
const STAT_LABELS = {{ stat_labels_json | safe }};
const CURRENT_STAT = {{ stat | tojson }};

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
let minRounds = 0;

function filterRegion(region, btn) {
  activeRegion = region;
  document.querySelectorAll('#region-filter .filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  applyFilters();
}

function updateMinRounds(val) {
  minRounds = parseInt(val) || 0;
  document.getElementById('rounds-val').textContent = minRounds === 0 ? 'Any' : minRounds + '+';
  applyFilters();
}

function applyFilters() {
  const query = document.getElementById('search').value.trim().toLowerCase();
  let visible = 0;
  document.querySelectorAll('#tbody tr:not(#no-results)').forEach(row => {
    const regionMatch = activeRegion === 'All' || row.dataset.region === activeRegion;
    const nameMatch = !query || row.dataset.player.includes(query);
    const roundsMatch = minRounds === 0 || (parseInt(row.dataset.rounds) || 0) >= minRounds;
    const show = regionMatch && nameMatch && roundsMatch;
    row.style.display = show ? '' : 'none';
    if (show) visible++;
  });
  document.getElementById('no-results').style.display = visible === 0 ? '' : 'none';
}

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
  document.getElementById('modal-player').innerHTML = `
    ${avatarEl}
    <div>
      <div class="modal-name">${esc(name)}</div>
      <div class="modal-meta">${esc(org)} &middot; ${esc(region)}</div>
      <div class="modal-stat-badge">${esc(stat)} &nbsp; ${esc(statVal)}</div>
    </div>`;

  document.getElementById('modal-match').innerHTML = '<div class="modal-loading">Loading match data&hellip;</div>';

  const playerVal = parseFloat(String(statVal).replace('%','')) || 0;
  document.getElementById('modal-dist-title').textContent = `${STAT_LABELS[stat]||stat} Distribution — ${STAT_VALUES.length} players`;
  drawDistribution(STAT_VALUES, playerVal, stat, STAT_PLAYERS);

  document.getElementById('modal-backdrop').style.display = 'flex';
  document.body.style.overflow = 'hidden';

  if (profileUrl) {
    fetch(`/vct/api/player_best_match?url=${encodeURIComponent(profileUrl)}&event=${encodeURIComponent(EVENT_ID)}`)
      .then(r => r.json())
      .then(renderBestMatch)
      .catch(() => renderBestMatch({error: 'Request failed'}));
  } else {
    document.getElementById('modal-match').innerHTML = '<div class="modal-loading" style="color:#ccc">No profile link available</div>';
  }
}

function closeModal(e) {
  if (e && e.target !== document.getElementById('modal-backdrop')) return;
  document.getElementById('modal-backdrop').style.display = 'none';
  document.body.style.overflow = '';
}

function renderBestMatch(data) {
  const el = document.getElementById('modal-match');
  if (data.error) {
    el.innerHTML = `<div class="modal-loading" style="color:#ccc">${esc(data.error)}</div>`;
    return;
  }
  const agentChips = (data.agents||[]).map(a => `<span class="agent-chip">${esc(a)}</span>`).join('');
  el.innerHTML = `
    <div class="best-match-card">
      <div class="best-match-vs">vs ${esc(data.opponent||'?')}</div>
      <div class="best-match-stats">
        <div class="best-match-agents">${agentChips||'<span class="agent-chip">—</span>'}</div>
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
    </div>`;
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

  ctx.font = 'bold 12px "Syne",sans-serif'; ctx.fillStyle = '#7c3aed';
  ctx.textAlign = pPx > W*0.7 ? 'right' : 'left';
  ctx.fillText(String(playerVal), pPx + (pPx > W*0.7 ? -8 : 8), toY(pdf(playerVal)) - 8);

  const below  = values.filter(v => v < playerVal).length;
  const topPct = Math.round((1 - below/values.length)*100);
  const pctTxt = topPct <= 50
    ? `Top ${topPct}% — better than ${100-topPct}% of players at this event`
    : `Bottom ${100-topPct}% — better than ${100-topPct}% of players at this event`;
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
    tooltip.innerHTML = `<strong style="font-family:'Syne',sans-serif">${esc(nearest.name)}</strong>${nearest.org ? ` <span style="color:#9e96a8;font-weight:400">${esc(nearest.org)}</span>` : ''}<br><span style="color:#7c3aed;font-weight:700">${nearest.val.toFixed(2)}</span> · <span style="color:#9e96a8">${pctLabel}</span>`;
    const wrapEl = canvas.parentElement;
    const wrapRect = wrapEl.getBoundingClientRect();
    const tipX = e.clientX - wrapRect.left;
    const tipY = e.clientY - wrapRect.top;
    tooltip.style.display = 'block';
    tooltip.style.left = Math.min(tipX + 14, wrapEl.offsetWidth - 190) + 'px';
    tooltip.style.top = Math.max(tipY - 58, 4) + 'px';
  });
  canvas.addEventListener('mouseleave', () => { tooltip.style.display='none'; });
})();
</script>
</body>
</html>
"""


@vct_bp.route("/")
def index():
    _ensure_headshots_loaded()

    default_id = ALL_EVENTS[0]["id"]
    event_id = request.args.get("event", default_id)
    event = next((e for e in ALL_EVENTS if e["id"] == event_id), ALL_EVENTS[0])

    cache = load_event(event)
    data, available_regions = build_data(cache, event)

    return render_template_string(
        MAIN_HTML,
        data_json=json.dumps(data),
        stat_labels_json=json.dumps(STAT_LABELS),
        event=event,
        event_id=event_id,
        events_by_year=get_events_by_year(),
        available_regions=available_regions,
    )


@vct_bp.route("/ranking/<stat>")
def ranking(stat):
    _ensure_headshots_loaded()

    if stat not in STAT_LABELS:
        return "Unknown stat", 404

    default_id = ALL_EVENTS[0]["id"]
    event_id = request.args.get("event", default_id)
    event = next((e for e in ALL_EVENTS if e["id"] == event_id), ALL_EVENTS[0])
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

    stat_values = [float(p[stat]) for p in players
                   if p.get(stat) not in (None, '', float('nan'))]
    players_hover = [{"name": p["Player"], "org": p.get("Org", ""), "val": float(p[stat])}
                     for p in players
                     if p.get(stat) not in (None, '', float('nan'))]

    return render_template_string(
        RANKING_HTML,
        stat=stat,
        stat_label=STAT_LABELS[stat],
        players=players,
        active_region=active_region,
        event=event,
        event_id=event_id,
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
    result = scrape_player_best_match(profile_url, event_id)
    return json.dumps(result)


