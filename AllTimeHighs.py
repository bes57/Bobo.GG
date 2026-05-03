import os
import json
import pandas as pd
from flask import Blueprint, render_template_string, request, jsonify
from MoreTestingMaybeFiles import ALL_EVENTS

highs_bp = Blueprint('highs', __name__)

DATA_DIR    = os.path.join(os.path.dirname(__file__), "data")
MAPS_DIR    = os.path.join(DATA_DIR, "maps")
SERIES_DIR  = os.path.join(DATA_DIR, "series")
HEADSHOTS_FILE = os.path.join(os.path.dirname(__file__), "headshots.json")

_headshot_cache   = {}
_headshots_loaded = False
_event_data     = None
_map_data       = None
_series_data    = None
_match_results  = None

STAT_COLS = {
    "VLR Rating":       "R2.0",
    "Kills":            "K",
    "Deaths":           "D",
    "Kill/Death Ratio": "K:D",
    "Assists":          "A",
}

MATCH_UNSUPPORTED_STATS = set()

INTERNATIONAL_IDS = {e["id"] for e in ALL_EVENTS if list(e["regions"].keys()) == ["International"]}
YEAR_MAP  = {e["id"]: e["year"]  for e in ALL_EVENTS}
LABEL_MAP = {e["id"]: e["label"] for e in ALL_EVENTS}


def _load_headshots():
    global _headshots_loaded
    if not _headshots_loaded:
        if os.path.exists(HEADSHOTS_FILE):
            with open(HEADSHOTS_FILE) as f:
                _headshot_cache.update(json.load(f))
        _headshots_loaded = True


def _parse_cl(val):
    try:
        return int(str(val).split("/")[0])
    except Exception:
        return None


def _attach_headshots(df):
    if "ProfileURL" in df.columns:
        df["HeadshotURL"] = df["ProfileURL"].map(lambda u: _headshot_cache.get(u, ""))
    else:
        df["HeadshotURL"] = ""
    return df


def _read_event_csvs(subdir=None):
    """Read all event CSVs from DATA_DIR (subdir=None) or a subdirectory."""
    folder = os.path.join(DATA_DIR, subdir) if subdir else DATA_DIR
    if not os.path.isdir(folder):
        return pd.DataFrame()

    frames = []
    for event in ALL_EVENTS:
        csv_path = os.path.join(folder, f"{event['id']}.csv")
        if not os.path.exists(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        df["_event_id"]    = event["id"]
        df["_event_label"] = LABEL_MAP[event["id"]]
        df["_year"]        = YEAR_MAP[event["id"]]
        df["_intl"]        = event["id"] in INTERNATIONAL_IDS
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_event_data():
    global _event_data
    if _event_data is not None:
        return _event_data

    _load_headshots()
    combined = _read_event_csvs()
    if combined.empty:
        _event_data = combined
        return _event_data

    if "CL" in combined.columns:
        combined["CL"] = combined["CL"].apply(_parse_cl)
    for col in ["R2.0", "K", "D", "A"]:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")
    if "K:D" in combined.columns:
        combined["K:D"] = pd.to_numeric(
            combined["K:D"].astype(str).str.replace("%", ""), errors="coerce"
        )

    _event_data = _attach_headshots(combined)
    return _event_data


def _load_match_data(subdir):
    """Load and cache per-map or per-series combined data."""
    _load_headshots()
    combined = _read_event_csvs(subdir)
    if combined.empty:
        return combined

    for col in ["R2.0", "K", "D", "A", "K:D"]:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    if "MapName" in combined.columns:
        combined["MapName"] = combined["MapName"].fillna("").astype(str).str.replace("PICK$", "", regex=True).str.strip()

    return _attach_headshots(combined)


def _load_map_data():
    global _map_data
    if _map_data is not None:
        return _map_data
    _map_data = _load_match_data("maps")
    return _map_data


def _load_series_data():
    global _series_data
    if _series_data is not None:
        return _series_data
    _series_data = _load_match_data("series")
    return _series_data


def _load_match_results():
    global _match_results
    if _match_results is not None and not _match_results.empty:
        return _match_results
    path = os.path.join(DATA_DIR, "match_results.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, dtype=str)
        df["MatchID"] = df["MatchID"].str.strip()
        df["MapNum"]  = df["MapNum"].str.strip()
        _match_results = df
    except Exception:
        return pd.DataFrame()
    return _match_results


PAGE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>All-Time Highs (and Lows)</title>
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
  .top-nav { padding:32px 32px 0; position:relative; z-index:1; }
  .home-logo { height:80px; width:auto; display:block; opacity:.85; transition:opacity .2s; }
  .home-logo:hover { opacity:1; }
  .page { position:relative; z-index:1; padding:32px 32px 60px; max-width:1100px; margin:0 auto; }
  header { margin-bottom:32px; }
  header h1 { font-family:'Syne',sans-serif; font-size:clamp(2rem,5vw,3.2rem); font-weight:700; letter-spacing:-1px; }
  header p { color:var(--soft); font-size:.88rem; margin-top:8px; font-weight:300; }
  .filters { display:flex; flex-wrap:wrap; gap:12px; margin-bottom:32px; align-items:flex-start; }
  .filter-group { display:flex; flex-direction:column; gap:4px; }
  .filter-label { font-size:.68rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; color:var(--soft); }
  .filter-select { -webkit-appearance:none; appearance:none; padding:8px 32px 8px 16px; border-radius:99px; border:2px solid #f0ecf4; background:white url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%237a6e7e'/%3E%3C/svg%3E") no-repeat right 12px center; font-family:'DM Sans',sans-serif; font-size:.85rem; font-weight:500; color:var(--ink); cursor:pointer; box-shadow:0 2px 8px #0001; outline:none; transition:border-color .2s; min-width:160px; }
  .filter-select:focus { border-color:var(--lavender); }
  .filter-select:disabled { opacity:.45; cursor:not-allowed; }
  .results-wrap { background:white; border-radius:20px; overflow:hidden; box-shadow:0 4px 24px #0000000a; }
  table { width:100%; border-collapse:collapse; }
  thead th { padding:13px 18px; text-align:left; font-family:'Syne',sans-serif; font-size:.7rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase; color:var(--soft); border-bottom:2px solid #f0ecf4; }
  thead th.num { text-align:right; }
  tbody tr { transition:background .15s; }
  tbody tr:hover { background:#fdf6f0; }
  tbody td { padding:11px 18px; border-bottom:1px solid #f6f2fa; font-size:.88rem; vertical-align:middle; }
  tbody td.num { text-align:right; font-family:'Syne',sans-serif; font-weight:700; font-size:1rem; }
  tbody tr:last-child td { border-bottom:none; }
  .rank-cell { font-family:'Syne',sans-serif; font-weight:800; color:#ccc; width:44px; text-align:center; }
  .r1{color:#f0b429} .r2{color:#9eaab5} .r3{color:#c07c3a}
  .player-cell { display:flex; align-items:center; gap:12px; }
  .avatar-ph { border-radius:50%; flex-shrink:0; display:flex; align-items:center; justify-content:center; font-family:'Syne',sans-serif; font-weight:800; color:white; font-size:14px; width:40px; height:40px; }
  .avatar-img { width:40px; height:40px; border-radius:50%; object-fit:cover; flex-shrink:0; }
  .badge { display:inline-block; padding:2px 8px; border-radius:99px; font-size:.7rem; font-weight:600; background:#f0ecf4; color:var(--soft); }
  .event-badge { display:inline-block; padding:2px 8px; border-radius:99px; font-size:.7rem; font-weight:500; background:#f4edb8; color:#6a5a1a; }
  .map-badge { display:inline-block; padding:2px 8px; border-radius:99px; font-size:.7rem; font-weight:500; background:#d4f4e8; color:#1a5a3a; margin-left:4px; }
  .empty { text-align:center; padding:40px; color:var(--soft); font-size:.88rem; }
  @keyframes fadeDown{from{opacity:0;transform:translateY(-12px)}to{opacity:1;transform:translateY(0)}}
  .page { animation:fadeDown .5s ease both; }
</style>
</head>
<body>
<div class="top-nav">
  <a href="/"><img src="/logo.svg" alt="Home" class="home-logo"></a>
</div>
<div class="page">
  <header>
    <h1>All-Time Highs (and Lows)</h1>
    <p>Records across all VCT franchised events, 2023&ndash;2026 Masters Santiago</p>
  </header>

  <div class="filters">
    <div class="filter-group">
      <span class="filter-label">Direction</span>
      <select class="filter-select" id="f-direction" onchange="fetchResults()">
        <option value="high">Highest</option>
        <option value="low">Lowest</option>
      </select>
    </div>
    <div class="filter-group">
      <span class="filter-label">Stat</span>
      <select class="filter-select" id="f-stat" onchange="onStatChange()">
        <option value="VLR Rating">VLR Rating</option>
        <option value="Kills">Kills</option>
        <option value="Deaths">Deaths</option>
        <option value="Kill/Death Ratio">Kill/Death Ratio</option>
        <option value="Assists">Assists</option>
      </select>
    </div>
    <div class="filter-group">
      <span class="filter-label">Format</span>
      <select class="filter-select" id="f-format" onchange="onFormatChange()">
        <option value="map">One Map</option>
        <option value="bo3">Bo3</option>
        <option value="bo5">Bo5</option>
      </select>
    </div>
    <div class="filter-group">
      <span class="filter-label">Year</span>
      <select class="filter-select" id="f-year" onchange="fetchResults()">
        <option value="all">All-Time</option>
        <option value="2026">2026</option>
        <option value="2025">2025</option>
        <option value="2024">2024</option>
        <option value="2023">2023</option>
      </select>
    </div>
    <div class="filter-group">
      <span class="filter-label">Context</span>
      <select class="filter-select" id="f-context" onchange="fetchResults()">
        <option value="all">All Events</option>
        <option value="intl">At an International</option>
        <option value="regional">Regional Only</option>
        <option value="win">In a Win</option>
        <option value="loss">In a Loss</option>
      </select>
    </div>
  </div>

  <div class="results-wrap">
    <table>
      <thead>
        <tr>
          <th style="width:44px">#</th>
          <th>Player</th>
          <th>Team</th>
          <th>Region</th>
          <th>Event</th>
          <th id="map-col-header" style="display:none">Map</th>
          <th>Result</th>
          <th class="num" id="stat-col-header">Value</th>
        </tr>
      </thead>
      <tbody id="results-body">
        <tr><td colspan="8" class="empty">Loading&hellip;</td></tr>
      </tbody>
    </table>
  </div>
</div>

<script>
const MATCH_UNSUPPORTED = new Set(["# of Clutches"]);

function esc(s) {
  return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
function avatarColor(name) {
  const colors = ['#f4a0ae','#90b8e8','#90d4b4','#f4b878','#b498e8','#e8d478','#78c8e8','#e898c8'];
  let hash = 0;
  for (let i = 0; i < name.length; i++) hash = name.charCodeAt(i) + ((hash << 5) - hash);
  return colors[Math.abs(hash) % colors.length];
}
function rankClass(i) { return i===0?'r1':i===1?'r2':i===2?'r3':''; }

function onFormatChange() {
  const fmt  = document.getElementById('f-format').value;
  const stat = document.getElementById('f-stat').value;
  if (MATCH_UNSUPPORTED.has(stat)) {
    document.getElementById('f-stat').value = 'Kills';
  }
  for (const opt of document.getElementById('f-stat').options) {
    opt.disabled = MATCH_UNSUPPORTED.has(opt.value);
  }
  document.getElementById('map-col-header').style.display = fmt === 'map' ? '' : 'none';
  fetchResults();
}

function onStatChange() {
  fetchResults();
}

function fetchResults() {
  const direction = document.getElementById('f-direction').value;
  const stat      = document.getElementById('f-stat').value;
  const fmt       = document.getElementById('f-format').value;
  const year      = document.getElementById('f-year').value;
  const context   = document.getElementById('f-context').value;

  document.getElementById('stat-col-header').textContent = stat;
  document.getElementById('results-body').innerHTML = '<tr><td colspan="8" class="empty">Loading&hellip;</td></tr>';

  fetch(`/highs/api/results?direction=${encodeURIComponent(direction)}&stat=${encodeURIComponent(stat)}&format=${encodeURIComponent(fmt)}&year=${encodeURIComponent(year)}&context=${encodeURIComponent(context)}`)
    .then(r => r.json())
    .then(data => {
      if (!data.length) {
        document.getElementById('results-body').innerHTML = '<tr><td colspan="8" class="empty">No data found for this combination.</td></tr>';
        return;
      }
      const showMap = document.getElementById('f-format').value === 'map';
      document.getElementById('map-col-header').style.display = showMap ? '' : 'none';
      document.getElementById('results-body').innerHTML = data.map((row, i) => {
        const avatar = row.headshot
          ? `<img class="avatar-img" src="${esc(row.headshot)}" onerror="this.replaceWith(Object.assign(document.createElement('div'),{className:'avatar-ph',style:'background:'+${JSON.stringify(avatarColor(row.player))},textContent:${JSON.stringify((row.player||'').slice(0,2).toUpperCase())}}))">`
          : `<div class="avatar-ph" style="background:${avatarColor(row.player)}">${esc((row.player||'').slice(0,2).toUpperCase())}</div>`;
        const mapCell = showMap ? `<td>${esc(row.map_name||'')}</td>` : '';
        const isKD = document.getElementById('f-stat').value === 'Kill/Death Ratio';
        const valDisplay = (isKD && row.kills != null && row.deaths != null)
          ? `${esc(String(row.value))} <span style="font-size:.75rem;font-weight:400;color:var(--soft)">(${row.kills}/${row.deaths})</span>`
          : esc(String(row.value));
        let resultCell = '<td></td>';
        if (row.result) {
          const won = row.result.startsWith('W');
          resultCell = `<td><span style="display:inline-block;padding:2px 8px;border-radius:99px;font-size:.7rem;font-weight:600;background:${won?'#d4f4e8':'#fde8e8'};color:${won?'#1a5a3a':'#7a1a1a'}">${esc(row.result)}</span></td>`;
        }
        return `<tr>
          <td class="rank-cell ${rankClass(i)}">${i+1}</td>
          <td><div class="player-cell">${avatar}<span>${esc(row.player)}</span></div></td>
          <td>${esc(row.org||'')}</td>
          <td><span class="badge">${esc(row.region||'')}</span></td>
          <td><span class="event-badge">${esc(row.event)}</span>${row.match_name ? `<div style="font-size:.7rem;color:var(--soft);margin-top:3px">${esc(row.match_name)}</div>` : ''}</td>
          ${mapCell}
          ${resultCell}
          <td class="num">${valDisplay}</td>
        </tr>`;
      }).join('');
    })
    .catch(() => {
      document.getElementById('results-body').innerHTML = '<tr><td colspan="8" class="empty">Failed to load results.</td></tr>';
    });
}

fetchResults();
</script>
</body>
</html>
"""


@highs_bp.route("/")
def index():
    return render_template_string(PAGE_HTML)


@highs_bp.route("/api/results")
def api_results():
    direction = request.args.get("direction", "high")
    stat_name = request.args.get("stat", "VLR Rating")
    fmt       = request.args.get("format", "event")
    year      = request.args.get("year", "all")
    context   = request.args.get("context", "all")

    col = STAT_COLS.get(stat_name)
    if not col:
        return jsonify([])

    # Pick the right dataset
    if fmt == "map":
        df = _load_map_data()
    elif fmt in ("bo3", "bo5"):
        df = _load_series_data()
        if not df.empty and "SeriesFormat" in df.columns:
            df = df[df["SeriesFormat"] == fmt]
    else:
        df = _load_event_data()

    if df.empty or col not in df.columns:
        return jsonify([])

    # Year filter
    if year != "all":
        df = df[df["_year"] == int(year)]

    # Context filter
    if context == "intl":
        df = df[df["_intl"] == True]
    elif context == "regional":
        df = df[df["_intl"] == False]
    elif context in ("win", "loss"):
        results_df = _load_match_results()
        if results_df.empty:
            return jsonify([])
        df = df.copy()
        df["MatchID"] = df["MatchID"].astype(str).str.strip()
        if fmt == "map":
            df["MapNum"] = df["MapNum"].astype(str).str.strip()
            lookup = results_df[results_df["MapNum"] != "all"][["MatchID", "MapNum", "WinnerOrg"]]
            merged = df.merge(lookup, on=["MatchID", "MapNum"], how="left")
        else:
            lookup = results_df[results_df["MapNum"] == "all"][["MatchID", "WinnerOrg"]]
            merged = df.merge(lookup, on="MatchID", how="left")
        if context == "win":
            df = merged[merged["WinnerOrg"] == merged["Org"]].drop(columns=["WinnerOrg"])
        else:
            df = merged[(merged["WinnerOrg"].notna()) & (merged["WinnerOrg"] != merged["Org"])].drop(columns=["WinnerOrg"])

    df = df.dropna(subset=[col])

    ascending = (direction == "low")
    df = df.sort_values(col, ascending=ascending).head(50)

    # Build match results lookup for the result + match name columns
    results_df = _load_match_results()
    if not results_df.empty and "Score" in results_df.columns:
        if fmt == "map":
            res_lookup = results_df[results_df["MapNum"] != "all"].set_index(["MatchID", "MapNum"])
        else:
            res_lookup = results_df[results_df["MapNum"] == "all"].set_index("MatchID")
    else:
        res_lookup = None

    is_kd = (col == "K:D")

    results = []
    for _, row in df.iterrows():
        val = row[col]
        if isinstance(val, float) and val == int(val):
            val = int(val)

        map_name   = ""
        result_str = ""
        match_name = ""

        if fmt == "map":
            map_name = str(row.get("MapName", "")) or ""

        if res_lookup is not None:
            try:
                mid = str(row.get("MatchID", "")).strip()
                org = str(row.get("Org", ""))
                if fmt == "map":
                    mnum = str(row.get("MapNum", "")).strip()
                    res_row = res_lookup.loc[(mid, mnum)]
                else:
                    res_row = res_lookup.loc[mid]
                winner_org = res_row["WinnerOrg"]
                score      = res_row["Score"]
                w_score, l_score = score.split("-")
                result_str = f"W {w_score}-{l_score}" if org == winner_org else f"L {l_score}-{w_score}"
                match_name = str(res_row.get("MatchName", "") or "")
            except Exception:
                pass

        entry = {
            "player":      row.get("Player", ""),
            "org":         row.get("Org", ""),
            "region":      row.get("Region", ""),
            "event":       row.get("_event_label", ""),
            "match_name":  match_name,
            "map_name":    map_name,
            "result":      result_str,
            "value":       round(val, 3) if isinstance(val, float) else val,
            "headshot":    row.get("HeadshotURL", ""),
        }

        if is_kd:
            try:
                entry["kills"]  = int(row["K"])
                entry["deaths"] = int(row["D"])
            except Exception:
                pass

        results.append(entry)

    return jsonify(results)
