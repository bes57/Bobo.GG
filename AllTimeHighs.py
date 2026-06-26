import os
import json
import pandas as pd
import datetime as _dt
from flask import Blueprint, render_template_string, request, jsonify
from MoreTestingMaybeFiles import ALL_EVENTS

highs_bp = Blueprint('highs', __name__)

DATA_DIR    = os.path.join(os.path.dirname(__file__), "data")
MAPS_DIR    = os.path.join(DATA_DIR, "maps")
SERIES_DIR  = os.path.join(DATA_DIR, "series")
HEADSHOTS_FILE = os.path.join(os.path.dirname(__file__), "data", "headshots.json")

_headshot_cache   = {}
_headshots_loaded = False
_event_data         = None
_event_data_mtime   = 0.0
_map_data           = None
_map_data_mtime     = 0.0
_series_data        = None
_series_data_mtime  = 0.0
_match_results      = None
_match_results_mtime = 0.0


def _csv_dir_mtime(folder, top_level_only_year_csvs=False):
    """Latest mtime across all CSVs in the folder (non-recursive). Cheap on
    the ~30 event files. Returns 0.0 if folder is missing or unreadable. Used
    to auto-invalidate the in-memory caches when the live scrape writes new
    rows — so /highs/ picks up Masters London matches as they're scraped
    without needing a Flask restart."""
    if not os.path.isdir(folder):
        return 0.0
    try:
        latest = 0.0
        for entry in os.scandir(folder):
            if not entry.name.endswith('.csv'):
                continue
            # When scanning DATA_DIR, ignore non-event CSVs (e.g. match_results,
            # map_vetos) so we only invalidate the event-level cache on
            # event-CSV changes.
            if top_level_only_year_csvs and ('match_results' in entry.name or 'map_vetos' in entry.name):
                continue
            try:
                latest = max(latest, entry.stat().st_mtime)
            except OSError:
                continue
        return latest
    except OSError:
        return 0.0

STAT_COLS = {
    "VLR Rating":       "R2.0",
    "Kills":            "K",
    "Deaths":           "D",
    "Kill/Death Ratio": "K:D",
    "Assists":          "A",
    # Per-map normalizations — only meaningful for series formats. Computed
    # as raw_value / map_count, where map_count comes from the series'
    # MapNum="all" row in match_results.csv (Score "2-1" → 3 maps).
    "Kills/Map":        "K",
    "Deaths/Map":       "D",
    "Assists/Map":      "A",
}

# Stats that divide the raw column by the series map count. Only valid when
# format is bo3 / bo5 / all_series (not "One Map" — already per-map by def).
PER_MAP_STATS = {"Kills/Map", "Deaths/Map", "Assists/Map"}

MATCH_UNSUPPORTED_STATS = set()

INTERNATIONAL_IDS = {e["id"] for e in ALL_EVENTS if list(e["regions"].keys()) == ["International"]}
# CN-only events feed BenPom (team ratings) but their player stats are hidden
# from this page — user wants CN integration scoped to ratings, not leaderboards.
CN_ONLY_IDS = {e["id"] for e in ALL_EVENTS if list(e["regions"].keys()) == ["CN"]}
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
        if event["id"] in CN_ONLY_IDS:
            continue
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
    global _map_data, _map_data_mtime
    cur = _csv_dir_mtime(MAPS_DIR)
    if _map_data is not None and cur <= _map_data_mtime:
        return _map_data
    _map_data = _load_match_data("maps")
    _map_data_mtime = cur
    return _map_data


def _load_series_data():
    global _series_data, _series_data_mtime
    cur = _csv_dir_mtime(SERIES_DIR)
    if _series_data is not None and cur <= _series_data_mtime:
        return _series_data
    _series_data = _load_match_data("series")
    _series_data_mtime = cur
    return _series_data


def _load_match_results():
    global _match_results, _match_results_mtime
    path = os.path.join(DATA_DIR, "match_results.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        cur = os.path.getmtime(path)
    except OSError:
        cur = 0.0
    if _match_results is not None and not _match_results.empty and cur <= _match_results_mtime:
        return _match_results
    try:
        df = pd.read_csv(path, dtype=str)
        df["MatchID"] = df["MatchID"].str.strip()
        df["MapNum"]  = df["MapNum"].str.strip()
        _match_results = df
        _match_results_mtime = cur
    except Exception:
        return pd.DataFrame()
    return _match_results


PAGE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=800">
<title>All-Time Highs (and Lows)</title>
<link rel="icon" type="image/svg+xml" href="/favicon.svg">
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="/static/base.css">
<style>
  .top-nav { padding:32px 32px 0; position:relative; z-index:1; }
  .home-logo { height:80px; width:auto; display:block; opacity:.85; transition:opacity .2s; }
  .home-logo:hover { opacity:1; }
  .page { position:relative; z-index:1; padding:32px 32px 60px; max-width:1100px; margin:0 auto; }
  header { margin-bottom:32px; text-align:center; }
  header h1 { font-family:'Plus Jakarta Sans',sans-serif; font-size:clamp(1.76rem,4vw,3.08rem); font-weight:800; letter-spacing:-1px; text-align:center; }
  header p { color:#111; font-size:1.02rem; margin-top:8px; font-weight:500; text-align:center; }
  .filters { display:flex; flex-wrap:wrap; gap:12px; margin-bottom:32px; align-items:flex-start; justify-content:center; }
  .filter-group { display:flex; flex-direction:column; gap:4px; }
  .filter-label { font-size:.68rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; color:var(--soft); text-align:center; }
  .filter-select { -webkit-appearance:none; appearance:none; padding:8px 32px 8px 16px; border-radius:99px; border:2px solid #f0ecf4; background:white url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%237a6e7e'/%3E%3C/svg%3E") no-repeat right 12px center; font-family:'DM Sans',sans-serif; font-size:.85rem; font-weight:500; color:var(--ink); cursor:pointer; box-shadow:0 2px 8px #0001; outline:none; transition:border-color .2s; min-width:160px; }
  .filter-select:focus { border-color:var(--lavender); }
  .filter-select:disabled { opacity:.45; cursor:not-allowed; }
  /* Custom dropdown (Context + the cloned single-selects) share one look */
  .ctx-dd, .fdd { position:relative; }
  .ctx-toggle, .fdd-toggle { width:100%; text-align:center; padding-left:32px; }
  .ctx-menu, .fdd-menu { position:absolute; top:calc(100% + 6px); left:0; min-width:100%; background:white; border:2px solid #f0ecf4; border-radius:16px; box-shadow:0 10px 30px #00000022; padding:6px; z-index:50; display:none; }
  .ctx-menu.open, .fdd-menu.open { display:block; }
  .ctx-opt, .fdd-opt { display:flex; align-items:center; gap:9px; padding:8px 12px; border-radius:10px; cursor:pointer; font-size:.85rem; font-weight:500; color:var(--ink); user-select:none; white-space:nowrap; }
  .ctx-opt:hover, .fdd-opt:hover { background:#faf6ff; }
  .ctx-opt input { width:16px; height:16px; accent-color:#8b5cf6; cursor:pointer; flex-shrink:0; }
  .ctx-opt.disabled, .fdd-opt.disabled { opacity:.38; cursor:not-allowed; }
  .ctx-opt.disabled input { cursor:not-allowed; }
  .ctx-sep { height:1px; background:#f0ecf4; margin:5px 8px; }
  .fdd-check { width:16px; height:16px; border-radius:5px; border:2px solid #d8d0e0; flex-shrink:0; position:relative; box-sizing:border-box; }
  .fdd-opt.active .fdd-check { background:#8b5cf6; border-color:#8b5cf6; }
  .fdd-opt.active .fdd-check::after { content:''; position:absolute; left:50%; top:50%; width:4px; height:8px; border:solid white; border-width:0 2px 2px 0; transform:translate(-50%,-58%) rotate(45deg); }
  .results-wrap { background:white; border-radius:20px; overflow:hidden; box-shadow:0 4px 24px #0000000a; }
  table { width:100%; border-collapse:collapse; }
  thead th { padding:13px 18px; text-align:center; font-family:'Plus Jakarta Sans',sans-serif; font-size:0.77rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase; color:var(--soft); border-bottom:2px solid #f0ecf4; }
  thead th.num { text-align:center; }
  tbody tr { transition:background .15s; }
  tbody tr:hover { background:#fdf6f0; }
  tbody td { padding:11px 18px; border-bottom:1px solid #f6f2fa; font-size:.88rem; vertical-align:middle; text-align:center; }
  tbody td.num { text-align:center; font-family:'DM Sans',sans-serif; font-weight:700; font-size:1rem; }
  tbody tr:last-child td { border-bottom:none; }
  .rank-cell { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; color:#ccc; width:44px; text-align:center; }
  .r1{color:#f0b429} .r2{color:#9eaab5} .r3{color:#c07c3a}
  .player-cell { display:flex; align-items:center; justify-content:center; gap:12px; }
  .team-cell { display:flex; align-items:center; justify-content:center; gap:8px; }
  .team-logo { height:22px; width:auto; object-fit:contain; flex-shrink:0; }
  .result-cell { display:grid; grid-template-columns:1fr auto 1fr; align-items:center; column-gap:8px; white-space:nowrap; }
  .result-cell > .result-team:first-child { justify-self:end; }
  .result-cell > .result-team:last-child { justify-self:start; }
  .result-team { display:inline-flex; align-items:center; gap:5px; font-size:.82rem; font-weight:700; }
  .result-logo { height:20px; width:auto; object-fit:contain; flex-shrink:0; }
  .result-score { display:inline-flex; flex-direction:column; align-items:center; gap:1px; line-height:1.1; padding:3px 10px; border-radius:12px; }
  .result-wl { font-family:'DM Sans',sans-serif; font-weight:700; font-size:.72rem; letter-spacing:.05em; opacity:.85; }
  .result-num { display:inline-flex; align-items:center; gap:3px; font-family:'DM Sans',sans-serif; font-weight:700; font-size:.78rem; }
  .result-dash { opacity:.45; font-weight:400; }
  .avatar-ph { border-radius:50%; flex-shrink:0; display:flex; align-items:center; justify-content:center; font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; color:white; font-size:14px; width:40px; height:40px; }
  .avatar-img { width:40px; height:40px; border-radius:50%; object-fit:cover; flex-shrink:0; }
  .badge { display:inline-block; padding:2px 8px; border-radius:99px; font-size:.7rem; font-weight:600; background:#f0ecf4; color:var(--soft); }
  .event-badge { display:inline-block; padding:2px 8px; border-radius:99px; font-size:.7rem; font-weight:500; background:#f4edb8; color:#6a5a1a; }
  .map-badge { display:inline-block; padding:2px 8px; border-radius:99px; font-size:.7rem; font-weight:500; background:#d4f4e8; color:#1a5a3a; margin-left:4px; }
  .empty { text-align:center; padding:40px; color:var(--soft); font-size:.88rem; }
  @keyframes fadeDown{from{opacity:0;transform:translateY(-12px)}to{opacity:1;transform:translateY(0)}}
  .page { animation:fadeDown .5s ease both; }
  .refresh-bar { margin-top:14px; display:flex; align-items:center; gap:12px; justify-content:center; flex-wrap:wrap; }
  .refresh-btn { background:transparent; border:1.5px solid var(--accent); color:var(--accent); padding:6px 14px; border-radius:99px; font-size:.78rem; font-weight:600; cursor:pointer; transition:all .15s; font-family:'DM Sans',sans-serif; }
  .refresh-btn:hover:not(:disabled) { background:var(--accent); color:#fff; }
  .refresh-btn:disabled { opacity:.6; cursor:wait; }
  .refresh-icon { display:inline-block; transition:transform .3s; }
  .refresh-btn.spinning .refresh-icon { animation:rspin 1s linear infinite; }
  @keyframes rspin { to { transform:rotate(360deg); } }
  .refresh-status { font-size:.78rem; color:var(--soft); }
  .refresh-progress-wrap { margin:10px auto 0; width:320px; max-width:90%; height:6px; background:rgba(0,0,0,.07); border-radius:99px; overflow:hidden; opacity:0; transition:opacity .25s; pointer-events:none; }
  .refresh-progress-wrap.active { opacity:1; }
  .refresh-progress-fill { height:100%; width:0%; background:linear-gradient(90deg, var(--accent), #a78bfa); transition:width .35s ease; border-radius:99px; }
  tbody tr.clickable { cursor:pointer; }
  tbody tr.clickable:hover { background:#faf6ff; }
  /* ── Mobile ─────────────────────────────────────────────── */
  @media (max-width:640px){
    .top-nav { padding:20px 16px 0; }
    .page { padding:20px 12px 48px; }
    .filters { gap:10px; margin-bottom:22px; }
    .filter-group { flex:1 1 140px; }
    .filter-select { min-width:0; width:100%; }
    thead th { padding:10px 7px; font-size:.6rem; letter-spacing:.03em; }
    tbody td { padding:9px 7px; font-size:.8rem; }
    tbody td.num { font-size:.9rem; }
    .result-cell { column-gap:5px; }
    .badge, .event-badge, .map-badge { font-size:.62rem; padding:2px 6px; }
  }
</style>
</head>
<body>
<div class="top-nav">
  <a href="/"><img src="/logo.svg" alt="Home" class="home-logo"></a>
</div>
<div class="page">
  <header>
    <h1>All-Time Highs (and Lows)</h1>
    <p>Individual map/match records across all VCT franchised events, 2023&ndash;{{ latest_event_label }}</p>
    <div class="refresh-bar">
      <button class="refresh-btn" id="refreshBtn" onclick="triggerRefresh()">
        <span class="refresh-icon">&#x21bb;</span> <span class="refresh-label">Check for new matches</span>
      </button>
      <span class="refresh-status" id="refreshStatus"></span>
    </div>
    <div class="refresh-progress-wrap" id="refreshProgressWrap">
      <div class="refresh-progress-fill" id="refreshProgressFill"></div>
    </div>
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
        <option value="Kill/Death Ratio">Kill/Death Ratio</option>
        <option value="Kills">Kills</option>
        <option value="Deaths">Deaths</option>
        <option value="Assists">Assists</option>
        <option value="Kills/Map">Kills/Map</option>
        <option value="Deaths/Map">Deaths/Map</option>
        <option value="Assists/Map">Assists/Map</option>
      </select>
    </div>
    <div class="filter-group">
      <span class="filter-label">Format</span>
      <select class="filter-select" id="f-format" onchange="onFormatChange()">
        <option value="map">One Map</option>
        <option value="all_series">One Match (Bo3 + Bo5)</option>
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
      <div class="ctx-dd" id="ctxDD">
        <button type="button" class="filter-select ctx-toggle" id="ctxToggle" onclick="toggleCtxMenu(event)"><span id="ctxLabel">All Events</span></button>
        <div class="ctx-menu" id="ctxMenu">
          <label class="ctx-opt"><input type="checkbox" class="ctx-cb" value="intl"><span>At an International</span></label>
          <label class="ctx-opt"><input type="checkbox" class="ctx-cb" value="regional"><span>Regional Only</span></label>
          <div class="ctx-sep"></div>
          <label class="ctx-opt"><input type="checkbox" class="ctx-cb" value="win"><span>In a Win</span></label>
          <label class="ctx-opt"><input type="checkbox" class="ctx-cb" value="loss"><span>In a Loss</span></label>
        </div>
      </div>
    </div>
  </div>

  <div class="results-wrap">
    <table>
      <thead>
        <tr>
          <th style="width:44px">#</th>
          <th>Player</th>
          <th>Team</th>
          <th>Event</th>
          <th id="map-col-header" style="display:none">Map</th>
          <th>Result</th>
          <th class="num" id="stat-col-header">Value</th>
        </tr>
      </thead>
      <tbody id="results-body">
        <tr><td colspan="7" class="empty">Loading&hellip;</td></tr>
      </tbody>
    </table>
  </div>
</div>

<script>
const MATCH_UNSUPPORTED = new Set(["# of Clutches"]);
const PER_MAP_STATS = new Set(["Kills/Map", "Deaths/Map", "Assists/Map"]);

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
  const isSeriesFmt = (fmt === 'bo3' || fmt === 'bo5' || fmt === 'all_series');
  // /Map stats only make sense for series formats.
  for (const opt of document.getElementById('f-stat').options) {
    opt.disabled = MATCH_UNSUPPORTED.has(opt.value) ||
                   (PER_MAP_STATS.has(opt.value) && !isSeriesFmt);
  }
  if (MATCH_UNSUPPORTED.has(stat) || (PER_MAP_STATS.has(stat) && !isSeriesFmt)) {
    document.getElementById('f-stat').value = 'Kills';
  }
  document.getElementById('map-col-header').style.display = fmt === 'map' ? '' : 'none';
  fetchResults();
}

function onStatChange() {
  const stat   = document.getElementById('f-stat').value;
  const fmtSel = document.getElementById('f-format');
  const isPerMap = PER_MAP_STATS.has(stat);
  // /Map stats are per-series-map averages, so "One Map" makes no sense.
  // Disable the option and bump the format to Bo3 if it's currently selected.
  for (const opt of fmtSel.options) {
    if (opt.value === 'map') opt.disabled = isPerMap;
  }
  if (isPerMap && fmtSel.value === 'map') {
    fmtSel.value = 'bo3';
    document.getElementById('map-col-header').style.display = 'none';
  }
  fetchResults();
}

// Context multi-select: two mutually-exclusive pairs (intl/regional, win/loss).
// Checking one disables its contradiction so impossible combos can't be set.
const CTX_EXCLUDE = { intl:'regional', regional:'intl', win:'loss', loss:'win' };
const CTX_LABELS  = { intl:'At an International', regional:'Regional Only', win:'In a Win', loss:'In a Loss' };

function closeAllMenus(except) {
  document.querySelectorAll('.ctx-menu.open, .fdd-menu.open').forEach(m => { if (m !== except) m.classList.remove('open'); });
}
function toggleCtxMenu(ev) {
  ev.stopPropagation();
  const m = document.getElementById('ctxMenu');
  closeAllMenus(m);
  m.classList.toggle('open');
}
function getContextValue() {
  const sel = Array.from(document.querySelectorAll('.ctx-cb:checked')).map(c => c.value);
  return sel.length ? sel.join(',') : 'all';
}
function updateCtxState() {
  const checked = new Set(Array.from(document.querySelectorAll('.ctx-cb:checked')).map(c => c.value));
  document.querySelectorAll('.ctx-cb').forEach(cb => {
    const opp = CTX_EXCLUDE[cb.value];
    const blocked = opp && checked.has(opp);
    cb.disabled = !!blocked;
    cb.closest('.ctx-opt').classList.toggle('disabled', !!blocked);
  });
  const labels = Array.from(checked).map(v => CTX_LABELS[v]);
  document.getElementById('ctxLabel').textContent = labels.length ? labels.join(', ') : 'All Events';
}
document.querySelectorAll('.ctx-cb').forEach(cb => {
  cb.addEventListener('change', () => { updateCtxState(); fetchResults(); });
});
document.addEventListener('click', (e) => {
  if (!e.target.closest('.ctx-dd') && !e.target.closest('.fdd')) closeAllMenus(null);
});

// Clone each native single-select into a custom dropdown styled like Context.
// The native <select> stays as the source of truth (hidden); the custom menu
// just mirrors it and dispatches `change` so existing onchange handlers fire.
function buildCustomDropdowns() {
  document.querySelectorAll('select.filter-select').forEach(sel => {
    sel.style.display = 'none';
    const wrap = document.createElement('div');
    wrap.className = 'fdd';
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'filter-select fdd-toggle';
    btn.innerHTML = '<span class="fdd-label"></span>';
    const menu = document.createElement('div');
    menu.className = 'fdd-menu';
    Array.from(sel.options).forEach(opt => {
      const item = document.createElement('div');
      item.className = 'fdd-opt';
      item.dataset.value = opt.value;
      item.innerHTML = '<span class="fdd-check"></span><span>' + esc(opt.textContent) + '</span>';
      item.addEventListener('click', () => {
        if (item.classList.contains('disabled')) return;
        sel.value = opt.value;
        sel.dispatchEvent(new Event('change'));
        menu.classList.remove('open');
        syncCustomDropdowns();
      });
      menu.appendChild(item);
    });
    btn.addEventListener('click', (e) => { e.stopPropagation(); closeAllMenus(menu); menu.classList.toggle('open'); });
    wrap.appendChild(btn);
    wrap.appendChild(menu);
    sel.parentNode.insertBefore(wrap, sel.nextSibling);
    sel._fdd = { btn, menu };
  });
  syncCustomDropdowns();
}
function syncCustomDropdowns() {
  document.querySelectorAll('select.filter-select').forEach(sel => {
    if (!sel._fdd) return;
    sel._fdd.btn.querySelector('.fdd-label').textContent = sel.options[sel.selectedIndex].textContent;
    sel._fdd.menu.querySelectorAll('.fdd-opt').forEach(item => {
      const opt = Array.from(sel.options).find(o => o.value === item.dataset.value);
      item.classList.toggle('active', sel.value === item.dataset.value);
      item.classList.toggle('disabled', !!(opt && opt.disabled));
    });
  });
}

function fetchResults() {
  const direction = document.getElementById('f-direction').value;
  const stat      = document.getElementById('f-stat').value;
  const fmt       = document.getElementById('f-format').value;
  const year      = document.getElementById('f-year').value;
  const context   = getContextValue();

  document.getElementById('stat-col-header').textContent = stat;
  document.getElementById('results-body').innerHTML = '<tr><td colspan="7" class="empty">Loading&hellip;</td></tr>';

  fetch(`/highs/api/results?direction=${encodeURIComponent(direction)}&stat=${encodeURIComponent(stat)}&format=${encodeURIComponent(fmt)}&year=${encodeURIComponent(year)}&context=${encodeURIComponent(context)}`)
    .then(r => r.json())
    .then(data => {
      if (!data.length) {
        document.getElementById('results-body').innerHTML = '<tr><td colspan="7" class="empty">No data found for this combination.</td></tr>';
        return;
      }
      const showMap = document.getElementById('f-format').value === 'map';
      document.getElementById('map-col-header').style.display = showMap ? '' : 'none';
      document.getElementById('results-body').innerHTML = data.map((row, i) => {
        const avatar = row.headshot
          ? `<img class="avatar-img" src="${esc(row.headshot)}" onerror="this.replaceWith(Object.assign(document.createElement('div'),{className:'avatar-ph',style:'background:'+${JSON.stringify(avatarColor(row.player))},textContent:${JSON.stringify((row.player||'').slice(0,2).toUpperCase())}}))">`
          : `<div class="avatar-ph" style="background:${avatarColor(row.player)}">${esc((row.player||'').slice(0,2).toUpperCase())}</div>`;
        const mapCell = showMap ? `<td>${esc(row.map_name||'')}</td>` : '';
        const curStat = document.getElementById('f-stat').value;
        const isKD = curStat === 'Kill/Death Ratio';
        // VLR Rating is conventionally shown to the hundredths (1.00, 1.45).
        const valStr = (curStat === 'VLR Rating' && typeof row.value === 'number')
          ? row.value.toFixed(2)
          : String(row.value);
        const valDisplay = (isKD && row.kills != null && row.deaths != null)
          ? `${esc(valStr)} <span style="font-size:.75rem;font-weight:400;color:var(--soft)">(${row.kills}/${row.deaths})</span>`
          : esc(valStr);
        let resultCell = '<td></td>';
        if (row.result) {
          const won = (row.won != null) ? row.won : row.result.startsWith('W');
          const bg = won ? '#d4f4e8' : '#fde8e8';
          const fg = won ? '#1a5a3a' : '#7a1a1a';
          const wl = won ? 'W' : 'L';
          if (row.opp) {
            const logo = (o) => `<img class="result-logo" src="/logos/${esc(o)}.png" alt="${esc(o)}" onerror="this.style.display='none'">`;
            resultCell = `<td><div class="result-cell">`
              + `<span class="result-team">${logo(row.org)}<b>${esc(row.org)}</b></span>`
              + `<span class="result-score" style="background:${bg};color:${fg}"><span class="result-wl">${wl}</span><span class="result-num">${esc(row.team_score)}<span class="result-dash">&ndash;</span>${esc(row.opp_score)}</span></span>`
              + `<span class="result-team"><b>${esc(row.opp)}</b>${logo(row.opp)}</span>`
              + `</div></td>`;
          } else {
            resultCell = `<td><span style="display:inline-block;padding:2px 8px;border-radius:99px;font-size:.7rem;font-weight:600;background:${bg};color:${fg}">${esc(row.result)}</span></td>`;
          }
        }
        // Inline `onclick="window.open(JSON.stringify(url)...)"` breaks the
        // attribute parser — the double quote inside JSON.stringify ends the
        // attribute early. Use a data attribute + delegated handler instead.
        const clickAttrs = row.vlr_url ? ` class="clickable" data-vlr="${esc(row.vlr_url)}"` : '';
        return `<tr${clickAttrs}>
          <td class="rank-cell ${rankClass(i)}">${i+1}</td>
          <td><div class="player-cell">${avatar}<span>${esc(row.player)}</span></div></td>
          <td><div class="team-cell"><img class="team-logo" src="/logos/${esc(row.org||'')}.png" alt="${esc(row.org||'')}" onerror="this.style.display='none'"><span>${esc(row.org||'')}</span></div></td>
          <td><span class="event-badge">${esc(row.event)}</span>${row.match_name ? `<div style="font-size:.7rem;color:var(--soft);margin-top:3px">${esc(row.match_name)}</div>` : ''}</td>
          ${mapCell}
          ${resultCell}
          <td class="num">${valDisplay}</td>
        </tr>`;
      }).join('');
    })
    .catch((err) => {
      console.error('[highs] fetchResults failed:', err);
      const msg = err && err.message ? err.message : String(err || 'unknown');
      document.getElementById('results-body').innerHTML = `<tr><td colspan="7" class="empty">Failed to load results: ${esc(msg)}</td></tr>`;
    });
}

// Delegated click handler — every row with data-vlr opens the match page in
// a new tab. Single listener on the tbody so it survives re-renders.
document.getElementById('results-body').addEventListener('click', (ev) => {
  const tr = ev.target.closest('tr.clickable');
  if (!tr) return;
  const url = tr.dataset.vlr;
  if (url) window.open(url, '_blank', 'noopener');
});

// ── Refresh button: trigger the Modern Hub's live-scrape pipeline, poll
// progress, then re-fetch results once it's done. Reuses /mapelo/modern/
// refresh + /progress so we don't duplicate the scrape infrastructure;
// the AllTimeHighs cache auto-invalidates via mtime on the next API call.
let _refreshing = false;
async function triggerRefresh() {
  if (_refreshing) return;
  _refreshing = true;
  const btn      = document.getElementById('refreshBtn');
  const lbl      = btn.querySelector('.refresh-label');
  const status   = document.getElementById('refreshStatus');
  const barWrap  = document.getElementById('refreshProgressWrap');
  const barFill  = document.getElementById('refreshProgressFill');
  const origLbl  = lbl.textContent;
  btn.classList.add('spinning');
  btn.disabled = true;
  lbl.textContent = 'Refreshing…';
  status.textContent = '';
  // Reveal the bar at 0% so it's visible while we wait for the first poll.
  barFill.style.width = '0%';
  barWrap.classList.add('active');
  let lastPct = 0;

  try {
    await fetch('/mapelo/modern/refresh', { cache: 'no-store' });
    // Poll progress every 1.5s, watch for terminal phase.
    let phase = '';
    let safety = 80; // ~2 minutes max
    while (phase !== 'done' && phase !== 'error' && safety-- > 0) {
      await new Promise(r => setTimeout(r, 1500));
      try {
        const r = await fetch('/mapelo/modern/progress', { cache: 'no-store' });
        const wrap = await r.json();
        // /modern/progress wraps the actual progress object:
        // { progress: {pct, phase, message, ...}, stderr_tail, build_running }
        const p = (wrap && wrap.progress) ? wrap.progress : {};
        if (p.message) status.textContent = p.message;
        if (typeof p.pct === 'number') {
          // Monotonic — don't let a stale poll yank the bar backwards.
          const next = Math.max(lastPct, Math.min(99, p.pct));
          lastPct = next;
          barFill.style.width = next + '%';
        }
        if (p.phase) phase = p.phase;
      } catch (e) { /* keep polling */ }
    }
    if (phase === 'error') {
      status.textContent = 'Refresh failed — try again.';
      // Leave the bar visible at last position so it's clear something stalled.
    } else {
      // Finish the bar before re-fetching so the user sees it complete.
      barFill.style.width = '100%';
      status.textContent = 'Updated.';
      await fetchResults();
      setTimeout(() => {
        status.textContent = '';
        barWrap.classList.remove('active');
        // Reset width after fade-out so the next refresh starts clean.
        setTimeout(() => { barFill.style.width = '0%'; }, 350);
      }, 1200);
    }
  } catch (e) {
    status.textContent = 'Refresh failed: ' + e.message;
  } finally {
    btn.classList.remove('spinning');
    btn.disabled = false;
    lbl.textContent = origLbl;
    _refreshing = false;
  }
}

// Deep-link support: /highs/?direction=&stat=&format=&year=&context= preselects
// the filters (used by the home page's "Recent VCT Records" cards so a click
// lands on that exact all-time leaderboard).
function applyUrlParams() {
  const q = new URLSearchParams(location.search);
  if (!['stat','format','direction','context','year'].some(k => q.has(k))) return;
  const setSel = (id,v) => { const el=document.getElementById(id); if(el&&v!=null) el.value=v; };
  setSel('f-direction', q.get('direction'));
  setSel('f-format',    q.get('format'));
  setSel('f-stat',      q.get('stat'));
  setSel('f-year',      q.get('year'));
  const ctx = (q.get('context')||'').split(',').map(s=>s.trim()).filter(Boolean);
  document.querySelectorAll('.ctx-cb').forEach(cb => { cb.checked = ctx.indexOf(cb.value) >= 0; });
  // Reconcile the format/stat option enabled-state with the chosen values.
  const fmt = document.getElementById('f-format').value;
  const isSeries = (fmt==='bo3'||fmt==='bo5'||fmt==='all_series');
  for (const opt of document.getElementById('f-stat').options)
    opt.disabled = MATCH_UNSUPPORTED.has(opt.value) || (PER_MAP_STATS.has(opt.value) && !isSeries);
  const stat = document.getElementById('f-stat').value;
  for (const opt of document.getElementById('f-format').options)
    if (opt.value==='map') opt.disabled = PER_MAP_STATS.has(stat);
  document.getElementById('map-col-header').style.display = (fmt==='map') ? '' : 'none';
}

applyUrlParams();
buildCustomDropdowns();
updateCtxState();
fetchResults();
</script>
<footer style="text-align:center;padding:24px 16px 28px;color:#7a6e7e;font-size:.75rem;font-weight:300;line-height:1.55;font-family:'DM Sans',sans-serif;">
  Data sourced from VLR.gg
  <div style="margin-top:8px;">Like my work? Tips are appreciated! <a href="https://ko-fi.com/bobovct" target="_blank" rel="noopener" style="color:#7a6e7e;text-decoration:underline;">ko-fi.com/bobovct</a></div>
</footer>
</body>
</html>
"""


def _latest_event_label():
    """Most recent non-CN-only event whose maps CSV exists. ALL_EVENTS is
    most-recent-first, so the first match wins. We check the maps CSV (not
    the top-level event CSV) because the live scrape pipeline only writes
    maps/series/match_results — never the top-level event CSV (that's
    ScrapeAllEvents only). Without this, the subtitle stays stuck at the
    last full-scrape event even after live data has been pulled."""
    for event in ALL_EVENTS:
        if event["id"] in CN_ONLY_IDS:
            continue
        if os.path.exists(os.path.join(MAPS_DIR, f"{event['id']}.csv")):
            return event["label"]
    return "2023"


@highs_bp.route("/")
def index():
    return render_template_string(PAGE_HTML, latest_event_label=_latest_event_label())


@highs_bp.route("/api/results")
def api_results():
    direction = request.args.get("direction", "high")
    stat_name = request.args.get("stat", "VLR Rating")
    fmt       = request.args.get("format", "event")
    year      = request.args.get("year", "all")
    context   = request.args.get("context", "all")
    df, col, is_kd = _rank_df(direction, stat_name, fmt, year, context)
    if df is None or df.empty:
        return jsonify([])
    return jsonify(_format_entries(df, fmt, col, is_kd))


def _rank_df(direction, stat_name, fmt, year, context):
    """Shared ranking core for one leaderboard cell. Returns
    (df_top50, value_col, is_kd) sorted best-first with a reset index (so row
    position == rank-1), or (None, None, None) when invalid/empty."""
    col = STAT_COLS.get(stat_name)
    if not col:
        return None, None, None

    is_per_map = stat_name in PER_MAP_STATS
    if is_per_map and fmt not in ("bo3", "bo5", "all_series"):
        return None, None, None

    if fmt == "map":
        df = _load_map_data()
    elif fmt in ("bo3", "bo5", "all_series"):
        df = _load_series_data()
        mr_fmt = _load_match_results()
        if df is not None and not df.empty and not mr_fmt.empty and "Score" in mr_fmt.columns:
            ss = mr_fmt[mr_fmt["MapNum"] == "all"][["MatchID", "Score"]].copy()
            def _max_score(s):
                try:
                    a, b = str(s).split("-")
                    return max(int(a), int(b))
                except Exception:
                    return None
            ss["MaxScore"] = ss["Score"].apply(_max_score)
            ss = ss.dropna(subset=["MaxScore"])
            ss["MatchID"] = ss["MatchID"].astype(str).str.strip()
            if fmt == "bo3":
                keep = set(ss.loc[ss["MaxScore"] == 2, "MatchID"])
            elif fmt == "bo5":
                keep = set(ss.loc[ss["MaxScore"] == 3, "MatchID"])
            else:
                keep = set(ss.loc[ss["MaxScore"].isin([2, 3]), "MatchID"])
            df = df[df["MatchID"].astype(str).str.strip().isin(keep)]
    else:
        df = _load_event_data()

    if df is None or df.empty or col not in df.columns:
        return None, None, None

    if "MatchID" in df.columns:
        mr_sm = _load_match_results()
        if not mr_sm.empty and "MatchName" in mr_sm.columns:
            show_ids = set(
                mr_sm.loc[mr_sm["MatchName"].astype(str).str.startswith("Showmatch", na=False), "MatchID"]
                     .astype(str).str.strip().tolist()
            )
            if show_ids:
                df = df[~df["MatchID"].astype(str).str.strip().isin(show_ids)]

    if "MatchID" in df.columns and "Org" in df.columns:
        pid = "ProfileURL" if "ProfileURL" in df.columns else "Player"
        if pid in df.columns:
            df = df.copy()
            df["MatchID"] = df["MatchID"].astype(str).str.strip()
            roster = df.groupby(["MatchID", "Org"])[pid].nunique()
            bad = set(roster[roster > 5].index)
            if bad:
                keep = [(m, o) not in bad for m, o in zip(df["MatchID"], df["Org"])]
                df = df[pd.Series(keep, index=df.index)]

    if is_per_map:
        mr = _load_match_results()
        if mr.empty:
            return None, None, None
        series_scores = mr[mr["MapNum"] == "all"][["MatchID", "Score"]].copy()
        def _map_count(score):
            try:
                a, b = str(score).split("-")
                return int(a) + int(b)
            except Exception:
                return None
        series_scores["MapCount"] = series_scores["Score"].apply(_map_count)
        series_scores = series_scores.dropna(subset=["MapCount"])
        series_scores["MatchID"] = series_scores["MatchID"].astype(str).str.strip()
        df = df.copy()
        df["MatchID"] = df["MatchID"].astype(str).str.strip()
        df = df.merge(series_scores[["MatchID", "MapCount"]], on="MatchID", how="left")
        df = df.dropna(subset=["MapCount"])
        derived_col = f"__{col}_per_map"
        df[derived_col] = df[col] / df["MapCount"]
        col = derived_col

    if year != "all":
        df = df[df["_year"] == int(year)]

    ctx = {t for t in str(context).split(",") if t and t != "all"}
    if "intl" in ctx:
        df = df[df["_intl"] == True]
    elif "regional" in ctx:
        df = df[df["_intl"] == False]
    if ("win" in ctx) or ("loss" in ctx):
        results_df = _load_match_results()
        if results_df.empty:
            return None, None, None
        df = df.copy()
        df["MatchID"] = df["MatchID"].astype(str).str.strip()
        if fmt == "map":
            df["MapNum"] = df["MapNum"].astype(str).str.strip()
            lookup = results_df[results_df["MapNum"] != "all"][["MatchID", "MapNum", "WinnerOrg"]]
            merged = df.merge(lookup, on=["MatchID", "MapNum"], how="left")
        else:
            lookup = results_df[results_df["MapNum"] == "all"][["MatchID", "WinnerOrg"]]
            merged = df.merge(lookup, on="MatchID", how="left")
        if "win" in ctx:
            df = merged[merged["WinnerOrg"] == merged["Org"]].drop(columns=["WinnerOrg"])
        else:
            df = merged[(merged["WinnerOrg"].notna()) & (merged["WinnerOrg"] != merged["Org"])].drop(columns=["WinnerOrg"])

    df = df.dropna(subset=[col])
    if df.empty:
        return None, None, None
    ascending = (direction == "low")
    df = df.sort_values(col, ascending=ascending).head(50).reset_index(drop=True)
    return df, col, (col == "K:D")


_FMT_LOOKUP_CACHE = {}   # group -> (mtime_key, res_lookup, opp_map)


def _fmt_lookups(fmt):
    """Per-format result lookup + opponent map. These depend only on the format
    GROUP (map vs series) and the underlying CSVs — not on the stat/context — so
    cache them. The home-page records scan calls _format_entries ~30x and was
    rebuilding a full-dataset groupby each time (the bulk of its cold-build cost)."""
    grp = "map" if fmt == "map" else "series"
    try:
        _mr_mt = os.path.getmtime(os.path.join(DATA_DIR, "match_results.csv"))
    except OSError:
        _mr_mt = 0.0
    key = (_csv_dir_mtime(MAPS_DIR if grp == "map" else SERIES_DIR), _mr_mt)
    cached = _FMT_LOOKUP_CACHE.get(grp)
    if cached and cached[0] == key:
        return cached[1], cached[2]

    results_df = _load_match_results()
    if not results_df.empty and "Score" in results_df.columns:
        if grp == "map":
            res_lookup = results_df[results_df["MapNum"] != "all"].set_index(["MatchID", "MapNum"])
        else:
            res_lookup = results_df[results_df["MapNum"] == "all"].set_index("MatchID")
    else:
        res_lookup = None

    opp_map = {}
    if grp == "map":
        _full = _load_map_data()
        if _full is not None and not _full.empty and {"MatchID", "MapNum", "Org"} <= set(_full.columns):
            _g = _full[["MatchID", "MapNum", "Org"]].copy()
            _g["MatchID"] = _g["MatchID"].astype(str).str.strip()
            _g["MapNum"]  = _g["MapNum"].astype(str).str.strip()
            for (mid_k, mnum_k), grp_rows in _g.groupby(["MatchID", "MapNum"]):
                opp_map[(mid_k, mnum_k)] = list(pd.unique(grp_rows["Org"].dropna()))
    else:
        _full = _load_series_data()
        if _full is not None and not _full.empty and {"MatchID", "Org"} <= set(_full.columns):
            _g = _full[["MatchID", "Org"]].copy()
            _g["MatchID"] = _g["MatchID"].astype(str).str.strip()
            for mid_k, grp_rows in _g.groupby("MatchID"):
                opp_map[mid_k] = list(pd.unique(grp_rows["Org"].dropna()))

    _FMT_LOOKUP_CACHE[grp] = (key, res_lookup, opp_map)
    return res_lookup, opp_map


def _format_entries(df, fmt, col, is_kd):
    """Turn a ranked df (from _rank_df) into the list of result dicts the page
    and the home-page records preview both consume."""
    res_lookup, opp_map = _fmt_lookups(fmt)

    results = []
    for _, row in df.iterrows():
        val = row[col]
        if isinstance(val, float) and val == int(val):
            val = int(val)

        map_name   = ""
        result_str = ""
        match_name = ""
        opp        = ""
        team_score = ""
        opp_score  = ""
        won        = None

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
                won        = (org == winner_org)
                result_str = f"W {w_score}-{l_score}" if won else f"L {l_score}-{w_score}"
                match_name = str(res_row.get("MatchName", "") or "")
                team_score, opp_score = (w_score, l_score) if won else (l_score, w_score)
                orgs = opp_map.get((mid, mnum)) if fmt == "map" else opp_map.get(mid)
                if orgs:
                    opp = next((o for o in orgs if o and o != org), "")
            except Exception:
                pass

        vlr_url = ""
        mid_str = str(row.get("MatchID", "")).strip()
        if mid_str and fmt != "event":
            if fmt == "map":
                mnum_link = str(row.get("MapNum", "")).strip()
                vlr_url = f"https://www.vlr.gg/{mid_str}/?game={mnum_link}&tab=overview" if mnum_link else f"https://www.vlr.gg/{mid_str}/"
            else:
                vlr_url = f"https://www.vlr.gg/{mid_str}/"

        def _s(v):
            try:
                if pd.isna(v):
                    return ""
            except Exception:
                pass
            return v if v is not None else ""

        entry = {
            "player":      _s(row.get("Player", "")),
            "profile":     _s(row.get("ProfileURL", "")),
            "org":         _s(row.get("Org", "")),
            "region":      _s(row.get("Region", "")),
            "event":       _s(row.get("_event_label", "")),
            "event_id":    _s(row.get("_event_id", "")),
            "match_name":  match_name,
            "map_name":    map_name,
            "result":      result_str,
            "opp":         _s(opp),
            "team_score":  _s(team_score),
            "opp_score":   _s(opp_score),
            "won":         won,
            "value":       round(val, 3) if isinstance(val, float) else val,
            "headshot":    _s(row.get("HeadshotURL", "")),
            "vlr_url":     vlr_url,
        }

        if is_kd:
            try:
                entry["kills"]  = int(row["K"])
                entry["deaths"] = int(row["D"])
            except Exception:
                pass

        results.append(entry)

    return results


# ── Recent records preview (home page) ───────────────────────────────────────
# Scan a curated slice of the leaderboard space (good-stat HIGHS across the
# meaningful format/context combinations) and surface performances from the most
# recent event that have cracked an all-time top-50.
# NOTE: only formats the /highs/ page actually exposes — so every surfaced record
# deep-links to a leaderboard the user can open and verify (there is no "event"
# format on that page, so event totals are intentionally excluded).
_REC_STATS    = ["VLR Rating", "Kills", "Kill/Death Ratio", "Assists", "Kills/Map"]
_REC_FORMATS  = ["map", "bo3", "bo5"]
_REC_CONTEXTS = ["all", "intl", "regional", "win"]
_REC_STAT_WORD = {
    "VLR Rating":       ("rating",    "highest"),
    "Kills":            ("kills",     "most"),
    "Kill/Death Ratio": ("K/D",       "highest"),
    "Assists":          ("assists",   "most"),
    "Kills/Map":        ("kills/map", "most"),
}
_REC_FMT_NOUN = {"event": "event", "map": "map", "bo3": "Bo3", "bo5": "Bo5", "all_series": "series"}
_RECENT_RECORDS_CACHE = {"data": None, "key": None}
_RECENT_RECORDS_DISK  = os.path.join(DATA_DIR, "recent_records.json")


def _recent_event_ids(window_days=21):
    """Event ids in the most recent competitive cluster (the latest event with
    data, plus anything ending within `window_days` of it) — these are the
    'recent' performances whose top-50 entries we show off."""
    dated = []
    for e in ALL_EVENTS:
        if e["id"] in CN_ONLY_IDS:
            continue
        if not os.path.exists(os.path.join(MAPS_DIR, f"{e['id']}.csv")):
            continue
        raw = e.get("end") or e.get("start") or ""
        try:
            d = _dt.date.fromisoformat(str(raw)[:10])
        except Exception:
            continue
        dated.append((d, e["id"]))
    if not dated:
        return set()
    latest = max(d for d, _ in dated)
    cutoff = latest - _dt.timedelta(days=window_days)
    return {eid for d, eid in dated if d >= cutoff}


def _ordinal(n):
    if 10 <= (n % 100) <= 20:
        suf = "th"
    else:
        suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def _rec_scope(fmt, context):
    ctx = {t for t in str(context).split(",") if t and t != "all"}
    noun   = _REC_FMT_NOUN.get(fmt, fmt)
    adj    = "international " if "intl" in ctx else ("domestic " if "regional" in ctx else "")
    suffix = " win" if "win" in ctx else (" loss" if "loss" in ctx else "")
    return (adj + noun + suffix).strip()


def build_recent_records(limit=8):
    # Cache key spans every data file the records derive from, so the moment the
    # live scrape writes new maps/series/event rows OR updates match_results, the
    # next page load rebuilds with the fresh data (and picks up a new latest event
    # automatically via _recent_event_ids).
    try:
        _mr_mtime = os.path.getmtime(os.path.join(DATA_DIR, "match_results.csv"))
    except OSError:
        _mr_mtime = 0.0
    key = (_csv_dir_mtime(MAPS_DIR), _csv_dir_mtime(SERIES_DIR),
           _csv_dir_mtime(DATA_DIR, True), _mr_mtime)
    if _RECENT_RECORDS_CACHE["data"] is not None and _RECENT_RECORDS_CACHE["key"] == key:
        return _RECENT_RECORDS_CACHE["data"]

    # Disk-backed cache: a COLD process (server start / new gunicorn worker) reads
    # the precomputed JSON instead of paying the ~1.3s rebuild. Keyed on the same
    # data-file mtimes, so it self-invalidates the instant the scrape writes new
    # data, and it's shared across workers via the filesystem.
    try:
        with open(_RECENT_RECORDS_DISK) as f:
            disk = json.load(f)
        if disk.get("key") == list(key) and disk.get("limit", 0) >= limit:
            out = disk["data"][:limit]
            _RECENT_RECORDS_CACHE["data"] = out
            _RECENT_RECORDS_CACHE["key"]  = key
            return out
    except (OSError, ValueError, KeyError, TypeError):
        pass

    recent = _recent_event_ids()
    best = {}   # (profile, matchid, mapnum, stat, fmt) -> record (best rank framing)
    if recent:
        for stat in _REC_STATS:
            word, verb = _REC_STAT_WORD[stat][0], _REC_STAT_WORD[stat][1]
            for fmt in _REC_FORMATS:
                if stat in PER_MAP_STATS and fmt not in ("bo3", "bo5"):
                    continue
                for context in _REC_CONTEXTS:
                    try:
                        df, col, is_kd = _rank_df("high", stat, fmt, "all", context)
                    except Exception:
                        df = None
                    if df is None or df.empty or "_event_id" not in df.columns:
                        continue
                    evs = df["_event_id"].tolist()
                    idxs = [i for i, e in enumerate(evs) if e in recent]
                    if not idxs:
                        continue
                    sub = df.iloc[idxs]
                    entries = _format_entries(sub, fmt, col, is_kd)
                    for j, pos in enumerate(idxs):
                        row  = df.iloc[pos]
                        ent  = entries[j]
                        rank = pos + 1
                        prof = str(row.get("ProfileURL", "") or row.get("Player", ""))
                        mid  = str(row.get("MatchID", "")).strip()
                        mnum = str(row.get("MapNum", "")).strip() if fmt == "map" else ""
                        rkey = (prof, mid, mnum, stat, fmt)
                        prev = best.get(rkey)
                        if prev is None or rank < prev["rank"]:
                            scope = _rec_scope(fmt, context)
                            art = "an" if scope[:1].lower() in "aeiou" else "a"
                            best[rkey] = dict(ent,
                                rank=rank, stat=stat, stat_label=stat, fmt=fmt,
                                context=context, direction="high",
                                scope=scope, matchid=mid, mapnum=mnum,
                                desc=f"{_ordinal(rank)}-{verb} {word} in {art} {scope} of all time")

    # Collapse to one headline per actual performance (best stat-rank), then
    # order by rank and cap repeats from a single player for variety.
    by_perf = {}
    for r in best.values():
        pk = (r.get("player", ""), r.get("matchid", ""), r.get("mapnum", ""), r.get("fmt", ""))
        if pk not in by_perf or r["rank"] < by_perf[pk]["rank"]:
            by_perf[pk] = r

    ordered = sorted(by_perf.values(), key=lambda r: (r["rank"], r.get("player", "")))
    out, per_player = [], {}
    for r in ordered:
        p = r.get("player", "")
        if per_player.get(p, 0) >= 2:
            continue
        per_player[p] = per_player.get(p, 0) + 1
        out.append(r)
        if len(out) >= limit:
            break

    _RECENT_RECORDS_CACHE["data"] = out
    _RECENT_RECORDS_CACHE["key"]  = key
    # Persist for cold processes / other workers (atomic write).
    try:
        tmp = _RECENT_RECORDS_DISK + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"key": list(key), "limit": limit, "data": out}, f)
        os.replace(tmp, _RECENT_RECORDS_DISK)
    except OSError:
        pass
    return out
