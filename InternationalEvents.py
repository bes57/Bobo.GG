import os
import json
import pandas as pd
from flask import Blueprint, render_template_string

intl_bp = Blueprint('intl', __name__)

JUNK_ORGS = {'tarik','Team','INTL','THAi','fugu','jisou','sergioferra','yjj',
             'heart bus','karsaj','FRTTT','NaN','nan'}

INTL_EVENT_META = [
    ('2023_lock_in',          'LOCK//IN São Paulo',  '2023', 'February 2023'),
    ('2023_masters_tokyo',    'Masters Tokyo',        '2023', 'June 2023'),
    ('2023_champions',        'Champions Los Angeles','2023', 'August 2023'),
    ('2024_masters_madrid',   'Masters Madrid',       '2024', 'February 2024'),
    ('2024_champions',        'Champions Seoul',      '2024', 'August 2024'),
    ('2025_masters_bangkok',  'Masters Bangkok',      '2025', 'February 2025'),
    ('2025_masters_toronto',  'Masters Toronto',      '2025', 'May 2025'),
    ('2025_champions',        'Champions Bangkok',    '2025', 'August 2025'),
    ('2026_masters_santiago', 'Masters Santiago',     '2026', 'April 2026'),
]

SLUG_TO_ORG = {
    'paper-rex': 'PRX', 'fnatic': 'FNC', 'loud': 'LOUD', 'kiwoom-drx': 'KRX',
    'natus-vincere': 'NAVI', 'evil-geniuses': 'EG', 'nrg': 'NRG',
    'sentinels': 'SEN', 'gen-g': 'GEN', 'team-heretics': 'TH',
    'leviat-n': 'LEV', 'edward-gaming': 'EDG', 't1': 'T1', 'g2-esports': 'G2',
    'team-vitality': 'VIT', 'wolves-esports': 'WOL', 'xi-lai-gaming': 'XLG',
    'rex-regum-qeon': 'RRQ', 'mibr': 'MIBR', 'giantx': 'GX',
    'nongshim-redforce': 'NS', 'all-gamers': 'AG', 'bbl-esports': 'BBL',
    'gentle-mates': 'M8', 'furia': 'FUR', 'dragonx': 'KRX', 'drx': 'KRX',
}

_cache = None

def _load_data():
    global _cache
    if _cache is not None:
        return _cache

    base = os.path.dirname(__file__)
    mr = pd.read_csv(os.path.join(base, 'data', 'match_results.csv'))

    with open(os.path.join(base, 'data', 'intl_placements.json')) as f:
        placements = json.load(f)

    with open(os.path.join(base, 'static', 'logos', 'logos.json')) as f:
        logos = json.load(f)

    all_events = {}

    for key, label, year, date_str in INTL_EVENT_META:
        csv_path = os.path.join(base, 'data', 'maps', f'{key}.csv')
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        df = df[~df['Org'].isin(JUNK_ORGS) & df['Org'].notna()].copy()
        df = df.drop_duplicates(['Player', 'MatchID', 'MapNum'])

        # Filter out showmatch data using match_results
        match_ids_all = set(df['MatchID'].unique())
        mr_all = mr[mr['MatchID'].isin(match_ids_all)]
        showmatch_ids = set(mr_all[mr_all['MatchName'].str.contains('Showmatch|Main Event', case=False, na=False)]['MatchID'].unique())
        if showmatch_ids:
            df = df[~df['MatchID'].isin(showmatch_ids)]

        match_ids = set(df['MatchID'].unique())
        mr_ev = mr[mr['MatchID'].isin(match_ids)].copy()
        # Filter out showmatches from match results
        mr_ev = mr_ev[~mr_ev['MatchName'].str.contains('Showmatch|Main Event', case=False, na=False)]

        # Build match list
        matches = []
        for mid, grp in df.groupby('MatchID'):
            orgs = list(grp['Org'].unique())
            if len(orgs) < 2:
                continue
            mr_rows = mr_ev[mr_ev['MatchID'] == mid]
            if mr_rows.empty:
                continue

            winner_org = mr_rows.iloc[0]['WinnerOrg']
            match_name = mr_rows.iloc[0]['MatchName']

            # Skip showmatches (from player CSV side)
            if 'Showmatch' in match_name or 'Main Event' in match_name:
                continue

            orgs_set = set(orgs) - JUNK_ORGS
            losers = orgs_set - {winner_org}
            loser_org = list(losers)[0] if losers else list(orgs_set - {winner_org})[0] if len(orgs_set) > 1 else '?'

            # Use 'all' row for series score, individual rows for map details
            all_row = mr_rows[mr_rows['MapNum'].astype(str) == 'all']
            map_rows = mr_rows[mr_rows['MapNum'].astype(str) != 'all']

            if not all_row.empty:
                series_score = all_row.iloc[0]['Score']
            else:
                winner_maps = len(map_rows[map_rows['WinnerOrg'] == winner_org])
                loser_maps = len(map_rows[map_rows['WinnerOrg'] == loser_org])
                series_score = f'{winner_maps}-{loser_maps}'

            # Build mapnum → map name lookup from player CSV (keyed as string)
            mapnum_to_name = {}
            for _, pr in grp[['MapNum','MapName']].drop_duplicates().iterrows():
                mapnum_to_name[str(pr['MapNum'])] = str(pr['MapName']).replace('PICK','').strip()

            # Per-map details
            map_details = []
            for _, row in map_rows.iterrows():
                map_name = mapnum_to_name.get(str(row['MapNum']), '')
                map_details.append({
                    'winner': row['WinnerOrg'],
                    'score': row['Score'],
                    'map': map_name,
                })

            matches.append({
                'match_id': int(mid),
                'stage': match_name,
                'winner': winner_org,
                'loser': loser_org,
                'series_score': series_score,
                'maps': map_details,
            })

        matches.sort(key=lambda x: x['match_id'])

        # Build player stats (aggregated per player)
        numeric_cols = ['R2.0', 'ACS', 'K', 'D', 'A', 'ADR', 'FK', 'FD']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        def pct_to_float(s):
            try:
                return float(str(s).replace('%', '')) / 100
            except:
                return None

        df['KAST_f'] = df['KAST'].apply(pct_to_float)
        df['HS_f'] = df['HS%'].apply(pct_to_float)

        player_stats = []
        for (player, org), pgrp in df.groupby(['Player', 'Org']):
            maps_played = len(pgrp)
            def flt(series, mult=1, ndigits=2):
                if series.notna().any():
                    return round(float(series.mean()) * mult, ndigits)
                return None

            stats = {
                'player': player,
                'org': org,
                'profile_url': pgrp['ProfileURL'].iloc[0] if 'ProfileURL' in pgrp else '',
                'maps': maps_played,
                'rating': flt(pgrp['R2.0']),
                'acs': flt(pgrp['ACS'], ndigits=1),
                'k': flt(pgrp['K'], ndigits=1),
                'd': flt(pgrp['D'], ndigits=1),
                'a': flt(pgrp['A'], ndigits=1),
                'kast': flt(pgrp['KAST_f'], mult=100, ndigits=1),
                'adr': flt(pgrp['ADR'], ndigits=1),
                'hs': flt(pgrp['HS_f'], mult=100, ndigits=1),
                'fk': flt(pgrp['FK'], ndigits=1),
                'fd': flt(pgrp['FD'], ndigits=1),
            }
            player_stats.append(stats)

        player_stats.sort(key=lambda x: (x['rating'] or 0), reverse=True)

        # Standings with org code lookup
        raw_standings = placements.get(key, {}).get('standings', [])
        standings = []
        for s in raw_standings:
            org_code = SLUG_TO_ORG.get(s['slug'], s['slug'].upper()[:4])
            standings.append({
                'place': s['place'],
                'team': s['team'],
                'org': org_code,
                'logo': logos.get(org_code),
            })

        # All orgs that participated
        all_orgs = sorted(set(df['Org'].unique()) - JUNK_ORGS)

        all_events[key] = {
            'key': key,
            'label': label,
            'year': year,
            'date': date_str,
            'standings': standings,
            'matches': matches,
            'player_stats': player_stats,
            'orgs': all_orgs,
        }

    _cache = {'events': all_events, 'logos': logos, 'meta': INTL_EVENT_META}
    return _cache


INTL_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>International Events — Bobo's VCT Database</title>
<link rel="icon" type="image/svg+xml" href="/favicon.svg">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
:root {
  --rose:#f4b8c1; --peach:#f9cba7; --mint:#b8e8d4;
  --sky:#b8d8f4; --lavender:#d4b8f4; --lemon:#f4edb8;
  --cream:#fdf6f0; --ink:#2a1f2d; --soft:#7a6e7e; --card:#fff;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--cream);font-family:'DM Sans',sans-serif;color:var(--ink);min-height:100vh;}
body::before{content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
  background:radial-gradient(ellipse 60% 50% at 10% 10%,#f4b8c133 0%,transparent 70%),
             radial-gradient(ellipse 50% 60% at 90% 20%,#b8d8f433 0%,transparent 70%),
             radial-gradient(ellipse 55% 45% at 15% 85%,#b8e8d433 0%,transparent 70%),
             radial-gradient(ellipse 60% 50% at 85% 80%,#d4b8f433 0%,transparent 70%);}
.top-nav{position:relative;z-index:10;display:flex;align-items:center;padding:16px 32px;gap:12px;}
.home-logo{display:flex;align-items:center;gap:8px;text-decoration:none;color:var(--ink);}
.home-logo img{height:32px;width:auto;}
.home-logo span{font-family:'Syne',sans-serif;font-size:.95rem;font-weight:700;opacity:.7;}
.page{position:relative;z-index:1;max-width:1200px;margin:0 auto;padding:0 24px 60px;}
h1{font-family:'Syne',sans-serif;font-size:clamp(1.8rem,4vw,2.8rem);font-weight:800;letter-spacing:-1px;margin-bottom:6px;}
.subtitle{color:var(--soft);font-size:.9rem;margin-bottom:28px;}

/* Event tabs */
.event-tabs{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:28px;}
.event-tab{background:white;border:none;border-radius:12px;padding:8px 16px;font-family:'DM Sans',sans-serif;
  font-size:.8rem;cursor:pointer;color:var(--soft);box-shadow:0 2px 8px #0000000a;transition:all .2s;}
.event-tab.active{background:var(--ink);color:white;box-shadow:0 4px 12px #0000001a;}
.event-tab:hover:not(.active){transform:translateY(-2px);box-shadow:0 4px 12px #0000001a;}
.year-group{display:flex;align-items:center;gap:8px;flex-wrap:wrap;}
.year-label{font-family:'Syne',sans-serif;font-size:.75rem;font-weight:800;color:var(--soft);text-transform:uppercase;letter-spacing:1px;padding:4px 0;}

/* Sections */
.section{margin-bottom:32px;}
.section-title{font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;margin-bottom:14px;color:var(--ink);}

/* Standings */
.standings-grid{display:flex;flex-wrap:wrap;gap:12px;}
.standing-card{background:white;border-radius:16px;padding:14px 18px;display:flex;align-items:center;
  gap:12px;box-shadow:0 2px 12px #0000000a;min-width:160px;}
.standing-place{font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;color:var(--soft);width:32px;}
.standing-place.gold{color:#c89b3c;}
.standing-place.silver{color:#9ca3af;}
.standing-place.bronze{color:#c87c3c;}
.standing-logo{width:32px;height:32px;object-fit:contain;}
.standing-logo-placeholder{width:32px;height:32px;background:var(--cream);border-radius:6px;}
.standing-name{font-size:.85rem;font-weight:500;}

/* Matches */
.stage-group{margin-bottom:20px;}
.stage-label{font-size:.75rem;font-weight:700;text-transform:uppercase;letter-spacing:.8px;color:var(--soft);margin-bottom:8px;}
.match-row{background:white;border-radius:12px;padding:10px 16px;display:flex;align-items:center;
  gap:12px;margin-bottom:6px;box-shadow:0 2px 8px #0000000a;cursor:pointer;transition:all .15s;}
.match-row:hover{transform:translateX(3px);box-shadow:0 4px 12px #0000001a;}
.match-team{display:flex;align-items:center;gap:8px;flex:1;}
.match-team.loser{opacity:.5;}
.match-logo{width:22px;height:22px;object-fit:contain;}
.match-org{font-size:.85rem;font-weight:500;}
.match-score{font-family:'Syne',sans-serif;font-size:.95rem;font-weight:800;color:var(--ink);min-width:40px;text-align:center;}
.match-vs{color:var(--soft);font-size:.75rem;min-width:14px;text-align:center;}

/* Map breakdown row */
.map-breakdown{display:none;background:#f9f6f3;border-radius:0 0 12px 12px;padding:10px 16px 12px;
  margin-top:-6px;margin-bottom:6px;border-top:1px solid #f0ebe6;}
.map-breakdown.open{display:block;}
.map-row{display:flex;align-items:center;gap:10px;padding:4px 0;font-size:.8rem;border-bottom:1px solid #ede8e3;}
.map-row:last-child{border-bottom:none;}
.map-name{flex:1;color:var(--soft);}
.map-winner-label{font-weight:600;font-size:.75rem;}
.map-score{font-size:.75rem;color:var(--soft);}

/* Stats table */
.filter-bar{display:flex;align-items:center;gap:10px;margin-bottom:14px;flex-wrap:wrap;}
.filter-bar select, .filter-bar input{background:white;border:none;border-radius:10px;padding:7px 12px;
  font-family:'DM Sans',sans-serif;font-size:.82rem;color:var(--ink);box-shadow:0 2px 8px #0000000a;outline:none;}
.stats-wrap{overflow-x:auto;}
table{width:100%;border-collapse:collapse;background:white;border-radius:16px;overflow:hidden;
  box-shadow:0 2px 12px #0000000a;font-size:.8rem;}
thead{background:var(--ink);color:white;}
th{padding:10px 12px;text-align:left;font-weight:600;white-space:nowrap;cursor:pointer;user-select:none;
  font-family:'Syne',sans-serif;font-size:.72rem;letter-spacing:.3px;}
th:hover{background:#3d2f41;}
th.active-sort{background:#503958;}
th .sort-arrow{margin-left:4px;opacity:.6;}
td{padding:9px 12px;border-bottom:1px solid #f5f0ec;white-space:nowrap;}
tr:last-child td{border-bottom:none;}
tr:hover td{background:#faf6f3;}
.org-cell{display:flex;align-items:center;gap:7px;}
.org-logo{width:18px;height:18px;object-fit:contain;}
.player-link{color:var(--ink);text-decoration:none;font-weight:500;}
.player-link:hover{text-decoration:underline;}
.stat-good{color:#2d8a4e;}
.stat-bad{color:#c0392b;}
.rank-cell{color:var(--soft);font-size:.72rem;width:28px;}

/* Modal */
.modal-bg{display:none;position:fixed;inset:0;background:#0006;z-index:100;align-items:center;justify-content:center;}
.modal-bg.open{display:flex;}
.modal{background:white;border-radius:20px;padding:28px;max-width:520px;width:90%;max-height:80vh;overflow-y:auto;}
.modal-title{font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;margin-bottom:16px;}
.modal-close{float:right;background:none;border:none;font-size:1.2rem;cursor:pointer;color:var(--soft);}
.modal-map-row{display:grid;grid-template-columns:auto 1fr auto;gap:8px;align-items:center;
  padding:8px 0;border-bottom:1px solid #f0ebe6;font-size:.82rem;}
.modal-map-row:last-child{border-bottom:none;}
.modal-map-name{color:var(--soft);font-size:.75rem;}
.modal-map-score{font-weight:600;text-align:right;}

@media(max-width:600px){
  .top-nav{padding:12px 16px;}
  .page{padding:0 16px 40px;}
  .standings-grid{gap:8px;}
  .standing-card{min-width:130px;padding:10px 14px;}
}
</style>
</head>
<body>
<nav class="top-nav">
  <a class="home-logo" href="/">
    <img src="/logo.svg" alt="Bobo">
    <span>Bobo's VCT Database</span>
  </a>
</nav>
<div class="page">
  <h1>International Events</h1>
  <p class="subtitle">Match results, final standings, and player stats for every VCT international event.</p>

  <div class="event-tabs" id="event-tabs"></div>

  <div id="event-content"></div>
</div>

<!-- Map detail modal -->
<div class="modal-bg" id="map-modal">
  <div class="modal">
    <button class="modal-close" onclick="closeModal()">✕</button>
    <div class="modal-title" id="modal-title"></div>
    <div id="modal-body"></div>
  </div>
</div>

<script>
var DATA = {{ data_json }};
var LOGOS = {{ logos_json }};

var currentEvent = null;
var sortCol = 'rating';
var sortAsc = false;
var orgFilter = '';
var searchFilter = '';

function logoSrc(org) {
  return LOGOS[org] ? '/logos/' + LOGOS[org] : null;
}

function logoImg(org, size) {
  var src = logoSrc(org);
  size = size || 22;
  if (src) return '<img class="match-logo" src="' + src + '" style="width:' + size + 'px;height:' + size + 'px" alt="' + org + '">';
  return '<div style="width:' + size + 'px;height:' + size + 'px;background:#f0ebe6;border-radius:4px;display:inline-block"></div>';
}

function buildTabs() {
  var container = document.getElementById('event-tabs');
  var byYear = {};
  DATA.meta.forEach(function(m) {
    var key = m[0], label = m[1], year = m[2];
    if (!byYear[year]) byYear[year] = [];
    byYear[year].push({key: key, label: label});
  });

  ['2023','2024','2025','2026'].forEach(function(yr) {
    if (!byYear[yr]) return;
    var wrap = document.createElement('div');
    wrap.className = 'year-group';
    var lbl = document.createElement('span');
    lbl.className = 'year-label';
    lbl.textContent = yr;
    wrap.appendChild(lbl);
    byYear[yr].forEach(function(ev) {
      var btn = document.createElement('button');
      btn.className = 'event-tab';
      btn.textContent = ev.label;
      btn.dataset.key = ev.key;
      btn.onclick = function() { selectEvent(ev.key); };
      wrap.appendChild(btn);
    });
    container.appendChild(wrap);
  });
}

function selectEvent(key) {
  currentEvent = key;
  orgFilter = '';
  sortCol = 'rating';
  sortAsc = false;

  document.querySelectorAll('.event-tab').forEach(function(b) {
    b.classList.toggle('active', b.dataset.key === key);
  });

  renderEvent(DATA.events[key]);
}

function renderEvent(ev) {
  var html = '';

  // Standings
  if (ev.standings && ev.standings.length) {
    html += '<div class="section"><div class="section-title">Final Standings</div><div class="standings-grid">';
    ev.standings.forEach(function(s) {
      var placeClass = s.place === 1 ? 'gold' : s.place === 2 ? 'silver' : s.place === 3 ? 'bronze' : '';
      var suffix = s.place === 1 ? 'st' : s.place === 2 ? 'nd' : s.place === 3 ? 'rd' : 'th';
      var logoHtml = s.logo
        ? '<img class="standing-logo" src="/logos/' + s.logo + '" alt="' + s.org + '">'
        : '<div class="standing-logo-placeholder"></div>';
      html += '<div class="standing-card">' +
        '<div class="standing-place ' + placeClass + '">' + s.place + '<sup style="font-size:.6em">' + suffix + '</sup></div>' +
        logoHtml +
        '<div class="standing-name">' + s.org + '</div>' +
        '</div>';
    });
    html += '</div></div>';
  }

  // Matches by stage
  var stages = {};
  var stageOrder = [];
  ev.matches.forEach(function(m) {
    if (!stages[m.stage]) { stages[m.stage] = []; stageOrder.push(m.stage); }
    stages[m.stage].push(m);
  });

  html += '<div class="section"><div class="section-title">Match Results</div>';
  stageOrder.forEach(function(stage) {
    html += '<div class="stage-group"><div class="stage-label">' + stage + '</div>';
    stages[stage].forEach(function(m) {
      html += '<div class="match-row" onclick="toggleMapDetail(' + m.match_id + ')">' +
        '<div class="match-team">' + logoImg(m.winner, 22) + '<span class="match-org">' + m.winner + '</span></div>' +
        '<span class="match-score">' + m.series_score + '</span>' +
        '<span class="match-vs">vs</span>' +
        '<div class="match-team loser">' + logoImg(m.loser, 22) + '<span class="match-org">' + m.loser + '</span></div>' +
        '</div>' +
        '<div class="map-breakdown" id="maps-' + m.match_id + '">' +
        buildMapBreakdown(m) +
        '</div>';
    });
    html += '</div>';
  });
  html += '</div>';

  // Player stats
  html += '<div class="section"><div class="section-title">Player Stats</div>' +
    '<div class="filter-bar">' +
    '<select id="org-filter" onchange="setOrgFilter(this.value)"><option value="">All Teams</option>' +
    ev.orgs.map(function(o) { return '<option value="' + o + '">' + o + '</option>'; }).join('') +
    '</select>' +
    '<input id="player-search" type="text" placeholder="Search player..." oninput="setSearchFilter(this.value)">' +
    '</div>' +
    '<div class="stats-wrap"><table><thead><tr>' +
    '<th></th>' +
    '<th onclick="setSort(\'player\')" data-col="player">Player<span class="sort-arrow"></span></th>' +
    '<th onclick="setSort(\'org\')" data-col="org">Team<span class="sort-arrow"></span></th>' +
    '<th onclick="setSort(\'maps\')" data-col="maps">Maps<span class="sort-arrow"></span></th>' +
    '<th onclick="setSort(\'rating\')" data-col="rating">Rating<span class="sort-arrow"></span></th>' +
    '<th onclick="setSort(\'acs\')" data-col="acs">ACS<span class="sort-arrow"></span></th>' +
    '<th onclick="setSort(\'k\')" data-col="k">K<span class="sort-arrow"></span></th>' +
    '<th onclick="setSort(\'d\')" data-col="d">D<span class="sort-arrow"></span></th>' +
    '<th onclick="setSort(\'a\')" data-col="a">A<span class="sort-arrow"></span></th>' +
    '<th onclick="setSort(\'kast\')" data-col="kast">KAST%<span class="sort-arrow"></span></th>' +
    '<th onclick="setSort(\'adr\')" data-col="adr">ADR<span class="sort-arrow"></span></th>' +
    '<th onclick="setSort(\'hs\')" data-col="hs">HS%<span class="sort-arrow"></span></th>' +
    '<th onclick="setSort(\'fk\')" data-col="fk">FK<span class="sort-arrow"></span></th>' +
    '<th onclick="setSort(\'fd\')" data-col="fd">FD<span class="sort-arrow"></span></th>' +
    '</tr></thead><tbody id="stats-tbody"></tbody></table></div></div>';

  document.getElementById('event-content').innerHTML = html;
  renderStats(ev);
  updateSortArrows();
}

function buildMapBreakdown(match) {
  if (!match.maps || !match.maps.length) return '<div style="color:var(--soft);font-size:.8rem">No map data</div>';
  var html = '';
  match.maps.forEach(function(mp) {
    html += '<div class="map-row">' +
      '<span class="map-name">' + (mp.map || '') + '</span>' +
      '<span class="map-winner-label">' + mp.winner + '</span>' +
      '<span class="map-score">' + mp.score + '</span>' +
      '</div>';
  });
  return html;
}

function toggleMapDetail(matchId) {
  var el = document.getElementById('maps-' + matchId);
  if (el) el.classList.toggle('open');
}

function renderStats(ev) {
  var stats = ev.player_stats.filter(function(p) {
    if (orgFilter && p.org !== orgFilter) return false;
    if (searchFilter && p.player.toLowerCase().indexOf(searchFilter) === -1) return false;
    return true;
  });

  stats = stats.slice().sort(function(a, b) {
    var va = a[sortCol], vb = b[sortCol];
    if (va === null || va === undefined) va = sortAsc ? Infinity : -Infinity;
    if (vb === null || vb === undefined) vb = sortAsc ? Infinity : -Infinity;
    if (typeof va === 'string') return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
    return sortAsc ? va - vb : vb - va;
  });

  var median_rating = getMedian(ev.player_stats.map(function(p) { return p.rating; }).filter(function(v) { return v !== null; }));

  var html = '';
  stats.forEach(function(p, i) {
    var ratingClass = p.rating !== null ? (p.rating > median_rating * 1.05 ? 'stat-good' : p.rating < median_rating * 0.95 ? 'stat-bad' : '') : '';
    var logoH = logoSrc(p.org) ? '<img class="org-logo" src="/logos/' + LOGOS[p.org] + '" alt="' + p.org + '">' : '';
    html += '<tr>' +
      '<td class="rank-cell">' + (i+1) + '</td>' +
      '<td><a class="player-link" href="' + p.profile_url + '" target="_blank">' + p.player + '</a></td>' +
      '<td><div class="org-cell">' + logoH + p.org + '</div></td>' +
      '<td>' + p.maps + '</td>' +
      '<td class="' + ratingClass + '">' + (p.rating !== null ? p.rating : '—') + '</td>' +
      '<td>' + (p.acs !== null ? p.acs : '—') + '</td>' +
      '<td>' + (p.k !== null ? p.k : '—') + '</td>' +
      '<td>' + (p.d !== null ? p.d : '—') + '</td>' +
      '<td>' + (p.a !== null ? p.a : '—') + '</td>' +
      '<td>' + (p.kast !== null ? p.kast + '%' : '—') + '</td>' +
      '<td>' + (p.adr !== null ? p.adr : '—') + '</td>' +
      '<td>' + (p.hs !== null ? p.hs + '%' : '—') + '</td>' +
      '<td>' + (p.fk !== null ? p.fk : '—') + '</td>' +
      '<td>' + (p.fd !== null ? p.fd : '—') + '</td>' +
      '</tr>';
  });

  document.getElementById('stats-tbody').innerHTML = html || '<tr><td colspan="14" style="text-align:center;color:var(--soft);padding:20px">No players found</td></tr>';
}

function getMedian(arr) {
  if (!arr.length) return 1;
  var s = arr.slice().sort(function(a,b){return a-b;});
  var m = Math.floor(s.length/2);
  return s.length % 2 ? s[m] : (s[m-1]+s[m])/2;
}

function setSort(col) {
  if (sortCol === col) sortAsc = !sortAsc;
  else { sortCol = col; sortAsc = false; }
  updateSortArrows();
  var ev = DATA.events[currentEvent];
  if (ev) renderStats(ev);
}

function updateSortArrows() {
  document.querySelectorAll('th[data-col]').forEach(function(th) {
    var col = th.dataset.col;
    var arrow = th.querySelector('.sort-arrow');
    if (!arrow) return;
    th.classList.toggle('active-sort', col === sortCol);
    arrow.textContent = col === sortCol ? (sortAsc ? ' ↑' : ' ↓') : '';
  });
}

function setOrgFilter(val) { orgFilter = val; var ev = DATA.events[currentEvent]; if (ev) renderStats(ev); }
function setSearchFilter(val) { searchFilter = val.toLowerCase(); var ev = DATA.events[currentEvent]; if (ev) renderStats(ev); }

function closeModal() { document.getElementById('map-modal').classList.remove('open'); }

// Init
buildTabs();
// Default: select most recent event
var defaultKey = DATA.meta[DATA.meta.length-1][0];
selectEvent(defaultKey);
</script>
</body>
</html>
"""


@intl_bp.route('/')
def intl_home():
    data = _load_data()

    # Serialize events (player_stats can be large; send all)
    events_js = {}
    for key, ev in data['events'].items():
        events_js[key] = {
            'key': ev['key'],
            'label': ev['label'],
            'year': ev['year'],
            'standings': ev['standings'],
            'matches': ev['matches'],
            'player_stats': ev['player_stats'],
            'orgs': ev['orgs'],
        }

    data_obj = {
        'meta': [[k, label, yr, date] for k, label, yr, date in INTL_EVENT_META if k in events_js],
        'events': events_js,
    }

    return INTL_HTML.replace('{{ data_json }}', json.dumps(data_obj)).replace('{{ logos_json }}', json.dumps(data['logos']))
