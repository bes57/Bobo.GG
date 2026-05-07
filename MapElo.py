import os
import json
import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
from flask import Blueprint

mapelo_bp = Blueprint('mapelo_bp', __name__)

ROOT = os.path.dirname(os.path.abspath(__file__))


def _build_team_stats(rdf, k, pdf=None, min_maps=3):
    """Aggregate rounds/wins for each team in rdf, return sorted list of dicts."""
    records = {}
    for _, row in rdf.iterrows():
        map_name   = row.get('map_name',   'Unknown') if hasattr(row, 'get') else getattr(row, 'map_name',   'Unknown')
        match_id   = int(row.get('match_id', 0))      if hasattr(row, 'get') else int(getattr(row, 'match_id',   0))
        match_name = row.get('match_name', '')         if hasattr(row, 'get') else getattr(row, 'match_name', '')
        for team, rw, rl, opp, is_win in [
            (row['winner'], row['wr'], row['lr'], row['loser'],  True),
            (row['loser'],  row['lr'], row['wr'], row['winner'], False),
        ]:
            if team not in records:
                records[team] = {'wins': 0, 'losses': 0, 'rw': 0, 'rl': 0, 'matches': [], 'map_stats': {}}
            records[team]['wins']   += int(is_win)
            records[team]['losses'] += int(not is_win)
            records[team]['rw']     += rw
            records[team]['rl']     += rl
            records[team]['matches'].append({
                'opponent':   opp,
                'score':      f'{rw}-{rl}',
                'win':        is_win,
                'diff':       rw - rl,
                'map':        map_name,
                'match_id':   match_id,
                'match_name': match_name,
            })
            ms = records[team]['map_stats']
            if map_name not in ms:
                ms[map_name] = {'wins': 0, 'losses': 0, 'rw': 0, 'rl': 0}
            ms[map_name]['wins']   += int(is_win)
            ms[map_name]['losses'] += int(not is_win)
            ms[map_name]['rw']     += rw
            ms[map_name]['rl']     += rl

    # Load headshots cache
    _hs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/headshots.json')
    try:
        with open(_hs_path) as _f:
            _headshots = json.load(_f)
    except Exception:
        _headshots = {}

    # Compute most common 5-player lineup per team from player data
    rosters = {}
    if pdf is not None and not pdf.empty:
        match_ids = set(int(m) for m in rdf['match_id'].unique())
        pf = pdf[pdf['MatchID'].isin(match_ids)].copy()
        from collections import Counter
        for org, grp in pf.groupby('Org'):
            url_map = dict(zip(grp['Player'], grp['ProfileURL']))
            lineup_counts = Counter()
            for (mid, mnum), mgrp in grp.groupby(['MatchID', 'MapNum']):
                players = tuple(sorted(mgrp['Player'].unique()))
                if len(players) >= 4:
                    lineup_counts[players] += 1
            if not lineup_counts:
                continue
            best_lineup = lineup_counts.most_common(1)[0][0]
            rosters[org] = [{'player': p, 'url': url_map.get(p, ''),
                              'headshot': _headshots.get(p, '')} for p in best_lineup]

    out = []
    for team, v in records.items():
        total = v['wins'] + v['losses']
        if total < min_maps:
            continue
        win_pct  = v['wins'] / total
        pyth_pct = v['rw'] ** k / (v['rw'] ** k + v['rl'] ** k)

        map_list = []
        for mn, ms in v['map_stats'].items():
            map_list.append({
                'map':      mn,
                'wins':     ms['wins'],
                'losses':   ms['losses'],
                'rw':       ms['rw'],
                'rl':       ms['rl'],
                'rd':       ms['rw'] - ms['rl'],
                'rw_pct':   round(ms['rw'] / (ms['rw'] + ms['rl']), 4) if (ms['rw'] + ms['rl']) > 0 else 0,
            })
        map_list.sort(key=lambda m: m['rd'], reverse=True)

        out.append({
            'org':      team,
            'wins':     v['wins'],
            'losses':   v['losses'],
            'rw':       v['rw'],
            'rl':       v['rl'],
            'win_pct':  round(win_pct, 4),
            'pyth_pct': round(pyth_pct, 4),
            'luck':     round(win_pct - pyth_pct, 4),
            'matches':  sorted(v['matches'], key=lambda m: -m['match_id']),
            'map_stats': map_list,
            'roster':   rosters.get(team, []),
        })

    out.sort(key=lambda r: r['pyth_pct'], reverse=True)
    for i, r in enumerate(out):
        r['rank'] = i + 1
    return out


def _compute_pyth_data():
    from MoreTestingMaybeFiles import ALL_EVENTS

    # Chronological list of regional events per year
    regional_chron = [e for e in reversed(ALL_EVENTS) if 'International' not in e['regions']]

    events_by_year = {}
    for e in regional_chron:
        y = str(e['year'])
        if y not in events_by_year:
            events_by_year[y] = []
        short_label = e['label'].split(' ', 1)[1] if ' ' in e['label'] else e['label']
        events_by_year[y].append({'id': e['id'], 'label': short_label})

    # Load all map frames tagged with event_id
    mr = pd.read_csv(os.path.join(ROOT, 'data/match_results.csv'))
    mr = mr[mr['MapNum'] != 'all'].copy()
    mr['MapNum'] = mr['MapNum'].astype(str)

    map_frames  = []
    player_frames = []
    for e in regional_chron:
        path = os.path.join(ROOT, f'data/maps/{e["id"]}.csv')
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df['year']     = e['year']
        df['event_id'] = e['id']
        df['MapNum']   = df['MapNum'].astype(str)
        df['MapName'] = df['MapName'].str.replace('PICK', '', regex=False).str.strip()
        map_frames.append(df[['MatchID', 'MapNum', 'Org', 'MapName', 'year', 'event_id']])
        if 'Player' in df.columns and 'ProfileURL' in df.columns:
            player_frames.append(df[['MatchID', 'MapNum', 'Org', 'Player', 'ProfileURL']].copy())

    if not map_frames:
        return {'exponent': 2.0, 'k_curve': {'k': [], 'mse': []},
                'years': [], 'events_by_year': {}, 'data': {}}

    pdf = pd.concat(player_frames) if player_frames else pd.DataFrame()

    all_maps = pd.concat(map_frames)
    orgs_per = all_maps.groupby(['MatchID', 'MapNum']).agg(
        Orgs=('Org',      lambda x: list(x.unique())),
        MapName=('MapName', 'first'),
        year=('year',     'first'),
        event_id=('event_id', 'first'),
    ).reset_index()

    merged = mr.merge(orgs_per, on=['MatchID', 'MapNum'], how='inner')

    rows = []
    for _, row in merged.iterrows():
        orgs   = row['Orgs']
        winner = row['WinnerOrg']
        losers = [o for o in orgs if o != winner]
        if not losers:
            continue
        w_rounds, l_rounds = map(int, row['Score'].split('-'))
        rows.append({
            'year':       int(row['year']),
            'event_id':   row['event_id'],
            'map_name':   row['MapName'],
            'match_id':   int(row['MatchID']),
            'match_name': row.get('MatchName', ''),
            'winner':     winner,
            'loser':      losers[0],
            'wr':         w_rounds,
            'lr':         l_rounds,
        })

    rdf = pd.DataFrame(rows)

    # Fit optimal exponent globally (no match detail needed for this)
    global_df = rdf.copy()
    rec = {}
    for _, row in global_df.iterrows():
        for team, rw, rl, is_win in [
            (row['winner'], row['wr'], row['lr'], True),
            (row['loser'],  row['lr'], row['wr'], False),
        ]:
            if team not in rec:
                rec[team] = {'wins': 0, 'losses': 0, 'rw': 0, 'rl': 0}
            rec[team]['wins']   += int(is_win)
            rec[team]['losses'] += int(not is_win)
            rec[team]['rw']     += rw
            rec[team]['rl']     += rl

    fit_rows = [{'org': t, **v} for t, v in rec.items()]
    fit_df = pd.DataFrame(fit_rows)
    fit_df['total']   = fit_df['wins'] + fit_df['losses']
    fit_df['win_pct'] = fit_df['wins'] / fit_df['total']
    fit_df = fit_df[fit_df['total'] >= 5]

    def mse(k):
        p = fit_df['rw'] ** k / (fit_df['rw'] ** k + fit_df['rl'] ** k)
        return ((p - fit_df['win_pct']) ** 2).mean()

    res = minimize_scalar(mse, bounds=(0.5, 10.0), method='bounded')
    k = round(float(res.x), 3)

    k_vals  = [round(x * 0.1, 1) for x in range(5, 101)]
    mse_vals = [round(float(mse(kv)), 6) for kv in k_vals]

    # Known event date ranges for time-frame filtering
    EVENT_DATES = {
        '2023_league':           ('2023-01-23', '2023-10-01'),
        '2024_kickoff':          ('2024-01-08', '2024-02-11'),
        '2024_stage1':           ('2024-03-15', '2024-05-19'),
        '2024_stage2':           ('2024-06-20', '2024-08-25'),
        '2025_kickoff':          ('2025-01-13', '2025-02-09'),
        '2025_stage1':           ('2025-03-14', '2025-05-18'),
        '2025_stage2':           ('2025-07-14', '2025-08-24'),
        '2026_kickoff':          ('2026-01-07', '2026-02-09'),
        '2026_stage1':           ('2026-04-23', '2026-06-15'),
        '2023_lock_in':          ('2023-02-13', '2023-02-26'),
        '2023_masters_tokyo':    ('2023-06-11', '2023-06-25'),
        '2023_champions':        ('2023-08-06', '2023-08-26'),
        '2024_masters_madrid':   ('2024-02-14', '2024-03-03'),
        '2024_champions':        ('2024-08-01', '2024-08-25'),
        '2025_masters_bangkok':  ('2025-02-05', '2025-02-23'),
        '2025_masters_toronto':  ('2025-05-13', '2025-06-01'),
        '2025_champions':        ('2025-08-07', '2025-08-24'),
        '2026_masters_santiago': ('2026-03-26', '2026-04-06'),
    }

    INTL_EVENT_DATES = {
        '2023_lock_in':          ('LOCK//IN São Paulo', '2023'),
        '2023_masters_tokyo':    ('Masters Tokyo',      '2023'),
        '2023_champions':        ('Champions 2023',     '2023'),
        '2024_masters_madrid':   ('Masters Madrid',     '2024'),
        '2024_champions':        ('Champions 2024',     '2024'),
        '2025_masters_bangkok':  ('Masters Bangkok',    '2025'),
        '2025_masters_toronto':  ('Masters Toronto',    '2025'),
        '2025_champions':        ('Champions 2025',     '2025'),
        '2026_masters_santiago': ('Masters Santiago',   '2026'),
    }
    # International events that fall between regional events, keyed by year
    INTL_EVENTS = {
        '2024': [{'label': 'Masters Madrid',   'end': '2024-03-03'}],
        '2025': [{'label': 'Masters Bangkok',  'end': '2025-02-23'},
                 {'label': 'Masters Toronto',  'end': '2025-06-01'}],
        '2026': [{'label': 'Masters Santiago', 'end': '2026-04-06'}],
    }

    # Assign approximate date using event start date
    rdf['date'] = rdf['event_id'].map(lambda eid: EVENT_DATES.get(eid, ('',))[0])

    # Build all data slices
    data = {}
    years = sorted(rdf['year'].unique())

    for year in years:
        y = str(year)
        year_rows = rdf[rdf['year'] == year]
        year_data = _build_team_stats(year_rows, k, pdf)
        for r in year_data:
            r['year'] = int(year)
        data[y] = year_data

        evs = events_by_year.get(y, [])
        ev_ids = [e['id'] for e in evs]

        for ev in evs:
            ev_rows = year_rows[year_rows['event_id'] == ev['id']]
            if len(ev_rows):
                ev_data = _build_team_stats(ev_rows, k, pdf)
                for r in ev_data:
                    r['year'] = int(year)
                data[ev['id']] = ev_data

        # "from X onwards" only for middle events (not first, not last)
        for i in range(1, len(ev_ids) - 1):
            onwards_ids  = ev_ids[i:]
            onwards_rows = year_rows[year_rows['event_id'].isin(onwards_ids)]
            if len(onwards_rows):
                on_data = _build_team_stats(onwards_rows, k, pdf)
                for r in on_data:
                    r['year'] = int(year)
                data[ev_ids[i] + '+'] = on_data

    # All-time: only include years that are fully completed (last event end date < today)
    from datetime import date as _date
    today_str = _date.today().isoformat()
    def year_is_complete(y_int):
        y = str(y_int)
        ev_ids = [e['id'] for e in events_by_year.get(y, [])]
        if not ev_ids:
            return False
        last_end = max((EVENT_DATES[eid][1] for eid in ev_ids if eid in EVENT_DATES), default='')
        return bool(last_end) and last_end < today_str

    all_time = []
    for y_int in years:
        if not year_is_complete(y_int):
            continue
        for r in data.get(str(y_int), []):
            all_time.append(dict(r))
    all_time.sort(key=lambda r: r['pyth_pct'], reverse=True)
    for i, r in enumerate(all_time):
        r['rank'] = i + 1
    data['all_time'] = all_time

    # All-time by splits: each (team, event) combo ranked together, completed events only
    all_time_splits = []
    for y_int in years:
        y = str(y_int)
        for ev in events_by_year.get(y, []):
            eid = ev['id']
            ev_dates = EVENT_DATES.get(eid)
            if not ev_dates or ev_dates[1] >= today_str:
                continue  # skip incomplete events
            for r in data.get(eid, []):
                entry = dict(r)
                entry['split_label'] = ev['label']
                all_time_splits.append(entry)
    all_time_splits.sort(key=lambda r: r['pyth_pct'], reverse=True)
    for i, r in enumerate(all_time_splits):
        r['rank'] = i + 1
    data['all_time_splits'] = all_time_splits

    # All-time internationals: one entry per (team, intl event), completed events only
    JUNK_ORGS = {'tarik','Team','INTL','THAi','fugu','jisou','sergioferra','yjj',
                 'heart bus','karsaj','FRTTT','NaN','nan'}
    mr_intl = mr.copy()
    showmatch_ids = set(mr_intl[mr_intl['MatchName'].str.contains('Showmatch|Main Event', case=False, na=False)]['MatchID'].unique()) if 'MatchName' in mr_intl.columns else set()

    # Load placement data for international events
    _placements_path = os.path.join(ROOT, 'data/intl_placements.json')
    try:
        with open(_placements_path) as _pf:
            _intl_placements = json.load(_pf)
    except Exception:
        _intl_placements = {}
    SLUG_TO_ORG = {
        'paper-rex': 'PRX', 'fnatic': 'FNC', 'loud': 'LOUD', 'kiwoom-drx': 'KRX',
        'natus-vincere': 'NAVI', 'evil-geniuses': 'EG', 'nrg': 'NRG',
        'sentinels': 'SEN', 'gen-g': 'GEN', 'team-heretics': 'TH',
        'leviat-n': 'LEV', 'edward-gaming': 'EDG', 't1': 'T1', 'g2-esports': 'G2',
        'team-vitality': 'VIT', 'wolves-esports': 'WOL', 'xi-lai-gaming': 'XLG',
        'rex-regum-qeon': 'RRQ', 'mibr': 'MIBR', 'giantx': 'GX',
        'nongshim-redforce': 'NS', 'all-gamers': 'AG', 'bbl-esports': 'BBL',
        'gentle-mates': 'M8', 'furia': 'FUR',
    }
    # Build org→place lookup per event
    intl_org_place = {}
    for eid, pdata in _intl_placements.items():
        intl_org_place[eid] = {}
        for s in pdata.get('standings', []):
            org = SLUG_TO_ORG.get(s['slug'], s['slug'].upper()[:4])
            intl_org_place[eid][org] = s['place']

    all_time_intl = []
    for eid, (ev_label, ev_year) in INTL_EVENT_DATES.items():
        ev_dates = EVENT_DATES.get(eid)
        if not ev_dates or ev_dates[1] >= today_str:
            continue  # skip incomplete/future events
        csv_path = os.path.join(ROOT, f'data/maps/{eid}.csv')
        if not os.path.exists(csv_path):
            continue
        idf = pd.read_csv(csv_path)
        idf = idf[~idf['Org'].isin(JUNK_ORGS) & idf['Org'].notna()].copy()
        idf = idf.drop_duplicates(['Player', 'MatchID', 'MapNum'])
        idf = idf[~idf['MatchID'].isin(showmatch_ids)]
        idf['MapNum'] = idf['MapNum'].astype(str)
        idf['MapName'] = idf['MapName'].str.replace('PICK', '', regex=False).str.strip()

        intl_match_ids = set(idf['MatchID'].unique())
        mr_ev = mr_intl[mr_intl['MatchID'].isin(intl_match_ids)].copy()
        if 'MatchName' in mr_ev.columns:
            mr_ev = mr_ev[~mr_ev['MatchName'].str.contains('Showmatch|Main Event', case=False, na=False)]

        orgs_per_i = idf.groupby(['MatchID', 'MapNum']).agg(
            Orgs=('Org', lambda x: list(x.unique())),
            MapName=('MapName', 'first'),
        ).reset_index()

        merged_i = mr_ev.merge(orgs_per_i, on=['MatchID', 'MapNum'], how='inner')

        i_rows = []
        for _, row in merged_i.iterrows():
            orgs   = [o for o in row['Orgs'] if o not in JUNK_ORGS]
            winner = row['WinnerOrg']
            losers = [o for o in orgs if o != winner]
            if not losers:
                continue
            try:
                w_rounds, l_rounds = map(int, str(row['Score']).split('-'))
            except Exception:
                continue
            i_rows.append({
                'year':       int(ev_year),
                'event_id':   eid,
                'map_name':   row.get('MapName', ''),
                'match_id':   int(row['MatchID']),
                'match_name': row.get('MatchName', ''),
                'winner':     winner,
                'loser':      losers[0],
                'wr':         w_rounds,
                'lr':         l_rounds,
            })

        if not i_rows:
            continue
        irdf = pd.DataFrame(i_rows)
        irdf['date'] = ev_dates[0]

        event_results = _build_team_stats(irdf, k, pdf=idf, min_maps=1)
        place_map = intl_org_place.get(eid, {})
        for r in event_results:
            entry = dict(r)
            entry['split_label'] = ev_label
            entry['year'] = int(ev_year)
            entry['placement'] = place_map.get(r['org'], None)
            all_time_intl.append(entry)

    all_time_intl.sort(key=lambda r: r['pyth_pct'], reverse=True)
    for i, r in enumerate(all_time_intl):
        r['rank'] = i + 1
    data['all_time_intl'] = all_time_intl

    incomplete_years = [int(y) for y in years if not year_is_complete(y)]

    return {
        'exponent':         k,
        'k_curve':          {'k': k_vals, 'mse': mse_vals},
        'years':            [int(y) for y in years],
        'incomplete_years': incomplete_years,
        'events_by_year':   events_by_year,
        'event_dates':      {eid: {'start': d[0], 'end': d[1]} for eid, d in EVENT_DATES.items()},
        'intl_events':      INTL_EVENTS,
        'data':             data,
    }


_pyth_cache = None

def get_pyth_data():
    global _pyth_cache
    if _pyth_cache is None:
        _pyth_cache = _compute_pyth_data()
    return _pyth_cache


PW_JS = """
var PW_KEY = 'mapelo_unlocked';
var wrap = document.getElementById('content-wrap');
(function() {
  if (sessionStorage.getItem(PW_KEY) !== '1') {
    wrap.classList.add('blurred');
    var overlay = document.createElement('div');
    overlay.id = 'pw-overlay';
    overlay.style.cssText = 'position:fixed;inset:0;z-index:200;display:flex;align-items:center;justify-content:center;';
    overlay.innerHTML = '<div style="background:white;border-radius:24px;padding:36px 40px;box-shadow:0 8px 48px #00000018;display:flex;flex-direction:column;align-items:center;gap:16px;min-width:280px;">' +
      '<h2 style="font-family:Syne,sans-serif;font-size:1.1rem;font-weight:800;">Enter password</h2>' +
      '<input id="pw-input" type="password" placeholder="Password" autofocus style="width:100%;padding:10px 16px;border-radius:99px;border:2px solid #f0ecf4;font-family:DM Sans,sans-serif;font-size:.95rem;text-align:center;outline:none;">' +
      '<button onclick="checkPw()" style="padding:10px 28px;border-radius:99px;border:none;background:#2a1f2d;color:white;font-family:DM Sans,sans-serif;font-size:.88rem;font-weight:500;cursor:pointer;">Enter</button>' +
      '</div>';
    document.body.appendChild(overlay);
    document.getElementById('pw-input').addEventListener('keydown', function(e){ if(e.key==='Enter') checkPw(); });
  }
})();
function checkPw() {
  var input = document.getElementById('pw-input');
  if (input.value === 'TenZ') {
    sessionStorage.setItem(PW_KEY, '1');
    document.getElementById('pw-overlay').remove();
    wrap.classList.remove('blurred');
  } else {
    input.style.borderColor = '#f4b8c1';
    input.value = '';
    setTimeout(function(){ input.style.borderColor = '#f0ecf4'; }, 400);
  }
}
"""

SHARED_CSS = """
  :root {
    --rose:#f4b8c1; --peach:#f9cba7; --mint:#b8e8d4;
    --sky:#b8d8f4; --lavender:#d4b8f4; --lemon:#f4edb8;
    --cream:#fdf6f0; --ink:#2a1f2d; --soft:#7a6e7e;
  }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--cream); font-family:'DM Sans',sans-serif; color:var(--ink); min-height:100vh; display:flex; flex-direction:column; }
  body::before {
    content:''; position:fixed; inset:0; pointer-events:none; z-index:0;
    background:
      radial-gradient(ellipse 60% 50% at 10% 10%,#f4b8c155 0%,transparent 70%),
      radial-gradient(ellipse 50% 60% at 90% 20%,#b8d8f455 0%,transparent 70%),
      radial-gradient(ellipse 55% 45% at 15% 85%,#b8e8d455 0%,transparent 70%),
      radial-gradient(ellipse 60% 50% at 85% 80%,#d4b8f455 0%,transparent 70%);
  }
  body::after {
    content:''; position:fixed; inset:-50%; pointer-events:none; z-index:0;
    background:
      radial-gradient(ellipse 60% 50% at 60% 55%,#c4a0f099 0%,transparent 55%),
      radial-gradient(ellipse 50% 60% at 38% 42%,#d4a97477 0%,transparent 55%);
    animation:purpleFloat 12s ease-in-out infinite alternate;
  }
  @keyframes purpleFloat {
    0%   { transform:translate(0,0) scale(1); }
    33%  { transform:translate(10%,-9%) scale(1.14); }
    66%  { transform:translate(-9%,12%) scale(0.9); }
    100% { transform:translate(7%,5%) scale(1.1); }
  }
  .top-nav { padding:32px 32px 0; position:relative; z-index:1; }
  .home-logo { height:80px; width:auto; display:block; opacity:.85; transition:opacity .2s; }
  .home-logo:hover { opacity:1; }
  #content-wrap { transition:filter .4s ease; }
  #content-wrap.blurred { filter:blur(12px); pointer-events:none; user-select:none; }
"""

MAPELO_HOME_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VCT Map Model — Bobo's VCT Database</title>
<link rel="icon" type="image/svg+xml" href="/favicon.svg">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  SHARED_CSS
  .page { position:relative; z-index:1; flex:1; display:flex; flex-direction:column; align-items:center; padding:40px 32px 80px; }
  .page-title { font-family:'Syne',sans-serif; font-size:clamp(2rem,6vw,4rem); font-weight:800; letter-spacing:-1.5px; margin-bottom:48px; text-align:center; }
  .section-label { font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:800; color:var(--ink); margin-bottom:16px; text-align:left; width:100%; max-width:800px; letter-spacing:-0.3px; }
  .cards { display:flex; gap:20px; flex-wrap:wrap; justify-content:flex-start; width:100%; max-width:800px; }
  .nav-card { background:white; border-radius:24px; padding:28px 24px; width:260px; text-decoration:none; color:var(--ink); box-shadow:0 4px 24px #0000000a; transition:transform .2s,box-shadow .2s; text-align:left; }
  .nav-card:hover { transform:translateY(-6px); box-shadow:0 16px 40px #00000014; }
  .nav-card-title { font-family:'Syne',sans-serif; font-size:1rem; font-weight:800; margin-bottom:6px; }
  .nav-card-desc { font-size:.8rem; color:var(--soft); font-weight:300; line-height:1.5; }
  .nav-card-arrow { margin-top:16px; font-size:.8rem; color:#ccc; }
</style>
</head>
<body>
<div id="content-wrap">
  <div class="top-nav">
    <a href="/"><img src="/logo.svg" alt="Home" class="home-logo"></a>
  </div>
  <div class="page">
    <div class="page-title">VCT Map Model</div>
    <div class="section-label">Tools</div>
    <div class="cards">
      <a class="nav-card" href="/mapelo/pythagorean/">
        <div class="nav-card-title">Pythagorean Win%</div>
        <div class="nav-card-desc">Expected win% from rounds scored vs. allowed, with the optimal exponent fit to VCT data.</div>
        <div class="nav-card-arrow">Explore &rarr;</div>
      </a>
    </div>
  </div>
</div>
<script>PW_JS</script>
</body>
</html>
""".replace('SHARED_CSS', SHARED_CSS).replace('PW_JS', PW_JS)


MAPELO_PYTH_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pythagorean Win% — VCT Map Model</title>
<link rel="icon" type="image/svg+xml" href="/favicon.svg">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
  SHARED_CSS
  .page { position:relative; z-index:1; padding:32px; max-width:1000px; margin:0 auto; width:100%; }
  .back-link { display:inline-block; font-size:.8rem; color:var(--soft); text-decoration:none; margin-bottom:24px; }
  .back-link:hover { color:var(--ink); }
  .page-title { font-family:'Syne',sans-serif; font-size:clamp(1.6rem,4vw,2.5rem); font-weight:800; letter-spacing:-1px; margin-bottom:28px; }
  .card { background:white; border-radius:24px; padding:28px 32px; box-shadow:0 4px 24px #0000000a; }
  .card-header { display:flex; align-items:baseline; gap:14px; margin-bottom:6px; flex-wrap:wrap; }
  .exponent-badge { font-size:.75rem; font-weight:500; background:#f4edb8; color:#6a5a1a; padding:3px 10px; border-radius:99px; }
  .card-desc { font-size:.82rem; color:var(--soft); line-height:1.6; margin-bottom:18px; }
  .formula-block { background:#f8f4fc; border-radius:14px; padding:14px 20px; margin-bottom:16px; text-align:center; }
  .formula { font-family:Georgia,serif; font-size:1rem; color:var(--ink); margin-bottom:6px; }
  .formula-caption { font-size:.75rem; color:var(--soft); }
  .chart-wrap { margin-bottom:24px; }
  .filter-row { display:flex; align-items:center; gap:8px; margin-bottom:8px; flex-wrap:wrap; }
  .filter-label { font-size:.7rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; color:var(--soft); white-space:nowrap; min-width:52px; }
  .tab-btn { padding:5px 14px; border-radius:99px; border:2px solid #f0ecf4; background:white; font-family:'DM Sans',sans-serif; font-size:.78rem; font-weight:500; cursor:pointer; transition:all .18s; color:var(--soft); white-space:nowrap; }
  .tab-btn:hover { border-color:#d4b8f4; color:var(--ink); }
  .tab-btn.active { background:var(--ink); color:white; border-color:var(--ink); }
  .tab-btn.onwards { border-style:dashed; }
  .tab-btn.onwards.active { border-style:solid; }
  .filter-divider { width:1px; height:20px; background:#f0ecf4; margin:0 4px; }
  .table-wrap { overflow-x:auto; margin-top:20px; }
  table { width:100%; border-collapse:collapse; font-size:.85rem; }
  thead th { font-family:'Syne',sans-serif; font-size:.7rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase; color:var(--soft); padding:8px 12px; text-align:right; border-bottom:2px solid #f0ecf4; cursor:pointer; user-select:none; white-space:nowrap; }
  thead th:nth-child(2) { text-align:left; }
  thead th.sorted-asc::after  { content:' ▲'; font-size:.6rem; }
  thead th.sorted-desc::after { content:' ▼'; font-size:.6rem; }
  tbody tr { border-bottom:1px solid #f8f4fc; transition:background .12s; }
  tbody tr:last-child { border-bottom:none; }
  tbody tr:hover { background:#fdf6f0; }
  td { padding:10px 12px; text-align:right; }
  td:nth-child(2) { text-align:left; }
  .rank-cell { color:var(--soft); font-size:.78rem; width:32px; }
  .org-cell { font-family:'Syne',sans-serif; font-weight:800; font-size:.88rem; cursor:pointer; display:flex; align-items:center; gap:8px; }
  .org-cell:hover .org-name { text-decoration:underline dotted; text-underline-offset:3px; color:#5a3a8a; }
  .team-logo { width:22px; height:22px; object-fit:contain; flex-shrink:0; }
  .wl-cell { color:var(--ink); font-size:.8rem; cursor:pointer; text-decoration:underline dotted; text-underline-offset:3px; }
  .wl-cell:hover { color:#5a3a8a; }
  /* Team card modal */
  .team-modal-header { font-family:'Syne',sans-serif; font-size:1rem; font-weight:800; margin-bottom:20px; display:flex; align-items:center; gap:10px; }
  .team-modal-logo { width:32px; height:32px; object-fit:contain; }
  .map-cards { display:flex; gap:14px; flex-direction:column; margin-top:20px; }
  .map-card { border-radius:16px; overflow:hidden; background:#fdf6f0; }
  .map-card-img { width:100%; height:110px; object-fit:cover; object-position:center; display:block; }
  .map-card-body { padding:12px 16px; }
  .map-card-label { font-size:.6rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; color:var(--soft); margin-bottom:4px; }
  .map-card-name { font-family:'Syne',sans-serif; font-weight:800; font-size:1rem; margin-bottom:10px; }
  .map-card-stats { display:flex; gap:18px; }
  .map-stat { display:flex; flex-direction:column; gap:2px; }
  .map-stat-val { font-family:'Syne',sans-serif; font-weight:800; font-size:.95rem; }
  .map-stat-lbl { font-size:.65rem; color:var(--soft); text-transform:uppercase; letter-spacing:.06em; }
  .map-rd-pos { color:#1a6a4a; }
  .map-rd-neg { color:#7a1a1a; }
  .roster-section { margin-top:20px; }
  .roster-label { font-size:.65rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; color:var(--soft); margin-bottom:10px; }
  .roster-list { display:flex; flex-direction:column; gap:6px; }
  .roster-player { display:flex; align-items:center; gap:10px; text-decoration:none; color:var(--ink); font-size:.85rem; padding:6px 10px; border-radius:10px; transition:background .15s; }
  .roster-player:hover { background:#f8f4fc; }
  .roster-headshot { width:36px; height:36px; border-radius:50%; object-fit:cover; object-position:top; background:#f0ecf4; flex-shrink:0; }
  .roster-player-name { font-family:'Syne',sans-serif; font-weight:800; font-size:.85rem; }
  .pct-cell { font-weight:500; }
  .pyth-cell { font-weight:700; }
  .luck-pos { color:#1a6a4a; font-weight:500; }
  .luck-neg { color:#7a1a1a; font-weight:500; }
  .luck-neu { color:var(--soft); font-weight:500; }
  /* Match modal */
  .modal-backdrop { position:fixed; inset:0; background:#2a1f2daa; backdrop-filter:blur(4px); z-index:300; display:flex; align-items:center; justify-content:center; padding:20px; }
  .modal-box { background:white; border-radius:24px; padding:28px 32px; max-width:480px; width:100%; max-height:80vh; overflow-y:auto; box-shadow:0 24px 60px #0003; position:relative; animation:modalIn .2s ease; }
  @keyframes modalIn { from{opacity:0;transform:scale(.96)} to{opacity:1;transform:scale(1)} }
  .modal-close { position:absolute; top:14px; right:18px; background:none; border:none; font-size:1.4rem; cursor:pointer; color:var(--soft); padding:4px; line-height:1; }
  .modal-title { font-family:'Syne',sans-serif; font-size:1rem; font-weight:800; margin-bottom:16px; }
  .series-group { margin-bottom:14px; }
  .series-group:last-child { margin-bottom:0; }
  .series-header { display:flex; align-items:center; gap:10px; padding-bottom:7px; border-bottom:2px solid #f0ecf4; margin-bottom:4px; }
  .series-result { font-weight:700; font-size:.72rem; padding:2px 8px; border-radius:99px; white-space:nowrap; flex-shrink:0; }
  .series-result.w { background:#d4f4e8; color:#1a5a3a; }
  .series-result.l { background:#fde8e8; color:#7a1a1a; }
  .series-opp { font-family:'Syne',sans-serif; font-weight:800; font-size:.88rem; flex:1; }
  .series-score { font-size:.78rem; color:var(--soft); white-space:nowrap; }
  .map-row { display:grid; grid-template-columns:1fr auto auto; align-items:center; padding:5px 0 5px 10px; border-bottom:1px solid #faf6fc; font-size:.8rem; gap:10px; }
  .map-row:last-child { border-bottom:none; }
  .map-name { color:var(--soft); }
  .map-score { color:var(--soft); font-size:.78rem; white-space:nowrap; }
  .map-diff { font-size:.76rem; font-weight:500; white-space:nowrap; }
  .map-diff.pos { color:#1a6a4a; }
  .map-diff.neg { color:#7a1a1a; }
</style>
</head>
<body>
<div id="content-wrap">
  <div class="top-nav">
    <a href="/"><img src="/logo.svg" alt="Home" class="home-logo"></a>
  </div>
  <div class="page">
    <a class="back-link" href="/mapelo/">&larr; VCT Map Model</a>
    <div class="page-title">Pythagorean Win%</div>
    <div class="card">
      <p class="card-desc"></p>
      <div class="formula-block">
        <div class="formula">Win% &asymp; RW<sup>k</sup> / (RW<sup>k</sup> + RL<sup>k</sup>)</div>
        <div class="formula-caption">RW = rounds won &nbsp;|&nbsp; RL = rounds lost &nbsp;|&nbsp; k = optimal exponent fit to VCT data</div>
      </div>
      <div class="chart-wrap">
        <canvas id="exp-chart" height="90"></canvas>
      </div>

      <div class="filter-row" id="year-row"></div>
      <div class="filter-row" id="alltime-sub" style="display:none"></div>
      <div class="filter-row" id="split-row" style="display:none"></div>

      <div id="alltime-note" style="display:none; font-size:.75rem; color:var(--soft); margin-bottom:8px; font-style:italic;"></div>

      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th data-col="rank">#</th>
              <th data-col="org">Team</th>
              <th data-col="placement" id="th-placement" style="display:none">Place</th>
              <th data-col="wl">W-L</th>
              <th data-col="pyth_pct">Pyth%</th>
              <th data-col="win_pct">Actual Win%</th>
              <th data-col="luck">Luck</th>
              <th data-col="rw">RW</th>
              <th data-col="rl">RL</th>
            </tr>
          </thead>
          <tbody id="pyth-body"></tbody>
        </table>
      </div>
    </div>
  </div>
</div>

<script>
var PYTH = PYTH_JSON;


// Exponent curve chart
(function() {
  var curve = PYTH.k_curve;
  var optK = PYTH.exponent;
  var minDist = Infinity, minIdx = 0;
  curve.k.forEach(function(v, i) { var d = Math.abs(v - optK); if (d < minDist) { minDist = d; minIdx = i; } });
  new Chart(document.getElementById('exp-chart'), {
    type: 'line',
    data: {
      labels: curve.k,
      datasets: [{
        data: curve.mse,
        borderColor: '#d4b8f4', borderWidth: 2,
        pointRadius: curve.k.map(function(_, i) { return i === minIdx ? 6 : 0; }),
        pointBackgroundColor: curve.k.map(function(_, i) { return i === minIdx ? '#2a1f2d' : 'transparent'; }),
        fill: true, backgroundColor: 'rgba(212,184,244,0.08)', tension: 0.4,
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: function(items) { return 'k = ' + items[0].label; },
            label: function(item)  { return 'MSE = ' + item.raw.toFixed(5); }
          }
        }
      },
      scales: {
        x: { title: { display:true, text:'Exponent (k)', font:{family:'DM Sans',size:11}, color:'#7a6e7e' }, ticks:{maxTicksLimit:10,font:{size:10}}, grid:{color:'#f0ecf4'} },
        y: { title: { display:true, text:'Mean Squared Error', font:{family:'DM Sans',size:11}, color:'#7a6e7e' }, ticks:{font:{size:10}}, grid:{color:'#f0ecf4'} }
      }
    }
  });
})();

// ── State ──────────────────────────────────────────────────────────────────
var sortCol = 'pyth_pct', sortDir = -1;
var activeYear = String(PYTH.years[PYTH.years.length - 1]);
var activeKey  = activeYear;
var isAllTime  = false;

// ── Helpers ────────────────────────────────────────────────────────────────
function fmt(v, d) { return (v * 100).toFixed(d) + '%'; }

function makeTab(label, isActive, cls, onClick) {
  var btn = document.createElement('button');
  btn.className = 'tab-btn' + (isActive ? ' active' : '') + (cls ? ' ' + cls : '');
  btn.textContent = label;
  btn.addEventListener('click', onClick);
  return btn;
}

function setKey(key) {
  isAllTime = false;
  activeKey = key;
  buildSplitRow();
  renderTable();
}

// ── Year tabs (incl. All-Time) ──────────────────────────────────────────────
var yearRow  = document.getElementById('year-row');
var yearBtns = [];

PYTH.years.forEach(function(y) {
  var btn = makeTab(String(y), false, '', function() {
    yearBtns.forEach(function(b) { b.classList.remove('active'); });
    allTimeBtn.classList.remove('active');
    btn.classList.add('active');
    isAllTime  = false;
    activeYear = String(y);
    activeKey  = activeYear;
    document.getElementById('alltime-note').style.display = 'none';
    document.getElementById('alltime-sub').style.display  = 'none';
    buildSplitRow();
    renderTable();
  });
  yearBtns.push(btn);
  yearRow.appendChild(btn);
});

// allTimeSplits: false = Full Year, true = By Splits, 'intl' = Internationals
var allTimeSplits = false;

function showAllTime(mode) {
  yearBtns.forEach(function(b) { b.classList.remove('active'); });
  allTimeBtn.classList.add('active');
  isAllTime = true;
  allTimeSplits = mode;
  activeKey = mode === 'intl' ? 'all_time_intl' : (mode ? 'all_time_splits' : 'all_time');
  document.getElementById('split-row').style.display = 'none';
  document.getElementById('alltime-sub').style.display = 'flex';
  allTimeFullBtn.classList.toggle('active', !mode);
  allTimeSplitBtn.classList.toggle('active', mode === true);
  allTimeIntlBtn.classList.toggle('active',  mode === 'intl');
  var note = document.getElementById('alltime-note');
  var incomplete = PYTH.incomplete_years || [];
  if (!mode && incomplete.length) {
    note.textContent = incomplete.join(', ') + (incomplete.length === 1 ? ' is' : ' are') + ' still in progress and not included.';
    note.style.display = 'block';
  } else {
    note.style.display = 'none';
  }
  renderTable();
}

var allTimeBtn = makeTab('All-Time', false, '', function() { showAllTime(false); });
yearRow.appendChild(allTimeBtn);

// All-time sub-row: Full Year / By Splits / Internationals
var allTimeSubRow  = document.getElementById('alltime-sub');
var allTimeFullBtn  = makeTab('Full Year',      true,  '', function() { showAllTime(false);   });
var allTimeSplitBtn = makeTab('By Splits',      false, '', function() { showAllTime(true);    });
var allTimeIntlBtn  = makeTab('Internationals', false, '', function() { showAllTime('intl');  });
allTimeSubRow.appendChild(allTimeFullBtn);
allTimeSubRow.appendChild(allTimeSplitBtn);
allTimeSubRow.appendChild(allTimeIntlBtn);

// ── Split row ─────────────────────────────────────────────────────────────
function buildSplitRow() {
  var splitRow = document.getElementById('split-row');
  splitRow.innerHTML = '';

  var evs = PYTH.events_by_year[activeYear] || [];
  if (evs.length <= 1) { splitRow.style.display = 'none'; return; }

  splitRow.style.display = 'flex';

  var allBtn = makeTab('All', activeKey === activeYear, '', function() { setKey(activeYear); });
  splitRow.appendChild(allBtn);

  var today = new Date().toISOString().slice(0, 10);
  evs.forEach(function(ev) {
    var d = PYTH.event_dates[ev.id];
    if (d && d.end > today) return;
    var btn = makeTab(ev.label, activeKey === ev.id, '', function() { setKey(ev.id); });
    splitRow.appendChild(btn);
  });
}


buildSplitRow();

// ── Sort ───────────────────────────────────────────────────────────────────
document.querySelectorAll('thead th').forEach(function(th) {
  th.addEventListener('click', function() {
    var col = th.dataset.col;
    if (col === 'rank') return;
    if (sortCol === col) { sortDir *= -1; }
    else { sortCol = col; sortDir = col === 'org' ? 1 : -1; }
    renderTable();
  });
});

// ── Render ─────────────────────────────────────────────────────────────────
function placementCell(p, show) {
  var suffix = p === 1 ? 'st' : p === 2 ? 'nd' : p === 3 ? 'rd' : 'th';
  var disp  = show ? '' : 'display:none;';
  var content = p ? (p + '<sup style="font-size:.55em">' + suffix + '</sup>') : '—';
  return '<td class="placement-cell" style="' + disp + 'text-align:right;color:var(--soft)">' + content + '</td>';
}

function renderTable() {
  var isIntl = allTimeSplits === 'intl';
  var thPlace = document.getElementById('th-placement');
  if (thPlace) thPlace.style.display = isIntl ? '' : 'none';

  var rows = (PYTH.data[activeKey] || []).slice();
  rows.sort(function(a, b) {
    var av, bv;
    if (sortCol === 'org') {
      av = (isAllTime || allTimeSplits) ? (String(a.year||'') + ' ' + a.org + ' ' + (a.split_label||'')) : a.org;
      bv = (isAllTime || allTimeSplits) ? (String(b.year||'') + ' ' + b.org + ' ' + (b.split_label||'')) : b.org;
      return sortDir * av.localeCompare(bv);
    }
    if (sortCol === 'placement') {
      av = a.placement !== null && a.placement !== undefined ? a.placement : 9999;
      bv = b.placement !== null && b.placement !== undefined ? b.placement : 9999;
      return sortDir * (av - bv);
    }
    var col = sortCol === 'wl' ? 'win_pct' : sortCol;
    av = a[col] !== undefined ? a[col] : 0;
    bv = b[col] !== undefined ? b[col] : 0;
    return sortDir * (av - bv);
  });
  document.querySelectorAll('thead th').forEach(function(th) {
    th.classList.remove('sorted-asc', 'sorted-desc');
    if (th.dataset.col === sortCol) th.classList.add(sortDir === 1 ? 'sorted-asc' : 'sorted-desc');
  });

  var html = rows.map(function(r, i) {
    var luck    = r.luck;
    var luckCls = luck > 0.01 ? 'luck-pos' : luck < -0.01 ? 'luck-neg' : 'luck-neu';
    var luckStr = (luck >= 0 ? '+' : '') + fmt(luck, 1);
    var displayOrg = allTimeSplits
      ? (r.year + ' ' + r.org + ' ' + (r.split_label || ''))
      : isAllTime ? (r.year + ' ' + r.org) : r.org;
    var logoHtml = '<img src="/logos/' + r.org + '.png" class="team-logo" onerror="this.style.display=&apos;none&apos;">';
    var place = isIntl ? r.placement : null;
    var glowStyle = '';
    if (place === 1) glowStyle = 'text-shadow:0 0 8px #c9960ccc,0 0 2px #c9960c88;color:#8a6200;';
    else if (place === 2) glowStyle = 'text-shadow:0 0 8px #90909099,0 0 2px #90909066;color:#555;';
    else if (place === 3) glowStyle = 'text-shadow:0 0 8px #a0522d88,0 0 2px #a0522d55;color:#7a3a1a;';
    return '<tr>' +
      '<td class="rank-cell">' + (i + 1) + '</td>' +
      '<td class="org-cell" data-org="' + r.org + '" data-year="' + (r.year||activeYear) + '" data-split="' + (r.split_label||'') + '">' +
        logoHtml + '<span class="org-name" style="' + glowStyle + '">' + displayOrg + '</span>' +
      '</td>' +
      placementCell(isIntl ? r.placement : null, isIntl) +
      '<td class="wl-cell" data-org="' + r.org + '" data-year="' + (r.year||activeYear) + '" data-split="' + (r.split_label||'') + '">' + r.wins + '-' + r.losses + '</td>' +
      '<td class="pyth-cell">' + fmt(r.pyth_pct, 1) + '</td>' +
      '<td class="pct-cell">' + fmt(r.win_pct, 1) + '</td>' +
      '<td class="' + luckCls + '">' + luckStr + '</td>' +
      '<td>' + r.rw + '</td>' +
      '<td>' + r.rl + '</td>' +
      '</tr>';
  }).join('');
  document.getElementById('pyth-body').innerHTML = html;

  document.querySelectorAll('.wl-cell').forEach(function(cell) {
    cell.addEventListener('click', function(e) {
      e.stopPropagation();
      openMatchModal(cell.dataset.org, cell.dataset.year, cell.dataset.split);
    });
  });
  document.querySelectorAll('.org-cell').forEach(function(cell) {
    cell.addEventListener('click', function() {
      openTeamModal(cell.dataset.org, cell.dataset.year, cell.dataset.split);
    });
  });
}

// ── Match modal ─────────────────────────────────────────────────────────────
function openMatchModal(org, year, split) {
  var rows  = PYTH.data[activeKey] || [];
  var entry = rows.find(function(r) {
    if (r.org !== org) return false;
    if (String(r.year||activeYear) !== String(year||activeYear)) return false;
    if (allTimeSplits && split) return (r.split_label||'') === split;
    return true;
  });
  if (!entry) return;

  // Sort chronologically by stage priority, then match_id as tiebreaker
  function stagePriority(name) {
    name = (name || '').toLowerCase();
    if (/regular season|league play|group stage.*week|week \d/.test(name)) {
      var wm = name.match(/week (\d+)/); return wm ? parseInt(wm[1]) : 10;
    }
    if (/group stage.*opening|opening/.test(name))    return 20;
    if (/group stage.*winner/.test(name))             return 21;
    if (/group stage.*elimination/.test(name))        return 22;
    if (/group stage.*decider/.test(name))            return 23;
    if (/swiss.*round 1/.test(name))                  return 30;
    if (/swiss.*round 2/.test(name))                  return 31;
    if (/swiss.*round 3/.test(name))                  return 32;
    if (/play.in|bracket.*round of 16/.test(name))    return 40;
    if (/bracket.*quarterfinal|upper.*round 1/.test(name)) return 50;
    if (/bracket.*semifinal|upper.*round 2|lower.*round 1/.test(name)) return 55;
    if (/playoff.*knockout|upper.*quarterfinal/.test(name)) return 57;
    if (/lower.*round 2|upper.*semifinal/.test(name)) return 60;
    if (/lower.*round 3|upper.*final(?! grand)/.test(name)) return 65;
    if (/lower.*round 4/.test(name))                  return 67;
    if (/lower.*round 5|playoff.*semifinal|playoff.*upper.*semifinal/.test(name)) return 70;
    if (/lower.*final|playoff.*lower.*final/.test(name)) return 75;
    if (/middle.*round|playoff.*upper.*final(?! grand)/.test(name)) return 77;
    if (/semifinal/.test(name))                       return 80;
    if (/grand final|championship/.test(name))        return 100;
    return 50;
  }
  var matches = (entry.matches || []).slice().sort(function(a,b){
    var pa = stagePriority(a.match_name), pb = stagePriority(b.match_name);
    if (pa !== pb) return pa - pb;
    return (a.match_id||0) - (b.match_id||0);
  });

  // Group maps by match_id, preserving order
  var seriesOrder = [];
  var seriesMap = {};
  matches.forEach(function(m) {
    var id = m.match_id;
    if (!seriesMap[id]) { seriesMap[id] = []; seriesOrder.push(id); }
    seriesMap[id].push(m);
  });

  function buildSeries(mid) {
    var maps = seriesMap[mid];
    var opponent = maps[0].opponent;
    var seriesWins   = maps.filter(function(m){ return  m.win; }).length;
    var seriesLosses = maps.filter(function(m){ return !m.win; }).length;
    var seriesWon    = seriesWins > seriesLosses;
    var resCls = seriesWon ? 'w' : 'l';
    var resLbl = seriesWon ? 'W' : 'L';

    var header = '<div class="series-header">' +
      '<span class="series-result ' + resCls + '">' + resLbl + '</span>' +
      '<span class="series-opp">' + opponent + '</span>' +
      '<span class="series-score">' + seriesWins + '–' + seriesLosses + '</span>' +
      '</div>';

    var mapRows = maps.map(function(m) {
      var diff    = m.diff >= 0 ? '+' + m.diff : String(m.diff);
      var diffCls = m.diff > 0 ? 'pos' : m.diff < 0 ? 'neg' : '';
      var mapName = (m.map && m.map !== 'Unknown') ? m.map : '—';
      return '<div class="map-row">' +
        '<span class="map-name">' + mapName + '</span>' +
        '<span class="map-score">' + m.score + '</span>' +
        '<span class="map-diff ' + diffCls + '">' + diff + '</span>' +
        '</div>';
    }).join('');

    return '<div class="series-group">' + header + mapRows + '</div>';
  }

  var html = seriesOrder.slice().reverse().map(buildSeries).join('');

  var backdrop = document.createElement('div');
  backdrop.className = 'modal-backdrop';
  backdrop.innerHTML = '<div class="modal-box">' +
    '<button class="modal-close">&times;</button>' +
    '<div class="modal-title">' + year + ' ' + org + ' &mdash; Map Results</div>' +
    html + '</div>';
  backdrop.querySelector('.modal-close').addEventListener('click', function() { backdrop.remove(); });
  backdrop.addEventListener('click', function(e) { if (e.target === backdrop) backdrop.remove(); });
  document.body.appendChild(backdrop);
}

// ── Team map modal ────────────────────────────────────────────────────────
function openTeamModal(org, year, split) {
  var rows  = PYTH.data[activeKey] || [];
  var entry = rows.find(function(r) {
    if (r.org !== org) return false;
    if (String(r.year||activeYear) !== String(year||activeYear)) return false;
    if (allTimeSplits && split) return (r.split_label||'') === split;
    return true;
  });
  if (!entry) return;

  var mapStats = (entry.map_stats || []).filter(function(m) { return m.map && m.map !== 'Unknown'; });
  if (!mapStats.length) return;

  var best  = mapStats[0];
  var worst = mapStats[mapStats.length - 1];

  function mapCard(ms, label) {
    var rdCls = ms.rd >= 0 ? 'map-rd-pos' : 'map-rd-neg';
    var rdStr = (ms.rd >= 0 ? '+' : '') + ms.rd;
    return '<div class="map-card">' +
      '<img class="map-card-img" src="/maps/' + ms.map.toLowerCase() + '.png" onerror="this.style.display=&apos;none&apos;">' +
      '<div class="map-card-body">' +
        '<div class="map-card-label">' + label + '</div>' +
        '<div class="map-card-name">' + ms.map + '</div>' +
        '<div class="map-card-stats">' +
          '<div class="map-stat"><div class="map-stat-val">' + ms.wins + '-' + ms.losses + '</div><div class="map-stat-lbl">W-L</div></div>' +
          '<div class="map-stat"><div class="map-stat-val">' + fmt(ms.rw_pct, 1) + '</div><div class="map-stat-lbl">RW%</div></div>' +
          '<div class="map-stat"><div class="map-stat-val ' + rdCls + '">' + rdStr + '</div><div class="map-stat-lbl">Round Diff</div></div>' +
        '</div>' +
      '</div></div>';
  }

  var roster = entry.roster || [];
  var rosterHtml = '';
  if (roster.length) {
    var playerItems = roster.map(function(p) {
      var name = p.player || '';
      var url  = p.url || '';
      var hs   = p.headshot || '';
      var imgTag = hs
        ? '<img class="roster-headshot" src="' + hs + '" alt="' + name + '" onerror="this.style.visibility=&apos;hidden&apos;">'
        : '<div class="roster-headshot" style="background:#e8e4f0;"></div>';
      return '<a class="roster-player" href="' + url + '" target="_blank" rel="noopener">' +
        imgTag +
        '<span class="roster-player-name">' + name + '</span>' +
        '</a>';
    }).join('');
    rosterHtml = '<div class="roster-section"><div class="roster-label">Roster</div><div class="roster-list">' + playerItems + '</div></div>';
  }

  var logoTag = '<img src="/logos/' + org + '.png" class="team-modal-logo" onerror="this.style.display=&apos;none&apos;">';
  var backdrop = document.createElement('div');
  backdrop.className = 'modal-backdrop';
  backdrop.innerHTML = '<div class="modal-box">' +
    '<button class="modal-close">&times;</button>' +
    '<div class="team-modal-header">' + logoTag + year + ' ' + org + '</div>' +
    rosterHtml +
    '<div class="map-cards">' +
      mapCard(best, 'Best Map') +
      (best !== worst ? mapCard(worst, 'Worst Map') : '') +
    '</div></div>';
  backdrop.querySelector('.modal-close').addEventListener('click', function() { backdrop.remove(); });
  backdrop.addEventListener('click', function(e) { if (e.target === backdrop) backdrop.remove(); });
  document.body.appendChild(backdrop);
}

showAllTime(false);
</script>
</body>
</html>
""".replace('SHARED_CSS', SHARED_CSS)


@mapelo_bp.route('/')
def mapelo_home():
    return MAPELO_HOME_HTML

@mapelo_bp.route('/pythagorean/')
def mapelo_pythagorean():
    data = get_pyth_data()
    return MAPELO_PYTH_HTML.replace('PYTH_JSON', json.dumps(data))
