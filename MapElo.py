import os
import re
import json
import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
from flask import Blueprint, Response

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
_PYTH_JSON_PATH = os.path.join(ROOT, 'data', 'pyth_data.json')

_ratings_cache = None
_RATINGS_JSON_PATH = os.path.join(ROOT, 'data', 'map_ratings.json')

def get_ratings():
    global _ratings_cache
    if _ratings_cache is None:
        with open(_RATINGS_JSON_PATH) as f:
            _ratings_cache = json.load(f)
    return _ratings_cache

_veto_cache = None
_VETO_JSON_PATH = os.path.join(ROOT, 'data', 'veto_model.json')

def get_veto_model():
    global _veto_cache
    if _veto_cache is None:
        with open(_VETO_JSON_PATH) as f:
            _veto_cache = json.load(f)
    return _veto_cache

_intl_cache = None
_INTL_JSON_PATH = os.path.join(ROOT, 'data', 'intl_calibration.json')

def get_intl_calibration():
    global _intl_cache
    if _intl_cache is None:
        with open(_INTL_JSON_PATH) as f:
            _intl_cache = json.load(f)
    return _intl_cache

# Static region lookup — teams that have competed in VCT franchised play
ORG_REGIONS = {
    "TL":   "EMEA",  "FNC":  "EMEA",  "NAVI": "EMEA",  "VIT":  "EMEA",
    "BBL":  "EMEA",  "GX":   "EMEA",  "KC":   "EMEA",  "TH":   "EMEA",
    "FUT":  "EMEA",  "GIA":  "EMEA",  "MKOI": "EMEA",  "WOL":  "EMEA",
    "M8":   "EMEA",  "FPX":  "EMEA",  "BME":  "EMEA",
    "SEN":  "Americas",  "G2":   "Americas",  "MIBR": "Americas",
    "NRG":  "Americas",  "100T": "Americas",  "C9":   "Americas",
    "EG":   "Americas",  "KRÜ":  "Americas",  "LEV":  "Americas",
    "FUR":  "Americas",  "LOUD": "Americas",  "2G":   "Americas",
    "APK":  "Americas",
    "PRX":  "Pacific",  "DRX":  "Pacific",  "T1":   "Pacific",
    "TLN":  "Pacific",  "GEN":  "Pacific",  "DFM":  "Pacific",
    "ZETA": "Pacific",  "RRQ":  "Pacific",  "TS":   "Pacific",
    "GE":   "Pacific",  "NS":   "Pacific",
}

def get_pyth_data():
    global _pyth_cache
    if _pyth_cache is None:
        if os.path.exists(_PYTH_JSON_PATH):
            with open(_PYTH_JSON_PATH) as f:
                _pyth_cache = json.load(f)
        else:
            _pyth_cache = _compute_pyth_data()
    return _pyth_cache


_SNAPSHOT_EVENTS = {
    '2023': {
        'after_tokyo':      ['2023_lock_in', '2023_masters_tokyo'],
        'before_champions': ['2023_lock_in', '2023_masters_tokyo', '2023_league'],
        'after_champions':  ['2023_lock_in', '2023_masters_tokyo', '2023_league', '2023_champions'],
    },
    '2024': {
        'before_madrid':    ['2024_kickoff'],
        'after_madrid':     ['2024_kickoff', '2024_masters_madrid'],
        'before_shanghai':  ['2024_kickoff', '2024_masters_madrid', '2024_stage1'],
        'after_shanghai':   ['2024_kickoff', '2024_masters_madrid', '2024_stage1', '2024_masters_shanghai'],
        'before_champions': ['2024_kickoff', '2024_masters_madrid', '2024_stage1', '2024_masters_shanghai', '2024_stage2'],
        'after_champions':  ['2024_kickoff', '2024_masters_madrid', '2024_stage1', '2024_masters_shanghai', '2024_stage2', '2024_champions'],
    },
    '2025': {
        'before_bangkok':   ['2025_kickoff'],
        'after_bangkok':    ['2025_kickoff', '2025_masters_bangkok'],
        'before_toronto':   ['2025_kickoff', '2025_masters_bangkok', '2025_stage1'],
        'after_toronto':    ['2025_kickoff', '2025_masters_bangkok', '2025_stage1', '2025_masters_toronto'],
        'before_champions': ['2025_kickoff', '2025_masters_bangkok', '2025_stage1', '2025_masters_toronto', '2025_stage2'],
        'after_champions':  ['2025_kickoff', '2025_masters_bangkok', '2025_stage1', '2025_masters_toronto', '2025_stage2', '2025_champions'],
    },
    '2026': {
        'before_santiago': ['2026_kickoff'],
        'after_santiago':  ['2026_kickoff', '2026_masters_santiago'],
    },
}

_map_name_index   = None
_headshots_cache  = None
_TEAM_INFO_VER    = 2   # bump this to bust _team_info_cache across all keys
_team_info_cache  = {}

def _get_headshots():
    global _headshots_cache
    if _headshots_cache is None:
        path = os.path.join(ROOT, 'data', 'headshots.json')
        _headshots_cache = json.load(open(path)) if os.path.exists(path) else {}
    return _headshots_cache


def _build_map_name_index():
    global _map_name_index
    if _map_name_index is not None:
        return _map_name_index
    from MoreTestingMaybeFiles import ALL_EVENTS
    _map_name_index = {}
    for event in ALL_EVENTS:
        path = os.path.join(ROOT, 'data', 'maps', f"{event['id']}.csv")
        if not os.path.exists(path):
            continue
        try:
            mdf = pd.read_csv(path, usecols=['MatchID', 'MapNum', 'MapName'])
            for _, row in mdf.drop_duplicates(['MatchID', 'MapNum']).iterrows():
                try:
                    key = (int(row['MatchID']), int(row['MapNum']))
                    name = re.sub(r'(?i)(PICK|BAN|REMAINS?|DECIDER)$', '', str(row['MapName'])).strip()
                    _map_name_index[key] = name
                except (ValueError, TypeError):
                    pass
        except Exception:
            pass
    return _map_name_index


def _get_team_info(org, year='2025', snap='after_champions'):
    cache_key = (_TEAM_INFO_VER, org, year, snap)
    if cache_key in _team_info_cache:
        return _team_info_cache[cache_key]

    data_dir    = os.path.join(ROOT, 'data')
    snap_events = _SNAPSHOT_EVENTS.get(year, {}).get(snap, [])
    headshots   = _get_headshots()

    # Roster from the last event in the snapshot that has data for this org
    roster = []
    for event_id in reversed(snap_events):
        path = os.path.join(data_dir, f'{event_id}.csv')
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path, usecols=['Org', 'Player', 'ProfileURL'])
            rows = df[df['Org'] == org].drop_duplicates('Player').sort_values('Player')
            if not rows.empty:
                for _, r in rows.iterrows():
                    roster.append({
                        'name':     r['Player'],
                        'headshot': headshots.get(r['ProfileURL'], ''),
                    })
                break
        except Exception:
            continue

    # Recent matches: only from snapshot events, most recent first
    recent_matches = []
    mr_path = os.path.join(data_dir, 'match_results.csv')
    if os.path.exists(mr_path) and snap_events:
        mr      = pd.read_csv(mr_path)
        mr_all  = mr[mr['MapNum'] == 'all'].set_index('MatchID')
        mr_maps = mr[mr['MapNum'] != 'all']
        mni     = _build_map_name_index()
        seen    = set()

        for event_id in reversed(snap_events):  # most recent event first
            series_path = os.path.join(data_dir, 'series', f'{event_id}.csv')
            if not os.path.exists(series_path):
                continue
            try:
                sdf = pd.read_csv(series_path, usecols=['Org', 'MatchID'])
            except Exception:
                continue

            # Find event label for display
            from MoreTestingMaybeFiles import ALL_EVENTS
            ev_label = next((e.get('label', event_id) for e in ALL_EVENTS if e['id'] == event_id), event_id)

            # CSV row order is chronological; MatchIDs are NOT (VLR assigns them at creation, not play time)
            for mid in reversed(sdf[sdf['Org'] == org]['MatchID'].unique().tolist()):
                if mid in seen or mid not in mr_all.index:
                    continue
                seen.add(mid)
                sr       = mr_all.loc[mid]
                opponent = next((o for o in sdf[sdf['MatchID'] == mid]['Org'].unique() if o != org), '?')
                won      = (sr['WinnerOrg'] == org)

                maps = []
                for _, mrow in mr_maps[mr_maps['MatchID'] == mid].sort_values('MapNum').iterrows():
                    try:
                        map_name = mni.get((int(mid), int(mrow['MapNum'])), '')
                    except (ValueError, TypeError):
                        map_name = ''
                    maps.append({
                        'map_name': map_name,
                        'score':    str(mrow['Score']),
                        'result':   'W' if mrow['WinnerOrg'] == org else 'L',
                    })

                recent_matches.append({
                    'match_id':      int(mid),
                    'event_label':   ev_label,
                    'match_name':    str(sr.get('MatchName', '') or ''),
                    'opponent':      str(opponent),
                    'series_score':  str(sr['Score']),
                    'series_result': 'W' if won else 'L',
                    'maps':          maps,
                })
                if len(recent_matches) >= 3:
                    break
            if len(recent_matches) >= 3:
                break

    result = {'roster': roster, 'recent_matches': recent_matches}
    _team_info_cache[cache_key] = result
    return result


def _get_map_matches(org, map_name, year='2025', snap='after_champions'):
    cache_key = (_TEAM_INFO_VER, 'map_matches', org, map_name, year, snap)
    if cache_key in _team_info_cache:
        return _team_info_cache[cache_key]

    data_dir    = os.path.join(ROOT, 'data')
    snap_events = _SNAPSHOT_EVENTS.get(year, {}).get(snap, [])
    matches     = []

    mr_path = os.path.join(data_dir, 'match_results.csv')
    if not os.path.exists(mr_path) or not snap_events:
        result = {'map_name': map_name, 'matches': matches}
        _team_info_cache[cache_key] = result
        return result

    from MoreTestingMaybeFiles import ALL_EVENTS
    mr      = pd.read_csv(mr_path)
    mr_all  = mr[mr['MapNum'] == 'all'].set_index('MatchID')
    mr_maps = mr[mr['MapNum'] != 'all']
    mni     = _build_map_name_index()

    for event_id in snap_events:  # chronological; reversed at end
        series_path = os.path.join(data_dir, 'series', f'{event_id}.csv')
        if not os.path.exists(series_path):
            continue
        try:
            sdf = pd.read_csv(series_path, usecols=['Org', 'MatchID'])
        except Exception:
            continue

        ev_label = next((e.get('label', event_id) for e in ALL_EVENTS if e['id'] == event_id), event_id)
        # CSV order is chronological; MatchIDs are NOT
        org_mids = sdf[sdf['Org'] == org]['MatchID'].unique().tolist()

        for mid in org_mids:
            if mid not in mr_all.index:
                continue
            for _, mrow in mr_maps[mr_maps['MatchID'] == mid].iterrows():
                try:
                    mn = mni.get((int(mid), int(mrow['MapNum'])), '')
                except (ValueError, TypeError):
                    continue
                if mn.lower() != map_name.lower():
                    continue

                sr       = mr_all.loc[mid]
                opponent = next((o for o in sdf[sdf['MatchID'] == mid]['Org'].unique() if o != org), '?')
                won      = (str(mrow.get('WinnerOrg', '')) == org)
                score_str = str(mrow['Score'])
                round_diff = 0
                try:
                    a, b = [int(x) for x in score_str.split('-')]
                    round_diff = (a - b) if won else (b - a)
                except Exception:
                    pass

                matches.append({
                    'match_id':    int(mid),
                    'event_label': str(ev_label),
                    'match_name':  str(sr.get('MatchName', '') or ''),
                    'opponent':    str(opponent),
                    'result':      'W' if won else 'L',
                    'score':       score_str,
                    'round_diff':  round_diff,
                })
                break  # one game per map per match

    matches.reverse()  # most recent first
    result = {'map_name': map_name, 'matches': matches}
    _team_info_cache[cache_key] = result
    return result


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
  .top-nav { padding:32px 32px 0; position:relative; z-index:1; display:flex; flex-direction:column; align-items:flex-start; gap:10px; }
  .home-logo { height:80px; width:auto; display:block; opacity:.85; transition:opacity .2s; }
  .home-logo:hover { opacity:1; }
  .back-link { display:inline-flex; align-items:center; gap:4px; font-family:'Syne',sans-serif; font-size:.8rem; font-weight:800; letter-spacing:.03em; color:var(--soft); text-decoration:none; padding:5px 14px 5px 10px; border:1.5px solid #ece6f4; border-radius:99px; background:rgba(255,255,255,.75); backdrop-filter:blur(4px); transition:all .15s; }
  .back-link:hover { color:var(--ink); border-color:#d4b8f4; background:white; }
  #content-wrap { transition:filter .4s ease; }
  #content-wrap.blurred { filter:blur(12px); pointer-events:none; user-select:none; }
"""

MAPELO_HUB_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BenPom &mdash; Bobo's VCT Database</title>
<link rel="icon" type="image/svg+xml" href="/favicon.svg">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  SHARED_CSS
  .hub-page { position:relative; z-index:1; padding:48px 32px; max-width:700px; margin:0 auto; }
  .hub-title { font-family:'Syne',sans-serif; font-size:clamp(1.8rem,4vw,3rem); font-weight:800; letter-spacing:-1px; margin-bottom:6px; }
  .hub-sub { font-size:.85rem; color:var(--soft); margin-bottom:36px; }
  .hub-cards { display:flex; gap:20px; flex-wrap:wrap; }
  .hub-card { background:white; border-radius:24px; padding:28px 24px; width:280px; text-decoration:none; color:var(--ink); box-shadow:0 4px 24px #0000000a; transition:transform .2s,box-shadow .2s; }
  .hub-card:hover { transform:translateY(-5px); box-shadow:0 16px 40px #00000014; }
  .hub-card-title { font-family:'Syne',sans-serif; font-size:1.05rem; font-weight:800; margin-bottom:8px; }
  .hub-card-desc { font-size:.82rem; color:var(--soft); line-height:1.55; }
  .hub-card-arrow { margin-top:18px; font-size:.8rem; color:#ccc; }
</style>
</head>
<body>
<div id="content-wrap">
  <div class="top-nav">
    <a href="/"><img src="/logo.svg" alt="Home" class="home-logo"></a>
  </div>
  <div class="hub-page">
    <div class="hub-title">BenPom</div>
    <p class="hub-sub">Opponent-adjusted map ratings for VCT franchised teams, 2023&ndash;2025.</p>
    <div class="hub-cards">
      <a class="hub-card" href="/mapelo/rankings/">
        <div class="hub-card-title">Rankings</div>
        <div class="hub-card-desc">Per-map Massey ratings with decay, James&ndash;Stein shrinkage, and pick/ban-adjusted overall scores.</div>
        <div class="hub-card-arrow">Explore &rarr;</div>
      </a>
      <a class="hub-card" href="/mapelo/matchup/">
        <div class="hub-card-title">Matchup Predictor</div>
        <div class="hub-card-desc">Monte Carlo BO3 veto simulation &mdash; pick two teams and see head-to-head win probability with a map-by-map breakdown.</div>
        <div class="hub-card-arrow">Explore &rarr;</div>
      </a>
    </div>
  </div>
</div>
<script>PW_JS</script>
</body>
</html>
""".replace('SHARED_CSS', SHARED_CSS).replace('PW_JS', PW_JS)

MAPELO_HOME_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BenPom &mdash; Bobo's VCT Database</title>
<link rel="icon" type="image/svg+xml" href="/favicon.svg">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
  SHARED_CSS
  .page { position:relative; z-index:1; padding:32px; max-width:900px; margin:0 auto; width:100%; }
  .page-title { font-family:'Syne',sans-serif; font-size:clamp(1.6rem,4vw,2.8rem); font-weight:800; letter-spacing:-1px; margin-bottom:6px; }
  .page-sub  { font-size:.83rem; color:var(--soft); margin-bottom:22px; line-height:1.5; }
  /* Model explanation + animated pipeline */
  .model-card { background:white; border-radius:24px; padding:24px 28px; box-shadow:0 4px 24px #0000000a; margin-bottom:20px; }
  .model-card-header { display:flex; align-items:center; justify-content:space-between; margin-bottom:0; }
  .model-card-title { font-family:'Syne',sans-serif; font-size:.85rem; font-weight:800; letter-spacing:.04em; text-transform:uppercase; color:var(--soft); }
  .model-card-toggle { background:none; border:none; cursor:pointer; font-family:'Syne',sans-serif; font-size:.65rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; color:#9a7ab4; display:flex; align-items:center; gap:6px; padding:0; }
  .model-card-toggle .toggle-arrow { display:inline-block; transition:transform .2s; font-style:normal; }
  .model-card-toggle.open .toggle-arrow { transform:rotate(90deg); }
  .model-collapsible { overflow:hidden; transition:max-height .6s ease, opacity .3s ease; max-height:0; opacity:0; }
  .model-collapsible.open { max-height:2400px; opacity:1; overflow:visible; }
  /* Pipeline */
  .pipeline-wrap { position:relative; padding:20px 0 4px; }
  .pipe-stage { display:flex; gap:16px; align-items:flex-start; padding:14px 16px; border-radius:16px; cursor:pointer; transition:background .2s, box-shadow .2s; position:relative; z-index:1; }
  .pipe-stage:hover { background:#faf8fc; }
  .pipe-stage.active { background:linear-gradient(135deg,#f3ecfc,#fff); box-shadow:0 0 0 1.5px #c89ee8, 0 4px 16px #9a4ab41a; }
  .pipe-stage.active-current { box-shadow:0 0 0 2px #a060d0, 0 6px 28px #9a4ab438 !important; }
  @keyframes stageGlow { 0%{box-shadow:0 0 0 2px #a060d0,0 6px 28px #9a4ab438} 50%{box-shadow:0 0 0 2.5px #8040c0,0 8px 36px #9a4ab450} 100%{box-shadow:0 0 0 2px #a060d0,0 6px 28px #9a4ab438} }
  .pipe-stage.active-current { animation:stageGlow 2s ease-in-out infinite; }
  .pipe-num { flex-shrink:0; width:36px; height:36px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-family:'Syne',sans-serif; font-weight:800; font-size:.85rem; color:white; transition:transform .3s, box-shadow .3s; }
  .pipe-stage.active .pipe-num { transform:scale(1.12); box-shadow:0 4px 12px #9a4ab444; }
  .pipe-n0 { background:linear-gradient(135deg,#e8a060,#d4804a); }
  .pipe-n1 { background:linear-gradient(135deg,#7a60e8,#5a3ab4); }
  .pipe-n2 { background:linear-gradient(135deg,#60a8e8,#3a78c8); }
  .pipe-n3 { background:linear-gradient(135deg,#60c8a0,#3a9470); }
  .pipe-n4 { background:linear-gradient(135deg,#e860a8,#b43a78); }
  .pipe-n5 { background:linear-gradient(135deg,#e8c060,#c89a30); }
  .pipe-n6 { background:linear-gradient(135deg,#9a4ab4,#5a2a7a); }
  .pipe-content { flex:1; min-width:0; }
  .pipe-title { font-family:'Syne',sans-serif; font-size:.88rem; font-weight:800; color:var(--ink); margin-bottom:3px; transition:color .2s; }
  .pipe-stage.active .pipe-title { color:#5a2a7a; }
  .pipe-desc { font-size:.79rem; color:var(--soft); line-height:1.6; max-height:0; overflow:hidden; opacity:0; transition:max-height .4s ease, opacity .35s ease; }
  .pipe-stage.active .pipe-desc { max-height:200px; opacity:1; }
  .pipe-desc code { background:#efe8f8; border-radius:4px; padding:1px 5px; font-size:.73rem; color:#5a2a7a; }
  /* Stage graphics */
  .pipe-graphic { max-height:0; overflow:hidden; opacity:0; margin-top:8px; transition:max-height .5s ease .1s, opacity .4s ease .15s; }
  .pipe-stage.active .pipe-graphic { max-height:340px; opacity:1; }
  .pg-note { font-size:.64rem; color:var(--soft); padding-top:4px; }
  /* Score bars (Stage 1) */
  .pg-scorebar { display:flex; flex-direction:column; gap:7px; padding:4px 0 2px; }
  .pg-score-row { display:flex; align-items:center; gap:10px; }
  .pg-score-label { font-family:'Syne',sans-serif; font-size:.72rem; font-weight:800; color:var(--soft); width:34px; text-align:right; flex-shrink:0; }
  .pg-bar-track { flex:1; height:10px; background:#f0ecf8; border-radius:5px; overflow:hidden; }
  .pg-bar-fill { height:100%; border-radius:5px; width:0; transition:width 1.1s cubic-bezier(.4,0,.2,1); }
  .pg-bar-big { background:linear-gradient(90deg,#a060d0,#d080f8); }
  .pg-bar-small { background:linear-gradient(90deg,#c8b0e0,#dcccea); }
  .pg-score-diff { font-family:'Syne',sans-serif; font-size:.72rem; font-weight:800; width:28px; flex-shrink:0; }
  .pg-score-diff-big { color:#5a2a7a; } .pg-score-diff-small { color:#b4a0c8; }
  /* Decay canvas (Stage 3) */
  .pg-decay-wrap { padding:4px 0 2px; }
  .pg-decay-canvas { display:block; width:100%; height:120px; }
  /* Map veto (Stage 5) */
  .pg-veto { display:flex; flex-wrap:wrap; gap:5px; padding:6px 0 2px; }
  .pg-map-chip { font-family:'Syne',sans-serif; font-size:.65rem; font-weight:800; padding:3px 9px; border-radius:6px; background:#f0ecf8; color:var(--ink); transition:all .4s; }
  .pg-map-chip.banned { background:#f5e8e8; color:#c07070; text-decoration:line-through; opacity:.45; }
  .pg-map-chip.picked { background:linear-gradient(135deg,#a060d0,#7040a0); color:white; box-shadow:0 2px 8px #9a4ab455; transform:scale(1.06); }
  .pg-map-chip.float  { background:linear-gradient(135deg,#f0e8ff,#e0d0f8); color:#5a2a7a; box-shadow:0 0 0 1.5px #c8a0e8; }
  .pg-map-chip.dimmed { opacity:.3; }
  /* Region offsets (Stage 6) */
  .pg-regions { display:flex; gap:10px; padding:6px 6px 2px 6px; align-items:center; flex-wrap:wrap; }
  .pg-region { display:flex; flex-direction:column; align-items:center; gap:4px; width:66px; }
  .pg-region-name { font-family:'Syne',sans-serif; font-size:.63rem; font-weight:800; color:var(--soft); }
  .pg-region-bubble { width:50px; height:50px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-family:'Syne',sans-serif; font-size:.72rem; font-weight:800; transition:transform .5s, box-shadow .5s; }
  .pg-region-bubble.r-emea { background:linear-gradient(135deg,#ebe5ff,#d8ccf8); color:#5a2a9a; }
  .pg-region-bubble.r-am   { background:linear-gradient(135deg,#e5f5e5,#c8edd8); color:#2a6a4a; }
  .pg-region-bubble.r-pac  { background:linear-gradient(135deg,#e5f0ff,#c8daf8); color:#2a4a9a; }
  .pg-region-bubble.show { transform:scale(1.08); box-shadow:0 4px 14px #9a4ab440; }
  .pg-intl-arrow { font-size:.9rem; color:#c8b8e0; margin-bottom:15px; }
  /* Formula assembly (Stage 7) */
  .pg-formula { display:flex; align-items:center; gap:5px; padding:8px 0 2px; flex-wrap:wrap; }
  .pg-formula-part { font-family:'Syne',sans-serif; font-size:.72rem; font-weight:800; padding:4px 10px; border-radius:8px; opacity:0; transform:translateY(6px); transition:opacity .45s, transform .45s; }
  .pg-formula-part.show { opacity:1; transform:translateY(0); }
  .pg-formula-dom { background:#f0ecf8; color:var(--ink); }
  .pg-formula-op  { background:none; color:var(--soft); padding:4px 3px; }
  .pg-formula-intl { background:#e8f0f8; color:#2a4a9a; }
  .pg-formula-global { background:linear-gradient(135deg,#9a4ab4,#5a2a7a); color:white; box-shadow:0 2px 8px #9a4ab455; }
  /* Connector — redesigned as glowing data tube */
  .pipe-connector { position:relative; margin-left:34px; width:6px; height:42px; border-radius:3px; background:#ede8f4; overflow:hidden; transition:background .6s, box-shadow .6s; }
  .pipe-connector.lit { background:linear-gradient(to bottom,#b870e8,#6a30a0); box-shadow:0 0 10px #9a4ab468; }
  .pipe-particle { position:absolute; left:0; right:0; height:12px; border-radius:6px; background:linear-gradient(to bottom,#f0d0ff,#b840e8); opacity:0; }
  .pipe-particle-b { height:8px; background:linear-gradient(to bottom,#d8b8f8,#8830c8); }
  @keyframes particleFlow  { 0%{top:-14px;opacity:0} 12%{opacity:1} 88%{opacity:.9} 100%{top:48px;opacity:0} }
  @keyframes particleFlowB { 0%{top:-10px;opacity:0} 15%{opacity:.65} 85%{opacity:.65} 100%{top:48px;opacity:0} }
  .pipe-particle.flowing   { animation:particleFlow  .62s ease-in-out forwards; }
  .pipe-particle-b.flowing { animation:particleFlowB .62s ease-in-out .22s forwards; }
  /* Tech details / stats */
  .model-stats { display:flex; gap:12px; flex-wrap:wrap; margin:18px 0 0; }
  .stat-pill { background:#f8f4fc; border-radius:99px; padding:6px 16px; font-size:.78rem; display:flex; gap:6px; align-items:center; }
  .stat-pill-label { color:var(--soft); }
  .stat-pill-value { font-family:'Syne',sans-serif; font-weight:800; color:var(--ink); }
  .stat-pill-value.good { color:#1a6a4a; }
  .chart-section { margin-top:18px; }
  .chart-title { font-family:'Syne',sans-serif; font-size:.78rem; font-weight:800; color:var(--soft); letter-spacing:.04em; text-transform:uppercase; margin-bottom:10px; }
  .chart-wrap { position:relative; height:160px; }
  .pipe-replay-btn { background:none; border:1.5px solid #e0d8ec; border-radius:99px; padding:4px 14px; font-family:'Syne',sans-serif; font-size:.62rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase; color:#9a7ab4; cursor:pointer; transition:all .15s; }
  .pipe-replay-btn:hover { border-color:#c89ee8; color:#5a2a7a; background:#f8f4fc; }
  /* Filters */
  .filter-row { display:flex; align-items:center; gap:8px; margin-bottom:10px; flex-wrap:wrap; }
  .filter-row-maps { display:flex; align-items:center; gap:8px; margin-bottom:16px; flex-wrap:wrap; }
  .filter-label { font-size:.7rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; color:var(--soft); white-space:nowrap; min-width:44px; flex-shrink:0; }
  .period-select { appearance:none; -webkit-appearance:none; padding:5px 32px 5px 14px; border-radius:99px; border:2px solid #f0ecf4; background:white url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%237a6e7e'/%3E%3C/svg%3E") no-repeat right 12px center; font-family:'DM Sans',sans-serif; font-size:.78rem; font-weight:500; color:var(--ink); cursor:pointer; transition:border-color .18s; outline:none; }
  .period-select:hover, .period-select:focus { border-color:#d4b8f4; }
  .tab-btn { padding:5px 14px; border-radius:99px; border:2px solid #f0ecf4; background:white; font-family:'DM Sans',sans-serif; font-size:.78rem; font-weight:500; cursor:pointer; transition:all .18s; color:var(--soft); white-space:nowrap; }
  .tab-btn:hover { border-color:#d4b8f4; color:var(--ink); }
  .tab-btn.active { background:var(--ink); color:white; border-color:var(--ink); }
  /* Ratings card */
  .card { background:white; border-radius:24px; padding:24px 28px; box-shadow:0 4px 24px #0000000a; }
  .table-wrap { overflow-x:auto; }
  table { width:100%; border-collapse:collapse; font-size:.85rem; }
  thead th { font-family:'Syne',sans-serif; font-size:.68rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase; color:var(--soft); padding:8px 12px; text-align:right; border-bottom:2px solid #f0ecf4; cursor:pointer; user-select:none; white-space:nowrap; }
  thead th:nth-child(2) { text-align:left; }
  thead th[style*="cursor:default"] { cursor:default !important; }
  thead th.sorted-asc::after  { content:' ▲'; font-size:.6rem; }
  thead th.sorted-desc::after { content:' ▼'; font-size:.6rem; }
  tbody tr { border-bottom:1px solid #f8f4fc; transition:background .12s; cursor:pointer; }
  tbody tr:last-child { border-bottom:none; }
  tbody tr:hover { background:#fdf6f0; }
  td { padding:10px 12px; text-align:right; vertical-align:middle; }
  td:nth-child(2) { text-align:left; }
  .rank-cell { color:var(--soft); font-size:.78rem; width:32px; }
  .org-cell { font-family:'Syne',sans-serif; font-weight:800; font-size:.88rem; display:flex; align-items:center; gap:8px; }
  .org-cell:hover .org-name { text-decoration:underline dotted; text-underline-offset:3px; color:#5a3a8a; }
  .team-logo { width:22px; height:22px; object-fit:contain; flex-shrink:0; }
  .rating-pos { color:#1a6a4a; font-weight:700; }
  .rating-neg { color:#7a1a1a; font-weight:700; }
  .rating-neu { color:var(--soft); font-weight:700; }
  .wl-cell { font-size:.82rem; color:var(--ink); }
  .pct-cell { font-weight:500; }
  /* Modal */
  .modal-backdrop { position:fixed; inset:0; background:#2a1f2daa; backdrop-filter:blur(4px); z-index:300; display:flex; align-items:center; justify-content:center; padding:20px; }
  .modal-box { background:white; border-radius:24px; padding:28px 32px; max-width:780px; width:100%; max-height:88vh; overflow-y:auto; box-shadow:0 24px 60px #0003; position:relative; animation:modalIn .2s ease; }
  @keyframes modalIn { from{opacity:0;transform:scale(.96)} to{opacity:1;transform:scale(1)} }
  .modal-close { position:absolute; top:14px; right:18px; background:none; border:none; font-size:1.4rem; cursor:pointer; color:var(--soft); padding:4px; line-height:1; }
  .modal-header { font-family:'Syne',sans-serif; font-size:1.05rem; font-weight:800; margin-bottom:4px; display:flex; align-items:center; gap:10px; }
  .modal-sub { font-size:.78rem; color:var(--soft); margin-bottom:20px; }
  .modal-logo { width:30px; height:30px; object-fit:contain; }
  .map-table { width:100%; border-collapse:collapse; font-size:.82rem; }
  .map-table thead th { font-family:'Syne',sans-serif; font-size:.65rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase; color:var(--soft); padding:6px 10px; text-align:right; border-bottom:1px solid #f0ecf4; }
  .map-table thead th:first-child { text-align:left; }
  .map-table tbody td { padding:7px 10px; text-align:right; border-bottom:1px solid #f8f4fc; }
  .map-table tbody td:first-child { text-align:left; font-weight:500; }
  .map-table tbody tr:last-child td { border-bottom:none; }
  .overall-row td { border-top:2px solid #f0ecf4 !important; font-weight:700; padding-top:10px !important; }
  /* Roster & recent matches in modal */
  .team-section { margin-top:20px; border-top:1px solid #f0ecf4; padding-top:16px; }
  .team-section-title { font-family:'Syne',sans-serif; font-size:.65rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; color:var(--soft); margin-bottom:12px; }
  .roster-list { display:flex; flex-wrap:wrap; gap:14px; }
  .roster-player { display:flex; flex-direction:column; align-items:center; gap:7px; width:96px; }
  .roster-headshot { width:72px; height:72px; border-radius:50%; object-fit:cover; object-position:top center; background:#f0ecf4; flex-shrink:0; }
  .roster-player-name { font-size:.78rem; font-weight:600; color:var(--ink); text-align:center; line-height:1.2; }
  .recent-match { border:1px solid #f0ecf4; border-radius:12px; padding:11px 13px; margin-bottom:8px; }
  .recent-match:last-child { margin-bottom:0; }
  .recent-match-header { display:flex; align-items:center; justify-content:space-between; margin-bottom:3px; }
  .recent-match-opp { font-family:'Syne',sans-serif; font-weight:800; font-size:.88rem; }
  .result-badge { font-family:'Syne',sans-serif; font-weight:800; font-size:.78rem; border-radius:6px; padding:2px 8px; }
  .result-badge.rw { background:#e8f5ee; color:#1a6a4a; }
  .result-badge.rl { background:#f5e8e8; color:#7a1a1a; }
  .recent-match-sub { font-size:.71rem; color:var(--soft); margin-bottom:7px; line-height:1.4; }
  .recent-match-maps { display:flex; flex-wrap:wrap; gap:4px; }
  .map-chip { font-size:.72rem; padding:2px 8px; border-radius:5px; font-weight:500; white-space:nowrap; }
  .map-chip-w { background:#e8f5ee; color:#1a6a4a; }
  .map-row { cursor:pointer; }
  .map-row:hover td { background:#f8f4fc; }
  .map-row-arrow { display:inline-block; font-size:.55rem; color:var(--soft); margin-right:5px; transition:transform .2s; line-height:1; vertical-align:middle; }
  .map-row.open .map-row-arrow { transform:rotate(90deg); }
  .map-history-row td { padding:0 !important; border:none !important; }
  .map-history-body { max-height:0; overflow:hidden; transition:max-height .3s ease; background:#faf8fc; }
  .map-history-body.open { /* max-height set by JS */ }
  .mh-inner { padding:14px 16px 18px; }
  .mh-table { width:100%; border-collapse:collapse; font-size:.8rem; }
  .mh-table thead th { font-family:'Syne',sans-serif; font-size:.62rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase; color:var(--soft); padding:6px 10px; text-align:center; border-bottom:1px solid #ede8f4; }
  .mh-table thead th:first-child { text-align:left; }
  .mh-table tbody tr { height:38px; }
  .mh-table tbody td { padding:6px 10px; border-bottom:1px solid #f0ecf8; vertical-align:middle; white-space:nowrap; text-align:center; }
  .mh-table tbody td.mh-label { white-space:normal; text-align:left; max-width:220px; line-height:1.45; }
  .mh-table tbody tr:last-child td { border-bottom:none; }
  .mh-label { font-size:.75rem; color:var(--soft); }
  .mh-opp { display:inline-flex; align-items:center; gap:6px; font-family:'Syne',sans-serif; font-weight:800; font-size:.82rem; }
  .mh-opp-logo { width:18px; height:18px; object-fit:contain; }
  .map-chip-l { background:#f5e8e8; color:#7a1a1a; }
  .team-extra-loading { color:var(--soft); font-size:.78rem; padding:8px 0; }
  /* International adjustment badge in rankings table */
  .intl-badge { display:inline-block; margin-left:5px; font-family:'Syne',sans-serif; font-size:.62rem; font-weight:800; padding:1px 6px; border-radius:99px; vertical-align:middle; }
  .intl-badge-pos { background:#e8f5ee; color:#1a6a4a; }
  .intl-badge-neg { background:#f5e8e8; color:#7a1a1a; }
  /* Intl breakdown row in modal */
  .intl-row { display:flex; align-items:center; gap:6px; padding:10px 0; border-top:1px solid #f0ecf4; margin-top:4px; flex-wrap:wrap; }
  .intl-row-label { font-family:'Syne',sans-serif; font-size:.62rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase; color:var(--soft); min-width:100px; }
  .intl-chip { font-size:.75rem; padding:2px 10px; border-radius:99px; font-weight:700; }
  .intl-chip-dom { background:#f0ecf8; color:var(--ink); }
  .intl-chip-reg-pos { background:#e8f5ee; color:#1a6a4a; }
  .intl-chip-reg-neg { background:#f5e8e8; color:#7a1a1a; }
  .intl-chip-reg-neu { background:#f0ecf8; color:var(--soft); }
  .intl-chip-ind-pos { background:#e8eef8; color:#1a3a7a; }
  .intl-chip-ind-neg { background:#f8e8f4; color:#7a1a5a; }
  .intl-chip-ind-neu { background:#f0ecf8; color:var(--soft); }
  .intl-chip-total { background:var(--ink); color:white; }
  .intl-chip-arrow { color:var(--soft); font-size:.8rem; }
</style>
</head>
<body>
<div id="content-wrap">
  <div class="top-nav">
    <a href="/"><img src="/logo.svg" alt="Home" class="home-logo"></a>
    <a class="back-link" href="/mapelo/">&larr; BenPom</a>
  </div>
  <div class="page">
    <div class="page-title">Rankings</div>
    <p class="page-sub">Opponent-adjusted round differential ratings for VCT franchised teams, 2023&ndash;2025 domestic events.</p>

    <!-- Animated model pipeline -->
    <div class="model-card">
      <div class="model-card-header">
        <span class="model-card-title">How the model works</span>
        <button class="model-card-toggle" id="model-toggle" onclick="toggleModel()"><i class="toggle-arrow">&#9654;</i> show</button>
      </div>
      <div class="model-collapsible" id="model-collapsible">
      <div class="pipeline-wrap" id="pipeline-wrap">

        <!-- Stage 1: Round Differential -->
        <div class="pipe-stage" id="ps0" data-idx="0" onclick="focusPipe(0)">
          <div class="pipe-num pipe-n0">1</div>
          <div class="pipe-content">
            <div class="pipe-title">Round Differential</div>
            <div class="pipe-desc">Every map is scored by round margin &mdash; a 13&ndash;2 win carries far more signal than a 13&ndash;11 grind. This <em>dominance signal</em> is what feeds the solver, not just win/loss.</div>
            <div class="pipe-graphic">
              <div class="pg-scorebar">
                <div class="pg-score-row">
                  <div class="pg-score-label">13&ndash;2</div>
                  <div class="pg-bar-track"><div class="pg-bar-fill pg-bar-big" id="pg0-b1"></div></div>
                  <div class="pg-score-diff pg-score-diff-big">+11</div>
                </div>
                <div class="pg-score-row">
                  <div class="pg-score-label">13&ndash;11</div>
                  <div class="pg-bar-track"><div class="pg-bar-fill pg-bar-small" id="pg0-b2"></div></div>
                  <div class="pg-score-diff pg-score-diff-small">+2</div>
                </div>
              </div>
              <div class="pg-note">bigger margin &rarr; larger weight in the Massey solve</div>
            </div>
          </div>
        </div>

        <div class="pipe-connector" id="pc0"><div class="pipe-particle" id="pp0a"></div><div class="pipe-particle pipe-particle-b" id="pp0b"></div></div>

        <!-- Stage 2: Massey System -->
        <div class="pipe-stage" id="ps1" data-idx="1" onclick="focusPipe(1)">
          <div class="pipe-num pipe-n1">2</div>
          <div class="pipe-content">
            <div class="pipe-title">Massey Rating System</div>
            <div class="pipe-desc">A linear algebra solve finds the rating vector that best explains all observed round differentials simultaneously &mdash; accounting for every opponent&rsquo;s strength. One solve per map. Mean-zero constraint.</div>
            <div class="pipe-graphic">
              <div style="padding:4px 0 2px;display:flex;align-items:flex-start;gap:14px;flex-wrap:wrap">
                <svg width="134" height="62" style="display:block;flex-shrink:0;overflow:visible">
                  <rect x="0" y="2" width="58" height="54" rx="6" fill="#ede8f8" id="pg1-m" style="opacity:0;transition:opacity .35s"/>
                  <text x="4"  y="16" font-size="7.5" font-family="monospace" fill="#9a7ab4" id="pg1-m1" style="opacity:0;transition:opacity .35s"> 1  0 -1  0</text>
                  <text x="4"  y="27" font-size="7.5" font-family="monospace" fill="#9a7ab4" id="pg1-m2" style="opacity:0;transition:opacity .35s">-1  1  0  0</text>
                  <text x="4"  y="38" font-size="7.5" font-family="monospace" fill="#9a7ab4" id="pg1-m3" style="opacity:0;transition:opacity .35s"> 0 -1  1  0</text>
                  <text x="4"  y="49" font-size="7.5" font-family="monospace" fill="#9a7ab4" id="pg1-m4" style="opacity:0;transition:opacity .35s"> 0  0 -1  1</text>
                  <text x="22" y="62" font-size="7" font-family="Syne,sans-serif" font-weight="800" fill="#b0a0c8" id="pg1-ml" style="opacity:0;transition:opacity .35s">M</text>
                  <text x="65" y="36" font-size="18" font-family="sans-serif" fill="#c8b8e0" id="pg1-dot" style="opacity:0;transition:opacity .35s">&middot;</text>
                  <rect x="74" y="6"  width="16" height="50" rx="4" fill="#e8e0f8" id="pg1-rv" style="opacity:0;transition:opacity .35s"/>
                  <text x="77" y="20" font-size="7.5" font-family="monospace" fill="#7a60d0" id="pg1-r1" style="opacity:0;transition:opacity .35s">r&#8321;</text>
                  <text x="77" y="32" font-size="7.5" font-family="monospace" fill="#7a60d0" id="pg1-r2" style="opacity:0;transition:opacity .35s">r&#8322;</text>
                  <text x="77" y="44" font-size="7.5" font-family="monospace" fill="#7a60d0" id="pg1-r3" style="opacity:0;transition:opacity .35s">r&#8323;</text>
                  <text x="76" y="62" font-size="7" font-family="Syne,sans-serif" font-weight="800" fill="#b0a0c8" id="pg1-rl" style="opacity:0;transition:opacity .35s">r</text>
                  <text x="96" y="36" font-size="14" font-family="sans-serif" fill="#c8b8e0" id="pg1-eq" style="opacity:0;transition:opacity .35s">=</text>
                  <rect x="109" y="6" width="16" height="50" rx="4" fill="#dff0e8" id="pg1-pv" style="opacity:0;transition:opacity .35s"/>
                  <text x="111" y="20" font-size="7.5" font-family="monospace" fill="#2a7a50" id="pg1-p1" style="opacity:0;transition:opacity .35s">+8</text>
                  <text x="111" y="32" font-size="7.5" font-family="monospace" fill="#2a7a50" id="pg1-p2" style="opacity:0;transition:opacity .35s">-3</text>
                  <text x="111" y="44" font-size="7.5" font-family="monospace" fill="#2a7a50" id="pg1-p3" style="opacity:0;transition:opacity .35s">+5</text>
                  <text x="110" y="62" font-size="7" font-family="Syne,sans-serif" font-weight="800" fill="#b0a0c8" id="pg1-pl" style="opacity:0;transition:opacity .35s">p</text>
                </svg>
                <div style="font-size:.68rem;color:var(--soft);line-height:1.8;padding-top:4px">
                  <strong style="color:var(--ink)">M</strong> = matchup matrix<br>
                  <strong style="color:var(--ink)">r</strong> = ratings (unknown)<br>
                  <strong style="color:var(--ink)">p</strong> = round differentials
                </div>
              </div>
            </div>
          </div>
        </div>

        <div class="pipe-connector" id="pc1"><div class="pipe-particle" id="pp1a"></div><div class="pipe-particle pipe-particle-b" id="pp1b"></div></div>

        <!-- Stage 3: Recency Decay -->
        <div class="pipe-stage" id="ps2" data-idx="2" onclick="focusPipe(2)">
          <div class="pipe-num pipe-n2">3</div>
          <div class="pipe-content">
            <div class="pipe-title">Recency Decay</div>
            <div class="pipe-desc">Game weights follow <code>exp(&minus;&lambda;&thinsp;&times;&thinsp;weeks&thinsp;ago)</code>. International games get 4&times; multiplier so attending Masters/Champions boosts ratings above domestic-only teams. Half-life &asymp;&thinsp;6 weeks, CV-optimized.</div>
            <div class="pipe-graphic">
              <div class="pg-decay-wrap"><canvas class="pg-decay-canvas" id="pg2-canvas"></canvas></div>
            </div>
          </div>
        </div>

        <div class="pipe-connector" id="pc2"><div class="pipe-particle" id="pp2a"></div><div class="pipe-particle pipe-particle-b" id="pp2b"></div></div>

        <!-- Stage 4: James-Stein Shrinkage -->
        <div class="pipe-stage" id="ps3" data-idx="3" onclick="focusPipe(3)">
          <div class="pipe-num pipe-n3">4</div>
          <div class="pipe-content">
            <div class="pipe-title">James&ndash;Stein Shrinkage</div>
            <div class="pipe-desc">Per-map ratings with thin data are blended toward the team&rsquo;s overall rating &mdash; preventing wild swings from 2&ndash;3 maps. More games = less shrinkage = more trust in the map-specific signal.</div>
            <div class="pipe-graphic">
              <div style="padding:4px 0 2px;display:flex;gap:10px;flex-wrap:wrap">
                <div style="background:#f8f4fc;border-radius:10px;padding:6px 11px;font-size:.7rem;line-height:1.85;flex:1;min-width:110px">
                  <strong style="font-family:'Syne',sans-serif;color:var(--ink)">2 games</strong><br>
                  <span style="color:var(--soft)">&alpha; &asymp; 0.14<br>heavy pull &rarr; overall</span>
                </div>
                <div style="background:#f0f8f4;border-radius:10px;padding:6px 11px;font-size:.7rem;line-height:1.85;flex:1;min-width:110px">
                  <strong style="font-family:'Syne',sans-serif;color:var(--ink)">20 games</strong><br>
                  <span style="color:var(--soft)">&alpha; &asymp; 0.63<br>mostly raw map signal</span>
                </div>
              </div>
              <div class="pg-note">&alpha; = n&thinsp;/&thinsp;(n+k) where k=12</div>
            </div>
          </div>
        </div>

        <div class="pipe-connector" id="pc3"><div class="pipe-particle" id="pp3a"></div><div class="pipe-particle pipe-particle-b" id="pp3b"></div></div>

        <!-- Stage 5: Monte Carlo Veto -->
        <div class="pipe-stage" id="ps4" data-idx="4" onclick="focusPipe(4)">
          <div class="pipe-num pipe-n4">5</div>
          <div class="pipe-content">
            <div class="pipe-title">Monte Carlo Veto Simulation</div>
            <div class="pipe-desc">10,000 simulated BO3 vetoes against a league-average opponent using historical ban/pick patterns. Expected round-diff across the surviving maps becomes the headline rating. A great ban target is worth as much as a great map.</div>
            <div class="pipe-graphic">
              <div style="display:flex;gap:5px;margin-bottom:5px;flex-wrap:wrap">
                <span style="font-family:'Syne',sans-serif;font-size:.6rem;font-weight:800;color:#c07070;padding:2px 7px;background:#f5e8e8;border-radius:5px">ban</span>
                <span style="font-family:'Syne',sans-serif;font-size:.6rem;font-weight:800;color:#c07070;padding:2px 7px;background:#f5e8e8;border-radius:5px">ban</span>
                <span style="font-family:'Syne',sans-serif;font-size:.6rem;font-weight:800;color:white;padding:2px 7px;background:linear-gradient(135deg,#a060d0,#7040a0);border-radius:5px">pick</span>
                <span style="font-family:'Syne',sans-serif;font-size:.6rem;font-weight:800;color:white;padding:2px 7px;background:linear-gradient(135deg,#a060d0,#7040a0);border-radius:5px">pick</span>
                <span style="font-family:'Syne',sans-serif;font-size:.6rem;font-weight:800;color:#c07070;padding:2px 7px;background:#f5e8e8;border-radius:5px">ban</span>
                <span style="font-family:'Syne',sans-serif;font-size:.6rem;font-weight:800;color:#c07070;padding:2px 7px;background:#f5e8e8;border-radius:5px">ban</span>
                <span style="font-family:'Syne',sans-serif;font-size:.6rem;font-weight:800;color:#5a2a7a;padding:2px 7px;background:linear-gradient(135deg,#f0e8ff,#e0d0f8);border-radius:5px;box-shadow:0 0 0 1px #c8a0e8">float</span>
              </div>
              <div class="pg-veto" id="pg4-veto">
                <div class="pg-map-chip">Abyss</div>
                <div class="pg-map-chip">Ascent</div>
                <div class="pg-map-chip">Bind</div>
                <div class="pg-map-chip">Haven</div>
                <div class="pg-map-chip">Lotus</div>
                <div class="pg-map-chip">Pearl</div>
                <div class="pg-map-chip">Split</div>
              </div>
              <div class="pg-note">10,000 simulated sequences &rarr; expected round diff across picked maps</div>
            </div>
          </div>
        </div>

        <div class="pipe-connector" id="pc4"><div class="pipe-particle" id="pp4a"></div><div class="pipe-particle pipe-particle-b" id="pp4b"></div></div>

        <!-- Stage 6: International Calibration -->
        <div class="pipe-stage" id="ps5" data-idx="5" onclick="focusPipe(5)">
          <div class="pipe-num pipe-n5">6</div>
          <div class="pipe-content">
            <div class="pipe-title">International Calibration</div>
            <div class="pipe-desc">Masters &amp; Champions maps receive a 4&times; weight in the Massey solve, directly inflating ratings for teams that perform well internationally. Cross-regional strength is baked into the rating number, not a side note.</div>
            <div class="pipe-graphic">
              <div class="pg-regions" id="pg5-regions">
                <div class="pg-region">
                  <div class="pg-region-bubble r-emea" id="pg5-emea">+&delta;</div>
                  <div class="pg-region-name">EMEA</div>
                </div>
                <div class="pg-intl-arrow">&#8596;</div>
                <div class="pg-region">
                  <div class="pg-region-bubble r-am" id="pg5-am">&minus;&delta;</div>
                  <div class="pg-region-name">Americas</div>
                </div>
                <div class="pg-intl-arrow">&#8596;</div>
                <div class="pg-region">
                  <div class="pg-region-bubble r-pac" id="pg5-pac">+&delta;</div>
                  <div class="pg-region-name">Pacific</div>
                </div>
              </div>
              <div class="pg-note">4&times; multiplier on intl maps &rarr; top-3 finishers always in global top 5&ndash;10</div>
            </div>
          </div>
        </div>

        <div class="pipe-connector" id="pc5"><div class="pipe-particle" id="pp5a"></div><div class="pipe-particle pipe-particle-b" id="pp5b"></div></div>

        <!-- Stage 7: Global Rating -->
        <div class="pipe-stage" id="ps6" data-idx="6" onclick="focusPipe(6)">
          <div class="pipe-num pipe-n6">&#10003;</div>
          <div class="pipe-content">
            <div class="pipe-title">Global Rating &amp; Win Probability</div>
            <div class="pipe-desc">The final rating reflects domestic results + international performance in a single number. The matchup predictor runs 10,000 Monte Carlo veto sims using these global ratings, returning a calibrated series win probability.</div>
            <div class="pipe-graphic">
              <div class="pg-formula" id="pg6-formula">
                <div class="pg-formula-part pg-formula-dom"  id="pg6-p0">domestic</div>
                <div class="pg-formula-part pg-formula-op"   id="pg6-p1">+</div>
                <div class="pg-formula-part pg-formula-intl" id="pg6-p2">4&times; intl maps</div>
                <div class="pg-formula-part pg-formula-op"   id="pg6-p3">=</div>
                <div class="pg-formula-part pg-formula-global" id="pg6-p4">global rating</div>
              </div>
              <div class="pg-note">win probability: 10,000 veto sims using global ratings on both sides</div>
            </div>
          </div>
        </div>

      </div>
      <div style="display:flex;justify-content:space-between;align-items:center;margin-top:8px;flex-wrap:wrap;gap:8px">
        <button class="pipe-replay-btn" id="pipe-replay-btn" onclick="replayPipeline()">&#9654; Replay</button>
        <button class="model-card-toggle" id="model-details-toggle" onclick="toggleModelDetails()" style="font-size:.6rem"><i class="toggle-arrow" id="details-arrow">&#9654;</i> details</button>
      </div>
      <div class="model-collapsible" id="model-details-collapsible">
        <div class="chart-section" id="lambda-chart-section" style="margin-top:8px;">
          <div class="chart-title">Decay optimization &mdash; CV Brier score vs. half-life</div>
          <div class="chart-wrap"><canvas id="lambda-chart"></canvas></div>
        </div>
        <div class="model-stats">
          <div class="stat-pill"><span class="stat-pill-label">Half-life</span><span class="stat-pill-value" id="stat-hl">&mdash;</span></div>
          <div class="stat-pill"><span class="stat-pill-label">Brier (held-out 2025)</span><span class="stat-pill-value good" id="stat-brier">&mdash;</span></div>
          <div class="stat-pill"><span class="stat-pill-label">Baseline</span><span class="stat-pill-value">0.2500</span></div>
          <div class="stat-pill"><span class="stat-pill-label">Train maps</span><span class="stat-pill-value" id="stat-train">&mdash;</span></div>
          <div class="stat-pill"><span class="stat-pill-label">Test maps</span><span class="stat-pill-value" id="stat-test">&mdash;</span></div>
          <div class="stat-pill"><span class="stat-pill-label">Veto sims/team</span><span class="stat-pill-value" id="stat-sims">&mdash;</span></div>
        </div>
      </div>
      </div>
    </div>

    <!-- Year / Period / Map filters -->
    <div class="filter-row">
      <span class="filter-label">Year</span>
      <button class="tab-btn year-btn active" data-year="2026">2026</button>
      <button class="tab-btn year-btn" data-year="2025">2025</button>
      <button class="tab-btn year-btn" data-year="2024">2024</button>
      <button class="tab-btn year-btn" data-year="2023">2023</button>
    </div>
    <div class="filter-row" id="period-filter-row" style="display:none">
      <span class="filter-label">Period</span>
      <select id="period-select" class="period-select"></select>
    </div>
    <div class="filter-row-maps" id="map-filter-row">
      <span class="filter-label">Map</span>
    </div>

    <div class="card">
      <div class="table-wrap">
        <table id="ratings-table">
          <thead>
            <tr>
              <th data-col="rank" style="cursor:default">#</th>
              <th data-col="org" style="cursor:default">Team</th>
              <th id="rating-th" data-col="overall_rating">Rating</th>
              <th data-col="w" style="cursor:default">W&ndash;L</th>
              <th data-col="win_pct">Win%</th>
            </tr>
          </thead>
          <tbody id="ratings-body"></tbody>
        </table>
      </div>
    </div>
  </div>
</div>

<div id="team-modal" class="modal-backdrop" style="display:none" onclick="if(event.target===this)closeModal()">
  <div class="modal-box">
    <button class="modal-close" onclick="closeModal()">&times;</button>
    <div id="modal-content"></div>
  </div>
</div>

<script>
var DATA = RATINGS_JSON;
var INTL = DATA.intl_calib || {};
var INTL_PARAMS = DATA.intl_params || {};
var ORG_REGIONS = DATA.org_regions || {};
var currentYear     = '2026';
var currentSnap     = (function(){ var keys=Object.keys((DATA.ratings['2026']||{}).snapshots||{}); return keys.indexOf('after_champions')>=0?'after_champions':keys[keys.length-1]||'after_champions'; })();
var currentMap      = null;
var currentModalOrg = '';
var sortCol = 'overall_rating';
var sortDir = -1;

function getGlobalRating(org, snapKey, domesticRating) {
  var cal = INTL[snapKey] || {};
  var region = ORG_REGIONS[org] || '';
  var regOff = (cal.regional_offsets || {})[region] || 0;
  var indBonus = (cal.individual_bonuses || {})[org] || 0;
  return domesticRating + regOff + indBonus;
}
function getIntlBreakdown(org, snapKey) {
  var cal = INTL[snapKey] || {};
  var region = ORG_REGIONS[org] || '';
  var regOff = (cal.regional_offsets || {})[region] || 0;
  var indBonus = (cal.individual_bonuses || {})[org] || 0;
  return {region: region, regOff: regOff, indBonus: indBonus, total: regOff + indBonus};
}

var _pipelineStarted = false;
function toggleModel() {
  var c = document.getElementById('model-collapsible');
  var btn = document.getElementById('model-toggle');
  var open = c.classList.toggle('open');
  btn.classList.toggle('open', open);
  btn.querySelector('.toggle-arrow').style.transform = open ? 'rotate(90deg)' : '';
  btn.lastChild.textContent = ' ' + (open ? 'hide' : 'show');
  if (open && !_pipelineStarted) {
    _pipelineStarted = true;
    setTimeout(function(){ _runPipeStep(0, _pipelineDone); }, 200);
  }
}
function toggleModelDetails() {
  var c = document.getElementById('model-details-collapsible');
  var arrow = document.getElementById('details-arrow');
  var btn = document.getElementById('model-details-toggle');
  var open = c.classList.toggle('open');
  btn.classList.toggle('open', open);
  arrow.style.transform = open ? 'rotate(90deg)' : '';
  btn.lastChild.textContent = ' ' + (open ? 'hide' : 'details');
}

// ── Pipeline animation ──────────────────────────────────────────────────────
var _pipeTimer = null, _pipeActive = -1;
var PIPE_N = 7;

// ── Stage-specific graphic animations ────────────────────────────────────────
function _animateScoreBars() {
  var b1 = document.getElementById('pg0-b1'), b2 = document.getElementById('pg0-b2');
  if (b1) { b1.style.width = '0'; void b1.offsetWidth; setTimeout(function(){ b1.style.width='100%'; }, 80); }
  if (b2) { b2.style.width = '0'; void b2.offsetWidth; setTimeout(function(){ b2.style.width='18%'; }, 180); }
}

function _animateMasseyMatrix() {
  var ids = ['pg1-m','pg1-m1','pg1-m2','pg1-m3','pg1-m4','pg1-ml',
             'pg1-dot','pg1-rv','pg1-r1','pg1-r2','pg1-r3','pg1-rl',
             'pg1-eq','pg1-pv','pg1-p1','pg1-p2','pg1-p3','pg1-pl'];
  ids.forEach(function(id, i) {
    setTimeout(function() { var el=document.getElementById(id); if(el) el.style.opacity='1'; }, i * 75);
  });
}

function _resetMasseyMatrix() {
  ['pg1-m','pg1-m1','pg1-m2','pg1-m3','pg1-m4','pg1-ml',
   'pg1-dot','pg1-rv','pg1-r1','pg1-r2','pg1-r3','pg1-rl',
   'pg1-eq','pg1-pv','pg1-p1','pg1-p2','pg1-p3','pg1-pl'].forEach(function(id) {
    var el = document.getElementById(id); if(el) el.style.opacity='0';
  });
}

function _drawDecayCanvas() {
  var c = document.getElementById('pg2-canvas');
  if (!c) return;
  if (!c.offsetWidth) { requestAnimationFrame(_drawDecayCanvas); return; }
  var dpr = window.devicePixelRatio || 1;
  var W = c.offsetWidth, H = 120;
  c.width = W * dpr; c.height = H * dpr;
  var ctx = c.getContext('2d');
  ctx.scale(dpr, dpr);
  var PL=46, PB=22, PT=12, PR=12;
  var cW=W-PL-PR, cH=H-PB-PT;
  var MAX_W=4.5, MAX_WKS=20, STEPS=60;
  var lam = Math.LN2/6;
  function toX(wk){ return PL+(wk/MAX_WKS)*cW; }
  function toY(wt){ return (H-PB)-(wt/MAX_W)*cH; }
  function drawCurvePath(mult, steps) {
    ctx.beginPath();
    for(var i=0;i<=steps;i++){
      var wk=(i/STEPS)*MAX_WKS, wt=mult*Math.exp(-lam*wk);
      i===0?ctx.moveTo(toX(wk),toY(wt)):ctx.lineTo(toX(wk),toY(wt));
    }
  }
  function drawStatic() {
    // Grid lines + y-axis tick labels
    [1,2,3,4].forEach(function(wt){
      ctx.strokeStyle='#edeaf4'; ctx.lineWidth=.6;
      ctx.beginPath(); ctx.moveTo(PL,toY(wt)); ctx.lineTo(W-PR,toY(wt)); ctx.stroke();
      ctx.fillStyle='#b0a0c0'; ctx.font='8px DM Sans,sans-serif'; ctx.textAlign='right';
      ctx.fillText(wt, PL-5, toY(wt)+3);
    });
    // Axes
    ctx.strokeStyle='#ddd8e8'; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(PL,PT); ctx.lineTo(PL,H-PB); ctx.lineTo(W-PR,H-PB); ctx.stroke();
    // X-axis ticks + label
    [0,6,12,18].forEach(function(wk){
      ctx.fillStyle='#c0b8cc'; ctx.font='8px DM Sans,sans-serif'; ctx.textAlign='center';
      ctx.fillText(wk, toX(wk), H-PB+10);
    });
    ctx.fillStyle='#a090b8'; ctx.font='8px DM Sans,sans-serif'; ctx.textAlign='center';
    ctx.fillText('weeks ago', PL+cW/2, H-3);
  }
  var t=0;
  function step() {
    ctx.clearRect(0,0,W,H);
    drawStatic();
    if (t===0) { requestAnimationFrame(step); t=1; return; }
    // ── Intl curve (pink, 4×) ────────────────────
    drawCurvePath(4, t);
    var gI=ctx.createLinearGradient(PL,0,W-PR,0);
    gI.addColorStop(0,'#d83898'); gI.addColorStop(1,'#f060b8');
    ctx.strokeStyle=gI; ctx.lineWidth=2.5; ctx.stroke();
    drawCurvePath(4,t);
    ctx.lineTo(toX((t/STEPS)*MAX_WKS),H-PB); ctx.lineTo(PL,H-PB); ctx.closePath();
    var fI=ctx.createLinearGradient(0,PT,0,H-PB);
    fI.addColorStop(0,'rgba(216,56,152,.14)'); fI.addColorStop(1,'rgba(216,56,152,0)');
    ctx.fillStyle=fI; ctx.fill();
    // ── Domestic curve (purple, 1×) ───────────────
    drawCurvePath(1, t);
    var gD=ctx.createLinearGradient(PL,0,W-PR,0);
    gD.addColorStop(0,'#a060d0'); gD.addColorStop(1,'#d080f8');
    ctx.strokeStyle=gD; ctx.lineWidth=2.5; ctx.stroke();
    drawCurvePath(1,t);
    ctx.lineTo(toX((t/STEPS)*MAX_WKS),H-PB); ctx.lineTo(PL,H-PB); ctx.closePath();
    var fD=ctx.createLinearGradient(0,PT,0,H-PB);
    fD.addColorStop(0,'rgba(160,96,208,.15)'); fD.addColorStop(1,'rgba(160,96,208,0)');
    ctx.fillStyle=fD; ctx.fill();
    // ── Curve labels ─────────────────────────────
    if (t>=10) {
      ctx.font='bold 8.5px Syne,sans-serif'; ctx.textAlign='left';
      ctx.fillStyle='#c02888'; ctx.fillText('intl ×4', PL+6, toY(4)-5);
      ctx.fillStyle='#7030b8'; ctx.fillText('domestic', PL+6, toY(1)-5);
    }
    // ── 4× bracket on left ───────────────────────
    if (t>=14) {
      var bx=PL-14, y1=toY(4), y4=toY(1), mid=(y1+y4)/2;
      ctx.strokeStyle='#d090c0'; ctx.lineWidth=1;
      ctx.beginPath();
      ctx.moveTo(bx+5,y1); ctx.lineTo(bx,y1); ctx.lineTo(bx,y4); ctx.lineTo(bx+5,y4);
      ctx.stroke();
      ctx.fillStyle='#b05090'; ctx.font='bold 8px Syne,sans-serif'; ctx.textAlign='right';
      ctx.fillText('4×', bx-2, mid+3);
    }
    // ── Half-life dashed line ─────────────────────
    if (t >= Math.round((6/MAX_WKS)*STEPS)) {
      var hlX=toX(6), hlY=toY(0.5);
      ctx.strokeStyle='#ccc0e0'; ctx.lineWidth=1; ctx.setLineDash([3,3]);
      ctx.beginPath(); ctx.moveTo(hlX,H-PB); ctx.lineTo(hlX,hlY); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle='#9060b8'; ctx.font='bold 8px Syne,sans-serif'; ctx.textAlign='center';
      ctx.fillText('t½=6w', hlX, hlY-4);
    }
    t=Math.min(t+2, STEPS);
    if (t<STEPS) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

function _animateVeto() {
  var chips = document.querySelectorAll('#pg4-veto .pg-map-chip');
  if (!chips.length) return;
  chips.forEach(function(c) { c.className = 'pg-map-chip'; });
  // Correct BO3 veto: ban ban pick pick ban ban float
  // 0=Abyss 1=Ascent 2=Bind 3=Haven 4=Lotus 5=Pearl 6=Split
  var seq = [
    [2, 'banned'],  // A bans
    [5, 'banned'],  // B bans
    [1, 'picked'],  // A picks
    [4, 'picked'],  // B picks
    [0, 'banned'],  // A bans
    [6, 'banned'],  // B bans
    [3, 'float']    // Haven = decider
  ];
  seq.forEach(function(s, i) {
    setTimeout(function() { if(chips[s[0]]) chips[s[0]].classList.add(s[1]); }, 350 + i*400);
  });
}

function _animateRegions() {
  ['pg5-emea','pg5-am','pg5-pac'].forEach(function(id, i) {
    setTimeout(function() { var el=document.getElementById(id); if(el) el.classList.add('show'); }, 200 + i*300);
  });
}

function _animateFormula() {
  var parts = ['pg6-p0','pg6-p1','pg6-p2','pg6-p3','pg6-p4'];
  parts.forEach(function(id) { var el=document.getElementById(id); if(el) el.classList.remove('show'); });
  parts.forEach(function(id, i) {
    setTimeout(function() { var el=document.getElementById(id); if(el) el.classList.add('show'); }, 250 + i*240);
  });
}

function focusPipe(idx) {
  for(var i=0;i<PIPE_N;i++) {
    var s=document.getElementById('ps'+i);
    if(!s) continue;
    if(i < idx) { s.classList.add('active'); s.classList.remove('active-current'); }
    else if(i === idx) { s.classList.add('active', 'active-current'); }
    else { s.classList.remove('active', 'active-current'); }
    // Light up connectors between active stages
    var pc = document.getElementById('pc'+i);
    if (pc) { if(i < idx) pc.classList.add('lit'); else pc.classList.remove('lit'); }
  }
  _pipeActive = idx;
  // Trigger graphics per stage
  if (idx === 0) _animateScoreBars();
  if (idx === 1) _animateMasseyMatrix();
  if (idx === 2) requestAnimationFrame(_drawDecayCanvas);
  if (idx === 4) _animateVeto();
  if (idx === 5) _animateRegions();
  if (idx === 6) _animateFormula();
}

function _runPipeStep(idx, done) {
  if(idx >= PIPE_N) { if(done) done(); return; }
  if(idx > 0) {
    var ci = idx-1;
    ['a','b'].forEach(function(s) {
      var pp = document.getElementById('pp'+ci+s);
      if(pp) { pp.classList.remove('flowing'); void pp.offsetWidth; pp.classList.add('flowing'); }
    });
  }
  _pipeTimer = setTimeout(function() {
    focusPipe(idx);
    _pipeTimer = setTimeout(function() { _runPipeStep(idx+1, done); }, 950);
  }, idx===0 ? 200 : 720);
}

function _pipelineDone() {
  for(var i=0;i<PIPE_N;i++) {
    var s=document.getElementById('ps'+i);
    if(!s) continue;
    s.classList.add('active');
    if(i === PIPE_N-1) s.classList.add('active-current'); else s.classList.remove('active-current');
    var pc = document.getElementById('pc'+i);
    if(pc) pc.classList.add('lit');
  }
}

// Pipeline animation starts on first open of the model section (see toggleModel)

function replayPipeline() {
  if(_pipeTimer) { clearTimeout(_pipeTimer); _pipeTimer=null; }
  for(var i=0;i<PIPE_N;i++) {
    var s=document.getElementById('ps'+i);
    if(s) s.classList.remove('active','active-current');
    ['a','b'].forEach(function(sf) { var pp=document.getElementById('pp'+i+sf); if(pp) pp.classList.remove('flowing'); });
    var pc=document.getElementById('pc'+i); if(pc) pc.classList.remove('lit');
  }
  // Reset graphics
  var b1=document.getElementById('pg0-b1'); if(b1) b1.style.width='0';
  var b2=document.getElementById('pg0-b2'); if(b2) b2.style.width='0';
  _resetMasseyMatrix();
  var dc=document.getElementById('pg2-canvas'); if(dc){ var ctx=dc.getContext('2d'); ctx.clearRect(0,0,dc.width,dc.height); }
  document.querySelectorAll('#pg4-veto .pg-map-chip').forEach(function(c){ c.className='pg-map-chip'; });
  ['pg5-emea','pg5-am','pg5-pac'].forEach(function(id){ var el=document.getElementById(id); if(el) el.classList.remove('show'); });
  ['pg6-p0','pg6-p1','pg6-p2','pg6-p3','pg6-p4'].forEach(function(id){ var el=document.getElementById(id); if(el) el.classList.remove('show'); });
  _pipeActive=-1;
  _runPipeStep(0, _pipelineDone);
}

// ── Data accessors ──────────────────────────────────────────────────────────

function getSnaps() {
  var yr = DATA.ratings[currentYear];
  return (yr && yr.snapshots) ? yr.snapshots : {};
}

function getSnap() {
  var snaps = getSnaps();
  return snaps[currentSnap] || snaps['after_champions'] || snaps['before_champions'] || snaps[Object.keys(snaps)[Object.keys(snaps).length-1]] || {};
}

function getTeams() {
  var snap = getSnap();
  if (!snap.teams) return [];
  return Object.entries(snap.teams).map(function(e) {
    return Object.assign({org: e[0]}, e[1]);
  });
}

function getMaps() {
  var seen = {};
  getTeams().forEach(function(t) {
    Object.keys(t.maps || {}).forEach(function(m) { seen[m] = true; });
  });
  return Object.keys(seen).sort();
}

function getVal(t, col) {
  if (currentMap && t.maps && t.maps[currentMap]) {
    var md = t.maps[currentMap];
    if (col === 'overall_rating') return md.rating;
    if (col === 'win_pct')        return md.win_pct;
  }
  if (col === 'overall_rating' && !currentMap) {
    return t.overall_rating;
  }
  return t[col];
}

function ratingClass(v) {
  if (v > 0.05) return 'rating-pos';
  if (v < -0.05) return 'rating-neg';
  return 'rating-neu';
}

// ── Filter renderers ────────────────────────────────────────────────────────

function renderPeriodFilter() {
  var snaps = getSnaps();
  var keys  = Object.keys(snaps);
  var row   = document.getElementById('period-filter-row');
  var sel   = document.getElementById('period-select');

  if (keys.length <= 1) { row.style.display = 'none'; return; }
  row.style.display = 'flex';

  sel.innerHTML = '';
  Object.keys(snaps).forEach(function(k) {
    var opt = document.createElement('option');
    opt.value = k;
    opt.textContent = snaps[k].label;
    opt.selected = (currentSnap === k);
    sel.appendChild(opt);
  });
}

document.getElementById('period-select').addEventListener('change', function() {
  if (currentSnap === this.value) return;
  currentSnap = this.value;
  currentMap  = null;
  sortCol = 'overall_rating'; sortDir = -1;
  renderPeriodFilter();
  renderMapFilter();
  renderTable();
});

function renderMapFilter() {
  var maps = getMaps();
  var row  = document.getElementById('map-filter-row');
  row.querySelectorAll('.map-btn').forEach(function(b) { b.remove(); });

  function addBtn(label, mapVal) {
    var btn = document.createElement('button');
    btn.className = 'tab-btn map-btn' + (currentMap === mapVal ? ' active' : '');
    btn.textContent = label;
    row.appendChild(btn);
    btn.addEventListener('click', function() {
      if (currentMap === mapVal) return;
      currentMap = mapVal;
      sortCol = 'overall_rating'; sortDir = -1;
      renderMapFilter();
      renderTable();
    });
  }

  addBtn('All', null);
  maps.forEach(function(m) { addBtn(m, m); });
}

// ── Table renderer ──────────────────────────────────────────────────────────

function renderTable() {
  var teams = getTeams();

  if (currentMap) {
    teams = teams.filter(function(t) { return t.maps && t.maps[currentMap]; });
  }

  teams.sort(function(a, b) {
    var av = getVal(a, sortCol), bv = getVal(b, sortCol);
    if (av == null) av = 0;
    if (bv == null) bv = 0;
    return sortDir * (av - bv);
  });

  var snapKey = currentYear + '_' + currentSnap;
  var html = '';
  teams.forEach(function(t, i) {
    var rating  = currentMap ? t.maps[currentMap].rating   : t.overall_rating;
    var w       = currentMap ? t.maps[currentMap].w        : t.w;
    var l       = currentMap ? t.maps[currentMap].l        : t.l;
    var winPct  = currentMap ? t.maps[currentMap].win_pct  : t.win_pct;
    var sign = rating >= 0 ? '+' : '';
    var cls  = ratingClass(rating);
    // International adjustment badge
    var bd = getIntlBreakdown(t.org, snapKey);
    var intlBadge = '';
    if(!currentMap && Math.abs(bd.total) > 0.005) {
      var badgeCls = bd.total > 0 ? 'intl-badge-pos' : 'intl-badge-neg';
      intlBadge = '<span class="intl-badge '+badgeCls+'">'+(bd.total>0?'+':'')+bd.total.toFixed(2)+'</span>';
    }
    html += '<tr data-org="' + t.org + '" onclick="openModal(this.dataset.org)">' +
      '<td class="rank-cell">' + (i + 1) + '</td>' +
      '<td><div class="org-cell">' +
        '<img src="/logos/' + t.org + '.png" class="team-logo" onerror="this.hidden=true">' +
        '<span class="org-name">' + t.org + '</span>' +
      '</div></td>' +
      '<td class="' + cls + '">' + sign + rating.toFixed(2) + intlBadge + '</td>' +
      '<td class="wl-cell">' + w + '&ndash;' + l + '</td>' +
      '<td class="pct-cell">' + (winPct * 100).toFixed(1) + '%</td>' +
      '</tr>';
  });
  document.getElementById('ratings-body').innerHTML = html;

  document.getElementById('rating-th').textContent = currentMap || 'Rating';

  document.querySelectorAll('#ratings-table thead th[data-col]').forEach(function(th) {
    th.classList.remove('sorted-asc', 'sorted-desc');
    if (th.dataset.col === sortCol && th.style.cursor !== 'default') {
      th.classList.add(sortDir === -1 ? 'sorted-desc' : 'sorted-asc');
    }
  });
}

// ── Modal ───────────────────────────────────────────────────────────────────

// ── Monte Carlo pick/ban simulation ────────────────────────────────────────

function renderTeamExtra(org, data) {
  var el = document.getElementById('team-extra');
  if (!el) return;
  var html = '';

  if (data.roster && data.roster.length) {
    html += '<div class="team-section">' +
      '<div class="team-section-title">Roster</div>' +
      '<div class="roster-list">' +
      data.roster.map(function(p) {
        var name = p.name || p;
        var hs = p.headshot || '';
        var img = hs
          ? '<img class="roster-headshot" src="' + hs + '" alt="' + name + '" onerror="this.style.visibility=\\'hidden\\'">'
          : '<div class="roster-headshot" style="background:#e8e4f0;"></div>';
        return '<div class="roster-player">' + img + '<span class="roster-player-name">' + name + '</span></div>';
      }).join('') +
      '</div></div>';
  }

  if (data.recent_matches && data.recent_matches.length) {
    html += '<div class="team-section"><div class="team-section-title">Recent Matches</div>';
    data.recent_matches.forEach(function(m) {
      var badgeCls = m.series_result === 'W' ? 'rw' : 'rl';
      var sub = m.event_label + (m.match_name ? ' — ' + m.match_name : '');
      var chips = (m.maps || []).map(function(mp) {
        var cc = mp.result === 'W' ? 'map-chip-w' : 'map-chip-l';
        return '<span class="map-chip ' + cc + '">' + (mp.map_name || '?') + ' ' + mp.score + '</span>';
      }).join('');
      html += '<div class="recent-match">' +
        '<div class="recent-match-header">' +
          '<span class="recent-match-opp">' + m.opponent + '</span>' +
          '<span class="result-badge ' + badgeCls + '">' + m.series_result + ' ' + m.series_score + '</span>' +
        '</div>' +
        '<div class="recent-match-sub">' + sub + '</div>' +
        '<div class="recent-match-maps">' + chips + '</div>' +
      '</div>';
    });
    html += '</div>';
  }

  el.innerHTML = html;
}

function toggleMapHistory(row) {
  var histRow = row.nextElementSibling;
  if (!histRow || !histRow.classList.contains('map-history-row')) return;
  var body = histRow.querySelector('.map-history-body');
  var isOpen = row.classList.contains('open');

  // Close any other open rows first
  document.querySelectorAll('.map-row.open').forEach(function(r) {
    if (r !== row) {
      r.classList.remove('open');
      var b = r.nextElementSibling && r.nextElementSibling.querySelector('.map-history-body');
      if (b) b.style.maxHeight = '0';
    }
  });

  if (isOpen) {
    row.classList.remove('open');
    body.style.maxHeight = '0';
  } else {
    row.classList.add('open');
    var mapName = row.dataset.map;
    if (body.dataset.loaded) {
      body.style.maxHeight = body.scrollHeight + 'px';
      return;
    }
    body.innerHTML = '<div class="mh-inner" style="color:var(--soft);font-size:.8rem;">Loading…</div>';
    body.style.maxHeight = '48px';
    fetch('/mapelo/map-matches/' + encodeURIComponent(currentModalOrg) + '/' + encodeURIComponent(mapName) +
          '?year=' + encodeURIComponent(currentYear) + '&snap=' + encodeURIComponent(currentSnap))
      .then(function(r) { return r.json(); })
      .then(function(data) { renderMapHistoryInline(mapName, data, body); })
      .catch(function() { body.style.maxHeight = '0'; row.classList.remove('open'); });
  }
}

function renderMapHistoryInline(mapName, data, body) {
  var teams = (getSnap().teams || {});
  var rows = (data.matches || []).map(function(m) {
    var oppTeam = teams[m.opponent];
    var oppRating = '&mdash;';
    if (oppTeam && oppTeam.maps && oppTeam.maps[mapName]) {
      var r = oppTeam.maps[mapName].rating;
      oppRating = '<span class="' + ratingClass(r) + '">' + (r >= 0 ? '+' : '') + r.toFixed(2) + '</span>';
    }
    var rdSign = m.round_diff > 0 ? '+' : '';
    var rdCls  = m.round_diff > 0 ? 'rating-pos' : (m.round_diff < 0 ? 'rating-neg' : '');
    var badgeCls = m.result === 'W' ? 'rw' : 'rl';
    var label = m.event_label + (m.match_name ? ' &mdash; ' + m.match_name : '');
    return '<tr>' +
      '<td class="mh-label">' + label + '</td>' +
      '<td><span class="mh-opp"><img class="mh-opp-logo" src="/logos/' + m.opponent + '.png" onerror="this.hidden=true">' + m.opponent + '</span></td>' +
      '<td>' + oppRating + '</td>' +
      '<td><span class="result-badge ' + badgeCls + '">' + m.result + ' ' + m.score + '</span></td>' +
      '<td class="' + rdCls + '">' + rdSign + m.round_diff + '</td>' +
    '</tr>';
  }).join('');
  var tbody = rows || '<tr><td colspan="5" style="color:var(--soft);padding:10px 8px;">No matches found.</td></tr>';
  body.innerHTML =
    '<div class="mh-inner">' +
      '<table class="mh-table">' +
        '<thead><tr><th>Match</th><th>Opponent</th><th>Opp. rating</th><th>Result</th><th>Rd diff</th></tr></thead>' +
        '<tbody>' + tbody + '</tbody>' +
      '</table>' +
    '</div>';
  body.dataset.loaded = '1';
  body.style.maxHeight = body.scrollHeight + 'px';
}

function openModal(org) {
  currentModalOrg = org;
  var snap = getSnap();
  if (!snap.teams || !snap.teams[org]) return;
  var t    = snap.teams[org];
  var sign = t.overall_rating >= 0 ? '+' : '';

  var maps = Object.entries(t.maps || {}).sort(function(a, b) { return b[1].rating - a[1].rating; });
  var mapRows = maps.map(function(e) {
    var mn = e[0], md = e[1];
    var s = md.rating >= 0 ? '+' : '';
    var cls = ratingClass(md.rating);
    return '<tr class="map-row" data-map="' + mn + '" onclick="toggleMapHistory(this)">' +
      '<td><span class="map-row-arrow">&#9654;</span>' + mn + '</td>' +
      '<td class="' + cls + '">' + s + md.rating.toFixed(2) + '</td>' +
      '<td>' + md.w + '&ndash;' + md.l + '</td>' +
      '<td>' + (md.win_pct * 100).toFixed(1) + '%</td>' +
    '</tr>' +
    '<tr class="map-history-row"><td colspan="4"><div class="map-history-body"></div></td></tr>';
  }).join('');

  var snap_label = (getSnaps()[currentSnap] || {}).label || '';
  var logoHtml = '<img src="/logos/' + org + '.png" class="modal-logo" onerror="this.hidden=true">';

  var teams = Object.keys(snap.teams || {}).filter(function(o){ return o !== org; }).sort();
  var opponentOpts = teams.map(function(o) {
    return '<option value="' + o + '">' + o + '</option>';
  }).join('');

  // International rating breakdown
  var snapKey = currentYear + '_' + currentSnap;
  var bd = getIntlBreakdown(org, snapKey);
  var globalRating = t.overall_rating + bd.total;
  var intlRowHtml = '';
  if(bd.region) {
    var regCls = bd.regOff > 0.005 ? 'intl-chip-reg-pos' : bd.regOff < -0.005 ? 'intl-chip-reg-neg' : 'intl-chip-reg-neu';
    var indCls = bd.indBonus > 0.005 ? 'intl-chip-ind-pos' : bd.indBonus < -0.005 ? 'intl-chip-ind-neg' : 'intl-chip-ind-neu';
    intlRowHtml = '<div class="intl-row">' +
      '<span class="intl-row-label">Intl rating</span>' +
      '<span class="intl-chip intl-chip-dom">'+(t.overall_rating>=0?'+':'')+t.overall_rating.toFixed(2)+' domestic</span>' +
      '<span class="intl-chip-arrow">+</span>' +
      '<span class="intl-chip '+regCls+'">'+(bd.regOff>=0?'+':'')+bd.regOff.toFixed(2)+' '+bd.region+'</span>' +
      '<span class="intl-chip-arrow">+</span>' +
      '<span class="intl-chip '+indCls+'">'+(bd.indBonus>=0?'+':'')+bd.indBonus.toFixed(2)+' indiv.</span>' +
      '<span class="intl-chip-arrow">=</span>' +
      '<span class="intl-chip intl-chip-total">'+(globalRating>=0?'+':'')+globalRating.toFixed(2)+' global</span>' +
    '</div>';
  }

  document.getElementById('modal-content').innerHTML =
    '<div class="modal-header">' + logoHtml + org + '</div>' +
    '<div class="modal-sub">' + currentYear +
      (snap_label ? ' &mdash; ' + snap_label : '') +
      ' &mdash; ' + t.w + '&ndash;' + t.l +
      ' (' + (t.win_pct * 100).toFixed(1) + '% win rate)</div>' +
    '<table class="map-table">' +
      '<thead><tr><th>Map</th><th>Rating</th><th>W&ndash;L</th><th>Win%</th></tr></thead>' +
      '<tbody>' + mapRows +
        '<tr class="overall-row">' +
          '<td>Overall (Pick/Ban)</td>' +
          '<td class="' + ratingClass(t.overall_rating) + '">' + sign + t.overall_rating.toFixed(2) + '</td>' +
          '<td>' + t.w + '&ndash;' + t.l + '</td>' +
          '<td>' + (t.win_pct * 100).toFixed(1) + '%</td>' +
        '</tr>' +
      '</tbody>' +
    '</table>' +
    intlRowHtml +
    '<div id="team-extra"><div class="team-extra-loading">Loading roster &amp; recent matches…</div></div>';

  document.getElementById('team-modal').style.display = 'flex';

  fetch('/mapelo/team-info/' + encodeURIComponent(org) + '?year=' + encodeURIComponent(currentYear) + '&snap=' + encodeURIComponent(currentSnap))
    .then(function(r) { return r.json(); })
    .then(function(data) { renderTeamExtra(org, data); })
    .catch(function() {
      var el = document.getElementById('team-extra');
      if (el) el.innerHTML = '';
    });
}

function closeModal() { document.getElementById('team-modal').style.display = 'none'; }
document.addEventListener('keydown', function(e) { if (e.key === 'Escape') closeModal(); });

// ── Sort on header click ────────────────────────────────────────────────────

document.querySelectorAll('#ratings-table thead th[data-col]').forEach(function(th) {
  if (th.style.cursor === 'default') return;
  th.addEventListener('click', function() {
    var col = th.dataset.col;
    if (sortCol === col) { sortDir *= -1; }
    else { sortCol = col; sortDir = -1; }
    renderTable();
  });
});

// ── Year tabs ───────────────────────────────────────────────────────────────

document.querySelectorAll('.year-btn').forEach(function(btn) {
  btn.addEventListener('click', function() {
    if (currentYear === btn.dataset.year) return;
    currentYear = btn.dataset.year;
    var snapsForYear = Object.keys((DATA.ratings[currentYear]||{}).snapshots||{});
    currentSnap = snapsForYear.indexOf('after_champions')>=0 ? 'after_champions'
                : snapsForYear[snapsForYear.length-1] || 'after_champions';
    currentMap  = null;
    sortCol = 'overall_rating'; sortDir = -1;
    document.querySelectorAll('.year-btn').forEach(function(b) { b.classList.remove('active'); });
    btn.classList.add('active');
    renderPeriodFilter();
    renderMapFilter();
    renderTable();
  });
});

// ── Init ────────────────────────────────────────────────────────────────────

(function() {
  var meta = DATA.metadata;
  var hl   = meta.optimal_half_life_weeks;
  document.getElementById('stat-hl').textContent    = hl + ' weeks';
  document.getElementById('stat-brier').textContent = meta.brier_test ? meta.brier_test.toFixed(4) : '--';
  document.getElementById('stat-train').textContent = meta.n_train || '--';
  document.getElementById('stat-test').textContent  = meta.n_test  || '--';
  document.getElementById('stat-sims').textContent  = meta.mc_n_sims ? meta.mc_n_sims.toLocaleString() : '--';

  var grid   = DATA.lambda_grid || [];
  if (!grid.length) {
    document.getElementById('lambda-chart-section').style.display = 'none';
  } else {
  var labels = grid.map(function(r) { return r.half_life_weeks; });
  var briers = grid.map(function(r) { return r.brier_cv; });
  var targetHl = meta.optimal_half_life_weeks;
  var optIdx = 0, minDist = Infinity;
  labels.forEach(function(hl, i) { var d = Math.abs(hl - targetHl); if (d < minDist) { minDist = d; optIdx = i; } });

  new Chart(document.getElementById('lambda-chart').getContext('2d'), {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'CV Brier',
          data: briers,
          borderColor: '#5a2a7a',
          backgroundColor: '#d4b8f422',
          borderWidth: 2,
          pointBackgroundColor: labels.map(function(_, i) { return i === optIdx ? '#5a2a7a' : 'transparent'; }),
          pointRadius: labels.map(function(_, i) { return i === optIdx ? 6 : 2; }),
          tension: 0.3,
          fill: true,
        },
        {
          label: 'Baseline',
          data: labels.map(function() { return 0.25; }),
          borderColor: '#f4b8c1',
          borderWidth: 1.5,
          borderDash: [4, 4],
          pointRadius: 0,
          fill: false,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      scales: {
        x: {
          type: 'logarithmic',
          title: { display: true, text: 'Half-life (matches)', font: { size: 10 }, color: '#7a6e7e' },
          ticks: { font: { size: 9 }, color: '#7a6e7e',
            callback: function(v) { return [5,10,20,50,100,200,500].indexOf(Math.round(v)) >= 0 ? Math.round(v) : ''; }
          },
          grid: { color: '#f0ecf4' }
        },
        y: {
          title: { display: true, text: 'Brier score', font: { size: 10 }, color: '#7a6e7e' },
          ticks: { font: { size: 9 }, color: '#7a6e7e', callback: function(v) { return v.toFixed(3); } },
          grid: { color: '#f0ecf4' }
        }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: function(items) { return 'Half-life: ' + items[0].label + ' matches'; },
            label: function(item) { return item.dataset.label + ': ' + item.raw.toFixed(5); }
          }
        }
      }
    }
  });
  } // end if grid.length

  renderPeriodFilter();
  renderMapFilter();
  renderTable();
})();
</script>
<script>PW_JS</script>
</body>
</html>
""".replace('SHARED_CSS', SHARED_CSS).replace('PW_JS', PW_JS)

MAPELO_MATCHUP_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Matchup Predictor &mdash; BenPom</title>
<link rel="icon" type="image/svg+xml" href="/favicon.svg">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  SHARED_CSS
  .page { position:relative; z-index:1; padding:32px; max-width:980px; margin:0 auto; }
  .page-title { font-family:'Syne',sans-serif; font-size:clamp(1.4rem,3vw,2.2rem); font-weight:800; letter-spacing:-1px; margin-bottom:6px; }
  .page-sub { font-size:.83rem; color:var(--soft); margin-bottom:28px; line-height:1.5; }
  /* Team selector panels */
  .teams-grid { display:grid; grid-template-columns:1fr 96px 1fr; gap:0; align-items:start; margin-bottom:24px; }
  .team-panel { background:white; border-radius:24px; padding:22px 24px; box-shadow:0 4px 24px #0000000a; }
  .tp-side { font-family:'Syne',sans-serif; font-size:.58rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase; color:var(--soft); margin-bottom:14px; }
  .yr-row { display:flex; gap:5px; margin-bottom:10px; }
  .yr-btn { padding:3px 10px; border-radius:99px; border:1.5px solid #f0ecf4; background:white; font-family:'DM Sans',sans-serif; font-size:.72rem; font-weight:500; cursor:pointer; color:var(--soft); transition:all .15s; }
  .yr-btn:hover { border-color:#d4b8f4; color:var(--ink); }
  .yr-btn.active { background:var(--ink); color:white; border-color:var(--ink); }
  .snap-sel { appearance:none; padding:5px 26px 5px 11px; border-radius:99px; border:1.5px solid #f0ecf4; background:white url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='9' height='5'%3E%3Cpath d='M0 0l4.5 5 4.5-5z' fill='%237a6e7e'/%3E%3C/svg%3E") no-repeat right 10px center; font-family:'DM Sans',sans-serif; font-size:.74rem; color:var(--ink); cursor:pointer; outline:none; margin-bottom:10px; display:block; }
  .snap-sel:focus { border-color:#d4b8f4; }
  .team-sel { width:100%; border:2px solid #f0ecf4; border-radius:12px; padding:9px 12px; font-size:.92rem; font-family:'Syne',sans-serif; font-weight:800; background:white; color:var(--ink); cursor:pointer; appearance:none; outline:none; transition:border-color .15s; }
  .team-sel:focus { border-color:#d4b8f4; }
  /* VS column */
  .vs-col { display:flex; flex-direction:column; align-items:center; justify-content:center; padding-top:0; }
  .vs-text { font-family:'Syne',sans-serif; font-weight:800; font-size:.95rem; color:#c0b8c8; }
  .sim-btn { background:#2a1f2d; color:white; border:none; border-radius:99px; padding:9px 18px; font-family:'Syne',sans-serif; font-size:.68rem; font-weight:800; letter-spacing:.07em; text-transform:uppercase; cursor:pointer; transition:background .15s; white-space:nowrap; }
  .sim-btn:hover { background:#5a2a7a; }
  .controls-row { display:flex; align-items:center; justify-content:center; gap:10px; margin-bottom:24px; margin-top:-10px; }
  .fmt-row { display:flex; gap:5px; }
  .fmt-btn { padding:5px 16px; border-radius:99px; border:1.5px solid #f0ecf4; background:white; font-family:'Syne',sans-serif; font-size:.72rem; font-weight:800; cursor:pointer; color:var(--soft); transition:all .15s; white-space:nowrap; }
  .fmt-btn:hover { border-color:#d4b8f4; color:var(--ink); }
  .fmt-btn.active { background:var(--ink); color:white; border-color:var(--ink); }
  /* Result card */
  .result-card { background:white; border-radius:24px; box-shadow:0 4px 24px #0000000a; overflow:hidden; }
  .result-top { padding:28px 32px 22px; }
  .result-teams-row { display:flex; align-items:center; gap:0; margin-bottom:18px; }
  .result-team-block { flex:1; display:flex; flex-direction:column; align-items:center; gap:5px; }
  .result-logo { width:44px; height:44px; object-fit:contain; }
  .result-org { font-family:'Syne',sans-serif; font-weight:800; font-size:.85rem; }
  .result-ctx { font-size:.68rem; color:var(--soft); text-align:center; line-height:1.4; }
  .result-pct { font-family:'Syne',sans-serif; font-weight:800; font-size:2.4rem; line-height:1; }
  .result-pct.fav { color:#2a1f2d; }
  .result-pct.dog { color:#d0c8d8; }
  .result-mid { flex:0 0 120px; display:flex; flex-direction:column; align-items:center; gap:8px; }
  .result-bar-outer { width:100%; height:8px; border-radius:99px; overflow:hidden; display:flex; }
  .result-bar-a { background:#5a2a7a; height:100%; transition:width .6s ease; }
  .result-bar-b { background:#e0d8ec; height:100%; transition:width .6s ease; }
  .result-bar-label { font-family:'Syne',sans-serif; font-size:.55rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; color:var(--soft); }
  /* Legend */
  .fate-legend { display:flex; gap:12px; flex-wrap:wrap; padding:0 32px 14px; }
  .fate-legend-item { display:flex; align-items:center; gap:5px; font-size:.67rem; color:var(--soft); }
  .fate-dot { width:10px; height:10px; border-radius:2px; flex-shrink:0; }
  /* Map table */
  .map-tbl { width:100%; border-collapse:collapse; font-size:.81rem; }
  .map-tbl thead th { font-family:'Syne',sans-serif; font-size:.6rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase; color:var(--soft); padding:8px 14px; border-top:1px solid #f0ecf4; border-bottom:1px solid #f0ecf4; text-align:center; background:#faf8fc; white-space:nowrap; }
  .map-tbl thead th:first-child { text-align:left; }
  .map-tbl tbody tr { border-bottom:1px solid #f8f4fc; transition:background .1s; }
  .map-tbl tbody tr:last-child { border-bottom:none; }
  .map-tbl tbody tr:hover { background:#fdf6f0; }
  .map-tbl tbody td { padding:10px 14px; text-align:center; vertical-align:middle; }
  .map-tbl tbody td:first-child { text-align:left; font-family:'Syne',sans-serif; font-weight:800; }
  /* Fate bar */
  .fate-bar-wrap { display:flex; flex-direction:column; align-items:center; gap:3px; }
  .fate-bar { display:flex; border-radius:4px; overflow:hidden; height:8px; width:110px; background:#f0ecf4; }
  .fate-seg { height:100%; }
  .fs-banA  { background:#f4b8c1; }
  .fs-pickA { background:#5a2a7a; }
  .fs-dec   { background:#c8b8d8; }
  .fs-pickB { background:#7ab8e8; }
  .fs-banB  { background:#b8e8d4; }
  .fate-txt { font-size:.67rem; color:var(--soft); white-space:nowrap; }
  /* Win% bar */
  .wp-cell { display:flex; align-items:center; gap:7px; justify-content:center; }
  .wp-bg { width:52px; height:5px; border-radius:3px; background:#f0ecf4; overflow:hidden; flex-shrink:0; }
  .wp-fill { height:100%; border-radius:3px; background:#5a2a7a; }
  .wp-num { font-size:.78rem; font-weight:600; min-width:30px; text-align:left; }
  /* Rating colors */
  .rt-pos { color:#1a6a4a; font-weight:700; }
  .rt-neg { color:#7a1a1a; font-weight:700; }
  .rt-neu { color:var(--soft); }
  .rd-pos { color:#1a6a4a; font-weight:700; }
  .rd-neg { color:#7a1a1a; font-weight:700; }
  .result-note { font-size:.68rem; color:var(--soft); text-align:center; padding:12px 32px 18px; opacity:.75; }
  .intl-adj { display:flex; align-items:center; gap:4px; margin:3px 0 5px; font-size:.72rem; flex-wrap:wrap; }
  .intl-adj-label { color:var(--soft); font-size:.65rem; letter-spacing:.05em; text-transform:uppercase; font-weight:700; }
  .intl-adj-val { font-family:'Syne',sans-serif; font-weight:800; font-size:.75rem; }
  .intl-adj-tip { color:var(--soft); font-size:.65rem; }
  /* Predicted veto */
  .veto-pred-card { background:white; border-radius:24px; box-shadow:0 4px 24px #0000000a; padding:24px 28px; margin-bottom:20px; }
  .veto-pred-title { font-family:'Syne',sans-serif; font-size:.65rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; color:var(--soft); margin-bottom:16px; }
  .veto-seq { margin-bottom:14px; padding-bottom:14px; border-bottom:1px solid #f0ecf4; }
  .veto-seq:last-child { margin-bottom:0; padding-bottom:0; border-bottom:none; }
  .veto-seq-header { display:flex; align-items:center; gap:10px; margin-bottom:8px; }
  .veto-seq-rank { font-family:'Syne',sans-serif; font-size:.68rem; font-weight:800; color:var(--soft); }
  .veto-seq-pct { font-family:'Syne',sans-serif; font-size:.78rem; font-weight:800; color:#2a1f2d; background:#f4f0fa; border-radius:99px; padding:2px 10px; }
  .veto-steps { display:flex; gap:6px; flex-wrap:wrap; align-items:center; }
  .veto-step { display:flex; flex-direction:column; align-items:center; gap:2px; }
  .step-lbl { font-size:.55rem; font-weight:800; letter-spacing:.07em; text-transform:uppercase; border-radius:4px; padding:2px 5px; white-space:nowrap; }
  .step-lbl-banA  { background:#fde8ec; color:#b03050; }
  .step-lbl-banB  { background:#e8f8ee; color:#206040; }
  .step-lbl-pickA { background:#ede0f8; color:#5a2a7a; }
  .step-lbl-pickB { background:#deeef8; color:#1a508a; }
  .step-lbl-dec   { background:#f0ecf4; color:#7a6e7e; }
  .step-map { font-family:'Syne',sans-serif; font-weight:800; font-size:.72rem; color:#2a1f2d; white-space:nowrap; }
  .step-arrow { font-size:.7rem; color:#ccc; align-self:center; margin-top:10px; }
  .no-veto-data { font-size:.78rem; color:var(--soft); font-style:italic; }
  @media(max-width:700px){
    .teams-grid { grid-template-columns:1fr; }
    .vs-col { flex-direction:row; padding:10px 0; justify-content:center; }
    .result-mid { flex:0 0 80px; }
    .result-pct { font-size:1.8rem; }
  }
</style>
</head>
<body>
<div id="content-wrap">
  <div class="top-nav">
    <a href="/"><img src="/logo.svg" alt="Home" class="home-logo"></a>
    <a class="back-link" href="/mapelo/">&larr; BenPom</a>
  </div>
  <div class="page">
    <div class="page-title">Matchup Predictor</div>
    <p class="page-sub">Monte Carlo veto simulation. Each team can be set to a different season and timeframe.</p>

    <div class="teams-grid">
      <div class="team-panel">
        <div class="tp-side">Team A</div>
        <div class="yr-row">
          <button class="yr-btn" data-side="a" data-year="2026">2026</button>
          <button class="yr-btn active" data-side="a" data-year="2025">2025</button>
          <button class="yr-btn" data-side="a" data-year="2024">2024</button>
          <button class="yr-btn" data-side="a" data-year="2023">2023</button>
        </div>
        <select id="snap-a" class="snap-sel"></select>
        <select id="team-a" class="team-sel"></select>
      </div>
      <div class="vs-col">
        <div class="vs-text">vs</div>
      </div>
      <div class="team-panel">
        <div class="tp-side">Team B</div>
        <div class="yr-row">
          <button class="yr-btn" data-side="b" data-year="2026">2026</button>
          <button class="yr-btn active" data-side="b" data-year="2025">2025</button>
          <button class="yr-btn" data-side="b" data-year="2024">2024</button>
          <button class="yr-btn" data-side="b" data-year="2023">2023</button>
        </div>
        <select id="snap-b" class="snap-sel"></select>
        <select id="team-b" class="team-sel"></select>
      </div>
    </div>

    <div class="controls-row">
      <div class="fmt-row">
        <button class="fmt-btn" data-fmt="bo1">Bo1</button>
        <button class="fmt-btn active" data-fmt="bo3">Bo3</button>
        <button class="fmt-btn" data-fmt="bo5">Bo5</button>
      </div>
      <button class="sim-btn" onclick="runMatchup()">Run</button>
    </div>

    <div id="result-section"></div>
  </div>
</div>
<script>
var DATA = RATINGS_JSON;
var VETO = DATA.veto_model || {teams:{}, snap_pools:{}};
var INTL = DATA.intl_calib || {};
var INTL_PARAMS = DATA.intl_params || {};
var ORG_REGIONS = DATA.org_regions || {};
var yearA = '2025', snapA = 'after_champions';
var yearB = '2025', snapB = 'after_champions';
var fmt = 'bo3';

function getGlobalRating(org, snapKey, domesticRating) {
  var cal = INTL[snapKey] || {};
  var region = ORG_REGIONS[org] || '';
  var regOff = (cal.regional_offsets || {})[region] || 0;
  var indBonus = (cal.individual_bonuses || {})[org] || 0;
  return domesticRating + regOff + indBonus;
}
function getIntlBreakdown(org, snapKey) {
  var cal = INTL[snapKey] || {};
  var region = ORG_REGIONS[org] || '';
  var regOff = (cal.regional_offsets || {})[region] || 0;
  var indBonus = (cal.individual_bonuses || {})[org] || 0;
  return {region: region, regOff: regOff, indBonus: indBonus, total: regOff + indBonus};
}

var VETO_STEPS = {
  bo1: [
    {side:'A',action:'ban'},{side:'B',action:'ban'},
    {side:'A',action:'ban'},{side:'B',action:'ban'},
    {side:'A',action:'ban'},{side:'B',action:'ban'},
  ],
  bo3: [
    {side:'A',action:'ban'},{side:'B',action:'ban'},
    {side:'A',action:'pick'},{side:'B',action:'pick'},
    {side:'A',action:'ban'},{side:'B',action:'ban'},
  ],
  bo5: [
    {side:'A',action:'ban'},{side:'B',action:'ban'},
    {side:'A',action:'pick'},{side:'B',action:'pick'},
    {side:'A',action:'pick'},{side:'B',action:'pick'},
  ],
};
var SERIES_THRESH = {bo1:1, bo3:2, bo5:3};

function getSnapsFor(year) {
  var yr = DATA.ratings[year];
  return (yr && yr.snapshots) ? yr.snapshots : {};
}
function getLastSnap(year) {
  var keys = Object.keys(getSnapsFor(year));
  if (keys.indexOf('after_champions') >= 0) return 'after_champions';
  return keys[keys.length - 1] || 'after_champions';
}
function getSnapData(year, snap) {
  var snaps = getSnapsFor(year);
  return snaps[snap] || snaps['after_champions'] || snaps['before_champions'] || snaps[Object.keys(snaps)[0]] || {};
}
function populateSnapSel(side) {
  var year = side==='a' ? yearA : yearB;
  var cur  = side==='a' ? snapA  : snapB;
  var snaps = getSnapsFor(year), keys = Object.keys(snaps);
  var sel = document.getElementById('snap-'+side);
  sel.innerHTML = keys.map(function(k){
    return '<option value="'+k+'"'+(k===cur?' selected':'')+'>'+((snaps[k]||{}).label||k)+'</option>';
  }).join('');
  if (!snaps[cur]){ var f=keys[0]; if(side==='a') snapA=f; else snapB=f; sel.value=f; }
}
function populateTeamSel(side) {
  var year=side==='a'?yearA:yearB, snap=side==='a'?snapA:snapB;
  var teams=Object.keys((getSnapData(year,snap).teams)||{}).sort();
  var sel=document.getElementById('team-'+side), cur=sel.value;
  sel.innerHTML=teams.map(function(t){ return '<option value="'+t+'">'+t+'</option>'; }).join('');
  if(teams.indexOf(cur)>=0) sel.value=cur;
  else sel.value=(side==='a'?teams[0]:teams[1])||teams[0]||'';
}
document.querySelectorAll('.yr-btn').forEach(function(btn){
  btn.addEventListener('click', function(){
    var side=btn.dataset.side, year=btn.dataset.year;
    if(side==='a'){ if(yearA===year) return; yearA=year; snapA=getLastSnap(year); }
    else           { if(yearB===year) return; yearB=year; snapB=getLastSnap(year); }
    document.querySelectorAll('.yr-btn[data-side="'+side+'"]').forEach(function(b){ b.classList.remove('active'); });
    btn.classList.add('active');
    populateSnapSel(side); populateTeamSel(side);
    document.getElementById('result-section').innerHTML='';
  });
});
document.getElementById('snap-a').addEventListener('change', function(){ snapA=this.value; populateTeamSel('a'); document.getElementById('result-section').innerHTML=''; });
document.getElementById('snap-b').addEventListener('change', function(){ snapB=this.value; populateTeamSel('b'); document.getElementById('result-section').innerHTML=''; });
document.querySelectorAll('.fmt-btn').forEach(function(btn){
  btn.addEventListener('click', function(){
    if(fmt===btn.dataset.fmt) return;
    fmt=btn.dataset.fmt;
    document.querySelectorAll('.fmt-btn').forEach(function(b){ b.classList.remove('active'); });
    btn.classList.add('active');
    document.getElementById('result-section').innerHTML='';
  });
});

// ── Veto model ───────────────────────────────────────────────────────────────

function getActivePool(year, snap) {
  return (VETO.snap_pools||{})[year+'_'+snap] || null;
}

function getBanProbs(patt, oppTeam, rem) {
  var scores = {};
  rem.forEach(function(m) {
    var rate = (patt && patt.bans && patt.bans[m] != null) ? patt.bans[m] : 0;
    var oppWin = (oppTeam && oppTeam.maps && oppTeam.maps[m]) ? (oppTeam.maps[m].win_pct||0.5) : 0.5;
    // Primary: historical ban tendency. Secondary: boost for opponent's strong maps.
    scores[m] = (rate + 0.02) * (0.75 + oppWin);
  });
  var tot = rem.reduce(function(s,m){ return s+scores[m]; }, 0);
  if(tot===0) rem.forEach(function(m){ scores[m]=1/rem.length; });
  else rem.forEach(function(m){ scores[m]/=tot; });
  return scores;
}

function getPickProbs(patt, rem) {
  var scores = {};
  rem.forEach(function(m) {
    scores[m] = (patt && patt.picks && patt.picks[m] != null) ? (patt.picks[m]+0.02) : (1/rem.length);
  });
  var tot = rem.reduce(function(s,m){ return s+scores[m]; }, 0);
  if(tot===0) rem.forEach(function(m){ scores[m]=1/rem.length; });
  else rem.forEach(function(m){ scores[m]/=tot; });
  return scores;
}

function sampleFrom(probs) {
  var r=Math.random(), cum=0, keys=Object.keys(probs);
  for(var i=0;i<keys.length;i++){ cum+=probs[keys[i]]; if(r<=cum) return keys[i]; }
  return keys[keys.length-1];
}

function simulateVetoMC(tA, tB, orgA, orgB, pool, yA, yB, sA, sB, f) {
  var pA=((VETO.teams||{})[yA+'_'+sA]||{})[orgA]||null;
  var pB=((VETO.teams||{})[yB+'_'+sB]||{})[orgB]||null;
  var rem=pool.slice(), fate={};
  (VETO_STEPS[f]||VETO_STEPS.bo3).forEach(function(step){
    var patt=step.side==='A'?pA:pB, oppT=step.side==='A'?tB:tA;
    var lbl=step.action+(step.side==='A'?'A':'B');
    var m=step.action==='ban'?sampleFrom(getBanProbs(patt,oppT,rem)):sampleFrom(getPickProbs(patt,rem));
    fate[m]=lbl; rem=rem.filter(function(x){return x!==m;});
  });
  if(rem.length) fate[rem[0]]='dec';
  return fate;
}

function topVetoSequences(tA, tB, orgA, orgB, pool, yA, yB, sA, sB, f, K) {
  var pA=((VETO.teams||{})[yA+'_'+sA]||{})[orgA]||null;
  var pB=((VETO.teams||{})[yB+'_'+sB]||{})[orgB]||null;
  K = K||3;
  var steps = VETO_STEPS[f]||VETO_STEPS.bo3;
  var states=[{rem:pool.slice(),seq:[],prob:1.0}];
  steps.forEach(function(step){
    var next=[];
    states.forEach(function(st){
      var patt=step.side==='A'?pA:pB, oppT=step.side==='A'?tB:tA;
      var probs=step.action==='ban'?getBanProbs(patt,oppT,st.rem):getPickProbs(patt,st.rem);
      st.rem.forEach(function(m){
        var p=probs[m]||0;
        if(p>0.005) next.push({
          rem: st.rem.filter(function(x){return x!==m;}),
          seq: st.seq.concat([{side:step.side,action:step.action,map:m}]),
          prob: st.prob*p
        });
      });
    });
    next.sort(function(a,b){return b.prob-a.prob;});
    states=next.slice(0,K*3);
  });
  states.forEach(function(st){ if(st.rem.length) st.seq.push({side:'',action:'dec',map:st.rem[0]}); });
  states.sort(function(a,b){return b.prob-a.prob;});
  return states.slice(0,K);
}

// ── Main ─────────────────────────────────────────────────────────────────────

function runMatchup() {
  var orgA=document.getElementById('team-a').value;
  var orgB=document.getElementById('team-b').value;
  if(!orgA||!orgB) return;
  var sdA=getSnapData(yearA,snapA), sdB=getSnapData(yearB,snapB);
  var tA=(sdA.teams||{})[orgA], tB=(sdB.teams||{})[orgB];
  if(!tA||!tB) return;
  var rawBeta=((sdA.beta||0.08)+(sdB.beta||0.08))/2;
  var crossRegional = (ORG_REGIONS[orgA] && ORG_REGIONS[orgB] && ORG_REGIONS[orgA] !== ORG_REGIONS[orgB]);
  var beta = crossRegional ? rawBeta * (INTL_PARAMS.cross_regional_beta_mult || 1.0) : rawBeta;

  // Map pool: use the older era so both teams have played every map in it
  var rdA = sdA.ref_date || (yearA + '-01-01');
  var rdB = sdB.ref_date || (yearB + '-01-01');
  var olderYear = rdA <= rdB ? yearA : yearB;
  var olderSnap = rdA <= rdB ? snapA  : snapB;
  var pool = getActivePool(olderYear, olderSnap);
  if(!pool){
    var seen={}; Object.keys(tA.maps||{}).forEach(function(m){seen[m]=true;}); Object.keys(tB.maps||{}).forEach(function(m){seen[m]=true;});
    pool=Object.keys(seen).sort();
  }

  var thresh = SERIES_THRESH[fmt]||2;
  var snapKeyA = yearA+'_'+snapA, snapKeyB = yearB+'_'+snapB;
  var intlA = getIntlBreakdown(orgA, snapKeyA), intlB = getIntlBreakdown(orgB, snapKeyB);
  var nSims=10000, seriesWins=0;
  var fateCnt={banA:{},pickA:{},dec:{},pickB:{},banB:{}};
  var mapWins={}, mapPlays={};
  pool.forEach(function(m){ mapWins[m]=0; mapPlays[m]=0; Object.keys(fateCnt).forEach(function(fc){fateCnt[fc][m]=0;}); });

  for(var s=0;s<nSims;s++){
    var fm=simulateVetoMC(tA,tB,orgA,orgB,pool,yearA,yearB,snapA,snapB,fmt), sw=0;
    pool.forEach(function(m){
      var fc=fm[m]||'banA';
      if(fateCnt[fc]) fateCnt[fc][m]++;
      if(fc==='pickA'||fc==='pickB'||fc==='dec'){
        mapPlays[m]++;
        var dA=(tA.maps[m]||{}).rating!=null?(tA.maps[m]||{}).rating:tA.overall_rating;
        var dB=(tB.maps[m]||{}).rating!=null?(tB.maps[m]||{}).rating:tB.overall_rating;
        var rA=getGlobalRating(orgA, snapKeyA, dA), rB=getGlobalRating(orgB, snapKeyB, dB);
        if(Math.random()<1/(1+Math.exp(-beta*(rA-rB)))){ sw++; mapWins[m]++; }
      }
    });
    if(sw>=thresh) seriesWins++;
  }
  var pA_=seriesWins/nSims, pctA=Math.round(pA_*100), pctB=100-pctA;
  var lblA=((getSnapsFor(yearA)[snapA])||{}).label||snapA;
  var lblB=((getSnapsFor(yearB)[snapB])||{}).label||snapB;
  var fmtLabel = fmt==='bo1'?'Map win prob.':fmt==='bo5'?'Series win prob. (Bo5)':'Series win prob. (Bo3)';

  // ── Beam search for top predicted sequences ──
  var topSeqs = topVetoSequences(tA,tB,orgA,orgB,pool,yearA,yearB,snapA,snapB,fmt,3);
  var hasPatt = (((VETO.teams||{})[yearA+'_'+snapA]||{})[orgA] || ((VETO.teams||{})[yearB+'_'+snapB]||{})[orgB]);
  var RANK_LABELS = ['#1 Most likely','#2','#3'];
  var ACTION_LBCLS = {banA:['Ban A','step-lbl-banA'],banB:['Ban B','step-lbl-banB'],pickA:['Pick A','step-lbl-pickA'],pickB:['Pick B','step-lbl-pickB'],dec:['Decider','step-lbl-dec']};
  var vetoHtml;
  if(!hasPatt){
    vetoHtml='<div class="no-veto-data">No historical veto data available for these teams in the selected year.</div>';
  } else {
    vetoHtml = topSeqs.map(function(seq,idx){
      var pct = Math.round(seq.prob*100);
      var steps = seq.seq.map(function(step,si){
        var ac = ACTION_LBCLS[step.action+step.side]||['?','step-lbl-dec'];
        var arrow = si < seq.seq.length-1 ? '<span class="step-arrow">›</span>' : '';
        return '<div class="veto-step"><span class="step-lbl '+ac[1]+'">'+ac[0]+'</span><span class="step-map">'+step.map+'</span></div>'+arrow;
      }).join('');
      return '<div class="veto-seq">'+
        '<div class="veto-seq-header">'+
          '<span class="veto-seq-rank">'+RANK_LABELS[idx]+'</span>'+
          '<span class="veto-seq-pct">~'+pct+'%</span>'+
        '</div>'+
        '<div class="veto-steps">'+steps+'</div>'+
      '</div>';
    }).join('');
  }

  // ── Map table ──
  var sorted=pool.slice().sort(function(a,b){return mapPlays[b]-mapPlays[a];});
  var rows=sorted.map(function(m){
    var dA=(tA.maps[m]||{}).rating!=null?(tA.maps[m]||{}).rating:tA.overall_rating;
    var dB=(tB.maps[m]||{}).rating!=null?(tB.maps[m]||{}).rating:tB.overall_rating;
    var rA=getGlobalRating(orgA, snapKeyA, dA), rB=getGlobalRating(orgB, snapKeyB, dB);
    var bA_=fateCnt.banA[m]/nSims, pA_m=fateCnt.pickA[m]/nSims, dc=fateCnt.dec[m]/nSims, pB_m=fateCnt.pickB[m]/nSims, bB_=fateCnt.banB[m]/nSims;
    var bar=''; [[bA_,'fs-banA'],[pA_m,'fs-pickA'],[dc,'fs-dec'],[pB_m,'fs-pickB'],[bB_,'fs-banB']].forEach(function(p){ if(p[0]>0.005) bar+='<div class="fate-seg '+p[1]+'" style="width:'+(p[0]*100).toFixed(1)+'%"></div>'; });
    var fv={banA:bA_,pickA:pA_m,pickB:pB_m,banB:bB_,dec:dc};
    var dom='banA'; Object.keys(fv).forEach(function(k){if(fv[k]>fv[dom]) dom=k;});
    var fateLabels={banA:'Banned by A',pickA:'Picked by A',pickB:'Picked by B',banB:'Banned by B',dec:'Decider'};
    var rACls=rA>0.05?'rt-pos':rA<-0.05?'rt-neg':'rt-neu';
    var rBCls=rB>0.05?'rt-pos':rB<-0.05?'rt-neg':'rt-neu';
    var p_m=mapPlays[m]>0?1/(1+Math.exp(-beta*(rA-rB))):0.5;
    var projRd=(2*p_m-1)*13, rdCls=projRd>0.5?'rd-pos':projRd<-0.5?'rd-neg':'rt-neu';
    var wpHtml='<span class="rt-neu">—</span>';
    if(mapPlays[m]>0){
      var wp=mapWins[m]/mapPlays[m], wpCls=wp>=0.55?'rd-pos':wp<=0.45?'rd-neg':'rt-neu';
      wpHtml='<div class="wp-cell"><div class="wp-bg"><div class="wp-fill" style="width:'+Math.round(wp*52)+'px"></div></div><span class="wp-num '+wpCls+'">'+(wp*100).toFixed(0)+'%</span></div>';
    }
    return '<tr><td>'+m+'</td>'+
      '<td><div class="fate-bar-wrap"><div class="fate-bar">'+bar+'</div><div class="fate-txt">'+fateLabels[dom]+'</div></div></td>'+
      '<td class="'+rACls+'">'+(rA>=0?'+':'')+rA.toFixed(2)+'</td>'+
      '<td class="'+rBCls+'">'+(rB>=0?'+':'')+rB.toFixed(2)+'</td>'+
      '<td class="'+rdCls+'">'+(projRd>=0?'+':'')+projRd.toFixed(1)+'</td>'+
      '<td>'+wpHtml+'</td></tr>';
  }).join('');

  // ── International adjustment badges ──
  function intlBadgeHtml(org, bd) {
    if(!bd.region || (!bd.regOff && !bd.indBonus)) return '';
    var tot=bd.total, cls=tot>0.05?'rd-pos':tot<-0.05?'rd-neg':'rt-neu';
    var parts=[];
    if(Math.abs(bd.regOff)>0.005) parts.push((bd.regOff>=0?'+':'')+bd.regOff.toFixed(2)+' '+bd.region+' region');
    if(Math.abs(bd.indBonus)>0.005) parts.push((bd.indBonus>=0?'+':'')+bd.indBonus.toFixed(2)+' indiv.');
    return '<div class="intl-adj"><span class="intl-adj-label">Intl adj</span><span class="intl-adj-val '+cls+'">'+(tot>=0?'+':'')+tot.toFixed(2)+'</span><span class="intl-adj-tip">'+parts.join(', ')+'</span></div>';
  }

  document.getElementById('result-section').innerHTML=
    '<div class="result-card" style="margin-bottom:20px;">'+
      '<div class="result-top">'+
        '<div class="result-teams-row">'+
          '<div class="result-team-block">'+
            '<img src="/logos/'+orgA+'.png" class="result-logo" onerror="this.style.visibility=\\'hidden\\'">'+
            '<div class="result-org">'+orgA+'</div>'+
            '<div class="result-ctx">'+yearA+'&thinsp;&middot;&thinsp;'+lblA+'</div>'+
            intlBadgeHtml(orgA, intlA)+
            '<div class="result-pct '+(pctA>=50?'fav':'dog')+'">'+pctA+'%</div>'+
          '</div>'+
          '<div class="result-mid">'+
            '<div class="result-bar-label">'+fmtLabel+'</div>'+
            '<div class="result-bar-outer"><div class="result-bar-a" style="width:'+pctA+'%"></div><div class="result-bar-b" style="width:'+pctB+'%"></div></div>'+
          '</div>'+
          '<div class="result-team-block">'+
            '<img src="/logos/'+orgB+'.png" class="result-logo" onerror="this.style.visibility=\\'hidden\\'">'+
            '<div class="result-org">'+orgB+'</div>'+
            '<div class="result-ctx">'+yearB+'&thinsp;&middot;&thinsp;'+lblB+'</div>'+
            intlBadgeHtml(orgB, intlB)+
            '<div class="result-pct '+(pctB>=50?'fav':'dog')+'">'+pctB+'%</div>'+
          '</div>'+
        '</div>'+
      '</div>'+
    '</div>'+
    '<div class="veto-pred-card">'+
      '<div class="veto-pred-title">Predicted Veto — '+orgA+' vs '+orgB+'</div>'+
      vetoHtml+
    '</div>'+
    '<div class="result-card">'+
      '<div class="fate-legend">'+
        '<div class="fate-legend-item"><div class="fate-dot" style="background:#f4b8c1"></div>Banned by A</div>'+
        '<div class="fate-legend-item"><div class="fate-dot" style="background:#5a2a7a"></div>Picked by A</div>'+
        '<div class="fate-legend-item"><div class="fate-dot" style="background:#c8b8d8"></div>Decider</div>'+
        '<div class="fate-legend-item"><div class="fate-dot" style="background:#7ab8e8"></div>Picked by B</div>'+
        '<div class="fate-legend-item"><div class="fate-dot" style="background:#b8e8d4"></div>Banned by B</div>'+
      '</div>'+
      '<table class="map-tbl"><thead><tr>'+
        '<th>Map</th><th>Veto outcome</th>'+
        '<th>'+orgA+' rtg</th><th>'+orgB+' rtg</th>'+
        '<th>Proj. RD (A)</th><th>A win% if played</th>'+
      '</tr></thead><tbody>'+rows+'</tbody></table>'+
      '<div class="result-note">'+nSims.toLocaleString()+' simulations &middot; '+fmt.toUpperCase()+' &middot; veto driven by historical ban/pick patterns &middot; ratings not normalized across seasons</div>'+
    '</div>';
}

(function(){
  populateSnapSel('a'); populateSnapSel('b');
  populateTeamSel('a'); populateTeamSel('b');
})();
</script>
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
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
  SHARED_CSS
  .page { position:relative; z-index:1; padding:32px; max-width:1000px; margin:0 auto; width:100%; }
  .page-title { font-family:'Syne',sans-serif; font-size:clamp(1.6rem,4vw,2.5rem); font-weight:800; letter-spacing:-1px; margin-bottom:28px; }
  .card { background:white; border-radius:24px; padding:28px 32px; box-shadow:0 4px 24px #0000000a; }
  .card-header { display:flex; align-items:baseline; gap:14px; margin-bottom:6px; flex-wrap:wrap; }
  .exponent-badge { font-size:.75rem; font-weight:500; background:#f4edb8; color:#6a5a1a; padding:3px 10px; border-radius:99px; }
  .card-desc { font-size:.82rem; color:var(--soft); line-height:1.6; margin-bottom:18px; }
  .intro-details { max-width:780px; margin:0 auto 32px; }
  .intro-details summary { font-family:'Syne',sans-serif; font-weight:800; font-size:.95rem; letter-spacing:.02em; cursor:pointer; list-style:none; display:flex; align-items:center; gap:8px; color:var(--soft); user-select:none; margin-bottom:0; }
  .intro-details summary::-webkit-details-marker { display:none; }
  .intro-details summary::before { content:'▸'; font-size:.75rem; transition:transform .2s; display:inline-block; }
  .intro-details[open] summary::before { transform:rotate(90deg); }
  .intro-details[open] summary { margin-bottom:18px; }
  .intro-body { display:flex; flex-direction:column; gap:14px; overflow:hidden; transition:max-height .35s ease; }
  .intro-p { font-size:.9rem; color:var(--ink); line-height:1.75; }
  .intro-note { background:#f8f4fc; border-radius:16px; padding:18px 22px; display:flex; flex-direction:column; gap:10px; }
  .intro-note-label { font-family:'Syne',sans-serif; font-weight:800; font-size:.8rem; letter-spacing:.06em; text-transform:uppercase; color:var(--soft); }
  .intro-note-list { padding-left:1.4em; display:flex; flex-direction:column; gap:10px; }
  .intro-note-subp { margin-top:8px; }
  .intro-formula-block { background:#f8f4fc; border-radius:14px; padding:12px 20px; text-align:center; }
  .section-divider { border:none; border-top:1px solid #f0ecf4; margin:4px 0; }
  .formula-block { background:#f8f4fc; border-radius:14px; padding:14px 20px; margin-bottom:16px; text-align:center; }
  .formula { font-size:1.05rem; color:var(--ink); margin-bottom:6px; }
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
    <a class="back-link" href="/mapelo/">&larr; BenPom</a>
    <div class="page-title">Pythagorean Win%</div>

    <details class="intro-details" open>
      <summary>Explanation</summary>
      <div class="intro-body">
        <p class="intro-p">The Pythagorean Rating formula originates from baseball statistician Bill James, who crafted a formula that settles the discrepancy between how many games a team <em>should</em> win vs. how many they actually won by using a team&rsquo;s margins of victory over a season. Specifically, it looks like:</p>
        <div class="intro-formula-block">
          <div id="baseball-formula"></div>
        </div>
        <p class="intro-p">For instance, the 2023 Baltimore Orioles finished 101&ndash;61, the best record in the American League, but their margins were not quite as great as their record. They scored 807 runs and allowed 678, which works out to a Pythagorean record of just 94&ndash;68, seven full wins below their actual mark. This overperformance was immediately realized in the playoffs, where they were first-round exits.</p>
        <p class="intro-p">The brilliance of this framework is that it can be applied to any sport, so long as the exponent is tuned to minimize the MSE. For instance, basketball uses a team&rsquo;s point margins and has an exponent tuned to 13.91. Hockey uses a team&rsquo;s goal margins and has an exponent tuned to 2.15. In this school of thought, I personally tuned Bill James&rsquo; formula to VCT by using round-differentials.</p>
        <p class="intro-p">The below Pyth% is a mathematically-proven way of seeing the true strength level of a team relative to the year &mdash; the true rate at which they should win maps.</p>
        <hr class="section-divider">
        <div class="intro-note">
          <div class="intro-note-label">Additional Note</div>
          <ul class="intro-note-list">
            <li class="intro-p">In keeping with Bill James&rsquo; framework, only domestic events are used to calculate Pyth%. This is because in domestic splits, a team&rsquo;s schedule is balanced to play all opponents, or at least an even distribution of opponents by strength. Adding internationals would skew the teams&rsquo; Pyth%.
              <p class="intro-p intro-note-subp">For example, at LOCK//IN, NAVI played Kr&uuml; (2023 Pyth% of 37.6%), TS (2023 Pyth% of 47.7%), Lev (2023 Pyth% of 47.3%), and eventually Fnatic (who were the best team of 2023). Meanwhile, a team like Sentinels just played Fnatic (again, the best team of 2023), where they got stomped 6&ndash;13 and 7&ndash;13. The two teams clearly got different luck when it came to their LOCK//IN draw. If internationals were included in Pyth%, NAVI&rsquo;s 2023 value would be unfairly skewed upwards, and Sentinels&rsquo; 2023 value would be unfairly skewed downwards.</p>
            </li>
            <li class="intro-p">If you&rsquo;re interested in seeing international Pyth%, those are calculated separately and located within the all-time category in the &ldquo;Internationals&rdquo; filter.</li>
          </ul>
        </div>
      </div>
    </details>

    <div class="card">
      <p class="card-desc"></p>
      <div class="formula-block">
        <div class="formula" id="pyth-formula"></div>
        <div class="formula-caption" id="pyth-caption"></div>
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

document.addEventListener('DOMContentLoaded', function() {
  katex.render('\\\\text{Win\\\\%} \\\\approx \\\\dfrac{RS^{1.83}}{RS^{1.83} + RA^{1.83}}',
    document.getElementById('baseball-formula'),
    { throwOnError: false, displayMode: true });
  katex.render('\\\\text{Pyth\\\\%} \\\\approx \\\\dfrac{RW^k}{RW^k + RL^k}',
    document.getElementById('pyth-formula'),
    { throwOnError: false, displayMode: true });
  var cap = document.getElementById('pyth-caption');
  cap.innerHTML =
    katex.renderToString('RW', {throwOnError:false}) + ' = rounds won  |  ' +
    katex.renderToString('RL', {throwOnError:false}) + ' = rounds lost  |  ' +
    katex.renderToString('k',  {throwOnError:false}) + ' = optimal exponent fit to VCT data';

  var details = document.querySelector('.intro-details');
  var body    = details.querySelector('.intro-body');
  body.style.maxHeight = body.scrollHeight + 'px';
  details.querySelector('summary').addEventListener('click', function(e) {
    e.preventDefault();
    if (details.open) {
      body.style.maxHeight = body.scrollHeight + 'px';
      requestAnimationFrame(function() { body.style.maxHeight = '0'; });
      setTimeout(function() { details.removeAttribute('open'); }, 350);
    } else {
      details.setAttribute('open', '');
      body.style.maxHeight = '0';
      requestAnimationFrame(function() { body.style.maxHeight = body.scrollHeight + 'px'; });
    }
  });
});


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

function teamLabel(year, split) {
  if (split) return year + ' ' + split;
  if (isAllTime) return year + ' Domestic';
  if (activeKey === activeYear) return year + ' Domestic';
  var evs = PYTH.events_by_year[String(year || activeYear)] || [];
  var ev  = evs.find(function(e) { return e.id === activeKey; });
  return year + (ev ? ' ' + ev.label : '');
}

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
var allTimeFullBtn  = makeTab('Full Year (Domestic)', true,  '', function() { showAllTime(false);   });
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

  var allBtn = makeTab('All (Domestic)', activeKey === activeYear, '', function() { setKey(activeYear); });
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
    '<div class="modal-title">' + teamLabel(year, split) + ' ' + org + ' &mdash; Map Results</div>' +
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
    '<div class="team-modal-header">' + logoTag + teamLabel(year, split) + ' ' + org + '</div>' +
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
def mapelo_hub():
    return MAPELO_HUB_HTML

@mapelo_bp.route('/rankings/')
def mapelo_home():
    full = get_ratings()
    intl = get_intl_calibration()
    keep_meta = ('optimal_half_life_matches', 'brier_test', 'n_train', 'n_test', 'mc_n_sims', 'veto_noise_std')
    frontend_data = {
        'metadata':     {k: v for k, v in full['metadata'].items() if k in keep_meta},
        'lambda_grid':  full.get('lambda_grid', []),
        'ratings':      full['ratings'],
        'intl_calib':   intl.get('calibration', {}),
        'intl_params':  intl.get('params', {}),
        'org_regions':  ORG_REGIONS,
    }
    return MAPELO_HOME_HTML.replace('RATINGS_JSON', json.dumps(frontend_data))

@mapelo_bp.route('/matchup/')
def mapelo_matchup():
    full  = get_ratings()
    veto  = get_veto_model()
    intl  = get_intl_calibration()
    keep_meta = ('optimal_half_life_matches', 'brier_test', 'n_train', 'n_test', 'mc_n_sims', 'veto_noise_std')
    frontend_data = {
        'metadata':    {k: v for k, v in full['metadata'].items() if k in keep_meta},
        'ratings':     full['ratings'],
        'veto_model':  {'teams': veto.get('teams', {}), 'snap_pools': veto.get('snap_pools', {})},
        'intl_calib':  intl.get('calibration', {}),
        'intl_params': intl.get('params', {}),
        'org_regions': ORG_REGIONS,
    }
    return MAPELO_MATCHUP_HTML.replace('RATINGS_JSON', json.dumps(frontend_data))

@mapelo_bp.route('/pythagorean/')
def mapelo_pythagorean():
    data = get_pyth_data()
    return MAPELO_PYTH_HTML.replace('PYTH_JSON', json.dumps(data))

@mapelo_bp.route('/team-info/<org>')
def mapelo_team_info(org):
    from flask import request as _req
    year = _req.args.get('year', '2025')
    snap = _req.args.get('snap', 'after_champions')
    data = _get_team_info(org, year, snap)
    return Response(json.dumps(data), mimetype='application/json')

@mapelo_bp.route('/map-matches/<org>/<map_name>')
def mapelo_map_matches(org, map_name):
    from flask import request as _req
    year = _req.args.get('year', '2025')
    snap = _req.args.get('snap', 'after_champions')
    data = _get_map_matches(org, map_name, year, snap)
    return Response(json.dumps(data), mimetype='application/json')
