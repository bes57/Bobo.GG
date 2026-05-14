import os
import re
import json
import threading as _th
import time as _time_mod
import math as _math_mod
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

    # Canonical split id: CN events fold into their regional counterpart so
    # the page shows ONE "Kickoff" / "Stage 1" / "Stage 2" filter per year
    # that combines EMEA + Americas + Pacific + CN teams. We don't want CN
    # to appear as its own filter — it's part of the same competitive split.
    def canonical_split_id(eid):
        return eid.replace('_china_', '_')

    events_by_year = {}
    seen_per_year = {}
    for e in regional_chron:
        y = str(e['year'])
        canon = canonical_split_id(e['id'])
        if y not in events_by_year:
            events_by_year[y] = []
            seen_per_year[y] = set()
        if canon in seen_per_year[y]:
            continue  # already listed via the regional counterpart
        seen_per_year[y].add(canon)
        # Use the regional (non-CN) event for the label; if only CN exists for
        # this split (shouldn't happen but be safe), strip "China " from the label
        canon_event = next((ev for ev in regional_chron if ev['id'] == canon), e)
        short_label = canon_event['label'].split(' ', 1)[1] if ' ' in canon_event['label'] else canon_event['label']
        short_label = short_label.replace('China ', '')
        events_by_year[y].append({'id': canon, 'label': short_label})

    # Load all map frames — tag each with both the original event_id AND the
    # canonical split_id, so we can later filter by canonical split.
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
        df['event_id'] = canonical_split_id(e['id'])  # canonical: CN folds in
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
        '2023_champions':        ('Champions',          '2023'),
        '2024_masters_madrid':   ('Masters Madrid',     '2024'),
        '2024_champions':        ('Champions',          '2024'),
        '2025_masters_bangkok':  ('Masters Bangkok',    '2025'),
        '2025_masters_toronto':  ('Masters Toronto',    '2025'),
        '2025_champions':        ('Champions',          '2025'),
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

    from datetime import date as _date
    today_str = _date.today().isoformat()

    for year in years:
        y = str(year)
        year_rows = rdf[rdf['year'] == year]

        # Year aggregate ("All" period) excludes ongoing/uncompleted splits —
        # e.g. for 2026 mid-Stage 1, this folds in Kickoff only, not Stage 1.
        completed_ev_ids = [
            e['id'] for e in events_by_year.get(y, [])
            if EVENT_DATES.get(e['id']) and EVENT_DATES[e['id']][1] < today_str
        ]
        year_agg_rows = year_rows[year_rows['event_id'].isin(completed_ev_ids)]
        year_data = _build_team_stats(year_agg_rows, k, pdf)
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
_veto_cache_mtime = 0.0
_VETO_JSON_PATH = os.path.join(ROOT, 'data', 'veto_model.json')

def get_veto_model():
    """Reload when the file's mtime advances — RefreshLiveData rewrites this
    every refresh, so an in-memory cache would freeze the simulator's veto
    patterns to whatever was on disk at server start."""
    global _veto_cache, _veto_cache_mtime
    try:
        mtime = os.path.getmtime(_VETO_JSON_PATH)
    except OSError:
        mtime = 0.0
    if _veto_cache is None or mtime > _veto_cache_mtime:
        with open(_VETO_JSON_PATH) as f:
            _veto_cache = json.load(f)
        _veto_cache_mtime = mtime
    return _veto_cache

_intl_cache = None
_INTL_JSON_PATH = os.path.join(ROOT, 'data', 'intl_calibration.json')

def get_intl_calibration():
    global _intl_cache
    if _intl_cache is None:
        with open(_INTL_JSON_PATH) as f:
            _intl_cache = json.load(f)
    return _intl_cache

# Active 2026 VCT league teams (48 total, 12 per region — EMEA + Americas + Pacific + CN).
# Used as a display filter for the Modern Hub leaderboard. CN added 2026-05-13 —
# user wants them visible in BenPom rankings (still excluded from upcoming/recent
# matches by separate logic since those pages don't follow CN league play).
ACTIVE_2026_ORGS = {
    # EMEA
    "TL", "FNC", "NAVI", "VIT", "BBL", "GX", "KC", "TH", "FUT", "M8", "EF", "PCF",
    # Americas
    "SEN", "G2", "MIBR", "NRG", "100T", "C9", "EG", "KRÜ", "LEV", "FUR", "LOUD", "ENVY",
    # Pacific
    "PRX", "T1", "GEN", "DFM", "ZETA", "RRQ", "TS", "GE", "NS", "FS", "VL", "KRX",
    # CN
    "AG", "BLG", "DRG", "EDG", "FPX", "JDG", "NOVA", "TE", "TEC", "TYL", "WOL", "XLG",
}

# Static region lookup — includes historical teams for match data context
ORG_REGIONS = {
    "TL":   "EMEA",  "FNC":  "EMEA",  "NAVI": "EMEA",  "VIT":  "EMEA",
    "BBL":  "EMEA",  "GX":   "EMEA",  "KC":   "EMEA",  "TH":   "EMEA",
    "FUT":  "EMEA",  "GIA":  "EMEA",  "MKOI": "EMEA",  "WOL":  "EMEA",
    "M8":   "EMEA",
    "PCF":  "EMEA",  "ULF":  "EMEA",  "EF":   "EMEA",
    "SEN":  "Americas",  "G2":   "Americas",  "MIBR": "Americas",
    "NRG":  "Americas",  "100T": "Americas",  "C9":   "Americas",
    "EG":   "Americas",  "KRÜ":  "Americas",  "LEV":  "Americas",
    "FUR":  "Americas",  "LOUD": "Americas",  "2G":   "Americas",
    "APK":  "Americas",  "ENVY": "Americas",
    "PRX":  "Pacific",  "DRX":  "Pacific",  "T1":   "Pacific",
    "TLN":  "Pacific",  "GEN":  "Pacific",  "DFM":  "Pacific",
    "ZETA": "Pacific",  "RRQ":  "Pacific",  "TS":   "Pacific",
    "GE":   "Pacific",  "NS":   "Pacific",
    "FS":   "Pacific",  "VL":   "Pacific",  "KRX":  "Pacific",
    "BME":  "Pacific",
    # CN
    "EDG":  "CN",  "BLG":  "CN",  "TE":   "CN",  "DRG":  "CN",
    "ASE":  "CN",  "AG":   "CN",  "XLG":  "CN",  "WOL":  "CN",
    "FPX":  "CN",  "JDG":  "CN",  "NOVA": "CN",  "TEC":  "CN",
    "TYL":  "CN",  "TYLOO":"CN",
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
        'before_madrid':    ['2024_kickoff', '2024_china_kickoff'],
        'after_madrid':     ['2024_kickoff', '2024_china_kickoff', '2024_masters_madrid'],
        'before_shanghai':  ['2024_kickoff', '2024_china_kickoff', '2024_masters_madrid', '2024_stage1', '2024_china_stage1'],
        'after_shanghai':   ['2024_kickoff', '2024_china_kickoff', '2024_masters_madrid', '2024_stage1', '2024_china_stage1', '2024_masters_shanghai'],
        'before_champions': ['2024_kickoff', '2024_china_kickoff', '2024_masters_madrid', '2024_stage1', '2024_china_stage1', '2024_masters_shanghai', '2024_stage2', '2024_china_stage2'],
        'after_champions':  ['2024_kickoff', '2024_china_kickoff', '2024_masters_madrid', '2024_stage1', '2024_china_stage1', '2024_masters_shanghai', '2024_stage2', '2024_china_stage2', '2024_champions'],
    },
    '2025': {
        'before_bangkok':   ['2025_kickoff', '2025_china_kickoff'],
        'after_bangkok':    ['2025_kickoff', '2025_china_kickoff', '2025_masters_bangkok'],
        'before_toronto':   ['2025_kickoff', '2025_china_kickoff', '2025_masters_bangkok', '2025_stage1', '2025_china_stage1'],
        'after_toronto':    ['2025_kickoff', '2025_china_kickoff', '2025_masters_bangkok', '2025_stage1', '2025_china_stage1', '2025_masters_toronto'],
        'before_champions': ['2025_kickoff', '2025_china_kickoff', '2025_masters_bangkok', '2025_stage1', '2025_china_stage1', '2025_masters_toronto', '2025_stage2', '2025_china_stage2'],
        'after_champions':  ['2025_kickoff', '2025_china_kickoff', '2025_masters_bangkok', '2025_stage1', '2025_china_stage1', '2025_masters_toronto', '2025_stage2', '2025_china_stage2', '2025_champions'],
    },
    '2026': {
        'before_santiago': ['2026_kickoff', '2026_china_kickoff'],
        'after_santiago':  ['2026_kickoff', '2026_china_kickoff', '2026_masters_santiago'],
        'after_stage1':    ['2026_kickoff', '2026_china_kickoff', '2026_masters_santiago', '2026_stage1'],
    },
}

_map_name_index   = None
_headshots_cache  = None
_TEAM_INFO_VER    = 3   # bump this to bust _team_info_cache across all keys
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


def _i(v):
    try: return int(float(v))
    except Exception: return None
def _f(v):
    try: return float(v)
    except Exception: return None
def _mvp_of(rows):
    """Return the row with the highest R2.0; falls back to a random row."""
    import random
    if rows.empty: return None
    try:
        rows = rows.copy()
        rows['_r'] = pd.to_numeric(rows['R2.0'], errors='coerce')
        if rows['_r'].notna().any():
            return rows.loc[rows['_r'].idxmax()]
    except Exception:
        pass
    return rows.sample(1).iloc[0]

def _get_mvp_stat(org, year='2025', snap='after_champions', n_maps=3):
    """MVP function. Pool = the team's most recent 20 *maps played* (win or loss,
    any opponent). Sample `n_maps` of them at random, aggregate every player's
    stats across the sample (sums for K/D/A, averages for ACS and R 2.0), and
    return the highest-average-rated player's combined statline.

    `n_maps` is the number of maps that were played in the simulated series
    (e.g. 4 for a 3-1 Bo5), NOT the number won."""
    import random

    try: n_maps = max(1, int(n_maps))
    except Exception: n_maps = 3

    data_dir    = os.path.join(ROOT, 'data')
    snap_events = _SNAPSHOT_EVENTS.get(year, {}).get(snap, [])
    if not snap_events:
        return None

    # Walk events newest-first; collect up to 20 unique (MatchID, MapNum) pairs the team played.
    pool = []   # list of dataframe slices (one slice per map, holding all of the team's player rows)
    seen = set()
    for event_id in reversed(snap_events):
        maps_path = os.path.join(data_dir, 'maps', f'{event_id}.csv')
        if not os.path.exists(maps_path):
            continue
        try:
            mdf = pd.read_csv(maps_path)
        except Exception:
            continue
        org_rows = mdf[mdf['Org'] == org]
        ordered = []
        ev_seen = set()
        for _, row in org_rows.iterrows():
            try:
                key = (int(row['MatchID']), str(row['MapNum']))
            except Exception:
                continue
            if key in ev_seen: continue
            ev_seen.add(key); ordered.append(key)
        for mid, mn in reversed(ordered):  # newest first within the event
            if (mid, mn) in seen: continue
            seen.add((mid, mn))
            grp = mdf[(mdf['MatchID'] == mid) & (mdf['MapNum'].astype(str) == mn) & (mdf['Org'] == org)]
            if grp.empty: continue
            # Skip maps with no usable rating data (e.g. Shanghai, where R2.0 is all NaN).
            try:
                if 'R2.0' in grp.columns and pd.to_numeric(grp['R2.0'], errors='coerce').dropna().empty:
                    continue
            except Exception:
                pass
            pool.append(grp)
            if len(pool) >= 20:
                break
        if len(pool) >= 20:
            break

    if not pool:
        return None

    n = min(n_maps, len(pool))
    sample = random.sample(pool, n)

    import math
    def _num(v):
        try:
            f = float(v)
            if math.isnan(f) or math.isinf(f): return None
            return f
        except Exception:
            return None

    # Aggregate per-player stats across the sampled maps. Rating uses its own
    # counter so NaN-rated rows don't pull the average toward zero.
    agg = {}  # player -> {K, D, A, ACS_sum, ACS_n, R_sum, R_n, n}
    for grp in sample:
        for _, row in grp.iterrows():
            name = str(row.get('Player', ''))
            if not name: continue
            e = agg.setdefault(name, {'K':0,'D':0,'A':0,'ACS_sum':0.0,'ACS_n':0,'R_sum':0.0,'R_n':0,'n':0})
            k = _num(row.get('K'));   e['K'] += int(k) if k is not None else 0
            d = _num(row.get('D'));   e['D'] += int(d) if d is not None else 0
            a = _num(row.get('A'));   e['A'] += int(a) if a is not None else 0
            ac = _num(row.get('ACS'));
            if ac is not None: e['ACS_sum'] += ac; e['ACS_n'] += 1
            r = _num(row.get('R2.0'))
            if r is not None: e['R_sum']   += r;  e['R_n']   += 1
            e['n'] += 1

    if not agg:
        return None

    def avg_r(e): return (e['R_sum'] / e['R_n']) if e['R_n'] else 0.0
    mvp_name, mvp = max(agg.items(), key=lambda kv: avg_r(kv[1]))

    return {
        'player': mvp_name,
        'org':    org,
        'K':      mvp['K'],
        'D':      mvp['D'],
        'A':      mvp['A'],
        'ACS':    (mvp['ACS_sum'] / mvp['ACS_n']) if mvp['ACS_n'] else None,
        'R':      (mvp['R_sum']   / mvp['R_n'])   if mvp['R_n']   else None,
        'maps_used': n,
    }


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


PW_JS = "/* password gate removed */"

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
  .top-nav { padding:32px 32px 0; position:relative; z-index:1; display:flex; flex-direction:row; align-items:center; gap:16px; }
  .home-logo { height:80px; width:auto; display:block; opacity:.85; transition:opacity .2s; }
  .home-logo:hover { opacity:1; }
  .back-link { display:inline-flex; align-items:center; gap:6px; font-family:'DM Sans',sans-serif; font-size:.8rem; font-weight:600; color:#7c3aed; text-decoration:none; padding:6px 14px; border-radius:99px; border:1.5px solid rgba(124,58,237,.25); background:rgba(124,58,237,.06); transition:background .18s,border-color .18s,color .18s; white-space:nowrap; }
  .back-link:hover { background:rgba(124,58,237,.12); border-color:rgba(124,58,237,.5); color:#5b21b6; }
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
<link rel="preload" as="image" fetchpriority="high" href="/static/MastersShanghaiFinal.jpg">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  SHARED_CSS
  .hub-hero { position:relative; width:100%; padding:24px 32px 170px; min-height:520px; text-align:center; overflow:hidden; isolation:isolate; background-color:#0e0a14; }
  .hub-hero-img { position:absolute; inset:0; background-size:cover; background-position:center 5%; background-repeat:no-repeat; z-index:-2; transform:scale(1.02); transition:transform 18s linear, opacity 2s ease; opacity:0; }
  .hub-hero:hover .hub-hero-img { transform:scale(1.06); }
  /* darken at top for legibility, fade to Modern VCT Hub bg at the bottom edge */
  .hub-hero::after { content:''; position:absolute; inset:0; background:linear-gradient(180deg, rgba(14,10,20,0.45) 0%, rgba(14,10,20,0.55) 30%, rgba(14,10,20,0.20) 55%, rgba(232,213,245,0.40) 72%, rgba(232,213,245,0.85) 88%, #e8d5f5 100%); z-index:-1; pointer-events:none; }
  .hub-hero-content { position:relative; z-index:1; max-width:840px; margin:0 auto; }
  .hub-hero-eyebrow { font-family:'Syne',sans-serif; font-size:.62rem; font-weight:800; letter-spacing:.22em; text-transform:uppercase; color:#e8dff4; margin-bottom:14px; display:inline-flex; align-items:center; gap:12px; }
  .hub-hero-eyebrow::before, .hub-hero-eyebrow::after { content:''; display:inline-block; width:36px; height:2px; background:linear-gradient(90deg, transparent, #d4b8f4, transparent); }
  .hub-hero-title { font-family:'Syne',sans-serif; font-size:clamp(2.6rem,7.5vw,5.4rem); font-weight:800; letter-spacing:-2px; line-height:1; margin-bottom:18px; background:linear-gradient(135deg,#fff 0%,#e6d6f7 60%,#d4b8f4 100%); -webkit-background-clip:text; background-clip:text; color:transparent; word-break:keep-all; text-shadow:0 8px 36px #0e0a1455; }
  .hub-hero-sub { font-family:'DM Sans',sans-serif; font-size:1rem; color:#f5eaf5; max-width:none; margin:0 auto; line-height:1.5; text-shadow:0 2px 14px #0e0a1466; white-space:nowrap; }
  .hub-hero-cap { position:absolute; top:22px; right:24px; z-index:2; font-family:'Syne',sans-serif; font-size:.58rem; font-weight:800; letter-spacing:.18em; text-transform:uppercase; color:#ffffffcc; padding:6px 12px; border-radius:99px; background:#0e0a1466; backdrop-filter:blur(6px); }
  .hub-hero-nav { position:absolute; top:0; left:0; right:0; z-index:3; padding:24px 32px 0; }
  .hub-hero-nav .home-logo { filter:drop-shadow(0 4px 18px #0e0a1466); }
  /* Top overscroll shows the dark hero color, bottom overscroll shows cream.
     Use a fixed-attached gradient on html so the top half always paints dark and the
     bottom half cream — body's solid cream paints over it for normal viewing. */
  /* No overscroll on the hub — both top and bottom rubber-band disabled. */
  html { background:#e8d5f5; overscroll-behavior:none; }
  body { background:#e8d5f5 !important; overscroll-behavior:none; }
  #content-wrap { width:100%; }
  /* Hub stays calm — kill BOTH SHARED_CSS body backdrops so nothing tints
     the area below the hero and breaks the gradient blend. */
  body::before, body::after { display:none !important; }

  .hub-page { position:relative; z-index:1; padding:0 32px 64px; max-width:760px; margin:0 auto; text-align:center; }
  .hub-cards { display:flex; gap:24px; flex-wrap:wrap; justify-content:center; }
  .hub-card { background:white; border-radius:24px; padding:32px 26px 26px; width:300px; text-decoration:none; color:var(--ink); box-shadow:0 4px 24px #0000000a; transition:transform .25s,box-shadow .25s; text-align:center; position:relative; overflow:hidden; display:flex; flex-direction:column; }
  .hub-card::after { content:''; position:absolute; inset:0; background:linear-gradient(135deg, transparent 60%, #d4b8f422 100%); opacity:0; transition:opacity .25s; pointer-events:none; }
  .hub-card:hover { transform:translateY(-6px); box-shadow:0 16px 44px #00000018; }
  .hub-card:hover::after { opacity:1; }
  .hub-card-title { font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:800; margin-bottom:8px; letter-spacing:-.01em; }
  .hub-card-title--sm { font-size:.92rem; }
  .hub-card-desc { font-size:.82rem; color:var(--soft); line-height:1.55; }
  .hub-card-arrow { margin-top:auto; padding-top:20px; font-size:.8rem; color:#9a7ab4; font-family:'Syne',sans-serif; font-weight:800; letter-spacing:.04em; }

  /* Wide hero-card row underneath the two regular cards (Modern VCT Hub) */
  .hub-cards-wide { display:flex; justify-content:center; margin-top:24px; padding:0 4px; }
  .hub-card-wide { position:relative; display:flex; align-items:center; justify-content:center; width:100%; max-width:660px; height:240px; border-radius:24px; overflow:hidden; text-decoration:none; color:white; box-shadow:0 6px 28px #00000018; transition:transform .25s, box-shadow .25s; isolation:isolate; }
  .hub-card-wide:hover { transform:translateY(-6px); box-shadow:0 18px 48px #00000028; }
  .hub-card-wide-bg { position:absolute; inset:0; background-size:cover; background-position:center 40%; background-repeat:no-repeat; transform:scale(1.03); transition:transform 14s linear, opacity .8s ease; z-index:-2; opacity:0; }
  .hub-card-wide:hover .hub-card-wide-bg { transform:scale(1.10); }
  .hub-card-wide::after { content:''; position:absolute; inset:0; background:linear-gradient(180deg, #0e0a1422 0%, #0e0a1488 70%, #0e0a14bb 100%), radial-gradient(ellipse 60% 40% at 50% 60%, #00000044 0%, transparent 70%); z-index:-1; pointer-events:none; }
  .hub-card-wide-title { position:relative; font-family:'Syne',sans-serif; font-size:clamp(1.8rem, 4vw, 2.6rem); font-weight:800; letter-spacing:-.02em; line-height:1; text-shadow:0 4px 22px #0e0a14cc; background:linear-gradient(135deg,#fff 0%,#ffd9b3 100%); -webkit-background-clip:text; background-clip:text; color:transparent; padding:0 24px; text-align:center; }
  .hub-logo-strip { width:100vw; position:relative; left:50%; transform:translateX(-50%); display:flex; justify-content:space-evenly; align-items:center; flex-wrap:nowrap; padding:14px 24px; margin-bottom:20px; opacity:.85; }
  .hub-logo-strip img { height:28px; width:28px; object-fit:contain; flex-shrink:0; filter:grayscale(.4); transition:filter .2s, transform .2s; cursor:pointer; user-select:none; }
  .hub-logo-strip img:hover { filter:none; transform:scale(1.18); }
  .hub-logo-strip img.shaking { animation:logoShake .6s cubic-bezier(.36,.07,.19,.97); transform-origin:center; filter:none; }
  @keyframes logoShake {
    0%   { transform:translateX(0) rotate(0) scale(1); }
    15%  { transform:translateX(-3px) rotate(-12deg) scale(1.18); }
    30%  { transform:translateX(3px)  rotate(10deg)  scale(1.20); }
    45%  { transform:translateX(-2px) rotate(-8deg)  scale(1.15); }
    60%  { transform:translateX(2px)  rotate(6deg)   scale(1.12); }
    80%  { transform:translateX(-1px) rotate(-3deg)  scale(1.08); }
    100% { transform:translateX(0) rotate(0) scale(1); }
  }
  .hub-confetti { position:fixed; width:8px; height:8px; border-radius:2px; pointer-events:none; z-index:1000; will-change:transform, opacity; animation:confettiFly .9s cubic-bezier(.2,.8,.4,1) forwards; }
  @keyframes confettiFly {
    0%   { transform:translate(-50%,-50%) rotate(0deg); opacity:1; }
    100% { transform:translate(calc(-50% + var(--dx,0px)), calc(-50% + var(--dy,0px))) rotate(var(--rot,360deg)); opacity:0; }
  }
  @media(max-width:640px){
    .hub-hero { height:280px; margin:16px 16px 24px; border-radius:18px; }
    .hub-hero-content { padding:24px 22px 20px; }
  }
</style>
</head>
<body>
<div id="content-wrap">
  <section class="hub-hero">
    <div class="hub-hero-img"></div>
    <div class="top-nav hub-hero-nav">
      <a href="/"><img src="/logo.svg" alt="Home" class="home-logo"></a>
    </div>
    <div class="hub-hero-cap">2024 Masters Shanghai</div>
    <div class="hub-hero-content">
      <div class="hub-hero-eyebrow">Bobo&rsquo;s VCT Database</div>
      <h1 class="hub-hero-title">BenPom</h1>
      <p class="hub-hero-sub">Kenpom-Style ratings and analyses of VCT teams. Explore VCT history through BenPom.</p>
    </div>
  </section>
  <div class="hub-page">
    <div class="hub-logo-strip" id="hub-logo-strip"></div>
    <div class="hub-cards">
      <a class="hub-card" href="/mapelo/rankings/">
        <div class="hub-card-title hub-card-title--sm">Historical Rankings</div>
        <div class="hub-card-desc">Per-map Massey ratings with decay, James&ndash;Stein shrinkage, and pick/ban-adjusted overall scores.</div>
        <div class="hub-card-arrow">Explore &rarr;</div>
      </a>
      <a class="hub-card" href="/mapelo/matchup/">
        <div class="hub-card-title hub-card-title--sm">Historical Matchup Predictor</div>
        <div class="hub-card-desc">Test matchups between every VCT team throughout history using Monte Carlo simulations; this includes a statistical breakdown of picks/bans, map differences, and win/loss frequencies.</div>
        <div class="hub-card-arrow">Explore &rarr;</div>
      </a>
    </div>
    <div class="hub-cards-wide">
      <a class="hub-card-wide" href="/mapelo/modern/">
        <div class="hub-card-wide-bg"></div>
        <div class="hub-card-wide-title">Modern VCT Hub</div>
      </a>
    </div>
  </div>
  <script>
  (function(){
    var heroImg = document.querySelector('.hub-hero-img');
    if (heroImg) {
      var src1 = '/static/MastersShanghaiFinal.jpg';
      var img1 = new Image();
      img1.onload = function() {
        heroImg.style.backgroundImage = 'url(' + src1 + ')';
        requestAnimationFrame(function() { requestAnimationFrame(function() { heroImg.style.opacity = '1'; }); });
      };
      img1.src = src1;
    }
    var wideImg = document.querySelector('.hub-card-wide-bg');
    if (wideImg) {
      var src2 = '/static/Champs25Arena.jpg';
      var img2 = new Image();
      img2.onload = function() {
        wideImg.style.backgroundImage = 'url(' + src2 + ')';
        requestAnimationFrame(function() { wideImg.style.opacity = '1'; });
      };
      img2.src = src2;
    }
  })();
  (function(){
    var teams = ['FNC','LOUD','KRX','FNC','EG','PRX','EG','PRX','LOUD','SEN','GEN','PRX','GEN','TH','G2','EDG','TH','LEV','T1','G2','EDG','PRX','FNC','WOL','NRG','FNC','KRX','NS','PRX','NRG'];
    var strip = document.getElementById('hub-logo-strip');
    if(!strip) return;
    var html = '';
    teams.forEach(function(t){
      html += '<img src="/logos/'+t+'.png" alt="'+t+'" onerror="this.style.display=\\'none\\'">';
    });
    strip.innerHTML = html;

    var TEAM_COLORS = {
      'SEN':  ['#c8102e','#f5d6a8','#000000','#ffffff'],
      'LOUD': ['#2dff5d','#000000','#ffffff','#52ff8d'],
      'PRX':  ['#7b2fff','#c0392b','#e040fb','#ff6b6b'],
      'GEN':  ['#f6c61b','#000000','#ffffff','#fbe085'],
      'FNC':  ['#ff5c00','#000000','#ffffff','#ff8c40'],
      'EG':   ['#0089d0','#fbb521','#ffffff','#003d7a'],
      'NRG':  ['#ff3c3c','#000000','#ffffff','#ffa1a1'],
      'TH':   ['#fbb521','#000000','#ffffff','#fdd97e'],
      'T1':   ['#e2012d','#000000','#ffffff','#fb8a9c'],
      'DRX':  ['#0080a8','#1ed8e6','#ffffff','#000000'],
      'KC':   ['#0099ff','#e60014','#000000','#ffffff'],
      'C9':   ['#00a3e0','#ffffff','#0050a0','#73d2ff'],
      'LEV':  ['#5bc8e8','#a0b0bc','#d0e8f0','#7a9aaa'],
      'G2':   ['#000000','#ffffff','#cccccc','#ed1c24'],
      'TL':   ['#0033a0','#ffd400','#ffffff','#000000'],
      'BBL':  ['#00b4d8','#0e132d','#ffffff','#5cdfff'],
      '100T': ['#e80024','#000000','#ffffff','#ff6680'],
      'VIT':  ['#fff200','#000000','#ffffff','#fff066'],
      'GX':   ['#fbb121','#000000','#ffffff','#fdd97e'],
      'DRG':  ['#56e84d','#000000','#ffffff','#aaff9c'],
      'NS':   ['#e00000','#111111','#ff4444','#333333'],
      'KRX':  ['#005bac','#3a9edb','#ffffff','#7ec8f0'],
      'WOL':  ['#f5c400','#111111','#ffe066','#333333']
    };
    var DEFAULT_COLORS = ['#f4b8c1','#f9cba7','#b8e8d4','#b8d8f4','#d4b8f4','#f4edb8','#5a2a7a','#9a4ab4'];

    function spawnConfetti(cx, cy, palette){
      var colors = palette && palette.length ? palette : DEFAULT_COLORS;
      for(var i=0;i<22;i++){
        var p = document.createElement('div');
        p.className = 'hub-confetti';
        var angle = Math.random() * Math.PI * 2;
        var dist  = 70 + Math.random() * 80;
        var dx = Math.cos(angle) * dist;
        var dy = Math.sin(angle) * dist - 28;
        p.style.left = cx + 'px';
        p.style.top  = cy + 'px';
        p.style.background = colors[Math.floor(Math.random()*colors.length)];
        p.style.width  = (5 + Math.random()*6) + 'px';
        p.style.height = (5 + Math.random()*6) + 'px';
        p.style.setProperty('--dx', dx + 'px');
        p.style.setProperty('--dy', dy + 'px');
        p.style.setProperty('--rot', (Math.random()*720 - 360) + 'deg');
        p.style.animationDuration = (.7 + Math.random()*.4) + 's';
        document.body.appendChild(p);
        (function(el){ setTimeout(function(){ el.remove(); }, 1200); })(p);
      }
    }

    strip.querySelectorAll('img').forEach(function(img){
      img.addEventListener('click', function(e){
        e.preventDefault();
        img.classList.remove('shaking');
        // restart animation
        void img.offsetWidth;
        img.classList.add('shaking');
        var r = img.getBoundingClientRect();
        var team = img.getAttribute('alt') || '';
        spawnConfetti(r.left + r.width/2, r.top + r.height/2, TEAM_COLORS[team]);
        setTimeout(function(){ img.classList.remove('shaking'); }, 650);
      });
    });
  })();
  </script>
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
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
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
  .pipe-stage.active .pipe-graphic { max-height:380px; opacity:1; padding-top:6px; }
  .pg-note { font-size:.64rem; color:var(--soft); padding-top:4px; }
  /* Score bars (Stage 1) */
  .pg-scorebar { display:flex; flex-direction:column; gap:7px; padding:4px 0 2px; }
  .pg-score-row { display:flex; align-items:center; gap:10px; }
  .pg-score-label { font-family:'Syne',sans-serif; font-size:.72rem; font-weight:800; color:var(--soft); min-width:46px; text-align:right; flex-shrink:0; white-space:nowrap; }
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
  .pg-region-bubble.r-cn   { background:linear-gradient(135deg,#fce7f3,#fbcfe8); color:#9d174d; }
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
    <a class="back-link" href="/mapelo/"><svg width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M9 2L4 7l5 5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg> Back to BenPom</a>
  </div>
  <div class="page">
    <div class="page-title">Historical Rankings</div>
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
            <div class="pipe-desc">Every map is scored by round margin. A 13&ndash;2 win carries far more signal than a 13&ndash;11 win, but big margins are <strong>sqrt-compressed</strong> &mdash; blowouts give diminishing returns rather than scaling linearly.</div>
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
            <div class="pipe-desc">A linear algebra solve finds the rating vector that best explains all observed round differentials simultaneously. One solve per map. Mean-zero constraint. <strong>International maps carry higher weight</strong> (Masters &times;2, Champions &times;4), and each historical game is further reweighted by <strong>roster continuity</strong> &mdash; how much of each team&rsquo;s current starting five played in that match.</div>
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
            <div class="pipe-desc">Game weights follow <code>exp(&minus;&lambda;&thinsp;&times;&thinsp;weeks&thinsp;ago)</code>. Half-life = 5 weeks.</div>
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
            <div class="pipe-desc">Per-map ratings with smaller sample sizes are blended toward the team&rsquo;s overall rating.</div>
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
              <div class="pg-note"><span id="pg3-alpha-formula"></span></div>
            </div>
          </div>
        </div>

        <div class="pipe-connector" id="pc3"><div class="pipe-particle" id="pp3a"></div><div class="pipe-particle pipe-particle-b" id="pp3b"></div></div>

        <!-- Stage 5: Monte Carlo Veto -->
        <div class="pipe-stage" id="ps4" data-idx="4" onclick="focusPipe(4)">
          <div class="pipe-num pipe-n4">5</div>
          <div class="pipe-content">
            <div class="pipe-title">Monte Carlo Veto Simulation</div>
            <div class="pipe-desc">Each team runs through 10,000 simulated BO3 vetoes against league-average opponents using historical ban/pick patterns. Expected round-diff across the surviving maps becomes the headline rating. Thus, a great ban target is worth as much as a great map.</div>
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
            <div class="pipe-desc">All Masters, Champions, and VCT CN maps from 2023&ndash;2026 are included as time-decayed prior anchors. Older games carry less weight via the same exp(&minus;&lambda;t) decay, stabilizing inter-regional offsets. <strong>CN teams get intl-confidence shrinkage</strong>: teams with deep international r&eacute;sum&eacute;s (EDG, XLG) keep their raw signal; teams with no intl exposure are pulled toward a CN-baseline prior. A <strong>regional spillover dampener</strong> also prevents one team&rsquo;s intl run from inflating their entire league&rsquo;s non-attendees.</div>
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
                <div class="pg-intl-arrow">&#8596;</div>
                <div class="pg-region">
                  <div class="pg-region-bubble r-cn" id="pg5-cn">&minus;&delta;</div>
                  <div class="pg-region-name">China</div>
                </div>
              </div>
              <div class="pg-note">2023&ndash;2026 intl priors &rarr; calibrated inter-regional offsets + CN shrinkage for unproven teams</div>
            </div>
          </div>
        </div>

        <div class="pipe-connector" id="pc5"><div class="pipe-particle" id="pp5a"></div><div class="pipe-particle pipe-particle-b" id="pp5b"></div></div>

        <!-- Stage 7: Global Rating -->
        <div class="pipe-stage" id="ps6" data-idx="6" onclick="focusPipe(6)">
          <div class="pipe-num pipe-n6">&#10003;</div>
          <div class="pipe-content">
            <div class="pipe-title">Global Rating &amp; Win Probability</div>
            <div class="pipe-desc">The final rating combines domestic league results with inter-regional calibration from decayed international priors into a single globally-comparable number.</div>
            <div class="pipe-graphic">
              <div class="pg-formula" id="pg6-formula">
                <div class="pg-formula-part pg-formula-dom"  id="pg6-p0">domestic</div>
                <div class="pg-formula-part pg-formula-op"   id="pg6-p1">+</div>
                <div class="pg-formula-part pg-formula-intl" id="pg6-p2">intl priors</div>
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
  var PL=32, PB=22, PT=12, PR=12;
  var cW=W-PL-PR, cH=H-PB-PT;
  var MAX_W=1.15, MAX_WKS=20, STEPS=60;
  var lam = Math.LN2/5;
  function toX(wk){ return PL+(wk/MAX_WKS)*cW; }
  function toY(wt){ return (H-PB)-(wt/MAX_W)*cH; }
  function drawCurvePath(steps) {
    ctx.beginPath();
    for(var i=0;i<=steps;i++){
      var wk=(i/STEPS)*MAX_WKS, wt=Math.exp(-lam*wk);
      i===0?ctx.moveTo(toX(wk),toY(wt)):ctx.lineTo(toX(wk),toY(wt));
    }
  }
  function drawStatic() {
    // Axes
    ctx.strokeStyle='#ddd8e8'; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(PL,PT); ctx.lineTo(PL,H-PB); ctx.lineTo(W-PR,H-PB); ctx.stroke();
    // X-axis ticks
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
    // ── Single decay curve (all games, same weight) ──
    drawCurvePath(t);
    var gC=ctx.createLinearGradient(PL,0,W-PR,0);
    gC.addColorStop(0,'#a060d0'); gC.addColorStop(1,'#d080f8');
    ctx.strokeStyle=gC; ctx.lineWidth=2.5; ctx.stroke();
    drawCurvePath(t);
    ctx.lineTo(toX((t/STEPS)*MAX_WKS),H-PB); ctx.lineTo(PL,H-PB); ctx.closePath();
    var fC=ctx.createLinearGradient(0,PT,0,H-PB);
    fC.addColorStop(0,'rgba(160,96,208,.18)'); fC.addColorStop(1,'rgba(160,96,208,0)');
    ctx.fillStyle=fC; ctx.fill();
    // ── Half-life dashed line at 5 weeks ──────────
    if (t >= Math.round((5/MAX_WKS)*STEPS)) {
      var hlX=toX(5), hlY=toY(Math.exp(-lam*5));
      ctx.strokeStyle='#ccc0e0'; ctx.lineWidth=1; ctx.setLineDash([3,3]);
      ctx.beginPath(); ctx.moveTo(hlX,H-PB); ctx.lineTo(hlX,hlY); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle='#9060b8'; ctx.font='bold 8px Syne,sans-serif'; ctx.textAlign='center';
      ctx.fillText('t½=5w', hlX, hlY-4);
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
  ['pg5-emea','pg5-am','pg5-pac','pg5-cn'].forEach(function(id, i) {
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
  ['pg5-emea','pg5-am','pg5-pac','pg5-cn'].forEach(function(id){ var el=document.getElementById(id); if(el) el.classList.remove('show'); });
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
  // The "Live" snapshot belongs to the Modern Hub, not the historical rankings
  // page — hide it from the period dropdown here. Any in-progress event keeps
  // its data accessible via the prior "after_<event>" snapshot.
  var keys  = Object.keys(snaps).filter(function(k){
    return (snaps[k].label || '').toLowerCase() !== 'live';
  });
  var row   = document.getElementById('period-filter-row');
  var sel   = document.getElementById('period-select');

  if (keys.length <= 1) { row.style.display = 'none'; return; }
  row.style.display = 'flex';

  // If currentSnap is the now-hidden "Live" snap, fall back to the last
  // non-live snap so the rest of the page renders sensibly.
  if (keys.indexOf(currentSnap) < 0) currentSnap = keys[keys.length - 1];

  sel.innerHTML = '';
  keys.forEach(function(k) {
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

  // Render the shrinkage formula in LaTeX (KaTeX is loaded with `defer`).
  function renderAlphaFormula(){
    var el = document.getElementById('pg3-alpha-formula');
    if(!el) return;
    if(typeof katex === 'undefined'){ setTimeout(renderAlphaFormula, 50); return; }
    try {
      katex.render('\\\\alpha = \\\\dfrac{n}{n + k} \\\\;\\\\text{where}\\\\; k = 12', el, {throwOnError:false, displayMode:false});
    } catch(e){
      el.textContent = 'α = n / (n + k) where k = 12';
    }
  }
  renderAlphaFormula();
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
<title>Historical Matchup Predictor &mdash; BenPom</title>
<link rel="icon" type="image/svg+xml" href="/favicon.svg">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  SHARED_CSS
  .page { position:relative; z-index:1; padding:32px; max-width:980px; margin:0 auto; }
  .page-title { font-family:'Syne',sans-serif; font-size:clamp(1.4rem,3vw,2.2rem); font-weight:800; letter-spacing:-1px; margin-bottom:28px; text-align:center; min-height:1.2em; line-height:1; transition:opacity .2s; }
  .type-cursor{opacity:1;animation:blink .55s step-end infinite}
  .dot-seed{font-size:1.6rem}
  @keyframes blink{0%,100%{opacity:1}50%{opacity:0}}
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
  .controls-row { display:flex; align-items:center; justify-content:center; gap:14px; margin-bottom:28px; margin-top:18px; }
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
  .fate-legend { display:flex; gap:12px; flex-wrap:wrap; padding:22px 32px 14px; }
  .fate-legend-item { display:flex; align-items:center; gap:5px; font-size:.67rem; color:var(--soft); }
  .fate-dot { width:10px; height:10px; border-radius:2px; flex-shrink:0; }
  /* Map table */
  .map-tbl { width:100%; border-collapse:collapse; font-size:.86rem; }
  .map-tbl thead th { font-family:'Syne',sans-serif; font-size:.6rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase; color:var(--soft); padding:10px 14px; border-top:1px solid #f0ecf4; border-bottom:1px solid #f0ecf4; text-align:center; background:#faf8fc; white-space:nowrap; }
  .map-tbl thead th:first-child { text-align:left; padding-left:24px; }
  .map-tbl thead th:nth-child(2) { text-align:left; }
  .map-tbl tbody tr { border-bottom:1px solid #f8f4fc; transition:background .1s; }
  .map-tbl tbody tr:last-child { border-bottom:none; }
  .map-tbl tbody tr:hover { background:#fdf6f0; }
  .map-tbl tbody td { padding:18px 14px; text-align:center; vertical-align:middle; }
  .map-tbl tbody td:first-child { text-align:left; font-family:'Syne',sans-serif; font-weight:800; padding-left:24px; }
  .bd-map-mini { display:flex; align-items:center; gap:12px; font-size:1.05rem; }
  .bd-map-mini img { width:38px; height:38px; object-fit:cover; border-radius:8px; }
  /* Prominent probability cell */
  .wp-prom { display:flex; align-items:center; gap:10px; justify-content:flex-start; }
  .wp-prom-num { font-family:'Syne',sans-serif; font-size:1.45rem; font-weight:800; min-width:62px; text-align:left; line-height:1; }
  .wp-prom-num.fav { color:#1a6a4a; }
  .wp-prom-num.dog { color:#7a1a1a; }
  .wp-prom-num.neu { color:var(--soft); }
  .wp-prom-bg { width:90px; height:9px; border-radius:99px; background:#f0ecf4; overflow:hidden; }
  .wp-prom-fill { height:100%; background:linear-gradient(90deg,#5a2a7a,#9a4ab4); border-radius:99px; transition:width .4s; }
  .wp-prom-empty { font-family:'Syne',sans-serif; font-size:1rem; color:var(--soft); }
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
  .step-lbl-banA, .step-lbl-banB   { background:#fde8ec; color:#b03050; }
  .step-lbl-pickA, .step-lbl-pickB { background:#e3f6ea; color:#206040; }
  .step-lbl-dec                    { background:#f0ecf4; color:#7a6e7e; }
  .step-map { font-family:'Syne',sans-serif; font-weight:800; font-size:.72rem; color:#2a1f2d; white-space:nowrap; }
  .step-arrow { font-size:.7rem; color:#ccc; align-self:center; margin-top:10px; }
  .no-veto-data { font-size:.78rem; color:var(--soft); font-style:italic; }

  /* === MODE TOGGLE === */
  .mode-toggle-row { display:flex; justify-content:center; margin-bottom:22px; }
  .mode-toggle { display:inline-flex; background:white; border-radius:99px; padding:4px; box-shadow:0 4px 18px #0000000a; gap:2px; }
  .mode-btn { background:transparent; border:none; padding:8px 18px; border-radius:99px; font-family:'Syne',sans-serif; font-size:.7rem; font-weight:800; letter-spacing:.06em; text-transform:uppercase; color:var(--soft); cursor:pointer; transition:all .2s; }
  .mode-btn.active { background:linear-gradient(135deg,#5a2a7a,#9a4ab4); color:white; box-shadow:0 4px 12px #5a2a7a33; }
  .mode-btn:not(.active):hover { color:var(--ink); }

  /* === SIDE PANEL (replaces team-panel) === */
  .side-grid { display:grid; grid-template-columns:1fr 80px 1fr; gap:0; align-items:stretch; margin-bottom:36px; }
  .side-panel { background:white; border-radius:24px; padding:18px 16px 22px; box-shadow:0 4px 24px #0000000a; display:flex; flex-direction:column; align-items:stretch; }
  .side-label { font-family:'Syne',sans-serif; font-size:.58rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase; color:var(--soft); margin-bottom:10px; text-align:center; }

  /* === YEAR SCRUBBER === */
  .yr-scrubber { position:relative; padding:6px 12px 22px; user-select:none; }
  .yr-track { position:relative; height:4px; border-radius:99px; background:linear-gradient(90deg,#f4b8c1,#d4b8f4,#b8d8f4,#b8e8d4); margin:14px 0 4px; }
  .yr-tick { position:absolute; top:50%; width:8px; height:8px; border-radius:50%; background:white; border:2px solid #d4b8f4; transform:translate(-50%,-50%); transition:transform .15s; cursor:pointer; }
  .yr-tick.active { background:var(--ink); border-color:var(--ink); transform:translate(-50%,-50%) scale(1.4); }
  .yr-tick:hover { transform:translate(-50%,-50%) scale(1.3); }
  .yr-knob { position:absolute; top:50%; width:18px; height:18px; border-radius:50%; background:linear-gradient(135deg,#5a2a7a,#9a4ab4); transform:translate(-50%,-50%); box-shadow:0 4px 12px #5a2a7a55, 0 0 0 4px white; transition:left .35s cubic-bezier(.5,1.6,.4,1); pointer-events:none; }
  .yr-labels { display:flex; justify-content:space-between; font-family:'Syne',sans-serif; font-size:.65rem; font-weight:800; color:var(--soft); margin-top:8px; padding:0 4px; }
  .yr-labels span { cursor:pointer; padding:2px 4px; transition:color .15s; }
  .yr-labels span.active { color:var(--ink); }
  .yr-labels span:hover { color:var(--ink); }

  /* === SNAPSHOT SEGMENTED === */
  .snap-seg { display:flex; gap:3px; flex-wrap:wrap; justify-content:center; margin:6px 0 14px; }
  .snap-seg-btn { background:#faf6fc; border:1.5px solid transparent; padding:4px 10px; border-radius:99px; font-family:'DM Sans',sans-serif; font-size:.68rem; color:var(--soft); cursor:pointer; transition:all .15s; }
  .snap-seg-btn:hover { color:var(--ink); border-color:#e0d0ec; }
  .snap-seg-btn.active { background:var(--ink); color:white; border-color:var(--ink); }

  /* === TEAM SEARCH === */
  .cf-search-wrap { position:relative; margin:6px 8px 12px; }
  .cf-search { width:100%; padding:7px 30px 7px 30px; border-radius:99px; border:1.5px solid #f0ecf4; background:#faf8fc; font-family:'DM Sans',sans-serif; font-size:.78rem; color:var(--ink); outline:none; transition:border-color .15s, background .15s; }
  .cf-search:focus { border-color:#d4b8f4; background:white; }
  .cf-search::placeholder { color:#bcb2c4; }
  .cf-search-icon { position:absolute; left:11px; top:50%; transform:translateY(-50%); color:#bcb2c4; font-size:.75rem; pointer-events:none; }
  .cf-search-clear { position:absolute; right:8px; top:50%; transform:translateY(-50%); width:18px; height:18px; border-radius:50%; background:#e8dff4; color:var(--soft); font-size:.7rem; border:none; cursor:pointer; display:none; align-items:center; justify-content:center; }
  .cf-search-clear.visible { display:flex; }
  .cf-search-clear:hover { background:var(--ink); color:white; }

  /* === CYLINDER (drum) === */
  .cf-stage { position:relative; height:170px; perspective:900px; perspective-origin:center 50%; overflow:hidden; cursor:grab; touch-action:pan-y; }
  .cf-stage:active { cursor:grabbing; }
  .cf-stage::before, .cf-stage::after { content:''; position:absolute; top:0; bottom:0; width:50px; pointer-events:none; z-index:5; }
  .cf-stage::before { left:0; background:linear-gradient(90deg,white 5%,transparent); }
  .cf-stage::after { right:0; background:linear-gradient(-90deg,white 5%,transparent); }
  .cf-track { position:absolute; left:50%; top:50%; width:0; height:0; transform-style:preserve-3d; transition:transform .45s cubic-bezier(.45,1.5,.4,1); }
  .cf-item { position:absolute; left:0; top:0; width:96px; height:96px; margin:-48px 0 0 -48px; display:flex; flex-direction:column; align-items:center; justify-content:center; transform-style:preserve-3d; backface-visibility:hidden; transition:opacity .3s; cursor:pointer; }
  .cf-item .cf-card { width:96px; height:96px; border-radius:18px; background:white; display:flex; align-items:center; justify-content:center; padding:8px; box-shadow:0 4px 14px #00000014; transition:box-shadow .2s, transform .2s; }
  .cf-item.center .cf-card { box-shadow:0 10px 28px #5a2a7a33, 0 0 0 2px #d4b8f4; transform:scale(1.04); }
  .cf-item img { max-width:74px; max-height:74px; object-fit:contain; }
  .cf-item .cf-fallback { font-family:'Syne',sans-serif; font-weight:800; font-size:.85rem; color:var(--ink); text-align:center; }
  .cf-rtg { font-family:'Syne',sans-serif; font-weight:800; font-size:.72rem; font-variant-numeric:tabular-nums; margin-top:6px; padding:2px 8px; border-radius:99px; background:rgba(0,0,0,.04); letter-spacing:.02em; }
  .cf-rtg-pos { color:#16a34a; }
  .cf-rtg-neg { color:#dc2626; }
  .cf-item.center .cf-rtg { background:rgba(124,58,237,.12); font-size:.78rem; }
  .cf-name { text-align:center; font-family:'Syne',sans-serif; font-weight:800; font-size:1.1rem; letter-spacing:-.02em; color:var(--ink); margin-top:6px; min-height:1.4em; }
  .cf-region { text-align:center; font-family:'Syne',sans-serif; font-size:.6rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase; color:var(--soft); margin-top:2px; min-height:1em; }
  .cf-arrows { display:flex; justify-content:space-between; padding:0 6px; pointer-events:none; position:absolute; left:0; right:0; top:50%; transform:translateY(-50%); z-index:6; }
  .cf-arrow { pointer-events:all; background:white; border:none; width:32px; height:32px; border-radius:50%; box-shadow:0 2px 10px #00000018; cursor:pointer; font-size:1rem; color:var(--ink); display:flex; align-items:center; justify-content:center; transition:transform .15s, background .15s; }
  .cf-arrow:hover { background:var(--ink); color:white; transform:scale(1.1); }
  .cf-arrow:disabled { opacity:.3; cursor:not-allowed; }
  .cf-arrow:disabled:hover { background:white; color:var(--ink); transform:none; }

  /* === VS DIVIDER === */
  .vs-divider { display:flex; align-items:center; justify-content:center; }
  .vs-badge { width:48px; height:48px; border-radius:50%; background:linear-gradient(135deg,#5a2a7a,#9a4ab4); color:white; font-family:'Syne',sans-serif; font-weight:800; font-size:.9rem; display:flex; align-items:center; justify-content:center; box-shadow:0 6px 18px #5a2a7a44; letter-spacing:.05em; }

  /* === REVEAL OVERLAY === */
  .reveal-stage { background:white; border-radius:24px; box-shadow:0 4px 24px #0000000a; padding:30px; margin-bottom:20px; position:relative; min-height:240px; overflow:hidden; }
  .reveal-skip { position:absolute; top:14px; right:16px; background:transparent; border:1.5px solid #e8dff4; color:var(--soft); border-radius:99px; padding:5px 14px; font-family:'Syne',sans-serif; font-size:.6rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase; cursor:pointer; transition:all .15s; z-index:10; }
  .reveal-skip:hover { color:var(--ink); border-color:#5a2a7a; }
  .rv-step { font-family:'Syne',sans-serif; font-size:.65rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; color:var(--soft); margin-bottom:14px; text-align:center; }

  /* Sim intro */
  .rv-intro { display:flex; align-items:center; justify-content:center; gap:40px; padding:30px 0; }
  .rv-intro-team { display:flex; flex-direction:column; align-items:center; gap:8px; opacity:0; transform:translateX(-60px); animation:rvSlideIn .5s ease forwards; }
  .rv-intro-team.b { transform:translateX(60px); animation-name:rvSlideInR; }
  @keyframes rvSlideIn { to { opacity:1; transform:translateX(0); } }
  @keyframes rvSlideInR { to { opacity:1; transform:translateX(0); } }
  .rv-intro-team img { width:72px; height:72px; object-fit:contain; }
  .rv-intro-vs { font-family:'Syne',sans-serif; font-weight:800; font-size:1.6rem; background:linear-gradient(135deg,#5a2a7a,#9a4ab4); -webkit-background-clip:text; background-clip:text; color:transparent; opacity:0; animation:rvFadeIn .4s .3s forwards; }
  @keyframes rvFadeIn { to { opacity:1; } }
  .rv-shimmer { position:absolute; inset:0; background:linear-gradient(110deg,transparent 30%,#d4b8f455 50%,transparent 70%); transform:translateX(-100%); animation:rvShimmer 1.6s ease-in-out infinite; pointer-events:none; }
  @keyframes rvShimmer { to { transform:translateX(100%); } }

  /* Veto reveal grid */
  .rv-veto-grid { display:flex; gap:8px; flex-wrap:wrap; justify-content:center; padding:14px 0; }
  .rv-veto-slot { width:104px; height:108px; border-radius:14px; background:#faf6fc; display:flex; flex-direction:column; align-items:center; justify-content:center; gap:5px; padding:8px 6px; opacity:.55; transition:opacity .3s, transform .35s, background .35s, box-shadow .35s; transform:scale(.92); position:relative; overflow:hidden; }
  .rv-veto-slot.rv-vs-pending .rv-vs-q { font-family:'Syne',sans-serif; font-weight:800; font-size:2.4rem; color:#bcb2c4; line-height:1; }
  .rv-veto-slot.revealed { opacity:1; transform:scale(1); animation:rvPop .4s ease; }
  @keyframes rvPop { 0%{transform:scale(.7);} 60%{transform:scale(1.08);} 100%{transform:scale(1);} }
  .rv-veto-slot.banned::before { content:''; position:absolute; inset:0; background:repeating-linear-gradient(45deg,transparent 0 6px,#f4b8c133 6px 12px); pointer-events:none; }
  .rv-veto-slot.banned img { filter:grayscale(1); opacity:.55; }
  .rv-veto-slot img { width:54px; height:54px; object-fit:cover; border-radius:8px; }
  .rv-veto-slot .rv-vs-map { font-family:'Syne',sans-serif; font-weight:800; font-size:.75rem; color:var(--ink); text-align:center; }
  .rv-veto-slot .rv-vs-act { font-family:'Syne',sans-serif; font-weight:800; font-size:.55rem; letter-spacing:.08em; text-transform:uppercase; padding:2px 8px; border-radius:99px; white-space:nowrap; max-width:100%; text-align:center; }
  .rv-act-banA, .rv-act-banB   { background:#fde8ec; color:#b03050; }
  .rv-act-pickA, .rv-act-pickB { background:#e3f6ea; color:#206040; }
  .rv-act-dec                  { background:#f0ecf4; color:#7a6e7e; }

  /* Map result reveal cards */
  .rv-maps { display:flex; flex-direction:column; gap:14px; padding:8px 0; }
  .rv-map-card { background:#0e0a14; color:white; border-radius:18px; overflow:hidden; position:relative; min-height:140px; display:flex; align-items:stretch; opacity:0; transform:translateY(24px); transition:opacity .55s, transform .55s; }
  .rv-map-card.shown { opacity:1; transform:translateY(0); }
  .rv-map-card .rv-map-bg { position:absolute; inset:0; background-size:cover; background-position:center; opacity:.55; }
  .rv-map-card .rv-map-bg::after { content:''; position:absolute; inset:0; background:linear-gradient(180deg,#0e0a1499 0%,#0e0a14ee 100%); }
  .rv-map-inner { position:relative; z-index:1; padding:18px 22px; display:flex; align-items:center; gap:18px; width:100%; }
  .rv-map-name { font-family:'Syne',sans-serif; font-weight:800; font-size:1.3rem; letter-spacing:.04em; text-transform:uppercase; flex:0 0 auto; }
  .rv-map-num { font-family:'Syne',sans-serif; font-weight:800; font-size:.6rem; letter-spacing:.18em; text-transform:uppercase; color:#a08fbf; margin-bottom:3px; }
  .rv-map-pickedby { font-family:'Syne',sans-serif; font-size:.55rem; letter-spacing:.1em; text-transform:uppercase; color:#9a7ab4; margin-top:2px; }
  .rv-map-h2h { display:flex; flex:1; align-items:center; justify-content:flex-end; gap:18px; }
  .rv-map-team { display:flex; flex-direction:column; align-items:center; gap:4px; min-width:84px; }
  .rv-map-team img { width:38px; height:38px; object-fit:contain; filter:drop-shadow(0 2px 6px #00000060); }
  .rv-map-team-name { font-family:'Syne',sans-serif; font-weight:800; font-size:.6rem; letter-spacing:.08em; color:#a08fbf; }
  .rv-map-score { font-family:'Syne',sans-serif; font-weight:800; font-size:2.4rem; line-height:1; color:white; transition:color .3s, transform .25s, text-shadow .3s; }
  .rv-map-score.bumped { transform:scale(1.18); }
  .rv-map-score.win { color:#9affd0; text-shadow:0 0 22px #9affd088; }
  .rv-map-score.lose { color:#7d6a8e; }
  .rv-map-team-pct { font-family:'DM Sans',sans-serif; font-weight:600; font-size:.78rem; color:#a08fbf; min-height:1.1em; opacity:0; transition:opacity .35s; }
  .rv-map-team-pct.shown { opacity:1; }
  .rv-map-team-pct.win { color:#9affd0; }
  .rv-map-vs-mini { font-family:'Syne',sans-serif; font-weight:800; font-size:.7rem; color:#796a89; align-self:center; }
  .rv-map-result-badge { font-family:'Syne',sans-serif; font-weight:800; font-size:.7rem; letter-spacing:.08em; text-transform:uppercase; padding:5px 10px; border-radius:99px; background:#9a4ab4; color:white; align-self:center; opacity:0; transform:scale(.6); transition:opacity .35s, transform .35s; }
  .rv-map-result-badge.shown { opacity:1; transform:scale(1); }

  /* Series clinch */
  .rv-clinch { text-align:center; padding:20px 20px 4px; font-family:'Syne',sans-serif; font-weight:800; font-size:1.5rem; background:linear-gradient(135deg,#5a2a7a,#9a4ab4); -webkit-background-clip:text; background-clip:text; color:transparent; opacity:0; transform:scale(.85); transition:opacity .4s, transform .4s; }
  .rv-clinch.shown { opacity:1; transform:scale(1); }
  .rv-statline { text-align:center; padding:6px 24px 22px; font-family:'DM Sans',sans-serif; font-size:.95rem; color:var(--soft); opacity:0; transform:translateY(8px); transition:opacity .5s, transform .5s; line-height:1.45; max-width:640px; margin:0 auto; }
  .rv-statline.shown { opacity:1; transform:translateY(0); }
  .rv-statline strong { font-family:'Syne',sans-serif; font-weight:800; color:var(--ink); }
  .rv-statline em { font-style:italic; color:#5a2a7a; font-weight:600; }
  .rv-mvp-tag { display:inline-block; font-family:'Syne',sans-serif; font-weight:800; font-size:.62rem; letter-spacing:.14em; color:white; background:linear-gradient(135deg,#9a4ab4,#5a2a7a); padding:3px 10px; border-radius:99px; vertical-align:1px; margin-right:6px; box-shadow:0 2px 8px #5a2a7a44; }

  /* Final breakdown card grid */
  .breakdown-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:14px; margin-top:18px; }
  .bd-card { background:white; border-radius:18px; padding:16px 18px; box-shadow:0 4px 18px #00000008; }
  .bd-card-title { font-family:'Syne',sans-serif; font-size:.6rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; color:var(--soft); margin-bottom:10px; }
  .bd-mini-row { display:flex; align-items:center; justify-content:space-between; padding:5px 0; font-size:.78rem; border-bottom:1px solid #f6f1fa; }
  .bd-mini-row:last-child { border-bottom:none; }
  .bd-map-mini { display:flex; align-items:center; gap:8px; }
  .bd-map-mini img { width:22px; height:22px; object-fit:cover; border-radius:5px; }

  .replay-btn { display:inline-flex; align-items:center; gap:6px; background:transparent; border:1.5px solid #e0d0ec; color:#5a2a7a; padding:6px 14px; border-radius:99px; font-family:'Syne',sans-serif; font-size:.65rem; font-weight:800; letter-spacing:.07em; text-transform:uppercase; cursor:pointer; transition:all .15s; margin-top:10px; }
  .replay-btn:hover { background:#5a2a7a; color:white; border-color:#5a2a7a; }

  /* Smooth result-section dismissal */
  #result-section { transition:opacity .32s ease, transform .32s ease, max-height .42s ease; max-height:9999px; overflow:hidden; }
  #result-section.rs-fade-out { opacity:0; transform:translateY(-8px); max-height:0 !important; }

  @media(max-width:740px){
    .side-grid { grid-template-columns:1fr; }
    .vs-divider { padding:6px 0; }
    .result-mid { flex:0 0 80px; }
    .result-pct { font-size:1.8rem; }
    .rv-map-h2h { flex-wrap:wrap; gap:8px; }
  }
</style>
</head>
<body>
<div id="content-wrap">
  <div class="top-nav">
    <a href="/"><img src="/logo.svg" alt="Home" class="home-logo"></a>
    <a class="back-link" href="/mapelo/"><svg width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M9 2L4 7l5 5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg> Back to BenPom</a>
  </div>
  <div class="page">
    <h1 class="page-title" id="matchupTitle" style="opacity:0">&middot;</h1>

    <div class="mode-toggle-row">
      <div class="mode-toggle">
        <button class="mode-btn active" data-mode="dramatic">Full Reveal</button>
        <button class="mode-btn" data-mode="straight">Straightforward</button>
      </div>
    </div>

    <div class="side-grid">
      <div class="side-panel">
        <div class="side-label">Team A</div>
        <div class="yr-scrubber" data-side="a">
          <div class="yr-track">
            <div class="yr-tick active" data-year="2023" style="left:0%"></div>
            <div class="yr-tick" data-year="2024" style="left:33.33%"></div>
            <div class="yr-tick" data-year="2025" style="left:66.66%"></div>
            <div class="yr-tick" data-year="2026" style="left:100%"></div>
            <div class="yr-knob" style="left:0%"></div>
          </div>
          <div class="yr-labels">
            <span class="active" data-year="2023">2023</span>
            <span data-year="2024">2024</span>
            <span data-year="2025">2025</span>
            <span data-year="2026">2026</span>
          </div>
        </div>
        <div class="snap-seg" id="snap-a"></div>
        <div class="cf-search-wrap">
          <span class="cf-search-icon">&#9906;</span>
          <input class="cf-search" id="cf-search-a" data-side="a" placeholder="Search teams…" autocomplete="off">
          <button class="cf-search-clear" data-side="a" type="button">&times;</button>
        </div>
        <div class="cf-stage" id="cf-a">
          <div class="cf-track"></div>
          <div class="cf-arrows"><button class="cf-arrow" data-side="a" data-dir="-1">&lsaquo;</button><button class="cf-arrow" data-side="a" data-dir="1">&rsaquo;</button></div>
        </div>
        <div class="cf-name" id="cf-name-a"></div>
        <div class="cf-region" id="cf-region-a"></div>
      </div>
      <div class="vs-divider">
        <div class="vs-badge">VS</div>
      </div>
      <div class="side-panel">
        <div class="side-label">Team B</div>
        <div class="yr-scrubber" data-side="b">
          <div class="yr-track">
            <div class="yr-tick" data-year="2023" style="left:0%"></div>
            <div class="yr-tick" data-year="2024" style="left:33.33%"></div>
            <div class="yr-tick active" data-year="2025" style="left:66.66%"></div>
            <div class="yr-tick" data-year="2026" style="left:100%"></div>
            <div class="yr-knob" style="left:66.66%"></div>
          </div>
          <div class="yr-labels">
            <span data-year="2023">2023</span>
            <span data-year="2024">2024</span>
            <span class="active" data-year="2025">2025</span>
            <span data-year="2026">2026</span>
          </div>
        </div>
        <div class="snap-seg" id="snap-b"></div>
        <div class="cf-search-wrap">
          <span class="cf-search-icon">&#9906;</span>
          <input class="cf-search" id="cf-search-b" data-side="b" placeholder="Search teams…" autocomplete="off">
          <button class="cf-search-clear" data-side="b" type="button">&times;</button>
        </div>
        <div class="cf-stage" id="cf-b">
          <div class="cf-track"></div>
          <div class="cf-arrows"><button class="cf-arrow" data-side="b" data-dir="-1">&lsaquo;</button><button class="cf-arrow" data-side="b" data-dir="1">&rsaquo;</button></div>
        </div>
        <div class="cf-name" id="cf-name-b"></div>
        <div class="cf-region" id="cf-region-b"></div>
      </div>
    </div>

    <div class="controls-row">
      <div class="fmt-row">
        <button class="fmt-btn" data-fmt="bo1">Bo1</button>
        <button class="fmt-btn active" data-fmt="bo3">Bo3</button>
        <button class="fmt-btn" data-fmt="bo5">Bo5</button>
      </div>
      <button class="sim-btn" onclick="runMatchup()">Run Simulation</button>
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
var LOCK_CURRENT = LOCK_CURRENT_FLAG;
function _latestSnapFor(y){
  var snaps = ((DATA.ratings||{})[y]||{}).snapshots || {};
  var keys = Object.keys(snaps);
  return keys[keys.length-1] || 'after_champions';
}
function _latestYear(){
  var years = Object.keys(DATA.ratings || {}).sort();
  return years[years.length-1] || '2026';
}
var yearA = LOCK_CURRENT ? _latestYear() : '2023';
var snapA = LOCK_CURRENT ? _latestSnapFor(yearA) : 'after_champions';
var yearB = LOCK_CURRENT ? _latestYear() : '2025';
var snapB = LOCK_CURRENT ? _latestSnapFor(yearB) : 'after_champions';
var fmt = 'bo3';

// When embedded as the Modern Hub "Simulator" tab, hide year/snap pickers,
// the home-button nav, and the snapshot label under each team — the modern
// hub is the live/dynamic view, so no fixed-snapshot text should appear.
if (LOCK_CURRENT) {
  document.addEventListener('DOMContentLoaded', function(){
    var css = document.createElement('style');
    css.textContent =
      '.yr-scrubber, .snap-seg { display: none !important; }' +
      '.top-nav { display: none !important; }' +
      '.page > .subtitle { display: none !important; }' +
      '.result-ctx { display: none !important; }' +
      'body { background: transparent !important; }' +
      'body::before, body::after { display: none !important; }';
    document.head.appendChild(css);
  });
}

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
var mode = 'dramatic';
var YEARS = ['2023','2024','2025','2026'];
var CF = {a:{teams:[],idx:0,startX:0,dragging:false,startIdx:0}, b:{teams:[],idx:0,startX:0,dragging:false,startIdx:0}};

// ── Tick SFX (Web Audio) ─────────────────────────────────────────────────────
var audioCtx=null, audioOn=true;
function ensureAudio(){ if(audioCtx) return; try{ audioCtx=new (window.AudioContext||window.webkitAudioContext)(); }catch(e){} }
function tick(opts){
  if(!audioOn) return;
  ensureAudio(); if(!audioCtx) return;
  opts = opts||{};
  var t=audioCtx.currentTime, freq=opts.freq||1100, dur=opts.dur||0.045, vol=opts.vol||0.06, type=opts.type||'square';
  var o=audioCtx.createOscillator(), g=audioCtx.createGain();
  o.type=type; o.frequency.value=freq;
  g.gain.setValueAtTime(0,t);
  g.gain.linearRampToValueAtTime(vol,t+0.004);
  g.gain.exponentialRampToValueAtTime(0.0001,t+dur);
  o.connect(g); g.connect(audioCtx.destination);
  o.start(t); o.stop(t+dur+0.02);
}
function tickSeq(seq){ seq.forEach(function(s,i){ setTimeout(function(){ tick(s); }, s.delay||(i*60)); }); }

// ── Snapshot segmented control ───────────────────────────────────────────────
function populateSnapSeg(side){
  var year = side==='a'?yearA:yearB, cur = side==='a'?snapA:snapB;
  var snaps = getSnapsFor(year);
  // Hide the "Live" snapshot from the standalone /mapelo/matchup/ page —
  // that page is for historical/retrospective matchups. The Modern Hub
  // simulator (LOCK_CURRENT) is the OPPOSITE — it specifically wants the
  // Live snap, so we keep all keys there. Without this guard, the simulator
  // would silently fall back to the oldest snap in the year because its
  // pre-set snapA="after_stage1" (the Live snap) wouldn't be in the
  // filtered list.
  var keys = Object.keys(snaps).filter(function(k){
    if (LOCK_CURRENT) return true;
    return (snaps[k].label || '').toLowerCase() !== 'live';
  });
  var host = document.getElementById('snap-'+side);
  if(!keys.length){ host.innerHTML=''; return; }
  if(keys.indexOf(cur)<0){ cur=keys[0]; if(side==='a') snapA=cur; else snapB=cur; }
  host.innerHTML = keys.map(function(k){
    var lbl = (snaps[k]||{}).label || k;
    return '<button class="snap-seg-btn'+(k===cur?' active':'')+'" data-side="'+side+'" data-snap="'+k+'">'+lbl+'</button>';
  }).join('');
  host.querySelectorAll('.snap-seg-btn').forEach(function(b){
    b.addEventListener('click', function(){
      var s=b.dataset.side, sn=b.dataset.snap;
      if(s==='a'){ if(snapA===sn) return; snapA=sn; } else { if(snapB===sn) return; snapB=sn; }
      populateSnapSeg(s); populateTeams(s);
      clearResult();
    });
  });
}

// ── Coverflow ────────────────────────────────────────────────────────────────
function populateTeams(side){
  var year=side==='a'?yearA:yearB, snap=side==='a'?snapA:snapB;
  var teams=Object.keys((getSnapData(year,snap).teams)||{}).sort();
  var st = CF[side];
  var prev = st.teams[st.idx];
  st.teams = teams;
  var newIdx = teams.indexOf(prev);
  if(newIdx<0){
    // Modern Hub default: FNC on the left, KRÜ on the right. Falls back to
    // the first/second alphabetical teams if either isn't in the snapshot.
    if (LOCK_CURRENT) {
      var pref = side==='a' ? 'FNC' : 'KRÜ';
      newIdx = teams.indexOf(pref);
    }
    if (newIdx < 0) {
      if(side==='a') newIdx = 0;
      else newIdx = teams.length>1 ? 1 : 0;
    }
  }
  st.idx = Math.max(0, newIdx);
  buildCoverflow(side);
  // suppress the cylinder-rotation transition during a rebuild — items appear in place
  var stage = document.getElementById('cf-'+side);
  var track = stage.querySelector('.cf-track');
  var prevTrans = track.style.transition;
  track.style.transition = 'none';
  updateCoverflow(side);
  void track.offsetWidth; // force reflow so the suppression takes effect
  track.style.transition = prevTrans;
}

var CF_ANGLE = 26;   // degrees per item slot
var CF_RADIUS = 200; // cylinder radius (px)

function buildCoverflow(side){
  var stage = document.getElementById('cf-'+side);
  var track = stage.querySelector('.cf-track');
  var st = CF[side];
  var year = side==='a' ? yearA : yearB;
  var snap = side==='a' ? snapA : snapB;
  var sd = getSnapData(year, snap);
  var sdTeams = (sd && sd.teams) || {};
  track.innerHTML = st.teams.map(function(t,i){
    var teamObj = sdTeams[t] || {};
    var rating = (teamObj.overall_rating != null) ? teamObj.overall_rating : 0;
    var rStr = (rating >= 0 ? '+' : '') + rating.toFixed(2);
    var rCls = rating >= 0 ? 'cf-rtg-pos' : 'cf-rtg-neg';
    return '<div class="cf-item" data-side="'+side+'" data-idx="'+i+'">'+
      '<div class="cf-card">'+
        '<img src="/logos/'+t+'.png" alt="'+t+'" onerror="this.outerHTML=\\'<div class=cf-fallback>'+t+'</div>\\'">'+
      '</div>'+
      '<div class="cf-rtg '+rCls+'">'+rStr+'</div>'+
    '</div>';
  }).join('');
  track.querySelectorAll('.cf-item').forEach(function(el){
    el.addEventListener('click', function(){
      var idx = parseInt(el.dataset.idx,10);
      if(idx === CF[side].idx) return;
      CF[side].idx = idx;
      updateCoverflow(side);
      clearResult();
    });
  });
}

function updateCoverflow(side){
  var st = CF[side];
  var stage = document.getElementById('cf-'+side);
  var track = stage.querySelector('.cf-track');
  // rotate the cylinder so the selected item sits at the front
  track.style.transform = 'translateZ(-'+CF_RADIUS+'px) rotateY('+(-st.idx * CF_ANGLE)+'deg)';
  var items = track.querySelectorAll('.cf-item');
  items.forEach(function(el, i){
    var off = i - st.idx;
    var abs = Math.abs(off);
    el.classList.toggle('center', off===0);
    el.style.transform = 'rotateY('+(i*CF_ANGLE)+'deg) translateZ('+CF_RADIUS+'px)';
    if(abs > 5){
      el.style.opacity = 0;
      el.style.pointerEvents = 'none';
    } else {
      el.style.opacity = Math.max(0.12, 1 - abs*0.22);
      el.style.pointerEvents = abs<=3 ? 'auto' : 'none';
    }
  });
  var prevBtn = stage.querySelector('.cf-arrow[data-dir="-1"]');
  var nextBtn = stage.querySelector('.cf-arrow[data-dir="1"]');
  if(prevBtn) prevBtn.disabled = (st.idx <= 0);
  if(nextBtn) nextBtn.disabled = (st.idx >= st.teams.length-1);
  var org = st.teams[st.idx] || '';
  document.getElementById('cf-name-'+side).textContent = org;
  document.getElementById('cf-region-'+side).textContent = ORG_REGIONS[org] || '';
}

function shiftCoverflow(side, delta){
  var st = CF[side];
  if(!st.teams.length) return;
  var n = st.teams.length;
  var ni = Math.max(0, Math.min(n-1, st.idx + delta));
  if(ni === st.idx) return;
  st.idx = ni;
  updateCoverflow(side);
  clearResult();
}

var _clearTimer = null;
function clearResult(){
  var sec = document.getElementById('result-section');
  if(!sec || !sec.children.length) return;
  if(_clearTimer){ clearTimeout(_clearTimer); _clearTimer=null; }
  // bring the team selectors into view so the fade has somewhere to "go" instead of jolting
  var sg = document.querySelector('.side-grid');
  if(sg){
    var r = sg.getBoundingClientRect();
    var iframeOff = false;
    if (window !== window.top) {
      try {
        var fr = window.frameElement;
        if (fr) {
          var pr = fr.getBoundingClientRect();
          var pw = window.parent.innerWidth, ph = window.parent.innerHeight;
          if (pr.right <= 0 || pr.left >= pw || pr.bottom <= 0 || pr.top >= ph) iframeOff = true;
        }
      } catch(e) {}
    }
    if(!iframeOff && (r.top < -20 || r.top > window.innerHeight - 200)){
      try { sg.scrollIntoView({behavior:'smooth', block:'start'}); } catch(e){ sg.scrollIntoView(); }
    }
  }
  sec.classList.add('rs-fade-out');
  _clearTimer = setTimeout(function(){
    sec.innerHTML = '';
    sec.classList.remove('rs-fade-out');
    _clearTimer = null;
  }, 340);
}

// ── Year scrubber ────────────────────────────────────────────────────────────
function setYear(side, year){
  if(YEARS.indexOf(year)<0) return;
  if(side==='a' && yearA===year) return;
  if(side==='b' && yearB===year) return;
  if(side==='a'){ yearA=year; snapA=getLastSnap(year); }
  else { yearB=year; snapB=getLastSnap(year); }
  // update scrubber visuals
  var pct = (YEARS.indexOf(year)/(YEARS.length-1))*100;
  var scrubber = document.querySelector('.yr-scrubber[data-side="'+side+'"]');
  scrubber.querySelectorAll('.yr-tick').forEach(function(t){ t.classList.toggle('active', t.dataset.year===year); });
  scrubber.querySelectorAll('.yr-labels span').forEach(function(s){ s.classList.toggle('active', s.dataset.year===year); });
  scrubber.querySelector('.yr-knob').style.left = pct+'%';
  populateSnapSeg(side); populateTeams(side);
  clearResult();
}

document.querySelectorAll('.yr-scrubber').forEach(function(sc){
  var side = sc.dataset.side;
  sc.querySelectorAll('.yr-tick, .yr-labels span').forEach(function(el){
    el.addEventListener('click', function(){ setYear(side, el.dataset.year); });
  });
  // drag knob
  var track = sc.querySelector('.yr-track');
  var dragging = false;
  function knobFromX(clientX){
    var r = track.getBoundingClientRect();
    var ratio = Math.max(0, Math.min(1, (clientX - r.left) / r.width));
    var idx = Math.round(ratio * (YEARS.length-1));
    return YEARS[idx];
  }
  sc.querySelector('.yr-knob').addEventListener('mousedown', function(e){ dragging=true; e.preventDefault(); });
  track.addEventListener('mousedown', function(e){ var y=knobFromX(e.clientX); setYear(side, y); dragging=true; });
  document.addEventListener('mousemove', function(e){ if(!dragging) return; setYear(side, knobFromX(e.clientX)); });
  document.addEventListener('mouseup', function(){ dragging=false; });
  // touch
  track.addEventListener('touchstart', function(e){ var t=e.touches[0]; setYear(side, knobFromX(t.clientX)); dragging=true; }, {passive:true});
  document.addEventListener('touchmove', function(e){ if(!dragging) return; var t=e.touches[0]; setYear(side, knobFromX(t.clientX)); }, {passive:true});
  document.addEventListener('touchend', function(){ dragging=false; });
});

// ── Coverflow controls ───────────────────────────────────────────────────────
document.querySelectorAll('.cf-arrow').forEach(function(b){
  b.addEventListener('click', function(){ shiftCoverflow(b.dataset.side, parseInt(b.dataset.dir,10)); });
});
['a','b'].forEach(function(side){
  var stage = document.getElementById('cf-'+side);
  // wheel
  stage.addEventListener('wheel', function(e){
    e.preventDefault();
    var d = (e.deltaY||e.deltaX) > 0 ? 1 : -1;
    if(stage._wt && Date.now()-stage._wt < 90) return;
    stage._wt = Date.now();
    shiftCoverflow(side, d);
  }, {passive:false});
  // drag
  var st = CF[side];
  stage.addEventListener('mousedown', function(e){ st.dragging=true; st.startX=e.clientX; st.startIdx=st.idx; });
  document.addEventListener('mousemove', function(e){
    if(!st.dragging) return;
    var dx = e.clientX - st.startX;
    var ni = Math.max(0, Math.min(st.teams.length-1, st.startIdx - Math.round(dx/60)));
    if(ni !== st.idx){ st.idx = ni; updateCoverflow(side); clearResult(); }
  });
  document.addEventListener('mouseup', function(){ st.dragging=false; });
  stage.addEventListener('touchstart', function(e){ var t=e.touches[0]; st.dragging=true; st.startX=t.clientX; st.startIdx=st.idx; }, {passive:true});
  document.addEventListener('touchmove', function(e){
    if(!st.dragging) return;
    var t=e.touches[0]; var dx = t.clientX - st.startX;
    var ni = Math.max(0, Math.min(st.teams.length-1, st.startIdx - Math.round(dx/55)));
    if(ni !== st.idx){ st.idx = ni; updateCoverflow(side); clearResult(); }
  }, {passive:true});
  document.addEventListener('touchend', function(){ st.dragging=false; });
});

// ── Team search ──────────────────────────────────────────────────────────────
['a','b'].forEach(function(side){
  var input = document.getElementById('cf-search-'+side);
  var clear = document.querySelector('.cf-search-clear[data-side="'+side+'"]');
  if(!input) return;
  function applySearch(q){
    q = (q||'').trim().toLowerCase();
    clear.classList.toggle('visible', q.length>0);
    if(!q) return;
    var teams = CF[side].teams;
    var prefix = -1, contain = -1;
    for(var i=0;i<teams.length;i++){
      var t = teams[i].toLowerCase();
      if(prefix<0 && t.indexOf(q)===0){ prefix=i; break; }
      if(contain<0 && t.indexOf(q)>=0) contain=i;
    }
    var hit = prefix>=0 ? prefix : contain;
    if(hit>=0 && hit !== CF[side].idx){ CF[side].idx = hit; updateCoverflow(side); clearResult(); }
  }
  input.addEventListener('input', function(){ applySearch(input.value); });
  input.addEventListener('keydown', function(e){
    if(e.key==='Enter'){ e.preventDefault(); applySearch(input.value); input.blur(); }
    if(e.key==='Escape'){ input.value=''; applySearch(''); input.blur(); }
  });
  if(clear) clear.addEventListener('click', function(){ input.value=''; applySearch(''); input.focus(); });
});

// keyboard: left/right cycles last-focused side
var lastSide = 'a';
document.querySelectorAll('.cf-stage').forEach(function(s){
  s.addEventListener('mouseenter', function(){ lastSide = s.id.split('-')[1]; });
});
document.addEventListener('keydown', function(e){
  if(e.target && /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName)) return;
  if(e.key==='ArrowLeft'){ shiftCoverflow(lastSide,-1); e.preventDefault(); }
  else if(e.key==='ArrowRight'){ shiftCoverflow(lastSide,1); e.preventDefault(); }
});

// ── Mode toggle ──────────────────────────────────────────────────────────────
document.querySelectorAll('.mode-btn').forEach(function(b){
  b.addEventListener('click', function(){
    if(mode===b.dataset.mode) return;
    mode = b.dataset.mode;
    document.querySelectorAll('.mode-btn').forEach(function(x){ x.classList.remove('active'); });
    b.classList.add('active');
  });
});

document.querySelectorAll('.fmt-btn').forEach(function(btn){
  btn.addEventListener('click', function(){
    if(fmt===btn.dataset.fmt) return;
    fmt=btn.dataset.fmt;
    document.querySelectorAll('.fmt-btn').forEach(function(b){ b.classList.remove('active'); });
    btn.classList.add('active');
    clearResult();
  });
});

// ── Veto model ───────────────────────────────────────────────────────────────

function getActivePool(year, snap) {
  var key = year+'_'+snap;
  var cp = (VETO.computed_pools||{})[key];
  if (cp && cp.length >= 7) return cp;
  return (VETO.snap_pools||{})[key] || null;
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

var ACTION_CLS = {banA:['step-lbl-banA','rv-act-banA'], banB:['step-lbl-banB','rv-act-banB'], pickA:['step-lbl-pickA','rv-act-pickA'], pickB:['step-lbl-pickB','rv-act-pickB'], dec:['step-lbl-dec','rv-act-dec']};
function actionLabel(orgA, orgB, key){
  if(key==='dec') return 'Decider';
  var verb = key.indexOf('ban')===0 ? 'Ban' : 'Pick';
  var team = key.charAt(key.length-1)==='A' ? orgA : orgB;
  return verb+' '+team;
}
function fateLabel(orgA, orgB, key){
  if(key==='dec') return 'Decider';
  var verb = key.indexOf('ban')===0 ? 'Banned by' : 'Picked by';
  var team = key.charAt(key.length-1)==='A' ? orgA : orgB;
  return verb+' '+team;
}
var RANK_LABELS = ['#1 Most likely','#2','#3'];
var MAP_IMG_OVERRIDES = {}; // map names match files in /static/maps/ (lowercase). For odd casing, fall back gracefully.

function mapImg(name){ return '/maps/' + (name||'').toLowerCase() + '.png'; }
function logoImg(org){ return '/logos/' + org + '.png'; }
function logoTag(org, cls){
  cls = cls || '';
  return '<img src="'+logoImg(org)+'" class="'+cls+'" alt="'+org+'" onerror="this.style.visibility=\\'hidden\\'">';
}
function intlBadgeHtml(org, bd) {
  if(!bd.region || (!bd.regOff && !bd.indBonus)) return '';
  var tot=bd.total, cls=tot>0.05?'rd-pos':tot<-0.05?'rd-neg':'rt-neu';
  var parts=[];
  if(Math.abs(bd.regOff)>0.005) parts.push((bd.regOff>=0?'+':'')+bd.regOff.toFixed(2)+' '+bd.region+' region');
  if(Math.abs(bd.indBonus)>0.005) parts.push((bd.indBonus>=0?'+':'')+bd.indBonus.toFixed(2)+' indiv.');
  return '<div class="intl-adj"><span class="intl-adj-label">Intl adj</span><span class="intl-adj-val '+cls+'">'+(tot>=0?'+':'')+tot.toFixed(2)+'</span><span class="intl-adj-tip">'+parts.join(', ')+'</span></div>';
}

function simulate() {
  var orgA = CF.a.teams[CF.a.idx], orgB = CF.b.teams[CF.b.idx];
  if(!orgA||!orgB) return null;
  var sdA=getSnapData(yearA,snapA), sdB=getSnapData(yearB,snapB);
  var tA=(sdA.teams||{})[orgA], tB=(sdB.teams||{})[orgB];
  if(!tA||!tB) return null;
  // β = 0.25 — paired with RD_POWER=0.35 in BuildMapRatings.py (2026-05-14).
  // The new RD transform compresses rating signal more than sqrt; β=0.25
  // extracts cleaner predictions from the compressed space. Joint sweep
  // (BacktestSharpness3.py) found this strictly dominates the prior config
  // (Brier -1.07%, LL -0.85%, ECE -3.2%, σ -0.4%, sharpness +0.5%) while
  // keeping GEN #1 after Shanghai and all trophy winners at their ranks.
  // Same β for same-region and cross-region — no manipulation.
  var beta = 0.25;

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
  var topSeqs = topVetoSequences(tA,tB,orgA,orgB,pool,yearA,yearB,snapA,snapB,fmt,3);
  var hasPatt = !!((((VETO.teams||{})[yearA+'_'+snapA]||{})[orgA]) || (((VETO.teams||{})[yearB+'_'+snapB]||{})[orgB]));

  return {
    orgA:orgA, orgB:orgB, tA:tA, tB:tB, beta:beta,
    yearA:yearA, snapA:snapA, lblA:lblA, snapKeyA:snapKeyA, intlA:intlA,
    yearB:yearB, snapB:snapB, lblB:lblB, snapKeyB:snapKeyB, intlB:intlB,
    fmt:fmt, fmtLabel:fmtLabel, thresh:thresh, pool:pool,
    pctA:pctA, pctB:pctB,
    fateCnt:fateCnt, mapPlays:mapPlays, mapWins:mapWins, nSims:nSims,
    topSeqs:topSeqs, hasPatt:hasPatt
  };
}

function buildMapRows(R){
  var sorted = R.pool.slice().sort(function(a,b){ return R.mapPlays[b] - R.mapPlays[a]; });
  return sorted.map(function(m){
    var dA=(R.tA.maps[m]||{}).rating!=null?(R.tA.maps[m]||{}).rating:R.tA.overall_rating;
    var dB=(R.tB.maps[m]||{}).rating!=null?(R.tB.maps[m]||{}).rating:R.tB.overall_rating;
    var rA=getGlobalRating(R.orgA, R.snapKeyA, dA), rB=getGlobalRating(R.orgB, R.snapKeyB, dB);
    var bA_=R.fateCnt.banA[m]/R.nSims, pA_m=R.fateCnt.pickA[m]/R.nSims, dc=R.fateCnt.dec[m]/R.nSims, pB_m=R.fateCnt.pickB[m]/R.nSims, bB_=R.fateCnt.banB[m]/R.nSims;
    var bar=''; [[bA_,'fs-banA'],[pA_m,'fs-pickA'],[dc,'fs-dec'],[pB_m,'fs-pickB'],[bB_,'fs-banB']].forEach(function(p){ if(p[0]>0.005) bar+='<div class="fate-seg '+p[1]+'" style="width:'+(p[0]*100).toFixed(1)+'%"></div>'; });
    var fv={banA:bA_,pickA:pA_m,pickB:pB_m,banB:bB_,dec:dc};
    var dom='banA'; Object.keys(fv).forEach(function(k){if(fv[k]>fv[dom]) dom=k;});
    var fateLabels = {banA:fateLabel(R.orgA,R.orgB,'banA'), pickA:fateLabel(R.orgA,R.orgB,'pickA'), pickB:fateLabel(R.orgA,R.orgB,'pickB'), banB:fateLabel(R.orgA,R.orgB,'banB'), dec:'Decider'};
    var rACls=rA>0.05?'rt-pos':rA<-0.05?'rt-neg':'rt-neu';
    var rBCls=rB>0.05?'rt-pos':rB<-0.05?'rt-neg':'rt-neu';
    var p_m=R.mapPlays[m]>0?1/(1+Math.exp(-R.beta*(rA-rB))):0.5;
    var projRd=(2*p_m-1)*13, rdCls=projRd>0.5?'rd-pos':projRd<-0.5?'rd-neg':'rt-neu';
    var probHtml='<span class="wp-prom-empty">— map banned —</span>';
    if(R.mapPlays[m]>0){
      var wp=R.mapWins[m]/R.mapPlays[m];
      var wpCls = wp>=0.55?'fav':(wp<=0.45?'dog':'neu');
      probHtml = '<div class="wp-prom">'+
        '<div class="wp-prom-num '+wpCls+'">'+(wp*100).toFixed(0)+'%</div>'+
        '<div class="wp-prom-bg"><div class="wp-prom-fill" style="width:'+Math.round(wp*100)+'%"></div></div>'+
      '</div>';
    }
    return '<tr><td><div class="bd-map-mini"><img src="'+mapImg(m)+'" onerror="this.style.display=\\'none\\'">'+m+'</div></td>'+
      '<td style="text-align:left;">'+probHtml+'</td>'+
      '<td><div class="fate-bar-wrap"><div class="fate-bar">'+bar+'</div><div class="fate-txt">'+fateLabels[dom]+'</div></div></td>'+
      '<td class="'+rACls+'">'+(rA>=0?'+':'')+rA.toFixed(2)+'</td>'+
      '<td class="'+rBCls+'">'+(rB>=0?'+':'')+rB.toFixed(2)+'</td>'+
      '<td class="'+rdCls+'">'+(projRd>=0?'+':'')+projRd.toFixed(1)+'</td></tr>';
  }).join('');
}

function vetoListHtml(R){
  if(!R.hasPatt) return '<div class="no-veto-data">No historical veto data available for these teams in the selected year.</div>';
  return R.topSeqs.map(function(seq,idx){
    var steps = seq.seq.map(function(step,si){
      var key = step.action + step.side;
      var cls = (ACTION_CLS[key] || ACTION_CLS.dec)[0];
      var lbl = actionLabel(R.orgA, R.orgB, key);
      var arrow = si < seq.seq.length-1 ? '<span class="step-arrow">›</span>' : '';
      return '<div class="veto-step"><span class="step-lbl '+cls+'">'+lbl+'</span><span class="step-map">'+step.map+'</span></div>'+arrow;
    }).join('');
    return '<div class="veto-seq">'+
      '<div class="veto-seq-header">'+
        '<span class="veto-seq-rank">'+RANK_LABELS[idx]+'</span>'+
      '</div>'+
      '<div class="veto-steps">'+steps+'</div>'+
    '</div>';
  }).join('');
}

function topHeaderHtml(R){
  return '<div class="result-card" style="margin-bottom:20px;">'+
    '<div class="result-top">'+
      '<div class="result-teams-row">'+
        '<div class="result-team-block">'+
          logoTag(R.orgA,'result-logo')+
          '<div class="result-org">'+R.orgA+'</div>'+
          '<div class="result-ctx">'+R.yearA+'&thinsp;&middot;&thinsp;'+R.lblA+'</div>'+
          intlBadgeHtml(R.orgA, R.intlA)+
          '<div class="result-pct '+(R.pctA>=50?'fav':'dog')+'">'+R.pctA+'%</div>'+
        '</div>'+
        '<div class="result-mid">'+
          '<div class="result-bar-label">'+R.fmtLabel+'</div>'+
          '<div class="result-bar-outer"><div class="result-bar-a" style="width:'+R.pctA+'%"></div><div class="result-bar-b" style="width:'+R.pctB+'%"></div></div>'+
        '</div>'+
        '<div class="result-team-block">'+
          logoTag(R.orgB,'result-logo')+
          '<div class="result-org">'+R.orgB+'</div>'+
          '<div class="result-ctx">'+R.yearB+'&thinsp;&middot;&thinsp;'+R.lblB+'</div>'+
          intlBadgeHtml(R.orgB, R.intlB)+
          '<div class="result-pct '+(R.pctB>=50?'fav':'dog')+'">'+R.pctB+'%</div>'+
        '</div>'+
      '</div>'+
    '</div>'+
  '</div>';
}

function breakdownHtml(R){
  return '<div class="veto-pred-card">'+
    '<div class="veto-pred-title">Predicted Veto — '+R.orgA+' vs '+R.orgB+'</div>'+
    vetoListHtml(R)+
  '</div>'+
  '<div class="result-card">'+
    '<div class="fate-legend">'+
      '<div class="fate-legend-item"><div class="fate-dot" style="background:#f4b8c1"></div>Banned by '+R.orgA+'</div>'+
      '<div class="fate-legend-item"><div class="fate-dot" style="background:#5a2a7a"></div>Picked by '+R.orgA+'</div>'+
      '<div class="fate-legend-item"><div class="fate-dot" style="background:#c8b8d8"></div>Decider</div>'+
      '<div class="fate-legend-item"><div class="fate-dot" style="background:#7ab8e8"></div>Picked by '+R.orgB+'</div>'+
      '<div class="fate-legend-item"><div class="fate-dot" style="background:#b8e8d4"></div>Banned by '+R.orgB+'</div>'+
    '</div>'+
    '<table class="map-tbl"><thead><tr>'+
      '<th>Map</th>'+
      '<th>'+R.orgA+' win% if played</th>'+
      '<th>Veto outcome</th>'+
      '<th>'+R.orgA+' rtg</th><th>'+R.orgB+' rtg</th>'+
      '<th>Proj. RD ('+R.orgA+')</th>'+
    '</tr></thead><tbody>'+buildMapRows(R)+'</tbody></table>'+
    '<div class="result-note">'+R.nSims.toLocaleString()+' simulations &middot; '+R.fmt.toUpperCase()+' &middot; veto driven by historical ban/pick patterns &middot; ratings not normalized across seasons</div>'+
  '</div>';
}

function renderStraight(R){
  document.getElementById('result-section').innerHTML = topHeaderHtml(R) + breakdownHtml(R);
}

// ── Dramatic reveal ──────────────────────────────────────────────────────────
var revealAbort = false;

function wait(ms){ return new Promise(function(res){ setTimeout(res, ms); }); }
function abortable(ms){
  return new Promise(function(res){
    var t0=Date.now();
    (function step(){
      if(revealAbort) return res();
      if(Date.now()-t0 >= ms) return res();
      setTimeout(step, Math.min(50, ms-(Date.now()-t0)));
    })();
  });
}

function rvSeqFor(R){
  if(R.topSeqs && R.topSeqs.length) return R.topSeqs[0].seq;
  // fallback: build deterministic sequence from VETO_STEPS using uniform pool
  var rem=R.pool.slice(), seq=[];
  (VETO_STEPS[R.fmt]||VETO_STEPS.bo3).forEach(function(step){
    var m = rem[Math.floor(Math.random()*rem.length)];
    seq.push({side:step.side, action:step.action, map:m});
    rem = rem.filter(function(x){return x!==m;});
  });
  if(rem.length) seq.push({side:'',action:'dec',map:rem[0]});
  return seq;
}

function renderDramatic(R){
  revealAbort = false;
  R._finished = false;
  var section = document.getElementById('result-section');
  section.innerHTML =
    '<div class="reveal-stage" id="reveal-stage">'+
      '<button class="reveal-skip" id="reveal-skip">Skip &raquo;</button>'+
      '<div class="rv-step" id="rv-step-label">Initializing simulation</div>'+
      '<div id="rv-body"></div>'+
    '</div>';
  document.getElementById('reveal-skip').addEventListener('click', function(){
    revealAbort = true;
    finishReveal(R);
  });
  rvScroll(document.getElementById('reveal-stage'), 'start');
  playReveal(R);
}

function rvScroll(el, block){
  if(!el) return;
  // When iframed into the Modern Hub (LOCK_CURRENT), the simulator's reveal
  // animation keeps firing even after the user has switched to another tab.
  // scrollIntoView bubbles through ancestor scroll containers, including the
  // parent window — so a still-running sim would yank the parent page back
  // to the simulator panel every few hundred ms. Bail out if our iframe is
  // currently off-screen in the parent's viewport.
  if (window !== window.top) {
    try {
      var fr = window.frameElement;
      if (fr) {
        var r = fr.getBoundingClientRect();
        var pw = window.parent.innerWidth, ph = window.parent.innerHeight;
        if (r.right <= 0 || r.left >= pw || r.bottom <= 0 || r.top >= ph) return;
      }
    } catch(e) { /* cross-origin parent — fall through */ }
  }
  try { el.scrollIntoView({behavior:'smooth', block: block || 'center'}); }
  catch(e){ el.scrollIntoView(); }
}

function fetchMvpStat(org, year, snap, nMaps, host){
  fetch('/mapelo/mvp-stat/' + encodeURIComponent(org)
        + '?year='   + encodeURIComponent(year)
        + '&snap='   + encodeURIComponent(snap)
        + '&n_maps=' + encodeURIComponent(nMaps))
    .then(function(r){ return r.ok ? r.json() : null; })
    .then(function(s){
      if(!s || !s.player) return;
      var line = document.createElement('div');
      line.className = 'rv-statline';
      var kda = (s.K!=null && s.D!=null && s.A!=null) ? (s.K+'/'+s.D+'/'+s.A) : '';
      var extras = [];
      if(s.ACS) extras.push(Math.round(s.ACS)+' ACS');
      if(s.R)   extras.push((+s.R).toFixed(2)+' rtg');
      var suffix = extras.length ? ' ('+extras.join(', ')+')' : '';
      line.innerHTML = '<span class="rv-mvp-tag">MVP:</span> <strong>'+s.player+'</strong> goes <em>'+kda+suffix+'</em>.';
      host.appendChild(line);
      setTimeout(function(){ line.classList.add('shown'); rvScroll(line, 'center'); }, 80);
    })
    .catch(function(){});
}

function setStepLabel(txt){
  var el = document.getElementById('rv-step-label');
  if(el) el.textContent = txt;
}

function playReveal(R){
  var body = document.getElementById('rv-body');
  // Phase 1 — intro
  setStepLabel('Simulating');
  body.innerHTML =
    '<div class="rv-shimmer"></div>'+
    '<div class="rv-intro">'+
      '<div class="rv-intro-team">'+logoTag(R.orgA,'')+'<div style="font-family:Syne,sans-serif;font-weight:800;">'+R.orgA+'</div></div>'+
      '<div class="rv-intro-vs">VS</div>'+
      '<div class="rv-intro-team b">'+logoTag(R.orgB,'')+'<div style="font-family:Syne,sans-serif;font-weight:800;">'+R.orgB+'</div></div>'+
    '</div>';
  tick({freq:520,dur:.18,vol:.04,type:'sine'});

  abortable(1100).then(function(){
    if(revealAbort) return;
    // Phase 2 — veto reveal
    setStepLabel('Predicted veto sequence');
    var seq = rvSeqFor(R);
    body.innerHTML = '<div class="rv-veto-grid" id="rv-veto-grid"></div>';
    var grid = document.getElementById('rv-veto-grid');
    rvScroll(grid, 'center');
    seq.forEach(function(step){
      var key = step.action + step.side;
      var cls = (ACTION_CLS[key] || ACTION_CLS.dec)[1];
      var lbl = actionLabel(R.orgA, R.orgB, key);
      var slot = document.createElement('div');
      slot.className = 'rv-veto-slot rv-vs-pending';
      // Placeholder content — actual map is hidden until this slot is revealed.
      slot.innerHTML = '<div class="rv-vs-q">?</div>';
      // Stash the real content for the reveal step.
      slot.dataset.map = step.map;
      slot.dataset.action = step.action;
      slot.dataset.cls = cls;
      slot.dataset.lbl = lbl;
      grid.appendChild(slot);
    });
    return revealVetoSlots(grid).then(function(){
      if(revealAbort) return;
      return revealMaps(R, seq, body);
    });
  }).then(function(){
    finishReveal(R);
  });
}

function revealVetoSlots(grid){
  var slots = grid.querySelectorAll('.rv-veto-slot');
  return new Promise(function(res){
    var i = 0;
    function next(){
      if(revealAbort || i>=slots.length){ return res(); }
      var slot = slots[i];
      var map    = slot.dataset.map || '';
      var action = slot.dataset.action || 'dec';
      var cls    = slot.dataset.cls || 'rv-act-dec';
      var lbl    = slot.dataset.lbl || 'Decider';
      slot.classList.remove('rv-vs-pending');
      if(action === 'ban') slot.classList.add('banned');
      slot.innerHTML =
        '<img src="'+mapImg(map)+'" onerror="this.style.visibility=\\'hidden\\'">'+
        '<div class="rv-vs-map">'+map+'</div>'+
        '<div class="rv-vs-act '+cls+'">'+lbl+'</div>';
      slot.classList.add('revealed');
      if(action === 'ban')      tick({freq:380, dur:.07, vol:.05, type:'square'});
      else if(action === 'pick') tick({freq:1200,dur:.06, vol:.06, type:'square'});
      else                       tick({freq:900, dur:.08, vol:.05, type:'square'});
      i++;
      setTimeout(next, 360);
    }
    next();
  });
}

function revealMaps(R, seq, body){
  // collect picked + decider in order
  var played = seq.filter(function(s){ return s.action==='pick' || s.action==='dec'; });
  if(!played.length) return Promise.resolve();
  setStepLabel('Map results');
  var mapsHost = document.createElement('div');
  mapsHost.className = 'rv-maps';
  mapsHost.id = 'rv-maps-host';
  body.appendChild(mapsHost);

  var seriesA = 0, seriesB = 0;
  return played.reduce(function(p, step, idx){
    return p.then(function(){
      if(revealAbort) return;
      var m = step.map;
      var dA=(R.tA.maps[m]||{}).rating!=null?(R.tA.maps[m]||{}).rating:R.tA.overall_rating;
      var dB=(R.tB.maps[m]||{}).rating!=null?(R.tB.maps[m]||{}).rating:R.tB.overall_rating;
      var rA=getGlobalRating(R.orgA, R.snapKeyA, dA), rB=getGlobalRating(R.orgB, R.snapKeyB, dB);
      var pA = 1/(1+Math.exp(-R.beta*(rA-rB)));
      var winA = Math.random() < pA;
      if(winA) seriesA++; else seriesB++;
      var pickedBy = step.action==='dec' ? 'Decider' :
                      (step.side==='A' ? R.orgA+' pick' : R.orgB+' pick');
      var pAFav = Math.max(pA, 1-pA);
      var score = sampleScore(pAFav);

      var card = document.createElement('div');
      card.className = 'rv-map-card';
      card.innerHTML =
        '<div class="rv-map-bg" style="background-image:url('+mapImg(m)+')"></div>'+
        '<div class="rv-map-inner">'+
          '<div>'+
            '<div class="rv-map-num">Map '+(idx+1)+'</div>'+
            '<div class="rv-map-name">'+m+'</div>'+
            '<div class="rv-map-pickedby">'+pickedBy+'</div>'+
          '</div>'+
          '<div class="rv-map-h2h">'+
            '<div class="rv-map-team">'+
              logoTag(R.orgA,'')+
              '<div class="rv-map-team-name">'+R.orgA+'</div>'+
              '<div class="rv-map-score" id="rv-score-A-'+idx+'">0</div>'+
              '<div class="rv-map-team-pct" id="rv-pct-A-'+idx+'"></div>'+
            '</div>'+
            '<div class="rv-map-vs-mini">VS</div>'+
            '<div class="rv-map-team">'+
              logoTag(R.orgB,'')+
              '<div class="rv-map-team-name">'+R.orgB+'</div>'+
              '<div class="rv-map-score" id="rv-score-B-'+idx+'">0</div>'+
              '<div class="rv-map-team-pct" id="rv-pct-B-'+idx+'"></div>'+
            '</div>'+
          '</div>'+
          '<div class="rv-map-result-badge" id="rv-badge-'+idx+'">'+(winA?R.orgA:R.orgB)+' takes it</div>'+
        '</div>';
      mapsHost.appendChild(card);
      setTimeout(function(){ card.classList.add('shown'); }, 20);
      rvScroll(card, 'center');
      tick({freq:660,dur:.12,vol:.06,type:'sine'});
      return abortable(420).then(function(){
        if(revealAbort) return;
        return animateRoundTally(idx, winA, score);
      }).then(function(){
        if(revealAbort) return;
        return revealMapPct(idx, pA, winA);
      }).then(function(){
        if(revealAbort) return;
        var thresh = R.thresh;
        if(seriesA >= thresh || seriesB >= thresh){
          var winnerSide = seriesA > seriesB ? 'A' : 'B';
          var winnerOrg = winnerSide==='A' ? R.orgA : R.orgB;
          var winnerYear = winnerSide==='A' ? R.yearA : R.yearB;
          var winnerSnap = winnerSide==='A' ? R.snapA : R.snapB;
          var clinch = document.createElement('div');
          clinch.className = 'rv-clinch';
          clinch.textContent = winnerOrg + ' clinches the series ' +
            Math.max(seriesA,seriesB) + '-' + Math.min(seriesA,seriesB);
          mapsHost.appendChild(clinch);
          setTimeout(function(){ clinch.classList.add('shown'); }, 20);
          rvScroll(clinch, 'center');
          tick({freq:1500,dur:.18,vol:.07,type:'sine'});
          setTimeout(function(){ tick({freq:1900,dur:.22,vol:.07,type:'sine'}); }, 110);
          // n_maps = total maps played in the simulated series (not maps won)
          fetchMvpStat(winnerOrg, winnerYear, winnerSnap, seriesA + seriesB, mapsHost);
          revealAbort = true; // halt remaining maps after clinch
        }
        return abortable(700);
      });
    });
  }, Promise.resolve());
}

function sampleScore(pFav){
  // pFav in [0.5, 1]. Stronger favorite → larger margin.
  // Valorant: first to 13, win by 2. Tie at 12-12 → OT, must win by 2 (14-12, 15-13, ...).
  var projRD = (2*pFav - 1) * 13;
  var jitter = (Math.random()*2 - 1) * 2.0;
  var loser = Math.round(13 - projRD + jitter);
  loser = Math.max(3, Math.min(13, loser));
  if(loser <= 11) return {winner: 13, loser: loser};
  // OT zone — score becomes 14-12, 15-13, 16-14, ...
  var loserOT = 12;
  var r = Math.random();
  if(r > 0.78){ loserOT = 13; }   // ~22% chance of an extended OT (15-13)
  if(r > 0.94){ loserOT = 14; }   // ~6% even longer (16-14)
  return {winner: loserOT + 2, loser: loserOT};
}

function genRoundSequence(winnerScore, loserScore){
  // Build a credible round sequence ending with a winner round (the clincher).
  // For OT we still just shuffle then append a final winner round.
  var seq = [];
  for(var i=0;i<winnerScore-1;i++) seq.push(true);
  for(var i=0;i<loserScore;i++) seq.push(false);
  // Fisher-Yates shuffle
  for(var i=seq.length-1; i>0; i--){
    var j = Math.floor(Math.random()*(i+1));
    var tmp = seq[i]; seq[i]=seq[j]; seq[j]=tmp;
  }
  seq.push(true); // clincher
  return seq;
}

function animateRoundTally(idx, winA, score){
  return new Promise(function(res){
    var elA = document.getElementById('rv-score-A-'+idx);
    var elB = document.getElementById('rv-score-B-'+idx);
    if(!elA || !elB) return res();
    var seq = genRoundSequence(score.winner, score.loser);
    var winnerEl = winA ? elA : elB;
    var loserEl  = winA ? elB : elA;
    var sW = 0, sL = 0;
    var i = 0;
    var perRound = 95;
    function bump(el){
      el.classList.add('bumped');
      setTimeout(function(){ el.classList.remove('bumped'); }, 110);
    }
    function step(){
      if(revealAbort){
        winnerEl.textContent = score.winner;
        loserEl.textContent = score.loser;
        return finish();
      }
      if(i >= seq.length) return finish();
      var winnerRd = seq[i];
      if(winnerRd){ sW++; winnerEl.textContent = sW; bump(winnerEl); }
      else        { sL++; loserEl.textContent  = sL; bump(loserEl);  }
      var isClincher = (i === seq.length - 1);
      if(isClincher){
        tick({freq:1500,dur:.16,vol:.07,type:'sine'});
        setTimeout(function(){ tick({freq:1850,dur:.18,vol:.06,type:'sine'}); }, 90);
      } else {
        tick({freq: 1080 + Math.random()*180, dur:.028, vol:.035, type:'square'});
      }
      i++;
      setTimeout(step, perRound);
    }
    function finish(){
      winnerEl.classList.add('win');
      loserEl.classList.add('lose');
      setTimeout(res, 320);
    }
    step();
  });
}

function revealMapPct(idx, pA, winA){
  return new Promise(function(res){
    var elA = document.getElementById('rv-pct-A-'+idx);
    var elB = document.getElementById('rv-pct-B-'+idx);
    var b   = document.getElementById('rv-badge-'+idx);
    if(elA){ elA.textContent = Math.round(pA*100)+'% to win'; if(winA) elA.classList.add('win'); elA.classList.add('shown'); }
    if(elB){ elB.textContent = Math.round((1-pA)*100)+'% to win'; if(!winA) elB.classList.add('win'); elB.classList.add('shown'); }
    if(b) b.classList.add('shown');
    setTimeout(res, 360);
  });
}

function finishReveal(R){
  if(R._finished) return;
  R._finished = true;
  revealAbort = true;
  var section = document.getElementById('result-section');
  // pop the skip button — reveal is done
  var skip = document.getElementById('reveal-skip');
  if(skip) skip.remove();
  // remove any old breakdown (replay edge case)
  var oldBd = document.getElementById('reveal-breakdown');
  if(oldBd) oldBd.remove();
  var bd = document.createElement('div');
  bd.id = 'reveal-breakdown';
  bd.style.marginTop = '20px';
  bd.innerHTML = topHeaderHtml(R) + breakdownHtml(R) +
    '<div style="text-align:center;margin-top:6px;"><button class="replay-btn" id="replay-btn">&#9654; Replay reveal</button></div>';
  section.appendChild(bd);
  var rb = document.getElementById('replay-btn');
  if(rb) rb.addEventListener('click', function(){ renderDramatic(R); });
  rvScroll(bd, 'start');
}

function runMatchup() {
  ensureAudio();
  // cancel any in-flight fade-out so we don't wipe the new content
  if(_clearTimer){ clearTimeout(_clearTimer); _clearTimer = null; }
  var sec = document.getElementById('result-section');
  if(sec){ sec.classList.remove('rs-fade-out'); sec.innerHTML = ''; }
  var R = simulate();
  if(!R) return;
  if(mode==='dramatic') renderDramatic(R);
  else renderStraight(R);
}

(function(){
  populateSnapSeg('a'); populateSnapSeg('b');
  populateTeams('a'); populateTeams('b');
  // Default selection: EG (A) vs NRG (B) on the standalone matchup page.
  // Modern Hub Simulator (LOCK_CURRENT) keeps the FNC vs KRÜ default set
  // inside populateTeams — don't override it here.
  if (!LOCK_CURRENT) {
    var ai = CF.a.teams.indexOf('EG');
    var bi = CF.b.teams.indexOf('NRG');
    if(ai >= 0){ CF.a.idx = ai; updateCoverflow('a'); }
    if(bi >= 0){ CF.b.idx = bi; updateCoverflow('b'); }
  }
})();

// ── Title intro animation (dot → typewriter), mirrors the Modern Hub ─────────
// Runs in both the standalone page and the iframed Modern Hub Simulator so
// the simulator tab always opens with a real title instead of a blank slot.
(function introMatchupTitle(){
  var title = document.getElementById('matchupTitle');
  if (!title) return;
  function sleep(ms){ return new Promise(function(r){ setTimeout(r, ms); }); }
  (async function(){
    title.style.opacity = '1';
    title.innerHTML = '<span class="dot-seed">·</span>';
    await sleep(380);
    // "Historical" only on the standalone page; the Modern Hub simulator
    // (LOCK_CURRENT iframe) is for live/current matchups, so the typewriter
    // there should read "Matchup Predictor" without the prefix.
    var text = LOCK_CURRENT ? 'Matchup Predictor' : 'Historical Matchup Predictor';
    title.textContent = '';
    var built = '';
    for (var i = 0; i < text.length; i++) {
      built += text[i];
      title.innerHTML = built + '<span class="type-cursor">|</span>';
      await sleep(48 + Math.random() * 38);
    }
    for (var j = 0; j < 4; j++) {
      title.innerHTML = text + (j % 2 === 0 ? '<span class="type-cursor">|</span>' : '');
      await sleep(210);
    }
    title.textContent = text;
  })();
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
  .page-title { font-family:'Syne',sans-serif; font-size:clamp(1.6rem,4vw,2.5rem); font-weight:700; letter-spacing:-1px; margin-bottom:28px; }
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
  table { width:100%; border-collapse:collapse; font-size:.88rem; }
  thead th { font-family:'Syne',sans-serif; font-size:.7rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; color:var(--soft); padding:8px 12px; text-align:right; border-bottom:2px solid #f0ecf4; cursor:pointer; user-select:none; white-space:nowrap; }
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
    from flask import request as _req
    full  = get_ratings()
    veto  = get_veto_model()
    intl  = get_intl_calibration()
    keep_meta = ('optimal_half_life_matches', 'brier_test', 'n_train', 'n_test', 'mc_n_sims', 'veto_noise_std')
    frontend_data = {
        'metadata':    {k: v for k, v in full['metadata'].items() if k in keep_meta},
        'ratings':     full['ratings'],
        'veto_model':  {'teams': veto.get('teams', {}), 'snap_pools': veto.get('snap_pools', {}), 'computed_pools': _build_computed_pools()},
        'intl_calib':  intl.get('calibration', {}),
        'intl_params': intl.get('params', {}),
        'org_regions': ORG_REGIONS,
    }
    lock_current = _req.args.get('lockCurrent') == '1'

    # When embedded in the Modern Hub Simulator tab:
    #   (1) augment the snapshot's team list to include every 2026 active org
    #   (2) override the snap_pool with the LIVE current pool, derived from the
    #       most recent VCT match's actual pick/ban sequence — the veto IS the
    #       pool, by definition. Beats any play-count heuristic.
    if lock_current:
        try:
            _live_pool = _current_pool_from_latest_veto()
            if _live_pool and len(_live_pool) >= 7:
                # Latest 2026 snapshot by ref_date — same rule the frontend uses
                # to pick which snap to render, so the pool lands where it'll be
                # read.
                _2026_snaps = (frontend_data.get("ratings", {}).get("2026") or {}).get("snapshots") or {}
                if _2026_snaps:
                    _latest_snap_id = max(
                        _2026_snaps.items(),
                        key=lambda kv: (kv[1].get("ref_date") or "", kv[0]),
                    )[0]
                    _target_key = f"2026_{_latest_snap_id}"
                    frontend_data["veto_model"].setdefault("snap_pools", {})[_target_key] = _live_pool
                    frontend_data["veto_model"].setdefault("computed_pools", {})[_target_key] = _live_pool
        except Exception:
            pass
        try:
            tl_path = os.path.join(ROOT, "data", "rating_timeline.json")
            if os.path.exists(tl_path):
                with open(tl_path) as _f:
                    _tl = json.load(_f)
                _cps = _tl.get("checkpoints", []) or []
                _last_ratings = _cps[-1].get("ratings", {}) if _cps else {}
            else:
                _last_ratings = {}
            _snaps = ((frontend_data["ratings"].get("2026") or {}).get("snapshots") or {})
            # Latest 2026 snapshot by ref_date — same selection rule the
            # simulator frontend uses, so the live-rating overlay lands on
            # the same snap the user will be looking at.
            _latest_snap_id = max(
                _snaps.items(),
                key=lambda kv: (kv[1].get("ref_date") or "", kv[0]),
            )[0] if _snaps else None
            _target = _snaps.get(_latest_snap_id) if _latest_snap_id else None
            if _target is not None:
                _existing_teams = _target.get("teams") or {}
                # Overwrite each team's overall_rating with the live (last-
                # checkpoint) value so the coverflow and headers show the same
                # number as the leaderboard. Per-map ratings are left at their
                # RAW snap values — backtesting showed shifting them by the
                # live-vs-snap delta hurt predictions (Brier 0.246 → 0.241 by
                # dropping the rebase, all else equal). The simulator's per-map
                # sim now uses the same raw map ratings the upcoming-card sim
                # uses, so the two views produce identical win probabilities.
                for _org, _td in list(_existing_teams.items()):
                    if _org not in _last_ratings:
                        continue
                    _cur = float(_last_ratings[_org])
                    _td["overall_rating"] = round(_cur, 4)
                # (b) Augment with active 2026 orgs that weren't in the
                # snapshot at all — use last-checkpoint rating as overall
                # and neutral map ratings.
                _example_team = next(iter(_existing_teams.values()), None)
                _map_keys = list((_example_team or {}).get("maps", {}).keys()) if _example_team else []
                _existing = set(_existing_teams.keys())
                for _org in ACTIVE_2026_ORGS:
                    if _org in _existing:
                        continue
                    _r = float(_last_ratings.get(_org, 0.0))
                    _existing_teams[_org] = {
                        "overall_rating": _r,
                        "w": 0, "l": 0,
                        "maps": {_m: {"rating": _r, "w": 0, "l": 0, "win_pct": 0.5} for _m in _map_keys},
                    }
                _target["teams"] = _existing_teams
        except Exception:
            pass

    html = MAPELO_MATCHUP_HTML.replace('RATINGS_JSON', json.dumps(frontend_data))
    html = html.replace('LOCK_CURRENT_FLAG', 'true' if lock_current else 'false')
    return html

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

@mapelo_bp.route('/mvp-stat/<org>')
def mapelo_mvp_stat(org):
    from flask import request as _req
    year   = _req.args.get('year', '2025')
    snap   = _req.args.get('snap', 'after_champions')
    n_maps = _req.args.get('n_maps', '3')  # maps PLAYED in the simulated series
    data = _get_mvp_stat(org, year, snap, n_maps) or {}
    return Response(json.dumps(data), mimetype='application/json')

# ─── Modern VCT Hub — backend ────────────────────────────────────────────────

_RATING_TIMELINE_PATH = os.path.join(ROOT, "data", "rating_timeline.json")
_MAP_RATINGS_PATH     = os.path.join(ROOT, "data", "map_ratings.json")

_mhub_cache        = {"data": None, "ts": 0.0}
_mhub_cache_lock   = _th.Lock()
_mhub_build_running = False
_mhub_last_trigger = 0.0
_MHUB_TTL          = 1800  # 30 min
_MHUB_TRIGGER_COOLDOWN = 120  # don't spawn a new RefreshLiveData more than once / 2 min per worker
_MHUB_PROGRESS_FILE = "/tmp/mhub_refresh_progress.json"
_MHUB_STDERR_FILE   = "/tmp/mhub_refresh_stderr.log"


def _live_event_ids_by_date():
    """Live event CSV ids, most-recent end-date first.  Used for any place that
    needs to know which event(s) are currently active (live map stats, roster
    lookup, current pool detection) without hardcoding event ids."""
    from MoreTestingMaybeFiles import live_events_today as _lt
    try:
        evs = _lt()
    except Exception:
        return []
    def _end(ev):
        return ev.get("end") or ev.get("start") or ""
    return [ev["id"] for ev in sorted(evs, key=_end, reverse=True)]

_MAPS_DIR       = os.path.join(ROOT, "data", "maps")
_MATCH_DATES_PATH = os.path.join(ROOT, "data", "match_dates.json")
_VETOS_PATH       = os.path.join(ROOT, "data", "map_vetos.csv")


def _current_pool_from_latest_veto():
    """Return the 7-map pool from the most recent VCT match with a complete
    pick/ban sequence. This is the authoritative source: a match's veto IS
    the active pool, by definition.

    Looks at data/map_vetos.csv (populated by scrapers/ScrapeMapVetos.py),
    joins to match dates, finds the latest match with ≥7 veto steps, and
    returns its distinct map names. Returns [] if no usable data."""
    if not os.path.exists(_VETOS_PATH) or not os.path.exists(_MATCH_DATES_PATH):
        return []
    try:
        with open(_MATCH_DATES_PATH) as f:
            match_dates = json.load(f)
        vetos = pd.read_csv(_VETOS_PATH)
    except Exception:
        return []
    if vetos.empty:
        return []
    vetos = vetos.copy()
    vetos["date"] = vetos["MatchID"].astype(str).map(match_dates)
    vetos = vetos.dropna(subset=["date"])
    if vetos.empty:
        return []
    counts = vetos.groupby("MatchID").size()
    valid_mids = counts[counts >= 7].index
    if len(valid_mids) == 0:
        return []
    vetos = vetos[vetos["MatchID"].isin(valid_mids)]
    # Most recent match wins; tie-break on higher MatchID (VLR IDs grow over time)
    latest = vetos.sort_values(["date", "MatchID"], ascending=False).iloc[0]
    latest_mid = int(latest["MatchID"])
    pool_maps = sorted(set(vetos[vetos["MatchID"] == latest_mid]["map"].dropna().tolist()))
    # Strip stray PICK/BAN/DECIDER/REMAIN suffixes if present (defensive)
    cleaned = []
    for m in pool_maps:
        for sfx in ("PICK", "BAN", "DECIDER", "REMAIN"):
            if m.endswith(sfx):
                m = m[:-len(sfx)]
                break
        cleaned.append(m)
    return sorted(set(cleaned))

# Snap key → event CSV stems for pool detection. Historical entries are frozen
# (those snapshots are fixed); 2026+ entries are auto-derived from
# BuildMapRatings.YEAR_CONFIGS so adding a new event (Masters London, Stage 2,
# Champions) in MoreTestingMaybeFiles.py is enough to propagate everywhere —
# no edits needed here.
_HISTORICAL_SNAP_POOL_EVENTS = {
    "2025_before_bangkok":   ["2025_kickoff"],
    "2025_after_bangkok":    ["2025_masters_bangkok"],
    "2025_before_toronto":   ["2025_stage1"],
    "2025_after_toronto":    ["2025_masters_toronto"],
    "2025_before_champions": ["2025_stage2"],
    "2025_after_champions":  ["2025_champions"],
    "2024_before_madrid":    ["2024_kickoff"],
    "2024_after_madrid":     ["2024_masters_madrid"],
    "2024_before_shanghai":  ["2024_stage1"],
    "2024_after_shanghai":   ["2024_masters_shanghai"],
    "2024_before_champions": ["2024_stage2"],
    "2024_after_champions":  ["2024_champions"],
}

def _build_dynamic_snap_pool_events():
    """For each dynamically-generated snapshot in BuildMapRatings.YEAR_CONFIGS,
    map ``<year>_<snap_id>`` to the snapshot's most recent event id. That event
    is the one whose matches define the pool era for that snap."""
    out = {}
    try:
        from scrapers.BuildMapRatings import YEAR_CONFIGS as _YC
    except Exception:
        return out
    for year, cfg in _YC.items():
        if int(year) < 2026:
            continue  # past years are frozen above
        for snap_id, snap in (cfg.get("snapshots") or {}).items():
            evs = snap.get("events") or []
            if evs:
                out[f"{year}_{snap_id}"] = [evs[-1]]
    return out

_SNAP_POOL_EVENTS = dict(_HISTORICAL_SNAP_POOL_EVENTS)
_SNAP_POOL_EVENTS.update(_build_dynamic_snap_pool_events())

def _load_event_map_records(event_ids):
    """
    Load (date, match_id, frozenset_of_maps) for the given event CSV stems.
    Map names in the CSVs are like 'BreezePICK', 'SplitBAN', 'Haven' (bare = decider).
    """
    try:
        with open(_MATCH_DATES_PATH) as f:
            match_dates = json.load(f)
    except Exception:
        return []

    records = []
    for eid in event_ids:
        fpath = os.path.join(_MAPS_DIR, f"{eid}.csv")
        if not os.path.exists(fpath):
            continue
        try:
            df = pd.read_csv(fpath, usecols=lambda c: c in ("MatchID", "MapName"))
            if "MapName" not in df.columns:
                continue
            for mid, grp in df.groupby("MatchID"):
                date = match_dates.get(str(int(mid)), "")
                if not date:
                    continue
                maps = set()
                for mn in grp["MapName"].unique():
                    clean = mn
                    for sfx in ("PICK", "BAN", "DECIDER", "REMAIN"):
                        if mn.endswith(sfx):
                            clean = mn[:-len(sfx)]
                            break
                    maps.add(clean)
                records.append((date, int(mid), frozenset(maps)))
        except Exception:
            pass

    records.sort()
    return records


def _detect_pool(records_sorted, as_of_date=None, target_size=7):
    """
    Derive the active map pool from played-map records.

    Algorithm: the most recently introduced map marks the start of the current
    pool era.  Pool = union of all maps played since that date, trimmed to
    `target_size` by frequency when a pool transition produces more than
    target_size maps (the least-played map is the one that was replaced).
    """
    if as_of_date:
        recs = [(d, mid, m) for d, mid, m in records_sorted if d <= as_of_date]
    else:
        recs = records_sorted[:]

    if not recs:
        return []

    # Find when each map was first seen
    first_seen: dict = {}
    for date, mid, maps in recs:
        for m in maps:
            if m not in first_seen:
                first_seen[m] = date

    if not first_seen:
        return []

    # The most-recently introduced map defines the start of the current era
    era_start = max(first_seen.values())

    # Pool = every map that appeared on or after era_start
    pool: set = set()
    freq: dict = {}
    for date, mid, maps in recs:
        if date >= era_start:
            pool |= maps
            for m in maps:
                freq[m] = freq.get(m, 0) + 1

    # If pool transition produced more maps than expected, keep only the
    # target_size most-frequently played ones (the dropped map is least frequent)
    if len(pool) > target_size:
        pool = set(sorted(pool, key=lambda m: -freq.get(m, 0))[:target_size])

    return sorted(pool)


def _build_computed_pools():
    """
    Return {snap_key: [map, ...]} for all known snaps using event map CSVs.
    Only emits pools with ≥ 7 maps; smaller results are omitted so the
    frontend falls back to the veto_model snap_pools for that key.
    """
    computed = {}
    for snap_key, event_ids in _SNAP_POOL_EVENTS.items():
        recs = _load_event_map_records(event_ids)
        pool = _detect_pool(recs)
        if len(pool) >= 7:
            computed[snap_key] = pool
    return computed


def _build_live_map_stats(beta=0.3237, min_games=2, shrink_prior=4):
    """
    Build per-team, per-map win rates from the most recent live event(s).
    Returns {org: {map: {w, l, win_pct, rating}}} where `rating` is a
    Massey-scale estimate (logit(blended_win_pct) / beta), shrunk toward 0.5
    for small samples.

    Live events are resolved dynamically from ALL_EVENTS via
    _live_event_ids_by_date(), so when Masters London / Stage 2 / Champions
    start, this picks them up with no code change.
    """
    import math as _m
    results_path = os.path.join(ROOT, "data", "match_results.csv")
    if not os.path.exists(results_path):
        return {}
    live_csv_paths = []
    for eid in _live_event_ids_by_date():
        p = os.path.join(ROOT, "data", "maps", f"{eid}.csv")
        if os.path.exists(p):
            live_csv_paths.append(p)
    if not live_csv_paths:
        return {}
    try:
        mr = pd.read_csv(results_path)
        s1 = pd.concat([
            pd.read_csv(p, usecols=lambda c: c in ("MatchID","MapNum","MapName","Org"))
            for p in live_csv_paths
        ], ignore_index=True)
        mr_maps = mr[mr["MapNum"].astype(str) != "all"].copy()
        mr_maps["MapNum"] = mr_maps["MapNum"].astype(int)
        s1["MapNum"] = s1["MapNum"].astype(int)
        joined = s1.merge(mr_maps[["MatchID","MapNum","WinnerOrg"]],
                          on=["MatchID","MapNum"], how="inner")
        # Deduplicate to one row per (MatchID, MapNum, Org) — CSV has one row per player
        map_level = joined[["MatchID","MapNum","MapName","Org","WinnerOrg"]].drop_duplicates(
            subset=["MatchID","MapNum","Org"])
        raw: dict = {}
        for _, row in map_level.iterrows():
            mn = row["MapName"]
            for sfx in ("PICK","BAN","DECIDER","REMAIN"):
                if mn.endswith(sfx): mn = mn[:-len(sfx)]; break
            org    = str(row["Org"])
            winner = row["WinnerOrg"]
            raw.setdefault(org, {}).setdefault(mn, {"w":0,"l":0})
            if org == winner:
                raw[org][mn]["w"] += 1
            else:
                raw[org][mn]["l"] += 1
        out: dict = {}
        for org, maps in raw.items():
            out[org] = {}
            for mp, v in maps.items():
                n   = v["w"] + v["l"]
                if n < min_games:
                    continue
                # Bayesian shrink toward 50%
                wp  = (v["w"] + shrink_prior * 0.5) / (n + shrink_prior)
                wp  = max(0.05, min(0.95, wp))
                rtg = round(_m.log(wp / (1 - wp)) / beta, 4)
                out[org][mp] = {"w": v["w"], "l": v["l"],
                                "win_pct": round(wp, 4), "rating": rtg}
        return out
    except Exception as e:
        print(f"[live_map_stats] {e}")
        return {}


_MHUB_STALE_HOURS  = 6     # refresh if last checkpoint > 6h old

MHUB_EVENT_BANDS = [
    {"id": "kickoff",   "label": "Kickoff",         "start": "2026-01-15", "end": "2026-02-16"},
    {"id": "santiago",  "label": "Masters Santiago", "start": "2026-02-28", "end": "2026-03-15"},
    {"id": "stage1",    "label": "Stage 1",          "start": "2026-04-01", "end": "2026-05-25"},
    {"id": "london",    "label": "Masters London",   "start": "2026-06-05", "end": "2026-06-21"},
    {"id": "stage2",    "label": "Stage 2",          "start": "2026-07-15", "end": "2026-09-06"},
    {"id": "champions", "label": "Champions",        "start": "2026-09-24", "end": "2026-10-18"},
]

MHUB_COLORS = {
    "SEN":  "#e3001a", "G2":   "#e8b800", "NRG":  "#ff6600", "100T": "#be0000",
    "C9":   "#0d8ac8", "EG":   "#1565c0", "MIBR": "#18a040", "KRÜ":  "#f9c200",
    "LEV":  "#7c3aed", "FUR":  "#ff4500", "LOUD": "#a3e635", "ENVY": "#9333ea",
    "TL":   "#f59e0b", "FNC":  "#f97316", "NAVI": "#fbbf24", "VIT":  "#dc2626",
    "BBL":  "#db2777", "GX":   "#ec4899", "KC":   "#ef4444", "TH":   "#6d28d9",
    "FUT":  "#0d9488", "M8":   "#be185d", "PCF":  "#65a30d", "ULF":  "#0284c7",
    "EF":   "#b91c1c",
    "PRX":  "#0ea5e9", "DRX":  "#c53030", "T1":   "#cc0000", "GEN":  "#15803d",
    "ZETA": "#7e22ce", "RRQ":  "#9f1239", "TS":   "#475569", "GE":   "#0e7490",
    "NS":   "#134e4a", "FS":   "#ea580c", "VL":   "#1d4ed8", "KRX":  "#1e40af",
    "DFM":  "#b91c1c", "ZETA": "#7e22ce", "TLN":  "#0369a1",
}


def _mhub_load():
    """Load rating_timeline.json + map_ratings.json and merge into hub payload."""
    result = {
        "status":      "ready",
        "event_bands": MHUB_EVENT_BANDS,
        "chart":       {"checkpoints": [], "match_events": []},
        "leaderboard": {"teams": [], "beta": 0.3237, "as_of_date": None},
    }

    # ── Chart data ─────────────────────────────────────────────────────────────
    if os.path.exists(_RATING_TIMELINE_PATH):
        with open(_RATING_TIMELINE_PATH) as f:
            tl = json.load(f)
        checkpoints   = tl.get("checkpoints",  [])
        match_events  = tl.get("match_events", [])
        result["chart"]["checkpoints"]  = checkpoints
        result["chart"]["match_events"] = match_events
        if checkpoints:
            result["as_of_date"] = checkpoints[-1]["date"]
            # Three states to distinguish:
            #   (a) recent terminal (done/error within 15 min) → data is
            #       fresh, status=ready, don't re-trigger.
            #   (b) recent non-terminal (any phase != done/error written in
            #       the last 60 s) → scrape is IN-FLIGHT, status=building so
            #       the frontend keeps polling, but don't re-trigger.
            #   (c) old or missing → status=building AND re-trigger.
            #
            # Earlier this collapsed (b) and (a) onto the same path, which
            # caused the page to render with stale on-disk data while the
            # scrape was still updating it.
            needs_trigger = True
            in_flight     = False
            try:
                if os.path.exists(_MHUB_PROGRESS_FILE):
                    with open(_MHUB_PROGRESS_FILE) as _pf:
                        _pd = json.load(_pf)
                    _phase = _pd.get("phase", "")
                    _age   = _time_mod.time() - _pd.get("ts", 0)
                    if _phase in ("done", "error") and _age < 900:
                        needs_trigger = False
                    elif _age < 60:
                        # scrape is actively running — keep frontend polling
                        # by reporting building, but don't re-trigger.
                        needs_trigger = False
                        in_flight     = True
            except Exception:
                pass
            if needs_trigger or in_flight:
                result["status"]      = "building"
                result["needs_trigger"] = needs_trigger
    else:
        result["status"] = "building"

    # ── Leaderboard from map_ratings.json (latest snapshot of the latest year) ─
    # No hardcoded snap names: pick the snapshot with the most recent ref_date,
    # falling back to insertion order if ref_date is missing. As new events get
    # scraped and BuildMapRatings rebuilds, this auto-promotes to the freshest.
    snap_data = None
    snap_name = None
    if os.path.exists(_MAP_RATINGS_PATH):
        with open(_MAP_RATINGS_PATH) as f:
            mr_json = json.load(f)
        all_ratings = mr_json.get("ratings", {}) or {}
        latest_year = max(all_ratings.keys()) if all_ratings else None
        snaps = (all_ratings.get(latest_year) or {}).get("snapshots", {}) if latest_year else {}
        if snaps:
            def _snap_sort_key(item):
                sid, sdata = item
                return (sdata.get("ref_date") or "", sid)
            snap_name, snap_data = max(snaps.items(), key=_snap_sort_key)

    # If timeline is available, override overall ratings with last checkpoint
    last_checkpoint_ratings = {}
    if result["chart"]["checkpoints"]:
        last_checkpoint_ratings = result["chart"]["checkpoints"][-1]["ratings"]

    # Build per-team recent-matches from match_events (include maps + event)
    recent_by_org: dict = {}
    for me in reversed(result["chart"]["match_events"]):
        for role in ("winner", "loser"):
            org = me[role]
            if org not in recent_by_org:
                recent_by_org[org] = []
            if len(recent_by_org[org]) < 4:
                is_winner = (role == "winner")
                recent_by_org[org].append({
                    "date":     me["date"],
                    "opponent": me["loser"] if is_winner else me["winner"],
                    "result":   "W" if is_winner else "L",
                    "score":    me["series_score"],
                    "delta":    me["winner_delta"] if is_winner else me["loser_delta"],
                    "maps":     me.get("maps", []),
                    "event_id": me.get("event_id", ""),
                    "match_id": me.get("match_id", ""),
                })

    # Build most-recently-used roster for each org from 2026 player CSVs
    _roster_by_org = {}
    _hs_cache = {}
    try:
        _hs_path = os.path.join(ROOT, 'data', 'headshots.json')
        if os.path.exists(_hs_path):
            with open(_hs_path) as _hf:
                _hs_cache = json.load(_hf)
    except Exception:
        pass
    _player_frames = []
    # Roster lookup: walk every live event by recency, then fall back to all
    # 2026 events on disk so we still get rosters even between splits.
    _live_ids   = _live_event_ids_by_date()
    _fallback_2026 = ["2026_stage2", "2026_masters_london",
                      "2026_stage1", "2026_masters_santiago", "2026_kickoff"]
    _roster_ids = list(dict.fromkeys(_live_ids + _fallback_2026))
    for _eid in _roster_ids:
        _ep = os.path.join(ROOT, 'data', 'maps', f'{_eid}.csv')
        if os.path.exists(_ep):
            try:
                _pf = pd.read_csv(_ep, usecols=['MatchID', 'Org', 'Player', 'ProfileURL'])
                _player_frames.append(_pf)
            except Exception:
                pass
    if _player_frames:
        _pdf = pd.concat(_player_frames, ignore_index=True).dropna(subset=['Player'])
        _pdf = _pdf.sort_values('MatchID', ascending=False)
        for _org, _grp in _pdf.groupby('Org'):
            _url_map = dict(zip(_grp['Player'], _grp['ProfileURL']))
            _seen = []
            for _p in _grp['Player']:
                if _p not in _seen:
                    _seen.append(_p)
                    if len(_seen) >= 5:
                        break
            _roster_by_org[_org] = [
                {'player': _p, 'url': _url_map.get(_p, ''),
                 'headshot': _hs_cache.get(_url_map.get(_p, ''), '')}
                for _p in _seen
            ]

    teams_list = []
    if snap_data:
        beta       = snap_data.get("beta", 0.3237)
        teams_raw  = snap_data.get("teams", {})
        for org, td in teams_raw.items():
            region   = ORG_REGIONS.get(org, "Unknown")
            maps_d   = td.get("maps", {})
            eligible = [(m, v) for m, v in maps_d.items() if v.get("w", 0) + v.get("l", 0) >= 3]
            all_maps   = sorted(maps_d.items(), key=lambda x: -x[1]["rating"])
            best_maps  = sorted(eligible, key=lambda x: -x[1]["rating"])[:3]
            worst_maps = sorted(eligible, key=lambda x:  x[1]["rating"])[:3]

            # Use live timeline rating if available, else snapshot
            rating = last_checkpoint_ratings.get(org, td.get("overall_rating", 0.0))

            teams_list.append({
                "org":    org,
                "region": region,
                "rating": round(float(rating), 4),
                "w":      td.get("w", 0),
                "l":      td.get("l", 0),
                "all_maps":  [{"map": m, "rating": round(v["rating"], 2),
                               "w": v["w"], "l": v["l"]} for m, v in all_maps],
                "best_maps":  [{"map": m, "rating": round(v["rating"], 2),
                                "w": v["w"], "l": v["l"]} for m, v in best_maps],
                "worst_maps": [{"map": m, "rating": round(v["rating"], 2),
                                "w": v["w"], "l": v["l"]} for m, v in worst_maps],
                "recent_matches": recent_by_org.get(org, []),
                "roster": _roster_by_org.get(org, []),
            })
        # Add any timeline teams not in snapshot (e.g. EMEA teams that missed Santiago)
        snap_orgs = {t["org"] for t in teams_list}
        for org, rating in last_checkpoint_ratings.items():
            if org in snap_orgs or org not in ACTIVE_2026_ORGS:
                continue
            teams_list.append({
                "org":    org,
                "region": ORG_REGIONS.get(org, "Unknown"),
                "rating": round(float(rating), 4),
                "w": 0, "l": 0,
                "all_maps": [], "best_maps": [], "worst_maps": [],
                "recent_matches": recent_by_org.get(org, []),
                "roster": _roster_by_org.get(org, []),
            })
        teams_list = [t for t in teams_list if t["org"] in ACTIVE_2026_ORGS]
        teams_list.sort(key=lambda x: -x["rating"])
        for i, t in enumerate(teams_list):
            t["rank"] = i + 1
        result["leaderboard"] = {
            "teams":       teams_list,
            "beta":        snap_data.get("beta", 0.3237),
            "snapshot":    snap_name,
            "as_of_date":  result.get("as_of_date"),
        }
    elif last_checkpoint_ratings:
        # Fallback: leaderboard from timeline ratings only (no map breakdown)
        for i, (org, rating) in enumerate(
                sorted(((o, r) for o, r in last_checkpoint_ratings.items() if o in ACTIVE_2026_ORGS), key=lambda x: -x[1]), 1):
            teams_list.append({
                "org": org, "region": ORG_REGIONS.get(org, "Unknown"),
                "rating": round(rating, 4), "rank": i,
                "w": 0, "l": 0, "all_maps": [], "best_maps": [], "worst_maps": [],
                "recent_matches": recent_by_org.get(org, []),
                "roster": _roster_by_org.get(org, []),
            })
        result["leaderboard"] = {
            "teams":    teams_list,
            "beta":     0.3237,
            "snapshot": "timeline",
            "as_of_date": result.get("as_of_date"),
        }

    # ── Upcoming matches — compute win probs from timeline ratings ────────────
    upcoming_path = os.path.join(ROOT, "data", "upcoming_matches.json")
    if os.path.exists(upcoming_path):
        try:
            with open(upcoming_path) as f:
                upcoming_raw = json.load(f)
        except Exception:
            upcoming_raw = []
    else:
        upcoming_raw = []

    # Compute series win probs for past matches using the same CV-optimal β
    # the upcoming-card sim, simulator iframe, and renderRecent JS sim use.
    # NLL-fitting β on this timeline used to land near 0.28, which is in the
    # overfit zone — a daily-rolling 2025 backtest of 1,922 leak-free per-map
    # predictions showed 0.17 minimizes Brier (0.240 vs 0.246 at 0.28).
    # Keeping all four prediction surfaces on the same β = same matchup,
    # same probability, every surface.
    if last_checkpoint_ratings and result["chart"]["match_events"]:
        from scipy.special import expit as _expit
        _tl_beta = 0.25

        def _series_wp(p, fmt):
            if fmt == "bo5":
                return p**3 * (1 + 3*(1-p) + 6*(1-p)**2)
            return p**2 * (3 - 2*p)  # bo3

        for _m in upcoming_raw:
            _ra = last_checkpoint_ratings.get(_m.get("org_a", ""), 0.0)
            _rb = last_checkpoint_ratings.get(_m.get("org_b", ""), 0.0)
            _m["rating_a"]   = round(_ra, 3)
            _m["rating_b"]   = round(_rb, 3)
            # NOTE: win_prob_a is intentionally NOT pre-computed for upcoming
            # matches. The frontend's per-map MC veto sim (in renderUpcoming)
            # is the authoritative win prob — same model the simulator uses,
            # so the two surfaces produce matching predictions. Pre-computing
            # here with a LIVE-only sigmoid used a different β and skipped
            # per-map info, giving the upcoming card a different answer than
            # the simulator on the same matchup.

    result["upcoming"] = upcoming_raw

    # ── Past matches — replay each match with 12:01-AM ratings ───────────────
    # All matches on date X use the SAME rating snapshot: the checkpoint from
    # the previous match day (= end of day X-1 = start of day X = 12:01 AM
    # local of day X). This removes a bias where, on days with multiple
    # matches, the model's projection for a 9 PM match would have already
    # absorbed the outcomes of the 5 AM matches that ran earlier the same day.
    past_matches = []
    if last_checkpoint_ratings and result["chart"]["match_events"]:
        from datetime import datetime as _dt, timedelta as _td
        try:
            from MoreTestingMaybeFiles import ALL_EVENTS as _ALL_EVENTS_PAST
            _event_label_by_id = {e["id"]: e.get("label", e["id"]) for e in _ALL_EVENTS_PAST}
        except Exception:
            _event_label_by_id = {}

        _as_of_str = result.get("as_of_date")
        try:
            _as_of_dt = _dt.strptime(_as_of_str, "%Y-%m-%d") if _as_of_str else _dt.utcnow()
        except Exception:
            _as_of_dt = _dt.utcnow()
        _cutoff = _as_of_dt - _td(days=7)

        # Build a "morning-of" rating lookup keyed by match date.  Checkpoints
        # are sorted ascending by date; checkpoint(X) represents ratings at
        # the END of day X.  So for a match on date X, the unbiased pre-match
        # snapshot is the most recent checkpoint with date < X.
        _cps_sorted = result["chart"]["checkpoints"]  # already sorted ascending
        _morning_cache = {}
        def _morning_ratings_for(date_str):
            if date_str in _morning_cache:
                return _morning_cache[date_str]
            _best = {}
            for _cp in _cps_sorted:
                if _cp.get("date", "") < date_str:
                    _best = _cp.get("ratings", {})
                else:
                    break
            _morning_cache[date_str] = _best
            return _best

        for _me in result["chart"]["match_events"]:
            try:
                _md = _dt.strptime(_me["date"], "%Y-%m-%d")
            except Exception:
                continue
            if _md < _cutoff or _md > _as_of_dt:
                continue
            _winner = _me.get("winner", "")
            _loser  = _me.get("loser", "")
            if not _winner or not _loser:
                continue

            # Unbiased "12:01 AM of match day" ratings — same for every match
            # on the same date, regardless of earlier-same-day results.
            _morning = _morning_ratings_for(_me["date"])
            _r_win  = _morning.get(_winner, _me.get("winner_before", 0.0))
            _r_lose = _morning.get(_loser,  _me.get("loser_before",  0.0))

            _org_a, _org_b = sorted([_winner, _loser])
            if _org_a == _winner:
                _ra_p, _rb_p  = _r_win, _r_lose
                _actual_winner = "a"
            else:
                _ra_p, _rb_p  = _r_lose, _r_win
                _actual_winner = "b"

            _ss = str(_me.get("series_score", "")).strip()
            _ws, _ls = "0", "0"
            if "-" in _ss:
                _ws, _ls = _ss.split("-", 1)
            try:
                _first = int(_ws)
            except Exception:
                _first = 2
            _fmt = "bo5" if _first >= 3 else "bo3"
            # Display score in org_a / org_b order
            if _org_a == _winner:
                _disp_score = f"{_ws}-{_ls}"
            else:
                _disp_score = f"{_ls}-{_ws}"

            _p_map = float(_expit(_tl_beta * (_ra_p - _rb_p)))
            _p_series = _series_wp(_p_map, _fmt)

            _region = ORG_REGIONS.get(_winner, ORG_REGIONS.get(_loser, "Unknown"))
            _evt_label = _event_label_by_id.get(_me.get("event_id", ""), _me.get("event_id", ""))

            past_matches.append({
                "match_id":    _me.get("match_id"),
                "org_a":       _org_a,
                "org_b":       _org_b,
                "team_a":      _org_a,
                "team_b":      _org_b,
                "date":        _me.get("date"),
                "region":      _region,
                "event":       _evt_label,
                "event_id":    _me.get("event_id", ""),
                "format":      _fmt,
                "rating_a":    round(_ra_p, 3),
                "rating_b":    round(_rb_p, 3),
                "win_prob_a":  round(_p_series, 3),
                "win_prob_b":  round(1.0 - _p_series, 3),
                "actual_winner": _actual_winner,
                "actual_score":  _disp_score,
                "maps_played":   _me.get("maps", []),
            })

        past_matches.sort(key=lambda x: x["date"] or "", reverse=True)

    # Per-region pool — Stage 1 runs three regional leagues, each with its own
    # 7-map pool. Derive each pool from the maps played in past-7-day matches
    # within that region, capped at 7 maps by play count (drops anything
    # borderline that snuck in but isn't really in pool).
    event_pools = {}        # by event_id (combined, kept as fallback)
    region_pools = {}       # by region (the real per-match pool source)
    region_event_pools = {} # by f"{event_id}:{region}"

    def _top7(name_counts):
        items = sorted(name_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return [n for n, _c in items[:7]]

    for _eid in sorted(set(m.get("event_id", "") for m in past_matches)):
        if not _eid:
            continue
        _all_counts = {}
        for _pm in past_matches:
            if _pm.get("event_id") != _eid:
                continue
            for _mp in _pm.get("maps_played", []):
                _name = (_mp or {}).get("map")
                if _name:
                    _all_counts[_name] = _all_counts.get(_name, 0) + 1
        if _all_counts:
            event_pools[_eid] = _top7(_all_counts)

    for _rg in sorted(set(m.get("region", "") for m in past_matches)):
        if not _rg:
            continue
        _counts = {}
        for _pm in past_matches:
            if _pm.get("region") != _rg:
                continue
            for _mp in _pm.get("maps_played", []):
                _name = (_mp or {}).get("map")
                if _name:
                    _counts[_name] = _counts.get(_name, 0) + 1
        if _counts:
            region_pools[_rg] = _top7(_counts)

    for _pm in past_matches:
        _key = f"{_pm.get('event_id','')}:{_pm.get('region','')}"
        if _key in region_event_pools:
            continue
        _counts = {}
        for _pm2 in past_matches:
            if _pm2.get("event_id") != _pm.get("event_id") or _pm2.get("region") != _pm.get("region"):
                continue
            for _mp in _pm2.get("maps_played", []):
                _name = (_mp or {}).get("map")
                if _name:
                    _counts[_name] = _counts.get(_name, 0) + 1
        if _counts:
            region_event_pools[_key] = _top7(_counts)

    result["past_matches"]       = past_matches
    result["past_event_pools"]   = event_pools
    result["past_region_pools"]  = region_pools
    result["past_region_event_pools"] = region_event_pools

    # ── Veto simulation data for upcoming predictions ─────────────────────────
    veto = get_veto_model()
    computed_pools   = _build_computed_pools()
    live_map_stats   = _build_live_map_stats()
    # Derive the live current pool — walk live events by recency, then fall back
    # to the standard 2026 ladder if nothing live has enough data yet.
    _live_pool = None
    _pool_candidates = list(dict.fromkeys(
        _live_event_ids_by_date() +
        ["2026_stage2", "2026_masters_london",
         "2026_stage1", "2026_masters_santiago", "2026_kickoff"]
    ))
    for _eid in _pool_candidates:
        _recs = _load_event_map_records([_eid])
        _p = _detect_pool(_recs)
        if len(_p) >= 7:
            _live_pool = _p
            break
    result["veto_model"]  = {
        "teams":           veto.get("teams", {}),
        "snap_pools":      veto.get("snap_pools", {}),
        "computed_pools":  computed_pools,
        "current_pool":    _live_pool or [],
        "live_map_stats":  live_map_stats,
    }
    result["org_regions"] = ORG_REGIONS
    result["snap_teams"]  = snap_data.get("teams", {}) if snap_data else {}
    result["snap_beta"]   = snap_data.get("beta", 0.3237) if snap_data else 0.3237
    result["snap_key"]    = snap_name or "after_stage1"
    if os.path.exists(_MAP_RATINGS_PATH):
        try:
            with open(_MAP_RATINGS_PATH) as _mrf:
                _mrd = json.load(_mrf)
            result["intl_calib"] = _mrd.get("intl_calib", {})
        except Exception:
            result["intl_calib"] = {}
    else:
        result["intl_calib"] = {}

    return result


def _mhub_event_for_date(date_str):
    """Return the event label whose date range contains date_str (YYYY-MM-DD)."""
    if not date_str:
        return None
    for band in MHUB_EVENT_BANDS:
        if band["start"] <= date_str <= band["end"]:
            return band["label"]
    return None


def _mhub_scrape_progress():
    """Read build progress from RefreshLiveData.py's progress file."""
    try:
        if os.path.exists(_MHUB_PROGRESS_FILE):
            with open(_MHUB_PROGRESS_FILE) as f:
                p = json.load(f)
            if _time_mod.time() - p.get("ts", 0) < 1800:
                return p
    except Exception:
        pass
    return {"phase": "init", "pct": 3, "message": "Initializing…"}


def _mhub_write_progress_error(message, detail=""):
    """Write a synthetic 'error' progress record so the UI can surface what broke
    when the subprocess itself never gets to write its own progress."""
    try:
        payload = {
            "phase":   "error",
            "pct":     100,
            "message": message,
            "log":     [message] + ([detail] if detail else []),
            "errors":  [detail] if detail else [],
            "ts":      _time_mod.time(),
        }
        with open(_MHUB_PROGRESS_FILE, "w") as f:
            json.dump(payload, f)
    except Exception as e:
        print(f"[mhub] could not write progress error: {e}")


def _mhub_recent_progress_age():
    """Seconds since the on-disk progress file was last written; None if missing."""
    try:
        if os.path.exists(_MHUB_PROGRESS_FILE):
            with open(_MHUB_PROGRESS_FILE) as f:
                p = json.load(f)
            ts = float(p.get("ts", 0) or 0)
            if ts > 0:
                return _time_mod.time() - ts
    except Exception:
        pass
    return None


def _mhub_trigger_build(force=False):
    """Kick off the full RefreshLiveData pipeline on page load.

    Throttled three ways so multi-worker gunicorn / rapid polling can't
    spiral into a re-spawn loop:
      1. In-process `_mhub_build_running` flag (single-worker protection).
      2. Per-worker 2-min cooldown via `_mhub_last_trigger` (rapid-poll).
      3. Cross-worker check of the on-disk progress file age — if another
         worker wrote progress within the last 30 s, skip.

    `force=True` is for the manual /modern/refresh endpoint only."""
    global _mhub_build_running, _mhub_last_trigger
    now = _time_mod.time()
    with _mhub_cache_lock:
        if _mhub_build_running:
            return
        if not force and (now - _mhub_last_trigger) < _MHUB_TRIGGER_COOLDOWN:
            return
        if not force:
            age = _mhub_recent_progress_age()
            if age is not None and age < 30:
                # Another worker is actively scraping; let it finish.
                return
        _mhub_build_running = True
        _mhub_last_trigger  = now

    def _run():
        global _mhub_build_running
        try:
            import subprocess as _sp
            import sys as _sys
            script = os.path.join(ROOT, "scrapers", "RefreshLiveData.py")
            if not os.path.exists(script):
                _mhub_write_progress_error("RefreshLiveData.py not found",
                                           f"missing at {script}")
                return
            try:
                with open(_MHUB_STDERR_FILE, "w") as log:
                    cp = _sp.run(
                        [_sys.executable, script],
                        cwd=ROOT,
                        stdout=log, stderr=_sp.STDOUT,
                        start_new_session=True,  # detach so worker recycling doesn't kill it
                        timeout=1800,            # 30-min hard ceiling
                    )
                if cp.returncode != 0:
                    tail = ""
                    try:
                        with open(_MHUB_STDERR_FILE) as f:
                            tail = f.read()[-600:]
                    except Exception:
                        pass
                    _mhub_write_progress_error(
                        f"RefreshLiveData exited {cp.returncode}", tail)
            except FileNotFoundError as e:
                _mhub_write_progress_error("Python interpreter not found",
                                           f"{_sys.executable}: {e}")
            except _sp.TimeoutExpired:
                _mhub_write_progress_error("RefreshLiveData timed out after 30m",
                                           "consider running scrapers offline and committing data")
        except Exception as e:
            print(f"[mhub] RefreshLiveData failed: {e}")
            _mhub_write_progress_error("RefreshLiveData crashed in launcher", str(e))
        finally:
            _mhub_build_running = False
            # Only invalidate the cache when the subprocess actually finished —
            # if it bailed because another worker held the lock, we shouldn't
            # invalidate (the OTHER worker is the one doing work and will write
            # phase=done).  Cache invalidation here used to cause the rapid
            # re-trigger spiral on Render's multi-worker gunicorn.
            try:
                age = _mhub_recent_progress_age()
                with open(_MHUB_PROGRESS_FILE) as _pf:
                    _pd = json.load(_pf)
                if _pd.get("phase") in ("done", "error") and (age or 999) < 60:
                    with _mhub_cache_lock:
                        _mhub_cache["ts"] = 0.0
            except Exception:
                pass

    _th.Thread(target=_run, daemon=True).start()


def _mhub_get():
    """Return cached modern-hub data; short TTL when building for live progress.

    Cache is also invalidated whenever any source file (rating_timeline.json,
    upcoming_matches.json) has a newer mtime than the cache timestamp — that's
    how a scrape completed by ONE gunicorn worker becomes visible to all the
    OTHERS without each needing to re-trigger.  Without this, a worker that
    cached a 'ready' payload before the scrape completed would keep serving
    stale data for up to 30 min."""
    now = _time_mod.time()
    with _mhub_cache_lock:
        cached   = _mhub_cache["data"]
        cache_ts = _mhub_cache["ts"]
        # Cross-worker invalidation: if any source file was touched after we
        # last cached, drop the cache and re-read.
        if cached is not None:
            try:
                for p in (_RATING_TIMELINE_PATH,
                          os.path.join(ROOT, "data", "upcoming_matches.json"),
                          _MAP_RATINGS_PATH):
                    if os.path.exists(p) and os.path.getmtime(p) > cache_ts:
                        cached = None
                        _mhub_cache["data"] = None
                        break
            except OSError:
                pass
        building = bool(cached and cached.get("status") == "building")
        # 3s TTL while building (live progress), 5s when idle-but-running, else 30min
        ttl = 3 if building else (5 if _mhub_build_running else _MHUB_TTL)
        if cached is not None and (now - cache_ts) < ttl:
            return cached

    data = _mhub_load()

    # Only spawn a subprocess when _mhub_load explicitly asked for one.  An
    # in-flight scrape from another worker still reports status=building
    # (so the frontend keeps polling) but needs_trigger=False (so we don't
    # spam new subprocesses).
    if data.get("status") == "building" and data.get("needs_trigger", True):
        _mhub_trigger_build()
    # Strip the internal flag — frontend doesn't need it.
    data.pop("needs_trigger", None)

    # Always attach progress so frontend can show log even on "ready"
    data["progress"] = _mhub_scrape_progress()
    data["as_of_event"] = _mhub_event_for_date(data.get("as_of_date"))

    with _mhub_cache_lock:
        _mhub_cache["data"] = data
        _mhub_cache["ts"]   = now

    return data


MAPELO_MODERN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Modern VCT Hub — Bobo.GG</title>
<link rel="icon" type="image/svg+xml" href="/favicon.svg">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3.0.0/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
<!-- v2 -->
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%}
body{background:#fdf6f0;font-family:'DM Sans',sans-serif;color:#000;min-height:100vh}
body::before{content:'';position:fixed;inset:0;pointer-events:none;z-index:0;background:radial-gradient(ellipse 60% 50% at 10% 10%,#f4b8c155 0%,transparent 70%),radial-gradient(ellipse 50% 60% at 90% 20%,#b8d8f455 0%,transparent 70%),radial-gradient(ellipse 55% 45% at 15% 85%,#b8e8d455 0%,transparent 70%),radial-gradient(ellipse 60% 50% at 85% 80%,#d4b8f455 0%,transparent 70%)}
body::after{content:'';position:fixed;inset:-50%;pointer-events:none;z-index:0;background:radial-gradient(ellipse 60% 50% at 60% 55%,#c4a0f099 0%,transparent 55%),radial-gradient(ellipse 50% 60% at 38% 42%,#d4a97477 0%,transparent 55%);animation:purpleFloat 12s ease-in-out infinite alternate}
@keyframes purpleFloat{0%{transform:translate(0,0) scale(1)}33%{transform:translate(10%,-9%) scale(1.14)}66%{transform:translate(-9%,12%) scale(.9)}100%{transform:translate(7%,5%) scale(1.1)}}

.top-nav{padding:24px 32px 0;display:flex;align-items:center;gap:16px;position:relative;z-index:1}
.home-logo{display:block;height:72px;width:auto;opacity:.85;transition:opacity .2s}
.home-logo:hover{opacity:1}
.back-btn{display:inline-flex;align-items:center;gap:6px;font-family:'DM Sans',sans-serif;font-size:.8rem;font-weight:600;color:#7c3aed;text-decoration:none;padding:6px 14px;border-radius:99px;border:1.5px solid rgba(124,58,237,.25);background:rgba(124,58,237,.06);transition:background .18s,border-color .18s,color .18s;white-space:nowrap}
.back-btn:hover{background:rgba(124,58,237,.12);border-color:rgba(124,58,237,.5);color:#5b21b6}
.back-btn svg{flex-shrink:0}

.hub-main{padding:20px 0 60px;width:100%;position:relative;z-index:1}
.hub-header{text-align:center;margin-bottom:20px}
.hub-title{font-family:'Syne',sans-serif;font-size:clamp(1.8rem,5vw,3rem);font-weight:800;letter-spacing:-.03em;color:#000;line-height:1;min-height:1.2em;transition:opacity .2s}
.type-cursor{opacity:1;animation:blink .55s step-end infinite}
.dot-seed{font-size:2rem}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}
.hub-sub{color:#444;font-size:.9rem;margin-top:6px;transition:opacity .5s}

.tab-bar{display:flex;gap:8px;justify-content:center;margin-bottom:16px;transition:opacity .5s}
.tab{padding:9px 28px;border-radius:100px;border:2px solid #c8a8e8;background:transparent;color:#444;font-family:'DM Sans',sans-serif;font-size:.88rem;font-weight:500;cursor:pointer;transition:all .2s}
.tab.active{background:#3d1a6e;border-color:#3d1a6e;color:#fff}
.tab:hover:not(.active){border-color:#9c6ec8;color:#000}

.panels-outer{overflow:hidden}
/* Slide curve = ease-out-quint. Snappier finish than the symmetric ease,
   so the panel "lands" without that mid-slide hesitation that read as
   a stutter. transform-only animation runs on the compositor. */
.panel-track{display:flex;width:400%;transition:transform .55s cubic-bezier(.22,1,.36,1);will-change:transform;transform:translate3d(0,0,0);backface-visibility:hidden}
.panel-track.show-b{transform:translate3d(-25%,0,0)}
.panel-track.show-c{transform:translate3d(-50%,0,0)}
.panel-track.show-d{transform:translate3d(-75%,0,0)}
/* contain: isolates each panel's layout/paint from its neighbors so the
   simulator iframe (2400px tall) can't trigger a layout recalc of the
   sibling panels during the slide. */
.panel{width:25%;min-width:0;contain:layout paint style}

.region-pills{display:flex;gap:8px;margin-bottom:14px;flex-wrap:wrap;transition:opacity .3s,max-height .5s cubic-bezier(.55,.06,.36,.98),margin-bottom .5s cubic-bezier(.55,.06,.36,.98);justify-content:center;padding:0 24px;overflow:hidden;max-height:60px}
.region-pills.hidden-panel{opacity:0 !important;max-height:0;margin-bottom:0;pointer-events:none;transition:opacity .25s,max-height .45s cubic-bezier(.55,.06,.36,.98) .15s,margin-bottom .45s cubic-bezier(.55,.06,.36,.98) .15s}
.region-pills .pill{transition:transform .35s cubic-bezier(.34,1.56,.64,1),opacity .3s ease,background .15s,border-color .15s,color .15s}
.region-pills.hidden-panel .pill{transform:translateY(-14px) scale(.85);opacity:0}
.region-pills.hidden-panel .pill:nth-child(1){transition-delay:0s}
.region-pills.hidden-panel .pill:nth-child(2){transition-delay:.04s}
.region-pills.hidden-panel .pill:nth-child(3){transition-delay:.08s}
.region-pills.hidden-panel .pill:nth-child(4){transition-delay:.12s}
.region-pills.hidden-panel .pill:nth-child(5){transition-delay:.16s}
.pill{padding:5px 16px;border-radius:100px;border:1.5px solid #c8a8e8;background:transparent;color:#444;font-size:.8rem;font-family:'DM Sans',sans-serif;font-weight:500;cursor:pointer;transition:all .15s}
.pill.active{background:#3d1a6e;border-color:#3d1a6e;color:#fff}
.pill:hover:not(.active){border-color:#9c6ec8}

/* Progress */
.progress-card{background:#1a0a2e;border-radius:20px;padding:44px 48px;margin:0 auto 18px;max-width:560px;text-align:center}
.progress-label{color:rgba(232,213,245,.95);font-family:'Syne',sans-serif;font-weight:800;font-size:1.25rem;margin-bottom:6px;letter-spacing:.01em}
.progress-msg{color:rgba(232,213,245,.55);font-size:.82rem;margin-bottom:28px;font-variant-numeric:tabular-nums}
.progress-track{height:10px;background:rgba(255,255,255,.08);border-radius:6px;overflow:hidden;margin-bottom:10px;position:relative}
.progress-fill{height:100%;border-radius:6px;background:linear-gradient(90deg,#5b21b6,#7c3aed,#a78bfa,#c4b5fd);background-size:200% 100%;transition:width .6s cubic-bezier(.4,0,.2,1);width:0%;animation:progressShimmer 1.8s linear infinite}
@keyframes progressShimmer{0%{background-position:200% 0}100%{background-position:-200% 0}}
.progress-pct{color:rgba(232,213,245,.5);font-size:.75rem;font-variant-numeric:tabular-nums;margin-bottom:0}
@keyframes fillDone{0%{background:linear-gradient(90deg,#5b21b6,#7c3aed,#a78bfa,#c4b5fd)}50%{background:#fff;box-shadow:0 0 18px 8px rgba(255,255,255,.7)}100%{background:#e9d5ff;box-shadow:0 0 6px 2px rgba(255,255,255,.15)}}
.progress-fill.done{animation:fillDone .45s ease forwards!important;width:100%!important;transition:none!important}
@keyframes cardExit{
  0%  {opacity:1;transform:translateY(0);filter:none}
  100%{opacity:0;transform:translateY(48px);filter:blur(4px)}
}
.progress-card.exiting{animation:cardExit .55s cubic-bezier(.4,0,1,1) forwards;pointer-events:none}
@keyframes chartEnter{from{transform:translateX(-100vw)}to{transform:none}}
.chart-card.entering{animation:chartEnter 2.4s cubic-bezier(.16,1,.3,1) forwards}
#progressLog{margin-top:20px;text-align:center;max-height:140px;overflow:hidden;display:flex;flex-direction:column;gap:3px;border-top:1px solid rgba(167,139,250,.12);padding-top:14px}
.plog-entry{font-size:.78rem;color:rgba(167,139,250,.75);padding:1px 0;font-family:'DM Sans',sans-serif;line-height:1.5;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;opacity:.5}
.plog-entry:last-child{opacity:1;color:rgba(200,180,255,.95)}
.plog-entry.new{animation:plog-in .3s ease}
@keyframes plog-in{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}

/* Chart card */
.chart-hint{font-size:.72rem;color:rgba(0,0,0,.38);text-align:center;padding:6px 0 18px;letter-spacing:.01em}
.chart-hint kbd{display:inline-block;font-family:'DM Sans',sans-serif;font-size:.68rem;font-weight:700;background:rgba(0,0,0,.07);border-radius:4px;padding:1px 5px;margin:0 1px}
.chart-card{background:#fff;border-radius:16px;padding:12px 0 8px;margin:0 auto 18px;position:relative;max-width:85%}
.chart-header{display:flex;flex-direction:column;align-items:stretch;margin-bottom:10px;gap:6px;padding:0 20px;position:relative}
.chart-header-row{display:flex;justify-content:flex-end;align-items:center;gap:10px}
.chart-title{align-self:center;font-family:'Syne',sans-serif;font-size:1rem;font-weight:800;letter-spacing:-.02em;background:linear-gradient(135deg,#2a1f2d 0%,#7c3aed 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;white-space:nowrap;pointer-events:none}
.chart-asof{color:rgba(0,0,0,.4);font-size:.75rem}
.chart-controls{display:flex;gap:8px;align-items:center;flex-shrink:0}
.chart-btn{padding:5px 14px;border-radius:100px;border:1.5px solid rgba(0,0,0,.15);background:rgba(0,0,0,.03);color:rgba(0,0,0,.55);font-size:.75rem;font-family:'DM Sans',sans-serif;font-weight:500;cursor:pointer;transition:all .2s;white-space:nowrap}
.chart-btn:hover{border-color:rgba(0,0,0,.4);color:#000;background:rgba(0,0,0,.06)}
.chart-btn.active{border-color:#7c3aed;background:#7c3aed;color:#fff}
.chart-wrap{position:relative;height:650px;user-select:none}
#benpomChart{cursor:default}

/* Dot hover tooltip */
#dotTooltip{position:absolute;z-index:20;pointer-events:none;min-width:280px;max-width:380px;background:#1a0938;border:1px solid rgba(167,139,250,.28);border-radius:16px;padding:20px 24px;box-shadow:0 16px 60px rgba(0,0,0,.7);opacity:0;transform:translateY(8px);transition:opacity .18s ease,transform .18s ease}
#dotTooltip.visible{opacity:1;transform:translateY(0)}
#dotTooltip .popup-inner{text-align:center}
#dotTooltip .popup-event-label{font-size:.65rem;font-weight:600;color:rgba(167,139,250,.5);text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px}
#dotTooltip .popup-teams{display:flex;align-items:center;justify-content:center;gap:16px;margin-bottom:8px}
#dotTooltip .popup-team-block{display:flex;flex-direction:column;align-items:center;gap:5px;min-width:60px}
#dotTooltip .popup-logo{width:44px;height:44px;object-fit:contain}
#dotTooltip .popup-team-name{font-size:.7rem;color:rgba(232,213,245,.6);font-weight:500}
#dotTooltip .popup-score-block{display:flex;flex-direction:column;align-items:center;gap:3px}
#dotTooltip .popup-score{font-size:1.9rem;font-weight:800;font-family:'Syne',sans-serif;line-height:1}
#dotTooltip .popup-score.w{color:#4ade80}#dotTooltip .popup-score.l{color:#f87171}
#dotTooltip .popup-vs-label{font-size:.65rem;color:rgba(232,213,245,.3)}
#dotTooltip .popup-date{color:rgba(232,213,245,.3);font-size:.68rem;margin-bottom:4px}
#dotTooltip .popup-delta{font-size:.85rem;font-weight:600;margin-bottom:14px}
#dotTooltip .popup-delta.pos{color:#4ade80}#dotTooltip .popup-delta.neg{color:#f87171}
#dotTooltip .popup-maps-table{width:100%;border-collapse:collapse;margin-top:2px}
#dotTooltip .popup-maps-table th{font-size:.6rem;font-weight:600;color:rgba(167,139,250,.5);text-transform:uppercase;letter-spacing:.07em;padding:0 6px 6px;text-align:center}
#dotTooltip .popup-maps-table th:first-child{text-align:left}
#dotTooltip .popup-maps-table th:last-child{text-align:right}
#dotTooltip .popup-map-score{text-align:center}
#dotTooltip .popup-maps-table td{padding:5px 6px;font-size:.78rem;color:rgba(232,213,245,.8);border-top:1px solid rgba(255,255,255,.06)}
#dotTooltip .popup-map-name{font-weight:500;color:#e8d5f5}
#dotTooltip .popup-map-score{font-variant-numeric:tabular-nums;font-weight:600}
#dotTooltip .popup-map-score.w{color:#4ade80}#dotTooltip .popup-map-score.l{color:#f87171}
#dotTooltip .popup-map-diff{text-align:right;font-size:.7rem;color:rgba(232,213,245,.4)}

/* Popup */
.match-popup{position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);background:#1a0938;border:1px solid rgba(167,139,250,.25);border-radius:20px;padding:32px 36px;z-index:200;min-width:360px;max-width:480px;width:90vw;box-shadow:0 24px 80px rgba(0,0,0,.75)}
.match-popup.hidden{display:none}
.popup-overlay{position:fixed;inset:0;background:rgba(0,0,0,.55);z-index:199;backdrop-filter:blur(2px)}
.popup-overlay.hidden{display:none}
.popup-close{position:absolute;top:14px;right:18px;background:none;border:none;color:rgba(232,213,245,.4);font-size:1.5rem;cursor:pointer;line-height:1;transition:color .15s}
.popup-close:hover{color:#e8d5f5}
.popup-inner{text-align:center}
.popup-event-label{font-size:.7rem;font-weight:600;color:rgba(167,139,250,.5);text-transform:uppercase;letter-spacing:.08em;margin-bottom:14px}
.popup-teams{display:flex;align-items:center;justify-content:center;gap:22px;margin-bottom:10px}
.popup-team-block{display:flex;flex-direction:column;align-items:center;gap:7px;min-width:80px}
.popup-logo{width:60px;height:60px;object-fit:contain}
.popup-team-name{font-size:.75rem;color:rgba(232,213,245,.6);font-weight:500}
.popup-score-block{display:flex;flex-direction:column;align-items:center;gap:4px}
.popup-score{font-size:2.4rem;font-weight:800;font-family:'Syne',sans-serif;line-height:1}
.popup-score.w{color:#4ade80}.popup-score.l{color:#f87171}
.popup-vs-label{font-size:.7rem;color:rgba(232,213,245,.3)}
.popup-date{color:rgba(232,213,245,.3);font-size:.72rem;margin-bottom:6px}
.popup-delta{font-size:.95rem;font-weight:600;margin-bottom:20px}
.popup-delta.pos{color:#4ade80}.popup-delta.neg{color:#f87171}
.popup-maps-table{width:100%;border-collapse:collapse;margin-top:4px}
.popup-maps-table th{font-size:.65rem;font-weight:600;color:rgba(167,139,250,.5);text-transform:uppercase;letter-spacing:.07em;padding:0 8px 8px;text-align:left}
.popup-maps-table th:last-child{text-align:right}
.popup-maps-table td{padding:7px 8px;font-size:.82rem;color:rgba(232,213,245,.8);border-top:1px solid rgba(255,255,255,.06)}
.popup-map-name{font-weight:500;color:#e8d5f5}
.popup-map-score{font-variant-numeric:tabular-nums;font-weight:600}
.popup-map-score.w{color:#4ade80}.popup-map-score.l{color:#f87171}
.popup-map-diff{text-align:right;font-size:.75rem;color:rgba(232,213,245,.4)}

/* Chart + leaderboard full-width layout */
#chartSection{padding:0 48px}
.lb-card-wrap{padding:0 24px;max-width:780px;margin:0 auto}

/* Leaderboard */
.lb-card{background:rgba(255,255,255,.72);border-radius:16px;overflow:hidden;backdrop-filter:blur(10px)}
.lb-header-row{padding:14px 20px;display:flex;align-items:center;justify-content:center;position:relative;border-bottom:1px solid rgba(61,26,110,.1)}
.lb-title{font-family:'Syne',sans-serif;font-weight:700;font-size:.95rem;color:#000;text-align:center}
.lb-asof{position:absolute;right:20px;top:50%;transform:translateY(-50%);font-size:.7rem;color:#666;text-align:right;max-width:240px}
@keyframes lbRowSlideIn { from { opacity:0; transform:translateX(-60px); } to { opacity:1; transform:translateX(0); } }
.lb-row.slide-in { animation:lbRowSlideIn .55s cubic-bezier(.16,1,.3,1) backwards; }
.lb-col-hdr{display:grid;grid-template-columns:44px 2fr 1fr 1fr 24px;align-items:center;padding:8px 24px;gap:10px;border-bottom:2px solid rgba(61,26,110,.1)}
.lb-col-hdr span{font-family:'Syne',sans-serif;font-size:.62rem;font-weight:800;text-transform:uppercase;letter-spacing:.1em;color:#888;text-align:center}
.lb-row{display:grid;grid-template-columns:44px 2fr 1fr 1fr 24px;align-items:center;padding:13px 24px;cursor:pointer;transition:background .15s;border-bottom:1px solid rgba(61,26,110,.06);gap:10px}
.lb-row:last-child{border-bottom:none}
.lb-row:hover{background:rgba(61,26,110,.05)}
.lb-row.selected{background:rgba(61,26,110,.08)}
.lb-rank{color:#aaa;font-size:.78rem;font-weight:600;text-align:center}
.lb-team{display:flex;align-items:center;justify-content:center;gap:10px}
.lb-team img{width:30px;height:30px;object-fit:contain;flex-shrink:0}
.lb-name{font-weight:700;font-size:.92rem;color:#111}
.lb-rating{font-weight:700;font-size:1rem;text-align:center;justify-self:center;font-variant-numeric:tabular-nums;color:#111}
.lb-region{font-size:.68rem;font-weight:700;padding:3px 10px;border-radius:100px;text-align:center;justify-self:center}
.lb-region.americas{background:rgba(234,88,12,.12);color:#c2410c}
.lb-region.emea{background:rgba(22,163,74,.12);color:#15803d}
.lb-region.pacific{background:rgba(37,99,235,.12);color:#1d4ed8}
.lb-region.cn{background:rgba(219,39,119,.12);color:#be185d}
.lb-chevron{color:#bbb;font-size:.62rem;text-align:center;transition:transform .2s}
.lb-row.selected .lb-chevron{transform:rotate(180deg)}

.lb-detail{border-bottom:1px solid rgba(61,26,110,.07);animation:sd .18s ease}
@keyframes sd{from{opacity:0;transform:translateY(-3px)}to{opacity:1;transform:none}}
@keyframes su{from{opacity:1;max-height:800px}to{opacity:0;max-height:0}}
.lb-detail.closing{animation:su .22s ease forwards;pointer-events:none;overflow:hidden}
.lb-detail-inner{padding:16px 24px 20px}
.lb-sec-label{font-size:.68rem;font-weight:700;color:#555;text-transform:uppercase;letter-spacing:.1em;margin:16px 0 8px}
.lb-sec-label:first-child{margin-top:0}
.lb-match-card{background:rgba(61,26,110,.05);border-radius:10px;padding:10px 14px;margin-bottom:7px}
.lb-match-head{display:flex;align-items:center;gap:10px;margin-bottom:6px}
.lb-mr{font-weight:700;font-size:.82rem;min-width:14px}
.lb-match-card.win .lb-mr{color:#16a34a}.lb-match-card.loss .lb-mr{color:#dc2626}
.lb-mlogo{width:22px;height:22px;object-fit:contain;flex-shrink:0}
.lb-mopp{font-weight:600;font-size:.87rem;flex:1;color:#000}
.lb-mscore{font-weight:700;font-size:.9rem;font-variant-numeric:tabular-nums}
.lb-match-card.win .lb-mscore{color:#16a34a}.lb-match-card.loss .lb-mscore{color:#dc2626}
.lb-mdelta{font-weight:600;font-size:.8rem;font-variant-numeric:tabular-nums}
.lb-match-card.win .lb-mdelta{color:#16a34a}.lb-match-card.loss .lb-mdelta{color:#dc2626}
.lb-mmeta{display:flex;gap:10px;font-size:.7rem;color:#666;margin-bottom:6px}
.lb-mmaps{display:flex;flex-wrap:wrap;gap:5px}
.lb-mmap-chip{font-size:.72rem;padding:3px 8px;border-radius:6px;font-weight:500;font-variant-numeric:tabular-nums}
.lb-mmap-chip.mw{background:rgba(22,163,74,.1);color:#16a34a}
.lb-mmap-chip.ml{background:rgba(220,38,38,.1);color:#dc2626}
.lb-mmap-chip.mn{background:rgba(0,0,0,.06);color:#555}
.lb-maps-table{width:100%;border-collapse:collapse;margin-top:2px}
.lb-maps-table th{font-size:.64rem;font-weight:700;color:#666;text-transform:uppercase;letter-spacing:.07em;padding:0 6px 6px;text-align:left}
.lb-maps-table th:not(:first-child){text-align:right}
.lb-maps-table td{padding:6px 6px;font-size:.8rem;border-top:1px solid rgba(61,26,110,.06)}
.lb-mt-map{color:#000;font-weight:500}
.lb-mt-rat{text-align:right;font-weight:700;font-variant-numeric:tabular-nums}
.lb-mt-rat.pos{color:#16a34a}.lb-mt-rat.neg{color:#dc2626}
.lb-mt-wl{text-align:right;color:#666;font-size:.75rem}
.lb-mt-pct{text-align:right;color:#666;font-size:.74rem}
.lb-empty{padding:44px;text-align:center;color:#666;font-size:.88rem}
.lb-player-row{display:flex;gap:16px;flex-wrap:wrap;justify-content:center;margin-bottom:4px}
.lb-player-card{display:flex;flex-direction:column;align-items:center;gap:6px;width:72px}
.lb-player-hs{width:64px;height:64px;border-radius:50%;object-fit:cover;object-position:top;background:#f0ecf4;flex-shrink:0}
.lb-player-hs-empty{background:#e8e4f0}
.lb-player-name{font-size:.67rem;font-weight:600;text-align:center;color:#333;line-height:1.2;word-break:break-word}
.lb-map-row-click{cursor:pointer}
.lb-map-row-click:hover td{background:rgba(61,26,110,.04)}
.lb-map-chevron{display:inline-block;font-size:.55rem;color:#bbb;transition:transform .2s;margin-left:3px;vertical-align:middle}
.lb-map-row-click.open .lb-map-chevron{transform:rotate(180deg)}
.lb-map-games-tr>td{padding:0!important}
.lb-map-games-wrap{padding:2px 0 6px 4px;animation:sd .15s ease;overflow:hidden}
.lb-map-games-wrap.closing{animation:su .2s ease forwards}
.lb-map-games-tbl{width:100%;border-collapse:collapse}
.lb-mg-inner{display:flex;align-items:center;gap:7px;padding:4px 8px}
.lb-mg-result{font-weight:700;font-size:.76rem;min-width:11px}
.lb-map-game-row.win .lb-mg-result{color:#16a34a}.lb-map-game-row.loss .lb-mg-result{color:#dc2626}
.lb-mg-logo{width:16px;height:16px;object-fit:contain;flex-shrink:0}
.lb-mg-opp{font-size:.78rem;font-weight:600;flex:1;color:#111}
.lb-mg-score{font-size:.78rem;font-weight:700;font-variant-numeric:tabular-nums}
.lb-map-game-row.win .lb-mg-score{color:#16a34a}.lb-map-game-row.loss .lb-mg-score{color:#dc2626}
.lb-mg-diff{font-size:.72rem;font-weight:600;font-variant-numeric:tabular-nums;min-width:28px;text-align:right}
.lb-mg-diff.pos{color:#16a34a}.lb-mg-diff.neg{color:#dc2626}
.lb-mg-meta{font-size:.67rem;color:#888;white-space:nowrap}
.lb-map-no-games{padding:6px 10px;color:#888;font-size:.73rem;font-style:italic}

/* Upcoming */
.upcoming-panel{padding:4px 0 20px}
.upcoming-heading{font-family:'Syne',sans-serif;font-weight:800;font-size:1.3rem;color:#000;margin-bottom:4px;text-align:center}
.upcoming-sub{color:#444;font-size:.83rem;margin-bottom:16px;text-align:center}
.no-upcoming{padding:60px;text-align:center;color:#666;font-size:.88rem}

/* Letter fly-in animation for Upcoming + Recent Matches heading + sub */
.upcoming-heading .fly-char,
.upcoming-sub .fly-char,
.past-heading .fly-char,
.past-sub .fly-char,
.sim-heading .fly-char,
.sim-sub .fly-char{display:inline-block;opacity:0;transform:translateX(60px);transition:transform .55s cubic-bezier(.16,.85,.34,1.02),opacity .45s ease}
.upcoming-heading.flying .fly-char,
.upcoming-sub.flying .fly-char,
.past-heading.flying .fly-char,
.past-sub.flying .fly-char,
.sim-heading.flying .fly-char,
.sim-sub.flying .fly-char{will-change:transform,opacity}
.upcoming-heading.fly-in .fly-char,
.upcoming-sub.fly-in .fly-char,
.past-heading.fly-in .fly-char,
.past-sub.fly-in .fly-char,
.sim-heading.fly-in .fly-char,
.sim-sub.fly-in .fly-char{opacity:1;transform:translateX(0)}
/* Match cards fly in from right with cascade */
.upc-list .upc-card{opacity:0;transform:translateX(80px);transition:transform .5s cubic-bezier(.16,.85,.34,1.02),opacity .4s ease;will-change:transform,opacity}
.upc-list.fly-in .upc-card{opacity:1;transform:translateX(0)}
/* Per-card slide-in for progressive loading (used by renderPast — each match
   card is filled in then revealed once its 20k-sim MC completes). */
.upc-list .upc-card.card-loaded{opacity:1;transform:translateX(0);transition-delay:0ms !important}
/* Drop will-change once animation finishes so we don't pay the compositing
   cost forever (added back by JS for the flight, removed after) */
.upc-list.anim-done .upc-card{will-change:auto}

/* Recent Matches heading (mirror upcoming-heading) */
.past-heading{font-family:'Syne',sans-serif;font-weight:800;font-size:1.3rem;color:#000;margin-bottom:4px;text-align:center}
.past-sub{color:#444;font-size:.83rem;margin-bottom:16px;text-align:center;max-width:560px;margin-left:auto;margin-right:auto}

/* Simulator panel — full historical-matchup tool via iframe */
.sim-iframe{width:100%;min-height:2400px;border:0;background:transparent;display:block}

/* Result strip on past-match cards */
.upc-result-strip{display:flex;align-items:center;justify-content:center;gap:10px;margin-top:8px;padding:6px 10px;border-radius:8px;background:rgba(0,0,0,.04);font-size:.74rem;font-weight:700;letter-spacing:.02em}
.upc-result-strip .upc-result-label{color:#666;font-weight:600;font-size:.66rem;letter-spacing:.08em;text-transform:uppercase}
.upc-result-strip .upc-result-score{font-variant-numeric:tabular-nums;color:#111;font-size:.86rem}
.upc-result-strip .upc-result-winner{color:#fff;background:#16a34a;padding:2px 8px;border-radius:100px;font-size:.66rem;letter-spacing:.05em;text-transform:uppercase}
.upc-result-strip .upc-result-upset{background:#dc2626}
.upc-card .upc-pre-label{font-size:.58rem;color:#888;font-weight:600;letter-spacing:.08em;text-transform:uppercase;text-align:center;margin-top:3px;margin-bottom:1px}

/* Upcoming + Recent match cards — vertical list. Day groups get plenty
   of breathing room so each date reads as its own section. */
.upc-list{display:flex;flex-direction:column;gap:36px;max-width:680px;margin:0 auto}
.upc-day-group{display:flex;flex-direction:column;gap:10px}
/* Mirrors the day-group's own gap. renderPast appends cards into this
   wrapper one at a time (progressive load) — without explicit gap here,
   the day-group's gap doesn't reach the grand-children. */
.upc-day-cards{display:flex;flex-direction:column;gap:10px}
.upc-day-label{font-family:'Syne',sans-serif;font-weight:800;font-size:.85rem;color:#555;text-transform:uppercase;letter-spacing:.08em;margin-top:6px;padding-bottom:8px;border-bottom:1px solid rgba(0,0,0,.1);margin-bottom:2px}
/* No backdrop-filter — at 8% bg opacity the blur is invisible but costs
   a full GPU recompute per card per frame during the cascade animation
   (FPS would drop to ~15 on the 26-card Recent Matches panel). */
.upc-card{border-radius:14px;padding:13px 16px;background:rgba(255,255,255,.55);box-shadow:0 2px 10px rgba(61,26,110,.08);cursor:pointer;user-select:none;transition:box-shadow .15s;border-left:4px solid transparent;contain:layout style}
.upc-card:hover{box-shadow:0 4px 18px rgba(61,26,110,.15)}
.upc-card.rgn-emea{background:rgba(34,197,94,.08);border-left-color:#16a34a}
.upc-card.rgn-americas{background:rgba(249,115,22,.08);border-left-color:#ea580c}
.upc-card.rgn-pacific{background:rgba(59,130,246,.08);border-left-color:#2563eb}
.upc-card.rgn-cn{background:rgba(219,39,119,.08);border-left-color:#db2777}
.upc-header{display:flex;align-items:center;gap:12px}
.upc-team-a,.upc-team-b{display:flex;flex-direction:column;align-items:center;gap:3px;min-width:60px}
.upc-logo{width:36px;height:36px;object-fit:contain}
.upc-org{font-family:'Syne',sans-serif;font-weight:800;font-size:.84rem;color:#000;text-align:center}
.upc-rtg{font-size:.95rem;color:#111;font-weight:800;font-variant-numeric:tabular-nums}
.upc-center{flex:1;text-align:center;padding:0 4px}
.upc-date-event{font-size:.65rem;color:#666;margin-bottom:6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.upc-bar-wrap{height:7px;border-radius:4px;overflow:hidden;display:flex;margin-bottom:4px}
.upc-bar-a{height:100%;background:linear-gradient(90deg,#3d1a6e,#7c3aed);border-radius:4px 0 0 4px;transition:width .3s}
.upc-bar-b{flex:1;height:100%;background:linear-gradient(90deg,#9c6ec8,#c8b8e8);border-radius:0 4px 4px 0}
.upc-pcts{display:flex;justify-content:space-between;font-size:.74rem;font-weight:800}
.upc-pct.fav{color:#000}
.upc-pct.dog{color:#888}
.upc-expand-hint{text-align:center;font-size:.6rem;color:#bbb;margin-top:7px;letter-spacing:.04em}
.upc-card.open .upc-expand-hint{color:#999}

/* Expandable details — use the grid-template-rows 0fr→1fr trick so the
   panel animates to its real content height (no max-height overshoot or
   abrupt finish when content is shorter than the cap). */
.upc-details{display:grid;grid-template-rows:0fr;transition:grid-template-rows .35s cubic-bezier(.22,1,.36,1)}
.upc-details > .upc-details-inner{overflow:hidden;min-height:0}
.upc-card.open .upc-details{grid-template-rows:1fr}
.upc-details-inner{padding-top:0;margin-top:0;border-top:0;transition:padding-top .35s ease,margin-top .35s ease,border-top-color .35s ease}
.upc-card.open .upc-details-inner{padding-top:12px;margin-top:10px;border-top:1px solid rgba(0,0,0,.07)}

/* Map breakdown table */
.upc-section-lbl{font-size:.62rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase;color:#888;margin-bottom:6px}
.upc-map-table{width:100%;border-collapse:collapse;font-size:.72rem;margin-bottom:12px}
.upc-map-table th{font-weight:700;color:#888;font-size:.63rem;text-transform:uppercase;letter-spacing:.05em;padding:3px 6px;text-align:center;border-bottom:1px solid rgba(0,0,0,.08)}
.upc-map-table th:first-child{text-align:left}
.upc-map-table td{padding:4px 6px;text-align:center;border-bottom:1px solid rgba(0,0,0,.04)}
.upc-map-table td:first-child{text-align:left;font-weight:600;color:#111}
.upc-map-td-wp{font-weight:700}
.upc-map-td-wp.fav{color:#1a7a40}
.upc-map-td-wp.dog{color:#b03030}
.upc-map-td-wp.neu{color:#555}
.upc-map-td-veto{font-size:.65rem}

/* Veto sequences in expanded */
.upc-veto-seqs{margin-bottom:12px}
.upc-veto-seq-row{display:flex;align-items:center;gap:4px;flex-wrap:wrap;margin-bottom:5px}
.upc-veto-seq-prob{font-size:.62rem;color:#888;font-weight:700;min-width:28px}

/* Recent form */
.upc-recent-row{display:flex;gap:10px;margin-top:4px}
.upc-recent-col{flex:1;min-width:0}
.upc-recent-col-hdr{font-family:'Syne',sans-serif;font-weight:800;font-size:.72rem;color:#111;margin-bottom:5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.upc-recent-match{display:flex;align-items:center;gap:5px;padding:3px 0;border-bottom:1px solid rgba(0,0,0,.05);font-size:.68rem}
.upc-recent-match:last-child{border-bottom:none}
.upc-recent-result{font-weight:800;font-size:.72rem;min-width:14px}
.upc-recent-result.w{color:#1a7a40}
.upc-recent-result.l{color:#b03030}
.upc-recent-opp{font-weight:600;color:#111;flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.upc-recent-score{color:#555;font-size:.65rem;white-space:nowrap}
.upc-recent-evt{color:#aaa;font-size:.6rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:80px}

/* Upcoming veto sequence (shared) */
.upc-veto-row{display:flex;align-items:center;gap:4px;margin-top:8px;flex-wrap:wrap;font-size:.68rem}
.upc-veto-step{display:flex;flex-direction:column;align-items:center;gap:2px}
.upc-veto-map{font-weight:700;color:#000;font-size:.7rem}
/* Veto step labels (shared with historical predictor) */
.step-lbl{font-size:.52rem;font-weight:800;letter-spacing:.06em;text-transform:uppercase;border-radius:4px;padding:2px 5px;white-space:nowrap}
.step-lbl-banA,.step-lbl-banB{background:#fde8ec;color:#b03050}
.step-lbl-pickA,.step-lbl-pickB{background:#e3f6ea;color:#206040}
.step-lbl-dec{background:#f0ecf4;color:#7a6e7e}
.step-arrow{color:#666;font-weight:700;font-size:.9rem;line-height:1}

.hidden{display:none}
</style>
</head>
<body>
<div class="top-nav">
  <a href="/"><img src="/logo.svg" alt="Home" class="home-logo"></a>
  <a href="/mapelo/" class="back-btn"><svg width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M9 2L4 7l5 5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg> Back to BenPom</a>
</div>

<main class="hub-main">
  <div class="hub-header">
    <h1 class="hub-title" id="hubTitle" style="opacity:0">&middot;</h1>
  </div>

  <div class="tab-bar" id="tabBar" style="opacity:0">
    <button class="tab active" data-panel="a">BenPom Ratings</button>
    <button class="tab" data-panel="b">Upcoming Matches</button>
    <button class="tab" data-panel="c">Recent Matches</button>
    <button class="tab" data-panel="d">Simulator</button>
  </div>

  <div class="region-pills" id="regionPills" style="opacity:0">
    <button class="pill active" data-region="All">All Regions</button>
    <button class="pill" data-region="Americas">Americas</button>
    <button class="pill" data-region="EMEA">EMEA</button>
    <button class="pill" data-region="Pacific">Pacific</button>
    <button class="pill" data-region="CN">China</button>
    <button class="pill" data-region="Top10" id="top10Pill">Top 10 Globally</button>
  </div>

  <div class="panels-outer">
    <div class="panel-track" id="panelTrack">

      <!-- Panel A -->
      <div class="panel" id="panelA">

        <!-- Progress section (shown while building) -->
        <div id="progressSection" class="hidden">
          <div class="progress-card">
            <div class="progress-label">Verifying VCT Data</div>
            <div class="progress-msg" id="progressMsg">Initializing&hellip;</div>
            <div class="progress-track"><div class="progress-fill" id="progressFill"></div></div>
            <div class="progress-pct" id="progressPct">0%</div>
            <div id="progressLog"></div>
          </div>
        </div>

        <!-- Chart section (shown when ready) -->
        <div id="chartSection" class="hidden">
          <div class="chart-card">
            <p class="chart-hint"><kbd>W</kbd> up &nbsp;<kbd>S</kbd> down &nbsp;<kbd>X</kbd> clear selection</p>
            <div class="chart-header">
              <span class="chart-title">BenPom Rating &mdash; 2026 Season</span>
              <div class="chart-header-row">
                <span class="chart-asof" id="chartAsOf"></span>
                <div class="chart-controls">
                  <button class="chart-btn" id="replayBtn" onclick="replayChart()" title="Replay season animation">&#8635; Replay</button>
                  <button class="chart-btn" id="zoomBtn" onclick="toggleZoom()" title="Zoom to current split">&#x2316; Zoom Split</button>
                  <button class="chart-btn" id="resetZoomBtn" onclick="resetZoom()" title="Reset zoom">&#x2715; Reset</button>
                </div>
              </div>
            </div>
            <div class="chart-wrap" id="chartWrap">
              <canvas id="benpomChart"></canvas>
              <div id="dotTooltip"><div class="popup-inner" id="dotTooltipContent"></div></div>
            </div>
          </div>
        </div>

        <!-- Leaderboard -->
        <div class="lb-card-wrap">
          <div class="lb-card hidden" id="lbCard">
            <div class="lb-header-row">
              <span class="lb-title">Current Rankings</span>
              <span class="lb-asof" id="lbAsOf"></span>
            </div>
            <div class="lb-col-hdr">
              <span>#</span><span>Team</span><span>Rating</span><span>Region</span><span></span>
            </div>
            <div id="lbBody"></div>
          </div>
        </div>
      </div>

      <!-- Panel B -->
      <div class="panel" id="panelB">
        <div class="upcoming-panel">
          <div class="upcoming-heading">Upcoming Matches</div>
          <div class="upcoming-sub">Next 7 days across all regions</div>
          <div id="upcomingBody"><div class="no-upcoming">Loading&hellip;</div></div>
        </div>
      </div>

      <!-- Panel C — Recent Matches -->
      <div class="panel" id="panelC">
        <div class="upcoming-panel">
          <div class="past-heading">Recent Matches</div>
          <div class="past-sub">Last 7 days &middot; projected probability uses ratings from the morning before each match</div>
          <div id="pastBody"><div class="no-upcoming">Loading&hellip;</div></div>
        </div>
      </div>

      <!-- Panel D — Match Simulator (embeds the full historical matchup tool,
           with year/snap pickers hidden so both sides are pinned to current) -->
      <div class="panel" id="panelD">
        <!-- loading="eager" so preloadSimulator() actually fetches the iframe
             during idle time; lazy would defer until the panel scrolls into view. -->
        <iframe class="sim-iframe" id="simIframe" src="about:blank" loading="eager"></iframe>
      </div>

    </div><!-- panel-track -->
  </div><!-- panels-outer -->
</main>

<!-- Popup (outside panels-outer so it isn't clipped) -->
<div class="popup-overlay hidden" id="popupOverlay" onclick="closePopup()"></div>
<div class="match-popup hidden" id="matchPopup">
  <button class="popup-close" onclick="closePopup()">&times;</button>
  <div class="popup-inner" id="popupContent"></div>
</div>

<script>
// ── Utilities ────────────────────────────────────────────────────────────────
const sleep  = ms => new Promise(r => setTimeout(r, ms));
const easeOut = t => 1 - Math.pow(1 - t, 2.5);

function showEl(id) {
  const el = document.getElementById(id);
  el.classList.remove('hidden');
  el.style.opacity = '0';
  el.style.transition = 'opacity 0.4s';
  requestAnimationFrame(() => requestAnimationFrame(() => { el.style.opacity = '1'; }));
}
function fadeIn(id, dur) {
  const el = document.getElementById(id);
  el.style.transition = `opacity ${dur||0.4}s`;
  el.style.opacity = '1';
}

// ── Constants ────────────────────────────────────────────────────────────────
const TEAM_COLORS = {
  // Pacific
  PRX:'#ED1C7C', T1:'#E2012D', FS:'#FF6A00', GE:'#1E90FF',
  GEN:'#AA8E4F', NS:'#DC0000', DFM:'#1565C0', RRQ:'#FFA500',
  KRX:'#0B1F4D', TS:'#FFCC00', ZETA:'#000000', VL:'#8C8C8C',
  // Americas
  G2:'#000000', '100T':'#E21F26', LEV:'#00D4D4', NRG:'#FF6B00',
  'KRÜ':'#FF1493', FUR:'#000000', SEN:'#C8102E', MIBR:'#000000',
  LOUD:'#00FF7F', C9:'#00B6E8', EG:'#0073CF', ENVY:'#6A0DAD',
  // EMEA
  VIT:'#FFD100', TH:'#FFD700', FNC:'#FF5900', TL:'#002B5C',
  NAVI:'#F7D417', FUT:'#E10600', KC:'#1B6FE2', GX:'#4FC3F7',
  M8:'#39FF14', BBL:'#D4AF37', EF:'#D4AF37', PCF:'#87CEEB',
  // CN (provisional brand colors — confirm with user)
  EDG:'#E60012', BLG:'#FB7299', TE:'#00B0FF', DRG:'#FFD600',
  ASE:'#FF6F00', AG:'#FF8800', XLG:'#1A1A1A', WOL:'#F5C400',
  FPX:'#E60012', JDG:'#00C853', NOVA:'#7B1FA2', TEC:'#1565C0',
  TYL:'#D32F2F', TYLOO:'#D32F2F',
  // Team Secret (grey — appears as 'Secret' in older data)
  Secret:'#808080', SCRT:'#808080', TSEC:'#808080',
  // Legacy / extras kept for older event data
  DRX:'#c53030', ULF:'#0284c7', TLN:'#0369a1',
};

// Per-team logo size multipliers — some logos render too large inside the circle
const LOGO_SCALES = {
  ZETA: 0.72,
};

// ── Veto simulation helpers ───────────────────────────────────────────────────
// Initialized once hubData is loaded (see showChartAndLeaderboard)
var VETO_HUB   = {teams:{}, snap_pools:{}};
var ORG_REGIONS_HUB = {};
var INTL_HUB   = {};
var SNAP_TEAMS = {};
// β = 0.25 — paired with RD_TRANSFORM=power, RD_POWER=0.35, RD_SCALE=1.25
// in BuildMapRatings.py. Joint sweep found this strict-Pareto improvement
// (Brier -1.07%, LL -0.85%, ECE -3.2%, σ -0.4%, sharpness +0.5%) on the
// 2026 holdout — strictly better than the prior β=0.17 / sqrt config.
// Hardcoded so all four prediction surfaces use the same β.
var SNAP_BETA  = 0.25;
var SNAP_KEY   = 'after_santiago';

var VETO_STEPS_HUB = {
  bo1:[{side:'A',action:'ban'},{side:'B',action:'ban'},{side:'A',action:'ban'},{side:'B',action:'ban'},{side:'A',action:'ban'},{side:'B',action:'ban'}],
  bo3:[{side:'A',action:'ban'},{side:'B',action:'ban'},{side:'A',action:'pick'},{side:'B',action:'pick'},{side:'A',action:'ban'},{side:'B',action:'ban'}],
  bo5:[{side:'A',action:'ban'},{side:'B',action:'ban'},{side:'A',action:'pick'},{side:'B',action:'pick'},{side:'A',action:'pick'},{side:'B',action:'pick'}],
};
var SERIES_THRESH_HUB = {bo1:1, bo3:2, bo5:3};
var ACTION_CLS = {banA:['step-lbl-banA','rv-act-banA'], banB:['step-lbl-banB','rv-act-banB'], pickA:['step-lbl-pickA','rv-act-pickA'], pickB:['step-lbl-pickB','rv-act-pickB'], dec:['step-lbl-dec','rv-act-dec']};

function getGlobalRatingHUB(org, snapKey, domesticRating) {
  var cal = INTL_HUB[snapKey] || {};
  var region = ORG_REGIONS_HUB[org] || '';
  var regOff = (cal.regional_offsets || {})[region] || 0;
  var indBonus = (cal.individual_bonuses || {})[org] || 0;
  return domesticRating + regOff + indBonus;
}
function getActivePoolHUB(snap) {
  var key = '2026_'+snap;
  var cp = (VETO_HUB.computed_pools||{})[key];
  if (cp && cp.length >= 7) return cp;
  return (VETO_HUB.snap_pools||{})[key] || null;
}
function getBanProbsHUB(patt, oppTeam, rem) {
  var scores = {};
  rem.forEach(function(m){
    var rate=(patt&&patt.bans&&patt.bans[m]!=null)?patt.bans[m]:0;
    var oppWin=(oppTeam&&oppTeam.maps&&oppTeam.maps[m])?(oppTeam.maps[m].win_pct||0.5):0.5;
    scores[m]=(rate+0.02)*(0.75+oppWin);
  });
  var tot=rem.reduce(function(s,m){return s+scores[m];},0);
  if(tot===0) rem.forEach(function(m){scores[m]=1/rem.length;});
  else rem.forEach(function(m){scores[m]/=tot;});
  return scores;
}
function getPickProbsHUB(patt, rem) {
  var scores={};
  rem.forEach(function(m){scores[m]=(patt&&patt.picks&&patt.picks[m]!=null)?(patt.picks[m]+0.02):(1/rem.length);});
  var tot=rem.reduce(function(s,m){return s+scores[m];},0);
  if(tot===0) rem.forEach(function(m){scores[m]=1/rem.length;});
  else rem.forEach(function(m){scores[m]/=tot;});
  return scores;
}
function sampleFromHUB(probs) {
  var r=Math.random(),cum=0,keys=Object.keys(probs);
  for(var i=0;i<keys.length;i++){cum+=probs[keys[i]];if(r<=cum) return keys[i];}
  return keys[keys.length-1];
}

// Deterministic PRNG seeded per matchup (mulberry32). Used by the upcoming /
// past card sims so the win prob shown for "G2 vs 100T" is identical on every
// page load — eliminates the visual jitter from Math.random()'s reseeding
// while keeping the MC unbiased across different matchups.
function _seededRng(seed) {
  var s = seed >>> 0;
  return function() {
    s = (s + 0x6D2B79F5) | 0;
    var t = Math.imul(s ^ s >>> 15, 1 | s);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
function _matchSeed() {
  var h = 2166136261;
  for (var i = 0; i < arguments.length; i++) {
    var s = String(arguments[i]);
    for (var j = 0; j < s.length; j++) {
      h ^= s.charCodeAt(j);
      h = Math.imul(h, 16777619);
    }
  }
  return h >>> 0;
}
// Wrap a block of MC code with a seeded Math.random. Anything inside fn —
// including simulateVetoHUB / sampleFromHUB — consumes deterministic numbers.
// Restored in a finally so a sim error can't leak the override.
function _withSeededRand(seed, fn) {
  var orig = Math.random;
  Math.random = _seededRng(seed);
  try { return fn(); } finally { Math.random = orig; }
}
function simulateVetoHUB(tA, tB, orgA, orgB, pool, snap, fmt) {
  var pA=((VETO_HUB.teams||{})['2026_'+snap]||{})[orgA]||null;
  var pB=((VETO_HUB.teams||{})['2026_'+snap]||{})[orgB]||null;
  var rem=pool.slice(), fate={};
  (VETO_STEPS_HUB[fmt]||VETO_STEPS_HUB.bo3).forEach(function(step){
    var patt=step.side==='A'?pA:pB, oppT=step.side==='A'?tB:tA;
    var m=step.action==='ban'?sampleFromHUB(getBanProbsHUB(patt,oppT,rem)):sampleFromHUB(getPickProbsHUB(patt,rem));
    fate[m]=step.action+step.side; rem=rem.filter(function(x){return x!==m;});
  });
  if(rem.length) fate[rem[0]]='dec';
  return fate;
}
function topVetoHUB(tA, tB, orgA, orgB, pool, snap, fmt, K) {
  var pA=((VETO_HUB.teams||{})['2026_'+snap]||{})[orgA]||null;
  var pB=((VETO_HUB.teams||{})['2026_'+snap]||{})[orgB]||null;
  K=K||3;
  var steps=VETO_STEPS_HUB[fmt]||VETO_STEPS_HUB.bo3;
  var states=[{rem:pool.slice(),seq:[],prob:1.0}];
  steps.forEach(function(step){
    var next=[];
    states.forEach(function(st){
      var patt=step.side==='A'?pA:pB, oppT=step.side==='A'?tB:tA;
      var probs=step.action==='ban'?getBanProbsHUB(patt,oppT,st.rem):getPickProbsHUB(patt,st.rem);
      st.rem.forEach(function(m){
        var p=probs[m]||0;
        if(p>0.005) next.push({rem:st.rem.filter(function(x){return x!==m;}),seq:st.seq.concat([{side:step.side,action:step.action,map:m}]),prob:st.prob*p});
      });
    });
    next.sort(function(a,b){return b.prob-a.prob;});
    states=next.slice(0,K*3);
  });
  states.forEach(function(st){if(st.rem.length) st.seq.push({side:'',action:'dec',map:st.rem[0]});});
  states.sort(function(a,b){return b.prob-a.prob;});
  return states.slice(0,K);
}
function actionLabelHUB(orgA, orgB, key) {
  if(key==='dec') return 'Decider';
  var verb=key.indexOf('ban')===0?'Ban':'Pick';
  var team=key.charAt(key.length-1)==='A'?orgA:orgB;
  return verb+' '+team;
}

// ── State ────────────────────────────────────────────────────────────────────
let hubData      = null;
let myChart      = null;
let logos        = {};
let selectedTeam = null;
let activeRegion = 'All';
let expandedOrg  = null;
let activePanel  = 'a';

// ── Tab switching ────────────────────────────────────────────────────────────
document.querySelectorAll('.tab').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    activePanel = btn.dataset.panel;
    const track = document.getElementById('panelTrack');
    track.classList.toggle('show-b', activePanel === 'b');
    track.classList.toggle('show-c', activePanel === 'c');
    track.classList.toggle('show-d', activePanel === 'd');
    const rp = document.getElementById('regionPills');
    // Override the inline transition (set by fadeIn) so max-height + margin animate too
    rp.style.transition = 'opacity .25s ease, max-height .5s cubic-bezier(.55,.06,.36,.98), margin-bottom .5s cubic-bezier(.55,.06,.36,.98)';
    rp.classList.toggle('hidden-panel', activePanel !== 'a');
    if (activePanel === 'b' && hubData) renderUpcoming(hubData);
    if (activePanel === 'c' && hubData) renderPast(hubData);
    // Sim is normally preloaded on init; if the user clicks before that
    // happens, defer the iframe-src assignment until after the slide so
    // the transform animation keeps the main thread to itself.
    if (activePanel === 'd' && !_simInitialized) {
      setTimeout(renderSimulator, 560);
    }
  });
});

// ── Region filter ────────────────────────────────────────────────────────────
document.querySelectorAll('.pill').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.pill').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    activeRegion = btn.dataset.region;
    selectedTeam = null;
    expandedOrg  = null;
    if (hubData) {
      buildChart(hubData);
      renderLeaderboard(hubData);
    }
  });
});

// ── Intro animation ──────────────────────────────────────────────────────────
async function introAnimation() {
  const title = document.getElementById('hubTitle');

  // Phase 1 — dot
  title.style.opacity = '1';
  title.innerHTML = '<span class="dot-seed">&middot;</span>';
  await sleep(380);

  // Phase 2 — typewriter
  const text = 'VCT Hub 2026';
  title.textContent = '';
  let built = '';
  for (const ch of text) {
    built += ch;
    title.innerHTML = built + '<span class="type-cursor">|</span>';
    await sleep(48 + Math.random() * 38);
  }
  for (let i = 0; i < 4; i++) {
    title.innerHTML = text + (i % 2 === 0 ? '<span class="type-cursor">|</span>' : '');
    await sleep(210);
  }
  title.textContent = text;

  // Phase 3 — fade rest in
  fadeIn('tabBar', 0.5);
  await sleep(80);
  // regionPills start invisible; shown later after chart is ready
}

// ── Data fetch ───────────────────────────────────────────────────────────────
async function fetchData() {
  try {
    const r = await fetch('/mapelo/modern/data');
    if (!r.ok) return null;
    return await r.json();
  } catch(e) { return null; }
}

// ── Progress bar ─────────────────────────────────────────────────────────────
// Persistent across polls — tracks every log line we've EVER rendered, so a
// line that was trimmed out of the visible 4-slot window doesn't re-appear
// as "new" on the next poll (which made the log look like it was looping).
window._mhubSeenLogLines = window._mhubSeenLogLines || new Set();
function updateProgress(prog) {
  if (!prog) return;
  document.getElementById('progressFill').style.width = (prog.pct || 0) + '%';
  document.getElementById('progressMsg').textContent  = prog.message || 'Checking…';
  document.getElementById('progressPct').textContent  = (prog.pct || 0) + '%';

  const logEl = document.getElementById('progressLog');
  (prog.log || []).forEach(line => {
    if (window._mhubSeenLogLines.has(line)) return;
    window._mhubSeenLogLines.add(line);
    const d = document.createElement('div');
    d.className = 'plog-entry new';
    d.textContent = line;
    logEl.appendChild(d);
  });
  // Trim DOM to the last 4 entries (visible) — the seen-set keeps the
  // earlier ones out of the "re-add" path even after they leave the DOM.
  const entries = logEl.querySelectorAll('.plog-entry');
  if (entries.length > 4) {
    for (let i = 0; i < entries.length - 4; i++) entries[i].remove();
  }
}

async function pollUntilReady() {
  let retries = 0;
  while (retries < 400) {
    await sleep(2000);
    retries++;
    const data = await fetchData();
    if (!data) continue;
    hubData = data;
    if (data.progress) updateProgress(data.progress);
    if (data.status === 'ready') {
      updateProgress({pct: 100, message: 'All data verified!', log: data.progress?.log || []});
      await sleep(900);
      return;
    }
  }
}

// ── Logo preloader ───────────────────────────────────────────────────────────
async function preloadLogos(teams) {
  await Promise.all(teams.map(t => new Promise(res => {
    const img = new Image();
    img.onload = () => { logos[t.org] = img; res(); };
    img.onerror = res;
    img.src = `/static/logos/${t.org}.png`;
  })));
}

// ── Axis animation overlay ───────────────────────────────────────────────────
async function animateAxesOverlay() {
  const canvas = document.getElementById('benpomChart');
  const wrap   = document.getElementById('chartWrap');
  const dpr    = window.devicePixelRatio || 1;
  const W      = wrap.offsetWidth;
  const H      = wrap.offsetHeight;

  const ov  = document.createElement('canvas');
  ov.width  = W * dpr;
  ov.height = H * dpr;
  ov.style.cssText = `position:absolute;top:0;left:0;width:${W}px;height:${H}px;pointer-events:none;z-index:5`;
  wrap.appendChild(ov);
  const oc = ov.getContext('2d');
  oc.scale(dpr, dpr);

  const ca = myChart.chartArea;
  const ox = myChart.scales.x.getPixelForValue(new Date('2026-01-15'));
  const oy = myChart.scales.y.getPixelForValue(0);

  await new Promise(resolve => {
    const dur = 1050, start = performance.now();
    function frame(now) {
      const p  = Math.min((now - start) / dur, 1);
      oc.clearRect(0, 0, W, H);

      // Cover chart with dark background (hides Chart.js rendering underneath)
      oc.fillStyle = '#ffffff';
      oc.fillRect(ca.left, ca.top, ca.right - ca.left, ca.bottom - ca.top);

      oc.save();

      if (p < 0.12) {
        // Dot phase
        const r = easeOut(p / 0.12) * 6;
        oc.shadowColor = '#8b5cf6'; oc.shadowBlur = 14;
        oc.beginPath(); oc.arc(ox, oy, r, 0, Math.PI * 2);
        oc.fillStyle = '#a78bfa'; oc.fill();
      } else {
        const lp = easeOut((p - 0.12) / 0.88);
        oc.shadowColor = '#8b5cf6'; oc.shadowBlur = 10;
        oc.strokeStyle = 'rgba(167,139,250,.9)'; oc.lineWidth = 1.5;

        // Y-axis (extends up AND down simultaneously)
        oc.beginPath();
        oc.moveTo(ox, oy - (oy - ca.top) * lp);
        oc.lineTo(ox, oy + (ca.bottom - oy) * lp);
        oc.stroke();

        // Zero line right
        oc.beginPath();
        oc.moveTo(ox, oy);
        oc.lineTo(ox + (ca.right - ox) * lp, oy);
        oc.stroke();

        // Zero line left (dim)
        oc.shadowBlur = 0; oc.globalAlpha = .28;
        oc.beginPath();
        oc.moveTo(ox, oy);
        oc.lineTo(ca.left + (ox - ca.left) * (1 - lp * .55), oy);
        oc.stroke();
        oc.globalAlpha = 1; oc.shadowBlur = 8;

        // Origin dot
        oc.beginPath(); oc.arc(ox, oy, 3.5, 0, Math.PI * 2);
        oc.fillStyle = '#c4b5fd'; oc.fill();

        // Late: hint grid lines
        if (lp > .78) {
          const tp = (lp - .78) / .22;
          oc.shadowBlur = 0; oc.globalAlpha = tp * .15;
          oc.strokeStyle = 'rgba(167,139,250,1)'; oc.lineWidth = .5;
          for (const v of [-10, -5, 5, 10]) {
            const py = myChart.scales.y.getPixelForValue(v);
            if (py < ca.top || py > ca.bottom) continue;
            oc.beginPath();
            oc.moveTo(ca.left, py);
            oc.lineTo(ca.left + (ca.right - ca.left) * lp, py);
            oc.stroke();
          }
          oc.globalAlpha = 1;
        }
      }
      oc.restore();
      if (p < 1) requestAnimationFrame(frame);
      else resolve();
    }
    requestAnimationFrame(frame);
  });

  // Phase 2: curtain sweeps right, revealing chart lines underneath
  await new Promise(resolve => {
    const dur  = 1700;
    const cw   = ca.right - ca.left;
    const ch   = ca.bottom - ca.top;
    let startT = null;
    function frame(ts) {
      if (!startT) startT = ts;
      const raw = Math.min((ts - startT) / dur, 1);
      const p   = 1 - Math.pow(1 - raw, 3);   // ease-out cubic
      const rx  = cw * p;
      oc.clearRect(0, 0, W, H);
      // dark fill covers what hasn't been revealed yet (right side)
      if (rx < cw) {
        oc.fillStyle = '#ffffff';
        oc.fillRect(ca.left + rx, ca.top, cw - rx + 1, ch);
      }
      // glowing edge at the reveal boundary
      if (rx > 0 && rx < cw) {
        const grd = oc.createLinearGradient(ca.left + rx - 28, 0, ca.left + rx + 4, 0);
        grd.addColorStop(0, 'rgba(167,139,250,0)');
        grd.addColorStop(1, 'rgba(167,139,250,0.3)');
        oc.fillStyle = grd;
        oc.fillRect(ca.left + rx - 28, ca.top, 32, ch);
      }
      if (raw < 1) requestAnimationFrame(frame);
      else resolve();
    }
    requestAnimationFrame(frame);
  });

  // quick fade out
  ov.style.transition = 'opacity 0.25s ease';
  ov.style.opacity = '0';
  await sleep(250);
  ov.remove();
}

// ── Band plugin ──────────────────────────────────────────────────────────────
function makeBandsPlugin(bands) {
  const COLS = [
    'rgba(147,112,219,.08)','rgba(100,149,237,.08)','rgba(128,200,100,.08)',
    'rgba(255,180,100,.08)','rgba(100,200,220,.08)','rgba(200,120,180,.08)',
  ];
  return {
    id:'eventBands',
    beforeDraw(chart) {
      const {ctx, chartArea:{left,top,right,bottom}, scales:{x}} = chart;
      bands.forEach((band, i) => {
        const x1 = Math.max(left,  x.getPixelForValue(new Date(band.start)));
        const x2 = Math.min(right, x.getPixelForValue(new Date(band.end)));
        if (x2 <= x1) return;
        ctx.fillStyle = COLS[i % COLS.length];
        ctx.fillRect(x1, top, x2 - x1, bottom - top);
        ctx.save();
        ctx.font = 'bold 10px DM Sans,sans-serif';
        ctx.fillStyle = 'rgba(60,30,100,.35)';
        ctx.textAlign = 'center';
        ctx.fillText(band.label, (x1 + x2) / 2, top + 14);
        ctx.restore();
      });
    },
  };
}

// ── Logo-endpoint plugin ─────────────────────────────────────────────────────
// ── Per-org grow/shrink animation between small dot and full logo ──────────
// progress 0 = small dot at the data endpoint
// progress 1 = full logo + halo (offset 15px to the right)
// Map<org, {progress, target, startProg, startTime}>. The plugin pushes the
// target each frame based on the current hover/selection state; a single
// shared RAF loop interpolates progress toward target on each tick.
const _logoAnimState = new Map();
let _logoAnimRaf = null;
// Snappy: 85ms feels instant but still smooth enough to read as motion.
// ease-out-quart (no ease-in) so the dot starts moving the same frame your
// cursor lands — no perceptible "wait, then grow" delay.
const _LOGO_ANIM_MS = 85;

function _tickLogoAnim() {
  let busy = false;
  const now = performance.now();
  _logoAnimState.forEach(st => {
    if (st.progress === st.target) return;
    const p = Math.min((now - st.startTime) / _LOGO_ANIM_MS, 1);
    const ep = 1 - Math.pow(1 - p, 4);  // ease-out-quart: fast start, soft land
    st.progress = st.startProg + ep * (st.target - st.startProg);
    if (p < 1) busy = true;
    else st.progress = st.target;
  });
  if (myChart) try { myChart.draw(); } catch (_) {}
  _logoAnimRaf = busy ? requestAnimationFrame(_tickLogoAnim) : null;
}

function _setLogoTarget(org, target) {
  let st = _logoAnimState.get(org);
  if (!st) {
    st = {progress: target, target: target, startProg: target, startTime: 0};
    _logoAnimState.set(org, st);
    return;
  }
  if (st.target === target) return;
  st.startProg = st.progress;
  st.target    = target;
  st.startTime = performance.now();
  if (!_logoAnimRaf) _logoAnimRaf = requestAnimationFrame(_tickLogoAnim);
}

const logoPlugin = {
  id:'teamLogos',
  afterDatasetsDraw(chart) {
    const {ctx, chartArea, scales:{x,y}} = chart;
    chart.data.datasets.forEach(ds => {
      if (!ds.data?.length || !ds.org || !logos[ds.org] || ds.type === 'scatter' || ds._dimmed || ds._noLogo) return;
      const last = ds.data[ds.data.length - 1];
      const px   = x.getPixelForValue(new Date(last.x));
      const py   = y.getPixelForValue(last.y);
      if (px < chartArea.left || px > chartArea.right + 30) return;
      const _isFocused = (selectedTeam === ds.org) || (_logoHoverOrg === ds.org);
      _setLogoTarget(ds.org, _isFocused ? 1 : 0);
      const prog = (_logoAnimState.get(ds.org) || {progress: _isFocused ? 1 : 0}).progress;
      const sz = 22;

      // Small dot at the endpoint — fades out while the logo grows in.
      if (prog < 0.999) {
        ctx.save();
        ctx.globalAlpha = 1 - prog;
        ctx.beginPath(); ctx.arc(px, py, 3.5, 0, Math.PI * 2);
        ctx.fillStyle = ds.borderColor; ctx.fill();
        ctx.restore();
      }
      if (prog <= 0.001) return;

      // Halo + logo at the offset position. Scale + fade animate from
      // dot-center (px) out to the logo's resting position so it visually
      // grows out of the dot rather than appearing in midair.
      const cx     = px + 4 + sz / 2;
      const ringR  = sz / 2 + 4;
      const drawCx = px + (cx - px) * prog;  // slide from dot to logo position
      ctx.save();
      ctx.globalAlpha = prog;
      ctx.translate(drawCx, py);
      ctx.scale(prog, prog);
      ctx.translate(-drawCx, -py);
      const grd = ctx.createRadialGradient(drawCx, py, 0, drawCx, py, ringR);
      grd.addColorStop(0, '#ffffff');
      grd.addColorStop(0.62, '#ffffff');
      grd.addColorStop(1, ds.borderColor);
      ctx.beginPath(); ctx.arc(drawCx, py, ringR, 0, Math.PI * 2);
      ctx.fillStyle = grd; ctx.fill();
      ctx.beginPath(); ctx.arc(drawCx, py, sz / 2, 0, Math.PI * 2); ctx.clip();
      const _logoScale = (LOGO_SCALES[ds.org] != null) ? LOGO_SCALES[ds.org] : 1;
      const _drawSz = sz * _logoScale;
      ctx.drawImage(logos[ds.org], drawCx - _drawSz / 2, py - _drawSz / 2, _drawSz, _drawSz);
      ctx.restore();

      // Info card only when fully grown — avoids drawing card while logo is
      // still scaling in (and prevents jitter as we round position pixels).
      if (prog < 0.98) return;
      // Mini info card for the selected team
      if (selectedTeam !== ds.org || !hubData) return;
      const team = (hubData.leaderboard.teams || []).find(t => t.org === ds.org);
      if (!team) return;
      const rStr = (team.rating >= 0 ? '+' : '') + team.rating.toFixed(2);

      // W-L in current event
      const asOf  = hubData.as_of_date || '';
      const bands = hubData.event_bands || [];
      const curBand = bands.find(b => b.start <= asOf && asOf <= b.end)
                   || [...bands].reverse().find(b => b.start <= asOf);
      const mes   = hubData.chart.match_events || [];
      const evMes = curBand ? mes.filter(m => m.date >= curBand.start && m.date <= curBand.end) : mes;
      const wins   = evMes.filter(m => m.winner === ds.org).length;
      const losses = evMes.filter(m => m.loser  === ds.org).length;
      const wlStr  = `${wins}W – ${losses}L`;
      const evLabel = curBand ? curBand.label.replace(' 2026','').replace(' 2025','') : '';

      const cardX = cx + sz / 2 + 8;
      const cardW = 88, cardH = 52, cardR = 8;
      const cardY = py - cardH / 2;

      ctx.save();
      ctx.shadowBlur = 10; ctx.shadowColor = 'rgba(0,0,0,.14)';
      ctx.beginPath();
      ctx.roundRect(cardX, cardY, cardW, cardH, cardR);
      ctx.fillStyle = 'rgba(255,255,255,.97)'; ctx.fill();
      ctx.shadowBlur = 0;
      ctx.beginPath();
      ctx.roundRect(cardX, cardY, cardW, cardH, cardR);
      ctx.strokeStyle = ds.borderColor + '55'; ctx.lineWidth = 1.5; ctx.stroke();
      // Colored top stripe
      ctx.beginPath();
      ctx.roundRect(cardX, cardY, cardW, 4, [cardR, cardR, 0, 0]);
      ctx.fillStyle = ds.borderColor; ctx.fill();
      // Rating
      ctx.font = 'bold 14px "DM Sans",sans-serif';
      ctx.fillStyle = ds.borderColor; ctx.textAlign = 'center';
      ctx.fillText(rStr, cardX + cardW / 2, cardY + 22);
      // W-L
      ctx.font = '10.5px "DM Sans",sans-serif';
      ctx.fillStyle = '#555';
      ctx.fillText(wlStr, cardX + cardW / 2, cardY + 36);
      // Event label
      if (evLabel) {
        ctx.font = '9px "DM Sans",sans-serif';
        ctx.fillStyle = '#999';
        ctx.fillText(evLabel, cardX + cardW / 2, cardY + 48);
      }
      ctx.restore();
    });
  },
};

// ── Chart build ──────────────────────────────────────────────────────────────
var _chartYMin = null, _chartYMax = null;
function _computeGlobalYRange(data) {
  // Y-axis is locked centered at 0, range [-12, +12]. Keeps the visual
  // scale stable across snapshots regardless of where the league's
  // current peak rating happens to sit.
  _chartYMin = -12;
  _chartYMax =  12;
}

function buildChart(data, noLines = false) {
  const checkpoints = data.chart.checkpoints || [];
  const matchEvents = data.chart.match_events || [];
  const allTeams    = data.leaderboard.teams  || [];

  const visible = activeRegion === 'All' ? allTeams
    : activeRegion === 'Top10'           ? allTeams.slice(0, 10)
    : allTeams.filter(t => t.region === activeRegion);
  const visOrgs = new Set(visible.map(t => t.org));

  const datasets = [];
  visible.forEach(team => {
    const org = team.org;
    const pts = checkpoints.filter(cp => org in cp.ratings)
                           .map(cp => ({x: cp.date, y: cp.ratings[org]}));
    if (!pts.length) return;
    const color    = TEAM_COLORS[org] || '#888';
    const isSel    = selectedTeam === org;
    const isDimmed = selectedTeam !== null && !isSel;
    datasets.push({
      label: org, org,
      data: pts,
      borderColor: noLines ? 'transparent' : (isDimmed ? color + '28' : color),
      backgroundColor: 'transparent',
      borderWidth: noLines ? 0 : (isSel ? 2.5 : (selectedTeam ? 1 : 1.5)),
      pointRadius: 0, pointHoverRadius: 0,
      tension: 0.25, _dimmed: isDimmed, _noLogo: noLines,
    });
  });

  // Match dots: selected team (large) or all teams when zoomed into a split (small)
  if (selectedTeam && visOrgs.has(selectedTeam)) {
    const tm = matchEvents.filter(m => m.winner === selectedTeam || m.loser === selectedTeam);
    const wins = [], losses = [];
    tm.forEach(m => {
      const won = m.winner === selectedTeam;
      const pt  = {x:m.date, y:won?m.winner_after:m.loser_after, _m:m, _won:won};
      (won ? wins : losses).push(pt);
    });
    if (wins.length)   datasets.push({type:'scatter',label:'Win',  org:selectedTeam,data:wins,  backgroundColor:'#4ade80',pointRadius:7,pointHoverRadius:9,borderWidth:0,_dimmed:false});
    if (losses.length) datasets.push({type:'scatter',label:'Loss', org:selectedTeam,data:losses,backgroundColor:'#f87171',pointRadius:7,pointHoverRadius:9,borderWidth:0,_dimmed:false});
  } else if (_isZoomed) {
    const allWins = [], allLosses = [];
    matchEvents.forEach(m => {
      if (visOrgs.has(m.winner)) allWins.push({x:m.date, y:m.winner_after, _m:m, _won:true});
      if (visOrgs.has(m.loser))  allLosses.push({x:m.date, y:m.loser_after,  _m:m, _won:false});
    });
    if (allWins.length)   datasets.push({type:'scatter',label:'Win',  org:null,data:allWins,  backgroundColor:'rgba(74,222,128,.7)', pointRadius:5,pointHoverRadius:7,borderWidth:0,_dimmed:false});
    if (allLosses.length) datasets.push({type:'scatter',label:'Loss', org:null,data:allLosses, backgroundColor:'rgba(248,113,113,.7)',pointRadius:5,pointHoverRadius:7,borderWidth:0,_dimmed:false});
  }

  const bandsPlugin = makeBandsPlugin(data.event_bands || []);

  if (myChart) myChart.destroy();
  const ctx = document.getElementById('benpomChart').getContext('2d');
  myChart = new Chart(ctx, {
    type: 'line',
    data: {datasets},
    options: {
      animation: false,
      responsive: true, maintainAspectRatio: false,
      interaction: {mode:'point', intersect:true},
      plugins: {
        legend: {display:false},
        tooltip: { enabled: false },
      },
      scales: {
        x: {
          type:'time',
          min: (_savedZoomMin && _savedZoomMin >= new Date('2026-01-01').getTime()) ? _savedZoomMin : '2026-01-07',
          max: (_savedZoomMax && _savedZoomMax <= new Date('2026-11-01').getTime()) ? _savedZoomMax : '2026-10-25',
          time:{unit:'month', displayFormats:{month:'MMM'}},
          grid:{color:'rgba(0,0,0,.07)'},
          ticks:{color:'rgba(0,0,0,.45)', font:{size:11}},
          border:{color:'rgba(0,0,0,.12)'},
        },
        y: {
          min: _chartYMin,
          max: _chartYMax,
          grid:{color:'rgba(0,0,0,.07)'},
          ticks:{color:'rgba(0,0,0,.45)', font:{size:11}, callback:v => v===0 ? '0' : (v>0?'+':'')+v.toFixed(1)},
          afterBuildTicks(scale) {
            const ticks = [];
            for (let v = _chartYMin; v <= _chartYMax + 0.001; v += 0.5) {
              ticks.push({value: Math.round(v * 10) / 10});
            }
            scale.ticks = ticks;
          },
          border:{color:'rgba(0,0,0,.12)'},
        },
      },
      layout:{padding:{right:32}},
    },
    plugins: [bandsPlugin, logoPlugin],
  });

}

let _logoHoverOrg   = null;
let _lastHoveredDot = null;

// ── Canvas listeners (registered once) ──────────────────────────────────────
function _initCanvasListeners() {
  const canvas = document.getElementById('benpomChart');

  // ── Hit test logos ────────────────────────────────────────────────────────
  function _hitTestLogos(mx, my) {
    if (!myChart || !logos) return null;
    const {scales: {x, y}} = myChart;
    // Two hit zones, with very different purposes:
    //   - SMALL_HIT = the unfocused-state hit zone, centered on the actual
    //     dot. This is the only thing that promotes a team into the focused
    //     state. Kept tight so cursoring near (but not on) a dot doesn't
    //     trigger the "snap large" effect.
    //   - FOCUSED_HIT = the focused-state hit zone, covers the expanded
    //     logo offset 15px right. Only consulted once a team is already
    //     focused, so the user can move from the dot into the logo without
    //     losing the hover state.
    const SMALL_HIT   = 5;
    const FOCUSED_HIT = 15;
    let hit = null;
    myChart.data.datasets.forEach(ds => {
      if (!ds.data?.length || !ds.org || !logos[ds.org] || ds.type === 'scatter' || ds._dimmed) return;
      const last = ds.data[ds.data.length - 1];
      const px = x.getPixelForValue(new Date(last.x));
      const py = y.getPixelForValue(last.y);
      // Small-dot test — always available, this is what triggers focus.
      if (Math.sqrt((mx - px) ** 2 + (my - py) ** 2) <= SMALL_HIT) { hit = ds.org; return; }
      // Expanded-logo test — only when this team is ALREADY focused.
      const isFocused = (selectedTeam === ds.org) || (_logoHoverOrg === ds.org);
      if (isFocused) {
        const cxFocused = px + 4 + 11;
        if (Math.sqrt((mx - cxFocused) ** 2 + (my - py) ** 2) <= FOCUSED_HIT) hit = ds.org;
      }
    });
    return hit;
  }

  // ── Hover (dots + logos) ──────────────────────────────────────────────────
  canvas.addEventListener('mousemove', e => {
    if (!myChart) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    // Dot hover
    const els = myChart.getElementsAtEventForMode(e, 'point', {intersect: true}, false);
    const dotEl = els.find(el => myChart.data.datasets[el.datasetIndex]?.data[el.index]?._m);
    if (dotEl) {
      const pt  = myChart.data.datasets[dotEl.datasetIndex].data[dotEl.index];
      const key = pt._m.date + pt._m.winner + pt._m.loser;
      if (key !== _lastHoveredDot) {
        _lastHoveredDot = key;
        const wr = document.getElementById('chartWrap').getBoundingClientRect();
        showDotTooltip(pt._m, pt._won, e.clientX - wr.left, e.clientY - wr.top);
      }
      canvas.style.cursor = 'pointer';
      return;
    }
    if (_lastHoveredDot) { _lastHoveredDot = null; hideDotTooltip(); }

    // Logo hover
    const hovered = _hitTestLogos(mx, my);
    if (hovered) {
      canvas.style.cursor = 'pointer';
      if (hovered !== _logoHoverOrg) { _logoHoverOrg = hovered; _applyLogoHover(hovered); }
    } else {
      if (_logoHoverOrg) { _logoHoverOrg = null; _applyLogoHover(null); }
    }
  });

  canvas.addEventListener('mouseleave', () => {
    _lastHoveredDot = null;
    hideDotTooltip();
    if (_logoHoverOrg) { _logoHoverOrg = null; _applyLogoHover(null); }
  });

  // ── Logo click ────────────────────────────────────────────────────────────
  canvas.addEventListener('click', e => {
    if (!_logoHoverOrg) return;
    e.stopPropagation();
    selectedTeam  = _logoHoverOrg;
    expandedOrg   = null;
    _logoHoverOrg = null;
    buildChart(hubData);
    renderLeaderboard(hubData);
  });
}

function _applyLogoHover(org) {
  if (!myChart) return;
  myChart.data.datasets.forEach(ds => {
    if (ds.type === 'scatter') return;
    const base = TEAM_COLORS[ds.org] || '#888';
    if (!org) {
      // restore to current selectedTeam state
      const isSel    = ds.org === selectedTeam;
      const isDimmed = selectedTeam !== null && !isSel;
      ds.borderColor = isDimmed ? base + '28' : base;
      ds.borderWidth = isSel ? 2.5 : (selectedTeam ? 1 : 1.5);
      ds._dimmed = isDimmed;
    } else {
      const isHov = ds.org === org;
      ds.borderColor = isHov ? base : base + '28';
      ds.borderWidth = isHov ? 2.5 : 1;
      ds._dimmed = !isHov;
    }
  });
  myChart.update('none');
}

// Clicking anywhere outside a logo clears the selection
document.addEventListener('click', () => {
  if (!selectedTeam || !hubData) return;
  const det = document.querySelector('.lb-detail');
  selectedTeam = null;
  expandedOrg  = null;
  if (det) {
    document.querySelectorAll('.lb-row.selected').forEach(r => r.classList.remove('selected'));
    det.classList.add('closing');
    setTimeout(() => {
      if (hubData) { renderLeaderboard(hubData); buildChart(hubData); }
    }, 220);
    return;
  }
  buildChart(hubData);
  renderLeaderboard(hubData);
});

// ── Keyboard navigation: W/S to cycle teams, X to clear ──────────────────────
document.addEventListener('keydown', e => {
  if (!hubData) return;
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  const key = e.key.toLowerCase();
  if (key !== 'w' && key !== 's' && key !== 'x') return;
  e.preventDefault();

  if (key === 'x') {
    selectedTeam = null; expandedOrg = null;
    buildChart(hubData); renderLeaderboard(hubData);
    return;
  }

  const teams = hubData.leaderboard.teams || [];
  const visible = activeRegion === 'All' ? teams
    : activeRegion === 'Top10'           ? teams.slice(0, 10)
    : teams.filter(t => t.region === activeRegion);
  if (!visible.length) return;

  const idx = selectedTeam ? visible.findIndex(t => t.org === selectedTeam) : -1;
  let next;
  if (key === 'w') next = idx <= 0 ? visible[0] : visible[idx - 1];
  else             next = idx < 0 || idx >= visible.length - 1 ? visible[visible.length - 1] : visible[idx + 1];

  selectedTeam = next.org;
  expandedOrg  = null;
  buildChart(hubData);
  renderLeaderboard(hubData);
});

// ── as-of label ──────────────────────────────────────────────────────────────
function setAsOf(data) {
  const date   = data.as_of_date  || '';
  const event  = data.as_of_event || '';
  if (!date) return;
  const d    = new Date(date + 'T12:00:00');
  const dStr = d.toLocaleDateString('en-US', {month:'short', day:'numeric', year:'numeric'});
  document.getElementById('chartAsOf').textContent =
    event ? `Through ${event} · ${dStr}` : `Through ${dStr}`;
  document.getElementById('lbAsOf').textContent =
    `Has all 2026 matches through ${dStr}`;
}

// ── Chart animation: progressive x.max reveal ────────────────────────────────
let _isReplaying  = false;
let _isZoomed     = false;
let _savedZoomMin = null;
let _savedZoomMax = null;

async function revealChart(duration = 2500, startFromLeft = false) {
  if (!myChart || !hubData) return;
  _isReplaying = true;
  const btn = document.getElementById('replayBtn');
  if (btn) { btn.textContent = '⏸ Playing…'; btn.disabled = true; }

  _isZoomed = false;
  _savedZoomMin = null; _savedZoomMax = null;
  const zBtn = document.getElementById('zoomBtn');
  if (zBtn) { zBtn.textContent = '⊕ Zoom Split'; zBtn.classList.remove('active'); }
  myChart.options.scales.x.min = '2026-01-07';
  myChart.options.scales.x.max = '2026-10-25';
  myChart.update('none');

  const cps     = hubData.chart.checkpoints || [];
  const firstMs = cps.length ? new Date(cps[0].date).getTime() : new Date('2026-01-15').getTime();
  const lastMs  = new Date(hubData.as_of_date || '2026-05-10').getTime();

  // Overlay canvas — sits on top of chart, sweeps a white curtain left→right
  const mainCanvas = document.getElementById('benpomChart');
  const wrap = document.getElementById('chartWrap');
  const dpr  = window.devicePixelRatio || 1;
  const ov   = document.createElement('canvas');
  ov.style.cssText = 'position:absolute;top:0;left:0;pointer-events:none;z-index:5;';
  ov.width  = mainCanvas.offsetWidth  * dpr;
  ov.height = mainCanvas.offsetHeight * dpr;
  ov.style.width  = mainCanvas.offsetWidth  + 'px';
  ov.style.height = mainCanvas.offsetHeight + 'px';
  wrap.appendChild(ov);
  const oc = ov.getContext('2d');
  oc.scale(dpr, dpr);

  const ca   = myChart.chartArea;
  const endX = startFromLeft ? myChart.scales.x.getPixelForValue(lastMs) : null;

  await new Promise(resolve => {
    const startT = performance.now();
    function frame(ts) {
      const p      = Math.min((ts - startT) / duration, 1);
      const ep     = 1 - Math.pow(1 - p, 3);
      const nowMs  = firstMs + ep * (lastMs - firstMs);
      const revX   = startFromLeft
        ? ca.left + ep * (endX - ca.left)
        : myChart.scales.x.getPixelForValue(nowMs);

      oc.clearRect(0, 0, ov.offsetWidth, ov.offsetHeight);

      // White curtain covers everything to the right of the reveal line
      if (revX < ca.right) {
        oc.fillStyle = '#ffffff';
        oc.fillRect(revX, ca.top, ca.right - revX + 2, ca.bottom - ca.top);

        // Re-draw event bands on top of curtain so they stay visible
        const _bands = hubData.event_bands || [];
        const _BCOLS = ['rgba(147,112,219,.08)','rgba(100,149,237,.08)','rgba(128,200,100,.08)',
                        'rgba(255,180,100,.08)','rgba(100,200,220,.08)','rgba(200,120,180,.08)'];
        _bands.forEach((_b, _i) => {
          const _truex1 = myChart.scales.x.getPixelForValue(new Date(_b.start));
          const _truex2 = myChart.scales.x.getPixelForValue(new Date(_b.end));
          const _bx1 = Math.max(revX, _truex1);
          const _bx2 = Math.min(ca.right, _truex2);
          if (_bx2 <= _bx1) return;
          oc.fillStyle = _BCOLS[_i % _BCOLS.length];
          oc.fillRect(_bx1, ca.top, _bx2 - _bx1, ca.bottom - ca.top);
          // Label always at true band center — never shifts with curtain
          const _labelX = Math.max(_truex1, Math.min(_truex2, (_truex1 + _truex2) / 2));
          if (_labelX >= revX && _labelX <= ca.right) {
            oc.save();
            oc.font = 'bold 10px DM Sans,sans-serif';
            oc.fillStyle = 'rgba(60,30,100,.35)';
            oc.textAlign = 'center';
            oc.fillText(_b.label, _labelX, ca.top + 14);
            oc.restore();
          }
        });

        // Re-draw the axis grid on top of the curtain so the gridlines
        // stay visible throughout the reveal animation.  Match Chart.js's
        // grid config EXACTLY (rgba(0,0,0,.07), 1px) so the lines on the
        // covered side look identical to the lines on the uncovered side
        // — any deviation reads as "the right side is bolder than the
        // left" once the curtain crosses over.
        oc.save();
        oc.strokeStyle = 'rgba(0,0,0,0.07)';
        oc.lineWidth   = 1;

        // Horizontal grid — must hit the SAME pixel rows as Chart.js's own
        // gridlines on the uncovered side, otherwise the seam at the curtain
        // edge shows two grids running at half-step offsets. Chart.js draws
        // a line at every integer y-tick (see the +12.0 / +11.0 / +10.0 axis
        // labels), so iterate integer values that fall inside the visible y
        // range. Walk the actual rendered ticks if Chart.js exposes them so
        // that any future tick-density change (e.g. step=2) stays in sync.
        const _yMin = myChart.scales.y.min;
        const _yMax = myChart.scales.y.max;
        const _yTickVals = (myChart.scales.y.ticks || [])
          .map(t => (typeof t === 'object' ? t.value : t))
          .filter(v => typeof v === 'number');
        const _yIter = _yTickVals.length
          ? _yTickVals
          : (() => {
              const out = [];
              for (let _v = Math.ceil(_yMin); _v <= Math.floor(_yMax) + 1e-6; _v += 1) out.push(_v);
              return out;
            })();
        for (const _v of _yIter) {
          if (_v < _yMin - 1e-6 || _v > _yMax + 1e-6) continue;
          const _py = myChart.scales.y.getPixelForValue(_v);
          if (_py < ca.top - 0.5 || _py > ca.bottom + 0.5) continue;
          oc.beginPath();
          oc.moveTo(revX, _py);
          oc.lineTo(ca.right, _py);
          oc.stroke();
        }

        // Vertical grid: mirror Chart.js's own x-axis ticks so the lines
        // under the curtain land at the exact same pixels as the lines on
        // the uncovered side. Falls back to first-of-month if for some
        // reason the chart doesn't expose ticks yet.
        const _xTickVals = (myChart.scales.x.ticks || [])
          .map(t => (typeof t === 'object' ? t.value : t))
          .filter(v => typeof v === 'number');
        const _xIter = _xTickVals.length
          ? _xTickVals.map(v => new Date(v))
          : (() => {
              const out = [];
              const _xMin = new Date(myChart.scales.x.min);
              const _xMax = new Date(myChart.scales.x.max);
              const _m0   = new Date(_xMin.getFullYear(), _xMin.getMonth(), 1);
              if (_m0 < _xMin) _m0.setMonth(_m0.getMonth() + 1);
              for (let _d = new Date(_m0); _d <= _xMax; _d.setMonth(_d.getMonth() + 1)) out.push(new Date(_d));
              return out;
            })();
        for (const _d of _xIter) {
          const _px = myChart.scales.x.getPixelForValue(_d.getTime());
          if (_px < revX - 0.5 || _px > ca.right + 0.5) continue;
          oc.beginPath();
          oc.moveTo(_px, ca.top);
          oc.lineTo(_px, ca.bottom);
          oc.stroke();
        }
        oc.restore();
      }

      // Glowing edge at the reveal boundary
      if (revX > ca.left && revX < ca.right) {
        const grd = oc.createLinearGradient(revX - 22, 0, revX + 4, 0);
        grd.addColorStop(0, 'rgba(167,139,250,0)');
        grd.addColorStop(1, 'rgba(167,139,250,0.32)');
        oc.fillStyle = grd;
        oc.fillRect(revX - 22, ca.top, 26, ca.bottom - ca.top);

        // Small colored dot on each team's line at the reveal position
        myChart.data.datasets.forEach(ds => {
          if (ds.type === 'scatter' || ds._dimmed || !ds.data?.length) return;
          const pts = ds.data;
          let yVal = null;
          for (let i = 0; i < pts.length - 1; i++) {
            const t0 = new Date(pts[i].x).getTime();
            const t1 = new Date(pts[i+1].x).getTime();
            if (nowMs >= t0 && nowMs <= t1) {
              yVal = pts[i].y + (nowMs - t0) / (t1 - t0) * (pts[i+1].y - pts[i].y);
              break;
            }
          }
          if (yVal === null) {
            const last = pts[pts.length - 1];
            if (new Date(last.x).getTime() <= nowMs) yVal = last.y;
          }
          if (yVal !== null) {
            const py = myChart.scales.y.getPixelForValue(yVal);
            if (py >= ca.top && py <= ca.bottom) {
              oc.save();
              oc.shadowBlur = 5; oc.shadowColor = ds.borderColor || '#c4b5fd';
              oc.beginPath(); oc.arc(revX, py, 3, 0, Math.PI * 2);
              oc.fillStyle = ds.borderColor || '#c4b5fd'; oc.fill();
              oc.restore();
            }
          }
        });
      }

      if (p < 1) requestAnimationFrame(frame);
      else resolve();
    }
    requestAnimationFrame(frame);
  });

  ov.style.transition = 'opacity 0.25s';
  ov.style.opacity = '0';
  await sleep(260);
  ov.remove();
  _isReplaying = false;
  if (btn) { btn.innerHTML = '&#8635; Replay'; btn.disabled = false; }
}

async function replayChart() {
  if (_isReplaying) return;
  await revealChart(3700);
}

async function animateZoom(toMin, toMax, duration) {
  if (!myChart) return;
  const fmn = new Date(myChart.options.scales.x.min).getTime();
  const fmx = new Date(myChart.options.scales.x.max).getTime();
  const tmn = new Date(toMin).getTime();
  const tmx = new Date(toMax).getTime();
  await new Promise(resolve => {
    const startT = performance.now();
    function frame(ts) {
      const p  = Math.min((ts - startT) / duration, 1);
      const ep = 1 - Math.pow(1 - p, 3);
      myChart.options.scales.x.min = new Date(fmn + ep * (tmn - fmn));
      myChart.options.scales.x.max = new Date(fmx + ep * (tmx - fmx));
      myChart.update('none');
      if (p < 1) requestAnimationFrame(frame);
      else resolve();
    }
    requestAnimationFrame(frame);
  });
}

async function toggleZoom() {
  if (_isReplaying || !myChart) return;
  const btn = document.getElementById('zoomBtn');
  if (_isZoomed) {
    const prevMin = myChart.options.scales.x.min;
    const prevMax = myChart.options.scales.x.max;
    _isZoomed = false;
    _savedZoomMin = null; _savedZoomMax = null;
    buildChart(hubData);
    myChart.options.scales.x.min = prevMin;
    myChart.options.scales.x.max = prevMax;
    myChart.update('none');
    await animateZoom('2026-01-07', '2026-10-25', 600);
    if (btn) { btn.textContent = '⊕ Zoom Split'; btn.classList.remove('active'); }
  } else {
    const asOf  = hubData?.as_of_date || '2026-05-10';
    const bands = hubData?.event_bands || [];
    let band = bands.find(b => b.start <= asOf && asOf <= b.end);
    if (!band) band = [...bands].reverse().find(b => b.start <= asOf);
    if (!band && bands.length) band = bands[bands.length - 1];
    const zStart = band ? band.start : asOf;
    const zEnd   = band ? band.end   : asOf;
    _isZoomed = true;
    buildChart(hubData);
    await animateZoom(zStart, zEnd, 600);
    _savedZoomMin = myChart.scales.x.min;
    _savedZoomMax = myChart.scales.x.max;
    if (btn) { btn.textContent = '⊖ Zoom Out'; btn.classList.add('active'); }
  }
}

async function resetZoom() {
  if (!myChart) return;
  const prevMin = myChart.options.scales.x.min;
  const prevMax = myChart.options.scales.x.max;
  _isZoomed     = false;
  _savedZoomMin = null; _savedZoomMax = null;
  buildChart(hubData);
  myChart.options.scales.x.min = prevMin;
  myChart.options.scales.x.max = prevMax;
  myChart.update('none');
  await animateZoom('2026-01-07', '2026-10-25', 400);
  const zBtn = document.getElementById('zoomBtn');
  if (zBtn) { zBtn.textContent = '⊕ Zoom Split'; zBtn.classList.remove('active'); }
}

// ── Chart section reveal ─────────────────────────────────────────────────────
async function showChartAndLeaderboard(data) {
  // Initialize veto simulation globals from hub data
  VETO_HUB       = data.veto_model   || {teams:{}, snap_pools:{}};
  ORG_REGIONS_HUB = data.org_regions || {};
  INTL_HUB       = data.intl_calib   || {};
  SNAP_TEAMS     = data.snap_teams   || {};
  // Don't overwrite with data.snap_beta — that value (~0.32) is the
  // overfit training-set β. Keep the hardcoded CV-optimal 0.17 from the
  // module-level constant.
  SNAP_KEY       = data.snap_key     || 'after_santiago';

  await preloadLogos(data.leaderboard.teams || []);
  setAsOf(data);

  _computeGlobalYRange(data);

  // Build chart without lines so shading/axes are visible during slide-in
  buildChart(data, true);
  _initCanvasListeners();   // register once after first build

  // Slide card in from the left — lines hidden, bands/axes visible
  showEl('chartSection');
  const chartCard = document.querySelector('.chart-card');
  if (chartCard) chartCard.classList.add('entering');

  // Auto-scroll FIRST so the user lands on a centered chart before the
  // reveal animation plays. Skip if the user already scrolled themselves.
  if (window.scrollY < 40 && chartCard) {
    const _rect = chartCard.getBoundingClientRect();
    const _target = window.scrollY + _rect.top + _rect.height / 2 - window.innerHeight / 2;
    window.scrollTo({top: Math.max(0, _target), behavior: 'smooth'});
  }

  await sleep(2500);

  // Rebuild with real lines, then immediately sweep curtain from left
  buildChart(data);
  await revealChart(4400, true);

  // Show leaderboard and pills after reveal completes
  showEl('lbCard');
  renderLeaderboard(data, {animate: true});
  fadeIn('regionPills', 0.4);
}

// ── Main init ────────────────────────────────────────────────────────────────
async function init() {
  const dataP = fetchData();     // start fetch immediately
  await introAnimation();        // run intro in parallel

  const data = await dataP;
  if (!data) {
    document.getElementById('lbBody').innerHTML = '<div class="lb-empty">Could not load data. Please refresh.</div>';
    showEl('lbCard');
    return;
  }
  hubData = data;

  // Always show real progress — poll until backend says ready
  showEl('progressSection');
  // Scroll to the top so the user sees the progress bar / "Loading…" state
  // as soon as the backend says it's building. Without this, if the user
  // already scrolled while waiting, the progress card animates in offscreen.
  window.scrollTo({top: 0, behavior: 'smooth'});
  if (data.progress) updateProgress(data.progress);
  await pollUntilReady();

  // Bar done — flash white, then glitch-exit the card
  const fill = document.getElementById('progressFill');
  if (fill) fill.classList.add('done');
  const pLabel = document.querySelector('.progress-label');
  if (pLabel) pLabel.textContent = 'Ready';
  await sleep(520);
  const pSec = document.getElementById('progressSection');
  if (pSec) pSec.style.overflow = 'visible';
  const pOuter = document.querySelector('.panels-outer');
  if (pOuter) pOuter.style.overflow = 'visible';
  const pCard = document.querySelector('.progress-card');
  if (pCard) pCard.classList.add('exiting');
  await sleep(560);
  document.getElementById('progressSection').classList.add('hidden');
  // Restore overflow:hidden so the chart card clips correctly during slide-in
  if (pOuter) pOuter.style.overflow = '';

  await showChartAndLeaderboard(hubData);

  // Warm the simulator iframe in the background once the main view has
  // settled. Lets the click-to-open animation slide a fully-rendered panel
  // instead of doing a network fetch + layout pass mid-slide.
  preloadSimulator();
}

// ── Leaderboard ──────────────────────────────────────────────────────────────
function renderLeaderboard(data, opts) {
  const animate = !!(opts && opts.animate);
  const teams   = data.leaderboard.teams || [];
  const visible = activeRegion === 'All' ? teams
    : activeRegion === 'Top10'          ? teams.slice(0, 10)
    : teams.filter(t => t.region === activeRegion);
  const body    = document.getElementById('lbBody');
  body.innerHTML = '';

  if (!visible.length) {
    body.innerHTML = '<div class="lb-empty">No teams found.</div>';
    return;
  }

  visible.forEach((team, _idx) => {
    const org   = team.org;
    const color = TEAM_COLORS[org] || '#888';
    const rStr  = (team.rating >= 0 ? '+' : '') + team.rating.toFixed(2);
    const regCls = (team.region || '').toLowerCase();
    const isSel  = org === selectedTeam;
    const isExp  = org === expandedOrg;

    const row = document.createElement('div');
    row.className = 'lb-row' + (isSel ? ' selected' : '') + (animate ? ' slide-in' : '');
    if (animate) row.style.animationDelay = (_idx * 55) + 'ms';
    row.innerHTML = `
      <div class="lb-rank">${team.rank}</div>
      <div class="lb-team">
        <img src="/static/logos/${org}.png" onerror="this.style.display='none'" alt="${org}">
        <span class="lb-name">${org}</span>
      </div>
      <div class="lb-rating">${rStr}</div>
      <div class="lb-region ${regCls}">${team.region || ''}</div>
      <div class="lb-chevron">&#9660;</div>`;
    row.onclick = e => { e.stopPropagation(); toggleTeam(org); };
    body.appendChild(row);

    if (isExp) {
      const det = document.createElement('div');
      det.className = 'lb-detail';
      det.innerHTML = buildDetailHTML(team);
      det.addEventListener('click', e => e.stopPropagation());
      body.appendChild(det);
    }
  });
}

const EVENT_LABELS = {
  // 2026
  '2026_kickoff':           'Kickoff 2026',
  '2026_masters_santiago':  'Masters Santiago',
  '2026_stage1':            'Stage 1 2026',
  '2026_masters_london':    'Masters London',
  '2026_stage2':            'Stage 2 2026',
  '2026_champions':         'Champions 2026',
  // 2025
  '2025_kickoff':           'Kickoff 2025',
  '2025_masters_bangkok':   'Masters Bangkok',
  '2025_stage1':            'Stage 1 2025',
  '2025_masters_toronto':   'Masters Toronto',
  '2025_stage2':            'Stage 2 2025',
  '2025_champions':         'Champions 2025',
  // 2024
  '2024_kickoff':           'Kickoff 2024',
  '2024_masters_madrid':    'Masters Madrid',
  '2024_stage1':            'Stage 1 2024',
  '2024_masters_shanghai':  'Masters Shanghai',
  '2024_stage2':            'Stage 2 2024',
  '2024_champions':         'Champions 2024',
  // 2023
  '2023_lock_in':           'LOCK//IN 2023',
  '2023_masters_tokyo':     'Masters Tokyo',
  '2023_league':            'League 2023',
  '2023_champions':         'Champions 2023',
};

function buildDetailHTML(team) {
  const recent  = (team.recent_matches || []).slice(0, 4);
  const allMaps = team.all_maps || [];
  const org     = team.org;

  // Map result chips: green + 13-first if team won map, red + 13-last if lost
  const mapChips = (maps, o) => (maps || []).map(mp => {
    const mapWon = mp.winner === o;
    const cls = mapWon ? 'mw' : 'ml';
    const scoreStr = mapWon ? `${mp.wr}–${mp.lr}` : `${mp.lr}–${mp.wr}`;
    return `<span class="lb-mmap-chip ${cls}">${mp.map} ${scoreStr}</span>`;
  }).join('');

  // Player headshots row
  const roster = (team.roster || []).slice(0, 5);
  const rosterHtml = roster.length ? `
    <div class="lb-sec-label">Players</div>
    <div class="lb-player-row">${roster.map(p => {
      const name = p.player || '';
      const hs   = p.headshot || '';
      const img  = hs
        ? `<img class="lb-player-hs" src="${hs}" alt="${name}" onerror="this.style.visibility='hidden'">`
        : `<div class="lb-player-hs lb-player-hs-empty"></div>`;
      return `<div class="lb-player-card">${img}<span class="lb-player-name">${name}</span></div>`;
    }).join('')}</div>` : '';

  const recentHtml = recent.map(m => {
    const d    = (m.delta >= 0 ? '+' : '') + parseFloat(m.delta).toFixed(2);
    const won  = m.result === 'W';
    const evt  = EVENT_LABELS[m.event_id] || '';
    const chips = mapChips(m.maps || [], org);
    const scoreParts = (m.score||'').split('-');
    const displayScore = (!won && scoreParts.length===2)
      ? scoreParts[1]+'-'+scoreParts[0] : m.score;
    return `<div class="lb-match-card ${won?'win':'loss'}">
      <div class="lb-match-head">
        <span class="lb-mr">${m.result}</span>
        <img class="lb-mlogo" src="/static/logos/${m.opponent}.png" onerror="this.style.display='none'" alt="">
        <span class="lb-mopp">vs ${m.opponent}</span>
        <span class="lb-mscore">${displayScore}</span>
        <span class="lb-mdelta">${d}</span>
      </div>
      <div class="lb-mmeta">${evt ? `<span>${evt}</span>` : ''}<span>${m.date}</span></div>
      ${chips ? `<div class="lb-mmaps">${chips}</div>` : ''}
    </div>`;
  }).join('');

  const mapsHtml = allMaps.length ? `
    <table class="lb-maps-table">
      <thead><tr><th>Map</th><th>Rating</th><th>W–L</th><th>Win%</th></tr></thead>
      <tbody>${allMaps.map(m => {
        const r   = (m.rating >= 0 ? '+' : '') + parseFloat(m.rating).toFixed(2);
        const tot = m.w + m.l;
        const pct = tot ? Math.round(100 * m.w / tot) + '%' : '—';
        const cls = m.rating >= 0 ? 'pos' : 'neg';
        const sid = 'mdr_' + org.replace(/[^a-z0-9]/gi,'_') + '_' + m.map.replace(/[^a-z0-9]/gi,'_');
        const eOrg = encodeURIComponent(org);
        const eMap = encodeURIComponent(m.map);
        return `<tr id="${sid}" class="lb-map-row-click" onclick="_expandMapRow('${eOrg}','${eMap}','${sid}')">
          <td class="lb-mt-map">${m.map}<span class="lb-map-chevron">▾</span></td>
          <td class="lb-mt-rat ${cls}">${r}</td>
          <td class="lb-mt-wl">${m.w}–${m.l}</td>
          <td class="lb-mt-pct">${pct}</td>
        </tr>`;
      }).join('')}</tbody>
    </table>` : '';

  return `<div class="lb-detail-inner">
    ${rosterHtml}
    ${recent.length ? `<div class="lb-sec-label">Recent Matches</div>${recentHtml}` : ''}
    ${mapsHtml ? `<div class="lb-sec-label">Map Breakdown</div>${mapsHtml}` : ''}
  </div>`;
}

function _expandMapRow(encOrg, encMap, rowId) {
  const org    = decodeURIComponent(encOrg);
  const map    = decodeURIComponent(encMap);
  const detId  = rowId + '_d';
  const tr     = document.getElementById(rowId);
  if (!tr) return;
  const existing = document.getElementById(detId);
  if (existing) {
    tr.classList.remove('open');
    const wrap = existing.querySelector('.lb-map-games-wrap');
    if (wrap) {
      wrap.classList.add('closing');
      setTimeout(() => existing.remove(), 200);
    } else {
      existing.remove();
    }
    return;
  }
  tr.classList.add('open');
  const events = (hubData?.chart?.match_events || []);
  const games  = events.filter(me =>
    (me.winner === org || me.loser === org) &&
    (me.maps  || []).some(m => m.map === map)
  ).sort((a, b) => (b.match_id || 0) - (a.match_id || 0));

  let innerHtml;
  if (!games.length) {
    innerHtml = `<td colspan="4" class="lb-map-no-games">No recorded games</td>`;
  } else {
    const rows = games.map(me => {
      const mInfo  = (me.maps || []).find(m => m.map === map);
      // W/L reflects the MAP outcome (not the series outcome) — a team can lose
      // the series but win this specific map. Falls back to series winner only
      // if per-map info is missing.
      const won    = mInfo ? (mInfo.winner === org) : (me.winner === org);
      const opp    = (me.winner === org) ? me.loser : me.winner;
      const orgRd  = mInfo ? (mInfo.winner === org ? mInfo.wr : mInfo.lr) : '?';
      const oppRd  = mInfo ? (mInfo.winner === org ? mInfo.lr : mInfo.wr) : '?';
      const diff   = (typeof orgRd === 'number' && typeof oppRd === 'number') ? orgRd - oppRd : null;
      const diffStr = diff !== null ? (diff >= 0 ? '+' : '') + diff : '';
      const diffCls = diff !== null ? (diff >= 0 ? 'pos' : 'neg') : '';
      const evt    = EVENT_LABELS[me.event_id] || '';
      return `<tr class="lb-map-game-row ${won?'win':'loss'}">
        <td colspan="4"><div class="lb-mg-inner">
          <span class="lb-mg-result">${won?'W':'L'}</span>
          <img class="lb-mg-logo" src="/static/logos/${opp}.png" onerror="this.style.display='none'" alt="">
          <span class="lb-mg-opp">${opp}</span>
          <span class="lb-mg-score">${orgRd}–${oppRd}</span>
          <span class="lb-mg-diff ${diffCls}">${diffStr}</span>
          <span class="lb-mg-meta">${me.date}${evt?' · '+evt:''}</span>
        </div></td>
      </tr>`;
    }).join('');
    innerHtml = `<td colspan="4"><div class="lb-map-games-wrap">
      <table class="lb-map-games-tbl">${rows}</table>
    </div></td>`;
  }
  const gamesTr = document.createElement('tr');
  gamesTr.id = detId;
  gamesTr.className = 'lb-map-games-tr';
  gamesTr.innerHTML = innerHtml;
  tr.after(gamesTr);
}

function toggleTeam(org) {
  const wasOpen = expandedOrg === org;
  if (wasOpen) {
    const det = document.querySelector('.lb-detail');
    expandedOrg  = null;
    selectedTeam = null;
    document.querySelectorAll('.lb-row.selected').forEach(r => r.classList.remove('selected'));
    if (det) {
      det.classList.add('closing');
      setTimeout(() => {
        if (hubData) { renderLeaderboard(hubData); buildChart(hubData); }
      }, 230);
      return;
    }
  }
  expandedOrg  = org;
  selectedTeam = org;
  if (hubData) {
    renderLeaderboard(hubData);
    buildChart(hubData);
  }
}

// ── Dot hover tooltip ────────────────────────────────────────────────────────
let _dotTooltipHideTimer = null;

function _matchTooltipHTML(m, won) {
  const org  = won ? m.winner : m.loser;
  const opp  = won ? m.loser  : m.winner;
  const d    = won ? m.winner_delta : m.loser_delta;
  const rat  = won ? m.winner_after  : m.loser_after;
  const dStr = (d >= 0 ? '+' : '') + d.toFixed(2);
  const evt  = EVENT_LABELS[m.event_id] || m.event_id || '';
  // Series score always shown as [org score]-[opp score]
  const rawParts = (m.series_score || '0-0').split('-');
  const displayScore = won
    ? m.series_score
    : `${rawParts[1]}-${rawParts[0]}`;

  const mapsRows = (m.maps || []).map(mp => {
    const mapWon = mp.winner === org;
    const orgRd = mapWon ? mp.wr : mp.lr;
    const oppRd = mapWon ? mp.lr : mp.wr;
    const diff   = orgRd - oppRd;
    return `<tr>
      <td class="popup-map-name">${mp.map}</td>
      <td class="popup-map-score ${mapWon?'w':'l'}">${orgRd}</td>
      <td class="popup-map-score ${mapWon?'l':'w'}">${oppRd}</td>
      <td class="popup-map-diff">${diff >= 0 ? '+' : ''}${diff}</td>
    </tr>`;
  }).join('');
  return `
    ${evt ? `<div class="popup-event-label">${evt}</div>` : ''}
    <div class="popup-teams">
      <div class="popup-team-block">
        <img class="popup-logo" src="/static/logos/${org}.png" onerror="this.style.display='none'" alt="${org}">
        <span class="popup-team-name">${org}</span>
      </div>
      <div class="popup-score-block">
        <span class="popup-score ${won?'w':'l'}">${displayScore}</span>
        <span class="popup-vs-label">series</span>
      </div>
      <div class="popup-team-block">
        <img class="popup-logo" src="/static/logos/${opp}.png" onerror="this.style.display='none'" alt="${opp}">
        <span class="popup-team-name">${opp}</span>
      </div>
    </div>
    <div class="popup-date">${m.date}</div>
    <div class="popup-delta ${d>=0?'pos':'neg'}">BenPom ${rat.toFixed(2)} &nbsp;(${dStr})</div>
    ${mapsRows ? `<table class="popup-maps-table">
      <thead><tr><th>Map</th><th>${org}</th><th>${opp}</th><th>Diff</th></tr></thead>
      <tbody>${mapsRows}</tbody>
    </table>` : ''}`;
}

function showDotTooltip(m, won, dotX, dotY) {
  const tt = document.getElementById('dotTooltip');
  document.getElementById('dotTooltipContent').innerHTML = _matchTooltipHTML(m, won);
  tt.style.visibility = 'hidden';
  tt.classList.add('visible');
  const wrap = document.getElementById('chartWrap');
  const ttW  = tt.offsetWidth  || 300;
  const ttH  = tt.offsetHeight || 300;
  const gap  = 20;
  let left = dotX - ttW / 2;
  let top  = dotY - ttH - gap;
  if (top < 4)                             top  = dotY + gap;  // flip below dot
  if (left < 4)                            left = 4;
  if (left + ttW > wrap.offsetWidth - 4)   left = wrap.offsetWidth - ttW - 4;
  tt.style.left       = left + 'px';
  tt.style.top        = top  + 'px';
  tt.style.visibility = '';
}

function hideDotTooltip() {
  document.getElementById('dotTooltip').classList.remove('visible');
}

// Legacy click popup (kept for non-hover contexts)
function showMatchPopup(m, won) {
  showDotTooltip(m, won, 0, 0);
}
function closePopup() { hideDotTooltip(); }

// ── Upcoming ─────────────────────────────────────────────────────────────────
function renderUpcoming(data) {
  var upcoming = (data.upcoming || []).slice().sort(function(a,b){
    return (a.date||'') < (b.date||'') ? -1 : (a.date||'') > (b.date||'') ? 1 : 0;
  });
  var body = document.getElementById('upcomingBody');
  if (!upcoming.length) {
    body.innerHTML = '<div class="no-upcoming">No upcoming matches found.<br><span style="font-size:.78rem;opacity:.6">Data updates on page load.</span></div>';
    return;
  }

  var lbTeams = {};
  (data.leaderboard.teams || []).forEach(function(t){ lbTeams[t.org] = t; });

  var snapKey = SNAP_KEY;
  var beta = SNAP_BETA;

  // Upcoming matches always use the live current pool, not the historical snap pool
  var pool = (VETO_HUB.current_pool && VETO_HUB.current_pool.length >= 7)
    ? VETO_HUB.current_pool
    : getActivePoolHUB(snapKey);
  if (!pool || !pool.length) {
    var seen = {};
    Object.values(SNAP_TEAMS).forEach(function(t){ Object.keys(t.maps||{}).forEach(function(m){ seen[m]=1; }); });
    pool = Object.keys(seen).sort();
  }
  if (!pool || !pool.length) {
    pool = ['Ascent','Bind','Breeze','Fracture','Haven','Lotus','Pearl','Split'];
  }

  var liveMapStats = VETO_HUB.live_map_stats || {};

  function getTeamObj(org) {
    var lb = lbTeams[org];
    var overall = lb ? lb.rating : 0;
    // Start with snapshot map ratings (if available), then overlay live Stage-1 data
    var maps = {};
    var st = SNAP_TEAMS[org];
    if (st && st.maps) {
      Object.keys(st.maps).forEach(function(mp){ maps[mp] = Object.assign({}, st.maps[mp]); });
    } else if (lb) {
      (lb.all_maps||[]).forEach(function(m){
        maps[m.map] = {rating:m.rating, w:m.w, l:m.l,
                       win_pct: m.w/Math.max(1,m.w+m.l)};
      });
    }
    // Overlay live Stage-1 win rates for veto-heuristic purposes (win_pct/w/l)
    // but DO NOT replace the rating — small-sample live ratings (e.g. a 3-0
    // record yielding +2.83) blow out the per-map probability calculation
    // and the historical matchup algorithm — which we want to match — only
    // uses the calibrated snapshot rating.
    var live = liveMapStats[org];
    if (live) {
      Object.keys(live).forEach(function(mp){
        var base = maps[mp] || {};
        var liveData = live[mp] || {};
        maps[mp] = {
          rating:  base.rating,                            // preserve snap rating
          w:       (liveData.w != null) ? liveData.w : base.w,
          l:       (liveData.l != null) ? liveData.l : base.l,
          win_pct: (liveData.win_pct != null) ? liveData.win_pct : base.win_pct,
        };
      });
    }
    if (!Object.keys(maps).length && !overall) return null;
    return {overall_rating: overall, maps: maps};
  }

  // 20000 sims: stderr ~0.35% per match — well below the rounding precision
  // shown to the user. Seeded RNG (_withSeededRand below) means each matchup's
  // prediction is byte-identical on every reload anyway.
  var nSims = 20000;
  var vetoSnapKey = '2026_'+snapKey;

  var REGION_CLS = {'EMEA':'rgn-emea','Americas':'rgn-americas','Pacific':'rgn-pacific','CN':'rgn-cn'};

  var cardHtmlArr = upcoming.map(function(m) {
    var orgA = m.org_a || m.team_a;
    var orgB = m.org_b || m.team_b;
    var matchFmt = m.format || 'bo3';
    var matchThresh = SERIES_THRESH_HUB[matchFmt] || 2;
    var tA = getTeamObj(orgA), tB = getTeamObj(orgB);
    var lbA = lbTeams[orgA], lbB = lbTeams[orgB];
    var ratingA = lbA ? lbA.rating : (tA ? (tA.overall_rating||0) : 0);
    var ratingB = lbB ? lbB.rating : (tB ? (tB.overall_rating||0) : 0);
    var region = m.region || (lbA ? lbA.region : '') || '';
    var rgnCls = REGION_CLS[region] || '';

    var seriesWins=0;
    var mapWins={}, mapPlays={};
    pool.forEach(function(mp){ mapWins[mp]=0; mapPlays[mp]=0; });

    if (tA && tB) {
      for (var s=0; s<nSims; s++) {
        var fm = simulateVetoHUB(tA,tB,orgA,orgB,pool,snapKey,matchFmt);
        var sw = 0;
        pool.forEach(function(mp){
          var fc = fm[mp] || 'banA';
          if (fc==='pickA'||fc==='pickB'||fc==='dec') {
            mapPlays[mp]++;
            var dA=(tA.maps&&tA.maps[mp]&&tA.maps[mp].rating!=null)?tA.maps[mp].rating:(tA.overall_rating||ratingA);
            var dB=(tB.maps&&tB.maps[mp]&&tB.maps[mp].rating!=null)?tB.maps[mp].rating:(tB.overall_rating||ratingB);
            var gA=getGlobalRatingHUB(orgA,vetoSnapKey,dA), gB=getGlobalRatingHUB(orgB,vetoSnapKey,dB);
            if (Math.random()<1/(1+Math.exp(-beta*(gA-gB)))) { sw++; mapWins[mp]++; }
          }
        });
        if (sw >= matchThresh) seriesWins++;
      }
    }

    // Always run the frontend MC sim — same model as the simulator iframe —
    // so the two surfaces give matching win probs. Fall back to a sigmoid on
    // live overalls only if a team isn't in the snap (no per-map data).
    var pctA = (tA&&tB) ? (seriesWins/nSims*100).toFixed(1)
                        : (100/(1+Math.exp(-beta*(ratingA-ratingB)))).toFixed(1);
    var pctB = (100 - parseFloat(pctA)).toFixed(1);
    var hasPatt = !!( ((VETO_HUB.teams||{})[vetoSnapKey]||{})[orgA] || ((VETO_HUB.teams||{})[vetoSnapKey]||{})[orgB] );
    var topSeqs = (tA&&tB&&pool.length) ? topVetoHUB(tA,tB,orgA,orgB,pool,snapKey,matchFmt,1) : [];

    // Played maps sorted by frequency
    var playedMaps = pool.filter(function(mp){ return mapPlays[mp]>0; })
                        .sort(function(a,b){ return mapPlays[b]-mapPlays[a]; });

    // Build veto sequences section
    var vetoSeqsHtml = '';
    if (hasPatt && topSeqs.length) {
      var sq = topSeqs[0];
      var seqRow = sq.seq.map(function(step, idx){
        var key = step.action+step.side;
        var cls = (ACTION_CLS[key]||ACTION_CLS.dec)[0];
        var lbl = actionLabelHUB(orgA, orgB, key);
        return '<div class="upc-veto-step">'+
          '<span class="step-lbl '+cls+'">'+lbl+'</span>'+
          '<span class="upc-veto-map">'+step.map+'</span>'+
        '</div>'+(idx<sq.seq.length-1?'<span class="step-arrow">›</span>':'');
      }).join('');
      vetoSeqsHtml = '<div class="upc-section-lbl">Predicted Veto</div>'+
        '<div class="upc-veto-seqs"><div class="upc-veto-seq-row">'+seqRow+'</div></div>';
    }

    // Build map breakdown table
    var mapTableHtml = '';
    if (playedMaps.length) {
      var totalSims = nSims;
      var vetoSeqForMap = {};
      if (topSeqs.length) {
        topSeqs[0].seq.forEach(function(step){
          vetoSeqForMap[step.map] = step.action + step.side;
        });
      }
      mapTableHtml = '<div class="upc-section-lbl">Map Breakdown</div>'+
        '<table class="upc-map-table">'+
        '<thead><tr>'+
          '<th>Map</th>'+
          '<th>Played</th>'+
          '<th>'+orgA+'</th>'+
          '<th>'+orgB+'</th>'+
          (Object.keys(vetoSeqForMap).length?'<th>Veto</th>':'')+
        '</tr></thead><tbody>';
      playedMaps.forEach(function(mp){
        var wp = mapWins[mp]/mapPlays[mp];
        var wpPctA = Math.round(wp*100);
        var wpPctB = 100-wpPctA;
        var clsA = wp>=0.55?'fav':(wp<=0.45?'dog':'neu');
        var clsB = (1-wp)>=0.55?'fav':((1-wp)<=0.45?'dog':'neu');
        var playedPct = Math.round(mapPlays[mp]/totalSims*100);
        var vetoKey = vetoSeqForMap[mp] || '';
        var vetoLbl = vetoKey ? actionLabelHUB(orgA, orgB, vetoKey) : '';
        var vetoCls = vetoKey ? (ACTION_CLS[vetoKey]||ACTION_CLS.dec)[0] : '';
        mapTableHtml += '<tr>'+
          '<td><img src="/maps/'+mp.toLowerCase()+'.png" style="width:20px;height:14px;object-fit:cover;border-radius:2px;vertical-align:middle;margin-right:5px" onerror="this.style.display=\\'none\\'">'+mp+'</td>'+
          '<td style="color:#888;font-size:.65rem">'+playedPct+'%</td>'+
          '<td class="upc-map-td-wp '+clsA+'">'+wpPctA+'%</td>'+
          '<td class="upc-map-td-wp '+clsB+'">'+wpPctB+'%</td>'+
          (Object.keys(vetoSeqForMap).length?'<td class="upc-map-td-veto">'+(vetoLbl?'<span class="step-lbl '+vetoCls+'">'+vetoLbl+'</span>':'—')+'</td>':'')+
        '</tr>';
      });
      mapTableHtml += '</tbody></table>';
    }

    // Recent form — 3 past matches per team
    function recentMatchesHtml(org, side) {
      var lb = lbTeams[org];
      var recent = lb ? (lb.recent_matches || []) : [];
      recent = recent.slice(0, 3);
      if (!recent.length) return '<div style="color:#aaa;font-size:.68rem">No data</div>';
      return recent.map(function(r){
        var resultCls = r.result==='W' ? 'w' : 'l';
        var dateStr = r.date ? new Date(r.date+'T12:00:00').toLocaleDateString('en-US',{month:'short',day:'numeric'}) : '';
        // Score is always winner-first; flip it for losses so team's score is always on the left
        var scoreParts = (r.score||'').split('-');
        var displayScore = (r.result==='L' && scoreParts.length===2)
          ? scoreParts[1]+'-'+scoreParts[0] : r.score;
        return '<div class="upc-recent-match">'+
          '<span class="upc-recent-result '+resultCls+'">'+r.result+'</span>'+
          '<span class="upc-recent-opp">vs '+r.opponent+'</span>'+
          '<span class="upc-recent-score">'+displayScore+'</span>'+
          '<span class="upc-recent-evt">'+dateStr+'</span>'+
        '</div>';
      }).join('');
    }
    var recentHtml = '<div class="upc-section-lbl">Recent Form</div>'+
      '<div class="upc-recent-row">'+
        '<div class="upc-recent-col">'+
          '<div class="upc-recent-col-hdr">'+orgA+'</div>'+
          recentMatchesHtml(orgA,'a')+
        '</div>'+
        '<div class="upc-recent-col">'+
          '<div class="upc-recent-col-hdr">'+orgB+'</div>'+
          recentMatchesHtml(orgB,'b')+
        '</div>'+
      '</div>';

    var dateLabel = m.date ? new Date(m.date+'T12:00:00').toLocaleDateString('en-US',{month:'short',day:'numeric'}) : '';
    var rtgA = '<span class="upc-rtg">'+(ratingA>=0?'+':'')+ratingA.toFixed(2)+'</span>';
    var rtgB = '<span class="upc-rtg">'+(ratingB>=0?'+':'')+ratingB.toFixed(2)+'</span>';
    var fmtLabel = matchFmt==='bo5'?'Bo5':matchFmt==='bo1'?'Bo1':'Bo3';

    return '<div class="upc-card '+rgnCls+'">'+
      '<div class="upc-header">'+
        '<div class="upc-team-a">'+
          '<img class="upc-logo" src="/static/logos/'+orgA+'.png" onerror="this.style.opacity=\\'0\\'">'+
          '<span class="upc-org">'+orgA+'</span>'+
          rtgA+
        '</div>'+
        '<div class="upc-center">'+
          '<div class="upc-date-event">'+dateLabel+(m.event?' · '+m.event:'')+' · '+fmtLabel+'</div>'+
          '<div class="upc-bar-wrap">'+
            '<div class="upc-bar-a" style="width:'+pctA+'%"></div>'+
            '<div class="upc-bar-b" style="width:'+pctB+'%"></div>'+
          '</div>'+
          '<div class="upc-pcts">'+
            '<span class="upc-pct '+(pctA>=50?'fav':'dog')+'">'+pctA+'%</span>'+
            '<span class="upc-pct '+(pctB>=50?'fav':'dog')+'">'+pctB+'%</span>'+
          '</div>'+
        '</div>'+
        '<div class="upc-team-b">'+
          '<img class="upc-logo" src="/static/logos/'+orgB+'.png" onerror="this.style.opacity=\\'0\\'">'+
          '<span class="upc-org">'+orgB+'</span>'+
          rtgB+
        '</div>'+
      '</div>'+
      '<div class="upc-details">'+
        '<div class="upc-details-inner">'+
          (vetoSeqsHtml || '')+
          (mapTableHtml || '')+
          recentHtml+
        '</div>'+
      '</div>'+
      '<div class="upc-expand-hint">▸ expand</div>'+
    '</div>';
  });

  // Group by date and render
  var groups = [];
  var curDate = null;
  upcoming.forEach(function(m, i) {
    var d = m.date || '';
    if (d !== curDate) { groups.push({date: d, indices: []}); curDate = d; }
    groups[groups.length-1].indices.push(i);
  });
  var groupedHtml = groups.map(function(g) {
    var dateLabel = g.date ? new Date(g.date+'T12:00:00').toLocaleDateString('en-US',{weekday:'long',month:'long',day:'numeric'}) : '';
    return '<div class="upc-day-group">'+
      '<div class="upc-day-label">'+dateLabel+'</div>'+
      g.indices.map(function(i){ return cardHtmlArr[i]; }).join('')+
    '</div>';
  }).join('');
  body.innerHTML = '<div class="upc-list">'+groupedHtml+'</div>';

  body.querySelectorAll('.upc-card').forEach(function(card) {
    card.addEventListener('click', function() {
      card.classList.toggle('open');
      var hint = card.querySelector('.upc-expand-hint');
      if (hint) hint.textContent = card.classList.contains('open') ? '▾ collapse' : '▸ expand';
    });
  });

  triggerUpcomingFlyIn();
}

// ── Past matches ─────────────────────────────────────────────────────────────
function renderPast(data) {
  var past = (data.past_matches || []).slice().sort(function(a,b){
    return (b.date||'').localeCompare(a.date||'');
  });
  var body = document.getElementById('pastBody');
  if (!past.length) {
    body.innerHTML = '<div class="no-upcoming">No matches played in the past 7 days.</div>';
    triggerPastFlyIn();
    return;
  }

  // Reuse leaderboard like upcoming, for dropdown veto/map sim
  var snapKey   = data.snap_key || 'live';
  var lbTeams = {};
  ((data.leaderboard||{}).teams || []).forEach(function(t){ lbTeams[t.org] = t; });
  // Same CV-optimal β as the upcoming-card and simulator sims — keep all three
  // surfaces in sync so they produce identical win-prob predictions.
  var beta = 0.25;
  var livePool = ((data.snapshots||{})[snapKey] || {}).current_pool
              || ['Abyss','Bind','Haven','Lotus','Split','Sunset','Ascent'];
  var liveMapStats = (typeof VETO_HUB!=='undefined' && VETO_HUB.live_map_stats) || {};
  var REGION_CLS = {'EMEA':'rgn-emea','Americas':'rgn-americas','Pacific':'rgn-pacific','CN':'rgn-cn'};
  // Match the upcoming-card sim count (20000) so recent matches and upcoming
  // matches have the same statistical precision.
  var nSims = 20000;

  // Per-match pool: each regional league runs its own 7-map pool, so prefer
  // event+region match, then region, then event-wide fallback.
  var eventPools       = data.past_event_pools || {};
  var regionPools      = data.past_region_pools || {};
  var regionEventPools = data.past_region_event_pools || {};
  function getMatchPool(m) {
    var evId   = m.event_id || '';
    var region = m.region   || '';
    var p = regionEventPools[evId + ':' + region];
    if (p && p.length) return p;
    p = regionPools[region];
    if (p && p.length) return p;
    p = eventPools[evId];
    if (p && p.length) return p;
    p = (VETO_HUB.computed_pools||{})[evId];
    if (!p || !p.length) p = (VETO_HUB.snap_pools||{})[evId];
    if (p && p.length) return p;
    return livePool;
  }

  function getTeamObj(org) {
    var lb = lbTeams[org];
    var overall = lb ? lb.rating : 0;
    var maps = {};
    var st = (typeof SNAP_TEAMS!=='undefined') ? SNAP_TEAMS[org] : null;
    if (st && st.maps) {
      Object.keys(st.maps).forEach(function(mp){ maps[mp] = Object.assign({}, st.maps[mp]); });
    } else if (lb) {
      (lb.all_maps||[]).forEach(function(mm){
        maps[mm.map] = {rating:mm.rating, w:mm.w, l:mm.l, win_pct: mm.w/Math.max(1,mm.w+mm.l)};
      });
    }
    // Overlay live win%/w/l only — preserve the calibrated snap rating so the
    // per-map sim matches the historical matchup algorithm (no small-sample
    // rating extremes).
    var live = liveMapStats[org];
    if (live) {
      Object.keys(live).forEach(function(mp){
        var base = maps[mp] || {};
        var ld   = live[mp] || {};
        maps[mp] = {
          rating:  base.rating,
          w:       (ld.w != null) ? ld.w : base.w,
          l:       (ld.l != null) ? ld.l : base.l,
          win_pct: (ld.win_pct != null) ? ld.win_pct : base.win_pct,
        };
      });
    }
    if (!Object.keys(maps).length && !overall) return null;
    return {overall_rating: overall, maps: maps};
  }

  function buildCard(m) {
    var orgA = m.org_a || m.team_a;
    var orgB = m.org_b || m.team_b;
    var matchFmt = m.format || 'bo3';
    var matchThresh = SERIES_THRESH_HUB[matchFmt] || 2;
    var tA = getTeamObj(orgA), tB = getTeamObj(orgB);
    var ratingA = (m.rating_a != null) ? m.rating_a : 0;
    var ratingB = (m.rating_b != null) ? m.rating_b : 0;
    var region = m.region || '';
    var rgnCls = REGION_CLS[region] || '';

    // Pool comes from the match's event/region, but veto-pattern data is only
    // stored under snapshot keys like "2026_after_santiago" — fall back to the
    // live snap key (which IS that snapshot for current matches).
    var pool = getMatchPool(m);
    var vetoSnapKey = '2026_' + (data.snap_key || 'live');

    var seriesWins=0;
    var mapWins={}, mapPlays={};
    pool.forEach(function(mp){ mapWins[mp]=0; mapPlays[mp]=0; });

    // Per-map sim mirrors the historical matchup algorithm exactly: raw map
    // rating → intl global rating → sigmoid with beta.
    if (tA && tB) {
      for (var s=0; s<nSims; s++) {
        var fm = simulateVetoHUB(tA,tB,orgA,orgB,pool,vetoSnapKey,matchFmt);
        var sw = 0;
        pool.forEach(function(mp){
          var fc = fm[mp] || 'banA';
          if (fc==='pickA'||fc==='pickB'||fc==='dec') {
            mapPlays[mp]++;
            var dA=(tA.maps&&tA.maps[mp]&&tA.maps[mp].rating!=null)?tA.maps[mp].rating:(tA.overall_rating||ratingA);
            var dB=(tB.maps&&tB.maps[mp]&&tB.maps[mp].rating!=null)?tB.maps[mp].rating:(tB.overall_rating||ratingB);
            var gA=getGlobalRatingHUB(orgA,vetoSnapKey,dA), gB=getGlobalRatingHUB(orgB,vetoSnapKey,dB);
            if (Math.random()<1/(1+Math.exp(-beta*(gA-gB)))) { sw++; mapWins[mp]++; }
          }
        });
        if (sw >= matchThresh) seriesWins++;
      }
    }

    // Projected probabilities — use the morning-of value from backend (m.win_prob_*)
    var pctA = (m.win_prob_a != null) ? (m.win_prob_a*100).toFixed(1)
             : (100/(1+Math.exp(-beta*(ratingA-ratingB)))).toFixed(1);
    var pctB = (100 - parseFloat(pctA)).toFixed(1);

    var hasPatt = !!( ((VETO_HUB.teams||{})[vetoSnapKey]||{})[orgA] || ((VETO_HUB.teams||{})[vetoSnapKey]||{})[orgB] );
    var topSeqs = (tA&&tB&&pool.length) ? topVetoHUB(tA,tB,orgA,orgB,pool,vetoSnapKey,matchFmt,1) : [];
    var playedMaps = pool.filter(function(mp){ return mapPlays[mp]>0; })
                        .sort(function(a,b){ return mapPlays[b]-mapPlays[a]; });

    var vetoSeqsHtml = '';
    if (hasPatt && topSeqs.length) {
      var sq = topSeqs[0];
      var seqRow = sq.seq.map(function(step, idx){
        var key = step.action+step.side;
        var cls = (ACTION_CLS[key]||ACTION_CLS.dec)[0];
        var lbl = actionLabelHUB(orgA, orgB, key);
        return '<div class="upc-veto-step">'+
          '<span class="step-lbl '+cls+'">'+lbl+'</span>'+
          '<span class="upc-veto-map">'+step.map+'</span>'+
        '</div>'+(idx<sq.seq.length-1?'<span class="step-arrow">›</span>':'');
      }).join('');
      vetoSeqsHtml = '<div class="upc-section-lbl">Predicted Veto</div>'+
        '<div class="upc-veto-seqs"><div class="upc-veto-seq-row">'+seqRow+'</div></div>';
    }

    // Actual maps played — shown alongside model breakdown for past matches
    var actualMapsHtml = '';
    if (m.maps_played && m.maps_played.length) {
      actualMapsHtml = '<div class="upc-section-lbl">Maps Played (Result)</div>'+
        '<table class="upc-map-table"><thead><tr>'+
          '<th>Map</th><th>Winner</th><th>Score</th>'+
        '</tr></thead><tbody>'+
        m.maps_played.map(function(mp){
          var winLbl = mp.winner || '';
          var score  = (mp.wr != null && mp.lr != null) ? (mp.wr+'-'+mp.lr) : '';
          var winCls = winLbl===orgA ? 'fav' : (winLbl===orgB ? 'dog' : 'neu');
          return '<tr>'+
            '<td><img src="/maps/'+(mp.map||'').toLowerCase()+'.png" style="width:20px;height:14px;object-fit:cover;border-radius:2px;vertical-align:middle;margin-right:5px" onerror="this.style.display=\\'none\\'">'+(mp.map||'')+'</td>'+
            '<td class="upc-map-td-wp '+winCls+'">'+winLbl+'</td>'+
            '<td style="color:#444;font-size:.7rem">'+score+'</td>'+
          '</tr>';
        }).join('')+
        '</tbody></table>';
    }

    var mapTableHtml = '';
    if (playedMaps.length) {
      var totalSims = nSims;
      var vetoSeqForMap = {};
      if (topSeqs.length) {
        topSeqs[0].seq.forEach(function(step){
          vetoSeqForMap[step.map] = step.action + step.side;
        });
      }
      mapTableHtml = '<div class="upc-section-lbl">Model&rsquo;s Map Projection</div>'+
        '<table class="upc-map-table">'+
        '<thead><tr>'+
          '<th>Map</th><th>Played</th><th>'+orgA+'</th><th>'+orgB+'</th>'+
          (Object.keys(vetoSeqForMap).length?'<th>Veto</th>':'')+
        '</tr></thead><tbody>';
      playedMaps.forEach(function(mp){
        var wp = mapWins[mp]/mapPlays[mp];
        var wpPctA = Math.round(wp*100);
        var wpPctB = 100-wpPctA;
        var clsA = wp>=0.55?'fav':(wp<=0.45?'dog':'neu');
        var clsB = (1-wp)>=0.55?'fav':((1-wp)<=0.45?'dog':'neu');
        var playedPct = Math.round(mapPlays[mp]/totalSims*100);
        var vetoKey = vetoSeqForMap[mp] || '';
        var vetoLbl = vetoKey ? actionLabelHUB(orgA, orgB, vetoKey) : '';
        var vetoCls = vetoKey ? (ACTION_CLS[vetoKey]||ACTION_CLS.dec)[0] : '';
        mapTableHtml += '<tr>'+
          '<td><img src="/maps/'+mp.toLowerCase()+'.png" style="width:20px;height:14px;object-fit:cover;border-radius:2px;vertical-align:middle;margin-right:5px" onerror="this.style.display=\\'none\\'">'+mp+'</td>'+
          '<td style="color:#888;font-size:.65rem">'+playedPct+'%</td>'+
          '<td class="upc-map-td-wp '+clsA+'">'+wpPctA+'%</td>'+
          '<td class="upc-map-td-wp '+clsB+'">'+wpPctB+'%</td>'+
          (Object.keys(vetoSeqForMap).length?'<td class="upc-map-td-veto">'+(vetoLbl?'<span class="step-lbl '+vetoCls+'">'+vetoLbl+'</span>':'—')+'</td>':'')+
        '</tr>';
      });
      mapTableHtml += '</tbody></table>';
    }

    // For past matches, "Recent Form" should reflect what the model knew at
    // the time — i.e., each team's three matches BEFORE this one — not their
    // globally most-recent matches today.
    function recentMatchesHtml(org) {
      var cutoff = m.date || '';
      var evts = (data.chart && data.chart.match_events) ? data.chart.match_events : [];
      var teamMatches = [];
      for (var i = evts.length - 1; i >= 0; i--) {
        var ev = evts[i];
        if (!ev || !ev.date) continue;
        if (cutoff && ev.date >= cutoff) continue;   // strictly before this match
        if (ev.winner !== org && ev.loser !== org) continue;
        var isWin = (ev.winner === org);
        teamMatches.push({
          date:     ev.date,
          opponent: isWin ? ev.loser : ev.winner,
          result:   isWin ? 'W' : 'L',
          score:    ev.series_score || '',
        });
        if (teamMatches.length >= 3) break;
      }
      if (!teamMatches.length) return '<div style="color:#aaa;font-size:.68rem">No data</div>';
      return teamMatches.map(function(r){
        var resultCls = r.result==='W' ? 'w' : 'l';
        var dateStr = r.date ? new Date(r.date+'T12:00:00').toLocaleDateString('en-US',{month:'short',day:'numeric'}) : '';
        var scoreParts = (r.score||'').split('-');
        var displayScore = (r.result==='L' && scoreParts.length===2)
          ? scoreParts[1]+'-'+scoreParts[0] : r.score;
        return '<div class="upc-recent-match">'+
          '<span class="upc-recent-result '+resultCls+'">'+r.result+'</span>'+
          '<span class="upc-recent-opp">vs '+r.opponent+'</span>'+
          '<span class="upc-recent-score">'+displayScore+'</span>'+
          '<span class="upc-recent-evt">'+dateStr+'</span>'+
        '</div>';
      }).join('');
    }
    var recentHtml = '<div class="upc-section-lbl">Recent Form (Before This Match)</div>'+
      '<div class="upc-recent-row">'+
        '<div class="upc-recent-col"><div class="upc-recent-col-hdr">'+orgA+'</div>'+recentMatchesHtml(orgA)+'</div>'+
        '<div class="upc-recent-col"><div class="upc-recent-col-hdr">'+orgB+'</div>'+recentMatchesHtml(orgB)+'</div>'+
      '</div>';

    var dateLabel = m.date ? new Date(m.date+'T12:00:00').toLocaleDateString('en-US',{month:'short',day:'numeric'}) : '';
    var rtgA = '<span class="upc-rtg">'+(ratingA>=0?'+':'')+ratingA.toFixed(2)+'</span>';
    var rtgB = '<span class="upc-rtg">'+(ratingB>=0?'+':'')+ratingB.toFixed(2)+'</span>';
    var fmtLabel = matchFmt==='bo5'?'Bo5':matchFmt==='bo1'?'Bo1':'Bo3';

    // Result strip — actual winner + score. Upset if the underdog won.
    var winnerOrg = m.actual_winner === 'a' ? orgA : orgB;
    var winnerPct = m.actual_winner === 'a' ? parseFloat(pctA) : parseFloat(pctB);
    var isUpset = winnerPct < 50;
    var resultHtml = '<div class="upc-result-strip">'+
      '<span class="upc-result-label">Final</span>'+
      '<span class="upc-result-score">'+orgA+' '+(m.actual_score||'')+' '+orgB+'</span>'+
      '<span class="upc-result-winner'+(isUpset?' upc-result-upset':'')+'">'+winnerOrg+(isUpset?' &middot; upset':' wins')+'</span>'+
    '</div>';

    return '<div class="upc-card '+rgnCls+'">'+
      '<div class="upc-header">'+
        '<div class="upc-team-a">'+
          '<img class="upc-logo" src="/static/logos/'+orgA+'.png" onerror="this.style.opacity=\\'0\\'">'+
          '<span class="upc-org">'+orgA+'</span>'+rtgA+
        '</div>'+
        '<div class="upc-center">'+
          '<div class="upc-date-event">'+dateLabel+(m.event?' · '+m.event:'')+' · '+fmtLabel+'</div>'+
          '<div class="upc-pre-label">Pre-match projection</div>'+
          '<div class="upc-bar-wrap">'+
            '<div class="upc-bar-a" style="width:'+pctA+'%"></div>'+
            '<div class="upc-bar-b" style="width:'+pctB+'%"></div>'+
          '</div>'+
          '<div class="upc-pcts">'+
            '<span class="upc-pct '+(pctA>=50?'fav':'dog')+'">'+pctA+'%</span>'+
            '<span class="upc-pct '+(pctB>=50?'fav':'dog')+'">'+pctB+'%</span>'+
          '</div>'+
          resultHtml+
        '</div>'+
        '<div class="upc-team-b">'+
          '<img class="upc-logo" src="/static/logos/'+orgB+'.png" onerror="this.style.opacity=\\'0\\'">'+
          '<span class="upc-org">'+orgB+'</span>'+rtgB+
        '</div>'+
      '</div>'+
      '<div class="upc-details">'+
        '<div class="upc-details-inner">'+
          (actualMapsHtml || '')+
          (vetoSeqsHtml || '')+
          (mapTableHtml || '')+
          recentHtml+
        '</div>'+
      '</div>'+
      '<div class="upc-expand-hint">▸ expand</div>'+
    '</div>';
  }

  // ── Progressive render ─────────────────────────────────────────────────
  // Build the day-group frames synchronously (so the tab switch is instant),
  // then process each match's 20k-sim MC one at a time via setTimeout. Each
  // card gets inserted + slid in from the right as its sim completes.
  // Result: no main-thread block on tab switch; the user sees cards stream in.
  var groups = [];
  var curDate = null;
  past.forEach(function(m, i) {
    if (m.date !== curDate) {
      curDate = m.date;
      groups.push({date: m.date, indices: []});
    }
    groups[groups.length-1].indices.push(i);
  });
  // Sanitize date for use as a CSS id (no spaces / weird chars expected, but safe).
  function _groupId(d) { return 'past-group-' + (d || 'undated').replace(/[^a-zA-Z0-9_-]/g, '_'); }
  var groupedHtml = groups.map(function(g) {
    var dateLabel = g.date ? new Date(g.date+'T12:00:00').toLocaleDateString('en-US',{weekday:'long',month:'long',day:'numeric'}) : '';
    return '<div class="upc-day-group" id="'+_groupId(g.date)+'">'+
      '<div class="upc-day-label">'+dateLabel+'</div>'+
      '<div class="upc-day-cards"></div>'+
    '</div>';
  }).join('');
  body.innerHTML = '<div class="upc-list">'+groupedHtml+'</div>';

  // Heading fly-in fires immediately (cheap, no MC). The list-level fly-in is
  // skipped — we handle per-card reveal below.
  triggerPastFlyIn();

  function _attachCardHandler(card) {
    card.addEventListener('click', function() {
      card.classList.toggle('open');
      var hint = card.querySelector('.upc-expand-hint');
      if (hint) hint.textContent = card.classList.contains('open') ? '▾ collapse' : '▸ expand';
    });
  }

  function processMatch(idx) {
    if (idx >= past.length) return;
    var m = past[idx];
    var cardHtml = buildCard(m);  // 20k sims for this one match
    var groupEl = document.getElementById(_groupId(m.date));
    if (groupEl) {
      var container = groupEl.querySelector('.upc-day-cards');
      if (container) {
        var wrap = document.createElement('div');
        wrap.innerHTML = cardHtml;
        var newCard = wrap.firstElementChild;
        if (newCard) {
          container.appendChild(newCard);
          _attachCardHandler(newCard);
          // Trigger slide-in on the next frame (after the browser sees the
          // initial opacity:0 / translateX(80) state). The double-rAF avoids
          // races where the class is added before the initial state paints.
          requestAnimationFrame(function() {
            requestAnimationFrame(function() {
              newCard.classList.add('card-loaded');
            });
          });
        }
      }
    }
    // Yield to the browser before kicking off the next match's sim, so the
    // slide-in animation actually paints and the UI stays responsive.
    setTimeout(function(){ processMatch(idx + 1); }, 0);
  }
  processMatch(0);
}

// ── Match Simulator (iframes the historical matchup tool, locked to current) ─
// The iframe is preloaded in the background after init() — see preloadSimulator() —
// so by the time the user clicks the tab, the matchup tool is already rendered
// and the slide animation has nothing competing for the main thread.
var _simInitialized = false;
function renderSimulator() {
  if (_simInitialized) return;
  _simInitialized = true;
  var f = document.getElementById('simIframe');
  if (f && f.src.indexOf('lockCurrent=1') < 0) f.src = '/mapelo/matchup/?lockCurrent=1';
}
function preloadSimulator() {
  // Idle-time preload after main UI is settled. requestIdleCallback gives us
  // a chunk of free main-thread time; falls back to setTimeout where unsupported.
  var fire = function(){ renderSimulator(); };
  if (window.requestIdleCallback) {
    window.requestIdleCallback(fire, {timeout: 3000});
  } else {
    setTimeout(fire, 600);
  }
}
function triggerSimFlyIn(){}

// ── Letter-by-letter fly-in for Upcoming Matches panel ───────────────────────
function _splitIntoChars(el) {
  if (!el || el.dataset.split === '1') return;
  var text = el.textContent;
  el.textContent = '';
  for (var i = 0; i < text.length; i++) {
    var ch = text[i];
    var span = document.createElement('span');
    span.className = 'fly-char';
    span.textContent = ch === ' ' ? ' ' : ch;
    el.appendChild(span);
  }
  el.dataset.split = '1';
}
function _flyInPanel(panelSel, headingSel, subSel, opts) {
  // opts.skipList=true: animate heading/sub but NOT the .upc-list. Used by
  // renderPast which inserts cards one-at-a-time and reveals each as its
  // MC sim completes (per-card .card-loaded class, not a list-wide fly-in).
  opts = opts || {};
  var skipList = !!opts.skipList;
  var root    = document.querySelector(panelSel);
  if (!root) return;
  var heading = root.querySelector(headingSel);
  var sub     = root.querySelector(subSel);
  var list    = skipList ? null : root.querySelector('.upc-list');
  _splitIntoChars(heading);
  _splitIntoChars(sub);

  // Reset all state — strip transition classes
  [heading, sub, list].forEach(function(el){
    if (!el) return;
    el.classList.remove('fly-in', 'flying', 'anim-done');
  });

  var hChars = heading ? heading.querySelectorAll('.fly-char') : [];
  var sChars = sub ? sub.querySelectorAll('.fly-char') : [];
  var cards  = list ? list.querySelectorAll('.upc-card') : [];

  // Snap every animated element back to its initial state INSTANTLY by
  // disabling transitions. Without this, on a replay the elements are
  // still partway through a "go back to start" transition when we
  // re-trigger, so the animation looks like it skips.
  function _snapReset(el){ el.style.transition = 'none'; }
  hChars.forEach(_snapReset);
  sChars.forEach(_snapReset);
  cards.forEach(_snapReset);
  // Force reflow so the no-transition state actually paints before we
  // restore transitions and re-trigger.
  void document.body.offsetWidth;

  // Restore transitions + assign per-element stagger delays
  hChars.forEach(function(c, i) {
    c.style.transition = '';
    c.style.transitionDelay = (i * 35) + 'ms';
  });
  var headingDur = hChars.length * 35 + 450;

  sChars.forEach(function(c, i) {
    c.style.transition = '';
    c.style.transitionDelay = (headingDur * 0.4 + i * 14) + 'ms';
  });
  var subDur = headingDur * 0.4 + sChars.length * 14 + 350;

  // Cap the cascade — beyond ~20 cards in flight at once, the GPU starts
  // dropping frames. Stagger small for the first 12, snap the rest in fast.
  cards.forEach(function(c, i) {
    c.style.transition = '';
    var delay = (subDur * 0.55) + (i < 12 ? i * 50 : 12 * 50 + (i - 12) * 18);
    c.style.transitionDelay = delay + 'ms';
  });
  var totalDur = (subDur * 0.55) + (cards.length < 12 ? cards.length * 50 : 12 * 50 + (cards.length - 12) * 18) + 600;

  // Apply will-change for the flight, then strip it once the animation is done
  if (heading) heading.classList.add('flying');
  if (sub)     sub.classList.add('flying');
  requestAnimationFrame(function() {
    if (heading) heading.classList.add('fly-in');
    if (sub)     sub.classList.add('fly-in');
    if (list)    list.classList.add('fly-in');
  });
  setTimeout(function(){
    if (heading) heading.classList.remove('flying');
    if (sub)     sub.classList.remove('flying');
    if (list)    list.classList.add('anim-done');
  }, totalDur);
}
function triggerUpcomingFlyIn() {
  _flyInPanel('#panelB', '.upcoming-heading', '.upcoming-sub');
}
function triggerPastFlyIn() {
  // skipList: cards are added + revealed individually via the progressive-load
  // path in renderPast (.card-loaded class per match). The list-wide fly-in
  // would snap everything visible immediately.
  _flyInPanel('#panelC', '.past-heading', '.past-sub', {skipList: true});
}

// ── Boot ─────────────────────────────────────────────────────────────────────
init();
</script>
</body>
</html>
"""

@mapelo_bp.route('/modern/')
def mapelo_modern():
    return MAPELO_MODERN_HTML

@mapelo_bp.route('/modern/data')
def mapelo_modern_data():
    data = _mhub_get()
    return Response(json.dumps(data), mimetype='application/json')

@mapelo_bp.route('/modern/progress')
def mapelo_modern_progress():
    """Surface the live refresh progress + stderr tail so operators can diagnose
    Render scraping problems without shell access."""
    payload = {"progress": None, "stderr_tail": ""}
    try:
        if os.path.exists(_MHUB_PROGRESS_FILE):
            with open(_MHUB_PROGRESS_FILE) as f:
                payload["progress"] = json.load(f)
    except Exception as e:
        payload["progress_error"] = str(e)
    try:
        if os.path.exists(_MHUB_STDERR_FILE):
            with open(_MHUB_STDERR_FILE) as f:
                payload["stderr_tail"] = f.read()[-4000:]
    except Exception as e:
        payload["stderr_error"] = str(e)
    payload["build_running"] = _mhub_build_running
    return Response(json.dumps(payload, indent=2), mimetype='application/json')


@mapelo_bp.route('/modern/refresh')
def mapelo_modern_refresh():
    """Force-trigger a refresh (bypasses cooldown).  Poll /modern/progress."""
    with _mhub_cache_lock:
        _mhub_cache["ts"] = 0.0
    _mhub_trigger_build(force=True)
    return Response(json.dumps({"triggered": True}), mimetype='application/json')


@mapelo_bp.route('/modern/run-sync')
def mapelo_modern_run_sync():
    """
    Run RefreshLiveData synchronously inside the request, capture EVERYTHING,
    return it.  No subprocess, no /tmp writes, no race conditions — the only
    diagnostic that can't be invisibly swallowed.

    Useful for: confirming the scrape itself works on this host, seeing exactly
    where it bails when it doesn't, and verifying that data files get written
    (or proving the filesystem is read-only).
    """
    import sys as _sys, traceback as _tb, io as _io, time as _time
    if ROOT not in _sys.path:
        _sys.path.insert(0, ROOT)

    out = {
        "started":  _time.time(),
        "env":      {
            "RENDER":            os.environ.get("RENDER"),
            "RENDER_SERVICE_ID": os.environ.get("RENDER_SERVICE_ID"),
            "cwd":               os.getcwd(),
            "ROOT":              ROOT,
            "python":            _sys.executable,
        },
        "writable": {},
        "data_files": {},
        "scrape_log": None,
        "after_progress": None,
        "error": None,
    }

    # Probe filesystem writability
    for path in ("/tmp", "/tmp/mhub_test.txt",
                 os.path.join(ROOT, "data"),
                 os.path.join(ROOT, "data", "mhub_test.txt")):
        try:
            if path.endswith(".txt"):
                with open(path, "w") as f:
                    f.write("ok")
                os.remove(path)
                out["writable"][path] = "ok"
            else:
                out["writable"][path] = ("exists" if os.path.exists(path) else "missing") + \
                    (" / writable" if os.access(path, os.W_OK) else " / NOT writable")
        except Exception as e:
            out["writable"][path] = f"err: {type(e).__name__}: {e}"

    # Snapshot key data files
    for rel in ("data/rating_timeline.json",
                "data/maps/2026_stage1.csv",
                "data/match_results.csv",
                "data/upcoming_matches.json"):
        full = os.path.join(ROOT, rel)
        try:
            if os.path.exists(full):
                st = os.stat(full)
                out["data_files"][rel] = {"size": st.st_size, "mtime": st.st_mtime}
            else:
                out["data_files"][rel] = "MISSING"
        except Exception as e:
            out["data_files"][rel] = f"err: {e}"

    # Run the scraper in-process and capture stdout
    buf = _io.StringIO()
    old_stdout = _sys.stdout
    try:
        _sys.stdout = buf
        from scrapers import RefreshLiveData as _rld
        # Reset its module-level log so we get a clean slate
        _rld._log_entries = []
        _rld._error_entries = []
        _rld._strategy_log["attempts"] = []
        _rld._strategy_log["first_success"] = None
        try:
            _rld.main()
        except SystemExit:
            pass
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        out["traceback"] = _tb.format_exc()
    finally:
        _sys.stdout = old_stdout
    out["scrape_log"] = buf.getvalue()[-8000:]

    # Snapshot progress file after
    try:
        if os.path.exists(_MHUB_PROGRESS_FILE):
            with open(_MHUB_PROGRESS_FILE) as f:
                out["after_progress"] = json.load(f)
    except Exception as e:
        out["after_progress"] = f"read err: {e}"

    # Re-snapshot data files to see what changed
    out["data_files_after"] = {}
    for rel in ("data/rating_timeline.json",
                "data/maps/2026_stage1.csv",
                "data/match_results.csv",
                "data/upcoming_matches.json"):
        full = os.path.join(ROOT, rel)
        try:
            if os.path.exists(full):
                st = os.stat(full)
                out["data_files_after"][rel] = {"size": st.st_size, "mtime": st.st_mtime}
            else:
                out["data_files_after"][rel] = "MISSING"
        except Exception as e:
            out["data_files_after"][rel] = f"err: {e}"

    out["elapsed"] = _time.time() - out["started"]
    return Response(json.dumps(out, indent=2, default=str), mimetype='application/json')


@mapelo_bp.route('/modern/debug-fetch')
def mapelo_modern_debug_fetch():
    """
    Synchronously fetch one VLR URL and return everything we know about the
    response — status, length, Cloudflare detection, parsed match-item count,
    and which bypass strategies are available + which succeeded.  Lets us
    diagnose Cloudflare/Render issues without waiting for a full scrape.

    Usage:  /mapelo/modern/debug-fetch
           ?url=https://www.vlr.gg/event/matches/2863/vct-2026-emea-stage-1/
    """
    from flask import request as _req
    url = _req.args.get('url',
                        'https://www.vlr.gg/event/matches/2863/vct-2026-emea-stage-1/')
    try:
        # Import lazily so this endpoint stays usable even if scrapers fail
        import sys as _sys
        if ROOT not in _sys.path:
            _sys.path.insert(0, ROOT)
        from scrapers.RefreshLiveData import (
            _CFFI_AVAILABLE, _CFFI_VERSION, _curl_cffi_err,
            _CS_AVAILABLE, _cloudscraper_err,
            _try_strategy, _looks_like_cloudflare,
        )
    except Exception as e:
        return Response(json.dumps({
            "error": f"import failed: {type(e).__name__}: {e}"
        }), mimetype='application/json')

    out = {
        "url": url,
        "available": {
            "curl_cffi": _CFFI_AVAILABLE,
            "curl_cffi_version": _CFFI_VERSION,
            "curl_cffi_err": _curl_cffi_err,
            "cloudscraper": _CS_AVAILABLE,
            "cloudscraper_err": _cloudscraper_err,
        },
        "strategies": [],
    }
    strats = []
    if _CFFI_AVAILABLE:
        strats += ["curl_cffi:chrome131", "curl_cffi:chrome120", "curl_cffi:chrome"]
    if _CS_AVAILABLE:
        strats.append("cloudscraper")
    strats.append("requests")

    for s in strats:
        status, text, err = _try_strategy(s, url, 15)
        entry = {
            "strategy": s,
            "status":   status,
            "len":      len(text) if text else 0,
            "cloudflare_detected": _looks_like_cloudflare(text),
            "match_items": text.count("wf-module-item match-item") if text else 0,
            "err":      err,
            "head":     (text or "")[:400],
        }
        out["strategies"].append(entry)
        # If we got real HTML, stop — we have what we need.
        if status and 200 <= status < 300 and not entry["cloudflare_detected"]:
            break
    return Response(json.dumps(out, indent=2), mimetype='application/json')


@mapelo_bp.route('/map-matches/<org>/<map_name>')
def mapelo_map_matches(org, map_name):
    from flask import request as _req
    year = _req.args.get('year', '2025')
    snap = _req.args.get('snap', 'after_champions')
    data = _get_map_matches(org, map_name, year, snap)
    return Response(json.dumps(data), mimetype='application/json')
