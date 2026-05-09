"""
BuildVetoModel.py
Builds empirical map ban/pick preference profiles for each team × (year, snap)
from actual VCT veto sequences in map_vetos.csv.

Core idea:
  - Primary signal: team's historical ban/pick rates per map (dominant)
  - Secondary signal: opponent strength adjustment at ban time
  - Year boundary is a hard reset (different rosters/meta)
  - Rates are CONDITIONAL: given map was in the pool, how often did team ban/pick it?
  - Profiles are PER-SNAP: "Before Champions" only uses data from pre-Champions events
  - Per-event map pools derived from data (pool changes each event/stage)

Output: data/veto_model.json

Usage: python scrapers/BuildVetoModel.py
"""

import os, json, glob
import pandas as pd
from collections import defaultdict, Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')

GARBAGE_MAPS = {
    'Map 1 stats are unavailable due to lobby remake.',
    'Map 1 stats unavailable due to lobby remake.',
    'Stats unavailable for Map 1 and Map 2 due to inaccessible Riot API.',
    'Stats unavailable for Map 1 due to lobby remake.',
    'Stats unavailable for Map 2 due to lobby remake.',
    'VOD Unavaliable', 'unknown', 'Unknown',
}

# Events in chronological order per year (hard year boundaries)
YEAR_EVENTS = {
    '2023': ['2023_lock_in', '2023_masters_tokyo', '2023_league', '2023_champions'],
    '2024': ['2024_kickoff', '2024_masters_madrid', '2024_stage1', '2024_masters_shanghai', '2024_stage2', '2024_champions'],
    '2025': ['2025_kickoff', '2025_masters_bangkok', '2025_stage1', '2025_masters_toronto',
             '2025_stage2', '2025_champions'],
    '2026': ['2026_kickoff', '2026_masters_santiago'],
}

# Which events to include for each (year, snap) — snapshot-aware profiles
# "Before X" snaps exclude X and everything after
SNAP_EVENTS = {
    # 2023
    ('2023', 'after_tokyo'):      ['2023_lock_in', '2023_masters_tokyo'],
    ('2023', 'before_champions'): ['2023_lock_in', '2023_masters_tokyo', '2023_league'],
    ('2023', 'after_champions'):  ['2023_lock_in', '2023_masters_tokyo', '2023_league', '2023_champions'],
    # 2024
    ('2024', 'before_madrid'):    ['2024_kickoff'],
    ('2024', 'after_madrid'):     ['2024_kickoff', '2024_masters_madrid'],
    ('2024', 'before_shanghai'):  ['2024_kickoff', '2024_masters_madrid', '2024_stage1'],
    ('2024', 'after_shanghai'):   ['2024_kickoff', '2024_masters_madrid', '2024_stage1', '2024_masters_shanghai'],
    ('2024', 'before_champions'): ['2024_kickoff', '2024_masters_madrid', '2024_stage1', '2024_masters_shanghai', '2024_stage2'],
    ('2024', 'after_champions'):  ['2024_kickoff', '2024_masters_madrid', '2024_stage1', '2024_masters_shanghai', '2024_stage2', '2024_champions'],
    # 2025
    ('2025', 'before_bangkok'):   ['2025_kickoff'],
    ('2025', 'after_bangkok'):    ['2025_kickoff', '2025_masters_bangkok'],
    ('2025', 'before_toronto'):   ['2025_kickoff', '2025_masters_bangkok', '2025_stage1'],
    ('2025', 'after_toronto'):    ['2025_kickoff', '2025_masters_bangkok', '2025_stage1', '2025_masters_toronto'],
    ('2025', 'before_champions'): ['2025_kickoff', '2025_masters_bangkok', '2025_stage1', '2025_masters_toronto', '2025_stage2'],
    ('2025', 'after_champions'):  ['2025_kickoff', '2025_masters_bangkok', '2025_stage1', '2025_masters_toronto', '2025_stage2', '2025_champions'],
    # 2026
    ('2026', 'before_santiago'):  ['2026_kickoff'],
    ('2026', 'after_santiago'):   ['2026_kickoff', '2026_masters_santiago'],
}

# Snap → representative event (last event in that snap's window) for pool lookup
SNAP_POOL_EVENT = {
    ('2023', 'after_tokyo'):      '2023_masters_tokyo',
    ('2023', 'before_champions'): '2023_masters_tokyo',
    ('2023', 'after_champions'):  '2023_champions',
    ('2024', 'before_madrid'):    '2024_kickoff',
    ('2024', 'after_madrid'):     '2024_masters_madrid',
    ('2024', 'before_shanghai'):  '2024_stage1',
    ('2024', 'after_shanghai'):   '2024_masters_shanghai',
    ('2024', 'before_champions'): '2024_stage2',
    ('2024', 'after_champions'):  '2024_champions',
    ('2025', 'before_bangkok'):   '2025_kickoff',
    ('2025', 'after_bangkok'):    '2025_masters_bangkok',
    ('2025', 'before_toronto'):   '2025_stage1',
    ('2025', 'after_toronto'):    '2025_masters_toronto',
    ('2025', 'before_champions'): '2025_stage2',
    ('2025', 'after_champions'):  '2025_champions',
    ('2026', 'before_santiago'):  '2026_kickoff',
    ('2026', 'after_santiago'):   '2026_masters_santiago',
}


def load_match_event_map():
    """Build MatchID -> event_id lookup from per-event maps CSVs."""
    match_event = {}
    maps_dir = os.path.join(DATA, 'maps')
    for fpath in glob.glob(os.path.join(maps_dir, '*.csv')):
        event_id = os.path.basename(fpath).replace('.csv', '')
        try:
            df = pd.read_csv(fpath, usecols=['MatchID'])
            for mid in df['MatchID'].dropna().unique():
                match_event[int(mid)] = event_id
        except Exception:
            pass
    return match_event


def derive_event_pools(veto_df, match_event):
    """
    For each event, find the modal 7-map pool.
    For each match, pool = the set of maps in that match's veto steps.
    Event pool = most common pool across all matches in that event.
    """
    clean = veto_df[~veto_df['map'].isin(GARBAGE_MAPS)].copy()
    match_pools = {}
    for mid, grp in clean.groupby('MatchID'):
        maps_in_match = tuple(sorted(grp['map'].unique()))
        if len(maps_in_match) >= 5:
            match_pools[int(mid)] = maps_in_match

    event_pool_counter = defaultdict(Counter)
    for mid, pool in match_pools.items():
        event = match_event.get(mid)
        if event:
            event_pool_counter[event][pool] += 1

    event_pools = {}
    for event, counter in event_pool_counter.items():
        best_pool = counter.most_common(1)[0][0]
        event_pools[event] = sorted(best_pool)

    return match_pools, event_pools


def build_snap_profiles(veto_df, match_event, match_pools):
    """
    For each (year, snap), compute conditional ban/pick rates per map per team.
    Only matches from events in SNAP_EVENTS[(year, snap)] are included.
    """
    clean = veto_df[~veto_df['map'].isin(GARBAGE_MAPS)].copy()
    clean = clean[clean['action'].isin(['ban', 'pick'])]
    clean = clean[clean['team'].notna() & (clean['team'].astype(str).str.strip() != '')]

    # Pre-build event -> list of (year, snap) keys it contributes to
    event_to_snaps = defaultdict(list)
    for (year, snap), events in SNAP_EVENTS.items():
        for e in events:
            event_to_snaps[e].append((year, snap))

    # Accumulators keyed by (year_snap, org, map)
    ban_num  = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    ban_den  = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    pick_num = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    pick_den = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for mid, grp in clean.groupby('MatchID'):
        mid_int = int(mid)
        event = match_event.get(mid_int)
        if not event:
            continue
        snap_keys = event_to_snaps.get(event)
        if not snap_keys:
            continue
        pool = match_pools.get(mid_int)
        if not pool:
            continue

        teams_in_match = grp['team'].astype(str).str.strip().unique()

        for org in teams_in_match:
            team_actions = grp[grp['team'].astype(str).str.strip() == org]
            team_bans  = set(team_actions[team_actions['action'] == 'ban']['map'].tolist())
            team_picks = set(team_actions[team_actions['action'] == 'pick']['map'].tolist())

            for (year, snap) in snap_keys:
                key = f'{year}_{snap}'
                for m in pool:
                    ban_den[key][org][m]  += 1
                    pick_den[key][org][m] += 1
                    if m in team_bans:
                        ban_num[key][org][m] += 1
                    if m in team_picks:
                        pick_num[key][org][m] += 1

    profiles = {}
    for key in ban_den:
        profiles[key] = {}
        for org in ban_den[key]:
            n_matches = max(ban_den[key][org].values()) if ban_den[key][org] else 0
            if n_matches < 3:
                continue
            bans, picks = {}, {}
            for m in set(list(ban_den[key][org].keys()) + list(pick_den[key][org].keys())):
                bd  = ban_den[key][org].get(m, 0)
                pd_ = pick_den[key][org].get(m, 0)
                if bd > 0:
                    bans[m]  = round(ban_num[key][org].get(m, 0)  / bd, 4)
                if pd_ > 0:
                    picks[m] = round(pick_num[key][org].get(m, 0) / pd_, 4)
            profiles[key][org] = {'n': n_matches, 'bans': bans, 'picks': picks}

    return profiles


def build_snap_pools(event_pools):
    """Map each (year, snap) key to the representative event's pool."""
    snap_pools = {}
    for (year, snap), event in SNAP_POOL_EVENT.items():
        pool = event_pools.get(event)
        if pool:
            snap_pools[f'{year}_{snap}'] = pool
    return snap_pools


def main():
    print('Loading veto data...')
    veto_df = pd.read_csv(os.path.join(DATA, 'map_vetos.csv'))
    print(f'  {len(veto_df)} veto rows, {veto_df["MatchID"].nunique()} matches')

    print('Building MatchID → event map...')
    match_event = load_match_event_map()
    print(f'  Mapped {len(match_event)} match IDs')

    print('Deriving per-event map pools...')
    match_pools, event_pools = derive_event_pools(veto_df, match_event)
    for event, pool in sorted(event_pools.items()):
        print(f'  {event}: {pool}')

    print('Building per-snap team ban/pick profiles...')
    profiles = build_snap_profiles(veto_df, match_event, match_pools)
    for key in sorted(profiles.keys()):
        print(f'  {key}: {len(profiles[key])} teams')

    # Sample G2 across snaps
    for key in ['2025_early', '2025_after_stage1', '2025_full']:
        g2 = profiles.get(key, {}).get('G2')
        if g2:
            top_bans  = sorted(g2['bans'].items(),  key=lambda x: -x[1])[:4]
            top_picks = sorted(g2['picks'].items(), key=lambda x: -x[1])[:3]
            print(f'  G2 {key} (n={g2["n"]}): bans={top_bans}')
            print(f'           picks={top_picks}')

    snap_pools = build_snap_pools(event_pools)
    print(f'\nSnap pools: {list(snap_pools.keys())}')

    out = {
        'teams':       profiles,
        'snap_pools':  snap_pools,
        'event_pools': event_pools,
    }

    out_path = os.path.join(DATA, 'veto_model.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, separators=(',', ':'))
    size_kb = os.path.getsize(out_path) / 1024
    print(f'\nWrote {out_path} ({size_kb:.1f} KB)')


if __name__ == '__main__':
    main()
