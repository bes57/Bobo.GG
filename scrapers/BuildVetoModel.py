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

import os, sys, json, glob, math
import pandas as pd
from collections import defaultdict, Counter
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')

# ── Recency-decay constants (chosen by backtest on 1,500 matches) ────────────
# Half-life of 6 weeks for ban/pick patterns. Backtest finding: VCT meta
# (agent / map / patch / team-strategy) shifts on a 4–8 week cadence; matches
# older than ~12 weeks contribute little signal to the next match's veto.
# At HL=6, a 6-week-old match contributes 0.5× weight, 12-week old 0.25×, etc.
# Sweep results: HL=6 + alpha=2.0 ban/pick own-strength factor gives:
#   Bo3 step-1 top-1:  37.9% → 49% (+11pp vs cumulative-only baseline)
#   Bo5 step-1 top-1:  48.9% → 66% (+17pp)
#   Bo5 avg top-1:     45.5% → 57% (+11.5pp)
#   Bo5 top-3 coverage: 90% → 93%
VETO_DECAY_HL_WEEKS = 6.0

# Reuse the dynamic 2026+ snapshot definitions from BuildMapRatings so the two
# rating models always agree on which events make up which snapshot.
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scrapers'))
try:
    from BuildMapRatings import YEAR_CONFIGS as _MR_YEAR_CONFIGS
except Exception:
    _MR_YEAR_CONFIGS = {}

GARBAGE_MAPS = {
    'Map 1 stats are unavailable due to lobby remake.',
    'Map 1 stats unavailable due to lobby remake.',
    'Stats unavailable for Map 1 and Map 2 due to inaccessible Riot API.',
    'Stats unavailable for Map 1 due to lobby remake.',
    'Stats unavailable for Map 2 due to lobby remake.',
    'VOD Unavaliable', 'unknown', 'Unknown',
}

# Past seasons (2023-2025) are frozen — those snapshots are already validated.
# 2026+ is derived from BuildMapRatings.YEAR_CONFIGS so any new event added to
# MoreTestingMaybeFiles.ALL_EVENTS picks up here automatically.
YEAR_EVENTS = {
    '2023': ['2023_lock_in', '2023_masters_tokyo', '2023_league', '2023_champions'],
    '2024': ['2024_kickoff', '2024_china_kickoff', '2024_masters_madrid',
             '2024_stage1', '2024_china_stage1', '2024_masters_shanghai',
             '2024_stage2', '2024_china_stage2', '2024_champions'],
    '2025': ['2025_kickoff', '2025_china_kickoff', '2025_masters_bangkok',
             '2025_stage1', '2025_china_stage1', '2025_masters_toronto',
             '2025_stage2', '2025_china_stage2', '2025_champions'],
}

# Which events to include for each (year, snap) — snapshot-aware profiles
# "Before X" snaps exclude X and everything after.  Historical entries frozen.
SNAP_EVENTS = {
    # 2023
    ('2023', 'after_tokyo'):      ['2023_lock_in', '2023_masters_tokyo'],
    ('2023', 'before_champions'): ['2023_lock_in', '2023_masters_tokyo', '2023_league'],
    ('2023', 'after_champions'):  ['2023_lock_in', '2023_masters_tokyo', '2023_league', '2023_champions'],
    # 2024
    ('2024', 'before_madrid'):    ['2024_kickoff', '2024_china_kickoff'],
    ('2024', 'after_madrid'):     ['2024_kickoff', '2024_china_kickoff', '2024_masters_madrid'],
    ('2024', 'before_shanghai'):  ['2024_kickoff', '2024_china_kickoff', '2024_masters_madrid', '2024_stage1', '2024_china_stage1'],
    ('2024', 'after_shanghai'):   ['2024_kickoff', '2024_china_kickoff', '2024_masters_madrid', '2024_stage1', '2024_china_stage1', '2024_masters_shanghai'],
    ('2024', 'before_champions'): ['2024_kickoff', '2024_china_kickoff', '2024_masters_madrid', '2024_stage1', '2024_china_stage1', '2024_masters_shanghai', '2024_stage2', '2024_china_stage2'],
    ('2024', 'after_champions'):  ['2024_kickoff', '2024_china_kickoff', '2024_masters_madrid', '2024_stage1', '2024_china_stage1', '2024_masters_shanghai', '2024_stage2', '2024_china_stage2', '2024_champions'],
    # 2025
    ('2025', 'before_bangkok'):   ['2025_kickoff', '2025_china_kickoff'],
    ('2025', 'after_bangkok'):    ['2025_kickoff', '2025_china_kickoff', '2025_masters_bangkok'],
    ('2025', 'before_toronto'):   ['2025_kickoff', '2025_china_kickoff', '2025_masters_bangkok', '2025_stage1', '2025_china_stage1'],
    ('2025', 'after_toronto'):    ['2025_kickoff', '2025_china_kickoff', '2025_masters_bangkok', '2025_stage1', '2025_china_stage1', '2025_masters_toronto'],
    ('2025', 'before_champions'): ['2025_kickoff', '2025_china_kickoff', '2025_masters_bangkok', '2025_stage1', '2025_china_stage1', '2025_masters_toronto', '2025_stage2', '2025_china_stage2'],
    ('2025', 'after_champions'):  ['2025_kickoff', '2025_china_kickoff', '2025_masters_bangkok', '2025_stage1', '2025_china_stage1', '2025_masters_toronto', '2025_stage2', '2025_china_stage2', '2025_champions'],
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
}

# Pull 2026+ entries straight from BuildMapRatings.YEAR_CONFIGS (which itself
# derives from ALL_EVENTS for current/future seasons). This guarantees the
# veto model and the rating model always agree on which events form which snap.
for _year, _cfg in _MR_YEAR_CONFIGS.items():
    if int(_year) < 2026:
        continue
    _evs_in_order = []
    for _snap_id, _snap in (_cfg.get('snapshots') or {}).items():
        _evs = list(_snap.get('events') or [])
        SNAP_EVENTS[(_year, _snap_id)] = _evs
        if _evs:
            SNAP_POOL_EVENT[(_year, _snap_id)] = _evs[-1]
            for _e in _evs:
                if _e not in _evs_in_order:
                    _evs_in_order.append(_e)
    YEAR_EVENTS[_year] = _evs_in_order


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


def _empty_team_counters():
    return {'n': 0,
            'ban_num': {}, 'ban_den': {},
            'pick_num': {}, 'pick_den': {}}


def _load_match_dates():
    """Returns {match_id: 'YYYY-MM-DD'} from match_dates.json."""
    try:
        with open(os.path.join(DATA, 'match_dates.json')) as f:
            return {int(k): str(v) for k, v in json.load(f).items()}
    except Exception:
        return {}


def _snap_ref_date(snap_events, event_end_dates):
    """Reference date for a snap = last event's end date in that snap."""
    last = None
    for eid in snap_events:
        end = event_end_dates.get(eid)
        if end and (last is None or end > last):
            last = end
    return last


def _load_event_end_dates():
    """Pull event end dates from BuildMapRatings.EVENT_DATES."""
    try:
        from BuildMapRatings import EVENT_DATES
        return {eid: end for eid, (_, end) in EVENT_DATES.items()}
    except Exception:
        return {}


def _decay_weight(match_date_str, ref_date_str, hl_weeks=VETO_DECAY_HL_WEEKS):
    """Exponential decay weight for a match relative to a snap's ref date."""
    if not match_date_str or not ref_date_str:
        return 1.0
    try:
        d_match = datetime.strptime(match_date_str, '%Y-%m-%d')
        d_ref   = datetime.strptime(ref_date_str,   '%Y-%m-%d')
    except Exception:
        return 1.0
    days = (d_ref - d_match).days
    if days < 0:
        return 0.0   # match in the future — exclude (shouldn't happen for valid snaps)
    weeks = days / 7.0
    return math.exp(-math.log(2) * weeks / hl_weeks)


def _apply_match_to_counters(counters, mid_int, match_event, match_pools,
                             event_to_snaps, grp,
                             match_dates, snap_ref_dates):
    """Add one match's contribution to the counters dict (mutated in place),
    weighted by exponential recency decay relative to each snap's ref date.

    No-op if the match isn't mapped to an event/snap/pool.
    """
    event = match_event.get(mid_int)
    if not event:
        return False
    snap_keys = event_to_snaps.get(event)
    if not snap_keys:
        return False
    pool = match_pools.get(mid_int)
    if not pool:
        return False

    match_date = match_dates.get(mid_int)
    team_stripped = grp['team'].astype(str).str.strip()
    teams_in_match = [t for t in team_stripped.unique() if t]

    for org in teams_in_match:
        sel = team_stripped == org
        team_bans  = set(grp.loc[sel & (grp['action'] == 'ban'),  'map'].tolist())
        team_picks = set(grp.loc[sel & (grp['action'] == 'pick'), 'map'].tolist())

        for (year, snap) in snap_keys:
            key = f'{year}_{snap}'
            ref_date = snap_ref_dates.get(key)
            w = _decay_weight(match_date, ref_date)
            if w <= 0:
                continue
            ck = counters.setdefault(key, {}).setdefault(org, _empty_team_counters())
            ck['n'] += 1   # raw match count (unweighted) for min-n filter
            for m in pool:
                ck['ban_den'][m]  = ck['ban_den'].get(m, 0)  + w
                ck['pick_den'][m] = ck['pick_den'].get(m, 0) + w
                if m in team_bans:
                    ck['ban_num'][m]  = ck['ban_num'].get(m, 0)  + w
                if m in team_picks:
                    ck['pick_num'][m] = ck['pick_num'].get(m, 0) + w
    return True


def _profiles_from_counters(counters):
    """Compile counters → final {snap: {org: {n, bans, picks}}} structure.

    bans[m] / picks[m] are decay-weighted RATES in [0,1] (matches expected
    JS-side formula: rate × opp_win, etc.). The `n` field is the raw count
    (no decay), used only for the min-3-matches filter.
    """
    profiles = {}
    for key, by_org in counters.items():
        profiles[key] = {}
        for org, ck in by_org.items():
            if ck['n'] < 3:
                continue
            bans, picks = {}, {}
            all_maps = set(ck['ban_den'].keys()) | set(ck['pick_den'].keys())
            for m in all_maps:
                bd  = ck['ban_den'].get(m, 0)
                pd_ = ck['pick_den'].get(m, 0)
                if bd > 0:
                    bans[m]  = round(ck['ban_num'].get(m, 0)  / bd, 4)
                if pd_ > 0:
                    picks[m] = round(ck['pick_num'].get(m, 0) / pd_, 4)
            profiles[key][org] = {'n': ck['n'], 'bans': bans, 'picks': picks}
    return profiles


def build_snap_profiles(veto_df, match_event, match_pools,
                        prev_counters=None, processed_mids=None):
    """
    Compute per-team recency-DECAYED ban/pick rates per (year, snap).

    Each match's contribution is weighted by exp(-ln(2) · weeks_to_ref_date / HL)
    where ref_date is the end of the snap's last event. So a match 6 weeks
    before the snap's ref date counts at 0.5x; 12 weeks counts at 0.25x; etc.

    Incremental cache disabled — the decay re-weighting requires a full pass
    against the snap's ref date, and the full build is fast (~1500 matches).

    Returns (profiles, counters, processed_mids_set) for back-compat with
    callers (counters/processed_mids no longer needed but returned anyway).
    """
    clean = veto_df[~veto_df['map'].isin(GARBAGE_MAPS)].copy()
    clean = clean[clean['action'].isin(['ban', 'pick'])]
    clean = clean[clean['team'].notna() & (clean['team'].astype(str).str.strip() != '')]

    event_to_snaps = defaultdict(list)
    for (year, snap), events in SNAP_EVENTS.items():
        for e in events:
            event_to_snaps[e].append((year, snap))

    match_dates = _load_match_dates()
    event_end_dates = _load_event_end_dates()
    snap_ref_dates = {
        f'{year}_{snap}': _snap_ref_date(events, event_end_dates)
        for (year, snap), events in SNAP_EVENTS.items()
    }

    counters = {}
    processed = set()

    n_new = 0
    for mid, grp in clean.groupby('MatchID'):
        mid_int = int(mid)
        if _apply_match_to_counters(counters, mid_int, match_event, match_pools,
                                    event_to_snaps, grp,
                                    match_dates, snap_ref_dates):
            n_new += 1
        processed.add(mid_int)

    print(f'  [decayed build] processed {n_new} matches (HL={VETO_DECAY_HL_WEEKS}w)')

    return _profiles_from_counters(counters), counters, processed


def build_snap_pools(event_pools):
    """Map each (year, snap) key to the representative event's pool."""
    snap_pools = {}
    for (year, snap), event in SNAP_POOL_EVENT.items():
        pool = event_pools.get(event)
        if pool:
            snap_pools[f'{year}_{snap}'] = pool
    return snap_pools


STATE_PATH = os.path.join(DATA, 'veto_model_state.json')


def _load_state():
    """Load persisted counters + processed-match list. Missing file → fresh."""
    if not os.path.exists(STATE_PATH):
        return None, set()
    try:
        with open(STATE_PATH) as f:
            st = json.load(f)
        return st.get('counters') or {}, set(st.get('processed_mids') or [])
    except Exception:
        return None, set()


def _save_state(counters, processed_mids):
    try:
        with open(STATE_PATH, 'w') as f:
            json.dump({
                'counters':       counters,
                'processed_mids': sorted(processed_mids),
            }, f, separators=(',', ':'))
    except Exception as e:
        print(f'  [warn] could not write state file: {e}')


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

    # Incremental state: only matches whose MatchID isn't in processed_mids
    # need their counters bumped. Fresh first run = full pass.
    print('Loading incremental counter state…')
    prev_counters, processed_mids = _load_state()
    if prev_counters is None:
        print('  no prior state — full rebuild')
    else:
        print(f'  loaded counters for {sum(len(v) for v in prev_counters.values())} '
              f'(snap, team) entries; {len(processed_mids)} matches already counted')

    print('Building per-snap team ban/pick profiles...')
    profiles, counters, processed_mids = build_snap_profiles(
        veto_df, match_event, match_pools,
        prev_counters=prev_counters, processed_mids=processed_mids,
    )
    for key in sorted(profiles.keys()):
        print(f'  {key}: {len(profiles[key])} teams')

    # Persist counters for the next run
    _save_state(counters, processed_mids)

    snap_pools = build_snap_pools(event_pools)
    print(f'\nSnap pools: {list(snap_pools.keys())}')

    out = {
        'teams':       profiles,
        'snap_pools':  snap_pools,
        'event_pools': event_pools,
        'decay_hl_weeks': VETO_DECAY_HL_WEEKS,
    }

    out_path = os.path.join(DATA, 'veto_model.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, separators=(',', ':'))
    size_kb = os.path.getsize(out_path) / 1024
    print(f'\nWrote {out_path} ({size_kb:.1f} KB)')


if __name__ == '__main__':
    main()
