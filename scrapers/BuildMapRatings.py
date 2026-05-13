"""
BuildMapRatings.py
Massey-based map rating system for VCT domestic events (2023-2025).

Model:
  - Opponent-adjusted Massey ratings using round differential as the margin signal
  - Exponential recency decay: weight(game) = exp(-λ * weeks_ago)
    Calendar-date-based: weeks_ago = (ref_date - game_date).days / 7
    Per-map and overall Massey both use date-based decay.
  - λ optimized via temporal cross-validation (rolling-window across events)
  - Per-map ratings with James-Stein shrinkage toward overall rating
  - Pick/ban-adjusted overall rating: Monte Carlo BO3 veto simulation vs.
    league-average opponent. Teams rationally ban worst maps and pick best.
    Overall rating = expected round-diff advantage on the 3 maps that survive.
  - Win probability via logistic: P(A beats B) = sigmoid(β * (r_A - r_B))
  - β calibrated on training data via log-loss minimization

Output: data/map_ratings.json

Usage: python scrapers/BuildMapRatings.py [--reoptimize]
"""

import os, sys, json, math, random
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime, timedelta
from scipy.special import expit
from scipy.optimize import minimize_scalar

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from MoreTestingMaybeFiles import ALL_EVENTS

DATA_DIR = os.path.join(ROOT, 'data')
OUT_PATH = os.path.join(DATA_DIR, 'map_ratings.json')

# International events — cross-regional calibrators; never apply recency decay.
# Derived from ALL_EVENTS: any event whose `regions` dict has an "International"
# key is treated as international (cross-regional). No hand-maintained list.
INTL_EVENTS = {e['id'] for e in ALL_EVENTS
               if 'International' in (e.get('regions') or {})}

# VCT franchised-era regional assignments (org abbreviation → league region)
TEAM_REGIONS = {
    # EMEA
    "TL": "EMEA", "FNC": "EMEA", "NAVI": "EMEA", "VIT": "EMEA",
    "BBL": "EMEA", "GX": "EMEA", "KC": "EMEA", "TH": "EMEA",
    "FUT": "EMEA", "GIA": "EMEA", "MKOI": "EMEA", "WOL": "EMEA",
    "M8": "EMEA", "FPX": "EMEA",
    # Americas
    "SEN": "Americas", "G2": "Americas", "MIBR": "Americas",
    "NRG": "Americas", "100T": "Americas", "C9": "Americas",
    "EG": "Americas", "KRÜ": "Americas", "LEV": "Americas",
    "FUR": "Americas", "LOUD": "Americas",
    # Pacific
    "PRX": "Pacific", "DRX": "Pacific", "T1": "Pacific",
    "TLN": "Pacific", "GEN": "Pacific", "DFM": "Pacific",
    "ZETA": "Pacific", "RRQ": "Pacific", "TS": "Pacific", "GE": "Pacific",
    # CN
    "EDG": "CN", "BLG": "CN", "KRX": "CN", "TE": "CN",
    "DRG": "CN", "ASE": "CN", "NS": "CN", "AG": "CN", "XLG": "CN",
}

# ── Event metadata ─────────────────────────────────────────────────────────────
# Past seasons are frozen below (these dates pre-date the start/end fields on
# ALL_EVENTS). Current and future seasons are auto-loaded from ALL_EVENTS at
# import time — any new event added to MoreTestingMaybeFiles.py with start/end
# dates picks up here automatically, no edit required.

_HISTORICAL_EVENT_DATES = {
    # 2023
    '2023_lock_in':        ('2023-01-10', '2023-02-12'),
    '2023_league':         ('2023-01-23', '2023-10-01'),
    '2023_masters_tokyo':  ('2023-06-11', '2023-06-25'),
    '2023_champions':      ('2023-08-06', '2023-08-27'),
    # 2024
    '2024_kickoff':           ('2024-01-08', '2024-02-11'),
    '2024_masters_madrid':    ('2024-02-14', '2024-03-10'),
    '2024_stage1':            ('2024-03-15', '2024-05-19'),
    '2024_masters_shanghai':  ('2024-06-02', '2024-06-16'),
    '2024_stage2':            ('2024-06-20', '2024-08-25'),
    '2024_champions':         ('2024-08-01', '2024-09-22'),
    # 2025
    '2025_kickoff':         ('2025-01-13', '2025-02-09'),
    '2025_masters_bangkok': ('2025-02-12', '2025-03-09'),
    '2025_stage1':          ('2025-03-14', '2025-05-18'),
    '2025_masters_toronto': ('2025-06-07', '2025-06-29'),
    '2025_stage2':          ('2025-07-14', '2025-08-24'),
    '2025_champions':       ('2025-08-28', '2025-09-21'),
}

EVENT_DATES = dict(_HISTORICAL_EVENT_DATES)
for _e in ALL_EVENTS:
    if _e.get('start') and _e.get('end'):
        EVENT_DATES[_e['id']] = (_e['start'], _e['end'])

# Chronological train/test split for β calibration. Hold out the most recent
# ~16 months so β re-fits as the season progresses without manual upkeep.
_today_str = datetime.now().strftime('%Y-%m-%d')
_holdout_start = (datetime.now() - timedelta(days=480)).strftime('%Y-%m-%d')
TRAIN_EVENTS = [eid for eid, (_, end) in EVENT_DATES.items() if end < _holdout_start]
TEST_EVENTS  = [eid for eid, (_, end) in EVENT_DATES.items() if end >= _holdout_start]

# Fixed hyper-parameters (CV-validated; same Brier score as optimised values)
HALF_LIFE_WEEKS   = 8      # calendar-weeks half-life — shorter to punish stale domestic records
INTL_WIN_MULT     = 4.0    # intl games: both sides get 4x — symmetric, full-weight international games
INTL_LOSS_MULT    = 4.0    # symmetric with win mult — losing at intl hurts as much as winning helps
INTL_MULTIPLIER   = INTL_WIN_MULT  # legacy alias used in shrinkage weight counts
SHRINK_K          = 12     # James-Stein shrinkage strength for per-map ratings
MC_N_SIMS         = 10000 # Monte Carlo veto simulations per team per snapshot
VETO_NOISE_STD    = 2.0  # Gaussian noise on map advantages during veto (round diff units)

# Per-year snapshot configuration.
# Snapshots are named for the last international included:
#   before_1st  = data up to (not including) the 1st international
#   after_1st   = through 1st international
#   before_2nd  = through domestic events between 1st and 2nd internationals
#   after_2nd   = through 2nd international
#   before_champs = everything before Champions
#   after_champs  = full year including Champions
def _short_id(event_id):
    """Snapshot suffix: strip the year prefix and the 'masters_' marker so the
    snapshot keys read naturally ('after_santiago' not 'after_2026_masters_santiago')."""
    s = event_id.split('_', 1)[1] if '_' in event_id else event_id
    if s.startswith('masters_'):
        s = s[len('masters_'):]
    return s


def _short_label(label, year):
    """Drop the leading year from an event label: '2026 Masters Santiago' -> 'Masters Santiago'."""
    return label.replace(f'{year} ', '', 1)


def _build_year_configs():
    """Dynamically generate snapshot configs from ALL_EVENTS.

    For each year, snapshots are emitted in chronological order:
      - International events get both ``before_<short>`` (cumulative through
        the previous event) and ``after_<short>`` (cumulative through this
        international).
      - Domestic events accumulate silently UNLESS they are the year's most
        recent event with data — in that case we emit ``after_<short>`` so
        the current stage has a snapshot. This is what makes "Live" appear
        mid-stage without any hardcoded list.

    Events with no corresponding ``data/maps/<id>.csv`` (future events whose
    data hasn't been scraped yet) are skipped.
    """
    by_year = {}
    for e in ALL_EVENTS:
        if not e.get('start') or not e.get('end'):
            continue
        if not os.path.exists(os.path.join(DATA_DIR, 'maps', f"{e['id']}.csv")):
            continue
        by_year.setdefault(str(e['year']), []).append(e)

    configs = {}
    for year, events in by_year.items():
        events = sorted(events, key=lambda e: e['start'])
        if not events:
            continue
        snapshots = {}
        cumulative = []
        last_event = events[-1]
        last_is_intl = 'International' in (last_event.get('regions') or {})
        for e in events:
            is_intl = 'International' in (e.get('regions') or {})
            short = _short_id(e['id'])
            short_label = _short_label(e['label'], e['year'])
            if is_intl:
                if cumulative:
                    snapshots[f'before_{short}'] = {
                        'events': list(cumulative),
                        'label':  f'Before {short_label}',
                    }
                cumulative.append(e['id'])
                snapshots[f'after_{short}'] = {
                    'events': list(cumulative),
                    'label':  f'After {short_label}',
                }
            else:
                cumulative.append(e['id'])
        if not last_is_intl:
            short = _short_id(last_event['id'])
            short_label = _short_label(last_event['label'], last_event['year'])
            snapshots[f'after_{short}'] = {
                'events': list(cumulative),
                'label':  short_label,
            }
        # If the year's latest event is still ongoing (end date hasn't passed),
        # mark the latest snapshot as "Live". Past, completed years keep their
        # event-name labels (e.g. "After Champions").
        if snapshots and last_event['end'] >= _today_str:
            latest_key = list(snapshots.keys())[-1]
            snapshots[latest_key] = dict(snapshots[latest_key], label='Live')
        configs[year] = {'snapshots': snapshots, 'min_games': 5}
    return configs


# Frozen snapshot definitions for past seasons (these pre-date the dynamic
# generator; preserving them keeps historical kenpom ratings identical to
# what's already cached and validated).
_HISTORICAL_YEAR_CONFIGS = {
    '2023': {
        'snapshots': {
            'after_tokyo':      {'events': ['2023_lock_in', '2023_masters_tokyo'],
                                 'label': 'After Masters Tokyo'},
            'before_champions': {'events': ['2023_lock_in', '2023_masters_tokyo', '2023_league'],
                                 'label': 'Before Champions'},
            'after_champions':  {'events': ['2023_lock_in', '2023_masters_tokyo', '2023_league', '2023_champions'],
                                 'label': 'After Champions'},
        },
        'min_games': 5,
    },
    '2024': {
        'snapshots': {
            'before_madrid':    {'events': ['2024_kickoff'],
                                 'label': 'Before Masters Madrid'},
            'after_madrid':     {'events': ['2024_kickoff', '2024_masters_madrid'],
                                 'label': 'After Masters Madrid'},
            'before_shanghai':  {'events': ['2024_kickoff', '2024_masters_madrid', '2024_stage1'],
                                 'label': 'Before Masters Shanghai'},
            'after_shanghai':   {'events': ['2024_kickoff', '2024_masters_madrid', '2024_stage1', '2024_masters_shanghai'],
                                 'label': 'After Masters Shanghai'},
            'before_champions': {'events': ['2024_kickoff', '2024_masters_madrid', '2024_stage1', '2024_masters_shanghai', '2024_stage2'],
                                 'label': 'Before Champions'},
            'after_champions':  {'events': ['2024_kickoff', '2024_masters_madrid', '2024_stage1', '2024_masters_shanghai', '2024_stage2', '2024_champions'],
                                 'label': 'After Champions'},
        },
        'min_games': 5,
    },
    '2025': {
        'snapshots': {
            'before_bangkok':   {'events': ['2025_kickoff'],
                                 'label': 'Before Masters Bangkok'},
            'after_bangkok':    {'events': ['2025_kickoff', '2025_masters_bangkok'],
                                 'label': 'After Masters Bangkok'},
            'before_toronto':   {'events': ['2025_kickoff', '2025_masters_bangkok', '2025_stage1'],
                                 'label': 'Before Masters Toronto'},
            'after_toronto':    {'events': ['2025_kickoff', '2025_masters_bangkok', '2025_stage1', '2025_masters_toronto'],
                                 'label': 'After Masters Toronto'},
            'before_champions': {'events': ['2025_kickoff', '2025_masters_bangkok', '2025_stage1', '2025_masters_toronto', '2025_stage2'],
                                 'label': 'Before Champions'},
            'after_champions':  {'events': ['2025_kickoff', '2025_masters_bangkok', '2025_stage1', '2025_masters_toronto', '2025_stage2', '2025_champions'],
                                 'label': 'After Champions'},
        },
        'min_games': 5,
    },
}

YEAR_CONFIGS = dict(_HISTORICAL_YEAR_CONFIGS)
for _year, _cfg in _build_year_configs().items():
    # Don't overwrite frozen historical configs; only add years they don't cover.
    if _year not in YEAR_CONFIGS:
        YEAR_CONFIGS[_year] = _cfg


def _parse_date(s):
    return datetime.strptime(s, '%Y-%m-%d')


# ── Data loading ───────────────────────────────────────────────────────────────

def load_games(only_events=None):
    """
    Load domestic map-level games and return a list of dicts:
      match_id, event_id, map_name, winner, loser, wr, lr, date, match_rank
    Dates are interpolated from match_id rank within each event.
    match_rank is the global ordinal position across all events (1 = oldest).

    ``only_events``: optional iterable of event_ids — when set, skip all
    other event CSVs. Used in --refresh mode to avoid parsing 4 years of
    historical CSVs every page-load refresh.
    """
    only_set = set(only_events) if only_events else None
    mr = pd.read_csv(os.path.join(DATA_DIR, 'match_results.csv'))
    mr = mr[mr['MapNum'] != 'all'].copy()
    mr['MapNum'] = mr['MapNum'].astype(str)
    mr_idx = mr.set_index(['MatchID', 'MapNum'])

    frames = []
    for event in ALL_EVENTS:
        eid = event['id']
        if only_set is not None and eid not in only_set:
            continue
        if eid not in EVENT_DATES:
            continue
        path = os.path.join(DATA_DIR, 'maps', f'{eid}.csv')
        if not os.path.exists(path):
            print(f'  [warn] missing maps/{eid}.csv — skipping')
            continue
        df = pd.read_csv(path)
        df['event_id'] = eid
        df['MapNum']   = df['MapNum'].astype(str)
        df['MapName']  = df['MapName'].str.replace('PICK', '', regex=False).str.strip()
        frames.append(df)

    if not frames:
        raise RuntimeError('No map data found')

    all_maps = pd.concat(frames, ignore_index=True)

    meta = all_maps.groupby(['MatchID', 'MapNum']).agg(
        orgs=('Org',      lambda x: list(x.unique())),
        map_name=('MapName',  'first'),
        event_id=('event_id', 'first'),
    ).reset_index()

    games = []
    for _, row in meta.iterrows():
        key = (int(row['MatchID']), row['MapNum'])
        if key not in mr_idx.index:
            continue
        mr_row = mr_idx.loc[key]
        winner = mr_row['WinnerOrg']
        losers = [o for o in row['orgs'] if o != winner]
        if not losers:
            continue
        try:
            wr, lr = map(int, str(mr_row['Score']).split('-'))
        except Exception:
            continue
        games.append({
            'match_id':  int(row['MatchID']),
            'event_id':  row['event_id'],
            'map_name':  row['map_name'],
            'winner':    winner,
            'loser':     losers[0],
            'wr':        wr,
            'lr':        lr,
            'date':      None,
        })

    # Assign approximate dates: interpolate match_id rank within each event
    gdf = pd.DataFrame(games)
    for eid, (start_str, end_str) in EVENT_DATES.items():
        mask = gdf['event_id'] == eid
        if not mask.any():
            continue
        start = _parse_date(start_str)
        end   = _parse_date(end_str)
        span  = (end - start).days

        mids = gdf.loc[mask, 'match_id'].values
        sorted_unique = sorted(set(mids))
        rank_map = {mid: i for i, mid in enumerate(sorted_unique)}
        max_rank  = max(rank_map.values()) or 1

        indices = gdf.index[mask].tolist()
        for i, mid in zip(indices, mids):
            frac = rank_map[mid] / max_rank
            gdf.at[i, 'date'] = start + timedelta(days=int(frac * span))

    gdf = gdf.dropna(subset=['date'])

    # Assign global match_rank: unique matches ordered by (date, match_id)
    # Rank 1 = oldest, rank N = most recent
    unique_matches = (
        gdf[['match_id', 'date']].drop_duplicates('match_id')
        .sort_values(['date', 'match_id'])
        .reset_index(drop=True)
    )
    unique_matches['match_rank'] = np.arange(1, len(unique_matches) + 1)
    rank_by_mid = dict(zip(unique_matches['match_id'], unique_matches['match_rank']))
    gdf['match_rank'] = gdf['match_id'].map(rank_by_mid)

    print(f'  match_rank range: 1 – {unique_matches["match_rank"].max()} '
          f'({len(unique_matches)} unique matches)')

    # Assign per-map match ranks: ordinal within each map's own history.
    # Used for per-map recency decay so "Ascent games ago" ≠ "all games ago".
    games_list = gdf.to_dict('records')
    by_map = defaultdict(list)
    for g in games_list:
        by_map[g['map_name']].append(g)
    for mn, mg in by_map.items():
        sorted_mg = sorted(mg, key=lambda x: (x['date'], x['match_id']))
        for i, g in enumerate(sorted_mg):
            g['map_match_rank'] = i + 1

    return games_list


# ── Massey solver ──────────────────────────────────────────────────────────────

def massey_ratings(games, lambda_decay, ref_date, min_games=0):
    """
    Solve the Massey system with exponential recency decay.

    weight = exp(-λ * weeks_ago)  where weeks_ago = (ref_date - game_date).days / 7
    λ is in units of 1/week; half-life = ln(2) / λ weeks.

    Returns dict: team -> rating (mean-zero, units = round differential)
    """
    if not games:
        return {}

    teams = sorted({g['winner'] for g in games} | {g['loser'] for g in games})

    if min_games > 0:
        counts = {}
        for g in games:
            counts[g['winner']] = counts.get(g['winner'], 0) + 1
            counts[g['loser']]  = counts.get(g['loser'],  0) + 1
        teams = [t for t in teams if counts.get(t, 0) >= min_games]
        games  = [g for g in games if g['winner'] in teams and g['loser'] in teams]
        if not games:
            return {}

    n   = len(teams)
    idx = {t: i for i, t in enumerate(teams)}
    M   = np.zeros((n, n))
    p   = np.zeros(n)

    for g in games:
        if g['winner'] not in idx or g['loser'] not in idx:
            continue
        weeks_ago = max(0, (ref_date - g['date']).days / 7.0)
        base_w = math.exp(-lambda_decay * weeks_ago)
        is_intl = g.get('event_id') in INTL_EVENTS
        w_win = base_w * (INTL_WIN_MULT  if is_intl else 1.0)
        w_los = base_w * (INTL_LOSS_MULT if is_intl else 1.0)
        w_sym = min(w_win, w_los)  # symmetric part for M matrix (ensures PSD)
        rd = g['wr'] - g['lr']
        i, j = idx[g['winner']], idx[g['loser']]
        M[i, i] += w_sym;  M[j, j] += w_sym
        M[i, j] -= w_sym;  M[j, i] -= w_sym
        p[i] += w_win * rd;  p[j] -= w_los * rd

    # Anchor: mean(r) = 0
    M[-1, :] = 1.0
    p[-1]    = 0.0

    # Ridge: regularize toward zero; 0.5 ≈ one "virtual" draw against a neutral opponent
    ridge = 0.5
    for i in range(n - 1):
        M[i, i] += ridge

    M[-1, :] = 1.0
    p[-1]    = 0.0

    try:
        r = np.linalg.solve(M, p)
    except np.linalg.LinAlgError:
        r, *_ = np.linalg.lstsq(M, p, rcond=None)

    return {t: float(r[idx[t]]) for t in teams}


# ── Per-map ratings with shrinkage ─────────────────────────────────────────────

def compute_per_map_ratings(games, overall_ratings, lambda_decay,
                            shrink_k=5, min_map_games=4):
    """
    Separate Massey solve per map, blended with overall via James-Stein shrinkage:
        rating = (n_eff / (n_eff + k)) * map_rating + (k / (n_eff + k)) * overall_rating

    Decay uses date-based weeks_ago measured from the most recent game on that map.
    """
    all_maps = sorted({g['map_name'] for g in games})
    result = {}

    for map_name in all_maps:
        mg = [g for g in games if g['map_name'] == map_name]
        if len(mg) < min_map_games:
            result[map_name] = {}
            continue

        # Ref date = most recent game on this specific map
        map_ref_date = max(g['date'] for g in mg)
        map_rtgs = massey_ratings(mg, lambda_decay, map_ref_date)

        # Effective weights using date-based decay for shrinkage alpha
        w_counts = {}
        for g in mg:
            weeks_ago = max(0, (map_ref_date - g['date']).days / 7.0)
            w = math.exp(-lambda_decay * weeks_ago)
            if g.get('event_id') in INTL_EVENTS:
                w *= INTL_MULTIPLIER
            w_counts[g['winner']] = w_counts.get(g['winner'], 0.0) + w
            w_counts[g['loser']]  = w_counts.get(g['loser'],  0.0) + w

        blended = {}
        all_teams = set(overall_ratings) | set(map_rtgs)
        for team in all_teams:
            n_eff  = w_counts.get(team, 0.0)
            alpha  = n_eff / (n_eff + shrink_k)
            map_r  = map_rtgs.get(team, 0.0)
            ovr_r  = overall_ratings.get(team, 0.0)
            blended[team] = alpha * map_r + (1 - alpha) * ovr_r

        result[map_name] = blended

    return result


# ── Pick/ban simulation ────────────────────────────────────────────────────────

def simulate_veto(advantages, pool, noise_std=VETO_NOISE_STD):
    """
    Simulate one BO3 veto. Returns [A_pick, B_pick, decider] (3 maps).

    advantages: {map: rating_A - rating_B}  (positive = better for A)
    pool: collection of map names

    BO3 sequence generalised to any pool size N:
      Phase 1: alternate bans (A, B, ...) until 5 maps remain
      Phase 2: A picks best, B picks best for B
      Phase 3: alternate bans until 1 map remains (decider)

    Gaussian noise on advantages models imperfect information and tactical variance.
    """
    pool = list(pool)
    n = len(pool)
    if n < 3:
        return pool

    noisy = {m: advantages.get(m, 0.0) + random.gauss(0, noise_std) for m in pool}
    remaining = list(pool)
    selected = []

    # Phase 1: ban down to 5 maps (standard 7-map veto = 2 bans here)
    phase1_bans = max(0, n - 5)
    turn = 0  # 0=A bans min, 1=B bans max
    for _ in range(phase1_bans):
        m = (min if turn == 0 else max)(remaining, key=lambda x: noisy[x])
        remaining.remove(m)
        turn ^= 1

    # Phase 2: A picks best remaining, B picks best for B (worst for A)
    m = max(remaining, key=lambda x: noisy[x])
    remaining.remove(m); selected.append(m)
    m = min(remaining, key=lambda x: noisy[x])
    remaining.remove(m); selected.append(m)

    # Phase 3: alternate bans until 1 map remains (= decider)
    turn = 0
    while len(remaining) > 1:
        m = (min if turn == 0 else max)(remaining, key=lambda x: noisy[x])
        remaining.remove(m)
        turn ^= 1

    selected.append(remaining[0])
    return selected  # [A_pick, B_pick, decider]


def pb_adjusted_rating(team_map_rtgs, avg_map_rtgs, all_maps,
                       n_sims=None, noise_std=VETO_NOISE_STD):
    """
    Pick/ban-adjusted overall rating.

    Simulates n_sims BO3 vetos vs league-average opponent. Returns expected
    round-differential advantage per map across the 3 maps that survive the veto.

    Same units as Massey ratings (round diff per map game vs average).
    Positive = beats average even after optimal pick/ban by the opponent.

    n_sims defaults to the current MC_N_SIMS module constant — read at call
    time so refresh-mode monkey-patches in main() take effect.
    """
    if n_sims is None:
        n_sims = MC_N_SIMS
    advantages = {m: team_map_rtgs.get(m, 0.0) - avg_map_rtgs.get(m, 0.0)
                  for m in all_maps}
    total = 0.0
    for _ in range(n_sims):
        played = simulate_veto(advantages, all_maps, noise_std=noise_std)
        for m in played:
            total += advantages[m]
    return total / (n_sims * 3)


# ── Win probability calibration ────────────────────────────────────────────────

def fit_beta(games, ratings):
    """Fit logistic β via log-loss: P(winner) = sigmoid(β × Δrating)."""
    diffs, outcomes = [], []
    for g in games:
        rw = ratings.get(g['winner'], 0.0)
        rl = ratings.get(g['loser'],  0.0)
        diffs.append(rw - rl);  outcomes.append(1.0)
        diffs.append(rl - rw);  outcomes.append(0.0)

    diffs    = np.array(diffs)
    outcomes = np.array(outcomes)

    def neg_log_loss(beta):
        probs = np.clip(expit(beta * diffs), 1e-9, 1 - 1e-9)
        return -np.mean(outcomes * np.log(probs) + (1 - outcomes) * np.log(1 - probs))

    res = minimize_scalar(neg_log_loss, bounds=(0.01, 10.0), method='bounded')
    return float(res.x)


def brier_score(games, ratings, beta):
    """Mean Brier score. Lower = better. Baseline (always 50%) = 0.25."""
    if not games:
        return 1.0
    scores = []
    for g in games:
        rw = ratings.get(g['winner'], 0.0)
        rl = ratings.get(g['loser'],  0.0)
        p  = float(expit(beta * (rw - rl)))
        scores.append((p - 1.0) ** 2)
    return float(np.mean(scores))


def log_loss(games, ratings, beta):
    """Log-loss. Baseline = ln(2) ≈ 0.693."""
    if not games:
        return math.log(2)
    ll = 0.0
    for g in games:
        rw = ratings.get(g['winner'], 0.0)
        rl = ratings.get(g['loser'],  0.0)
        p  = float(np.clip(expit(beta * (rw - rl)), 1e-9, 1 - 1e-9))
        ll -= math.log(p)
    return ll / len(games)


# ── λ grid search ──────────────────────────────────────────────────────────────

def optimize_lambda(all_games, n_grid=40):
    """
    Grid search over half-lives 2–52 weeks using rolling-window CV.

    Folds: each domestic event is the test set; all earlier events are train.
    ref_date for each fold = latest game date in training set.
    """
    # Map event_id -> latest date in that event
    event_max_date = {}
    event_min_date = {}
    for g in all_games:
        eid = g['event_id']
        d   = g['date']
        if eid not in event_max_date or d > event_max_date[eid]:
            event_max_date[eid] = d
        if eid not in event_min_date or d < event_min_date[eid]:
            event_min_date[eid] = d

    ordered_events = sorted(event_max_date, key=lambda e: event_max_date[e])

    folds = []
    for eid in ordered_events:
        test_games  = [g for g in all_games if g['event_id'] == eid]
        this_start  = event_min_date[eid]
        train_games = [g for g in all_games if g['date'] < this_start]
        if len(train_games) < 20 or len(test_games) < 10:
            continue
        train_teams = {g['winner'] for g in train_games} | {g['loser'] for g in train_games}
        test_teams  = {g['winner'] for g in test_games}  | {g['loser'] for g in test_games}
        if len(train_teams & test_teams) < 5:
            continue
        ref_date = max(g['date'] for g in train_games)
        folds.append({'eid': eid, 'train': train_games, 'test': test_games, 'ref_date': ref_date})

    print(f'  CV folds: {len(folds)} ({", ".join(f["eid"] for f in folds)})')

    # Half-life range 2–52 weeks
    half_lives = np.logspace(np.log10(2), np.log10(52), n_grid)
    lambdas    = np.log(2) / half_lives

    results = []
    print(f'\n  {"half-life":>12}  {"λ":>9}  {"β_mean":>7}  {"brier_cv":>10}')
    print(f'  {"-"*12}  {"-"*9}  {"-"*7}  {"-"*10}')

    for hl, lam in zip(half_lives, lambdas):
        fold_briers = []
        fold_betas  = []
        for fold in folds:
            rtgs = massey_ratings(fold['train'], lam, fold['ref_date'])
            beta = fit_beta(fold['train'], rtgs)
            b    = brier_score(fold['test'], rtgs, beta)
            fold_briers.append(b)
            fold_betas.append(beta)
        cv_brier  = float(np.mean(fold_briers))
        mean_beta = float(np.mean(fold_betas))
        print(f'  {hl:12.1f}w  {lam:9.6f}  {mean_beta:7.3f}  {cv_brier:10.5f}')
        results.append({
            'half_life_weeks': round(float(hl), 1),
            'lambda':          float(lam),
            'beta_mean':       round(mean_beta, 4),
            'brier_cv':        round(cv_brier,  5),
            'fold_briers':     [round(b, 5) for b in fold_briers],
        })

    best = min(results, key=lambda r: r['brier_cv'])
    return best['lambda'], results


# ── Win/loss record helpers ────────────────────────────────────────────────────

def build_records(games):
    """Return {team: {w, l, maps: {map: {w, l}}}} across all games."""
    rec = {}
    for g in games:
        for team, is_win in [(g['winner'], True), (g['loser'], False)]:
            if team not in rec:
                rec[team] = {'w': 0, 'l': 0, 'maps': {}}
            rec[team]['w' if is_win else 'l'] += 1
            mn = g['map_name']
            if mn not in rec[team]['maps']:
                rec[team]['maps'][mn] = {'w': 0, 'l': 0}
            rec[team]['maps'][mn]['w' if is_win else 'l'] += 1
    return rec


# ── Effective game count ────────────────────────────────────────────────────────

def effective_counts(games, lambda_decay, ref_date):
    """Sum of decay weights per team — proxy for how much recent data is active."""
    counts = {}
    for g in games:
        weeks_ago = max(0, (ref_date - g['date']).days / 7.0)
        w = math.exp(-lambda_decay * weeks_ago)
        counts[g['winner']] = counts.get(g['winner'], 0.0) + w
        counts[g['loser']]  = counts.get(g['loser'],  0.0) + w
    return {t: round(v, 2) for t, v in counts.items()}


# ── Within-region qualification cap ───────────────────────────────────────────

def apply_qualification_cap(teams_out, intl_event_id, all_games, epsilon=0.001):
    """
    Within-region qualification adjustment.

    Non-qualifying teams that sit above the lowest qualifier from their region
    are compressed into the gap between that qualifier and the highest
    non-qualifier already below it.  Teams already below the threshold are
    unchanged.  Cross-regional ordering is never touched.

    When some non-qualifiers are already below cap:
        floor = below_cap_max + ε
        new_r = floor + (r - floor) / (max_above - floor) * (cap - ε - floor)

    When all non-qualifiers are above cap: uniform shift so max → cap - ε.
    """
    qualifiers = set()
    for g in all_games:
        if g['event_id'] == intl_event_id:
            qualifiers.add(g['winner'])
            qualifiers.add(g['loser'])

    if not qualifiers:
        return teams_out, 0

    # Minimum overall_rating per region among qualifying teams
    region_min = {}
    for team, data in teams_out.items():
        if team not in qualifiers:
            continue
        region = TEAM_REGIONS.get(team)
        if not region:
            continue
        r = data['overall_rating']
        if region not in region_min or r < region_min[region]:
            region_min[region] = r

    if not region_min:
        return teams_out, 0

    n_adjusted = 0
    for region, min_qual in region_min.items():
        cap = min_qual - epsilon

        nonquals = [(t, teams_out[t]['overall_rating'])
                    for t in teams_out
                    if t not in qualifiers and TEAM_REGIONS.get(t) == region]
        if not nonquals:
            continue

        above = [(t, r) for t, r in nonquals if r > cap]
        if not above:
            continue

        below = [(t, r) for t, r in nonquals if r <= cap]
        max_above = max(r for _, r in above)

        if below:
            below_max = max(r for _, r in below)
            floor     = below_max + epsilon
            cap_adj   = cap - epsilon
            denom     = max_above - floor
            if denom <= 0 or cap_adj <= floor:
                for t, _ in above:
                    teams_out[t]['overall_rating'] = round(cap - epsilon, 3)
                    n_adjusted += 1
            else:
                scale = (cap_adj - floor) / denom
                for t, r in above:
                    new_r = floor + (r - floor) * scale
                    teams_out[t]['overall_rating'] = round(new_r, 3)
                    n_adjusted += 1
        else:
            # All non-qualifiers above cap: uniform shift preserves spacing
            shift = max_above - (cap - epsilon)
            for t, r in above:
                teams_out[t]['overall_rating'] = round(r - shift, 3)
                n_adjusted += 1

    if n_adjusted:
        teams_out = dict(sorted(teams_out.items(), key=lambda x: -x[1]['overall_rating']))

    return teams_out, n_adjusted


# ── Per-year ratings builder ───────────────────────────────────────────────────

CN_TEAMS = {
    "EDG", "BLG", "TE", "DRG", "ASE", "AG", "XLG",
    "WOL",  # Wolves Esports — VCT CN 2025
}

def build_year_ratings(games, lam, ref_date, shrink_k, min_games, filter_teams=None):
    """
    Compute pick/ban-adjusted ratings for one snapshot's games.

    overall_rating = Monte Carlo pick/ban-adjusted rating vs league-average opponent.
    per-map ratings use per-map recency decay (date-based).
    filter_teams: if set, only emit these teams in the output (Massey solve uses all teams).
    CN teams are always excluded from the solve.
    """
    if not games:
        return {'n_games': 0, 'beta': 0, 'ref_date': None, 'teams': {}}

    # Overall Massey: pooled across all maps, used as shrinkage prior only
    rtgs     = massey_ratings(games, lam, ref_date, min_games=min_games)
    beta     = fit_beta(games, rtgs)
    map_rtgs = compute_per_map_ratings(games, rtgs, lam, shrink_k=shrink_k)
    records  = build_records(games)
    eff_cnts = effective_counts(games, lam, ref_date)

    # Maps with enough data for the veto simulation pool
    active_maps = [m for m, mrat in map_rtgs.items() if mrat]

    # League-average per-map rating (baseline opponent for pb_rating)
    avg_map_rtgs = {}
    for m in active_maps:
        vals = list(map_rtgs[m].values())
        avg_map_rtgs[m] = float(np.mean(vals)) if vals else 0.0

    # Headline overall_rating = raw decay-weighted Massey — same number the
    # live BenPom rating (rating_timeline.json) uses, so the historical
    # rankings page agrees with the modern hub & simulator on what an
    # "overall rating" means. The earlier Monte Carlo pick/ban adjustment
    # (pb_adjusted_rating) sometimes produced large deviations from a
    # team's actual record (e.g. TLN 27-27 rated +2.58) which read as a
    # bug to users even though the math was internally consistent. The
    # per-map ratings stored below are still shrunken toward this same
    # rtgs anchor (see compute_per_map_ratings), so the whole stack is
    # self-consistent.
    pb_ratings = {}
    for team in rtgs:
        rec = records.get(team, {'w': 0, 'l': 0, 'maps': {}})
        if rec['w'] + rec['l'] < min_games:
            continue
        pb_ratings[team] = rtgs.get(team, 0.0)

    teams_out = {}
    for team in sorted(pb_ratings, key=lambda t: -pb_ratings[t]):
        if team in CN_TEAMS:
            continue
        if filter_teams is not None and team not in filter_teams:
            continue
        rec = records.get(team, {'w': 0, 'l': 0, 'maps': {}})
        maps_out = {}
        for map_name, mrat in map_rtgs.items():
            if team not in mrat:
                continue
            mr = rec['maps'].get(map_name, {'w': 0, 'l': 0})
            total = mr['w'] + mr['l']
            if total == 0:
                continue
            maps_out[map_name] = {
                'rating':  round(mrat[team], 3),
                'w':       mr['w'],
                'l':       mr['l'],
                'win_pct': round(mr['w'] / total, 3),
            }
        total = rec['w'] + rec['l']
        teams_out[team] = {
            'overall_rating':  round(pb_ratings[team], 3),
            'w':               rec['w'],
            'l':               rec['l'],
            'win_pct':         round(rec['w'] / total, 3) if total else 0,
            'effective_games': eff_cnts.get(team, 0.0),
            'maps':            maps_out,
        }

    return {
        'n_games':  len(games),
        'beta':     round(beta, 4),
        'ref_date': ref_date.strftime('%Y-%m-%d') if ref_date else None,
        'teams':    teams_out,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    reoptimize = '--reoptimize' in sys.argv
    # Refresh-mode flag: only rebuild the current-year snapshots and reuse
    # historical snapshots verbatim from the existing JSON. Historical match
    # data is immutable, so re-doing 4 years of Monte Carlo each refresh is
    # pure waste. Cuts wall time from ~120s to ~5-10s on a typical refresh.
    refresh_mode = '--refresh' in sys.argv

    print('=' * 60)
    print('BuildMapRatings — VCT Domestic 2023-2025')
    if reoptimize:
        print('Mode: FULL (--reoptimize: running CV grid search)')
    elif refresh_mode:
        print(f'Mode: REFRESH (current-year only, fewer MC sims)')
    else:
        print(f'Mode: FAST (half-life={HALF_LIFE_WEEKS}w, shrink_k={SHRINK_K} — use --reoptimize to re-run CV)')
    print('=' * 60)

    # Detect current year + the events that make it up — used to decide whether
    # to load the full historical CSV pile or just the current-year subset.
    _years_sorted_int = sorted(int(y) for y in YEAR_CONFIGS.keys())
    _current_year_str = str(_years_sorted_int[-1]) if _years_sorted_int else None
    _current_year_events = set()
    if _current_year_str:
        for _snap_cfg in YEAR_CONFIGS[_current_year_str]['snapshots'].values():
            _current_year_events.update(_snap_cfg.get('events') or [])

    # Existing JSON — refresh mode reuses its metadata + historical snapshots
    _existing = {}
    if os.path.exists(OUT_PATH):
        try:
            with open(OUT_PATH) as _f:
                _existing = json.load(_f)
        except Exception:
            _existing = {}

    print('\nLoading games...')
    if refresh_mode and _existing:
        # Current-year-only load — historical CSVs aren't needed for the
        # rebuild (we reuse those snapshots verbatim) or for β calibration
        # (reused from the existing JSON's metadata).
        all_games = load_games(only_events=_current_year_events)
        train_games = []
        test_games  = all_games
        print(f'  [refresh] current-year only: {len(all_games)} map games')
    else:
        all_games    = load_games()
        train_games  = [g for g in all_games if g['event_id'] in TRAIN_EVENTS]
        test_games   = [g for g in all_games if g['event_id'] in TEST_EVENTS]
        print(f'  total: {len(all_games)} map games')
        print(f'  train: {len(train_games)}  ({", ".join(TRAIN_EVENTS)})')
        print(f'  test:  {len(test_games)}   ({", ".join(TEST_EVENTS)})')

    all_teams = sorted({g['winner'] for g in all_games} | {g['loser'] for g in all_games})
    print(f'  unique teams: {len(all_teams)}')

    train_ref_date = max((g['date'] for g in train_games), default=datetime.now())

    if not reoptimize:
        # ── Fast path: use fixed hyper-parameters ───────────────────────────────
        optimal_lambda = math.log(2) / HALF_LIFE_WEEKS
        best_sk        = SHRINK_K
        best_hl        = HALF_LIFE_WEEKS
        grid       = _existing.get('lambda_grid',   []) if _existing else []
        sk_results = _existing.get('shrink_k_grid', []) if _existing else []
        print(f'\n  [fast] λ = {optimal_lambda:.7f}  '
              f'half-life = {best_hl} weeks  shrink_k = {best_sk}')

        if refresh_mode and _existing and _existing.get('metadata'):
            # ── Step 2 (skipped): β reused from existing JSON ──────────────────
            _meta = _existing['metadata']
            beta  = float(_meta.get('beta_test') or _meta.get('beta') or 0.32)
            b_tr  = float(_meta.get('brier_train', 0.0))
            b_te  = float(_meta.get('brier_test',  0.0))
            ll_tr = float(_meta.get('logloss_train', 0.0))
            ll_te = float(_meta.get('logloss_test',  0.0))
            print(f'\n  [refresh] reusing β = {beta:.4f} from existing metadata '
                  f'(skipping calibration)')
        else:
            # ── Step 2: β calibration (cheap) ──────────────────────────────────
            print(f'\n{"─"*60}')
            print('Step 2: β calibration (held-out 2023-2024 → 2025)')
            rtgs_train = massey_ratings(train_games, optimal_lambda, train_ref_date)
            beta = fit_beta(train_games, rtgs_train)
            b_tr = brier_score(train_games, rtgs_train, beta)
            b_te = brier_score(test_games,  rtgs_train, beta)
            ll_tr = log_loss(train_games, rtgs_train, beta)
            ll_te = log_loss(test_games,  rtgs_train, beta)
            print(f'  β = {beta:.4f}')
            print(f'  Brier   — train: {b_tr:.5f}  test: {b_te:.5f}  (baseline 0.25)')
            print(f'  LogLoss — train: {ll_tr:.5f}  test: {ll_te:.5f}  (baseline {math.log(2):.5f})')

    else:
        # ── Step 1: CV grid search (for visualization only) ─────────────────────
        print(f'\n{"─"*60}')
        print('Step 1: λ grid search — rolling-window CV (date-based decay, weeks)')
        print(f'  Baseline Brier (always predict 50%): 0.25000')
        print(f'{"─"*60}')
        cv_lambda, grid = optimize_lambda(all_games)
        cv_hl = math.log(2) / cv_lambda
        print(f'\n  CV-optimal half-life: {cv_hl:.1f} weeks  (λ = {cv_lambda:.6f})')
        print(f'  Using fixed half-life: {HALF_LIFE_WEEKS} weeks  (HALF_LIFE_WEEKS constant)')
        # Computation always uses the fixed constant, not the CV result
        optimal_lambda = math.log(2) / HALF_LIFE_WEEKS
        best_hl        = HALF_LIFE_WEEKS

        # ── Step 2: calibrate β on train data ──────────────────────────────────
        print(f'\n{"─"*60}')
        print('Step 2: β calibration (held-out 2023-2024 → 2025)')
        rtgs_train = massey_ratings(train_games, optimal_lambda, train_ref_date)
        beta = fit_beta(train_games, rtgs_train)
        b_tr = brier_score(train_games, rtgs_train, beta)
        b_te = brier_score(test_games,  rtgs_train, beta)
        ll_tr = log_loss(train_games, rtgs_train, beta)
        ll_te = log_loss(test_games,  rtgs_train, beta)
        print(f'  β = {beta:.4f}')
        print(f'  Brier   — train: {b_tr:.5f}  test: {b_te:.5f}  (baseline 0.25)')
        print(f'  LogLoss — train: {ll_tr:.5f}  test: {ll_te:.5f}  (baseline {math.log(2):.5f})')

        # ── Step 3: tune shrink_k ────────────────────────────────────────────────
        print(f'\n{"─"*60}')
        print('Step 3: shrink_k tuning for per-map ratings')
        best_sk, best_sk_brier = SHRINK_K, 999.0
        sk_results = []
        for sk in [2, 4, 6, 8, 12]:
            pm = compute_per_map_ratings(train_games, rtgs_train, optimal_lambda,
                                         shrink_k=sk)
            per_map_preds = []
            for g in test_games:
                mn = g['map_name']
                rw = (pm.get(mn) or {}).get(g['winner'])
                rl = (pm.get(mn) or {}).get(g['loser'])
                if rw is None: rw = rtgs_train.get(g['winner'], 0.0)
                if rl is None: rl = rtgs_train.get(g['loser'],  0.0)
                per_map_preds.append((float(expit(beta * (rw - rl))) - 1.0) ** 2)
            sk_brier = float(np.mean(per_map_preds))
            sk_results.append({'shrink_k': sk, 'brier_test': round(sk_brier, 5)})
            flag = ' <--' if sk_brier < best_sk_brier else ''
            print(f'  shrink_k={sk}  brier_test={sk_brier:.5f}{flag}')
            if sk_brier < best_sk_brier:
                best_sk, best_sk_brier = sk, sk_brier
        print(f'  Optimal shrink_k: {best_sk}')

    # ── Step 4: per-year ratings ─────────────────────────────────────────────────
    print(f'\n{"─"*60}')
    print('Step 4: computing per-year ratings (each snapshot)')

    # Refresh mode: rebuild only the latest year (live data); historical years
    # come straight from the existing on-disk JSON since their matches are
    # immutable.
    skip_years = set()
    existing_ratings = (_existing.get('ratings') or {}) if _existing else {}
    if refresh_mode and existing_ratings and _current_year_str:
        for _y in YEAR_CONFIGS:
            if _y != _current_year_str and _y in existing_ratings:
                skip_years.add(_y)
        if skip_years:
            print(f'  [refresh] rebuilding {_current_year_str} only; '
                  f'reusing snapshots for {sorted(skip_years)}')

    # In refresh mode, drop MC sims hard — precision still well within the
    # noise floor of the simulator's win-prob output, and the inner-loop sim
    # dominates wall time. Monkey-patches the module constant so
    # build_year_ratings → pb_adjusted_rating picks up the reduced count
    # without threading a new kwarg through.
    if refresh_mode:
        globals()['MC_N_SIMS'] = 600

    ratings_out = {}
    for year, cfg in YEAR_CONFIGS.items():
        if year in skip_years:
            ratings_out[year] = existing_ratings[year]
            continue
        snaps_out = {}
        for snap_id, snap_cfg in cfg['snapshots'].items():
            snap_games = [g for g in all_games if g['event_id'] in snap_cfg['events']]
            ref_date   = max(g['date'] for g in snap_games) if snap_games else datetime.now()

            # For after_champions snapshots, only emit teams that attended Champions
            champs_filter_eid = snap_cfg.get('champs_filter')
            filter_teams = None
            if champs_filter_eid:
                filter_teams = (
                    {g['winner'] for g in all_games if g['event_id'] == champs_filter_eid} |
                    {g['loser']  for g in all_games if g['event_id'] == champs_filter_eid}
                )

            snap_data = build_year_ratings(
                snap_games, optimal_lambda, ref_date,
                shrink_k=best_sk, min_games=cfg['min_games'],
                filter_teams=filter_teams,
            )
            snap_data['label'] = snap_cfg['label']

            # Qualification cap: only for snapshots ending with a completed international
            last_event = snap_cfg['events'][-1] if snap_cfg['events'] else None
            n_capped = 0
            if last_event and last_event in INTL_EVENTS:
                snap_data['teams'], n_capped = apply_qualification_cap(
                    snap_data['teams'], last_event, all_games
                )

            snaps_out[snap_id] = snap_data
            n_filtered = len(filter_teams) if filter_teams else '—'
            cap_note = f', qual cap: {n_capped} capped' if n_capped else ''
            print(f'  {year}/{snap_id}: {len(snap_games)} games  '
                  f'({len(snap_data["teams"])} teams shown'
                  f'{", champs filter: " + str(n_filtered) + " attendees" if filter_teams else ""}'
                  f'{cap_note},'
                  f' beta={snap_data["beta"]})')
        ratings_out[year] = {'snapshots': snaps_out}

    # ── Step 5: assemble and save ────────────────────────────────────────────────
    print(f'\n{"─"*60}')
    print('Step 5: saving output JSON')

    out = {
        'metadata': {
            'generated':                   datetime.now().isoformat()[:10],
            'train_events':                TRAIN_EVENTS,
            'test_events':                 TEST_EVENTS,
            'optimal_lambda':              round(optimal_lambda, 7),
            'optimal_half_life_weeks':      round(best_hl, 1),
            'shrink_k':                    best_sk,
            'mc_n_sims':                   MC_N_SIMS,
            'veto_noise_std':              VETO_NOISE_STD,
            'brier_train':                 round(b_tr, 5),
            'brier_test':                  round(b_te, 5),
            'logloss_train':               round(ll_tr, 5),
            'logloss_test':                round(ll_te, 5),
            'n_train':                     len(train_games),
            'n_test':                      len(test_games),
        },
        'lambda_grid':   grid,
        'shrink_k_grid': sk_results,
        'ratings':       ratings_out,
    }

    with open(OUT_PATH, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'  Saved -> {OUT_PATH}')

    year_last_snap = {
        '2026': 'after_santiago',
        '2025': 'after_champions',
        '2024': 'after_champions',
        '2023': 'after_champions',
    }
    for year in ['2026', '2025', '2024', '2023']:
        snap_id = year_last_snap[year]
        snap = ratings_out.get(year, {}).get('snapshots', {}).get(snap_id, {})
        print(f'\nTop 10 — {year} ({snap_id}):')
        print(f'  {"#":>2}  {"Team":>6}  {"Rating":>7}  {"W-L":>7}  {"Win%":>5}')
        for i, (team, d) in enumerate(list(snap.get('teams', {}).items())[:10], 1):
            print(f'  {i:2}.  {team:>6}  {d["overall_rating"]:+7.3f}  '
                  f'{d["w"]}-{d["l"]:>2}  {d["win_pct"]:.3f}')

    print(f'\nDone.')


if __name__ == '__main__':
    main()
