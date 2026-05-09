"""
intl_calib_research.py

Deep research on alternative international calibration approaches.
Tests 10+ methods against domestic-only Brier baseline at 2024_champions and 2025_champions.

Usage: python3 intl_calib_research.py
"""

import os, sys, json, math, itertools
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime
from scipy.special import expit

ROOT = '/Users/benny_es1/PythonTest'
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, 'data')

from EventLeaderboards import ORG_REGIONS

ACTIVE_REGIONS = {'EMEA', 'Americas', 'Pacific', 'CN'}

# ── Replicate config from BuildIntlCalibration ───────────────────────────────

INTL_CONFIG = {
    '2023_masters_tokyo':   {'year': '2023', 'snap': 'after_tokyo',      'date': '2023-06-11'},
    '2023_champions':       {'year': '2023', 'snap': 'before_champions', 'date': '2023-08-06'},
    '2024_masters_madrid':  {'year': '2024', 'snap': 'before_madrid',    'date': '2024-02-14'},
    '2024_masters_shanghai':{'year': '2024', 'snap': 'before_shanghai',  'date': '2024-06-02'},
    '2024_champions':       {'year': '2024', 'snap': 'before_champions', 'date': '2024-08-01'},
    '2025_masters_bangkok': {'year': '2025', 'snap': 'before_bangkok',   'date': '2025-02-19'},
    '2025_masters_toronto': {'year': '2025', 'snap': 'before_toronto',   'date': '2025-06-19'},
    '2025_champions':       {'year': '2025', 'snap': 'before_champions', 'date': '2025-08-22'},
    '2026_masters_santiago':{'year': '2026', 'snap': 'before_santiago',  'date': '2026-03-26'},
}

SNAP_CONFIG = {
    ('2023', 'after_tokyo'):      {'prior_intl': ['2023_masters_tokyo'],                              'date': '2023-06-26'},
    ('2023', 'before_champions'): {'prior_intl': ['2023_masters_tokyo'],                              'date': '2023-08-05'},
    ('2023', 'after_champions'):  {'prior_intl': ['2023_masters_tokyo', '2023_champions'],             'date': '2023-09-01'},
    ('2024', 'before_madrid'):    {'prior_intl': [],                                                   'date': '2024-02-12'},
    ('2024', 'after_madrid'):     {'prior_intl': ['2024_masters_madrid'],                              'date': '2024-03-12'},
    ('2024', 'before_shanghai'):  {'prior_intl': ['2024_masters_madrid'],                              'date': '2024-06-01'},
    ('2024', 'after_shanghai'):   {'prior_intl': ['2024_masters_madrid', '2024_masters_shanghai'],     'date': '2024-06-18'},
    ('2024', 'before_champions'): {'prior_intl': ['2024_masters_madrid', '2024_masters_shanghai'],     'date': '2024-08-01'},
    ('2024', 'after_champions'):  {'prior_intl': ['2024_masters_madrid', '2024_masters_shanghai', '2024_champions'], 'date': '2024-09-25'},
    ('2025', 'before_bangkok'):   {'prior_intl': [],                                                   'date': '2025-02-11'},
    ('2025', 'after_bangkok'):    {'prior_intl': ['2025_masters_bangkok'],                             'date': '2025-03-11'},
    ('2025', 'before_toronto'):   {'prior_intl': ['2025_masters_bangkok'],                             'date': '2025-06-06'},
    ('2025', 'after_toronto'):    {'prior_intl': ['2025_masters_bangkok', '2025_masters_toronto'],     'date': '2025-07-01'},
    ('2025', 'before_champions'): {'prior_intl': ['2025_masters_bangkok', '2025_masters_toronto'],     'date': '2025-08-27'},
    ('2025', 'after_champions'):  {'prior_intl': ['2025_masters_bangkok', '2025_masters_toronto', '2025_champions'], 'date': '2025-09-25'},
    ('2026', 'before_santiago'):  {'prior_intl': [],                                                   'date': '2026-02-09'},
    ('2026', 'after_santiago'):   {'prior_intl': ['2026_masters_santiago'],                            'date': '2026-04-08'},
}

BACKTEST_HOLDOUTS = ['2024_champions', '2025_champions']

# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_date(s):
    return datetime.strptime(s, '%Y-%m-%d')

def _weeks_between(d1_str, d2_str):
    return abs((_parse_date(d2_str) - _parse_date(d1_str)).days) / 7.0

def _decay(weeks, hl):
    return math.exp(-math.log(2) / hl * weeks)

def load_ratings():
    with open(os.path.join(DATA, 'map_ratings.json')) as f:
        return json.load(f)

def get_snap_ratings(ratings_data, year, snap):
    try:
        return {org: d['overall_rating']
                for org, d in ratings_data['ratings'][year]['snapshots'][snap]['teams'].items()}
    except (KeyError, TypeError):
        return {}

def get_snap_beta(ratings_data, year, snap):
    try:
        return ratings_data['ratings'][year]['snapshots'][snap]['beta']
    except Exception:
        return 0.45

# ── Load international game results (map level) ───────────────────────────────

def load_intl_games(event_id, include_cn=False):
    """
    Returns list of map-game dicts (cross-regional only).
    include_cn=False skips CN teams.
    """
    maps_path = os.path.join(DATA, 'maps', f'{event_id}.csv')
    mr_path   = os.path.join(DATA, 'match_results.csv')
    if not os.path.exists(maps_path):
        return []

    mr = pd.read_csv(mr_path)
    mr = mr[mr['MapNum'] != 'all'].copy()
    mr_idx = mr.set_index(['MatchID', 'MapNum'])

    df = pd.read_csv(maps_path)
    df['MapNum'] = df['MapNum'].astype(str)

    meta = df.groupby(['MatchID', 'MapNum']).agg(
        orgs=('Org', lambda x: list(x.unique()))
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
        loser  = losers[0]
        wr_reg = ORG_REGIONS.get(winner)
        lr_reg = ORG_REGIONS.get(loser)
        if not wr_reg or not lr_reg:
            continue
        if wr_reg not in ACTIVE_REGIONS or lr_reg not in ACTIVE_REGIONS:
            continue
        if not include_cn and (wr_reg == 'CN' or lr_reg == 'CN'):
            continue
        if wr_reg == lr_reg:
            continue
        try:
            wr, lr = map(int, str(mr_row['Score']).split('-'))
        except Exception:
            continue
        games.append({
            'winner': winner, 'loser': loser,
            'wr': wr, 'lr': lr,
            'winner_region': wr_reg, 'loser_region': lr_reg,
            'match_id': int(row['MatchID']),
        })
    return games


def load_intl_matches(event_id, include_cn=False):
    """
    Load match-level (series) results for an international event.
    Returns list of {winner, loser, w_maps, l_maps, winner_region, loser_region}.
    """
    mr_path = os.path.join(DATA, 'match_results.csv')
    maps_path = os.path.join(DATA, 'maps', f'{event_id}.csv')
    if not os.path.exists(maps_path):
        return []

    mr = pd.read_csv(mr_path)
    # Match-level rows have MapNum == 'all'
    mr_all = mr[mr['MapNum'] == 'all'].copy()

    # Get orgs from maps CSV
    df = pd.read_csv(maps_path)
    match_orgs = df.groupby('MatchID')['Org'].apply(lambda x: list(x.unique())).reset_index()
    match_orgs.columns = ['MatchID', 'orgs']
    match_orgs_idx = match_orgs.set_index('MatchID')

    matches = []
    for _, row in mr_all.iterrows():
        mid = int(row['MatchID'])
        if mid not in match_orgs_idx.index:
            continue
        orgs   = match_orgs_idx.loc[mid, 'orgs']
        winner = row['WinnerOrg']
        losers = [o for o in orgs if o != winner]
        if not losers:
            continue
        loser  = losers[0]
        wr_reg = ORG_REGIONS.get(winner)
        lr_reg = ORG_REGIONS.get(loser)
        if not wr_reg or not lr_reg:
            continue
        if wr_reg not in ACTIVE_REGIONS or lr_reg not in ACTIVE_REGIONS:
            continue
        if not include_cn and (wr_reg == 'CN' or lr_reg == 'CN'):
            continue
        if wr_reg == lr_reg:
            continue
        score_str = str(row['Score'])
        try:
            wm, lm = map(int, score_str.split('-'))
        except Exception:
            wm, lm = 1, 0
        matches.append({
            'winner': winner, 'loser': loser,
            'w_maps': wm, 'l_maps': lm,
            'winner_region': wr_reg, 'loser_region': lr_reg,
        })
    return matches


# ── Core prediction / scoring ─────────────────────────────────────────────────

def win_prob(rA, rB, beta):
    return float(np.clip(expit(beta * (rA - rB)), 1e-9, 1 - 1e-9))

def brier(p):   return (p - 1.0) ** 2
def logloss(p): return -math.log(p)


# ── Backtest engine ───────────────────────────────────────────────────────────

def run_backtest(method_name, regional_offsets_by_snap, individual_bonuses_by_snap,
                 ratings_data, verbose=False):
    """
    Returns dict: event -> {brier, logloss, n}  for calibrated model.
    Also returns domestic-only scores for comparison.
    """
    results = {}

    for eid in BACKTEST_HOLDOUTS:
        cfg    = INTL_CONFIG.get(eid, {})
        yr, sn = cfg.get('year'), cfg.get('snap')
        if not yr or not sn:
            continue

        pre_rtgs = get_snap_ratings(ratings_data, yr, sn)
        beta     = get_snap_beta(ratings_data, yr, sn)
        snap_key = f'{yr}_{sn}'

        reg_off = regional_offsets_by_snap.get(snap_key, {})
        ind_bon = individual_bonuses_by_snap.get(snap_key, {})

        games = load_intl_games(eid)
        if not games:
            continue

        brier_dom, brier_cal = [], []
        ll_dom,    ll_cal    = [], []

        for g in games:
            rw_dom = pre_rtgs.get(g['winner'])
            rl_dom = pre_rtgs.get(g['loser'])
            if rw_dom is None or rl_dom is None:
                continue

            rw_reg = ORG_REGIONS.get(g['winner'], '')
            rl_reg = ORG_REGIONS.get(g['loser'],  '')

            rw_cal = rw_dom + reg_off.get(rw_reg, 0) + ind_bon.get(g['winner'], 0)
            rl_cal = rl_dom + reg_off.get(rl_reg, 0) + ind_bon.get(g['loser'],  0)

            p_dom = win_prob(rw_dom, rl_dom, beta)
            p_cal = win_prob(rw_cal, rl_cal, beta)

            brier_dom.append(brier(p_dom));  ll_dom.append(logloss(p_dom))
            brier_cal.append(brier(p_cal));  ll_cal.append(logloss(p_cal))

        if not brier_dom:
            continue

        results[eid] = {
            'n':            len(brier_dom),
            'brier_dom':    float(np.mean(brier_dom)),
            'brier_cal':    float(np.mean(brier_cal)),
            'll_dom':       float(np.mean(ll_dom)),
            'll_cal':       float(np.mean(ll_cal)),
        }

    return results


def aggregate_brier(results):
    """Weighted average brier_cal across holdout events."""
    total_b, total_n = 0.0, 0
    for r in results.values():
        total_b += r['brier_cal'] * r['n']
        total_n += r['n']
    return total_b / total_n if total_n else float('nan')

def aggregate_brier_dom(results):
    total_b, total_n = 0.0, 0
    for r in results.values():
        total_b += r['brier_dom'] * r['n']
        total_n += r['n']
    return total_b / total_n if total_n else float('nan')


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL COMPUTATION METHODS
# Each method returns (regional_offsets_by_snap, individual_bonuses_by_snap)
# where keys are like '2024_before_champions'
# ═══════════════════════════════════════════════════════════════════════════════

# ── Helper: normalise regional offsets so mean = 0 ────────────────────────────

def _normalise(raw_delta):
    if not raw_delta:
        return {}
    gm = float(np.mean(list(raw_delta.values())))
    return {r: v - gm for r, v in raw_delta.items()}


# ── Method 0: Baseline — domestic only ───────────────────────────────────────

def method_baseline(ratings_data):
    reg_snaps, ind_snaps = {}, {}
    for (yr, sn) in SNAP_CONFIG:
        key = f'{yr}_{sn}'
        reg_snaps[key] = {}
        ind_snaps[key] = {}
    return reg_snaps, ind_snaps


# ── Method 1: Win-rate residuals (map level) ──────────────────────────────────

def method_winrate_residuals(ratings_data, hl_weeks=10, reg_scale=0.12, ind_scale=0.10,
                              ind_shrink=3, regional_only=False, individual_only=False):
    """
    Residual = actual_win (1/0) - predicted_win_prob from domestic ratings.
    Less noisy than round-diff; bounded [−1, 1].
    """
    event_cache = {}

    def get_event_wr_resids(eid):
        if eid in event_cache:
            return event_cache[eid]
        cfg    = INTL_CONFIG.get(eid, {})
        yr, sn = cfg.get('year'), cfg.get('snap')
        pre    = get_snap_ratings(ratings_data, yr, sn) if yr and sn else {}
        beta   = get_snap_beta(ratings_data, yr, sn) if yr and sn else 0.45
        games  = load_intl_games(eid)

        team_resids = defaultdict(list)
        for g in games:
            rw = pre.get(g['winner'])
            rl = pre.get(g['loser'])
            if rw is None or rl is None:
                continue
            p_win  = win_prob(rw, rl, beta)
            resid_w = 1.0 - p_win   # winner: got 1, expected p
            resid_l = 0.0 - (1 - p_win)  # loser: got 0, expected (1-p)
            team_resids[g['winner']].append(resid_w)
            team_resids[g['loser']].append(resid_l)

        event_cache[eid] = team_resids
        return team_resids

    reg_snaps, ind_snaps = {}, {}

    for (yr, sn), cfg in SNAP_CONFIG.items():
        key         = f'{yr}_{sn}'
        snap_date   = cfg['date']
        prior_events = cfg['prior_intl']

        # skip holdouts
        skip_events = set(BACKTEST_HOLDOUTS)
        prior_events = [e for e in prior_events if e not in skip_events]

        acc_reg = defaultdict(float)
        acc_ind = defaultdict(float)

        for eid in prior_events:
            team_resids = get_event_wr_resids(eid)
            intl_date   = INTL_CONFIG[eid]['date']
            w           = _decay(_weeks_between(intl_date, snap_date), hl_weeks)

            # Regional averages
            reg_raw = defaultdict(list)
            for team, resids in team_resids.items():
                reg = ORG_REGIONS.get(team)
                if reg in ACTIVE_REGIONS:
                    reg_raw[reg].extend(resids)

            raw_delta = {reg: float(np.mean(v)) for reg, v in reg_raw.items() if v}
            delta     = _normalise(raw_delta)

            # Individual bonuses
            ind_raw = {}
            for team, resids in team_resids.items():
                reg = ORG_REGIONS.get(team)
                if reg not in ACTIVE_REGIONS:
                    continue
                n     = len(resids)
                raw_b = float(np.mean(resids)) - delta.get(reg, 0.0)
                alpha = n / (n + ind_shrink)
                ind_raw[team] = alpha * raw_b

            if not regional_only:
                for team, bonus in ind_raw.items():
                    acc_ind[team] += bonus * ind_scale * w
            if not individual_only:
                for reg, offset in delta.items():
                    acc_reg[reg] += offset * reg_scale * w

        reg_snaps[key] = dict(acc_reg)
        ind_snaps[key] = dict(acc_ind)

    return reg_snaps, ind_snaps


# ── Method 2: Logistic (log-loss) residuals ───────────────────────────────────

def method_logloss_residuals(ratings_data, hl_weeks=10, reg_scale=0.08, ind_scale=0.06,
                               ind_shrink=3, regional_only=False):
    """
    Residual = gradient of log-loss w.r.t. rating:
      For winner: +(1 - p_win)  (bounded 0..1, gradient direction pushes up)
      For loser:  -(p_win)      (bounded 0..1, gradient pushes down)
    Equivalent to win-rate residuals — same formula, just named differently.
    Included separately to test different scales.
    """
    return method_winrate_residuals(ratings_data, hl_weeks=hl_weeks,
                                    reg_scale=reg_scale, ind_scale=ind_scale,
                                    ind_shrink=ind_shrink, regional_only=regional_only)


# ── Method 3: Match-level (series) win-rate residuals ─────────────────────────

def method_match_winrate(ratings_data, hl_weeks=10, reg_scale=0.12, ind_scale=0.10,
                          ind_shrink=3, regional_only=False):
    """
    Use series (match) wins instead of map wins. Reduces map-level noise.
    Uses the 'all' rows from match_results.csv.
    """
    event_cache = {}

    def get_event_match_resids(eid):
        if eid in event_cache:
            return event_cache[eid]
        cfg    = INTL_CONFIG.get(eid, {})
        yr, sn = cfg.get('year'), cfg.get('snap')
        pre    = get_snap_ratings(ratings_data, yr, sn) if yr and sn else {}
        beta   = get_snap_beta(ratings_data, yr, sn) if yr and sn else 0.45
        matches = load_intl_matches(eid)

        team_resids = defaultdict(list)
        for m in matches:
            rw = pre.get(m['winner'])
            rl = pre.get(m['loser'])
            if rw is None or rl is None:
                continue
            p_win   = win_prob(rw, rl, beta)
            resid_w = 1.0 - p_win
            resid_l = 0.0 - (1 - p_win)
            team_resids[m['winner']].append(resid_w)
            team_resids[m['loser']].append(resid_l)

        event_cache[eid] = team_resids
        return team_resids

    reg_snaps, ind_snaps = {}, {}

    for (yr, sn), cfg in SNAP_CONFIG.items():
        key          = f'{yr}_{sn}'
        snap_date    = cfg['date']
        prior_events = [e for e in cfg['prior_intl'] if e not in BACKTEST_HOLDOUTS]

        acc_reg = defaultdict(float)
        acc_ind = defaultdict(float)

        for eid in prior_events:
            team_resids = get_event_match_resids(eid)
            intl_date   = INTL_CONFIG[eid]['date']
            w           = _decay(_weeks_between(intl_date, snap_date), hl_weeks)

            reg_raw = defaultdict(list)
            for team, resids in team_resids.items():
                reg = ORG_REGIONS.get(team)
                if reg in ACTIVE_REGIONS:
                    reg_raw[reg].extend(resids)

            raw_delta = {reg: float(np.mean(v)) for reg, v in reg_raw.items() if v}
            delta     = _normalise(raw_delta)

            ind_raw = {}
            for team, resids in team_resids.items():
                reg = ORG_REGIONS.get(team)
                if reg not in ACTIVE_REGIONS:
                    continue
                n     = len(resids)
                raw_b = float(np.mean(resids)) - delta.get(reg, 0.0)
                alpha = n / (n + ind_shrink)
                ind_raw[team] = alpha * raw_b

            if not regional_only:
                for team, bonus in ind_raw.items():
                    acc_ind[team] += bonus * ind_scale * w
            for reg, offset in delta.items():
                acc_reg[reg] += offset * reg_scale * w

        reg_snaps[key] = dict(acc_reg)
        ind_snaps[key] = dict(acc_ind)

    return reg_snaps, ind_snaps


# ── Method 4: Regional dominance ratio ───────────────────────────────────────

def method_dominance_ratio(ratings_data, hl_weeks=10, reg_scale=0.12):
    """
    For each region: ratio = actual_cross_wins / expected_cross_wins
    Signal = log(ratio) (or ratio - 1 for small values).
    Regional-only; no individual bonus.
    """
    event_cache = {}

    def get_event_dom_ratio(eid):
        if eid in event_cache:
            return event_cache[eid]
        cfg    = INTL_CONFIG.get(eid, {})
        yr, sn = cfg.get('year'), cfg.get('snap')
        pre    = get_snap_ratings(ratings_data, yr, sn) if yr and sn else {}
        beta   = get_snap_beta(ratings_data, yr, sn) if yr and sn else 0.45
        games  = load_intl_games(eid)

        actual_wins  = defaultdict(int)
        expected_wins = defaultdict(float)

        for g in games:
            rw = pre.get(g['winner'])
            rl = pre.get(g['loser'])
            if rw is None or rl is None:
                continue
            p = win_prob(rw, rl, beta)
            wr_reg, lr_reg = g['winner_region'], g['loser_region']
            actual_wins[wr_reg]   += 1
            actual_wins[lr_reg]   += 0
            expected_wins[wr_reg] += p
            expected_wins[lr_reg] += (1 - p)

        # ratio of actual wins to expected wins (for each region)
        dom_signal = {}
        for reg in set(actual_wins) | set(expected_wins):
            exp = expected_wins.get(reg, 0)
            act = actual_wins.get(reg, 0)
            if exp < 1:
                continue
            # Use (act - exp) / exp  as percentage over/underperformance
            dom_signal[reg] = (act - exp) / exp

        event_cache[eid] = dom_signal
        return dom_signal

    reg_snaps, ind_snaps = {}, {}

    for (yr, sn), cfg in SNAP_CONFIG.items():
        key          = f'{yr}_{sn}'
        snap_date    = cfg['date']
        prior_events = [e for e in cfg['prior_intl'] if e not in BACKTEST_HOLDOUTS]

        acc_reg = defaultdict(float)

        for eid in prior_events:
            dom_signal = get_event_dom_ratio(eid)
            intl_date  = INTL_CONFIG[eid]['date']
            w          = _decay(_weeks_between(intl_date, snap_date), hl_weeks)

            raw_delta = dict(dom_signal)
            delta     = _normalise(raw_delta)

            for reg, offset in delta.items():
                acc_reg[reg] += offset * reg_scale * w

        reg_snaps[key] = dict(acc_reg)
        ind_snaps[key] = {}

    return reg_snaps, ind_snaps


# ── Method 5: Median-based residuals (robust to outliers) ─────────────────────

def method_median_residuals(ratings_data, hl_weeks=10, reg_scale=0.12, ind_scale=0.10,
                             ind_shrink=3, use_winrate=True):
    """
    Use median instead of mean for regional offsets.
    Robust to one dominant team distorting the region signal.
    """
    event_cache = {}

    def get_event_resids(eid):
        if eid in event_cache:
            return event_cache[eid]
        cfg    = INTL_CONFIG.get(eid, {})
        yr, sn = cfg.get('year'), cfg.get('snap')
        pre    = get_snap_ratings(ratings_data, yr, sn) if yr and sn else {}
        beta   = get_snap_beta(ratings_data, yr, sn) if yr and sn else 0.45
        games  = load_intl_games(eid)

        team_resids = defaultdict(list)
        for g in games:
            rw = pre.get(g['winner'])
            rl = pre.get(g['loser'])
            if rw is None or rl is None:
                continue
            if use_winrate:
                p_win   = win_prob(rw, rl, beta)
                resid_w = 1.0 - p_win
                resid_l = 0.0 - (1 - p_win)
            else:
                actual_rd  = g['wr'] - g['lr']
                pred_rd    = rw - rl
                resid_w    = actual_rd - pred_rd
                resid_l    = -resid_w
            team_resids[g['winner']].append(resid_w)
            team_resids[g['loser']].append(resid_l)

        event_cache[eid] = team_resids
        return team_resids

    reg_snaps, ind_snaps = {}, {}

    for (yr, sn), cfg in SNAP_CONFIG.items():
        key          = f'{yr}_{sn}'
        snap_date    = cfg['date']
        prior_events = [e for e in cfg['prior_intl'] if e not in BACKTEST_HOLDOUTS]

        acc_reg = defaultdict(float)
        acc_ind = defaultdict(float)

        for eid in prior_events:
            team_resids = get_event_resids(eid)
            intl_date   = INTL_CONFIG[eid]['date']
            w           = _decay(_weeks_between(intl_date, snap_date), hl_weeks)

            # Build per-team average, then take MEDIAN across teams per region
            team_avgs = {team: float(np.mean(rs)) for team, rs in team_resids.items()}

            reg_team_avgs = defaultdict(list)
            for team, avg in team_avgs.items():
                reg = ORG_REGIONS.get(team)
                if reg in ACTIVE_REGIONS:
                    reg_team_avgs[reg].append(avg)

            raw_delta = {reg: float(np.median(v)) for reg, v in reg_team_avgs.items() if v}
            delta     = _normalise(raw_delta)

            ind_raw = {}
            for team, resids in team_resids.items():
                reg = ORG_REGIONS.get(team)
                if reg not in ACTIVE_REGIONS:
                    continue
                n     = len(resids)
                raw_b = float(np.mean(resids)) - delta.get(reg, 0.0)
                alpha = n / (n + 3)
                ind_raw[team] = alpha * raw_b

            for reg, offset in delta.items():
                acc_reg[reg] += offset * reg_scale * w
            for team, bonus in ind_raw.items():
                acc_ind[team] += bonus * ind_scale * w

        reg_snaps[key] = dict(acc_reg)
        ind_snaps[key] = dict(acc_ind)

    return reg_snaps, ind_snaps


# ── Method 6: Sample-size scaled regional (sqrt(n_games)) ─────────────────────

def method_sample_scaled(ratings_data, hl_weeks=10, base_reg_scale=0.04, ind_scale=0.10,
                          ind_shrink=3):
    """
    Trust regional signal more when more cross-regional games were played.
    Regional offset scale = base_scale * sqrt(n_cross_games_for_region).
    """
    event_cache = {}

    def get_event_data(eid):
        if eid in event_cache:
            return event_cache[eid]
        cfg    = INTL_CONFIG.get(eid, {})
        yr, sn = cfg.get('year'), cfg.get('snap')
        pre    = get_snap_ratings(ratings_data, yr, sn) if yr and sn else {}
        beta   = get_snap_beta(ratings_data, yr, sn) if yr and sn else 0.45
        games  = load_intl_games(eid)

        team_resids = defaultdict(list)
        for g in games:
            rw = pre.get(g['winner'])
            rl = pre.get(g['loser'])
            if rw is None or rl is None:
                continue
            p_win   = win_prob(rw, rl, beta)
            team_resids[g['winner']].append(1.0 - p_win)
            team_resids[g['loser']].append(0.0 - (1 - p_win))

        event_cache[eid] = team_resids
        return team_resids

    reg_snaps, ind_snaps = {}, {}

    for (yr, sn), cfg in SNAP_CONFIG.items():
        key          = f'{yr}_{sn}'
        snap_date    = cfg['date']
        prior_events = [e for e in cfg['prior_intl'] if e not in BACKTEST_HOLDOUTS]

        acc_reg = defaultdict(float)
        acc_ind = defaultdict(float)

        for eid in prior_events:
            team_resids = get_event_data(eid)
            intl_date   = INTL_CONFIG[eid]['date']
            w           = _decay(_weeks_between(intl_date, snap_date), hl_weeks)

            reg_raw = defaultdict(list)
            for team, resids in team_resids.items():
                reg = ORG_REGIONS.get(team)
                if reg in ACTIVE_REGIONS:
                    reg_raw[reg].extend(resids)

            raw_delta = {reg: float(np.mean(v)) for reg, v in reg_raw.items() if v}
            delta     = _normalise(raw_delta)

            ind_raw = {}
            for team, resids in team_resids.items():
                reg = ORG_REGIONS.get(team)
                if reg not in ACTIVE_REGIONS:
                    continue
                n     = len(resids)
                raw_b = float(np.mean(resids)) - delta.get(reg, 0.0)
                alpha = n / (n + ind_shrink)
                ind_raw[team] = alpha * raw_b

            for reg, offset in delta.items():
                n_games = len(reg_raw.get(reg, []))
                scale   = base_reg_scale * math.sqrt(n_games)
                acc_reg[reg] += offset * scale * w

            for team, bonus in ind_raw.items():
                acc_ind[team] += bonus * ind_scale * w

        reg_snaps[key] = dict(acc_reg)
        ind_snaps[key] = dict(acc_ind)

    return reg_snaps, ind_snaps


# ── Method 7: Regional-only (no individual bonuses) ───────────────────────────

def method_regional_only(ratings_data, hl_weeks=10, reg_scale=0.12, use_winrate=True):
    """Pure regional signal, zero individual bonuses."""
    r, i = method_winrate_residuals(ratings_data, hl_weeks=hl_weeks,
                                    reg_scale=reg_scale, ind_scale=0.0,
                                    ind_shrink=999, regional_only=True)
    return r, i


# ── Method 8: Individual-only (no regional) ───────────────────────────────────

def method_individual_only(ratings_data, hl_weeks=10, ind_scale=0.10, ind_shrink=3):
    """Individual bonuses only, zero regional offsets."""
    r, i = method_winrate_residuals(ratings_data, hl_weeks=hl_weeks,
                                    reg_scale=0.0, ind_scale=ind_scale,
                                    ind_shrink=ind_shrink, individual_only=True)
    return r, i


# ── Method 9: Multi-event long history (20-week half-life) ───────────────────

def method_long_hl(ratings_data, hl_weeks=26, reg_scale=0.12, ind_scale=0.10, ind_shrink=3):
    """Longer half-life — slower decay, value older events more."""
    return method_winrate_residuals(ratings_data, hl_weeks=hl_weeks,
                                    reg_scale=reg_scale, ind_scale=ind_scale,
                                    ind_shrink=ind_shrink)


# ── Method 10: Short half-life (3-week) ──────────────────────────────────────

def method_short_hl(ratings_data, hl_weeks=3, reg_scale=0.15, ind_scale=0.12, ind_shrink=3):
    """Very short half-life — only very recent internationals matter much."""
    return method_winrate_residuals(ratings_data, hl_weeks=hl_weeks,
                                    reg_scale=reg_scale, ind_scale=ind_scale,
                                    ind_shrink=ind_shrink)


# ── Method 11: Match-level regional only ─────────────────────────────────────

def method_match_regional_only(ratings_data, hl_weeks=10, reg_scale=0.12):
    """Match-level wins, regional offset only."""
    return method_match_winrate(ratings_data, hl_weeks=hl_weeks,
                                reg_scale=reg_scale, ind_scale=0.0,
                                ind_shrink=999, regional_only=True)


# ═══════════════════════════════════════════════════════════════════════════════
# GRID SEARCH
# ═══════════════════════════════════════════════════════════════════════════════

def grid_search_winrate(ratings_data):
    """
    Grid search over (hl_weeks, reg_scale, ind_scale, ind_shrink, regional_only).
    Return best params and scores.
    """
    best_brier = float('inf')
    best_params = None
    best_rows   = []

    all_rows = []

    for hl in [3, 6, 10, 16, 26, 52]:
        for reg_scale in [0.0, 0.04, 0.08, 0.12, 0.18, 0.25]:
            for ind_scale in [0.0, 0.05, 0.10, 0.15]:
                for ind_shrink in [2, 3, 5]:
                    for reg_only in [False]:
                        for ind_only in [False]:
                            try:
                                r, i = method_winrate_residuals(
                                    ratings_data,
                                    hl_weeks=hl,
                                    reg_scale=reg_scale,
                                    ind_scale=ind_scale,
                                    ind_shrink=ind_shrink,
                                    regional_only=reg_only,
                                    individual_only=ind_only,
                                )
                                res = run_backtest('wr', r, i, ratings_data)
                                b   = aggregate_brier(res)
                                row = {
                                    'hl': hl, 'reg_scale': reg_scale,
                                    'ind_scale': ind_scale, 'ind_shrink': ind_shrink,
                                    'brier': b,
                                }
                                all_rows.append(row)
                                if b < best_brier:
                                    best_brier  = b
                                    best_params = row.copy()
                            except Exception as e:
                                pass

    return best_params, sorted(all_rows, key=lambda x: x['brier'])[:15]


def grid_search_match(ratings_data):
    """Grid search for match-level approach."""
    best_brier = float('inf')
    best_params = None
    all_rows    = []

    for hl in [3, 6, 10, 16, 26]:
        for reg_scale in [0.0, 0.06, 0.12, 0.18, 0.25]:
            for ind_scale in [0.0, 0.06, 0.12]:
                for ind_shrink in [2, 3, 5]:
                    try:
                        r, i = method_match_winrate(
                            ratings_data,
                            hl_weeks=hl,
                            reg_scale=reg_scale,
                            ind_scale=ind_scale,
                            ind_shrink=ind_shrink,
                        )
                        res = run_backtest('match', r, i, ratings_data)
                        b   = aggregate_brier(res)
                        row = {
                            'hl': hl, 'reg_scale': reg_scale,
                            'ind_scale': ind_scale, 'ind_shrink': ind_shrink,
                            'brier': b,
                        }
                        all_rows.append(row)
                        if b < best_brier:
                            best_brier  = b
                            best_params = row.copy()
                    except Exception:
                        pass

    return best_params, sorted(all_rows, key=lambda x: x['brier'])[:10]


def grid_search_dominance(ratings_data):
    """Grid search for dominance ratio."""
    best_brier = float('inf')
    best_params = None
    all_rows    = []

    for hl in [3, 6, 10, 16, 26, 52]:
        for reg_scale in [0.04, 0.08, 0.12, 0.18, 0.25, 0.35]:
            try:
                r, i = method_dominance_ratio(ratings_data, hl_weeks=hl, reg_scale=reg_scale)
                res  = run_backtest('dom', r, i, ratings_data)
                b    = aggregate_brier(res)
                row  = {'hl': hl, 'reg_scale': reg_scale, 'brier': b}
                all_rows.append(row)
                if b < best_brier:
                    best_brier  = b
                    best_params = row.copy()
            except Exception:
                pass

    return best_params, sorted(all_rows, key=lambda x: x['brier'])[:10]


def grid_search_median(ratings_data):
    """Grid search for median-based residuals."""
    best_brier = float('inf')
    best_params = None
    all_rows    = []

    for hl in [6, 10, 16, 26]:
        for reg_scale in [0.06, 0.12, 0.18, 0.25]:
            for ind_scale in [0.0, 0.08, 0.12]:
                try:
                    r, i = method_median_residuals(
                        ratings_data, hl_weeks=hl,
                        reg_scale=reg_scale, ind_scale=ind_scale,
                    )
                    res = run_backtest('median', r, i, ratings_data)
                    b   = aggregate_brier(res)
                    row = {'hl': hl, 'reg_scale': reg_scale, 'ind_scale': ind_scale, 'brier': b}
                    all_rows.append(row)
                    if b < best_brier:
                        best_brier  = b
                        best_params = row.copy()
                except Exception:
                    pass

    return best_params, sorted(all_rows, key=lambda x: x['brier'])[:10]


def grid_search_sample_scaled(ratings_data):
    """Grid search for sample-size-scaled method."""
    best_brier = float('inf')
    best_params = None
    all_rows    = []

    for hl in [6, 10, 16, 26]:
        for base_reg_scale in [0.01, 0.02, 0.04, 0.06]:
            for ind_scale in [0.0, 0.06, 0.10]:
                try:
                    r, i = method_sample_scaled(
                        ratings_data, hl_weeks=hl,
                        base_reg_scale=base_reg_scale, ind_scale=ind_scale,
                    )
                    res = run_backtest('sscaled', r, i, ratings_data)
                    b   = aggregate_brier(res)
                    row = {'hl': hl, 'base_reg_scale': base_reg_scale, 'ind_scale': ind_scale, 'brier': b}
                    all_rows.append(row)
                    if b < best_brier:
                        best_brier  = b
                        best_params = row.copy()
                except Exception:
                    pass

    return best_params, sorted(all_rows, key=lambda x: x['brier'])[:10]


# ═══════════════════════════════════════════════════════════════════════════════
# REPORTING
# ═══════════════════════════════════════════════════════════════════════════════

def print_results(method_name, results, dom_brier=None):
    """Print per-event and aggregate results for a method."""
    print(f'\n  Method: {method_name}')
    total_b, total_b_dom, total_n = 0.0, 0.0, 0
    for eid, r in results.items():
        delta = r['brier_dom'] - r['brier_cal']
        print(f'    {eid}: n={r["n"]:3d}  dom={r["brier_dom"]:.5f}  cal={r["brier_cal"]:.5f}  '
              f'Δ={delta:+.5f} {"BETTER" if delta > 0 else "worse"}')
        total_b     += r['brier_cal'] * r['n']
        total_b_dom += r['brier_dom'] * r['n']
        total_n     += r['n']
    if total_n:
        agg_cal = total_b / total_n
        agg_dom = total_b_dom / total_n
        print(f'    AGGREGATE: dom={agg_dom:.5f}  cal={agg_cal:.5f}  '
              f'Δ={agg_dom - agg_cal:+.5f}')
        return agg_cal
    return float('nan')


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print('=' * 70)
    print('VCT International Calibration — Alternative Signal Research')
    print('=' * 70)

    print('\nLoading domestic ratings...')
    ratings_data = load_ratings()

    # ── Compute domestic baseline ─────────────────────────────────────────────
    print('\n── BASELINE: Domestic-only ──────────────────────────────────────────')
    r0, i0 = method_baseline(ratings_data)
    res0   = run_backtest('baseline', r0, i0, ratings_data)
    dom_brier = aggregate_brier_dom(res0)
    print_results('Domestic-only (baseline)', res0)
    print(f'\n  BASELINE Brier (domestic-only): {dom_brier:.5f}')

    all_scores = [('BASELINE (domestic-only)', dom_brier)]

    # ── Method 1: Win-rate residuals (map level) ──────────────────────────────
    print('\n\n── METHOD 1: Win-rate residuals (map-level, default params) ─────────')
    r1, i1 = method_winrate_residuals(ratings_data, hl_weeks=10, reg_scale=0.12, ind_scale=0.10)
    res1   = run_backtest('wr_default', r1, i1, ratings_data)
    b1     = print_results('WinRate-MapLevel [hl=10, reg=0.12, ind=0.10]', res1)
    all_scores.append(('WinRate-MapLevel default', b1))

    # ── Method 2: Regional only ───────────────────────────────────────────────
    print('\n── METHOD 2: Regional-only (no individual bonuses) ──────────────────')
    r2, i2 = method_regional_only(ratings_data, hl_weeks=10, reg_scale=0.12)
    res2   = run_backtest('reg_only', r2, i2, ratings_data)
    b2     = print_results('Regional-only [hl=10, reg=0.12]', res2)
    all_scores.append(('Regional-only', b2))

    # ── Method 3: Individual only ─────────────────────────────────────────────
    print('\n── METHOD 3: Individual-only (no regional offsets) ──────────────────')
    r3, i3 = method_individual_only(ratings_data, hl_weeks=10, ind_scale=0.10)
    res3   = run_backtest('ind_only', r3, i3, ratings_data)
    b3     = print_results('Individual-only [hl=10, ind=0.10]', res3)
    all_scores.append(('Individual-only', b3))

    # ── Method 4: Match-level win-rate ────────────────────────────────────────
    print('\n── METHOD 4: Match-level (series) win-rate residuals ────────────────')
    r4, i4 = method_match_winrate(ratings_data, hl_weeks=10, reg_scale=0.12, ind_scale=0.10)
    res4   = run_backtest('match_wr', r4, i4, ratings_data)
    b4     = print_results('Match-level WinRate [hl=10, reg=0.12, ind=0.10]', res4)
    all_scores.append(('Match-level WinRate', b4))

    # ── Method 4b: Match regional only ───────────────────────────────────────
    print('\n── METHOD 4b: Match-level regional-only ─────────────────────────────')
    r4b, i4b = method_match_regional_only(ratings_data, hl_weeks=10, reg_scale=0.12)
    res4b    = run_backtest('match_reg', r4b, i4b, ratings_data)
    b4b      = print_results('Match-level Regional-only [hl=10, reg=0.12]', res4b)
    all_scores.append(('Match-level Regional-only', b4b))

    # ── Method 5: Dominance ratio ─────────────────────────────────────────────
    print('\n── METHOD 5: Regional dominance ratio ───────────────────────────────')
    r5, i5 = method_dominance_ratio(ratings_data, hl_weeks=10, reg_scale=0.12)
    res5   = run_backtest('dom_ratio', r5, i5, ratings_data)
    b5     = print_results('Dominance-ratio [hl=10, reg=0.12]', res5)
    all_scores.append(('Dominance-ratio', b5))

    # ── Method 6: Median residuals ────────────────────────────────────────────
    print('\n── METHOD 6: Median-based residuals (robust) ────────────────────────')
    r6, i6 = method_median_residuals(ratings_data, hl_weeks=10, reg_scale=0.12, ind_scale=0.10)
    res6   = run_backtest('median', r6, i6, ratings_data)
    b6     = print_results('Median-residuals [hl=10, reg=0.12, ind=0.10]', res6)
    all_scores.append(('Median-residuals', b6))

    # ── Method 7: Sample-size scaled ─────────────────────────────────────────
    print('\n── METHOD 7: Sample-size scaled regional ────────────────────────────')
    r7, i7 = method_sample_scaled(ratings_data, hl_weeks=10, base_reg_scale=0.04, ind_scale=0.10)
    res7   = run_backtest('sscale', r7, i7, ratings_data)
    b7     = print_results('SampleScaled [hl=10, base_reg=0.04, ind=0.10]', res7)
    all_scores.append(('SampleScaled', b7))

    # ── Method 8: Long half-life ──────────────────────────────────────────────
    print('\n── METHOD 8: Long half-life (26 weeks) ──────────────────────────────')
    r8, i8 = method_long_hl(ratings_data, hl_weeks=26, reg_scale=0.12, ind_scale=0.10)
    res8   = run_backtest('long_hl', r8, i8, ratings_data)
    b8     = print_results('LongHL [hl=26, reg=0.12, ind=0.10]', res8)
    all_scores.append(('LongHL-26wk', b8))

    # ── Method 9: Short half-life ─────────────────────────────────────────────
    print('\n── METHOD 9: Short half-life (3 weeks) ──────────────────────────────')
    r9, i9 = method_short_hl(ratings_data, hl_weeks=3, reg_scale=0.15, ind_scale=0.12)
    res9   = run_backtest('short_hl', r9, i9, ratings_data)
    b9     = print_results('ShortHL [hl=3, reg=0.15, ind=0.12]', res9)
    all_scores.append(('ShortHL-3wk', b9))

    # ── Method 10: HL=52 (very long) ─────────────────────────────────────────
    print('\n── METHOD 10: Very long half-life (52 weeks) ────────────────────────')
    r10, i10 = method_long_hl(ratings_data, hl_weeks=52, reg_scale=0.12, ind_scale=0.10)
    res10    = run_backtest('hl52', r10, i10, ratings_data)
    b10      = print_results('VeryLongHL [hl=52, reg=0.12, ind=0.10]', res10)
    all_scores.append(('VeryLongHL-52wk', b10))

    # ═══════════════════════════════════════════════════════════════════════
    # GRID SEARCHES
    # ═══════════════════════════════════════════════════════════════════════

    print('\n\n' + '=' * 70)
    print('GRID SEARCH — Win-rate residuals (map level)')
    print('=' * 70)
    best_wr, top_wr = grid_search_winrate(ratings_data)
    print(f'\n  Best params: {best_wr}')
    print(f'\n  Top 15 configs (sorted by brier_cal):')
    for row in top_wr:
        beat = '*BEATS DOM*' if row['brier'] < dom_brier else ''
        print(f'    hl={row["hl"]:2d}  reg={row["reg_scale"]:.2f}  '
              f'ind={row["ind_scale"]:.2f}  shrink={row["ind_shrink"]}  '
              f'brier={row["brier"]:.5f} {beat}')
    all_scores.append((f'GridBest-WinRate {best_wr}', best_wr['brier']))

    print('\n\n' + '=' * 70)
    print('GRID SEARCH — Match-level win-rate')
    print('=' * 70)
    best_match, top_match = grid_search_match(ratings_data)
    print(f'\n  Best params: {best_match}')
    print(f'\n  Top 10 configs:')
    for row in top_match:
        beat = '*BEATS DOM*' if row['brier'] < dom_brier else ''
        print(f'    hl={row["hl"]:2d}  reg={row["reg_scale"]:.2f}  '
              f'ind={row["ind_scale"]:.2f}  shrink={row["ind_shrink"]}  '
              f'brier={row["brier"]:.5f} {beat}')
    all_scores.append((f'GridBest-Match {best_match}', best_match['brier']))

    print('\n\n' + '=' * 70)
    print('GRID SEARCH — Dominance ratio')
    print('=' * 70)
    best_dom, top_dom = grid_search_dominance(ratings_data)
    print(f'\n  Best params: {best_dom}')
    print(f'\n  Top 10 configs:')
    for row in top_dom:
        beat = '*BEATS DOM*' if row['brier'] < dom_brier else ''
        print(f'    hl={row["hl"]:2d}  reg={row["reg_scale"]:.2f}  '
              f'brier={row["brier"]:.5f} {beat}')
    all_scores.append((f'GridBest-DomRatio {best_dom}', best_dom['brier']))

    print('\n\n' + '=' * 70)
    print('GRID SEARCH — Median residuals')
    print('=' * 70)
    best_med, top_med = grid_search_median(ratings_data)
    print(f'\n  Best params: {best_med}')
    print(f'\n  Top 10 configs:')
    for row in top_med:
        beat = '*BEATS DOM*' if row['brier'] < dom_brier else ''
        print(f'    hl={row["hl"]:2d}  reg={row["reg_scale"]:.2f}  '
              f'ind={row["ind_scale"]:.2f}  '
              f'brier={row["brier"]:.5f} {beat}')
    all_scores.append((f'GridBest-Median {best_med}', best_med['brier']))

    print('\n\n' + '=' * 70)
    print('GRID SEARCH — Sample-size scaled')
    print('=' * 70)
    best_ss, top_ss = grid_search_sample_scaled(ratings_data)
    print(f'\n  Best params: {best_ss}')
    print(f'\n  Top 10 configs:')
    for row in top_ss:
        beat = '*BEATS DOM*' if row['brier'] < dom_brier else ''
        print(f'    hl={row["hl"]:2d}  base_reg={row["base_reg_scale"]:.3f}  '
              f'ind={row["ind_scale"]:.2f}  '
              f'brier={row["brier"]:.5f} {beat}')
    all_scores.append((f'GridBest-SampleScaled {best_ss}', best_ss['brier']))

    # ═══════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════════════════

    print('\n\n' + '=' * 70)
    print('FINAL SUMMARY — All methods vs domestic-only baseline')
    print('=' * 70)
    print(f'\n  {"Method":<50s} {"Brier":>8s}  {"vs Baseline":>12s}')
    print(f'  {"-"*50}  {"-"*8}  {"-"*12}')
    for name, b in sorted(all_scores, key=lambda x: x[1]):
        delta = dom_brier - b
        marker = ' <-- BEST' if b == min(x[1] for x in all_scores) else ''
        better = 'BETTER' if delta > 0 else ('SAME' if delta == 0 else 'worse')
        print(f'  {name:<50s} {b:>8.5f}  {delta:>+8.5f} {better}{marker}')

    best_method = min(all_scores, key=lambda x: x[1])
    print(f'\n  WINNER: {best_method[0]}')
    print(f'  Best Brier:     {best_method[1]:.5f}')
    print(f'  Baseline Brier: {dom_brier:.5f}')
    print(f'  Improvement:    {dom_brier - best_method[1]:+.5f} ({(dom_brier - best_method[1])/dom_brier*100:+.3f}%)')

    # Per-event breakdown for best calibrated method
    print(f'\n  Per-event detail for best method:')

    # Re-run best grid search winner
    best_all = []
    for name, b in all_scores:
        if b == best_method[1]:
            best_all.append(name)
    print(f'  (Best = {best_all[0]})')

    print('\n  Note: if best_brier >= dom_brier, no calibration helps.')
    print(f'  Domestic-only {dom_brier:.5f} is the benchmark.')


if __name__ == '__main__':
    main()
