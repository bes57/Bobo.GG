"""
intl_calib_research3.py

Final validation: rating compression / beta reduction as the calibration signal.
Key question: can we learn the compression factor FROM TRAINING DATA (no leakage)?
Also: is the effect consistent event-to-event, or is 2025_champions an outlier?

Usage: python3 intl_calib_research3.py
"""

import os, sys, json, math
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime
from scipy.special import expit
from scipy.optimize import minimize_scalar, minimize

ROOT = '/Users/benny_es1/PythonTest'
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, 'data')

from EventLeaderboards import ORG_REGIONS

ACTIVE_REGIONS = {'EMEA', 'Americas', 'Pacific', 'CN'}

INTL_CONFIG = {
    '2023_masters_tokyo':   {'year': '2023', 'snap': 'after_tokyo',      'date': '2023-06-11'},
    '2023_champions':       {'year': '2023', 'snap': 'before_champions', 'date': '2023-08-06'},
    '2024_masters_madrid':  {'year': '2024', 'snap': 'before_madrid',    'date': '2024-02-14'},
    '2024_masters_shanghai':{'year': '2024', 'snap': 'before_shanghai',  'date': '2024-06-02'},
    '2024_champions':       {'year': '2024', 'snap': 'before_champions', 'date': '2024-08-01'},
    '2025_masters_bangkok': {'year': '2025', 'snap': 'before_bangkok',   'date': '2025-02-19'},
    '2025_masters_toronto': {'year': '2025', 'snap': 'before_toronto',   'date': '2025-06-19'},
    '2025_champions':       {'year': '2025', 'snap': 'before_champions', 'date': '2025-08-22'},
}

BACKTEST_HOLDOUTS = ['2024_champions', '2025_champions']
ALL_INTL_EVENTS = list(INTL_CONFIG.keys())

def load_ratings():
    with open(os.path.join(DATA, 'map_ratings.json')) as f:
        return json.load(f)

def get_snap_ratings(rd, yr, sn):
    try:
        return {org: d['overall_rating']
                for org, d in rd['ratings'][yr]['snapshots'][sn]['teams'].items()}
    except:
        return {}

def get_snap_beta(rd, yr, sn):
    try:
        return rd['ratings'][yr]['snapshots'][sn]['beta']
    except:
        return 0.45

def load_intl_games(event_id, include_cn=False):
    maps_path = os.path.join(DATA, 'maps', f'{event_id}.csv')
    mr_path   = os.path.join(DATA, 'match_results.csv')
    if not os.path.exists(maps_path):
        return []
    mr    = pd.read_csv(mr_path)
    mr    = mr[mr['MapNum'] != 'all'].copy()
    mr_idx = mr.set_index(['MatchID','MapNum'])
    df    = pd.read_csv(maps_path)
    df['MapNum'] = df['MapNum'].astype(str)
    meta  = df.groupby(['MatchID','MapNum']).agg(orgs=('Org', lambda x: list(x.unique()))).reset_index()
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
        except:
            continue
        games.append({'winner': winner, 'loser': loser,
                      'wr': wr, 'lr': lr,
                      'winner_region': wr_reg, 'loser_region': lr_reg})
    return games

def win_prob(rA, rB, beta):
    return float(np.clip(expit(beta * (rA - rB)), 1e-9, 1-1e-9))

def brier(p): return (p - 1.0)**2
def logloss(p): return -math.log(p)

def score_games(games, pre, beta, shrink=0.0, beta_mult=1.0):
    """Score games with optional compression and beta scaling."""
    if shrink > 0 and pre:
        mean_r = float(np.mean(list(pre.values())))
        pre = {t: mean_r + (r - mean_r) * (1 - shrink) for t, r in pre.items()}
    adj_beta = beta * beta_mult
    bs = []
    for g in games:
        rw = pre.get(g['winner'])
        rl = pre.get(g['loser'])
        if rw is None or rl is None:
            continue
        p = win_prob(rw, rl, adj_beta)
        bs.append(brier(p))
    return bs

def main():
    print('='*70)
    print('Validation: Can Rating Compression Be Learned Without Leakage?')
    print('='*70)

    rd = load_ratings()

    # ── Part 1: Per-event optimal beta/shrink ────────────────────────────────
    print('\n\n── Part 1: Per-event oracle beta multiplier ─────────────────────────')
    print('What beta_mult minimizes brier at EACH event individually?')
    print('(Shows whether overconfidence is consistent across events)')
    print()
    print(f'  {"Event":<28s}  {"n":>3s}  {"beta0":>6s}  {"opt_bm":>7s}  '
          f'{"dom_brier":>9s}  {"opt_brier":>9s}  {"improve":>8s}')
    print(f'  {"-"*28}  {"---":>3s}  {"------":>6s}  {"-------":>7s}  '
          f'{"--------":>9s}  {"---------":>9s}  {"--------":>8s}')

    event_oracle_bm = {}
    for eid in ALL_INTL_EVENTS:
        cfg    = INTL_CONFIG.get(eid, {})
        yr, sn = cfg.get('year'), cfg.get('snap')
        pre    = get_snap_ratings(rd, yr, sn)
        beta0  = get_snap_beta(rd, yr, sn)
        games  = load_intl_games(eid)
        if not games:
            continue

        # Domestic brier
        bs_dom = score_games(games, pre, beta0)
        dom_b  = float(np.mean(bs_dom)) if bs_dom else float('nan')

        # Optimize beta_mult
        def neg_b(bm):
            bs = score_games(games, pre, beta0, beta_mult=bm)
            return float(np.mean(bs)) if bs else 0.5

        result = minimize_scalar(neg_b, bounds=(0.05, 3.0), method='bounded')
        opt_bm = result.x
        opt_b  = result.fun
        improve = dom_b - opt_b

        event_oracle_bm[eid] = opt_bm
        marker = ' <HOLDOUT>' if eid in BACKTEST_HOLDOUTS else ''
        print(f'  {eid:<28s}  {len(bs_dom):>3d}  {beta0:>6.3f}  {opt_bm:>7.3f}  '
              f'{dom_b:>9.5f}  {opt_b:>9.5f}  {improve:>+8.5f}{marker}')

    # ── Part 2: Cross-validation — train on all-but-one, test on left-out ────
    print('\n\n── Part 2: Cross-validation (leave-one-out) ─────────────────────────')
    print('Train beta_mult on all events except one, test on the left-out event')
    print('This is proper out-of-sample validation (like our backtest, but for all events)')

    all_events_with_games = [eid for eid in ALL_INTL_EVENTS
                              if load_intl_games(eid)]

    cv_results = []
    for test_eid in all_events_with_games:
        train_eids = [e for e in all_events_with_games if e != test_eid]

        # Build training set
        train_games_rated = []
        for eid in train_eids:
            cfg   = INTL_CONFIG.get(eid, {})
            yr,sn = cfg.get('year'), cfg.get('snap')
            pre   = get_snap_ratings(rd, yr, sn)
            beta0 = get_snap_beta(rd, yr, sn)
            games = load_intl_games(eid)
            for g in games:
                rw = pre.get(g['winner'])
                rl = pre.get(g['loser'])
                if rw is not None and rl is not None:
                    train_games_rated.append((rw, rl, beta0))

        def train_brier(bm):
            bs = []
            for rw, rl, beta0 in train_games_rated:
                p = win_prob(rw, rl, beta0 * bm)
                bs.append(brier(p))
            return float(np.mean(bs)) if bs else 0.5

        result = minimize_scalar(train_brier, bounds=(0.05, 3.0), method='bounded')
        trained_bm = result.x

        # Test on left-out event
        cfg   = INTL_CONFIG.get(test_eid, {})
        yr,sn = cfg.get('year'), cfg.get('snap')
        pre   = get_snap_ratings(rd, yr, sn)
        beta0 = get_snap_beta(rd, yr, sn)
        games = load_intl_games(test_eid)

        bs_dom = score_games(games, pre, beta0)
        bs_cal = score_games(games, pre, beta0, beta_mult=trained_bm)

        if bs_dom:
            dom_b = float(np.mean(bs_dom))
            cal_b = float(np.mean(bs_cal))
            marker = ' <HOLDOUT>' if test_eid in BACKTEST_HOLDOUTS else ''
            print(f'  {test_eid:<28s}: trained_bm={trained_bm:.3f}  '
                  f'dom={dom_b:.5f}  cal={cal_b:.5f}  Δ={dom_b-cal_b:+.5f}{marker}')
            cv_results.append((test_eid, dom_b, cal_b))

    print(f'\n  Cross-validation summary:')
    total_dom = sum(d * (1/len(cv_results)) for _, d, _ in cv_results)
    total_cal = sum(c * (1/len(cv_results)) for _, _, c in cv_results)
    n_better  = sum(1 for _, d, c in cv_results if c < d)
    print(f'  Avg dom brier (unweighted): {total_dom:.5f}')
    print(f'  Avg cal brier (unweighted): {total_cal:.5f}')
    print(f'  Events where calibration helps: {n_better}/{len(cv_results)}')
    print(f'  Avg improvement: {total_dom - total_cal:+.5f}')

    # ── Part 3: Focused backtest — proper train → test ────────────────────────
    print('\n\n── Part 3: Proper backtest (no leakage) ─────────────────────────────')
    print('For 2024_champions: train bm on 2023+2024_masters events')
    print('For 2025_champions: train bm on 2023+2024+2025_masters events')

    for holdout_eid, train_eids in [
        ('2024_champions', ['2023_masters_tokyo', '2023_champions',
                            '2024_masters_madrid', '2024_masters_shanghai']),
        ('2025_champions', ['2023_masters_tokyo', '2023_champions',
                            '2024_masters_madrid', '2024_masters_shanghai',
                            '2025_masters_bangkok', '2025_masters_toronto']),
    ]:
        train_games_rated = []
        for eid in train_eids:
            cfg   = INTL_CONFIG.get(eid, {})
            yr,sn = cfg.get('year'), cfg.get('snap')
            pre   = get_snap_ratings(rd, yr, sn)
            beta0 = get_snap_beta(rd, yr, sn)
            games = load_intl_games(eid)
            for g in games:
                rw = pre.get(g['winner'])
                rl = pre.get(g['loser'])
                if rw is not None and rl is not None:
                    train_games_rated.append((rw, rl, beta0))

        def train_brier2(bm):
            bs = []
            for rw, rl, b0 in train_games_rated:
                p = win_prob(rw, rl, b0 * bm)
                bs.append(brier(p))
            return float(np.mean(bs)) if bs else 0.5

        result = minimize_scalar(train_brier2, bounds=(0.05, 3.0), method='bounded')
        trained_bm = result.x

        # Also train a joint (beta_mult, shrink) to be thorough
        def train_brier_both(params):
            bm, sf = params
            if bm < 0.05 or bm > 3.0 or sf < 0.0 or sf > 0.99:
                return 1.0
            bs = []
            for rw, rl, b0 in train_games_rated:
                # compression doesn't matter if we only have rw-rl difference
                # shrink: rw_new = mean + (rw-mean)*(1-sf), same for rl
                # But we don't have all ratings here — let's just use beta_mult
                p = win_prob(rw, rl, b0 * bm)
                bs.append(brier(p))
            return float(np.mean(bs)) if bs else 0.5

        # Test on holdout
        cfg_h  = INTL_CONFIG.get(holdout_eid, {})
        yr_h, sn_h = cfg_h.get('year'), cfg_h.get('snap')
        pre_h  = get_snap_ratings(rd, yr_h, sn_h)
        beta0_h = get_snap_beta(rd, yr_h, sn_h)
        games_h = load_intl_games(holdout_eid)

        bs_dom_h = score_games(games_h, pre_h, beta0_h)
        bs_cal_h = score_games(games_h, pre_h, beta0_h, beta_mult=trained_bm)

        dom_b = float(np.mean(bs_dom_h))
        cal_b = float(np.mean(bs_cal_h))

        print(f'\n  {holdout_eid}:')
        print(f'    Training events: {train_eids}')
        print(f'    Trained beta_mult: {trained_bm:.4f}  (default=1.0)')
        print(f'    Holdout domestic brier: {dom_b:.5f}')
        print(f'    Holdout calibrated brier: {cal_b:.5f}')
        print(f'    Improvement: {dom_b - cal_b:+.5f} {"BETTER" if dom_b > cal_b else "WORSE"}')

        # Also test a range around the trained bm
        print(f'\n    Sensitivity around trained bm={trained_bm:.3f}:')
        for bm_test in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, trained_bm, 1.0]:
            bs_t = score_games(games_h, pre_h, beta0_h, beta_mult=bm_test)
            b_t  = float(np.mean(bs_t))
            mark = ' <-- trained' if abs(bm_test - trained_bm) < 0.01 else ''
            mark += ' <-- default' if abs(bm_test - 1.0) < 0.01 else ''
            print(f'      bm={bm_test:.2f}  brier={b_t:.5f}  Δ={dom_b - b_t:+.5f}{mark}')

    # ── Part 4: Combined backtest summary ─────────────────────────────────────
    print('\n\n── Part 4: Final combined backtest summary ──────────────────────────')

    # Best configs we've found
    configs = [
        ('Domestic-only (baseline)', 0.0, 1.0),
        ('Compress 30% only', 0.3, 1.0),
        ('Beta 0.7x only', 0.0, 0.7),
        ('Trained beta_mult (per event)', 'trained', 'trained'),
        ('Beta 0.2x extreme', 0.0, 0.2),
        ('Compress 80% + beta 0.8x', 0.8, 0.8),
    ]

    # Per-event trained bm from part 3
    per_event_trained_bm = {}
    for holdout_eid, train_eids in [
        ('2024_champions', ['2023_masters_tokyo', '2023_champions',
                            '2024_masters_madrid', '2024_masters_shanghai']),
        ('2025_champions', ['2023_masters_tokyo', '2023_champions',
                            '2024_masters_madrid', '2024_masters_shanghai',
                            '2025_masters_bangkok', '2025_masters_toronto']),
    ]:
        train_games_rated = []
        for eid in train_eids:
            cfg   = INTL_CONFIG.get(eid, {})
            yr,sn = cfg.get('year'), cfg.get('snap')
            pre   = get_snap_ratings(rd, yr, sn)
            beta0 = get_snap_beta(rd, yr, sn)
            games = load_intl_games(eid)
            for g in games:
                rw = pre.get(g['winner'])
                rl = pre.get(g['loser'])
                if rw is not None and rl is not None:
                    train_games_rated.append((rw, rl, beta0))

        def tb(bm):
            bs = []
            for rw, rl, b0 in train_games_rated:
                p = win_prob(rw, rl, b0 * bm)
                bs.append(brier(p))
            return float(np.mean(bs)) if bs else 0.5

        result = minimize_scalar(tb, bounds=(0.05, 3.0), method='bounded')
        per_event_trained_bm[holdout_eid] = result.x

    print()
    print(f'  {"Config":<45s}  {"2024":>9s}  {"2025":>9s}  {"Aggregate":>9s}')
    print(f'  {"-"*45}  {"-"*9}  {"-"*9}  {"-"*9}')

    for config_name, sf, bm in configs:
        results_by_event = {}
        for eid in BACKTEST_HOLDOUTS:
            cfg    = INTL_CONFIG.get(eid, {})
            yr, sn = cfg.get('year'), cfg.get('snap')
            pre    = get_snap_ratings(rd, yr, sn)
            beta0  = get_snap_beta(rd, yr, sn)
            games  = load_intl_games(eid)

            if sf == 'trained':
                actual_sf = 0.0
            else:
                actual_sf = sf

            if bm == 'trained':
                actual_bm = per_event_trained_bm.get(eid, 1.0)
            else:
                actual_bm = bm

            bs = score_games(games, pre, beta0, shrink=actual_sf, beta_mult=actual_bm)
            results_by_event[eid] = float(np.mean(bs)) if bs else float('nan')

        b24 = results_by_event.get('2024_champions', float('nan'))
        b25 = results_by_event.get('2025_champions', float('nan'))
        # weighted aggregate (36 + 39 games)
        agg  = (b24 * 36 + b25 * 39) / (36 + 39)
        beat = ' ***' if agg < 0.26432 else ''
        print(f'  {config_name:<45s}  {b24:>9.5f}  {b25:>9.5f}  {agg:>9.5f}{beat}')

    # ── Part 5: Is the overconfidence consistent event-to-event? ─────────────
    print('\n\n── Part 5: Is beta < 1 consistently better across ALL events? ───────')
    print('Compare domestic (bm=1.0) vs bm=0.5 at every event:')
    print()
    print(f'  {"Event":<28s}  {"n":>3s}  {"dom_brier":>9s}  {"bm0.5_brier":>10s}  {"Δ":>8s}')
    print(f'  {"-"*28}  {"-"*3}  {"-"*9}  {"-"*10}  {"-"*8}')

    for eid in ALL_INTL_EVENTS:
        cfg   = INTL_CONFIG.get(eid, {})
        yr,sn = cfg.get('year'), cfg.get('snap')
        pre   = get_snap_ratings(rd, yr, sn)
        beta0 = get_snap_beta(rd, yr, sn)
        games = load_intl_games(eid)
        if not games:
            continue

        bs_dom = score_games(games, pre, beta0)
        bs_05  = score_games(games, pre, beta0, beta_mult=0.5)

        if bs_dom:
            dom_b = float(np.mean(bs_dom))
            b05   = float(np.mean(bs_05))
            delta = dom_b - b05
            marker = ' BETTER' if delta > 0 else ' worse'
            marker2 = ' <HOLDOUT>' if eid in BACKTEST_HOLDOUTS else ''
            print(f'  {eid:<28s}  {len(bs_dom):>3d}  {dom_b:>9.5f}  {b05:>10.5f}  {delta:>+8.5f}{marker}{marker2}')

    print('\n\nDone.')


if __name__ == '__main__':
    main()
