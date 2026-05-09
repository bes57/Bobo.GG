"""
intl_calib_research2.py

Follow-up research based on diagnostics:
1. Beta recalibration for cross-regional games (shared beta, or different at Champions)
2. Rating compression: scale all ratings toward mean before applying at Champions
3. Upset-weighting: flatter predictions for big rating gaps
4. Hierarchical beta: use different beta for cross-regional vs same-regional
5. Cross-year signal: test if 2023 Masters/Champions predicts 2024 signal direction
6. Uncertainty-based flattening: high-variance teams → flatter predictions
7. Test "naive partial regression to mean" approaches

Usage: python3 intl_calib_research2.py
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

# Generic backtest with per-event (beta, rating_transform)
def backtest_generic(ratings_data, transform_fn, beta_fn=None):
    """
    transform_fn(ratings_dict, event_id) -> new_ratings_dict
    beta_fn(yr, sn) -> float or None (use default)
    """
    results = {}
    for eid in BACKTEST_HOLDOUTS:
        cfg    = INTL_CONFIG.get(eid, {})
        yr, sn = cfg.get('year'), cfg.get('snap')
        pre    = get_snap_ratings(ratings_data, yr, sn)
        beta0  = get_snap_beta(ratings_data, yr, sn)
        beta   = beta_fn(yr, sn) if beta_fn else beta0
        games  = load_intl_games(eid)

        transformed = transform_fn(pre, eid)

        bs_dom, bs_cal = [], []
        for g in games:
            rw_d = pre.get(g['winner'])
            rl_d = pre.get(g['loser'])
            if rw_d is None or rl_d is None:
                continue
            rw_t = transformed.get(g['winner'], rw_d)
            rl_t = transformed.get(g['loser'],  rl_d)
            p_d  = win_prob(rw_d, rl_d, beta0)
            p_t  = win_prob(rw_t, rl_t, beta)
            bs_dom.append(brier(p_d))
            bs_cal.append(brier(p_t))

        if bs_dom:
            results[eid] = {
                'n':         len(bs_dom),
                'brier_dom': float(np.mean(bs_dom)),
                'brier_cal': float(np.mean(bs_cal)),
            }
    return results

def print_results(name, results):
    total_cal, total_dom, total_n = 0.0, 0.0, 0
    print(f'\n  {name}:')
    for eid, r in results.items():
        d = r['brier_dom'] - r['brier_cal']
        print(f'    {eid}: dom={r["brier_dom"]:.5f}  cal={r["brier_cal"]:.5f}  Δ={d:+.5f}')
        total_cal += r['brier_cal'] * r['n']
        total_dom += r['brier_dom'] * r['n']
        total_n   += r['n']
    if total_n:
        agg_cal = total_cal / total_n
        agg_dom = total_dom / total_n
        d       = agg_dom - agg_cal
        print(f'    AGGREGATE: dom={agg_dom:.5f}  cal={agg_cal:.5f}  Δ={d:+.5f} {"BETTER" if d>0 else "WORSE"}')
        return agg_cal
    return float('nan')

def main():
    print('='*70)
    print('Follow-up Research: Beta Recalibration + Rating Compression')
    print('='*70)

    rd = load_ratings()

    # Get domestic baseline
    def no_transform(pre, eid):
        return pre
    res_base = backtest_generic(rd, no_transform)
    dom_brier = sum(r['brier_dom']*r['n'] for r in res_base.values()) / sum(r['n'] for r in res_base.values())
    print(f'\n  Domestic baseline Brier: {dom_brier:.5f}')

    all_scores = [('Baseline (domestic-only)', dom_brier)]

    # ── A: Rating compression (shrink toward mean) ────────────────────────────
    print('\n\n── A: Rating compression (shrink all ratings toward team mean) ───────')
    print('Hypothesis: domestic ratings are "too confident" about cross-regional gaps')

    for shrink_factor in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        def make_compress(sf):
            def transform(pre, eid):
                if not pre:
                    return pre
                mean_r = float(np.mean(list(pre.values())))
                return {t: mean_r + (r - mean_r) * (1 - sf) for t, r in pre.items()}
            return transform

        results = backtest_generic(rd, make_compress(shrink_factor))
        cal = sum(r['brier_cal']*r['n'] for r in results.values()) / sum(r['n'] for r in results.values())
        beat = ' <-- BEATS DOM' if cal < dom_brier else ''
        print(f'    shrink={shrink_factor:.1f}  brier={cal:.5f}{beat}')

    # ── B: Cross-regional beta recalibration ──────────────────────────────────
    print('\n\n── B: Beta recalibration for cross-regional games ──────────────────')
    print('Use a different (lower) beta for cross-regional predictions only')

    best_beta_b = float('inf')
    best_beta_val = None
    for beta_mult in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
        # Use default ratings, but multiply beta by factor
        def make_beta_fn(mult):
            def beta_fn(yr, sn):
                return get_snap_beta(rd, yr, sn) * mult
            return beta_fn

        results = backtest_generic(rd, no_transform, make_beta_fn(beta_mult))
        cal = sum(r['brier_cal']*r['n'] for r in results.values()) / sum(r['n'] for r in results.values())
        beat = ' <-- BEATS DOM' if cal < dom_brier else ''
        print(f'    beta_mult={beta_mult:.1f}  brier={cal:.5f}{beat}')
        if cal < best_beta_b:
            best_beta_b = cal
            best_beta_val = beta_mult

    all_scores.append((f'BetaMult={best_beta_val}', best_beta_b))

    # ── C: Combined compression + beta reduction ──────────────────────────────
    print('\n\n── C: Combined compression + beta reduction ─────────────────────────')
    best_combined = float('inf')
    best_sf, best_bm = 0.0, 1.0

    for sf in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        for bm in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            def make_both(sf_, bm_):
                def transform(pre, eid):
                    if not pre:
                        return pre
                    mean_r = float(np.mean(list(pre.values())))
                    return {t: mean_r + (r - mean_r) * (1 - sf_) for t, r in pre.items()}
                def beta_fn(yr, sn):
                    return get_snap_beta(rd, yr, sn) * bm_
                return transform, beta_fn

            tf, bf = make_both(sf, bm)
            results = backtest_generic(rd, tf, bf)
            cal = sum(r['brier_cal']*r['n'] for r in results.values()) / sum(r['n'] for r in results.values())
            beat = ' <-- BEATS DOM' if cal < dom_brier else ''
            if beat or abs(sf - best_sf) < 0.01:
                print(f'    shrink={sf:.1f}  beta_mult={bm:.1f}  brier={cal:.5f}{beat}')
            if cal < best_combined:
                best_combined = cal
                best_sf, best_bm = sf, bm

    print(f'\n  Best combined: shrink={best_sf}  beta_mult={best_bm}  brier={best_combined:.5f}')
    all_scores.append((f'Combined shrink={best_sf} bm={best_bm}', best_combined))

    # ── D: Per-event analysis of beta training ────────────────────────────────
    print('\n\n── D: Train beta on prior Champions events ──────────────────────────')
    print('Use 2023_champions games to choose beta for 2024_champions etc.')

    # Training events for beta calibration
    training_events = ['2023_masters_tokyo', '2023_champions', '2024_masters_madrid', '2024_masters_shanghai']

    def compute_optimal_beta_from_training(training_eids):
        """Find beta that minimizes brier on training events' cross-regional games."""
        all_games_rated = []
        for eid in training_eids:
            cfg   = INTL_CONFIG.get(eid, {})
            yr,sn = cfg.get('year'), cfg.get('snap')
            pre   = get_snap_ratings(rd, yr, sn)
            games = load_intl_games(eid)
            for g in games:
                rw = pre.get(g['winner'])
                rl = pre.get(g['loser'])
                if rw is not None and rl is not None:
                    all_games_rated.append((rw, rl, 1.0))

        if not all_games_rated:
            return 0.45

        def neg_brier(beta_val):
            bs = []
            for rw, rl, _ in all_games_rated:
                p = win_prob(rw, rl, beta_val)
                bs.append(brier(p))
            return float(np.mean(bs))

        result = minimize_scalar(neg_brier, bounds=(0.05, 2.0), method='bounded')
        return result.x

    # For 2024_champions: train on all prior events except holdouts
    beta_2024 = compute_optimal_beta_from_training(
        ['2023_masters_tokyo', '2023_champions', '2024_masters_madrid', '2024_masters_shanghai']
    )
    # For 2025_champions: train on all prior events except holdouts
    beta_2025 = compute_optimal_beta_from_training(
        ['2023_masters_tokyo', '2023_champions', '2024_masters_madrid', '2024_masters_shanghai',
         '2025_masters_bangkok', '2025_masters_toronto']
    )

    print(f'  Trained beta for 2024_champions: {beta_2024:.3f}')
    print(f'  Trained beta for 2025_champions: {beta_2025:.3f}')

    per_event_beta = {'2024_champions': beta_2024, '2025_champions': beta_2025}

    def trained_beta_fn(yr, sn):
        eid_map = {'2024_before_champions': '2024_champions', '2025_before_champions': '2025_champions'}
        eid = eid_map.get(f'{yr}_{sn}')
        return per_event_beta.get(eid, get_snap_beta(rd, yr, sn))

    results_d = backtest_generic(rd, no_transform, trained_beta_fn)
    b_d = print_results('Trained cross-regional beta (from prior Masters+Champions)', results_d)
    all_scores.append(('Trained beta', b_d))

    # ── E: Separate beta for each event type ──────────────────────────────────
    print('\n\n── E: Grid search beta multiplier vs shrink (all combinations) ───────')
    print('Full 2D grid — show all configs that beat domestic baseline')

    beats_dom = []
    for sf in np.arange(0.0, 0.9, 0.05):
        for bm in np.arange(0.2, 1.5, 0.05):
            def make_both2(sf_, bm_):
                def tf(pre, eid):
                    if not pre: return pre
                    mean_r = float(np.mean(list(pre.values())))
                    return {t: mean_r + (r - mean_r) * (1 - sf_) for t, r in pre.items()}
                def bf(yr, sn):
                    return get_snap_beta(rd, yr, sn) * bm_
                return tf, bf
            tf, bf = make_both2(sf, bm)
            results = backtest_generic(rd, tf, bf)
            cal = sum(r['brier_cal']*r['n'] for r in results.values()) / sum(r['n'] for r in results.values())
            if cal < dom_brier:
                beats_dom.append((sf, bm, cal))

    if beats_dom:
        beats_dom.sort(key=lambda x: x[2])
        print(f'\n  Configs that BEAT domestic baseline ({dom_brier:.5f}):')
        for sf, bm, cal in beats_dom[:20]:
            print(f'    shrink={sf:.2f}  beta_mult={bm:.2f}  brier={cal:.5f}  Δ={dom_brier-cal:+.5f}')
        best_sf2, best_bm2, best_cal2 = beats_dom[0]
        all_scores.append((f'Grid2D best: shrink={best_sf2:.2f} bm={best_bm2:.2f}', best_cal2))
    else:
        print('  NO configs beat domestic baseline!')
        print('  Best config from grid:')
        all_results_grid = []
        for sf in np.arange(0.0, 0.9, 0.1):
            for bm in np.arange(0.2, 1.5, 0.1):
                def make_both3(sf_, bm_):
                    def tf(pre, eid):
                        if not pre: return pre
                        mean_r = float(np.mean(list(pre.values())))
                        return {t: mean_r + (r - mean_r) * (1 - sf_) for t, r in pre.items()}
                    def bf(yr, sn):
                        return get_snap_beta(rd, yr, sn) * bm_
                    return tf, bf
                tf, bf = make_both3(sf, bm)
                results = backtest_generic(rd, tf, bf)
                cal = sum(r['brier_cal']*r['n'] for r in results.values()) / sum(r['n'] for r in results.values())
                all_results_grid.append((sf, bm, cal))
        all_results_grid.sort(key=lambda x: x[2])
        for sf, bm, cal in all_results_grid[:5]:
            print(f'    shrink={sf:.1f}  beta_mult={bm:.1f}  brier={cal:.5f}')

    # ── F: Per-event residual detail ──────────────────────────────────────────
    print('\n\n── F: Per-game prediction errors at holdout events ─────────────────')
    print('Which specific matchups hurt/help most?')

    for eid in BACKTEST_HOLDOUTS:
        cfg    = INTL_CONFIG.get(eid, {})
        yr, sn = cfg.get('year'), cfg.get('snap')
        pre    = get_snap_ratings(rd, yr, sn)
        beta   = get_snap_beta(rd, yr, sn)
        games  = load_intl_games(eid)

        print(f'\n  {eid}:')
        rows = []
        for g in games:
            rw = pre.get(g['winner'])
            rl = pre.get(g['loser'])
            if rw is None or rl is None:
                continue
            p = win_prob(rw, rl, beta)
            b = brier(p)
            rows.append({
                'winner': g['winner'], 'loser': g['loser'],
                'wr_reg': g['winner_region'], 'lr_reg': g['loser_region'],
                'rw': rw, 'rl': rl, 'diff': rw - rl, 'p': p, 'brier': b,
            })

        rows.sort(key=lambda x: -x['brier'])
        print(f'  {"Winner":6s} {"Loser":6s}  {"WReg":8s} {"LReg":8s}  '
              f'{"Diff":6s}  {"p_win":6s}  {"Brier":7s}  Note')
        for r in rows[:10]:
            upset = ' UPSET' if r['p'] < 0.5 else ''
            print(f'  {r["winner"]:6s} {r["loser"]:6s}  {r["wr_reg"]:8s} {r["lr_reg"]:8s}  '
                  f'{r["diff"]:+6.2f}  {r["p"]:.3f}  {r["brier"]:.5f}{upset}')

    # ── G: What's the irreducible Brier? ─────────────────────────────────────
    print('\n\n── G: Information theory bound ─────────────────────────────────────')
    print('If we predict p=0.5 for everything (ignorance), what is Brier?')

    for eid in BACKTEST_HOLDOUTS:
        games  = load_intl_games(eid)
        n      = len(games)
        brier_ignorance = sum(brier(0.5) for _ in games) / max(n, 1)
        cfg    = INTL_CONFIG.get(eid, {})
        yr, sn = cfg.get('year'), cfg.get('snap')
        pre    = get_snap_ratings(rd, yr, sn)
        beta   = get_snap_beta(rd, yr, sn)
        bs_dom = []
        for g in games:
            rw = pre.get(g['winner'])
            rl = pre.get(g['loser'])
            if rw is None or rl is None:
                continue
            bs_dom.append(brier(win_prob(rw, rl, beta)))

        print(f'  {eid}: n={n}')
        print(f'    Ignorance (p=0.5): {brier_ignorance:.5f}')
        print(f'    Domestic:          {np.mean(bs_dom):.5f}')
        print(f'    Domestic advantage over ignorance: {brier_ignorance - np.mean(bs_dom):+.5f}')

    # ── H: Simulate what happens if we use past Champions as calibration ───────
    print('\n\n── H: Use 2023_champions to calibrate 2024_champions signal ─────────')
    print('Train regional offsets on 2023 champions, apply to 2024 and 2025')

    # Compute regional signal from 2023_champions
    champ_2023_eid = '2023_champions'
    cfg23   = INTL_CONFIG.get(champ_2023_eid, {})
    yr23,sn23 = cfg23.get('year'), cfg23.get('snap')
    pre23   = get_snap_ratings(rd, yr23, sn23)
    beta23  = get_snap_beta(rd, yr23, sn23)
    games23 = load_intl_games(champ_2023_eid)

    reg_actual_wins = defaultdict(int)
    reg_expected_wins = defaultdict(float)
    reg_n = defaultdict(int)
    for g in games23:
        rw = pre23.get(g['winner'])
        rl = pre23.get(g['loser'])
        if rw is None or rl is None:
            continue
        p = win_prob(rw, rl, beta23)
        reg_actual_wins[g['winner_region']]   += 1
        reg_expected_wins[g['winner_region']] += p
        reg_expected_wins[g['loser_region']]  += (1-p)
        reg_n[g['winner_region']]  += 1
        reg_n[g['loser_region']]   += 1

    # signal = (actual - expected) / n_games normalized
    raw_signal_23 = {}
    for reg in ['EMEA', 'Americas', 'Pacific']:
        n = reg_n.get(reg, 0)
        if n > 0:
            raw_signal_23[reg] = (reg_actual_wins.get(reg, 0) - reg_expected_wins.get(reg, 0)) / n
    gm = float(np.mean(list(raw_signal_23.values())))
    signal_23 = {r: v - gm for r, v in raw_signal_23.items()}
    print(f'  2023_champions regional signal: {signal_23}')

    # Apply scaled versions of this signal as offset at 2024_champions and 2025_champions
    print(f'\n  Applying 2023 signal to 2024_champions and 2025_champions:')
    for scale in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
        def make_signal_transform(sc, sig):
            def tf(pre, eid):
                if eid not in BACKTEST_HOLDOUTS:
                    return pre
                return {t: r + sig.get(ORG_REGIONS.get(t, ''), 0) * sc
                        for t, r in pre.items()}
            return tf
        results_h = backtest_generic(rd, make_signal_transform(scale, signal_23))
        cal = sum(r['brier_cal']*r['n'] for r in results_h.values()) / sum(r['n'] for r in results_h.values())
        beat = ' <-- BEATS DOM' if cal < dom_brier else ''
        print(f'    scale={scale:.1f}  brier={cal:.5f}{beat}')

    # ── Final Summary ──────────────────────────────────────────────────────────
    print('\n\n' + '='*70)
    print('FINAL SUMMARY')
    print('='*70)
    print(f'\n  Domestic baseline: {dom_brier:.5f}')
    print(f'\n  {"Method":<50s} {"Brier":>8s}  {"vs Baseline":>12s}')
    print(f'  {"-"*50}  {"-"*8}  {"-"*12}')
    for name, b in sorted(all_scores, key=lambda x: x[1]):
        d = dom_brier - b
        marker = ' <-- BEST' if b == min(x[1] for x in all_scores) else ''
        better = 'BETTER' if d > 0 else ('SAME' if abs(d) < 1e-8 else 'WORSE')
        print(f'  {name:<50s} {b:>8.5f}  {d:>+8.5f} {better}{marker}')

    best_method = min(all_scores, key=lambda x: x[1])
    print(f'\n  WINNER: {best_method[0]}  ({best_method[1]:.5f})')
    print(f'  Domestic: {dom_brier:.5f}')
    print(f'  Improvement: {dom_brier - best_method[1]:+.5f}')

    if best_method[1] >= dom_brier - 1e-7:
        print('\n  CONCLUSION: No calibration method beats domestic-only.')
        print('  The domestic Massey ratings are already as good as any cross-regional calibration.')
        print('  Likely reasons:')
        print('   - Only 3-4 teams per region per event (very high variance)')
        print('   - Champions regularly has major upsets (format/pressure effects)')
        print('   - Signal from prior Masters contradicts signal at Champions (2025 Pacific)')
        print('   - The oracle per-team Brier shows improvement IS possible,')
        print('     but it requires knowing future performance in advance.')


if __name__ == '__main__':
    main()
