"""
intl_calib_diagnostic.py

Deeper diagnostic: why does calibration fail?
- Inspect what signals exist in the data
- Test if ANY static offset helps at all
- Explore what the oracle offset would be
- Check whether the domestic ratings are already well-calibrated for cross-regional games

Usage: python3 intl_calib_diagnostic.py
"""

import os, sys, json, math
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime
from scipy.special import expit
from scipy.optimize import minimize, minimize_scalar

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

def get_snap_ratings(ratings_data, year, snap):
    try:
        return {org: d['overall_rating']
                for org, d in ratings_data['ratings'][year]['snapshots'][snap]['teams'].items()}
    except:
        return {}

def get_snap_beta(ratings_data, year, snap):
    try:
        return ratings_data['ratings'][year]['snapshots'][snap]['beta']
    except:
        return 0.45

def load_intl_games(event_id, include_cn=False):
    maps_path = os.path.join(DATA, 'maps', f'{event_id}.csv')
    mr_path   = os.path.join(DATA, 'match_results.csv')
    if not os.path.exists(maps_path):
        return []
    mr = pd.read_csv(mr_path)
    mr = mr[mr['MapNum'] != 'all'].copy()
    mr_idx = mr.set_index(['MatchID', 'MapNum'])
    df = pd.read_csv(maps_path)
    df['MapNum'] = df['MapNum'].astype(str)
    meta = df.groupby(['MatchID', 'MapNum']).agg(orgs=('Org', lambda x: list(x.unique()))).reset_index()
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
        loser = losers[0]
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
        games.append({
            'winner': winner, 'loser': loser,
            'wr': wr, 'lr': lr,
            'winner_region': wr_reg, 'loser_region': lr_reg,
        })
    return games

def win_prob(rA, rB, beta):
    return float(np.clip(expit(beta * (rA - rB)), 1e-9, 1-1e-9))

def brier(p): return (p - 1.0)**2
def logloss(p): return -math.log(p)

def main():
    print('='*70)
    print('DIAGNOSTIC: Why Does Calibration Fail?')
    print('='*70)

    ratings_data = load_ratings()

    # ── Part 1: What does the oracle offset say? ──────────────────────────────
    # For each holdout event: what's the BEST POSSIBLE constant regional offset?
    print('\n\n── PART 1: Oracle regional offset search ────────────────────────────')
    print('What regional offset would have minimized Brier AT each holdout event?')
    print('(This is the absolute ceiling — uses future data)')

    for eid in BACKTEST_HOLDOUTS:
        cfg    = INTL_CONFIG.get(eid, {})
        yr, sn = cfg.get('year'), cfg.get('snap')
        pre    = get_snap_ratings(ratings_data, yr, sn)
        beta   = get_snap_beta(ratings_data, yr, sn)
        games  = load_intl_games(eid)

        print(f'\n  {eid} (n={len(games)} cross-regional maps)')
        print(f'  Using snapshot: {yr}/{sn}  beta={beta:.3f}')

        # First compute domestic-only brier
        b_dom = []
        for g in games:
            rw = pre.get(g['winner'])
            rl = pre.get(g['loser'])
            if rw is None or rl is None:
                continue
            p = win_prob(rw, rl, beta)
            b_dom.append(brier(p))
        print(f'  Domestic Brier: {np.mean(b_dom):.5f}')

        # Oracle: optimize a single shared offset for all teams in this event
        # (unrealistic, but sets upper bound)
        def neg_brier_global_offset(offset):
            bs = []
            for g in games:
                rw = pre.get(g['winner'])
                rl = pre.get(g['loser'])
                if rw is None or rl is None:
                    continue
                # apply offset symmetrically shifts nothing unless asymmetric
                p = win_prob(rw, rl, beta)
                bs.append(brier(p))
            return float(np.mean(bs))

        # Oracle: optimize per-region offsets
        regions = ['EMEA', 'Americas', 'Pacific']

        def brier_with_offsets(offsets):
            off = dict(zip(regions, offsets))
            bs  = []
            for g in games:
                rw = pre.get(g['winner'])
                rl = pre.get(g['loser'])
                if rw is None or rl is None:
                    continue
                rw_cal = rw + off.get(g['winner_region'], 0)
                rl_cal = rl + off.get(g['loser_region'],  0)
                p = win_prob(rw_cal, rl_cal, beta)
                bs.append(brier(p))
            return float(np.mean(bs)) if bs else 0.5

        result = minimize(brier_with_offsets, [0.0, 0.0, 0.0], method='Nelder-Mead',
                          options={'xatol': 1e-5, 'fatol': 1e-7, 'maxiter': 5000})
        oracle_b = result.fun
        oracle_off = dict(zip(regions, result.x))
        print(f'  Oracle Brier (3-region):  {oracle_b:.5f}  (offsets: {oracle_off})')
        print(f'  Oracle improvement: {np.mean(b_dom) - oracle_b:+.5f}')

        # Per-team oracle offsets (absolute upper bound)
        teams = list(set([g['winner'] for g in games] + [g['loser'] for g in games]))
        teams = [t for t in teams if pre.get(t) is not None]

        def brier_with_team_offsets(offsets):
            off = dict(zip(teams, offsets))
            bs  = []
            for g in games:
                rw = pre.get(g['winner'])
                rl = pre.get(g['loser'])
                if rw is None or rl is None:
                    continue
                rw_cal = rw + off.get(g['winner'], 0)
                rl_cal = rl + off.get(g['loser'],  0)
                p = win_prob(rw_cal, rl_cal, beta)
                bs.append(brier(p))
            return float(np.mean(bs)) if bs else 0.5

        result_t = minimize(brier_with_team_offsets, [0.0]*len(teams), method='Nelder-Mead',
                            options={'xatol': 1e-4, 'fatol': 1e-6, 'maxiter': 50000})
        oracle_team_b = result_t.fun
        print(f'  Oracle Brier (per-team):  {oracle_team_b:.5f}')
        print(f'  Oracle per-team improvement: {np.mean(b_dom) - oracle_team_b:+.5f}')

    # ── Part 2: Cross-event signal consistency ────────────────────────────────
    print('\n\n── PART 2: Cross-event signal consistency ───────────────────────────')
    print('Do regional win-rates at Masters predict win-rates at Champions?')

    training_events = ['2023_masters_tokyo', '2023_champions',
                       '2024_masters_madrid', '2024_masters_shanghai',
                       '2025_masters_bangkok', '2025_masters_toronto']

    # For each event, compute actual vs expected win rate by region
    print('\n  Per-event actual vs expected win rates by region:')
    print(f'  {"Event":<25s}  {"EMEA":>8s}  {"Americas":>8s}  {"Pacific":>8s}  n_games')

    for eid in training_events + BACKTEST_HOLDOUTS:
        cfg    = INTL_CONFIG.get(eid, {})
        yr, sn = cfg.get('year'), cfg.get('snap')
        if not yr or not sn:
            continue
        pre  = get_snap_ratings(ratings_data, yr, sn)
        beta = get_snap_beta(ratings_data, yr, sn)
        games = load_intl_games(eid)

        actual_w   = defaultdict(int)
        expected_w = defaultdict(float)
        n_games_r  = defaultdict(int)

        for g in games:
            rw = pre.get(g['winner'])
            rl = pre.get(g['loser'])
            if rw is None or rl is None:
                continue
            p = win_prob(rw, rl, beta)
            actual_w[g['winner_region']]   += 1
            actual_w[g['loser_region']]    += 0
            expected_w[g['winner_region']] += p
            expected_w[g['loser_region']]  += (1-p)
            n_games_r[g['winner_region']]  += 1
            n_games_r[g['loser_region']]   += 1

        # residual = actual - expected for each region
        resids = {}
        for reg in ['EMEA', 'Americas', 'Pacific']:
            exp = expected_w.get(reg, 0)
            act = actual_w.get(reg, 0)
            n   = n_games_r.get(reg, 0)
            resids[reg] = (act - exp) / max(n, 1) if n > 0 else float('nan')

        total_n = sum(n_games_r.values()) // 2
        marker = ' <-- HOLDOUT' if eid in BACKTEST_HOLDOUTS else ''
        print(f'  {eid:<25s}  {resids.get("EMEA", float("nan")):>+8.3f}  '
              f'{resids.get("Americas", float("nan")):>+8.3f}  '
              f'{resids.get("Pacific", float("nan")):>+8.3f}  '
              f'{total_n:3d}{marker}')

    # ── Part 3: Is signal at Masters predictive of signal at Champions? ───────
    print('\n\n── PART 3: Masters → Champions predictability ───────────────────────')

    # 2024: Madrid + Shanghai → Champions
    # 2025: Bangkok + Toronto → Champions

    test_pairs = [
        ('2024', '2024_champions',
         ['2024_masters_madrid', '2024_masters_shanghai'], 'before_champions'),
        ('2025', '2025_champions',
         ['2025_masters_bangkok', '2025_masters_toronto'], 'before_champions'),
    ]

    for yr, holdout_eid, prior_eids, snap in test_pairs:
        print(f'\n  Year {yr}: prior={prior_eids} → {holdout_eid}')

        # Compute regional residual signal in training events
        prior_resids = {}
        for reg in ['EMEA', 'Americas', 'Pacific']:
            signals = []
            for eid in prior_eids:
                cfg2   = INTL_CONFIG.get(eid, {})
                yr2,sn2 = cfg2.get('year'), cfg2.get('snap')
                pre2   = get_snap_ratings(ratings_data, yr2, sn2)
                beta2  = get_snap_beta(ratings_data, yr2, sn2)
                games2 = load_intl_games(eid)

                actual_w2   = defaultdict(int)
                expected_w2 = defaultdict(float)
                n_games_r2  = defaultdict(int)

                for g in games2:
                    rw = pre2.get(g['winner'])
                    rl = pre2.get(g['loser'])
                    if rw is None or rl is None:
                        continue
                    p = win_prob(rw, rl, beta2)
                    actual_w2[g['winner_region']]   += 1
                    expected_w2[g['winner_region']] += p
                    expected_w2[g['loser_region']]  += (1-p)
                    n_games_r2[g['winner_region']]  += 1
                    n_games_r2[g['loser_region']]   += 1

                n = n_games_r2.get(reg, 0)
                if n > 0:
                    act = actual_w2.get(reg, 0)
                    exp = expected_w2.get(reg, 0)
                    signals.append((act - exp) / n)

            prior_resids[reg] = float(np.mean(signals)) if signals else 0.0

        print(f'  Prior signal (win-rate residual per game by region):')
        for reg, sig in prior_resids.items():
            print(f'    {reg}: {sig:+.4f}')

        # Now check actual at Champions
        cfg_h = INTL_CONFIG.get(holdout_eid, {})
        yr_h, sn_h = cfg_h.get('year'), cfg_h.get('snap')
        pre_h  = get_snap_ratings(ratings_data, yr_h, sn_h)
        beta_h = get_snap_beta(ratings_data, yr_h, sn_h)
        games_h = load_intl_games(holdout_eid)

        actual_wh   = defaultdict(int)
        expected_wh = defaultdict(float)
        n_games_rh  = defaultdict(int)
        for g in games_h:
            rw = pre_h.get(g['winner'])
            rl = pre_h.get(g['loser'])
            if rw is None or rl is None:
                continue
            p = win_prob(rw, rl, beta_h)
            actual_wh[g['winner_region']]   += 1
            expected_wh[g['winner_region']] += p
            expected_wh[g['loser_region']]  += (1-p)
            n_games_rh[g['winner_region']]  += 1
            n_games_rh[g['loser_region']]   += 1

        print(f'  Actual at Champions (win-rate residual per game by region):')
        for reg in ['EMEA', 'Americas', 'Pacific']:
            n = n_games_rh.get(reg, 0)
            if n > 0:
                act = actual_wh.get(reg, 0)
                exp = expected_wh.get(reg, 0)
                actual_sig = (act - exp) / n
                prior_sig  = prior_resids.get(reg, 0)
                same_dir   = (prior_sig > 0) == (actual_sig > 0)
                print(f'    {reg}: prior={prior_sig:+.4f}  actual_at_champs={actual_sig:+.4f}  '
                      f'{"SAME DIR" if same_dir else "OPPOSITE DIR"}')

    # ── Part 4: Per-team rating calibration ───────────────────────────────────
    print('\n\n── PART 4: Domestic rating accuracy at holdout events ────────────────')
    print('How well-calibrated are the domestic ratings for cross-regional games?')

    for eid in BACKTEST_HOLDOUTS:
        cfg    = INTL_CONFIG.get(eid, {})
        yr, sn = cfg.get('year'), cfg.get('snap')
        pre    = get_snap_ratings(ratings_data, yr, sn)
        beta   = get_snap_beta(ratings_data, yr, sn)
        games  = load_intl_games(eid)

        print(f'\n  {eid}:')

        # What fraction of games does the favorite win?
        fav_wins = 0
        total    = 0
        upset_briers = []
        fav_briers   = []

        for g in games:
            rw = pre.get(g['winner'])
            rl = pre.get(g['loser'])
            if rw is None or rl is None:
                continue
            p = win_prob(rw, rl, beta)
            total += 1
            if p > 0.5:
                fav_wins += 1
                fav_briers.append(brier(p))
            else:
                upset_briers.append(brier(p))

        print(f'  Favorite wins: {fav_wins}/{total} = {fav_wins/max(total,1):.1%}')
        print(f'  Avg Brier (favorite games): {np.mean(fav_briers):.5f}' if fav_briers else '  N/A')
        print(f'  Avg Brier (upset games):    {np.mean(upset_briers):.5f}' if upset_briers else '  N/A')

        # Calibration by probability bucket
        print(f'  Calibration by predicted probability:')
        buckets = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
        for lo, hi in buckets:
            matches_in_bucket = []
            for g in games:
                rw = pre.get(g['winner'])
                rl = pre.get(g['loser'])
                if rw is None or rl is None:
                    continue
                p = win_prob(rw, rl, beta)
                # winner's perspective
                if lo <= p < hi:
                    matches_in_bucket.append(1)
                elif lo <= (1-p) < hi:
                    matches_in_bucket.append(0)  # upset
            if matches_in_bucket:
                mid = (lo + hi) / 2
                actual_rate = np.mean(matches_in_bucket)
                print(f'    [{lo:.1f},{hi:.1f}): n={len(matches_in_bucket):2d}  '
                      f'predicted_avg≈{mid:.2f}  actual_win_rate={actual_rate:.3f}')

        # Team-level ratings
        print(f'\n  Team ratings at {eid}:')
        teams_at_event = set()
        for g in games:
            teams_at_event.add(g['winner'])
            teams_at_event.add(g['loser'])

        team_data = []
        for t in teams_at_event:
            if t in pre:
                reg = ORG_REGIONS.get(t, '?')
                team_data.append((t, reg, pre[t]))

        for t, reg, r in sorted(team_data, key=lambda x: -x[2]):
            print(f'    {t:6s} ({reg:8s}): {r:+.2f}')

    # ── Part 5: Information-theoretic limit ───────────────────────────────────
    print('\n\n── PART 5: How much signal do prior Masters give? ────────────────────')
    print('Correlation between prior Masters performance and Champions performance')

    for yr, holdout_eid, prior_eids, snap in test_pairs:
        print(f'\n  Year {yr}:')

        cfg_h = INTL_CONFIG.get(holdout_eid, {})
        yr_h, sn_h = cfg_h.get('year'), cfg_h.get('snap')
        pre_h  = get_snap_ratings(ratings_data, yr_h, sn_h)
        beta_h = get_snap_beta(ratings_data, yr_h, sn_h)
        games_h = load_intl_games(holdout_eid)

        # Per-team: win rate residual at prior Masters
        prior_perf = {}
        for eid in prior_eids:
            cfg2   = INTL_CONFIG.get(eid, {})
            yr2,sn2 = cfg2.get('year'), cfg2.get('snap')
            pre2   = get_snap_ratings(ratings_data, yr2, sn2)
            beta2  = get_snap_beta(ratings_data, yr2, sn2)
            games2 = load_intl_games(eid)

            for g in games2:
                rw = pre2.get(g['winner'])
                rl = pre2.get(g['loser'])
                if rw is None or rl is None:
                    continue
                p    = win_prob(rw, rl, beta2)
                if g['winner'] not in prior_perf:
                    prior_perf[g['winner']] = []
                if g['loser'] not in prior_perf:
                    prior_perf[g['loser']] = []
                prior_perf[g['winner']].append(1.0 - p)
                prior_perf[g['loser']].append(0.0 - (1-p))

        # Per-team: win rate residual at holdout Champions
        champ_perf = {}
        for g in games_h:
            rw = pre_h.get(g['winner'])
            rl = pre_h.get(g['loser'])
            if rw is None or rl is None:
                continue
            p = win_prob(rw, rl, beta_h)
            if g['winner'] not in champ_perf:
                champ_perf[g['winner']] = []
            if g['loser'] not in champ_perf:
                champ_perf[g['loser']] = []
            champ_perf[g['winner']].append(1.0 - p)
            champ_perf[g['loser']].append(0.0 - (1-p))

        # Correlate
        common_teams = set(prior_perf) & set(champ_perf)
        if len(common_teams) < 3:
            print(f'  Not enough teams to correlate (found {len(common_teams)})')
            continue

        x = [float(np.mean(prior_perf[t]))  for t in common_teams]
        y = [float(np.mean(champ_perf[t]))  for t in common_teams]

        corr = np.corrcoef(x, y)[0,1]
        print(f'  Teams in both prior Masters + Champions: {len(common_teams)}')
        print(f'  Pearson correlation (prior_resid vs champs_resid): {corr:.3f}')
        print(f'  (near 0 = prior Masters not predictive of Champions performance)')

        print(f'\n  Per-team (prior_masters_resid → champs_resid):')
        combined = [(t, float(np.mean(prior_perf[t])), float(np.mean(champ_perf[t])))
                    for t in sorted(common_teams)]
        for t, pr, cr in sorted(combined, key=lambda x: -x[1]):
            reg = ORG_REGIONS.get(t, '?')
            print(f'    {t:6s} ({reg:8s}): prior={pr:+.4f}  champs={cr:+.4f}')

    # ── Part 6: Can beta calibration help? ────────────────────────────────────
    print('\n\n── PART 6: Beta (logistic slope) recalibration ──────────────────────')
    print('What if we recalibrate beta for cross-regional games?')

    for eid in BACKTEST_HOLDOUTS:
        cfg    = INTL_CONFIG.get(eid, {})
        yr, sn = cfg.get('year'), cfg.get('snap')
        pre    = get_snap_ratings(ratings_data, yr, sn)
        beta0  = get_snap_beta(ratings_data, yr, sn)
        games  = load_intl_games(eid)

        games_rated = [(g, pre.get(g['winner']), pre.get(g['loser']))
                       for g in games if pre.get(g['winner']) and pre.get(g['loser'])]

        def brier_at_beta(beta_val):
            bs = []
            for g, rw, rl in games_rated:
                p = win_prob(rw, rl, beta_val)
                bs.append(brier(p))
            return float(np.mean(bs))

        # Grid search beta
        betas = np.arange(0.05, 1.5, 0.05)
        brier_by_beta = [(b_val, brier_at_beta(b_val)) for b_val in betas]
        best_b_beta, best_b_brier = min(brier_by_beta, key=lambda x: x[1])

        print(f'\n  {eid}:')
        print(f'  Default beta={beta0:.3f}  Brier={brier_at_beta(beta0):.5f}')
        print(f'  Oracle  beta={best_b_beta:.3f}  Brier={best_b_brier:.5f}')
        print(f'  Oracle improvement: {brier_at_beta(beta0) - best_b_brier:+.5f}')

        # Show beta landscape
        print(f'  Beta landscape (brier):')
        for b_val, b_brier in brier_by_beta:
            marker = ' <-- oracle' if b_val == best_b_beta else ''
            marker2 = ' <-- default' if abs(b_val - beta0) < 0.03 else ''
            if abs(b_val - best_b_beta) < 0.2 or abs(b_val - beta0) < 0.15:
                print(f'    beta={b_val:.2f}  brier={b_brier:.5f}{marker}{marker2}')

    print('\n\nDiagnostic complete.')


if __name__ == '__main__':
    main()
