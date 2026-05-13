"""
Comprehensive backtest of series-level win-probability predictions on the
current post-CN BenPom model. Used to find:

  1. Optimal β (logit scale) for series predictions
  2. Cross-region β multiplier vs same-region
  3. Per-segment calibration biases (region, era, CN, rating gap)
  4. Whether temperature scaling / Platt scaling further reduces Brier
  5. The std-dev of probability error per prediction (variance bound)

Walks all historical SERIES matches (MapNum='all'). For each match, uses the
latest snapshot whose ref_date is strictly before the match date — strictly
leak-free. Reports Brier, log loss, ECE, decomposition, and a sweep over β.

Usage:
    python3 scrapers/BacktestSeriesPredictions.py
"""

import json, os, sys, math
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from MoreTestingMaybeFiles import ALL_EVENTS

REGIONS = {}
for _e in ALL_EVENTS:
    pass
# Use the same ORG_REGIONS used elsewhere
ORG_REGIONS = {
    'TL':'EMEA','FNC':'EMEA','NAVI':'EMEA','VIT':'EMEA','BBL':'EMEA','GX':'EMEA','KC':'EMEA','TH':'EMEA',
    'FUT':'EMEA','GIA':'EMEA','MKOI':'EMEA','M8':'EMEA','EF':'EMEA','PCF':'EMEA',
    'SEN':'Americas','G2':'Americas','MIBR':'Americas','NRG':'Americas','100T':'Americas','C9':'Americas',
    'EG':'Americas','KRÜ':'Americas','LEV':'Americas','FUR':'Americas','LOUD':'Americas','ENVY':'Americas',
    'PRX':'Pacific','DRX':'Pacific','T1':'Pacific','TLN':'Pacific','GEN':'Pacific','DFM':'Pacific',
    'ZETA':'Pacific','RRQ':'Pacific','TS':'Pacific','GE':'Pacific','NS':'Pacific','FS':'Pacific',
    'KRX':'Pacific','VL':'Pacific',
}
CN = {'EDG','BLG','TE','DRG','ASE','AG','XLG','FPX','JDG','NOVA','TEC','TYL','TYLOO','WOL'}
for t in CN:
    ORG_REGIONS[t] = 'CN'


def load_snapshots():
    """Returns list of (ref_date, year, snap_name, teams_dict) sorted by ref_date."""
    with open(os.path.join(ROOT, 'data/map_ratings.json')) as f:
        mr = json.load(f)
    out = []
    for y, yblock in mr['ratings'].items():
        for snap, sdata in yblock['snapshots'].items():
            rd = sdata.get('ref_date')
            if not rd:
                continue
            teams = sdata.get('teams', {})
            ratings = {t: v.get('overall_rating', 0.0) for t, v in teams.items()}
            out.append((rd, y, snap, ratings))
    out.sort(key=lambda x: x[0])
    return out


def load_match_data():
    """Returns dataframe: MatchID, date, a, b, a_wins.

    Both teams are pulled from the per-match player-level maps CSVs so
    series sweeps (2-0) are included along with 2-1s. To remove
    A-vs-B ordering bias entirely we then sort (a,b) alphabetically so the
    set of (a,b) is canonical and we assign a_wins based on the alpha
    ordering — NOT on appearance order in the source data.
    """
    mr = pd.read_csv(os.path.join(ROOT, 'data/match_results.csv'))
    series = mr[mr['MapNum'] == 'all'].copy()
    series['MatchID'] = series['MatchID'].astype(int)

    with open(os.path.join(ROOT, 'data/match_dates.json')) as f:
        dates = json.load(f)

    # Build MatchID → (teamA, teamB) from per-match-per-player maps CSVs
    teams_per = {}
    maps_dir = os.path.join(ROOT, 'data/maps')
    for fname in os.listdir(maps_dir):
        if not fname.endswith('.csv'):
            continue
        try:
            mdf = pd.read_csv(os.path.join(maps_dir, fname), usecols=['MatchID', 'Org'])
        except Exception:
            continue
        for mid, grp in mdf.groupby('MatchID'):
            orgs = list(grp['Org'].dropna().unique())
            if len(orgs) == 2:
                a, b = sorted(orgs)  # canonical alphabetical ordering — kills any A/B bias
                teams_per[int(mid)] = (a, b)

    rows = []
    for _, r in series.iterrows():
        mid = int(r['MatchID'])
        d = dates.get(str(mid)) or dates.get(mid)
        if not d:
            continue
        pair = teams_per.get(mid)
        if not pair:
            continue
        a, b = pair  # alphabetically ordered
        winner = r['WinnerOrg']
        if winner not in (a, b):
            continue
        a_wins = int(winner == a)
        rows.append({'mid': mid, 'date': d, 'a': a, 'b': b, 'a_wins': a_wins})
    df = pd.DataFrame(rows)
    df.sort_values('date', inplace=True)
    return df.reset_index(drop=True)


def find_snapshot_for(date, snapshots):
    """Latest snapshot with ref_date < date."""
    best = None
    for rd, y, snap, ratings in snapshots:
        if rd < date:
            best = (rd, y, snap, ratings)
        else:
            break
    return best


def brier(probs, outs):
    return float(np.mean((np.asarray(probs) - np.asarray(outs)) ** 2))


def logloss(probs, outs):
    p = np.clip(np.asarray(probs), 1e-9, 1 - 1e-9)
    o = np.asarray(outs)
    return float(-np.mean(o * np.log(p) + (1 - o) * np.log(1 - p)))


def expected_calibration_error(probs, outs, n_bins=10):
    """ECE with equal-width bins."""
    p = np.asarray(probs)
    o = np.asarray(outs)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    ece = 0.0
    n = len(p)
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        bin_acc = o[mask].mean()
        bin_conf = p[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def brier_decomposition(probs, outs, n_bins=10):
    """Returns (reliability, resolution, uncertainty)."""
    p = np.asarray(probs)
    o = np.asarray(outs)
    obar = o.mean()
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    rel = 0.0
    res = 0.0
    n = len(p)
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        nb = mask.sum()
        pbar = p[mask].mean()
        obar_b = o[mask].mean()
        rel += (nb / n) * (pbar - obar_b) ** 2
        res += (nb / n) * (obar_b - obar) ** 2
    unc = obar * (1 - obar)
    return rel, res, unc


def predict_series(rA, rB, beta, cross_region_mult=1.0, cross=False, fmt='bo3'):
    """Map-level P(A wins) → series P(A wins)."""
    b = beta * cross_region_mult if cross else beta
    p = 1.0 / (1.0 + math.exp(-b * (rA - rB)))
    p = max(1e-6, min(1 - 1e-6, p))
    if fmt == 'bo5':
        # P(A wins BO5) = sum_{k=0..2} C(2+k,k) p^3 (1-p)^k = p^3 + 3 p^3 (1-p) + 6 p^3 (1-p)^2
        return p**3 + 3*p**3*(1-p) + 6*p**3*(1-p)**2
    elif fmt == 'bo1':
        return p
    else:  # bo3 default
        return p**2 + 2*p*(1-p)*p  # = p^2 * (3 - 2p)


def gen_predictions(matches, snapshots, beta, cross_mult=1.0, fmt='bo3', require_both=True):
    """Returns list of dicts: {p, y, cross, region_a, region_b, ...}"""
    out = []
    for _, m in matches.iterrows():
        snap = find_snapshot_for(m['date'], snapshots)
        if not snap:
            continue
        rd, y, sname, rt = snap
        ra = rt.get(m['a'])
        rb = rt.get(m['b'])
        if require_both and (ra is None or rb is None):
            continue
        if ra is None: ra = 0.0
        if rb is None: rb = 0.0
        regA = ORG_REGIONS.get(m['a'], '?')
        regB = ORG_REGIONS.get(m['b'], '?')
        cross = (regA != regB and regA != '?' and regB != '?')
        p = predict_series(ra, rb, beta, cross_region_mult=cross_mult, cross=cross, fmt=fmt)
        out.append({
            'mid': m['mid'],
            'date': m['date'],
            'p': p,
            'y': m['a_wins'],
            'a': m['a'], 'b': m['b'],
            'ra': ra, 'rb': rb,
            'gap': abs(ra - rb),
            'cross': cross,
            'regA': regA, 'regB': regB,
            'snap_used': f'{y}_{sname}',
            'cn_involved': (regA == 'CN' or regB == 'CN'),
        })
    return out


def main():
    print('Loading snapshots and matches...')
    snapshots = load_snapshots()
    matches = load_match_data()
    print(f'  {len(snapshots)} snapshots, {len(matches)} series-level matches')
    print(f'  Date range: {matches["date"].min()} to {matches["date"].max()}')

    # Reserve 2026 for final report; use 2024+2025 for tuning where possible
    train_matches = matches[matches['date'] < '2026-01-01'].copy()
    test_matches  = matches[matches['date'] >= '2026-01-01'].copy()
    print(f'  Train (pre-2026): {len(train_matches)}, Test (2026): {len(test_matches)}')

    # ─── Step 1: Sweep β on TRAIN, find optimum ─────────────────────────────
    print('\n━━━ Step 1: β sweep on train data (single β, no cross mult) ━━━')
    betas = [0.05, 0.08, 0.10, 0.12, 0.14, 0.16, 0.17, 0.18, 0.20, 0.22, 0.25, 0.30, 0.35, 0.40]
    print(f'{"β":>6}  {"n":>5}  {"Brier":>7}  {"LogLoss":>8}  {"ECE":>6}  {"Rel":>7}  {"Res":>7}')
    best_beta = None
    best_brier = 1.0
    for b in betas:
        preds = gen_predictions(train_matches, snapshots, b)
        if not preds:
            continue
        probs = [d['p'] for d in preds]
        outs  = [d['y'] for d in preds]
        br = brier(probs, outs)
        ll = logloss(probs, outs)
        ece = expected_calibration_error(probs, outs)
        rel, res, unc = brier_decomposition(probs, outs)
        print(f'{b:>6.2f}  {len(preds):>5}  {br:.5f}  {ll:.5f}  {ece:.4f}  {rel:.5f}  {res:.5f}')
        if br < best_brier:
            best_brier = br
            best_beta = b
    print(f'\n  Best single-β: β={best_beta}, Brier={best_brier:.5f}')

    # Finer sweep around best
    print('\n━━━ Step 1b: Fine β sweep around best ━━━')
    if best_beta is not None:
        fine = np.arange(max(0.02, best_beta - 0.05), best_beta + 0.06, 0.01)
        print(f'{"β":>6}  {"Brier":>7}  {"LogLoss":>8}  {"ECE":>6}')
        for b in fine:
            preds = gen_predictions(train_matches, snapshots, b)
            probs = [d['p'] for d in preds]; outs = [d['y'] for d in preds]
            print(f'{b:>6.3f}  {brier(probs, outs):.5f}  {logloss(probs, outs):.5f}  {expected_calibration_error(probs, outs):.4f}')

    # ─── Step 2: Split by region / cross / CN ────────────────────────────────
    print('\n━━━ Step 2: Segment analysis at β=best ━━━')
    preds = gen_predictions(train_matches, snapshots, best_beta)
    print(f'{"Segment":<22}  {"n":>5}  {"Brier":>7}  {"LogLoss":>8}  {"ECE":>6}  {"Mean p":>7}  {"Mean y":>7}')
    segments = [
        ('All', lambda d: True),
        ('Same-region', lambda d: not d['cross']),
        ('Cross-region', lambda d: d['cross']),
        ('CN involved', lambda d: d['cn_involved']),
        ('No CN', lambda d: not d['cn_involved']),
        ('Domestic non-CN', lambda d: (not d['cross']) and (not d['cn_involved'])),
        ('|gap| < 1', lambda d: d['gap'] < 1.0),
        ('|gap| 1-2', lambda d: 1.0 <= d['gap'] < 2.0),
        ('|gap| 2-3', lambda d: 2.0 <= d['gap'] < 3.0),
        ('|gap| > 3', lambda d: d['gap'] >= 3.0),
    ]
    for name, fn in segments:
        sub = [d for d in preds if fn(d)]
        if not sub:
            continue
        ps = [d['p'] for d in sub]; ys = [d['y'] for d in sub]
        print(f'{name:<22}  {len(sub):>5}  {brier(ps,ys):.5f}  {logloss(ps,ys):.5f}  {expected_calibration_error(ps,ys):.4f}  {np.mean(ps):.4f}  {np.mean(ys):.4f}')

    # ─── Step 3: Tune cross-region multiplier ───────────────────────────────
    print('\n━━━ Step 3: Cross-region β multiplier sweep ━━━')
    print(f'  Sweeping with base β={best_beta}')
    print(f'{"x-mult":>7}  {"x-Brier":>9}  {"All Brier":>10}')
    best_xmult = 1.0
    best_xbrier = 1.0
    for xm in [0.3, 0.4, 0.5, 0.6, 0.664, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
        preds = gen_predictions(train_matches, snapshots, best_beta, cross_mult=xm)
        xpreds = [d for d in preds if d['cross']]
        xps = [d['p'] for d in xpreds]; xys = [d['y'] for d in xpreds]
        all_ps = [d['p'] for d in preds]; all_ys = [d['y'] for d in preds]
        xb = brier(xps, xys) if xpreds else float('nan')
        ab = brier(all_ps, all_ys)
        marker = ''
        if xb < best_xbrier:
            best_xbrier = xb
            best_xmult = xm
            marker = '  ←'
        print(f'{xm:>7.3f}  {xb:.5f}    {ab:.5f}{marker}')

    # ─── Step 4: Year segments ──────────────────────────────────────────────
    print('\n━━━ Step 4: Per-year breakdown @ best β + best xmult ━━━')
    preds = gen_predictions(train_matches, snapshots, best_beta, cross_mult=best_xmult)
    print(f'{"Year":>6}  {"n":>4}  {"Brier":>7}  {"LogLoss":>8}  {"ECE":>6}')
    yr_groups = defaultdict(list)
    for d in preds:
        yr_groups[d['date'][:4]].append(d)
    for yr in sorted(yr_groups):
        sub = yr_groups[yr]
        ps = [d['p'] for d in sub]; ys = [d['y'] for d in sub]
        print(f'{yr:>6}  {len(sub):>4}  {brier(ps,ys):.5f}  {logloss(ps,ys):.5f}  {expected_calibration_error(ps,ys):.4f}')

    # ─── Step 5: Held-out 2026 test ─────────────────────────────────────────
    print('\n━━━ Step 5: Held-out 2026 test (NEVER USED FOR TUNING) ━━━')
    if len(test_matches):
        preds_test = gen_predictions(test_matches, snapshots, best_beta, cross_mult=best_xmult)
        ps = [d['p'] for d in preds_test]; ys = [d['y'] for d in preds_test]
        if ps:
            print(f'  n={len(preds_test)}')
            print(f'  Brier:   {brier(ps,ys):.5f}')
            print(f'  LogLoss: {logloss(ps,ys):.5f}')
            print(f'  ECE:     {expected_calibration_error(ps,ys):.4f}')
            rel, res, unc = brier_decomposition(ps, ys)
            print(f'  Decomp:  rel={rel:.5f}  res={res:.5f}  unc={unc:.5f}')
            # Distribution of |error|
            errs = np.abs(np.array(ps) - np.array(ys))
            print(f'  |p - y|  mean={errs.mean():.4f}  median={np.median(errs):.4f}  std={errs.std():.4f}  max={errs.max():.4f}')

    # ─── Step 6: Reliability bins on FULL data ──────────────────────────────
    print('\n━━━ Step 6: Reliability bins (full train @ best β + xmult) ━━━')
    preds = gen_predictions(train_matches, snapshots, best_beta, cross_mult=best_xmult)
    ps = np.array([d['p'] for d in preds]); ys = np.array([d['y'] for d in preds])
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(ps, bins) - 1, 0, 9)
    print(f'{"Bin":>10}  {"n":>5}  {"Mean p":>8}  {"Actual":>8}  {"Diff":>7}')
    for b in range(10):
        mask = idx == b
        if not mask.any():
            continue
        mp = ps[mask].mean(); ma = ys[mask].mean()
        print(f'[{b*0.1:.1f}-{(b+1)*0.1:.1f}]  {mask.sum():>5}  {mp:.4f}    {ma:.4f}    {ma-mp:+.4f}')

    # ─── Step 7: Temperature scaling on validation split ────────────────────
    print('\n━━━ Step 7: Temperature scaling (logit*T) ━━━')
    # Cut: 80% chronological train, 20% calib
    n = len(preds)
    split = int(n * 0.8)
    cal_preds = preds[split:]
    logits = np.array([math.log(d['p'] / (1 - d['p'])) for d in cal_preds])
    ys_cal = np.array([d['y'] for d in cal_preds])

    def nll_T(T):
        z = expit(logits * T)
        z = np.clip(z, 1e-9, 1 - 1e-9)
        return -np.mean(ys_cal * np.log(z) + (1 - ys_cal) * np.log(1 - z))

    res = minimize_scalar(nll_T, bounds=(0.1, 5.0), method='bounded')
    T = float(res.x)
    print(f'  Optimal T (on 20% calibration tail): {T:.4f}')
    # Apply T to ALL predictions and re-measure
    full_logits = np.array([math.log(d['p'] / (1 - d['p'])) for d in preds])
    full_ys = np.array([d['y'] for d in preds])
    scaled = expit(full_logits * T)
    print(f'  Pre-T Brier:  {brier(ps, ys):.5f}')
    print(f'  Post-T Brier: {brier(scaled, full_ys):.5f}')
    print(f'  Pre-T ECE:   {expected_calibration_error(ps, ys):.4f}')
    print(f'  Post-T ECE:  {expected_calibration_error(scaled, full_ys):.4f}')

    # Apply T to held-out 2026
    if len(test_matches):
        preds_test = gen_predictions(test_matches, snapshots, best_beta, cross_mult=best_xmult)
        tlog = np.array([math.log(d['p'] / (1 - d['p'])) for d in preds_test])
        tys = np.array([d['y'] for d in preds_test])
        tps = np.array([d['p'] for d in preds_test])
        ts = expit(tlog * T)
        print(f'  2026 Pre-T Brier:  {brier(tps, tys):.5f}')
        print(f'  2026 Post-T Brier: {brier(ts, tys):.5f}')

    # ─── Step 8: Final summary ───────────────────────────────────────────────
    print('\n━━━ FINAL RECOMMENDED PARAMETERS ━━━')
    print(f'  β = {best_beta}')
    print(f'  cross_region_mult = {best_xmult}')
    print(f'  temperature_scale = {T:.4f}')
    print(f'  → effective same-region β = {best_beta * T:.4f}')
    print(f'  → effective cross-region β = {best_beta * best_xmult * T:.4f}')
    print(f'\n  Production currently uses β=0.17 (hardcoded in MapElo.py:3329)')


if __name__ == '__main__':
    main()
