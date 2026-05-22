#!/usr/bin/env python3
"""
ResearchPostCnResiduals.py

Hunt for OTHER systematic residuals in the optimized BenPom model.

Optimized model:
    p_series = sigmoid(0.154 * delta_rating
                       + 0.22 * intl_exp_diff
                       + 0.47 * 1[CN-is-dog at intl])
        - intl_exp_diff applied only at intl events
        - CN-dog bump applied only at intl events when fav region != CN

For each bucket, compute n, mean predicted, mean actual, residual, Brier,
95% binomial CI, and LR-test p-value of `y ~ logit(p_base) + bucket_indicator`.

Buckets:
  1. Other "X-is-dog" by region at intl (Pacific/EMEA/Americas dog)
  2. CN-fav at intl (asymmetric counterpart to CN-dog)
  3. Per-CN team residual at intl
  4. Per non-CN team facing CN at intl
  5. bo5 vs bo3 conditional on cross-region intl
  6. Champions vs Masters vs non-intl event-type residual
  7. First match of the team at an event
  8. Group stage vs playoffs (proxy: match-position in event)
  9. Cross-region intl pair (regionA vs regionB) per-pair fav over/under
"""

import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from MapElo import ORG_REGIONS  # noqa: E402

# ---- config ----
BETA = 0.154
INTL_EXP_COEF = 0.22
CN_DOG_BUMP = 0.47

INTL_EVENTS = {
    '2024_masters_madrid', '2024_masters_shanghai', '2024_champions',
    '2025_masters_bangkok', '2025_masters_toronto', '2025_champions',
    '2026_masters_santiago',
}

CHAMPIONS_EVENTS = {'2024_champions', '2025_champions'}
MASTERS_EVENTS = INTL_EVENTS - CHAMPIONS_EVENTS

DATA_FILES = [
    'data/rating_timeline_2024.json',
    'data/rating_timeline_2025.json',
    'data/rating_timeline.json',  # 2026
]

OUT_JSON = 'data/projection_research_post_cn.json'


# ---- helpers ----
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def logit(p):
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def bo3_p_series(p_map):
    # P(win bo3 | p_map) = p^2 (1 + 2(1-p))  -> p^2*(3-2p)
    return p_map * p_map * (3 - 2 * p_map)


def bo5_p_series(p_map):
    # Closed form: p^3*(1 + 3(1-p) + 6(1-p)^2)
    q = 1 - p_map
    return p_map ** 3 * (1 + 3 * q + 6 * q * q)


def region_of(team):
    return ORG_REGIONS.get(team)


def load_matches():
    out = []
    for fp in DATA_FILES:
        d = json.load(open(fp))
        for m in d.get('match_events', []):
            out.append(m)
    return out


def event_year(eid):
    # eid like '2024_masters_madrid'
    return int(eid.split('_')[0])


def build_intl_attendance(matches):
    """Map team -> set of intl event_ids attended."""
    intl_att = defaultdict(set)
    intl_att_dates = defaultdict(list)  # team -> list of (event_id, first_match_date)
    first_date_in_event = {}  # (team, event_id) -> earliest date
    for m in matches:
        if m['event_id'] not in INTL_EVENTS:
            continue
        for side in ('winner', 'loser'):
            t = m[side]
            eid = m['event_id']
            d = m['date']
            intl_att[t].add(eid)
            key = (t, eid)
            if key not in first_date_in_event or d < first_date_in_event[key]:
                first_date_in_event[key] = d
    # Build (team, year) -> list of (event_id, first_match_date) for prior-intl lookup
    team_year_intl = defaultdict(list)
    for (t, eid), d in first_date_in_event.items():
        y = event_year(eid)
        team_year_intl[(t, y)].append((eid, d))
    for k in team_year_intl:
        team_year_intl[k].sort(key=lambda x: x[1])
    return team_year_intl


def has_prior_intl_in_year(team, year, match_date, team_year_intl):
    for eid, d in team_year_intl.get((team, year), []):
        if d < match_date:
            return True
    return False


def build_predictions(matches, team_year_intl):
    rows = []
    # Build event match-position index (for group/playoff proxy)
    event_match_idx = defaultdict(list)
    for m in matches:
        event_match_idx[m['event_id']].append(m)
    for eid in event_match_idx:
        event_match_idx[eid].sort(key=lambda x: (x['date'], x['match_id']))

    # team-first-match-in-event lookup
    team_event_first = {}
    for eid, lst in event_match_idx.items():
        seen = set()
        for m in lst:
            for side in ('winner', 'loser'):
                t = m[side]
                key = (t, eid)
                if key not in seen:
                    seen.add(key)
                    team_event_first[(m['match_id'], t)] = True

    # match position fraction within event
    event_pos = {}
    for eid, lst in event_match_idx.items():
        n = len(lst)
        for i, m in enumerate(lst):
            event_pos[m['match_id']] = (i, n)

    for m in matches:
        wb, lb = m['winner_before'], m['loser_before']
        if wb == lb:
            continue
        # Optional: skip very early matches where both ratings are 0 (no signal)
        # Keep them — the model still produces a 0.5 baseline.

        # fav/dog
        if wb > lb:
            fav, dog = m['winner'], m['loser']
            fav_r, dog_r = wb, lb
            fav_won = 1
        else:
            fav, dog = m['loser'], m['winner']
            fav_r, dog_r = lb, wb
            fav_won = 0

        delta = abs(wb - lb)
        # baseline map logit from rating
        logit_map = BETA * delta  # fav perspective

        # Adjustments only valid at series logit, not map logit — but the
        # original model applies adjustments to the series logit per the spec.
        # We follow the spec: compute p_map from logit_map, then bo3/bo5
        # series prob, then convert to logit and add adjustments.
        p_map = sigmoid(logit_map)
        max_score = max(int(c) for c in m['series_score'].split('-'))
        is_bo5 = (max_score == 3)
        p_series_base = bo5_p_series(p_map) if is_bo5 else bo3_p_series(p_map)
        series_logit = logit(p_series_base)

        eid = m['event_id']
        is_intl = eid in INTL_EVENTS
        fav_reg = region_of(fav)
        dog_reg = region_of(dog)
        y = event_year(eid)
        match_date = m['date']

        intl_exp_diff = 0
        cn_dog_bump = 0
        if is_intl:
            fav_prior = has_prior_intl_in_year(fav, y, match_date, team_year_intl)
            dog_prior = has_prior_intl_in_year(dog, y, match_date, team_year_intl)
            intl_exp_diff = int(fav_prior) - int(dog_prior)
            if dog_reg == 'CN' and fav_reg != 'CN':
                cn_dog_bump = 1

        adj_logit = series_logit + INTL_EXP_COEF * intl_exp_diff + CN_DOG_BUMP * cn_dog_bump
        p_fav = sigmoid(adj_logit)

        rows.append({
            'match_id': m['match_id'],
            'date': match_date,
            'event_id': eid,
            'is_intl': is_intl,
            'is_bo5': is_bo5,
            'fav': fav, 'dog': dog,
            'fav_reg': fav_reg, 'dog_reg': dog_reg,
            'delta': delta,
            'p_fav': p_fav,
            'fav_won': fav_won,
            'cn_dog_bump': cn_dog_bump,
            'intl_exp_diff': intl_exp_diff,
            'event_pos_frac': event_pos[m['match_id']][0] / max(event_pos[m['match_id']][1] - 1, 1),
            'event_pos_n': event_pos[m['match_id']][1],
            'is_champions': eid in CHAMPIONS_EVENTS,
            'is_masters': eid in MASTERS_EVENTS,
        })
    return pd.DataFrame(rows)


def filter_pool(df):
    """2024 → 2026 Stage 1 — that's all rows except 2026_masters_santiago (if any future)."""
    # 2026 Stage 1 means up through 2026_stage1. We have 2026_kickoff, 2026_china_kickoff,
    # 2026_masters_santiago, 2026_stage1. We want everything except future events beyond stage1.
    # All four 2026 events are already historical (today is 2026-05-18), so include all.
    # Per spec the pool size is 1217. Let's match that.
    return df.copy()


def brier(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    return float(np.mean((p - y) ** 2))


def binom_ci(k, n, alpha=0.05):
    if n == 0:
        return (0.0, 1.0)
    # Wilson interval
    from math import sqrt
    z = 1.959963984540054
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    halfw = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return (max(0.0, center - halfw), min(1.0, center + halfw))


def lr_test(df_pool, mask, label):
    """Fit y ~ logit(p_base) + bucket_indicator and return LR p, delta_brier."""
    y = df_pool['fav_won'].values.astype(float)
    p = df_pool['p_fav'].values
    z = np.log(np.clip(p, 1e-9, 1 - 1e-9) / np.clip(1 - p, 1e-9, 1 - 1e-9))
    b = mask.astype(float).values

    # base: intercept + z (Platt-like)
    X0 = sm.add_constant(np.column_stack([z]))
    X1 = sm.add_constant(np.column_stack([z, b]))
    try:
        m0 = sm.Logit(y, X0).fit(disp=0, maxiter=200)
        m1 = sm.Logit(y, X1).fit(disp=0, maxiter=200)
    except Exception as e:
        return {'lr_p': float('nan'), 'beta': float('nan'),
                'brier_base': float('nan'), 'brier_adj': float('nan'),
                'delta_brier': float('nan'), 'err': str(e)}

    lr_stat = 2 * (m1.llf - m0.llf)
    from scipy.stats import chi2
    lr_p = 1 - chi2.cdf(lr_stat, df=1)

    p_base_adj = m0.predict(X0)
    p_full_adj = m1.predict(X1)
    return {
        'lr_p': float(lr_p),
        'beta': float(m1.params[2]),
        'brier_base': brier(y, p_base_adj),
        'brier_adj': brier(y, p_full_adj),
        'delta_brier': brier(y, p_base_adj) - brier(y, p_full_adj),
    }


def bucket_stats(sub):
    n = len(sub)
    if n == 0:
        return None
    p_mean = float(sub['p_fav'].mean())
    a_mean = float(sub['fav_won'].mean())
    k = int(sub['fav_won'].sum())
    ci = binom_ci(k, n)
    return {
        'n': n,
        'mean_pred': p_mean,
        'mean_actual': a_mean,
        'residual': a_mean - p_mean,
        'wins': k,
        'ci_low': ci[0], 'ci_high': ci[1],
        'brier_bucket': brier(sub['fav_won'].values, sub['p_fav'].values),
    }


def bootstrap_residual(df_pool, mask, reps=200, seed=42):
    """Cluster (by series/match_id) bootstrap — but each row IS a series, so just resample rows."""
    rng = np.random.default_rng(seed)
    sub = df_pool[mask]
    n = len(sub)
    if n < 5:
        return None
    deltas = []
    for _ in range(reps):
        idx = rng.integers(0, n, size=n)
        s = sub.iloc[idx]
        deltas.append(float(s['fav_won'].mean() - s['p_fav'].mean()))
    deltas.sort()
    lo = deltas[int(0.025 * reps)]
    hi = deltas[int(0.975 * reps) - 1]
    return {'boot_resid_lo': lo, 'boot_resid_hi': hi, 'reps': reps}


# ---- bucket definitions ----
def define_buckets(df):
    buckets = []

    # 1. Other "X-is-dog" by region at intl (cross-region only)
    intl = df['is_intl']
    cross = intl & (df['fav_reg'].notna()) & (df['dog_reg'].notna()) & (df['fav_reg'] != df['dog_reg'])
    for reg in ['Pacific', 'EMEA', 'Americas']:
        m = cross & (df['dog_reg'] == reg) & (df['fav_reg'] != 'CN')  # mirror of CN-dog
        buckets.append((f'1_intl_{reg}_dog_vs_nonCN_fav', m))
        # Also full version: dog == reg at intl regardless of fav region
        m2 = cross & (df['dog_reg'] == reg)
        buckets.append((f'1b_intl_{reg}_dog_anyfav', m2))

    # 2. CN-fav at intl (asymmetric counterpart)
    m = intl & (df['fav_reg'] == 'CN') & (df['dog_reg'] != 'CN') & df['dog_reg'].notna()
    buckets.append(('2_intl_CN_fav_vs_nonCN_dog', m))

    # 3. Per-CN team residual at intl (team appears on either side)
    cn_teams = [t for t, r in ORG_REGIONS.items() if r == 'CN']
    for t in cn_teams:
        m = intl & ((df['fav'] == t) | (df['dog'] == t))
        if m.sum() >= 8:  # skip tiny
            buckets.append((f'3_intl_CNteam_{t}', m))

    # 4. Per non-CN team facing CN at intl
    cross_cn = intl & (((df['fav_reg'] == 'CN') & (df['dog_reg'] != 'CN') & df['dog_reg'].notna()) |
                      ((df['dog_reg'] == 'CN') & (df['fav_reg'] != 'CN') & df['fav_reg'].notna()))
    facing_team = df.apply(lambda r: r['dog'] if r['fav_reg'] == 'CN' else r['fav'], axis=1)
    for t in sorted(set(facing_team[cross_cn])):
        m = cross_cn & (facing_team == t)
        if m.sum() >= 5:
            buckets.append((f'4_intl_nonCNteam_vs_CN_{t}', m))

    # 5. bo5 vs bo3 conditional on cross-region intl
    cross_intl = intl & (df['fav_reg'].notna()) & (df['dog_reg'].notna()) & (df['fav_reg'] != df['dog_reg'])
    buckets.append(('5_cross_intl_bo5', cross_intl & df['is_bo5']))
    buckets.append(('5_cross_intl_bo3', cross_intl & ~df['is_bo5']))

    # 6. Champions vs Masters vs non-intl
    buckets.append(('6_champions', df['is_champions']))
    buckets.append(('6_masters', df['is_masters']))
    buckets.append(('6_nonintl', ~intl))

    # 7. First match of the team at an event — proxy: event_pos_frac < 0.1
    buckets.append(('7_first10pct_event', df['event_pos_frac'] < 0.1))
    buckets.append(('7_last20pct_event', df['event_pos_frac'] > 0.8))

    # 8. Group vs playoffs proxy
    buckets.append(('8_group_first50pct', df['event_pos_frac'] < 0.5))
    buckets.append(('8_playoff_last50pct', df['event_pos_frac'] >= 0.5))
    buckets.append(('8_intl_group_first50pct', intl & (df['event_pos_frac'] < 0.5)))
    buckets.append(('8_intl_playoff_last50pct', intl & (df['event_pos_frac'] >= 0.5)))

    # 9. Cross-region intl pair (unordered)
    def pair_key(r):
        if pd.isna(r['fav_reg']) or pd.isna(r['dog_reg']):
            return None
        if r['fav_reg'] == r['dog_reg']:
            return None
        a, b = sorted([r['fav_reg'], r['dog_reg']])
        return f'{a}-vs-{b}'
    pkeys = df.apply(pair_key, axis=1)
    for pk in sorted({p for p in pkeys if isinstance(p, str)}):
        m = intl & (pkeys == pk)
        if m.sum() >= 15:
            buckets.append((f'9_intl_pair_{pk}', m))

    return buckets


def main():
    matches = load_matches()
    team_year_intl = build_intl_attendance(matches)
    df = build_predictions(matches, team_year_intl)
    df = filter_pool(df)
    print(f'Pool size: {len(df)} matches')
    print(f'Pool Brier: {brier(df.fav_won.values, df.p_fav.values):.4f}')
    print(f'Pool mean pred: {df.p_fav.mean():.4f}, mean actual: {df.fav_won.mean():.4f}')

    buckets = define_buckets(df)
    print(f'\nTesting {len(buckets)} buckets...\n')

    results = []
    for name, mask in buckets:
        sub = df[mask]
        stats = bucket_stats(sub)
        if stats is None or stats['n'] < 5:
            continue
        lr = lr_test(df, mask, name)
        rec = {'bucket': name, **stats, **lr}
        # bootstrap only on candidates
        if stats['n'] >= 15 and lr['lr_p'] < 0.05:
            boot = bootstrap_residual(df, mask, reps=200)
            if boot:
                rec.update(boot)
        results.append(rec)

    # Pass criteria
    BONF_P = 0.005
    MIN_N = 15
    MIN_DBRIER = 0.0010

    for r in results:
        r['passes_gate'] = (
            r['n'] >= MIN_N and
            r['lr_p'] < BONF_P and
            r['delta_brier'] >= MIN_DBRIER
        )

    results.sort(key=lambda r: (-r['delta_brier'], r['lr_p']))

    print(f'{"bucket":50s} {"n":>5s} {"pred":>6s} {"act":>6s} {"resid":>7s} {"lr_p":>9s} {"dBrier":>8s}  pass')
    print('-' * 110)
    for r in results:
        print(f'{r["bucket"][:50]:50s} {r["n"]:5d} {r["mean_pred"]:6.3f} {r["mean_actual"]:6.3f} '
              f'{r["residual"]:+7.3f} {r["lr_p"]:9.4f} {r["delta_brier"]:+8.4f}  '
              f'{"PASS" if r["passes_gate"] else ""}')

    passers = [r for r in results if r['passes_gate']]
    print(f'\nBuckets passing gate (n>={MIN_N}, lr_p<{BONF_P}, dBrier>={MIN_DBRIER}): {len(passers)}')
    for r in passers:
        print(f'  - {r["bucket"]}: n={r["n"]} resid={r["residual"]:+.3f} '
              f'lr_p={r["lr_p"]:.4g} dBrier={r["delta_brier"]:+.4f} beta={r["beta"]:+.3f}')

    if len(passers) == 0:
        verdict = 'no signal — current model is well-calibrated on tested sub-buckets'
    elif len(passers) == 1:
        verdict = f'1 thing to ship: {passers[0]["bucket"]}'
    else:
        verdict = f'{len(passers)} things to ship: ' + ', '.join(r['bucket'] for r in passers)
    print(f'\nVERDICT: {verdict}')

    out = {
        'pool_size': int(len(df)),
        'pool_brier': brier(df.fav_won.values, df.p_fav.values),
        'pool_mean_pred': float(df.p_fav.mean()),
        'pool_mean_actual': float(df.fav_won.mean()),
        'gate': {'min_n': MIN_N, 'bonferroni_p': BONF_P, 'min_delta_brier': MIN_DBRIER},
        'buckets': results,
        'passers': [r['bucket'] for r in passers],
        'verdict': verdict,
    }
    with open(OUT_JSON, 'w') as f:
        json.dump(out, f, indent=2, default=float)
    print(f'\nWrote {OUT_JSON}')


if __name__ == '__main__':
    main()
