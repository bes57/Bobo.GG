#!/usr/bin/env python3
"""
ResearchEventContext.py
-----------------------
Test whether event-context features improve BenPom's pre-match win-probability
predictions (baseline: p = sigmoid(0.154 * (winner_before - loser_before))).

Features tested:
  1. intl_exp_diff: signed {-1,0,1} based on whether each side has attended an
                    international event THIS calendar year already.
  2. first_match_diff: signed (fav first in event ? 1 : 0) - (und first ? 1 : 0).
  3. rest_diff: days since each side's prior match -> fav_rest - und_rest.
  4. match_idx_diff: # matches played within current event so far per side.
  5. is_intl: international event indicator.
  6. cross_region: 1 if favorite and underdog are from different regions.

Plus a special test:
  intl_interaction: outcome ~ logit(p_baseline) * is_intl
"""

import json
import os
import sys
import math
from datetime import datetime
from collections import defaultdict

import numpy as np

# Statsmodels for clean logistic regression + LR test
import statsmodels.api as sm
from scipy.stats import chi2, norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')

# ---- baseline calibration constant from project memory ----
BASELINE_BETA = 0.154

INTL_EVENTS = {
    '2024_masters_madrid', '2024_masters_shanghai', '2024_champions',
    '2025_masters_bangkok', '2025_masters_toronto', '2025_champions',
    '2026_masters_santiago',
}

# Static region map mirroring MapElo.ORG_REGIONS.  Inlined to avoid importing
# the Flask app (which has heavy side effects on import).
ORG_REGIONS = {
    "TL":"EMEA","FNC":"EMEA","NAVI":"EMEA","VIT":"EMEA","BBL":"EMEA","GX":"EMEA",
    "KC":"EMEA","TH":"EMEA","FUT":"EMEA","GIA":"EMEA","MKOI":"EMEA","M8":"EMEA",
    "PCF":"EMEA","ULF":"EMEA","EF":"EMEA",
    "SEN":"Americas","G2":"Americas","MIBR":"Americas","NRG":"Americas",
    "100T":"Americas","C9":"Americas","EG":"Americas","KRÜ":"Americas",
    "LEV":"Americas","FUR":"Americas","LOUD":"Americas","2G":"Americas",
    "APK":"Americas","ENVY":"Americas",
    "PRX":"Pacific","DRX":"Pacific","T1":"Pacific","TLN":"Pacific","GEN":"Pacific",
    "DFM":"Pacific","ZETA":"Pacific","RRQ":"Pacific","TS":"Pacific","GE":"Pacific",
    "NS":"Pacific","FS":"Pacific","VL":"Pacific","KRX":"Pacific","BME":"Pacific",
    "EDG":"CN","BLG":"CN","TE":"CN","DRG":"CN","ASE":"CN","AG":"CN","XLG":"CN",
    "WOL":"CN","FPX":"CN","JDG":"CN","NOVA":"CN","TEC":"CN","TYL":"CN","TYLOO":"CN",
}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def brier(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


def load_all_matches():
    """Load 2024, 2025, 2026 (skip 2023) and return a single chronological list."""
    files = [
        os.path.join(DATA, 'rating_timeline_2024.json'),
        os.path.join(DATA, 'rating_timeline_2025.json'),
        os.path.join(DATA, 'rating_timeline.json'),  # 2026
    ]
    matches = []
    for fp in files:
        d = json.load(open(fp))
        for m in d['match_events']:
            matches.append(m)
    # Sort by date then match_id for deterministic ordering
    matches.sort(key=lambda m: (m['date'], m['match_id']))
    return matches


def build_feature_table(matches):
    """For each series compute baseline p, outcome, and all candidate features.

    Walk through matches in chronological order so per-team running counters
    (last-match-date, intl-this-season, count-within-event) are correct.
    """
    last_match_date = {}           # org -> date string of last match
    intl_this_year = defaultdict(set)   # year (int) -> set of orgs that have played intl this year
    event_team_match_count = defaultdict(lambda: defaultdict(int))  # (event_id, org) -> count
    event_team_first_done = defaultdict(set)  # event_id -> set of orgs that have already played

    rows = []
    for m in matches:
        date_s = m['date']
        try:
            dt = datetime.strptime(date_s, '%Y-%m-%d')
        except Exception:
            continue
        year = dt.year
        event_id = m['event_id']
        w = m['winner']; l = m['loser']
        wb = float(m['winner_before']); lb = float(m['loser_before'])

        # ---- favorite/underdog by rating-before ----
        if wb == lb:
            # tie in rating: skip — undefined fav/und
            fav, und = w, l
            fav_b, und_b = wb, lb
            fav_won = 1
            tied = True
        else:
            tied = False
            if wb > lb:
                fav, und = w, l
                fav_b, und_b = wb, lb
                fav_won = 1
            else:
                fav, und = l, w
                fav_b, und_b = lb, wb
                fav_won = 0

        rating_diff = fav_b - und_b
        p_base = sigmoid(BASELINE_BETA * rating_diff)

        # ---- intl_exp_diff (BEFORE this match) ----
        fav_intl = 1 if fav in intl_this_year[year] else 0
        und_intl = 1 if und in intl_this_year[year] else 0
        intl_exp_diff = fav_intl - und_intl

        # ---- first_match_diff (within current event) ----
        fav_first = 0 if fav in event_team_first_done[event_id] else 1
        und_first = 0 if und in event_team_first_done[event_id] else 1
        first_match_diff = fav_first - und_first

        # ---- rest_diff ----
        def rest_for(org):
            if org not in last_match_date:
                return None
            try:
                prev = datetime.strptime(last_match_date[org], '%Y-%m-%d')
            except Exception:
                return None
            return (dt - prev).days

        fav_rest = rest_for(fav)
        und_rest = rest_for(und)
        if fav_rest is not None and und_rest is not None:
            rest_diff = float(fav_rest - und_rest)
        else:
            rest_diff = None

        # ---- match_idx_diff (matches in event already completed) ----
        fav_cnt = event_team_match_count[event_id][fav]
        und_cnt = event_team_match_count[event_id][und]
        match_idx_diff = float(fav_cnt - und_cnt)

        # ---- is_intl ----
        is_intl = 1 if event_id in INTL_EVENTS else 0

        # ---- cross_region ----
        fav_reg = ORG_REGIONS.get(fav)
        und_reg = ORG_REGIONS.get(und)
        if fav_reg is None or und_reg is None:
            cross_region = None
        else:
            cross_region = 1 if fav_reg != und_reg else 0

        rows.append({
            'match_id': m['match_id'],
            'date': date_s,
            'event_id': event_id,
            'fav': fav, 'und': und,
            'fav_b': fav_b, 'und_b': und_b,
            'rating_diff': rating_diff,
            'p_base': p_base,
            'y': fav_won,
            'tied': tied,
            'intl_exp_diff': intl_exp_diff,
            'first_match_diff': first_match_diff,
            'rest_diff': rest_diff,
            'match_idx_diff': match_idx_diff,
            'is_intl': is_intl,
            'cross_region': cross_region,
        })

        # ---- update counters AFTER recording the row (features are pre-match) ----
        last_match_date[w] = date_s
        last_match_date[l] = date_s
        event_team_match_count[event_id][w] += 1
        event_team_match_count[event_id][l] += 1
        event_team_first_done[event_id].add(w)
        event_team_first_done[event_id].add(l)
        if is_intl:
            intl_this_year[year].add(w)
            intl_this_year[year].add(l)

    return rows


def fit_logit(y, X, names):
    """Fit logistic regression. X must already include constant if wanted.

    Returns dict with params, ses, p_values, conf_int, loglike, predicted probs.
    """
    model = sm.Logit(y, X)
    res = model.fit(disp=False, method='newton', maxiter=100)
    ci = res.conf_int(alpha=0.05)
    out = {
        'params': dict(zip(names, res.params)),
        'bse':    dict(zip(names, res.bse)),
        'pvalues':dict(zip(names, res.pvalues)),
        'ci':     {names[i]: [float(ci[i][0]), float(ci[i][1])] for i in range(len(names))},
        'llf':    float(res.llf),
        'fitted': np.asarray(res.predict(X)),
        'res':    res,
    }
    return out


def main():
    matches = load_all_matches()
    rows_all = build_feature_table(matches)
    # Drop ties (rating_diff == 0) where favorite is undefined
    rows = [r for r in rows_all if not r['tied']]

    n = len(rows)
    y = np.array([r['y'] for r in rows], dtype=float)
    logit_p = np.array([math.log(r['p_base'] / (1 - r['p_base'])) for r in rows])
    p_base = np.array([r['p_base'] for r in rows])
    baseline_brier = brier(y, p_base)

    # ----- Baseline-only logistic (offset-style: re-fit intercept+slope on logit(p_base))
    # We use the BASELINE model = p_base directly (β fixed at 0.154). To get a clean
    # likelihood for LR tests, we fit the baseline as `outcome ~ logit_p_base` and
    # treat THAT as the null for the LR test. This is the standard approach for
    # testing incremental features.
    X_base = sm.add_constant(logit_p.reshape(-1, 1), has_constant='add')
    base_fit = fit_logit(y, X_base, ['const', 'logit_p'])
    llf_base = base_fit['llf']
    # Predicted probs from the re-fit baseline (used as fair LR-test null)
    p_base_refit = base_fit['fitted']
    base_refit_brier = brier(y, p_base_refit)

    features_to_test = [
        ('intl_exp_diff',    'intl_exp_diff'),
        ('first_match_diff', 'first_match_diff'),
        ('rest_diff',        'rest_diff'),
        ('match_idx_diff',   'match_idx_diff'),
        ('is_intl',          'is_intl'),
        ('cross_region',     'cross_region'),
    ]

    bonf_alpha = 0.05 / len(features_to_test)
    feature_results = []
    fit_cache = {}

    for fname, key in features_to_test:
        # subset rows where feature is defined
        sub = [r for r in rows if r[key] is not None]
        n_used = len(sub)
        y_s   = np.array([r['y'] for r in sub], dtype=float)
        lp_s  = np.array([math.log(r['p_base'] / (1 - r['p_base'])) for r in sub])
        feat  = np.array([float(r[key]) for r in sub])
        p_s   = sigmoid(lp_s)

        X_null = sm.add_constant(lp_s.reshape(-1, 1), has_constant='add')
        X_alt  = sm.add_constant(np.column_stack([lp_s, feat]), has_constant='add')
        null_fit = fit_logit(y_s, X_null, ['const', 'logit_p'])
        alt_fit  = fit_logit(y_s, X_alt,  ['const', 'logit_p', fname])

        coef = float(alt_fit['params'][fname])
        se   = float(alt_fit['bse'][fname])
        pval = float(alt_fit['pvalues'][fname])
        ci   = alt_fit['ci'][fname]

        # Brier comparisons (on the SAME subset, both refit)
        brier_null = brier(y_s, null_fit['fitted'])
        brier_alt  = brier(y_s, alt_fit['fitted'])

        # LR test
        lr_stat = 2 * (alt_fit['llf'] - null_fit['llf'])
        lr_p = 1 - chi2.cdf(lr_stat, df=1) if lr_stat > 0 else 1.0

        feature_results.append({
            'name': fname,
            'coef': coef,
            'se':   se,
            'p_value': pval,
            'ci':   [float(ci[0]), float(ci[1])],
            'brier_with_feature': brier_alt,
            'brier_improvement': brier_null - brier_alt,
            'lr_p': float(lr_p),
            'n_used': int(n_used),
            'significant_bonferroni': bool(pval < bonf_alpha),
        })
        fit_cache[fname] = (alt_fit, sub)

    # ----- Joint model with all features (rows where ALL features defined) -----
    joint_keys = [k for _, k in features_to_test]
    joint_rows = [r for r in rows if all(r[k] is not None for k in joint_keys)]
    n_joint = len(joint_rows)
    y_j  = np.array([r['y'] for r in joint_rows], dtype=float)
    lp_j = np.array([math.log(r['p_base'] / (1 - r['p_base'])) for r in joint_rows])
    feat_mat = np.column_stack([
        np.array([float(r[k]) for r in joint_rows]) for k in joint_keys
    ])
    X_joint_null = sm.add_constant(lp_j.reshape(-1, 1), has_constant='add')
    X_joint_full = sm.add_constant(np.column_stack([lp_j, feat_mat]), has_constant='add')
    joint_null_fit = fit_logit(y_j, X_joint_null, ['const', 'logit_p'])
    joint_full_fit = fit_logit(y_j, X_joint_full,
                               ['const', 'logit_p'] + [n for n, _ in features_to_test])

    joint_brier_null = brier(y_j, joint_null_fit['fitted'])
    joint_brier_full = brier(y_j, joint_full_fit['fitted'])

    retained = []
    for fname, _ in features_to_test:
        p = float(joint_full_fit['pvalues'][fname])
        if p < bonf_alpha:
            retained.append({'name': fname, 'coef': float(joint_full_fit['params'][fname]),
                             'p': p})

    # ----- intl_interaction: outcome ~ logit_p * is_intl (i.e. main effects + product) -----
    sub_ix = [r for r in rows if r['is_intl'] is not None]
    y_i  = np.array([r['y'] for r in sub_ix], dtype=float)
    lp_i = np.array([math.log(r['p_base'] / (1 - r['p_base'])) for r in sub_ix])
    isi  = np.array([float(r['is_intl']) for r in sub_ix])
    inter = lp_i * isi
    X_int_null = sm.add_constant(np.column_stack([lp_i, isi]), has_constant='add')
    X_int_full = sm.add_constant(np.column_stack([lp_i, isi, inter]), has_constant='add')
    int_null = fit_logit(y_i, X_int_null, ['const', 'logit_p', 'is_intl'])
    int_full = fit_logit(y_i, X_int_full, ['const', 'logit_p', 'is_intl', 'logit_p_x_is_intl'])
    int_coef = float(int_full['params']['logit_p_x_is_intl'])
    int_se   = float(int_full['bse']['logit_p_x_is_intl'])
    int_p    = float(int_full['pvalues']['logit_p_x_is_intl'])
    int_brier_null = brier(y_i, int_null['fitted'])
    int_brier_full = brier(y_i, int_full['fitted'])

    # ----- Verdict logic -----
    sig_count = sum(1 for f in feature_results if f['significant_bonferroni'])
    any_marginal = any(f['p_value'] < 0.05 for f in feature_results)
    best_brier_gain = max((f['brier_improvement'] for f in feature_results), default=0.0)

    if sig_count >= 1 and best_brier_gain > 0.002:
        verdict = "promising"
    elif any_marginal or best_brier_gain > 0.001:
        verdict = "marginal"
    else:
        verdict = "no signal"

    # Headline
    best = max(feature_results, key=lambda f: f['brier_improvement'])
    headline = (
        f"Best single feature: {best['name']} (coef={best['coef']:.3f}, p={best['p_value']:.3g}, "
        f"Brier {best['brier_improvement']:+.5f}); {sig_count}/{len(features_to_test)} pass Bonferroni "
        f"alpha={bonf_alpha:.4f}."
    )

    out = {
        'n_series': int(n),
        'baseline_brier': float(baseline_brier),
        'baseline_refit_brier': float(base_refit_brier),
        'baseline_beta_fixed': BASELINE_BETA,
        'bonferroni_alpha': bonf_alpha,
        'features': feature_results,
        'intl_interaction': {
            'coef': int_coef,
            'se':   int_se,
            'p':    int_p,
            'brier_improvement': float(int_brier_null - int_brier_full),
            'n_used': int(len(sub_ix)),
        },
        'joint_model': {
            'n_used': int(n_joint),
            'features_retained': retained,
            'joint_brier_improvement': float(joint_brier_null - joint_brier_full),
            'all_coefs': {n: float(joint_full_fit['params'][n]) for n, _ in features_to_test},
            'all_pvalues': {n: float(joint_full_fit['pvalues'][n]) for n, _ in features_to_test},
        },
        'headline': headline,
        'verdict': verdict,
    }

    out_path = os.path.join(DATA, 'projection_research_event_context.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)

    # Pretty print to stdout for the caller
    print(f"n_series: {n}")
    print(f"baseline brier (fixed beta={BASELINE_BETA}): {baseline_brier:.5f}")
    print(f"baseline brier (refit intercept+slope):     {base_refit_brier:.5f}")
    print(f"Bonferroni alpha: {bonf_alpha:.4f}")
    print()
    print(f"{'feature':<22} {'coef':>8} {'se':>7} {'p':>9} {'dBrier':>10} {'n':>5} {'BF*'}")
    for f_ in feature_results:
        star = '*' if f_['significant_bonferroni'] else ''
        print(f"{f_['name']:<22} {f_['coef']:>+8.4f} {f_['se']:>7.4f} {f_['p_value']:>9.4g} "
              f"{f_['brier_improvement']:>+10.5f} {f_['n_used']:>5d}  {star}")
    print()
    print(f"intl x logit interaction: coef={int_coef:+.4f} se={int_se:.4f} p={int_p:.4g} "
          f"dBrier={int_brier_null - int_brier_full:+.5f}")
    print()
    print(f"Joint model n={n_joint} dBrier={joint_brier_null - joint_brier_full:+.5f} "
          f"retained @ BF: {[r['name'] for r in retained]}")
    print()
    print(headline)
    print(f"verdict: {verdict}")
    print(f"wrote {out_path}")


if __name__ == '__main__':
    main()
