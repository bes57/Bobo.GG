#!/usr/bin/env python3
"""
Research2_Interactions.py
-------------------------
Phase-2 research: test whether INTERACTION terms and REGIME-SPECIFIC corrections
can lower series Brier below the optimized additive baseline:

    p = sigmoid(0.154 * (winner_before - loser_before) + 0.36 * intl_exp_diff_signed)

Interactions tested:
  1. rating_diff x format (bo3 vs bo5)
  2. rating_diff x cross_region
  3. rating_diff x season (2024 / 2025 / 2026)
  4. rating_diff x event_type (kickoff / stage / masters / champions)
  5. intl_exp_diff x cross_region
  6. region-pair specific betas (rating_diff per (fav_region, und_region) pair)
  7. magnitude-asymmetric (|d|<1.5 vs |d|>=1.5)
  8. time-decayed (sample-weighted) MLE
  9. region-conditional intl_exp_diff

Methodology:
  - Chronological 75 / 25 holdout (older = train, newer = test).
  - Coefficients fit on TRAIN, refit baseline + alt; report TEST Brier and Platt b.
  - Bootstrap CI on the interaction coefficient (resample series w/ replacement, 200 reps).
  - LR test vs baseline on TRAIN (test of in-sample fit improvement).
  - Bonferroni alpha = 0.05 / 9 = 0.00556.

Ship criteria:
  - Test Brier improvement >= 0.0008
  - LR p < 0.00556
  - Bootstrap 95% CI on coefficient excludes 0
  - Effect direction physically sensible.
"""

import json
import math
import os
import random
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
import statsmodels.api as sm
from scipy.stats import chi2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')

# -- Baseline calibration (current shipping numbers) ------------------------
BASELINE_BETA = 0.154            # coef on rating_diff
BASELINE_INTL_BONUS = 0.36       # additive logit bonus per +1 of intl_exp_diff

INTL_EVENTS = {
    '2024_masters_madrid', '2024_masters_shanghai', '2024_champions',
    '2025_masters_bangkok', '2025_masters_toronto', '2025_champions',
    '2026_masters_santiago',
}

# Region map -- copy from ResearchEventContext.py to avoid Flask import side effects
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


# ── helpers ────────────────────────────────────────────────────────────────
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def brier(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


def fit_platt(y, p):
    """Fit Platt 'a + b*logit(p)' and return (a, b). Returns (0,1) on failure."""
    try:
        lp = logit(p)
        X = sm.add_constant(lp.reshape(-1, 1), has_constant='add')
        res = sm.Logit(y.astype(float), X).fit(disp=False, method='newton', maxiter=100)
        a = float(res.params[0]); b = float(res.params[1])
        return a, b
    except Exception:
        return 0.0, 1.0


def event_type_of(eid):
    if 'kickoff' in eid:    return 'kickoff'
    if 'masters' in eid:    return 'masters'
    if 'champions' in eid:  return 'champions'
    if 'stage' in eid:      return 'stage'
    if 'lock_in' in eid:    return 'lock_in'
    if 'league' in eid:     return 'league'
    return 'other'


def format_of(series_score):
    """Infer bo3 vs bo5 from final series score."""
    try:
        a, b = series_score.split('-')
        a, b = int(a), int(b)
    except Exception:
        return 'unknown'
    total = a + b
    winner_score = max(a, b)
    if winner_score == 3:
        return 'bo5'
    if winner_score == 2:
        return 'bo3'
    return f'bo{2*winner_score - 1}'


# ── load + build features ──────────────────────────────────────────────────
def load_all_matches():
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
    matches.sort(key=lambda m: (m['date'], m['match_id']))
    return matches


def build_rows(matches):
    """Walk chronologically; produce one row per series with pre-match features.

    Drops series where winner_before == loser_before (model had no opinion).
    """
    intl_this_year = defaultdict(set)  # year -> set of orgs that already played intl this year
    rows = []
    for m in matches:
        date_s = m['date']
        try:
            dt = datetime.strptime(date_s, '%Y-%m-%d')
        except Exception:
            continue
        year = dt.year
        event_id = m['event_id']
        w, l = m['winner'], m['loser']
        wb, lb = float(m['winner_before']), float(m['loser_before'])
        if wb == lb:
            # no opinion -> skip
            if event_id in INTL_EVENTS:
                intl_this_year[year].add(w); intl_this_year[year].add(l)
            continue

        if wb > lb:
            fav, und, fav_b, und_b, fav_won = w, l, wb, lb, 1
        else:
            fav, und, fav_b, und_b, fav_won = l, w, lb, wb, 0
        rating_diff = fav_b - und_b

        fav_intl = 1 if fav in intl_this_year[year] else 0
        und_intl = 1 if und in intl_this_year[year] else 0
        intl_exp_diff = fav_intl - und_intl

        fav_reg = ORG_REGIONS.get(fav)
        und_reg = ORG_REGIONS.get(und)
        cross_region = None
        if fav_reg and und_reg:
            cross_region = 1 if fav_reg != und_reg else 0

        is_intl = 1 if event_id in INTL_EVENTS else 0
        fmt = format_of(m.get('series_score', ''))

        rows.append({
            'match_id': m['match_id'],
            'date': date_s,
            'date_ord': dt.toordinal(),
            'year': year,
            'event_id': event_id,
            'event_type': event_type_of(event_id),
            'fav': fav, 'und': und,
            'fav_region': fav_reg, 'und_region': und_reg,
            'rating_diff': rating_diff,
            'intl_exp_diff': intl_exp_diff,
            'cross_region': cross_region,
            'is_intl': is_intl,
            'format': fmt,
            'y': fav_won,
        })

        # update intl experience AFTER recording
        if is_intl:
            intl_this_year[year].add(w); intl_this_year[year].add(l)

    return rows


# ── baseline prediction (current shipping model) ──────────────────────────
def baseline_logit(r):
    # Note: intl_exp_diff already signed from fav perspective
    return (BASELINE_BETA * r['rating_diff']
            + BASELINE_INTL_BONUS * r['intl_exp_diff'])


# ── core LR + bootstrap utilities ──────────────────────────────────────────
def fit_logit_safe(y, X):
    try:
        return sm.Logit(y, X).fit(disp=False, method='newton', maxiter=100)
    except Exception:
        try:
            return sm.Logit(y, X).fit(disp=False, method='bfgs', maxiter=200)
        except Exception:
            return None


def bootstrap_ci_coef(rows, build_xy_fn, coef_name, n_boot=200, seed=42):
    """Resample rows w/ replacement, refit on resampled, collect coef_name. Return (lo, hi)."""
    rng = random.Random(seed)
    coefs = []
    n = len(rows)
    for b in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        sub = [rows[i] for i in idx]
        y, X, names = build_xy_fn(sub)
        if X.shape[1] != len(names):
            continue
        res = fit_logit_safe(y.astype(float), X)
        if res is None:
            continue
        if coef_name in names:
            coefs.append(float(res.params[names.index(coef_name)]))
    if len(coefs) < 20:
        return None, None
    lo = float(np.percentile(coefs, 2.5))
    hi = float(np.percentile(coefs, 97.5))
    return lo, hi


# ── test/train split ──────────────────────────────────────────────────────
def chrono_split(rows, train_frac=0.75):
    rs = sorted(rows, key=lambda r: (r['date_ord'], r['match_id']))
    cut = int(len(rs) * train_frac)
    return rs[:cut], rs[cut:]


# ── per-interaction evaluation ────────────────────────────────────────────
def eval_interaction(name, train, test, build_xy_fn_train, build_xy_fn_test,
                     extra_coef_names, alt_coef_names_for_ci,
                     baseline_xy_fn_train, baseline_xy_fn_test,
                     direction_note=""):
    """Fit alt model on TRAIN, evaluate on TEST, run LR vs baseline, bootstrap CI.

    extra_coef_names: list of new coef names beyond the baseline columns.
    alt_coef_names_for_ci: which coef(s) to bootstrap (we summarise as 1 number,
        using the first; full list reported in notes).
    """
    # ---- Baseline (refit) on TRAIN ----
    y_tr, X_b_tr, names_b = baseline_xy_fn_train(train)
    res_b = fit_logit_safe(y_tr.astype(float), X_b_tr)
    if res_b is None:
        return None
    llf_b = float(res_b.llf)

    # ---- Alt on TRAIN ----
    y_tr2, X_a_tr, names_a = build_xy_fn_train(train)
    res_a = fit_logit_safe(y_tr2.astype(float), X_a_tr)
    if res_a is None:
        return None
    llf_a = float(res_a.llf)

    # LR test (df = number of new columns)
    df_diff = len(names_a) - len(names_b)
    lr_stat = 2 * (llf_a - llf_b)
    lr_p = float(1 - chi2.cdf(lr_stat, df=df_diff)) if lr_stat > 0 and df_diff > 0 else 1.0

    # ---- TEST evaluation ----
    y_te, X_b_te, _ = baseline_xy_fn_test(test)
    y_te2, X_a_te, _ = build_xy_fn_test(test)
    p_b_te = np.asarray(res_b.predict(X_b_te))
    p_a_te = np.asarray(res_a.predict(X_a_te))
    brier_b_te = brier(y_te, p_b_te)
    brier_a_te = brier(y_te2, p_a_te)
    a_pl, b_pl = fit_platt(y_te2, p_a_te)

    # ---- coefficient details ----
    coef_info = {}
    for cn in extra_coef_names:
        if cn in names_a:
            i = names_a.index(cn)
            coef_info[cn] = {
                'coef': float(res_a.params[i]),
                'se': float(res_a.bse[i]),
                'p': float(res_a.pvalues[i]),
            }

    # ---- bootstrap CI on the primary coef ----
    primary = alt_coef_names_for_ci[0]
    lo, hi = bootstrap_ci_coef(train, build_xy_fn_train, primary, n_boot=200, seed=137)

    primary_coef = coef_info.get(primary, {}).get('coef', float('nan'))
    primary_p = coef_info.get(primary, {}).get('p', float('nan'))

    return {
        'name': name,
        'coef': primary_coef,
        'coef_name': primary,
        'p': primary_p,
        'lr_p': lr_p,
        'lr_stat': float(lr_stat),
        'df': int(df_diff),
        'test_brier': brier_a_te,
        'baseline_test_brier': brier_b_te,
        'delta_test_brier': brier_b_te - brier_a_te,  # positive = improvement
        'test_platt_b': b_pl,
        'test_platt_a': a_pl,
        'n_train': int(len(train)),
        'n_test': int(len(test)),
        'all_coefs': coef_info,
        'bootstrap_ci_coef': [lo, hi],
        'notes': direction_note,
    }


# ── helpers to build X-matrices for the various designs ───────────────────
def build_baseline_xy(rows):
    """Baseline columns: [const, rating_diff, intl_exp_diff]."""
    y = np.array([r['y'] for r in rows], dtype=float)
    rd = np.array([r['rating_diff'] for r in rows], dtype=float)
    ix = np.array([r['intl_exp_diff'] for r in rows], dtype=float)
    X = sm.add_constant(np.column_stack([rd, ix]), has_constant='add')
    return y, X, ['const', 'rating_diff', 'intl_exp_diff']


def make_filter(predicate):
    def f(rows):
        return [r for r in rows if predicate(r)]
    return f


# Each interaction is described by a builder for the ALT design.
# All builders include rating_diff + intl_exp_diff baseline columns.

def build_format_xy(rows):
    """rating_diff + intl_exp_diff + is_bo5 + rating_diff*is_bo5  (drops 'unknown' fmt)."""
    rs = [r for r in rows if r['format'] in ('bo3', 'bo5')]
    y = np.array([r['y'] for r in rs], dtype=float)
    rd = np.array([r['rating_diff'] for r in rs], dtype=float)
    ix = np.array([r['intl_exp_diff'] for r in rs], dtype=float)
    bo5 = np.array([1.0 if r['format'] == 'bo5' else 0.0 for r in rs])
    inter = rd * bo5
    X = sm.add_constant(np.column_stack([rd, ix, bo5, inter]), has_constant='add')
    return y, X, ['const', 'rating_diff', 'intl_exp_diff', 'is_bo5', 'rd_x_bo5']


def build_format_baseline_xy(rows):
    rs = [r for r in rows if r['format'] in ('bo3', 'bo5')]
    return build_baseline_xy(rs)


def build_crossreg_xy(rows):
    rs = [r for r in rows if r['cross_region'] is not None]
    y = np.array([r['y'] for r in rs], dtype=float)
    rd = np.array([r['rating_diff'] for r in rs], dtype=float)
    ix = np.array([r['intl_exp_diff'] for r in rs], dtype=float)
    cr = np.array([float(r['cross_region']) for r in rs])
    inter = rd * cr
    X = sm.add_constant(np.column_stack([rd, ix, cr, inter]), has_constant='add')
    return y, X, ['const', 'rating_diff', 'intl_exp_diff', 'cross_region', 'rd_x_cross']


def build_crossreg_baseline_xy(rows):
    rs = [r for r in rows if r['cross_region'] is not None]
    return build_baseline_xy(rs)


def build_season_xy(rows):
    """rating_diff + intl_exp_diff + year dummies + rd * year dummies (2026 ref)."""
    rs = list(rows)
    y = np.array([r['y'] for r in rs], dtype=float)
    rd = np.array([r['rating_diff'] for r in rs], dtype=float)
    ix = np.array([r['intl_exp_diff'] for r in rs], dtype=float)
    y2024 = np.array([1.0 if r['year'] == 2024 else 0.0 for r in rs])
    y2025 = np.array([1.0 if r['year'] == 2025 else 0.0 for r in rs])
    rd24 = rd * y2024
    rd25 = rd * y2025
    X = sm.add_constant(np.column_stack([rd, ix, y2024, y2025, rd24, rd25]), has_constant='add')
    return y, X, ['const', 'rating_diff', 'intl_exp_diff', 'is_2024', 'is_2025',
                  'rd_x_2024', 'rd_x_2025']


def build_event_type_xy(rows):
    """rating_diff + intl_exp_diff + dummies + rd*dummies. Reference = 'stage'."""
    rs = list(rows)
    y = np.array([r['y'] for r in rs], dtype=float)
    rd = np.array([r['rating_diff'] for r in rs], dtype=float)
    ix = np.array([r['intl_exp_diff'] for r in rs], dtype=float)
    types = ['kickoff', 'masters', 'champions']
    cols = [rd, ix]
    names = ['rating_diff', 'intl_exp_diff']
    for t in types:
        d = np.array([1.0 if r['event_type'] == t else 0.0 for r in rs])
        cols.append(d); names.append(f'is_{t}')
    for t in types:
        d = np.array([1.0 if r['event_type'] == t else 0.0 for r in rs])
        cols.append(rd * d); names.append(f'rd_x_{t}')
    X = sm.add_constant(np.column_stack(cols), has_constant='add')
    return y, X, ['const'] + names


def build_intl_x_cross_xy(rows):
    """rating_diff + intl_exp_diff + cross_region + intl_exp_diff * cross_region."""
    rs = [r for r in rows if r['cross_region'] is not None]
    y = np.array([r['y'] for r in rs], dtype=float)
    rd = np.array([r['rating_diff'] for r in rs], dtype=float)
    ix = np.array([r['intl_exp_diff'] for r in rs], dtype=float)
    cr = np.array([float(r['cross_region']) for r in rs])
    inter = ix * cr
    X = sm.add_constant(np.column_stack([rd, ix, cr, inter]), has_constant='add')
    return y, X, ['const', 'rating_diff', 'intl_exp_diff', 'cross_region', 'ix_x_cross']


REGION_LIST = ['Americas', 'EMEA', 'Pacific', 'CN']


def build_regionpair_xy(rows):
    """rating_diff main + per (fav_region, und_region) ordered pair interaction
    (only cross_region pairs; same-region uses the main rd coef).

    Adds rd * indicator for each ordered pair, dropping one as reference.
    """
    rs = [r for r in rows if r['fav_region'] in REGION_LIST and r['und_region'] in REGION_LIST]
    y = np.array([r['y'] for r in rs], dtype=float)
    rd = np.array([r['rating_diff'] for r in rs], dtype=float)
    ix = np.array([r['intl_exp_diff'] for r in rs], dtype=float)
    cols = [rd, ix]
    names = ['rating_diff', 'intl_exp_diff']
    # ordered cross-region pairs, skip Americas->EMEA as reference
    ordered_pairs = [(a, b) for a in REGION_LIST for b in REGION_LIST if a != b]
    ref = ('Americas', 'EMEA')
    for fr, ur in ordered_pairs:
        if (fr, ur) == ref:
            continue
        d = np.array([1.0 if (r['fav_region'] == fr and r['und_region'] == ur) else 0.0
                      for r in rs])
        cols.append(d * rd)
        names.append(f'rd_x_{fr}_v_{ur}')
    X = sm.add_constant(np.column_stack(cols), has_constant='add')
    return y, X, ['const'] + names


def build_regionpair_baseline_xy(rows):
    rs = [r for r in rows if r['fav_region'] in REGION_LIST and r['und_region'] in REGION_LIST]
    return build_baseline_xy(rs)


def build_magnitude_xy(rows):
    """rating_diff + intl_exp_diff + |d|>=1.5 + rd*(|d|>=1.5)."""
    rs = list(rows)
    y = np.array([r['y'] for r in rs], dtype=float)
    rd = np.array([r['rating_diff'] for r in rs], dtype=float)
    ix = np.array([r['intl_exp_diff'] for r in rs], dtype=float)
    big = (np.abs(rd) >= 1.5).astype(float)
    inter = rd * big
    X = sm.add_constant(np.column_stack([rd, ix, big, inter]), has_constant='add')
    return y, X, ['const', 'rating_diff', 'intl_exp_diff', 'is_big_gap', 'rd_x_big']


def build_region_intl_xy(rows):
    """rating_diff + intl_exp_diff + per-region intl_exp_diff interaction.

    Uses CN, Pacific, EMEA interactions with Americas as reference.
    Region indicator is fav region.
    """
    rs = [r for r in rows if r['fav_region'] in REGION_LIST]
    y = np.array([r['y'] for r in rs], dtype=float)
    rd = np.array([r['rating_diff'] for r in rs], dtype=float)
    ix = np.array([r['intl_exp_diff'] for r in rs], dtype=float)
    cols = [rd, ix]
    names = ['rating_diff', 'intl_exp_diff']
    for region in ['EMEA', 'Pacific', 'CN']:
        d = np.array([1.0 if r['fav_region'] == region else 0.0 for r in rs])
        cols.append(ix * d)
        names.append(f'ix_x_fav_{region}')
    X = sm.add_constant(np.column_stack(cols), has_constant='add')
    return y, X, ['const'] + names


def build_region_intl_baseline_xy(rows):
    rs = [r for r in rows if r['fav_region'] in REGION_LIST]
    return build_baseline_xy(rs)


# ── Time-decay weighted MLE: special-case test ─────────────────────────────
def time_decay_fit_and_eval(rows_train, rows_test, half_life_days):
    """Refit baseline (rating_diff + intl_exp_diff) with exp(-lambda*days_old) weights.
    Returns dict comparable to other entries; uses train weighted Brier improvement
    on test and a 'pseudo' LR using weighted log-lik.
    """
    if not rows_train or not rows_test:
        return None
    # Anchor 'now' at the LAST date in train; weights = exp(-ln2 * days_old / half_life)
    now = max(r['date_ord'] for r in rows_train)
    lam = math.log(2.0) / half_life_days
    weights = np.array([math.exp(-lam * (now - r['date_ord'])) for r in rows_train])
    y_tr = np.array([r['y'] for r in rows_train], dtype=float)
    rd_tr = np.array([r['rating_diff'] for r in rows_train], dtype=float)
    ix_tr = np.array([r['intl_exp_diff'] for r in rows_train], dtype=float)
    X_tr = sm.add_constant(np.column_stack([rd_tr, ix_tr]), has_constant='add')

    try:
        res = sm.GLM(y_tr, X_tr,
                     family=sm.families.Binomial(),
                     freq_weights=weights).fit()
    except Exception:
        return None
    y_te = np.array([r['y'] for r in rows_test], dtype=float)
    rd_te = np.array([r['rating_diff'] for r in rows_test], dtype=float)
    ix_te = np.array([r['intl_exp_diff'] for r in rows_test], dtype=float)
    X_te = sm.add_constant(np.column_stack([rd_te, ix_te]), has_constant='add')
    p_te = np.asarray(res.predict(X_te))
    a_pl, b_pl = fit_platt(y_te, p_te)
    return {
        'half_life_days': half_life_days,
        'rating_diff_coef': float(res.params[1]),
        'intl_exp_diff_coef': float(res.params[2]),
        'test_brier': brier(y_te, p_te),
        'test_platt_b': b_pl,
        'test_platt_a': a_pl,
        'n_train': int(len(rows_train)),
        'n_test': int(len(rows_test)),
    }


# ── MAIN ────────────────────────────────────────────────────────────────────
def main():
    matches = load_all_matches()
    rows = build_rows(matches)
    n_total = len(rows)
    print(f"Total series with non-zero rating_diff: {n_total}")

    train, test = chrono_split(rows, train_frac=0.75)
    n_tr = len(train); n_te = len(test)
    print(f"Chrono split: train={n_tr}  test={n_te}")
    print(f"Train dates: {train[0]['date']} .. {train[-1]['date']}")
    print(f"Test  dates: {test[0]['date']} .. {test[-1]['date']}")

    # ---- Baseline on test (refit on train) -----------------------------------
    y_tr, X_b_tr, _ = build_baseline_xy(train)
    res_b = fit_logit_safe(y_tr.astype(float), X_b_tr)
    y_te, X_b_te, _ = build_baseline_xy(test)
    p_b_te = np.asarray(res_b.predict(X_b_te))
    baseline_test_brier = brier(y_te, p_b_te)
    baseline_test_platt_a, baseline_test_platt_b = fit_platt(y_te, p_b_te)
    print(f"Baseline TEST Brier: {baseline_test_brier:.5f}  Platt b: {baseline_test_platt_b:.3f}")
    print(f"Baseline coefs: const={res_b.params[0]:+.3f}  "
          f"rating_diff={res_b.params[1]:+.3f}  intl_exp_diff={res_b.params[2]:+.3f}")

    BONF = 0.05 / 9.0
    print(f"Bonferroni alpha: {BONF:.5f}\n")

    interactions = []

    # ---- 1. rating_diff × format (bo3 vs bo5) ----
    print("=" * 60)
    print("[1] rating_diff × format (bo3 vs bo5)")
    r = eval_interaction(
        'rd_x_format(bo5)', train, test,
        build_format_xy, build_format_xy,
        extra_coef_names=['is_bo5', 'rd_x_bo5'],
        alt_coef_names_for_ci=['rd_x_bo5'],
        baseline_xy_fn_train=build_format_baseline_xy,
        baseline_xy_fn_test=build_format_baseline_xy,
        direction_note="Hypothesis: bo5 amplifies rating gaps (rd_x_bo5 > 0)."
    )
    if r:
        interactions.append(r)
        print(f"  coef={r['coef']:+.4f}  p={r['p']:.4g}  LR p={r['lr_p']:.4g}  "
              f"Δtest Brier={r['delta_test_brier']:+.5f}  bootCI={r['bootstrap_ci_coef']}")

    # ---- 2. rating_diff × cross_region ----
    print("\n[2] rating_diff × cross_region")
    r = eval_interaction(
        'rd_x_cross_region', train, test,
        build_crossreg_xy, build_crossreg_xy,
        extra_coef_names=['cross_region', 'rd_x_cross'],
        alt_coef_names_for_ci=['rd_x_cross'],
        baseline_xy_fn_train=build_crossreg_baseline_xy,
        baseline_xy_fn_test=build_crossreg_baseline_xy,
        direction_note="Hypothesis: bridge-solve noise -> rd_x_cross < 0."
    )
    if r:
        interactions.append(r)
        print(f"  coef={r['coef']:+.4f}  p={r['p']:.4g}  LR p={r['lr_p']:.4g}  "
              f"Δtest Brier={r['delta_test_brier']:+.5f}  bootCI={r['bootstrap_ci_coef']}")

    # ---- 3. rating_diff × season ----
    print("\n[3] rating_diff × season (per-year β)")
    r = eval_interaction(
        'rd_x_season', train, test,
        build_season_xy, build_season_xy,
        extra_coef_names=['rd_x_2024', 'rd_x_2025'],
        alt_coef_names_for_ci=['rd_x_2024'],
        baseline_xy_fn_train=build_baseline_xy,
        baseline_xy_fn_test=build_baseline_xy,
        direction_note="Tests per-season β (df=4). Primary coef = rd_x_2024."
    )
    if r:
        interactions.append(r)
        print(f"  primary coef={r['coef']:+.4f}  p={r['p']:.4g}  LR p={r['lr_p']:.4g}  "
              f"Δtest Brier={r['delta_test_brier']:+.5f}  bootCI={r['bootstrap_ci_coef']}")
        print(f"  all coefs: {r['all_coefs']}")

    # ---- 4. rating_diff × event_type ----
    print("\n[4] rating_diff × event_type")
    r = eval_interaction(
        'rd_x_event_type', train, test,
        build_event_type_xy, build_event_type_xy,
        extra_coef_names=['rd_x_kickoff', 'rd_x_masters', 'rd_x_champions'],
        alt_coef_names_for_ci=['rd_x_champions'],
        baseline_xy_fn_train=build_baseline_xy,
        baseline_xy_fn_test=build_baseline_xy,
        direction_note="Per-event-type β (ref=stage). df=6."
    )
    if r:
        interactions.append(r)
        print(f"  primary coef={r['coef']:+.4f}  p={r['p']:.4g}  LR p={r['lr_p']:.4g}  "
              f"Δtest Brier={r['delta_test_brier']:+.5f}  bootCI={r['bootstrap_ci_coef']}")
        print(f"  all coefs: {r['all_coefs']}")

    # ---- 5. intl_exp_diff × cross_region ----
    print("\n[5] intl_exp_diff × cross_region")
    r = eval_interaction(
        'ix_x_cross_region', train, test,
        build_intl_x_cross_xy, build_intl_x_cross_xy,
        extra_coef_names=['cross_region', 'ix_x_cross'],
        alt_coef_names_for_ci=['ix_x_cross'],
        baseline_xy_fn_train=build_crossreg_baseline_xy,
        baseline_xy_fn_test=build_crossreg_baseline_xy,
        direction_note="Hypothesis: intl-exp bonus fires hardest on cross-region (>0)."
    )
    if r:
        interactions.append(r)
        print(f"  coef={r['coef']:+.4f}  p={r['p']:.4g}  LR p={r['lr_p']:.4g}  "
              f"Δtest Brier={r['delta_test_brier']:+.5f}  bootCI={r['bootstrap_ci_coef']}")

    # ---- 6. region-pair specific β ----
    print("\n[6] rating_diff × region pair (full ordered matrix)")
    r = eval_interaction(
        'rd_x_region_pair', train, test,
        build_regionpair_xy, build_regionpair_xy,
        extra_coef_names=[f'rd_x_{a}_v_{b}'
                          for a in REGION_LIST for b in REGION_LIST
                          if a != b and (a, b) != ('Americas', 'EMEA')],
        alt_coef_names_for_ci=['rd_x_Pacific_v_Americas'],
        baseline_xy_fn_train=build_regionpair_baseline_xy,
        baseline_xy_fn_test=build_regionpair_baseline_xy,
        direction_note="11 ordered cross-region pairs vs Americas-vs-EMEA reference."
    )
    if r:
        interactions.append(r)
        print(f"  primary coef={r['coef']:+.4f}  p={r['p']:.4g}  LR p={r['lr_p']:.4g}  "
              f"Δtest Brier={r['delta_test_brier']:+.5f}")
        for k, v in r['all_coefs'].items():
            mark = "  *" if v['p'] < 0.05 else ""
            print(f"    {k:>32}: coef={v['coef']:+.4f}  p={v['p']:.3g}{mark}")

    # ---- 7. magnitude-asymmetric ----
    print("\n[7] rating_diff × |Δ|≥1.5 (blowout vs toss-up)")
    r = eval_interaction(
        'rd_x_big_gap', train, test,
        build_magnitude_xy, build_magnitude_xy,
        extra_coef_names=['is_big_gap', 'rd_x_big'],
        alt_coef_names_for_ci=['rd_x_big'],
        baseline_xy_fn_train=build_baseline_xy,
        baseline_xy_fn_test=build_baseline_xy,
        direction_note="If coef<0, blowouts shrink toward 0.5 (need smaller β)."
    )
    if r:
        interactions.append(r)
        print(f"  coef={r['coef']:+.4f}  p={r['p']:.4g}  LR p={r['lr_p']:.4g}  "
              f"Δtest Brier={r['delta_test_brier']:+.5f}  bootCI={r['bootstrap_ci_coef']}")

    # ---- 8. Time-decayed MLE ----
    print("\n[8] Time-decayed MLE (sample weights)")
    td_results = []
    best_td = None
    for hl_years in [0.5, 1.0, 2.0]:
        td = time_decay_fit_and_eval(train, test, half_life_days=hl_years * 365.25)
        if td:
            td['half_life_years'] = hl_years
            td_results.append(td)
            print(f"  half-life={hl_years}y  rd_coef={td['rating_diff_coef']:+.3f}  "
                  f"ix_coef={td['intl_exp_diff_coef']:+.3f}  "
                  f"test Brier={td['test_brier']:.5f}  Platt b={td['test_platt_b']:.3f}")
            if best_td is None or td['test_brier'] < best_td['test_brier']:
                best_td = td
    if best_td:
        delta = baseline_test_brier - best_td['test_brier']
        interactions.append({
            'name': f'time_decay_{best_td["half_life_years"]}y',
            'coef': float(best_td['rating_diff_coef']),
            'coef_name': 'rating_diff_weighted',
            'p': float('nan'),
            'lr_p': float('nan'),
            'lr_stat': float('nan'),
            'df': 0,
            'test_brier': float(best_td['test_brier']),
            'baseline_test_brier': float(baseline_test_brier),
            'delta_test_brier': float(delta),
            'test_platt_b': float(best_td['test_platt_b']),
            'test_platt_a': float(best_td['test_platt_a']),
            'n_train': int(best_td['n_train']),
            'n_test': int(best_td['n_test']),
            'all_coefs': {'rd': {'coef': best_td['rating_diff_coef']},
                          'ix': {'coef': best_td['intl_exp_diff_coef']}},
            'bootstrap_ci_coef': [None, None],
            'notes': (f"Best half-life={best_td['half_life_years']}y; "
                      f"all hl results: {td_results}")
        })

    # ---- 9. Region-conditional intl_exp_diff ----
    print("\n[9] intl_exp_diff × fav region")
    r = eval_interaction(
        'ix_x_fav_region', train, test,
        build_region_intl_xy, build_region_intl_xy,
        extra_coef_names=['ix_x_fav_EMEA', 'ix_x_fav_Pacific', 'ix_x_fav_CN'],
        alt_coef_names_for_ci=['ix_x_fav_CN'],
        baseline_xy_fn_train=build_region_intl_baseline_xy,
        baseline_xy_fn_test=build_region_intl_baseline_xy,
        direction_note="Hypothesis: CN gains MORE intl bonus (coef>0)."
    )
    if r:
        interactions.append(r)
        print(f"  primary coef={r['coef']:+.4f}  p={r['p']:.4g}  LR p={r['lr_p']:.4g}  "
              f"Δtest Brier={r['delta_test_brier']:+.5f}  bootCI={r['bootstrap_ci_coef']}")
        for k, v in r['all_coefs'].items():
            mark = "  *" if v['p'] < 0.05 else ""
            print(f"    {k:>20}: coef={v['coef']:+.4f}  p={v['p']:.3g}{mark}")

    # ---- Ship decisions ------------------------------------------------------
    DELTA_MIN = 0.0008
    for r in interactions:
        ci = r.get('bootstrap_ci_coef') or [None, None]
        lr_p = r.get('lr_p')
        delta = r.get('delta_test_brier', 0.0)
        ci_excludes_0 = (ci[0] is not None and ci[1] is not None
                         and ((ci[0] > 0 and ci[1] > 0) or (ci[0] < 0 and ci[1] < 0)))
        meets_delta = delta >= DELTA_MIN
        meets_lr = (lr_p is not None) and (not math.isnan(lr_p)) and (lr_p < BONF)
        r['ship_worthy'] = bool(meets_delta and meets_lr and ci_excludes_0)

    # ---- Best interaction + verdict ------------------------------------------
    shippable = [r for r in interactions if r['ship_worthy']]
    if shippable:
        best = max(shippable, key=lambda r: r['delta_test_brier'])
        best_name = best['name']
        verdict = f"ship: {best_name}"
    else:
        best = max(interactions, key=lambda r: r['delta_test_brier'])
        best_name = best['name'] if best['delta_test_brier'] > 0 else None
        if best['delta_test_brier'] >= DELTA_MIN / 2 or (best.get('lr_p') is not None
                                                        and not math.isnan(best['lr_p'])
                                                        and best['lr_p'] < 0.05):
            verdict = "marginal"
        else:
            verdict = "no signal"

    headline = (
        f"Best interaction: {best['name']} "
        f"Δtest Brier={best['delta_test_brier']:+.5f} "
        f"(baseline={baseline_test_brier:.5f}); "
        f"ship_worthy={best['ship_worthy']}; verdict='{verdict}'."
    )
    print("\n" + "=" * 60)
    print(headline)

    # ---- emit JSON ---------------------------------------------------------
    # Strip non-JSONable items
    serial_interactions = []
    for r in interactions:
        serial_interactions.append({
            'name': r['name'],
            'coef': r['coef'],
            'coef_name': r['coef_name'],
            'p': r['p'],
            'lr_p': r['lr_p'],
            'lr_stat': r.get('lr_stat'),
            'df': r['df'],
            'test_brier': r['test_brier'],
            'baseline_test_brier': r['baseline_test_brier'],
            'delta_test_brier': r['delta_test_brier'],
            'test_platt_b': r['test_platt_b'],
            'test_platt_a': r['test_platt_a'],
            'bootstrap_ci_coef': r['bootstrap_ci_coef'],
            'n_train': r['n_train'],
            'n_test': r['n_test'],
            'all_coefs': r['all_coefs'],
            'ship_worthy': r['ship_worthy'],
            'notes': r['notes'],
        })

    out = {
        'n_series': int(n_total),
        'n_train': int(n_tr),
        'n_test': int(n_te),
        'train_date_range': [train[0]['date'], train[-1]['date']],
        'test_date_range':  [test[0]['date'], test[-1]['date']],
        'baseline_test_brier': float(baseline_test_brier),
        'baseline_test_platt': float(baseline_test_platt_b),
        'baseline_train_coefs': {
            'const': float(res_b.params[0]),
            'rating_diff': float(res_b.params[1]),
            'intl_exp_diff': float(res_b.params[2]),
        },
        'bonferroni_alpha': float(BONF),
        'ship_threshold_delta_brier': DELTA_MIN,
        'time_decay_results': td_results,
        'interactions': serial_interactions,
        'best_interaction': best_name,
        'headline': headline,
        'verdict': verdict,
    }

    out_path = os.path.join(DATA, 'projection_research_phase2_interactions.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == '__main__':
    main()
