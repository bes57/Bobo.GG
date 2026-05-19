"""
Walk-forward CV harness for BenPom optimization.

Uses leak-free pre-match ratings from rating_timeline*.json (computed via the
12:01 AM rating snapshot — see MapElo.py for the design note). This is
effectively walk-forward already: each match's prediction uses only data
strictly before that match's date.

This module is the ONE source of truth for Brier / Platt / ECE during the
hour-pass. Phase-2 subagents import from here.
"""
from __future__ import annotations
import json
import math
import os
import re
import ast
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path('/Users/benny_es1/PythonTest')
TIMELINE_FILES = [
    ROOT / 'data' / 'rating_timeline.json',
    ROOT / 'data' / 'rating_timeline_2023.json',
    ROOT / 'data' / 'rating_timeline_2024.json',
    ROOT / 'data' / 'rating_timeline_2025.json',
]

INTL_EVENTS = {
    '2024_masters_madrid', '2024_masters_shanghai', '2024_champions',
    '2025_masters_bangkok', '2025_masters_toronto', '2025_champions',
    '2026_masters_santiago',
}

# Pull TEAM_REGIONS from BuildMapRatings so the harness doesn't drift
def _load_team_regions():
    src = (ROOT / 'scrapers' / 'BuildMapRatings.py').read_text()
    m = re.search(r'TEAM_REGIONS = \{(.+?)^\}', src, re.DOTALL | re.MULTILINE)
    return ast.literal_eval('{' + m.group(1) + '}')

TEAM_REGIONS = _load_team_regions()


def sigmoid(x):
    if isinstance(x, np.ndarray):
        return 1.0 / (1.0 + np.exp(-x))
    return 1.0 / (1.0 + math.exp(-x))


def series_prob(p_map, fmt='bo3'):
    if fmt == 'bo5':
        return p_map**3 * (1 + 3 * (1 - p_map) + 6 * (1 - p_map)**2)
    return p_map**2 * (3 - 2 * p_map)


def load_matches(timeline_files=None):
    """Load all matches with pre-match ratings, season, intl flag, region pair.

    Returns list of dicts with keys:
      date, event_id, season, winner, loser, delta_before, abs_delta,
      fav, dog, fav_won, fmt, intl, fav_region, dog_region, n_maps, maps[]
    """
    files = timeline_files or TIMELINE_FILES
    matches = []
    attendance = defaultdict(list)  # {org: [(date, season)]}

    # First pass: build attendance lookup
    for path in files:
        d = json.load(open(path))
        yr = d.get('year') or 2026
        for me in d.get('match_events', []):
            if me.get('event_id') in INTL_EVENTS:
                attendance[me['winner']].append((me['date'], yr))
                attendance[me['loser']].append((me['date'], yr))

    # Second pass: build matches
    for path in files:
        d = json.load(open(path))
        yr = d.get('year') or 2026
        for me in d.get('match_events', []):
            w_before = me.get('winner_before', 0.0)
            l_before = me.get('loser_before', 0.0)
            delta = w_before - l_before
            if delta == 0:
                continue  # model has no opinion

            ss = str(me.get('series_score', '')).split('-')
            try:
                wins_w = int(ss[0]) if ss else 2
            except Exception:
                wins_w = 2
            fmt = 'bo5' if wins_w >= 3 else 'bo3'

            fav = me['winner'] if delta > 0 else me['loser']
            dog = me['loser'] if delta > 0 else me['winner']
            fav_won = delta > 0

            intl = me.get('event_id', '') in INTL_EVENTS

            # Intl-exp-diff: did each side attend an intl event in this same
            # season strictly before this match's date?
            def attended(org):
                for d_, seas in attendance.get(org, []):
                    if seas == yr and d_ < me['date']:
                        return True
                return False

            intl_exp_diff = 0
            if intl:
                intl_exp_diff = (1 if attended(fav) else 0) - (1 if attended(dog) else 0)

            matches.append({
                'date': me.get('date', ''),
                'event_id': me.get('event_id', ''),
                'season': yr,
                'winner': me['winner'],
                'loser': me['loser'],
                'w_before': w_before,
                'l_before': l_before,
                'delta_before': delta,
                'abs_delta': abs(delta),
                'fav': fav,
                'dog': dog,
                'fav_won': fav_won,
                'fmt': fmt,
                'intl': intl,
                'intl_exp_diff': intl_exp_diff,
                'fav_region': TEAM_REGIONS.get(fav),
                'dog_region': TEAM_REGIONS.get(dog),
                'maps': me.get('maps', []),
                'match_id': me.get('match_id', ''),
            })
    matches.sort(key=lambda m: m['date'])
    return matches


def predict_series(matches, beta=0.140, beta_bo5=None, intl_bonus=0.22,
                   cn_dog_offset=0.47):
    """Vectorized prediction of fav series-win probability for every match.

    Returns (probs, outs, matches_in_order).
    """
    probs = np.empty(len(matches))
    outs = np.empty(len(matches), dtype=int)
    for i, m in enumerate(matches):
        b = beta_bo5 if (beta_bo5 is not None and m['fmt'] == 'bo5') else beta
        p_map = sigmoid(b * m['abs_delta'])
        ps = series_prob(p_map, m['fmt'])
        if m['intl'] and m['intl_exp_diff'] != 0 and intl_bonus != 0:
            ps = sigmoid(math.log(ps / (1 - ps)) + intl_bonus * m['intl_exp_diff'])
        if m['intl'] and m['dog_region'] == 'CN' and m['fav_region'] != 'CN' and cn_dog_offset != 0:
            ps = sigmoid(math.log(ps / (1 - ps)) + cn_dog_offset)
        probs[i] = ps
        outs[i] = 1 if m['fav_won'] else 0
    return probs, outs


def predict_maps(matches, beta=0.140, beta_bo5=None, intl_bonus=0.22,
                 cn_dog_offset=0.47):
    """Per-map probs (from favorite's perspective)."""
    probs, outs = [], []
    for m in matches:
        b = beta_bo5 if (beta_bo5 is not None and m['fmt'] == 'bo5') else beta
        p_map = sigmoid(b * m['abs_delta'])
        if m['intl'] and m['intl_exp_diff'] != 0 and intl_bonus != 0:
            p_map = sigmoid(math.log(p_map / (1 - p_map)) + intl_bonus * m['intl_exp_diff'])
        if m['intl'] and m['dog_region'] == 'CN' and m['fav_region'] != 'CN' and cn_dog_offset != 0:
            p_map = sigmoid(math.log(p_map / (1 - p_map)) + cn_dog_offset)
        for mp in m['maps']:
            w = mp.get('winner', '')
            probs.append(p_map)
            outs.append(1 if w == m['fav'] else 0)
    return np.array(probs), np.array(outs)


def brier(p, o):
    p = np.asarray(p); o = np.asarray(o)
    return float(np.mean((p - o) ** 2))


def platt_slope(p, o):
    """Logistic regression of o ~ a + b * logit(p). Returns (a, b)."""
    eps = 1e-9
    p = np.clip(np.asarray(p), eps, 1 - eps)
    o = np.asarray(o)
    z = np.log(p / (1 - p))
    a, b = 0.0, 1.0
    for _ in range(80):
        eta = a + b * z
        pi = 1 / (1 + np.exp(-eta))
        W = pi * (1 - pi)
        g = np.array([np.sum(o - pi), np.sum(z * (o - pi))])
        H = np.array([
            [-np.sum(W),     -np.sum(W * z)],
            [-np.sum(W * z), -np.sum(W * z * z)],
        ])
        delta = np.linalg.solve(H, -g)
        a += delta[0]; b += delta[1]
        if abs(delta).max() < 1e-8:
            break
    return float(a), float(b)


def ece(probs, outs, n_bins=10):
    """Expected Calibration Error with equal-frequency bins."""
    probs = np.asarray(probs); outs = np.asarray(outs)
    order = np.argsort(probs)
    probs = probs[order]; outs = outs[order]
    n = len(probs)
    bin_size = n // n_bins
    total = 0.0
    for i in range(n_bins):
        lo = i * bin_size
        hi = (i + 1) * bin_size if i < n_bins - 1 else n
        if hi <= lo:
            continue
        p_avg = probs[lo:hi].mean()
        o_avg = outs[lo:hi].mean()
        total += (hi - lo) * abs(p_avg - o_avg)
    return float(total / n)


def per_bucket_brier(matches, probs, outs):
    """Return dict: {bucket_key: {brier, n}} for (fmt, region-bucket)."""
    buckets = defaultdict(lambda: {'p': [], 'o': []})
    for m, p, o in zip(matches, probs, outs):
        fmt = m['fmt']
        if m['intl']:
            if m['fav_region'] == 'CN' or m['dog_region'] == 'CN':
                rb = 'intl-CN'
            else:
                rb = 'intl-noCN'
        else:
            rb = 'domestic'
        k = f'{fmt}_{rb}'
        buckets[k]['p'].append(p)
        buckets[k]['o'].append(o)
    out = {}
    for k, v in buckets.items():
        out[k] = {'brier': brier(v['p'], v['o']), 'n': len(v['p'])}
    return out


def paired_bootstrap_brier(p_a, p_b, outs, n_boot=1000, seed=0):
    """Paired bootstrap on Brier difference (p_b - p_a)."""
    rng = np.random.default_rng(seed)
    n = len(p_a)
    err_a = (p_a - outs) ** 2
    err_b = (p_b - outs) ** 2
    diffs = err_b - err_a
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = diffs[idx].mean()
    mean = diffs.mean()
    se = diffs.std(ddof=1) / np.sqrt(n)
    lo = np.quantile(boots, 0.025)
    hi = np.quantile(boots, 0.975)
    # one-sample test: H0 mean=0
    t = mean / se if se > 0 else 0
    p_two = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return {
        'mean': float(mean), 'se': float(se),
        'ci_lo': float(lo), 'ci_hi': float(hi),
        't': float(t), 'p_two_sided': float(p_two),
    }


def compute_baseline(matches=None, **kwargs):
    """Compute all baseline metrics, return as dict."""
    if matches is None:
        matches = load_matches()
    probs, outs = predict_series(matches, **kwargs)
    map_probs, map_outs = predict_maps(matches, **kwargs)
    a, b = platt_slope(probs, outs)
    a_m, b_m = platt_slope(map_probs, map_outs)
    return {
        'n_series': len(matches),
        'n_maps': len(map_probs),
        'series_brier': brier(probs, outs),
        'map_brier': brier(map_probs, map_outs),
        'series_platt_a': a,
        'series_platt_b': b,
        'map_platt_a': a_m,
        'map_platt_b': b_m,
        'series_ece': ece(probs, outs),
        'map_ece': ece(map_probs, map_outs),
        'per_bucket': per_bucket_brier(matches, probs, outs),
        'config': kwargs,
    }


if __name__ == '__main__':
    matches = load_matches()
    print(f'Loaded {len(matches)} series, {sum(len(m["maps"]) for m in matches)} maps')
    baseline = compute_baseline(matches)
    Path('artifacts').mkdir(exist_ok=True)
    json.dump(baseline, open('artifacts/baseline.json', 'w'), indent=2)
    print(json.dumps(baseline, indent=2))
