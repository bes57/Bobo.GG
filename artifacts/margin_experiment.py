"""
Phase 2A margin-function experiment.

Tests alternate per-map round-differential transforms (sqrt vs log1p, tanh4,
capped_linear_8, huber4). For each, rebuilds all 4-year rating timelines
in-memory using a patched massey_ratings, then evaluates with the harness
across an inner BETA grid to pick the best beta per variant.

Writes results to artifacts/margin_experiment.json.
"""
from __future__ import annotations
import os, sys, json, math, time, copy
from pathlib import Path
from datetime import datetime

import numpy as np

ROOT = Path('/Users/benny_es1/PythonTest')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scrapers'))

import BuildMapRatings as BMR
import BuildRatingTimeline as BRT
# harness lives at /artifacts/harness.py
sys.path.insert(0, str(ROOT / 'artifacts'))
import harness as H

# ---------------------------------------------------------------------------
# 1. Define a custom massey_ratings that accepts a margin transform fn.
# ---------------------------------------------------------------------------

def make_massey(margin_fn):
    """Return a massey_ratings clone that applies margin_fn(raw_rd) instead of
    the current power transform. RD_SCALE is still multiplied in for fairness
    (it's a free scale; beta will absorb anyway, but stay consistent)."""
    INTL_EVENTS = BMR.INTL_EVENTS
    CHAMPIONS_MULT = BMR.CHAMPIONS_MULT
    INTL_WIN_MULT = BMR.INTL_WIN_MULT
    INTL_LOSS_MULT = BMR.INTL_LOSS_MULT
    RD_SCALE = BMR.RD_SCALE
    _effective_weeks_ago = BMR._effective_weeks_ago
    _team_continuity_factor = BMR._team_continuity_factor

    def massey_ratings(games, lambda_decay, ref_date, min_games=0):
        if not games:
            return {}
        teams = sorted({g['winner'] for g in games} | {g['loser'] for g in games})
        if min_games > 0:
            counts = {}
            for g in games:
                counts[g['winner']] = counts.get(g['winner'], 0) + 1
                counts[g['loser']] = counts.get(g['loser'], 0) + 1
            teams = [t for t in teams if counts.get(t, 0) >= min_games]
            games = [g for g in games if g['winner'] in teams and g['loser'] in teams]
            if not games:
                return {}
        n = len(teams)
        idx = {t: i for i, t in enumerate(teams)}
        M = np.zeros((n, n))
        p = np.zeros(n)
        for g in games:
            if g['winner'] not in idx or g['loser'] not in idx:
                continue
            is_intl = g.get('event_id') in INTL_EVENTS
            is_champions = 'champions' in g.get('event_id', '')
            eff_w = _effective_weeks_ago(g['winner'], g['date'], ref_date)
            eff_l = _effective_weeks_ago(g['loser'], g['date'], ref_date)
            w_winner = math.exp(-lambda_decay * eff_w)
            w_loser = math.exp(-lambda_decay * eff_l)
            base_w = math.sqrt(w_winner * w_loser)
            cont_w = _team_continuity_factor(g['winner'], g['date'], ref_date)
            cont_l = _team_continuity_factor(g['loser'], g['date'], ref_date)
            base_w *= math.sqrt(cont_w * cont_l)
            if is_champions:
                win_mult = CHAMPIONS_MULT; los_mult = CHAMPIONS_MULT
            elif is_intl:
                win_mult = INTL_WIN_MULT; los_mult = INTL_LOSS_MULT
            else:
                win_mult = 1.0; los_mult = 1.0
            w_win = base_w * win_mult
            w_los = base_w * los_mult
            w_sym = min(w_win, w_los)
            raw_rd = g['wr'] - g['lr']
            rd = margin_fn(raw_rd) * RD_SCALE
            i, j = idx[g['winner']], idx[g['loser']]
            M[i, i] += w_sym; M[j, j] += w_sym
            M[i, j] -= w_sym; M[j, i] -= w_sym
            p[i] += w_win * rd; p[j] -= w_los * rd
        M[-1, :] = 1.0
        p[-1] = 0.0
        ridge = 0.5
        for i in range(n - 1):
            M[i, i] += ridge
        M[-1, :] = 1.0
        p[-1] = 0.0
        try:
            r = np.linalg.solve(M, p)
        except np.linalg.LinAlgError:
            r, *_ = np.linalg.lstsq(M, p, rcond=None)
        return {t: float(r[idx[t]]) for t in teams}
    return massey_ratings


# Margin transforms
def margin_sqrt(d):
    return math.copysign(math.sqrt(abs(d)), d)

def margin_log1p(d):
    return math.copysign(math.log1p(abs(d)), d)

def margin_tanh4(d):
    return 4.0 * math.tanh(d / 4.0)

def margin_capped_linear_8(d):
    if d > 8:
        return 8.0
    if d < -8:
        return -8.0
    return float(d)

def margin_huber4(d):
    a = abs(d)
    if a < 4:
        return float(d)
    # 4*sqrt(|d|/2 - 1) * sign
    val = 4.0 * math.sqrt(a / 2.0 - 1.0)
    return math.copysign(val, d)


MARGINS = {
    'sqrt': margin_sqrt,
    'log1p': margin_log1p,
    'tanh4': margin_tanh4,
    'capped_linear_8': margin_capped_linear_8,
    'huber4': margin_huber4,
}


# ---------------------------------------------------------------------------
# 2. Build timelines for all 4 years using the patched massey.
# ---------------------------------------------------------------------------

def build_all_timelines(margin_name, margin_fn, all_games, scratch_dir):
    """Rebuild timelines for 2023,2024,2025,2026 into scratch_dir.
    Returns dict {year: out_path}."""
    custom_massey = make_massey(margin_fn)
    # Patch: BRT uses module-level reference. Patch BRT.massey_ratings since
    # build_year_timeline calls it as massey_ratings(...).
    orig = BRT.massey_ratings
    BRT.massey_ratings = custom_massey
    # Also patch BMR.massey_ratings — _compute_intl_weights, _compute_indirect_intl_w
    # use BMR.massey_ratings? Let's check… Actually those helpers don't call
    # massey_ratings; they just compute weighted counts. Safe.
    try:
        out_paths = {}
        scratch_dir.mkdir(parents=True, exist_ok=True)
        for year in [2023, 2024, 2025, 2026]:
            t0 = time.time()
            checkpoints, match_events = BRT.build_year_timeline(all_games, year, existing=None)
            out = scratch_dir / f'rating_timeline_{margin_name}_{year}.json'
            with open(out, 'w') as f:
                json.dump({
                    'year': year,
                    'checkpoints': checkpoints,
                    'match_events': match_events,
                }, f)
            out_paths[year] = out
            print(f'  [{margin_name}] {year}: {len(match_events)} matches in {time.time()-t0:.1f}s', flush=True)
        return out_paths
    finally:
        BRT.massey_ratings = orig


def evaluate_variant(timeline_paths, betas):
    """Load timelines via harness, evaluate at each beta. Returns
    {beta: {brier, platt_b}}."""
    files = [timeline_paths[y] for y in [2026, 2023, 2024, 2025]]
    matches = H.load_matches(files)
    out = {}
    for beta in betas:
        probs, outs = H.predict_series(matches, beta=beta)
        b = H.brier(probs, outs)
        a, slope = H.platt_slope(probs, outs)
        out[beta] = {'brier': b, 'platt_b': slope, 'platt_a': a,
                     'probs': probs, 'outs': outs, 'n': len(matches)}
    return out


# ---------------------------------------------------------------------------
# 3. IRLS — Huber-weighted re-solve.
# ---------------------------------------------------------------------------

def make_massey_irls(margin_fn, n_iters=5, huber_k_mult=1.345):
    """Build a massey_ratings that does n_iters IRLS rounds with Huber weights."""
    INTL_EVENTS = BMR.INTL_EVENTS
    CHAMPIONS_MULT = BMR.CHAMPIONS_MULT
    INTL_WIN_MULT = BMR.INTL_WIN_MULT
    INTL_LOSS_MULT = BMR.INTL_LOSS_MULT
    RD_SCALE = BMR.RD_SCALE
    _effective_weeks_ago = BMR._effective_weeks_ago
    _team_continuity_factor = BMR._team_continuity_factor

    def massey_ratings(games, lambda_decay, ref_date, min_games=0):
        if not games:
            return {}
        teams = sorted({g['winner'] for g in games} | {g['loser'] for g in games})
        if min_games > 0:
            counts = {}
            for g in games:
                counts[g['winner']] = counts.get(g['winner'], 0) + 1
                counts[g['loser']] = counts.get(g['loser'], 0) + 1
            teams = [t for t in teams if counts.get(t, 0) >= min_games]
            games = [g for g in games if g['winner'] in teams and g['loser'] in teams]
            if not games:
                return {}
        n = len(teams)
        idx = {t: i for i, t in enumerate(teams)}

        # Pre-compute per-game weights / margins (independent of ratings)
        rows = []
        for g in games:
            if g['winner'] not in idx or g['loser'] not in idx:
                continue
            is_intl = g.get('event_id') in INTL_EVENTS
            is_champions = 'champions' in g.get('event_id', '')
            eff_w = _effective_weeks_ago(g['winner'], g['date'], ref_date)
            eff_l = _effective_weeks_ago(g['loser'], g['date'], ref_date)
            w_winner = math.exp(-lambda_decay * eff_w)
            w_loser = math.exp(-lambda_decay * eff_l)
            base_w = math.sqrt(w_winner * w_loser)
            cont_w = _team_continuity_factor(g['winner'], g['date'], ref_date)
            cont_l = _team_continuity_factor(g['loser'], g['date'], ref_date)
            base_w *= math.sqrt(cont_w * cont_l)
            if is_champions:
                wm = lm = CHAMPIONS_MULT
            elif is_intl:
                wm = INTL_WIN_MULT; lm = INTL_LOSS_MULT
            else:
                wm = lm = 1.0
            w_win = base_w * wm
            w_los = base_w * lm
            w_sym = min(w_win, w_los)
            raw_rd = g['wr'] - g['lr']
            rd = margin_fn(raw_rd) * RD_SCALE
            rows.append((idx[g['winner']], idx[g['loser']], w_sym, w_win, w_los, rd))

        def solve(extra_weight_mult):
            """extra_weight_mult: per-row multiplier (length len(rows))."""
            M = np.zeros((n, n))
            p = np.zeros(n)
            for k, (i, j, w_sym, w_win, w_los, rd) in enumerate(rows):
                em = extra_weight_mult[k]
                ws = w_sym * em
                ww = w_win * em
                wl = w_los * em
                M[i, i] += ws; M[j, j] += ws
                M[i, j] -= ws; M[j, i] -= ws
                p[i] += ww * rd; p[j] -= wl * rd
            M[-1, :] = 1.0
            p[-1] = 0.0
            ridge = 0.5
            for ii in range(n - 1):
                M[ii, ii] += ridge
            M[-1, :] = 1.0
            p[-1] = 0.0
            try:
                r = np.linalg.solve(M, p)
            except np.linalg.LinAlgError:
                r, *_ = np.linalg.lstsq(M, p, rcond=None)
            return r

        # Iteration 0: equal weights
        mults = np.ones(len(rows))
        r = solve(mults)
        for it in range(n_iters):
            # Residuals: predicted margin = r[i] - r[j]; observed = rd
            resid = np.empty(len(rows))
            for k, (i, j, _, _, _, rd) in enumerate(rows):
                resid[k] = rd - (r[i] - r[j])
            # MAD-based sigma
            sigma = np.median(np.abs(resid - np.median(resid))) * 1.4826
            if sigma <= 0:
                break
            k_h = huber_k_mult * sigma
            abs_r = np.abs(resid)
            mults = np.where(abs_r <= k_h, 1.0, k_h / np.maximum(abs_r, 1e-12))
            r_new = solve(mults)
            if np.max(np.abs(r_new - r)) < 1e-5:
                r = r_new
                break
            r = r_new
        return {t: float(r[idx[t]]) for t in teams}

    return massey_ratings


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------

def main():
    start = time.time()
    DEADLINE = 13 * 60  # 13 min hard stop

    print('Loading games via BRT.load_all_games()...', flush=True)
    all_games = BRT.load_all_games()
    print(f'  {len(all_games)} game-maps loaded in {time.time()-start:.1f}s', flush=True)

    scratch = ROOT / 'artifacts' / 'margin_scratch'
    scratch.mkdir(parents=True, exist_ok=True)

    BETAS = [0.10, 0.12, 0.14, 0.16, 0.18]
    BASELINE = 0.23066798182013623  # from artifacts/baseline.json

    results = {
        'baseline_sqrt_beta_0.140_brier': BASELINE,
        'variants': {},
        'irls_on_winner': None,
    }

    # Run each variant
    variant_order = ['sqrt', 'log1p', 'tanh4', 'capped_linear_8', 'huber4']

    cached_probs = {}  # for paired bootstrap later

    for name in variant_order:
        elapsed = time.time() - start
        if elapsed > DEADLINE:
            print(f'  DEADLINE hit before {name}, writing partial', flush=True)
            break
        print(f'\n=== variant: {name} (elapsed {elapsed:.0f}s) ===', flush=True)
        try:
            paths = build_all_timelines(name, MARGINS[name], all_games, scratch)
        except Exception as e:
            print(f'  build failed: {e}', flush=True)
            results['variants'][name] = {'error': str(e)}
            continue

        beta_results = evaluate_variant(paths, BETAS)
        best_beta = min(beta_results, key=lambda b: beta_results[b]['brier'])
        bb = beta_results[best_beta]
        results['variants'][name] = {
            'best_beta': best_beta,
            'best_brier': bb['brier'],
            'platt_b': bb['platt_b'],
            'platt_a': bb['platt_a'],
            'delta_vs_baseline': bb['brier'] - BASELINE,
            'all_betas': {f'{b:.2f}': {'brier': beta_results[b]['brier'],
                                       'platt_b': beta_results[b]['platt_b']}
                          for b in BETAS},
            'n': bb['n'],
        }
        cached_probs[name] = (bb['probs'], bb['outs'])

        # Snapshot to disk after each variant
        out_path = ROOT / 'artifacts' / 'margin_experiment.json'
        with open(out_path, 'w') as f:
            json.dump({k: v for k, v in results.items()
                       if k != 'irls_on_winner' or v is not None}, f, indent=2,
                      default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
        print(f'  best_beta={best_beta} brier={bb["brier"]:.5f} '
              f'(delta {bb["brier"]-BASELINE:+.5f}) platt_b={bb["platt_b"]:.3f}',
              flush=True)

    # Pick winner — lowest Brier among completed variants
    completed = {k: v for k, v in results['variants'].items() if 'best_brier' in v}
    if completed:
        winner = min(completed, key=lambda k: completed[k]['best_brier'])
        results['winning_variant'] = winner
        results['winning_beta'] = completed[winner]['best_beta']
        results['winning_brier'] = completed[winner]['best_brier']
        best_betas = [v['best_beta'] for v in completed.values()]
        results['beta_spread_across_variants'] = max(best_betas) - min(best_betas)
        results['scale_coupling_confirmed'] = (
            (max(best_betas) - min(best_betas)) > 0.03
        )

        # Paired bootstrap vs sqrt at sqrt's best beta.
        # NOTE: outs *can* differ across variants because "fav_won" is defined
        # by which side had the higher pre-match rating, and ratings change with
        # the margin function. To get a paired comparison we work in "winner-
        # of-series" space: prob that the actual series-winner wins =
        # fav_won ? p : 1-p. That target is always 1, identical across variants.
        if 'sqrt' in cached_probs and winner != 'sqrt':
            p_sqrt, o_sqrt = cached_probs['sqrt']
            p_win, o_win = cached_probs[winner]
            # Convert to "p(actual winner wins)" with target=1 each, so the
            # squared error is (1 - p_aw)^2 where p_aw = p if fav_won else 1-p.
            paw_sqrt = np.where(o_sqrt == 1, p_sqrt, 1 - p_sqrt)
            paw_win = np.where(o_win == 1, p_win, 1 - p_win)
            targets = np.ones_like(paw_sqrt)
            boot = H.paired_bootstrap_brier(paw_sqrt, paw_win, targets, n_boot=1000)
            results['paired_bootstrap_vs_sqrt'] = boot

        # IRLS on winner — only if we have time
        elapsed = time.time() - start
        if elapsed < DEADLINE - 60 and winner in MARGINS:
            print(f'\n=== IRLS on winner: {winner} (elapsed {elapsed:.0f}s) ===',
                  flush=True)
            try:
                # Patch BRT.massey_ratings with the IRLS variant
                irls_massey = make_massey_irls(MARGINS[winner], n_iters=5)
                orig = BRT.massey_ratings
                BRT.massey_ratings = irls_massey
                try:
                    out_paths = {}
                    for year in [2023, 2024, 2025, 2026]:
                        t0 = time.time()
                        cps, mes = BRT.build_year_timeline(all_games, year, existing=None)
                        out = scratch / f'rating_timeline_irls_{winner}_{year}.json'
                        with open(out, 'w') as f:
                            json.dump({'year': year, 'checkpoints': cps,
                                       'match_events': mes}, f)
                        out_paths[year] = out
                        print(f'  IRLS {year}: {time.time()-t0:.1f}s', flush=True)
                    irls_results = evaluate_variant(out_paths, BETAS)
                    best_b = min(irls_results, key=lambda b: irls_results[b]['brier'])
                    bbb = irls_results[best_b]
                    results['irls_on_winner'] = {
                        'best_beta': best_b,
                        'brier': bbb['brier'],
                        'platt_b': bbb['platt_b'],
                        'delta_vs_winner': bbb['brier'] - completed[winner]['best_brier'],
                        'delta_vs_baseline': bbb['brier'] - BASELINE,
                    }
                    # Paired bootstrap IRLS vs winner (via p(actual winner wins))
                    p_w, o_w = cached_probs[winner]
                    p_i, o_i = bbb['probs'], bbb['outs']
                    paw_w = np.where(o_w == 1, p_w, 1 - p_w)
                    paw_i = np.where(o_i == 1, p_i, 1 - p_i)
                    targets = np.ones_like(paw_w)
                    boot2 = H.paired_bootstrap_brier(paw_w, paw_i, targets, n_boot=1000)
                    results['irls_paired_bootstrap_vs_winner'] = boot2
                finally:
                    BRT.massey_ratings = orig
            except Exception as e:
                results['irls_on_winner'] = {'error': str(e)}
                print(f'  IRLS failed: {e}', flush=True)
        else:
            results['irls_on_winner'] = {'skipped': 'deadline'}

    results['notes'] = (
        f'Walk-forward via leak-free pre-match ratings from rebuilt timelines. '
        f'Wall-clock {time.time()-start:.1f}s.'
    )

    # Final write (strip numpy)
    out_path = ROOT / 'artifacts' / 'margin_experiment.json'
    def _safe(o):
        if isinstance(o, dict):
            return {k: _safe(v) for k, v in o.items() if k not in ('probs', 'outs')}
        if isinstance(o, list):
            return [_safe(x) for x in o]
        if hasattr(o, 'tolist'):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        return o
    safe_results = _safe(results)
    with open(out_path, 'w') as f:
        json.dump(safe_results, f, indent=2)
    print(f'\nWrote {out_path}', flush=True)
    print(f'Total elapsed: {time.time()-start:.1f}s', flush=True)


if __name__ == '__main__':
    main()
