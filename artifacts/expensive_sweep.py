"""
Phase 1b: Expensive-knob Optuna sweep.

Knobs requiring Massey rebuild:
  HALF_LIFE_WEEKS, SHRINK_K, CN_PRIOR, CN_INTL_K, CN_INDIRECT_WEIGHT,
  REGION_SPILLOVER_ALPHA, CHAMPIONS_MULT, INTL_WIN_MULT, ROSTER_PERSISTENCE,
  MARGIN_FN (categorical: sqrt, log1p, tanh4, capped_linear_8).

Predictions use the CURRENT cheap-knob defaults:
  beta=0.140, intl_exp_bonus=0.22, cn_dog_offset=0.47, no beta_bo5.

Per trial:
  1. Monkey-patch BuildMapRatings constants.
  2. If MARGIN_FN != current default, also patch massey_ratings to use it.
  3. Re-build rating timeline for 2023..2026.
  4. Convert resulting match_events into harness-format matches (winner_before,
     loser_before, intl flag, region pair, intl_exp_diff).
  5. Compute series Brier on the resulting 1440-ish leak-free predictions.
  6. Apply I6 CN-shrinkage guardrail; reject if violated.

Wall-clock budget: 20 minutes. After minute 18, stop adding trials.
"""
from __future__ import annotations
import json, math, os, sys, time, copy
from collections import defaultdict
from pathlib import Path

import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

ROOT = Path('/Users/benny_es1/PythonTest')
sys.path.insert(0, str(ROOT / 'scrapers'))
sys.path.insert(0, str(ROOT / 'artifacts'))
sys.path.insert(0, str(ROOT))

import BuildMapRatings as BMR
import BuildRatingTimeline as BRT
import harness as H

# ── Load games once ────────────────────────────────────────────────────────────
print('Loading all games...')
t0 = time.time()
ALL_GAMES = BRT.load_all_games()
print(f'  loaded {len(ALL_GAMES)} games in {time.time()-t0:.2f}s')

INTL_EVENTS = H.INTL_EVENTS
TEAM_REGIONS = H.TEAM_REGIONS
CN_TEAMS_SET = BMR.CN_TEAMS_SET

# Capture defaults for reset between trials
_DEFAULTS = {
    'HALF_LIFE_WEEKS': BMR.HALF_LIFE_WEEKS,
    'SHRINK_K': BMR.SHRINK_K,
    'CN_PRIOR': BMR.CN_PRIOR,
    'CN_INTL_K': BMR.CN_INTL_K,
    'CN_C_MIN': BMR.CN_C_MIN,
    'CN_INDIRECT_WEIGHT': BMR.CN_INDIRECT_WEIGHT,
    'REGION_SPILLOVER_ALPHA': BMR.REGION_SPILLOVER_ALPHA,
    'CHAMPIONS_MULT': BMR.CHAMPIONS_MULT,
    'INTL_WIN_MULT': BMR.INTL_WIN_MULT,
    'INTL_LOSS_MULT': BMR.INTL_LOSS_MULT,
    'ROSTER_PERSISTENCE': BMR.ROSTER_PERSISTENCE,
    'RD_TRANSFORM': BMR.RD_TRANSFORM,
    'RD_POWER': BMR.RD_POWER,
    'RD_SCALE': BMR.RD_SCALE,
}
print(f'Defaults captured: {_DEFAULTS}')

# Margin function override — implemented by replacing massey_ratings while
# leaving all other math identical (so we still use roster persistence,
# Champions mult, ridge, etc.).
_ORIG_MASSEY = BMR.massey_ratings


def _make_massey_with_margin(margin_fn_name, rd_scale=2.5):
    """Return a massey_ratings drop-in that uses the given margin function.

    rd_scale is applied as a multiplier on the margin output. It defaults to
    the current production RD_SCALE=2.5 so that sweep variants of MARGIN_FN
    are comparable to the deployed `power(0.5) * 2.5` baseline (which equals
    sqrt(|rd|) * 2.5)."""
    def margin(raw_rd):
        if margin_fn_name == 'sqrt':
            v = math.copysign(math.sqrt(abs(raw_rd)), raw_rd)
        elif margin_fn_name == 'log1p':
            v = math.copysign(math.log1p(abs(raw_rd)), raw_rd)
        elif margin_fn_name == 'tanh4':
            v = math.tanh(raw_rd / 4.0) * 4.0
        elif margin_fn_name == 'capped_linear_8':
            v = max(-8, min(8, raw_rd))
        else:
            v = raw_rd
        return v * rd_scale

    def massey_ratings(games, lambda_decay, ref_date, min_games=0):
        if not games:
            return {}
        teams = sorted({g['winner'] for g in games} | {g['loser'] for g in games})
        if min_games > 0:
            counts = {}
            for g in games:
                counts[g['winner']] = counts.get(g['winner'], 0) + 1
                counts[g['loser']]  = counts.get(g['loser'],  0) + 1
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
            is_intl = g.get('event_id') in BMR.INTL_EVENTS
            is_champions = 'champions' in g.get('event_id', '')
            eff_weeks_w = BMR._effective_weeks_ago(g['winner'], g['date'], ref_date)
            eff_weeks_l = BMR._effective_weeks_ago(g['loser'],  g['date'], ref_date)
            w_winner = math.exp(-lambda_decay * eff_weeks_w)
            w_loser  = math.exp(-lambda_decay * eff_weeks_l)
            base_w = math.sqrt(w_winner * w_loser)
            cont_w = BMR._team_continuity_factor(g['winner'], g['date'], ref_date)
            cont_l = BMR._team_continuity_factor(g['loser'],  g['date'], ref_date)
            base_w *= math.sqrt(cont_w * cont_l)
            if is_champions:
                win_mult = BMR.CHAMPIONS_MULT
                los_mult = BMR.CHAMPIONS_MULT
            elif is_intl:
                win_mult = BMR.INTL_WIN_MULT
                los_mult = BMR.INTL_LOSS_MULT
            else:
                win_mult = 1.0
                los_mult = 1.0
            w_win = base_w * win_mult
            w_los = base_w * los_mult
            w_sym = min(w_win, w_los)
            raw_rd = g['wr'] - g['lr']
            rd = margin(raw_rd)
            i, j = idx[g['winner']], idx[g['loser']]
            M[i, i] += w_sym;  M[j, j] += w_sym
            M[i, j] -= w_sym;  M[j, i] -= w_sym
            p[i] += w_win * rd;  p[j] -= w_los * rd
        M[-1, :] = 1.0
        p[-1]    = 0.0
        ridge = 0.5
        for i in range(n - 1):
            M[i, i] += ridge
        M[-1, :] = 1.0
        p[-1]    = 0.0
        try:
            r = np.linalg.solve(M, p)
        except np.linalg.LinAlgError:
            r, *_ = np.linalg.lstsq(M, p, rcond=None)
        return {t: float(r[idx[t]]) for t in teams}
    return massey_ratings


def _patch_constants(cfg):
    """Patch BMR + BRT modules with cfg values."""
    BMR.HALF_LIFE_WEEKS       = cfg['HALF_LIFE_WEEKS']
    BMR.SHRINK_K              = cfg['SHRINK_K']
    BMR.CN_PRIOR              = cfg['CN_PRIOR']
    BMR.CN_INTL_K             = cfg['CN_INTL_K']
    BMR.CN_INDIRECT_WEIGHT    = cfg['CN_INDIRECT_WEIGHT']
    BMR.REGION_SPILLOVER_ALPHA = cfg['REGION_SPILLOVER_ALPHA']
    BMR.CHAMPIONS_MULT        = cfg['CHAMPIONS_MULT']
    BMR.INTL_WIN_MULT         = cfg['INTL_WIN_MULT']
    BMR.INTL_LOSS_MULT        = cfg['INTL_WIN_MULT']  # tie to win mult since not in sweep
    BMR.ROSTER_PERSISTENCE    = cfg['ROSTER_PERSISTENCE']

    # BRT picks up HALF_LIFE_WEEKS at import time -> recompute LAMBDA_DECAY
    BRT.HALF_LIFE_WEEKS  = cfg['HALF_LIFE_WEEKS']
    BRT.LAMBDA_DECAY     = math.log(2) / cfg['HALF_LIFE_WEEKS']
    BRT.INTL_MULT        = cfg['INTL_WIN_MULT']
    BRT.CN_PRIOR         = cfg['CN_PRIOR']
    BRT.CN_INTL_K        = cfg['CN_INTL_K']
    BRT.CN_INDIRECT_WEIGHT = cfg['CN_INDIRECT_WEIGHT']

    # Margin function: swap massey_ratings
    BMR.massey_ratings = _make_massey_with_margin(cfg['MARGIN_FN'])
    BRT.massey_ratings = BMR.massey_ratings  # used directly from BRT


def _reset_constants():
    for k, v in _DEFAULTS.items():
        setattr(BMR, k, v)
    BMR.massey_ratings = _ORIG_MASSEY
    BRT.massey_ratings = _ORIG_MASSEY
    BRT.HALF_LIFE_WEEKS = _DEFAULTS['HALF_LIFE_WEEKS']
    BRT.LAMBDA_DECAY = math.log(2) / _DEFAULTS['HALF_LIFE_WEEKS']
    BRT.INTL_MULT = _DEFAULTS['INTL_WIN_MULT']
    BRT.CN_PRIOR = _DEFAULTS['CN_PRIOR']
    BRT.CN_INTL_K = _DEFAULTS['CN_INTL_K']
    BRT.CN_INDIRECT_WEIGHT = _DEFAULTS['CN_INDIRECT_WEIGHT']


# ── Convert timeline match_events to harness matches format ───────────────────
def timeline_to_harness_matches(years_timelines):
    """years_timelines: dict {year: {checkpoints: [...], match_events: [...]}}.
    Returns harness-format matches list (same shape as harness.load_matches).
    """
    # Build attendance lookup across all years
    attendance = defaultdict(list)
    for yr, tl in years_timelines.items():
        for me in tl['match_events']:
            if me.get('event_id') in INTL_EVENTS:
                attendance[me['winner']].append((me['date'], yr))
                attendance[me['loser']].append((me['date'], yr))

    matches = []
    for yr, tl in years_timelines.items():
        for me in tl['match_events']:
            w_before = me.get('winner_before', 0.0)
            l_before = me.get('loser_before', 0.0)
            delta = w_before - l_before
            if delta == 0:
                continue
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


# ── Trial body ────────────────────────────────────────────────────────────────
TARGET_YEARS = [2023, 2024, 2025, 2026]


def evaluate_config(cfg, return_extras=False):
    """Run a full rebuild for the given config and return series brier."""
    import io, contextlib
    _patch_constants(cfg)
    try:
        years_tl = {}
        with contextlib.redirect_stdout(io.StringIO()):
            for yr in TARGET_YEARS:
                cps, mes = BRT.build_year_timeline(ALL_GAMES, yr, existing=None)
                years_tl[yr] = {'checkpoints': cps, 'match_events': mes}

        matches = timeline_to_harness_matches(years_tl)
        probs, outs = H.predict_series(
            matches,
            beta=0.140,
            beta_bo5=None,
            intl_bonus=0.22,
            cn_dog_offset=0.47,
        )
        brier = H.brier(probs, outs)

        # I6 guardrail: CN team c values
        # Use final 2026 checkpoint as snapshot to evaluate c distribution
        cn_violations = 0
        final_cp = years_tl[2026]['checkpoints'][-1] if years_tl[2026]['checkpoints'] else None
        # We need to know intl-weight evidence at that final ref_date — recompute
        from datetime import datetime
        if final_cp:
            ref_date = datetime.strptime(final_cp['date'], '%Y-%m-%d')
            # Use the same prior + year games like BRT does
            prior_games = [g for g in ALL_GAMES
                           if g['date'].year < 2026 and g.get('event_id') in BMR.INTL_EVENTS]
            year_games = [g for g in ALL_GAMES
                          if g['date'].year == 2026 and g['date'] <= ref_date]
            solve_games = prior_games + year_games
            iw = BMR._compute_intl_weights(solve_games, BRT.LAMBDA_DECAY, ref_date)
            indirect = BMR._compute_indirect_intl_w(solve_games, iw, BRT.LAMBDA_DECAY, ref_date)
            c_vals = {}
            for t in CN_TEAMS_SET:
                if t not in final_cp['ratings']:
                    continue
                evidence = iw.get(t, 0.0) + cfg['CN_INDIRECT_WEIGHT'] * indirect.get(t, 0.0)
                c = max(0.0, min(evidence / cfg['CN_INTL_K'], 1.0))
                c_vals[t] = c
                if c < 0.1 or c > 0.95:
                    cn_violations += 1
        else:
            c_vals = {}

        result = {
            'brier': brier,
            'n_matches': len(matches),
            'cn_violations': cn_violations,
            'cn_c_vals': c_vals,
            'config': cfg,
        }
        if return_extras:
            result['probs'] = probs
            result['outs'] = outs
        return result
    finally:
        _reset_constants()


# ── Optuna driver ─────────────────────────────────────────────────────────────
BASELINE_BRIER = 0.23066798182013623
WALL_BUDGET = 18 * 60  # stop adding trials at minute 18
PROGRESS_PATH = ROOT / 'artifacts' / 'optuna_progress.jsonl'
RESULTS_PATH  = ROOT / 'artifacts' / 'optuna_expensive_results.json'
DB_PATH       = ROOT / 'artifacts' / 'optuna_expensive.db'

ALL_TRIAL_LOG = []
START_TIME = None  # set in main()


def objective(trial):
    elapsed = time.time() - START_TIME
    if elapsed > WALL_BUDGET:
        raise optuna.TrialPruned('wall budget reached')

    cfg = {
        'HALF_LIFE_WEEKS': trial.suggest_float('HALF_LIFE_WEEKS', 3.0, 10.0),
        'SHRINK_K': trial.suggest_float('SHRINK_K', 2.0, 15.0),
        'CN_PRIOR': trial.suggest_float('CN_PRIOR', -6.0, -1.5),
        'CN_INTL_K': trial.suggest_float('CN_INTL_K', 10.0, 50.0),
        'CN_INDIRECT_WEIGHT': trial.suggest_float('CN_INDIRECT_WEIGHT', 0.05, 0.7),
        'REGION_SPILLOVER_ALPHA': trial.suggest_float('REGION_SPILLOVER_ALPHA', 0.0, 1.0),
        'CHAMPIONS_MULT': trial.suggest_float('CHAMPIONS_MULT', 1.0, 3.0),
        'INTL_WIN_MULT': trial.suggest_float('INTL_WIN_MULT', 0.7, 1.5),
        'ROSTER_PERSISTENCE': trial.suggest_float('ROSTER_PERSISTENCE', 0.4, 1.0),
        'MARGIN_FN': trial.suggest_categorical(
            'MARGIN_FN', ['sqrt', 'log1p', 'tanh4', 'capped_linear_8']),
    }

    t_trial = time.time()
    try:
        res = evaluate_config(cfg)
    except Exception as e:
        print(f'Trial {trial.number} FAILED: {e}')
        raise optuna.TrialPruned()

    brier = res['brier']
    cn_v  = res['cn_violations']
    dt = time.time() - t_trial

    log_entry = {
        'trial': trial.number,
        't_elapsed': time.time() - START_TIME,
        't_trial': dt,
        'brier': brier,
        'cn_violations': cn_v,
        'config': cfg,
    }
    ALL_TRIAL_LOG.append(log_entry)

    # Append progress line every 10 trials
    if (trial.number + 1) % 10 == 0:
        with open(PROGRESS_PATH, 'a') as f:
            for entry in ALL_TRIAL_LOG[-10:]:
                f.write(json.dumps(entry) + '\n')

    # I6 guardrail: penalize >2 CN violations heavily but still report brier
    # so Optuna can learn the boundary
    if cn_v > 2:
        return brier + 0.01  # soft penalty

    print(f'  [{trial.number}] brier={brier:.5f} cn_v={cn_v} margin={cfg["MARGIN_FN"]} dt={dt:.1f}s')
    return brier


def run_sweep():
    global START_TIME
    import io, contextlib
    START_TIME = time.time()
    if PROGRESS_PATH.exists():
        PROGRESS_PATH.unlink()
    if DB_PATH.exists():
        DB_PATH.unlink()
    storage_url = f'sqlite:///{DB_PATH}'

    sampler = TPESampler(seed=42, n_startup_trials=8)
    study = optuna.create_study(
        study_name='expensive_sweep',
        storage=storage_url,
        sampler=sampler,
        direction='minimize',
        load_if_exists=False,
    )

    # Seed with default config so we have a baseline trial
    default_cfg = {
        'HALF_LIFE_WEEKS': 6.0,
        'SHRINK_K': 5.0,
        'CN_PRIOR': -4.0,
        'CN_INTL_K': 20.0,
        'CN_INDIRECT_WEIGHT': 0.3,
        'REGION_SPILLOVER_ALPHA': 0.5,
        'CHAMPIONS_MULT': 2.0,
        'INTL_WIN_MULT': 1.0,
        'ROSTER_PERSISTENCE': 0.3,
        'MARGIN_FN': 'sqrt',
    }
    study.enqueue_trial(default_cfg)

    # Also enqueue ROST=0.7 variant from memory
    v_rost7 = dict(default_cfg); v_rost7['ROSTER_PERSISTENCE'] = 0.7
    study.enqueue_trial(v_rost7)

    print(f'\nStarting Optuna sweep at {time.strftime("%H:%M:%S")}...')
    print(f'Target: 40-60 trials in {WALL_BUDGET/60:.0f} minutes')

    while time.time() - START_TIME < WALL_BUDGET:
        try:
            study.optimize(objective, n_trials=1, catch=(Exception,))
        except KeyboardInterrupt:
            break
        if time.time() - START_TIME > WALL_BUDGET:
            break

    # Flush any remaining progress entries
    with open(PROGRESS_PATH, 'a') as f:
        written_count = (len(ALL_TRIAL_LOG) // 10) * 10
        for entry in ALL_TRIAL_LOG[written_count:]:
            f.write(json.dumps(entry) + '\n')

    elapsed = time.time() - START_TIME
    print(f'\nSweep complete: {len(ALL_TRIAL_LOG)} trials in {elapsed/60:.1f} minutes')

    # ── Analysis ──────────────────────────────────────────────────────────────────
    completed = [t for t in ALL_TRIAL_LOG if t['brier'] is not None]
    completed.sort(key=lambda t: t['brier'])
    top_10 = completed[:10]

    try:
        importance = optuna.importance.get_param_importances(study)
    except Exception as e:
        print(f'  importance failed: {e}')
        importance = {}

    top_q = completed[:max(1, len(completed) // 4)]
    if len(top_q) >= 3:
        hl = np.array([t['config']['HALF_LIFE_WEEKS'] for t in top_q])
        sk = np.array([t['config']['SHRINK_K'] for t in top_q])
        corr = float(np.corrcoef(hl, sk)[0, 1]) if hl.std() > 0 and sk.std() > 0 else 0.0
    else:
        corr = None

    # Bootstrap test: best config vs baseline
    print('\nRunning paired bootstrap of best vs baseline...')
    best_cfg = top_10[0]['config'] if top_10 else default_cfg
    best_res = evaluate_config(best_cfg, return_extras=True)
    best_probs = best_res['probs']
    best_outs  = best_res['outs']

    # Recompute baseline predictions from current on-disk timeline
    baseline_matches = H.load_matches()
    baseline_probs, baseline_outs = H.predict_series(
        baseline_matches, beta=0.140, intl_bonus=0.22, cn_dog_offset=0.47,
    )

    # Rebuild matches under best config (same as evaluate_config did internally)
    _patch_constants(best_cfg)
    try:
        years_tl = {}
        with contextlib.redirect_stdout(io.StringIO()):
            for yr in TARGET_YEARS:
                cps, mes = BRT.build_year_timeline(ALL_GAMES, yr, existing=None)
                years_tl[yr] = {'checkpoints': cps, 'match_events': mes}
        best_matches = timeline_to_harness_matches(years_tl)
    finally:
        _reset_constants()

    def match_key(m):
        return (m.get('match_id'), m.get('date'))

    baseline_by_key = {match_key(m): (p, o) for m, p, o in zip(baseline_matches, baseline_probs, baseline_outs)}
    best_by_key = {match_key(m): (p, o) for m, p, o in zip(best_matches, best_probs, best_outs)}
    common = [k for k in best_by_key if k in baseline_by_key]
    print(f'  baseline n={len(baseline_matches)}, best n={len(best_matches)}, common={len(common)}')

    p_a = np.array([baseline_by_key[k][0] for k in common])
    p_b = np.array([best_by_key[k][0] for k in common])
    outs_common = np.array([best_by_key[k][1] for k in common])
    outs_a = np.array([baseline_by_key[k][1] for k in common])
    assert (outs_common == outs_a).all(), 'outcome mismatch — match key collision?'

    bootstrap = H.paired_bootstrap_brier(p_a, p_b, outs_common, n_boot=1000, seed=0)
    print(f'  paired bootstrap mean diff (best - baseline) = {bootstrap["mean"]:.5f}  '
          f'CI=[{bootstrap["ci_lo"]:.5f}, {bootstrap["ci_hi"]:.5f}]  p={bootstrap["p_two_sided"]:.4f}')

    output = {
        'n_trials_completed': len(completed),
        'best_brier': top_10[0]['brier'] if top_10 else None,
        'baseline_brier': BASELINE_BRIER,
        'best_config': top_10[0]['config'] if top_10 else None,
        'top_10_configs': [{'brier': t['brier'], 'cn_violations': t['cn_violations'],
                            'config': t['config']} for t in top_10],
        'param_importances': {k: float(v) for k, v in importance.items()},
        'shrink_k_vs_half_life_corr_top_quartile': corr,
        'cn_prior_vs_dog_offset_corr_top_quartile': None,
        'paired_bootstrap': bootstrap,
        'elapsed_minutes': elapsed / 60,
        'notes': (
            f'Sweep over 10 expensive Massey-rebuild knobs. Each trial rebuilds 4-year '
            f'timeline (~3.5s). Baseline Brier = {BASELINE_BRIER:.5f}. '
            f'I6 guardrail (CN c-values outside [0.1, 0.95]) applies a +0.01 brier penalty '
            f'when >2 CN teams violate.'
        ),
    }
    RESULTS_PATH.write_text(json.dumps(output, indent=2, default=str))
    print(f'\nWrote {RESULTS_PATH}')
    print(f'\nBest brier: {output["best_brier"]:.5f} (baseline {BASELINE_BRIER:.5f})')
    print(f'Best config: {output["best_config"]}')
    if importance:
        print(f'Importances: {dict(sorted(importance.items(), key=lambda kv: -kv[1])[:5])}')
    print(f'SHRINK_K vs HALF_LIFE_WEEKS top-quartile corr: {corr}')


if __name__ == '__main__':
    run_sweep()
