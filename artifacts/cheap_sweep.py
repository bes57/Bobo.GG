"""
Phase 1a: Cheap-knob sweep.

Knobs: BETA, BETA_BO5, INTL_EXP_BONUS, CN_DOG_OFFSET.
All applied at prediction time. No rebuild needed.

70/30 temporal split:
  - Train: matches before cutoff (used for parameter selection)
  - Test:  matches after cutoff  (held out, only evaluated at the end)

Plus full-data sweep for comparison.
"""
from __future__ import annotations
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import optuna
from optuna.samplers import TPESampler

sys.path.insert(0, '/Users/benny_es1/PythonTest/artifacts')
import harness as H

# ───────────────────────────────────────────────────────────────────
# Setup
# ───────────────────────────────────────────────────────────────────
matches = H.load_matches()
matches.sort(key=lambda m: m['date'])
N = len(matches)
cutoff_idx = int(0.7 * N)
cutoff_date = matches[cutoff_idx]['date']
train = matches[:cutoff_idx]
test  = matches[cutoff_idx:]
print(f'n_total={N}  train={len(train)} (≤ {matches[cutoff_idx-1]["date"]})  test={len(test)} (≥ {cutoff_date})')


def eval_brier(matches_subset, beta, beta_bo5, intl_bonus, cn_dog_offset):
    """Cheap: just re-predict with new constants."""
    probs, outs = H.predict_series(
        matches_subset,
        beta=beta,
        beta_bo5=beta_bo5 if beta_bo5 != beta else None,
        intl_bonus=intl_bonus,
        cn_dog_offset=cn_dog_offset,
    )
    return H.brier(probs, outs), probs, outs


def objective(trial):
    beta          = trial.suggest_float('BETA',          0.08, 0.22)
    beta_bo5      = trial.suggest_float('BETA_BO5',      0.08, 0.22)
    intl_bonus    = trial.suggest_float('INTL_EXP_BONUS', 0.0, 0.5)
    cn_dog_offset = trial.suggest_float('CN_DOG_OFFSET',  0.0, 0.8)
    br, _, _ = eval_brier(train, beta, beta_bo5, intl_bonus, cn_dog_offset)
    return br


optuna.logging.set_verbosity(optuna.logging.WARNING)
sampler = TPESampler(seed=42, n_startup_trials=50)
study = optuna.create_study(direction='minimize', sampler=sampler)
study.optimize(objective, n_trials=3000, show_progress_bar=False)

print(f'\nbest train brier: {study.best_value:.5f}')
print(f'best params:      {study.best_params}')

# Held-out test set evaluation
bp = study.best_params
bp_kw = dict(beta=bp['BETA'], beta_bo5=bp['BETA_BO5'],
             intl_bonus=bp['INTL_EXP_BONUS'], cn_dog_offset=bp['CN_DOG_OFFSET'])
train_br_best, _, _ = eval_brier(train, **bp_kw)
test_br_best, p_test, o_test = eval_brier(test, **bp_kw)
train_br_base, _, _ = eval_brier(train, beta=0.140, beta_bo5=0.140, intl_bonus=0.22, cn_dog_offset=0.47)
test_br_base, p_test_base, o_test_base = eval_brier(test, beta=0.140, beta_bo5=0.140, intl_bonus=0.22, cn_dog_offset=0.47)

print(f'\nTrain:  baseline={train_br_base:.5f}  best={train_br_best:.5f}  Δ={train_br_best-train_br_base:+.5f}')
print(f'Test:   baseline={test_br_base:.5f}  best={test_br_best:.5f}  Δ={test_br_best-test_br_base:+.5f}')

# Paired bootstrap on test set
btr = H.paired_bootstrap_brier(p_test_base, p_test, o_test, n_boot=1000, seed=1)
print(f'Paired test bootstrap Δ Brier (new − baseline): {btr["mean"]:+.5f}  95% CI ({btr["ci_lo"]:+.5f}, {btr["ci_hi"]:+.5f})  p={btr["p_two_sided"]:.3f}')

# Full-set evaluation
full_br_best, p_full, o_full = eval_brier(matches, **bp_kw)
full_br_base, p_full_base, _ = eval_brier(matches, beta=0.140, beta_bo5=0.140, intl_bonus=0.22, cn_dog_offset=0.47)
btr_full = H.paired_bootstrap_brier(p_full_base, p_full, o_full, n_boot=1000, seed=2)
print(f'Full:   baseline={full_br_base:.5f}  best={full_br_best:.5f}  Δ={full_br_best-full_br_base:+.5f}  p={btr_full["p_two_sided"]:.3f}')

# Platt slope of best on test set
a_best, b_best = H.platt_slope(p_test, o_test)
a_base, b_base = H.platt_slope(p_test_base, o_test_base)
print(f'Test Platt: baseline a={a_base:+.3f} b={b_base:.4f} | best a={a_best:+.3f} b={b_best:.4f}')

# Save
out = {
    'n_trials': len(study.trials),
    'best_params': bp,
    'best_train_brier': float(train_br_best),
    'best_test_brier':  float(test_br_best),
    'best_full_brier':  float(full_br_best),
    'baseline_train_brier': float(train_br_base),
    'baseline_test_brier':  float(test_br_base),
    'baseline_full_brier':  float(full_br_base),
    'test_bootstrap':  btr,
    'full_bootstrap':  btr_full,
    'test_platt_slope_baseline': b_base,
    'test_platt_slope_best':     b_best,
    'cutoff_date': cutoff_date,
}
Path('/Users/benny_es1/PythonTest/artifacts').mkdir(exist_ok=True)
json.dump(out, open('/Users/benny_es1/PythonTest/artifacts/cheap_sweep.json', 'w'), indent=2, default=float)
print('\nSaved → artifacts/cheap_sweep.json')

# Top-10 configs (excluding the best, for ensemble use)
trials_sorted = sorted([t for t in study.trials if t.value is not None], key=lambda t: t.value)
top10 = [{'rank': i+1, 'train_brier': t.value, 'params': t.params}
         for i, t in enumerate(trials_sorted[:10])]
json.dump(top10, open('/Users/benny_es1/PythonTest/artifacts/cheap_sweep_top10.json', 'w'), indent=2)
