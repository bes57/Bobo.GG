"""Validate the best Optuna expensive-sweep config on held-out test split + sanity."""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path
import numpy as np

ROOT = Path('/Users/benny_es1/PythonTest')
sys.path.insert(0, str(ROOT / 'artifacts'))
sys.path.insert(0, str(ROOT / 'scrapers'))
sys.path.insert(0, str(ROOT))

# Re-use the subagent's evaluate_config function (full rebuild + harness eval)
from expensive_sweep import evaluate_config, ALL_GAMES, timeline_to_harness_matches
import harness as H

# Best trial from progress (cn_violations<=2)
trials = [json.loads(l) for l in open(ROOT / 'artifacts' / 'optuna_progress.jsonl')]
trials_ok = [t for t in trials if t.get('cn_violations', 99) <= 2]
trials_ok.sort(key=lambda t: t['brier'])
print(f'Trials with cn_violations<=2: {len(trials_ok)} / {len(trials)}')

# Top 3 candidates to validate
candidates = trials_ok[:3]
print(f'\nTop 3 candidates (in-sample Brier):')
for i, t in enumerate(candidates):
    print(f'  #{i+1}: trial {t["trial"]}, in-sample brier {t["brier"]:.5f}, cnv {t["cn_violations"]}')


def evaluate_with_split(cfg, frac=0.7):
    r = evaluate_config(cfg, return_extras=True)
    probs, outs = r['probs'], r['outs']
    n = len(probs)
    cut = int(frac * n)
    train_b = float(H.brier(probs[:cut], outs[:cut]))
    test_b = float(H.brier(probs[cut:], outs[cut:]))
    full_b = float(H.brier(probs, outs))
    return {
        'train_brier': train_b,
        'test_brier': test_b,
        'full_brier': full_b,
        'cn_violations': r['cn_violations'],
        'cn_c_vals': r['cn_c_vals'],
        'n_train': cut, 'n_test': n - cut,
        'probs': probs, 'outs': outs,
    }


# Baseline reference
print('\nBaseline (current constants):')
baseline_matches = H.load_matches()
p_base, o_base = H.predict_series(baseline_matches)
n = len(baseline_matches)
cut = int(0.7 * n)
print(f'  train Brier (n={cut}):    {H.brier(p_base[:cut], o_base[:cut]):.5f}')
print(f'  test  Brier (n={n-cut}):  {H.brier(p_base[cut:], o_base[cut:]):.5f}')
print(f'  full  Brier:               {H.brier(p_base, o_base):.5f}')
base_train = float(H.brier(p_base[:cut], o_base[:cut]))
base_test = float(H.brier(p_base[cut:], o_base[cut:]))

# Validate top candidates
results = []
for i, t in enumerate(candidates):
    print(f'\n=== Candidate #{i+1} (trial {t["trial"]}) ===')
    cfg = t['config']
    cfg['CN_C_MIN'] = 0.0  # match expensive_sweep's default
    cfg['INTL_LOSS_MULT'] = 1.0
    cfg['RD_TRANSFORM'] = 'sqrt' if cfg['MARGIN_FN'] == 'sqrt' else cfg['MARGIN_FN']
    cfg['RD_POWER'] = 0.5
    cfg['RD_SCALE'] = 2.5
    print('Config:', {k: round(v, 3) if isinstance(v, float) else v for k, v in cfg.items()})
    t0 = time.time()
    r = evaluate_with_split(cfg)
    print(f'  rebuild + eval: {time.time()-t0:.1f}s')
    print(f'  train Brier (n={r["n_train"]}):  {r["train_brier"]:.5f}  (Δ vs base: {r["train_brier"]-base_train:+.5f})')
    print(f'  test  Brier (n={r["n_test"]}):  {r["test_brier"]:.5f}  (Δ vs base: {r["test_brier"]-base_test:+.5f})')
    print(f'  full  Brier:                    {r["full_brier"]:.5f}')
    print(f'  CN violations: {r["cn_violations"]}')

    # Paired bootstrap on test set
    test_idx = slice(cut, n)
    btr = H.paired_bootstrap_brier(
        p_base[test_idx], r['probs'][test_idx], o_base[test_idx],
        n_boot=1000, seed=i,
    )
    print(f'  paired bootstrap (test): mean Δ {btr["mean"]:+.5f}, 95% CI ({btr["ci_lo"]:+.5f}, {btr["ci_hi"]:+.5f}), p={btr["p_two_sided"]:.3f}')

    results.append({
        'trial': t['trial'],
        'config': cfg,
        'baseline_train': base_train,
        'baseline_test': base_test,
        'baseline_full': float(H.brier(p_base, o_base)),
        'candidate_train': r['train_brier'],
        'candidate_test': r['test_brier'],
        'candidate_full': r['full_brier'],
        'cn_violations': r['cn_violations'],
        'test_bootstrap': btr,
    })

# Pick winner: best held-out test Brier with cn_violations <= 2
winners = sorted(results, key=lambda r: r['candidate_test'])
print('\nRanked by test Brier:')
for r in winners:
    print(f'  trial {r["trial"]}: test={r["candidate_test"]:.5f}, Δ test={r["candidate_test"]-base_test:+.5f}, p={r["test_bootstrap"]["p_two_sided"]:.3f}')

json.dump({'baseline_test_brier': base_test, 'candidates': results},
          open(ROOT / 'artifacts' / 'expensive_validation.json', 'w'),
          indent=2, default=float)
print('\nSaved → artifacts/expensive_validation.json')
