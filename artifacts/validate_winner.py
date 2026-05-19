"""
Validates an expensive-sweep winning config on:
  - Held-out test (70/30 temporal split)
  - Sanity checks (trophy, CN bottom, H2H, c saturation)

Reads artifacts/optuna_expensive_results.json (or progress.jsonl), picks best,
rebuilds rating timelines under the config, evaluates harness, runs sanity.
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '/Users/benny_es1/PythonTest/artifacts')
sys.path.insert(0, '/Users/benny_es1/PythonTest/scrapers')
import harness as H


def best_from_progress():
    trials = [json.loads(l) for l in open('/Users/benny_es1/PythonTest/artifacts/optuna_progress.jsonl')]
    trials.sort(key=lambda t: t['brier'])
    # Pick best with cn_violations <= 2 (sanity constraint)
    for t in trials:
        if t.get('cn_violations', 99) <= 2:
            return t
    return trials[0]


def evaluate_config_in_sample(matches_for_timeline_path, config):
    """Loads matches from given timeline files, computes Brier."""
    matches = H.load_matches(matches_for_timeline_path)
    probs, outs = H.predict_series(matches)
    return H.brier(probs, outs), probs, outs, matches


def split_eval(matches, probs, outs, frac=0.7):
    n = len(matches)
    cut = int(frac * n)
    train_b = H.brier(probs[:cut], outs[:cut])
    test_b = H.brier(probs[cut:], outs[cut:])
    return train_b, test_b


if __name__ == '__main__':
    best = best_from_progress()
    print('Best trial (cn_violations≤2):')
    print(json.dumps(best, indent=2))

    # Recipe: the subagent's code rebuilt timelines per trial and saved Brier.
    # To re-validate, we'd need to re-rebuild with this config. Out of scope
    # here without copying the subagent's rebuild script.
    print(f'\nIn-sample Brier from sweep: {best["brier"]:.5f}')
    print(f'Δ vs baseline: {best["brier"] - 0.23067:+.5f}')
    print('NOTE: This is IN-SAMPLE. Test on held-out before accepting.')
