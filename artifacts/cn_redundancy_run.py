"""
CN redundancy experiment: test whether CN_PRIOR and CN_DOG_OFFSET are partially
redundant by running 3 configs and comparing series Brier on CN-touching intl
matches and on all matches.
"""
from __future__ import annotations
import json
import os
import sys
import shutil
from pathlib import Path

import numpy as np

ROOT = Path('/Users/benny_es1/PythonTest')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scrapers'))
sys.path.insert(0, str(ROOT / 'artifacts'))

import harness
from harness import (
    load_matches, predict_series, brier, paired_bootstrap_brier
)

DATA_DIR = ROOT / 'data'
TMP_DIR = Path('/tmp')

# Targets for alt rebuild outputs
ALT_NAMES = {
    2023: 'rating_timeline_2023.json',
    2024: 'rating_timeline_2024.json',
    2025: 'rating_timeline_2025.json',
    2026: 'rating_timeline.json',
}


def rebuild_timelines_with_cn_prior(cn_prior_value, tag):
    """Rebuild rating_timeline*.json with the given CN_PRIOR. Writes the alt
    timelines to /tmp/<tag>/ and returns the list of those file paths."""
    import importlib
    # Fresh-import the builder modules so we can monkey-patch and reload
    if 'BuildMapRatings' in sys.modules:
        del sys.modules['BuildMapRatings']
    if 'BuildRatingTimeline' in sys.modules:
        del sys.modules['BuildRatingTimeline']

    import BuildMapRatings
    BuildMapRatings.CN_PRIOR = cn_prior_value
    # Force reload of BuildRatingTimeline so it picks up patched module
    import BuildRatingTimeline
    # Patch local CN_PRIOR reference inside BuildRatingTimeline too
    BuildRatingTimeline.CN_PRIOR = cn_prior_value

    out_dir = TMP_DIR / tag
    out_dir.mkdir(exist_ok=True)

    print(f"\n=== Rebuilding timelines with CN_PRIOR={cn_prior_value} → {out_dir} ===")
    all_games = BuildRatingTimeline.load_all_games()
    print(f"Loaded {len(all_games)} map games")

    out_paths = []
    for year in [2023, 2024, 2025, 2026]:
        print(f"  Building {year}...")
        checkpoints, match_events = BuildRatingTimeline.build_year_timeline(
            all_games, year, existing=None
        )
        out = {
            "year": year,
            "lambda_decay": round(BuildRatingTimeline.LAMBDA_DECAY, 6),
            "checkpoints": checkpoints,
            "match_events": match_events,
            "generated": "experiment",
        }
        out_path = out_dir / ALT_NAMES[year]
        with open(out_path, "w") as f:
            json.dump(out, f, separators=(",", ":"))
        print(f"    saved {len(match_events)} match events to {out_path.name}")
        out_paths.append(out_path)

    return out_paths


def compute_cfg_metrics(timeline_files, cn_dog_offset, label):
    print(f"\n=== [{label}] timelines={[p.name for p in timeline_files]}, cn_dog_offset={cn_dog_offset} ===")
    matches = load_matches(timeline_files=timeline_files)
    probs, outs = predict_series(matches, cn_dog_offset=cn_dog_offset)

    cn_intl_idx = [
        i for i, m in enumerate(matches)
        if m['intl'] and (m['fav_region'] == 'CN' or m['dog_region'] == 'CN')
    ]
    cn_probs = probs[cn_intl_idx]
    cn_outs = outs[cn_intl_idx]

    res = {
        'all_brier': brier(probs, outs),
        'cn_intl_brier': brier(cn_probs, cn_outs),
        'n_all': int(len(probs)),
        'n_cn_intl': int(len(cn_intl_idx)),
        'probs_all': probs,
        'outs_all': outs,
        'probs_cn': cn_probs,
        'outs_cn': cn_outs,
    }
    print(f"  all: n={res['n_all']}, brier={res['all_brier']:.5f}")
    print(f"  cn_intl: n={res['n_cn_intl']}, brier={res['cn_intl_brier']:.5f}")
    return res


def main():
    # CONFIG 1: current state (existing timelines, cn_dog_offset=0.47)
    current_timelines = [
        DATA_DIR / 'rating_timeline.json',
        DATA_DIR / 'rating_timeline_2023.json',
        DATA_DIR / 'rating_timeline_2024.json',
        DATA_DIR / 'rating_timeline_2025.json',
    ]
    r1 = compute_cfg_metrics(current_timelines, cn_dog_offset=0.47, label="1_current_split")

    # CONFIG 2: CN_PRIOR=-6.0, offset=0.0
    cfg2_files = rebuild_timelines_with_cn_prior(-6.0, 'cn_prior_neg6')
    # cfg2_files comes back in [2023, 2024, 2025, 2026] order; harness expects
    # any list. The order matters only for attendance lookup keys; both ways
    # work since lookups are global.
    r2 = compute_cfg_metrics(cfg2_files, cn_dog_offset=0.0, label="2_rating_only")

    # CONFIG 3: CN_PRIOR=-2.0, offset=0.94
    cfg3_files = rebuild_timelines_with_cn_prior(-2.0, 'cn_prior_neg2')
    r3 = compute_cfg_metrics(cfg3_files, cn_dog_offset=0.94, label="3_offset_only")

    # Pick winners
    cfgs = {
        '1_current_split': r1,
        '2_rating_only': r2,
        '3_offset_only': r3,
    }
    winner_all = min(cfgs, key=lambda k: cfgs[k]['all_brier'])
    winner_cn = min(cfgs, key=lambda k: cfgs[k]['cn_intl_brier'])

    # Paired bootstrap: winner_all vs next-best on all matches
    ordered_all = sorted(cfgs.items(), key=lambda kv: kv[1]['all_brier'])
    best_all, runner_all = ordered_all[0], ordered_all[1]
    pb_all = paired_bootstrap_brier(
        best_all[1]['probs_all'], runner_all[1]['probs_all'],
        best_all[1]['outs_all'].astype(float)
    )

    ordered_cn = sorted(cfgs.items(), key=lambda kv: kv[1]['cn_intl_brier'])
    best_cn, runner_cn = ordered_cn[0], ordered_cn[1]
    pb_cn = paired_bootstrap_brier(
        best_cn[1]['probs_cn'], runner_cn[1]['probs_cn'],
        best_cn[1]['outs_cn'].astype(float)
    )

    # Verdict
    if winner_all == '1_current_split':
        verdict = 'principled-split'
        rec_prior, rec_offset = -4.0, 0.47
    elif winner_all == '2_rating_only':
        verdict = 'drop-offset'
        rec_prior, rec_offset = -6.0, 0.0
    elif winner_all == '3_offset_only':
        verdict = 'drop-rating'
        rec_prior, rec_offset = -2.0, 0.94
    else:
        verdict = 'ambiguous'
        rec_prior, rec_offset = -4.0, 0.47

    # If the all-data win is tiny and cn_intl winner differs, flag ambiguous
    spread_all = ordered_all[-1][1]['all_brier'] - ordered_all[0][1]['all_brier']
    if spread_all < 1e-5 and winner_all != winner_cn:
        verdict = 'ambiguous'

    output = {
        'configs': {
            '1_current_split': {
                'cn_intl_brier': r1['cn_intl_brier'],
                'all_brier':     r1['all_brier'],
                'n_cn_intl':     r1['n_cn_intl'],
                'n_all':         r1['n_all'],
            },
            '2_rating_only': {
                'cn_intl_brier': r2['cn_intl_brier'],
                'all_brier':     r2['all_brier'],
                'n_cn_intl':     r2['n_cn_intl'],
                'n_all':         r2['n_all'],
            },
            '3_offset_only': {
                'cn_intl_brier': r3['cn_intl_brier'],
                'all_brier':     r3['all_brier'],
                'n_cn_intl':     r3['n_cn_intl'],
                'n_all':         r3['n_all'],
            },
        },
        'winner_all': winner_all,
        'winner_cn_intl': winner_cn,
        'redundancy_verdict': verdict,
        'recommended_cn_prior': rec_prior,
        'recommended_cn_dog_offset': rec_offset,
        'paired_bootstrap_all': {
            'winner': best_all[0],
            'runner_up': runner_all[0],
            **pb_all,
        },
        'paired_bootstrap_cn_intl': {
            'winner': best_cn[0],
            'runner_up': runner_cn[0],
            **pb_cn,
        },
        'notes': (
            f"Winner on all (n={r1['n_all']}): {winner_all} "
            f"(Brier={cfgs[winner_all]['all_brier']:.5f}); "
            f"Winner on CN-intl (n={r1['n_cn_intl']}): {winner_cn} "
            f"(Brier={cfgs[winner_cn]['cn_intl_brier']:.5f}). "
            f"Paired-bootstrap p (all, winner vs runner-up) = {pb_all['p_two_sided']:.4f}; "
            f"p (CN-intl, winner vs runner-up) = {pb_cn['p_two_sided']:.4f}."
        ),
    }

    out_file = ROOT / 'artifacts' / 'cn_redundancy.json'
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {out_file}")
    print(json.dumps({k: v for k, v in output.items() if k != 'notes'}, indent=2))
    print(output['notes'])


if __name__ == '__main__':
    main()
