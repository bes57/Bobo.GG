"""
Fix the CN_PRIOR/CN_INTL_K sweep (Python default-args bug) and verify the
REGION_SPILLOVER_ALPHA finding from SecondaryParamSweep.

REGION_SPILLOVER_ALPHA changes DO show real effects — 0.5 looks like a clean
upgrade. Verify, then test joint moves to confirm.
"""
import os, sys, math, json, importlib, contextlib, io
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scrapers'))

from BacktestSeriesPredictions import (
    load_match_data, load_snapshots, find_snapshot_for, brier, logloss,
    expected_calibration_error,
)

DEPLOYED = dict(
    RD_TRANSFORM='power', RD_POWER=0.5, RD_SCALE=2.0,
    HALF_LIFE_WEEKS=6.0, ROSTER_PERSISTENCE=0.3,
    INTL_WIN_MULT=1.0, INTL_LOSS_MULT=1.0, CHAMPIONS_MULT=2.0,
    CN_PRIOR=-2.0, CN_INTL_K=4.0, CN_C_MIN=0.5,
    REGION_SPILLOVER_ALPHA=0.30,
)
BETA = 0.17


def predict(rA, rB, beta=BETA):
    p = 1.0 / (1.0 + math.exp(-beta * (rA - rB)))
    return p**2 * (3 - 2*p)


def gen(matches, snaps, beta=BETA):
    p, y = [], []
    for _, m in matches.iterrows():
        s = find_snapshot_for(m['date'], snaps)
        if not s: continue
        _, _, _, ratings = s
        a, b = m['a'], m['b']
        if a not in ratings or b not in ratings: continue
        rA = ratings[a].get('overall_rating', 0.0) if isinstance(ratings[a], dict) else ratings[a]
        rB = ratings[b].get('overall_rating', 0.0) if isinstance(ratings[b], dict) else ratings[b]
        p.append(predict(rA, rB, beta)); y.append(m['a_wins'])
    return np.array(p), np.array(y)


def platt(p, y, iters=2000, lr=0.05):
    eps = 1e-6
    pc = np.clip(p, eps, 1-eps)
    z = np.log(pc/(1-pc))
    A, B = 1.0, 0.0
    for _ in range(iters):
        pred = 1/(1+np.exp(-(A*z + B)))
        err = pred - y
        A -= lr * (err * z).mean()
        B -= lr * err.mean()
    return float(A), float(B)


def rebuild(**overrides):
    """Rebuild with proper handling of default-arg parameters."""
    saved_argv = sys.argv[:]
    sys.argv = ['BuildMapRatings.py']
    try:
        import BuildMapRatings
        importlib.reload(BuildMapRatings)
        cfg = dict(DEPLOYED, **overrides)
        for k, v in cfg.items():
            setattr(BuildMapRatings, k, v)
        # Patch __defaults__ for functions that bind CN_* / SHRINK_* as default args
        # _apply_cn_shrinkage(ratings, intl_weights, prior=CN_PRIOR, K=CN_INTL_K, c_min=CN_C_MIN)
        BuildMapRatings._apply_cn_shrinkage.__defaults__ = (
            cfg['CN_PRIOR'], cfg['CN_INTL_K'], cfg['CN_C_MIN']
        )
        with contextlib.redirect_stdout(io.StringIO()):
            BuildMapRatings.main()
    finally:
        sys.argv = saved_argv


def winner_ranks(mr_dict):
    SNAPS = [(2024,'after_madrid','SEN'),(2024,'after_shanghai','GEN'),
             (2024,'after_champions','EDG'),(2025,'after_bangkok','T1'),
             (2025,'after_toronto','PRX'),(2025,'after_champions','NRG'),
             (2026,'after_santiago','NS')]
    rks = []
    for year, snap, winner in SNAPS:
        s = mr_dict['ratings'][str(year)]['snapshots'].get(snap, {}).get('teams', {})
        if not s: rks.append(50); continue
        items = sorted(s.items(), key=lambda x:-x[1]['overall_rating'])
        rk = next((i+1 for i,(t,_) in enumerate(items) if t==winner), 50)
        rks.append(rk)
    return rks


def cn_team_count_below(mr_dict, threshold=-2.5):
    """How many CN teams sit below threshold (deeper = more negative)?"""
    CN = {'EDG','BLG','TE','DRG','ASE','AG','XLG','WOL','FPX','JDG','NOVA','TEC','TYL','TYLOO'}
    teams = mr_dict['ratings']['2026']['snapshots']['after_santiago']['teams']
    cn_r = [teams[t]['overall_rating'] for t in teams if t in CN]
    return sum(1 for r in cn_r if r < threshold), min(cn_r) if cn_r else 0


def evaluate(matches, **overrides):
    rebuild(**overrides)
    snaps = load_snapshots()
    with open(os.path.join(ROOT, 'data/map_ratings.json')) as f:
        mr = json.load(f)
    teams26 = mr['ratings']['2026']['snapshots']['after_santiago']['teams']
    top_26 = max(t['overall_rating'] for t in teams26.values())
    bot_26 = min(t['overall_rating'] for t in teams26.values())
    cn_below, cn_min = cn_team_count_below(mr)
    trophy_avg = float(np.mean(winner_ranks(mr)))
    matches_tr = matches[matches['date'] < '2026-01-01'].reset_index(drop=True)
    matches_te = matches[matches['date'] >= '2026-01-01'].reset_index(drop=True)
    p, y = gen(matches, snaps)
    p_tr, y_tr = gen(matches_tr, snaps)
    p_te, y_te = gen(matches_te, snaps)
    A, _ = platt(p, y)
    return {
        'top26': top_26, 'bot26': bot_26, 'cn_below_-2.5': cn_below, 'cn_min': cn_min,
        'trophy_avg': trophy_avg,
        'brier_full': float(brier(p, y)),
        'brier_tr':   float(brier(p_tr, y_tr)),
        'brier_te':   float(brier(p_te, y_te)) if len(p_te) >= 30 else float('nan'),
        'platt_A':    A,
        'ece':        float(expected_calibration_error(p, y)),
        'sharp':      float(np.abs(p-0.5).mean()),
    }


def fmt(r, label):
    return (f"  {label:<18}  top26={r['top26']:>+5.2f}  bot26={r['bot26']:>+5.2f}  "
            f"cn_min={r['cn_min']:>+5.2f}  troph={r['trophy_avg']:>4.2f}  "
            f"Brier_tr={r['brier_tr']:.5f}  Brier_te={r['brier_te']:.5f}  "
            f"A={r['platt_A']:>4.2f}  ECE={r['ece']:.4f}")


def main():
    matches = load_match_data()
    print(f"Loaded {len(matches)} series\n")

    # Baseline
    print("━━━ BASELINE (current deployed) ━━━")
    base = evaluate(matches)
    print(fmt(base, '(deployed)'))
    print()

    # === CN_PRIOR (fixed override) ===
    print("━━━ CN_PRIOR (proper override) ━━━")
    for cp in [-1.0, -1.5, -2.0, -2.5, -3.0, -4.0, -5.0]:
        r = evaluate(matches, CN_PRIOR=cp)
        print(fmt(r, f'CN_PRIOR={cp}'), flush=True)

    # === CN_INTL_K ===
    print("\n━━━ CN_INTL_K (proper override) ━━━")
    for ck in [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]:
        r = evaluate(matches, CN_INTL_K=ck)
        print(fmt(r, f'CN_INTL_K={ck}'), flush=True)

    # === CN_C_MIN (the floor) ===
    print("\n━━━ CN_C_MIN (floor on shrinkage) ━━━")
    for cm in [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
        r = evaluate(matches, CN_C_MIN=cm)
        print(fmt(r, f'CN_C_MIN={cm}'), flush=True)

    # === REGION_SPILLOVER_ALPHA — re-verify finding ===
    print("\n━━━ REGION_SPILLOVER_ALPHA (re-verify the genuine improvement) ━━━")
    for a in [0.0, 0.15, 0.30, 0.40, 0.50, 0.60, 0.75, 1.0]:
        r = evaluate(matches, REGION_SPILLOVER_ALPHA=a)
        print(fmt(r, f'SPILL={a}'), flush=True)

    # === JOINT: SPILL=0.5 + CN_PRIOR variations ===
    print("\n━━━ JOINT: SPILL=0.5 + CN_PRIOR variations ━━━")
    for cp in [-2.0, -2.5, -3.0, -4.0]:
        r = evaluate(matches, REGION_SPILLOVER_ALPHA=0.5, CN_PRIOR=cp)
        print(fmt(r, f'SPILL=0.5 CN_PR={cp}'), flush=True)

    print(f"\n━━━ Restoring deployed config ━━━")
    rebuild()
    print("Done.")


if __name__ == '__main__':
    main()
