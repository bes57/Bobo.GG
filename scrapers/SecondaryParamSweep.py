"""
After deploying RD_POWER=0.5, RD_SCALE=2.0, β=0.17, the optimal positions of
other knobs may have shifted. Sweep each independently around its current
value to see if any want to move.

Each sweep holds everything else at the deployed value and varies one knob.
Track: Brier_full, Brier_train (≤2025), Brier_test (2026), Platt A, ECE,
sharpness, top_2026, top_2024, trophy_avg.

Output: for each knob, the value that minimizes Brier_train, and the value
that gives best calibration (|Platt-1| closest to 0). Flag if those agree
with the deployed value (no change) or suggest a move.
"""
import os, sys, math, json, time, importlib, contextlib, io
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scrapers'))

from BacktestSeriesPredictions import (
    load_match_data, load_snapshots, find_snapshot_for, brier, logloss,
    expected_calibration_error,
)

# Deployed values — baseline
DEPLOYED = dict(
    RD_TRANSFORM='power', RD_POWER=0.5, RD_SCALE=2.0,
    HALF_LIFE_WEEKS=6.0, ROSTER_PERSISTENCE=0.3,
    INTL_WIN_MULT=1.0, INTL_LOSS_MULT=1.0, CHAMPIONS_MULT=2.0,
    CN_PRIOR=-2.0, CN_INTL_K=4.0,
    REGION_SPILLOVER_ALPHA=0.30,
    SHRINK_K=12,
)
BETA = 0.17


def predict_series(rA, rB, beta=BETA, fmt='bo3'):
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
        p.append(predict_series(rA, rB, beta)); y.append(m['a_wins'])
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
    saved_argv = sys.argv[:]
    sys.argv = ['BuildMapRatings.py']
    try:
        import BuildMapRatings
        importlib.reload(BuildMapRatings)
        cfg = dict(DEPLOYED, **overrides)
        for k, v in cfg.items():
            setattr(BuildMapRatings, k, v)
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


def evaluate(matches, **overrides):
    rebuild(**overrides)
    snaps = load_snapshots()
    with open(os.path.join(ROOT, 'data/map_ratings.json')) as f:
        mr = json.load(f)
    top_26 = max(t['overall_rating'] for t in
                 mr['ratings']['2026']['snapshots']['after_santiago']['teams'].values())
    bot_26 = min(t['overall_rating'] for t in
                 mr['ratings']['2026']['snapshots']['after_santiago']['teams'].values())
    top_24 = max(t['overall_rating'] for t in
                 mr['ratings']['2024']['snapshots']['after_champions']['teams'].values())
    trophy_avg = float(np.mean(winner_ranks(mr)))
    matches_tr = matches[matches['date'] < '2026-01-01'].reset_index(drop=True)
    matches_te = matches[matches['date'] >= '2026-01-01'].reset_index(drop=True)
    p, y = gen(matches, snaps)
    p_tr, y_tr = gen(matches_tr, snaps)
    p_te, y_te = gen(matches_te, snaps)
    if len(p) < 100: return None
    A, B = platt(p, y)
    return {
        'top26': top_26, 'bot26': bot_26, 'top24': top_24, 'trophy_avg': trophy_avg,
        'brier_full': float(brier(p, y)),
        'brier_tr':   float(brier(p_tr, y_tr)),
        'brier_te':   float(brier(p_te, y_te)) if len(p_te) >= 30 else float('nan'),
        'platt_A':    A,
        'ece':        float(expected_calibration_error(p, y)),
        'sharp':      float(np.abs(p-0.5).mean()),
    }


def fmt(r, val_lbl, val):
    return (f"  {val_lbl}={val!s:<6}  top26={r['top26']:>+5.2f}  bot26={r['bot26']:>+5.2f}  "
            f"top24={r['top24']:>+5.2f}  troph={r['trophy_avg']:>4.2f}  "
            f"Brier(tr/te/full)={r['brier_tr']:.5f}/{r['brier_te']:.5f}/{r['brier_full']:.5f}  "
            f"A={r['platt_A']:>4.2f}  ECE={r['ece']:.4f}  sharp={r['sharp']:.3f}")


def main():
    matches = load_match_data()
    print(f"Loaded {len(matches)} series")
    print(f"Deployed config: RD_POWER=0.5 RD_SCALE=2.0 β=0.17, HL=6.0 RP=0.3, INTL=1.0 CH=2.0,")
    print(f"                 CN_PRIOR=-2.0 CN_INTL_K=4.0 SPILL_α=0.3 SHRINK_K=12")
    print()

    # === BASELINE ===
    print("━━━ BASELINE (current deployed) ━━━")
    base = evaluate(matches)
    print(fmt(base, '(deployed)', ''))
    print()

    sweeps = []

    # === Knob 1: HALF_LIFE_WEEKS ===
    print("━━━ KNOB 1: HALF_LIFE_WEEKS (currently 6.0) ━━━")
    knob_results = []
    for hl in [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0]:
        r = evaluate(matches, HALF_LIFE_WEEKS=hl)
        knob_results.append(('HL', hl, r))
        if r: print(fmt(r, 'HL', hl), flush=True)
    sweeps.append(('HALF_LIFE_WEEKS', 6.0, knob_results))

    # === Knob 2: ROSTER_PERSISTENCE ===
    print("\n━━━ KNOB 2: ROSTER_PERSISTENCE (currently 0.3) ━━━")
    knob_results = []
    for rp in [0.0, 0.15, 0.3, 0.5, 0.7]:
        r = evaluate(matches, ROSTER_PERSISTENCE=rp)
        knob_results.append(('RP', rp, r))
        if r: print(fmt(r, 'RP', rp), flush=True)
    sweeps.append(('ROSTER_PERSISTENCE', 0.3, knob_results))

    # === Knob 3: INTL_WIN_MULT / INTL_LOSS_MULT (paired) ===
    print("\n━━━ KNOB 3: INTL_WIN_MULT / INTL_LOSS_MULT (currently 1.0/1.0) ━━━")
    knob_results = []
    for im in [0.5, 1.0, 1.5, 2.0, 3.0]:
        r = evaluate(matches, INTL_WIN_MULT=im, INTL_LOSS_MULT=im)
        knob_results.append(('INTL', im, r))
        if r: print(fmt(r, 'INTL', im), flush=True)
    sweeps.append(('INTL_WIN_MULT', 1.0, knob_results))

    # === Knob 4: CHAMPIONS_MULT ===
    print("\n━━━ KNOB 4: CHAMPIONS_MULT (currently 2.0) ━━━")
    knob_results = []
    for cm in [1.0, 1.5, 2.0, 3.0, 4.0]:
        r = evaluate(matches, CHAMPIONS_MULT=cm)
        knob_results.append(('CH', cm, r))
        if r: print(fmt(r, 'CH', cm), flush=True)
    sweeps.append(('CHAMPIONS_MULT', 2.0, knob_results))

    # === Knob 5: CN_PRIOR ===
    print("\n━━━ KNOB 5: CN_PRIOR (currently -2.0) ━━━")
    knob_results = []
    for cp in [-1.0, -1.5, -2.0, -2.5, -3.0, -4.0]:
        r = evaluate(matches, CN_PRIOR=cp)
        knob_results.append(('CN_PR', cp, r))
        if r: print(fmt(r, 'CN_PR', cp), flush=True)
    sweeps.append(('CN_PRIOR', -2.0, knob_results))

    # === Knob 6: CN_INTL_K ===
    print("\n━━━ KNOB 6: CN_INTL_K (currently 4.0) ━━━")
    knob_results = []
    for ck in [2.0, 3.0, 4.0, 6.0, 8.0, 12.0]:
        r = evaluate(matches, CN_INTL_K=ck)
        knob_results.append(('CN_K', ck, r))
        if r: print(fmt(r, 'CN_K', ck), flush=True)
    sweeps.append(('CN_INTL_K', 4.0, knob_results))

    # === Knob 7: REGION_SPILLOVER_ALPHA ===
    print("\n━━━ KNOB 7: REGION_SPILLOVER_ALPHA (currently 0.30) ━━━")
    knob_results = []
    for a in [0.0, 0.15, 0.30, 0.50, 0.75, 1.0]:
        r = evaluate(matches, REGION_SPILLOVER_ALPHA=a)
        knob_results.append(('SPILL', a, r))
        if r: print(fmt(r, 'SPILL', a), flush=True)
    sweeps.append(('REGION_SPILLOVER_ALPHA', 0.30, knob_results))

    # === Knob 8: SHRINK_K (per-map James-Stein) ===
    print("\n━━━ KNOB 8: SHRINK_K (currently 12) ━━━")
    knob_results = []
    for sk in [4, 8, 12, 18, 24, 36]:
        r = evaluate(matches, SHRINK_K=sk)
        knob_results.append(('SK', sk, r))
        if r: print(fmt(r, 'SK', sk), flush=True)
    sweeps.append(('SHRINK_K', 12, knob_results))

    # === SUMMARY ===
    print("\n" + "="*100)
    print("SUMMARY — best-by-train-Brier vs best-by-calibration for each knob")
    print("="*100)
    for knob_name, deployed_val, knob_results in sweeps:
        valid = [(v, r) for _, v, r in knob_results if r is not None]
        if not valid: continue
        # Best train Brier
        best_brier = min(valid, key=lambda x: x[1]['brier_tr'])
        # Best calibration
        best_calib = min(valid, key=lambda x: abs(x[1]['platt_A'] - 1))
        # Deployed result
        deployed = next((r for v, r in valid if v == deployed_val), None)
        if not deployed: continue
        # Should we move?
        d_brier = deployed['brier_tr']
        b_val, b_r = best_brier
        c_val, c_r = best_calib

        improve_brier = (d_brier - b_r['brier_tr']) * 1000  # in basis points
        d_platt = abs(deployed['platt_A'] - 1)
        c_platt = abs(c_r['platt_A'] - 1)
        improve_platt = d_platt - c_platt

        verdict = "STAY"
        if (b_val != deployed_val and improve_brier > 1.0
            and abs(b_r['platt_A'] - 1) <= 0.15
            and b_r['trophy_avg'] <= 2.5):
            verdict = f"MOVE → {b_val} (Brier −{improve_brier:.1f}bp)"
        elif (c_val != deployed_val and improve_platt > 0.05
              and c_r['brier_tr'] <= d_brier * 1.005):
            verdict = f"MOVE → {c_val} (Platt closer to 1.0, Δ={improve_platt:+.2f})"

        print(f"\n  {knob_name} (deployed: {deployed_val})")
        print(f"    deployed:        Brier_tr={d_brier:.5f}  Platt={deployed['platt_A']:.2f}  top26={deployed['top26']:+.2f}")
        print(f"    best train Brier: {b_val:<6} Brier_tr={b_r['brier_tr']:.5f}  Platt={b_r['platt_A']:.2f}  top26={b_r['top26']:+.2f}")
        print(f"    best Platt:       {c_val:<6} Brier_tr={c_r['brier_tr']:.5f}  Platt={c_r['platt_A']:.2f}  top26={c_r['top26']:+.2f}")
        print(f"    verdict: {verdict}")

    print(f"\n━━━ Restoring deployed config ━━━")
    rebuild()
    print("Done.")


if __name__ == '__main__':
    main()
