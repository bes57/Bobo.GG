"""
Deeper magnitude exploration:
  1. Push RD_SCALE much higher (1.5 → 4.0)
  2. Test RD_POWER beyond sqrt (0.5 → 1.0 = linear)
  3. Test RD_TRANSFORM='diff' (no compression at all)
  4. Test lower ridge (BuildMapRatings has it hardcoded — patch via setattr)
  5. Find the magnitude "knee" — the point where pushing further hurts predictions

For each config:
  - Top/bot/sd at each year
  - Brier (full/train/test)
  - Platt calibration A (1.0 = perfect)
  - ECE
  - Sharpness
  - Trophy-winner ranks preserved?

Output: Pareto frontier of (top_2026 magnitude × Brier × |Platt-1|).
Recommend the most aggressive config that doesn't degrade predictions.
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


def predict_map(rA, rB, beta):
    return 1.0 / (1.0 + math.exp(-beta * (rA - rB)))

def predict_series(rA, rB, beta, fmt='bo3'):
    p = predict_map(rA, rB, beta)
    return p**2 * (3 - 2*p)


def gen(matches, snaps, beta):
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


def rebuild(ridge_override=None, **overrides):
    """Rebuild ratings. ridge_override patches the hardcoded ridge inside massey_ratings."""
    saved_argv = sys.argv[:]
    sys.argv = ['BuildMapRatings.py']
    try:
        import BuildMapRatings
        importlib.reload(BuildMapRatings)
        for k, v in overrides.items():
            setattr(BuildMapRatings, k, v)
        # Optionally patch ridge inside massey_ratings (currently hardcoded to 0.5)
        if ridge_override is not None:
            # We monkey-patch the function source — risky. Better: pass via a global.
            # Easiest path: shim by inspecting massey_ratings code.
            # For simplicity, patch via a module-level attribute the function will look up.
            BuildMapRatings._RIDGE_OVERRIDE = ridge_override
            # Patch the function to honor the override
            orig = BuildMapRatings.massey_ratings
            import functools
            @functools.wraps(orig)
            def patched(*args, **kwargs):
                # Re-read source quickly: the function uses local var `ridge = 0.5`
                # Easier approach: write a clone here
                return _massey_with_ridge(*args, ridge_value=ridge_override, **kwargs)
            BuildMapRatings.massey_ratings = patched
        with contextlib.redirect_stdout(io.StringIO()):
            BuildMapRatings.main()
    finally:
        sys.argv = saved_argv


def _massey_with_ridge(games, lambda_decay, ref_date, min_games=5, ridge_value=0.5):
    """Clone of massey_ratings with configurable ridge."""
    import BuildMapRatings as B
    # Filter teams
    counts = {}
    for g in games:
        counts[g['winner']] = counts.get(g['winner'], 0) + 1
        counts[g['loser']]  = counts.get(g['loser'], 0) + 1
    teams = sorted(t for t, c in counts.items() if c >= min_games)
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    if n < 2: return {t: 0.0 for t in teams}
    M = np.zeros((n, n)); p = np.zeros(n)
    for g in games:
        if g['winner'] not in idx or g['loser'] not in idx: continue
        is_intl = g.get('event_id') in B.INTL_EVENTS
        is_champions = 'champions' in g.get('event_id', '')
        eff_weeks_w = B._effective_weeks_ago(g['winner'], g['date'], ref_date)
        eff_weeks_l = B._effective_weeks_ago(g['loser'],  g['date'], ref_date)
        w_winner = math.exp(-lambda_decay * eff_weeks_w)
        w_loser  = math.exp(-lambda_decay * eff_weeks_l)
        base_w = math.sqrt(w_winner * w_loser)
        cont_w = B._team_continuity_factor(g['winner'], g['date'], ref_date)
        cont_l = B._team_continuity_factor(g['loser'],  g['date'], ref_date)
        base_w *= math.sqrt(cont_w * cont_l)
        if is_champions:
            win_mult = los_mult = B.CHAMPIONS_MULT
        elif is_intl:
            win_mult = B.INTL_WIN_MULT; los_mult = B.INTL_LOSS_MULT
        else:
            win_mult = los_mult = 1.0
        w_win = base_w * win_mult
        w_los = base_w * los_mult
        w_sym = min(w_win, w_los)
        raw_rd = g['wr'] - g['lr']
        if B.RD_TRANSFORM == 'sqrt':
            rd = math.copysign(math.sqrt(abs(raw_rd)) * B.RD_SCALE, raw_rd)
        elif B.RD_TRANSFORM == 'power':
            rd = math.copysign((abs(raw_rd) ** B.RD_POWER) * B.RD_SCALE, raw_rd)
        else:  # 'diff'
            rd = raw_rd * B.RD_SCALE
        i, j = idx[g['winner']], idx[g['loser']]
        M[i, i] += w_sym;  M[j, j] += w_sym
        M[i, j] -= w_sym;  M[j, i] -= w_sym
        p[i] += w_win * rd; p[j] -= w_los * rd
    M[-1, :] = 1.0; p[-1] = 0.0
    for i in range(n - 1):
        M[i, i] += ridge_value
    M[-1, :] = 1.0; p[-1] = 0.0
    try:
        r = np.linalg.solve(M, p)
    except np.linalg.LinAlgError:
        r, *_ = np.linalg.lstsq(M, p, rcond=None)
    return {t: float(r[idx[t]]) for t in teams}


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


def evaluate(matches, beta_grid, ridge_override=None, **cfg):
    rebuild(ridge_override=ridge_override, **cfg)
    snaps = load_snapshots()
    with open(os.path.join(ROOT, 'data/map_ratings.json')) as f:
        mr = json.load(f)

    def snap_stats(year, snap_key):
        teams = mr['ratings'][year]['snapshots'].get(snap_key, {}).get('teams', {})
        if not teams: return None
        rs = [t['overall_rating'] for t in teams.values()]
        return {'top': max(rs), 'bot': min(rs), 'sd': float(np.std(rs)), 'range': max(rs)-min(rs)}

    s26 = snap_stats('2026', 'after_santiago')
    s24 = snap_stats('2024', 'after_champions')
    s25 = snap_stats('2025', 'after_champions')
    trophy_avg = float(np.mean(winner_ranks(mr)))

    out = {'cfg': cfg, 'ridge': ridge_override or 0.5,
           's26': s26, 's24': s24, 's25': s25, 'trophy_avg': trophy_avg, 'betas': {}}

    matches_train = matches[matches['date'] < '2026-01-01'].reset_index(drop=True)
    matches_test  = matches[matches['date'] >= '2026-01-01'].reset_index(drop=True)

    for b in beta_grid:
        p_full, y_full = gen(matches, snaps, b)
        p_tr, y_tr     = gen(matches_train, snaps, b)
        p_te, y_te     = gen(matches_test, snaps, b)
        if len(p_full) < 100: continue
        A, B = platt(p_full, y_full)
        out['betas'][b] = {
            'brier_full': float(brier(p_full, y_full)),
            'brier_train': float(brier(p_tr, y_tr)),
            'brier_test': float(brier(p_te, y_te)) if len(p_te) >= 30 else float('nan'),
            'platt_A': A,
            'ece': float(expected_calibration_error(p_full, y_full)),
            'sharp': float(np.abs(p_full-0.5).mean()),
            'maxP': float(p_full.max()),
        }
    return out


def fmt_row(r, beta):
    bm = r['betas'].get(beta, {})
    if not bm: return None
    cfg = r['cfg']
    pw = cfg.get('RD_POWER', '-')
    sc = cfg.get('RD_SCALE', '-')
    tr = cfg.get('RD_TRANSFORM', 'power')[:5]
    rd = r['ridge']
    top26 = r['s26']['top'] if r['s26'] else 0
    bot26 = r['s26']['bot'] if r['s26'] else 0
    sd26 = r['s26']['sd'] if r['s26'] else 0
    return f"  {tr:>5}  {pw:>4}  {sc:>4}  {rd:>4}  {beta:>5.3f}  {top26:>+6.2f}  {bot26:>+6.2f}  {sd26:>4.2f}  {r['trophy_avg']:>4.1f}  {bm['brier_full']:.5f}  {bm['platt_A']:>5.2f}  {bm['ece']:.4f}  {bm['sharp']:.3f}  {bm['maxP']:.3f}"


def main():
    matches = load_match_data()
    print(f"Loaded {len(matches)} historical series")
    print()
    print("Goal: find true Pareto frontier of magnitude × Brier × |Platt-1|")
    print("Knee = max magnitude where Brier_full ≤ 0.232 AND |Platt-1| ≤ 0.15 AND trophy ≤ 2.5")
    print()
    print(f"  {'tform':>5}  {'pow':>4}  {'sc':>4}  {'ridg':>4}  {'β':>5}  {'top26':>6}  {'bot26':>6}  {'sd26':>4}  "
          f"{'troph':>4}  {'Brier':>7}  {'PltA':>5}  {'ECE':>6}  {'sharp':>5}  {'maxP':>5}")

    # Current new baseline (just deployed)
    print("\n━━━ Current deployed baseline (RD_POWER=0.5, RD_SCALE=1.5) ━━━")
    base = evaluate(matches, [0.22],
                    RD_TRANSFORM='power', RD_POWER=0.5, RD_SCALE=1.5,
                    HALF_LIFE_WEEKS=6.0, INTL_WIN_MULT=1.0, CHAMPIONS_MULT=2.0,
                    ROSTER_PERSISTENCE=0.3)
    print(fmt_row(base, 0.22))

    all_results = [base]

    # === EXPERIMENT 1: Push RD_SCALE much higher with RD_POWER=0.5 (sqrt) ===
    print("\n━━━ EXP 1: Push RD_SCALE with RD_POWER=0.5 (sqrt) ━━━")
    for sc in [1.5, 2.0, 2.5, 3.0, 4.0]:
        r = evaluate(matches, [0.10, 0.14, 0.17, 0.20, 0.22, 0.25],
                     RD_TRANSFORM='power', RD_POWER=0.5, RD_SCALE=sc,
                     HALF_LIFE_WEEKS=6.0, INTL_WIN_MULT=1.0, CHAMPIONS_MULT=2.0,
                     ROSTER_PERSISTENCE=0.3)
        all_results.append(r)
        for b in [0.10, 0.14, 0.17, 0.20, 0.22, 0.25]:
            row = fmt_row(r, b)
            if row: print(row, flush=True)

    # === EXPERIMENT 2: Linear transform (no compression) ===
    print("\n━━━ EXP 2: RD_TRANSFORM='diff' (no compression, linear) ━━━")
    for sc in [0.05, 0.10, 0.15, 0.20, 0.30]:
        r = evaluate(matches, [0.10, 0.14, 0.17, 0.20, 0.25],
                     RD_TRANSFORM='diff', RD_POWER=1.0, RD_SCALE=sc,
                     HALF_LIFE_WEEKS=6.0, INTL_WIN_MULT=1.0, CHAMPIONS_MULT=2.0,
                     ROSTER_PERSISTENCE=0.3)
        all_results.append(r)
        for b in [0.10, 0.14, 0.17, 0.20, 0.25]:
            row = fmt_row(r, b)
            if row: print(row, flush=True)

    # === EXPERIMENT 3: Power > 0.5 (less compressive than sqrt) ===
    print("\n━━━ EXP 3: RD_POWER > 0.5 (less compressive than sqrt) ━━━")
    for pw, sc in [(0.7, 1.5), (0.7, 2.0), (0.7, 2.5),
                   (0.9, 1.0), (0.9, 1.5), (0.9, 2.0),
                   (1.1, 0.5), (1.1, 1.0)]:
        r = evaluate(matches, [0.08, 0.10, 0.13, 0.16, 0.20],
                     RD_TRANSFORM='power', RD_POWER=pw, RD_SCALE=sc,
                     HALF_LIFE_WEEKS=6.0, INTL_WIN_MULT=1.0, CHAMPIONS_MULT=2.0,
                     ROSTER_PERSISTENCE=0.3)
        all_results.append(r)
        for b in [0.08, 0.10, 0.13, 0.16, 0.20]:
            row = fmt_row(r, b)
            if row: print(row, flush=True)

    # === EXPERIMENT 4: Lower ridge (lets ratings spread further) ===
    print("\n━━━ EXP 4: Lower ridge (default 0.5) ━━━")
    for rg in [0.1, 0.25, 1.0]:
        r = evaluate(matches, [0.18, 0.20, 0.22, 0.25],
                     ridge_override=rg,
                     RD_TRANSFORM='power', RD_POWER=0.5, RD_SCALE=1.5,
                     HALF_LIFE_WEEKS=6.0, INTL_WIN_MULT=1.0, CHAMPIONS_MULT=2.0,
                     ROSTER_PERSISTENCE=0.3)
        all_results.append(r)
        for b in [0.18, 0.20, 0.22, 0.25]:
            row = fmt_row(r, b)
            if row: print(row, flush=True)

    # === PARETO ANALYSIS ===
    print("\n" + "="*100)
    print("PARETO ANALYSIS — sorted by top_2026 magnitude")
    print("="*100)
    # For each result, pick the β that maximizes magnitude while keeping
    # Brier_full ≤ 0.232 AND |Platt-1| ≤ 0.15 AND trophy_avg ≤ 2.5
    pareto = []
    for r in all_results:
        for b, bm in r['betas'].items():
            if (bm['brier_full'] <= 0.232 and
                abs(bm['platt_A'] - 1) <= 0.15 and
                r['trophy_avg'] <= 2.5 and
                r['s26']):
                pareto.append((b, r, bm))
    # Best magnitude at each "Brier budget"
    pareto.sort(key=lambda x: -x[1]['s26']['top'])
    print(f"\n  Configs passing all gates, sorted by top_2026 magnitude:")
    print(f"  {'tform':>5}  {'pow':>4}  {'sc':>4}  {'ridg':>4}  {'β':>5}  {'top26':>6}  {'bot26':>6}  {'top24':>6}  "
          f"{'sd26':>4}  {'troph':>5}  {'Brier':>7}  {'PltA':>5}  {'ECE':>6}  {'sharp':>5}")
    for b, r, bm in pareto[:25]:
        cfg = r['cfg']
        print(f"  {cfg.get('RD_TRANSFORM','power')[:5]:>5}  {cfg.get('RD_POWER','-'):>4}  "
              f"{cfg.get('RD_SCALE','-'):>4}  {r['ridge']:>4}  {b:>5.3f}  "
              f"{r['s26']['top']:>+6.2f}  {r['s26']['bot']:>+6.2f}  {r['s24']['top']:>+6.2f}  "
              f"{r['s26']['sd']:>4.2f}  {r['trophy_avg']:>5.2f}  "
              f"{bm['brier_full']:.5f}  {bm['platt_A']:>5.2f}  {bm['ece']:.4f}  {bm['sharp']:.3f}")

    # Find the "knee" — max magnitude with Platt closest to 1.0
    print(f"\n  Configs with Platt A in [0.95, 1.05] (tightest calibration band):")
    tight = [(b, r, bm) for b, r, bm in pareto if 0.95 <= bm['platt_A'] <= 1.05]
    tight.sort(key=lambda x: -x[1]['s26']['top'])
    for b, r, bm in tight[:10]:
        cfg = r['cfg']
        print(f"  {cfg.get('RD_TRANSFORM','power')[:5]:>5}  {cfg.get('RD_POWER','-'):>4}  "
              f"{cfg.get('RD_SCALE','-'):>4}  {r['ridge']:>4}  {b:>5.3f}  "
              f"{r['s26']['top']:>+6.2f}  {r['s26']['bot']:>+6.2f}  {r['s24']['top']:>+6.2f}  "
              f"{r['s26']['sd']:>4.2f}  {r['trophy_avg']:>5.2f}  "
              f"{bm['brier_full']:.5f}  {bm['platt_A']:>5.2f}  {bm['ece']:.4f}  {bm['sharp']:.3f}")

    # Save raw for inspection
    out_path = '/tmp/magnitude_knee.json'
    with open(out_path, 'w') as f:
        save = []
        for r in all_results:
            o = {k: v for k, v in r.items() if k != 'betas'}
            o['betas'] = {str(b): bm for b, bm in r['betas'].items()}
            save.append(o)
        json.dump(save, f, indent=2)
    print(f"\nSaved raw results to {out_path}")

    print("\n━━━ Restoring deployed config ━━━")
    rebuild(RD_TRANSFORM='power', RD_POWER=0.5, RD_SCALE=1.5)
    print("Done.")


if __name__ == '__main__':
    main()
