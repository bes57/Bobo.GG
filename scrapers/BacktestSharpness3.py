"""
Structural-change sweep: try power-RD transform with various exponents.
sqrt (power=0.5) is highly compressive — power=0.7-1.0 keeps more of the
blowout signal which may give more INFORMATIVE ratings → sharper-and-
accurate predictions.

Plus: joint with β and RD_SCALE.
"""
import sys, os, json, importlib, math, time
import numpy as np
import itertools
sys.path.insert(0, 'scrapers')
from BacktestSeriesPredictions import load_match_data


def rebuild(**overrides):
    saved_argv = sys.argv[:]
    sys.argv = ['BuildMapRatings.py']
    try:
        import BuildMapRatings
        importlib.reload(BuildMapRatings)
        for k, v in overrides.items():
            setattr(BuildMapRatings, k, v)
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            BuildMapRatings.main()
    finally:
        sys.argv = saved_argv


def load_snaps():
    with open('data/map_ratings.json') as f: mr = json.load(f)
    snaps = []
    for y, yblock in mr['ratings'].items():
        for snap, sdata in yblock['snapshots'].items():
            rd = sdata.get('ref_date')
            if not rd: continue
            d = {t: v.get('overall_rating', 0.0) for t, v in sdata.get('teams', {}).items()}
            snaps.append((rd, d))
    snaps.sort(key=lambda x: x[0])
    return snaps, mr


def predict(rA, rB, beta):
    p = 1/(1+math.exp(-beta*(rA-rB)))
    return p**2*(3-2*p)


def brier(p,y): return float(np.mean((np.asarray(p)-np.asarray(y))**2))
def logloss(p,y):
    p = np.clip(np.asarray(p), 1e-9, 1-1e-9); y = np.asarray(y)
    return float(-np.mean(y*np.log(p) + (1-y)*np.log(1-p)))
def ece(p,y,n_bins=10):
    p = np.asarray(p); y = np.asarray(y)
    bins = np.linspace(0,1,n_bins+1)
    idx = np.clip(np.digitize(p,bins)-1, 0, n_bins-1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any(): continue
        e += (m.sum()/len(p)) * abs(p[m].mean() - y[m].mean())
    return float(e)


def evaluate(matches, beta, **cfg):
    rebuild(**cfg)
    snaps, mr = load_snaps()
    test = matches[matches['date'] >= '2026-01-01']
    ps, ys = [], []
    for _, m in test.iterrows():
        best = None
        for rd, d in snaps:
            if rd < m['date']: best = d
            else: break
        if not best or m['a'] not in best or m['b'] not in best: continue
        p = predict(best[m['a']], best[m['b']], beta)
        ps.append(p); ys.append(m['a_wins'])
    ep = np.array(ps); ey = np.array(ys)

    s = mr['ratings']['2024']['snapshots']['after_shanghai']['teams']
    items = sorted(s.items(), key=lambda x:-x[1]['overall_rating'])
    rk_gen = next((i+1 for i,(t,_) in enumerate(items) if t=='GEN'), 99)

    SNAPS = [(2024,'after_madrid','SEN'),(2024,'after_shanghai','GEN'),(2024,'after_champions','EDG'),
             (2025,'after_bangkok','T1'),(2025,'after_toronto','PRX'),(2025,'after_champions','NRG'),
             (2026,'after_santiago','NS')]
    trophy_rks = []
    for year, snap, winner in SNAPS:
        s = mr['ratings'][str(year)]['snapshots'][snap]['teams']
        items = sorted(s.items(), key=lambda x:-x[1]['overall_rating'])
        rk = next((i+1 for i,(t,_) in enumerate(items) if t==winner), 50)
        trophy_rks.append(rk)

    return {
        'cfg': cfg, 'beta': beta,
        'test_brier': brier(ep, ey),
        'test_ll': logloss(ep, ey),
        'test_ece': ece(ep, ey),
        'test_sigma': float(np.abs(ep - ey).std()),
        'test_sharpness': float(np.abs(ep - 0.5).mean()),
        'rk_gen': rk_gen,
        'trophy_avg': sum(trophy_rks)/len(trophy_rks),
    }


def main():
    matches = load_match_data()
    print('━━━ Baseline (current) ━━━')
    base = evaluate(matches, beta=0.17)
    print(f'  Brier={base["test_brier"]:.5f}  LL={base["test_ll"]:.5f}  ECE={base["test_ece"]:.4f}  '
          f'σ={base["test_sigma"]:.4f}  sharp={base["test_sharpness"]:.4f}  GEN#{base["rk_gen"]}')

    # ─── RD_POWER × β joint sweep ───
    print('\n━━━ RD_POWER × β × RD_SCALE sweep (transform=power) ━━━')
    results = []
    powers = [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    rd_scales = [0.5, 1.0, 1.5, 2.0]
    betas = [0.12, 0.15, 0.17, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35]
    total = len(powers) * len(rd_scales) * len(betas)
    print(f'  Total configs: {total}')
    t0 = time.time()
    for i, (pw, rd, b) in enumerate(itertools.product(powers, rd_scales, betas), 1):
        try:
            r = evaluate(matches, beta=b,
                         RD_TRANSFORM='power', RD_POWER=pw, RD_SCALE=rd)
            results.append(r)
        except Exception:
            continue
        if i % 30 == 0:
            elapsed = time.time() - t0
            eta = elapsed * (total - i) / i
            print(f'  [{i}/{total}] elapsed={elapsed:.0f}s eta={eta:.0f}s', flush=True)

    print(f'\n  Done in {time.time()-t0:.0f}s')

    # ─── Find Pareto+sharper+GEN#1 ───
    t1 = [r for r in results
          if r['test_brier'] <= base['test_brier']
          and r['test_ll'] <= base['test_ll']
          and r['test_ece'] <= base['test_ece']
          and r['test_sigma'] <= base['test_sigma']
          and r['test_sharpness'] > base['test_sharpness']
          and r['rk_gen'] == 1]
    print(f'\n  Strict Pareto + sharper + GEN#1: {len(t1)} configs')
    if t1:
        t1.sort(key=lambda r: r['test_brier'] + 2*r['test_ll'] + 5*r['test_ece']
                              + r['test_sigma'] - 0.5*r['test_sharpness'])
        print(f'\n  Top 10:')
        print(f'  {"power":>5}  {"RD":>4}  {"β":>5}  {"Brier":>7}  {"LL":>7}  {"ECE":>6}  '
              f'{"σ":>6}  {"sharp":>6}  {"trophy":>6}')
        for r in t1[:10]:
            c = r['cfg']
            print(f'  {c.get("RD_POWER",0.5):>5.2f}  {c.get("RD_SCALE",1.5):>4.2f}  {r["beta"]:>5.2f}  '
                  f'{r["test_brier"]:.5f}  {r["test_ll"]:.5f}  '
                  f'{r["test_ece"]:.4f}  {r["test_sigma"]:.4f}  '
                  f'{r["test_sharpness"]:.4f}  {r["trophy_avg"]:>6.2f}')

    # Also save full results
    with open('/tmp/sharpness3.json', 'w') as f:
        out = []
        for r in results:
            o = {k: v for k, v in r.items() if k != 'cfg'}
            o['cfg'] = r['cfg']
            out.append(o)
        json.dump(out, f)

    # Restore baseline
    print('\n━━━ Restoring baseline ━━━')
    rebuild()
    print('Done.')


if __name__ == '__main__':
    main()
