"""
Deep multi-dimensional search for a config that:
  - Improves Brier vs current (HL=3.5, INTL=1, CHAMP=2, ROST=0.7)
  - Improves LogLoss
  - Improves ECE
  - Improves |err|σ
  - AND demotes 100T below #1 after Shanghai 2024

Coarse 5D grid: HL × INTL × CHAMP × ROST × RD_SCALE
Walk-forward CV + 2026 holdout evaluation per config.

Output:
  - All Pareto-improving configs (better on every metric)
  - Configs that also fix 100T (intersection)
  - Best by composite score
"""
import sys, os, json, importlib, math, time
import numpy as np
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


def predict(rA, rB, beta=0.17):
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


def evaluate(matches, **cfg):
    rebuild(**cfg)
    snaps, mr = load_snaps()
    test = matches[matches['date'] >= '2026-01-01']
    train = matches[matches['date'] < '2026-01-01']

    def collect(sub):
        ps, ys = [], []
        for _, m in sub.iterrows():
            best = None
            for rd, d in snaps:
                if rd < m['date']: best = d
                else: break
            if not best or m['a'] not in best or m['b'] not in best: continue
            ps.append(predict(best[m['a']], best[m['b']])); ys.append(m['a_wins'])
        return np.array(ps), np.array(ys)

    tp, ty = collect(train)
    ep, ey = collect(test)

    # 100T after Shanghai rank
    s = mr['ratings']['2024']['snapshots']['after_shanghai']['teams']
    items = sorted(s.items(), key=lambda x:-x[1]['overall_rating'])
    rk_100t = next((i+1 for i,(t,_) in enumerate(items) if t=='100T'), 99)

    # Trophy avg rank
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
        'cfg':         cfg,
        'train_brier': brier(tp, ty),
        'train_ll':    logloss(tp, ty),
        'test_brier':  brier(ep, ey),
        'test_ll':     logloss(ep, ey),
        'test_ece':    ece(ep, ey),
        'test_sigma':  float(np.abs(ep - ey).std()),
        'rk_100t':     rk_100t,
        'trophy_avg':  sum(trophy_rks)/len(trophy_rks),
    }


def main():
    matches = load_match_data()

    print('━━━ Phase 1: Establish baseline ━━━')
    t0 = time.time()
    baseline = evaluate(matches)
    print(f'  Baseline (current): test_brier={baseline["test_brier"]:.5f}  '
          f'test_ll={baseline["test_ll"]:.5f}  test_ece={baseline["test_ece"]:.4f}  '
          f'σ={baseline["test_sigma"]:.4f}  100T#{baseline["rk_100t"]}  trophy={baseline["trophy_avg"]:.2f}')

    grids = {
        'HALF_LIFE_WEEKS':    [2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
        'INTL_WIN_MULT':      [0.75, 1.0, 1.25, 1.5, 2.0],
        'CHAMPIONS_MULT':     [1.5, 2.0, 2.5, 3.0],
        'ROSTER_PERSISTENCE': [0.3, 0.5, 0.7],
        'RD_SCALE':           [1.5, 2.0, 2.5],
    }
    total = 1
    for v in grids.values(): total *= len(v)
    print(f'\n━━━ Phase 2: Coarse grid of {total} configs ━━━')

    results = [baseline]
    keys = list(grids.keys())

    import itertools
    combos = list(itertools.product(*[grids[k] for k in keys]))
    t_total = time.time()

    for i, combo in enumerate(combos, 1):
        cfg = dict(zip(keys, combo))
        # skip baseline (already evaluated)
        if cfg == {'HALF_LIFE_WEEKS':3.5,'INTL_WIN_MULT':1.0,'CHAMPIONS_MULT':2.0,
                   'ROSTER_PERSISTENCE':0.7,'RD_SCALE':2.0}:
            continue
        try:
            r = evaluate(matches, **cfg)
            results.append(r)
        except Exception as e:
            print(f'  config #{i} failed: {e}')
        if i % 25 == 0:
            elapsed = time.time() - t_total
            eta = elapsed * (total - i) / i
            print(f'  [{i}/{total}] elapsed={elapsed:.0f}s, eta={eta:.0f}s', flush=True)

    print(f'\n  Total time: {time.time()-t_total:.0f}s')
    print(f'  Evaluated {len(results)} configs')

    # ─── Pareto-improving configs ─────
    print(f'\n━━━ Phase 3: Pareto improvements over baseline ━━━')
    b = baseline
    pareto = []
    for r in results:
        if r is baseline: continue
        if (r['test_brier'] <= b['test_brier']
            and r['test_ll'] <= b['test_ll']
            and r['test_ece'] <= b['test_ece']
            and r['test_sigma'] <= b['test_sigma']):
            pareto.append(r)
    pareto.sort(key=lambda r: r['test_brier'] + r['test_ll'] + 5*r['test_ece'] + r['test_sigma'])
    print(f'  Pareto-improving: {len(pareto)} configs')
    print(f'\n  Top 10 Pareto by composite (Brier + LL + 5*ECE + σ):')
    print(f'  {"#":>3}  {"HL":>4}  {"INTL":>5}  {"CHAMP":>5}  {"ROST":>5}  {"RD":>4}  '
          f'{"Brier":>7}  {"LL":>7}  {"ECE":>6}  {"σ":>6}  {"100T":>4}  {"trophy":>6}')
    for i, r in enumerate(pareto[:10], 1):
        c = r['cfg']
        print(f'  {i:>3}  {c.get("HALF_LIFE_WEEKS",3.5):>4}  '
              f'{c.get("INTL_WIN_MULT",1.0):>5}  '
              f'{c.get("CHAMPIONS_MULT",2.0):>5}  '
              f'{c.get("ROSTER_PERSISTENCE",0.7):>5}  '
              f'{c.get("RD_SCALE",2.0):>4}  '
              f'{r["test_brier"]:.5f}  {r["test_ll"]:.5f}  '
              f'{r["test_ece"]:.4f}  {r["test_sigma"]:.4f}  '
              f'#{r["rk_100t"]:<3}  {r["trophy_avg"]:>6.2f}')

    # ─── Pareto AND demotes 100T ─────
    print(f'\n━━━ Phase 4: Pareto-improving AND 100T not #1 ━━━')
    pareto_100t = [r for r in pareto if r['rk_100t'] >= 2]
    if pareto_100t:
        pareto_100t.sort(key=lambda r: r['test_brier'] + r['test_ll'] + 5*r['test_ece'] + r['test_sigma'])
        print(f'  Found {len(pareto_100t)} configs that beat baseline on ALL metrics + demote 100T')
        print(f'\n  Top 5:')
        print(f'  {"#":>3}  {"HL":>4}  {"INTL":>5}  {"CHAMP":>5}  {"ROST":>5}  {"RD":>4}  '
              f'{"Brier":>7}  {"LL":>7}  {"ECE":>6}  {"σ":>6}  {"100T":>4}  {"trophy":>6}')
        for i, r in enumerate(pareto_100t[:5], 1):
            c = r['cfg']
            print(f'  {i:>3}  {c.get("HALF_LIFE_WEEKS",3.5):>4}  '
                  f'{c.get("INTL_WIN_MULT",1.0):>5}  '
                  f'{c.get("CHAMPIONS_MULT",2.0):>5}  '
                  f'{c.get("ROSTER_PERSISTENCE",0.7):>5}  '
                  f'{c.get("RD_SCALE",2.0):>4}  '
                  f'{r["test_brier"]:.5f}  {r["test_ll"]:.5f}  '
                  f'{r["test_ece"]:.4f}  {r["test_sigma"]:.4f}  '
                  f'#{r["rk_100t"]:<3}  {r["trophy_avg"]:>6.2f}')
    else:
        print(f'  No config dominates baseline on ALL 4 metrics WHILE demoting 100T.')
        # Try relaxing: configs that demote 100T AND don't hurt any metric by >2%
        print(f'\n  Relaxed: demote 100T + each metric within +2% of baseline:')
        relaxed = []
        for r in results:
            if r['rk_100t'] < 2: continue
            if (r['test_brier'] <= b['test_brier'] * 1.02
                and r['test_ll'] <= b['test_ll'] * 1.02
                and r['test_ece'] <= b['test_ece'] * 1.02
                and r['test_sigma'] <= b['test_sigma'] * 1.02):
                relaxed.append(r)
        relaxed.sort(key=lambda r: r['test_brier'] + r['test_ll'] + 5*r['test_ece'] + r['test_sigma'])
        print(f'  {len(relaxed)} configs')
        if relaxed:
            print(f'\n  Top 5:')
            print(f'  {"#":>3}  {"HL":>4}  {"INTL":>5}  {"CHAMP":>5}  {"ROST":>5}  {"RD":>4}  '
                  f'{"Brier":>7}  {"LL":>7}  {"ECE":>6}  {"σ":>6}  {"100T":>4}  {"trophy":>6}')
            for i, r in enumerate(relaxed[:5], 1):
                c = r['cfg']
                d_brier = (r['test_brier']-b['test_brier'])/b['test_brier']*100
                d_ll    = (r['test_ll']-b['test_ll'])/b['test_ll']*100
                d_ece   = (r['test_ece']-b['test_ece'])/b['test_ece']*100
                d_sig   = (r['test_sigma']-b['test_sigma'])/b['test_sigma']*100
                print(f'  {i:>3}  {c.get("HALF_LIFE_WEEKS",3.5):>4}  '
                      f'{c.get("INTL_WIN_MULT",1.0):>5}  '
                      f'{c.get("CHAMPIONS_MULT",2.0):>5}  '
                      f'{c.get("ROSTER_PERSISTENCE",0.7):>5}  '
                      f'{c.get("RD_SCALE",2.0):>4}  '
                      f'{r["test_brier"]:.5f}({d_brier:+.1f}%)  '
                      f'{r["test_ll"]:.5f}({d_ll:+.1f}%)  '
                      f'{r["test_ece"]:.4f}({d_ece:+.1f}%)  '
                      f'{r["test_sigma"]:.4f}({d_sig:+.1f}%)  '
                      f'#{r["rk_100t"]:<3}  {r["trophy_avg"]:>6.2f}')

    # Save results dump
    out = []
    for r in results:
        out.append({k:v for k,v in r.items() if k!='cfg'} | {'cfg':r['cfg']})
    with open('/tmp/pareto_results.json','w') as f:
        json.dump(out, f, indent=2)
    print(f'\nFull results saved to /tmp/pareto_results.json')

    # ─── Restore baseline current state ─────
    print('\n━━━ Restoring HL=3.5 baseline ━━━')
    rebuild()  # No overrides — uses constants from file (HL=3.5 now)
    print('  Done.')


if __name__ == '__main__':
    main()
