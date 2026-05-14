"""
Find a config that's Pareto-better than current + MORE SHARP.

Strategy:
  Phase 1: Sweep β at current rating config. RD_SCALE=1.5 compressed
           ratings; β=0.17 may now be too low. Higher β = more sharpness.
  Phase 2: Joint β × RD_SCALE sweep. Maybe higher RD_SCALE with proper β
           balances sharpness AND calibration.
  Phase 3: Joint β × RD_SCALE × HL × ROST sweep around the Phase 2 best.
  Phase 4: Validate winners with full metrics + trophy-rank check.

Hard constraint: GEN #1 after 2024 Shanghai (don't regress trophy ranks).
Soft goals: lower Brier, LL, ECE, σ AND higher mean|p-0.5|.
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
    train = matches[matches['date'] < '2026-01-01']
    test = matches[matches['date'] >= '2026-01-01']

    def collect(sub):
        ps, ys = [], []
        for _, m in sub.iterrows():
            best = None
            for rd, d in snaps:
                if rd < m['date']: best = d
                else: break
            if not best or m['a'] not in best or m['b'] not in best: continue
            p = predict(best[m['a']], best[m['b']], beta)
            ps.append(p); ys.append(m['a_wins'])
        return np.array(ps), np.array(ys)

    tp, ty = collect(train)
    ep, ey = collect(test)

    # GEN #1 check (hard constraint)
    s = mr['ratings']['2024']['snapshots']['after_shanghai']['teams']
    items = sorted(s.items(), key=lambda x:-x[1]['overall_rating'])
    rk_gen = next((i+1 for i,(t,_) in enumerate(items) if t=='GEN'), 99)

    # Trophy avg
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
        'train_brier':    brier(tp, ty),
        'train_ll':       logloss(tp, ty),
        'test_brier':     brier(ep, ey),
        'test_ll':        logloss(ep, ey),
        'test_ece':       ece(ep, ey),
        'test_sigma':     float(np.abs(ep - ey).std()),
        'test_sharpness': float(np.abs(ep - 0.5).mean()),
        'rk_gen':         rk_gen,
        'trophy_avg':     sum(trophy_rks)/len(trophy_rks),
    }


def main():
    matches = load_match_data()

    print('━━━ Baseline (current production) ━━━')
    t0 = time.time()
    base = evaluate(matches, beta=0.17)  # current uses HL=5, ROST=0.3, RD=1.5
    print(f'  test: Brier={base["test_brier"]:.5f}  LL={base["test_ll"]:.5f}  ECE={base["test_ece"]:.4f}  '
          f'σ={base["test_sigma"]:.4f}  sharpness={base["test_sharpness"]:.4f}')
    print(f'  GEN rank after Shanghai: #{base["rk_gen"]}  (must stay #1)')
    print(f'  ({time.time()-t0:.1f}s)')

    # ─── Phase 1: β sweep at current rating config ────
    print('\n━━━ Phase 1: β sweep at current rating config ━━━')
    print(f'  {"β":>5}  {"Brier":>7}  {"LL":>7}  {"ECE":>6}  {"σ":>6}  {"sharp":>6}  {"GEN":>4}')
    phase1 = [base]
    for b in [0.15, 0.17, 0.18, 0.20, 0.22, 0.24, 0.25, 0.27, 0.30, 0.32, 0.35]:
        r = evaluate(matches, beta=b)
        phase1.append(r)
        print(f'  {b:>5.2f}  {r["test_brier"]:.5f}  {r["test_ll"]:.5f}  '
              f'{r["test_ece"]:.4f}  {r["test_sigma"]:.4f}  {r["test_sharpness"]:.4f}  #{r["rk_gen"]}', flush=True)
    # Find Pareto-improving in Phase 1
    pareto1 = [r for r in phase1[1:]
               if r['test_brier'] <= base['test_brier']
               and r['test_ll'] <= base['test_ll']
               and r['test_ece'] <= base['test_ece']
               and r['test_sigma'] <= base['test_sigma']
               and r['test_sharpness'] > base['test_sharpness']
               and r['rk_gen'] == 1]
    if pareto1:
        pareto1.sort(key=lambda r: r['test_brier'] + 2*r['test_ll'] + 5*r['test_ece'] + r['test_sigma'])
        print(f'\n  Pareto+sharper (Phase 1): {len(pareto1)} configs')
        for r in pareto1[:5]:
            print(f'    β={r["beta"]:.3f}  Brier={r["test_brier"]:.5f}  LL={r["test_ll"]:.5f}  '
                  f'ECE={r["test_ece"]:.4f}  σ={r["test_sigma"]:.4f}  sharp={r["test_sharpness"]:.4f}')

    # ─── Phase 2: Joint β × RD_SCALE ────
    print('\n━━━ Phase 2: β × RD_SCALE joint sweep ━━━')
    print(f'  {"RD":>4}  {"β":>5}  {"Brier":>7}  {"LL":>7}  {"ECE":>6}  {"σ":>6}  {"sharp":>6}  {"GEN":>4}')
    phase2 = []
    for rd in [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0]:
        for b in [0.12, 0.15, 0.17, 0.20, 0.22, 0.25, 0.28, 0.30]:
            r = evaluate(matches, beta=b, RD_SCALE=rd)
            phase2.append(r)
            tag = ''
            if (r['test_brier'] <= base['test_brier']
                and r['test_ll'] <= base['test_ll']
                and r['test_ece'] <= base['test_ece']
                and r['test_sigma'] <= base['test_sigma']
                and r['test_sharpness'] > base['test_sharpness']
                and r['rk_gen'] == 1):
                tag = '  ★'
            print(f'  {rd:>4.2f}  {b:>5.2f}  {r["test_brier"]:.5f}  {r["test_ll"]:.5f}  '
                  f'{r["test_ece"]:.4f}  {r["test_sigma"]:.4f}  {r["test_sharpness"]:.4f}  #{r["rk_gen"]}{tag}',
                  flush=True)

    # Find Pareto+sharper+GEN#1 in Phase 2
    pareto2 = [r for r in phase2
               if r['test_brier'] <= base['test_brier']
               and r['test_ll'] <= base['test_ll']
               and r['test_ece'] <= base['test_ece']
               and r['test_sigma'] <= base['test_sigma']
               and r['test_sharpness'] > base['test_sharpness']
               and r['rk_gen'] == 1]
    print(f'\n  Phase 2 Pareto+sharper+GEN#1: {len(pareto2)} configs')
    pareto2.sort(key=lambda r: -r['test_sharpness'])  # most sharpness first among Pareto
    if pareto2:
        print(f'  Top 5 by sharpness:')
        for r in pareto2[:5]:
            print(f'    β={r["beta"]:.3f} RD={r["cfg"].get("RD_SCALE",1.5):.2f}  Brier={r["test_brier"]:.5f}  '
                  f'LL={r["test_ll"]:.5f}  ECE={r["test_ece"]:.4f}  σ={r["test_sigma"]:.4f}  '
                  f'sharp={r["test_sharpness"]:.4f}  trophy={r["trophy_avg"]:.2f}')

    # Save Phase 2 results
    with open('/tmp/sharpness_phase2.json', 'w') as f:
        out = []
        for r in phase2:
            o = {k: v for k, v in r.items() if k != 'cfg'}
            o['cfg'] = r['cfg']
            out.append(o)
        json.dump(out, f)

    # ─── Phase 3: Joint sweep refining Phase 2 best ────
    if pareto2:
        best = pareto2[0]
        best_rd = best['cfg'].get('RD_SCALE', 1.5)
        best_b = best['beta']
        print(f'\n━━━ Phase 3: refine around β={best_b:.3f}, RD={best_rd:.2f} with HL, ROST sweep ━━━')
        print(f'  {"HL":>4}  {"ROST":>5}  {"β":>5}  {"RD":>4}  {"Brier":>7}  {"LL":>7}  {"ECE":>6}  '
              f'{"σ":>6}  {"sharp":>6}  {"GEN":>4}  {"trophy":>6}')
        phase3 = []
        for hl in [3.5, 4.0, 4.5, 5.0, 5.5, 6.0]:
            for rost in [0.0, 0.15, 0.3, 0.5]:
                for db in [-0.02, 0, 0.02]:
                    for drd in [-0.25, 0, 0.25]:
                        b = best_b + db
                        rd = best_rd + drd
                        if b < 0.05 or rd < 0.5: continue
                        r = evaluate(matches, beta=b, HALF_LIFE_WEEKS=hl,
                                     ROSTER_PERSISTENCE=rost, RD_SCALE=rd)
                        phase3.append(r)
                        is_paretoplus = (r['test_brier'] <= base['test_brier']
                                          and r['test_ll'] <= base['test_ll']
                                          and r['test_ece'] <= base['test_ece']
                                          and r['test_sigma'] <= base['test_sigma']
                                          and r['test_sharpness'] > base['test_sharpness']
                                          and r['rk_gen'] == 1)
                        tag = '  ★' if is_paretoplus else ''
                        # Only print Pareto+ to keep output tractable
                        if is_paretoplus:
                            print(f'  {hl:>4.1f}  {rost:>5.2f}  {b:>5.2f}  {rd:>4.2f}  '
                                  f'{r["test_brier"]:.5f}  {r["test_ll"]:.5f}  '
                                  f'{r["test_ece"]:.4f}  {r["test_sigma"]:.4f}  '
                                  f'{r["test_sharpness"]:.4f}  #{r["rk_gen"]}  {r["trophy_avg"]:.2f}{tag}',
                                  flush=True)

        pareto3 = [r for r in phase3
                   if r['test_brier'] <= base['test_brier']
                   and r['test_ll'] <= base['test_ll']
                   and r['test_ece'] <= base['test_ece']
                   and r['test_sigma'] <= base['test_sigma']
                   and r['test_sharpness'] > base['test_sharpness']
                   and r['rk_gen'] == 1]
        # Sort by COMPOSITE: lowest (Brier + 2*LL + 5*ECE + σ - sharpness) — favors sharpness too
        pareto3.sort(key=lambda r: r['test_brier'] + 2*r['test_ll'] + 5*r['test_ece']
                                  + r['test_sigma'] - r['test_sharpness'])
        print(f'\n  Phase 3 Pareto+sharper+GEN#1: {len(pareto3)} configs')
        if pareto3:
            print(f'\n  Best 5 by composite (lower is better, sharpness adds negative):')
            for r in pareto3[:5]:
                c = r['cfg']
                print(f'    HL={c.get("HALF_LIFE_WEEKS",5):.1f}  '
                      f'ROST={c.get("ROSTER_PERSISTENCE",0.3):.2f}  '
                      f'β={r["beta"]:.3f}  RD={c.get("RD_SCALE",1.5):.2f}  '
                      f'Brier={r["test_brier"]:.5f}  LL={r["test_ll"]:.5f}  '
                      f'ECE={r["test_ece"]:.4f}  σ={r["test_sigma"]:.4f}  '
                      f'sharp={r["test_sharpness"]:.4f}  trophy={r["trophy_avg"]:.2f}')
        # Save phase3
        with open('/tmp/sharpness_phase3.json', 'w') as f:
            out = []
            for r in phase3:
                o = {k: v for k, v in r.items() if k != 'cfg'}
                o['cfg'] = r['cfg']
                out.append(o)
            json.dump(out, f)
    else:
        print('  Phase 2 found nothing — Phase 3 skipped, will try structural changes next.')

    # ─── Restore current ────
    print('\n━━━ Restoring current ━━━')
    rebuild()
    print('Done.')


if __name__ == '__main__':
    main()
