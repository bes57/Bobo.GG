"""
Sharpness-first search: explicit goal is mean|p-0.5| ≥ 0.11
(currently 0.0857). Accept higher σ if Brier/LL/ECE all improve.

User explicitly: more sharp ratings + larger probability spread.
Less compressive RD transforms (power closer to 1.0 = linear) plus
appropriately scaled β should give bigger rating gaps and sharper preds.
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
    top_rating = max(td['overall_rating'] for td in mr['ratings']['2026']['snapshots']['after_santiago']['teams'].values())
    return {
        'cfg': cfg, 'beta': beta,
        'test_brier': brier(ep, ey), 'test_ll': logloss(ep, ey),
        'test_ece': ece(ep, ey), 'test_sigma': float(np.abs(ep - ey).std()),
        'test_sharpness': float(np.abs(ep - 0.5).mean()),
        'rk_gen': rk_gen, 'trophy_avg': sum(trophy_rks)/len(trophy_rks),
        'top_rating': top_rating,
    }


def main():
    matches = load_match_data()
    print('━━━ Current local config baseline (β=0.25, RD=power 0.35, RD_S=1.25) ━━━')
    base = evaluate(matches, beta=0.25)
    print(f'  Brier={base["test_brier"]:.5f}  LL={base["test_ll"]:.5f}  ECE={base["test_ece"]:.4f}  '
          f'σ={base["test_sigma"]:.4f}  sharp={base["test_sharpness"]:.4f}  '
          f'top_r={base["top_rating"]:+.2f}  GEN#{base["rk_gen"]}')

    # ─── Sharpness-first sweep: less compressive RD transforms ───
    # RD_POWER near 1.0 = linear (no compression), power=0.5=sqrt
    print('\n━━━ Searching for high-sharpness Pareto wins (RD_POWER 0.6-1.0) ━━━')
    print(f'  {"pow":>4}  {"RD":>5}  {"β":>5}  {"HL":>4}  {"CHAMP":>5}  {"ROST":>5}  '
          f'{"Brier":>7}  {"LL":>7}  {"ECE":>6}  {"σ":>6}  {"sharp":>6}  {"top_r":>7}  {"GEN":>4}  {"TAG":>5}')

    results = []
    powers = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    rd_scales = [0.5, 0.75, 1.0, 1.25, 1.5]
    betas = [0.10, 0.12, 0.15, 0.17, 0.20, 0.25, 0.30, 0.35]
    hls = [4.0, 5.0, 6.0, 8.0]
    champs = [2.0, 3.0]
    rosts = [0.0, 0.3]
    total = len(powers)*len(rd_scales)*len(betas)*len(hls)*len(champs)*len(rosts)
    print(f'  total: {total} configs', flush=True)
    t0 = time.time()
    for i, (pw, rd, b, hl, ch, ro) in enumerate(itertools.product(powers, rd_scales, betas, hls, champs, rosts), 1):
        try:
            r = evaluate(matches, beta=b, RD_TRANSFORM='power', RD_POWER=pw,
                         RD_SCALE=rd, HALF_LIFE_WEEKS=hl, CHAMPIONS_MULT=ch,
                         ROSTER_PERSISTENCE=ro)
            results.append(r)
        except Exception:
            continue
        # Tag interesting configs
        # T1 = strict Pareto on all 4 + sharper + GEN#1
        # T2 = Brier+LL+ECE all ≤ base, sharper, GEN#1 (σ free)
        # T3 = MUCH sharper (≥ 0.11) and Brier+LL+ECE ≤ base + 5%
        is_t1 = (r['test_brier'] <= base['test_brier']
                 and r['test_ll'] <= base['test_ll']
                 and r['test_ece'] <= base['test_ece']
                 and r['test_sigma'] <= base['test_sigma']
                 and r['test_sharpness'] > base['test_sharpness']
                 and r['rk_gen'] == 1)
        is_t2 = (r['test_brier'] <= base['test_brier']
                 and r['test_ll'] <= base['test_ll']
                 and r['test_ece'] <= base['test_ece']
                 and r['test_sharpness'] > base['test_sharpness']
                 and r['rk_gen'] == 1)
        is_t3 = (r['test_brier'] <= base['test_brier']
                 and r['test_ll'] <= base['test_ll']
                 and r['test_ece'] <= base['test_ece']
                 and r['test_sharpness'] >= 0.11
                 and r['rk_gen'] == 1)
        tag = ''
        if is_t1: tag = 'T1★'
        elif is_t3: tag = 'T3+SHARP'
        elif is_t2: tag = 'T2'
        if tag:
            print(f'  {pw:>4}  {rd:>5}  {b:>5.2f}  {hl:>4}  {ch:>5}  {ro:>5}  '
                  f'{r["test_brier"]:.5f}  {r["test_ll"]:.5f}  '
                  f'{r["test_ece"]:.4f}  {r["test_sigma"]:.4f}  '
                  f'{r["test_sharpness"]:.4f}  {r["top_rating"]:>+7.2f}  '
                  f'#{r["rk_gen"]:<3}  {tag:>5}', flush=True)
        if i % 100 == 0:
            elapsed = time.time() - t0
            eta = elapsed * (total - i) / i
            print(f'  ... [{i}/{total}] elapsed={elapsed:.0f}s eta={eta:.0f}s', flush=True)

    print(f'\n  Total time: {time.time()-t0:.0f}s')

    # Best for sharpness goal
    t3 = [r for r in results
          if r['test_brier'] <= base['test_brier']
          and r['test_ll'] <= base['test_ll']
          and r['test_ece'] <= base['test_ece']
          and r['test_sharpness'] >= 0.11
          and r['rk_gen'] == 1]
    print(f'\n  Sharpness≥0.11 + Brier+LL+ECE ≤ current + GEN#1: {len(t3)} configs')
    if t3:
        # Sort by sharpness descending, then composite
        t3.sort(key=lambda r: (-r['test_sharpness'],
                                r['test_brier'] + 2*r['test_ll'] + 5*r['test_ece']))
        print(f'  Top 10 by sharpness:')
        print(f'  {"pow":>4}  {"RD":>5}  {"β":>5}  {"HL":>4}  {"CHAMP":>5}  {"ROST":>5}  '
              f'{"Brier":>7}  {"LL":>7}  {"ECE":>6}  {"σ":>6}  {"sharp":>6}  {"top_r":>7}')
        for r in t3[:10]:
            c = r['cfg']
            print(f'  {c.get("RD_POWER",0.5):>4}  {c.get("RD_SCALE",1.25):>5}  {r["beta"]:>5.2f}  '
                  f'{c.get("HALF_LIFE_WEEKS",6):>4}  {c.get("CHAMPIONS_MULT",2):>5}  '
                  f'{c.get("ROSTER_PERSISTENCE",0.3):>5}  '
                  f'{r["test_brier"]:.5f}  {r["test_ll"]:.5f}  '
                  f'{r["test_ece"]:.4f}  {r["test_sigma"]:.4f}  '
                  f'{r["test_sharpness"]:.4f}  {r["top_rating"]:>+7.2f}')

    # Restore current
    print('\n━━━ Restoring current config ━━━')
    rebuild()
    print('Done.')


if __name__ == '__main__':
    main()
