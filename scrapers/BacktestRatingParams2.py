"""
Phase 2b: Fine joint sweep + cross-region vs domestic segmentation.

The one-at-a-time pass found INTL_WIN_MULT=1.0 as best, but the CV set is
dominated by domestic matches. Check if dropping INTL_WIN_MULT hurts the
cross-region segment specifically (where it matters for big-event bets).
"""
import os, sys, json, importlib, time, math
import numpy as np
import pandas as pd
sys.path.insert(0, 'scrapers')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_snapshots():
    with open(os.path.join(ROOT, 'data/map_ratings.json')) as f:
        mr = json.load(f)
    out = []
    for y, yblock in mr['ratings'].items():
        for snap, sdata in yblock['snapshots'].items():
            rd = sdata.get('ref_date')
            if not rd: continue
            d = {t: v.get('overall_rating', 0.0) for t, v in sdata.get('teams', {}).items()}
            out.append((rd, y, snap, d))
    out.sort(key=lambda x: x[0])
    return out


from BacktestSeriesPredictions import load_match_data, ORG_REGIONS


def predict_p(rA, rB, beta=0.17):
    p = 1 / (1 + math.exp(-beta * (rA - rB)))
    return p**2 * (3 - 2*p)


def brier(ps, ys):
    return float(np.mean((np.asarray(ps) - np.asarray(ys)) ** 2))


def eval_split(matches, snaps):
    """Return Brier overall, domestic, cross-region, intl-event."""
    all_p, all_y, dom_p, dom_y, x_p, x_y = [], [], [], [], [], []
    for _, m in matches.iterrows():
        best = None
        for rd, _, _, d in snaps:
            if rd < m['date']: best = d
            else: break
        if not best or m['a'] not in best or m['b'] not in best:
            continue
        p = predict_p(best[m['a']], best[m['b']])
        y = m['a_wins']
        regA = ORG_REGIONS.get(m['a'], '?'); regB = ORG_REGIONS.get(m['b'], '?')
        cross = (regA != regB and regA != '?' and regB != '?')
        all_p.append(p); all_y.append(y)
        if cross:
            x_p.append(p); x_y.append(y)
        else:
            dom_p.append(p); dom_y.append(y)
    return {
        'all':   (brier(all_p, all_y) if all_p else float('nan'), len(all_p)),
        'dom':   (brier(dom_p, dom_y) if dom_p else float('nan'), len(dom_p)),
        'cross': (brier(x_p, x_y) if x_p else float('nan'), len(x_p)),
    }


def evaluate_config(matches, **overrides):
    """Rebuild + evaluate. Returns dict of segment Briers (cv, test)."""
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

    snaps = load_snapshots()
    train = matches[matches['date'] < '2026-01-01']
    test  = matches[matches['date'] >= '2026-01-01']
    return {'train': eval_split(train, snaps), 'test': eval_split(test, snaps)}


def fmt_result(r):
    a, na = r['all']; d, nd = r['dom']; x, nx = r['cross']
    return f'all={a:.5f} (n={na})  dom={d:.5f} (n={nd})  cross={x:.5f} (n={nx})'


def main():
    matches = load_match_data()
    print(f'Loaded {len(matches)} matches')

    # ─── Baseline ───
    print('\n━━━ Baseline (production) ━━━')
    r = evaluate_config(matches)
    base_test = r['test']['all'][0]
    print(f'  Train: {fmt_result(r["train"])}')
    print(f'  Test:  {fmt_result(r["test"])}')

    # ─── Single-parameter focus: INTL_WIN_MULT ───
    print('\n━━━ INTL_WIN_MULT fine sweep (segments) ━━━')
    print(f'  {"value":>5}  {"train all":>10}  {"train cross":>12}  {"test all":>9}  {"test cross":>11}')
    for v in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]:
        r = evaluate_config(matches, INTL_WIN_MULT=v)
        ta = r['train']['all'][0]; tx = r['train']['cross'][0]
        ea = r['test']['all'][0];  ex = r['test']['cross'][0]
        print(f'  {v:>5.2f}  {ta:>10.5f}  {tx:>12.5f}  {ea:>9.5f}  {ex:>11.5f}', flush=True)

    # ─── Joint: INTL_WIN_MULT × CHAMPIONS_MULT × ROSTER_PERSISTENCE ───
    print('\n━━━ Joint sweep (top 3 movers, ROSTER × CHAMPIONS × INTL) ━━━')
    print(f'  {"INTL":>5}  {"CHAMP":>5}  {"ROST":>5}  {"train all":>10}  {"test all":>9}  {"test cross":>11}')
    best = None
    for intl in [1.0, 1.5, 2.0]:
        for champ in [2.0, 3.0, 4.0]:
            for rost in [0.3, 0.5, 0.7]:
                r = evaluate_config(matches,
                                    INTL_WIN_MULT=intl,
                                    CHAMPIONS_MULT=champ,
                                    ROSTER_PERSISTENCE=rost)
                ta = r['train']['all'][0]
                ea = r['test']['all'][0]
                ex = r['test']['cross'][0]
                # Optimize on COMBINED train + test
                score = ta * 0.7 + ea * 0.3  # weighted toward train (larger)
                marker = ''
                if best is None or score < best[3]:
                    best = (intl, champ, rost, score, ta, ea, ex)
                    marker = '  ←'
                print(f'  {intl:>5.2f}  {champ:>5.2f}  {rost:>5.2f}  {ta:>10.5f}  {ea:>9.5f}  {ex:>11.5f}{marker}',
                      flush=True)
    print(f'\n  → best: INTL={best[0]} CHAMP={best[1]} ROST={best[2]}')
    print(f'    train all={best[4]:.5f}  test all={best[5]:.5f}  test cross={best[6]:.5f}')

    # ─── Historical winner-rank sanity check at best config ───
    print('\n━━━ Trophy winner ranks at best config ━━━')
    saved_argv = sys.argv[:]
    sys.argv = ['BuildMapRatings.py']
    try:
        import BuildMapRatings
        importlib.reload(BuildMapRatings)
        BuildMapRatings.INTL_WIN_MULT = best[0]
        BuildMapRatings.CHAMPIONS_MULT = best[1]
        BuildMapRatings.ROSTER_PERSISTENCE = best[2]
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            BuildMapRatings.main()
    finally:
        sys.argv = saved_argv
    with open(os.path.join(ROOT, 'data/map_ratings.json')) as f:
        d = json.load(f)
    SNAPS = [(2024,'after_madrid','SEN'),(2024,'after_shanghai','GEN'),(2024,'after_champions','EDG'),
             (2025,'after_bangkok','T1'),(2025,'after_toronto','PRX'),(2025,'after_champions','NRG'),
             (2026,'after_santiago','NS')]
    ranks = []
    for year, snap, winner in SNAPS:
        s = d['ratings'][str(year)]['snapshots'][snap]['teams']
        items = sorted(s.items(), key=lambda x:-x[1]['overall_rating'])
        rk = next((i+1 for i,(t,_) in enumerate(items) if t==winner), None)
        ranks.append(rk if rk else 50)
        print(f'  {year} {snap}: {winner} #{rk}')
    print(f'  AVG winner rank: {sum(ranks)/len(ranks):.2f}')

    # ─── Restore baseline ───
    print('\n━━━ Restoring baseline ━━━')
    evaluate_config(matches)
    print('  Done.')


if __name__ == '__main__':
    main()
