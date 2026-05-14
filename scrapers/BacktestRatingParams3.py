"""
Phase 2c: Final candidate selection.

Compare four candidate configs head-to-head with full metrics + ranks:
  1. PROD: current production
  2. BRIER:  optimize purely for Brier (may hurt rankings)
  3. BLEND:  middle ground (INTL=1.0, CHAMP=3.0, ROST=0.5)
  4. SOFT:   keep rankings high (INTL=1.5, CHAMP=4.0, ROST=0.5)

All metrics:
  - 2026 holdout Brier, LogLoss, ECE, |err|σ
  - Cross-region segment Brier
  - Avg trophy-winner rank across 7 intl events
  - High-confidence calibration (where pred >0.7, actual win rate)
"""
import os, sys, json, importlib, math
import numpy as np
sys.path.insert(0, 'scrapers')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from BacktestSeriesPredictions import load_match_data, ORG_REGIONS
import io, contextlib


def rebuild(**overrides):
    saved_argv = sys.argv[:]
    sys.argv = ['BuildMapRatings.py']
    try:
        import BuildMapRatings
        importlib.reload(BuildMapRatings)
        for k, v in overrides.items():
            setattr(BuildMapRatings, k, v)
        with contextlib.redirect_stdout(io.StringIO()):
            BuildMapRatings.main()
    finally:
        sys.argv = saved_argv


def load_snaps():
    with open(os.path.join(ROOT, 'data/map_ratings.json')) as f:
        mr = json.load(f)
    out = []
    for y, yblock in mr['ratings'].items():
        for snap, sdata in yblock['snapshots'].items():
            rd = sdata.get('ref_date')
            if not rd: continue
            d = {t: v.get('overall_rating', 0.0) for t, v in sdata.get('teams', {}).items()}
            out.append((rd, y, snap, d))
    out.sort()
    return out, mr


def predict(rA, rB, beta=0.17):
    p = 1 / (1 + math.exp(-beta * (rA - rB)))
    return p**2 * (3 - 2*p)


def brier(p, y): return float(np.mean((np.asarray(p) - np.asarray(y))**2))
def logloss(p, y):
    p = np.clip(np.asarray(p), 1e-9, 1-1e-9)
    y = np.asarray(y)
    return float(-np.mean(y*np.log(p) + (1-y)*np.log(1-p)))


def ece(p, y, n_bins=10):
    p = np.asarray(p); y = np.asarray(y)
    bins = np.linspace(0, 1, n_bins+1)
    idx = np.clip(np.digitize(p, bins)-1, 0, n_bins-1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any(): continue
        e += (m.sum()/len(p)) * abs(p[m].mean() - y[m].mean())
    return float(e)


def evaluate(matches, label, **cfg):
    rebuild(**cfg)
    snaps, mr = load_snaps()
    train = matches[matches['date'] < '2026-01-01']
    test  = matches[matches['date'] >= '2026-01-01']

    def get_preds(sub):
        all_p, all_y, x_p, x_y, dom_p, dom_y = [], [], [], [], [], []
        for _, m in sub.iterrows():
            best = None
            for rd, _, _, dd in snaps:
                if rd < m['date']: best = dd
                else: break
            if not best or m['a'] not in best or m['b'] not in best:
                continue
            p = predict(best[m['a']], best[m['b']])
            y = m['a_wins']
            all_p.append(p); all_y.append(y)
            regA = ORG_REGIONS.get(m['a'], '?'); regB = ORG_REGIONS.get(m['b'], '?')
            if regA != regB and regA != '?' and regB != '?':
                x_p.append(p); x_y.append(y)
            else:
                dom_p.append(p); dom_y.append(y)
        return all_p, all_y, x_p, x_y, dom_p, dom_y

    tr_all_p, tr_all_y, tr_x_p, tr_x_y, tr_d_p, tr_d_y = get_preds(train)
    te_all_p, te_all_y, te_x_p, te_x_y, te_d_p, te_d_y = get_preds(test)

    # Winner ranks
    SNAPS = [(2024,'after_madrid','SEN'),(2024,'after_shanghai','GEN'),(2024,'after_champions','EDG'),
             (2025,'after_bangkok','T1'),(2025,'after_toronto','PRX'),(2025,'after_champions','NRG'),
             (2026,'after_santiago','NS')]
    ranks = []
    for year, snap, winner in SNAPS:
        s = mr['ratings'][str(year)]['snapshots'][snap]['teams']
        items = sorted(s.items(), key=lambda x:-x[1]['overall_rating'])
        rk = next((i+1 for i,(t,_) in enumerate(items) if t==winner), 50)
        ranks.append(rk)
    avg_rank = sum(ranks)/len(ranks)

    # High-confidence calibration: bin (0.6, 0.7], (0.7, 1.0]
    hp = np.array(te_all_p); hy = np.array(te_all_y)
    bin_70_80 = (hp >= 0.7) & (hp < 0.8)
    bin_60_70 = (hp >= 0.6) & (hp < 0.7)
    cal_70 = (hy[bin_70_80].mean() if bin_70_80.any() else None, int(bin_70_80.sum()))
    cal_60 = (hy[bin_60_70].mean() if bin_60_70.any() else None, int(bin_60_70.sum()))

    errs = np.abs(np.array(te_all_p) - np.array(te_all_y))

    return {
        'label': label, 'cfg': cfg,
        'train_all':   brier(tr_all_p, tr_all_y),
        'train_cross': brier(tr_x_p, tr_x_y) if tr_x_p else None,
        'test_all':    brier(te_all_p, te_all_y),
        'test_dom':    brier(te_d_p, te_d_y) if te_d_p else None,
        'test_cross':  brier(te_x_p, te_x_y) if te_x_p else None,
        'test_logloss': logloss(te_all_p, te_all_y),
        'test_ece':    ece(te_all_p, te_all_y),
        'err_sigma':   float(errs.std()),
        'err_p90':     float(np.percentile(errs, 90)),
        'winner_ranks': ranks,
        'avg_rank':    avg_rank,
        'cal_60_70':   cal_60,
        'cal_70_80':   cal_70,
    }


def main():
    matches = load_match_data()
    configs = [
        ('PROD',  dict()),
        ('BRIER', dict(INTL_WIN_MULT=1.0, CHAMPIONS_MULT=2.0, ROSTER_PERSISTENCE=0.7)),
        ('BLEND', dict(INTL_WIN_MULT=1.0, CHAMPIONS_MULT=3.0, ROSTER_PERSISTENCE=0.5)),
        ('SOFT',  dict(INTL_WIN_MULT=1.5, CHAMPIONS_MULT=4.0, ROSTER_PERSISTENCE=0.5)),
    ]
    results = []
    for label, cfg in configs:
        print(f'Evaluating {label}: {cfg}', flush=True)
        r = evaluate(matches, label, **cfg)
        results.append(r)

    print('\n━━━ HEAD-TO-HEAD ━━━\n')
    print(f'{"Metric":<22}  ' + '  '.join(f'{r["label"]:>9}' for r in results))
    print('-' * 70)
    for metric, fmt in [
        ('train_all',     '.5f'),
        ('train_cross',   '.5f'),
        ('test_all',      '.5f'),
        ('test_dom',      '.5f'),
        ('test_cross',    '.5f'),
        ('test_logloss',  '.5f'),
        ('test_ece',      '.4f'),
        ('err_sigma',     '.4f'),
        ('err_p90',       '.3f'),
        ('avg_rank',      '.2f'),
    ]:
        vals = [r[metric] for r in results]
        valstrs = []
        for v in vals:
            if v is None: valstrs.append(f'{"n/a":>9}')
            else: valstrs.append(f'{v:>9{fmt}}')
        print(f'{metric:<22}  ' + '  '.join(valstrs))

    print('\nWinner-rank breakdown:')
    print(f'{"Event":<28}  ' + '  '.join(f'{r["label"]:>5}' for r in results))
    for i, (y, s, w) in enumerate([(2024,'after_madrid','SEN'),(2024,'after_shanghai','GEN'),
                                    (2024,'after_champions','EDG'),(2025,'after_bangkok','T1'),
                                    (2025,'after_toronto','PRX'),(2025,'after_champions','NRG'),
                                    (2026,'after_santiago','NS')]):
        print(f'{w} {y} {s}'.ljust(28) + '  ' + '  '.join(f'#{r["winner_ranks"][i]:>4}' for r in results))

    print('\nHigh-confidence test-bin calibration:')
    print(f'  Bin           ' + '  '.join(f'{r["label"]:>14}' for r in results))
    for bin_name, key in [('[0.60-0.70)','cal_60_70'), ('[0.70-0.80)','cal_70_80')]:
        line = f'  {bin_name:<13}'
        for r in results:
            actual, n = r[key]
            if actual is None or n == 0:
                line += f'   {"n=0":>11}'
            else:
                line += f'  {actual:>6.3f} (n={n:>3})'
        print(line)

    # Restore baseline
    print('\nRestoring baseline...')
    rebuild()
    print('Done.')


if __name__ == '__main__':
    main()
