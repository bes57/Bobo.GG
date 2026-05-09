"""
TestModels.py — Compare rating model variants for predictive accuracy and face validity.

Variants tested:
  Baseline  : current model, symmetric INTL_MULT=4.0 on all international games
  Option 1  : domestic-only Massey (exclude all international games)
  Option 2  : asymmetric multiplier — intl wins get win_mult, intl losses get loss_mult

For Option 2, intl game matrix contributions:
  M[winner, winner] += w * win_mult
  M[loser,  loser]  += w * loss_mult
  M[winner, loser]  -= w * min(win_mult, loss_mult)   (symmetric off-diag, PSD guaranteed)
  p[winner]         += w * win_mult * rd
  p[loser]          -= w * loss_mult * rd

Usage: python3 scrapers/TestModels.py
"""

import os, sys, math, json
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime, timedelta
from scipy.special import expit
from scipy.optimize import minimize_scalar

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from MoreTestingMaybeFiles import ALL_EVENTS

INTL_EVENTS = {
    '2023_lock_in', '2023_masters_tokyo', '2023_champions',
    '2024_masters_madrid', '2024_masters_shanghai', '2024_champions',
    '2025_masters_bangkok', '2025_masters_toronto', '2025_champions',
    '2026_masters_santiago',
}

EVENT_DATES = {
    '2023_lock_in':           ('2023-01-10', '2023-02-12'),
    '2023_league':            ('2023-01-23', '2023-10-01'),
    '2023_masters_tokyo':     ('2023-06-11', '2023-06-25'),
    '2023_champions':         ('2023-08-06', '2023-08-27'),
    '2024_kickoff':           ('2024-01-08', '2024-02-11'),
    '2024_masters_madrid':    ('2024-02-14', '2024-03-10'),
    '2024_stage1':            ('2024-03-15', '2024-05-19'),
    '2024_masters_shanghai':  ('2024-06-02', '2024-06-16'),
    '2024_stage2':            ('2024-06-20', '2024-08-25'),
    '2024_champions':         ('2024-08-01', '2024-09-22'),
    '2025_kickoff':           ('2025-01-13', '2025-02-09'),
    '2025_masters_bangkok':   ('2025-02-12', '2025-03-09'),
    '2025_stage1':            ('2025-03-14', '2025-05-18'),
    '2025_masters_toronto':   ('2025-06-07', '2025-06-29'),
    '2025_stage2':            ('2025-07-14', '2025-08-24'),
    '2025_champions':         ('2025-08-28', '2025-09-21'),
    '2026_kickoff':           ('2026-01-07', '2026-02-09'),
    '2026_masters_santiago':  ('2026-03-26', '2026-04-06'),
}

TRAIN_EVENTS = [
    '2023_lock_in', '2023_league', '2023_masters_tokyo', '2023_champions',
    '2024_kickoff', '2024_masters_madrid', '2024_stage1',
    '2024_masters_shanghai', '2024_stage2', '2024_champions',
]
TEST_EVENTS = [
    '2025_kickoff', '2025_masters_bangkok', '2025_stage1',
    '2025_masters_toronto', '2025_stage2', '2025_champions',
    '2026_kickoff', '2026_masters_santiago',
]

ORG_REGIONS = {
    "TL":"EMEA","FNC":"EMEA","NAVI":"EMEA","VIT":"EMEA","BBL":"EMEA",
    "GX":"EMEA","KC":"EMEA","TH":"EMEA","FUT":"EMEA","GIA":"EMEA",
    "MKOI":"EMEA","WOL":"EMEA","M8":"EMEA","FPX":"EMEA","BLD":"EMEA",
    "SEN":"Americas","G2":"Americas","MIBR":"Americas","NRG":"Americas",
    "100T":"Americas","C9":"Americas","EG":"Americas","KRÜ":"Americas",
    "LEV":"Americas","FUR":"Americas","LOUD":"Americas","BME":"Americas",
    "2G":"Americas",
    "PRX":"Pacific","T1":"Pacific","TLN":"Pacific","GEN":"Pacific",
    "DFM":"Pacific","ZETA":"Pacific","RRQ":"Pacific","TS":"Pacific",
    "GE":"Pacific","KRX":"Pacific","NS":"Pacific","APK":"Pacific",
    "EDG":"CN","BLG":"CN","TE":"CN","DRG":"CN","ASE":"CN",
    "XLG":"CN","AG":"CN","NS":"CN",
}
# NS is CN, not Pacific — fix overlap
ORG_REGIONS["NS"] = "CN"

DATA_DIR = os.path.join(ROOT, 'data')


# ── Data loading ───────────────────────────────────────────────────────────────

def _parse_date(s):
    return datetime.strptime(s, '%Y-%m-%d')


def load_games():
    mr = pd.read_csv(os.path.join(DATA_DIR, 'match_results.csv'))
    mr = mr[mr['MapNum'] != 'all'].copy()
    mr['MapNum'] = mr['MapNum'].astype(str)
    mr_idx = mr.set_index(['MatchID', 'MapNum'])

    frames = []
    for event in ALL_EVENTS:
        eid = event['id']
        if eid not in EVENT_DATES:
            continue
        path = os.path.join(DATA_DIR, 'maps', f'{eid}.csv')
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df['event_id'] = eid
        df['MapNum'] = df['MapNum'].astype(str)
        df['MapName'] = df['MapName'].str.replace('PICK', '', regex=False).str.strip()
        frames.append(df)

    all_maps = pd.concat(frames, ignore_index=True)
    meta = all_maps.groupby(['MatchID', 'MapNum']).agg(
        orgs=('Org', lambda x: list(x.unique())),
        map_name=('MapName', 'first'),
        event_id=('event_id', 'first'),
    ).reset_index()

    games = []
    for _, row in meta.iterrows():
        key = (int(row['MatchID']), row['MapNum'])
        if key not in mr_idx.index:
            continue
        mr_row = mr_idx.loc[key]
        winner = mr_row['WinnerOrg']
        losers = [o for o in row['orgs'] if o != winner]
        if not losers:
            continue
        try:
            wr, lr = map(int, str(mr_row['Score']).split('-'))
        except Exception:
            continue
        games.append({
            'match_id': int(row['MatchID']),
            'event_id': row['event_id'],
            'map_name': row['map_name'],
            'winner': winner,
            'loser': losers[0],
            'wr': wr, 'lr': lr,
            'date': None,
        })

    gdf = pd.DataFrame(games)
    for eid, (s, e) in EVENT_DATES.items():
        mask = gdf['event_id'] == eid
        if not mask.any():
            continue
        start, end = _parse_date(s), _parse_date(e)
        span = (end - start).days
        mids = gdf.loc[mask, 'match_id'].values
        su = sorted(set(mids))
        rm = {m: i for i, m in enumerate(su)}
        mr2 = max(rm.values()) or 1
        for idx2, mid in zip(gdf.index[mask].tolist(), mids):
            gdf.at[idx2, 'date'] = start + timedelta(days=int(rm[mid] / mr2 * span))

    gdf = gdf.dropna(subset=['date'])
    return gdf.to_dict('records')


# ── Core solver (supports all three modes) ─────────────────────────────────────

def massey_solve(games, lam, ref_date, mode='symmetric',
                 win_mult=4.0, loss_mult=4.0, min_games=3):
    """
    mode='symmetric'  : intl games use win_mult=loss_mult (current baseline)
    mode='exclude'    : Option 1 — drop intl games entirely
    mode='asymmetric' : Option 2 — intl wins get win_mult, losses get loss_mult
    """
    if mode == 'exclude':
        games = [g for g in games if g['event_id'] not in INTL_EVENTS]

    if not games:
        return {}

    teams = sorted({g['winner'] for g in games} | {g['loser'] for g in games})

    if min_games > 0:
        counts = defaultdict(int)
        for g in games:
            counts[g['winner']] += 1
            counts[g['loser']]  += 1
        teams = [t for t in teams if counts[t] >= min_games]
        games  = [g for g in games if g['winner'] in teams and g['loser'] in teams]
        if not games:
            return {}

    n   = len(teams)
    idx = {t: i for i, t in enumerate(teams)}
    M   = np.zeros((n, n))
    p   = np.zeros(n)

    for g in games:
        if g['winner'] not in idx or g['loser'] not in idx:
            continue
        weeks_ago = max(0.0, (ref_date - g['date']).days / 7.0)
        w_base    = math.exp(-lam * weeks_ago)
        rd        = g['wr'] - g['lr']
        i, j      = idx[g['winner']], idx[g['loser']]

        is_intl = g['event_id'] in INTL_EVENTS
        if is_intl and mode != 'exclude':
            if mode == 'symmetric':
                w_i = w_j = w_base * win_mult
            else:  # asymmetric
                w_i = w_base * win_mult
                w_j = w_base * loss_mult
        else:
            w_i = w_j = w_base

        w_off = min(w_i, w_j)  # min ensures PSD per-game contribution

        M[i, i] += w_i;  M[j, j] += w_j
        M[i, j] -= w_off; M[j, i] -= w_off
        p[i]    += w_i * rd
        p[j]    -= w_j * rd

    M[-1, :] = 1.0; p[-1] = 0.0
    for k in range(n - 1):
        M[k, k] += 1e-4

    try:
        r = np.linalg.solve(M, p)
    except np.linalg.LinAlgError:
        r, *_ = np.linalg.lstsq(M, p, rcond=None)

    return {t: float(r[idx[t]]) for t in teams}


# ── Metrics helpers ────────────────────────────────────────────────────────────

def fit_beta(games, ratings):
    diffs, outs = [], []
    for g in games:
        d = ratings.get(g['winner'], 0.0) - ratings.get(g['loser'], 0.0)
        diffs += [d, -d]; outs += [1.0, 0.0]
    diffs = np.array(diffs); outs = np.array(outs)
    def nll(b):
        pr = np.clip(expit(b * diffs), 1e-9, 1-1e-9)
        return -np.mean(outs * np.log(pr) + (1-outs) * np.log(1-pr))
    return float(minimize_scalar(nll, bounds=(0.01, 10.0), method='bounded').x)


def brier(preds):
    if not preds: return 0.25
    return float(np.mean([(p - 1.0)**2 for p in preds]))


def logloss(preds):
    if not preds: return math.log(2)
    return float(np.mean([-math.log(max(p, 1e-9)) for p in preds]))


def is_xreg(g):
    r1 = ORG_REGIONS.get(g['winner'], '')
    r2 = ORG_REGIONS.get(g['loser'],  '')
    return r1 and r2 and r1 != r2


# ── Rolling-window CV evaluator ─────────────────────────────────────────────────

def cv_evaluate(all_games, lam, mode, win_mult=4.0, loss_mult=4.0):
    """
    Rolling-window CV: train on TRAIN_EVENTS + all earlier TEST_EVENTS,
    predict each TEST_EVENT game in chronological order.
    Returns per-game predictions bucketed by type.
    """
    event_min = {}
    for g in all_games:
        e = g['event_id']
        if e not in event_min or g['date'] < event_min[e]:
            event_min[e] = g['date']

    test_set = set(TEST_EVENTS)
    ordered_test = sorted(test_set, key=lambda e: event_min.get(e, datetime.max))

    by_event = defaultdict(list)
    for g in all_games:
        by_event[g['event_id']].append(g)

    train = [g for g in all_games if g['event_id'] in set(TRAIN_EVENTS)]
    preds_all = []; preds_dom = []; preds_intl = []; preds_xreg = []

    for eid in ordered_test:
        test_g = by_event.get(eid, [])
        if not test_g or not train:
            train += test_g; continue

        ref_date = max(g['date'] for g in train)
        rtgs = massey_solve(train, lam, ref_date, mode=mode,
                            win_mult=win_mult, loss_mult=loss_mult, min_games=3)
        if not rtgs:
            train += test_g; continue

        beta = fit_beta(train, rtgs)

        for g in test_g:
            d = rtgs.get(g['winner'], 0.0) - rtgs.get(g['loser'], 0.0)
            pr = float(np.clip(expit(beta * d), 1e-9, 1-1e-9))
            preds_all.append(pr)
            if g['event_id'] in INTL_EVENTS:
                preds_intl.append(pr)
                if is_xreg(g):
                    preds_xreg.append(pr)
            else:
                preds_dom.append(pr)

        train += test_g

    return {
        'all':    {'brier': brier(preds_all),  'n': len(preds_all)},
        'dom':    {'brier': brier(preds_dom),  'n': len(preds_dom)},
        'intl':   {'brier': brier(preds_intl), 'n': len(preds_intl)},
        'xreg':   {'brier': brier(preds_xreg), 'n': len(preds_xreg)},
    }


# ── Rankings builder ───────────────────────────────────────────────────────────

def get_rankings(all_games, events, lam, mode, win_mult, loss_mult):
    snap_games = [g for g in all_games if g['event_id'] in set(events)]
    if not snap_games:
        return {}
    ref = max(g['date'] for g in snap_games)
    rtgs = massey_solve(snap_games, lam, ref, mode=mode,
                        win_mult=win_mult, loss_mult=loss_mult, min_games=5)
    return dict(sorted(rtgs.items(), key=lambda x: -x[1]))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print('Loading games...')
    all_games = load_games()
    print(f'  {len(all_games)} map games\n')

    BASE_LAM = math.log(2) / 6.0

    # ─────────────────────────────────────────────────────────────────────────
    print('=' * 70)
    print('PART 1 — BASELINE vs OPTION 1 vs OPTION 2 (representative configs)')
    print('         Rolling-window CV on 2025–2026 test events')
    print('=' * 70)

    quick_configs = [
        ('Baseline  (sym 4×)',    'symmetric',  4.0, 4.0),
        ('Option 1  (dom-only)',  'exclude',    1.0, 1.0),
        ('Opt2 win4 loss1',       'asymmetric', 4.0, 1.0),
        ('Opt2 win4 loss0.5',     'asymmetric', 4.0, 0.5),
        ('Opt2 win6 loss1',       'asymmetric', 6.0, 1.0),
        ('Opt2 win6 loss0.5',     'asymmetric', 6.0, 0.5),
        ('Opt2 win3 loss1',       'asymmetric', 3.0, 1.0),
        ('Opt2 win3 loss0.5',     'asymmetric', 3.0, 0.5),
        ('Opt2 win2 loss1',       'asymmetric', 2.0, 1.0),
        ('Opt2 win4 loss0 (extreme)', 'asymmetric', 4.0, 0.0),
    ]

    print(f'\n  {"Model":28}  {"All":>7}  {"Dom":>7}  {"Intl":>7}  {"Xreg":>7}  n')
    print(f'  {"-"*28}  {"-"*7}  {"-"*7}  {"-"*7}  {"-"*7}  ---')
    print(f'  {"Random (baseline ref)":28}  {"0.25000":>7}  {"0.25000":>7}  {"0.25000":>7}  {"0.25000":>7}')

    part1_results = {}
    for name, mode, wm, lm in quick_configs:
        r = cv_evaluate(all_games, BASE_LAM, mode, wm, lm)
        part1_results[name] = r
        print(f'  {name:28}  {r["all"]["brier"]:.5f}  {r["dom"]["brier"]:.5f}  '
              f'{r["intl"]["brier"]:.5f}  {r["xreg"]["brier"]:.5f}  {r["all"]["n"]}')

    # ─────────────────────────────────────────────────────────────────────────
    print(f'\n{"=" * 70}')
    print('PART 2 — OPTION 2 FULL GRID SEARCH  (fixed λ = 6w half-life)')
    print('         Metric: cross-regional Brier (the key failure mode)')
    print('=' * 70)

    win_mults  = [1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
    loss_mults = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

    print(f'\n  {"win":>5}  {"loss":>5}  {"xreg":>8}  {"intl":>8}  {"all":>8}  {"dom":>8}')
    print(f'  {"-"*5}  {"-"*5}  {"-"*8}  {"-"*8}  {"-"*8}  {"-"*8}')

    grid_results = []
    for wm in win_mults:
        for lm in loss_mults:
            if lm > wm:
                continue  # loss > win makes no conceptual sense
            r = cv_evaluate(all_games, BASE_LAM, 'asymmetric', wm, lm)
            grid_results.append({
                'win': wm, 'loss': lm,
                'xreg': r['xreg']['brier'], 'intl': r['intl']['brier'],
                'all':  r['all']['brier'],  'dom':  r['dom']['brier'],
            })
            print(f'  {wm:5.1f}  {lm:5.2f}  {r["xreg"]["brier"]:.5f}   '
                  f'{r["intl"]["brier"]:.5f}   {r["all"]["brier"]:.5f}   '
                  f'{r["dom"]["brier"]:.5f}')
        print()

    best_xreg = min(grid_results, key=lambda x: x['xreg'])
    best_all  = min(grid_results, key=lambda x: x['all'])
    best_intl = min(grid_results, key=lambda x: x['intl'])

    print(f'  Baseline xreg Brier : {part1_results["Baseline  (sym 4×)"]["xreg"]["brier"]:.5f}')
    print(f'  Best xreg  : win={best_xreg["win"]}, loss={best_xreg["loss"]}  → {best_xreg["xreg"]:.5f}')
    print(f'  Best overall: win={best_all["win"]},  loss={best_all["loss"]}  → {best_all["all"]:.5f}')
    print(f'  Best intl  : win={best_intl["win"]}, loss={best_intl["loss"]}  → {best_intl["intl"]:.5f}')

    # ─────────────────────────────────────────────────────────────────────────
    print(f'\n{"=" * 70}')
    print('PART 3 — λ OPTIMIZATION for top Option 2 configs')
    print('=' * 70)

    # Take top-3 by xreg Brier and optimize λ for each
    top3 = sorted(grid_results, key=lambda x: x['xreg'])[:3]

    lam_opt_results = {}
    half_lives = [2, 3, 4, 5, 6, 7, 8, 10, 13, 16, 20, 26]

    for cfg in top3:
        wm, lm = cfg['win'], cfg['loss']
        print(f'\n  Optimizing λ for win={wm}, loss={lm} ...')
        print(f'  {"hl(w)":>6}  {"xreg":>8}  {"all":>8}')
        best_hl_r = None
        for hl in half_lives:
            lam_v = math.log(2) / hl
            r = cv_evaluate(all_games, lam_v, 'asymmetric', wm, lm)
            print(f'  {hl:6}  {r["xreg"]["brier"]:.5f}   {r["all"]["brier"]:.5f}')
            if best_hl_r is None or r['xreg']['brier'] < best_hl_r['xreg']['brier']:
                best_hl_r = {'hl': hl, 'lam': lam_v, **{k: v for k, v in r.items()}}
        lam_opt_results[(wm, lm)] = best_hl_r
        print(f'  → optimal hl = {best_hl_r["hl"]}w  (xreg={best_hl_r["xreg"]["brier"]:.5f})')

    # Also optimize λ for Option 1
    print(f'\n  Optimizing λ for Option 1 (dom-only) ...')
    print(f'  {"hl(w)":>6}  {"xreg":>8}  {"all":>8}')
    best_opt1 = None
    for hl in half_lives:
        lam_v = math.log(2) / hl
        r = cv_evaluate(all_games, lam_v, 'exclude', 1.0, 1.0)
        print(f'  {hl:6}  {r["xreg"]["brier"]:.5f}   {r["all"]["brier"]:.5f}')
        if best_opt1 is None or r['xreg']['brier'] < best_opt1['xreg']['brier']:
            best_opt1 = {'hl': hl, 'lam': lam_v, **{k: v for k, v in r.items()}}
    print(f'  → optimal hl = {best_opt1["hl"]}w  (xreg={best_opt1["xreg"]["brier"]:.5f})')

    # ─────────────────────────────────────────────────────────────────────────
    print(f'\n{"=" * 70}')
    print('PART 4 — ACCURACY SUMMARY (each model at its optimal λ)')
    print('=' * 70)

    best_cfg_key = min(lam_opt_results, key=lambda k: lam_opt_results[k]['xreg']['brier'])
    best_wm, best_lm = best_cfg_key
    best_lam_val = lam_opt_results[best_cfg_key]['lam']
    best_hl_val  = lam_opt_results[best_cfg_key]['hl']

    baseline_r = cv_evaluate(all_games, BASE_LAM,   'symmetric',  4.0, 4.0)
    opt1_r     = cv_evaluate(all_games, best_opt1['lam'], 'exclude', 1.0, 1.0)
    opt2_best_r = lam_opt_results[best_cfg_key]

    print(f'\n  {"Model":35}  {"All":>8}  {"Dom":>8}  {"Intl":>8}  {"Xreg":>8}')
    print(f'  {"-"*35}  {"-"*8}  {"-"*8}  {"-"*8}  {"-"*8}')
    print(f'  {"Random baseline":35}  {"0.25000":>8}  {"0.25000":>8}  {"0.25000":>8}  {"0.25000":>8}')

    def row(label, r):
        print(f'  {label:35}  {r["all"]["brier"]:.5f}   {r["dom"]["brier"]:.5f}   '
              f'{r["intl"]["brier"]:.5f}   {r["xreg"]["brier"]:.5f}')

    row(f'Baseline  (sym 4×, hl=6w)',      baseline_r)
    row(f'Option 1  (dom-only, hl={best_opt1["hl"]}w)',  opt1_r)
    row(f'Option 2  (win={best_wm}, loss={best_lm}, hl={best_hl_val}w)', opt2_best_r)

    # ─────────────────────────────────────────────────────────────────────────
    print(f'\n{"=" * 70}')
    print('PART 5 — FACE VALIDITY: 2025 After Champions Rankings')
    print('=' * 70)

    events_2025 = ['2025_kickoff','2025_masters_bangkok','2025_stage1',
                   '2025_masters_toronto','2025_stage2','2025_champions']

    baseline_rtgs = get_rankings(all_games, events_2025, BASE_LAM, 'symmetric', 4.0, 4.0)
    opt1_rtgs     = get_rankings(all_games, events_2025, best_opt1['lam'], 'exclude', 1.0, 1.0)
    opt2_rtgs     = get_rankings(all_games, events_2025, best_lam_val, 'asymmetric', best_wm, best_lm)

    def rank_map(rtgs):
        return {t: i+1 for i, t in enumerate(rtgs)}

    rm_b = rank_map(baseline_rtgs)
    rm_1 = rank_map(opt1_rtgs)
    rm_2 = rank_map(opt2_rtgs)

    all_teams = list(baseline_rtgs.keys())

    print(f'\n  {"#":>2}  {"Team":>6}  {"Baseline":>10}  {"Option 1":>10}  {"Option 2":>10}')
    print(f'  {"Baseline config — win=4 loss=4 hl=6w":55}')
    print(f'  {"Option 1   config — dom-only hl=" + str(best_opt1["hl"]) + "w":55}')
    print(f'  {"Option 2   config — win=" + str(best_wm) + " loss=" + str(best_lm) + " hl=" + str(best_hl_val) + "w":55}')
    print()
    print(f'  {"#":>2}  {"Team":>6}  {"Baseline":>10}  {"Option 1":>10}  {"Option 2":>10}')
    print(f'  {"--":>2}  {"------":>6}  {"----------":>10}  {"----------":>10}  {"----------":>10}')

    for i, team in enumerate(all_teams[:25], 1):
        r_b = baseline_rtgs.get(team, 0.0)
        r_1 = opt1_rtgs.get(team, 0.0)
        r_2 = opt2_rtgs.get(team, 0.0)
        rk1 = rm_1.get(team, 99)
        rk2 = rm_2.get(team, 99)
        print(f'  {i:2}. {team:>6}  {r_b:+7.3f}     #{rk1:2} {r_1:+7.3f}   #{rk2:2} {r_2:+7.3f}')

    print(f'\n  ── Focus: Gen.G, KRX, PRX ──')
    focus = ['GEN', 'KRX', 'PRX', 'T1', 'RRQ', 'GE']
    for team in focus:
        if team not in baseline_rtgs and team not in opt1_rtgs and team not in opt2_rtgs:
            continue
        rb = baseline_rtgs.get(team, 0.0)
        r1 = opt1_rtgs.get(team, 0.0)
        r2 = opt2_rtgs.get(team, 0.0)
        print(f'  {team:>6}:  baseline #{rm_b.get(team,"?"):>2} ({rb:+.3f})  '
              f'| opt1 #{rm_1.get(team,"?"):>2} ({r1:+.3f})  '
              f'| opt2 #{rm_2.get(team,"?"):>2} ({r2:+.3f})')

    print(f'\n  ── 2026 After Santiago — T1 vs GE/RRQ case ──')
    events_2026 = ['2026_kickoff', '2026_masters_santiago']
    b26  = get_rankings(all_games, events_2026, BASE_LAM, 'symmetric', 4.0, 4.0)
    o1_26 = get_rankings(all_games, events_2026, best_opt1['lam'], 'exclude', 1.0, 1.0)
    o2_26 = get_rankings(all_games, events_2026, best_lam_val, 'asymmetric', best_wm, best_lm)
    rm_b26 = rank_map(b26); rm_1_26 = rank_map(o1_26); rm_2_26 = rank_map(o2_26)

    for team in ['PRX', 'T1', 'RRQ', 'GE', 'NS', 'NRG', 'G2']:
        if team not in b26 and team not in o1_26 and team not in o2_26:
            continue
        rb = b26.get(team, 0.0)
        r1 = o1_26.get(team, 0.0)
        r2 = o2_26.get(team, 0.0)
        print(f'  {team:>6}:  baseline #{rm_b26.get(team,"?"):>2} ({rb:+.3f})  '
              f'| opt1 #{rm_1_26.get(team,"?"):>2} ({r1:+.3f})  '
              f'| opt2 #{rm_2_26.get(team,"?"):>2} ({r2:+.3f})')

    # ─────────────────────────────────────────────────────────────────────────
    print(f'\n{"=" * 70}')
    print('SUMMARY')
    print('=' * 70)
    print(f'  Baseline  (sym 4×,          hl=6w)  '
          f'xreg={baseline_r["xreg"]["brier"]:.5f}  all={baseline_r["all"]["brier"]:.5f}')
    print(f'  Option 1  (dom-only,         hl={best_opt1["hl"]}w)  '
          f'xreg={opt1_r["xreg"]["brier"]:.5f}  all={opt1_r["all"]["brier"]:.5f}')
    print(f'  Option 2  (win={best_wm} loss={best_lm}, hl={best_hl_val}w)  '
          f'xreg={opt2_best_r["xreg"]["brier"]:.5f}  all={opt2_best_r["all"]["brier"]:.5f}')
    print(f'\n  Random baseline: xreg=0.25000  all=0.25000')

    print(f'\n  Improvement over baseline:')
    b_xreg = baseline_r['xreg']['brier']
    print(f'  Option 1 xreg: {(b_xreg - opt1_r["xreg"]["brier"])*10000:+.1f} pts  '
          f'({"better" if opt1_r["xreg"]["brier"] < b_xreg else "worse"})')
    print(f'  Option 2 xreg: {(b_xreg - opt2_best_r["xreg"]["brier"])*10000:+.1f} pts  '
          f'({"better" if opt2_best_r["xreg"]["brier"] < b_xreg else "worse"})')


if __name__ == '__main__':
    main()
