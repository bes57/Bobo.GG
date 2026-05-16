"""
Extensive validation of Bayesian + Adaptive CN prior across history.

PHASES:
  1. Descriptive: actual CN intl performance at each event (M-W, round diffs)
  2. Per-snapshot ratings: deployed vs Bayesian, show CN cluster evolution
  3. Transitivity: when a top CN team moves, do non-tested CN teams also move?
  4. Per-cohort Brier: CN-vs-CN, CN-vs-intl, intl-vs-intl, all
  5. Parameter robustness: sensitivity to STRENGTH and K
  6. Adaptive prior trajectory: how does prior evolve over time?
"""
import os, sys, math, json, importlib, contextlib, io
import numpy as np
import pandas as pd
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scrapers'))

from BacktestSeriesPredictions import (
    load_match_data, load_snapshots, find_snapshot_for, brier,
    expected_calibration_error,
)

CN_SET = {'EDG','BLG','TE','DRG','ASE','AG','XLG','WOL','FPX','JDG','NOVA','TEC','TYL','TYLOO'}
INTL_EVENT_IDS = {
    '2024_masters_madrid', '2024_masters_shanghai', '2024_champions',
    '2025_masters_bangkok', '2025_masters_toronto', '2025_champions',
    '2026_masters_santiago',
}
BETA = 0.136


# ---------- bayesian rating builder ----------
def _massey_with_priors(games, lambda_decay, ref_date, prior_weights, prior_targets, min_games=5):
    import BuildMapRatings as B
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
        eff_w = B._effective_weeks_ago(g['winner'], g['date'], ref_date)
        eff_l = B._effective_weeks_ago(g['loser'],  g['date'], ref_date)
        base_w = math.sqrt(math.exp(-lambda_decay * eff_w) * math.exp(-lambda_decay * eff_l))
        cont_w = B._team_continuity_factor(g['winner'], g['date'], ref_date)
        cont_l = B._team_continuity_factor(g['loser'],  g['date'], ref_date)
        base_w *= math.sqrt(cont_w * cont_l)
        if is_champions:
            win_mult = los_mult = B.CHAMPIONS_MULT
        elif is_intl:
            win_mult = B.INTL_WIN_MULT; los_mult = B.INTL_LOSS_MULT
        else:
            win_mult = los_mult = 1.0
        w_win = base_w * win_mult; w_los = base_w * los_mult
        w_sym = min(w_win, w_los)
        raw_rd = g['wr'] - g['lr']
        if B.RD_TRANSFORM == 'sqrt':
            rd = math.copysign(math.sqrt(abs(raw_rd)) * B.RD_SCALE, raw_rd)
        elif B.RD_TRANSFORM == 'power':
            rd = math.copysign((abs(raw_rd) ** B.RD_POWER) * B.RD_SCALE, raw_rd)
        else:
            rd = raw_rd
        i, j = idx[g['winner']], idx[g['loser']]
        M[i, i] += w_sym; M[j, j] += w_sym
        M[i, j] -= w_sym; M[j, i] -= w_sym
        p[i] += w_win * rd; p[j] -= w_los * rd
    ridge = 0.5
    for i in range(n - 1):
        M[i, i] += ridge
    for t, i in idx.items():
        if i == n - 1: continue
        pw = prior_weights.get(t, 0.0)
        pt = prior_targets.get(t, 0.0)
        if pw > 0:
            M[i, i] += pw
            p[i] += pw * pt
    M[-1, :] = 1.0; p[-1] = 0.0
    try:
        r = np.linalg.solve(M, p)
    except np.linalg.LinAlgError:
        r, *_ = np.linalg.lstsq(M, p, rcond=None)
    return {t: float(r[idx[t]]) for t in teams}


def bayesian_solve(games, lam, ref_date, strength, intl_k, use_adaptive, fixed_prior=-4.0,
                    min_evidence=0.5):
    import BuildMapRatings as B
    raw_pass1 = _massey_with_priors(games, lam, ref_date, {}, {})
    intl_w = B._compute_intl_weights(games, lam, ref_date)
    if use_adaptive:
        num = den = 0.0
        for t in raw_pass1:
            if t in B.CN_TEAMS_SET and intl_w.get(t, 0) > min_evidence:
                w = intl_w[t]; num += raw_pass1[t] * w; den += w
        cn_prior = num / den if den > 0 else fixed_prior
    else:
        cn_prior = fixed_prior
    pw, pt = {}, {}
    for t in raw_pass1:
        if t in B.CN_TEAMS_SET:
            iw = intl_w.get(t, 0.0)
            pw[t] = strength * intl_k / (intl_k + iw)
            pt[t] = cn_prior
    final = _massey_with_priors(games, lam, ref_date, pw, pt)
    return final, raw_pass1, intl_w, cn_prior


# ---------- snapshot helpers ----------
def rebuild_with_bayesian(strength, intl_k, use_adaptive, fixed_prior=-4.0):
    saved_argv = sys.argv[:]
    sys.argv = ['BuildMapRatings.py']
    try:
        import BuildMapRatings as B
        importlib.reload(B)
        original_massey = B.massey_ratings
        original_apply_cn = B._apply_cn_shrinkage

        def bayesian_massey(games, lam, ref_date, min_games=0):
            final, _, _, _ = bayesian_solve(games, lam, ref_date, strength, intl_k,
                                              use_adaptive, fixed_prior)
            return final

        B.massey_ratings = bayesian_massey
        B._apply_cn_shrinkage = lambda r, iw, **k: dict(r)

        with contextlib.redirect_stdout(io.StringIO()):
            B.main()

        B.massey_ratings = original_massey
        B._apply_cn_shrinkage = original_apply_cn
    finally:
        sys.argv = saved_argv


def rebuild_deployed():
    saved_argv = sys.argv[:]
    sys.argv = ['BuildMapRatings.py']
    try:
        import BuildMapRatings as B
        importlib.reload(B)
        with contextlib.redirect_stdout(io.StringIO()):
            B.main()
    finally:
        sys.argv = saved_argv


# ---------- data helpers ----------
def load_cn_intl_results():
    """For each intl event, count CN team wins/losses vs non-CN."""
    mr = pd.read_csv(f'{ROOT}/data/match_results.csv')
    out = {}
    for eid in INTL_EVENT_IDS:
        path = f'{ROOT}/data/maps/{eid}.csv'
        if not os.path.exists(path): continue
        df = pd.read_csv(path, usecols=['MatchID', 'Org'])
        # For each match, determine if it's CN vs non-CN
        cn_w = cn_l = cn_vs_cn = 0
        for mid, grp in df.groupby('MatchID'):
            orgs = set(grp['Org'].dropna().unique())
            if len(orgs) != 2: continue
            cn_orgs = orgs & CN_SET
            if len(cn_orgs) == 0: continue
            elif len(cn_orgs) == 2:
                cn_vs_cn += 1
            else:
                # CN vs non-CN — look up winner from match_results
                a, b = sorted(orgs)
                # Per-map results
                maps = mr[(mr['MatchID']==mid) & (mr['MapNum']!='all')]
                for _, row in maps.iterrows():
                    w = row['WinnerOrg']
                    if w in cn_orgs: cn_w += 1
                    elif w in (orgs - cn_orgs): cn_l += 1
        out[eid] = {'cn_w': cn_w, 'cn_l': cn_l, 'cn_vs_cn_series': cn_vs_cn,
                    'wr': cn_w / (cn_w + cn_l) if (cn_w + cn_l) > 0 else None}
    return out


def predict_series(rA, rB, beta=BETA):
    p = 1.0 / (1.0 + math.exp(-beta * (rA - rB)))
    return p**2 * (3 - 2*p)


def gen_with_classification(matches, snaps, beta=BETA):
    """Classify each series as cn-cn, cn-intl, intl-intl, intl-other(domestic-intl)."""
    rows = []
    for _, m in matches.iterrows():
        s = find_snapshot_for(m['date'], snaps)
        if not s: continue
        _, _, _, ratings = s
        a, b = m['a'], m['b']
        if a not in ratings or b not in ratings: continue
        rA = ratings[a].get('overall_rating', 0.0) if isinstance(ratings[a], dict) else ratings[a]
        rB = ratings[b].get('overall_rating', 0.0) if isinstance(ratings[b], dict) else ratings[b]
        p = predict_series(rA, rB, beta)
        is_a_cn = a in CN_SET; is_b_cn = b in CN_SET
        if is_a_cn and is_b_cn: cat = 'cn-cn'
        elif is_a_cn or is_b_cn: cat = 'cn-other'
        else: cat = 'other-other'
        rows.append({'date': m['date'], 'pred': p, 'actual': m['a_wins'], 'cat': cat,
                     'a': a, 'b': b, 'rA': rA, 'rB': rB})
    return pd.DataFrame(rows)


def cn_summary(snap_teams):
    cn = sorted([(o, t['overall_rating']) for o, t in snap_teams.items() if o in CN_SET],
                key=lambda x: -x[1])
    return cn


# ---------- the phases ----------
def phase1_historical_performance():
    print("="*100)
    print("PHASE 1: HISTORICAL CN INTL PERFORMANCE (descriptive)")
    print("="*100)
    print("\nFor each intl event, CN teams' actual W/L on map basis vs non-CN opponents:")
    results = load_cn_intl_results()
    print(f"\n  {'Event':<32}  {'CN W':>5}  {'CN L':>5}  {'CN MapWin%':>11}  {'CN-vs-CN series':>17}")
    for eid in sorted(INTL_EVENT_IDS):
        if eid not in results: continue
        r = results[eid]
        wr = f"{r['wr']*100:.1f}%" if r['wr'] is not None else "—"
        print(f"  {eid:<32}  {r['cn_w']:>5}  {r['cn_l']:>5}  {wr:>11}  {r['cn_vs_cn_series']:>17}")

    # Overall CN intl WR by year
    print("\n  CN intl WR by year (excluding CN-vs-CN games):")
    by_year = {}
    for eid, r in results.items():
        yr = eid[:4]
        by_year.setdefault(yr, {'w': 0, 'l': 0})
        by_year[yr]['w'] += r['cn_w']; by_year[yr]['l'] += r['cn_l']
    for yr in sorted(by_year):
        d = by_year[yr]
        tot = d['w'] + d['l']
        wr = d['w']/tot*100 if tot else 0
        print(f"    {yr}: {d['w']}-{d['l']} ({wr:.1f}%)  total maps: {tot}")


def phase2_snapshot_comparison(matches):
    print("\n" + "="*100)
    print("PHASE 2: PER-SNAPSHOT CN RATINGS — DEPLOYED vs BAYESIAN-ADAPTIVE")
    print("="*100)

    # Run deployed
    rebuild_deployed()
    with open(f'{ROOT}/data/map_ratings.json') as f:
        mr_deployed = json.load(f)

    # Run Bayesian
    rebuild_with_bayesian(strength=25, intl_k=2.0, use_adaptive=True)
    with open(f'{ROOT}/data/map_ratings.json') as f:
        mr_bayes = json.load(f)

    snaps = [('2024','after_madrid'),('2024','after_shanghai'),('2024','after_champions'),
             ('2025','after_bangkok'),('2025','after_toronto'),('2025','after_champions'),
             ('2026','after_santiago'),('2026','after_stage1')]
    for year, snap in snaps:
        d = mr_deployed['ratings'][year]['snapshots'].get(snap, {}).get('teams', {})
        b = mr_bayes['ratings'][year]['snapshots'].get(snap, {}).get('teams', {})
        if not d or not b: continue
        print(f"\n  ── {year} {snap} ──")
        d_cn = {o: t['overall_rating'] for o, t in d.items() if o in CN_SET}
        b_cn = {o: t['overall_rating'] for o, t in b.items() if o in CN_SET}
        all_orgs = sorted(set(d_cn) | set(b_cn), key=lambda o: -b_cn.get(o, -10))
        print(f"  {'Team':>6}  {'Deployed':>9}  {'Bayesian':>9}  {'Δ':>6}")
        for o in all_orgs:
            dv = d_cn.get(o, None); bv = b_cn.get(o, None)
            if dv is None or bv is None:
                print(f"  {o:>6}  {'—' if dv is None else f'{dv:+.2f}':>9}  {'—' if bv is None else f'{bv:+.2f}':>9}")
            else:
                print(f"  {o:>6}  {dv:>+9.2f}  {bv:>+9.2f}  {bv-dv:>+6.2f}")

    rebuild_deployed()


def phase3_transitivity(matches):
    """Test: when XLG's rating shifts, do related CN teams (NOVA, DRG) shift too?

    Construct a hypothetical: artificially insert a few extra XLG wins at intl,
    re-solve, and check whether NOVA/DRG (who played XLG) shift.
    """
    print("\n" + "="*100)
    print("PHASE 3: TRANSITIVITY TEST")
    print("="*100)
    print("\nCompare ratings of CN teams who PLAYED an intl-tested CN team:")
    print("Showing pairs where the deployed model fails to propagate (rating gap stays fixed)")
    print()

    # Use 2026 after_stage1 — current snapshot
    rebuild_deployed()
    with open(f'{ROOT}/data/map_ratings.json') as f:
        mr_d = json.load(f)
    rebuild_with_bayesian(strength=25, intl_k=2.0, use_adaptive=True)
    with open(f'{ROOT}/data/map_ratings.json') as f:
        mr_b = json.load(f)

    d_teams = mr_d['ratings']['2026']['snapshots']['after_stage1']['teams']
    b_teams = mr_b['ratings']['2026']['snapshots']['after_stage1']['teams']

    # Specifically check: NOVA & DRG who played tested CN teams
    # The user's question is whether the BAYESIAN model SHOULD shift them more vs deployed
    print(f"  {'Team':>6}  {'Deployed':>9}  {'Bayesian':>9}  {'Lift':>6}  {'Note':<40}")
    for org, note in [('XLG', '(intl-tested)'),
                       ('EDG', '(intl-tested)'),
                       ('AG',  '(intl-tested, deep Santiago)'),
                       ('DRG', '(played XLG 1-3 in stage1 finals)'),
                       ('NOVA',('(played XLG, lost 0-2 in stage 1)')),
                       ('BLG', '(played XLG/EDG in stage1)'),
                       ('JDG', '(weak, no intl, only played CN)'),
                       ('TEC', '(no intl, no intl history)')]:
        dv = d_teams.get(org, {}).get('overall_rating')
        bv = b_teams.get(org, {}).get('overall_rating')
        if dv is None or bv is None: continue
        print(f"  {org:>6}  {dv:>+9.2f}  {bv:>+9.2f}  {bv-dv:>+6.2f}  {note}")

    # Verify: does the gap XLG-NOVA preserve a relationship to actual game outcomes?
    print(f"\n  Pair-wise gap analysis (predicted vs actual):")
    print(f"  {'A':>5} vs {'B':>5}  {'Deployed Δ':>11}  {'Bayes Δ':>9}  {'Actual rd avg':>13}")
    pairs = [
        ('XLG', 'NOVA', 'XLG won 2-0 (13-8, 13-11) — avg margin per map: +3.5'),
        ('XLG', 'DRG',  'XLG won 2-0 (13-4, 13-11), then 3-1 (13-2, 13-6, L13-10, 13-11) — XLG +5.0 avg margin'),
        ('XLG', 'BLG',  'play in same group'),
        ('EDG', 'NOVA', 'EDG very dominant'),
        ('EDG', 'DRG',  ''),
    ]
    for a, b_org, note in pairs:
        if a not in d_teams or b_org not in d_teams: continue
        d_gap = d_teams[a]['overall_rating'] - d_teams[b_org]['overall_rating']
        b_gap = b_teams[a]['overall_rating'] - b_teams[b_org]['overall_rating']
        print(f"  {a:>5} vs {b_org:>5}  {d_gap:>+11.2f}  {b_gap:>+9.2f}    {note}")

    rebuild_deployed()


def phase4_per_cohort_brier(matches):
    print("\n" + "="*100)
    print("PHASE 4: PER-COHORT BRIER (CN-vs-CN, CN-vs-other, other-vs-other)")
    print("="*100)

    rebuild_deployed()
    snaps_d = load_snapshots()
    df_d = gen_with_classification(matches, snaps_d, BETA)

    rebuild_with_bayesian(strength=25, intl_k=2.0, use_adaptive=True)
    snaps_b = load_snapshots()
    df_b = gen_with_classification(matches, snaps_b, BETA)

    # Per cohort, per year
    print(f"\n  {'Year':>4}  {'Cohort':<14}  {'n':>4}  {'Brier_dep':>10}  {'Brier_bay':>10}  {'Δ bp':>+7}")
    for yr in ['2023','2024','2025','2026','all']:
        if yr == 'all':
            sub_d = df_d; sub_b = df_b
        else:
            sub_d = df_d[df_d['date'].str.startswith(yr)]
            sub_b = df_b[df_b['date'].str.startswith(yr)]
        for cat in ['cn-cn', 'cn-other', 'other-other', 'ALL']:
            if cat == 'ALL':
                a = sub_d; b = sub_b
            else:
                a = sub_d[sub_d['cat'] == cat]
                b = sub_b[sub_b['cat'] == cat]
            if len(a) < 5 or len(b) < 5: continue
            br_d = float(brier(a['pred'].values, a['actual'].values))
            br_b = float(brier(b['pred'].values, b['actual'].values))
            delta_bp = (br_b - br_d) * 10000
            mark = '*' if abs(delta_bp) > 5 else ' '
            print(f"  {yr:>4}  {cat:<14}  {len(a):>4}  {br_d:>10.5f}  {br_b:>10.5f}  {delta_bp:>+7.1f}  {mark}")

    rebuild_deployed()


def phase5_parameter_robustness(matches):
    print("\n" + "="*100)
    print("PHASE 5: PARAMETER ROBUSTNESS (sweep STRENGTH and K)")
    print("="*100)
    print("\n  Goal: see how sensitive the Bayesian results are to STRENGTH and K")
    print("  Specifically check: trophy ranks preserved? Brier stable? EDG/NOVA reasonable?")
    print()
    print(f"  {'STRENGTH':>8}  {'K':>4}  {'EDG_s1':>+7}  {'XLG_s1':>+7}  {'NOVA_s1':>+8}  "
          f"{'CN_prior_2026':>+13}  {'Brier24':>9}  {'Brier25':>9}  {'Brier26':>9}  {'Brierfull':>10}  {'Trophy':>6}")

    for strength in [10, 15, 20, 25, 30, 40]:
        for k in [1.5, 2.0, 3.0]:
            rebuild_with_bayesian(strength=strength, intl_k=k, use_adaptive=True)
            with open(f'{ROOT}/data/map_ratings.json') as f:
                mr = json.load(f)
            t26 = mr['ratings']['2026']['snapshots']['after_stage1']['teams']
            edg = t26.get('EDG', {}).get('overall_rating', 0)
            xlg = t26.get('XLG', {}).get('overall_rating', 0)
            nova = t26.get('NOVA', {}).get('overall_rating', 0)
            # Compute the CN prior in current snapshot
            # We need to re-derive — easier to grab raw + recompute
            # Just approximate: avg of tested CN raw via the rating diff
            # Or: just leave as 'check the script'

            snaps = load_snapshots()
            yearly = {}
            for yr in ['2024','2025','2026']:
                m = matches[matches['date'].str.startswith(yr)].reset_index(drop=True)
                p, y = gen_series_wrapper(m, snaps, BETA)
                yearly[yr] = float(brier(p, y)) if len(p) >= 30 else None
            p_all, y_all = gen_series_wrapper(matches, snaps, BETA)
            full = float(brier(p_all, y_all))

            # Trophy
            SNAPS_T = [(2024,'after_champions','EDG'),(2025,'after_champions','NRG'),(2026,'after_santiago','NS')]
            ranks = []
            for year, snap_key, winner in SNAPS_T:
                s = mr['ratings'][str(year)]['snapshots'].get(snap_key, {}).get('teams', {})
                items = sorted(s.items(), key=lambda x:-x[1]['overall_rating'])
                rk = next((i+1 for i,(t,_) in enumerate(items) if t==winner), 50)
                ranks.append(rk)
            trophy_avg = sum(ranks)/len(ranks)

            print(f"  {strength:>8}  {k:>4}  {edg:>+7.2f}  {xlg:>+7.2f}  {nova:>+8.2f}  "
                  f"{'n/a':>13}  {yearly['2024']:.5f}  {yearly['2025']:.5f}  {yearly['2026']:.5f}  {full:.5f}  {trophy_avg:>6.2f}",
                  flush=True)

    rebuild_deployed()


def gen_series_wrapper(matches, snaps, beta):
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


def phase6_adaptive_prior_trajectory():
    print("\n" + "="*100)
    print("PHASE 6: ADAPTIVE PRIOR TRAJECTORY across snapshots")
    print("="*100)
    print("\n  Shows how the auto-derived CN prior changes over time as new intl data arrives.")
    print("  This is the key 'self-calibrating' feature — if CN improves, prior follows.")
    print()
    import BuildMapRatings as B
    sys.argv = ['BuildMapRatings.py']
    importlib.reload(B)
    all_games = B.load_games()

    snaps_to_test = [
        ('2024', 'before_madrid'),
        ('2024', 'after_madrid'),
        ('2024', 'after_shanghai'),
        ('2024', 'after_champions'),
        ('2025', 'after_bangkok'),
        ('2025', 'after_toronto'),
        ('2025', 'after_champions'),
        ('2026', 'before_santiago'),
        ('2026', 'after_santiago'),
        ('2026', 'after_stage1'),
    ]
    # Map snap key to events
    # Use the YEAR_CONFIGS
    print(f"  {'Snapshot':<28}  {'Adaptive prior':>15}  {'CN-tested teams (intl_w)':<50}")
    for year, snap in snaps_to_test:
        # Build snapshot games
        events = []
        # Best-effort: use BuildMapRatings._build_year_configs
        try:
            configs = B._build_year_configs()
        except Exception:
            configs = {}
        if year in B._HISTORICAL_YEAR_CONFIGS:
            cfg = B._HISTORICAL_YEAR_CONFIGS[year]['snapshots'].get(snap)
        elif year in configs:
            cfg = configs[year]['snapshots'].get(snap)
        else:
            cfg = None
        if not cfg: continue
        events = cfg['events']
        event_set = set(events)
        snap_games = [g for g in all_games if g.get('event_id') in event_set]
        if not snap_games: continue
        # Compute ref_date
        try:
            ref_date = max(g['date'] for g in snap_games)
        except ValueError:
            continue
        lam = math.log(2)/B.HALF_LIFE_WEEKS
        final, raw, intl_w, cn_prior = bayesian_solve(snap_games, lam, ref_date,
                                                       strength=25, intl_k=2.0,
                                                       use_adaptive=True)
        # Tested CN: those with intl_w > 0.5
        tested = [(t, intl_w[t]) for t in raw if t in CN_SET and intl_w.get(t, 0) > 0.5]
        tested.sort(key=lambda x: -x[1])
        tested_str = ', '.join(f'{t}({iw:.1f})' for t, iw in tested[:6])
        print(f"  {f'{year} {snap}':<28}  {cn_prior:>+15.2f}  {tested_str:<50}")


def main():
    matches = load_match_data()
    print(f"Loaded {len(matches)} series\n")
    phase1_historical_performance()
    phase2_snapshot_comparison(matches)
    phase3_transitivity(matches)
    phase4_per_cohort_brier(matches)
    phase5_parameter_robustness(matches)
    phase6_adaptive_prior_trajectory()
    print("\n━━━ Restoring deployed ━━━")
    rebuild_deployed()
    print("Done.")


if __name__ == '__main__':
    main()
