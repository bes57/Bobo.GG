"""
BuildIntlCalibration.py

Computes cross-regional calibration offsets from VCT international events.

Architecture:
  global_rating(team, snap) = domestic_rating + regional_offset + individual_bonus

  regional_offset (δ_R): region-wide shift derived from how a region's teams
    performed vs expectations at internationals. Applies to ALL teams in the
    region, including those who didn't attend — so if EMEA underperforms at
    Bangkok, all EMEA teams' global ratings drop, not just attendees.

  individual_bonus (b_i): attending team's deviation above/below their region's
    average residual. Captures "Fnatic specifically is better than average EMEA"
    vs "EMEA just happened to be strong that event." Shrunk toward zero for
    teams with few games (noisy). Decays over time and carries across years
    weighted by roster similarity.

  attendees:    get δ_R + b_i  (full personal adjustment, region + individual)
  non-attendees: get δ_R only   (regional signal only)

  This avoids double-penalizing: if GX is exposed at Champions, their b_i is
  already negative; the regional δ_EMEA captures the collective EMEA signal.

Backtest:
  Holdout = Champions events (cross-regional games only).
  Compare domestic-only vs calibrated Brier/LogLoss.
  Also tests "always global" vs "global only for cross-regional matchups".

Output: data/intl_calibration.json
Usage:  python scrapers/BuildIntlCalibration.py
"""

import os, sys, json, math
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime
from scipy.special import expit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, 'data')

from EventLeaderboards import ORG_REGIONS

ACTIVE_REGIONS = {'EMEA', 'Americas', 'Pacific', 'CN'}

# ── International event config ──────────────────────────────────────────────────
# Maps each international event to the domestic snapshot that best represents
# team strength just before that event.
INTL_CONFIG = {
    '2023_masters_tokyo':  {'year': '2023', 'snap': 'after_tokyo',     'date': '2023-06-11'},
    '2023_champions':      {'year': '2023', 'snap': 'before_champions','date': '2023-08-06'},
    '2024_masters_madrid':   {'year': '2024', 'snap': 'before_madrid',   'date': '2024-02-14'},
    '2024_masters_shanghai': {'year': '2024', 'snap': 'before_shanghai', 'date': '2024-06-02'},
    '2024_champions':        {'year': '2024', 'snap': 'before_champions','date': '2024-08-01'},
    '2025_masters_bangkok':{'year': '2025', 'snap': 'before_bangkok',  'date': '2025-02-19'},
    '2025_masters_toronto':{'year': '2025', 'snap': 'before_toronto',  'date': '2025-06-19'},
    '2025_champions':      {'year': '2025', 'snap': 'before_champions','date': '2025-08-22'},
    '2026_masters_santiago':{'year':'2026', 'snap': 'before_santiago', 'date': '2026-03-26'},
}

# Which prior internationals feed into each (year, snap) calibration
# and approximately when that snapshot's "present" is
SNAP_CONFIG = {
    # 2023
    ('2023', 'after_tokyo'):      {'prior_intl': ['2023_masters_tokyo'],
                                   'date': '2023-06-26'},
    ('2023', 'before_champions'): {'prior_intl': ['2023_masters_tokyo'],
                                   'date': '2023-08-05'},
    ('2023', 'after_champions'):  {'prior_intl': ['2023_masters_tokyo', '2023_champions'],
                                   'date': '2023-09-01'},
    # 2024
    ('2024', 'before_madrid'):    {'prior_intl': [],
                                   'date': '2024-02-12'},
    ('2024', 'after_madrid'):     {'prior_intl': ['2024_masters_madrid'],
                                   'date': '2024-03-12'},
    ('2024', 'before_shanghai'):  {'prior_intl': ['2024_masters_madrid'],
                                   'date': '2024-06-01'},
    ('2024', 'after_shanghai'):   {'prior_intl': ['2024_masters_madrid', '2024_masters_shanghai'],
                                   'date': '2024-06-18'},
    ('2024', 'before_champions'): {'prior_intl': ['2024_masters_madrid', '2024_masters_shanghai'],
                                   'date': '2024-08-01'},
    ('2024', 'after_champions'):  {'prior_intl': ['2024_masters_madrid', '2024_masters_shanghai', '2024_champions'],
                                   'date': '2024-09-25'},
    # 2025
    ('2025', 'before_bangkok'):   {'prior_intl': [],
                                   'date': '2025-02-11'},
    ('2025', 'after_bangkok'):    {'prior_intl': ['2025_masters_bangkok'],
                                   'date': '2025-03-11'},
    ('2025', 'before_toronto'):   {'prior_intl': ['2025_masters_bangkok'],
                                   'date': '2025-06-06'},
    ('2025', 'after_toronto'):    {'prior_intl': ['2025_masters_bangkok', '2025_masters_toronto'],
                                   'date': '2025-07-01'},
    ('2025', 'before_champions'): {'prior_intl': ['2025_masters_bangkok', '2025_masters_toronto'],
                                   'date': '2025-08-27'},
    ('2025', 'after_champions'):  {'prior_intl': ['2025_masters_bangkok', '2025_masters_toronto', '2025_champions'],
                                   'date': '2025-09-25'},
    # 2026
    ('2026', 'before_santiago'):  {'prior_intl': [],
                                   'date': '2026-02-09'},
    ('2026', 'after_santiago'):   {'prior_intl': ['2026_masters_santiago'],
                                   'date': '2026-04-08'},
}

# Backtest holdouts: cross-regional games at these events test our predictions
BACKTEST_HOLDOUTS = ['2024_champions', '2025_champions']

# ── Tunable parameters (validated via backtest) ─────────────────────────────────
INTL_HL_WEEKS     = 10   # half-life of intl signal decay within a year (weeks)
REGIONAL_SCALE    = 0.12 # conservative — regional signal is noisy with few teams
INDIVIDUAL_SCALE  = 0.10 # conservative — individual signal is interpretable but volatile
INDIVIDUAL_SHRINK = 3    # James-Stein shrink for b_i (games of shrinkage)
YEAR_CARRY_BASE   = 0.45 # base carry of individual bonus across year boundary
                          # times roster similarity (0–1)


def _parse_date(s):
    return datetime.strptime(s, '%Y-%m-%d')


def _weeks_between(d1_str, d2_str):
    d1, d2 = _parse_date(d1_str), _parse_date(d2_str)
    return abs((d2 - d1).days) / 7.0


def _decay(weeks, hl=INTL_HL_WEEKS):
    return math.exp(-math.log(2) / hl * weeks)


# ── Load domestic ratings ────────────────────────────────────────────────────────

def load_ratings():
    path = os.path.join(DATA, 'map_ratings.json')
    with open(path) as f:
        return json.load(f)


def get_snap_ratings(ratings_data, year, snap):
    """Return {org: overall_rating} for (year, snap), or {} if missing."""
    try:
        return {
            org: d['overall_rating']
            for org, d in ratings_data['ratings'][year]['snapshots'][snap]['teams'].items()
        }
    except (KeyError, TypeError):
        return {}


# ── Load international game-level results ────────────────────────────────────────

def load_intl_games(event_id):
    """
    Load map-level match results for an international event.
    Returns list of {winner, loser, wr, lr, winner_region, loser_region}.
    Skips matches involving CN or unknown orgs, or same-region matches.
    """
    maps_path = os.path.join(DATA, 'maps', f'{event_id}.csv')
    mr_path   = os.path.join(DATA, 'match_results.csv')
    if not os.path.exists(maps_path):
        print(f'  [warn] no maps/{event_id}.csv')
        return []

    mr = pd.read_csv(mr_path)
    mr = mr[mr['MapNum'] != 'all'].copy()
    mr_idx = mr.set_index(['MatchID', 'MapNum'])

    df = pd.read_csv(maps_path)
    df['MapNum'] = df['MapNum'].astype(str)
    df['MapName'] = df['MapName'].str.replace('PICK', '', regex=False).str.strip()

    # Build MatchID,MapNum -> set of orgs
    meta = df.groupby(['MatchID', 'MapNum']).agg(
        orgs=('Org', lambda x: list(x.unique()))
    ).reset_index()

    games = []
    for _, row in meta.iterrows():
        key = (int(row['MatchID']), row['MapNum'])
        if key not in mr_idx.index:
            continue
        mr_row  = mr_idx.loc[key]
        winner  = mr_row['WinnerOrg']
        losers  = [o for o in row['orgs'] if o != winner]
        if not losers:
            continue
        loser = losers[0]
        wr_reg = ORG_REGIONS.get(winner)
        lr_reg = ORG_REGIONS.get(loser)
        # Skip CN, unknown, or same-region
        if not wr_reg or not lr_reg:
            continue
        if wr_reg not in ACTIVE_REGIONS or lr_reg not in ACTIVE_REGIONS:
            continue
        if wr_reg == lr_reg:
            continue
        try:
            wr, lr = map(int, str(mr_row['Score']).split('-'))
        except Exception:
            continue
        games.append({
            'winner': winner, 'loser': loser,
            'wr': wr, 'lr': lr,
            'winner_region': wr_reg, 'loser_region': lr_reg,
        })
    return games


# ── Compute residuals for one international event ────────────────────────────────

def compute_event_residuals(games, pre_ratings):
    """
    For each cross-regional map game, compute:
      residual = actual_round_diff - predicted_round_diff
    from winner's perspective. Returns per-team residuals.
    """
    team_residuals = defaultdict(list)

    for g in games:
        rw = pre_ratings.get(g['winner'])
        rl = pre_ratings.get(g['loser'])
        if rw is None or rl is None:
            continue  # one team unrated — skip

        actual_rd    = g['wr'] - g['lr']
        predicted_rd = rw - rl
        resid        = actual_rd - predicted_rd

        team_residuals[g['winner']].append(resid)
        team_residuals[g['loser']].append(-resid)  # loser's perspective

    return team_residuals


def residuals_to_offsets(team_residuals):
    """
    Decompose per-team residuals into:
      δ_R  = regional average residual (normalised so global mean = 0)
      b_i  = team's deviation from its region's average (shrunk)

    Returns (regional_offsets, individual_bonuses, details).
    """
    # Regional averages
    reg_resids = defaultdict(list)
    for team, resids in team_residuals.items():
        reg = ORG_REGIONS.get(team)
        if reg in ACTIVE_REGIONS:
            reg_resids[reg].extend(resids)

    raw_delta = {reg: float(np.mean(v)) for reg, v in reg_resids.items() if v}
    # Normalise: regional offsets are relative (mean across regions = 0)
    if raw_delta:
        global_mean = float(np.mean(list(raw_delta.values())))
        delta = {reg: raw_delta[reg] - global_mean for reg in raw_delta}
    else:
        delta = {}

    # Individual bonuses: team deviation above/below their region
    bonuses = {}
    details = {}
    for team, resids in team_residuals.items():
        reg = ORG_REGIONS.get(team)
        if reg not in ACTIVE_REGIONS:
            continue
        n          = len(resids)
        raw_b      = float(np.mean(resids)) - delta.get(reg, 0.0)
        # James-Stein shrinkage toward 0
        alpha      = n / (n + INDIVIDUAL_SHRINK)
        bonuses[team] = alpha * raw_b
        details[team] = {
            'n_games': n, 'mean_resid': round(float(np.mean(resids)), 3),
            'regional_offset': round(delta.get(reg, 0.0), 3),
            'raw_individual': round(raw_b, 3),
            'individual_bonus': round(bonuses[team], 3),
            'region': reg,
        }

    return delta, bonuses, details


# ── Roster similarity ────────────────────────────────────────────────────────────

def build_roster_map(year_events):
    """
    Returns {org: set_of_players} from the LAST domestic event of a year.
    Uses players from the maps CSVs.
    """
    rosters = {}
    for eid in reversed(year_events):
        path = os.path.join(DATA, 'maps', f'{eid}.csv')
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if 'Player' not in df.columns or 'Org' not in df.columns:
            continue
        for org, grp in df.groupby('Org'):
            if org in ORG_REGIONS and ORG_REGIONS[org] in ACTIVE_REGIONS:
                if org not in rosters:
                    rosters[org] = set(grp['Player'].dropna().unique())
    return rosters


def roster_carry_factor(org, from_rosters, to_rosters):
    """
    Fraction of 5-player lineup shared between end-of-year and start-of-year.
    """
    r_from = from_rosters.get(org, set())
    r_to   = to_rosters.get(org, set())
    if not r_from or not r_to:
        return 0.3  # assume partial carry if no data
    overlap = len(r_from & r_to)
    return overlap / 5.0


# ── Accumulate offsets across internationals with decay ──────────────────────────

def accumulate_offsets(ratings_data):
    """
    For every (year, snap) in SNAP_CONFIG, compute the accumulated
    regional_offsets and individual_bonuses by summing over prior internationals
    with temporal decay.

    Also handles cross-year carry for individual bonuses.
    """
    # Cache per-event results
    event_cache = {}

    # Domestic event ordering per year (for roster comparison)
    domestic_events_by_year = {
        '2023': ['2023_league'],
        '2024': ['2024_kickoff', '2024_stage1', '2024_stage2'],
        '2025': ['2025_kickoff', '2025_stage1', '2025_stage2'],
        '2026': ['2026_kickoff'],
    }

    def get_event_result(eid):
        if eid in event_cache:
            return event_cache[eid]
        cfg    = INTL_CONFIG.get(eid, {})
        yr, sn = cfg.get('year'), cfg.get('snap')
        pre    = get_snap_ratings(ratings_data, yr, sn) if yr and sn else {}
        games  = load_intl_games(eid)
        cross  = [g for g in games]  # already filtered in load_intl_games
        resids = compute_event_residuals(cross, pre)
        delta, bonuses, details = residuals_to_offsets(resids)
        event_cache[eid] = (delta, bonuses, details, games)
        return event_cache[eid]

    # Year-end individual bonuses (for carry into next year)
    prev_year_bonuses = {}  # org -> bonus at end of prior year

    calibration = {}

    for year in ['2023', '2024', '2025', '2026']:
        for snap_key, cfg in sorted(SNAP_CONFIG.items(), key=lambda x: x[1]['date']):
            yr, sn = snap_key
            if yr != year:
                continue

            snap_date    = cfg['date']
            prior_events = cfg['prior_intl']

            # ── Accumulate from prior internationals this year ──────────────────
            acc_regional  = defaultdict(float)
            acc_individual = defaultdict(float)

            for eid in prior_events:
                delta, bonuses, _, _ = get_event_result(eid)
                intl_date = INTL_CONFIG[eid]['date']
                w = _decay(_weeks_between(intl_date, snap_date))

                for reg, offset in delta.items():
                    acc_regional[reg] += offset * REGIONAL_SCALE * w

                for org, bonus in bonuses.items():
                    acc_individual[org] += bonus * INDIVIDUAL_SCALE * w

            # ── Carry individual bonuses from prior year ─────────────────────────
            if sn == list(SNAP_CONFIG.items())[0][1].get('first_snap') or True:
                # For early snap (no prior intl this year), blend in year-carry
                is_year_start = (len(prior_events) == 0 and year != '2023')
                if is_year_start and prev_year_bonuses.get(year):
                    from_rosters = build_roster_map(domestic_events_by_year.get(str(int(year)-1), []))
                    to_rosters   = build_roster_map(domestic_events_by_year.get(year, []))
                    for org, bonus in prev_year_bonuses[year].items():
                        cf = roster_carry_factor(org, from_rosters, to_rosters)
                        carried = bonus * YEAR_CARRY_BASE * cf
                        if abs(carried) > 0.05:
                            acc_individual[org] = acc_individual.get(org, 0.0) + carried

            key = f'{year}_{sn}'
            calibration[key] = {
                'regional_offsets':   {r: round(v, 4) for r, v in acc_regional.items()},
                'individual_bonuses': {o: round(v, 4) for o, v in acc_individual.items()},
            }

        # Store year-end bonuses for carry into next year
        # Use 'full' snap if available, else the last snap of the year
        year_snaps = [(s, c) for (y, s), c in SNAP_CONFIG.items() if y == year]
        year_snaps_sorted = sorted(year_snaps, key=lambda x: x[1]['date'])
        if year_snaps_sorted:
            last_snap = year_snaps_sorted[-1][0]
            last_key  = f'{year}_{last_snap}'
            if last_key in calibration:
                prev_year_bonuses[str(int(year) + 1)] = calibration[last_key]['individual_bonuses'].copy()

    return calibration, event_cache


# ── Backtest ─────────────────────────────────────────────────────────────────────

def run_backtest(calibration, ratings_data, event_cache):
    """
    For each Champions event, evaluate:
      - domestic_only: predict using regional Massey ratings
      - calibrated_global: add regional_offset + individual_bonus
      - calibrated_cross_only: use global only for cross-regional, domestic for same

    Returns list of result dicts.
    """
    results = []

    for eid in BACKTEST_HOLDOUTS:
        cfg       = INTL_CONFIG.get(eid, {})
        yr, sn    = cfg.get('year'), cfg.get('snap')
        if not yr or not sn:
            continue

        # Pre-event snapshot
        pre_rtgs = get_snap_ratings(ratings_data, yr, sn)
        snap_key = f'{yr}_{sn}'
        calib    = calibration.get(snap_key, {})
        reg_off  = calib.get('regional_offsets', {})
        ind_bon  = calib.get('individual_bonuses', {})

        # Beta from snapshot
        try:
            beta = ratings_data['ratings'][yr]['snapshots'][sn]['beta']
        except Exception:
            beta = 0.45

        games = load_intl_games(eid)
        if not games:
            continue

        brier_dom, brier_cal, brier_cross = [], [], []
        ll_dom,    ll_cal,    ll_cross    = [], [], []

        for g in games:
            rw_dom = pre_rtgs.get(g['winner'])
            rl_dom = pre_rtgs.get(g['loser'])
            if rw_dom is None or rl_dom is None:
                continue

            rw_reg = ORG_REGIONS.get(g['winner'], '')
            rl_reg = ORG_REGIONS.get(g['loser'],  '')

            # Global ratings
            rw_glob = rw_dom + reg_off.get(rw_reg, 0) + ind_bon.get(g['winner'], 0)
            rl_glob = rl_dom + reg_off.get(rl_reg, 0) + ind_bon.get(g['loser'],  0)

            # Cross-regional only (same-region → use domestic)
            cross_regional = (rw_reg != rl_reg)
            if cross_regional:
                rw_cross, rl_cross = rw_glob, rl_glob
            else:
                rw_cross, rl_cross = rw_dom, rl_dom

            def score_pair(rA, rB):
                p = float(np.clip(expit(beta * (rA - rB)), 1e-9, 1 - 1e-9))
                brier = (p - 1.0) ** 2
                ll    = -math.log(p)
                return brier, ll

            b_d, l_d = score_pair(rw_dom,  rl_dom)
            b_c, l_c = score_pair(rw_glob, rl_glob)
            b_x, l_x = score_pair(rw_cross, rl_cross)

            brier_dom.append(b_d);  ll_dom.append(l_d)
            brier_cal.append(b_c);  ll_cal.append(l_c)
            brier_cross.append(b_x); ll_cross.append(l_x)

        if not brier_dom:
            continue

        n = len(brier_dom)
        row = {
            'event':           eid,
            'n_games':         n,
            'brier_domestic':  round(float(np.mean(brier_dom)),  5),
            'brier_global':    round(float(np.mean(brier_cal)),  5),
            'brier_cross_only':round(float(np.mean(brier_cross)),5),
            'logloss_domestic':round(float(np.mean(ll_dom)),     5),
            'logloss_global':  round(float(np.mean(ll_cal)),     5),
            'logloss_cross_only':round(float(np.mean(ll_cross)), 5),
        }
        results.append(row)

    return results


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    print('=' * 60)
    print('BuildIntlCalibration — VCT Cross-Regional Calibration')
    print('=' * 60)

    print('\nLoading domestic ratings...')
    ratings_data = load_ratings()

    print('\nComputing per-event residuals and accumulating offsets...')
    calibration, event_cache = accumulate_offsets(ratings_data)

    # ── Print per-event summaries ─────────────────────────────────────────────
    print('\n── Per-event regional offsets (raw, before scaling/decay): ──')
    for eid in INTL_CONFIG:
        if eid not in event_cache:
            continue
        delta, bonuses, details, games = event_cache[eid]
        if not delta:
            continue
        sorted_delta = sorted(delta.items(), key=lambda x: -x[1])
        print(f'\n  {eid}: {len(games)} cross-regional map games')
        for reg, off in sorted_delta:
            print(f'    {reg:10s}: {off:+.2f}')
        top_ind = sorted(details.items(), key=lambda x: -x[1]['individual_bonus'])
        print(f'  Top individual bonuses:')
        for org, d in top_ind[:5]:
            print(f'    {org:6s} ({d["region"]}): +{d["individual_bonus"]:.2f}  '
                  f'(n={d["n_games"]}, resid={d["mean_resid"]:+.2f})')
        bot_ind = sorted(details.items(), key=lambda x: x[1]['individual_bonus'])
        print(f'  Bottom individual bonuses:')
        for org, d in bot_ind[:3]:
            print(f'    {org:6s} ({d["region"]}): {d["individual_bonus"]:+.2f}  '
                  f'(n={d["n_games"]}, resid={d["mean_resid"]:+.2f})')

    # ── Print accumulated snapshot calibrations ───────────────────────────────
    print('\n── Accumulated offsets per snapshot: ──')
    for key in sorted(calibration):
        c = calibration[key]
        if not c['regional_offsets'] and not c['individual_bonuses']:
            print(f'  {key}: (no calibration — no prior intl data)')
            continue
        regs  = c['regional_offsets']
        inds  = c['individual_bonuses']
        top_i = sorted(inds.items(), key=lambda x: -x[1])[:3]
        bot_i = sorted(inds.items(), key=lambda x:  x[1])[:2]
        reg_str = '  '.join(f'{r}:{v:+.2f}' for r, v in sorted(regs.items()))
        ind_str = '  '.join(f'{o}:{v:+.2f}' for o, v in top_i + bot_i)
        print(f'  {key:30s} | regs: {reg_str}')
        if inds:
            print(f'  {"":30s} | ind: {ind_str}')

    # ── Backtest ───────────────────────────────────────────────────────────────
    print('\n── Backtest (cross-regional map games at Champions): ──')
    bt = run_backtest(calibration, ratings_data, event_cache)
    baseline_brier = 0.25000
    for r in bt:
        print(f'\n  {r["event"]}  (n={r["n_games"]} cross-regional map games)')
        print(f'    Brier:   domestic={r["brier_domestic"]:.5f}  '
              f'global={r["brier_global"]:.5f}  '
              f'cross_only={r["brier_cross_only"]:.5f}  '
              f'(baseline {baseline_brier:.5f})')
        print(f'    LogLoss: domestic={r["logloss_domestic"]:.5f}  '
              f'global={r["logloss_global"]:.5f}  '
              f'cross_only={r["logloss_cross_only"]:.5f}')
        b_imp  = r["brier_domestic"] - r["brier_global"]
        x_imp  = r["brier_domestic"] - r["brier_cross_only"]
        winner = 'global' if b_imp >= x_imp else 'cross_only'
        print(f'    → {winner} wins by {max(b_imp,x_imp)*100:.2f}pp vs domestic-only')

    # Decide global-always vs cross-only based on aggregate improvement
    if bt:
        agg_global = float(np.mean([r["brier_domestic"] - r["brier_global"]     for r in bt]))
        agg_cross  = float(np.mean([r["brier_domestic"] - r["brier_cross_only"] for r in bt]))
        best_mode  = 'global' if agg_global >= agg_cross else 'cross_only'
        print(f'\n  Aggregate: global={agg_global*100:+.2f}pp  cross_only={agg_cross*100:+.2f}pp')
        print(f'  → Recommended mode: {best_mode}')
    else:
        best_mode = 'global'

    # ── Write output ───────────────────────────────────────────────────────────
    out = {
        'params': {
            'intl_hl_weeks':     INTL_HL_WEEKS,
            'regional_scale':    REGIONAL_SCALE,
            'individual_scale':  INDIVIDUAL_SCALE,
            'individual_shrink': INDIVIDUAL_SHRINK,
            'year_carry_base':   YEAR_CARRY_BASE,
            'recommended_mode':  best_mode,
        },
        'calibration': calibration,
        'backtest':    bt,
    }

    out_path = os.path.join(DATA, 'intl_calibration.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, separators=(',', ':'))
    print(f'\nWrote {out_path} ({os.path.getsize(out_path)//1024} KB)')


if __name__ == '__main__':
    main()
