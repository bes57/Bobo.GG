"""
ResearchModel.py
Diagnostic research script for the Massey map rating model.
Investigates two documented failure cases and tests potential fixes.

Run: python3 scrapers/ResearchModel.py 2>&1
"""

import os, sys, json, math, random
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from MoreTestingMaybeFiles import ALL_EVENTS

DATA_DIR = os.path.join(ROOT, 'data')

# ── Copy constants from BuildMapRatings ──────────────────────────────────────

INTL_EVENTS = {
    '2023_lock_in', '2023_masters_tokyo', '2023_champions',
    '2024_masters_madrid', '2024_masters_shanghai', '2024_champions',
    '2025_masters_bangkok', '2025_masters_toronto', '2025_champions',
    '2026_masters_santiago',
}

EVENT_DATES = {
    '2023_lock_in':        ('2023-01-10', '2023-02-12'),
    '2023_league':         ('2023-01-23', '2023-10-01'),
    '2023_masters_tokyo':  ('2023-06-11', '2023-06-25'),
    '2023_champions':      ('2023-08-06', '2023-08-27'),
    '2024_kickoff':           ('2024-01-08', '2024-02-11'),
    '2024_masters_madrid':    ('2024-02-14', '2024-03-10'),
    '2024_stage1':            ('2024-03-15', '2024-05-19'),
    '2024_masters_shanghai':  ('2024-06-02', '2024-06-16'),
    '2024_stage2':            ('2024-06-20', '2024-08-25'),
    '2024_champions':         ('2024-08-01', '2024-09-22'),
    '2025_kickoff':         ('2025-01-13', '2025-02-09'),
    '2025_masters_bangkok': ('2025-02-12', '2025-03-09'),
    '2025_stage1':          ('2025-03-14', '2025-05-18'),
    '2025_masters_toronto': ('2025-06-07', '2025-06-29'),
    '2025_stage2':          ('2025-07-14', '2025-08-24'),
    '2025_champions':       ('2025-08-28', '2025-09-21'),
    '2026_kickoff':          ('2026-01-07', '2026-02-09'),
    '2026_masters_santiago': ('2026-03-26', '2026-04-06'),
}

HALF_LIFE_WEEKS = 6
INTL_MULTIPLIER = 4.0

PACIFIC_TEAMS  = {'PRX', 'T1', 'TLN', 'GEN', 'DFM', 'ZETA', 'RRQ', 'TS', 'GE', 'KRX', 'NS', 'APK'}
AMERICAS_TEAMS = {'SEN', 'G2', 'MIBR', 'NRG', '100T', 'C9', 'EG', 'KRÜ', 'LEV', 'FUR', 'LOUD', 'BME'}
EMEA_TEAMS     = {'TL', 'FNC', 'NAVI', 'VIT', 'BBL', 'GX', 'KC', 'TH', 'FUT', 'GIA', 'MKOI', 'WOL', 'M8', 'FPX'}
CN_TEAMS       = {'EDG', 'BLG', 'TE', 'DRG', 'ASE', 'XLG', 'AG'}

REGION_MAP = {}
for t in PACIFIC_TEAMS:  REGION_MAP[t] = 'Pacific'
for t in AMERICAS_TEAMS: REGION_MAP[t] = 'Americas'
for t in EMEA_TEAMS:     REGION_MAP[t] = 'EMEA'
for t in CN_TEAMS:       REGION_MAP[t] = 'CN'


def _parse_date(s):
    return datetime.strptime(s, '%Y-%m-%d')


# ── Data loading (copied from BuildMapRatings) ────────────────────────────────

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
        df['MapNum']   = df['MapNum'].astype(str)
        df['MapName']  = df['MapName'].str.replace('PICK', '', regex=False).str.strip()
        frames.append(df)

    all_maps = pd.concat(frames, ignore_index=True)

    meta = all_maps.groupby(['MatchID', 'MapNum']).agg(
        orgs=('Org',      lambda x: list(x.unique())),
        map_name=('MapName',  'first'),
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
            'match_id':  int(row['MatchID']),
            'event_id':  row['event_id'],
            'map_name':  row['map_name'],
            'winner':    winner,
            'loser':     losers[0],
            'wr':        wr,
            'lr':        lr,
            'date':      None,
        })

    gdf = pd.DataFrame(games)
    for eid, (start_str, end_str) in EVENT_DATES.items():
        mask = gdf['event_id'] == eid
        if not mask.any():
            continue
        start = _parse_date(start_str)
        end   = _parse_date(end_str)
        span  = (end - start).days
        mids = gdf.loc[mask, 'match_id'].values
        sorted_unique = sorted(set(mids))
        rank_map = {mid: i for i, mid in enumerate(sorted_unique)}
        max_rank  = max(rank_map.values()) or 1
        indices = gdf.index[mask].tolist()
        for i, mid in zip(indices, mids):
            frac = rank_map[mid] / max_rank
            gdf.at[i, 'date'] = start + timedelta(days=int(frac * span))

    gdf = gdf.dropna(subset=['date'])
    return gdf.to_dict('records')


# ── Core Massey solver (standard symmetric) ──────────────────────────────────

def massey_ratings(games, lambda_decay, ref_date, min_games=0, intl_mult=None):
    if intl_mult is None:
        intl_mult = INTL_MULTIPLIER
    if not games:
        return {}

    teams = sorted({g['winner'] for g in games} | {g['loser'] for g in games})

    if min_games > 0:
        counts = {}
        for g in games:
            counts[g['winner']] = counts.get(g['winner'], 0) + 1
            counts[g['loser']]  = counts.get(g['loser'],  0) + 1
        teams = [t for t in teams if counts.get(t, 0) >= min_games]
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
        weeks_ago = max(0, (ref_date - g['date']).days / 7.0)
        w = math.exp(-lambda_decay * weeks_ago)
        if g.get('event_id') in INTL_EVENTS:
            w *= intl_mult
        rd = g['wr'] - g['lr']
        i, j = idx[g['winner']], idx[g['loser']]
        M[i, i] += w;  M[j, j] += w
        M[i, j] -= w;  M[j, i] -= w
        p[i] += w * rd;  p[j] -= w * rd

    M[-1, :] = 1.0
    p[-1]    = 0.0
    ridge = 1e-4
    for i in range(n - 1):
        M[i, i] += ridge
    M[-1, :] = 1.0
    p[-1]    = 0.0

    try:
        r = np.linalg.solve(M, p)
    except np.linalg.LinAlgError:
        r, *_ = np.linalg.lstsq(M, p, rcond=None)

    return {t: float(r[idx[t]]) for t in teams}


# ── Asymmetric Massey solver ──────────────────────────────────────────────────

def massey_ratings_asymmetric(games, lambda_decay, ref_date,
                               win_mult=4.0, loss_mult=1.0, intl_base_mult=1.0):
    """
    Asymmetric Massey: wins and losses can have different weights.
    win_mult  = multiplier applied to the base weight when counting a WIN
    loss_mult = multiplier applied to the base weight when counting a LOSS
    intl_base_mult = base multiplier for intl games (before win/loss mult)

    For each game:
      base_w = decay_weight * (intl_base_mult if intl else 1)
      w_i (winner) = base_w * win_mult
      w_j (loser)  = base_w * loss_mult
      M[i,i] += w_i, M[j,j] += w_j
      M[i,j] -= min(w_i, w_j), M[j,i] -= min(w_i, w_j)
      p[i] += w_i * rd, p[j] -= w_j * rd
    """
    if not games:
        return {}

    teams = sorted({g['winner'] for g in games} | {g['loser'] for g in games})
    n   = len(teams)
    idx = {t: i for i, t in enumerate(teams)}
    M   = np.zeros((n, n))
    p   = np.zeros(n)

    for g in games:
        if g['winner'] not in idx or g['loser'] not in idx:
            continue
        weeks_ago = max(0, (ref_date - g['date']).days / 7.0)
        base_w = math.exp(-lambda_decay * weeks_ago)
        if g.get('event_id') in INTL_EVENTS:
            base_w *= intl_base_mult
        rd = g['wr'] - g['lr']
        w_i = base_w * win_mult
        w_j = base_w * loss_mult
        i, j = idx[g['winner']], idx[g['loser']]
        M[i, i] += w_i
        M[j, j] += w_j
        M[i, j] -= min(w_i, w_j)
        M[j, i] -= min(w_i, w_j)
        p[i] += w_i * rd
        p[j] -= w_j * rd

    M[-1, :] = 1.0
    p[-1]    = 0.0
    ridge = 1e-4
    for i in range(n - 1):
        M[i, i] += ridge
    M[-1, :] = 1.0
    p[-1]    = 0.0

    try:
        r = np.linalg.solve(M, p)
    except np.linalg.LinAlgError:
        r, *_ = np.linalg.lstsq(M, p, rcond=None)

    return {t: float(r[idx[t]]) for t in teams}


# ── Per-team game contribution breakdown ─────────────────────────────────────

def breakdown_team_contributions(team, games, lambda_decay, ref_date, intl_mult=None):
    """
    Show how each event contributes to a team's Massey row (weighted round-diff column).
    Returns a list of (event_id, n_games, total_weight, weighted_rd, cum_rating_signal)
    """
    if intl_mult is None:
        intl_mult = INTL_MULTIPLIER

    by_event = defaultdict(lambda: {'n': 0, 'total_w': 0.0, 'weighted_rd': 0.0})
    for g in games:
        if g['winner'] != team and g['loser'] != team:
            continue
        weeks_ago = max(0, (ref_date - g['date']).days / 7.0)
        w = math.exp(-lambda_decay * weeks_ago)
        if g.get('event_id') in INTL_EVENTS:
            w *= intl_mult
        if g['winner'] == team:
            rd = g['wr'] - g['lr']
        else:
            rd = -(g['wr'] - g['lr'])
        eid = g['event_id']
        by_event[eid]['n'] += 1
        by_event[eid]['total_w'] += w
        by_event[eid]['weighted_rd'] += w * rd

    return dict(by_event)


def get_snapshot_games(all_games, events_list):
    return [g for g in all_games if g['event_id'] in events_list]


def rank_teams(ratings, teams_of_interest=None):
    """Sort ratings descending, return list of (rank, team, rating)."""
    sorted_teams = sorted(ratings.items(), key=lambda x: -x[1])
    result = []
    for rank, (team, rating) in enumerate(sorted_teams, 1):
        result.append((rank, team, rating))
    return result


def print_rankings(ratings, title, teams_highlight=None, top_n=None):
    ranked = rank_teams(ratings)
    if top_n:
        ranked = ranked[:top_n]
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")
    print(f"  {'#':>3}  {'Team':>6}  {'Rating':>8}  {'Region':>8}")
    for rank, team, rating in ranked:
        flag = " ◄" if teams_highlight and team in teams_highlight else ""
        region = REGION_MAP.get(team, '?')
        print(f"  {rank:>3}.  {team:>6}  {rating:+8.3f}  {region:>8}{flag}")


# ── MAIN RESEARCH ─────────────────────────────────────────────────────────────

def main():
    lam = math.log(2) / HALF_LIFE_WEEKS

    print("=" * 60)
    print("RESEARCH MODEL — Massey Rating Failure Case Diagnostics")
    print("=" * 60)

    print("\nLoading games...")
    all_games = load_games()
    print(f"  Total games loaded: {len(all_games)}")

    # ── 2025 after_champions snapshot ────────────────────────────────────────
    events_2025_ac = [
        '2025_kickoff', '2025_masters_bangkok', '2025_stage1',
        '2025_masters_toronto', '2025_stage2', '2025_champions'
    ]
    games_2025_ac = get_snapshot_games(all_games, events_2025_ac)
    ref_2025_ac   = max(g['date'] for g in games_2025_ac)

    # Which teams played at Champions
    champs_2025_teams = (
        {g['winner'] for g in all_games if g['event_id'] == '2025_champions'} |
        {g['loser']  for g in all_games if g['event_id'] == '2025_champions'}
    )
    print(f"\n  2025 Champions attendees: {sorted(champs_2025_teams)}")

    # Which Pacific teams did NOT qualify
    pacific_no_champs_2025 = PACIFIC_TEAMS & {g['winner'] for g in games_2025_ac} | \
                              PACIFIC_TEAMS & {g['loser']  for g in games_2025_ac}
    pacific_no_champs_2025 = {t for t in pacific_no_champs_2025 if t not in champs_2025_teams}
    print(f"  Pacific non-qualifiers 2025: {sorted(pacific_no_champs_2025)}")

    # ── 2026 after_santiago snapshot ─────────────────────────────────────────
    events_2026_as = ['2026_kickoff', '2026_masters_santiago']
    games_2026_as = get_snapshot_games(all_games, events_2026_as)
    ref_2026_as   = max(g['date'] for g in games_2026_as)

    santiago_teams = (
        {g['winner'] for g in all_games if g['event_id'] == '2026_masters_santiago'} |
        {g['loser']  for g in all_games if g['event_id'] == '2026_masters_santiago'}
    )
    print(f"\n  2026 Santiago attendees: {sorted(santiago_teams)}")

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 1: DIAGNOSE CASE 1 — 2025 after_champions Pacific breakdown
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SECTION 1: CASE 1 DIAGNOSIS — 2025 after_champions")
    print("= Why is Gen.G above PRX/KRX? =")
    print("=" * 60)

    ratings_2025_ac = massey_ratings(games_2025_ac, lam, ref_2025_ac)
    print_rankings(ratings_2025_ac, "2025 after_champions — ALL TEAMS (top 20)",
                   teams_highlight={'GEN', 'PRX', 'KRX', 'RRQ', 'NRG', 'MIBR', 'FNC'},
                   top_n=20)

    print("\n\n--- Per-Event Contribution Breakdown for Pacific teams ---")
    print(f"(ref_date = {ref_2025_ac.strftime('%Y-%m-%d')}, lambda={lam:.5f}, intl_mult={INTL_MULTIPLIER})")

    focus_teams = ['GEN', 'PRX', 'KRX', 'RRQ', 'T1', 'NS']
    for team in focus_teams:
        if team not in ratings_2025_ac:
            print(f"\n  {team}: NOT IN RATINGS")
            continue
        contrib = breakdown_team_contributions(team, games_2025_ac, lam, ref_2025_ac)
        print(f"\n  {team}  (overall rating = {ratings_2025_ac[team]:+.3f})")
        total_w = sum(v['total_w'] for v in contrib.values())
        total_wrd = sum(v['weighted_rd'] for v in contrib.values())
        print(f"  {'Event':<28} {'N':>4} {'Weight':>8} {'Wt%':>6} {'Wtd RD':>8} {'p[i] share':>10}")
        for eid in sorted(contrib, key=lambda e: contrib[e]['total_w'], reverse=True):
            v = contrib[eid]
            wt_pct = 100.0 * v['total_w'] / total_w if total_w > 0 else 0
            intl_marker = " (INTL)" if eid in INTL_EVENTS else ""
            print(f"  {eid+intl_marker:<28} {v['n']:>4} {v['total_w']:>8.2f} {wt_pct:>5.1f}% "
                  f"{v['weighted_rd']:>8.2f}  "
                  f"p[i] portion: {v['weighted_rd']:+.2f}")
        print(f"  {'TOTAL':<28} {sum(v['n'] for v in contrib.values()):>4} "
              f"{total_w:>8.2f} {'100%':>6} {total_wrd:>8.2f}")

    # Show Gen.G vs PRX: which events pull them apart
    print("\n\n--- Gen.G vs PRX: Delta analysis ---")
    gen_contrib = breakdown_team_contributions('GEN', games_2025_ac, lam, ref_2025_ac)
    prx_contrib = breakdown_team_contributions('PRX', games_2025_ac, lam, ref_2025_ac)
    all_events_seen = sorted(set(gen_contrib) | set(prx_contrib),
                             key=lambda e: (e not in INTL_EVENTS, e))
    print(f"  {'Event':<28} {'GEN wtd_rd':>12} {'PRX wtd_rd':>12} {'GEN-PRX Δ':>12}")
    for eid in all_events_seen:
        g_rd = gen_contrib.get(eid, {}).get('weighted_rd', 0.0)
        p_rd = prx_contrib.get(eid, {}).get('weighted_rd', 0.0)
        delta = g_rd - p_rd
        flag = " ◄" if abs(delta) > 1 else ""
        print(f"  {eid:<28} {g_rd:>+12.2f} {p_rd:>+12.2f} {delta:>+12.2f}{flag}")

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 2: DIAGNOSE CASE 2 — 2026 after_santiago
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SECTION 2: CASE 2 DIAGNOSIS — 2026 after_santiago")
    print("= Why is T1 ranked so low vs GE/RRQ who didn't qualify? =")
    print("=" * 60)

    ratings_2026_as = massey_ratings(games_2026_as, lam, ref_2026_as)
    print_rankings(ratings_2026_as, "2026 after_santiago — ALL TEAMS",
                   teams_highlight={'T1', 'GE', 'RRQ', 'GEN', 'PRX', 'NS'},
                   top_n=25)

    print("\n--- Per-Event Breakdown for T1, GE, RRQ ---")
    for team in ['T1', 'GE', 'RRQ']:
        if team not in ratings_2026_as:
            print(f"\n  {team}: NOT IN RATINGS")
            continue
        contrib = breakdown_team_contributions(team, games_2026_as, lam, ref_2026_as)
        print(f"\n  {team}  (overall rating = {ratings_2026_as.get(team, 'N/A')})")
        total_w = sum(v['total_w'] for v in contrib.values())
        print(f"  {'Event':<28} {'N':>4} {'Weight':>8} {'Wt%':>6} {'Wtd RD':>8}")
        for eid in sorted(contrib, key=lambda e: contrib[e]['total_w'], reverse=True):
            v = contrib[eid]
            wt_pct = 100.0 * v['total_w'] / total_w if total_w > 0 else 0
            intl_marker = " (INTL)" if eid in INTL_EVENTS else ""
            print(f"  {eid+intl_marker:<28} {v['n']:>4} {v['total_w']:>8.2f} {wt_pct:>5.1f}% "
                  f"{v['weighted_rd']:>+8.2f}")

    print("\n--- T1 vs GE vs RRQ: Santiago game details ---")
    for team in ['T1', 'GE', 'RRQ']:
        santiago_games = [g for g in games_2026_as
                          if g['event_id'] == '2026_masters_santiago'
                          and (g['winner'] == team or g['loser'] == team)]
        kickoff_games  = [g for g in games_2026_as
                          if g['event_id'] == '2026_kickoff'
                          and (g['winner'] == team or g['loser'] == team)]
        print(f"\n  {team}:")
        print(f"    Kickoff games: {len(kickoff_games)}")
        for g in kickoff_games:
            outcome = 'W' if g['winner'] == team else 'L'
            opp = g['loser'] if g['winner'] == team else g['winner']
            weeks_ago = max(0, (ref_2026_as - g['date']).days / 7.0)
            w = math.exp(-lam * weeks_ago)
            print(f"      [{outcome}] vs {opp:>6}  {g['wr']}-{g['lr']}  "
                  f"decay_w={w:.3f}  weeks_ago={weeks_ago:.1f}")
        print(f"    Santiago games: {len(santiago_games)}")
        for g in santiago_games:
            outcome = 'W' if g['winner'] == team else 'L'
            opp = g['loser'] if g['winner'] == team else g['winner']
            weeks_ago = max(0, (ref_2026_as - g['date']).days / 7.0)
            w = math.exp(-lam * weeks_ago) * INTL_MULTIPLIER
            rd = g['wr'] - g['lr'] if g['winner'] == team else -(g['wr'] - g['lr'])
            print(f"      [{outcome}] vs {opp:>6}  {g['wr']}-{g['lr']}  "
                  f"eff_w={w:.3f}  rd={rd:+d}  contribution={w*rd:+.2f}")

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 3: INTL_MULTIPLIER SWEEP — 2025 after_champions
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SECTION 3: INTL_MULTIPLIER SWEEP — 2025 after_champions")
    print("= At what multiplier does PRX overtake Gen.G? =")
    print("=" * 60)

    sweep_teams = ['GEN', 'PRX', 'KRX', 'RRQ', 'NRG', 'MIBR', 'FNC', 'T1']
    mults = [1, 2, 4, 6, 8, 12, 16, 24]

    header = f"  {'Mult':>4}  " + "  ".join(f"{t:>6}" for t in sweep_teams)
    print(header)
    print("  " + "-" * (len(header) - 2))

    prev_gen_above_prx = None
    for mult in mults:
        rtgs = massey_ratings(games_2025_ac, lam, ref_2025_ac, intl_mult=mult)
        ranked = rank_teams(rtgs)
        rank_by_team = {t: r for r, t, _ in ranked}

        gen_rank = rank_by_team.get('GEN', 99)
        prx_rank = rank_by_team.get('PRX', 99)
        krx_rank = rank_by_team.get('KRX', 99)
        gen_above = gen_rank < prx_rank

        ranks_str = "  ".join(f"{rank_by_team.get(t, '—'):>6}" for t in sweep_teams)
        flip_note = ""
        if prev_gen_above_prx is not None and prev_gen_above_prx and not gen_above:
            flip_note = f"  ← PRX overtakes GEN here!"
        print(f"  {mult:>4}  {ranks_str}{flip_note}")
        prev_gen_above_prx = gen_above

    print("\n  (Rating values at each mult:)")
    header2 = f"  {'Mult':>4}  " + "  ".join(f"{t:>7}" for t in sweep_teams)
    print(header2)
    for mult in mults:
        rtgs = massey_ratings(games_2025_ac, lam, ref_2025_ac, intl_mult=mult)
        vals_str = "  ".join(f"{rtgs.get(t, 0):>+7.2f}" for t in sweep_teams)
        print(f"  {mult:>4}  {vals_str}")

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 4: ASYMMETRIC MULTIPLIER GRID — 2025 after_champions
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SECTION 4: ASYMMETRIC MULTIPLIER GRID — 2025 after_champions")
    print("= win_mult x loss_mult combinations for GEN/PRX/KRX ranks =")
    print("=" * 60)

    win_mults  = [4, 6, 8, 12]
    loss_mults = [0, 0.5, 1, 2]

    print(f"\n  Rankings table: (GEN rank / PRX rank / KRX rank)")
    print(f"  {'':>10}", end="")
    for wm in win_mults:
        print(f"  win={wm:<4}", end="")
    print()
    print("  " + "-" * (10 + len(win_mults) * 10))

    for lm in loss_mults:
        print(f"  loss={lm:<5}", end="")
        for wm in win_mults:
            rtgs = massey_ratings_asymmetric(
                games_2025_ac, lam, ref_2025_ac,
                win_mult=wm, loss_mult=lm, intl_base_mult=1.0
            )
            ranked = rank_teams(rtgs)
            rank_by_team = {t: r for r, t, _ in ranked}
            gen_r = rank_by_team.get('GEN', 99)
            prx_r = rank_by_team.get('PRX', 99)
            krx_r = rank_by_team.get('KRX', 99)
            # Mark if PRX > GEN (correct ordering)
            mark = "✓" if prx_r < gen_r else " "
            print(f"  {mark}{gen_r}/{prx_r}/{krx_r:<3}", end="")
        print()

    print(f"\n  Ratings table: GEN | PRX | KRX  (✓ = PRX rank > GEN rank, i.e., PRX is ahead)")
    for lm in loss_mults:
        for wm in win_mults:
            rtgs = massey_ratings_asymmetric(
                games_2025_ac, lam, ref_2025_ac,
                win_mult=wm, loss_mult=lm, intl_base_mult=1.0
            )
            gen_r = rtgs.get('GEN', 0)
            prx_r = rtgs.get('PRX', 0)
            krx_r = rtgs.get('KRX', 0)
            mark = "✓" if prx_r > gen_r else " "
            print(f"  {mark} win={wm}, loss={lm}:  GEN={gen_r:+.3f}  PRX={prx_r:+.3f}  KRX={krx_r:+.3f}")

    # Also test with intl_base_mult = 4 (symmetric intl boost, then asymmetric win/loss)
    print(f"\n  Same grid but with intl_base_mult=4 (INTL games get 4x base, then asymmetric win/loss):")
    print(f"  {'':>10}", end="")
    for wm in win_mults:
        print(f"  win={wm:<4}", end="")
    print()
    for lm in loss_mults:
        print(f"  loss={lm:<5}", end="")
        for wm in win_mults:
            rtgs = massey_ratings_asymmetric(
                games_2025_ac, lam, ref_2025_ac,
                win_mult=wm, loss_mult=lm, intl_base_mult=4.0
            )
            ranked = rank_teams(rtgs)
            rank_by_team = {t: r for r, t, _ in ranked}
            gen_r = rank_by_team.get('GEN', 99)
            prx_r = rank_by_team.get('PRX', 99)
            krx_r = rank_by_team.get('KRX', 99)
            mark = "✓" if prx_r < gen_r else " "
            print(f"  {mark}{gen_r}/{prx_r}/{krx_r:<3}", end="")
        print()

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 5: QUALIFICATION CONSTRAINT POST-PROCESSING
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SECTION 5: QUALIFICATION CONSTRAINT POST-PROCESSING")
    print("= Cap non-qualifiers below any qualifier from same region =")
    print("=" * 60)

    ratings_base = massey_ratings(games_2025_ac, lam, ref_2025_ac)

    # Build region -> {qualifiers, non-qualifiers}
    region_qualifiers = defaultdict(set)
    region_non_qual   = defaultdict(set)
    for team, rating in ratings_base.items():
        region = REGION_MAP.get(team)
        if region is None or region == 'CN':
            continue
        if team in champs_2025_teams:
            region_qualifiers[region].add(team)
        else:
            region_non_qual[region].add(team)

    print("\n  Per region — qualifiers vs non-qualifiers in this snapshot:")
    for region in ['Pacific', 'Americas', 'EMEA']:
        qs = sorted(region_qualifiers[region], key=lambda t: -ratings_base.get(t, 0))
        nqs = sorted(region_non_qual[region], key=lambda t: -ratings_base.get(t, 0))
        print(f"  {region}:")
        print(f"    Qualifiers:     {qs}")
        print(f"    Non-qualifiers: {nqs}")

    # Apply cap: for each non-qualifier, cap their rating below the
    # LOWEST-rated qualifier in their region
    epsilon = 0.01
    ratings_capped = dict(ratings_base)
    for region in ['Pacific', 'Americas', 'EMEA']:
        qs = region_qualifiers[region]
        nqs = region_non_qual[region]
        if not qs:
            continue
        min_qual_rating = min(ratings_base[q] for q in qs if q in ratings_base)
        for team in nqs:
            if team in ratings_capped:
                if ratings_capped[team] >= min_qual_rating:
                    old = ratings_capped[team]
                    ratings_capped[team] = min_qual_rating - epsilon
                    print(f"  [CAP] {team:>6}: {old:+.3f} → {ratings_capped[team]:+.3f} "
                          f"(below {region} min qualifier {min_qual_rating:+.3f})")

    # Re-normalize ratings so mean = 0
    vals = list(ratings_capped.values())
    mean_v = np.mean(vals)
    ratings_capped = {t: v - mean_v for t, v in ratings_capped.items()}

    print_rankings(ratings_capped,
                   "2025 after_champions WITH QUALIFICATION CAP (top 20)",
                   teams_highlight={'GEN', 'PRX', 'KRX', 'NRG', 'MIBR', 'FNC'},
                   top_n=20)

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 6: SYNTHETIC QUALIFICATION LOSS GAMES
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SECTION 6: SYNTHETIC QUALIFICATION LOSS GAMES")
    print("= Add fake 13-9 loss for each non-qualifier =")
    print("=" * 60)

    # Determine league-average team proxy.
    # We'll use a synthetic team "LEAGUE_AVG" with rating 0 by definition.
    # Synthetic game: non-qualifier loses 9-13 to LEAGUE_AVG.
    # Weight equivalent to a Stage 2 game at the time of Stage 2 (roughly 0.6 decay).
    # We place the synthetic game date at the END of stage2 (2025-08-24)
    # so it's as recent as possible to have meaningful weight.

    stage2_end = _parse_date('2025-08-24')
    weeks_ago_synthetic = max(0, (ref_2025_ac - stage2_end).days / 7.0)
    synthetic_base_w = math.exp(-lam * weeks_ago_synthetic)
    print(f"\n  Synthetic game date: {stage2_end.strftime('%Y-%m-%d')}")
    print(f"  weeks_ago from ref: {weeks_ago_synthetic:.2f}")
    print(f"  decay weight:       {synthetic_base_w:.4f}  (roughly)")
    print(f"  (Stage 2 game weight ≈ 0.6, this is: {synthetic_base_w:.4f})")

    # Build list of non-qualifying teams that appear in games_2025_ac
    teams_in_snapshot = {g['winner'] for g in games_2025_ac} | {g['loser'] for g in games_2025_ac}
    non_qual_in_snap  = teams_in_snapshot - champs_2025_teams

    # Filter to non-CN teams only (mirroring the model behavior)
    non_qual_in_snap = {t for t in non_qual_in_snap if REGION_MAP.get(t) != 'CN'}

    print(f"\n  Non-qualifiers who get synthetic losses: {sorted(non_qual_in_snap)}")

    # Build augmented games list
    SYNTHETIC_TEAM = 'LEAGUE_AVG'
    synthetic_games = []
    for team in non_qual_in_snap:
        synthetic_games.append({
            'match_id':  -1,  # synthetic
            'event_id':  '2025_stage2',  # treated as domestic (no intl multiplier)
            'map_name':  'Ascent',
            'winner':    SYNTHETIC_TEAM,
            'loser':     team,
            'wr':        13,
            'lr':        9,
            'date':      stage2_end,
        })

    games_2025_ac_synth = games_2025_ac + synthetic_games
    ratings_synth = massey_ratings(games_2025_ac_synth, lam, ref_2025_ac)
    print_rankings(ratings_synth,
                   "2025 after_champions WITH SYNTHETIC LOSSES (top 25)",
                   teams_highlight={'GEN', 'PRX', 'KRX', 'NRG', 'MIBR', 'FNC', SYNTHETIC_TEAM},
                   top_n=25)

    print(f"\n  Focus teams comparison (Base vs Synthetic):")
    focus = ['NRG', 'MIBR', 'FNC', 'GEN', 'PRX', 'KRX', 'RRQ', 'T1', 'NS']
    rated_base  = rank_teams(ratings_base)
    rated_synth = rank_teams(ratings_synth)
    rank_base   = {t: (r, v) for r, t, v in rated_base}
    rank_synth  = {t: (r, v) for r, t, v in rated_synth if t != SYNTHETIC_TEAM}
    print(f"  {'Team':>6}  {'Base Rank':>10} {'Base Rtg':>10}  {'Synth Rank':>10} {'Synth Rtg':>10}  {'Δ Rank':>8}")
    for team in focus:
        br, bv = rank_base.get(team, (99, 0))
        sr, sv = rank_synth.get(team, (99, 0))
        print(f"  {team:>6}  {br:>10} {bv:>+10.3f}  {sr:>10} {sv:>+10.3f}  {sr-br:>+8}")

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 7: TEST CASE 2 FIXES — 2026 after_santiago
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SECTION 7: APPLY FIXES TO CASE 2 — 2026 after_santiago")
    print("= Do the same fixes help T1 vs GE/RRQ? =")
    print("=" * 60)

    santiago_qualifiers = santiago_teams
    non_qual_2026 = {g['winner'] for g in games_2026_as} | {g['loser'] for g in games_2026_as}
    non_qual_2026 = non_qual_2026 - santiago_qualifiers
    non_qual_2026 = {t for t in non_qual_2026 if REGION_MAP.get(t) != 'CN'}

    print(f"\n  Santiago qualifiers: {sorted(santiago_qualifiers)}")
    print(f"  Non-qualifiers in snapshot: {sorted(non_qual_2026)}")

    # 7a: Qualification cap for Case 2
    region_qual_2026   = defaultdict(set)
    region_nonqual_2026 = defaultdict(set)
    for team in (
        {g['winner'] for g in games_2026_as} | {g['loser'] for g in games_2026_as}
    ):
        region = REGION_MAP.get(team)
        if region is None or region == 'CN':
            continue
        if team in santiago_qualifiers:
            region_qual_2026[region].add(team)
        else:
            region_nonqual_2026[region].add(team)

    ratings_2026_capped = dict(ratings_2026_as)
    for region in ['Pacific', 'Americas', 'EMEA']:
        qs = region_qual_2026[region]
        nqs = region_nonqual_2026[region]
        if not qs:
            continue
        min_qual = min(ratings_2026_as[q] for q in qs if q in ratings_2026_as)
        for team in nqs:
            if team in ratings_2026_capped and ratings_2026_capped[team] >= min_qual:
                old = ratings_2026_capped[team]
                ratings_2026_capped[team] = min_qual - epsilon
                print(f"  [CAP 2026] {team:>6}: {old:+.3f} → {ratings_2026_capped[team]:+.3f}")

    vals_2026 = list(ratings_2026_capped.values())
    mean_2026 = np.mean(vals_2026)
    ratings_2026_capped = {t: v - mean_2026 for t, v in ratings_2026_capped.items()}

    print_rankings(ratings_2026_capped,
                   "2026 after_santiago WITH QUALIFICATION CAP",
                   teams_highlight={'T1', 'GE', 'RRQ', 'NRG', 'G2', 'PRX'},
                   top_n=20)

    # 7b: Asymmetric multiplier test for Case 2
    print("\n--- Asymmetric multiplier on 2026 after_santiago (win=4, loss=0.5, intl_base=4) ---")
    rtgs_asym_2026 = massey_ratings_asymmetric(
        games_2026_as, lam, ref_2026_as, win_mult=4, loss_mult=0.5, intl_base_mult=4.0
    )
    print_rankings(rtgs_asym_2026,
                   "2026 after_santiago — Asymmetric (win=4, loss=0.5, intl_base=4)",
                   teams_highlight={'T1', 'GE', 'RRQ'},
                   top_n=20)

    # 7c: Synthetic losses for Case 2 non-qualifiers
    kickoff_end_2026 = _parse_date('2026-02-09')
    weeks_ago_synth_2026 = max(0, (ref_2026_as - kickoff_end_2026).days / 7.0)
    w_synth_2026 = math.exp(-lam * weeks_ago_synth_2026)
    print(f"\n  Synthetic loss weight for 2026 non-qualifiers: {w_synth_2026:.4f}")

    synth_2026 = []
    for team in non_qual_2026:
        synth_2026.append({
            'match_id': -2,
            'event_id': '2026_kickoff',
            'map_name': 'Ascent',
            'winner':   SYNTHETIC_TEAM,
            'loser':    team,
            'wr':       13,
            'lr':       9,
            'date':     kickoff_end_2026,
        })

    games_2026_synth = games_2026_as + synth_2026
    ratings_2026_synth = massey_ratings(games_2026_synth, lam, ref_2026_as)
    print_rankings(ratings_2026_synth,
                   "2026 after_santiago WITH SYNTHETIC LOSSES",
                   teams_highlight={'T1', 'GE', 'RRQ', SYNTHETIC_TEAM},
                   top_n=20)

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 8: COMBINED BEST FIX TEST — Both cases
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SECTION 8: COMBINED BEST FIX — Both Cases Simultaneously")
    print("= Qualification cap applied to both 2025 and 2026 =")
    print("=" * 60)

    print("\n  The qualification cap approach:")
    print("  - For 2025 after_champions: caps non-qualifiers below lowest qualifier per region")
    print("  - For 2026 after_santiago: same")
    print("  - Model distortion: ZERO — only post-processing, Massey solve unchanged")
    print("  - Correctness: directly encodes the information 'qualification is an outcome'")
    print("")
    print("  The asymmetric multiplier (win=4, loss=0.5, intl_base=4):")
    print("  - Changes the Massey solve itself")
    print("  - Makes INTL wins count 16x more, INTL losses count 2x more")
    print("  - Partially penalizes teams that attend and lose, less than symmetric 4x")
    print("  - But still penalizes T1 for attending — just less")
    print("")
    print("  The synthetic loss approach:")
    print("  - Adds fake games — model distortion is moderate")
    print("  - Weight ~0.17 (far-past kickoff weight) makes it very weak signal")
    print("  - Non-qualifiers are assumed to lose league-average, which may be wrong")
    print("  - This infers Champions performance rather than encoding qualification")

    # Summary table: base vs each fix for the focus teams in Case 1
    print("\n--- Summary: Case 1 (2025) Rankings under each approach ---")
    focus_c1 = ['NRG', 'MIBR', 'FNC', 'GEN', 'PRX', 'KRX', 'RRQ', 'T1', 'NS']
    rank_base_c1 = {t: r for r, t, _ in rank_teams(ratings_base)}
    rank_cap_c1  = {t: r for r, t, _ in rank_teams(ratings_capped)}
    rank_syn_c1  = {t: r for r, t, _ in rank_teams(ratings_synth) if t != SYNTHETIC_TEAM}
    # Best asymmetric: win=8, loss=0.5, intl_base=4
    rtgs_asym_best = massey_ratings_asymmetric(
        games_2025_ac, lam, ref_2025_ac, win_mult=8, loss_mult=0.5, intl_base_mult=4.0
    )
    rank_asym_c1 = {t: r for r, t, _ in rank_teams(rtgs_asym_best)}

    print(f"\n  {'Team':>6}  {'Base':>5}  {'Cap':>5}  {'Synth':>5}  {'Asym(8,0.5,4)':>14}  {'In Champs?':>10}")
    for team in focus_c1:
        in_champs = 'YES' if team in champs_2025_teams else 'no'
        print(f"  {team:>6}  {rank_base_c1.get(team,99):>5}  {rank_cap_c1.get(team,99):>5}  "
              f"{rank_syn_c1.get(team,99):>5}  {rank_asym_c1.get(team,99):>14}  {in_champs:>10}")

    print("\n--- Summary: Case 2 (2026) Rankings under each approach ---")
    focus_c2 = ['NRG', 'G2', 'PRX', 'T1', 'GE', 'RRQ', 'NS', 'M8', 'TL']
    rank_base_c2 = {t: r for r, t, _ in rank_teams(ratings_2026_as)}
    rank_cap_c2  = {t: r for r, t, _ in rank_teams(ratings_2026_capped)}
    rank_syn_c2  = {t: r for r, t, _ in rank_teams(ratings_2026_synth) if t != SYNTHETIC_TEAM}
    rank_asym_c2 = {t: r for r, t, _ in rank_teams(rtgs_asym_2026)}

    print(f"\n  {'Team':>6}  {'Base':>5}  {'Cap':>5}  {'Synth':>5}  {'Asym(4,0.5,4)':>14}  {'In Santiago?':>12}")
    for team in focus_c2:
        in_santiago = 'YES' if team in santiago_teams else 'no'
        print(f"  {team:>6}  {rank_base_c2.get(team,99):>5}  {rank_cap_c2.get(team,99):>5}  "
              f"{rank_syn_c2.get(team,99):>5}  {rank_asym_c2.get(team,99):>14}  {in_santiago:>12}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == '__main__':
    main()
