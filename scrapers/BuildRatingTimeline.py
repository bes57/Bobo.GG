"""
BuildRatingTimeline.py
Build a per-match-day BenPom rating timeline for the Modern VCT Hub chart.

For each day in 2026 where at least one match was played, computes the Massey
rating for all active teams using all games up to and including that day.
Also computes per-match rating deltas for the interactive chart dots.

Requires: data/match_dates.json (from ScrapeMatchDates.py)
Output:   data/rating_timeline.json

Usage: python scrapers/BuildRatingTimeline.py
"""

import os, sys, json, math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from MoreTestingMaybeFiles import ALL_EVENTS

DATA_DIR = os.path.join(ROOT, "data")
OUT_PATH  = os.path.join(DATA_DIR, "rating_timeline.json")

# ── Model constants (match BuildMapRatings.py) ─────────────────────────────────
LAMBDA_DECAY  = math.log(2) / 5.0   # 5-week half-life — short recency window so recent results dominate
CN_TEAMS      = {"EDG","BLG","TE","DRG","ASE","AG","XLG"}  # actual CN teams — NS/KRX are Pacific
INTL_EVENTS   = {
    "2023_lock_in", "2023_masters_tokyo", "2023_champions",
    "2024_masters_madrid", "2024_masters_shanghai", "2024_champions",
    "2025_masters_bangkok", "2025_masters_toronto", "2025_champions",
    "2026_masters_santiago",
}
INTL_MULT = 1.0       # intl games same weight as domestic (best symmetric calibration)
MIN_GAMES     = 5     # min games for a team to appear in the timeline

EVENT_DATES = {
    "2023_lock_in":          ("2023-01-10", "2023-02-12"),
    "2023_league":           ("2023-01-23", "2023-10-01"),
    "2023_masters_tokyo":    ("2023-06-11", "2023-06-25"),
    "2023_champions":        ("2023-08-06", "2023-08-27"),
    "2024_kickoff":          ("2024-01-08", "2024-02-11"),
    "2024_masters_madrid":   ("2024-02-14", "2024-03-10"),
    "2024_stage1":           ("2024-03-15", "2024-05-19"),
    "2024_masters_shanghai": ("2024-06-02", "2024-06-16"),
    "2024_stage2":           ("2024-06-20", "2024-08-25"),
    "2024_champions":        ("2024-08-01", "2024-09-22"),
    "2025_kickoff":          ("2025-01-13", "2025-02-09"),
    "2025_masters_bangkok":  ("2025-02-12", "2025-03-09"),
    "2025_stage1":           ("2025-03-14", "2025-05-18"),
    "2025_masters_toronto":  ("2025-06-07", "2025-06-29"),
    "2025_stage2":           ("2025-07-14", "2025-08-24"),
    "2025_champions":        ("2025-08-28", "2025-09-21"),
    "2026_kickoff":          ("2026-01-15", "2026-02-16"),
    "2026_masters_santiago": ("2026-02-28", "2026-03-15"),
    "2026_stage1":           ("2026-04-01", "2026-05-25"),
}


# ── Massey solver ──────────────────────────────────────────────────────────────

def massey_ratings(games, lambda_decay, ref_date, min_games=0):
    """
    Opponent-adjusted Massey rating with exponential recency decay.
    ref_date: datetime object (used as "now" for decay computation).
    games: list of dicts with keys: winner, loser, wr, lr, date (datetime), event_id.
    Returns {team: float} (mean-zero).
    """
    if not games:
        return {}

    teams = sorted({g["winner"] for g in games} | {g["loser"] for g in games})

    if min_games > 0:
        counts = defaultdict(int)
        for g in games:
            counts[g["winner"]] += 1
            counts[g["loser"]]  += 1
        teams = [t for t in teams if counts[t] >= min_games]
        games = [g for g in games if g["winner"] in teams and g["loser"] in teams]
        if not games:
            return {}

    n   = len(teams)
    idx = {t: i for i, t in enumerate(teams)}
    M   = np.zeros((n, n))
    p   = np.zeros(n)

    for g in games:
        if g["winner"] not in idx or g["loser"] not in idx:
            continue
        g_date = g["date"] if isinstance(g["date"], datetime) else datetime.strptime(g["date"], "%Y-%m-%d")
        weeks_ago = max(0.0, (ref_date - g_date).days / 7.0)
        w = math.exp(-lambda_decay * weeks_ago)
        if g.get("event_id") in INTL_EVENTS:
            w *= INTL_MULT
        rd = g["wr"] - g["lr"]
        i, j = idx[g["winner"]], idx[g["loser"]]
        M[i, i] += w;  M[j, j] += w
        M[i, j] -= w;  M[j, i] -= w
        p[i] += w * rd;  p[j] -= w * rd

    # Mean-zero anchor on last row
    M[-1, :] = 1.0
    p[-1]    = 0.0
    # Ridge: prevent near-singular matrix
    for i in range(n - 1):
        M[i, i] += 1e-4

    try:
        r = np.linalg.solve(M, p)
    except np.linalg.LinAlgError:
        r, *_ = np.linalg.lstsq(M, p, rcond=None)

    return {t: float(r[idx[t]]) for t in teams}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_all_games():
    """
    Load every scraped map-level game and attach an actual date.
    Falls back to match_id-rank interpolation within an event if the date
    is not in match_dates.json.
    Returns list of game dicts.
    """
    dates_path = os.path.join(DATA_DIR, "match_dates.json")
    match_dates = {}
    if os.path.exists(dates_path):
        with open(dates_path) as f:
            match_dates = json.load(f)
        print(f"  Loaded {len(match_dates)} match dates from match_dates.json")
    else:
        print("  WARNING: match_dates.json not found — dates will be interpolated")

    mr = pd.read_csv(os.path.join(DATA_DIR, "match_results.csv"))
    mr = mr[mr["MapNum"] != "all"].copy()
    mr["MapNum"] = mr["MapNum"].astype(str)
    mr_idx = mr.set_index(["MatchID", "MapNum"])

    games = []

    for event in ALL_EVENTS:
        eid  = event["id"]
        path = os.path.join(DATA_DIR, "maps", f"{eid}.csv")
        if not os.path.exists(path):
            continue
        if eid not in EVENT_DATES:
            continue

        df = pd.read_csv(path)
        df["event_id"] = eid
        df["MapNum"]   = df["MapNum"].astype(str)
        df["MapName"]  = df["MapName"].str.replace("PICK", "", regex=False).str.strip()

        meta = df.groupby(["MatchID", "MapNum"]).agg(
            orgs=("Org",     lambda x: list(x.unique())),
            map_name=("MapName",  "first"),
            event_id=("event_id", "first"),
        ).reset_index()

        for _, row in meta.iterrows():
            key = (int(row["MatchID"]), row["MapNum"])
            if key not in mr_idx.index:
                continue
            mr_row = mr_idx.loc[key]
            winner = mr_row["WinnerOrg"]
            losers = [o for o in row["orgs"] if o != winner]
            if not losers:
                continue
            try:
                wr, lr = map(int, str(mr_row["Score"]).split("-"))
            except Exception:
                continue

            mid_str = str(int(row["MatchID"]))
            date_str = match_dates.get(mid_str)  # may be None

            games.append({
                "match_id":   int(row["MatchID"]),
                "event_id":   eid,
                "map_name":   row["map_name"],
                "winner":     winner,
                "loser":      losers[0],
                "wr":         wr,
                "lr":         lr,
                "date":       datetime.strptime(date_str, "%Y-%m-%d") if date_str else None,
                "_date_known": date_str is not None,
            })

    # Fill interpolated dates for games without real dates
    gdf = pd.DataFrame(games)
    for eid, (start_str, end_str) in EVENT_DATES.items():
        mask = (gdf["event_id"] == eid) & gdf["date"].isna()
        if not mask.any():
            continue
        start_dt = datetime.strptime(start_str, "%Y-%m-%d")
        end_dt   = datetime.strptime(end_str,   "%Y-%m-%d")
        span     = max(1, (end_dt - start_dt).days)

        mids         = gdf.loc[mask, "match_id"].values
        sorted_uniq  = sorted(set(mids))
        rank_map     = {mid: i for i, mid in enumerate(sorted_uniq)}
        max_rank     = max(rank_map.values()) if rank_map else 1

        for i_row, mid in zip(gdf.index[mask], mids):
            frac = rank_map[mid] / max_rank
            gdf.at[i_row, "date"] = start_dt + timedelta(days=int(frac * span))

    gdf = gdf.dropna(subset=["date"])
    return gdf.to_dict("records")


# ── Timeline builder ───────────────────────────────────────────────────────────

def build_2026_timeline(all_games):
    """
    Build checkpoint ratings at every match day in 2026, plus per-match deltas.

    Pure fresh-slate: only 2026 games are used. Each team starts at 0 on
    Jan 1 and ratings evolve entirely from 2026 results. 5-week half-life
    ensures recent results dominate (Kickoff from 17 weeks ago has ~10% weight
    by May; Stage 1 from 2 weeks ago has ~76% weight).

    Returns:
      checkpoints:   [{date, ratings:{org:float}}]
      match_events:  [{match_id, date, winner, loser, maps, series_score,
                       winner_before, winner_after, winner_delta,
                       loser_before,  loser_after,  loser_delta}]
    """
    # Include CN-team games in the solve (they affect opponent ratings) but
    # strip CN teams from the displayed checkpoints/events below.
    year_games = [g for g in all_games if g["date"].year == 2026]

    if not year_games:
        print("  No 2026 games found — timeline will be empty.")
        return [], []

    match_days = sorted(set(g["date"].date() for g in year_games))
    print(f"  {len(year_games)} map games across {len(match_days)} match days in 2026")

    checkpoints  = []
    match_events = []
    prev_ratings = {}

    for i, day in enumerate(match_days):
        day_dt = datetime(day.year, day.month, day.day)

        # Fresh-slate: only 2026 games through today
        games_to_day = [g for g in year_games if g["date"].date() <= day]

        ratings_before = prev_ratings  # {} on day 0

        ratings_after = massey_ratings(games_to_day, LAMBDA_DECAY, day_dt, MIN_GAMES)
        prev_ratings  = ratings_after

        checkpoints.append({
            "date":    day.isoformat(),
            "ratings": {k: round(v, 4) for k, v in ratings_after.items()
                        if k not in CN_TEAMS},
        })

        # Group today's games by match_id for delta computation
        day_games = [g for g in year_games if g["date"].date() == day]
        by_match  = defaultdict(list)
        for g in day_games:
            by_match[g["match_id"]].append(g)

        for mid, maps in by_match.items():
            map_wins:   dict = {}
            teams_seen: set  = set()
            for g in maps:
                map_wins[g["winner"]] = map_wins.get(g["winner"], 0) + 1
                teams_seen.add(g["winner"])
                teams_seen.add(g["loser"])
            if len(teams_seen) < 2:
                continue
            teams = list(teams_seen)
            winner = max(teams, key=lambda t: map_wins.get(t, 0))
            loser  = min(teams, key=lambda t: map_wins.get(t, 0))

            w_maps = map_wins.get(winner, 0)
            l_maps = map_wins.get(loser, 0)

            match_events.append({
                "match_id":      mid,
                "date":          day.isoformat(),
                "event_id":      maps[0]["event_id"],
                "winner":        winner,
                "loser":         loser,
                "series_score":  f"{w_maps}-{l_maps}",
                "maps":          [{"map": g["map_name"], "wr": g["wr"], "lr": g["lr"]} for g in maps],
                "winner_before": round(ratings_before.get(winner, 0.0), 4),
                "winner_after":  round(ratings_after.get(winner, 0.0),  4),
                "winner_delta":  round(ratings_after.get(winner, 0.0) - ratings_before.get(winner, 0.0), 4),
                "loser_before":  round(ratings_before.get(loser, 0.0), 4),
                "loser_after":   round(ratings_after.get(loser, 0.0),  4),
                "loser_delta":   round(ratings_after.get(loser, 0.0)  - ratings_before.get(loser, 0.0),  4),
            })

        if (i + 1) % 5 == 0 or (i + 1) == len(match_days):
            print(f"  Day {i+1}/{len(match_days)}: {day.isoformat()} — {len(ratings_after)} teams rated")

    return checkpoints, match_events


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    print("Loading all scraped games with actual dates...")
    all_games = load_all_games()
    print(f"Loaded {len(all_games)} map games across all events\n")

    print("Building 2026 rating timeline...")
    checkpoints, match_events = build_2026_timeline(all_games)

    out = {
        "year":         2026,
        "lambda_decay": round(LAMBDA_DECAY, 6),
        "checkpoints":  checkpoints,
        "match_events": match_events,
        "generated":    datetime.now().strftime("%Y-%m-%d"),
    }

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, separators=(",", ":"))  # compact for smaller file

    print(f"\nSaved {len(checkpoints)} checkpoints and {len(match_events)} match events.")
    print(f"Output: {OUT_PATH}")


if __name__ == "__main__":
    main()
