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
sys.path.insert(0, os.path.join(ROOT, "scrapers"))
from MoreTestingMaybeFiles import ALL_EVENTS

# Re-use shared model constants + the massey solver + CN shrinkage from
# BuildMapRatings so the Modern Hub timeline and the historical snapshots
# are computed by *exactly* the same math (sqrt rd-transform, Champions
# prestige multiplier, roster-continuity decay, roster-persistence).
from BuildMapRatings import (
    massey_ratings,
    _compute_intl_weights,
    INTL_EVENTS,
    HALF_LIFE_WEEKS,
    INTL_WIN_MULT  as _BMR_INTL_MULT,
    CN_PRIOR,
    CN_INTL_K,
    CN_C_MIN,
    CN_TEAMS_SET,
)

DATA_DIR = os.path.join(ROOT, "data")
OUT_PATH  = os.path.join(DATA_DIR, "rating_timeline.json")

LAMBDA_DECAY  = math.log(2) / HALF_LIFE_WEEKS  # share half-life
INTL_MULT     = _BMR_INTL_MULT                 # share intl multiplier (for legacy intl_weights helper)
MIN_GAMES     = 5     # min games for a team to appear in the timeline


def _apply_cn_shrinkage(ratings, intl_weights, prior=CN_PRIOR, K=CN_INTL_K,
                         c_min=CN_C_MIN):
    """Identical formula to BuildMapRatings._apply_cn_shrinkage."""
    out = {}
    for t, r in ratings.items():
        if t in CN_TEAMS_SET:
            c = max(c_min, min(intl_weights.get(t, 0.0) / K, 1.0))
            out[t] = c * r + (1 - c) * prior
        else:
            out[t] = r
    return out

EVENT_DATES = {
    "2023_lock_in":          ("2023-01-10", "2023-02-12"),
    "2023_league":           ("2023-01-23", "2023-10-01"),
    "2023_masters_tokyo":    ("2023-06-11", "2023-06-25"),
    "2023_champions":        ("2023-08-06", "2023-08-27"),
    "2024_kickoff":          ("2024-01-08", "2024-02-11"),
    "2024_china_kickoff":    ("2024-02-22", "2024-03-02"),
    "2024_masters_madrid":   ("2024-02-14", "2024-03-10"),
    "2024_stage1":           ("2024-03-15", "2024-05-19"),
    "2024_china_stage1":     ("2024-04-05", "2024-05-12"),
    "2024_masters_shanghai": ("2024-06-02", "2024-06-16"),
    "2024_stage2":           ("2024-06-20", "2024-08-25"),
    "2024_china_stage2":     ("2024-06-15", "2024-07-21"),
    "2024_champions":        ("2024-08-01", "2024-09-22"),
    "2025_kickoff":          ("2025-01-13", "2025-02-09"),
    "2025_china_kickoff":    ("2025-01-10", "2025-01-25"),
    "2025_masters_bangkok":  ("2025-02-12", "2025-03-09"),
    "2025_stage1":           ("2025-03-14", "2025-05-18"),
    "2025_china_stage1":     ("2025-03-13", "2025-05-04"),
    "2025_masters_toronto":  ("2025-06-07", "2025-06-29"),
    "2025_stage2":           ("2025-07-14", "2025-08-24"),
    "2025_china_stage2":     ("2025-07-03", "2025-08-24"),
    "2025_champions":        ("2025-08-28", "2025-09-21"),
    "2026_kickoff":          ("2026-01-15", "2026-02-16"),
    "2026_china_kickoff":    ("2026-01-21", "2026-02-09"),
    "2026_masters_santiago": ("2026-02-28", "2026-03-15"),
    "2026_stage1":           ("2026-04-01", "2026-05-25"),
}


# ── Massey solver ──────────────────────────────────────────────────────────────
# `massey_ratings` is imported from BuildMapRatings.py so the live timeline
# uses identical math (sqrt-rd transform, Champions multiplier, roster
# continuity, roster persistence) as the historical snapshots. No local
# duplication needed — one source of truth.


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

def build_2026_timeline(all_games, existing=None):
    """
    Build checkpoint ratings at every match day in 2026, plus per-match deltas.

    Uses previous years' international games as a decayed prior so that the
    three regional clusters (Americas/EMEA/Pacific) are always connected before
    Masters Santiago begins.  Without this, the first 1-2 cross-regional games
    at Santiago create a "component fusion" shock that produces wild spikes.
    With the prior, those games are incremental updates to an existing calibration.

    Checkpoints and match-event deltas come from 2026 results only; prior games
    silently anchor the inter-regional offset without appearing in the output.

    If ``existing`` (the previous rating_timeline.json contents) is given AND the
    only new match days are strictly after the last existing checkpoint date,
    only those new days are solved; prior checkpoints are reused verbatim.
    Massey decay uses ref_date = day, so re-solving an old day with newer games
    layered in would only produce trivially different ratings — safe to skip.

    Returns:
      checkpoints:   [{date, ratings:{org:float}}]
      match_events:  [{match_id, date, winner, loser, maps, series_score,
                       winner_before, winner_after, winner_delta,
                       loser_before,  loser_after,  loser_delta}]
    """
    year_games  = [g for g in all_games if g["date"].year == 2026]
    # Prior: all previous-year international games.  They decay naturally via the
    # same LAMBDA_DECAY (5-week half-life).  By March 2026, 2025 Champions games
    # (~27 weeks old) carry ~2% weight each — enough to keep clusters connected
    # without distorting current-season ratings.
    prior_games = [g for g in all_games
                   if g["date"].year < 2026 and g.get("event_id") in INTL_EVENTS]
    print(f"  Prior anchor: {len(prior_games)} international map games from 2023-2025")

    if not year_games:
        print("  No 2026 games found — timeline will be empty.")
        return [], []

    match_days = sorted(set(g["date"].date() for g in year_games))
    print(f"  {len(year_games)} map games across {len(match_days)} match days in 2026")

    # Incremental path: reuse old checkpoints + match_events if all changes are
    # additions *after* the last existing checkpoint date.
    checkpoints  = []
    match_events = []
    prev_ratings = {}
    start_idx    = 0

    if existing and existing.get("checkpoints"):
        try:
            last_cp = existing["checkpoints"][-1]
            last_date = datetime.strptime(last_cp["date"], "%Y-%m-%d").date()
            new_days = [d for d in match_days if d > last_date]
            existing_days = {datetime.strptime(c["date"], "%Y-%m-%d").date()
                             for c in existing["checkpoints"]}
            old_days_match = all(d in existing_days for d in match_days if d <= last_date)
            if new_days and old_days_match:
                checkpoints  = list(existing["checkpoints"])
                match_events = list(existing.get("match_events", []))
                prev_ratings = dict(last_cp.get("ratings", {}))
                start_idx    = match_days.index(new_days[0])
                print(f"  [incremental] reusing {len(checkpoints)} checkpoints; "
                      f"solving {len(new_days)} new day(s) from {new_days[0].isoformat()}")
            elif not new_days and old_days_match and len(match_days) == len(existing_days):
                print(f"  [incremental] no new match days — reusing all {len(existing['checkpoints'])} checkpoints")
                return list(existing["checkpoints"]), list(existing.get("match_events", []))
            else:
                print(f"  [full rebuild] historical match days changed — solving all {len(match_days)} days")
        except Exception as e:
            print(f"  [full rebuild] could not parse existing timeline ({e})")

    for i in range(start_idx, len(match_days)):
        day = match_days[i]
        day_dt = datetime(day.year, day.month, day.day)

        # Solve includes prior + all 2026 games through today
        solve_games  = prior_games + [g for g in year_games if g["date"].date() <= day]
        ratings_before = prev_ratings

        ratings_after_raw = massey_ratings(solve_games, LAMBDA_DECAY, day_dt, MIN_GAMES)
        # CN-only intl-confidence shrinkage: CN teams with weak intl exposure
        # get pulled toward CN_PRIOR until they prove themselves at internationals.
        # _compute_intl_weights from BuildMapRatings mirrors massey's exact
        # weighting (roster, Champions mult, sqrt rd) so c uses the right input.
        intl_w        = _compute_intl_weights(solve_games, LAMBDA_DECAY, day_dt)
        ratings_after = _apply_cn_shrinkage(ratings_after_raw, intl_w)
        prev_ratings  = ratings_after

        checkpoints.append({
            "date":    day.isoformat(),
            "ratings": {k: round(v, 4) for k, v in ratings_after.items()},
        })

        # Match events use only today's 2026 games
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
            teams  = list(teams_seen)
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
                "maps":          [{"map": g["map_name"], "wr": g["wr"], "lr": g["lr"],
                                   "winner": g["winner"]} for g in maps],
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

    existing = None
    if os.path.exists(OUT_PATH):
        try:
            with open(OUT_PATH) as f:
                existing = json.load(f)
        except Exception:
            existing = None

    print("Building 2026 rating timeline...")
    checkpoints, match_events = build_2026_timeline(all_games, existing=existing)

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
