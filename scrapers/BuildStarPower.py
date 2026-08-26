#!/usr/bin/env python3
"""
BuildStarPower.py — the best player on each international-winning roster,
measured over the domestic split immediately before the tournament.

Writes data/enriched/star_power.json for the Championship DNA article.

Rating is a ROUNDS-WEIGHTED mean of per-map R2.0, which is how VLR aggregates
and is the reason this reads data/maps rather than data/series. The series rows
are also the ones VLR can publish mid-window with a single map's numbers in the
"all maps" line (see scrapers/MatchStatsIntegrity.py); per-map rows are not
exposed to that.

The rank is the player's position on that event's leaderboard, among everyone
who cleared the round minimum, so it says how good the performance was against
the field the team was actually playing in.

The basis is the domestic split immediately before the tournament, resolved per
TEAM exactly as BuildSideLandscape does it.

Champions 2023 is excluded, for the same reason the landscape chart excludes it:
no domestic split ran before it. Masters Tokyo sat between the Americas League
and Champions, so EG's last split was not the competition they last played, and
neither reading describes "their last split" honestly.
"""
import os, re, sys, json, glob
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from BuildSideLandscape import (INTERNATIONALS, _INTL_SET, _SPLIT_RE, _NOT_ORGS,
                                side_rates, _event_winners)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, "data", "enriched", "star_power.json")

# Enough of a split to be a rating rather than a hot night. Two Bo3s' worth.
MIN_ROUNDS = 100


def _map_rounds():
    """(match_id, map_num) -> rounds played, and -> event."""
    r = pd.read_csv(os.path.join(ROOT, "data", "enriched", "round_outcomes.csv"),
                    dtype=str, usecols=["match_id", "map_num", "event"])
    r["match_id"] = r.match_id.str.strip()
    r["map_num"]  = r.map_num.str.strip()
    g = r.groupby(["match_id", "map_num"])
    return g.size(), g.event.first()


def _player_maps():
    frames = []
    for p in glob.glob(os.path.join(ROOT, "data", "maps", "*.csv")):
        try:
            frames.append(pd.read_csv(p, dtype=str,
                          usecols=["Player", "Org", "ProfileURL", "MatchID", "MapNum", "R2.0"]))
        except Exception:
            continue
    m = pd.concat(frames)
    m["MatchID"] = m.MatchID.str.strip()
    m["MapNum"]  = m.MapNum.str.strip()
    m = m[m.MapNum.str.lower() != "all"]
    m["R2.0"] = pd.to_numeric(m["R2.0"], errors="coerce")
    return m.dropna(subset=["R2.0"]).drop_duplicates(["Player", "MatchID", "MapNum"])


def build():
    rounds, ev_of_map = _map_rounds()
    m = _player_maps()
    key = list(zip(m.MatchID, m.MapNum))
    m["rounds"] = [rounds.get(k, 0) for k in key]
    m["event"]  = [ev_of_map.get(k) for k in key]
    m = m[(m.rounds > 0) & m.event.notna() & ~m.Org.isin(_NOT_ORGS)]
    m["wr"] = m["R2.0"] * m.rounds

    # Rounds-weighted rating per (player, event).
    agg = (m.groupby(["event", "Player", "Org", "ProfileURL"], as_index=False)
             .agg(wr=("wr", "sum"), rounds=("rounds", "sum"), maps=("MapNum", "size")))
    agg["rating"] = agg.wr / agg.rounds

    sp = side_rates(); sp = sp[~sp.org.isin(_NOT_ORGS)]
    sp["is_split"] = sp.event.str.contains(_SPLIT_RE, regex=True) & ~sp.event.isin(_INTL_SET)
    starts  = sp[sp.event.isin(_INTL_SET)].groupby("event").date.min()
    label   = dict(INTERNATIONALS)
    winners = _event_winners()
    heads   = json.load(open(os.path.join(ROOT, "data", "headshots.json")))

    out, skipped = [], []
    for ev, _ in INTERNATIONALS:
        org = winners.get(ev)
        if org is None or ev not in starts.index:
            skipped.append(f"{label[ev]} (no winner or start date)"); continue
        prior = sp[(sp.org == org) & sp.is_split & (sp.date < starts[ev])].sort_values("date")
        if prior.empty:
            skipped.append(f"{org} @ {label[ev]} (no prior split)"); continue
        pe = prior.iloc[-1].event

        board = agg[(agg.event == pe) & (agg.rounds >= MIN_ROUNDS)].sort_values("rating", ascending=False)
        if board.empty:
            # Distinguish "nobody played enough" from "the split has no ratings
            # at all" -- VLR never published rating2 for the 2024 CN splits, so
            # those events carry full player rows with an empty R2.0 column.
            why = ("no published player ratings" if agg[agg.event == pe].empty
                   else f"nobody cleared {MIN_ROUNDS} rounds")
            skipped.append(f"{org} @ {label[ev]}: {why} in {pe}"); continue
        board = board.reset_index(drop=True)
        mine = board[board.Org == org]
        if mine.empty:
            skipped.append(f"{org} @ {label[ev]} (no qualified {org} player in {pe})"); continue

        i = mine.index[0]
        row = board.loc[i]
        out.append({
            "intl": label[ev], "org": org, "prior": pe,
            "player": row.Player, "profile": row.ProfileURL,
            "head": heads.get(row.ProfileURL) or "",
            "rating": round(float(row.rating), 2),
            "rank": int(i) + 1, "pool": int(len(board)),
            "rounds": int(row.rounds), "maps": int(row.maps),
        })

    payload = {"stars": out, "min_rounds": MIN_ROUNDS, "skipped": skipped}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=1)
    return payload


if __name__ == "__main__":
    p = build()
    print(f"wrote {OUT}   (min {p['min_rounds']} rounds to qualify)")
    for s in p["stars"]:
        print(f"  {s['org']:<4} {s['intl']:<22} {s['player']:<12} {s['rating']:.2f}  "
              f"#{s['rank']:<3} of {s['pool']:<3}  {s['rounds']:>4} rnd   "
              f"{'HEAD' if s['head'] else 'NO HEADSHOT':<11} {s['prior']}")
    if p["skipped"]:
        print(f"  skipped: {p['skipped']}")
