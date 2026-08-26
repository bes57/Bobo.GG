#!/usr/bin/env python3
"""
BuildMomentumStreaks.py — the streak each international winner carried out of
the domestic split immediately before the tournament.

Writes data/enriched/momentum_streaks.json for the Championship DNA article.

Modelled on the NCAA "End-of-Season Streaks for NCAA Champions" chart: one bar
per champion, height = the length of the run they ended the regular season on,
coloured by whether that run was wins or losses.

"Split prior" is resolved exactly as BuildSideLandscape does it -- per TEAM, the
most recent domestic split that team actually played before the international
starts -- so the two charts always describe the same stretch of matches.
"""
import os, json, glob
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from BuildSideLandscape import (INTERNATIONALS, _INTL_SET, _SPLIT_RE, _NOT_ORGS,
                                side_rates, _event_winners)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, "data", "enriched", "momentum_streaks.json")


def _series_table():
    """One row per (match, org): the org, whether it won, the event, the date.

    match_results names only the winner, so the loser is recovered from the
    series CSVs, which carry a player row per org for the MapNum='all' line.
    """
    frames = []
    for p in glob.glob(os.path.join(ROOT, "data", "series", "*.csv")):
        try:
            frames.append(pd.read_csv(p, dtype=str, usecols=["MatchID", "MapNum", "Org"]))
        except Exception:
            continue
    sp = pd.concat(frames).dropna()
    sp = sp[sp.MapNum.str.strip() == "all"]
    sp["MatchID"] = sp.MatchID.str.strip()
    parts = (sp.drop_duplicates(["MatchID", "Org"]).groupby("MatchID")["Org"].apply(list))
    parts = parts[parts.apply(len) == 2]

    mr = pd.read_csv(os.path.join(ROOT, "data", "match_results.csv"), dtype=str)
    mr["MatchID"] = mr.MatchID.str.strip()
    win = dict(zip(mr[mr.MapNum == "all"].MatchID, mr[mr.MapNum == "all"].WinnerOrg))

    ro = pd.read_csv(os.path.join(ROOT, "data", "enriched", "round_outcomes.csv"),
                     dtype=str, usecols=["match_id", "event"]).drop_duplicates()
    ev = dict(zip(ro.match_id.str.strip(), ro.event))

    dates = json.load(open(os.path.join(ROOT, "data", "match_dates.json")))

    rows = []
    for mid, orgs in parts.items():
        w, e, d = win.get(mid), ev.get(mid), dates.get(mid)
        if not w or not e or not d or w not in orgs:
            continue
        for o in orgs:
            rows.append((o, e, d, int(mid), o == w))
    return pd.DataFrame(rows, columns=["org", "event", "date", "mid", "won"])


def _terminal_streak(seq):
    """(length, 'W'|'L') for the run a sequence of booleans ends on."""
    last, n = seq[-1], 1
    for v in reversed(seq[:-1]):
        if v != last:
            break
        n += 1
    return n, ("W" if last else "L")


def build():
    sp = side_rates()
    sp = sp[~sp.org.isin(_NOT_ORGS)]
    sp["is_split"] = sp.event.str.contains(_SPLIT_RE, regex=True) & ~sp.event.isin(_INTL_SET)
    starts  = sp[sp.event.isin(_INTL_SET)].groupby("event").date.min()
    label   = dict(INTERNATIONALS)
    winners = _event_winners()
    series  = _series_table()

    out, skipped = [], []
    for ev, _ in INTERNATIONALS:
        org = winners.get(ev)
        if org is None or ev not in starts.index:
            skipped.append(f"{label[ev]} (no winner or no start date)"); continue
        prior = sp[(sp.org == org) & sp.is_split & (sp.date < starts[ev])].sort_values("date")
        if prior.empty:
            skipped.append(f"{org} @ {label[ev]} (no prior split)"); continue
        pe = prior.iloc[-1].event
        g = series[(series.org == org) & (series.event == pe)].sort_values(["date", "mid"])
        if g.empty:
            skipped.append(f"{org} @ {label[ev]} (no series rows for {pe})"); continue
        seq = list(g.won)
        n, res = _terminal_streak(seq)
        out.append({
            "org": org, "intl": label[ev], "prior": pe,
            "streak": n, "result": res,
            "record": f"{sum(seq)}-{len(seq) - sum(seq)}",
            "seq": "".join("W" if v else "L" for v in seq),
        })

    payload = {"winners": out, "skipped": skipped}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=1)
    return payload


if __name__ == "__main__":
    p = build()
    print(f"wrote {OUT}")
    for w in p["winners"]:
        print(f"  {w['org']:<4} {w['intl']:<22} {w['prior']:<40} "
              f"{w['seq']:<12} -> {w['streak']}{w['result']}  ({w['record']})")
    if p["skipped"]:
        print(f"  skipped: {p['skipped']}")
