#!/usr/bin/env python3
"""
BuildMomentumStreaks.py — the streak each international winner carried out of
the domestic split immediately before the tournament.

Writes data/enriched/momentum_streaks.json for the Championship DNA article.

Two views, both modelled on the NCAA originals:

  streaks -- the run a winner ended its domestic split on, measured inside that
    split. "Split prior" is resolved exactly as BuildSideLandscape does it, per
    TEAM, so this describes the same stretch of matches as the landscape chart.
    Champions 2023 is absent here for the reason it is absent from the landscape:
    no domestic split preceded it.

  last5 -- the winner's record over its last five FRANCHISED matches before the
    tournament, counted across events rather than inside the split. All TEN
    internationals appear, Champions 2023 included -- not needing a split before
    it is the whole point of counting across events. Crossing the
    event boundary matters because a run into an international routinely spans a
    split and a previous international. Third-party events are excluded: a team
    that entered EWC or Red Bull Home Ground would otherwise be judged on
    matches its rivals never had the chance to play.
"""
import os, re, json, glob
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from BuildSideLandscape import (INTERNATIONALS, INTERNATIONALS_ALL, _INTL_SET, _SPLIT_RE,
                                _NOT_ORGS, side_rates, _event_winners, _event_winners_all)

_INTL_ALL_SET = {e for e, _ in INTERNATIONALS_ALL}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The franchised circuit: domestic splits, their qualifiers, and the Masters /
# Champions events. Everything else in the data is third-party -- Esports World
# Cup, Red Bull Home Ground, the China Evolution Series, Shanghai Esports
# Masters, TEN, Convergence, the Radiant Invitationals -- and only some teams
# ever play them.
_FRANCHISED_RE = re.compile(
    r"^(?:Champions Tour \d{4}:|VCT \d{4}:|Valorant Champions \d{4}$|Valorant Masters )")
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


def _last_five(series, org, before):
    """Record over the org's last five matches before a date, across events."""
    g = series[(series.org == org) & (series.date < before)
               & series.event.str.match(_FRANCHISED_RE)].sort_values(["date", "mid"]).tail(5)
    if g.empty:
        return None
    w = int(g.won.sum())
    return {"w": w, "l": len(g) - w, "bucket": f"{w}-{len(g) - w}",
            "seq": "".join("W" if v else "L" for v in g.won),
            "events": sorted(set(g.event))}


def build():
    sp = side_rates()
    sp = sp[~sp.org.isin(_NOT_ORGS)]
    sp["is_split"] = sp.event.str.contains(_SPLIT_RE, regex=True) & ~sp.event.isin(_INTL_SET)
    starts  = sp[sp.event.isin(_INTL_ALL_SET)].groupby("event").date.min()
    label   = dict(INTERNATIONALS_ALL)
    winners = _event_winners_all()
    split_only = dict(INTERNATIONALS)
    series  = _series_table()

    out, skipped = [], []
    for ev, _ in INTERNATIONALS_ALL:
        org = winners.get(ev)
        if org is None or ev not in starts.index:
            skipped.append(f"{label[ev]} (no winner or no start date)"); continue
        rec = {"org": org, "intl": label[ev], "last5": _last_five(series, org, starts[ev])}

        # The within-split streak needs a split to have happened first, which is
        # exactly what Champions 2023 lacks. The last-5 above does not.
        if ev in split_only:
            prior = sp[(sp.org == org) & sp.is_split & (sp.date < starts[ev])].sort_values("date")
            if not prior.empty:
                pe = prior.iloc[-1].event
                g = series[(series.org == org) & (series.event == pe)].sort_values(["date", "mid"])
                if not g.empty:
                    seq = list(g.won)
                    n, res = _terminal_streak(seq)
                    rec.update({"prior": pe, "streak": n, "result": res,
                                "record": f"{sum(seq)}-{len(seq) - sum(seq)}",
                                "seq": "".join("W" if v else "L" for v in seq)})
        if "streak" not in rec:
            skipped.append(f"{org} @ {label[ev]} (no within-split streak)")
        out.append(rec)

    # Every 5-match bucket, including the empty ones -- a wedge missing from the
    # chart says something, but only if the reader can see where it would be.
    buckets = [f"{w}-{5 - w}" for w in range(5, -1, -1)]
    tally = {b: [] for b in buckets}
    for r in out:
        if r.get("last5"):
            tally[r["last5"]["bucket"]].append({"org": r["org"], "label": r["intl"]})

    payload = {"winners": out, "buckets": buckets,
               "tally": [{"bucket": b, "n": len(tally[b]), "who": tally[b]} for b in buckets],
               "skipped": skipped}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=1)
    return payload


if __name__ == "__main__":
    p = build()
    print(f"wrote {OUT}")
    for w in p["winners"]:
        l5 = w.get("last5") or {}
        st = f"{w['streak']}{w['result']}" if "streak" in w else "--"
        print(f"  {w['org']:<4} {w['intl']:<22} {w.get('seq','-'):<12} -> {st:<3}"
              f"   last5 {l5.get('seq','?'):<6} {l5.get('bucket','?')}")
    print("  last-5 buckets: " + ", ".join(f"{t['bucket']}={t['n']}" for t in p["tally"]))
    if p["skipped"]:
        print(f"  skipped: {p['skipped']}")
