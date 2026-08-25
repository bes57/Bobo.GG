#!/usr/bin/env python3
"""
BuildSideLandscape.py — attack/defense win% for every international-attending
team, measured over the domestic split immediately BEFORE that international.

Writes data/enriched/side_landscape.json for the Championship DNA article.

Why a builder and not a request-time computation: this reads ~115k rounds plus
every per-map CSV to recover who played each map (round_outcomes names only the
round winner, so a team's LOSSES have to be inferred from the map's other org).
That's seconds of work, far too slow to do on a page load, and it only changes
when a new event is scraped.

The measure: for one team in one event,
    attack rounds  = rounds it won on attack + rounds the opponent won on defense
    attack win%    = rounds won on attack / attack rounds
and symmetrically for defense. Globally these come out at 50.75% / 49.25%,
which is the sanity check the builder prints.

"Split prior" is resolved per TEAM, not per region — the most recent domestic
split that team actually played before the international starts. That handles
region moves and teams entering through a qualifier without a region lookup.
"""
import os, sys, json, glob, re
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, "data", "enriched", "side_landscape.json")

# International events, oldest first. LOCK//IN 2023 is absent for the same
# reason as Champions 2023 below: no domestic split preceded it.
INTERNATIONALS = [
    ("Champions Tour 2023: Masters Tokyo",     "Masters Tokyo 2023"),
    # Champions 2023 is deliberately absent. 2023 ran a single domestic League
    # split, and that split already feeds Masters Tokyo — only LCQs sat between
    # Tokyo and Champions. Including it would reuse the same numbers for two
    # different events.
    ("Champions Tour 2024: Masters Madrid",    "Masters Madrid 2024"),
    ("Champions Tour 2024: Masters Shanghai",  "Masters Shanghai 2024"),
    ("Valorant Champions 2024",                "Champions 2024"),
    ("Valorant Masters Bangkok 2025",          "Masters Bangkok 2025"),
    ("Valorant Masters Toronto 2025",          "Masters Toronto 2025"),
    ("Valorant Champions 2025",                "Champions 2025"),
    ("Valorant Masters Santiago 2026",         "Masters Santiago 2026"),
    ("Valorant Masters London 2026",           "Masters London 2026"),
]
_INTL_SET = {e for e, _ in INTERNATIONALS}

# A domestic split: the regional competition played between internationals.
_SPLIT_RE = re.compile(
    r"(?:Kickoff|Stage 1|Stage 2|:\s(?:Americas|EMEA|Pacific|China)\sLeague"
    r"|Champions China Qualifier)")

# Parsing artifacts in the upstream data — "Team tarik" vs "Team Toast" was the
# Toronto showmatch, not two orgs.
_NOT_ORGS = {"Team", "tarik", "Toast", "INTL"}


def _map_participants():
    """(match_id, map_num) -> [orgA, orgB], from the per-map stat CSVs."""
    frames = []
    for p in glob.glob(os.path.join(ROOT, "data", "maps", "*.csv")):
        try:
            frames.append(pd.read_csv(p, dtype=str, usecols=["MatchID", "MapNum", "Org"]))
        except Exception:
            continue
    mp = pd.concat(frames).dropna()
    mp["MatchID"] = mp.MatchID.str.strip()
    mp["MapNum"]  = mp.MapNum.str.strip()
    return (mp.drop_duplicates(["MatchID", "MapNum", "Org"])
              .groupby(["MatchID", "MapNum"])["Org"].apply(list))


def side_rates():
    """Per (org, event): attack/defense wins and rounds played on each side."""
    r = pd.read_csv(os.path.join(ROOT, "data", "enriched", "round_outcomes.csv"), dtype=str)
    r["match_id"] = r.match_id.str.strip()
    r["map_num"]  = r.map_num.str.strip()
    r["orgs"] = list(zip(r.match_id, r.map_num))
    r["orgs"] = r.orgs.map(_map_participants())
    r = r.dropna(subset=["orgs"])
    r = r[r.orgs.apply(lambda x: isinstance(x, list) and len(x) == 2)]

    recs = []
    for orgs, side, w, ev, dt in zip(r.orgs, r.winner_side, r.winner_org, r.event, r.date):
        if w not in orgs:
            continue
        loser = orgs[0] if orgs[1] == w else orgs[1]
        other = "defense" if side == "attack" else "attack"
        recs.append((w, ev, dt, side, 1))
        recs.append((loser, ev, dt, other, 0))
    d = pd.DataFrame(recs, columns=["org", "event", "date", "side", "won"])

    g = d.groupby(["org", "event", "side"]).agg(w=("won", "sum"), n=("won", "size")).reset_index()
    atk = g[g.side == "attack"].set_index(["org", "event"])[["w", "n"]].rename(columns={"w": "atk_w", "n": "atk_n"})
    dfn = g[g.side == "defense"].set_index(["org", "event"])[["w", "n"]].rename(columns={"w": "def_w", "n": "def_n"})
    out = atk.join(dfn, how="outer").reset_index()
    out["date"] = out.set_index(["org", "event"]).index.map(d.groupby(["org", "event"]).date.min())
    for c in ("atk_w", "atk_n", "def_w", "def_n"):
        out[c] = out[c].fillna(0).astype(int)
    return out


def _event_winners():
    """intl event -> the org that won its Grand Final. Read from match_results
    rather than hardcoded, so a new event needs no edit here."""
    mr = pd.read_csv(os.path.join(ROOT, "data", "match_results.csv"), dtype=str)
    mr["MatchID"] = mr.MatchID.str.strip()
    ro = pd.read_csv(os.path.join(ROOT, "data", "enriched", "round_outcomes.csv"),
                     dtype=str, usecols=["match_id", "event"]).drop_duplicates()
    ev = dict(zip(ro.match_id.str.strip(), ro.event))
    series = mr[mr.MapNum == "all"].assign(event=lambda d: d.MatchID.map(ev))
    out = {}
    for e, _ in INTERNATIONALS:
        sub = series[series.event == e]
        gf = sub[sub.MatchName.str.contains("Grand Final", case=False, na=False)]
        if len(gf) == 1:
            out[e] = gf.iloc[0].WinnerOrg
    return out


def build():
    sp = side_rates()
    winners = _event_winners()
    sp = sp[~sp.org.isin(_NOT_ORGS)]
    sp["is_split"] = sp.event.str.contains(_SPLIT_RE, regex=True) & ~sp.event.isin(_INTL_SET)

    starts = sp[sp.event.isin(_INTL_SET)].groupby("event").date.min()
    label  = dict(INTERNATIONALS)

    points, skipped = [], []
    for ev, _ in INTERNATIONALS:
        if ev not in starts.index:
            continue
        start = starts[ev]
        for _, row in sp[sp.event == ev].iterrows():
            prior = sp[(sp.org == row.org) & sp.is_split & (sp.date < start)].sort_values("date")
            if prior.empty:
                skipped.append(f"{row.org} @ {label[ev]} (no prior split)")
                continue
            pr = prior.iloc[-1]
            if pr.atk_n == 0 or pr.def_n == 0:
                skipped.append(f"{row.org} @ {label[ev]} (a side went unplayed)")
                continue
            points.append({
                "org": row.org,
                "intl": label[ev],
                "intl_event": ev,
                "prior": pr.event,
                "atk": round(pr.atk_w / pr.atk_n, 4),
                "dfn": round(pr.def_w / pr.def_n, 4),
                "atk_w": int(pr.atk_w), "atk_n": int(pr.atk_n),
                "def_w": int(pr.def_w), "def_n": int(pr.def_n),
                "rounds": int(pr.atk_n + pr.def_n),
                "won": winners.get(ev) == row.org,
            })

    payload = {
        "points": points,
        "internationals": [lbl for _, lbl in INTERNATIONALS],
        "global_attack_rate": round(sp.atk_w.sum() / max(1, sp.atk_n.sum()), 4),
        "winners": {label[e]: o for e, o in winners.items()},
        "skipped": skipped,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=1)
    return payload


if __name__ == "__main__":
    p = build()
    print(f"wrote {OUT}")
    print(f"  {len(p['points'])} observations, "
          f"{len({x['org'] for x in p['points']})} teams, "
          f"{len(p['internationals'])} internationals")
    print(f"  global attack round win rate: {p['global_attack_rate']} (expect ~.507)")
    print(f"  winners: {p['winners']}")
    # The article pins both axes to 40-70%. Anything outside would be clipped
    # off the chart with no visual hint, so say so loudly here.
    out = [f"{x['org']} @ {x['intl']} atk={x['atk']:.3f} def={x['dfn']:.3f}"
           for x in p["points"] if not (0.40 <= x["atk"] <= 0.70 and 0.40 <= x["dfn"] <= 0.70)]
    print(f"  outside the chart's pinned 40-70% window: {out if out else 'none'}")
    won = [x for x in p["points"] if x["won"]]
    print(f"  points flagged as tournament winners: {len(won)} (expect one per event)")
    for s in p["skipped"]:
        print(f"  skipped: {s}")
