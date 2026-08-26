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

Every international gets a card, including the ones that cannot be measured --
a gap the reader can see is worth more than a silently shorter row. Each card
carries a kind, and all four are DERIVED rather than hardcoded, so a new event
falls into the right one on its own:

  rating -- the normal case.
  acs    -- the split published ACS but no rating2. True of the 2024 CN splits,
            which carry full player rows with an empty R2.0 column.
  nodata -- no domestic split ran before the event at all (LOCK//IN).
  nostage - a split ran, but it was not the last thing the team played: an
            international sat in between, so "their last split" is not their
            last competition. True only of Champions 2023, where Masters Tokyo
            sat between the Americas League and the tournament.
"""
import os, re, sys, json, glob
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from BuildSideLandscape import (INTERNATIONALS_ALL, _INTL_SET, _SPLIT_RE, _NOT_ORGS,
                                side_rates, _event_winners_all, _event_winners)
from BuildMomentumStreaks import _FRANCHISED_RE

# LOCK//IN opened the franchised era and sits ahead of everything else. It is
# absent from BuildSideLandscape's lists because it has no split to measure
# against -- which is exactly what its card says.
_LOCKIN = ("Champions Tour 2023: LOCK//IN S\u00e3o Paulo", "LOCK//IN 2023")
EVENTS = [_LOCKIN] + list(INTERNATIONALS_ALL)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT  = os.path.join(ROOT, "data", "enriched", "star_power.json")

# Enough of a split to be a rating rather than a hot night. Two Bo3s' worth.
MIN_ROUNDS = 100


def _slug(name):
    """Event name -> comparable slug, with the circuit prefix removed.

    VLR is not consistent about that prefix: the data calls one event
    "VCT 2025: Pacific Kickoff" while its own stats page is slugged
    champions-tour-2025-pacific-kickoff. Everything after the prefix does match,
    so both sides get normalised down to "2025-pacific-kickoff".
    """
    t = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return re.sub(r"^(champions-tour|vct|valorant)-", "", t)


def _stats_urls():
    """slug -> the event's VLR stats page, from the scraper's own event config."""
    from MoreTestingMaybeFiles import ALL_EVENTS
    out = {}
    for e in ALL_EVENTS:
        for url in (e.get("regions") or {}).values():
            m = re.search(r"/event/stats/\d+/([^/?]+)", url or "")
            if m:
                out.setdefault(_slug(m.group(1)), url)
    return out


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
                          usecols=["Player", "Org", "ProfileURL", "MatchID", "MapNum",
                                   "R2.0", "ACS"]))
        except Exception:
            continue
    m = pd.concat(frames)
    m["MatchID"] = m.MatchID.str.strip()
    m["MapNum"]  = m.MapNum.str.strip()
    m = m[m.MapNum.str.lower() != "all"]
    for c in ("R2.0", "ACS"):
        m[c] = pd.to_numeric(m[c], errors="coerce")
    return m.drop_duplicates(["Player", "MatchID", "MapNum"])


def build():
    rounds, ev_of_map = _map_rounds()
    m = _player_maps()
    key = list(zip(m.MatchID, m.MapNum))
    m["rounds"] = [rounds.get(k, 0) for k in key]
    m["event"]  = [ev_of_map.get(k) for k in key]
    m = m[(m.rounds > 0) & m.event.notna() & ~m.Org.isin(_NOT_ORGS)]

    def board_for(stat):
        """Rounds-weighted mean of `stat` per (player, event)."""
        d = m.dropna(subset=[stat]).copy()
        d["w"] = d[stat] * d.rounds
        g = (d.groupby(["event", "Player", "Org", "ProfileURL"], as_index=False)
               .agg(w=("w", "sum"), rounds=("rounds", "sum"), maps=("MapNum", "size")))
        g["val"] = g.w / g.rounds
        return g

    agg = board_for("R2.0")
    agg_acs = board_for("ACS")

    sp = side_rates(); sp = sp[~sp.org.isin(_NOT_ORGS)]
    sp["is_split"]  = sp.event.str.contains(_SPLIT_RE, regex=True) & ~sp.event.isin(_INTL_SET)
    sp["is_franch"] = sp.event.str.match(_FRANCHISED_RE)
    all_events = {e for e, _ in EVENTS}
    starts  = sp[sp.event.isin(all_events)].groupby("event").date.min()
    label   = dict(EVENTS)
    winners = dict(_event_winners_all())
    winners.update(_event_winners([_LOCKIN]))
    heads   = json.load(open(os.path.join(ROOT, "data", "headshots.json")))
    stats   = _stats_urls()

    def top_of(board, org, event):
        """Best qualified player from `org` at `event`, with their overall rank."""
        b = board[(board.event == event) & (board.rounds >= MIN_ROUNDS)]
        b = b.sort_values("val", ascending=False).reset_index(drop=True)
        mine = b[b.Org == org]
        if b.empty or mine.empty:
            return None
        i = mine.index[0]
        row = b.loc[i]
        return {"player": row.Player, "profile": row.ProfileURL,
                "head": heads.get(row.ProfileURL) or "",
                "val": float(row.val), "rank": int(i) + 1, "pool": int(len(b)),
                "rounds": int(row.rounds), "maps": int(row.maps)}

    out, skipped = [], []
    for ev, _ in EVENTS:
        org = winners.get(ev)
        if org is None or ev not in starts.index:
            skipped.append(f"{label[ev]} (no winner or start date)"); continue
        card = {"intl": label[ev], "org": org}

        before = sp[(sp.org == org) & (sp.date < starts[ev]) & (sp.event != ev)]
        splits = before[before.is_split].sort_values("date")
        franch = before[before.is_franch].sort_values("date")

        if splits.empty:
            card.update(kind="nodata", note="No prior data")
        elif franch.empty or franch.iloc[-1].event != splits.iloc[-1].event:
            # A split ran, but an international came after it -- so the split is
            # not the last thing this roster played.
            card.update(kind="nostage", note="No domestic stage")
        else:
            pe = splits.iloc[-1].event
            card["prior"] = pe
            hit = top_of(agg, org, pe)
            if hit:
                card.update(kind="rating", stat="VLR-rating", **hit)
                card["val"] = round(card["val"], 2)
            else:
                hit = top_of(agg_acs, org, pe)
                if hit:
                    card.update(kind="acs", stat="ACS", note="No VLR-rating published", **hit)
                    card["val"] = round(card["val"], 1)
                else:
                    card.update(kind="nodata", note="No player data")
                    skipped.append(f"{org} @ {label[ev]}: nothing usable in {pe}")
        # Link target: the leaderboard the number came from. Cards with no
        # split fall back to the tournament's own stats page.
        card["url"] = stats.get(_slug(card.get("prior") or ev)) or stats.get(_slug(ev)) or ""
        if not card["url"]:
            skipped.append(f"{label[ev]}: no VLR stats URL for {card.get('prior') or ev}")
        out.append(card)

    payload = {"stars": out, "min_rounds": MIN_ROUNDS, "skipped": skipped}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=1)
    return payload


if __name__ == "__main__":
    p = build()
    print(f"wrote {OUT}   (min {p['min_rounds']} rounds to qualify)")
    for s in p["stars"]:
        if s["kind"] in ("rating", "acs"):
            print(f"  {s['org']:<4} {s['intl']:<22} {s['kind']:<7} {s['player']:<12} "
                  f"{s['val']:>6}  #{s['rank']:<3} of {s['pool']:<3} {s['rounds']:>4} rnd  "
                  f"{'HEAD' if s['head'] else 'NO HEADSHOT':<11} {s.get('prior','')}")
        else:
            print(f"  {s['org']:<4} {s['intl']:<22} {s['kind']:<7} -- {s['note']}")
    if p["skipped"]:
        print(f"  skipped: {p['skipped']}")
