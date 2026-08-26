#!/usr/bin/env python3
"""
BuildStarPower.py — the best player on each international-winning roster,
measured over the domestic split immediately before the tournament.

Writes data/enriched/star_power.json for the Championship DNA article.

The board is READ FROM VLR, not recomputed. An earlier version aggregated
per-map ratings locally and landed within 0.01 of VLR on almost every player --
but "almost" is not good enough for a rank, and the card links straight to the
page a reader can check it against:

  * VLR's round minimum for an event page is lower than any sensible local one.
    mwzera cleared it at Champions Tour 2024: Americas Kickoff on 90 rounds and
    sat top of the board; a 100-round cut dropped him and moved everyone below
    him up a place.
  * Ties at two decimals are common and VLR breaks them on values it does not
    publish. keznit and zekken both show 1.14 there, ordered keznit first,
    while the local numbers ordered them the other way.

Boards are cached to data/enriched/vlr_boards.json, so a rebuild costs nothing
unless a new event is added or the cache is deleted.

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


BOARDS = os.path.join(ROOT, "data", "enriched", "vlr_boards.json")


def _fetch_board(url):
    """One VLR event stats page -> [{player, profile, team, rounds, rating, acs}]."""
    from curl_cffi import requests as cr
    from bs4 import BeautifulSoup
    r = cr.get(url, impersonate="chrome131", timeout=30)
    r.raise_for_status()
    tbl = BeautifulSoup(r.text, "html.parser").find("table")
    if tbl is None:
        return []
    out = []
    for tr in tbl.find_all("tr")[1:]:
        cell = tr.find("td")
        a = cell.find("a") if cell else None
        name = cell.select_one(".st-pl-name") if cell else None
        if not a or not name:
            continue
        team = cell.select_one(".st-pl-country")

        def col(c):
            td = tr.find("td", attrs={"data-col": c})
            return td.get_text(strip=True) if td else ""

        def num(c):
            try:
                return float(col(c))
            except ValueError:
                return None

        out.append({
            "player":  name.get_text(strip=True),
            "profile": "https://www.vlr.gg" + a["href"],
            "team":    team.get_text(strip=True) if team else "",
            "rounds":  int(num("rnd") or 0),
            "rating":  num("rating2"),
            "acs":     num("acs"),
        })
    return out


def boards_for(urls, refresh=False):
    """Cached VLR boards, keyed by stats URL."""
    try:
        cache = json.load(open(BOARDS))
    except Exception:
        cache = {}
    fetched = 0
    for u in urls:
        if u and (refresh or u not in cache):
            try:
                cache[u] = _fetch_board(u)
                fetched += 1
                print(f"  fetched {len(cache[u]):>3} rows  {u}")
            except Exception as e:
                print(f"  [warn] {u}: {e}")
                cache.setdefault(u, [])
    if fetched:
        os.makedirs(os.path.dirname(BOARDS), exist_ok=True)
        json.dump(cache, open(BOARDS, "w"), indent=1)
    return cache


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


def build(refresh=False):
    skipped_early = []
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

    # Work out every card's target first, then fetch the boards in one pass.
    plan = []
    for ev, _ in EVENTS:
        org = winners.get(ev)
        if org is None or ev not in starts.index:
            skipped_early.append(f"{label[ev]} (no winner or start date)"); continue
        before = sp[(sp.org == org) & (sp.date < starts[ev]) & (sp.event != ev)]
        splits = before[before.is_split].sort_values("date")
        franch = before[before.is_franch].sort_values("date")
        if splits.empty:
            kind, prior = "nodata", None
        elif franch.empty or franch.iloc[-1].event != splits.iloc[-1].event:
            # A split ran, but an international came after it -- so the split is
            # not the last thing this roster played.
            kind, prior = "nostage", None
        else:
            kind, prior = "measured", splits.iloc[-1].event
        url = stats.get(_slug(prior or ev)) or stats.get(_slug(ev)) or ""
        plan.append((ev, org, kind, prior, url))

    cache = boards_for([u for _, _, k, _, u in plan if u and k == "measured"], refresh=refresh)

    out, skipped = [], list(skipped_early)
    for ev, org, kind, prior, url in plan:
        card = {"intl": label[ev], "org": org, "url": url}
        if kind != "measured":
            card.update(kind=kind,
                        note="No prior data" if kind == "nodata" else "No domestic stage")
            out.append(card); continue

        board = cache.get(url) or []
        # VLR publishes no rating2 for the 2024 CN splits -- full boards, empty R
        # column -- so those fall back to ACS, which is populated.
        stat = "rating" if any(r.get("rating") is not None for r in board) else "acs"
        ranked = [r for r in board if r.get(stat) is not None]
        if stat == "acs":
            # VLR sorts its page by rating. With the rating column empty that
            # order carries no meaning, so the ACS board has to be sorted here --
            # taking VLR's row order put EDG's 50th-best ACS on the card.
            ranked.sort(key=lambda r: -r["acs"])
        mine = [r for r in ranked if r["team"] == org]
        if not mine:
            card.update(kind="nodata", note="No player data", prior=prior)
            skipped.append(f"{org} @ {label[ev]}: no {org} row on {url}")
            out.append(card); continue

        best = mine[0]
        # Competition ranking on the PUBLISHED value: one better than the number
        # of players ahead of you, so players showing the same figure share a
        # place. VLR's row order breaks 1.14-vs-1.14 on precision it does not
        # print, and a reader counting rows sees a tie, not a winner.
        ahead = sum(1 for r in ranked if r[stat] > best[stat])
        tied  = sum(1 for r in ranked if r[stat] == best[stat])
        card.update(kind="rating" if stat == "rating" else "acs",
                    prior=prior, player=best["player"], profile=best["profile"],
                    head=heads.get(best["profile"]) or "",
                    val=best[stat], rank=ahead + 1, pool=len(ranked),
                    tied=tied > 1, rounds=best["rounds"])
        if stat == "acs":
            card["note"] = "No VLR-rating published"
        out.append(card)

    payload = {"stars": out, "skipped": skipped}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=1)
    return payload


if __name__ == "__main__":
    p = build()
    print(f"wrote {OUT}")
    for s in p["stars"]:
        if s["kind"] in ("rating", "acs"):
            print(f"  {s['org']:<4} {s['intl']:<22} {s['kind']:<7} {s['player']:<12} "
                  f"{s['val']:>6}  {'T-' if s.get('tied') else '#'}{s['rank']:<3} of {s['pool']:<3} {s['rounds']:>4} rnd  "
                  f"{'HEAD' if s['head'] else 'NO HEADSHOT':<11} {s.get('prior','')}")
        else:
            print(f"  {s['org']:<4} {s['intl']:<22} {s['kind']:<7} -- {s['note']}")
    if p["skipped"]:
        print(f"  skipped: {p['skipped']}")
