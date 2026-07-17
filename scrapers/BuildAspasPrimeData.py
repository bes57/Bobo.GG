"""
Build data/article_aspas_prime.json for the "Greatest Prime in VCT History"
article (AspasGreatestPrime.py).

Methodology (matches the article prose):
  * International events only.
  * Minimum 150 rounds played to be a "candidate".
  * Masters London is EXCLUDED from this retrospective.
  * 2024 Masters Shanghai is kept in the round-count but VLR never published its
    ratings, so its rows have no R2.0/K:D/KPR/KAST and contribute no rated
    candidates (they fall out of every leaderboard automatically).
  * Player "role" is the majority-agent role from the over/underperformers
    article (data/article_all_roles_data.json), applied to internationals.

Run from the project root:  python scrapers/BuildAspasPrimeData.py
"""

import os
import csv
import json
from collections import defaultdict, Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

# International events, oldest first. Masters London intentionally excluded.
INTL = [
    "2023_lock_in", "2023_masters_tokyo", "2023_champions",
    "2024_masters_madrid", "2024_masters_shanghai", "2024_champions",
    "2025_masters_bangkok", "2025_masters_toronto", "2025_champions",
    "2026_masters_santiago",
]
LABEL = {
    "2023_lock_in": "2023 LOCK//IN", "2023_masters_tokyo": "2023 Masters Tokyo",
    "2023_champions": "2023 Champions", "2024_masters_madrid": "2024 Masters Madrid",
    "2024_masters_shanghai": "2024 Masters Shanghai", "2024_champions": "2024 Champions",
    "2025_masters_bangkok": "2025 Masters Bangkok", "2025_masters_toronto": "2025 Masters Toronto",
    "2025_champions": "2025 Champions", "2026_masters_santiago": "2026 Masters Santiago",
    "2026_masters_london": "2026 Masters London",
}
MIN_RND = 150
ASPAS_KEY = ("aspas", "2025_champions")  # the performance the whole piece is about


def num(s):
    s = (s or "").strip().replace("%", "")
    try:
        return float(s)
    except ValueError:
        return None


def load_event(eid):
    with open(os.path.join(DATA, eid + ".csv")) as f:
        return list(csv.DictReader(f))


def main():
    with open(os.path.join(DATA, "headshots.json")) as f:
        headshots = json.load(f)
    with open(os.path.join(DATA, "article_all_roles_data.json")) as f:
        roles = json.load(f)
    with open(os.path.join(ROOT, "static", "logos", "logos.json")) as f:
        logos = json.load(f)

    # player -> majority role across the (domestic) over/underperformers dataset
    rc = defaultdict(Counter)
    for role, lst in roles.items():
        for e in lst:
            rc[e["player"]][role] += 1
    player_role = {p: c.most_common(1)[0][0] for p, c in rc.items()}

    # Rounds-weighted KAST from the per-map data (data/maps/*.csv) so KAST carries
    # real decimals instead of VLR's rounded whole-number event aggregate. Each
    # map's KAST is weighted by that map's round count (from match_results.csv),
    # which reproduces VLR's rounded value (a plain map-average would not).
    map_rounds = {}
    with open(os.path.join(DATA, "match_results.csv")) as f:
        for r in csv.DictReader(f):
            if r["MapNum"] != "all" and "-" in r["Score"]:
                try:
                    map_rounds[r["MapNum"]] = sum(int(x) for x in r["Score"].split("-"))
                except ValueError:
                    pass
    wkast = {}  # (player, event) -> rounds-weighted KAST (decimal)
    for eid in INTL:
        mp = os.path.join(DATA, "maps", eid + ".csv")
        if not os.path.exists(mp):
            continue
        acc = defaultdict(lambda: [0.0, 0.0])  # player -> [sum(kast*rnd), sum(rnd)]
        with open(mp) as f:
            for r in csv.DictReader(f):
                k = num(r.get("KAST"))
                if k is None:
                    continue
                w = map_rounds.get(r.get("MapNum")) or 1
                a = acc[r["Player"]]
                a[0] += k * w
                a[1] += w
        for player, (s, dd) in acc.items():
            if dd:
                wkast[(player, eid)] = s / dd

    total = 0
    pool = []  # rated candidates (Rnd >= MIN_RND and a published rating)
    for eid in INTL:
        for r in load_event(eid):
            total += 1
            if (num(r["Rnd"]) or 0) < MIN_RND:
                continue
            r20 = num(r["R2.0"])
            if r20 is None:  # drops unrated Shanghai rows
                continue
            pool.append(dict(
                # The user writes his name capitalized ("Aspas"); VLR stores the
                # handle lowercase. Display it capitalized everywhere it appears.
                player=("Aspas" if r["Player"] == "aspas" else r["Player"]),
                org=r["Org"], event=eid, evlabel=LABEL[eid],
                profile=r["ProfileURL"], headshot=headshots.get(r["ProfileURL"], ""),
                rnd=int(num(r["Rnd"])), r20=r20, kd=num(r["K:D"]), kpr=num(r["KPR"]),
                kast=wkast.get((r["Player"], eid), num(r["KAST"])), acs=num(r["ACS"]),
                k=int(num(r["K"])), d=int(num(r["D"])), role=player_role.get(r["Player"]),
            ))

    # number of candidates: all Rnd>=150 international rows including Masters
    # London and unrated Shanghai (the full 437 universe, matching the DPR stats).
    qualifying = 0       # all 150+ international rows (full 437 universe)
    qualifying_rated = 0  # of those, the ones with a published VLR rating (excl. Shanghai)
    for eid in INTL + ["2026_masters_london"]:
        for r in load_event(eid):
            if (num(r["Rnd"]) or 0) >= MIN_RND:
                qualifying += 1
                if num(r["R2.0"]) is not None:
                    qualifying_rated += 1

    def is_aspas(p):
        return (p["player"].lower(), p["event"]) == ASPAS_KEY

    def row(p, val, disp):
        lf = logos.get(p["org"], "")
        return dict(player=p["player"], org=p["org"], evlabel=p["evlabel"],
                    profile=p["profile"], headshot=p["headshot"],
                    logo=("/logos/" + lf) if lf else "",
                    value=val, disp=disp, is_aspas=is_aspas(p))

    def leaderboard(key, n, fmt, tiebreak):
        s = sorted(pool, key=lambda x: (x[key], tiebreak(x)), reverse=True)
        return [dict(rank=i, **row(p, p[key], fmt(p[key]))) for i, p in enumerate(s[:n], 1)]

    out = {}
    out["counts"] = dict(total=total, qualifying=qualifying,
                         qualifying_rated=qualifying_rated, rated_pool=len(pool))
    out["rating_top10"] = leaderboard("r20", 10, lambda v: "%.2f" % v, lambda x: x["kd"])
    out["kd_top10"] = leaderboard("kd", 10, lambda v: "%.2f" % v, lambda x: x["r20"])
    out["kpr_top10"] = leaderboard("kpr", 10, lambda v: "%.2f" % v, lambda x: x["r20"])
    out["kast_top15"] = leaderboard("kast", 15, lambda v: "%.1f%%" % v, lambda x: x["r20"])

    out["kd_strip"] = [dict(value=p["kd"], player=p["player"], org=p["org"],
                            evlabel=p["evlabel"], headshot=p["headshot"],
                            profile=p["profile"], is_aspas=is_aspas(p))
                       for p in pool if p["kd"] is not None]

    byrole = defaultdict(list)
    for p in pool:
        if p["role"] and p["kast"] is not None:
            byrole[p["role"]].append(p["kast"])
    out["role_kast"] = [dict(role=role, kast=round(sum(v) / len(v), 1), n=len(v))
                        for role, v in sorted(byrole.items(),
                                              key=lambda kv: -sum(kv[1]) / len(kv[1]))
                        if role != "Flex"]

    duel = [p for p in pool if p["role"] == "Duelist" and p["kast"] is not None]
    ds = sorted(duel, key=lambda x: (x["kast"], x["r20"]), reverse=True)
    out["duelist_kast_top5"] = [dict(rank=i, **row(p, p["kast"], "%.1f%%" % p["kast"]))
                                for i, p in enumerate(ds[:5], 1)]
    out["duelist_strip"] = [dict(value=p["kast"], player=p["player"], org=p["org"],
                                 evlabel=p["evlabel"], headshot=p["headshot"],
                                 profile=p["profile"], is_aspas=is_aspas(p)) for p in duel]
    out["duelist_n"] = len(duel)

    # average # of players per international with >=1 single MATCH at 1.3+ rating
    counts = []
    for eid in INTL:
        path = os.path.join(DATA, "series", eid + ".csv")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            hit = set(r["Player"] for r in csv.DictReader(f) if (num(r["R2.0"]) or 0) >= 1.3)
        counts.append(len(hit))
    out["avg_1p3_match"] = round(sum(counts) / len(counts), 1)

    # percentage of players per international with >=1 single MATCH at 1.3+ rating,
    # averaged across internationals. Shanghai is skipped (no published ratings).
    pcts = []
    for eid in INTL:
        path = os.path.join(DATA, "series", eid + ".csv")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            rows = list(csv.DictReader(f))
        rated = [r for r in rows if num(r["R2.0"]) is not None]
        if not rated:
            continue
        allp = set(r["Player"] for r in rows)
        hit = set(r["Player"] for r in rows if (num(r["R2.0"]) or 0) >= 1.3)
        pcts.append(100 * len(hit) / len(allp))
    out["avg_1p3_pct"] = round(sum(pcts) / len(pcts))

    club = sorted([p for p in pool if p["r20"] >= 1.30], key=lambda x: -x["r20"])
    out["club13"] = [dict(player=p["player"], org=p["org"], evlabel=p["evlabel"],
                          r20="%.2f" % p["r20"]) for p in club]

    asp = next(p for p in pool if is_aspas(p))
    out["aspas"] = dict(player="aspas", org="MIBR", evlabel="2025 Champions",
                        headshot=asp["headshot"], profile=asp["profile"],
                        r20="%.2f" % asp["r20"], kd="%.2f" % asp["kd"],
                        kpr="%.2f" % asp["kpr"], kast="%d%%" % int(asp["kast"]),
                        k=asp["k"], d=asp["d"], rnd=asp["rnd"], acs="%.1f" % asp["acs"])

    # ── "What About Baiting?" section ────────────────────────────────────
    # DPR / FIPR / FIWR over ALL 150+ international performances INCLUDING
    # Masters London (11 events) = the "437" universe. These stats don't need
    # VLR ratings, so Shanghai's unrated rows and London are all included.
    BINTL = INTL + ["2026_masters_london"]
    bpool = []
    for eid in BINTL:
        for r in load_event(eid):
            if (num(r["Rnd"]) or 0) < MIN_RND:
                continue
            bpool.append(dict(
                player=("Aspas" if r["Player"] == "aspas" else r["Player"]),
                raw=r["Player"], org=r["Org"], event=eid, evlabel=LABEL[eid],
                profile=r["ProfileURL"], rnd=num(r["Rnd"]),
                d=num(r["D"]), fk=num(r["FK"]), fd=num(r["FD"]),
                role=player_role.get(r["Player"])))

    def b_is_aspas(p):
        return p["raw"] == "aspas" and p["event"] == "2025_champions"

    def dpr_v(p):
        return p["d"] / p["rnd"] if p["d"] is not None else None

    def fipr_v(p):
        return (p["fk"] + p["fd"]) / p["rnd"] if (p["fk"] is not None and p["fd"] is not None) else None

    def fiwr_v(p):
        if p["fk"] is None or p["fd"] is None or (p["fk"] + p["fd"]) == 0:
            return None
        return p["fk"] / (p["fk"] + p["fd"])

    def strip(rows, valfn):
        pts = []
        for p in rows:
            v = valfn(p)
            if v is None:
                continue
            pt = dict(value=round(v, 4), is_aspas=b_is_aspas(p),
                      player=p["player"], org=p["org"], evlabel=p["evlabel"])
            if b_is_aspas(p):
                pt["headshot"] = headshots.get(p["profile"], "")
            pts.append(pt)
        return pts

    asp_b = next(p for p in bpool if b_is_aspas(p))
    a_fipr = fipr_v(asp_b)
    a_dpr = dpr_v(asp_b)
    duel_f = [fipr_v(p) for p in bpool if p["role"] == "Duelist" and fipr_v(p) is not None]
    dpr_all = [dpr_v(p) for p in bpool if dpr_v(p) is not None]
    fipr_all = [fipr_v(p) for p in bpool if fipr_v(p) is not None]

    fiwr_rows = sorted([p for p in bpool if fiwr_v(p) is not None],
                       key=lambda p: fiwr_v(p), reverse=True)

    def frow(p, i):
        lf = logos.get(p["org"], "")
        return dict(rank=i, player=p["player"], org=p["org"], evlabel=p["evlabel"],
                    headshot=headshots.get(p["profile"], ""),
                    logo=("/logos/" + lf) if lf else "",
                    fi=int(p["fk"] + p["fd"]),
                    disp="%.2f%%" % (fiwr_v(p) * 100), is_aspas=b_is_aspas(p))

    def role_avg(valfn, pct):
        by = defaultdict(list)
        for p in bpool:
            v = valfn(p)
            if p["role"] and p["role"] != "Flex" and v is not None:
                by[p["role"]].append(v)
        rows = sorted(((r, sum(v) / len(v)) for r, v in by.items()), key=lambda x: -x[1])
        return [dict(role=r, disp=("%.1f%%" % (a * 100)) if pct else ("%.2f" % a)) for r, a in rows]

    out["baiting"] = dict(
        pool_n=len(bpool),
        dpr_strip=strip(bpool, dpr_v),
        fipr_strip=strip(bpool, fipr_v),
        fipr_duelist_strip=strip([p for p in bpool if p["role"] == "Duelist"], fipr_v),
        fiwr_duelist_strip=strip([p for p in bpool if p["role"] == "Duelist"],
                                 lambda p: fiwr_v(p) * 100 if fiwr_v(p) is not None else None),
        # scatter: x = total first interactions, y = FIWR% (all 150+ performances)
        fiwr_scatter=[dict(x=int(p["fk"] + p["fd"]), y=round(fiwr_v(p) * 100, 2),
                           is_aspas=b_is_aspas(p), player=p["player"], org=p["org"],
                           evlabel=p["evlabel"]) for p in bpool if fiwr_v(p) is not None],
        # ranks/percentiles (competition rank = 1 + count strictly better)
        dpr_rank=1 + sum(1 for x in dpr_all if x > a_dpr),
        dpr_pool=len(dpr_all),
        fipr_pctile=round(100 * sum(1 for x in fipr_all if x < a_fipr) / len(fipr_all)),
        duelist_fipr_pctile=round(100 * sum(1 for x in duel_f if x < a_fipr) / len(duel_f)),
        fiwr_rank=1 + sum(1 for p in fiwr_rows if fiwr_v(p) > fiwr_v(asp_b)),
        fiwr_top10=[frow(p, i) for i, p in enumerate(fiwr_rows[:10], 1)],
        role_fiwr=role_avg(fiwr_v, True),
        role_fipr=role_avg(fipr_v, False),
    )

    with open(os.path.join(DATA, "article_aspas_prime.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote data/article_aspas_prime.json "
          "(total rows %d, candidates %d, rated %d, duelists %d)"
          % (total, qualifying, len(pool), out["duelist_n"]))


if __name__ == "__main__":
    main()
