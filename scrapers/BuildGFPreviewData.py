"""
Static snapshot for the Champions Shanghai Grand Finals preview article.

Freezes (a) the BenPom top-5 power rankings and (b) the BenPom grand-final
prediction for LOUD vs VIT (Bo5 GF, LOUD = upper bracket) exactly as the
Modern Hub computes them: v6 closed-form headline probability, v2 veto model
(decayed ban/pick rates x map strength), and per-map win probabilities.
The veto tree is enumerated exactly instead of Monte-Carlo sampled, so the
numbers are deterministic. Output: data/article_gf_preview.json.

Run from the repo root:  python3 scrapers/BuildGFPreviewData.py
"""
import os, sys, json, math, datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import MapElo  # noqa: E402

ORG_A, ORG_B, FMT, GF_UPPER = "LOUD", "VIT", "bo5_gf", "LOUD"
VETO_STEPS = {
    "bo5_gf": [("A", "ban"), ("A", "ban"), ("A", "pick"), ("B", "pick"), ("A", "pick"), ("B", "pick")],
}


def sig(x):
    return 1.0 / (1.0 + math.exp(-x))


def main():
    d = MapElo._mhub_load()
    sm = d["site_model"]
    beta, b_pick, gf_logit = sm["beta"], sm.get("b_pick", 0.0), sm.get("gf_upper_logit", 0.0)
    xoff = sm.get("xregion_offsets") or {}
    regions = d["org_regions"]
    lb = {t["org"]: t for t in d["leaderboard"]["teams"]}
    vm = d["veto_model"]
    snap = d["snap_key"]
    key = "2026_" + snap
    pool = vm.get("current_pool") if len(vm.get("current_pool") or []) >= 7 else None
    if not pool:
        cp = (vm.get("computed_pools") or {}).get(key)
        pool = cp if cp and len(cp) >= 7 else (vm.get("snap_pools") or {}).get(key)
    pool = list(pool)

    def team_obj(org):  # mirrors _getTeamObjBC: snapshot maps overlaid by live stats
        maps = {}
        st = (d.get("snap_teams") or {}).get(org)
        if st and st.get("maps"):
            for mp, v in st["maps"].items():
                maps[mp] = dict(v)
        elif org in lb:
            for m in lb[org].get("all_maps") or []:
                maps[m["map"]] = {"rating": m.get("rating"), "w": m.get("w"), "l": m.get("l"),
                                  "win_pct": m["w"] / max(1, m["w"] + m["l"])}
        for mp, ld in ((vm.get("live_map_stats") or {}).get(org) or {}).items():
            base = maps.get(mp, {})
            maps[mp] = {"rating": base.get("rating"),
                        "w": ld.get("w", base.get("w")), "l": ld.get("l", base.get("l")),
                        "win_pct": ld.get("win_pct", base.get("win_pct"))}
        return {"overall_rating": lb[org]["rating"], "maps": maps}

    tA, tB = team_obj(ORG_A), team_obj(ORG_B)
    pattA = ((vm.get("teams") or {}).get(key) or {}).get(ORG_A)
    pattB = ((vm.get("teams") or {}).get(key) or {}).get(ORG_B)

    def wp(team, m):
        mm = team["maps"].get(m)
        return (mm.get("win_pct") or 0.5) if mm else 0.5

    def ban_probs(patt, opp, rem):
        sc = {m: (((patt or {}).get("bans") or {}).get(m, 0) + 0.02) * (0.75 + wp(opp, m)) for m in rem}
        tot = sum(sc.values())
        return {m: (sc[m] / tot if tot else 1 / len(rem)) for m in rem}

    def pick_probs(patt, rem, own):
        sc = {m: (((patt or {}).get("picks") or {}).get(m, 0) + 0.02) * (0.3 + wp(own, m)) ** 2 for m in rem}
        tot = sum(sc.values())
        return {m: (sc[m] / tot if tot else 1 / len(rem)) for m in rem}

    def step_probs(side, act, rem):
        if act == "ban":
            return ban_probs(pattA if side == "A" else pattB, tB if side == "A" else tA, rem)
        return pick_probs(pattA if side == "A" else pattB, rem, tA if side == "A" else tB)

    # Most-likely full sequence: same beam as the Hub's topVetoHUB(K=1) (3 states/step, prune p<=0.005).
    states = [{"rem": list(pool), "seq": [], "prob": 1.0}]
    for side, act in VETO_STEPS[FMT]:
        nxt = []
        for st in states:
            probs = step_probs(side, act, st["rem"])
            for m in st["rem"]:
                p = probs.get(m, 0)
                if p > 0.005:
                    nxt.append({"rem": [x for x in st["rem"] if x != m],
                                "seq": st["seq"] + [{"side": side, "action": act, "map": m, "p": round(p, 4)}],
                                "prob": st["prob"] * p})
        nxt.sort(key=lambda s: -s["prob"])
        states = nxt[:3]
    for st in states:
        if st["rem"]:
            st["seq"].append({"side": "", "action": "dec", "map": st["rem"][0], "p": 1.0})
    states.sort(key=lambda s: -s["prob"])
    top = states[0]

    # Exact expectation over the whole veto tree (what the Hub's 20k-sim MC estimates).
    rA, rB = lb[ORG_A]["rating"], lb[ORG_B]["rating"]
    xadj = 0.0
    ra, rb = regions.get(ORG_A), regions.get(ORG_B)
    if ra and rb and ra != rb:
        xadj = xoff.get(ra, 0.0) - xoff.get(rb, 0.0)
    z_base = beta * (rA - rB + xadj)
    plays = {m: 0.0 for m in pool}
    wins = {m: 0.0 for m in pool}

    def walk(rem, i, prob, fate):
        if i == len(VETO_STEPS[FMT]):
            f = dict(fate)
            if rem:
                f[rem[0]] = "dec"
            for m, fc in f.items():
                if fc in ("pickA", "pickB", "dec"):
                    z = z_base + (b_pick if fc == "pickA" else (-b_pick if fc == "pickB" else 0.0))
                    plays[m] += prob
                    wins[m] += prob * sig(z)
            return
        side, act = VETO_STEPS[FMT][i]
        probs = step_probs(side, act, rem)
        for m in rem:
            p = probs.get(m, 0)
            if p <= 0:
                continue
            fate[m] = act + side
            walk([x for x in rem if x != m], i + 1, prob * p, fate)
            del fate[m]

    walk(list(pool), 0, 1.0, {})
    maps = sorted(pool, key=lambda m: -plays[m])
    per_map = [{"map": m, "p_play": round(plays[m], 4), "wp_a": round(wins[m] / plays[m], 4)} for m in maps if plays[m] > 0]

    # Headline: v6 closed form, bo5_gf with the upper-bracket logit toward LOUD.
    pA = MapElo._v6_series_prob_from_ratings(sm, rA, rB, ra, rb, FMT, upper_is_a=(GF_UPPER == ORG_A))
    pA_no_gf = MapElo._v6_series_prob_from_ratings(sm, rA, rB, ra, rb, "bo5", None)

    # "Optimal" veto for LOUD by BenPom's per-map ratings (the Hub's team-page
    # map ratings): LOUD bans the two maps where VIT's rating edge is largest and
    # picks the maps where its own edge is largest; VIT's picks follow BenPom's
    # predicted pick behaviour (argmax of the v2 pick model) among what's left.
    def map_ratings(org):
        return {m["map"]: {"rating": m.get("rating"), "w": m.get("w"), "l": m.get("l")}
                for m in (lb[org].get("all_maps") or [])}
    mrA, mrB = map_ratings(ORG_A), map_ratings(ORG_B)

    def edge(m):  # LOUD minus VIT map rating (0 if either side has no rating)
        a = (mrA.get(m) or {}).get("rating"); b = (mrB.get(m) or {}).get("rating")
        return (a - b) if (a is not None and b is not None) else 0.0

    rem = list(pool)
    opt = []
    for side, act in VETO_STEPS[FMT]:
        if side == "A":
            m = min(rem, key=edge) if act == "ban" else max(rem, key=edge)
        else:
            pp = pick_probs(pattB, rem, tB)
            m = max(rem, key=lambda x: pp[x])
        opt.append({"side": side, "action": act, "map": m})
        rem = [x for x in rem if x != m]
    if rem:
        opt.append({"side": "", "action": "dec", "map": rem[0]})
    for st in opt:
        m = st["map"]
        st["rating_a"] = (mrA.get(m) or {}).get("rating")
        st["rating_b"] = (mrB.get(m) or {}).get("rating")
        st["edge"] = round(edge(m), 2)
    optimal = {"basis": "map_ratings", "veto": opt,
               "map_ratings": {"a": {m: round(edge(m) + ((mrB.get(m) or {}).get("rating") or 0), 2) for m in pool},
                               "edge": {m: round(edge(m), 2) for m in pool}}}

    out = {
        "optimal": optimal,
        "as_of": d.get("as_of_date") or datetime.date.today().isoformat(),
        "snap_key": snap,
        "model": {"beta": beta, "b_pick": b_pick, "gf_upper_logit": gf_logit, "xadj_a_minus_b": round(xadj, 4)},
        "top5": [{"rank": t["rank"], "org": t["org"], "region": t["region"], "rating": round(t["rating"], 2),
                  "w": t.get("w"), "l": t.get("l")}
                 for t in sorted(d["leaderboard"]["teams"], key=lambda t: t["rank"])[:5]],
        "gf": {"a": ORG_A, "b": ORG_B, "fmt": FMT, "upper": GF_UPPER, "pool": pool,
               "rating_a": round(rA, 4), "rating_b": round(rB, 4),
               "pa": round(pA, 4), "pa_without_gf_logit": round(pA_no_gf, 4),
               "veto": top["seq"], "veto_seq_prob": round(top["prob"], 4),
               "maps": per_map,
               "map_winpct": {"a": {m: round(wp(tA, m), 3) for m in pool}, "b": {m: round(wp(tB, m), 3) for m in pool}}},
    }
    path = os.path.join(ROOT, "data", "article_gf_preview.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
