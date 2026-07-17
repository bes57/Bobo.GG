"""Match Data — an aggregated analytics dashboard over the VLR enrichment data.

Isolated Flask blueprint. Reads ONLY data/enriched/vlr/*.json (produced by
scrapers/enriched/) and serves organized, filterable statistics — NOT individual
match browsing. Everything is computed server-side and cached in memory; the
cache self-invalidates when the enriched set grows (so it stays live during a
backfill).

Routes:
  GET /match-data/                     the dashboard
  GET /match-data/api/stats?year=&event=&min_maps=   aggregated stats bundle
"""
import csv
import json
import os
import re
import threading

from flask import Blueprint, Response, request

match_data_bp = Blueprint("match_data_bp", __name__)

ROOT = os.path.dirname(os.path.abspath(__file__))
ENRICH_DIR = os.path.join(ROOT, "data", "enriched", "vlr")

_lock = threading.Lock()
_cache = {"ver": None, "maps": None}


# ─────────────────────────── event classification ───────────────────────────
def _year(date, event):
    if date and len(date) >= 4 and date[:4].isdigit():
        return date[:4]
    m = re.search(r"\b(20\d\d)\b", event or "")
    return m.group(1) if m else None


def _split(event):
    e = (event or "").lower()
    if "kickoff" in e: return "Kickoff"
    if "lock" in e: return "LOCK//IN"
    if "masters" in e:  # two Masters per year — name by host city
        m = re.search(r"masters[\s:]+([a-z]+)", e)
        return f"Masters {m.group(1).title()}" if m else "Masters"
    # the year-end Champions event ("Valorant Champions 2024", "…: Champions") —
    # NOT the "Champions Tour 20XX:" prefix every 2023 event carried
    if "champions" in e.replace("champions tour", ""): return "Champions"
    if "stage 1" in e or "stage1" in e: return "Stage 1"
    if "stage 2" in e or "stage2" in e: return "Stage 2"
    if "league" in e: return "League"   # 2023 domestic regular season (no stages)
    return "Other"


def _split_sort_key(s):
    base = {"Kickoff": 0, "LOCK//IN": 1, "League": 2, "Stage 1": 3, "Stage 2": 4,
            "Champions": 6}
    if s.startswith("Masters"):
        return (5, s)
    return (base.get(s, 7), s)


def _region(event):
    e = (event or "").lower()
    if "masters" in e or ("champions" in e and "tour" not in e) or "lock" in e:
        return "International"
    if "china" in e or " cn" in e or "(cn" in e: return "CN"
    if "americas" in e: return "Americas"
    if "emea" in e: return "EMEA"
    if "pacific" in e: return "Pacific"
    return "Other"


# ───────────────────────────── load + cache ─────────────────────────────────
def _dir_version():
    try:
        return sum(1 for f in os.listdir(ENRICH_DIR)
                   if f.endswith(".json") and not f.startswith("_"))
    except FileNotFoundError:
        return 0


def _load_maps():
    """Flat, compact list of every enriched map. Cached; rebuilt when the file
    count changes (kill matrices are dropped — not needed for aggregates)."""
    ver = _dir_version()
    with _lock:
        if _cache["ver"] == ver and _cache["maps"] is not None:
            return _cache["maps"]
    maps = []
    try:
        names = os.listdir(ENRICH_DIR)
    except FileNotFoundError:
        names = []
    for fn in names:
        if not fn.endswith(".json") or fn.startswith("_"):
            continue
        try:
            rec = json.load(open(os.path.join(ENRICH_DIR, fn), encoding="utf-8"))
        except Exception:
            continue
        ev, date = rec.get("event"), rec.get("date")
        yr, split, region = _year(date, ev), _split(ev), _region(ev)
        url = rec.get("url", "")
        for mp in rec.get("maps", []):
            maps.append({
                "event": ev, "date": date, "year": yr, "split": split, "region": region,
                "map_name": _clean_map(mp.get("map_name")),
                "match_id": rec.get("match_id"),
                "map_num": mp.get("map_num"),
                "url": f"{url}/?game={mp.get('map_num')}",
                "t1": mp.get("team1_org"), "t2": mp.get("team2_org"),
                "rounds": [(r.get("winner_org"), r.get("winner_side"), r.get("win_condition"))
                           for r in mp.get("rounds", [])],
                "economy": mp.get("economy", {}),
                "players": mp.get("players", []),
            })
    with _lock:
        _cache["ver"], _cache["maps"] = ver, maps
    return maps


def _clean_map(name):
    if not name:
        return name
    return re.sub(r"(PICK|BAN|DECIDER|REMAINING)$", "", name).strip() or name


# ─── per-map combat stats (from the existing data/maps CSVs, read-only) ───────
_combat_cache = {"loaded": False, "idx": {}}


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _i(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _pctf(v):
    if not v:
        return None
    return _f(str(v).replace("%", "").strip())


def _pid(url):
    m = re.search(r"/player/(\d+)", url or "")
    return m.group(1) if m else None


def _load_combat():
    """Index per-map fragging stats by (MatchID, MapNum) -> {pid: {...}} from the
    site's existing data/maps CSVs. These carry rating/ACS/KAST/ADR/K/D/HS/FK/FD
    for every player (including CN, which has no economy/perf data)."""
    if _combat_cache["loaded"]:
        return _combat_cache["idx"]
    idx = {}
    mdir = os.path.join(ROOT, "data", "maps")
    try:
        files = os.listdir(mdir)
    except FileNotFoundError:
        files = []
    for fn in files:
        if not fn.endswith(".csv"):
            continue
        with open(os.path.join(mdir, fn), encoding="utf-8") as f:
            for row in csv.DictReader(f):
                pid = _pid(row.get("ProfileURL"))
                key = (row.get("MatchID"), row.get("MapNum"))
                idx.setdefault(key, {})[pid or row.get("Player")] = {
                    "player": row.get("Player"), "org": row.get("Org"), "pid": pid,
                    "r2": _f(row.get("R2.0")), "acs": _f(row.get("ACS")),
                    "kast": _pctf(row.get("KAST")), "adr": _f(row.get("ADR")),
                    "hs": _pctf(row.get("HS%")), "k": _i(row.get("K")) or 0,
                    "d": _i(row.get("D")) or 0, "a": _i(row.get("A")) or 0,
                    "fk": _i(row.get("FK")) or 0, "fd": _i(row.get("FD")) or 0,
                }
    _combat_cache["idx"] = idx
    _combat_cache["loaded"] = True
    return idx


# ───────────────────────────── aggregation ──────────────────────────────────
def _pct(num, den):
    # None (not 0.0) when there's no sample — so "no data" renders as "—" and
    # never masquerades as a real 0% (e.g. CN matches carry no economy data).
    return round(100.0 * num / den, 1) if den else None


_SPLIT_ORDER = ["Kickoff", "Stage 1", "Stage 2", "Masters", "Champions", "LOCK//IN"]
_REGION_ORDER = ["Americas", "EMEA", "Pacific", "CN", "International"]


def build_stats(year=None, event=None, min_maps=5, region=None, split=None):
    all_maps = _load_maps()
    _COMBAT = _load_combat()
    maps = [m for m in all_maps
            if (not year or m["year"] == year)
            and (not event or m["event"] == event)
            and (not region or m["region"] == region)
            and (not split or m["split"] == split)]

    # global filter options (from the full set, not the filtered subset)
    years = sorted({m["year"] for m in all_maps if m["year"]}, reverse=True)
    events = sorted({m["event"] for m in all_maps if m["event"]})
    regions = [r for r in _REGION_ORDER if any(m["region"] == r for m in all_maps)]
    splits = sorted({m["split"] for m in all_maps if m["split"] and m["split"] != "Other"},
                    key=_split_sort_key)
    # chronological split order *within each year* (by earliest map date) so the
    # dropdown reads Kickoff → … → Masters (in the order they actually happened)
    _split_first = {}   # (year, split) -> earliest date seen
    for m in all_maps:
        y, s, d = m["year"], m["split"], m["date"]
        if y and s and s != "Other" and d:
            k = (y, s)
            if k not in _split_first or d < _split_first[k]:
                _split_first[k] = d
    split_order = {}
    for (y, s), d in _split_first.items():
        split_order.setdefault(y, []).append((d, s))
    split_order = {y: [s for _, s in sorted(v)] for y, v in split_order.items()}
    combos = sorted({(m["year"], m["region"], m["split"]) for m in all_maps
                     if m["year"] and m["region"] != "Other" and m["split"] != "Other"})

    ov = {"atk_wins": 0, "def_wins": 0, "cond": {}, "pistol_won": 0, "pistol_total": 0, "rounds": 0}
    map_agg, team_agg, player_agg = {}, {}, {}
    records = {"most_mk_map": [], "aces": [], "big_clutches": [],
              "most_clutch_map": [], "most_plants_map": [], "most_defuses_map": [], "top_econ_map": []}

    def team(org):
        return team_agg.setdefault(org, {"org": org, "maps": 0, "atk_w": 0, "atk_p": 0,
                                         "def_w": 0, "def_p": 0, "pistol_won": 0, "pistol_total": 0,
                                         "eco_w": 0, "eco_n": 0, "force_w": 0, "force_n": 0,
                                         "full_w": 0, "full_n": 0, "regions": set()})

    for m in maps:
        t1, t2 = m["t1"], m["t2"]
        ma = map_agg.setdefault(m["map_name"], {"map_name": m["map_name"], "n": 0,
                                                "atk": 0, "def": 0, "cond": {}})
        ma["n"] += 1
        if t1: tt = team(t1); tt["maps"] += 1; tt["regions"].add(m["region"])
        if t2: tt = team(t2); tt["maps"] += 1; tt["regions"].add(m["region"])

        for winner, side, cond in m["rounds"]:
            ov["rounds"] += 1
            ov["cond"][cond] = ov["cond"].get(cond, 0) + 1
            ma["cond"][cond] = ma["cond"].get(cond, 0) + 1
            loser = t2 if winner == t1 else t1
            if side == "attack":
                ov["atk_wins"] += 1; ma["atk"] += 1
                if winner: team(winner)["atk_w"] += 1; team(winner)["atk_p"] += 1
                if loser:  team(loser)["def_p"] += 1
            elif side == "defense":
                ov["def_wins"] += 1; ma["def"] += 1
                if winner: team(winner)["def_w"] += 1; team(winner)["def_p"] += 1
                if loser:  team(loser)["atk_p"] += 1

        # economy (per team per map)
        for org, e in (m["economy"] or {}).items():
            t = team(org)
            t["pistol_won"] += e.get("pistol_won", 0); t["pistol_total"] += 2
            ov["pistol_won"] += e.get("pistol_won", 0); ov["pistol_total"] += 2
            for key, src in (("eco", "eco"), ("force", "semi_eco"), ("full", "full_buy")):
                blk = e.get(src) or {}
                t[key + "_w"] += blk.get("won", 0); t[key + "_n"] += blk.get("n", 0)
            # fold semi_buy into "force" bucket too (non-full buys)
            sb = e.get("semi_buy") or {}
            t["force_w"] += sb.get("won", 0); t["force_n"] += sb.get("n", 0)

        # players — iterate the combat universe (so CN players appear with their
        # fragging stats) and merge enrichment performance where VLR provides it.
        combat_rows = _COMBAT.get((m["match_id"], m["map_num"]), {})
        perf_by = {(p.get("player_id") or ("n:" + (p.get("player") or ""))): p
                   for p in m["players"]}
        map_rounds = len(m["rounds"])
        universe = list(combat_rows.values()) or [
            {"player": p.get("player"), "org": p.get("org"), "pid": p.get("player_id")}
            for p in m["players"]]

        map_mk_leaders = []
        for cb in universe:
            pid = cb.get("pid")
            key = pid or ("n:" + (cb.get("player") or ""))
            perf = (perf_by.get(pid) if pid else None) or perf_by.get(key)
            pa = player_agg.get(key)
            if pa is None:
                pa = player_agg[key] = {
                    "player": cb.get("player"), "org": cb.get("org"), "regions": set(),
                    "maps": 0, "perf_maps": 0, "rounds": 0,
                    "mk2": 0, "mk3": 0, "mk4": 0, "mk5": 0,
                    "c1": 0, "c2": 0, "c3": 0, "c4": 0, "c5": 0,
                    "plants": 0, "defuses": 0, "econ_sum": 0,
                    "k": 0, "d": 0, "a": 0, "fk": 0, "fd": 0,
                    "r2_sum": 0.0, "acs_sum": 0.0, "kast_sum": 0.0,
                    "adr_sum": 0.0, "hs_sum": 0.0, "stat_maps": 0}
            pa["org"] = cb.get("org") or pa["org"]
            pa["regions"].add(m["region"])
            pa["maps"] += 1
            pa["rounds"] += map_rounds
            pa["k"] += cb.get("k", 0) or 0; pa["d"] += cb.get("d", 0) or 0
            pa["a"] += cb.get("a", 0) or 0
            pa["fk"] += cb.get("fk", 0) or 0; pa["fd"] += cb.get("fd", 0) or 0
            if cb.get("r2") is not None:
                pa["r2_sum"] += cb["r2"]; pa["acs_sum"] += cb.get("acs") or 0
                pa["kast_sum"] += cb.get("kast") or 0; pa["adr_sum"] += cb.get("adr") or 0
                pa["hs_sum"] += cb.get("hs") or 0; pa["stat_maps"] += 1

            if perf:
                pa["perf_maps"] += 1
                mk = perf.get("multikills") or {}; cl = perf.get("clutches") or {}
                m2, m3, m4, m5 = mk.get("2k", 0), mk.get("3k", 0), mk.get("4k", 0), mk.get("5k", 0)
                pa["mk2"] += m2; pa["mk3"] += m3; pa["mk4"] += m4; pa["mk5"] += m5
                for i in range(1, 6):
                    pa["c%d" % i] += cl.get("1v%d" % i, 0)
                pl, de, econ = perf.get("plants", 0), perf.get("defuses", 0), perf.get("econ", 0)
                pa["plants"] += pl; pa["defuses"] += de; pa["econ_sum"] += econ
                ctx = {"player": cb.get("player"), "org": cb.get("org"),
                       "map_name": m["map_name"], "event": m["event"], "date": m["date"], "url": m["url"]}
                mk_total = m2 + m3 + m4 + m5
                clutch_total = sum(cl.get("1v%d" % i, 0) for i in range(1, 6))
                map_mk_leaders.append((mk_total, ctx))
                if m5: records["aces"].append({**ctx, "value": m5})
                _push(records["most_clutch_map"], clutch_total, ctx)
                _push(records["most_plants_map"], pl, ctx)
                _push(records["most_defuses_map"], de, ctx)
                _push(records["top_econ_map"], econ, ctx)
                big = 5 if cl.get("1v5") else 4 if cl.get("1v4") else 0
                if big:
                    records["big_clutches"].append({**ctx, "value": big})
        for mk_total, ctx in map_mk_leaders:
            _push(records["most_mk_map"], mk_total, ctx)

    # finalize maps
    maps_out = []
    for ma in map_agg.values():
        rounds = ma["atk"] + ma["def"]
        c = ma["cond"]; ct = sum(c.values()) or 1
        maps_out.append({"map_name": ma["map_name"], "n": ma["n"],
                         "atk_pct": _pct(ma["atk"], rounds), "def_pct": _pct(ma["def"], rounds),
                         "elim_pct": _pct(c.get("elimination", 0), ct),
                         "defuse_pct": _pct(c.get("defuse", 0), ct),
                         "detonate_pct": _pct(c.get("detonate", 0), ct),
                         "time_pct": _pct(c.get("time", 0), ct)})
    maps_out.sort(key=lambda x: -x["n"])

    # finalize teams
    teams_out = []
    for t in team_agg.values():
        if t["maps"] < min_maps:
            continue
        # drop CN-domestic-only teams from cross-region/domestic views (VLR has
        # no economy/perf data for CN domestic); keep them if CN is chosen explicitly
        # or if they also have international maps in this filter.
        if region != "CN" and t["regions"] == {"CN"}:
            continue
        rw_w = t["atk_w"] + t["def_w"]; rw_p = t["atk_p"] + t["def_p"]
        teams_out.append({"org": t["org"], "maps": t["maps"],
                          "atk_pct": _pct(t["atk_w"], t["atk_p"]), "def_pct": _pct(t["def_w"], t["def_p"]),
                          "rw_pct": _pct(rw_w, rw_p),
                          "pistol_pct": _pct(t["pistol_won"], t["pistol_total"]),
                          "eco_pct": _pct(t["eco_w"], t["eco_n"]),
                          "force_pct": _pct(t["force_w"], t["force_n"]),
                          "full_pct": _pct(t["full_w"], t["full_n"])})
    teams_out.sort(key=lambda x: -x["rw_pct"])

    # finalize players (fragging from combat CSVs + perf-only stats as null when
    # VLR has none, e.g. CN). min_maps hides small samples.
    players_out = []
    for p in player_agg.values():
        if p["maps"] < min_maps:
            continue
        # same CN-domestic exclusion as teams (see above)
        if region != "CN" and p["regions"] == {"CN"}:
            continue
        sm, pm, d = p["stat_maps"], p["perf_maps"], p["d"]
        has = pm > 0
        mk_total = p["mk2"] + p["mk3"] + p["mk4"] + p["mk5"]
        clutch_total = p["c1"] + p["c2"] + p["c3"] + p["c4"] + p["c5"]
        players_out.append({
            "player": p["player"], "org": p["org"], "maps": p["maps"], "perf_maps": pm,
            "rating": round(p["r2_sum"] / sm, 2) if sm else None,
            "acs": round(p["acs_sum"] / sm) if sm else None,
            "kast": round(p["kast_sum"] / sm, 1) if sm else None,
            "adr": round(p["adr_sum"] / sm) if sm else None,
            "hs": round(p["hs_sum"] / sm, 1) if sm else None,
            "kd": round(p["k"] / d, 2) if d else None,
            "k": p["k"], "d": p["d"], "a": p["a"],
            "fk": p["fk"], "fd": p["fd"], "fk_diff": p["fk"] - p["fd"],
            "fb_pct": _pct(p["fk"], p["fk"] + p["fd"]),
            "fkpr": round(p["fk"] / p["rounds"], 3) if p["rounds"] else None,
            "survival": _pct(p["rounds"] - d, p["rounds"]),
            "mk_total": mk_total if has else None,
            "mk2": p["mk2"] if has else None, "mk3": p["mk3"] if has else None,
            "mk4": p["mk4"] if has else None, "mk5": p["mk5"] if has else None,
            "mk_per_map": round(mk_total / pm, 2) if has else None,
            "clutch_total": clutch_total if has else None,
            "chi": (p["c3"] + p["c4"] + p["c5"]) if has else None,
            "plants": p["plants"] if has else None,
            "defuses": p["defuses"] if has else None,
            "econ_avg": round(p["econ_sum"] / pm, 1) if has else None,
        })
    players_out.sort(key=lambda x: -(x["rating"] or 0))

    # trim records
    for k in records:
        records[k] = _top(records[k], 12)

    ov_out = {"matches": len({m["match_id"] for m in maps}), "maps": len(maps),
              "rounds": ov["rounds"], "players": len(player_agg),
              "atk_pct": _pct(ov["atk_wins"], ov["atk_wins"] + ov["def_wins"]),
              "def_pct": _pct(ov["def_wins"], ov["atk_wins"] + ov["def_wins"]),
              "pistol_pct": _pct(ov["pistol_won"], ov["pistol_total"]),
              "avg_rounds": round(ov["rounds"] / len(maps), 1) if maps else 0,
              "cond": ov["cond"]}

    return {"meta": {"years": years, "regions": regions, "splits": splits,
                     "split_order": split_order,
                     "combos": [list(c) for c in combos], "total_maps": len(all_maps)},
            "overview": ov_out, "maps": maps_out, "teams": teams_out,
            "players": players_out, "records": records}


# ─── team composition (agent) meta ──────────────────────────────────────────
_comps_cache = {"ver": None, "idx": None}
COMPS_DIR = os.path.join(ROOT, "data", "enriched", "comps")

# Real VALORANT agents — used to drop mis-scraped "agents" (player names, etc.).
VALID_AGENTS = {
    "astra", "breach", "brimstone", "chamber", "clove", "cypher", "deadlock",
    "fade", "gekko", "harbor", "iso", "jett", "kayo", "killjoy", "neon", "omen",
    "phoenix", "raze", "reyna", "sage", "skye", "sova", "tejo", "viper", "vyse",
    "waylay", "yoru",
}


def _load_comps():
    try:
        ver = len(os.listdir(COMPS_DIR))
    except FileNotFoundError:
        ver = 0
    if _comps_cache["ver"] == ver and _comps_cache["idx"] is not None:
        return _comps_cache["idx"]
    idx = {}
    try:
        names = os.listdir(COMPS_DIR)
    except FileNotFoundError:
        names = []
    for fn in names:
        if not fn.endswith(".json"):
            continue
        try:
            rec = json.load(open(os.path.join(COMPS_DIR, fn), encoding="utf-8"))
        except Exception:
            continue
        for m in rec.get("maps", []):
            idx[(rec.get("match_id"), m.get("map_num"))] = m.get("comps", {})
    _comps_cache["ver"], _comps_cache["idx"] = ver, idx
    return idx


def build_comps(year=None, event=None, region=None, split=None, min_maps=5):
    all_maps = _load_maps()
    comps_idx = _load_comps()
    maps = [m for m in all_maps
            if (not year or m["year"] == year) and (not event or m["event"] == event)
            and (not region or m["region"] == region) and (not split or m["split"] == split)]

    agent_stats, comp_stats = {}, {}
    map_agent, map_team_maps = {}, {}
    team_maps = covered = 0
    for m in maps:
        comps = comps_idx.get((m["match_id"], m["map_num"]))
        if not comps:
            continue
        covered += 1
        tally = {}
        for w, _s, _c in m["rounds"]:
            tally[w] = tally.get(w, 0) + 1
        winner = max(tally, key=tally.get) if tally else None
        mp = m["map_name"]
        for org, agents in comps.items():
            uniq = set(agents)
            won = org == winner
            team_maps += 1
            map_team_maps[mp] = map_team_maps.get(mp, 0) + 1
            for a in uniq:
                st = agent_stats.setdefault(a, {"picks": 0, "wins": 0})
                st["picks"] += 1
                st["wins"] += 1 if won else 0
                ma = map_agent.setdefault(mp, {}).setdefault(a, {"picks": 0, "wins": 0})
                ma["picks"] += 1
                ma["wins"] += 1 if won else 0
            if len(uniq) == 5:
                ck = (m["map_name"], tuple(sorted(uniq)))
                cs = comp_stats.setdefault(ck, {"n": 0, "wins": 0})
                cs["n"] += 1
                cs["wins"] += 1 if won else 0

    agents_out = [{"agent": a, "picks": s["picks"],
                   "pick_pct": _pct(s["picks"], team_maps),
                   "win_pct": _pct(s["wins"], s["picks"])}
                  for a, s in agent_stats.items() if a in VALID_AGENTS]
    agents_out.sort(key=lambda x: -(x["pick_pct"] or 0))

    comps_out = [{"map_name": mp, "agents": list(ag), "n": s["n"],
                  "win_pct": _pct(s["wins"], s["n"])}
                 for (mp, ag), s in comp_stats.items()
                 if s["n"] >= 3 and all(a in VALID_AGENTS for a in ag)]
    comps_out.sort(key=lambda x: -x["n"])

    by_map = {}
    for mp, agents in map_agent.items():
        tm = map_team_maps.get(mp, 0)
        lst = [{"agent": a, "pick_pct": _pct(s["picks"], tm), "win_pct": _pct(s["wins"], s["picks"]),
                "picks": s["picks"]} for a, s in agents.items() if a in VALID_AGENTS]
        lst.sort(key=lambda x: -(x["pick_pct"] or 0))
        by_map[mp] = lst

    return {"meta": {"covered_maps": covered, "filtered_maps": len(maps),
                     "team_maps": team_maps, "total_comps_files": _comps_cache["ver"] or 0},
            "agents": agents_out, "comps": comps_out[:60], "by_map": by_map}


def _push(lst, value, ctx):
    if value and value > 0:
        lst.append({**ctx, "value": value})


def _top(lst, n):
    return sorted(lst, key=lambda x: -x["value"])[:n]


# ─── deep stats from Riot official game data (2023-2024 international) ─────────
RIOT_GAMES = os.path.join(ROOT, "data", "riot", "games")
_riot_cache = {"ver": None, "games": None}


def _riot_region(n):
    n = (n or "").lower()
    if "americas" in n: return "Americas"
    if "emea" in n: return "EMEA"
    if "pacific" in n: return "Pacific"
    if "cn" in n or "china" in n: return "CN"
    return "International"


def _riot_split(n):
    n = (n or "").lower()
    if "kickoff" in n: return "Kickoff"
    if "stage_1" in n or "stage1" in n: return "Stage 1"
    if "stage_2" in n or "stage2" in n: return "Stage 2"
    if "lock_in" in n: return "LOCK//IN"
    if "masters" in n:
        for c in ("madrid", "shanghai", "tokyo", "bangkok", "london", "toronto"):
            if c in n: return "Masters " + c.title()
        return "Masters Tokyo" if "2023" in n else "Masters"
    if "champions" in n: return "Champions"
    return "League Play"


def _load_riot():
    try:
        ver = len(os.listdir(RIOT_GAMES))
    except FileNotFoundError:
        ver = 0
    if _riot_cache["ver"] == ver and _riot_cache["games"] is not None:
        return _riot_cache["games"]
    games = []
    try:
        names = os.listdir(RIOT_GAMES)
    except FileNotFoundError:
        names = []
    for fn in names:
        if not fn.endswith(".json"):
            continue
        try:
            g = json.load(open(os.path.join(RIOT_GAMES, fn), encoding="utf-8"))
        except Exception:
            continue
        tn = g.get("tournament")
        g["_region"], g["_split"] = _riot_region(tn), _riot_split(tn)
        g["_year"] = str(g.get("year") or "")
        games.append(g)
    _riot_cache["ver"], _riot_cache["games"] = ver, games
    return games


def _riot_filter(games, year, region, split):
    return [g for g in games
            if (not year or g["_year"] == year) and (not region or g["_region"] == region)
            and (not split or g["_split"] == split)]


def build_deep_stats(year=None, region=None, split=None, min_maps=3):
    games = _load_riot()
    sel = _riot_filter(games, year, region, split)
    agg = {}
    for g in sel:
        for epid, p in g["players"].items():
            a = agg.get(epid)
            if a is None:
                a = agg[epid] = {"handle": None, "team": None, "maps": 0, "map_deaths": {},
                                 "clutch_att": {str(i): 0 for i in range(1, 6)},
                                 "clutch_win": {str(i): 0 for i in range(1, 6)}}
                for k in ("rounds", "deaths", "save_opp", "saves", "fk", "fd", "fb_win",
                          "plants", "defuses", "last_alive", "alive_end", "alive_end_win"):
                    a[k] = 0
            a["handle"] = p.get("handle") or a["handle"]
            a["team"] = p.get("team") or a["team"]
            a["maps"] += 1
            for k in ("rounds", "deaths", "save_opp", "saves", "fk", "fd", "fb_win",
                      "plants", "defuses", "last_alive", "alive_end", "alive_end_win"):
                a[k] += p.get(k, 0)
            for i in "12345":
                a["clutch_att"][i] += (p.get("clutch_att") or {}).get(i, 0)
                a["clutch_win"][i] += (p.get("clutch_win") or {}).get(i, 0)
        for d in g.get("deaths", []):
            a = agg.get(d.get("pid"))
            if a is not None and d.get("x") is not None:
                a["map_deaths"][d["map"]] = a["map_deaths"].get(d["map"], 0) + 1

    out = []
    for a in agg.values():
        if a["maps"] < min_maps:
            continue
        catt = sum(a["clutch_att"].values())
        cwin = sum(a["clutch_win"].values())
        out.append({
            "player": a["handle"], "team": a["team"], "maps": a["maps"], "rounds": a["rounds"],
            "save_rate": _pct(a["saves"], a["save_opp"]), "saves": a["saves"], "save_opp": a["save_opp"],
            "fk": a["fk"], "fd": a["fd"], "fk_diff": a["fk"] - a["fd"], "fb_win": a["fb_win"],
            "fb_win_pct": _pct(a["fb_win"], a["fk"]),
            "clutch_win": cwin, "clutch_att": catt, "clutch_pct": _pct(cwin, catt),
            "cw3": a["clutch_win"]["3"] + a["clutch_win"]["4"] + a["clutch_win"]["5"],
            "last_alive": a["last_alive"], "plants": a["plants"], "defuses": a["defuses"],
            "alive_win": _pct(a["alive_end_win"], a["alive_end"]),
            "survival": _pct(a["alive_end"], a["rounds"]),
            "map_deaths": a["map_deaths"],
        })
    out.sort(key=lambda x: -(x["save_rate"] or 0))
    years = sorted({g["_year"] for g in games if g["_year"]}, reverse=True)
    regions = [r for r in _REGION_ORDER if any(g["_region"] == r for g in games)]
    maps = sorted({g["map"] for g in games if g.get("map")})
    return {"meta": {"games": len(games), "filtered": len(sel), "years": years,
                     "regions": regions, "maps": maps, "players": len(out)},
            "players": out}


def build_deep_deaths(player, mp, year=None, region=None, split=None):
    games = _riot_filter(_load_riot(), year, region, split)
    pts = []
    for g in games:
        if mp and g.get("map") != mp:
            continue
        for d in g.get("deaths", []):
            if (not player or d.get("handle") == player) and d.get("x") is not None:
                pts.append({"x": d["x"], "y": d["y"], "won": d["round_won"],
                            "first": d["first"], "map": g["map"]})
    return {"player": player, "map": mp, "points": pts}


# ─────────────────────────────── routes ─────────────────────────────────────
@match_data_bp.route("/")
def match_data_home():
    return Response(PAGE_HTML, mimetype="text/html")


@match_data_bp.route("/api/deep")
def match_data_api_deep():
    return Response(json.dumps(build_deep_stats(
        request.args.get("year") or None, request.args.get("region") or None,
        request.args.get("split") or None)), mimetype="application/json")


@match_data_bp.route("/api/deep/deaths")
def match_data_api_deep_deaths():
    return Response(json.dumps(build_deep_deaths(
        request.args.get("player") or None, request.args.get("map") or None,
        request.args.get("year") or None, request.args.get("region") or None,
        request.args.get("split") or None)), mimetype="application/json")


@match_data_bp.route("/api/deep/mapmeta")
def match_data_api_deep_mapmeta():
    p = os.path.join(ROOT, "data", "riot", "map_coords.json")
    return Response(open(p, encoding="utf-8").read() if os.path.isfile(p) else "{}",
                    mimetype="application/json")


@match_data_bp.route("/api/stats")
def match_data_api_stats():
    year = request.args.get("year") or None
    event = request.args.get("event") or None
    region = request.args.get("region") or None
    split = request.args.get("split") or None
    try:
        min_maps = int(request.args.get("min_maps", 5))
    except ValueError:
        min_maps = 5
    return Response(json.dumps(build_stats(year, event, min_maps, region, split)),
                    mimetype="application/json")


@match_data_bp.route("/api/comps")
def match_data_api_comps():
    year = request.args.get("year") or None
    event = request.args.get("event") or None
    region = request.args.get("region") or None
    split = request.args.get("split") or None
    return Response(json.dumps(build_comps(year, event, region, split)),
                    mimetype="application/json")


PAGE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Match Data — VCT Round, Economy &amp; Clutch Analytics</title>
<link rel="icon" type="image/svg+xml" href="/favicon.svg">
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@700;800&family=DM+Sans:wght@300;400;500;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="/static/base.css">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<style>
  :root{ --atk:#e0713f; --def:#2f9d78; --line:#f0ecf4; --chip:#f8f4fc; --acc:#7c4dd6; }
  .top-nav{ padding:18px 24px; }
  .home-logo{ height:30px; width:auto; display:block; }
  .page{ position:relative; z-index:1; padding:20px 28px 70px; max-width:1200px; margin:0 auto; width:100%; }
  .page-title{ font-family:'Plus Jakarta Sans',sans-serif; font-size:clamp(1.5rem,3.4vw,2.4rem); font-weight:800; letter-spacing:-1px; text-align:center; }
  .page-sub{ text-align:center; color:var(--soft); font-size:.9rem; max-width:720px; margin:8px auto 2px; line-height:1.6; }
  .statbar{ display:flex; flex-wrap:wrap; justify-content:center; gap:8px; margin:14px auto 2px; max-width:940px; }
  .statbar .s{ background:var(--chip); border-radius:13px; padding:8px 15px; text-align:center; min-width:82px; }
  .statbar .sv{ font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.06rem; letter-spacing:-.3px; line-height:1.1; }
  .statbar .sl{ font-size:.58rem; color:var(--soft); text-transform:uppercase; letter-spacing:.05em; margin-top:2px; }
  .sect-lbl{ font-family:'Plus Jakarta Sans',sans-serif; font-size:.7rem; font-weight:800; color:var(--soft); text-transform:uppercase; letter-spacing:.05em; margin-bottom:8px; }

  .filters{ display:flex; gap:10px; align-items:center; flex-wrap:wrap; justify-content:center; margin-bottom:8px; }
  .filters label{ font-size:.66rem; font-weight:700; letter-spacing:.07em; text-transform:uppercase; color:var(--soft); margin-right:-4px; }
  select{ appearance:none; -webkit-appearance:none; font-family:'DM Sans',sans-serif; font-size:.82rem; font-weight:500; padding:8px 32px 8px 14px; border:2px solid var(--line); border-radius:99px; color:var(--ink); outline:none; cursor:pointer; background:#fff url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='11' height='11' viewBox='0 0 24 24' fill='none' stroke='%237a6e7e' stroke-width='3.2'%3E%3Cpath d='M6 9l6 6 6-6'/%3E%3C/svg%3E") no-repeat right 12px center; }
  select:hover{ border-color:#d4b8f4; }
  select:focus{ border-color:#7c4dd6; }
  .controls{ display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
  .controls label{ font-size:.66rem; font-weight:700; letter-spacing:.07em; text-transform:uppercase; color:var(--soft); }
  .scope{ text-align:center; font-size:.74rem; color:var(--soft); margin:2px auto 0; max-width:820px; }
  .scope b{ color:var(--ink); font-weight:700; }

  .tabs{ display:flex; gap:6px; justify-content:center; flex-wrap:wrap; margin:18px 0 22px; }
  .tab{ font-family:'DM Sans',sans-serif; font-weight:700; font-size:.82rem; color:var(--soft); background:white; border:2px solid var(--line); padding:8px 18px; border-radius:99px; cursor:pointer; transition:all .15s; }
  .tab:hover{ color:var(--ink); border-color:#d4b8f4; }
  .tab.active{ background:var(--ink); color:white; border-color:var(--ink); }

  .card{ background:white; border-radius:20px; box-shadow:0 4px 24px #0000000a; padding:22px 26px; margin-bottom:20px; }
  .card h2{ font-family:'Plus Jakarta Sans',sans-serif; font-size:1.02rem; font-weight:800; margin-bottom:4px; }
  .card .sub{ color:var(--soft); font-size:.78rem; margin-bottom:16px; line-height:1.5; }

  .stat-grid{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:14px; }
  .stat{ background:var(--chip); border-radius:16px; padding:16px 18px; }
  .stat-val{ font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.5rem; letter-spacing:-.5px; }
  .stat-lbl{ font-size:.7rem; color:var(--soft); text-transform:uppercase; letter-spacing:.06em; margin-top:3px; }
  .split2{ display:grid; grid-template-columns:1fr 1fr; gap:20px; align-items:start; }
  .chart-box{ position:relative; height:300px; }

  table{ width:100%; border-collapse:collapse; font-size:.83rem; }
  th{ font-family:'Plus Jakarta Sans',sans-serif; font-size:.62rem; font-weight:700; letter-spacing:.05em; text-transform:uppercase; color:var(--soft); padding:8px 12px; text-align:center; border-bottom:2px solid var(--line); white-space:nowrap; cursor:pointer; user-select:none; }
  th:first-child, td:first-child{ text-align:left; }
  th.sa::after{ content:' ▲'; font-size:.55rem; } th.sd::after{ content:' ▼'; font-size:.55rem; }
  td{ padding:8px 12px; text-align:center; border-bottom:1px solid #faf7fc; white-space:nowrap; }
  tbody tr:hover{ background:#fdf6f0; }
  .rk{ color:var(--soft); font-size:.72rem; width:26px; }
  .org{ color:var(--soft); font-size:.72rem; }
  .nm{ font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; }
  .tcell{ display:inline-flex; align-items:center; gap:9px; }
  .tlink{ display:inline-flex; align-items:center; gap:8px; color:inherit; text-decoration:none; cursor:pointer; }
  .tlink:hover .nm{ color:var(--acc); }
  .tlogo{ width:22px; height:22px; object-fit:contain; flex-shrink:0; }
  .olink{ color:var(--soft); text-decoration:none; cursor:pointer; }
  .olink:hover{ color:var(--acc); }
  .z{ color:#cfc7d4; }
  .barfill{ display:none; }
  a.ext{ color:var(--acc); text-decoration:none; }
  .rec-grid{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:18px; }
  .rec h3{ font-family:'Plus Jakarta Sans',sans-serif; font-size:.8rem; font-weight:800; margin-bottom:8px; }
  .rec ol{ list-style:none; }
  .rec li{ display:flex; gap:8px; padding:6px 0; border-bottom:1px solid #faf7fc; font-size:.8rem; }
  .rec li:last-child{ border-bottom:none; }
  .rec .ri{ color:var(--soft); font-size:.68rem; width:16px; flex-shrink:0; padding-top:2px; }
  .rec .rbody{ flex:1; min-width:0; }
  .rec .rrow{ display:flex; align-items:baseline; gap:6px; }
  .rec .rv{ font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; margin-left:auto; white-space:nowrap; }
  .rec .rmeta{ color:var(--soft); font-size:.68rem; margin-top:1px; }
  .legend{ display:flex; gap:16px; font-size:.72rem; color:var(--soft); margin-top:10px; }
  .legend .sw{ width:11px; height:11px; border-radius:3px; display:inline-block; vertical-align:-1px; margin-right:4px; }
  .loading{ text-align:center; color:var(--soft); padding:50px; }
  .deep-banner{ background:linear-gradient(90deg,#f3ecfd,#fdf0ea); border:1px solid #e6dcf2; border-radius:16px; padding:12px 18px; font-size:.8rem; color:var(--ink); line-height:1.6; margin-bottom:16px; }
  .agent-chip{ display:inline-block; background:var(--chip); color:#5a3a8a; border-radius:7px; padding:2px 9px; font-size:.72rem; font-weight:700; text-transform:capitalize; margin:2px 5px 2px 0; }
  th[title]{ text-decoration:underline dotted var(--line); text-underline-offset:3px; }
  details.defs{ margin-top:14px; border-top:1px solid var(--line); padding-top:10px; }
  details.defs summary{ cursor:pointer; font-size:.72rem; font-weight:700; letter-spacing:.05em; text-transform:uppercase; color:var(--soft); }
  details.defs ul{ list-style:none; margin-top:10px; columns:2; column-gap:28px; }
  details.defs li{ font-size:.76rem; color:var(--soft); line-height:1.5; margin-bottom:6px; break-inside:avoid; }
  details.defs li b{ color:var(--ink); }
  @media (max-width:720px){ details.defs ul{ columns:1; } }
  @media (max-width:720px){ .split2{ grid-template-columns:1fr; } .page{ padding:16px 12px 50px; } }

  /* team card modal — mirrors the app-wide alpha-nav one (which doesn't init on this standalone page) */
  #tmodal{ position:fixed; inset:0; z-index:100000; display:none; align-items:center; justify-content:center; padding:26px; background:rgba(18,11,28,.55); backdrop-filter:blur(8px) saturate(1.1); -webkit-backdrop-filter:blur(8px) saturate(1.1); }
  #tmodal.on{ display:flex; }
  #tmodal .tm-card{ position:relative; width:min(1080px,96vw); height:70vh; max-height:92vh; background:#fff; border-radius:20px; overflow:hidden; box-shadow:0 34px 100px #00000066; animation:tmIn .22s cubic-bezier(.2,.8,.3,1); }
  @keyframes tmIn{ from{opacity:0;transform:scale(.97) translateY(10px)} to{opacity:1;transform:none} }
  #tmodal .tm-frame{ width:100%; height:100%; border:0; display:block; background:#fff; }
  #tmodal .tm-load{ position:absolute; inset:0; display:flex; align-items:center; justify-content:center; color:#9a93a6; font-family:'DM Sans',system-ui,sans-serif; font-size:.85rem; font-weight:600; }
  #tmodal .tm-x{ position:absolute; top:13px; right:13px; z-index:3; width:36px; height:36px; border:0; border-radius:50%; background:#fff; color:#16121d; font-size:1.35rem; line-height:1; cursor:pointer; box-shadow:0 4px 14px #0004; display:flex; align-items:center; justify-content:center; transition:background .15s,transform .15s; }
  #tmodal .tm-x:hover{ background:#f3eefb; transform:scale(1.06); }
  @media(max-width:600px){ #tmodal{padding:10px;} #tmodal .tm-card{width:100vw;height:96vh;border-radius:16px;} }
</style>
</head>
<body>
<div id="content-wrap">
  <div class="top-nav"><a href="/"><img src="/logo.svg" alt="Home" class="home-logo"></a></div>
  <div class="page">
    <div class="page-title">Match Data</div>
    <div class="page-sub">Round-by-round, economy, clutch &amp; multikill analytics aggregated across <b>every franchised VCT match since 2023</b> &mdash; international events (Masters, Champions, LOCK//IN) and all four domestic leagues (Americas, EMEA, Pacific, China). Use the filters to focus a year, region, split, or single event.</div>

    <div class="filters">
      <label>Year</label><select id="fYear"><option value="">…</option></select>
      <label>Region</label><select id="fRegion"><option value="">All</option></select>
      <label>Split</label><select id="fSplit"><option value="">All</option></select>
    </div>
    <div class="scope" id="scope"></div>
    <div class="statbar" id="statbar"></div>

    <div class="tabs" id="tabs">
      <div class="tab active" data-t="maps">Maps &amp; Agents</div>
      <div class="tab" data-t="teams">Team Rankings</div>
      <div class="tab" data-t="players">Players</div>
      <div class="tab" data-t="records">Records</div>
    </div>

    <div id="view"><div class="loading">Loading…</div></div>
  </div>
</div>

<div id="tmodal"><div class="tm-card"><button class="tm-x" aria-label="Close">&times;</button><div class="tm-load">Loading&hellip;</div><iframe class="tm-frame" title="Team profile"></iframe></div></div>

<script>
let DATA=null, TAB='maps', PCAT='multi', DEEP=null, COMPS=null, MAPCOORDS=null, FILTERMETA=null, MMAP='', charts=[];
async function fetchComps(){ if(COMPS===null){ try{ COMPS=await (await fetch('api/comps?'+deepQS())).json(); }catch(e){ COMPS={agents:[],comps:[],by_map:{},meta:{covered_maps:0,filtered_maps:0}}; } } return COMPS; }
const $=s=>document.querySelector(s);
// shared, branded Chart.js tooltip (replaces the default black box)
if(window.Chart){ Chart.Tooltip.positioners.cursor=(items,evt)=>({x:evt.x,y:evt.y}); }
const TT={ position:'cursor', backgroundColor:'#ffffff', titleColor:'#2a1f2d', bodyColor:'#2a1f2d',
  borderColor:'#e6dcf2', borderWidth:1, cornerRadius:12, boxPadding:6, usePointStyle:true, caretSize:0,
  padding:{top:9,bottom:9,left:12,right:12},
  titleFont:{family:'Plus Jakarta Sans',weight:'800',size:12.5},
  bodyFont:{family:'DM Sans',size:12} };
const isbad=v=>v==null||(typeof v==='number'&&isNaN(v));
const NV=v=>isbad(v)?'<span class="z">—</span>':v;
const PV=v=>isbad(v)?'<span class="z">—</span>':v+'%';
function esc(s){ return (s==null?'':String(s)).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
function cap(s){ return s?String(s).charAt(0).toUpperCase()+String(s).slice(1):s; }
function pct(v){ return (v==null||isNaN(v))?'—':v.toFixed(1)+'%'; }
function z(v){ return v?v:'<span class="z">0</span>'; }
function clearCharts(){ charts.forEach(c=>{try{c.destroy()}catch(e){}}); charts=[]; }

async function load(){
  const yr=$('#fYear').value;
  const qs=new URLSearchParams({year:yr, region:$('#fRegion').value,
    split:$('#fSplit').value, min_maps:5});   // fixed floor: hides 1–4 map noise
  $('#view').innerHTML='<div class="loading">Crunching…</div>';
  const r=await fetch('api/stats?'+qs); DATA=await r.json();
  DEEP=null; COMPS=null; MMAP='';   // deep + comp caches invalidated when filters change
  // populate filter dropdowns once; year defaults to most recent, so refetch it
  if(!FILTERMETA){ FILTERMETA=DATA.meta; refreshFilters(); if($('#fYear').value!==yr) return load(); }
  setScope(); setStatbar();
  render();
}
function setScope(){
  const y=$('#fYear').value, r=$('#fRegion').value, s=$('#fSplit').value;
  const o=DATA.overview;
  const parts=[ y||'2023–present (all-time)', r||'all regions (intl + domestic)', s||'all splits' ];
  $('#scope').innerHTML=`Showing: <b>${parts.map(esc).join('</b> · <b>')}</b>`;
}
// headline aggregates, always visible above the tabs (was the Overview tab)
function setStatbar(){
  const o=DATA.overview;
  const cells=[['Matches',o.matches.toLocaleString()],['Maps',o.maps.toLocaleString()],
    ['Rounds',o.rounds.toLocaleString()],['Players',o.players.toLocaleString()],
    ['Atk win%',pct(o.atk_pct)],['Def win%',pct(o.def_pct)],
    ['Pistol%',pct(o.pistol_pct)],['Rds/map',o.avg_rounds]];
  $('#statbar').innerHTML=cells.map(c=>`<div class="s"><div class="sv">${c[1]}</div><div class="sl">${c[0]}</div></div>`).join('');
}
const VIEWS={maps:vMaps,teams:vTeams,players:vPlayers,records:vRecords};
function render(){ clearCharts(); (VIEWS[TAB]||vMaps)(); }

function deepQS(){ return new URLSearchParams({year:$('#fYear').value, region:$('#fRegion').value, split:$('#fSplit').value}); }
async function initDeathMap(){
  if(!MAPCOORDS){ try{ MAPCOORDS=await (await fetch('api/deep/mapmeta')).json(); }catch(e){ MAPCOORDS={}; } }
  const pSel=$('#dmPlayer'), mSel=$('#dmMap'); if(!pSel||!mSel) return;
  const byName={}; DEEP.players.forEach(p=>byName[p.player]=p);
  function fillMaps(){
    const md=(byName[pSel.value]||{}).map_deaths||{};
    const maps=Object.keys(md).sort((a,b)=>md[b]-md[a]);
    mSel.innerHTML=maps.map(mp=>`<option value="${esc(mp)}">${esc(mp)} (${md[mp]})</option>`).join('');
  }
  async function draw(){
    if(!mSel.value){ if($('#dmCount'))$('#dmCount').textContent='0 deaths'; renderHeat([], ''); return; }
    const qs=new URLSearchParams({player:pSel.value, map:mSel.value, year:$('#fYear').value, region:$('#fRegion').value, split:$('#fSplit').value});
    const res=await (await fetch('api/deep/deaths?'+qs)).json();
    if($('#dmCount')) $('#dmCount').textContent=res.points.length+' deaths';
    renderHeat(res.points, mSel.value);
  }
  pSel.onchange=()=>{ fillMaps(); draw(); };
  mSel.onchange=draw;
  fillMaps(); draw();
}
function renderHeat(points, map){
  const c=$('#dmCanvas'); if(!c) return; const ctx=c.getContext('2d'), W=c.width, H=c.height;
  const mc=MAPCOORDS?MAPCOORDS[map]:null;
  function plot(){
    for(const p of points){
      if(!mc) break;
      const px=(p.y*mc.xMult+mc.xAdd)*W, py=(p.x*mc.yMult+mc.yAdd)*H;
      ctx.beginPath(); ctx.arc(px,py,5,0,7);
      ctx.fillStyle=p.first?'rgba(224,113,63,.55)':'rgba(230,40,40,.45)'; ctx.fill();
    }
  }
  ctx.clearRect(0,0,W,H); ctx.fillStyle='#0e0b14'; ctx.fillRect(0,0,W,H);
  if(mc&&mc.icon){ const img=new Image(); img.crossOrigin='anonymous';
    img.onload=()=>{ ctx.drawImage(img,0,0,W,H); plot(); };
    img.onerror=plot; img.src=mc.icon;
  } else plot();
}


/* ---- sortable table helper ---- */
function tableCard(title, sub, cols, rows, defSort){
  const id='t'+Math.random().toString(36).slice(2);
  const defs=cols.filter(c=>c.def);
  const defsHtml=defs.length?`<details class="defs"><summary>Stat definitions</summary><ul>${defs.map(c=>`<li><b>${esc(c.label)}</b> — ${esc(c.def)}</li>`).join('')}</ul></details>`:'';
  const html=`<div class="card"><h2>${esc(title)}</h2>${sub?`<div class="sub">${sub}</div>`:''}
    <div style="overflow-x:auto"><table id="${id}"><thead><tr>${cols.map((c,i)=>`<th data-i="${i}" data-n="${c.num?1:0}"${c.def?` title="${esc(c.def)}"`:''}>${esc(c.label)}</th>`).join('')}</tr></thead><tbody></tbody></table></div>${defsHtml}</div>`;
  setTimeout(()=>{
    const t=document.getElementById(id); if(!t) return;
    let sortI=defSort.i, dir=defSort.dir;
    const maxes=cols.map((c,i)=> c.bar? Math.max(...rows.map(r=>+((c.raw?c.raw(r):r[c.key]))||0)) : 0);
    function draw(){
      rows.sort((a,b)=>{ const c=cols[sortI]; let x=c.sortval?c.sortval(a):a[c.key], y=c.sortval?c.sortval(b):b[c.key];
        const xn=(x===null||x===undefined), yn=(y===null||y===undefined);
        if(xn&&yn) return 0; if(xn) return 1; if(yn) return -1;   // "no data" always last
        if(typeof x==='string'){ return dir*x.localeCompare(y); } return dir*(x-y); });
      t.tBodies[0].innerHTML=rows.map((r,ri)=>'<tr>'+cols.map((c,i)=>{
        let v=c.fmt?c.fmt(r,ri):r[c.key];
        if(isbad(v)) v='<span class="z">—</span>';
        if(c.bar && maxes[i]>0){ const raw=+((c.raw?c.raw(r):r[c.key]))||0; const w=Math.max(0,100*raw/maxes[i]);
          return `<td class="bar"><span class="barfill" style="width:${w}%"></span><span class="barval">${v}</span></td>`; }
        return `<td>${v}</td>`;
      }).join('')+'</tr>').join('');
      t.querySelectorAll('th').forEach((th,i)=>{ th.className=(i===sortI?(dir<0?'sd':'sa'):''); });
    }
    t.querySelectorAll('th').forEach(th=>{ th.onclick=()=>{ const i=+th.dataset.i;
      if(i===sortI) dir=-dir; else { sortI=i; dir=(+th.dataset.n)? -1: 1; } draw(); }; });
    draw();
  },0);
  return html;
}
const rank=(r,ri)=>`<span class="rk">${ri+1}</span>`;
// team acronym → logo + link. alpha-nav.js intercepts /team/ links app-wide and
// opens the team card modal, so no extra JS is needed here.
const teamCell=(org,ri)=>`<span class="tcell"><span class="rk">${ri+1}</span><a href="/team/${esc(org)}" class="tlink"><img class="tlogo" src="/static/logos/${esc(org)}.png" alt="" loading="lazy" onerror="this.style.visibility='hidden'"><span class="nm">${esc(org)}</span></a></span>`;
const orgLink=(org)=>org?`<a href="/team/${esc(org)}" class="olink org">${esc(org)}</a>`:'';
const nmeorg=(r)=>`<span class="nm">${esc(r.player)}</span> ${orgLink(r.org)}`;

// the two aggregate charts (side split + how-rounds-end) — used at the top of Maps
function drawOverviewCharts(){
  const o=DATA.overview;
  charts.push(new Chart($('#cSide'),{type:'bar',data:{labels:['Rounds'],datasets:[
    {label:'Attack',data:[o.atk_pct],backgroundColor:'#e0713f'},{label:'Defense',data:[o.def_pct],backgroundColor:'#2f9d78'}]},
    options:{indexAxis:'y',plugins:{legend:{position:'bottom'},
      tooltip:{...TT,callbacks:{title:()=>'Round-win share',label:ctx=>` ${ctx.dataset.label}: ${ctx.parsed.x}%`}}},
      scales:{x:{stacked:true,max:100,ticks:{callback:v=>v+'%'}},y:{stacked:true}},responsive:true,maintainAspectRatio:false}}));
  const cd=o.cond||{}; const cl=['elimination','defuse','detonate','time'];
  charts.push(new Chart($('#cCond'),{type:'doughnut',data:{labels:['Elimination','Defuse','Detonate','Time'],
    datasets:[{data:cl.map(k=>cd[k]||0),backgroundColor:['#7c4dd6','#2f9d78','#e0713f','#c9a227']}]},
    options:{plugins:{legend:{position:'bottom'},
      tooltip:{...TT,callbacks:{label:ctx=>{const t=ctx.dataset.data.reduce((a,b)=>a+b,0)||1;return ` ${ctx.label}: ${(100*ctx.parsed/t).toFixed(1)}% (${ctx.parsed.toLocaleString()} rounds)`;}}}},
      responsive:true,maintainAspectRatio:false}}));
}

/* ---- Map Meta ---- */
async function vMaps(){
  const rows=DATA.maps;
  const cols=[{label:'Map',key:'map_name',fmt:r=>`<span class="nm">${esc(r.map_name)}</span>`,sortval:r=>r.map_name},
    {label:'Maps',key:'n',num:1,def:'Times this map was played in the current filter.'},
    {label:'Attack Win%',key:'atk_pct',num:1,bar:1,fmt:r=>pct(r.atk_pct),def:'Share of all rounds on this map won by the attacking side.'},
    {label:'Defense Win%',key:'def_pct',num:1,fmt:r=>pct(r.def_pct),def:'Share of all rounds on this map won by the defending side.'},
    {label:'Elim%',key:'elim_pct',num:1,fmt:r=>pct(r.elim_pct),def:'Rounds ending by eliminating the enemy team ÷ all rounds.'},
    {label:'Defuse%',key:'defuse_pct',num:1,fmt:r=>pct(r.defuse_pct),def:'Rounds ending by the defenders defusing the spike ÷ all rounds.'},
    {label:'Detonate%',key:'detonate_pct',num:1,fmt:r=>pct(r.detonate_pct),def:'Rounds ending by the spike detonating ÷ all rounds.'},
    {label:'Time%',key:'time_pct',num:1,fmt:r=>pct(r.time_pct),def:'Rounds ending on the timer with no plant ÷ all rounds.'}];
  let h=`<div class="card"><div class="split2">
      <div><h2>Attack vs Defense</h2><div class="sub">Share of all rounds won by side in this filter.</div><div class="chart-box"><canvas id="cSide"></canvas></div></div>
      <div><h2>How rounds end</h2><div class="sub">Win-condition distribution across all rounds.</div><div class="chart-box"><canvas id="cCond"></canvas></div></div>
    </div></div>`;
  h+=`<div class="card"><h2>Attack sidedness by map</h2><div class="sub">Higher bar = more attack-favored. Bars are stacked attack (orange) vs defense (teal) round-win share.</div><div class="chart-box" style="height:${Math.max(220,rows.length*34)}px"><canvas id="cMaps"></canvas></div>
    <div class="legend"><span><span class="sw" style="background:#e0713f"></span>Attack win%</span><span><span class="sw" style="background:#2f9d78"></span>Defense win%</span></div></div>`;
  h+=tableCard('Map meta table','Round-win split and how rounds end, per map.',cols,rows,{i:0,dir:1});
  h+=`<div id="agentHub" class="card"><div class="loading">Loading agent &amp; composition meta…</div></div>`;
  $('#view').innerHTML=h;
  drawOverviewCharts();
  const sorted=rows.slice().sort((a,b)=>b.atk_pct-a.atk_pct);
  // dashed reference line at 50% — the even attack/defense split; bars past it are attack-favored
  const at50={id:'at50',afterDatasetsDraw(c){const sx=c.scales.x,a=c.chartArea,ctx=c.ctx;
    const x=sx.getPixelForValue(50);ctx.save();ctx.beginPath();ctx.setLineDash([5,4]);
    ctx.lineWidth=1.5;ctx.strokeStyle='rgba(30,25,40,.55)';ctx.moveTo(x,a.top);ctx.lineTo(x,a.bottom);ctx.stroke();
    ctx.setLineDash([]);ctx.fillStyle='rgba(30,25,40,.75)';ctx.font='700 10px system-ui';ctx.textAlign='center';
    ctx.fillText('50% even',x,a.top-4);ctx.restore();}};
  charts.push(new Chart($('#cMaps'),{type:'bar',data:{labels:sorted.map(r=>r.map_name),datasets:[
    {label:'Attack',data:sorted.map(r=>r.atk_pct),backgroundColor:'#e0713f'},
    {label:'Defense',data:sorted.map(r=>r.def_pct),backgroundColor:'#2f9d78'}]},
    plugins:[at50],
    options:{indexAxis:'y',layout:{padding:{top:14}},plugins:{legend:{display:false},
      tooltip:{...TT,callbacks:{label:ctx=>` ${ctx.dataset.label} win: ${ctx.parsed.x}%`}}},
      scales:{x:{stacked:true,max:100,ticks:{callback:v=>v+'%'}},y:{stacked:true}},responsive:true,maintainAspectRatio:false}}));
  await fetchComps();
  if(TAB==='maps') renderMapAgents();
}
// Agent & composition hub — one section (folds in the old Comp Meta tab): a
// global pick-rate chart plus a map picker that swaps the agent + comp tables
// between "all maps" and a single map.
function renderMapAgents(){
  const sec=$('#agentHub'); if(!sec) return;
  const d=COMPS||{}, meta=d.meta||{covered_maps:0,filtered_maps:0};
  const cov = meta.covered_maps<meta.filtered_maps
    ? `Composition data for <b>${meta.covered_maps.toLocaleString()}</b> of ${meta.filtered_maps.toLocaleString()} maps in this filter (still collecting — reload for more).`
    : `All <b>${(meta.filtered_maps||0).toLocaleString()}</b> maps in this filter covered.`;
  if(!d.agents||!d.agents.length){ sec.innerHTML=`<h2>Agent &amp; composition meta</h2><div class="sub">${cov}</div><div class="loading">No composition data for this filter yet.</div>`; return; }
  const bm=d.by_map||{};
  const mapsWithData=DATA.maps.map(r=>r.map_name).filter(mp=>bm[mp]&&bm[mp].length);
  const picker=`<select id="mmMapSel"><option value="">All maps</option>${mapsWithData.map(m=>`<option value="${esc(m)}"${m===MMAP?' selected':''}>${esc(m)}</option>`).join('')}</select>`;
  sec.innerHTML=`<h2>Agent &amp; composition meta</h2>
    <div class="sub">${cov} “Pick%” = share of team-maps that fielded the agent; “Win%” = map win rate when it was played. Pick a map to focus, or keep <b>All maps</b> for the combined meta.</div>
    <div class="chart-box" style="height:${Math.max(240,d.agents.length*21)}px"><canvas id="cAgents"></canvas></div>
    <div class="controls" style="justify-content:flex-start;margin:16px 0 14px"><label>Map</label>${picker}</div>
    <div id="agentTables"></div>`;
  const top=d.agents.slice().sort((a,b)=>(b.pick_pct||0)-(a.pick_pct||0));
  charts.push(new Chart($('#cAgents'),{type:'bar',data:{labels:top.map(a=>cap(a.agent)),datasets:[{label:'Pick%',data:top.map(a=>a.pick_pct),backgroundColor:'#7c4dd6'}]},
    options:{indexAxis:'y',plugins:{legend:{display:false},tooltip:{...TT,callbacks:{title:ctx=>ctx[0].label,label:ctx=>` Pick ${ctx.parsed.x}%`}}},scales:{x:{max:100,ticks:{callback:v=>v+'%'}}},responsive:true,maintainAspectRatio:false}}));
  renderAgentTables();
}
function renderAgentTables(){
  const box=$('#agentTables'); if(!box||!COMPS) return;
  const d=COMPS, bm=d.by_map||{};
  const mp=(MMAP&&bm[MMAP])?MMAP:'';            // '' = all maps combined
  const scopeLbl=mp?esc(mp):'all maps';
  const agents=(mp?bm[mp]:d.agents).slice(0,18);
  const comps=(mp?(d.comps||[]).filter(c=>c.map_name===mp):(d.comps||[])).slice(0,mp?10:16);
  const aRows=agents.map((a,i)=>`<tr><td><span class="rk">${i+1}</span> <span class="nm" style="text-transform:capitalize">${esc(a.agent)}</span></td><td>${PV(a.pick_pct)}</td><td>${PV(a.win_pct)}</td><td>${a.picks}</td></tr>`).join('');
  const cRows=comps.map((c,i)=>`<tr><td style="white-space:normal"><span class="rk">${i+1}</span>${mp?'':` <span class="org">${esc(c.map_name)}</span>`} ${c.agents.map(x=>`<span class="agent-chip">${esc(x)}</span>`).join(' ')}</td><td>${c.n}</td><td>${PV(c.win_pct)}</td></tr>`).join('')||'<tr><td colspan="3" class="org">Not enough data.</td></tr>';
  box.innerHTML=`<div class="split2">
      <div><div class="sect-lbl">Agent pick &amp; win — ${scopeLbl}</div><div style="overflow-x:auto"><table><thead><tr><th>Agent</th><th>Pick%</th><th>Win%</th><th>Maps</th></tr></thead><tbody>${aRows}</tbody></table></div></div>
      <div><div class="sect-lbl">Most-run comps — ${scopeLbl}</div><div style="overflow-x:auto"><table><thead><tr><th>Composition</th><th>Times</th><th>Win%</th></tr></thead><tbody>${cRows}</tbody></table></div></div>
    </div>`;
}

/* ---- Team Rankings ---- */
function vTeams(){
  const cols=[{label:'Team',key:'org',fmt:(r,ri)=>teamCell(r.org,ri),sortval:r=>r.org},
    {label:'Maps',key:'maps',num:1,def:'Maps played in the current filter.'},
    {label:'Round Win%',key:'rw_pct',num:1,bar:1,fmt:r=>pct(r.rw_pct),def:'Rounds won ÷ total rounds played.'},
    {label:'Attack%',key:'atk_pct',num:1,fmt:r=>pct(r.atk_pct),def:'Rounds won while attacking ÷ attack rounds played.'},
    {label:'Defense%',key:'def_pct',num:1,fmt:r=>pct(r.def_pct),def:'Rounds won while defending ÷ defense rounds played.'},
    {label:'Pistol%',key:'pistol_pct',num:1,fmt:r=>pct(r.pistol_pct),def:'The two pistol rounds (rounds 1 & 13) won ÷ pistol rounds played.'},
    {label:'Eco Win%',key:'eco_pct',num:1,fmt:r=>pct(r.eco_pct),def:'Full-save (eco) rounds won ÷ eco rounds played.'},
    {label:'Full-buy%',key:'full_pct',num:1,fmt:r=>pct(r.full_pct),def:'Full-buy rounds won ÷ full-buy rounds played.'}];
  $('#view').innerHTML=tableCard('Team rankings',
    'Round-win% overall and by side, plus economy conversion (eco = full-save rounds, full-buy = full-buy rounds). Sortable — click any header. Click a team to open its card. Teams with very few maps are hidden.',
    cols,DATA.teams,{i:2,dir:-1});
}

/* ---- Players (VLR-unique stats + Riot deep stats, unified) ---- */
async function vPlayers(){
  const cats=[{k:'multi',lbl:'Multikills'},{k:'clutch',lbl:'Clutches & Econ'},
    {k:'save',lbl:'Saving ✦',riot:1},{k:'impact',lbl:'Round Impact ✦',riot:1},{k:'deaths',lbl:'Death Maps ✦',riot:1}];
  const cur=cats.find(c=>c.k===PCAT)||cats[0];
  const btns=`<div class="tabs" style="margin:2px 0 12px">${cats.map(c=>`<div class="tab pc ${c.k===cur.k?'active':''}" data-pc="${c.k}">${c.lbl}</div>`).join('')}</div>`;
  const RP0={label:'Player',key:'player',fmt:(r,ri)=>`${rank(r,ri)} <span class="nm">${esc(r.player)}</span> <span class="org">${esc(r.team||'')}</span>`,sortval:r=>r.player};

  if(cur.riot){
    if(DEEP===null){
      $('#view').innerHTML=btns+'<div class="loading">Loading deep stats…</div>';
      try{ DEEP=await (await fetch('api/deep?'+deepQS())).json(); }
      catch(e){ if(TAB==='players')$('#view').innerHTML=btns+'<div class="loading err">Failed to load deep stats.</div>'; return; }
      if(TAB!=='players'||PCAT!==cur.k) return;
    }
    const d=DEEP, m=d.meta;
    const banner=`<div class="deep-banner">✦ From Riot's official game telemetry — positions, per-round survival &amp; economy. <b>2023–2024 international only</b> (${m.games.toLocaleString()} games${m.games<1588?' · still collecting':''}); set Year to 2023/2024 if a filter shows nothing. None of this is on VLR.</div>`;
    if(!d.players.length){ $('#view').innerHTML=btns+banner+'<div class="card"><div class="loading">No Riot data for this filter — it covers <b>2023–2024 international</b> only.</div></div>'; return; }
    let body='';
    if(cur.k==='save'){
      body=tableCard('Saving — who preserves for the next round',
        'A true save = you survived a round your team lost. From per-round survival in Riot telemetry (impossible from VLR).',
        [RP0,{label:'Maps',key:'maps',num:1,def:'Maps played (Riot data).'},
         {label:'Save Rate',key:'save_rate',num:1,bar:1,fmt:r=>PV(r.save_rate),def:'Rounds survived when your team LOST ÷ rounds your team lost.'},
         {label:'Saves',key:'saves',num:1,def:'Rounds survived despite your team losing.'},
         {label:'Lost Rounds',key:'save_opp',num:1,def:'Rounds your team lost.'},
         {label:'Survival%',key:'survival',num:1,fmt:r=>PV(r.survival),def:'Share of all rounds survived.'}],
        d.players.slice(),{i:2,dir:-1});
    } else if(cur.k==='impact'){
      body=tableCard('Round-Win Impact',
        'How a player swings rounds: does the opening kill convert, and how often is the round won when they stay alive? Includes last-alive clutch conversion.',
        [RP0,{label:'Maps',key:'maps',num:1},
         {label:'FK',key:'fk',num:1,def:'First kills — opening the round.'},
         {label:'FK +/−',key:'fk_diff',num:1,def:'First kills minus first deaths.'},
         {label:'FB→Win%',key:'fb_win_pct',num:1,fmt:r=>PV(r.fb_win_pct),def:'Rounds won when you got the opening kill ÷ your first kills.'},
         {label:'Win% Alive',key:'alive_win',num:1,bar:1,fmt:r=>PV(r.alive_win),def:'Round win rate when you were alive at round end.'},
         {label:'Clutch%',key:'clutch_pct',num:1,fmt:r=>PV(r.clutch_pct),def:'Last-alive rounds won ÷ last-alive situations faced.'},
         {label:'Plants',key:'plants',num:1},{label:'Defuses',key:'defuses',num:1}],
        d.players.slice(),{i:5,dir:-1});
    } else {
      const ps=d.players.slice().filter(p=>p.map_deaths&&Object.keys(p.map_deaths).length).sort((a,b)=>b.maps-a.maps);
      body=`<div class="card"><h2>Death map — where a player dies ◎</h2>
        <div class="sub">Every death location from Riot telemetry on the minimap. <b style="color:#e0713f">Orange</b> = opening deaths (first blood), <b style="color:#e62828">red</b> = other. Map list shows each player's death count per map.</div>
        <div class="controls" style="justify-content:flex-start;margin-bottom:0">
          <label>Player</label><select id="dmPlayer">${ps.map(p=>`<option value="${esc(p.player)}">${esc(p.player)} — ${esc(p.team||'')}</option>`).join('')}</select>
          <label>Map</label><select id="dmMap"></select>
          <span class="count-pill" id="dmCount">…</span></div>
        <div style="display:flex;justify-content:center;margin-top:16px"><canvas id="dmCanvas" width="512" height="512" style="max-width:100%;border-radius:16px;background:#0e0b14"></canvas></div></div>`;
    }
    $('#view').innerHTML=btns+banner+body;
    if(cur.k==='deaths') initDeathMap();
    return;
  }

  // VLR categories (all years) — only stats NOT already on the site's Leaderboards
  const P0={label:'Player',key:'player',fmt:(r,ri)=>`${rank(r,ri)} ${nmeorg(r)}`,sortval:r=>r.player};
  const CATS={
    multi:{sub:'Multikill rounds (2K–ace) — not on the standard Leaderboards.',
      cols:[P0,{label:'Maps',key:'perf_maps',num:1,def:'Maps played with performance data in the current filter.'},
        {label:'Multi (2K+)',key:'mk_total',num:1,bar:1,fmt:r=>NV(r.mk_total),def:'Total multikill rounds (2K, 3K, 4K and aces).'},
        {label:'2K',key:'mk2',num:1,fmt:r=>NV(r.mk2),def:'Rounds with exactly 2 kills.'},{label:'3K',key:'mk3',num:1,fmt:r=>NV(r.mk3),def:'Rounds with 3 kills.'},
        {label:'4K',key:'mk4',num:1,fmt:r=>NV(r.mk4),def:'Rounds with 4 kills.'},{label:'Aces',key:'mk5',num:1,fmt:r=>NV(r.mk5),def:'Aces — rounds with 5 kills.'},
        {label:'MK/Map',key:'mk_per_map',num:1,fmt:r=>NV(r.mk_per_map),def:'Multikill rounds per map played.'}], sort:2},
    clutch:{sub:'Clutches won (1vX), econ rating, plants and defuses — enrichment stats not on the standard Leaderboards.',
      cols:[P0,{label:'Maps',key:'maps',num:1,def:'Maps played in the current filter.'},
        {label:'Clutches',key:'clutch_total',num:1,bar:1,fmt:r=>NV(r.clutch_total),def:'Clutches won — last-alive rounds (1v1…1v5) the player won.'},
        {label:'1v3+',key:'chi',num:1,fmt:r=>NV(r.chi),def:'Won clutches against 3 or more enemies.'},
        {label:'Econ',key:'econ_avg',num:1,fmt:r=>NV(r.econ_avg),def:'VLR econ rating — combat score per 1000 credits spent.'},
        {label:'Plants',key:'plants',num:1,fmt:r=>NV(r.plants),def:'Total spike plants.'},
        {label:'Defuses',key:'defuses',num:1,fmt:r=>NV(r.defuses),def:'Total spike defuses.'}], sort:2}};
  const c=CATS[cur.k]||CATS.multi;
  $('#view').innerHTML=btns+tableCard('Players — '+cur.lbl,
    c.sub+' Click any column to sort; players with very few maps are hidden.',
    c.cols,DATA.players.slice(),{i:c.sort,dir:-1});
}

/* ---- Records ---- */
function vRecords(){
  const R=DATA.records;
  const blocks=[
    ['Most multikills in a map (2K+)','','most_mk_map',v=>v+' MK'],
    ['Most clutches in a map','','most_clutch_map',v=>v+' clutch'],
    ['Biggest clutches (1v4 / 1v5)','','big_clutches',v=>'1v'+v],
    ['Most plants in a map','','most_plants_map',v=>v+' PL'],
    ['Most defuses in a map','','most_defuses_map',v=>v+' DE'],
    ['Highest econ rating in a map','','top_econ_map',v=>v]
  ];
  const rec=(title,note,key,vf)=>{
    const list=(R[key]||[]);
    const items=list.map((x,i)=>`<li><span class="ri">${i+1}</span><div class="rbody">
        <div class="rrow"><span class="nm">${esc(x.player)}</span> ${orgLink(x.org)}<span class="rv">${esc(vf(x.value))}</span></div>
        <div class="rmeta">${esc(x.map_name||'')} &middot; ${esc(x.event||'')} ${x.url?`&middot; <a class="ext" href="${esc(x.url)}" target="_blank" rel="noopener">VLR&nbsp;&rarr;</a>`:''}</div>
      </div></li>`).join('') || '<li class="rmeta">No data yet.</li>';
    return `<div class="rec card"><h3>${esc(title)}</h3>${note?`<div class="sub" style="margin-top:-4px">${esc(note)}</div>`:''}<ol>${items}</ol></div>`;
  };
  $('#view').innerHTML=`<div class="rec-grid">${blocks.map(b=>rec(b[0],b[1],b[2],b[3])).join('')}</div>`;
}

/* ---- wiring ---- */
function setTab(name){
  if(!name) return;
  const t=document.querySelector('.tab[data-t="'+name+'"]'); if(!t) return;
  document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active')); t.classList.add('active');
  TAB=name; if(history.replaceState) history.replaceState(null,'','#'+name);
  if(DATA) render();
}
$('#tabs').addEventListener('click',e=>{ const t=e.target.closest('.tab'); if(t) setTab(t.dataset.t); });
$('#view').addEventListener('click',e=>{
  const pc=e.target.closest('.pc'); if(pc){ PCAT=pc.dataset.pc; vPlayers(); }
});
$('#view').addEventListener('change',e=>{
  if(e.target.id==='mmMapSel'){ MMAP=e.target.value; renderAgentTables(); }
});
function refreshFilters(){
  const cur={year:$('#fYear').value, region:$('#fRegion').value, split:$('#fSplit').value};
  const IDX={year:0,region:1,split:2};
  const dims=[['#fYear','year',FILTERMETA.years,'',true],['#fRegion','region',FILTERMETA.regions,'All',false],['#fSplit','split',FILTERMETA.splits,'All',false]];
  for(const [sel,dim,full,allLbl,noAll] of dims){
    const valid=new Set();
    for(const c of FILTERMETA.combos){
      let ok=true;
      for(const k of ['year','region','split']){ if(k!==dim && cur[k] && c[IDX[k]]!==cur[k]) ok=false; }
      if(ok) valid.add(c[IDX[dim]]);
    }
    // splits are ordered chronologically within the selected year (most recent
    // year is resolved first, since we process the year dim before split)
    let base=full;
    if(dim==='split'){ const so=FILTERMETA.split_order&&FILTERMETA.split_order[cur.year];
      if(so) base=so.concat(full.filter(v=>!so.includes(v))); }
    const opts=base.filter(v=>valid.has(v));
    let val=cur[dim];
    if(noAll){                                  // Year: always a specific year (most recent)
      if(!val || !valid.has(val)) val=opts[0]||'';
      $(sel).innerHTML=opts.map(v=>`<option value="${esc(v)}"${v===val?' selected':''}>${esc(v)}</option>`).join('');
    } else {
      if(val && !valid.has(val)) val='';
      $(sel).innerHTML=`<option value="">${allLbl}</option>`+opts.map(v=>`<option value="${esc(v)}"${v===val?' selected':''}>${esc(v)}</option>`).join('');
    }
    $(sel).value=val;
    cur[dim]=val;   // later dims (e.g. split order) see the resolved year
  }
}
function onFilterChange(){ refreshFilters(); load(); }
['#fYear','#fRegion','#fSplit'].forEach(s=>$(s).addEventListener('change',onFilterChange));

// Team card popup — clicking any /team/<org> link opens the team profile as an
// X-able card overlay (iframe), exactly like the home page. Self-contained here
// because alpha-nav.js's app-wide modal doesn't initialize on this page.
(function(){
  window.__teamModalSetup=true;   // stop alpha-nav from adding a second modal if it later runs
  const ov=$('#tmodal'), frame=ov.querySelector('.tm-frame'), load=ov.querySelector('.tm-load'), card=ov.querySelector('.tm-card');
  frame.addEventListener('load',()=>{ if(frame.src && frame.src.indexOf('about:blank')<0) load.style.display='none'; });
  window.addEventListener('message',e=>{ if(e.source!==frame.contentWindow) return;
    if(e.data && typeof e.data==='object' && typeof e.data.__teamH==='number'){
      card.style.height=Math.min(e.data.__teamH, Math.round(window.innerHeight*0.92))+'px'; } });
  function openTeam(org){ card.style.height='70vh'; load.style.display='flex'; frame.src='/team/'+encodeURIComponent(org); ov.classList.add('on'); document.documentElement.style.overflow='hidden'; }
  function closeTeam(){ ov.classList.remove('on'); frame.src='about:blank'; document.documentElement.style.overflow=''; }
  ov.addEventListener('click',e=>{ if(e.target===ov || (e.target.closest && e.target.closest('.tm-x'))) closeTeam(); });
  document.addEventListener('keydown',e=>{ if(e.key==='Escape' && ov.classList.contains('on')) closeTeam(); });
  document.addEventListener('click',e=>{
    if(e.metaKey||e.ctrlKey||e.shiftKey||e.altKey||e.button!==0) return;
    const a=e.target.closest && e.target.closest('a[href^="/team/"]'); if(!a || a.target==='_blank') return;
    const m=(a.getAttribute('href')||'').match(/^\/team\/([^?#]+)/);
    if(m){ e.preventDefault(); e.stopPropagation(); openTeam(decodeURIComponent(m[1])); }
  },true);
})();

let h0=(location.hash||'').replace('#','');
if(h0.includes('/')){ const parts=h0.split('/'); if(parts[0]==='deep'){ h0='players'; PCAT=parts[1]||'save'; } else { h0=parts[0]; if(parts[0]==='players') PCAT=parts[1]; } }
if(!VIEWS[h0]) h0='maps';   // old #overview / #comps (and empty) → Maps & Agents
TAB=h0;
load().then(()=>setTab(TAB)).catch(e=>{ $('#view').innerHTML='<div class="loading">Failed to load stats.</div>'; });
</script>
</body>
</html>"""
