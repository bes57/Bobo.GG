"""Parse one Riot official VCT game file into derived per-player deep stats.

Input: the decoded game-event list + that game's mapping_data_v2 entry + the
esports lookup dicts (players/teams/tournaments). Output: per-player aggregates
for the game plus a list of death points (for heatmaps).

Derives the things VLR cannot: true saves (survived a lost round), last-alive /
clutch situations (attempts AND wins, by 1vN), death locations, first-blood
impact, plants/defuses, and survival→round-win impact.
"""
from __future__ import annotations

import re

# Valorant internal map codename -> display name
MAP_CODENAMES = {
    "Ascent": "Ascent", "Bonsai": "Split", "Duality": "Bind", "Triad": "Haven",
    "Port": "Icebox", "Foxtrot": "Breeze", "Canyon": "Fracture", "Pitt": "Pearl",
    "Jam": "Lotus", "Juliett": "Sunset", "Infinity": "Abyss", "Rook": "Corrode",
}


def _seq(ev):
    return ev.get("metadata", {}).get("sequenceNumber", 0)


def _map_name(guid):
    code = (guid or "").rstrip("/").split("/")[-1]
    return MAP_CODENAMES.get(code, code or "Unknown")


def _year(name):
    m = re.search(r"20\d\d", name or "")
    return m.group() if m else None


def _new_player():
    return {
        "handle": None, "team": None, "rounds": 0, "deaths": 0,
        "save_opp": 0, "saves": 0,                       # survived a lost round
        "fk": 0, "fd": 0, "fb_win": 0,                   # opening duels
        "plants": 0, "defuses": 0,
        "clutch_att": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
        "clutch_win": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
        "last_alive": 0,
        "alive_end": 0, "alive_end_win": 0,              # survival -> win impact
    }


def parse_game(game, mapping, players_by_id, teams_by_id, tournaments_by_id):
    tour_name = (tournaments_by_id.get(mapping["tournamentId"]) or {}).get("name")
    year = _year(tour_name)

    part_map = mapping.get("participantMapping", {})   # ingame pid(str) -> esports playerId
    team_map = mapping.get("teamMapping", {})          # ingame teamId(str) -> esports teamId

    def handle(pid):
        pl = players_by_id.get(part_map.get(str(pid)))
        return pl["handle"] if pl else f"p{pid}"

    def acronym(ingame_team):
        tm = teams_by_id.get(team_map.get(str(ingame_team)))
        return tm["acronym"] if tm else str(ingame_team)

    events = sorted(game, key=_seq)

    # ---- configuration: map, participant->team ----
    map_name, date = "Unknown", None
    part_team = {}          # ingame pid -> ingame teamId
    for ev in events:
        if "configuration" in ev:
            c = ev["configuration"]
            map_name = _map_name((c.get("selectedMap") or {}).get("fallback", {}).get("guid"))
            for t in c.get("teams", []):
                tid = t["teamId"]["value"]
                for p in t.get("playersInTeam", []):
                    part_team[p["value"]] = tid
            break
    for ev in events:
        wt = ev.get("metadata", {}).get("wallTime")
        if wt:
            date = wt[:10]
            break

    players = {}   # esports playerId -> agg
    deaths_out = []

    def P(pid):
        epid = part_map.get(str(pid)) or f"ingame:{pid}"
        pa = players.get(epid)
        if pa is None:
            pa = players[epid] = _new_player()
            pa["handle"] = handle(pid)
            pa["team"] = acronym(part_team.get(pid))
        return pa

    latest_pos = {}   # ingame pid -> (x, y)
    round_ctx = None  # {num, atk, def, deaths:[(pid,killer)]}

    def finalize(winner_team, cause):
        if round_ctx is None:
            return
        # all participants that belong to a team this game
        all_parts = list(part_team.keys())
        died = [d[0] for d in round_ctx["deaths"]]
        died_set = set(died)
        for pid in all_parts:
            pa = P(pid)
            pa["rounds"] += 1
            my_team = part_team.get(pid)
            won = (my_team == winner_team)
            alive_end = pid not in died_set
            if alive_end:
                pa["alive_end"] += 1
                if won:
                    pa["alive_end_win"] += 1
            if not won:                       # save opportunity = a round you lost
                pa["save_opp"] += 1
                if alive_end:
                    pa["saves"] += 1
        # deaths -> counts + positions
        for i, (victim, killer) in enumerate(round_ctx["deaths"]):
            P(victim)["deaths"] += 1
            pos = latest_pos.get(victim)
            deaths_out.append({
                "pid": part_map.get(str(victim)) or f"ingame:{victim}",
                "handle": handle(victim), "map": map_name,
                "x": pos[0] if pos else None, "y": pos[1] if pos else None,
                "round_won": part_team.get(victim) == winner_team,
                "first": i == 0,
            })
        # first blood
        if round_ctx["deaths"]:
            fvic, fkill = round_ctx["deaths"][0]
            P(fvic)["fd"] += 1
            pk = P(fkill)
            pk["fk"] += 1
            if part_team.get(fkill) == winner_team:
                pk["fb_win"] += 1
        # last-alive / clutch: replay deaths tracking alive sets
        teams = {}
        for pid in all_parts:
            teams.setdefault(part_team.get(pid), set()).add(pid)
        alive = {t: set(ps) for t, ps in teams.items()}
        clutched = set()
        for victim, _killer in round_ctx["deaths"]:
            vt = part_team.get(victim)
            alive.get(vt, set()).discard(victim)
            # did a team just drop to exactly 1 alive vs an enemy with >=1?
            for t, aset in alive.items():
                if len(aset) == 1:
                    solo = next(iter(aset))
                    if solo in clutched:
                        continue
                    enemies = sum(len(a) for tt, a in alive.items() if tt != t)
                    if enemies >= 1:
                        clutched.add(solo)
                        n = min(enemies, 5)
                        pa = P(solo)
                        pa["clutch_att"][n] += 1
                        pa["last_alive"] += 1
                        if part_team.get(solo) == winner_team:
                            pa["clutch_win"][n] += 1

    for ev in events:
        if "snapshot" in ev:
            for p in ev["snapshot"].get("players", []):
                pid = (p.get("playerId") or {}).get("value")
                ts = p.get("timeseries")
                if pid is not None and ts:
                    pos = ts[-1].get("position")
                    if pos:
                        latest_pos[pid] = (round(pos["x"]), round(pos["y"]))
        elif "roundStarted" in ev:
            rs = ev["roundStarted"]
            sm = rs.get("spikeMode", {})
            round_ctx = {"num": rs.get("roundNumber"),
                         "atk": (sm.get("attackingTeam") or {}).get("value"),
                         "def": (sm.get("defendingTeam") or {}).get("value"),
                         "deaths": []}
        elif "playerDied" in ev and round_ctx is not None:
            d = ev["playerDied"]
            round_ctx["deaths"].append((d["deceasedId"]["value"], d["killerId"]["value"]))
        elif "spikePlantCompleted" in ev and round_ctx is not None:
            pl = (ev["spikePlantCompleted"].get("playerId") or {}).get("value")
            if pl is not None:
                P(pl)["plants"] += 1
        elif "spikeDefuseCheckpointReached" in ev and round_ctx is not None:
            # Riot emits no "defuse completed"; a checkpoint on a round that ends
            # in DEFUSE is the defuser. Buffer the last checkpoint's player.
            pl = (ev["spikeDefuseCheckpointReached"].get("playerId") or {}).get("value")
            if pl is not None:
                round_ctx["defuser"] = pl
        elif "roundDecided" in ev:
            res = ev["roundDecided"]["result"]
            smr = res.get("spikeModeResult", {})
            if smr.get("cause") == "DEFUSE" and round_ctx and round_ctx.get("defuser") is not None:
                P(round_ctx["defuser"])["defuses"] += 1
            finalize((res.get("winningTeam") or {}).get("value"), smr.get("cause"))
            round_ctx = None

    return {"map": map_name, "tournament": tour_name, "year": year, "date": date,
            "players": players, "deaths": deaths_out}
