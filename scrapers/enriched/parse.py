"""Parse VLR.gg match pages into structured enrichment data.

Three inputs (raw HTML strings): the overview page, the economy tab page, and
the performance tab page. Everything keys on the VLR match id (their MatchID)
and per-map game id (their MapNum), so outputs join directly to existing data.
"""
from __future__ import annotations

import re

from bs4 import BeautifulSoup

# round win-condition icon filename -> normalized meaning
_WIN_COND = {
    "elim": "elimination",   # won by killing the enemy team
    "defuse": "defuse",      # defenders defused the spike
    "boom": "detonate",      # attackers' spike detonated
    "time": "time",          # round timer expired (no plant)
}

_PERF_FIELDS = ["2K", "3K", "4K", "5K", "1v1", "1v2", "1v3", "1v4", "1v5", "ECON", "PL", "DE"]


def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")


def _game_containers(soup: BeautifulSoup):
    """Yield (game_id, container) for real maps (skips the 'all' aggregate)."""
    for g in soup.select(".vm-stats-game[data-game-id]"):
        gid = g.get("data-game-id")
        if gid and gid != "all":
            yield gid, g


def _int(text) -> int:
    if text is None:
        return 0
    m = re.search(r"-?\d+", text)
    return int(m.group()) if m else 0


def _player_cells(container):
    """Every player cell in a map container, in DOM order.

    VLR migrated the overview stats from `table.wf-table-inset.mod-overview`
    (rows of `td.mod-player`) to a `div.ovw-table` grid (rows of
    `.ovw-cell.mod-player`) ~mid-2026. Both are matched so the current layout
    and any archived HTML parse identically."""
    return container.select(".ovw-cell.mod-player, td.mod-player")


def _player_cell(el):
    """From an overview player cell -> (name, org, player_id)."""
    if el is None:
        return None, None, None
    name_el = el.select_one(".text-of")
    org_el = el.select_one(".ge-text-light")
    a = el.select_one("a[href^='/player/']")
    pid = None
    if a:
        m = re.search(r"/player/(\d+)", a.get("href", ""))
        pid = m.group(1) if m else None
    name = name_el.get_text(strip=True) if name_el else (el.get_text(strip=True) or None)
    org = org_el.get_text(strip=True) if org_el else None
    return name, org, pid


def _perf_player(cell):
    """Performance/matrix pages render players as a .team block
    (name text + .team-tag org). Returns (name, org)."""
    team = cell.select_one(".team")
    if not team:
        return None, None
    org_el = team.select_one(".team-tag")
    org = org_el.get_text(strip=True) if org_el else None
    full = team.get_text(" ", strip=True)
    if org and full.endswith(org):
        full = full[: -len(org)].strip()
    return (full or None), org


def _perf_val(cell):
    """Integer value of a performance cell, ignoring hover-popup contents."""
    sq = cell.select_one(".stats-sq")
    if sq is not None:
        direct = "".join(t for t in sq.contents if isinstance(t, str))
        return _int(direct)
    txt = cell.get_text(strip=True)
    return _int(txt) if re.fullmatch(r"-?\d+", txt or "") else 0


def _team_orgs(container):
    """(team1_org, team2_org) from the two per-team overview blocks in a map.

    Post-migration each block is a `div.ovw-table`; before it, a
    `table.wf-table-inset.mod-overview`. Either way the first player's tag gives
    the block's org."""
    blocks = (container.select("div.ovw-table")
              or container.select("table.wf-table-inset.mod-overview"))
    orgs = []
    for b in blocks[:2]:
        cell = b.select_one(".ovw-cell.mod-player, td.mod-player")
        _, org, _ = _player_cell(cell)
        orgs.append(org)
    while len(orgs) < 2:
        orgs.append(None)
    return orgs[0], orgs[1]


def _team_names(container):
    names = [n.get_text(strip=True) for n in container.select(".team-name")]
    while len(names) < 2:
        names.append(None)
    return names[0], names[1]


def _map_name(container):
    el = container.select_one(".map div span")
    if not el:
        return None
    # keep the raw "IceboxPICK"-style string (matches their existing MapName)
    return re.sub(r"\s+", "", el.get_text())


# ---------------------------------------------------------------- rounds -----
def parse_rounds(container, team1_org, team2_org):
    rounds = []
    strip = container.select_one(".vlr-rounds")
    if not strip:
        return rounds
    rnum = 0
    for col in strip.select(".vlr-rounds-row-col"):
        win_sq = col.select_one(".rnd-sq.mod-win")
        if not win_sq:
            continue  # team-label column or half/OT spacer
        rnum += 1
        squares = col.select(".rnd-sq")
        idx = squares.index(win_sq)
        winner_org = team1_org if idx == 0 else team2_org
        classes = win_sq.get("class", [])
        side = "attack" if "mod-t" in classes else ("defense" if "mod-ct" in classes else None)
        img = win_sq.select_one("img")
        raw = None
        if img and img.get("src"):
            m = re.search(r"/round/(\w+)\.webp", img["src"])
            raw = m.group(1) if m else None
        rounds.append({
            "round": rnum,
            "winner_org": winner_org,
            "winner_side": side,
            "win_condition": _WIN_COND.get(raw, raw),
            "raw_icon": raw,
            "score_after": col.get("title"),
        })
    return rounds


# --------------------------------------------------------------- economy -----
def _parse_econ_cell(text):
    """'3 (1)' -> {'n': 3, 'won': 1};  '1' -> {'n': 1, 'won': 0}."""
    if text is None:
        return {"n": 0, "won": 0}
    won = re.search(r"\((\d+)\)", text)
    total = re.search(r"-?\d+", text)
    return {"n": int(total.group()) if total else 0, "won": int(won.group(1)) if won else 0}


def parse_economy(container):
    """{org: {pistol_won, eco, semi_eco, semi_buy, full_buy}} for one map."""
    out = {}
    table = None
    for t in container.find_all("table"):
        if "Pistol Won" in t.get_text():
            table = t
            break
    if table is None:
        return out
    for tr in table.select("tr"):
        cells = tr.find_all(["td", "th"])
        if len(cells) < 6:
            continue
        org = cells[0].get_text(strip=True)
        if not org or org == "Pistol Won":
            continue
        out[org] = {
            "pistol_won": _int(cells[1].get_text()),
            "eco": _parse_econ_cell(cells[2].get_text()),
            "semi_eco": _parse_econ_cell(cells[3].get_text()),
            "semi_buy": _parse_econ_cell(cells[4].get_text()),
            "full_buy": _parse_econ_cell(cells[5].get_text()),
        }
    return out


# ------------------------------------------------------------ performance ----
def parse_performance(container):
    """Per-player multikills / clutches / plants / defuses / econ for one map,
    plus a best-effort player-vs-player kill matrix."""
    players = {}
    kill_matrix = []
    matrix_done = False

    for t in container.find_all("table"):
        head = t.find("tr")
        if not head:
            continue
        head_labels = [c.get_text(strip=True) for c in head.find_all(["td", "th"])]

        # ---- multikill / clutch summary table (has 2K + 1v1 columns) --------
        if "2K" in head_labels and "1v1" in head_labels:
            idx = {lbl: i for i, lbl in enumerate(head_labels)}
            for tr in t.select("tr"):
                cells = tr.find_all(["td", "th"])
                if not cells:
                    continue
                name, org = _perf_player(cells[0])
                if not name:
                    continue

                def val(lbl):
                    i = idx.get(lbl)
                    return _perf_val(cells[i]) if i is not None and i < len(cells) else 0

                players[name] = {
                    "player": name, "org": org, "player_id": None,
                    "multikills": {k.lower(): val(k) for k in ("2K", "3K", "4K", "5K")},
                    "clutches": {k: val(k) for k in ("1v1", "1v2", "1v3", "1v4", "1v5")},
                    "econ": val("ECON"), "plants": val("PL"), "defuses": val("DE"),
                }

        # ---- kill matrix: first table whose columns are opponent players ----
        elif not matrix_done:
            col_cells = head.find_all(["td", "th"])[1:]
            col_players = [_perf_player(c)[0] for c in col_cells]
            if sum(1 for p in col_players if p) >= 2:
                for tr in t.select("tr")[1:]:
                    cells = tr.find_all(["td", "th"])
                    if not cells:
                        continue
                    killer, _ = _perf_player(cells[0])
                    if not killer:
                        continue
                    for victim, cell in zip(col_players, cells[1:]):
                        if not victim or victim == killer:
                            continue
                        sqs = cell.select(".stats-sq")
                        kills = _int(sqs[0].get_text()) if sqs else 0
                        deaths = _int(sqs[1].get_text()) if len(sqs) > 1 else 0
                        kill_matrix.append({
                            "killer": killer, "victim": victim,
                            "kills": kills, "deaths": deaths,
                        })
                matrix_done = True

    return list(players.values()), kill_matrix


# --------------------------------------------------------------- combine -----
def parse_match(match_id, overview_html, economy_html, performance_html, *,
                date=None, event=None, enriched_at=None):
    """Combine the three pages into one enriched match record."""
    ov = _soup(overview_html)
    ec = _soup(economy_html)
    pf = _soup(performance_html)

    ec_by_gid = {gid: g for gid, g in _game_containers(ec)}
    pf_by_gid = {gid: g for gid, g in _game_containers(pf)}

    header_t1 = ov.select_one(".match-header-link.mod-1 .wf-title-med")
    header_t2 = ov.select_one(".match-header-link.mod-2 .wf-title-med")
    event_el = ov.select_one(".match-header-event div div")

    maps = []
    for gid, g in _game_containers(ov):
        t1_org, t2_org = _team_orgs(g)
        t1_name, t2_name = _team_names(g)
        rounds = parse_rounds(g, t1_org, t2_org)
        if not rounds:
            continue  # unplayed / forfeit map — nothing to enrich
        economy = parse_economy(ec_by_gid[gid]) if gid in ec_by_gid else {}
        if gid in pf_by_gid:
            players, matrix = parse_performance(pf_by_gid[gid])
            # player_id only exists on the overview page — join it in by name
            ov_pids = {}
            for pc in _player_cells(g):
                nm, _org, pid = _player_cell(pc)
                if nm and pid:
                    ov_pids[nm] = pid
            for p in players:
                p["player_id"] = ov_pids.get(p["player"])
        else:
            players, matrix = [], []
        score = {}
        for r in rounds:
            if r["winner_org"]:
                score[r["winner_org"]] = score.get(r["winner_org"], 0) + 1
        maps.append({
            "map_num": gid,
            "map_name": _map_name(g),
            "team1_org": t1_org, "team2_org": t2_org,
            "team1_name": t1_name, "team2_name": t2_name,
            "score": score,
            "rounds": rounds,
            "economy": economy,
            "players": players,
            "kill_matrix": matrix,
        })

    return {
        "match_id": str(match_id),
        "url": f"https://www.vlr.gg/{match_id}",
        "event": (event or (event_el.get_text(strip=True) if event_el else None)),
        "date": date,
        "team1": {"org": (maps[0]["team1_org"] if maps else None),
                  "name": header_t1.get_text(strip=True) if header_t1 else None},
        "team2": {"org": (maps[0]["team2_org"] if maps else None),
                  "name": header_t2.get_text(strip=True) if header_t2 else None},
        "n_maps": len(maps),
        "enriched_at": enriched_at,
        "maps": maps,
    }
