#!/usr/bin/env python3
"""Collect per-map team compositions (the 5 agents each team played) from VLR.

Separate from the main enrichment so it can run alongside it without contending
for the same files: reads the match list (read-only) and writes ONLY to
data/enriched/comps/<MatchID>.json. Incremental + resumable.

Agents come from the overview stat table (td.mod-agents), which VLR populates
for every match — including CN, which lacks economy/performance data.

Usage:
  python -m scrapers.enriched.agents            # collect all not-yet-done
  python -m scrapers.enriched.agents --limit 50
  python -m scrapers.enriched.agents --status
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

from bs4 import BeautifulSoup

from . import vlr_client as C
from .enrich import DATA, load_match_list, _is_recent, _save_json, _load_json
from .parse import _game_containers, _map_name, _player_cell

COMPS_DIR = DATA / "enriched" / "comps"
SLEEP = 0.6


def parse_comps(overview_html: str):
    """{map_num: {org: [agent, …]}} from the overview page.

    VLR's ~mid-2026 grid migration moved the agent icon out of its own
    `td.mod-agents` column and into the player cell itself (`.ovw-agents img`),
    so scan from the player cell and accept either shape. Missing it silently
    yields empty comps — which is why the collector defers recent matches rather
    than writing an empty record."""
    soup = BeautifulSoup(overview_html, "html.parser")
    maps = []
    for gid, g in _game_containers(soup):
        comps = {}
        for pcell in g.select(".ovw-cell.mod-player, td.mod-player"):
            agimg = pcell.select_one(".ovw-agents img")
            if agimg is None:
                row = pcell.find_parent("tr")
                agimg = row.select_one("td.mod-agents img") if row else None
            if agimg is None:
                continue
            _name, org, _pid = _player_cell(pcell)
            m = re.search(r"/agents/([^./]+)", agimg.get("src", ""))
            if org and m:
                comps.setdefault(org, []).append(m.group(1))
        if comps:
            maps.append({"map_num": gid, "map_name": _map_name(g),
                         "comps": {o: sorted(a) for o, a in comps.items()}})
    return maps


def collect_one(match_id: str):
    html = C.fetch(f"{C.BASE}/{match_id}")
    maps = parse_comps(html)
    return {"match_id": str(match_id), "maps": maps} if maps else None


def run(limit=None, deadline=None, defer_recent_days=0):
    """Collect comps for matches with no file yet, newest-first.

    deadline           — time.monotonic() value to stop at (None = no limit), so
                         the automated runner can cap how long a pass takes.
    defer_recent_days  — matches this recent with no agent table yet are skipped
                         rather than written as empty, so they get retried."""
    COMPS_DIR.mkdir(parents=True, exist_ok=True)
    todo = [m for m in load_match_list()
            if not (COMPS_DIR / f"{m['match_id']}.json").exists()]
    if limit:
        todo = todo[:limit]
    print(f"[comps] {len(todo)} match(es) to collect")
    done = fail = 0
    for n, m in enumerate(todo, 1):
        mid = m["match_id"]
        if deadline is not None and time.monotonic() >= deadline:
            print(f"[comps] time budget reached — stopping at {n - 1}/{len(todo)}")
            break
        try:
            rec = collect_one(mid)
            if rec:
                _save_json(COMPS_DIR / f"{mid}.json", rec)
                done += 1
                if n % 25 == 0:
                    print(f"  [{n}/{len(todo)}] {mid} · {len(rec['maps'])} maps")
            elif _is_recent(m["date"], defer_recent_days):
                # VLR hadn't published the stat table yet — leave no file so the
                # next pass retries instead of freezing an empty record in place.
                print(f"  [{n}/{len(todo)}] {mid} · no agents yet — will retry")
            else:
                _save_json(COMPS_DIR / f"{mid}.json", {"match_id": mid, "maps": []})
        except KeyboardInterrupt:
            print("\n[comps] interrupted"); break
        except Exception as e:  # noqa: BLE001
            fail += 1
            print(f"  [{n}/{len(todo)}] {mid} · FAILED {type(e).__name__}: {e}")
        time.sleep(SLEEP)
    print(f"[comps] done: {done} collected, {fail} failed")
    return done


def status():
    have = len([p for p in COMPS_DIR.glob("*.json")]) if COMPS_DIR.exists() else 0
    total = len(load_match_list())
    print(f"[comps] {have}/{total} matches collected")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args(argv)
    if args.status:
        status()
    else:
        run(limit=args.limit)


if __name__ == "__main__":
    sys.exit(main())
