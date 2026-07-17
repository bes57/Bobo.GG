#!/usr/bin/env python3
"""Stream Riot official VCT game files (2023-2024, franchised international),
parse each into deep per-player stats + death points, and write one compact file
per game to data/riot/games/. Raw 15 MB game files are parsed and discarded.

Resumable: skips games already parsed. Excludes 2022 (pre-franchising).

Usage:
  python -m scrapers.riot.pipeline            # process all remaining
  python -m scrapers.riot.pipeline --limit 20
  python -m scrapers.riot.pipeline --status
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import time
from pathlib import Path

from curl_cffi import requests as creq

from .parse_game import parse_game

ROOT = Path(__file__).resolve().parents[2]
RIOT = ROOT / "data" / "riot"
GAMES = RIOT / "games"
ESPORTS = RIOT / "esports"
S3 = "https://vcthackathon-data.s3.amazonaws.com"
LEAGUE = "vct-international"
YEARS = {"2023", "2024"}          # franchised only — no 2022


def _get(url, timeout=90):
    for imp in ("chrome131", "chrome124", "chrome120"):
        try:
            r = creq.get(url, impersonate=imp, timeout=timeout)
            if r.status_code == 200:
                return r.content
        except Exception:
            pass
    return None


def _getgz(url):
    raw = _get(url)
    return json.loads(gzip.decompress(raw)) if raw else None


def load_esports():
    ESPORTS.mkdir(parents=True, exist_ok=True)
    out = {}
    for name in ("players", "teams", "tournaments", "mapping_data_v2"):
        p = ESPORTS / f"{name}.json"
        if p.exists():
            out[name] = json.loads(p.read_text())
        else:
            d = _getgz(f"{S3}/{LEAGUE}/esports-data/{name}.json.gz")
            p.write_text(json.dumps(d))
            out[name] = d
    return out


def _year(name):
    m = re.search(r"20\d\d", name or "")
    return m.group() if m else None


def game_list(es):
    tours = {t["id"]: t for t in es["tournaments"]}
    out = []
    for mp in es["mapping_data_v2"]:
        yr = _year((tours.get(mp["tournamentId"]) or {}).get("name"))
        if yr in YEARS:
            out.append((mp["platformGameId"], yr, mp))
    return out


def run(limit=None):
    GAMES.mkdir(parents=True, exist_ok=True)
    es = load_esports()
    players = {p["id"]: p for p in es["players"]}
    teams = {t["id"]: t for t in es["teams"]}
    tours = {t["id"]: t for t in es["tournaments"]}
    games = game_list(es)

    todo = [(pg, yr, mp) for (pg, yr, mp) in games
            if not (GAMES / f"{pg.replace(':', '_')}.json").exists()]
    if limit:
        todo = todo[:limit]
    print(f"[riot] {len(todo)} of {len(games)} games to parse (2023-2024 international)")

    done = fail = miss = 0
    for n, (pg, yr, mp) in enumerate(todo, 1):
        try:
            game = _getgz(f"{S3}/{LEAGUE}/games/{yr}/{pg}.json.gz")
            if game is None:  # try the other year folder
                for alt in YEARS - {yr}:
                    game = _getgz(f"{S3}/{LEAGUE}/games/{alt}/{pg}.json.gz")
                    if game:
                        break
            if game is None:
                miss += 1
                continue
            rec = parse_game(game, mp, players, teams, tours)
            # slim clutch dicts keys to strings for JSON
            for p in rec["players"].values():
                p["clutch_att"] = {str(k): v for k, v in p["clutch_att"].items()}
                p["clutch_win"] = {str(k): v for k, v in p["clutch_win"].items()}
            (GAMES / f"{pg.replace(':', '_')}.json").write_text(
                json.dumps(rec, ensure_ascii=False, separators=(",", ":")))
            done += 1
            if n % 20 == 0:
                print(f"  [{n}/{len(todo)}] {rec['map']} · {rec['tournament']} · "
                      f"{len(rec['deaths'])} deaths")
        except KeyboardInterrupt:
            print("\n[riot] interrupted"); break
        except Exception as e:  # noqa: BLE001
            fail += 1
            print(f"  [{n}/{len(todo)}] {pg} FAILED {type(e).__name__}: {e}")
        time.sleep(0.1)
    print(f"[riot] done: {done} parsed, {miss} missing, {fail} failed")


def status():
    es = load_esports()
    total = len(game_list(es))
    have = len(list(GAMES.glob("*.json"))) if GAMES.exists() else 0
    print(f"[riot] {have}/{total} games parsed (2023-2024 international)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args(argv)
    status() if args.status else run(limit=args.limit)


if __name__ == "__main__":
    sys.exit(main())
