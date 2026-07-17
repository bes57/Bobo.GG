#!/usr/bin/env python3
"""Incremental, resumable VLR match-page enrichment orchestrator (Tier 1).

Reads the existing (read-only) match list from data/match_results.csv and dates
from data/match_dates.json, then for each match that hasn't been enriched yet:
fetch the 3 VLR pages, parse, and write data/enriched/vlr/<MatchID>.json.

Fully isolated: never writes outside data/enriched/, never imported by the app.
Re-running only processes matches that are new since last run (dynamic), so it
slots in right after the site's own scrape step.

Usage:
  python -m scrapers.enriched.enrich                 # enrich all new matches
  python -m scrapers.enriched.enrich --limit 25      # cap this run
  python -m scrapers.enriched.enrich --match 314622  # specific match(es)
  python -m scrapers.enriched.enrich --refresh       # re-enrich existing too
  python -m scrapers.enriched.enrich --rebuild-csv   # only rebuild flat CSVs
  python -m scrapers.enriched.enrich --status        # counts, no fetching
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import vlr_client as C
from .parse import parse_match

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
MATCH_RESULTS = DATA / "match_results.csv"
MATCH_DATES = DATA / "match_dates.json"

OUT_DIR = DATA / "enriched"
VLR_DIR = OUT_DIR / "vlr"
INDEX_PATH = VLR_DIR / "_index.json"
TOMB_PATH = VLR_DIR / "_tombstones.json"

SLEEP_BETWEEN_PAGES = 0.35
SLEEP_BETWEEN_MATCHES = 0.7
CHECKPOINT_EVERY = 20


# --------------------------------------------------------------- helpers -----
def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_json(path: Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return default
    return default


def _save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
    tmp.replace(path)


def load_match_list():
    """Distinct MatchIDs from the existing results file, newest-first by date."""
    dates = _load_json(MATCH_DATES, {})
    seen = {}
    with MATCH_RESULTS.open() as f:
        for row in csv.DictReader(f):
            mid = (row.get("MatchID") or "").strip()
            if mid and mid not in seen:
                seen[mid] = {"match_id": mid, "date": dates.get(mid)}
    matches = list(seen.values())
    matches.sort(key=lambda m: (m["date"] or "0000-00-00"), reverse=True)
    return matches


# --------------------------------------------------------------- enrich ------
def enrich_one(match_id: str, date=None):
    """Fetch + parse one match. Returns the record (>=1 map) or None (tombstone)."""
    urls = C.match_urls(match_id)
    html = {}
    for i, (name, url) in enumerate(urls.items()):
        html[name] = C.fetch(url)
        if i < len(urls) - 1:
            time.sleep(SLEEP_BETWEEN_PAGES)
    rec = parse_match(
        match_id, html["overview"], html["economy"], html["performance"],
        date=date, enriched_at=_now(),
    )
    return rec if rec["n_maps"] > 0 else None


def run(limit=None, only=None, refresh=False, event=None):
    VLR_DIR.mkdir(parents=True, exist_ok=True)
    index = _load_json(INDEX_PATH, {})
    tombstones = _load_json(TOMB_PATH, {})

    if only:
        dates = _load_json(MATCH_DATES, {})
        todo = [{"match_id": str(m), "date": dates.get(str(m))} for m in only]
    else:
        todo = load_match_list()
        if not refresh:
            todo = [m for m in todo
                    if not (VLR_DIR / f"{m['match_id']}.json").exists()
                    and m["match_id"] not in tombstones]

    if limit:
        todo = todo[:limit]

    total = len(todo)
    print(f"[enrich] {total} match(es) to process "
          f"(refresh={refresh}{', only='+str(only) if only else ''})")

    done = fail = tomb = 0
    for n, m in enumerate(todo, 1):
        mid = m["match_id"]
        try:
            rec = enrich_one(mid, date=m["date"])
            if rec is None:
                tombstones[mid] = {"reason": "no round data", "at": _now()}
                tomb += 1
                print(f"  [{n}/{total}] {mid}  · tombstoned (no rounds)")
            else:
                _save_json(VLR_DIR / f"{mid}.json", rec)
                index[mid] = {
                    "date": rec["date"], "event": rec["event"],
                    "team1": rec["team1"]["org"], "team2": rec["team2"]["org"],
                    "n_maps": rec["n_maps"], "enriched_at": rec["enriched_at"],
                }
                done += 1
                print(f"  [{n}/{total}] {mid}  · {rec['n_maps']} map(s) "
                      f"{rec['team1']['org']} vs {rec['team2']['org']}  ({rec['event']})")
        except KeyboardInterrupt:
            print("\n[enrich] interrupted — saving progress…")
            break
        except Exception as e:  # noqa: BLE001
            fail += 1
            print(f"  [{n}/{total}] {mid}  · FAILED: {type(e).__name__}: {e}")
        if n % CHECKPOINT_EVERY == 0:
            _save_json(INDEX_PATH, index)
            _save_json(TOMB_PATH, tombstones)
        time.sleep(SLEEP_BETWEEN_MATCHES)

    _save_json(INDEX_PATH, index)
    _save_json(TOMB_PATH, tombstones)
    print(f"[enrich] done: {done} enriched, {tomb} tombstoned, {fail} failed")
    return done, tomb, fail


# ------------------------------------------------------- flat CSV exports ----
def rebuild_csvs():
    """Regenerate analyst-friendly flat CSVs from all per-match JSONs."""
    rounds_rows, econ_rows, player_rows, matrix_rows = [], [], [], []
    files = sorted(p for p in VLR_DIR.glob("*.json") if not p.name.startswith("_"))
    for p in files:
        rec = json.loads(p.read_text())
        mid, date, event = rec["match_id"], rec.get("date"), rec.get("event")
        for mp in rec["maps"]:
            base = {"match_id": mid, "map_num": mp["map_num"],
                    "map_name": mp["map_name"], "date": date, "event": event}
            for r in mp["rounds"]:
                rounds_rows.append({**base, "round": r["round"],
                                    "winner_org": r["winner_org"],
                                    "winner_side": r["winner_side"],
                                    "win_condition": r["win_condition"],
                                    "score_after": r["score_after"]})
            for org, e in mp["economy"].items():
                econ_rows.append({**base, "org": org, "pistol_won": e["pistol_won"],
                                  "eco_n": e["eco"]["n"], "eco_won": e["eco"]["won"],
                                  "semi_eco_n": e["semi_eco"]["n"], "semi_eco_won": e["semi_eco"]["won"],
                                  "semi_buy_n": e["semi_buy"]["n"], "semi_buy_won": e["semi_buy"]["won"],
                                  "full_buy_n": e["full_buy"]["n"], "full_buy_won": e["full_buy"]["won"]})
            for pl in mp["players"]:
                player_rows.append({**base, "player": pl["player"], "org": pl["org"],
                                    "player_id": pl["player_id"],
                                    "mk2": pl["multikills"]["2k"], "mk3": pl["multikills"]["3k"],
                                    "mk4": pl["multikills"]["4k"], "mk5": pl["multikills"]["5k"],
                                    "c1v1": pl["clutches"]["1v1"], "c1v2": pl["clutches"]["1v2"],
                                    "c1v3": pl["clutches"]["1v3"], "c1v4": pl["clutches"]["1v4"],
                                    "c1v5": pl["clutches"]["1v5"], "econ": pl["econ"],
                                    "plants": pl["plants"], "defuses": pl["defuses"]})
            for km in mp["kill_matrix"]:
                matrix_rows.append({"match_id": mid, "map_num": mp["map_num"],
                                    "killer": km["killer"], "victim": km["victim"],
                                    "kills": km["kills"], "deaths": km.get("deaths")})

    def write(path, rows, cols):
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)

    write(OUT_DIR / "round_outcomes.csv", rounds_rows,
          ["match_id", "map_num", "map_name", "round", "winner_org", "winner_side",
           "win_condition", "score_after", "date", "event"])
    write(OUT_DIR / "map_economy.csv", econ_rows,
          ["match_id", "map_num", "map_name", "org", "pistol_won", "eco_n", "eco_won",
           "semi_eco_n", "semi_eco_won", "semi_buy_n", "semi_buy_won",
           "full_buy_n", "full_buy_won", "date", "event"])
    write(OUT_DIR / "player_map_advanced.csv", player_rows,
          ["match_id", "map_num", "map_name", "player", "org", "player_id",
           "mk2", "mk3", "mk4", "mk5", "c1v1", "c1v2", "c1v3", "c1v4", "c1v5",
           "econ", "plants", "defuses", "date", "event"])
    write(OUT_DIR / "kill_matrix.csv", matrix_rows,
          ["match_id", "map_num", "killer", "victim", "kills", "deaths"])
    print(f"[csv] {len(files)} matches -> {len(rounds_rows)} rounds, "
          f"{len(econ_rows)} econ rows, {len(player_rows)} player rows, "
          f"{len(matrix_rows)} matrix rows")


def status():
    matches = load_match_list()
    have = {p.stem for p in VLR_DIR.glob("*.json") if not p.name.startswith("_")}
    tombstones = _load_json(TOMB_PATH, {})
    remaining = [m for m in matches
                 if m["match_id"] not in have and m["match_id"] not in tombstones]
    print(f"[status] total matches in DB: {len(matches)}")
    print(f"         enriched:   {len(have)}")
    print(f"         tombstoned: {len(tombstones)}")
    print(f"         remaining:  {len(remaining)}")


# ----------------------------------------------------------------- cli -------
def main(argv=None):
    ap = argparse.ArgumentParser(description="VLR Tier-1 match enrichment")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--match", nargs="+", default=None, help="specific MatchID(s)")
    ap.add_argument("--refresh", action="store_true", help="re-enrich existing matches")
    ap.add_argument("--rebuild-csv", action="store_true", help="only regenerate flat CSVs")
    ap.add_argument("--status", action="store_true", help="print counts and exit")
    args = ap.parse_args(argv)

    if args.status:
        status()
        return
    if args.rebuild_csv:
        rebuild_csvs()
        return

    run(limit=args.limit, only=args.match, refresh=args.refresh)
    rebuild_csvs()


if __name__ == "__main__":
    sys.exit(main())
