#!/usr/bin/env python3
"""Keep the Match Data Explorer's dataset current with newly-scraped matches.

RefreshLiveData scrapes new results into data/maps|series/*.csv, which is what
BenPom, Event Leaderboards and All-Time Highs read — so those pages update
themselves on every refresh. Match Data (/match-data/) instead reads the deeper
per-map enrichment under data/enriched/, which used to be a manual pipeline and
therefore drifted behind. This runner closes that gap: RefreshLiveData spawns it
(detached) once its own work is done, so the enrichment never delays the visible
refresh but still tracks the same match list.

Both passes are incremental (matches already on disk are skipped) and ordered
newest-first, so the matches that just landed are always processed before any
older backlog.

Design notes:
  • Time-budgeted. A large backlog drains across several refreshes instead of
    pinning one process (and its VLR request budget) for a long time.
  • flock'd. Overlapping refreshes can't stack duplicate enrichers on VLR.
  • Recent matches are never tombstoned/blanked. VLR publishes round, economy
    and agent data some time after a match ends; a match scraped in that window
    is left for the next pass rather than being permanently marked "no data".

Usage:
  python scrapers/EnrichNewMatches.py                 # budgeted incremental pass
  python scrapers/EnrichNewMatches.py --budget 900    # allow a longer pass
  python scrapers/EnrichNewMatches.py --limit 25      # cap matches per pass
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scrapers.enriched import agents as comps_mod      # noqa: E402
from scrapers.enriched import enrich as enrich_mod     # noqa: E402

LOCK_FILE = os.path.join(ROOT, "data", "enriched", ".enrich.lock")
STATUS_FILE = os.path.join(ROOT, "data", "enriched", "_last_run.json")

# Total wall clock for one pass, split between the two passes. ~7 min keeps a
# backlog draining at a decent clip while staying far under RefreshLiveData's
# 30-min ceiling and well inside a normal gap between refreshes.
DEFAULT_BUDGET = 420
ENRICH_SHARE = 0.65        # deep enrichment is 3 fetches/match, comps is 1
# A match finished within this window whose page has no data yet is retried on a
# later pass instead of being written off.
DEFER_RECENT_DAYS = 4


def _write_status(payload):
    try:
        os.makedirs(os.path.dirname(STATUS_FILE), exist_ok=True)
        with open(STATUS_FILE, "w") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass


def main(argv=None):
    ap = argparse.ArgumentParser(description="Incremental Match Data enrichment")
    ap.add_argument("--budget", type=float, default=DEFAULT_BUDGET,
                    help="seconds of wall clock for this pass (default %(default)s)")
    ap.add_argument("--limit", type=int, default=None,
                    help="max matches per pass (default: budget-bound only)")
    args = ap.parse_args(argv)

    started = time.monotonic()
    enrich_deadline = started + args.budget * ENRICH_SHARE
    total_deadline = started + args.budget

    enriched = tombed = failed = 0
    try:
        enriched, tombed, failed = enrich_mod.run(
            limit=args.limit, deadline=enrich_deadline,
            defer_recent_days=DEFER_RECENT_DAYS)
    except Exception as e:  # noqa: BLE001 — never let this kill the pass
        print(f"[enrich] pass failed: {type(e).__name__}: {e}", flush=True)

    # Flat analyst CSVs are rebuilt from every JSON, so only pay for it when the
    # JSON set actually grew.
    if enriched:
        try:
            enrich_mod.rebuild_csvs()
        except Exception as e:  # noqa: BLE001
            print(f"[csv] rebuild failed: {type(e).__name__}: {e}", flush=True)

    comps = 0
    try:
        comps = comps_mod.run(limit=args.limit, deadline=total_deadline,
                              defer_recent_days=DEFER_RECENT_DAYS) or 0
    except Exception as e:  # noqa: BLE001
        print(f"[comps] pass failed: {type(e).__name__}: {e}", flush=True)

    _write_status({
        "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seconds": round(time.monotonic() - started, 1),
        "enriched": enriched, "tombstoned": tombed, "failed": failed,
        "comps": comps,
    })
    print(f"[enrich-new] {enriched} enriched, {comps} comps, "
          f"{round(time.monotonic() - started, 1)}s", flush=True)


if __name__ == "__main__":
    os.makedirs(os.path.dirname(LOCK_FILE), exist_ok=True)
    lock_fd = open(LOCK_FILE, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("[enrich-new] already running — exiting.", flush=True)
        sys.exit(0)
    try:
        main()
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()
