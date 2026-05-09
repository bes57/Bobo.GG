"""
Scrape map pick/ban (veto) sequences for all VCT matches.
Reads match IDs from data/match_results.csv, fetches each VLR match page,
and parses the .match-header-note text.

Output: data/map_vetos.csv
Columns: MatchID, Step, Team, Action, Map

Actions: ban | pick | remains (decider)
Team is empty string for 'remains' entries.

Resumable — already-scraped MatchIDs are skipped.
Usage: python scrapers/ScrapeMapVetos.py
"""

import os
import re
import sys
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUT_PATH = os.path.join(DATA_DIR, "map_vetos.csv")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

DELAY = 1.0  # seconds between requests


def parse_veto_note(text):
    """
    Parse a .match-header-note string into a list of step dicts.
    Example input: "NRG ban Ascent; MKOI ban Split; NRG pick Icebox; Pearl remains"
    Returns: [{'step': 1, 'team': 'NRG', 'action': 'ban', 'map': 'Ascent'}, ...]
    """
    steps = []
    segments = [s.strip() for s in text.split(';') if s.strip()]
    for i, seg in enumerate(segments, start=1):
        # "Map remains" — decider
        remains_m = re.match(r'^(.+?)\s+remains$', seg, re.IGNORECASE)
        if remains_m:
            steps.append({'step': i, 'team': '', 'action': 'remains', 'map': remains_m.group(1).strip()})
            continue
        # "Team ban/pick Map"
        action_m = re.match(r'^(.+?)\s+(ban|pick)\s+(.+)$', seg, re.IGNORECASE)
        if action_m:
            steps.append({
                'step':   i,
                'team':   action_m.group(1).strip(),
                'action': action_m.group(2).lower(),
                'map':    action_m.group(3).strip(),
            })
            continue
        # Unrecognised segment — store raw for debugging
        steps.append({'step': i, 'team': '', 'action': 'unknown', 'map': seg})
    return steps


def scrape_match_veto(match_id):
    """Fetch a VLR match page and return parsed veto steps, or None if unavailable."""
    url = f"https://www.vlr.gg/{match_id}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
    except Exception as e:
        print(f"  [error] {match_id}: {e}")
        return None

    soup = BeautifulSoup(r.text, "html.parser")
    note = soup.select_one(".match-header-note")
    if not note:
        return []  # match exists but has no veto note (e.g. some old matches)

    text = re.sub(r'\s+', ' ', note.get_text()).strip()
    steps = parse_veto_note(text)
    return steps


def main():
    # Load all unique match IDs (use series-level rows)
    mr = pd.read_csv(os.path.join(DATA_DIR, "match_results.csv"))
    all_ids = mr[mr["MapNum"] == "all"]["MatchID"].unique().tolist()
    print(f"Total matches in match_results.csv: {len(all_ids)}")

    # Load already-done IDs
    done_ids = set()
    if os.path.exists(OUT_PATH):
        existing = pd.read_csv(OUT_PATH)
        done_ids = set(existing["MatchID"].unique())
        print(f"Already scraped: {len(done_ids)} matches — skipping")

    todo = [m for m in all_ids if m not in done_ids]
    print(f"Remaining: {len(todo)} matches\n")

    rows = []
    for idx, match_id in enumerate(todo, start=1):
        print(f"[{idx}/{len(todo)}] match {match_id} ...", end=" ", flush=True)
        steps = scrape_match_veto(match_id)
        if steps is None:
            print("fetch error — skipping")
            time.sleep(DELAY)
            continue
        if not steps:
            print("no veto note")
        else:
            for s in steps:
                rows.append({"MatchID": match_id, **s})
            print(f"{len(steps)} steps")

        # Flush to disk every 50 matches so progress is saved
        if idx % 50 == 0 and rows:
            df_new = pd.DataFrame(rows, columns=["MatchID", "step", "team", "action", "map"])
            if os.path.exists(OUT_PATH):
                df_new.to_csv(OUT_PATH, mode="a", header=False, index=False)
            else:
                df_new.to_csv(OUT_PATH, index=False)
            done_ids.update(df_new["MatchID"].unique())
            rows = []
            print(f"  [saved checkpoint]")

        time.sleep(DELAY)

    # Final flush
    if rows:
        df_new = pd.DataFrame(rows, columns=["MatchID", "step", "team", "action", "map"])
        if os.path.exists(OUT_PATH):
            df_new.to_csv(OUT_PATH, mode="a", header=False, index=False)
        else:
            df_new.to_csv(OUT_PATH, index=False)

    total = len(pd.read_csv(OUT_PATH)) if os.path.exists(OUT_PATH) else 0
    print(f"\nDone. {OUT_PATH} has {total} rows.")


if __name__ == "__main__":
    main()
