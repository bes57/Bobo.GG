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
import json
import requests
import pandas as pd
from bs4 import BeautifulSoup

# Cloudflare bypass — same chain RefreshLiveData uses. Without curl_cffi's
# JA3 impersonation, datacenter IPs (Render in particular) get 403'd by
# Cloudflare every time.
try:
    from curl_cffi import requests as cffi_requests  # type: ignore
    _CFFI_OK = True
except Exception:
    _CFFI_OK = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUT_PATH = os.path.join(DATA_DIR, "map_vetos.csv")
# Matches whose VLR page has no veto note (old matches, lobby remakes, etc.)
# get tombstoned here so we don't re-fetch them on every refresh. Without this
# every page-load refresh fires off N stale requests at 1s each.
TOMBSTONE_PATH = os.path.join(DATA_DIR, "map_vetos_missing.json")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

DELAY = 0.25  # seconds between requests — matches RefreshLiveData's tuned cadence
              # (was 1.0s; Cloudflare still happy with sub-second sequential reqs)


def _fetch_html(url, timeout=12):
    """Try curl_cffi (multiple Chrome JA3 fingerprints) then fall back to
    plain requests. Returns response text or None."""
    if _CFFI_OK:
        for imp in ("chrome131", "chrome120", "chrome"):
            try:
                r = cffi_requests.get(url, headers=HEADERS, timeout=timeout,
                                      impersonate=imp, allow_redirects=True)
                if r.status_code == 200 and r.text:
                    return r.text
            except Exception:
                continue
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        r.raise_for_status()
        return r.text
    except Exception:
        return None


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
    html = _fetch_html(url)
    if html is None:
        print(f"  [error] {match_id}: fetch failed")
        return None

    soup = BeautifulSoup(html, "html.parser")
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

    # Load already-done IDs (have at least one veto row in the CSV)
    done_ids = set()
    if os.path.exists(OUT_PATH):
        existing = pd.read_csv(OUT_PATH)
        done_ids = set(existing["MatchID"].unique())
        print(f"Already scraped: {len(done_ids)} matches — skipping")

    # Load tombstoned IDs (matches we've fetched but that have no veto note —
    # mostly old matches with no '.match-header-note'). Without this, every
    # refresh re-fetches all of them at 1s each.
    missing_ids = set()
    if os.path.exists(TOMBSTONE_PATH):
        try:
            with open(TOMBSTONE_PATH) as _f:
                missing_ids = {int(k) for k in (json.load(_f) or {}).keys()}
            print(f"Tombstoned (no veto on VLR): {len(missing_ids)} matches — skipping")
        except Exception:
            missing_ids = set()
    missing_now = {}  # newly discovered no-veto matches this run

    todo = [m for m in all_ids if m not in done_ids and m not in missing_ids]
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
            print("no veto note — tombstoning")
            missing_now[str(int(match_id))] = time.strftime("%Y-%m-%d")
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

    # Persist tombstones from this run (merge with any pre-existing)
    if missing_now:
        try:
            existing_tomb = {}
            if os.path.exists(TOMBSTONE_PATH):
                with open(TOMBSTONE_PATH) as _f:
                    existing_tomb = json.load(_f) or {}
            existing_tomb.update(missing_now)
            with open(TOMBSTONE_PATH, "w") as _f:
                json.dump(existing_tomb, _f, indent=2)
            print(f"\nTombstoned {len(missing_now)} new no-veto matches.")
        except Exception as e:
            print(f"\n[warn] could not write tombstone file: {e}")

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
