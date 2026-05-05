"""
Build data/match_results.csv — per-map and per-series win/loss records with scores and match names.
Run once after ScrapeMatchData.py completes. Resumable.
Usage: python BuildMatchResults.py
"""

import os
import re
import sys
import time
import requests
import pandas as pd
from collections import Counter
from bs4 import BeautifulSoup

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

DATA_DIR     = os.path.join(ROOT, "data")
MAPS_DIR     = os.path.join(DATA_DIR, "maps")
SERIES_DIR   = os.path.join(DATA_DIR, "series")
RESULTS_FILE = os.path.join(DATA_DIR, "match_results.csv")


def collect_match_ids():
    ids = set()
    for d in [MAPS_DIR, SERIES_DIR]:
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            if not fname.endswith(".csv"):
                continue
            try:
                df = pd.read_csv(os.path.join(d, fname), usecols=["MatchID"])
                ids.update(df["MatchID"].dropna().astype(str).str.strip().tolist())
            except Exception:
                pass
    return ids


def _first_org(table):
    tbody = table.find("tbody")
    if not tbody:
        return None
    for tr in tbody.find_all("tr"):
        tds = tr.find_all("td")
        if not tds:
            continue
        porg = tds[0].select_one(".ge-text-light")
        if porg:
            return porg.get_text(strip=True)
    return None


def scrape_match_results(match_id):
    url = f"https://www.vlr.gg/{match_id}/"
    try:
        res = requests.get(url, headers=HEADERS, timeout=20)
        soup = BeautifulSoup(res.text, "html.parser")
    except Exception as e:
        print(f"    fetch failed {match_id}: {e}")
        return []

    # Match name from stage/round label e.g. "Playoffs: Grand Final"
    match_name = ""
    name_el = soup.select_one(".match-header-event-series")
    if name_el:
        match_name = re.sub(r"\s+", " ", name_el.get_text()).strip()

    # map_num -> (winner_org, "winner_rounds-loser_rounds")
    per_map = {}

    for game_div in soup.select("div.vm-stats-game"):
        game_id = game_div.get("data-game-id", "")
        if game_id == "all":
            continue

        tables = game_div.select("table.wf-table-inset.mod-overview")
        if len(tables) < 2:
            continue

        org1 = _first_org(tables[0])
        org2 = _first_org(tables[1])
        if not org1 or not org2:
            continue

        score_els = game_div.select(".vm-stats-game-header .team .score")
        if len(score_els) < 2:
            continue
        try:
            s1 = int(score_els[0].get_text(strip=True) or 0)
            s2 = int(score_els[1].get_text(strip=True) or 0)
        except Exception:
            continue

        if s1 > s2:
            per_map[game_id] = (org1, f"{s1}-{s2}")
        elif s2 > s1:
            per_map[game_id] = (org2, f"{s2}-{s1}")

    rows = []
    for map_num, (winner_org, score) in per_map.items():
        rows.append({
            "MatchID":   match_id,
            "MapNum":    map_num,
            "WinnerOrg": winner_org,
            "Score":     score,
            "MatchName": match_name,
        })

    if per_map:
        c = Counter(w for w, _ in per_map.values())
        series_winner = c.most_common(1)[0][0]
        winner_maps   = c[series_winner]
        loser_maps    = sum(c.values()) - winner_maps
        rows.append({
            "MatchID":   match_id,
            "MapNum":    "all",
            "WinnerOrg": series_winner,
            "Score":     f"{winner_maps}-{loser_maps}",
            "MatchName": match_name,
        })

    return rows


if __name__ == "__main__":
    done_ids = set()
    if os.path.exists(RESULTS_FILE):
        existing = pd.read_csv(RESULTS_FILE, dtype=str, usecols=["MatchID"])
        done_ids = set(existing["MatchID"].dropna().str.strip().tolist())
        print(f"Already have results for {len(done_ids)} matches.")

    all_ids   = collect_match_ids()
    remaining = sorted(all_ids - done_ids)
    print(f"Total matches: {len(all_ids)}, remaining: {len(remaining)}")

    buffer = []
    for i, mid in enumerate(remaining, 1):
        print(f"  [{i}/{len(remaining)}] {mid}")
        buffer.extend(scrape_match_results(mid))

        if i % 100 == 0 and buffer:
            df_chunk = pd.DataFrame(buffer)
            df_chunk.to_csv(RESULTS_FILE, mode="a", header=not os.path.exists(RESULTS_FILE), index=False)
            buffer = []
            print(f"  Checkpoint saved.")

        time.sleep(0.5)

    if buffer:
        df_chunk = pd.DataFrame(buffer)
        df_chunk.to_csv(RESULTS_FILE, mode="a", header=not os.path.exists(RESULTS_FILE), index=False)

    print("\nDone! match_results.csv updated.")
