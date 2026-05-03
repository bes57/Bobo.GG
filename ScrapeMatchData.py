"""
Scrape per-map and per-series match stats for all VCT franchised events.
Saves to:
  data/maps/{event_id}.csv   — one row per player per map
  data/series/{event_id}.csv — one row per player per series (all maps combined)

Run once; already-scraped events are skipped automatically.
Usage: python ScrapeMatchData.py
"""

import os
import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from MoreTestingMaybeFiles import ALL_EVENTS

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

BASE_DIR   = os.path.dirname(__file__)
DATA_DIR   = os.path.join(BASE_DIR, "data")
MAPS_DIR   = os.path.join(DATA_DIR, "maps")
SERIES_DIR = os.path.join(DATA_DIR, "series")
LIVE_EVENT_ID = "2026_stage1"

os.makedirs(MAPS_DIR,   exist_ok=True)
os.makedirs(SERIES_DIR, exist_ok=True)


def build_region_map():
    """ProfileURL → Region from all existing event CSVs."""
    region_map = {}
    for event in ALL_EVENTS:
        csv_path = os.path.join(DATA_DIR, f"{event['id']}.csv")
        if not os.path.exists(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path, usecols=["ProfileURL", "Region"])
            for _, row in df.iterrows():
                url = str(row.get("ProfileURL", ""))
                reg = str(row.get("Region", ""))
                if url and reg and url not in region_map:
                    region_map[url] = reg
        except Exception:
            pass
    return region_map


def _extract_id_slug(url):
    m = re.search(r'/event/stats/(\d+)/([^/?]+)', url)
    return (m.group(1), m.group(2)) if m else (None, None)


def get_match_urls(numeric_id, slug):
    url = f"https://www.vlr.gg/event/matches/{numeric_id}/{slug}/"
    try:
        res = requests.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(res.text, "html.parser")
    except Exception as e:
        print(f"  Failed matches page {url}: {e}")
        return []

    urls = []
    for a in soup.select("a.wf-module-item.match-item"):
        href = a.get("href", "")
        status_el = a.select_one(".ml-status")
        if status_el and status_el.get_text(strip=True).lower() != "completed":
            continue
        if re.match(r'^/\d+/', href):
            full = "https://www.vlr.gg" + href
            if full not in urls:
                urls.append(full)
    return urls


def _stat(td):
    """Get the overall (mod-both) stat value from a match stats cell."""
    span = td.find("span", class_=lambda c: c and "mod-both" in c.split())
    if span:
        return span.get_text(strip=True)
    return td.get_text(strip=True)


def scrape_match(match_url, region_tag, region_map):
    """
    Returns (map_rows, series_rows).
    region_tag: fixed region string for regional events; None for international.
    """
    try:
        res = requests.get(match_url, headers=HEADERS, timeout=20)
        soup = BeautifulSoup(res.text, "html.parser")
    except Exception as e:
        print(f"    fetch failed: {e}")
        return [], []

    fmt_el  = soup.select_one(".match-header-vs-note")
    fmt_raw = fmt_el.get_text(strip=True).lower() if fmt_el else ""
    if "5" in fmt_raw:
        series_fmt = "bo5"
    elif "3" in fmt_raw:
        series_fmt = "bo3"
    else:
        series_fmt = "bo1"

    m = re.search(r'vlr\.gg/(\d+)/', match_url)
    match_id = m.group(1) if m else ""

    map_rows, series_rows = [], []

    for game_div in soup.select("div.vm-stats-game"):
        game_id = game_div.get("data-game-id", "")
        is_all  = (game_id == "all")

        map_name = ""
        if not is_all:
            hdr = game_div.select_one(".vm-stats-game-header .map")
            if hdr:
                first_div = hdr.find("div")
                if first_div:
                    map_name = first_div.get_text(strip=True)

        for table in game_div.select("table.wf-table-inset.mod-overview"):
            tbody = table.find("tbody")
            if not tbody:
                continue
            for tr in tbody.find_all("tr"):
                tds = tr.find_all("td")
                if len(tds) < 10:
                    continue

                ptd        = tds[0]
                pname      = ptd.select_one(".text-of")
                porg       = ptd.select_one(".ge-text-light")
                pa         = ptd.find("a", href=True)
                player     = pname.get_text(strip=True) if pname else ""
                org        = porg.get_text(strip=True) if porg else ""
                profile_url = ("https://www.vlr.gg" + pa["href"]) if pa else ""
                if not player:
                    continue

                region = region_tag if region_tag else region_map.get(profile_url, "")

                rating = _stat(tds[2])
                acs    = _stat(tds[3])
                k      = _stat(tds[4])
                d      = _stat(tds[5])
                a_val  = _stat(tds[6])
                kast   = _stat(tds[8])  if len(tds) > 8  else ""
                adr    = _stat(tds[9])  if len(tds) > 9  else ""
                hs     = _stat(tds[10]) if len(tds) > 10 else ""
                fk     = _stat(tds[11]) if len(tds) > 11 else ""
                fd     = _stat(tds[12]) if len(tds) > 12 else ""

                try:
                    k_int = int(k)
                    d_int = int(d)
                    kd = round(k_int / d_int, 2) if d_int > 0 else float(k_int)
                except Exception:
                    kd = ""

                row = {
                    "Player":       player,
                    "Org":          org,
                    "ProfileURL":   profile_url,
                    "Region":       region,
                    "MatchID":      match_id,
                    "MapNum":       game_id,
                    "MapName":      map_name,
                    "SeriesFormat": series_fmt,
                    "R2.0":         rating,
                    "ACS":          acs,
                    "K":            k,
                    "D":            d,
                    "A":            a_val,
                    "K:D":          kd,
                    "KAST":         kast,
                    "ADR":          adr,
                    "HS%":          hs,
                    "FK":           fk,
                    "FD":           fd,
                }

                if is_all:
                    series_rows.append(row)
                else:
                    map_rows.append(row)

    return map_rows, series_rows


def scrape_event(event, region_map):
    event_id   = event["id"]
    maps_path   = os.path.join(MAPS_DIR,   f"{event_id}.csv")
    series_path = os.path.join(SERIES_DIR, f"{event_id}.csv")

    if os.path.exists(maps_path) and os.path.exists(series_path):
        print(f"  Skipping {event_id} (already done)")
        return

    print(f"\n=== {event['label']} ===")
    all_map_rows, all_series_rows = [], []

    for region_name, stats_url in event["regions"].items():
        numeric_id, slug = _extract_id_slug(stats_url)
        if not numeric_id:
            continue

        region_tag = None if region_name == "International" else region_name

        print(f"  {region_name}: getting match list...")
        match_urls = get_match_urls(numeric_id, slug)
        print(f"    {len(match_urls)} matches found")
        time.sleep(1)

        for i, murl in enumerate(match_urls, 1):
            print(f"    [{i}/{len(match_urls)}] {murl}")
            mrows, srows = scrape_match(murl, region_tag, region_map)
            all_map_rows.extend(mrows)
            all_series_rows.extend(srows)
            time.sleep(0.75)

    if all_map_rows:
        pd.DataFrame(all_map_rows).to_csv(maps_path, index=False)
        print(f"  Saved {len(all_map_rows)} map rows → {maps_path}")
    else:
        print(f"  No map rows for {event_id}")

    if all_series_rows:
        pd.DataFrame(all_series_rows).to_csv(series_path, index=False)
        print(f"  Saved {len(all_series_rows)} series rows → {series_path}")
    else:
        print(f"  No series rows for {event_id}")


if __name__ == "__main__":
    region_map = build_region_map()
    print(f"Region map: {len(region_map)} players")

    for event in ALL_EVENTS:
        if event["id"] == LIVE_EVENT_ID:
            print(f"\nSkipping live event: {event['label']}")
            continue
        scrape_event(event, region_map)

    print("\nAll done!")
