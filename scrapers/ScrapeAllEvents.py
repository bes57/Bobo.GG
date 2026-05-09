"""
Run once to pre-scrape all completed VCT events and save them as CSVs.
The live app loads past events from these files instantly instead of scraping.

Usage: python3 ScrapeAllEvents.py
Re-running is safe — already-scraped events are skipped.
"""

import os
import sys
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from MoreTestingMaybeFiles import ALL_EVENTS

LIVE_EVENT_ID = "2026_stage1"
DATA_DIR = os.path.join(ROOT, "data")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

ORG_REGIONS = {
    "TL":"EMEA","FNC":"EMEA","NAVI":"EMEA","VIT":"EMEA",
    "BBL":"EMEA","GX":"EMEA","KC":"EMEA","TH":"EMEA",
    "FUT":"EMEA","GIA":"EMEA","MKOI":"EMEA","WOL":"EMEA",
    "M8":"EMEA","FPX":"EMEA",
    "SEN":"Americas","G2":"Americas","MIBR":"Americas",
    "NRG":"Americas","100T":"Americas","C9":"Americas",
    "EG":"Americas","KRÜ":"Americas","LEV":"Americas",
    "FUR":"Americas","LOUD":"Americas",
    "PRX":"Pacific","DRX":"Pacific","T1":"Pacific",
    "TLN":"Pacific","GEN":"Pacific","DFM":"Pacific",
    "ZETA":"Pacific","RRQ":"Pacific","TS":"Pacific","GE":"Pacific",
    "EDG":"CN","BLG":"CN","KRX":"CN","TE":"CN",
    "DRG":"CN","ASE":"CN","NS":"CN","AG":"CN","XLG":"CN",
}

os.makedirs(DATA_DIR, exist_ok=True)


def scrape_stats(region, url):
    print(f"  Scraping {region} — {url}...")
    try:
        res = requests.get(url, headers=HEADERS, timeout=15)
    except Exception as e:
        print(f"  Request failed: {e}")
        return pd.DataFrame()
    soup = BeautifulSoup(res.text, "html.parser")
    table = soup.find("table")
    if not table:
        print(f"  No table found.")
        return pd.DataFrame()
    raw_headers = [th.get_text(strip=True) for th in table.find_all("th")]
    col_names = ["Player", "Org", "ProfileURL"] + raw_headers[1:]
    rows = []
    for tr in table.find("tbody").find_all("tr"):
        tds = tr.find_all("td")
        if not tds:
            continue
        row = []
        for i, td in enumerate(tds):
            if i == 0:
                lines = [l.strip() for l in td.get_text(separator="\n", strip=True).split("\n") if l.strip()]
                player_name = lines[0] if lines else ""
                org = lines[1] if len(lines) > 1 else ""
                a = td.find("a", href=True)
                profile_url = ("https://www.vlr.gg" + a["href"]) if a else ""
                row.extend([player_name, org, profile_url])
            else:
                row.append(td.get_text(strip=True))
        if row:
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=col_names[:len(rows[0])])
    df.insert(0, "Region", region)
    return df


def scrape_event(event):
    dfs = []
    for region_name, url in event["regions"].items():
        df = scrape_stats(region_name, url)
        if not df.empty:
            dfs.append(df)
        time.sleep(1)
    if not dfs:
        return pd.DataFrame()
    cache = pd.concat(dfs, ignore_index=True)
    if "R2.0" in cache.columns:
        r2 = pd.to_numeric(cache["R2.0"].astype(str).str.replace("%", ""), errors="coerce")
        if r2.notna().any():
            cache = cache[r2.notna() & (r2 > 0)].reset_index(drop=True)
        elif "ACS" in cache.columns:
            acs = pd.to_numeric(cache["ACS"], errors="coerce")
            cache = cache[acs.notna() & (acs > 0)].reset_index(drop=True)
    if list(event["regions"].keys()) == ["International"] and "Org" in cache.columns:
        cache["Region"] = cache["Org"].map(lambda org: ORG_REGIONS.get(org, "International"))
    return cache


def main():
    past_events = [e for e in ALL_EVENTS if e["id"] != LIVE_EVENT_ID]
    print(f"Scraping {len(past_events)} past events (skipping live: {LIVE_EVENT_ID})\n")

    for event in past_events:
        csv_path = os.path.join(DATA_DIR, f"{event['id']}.csv")
        if os.path.exists(csv_path):
            print(f"[SKIP] {event['label']} — already scraped")
            continue

        print(f"[SCRAPING] {event['label']}...")
        df = scrape_event(event)
        if df.empty:
            print(f"  No data — skipping save.\n")
            continue
        df.to_csv(csv_path, index=False)
        print(f"  Saved {len(df)} players to {csv_path}\n")
        time.sleep(1)

    print("Done.")


if __name__ == "__main__":
    main()
