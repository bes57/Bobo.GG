"""
ScrapeMatchDates.py
Scrape exact match dates from individual VLR match pages.

Uses the data-utc-ts attribute on .moment-tz-convert elements — the same
timestamp VLR displays in the top-right of each match page.

Output: data/match_dates.json  →  {"660388": "2026-05-09", ...}

Usage:
  python scrapers/ScrapeMatchDates.py            # fetch only new matches
  python scrapers/ScrapeMatchDates.py --force    # re-fetch everything
"""

import os, sys, json, time, argparse
import requests
from bs4 import BeautifulSoup

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import pandas as pd

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}
DATA_DIR = os.path.join(ROOT, "data")
OUT_PATH  = os.path.join(DATA_DIR, "match_dates.json")


def scrape_match_date(match_id):
    """
    Fetch a VLR match page and return the UTC date as YYYY-MM-DD.
    Returns None on any failure.

    VLR encodes the exact UTC timestamp in data-utc-ts on every
    .moment-tz-convert element, e.g. "2026-05-09 13:30:00".
    We grab the first such element (match start time) and strip the time part.
    """
    url = f"https://www.vlr.gg/{match_id}/"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            print(f"  HTTP {r.status_code} for {match_id}")
            return None
        soup = BeautifulSoup(r.text, "html.parser")
        el = soup.find("div", class_="moment-tz-convert", attrs={"data-utc-ts": True})
        if el:
            ts_str = el["data-utc-ts"]  # "2026-05-09 13:30:00"
            return ts_str[:10]          # "2026-05-09"
        # Fallback: parse .match-header-date text
        date_el = soup.select_one(".match-header-date")
        if date_el:
            from datetime import datetime
            import re
            txt = date_el.get_text(separator=" ", strip=True)
            m = re.search(r"(\w+ \d+,? \d{4})", txt)
            if m:
                for fmt in ("%B %d, %Y", "%B %d %Y"):
                    try:
                        return datetime.strptime(m.group(1), fmt).strftime("%Y-%m-%d")
                    except ValueError:
                        pass
    except Exception as e:
        print(f"  Error {match_id}: {e}")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="Re-fetch all matches")
    args = ap.parse_args()

    existing = {}
    if os.path.exists(OUT_PATH) and not args.force:
        with open(OUT_PATH) as f:
            existing = json.load(f)
        print(f"Loaded {len(existing)} cached dates.")

    mr = pd.read_csv(os.path.join(DATA_DIR, "match_results.csv"))
    all_ids = [str(int(m)) for m in mr["MatchID"].unique()]
    to_fetch = [m for m in all_ids if m not in existing]
    print(f"Need to fetch dates for {len(to_fetch)} matches (skipping {len(existing)} cached)...")

    failed = []
    for i, mid in enumerate(to_fetch, 1):
        d = scrape_match_date(mid)
        if d:
            existing[mid] = d
            print(f"  [{i}/{len(to_fetch)}] {mid} → {d}")
        else:
            failed.append(mid)
            print(f"  [{i}/{len(to_fetch)}] {mid} → FAILED")

        # Save every 25 fetches
        if i % 25 == 0:
            with open(OUT_PATH, "w") as f:
                json.dump(existing, f)
            print(f"  (checkpoint saved, {len(failed)} failures so far)")

        time.sleep(0.5)  # respectful rate-limiting

    with open(OUT_PATH, "w") as f:
        json.dump(existing, f, indent=2)

    found = sum(1 for v in existing.values() if v)
    print(f"\nDone. {found}/{len(all_ids)} matches have dates. Failures: {len(failed)}")
    if failed:
        print(f"  Failed IDs: {failed[:20]}{'...' if len(failed) > 20 else ''}")


if __name__ == "__main__":
    main()
