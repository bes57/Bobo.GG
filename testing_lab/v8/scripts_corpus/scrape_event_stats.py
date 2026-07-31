"""agent:corpus — build top-level data/<id>.csv for the new events
(ScrapeAllEvents.scrape_stats semantics, but fetched via vlr_client.fetch).
Skips ids whose CSV already exists. Usage: scrape_event_stats.py <id> [<id>...]
"""
import os
import sys
import time

ROOT = "/Users/benny_es1/PythonTest"
sys.path.insert(0, ROOT)
import pandas as pd  # noqa: E402
from bs4 import BeautifulSoup  # noqa: E402
from MoreTestingMaybeFiles import ALL_EVENTS  # noqa: E402
from scrapers.enriched.vlr_client import fetch  # noqa: E402
from scrapers.ScrapeAllEvents import ORG_REGIONS  # noqa: E402

CACHE = ("/private/tmp/claude-501/-Users-benny-es1-PythonTest/"
         "1a0d507c-1768-436d-a3aa-05cca76a816a/scratchpad/html_cache")
DATA_DIR = os.path.join(ROOT, "data")


def scrape_stats(region, url, cache_name):
    p = os.path.join(CACHE, cache_name)
    if os.path.exists(p):
        html = open(p).read()
    else:
        html = fetch(url)
        open(p, "w").write(html)
        time.sleep(1.0)
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if not table:
        print(f"  no stats table: {url}")
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
                player = lines[0] if lines else ""
                org = lines[1] if len(lines) > 1 else ""
                a = td.find("a", href=True)
                purl = ("https://www.vlr.gg" + a["href"]) if a else ""
                row.extend([player, org, purl])
            else:
                row.append(td.get_text(strip=True))
        if row:
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=col_names[:len(rows[0])])
    df.insert(0, "Region", region)
    return df


def main(ids):
    by_id = {e["id"]: e for e in ALL_EVENTS}
    for eid in ids:
        out = os.path.join(DATA_DIR, f"{eid}.csv")
        if os.path.exists(out):
            print(f"SKIP {eid} (exists)")
            continue
        ev = by_id[eid]
        dfs = []
        for region, url in ev["regions"].items():
            safe = url.replace("https://www.vlr.gg/", "").replace("/", "_").replace("?", "_")
            df = scrape_stats(region, url, f"stats_{safe}.html")
            if not df.empty:
                dfs.append(df)
        if not dfs:
            print(f"  NO DATA {eid} — not written")
            continue
        cache = pd.concat(dfs, ignore_index=True)
        if "R2.0" in cache.columns:
            r2 = pd.to_numeric(cache["R2.0"].astype(str).str.replace("%", ""), errors="coerce")
            if r2.notna().any():
                cache = cache[r2.notna() & (r2 > 0)].reset_index(drop=True)
        if list(ev["regions"].keys()) == ["International"] and "Org" in cache.columns:
            cache["Region"] = cache["Org"].map(lambda o: ORG_REGIONS.get(o, "International"))
        cache.to_csv(out, index=False)
        print(f"  saved {len(cache)} players -> {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
