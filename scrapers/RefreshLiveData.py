"""
RefreshLiveData.py — VCT live data refresh pipeline.

On every run:
  1. Check today's date vs last data checkpoint
  2. Scan VLR for new completed Stage 1 matches
  3. Scrape any new matches (per-match progress + name logged)
  4. Rebuild match_results.csv if new matches found
  5. Scrape dates for new match IDs
  6. Rebuild BenPom rating timeline

Writes live progress to /tmp/mhub_refresh_progress.json.
PID-locked so only one instance runs at a time.
"""
import os, sys, json, time, re, subprocess, datetime, requests
from bs4 import BeautifulSoup

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PROGRESS_FILE = "/tmp/mhub_refresh_progress.json"
LOCK_FILE     = "/tmp/mhub_refresh.lock"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

STAGE1_REGIONS = {
    "EMEA":     ("2863", "vct-2026-emea-stage-1"),
    "Americas": ("2860", "vct-2026-americas-stage-1"),
    "Pacific":  ("2775", "vct-2026-pacific-stage-1"),
}

VLR_NAME_TO_ORG = {
    '100 Thieves': '100T', 'BBL Esports': 'BBL', 'Cloud9': 'C9',
    'DetonatioN FocusMe': 'DFM', 'ENVY': 'ENVY', 'Eternal Fire': 'EF',
    'Evil Geniuses': 'EG', 'FNATIC': 'FNC', 'FULL SENSE': 'FS',
    'FURIA': 'FUR', 'FUT Esports': 'FUT', 'G2 Esports': 'G2',
    'GIANTX': 'GX', 'Gen.G': 'GEN', 'Gentle Mates': 'M8',
    'Global Esports': 'GE', 'KRÜ Esports': 'KRÜ', 'Karmine Corp': 'KC',
    'Kiwoom DRX': 'KRX', 'LEVIATÁN': 'LEV', 'LOUD': 'LOUD',
    'MIBR': 'MIBR', 'NRG': 'NRG', 'Natus Vincere': 'NAVI',
    'Nongshim RedForce': 'NS', 'PCIFIC Esports': 'PCF', 'Paper Rex': 'PRX',
    'Rex Regum Qeon': 'RRQ', 'Sentinels': 'SEN', 'T1': 'T1',
    'Team Heretics': 'TH', 'Team Liquid': 'TL', 'Team Secret': 'TS',
    'Team Vitality': 'VIT', 'VARREL': 'VL', 'ZETA DIVISION': 'ZETA',
}

_log_entries = []


def _write(phase, pct, message, extra_log=None):
    if extra_log:
        _log_entries.extend(extra_log if isinstance(extra_log, list) else [extra_log])
    data = {
        "phase":   phase,
        "pct":     pct,
        "message": message,
        "log":     list(_log_entries[-30:]),   # keep last 30 entries
        "ts":      time.time(),
    }
    with open(PROGRESS_FILE, "w") as f:
        json.dump(data, f)
    print(f"  [{pct:3d}%] {message}", flush=True)
    if extra_log:
        for line in (extra_log if isinstance(extra_log, list) else [extra_log]):
            print(f"         {line}", flush=True)


# ── VLR helpers ───────────────────────────────────────────────────────────────

def _get_completed_urls(num_id, slug):
    url = f"https://www.vlr.gg/event/matches/{num_id}/{slug}/"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(r.text, "html.parser")
    except Exception as e:
        print(f"  match list failed {url}: {e}", flush=True)
        return []
    out = []
    for a in soup.select("a.wf-module-item.match-item"):
        href = a.get("href", "")
        status_el = a.select_one(".ml-status")
        if not status_el or status_el.get_text(strip=True).lower() != "completed":
            continue
        if re.match(r"^/\d+/", href):
            full = "https://www.vlr.gg" + href
            if full not in out:
                out.append(full)
    return out


def _match_id_from_url(url):
    m = re.search(r"/(\d+)/", url)
    return m.group(1) if m else None


def _existing_match_ids():
    import pandas as pd
    p = os.path.join(ROOT, "data", "maps", "2026_stage1.csv")
    if not os.path.exists(p):
        return set()
    try:
        df = pd.read_csv(p, usecols=["MatchID"])
        return set(df["MatchID"].dropna().astype(str).tolist())
    except Exception:
        return set()


def _scrape_match_page(url, region_tag):
    """Returns (map_rows, series_rows, match_name_str)."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        soup = BeautifulSoup(r.text, "html.parser")
    except Exception as e:
        print(f"    fetch failed {url}: {e}", flush=True)
        return [], [], "?"

    fmt_el  = soup.select_one(".match-header-vs-note")
    fmt_raw = fmt_el.get_text(strip=True).lower() if fmt_el else ""
    series_fmt = "bo5" if "5" in fmt_raw else ("bo3" if "3" in fmt_raw else "bo1")

    mid = _match_id_from_url(url) or ""

    # Extract team names for display
    teams_el = soup.select(".match-header-link-name .wf-title-med")
    team_a = teams_el[0].get_text(strip=True) if len(teams_el) > 0 else "?"
    team_b = teams_el[1].get_text(strip=True) if len(teams_el) > 1 else "?"

    # Extract score
    scores_el = soup.select(".match-header-vs-score .js-spoiler")
    score_a = scores_el[0].get_text(strip=True) if len(scores_el) > 0 else "?"
    score_b = scores_el[1].get_text(strip=True) if len(scores_el) > 1 else "?"

    display = f"{team_a} {score_a}–{score_b} {team_b}"

    map_rows, series_rows = [], []

    for game_div in soup.select("div.vm-stats-game"):
        game_id = game_div.get("data-game-id", "")
        is_all  = (game_id == "all")
        map_name = ""
        if not is_all:
            hdr = game_div.select_one(".vm-stats-game-header .map")
            if hdr:
                fd = hdr.find("div")
                if fd:
                    map_name = fd.get_text(strip=True)

        for table in game_div.select("table.wf-table-inset.mod-overview"):
            tbody = table.find("tbody")
            if not tbody:
                continue
            for tr in tbody.find_all("tr"):
                tds = tr.find_all("td")
                if len(tds) < 10:
                    continue
                ptd    = tds[0]
                pname  = ptd.select_one(".text-of")
                porg   = ptd.select_one(".ge-text-light")
                pa     = ptd.find("a", href=True)
                player = pname.get_text(strip=True) if pname else ""
                org    = porg.get_text(strip=True)  if porg  else ""
                if not player:
                    continue

                def _s(td):
                    sp = td.find("span", class_=lambda c: c and "mod-both" in c.split())
                    return sp.get_text(strip=True) if sp else td.get_text(strip=True)

                row = {
                    "Player":       player,
                    "Org":          org,
                    "ProfileURL":   ("https://www.vlr.gg" + pa["href"]) if pa else "",
                    "Region":       region_tag,
                    "MatchID":      mid,
                    "MapNum":       game_id,
                    "MapName":      map_name,
                    "SeriesFormat": series_fmt,
                    "R2.0":   _s(tds[2]) if len(tds) > 2  else "",
                    "ACS":    _s(tds[3]) if len(tds) > 3  else "",
                    "K":      _s(tds[4]) if len(tds) > 4  else "",
                    "D":      _s(tds[5]) if len(tds) > 5  else "",
                    "A":      _s(tds[6]) if len(tds) > 6  else "",
                    "KAST":   _s(tds[8]) if len(tds) > 8  else "",
                    "ADR":    _s(tds[9]) if len(tds) > 9  else "",
                    "HS%":    _s(tds[10]) if len(tds) > 10 else "",
                    "FK":     _s(tds[11]) if len(tds) > 11 else "",
                    "FD":     _s(tds[12]) if len(tds) > 12 else "",
                }
                try:
                    k_i, d_i = int(row["K"]), int(row["D"])
                    row["K:D"] = round(k_i / d_i, 2) if d_i else float(k_i)
                except Exception:
                    row["K:D"] = ""

                (series_rows if is_all else map_rows).append(row)

    return map_rows, series_rows, display


# ── Match date scraper ─────────────────────────────────────────────────────────

def _scrape_date(mid):
    url = f"https://www.vlr.gg/{mid}/"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "html.parser")
            el = soup.find("div", class_="moment-tz-convert", attrs={"data-utc-ts": True})
            if el:
                return el["data-utc-ts"][:10]
    except Exception:
        pass
    return None


# ── Upcoming match scraper ────────────────────────────────────────────────────

def _scrape_upcoming(all_urls_by_region):
    """Scrape upcoming (not completed) matches for the next 7 days and save to data/upcoming_matches.json."""
    from datetime import datetime as _dt
    from bs4 import NavigableString
    today = datetime.date.today()
    cutoff = today + datetime.timedelta(days=7)
    upcoming = []

    for region, (num_id, slug) in STAGE1_REGIONS.items():
        url = f"https://www.vlr.gg/event/matches/{num_id}/{slug}/"
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            soup = BeautifulSoup(r.text, "html.parser")
        except Exception:
            continue

        # Date labels (.wf-label.mod-large) and match cards (.wf-card) are
        # direct siblings inside .col.mod-1 — walk children to track current date.
        container = soup.select_one(".col.mod-1") or soup.body
        current_date = None
        for el in container.children:
            if isinstance(el, NavigableString):
                continue
            classes = el.get("class") or []

            if "wf-label" in classes and "mod-large" in classes:
                txt = re.sub(r"(Today|Yesterday)$", "", el.get_text(strip=True)).strip()
                try:
                    current_date = _dt.strptime(txt, "%a, %B %d, %Y").date().isoformat()
                except ValueError:
                    current_date = None
                continue

            if "wf-card" not in classes:
                continue

            for a in el.select("a.wf-module-item.match-item"):
                status_el = a.select_one(".ml-status")
                status = status_el.get_text(strip=True).lower() if status_el else ""
                if status in ("completed", "live"):
                    continue

                # Use data-utc-ts if available, else section date header
                ts_el = a.select_one(".moment-tz-convert")
                match_date = None
                if ts_el and ts_el.get("data-utc-ts"):
                    match_date = ts_el["data-utc-ts"][:10]
                else:
                    match_date = current_date

                if not match_date:
                    continue
                try:
                    md = datetime.date.fromisoformat(match_date)
                except Exception:
                    continue
                if md < today or md > cutoff:
                    continue

                teams = a.select(".match-item-vs-team-name")
                if len(teams) < 2:
                    continue
                team_a = teams[0].get_text(strip=True)
                team_b = teams[1].get_text(strip=True)
                if not team_a or not team_b or "TBD" in team_a or "TBD" in team_b:
                    continue

                org_a = VLR_NAME_TO_ORG.get(team_a, team_a)
                org_b = VLR_NAME_TO_ORG.get(team_b, team_b)
                fmt_el = a.select_one(".match-item-event-series")
                fmt_raw = fmt_el.get_text(strip=True).lower() if fmt_el else ""
                fmt = "bo5" if "5" in fmt_raw else ("bo1" if "1" in fmt_raw else "bo3")
                upcoming.append({
                    "team_a": team_a,
                    "team_b": team_b,
                    "org_a":  org_a,
                    "org_b":  org_b,
                    "date":   match_date,
                    "region": region,
                    "event":  f"Stage 1 — {region}",
                    "format": fmt,
                })
        time.sleep(0.3)

    # Deduplicate and sort
    seen = set()
    deduped = []
    for m in sorted(upcoming, key=lambda x: x["date"]):
        key = f"{m['team_a']}-{m['team_b']}-{m['date']}"
        if key not in seen:
            seen.add(key)
            deduped.append(m)

    out = os.path.join(ROOT, "data", "upcoming_matches.json")
    with open(out, "w") as f:
        json.dump(deduped, f, indent=2)
    print(f"  Upcoming: {len(deduped)} matches in next 7 days", flush=True)


# ── Main pipeline ──────────────────────────────────────────────────────────────

def main():
    import pandas as pd

    today_str = datetime.date.today().isoformat()
    _write("checking", 2, f"Checking VCT data — today is {today_str}…",
           [f"Today: {today_str}"])

    # Read last checkpoint date
    tl_path = os.path.join(ROOT, "data", "rating_timeline.json")
    last_date = "unknown"
    try:
        with open(tl_path) as f:
            tl = json.load(f)
        cps = tl.get("checkpoints", [])
        if cps:
            last_date = cps[-1]["date"]
    except Exception:
        pass
    _write("checking", 5, f"Last ratings checkpoint: {last_date}",
           [f"Last checkpoint: {last_date}"])

    # ── Step 1: Scan VLR for completed matches ─────────────────────────────
    existing_ids = _existing_match_ids()
    _write("checking", 8, f"Scanning VLR — {len(existing_ids)} matches already in database…",
           [f"Database: {len(existing_ids)} Stage 1 matches"])

    all_urls_by_region = {}
    new_urls = []
    total_completed = 0

    for region, (num_id, slug) in STAGE1_REGIONS.items():
        _write("checking", 10 + list(STAGE1_REGIONS).index(region) * 4,
               f"Checking {region} Stage 1…")
        urls = _get_completed_urls(num_id, slug)
        all_urls_by_region[region] = urls
        total_completed += len(urls)
        region_new = [u for u in urls if _match_id_from_url(u) not in existing_ids]
        new_urls.extend([(u, region) for u in region_new])
        _write("checking", 10 + list(STAGE1_REGIONS).index(region) * 4 + 3,
               f"{region}: {len(urls)} completed, {len(region_new)} new",
               [f"✓ {region}: {len(urls)} completed ({len(region_new)} new)"])
        time.sleep(0.5)

    _write("checking", 28,
           f"Scan complete — {total_completed} total, {len(new_urls)} new matches",
           [f"Total completed on VLR: {total_completed}",
            f"New to scrape: {len(new_urls)}"])

    # Always scrape upcoming matches (regardless of new completed matches)
    _scrape_upcoming(all_urls_by_region)

    if not new_urls:
        _write("done", 100,
               f"All data current through {last_date}",
               [f"✓ No new matches found — ratings up to date"])
        print("\nNo new matches. Done.", flush=True)
        return

    # ── Step 2: Scrape new matches ─────────────────────────────────────────
    maps_path   = os.path.join(ROOT, "data", "maps",   "2026_stage1.csv")
    series_path = os.path.join(ROOT, "data", "series", "2026_stage1.csv")

    all_map_rows, all_series_rows = [], []
    total_new = len(new_urls)

    for i, (url, region) in enumerate(new_urls, 1):
        pct = 30 + int(i / total_new * 35)
        _write("scraping", pct, f"Scraping match {i}/{total_new}…")
        mr, sr, display = _scrape_match_page(url, region)
        all_map_rows.extend(mr)
        all_series_rows.extend(sr)
        _write("scraping", pct, f"Scraping {i}/{total_new}…",
               [f"  [{region}] {display}"])
        time.sleep(0.6)

    # Save scraped match data
    if all_map_rows:
        new_df = pd.DataFrame(all_map_rows)
        if os.path.exists(maps_path):
            old_df = pd.read_csv(maps_path)
            combined = pd.concat([old_df, new_df], ignore_index=True).drop_duplicates()
        else:
            combined = new_df
        combined.to_csv(maps_path, index=False)

    if all_series_rows:
        new_df = pd.DataFrame(all_series_rows)
        if os.path.exists(series_path):
            old_df = pd.read_csv(series_path)
            combined = pd.concat([old_df, new_df], ignore_index=True).drop_duplicates()
        else:
            combined = new_df
        combined.to_csv(series_path, index=False)

    # ── Step 3: Rebuild match_results.csv ─────────────────────────────────
    _write("building", 67, "Rebuilding match results…", ["Rebuilding match_results.csv…"])
    subprocess.run(
        [sys.executable, os.path.join(ROOT, "scrapers", "BuildMatchResults.py")],
        cwd=ROOT, check=True, capture_output=True,
    )

    # ── Step 4: Scrape dates for new match IDs ────────────────────────────
    _write("scraping_dates", 72, "Fetching match dates from VLR…",
           ["Fetching exact match dates…"])
    mr_df    = pd.read_csv(os.path.join(ROOT, "data", "match_results.csv"))
    all_ids  = [str(int(m)) for m in mr_df["MatchID"].unique()]
    out_path = os.path.join(ROOT, "data", "match_dates.json")
    existing_dates = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing_dates = json.load(f)
    to_fetch = [m for m in all_ids if m not in existing_dates]
    print(f"  {len(to_fetch)} new match dates to fetch", flush=True)

    for i, mid in enumerate(to_fetch, 1):
        d = _scrape_date(mid)
        if d:
            existing_dates[mid] = d
        pct = 72 + int(i / max(len(to_fetch), 1) * 15)
        _write("scraping_dates", min(pct, 87),
               f"Fetching dates… ({i}/{len(to_fetch)})")
        if i % 10 == 0:
            with open(out_path, "w") as f:
                json.dump(existing_dates, f)
        time.sleep(0.4)

    with open(out_path, "w") as f:
        json.dump(existing_dates, f, indent=2)

    # ── Step 5: Rebuild rating timeline ───────────────────────────────────
    _write("building_ratings", 90, "Rebuilding BenPom ratings…",
           ["Running BenPom model…"])
    subprocess.run(
        [sys.executable, os.path.join(ROOT, "scrapers", "BuildRatingTimeline.py")],
        cwd=ROOT, check=True, capture_output=True,
    )

    # Read new last date
    try:
        with open(tl_path) as f:
            tl2 = json.load(f)
        new_last = tl2["checkpoints"][-1]["date"] if tl2.get("checkpoints") else today_str
    except Exception:
        new_last = today_str

    _scrape_upcoming(all_urls_by_region)
    _write("done", 100,
           f"Ratings updated through {new_last}!",
           [f"✓ Scraped {total_new} new matches",
            f"✓ Ratings rebuilt through {new_last}"])
    print(f"\nDone! {total_new} new matches, ratings through {new_last}", flush=True)


if __name__ == "__main__":
    import fcntl
    lock_fd = open(LOCK_FILE, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Already running — exiting.", flush=True)
        sys.exit(0)
    try:
        main()
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()
