"""agent:corpus — enumeration step 1: full VLR /events archive sweep.

Fetches every archive page sequentially through vlr_client.fetch, parses all
event-item rows (id, name, status, prize, dates text, region flag, section),
infers the year of completed events by month-rollover walking (reverse-chron
list), and stops once an entire page's completed events fall before 2021
(plus writes everything seen). Output: scratchpad archive_events.json.
"""
import json
import os
import re
import sys
import time

ROOT = "/Users/benny_es1/PythonTest"
sys.path.insert(0, ROOT)
from bs4 import BeautifulSoup  # noqa: E402
from scrapers.enriched.vlr_client import fetch  # noqa: E402

CACHE = ("/private/tmp/claude-501/-Users-benny-es1-PythonTest/"
         "1a0d507c-1768-436d-a3aa-05cca76a816a/scratchpad")
OUT = os.path.join(CACHE, "archive_events.json")

MONTHS = {m: i + 1 for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}


def parse_dates_txt(txt):
    """'Jul 9—Aug 23' / 'Jun 7—29' / 'May 30' / 'TBD' -> (start_m, start_d, end_m, end_d)."""
    txt = txt.strip()
    if not txt or txt.upper() == "TBD":
        return None
    parts = re.split(r"[—–-]", txt)
    m = re.match(r"([A-Za-z]{3})\w*\s+(\d+)", parts[0].strip())
    if not m:
        return None
    sm, sd = MONTHS.get(m.group(1)[:3].title()), int(m.group(2))
    em, ed = sm, sd
    if len(parts) > 1:
        p2 = parts[1].strip()
        m2 = re.match(r"([A-Za-z]{3})\w*\s+(\d+)", p2)
        if m2:
            em, ed = MONTHS.get(m2.group(1)[:3].title()), int(m2.group(2))
        elif re.match(r"^\d+$", p2):
            ed = int(p2)
    return (sm, sd, em, ed)


def parse_page(html):
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    section = ""
    # walk labels + event items in document order
    for el in soup.select(".wf-label.mod-large, a.event-item"):
        classes = el.get("class") or []
        if "wf-label" in classes:
            section = el.get_text(strip=True).lower()
            continue
        href = el.get("href", "")
        m = re.match(r"/event/(\d+)/([^/?#]+)", href)
        if not m:
            continue
        title_el = el.select_one(".event-item-title")
        status_el = el.select_one(".event-item-desc-item-status")
        prize_el = el.select_one(".event-item-desc-item.mod-prize")
        dates_el = el.select_one(".event-item-desc-item.mod-dates")
        flag_el = el.select_one(".event-item-desc-item.mod-location i.flag")
        prize = ""
        if prize_el:
            prize = prize_el.get_text(" ", strip=True).replace("Prize Pool", "").strip()
        dates = ""
        if dates_el:
            dates = dates_el.get_text(" ", strip=True).replace("Dates", "").strip()
        flag = ""
        if flag_el:
            fc = [c for c in (flag_el.get("class") or []) if c.startswith("mod-")]
            flag = fc[0][4:] if fc else ""
        rows.append({
            "vlr_event_id": m.group(1),
            "slug": m.group(2),
            "name": title_el.get_text(strip=True) if title_el else "",
            "section": section,
            "status": status_el.get_text(strip=True).lower() if status_el else "",
            "prize": prize,
            "dates_txt": dates,
            "flag": flag,
        })
    return rows


def main():
    all_rows = []
    page = 1
    max_pages = 80
    stop_year_below = 2021
    year = None
    prev_em = None
    consecutive_old_pages = 0
    while page <= max_pages:
        url = f"https://www.vlr.gg/events/?page={page}"
        html = fetch(url)
        rows = parse_page(html)
        completed = [r for r in rows if r["section"] == "completed events"]
        page_years = []
        for r in rows:
            r["page"] = page
            if r["section"] != "completed events":
                r["year_inferred"] = None
                continue
            pd_ = parse_dates_txt(r["dates_txt"])
            if pd_ is None:
                r["year_inferred"] = year
                continue
            sm, sd, em, ed = pd_
            if year is None:
                year = 2026  # first completed item on page 1 is current year
            elif prev_em is not None and em > prev_em + 1:
                # months increase going down a reverse-chron list => crossed a year
                year -= 1
            prev_em = em
            r["year_inferred"] = year
            page_years.append(year)
        all_rows.extend(rows)
        with open(OUT, "w") as f:
            json.dump(all_rows, f, indent=1)
        ymin = min(page_years) if page_years else None
        ymax = max(page_years) if page_years else None
        print(f"page {page}: {len(rows)} rows ({len(completed)} completed), "
              f"inferred years {ymin}..{ymax}", flush=True)
        if page_years and max(page_years) < stop_year_below:
            consecutive_old_pages += 1
            if consecutive_old_pages >= 2:  # one confirmation page past cutoff
                print(f"stop: {consecutive_old_pages} consecutive pages fully "
                      f"before {stop_year_below}")
                break
        else:
            consecutive_old_pages = 0
        page += 1
        time.sleep(1.1)
    print(f"TOTAL rows: {len(all_rows)} -> {OUT}")


if __name__ == "__main__":
    main()
