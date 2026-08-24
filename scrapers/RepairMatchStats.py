#!/usr/bin/env python3
"""
RepairMatchStats.py — re-scrape matches whose stored stats were captured inside
VLR's round-data publish window, and replace their rows in place.

See MatchStatsIntegrity for the full description of the bug. Short version: a
match scraped before VLR finished processing every map's round data gets empty
rating/ACS/ADR on the unprocessed maps, and — the damaging part — a SERIES row
whose rating/ACS/ADR are only the processed maps' (usually just map 1),
attached to the full series' K/D/A. Those inflated series rows were topping the
all-time Bo3 leaderboard.

Usage
  python3 scrapers/RepairMatchStats.py --dry-run          # report only
  python3 scrapers/RepairMatchStats.py                    # repair everything stale
  python3 scrapers/RepairMatchStats.py --event 2026_stage2
  python3 scrapers/RepairMatchStats.py --match 724621 --event 2026_stage2
  python3 scrapers/RepairMatchStats.py --partial-only     # skip all-blank matches

A re-scrape is only committed when it is strictly more complete than what's on
disk, so a genuinely unrated event can never be made worse by running this.
"""
import os, sys, time, argparse, glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import MatchStatsIntegrity as MSI
from RefreshLiveData import _fetch, _parse_match_html


def _completeness_score(map_rows, series_rows):
    """How many round-derived stat values the scrape actually carries. Used to
    refuse a replacement that would lose data (e.g. VLR briefly 500ing)."""
    return sum(1 for r in list(map_rows) + list(series_rows)
               for c in MSI.ROUND_STATS if not MSI._blank(r.get(c)))


def _region_context(stored_rows):
    """(fallback_region_tag, {profile_url: region}) recovered from the rows we
    already have, so a repaired international match keeps its per-player regions
    instead of being stamped with one tag."""
    per_url, counts = {}, {}
    for r in stored_rows:
        reg = str(r.get("Region", "") or "").strip()
        if not reg or reg.lower() == "nan":
            continue
        url = str(r.get("ProfileURL", "") or "").strip()
        if url:
            per_url[url] = reg
        counts[reg] = counts.get(reg, 0) + 1
    tag = max(counts, key=counts.get) if counts else ""
    return tag, per_url


def repair_match(event_csv_id, match_id, mp, sr, dry_run=False, pause=0.4):
    """Re-scrape one match. Returns (before_state, after_state, action, note)."""
    mid = str(match_id)
    stored_map = mp[mp["MatchID"] == mid].to_dict("records")
    stored_ser = sr[sr["MatchID"] == mid].to_dict("records") if sr is not None else []
    before = MSI.classify_rows(stored_map, stored_ser)

    tag, per_url = _region_context(stored_map + stored_ser)
    url = f"https://www.vlr.gg/{mid}/"
    soup = _fetch(url, retries=2)
    time.sleep(pause)
    if soup is None:
        return before, before, "fetch-failed", ""

    new_map, new_ser, display = _parse_match_html(soup, url, tag)
    if not new_map and not new_ser:
        return before, before, "no-rows", display

    for r in new_map + new_ser:
        r["Region"] = per_url.get(str(r.get("ProfileURL", "")).strip(), r.get("Region") or tag)

    after = MSI.classify_rows(new_map, new_ser)
    old_score, new_score = _completeness_score(stored_map, stored_ser), _completeness_score(new_map, new_ser)
    if new_score <= old_score:
        return before, after, "unchanged", f"{display} ({new_score} vs {old_score} stats)"

    if dry_run:
        return before, after, "would-fix", f"{display} (+{new_score - old_score} stats)"

    _replace_rows(event_csv_id, mid, new_map, new_ser, mp, sr)
    return before, after, "fixed", f"{display} (+{new_score - old_score} stats)"


def _replace_rows(event_csv_id, mid, new_map, new_ser, mp, sr):
    """Swap a match's rows out of the event CSVs. A plain append would leave the
    poisoned rows in place next to the good ones — RefreshLiveData's
    concat+drop_duplicates only collapses rows that match on EVERY column."""
    mp_path = os.path.join(ROOT, "data", "maps",   f"{event_csv_id}.csv")
    sr_path = os.path.join(ROOT, "data", "series", f"{event_csv_id}.csv")
    for path, df, rows in ((mp_path, mp, new_map), (sr_path, sr, new_ser)):
        if df is None:
            continue
        cols = list(df.columns)
        kept = df[df["MatchID"] != mid]
        add = pd.DataFrame(rows).reindex(columns=cols)
        add["MatchID"] = add["MatchID"].astype(str)
        pd.concat([kept, add], ignore_index=True)[cols].to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", help="event csv id, e.g. 2026_stage2 (default: all)")
    ap.add_argument("--match", help="single MatchID (requires --event)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--partial-only", action="store_true",
                    help="only matches with SOME maps rated (skip all-blank ones)")
    ap.add_argument("--limit", type=int, default=0, help="max matches to touch")
    args = ap.parse_args()

    events = ([args.event] if args.event else
              sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(ROOT, "data", "maps", "*.csv"))))

    total = fixed = 0
    for ev in events:
        mp, sr = MSI.load_event(ev)
        if mp is None or "R2.0" not in mp.columns:
            continue
        if args.match:
            targets = [(args.match, MSI.classify_stored(mp, sr, args.match))]
        else:
            targets = MSI.scan_event(ev)
            if args.partial_only:
                targets = [t for t in targets if t[1] == "partial"]
        if not targets:
            continue
        print(f"\n=== {ev}: {len(targets)} match(es) to check")
        for mid, state in targets:
            if args.limit and total >= args.limit:
                print("  (limit reached)")
                break
            total += 1
            before, after, action, note = repair_match(ev, mid, mp, sr, dry_run=args.dry_run)
            if action in ("fixed", "would-fix"):
                fixed += 1
                # Re-read so the next match in this event sees the new file.
                if action == "fixed":
                    mp, sr = MSI.load_event(ev)
            if not args.dry_run:
                MSI.record_attempt(mid, after)
            print(f"  {mid}  {before:8s} → {after:8s}  {action:12s} {note}")

    print(f"\n{fixed}/{total} match(es) {'would be ' if args.dry_run else ''}repaired.")
    if fixed and not args.dry_run:
        print("Reminder: rebuild anything derived from these CSVs "
              "(BuildMatchResults.py / rating timeline) if the fix touched scores.")


if __name__ == "__main__":
    main()
