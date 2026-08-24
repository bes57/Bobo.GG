"""
MatchStatsIntegrity.py — detect (and let callers repair) match pages that were
scraped before VLR finished publishing their round-derived stats.

THE BUG THIS EXISTS FOR
-----------------------
VLR builds a match page from two different sources. K/D/A, KAST, HS% and the
first-blood counts come straight off the scoreboard and are final the moment
the match ends. Rating 2.0, ACS and ADR are derived from per-round data, which
VLR ingests separately and a few minutes later.

Scrape inside that window and the page serves *empty* rating/ACS/ADR cells for
every map whose round data hasn't landed yet — and, far more damaging, its
"all maps" tab aggregates only the maps that HAVE landed. So a Bo3 scraped
after map 1 was processed but before maps 2-3 were gives a series row whose
R2.0/ACS/ADR are literally map 1's numbers, attached to the full series' K/D/A.

That is how KovaQ's 1.97 on Summit (17-4) and crownfisher's 1.89 on Breeze
(26-9) ended up on the all-time *Bo3 series* leaderboard. Their real series
ratings are 1.22 and 1.51.

Because RefreshLiveData skips any MatchID already present in the maps CSV, a
match poisoned this way was frozen forever. The two halves of the fix:

  * classify a scrape/stored match as complete | partial | unrated, and
  * let an incomplete match be re-scraped (bounded by a retry ledger so a
    genuinely unrated event — most CN qualifiers — isn't hammered forever).

VOCABULARY
  complete → every map has round-derived stats, and so does the series row.
  partial  → some maps rated, some not. The series row is untrustworthy.
             Always worth re-scraping.
  unrated  → nothing rated anywhere. Either the same publish window caught the
             page before ANY map was processed, or VLR simply never rates this
             event. Indistinguishable from one match alone — callers decide
             using event-level context (see event_is_rated).
"""
import os, json, datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
LEDGER_PATH = os.path.join(DATA, "stats_recheck.json")

# The three columns VLR derives from round data, and therefore the three that
# can be missing while the scoreboard columns are already final.
ROUND_STATS = ("R2.0", "ACS", "ADR")

# Bounds on re-scraping an incomplete match, so an event VLR never rates costs
# a handful of fetches rather than one per refresh forever.
MAX_ATTEMPTS   = 6
COOLDOWN_HOURS = 6


def _blank(v):
    if v is None:
        return True
    s = str(v).strip()
    return s == "" or s.lower() in ("nan", "none")


def _rated(row):
    """A row carries any round-derived stat at all."""
    return any(not _blank(row.get(c)) for c in ROUND_STATS)


def _has(rows, col):
    return any(not _blank(r.get(col)) for r in rows)


def classify_rows(map_rows, series_rows):
    """Classify freshly-parsed rows (list-of-dicts, as _parse_match_html returns).

    Returns "complete" | "partial" | "unrated" | "empty".

    Each round-derived column is judged on its own, because VLR's coverage is
    per-column, not all-or-nothing: unrated events (most CN qualifiers) publish
    ACS and ADR but never a rating 2.0. A column absent from EVERY map of the
    match is that event's normal state; a column present on some maps and
    missing from others is the mid-publish window, and is what corrupts the
    series row.
    """
    if not map_rows and not series_rows:
        return "empty"

    by_map = {}
    for r in map_rows:
        by_map.setdefault(str(r.get("MapNum", "")), []).append(r)

    if not by_map:
        # Series row only (shouldn't happen, but don't call it partial).
        return "complete" if any(_rated(r) for r in series_rows) else "unrated"

    any_covered = False
    for col in ROUND_STATS:
        covered = [mn for mn, rows in by_map.items() if _has(rows, col)]
        if not covered:
            continue                      # event never publishes this column
        any_covered = True
        if len(covered) < len(by_map):
            return "partial"              # mid-publish: some maps still empty
        # Every map has it, so the "all maps" aggregate must too — otherwise the
        # page was still assembling and the series row can't be trusted.
        if series_rows and not _has(series_rows, col):
            return "partial"
    return "complete" if any_covered else "unrated"


def classify_stored(maps_df, series_df, match_id):
    """Same classification, against what's already on disk. `maps_df`/`series_df`
    are DataFrames read with dtype=str; `match_id` a string."""
    mrows = maps_df[maps_df["MatchID"] == match_id].to_dict("records") if maps_df is not None else []
    srows = series_df[series_df["MatchID"] == match_id].to_dict("records") if series_df is not None else []
    return classify_rows(mrows, srows)


def _event_paths(event_csv_id):
    return (os.path.join(DATA, "maps",   f"{event_csv_id}.csv"),
            os.path.join(DATA, "series", f"{event_csv_id}.csv"))


def load_event(event_csv_id):
    """(maps_df, series_df) as dtype=str with MatchID stripped, or (None, None)."""
    import pandas as pd
    mp_path, sr_path = _event_paths(event_csv_id)
    if not os.path.exists(mp_path):
        return None, None
    try:
        mp = pd.read_csv(mp_path, dtype=str)
        mp["MatchID"] = mp["MatchID"].astype(str).str.strip()
    except Exception:
        return None, None
    try:
        sr = pd.read_csv(sr_path, dtype=str)
        sr["MatchID"] = sr["MatchID"].astype(str).str.strip()
    except Exception:
        sr = None
    return mp, sr


def event_is_rated(maps_df, threshold=0.2):
    """Does VLR rate this event at all? True when a meaningful share of its
    matches carry round-derived stats. Used to decide whether an all-blank
    match is a scrape-timing casualty (repair it) or just how the event is
    (leave it alone) — most CN qualifier events are entirely unrated."""
    if maps_df is None or maps_df.empty or "R2.0" not in maps_df.columns:
        return False
    ids = maps_df["MatchID"].dropna().unique()
    if len(ids) == 0:
        return False
    rated = sum(1 for mid in ids
                if any(_rated(r) for r in maps_df[maps_df["MatchID"] == mid].to_dict("records")))
    return (rated / len(ids)) >= threshold


def scan_event(event_csv_id, include_unrated=None):
    """Every stored match of one event that needs re-scraping.

    Returns [(match_id, state), ...]. "partial" always qualifies. "unrated"
    qualifies only when the event is otherwise rated (auto-detected unless
    include_unrated is passed explicitly)."""
    mp, sr = load_event(event_csv_id)
    if mp is None or "R2.0" not in mp.columns:
        return []
    if include_unrated is None:
        include_unrated = event_is_rated(mp)
    out = []
    for mid in mp["MatchID"].dropna().unique():
        state = classify_stored(mp, sr, mid)
        if state == "partial" or (state == "unrated" and include_unrated):
            out.append((str(mid), state))
    return out


# ── Retry ledger ─────────────────────────────────────────────────────────────
def _load_ledger():
    try:
        with open(LEDGER_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_ledger(led):
    try:
        os.makedirs(os.path.dirname(LEDGER_PATH), exist_ok=True)
        with open(LEDGER_PATH, "w") as f:
            json.dump(led, f, indent=2, sort_keys=True)
    except Exception:
        pass


def due_for_recheck(match_id, led=None):
    """Is this incomplete match allowed another attempt right now?"""
    led = _load_ledger() if led is None else led
    rec = led.get(str(match_id))
    if not rec:
        return True
    if int(rec.get("attempts", 0)) >= MAX_ATTEMPTS:
        return False
    try:
        last = datetime.datetime.fromisoformat(rec.get("last", ""))
    except Exception:
        return True
    age = datetime.datetime.now(datetime.timezone.utc) - last
    return age >= datetime.timedelta(hours=COOLDOWN_HOURS)


def record_attempt(match_id, state):
    """Log one re-scrape attempt. A match that comes back complete is dropped
    from the ledger so a future regression starts from a clean slate."""
    led = _load_ledger()
    key = str(match_id)
    if state == "complete":
        led.pop(key, None)
    else:
        rec = led.get(key, {"attempts": 0})
        rec["attempts"] = int(rec.get("attempts", 0)) + 1
        rec["last"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        rec["state"] = state
        led[key] = rec
    _save_ledger(led)


def rescrapeable_ids(event_csv_id):
    """MatchIDs already on disk that should NOT count as 'already scraped',
    because their stored stats are incomplete and they're due another try."""
    led = _load_ledger()
    return {mid for mid, _state in scan_event(event_csv_id)
            if due_for_recheck(mid, led)}
