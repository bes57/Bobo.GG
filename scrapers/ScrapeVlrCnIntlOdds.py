"""
ScrapeVlrCnIntlOdds.py
======================

Scrape VLR.gg betting odds for every CN-involved international match
(Masters / Champions / 2024-2026 stage 1) in the BenPom dataset.

For each match, extract the bookmaker rows (Thunderpick preferred, Rainbet
fallback) of the form:

    Thunderpick  $100 on Paper Rex returned $112 at pre-match odds

Decimal odds are returned/100, implied prob is 100/returned.

We then attach BenPom's pre-match prediction (sigmoid(beta * delta)) — WITHOUT
the CN-debut boost — along with the intl_exp_diff sign, and write
data/vlr_cn_intl_odds.json.

Run:
    .venv/bin/python scrapers/ScrapeVlrCnIntlOdds.py
"""

import json
import math
import os
import re
import sys
import time
from typing import Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bs4 import BeautifulSoup  # type: ignore

# ── Config ────────────────────────────────────────────────────────────────
CACHE_DIR  = "/tmp/vlr_cache"
OUT_PATH   = os.path.join(ROOT, "data", "vlr_cn_intl_odds.json")
SLEEP_SECS = 2.0
TIMEOUT    = 12

# BenPom coefficients used for the "no cn-debut boost" prediction.
# Mirrors AnalyzeProjectionCalibration.py shipped values for INTL_BONUS,
# but with CN_INTL_EXP_BOOST = 0 (the whole point — we're calibrating it).
BETA           = 0.154
INTL_BONUS     = 0.22
CN_DEBUT_BOOST = 0.0   # explicit zero for the no-boost baseline

INTL_EVENTS = {
    "2024_masters_madrid", "2024_masters_shanghai", "2024_champions",
    "2025_masters_bangkok", "2025_masters_toronto", "2025_champions",
    "2026_masters_santiago",
}

# Browser headers (mirrors RefreshLiveData.py).
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/130.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,"
              "image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Cache-Control": "no-cache",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}

# curl_cffi for Cloudflare bypass
from curl_cffi import requests as cffi_requests  # type: ignore

# MapElo team -> region
from MapElo import ORG_REGIONS  # type: ignore


# ── HTTP / cache ──────────────────────────────────────────────────────────
def _cached_html(match_id: int) -> Optional[str]:
    p = os.path.join(CACHE_DIR, f"{match_id}.html")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    return None


def _save_cache(match_id: int, html: str) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    p = os.path.join(CACHE_DIR, f"{match_id}.html")
    with open(p, "w", encoding="utf-8") as f:
        f.write(html)


def fetch_match_html(match_id: int) -> Optional[str]:
    """Fetch (or load cached) VLR match HTML. Tries chrome131/120/chrome."""
    cached = _cached_html(match_id)
    if cached is not None:
        return cached
    url = f"https://www.vlr.gg/{match_id}/"
    last_err = None
    for imp in ("chrome131", "chrome120", "chrome"):
        try:
            r = cffi_requests.get(url, headers=HEADERS, timeout=TIMEOUT,
                                  impersonate=imp, allow_redirects=True)
            if r.status_code == 200 and r.text:
                _save_cache(match_id, r.text)
                return r.text
            last_err = f"HTTP {r.status_code}"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
    print(f"  fetch failed for {match_id}: {last_err}", flush=True)
    return None


# ── Parsing ───────────────────────────────────────────────────────────────
_BET_RE = re.compile(
    r"\$(?P<stake>\d+(?:\.\d+)?)\s+on\s+(?P<team>.+?)\s+returned\s+\$(?P<ret>\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def parse_betting(html: str) -> dict:
    """Return {bookmaker_name: {team, decimal_odds, implied_prob}}.

    On a VLR match page each bookmaker offer is rendered as a
    `<a class="match-bet-item-bet">` containing the bookmaker name and the
    "$X on TEAM returned $Y at pre-match odds" sentence.
    """
    soup = BeautifulSoup(html, "html.parser")
    out: dict[str, dict] = {}

    # The bookmaker rows: anchors with class `match-bet-item`. Each contains
    # an <img class="mod-{bookmaker}"> and the "$X on TEAM returned $Y" text.
    candidates = soup.find_all("a", class_=re.compile(r"match-bet-item"))
    if not candidates:
        candidates = [a for a in soup.find_all("a") if "returned $" in a.get_text(" ", strip=True)]

    # Use a list of (bookmaker, team) -> row so we can hold multiple sides.
    rows: list[tuple[str, dict]] = []
    for a in candidates:
        text = a.get_text(" ", strip=True)
        if "returned $" not in text:
            continue
        m = _BET_RE.search(text)
        if not m:
            continue
        stake = float(m.group("stake"))
        ret = float(m.group("ret"))
        team = m.group("team").strip()
        if stake <= 0 or ret <= 0:
            continue
        decimal_odds = ret / stake
        if decimal_odds < 1.01:
            continue
        implied = 1.0 / decimal_odds

        # Bookmaker name from <img class="mod-{name}">.
        bm = "Unknown"
        img = a.find("img", class_=re.compile(r"mod-"))
        if img:
            classes = img.get("class") or []
            for c in classes:
                if c.startswith("mod-"):
                    bm = c[len("mod-"):]
                    break
        bm_norm = bm.capitalize() if bm.islower() else bm
        rows.append((bm_norm, {
            "team": team,
            "decimal_odds": round(decimal_odds, 4),
            "implied_prob": round(implied, 4),
        }))

    # Flatten: if a single bookmaker quotes two different teams, keep BOTH
    # under keys like "Thunderpick" and "Thunderpick__other".
    for bm, row in rows:
        if bm not in out:
            out[bm] = row
        elif out[bm]["team"] != row["team"]:
            # Same bookmaker, other side — add as "<bm>__2" for downstream devig.
            key = f"{bm}__2"
            if key not in out:
                out[key] = row
        # else: duplicate of same side, skip.
    return out


# ── Team-name reconciliation ──────────────────────────────────────────────
# Map VLR's full names to the short codes used in rating_timeline / ORG_REGIONS.
# Built ad-hoc for the 58 CN-involved intl matches; extend as needed.
TEAM_ALIASES = {
    # CN
    "edward gaming": "EDG", "edg": "EDG",
    "bilibili gaming": "BLG", "blg": "BLG",
    "trace esports": "TE", "te": "TE",
    "dragon ranger gaming": "DRG", "drg": "DRG",
    "all gamers": "AG", "ag": "AG",
    "xi lai gaming": "XLG", "xlg": "XLG",
    "wolves esports": "WOL", "wol": "WOL",
    "funplus phoenix": "FPX", "fpx": "FPX",
    "jd gaming": "JDG", "jdg": "JDG",
    "titan esports club": "TEC", "tec": "TEC",
    "tyloo": "TYLOO", "tyl": "TYL",
    "nova esports": "NOVA",
    # EMEA
    "team liquid": "TL", "tl": "TL",
    "fnatic": "FNC", "fnc": "FNC",
    "team vitality": "VIT", "vit": "VIT",
    "team heretics": "TH", "th": "TH",
    "fut esports": "FUT", "fut": "FUT",
    "karmine corp": "KC", "kc": "KC",
    "giantx": "GX", "gx": "GX",
    "movistar koi": "MKOI", "mkoi": "MKOI",
    "koi": "KOI",
    "natus vincere": "NAVI", "navi": "NAVI",
    "bbl esports": "BBL", "bbl": "BBL",
    "gentle mates": "M8", "m8": "M8",
    # Americas
    "sentinels": "SEN", "sen": "SEN",
    "g2 esports": "G2", "g2": "G2",
    "mibr": "MIBR",
    "nrg esports": "NRG", "nrg": "NRG",
    "100 thieves": "100T", "100t": "100T",
    "cloud9": "C9", "c9": "C9",
    "evil geniuses": "EG", "eg": "EG",
    "krü esports": "KRÜ", "kru esports": "KRÜ", "krü": "KRÜ", "kru": "KRÜ",
    "leviatán": "LEV", "leviatan": "LEV", "lev": "LEV",
    "furia": "FUR", "fur": "FUR",
    "loud": "LOUD",
    # Pacific
    "paper rex": "PRX", "prx": "PRX",
    "drx": "DRX",
    "t1": "T1",
    "talon esports": "TLN", "tln": "TLN",
    "gen.g": "GEN", "geng": "GEN", "gen": "GEN",
    "detonation focusme": "DFM", "dfm": "DFM",
    "zeta division": "ZETA", "zeta": "ZETA",
    "rex regum qeon": "RRQ", "rrq": "RRQ",
    "team secret": "TS", "ts": "TS",
    "global esports": "GE", "ge": "GE",
    "drx (pacific)": "DRX",
    "drx changers": "KRX", "krx": "KRX",
    "nongshim redforce": "NS", "ns": "NS",
    "boom esports": "BOOM", "boom": "BOOM",
}


def normalize_team(vlr_name: str) -> Optional[str]:
    key = vlr_name.strip().lower()
    if key in TEAM_ALIASES:
        return TEAM_ALIASES[key]
    # Try the upper-cased token (already-a-short-code case).
    up = vlr_name.strip().upper()
    if up in ORG_REGIONS:
        return up
    return None


# ── Devig ─────────────────────────────────────────────────────────────────
def devig(prob_a: float, prob_b: Optional[float]) -> float:
    """If both probs known, divide by total. If only one, assume 1.05 overround."""
    if prob_b is not None and prob_b > 0:
        total = prob_a + prob_b
        if total > 0:
            return prob_a / total
    # Single-sided: assume 5% overround.
    return min(0.999, max(0.001, prob_a / 1.05))


# ── BenPom prediction (no CN-debut boost) ─────────────────────────────────
def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def series_prob_from_map(p: float, bo: int) -> float:
    if bo == 3:
        return (p ** 2) * (3 - 2 * p)
    if bo == 5:
        return (p ** 3) * (10 - 15 * p + 6 * p * p)
    return p


def infer_bo(series_score: str, n_maps: int) -> int:
    if not series_score:
        return 3 if n_maps <= 3 else 5
    try:
        a, b = series_score.split("-")
        m = max(int(a), int(b))
        if m >= 3: return 5
        if m == 2: return 3
    except Exception:
        pass
    return 3 if n_maps <= 3 else 5


def _build_intl_attendance(all_matches: list[dict]) -> dict:
    attendance: dict = {}
    for m in sorted(all_matches, key=lambda r: (r["date"], r["match_id"])):
        if m["event_id"] not in INTL_EVENTS:
            continue
        season = m["season"]
        for org in (m["winner"], m["loser"]):
            attendance.setdefault((org, season), []).append((m["date"], m["event_id"]))
    return attendance


def intl_exp_diff(fav_org: str, dog_org: str, season: int, match_date: str, attendance: dict) -> int:
    def attended_before(org):
        for d, _ in attendance.get((org, season), []):
            if d < match_date:
                return True
        return False
    f = attended_before(fav_org)
    d = attended_before(dog_org)
    return (1 if f else 0) - (1 if d else 0)


def benpom_p_fav_no_boost(m: dict, attendance: dict) -> tuple[float, int, str, str]:
    """Return (p_fav, intl_exp_diff_sign, fav_org, dog_org) — no CN-debut boost."""
    wb = float(m["winner_before"])
    lb = float(m["loser_before"])
    delta = wb - lb
    if wb >= lb:
        fav_org, dog_org = m["winner"], m["loser"]
    else:
        fav_org, dog_org = m["loser"], m["winner"]
    abs_delta = abs(delta)
    p_map = sigmoid(BETA * abs_delta)
    n_maps = len(m.get("maps", []))
    bo = infer_bo(m.get("series_score", ""), n_maps)
    p_series = series_prob_from_map(p_map, bo)

    sign = 0
    if INTL_BONUS > 0:
        sign = intl_exp_diff(fav_org, dog_org, m["season"], m["date"], attendance)
        if sign != 0:
            ps = max(min(p_series, 1 - 1e-9), 1e-9)
            logit_ps = math.log(ps / (1 - ps)) + INTL_BONUS * sign
            p_series = 1.0 / (1.0 + math.exp(-logit_ps))
    # CN_DEBUT_BOOST = 0.0 → no further adjustment
    return p_series, sign, fav_org, dog_org


# ── Match list ────────────────────────────────────────────────────────────
def load_all_matches() -> list[dict]:
    """Load all match_events with season tag."""
    files = [
        ("rating_timeline_2024.json", 2024),
        ("rating_timeline_2025.json", 2025),
        ("rating_timeline.json",      2026),
    ]
    out = []
    for fname, season in files:
        p = os.path.join(ROOT, "data", fname)
        if not os.path.exists(p):
            continue
        with open(p) as f:
            d = json.load(f)
        for m in d.get("match_events", []):
            m = dict(m)
            m["season"] = season
            out.append(m)
    return out


def build_cn_intl_match_list(all_matches: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for m in all_matches:
        if m["event_id"] not in INTL_EVENTS:
            continue
        w, l = m["winner"], m["loser"]
        rw = ORG_REGIONS.get(w, "UNK")
        rl = ORG_REGIONS.get(l, "UNK")
        if (rw == "CN") == (rl == "CN"):  # need exactly one CN
            continue
        if float(m.get("winner_before", 0)) == float(m.get("loser_before", 0)):
            # BenPom has no opinion (brand-new team) — skip for calibration
            continue
        mid = m["match_id"]
        if mid in seen:
            continue
        seen.add(mid)
        out.append({
            "match_id": mid, "date": m["date"], "event_id": m["event_id"],
            "winner": w, "loser": l, "winner_region": rw, "loser_region": rl,
            "_match": m,  # keep raw row for prediction
        })
    out.sort(key=lambda x: (x["date"], x["match_id"]))
    return out


# ── Main ──────────────────────────────────────────────────────────────────
def pick_bookmaker(bms: dict) -> Optional[str]:
    """Prefer Thunderpick; else Rainbet; else first non-duplicate key."""
    # Case-insensitive lookup of bookmaker names (excluding "__2" duplicates).
    base_keys = [k for k in bms.keys() if not k.endswith("__2")]
    norm = {k.lower(): k for k in base_keys}
    for pref in ("thunderpick", "rainbet"):
        if pref in norm:
            return norm[pref]
    if base_keys:
        return base_keys[0]
    return None


def reconcile_team_to_match(team_str: str, winner: str, loser: str) -> Optional[str]:
    """Given the VLR-rendered team text, return whether it's `winner` or `loser`."""
    short = normalize_team(team_str)
    if short == winner: return winner
    if short == loser: return loser
    # Substring fallback for nicknames not in alias map.
    low = team_str.lower()
    # Try to match by aliases that map to one of the two teams.
    for alias, code in TEAM_ALIASES.items():
        if alias in low and code == winner:
            return winner
        if alias in low and code == loser:
            return loser
    return None


def main():
    all_matches = load_all_matches()
    attendance = _build_intl_attendance(all_matches)
    targets = build_cn_intl_match_list(all_matches)
    print(f"Scraping {len(targets)} CN-involved intl matches…", flush=True)

    results = []
    failed = []
    no_odds = []
    for i, t in enumerate(targets, 1):
        mid = t["match_id"]
        already_cached = _cached_html(mid) is not None
        html = fetch_match_html(mid)
        if html is None:
            failed.append(mid)
            print(f"  [{i}/{len(targets)}] {mid} {t['winner']} vs {t['loser']}: FETCH FAILED", flush=True)
            continue
        bms_raw = parse_betting(html)
        # Reconcile team names → either winner or loser short code.
        bms_clean: dict = {}
        for name, row in bms_raw.items():
            side = reconcile_team_to_match(row["team"], t["winner"], t["loser"])
            if side is None:
                continue
            bms_clean[name] = {
                "team": side,
                "decimal_odds": row["decimal_odds"],
                "implied_prob": row["implied_prob"],
            }

        chosen_bm = pick_bookmaker(bms_clean)

        # BenPom no-boost prediction.
        p_fav_bp, sign, fav_org, dog_org = benpom_p_fav_no_boost(t["_match"], attendance)

        p_fav_book = None
        if chosen_bm:
            # Collect this bookmaker's offerings for both sides if present.
            # Note: parse_betting deduplicates per bookmaker+team — if the
            # bookmaker quoted both sides we'll have entries under DIFFERENT
            # bookmaker keys only if the page renders separate rows; we treat
            # the chosen bookmaker as a single row referencing one team.
            chosen = bms_clean[chosen_bm]
            chosen_side = chosen["team"]
            chosen_prob = chosen["implied_prob"]

            # Look for the SAME bookmaker quoting the other side (stored as
            # "<bm>__2" by parse_betting when both rows are listed).
            other_prob = None
            other_key = f"{chosen_bm}__2"
            if other_key in bms_clean and bms_clean[other_key]["team"] != chosen_side:
                other_prob = bms_clean[other_key]["implied_prob"]

            true_prob_chosen = devig(chosen_prob, other_prob)
            # Convert to favorite-side probability.
            if chosen_side == fav_org:
                p_fav_book = true_prob_chosen
            else:
                # chosen_side is the dog; fav prob = 1 - dog true prob.
                p_fav_book = 1.0 - true_prob_chosen

        rec = {
            "match_id": mid,
            "date": t["date"],
            "event_id": t["event_id"],
            "winner": t["winner"], "loser": t["loser"],
            "winner_region": t["winner_region"], "loser_region": t["loser_region"],
            "fav_org": fav_org, "dog_org": dog_org,
            "bookmakers": bms_clean,
            "p_fav_book_devigged": (round(p_fav_book, 4) if p_fav_book is not None else None),
            "p_fav_benpom": round(p_fav_bp, 4),
            "intl_exp_diff_sign": sign,
        }
        results.append(rec)
        if not bms_clean:
            no_odds.append(mid)
            tag = "no odds"
        else:
            tag = f"{chosen_bm} → p_book={p_fav_book:.3f}, p_bp={p_fav_bp:.3f}, dsign={sign}"
        print(f"  [{i}/{len(targets)}] {mid} {fav_org} fav vs {dog_org}: {tag}", flush=True)

        if not already_cached:
            time.sleep(SLEEP_SECS)

    out_doc = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "beta": BETA, "intl_bonus": INTL_BONUS,
            "cn_debut_boost": CN_DEBUT_BOOST,
            "overround_assumption": 1.05,
        },
        "n_targets": len(targets),
        "n_scraped": len(results),
        "n_with_odds": len(results) - len(no_odds),
        "n_failed": len(failed),
        "failed_match_ids": failed,
        "no_odds_match_ids": no_odds,
        "matches": results,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out_doc, f, indent=2)
    print(f"\nWrote {OUT_PATH}")
    print(f"  scraped={len(results)}  with_odds={out_doc['n_with_odds']}  no_odds={len(no_odds)}  failed={len(failed)}")


if __name__ == "__main__":
    main()
