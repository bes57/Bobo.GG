#!/usr/bin/env python3
"""Pull Kalshi VALORANT match markets + candlesticks and build a backtest dataset.

Public API, no auth. Resumable: markets already in markets_raw.jsonl are not
re-fetched into the cache; candle files already on disk are skipped.

Outputs (under testing_lab/kalshi/):
  markets_raw.jsonl            one line per market, raw API JSON
  candles/{market_ticker}.json raw candlestick responses (tier-1 events only)
  kalshi_matches.csv           one row per event (match)
  SUMMARY.md                   counts / distributions / anomalies

Usage: python3 testing_lab/pull_kalshi.py
"""

import csv
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, "/Users/benny_es1/VCTMM")
from vctmm.benpom.teams import resolve_team_name  # noqa: E402

# Kalshi display names that the BenPom resolver misses but are verified tier-1
# (checked against rules_primary competition text in the raw markets):
#   "DRX"       -> VCT Pacific 2026 org, BenPom code KRX (Kiwoom DRX rebrand)
#   "JD Gaming" -> VCT CN org JDG (JDG Esports parent brand)
# Exact-match only, so "DRX Academy"/"DRX Prospects" stay unresolved (tier-2).
KALSHI_ALIASES = {"DRX": "KRX", "JD Gaming": "JDG"}


def resolve(name):
    org = resolve_team_name(name)
    if org is None:
        org = KALSHI_ALIASES.get(name)
    return org

BASE = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = "KXVALORANTGAME"
HERE = Path(__file__).resolve().parent
OUT = HERE / "kalshi"
CANDLE_DIR = OUT / "candles"
RAW_PATH = OUT / "markets_raw.jsonl"
CSV_PATH = OUT / "kalshi_matches.csv"
SUMMARY_PATH = OUT / "SUMMARY.md"

RATE = 0.21          # seconds between requests (~5 req/s)
CANDLE_PERIOD = 1    # minute candles
MAX_CANDLES = 4900   # API chunk safety limit per request
_last_req = [0.0]

session = requests.Session()
session.headers["User-Agent"] = "kalshi-backtest-puller/1.0"


def api_get(path, params=None, retries=4):
    wait = RATE - (time.monotonic() - _last_req[0])
    if wait > 0:
        time.sleep(wait)
    for attempt in range(retries):
        _last_req[0] = time.monotonic()
        try:
            r = session.get(BASE + path, params=params, timeout=30)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"GET {path} failed after {retries} tries")


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def fnum(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------- markets ---

def pull_markets():
    """Fetch all settled+finalized markets; append new ones to the jsonl cache."""
    OUT.mkdir(parents=True, exist_ok=True)
    cached = {}
    if RAW_PATH.exists():
        with RAW_PATH.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    m = json.loads(line)
                    cached[m["ticker"]] = m

    fetched = {}
    # NOTE: status=finalized is rejected with 400 "invalid status filter";
    # status=settled already returns markets whose status field says "finalized".
    for status in ("settled",):
        cursor = None
        while True:
            params = {"series_ticker": SERIES, "status": status, "limit": 1000}
            if cursor:
                params["cursor"] = cursor
            data = api_get("/markets", params)
            for m in data.get("markets", []):
                fetched[m["ticker"]] = m
            cursor = data.get("cursor")
            if not cursor or not data.get("markets"):
                break

    new = [m for t, m in fetched.items() if t not in cached]
    if new:
        with RAW_PATH.open("a") as f:
            for m in new:
                f.write(json.dumps(m, separators=(",", ":")) + "\n")
    cached.update(fetched)  # fetched copies are freshest but we keep cache keys
    print(f"[markets] fetched={len(fetched)} new={len(new)} cache_total={len(cached)}")
    return list(cached.values())


# ---------------------------------------------------------------- candles ---

def pull_candles_for(market):
    """Fetch minute candles over the market's life; cache to disk. Resumable."""
    ticker = market["ticker"]
    path = CANDLE_DIR / f"{ticker}.json"
    if path.exists():
        return json.loads(path.read_text())

    open_ts = int(parse_ts(market["open_time"]).timestamp())
    close_ts = int(parse_ts(market["close_time"]).timestamp())
    end_ts = close_ts + 120
    all_candles = []
    chunk_start = open_ts - 60
    while chunk_start < end_ts:
        chunk_end = min(chunk_start + MAX_CANDLES * 60 * CANDLE_PERIOD, end_ts)
        data = api_get(
            f"/series/{SERIES}/markets/{ticker}/candlesticks",
            {"start_ts": chunk_start, "end_ts": chunk_end,
             "period_interval": CANDLE_PERIOD},
        )
        all_candles.extend(data.get("candlesticks", []))
        chunk_start = chunk_end
    # dedupe on end_period_ts, keep order
    seen, deduped = set(), []
    for c in all_candles:
        ts = c["end_period_ts"]
        if ts not in seen:
            seen.add(ts)
            deduped.append(c)
    payload = {"ticker": ticker, "period_interval": CANDLE_PERIOD,
               "candlesticks": deduped}
    path.write_text(json.dumps(payload, separators=(",", ":")))
    return payload


# ------------------------------------------------------------ price logic ---

def candle_prob(c):
    """Best probability estimate from one candle (0-1), or None."""
    bid = fnum(c.get("yes_bid", {}).get("close_dollars"), None)
    ask = fnum(c.get("yes_ask", {}).get("close_dollars"), None)
    trade = c.get("price", {}).get("close_dollars")
    trade = fnum(trade, None) if trade is not None else None
    if bid is not None and ask is not None:
        empty_book = bid <= 0.0 and ask >= 1.0
        if not empty_book and (ask - bid) <= 0.30:
            return (bid + ask) / 2.0
    if trade is not None:
        return trade
    if bid is not None and ask is not None and not (bid <= 0.0 and ask >= 1.0):
        return (bid + ask) / 2.0
    return None


def market_prices(market, candles):
    """Return (close_prob, t2h_prob, vwap, wide_spread_flag) for one market."""
    cs = sorted(candles.get("candlesticks", []), key=lambda c: c["end_period_ts"])
    if not cs:
        return None, None, None, False
    close_ts = int(parse_ts(market["close_time"]).timestamp())

    # (a) last candle at-or-before close_time - 5 min
    close_prob, wide = None, False
    eligible = [c for c in cs if c["end_period_ts"] <= close_ts - 300]
    for c in reversed(eligible):
        p = candle_prob(c)
        if p is not None:
            close_prob = p
            bid = fnum(c.get("yes_bid", {}).get("close_dollars"), 0)
            ask = fnum(c.get("yes_ask", {}).get("close_dollars"), 1)
            wide = (ask - bid) > 0.30
            break

    # (b) candle nearest to close_time - 2h (within 30 min)
    t2h_prob = None
    target = close_ts - 7200
    best = min(cs, key=lambda c: abs(c["end_period_ts"] - target))
    if abs(best["end_period_ts"] - target) <= 1800:
        t2h_prob = candle_prob(best)

    # (c) volume-weighted average trade price
    num = den = 0.0
    for c in cs:
        v = fnum(c.get("volume_fp"))
        mean = c.get("price", {}).get("mean_dollars")
        if v > 0 and mean is not None:
            num += fnum(mean) * v
            den += v
    vwap = num / den if den > 0 else None
    return close_prob, t2h_prob, vwap, wide


def combine(pa, pb, va, vb):
    """Combine market-A yes prob and market-B yes prob (as 1-pb), volume-weighted."""
    if pa is not None and pb is not None:
        w = va + vb
        if w <= 0:
            return (pa + (1 - pb)) / 2.0
        return (pa * va + (1 - pb) * vb) / w
    if pa is not None:
        return pa
    if pb is not None:
        return 1 - pb
    return None


# ------------------------------------------------------------------ build ---

def build(markets):
    CANDLE_DIR.mkdir(parents=True, exist_ok=True)

    # group markets by event
    events = {}
    for m in markets:
        events.setdefault(m["event_ticker"], []).append(m)

    anomalies = []
    rows = []
    tier1_events = []
    for ev, ms in sorted(events.items()):
        if len(ms) != 2:
            anomalies.append(f"event {ev} has {len(ms)} markets (expected 2)")
            if len(ms) < 2:
                continue
            ms = ms[:2]
        for m in ms:
            m["_team"] = m.get("yes_sub_title") or m.get("no_sub_title") or ""
            m["_org"] = resolve(m["_team"])
        excluded = any(m["_org"] is None for m in ms)
        # org_a = alphabetically first org (fall back to raw name if unresolved)
        ms.sort(key=lambda m: (m["_org"] or m["_team"]).upper())
        ma, mb = ms
        results = {m["ticker"]: m.get("result") for m in ms}
        base_notes = []
        if any(m.get("result") == "scalar" for m in ms):
            # cancelled/postponed/forfeit -> settled at fair market price
            base_notes.append("scalar_settlement")
            anomalies.append(f"event {ev}: scalar settlement (cancelled/forfeit) ({results})")
        yes_winners = [m for m in ms if m.get("result") == "yes"]
        winner = None
        if len(yes_winners) == 1:
            wm = yes_winners[0]
            winner = wm["_org"] or wm["_team"]
        elif len(yes_winners) == 0:
            no_losers = [m for m in ms if m.get("result") == "no"]
            if len(no_losers) == 1:
                wm = mb if no_losers[0] is ma else ma
                winner = wm["_org"] or wm["_team"]
            elif not base_notes:
                anomalies.append(f"event {ev}: no yes-result market ({results})")
        else:
            anomalies.append(f"event {ev}: both markets resolved yes ({results})")

        row = {
            "event_ticker": ev,
            "date_utc": parse_ts(ma["close_time"]).strftime("%Y-%m-%d %H:%M"),
            "org_a": ma["_org"] or "",
            "org_b": mb["_org"] or "",
            "team_a_raw": ma["_team"],
            "team_b_raw": mb["_team"],
            "winner_org": winner or "",
            "prob_a_close": "",
            "prob_a_t2h": "",
            "prob_a_vwap": "",
            "volume_total": round(fnum(ma.get("volume_fp")) + fnum(mb.get("volume_fp")), 2),
            "liquidity_note": ";".join(base_notes),
            "excluded": excluded,
        }
        rows.append(row)
        if not excluded:
            tier1_events.append((row, ma, mb))

    # candles for tier-1 events only
    print(f"[candles] pulling for {len(tier1_events)} tier-1 events "
          f"({2 * len(tier1_events)} markets)")
    for i, (row, ma, mb) in enumerate(tier1_events, 1):
        notes = [n for n in row["liquidity_note"].split(";") if n]
        try:
            ca = pull_candles_for(ma)
            cb = pull_candles_for(mb)
        except Exception as e:  # keep going; note the failure
            anomalies.append(f"event {row['event_ticker']}: candle fetch failed: {e}")
            row["liquidity_note"] = ";".join(notes + ["candle_fetch_failed"])
            continue
        pa_c, pa_t, pa_v, wa = market_prices(ma, ca)
        pb_c, pb_t, pb_v, wb = market_prices(mb, cb)
        va, vb = fnum(ma.get("volume_fp")), fnum(mb.get("volume_fp"))
        pc = combine(pa_c, pb_c, va, vb)
        pt = combine(pa_t, pb_t, va, vb)
        pv = combine(pa_v, pb_v, va, vb)
        row["prob_a_close"] = round(pc, 4) if pc is not None else ""
        row["prob_a_t2h"] = round(pt, 4) if pt is not None else ""
        row["prob_a_vwap"] = round(pv, 4) if pv is not None else ""
        if row["volume_total"] < 100:
            notes.append("thin<$100vol")
        if wa or wb:
            notes.append("wide_spread_at_close")
        # NB: liquidity_dollars is 0 on ALL finalized markets (book cleared at
        # settlement), so it is useless as a thinness signal — not noted.
        if pc is None:
            notes.append("no_close_price")
        row["liquidity_note"] = ";".join(notes)
        if i % 50 == 0:
            print(f"  ...{i}/{len(tier1_events)} events")

    # suspicious: winner's close prob disagrees with settlement
    suspicious = []
    for row, ma, mb in tier1_events:
        pc = row["prob_a_close"]
        if pc == "" or not row["winner_org"]:
            continue
        winner_prob = pc if row["winner_org"] == row["org_a"] else 1 - pc
        if winner_prob < 0.5:
            suspicious.append(
                f"{row['event_ticker']} ({row['org_a']} vs {row['org_b']}): "
                f"winner={row['winner_org']} close_prob={winner_prob:.3f} vol={row['volume_total']}")

    rows.sort(key=lambda r: r["date_utc"])
    fields = ["event_ticker", "date_utc", "org_a", "org_b", "team_a_raw",
              "team_b_raw", "winner_org", "prob_a_close", "prob_a_t2h",
              "prob_a_vwap", "volume_total", "liquidity_note", "excluded"]
    with CSV_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[csv] wrote {len(rows)} event rows -> {CSV_PATH}")
    return rows, tier1_events, anomalies, suspicious


def write_summary(markets, rows, tier1_events, anomalies, suspicious):
    dates = sorted(r["date_utc"] for r in rows)
    vols = sorted(r["volume_total"] for r in rows)
    t1 = [r for r, _, _ in tier1_events]
    n_close = sum(1 for r in t1 if r["prob_a_close"] != "")
    n_t2h = sum(1 for r in t1 if r["prob_a_t2h"] != "")
    n_both = sum(1 for r in t1 if r["prob_a_close"] != "" and r["prob_a_t2h"] != "")
    n_vwap = sum(1 for r in t1 if r["prob_a_vwap"] != "")

    def pctl(xs, p):
        if not xs:
            return 0
        return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]

    dup = len(rows) - len({r["event_ticker"] for r in rows})
    lines = [
        "# Kalshi VALORANT dataset summary",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        f"- Total markets in raw cache: **{len(markets)}**",
        f"- Total events (matches): **{len(rows)}** (duplicate event tickers: {dup})",
        f"- Tier-1 events (both teams resolve to BenPom orgs): **{len(t1)}**",
        f"- Excluded events (GC / tier-2 / college): **{len(rows) - len(t1)}**",
        f"- Date range (close_time UTC): **{dates[0]} .. {dates[-1]}**" if dates else "- no events",
        "",
        "## Volume (per event, contracts, all events)",
        f"- median: {statistics.median(vols):,.0f}" if vols else "",
        f"- p90: {pctl(vols, 0.90):,.0f}" if vols else "",
        f"- max: {vols[-1]:,.0f}" if vols else "",
        "",
        "## Price coverage (tier-1 events)",
        f"- with close (t-5m) price: {n_close}",
        f"- with t2h price: {n_t2h}",
        f"- with BOTH close and t2h: {n_both}",
        f"- with VWAP: {n_vwap}",
        "",
        "## Suspicious settlements (winner's close prob < 0.5)",
    ]
    lines += [f"- {s}" for s in suspicious] or ["- none"]
    lines += ["", "## Anomalies"]
    lines += [f"- {a}" for a in anomalies] or ["- none"]
    lines += [
        "",
        "## Notes",
        "- `status=finalized` is rejected by the API (400 invalid status filter); `status=settled`",
        "  returns everything, with the market `status` field reading \"finalized\".",
        "- Kalshi alias overlay applied (verified via rules_primary competition text):",
        "  \"DRX\" -> KRX (Kiwoom DRX, VCT Pacific), \"JD Gaming\" -> JDG (VCT CN).",
        "- All API prices are dollar-denominated strings (e.g. \"0.9900\"); no cents scaling needed.",
        "- `price` object in candles is empty when a minute had no trades; bid/ask always present.",
        "- Close prob = last minute-candle at/before close_time-5min; bid/ask midpoint preferred",
        "  (spread<=0.30 and non-empty book), else last trade close.",
        "- prob_a combines both markets: volume-weighted avg of P(yes_A) and 1-P(yes_B).",
        "- Markets close only after a winner is declared, so close prices reflect in-match trading;",
        "  t2h is the better pre/early-match snapshot.",
    ]
    SUMMARY_PATH.write_text("\n".join(lines) + "\n")
    print(f"[summary] -> {SUMMARY_PATH}")
    print(f"COUNTS: markets={len(markets)} events={len(rows)} tier1={len(t1)} "
          f"close={n_close} t2h={n_t2h} both={n_both} suspicious={len(suspicious)}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    CANDLE_DIR.mkdir(parents=True, exist_ok=True)
    markets = pull_markets()
    rows, tier1, anomalies, suspicious = build(markets)
    write_summary(markets, rows, tier1, anomalies, suspicious)


if __name__ == "__main__":
    main()
