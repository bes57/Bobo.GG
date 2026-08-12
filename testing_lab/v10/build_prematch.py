"""Build an honest pre-match Kalshi dataset: bid/ask at T-2h from the REAL
scheduled match start, not from market close.

The shipped kalshi_matches.csv anchors everything to close_time, and these
markets close when a winner is declared -- so 74% of its "t2h" prices are taken
mid-match (see v10_timing_artifact.json). This rebuilds from the raw candles
using the scheduled start encoded in the event ticker, which agrees with the
market's own rules text on every event.

Keeps bid AND ask, so a backtest can pay the spread instead of pretending the
mid is transactable.

Writes testing_lab/v10/stats/prematch_book.csv
"""
import csv
import datetime as dt
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
KD = os.path.join(os.path.dirname(HERE), "kalshi")
OUT = os.path.join(HERE, "stats")

MONS = dict(JAN=1, FEB=2, MAR=3, APR=4, MAY=5, JUN=6, JUL=7, AUG=8,
            SEP=9, OCT=10, NOV=11, DEC=12)


def start_utc(ticker):
    """Scheduled start from the ticker (ET), returned as a UTC timestamp."""
    m = re.match(r"KXVALORANTGAME-(\d{2})([A-Z]{3})(\d{2})(\d{4})", ticker)
    if not m:
        return None
    yy, mon, dd, hhmm = m.groups()
    try:
        et = dt.datetime(2000 + int(yy), MONS[mon], int(dd),
                         int(hhmm[:2]), int(hhmm[2:]))
    except Exception:
        return None
    return int((et + dt.timedelta(hours=4)).replace(
        tzinfo=dt.timezone.utc).timestamp())          # ET -> UTC (summer)


def book_at(cands, target_ts, tol=3600):
    """Closest candle to target with a two-sided book, within tol seconds."""
    best, bd = None, None
    for c in cands:
        ts = c["end_period_ts"]
        d = abs(ts - target_ts)
        if d > tol:
            continue
        try:
            b = float(c["yes_bid"]["close_dollars"])
            a = float(c["yes_ask"]["close_dollars"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (0 < b < 1 and 0 < a < 1 and a >= b):
            continue
        if bd is None or d < bd:
            best, bd = (b, a, ts), d
    return best


meta = {}
with open(os.path.join(KD, "kalshi_matches.csv"), newline="") as f:
    for r in csv.DictReader(f):
        meta[r["event_ticker"]] = r

# market_ticker -> the team that market's YES refers to. The candle FILENAME
# suffix is Kalshi's own short code (DRX, KRU, GM, TYLOO) and does NOT equal the
# BenPom org (KRX, KRU-with-umlaut, M8, TYL) -- matching on it silently scored
# both legs of 37 of 253 events as losses. yes_sub_title is the full team name,
# which maps cleanly onto team_a_raw / team_b_raw in the CSV.
yes_team = {}
with open(os.path.join(KD, "markets_raw.jsonl")) as f:
    for line in f:
        try:
            m = json.loads(line)
        except ValueError:
            continue
        if m.get("ticker") and m.get("yes_sub_title"):
            yes_team[m["ticker"]] = m["yes_sub_title"].strip()

rows, unresolved = [], []
for path in sorted(glob.glob(os.path.join(KD, "candles", "*.json"))):
    base = os.path.basename(path)[:-5]
    if "-" not in base:
        continue
    ev, _suffix = base.rsplit("-", 1)
    mr = meta.get(ev)
    if not mr or mr.get("excluded") == "True" or not mr.get("winner_org"):
        continue
    st = start_utc(ev)
    if st is None:
        continue
    with open(path) as f:
        cs = json.load(f).get("candlesticks") or []
    # resolve which ORG this market's YES is, via the full team name
    tname = yes_team.get(base)
    if tname is None:
        continue
    if tname == (mr.get("team_a_raw") or "").strip():
        side = mr["org_a"]
    elif tname == (mr.get("team_b_raw") or "").strip():
        side = mr["org_b"]
    else:
        unresolved.append((base, tname))
        continue

    got = book_at(cs, st - 7200)
    got1 = book_at(cs, st - 3600)          # also T-1h, per the simpler spec
    if not got:
        continue
    bid, ask, ts = got
    b1, a1, ts1 = got1 if got1 else (None, None, None)
    rows.append({"event_ticker": ev, "side_org": side,
                 "bid_1h": b1, "ask_1h": a1,
                 "mid_1h": (round((b1 + a1) / 2, 4) if b1 is not None else None),
                 "offset_1h_min": (round((ts1 - st) / 60, 1) if ts1 else None),
                 "org_a": mr["org_a"], "org_b": mr["org_b"],
                 "winner_org": mr["winner_org"],
                 "start_utc": st, "book_ts": ts,
                 "offset_min": round((ts - st) / 60, 1),
                 "yes_bid": bid, "yes_ask": ask,
                 "mid": round((bid + ask) / 2, 4),
                 "spread": round(ask - bid, 4),
                 "volume_total": mr["volume_total"]})

os.makedirs(OUT, exist_ok=True)
p = os.path.join(OUT, "prematch_book.csv")
with open(p, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

if unresolved:
    print(f"  [warn] {len(unresolved)} market(s) whose team name matched neither "
          f"side; skipped: {unresolved[:3]}")
offs = [r["offset_min"] for r in rows]
print(f"wrote {p}: {len(rows)} market-sides, "
      f"{len({r['event_ticker'] for r in rows})} events")
print(f"  book taken a median {sorted(offs)[len(offs)//2]:.0f} min from start "
      f"(target -120); all within +/-60 min of target")
print(f"  median spread {sorted(r['spread'] for r in rows)[len(rows)//2]:.3f}")
