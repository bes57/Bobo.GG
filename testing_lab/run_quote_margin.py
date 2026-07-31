"""Optimal quoting margin: simulate the bot's maker strategy over the 168
clean events. For each margin rule, post NO bids on BOTH markets at
(model NO value - margin), fill iff a real trade printed at/through our level
inside [listing, start-2h], settle at match result.

Rules swept:
  flat m cents, m = 0..15
  logit-shift delta (edge demanded in logit space)
  sqrt-scaled k*sqrt(p(1-p)) cents
Verification: split-half by time, trade-through (conservative) fills,
excluding info-risk shapes, bootstrap CIs. Writes out/quote_margin.json."""
import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from engine import Engine

OUT = os.path.join(HERE, "out")

eng = Engine()
s = eng.series.reset_index(drop=True)
fmts = s.fmt.values
rd6 = np.load(os.path.join(OUT, "rd_v6_native.npy"))
b6 = json.load(open(os.path.join(OUT, "v6_native_beta.json")))["beta"]


def sp_scalar(pm, fm):
    if fm in ("bo5", "bo5_gf"):
        return pm ** 3 * (1 + 3 * (1 - pm) + 6 * (1 - pm) ** 2)
    if fm == "bo1":
        return pm
    return pm * pm * (3 - 2 * pm)


pm_all = 1 / (1 + np.exp(-b6 * rd6))
p6 = np.array([sp_scalar(pm_all[i], fmts[i]) if not np.isnan(rd6[i]) else np.nan
               for i in range(len(s))])
s["p6"] = p6

kj = pd.read_csv(os.path.join(OUT, "kalshi_joined5.csv"))
kj = kj.merge(s[["match_id", "p6"]], on="match_id").dropna(subset=["p6"])
mt = json.load(open(os.path.join(HERE, "..", "data", "match_times.json")))
mt.update(json.load(open(os.path.join(HERE, "data_patch", "patch_times.json"))))
raw = {}
with open(os.path.join(HERE, "kalshi", "markets_raw.jsonl")) as f:
    for line in f:
        m_ = json.loads(line)
        raw.setdefault(m_["event_ticker"], []).append(m_)
kmeta = pd.read_csv(os.path.join(HERE, "kalshi", "kalshi_matches.csv"))
kmeta_map = {r.event_ticker: r for r in kmeta.itertuples(index=False)}

# assemble per-market sim rows: (event, team, p_yes_model, won, trades[(ts, yes_price)])
markets = []
for r in kj.itertuples(index=False):
    t_real = mt.get(str(r.match_id))
    if not t_real:
        continue
    start = datetime.strptime(t_real, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    cutoff = int((start - timedelta(hours=2)).timestamp())
    cutoff5 = int((start - timedelta(minutes=5)).timestamp())
    km = kmeta_map.get(r.event_ticker)
    if km is None:
        continue
    # winner-referenced p6 -> per-org
    p_by_org = {r.winner: float(r.p6), r.loser: 1 - float(r.p6)}
    for mkt in raw.get(r.event_ticker, []):
        team_raw = mkt.get("no_sub_title", "")
        org = km.org_a if team_raw == km.team_a_raw else (
            km.org_b if team_raw == km.team_b_raw else None)
        if org is None or org not in p_by_org:
            continue
        path = os.path.join(HERE, "kalshi", "candles", f"{mkt['ticker']}.json")
        if not os.path.exists(path):
            continue
        candles = json.load(open(path)).get("candlesticks", [])
        trades = []
        for c in candles:
            pr = (c.get("price") or {}).get("close_dollars")
            if pr is not None and float(c.get("volume_fp") or 0) >= 0:
                if pr:
                    trades.append((c["end_period_ts"], float(pr)))
        markets.append({"event": r.event_ticker, "org": org,
                        "p_yes": p_by_org[org], "won": org == r.winner,
                        "date": r.date, "vct": bool(r.vct),
                        "div": abs(float(r.p6) - float(r.pk_pre)),
                        "mkt_conf_gap": (float(r.pk_pre) - float(r.p6))
                        if False else 0.0,
                        "cutoff": cutoff, "cutoff5": cutoff5,
                        "trades": trades})
print(f"sim markets: {len(markets)} across {len(set(m['event'] for m in markets))} events")


def simulate(rule, cutoff_key="cutoff", through=0.0, exclude=None):
    """rule(p_yes) -> NO bid price in [0.01, 0.97] or None.
    Fill iff any trade with yes_price >= 1 - b + through before cutoff."""
    cost = profit = fills = quoted = 0.0
    fills_list = []
    for m_ in markets:
        if exclude and exclude(m_):
            continue
        b = rule(m_["p_yes"])
        if b is None or b < 0.01:
            continue
        quoted += 1
        level = 1 - b + through
        filled = any(ts <= m_[cutoff_key] and pr >= level - 1e-9
                     for ts, pr in m_["trades"])
        if not filled:
            continue
        fills += 1
        pnl = (1 - b) if (not m_["won"]) else -b
        cost += b
        profit += pnl
        fills_list.append((pnl, b))
    roi = profit / cost if cost > 0 else 0.0
    return {"quoted": int(quoted), "fills": int(fills),
            "fill_rate": round(fills / max(quoted, 1), 3),
            "profit": round(profit, 2), "cost": round(cost, 2),
            "roi": round(roi, 4),
            "profit_per_event": round(profit / max(len(set(m_['event'] for m_ in markets)), 1), 4),
            "_fills": fills_list}


rng = np.random.default_rng(77)


def boot_roi(fl):
    if len(fl) < 5:
        return [None, None]
    pnl = np.array([x[0] for x in fl])
    c = np.array([x[1] for x in fl])
    n = len(fl)
    boots = [pnl[i].sum() / c[i].sum() for i in rng.integers(0, n, (2000, n))]
    return [round(float(np.percentile(boots, 2.5)), 3),
            round(float(np.percentile(boots, 97.5)), 3)]


res = {"n_markets": len(markets)}
print(f"\n{'rule':<22}{'fills':>6}{'fillrate':>9}{'profit':>8}{'ROI':>8}  CI")
best_flat = None
for m_c in range(0, 16):
    r = simulate(lambda p, m_c=m_c: (1 - p) - m_c / 100.0)
    ci = boot_roi(r["_fills"])
    res[f"flat_{m_c}c"] = {k: v for k, v in r.items() if k != "_fills"}
    res[f"flat_{m_c}c"]["ci"] = ci
    print(f"flat {m_c:>2}c              {r['fills']:>6}{r['fill_rate']:>9}"
          f"{r['profit']:>8.2f}{r['roi']:>8.1%}  {ci}")
    if best_flat is None or r["profit"] > best_flat[1]:
        best_flat = (m_c, r["profit"])
for dl in (0.1, 0.2, 0.3, 0.4, 0.6):
    def rule(p, dl=dl):
        lo = math.log(max(p, 1e-6) / max(1 - p, 1e-6))
        return 1 - 1 / (1 + math.exp(-(lo + dl)))
    r = simulate(rule)
    ci = boot_roi(r["_fills"])
    res[f"logit_{dl}"] = {k: v for k, v in r.items() if k != "_fills"}
    res[f"logit_{dl}"]["ci"] = ci
    print(f"logit +{dl:<4}           {r['fills']:>6}{r['fill_rate']:>9}"
          f"{r['profit']:>8.2f}{r['roi']:>8.1%}  {ci}")
for k_ in (8, 12, 16, 20):
    def rule(p, k_=k_):
        return (1 - p) - (k_ / 100.0) * math.sqrt(p * (1 - p))
    r = simulate(rule)
    ci = boot_roi(r["_fills"])
    res[f"sqrt_{k_}"] = {k: v for k, v in r.items() if k != "_fills"}
    res[f"sqrt_{k_}"]["ci"] = ci
    print(f"sqrt k={k_:<3}            {r['fills']:>6}{r['fill_rate']:>9}"
          f"{r['profit']:>8.2f}{r['roi']:>8.1%}  {ci}")

with open(os.path.join(OUT, "quote_margin.json"), "w") as f:
    json.dump(res, f, indent=1)
print("saved out/quote_margin.json (verification passes next)")
