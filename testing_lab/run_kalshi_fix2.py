"""Kalshi vs BenPom, take 3 — correct pre-match anchors.

Anchor priority: (1) real VLR start time (match_times.json, UTC) minus 5m,
(2) close_time - 4h (bo3) / 5h (bo5). Rows whose anchor lands within 30min of
close are dropped (match must last longer than that; anchor unreliable).
Join: pair + closest date, each series row used once.
Writes out/kalshi_compare3.json."""
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from harness import (BETA_LIVE, brier, intl_attendance_asof, load_series,
                     logloss, paired_bootstrap, predict)

OUT = os.path.join(HERE, "out")

mt = json.load(open(os.path.join(HERE, "..", "data", "match_times.json")))

k = pd.read_csv(os.path.join(HERE, "kalshi", "kalshi_matches.csv"))
k = k[(k.excluded != True) & k.winner_org.notna()].copy()  # noqa: E712
raw = {}
with open(os.path.join(HERE, "kalshi", "markets_raw.jsonl")) as f:
    for line in f:
        m_ = json.loads(line)
        raw.setdefault(m_["event_ticker"], []).append(m_)

s = load_series()
att = intl_attendance_asof(s)
s["p_prod"] = predict(s, beta=BETA_LIVE, gating="backend", attendance=att)
s26 = s[s.year == 2026].copy()
s26["pair"] = [frozenset((a, b)) for a, b in zip(s26.winner, s26.loser)]
s26["d"] = pd.to_datetime(s26.date)

k["pair"] = [frozenset((a, b)) for a, b in zip(k.org_a, k.org_b)]
k["close_dt"] = pd.to_datetime(k.date_utc, utc=True)

# unique closest-date join
used = set()
joins = []
for _, kr in k.sort_values("date_utc").iterrows():
    cand = s26[(s26.pair == kr["pair"]) & (~s26.match_id.isin(used))].copy()
    if len(cand) == 0:
        continue
    cand["dd"] = (cand.d - kr.close_dt.tz_localize(None).normalize()).abs()
    cand = cand[cand.dd <= pd.Timedelta(days=1)].sort_values("dd")
    if len(cand) == 0:
        continue
    sr = cand.iloc[0]
    used.add(sr.match_id)
    joins.append((kr, sr))
print(f"joined uniquely: {len(joins)}")


def yes_mid_at(candles, ts_target, max_lookback_min=240):
    best = None
    for c in candles:
        if c["end_period_ts"] <= ts_target:
            if best is None or c["end_period_ts"] > best["end_period_ts"]:
                best = c
    if best is None or ts_target - best["end_period_ts"] > max_lookback_min * 60:
        return None
    try:
        bid = float(best["yes_bid"]["close_dollars"])
        ask = float(best["yes_ask"]["close_dollars"])
    except Exception:
        bid, ask = 0.0, 1.0
    if (ask - bid) <= 0.30 and not (bid <= 0.0 and ask >= 1.0):
        return (bid + ask) / 2.0
    pr = best.get("price") or {}
    for key in ("close_dollars", "previous_dollars"):
        if pr.get(key):
            return float(pr[key])
    return None


rows = []
n_realtime, n_fallback, n_dropped = 0, 0, 0
for kr, sr in joins:
    t_real = mt.get(str(sr.match_id))
    if t_real:
        start = datetime.strptime(t_real, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        n_realtime += 1
    else:
        hrs = 5 if sr.fmt in ("bo5", "bo5_gf") else 4
        start = kr.close_dt.to_pydatetime() - timedelta(hours=hrs)
        n_fallback += 1
    if (kr.close_dt.to_pydatetime() - start) < timedelta(minutes=30):
        n_dropped += 1
        continue
    t0 = int(start.timestamp())
    ps = {}
    for mkt in raw.get(kr.event_ticker, []):
        team = mkt.get("no_sub_title", "")
        path = os.path.join(HERE, "kalshi", "candles", f"{mkt['ticker']}.json")
        if not os.path.exists(path):
            continue
        candles = json.load(open(path)).get("candlesticks", [])
        p5 = yes_mid_at(candles, t0 - 300)
        p60 = yes_mid_at(candles, t0 - 3600)
        org = kr.org_a if team == kr.team_a_raw else (
            kr.org_b if team == kr.team_b_raw else None)
        if org:
            ps[org] = (p5, p60)

    def comb(idx):
        vals = []
        a = ps.get(kr.org_a, (None, None))[idx]
        b = ps.get(kr.org_b, (None, None))[idx]
        if a is not None:
            vals.append(a)
        if b is not None:
            vals.append(1 - b)
        return float(np.mean(vals)) if vals else np.nan

    pa5, pa60 = comb(0), comb(1)

    def wp(p_a):
        return np.nan if pd.isna(p_a) else (
            p_a if kr.winner_org == kr.org_a else 1 - p_a)
    rows.append({"event_ticker": kr.event_ticker, "date": sr.date,
                 "match_id": sr.match_id, "event_id": sr.event_id,
                 "stage": sr.stage, "fmt": sr.fmt, "intl": bool(sr.intl),
                 "winner": sr.winner, "loser": sr.loser,
                 "anchor": "real" if t_real else "fallback",
                 "p_benpom": float(sr.p_prod),
                 "pk_pre": wp(pa5), "pk_pre60": wp(pa60),
                 "volume": float(kr.volume_total)})

m = pd.DataFrame(rows).dropna(subset=["pk_pre"])
print(f"anchors: real={n_realtime} fallback={n_fallback} dropped={n_dropped}; "
      f"final n={len(m)}")
res = {"n": len(m), "n_real_anchor": n_realtime, "n_fallback": n_fallback}

for c in ("pk_pre", "pk_pre60"):
    m[c] = m[c].clip(0.01, 0.99)

res["benpom"] = {"logloss": round(logloss(m.p_benpom.values), 5),
                 "brier": round(brier(m.p_benpom.values), 5)}
res["kalshi_pre"] = {"logloss": round(logloss(m.pk_pre.values), 5),
                     "brier": round(brier(m.pk_pre.values), 5)}
res["boot"] = paired_bootstrap(m.pk_pre.values, m.p_benpom.values)
print("\nBenPom:", res["benpom"])
print("Kalshi pre-match:", res["kalshi_pre"])
print("boot (kalshi better?):", res["boot"])

# real-anchor-only (cleanest subsample)
mr = m[m.anchor == "real"]
res["real_only"] = {
    "n": len(mr),
    "benpom_ll": round(logloss(mr.p_benpom.values), 5),
    "kalshi_ll": round(logloss(mr.pk_pre.values), 5)}
res["boot_real_only"] = paired_bootstrap(mr.pk_pre.values, mr.p_benpom.values)
print("real-anchor only:", res["real_only"], res["boot_real_only"])

med = m.volume.median()
for lab, mask in [("high_vol", m.volume >= med), ("low_vol", m.volume < med)]:
    res[lab] = {"n": int(mask.sum()),
                "benpom_ll": round(logloss(m.p_benpom[mask].values), 5),
                "kalshi_ll": round(logloss(m.pk_pre[mask].values), 5)}
    print(lab, res[lab])

pb_fav = np.maximum(m.p_benpom, 1 - m.p_benpom)
pk_same = np.where(m.p_benpom >= 0.5, m.pk_pre, 1 - m.pk_pre)
res["fav_shift"] = []
for lo, hi in [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]:
    msk = (pb_fav >= lo) & (pb_fav < hi)
    if msk.sum():
        res["fav_shift"].append({
            "band": f"[{lo},{hi})", "n": int(msk.sum()),
            "benpom": round(float(pb_fav[msk].mean()), 4),
            "kalshi": round(float(pk_same[msk].mean()), 4),
            "emp": round(float((m.p_benpom[msk] >= 0.5).mean()), 4)})
print("\nfav bands:")
for r in res["fav_shift"]:
    print(" ", r)

from scipy.optimize import minimize_scalar
lb_ = np.log(np.clip(m.p_benpom, 1e-6, 1-1e-6) / (1 - np.clip(m.p_benpom, 1e-6, 1-1e-6)))
lk_ = np.log(m.pk_pre / (1 - m.pk_pre))


def bl(w):
    p = 1 / (1 + np.exp(-(w * lb_ + (1 - w) * lk_)))
    return -np.mean(np.log(np.clip(p, 1e-9, 1)))


r = minimize_scalar(bl, bounds=(0, 1), method="bounded")
res["blend_w_benpom"] = round(float(r.x), 4)
res["blend_ll"] = round(float(bl(r.x)), 5)
print(f"blend: w_benpom={res['blend_w_benpom']} ll={res['blend_ll']}")

m["div"] = (m.p_benpom - m.pk_pre).abs()
res["top_divergences"] = m.nlargest(15, "div")[
    ["date", "winner", "loser", "stage", "event_id", "anchor",
     "p_benpom", "pk_pre", "div", "volume"]].to_dict("records")
print("\nTOP DIVERGENCES:")
for r_ in res["top_divergences"]:
    print(f"  {r_['date']} {r_['winner']} bt {r_['loser']} [{r_['stage']},"
          f"{r_['anchor']}]: benpom {r_['p_benpom']:.2f} kalshi {r_['pk_pre']:.2f}")

# who was right on divergences > 0.15?
big = m[m["div"] > 0.15]
kal_right = ((big.pk_pre > big.p_benpom)).sum()  # market higher on actual winner
res["div_gt15"] = {"n": int(len(big)), "kalshi_right": int(kal_right),
                   "benpom_right": int(len(big) - kal_right)}
print(f"\ndivergences>0.15: {len(big)} — market closer on {kal_right}, "
      f"benpom closer on {len(big)-kal_right}")

m.to_csv(os.path.join(OUT, "kalshi_joined3.csv"), index=False)
with open(os.path.join(OUT, "kalshi_compare3.json"), "w") as f:
    json.dump(res, f, indent=1, default=str)
print("saved out/kalshi_compare3.json")
