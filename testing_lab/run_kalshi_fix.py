"""Rebuild Kalshi pre-match prices anchored on the SCHEDULED start time parsed
from the event ticker (ET wall clock -> UTC), not close_time. Extracts the
yes-mid at start-5m and start-60m from minute candles, then re-runs the
BenPom-vs-market comparison cleanly. Writes out/kalshi_compare2.json."""
import json
import os
import re
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from harness import (BETA_LIVE, brier, intl_attendance_asof, load_series,
                     logloss, paired_bootstrap, predict)

OUT = os.path.join(HERE, "out")
ET = ZoneInfo("US/Eastern")

TICK_RE = re.compile(r"KXVALORANTGAME-(\d\d)([A-Z]{3})(\d\d)(\d\d)(\d\d)")
MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}


def start_utc(event_ticker):
    m = TICK_RE.match(event_ticker)
    if not m:
        return None
    yy, mon, dd, hh, mi = m.groups()
    dt = datetime(2000 + int(yy), MONTHS[mon], int(dd), int(hh), int(mi), tzinfo=ET)
    return dt.astimezone(ZoneInfo("UTC"))


def yes_mid_at(candles, ts_target, max_lookback_min=180):
    """Last minute-candle at/before ts_target with a sane book; returns mid."""
    best = None
    for c in candles:
        if c["end_period_ts"] <= ts_target:
            best = c if best is None or c["end_period_ts"] > best["end_period_ts"] else best
        # candles sorted; could break, but files are small enough
    if best is None or ts_target - best["end_period_ts"] > max_lookback_min * 60:
        return None
    try:
        bid = float(best["yes_bid"]["close_dollars"])
        ask = float(best["yes_ask"]["close_dollars"])
    except Exception:
        return None
    if ask - bid > 0.30 or (bid <= 0.0 and ask >= 1.0):
        pr = best.get("price") or {}
        for k in ("close_dollars", "previous_dollars"):
            if pr.get(k):
                return float(pr[k])
        return None
    return (bid + ask) / 2.0


k = pd.read_csv(os.path.join(HERE, "kalshi", "kalshi_matches.csv"))
k = k[(k.excluded != True) & k.winner_org.notna()].copy()  # noqa: E712
raw = {}
with open(os.path.join(HERE, "kalshi", "markets_raw.jsonl")) as f:
    for line in f:
        mkt = json.loads(line)
        raw.setdefault(mkt["event_ticker"], []).append(mkt)

rows = []
for _, kr in k.iterrows():
    su = start_utc(kr.event_ticker)
    if su is None:
        continue
    mkts = raw.get(kr.event_ticker, [])
    ps = {}   # org -> (p_start, p_1h)
    for mkt in mkts:
        team = mkt.get("no_sub_title", "")
        path = os.path.join(HERE, "kalshi", "candles", f"{mkt['ticker']}.json")
        if not os.path.exists(path):
            continue
        candles = json.load(open(path)).get("candlesticks", [])
        t0 = int(su.timestamp())
        p_st = yes_mid_at(candles, t0 - 300)
        p_1h = yes_mid_at(candles, t0 - 3600)
        yes_org = None
        # market's yes side is the team in no_sub_title? no: yes = team wins.
        # kalshi_matches has org_a/org_b + raw names; match by raw name.
        if team == kr.team_a_raw:
            yes_org = kr.org_a
        elif team == kr.team_b_raw:
            yes_org = kr.org_b
        if yes_org:
            ps[yes_org] = (p_st, p_1h)

    def combine(idx):
        a = ps.get(kr.org_a, (None, None))[idx]
        b = ps.get(kr.org_b, (None, None))[idx]
        vals = []
        if a is not None:
            vals.append(a)
        if b is not None:
            vals.append(1.0 - b)
        return float(np.mean(vals)) if vals else np.nan

    rows.append({"event_ticker": kr.event_ticker, "start_utc": su.isoformat(),
                 "org_a": kr.org_a, "org_b": kr.org_b,
                 "winner_org": kr.winner_org, "volume": kr.volume_total,
                 "pa_start": combine(0), "pa_1h": combine(1)})

kk = pd.DataFrame(rows)
print(f"events with parsed start: {len(kk)}; "
      f"pa_start coverage: {kk.pa_start.notna().sum()}; "
      f"pa_1h: {kk.pa_1h.notna().sum()}")

# join to BenPom walk-forward
s = load_series()
att = intl_attendance_asof(s)
s["p_prod"] = predict(s, beta=BETA_LIVE, gating="backend", attendance=att)
s26 = s[s.year == 2026].copy()
s26["pair"] = [frozenset((a, b)) for a, b in zip(s26.winner, s26.loser)]
s26["d"] = pd.to_datetime(s26.date)
kk["pair"] = [frozenset((a, b)) for a, b in zip(kk.org_a, kk.org_b)]
kk["d"] = pd.to_datetime(kk.start_utc.str[:10])

rows = []
unjoined = []
for _, kr in kk.iterrows():
    cand = s26[(s26.pair == kr["pair"]) & (abs((s26.d - kr.d).dt.days) <= 1)]
    if len(cand) == 0:
        unjoined.append(kr.event_ticker)
        continue
    sr = cand.iloc[0]

    def wp(p_a):
        return np.nan if pd.isna(p_a) else (p_a if kr.winner_org == kr.org_a else 1 - p_a)
    rows.append({"event_ticker": kr.event_ticker, "date": sr.date,
                 "event_id": sr.event_id, "stage": sr.stage, "fmt": sr.fmt,
                 "winner": sr.winner, "loser": sr.loser,
                 "p_benpom": float(sr.p_prod), "pk_start": wp(kr.pa_start),
                 "pk_1h": wp(kr.pa_1h), "volume": float(kr.volume)})
m = pd.DataFrame(rows).dropna(subset=["pk_start"])
res = {"n_joined": len(m), "n_unjoined": len(unjoined),
       "unjoined_sample": unjoined[:10]}
print(f"joined: {len(m)}  unjoined: {len(unjoined)} (sample: {unjoined[:6]})")

for c in ("pk_start", "pk_1h"):
    m[c] = m[c].clip(0.01, 0.99)

res["benpom"] = {"logloss": round(logloss(m.p_benpom.values), 5),
                 "brier": round(brier(m.p_benpom.values), 5)}
res["kalshi_start"] = {"logloss": round(logloss(m.pk_start.values), 5),
                       "brier": round(brier(m.pk_start.values), 5)}
m1 = m.dropna(subset=["pk_1h"])
res["kalshi_1h"] = {"n": len(m1), "logloss": round(logloss(m1.pk_1h.values), 5),
                    "brier": round(brier(m1.pk_1h.values), 5)}
res["boot_kalshi_start_vs_benpom"] = paired_bootstrap(
    m.pk_start.values, m.p_benpom.values)
print("\nBenPom:", res["benpom"])
print("Kalshi @start-5m:", res["kalshi_start"])
print("Kalshi @start-60m:", res["kalshi_1h"])
print("boot (kalshi better?):", res["boot_kalshi_start_vs_benpom"])

med = m.volume.median()
for lab, mask in [("high_vol", m.volume >= med), ("low_vol", m.volume < med)]:
    res[lab] = {"n": int(mask.sum()),
                "benpom_ll": round(logloss(m.p_benpom[mask].values), 5),
                "kalshi_ll": round(logloss(m.pk_start[mask].values), 5)}
    print(lab, res[lab])

# favorite pricing comparison
pb_fav = np.maximum(m.p_benpom, 1 - m.p_benpom)
pk_same = np.where(m.p_benpom >= 0.5, m.pk_start, 1 - m.pk_start)
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
lk_ = np.log(m.pk_start / (1 - m.pk_start))


def bl(w):
    p = 1 / (1 + np.exp(-(w * lb_ + (1 - w) * lk_)))
    return -np.mean(np.log(np.clip(p, 1e-9, 1)))


r = minimize_scalar(bl, bounds=(0, 1), method="bounded")
res["blend_w_benpom"] = round(float(r.x), 4)
res["blend_ll"] = round(float(bl(r.x)), 5)
print(f"\nblend: w_benpom={res['blend_w_benpom']} ll={res['blend_ll']}")

m["div"] = (m.p_benpom - m.pk_start).abs()
res["top_divergences"] = m.nlargest(15, "div")[
    ["date", "winner", "loser", "stage", "event_id",
     "p_benpom", "pk_start", "div", "volume"]].to_dict("records")
print("\nTOP DIVERGENCES (pre-match, winner-referenced):")
for r_ in res["top_divergences"]:
    print(f"  {r_['date']} {r_['winner']} bt {r_['loser']} [{r_['stage']}]: "
          f"benpom {r_['p_benpom']:.2f} kalshi {r_['pk_start']:.2f}")

m.to_csv(os.path.join(OUT, "kalshi_joined2.csv"), index=False)
with open(os.path.join(OUT, "kalshi_compare2.json"), "w") as f:
    json.dump(res, f, indent=1, default=str)
print("\nsaved out/kalshi_compare2.json")
