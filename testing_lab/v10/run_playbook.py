"""A simple, honestly-backtested playbook.

Rule shape, deliberately one sentence:

    Two hours before a match, buy the side whose BenPom probability exceeds the
    Kalshi ASK by at least T cents. One contract. Hold to settlement.

Every discipline the earlier reads were missing is applied here:

  * price is the book at T-2h from the REAL scheduled start (prematch_book.csv),
    never close-anchored, so nothing is sampled mid-match
  * you pay the ASK, not the mid
  * Kalshi's taker fee ceil(0.07*C*P*(1-P)) is charged on every ticket
  * BOTH sides of every match are evaluated, so the rule never sees the result
  * T is chosen on the FIRST half of the window and scored on the SECOND,
    so the headline is out-of-sample rather than a swept maximum

Writes testing_lab/v10/stats/v10_playbook.json
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from engine import Engine   # noqa: E402

OUT = os.path.join(HERE, "stats")
V6 = {"decay": {"kind": "games", "consistency": (20.0, 12.0)},
      "rd": {"power": 0.75, "scale": 2.5}, "roster_mode": "year",
      "roster_persistence": 0.3, "ridge": 0.5, "champ_mult": 2.0,
      "region_prior_ridge": 1.5}
RNG = np.random.default_rng(20260812)


def fee_cents(price_dollars):
    """Kalshi taker fee, per contract, in cents."""
    return math.ceil(7.0 * price_dollars * (1.0 - price_dollars))


# ── model probabilities, walk-forward ────────────────────────────────────────
eng = Engine()
s = eng.series.reset_index(drop=True)
rdiff = eng.run(dict(V6))["rdiff"]
fmts, FIT1 = s.fmt.values, (s.date <= "2024-12-31").values
ok = np.isfinite(rdiff)


def p_vec(b, m):
    pm = 1 / (1 + np.exp(-b * rdiff[m])); fm = fmts[m]
    return np.where(np.isin(fm, ("bo5", "bo5_gf")),
                    pm ** 3 * (1 + 3 * (1 - pm) + 6 * (1 - pm) ** 2),
                    np.where(fm == "bo1", pm, pm ** 2 * (3 - 2 * pm)))


beta = float(minimize_scalar(
    lambda b: -np.mean(np.log(np.clip(p_vec(b, ok & FIT1), 1e-9, 1))),
    bounds=(0.001, 1.0), method="bounded").x)
pw = np.full(len(s), np.nan); pw[ok] = p_vec(beta, ok)
s = s.assign(pw=pw)
s["pair"] = [frozenset((a, b)) for a, b in zip(s.winner, s.loser)]
s["d"] = pd.to_datetime(s.date)

# ── the honest pre-match book ────────────────────────────────────────────────
bk = pd.read_csv(os.path.join(OUT, "prematch_book.csv"))
bk["d"] = pd.to_datetime(bk.start_utc, unit="s").dt.normalize()
bk["pair"] = [frozenset((a, b)) for a, b in zip(bk.org_a, bk.org_b)]

legs = []
for _, r in bk.iterrows():
    c = s[(s.pair == r["pair"]) & (abs((s.d - r.d).dt.days) <= 1) & s.pw.notna()]
    if not len(c):
        continue
    sr = c.iloc[0]
    won = (r.side_org == r.winner_org)
    p_model = float(sr.pw) if won else 1.0 - float(sr.pw)
    legs.append({"date": sr.date, "event": r.event_ticker, "side": r.side_org,
                 "won": bool(won), "p_model": p_model,
                 "ask": float(r.yes_ask), "mid": float(r.mid),
                 "bid": float(r.yes_bid),
                 "spread": float(r.spread), "fmt": sr.fmt,
                 "vol": float(r.volume_total)})
L = pd.DataFrame(legs)
print(f"legs (both sides of each match): {len(L)} over "
      f"{L.event.nunique()} matches, {L.date.min()} .. {L.date.max()}")

cut = sorted(L.date.unique())[len(L.date.unique()) // 2]
train, test = L[L.date <= cut], L[L.date > cut]
print(f"train {len(train)} legs (<= {cut})   test {len(test)} legs")


def run(df, T, price="ask", fees=True):
    e = (df.p_model * 100) - (df[price] * 100)
    sel = e >= T
    if sel.sum() == 0:
        return None
    cost = df[price][sel].values * 100
    if fees:
        cost = cost + np.array([fee_cents(p) for p in df[price][sel].values])
    won = df.won[sel].values
    prof = np.where(won, 100 - cost, -cost)
    roi = 100 * prof.sum() / cost.sum()
    idx = RNG.integers(0, len(cost), size=(4000, len(cost)))
    bs = 100 * prof[idx].sum(1) / cost[idx].sum(1)
    return {"T": int(T), "n": int(sel.sum()), "hit": round(float(100 * won.mean()), 1),
            "roi": round(float(roi), 2),
            "ci": [round(float(np.percentile(bs, 2.5)), 2),
                   round(float(np.percentile(bs, 97.5)), 2)],
            "p_profit": round(float((bs > 0).mean()), 3),
            "staked": round(float(cost.sum()), 1),
            "profit": round(float(prof.sum()), 1)}


GRID = list(range(0, 26, 1))
tr = [r for r in (run(train, T) for T in GRID) if r and r["n"] >= 15]
best = max(tr, key=lambda r: r["roi"])
T = best["T"]
print(f"\nthreshold chosen on TRAIN only: T = {T}c "
      f"(train ROI {best['roi']:+.2f}%, n={best['n']})")

res = {"beta": round(beta, 6), "n_legs": int(len(L)),
       "n_matches": int(L.event.nunique()),
       "window": [str(L.date.min()), str(L.date.max())],
       "split_date": str(cut),
       "chosen_T": T,
       "train_curve": tr,
       "train_at_T": best,
       "test_at_T": run(test, T),
       "full_at_T": run(L, T),
       "test_curve": [r for r in (run(test, t) for t in GRID) if r and r["n"] >= 10],
       "no_fee_test": run(test, T, fees=False),
       "mid_no_fee_test": run(test, T, price="mid", fees=False)}

# a flat "always bet the model's side" reference, no threshold at all
res["reference_T0"] = {"train": run(train, 0), "test": run(test, 0)}

# MAKER bound: instead of lifting the ask you post a bid and wait to be hit.
# Optimistic in two ways -- it assumes you always get filled, and maker fees on
# this series really are zero -- so treat it as a ceiling, not a forecast.
res["maker_bound"] = {}
for tag, df in (("train", train), ("test", test)):
    e = (df.p_model * 100) - (df["bid"] * 100)
    sel = e >= T
    if sel.sum():
        cost = df["bid"][sel].values * 100          # filled at your own bid, no fee
        won = df.won[sel].values
        prof = np.where(won, 100 - cost, -cost)
        idx = RNG.integers(0, len(cost), size=(4000, len(cost)))
        bs = 100 * prof[idx].sum(1) / cost[idx].sum(1)
        res["maker_bound"][tag] = {
            "T": int(T), "n": int(sel.sum()),
            "hit": round(float(100 * won.mean()), 1),
            "roi": round(float(100 * prof.sum() / cost.sum()), 2),
            "ci": [round(float(np.percentile(bs, 2.5)), 2),
                   round(float(np.percentile(bs, 97.5)), 2)],
            "p_profit": round(float((bs > 0).mean()), 3)}

with open(os.path.join(OUT, "v10_playbook.json"), "w") as f:
    json.dump(res, f, indent=1)

print("\n== OUT-OF-SAMPLE RESULT ==")
t = res["test_at_T"]
print(f"  T={T}c on unseen half: n={t['n']}  hit {t['hit']}%  "
      f"ROI {t['roi']:+.2f}%  CI [{t['ci'][0]:+.1f}, {t['ci'][1]:+.1f}]  "
      f"p(profit) {t['p_profit']}")
print("\n  same bets without the fee   :", res["no_fee_test"]["roi"], "%")
print("  same bets at the mid, no fee:", res["mid_no_fee_test"]["roi"], "%")
print("\n== test curve (for the record, NOT for choosing) ==")
for r in res["test_curve"]:
    print(f"    T={r['T']:2}c  n={r['n']:3}  hit {r['hit']:5.1f}%  "
          f"ROI {r['roi']:+7.2f}%  CI [{r['ci'][0]:+.0f},{r['ci'][1]:+.0f}]")
print(f"\nwrote {OUT}/v10_playbook.json")
