"""v10 §9 — the v6 model we are sticking with, priced against Kalshi.

Reporting only. Market data is NEVER a fitting target and never a selection
signal (v8 standing rule 9 / v9 integrity.market_data). Nothing here tunes
anything; it reads the frozen v6 probabilities and asks what they would have
returned against the book at a >= 5c edge threshold.

Method
  - one row per Kalshi tier-1 VALORANT match with a settled winner
  - BenPom probability is the walk-forward pre-match number (never uses the
    result), taken from the same engine run the rest of this lab scores
  - market price is prob_a_t2h. WARNING (established 2026-08-12): this is
    T-2h from market CLOSE, and these markets close when a winner is
    declared -- so 74% of the sample is taken DURING the match, median 39
    min in. It is NOT a pre-match price and the ROI computed from it is not
    tradeable. See testing_lab/v10/stats/v10_timing_artifact.json and the
    Edge Lab section 3. prob_a_close is worse still: it is a settlement
    label (72% of values are >=0.99 or <=0.01), not a robustness check.
  - edge = p_benpom - p_market, in cents, on whichever side BenPom prefers
  - stake 1 contract per qualifying match, buy at the quoted price, settle
    at 100 or 0. ROI = total profit / total staked.

Writes testing_lab/v10/stats/v10_roi.json.
"""
import json
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

eng = Engine()
s = eng.series.reset_index(drop=True)
out = eng.run(dict(V6))
rdiff = out["rdiff"]
fmts = s.fmt.values
FIT1 = (s.date <= "2024-12-31").values
ok = np.isfinite(rdiff)


def p_vec(b, m):
    pm = 1 / (1 + np.exp(-b * rdiff[m])); fm = fmts[m]
    return np.where(np.isin(fm, ("bo5", "bo5_gf")),
                    pm ** 3 * (1 + 3 * (1 - pm) + 6 * (1 - pm) ** 2),
                    np.where(fm == "bo1", pm, pm ** 2 * (3 - 2 * pm)))


beta = float(minimize_scalar(
    lambda b: -np.mean(np.log(np.clip(p_vec(b, ok & FIT1), 1e-9, 1))),
    bounds=(0.001, 1.0), method="bounded").x)
print(f"v6 beta (fit on FIT1 only, never on market data): {beta:.6f}")

p_win = np.full(len(s), np.nan)
p_win[ok] = p_vec(beta, ok)          # prob the eventual WINNER wins (pre-match)
s = s.assign(p_benpom=p_win)

k = pd.read_csv(os.path.join(os.path.dirname(HERE), "kalshi", "kalshi_matches.csv"))
k = k[(k.excluded != True) & k.winner_org.notna()].copy()   # noqa: E712
k["pair"] = [frozenset((a, b)) for a, b in zip(k.org_a, k.org_b)]
k["d"] = pd.to_datetime(k.date_utc.str[:10])
s["pair"] = [frozenset((a, b)) for a, b in zip(s.winner, s.loser)]
s["d"] = pd.to_datetime(s.date)

rows = []
for _, kr in k.iterrows():
    cand = s[(s.pair == kr["pair"]) & (abs((s.d - kr.d).dt.days) <= 1)
             & s.p_benpom.notna()]
    if not len(cand):
        continue
    sr = cand.iloc[0]

    def wprob(pa):        # market prob of the eventual winner
        return np.nan if pd.isna(pa) else (pa if kr.winner_org == kr.org_a
                                           else 1.0 - pa)
    rows.append({"date": sr.date, "winner": sr.winner, "loser": sr.loser,
                 "fmt": sr.fmt, "volume": float(kr.volume_total),
                 "p_benpom": float(sr.p_benpom),
                 "pk_t2h": wprob(kr.prob_a_t2h),
                 "pk_close": wprob(kr.prob_a_close)})
m = pd.DataFrame(rows).dropna(subset=["pk_t2h"])
for c in ("pk_t2h", "pk_close"):
    m[c] = m[c].clip(0.01, 0.99)
print(f"joined Kalshi matches: {len(m)}  ({m.date.min()} .. {m.date.max()})")


def simulate(price_col, thresh_c):
    """Bet 1 contract wherever |edge| >= thresh cents, on BenPom's side."""
    pk, pb = m[price_col].values, m.p_benpom.values
    edge_win = (pb - pk) * 100.0          # cents of edge on the WINNER side
    bets = []
    for i in range(len(m)):
        if edge_win[i] >= thresh_c:       # BenPom likes the eventual winner
            cost = pk[i] * 100.0
            bets.append((cost, 100.0 - cost, True))
        elif -edge_win[i] >= thresh_c:    # BenPom likes the eventual loser
            cost = (1.0 - pk[i]) * 100.0
            bets.append((cost, -cost, False))
    if not bets:
        return None
    staked = sum(b[0] for b in bets)
    profit = sum(b[1] for b in bets)
    wins = sum(1 for b in bets if b[2])
    # bootstrap the ROI over bets
    rng = np.random.default_rng(20260812)
    pr = np.array([b[1] for b in bets]); st = np.array([b[0] for b in bets])
    idx = rng.integers(0, len(bets), size=(4000, len(bets)))
    boot = (pr[idx].sum(1) / st[idx].sum(1)) * 100
    return {"threshold_cents": thresh_c, "n_bets": len(bets),
            "n_won": wins, "hit_rate": round(100 * wins / len(bets), 1),
            "staked_cents": round(staked, 1), "profit_cents": round(profit, 1),
            "roi_pct": round(100 * profit / staked, 2),
            "roi_ci95": [round(float(np.percentile(boot, 2.5)), 2),
                         round(float(np.percentile(boot, 97.5)), 2)],
            "p_profitable": round(float((boot > 0).mean()), 3)}


res = {"beta": round(beta, 6), "n_matches": int(len(m)),
       "window": [str(m.date.min()), str(m.date.max())],
       "median_volume": int(m.volume.median()),
       "note": "reporting only; market data is never a fitting or selection target",
       "headline": None, "by_threshold": {"t2h": [], "close": []}}

for col, tag in (("pk_t2h", "t2h"), ("pk_close", "close")):
    for t in (0, 3, 5, 7, 10, 15):
        r = simulate(col, t)
        if r:
            res["by_threshold"][tag].append(r)

res["headline"] = next(r for r in res["by_threshold"]["t2h"]
                       if r["threshold_cents"] == 5)

# ── diagnostics that qualify the headline ────────────────────────────────────
# The model is systematically LESS confident than the book, so "bet where we
# disagree" is very nearly "always take the underdog". That makes the bets one
# directional wager, not N independent edges, and the naive per-bet bootstrap
# badly overstates the precision. Everything below exists to say so on the page.
pk, pb = m.pk_t2h.values, m.p_benpom.values
res["discrimination"] = {
    "benpom_fav_hit_pct": round(float(100 * (pb > 0.5).mean()), 1),
    "kalshi_fav_hit_pct": round(float(100 * (pk > 0.5).mean()), 1),
    "mean_p_on_winner_benpom": round(float(pb.mean()), 4),
    "mean_p_on_winner_kalshi": round(float(pk.mean()), 4),
}
side, cost, won = [], [], []
for i in range(len(m)):
    e = (pb[i] - pk[i]) * 100
    if e >= 5:
        side.append("underdog" if pk[i] < 0.5 else "favourite")
        cost.append(pk[i] * 100); won.append(True)
    elif -e >= 5:
        side.append("underdog" if (1 - pk[i]) < 0.5 else "favourite")
        cost.append((1 - pk[i]) * 100); won.append(False)
side, cost, won = np.array(side), np.array(cost), np.array(won)
res["by_side"] = {}
for sd in ("underdog", "favourite"):
    sel = side == sd
    if sel.sum():
        pr = np.where(won[sel], 100 - cost[sel], -cost[sel])
        res["by_side"][sd] = {"n": int(sel.sum()),
                              "hit_rate": round(float(100 * won[sel].mean()), 1),
                              "roi_pct": round(float(100 * pr.sum() / cost[sel].sum()), 2)}
# longshot sensitivity: drop anything bought under 10c
keep = cost > 10
pr = np.where(won[keep], 100 - cost[keep], -cost[keep])
res["ex_longshots"] = {"n": int(keep.sum()),
                       "hit_rate": round(float(100 * won[keep].mean()), 1),
                       "roi_pct": round(float(100 * pr.sum() / cost[keep].sum()), 2)}
# block bootstrap by match DATE, so a good/bad day is resampled as one unit
dates = m.date.values
bet_date = np.array([dates[i] for i in range(len(m))
                     if abs((pb[i] - pk[i]) * 100) >= 5])
pr_all = np.where(won, 100 - cost, -cost)
udates = np.unique(bet_date)
rng2 = np.random.default_rng(20260813)
bl = []
for _ in range(4000):
    pick = rng2.choice(udates, size=len(udates), replace=True)
    idx = np.concatenate([np.where(bet_date == d)[0] for d in pick])
    bl.append(100 * pr_all[idx].sum() / cost[idx].sum())
bl = np.array(bl)
res["headline"]["roi_ci95_block_by_day"] = [round(float(np.percentile(bl, 2.5)), 2),
                                            round(float(np.percentile(bl, 97.5)), 2)]
res["headline"]["p_profitable_block"] = round(float((bl > 0).mean()), 3)
res["prior_lab_benchmark"] = {
    "source": "testing_lab/out/kalshi_compare.json",
    "n": 86, "benpom_ll": 0.68052, "kalshi_t2h_ll": 0.66099,
    "finding": "Kalshi BETTER than BenPom by ~19.5 milli-LL (p_better 0.678)"}

# calibration of the model against the book, for context
ll_b = float(-np.mean(np.log(np.clip(m.p_benpom.values, 1e-9, 1))))
ll_k = float(-np.mean(np.log(np.clip(m.pk_t2h.values, 1e-9, 1))))
res["logloss"] = {"benpom": round(ll_b, 5), "kalshi_t2h": round(ll_k, 5),
                  "delta_milli": round((ll_k - ll_b) * 1000, 2)}

with open(os.path.join(OUT, "v10_roi.json"), "w") as f:
    json.dump(res, f, indent=1)

h = res["headline"]
print(f"\n>= 5c edge, T-2h price: {h['n_bets']} bets, hit {h['hit_rate']}%, "
      f"ROI {h['roi_pct']}%  CI {h['roi_ci95']}  p(profit) {h['p_profitable']}")
print(f"log-loss  BenPom {ll_b:.5f}  vs Kalshi {ll_k:.5f}  "
      f"({res['logloss']['delta_milli']:+.1f}m for BenPom)")
for r in res["by_threshold"]["t2h"]:
    print(f"  >={r['threshold_cents']:2}c  n={r['n_bets']:3}  hit {r['hit_rate']:5.1f}%  "
          f"ROI {r['roi_pct']:+7.2f}%  CI [{r['roi_ci95'][0]:+.1f}, {r['roi_ci95'][1]:+.1f}]")
print(f"\nwrote {OUT}/v10_roi.json")
