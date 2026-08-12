"""Does a 2025+2026-only corpus beat the full 2023-2026 one?

The v10 lab killed HARD year isolation (each season solved alone) because of the
January cold start: -24.98 milli-LL in Jan-Feb. This is the middle ground the
operator asked for -- keep a rolling two-season corpus, but do NOT reset at the
year boundary, so a 2026 solve still has all of 2025 behind it and never starts
from nothing.

Evaluation cannot use the v9 T1/T2 protocol: FIT1 is 2023-24, which this arm
does not have. So the honest split is beta fit on 2025, scored on 2026, paired
against v6 refit identically on the same rows.

Writes testing_lab/v10/stats/v10_corpus2025.json
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from engine import Engine            # noqa: E402
from harness import paired_bootstrap  # noqa: E402

OUT = os.path.join(HERE, "stats")
V6 = {"decay": {"kind": "games", "consistency": (20.0, 12.0)},
      "rd": {"power": 0.75, "scale": 2.5}, "roster_mode": "year",
      "roster_persistence": 0.3, "ridge": 0.5, "champ_mult": 2.0,
      "region_prior_ridge": 1.5}

ARMS = {
    "v6_full_corpus": dict(V6),
    "corpus_2025on":  dict(V6, corpus_from="2025-01-01"),
    "corpus_2024on":  dict(V6, corpus_from="2024-01-01"),   # bracketing control
}

eng = Engine()
s = eng.series.reset_index(drop=True)
fmts, date = s.fmt.values, s.date.values
FIT = (date >= "2025-01-01") & (date <= "2025-12-31")    # beta fit here
TEST = (date > "2025-12-31") & (date <= "2026-07-28")    # scored here
print(f"beta-fit rows (2025): {FIT.sum()}   scored rows (2026H1): {TEST.sum()}")

rd = {}
for n, cfg in ARMS.items():
    print(f"  solving {n} ...", flush=True)
    rd[n] = eng.run(cfg)["rdiff"]
common = np.logical_and.reduce([np.isfinite(v) for v in rd.values()])
print(f"  scoreable in every arm: {common.sum()}")


def p_vec(b, r, m):
    pm = 1 / (1 + np.exp(-b * r[m])); fm = fmts[m]
    return np.where(np.isin(fm, ("bo5", "bo5_gf")),
                    pm ** 3 * (1 + 3 * (1 - pm) + 6 * (1 - pm) ** 2),
                    np.where(fm == "bo1", pm, pm ** 2 * (3 - 2 * pm)))


def nll(b, r, m):
    return -np.mean(np.log(np.clip(p_vec(b, r, m), 1e-9, 1)))


def fit(r, m):
    return float(minimize_scalar(lambda x: nll(x, r, m), bounds=(0.001, 1.0),
                                 method="bounded", options={"xatol": 1e-6}).x)


res = {"fit_window": "2025", "score_window": "2026-01-01..2026-07-28",
       "n_fit": int((common & FIT).sum()), "n_score": int((common & TEST).sum()),
       "arms": {}}
b6 = fit(rd["v6_full_corpus"], common & FIT)
ll6 = nll(b6, rd["v6_full_corpus"], common & TEST)
p6 = p_vec(b6, rd["v6_full_corpus"], common & TEST)

for n in ARMS:
    b = fit(rd[n], common & FIT)
    ll = nll(b, rd[n], common & TEST)
    pa = p_vec(b, rd[n], common & TEST)
    bs = paired_bootstrap(pa, p6)
    res["arms"][n] = {"beta": round(b, 6), "ll_test": round(float(ll), 6),
                      "delta_milli_ll": round(float((ll6 - ll) * 1000), 3),
                      "acc_pct": round(float(100 * np.mean(pa > 0.5)), 1),
                      "bootstrap": {k: (round(float(v), 6) if isinstance(v, (int, float)) else v)
                                    for k, v in bs.items()}}
    print(f"  {n:16} beta {b:.4f}  ll {ll:.5f}  "
          f"delta {res['arms'][n]['delta_milli_ll']:+8.3f} m  "
          f"acc {res['arms'][n]['acc_pct']:.1f}%")

# where does it differ? by month of the scored window
res["by_month"] = []
months = np.array([d[:7] for d in date])
for mo in sorted(set(months[common & TEST])):
    m = common & TEST & (months == mo)
    if m.sum() < 15:
        continue
    a = nll(fit(rd["corpus_2025on"], common & FIT), rd["corpus_2025on"], m)
    v = nll(b6, rd["v6_full_corpus"], m)
    res["by_month"].append({"month": mo, "n": int(m.sum()),
                            "delta_milli": round(float((v - a) * 1000), 2)})

with open(os.path.join(OUT, "v10_corpus2025.json"), "w") as f:
    json.dump(res, f, indent=1)
print("\nby month (positive = 2025+ corpus better):")
for r in res["by_month"]:
    print(f"  {r['month']}  n={r['n']:4}  {r['delta_milli']:+8.2f} m")
print(f"\nwrote {OUT}/v10_corpus2025.json")
