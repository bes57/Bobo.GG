"""Round 6 — crossovers on games-decay winner + prediction-layer ideas:
fine games-HL grid x margin power, champ mult, games-blends, saturating tanh
link, cold-start region priors, fitted cross-region offsets.
Writes out/experiments6.json."""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import Engine
from harness import paired_bootstrap

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
BASE = {"rd": {"power": 0.75, "scale": 2.5}, "roster_mode": "year",
        "roster_persistence": 0.3, "ridge": 0.5, "champ_mult": 2.0}

eng = Engine()
s = eng.series.reset_index(drop=True)
fmts = s.fmt.values
results, rdiffs, probs = {}, {}, {}
train_v = (s.date <= "2024-12-31").values
test_v = (s.date > "2024-12-31").values
y26 = (s.date > "2025-12-31").values


def p_series_vec(b, rd, mask):
    pm = 1 / (1 + np.exp(-b * rd[mask]))
    fm = fmts[mask]
    return np.where(np.isin(fm, ("bo5", "bo5_gf")),
                    pm ** 3 * (1 + 3 * (1 - pm) + 6 * (1 - pm) ** 2),
                    np.where(fm == "bo1", pm, pm ** 2 * (3 - 2 * pm)))


def fit_score(name, rdiff, valid, extra=None):
    from scipy.optimize import minimize_scalar
    train = valid & train_v
    test = valid & test_v

    def nll(b, mask):
        return -np.mean(np.log(np.clip(p_series_vec(b, rdiff, mask), 1e-9, 1)))

    b = float(minimize_scalar(lambda x: nll(x, train), bounds=(0.02, 0.6),
                              method="bounded").x)
    results[name] = {"beta": round(b, 4), "ll_test": round(float(nll(b, test)), 5),
                     "ll_26": round(float(nll(b, test & y26)), 5)}
    if extra:
        results[name].update(extra)
    rdiffs[name] = rdiff
    probs[name] = (b, valid)
    print(f"{name:<30} b={b:.3f} ll={results[name]['ll_test']:.5f} "
          f"26={results[name]['ll_26']:.5f}", flush=True)


def run(name, cfg):
    out = eng.run({**BASE, **cfg})
    fit_score(name, out["rdiff"], ~np.isnan(out["rdiff"]))
    return out


# fine grid: games HL x margin power
outs = {}
for hg in (14.0, 16.0, 18.0, 20.0):
    for pw in (0.75, 1.0):
        o = run(f"g{int(hg)}_p{pw}", {"decay": {"kind": "games", "hl_games": hg},
                                      "rd": {"power": pw, "scale": 2.5}})
        outs[(hg, pw)] = o

# champ mult under games decay
for cm in (1.0, 3.0):
    run(f"g16_cm{cm}", {"decay": {"kind": "games", "hl_games": 16.0},
                        "champ_mult": cm})

# games two-timescale
o_g8 = run("g8_ref", {"decay": {"kind": "games", "hl_games": 8.0}})
o_g32 = run("g32_ref", {"decay": {"kind": "games", "hl_games": 32.0}})
vboth = ~(np.isnan(o_g8["rdiff"]) | np.isnan(o_g32["rdiff"]))
for w in (0.3, 0.5):
    fit_score(f"gblend_w{w}", w * o_g8["rdiff"] + (1 - w) * o_g32["rdiff"], vboth)

# saturating tanh link on the best games config
best_key = min(results, key=lambda k: results[k]["ll_test"])
rd_best = rdiffs[best_key]
v_best = ~np.isnan(rd_best)
from scipy.optimize import minimize
def tanh_nll(params, mask):
    a, b = params
    if a <= 0.5 or b <= 0:
        return 10.0
    rd_eff = a * np.tanh(rd_best / a)
    pm = 1 / (1 + np.exp(-b * rd_eff[mask]))
    fm = fmts[mask]
    p = np.where(np.isin(fm, ("bo5", "bo5_gf")),
                 pm ** 3 * (1 + 3 * (1 - pm) + 6 * (1 - pm) ** 2),
                 np.where(fm == "bo1", pm, pm ** 2 * (3 - 2 * pm)))
    return -np.mean(np.log(np.clip(p, 1e-9, 1)))


r = minimize(tanh_nll, [8.0, 0.12], args=(v_best & train_v,), method="Nelder-Mead")
a_f, b_f = r.x
ll_tanh = float(tanh_nll(r.x, v_best & test_v))
results["tanh_link"] = {"a": round(float(a_f), 3), "beta": round(float(b_f), 4),
                        "ll_test": round(ll_tanh, 5),
                        "base": best_key}
print(f"tanh_link on {best_key}: a={a_f:.2f} b={b_f:.3f} ll={ll_tanh:.5f}")

# cold-start region prior (prediction layer): teams with rating==0 exactly and
# no prior games get their region's 25th-percentile rating instead
o_best = outs[(16.0, 0.75)] if (16.0, 0.75) in outs else None
daily_out = eng.run({**BASE, "decay": {"kind": "games", "hl_games": 16.0},
                     "daily_out": True})
daily = daily_out["daily_r"]
sys.path.insert(0, "/Users/benny_es1/VCTMM")
from vctmm.benpom.teams import ORG_REGIONS  # noqa: E402
first_game = {}
for org, rows_ in eng.team_game_rows.items():
    first_game[org] = min(eng.g_date[i] for i in rows_)
rd_cs = daily_out["rdiff"].copy()
n_fixed = 0
days_sorted = sorted(daily.keys())
import bisect
for i, row in enumerate(s.itertuples(index=False)):
    fixes = {}
    for org, sign in ((row.winner, 1.0), (row.loser, -1.0)):
        if first_game.get(org, "9999") < row.date:
            continue  # team had history
        j = bisect.bisect_left(days_sorted, row.date) - 1
        if j < 0:
            continue
        rv = daily[days_sorted[j]]
        reg = ORG_REGIONS.get(org)
        regs = [rv[eng.tidx[t]] for t in eng.teams
                if ORG_REGIONS.get(t) == reg and first_game.get(t, "9999") < row.date]
        if len(regs) >= 6:
            prior = float(np.percentile(regs, 25))
            fixes[sign] = prior
            n_fixed += 1
    if fixes and not np.isnan(rd_cs[i]):
        for sign, prior in fixes.items():
            rd_cs[i] += sign * prior  # team had rating 0 in rdiff; add prior
fit_score("coldstart_prior", rd_cs, ~np.isnan(rd_cs),
          extra={"n_fixed_sides": n_fixed})

# fitted cross-region offsets (walk-forward monthly): rdiff += d[regA]-d[regB]
rd_x = rdiffs[best_key].copy()
v_x = ~np.isnan(rd_x)
cross = (s.reg_w != s.reg_l).values
months = sorted({d[:7] for d in s.date})
deltas_by_month = {}
REGS = ["Americas", "EMEA", "Pacific", "CN"]
b_fixed = results[best_key]["beta"]
for mo in months:
    hist = v_x & cross & (s.date < mo + "-01").values
    if hist.sum() < 60:
        deltas_by_month[mo] = np.zeros(4)
        continue
    iw = s.reg_w.map({r_: i for i, r_ in enumerate(REGS)}).values
    il = s.reg_l.map({r_: i for i, r_ in enumerate(REGS)}).values

    def nll_d(d4):
        d4 = np.append(d4, 0.0)  # CN pinned (identifiability)
        adj = rd_x[hist] + d4[iw[hist]] - d4[il[hist]]
        pm = 1 / (1 + np.exp(-b_fixed * adj))
        fm = fmts[hist]
        p = np.where(np.isin(fm, ("bo5", "bo5_gf")),
                     pm ** 3 * (1 + 3 * (1 - pm) + 6 * (1 - pm) ** 2),
                     np.where(fm == "bo1", pm, pm ** 2 * (3 - 2 * pm)))
        return -np.mean(np.log(np.clip(p, 1e-9, 1)))

    rr = minimize(nll_d, np.zeros(3), method="Nelder-Mead")
    deltas_by_month[mo] = np.append(rr.x, 0.0)
iw = s.reg_w.map({r_: i for i, r_ in enumerate(REGS)}).fillna(0).astype(int).values
il = s.reg_l.map({r_: i for i, r_ in enumerate(REGS)}).fillna(0).astype(int).values
rd_adj = rd_x.copy()
for i in np.where(v_x & cross)[0]:
    d4 = deltas_by_month[s.date.iloc[i][:7]]
    rd_adj[i] = rd_x[i] + d4[iw[i]] - d4[il[i]]
fit_score("xregion_offsets", rd_adj, v_x)
res_x_cross_only = None
# cross-region-only comparison
tm = v_x & test_v & cross
bb, _ = probs["xregion_offsets"][0], None
ll_x = float(-np.mean(np.log(np.clip(p_series_vec(probs['xregion_offsets'][0], rd_adj, tm), 1e-9, 1))))
ll_b = float(-np.mean(np.log(np.clip(p_series_vec(probs[best_key][0], rd_x, tm), 1e-9, 1))))
results["xregion_offsets"]["ll_cross_only"] = round(ll_x, 5)
results["xregion_offsets"]["base_cross_only"] = round(ll_b, 5)
print(f"cross-only: base {ll_b:.5f} -> offsets {ll_x:.5f} (n={int(tm.sum())})")

# bootstraps: winner vs calendar-13 candidate and vs production
exp5 = np.load(os.path.join(OUT, "exp5_rdiffs.npz"))
for ref_name, ref_key, ref_b in [("hl13", "ref_hl13_cand", 0.116),
                                 ("prod", "ref_prod_hl6_pow05", 0.177)]:
    rd_ref = exp5[ref_key]
    vv = ~np.isnan(rd_best) & ~np.isnan(rd_ref) & test_v
    from scipy.optimize import minimize_scalar
    def nll_ref(b):
        tr = ~np.isnan(rd_ref) & train_v
        return -np.mean(np.log(np.clip(p_series_vec(b, rd_ref, tr), 1e-9, 1)))
    b_ref = float(minimize_scalar(nll_ref, bounds=(0.02, 0.6), method="bounded").x)
    pa = p_series_vec(probs[best_key][0], rd_best, vv)
    pb = p_series_vec(b_ref, rd_ref, vv)
    results[f"boot_{best_key}_vs_{ref_name}"] = paired_bootstrap(pa, pb)
    print(f"boot {best_key} vs {ref_name}:", results[f"boot_{best_key}_vs_{ref_name}"])

lb = sorted(((k, v) for k, v in results.items() if "ll_test" in v),
            key=lambda kv: kv[1]["ll_test"])
print("\n== TOP 12 ==")
for name, r_ in lb[:12]:
    print(f"  {r_['ll_test']:.5f}  {name}")

with open(os.path.join(OUT, "experiments6.json"), "w") as f:
    json.dump(results, f, indent=1, default=str)
np.savez_compressed(os.path.join(OUT, "exp6_rdiffs.npz"), **rdiffs)
print("saved out/experiments6.json")
