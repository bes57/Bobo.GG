"""Round 8a — decay functional forms in games-space:
exp vs power(heavy tail) vs boxexp, calendar envelopes on games decay,
win/loss asymmetric decay. All on the v2 base (pow0.75, roster 0.3, po x1.6).
Writes out/experiments8.json."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import Engine
from harness import paired_bootstrap

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

eng = Engine()
s = eng.series.reset_index(drop=True)
fmts = s.fmt.values
train_v = (s.date <= "2024-12-31").values
test_v = (s.date > "2024-12-31").values

stage_by_mid = dict(zip(s.match_id, s.stage))
g_stage = np.array([stage_by_mid.get(g["match_id"], "groups") for g in eng.games])
WC = np.where(np.isin(g_stage, ("playoffs", "grand_final")), 1.6, 1.0)
BASE = {"rd": {"power": 0.75, "scale": 2.5}, "roster_mode": "year",
        "roster_persistence": 0.3, "ridge": 0.5, "champ_mult": 2.0,
        "w_custom": WC}

results, rdiffs = {}, {}


def fit_score(name, rdiff, valid):
    from scipy.optimize import minimize_scalar

    def pv(b, mask):
        pm = 1 / (1 + np.exp(-b * rdiff[mask]))
        fm = fmts[mask]
        return np.where(np.isin(fm, ("bo5", "bo5_gf")),
                        pm ** 3 * (1 + 3 * (1 - pm) + 6 * (1 - pm) ** 2),
                        np.where(fm == "bo1", pm, pm ** 2 * (3 - 2 * pm)))

    def nll(b, mask):
        return -np.mean(np.log(np.clip(pv(b, mask), 1e-9, 1)))

    b = float(__import__("scipy.optimize", fromlist=["minimize_scalar"])
              .minimize_scalar(lambda x: nll(x, valid & train_v),
                               bounds=(0.02, 0.6), method="bounded").x)
    results[name] = {"beta": round(b, 4),
                     "ll_test": round(float(nll(b, valid & test_v)), 5)}
    rdiffs[name] = (rdiff, b, valid)
    print(f"{name:<30} b={b:.3f} ll={results[name]['ll_test']:.5f}", flush=True)


def run(name, dcfg):
    out = eng.run({**BASE, "decay": dcfg})
    fit_score(name, out["rdiff"], ~np.isnan(out["rdiff"]))


run("g16_exp (ref)", {"kind": "games", "hl_games": 16.0})
# power heavy-tail in games-space
for tau, al in ((6.0, 1.2), (8.0, 1.5), (10.0, 1.0), (12.0, 1.5)):
    run(f"gpower_t{int(tau)}_a{al}", {"kind": "games", "form": "power",
                                      "tau": tau, "alpha": al})
# boxexp in games-space: recent c games full weight
for c, hl in ((4.0, 12.0), (6.0, 12.0), (8.0, 16.0)):
    run(f"gbox_c{int(c)}_hl{int(hl)}", {"kind": "games", "form": "boxexp",
                                        "c": c, "hl_games": hl})
# calendar envelope over games decay (very-old info still fades)
for env in (30.0, 45.0, 65.0):
    run(f"g16_env{int(env)}w", {"kind": "games", "hl_games": 16.0,
                                "cal_env_hl": env})
# win/loss asymmetric games decay
for hlw, hll in ((20.0, 12.0), (12.0, 20.0), (16.0, 10.0)):
    run(f"gasym_w{int(hlw)}_l{int(hll)}", {"kind": "games", "hl_games": hlw,
                                           "hl_games_loss": hll})

lb = sorted(results.items(), key=lambda kv: kv[1]["ll_test"])
print("\n== LEADERBOARD ==")
for name, r in lb:
    print(f"  {r['ll_test']:.5f}  {name}")

# bootstrap top vs ref
ref_rd, ref_b, ref_v = rdiffs["g16_exp (ref)"]
top = lb[0][0]
if top != "g16_exp (ref)":
    rd_a, b_a, v_a = rdiffs[top]
    vv = v_a & ref_v & test_v

    def pv(rd, b, mask):
        pm = 1 / (1 + np.exp(-b * rd[mask]))
        fm = fmts[mask]
        return np.where(np.isin(fm, ("bo5", "bo5_gf")),
                        pm ** 3 * (1 + 3 * (1 - pm) + 6 * (1 - pm) ** 2),
                        np.where(fm == "bo1", pm, pm ** 2 * (3 - 2 * pm)))
    results["_boot_top_vs_ref"] = {"top": top,
                                   **paired_bootstrap(pv(rd_a, b_a, vv),
                                                      pv(ref_rd, ref_b, vv))}
    print("boot top vs g16_exp:", results["_boot_top_vs_ref"])

with open(os.path.join(OUT, "experiments8.json"), "w") as f:
    json.dump(results, f, indent=1)
np.savez_compressed(os.path.join(OUT, "exp8_rdiffs.npz"),
                    **{k: v[0] for k, v in rdiffs.items()})
print("saved out/experiments8.json")
