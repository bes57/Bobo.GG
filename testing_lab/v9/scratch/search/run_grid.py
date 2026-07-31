"""v9 search — fixtures S1-S4, then the 3240-config FIT1 grid.

Preregistered: v9/preregister.search.md sections 1a, 2, 6 (LOCKED first).
Scoring touches FIT1 rows ONLY — no VAL-row aggregate is computed anywhere
in this script. Output: stats/v9_search_grid.json + scratch cache of the
per-(policy,tau) state tables for the nomination step.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import searchlib as sl  # noqa: E402
from searchlib import OverlaySearch, direction_of  # noqa: E402
import v9lib  # noqa: E402
import runner  # noqa: E402

V9 = "/Users/benny_es1/PythonTest/testing_lab/v9"
LOG = os.path.join(V9, "logs", "search.log")
t0 = time.time()


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


frame = sl.load_frame_checked()
W = sl.windows(frame)
fmts = frame.fmt.values
event_ids = frame.event_id.values
years = frame.year.values

log(f"grid start — frame sha OK, windows OK [{time.time()-t0:.0f}s]")

corpus = v9lib.load_corpus()
base = v9lib.v6_run()
valid = base["valid"]
log(f"v6 base run: beta={base['beta']} valid={int(valid.sum())}/{len(frame)} "
    f"[{time.time()-t0:.0f}s]")

POLICIES = {"p1w3c5": ("p1", 3), "p1w5c5": ("p1", 5),
            "p1w8c5": ("p1", 8), "p3w5c5": ("p3", 5)}
B_GRID = [0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 0.65, 0.90, 1.20]
TAU_GRID = [2.0, 3.0, 5.0, 8.0, 13.0, 21.0]
GAMMA_GRID = [0.5, 1.0, 2.0]
VARIANT_GRID = [("delta2", 0), ("delta1", 0), ("delta1", 3),
                ("hybrid", 3), ("hybrid", 6)]

plans = {}
overlays = {}
for pol, (p, w) in POLICIES.items():
    plans[pol] = v9lib.build_plan(p, W=w, corpus=corpus)
    overlays[pol] = OverlaySearch(plans[pol], frame=frame)
log(f"plans + overlays built for {list(POLICIES)} [{time.time()-t0:.0f}s]")

# ── fixtures (preregister section 6; predicted PASS) ────────────────────────
fx = []


def check(name, ok, detail=""):
    fx.append({"name": name, "pass": bool(ok), "detail": str(detail)})
    log(f"fixture {'PASS' if ok else 'FAIL'}: {name} {detail}")
    assert ok, f"FIXTURE FAILED: {name} {detail}"


ovs = overlays["p1w5c5"]
fam = v9lib.Overlay(plans["p1w5c5"], frame=frame)
for var in ("delta1", "delta2"):
    a_fam = fam.run(base, var, 0.3, 5.0)
    a_gen = ovs.run_general(base, var, 0.3, 5.0, gamma=1.0, m=0)
    check(f"S1 nesting {var}: run_general(gamma=1,m=0) p_all bitwise == family Overlay.run",
          np.array_equal(a_fam["p_all"], a_gen["p_all"], equal_nan=True)
          and np.array_equal(a_fam["adj"], a_gen["adj"]))

state_cache = {}
for pol in POLICIES:
    for tau in TAU_GRID:
        state_cache[(pol, tau)] = overlays[pol].state_table(base, tau)
log(f"state tables built (24, via real active_state/evidence path — leak "
    f"assert executed on every evidence row) [{time.time()-t0:.0f}s]")

nll_v6 = base["nll_rows"]
FIT1v = W["FIT1"] & valid
m23 = FIT1v & (years == 2023)
m24 = FIT1v & (years == 2024)
n_f, n_23, n_24 = int(FIT1v.sum()), int(m23.sum()), int(m24.sum())
ev_f = event_ids[FIT1v]
fit1_events = list(dict.fromkeys(ev_f))
log(f"FIT1 valid rows {n_f} (2023: {n_23}, 2024: {n_24}), "
    f"{len(fit1_events)} events; MDE(FIT1)={sl.mde_milli(n_f)}m")

for var, m in (("delta2", 0), ("delta1", 0), ("hybrid", 3)):
    full = ovs.run_general(base, var, 0.3, 5.0, gamma=1.0, m=m)
    adj_c = ovs.adj_from_state(state_cache[("p1w5c5", 5.0)], 0.3, 5.0,
                               1.0, m, var, len(frame))
    p_c = sl.p_from_adj(base, adj_c, base["beta"], fmts)
    same_adj = np.array_equal(full["adj"], adj_c)
    same_p = np.array_equal(full["p_all"], p_c, equal_nan=True)
    check(f"S2 cache equivalence {var}(m={m}): adj + p_all bitwise",
          same_adj and same_p)

b_fit1 = sl.fit_beta(base["rdiff"], W["FIT1"], fmts, valid)
check("S3 beta(FIT1, v6) protocol fixture = 0.1152 +/- 1e-3",
      abs(b_fit1 - 0.1152) <= 1e-3, f"realized {b_fit1:.6f}")

p3_state = state_cache[("p3w5c5", 5.0)]
log(f"S4 p3w5c5 active pricings at tau=5: {len(p3_state)} "
    f"({'v6-identity rows will be reported plainly' if not p3_state else 'nonzero — p3 grid is live'})")
fx.append({"name": "S4 p3w5c5 sanity (report-only)", "pass": True,
           "detail": f"{len(p3_state)} active pricings at tau=5"})

# ── the grid (FIT1 aggregation only) ────────────────────────────────────────
rows = []
n_cfg = 0
for pol in POLICIES:
    for tau in TAU_GRID:
        state = state_cache[(pol, tau)]
        for var, m in VARIANT_GRID:
            for gamma in GAMMA_GRID:
                for b in B_GRID:
                    adj = overlays[pol].adj_from_state(
                        state, b, tau, gamma, m, var, len(frame))
                    aff = np.flatnonzero(adj)
                    p_c = base["p_all"].copy()
                    if aff.size:
                        lg = base["beta"] * base["rdiff"][aff] + adj[aff]
                        pm = 1.0 / (1.0 + np.exp(-lg))
                        p_c[aff] = v9lib.series_from_pm(pm, fmts[aff])
                    nll_c = nll_v6.copy()
                    nll_c[aff] = -np.log(np.clip(p_c[aff], 1e-9, 1.0))
                    d = nll_v6 - nll_c            # >0 = candidate better
                    dF = float(np.nanmean(d[FIT1v])) * 1000
                    d23 = float(np.nanmean(d[m23])) * 1000
                    d24 = float(np.nanmean(d[m24])) * 1000
                    dv = d[FIT1v]
                    S, N = dv.sum(), len(dv)
                    loeo = min(
                        float((S - dv[ev_f == e].sum())
                              / (N - (ev_f == e).sum())) * 1000
                        for e in fit1_events)
                    era_min = min(d23, d24)
                    eligible = (dF >= 1.0) and (era_min > 0) and (loeo > 0)
                    rows.append({
                        "policy": pol, "variant": var, "m": m, "b": b,
                        "tau": tau, "gamma": gamma,
                        "dF_milli": round(dF, 3), "d23_milli": round(d23, 3),
                        "d24_milli": round(d24, 3),
                        "era_min_milli": round(era_min, 3),
                        "loeo_min_milli": round(loeo, 3),
                        "n_affected_fit1": int((adj[FIT1v] != 0).sum()),
                        "eligible": bool(eligible)})
                    n_cfg += 1
        log(f"  {pol} tau={tau}: {n_cfg} configs done [{time.time()-t0:.0f}s]")

elig = [r for r in rows if r["eligible"]]
log(f"grid complete: {n_cfg} configs, {len(elig)} eligible "
    f"[{time.time()-t0:.0f}s]")

out = {
    "written_by": "agent:v9-search",
    "written": time.strftime("%Y-%m-%d %H:%M:%S"),
    "preregistered": "v9/preregister.search.md sections 1a + 2 (locked before run)",
    "epistemic_status": ("TRAIN-SIDE ONLY: every aggregate on FIT1 valid rows "
                         "(<= 2024-12-31); no VAL row aggregated anywhere in "
                         "this artifact; selection-grade input to nomination"),
    "base": {"beta_ref": base["beta"], "n_valid": int(valid.sum())},
    "windows_fit1": {"n_valid": n_f, "n_2023": n_23, "n_2024": n_24,
                     "n_events": len(fit1_events),
                     "mde_milli_fit1": sl.mde_milli(n_f),
                     "mde_milli_2023": sl.mde_milli(n_23),
                     "mde_milli_2024": sl.mde_milli(n_24)},
    "grid_def": {"policies": list(POLICIES), "b": B_GRID, "tau": TAU_GRID,
                 "gamma": GAMMA_GRID,
                 "variants": [f"{v}(m={m})" for v, m in VARIANT_GRID],
                 "magnitude": "b*(1-k/5)^gamma*exp(-n/tau), horizon ceil(5*tau)",
                 "eligibility": "dF>=1.0m AND era_min>0 AND loeo_min>0"},
    "fixtures": fx,
    "n_configs": n_cfg,
    "n_eligible": len(elig),
    "configs": rows,
}
path = os.path.join(V9, "stats", "v9_search_grid.json")
with open(path, "w") as f:
    json.dump(out, f, indent=1)
log(f"wrote {path} [{time.time()-t0:.0f}s]")

top = sorted(elig, key=lambda r: -r["era_min_milli"])[:12]
for r in top:
    log(f"  top: {r['policy']} {r['variant']}(m={r['m']}) b={r['b']} "
        f"tau={r['tau']} g={r['gamma']} dF={r['dF_milli']:+.2f} "
        f"d23={r['d23_milli']:+.2f} d24={r['d24_milli']:+.2f} "
        f"loeo={r['loeo_min_milli']:+.2f}")
