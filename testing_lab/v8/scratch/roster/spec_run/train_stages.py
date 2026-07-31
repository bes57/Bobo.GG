"""SPEC RUN train stages — ALL selection on train only (date <= 2024-12-31).

Step 0: nesting assertion — EngineSpec at a=s=0 is bit-identical to the plain
        v6 Engine run (sha256 on rdiff bytes + daily ratings) and reproduces
        the stored baseline to <= 1e-12. Published into the fixtures ledger.
Steps 1-4: per shippable policy (census rule: c=5 for P1; P3-W5; P2 NOT RUN):
        (a x tau) grid at s=0 (P1-W5 full; W3/W8 reuse tau-hat), s-profile at
        (a_raw, tau-hat), a-profile at s-hat if s-hat > 0.
Every run's holdout metrics are popped unseen by runner.run_config; only
beta + ll_train + rdiff are recorded. rdiff arrays go to stage npz for the
single later read (read_corpus.py) — no holdout aggregate exists until then.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V8 = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)

import runner  # noqa: E402
from speclib import SpecPlan, load_corpus  # noqa: E402
from engine import Engine  # noqa: E402

A_GRID = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.5, 6.0]
TAU_GRID = [2.0, 3.0, 5.0, 8.0, 13.0]
S_GRID = [0.0, 0.2, 0.4, 0.7, 1.0]
N_MIN, CAPS = 3, [1.5, 2.0, 3.0]

corpus = load_corpus()
t0 = time.time()

# ── step 0: nesting checksum ────────────────────────────────────────────────
print("=== step 0: nesting (a=s=0 bit-identical to v6) ===", flush=True)
eng, frame = runner.get_engine()
base = runner.run_config(None, 0.0, 5.0, 0.0, daily=True)
ck_spec_rd = runner.rd_checksum(base["rdiff"])
ck_spec_daily = runner.daily_checksum(base["daily_r"])

plain = Engine()
plain.series = frame.copy().reset_index(drop=True)
plain.pred_days = sorted(frame.date.unique())
cfgp = runner.v6_cfg(eng, daily=True)
outp = plain.run(dict(cfgp))
for kk in ("ll_test", "brier_test", "p_test"):
    outp.pop(kk, None)
ck_plain_rd = runner.rd_checksum(outp["rdiff"])
ck_plain_daily = runner.daily_checksum(outp["daily_r"])

v6npz = np.load(os.path.join(V8, "scratch", "bias_h3", "v6_baseline.npz"))
mx_stored = float(np.nanmax(np.abs(base["p_all"] - v6npz["p_all"])))
nest_ok = (ck_spec_rd == ck_plain_rd and ck_spec_daily == ck_plain_daily
           and mx_stored <= 1e-12)
print(f"  rdiff sha equal: {ck_spec_rd == ck_plain_rd}; daily sha equal: "
      f"{ck_spec_daily == ck_plain_daily}; max|dp| vs stored: {mx_stored:.1e}",
      flush=True)
fx_path = os.path.join(V8, "stats", "roster_spec_fixtures.json")
fx = json.load(open(fx_path))
fx["assertions"].append({
    "name": "NESTING (spec 3.5): EngineSpec at a=s=0 bit-identical to plain "
            "v6 (sha256 rdiff + sha256 stacked daily ratings) and <=1e-12 "
            "vs stored baseline p_all",
    "expected": "sha equal x2, <=1e-12",
    "got": f"rdiff_sha_eq={ck_spec_rd == ck_plain_rd} "
           f"daily_sha_eq={ck_spec_daily == ck_plain_daily} "
           f"max_dp={mx_stored:.2e}",
    "pass": bool(nest_ok)})
fx["n_pass"] = sum(1 for a in fx["assertions"] if a["pass"])
fx["n_fail"] = sum(1 for a in fx["assertions"] if not a["pass"])
fx["nesting_checksums"] = {"rdiff_sha256": ck_spec_rd,
                           "daily_sha256": ck_spec_daily}
with open(fx_path, "w") as f:
    json.dump(fx, f, indent=1)
assert nest_ok, "NESTING ASSERTION FAILED"

# ── plans (shippable set per census + P2 harness decision) ──────────────────
plans = {
    "p1w3c5": SpecPlan("p1", W=3, c=5, corpus=corpus),
    "p1w5c5": SpecPlan("p1", W=5, c=5, corpus=corpus),
    "p1w8c5": SpecPlan("p1", W=8, c=5, corpus=corpus),
    "p3w5":   SpecPlan("p3", W=5, corpus=corpus),
}

manifest = {}
store = {}


def run_and_log(pol, a, tau, s):
    cid = f"{pol}_a{a}_t{tau}_s{s}"
    if cid in manifest:
        return manifest[cid]
    if a == 0 and s == 0:
        res = {"beta": base["beta"], "ll_train": base["ll_train"]}
        store[cid] = base["rdiff"]
    else:
        out = runner.run_config(plans[pol], a, tau, s, n_min=N_MIN, cap=None)
        res = {"beta": out["beta"], "ll_train": out["ll_train"]}
        store[cid] = out["rdiff"]
    manifest[cid] = dict(res, policy=pol, a=a, tau=tau, s=s)
    print(f"  {cid}: ll_train={res['ll_train']} beta={res['beta']} "
          f"[{time.time()-t0:.0f}s]", flush=True)
    return manifest[cid]


print("=== stage 1: P1-W5 (a x tau) grid at s=0 ===", flush=True)
for a in A_GRID:
    for tau in (TAU_GRID if a > 0 else [TAU_GRID[0]]):
        run_and_log("p1w5c5", a, tau, 0.0)
best_w5 = min((m for m in manifest.values()
               if m["policy"] == "p1w5c5" and m["a"] > 0),
              key=lambda m: m["ll_train"])
tau_hat = best_w5["tau"]
print(f"  -> tau_hat={tau_hat} (train argmin at a={best_w5['a']})", flush=True)

print("=== stage 2: a-profiles W3/W8 at tau_hat, s=0 ===", flush=True)
for pol in ("p1w3c5", "p1w8c5"):
    for a in A_GRID:
        run_and_log(pol, a, tau_hat, 0.0)

print("=== stage 3: s-profiles ===", flush=True)
araw = {}
for pol in ("p1w3c5", "p1w5c5", "p1w8c5"):
    cand = [m for m in manifest.values() if m["policy"] == pol
            and (m["tau"] == tau_hat or m["a"] == 0) and m["s"] == 0]
    araw[pol] = min(cand, key=lambda m: m["ll_train"])["a"]
    print(f"  {pol}: a_raw={araw[pol]}", flush=True)
    for s in S_GRID[1:]:
        run_and_log(pol, araw[pol] if araw[pol] > 0 else 1.0, tau_hat, s)
for s in S_GRID:
    run_and_log("p3w5", 0.0, tau_hat, s)

print("=== stage 4: a-profile at s_hat where s_hat > 0 ===", flush=True)
shat = {}
for pol in ("p1w3c5", "p1w5c5", "p1w8c5"):
    aa = araw[pol] if araw[pol] > 0 else 1.0
    cand = [(m["s"], m["ll_train"]) for m in manifest.values()
            if m["policy"] == pol and m["a"] == aa
            and (m["tau"] == tau_hat or m["a"] == 0)]
    base_ll = min(ll for s_, ll in cand if s_ == 0.0) if any(
        s_ == 0.0 for s_, _ in cand) else None
    s_best, ll_best = min(cand, key=lambda x: x[1])
    shat[pol] = s_best
    print(f"  {pol}: s_hat={s_best} (ll {ll_best} vs s0 {base_ll})", flush=True)
    if s_best > 0:
        for a in A_GRID:
            run_and_log(pol, a, tau_hat, s_best)
p3cand = [(m["s"], m["ll_train"]) for m in manifest.values()
          if m["policy"] == "p3w5"]
shat["p3w5"] = min(p3cand, key=lambda x: x[1])[0]
print(f"  p3w5: s_hat={shat['p3w5']}", flush=True)

# widen-if-edge rule (preregistered): if any policy's raw argmin sits at a=6
for pol in ("p1w3c5", "p1w5c5", "p1w8c5"):
    s_sel = shat[pol] if shat[pol] > 0 else 0.0
    prof = {m["a"]: m["ll_train"] for m in manifest.values()
            if m["policy"] == pol and m["s"] == s_sel
            and (m["tau"] == tau_hat or m["a"] == 0)}
    if prof and min(prof, key=prof.get) >= 6.0:
        print(f"  {pol}: argmin at a=6 EDGE — widening to 8, 10", flush=True)
        for a in (8.0, 10.0):
            run_and_log(pol, a, tau_hat, s_sel)

np.savez_compressed(os.path.join(HERE, "stage_runs.npz"),
                    **{k: v for k, v in store.items()})
with open(os.path.join(HERE, "stage_manifest.json"), "w") as f:
    json.dump({"tau_hat": tau_hat, "a_raw": araw, "s_hat": shat,
               "n_min": N_MIN, "caps_grid": CAPS,
               "configs": manifest}, f, indent=1)
print(f"DONE: {len(manifest)} configs in {time.time()-t0:.0f}s "
      f"(train-only; holdout scrubbed)", flush=True)
