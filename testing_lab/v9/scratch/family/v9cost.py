"""v9 cost model — wall-clock for one unit of every family operation, so
the search agent can budget its grid honestly. Preregistered section 5.
NO metric is recorded anywhere here (run outputs are used for timing and
discarded; ll_train etc. are never written)."""
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V9 = os.path.dirname(os.path.dirname(HERE))
STATS = os.path.join(V9, "stats")
sys.path.insert(0, HERE)

import v9lib  # noqa: E402
from v9lib import Overlay, build_plan, load_corpus, solve_side_run, v6_run  # noqa: E402
import runner  # noqa: E402

T = {}

t0 = time.time()
runner.get_engine()
T["engine_load_s"] = round(time.time() - t0, 2)

t0 = time.time()
base = v6_run()
T["v6_run_s"] = round(time.time() - t0, 2)

corpus = load_corpus()
T["specplan_build_s"] = {}
plans = {}
for pol, W in (("p1", 3), ("p1", 5), ("p1", 8), ("p3", 5)):
    t0 = time.time()
    plans[(pol, W)] = build_plan(pol, W=W, c=5, corpus=corpus)
    T["specplan_build_s"][f"{pol}w{W}c5"] = round(time.time() - t0, 2)

plan5 = plans[("p1", 5)]
t0 = time.time()
solve_side_run(plan5, 2.0, 5.0, 0.3, n_min=3, cap=1.5)
T["solve_side_run_s"] = round(time.time() - t0, 2)

t0 = time.time()
ov = Overlay(plan5)
T["overlay_index_build_s"] = round(time.time() - t0, 2)
t0 = time.time()
ov.run(base, "delta2", 1.0, 5.0)
T["overlay_delta2_full_s"] = round(time.time() - t0, 2)
t0 = time.time()
ov.run(base, "delta1", 1.0, 5.0)
T["overlay_delta1_full_s"] = round(time.time() - t0, 2)


def _brand():
    try:
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        return platform.processor()


out = {
    "written_by": "agent:v9-family",
    "written": time.strftime("%Y-%m-%d %H:%M:%S"),
    "preregistered": "v9/preregister.family.md section 5",
    "machine": {"platform": platform.platform(), "cpu": _brand(),
                "cores": os.cpu_count(), "python": platform.python_version(),
                "numpy": np.__version__},
    "timings_s": T,
    "budget_guidance": {
        "solve_side_per_config": "one full engine walk-forward run "
            f"(~{T['solve_side_run_s']}s) per (a,tau,s,n_min,cap) config; "
            "beta refits inside the run (train-only) at no extra step",
        "prediction_layer_per_config": "ONE shared pure-v6 base run "
            f"(~{T['v6_run_s']}s, amortized), then ~"
            f"{max(T['overlay_delta2_full_s'], T['overlay_delta1_full_s'])}s "
            "per (b,tau,variant) overlay config",
        "per_policy_overhead": "one SpecPlan build per (policy,W) "
            f"(~{max(T['specplan_build_s'].values())}s worst observed), "
            "reusable across every config of that policy",
        "solve_side_configs_per_hour": int(3600 / max(
            T["solve_side_run_s"], 0.01)),
        "prediction_layer_configs_per_hour": int(3600 / max(
            T["overlay_delta1_full_s"], T["overlay_delta2_full_s"], 0.01)),
        "note": "walk-forward era-transfer protocol (laws #2) multiplies "
                "solve-side cost by the number of fit/validate splits; "
                "prediction-layer overlays reuse one base run per split."},
}
with open(os.path.join(STATS, "v9_cost.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(T, indent=1))
print("stats/v9_cost.json written")
