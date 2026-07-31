"""agent:compose — multiple-looks accounting (compose brief rule 4).

Counts every holdout scoring the v8 program has made, per agent, with the
counted names listed so the tally is auditable. Primary unit = a distinct
candidate configuration whose holdout LL (expanded or frozen frame) was
computed and recorded. Baseline replays of v6 itself and reproductions of
already-published numbers are tallied separately. Slice/bucket/subpop tables
of an already-counted vector are NOT extra primary looks (secondary note).
Emits stats/compose_looks.json.
"""
import json
import os
from datetime import datetime

V8 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ST = os.path.join(V8, "stats")
SC = os.path.join(V8, "scratch")


def j(p):
    with open(p) as f:
        return json.load(f)


agents = {}


def add(agent, source, names, kind="candidate_config"):
    agents.setdefault(agent, []).append(
        {"source": source, "kind": kind, "n": len(names),
         "names": sorted(names)})


# ── decay ───────────────────────────────────────────────────────────────────
d = j(os.path.join(ST, "decay_axes.json"))
names = []
for ax, o in d["axes"].items():
    names += [f"{ax}:{k}" for k in o.get("grid", {})]
    if "on_top_of_v6" in o:
        names.append(f"{ax}:{o['on_top_of_v6']['name']}")
add("decay", "stats/decay_axes.json (every grid point carries ll_test)", names)
add("decay", "stats/decay_form.json", list(j(os.path.join(ST, "decay_form.json"))["results"]))
add("decay", "stats/decay_rerace.json (5a re-race table)",
    list(j(os.path.join(ST, "decay_rerace.json"))["table"]))
add("decay", "stats/decay_subpops.json", ["(subpop slices of already-counted "
    "configs — no new vectors)"], kind="slice_tables_only")

# ── context ─────────────────────────────────────────────────────────────────
c = j(os.path.join(ST, "context_seriousness.json"))
add("context", "stats/context_seriousness.json",
    [f"3a_integrity_w0[{i}]" for i in range(len(c["grid_integrity_w0"]))]
    + [f"3a_blanket_we[{i}]" for i in range(len(c["grid_blanket_we"]))])
c = j(os.path.join(ST, "context_stakes.json"))
add("context", "stats/context_stakes.json",
    [f"3cA_solve_weight[{i}]" for i in range(len(c["testA_solve_weight"]["grid"]))]
    + ["3cB_elim_variance"])
c = j(os.path.join(ST, "context_exposure.json"))
add("context", "stats/context_exposure.json",
    ["3ba_exposure_term", "3ba_exposure_ewcLAN",
     "3bb_form3_alone", "3bb_form3_with", "3bb_form5_alone", "3bb_form5_with"])
c = j(os.path.join(ST, "context_weights.json"))
add("context", "stats/context_weights.json (per-class weight profiles + "
    "selected + sensitivity)",
    [f"2b_profile_{cls}[{i}]" for cls, arr in c["profiles"].items()
     for i in range(len(arr))]
    + ["2b_fitted_classweights_holdout"]
    + [f"2b_ewc_sens[{i}]" for i in
       range(len(c["ewc_weight_sensitivity_v6_conditioned"]["grid"]))])
add("context", "stats/context_shrink.json",
    ["3e_X1_standin", "3e_X2_prepasym", "3e_joint", "3e_class_dummy_falsifier"])
add("context", "stats/context_adjacency.json",
    ["(coefficient inference only — no holdout scoring)"], kind="none")

# ── bias_h1 ─────────────────────────────────────────────────────────────────
add("bias_h1", "stats/h1_roundbt.json", ["roundbt_primary"])
t = j(os.path.join(ST, "h1_tobit.json"))
add("bias_h1", "stats/h1_tobit.json", ["tobit_s1.0_primary", "tobit_s0.8_sens",
                                       "tobit_s1.25_sens"])
add("bias_h1", "stats/h1_tobit.json em_history",
    [f"em_iter:{k}[{i}]" for k, v in t["em_history"].items()
     for i in range(len(v))], kind="per_iteration_diagnostic")

# ── bias_h2 ─────────────────────────────────────────────────────────────────
for fn in ("h2_hierarchical.json", "h2_ridge_ablation.json", "h2_centrality.json"):
    o = j(os.path.join(ST, fn))
    txt = json.dumps(o)
    n = txt.count("ll_holdout") + txt.count("ll_test")
    # gated/stopped experiments carry no candidate holdout scores
    cand = []
    if fn == "h2_centrality.json" and "ll_holdout" in txt:
        cand = ["(v6 baseline replay only)"]
        add("bias_h2", f"stats/{fn}", cand, kind="baseline_replay")
    else:
        add("bias_h2", f"stats/{fn}",
            [f"(status={o.get('status', '?')}; no candidate holdout scoring)"],
            kind="none" if n == 0 else "check")

# ── bias_h3 ─────────────────────────────────────────────────────────────────
add("bias_h3", "stats/h3_statespace.json",
    ["ss_1a", "ss_1b_qcal", "ss_1c_debutprior", "ss_5d_roster"])
add("bias_h3", "stats/h3_process_noise.json (per-axis ll_holdout_record)",
    ["axis_A_roster", "axis_B_orgage", "axis_C_volatility"])
add("bias_h3", "stats/h3_ensemble.json (one scored composite)",
    ["ss_1b|soft_blend_neff composite"])
swc = j(os.path.join(SC, "bias_h3", "sweep_core.json"))
add("bias_h3", "scratch/bias_h3/sweep_core.json (checkpointed core sweep — "
    "ll_holdout recorded per point; selection was train-only)",
    [f"core:{k}" for k in swc.get("points", {})], kind="sweep_checkpoint")

# ── bias_h4 ─────────────────────────────────────────────────────────────────
h4 = j(os.path.join(ST, "h4_series_link.json"))
add("bias_h4", "stats/h4_series_link.json", [f"link:{k}" for k in h4["links"]])

# ── power / referee / corpus / autopsy ──────────────────────────────────────
add("power", "stats/power_mde*.json", ["(24 published v7 ll_test values "
    "reproduced for npz alignment — reproductions, not new candidates)"],
    kind="reproduction")
add("referee", "stats/referee_selftest.json", ["(canonical baseline "
    "reproductions: 4 npz configs + native v6 build)"], kind="reproduction")
add("corpus", "stats/corpus_*.json", ["(data provenance only — no holdout "
    "scoring)"], kind="none")
add("autopsy", "stats/autopsy_*.json", ["(market-side diagnostics on n=38 "
    "settled fills — separate family, never a fitting target; no expanded-"
    "holdout LL scorings)"], kind="market_diagnostics")

# ── compose (this wave) ─────────────────────────────────────────────────────
add("compose", "stats/compose_stacks.json (this wave, preregistered, one "
    "scoring each)", ["S1_gate5d", "S2_fade_shrink", "S3_full"])

# ── totals ──────────────────────────────────────────────────────────────────
PRIMARY = ("candidate_config",)
tot_primary = sum(e["n"] for a in agents.values() for e in a
                  if e["kind"] in PRIMARY)
tot_sweep = sum(e["n"] for a in agents.values() for e in a
                if e["kind"] == "sweep_checkpoint")
tot_diag = sum(e["n"] for a in agents.values() for e in a
               if e["kind"] == "per_iteration_diagnostic")
per_agent = {a: sum(e["n"] for e in lst if e["kind"] in PRIMARY)
             for a, lst in agents.items()}

out = {
    "written_by": "agent:compose", "written": datetime.now().strftime("%F %T"),
    "rule": "compose brief rule 4 — the garden of forking paths, measured "
            "not hidden",
    "unit_definition": {
        "primary": "distinct candidate config whose holdout LL was computed "
                   "and recorded (selection may still have been train-only — "
                   "a recorded holdout number is a look regardless)",
        "sweep_checkpoint": "holdout LL recorded per sweep point in scratch "
                            "checkpoints (h3 core sweep; train-only selection "
                            "but numbers exist on disk)",
        "per_iteration_diagnostic": "same config re-scored across EM "
                                    "iterations (h1)",
        "excluded": "slice/bucket/subpop tables of an already-counted "
                    "vector; v6 baseline replays; reproductions of published "
                    "v7 numbers; market-side autopsy (separate family)"},
    "per_agent_primary": per_agent,
    "totals": {
        "primary_candidate_looks": tot_primary,
        "sweep_checkpoint_looks": tot_sweep,
        "per_iteration_diagnostic_looks": tot_diag,
        "grand_total_recorded_holdout_numbers":
            tot_primary + tot_sweep + tot_diag},
    "family_wise_context": {
        "note": "With K primary looks at alpha=0.05 one expects ~K/20 "
                "spurious 'wins' by chance; the program's guardrails are "
                "(a) train-only selection, (b) the MDE noise floor, "
                "(c) the promotion gate (mean dLL >= family MDE AND "
                "p_better >= 0.95 in BOTH CRN modes AND bias AND buckets). "
                "No look in this program cleared that bar. Bonferroni at "
                "the primary-look count is quoted with compose's p-values "
                "in phase_compose.md.",
        "K_primary": tot_primary,
        "bonferroni_alpha_at_K": round(0.05 / max(tot_primary, 1), 6)},
    "detail": agents}
with open(os.path.join(ST, "compose_looks.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps({"per_agent_primary": per_agent, "totals": out["totals"]},
                 indent=1))
