"""SPEC RUN — floor-cap fit (train) + THE corpus-wide holdout read + slices
+ coupling. This is the ONLY script that computes holdout aggregates.

Read budget (ADDENDUM 4): ONE headline read at the gate-selected config (or,
if no gate fired, at the unshrunk train-argmin as DOCUMENTATION-ONLY), plus
conditional policy contrasts ONLY if the top-2 CV improvements are within
1e-4 nats and both gates fired. Every read appended to roster_looks.json.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V8 = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
TL = os.path.dirname(V8)
STATS = os.path.join(V8, "stats")
sys.path.insert(0, HERE)
sys.path.insert(0, V8)

import referee  # noqa: E402
import runner  # noqa: E402
from speclib import SpecPlan, load_corpus  # noqa: E402

CAPS = [1.5, 2.0, 3.0]
MDE = {"within_milli": 1.773, "cross_milli": 5.889,
       "source": "stats/power_mde_expanded.json checkpoint_quote (n=1217)"}
EXPL = ("EXPLORATORY — spent holdout (402 prior recorded looks); "
        "preregistered ADDENDUM 4; not confirmatory, not promotable. "
        "Adjudication belongs to the frozen prospective arm H_specrun.")

gate = json.load(open(os.path.join(HERE, "gate_decision.json")))
sel = gate["selection"]
corpus = load_corpus()

if sel["deployed_is_v6"]:
    pol = sel["documentation_policy"]
    a_c, tau_c, s_c = (sel["documentation_a"], sel["documentation_tau"],
                       sel["documentation_s"])
    read_label = ("DOCUMENTATION-ONLY read: gate did NOT clear -> the "
                  "deployed model IS v6 (delta == 0 by construction). This "
                  "read documents what the gate declined: the unshrunk "
                  f"train-argmin {pol} a={a_c} tau={tau_c} s={s_c}.")
else:
    pol = sel["policy"]
    a_c, tau_c, s_c = sel["a"], sel["tau"], sel["s"]
    read_label = f"headline read: shipped config {pol} a={a_c} tau={tau_c} s={s_c}"

W = int(pol[3]) if pol.startswith("p1") else 5
Cc = int(pol.split("c")[1]) if pol.startswith("p1") else 5
plan = SpecPlan("p1", W=W, c=Cc, corpus=corpus) if pol.startswith("p1") \
    else SpecPlan("p3", W=5, corpus=corpus)
N_MIN = sel.get("n_min", 3)

# ── floor cap: train-only fit at the final config ───────────────────────────
print(f"=== floor cap fit (train-only) at {pol} a={a_c} tau={tau_c} "
      f"s={s_c} ===", flush=True)
cap_rows = []
boost_max = 1.0 + a_c * 1.0
if a_c == 0:
    cap_hat, cap_note = None, "no boost (a=0) -> floor moot"
else:
    for C in CAPS:
        out = runner.run_config(plan, a_c, tau_c, s_c, n_min=N_MIN, cap=C)
        cap_rows.append({"cap": C, "ll_train": out["ll_train"],
                         "beta": out["beta"]})
        print(f"  cap={C}: ll_train={out['ll_train']}", flush=True)
    best_ll = min(r["ll_train"] for r in cap_rows)
    cap_hat = min(r["cap"] for r in cap_rows if r["ll_train"] == best_ll)
    cap_note = ("cap grid tie -> smallest (most conservative)"
                if sum(1 for r in cap_rows if r["ll_train"] == best_ll) > 1
                else "train argmin")
    if boost_max <= min(CAPS):
        cap_note += f"; NON-BINDING (max boost {boost_max:.2f} <= {min(CAPS)})"
print(f"  -> cap_hat={cap_hat} ({cap_note})", flush=True)
sel["cap"] = cap_hat
gate["selection"] = sel
with open(os.path.join(HERE, "gate_decision.json"), "w") as f:
    json.dump(gate, f, indent=1)

# ── the final corpus-wide run + v6 reference ────────────────────────────────
print("=== corpus-wide runs (final config + v6 reference, daily) ===",
      flush=True)
fin_run = runner.run_config(plan, a_c, tau_c, s_c, n_min=N_MIN, cap=cap_hat,
                            daily=True)
ref = runner.run_config(None, 0.0, 5.0, 0.0, daily=True)
frame = runner.load_frame()
fmts = frame.fmt.values
event_ids = frame.event_id.values
hold = (frame.date > "2024-12-31").values
v6npz = np.load(os.path.join(V8, "scratch", "bias_h3", "v6_baseline.npz"))
p_v6 = v6npz["p_all"]
assert float(np.nanmax(np.abs(ref["p_all"] - p_v6))) <= 1e-12
p_cfg = fin_run["p_all"]
valid = fin_run["valid"] & ref["valid"]

# ── slice masks (final-classification labels; retrospective, disclosed) ─────
N = len(frame)
side_state = []          # per row: [(org, n, k, o, is_boundary_start)] x2
for i, r in enumerate(frame.itertuples(index=False)):
    row = []
    for org in (r.winner, r.loser):
        fv = plan.final(org)
        seq = corpus["team_match_seq"].get(org, [])
        pos = next((j for j, (d, m) in enumerate(seq)
                    if m == r.match_id), None)
        if fv is None or pos is None or pos >= fv["nvis"]:
            row.append((org, -1, 5, 1.0, False))
        else:
            bstart = any(b["j"] == pos and not b["provisional"]
                         for b in fv["boundaries"])
            row.append((org, int(fv["n"][pos]), int(fv["k"][pos]),
                        float(fv["o"][pos]), bstart))
    side_state.append(row)

gated = np.zeros(N, dtype=bool)
band = {"k4": np.zeros(N, dtype=bool), "k3": np.zeros(N, dtype=bool),
        "k_le2": np.zeros(N, dtype=bool)}
subheavy = np.zeros(N, dtype=bool)
one_side_first3 = {}
for i in range(N):
    firsts = [(org, n, k) for org, n, k, o, bs in side_state[i]
              if 0 <= n <= 2]
    if firsts:
        gated[i] = True
        org, n, k = min(firsts, key=lambda x: x[1])
        band["k4" if k == 4 else ("k3" if k == 3 else "k_le2")][i] = True
        if len(firsts) == 1:
            one_side_first3[i] = firsts[0][0]
    if any(o < 1.0 and not bs for org, n, k, o, bs in side_state[i]):
        subheavy[i] = True

# improvement/degradation per boundary episode (stored v6 baseline p, prior
# report's definition: mean(won - p_team) over first 3 post-boundary rows)
p_team = {}
for i, r in enumerate(frame.itertuples(index=False)):
    if not np.isnan(p_v6[i]):
        p_team[(r.winner, r.match_id)] = (1.0, float(p_v6[i]))
        p_team[(r.loser, r.match_id)] = (0.0, 1.0 - float(p_v6[i]))
imp_of = {}
for org in plan.orgs:
    fv = plan.final(org)
    if fv is None:
        continue
    seq = corpus["team_match_seq"][org]
    for b in fv["boundaries"]:
        if b["provisional"]:
            continue
        vals = [p_team[(org, m)][0] - p_team[(org, m)][1]
                for d, m in seq[b["j"]:b["j"] + 3] if (org, m) in p_team]
        if vals:
            imp_of[(org, b["j"])] = bool(np.mean(vals) > 0)
improve = np.zeros(N, dtype=bool)
degrade = np.zeros(N, dtype=bool)
for i, org in one_side_first3.items():
    fv = plan.final(org)
    seq = corpus["team_match_seq"][org]
    pos = next(j for j, (d, m) in enumerate(seq)
               if m == frame.match_id.iloc[i])
    bj = max((b["j"] for b in fv["boundaries"]
              if not b["provisional"] and b["j"] <= pos), default=None)
    if bj is not None and (org, bj) in imp_of:
        (improve if imp_of[(org, bj)] else degrade)[i] = True

SLICES = [("change-gated (either side n<=2 post-boundary)", gated),
          ("retention k=4", band["k4"]), ("retention k=3", band["k3"]),
          ("retention k<=2", band["k_le2"]),
          ("improvement cases (retrospective)", improve),
          ("degradation cases (retrospective)", degrade),
          ("sub-heavy rows (deviation without boundary)", subheavy)]


def judged(p_new, mask):
    d = referee.delta_vector(p_new[mask], p_v6[mask])
    iid = referee.paired_bootstrap_crn(d, mode="iid")
    blk = referee.paired_bootstrap_crn(d, mode="block_event",
                                       event_ids=event_ids[mask])
    return d, iid, blk


# ═══ THE READ ═══
m = hold & valid
d, iid, blk = judged(p_cfg, m)
dll = float(d.mean())
roi = referee.expected_roi_of_dll(dll, p_v6[m])
sym = referee.expected_roi_of_dll(abs(dll), p_v6[m])
slice_rows = []
for name, mask in SLICES:
    mm = m & mask
    nn = int(mm.sum())
    if nn < 10:
        slice_rows.append({"name": name, "n": nn, "note": "n<10, suppressed"})
        continue
    db = referee.delta_vector(p_cfg[mm], p_v6[mm])
    bb = referee.paired_bootstrap_crn(db, mode="iid")
    slice_rows.append({"name": name, "n": nn,
                       "delta_milli": round(float(db.mean()) * 1000, 3),
                       "ci_lo_milli": round(bb["ci_lo"] * 1000, 2),
                       "ci_hi_milli": round(bb["ci_hi"] * 1000, 2),
                       "p_better": bb["p_better"]})
print(f"READ: {read_label}", flush=True)
print(f"  overall {dll*1000:+.3f}m iid [{iid['ci_lo']*1000:.2f},"
      f"{iid['ci_hi']*1000:.2f}] blk [{blk['ci_lo']*1000:.2f},"
      f"{blk['ci_hi']*1000:.2f}] p {iid['p_better']:.3f}/{blk['p_better']:.3f}",
      flush=True)

# ── coupling (spec §5, its own subsection) ──────────────────────────────────
days = sorted(ref["daily_r"].keys())
tidx = ref["tidx"]
nob = [o for o in plan.orgs
       if plan.final(o) is not None
       and not any(not b["provisional"]
                   for b in plan.final(o)["boundaries"])]
first_b = {o: min(b["date"] for b in plan.final(o)["boundaries"]
                  if not b["provisional"])
           for o in plan.orgs if plan.final(o) is not None
           and any(not b["provisional"] for b in plan.final(o)["boundaries"])}
first_play = {o: corpus["team_match_seq"][o][0][0] for o in plan.orgs}
cd_nochange, cd_prechange = [], []
for o, ti in tidx.items():
    fp = first_play.get(o)
    if fp is None:
        continue
    if o in (nob or []):
        for D in days:
            if D > fp:
                cd_nochange.append(abs(float(fin_run["daily_r"][D][ti]
                                             - ref["daily_r"][D][ti])))
    elif o in first_b:
        for D in days:
            if fp < D < first_b[o]:
                cd_prechange.append(abs(float(fin_run["daily_r"][D][ti]
                                              - ref["daily_r"][D][ti])))


def dist(v):
    if not v:
        return None
    v = np.array(v)
    return {"n_team_days": int(len(v)),
            "p50": round(float(np.percentile(v, 50)), 4),
            "p90": round(float(np.percentile(v, 90)), 4),
            "p99": round(float(np.percentile(v, 99)), 4),
            "max": round(float(v.max()), 4), "mean": round(float(v.mean()), 4)}


coupling = {
    "definition": "corpus-wide run vs v6, |delta rating| by (team, day)",
    "never_changing_teams": {"n_teams": len(nob), "dist": dist(cd_nochange)},
    "changing_teams_strictly_pre_first_boundary": {"dist": dist(cd_prechange)},
    "note": "a real property of the joint Massey solve under the corpus-wide "
            "rule; the case charts use per-team ablations and are NOT "
            "allowed to show this (their pre-boundary gate is exactly 0.0)"}

# ── conditional contrast reads (budget <= 2 more) ───────────────────────────
contrasts = []
fired = {p: r for p, r in gate["per_policy"].items() if r["gate_fired"]}
if len(fired) >= 2:
    top2 = sorted(fired, key=lambda p: -fired[p]["gate_mean_milli"])[:2]
    gap_nats = abs(fired[top2[0]]["gate_mean_milli"]
                   - fired[top2[1]]["gate_mean_milli"]) / 1000.0
    if gap_nats < 1e-4 and top2[1] != pol:
        p2pol = top2[1]
        W2 = int(p2pol[3]) if p2pol.startswith("p1") else 5
        C2 = int(p2pol.split("c")[1]) if p2pol.startswith("p1") else 5
        plan2 = SpecPlan("p1", W=W2, c=C2, corpus=corpus) \
            if p2pol.startswith("p1") else SpecPlan("p3", W=5, corpus=corpus)
        r2 = gate["per_policy"][p2pol]
        a2 = r2["a_shrunk"] if p2pol != "p3w5" else 0.0
        s2 = r2["a_shrunk"] if p2pol == "p3w5" else r2["s_hat_trainstage"]
        out2 = runner.run_config(plan2, a2, gate["per_policy"][p2pol]["tau_hat"],
                                 s2, n_min=N_MIN, cap=cap_hat)
        d2, iid2, blk2 = judged(out2["p_all"], m & out2["valid"])
        contrasts.append({"policy": p2pol, "a": a2, "s": s2,
                          "delta_milli": round(float(d2.mean()) * 1000, 3),
                          "iid": iid2, "block_event": blk2,
                          "reason": "train evidence within 1e-4 nats of the "
                                    "winner (preregistered condition)"})
        print(f"  contrast read {p2pol}: {float(d2.mean())*1000:+.3f}m",
              flush=True)

res = {
    "written_by": "agent:roster-g SPEC RUN",
    "epistemic_status": EXPL,
    "read_label": read_label,
    "gate_outcome": {
        "fired": not sel["deployed_is_v6"],
        "deployed_model": "v6 (subsystem OFF; a=s=0)" if sel["deployed_is_v6"]
        else f"v6 + subsystem {pol}",
        "selected_policy_or_doc": pol,
        "a_shrunk": a_c if not sel["deployed_is_v6"] else 0.0,
        "a_documentation": a_c if sel["deployed_is_v6"] else None,
        "tau": tau_c, "s": s_c, "n_min": N_MIN, "cap": cap_hat,
        "cap_fit": {"grid": cap_rows, "note": cap_note},
        "per_policy_gate": gate["per_policy"],
        "p2": gate["p2"],
    },
    "headline": {
        "n_scored": int(m.sum()),
        "ll_holdout_config": round(float(referee.per_series_ll(p_cfg[m]).mean()), 6),
        "ll_holdout_v6": round(float(referee.per_series_ll(p_v6[m]).mean()), 6),
        "delta_milli": round(dll * 1000, 3),
        "iid": iid, "block_event": blk,
        "mde_context": MDE,
        "inside_noise_floor": bool(abs(dll * 1000) < MDE["within_milli"]),
        "expected_roi_both_units": {
            "referee_ladder_at_signed_dll": roi["expected_roi_delta"],
            "symmetric_reading_roi_delta": (-sym["expected_roi_delta"]
                                            if dll < 0
                                            else sym["expected_roi_delta"]),
            "dll_milli": round(dll * 1000, 3),
            "ladder_source": sym["ladder_source"]},
    },
    "slices": slice_rows,
    "coupling": coupling,
    "contrast_reads": contrasts,
    "note_on_deployment": ("The deployed model is v6 exactly; this read is "
                           "documentation of the declined subsystem."
                           if sel["deployed_is_v6"] else
                           "The read scores the model that would ship "
                           "(corpus-wide, all boundaries active)."),
}
with open(os.path.join(STATS, "roster_spec_read.json"), "w") as f:
    json.dump(res, f, indent=1)

looks = json.load(open(os.path.join(STATS, "roster_looks.json")))
looks["new_reads"].append({
    "read": "spec_run_headline",
    "config": f"{pol}_a{a_c}_t{tau_c}_s{s_c}_cap{cap_hat}",
    "ll_holdout": res["headline"]["ll_holdout_config"],
    "delta_milli_vs_v6": res["headline"]["delta_milli"],
    "status": "EXPLORATORY", "operator_specified": True,
    "note": read_label})
for c in contrasts:
    looks["new_reads"].append({"read": "spec_run_contrast",
                               "config": c["policy"],
                               "delta_milli_vs_v6": c["delta_milli"],
                               "status": "EXPLORATORY"})
looks["new_primary_looks"] = len(looks["new_reads"])
looks["grand_total_after"] = (looks["prior_grand_total_recorded_holdout_numbers"]
                              + len(looks["new_reads"]))
with open(os.path.join(STATS, "roster_looks.json"), "w") as f:
    json.dump(looks, f, indent=1)
print(f"looks grand total now {looks['grand_total_after']}", flush=True)
print("stats/roster_spec_read.json written", flush=True)
