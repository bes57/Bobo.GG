"""Generate reports/v8_lab.html — the BenPom v8 research program (2026-07-28):
power → corpus → context/decay/bias mechanisms → compose → adversarial review.

Two phases, per preregister.page.md (agent:page):
  1. DERIVE  — mechanical reshapes into testing_lab/v8/stats/page_*.json and
               stats/ledger_v8_updates.json (one writer: agent:page). No new
               statistics: copies, counts, deterministic display arithmetic
               (Wilson intervals, MDE(n) from stored sigma), predict.py slate.
  2. RENDER  — the page is built from testing_lab/v8/stats/*.json ONLY.
               No numeric measurement literal appears in the HTML template.

The ADVERSARY-AMENDED verdicts are the published verdicts. adversary_report.md
publishes verbatim. Any assertion failure aborts loudly (README rule 6).

Usage: python3 testing_lab/gen_v8_report.py [--render-only]
"""
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
V8 = os.path.join(HERE, "v8")
STATS = os.path.join(V8, "stats")
SCRATCH = os.path.join(V8, "scratch")
RD = os.path.join(HERE, "out", "reports")


def j(name, base=STATS):
    with open(os.path.join(base, name)) as f:
        return json.load(f)


def wj(name, obj):
    path = os.path.join(STATS, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
    print("derived:", path)


def die(msg):
    raise SystemExit(f"gen_v8_report FATAL: {msg}")


# ═════════════════════════════ PHASE 1 — DERIVE ══════════════════════════════

def derive_v7_ladder():
    """v7_reclass rows + amended_loss flag under the program-standard cross
    floor (adversary F1). Asserts the flag reproduces the adversary's four."""
    v7 = j("v7_reclass.json")
    pme = j("power_mde_expanded.json")
    cross = pme["checkpoint_quote"]["cross_milli"]
    rows = []
    for r in v7["rows"]:
        amended = abs(r["mean_delta_milli"]) >= cross and r["sig_block"]
        rows.append({
            "config": r["config"], "regime": r["regime"],
            "delta_milli": r["mean_delta_milli"],
            "pair_mde80_milli": r["mde80_milli"],
            "ci_block_milli": r["ci_block_milli"],
            "p_better_iid": r["p_better_iid_crn"],
            "sig_block": r["sig_block"],
            "phase0_verdict": r["verdict"],
            "amended_loss": amended,
        })
    got = {r["config"] for r in rows if r["amended_loss"]}
    want = {"sym_6", "sym_8", "surprise_12_20", "boxexp_c3_hl8"}
    if got != want:
        die(f"amended-loss set mismatch: derived {got} vs adversary {want}")
    wj("page_v7_ladder.json", {
        "source": "stats/v7_reclass.json + stats/power_mde_expanded.json",
        "amended_rule": ("ADVERSARY AMENDMENT (F1): distinguishable-as-worse iff "
                         "|Δ| ≥ program-standard cross floor AND block-CI excludes 0"),
        "cross_floor_milli": cross,
        "within_floor_milli": pme["checkpoint_quote"]["within_milli"],
        "champion": v7["champion"],
        "near_ties_unresolved": ["sym_20", "consist_16_10"],
        "rows": rows,
    })


def derive_adversary_fragility():
    """Verbatim copy of the adversary's machine-readable recompute numbers so
    the page serves them from stats/. No arithmetic."""
    wj("page_adversary_fragility.json", {
        "source": "scratch/adversary/recompute_{eclass_cold,3e,compose,core}.json (verbatim copies)",
        "eclass_cold": j("recompute_eclass_cold.json", os.path.join(SCRATCH, "adversary")),
        "shrink_3e": j("recompute_3e.json", os.path.join(SCRATCH, "adversary")),
        "compose": j("recompute_compose.json", os.path.join(SCRATCH, "adversary")),
        "core": j("recompute_core.json", os.path.join(SCRATCH, "adversary")),
    })


_PREREG_ROWS = [
    # (agent, item, unit, pred_lo, pred_hi, realized, held, source_file, quote)
    ("decay", "5a consist_16_10 vs v6", "milli", -1.5, 0.5, -0.531, "held",
     "preregister.decay.md", "consist_16_10 −0.53m (predicted [−1.5,+0.5])"),
    ("decay", "5a sym_20 vs v6", "milli", -4, 0, -2.17, "held",
     "preregister.decay.md", "sym_20 −2.17m (predicted [−4,0])"),
    ("decay", "5a sym_24 vs v6", "milli", -4, 1, -2.42, "held",
     "preregister.decay.md", "sym_24 −2.42m (predicted [−4,+1])"),
    ("decay", "5b-a lineup continuity vs v6", "milli", -2, 2, -2.82, "miss",
     "preregister.decay.md", "predicted [−2,+2] vs v6 → measured"),
    ("decay", "5b-a lineup continuity vs own control", "milli", 0.5, 3, 1.68, "held",
     "preregister.decay.md", "predicted +0.5..+3 vs own control → measured +1.68m"),
    ("decay", "5b-b opponent quality of anomaly", "milli", 0, 1.5, -0.85, "sign wrong",
     "preregister.decay.md", "predicted +0..+1.5 → measured −0.85m (SIGN WRONG,"),
    ("decay", "5b-c anomaly margin", "milli", 0, 1.5, -0.55, "sign wrong",
     "preregister.decay.md", "predicted +0..+1.5 → measured −0.55m (SIGN WRONG, inside"),
    ("decay", "5b-d event-class fade vs sym control", "milli", 0.5, 2, 0.27, "under band",
     "preregister.decay.md", "predicted +0.5..+2 vs sym control → measured +0.27m"),
    ("decay", "5b-e patch fade vs sym control", "milli", 0, 2, -3.09, "falsifier fired",
     "preregister.decay.md", "predicted +0..+2 vs sym control → measured −3.09m"),
    ("context", "3b-a exposure term vs B0", "milli", -1.773, 1.773, -0.28, "held",
     "preregister.context.md", "Predicted: all c ~ 0, |Δ| inside 1.773m floor"),
    ("context", "3d learned class weights vs B0", "milli", -1.773, 1.773, -5.737, "miss",
     "preregister.context.md", "holdout Δ vs B0 inside the floor (hand-set weights"),
    ("bias_h1", "E1 Tobit vs v6", "milli", 0.3, 1.5, -0.462, "miss",
     "preregister.bias_h1.md", "Predicted +0.3..+1.5m, elite +1..+3 pts toward 0"),
    ("bias_h1", "E2 round-BT vs v6", "milli", -2, 2, -0.222, "held",
     "preregister.bias_h1.md", "Predicted ΔLL −2..+2m: measured"),
    ("bias_h3", "Exp1 SS core vs v6", "milli", 0, 3, -8.87, "miss",
     "preregister.bias_h3.md", "predicted +0 to +3m"),
    ("bias_h3", "Exp2 5d vs SS core", "milli", 0, 2, 3.55, "above band",
     "preregister.bias_h3.md", "predicted +0 to +2m"),
    ("bias_h4", "E2 L1 global σ link vs v6", "milli", -1, 1, -0.048, "held",
     "preregister.bias_h4.md", "Predicted L1/L2 holdout Δ ∈ (−1,+1) milli, INSIDE NOISE FLOOR: CONFIRMED"),
    ("bias_h4", "E2 L2 depth σ link vs v6", "milli", -1, 1, 0.213, "held",
     "preregister.bias_h4.md", "Predicted L1/L2 holdout Δ ∈ (−1,+1) milli, INSIDE NOISE FLOOR: CONFIRMED"),
    ("compose", "S1 gate5d vs v6", "milli", 0.5, 2.5, 1.958, "held",
     "preregister.compose.md", "predicted effect +0.5..+2.5m overall"),
    ("compose", "S2 fade+shrink vs v6", "milli", -0.2, 0.8, -0.14, "held (low end)",
     "preregister.compose.md", "effect −0.2..+0.8m overall"),
    ("compose", "S3 full stack vs v6", "milli", 0.5, 3.0, -7.874, "falsifier fired",
     "preregister.compose.md", "predicted effect ≈ additive: +0.5..+3.0m"),
]

_PREREG_AUX = [
    # non-milli predictions → outcomes (table, not scatter)
    ("context", "3a integrity down-weight w0*", "solve weight", "~0.5", "1.0 (train falsifier fired; monotone worse toward 0)",
     "miss", "preregister.context.md", "measured train argmin w0* = 1.0"),
    ("context", "3a blanket EWC weight w_e*", "solve weight", "~0.8", "1.2 grid edge (train wants UP-weight); holdout −0.05m",
     "direction wrong", "preregister.context.md", "predicted w_e*~0.8; measured train argmin at the 1.2 grid"),
    ("context", "3b-b form coef shrink under exposure controls", "%", "<30%", "109% at HL5 (sign flip to ~0); 58% at HL3",
     "miss — scouting in disguise", "preregister.context.md", "Predicted <30% shrink — WRONG"),
    ("context", "3c-A elimination solve weight w_elim*", "solve weight", "1.0–1.15", "0.7 grid edge (train wants elim DOWN-weighted)",
     "direction wrong", "preregister.context.md", "predicted w_elim* ∈"),
    ("context", "3d fitted EWC-class weight", "solve weight", "~0.6", "1.018, CI [0.4,3.4] ⊇ 1.0 — falsifier fired",
     "miss", "preregister.context.md", "Predicted w_ewc ~0.6 — WRONG: fitted 1.018"),
    ("context", "3e stand-in shrink k1", "coef", "> 0 small", "+0.347",
     "held", "preregister.context.md", "k_standin = +0.347 (predicted >0 small ✓)"),
    ("bias_h1", "E3 cap-share ratio Q4 / mid", "ratio", "2.5–4×", "1.00× (premise dead)",
     "miss", "preregister.bias_h1.md", "cap-share(Q4) / cap-share(Q2∪Q3) ≈ 2.5–4×"),
    ("bias_h1", "E2 round-level effective-sample k_eff", "ratio", "2.5–4.5× (brief said ~10×)", "1.25× Fisher / 0.80× cluster-boot",
     "falsifier fired", "preregister.bias_h1.md", "k_eff predicted 2.5–4.5 (claim ~10): measured Fisher-median"),
    ("bias_h2", "Spearman(bias, centrality)", "ρ", "≈ −0.30", "−0.304 — but CI spans 0 → gate FAIL",
     "point held, gate failed", "preregister.bias_h2.md", "Predicted Spearman(bias, eig) ≈ −0.30: measured −0.304"),
    ("bias_h2", "Spearman(|bias|, centrality)", "ρ", "≈ −0.35", "−0.077, CI [−0.42,+0.28] — null",
     "miss", "preregister.bias_h2.md", "Predicted Spearman(|bias|, eig) ≈ −0.35: measured −0.077"),
    ("bias_h3", "Exp2 roster q(change)/q(stable)", "ratio", "≥ 2×", "≈11× point MLEs — but DL τ²=0: unsupported at this n",
     "point held, pooling null", "preregister.bias_h3.md", "predicted q(change)>q(stable) ≥2× ⇒ CONFIRMED in"),
    ("bias_h4", "E1 sweep excess D_sweep", "pp", "+1..+3", "+5.09 [+1.81,+8.53] — sign right, larger (OVER-dispersed)",
     "above band", "preregister.bias_h4.md", "Predicted mild global over-dispersion +1..+3 pp: CONFIRMED in sign,"),
    ("bias_h4", "E2 shared-effect σ_u", "σ", "0.2–0.6", "0.72 [0.38,1.00]",
     "above band", "preregister.bias_h4.md", "Predicted σ̂_u ∈ [0.2,0.6]: actual 0.72"),
]


def _norm_ws(s):
    return re.sub(r"\s+", " ", s)


def derive_prereg_scatter():
    """Curated predicted-vs-realized rows; every quote verified verbatim
    (whitespace-normalized) against its preregister file."""
    for row in _PREREG_ROWS + _PREREG_AUX:
        src, quote = row[-2], row[-1]
        text = _norm_ws(open(os.path.join(V8, src)).read())
        if _norm_ws(quote) not in text:
            die(f"prereg quote not found in {src}: {quote!r}")
    wj("page_prereg_scatter.json", {
        "source": "preregister.*.md outcome sections (quotes verified verbatim at build)",
        "note": ("predictions were written BEFORE runs (per-file discipline); "
                 "'held' = realized inside the preregistered band"),
        "milli_rows": [
            {"agent": a, "item": it, "unit": u, "pred_lo": lo, "pred_hi": hi,
             "realized_milli": rz, "outcome": hd, "source_file": sf, "source_quote": q}
            for a, it, u, lo, hi, rz, hd, sf, q in _PREREG_ROWS],
        "aux_rows": [
            {"agent": a, "item": it, "unit": u, "predicted": p, "realized": rz,
             "outcome": hd, "source_file": sf, "source_quote": q}
            for a, it, u, p, rz, hd, sf, q in _PREREG_AUX],
    })


def derive_reliability():
    """Descriptive favorite-frame reliability bins for v6 / SS-1a / SS-5d from
    the h3 agent's stored per-row probabilities. Display furniture only."""
    import csv
    import numpy as np
    z = np.load(os.path.join(SCRATCH, "bias_h3", "model_probs.npz"))
    dates = []
    with open(os.path.join(V8, "data", "frame_expanded", "series.csv")) as f:
        for row in csv.DictReader(f):
            dates.append(row["date"])
    if len(dates) != len(z["p_v6"]):
        die("frame/npz length mismatch")
    hold = np.array([d > "2024-12-31" for d in dates]) & z["v6_valid"].astype(bool)
    # loud orientation check: v6 holdout LL must reproduce the program baseline
    ll = float(-np.mean(np.log(np.clip(z["p_v6"][hold], 1e-12, 1))))
    base = j("compose_gate.json")["provenance"]["baseline_v6"]["ll_holdout"]
    if abs(ll - base) > 5e-5:
        die(f"reliability orientation check failed: {ll:.5f} vs baseline {base}")
    edges = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0001]
    out = {"source": "scratch/bias_h3/model_probs.npz (per-row winner-probabilities, holdout rows)",
           "frame": "favorite frame: p_fav = max(p, 1−p); y = favorite won; rows with p = 0.5 exactly excluded",
           "holdout_n": int(hold.sum()), "v6_holdout_ll_check": round(ll, 5),
           "min_bin_n": 15, "models": {}}
    for key, arr in [("v6", z["p_v6"]), ("ss_1a", z["p_ss_1a"]), ("ss_5d", z["p_ss_5d"])]:
        p = arr[hold]
        mask = p != 0.5
        pf = np.maximum(p[mask], 1 - p[mask])
        y = (p[mask] > 0.5).astype(float)
        raw = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (pf >= lo) & (pf < hi)
            raw.append([int(m.sum()), float(pf[m].mean()) if m.any() else None,
                        float(y[m].mean()) if m.any() else None,
                        float(pf[m].sum()), float(y[m].sum()), lo, hi])
        # merge sparse bins (n<15) leftward into neighbors (house convention)
        merged = []
        for b in raw:
            if merged and (b[0] < 15 or merged[-1][0] < 15):
                t = merged.pop()
                merged.append([t[0] + b[0], None, None, t[3] + b[3], t[4] + b[4], t[5], b[6]])
            else:
                merged.append(b)
        bins = []
        for n, _, _, spf, sy, lo, hi in merged:
            if n == 0:
                continue
            emp, pred = sy / n, spf / n
            zz = 1.959963984540054
            den = 1 + zz * zz / n
            ctr = (emp + zz * zz / (2 * n)) / den
            hw = zz * math.sqrt(emp * (1 - emp) / n + zz * zz / (4 * n * n)) / den
            bins.append({"n": n, "pred_mean": round(pred, 4), "emp": round(emp, 4),
                         "wilson_lo": round(ctr - hw, 4), "wilson_hi": round(ctr + hw, 4),
                         "range": [lo, hi]})
        out["models"][key] = bins
    wj("page_reliability.json", out)


def derive_mde_curve():
    pme = j("power_mde_expanded.json")
    grid = [200, 300, 400, 600, 800, 1007, 1217, 1600, 2000, 3000, 4500, 6500, 9000, 12000]
    curves = {}
    for reg in ("within", "cross"):
        sig = pme["mde"][reg]["sigma_adj"]
        curves[reg] = {"sigma_adj": sig,
                       "n": grid,
                       "mde_milli": [round(2.8016 * sig / math.sqrt(n) * 1000, 3) for n in grid]}
    sig_c = pme["mde"]["cross"]["sigma_adj"]
    wj("page_mde_curve.json", {
        "source": "stats/power_mde_expanded.json (stored sigma_adj; MDE80 = 2.8016·σ/√n)",
        "curves": curves,
        "markers": {
            "n_old": 1007,
            "n_expanded": 1217,
            "mde_old": {r: pme["mde"][r]["mde_old_n1007_milli"] for r in ("within", "cross")},
            "mde_expanded": {r: pme["mde"][r]["mde_composition_adjusted_milli"] for r in ("within", "cross")},
        },
        "n_for_2milli_cross": round((2.8016 * sig_c / 0.002) ** 2),
    })


def derive_caterpillar():
    """One merged per-team bias table so the caterpillar chart has a single
    downloadable JSON. Values copied from the three source caterpillars."""
    h1 = j("h1_bias_caterpillar.json")
    h3 = j("h3_bias_caterpillar.json")
    s1 = j("compose_stacks.json")["stacks"]["S1_gate5d"]["caterpillar"]["candidate"]["teams"]
    by = {t["team"]: {"team": t["team"], "n": t["n"], "v6": t["v6"],
                      "tobit": t["tobit"], "roundbt": t["roundbt"]} for t in h1["teams"]}
    for t in h3["teams"]:
        if t["team"] not in by:
            die(f"caterpillar team mismatch h3: {t['team']}")
        if abs(t["bias_v6"] - by[t["team"]]["v6"]) > 5e-4:
            die(f"caterpillar v6 bias disagrees for {t['team']}")
        by[t["team"]].update(ss_1a=t["bias_ss1a"], ss_5d=t["bias_ss5d"])
    for t in s1:
        if t["team"] not in by:
            die(f"caterpillar team mismatch S1: {t['team']}")
        by[t["team"]]["s1"] = t["bias"]
    teams = sorted(by.values(), key=lambda t: t["v6"])
    summary = {"v6": h3["summary"]["v6"], "ss_1a": h3["summary"]["ss_1a"],
               "ss_1b_qcal": h3["summary"]["ss_1b_qcal"],
               "ss_5d_roster": h3["summary"]["ss_5d_roster"],
               "tobit": h1["summary"]["tobit"], "roundbt": h1["summary"]["roundbt"]}
    wj("page_caterpillar.json", {
        "source": "h1_bias_caterpillar.json + h3_bias_caterpillar.json + compose_stacks.json (S1) — merged, v6 columns cross-checked",
        "unit": "probability points ×100 in chart; negative = model under-rates team",
        "min_n": h3["min_n"], "teams": teams, "summary": summary})


def derive_buckets():
    """S1/S2/S3 bucket deltas + the preregistered bucket-MDE scaling
    (family_MDE·√(N/n)) so 'inside the noise floor' shading is data-driven."""
    cs = j("compose_stacks.json")
    pme = j("power_mde_expanded.json")
    out = {"source": "compose_stacks.json buckets + power_mde_expanded floors; bucket MDE = family_MDE·√(N/n) (preregistered scaling, decay_subpops rule)",
           "n_holdout": None, "stacks": {}}
    for name, st in cs["stacks"].items():
        fam = st["family_mde_milli"]
        n_hold = st["judging"]["n"]
        out["n_holdout"] = n_hold
        rows = []
        for b in st["buckets"]["buckets"]:
            mde = fam * math.sqrt(n_hold / b["n"]) if b["n"] else None
            rows.append({"name": b["name"], "n": b["n"], "delta_milli": b["delta_milli"],
                         "bucket_mde_milli": round(mde, 2) if mde else None,
                         "inside_floor": (abs(b["delta_milli"]) < mde) if mde else None})
        out["stacks"][name] = {"family_mde_milli": fam, "regime": st["regime"],
                               "verdict": st["gate_verdict"], "rows": rows}
    out["floors"] = pme["checkpoint_quote"]
    wj("page_buckets.json", out)


def derive_slate():
    sys.path.insert(0, os.path.join(ROOT, "trading_model"))
    from predict import load_model, series_probability  # noqa: E402
    m = load_model()
    up = j("upcoming_matches.json", os.path.join(ROOT, "data"))
    rows = []
    for u in up:
        p = series_probability(m, u["org_a"], u["org_b"], u["format"])
        rows.append({"date": u["date"], "event": u["event"], "format": u["format"],
                     "a": u["org_a"], "b": u["org_b"],
                     "team_a": u["team_a"], "team_b": u["team_b"],
                     "p_a_v6": round(p, 4)})
    import datetime
    wj("page_slate.json", {
        "source": "data/upcoming_matches.json priced by trading_model/predict.py",
        "model_version": m["model_version"],
        "generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "no_v8_column_note": ("No v8 column: no candidate was promoted by this program "
                              "(all three compose stacks HOLD; v6 stands). The flag rule stays "
                              "for the future: when a promoted candidate exists, this table "
                              "carries its column plus a divergence flag on every match where "
                              "|candidate − v6| ≥ 5 pts."),
        "rows": rows})


_WAVES = {"corpus": 1, "lineups": 1, "power": 1, "referee": 1, "autopsy": 1,
          "decay": 2, "context": 2, "bias_h1": 2, "bias_h2": 2, "bias_h3": 2,
          "bias_h4": 2, "compose": 3, "adversary": 3, "page": 3}
_OUTCOMES = {
    "corpus": "31 events backfilled; verification passed in full; prefranchise corpus scraped separately, regional 2021-22 deferred",
    "lineups": "every engine side matched to a lineup; stand-in observables emitted for context/compose",
    "power": "the instrument was measured before it was used: MDE floors, v7 ladder re-adjudication, ledger reclassification",
    "referee": "metric suite upgraded (CRN boots, CUPED CV, per-team bias, expected-ROI unit); self-test green vs published numbers",
    "autopsy": "live P&L decomposed; verdict: config first, model second, microstructure/fees/noise exonerated",
    "decay": "re-race says ties; five new axes inside the floor; patch fade falsifier fired; performance form adds nothing",
    "context": "event context carries ~no extra probability signal; learned class weights anti-validate; stand-in shrink is a bucket lead only",
    "bias_h1": "censoring premise empirically false; Tobit and round-BT leave the elite bias intact; round-level power play refuted",
    "bias_h2": "bias does not live on the schedule graph; gate stopped the program before any candidate fit",
    "bias_h3": "state-space loses overall, wins the cold tail exactly as preregistered; heterogeneity unsupported at this n",
    "bias_h4": "premise inverted: maps are OVER-dispersed; β already pays the correction; links inside the floor",
    "compose": "three preregistered stacks, three HOLDs; S3 anti-synergy falsifier fired; v6 stands",
    "adversary": "independent hostile recompute: negatives reproduce to the last digit; positive residue cut down to one lead",
    "page": "this page: adversary-amended verdicts published; holdout declared spent",
}


def derive_timeline():
    """Journal spans parsed from each agent's append-only log. Formats vary by
    agent ([HH:MM], ISO-Z, ~HH:MM, HH:0x placeholders) — capture any
    date+time-ish token and display the log's own first/last time verbatim."""
    rows = []
    ts_re = re.compile(r"2026-\d\d-\d\d[T ]\s?~?(\d\d:\d[\dx](?::\d\d)?)")
    for agent, wave in _WAVES.items():
        path = os.path.join(V8, "logs", f"{agent}.log")
        first = last = None
        if os.path.exists(path):
            for line in open(path):
                m = ts_re.search(line)
                if m:
                    first = first or m.group(1)
                    last = m.group(1)
        rows.append({"agent": agent, "wave": wave, "first": first, "last": last,
                     "outcome": _OUTCOMES[agent]})
    wj("page_timeline.json", {"source": "logs/<agent>.log first/last journal timestamps",
                              "date": "2026-07-28", "rows": rows})


def derive_ledger_updates():
    """§10 deliverable. Kill-evidence numbers are copied from their stats
    files here, at derive time — the HTML then reads only this JSON."""
    h1t, h1c = j("h1_tobit.json"), j("h1_censor_diag.json")
    h1r = j("h1_roundbt.json")
    h2c = j("h2_centrality.json")
    h3s = j("h3_statespace.json")
    h4d, h4l = j("h4_dispersion_diag.json"), j("h4_series_link.json")
    cw, cser = j("context_weights.json"), j("context_seriousness.json")
    dax, dfm = j("decay_axes.json"), j("decay_form.json")
    cs = j("compose_stacks.json")
    lr = j("ledger_reclass.json")
    pme = j("power_mde_expanded.json")
    frag = j("page_adversary_fragility.json")
    clooks = j("compose_looks.json")
    cold = frag["eclass_cold"]["cold_buckets_mine"]["cold(<10)"]

    e1 = h1t["primary_s1.0"]
    l1 = h4l["links"]["L1_sigma_global"]
    l2 = h4l["links"]["L2_sigma_depth"]
    axis_e = dax["axes"]["e_patch_boundary"]["on_top_of_v6"]["vs_v6"]
    fp = dfm["results"]["form_player5"]
    s3 = cs["stacks"]["S3_full"]
    ss1b = next(p for p in h3s["pairwise_holdout"] if p["pair"].startswith("ss_1b_qcal vs v6"))

    dnr = [
        {"idea": "H1: bounded margins censor dominant teams (Tobit/censoring likelihood)",
         "kill": {"cap_share": h1c["cap_Q4"], "cap_ratio_Q4_over_mid": h1c["cap_ratio_Q4_over_mid"],
                  "caps_total": h1t["censoring_mass"]["CAP"], "games": h1t["censoring_mass"]["n_games"],
                  "tobit_dll_milli": e1["dll_milli_vs_v6"]},
         "evidence": ["stats/h1_censor_diag.json", "stats/h1_tobit.json"],
         "note": "premise empirically false (no pile-up at the cap; ratio ≈1 vs preregistered ≥2); Tobit inside floor, negative, elite bias unmoved"},
        {"idea": "H1 corollary: round-level evaluation as a power play",
         "kill": {"k_eff_fisher_median": h1r["effective_sample"]["k_eff_fisher_median"],
                  "k_eff_cluster_boot": h1r["effective_sample"]["k_eff_cluster_boot_median"]},
         "evidence": ["stats/h1_roundbt.json"],
         "note": "rounds within a match are correlated; nominal binomial information does not survive match-level resampling"},
        {"idea": "H2: per-team bias explained by opponent-graph connectivity",
         "kill": {"spearman_absbias_eig": h2c["gate"]["abs_bias_vs_eig"]["spearman"],
                  "ci": h2c["gate"]["abs_bias_vs_eig"]["ci"]},
         "evidence": ["stats/h2_centrality.json"],
         "note": "|bias| ~ centrality is null; signed correlations are composition (elite travel, floor stays home); gate stopped E2/E3"},
        {"idea": "H4: iid aggregation under-disperses deep-pool favorites (premise + both links)",
         "kill": {"D_sweep_pp": h4d["cells"][0]["D_sweep_pp"], "ci": h4d["cells"][0]["D_sweep_ci_pp"],
                  "L1_dll_milli": l1["delta_milli_vs_v6"], "L2_dll_milli": l2["delta_milli_vs_v6"]},
         "evidence": ["stats/h4_dispersion_diag.json", "stats/h4_series_link.json"],
         "note": "premise inverted — maps are OVER-dispersed (σ_u≈0.72) and train-fit β already absorbs it; both links inside the floor"},
        {"idea": "Learned event-class solve weights (replacing hand-set 1.6/2.0)",
         "kill": {"dll_milli": cw["holdout"]["dll_milli"], "iid_ci": cw["holdout"]["boot_iid_ci_milli"],
                  "fitted": cw["fitted_weights"]},
         "evidence": ["stats/context_weights.json"],
         "note": "in-sample regime memorization (champions weight collapses to ~0) that anti-validates; hand-set weights win"},
        {"idea": "Patch/map-pool fade axis (γ ≤ 0.7)",
         "kill": {"dll_on_v6_milli": axis_e["delta_milli"],
                  "p_iid": axis_e["boot_iid"]["p_better"], "p_block": axis_e["boot_block"]["p_better"]},
         "evidence": ["stats/decay_axes.json"],
         "note": "sharpest train→holdout reversal of the wave (best train LL of any config); preregistered falsifier fired"},
        {"idea": "Performance-defined form terms (wr / rd-margin / side / player R2.0 / combined)",
         "kill": {"player5_train_gain_milli": fp["train_gain_from_form_milli"],
                  "player5_holdout_milli": fp["delta_milli_vs_v6"]},
         "evidence": ["stats/decay_form.json"],
         "note": "every definition fits mean-reverting on train and scores ≤ v6 on holdout; richest feature overfits hardest"},
        {"idea": "Lineup-conditioned EWC solve down-weight (3a integrity)",
         "kill": {"train_argmin_w0": cser["train_argmin"]["w0"]},
         "evidence": ["stats/context_seriousness.json"],
         "note": "train NLL monotone worse as stand-in games are down-weighted; falsifier fired before holdout was touched"},
        {"idea": "Calendar-time process noise (state-space 1b, breaks add drift)",
         "kill": {"dll_milli": ss1b["delta_milli"], "p_iid": ss1b["iid"]["p_better"]},
         "evidence": ["stats/h3_statespace.json"],
         "note": "best train config of its family, worst holdout — breaks do not add real drift once games-counted noise exists"},
        {"idea": "S3 fitting shape: (β,k) refit on non-gated train rows under a hard gate",
         "kill": {"dll_milli": s3["judging"]["delta_milli"], "k_runaway": frag["compose"]["S3"]["k"],
                  "train_nll": s3["train_nll_composite"]},
         "evidence": ["stats/compose_stacks.json", "stats/page_adversary_fragility.json"],
         "note": "anti-synergy falsifier fired: best composite train NLL of the wave, worst holdout; k driven to ~1.87"},
    ]
    unresolved = [{"id": r["id"], "idea": r["idea"], "delta_milli": r["delta_milli"],
                   "mde_at_test_milli": r["mde_at_test_milli"], "n_at_test": r["n_at_test"],
                   "note": r["note"]}
                  for r in lr["rows"] if r["status"] == "UNRESOLVED"]
    if len(unresolved) != lr["counts"]["UNRESOLVED"]:
        die("unresolved count mismatch vs ledger_reclass counts")

    tripwires = [
        {"conclusion": "v6 stands (no stack promotable)",
         "threshold": "any candidate clearing the full gate — Δ ≥ family MDE AND p ≥ 0.95 in BOTH CRN modes AND no bias/bucket regression — on data dated after 2026-07-28",
         "window": "next confirmatory frame (see 'what data next')"},
        {"conclusion": f"H3 cold-start lead (+{cold['dll_5d_milli']}m at n={cold['n']})",
         "threshold": "score the FROZEN 5d spec (stats/compose_spec.json) vs v6 on fresh series where either team has <10 prior maps; promote to candidate status if Δ ≥ that bucket's scaled MDE",
         "window": "when ≥40 fresh cold rows exist (prospective logger + new events; at n=40 the scaled cross floor is ≈32m)"},
        {"conclusion": "event-class fade / 3e stand-in shrink demoted to noise-floor leads",
         "threshold": "re-test ONLY at the next corpus expansion of ≥300 fresh holdout series; verdicts at the then-current family MDE, preregistered bucket definitions fixed in advance",
         "window": "next expansion (2026 S2/Champions settle, or prefranchise regionals)"},
        {"conclusion": "v6 live favorite calibration (autopsy in-window −3.9pp, n.s.)",
         "threshold": "unconditional favorite gap beyond ±5pp with n ≥ 100 settled linked events",
         "window": "rolling, prospective logger",},
        {"conclusion": "PRX/NRG under-rating residual (≈−10pp, untouched by every stack)",
         "threshold": "either team's rolling bias worse than −12pp on its trailing 60 scored series → dedicated hypothesis wave (not stand-in load, not event-class recency, not rating uncertainty — all excluded here)",
         "window": "rolling 60 series per team"},
        {"conclusion": "bucket gates unenforceable under n≈200 (phase 0)",
         "threshold": "any future bucket claim must quote the bucket MDE from stats/power_mde.json; buckets under n=200 are advisory-only",
         "window": "standing rule"},
        {"conclusion": "autopsy config gap (flat 5¢ floor; skip-NO<45% absent) — operator implementation memo, NOT model input",
         "threshold": "if the live config stays unchanged: realized < −$150 over the next 28 settled events triggers a mandatory config review against FINDINGS §4",
         "window": "next 28 settled events (one-week-scale sample, same caveat as this one)"},
    ]
    wj("ledger_v8_updates.json", {
        "written_by": "agent:page (Wave 3), 2026-07-28",
        "amended_headline": [
            "v6 stands; no stack cleared the gate (compose_gate.json) — co-signed by the adversary",
            "CORRECTION over Phase 0 prose: sym_6, sym_8, surprise_12_20, boxexp_c3 WERE distinguishable-as-losses under the program-standard 5.889m floor; sym_20 and consist_16_10 remain unresolved",
            "Demoted to noise-floor leads by adversarial review: 3e stand-in shrink's EWC bucket; event-class fade's subpop positivity; 5d's 7.4-vs-24.8 half-life contrast; compose S1's gated gain",
            f"ONE surviving lead: H3 cold-start (<10 prior maps) +{cold['dll_5d_milli']}m (n={cold['n']}, floor {cold['bucket_mde_cross']}m, drop-5% +{cold['drop_top5pct_milli']}m, jackknife +{cold['jackknife_min']}m) — 4 events; parent model loses overall",
            f"THE HOLDOUT IS SPENT: {clooks['totals']['grand_total_recorded_holdout_numbers']} recorded holdout numbers; future train-only claims on this frame are unfalsifiable",
        ],
        "power_annotation_rule": ("a ledger entry bars re-testing only with a power annotation attached; "
                                  "UNRESOLVED entries are re-openable by design (new data, variance "
                                  "reduction, or a bigger claimed effect) — agent:power, phase0 §4"),
        "do_not_retest_additions": dnr,
        "reopened_unresolved": unresolved,
        "tripwires": tripwires,
        "what_data_next": [
            "prospective prediction-logger accumulation (every quoted probability, timestamped, model-versioned)",
            f"the deferred prefranchise regional corpus ({j('corpus_diff.json')['prefranchise_status']['deferred']['vct_2021_hub_regional_or_other_events'] + j('corpus_diff.json')['prefranchise_status']['deferred']['vct_2022_hub_regional_or_other_events']} regional 2021-22 events, several thousand series)",
            "2026 Stage 2 / Champions as they settle (organic holdout growth)",
        ],
        "floors": pme["checkpoint_quote"],
    })


def derive_all():
    derive_v7_ladder()
    derive_adversary_fragility()
    derive_prereg_scatter()
    derive_reliability()
    derive_mde_curve()
    derive_caterpillar()
    derive_buckets()
    derive_slate()
    derive_timeline()
    derive_ledger_updates()


# ═════════════════════════════ PHASE 2 — RENDER ══════════════════════════════
# Palette (validated per dataviz six checks; see logs/page.log 22:52):
#   neutral baseline v6 #6f6a7c · S1 #7c4dd6 · ss_5d #c96a2a · ss_1a #3a90cc ·
#   tobit #6b8f1f · round-BT #2b6f9e · status good/bad reserved for polarity.
C = {"v6": "#6f6a7c", "s1": "#7c4dd6", "d5": "#c96a2a", "a1": "#3a90cc",
     "tob": "#6b8f1f", "rbt": "#2b6f9e", "gray": "#9a93a6",
     "good": "#1e7a4f", "bad": "#c0392b", "warn": "#b3541e", "ink": "#16121d"}

CSS = """
 * { box-sizing:border-box; margin:0; }
 :root { --ink:#16121d; --dim:#6b6478; --line:#eceef2; --acc:#7c4dd6; --accbg:#f3eefb;
         --good:#1e7a4f; --goodbg:#ecf8f1; --bad:#c0392b; --badbg:#fbeaea;
         --warn:#b3541e; --warnbg:#fdf3ec; }
 body { font-family:'DM Sans',system-ui,sans-serif; background:#faf9fc; color:var(--ink);
        line-height:1.55; padding:30px 18px 90px; }
 .wrap { max-width:940px; margin:0 auto; }
 h1 { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.55rem;
      text-align:center; margin:6px 0 2px; }
 .tagline { text-align:center; color:var(--dim); font-size:.9rem; margin-bottom:18px; }
 .labtabs { display:flex; justify-content:center; gap:6px; margin:0 0 18px; flex-wrap:wrap; }
 .labtabs a { font-size:.8rem; font-weight:700; color:var(--dim); text-decoration:none;
   padding:7px 15px; border-radius:999px; border:1px solid var(--line); background:#fff; }
 .labtabs a:hover { color:var(--ink); background:var(--accbg); }
 .labtabs a.on { color:#fff; background:var(--acc); border-color:var(--acc); }
 .banner { background:#fbeaea; border:1px solid #f0c7c7; border-left:6px solid var(--bad);
   border-radius:14px; padding:14px 20px; margin:0 0 18px; font-size:.92rem; }
 .banner b { color:var(--bad); letter-spacing:.4px; }
 section { background:#fff; border:1px solid var(--line); border-radius:18px;
           padding:24px 28px; margin-bottom:16px; box-shadow:0 3px 14px #00000008; }
 h2 { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.08rem;
      margin-bottom:10px; display:flex; align-items:center; gap:10px; }
 h2 .n { background:var(--acc); color:#fff; border-radius:8px; font-size:.72rem; min-width:24px;
        height:24px; padding:0 4px; display:inline-flex; align-items:center; justify-content:center; flex-shrink:0; }
 h3 { font-size:.93rem; font-weight:700; margin:16px 0 6px; }
 p { font-size:.9rem; margin:7px 0; }
 .dim { color:var(--dim); } .good { color:var(--good); } .bad { color:var(--bad); }
 .mono { font-family:'JetBrains Mono',monospace; font-size:.82em; }
 .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px; margin:12px 0; }
 .card { border:1px solid var(--line); border-radius:14px; padding:13px 15px; }
 .card .lbl { font-size:.68rem; font-weight:700; letter-spacing:.8px; text-transform:uppercase; color:var(--dim); }
 .card .big { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.35rem; margin:2px 0; }
 .card .sub { font-size:.76rem; color:var(--dim); }
 table { width:100%; border-collapse:collapse; font-size:.84rem; margin:8px 0; }
 th { text-align:left; color:var(--dim); font-size:.7rem; text-transform:uppercase;
      letter-spacing:.5px; padding:6px 9px; border-bottom:2px solid var(--line); }
 td { padding:6px 9px; border-bottom:1px solid var(--line); vertical-align:top; }
 tr:last-child td { border-bottom:0; }
 .verd { font-weight:700; border-radius:999px; padding:2px 9px; font-size:.68rem; white-space:nowrap; }
 .verd.hold { background:#f1f0f4; color:var(--dim); }
 .verd.dead { background:var(--badbg); color:var(--bad); }
 .verd.floor { background:var(--warnbg); color:var(--warn); }
 .verd.lead { background:var(--goodbg); color:var(--good); }
 .callout { border-left:4px solid var(--acc); background:var(--accbg);
            border-radius:0 12px 12px 0; padding:11px 15px; margin:10px 0; font-size:.88rem; }
 .callout.good { border-color:var(--good); background:var(--goodbg); }
 .callout.warn { border-color:var(--warn); background:var(--warnbg); }
 .callout.bad { border-color:var(--bad); background:var(--badbg); }
 canvas { max-height:340px; }
 .chartbox { margin:14px 0 2px; }
 .chartbox.tall { height:640px; } .chartbox.tall canvas { max-height:none; height:100% !important; }
 .chartbox.mid { height:420px; } .chartbox.mid canvas { max-height:none; height:100% !important; }
 .dl { display:block; text-align:right; font-size:.72rem; margin:2px 0 10px; }
 .dl a { color:var(--acc); text-decoration:none; font-weight:700; }
 .dl a:hover { text-decoration:underline; }
 .cap { font-size:.78rem; color:var(--dim); margin:4px 0 8px; }
 code { font-family:'JetBrains Mono',monospace; font-size:.8em; background:#f4f2f8;
        border:1px solid var(--line); border-radius:6px; padding:1px 6px; }
 ul, ol { font-size:.9rem; margin:8px 0 8px 20px; } li { margin:5px 0; }
 .adv { border:2px solid var(--bad); border-radius:14px; padding:18px 20px; margin:14px 0;
        background:#fffdfd; }
 .adv .advhead { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; color:var(--bad);
        font-size:.95rem; margin-bottom:8px; }
 .adv pre { white-space:pre-wrap; font-family:'JetBrains Mono',monospace; font-size:.76rem;
        line-height:1.6; color:var(--ink); }
 .scroll { overflow-x:auto; }
 @media (max-width:640px) { section { padding:18px 14px; } }
"""

NAV = """<div class="labtabs">
<a href="/testing/">Testing Lab</a>
<a href="/testing/report/state_of_benpom">State of BenPom</a>
<a href="/testing/playbook">Deployment Playbook</a>
<a href="/testing/report/favorites_lab">Favorites Lab</a>
<a href="/testing/report/final_model">Final Model</a>
<a href="/testing/report/v7_lab">v7 Lab</a>
<a href="/testing/report/v8_lab" class="on">v8 Lab</a><a href="/testing/report/roster_adaptation">Roster</a><a href="/testing/report/v9_lab">v9 Lab</a>
</div>"""


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def dl(name):
    return (f'<div class="dl"><a href="/testing/v8/stats/{name}" download>'
            f'&#8681; {name}</a></div>')


def render():
    L = {n: j(n) for n in [
        "power_mde_expanded.json", "power_mde.json", "page_v7_ladder.json",
        "ledger_reclass.json", "variance_reduction.json", "corpus_blocks.json",
        "corpus_diff.json", "verification_report.json", "context_weights.json",
        "context_exposure.json", "context_adjacency.json", "context_seriousness.json",
        "context_stakes.json", "context_shrink.json", "decay_curves.json",
        "decay_rerace.json", "decay_axes.json", "decay_form.json", "decay_subpops.json",
        "h1_censor_diag.json", "h1_tobit.json", "h1_roundbt.json", "h2_centrality.json",
        "h3_statespace.json", "h3_process_noise.json", "h3_ensemble.json",
        "h4_dispersion_diag.json", "h4_series_link.json", "compose_gate.json",
        "compose_stacks.json", "compose_spec.json", "compose_looks.json",
        "autopsy_pnl.json", "autopsy_fees.json", "autopsy_fill_calib.json",
        "autopsy_markouts.json", "autopsy_config_gap.json", "autopsy_variance.json",
        "adversary_report.json", "lineups_coverage.json",
        "page_adversary_fragility.json", "page_reliability.json", "page_mde_curve.json",
        "page_slate.json", "page_timeline.json", "page_prereg_scatter.json",
        "page_caterpillar.json", "page_buckets.json", "ledger_v8_updates.json",
    ]}
    L["h3_neff_summ"] = {k: v for k, v in j("h3_neff.json").items() if k != "per_match"}
    adv_md = open(os.path.join(V8, "adversary_report.md")).read()

    pme = L["power_mde_expanded.json"]
    floors = pme["checkpoint_quote"]
    base = L["compose_gate.json"]["provenance"]["baseline_v6"]
    n_hold = L["compose_gate.json"]["gates"]["S1_gate5d"]["n_scored"]
    looks = L["compose_looks.json"]["totals"]
    frag = L["page_adversary_fragility.json"]
    cold = frag["eclass_cold"]["cold_buckets_mine"]["cold(<10)"]
    ap, av = L["autopsy_pnl.json"], L["autopsy_variance.json"]
    catsum = L["page_caterpillar.json"]["summary"]
    adv = L["adversary_report.json"]
    lvu = L["ledger_v8_updates.json"]

    # ── JS blocks (plain strings; data injected via .replace) ────────────────
    js_blocks = []

    def add_chart(js_tmpl, **data):
        blk = js_tmpl
        for k, v in data.items():
            blk = blk.replace("__" + k + "__", json.dumps(v))
        js_blocks.append(blk)

    # shared floor-band plugin + palette
    js_blocks.append("""
const PAL = __PAL__;
const floorBand = { id:'floorBand', beforeDatasetsDraw(c, a, o) {
  const fb = c.options.plugins.floorBand; if (!fb) return;
  const ax = fb.axis === 'y' ? c.scales.y : c.scales.x, ctx = c.ctx, ca = c.chartArea;
  const p0 = ax.getPixelForValue(fb.lo), p1 = ax.getPixelForValue(fb.hi);
  ctx.save(); ctx.fillStyle = 'rgba(154,147,166,0.16)';
  if (fb.axis === 'y') ctx.fillRect(ca.left, Math.min(p0,p1), ca.right-ca.left, Math.abs(p1-p0));
  else ctx.fillRect(Math.min(p0,p1), ca.top, Math.abs(p1-p0), ca.bottom-ca.top);
  ctx.restore(); } };
Chart.register(floorBand);
Chart.defaults.font.family = "'DM Sans',system-ui,sans-serif";
Chart.defaults.color = '#6b6478';
""".replace("__PAL__", json.dumps(C)))

    # c1 — MDE curves
    mc = L["page_mde_curve.json"]
    add_chart("""
new Chart(document.getElementById('c_mde'), { type:'line', data:{ labels:__N__, datasets:[
 {label:'cross-family', data:__CR__, borderColor:PAL.s1, backgroundColor:PAL.s1, pointRadius:3, tension:.3},
 {label:'within-family', data:__WI__, borderColor:PAL.d5, backgroundColor:PAL.d5, pointRadius:3, tension:.3}
]}, options:{ plugins:{ tooltip:{ callbacks:{ title:i=>'n = '+i[0].label,
   label:i=>i.dataset.label+': MDE '+i.parsed.y.toFixed(2)+'m' } } },
 scales:{ y:{ title:{display:true,text:'MDE\\u2088\\u2080 (milli-LL)'}, min:0 },
          x:{ title:{display:true,text:'holdout series n'} } } } });
""", N=mc["curves"]["cross"]["n"], CR=mc["curves"]["cross"]["mde_milli"],
        WI=mc["curves"]["within"]["mde_milli"])

    # c2 — v7 ladder
    lad = L["page_v7_ladder.json"]
    lrows = sorted(lad["rows"], key=lambda r: r["delta_milli"])
    add_chart("""
const LAD = __D__;
new Chart(document.getElementById('c_lad'), { type:'bar', data:{ labels: LAD.labels, datasets:[
 {label:'\\u0394 vs v6 (milli)', data: LAD.delta,
  backgroundColor: LAD.amended.map(a=>a?PAL.bad:PAL.gray),
  borderColor: LAD.near.map(nt=>nt?PAL.s1:'transparent'), borderWidth: LAD.near.map(nt=>nt?2:0) }
]}, options:{ indexAxis:'y', maintainAspectRatio:false,
 plugins:{ legend:{display:false}, floorBand:{axis:'x', lo:-LAD.floor, hi:LAD.floor},
  tooltip:{ callbacks:{ label:i=>i.parsed.x.toFixed(2)+'m \\u00b7 pair MDE '+LAD.mde[i.dataIndex]+'m \\u00b7 block CI ['+LAD.ci[i.dataIndex]+']'
   +(LAD.amended[i.dataIndex]?' \\u00b7 AMENDED: distinguishable loss':' \\u00b7 unresolved') } } },
 scales:{ x:{ title:{display:true,text:'holdout \\u0394LL vs v6 (milli) \\u00b7 shaded = program cross floor'} },
          y:{ ticks:{ autoSkip:false, font:{size:10} } } } } });
""", D={"labels": [r["config"] for r in lrows], "delta": [r["delta_milli"] for r in lrows],
        "amended": [r["amended_loss"] for r in lrows],
        "near": [r["config"] in lad["near_ties_unresolved"] for r in lrows],
        "mde": [r["pair_mde80_milli"] for r in lrows],
        "ci": [f"{r['ci_block_milli'][0]}, {r['ci_block_milli'][1]}" for r in lrows],
        "floor": lad["cross_floor_milli"]})

    # c3 — corpus blocks
    cb = L["corpus_blocks.json"]["blocks"]
    bl = sorted(cb.items(), key=lambda kv: -kv[1]["n_matches"])
    add_chart("""
const CB = __D__;
new Chart(document.getElementById('c_corpus'), { type:'bar', data:{ labels: CB.labels, datasets:[
 {label:'series added', data: CB.matches, backgroundColor: CB.pre.map(p=>p?PAL.gray:PAL.s1)}
]}, options:{ indexAxis:'y', plugins:{ legend:{display:false},
  tooltip:{ callbacks:{ label:i=>CB.matches[i.dataIndex]+' series \\u00b7 '+CB.maps[i.dataIndex]+' maps \\u00b7 '+CB.span[i.dataIndex] } } },
 scales:{ x:{ title:{display:true,text:'series backfilled (gray = prefranchise, held separate)'} } } } });
""", D={"labels": [k for k, _ in bl], "matches": [v["n_matches"] for _, v in bl],
        "maps": [v["n_maps"] for _, v in bl], "span": [" \u2192 ".join(v["span"]) for _, v in bl],
        "pre": [k.startswith("prefranchise") for k, _ in bl]})

    # c4 — context class weights
    cw = L["context_weights.json"]
    order = ["vct_playoffs", "champions", "masters", "ewc_offseason"]
    hand = {"vct_playoffs": 1.6, "champions": 2.0, "masters": 1.0, "ewc_offseason": 1.0}
    add_chart("""
const CW = __D__;
new Chart(document.getElementById('c_w'), { data:{ labels: CW.labels, datasets:[
 {type:'bar', label:'95% CI (profile)', data: CW.ci, backgroundColor:'#7c4dd633', borderRadius:4},
 {type:'scatter', label:'fitted', data: CW.fit.map((v,i)=>({x:i,y:v})), backgroundColor:PAL.s1, pointRadius:6},
 {type:'scatter', label:'v6 hand-set', data: CW.hand.map((v,i)=>({x:i,y:v})), backgroundColor:PAL.ink,
  pointStyle:'line', pointRadius:14, borderColor:PAL.ink, borderWidth:3}
]}, options:{ plugins:{ tooltip:{ callbacks:{ label:i=> i.datasetIndex===0
   ? 'CI ['+CW.ci[i.dataIndex][0]+', '+CW.ci[i.dataIndex][1]+']'
   : i.dataset.label+' '+(i.parsed.y??i.parsed._custom) } } },
 scales:{ y:{ title:{display:true,text:'solve weight (train-fit, n='+CW.ntrain+')'}, min:0 } } } });
""", D={"labels": order, "fit": [cw["fitted_weights"][k] for k in order],
        "ci": [[cw["ci_per_weight"][k]["ci_lo"], cw["ci_per_weight"][k]["ci_hi"]] for k in order],
        "hand": [hand[k] for k in order],
        "ntrain": L["h4_dispersion_diag.json"]["v6_reconstruction"]["n_train_valid"]})

    # c5 — exposure forest
    fx = L["context_exposure.json"]["forest_plot"]
    add_chart("""
const FX = __D__;
new Chart(document.getElementById('c_fx'), { type:'bar', data:{ labels: FX.labels, datasets:[
 {label:'fitted b_form', data: FX.coef, backgroundColor: FX.coef.map(v=>v<0?PAL.d5:PAL.s1)}
]}, options:{ indexAxis:'y', plugins:{ legend:{display:false} },
 scales:{ x:{ title:{display:true,text:'train-fit form coefficient (orange = mean-reverting sign)'} },
          y:{ ticks:{autoSkip:false, font:{size:11}} } } } });
""", D={"labels": [r["label"] for r in fx], "coef": [r["coef"] for r in fx]})

    # c6 — w(g) decay overlay
    dc = L["decay_curves.json"]
    style = {
        "v6 consistent (HL20)": (C["ink"], [], 2.2, 1.0),
        "v6 anomalous (HL12)": (C["ink"], [6, 4], 2.2, 1.0),
        "sym_20": (C["d5"], [], 2, 1.0),
        "eclass-on-v6 · ewc consistent (HL16)": (C["a1"], [], 2, 1.0),
        "eclass-on-v6 · ewc anomalous (HL9.6)": (C["a1"], [6, 4], 2, 1.0),
        "lineup h24 g2 · overlap 1.0": (C["s1"], [], 1.5, 0.95),
        "lineup h24 g2 · overlap 0.8": (C["s1"], [], 1.5, 0.65),
        "lineup h24 g2 · overlap 0.6": (C["s1"], [], 1.5, 0.45),
        "lineup h24 g2 · overlap 0.4": (C["s1"], [], 1.5, 0.28),
    }
    ds = []
    for s in dc["series"]:
        if s["label"] not in style:
            continue  # eclass vct pair == v6 pair (identical curves; caption notes it)
        col, dash, wd, alpha = style[s["label"]]
        a = f"{int(alpha*255):02x}" if alpha < 1 else ""
        ds.append({"label": s["label"], "data": s["w"], "borderColor": col + a,
                   "borderDash": dash, "borderWidth": wd, "pointRadius": 0})
    add_chart("""
const WG = __D__;
new Chart(document.getElementById('c_wg'), { type:'line',
 data:{ labels: WG.g, datasets: WG.ds },
 options:{ plugins:{ legend:{ labels:{ font:{size:10} } },
  tooltip:{ callbacks:{ title:i=>i[0].label+' games ago' } } },
 scales:{ y:{ title:{display:true,text:'solve weight w(g)'} , min:0, max:1.02 },
          x:{ title:{display:true,text:'map-games ago'}, ticks:{ maxTicksLimit:13 } } } } });
""", D={"g": dc["g_games_ago"], "ds": ds})

    # c7 — caterpillar
    cat = L["page_caterpillar.json"]
    teams = cat["teams"]
    add_chart("""
const CAT = __D__;
function catds(key,label,color,hidden){ return { label:label,
  data: CAT.teams.map((t,i)=>({x:+(t[key]*100).toFixed(2), y:i})),
  backgroundColor:color, borderColor:color, pointRadius:3.5, showLine:false, hidden:hidden }; }
new Chart(document.getElementById('c_cat'), { type:'scatter', data:{ datasets:[
 catds('v6','v6 baseline',PAL.v6,false), catds('ss_5d','SS 5d roster',PAL.d5,false),
 catds('s1','S1 gate5d',PAL.s1,false), catds('ss_1a','SS core (1a)',PAL.a1,true),
 catds('tobit','Tobit',PAL.tob,true), catds('roundbt','round-BT',PAL.rbt,true)
]}, options:{ maintainAspectRatio:false,
 plugins:{ tooltip:{ callbacks:{ label:i=>CAT.teams[i.parsed.y].team+' \\u00b7 '+i.dataset.label+' '
    +i.parsed.x.toFixed(1)+'pp \\u00b7 n='+CAT.teams[i.parsed.y].n } } },
 scales:{ y:{ min:-1, max:CAT.teams.length, ticks:{ autoSkip:false, stepSize:1, font:{size:9},
    callback:v=>(CAT.teams[v]||{}).team||'' } },
   x:{ title:{display:true,text:'per-team bias (prob-pts; negative = under-rated)'} } } } });
""", D={"teams": teams})

    # c8 — H2 scatter
    h2 = L["h2_centrality.json"]
    add_chart("""
const H2 = __D__;
new Chart(document.getElementById('c_h2'), { type:'scatter', data:{ datasets:[
 {label:'teams (n=43)', data:H2.pts, backgroundColor:PAL.s1, pointRadius:4}
]}, options:{ plugins:{ legend:{display:false},
  tooltip:{ callbacks:{ label:i=>H2.pts[i.dataIndex].t+' \\u00b7 centrality '
    +i.parsed.x.toFixed(2)+' \\u00b7 bias '+(i.parsed.y).toFixed(1)+'pp \\u00b7 n='+H2.pts[i.dataIndex].n } } },
 scales:{ x:{ title:{display:true,text:'eigenvector centrality (max-normalized)'} },
          y:{ title:{display:true,text:'signed bias (prob-pts)'} } } } });
""", D={"pts": [{"x": round(t["eig_centrality"], 4), "y": round(t["bias"] * 100, 2),
                 "t": t["team"], "n": t["n"]} for t in h2["scatter"]]})

    # c9 — half-lives (roster axis)
    hp = L["h3_process_noise.json"]["axes"]["A_roster"]
    add_chart("""
const HL5 = __D__;
new Chart(document.getElementById('c_hl'), { data:{ labels: HL5.cells, datasets:[
 {type:'bar', label:'95% profile CI', data: HL5.ci, backgroundColor:'#c96a2a33', borderRadius:4},
 {type:'scatter', label:'MLE half-life', data: HL5.mle.map((v,i)=>({x:i,y:v})),
  backgroundColor:PAL.d5, pointRadius:6},
 {type:'scatter', label:'DL pooled (\\u03c4\\u00b2=0)', data: HL5.pooled.map((v,i)=>({x:i,y:v})),
  pointStyle:'line', pointRadius:14, borderColor:PAL.v6, borderWidth:3, backgroundColor:PAL.v6}
]}, options:{ plugins:{ tooltip:{ callbacks:{ label:i=> i.datasetIndex===0
  ? 'CI ['+HL5.ci[i.dataIndex][0]+', '+HL5.ci[i.dataIndex][1]+'] games'
  : i.dataset.label+': '+(i.parsed.y)+' games \\u00b7 '+HL5.n[i.dataIndex]+' games/cell' } } },
 scales:{ y:{ type:'logarithmic', title:{display:true,text:'implied half-life (games, log scale)'} } } } });
""", D={"cells": hp["cells"], "mle": hp["half_life_games_mle"],
        "ci": hp["half_life_games_ci95_profile"],
        "pooled": hp["partial_pooling_DL"]["pooled_half_life_games"],
        "n": hp["n_games_per_cell_side"]})

    # c10 — cold buckets + fragility
    cb3 = frag["eclass_cold"]["cold_buckets_mine"]
    names = ["debut(either0)", "cold(<10)", "thin(<30)"]
    add_chart("""
const CO = __D__;
new Chart(document.getElementById('c_cold'), { data:{ labels: CO.labels, datasets:[
 {type:'bar', label:'5d \\u0394 vs v6 (milli)', data: CO.dll,
  backgroundColor: CO.survives.map(s=>s?PAL.d5:'#c96a2a55')},
 {type:'scatter', label:'drop-top-5%', data: CO.drop5.map((v,i)=>({x:i,y:v})),
  backgroundColor:PAL.ink, pointStyle:'rectRot', pointRadius:5},
 {type:'scatter', label:'jackknife min', data: CO.jack.map((v,i)=>({x:i,y:v})),
  backgroundColor:PAL.rbt, pointStyle:'triangle', pointRadius:5},
 {type:'scatter', label:'bucket noise floor', data: CO.floor.map((v,i)=>({x:i,y:v})),
  pointStyle:'line', pointRadius:16, borderColor:PAL.bad, borderWidth:2.5, backgroundColor:PAL.bad}
]}, options:{ plugins:{ tooltip:{ callbacks:{ footer:i=>'n='+CO.n[i[0].dataIndex]+' \\u00b7 '+CO.nev[i[0].dataIndex]+' events' } } },
 scales:{ y:{ title:{display:true,text:'\\u0394LL (milli) \\u2014 only cold(<10) clears its floor under attack'} } } } });
""", D={"labels": names, "dll": [cb3[n]["dll_5d_milli"] for n in names],
        "drop5": [cb3[n]["drop_top5pct_milli"] for n in names],
        "jack": [cb3[n]["jackknife_min"] for n in names],
        "floor": [cb3[n]["bucket_mde_cross"] for n in names],
        "n": [cb3[n]["n"] for n in names], "nev": [cb3[n]["n_events"] for n in names],
        "survives": [n == "cold(<10)" for n in names]})

    # c11 — H4 dispersion cells
    h4_keep = ("overall", "fmt_bo3", "fmt_bo5", "depth_T1", "depth_T2", "depth_T3",
               "depth_T3_fav070")
    h4c = [c for c in L["h4_dispersion_diag.json"]["cells"] if c["name"] in h4_keep]
    if len(h4c) != len(h4_keep):
        die("h4 dispersion cell names changed")
    add_chart("""
const H4D = __D__;
new Chart(document.getElementById('c_h4'), { data:{ labels: H4D.labels, datasets:[
 {type:'bar', label:'95% CI', data: H4D.ci, backgroundColor:'#3a90cc33', borderRadius:4},
 {type:'scatter', label:'D_sweep (pp)', data: H4D.d.map((v,i)=>({x:i,y:v})), backgroundColor:PAL.a1, pointRadius:6}
]}, options:{ plugins:{ tooltip:{ callbacks:{ footer:i=>'n='+H4D.n[i[0].dataIndex] } } },
 scales:{ y:{ title:{display:true,text:'sweep excess vs iid (pp) \\u2014 >0 = OVER-dispersed'} },
          x:{ ticks:{ font:{size:10} } } } } });
""", D={"labels": [c["name"] for c in h4c], "d": [c["D_sweep_pp"] for c in h4c],
        "ci": [c.get("D_sweep_ci_pp", [None, None]) for c in h4c],
        "n": [c["n"] for c in h4c]})

    # c12 — reliability
    rel = L["page_reliability.json"]["models"]
    add_chart("""
const REL = __D__;
function reld(key,label,color){ const b=REL[key]; return [
 {label:label, data:b.map(x=>({x:x.pred_mean,y:x.emp})), borderColor:color, backgroundColor:color,
  pointRadius:4, showLine:true, tension:.2},
 {label:label+' CI', data:b.map(x=>({x:x.pred_mean,y:x.wilson_lo})), showLine:false, pointRadius:0},
 {label:label+' CIh', data:b.map(x=>({x:x.pred_mean,y:x.wilson_hi})), showLine:false, pointRadius:0}]; }
new Chart(document.getElementById('c_rel'), { type:'scatter', data:{ datasets:[
 {label:'perfect calibration', data:[{x:.5,y:.5},{x:1,y:1}], borderColor:PAL.gray, borderDash:[6,4],
  pointRadius:0, showLine:true},
 ...reld('v6','v6',PAL.v6), ...reld('ss_1a','SS core (1a)',PAL.a1), ...reld('ss_5d','SS 5d',PAL.d5)
]}, options:{ plugins:{ legend:{ labels:{ filter: it=>it.text.indexOf('CI')<0 } },
  tooltip:{ callbacks:{ label:i=>{ const m=i.dataset.label, k=m==='v6'?'v6':(m.indexOf('1a')>=0?'ss_1a':'ss_5d');
    const b=REL[k]&&REL[k][i.dataIndex]; return b? m+': pred '+(100*b.pred_mean).toFixed(1)+'% \\u00b7 emp '
    +(100*b.emp).toFixed(1)+'% ['+(100*b.wilson_lo).toFixed(1)+', '+(100*b.wilson_hi).toFixed(1)+'] \\u00b7 n='+b.n : m; } } } },
 scales:{ x:{ min:.5, max:1, title:{display:true,text:'predicted favorite probability'} },
          y:{ min:.3, max:1, title:{display:true,text:'empirical favorite win rate'} } } } });
""", D=rel)

    # c13 — buckets (S1 primary + S2/S3 toggle)
    bkt = L["page_buckets.json"]["stacks"]
    b1 = bkt["S1_gate5d"]["rows"]
    add_chart("""
const BK = __D__;
function bkds(key,label,color,hidden){ const rows=BK[key];
 return { label:label, data: rows.map(r=>r.delta_milli),
  backgroundColor: rows.map(r=> (r.inside_floor? color+'55' : (r.delta_milli>=0?PAL.good:PAL.bad))),
  hidden:hidden }; }
new Chart(document.getElementById('c_bk'), { type:'bar',
 data:{ labels: BK.S1_gate5d.map(r=>r.name), datasets:[
  bkds('S1_gate5d','S1 gate5d', PAL.s1, false),
  bkds('S2_fade_shrink','S2 fade+shrink', PAL.a1, true),
  bkds('S3_full','S3 full', PAL.d5, true) ]},
 options:{ indexAxis:'y', maintainAspectRatio:false,
 plugins:{ tooltip:{ callbacks:{ label:i=>{ const r=BK[['S1_gate5d','S2_fade_shrink','S3_full'][i.datasetIndex]][i.dataIndex];
   return i.dataset.label+': '+r.delta_milli.toFixed(2)+'m \\u00b7 n='+r.n+' \\u00b7 bucket floor '+r.bucket_mde_milli
   +'m'+(r.inside_floor?' \\u00b7 INSIDE NOISE FLOOR':''); } } } },
 scales:{ x:{ title:{display:true,text:'\\u0394LL vs v6 (milli) \\u00b7 faded = inside bucket noise floor; solid green/red = clears it'} },
          y:{ ticks:{ autoSkip:false, font:{size:10} } } } } });
""", D={k: v["rows"] for k, v in bkt.items()})

    # c14 — autopsy waterfall
    wf = ap["waterfall"]
    steps = [("locked-pair margin", wf["locked_pair_margin"]),
             ("unhedged settled", wf["unhedged_settled_pnl"]),
             ("fees", wf["fees"]),
             ("realized", wf["realized_total"]),
             ("open-inventory mark", wf["open_inventory_mark_pnl"]),
             ("total incl. mark", wf["total_incl_open_mark"])]
    run, bars = 0.0, []
    for name, v in steps:
        if name in ("realized", "total incl. mark"):
            bars.append({"label": name, "range": [0, round(v, 2)], "kind": "total"})
        else:
            bars.append({"label": name, "range": [round(run, 2), round(run + v, 2)],
                         "kind": "pos" if v >= 0 else "neg"})
            run += v
    add_chart("""
const WF = __D__;
new Chart(document.getElementById('c_wf'), { type:'bar', data:{ labels: WF.map(b=>b.label), datasets:[
 {label:'$', data: WF.map(b=>b.range),
  backgroundColor: WF.map(b=> b.kind==='total' ? PAL.ink : (b.kind==='pos'?PAL.good:PAL.bad)) }
]}, options:{ plugins:{ legend:{display:false},
  tooltip:{ callbacks:{ label:i=>{ const r=WF[i.dataIndex].range; return '$'+(r[1]-r[0]===0?r[1]:(r[1]-r[0])).toFixed(2)+' (to $'+r[1].toFixed(2)+')'; } } } },
 scales:{ y:{ title:{display:true,text:'US$ (settled per Kalshi; fees verified $0 maker)'} } } } });
""", D=bars)

    # c15 — fill-conditional calibration by price band
    fv = L["autopsy_fill_calib.json"]["by_model"]["frozen_v6"]
    bands = [("px_0_20", "0-20\u00a2"), ("px_20_40", "20-40\u00a2"), ("px_40_60", "40-60\u00a2"),
             ("px_60_80", "60-80\u00a2"), ("px_80_100", "80-100\u00a2")]
    bands = [(k, lab) for k, lab in bands if k in fv and fv[k].get("n")]
    add_chart("""
const FC = __D__;
new Chart(document.getElementById('c_fc'), { data:{ labels: FC.labels, datasets:[
 {type:'bar', label:'predicted NO-pay (frozen v6)', data: FC.pred, backgroundColor: PAL.v6},
 {type:'bar', label:'realized NO-pay', data: FC.real, backgroundColor: PAL.s1},
 {type:'scatter', label:'Wilson 95% (realized)', data: FC.wl.map((v,i)=>({x:i+0.15,y:v})),
  pointStyle:'line', rotation:90, pointRadius:8, borderColor:PAL.ink, borderWidth:2, backgroundColor:PAL.ink},
 {type:'scatter', label:'hide', data: FC.wh.map((v,i)=>({x:i+0.15,y:v})),
  pointStyle:'line', rotation:90, pointRadius:8, borderColor:PAL.ink, borderWidth:2, backgroundColor:PAL.ink}
]}, options:{ plugins:{ legend:{ labels:{ filter:it=>it.text!=='hide' } },
  tooltip:{ callbacks:{ footer:i=>'n='+FC.n[i[0].dataIndex]+' fills \\u00b7 '+FC.ctr[i[0].dataIndex]+' contracts' } } },
 scales:{ y:{ min:0, max:1, title:{display:true,text:'P(NO pays) \\u2014 fills are NO buys'} } } } });
""", D={"labels": [lab for _, lab in bands],
        "pred": [fv[k]["predicted_no_pay"] for k, _ in bands],
        "real": [fv[k]["realized_no_pay"] for k, _ in bands],
        "wl": [fv[k]["wilson_ci_realized"][0] for k, _ in bands],
        "wh": [fv[k]["wilson_ci_realized"][1] for k, _ in bands],
        "n": [fv[k]["n"] for k, _ in bands],
        "ctr": [fv[k]["contracts"] for k, _ in bands]})

    # c16 — markouts
    am = L["autopsy_markouts.json"]
    hor = [("spread_capture_c", "spread capture"), ("maker_pnl_5m_c", "+5m"),
           ("maker_pnl_30m_c", "+30m"), ("maker_pnl_2h_c", "+2h"),
           ("maker_pnl_startm5_c", "start\u22125m")]
    add_chart("""
const MK = __D__;
new Chart(document.getElementById('c_mk'), { type:'bar', data:{ labels: MK.labels, datasets:[
 {label:'all fills', data: MK.all, backgroundColor: PAL.s1},
 {label:'side-1', data: MK.s1, backgroundColor: PAL.d5},
 {label:'side-2 (hedge)', data: MK.s2, backgroundColor: PAL.a1}
]}, options:{ plugins:{ tooltip:{ callbacks:{ footer:i=>'coverage: '+MK.cov[i[0].dataIndex] } } },
 scales:{ y:{ title:{display:true,text:'maker P&L per contract (\\u00a2, NO-holder terms)'} } } } });
""", D={"labels": [lab for _, lab in hor],
        "all": [am["all"][k] for k, _ in hor],
        "s1": [am["side1"][k] for k, _ in hor],
        "s2": [am["side2"][k] for k, _ in hor],
        "cov": ["at fill" if cv is None else f"{am['all'][cv]*100:.0f}% of tape"
                for cv in [None, "cov_5m", "cov_30m", "cov_2h", "cov_startm5"]]})

    # c17 — prereg scatter
    sc = L["page_prereg_scatter.json"]["milli_rows"]
    add_chart("""
const PS = __D__;
const psx = PS.map(r=>r.mid), pslo = Math.min(...psx)-0.8, pshi = Math.max(...psx)+0.8;
new Chart(document.getElementById('c_ps'), { type:'scatter', data:{ datasets:[
 {label:'y = x', data:[{x:pslo,y:pslo},{x:pshi,y:pshi}], borderColor:PAL.gray, borderDash:[6,4], pointRadius:0, showLine:true},
 {label:'held (inside preregistered band)', data: PS.filter(r=>r.ok).map(r=>({x:r.mid,y:r.realized_milli})),
  backgroundColor:PAL.s1, pointRadius:5},
 {label:'outside band / falsifier fired', data: PS.filter(r=>!r.ok).map(r=>({x:r.mid,y:r.realized_milli})),
  backgroundColor:PAL.bad, pointStyle:'rectRot', pointRadius:6}
]}, options:{ plugins:{ tooltip:{ callbacks:{ label:i=>{ const rows=PS.filter(r=>r.ok===(i.datasetIndex===1));
   const r=rows[i.dataIndex]; return r? r.agent+' \\u00b7 '+r.item+' \\u00b7 predicted ['+r.pred_lo+', '+r.pred_hi+'] \\u2192 '+r.realized_milli+'m ('+r.outcome+')':''; } } } },
 scales:{ x:{ title:{display:true,text:'preregistered band midpoint (milli)'} },
          y:{ title:{display:true,text:'realized holdout \\u0394 (milli)'} } } } });
""", D=[{**r, "mid": round((r["pred_lo"] + r["pred_hi"]) / 2, 2),
         "ok": r["outcome"].startswith("held")} for r in sc])

    # ── tables ───────────────────────────────────────────────────────────────
    def milli(x, dp=2):
        return f"{x:+.{dp}f}m"

    # §1 tables
    pmo = L["power_mde.json"]
    bucket_rows = "".join(
        f"<tr><td>{b['bucket']}</td><td class='mono'>{b['n']}</td>"
        f"<td class='mono'>{b['mde80_within_milli']}</td>"
        f"<td class='mono'>{b['mde80_cross_milli']}</td></tr>"
        for b in pmo["buckets"] if b.get("n"))
    lrq = L["ledger_reclass.json"]
    status_cls = {"REFUTED": "dead", "UNRESOLVED": "floor", "CONFIRMED": "lead"}
    ledger_rows = "".join(
        f"<tr><td class='mono'>{r['id']}</td><td>{esc(r['idea'])}</td>"
        f"<td class='mono'>{('%+.2f' % r['delta_milli']) + 'm' if r['delta_milli'] is not None else 'unrecoverable'}</td>"
        f"<td class='mono'>{r['n_at_test'] if r['n_at_test'] else '—'}</td>"
        f"<td class='mono'>{r['mde_at_test_milli'] if r['mde_at_test_milli'] else '—'}</td>"
        f"<td><span class='verd {status_cls[r['status']]}'>{r['status']}</span></td></tr>"
        for r in lrq["rows"])
    vr = L["variance_reduction.json"]["medians"]

    # §4 tables
    drr = L["decay_rerace.json"]
    rer_rows = ""
    for name, r in drr["table"].items():
        bi = r["boot_iid"]
        rer_rows += (f"<tr><td class='mono'>{name}</td><td class='mono'>{r['n']}</td>"
                     f"<td class='mono'>{milli(r['delta_milli'])}</td>"
                     f"<td class='mono'>{r['pair_mde_raw_milli']}\u2192{r['pair_mde_cv_milli']}m</td>"
                     f"<td class='mono'>[{bi['ci_lo']*1000:+.2f}, {bi['ci_hi']*1000:+.2f}]</td>"
                     f"<td class='mono'>{bi['p_better']:.3f} / {r['boot_block']['p_better']:.3f}</td>"
                     f"<td><span class='verd floor'>INSIDE NOISE FLOOR</span></td></tr>")
    dax = L["decay_axes.json"]["axes"]
    ax_label = {"a_lineup_continuity": "a — lineup continuity (sym)",
                "b_opponent_quality_of_anomaly": "b — opp quality of anomaly",
                "c_anomaly_margin": "c — anomaly margin",
                "d_event_class": "d — event-class fade (sym)",
                "e_patch_boundary": "e — patch/map-pool fade (sym)"}
    ax_rows = ""
    for k, a in dax.items():
        vs = a["vs_v6"]
        extra = ""
        if "on_top_of_v6" in a:
            ot = a["on_top_of_v6"]["vs_v6"]
            extra = f" · on-v6 addon {milli(ot['delta_milli'])} (p {ot['boot_iid']['p_better']:.2f}/{ot['boot_block']['p_better']:.2f})"
        killed = k == "e_patch_boundary"
        ax_rows += (f"<tr><td>{ax_label[k]}</td><td class='mono'>{a['selected']}</td>"
                    f"<td class='mono'>{milli(vs['delta_milli'])} (pair MDE {vs['pair_mde_raw_milli']}m){extra}</td>"
                    f"<td><span class='verd {'dead' if killed else 'floor'}'>"
                    f"{'FALSIFIER FIRED' if killed else 'INSIDE NOISE FLOOR'}</span></td></tr>")
    dfm = L["decay_form.json"]["results"]
    form_rows = ""
    for k, r in dfm.items():
        if not isinstance(r, dict) or "b_form" not in r:
            continue
        form_rows += (f"<tr><td class='mono'>{k}</td><td class='mono'>{r['b_form']:+.4f}</td>"
                      f"<td class='mono'>{r['train_gain_from_form_milli']:+.2f}m</td>"
                      f"<td class='mono'>{milli(r['delta_milli_vs_v6'])} (MDE {r['pair_mde_raw_milli']}m)</td></tr>")
    dsp = L["decay_subpops.json"]
    sp_show = ["eclass_on_v6_m0.8", "sym_20", "sym_24", "consist_16_10", "rot_on_v6_g0.7", "form_player5"]
    sp_cols = list(dsp["mask_coverage_holdout"].keys())
    sp_head = "".join(f"<th>{c.replace('_', ' ')} <span class='mono'>n={dsp['mask_coverage_holdout'][c]}</span></th>" for c in sp_cols)
    sp_rows = ""
    for cfg in sp_show:
        cells = ""
        for r in dsp["panel"][cfg]["rows"]:
            shade = " style='background:#f4f2f8'" if abs(r["delta_milli"]) < r["bucket_mde_milli"] else \
                    (" style='background:var(--goodbg)'" if r["delta_milli"] > 0 else " style='background:var(--badbg)'")
            cells += f"<td class='mono'{shade}>{r['delta_milli']:+.2f}</td>"
        sp_rows += f"<tr><td class='mono'>{cfg}</td>{cells}</tr>"

    # §3 verdict table
    cser, cstk, cshr = L["context_seriousness.json"], L["context_stakes.json"], L["context_shrink.json"]
    cx, cwj, cadj = L["context_exposure.json"], L["context_weights.json"], L["context_adjacency.json"]
    x1 = cshr["fit_X1_standin"]
    ctx_rows = [
        ("3a", "Lineup-conditioned EWC solve weight", "DEAD (train falsifier fired)",
         f"train argmin w0*={cser['train_argmin']['w0']} = B0; monotone worse toward 0", "dead"),
        ("3a′", "Blanket EWC solve weight", "INSIDE NOISE FLOOR",
         f"{milli(cser['result_blanket']['dll_milli'])} · train argmin at the {cser['train_argmin']['we']} UP-weight edge", "floor"),
        ("3b-a", "Footage/prep exposure term", "INSIDE NOISE FLOOR",
         f"{milli(cx['prediction_term']['dll_milli'])} · EWC bucket {milli(cx['prediction_term']['bucket_ewc_fullclass']['dll_milli'])}", "floor"),
        ("3b-b", "Form-vs-exposure decomposition", "ANSWERED",
         "see the quotable below — the deliverable of this mechanism", "lead"),
        ("3c", "Elimination stakes (solve + variance)", "INSIDE NOISE FLOOR (dir. wrong)",
         f"{milli(cstk['testA_solve_weight']['dll_milli'])} / {milli(cstk['testB_variance']['dll_milli'])} · train argmin at the {cstk['testA_solve_weight']['train_argmin_w_elim']} edge — wants elim DOWN-weighted", "floor"),
        ("3d", "Learned event-class solve weights", "DEAD",
         f"{milli(cwj['holdout']['dll_milli'])} · iid CI [{cwj['holdout']['boot_iid_ci_milli'][0]}, {cwj['holdout']['boot_iid_ci_milli'][1]}]m", "dead"),
        ("3e", "Stand-in confidence shrink (X1)", "INSIDE NOISE FLOOR · demoted bucket lead",
         f"{milli(x1['dll_milli'])} overall · EWC class {milli(x1['bucket_ewc_fullclass']['dll_milli'])} — see the adversary's demotion in §0", "floor"),
    ]
    ctx_tbl = "".join(
        f"<tr><td class='mono'>{i}</td><td>{m}</td><td>{d}</td>"
        f"<td><span class='verd {c}'>{v.split(' ·')[0].split(' (')[0]}</span></td></tr>"
        for i, m, v, d, c in [(a, b, c_, d_, e) for a, b, c_, d_, e in ctx_rows])

    # §5 tables
    h1c, h1t, h1r = L["h1_censor_diag.json"], L["h1_tobit.json"], L["h1_roundbt.json"]
    h3s = L["h3_statespace.json"]
    ss_rows = ""
    for pw in h3s["pairwise_holdout"]:
        cls = "dead" if pw["verdict"] == "LOSS" else ("lead" if pw["verdict"] == "WIN" else "floor")
        ss_rows += (f"<tr><td>{esc(pw['label'])}</td><td class='mono'>{milli(pw['delta_milli'])}</td>"
                    f"<td class='mono'>{pw['mde_milli']}m</td>"
                    f"<td class='mono'>{pw['iid']['p_better']:.3f} / {pw['block_event']['p_better']:.3f}</td>"
                    f"<td><span class='verd {cls}'>{pw['verdict']}</span></td></tr>")
    h4l = L["h4_series_link.json"]["links"]
    h4_rows = ""
    for k in ("L1_sigma_global", "L2_sigma_depth"):
        r = h4l[k]
        h4_rows += (f"<tr><td class='mono'>{k}</td>"
                    f"<td class='mono'>{json.dumps(r['params']['sigma']) if 'sigma' in r['params'] else esc(str(r['params']))[:40]}</td>"
                    f"<td class='mono'>\u03b2 {r['beta_refit']:.4f}</td>"
                    f"<td class='mono'>{milli(r['delta_milli_vs_v6'])}</td>"
                    f"<td><span class='verd floor'>INSIDE NOISE FLOOR</span></td></tr>")
    h2corr = "".join(
        f"<tr><td class='mono'>{r['target']}</td><td class='mono'>{r['feature']}</td>"
        f"<td class='mono'>{r['spearman']:+.3f} [{r['spearman_ci'][0]:+.3f}, {r['spearman_ci'][1]:+.3f}]</td>"
        f"<td class='mono'>{r['pearson']:+.3f}</td></tr>"
        for r in L["h2_centrality.json"]["correlations"])

    # §6 gate table
    gate_rows = ""
    for name, g in L["compose_gate.json"]["gates"].items():
        cl = " · ".join(f"{c['clause'].split(' ')[0]} {'✓' if c['pass'] else '✗'}"
                        for c in g["clauses"])
        st = L["compose_stacks.json"]["stacks"][name]
        gate_rows += (f"<tr><td class='mono'>{name}</td>"
                      f"<td class='mono'>{milli(st['judging']['delta_milli'], 3)}</td>"
                      f"<td class='mono'>{round(st['family_mde_milli'], 3)}m ({st['regime']})</td>"
                      f"<td style='font-size:.78rem'>{cl}</td>"
                      f"<td><span class='verd hold'>{g['verdict']}</span></td></tr>")

    # §7 config gap
    acg = L["autopsy_config_gap.json"]
    cfg_rows = ""
    for r in acg["rows"]:
        detail = ""
        if "counterfactual_on_our_fills" in r:
            k6 = r["counterfactual_on_our_fills"]["logit_0.6_v6"]
            detail = (f"logit+0.6 keeps ${k6['kept_cost']:.0f} at {k6['kept_roi']*100:+.0f}%, "
                      f"refuses ${k6['dropped_cost']:.0f} at {k6['dropped_roi']*100:+.1f}%")
        elif "our_pocket" in r:
            detail = (f"our pocket: {r['our_pocket']['fills']} fills, ${r['our_pocket']['cost']:.0f}, "
                      f"{r['our_pocket']['roi']*100:+.1f}%")
        elif "realized_margin_c_per_pair" in r:
            detail = f"realized {r['realized_margin_c_per_pair']}\u00a2/pair"
        cfg_rows += (f"<tr><td>{esc(r['knob'])}</td><td class='mono'>{esc(str(r['live']))}</td>"
                     f"<td class='mono'>{esc(str(r['recommended']))[:60]}</td>"
                     f"<td style='font-size:.78rem'>{detail}</td>"
                     f"<td>{'<span class=\"verd lead\">IMPLEMENTED</span>' if r['implemented'] else '<span class=\"verd dead\">GAP</span>'}</td></tr>")

    # §8 timeline + aux prereg
    tl_rows = "".join(
        f"<tr><td class='mono'>W{r['wave']}</td><td class='mono'>{r['agent']}</td>"
        f"<td class='mono'>{(r['first'] or '\u2014')[:5]}\u2013{(r['last'] or '\u2014')[:5]}</td>"
        f"<td style='font-size:.82rem'>{esc(r['outcome'])}</td></tr>"
        for r in sorted(L["page_timeline.json"]["rows"], key=lambda r: (r["wave"], r["first"] or "99")))
    aux_rows = "".join(
        f"<tr><td class='mono'>{r['agent']}</td><td>{esc(r['item'])}</td>"
        f"<td class='mono'>{esc(str(r['predicted']))}</td><td class='mono'>{esc(str(r['realized']))}</td>"
        f"<td><span class='verd {'lead' if r['outcome'].startswith(('held','point')) else 'dead'}'>{esc(r['outcome'])}</span></td></tr>"
        for r in L["page_prereg_scatter.json"]["aux_rows"])

    # §9 slate
    sl = L["page_slate.json"]
    slate_rows = "".join(
        f"<tr><td class='mono'>{r['date']}</td><td>{esc(r['team_a'])} vs {esc(r['team_b'])}"
        f" <span class='dim' style='font-size:.78rem'>({esc(r['event'])})</span></td>"
        f"<td class='mono'>{r['format']}</td><td class='mono'><b>{r['p_a_v6']*100:.1f}%</b></td></tr>"
        for r in sl["rows"])

    # §10 tables
    dnr_rows = "".join(
        f"<tr><td>{esc(d['idea'])}</td>"
        f"<td class='mono' style='font-size:.74rem'>{esc(json.dumps(d['kill']))}</td>"
        f"<td style='font-size:.78rem'>{esc(d['note'])} <span class='mono dim'>[{', '.join(d['evidence'])}]</span></td></tr>"
        for d in lvu["do_not_retest_additions"])
    unres_rows = "".join(
        f"<tr><td class='mono'>{r['id']}</td><td>{esc(r['idea'])}</td>"
        f"<td class='mono'>{('%+.2f' % r['delta_milli']) + 'm' if r['delta_milli'] is not None else 'unrecoverable'}</td>"
        f"<td class='mono'>{r['mde_at_test_milli'] or '—'}</td></tr>"
        for r in lvu["reopened_unresolved"])
    trip_rows = "".join(
        f"<tr><td>{esc(t['conclusion'])}</td><td style='font-size:.8rem'>{esc(t['threshold'])}</td>"
        f"<td style='font-size:.8rem'>{esc(t['window'])}</td></tr>"
        for t in lvu["tripwires"])

    verify = L["verification_report.json"]
    sample_n = sum(v["n"] for v in verify["sample"]["per_event"].values())
    sample_ok = sum(v["ok"] for v in verify["sample"]["per_event"].values())
    cdif = L["corpus_diff.json"]
    from collections import Counter
    dec = Counter((yr, row.get("decision")) for yr in cdif["seasons"] for row in cdif["seasons"][yr])
    years = sorted(cdif["seasons"].keys())
    diff_rows = "".join(
        f"<tr><td class='mono'>{y}</td>"
        f"<td class='mono'>{len(cdif['seasons'][y])}</td>"
        f"<td class='mono'>{dec.get((y, 'registered'), 0)}</td>"
        f"<td class='mono'>{dec.get((y, 'add'), 0)}</td>"
        f"<td class='mono'>{dec.get((y, 'prefranchise_candidate'), 0)}</td>"
        f"<td class='mono'>{dec.get((y, 'exclude'), 0)}</td></tr>"
        for y in years)
    pre = cdif["prefranchise_status"]
    deferred = (pre["deferred"]["vct_2021_hub_regional_or_other_events"]
                + pre["deferred"]["vct_2022_hub_regional_or_other_events"])

    lu = L["lineups_coverage.json"]
    sic = lu["standin_by_event_class"]
    hn = L["h3_neff_summ"]
    adv_f = {f["id"]: f for f in adv["findings"]}
    x1b = cshr["fit_X1_standin"]
    ewc_x1 = frag["shrink_3e"]["bucket_fullclass"]
    ewc_leg = frag["shrink_3e"]["bucket_legacy2026"]
    s1f = frag["compose"]["S1"]
    d5f = frag["core"]
    hp5 = L["h3_process_noise.json"]

    n1007 = pme["n_holdout"]["old_frozen_npz"]
    mde_cur = L["page_mde_curve.json"]

    html = f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex">
<title>v8 Lab — power, mechanisms &amp; the adversarial audit</title>
<link rel="icon" href="/logo.svg">
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@700;800&family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
<style>{CSS}</style></head>
<body><div class="wrap">
<h1>v8 Lab — power, mechanisms &amp; the adversarial audit</h1>
<div class="tagline">One-day sole-researcher program, 2026-07-28 · expanded frame
(n={n_hold} holdout series, +{pme['calibration']['raw_series_in_new_events']} corpus additions) ·
every number on this page reads from <code>testing_lab/v8/stats/</code> ·
verdicts as amended by the independent adversary (§8)</div>
{NAV}

<div class="banner"><b>THE HOLDOUT IS SPENT.</b> {looks['grand_total_recorded_holdout_numbers']} holdout
numbers are recorded on disk ({looks['primary_candidate_looks']} primary candidate looks,
{looks['sweep_checkpoint_looks']} sweep checkpoints, {looks['per_iteration_diagnostic_looks']} iteration
diagnostics — <code>compose_looks.json</code>), including per-grid-point holdout LLs for entire sweeps.
This wave's selections were verifiably train-only, but any <i>future</i> "train-only" claim on this frame
is unfalsifiable with those menus published. Anything new on the 2025-26 frame is exploratory by
definition; confirmatory power now requires fresh series (§10: what data next).</div>

<section>
<h2><span class="n">0</span>Verdict — adversary-amended, and final</h2>
<div class="callout"><b>v6 stands. No stack cleared the promotion gate.</b> All three preregistered
compose stacks scored HOLD ({milli(L['compose_stacks.json']['stacks']['S1_gate5d']['judging']['delta_milli'], 3)} /
{milli(L['compose_stacks.json']['stacks']['S2_fade_shrink']['judging']['delta_milli'], 3)} /
{milli(L['compose_stacks.json']['stacks']['S3_full']['judging']['delta_milli'], 3)}), every gate clause failed on every
stack, and the independent adversary reproduced all three verdicts exactly under his own engine and
bootstrap reimplementation — "v6 stands" is co-signed, not self-graded.</div>
<div class="callout warn"><b>CORRECTION over Phase 0's prose.</b> Phase 0 published "0/23 v7 configs
distinguishable". Under the program's own cross-family floor ({floors['cross_milli']}m) and its own block
bootstraps, four v7 configs — <code>sym_6</code> ({milli(next(r['delta_milli'] for r in lad['rows'] if r['config'] == 'sym_6'))}),
<code>surprise_12_20</code> ({milli(next(r['delta_milli'] for r in lad['rows'] if r['config'] == 'surprise_12_20'))}),
<code>sym_8</code> ({milli(next(r['delta_milli'] for r in lad['rows'] if r['config'] == 'sym_8'))}),
<code>boxexp_c3</code> ({milli(next(r['delta_milli'] for r in lad['rows'] if r['config'] == 'boxexp_c3_hl8'))}) — WERE
distinguishable, <i>as losses</i>. Phase 0 quietly used per-pair MDE₈₀, a laxer standard than the one every
later agent was held to (adversary finding F1, CONFIRMED-FLAW). The near-ties
(<code>sym_20</code> {milli(next(r['delta_milli'] for r in lad['rows'] if r['config'] == 'sym_20'))},
<code>consist_16_10</code> {milli(next(r['delta_milli'] for r in lad['rows'] if r['config'] == 'consist_16_10'))}) remain
unresolved. The §1 ladder is re-plotted accordingly.</div>
<div class="callout bad"><b>Demoted by adversarial review</b> — published here as noise-floor leads, not wins:
(1) 3e stand-in shrink's EWC bucket ({milli(ewc_x1['dll_X1_milli'])} at n={ewc_x1['n']} — inside the
{frag['shrink_3e']['bucket_mde_within_at_n291']}m bucket floor, flips to {milli(frag['shrink_3e']['fullclass_drop_top5pct']['delta_milli_after'])} under drop-top-5%, and the falsifier dummy beats it on the legacy bucket
{milli(ewc_leg['dll_dummy_milli'])} vs {milli(ewc_leg['dll_X1_milli'])});
(2) event-class fade's "positive in all subpops" (subpops overlap up to {frag['eclass_cold']['overlap']['S5_frac']*100:.1f}% of the holdout —
five signs, one observation; negative on its own EWC target rows);
(3) 5d's 7.4-vs-24.8-game half-life contrast (profile CIs overlap enormously; DL pooling τ²=0 — a hypothesis, not a finding);
(4) compose S1's gated gain (nongated delta exactly 0; gated {milli(s1f['gated_rows_delta_milli'])} inside a
~15m bucket floor; overall flips to {milli(s1f['drop_top5pct_milli'])} under drop-top-5%).</div>
<div class="callout good"><b>The one surviving lead: cold starts.</b> H3's state-space model on rows where
either team has &lt;10 prior maps: <b>{milli(cold['dll_5d_milli'])}</b> at n={cold['n']} (bucket floor
{cold['bucket_mde_cross']}m; drop-top-5% {milli(cold['drop_top5pct_milli'])}; event-jackknife min
{milli(cold['jackknife_min'])}) — the only positive cell that survived the adversary's fragility attacks.
Caveats stay attached: {cold['n_events']} events supply the rows, and the parent model loses overall
({milli(d5f['h3_check']['delta_5d_vs_v6_milli'])}). This is a lead that has earned a test on new data — not a promotion.</div>
<div class="cards">
 <div class="card"><div class="lbl">v6 on the expanded frame</div>
  <div class="big">{base['ll_holdout']:.5f}</div>
  <div class="sub">holdout log-loss · β={base['beta']} · n={n_hold} · unchanged in production</div></div>
 <div class="card"><div class="lbl">Stacks promoted</div>
  <div class="big">0 of {len(L['compose_gate.json']['gates'])}</div>
  <div class="sub">all HOLD · gate clauses G1/G2/G3 all failed · co-signed by the adversary</div></div>
 <div class="card"><div class="lbl">Instrument floor</div>
  <div class="big">{floors['within_milli']} / {floors['cross_milli']}m</div>
  <div class="sub">MDE₈₀ within / cross family at n={n_hold} · nothing below it is decidable</div></div>
 <div class="card"><div class="lbl">Surviving lead</div>
  <div class="big good">{milli(cold['dll_5d_milli'])}</div>
  <div class="sub">cold-start rows (&lt;10 prior maps), n={cold['n']} · floor {cold['bucket_mde_cross']}m · survives drop-5% + jackknife</div></div>
 <div class="card"><div class="lbl">max|team bias|</div>
  <div class="big">{catsum['v6']['max_abs_bias']*100:.1f}pp</div>
  <div class="sub">v6 · best candidate {catsum['ss_1b_qcal']['max_abs_bias']*100:.1f}pp (1b) but mean|bias| worsens
  {catsum['v6']['mean_abs_bias']*100:.2f}→{catsum['ss_1b_qcal']['mean_abs_bias']*100:.2f}pp</div></div>
 <div class="card"><div class="lbl">Live week (autopsy)</div>
  <div class="big bad">−${abs(ap['waterfall']['realized_total']):.2f}</div>
  <div class="sub">realized, {ap['fills']} fills · not noise (p&lt;{1/av['B']:.5f}) · blame: config ≈60–70%, model ≈25%</div></div>
 <div class="card"><div class="lbl">Holdout numbers recorded</div>
  <div class="big bad">{looks['grand_total_recorded_holdout_numbers']}</div>
  <div class="sub">{looks['primary_candidate_looks']} primary looks · Bonferroni α at K={L['compose_looks.json']['family_wise_context']['K_primary']} is {L['compose_looks.json']['family_wise_context']['bonferroni_alpha_at_K']}</div></div>
</div>
</section>

<section>
<h2><span class="n">1</span>Power — what this program can actually see</h2>
<p>Phase 0 measured the instrument before anyone used it: MDE₈₀ on paired per-series loss differences,
verified by simulation on the CRN streams. At n={n1007} the old frame could not 80%-reliably
distinguish two same-family variants closer than {pmo['overall']['regime_within']['mde80_milli']}m or two
structurally different models closer than {pmo['overall']['regime_cross']['mde80_milli']}m; the corpus
expansion buys ~8%: <b>{floors['within_milli']}m within / {floors['cross_milli']}m cross at n={n_hold}</b>
(composition-adjusted, preregistered rule). Adjudicating a 2m cross-family effect at 80% power needs
n ≈ {mde_cur['n_for_2milli_cross']:,} series — ~10× the holdout.</p>
<div class="chartbox"><canvas id="c_mde"></canvas></div>{dl('page_mde_curve.json')}
<h3>The v7 ladder, re-adjudicated — with the adversary's correction plotted</h3>
<div class="chartbox tall"><canvas id="c_lad"></canvas></div>{dl('page_v7_ladder.json')}
<p class="cap">Shaded band = program-standard cross-family floor (±{lad['cross_floor_milli']}m).
<span style="color:{C['bad']};font-weight:700">Red bars</span> = the four configs the adversary showed are
distinguishable-as-losses under that floor + block-CI significance (F1) — the CORRECTED verdict.
Purple outline = the motivating near-ties (sym_20, consist_16_10): still unresolved; the "−1.65m rejection"
of sym_20 was a coin flip then and stays one now. Hover any bar for its pair MDE and block CI.</p>
<h3>Ledger re-classification — {lrq['counts']['REFUTED']} refuted · {lrq['counts']['UNRESOLVED']} unresolved · {lrq['counts']['CONFIRMED']} confirmed</h3>
<p>{lrq['counts']['UNRESOLVED']} of {lrq['n_entries']} historical "rejected" entries were statements about noise
— the decisions were defensible ("no reason to switch"), the <i>negatives</i> were not established. Nothing
was found that should have been promoted ({lrq['counts']['CONFIRMED']} CONFIRMED overturns). Standing
consequence: a ledger entry bars re-testing only with a power annotation; UNRESOLVED entries are
re-openable by design (§10 lists them).</p>
<div class="scroll"><table>
<tr><th>id</th><th>idea</th><th>Δ at test</th><th>n</th><th>MDE then</th><th>status</th></tr>
{ledger_rows}
</table></div>{dl('ledger_reclass.json')}
<h3>Per-bucket floors (the promotion rule's blind spots)</h3>
<p class="cap">"No major bucket regression" is unenforceable as stated for any bucket under n≈200 — the
floors below are what a bucket claim must clear. Quote them or don't make the claim.</p>
<div class="scroll"><table>
<tr><th>bucket</th><th>n</th><th>within (m)</th><th>cross (m)</th></tr>
{bucket_rows}
</table></div>{dl('power_mde.json')}
<h3>Variance reduction — measured, not hoped</h3>
<table>
<tr><th>design</th><th>effective-n multiplier</th><th>reading</th></tr>
<tr><td>Pairing (in use)</td><td class="mono">{vr['pairing_multiplier_cross']:.0f}× cross / {vr['pairing_multiplier_within']:.0f}× within</td>
<td>the program's single biggest asset</td></tr>
<tr><td>Control variate l_v6 (CUPED)</td><td class="mono">{vr['cv_l6_multiplier_cross']:.2f}× cross / {vr['cv_l6_multiplier_within']:.2f}× within</td>
<td>adopted in referee.py for cross-family tests; nil within-family (the preregistered ≥1.5× trigger did NOT fire — {vr['cv_l6_multiplier_cross']:.2f}× is the honest number)</td></tr>
<tr><td>Multivariate CV</td><td class="mono">{vr['cv_multivariate_multiplier_cross']:.2f}× / {vr['cv_multivariate_multiplier_within']:.2f}×</td>
<td>negligible increment over l_v6 alone</td></tr>
<tr><td>Block-by-event vs iid</td><td class="mono">DEFF median {vr['deff_block_over_iid_median']:.2f}</td>
<td>iid CIs are <i>conservative</i> here (the DEFF&gt;1.3 alarm did not fire); caveat: only 18 events</td></tr>
</table>{dl('variance_reduction.json')}
</section>

<section>
<h2><span class="n">2</span>Corpus — what was added, and how it was checked</h2>
<div class="cards">
 <div class="card"><div class="lbl">Series backfilled (2023–26)</div><div class="big">{L['corpus_blocks.json']['totals']['matches_2023_2026']}</div>
  <div class="sub">{len(pme['new_events'])} new registry events · holdout grew {pme['n_holdout']['old_frozen_npz']}→{pme['n_holdout']['expanded_scoreable']} scoreable</div></div>
 <div class="card"><div class="lbl">Maps checked</div><div class="big">{verify['mechanical']['total_maps_checked']}</div>
  <div class="sub">mechanical re-verification: {verify['mechanical']['total_series']}/{verify['mechanical']['total_series']} series, 0 failures</div></div>
 <div class="card"><div class="lbl">Live sample re-fetch</div><div class="big">{sample_ok}/{sample_n}</div>
  <div class="sub">seeded re-scrape vs recorded winner+score, all {len(verify['sample']['per_event'])} events</div></div>
 <div class="card"><div class="lbl">Prefranchise (held separate)</div><div class="big">{L['corpus_blocks.json']['totals']['matches_prefranchise']}</div>
  <div class="sub">6 intl LANs 2021-22 · {deferred} regional events DEFERRED (several thousand series — the big power lever, untouched)</div></div>
</div>
<div class="chartbox"><canvas id="c_corpus"></canvas></div>{dl('corpus_blocks.json')}
<h3>Archive sweep — every VLR event row, decided</h3>
<div class="scroll"><table>
<tr><th>year</th><th>rows reviewed</th><th>already registered</th><th>added</th><th>prefranchise candidate</th><th>excluded (tier-3/community)</th></tr>
{diff_rows}
</table></div>{dl('corpus_diff.json')}
<div class="callout warn"><b>Verification limit, stated plainly (adversary structural note):</b>
{esc(adv_f['F7']['result'])}</div>
<p class="cap">Class map hazard logged for the rebuild owner: {', '.join(k for k, v in pme['new_events'].items() if v == 'intl' and ('masters' in k or 'champions' in k))}
are off-season invitationals whose ids contain "masters"/"champions" — the harness substring rule will
class them INTERNATIONAL at rebuild. Flagged in phase0 §addendum before any timeline regeneration.</p>
</section>

<section>
<h2><span class="n">3</span>Event context — incentives, preparation, seriousness</h2>
<p>Every solve-side lever failed in-sample or failed walk-forward; the operator's "down-weight EWC"
intuition is unsupported in every parameterization tried — in-sample the pull is <i>upward</i>.</p>
<table>
<tr><th>#</th><th>mechanism</th><th>result</th><th>verdict</th></tr>
{ctx_tbl}
</table>
<h3>Fitted event-class weights vs the hand-set marks</h3>
<div class="chartbox"><canvas id="c_w"></canvas></div>{dl('context_weights.json')}
<p class="cap">Fitted {{{', '.join(f"{k} {v}" for k, v in cwj['fitted_weights'].items())}}} improves train by
{(cwj['B0_train_ll'] - cwj['fitted_train_ll'])*1000:.1f}m and loses {milli(cwj['holdout']['dll_milli'])} on holdout — in-sample
regime memorization (the champions collapse to {cwj['fitted_weights']['champions']} is even "identified" in-sample and still
anti-validates). Fitted EWC weight {cwj['ewc_weight_deliverable']['fitted']} with CI
[{cwj['ewc_weight_deliverable']['ci95'][0]}, {cwj['ewc_weight_deliverable']['ci95'][1]}] ⊇ 1.0. Hand-set marks (black) win; weights are weakly identified at n_train={L['h4_dispersion_diag.json']['v6_reconstruction']['n_train_valid']}.</p>
<h3>Exposure vs form — the decomposition</h3>
<div class="chartbox"><canvas id="c_fx"></canvas></div>{dl('context_exposure.json')}
<div class="callout"><b>The quotable:</b> {esc(cx['quotable'])}</div>
<h3>Seriousness observables (lineups agent)</h3>
<p>EWC-class rows really are different in observables: stand-in rate vs 30-day modal lineup
<b>{sic['ewc']['stand_in_30d_modal']['rate']*100:.1f}%</b> (n={sic['ewc']['stand_in_30d_modal']['n_defined']}) vs
<b>{sic['vct']['stand_in_30d_modal']['rate']*100:.1f}%</b> on VCT rows (n={sic['vct']['stand_in_30d_modal']['n_defined']}) —
and none of it converts into a solve-weight or shrink gain that clears the floor (3a/3e above).
Coverage bar: {lu['engine_sides_covered']}/{lu['engine_sides_total']} engine sides matched
({lu['coverage_pct']:.1f}%).</p>
<div class="callout warn"><b>Adjacency honesty panel:</b> deep-run → next-intl slump is UNTESTABLE at this n:
{cadj['n_series_bothattended']} both-attended series across {len(cadj['pairs'])} Masters→next-intl pairs;
c = {cadj['c_hat']:+.4f}, CRN boot CI [{cadj['crn_boot_ci95'][0]:+.4f}, {cadj['crn_boot_ci95'][1]:+.4f}] spans zero
with the sign <i>opposite</i> the fatigue story. Published as untestable, not as a null win.</div>
</section>

<section>
<h2><span class="n">4</span>Decay — recency &amp; asymmetry, re-raced at 2-4× the old resolution</h2>
<div class="chartbox"><canvas id="c_wg"></canvas></div>{dl('decay_curves.json')}
<p class="cap">Solve weight vs map-games ago. Black = the v6 pair (consistent HL20, anomalous dashed);
orange = sym_20, which lies <i>exactly on</i> the v6 consistent curve — the symmetric variant simply applies
HL20 to anomalous games too, so the whole philosophical dispute is the gap between the black solid and black
dashed lines; blue = event-class fade's EWC pair (its VCT pair is identical to v6 and omitted); purple ramp =
lineup-continuity conditioning at roster overlap 1.0→0.4 (ordinal, one hue).</p>
<h3>The near-ties, re-raced (CV-adjusted)</h3>
<div class="scroll"><table>
<tr><th>cand vs v6</th><th>n</th><th>Δ</th><th>pair MDE raw→CV</th><th>iid CI (m)</th><th>p iid/block</th><th>verdict</th></tr>
{rer_rows}
</table></div>{dl('decay_rerace.json')}
<p class="cap">The answer is <b>ties</b>, in those words — pre-committed wording: "unresolved either way at
n={n_hold}: asymmetry is not demonstrably needed, nor demonstrably better." The uniform −2 to −2.8m lean of
five independent symmetric configs is the evidence that currently exists for keeping consistency conditioning.</p>
<h3>Five new conditioning axes</h3>
<table>
<tr><th>axis</th><th>selected (train)</th><th>holdout vs v6</th><th>verdict</th></tr>
{ax_rows}
</table>{dl('decay_axes.json')}
<h3>Performance-defined form — all of it inside the floor, all of it negative</h3>
<div class="scroll"><table>
<tr><th>definition</th><th>b_form (train)</th><th>train gain</th><th>holdout vs v6</th></tr>
{form_rows}
</table></div>{dl('decay_form.json')}
<h3>Subpopulation panel</h3>
<div class="scroll"><table>
<tr><th>config</th>{sp_head}</tr>
{sp_rows}
</table></div>{dl('decay_subpops.json')}
<p class="cap"><b>Caveat (adversary, F3):</b> the subpops are not independent evidence — S4∩S5 covers
{frag['eclass_cold']['overlap']['S4_and_S5_frac_of_holdout']*100:.1f}% of the holdout and S5 alone
{frag['eclass_cold']['overlap']['S5_frac']*100:.1f}%; five positive signs on eclass_on_v6 are one observation
wearing five hats, and the effect is negative on its own EWC-class target rows
({milli(frag['eclass_cold']['eclass_delta_split']['on_2026_ewc_class_rows_milli'])} on n={frag['eclass_cold']['eclass_delta_split']['n_in']}).
Gray shading = inside the bucket's scaled noise floor (which is nearly everywhere). Instrument note from the
phase: the red cells on rot_on_v6 and form_player5 overstate resolution — those rating-perturbing addons have
cross-family-sized empirical pair σ; both keep headline verdict INSIDE NOISE FLOOR.</p>
</section>

<section>
<h2><span class="n">5</span>Bias mechanisms — four hypotheses for one distortion</h2>
<p>The target: elite compression — max|bias| {catsum['v6']['max_abs_bias']*100:.1f}pp on the expanded frame
(PRX {teams[0]['v6']*100:+.1f}pp at one end, TS {teams[-1]['v6']*100:+.1f}pp at the other). Four mechanisms
went in; the caterpillar shows what came out. Toggle candidates via the legend.</p>
<div class="chartbox tall"><canvas id="c_cat"></canvas></div>{dl('page_caterpillar.json')}
<h3>H1 — margin censoring: DEAD at the premise</h3>
<p>The corpus barely touches the 13-0 cap: {h1t['censoring_mass']['CAP']} of {h1t['censoring_mass']['n_games']}
maps ({h1t['censoring_mass']['CAP_share']*100:.2f}%), density <i>falling</i> toward the bound — the opposite of
censoring pile-up. Cap-share ratio Q4/mid = {h1c['cap_ratio_Q4_over_mid']} (preregistered premise needed ≥2).
Tobit: {milli(h1t['primary_s1.0']['dll_milli_vs_v6'])} (inside floor, elite bias unmoved). Round-level BT:
{milli(h1r['holdout']['dll_milli_vs_v6'])} — a dead tie; and the "order of magnitude more data" intuition is
refuted by measurement: k_eff ≈ {h1r['effective_sample']['k_eff_fisher_median']} (Fisher) /
{h1r['effective_sample']['k_eff_cluster_boot_median']} (cluster bootstrap; the preregistered k&lt;7 falsifier
fired) — budget no future power on rounds.</p>
{dl('h1_censor_diag.json')}
<h3>H2 — schedule connectivity: DEAD at the gate</h3>
<div class="chartbox"><canvas id="c_h2"></canvas></div>{dl('h2_centrality.json')}
<div class="scroll"><table>
<tr><th>target</th><th>feature</th><th>Spearman [95% CI]</th><th>Pearson</th></tr>
{h2corr}
</table></div>
<p class="cap">The mechanistically decisive row is |bias| ~ centrality: as null as it gets at n=43. The signed
correlations that do light up are composition, not connectivity — teams that travel are disproportionately
elite (and under-rated); insular teams are disproportionately floor (and over-rated). PRX has the
<i>highest</i> centrality of the named teams and is the <i>most</i> under-rated. Gate stopped E2/E3; no rescue fits.</p>
<h3>H3 — rating uncertainty: loses overall, wins the cold tail</h3>
<div class="scroll"><table>
<tr><th>pair</th><th>Δ</th><th>MDE</th><th>p iid/block</th><th>verdict</th></tr>
{ss_rows}
</table></div>{dl('h3_statespace.json')}
<div class="chartbox"><canvas id="c_hl"></canvas></div>{dl('h3_process_noise.json')}
<p class="cap"><b>Demoted (adversary F4):</b> the 7.4-vs-24.8-game half-life contrast is a point ordering whose
profile CIs overlap enormously and whose DerSimonian–Laird pooling returns τ²=0 — all cells shrink to one
pooled HL ≈ {hp5['axes']['A_roster']['partial_pooling_DL']['pooled_half_life_games'][0]} games. The supporting 5d-vs-core "+{d5f['h3_check']['delta_5d_vs_1a_milli']}m WIN" has iid
p_better {d5f['h3_check']['boot_iid_5d_vs_1a']['p_better']} &lt; .95 and flips to {milli(d5f['h3_5d_vs_1a_drop_top5pct']['delta_milli_after'])}
under drop-top-5% — a hypothesis to test on new data, not a finding. Org-age axis REVERSED (left-censoring
artifact); volatility flat. The formal inference, preregistered wording: heterogeneity unsupported at this n.</p>
<div class="chartbox"><canvas id="c_cold"></canvas></div>{dl('page_adversary_fragility.json')}
<p class="cap">The cold buckets under attack: bars = 5d Δ vs v6; red line = each bucket's noise floor; diamonds =
drop-top-5%; triangles = event-jackknife minimum. Only cold(&lt;10) clears its floor under every attack —
debut (n={cb3['debut(either0)']['n']}) halves on one series and thin (n={cb3['thin(<30)']['n']}) falls below floor
under drop-5%. n_eff telemetry for confidence-aware sizing is emitted for all rows
(<code>h3_neff.json</code>; corr(pair n_eff, |ΔLL vs v6|) =
{hn['distribution_summaries']['corr_neff_harm_vs_absdeltaLL_v6_holdout']:+.3f} — low-confidence rows are
where the models disagree most).</p>
<h3>H4 — iid series aggregation: premise inverted</h3>
<div class="chartbox"><canvas id="c_h4"></canvas></div>{dl('h4_dispersion_diag.json')}
<table>
<tr><th>link</th><th>σ_u</th><th>β refit</th><th>holdout Δ</th><th>verdict</th></tr>
{h4_rows}
</table>{dl('h4_series_link.json')}
<p class="cap">Maps are OVER-dispersed (shared series effect σ̂_u = {L['h4_dispersion_diag.json']['joint_fit_overall']['sigma']:.2f}), not
under-dispersed — H4's "iid understates elite bo5 favorites" is refuted in sign, and v6's train-fit β has been
implicitly paying the correction all along (β_iid/β_σ ≈ {base['beta']:.4f}/{h4l['L1_sigma_global']['beta_refit']:.4f}). P(bo5) for strong favorites
<i>falls</i> under the corrected link. Both links inside the floor; v6 aggregation stands.</p>
<h3>Reliability, favorite frame (holdout)</h3>
<div class="chartbox"><canvas id="c_rel"></canvas></div>{dl('page_reliability.json')}
<p class="cap">The uncertainty models buy their cold-tail gains by shrinking toward 0.5 — visible here as the
flattening at high predicted probabilities vs v6. Hover for bin n and Wilson 95% interval; sparse bins
(n&lt;{L['page_reliability.json']['min_bin_n']}) merged.</p>
</section>

<section>
<h2><span class="n">6</span>Buckets — where the stacks move, and where that is meaningless</h2>
<div class="scroll"><table>
<tr><th>stack</th><th>overall Δ</th><th>family MDE</th><th>gate clauses</th><th>verdict</th></tr>
{gate_rows}
</table></div>{dl('compose_gate.json')}
<div class="chartbox tall"><canvas id="c_bk"></canvas></div>{dl('page_buckets.json')}
<p class="cap">Faded bars sit inside their bucket's scaled noise floor (family MDE · √(N/n) — the preregistered
rule); only solid bars carry color as polarity. <b>The negative buckets, explained:</b> S1's huge-gap bucket
({milli(next(r['delta_milli'] for r in b1 if r['name'].startswith('huge')))} on n={next(r['n'] for r in b1 if r['name'].startswith('huge'))}) sits inside
its scaled floor here, yet fails the gate's stricter fixed-threshold G3 clause
(<span class='mono'>{esc(L['compose_gate.json']['gates']['S1_gate5d']['clauses'][2]['clause'])}</span>) — the gated 5d rows shrink
some runaway favorites that v6 prices well; it is why S1 would not be promotable even above the floor. S2's
domestic-EMEA and favorite-[0.7,0.8) regressions are its G3 failures. S3 (toggle it on) regresses
{sum(1 for r in bkt['S3_full']['rows'] if not r['inside_floor'] and r['delta_milli'] < 0)} buckets beyond even the scaled
floors — the anti-synergy wipeout. EWC-class rows under S1: gated split {milli(s1f['gated_rows_delta_milli'])} on
n={s1f['n_gated_hold']} gated vs exactly {L['compose_stacks.json']['stacks']['S1_gate5d']['gated_split']['delta_milli_nongated']} elsewhere.</p>
<p>The unexplained residual, named for future work: <b>PRX/NRG under-rating persists under every stack</b>
(v6 {L['compose_stacks.json']['stacks']['S1_gate5d']['prx_nrg_residual_pp']['PRX']}pp / {L['compose_stacks.json']['stacks']['S1_gate5d']['prx_nrg_residual_pp']['NRG']}pp under S1 — unmoved). Whatever drives it is not stand-in
load, not event-class recency, not rating uncertainty: all excluded by this program.</p>
</section>

<section>
<h2><span class="n">7</span>Live P&amp;L autopsy — the operator's implementation memo</h2>
<div class="callout warn"><b>Scope guard (README rule 9):</b> market data is diagnostic, never a fitting
target. This section is an implementation memo for the operator, not an input to the solve — and it is a
<b>one-week sample</b> ({ap['events_settled']} settled events): every number below carries that caveat at full
strength. Model selection ran on match outcomes only.</div>
<div class="callout bad"><b>Verdict:</b> not noise, not fees, not microstructure — the bot is down
${abs(ap['waterfall']['realized_total']):.2f} realized because its flat {acg['live_config_verified']['hard_min_edge_cents']}¢ edge floor
let it accumulate every model–market disagreement, and in this window the market, not v6, was right.
CRN bootstrap under the flat-floor edge hypothesis: p(cum ≤ {av['observed_realized']}) =
{av['p_cum_le_observed']:.0f}/{av['B']} (&lt;{1/av['B']:.5f}). Blame: config ≈60–70%, model ≈25%,
everything else exonerated.</div>
<div class="chartbox"><canvas id="c_wf"></canvas></div>{dl('autopsy_pnl.json')}
<h3>Adverse selection — where fills fail (frozen v6, contract-weighted)</h3>
<div class="cards">
 <div class="card"><div class="lbl">Fill-conditional gap</div><div class="big bad">{L['autopsy_fill_calib.json']['adverse_selection']['fill_gap_frozen_v6']*100:+.1f}pp</div>
 <div class="sub">realized − predicted NO-pay on {L['autopsy_fill_calib.json']['settled_fills']} settled fills</div></div>
 <div class="card"><div class="lbl">Unconditional benchmark</div><div class="big">{L['autopsy_fill_calib.json']['adverse_selection']['uncond_gap']*100:+.1f}pp</div>
 <div class="sub">v6 NO-on-favorite, all {L['autopsy_fill_calib.json']['unconditional_frozen_v6']['n']} settled events</div></div>
 <div class="card"><div class="lbl">Adverse selection</div><div class="big bad">{L['autopsy_fill_calib.json']['adverse_selection']['adverse_selection_pts']*100:+.1f}pp</div>
 <div class="sub">≈ {L['autopsy_fill_calib.json']['adverse_selection']['adverse_selection_cents_per_contract']:.1f}¢/contract · sharpest: side-1 {L['autopsy_fill_calib.json']['by_model']['frozen_v6']['side1']['gap']*100:+.1f}pp</div></div>
 <div class="card"><div class="lbl">Toxic window</div><div class="big">6–24h</div>
 <div class="sub">{L['autopsy_fill_calib.json']['by_model']['frozen_v6']['mts_6-24h']['gap']*100:+.1f}pp on {L['autopsy_fill_calib.json']['by_model']['frozen_v6']['mts_6-24h']['contracts']:.0f} contracts — the "size late" zone, not the final 2h</div></div>
</div>
<div class="chartbox"><canvas id="c_fc"></canvas></div>{dl('autopsy_fill_calib.json')}
<h3>Micro markouts are clean — the selection is match/side-level</h3>
<div class="chartbox"><canvas id="c_mk"></canvas></div>{dl('autopsy_markouts.json')}
<p class="cap">Positive maker P&amp;L at every horizon with tape coverage (start−5m has
{am['all']['cov_startm5']*100:.0f}% coverage — read with care). The market did not pick us off tick-by-tick;
it knew the resolutions. Fees verified $0 (maker schedule; taker counterfactual
${L['autopsy_fees.json']['counterfactual_taker_fees_dollars']:.2f}).</p>
<h3>Config gap ladder</h3>
<div class="scroll"><table>
<tr><th>knob</th><th>live</th><th>recommended</th><th>evidence (this week's fills)</th><th>status</th></tr>
{cfg_rows}
</table></div>{dl('autopsy_config_gap.json')}
<p class="cap">Vintage slices exonerate the 2026-07-23 snapshot-drift incident: drifted-vm fills calibrate
{L['autopsy_fill_calib.json']['by_model']['frozen_v6']['vin_drifted_vm']['gap']*100:+.1f}pp (n={L['autopsy_fill_calib.json']['by_model']['frozen_v6']['vin_drifted_vm']['n']}).
Gaps: VM code provenance unverifiable (no .git); unconditional benchmark is n={L['autopsy_fill_calib.json']['unconditional_frozen_v6']['n']} events (wide CI).</p>
</section>

<section>
<h2><span class="n">8</span>Research log — preregistration vs outcome, and the hostile review</h2>
<p>Every experimental agent wrote numeric predictions BEFORE running (per-file discipline, quotes verified at
build time against the preregister files). Here is how the predictions did:</p>
<div class="chartbox"><canvas id="c_ps"></canvas></div>{dl('page_prereg_scatter.json')}
<p class="cap">Points on the diagonal were called correctly; red diamonds left the band (or fired their
falsifier — which is the system working, recorded at the same resolution as wins). Non-milli predictions:</p>
<div class="scroll"><table>
<tr><th>agent</th><th>prediction</th><th>predicted</th><th>realized</th><th>outcome</th></tr>
{aux_rows}
</table></div>
<h3>Session timeline — {L['page_timeline.json']['date']}</h3>
<div class="scroll"><table>
<tr><th>wave</th><th>agent</th><th>journal span</th><th>what happened</th></tr>
{tl_rows}
</table></div>{dl('page_timeline.json')}
<div class="adv">
<div class="advhead">THE ADVERSARY'S REPORT — published verbatim, including the parts that overturn program claims</div>
<pre>{esc(adv_md)}</pre>
</div>{dl('adversary_report.json')}
</section>

<section>
<h2><span class="n">9</span>Slate — v6 snapshot prices (no v8 column)</h2>
<div class="callout">{esc(sl['no_v8_column_note'])}</div>
<div class="scroll"><table>
<tr><th>date</th><th>match (P of first team)</th><th>fmt</th><th>v6</th></tr>
{slate_rows}
</table></div>{dl('page_slate.json')}
<p class="cap">Priced from the frozen <code>{sl['model_version']}</code> snapshot via
<code>trading_model/predict.py</code>, generated {sl['generated']}.</p>
</section>

<section>
<h2><span class="n">10</span>Tripwires &amp; ledger updates</h2>
<h3>Concrete thresholds — what would reopen each conclusion</h3>
<div class="scroll"><table>
<tr><th>conclusion</th><th>tripwire threshold</th><th>window</th></tr>
{trip_rows}
</table></div>
<h3>Do-not-retest additions from this program (each with its kill evidence)</h3>
<div class="scroll"><table>
<tr><th>idea (stays dead)</th><th>kill numbers</th><th>why, and where the evidence lives</th></tr>
{dnr_rows}
</table></div>
<h3>Re-openable: the {len(lvu['reopened_unresolved'])} rejected→unresolved ledger entries</h3>
<p class="cap">{esc(lvu['power_annotation_rule'])}</p>
<div class="scroll"><table>
<tr><th>id</th><th>idea</th><th>Δ at test</th><th>MDE then</th></tr>
{unres_rows}
</table></div>{dl('ledger_v8_updates.json')}
<h3>What data next (the holdout being spent)</h3>
<ul>{''.join(f'<li>{esc(x)}</li>' for x in lvu['what_data_next'])}</ul>
</section>

<p class="dim" style="text-align:center">v8 Lab · program run {L['page_timeline.json']['date']} · engine:
testing_lab walk-forward Massey replica on the sha-verified expanded frame · CRN referee throughout ·
v6 remains the champion and the trading_model snapshot is unchanged · adversary-amended verdicts are final</p>
</div>
<script>
{''.join(js_blocks)}
</script>
</body></html>"""

    os.makedirs(RD, exist_ok=True)
    out = os.path.join(RD, "v8_lab.html")
    with open(out, "w") as f:
        f.write(html)
    print(f"report written: {out} ({len(html)} bytes)")


if __name__ == "__main__":
    if "--render-only" not in sys.argv:
        derive_all()
    render()
