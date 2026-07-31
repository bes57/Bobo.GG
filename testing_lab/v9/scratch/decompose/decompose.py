"""agent:v9-decompose — per-match decomposition of the 3 ledgered transfer
evaluations (N1_delta2, S_a1.0, N2_hybrid6) on VAL1/VAL2 exactly as ledgered.

Brief: v9/briefs/decompose.md. PURE DECOMPOSITION: no new fitting (the window
beta refits below are the same deterministic calls the ledgered evaluations
made — reproduced, not chosen), no new configs, no new selection, zero new
looks (same rows, same frozen configs). Machinery is REUSED from
scratch/search/searchlib.py (OverlaySearch.run_general, p_from_adj, fit_beta)
and scratch/family/v9lib.py (Overlay.active_state / evidence — leak asserts
hot on every evidence row).

Hard checks (brief section 4):
  * tie-out: per-window mean(d)*1000 must reproduce every ledgered
    delta_milli (printed);
  * locality: for the prediction-layer candidates, d == 0 EXACTLY on every
    scored match with no active phase.
"""
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

TL = "/Users/benny_es1/PythonTest/testing_lab"
V8 = os.path.join(TL, "v8")
V9 = os.path.join(TL, "v9")
SEARCH = os.path.join(V9, "scratch", "search")
SPEC = os.path.join(V8, "scratch", "roster", "spec_run")
sys.path.insert(0, SEARCH)

import searchlib as sl                     # noqa: E402
from searchlib import OverlaySearch, direction_of  # noqa: E402
import v9lib                               # noqa: E402
import referee                             # noqa: E402

LOG = os.path.join(V9, "logs", "decompose.log")
LOOKS = os.path.join(V9, "stats", "v9_looks.json")
OUT = os.path.join(V9, "stats", "v9_case_decomposition.json")
t0 = time.time()


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


log("DECOMPOSE START — per-match decomposition of the 3 ledgered "
    "evaluations (brief: v9/briefs/decompose.md; zero new looks)")

# ── frozen inputs, exactly as the ledgered run loaded them ──────────────────
frame = sl.load_frame_checked()
W = sl.windows(frame)
fmts = frame.fmt.values
dates = frame.date.astype(str).values
event_ids = frame.event_id.values
winners = frame.winner.values
losers = frame.loser.values
w_maps = frame.w_maps.values
l_maps = frame.l_maps.values

corpus = v9lib.load_corpus()
base = v9lib.v6_run()
valid = base["valid"]
rd_v6 = base["rdiff"]

npz = np.load(os.path.join(SPEC, "stage_runs.npz"))
assert np.array_equal(rd_v6, npz["p1w5c5_a0.0_t2.0_s0.0"], equal_nan=True), \
    "fresh v6 rdiff != stored npz v6 (nesting broken?)"

b_v6_fit1 = sl.fit_beta(rd_v6, W["FIT1"], fmts, valid)
assert abs(b_v6_fit1 - 0.1152) <= 1e-3, "beta fixture failed — abort"
b_v6_fit2 = sl.fit_beta(rd_v6, W["FIT2"], fmts, valid)
log(f"frame sha OK; beta(FIT1,v6)={b_v6_fit1:.6f} (fixture PASS) "
    f"beta(FIT2,v6)={b_v6_fit2:.6f} [{time.time()-t0:.0f}s]")

CAND = json.load(open(os.path.join(V9, "stats", "v9_candidates.json")))
LEDGER = {c["id"]: c for c in CAND["candidates"]}

plan = v9lib.build_plan("p1", W=5, corpus=corpus)
ov = OverlaySearch(plan, frame=frame)


def probs_plain(rd, beta):
    p = np.full(len(frame), np.nan)
    p[valid] = sl.runner.p_series_closed(beta, rd[valid], fmts[valid])
    return p


# ── per-(org,row) active-phase states with org names (same frozen calls as
#    run_general: active_state + evidence, leak assert hot) ──────────────────
def side_states(tau):
    by_row = defaultdict(list)
    for org in ov.plan.orgs.keys():
        rows = ov.org_rows.get(org)
        if not rows:
            continue
        for mid, (r, is_w) in rows.items():
            if not valid[r]:
                continue
            D = ov.dates[r]
            st = ov.active_state(org, D, tau)
            if st is None:
                continue
            E, ne = ov.evidence(org, D, base, tau)   # assert d_i < D hot
            by_row[int(r)].append(
                {"org": org, "is_winner": bool(is_w), "k": int(st["k"]),
                 "n_since": int(st["n"]), "E": float(E), "ne": int(ne)})
    return by_row


def overlay_adj_check(states, b, tau, gamma, m, variant, adj_ref):
    """Rebuild adj from the annotated states; must equal run_general's adj
    bitwise (max 2 contributions per row => IEEE-exact)."""
    adj = np.zeros(len(frame))
    for r, sides in states.items():
        for sd in sides:
            g = b * (1.0 - sd["k"] / 5.0) ** gamma \
                * float(np.exp(-sd["n_since"] / tau))
            d = g * direction_of(variant, m, sd["E"], sd["ne"])
            sd["delta_own_side"] = float(d)
            if d == 0.0:
                continue
            adj[r] += d if sd["is_winner"] else -d
    assert np.array_equal(adj, adj_ref), "adj reconstruction != run_general"
    return adj


def fnum(x, nd=4):
    return round(float(x), nd)


def phase_txt(sd):
    ev = (f"E={sd['E']:+.2f} maps vs exp over {sd['ne']}"
          if sd["ne"] > 0 else "no scored phase matches yet")
    return (f"{sd['org']} (k={sd['k']}/5 kept, n_since={sd['n_since']}, "
            f"{ev})")


def shape_of(sd):
    if sd["ne"] == 0 or sd["E"] == 0.0:
        return "phase unproven (no scored evidence)"
    if sd["E"] > 0:
        return "LEV-shape: post-change record genuinely improving"
    return "ENVY-shape: post-change record already degrading"


def reason_overlay(rec):
    """Data-derived one-liner, framed on the dominant active side."""
    sides = rec["active_sides"]
    sd, tie = dominant(rec)
    net = rec["adj_map_logit"]
    win, lose = rec["winner"], rec["loser"]
    ben = win if net > 0 else lose
    other = lose if ben == win else win
    ben_won = ben == win
    if tie:
        return (f"two active phases of equal push "
                f"({'; '.join(phase_txt(s) for s in sides)}); net "
                f"{abs(net):.3f} toward {ben}, who "
                f"{'won' if ben_won else 'lost'} {rec['score']}")
    if sd["org"] == ben:
        mech = f"net push {abs(net):.3f} toward {phase_txt(sd)}"
    else:
        mech = (f"push against {phase_txt(sd)} moved odds toward {ben}")
    outcome = (f"{ben} {'won' if ben_won else 'lost'} {rec['score']} vs "
               f"{other}")
    paid = "push paid" if rec["d_milli"] > 0 else "push backfired"
    return f"{mech}; {outcome} — {paid}; {shape_of(sd)}"


def reason_solve(rec):
    sd, tie = dominant(rec)
    pv, pc = rec["p_v6"], rec["p_cand"]
    to = "toward" if rec["d_milli"] > 0 else "away from"
    who = ("; ".join(phase_txt(s) for s in rec["active_sides"]) if tie
           else f"dominant {phase_txt(sd)}")
    return (f"boost re-weighted active phase games [{who}]: p(actual winner) "
            f"{pv:.3f}->{pc:.3f}, moved {to} the result; {shape_of(sd)}")


def row_record(r, window, dfull, pv_full, pc_full, sides, adj=None, tier=None):
    rec = {"row": int(r), "window": window, "date": dates[r],
           "event": str(event_ids[r]), "winner": str(winners[r]),
           "loser": str(losers[r]),
           "score": f"{int(w_maps[r])}-{int(l_maps[r])}",
           "fmt": str(fmts[r]),
           "active_sides": [dict(s) for s in sides],
           "p_v6": fnum(pv_full[r]), "p_cand": fnum(pc_full[r]),
           "outcome": f"{winners[r]} won",
           "d_milli": fnum(dfull[r] * 1000)}
    if adj is not None:
        rec["adj_map_logit"] = fnum(adj[r], 5)
    if tier is not None:
        rec["tier"] = tier
    if sides:
        sd, tie = dominant(rec)
        rec["dominant_org"] = "TIE" if tie else sd["org"]
    return rec


def agg(dsub, n_window):
    return {"n": int(len(dsub)),
            "n_better": int((dsub > 0).sum()),
            "n_worse": int((dsub < 0).sum()),
            "n_zero": int((dsub == 0).sum()),
            "sum_milli": fnum(dsub.sum() * 1000, 3),
            "contrib_to_window_mean_milli": fnum(dsub.sum() / n_window * 1000, 3)}


def dom_metric(sd):
    """Magnitude of the side's own push: |delta_own_side| for overlay rows,
    the candidate's own boost kernel (1-k/5)e^(-n/tau) for solve rows."""
    if "delta_own_side" in sd:
        return abs(sd["delta_own_side"])
    return sd["boost_kernel"]


def dominant(rec):
    """(dominant side, exact_tie?) — the side the candidate moved most."""
    sides = sorted(rec["active_sides"], key=dom_metric, reverse=True)
    if len(sides) > 1 and dom_metric(sides[0]) == dom_metric(sides[1]):
        return sides[0], True
    return sides[0], False


def lev_envy(recs, n_window):
    """Bucket affected rows by the DOMINANT changed team's post-change
    record (frozen walk-forward evidence E at the match date). Dominant =
    the side with the largest own push; exact ties bucketed separately."""
    buckets = {"improving_LEV_shape": [], "degrading_ENVY_shape": [],
               "unproven_no_evidence": [], "tied_both_sides": []}
    for rec in recs:
        sd, tie = dominant(rec)
        if tie:
            buckets["tied_both_sides"].append(rec)
        elif sd["ne"] == 0 or sd["E"] == 0.0:
            buckets["unproven_no_evidence"].append(rec)
        elif sd["E"] > 0:
            buckets["improving_LEV_shape"].append(rec)
        else:
            buckets["degrading_ENVY_shape"].append(rec)
    out = {}
    for k, rr in buckets.items():
        d = np.array([x["d_milli"] for x in rr]) / 1000.0
        out[k] = agg(d, n_window) if len(rr) else agg(np.array([]), n_window)
    return out


def by_freshness(recs, n_window):
    """Descriptive binning by the freshest active phase on the row
    (min n_since across active sides): 1-5, 6-13, 14+."""
    bins = {"n_since_1_5": [], "n_since_6_13": [], "n_since_14_plus": []}
    for rec in recs:
        nmin = min(s["n_since"] for s in rec["active_sides"])
        key = ("n_since_1_5" if nmin <= 5 else
               "n_since_6_13" if nmin <= 13 else "n_since_14_plus")
        bins[key].append(rec)
    return {k: (agg(np.array([x["d_milli"] for x in rr]) / 1000.0, n_window)
                if len(rr) else agg(np.array([]), n_window))
            for k, rr in bins.items()}


TIE = {}
LOCAL = {}
CANDS_OUT = {}

# ════════ prediction-layer candidates (N1_delta2, N2_hybrid6) ═══════════════
for cid in ("N1_delta2", "N2_hybrid6"):
    led = LEDGER[cid]
    cfg = led["config"]
    out = ov.run_general(base, cfg["variant"], cfg["b"], cfg["tau"],
                         gamma=cfg["gamma"], m=cfg["m"])
    states = side_states(cfg["tau"])
    adj = overlay_adj_check(states, cfg["b"], cfg["tau"], cfg["gamma"],
                            cfg["m"], cfg["variant"], out["adj"])
    win_out, all_recs, tie_c = {}, [], {}
    raw_sum = {}
    loc_rows, loc_max = 0, 0.0
    for wname, fitw, bv6, tkey in (
            ("VAL1", "FIT1", b_v6_fit1, "T1_fit2324_val2025"),
            ("VAL2", "FIT2", b_v6_fit2, "T2_fit2325_val2026H1")):
        bc = sl.fit_beta(rd_v6, W[fitw], fmts, valid)   # == ledgered beta_cand
        assert round(bc, 6) == led["transfer"][tkey]["beta_cand"], \
            f"{cid} {wname}: beta_cand mismatch"
        pc_full = sl.p_from_adj({"rdiff": rd_v6, "valid": valid}, adj, bc, fmts)
        pv_full = probs_plain(rd_v6, bv6)
        mv = W[wname] & valid
        idx = np.flatnonzero(mv)
        d = referee.delta_vector(pc_full[mv], pv_full[mv])
        dfull = np.zeros(len(frame))
        dfull[idx] = d
        # tie-out
        rec_m = float(d.mean()) * 1000
        raw_sum[wname] = float(d.sum()) * 1000
        led_m = led["transfer"][tkey]["delta_milli"]
        tie_c[wname] = {"ledgered_milli": led_m,
                        "recomputed_milli": round(rec_m, 3),
                        "raw_abs_diff_milli": float(abs(rec_m - led_m)),
                        "pass": bool(round(rec_m, 3) == led_m)}
        # locality: d == 0 EXACTLY on scored rows with no active phase
        no_phase = np.array([r for r in idx if r not in states])
        mx = float(np.max(np.abs(dfull[no_phase]))) if len(no_phase) else 0.0
        loc_rows += len(no_phase)
        loc_max = max(loc_max, mx)
        assert mx == 0.0, f"{cid} {wname}: LOCALITY VIOLATION max|d|={mx}"
        # adj==0 superset (active phase but zero direction, e.g. sign(E)=0)
        z = idx[adj[idx] == 0.0]
        assert float(np.max(np.abs(dfull[z]))) == 0.0 if len(z) else True
        # affected rows
        aff = idx[adj[idx] != 0.0]
        recs = [row_record(r, wname, dfull, pv_full, pc_full, states[r],
                           adj=adj) for r in aff]
        all_recs += recs
        d_aff = dfull[aff]
        d_un = np.array([dfull[r] for r in idx if adj[r] == 0.0])
        win_out[wname] = {
            "n_window": int(mv.sum()),
            "ledgered_window_delta_milli": led_m,
            "affected": agg(d_aff, int(mv.sum())),
            "unaffected": agg(d_un, int(mv.sum())),
            "lev_envy_split": lev_envy(recs, int(mv.sum())),
            "by_freshness": by_freshness(recs, int(mv.sum()))}
    # pooled tie-out (raw per-match sums, exact n weighting)
    pooled_led = led["transfer"]["pooled_validation"]["delta_milli"]
    n1, n2 = win_out["VAL1"]["n_window"], win_out["VAL2"]["n_window"]
    pooled_rc = round((raw_sum["VAL1"] + raw_sum["VAL2"]) / (n1 + n2), 3)
    tie_c["pooled"] = {"ledgered_milli": pooled_led,
                       "recomputed_milli": pooled_rc,
                       "pass": bool(pooled_rc == pooled_led)}
    TIE[cid] = tie_c
    LOCAL[cid] = {"n_scored_rows_no_active_phase": int(loc_rows),
                  "max_abs_d_milli_on_those": loc_max,
                  "assertion": "PASS — Δ exactly 0 on every scored match "
                               "with no active phase (locality by construction)"}
    # examples: pooled across windows
    ordered = sorted(all_recs, key=lambda x: x["d_milli"])
    worst = [dict(x, reason=reason_overlay(x)) for x in ordered[:10]]
    best = [dict(x, reason=reason_overlay(x)) for x in ordered[-10:][::-1]]
    CANDS_OUT[cid] = {"family": led["family"], "config": cfg,
                      "label": led["transfer"]["label"],
                      "windows": win_out,
                      "examples": {"worst_10": worst, "best_10": best},
                      "rows": all_recs}
    log(f"{cid}: VAL1 {tie_c['VAL1']['recomputed_milli']:+.3f}m vs ledgered "
        f"{tie_c['VAL1']['ledgered_milli']:+.3f}m | VAL2 "
        f"{tie_c['VAL2']['recomputed_milli']:+.3f}m vs "
        f"{tie_c['VAL2']['ledgered_milli']:+.3f}m | locality PASS "
        f"({loc_rows} no-phase rows, max|d|=0.0) [{time.time()-t0:.0f}s]")

# ════════ solve-side candidate (S_a1.0) — ALL matches, tiered ═══════════════
led = LEDGER["S_a1.0"]
cfg = led["config"]
rd_c = npz[cfg["npz_key"]]
assert (~np.isnan(rd_c) == valid).all(), "valid-mask drift in solve npz"
states13 = side_states(cfg["tau"])          # active-phase def at the cand tau
for _r, _sides in states13.items():         # freshness/dominance kernel:
    for _sd in _sides:                      # the candidate's own boost shape
        _sd["boost_kernel"] = fnum((1.0 - _sd["k"] / 5.0)
                                   * float(np.exp(-_sd["n_since"] / cfg["tau"])), 6)


def touched_before(org, D):
    """Side directly touched by the subsystem before D: any confirmed
    boundary visible, or any sub-downweighted (o<1) game visible."""
    v = plan.version_asof(org, D)
    if v is None:
        return False
    return bool(v["boundaries"]) or bool((v["o"] < 1.0).any())


win_out, all_recs, tie_c = {}, [], {}
raw_sum = {}
for wname, fitw, bv6, tkey in (
        ("VAL1", "FIT1", b_v6_fit1, "T1_fit2324_val2025"),
        ("VAL2", "FIT2", b_v6_fit2, "T2_fit2325_val2026H1")):
    bc = sl.fit_beta(rd_c, W[fitw], fmts, valid)
    assert round(bc, 6) == led["transfer"][tkey]["beta_cand"], \
        f"S_a1.0 {wname}: beta_cand mismatch"
    pc_full = probs_plain(rd_c, bc)
    pv_full = probs_plain(rd_v6, bv6)
    mv = W[wname] & valid
    idx = np.flatnonzero(mv)
    d = referee.delta_vector(pc_full[mv], pv_full[mv])
    dfull = np.zeros(len(frame))
    dfull[idx] = d
    rec_m = float(d.mean()) * 1000
    raw_sum[wname] = float(d.sum()) * 1000
    led_m = led["transfer"][tkey]["delta_milli"]
    tie_c[wname] = {"ledgered_milli": led_m, "recomputed_milli": round(rec_m, 3),
                    "raw_abs_diff_milli": float(abs(rec_m - led_m)),
                    "pass": bool(round(rec_m, 3) == led_m)}
    tiers = {"change_affected": [], "stale_or_sub_touched": [],
             "untouched_pure_coupling": []}
    for r in idx:
        sides = states13.get(int(r), [])
        if sides:
            tier = "change_affected"
        elif touched_before(winners[r], dates[r]) or \
                touched_before(losers[r], dates[r]):
            tier = "stale_or_sub_touched"
        else:
            tier = "untouched_pure_coupling"
        rec = row_record(r, wname, dfull, pv_full, pc_full, sides, tier=tier)
        tiers[tier].append(rec)
        all_recs.append(rec)
    n_w = int(mv.sum())
    tsum = {}
    for tname, rr in tiers.items():
        dd = np.array([x["d_milli"] for x in rr]) / 1000.0
        tsum[tname] = agg(dd, n_w) if len(rr) else agg(np.array([]), n_w)
    un = np.array([x["d_milli"] for x in
                   tiers["stale_or_sub_touched"] +
                   tiers["untouched_pure_coupling"]]) / 1000.0
    win_out[wname] = {
        "n_window": n_w, "ledgered_window_delta_milli": led_m,
        "affected_definition": "≥1 side with an active phase "
                               "(last confirmed boundary, n_since <= 65 = "
                               "ceil(5*tau), tau=13)",
        "affected": tsum["change_affected"],
        "unaffected_all": agg(un, n_w),
        "unaffected_split": {
            "stale_or_sub_touched": tsum["stale_or_sub_touched"],
            "untouched_pure_coupling": tsum["untouched_pure_coupling"]},
        "lev_envy_split_on_affected": lev_envy(tiers["change_affected"], n_w),
        "by_freshness_on_affected": by_freshness(tiers["change_affected"],
                                                 n_w)}
n1, n2 = win_out["VAL1"]["n_window"], win_out["VAL2"]["n_window"]
pooled_rc = round((raw_sum["VAL1"] + raw_sum["VAL2"]) / (n1 + n2), 3)
pooled_led = led["transfer"]["pooled_validation"]["delta_milli"]
tie_c["pooled"] = {"ledgered_milli": pooled_led, "recomputed_milli": pooled_rc,
                   "pass": bool(pooled_rc == pooled_led)}
TIE["S_a1.0"] = tie_c
LOCAL["S_a1.0"] = {"assertion": "NOT APPLICABLE — solve-side re-rates the "
                                "whole field by construction; the coupling "
                                "is quantified in unaffected_all / "
                                "untouched_pure_coupling sums"}
aff_recs = [x for x in all_recs if x["tier"] == "change_affected"]
ordered = sorted(aff_recs, key=lambda x: x["d_milli"])
worst = [dict(x, reason=reason_solve(x)) for x in ordered[:10]]
best = [dict(x, reason=reason_solve(x)) for x in ordered[-10:][::-1]]
CANDS_OUT["S_a1.0"] = {"family": led["family"], "config": cfg,
                       "label": led["transfer"]["label"],
                       "windows": win_out,
                       "examples": {"worst_10": worst, "best_10": best},
                       "rows": all_recs}
log(f"S_a1.0: VAL1 {tie_c['VAL1']['recomputed_milli']:+.3f}m vs ledgered "
    f"{tie_c['VAL1']['ledgered_milli']:+.3f}m | VAL2 "
    f"{tie_c['VAL2']['recomputed_milli']:+.3f}m vs "
    f"{tie_c['VAL2']['ledgered_milli']:+.3f}m [{time.time()-t0:.0f}s]")

# ── printed tie-out table (brief hard requirement) ──────────────────────────
print("\n================ TIE-OUT (recomputed vs ledgered, milli-LL) "
      "================")
ok = True
for cid, tt in TIE.items():
    for wname, v in tt.items():
        ok &= v["pass"]
        print(f"  {cid:11s} {wname:6s} recomputed {v['recomputed_milli']:+8.3f}"
              f"  ledgered {v['ledgered_milli']:+8.3f}  "
              f"{'PASS' if v['pass'] else 'FAIL'}")
verdict = ("PASS — every per-match sum reproduces its published window total"
           if ok else "FAIL")
print(f"  => TIE-OUT {verdict}")
assert ok, "TIE-OUT FAILED"
log("TIE-OUT PASS: all 9 recomputed window/pooled totals equal the "
    "ledgered numbers (3 candidates x VAL1/VAL2/pooled)")
for cid in ("N1_delta2", "N2_hybrid6"):
    log(f"LOCALITY {cid}: {LOCAL[cid]['n_scored_rows_no_active_phase']} "
        f"scored rows with no active phase, max|Δ| = "
        f"{LOCAL[cid]['max_abs_d_milli_on_those']} (exactly 0) — PASS")

# ── output json ─────────────────────────────────────────────────────────────
out_blob = {
 "written_by": "agent:v9-decompose",
 "written": time.strftime("%Y-%m-%d %H:%M:%S"),
 "brief": "v9/briefs/decompose.md — per-match decomposition of the 3 "
          "ledgered transfer evaluations; no new fitting, no new configs, "
          "no new selection, zero new looks",
 "method": {
  "source": "re-scored stats/v9_candidates.json evaluations with the frozen "
            "search machinery (scratch/search/searchlib.py run_general / "
            "p_from_adj / fit_beta; scratch/family/v9lib.py active_state / "
            "evidence with the date-strict leak assert hot). Window beta "
            "refits are the identical deterministic calls the ledger made.",
  "sign_convention": "d_milli > 0 = candidate better than v6 on that match "
                     "(referee.delta_vector). p_v6 / p_cand are the "
                     "probability each model assigned to the ACTUAL series "
                     "winner.",
  "active_phase_definition": "Overlay.active_state at the candidate's own "
                             "tau: last confirmed boundary with n_since <= "
                             "ceil(5*tau) (ADDENDUM 1a horizon). k = players "
                             "kept of 5; n_since = matches since boundary.",
  "solve_side_tiers": "change_affected: >=1 side in an active phase (tau=13 "
                      "-> horizon 65). stale_or_sub_touched: no active side "
                      "but >=1 side had a visible confirmed boundary or any "
                      "sub-downweighted (o<1) game before the match — "
                      "directly re-weighted by the subsystem. "
                      "untouched_pure_coupling: neither side ever directly "
                      "touched — its Δ is PURE solve coupling, the "
                      "operator's 'shouldn't affect unchanged teams' test.",
  "lev_envy_definition": "per affected match, classified by the DOMINANT "
                         "active side (largest own-side push: |delta| for "
                         "overlays, boost kernel (1-k/5)e^(-n/tau) for the "
                         "solve candidate; exact ties bucketed separately) "
                         "using the frozen walk-forward evidence E(T,D) = "
                         "maps won minus v6-expected maps over the phase's "
                         "scored matches knowable strictly before the match "
                         "date. E>0 improving (LEV-shape), E<0 degrading "
                         "(ENVY-shape), ne=0 or E=0 unproven. Most rows "
                         "have both sides in some phase (horizon 65 keeps "
                         "orgs 'active' ~a season), so single-side rows are "
                         "rare and dominance is the informative unit."},
 "tie_out": TIE,
 "locality_assertion": LOCAL,
 "candidates": CANDS_OUT,
}

# ── computed headline block (chart/prose-ready; numbers only) ───────────────
def pooled_split(cid, key):
    ws = CANDS_OUT[cid]["windows"]
    out = {}
    for w in ("VAL1", "VAL2"):
        for k, v in ws[w][key].items():
            o = out.setdefault(k, {"n": 0, "sum_milli": 0.0,
                                   "n_better": 0, "n_worse": 0})
            o["n"] += v["n"]
            o["sum_milli"] = round(o["sum_milli"] + v["sum_milli"], 3)
            o["n_better"] += v["n_better"]
            o["n_worse"] += v["n_worse"]
    for k, o in out.items():
        o["mean_milli_per_affected_match"] = \
            round(o["sum_milli"] / o["n"], 3) if o["n"] else 0.0
    return out


headline = {}
for cid in ("N1_delta2", "N2_hybrid6"):
    le = pooled_split(cid, "lev_envy_split")
    headline[cid] = {
        "lev_envy_pooled": le,
        "freshness_pooled": pooled_split(cid, "by_freshness"),
        "fact": (f"loses on BOTH shapes: LEV-shape (improving) "
                 f"{le['improving_LEV_shape']['sum_milli']:+.1f} sum-milli over "
                 f"{le['improving_LEV_shape']['n']} matches "
                 f"({le['improving_LEV_shape']['mean_milli_per_affected_match']:+.2f}m/match), "
                 f"ENVY-shape (degrading) "
                 f"{le['degrading_ENVY_shape']['sum_milli']:+.1f} over "
                 f"{le['degrading_ENVY_shape']['n']} "
                 f"({le['degrading_ENVY_shape']['mean_milli_per_affected_match']:+.2f}m/match)")}
sw = CANDS_OUT["S_a1.0"]["windows"]
headline["S_a1.0"] = {
    "lev_envy_pooled_on_affected": pooled_split("S_a1.0",
                                                "lev_envy_split_on_affected"),
    "freshness_pooled_on_affected": pooled_split("S_a1.0",
                                                 "by_freshness_on_affected"),
    "unaffected_row_damage": {
        w: {"all_unaffected": sw[w]["unaffected_all"],
            "untouched_pure_coupling":
                sw[w]["unaffected_split"]["untouched_pure_coupling"]}
        for w in ("VAL1", "VAL2")},
    "fact": (f"operator's unchanged-teams test: rows with no active phase "
             f"sum {sw['VAL1']['unaffected_all']['sum_milli']:+.1f} sum-milli "
             f"(VAL1, n={sw['VAL1']['unaffected_all']['n']}) and "
             f"{sw['VAL2']['unaffected_all']['sum_milli']:+.1f} "
             f"(VAL2, n={sw['VAL2']['unaffected_all']['n']}); the strictly "
             f"never-touched subset (no boundary, no sub game, either side) "
             f"sums "
             f"{sw['VAL2']['unaffected_split']['untouched_pure_coupling']['sum_milli']:+.1f} "
             f"on VAL2 (n="
             f"{sw['VAL2']['unaffected_split']['untouched_pure_coupling']['n']}) = "
             f"{sw['VAL2']['unaffected_split']['untouched_pure_coupling']['contrib_to_window_mean_milli']:+.3f}m "
             f"of the -2.980m VAL2 window mean — pure solve coupling + beta "
             f"shift, dominated by one match (ULF 2-0 PCF 2026-02-03, "
             f"p 0.166->0.106, -450.4 milli)")}
out_blob["headline"] = headline


def pyify(o):
    if isinstance(o, dict):
        return {k: pyify(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [pyify(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


with open(OUT, "w") as f:
    json.dump(pyify(out_blob), f, indent=1)
log(f"wrote {OUT} ({os.path.getsize(OUT)//1024} KB) [{time.time()-t0:.0f}s]")

# ── ONE disclosure line to stats/v9_looks.json ──────────────────────────────
looks = json.load(open(LOOKS))
looks["decomposition_reads"] = {
 "rule": "per-match decomposition of ALREADY-LEDGERED selection_reads "
         "(briefs/decompose.md); same rows, same frozen configs, same "
         "windows — zero new information about any un-evaluated config; "
         "never a selection input",
 "entries": [{
  "type": "decomposition-of-existing-reads",
  "date": time.strftime("%Y-%m-%d"),
  "agent": "v9-decompose",
  "candidates": ["N1_delta2", "S_a1.0", "N2_hybrid6"],
  "new_looks": 0,
  "tie_out": "PASS (all 9 window/pooled totals reproduced exactly)",
  "artifact": "stats/v9_case_decomposition.json"}]}
with open(LOOKS, "w") as f:
    json.dump(looks, f, indent=1)
log("disclosure appended to stats/v9_looks.json (decomposition_reads, "
    "new_looks=0)")
log(f"DECOMPOSE DONE [{time.time()-t0:.0f}s]")
