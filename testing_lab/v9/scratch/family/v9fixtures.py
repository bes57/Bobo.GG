"""v9 FAMILY conformance fixtures V1-V7 — run against v9lib BEFORE any
fitting exists anywhere in v9. Preregistered: v9/preregister.family.md
(LOCKED 2026-07-29 11:32; ADDENDUM 1 11:40). Any failure writes
stats/v9_fixtures.json with status=FAILED and raises; numbers produced
after a failed fixture are void.

No metric is computed anywhere here: assertions compare model OUTPUTS
(ratings, probabilities) for identity/behavior only. Holdout metrics are
popped unseen inside runner.run_config; the manual hook-exercise run pops
them immediately after eng.run().

mini_corpus / mini_solve / max_abs_diff are copied VERBATIM from
testing_lab/v8/scratch/roster/spec_run/fixtures.py (agent:roster-g, SPEC
RUN) — that module executes its fixture suite on import, so importing it
is not an option. Attribution per the reuse rule.
"""
import json
import os
import sys
from collections import defaultdict
from datetime import date as _date, timedelta

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V9 = os.path.dirname(os.path.dirname(HERE))
STATS = os.path.join(V9, "stats")
sys.path.insert(0, HERE)

import v9lib  # noqa: E402
from v9lib import (Overlay, build_plan, daily_checksum, hybrid_run,  # noqa: E402
                   load_corpus, rd_checksum, series_from_pm, solve_side_run,
                   v6_run)
import runner  # noqa: E402  (via v9lib sys.path)

LEDGER = []
FIX = {"written_by": "agent:v9-family",
       "preregistered": "v9/preregister.family.md (2026-07-29 11:32) + "
                        "ADDENDUM 1 (11:40), both before implementation",
       "order": "fixtures ran BEFORE any v9 fitting/scoring exists "
                "(logs/family.log)",
       "assertions": LEDGER}


def check(name, expected, got):
    ok = expected == got
    LEDGER.append({"name": name, "expected": repr(expected), "got": repr(got),
                   "pass": bool(ok)})
    if not ok:
        _flush("FAILED")
        raise AssertionError(f"FIXTURE FAIL: {name}: expected {expected!r} "
                             f"got {got!r}")
    return True


def check_true(name, cond, detail=""):
    LEDGER.append({"name": name, "expected": "True",
                   "got": f"{bool(cond)} {detail}".strip(),
                   "pass": bool(cond)})
    if not cond:
        _flush("FAILED")
        raise AssertionError(f"FIXTURE FAIL: {name} {detail}")
    return True


def _flush(status):
    FIX["status"] = status
    FIX["n_pass"] = sum(1 for a in LEDGER if a["pass"])
    FIX["n_fail"] = sum(1 for a in LEDGER if not a["pass"])
    with open(os.path.join(STATS, "v9_fixtures.json"), "w") as f:
        json.dump(FIX, f, indent=1)


def eq_arr(a, b):
    return bool(np.array_equal(a, b, equal_nan=True))


def neq_mask(a, b):
    return ~((a == b) | (np.isnan(a) & np.isnan(b)))


# ════════════════════════════════════════════════════════════════════════════
# Synthetic mini-corpus — copied VERBATIM from v8 spec_run/fixtures.py
# (agent:roster-g); see module docstring for attribution.
# ════════════════════════════════════════════════════════════════════════════
def mini_corpus(change_round, sustained):
    """T00 featured. Lineup A rounds < change_round; lineup N (2 swapped,
    k=3) at change_round, and onward iff sustained else one round only."""
    teams = [f"T{i:02d}" for i in range(16)]
    A = frozenset({f"T00_p{j}" for j in range(5)})
    N = frozenset({"T00_p0", "T00_p1", "T00_p2", "T00_s3", "T00_s4"})
    games, lineups, seq = [], {}, defaultdict(list)
    mid = 1000
    for r in range(12):
        d = f"2024-{1 + r // 4:02d}-{1 + 7 * (r % 4):02d}"
        order = [0] + [(i + r) % 15 + 1 for i in range(15)]
        for p in range(8):
            t1, t2 = teams[order[p]], teams[order[15 - p]]
            w, l = (t1, t2) if order[p] < order[15 - p] else (t2, t1)
            games.append({"date_s": d, "match_id": mid, "winner": w,
                          "loser": l, "event_id": "mini"})
            for org in (w, l):
                seq[org].append((d, mid))
                if org == "T00":
                    on = (r == change_round if not sustained
                          else r >= change_round)
                    lineups[(org, mid)] = N if on else A
                else:
                    lineups[(org, mid)] = frozenset(
                        {f"{org}_p{j}" for j in range(5)})
            mid += 1
    for org in seq:
        seq[org].sort()
    corpus = {"team_match_seq": dict(seq), "lineups": lineups,
              "games": games, "teams": teams}
    return corpus, A, N


def mini_solve(corpus, plan, a, tau, s):
    """Standalone walk-forward Massey using the SAME multipliers code path
    as the real engine hook (here: SpecPlanV9's capped multipliers)."""
    teams = corpus["teams"]
    tidx = {t: i for i, t in enumerate(teams)}
    games = corpus["games"]
    seq = corpus["team_match_seq"]
    pos_of = {org: {m: i for i, (d, m) in enumerate(sq)}
              for org, sq in seq.items()}
    days = sorted({g["date_s"] for g in games})
    days = days[1:] + ["2024-12-31"]
    out = {}
    for D in days:
        hist = [g for g in games if g["date_s"] < D]
        if len(hist) < 8:
            continue
        n_t = len(teams)
        M = np.zeros((n_t, n_t)); p = np.zeros(n_t)
        vers = {}
        for org in teams:
            v = plan.version_asof(org, D) if plan is not None else None
            if v is not None:
                vers[org] = (v, plan.multipliers(v, a, tau, s))
        for g in hist:
            w = 1.0
            for org in (g["winner"], g["loser"]):
                if org in vers:
                    v, mu = vers[org]
                    pos = pos_of[org][g["match_id"]]
                    if pos < v["nvis"]:
                        w *= mu[pos]
            wi, li = tidx[g["winner"]], tidx[g["loser"]]
            rd = 3.0
            M[wi, wi] += w; M[li, li] += w
            M[wi, li] -= w; M[li, wi] -= w
            p[wi] += w * rd; p[li] -= w * rd
        M[np.diag_indices(n_t)] += 0.5
        M[-1, :] = 1.0; p[-1] = 0.0
        out[D] = np.linalg.solve(M, p)
    return out


def max_abs_diff(sa, sb, team_i=None, day_filter=None):
    mx = 0.0
    for D in sa:
        if day_filter and not day_filter(D):
            continue
        d = np.abs(sa[D] - sb[D])
        mx = max(mx, float(d[team_i] if team_i is not None else d.max()))
    return mx


# ════════════════════════════════════════════════════════════════════════════
print("=== V1: nesting a=s=b=0 => bit-identical v6 ===", flush=True)
base_v6 = v6_run(daily=True)
sha_rd0 = rd_checksum(base_v6["rdiff"])
sha_dl0 = daily_checksum(base_v6["daily_r"])
sha_p0 = rd_checksum(base_v6["p_all"])
corpus = load_corpus()
plan5 = build_plan("p1", W=5, c=5, corpus=corpus)

# (i) hook-exercised: spec ENABLED with a=s=0, run through the real engine
eng, frame = runner.get_engine()
eng._prev_rvec = None
eng.enable_spec(plan5, 0.0, 5.0, 0.0)
out_hook = eng.run(runner.v6_cfg(eng, daily=True))
eng.disable_spec()
for _k in ("ll_test", "brier_test", "p_test", "test_mask"):
    out_hook.pop(_k, None)                  # LOOK HYGIENE, immediately
check("V1 hook-path a=s=0: sha256(rdiff) == pure v6", sha_rd0,
      rd_checksum(out_hook["rdiff"]))
check("V1 hook-path a=s=0: sha256(daily ratings) == pure v6", sha_dl0,
      daily_checksum(out_hook["daily_r"]))
del out_hook

# (ii) API-level: solve_side_run(a=0, s=0)
r_api = solve_side_run(plan5, 0.0, 5.0, 0.0, daily=True)
check("V1 solve_side_run a=s=0: sha256(rdiff) == pure v6", sha_rd0,
      rd_checksum(r_api["rdiff"]))
check("V1 solve_side_run a=s=0: sha256(daily) == pure v6", sha_dl0,
      daily_checksum(r_api["daily_r"]))
del r_api

# (iii) hybrid a=s=b=0 and overlay b=0
h0 = hybrid_run(plan5, 0.0, 5.0, 0.0, 0.0, 5.0)
check("V1 hybrid a=s=b=0: p_all bit-identical to v6", True,
      eq_arr(h0["p_all"], base_v6["p_all"]))
del h0
ov = Overlay(plan5)
o0 = ov.run(base_v6, "delta2", 0.0, 5.0)
check("V1 overlay b=0: zero affected rows", 0, int(o0["affected_rows"].size))
check("V1 overlay b=0: p_all bit-identical to v6", True,
      eq_arr(o0["p_all"], base_v6["p_all"]))
check("V1 overlay: rdiff passes through as the SAME object (never re-solved)",
      True, o0["rdiff"] is base_v6["rdiff"])
# series_from_pm consistency with the base series step, bitwise
valid = base_v6["valid"]
pm0 = 1.0 / (1.0 + np.exp(-(base_v6["beta"] * base_v6["rdiff"][valid])))
check("V1 series_from_pm(sigmoid(beta*rdiff)) bitwise == base p_all", True,
      eq_arr(series_from_pm(pm0, frame.fmt.values[valid]),
             base_v6["p_all"][valid]))

print("=== V2: SEN marved case (real corpus, c=5) ===", flush=True)
SEN_SEQ = corpus["team_match_seq"]["SEN"]
pos_of_sen = {m: i for i, (d, m) in enumerate(SEN_SEQ)}
J_MARVED = pos_of_sen[706350]
WINDOW = "2026-05-01"
names = {u.rsplit("/", 1)[-1] for u in corpus["lineups"][("SEN", 706350)]}
check("V2 data: marved fielded in m706350", True, "marved" in names)
for W in (3, 5, 8):
    plc = plan5 if W == 5 else build_plan("p1", W=W, c=5, corpus=corpus)
    fb = plc.final("SEN")["boundaries"]
    check(f"V2 p1w{W}c5: no boundary at the marved match", [],
          [b for b in fb if b["j"] == J_MARVED])
    check(f"V2 p1w{W}c5: zero boundaries dated >= {WINDOW}", [],
          [b["date"] for b in fb if b["date"] >= WINDOW])
    if W == 5:
        check("V2 p1w5c5: marved game IS a deviation (o=0.8)", 0.8,
              float(plc.final("SEN")["o"][J_MARVED]))
# ADDENDUM 1b counterfactual identity: marved one-off contributes NOTHING
corpus_nodev = dict(corpus)
corpus_nodev["lineups"] = dict(corpus["lineups"])
corpus_nodev["lineups"][("SEN", 706350)] = corpus["lineups"][("SEN", 706360)]
plan5_nodev = build_plan("p1", W=5, c=5, corpus=corpus_nodev)
ov_nodev = Overlay(plan5_nodev, frame=frame)
for variant in ("delta2", "delta1"):
    pa = ov.run(base_v6, variant, 1.0, 5.0)["p_all"]
    pb = ov_nodev.run(base_v6, variant, 1.0, 5.0)["p_all"]
    check(f"V2 {variant}: overlay p_all bit-identical, real vs marved-"
          "reverted corpus (one-off sub contributes nothing)", True,
          eq_arr(pa, pb))
sen_asof = ov.active_state("SEN", WINDOW, 5.0)
FIX["v2_sen_note"] = {
    "expected_behavior": "SEN carries a REAL confirmed boundary 2026-04-19 "
                         "(k=3) — active in the window by design; the "
                         "marved one-off creates no boundary and no delta "
                         "difference (counterfactual identity above)",
    "sen_state_at_window_start": {"n": sen_asof["n"], "k": sen_asof["k"]}}

print("=== V3: synthetic revert (lag-correct, c=5) ===", flush=True)
corp2, A2, N2 = mini_corpus(change_round=5, sustained=False)
for W in (3, 5, 8):
    pl = build_plan("p1", W=W, c=5, corpus=corp2)
    check(f"V3 p1w{W}c5: zero boundaries", [],
          pl.final("T00")["boundaries"])
pl_p3 = build_plan("p3", W=5, c=5, corpus=corp2)
check("V3 p3w5: frozen o flags exactly the one sub game", 1,
      int((pl_p3.final("T00")["o"] < 1).sum()))
pl15 = build_plan("p1", W=5, c=5, corpus=corp2)
sa = mini_solve(corp2, pl15, a=2.0, tau=5.0, s=0.0)
sb = mini_solve(corp2, None, a=0.0, tau=5.0, s=0.0)
check("V3 p1w5c5 mini-solve: max|dR| before/during/after == 0.0 exactly",
      0.0, max_abs_diff(sa, sb))

print("=== V4: synthetic sustained change (lag-correct, c=5) ===", flush=True)
corp3, A3, N3 = mini_corpus(change_round=5, sustained=True)
rd_date = {r: f"2024-{1 + r // 4:02d}-{1 + 7 * (r % 4):02d}"
           for r in range(12)}
LAG = {3: 6, 5: 7, 8: 8}
pl35 = None
for W, det_idx in LAG.items():
    pl = build_plan("p1", W=W, c=5, corpus=corp3)
    if W == 5:
        pl35 = pl
    evs = pl.orgs["T00"]["events"]
    check(f"V4 p1w{W}c5: exactly one detection", 1, len(evs))
    check(f"V4 p1w{W}c5: boundary index j == 5", 5, evs[0]["j"])
    check(f"V4 p1w{W}c5: detected at match index {det_idx}, never earlier",
          rd_date[det_idx], evs[0]["det_date"])
    check(f"V4 p1w{W}c5: k == 3", 3, evs[0]["k"])
    check(f"V4 p1w{W}c5: one final boundary at j=5, k=3", [(5, 3, False)],
          [(b["j"], b["k"], b["provisional"])
           for b in pl.final("T00")["boundaries"]])
check("V4 p1w5c5: new-phase games o==1 (not penalized)", True,
      bool((pl35.final("T00")["o"][5:] == 1.0).all()))
check("V4 p1w5c5: per-game n counts from the boundary",
      list(range(7)), pl35.final("T00")["n"][5:].tolist())
sa3 = mini_solve(corp3, pl35, a=2.0, tau=5.0, s=0.0)
sb3 = mini_solve(corp3, None, a=0.0, tau=5.0, s=0.0)
det5 = rd_date[7]
check("V4 p1w5c5 mini-solve: ALL teams identical through detection date",
      0.0, max_abs_diff(sa3, sb3, day_filter=lambda D: D <= det5))
check_true("V4 p1w5c5 mini-solve: featured team moves after detection",
           max_abs_diff(sa3, sb3, team_i=0,
                        day_filter=lambda D: D > det5) > 0)

print("=== V5: prediction-layer pre-change identity + zero coupling ===",
      flush=True)
dates_f = frame.date.astype(str).values
win_f = frame.winner.values
los_f = frame.loser.values
orgs_ev = sorted(o for o in plan5.orgs if plan5.orgs[o]["events"])
first_det = {o: min(e["det_date"] for e in plan5.orgs[o]["events"])
             for o in orgs_ev}
B_PROBE, TAU_PROBE = 1.0, 5.0
v5_counts = {}
for variant in ("delta2", "delta1"):
    out_ok = pre_ok = adj0_ok = True
    n_adj_rows = 0
    for org in orgs_ev:
        o1 = ov.run(base_v6, variant, B_PROBE, TAU_PROBE, team_filter=[org])
        inv = (win_f == org) | (los_f == org)
        out_ok &= eq_arr(o1["p_all"][~inv], base_v6["p_all"][~inv])
        adj0_ok &= bool((o1["adj"][~inv] == 0.0).all())
        pre = inv & (dates_f <= first_det[org])
        pre_ok &= eq_arr(o1["p_all"][pre], base_v6["p_all"][pre])
        n_adj_rows += int(o1["affected_rows"].size)
    check(f"V5 {variant}: per-team overlay touches NO row outside the team "
          f"(all {len(orgs_ev)} changed orgs)", True, out_ok)
    check(f"V5 {variant}: adj identically 0 outside the team", True, adj0_ok)
    check(f"V5 {variant}: exact pre-change identity — every row of the team "
          "dated <= its first detection is bit-identical", True, pre_ok)
    v5_counts[variant] = {"orgs_with_events": len(orgs_ev),
                          "affected_rows_sum_per_team_runs": n_adj_rows}
    # full overlay: every changed row has an active side; global pre-identity
    o_full = ov.run(base_v6, variant, B_PROBE, TAU_PROBE)
    ch = np.flatnonzero(neq_mask(o_full["p_all"], base_v6["p_all"]))
    act_ok = all(
        ov.active_state(win_f[r], dates_f[r], TAU_PROBE) is not None
        or ov.active_state(los_f[r], dates_f[r], TAU_PROBE) is not None
        for r in ch)
    check(f"V5 {variant} full overlay: every changed row has an active-phase "
          "side", True, act_ok)
    gmin = min(first_det.values())
    pre_g = dates_f <= gmin
    check(f"V5 {variant} full overlay: all rows dated <= first detection "
          f"anywhere ({gmin}) bit-identical", True,
          eq_arr(o_full["p_all"][pre_g], base_v6["p_all"][pre_g]))
    v5_counts[variant]["full_overlay_changed_rows"] = int(ch.size)
check("V5: base rdiff sha unchanged after all overlay work (no mutation, "
      "no re-solve)", sha_rd0, rd_checksum(base_v6["rdiff"]))
check("V5: base p_all sha unchanged (overlays copy, never write back)",
      sha_p0, rd_checksum(base_v6["p_all"]))
FIX["v5_counts"] = v5_counts

print("=== V6: the a<=6.0 law (no widening path) ===", flush=True)


def must_raise(name, fn):
    try:
        fn()
    except AssertionError:
        return check(name, "AssertionError", "AssertionError")
    _flush("FAILED")
    raise AssertionError(f"FIXTURE FAIL: {name}: no AssertionError raised")


ver_sen = plan5.final("SEN")
must_raise("V6 solve_side_run(a=6.000001) raises",
           lambda: solve_side_run(plan5, 6.000001, 5.0, 0.0))
must_raise("V6 solve_side_run(a=-0.1) raises",
           lambda: solve_side_run(plan5, -0.1, 5.0, 0.0))
must_raise("V6 multipliers(a=7) raises at the hook path",
           lambda: plan5.multipliers(ver_sen, 7.0, 5.0, 0.0))
must_raise("V6 multipliers(s=1.5) raises",
           lambda: plan5.multipliers(ver_sen, 2.0, 5.0, 1.5))
must_raise("V6 overlay b<0 raises",
           lambda: ov.run(base_v6, "delta2", -1.0, 5.0))
mu6 = plan5.multipliers(ver_sen, 6.0, 5.0, 0.0)
check("V6 a=6.0 exactly is accepted (cap inclusive), multipliers finite",
      True, bool(np.isfinite(mu6).all()))
src = open(os.path.join(HERE, "v9lib.py")).read()
check("V6 source: single A_CAP = 6.0 constant", 1, src.count("A_CAP = 6.0"))
check("V6 source: no environment/flag bypass of the cap (os.environ/"
      "getenv absent; prose mentions of 'environment' don't count)", True,
      "os.environ" not in src and "getenv" not in src)

print("=== V7: delta1 leak-proofness (future/same-day flips) ===", flush=True)
d1_full = ov.run(base_v6, "delta1", B_PROBE, TAU_PROBE)
by_org = defaultdict(list)
for e in d1_full["detail"]:
    by_org[e["org"]].append(e)
case = None
for org, ee in sorted(by_org.items()):
    ee = sorted(ee, key=lambda x: x["date"])
    if len(ee) < 2 or ee[0]["date"] >= ee[-1]["date"]:
        continue
    r1, rL = ee[0]["row"], ee[-1]["row"]
    D1, DL = ee[0]["date"], ee[-1]["date"]
    D3 = (_date.fromisoformat(DL) + timedelta(days=1)).isoformat()
    if ov.active_state(org, D3, TAU_PROBE) is not None:
        case = (org, r1, D1, rL, DL, D3)
        break
check_true("V7 found a case: org with 2+ delta1-affected rows at distinct "
           "dates and an active phase the day after", case is not None,
           str(case))
org, r1, D1, rL, DL, D3 = case
o_ref = ov.run(base_v6, "delta1", B_PROBE, TAU_PROBE, team_filter=[org])
wm2 = ov.w_maps.copy()
lm2 = ov.l_maps.copy()
wm2[rL] += 3.0                               # flip a FUTURE match's score
o_flip = ov.run(base_v6, "delta1", B_PROBE, TAU_PROBE, team_filter=[org],
                scores=(wm2, lm2))
check(f"V7 {org}: price of the earlier affected row ({D1}) bit-identical "
      f"after flipping the {DL} score", True,
      bool(o_ref["p_all"][r1] == o_flip["p_all"][r1]))
le_mask = dates_f <= DL
check(f"V7 {org}: EVERY row dated <= the flipped match is bit-identical "
      "(covers same-day: date-strict evidence)", True,
      eq_arr(o_ref["p_all"][le_mask], o_flip["p_all"][le_mask]))
E_a = ov.evidence(org, D3, base_v6, TAU_PROBE)[0]
E_b = ov.evidence(org, D3, base_v6, TAU_PROBE, scores=(wm2, lm2))[0]
check_true(f"V7 {org}: the flip IS visible to evidence priced after it "
           f"(E {E_a:.4f} -> {E_b:.4f})", E_a != E_b)
FIX["v7_case"] = {"org": org, "early_row_date": D1, "flipped_row_date": DL,
                  "probe_date": D3,
                  "note": "in-code assert d_i < D ran on every evidence row "
                          "of every delta1 overlay in this suite"}

_flush("PASSED")
print(f"ALL FIXTURES PASSED ({FIX['n_pass']} assertions) — "
      "stats/v9_fixtures.json written", flush=True)
