"""v9 FAMILY library — solve-side (hard-capped) + prediction-layer + hybrid.

Governing docs: testing_lab/v9/README.md (three laws), v9/briefs/family.md,
v9/preregister.family.md (LOCKED 2026-07-29 11:32; ADDENDUM 1 11:40 — both
BEFORE this file existed). Mechanics only: nothing in this module computes
a log-loss, Brier, or any aggregate over outcomes, train or holdout.

Reused machinery (imported, NOT reimplemented — brief requirement):
  speclib.SpecPlan    causal walk-forward classifier (P1/P3): date-strict
                      version_asof (matches dated < D only), chain merge,
                      per-game o/n/k, multipliers
  runner.run_config   EngineSpec per-side weight hook; frame sha verified
                      against crn.json; holdout metrics popped UNSEEN
  runner.v6_cfg / get_engine / p_series_closed / rd_checksum / daily_checksum

THE a <= 6.0 LAW (v9 README; the a=28 autopsy): asserted unconditionally in
SpecPlanV9.multipliers AND in solve_side_run. There is deliberately NO flag,
kwarg, or environment switch that widens it, and none may be added.

Prediction-layer member: the solve stays PURE v6 for every team (the base
run's rdiff is never modified). A changed team's map-level series logit
gets delta(n, k) = b*(1-k/5)*exp(-n/tau) before the closed-form series
step — delta2 unconditionally toward the changed team (atlas prior),
delta1 times sign(E(T, D)) with the FROZEN walk-forward evidence E of
preregister section 2. Zero cross-team coupling BY CONSTRUCTION: the
adjustment is written only into rows the changed team plays, ratings are
never re-solved, and each side's delta depends only on its own state.
Active-phase horizon (ADDENDUM 1a): a phase prices only while
n <= ceil(5*tau); beyond that delta = 0 exactly.
"""
import os
import sys
from collections import defaultdict

import numpy as np

TL = "/Users/benny_es1/PythonTest/testing_lab"
V8 = os.path.join(TL, "v8")
SPEC = os.path.join(V8, "scratch", "roster", "spec_run")
for _p in (TL, V8, SPEC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from speclib import SpecPlan, load_corpus  # noqa: E402,F401  (re-exported)
import runner  # noqa: E402
from runner import rd_checksum, daily_checksum  # noqa: E402,F401 (re-export)

A_CAP = 6.0     # THE LAW. No widening path exists anywhere in v9 code.


def _assert_a(a):
    assert np.isfinite(a) and 0.0 <= a <= A_CAP, (
        f"v9 LAW VIOLATION: a={a!r} outside [0, {A_CAP}]. The a<=6.0 cap is "
        "a law with no widening path (v9 README; a=28 autopsy).")


def _assert_s(s):
    assert np.isfinite(s) and 0.0 <= s <= 1.0, f"s={s!r} outside [0, 1]"


class SpecPlanV9(SpecPlan):
    """SpecPlan with the v9 hard cap enforced AT THE MULTIPLIER, so every
    v9 weight path — the engine hook included — passes the assert."""

    def multipliers(self, ver, a, tau, s, n_min=3, cap=None, boost_only=False,
                    min_boundary_date=None):
        _assert_a(a)
        _assert_s(s)
        return SpecPlan.multipliers(self, ver, a, tau, s, n_min=n_min,
                                    cap=cap, boost_only=boost_only,
                                    min_boundary_date=min_boundary_date)


def build_plan(policy="p1", W=5, c=5, corpus=None):
    """Family classifier. c=5 is fixed by the family definition; P2 is NOT
    RUN (v8 spec-run P2_DECISION: do not run it wrong)."""
    assert policy in ("p1", "p3"), \
        "P2 is NOT RUN (v8 spec_run P2_DECISION); family uses P1/P3 only"
    assert c == 5, "family definition fixes chain merge c=5"
    return SpecPlanV9(policy, W=W, c=c, corpus=corpus)


# ── solve-side member ───────────────────────────────────────────────────────

def v6_run(daily=False):
    """Pure v6 base run (plan None). Holdout metrics are popped unseen
    inside runner.run_config."""
    return runner.run_config(None, 0.0, 5.0, 0.0, daily=daily)


def solve_side_run(plan, a, tau, s, n_min=3, cap=None, daily=False,
                   team_filter=None):
    """v6 + capped solve-side subsystem, one full walk-forward run.
    Params (a, tau, s, n_min, cap); boost 1 + a(1-k/5)e^(-n/tau), thin-phase
    floor, sub down-weight x[1 - s(1-o)] — all via the reused spec-run
    machinery. The a<=6.0 law is asserted here AND inside the plan's own
    multipliers (engine-hook path)."""
    _assert_a(a)
    _assert_s(s)
    assert isinstance(plan, SpecPlanV9), \
        "v9 solve path requires SpecPlanV9 (capped multipliers)"
    assert np.isfinite(tau) and tau > 0.0
    return runner.run_config(plan, a, tau, s, n_min=n_min, cap=cap,
                             daily=daily, team_filter=team_filter)


# ── prediction-layer member ────────────────────────────────────────────────

def series_from_pm(pm, fm):
    """Closed-form series prob from a map prob. Shapes copied verbatim from
    runner.p_series_closed (attribution) with pm supplied directly, so the
    unadjusted path is bit-identical to the base run's series step."""
    return np.where(np.isin(fm, ("bo5", "bo5_gf")),
                    pm ** 3 * (1 + 3 * (1 - pm) + 6 * (1 - pm) ** 2),
                    np.where(fm == "bo1", pm, pm ** 2 * (3 - 2 * pm)))


def horizon(tau):
    """Active-phase pricing horizon N_max = ceil(5*tau) (ADDENDUM 1a) —
    derived from tau, never an independently tunable parameter."""
    return int(np.ceil(5.0 * tau))


class Overlay:
    """Prediction-layer member over a fixed base run.

    The base run's solve is untouched (rdiff/beta pass through). For a team
    T with an active phase at pricing date D — version_asof(T, D) has a
    confirmed boundary with n = nvis - j_last <= horizon(tau) — the
    map-level logit of T's rows gets +delta on T's side:
        l' = beta*rdiff + delta_Wside - delta_Lside,  pm' = sigmoid(l'),
    then the unchanged closed-form series step. Only affected row indices
    are ever written: rows without an active side are the base's own values
    (zero coupling / pre-change identity BY CONSTRUCTION, asserted in
    fixtures V5)."""

    def __init__(self, plan, frame=None):
        assert isinstance(plan, SpecPlanV9)
        self.plan = plan
        if frame is None:
            _, frame = runner.get_engine()
        self.frame = frame
        self.dates = frame.date.astype(str).values
        self.fmts = frame.fmt.values
        self.w_maps = frame.w_maps.values.astype(float)
        self.l_maps = frame.l_maps.values.astype(float)
        self.org_rows = defaultdict(dict)   # org -> {mid: (row, is_winner)}
        for r, (mid, w, l) in enumerate(zip(frame.match_id.values,
                                            frame.winner.values,
                                            frame.loser.values)):
            self.org_rows[w][int(mid)] = (r, True)
            self.org_rows[l][int(mid)] = (r, False)

    # ── state ──
    def active_state(self, org, D, tau):
        """Phase state knowable strictly before D (date-strict), or None.
        Active iff a confirmed boundary is visible and n <= horizon(tau)."""
        v = self.plan.version_asof(org, D)
        if v is None or not v["boundaries"]:
            return None
        b = v["boundaries"][-1]
        n = v["nvis"] - b["j"]
        assert n >= 1
        if n > horizon(tau):
            return None                      # stale phase: delta == 0 exactly
        return {"v": v, "j": b["j"], "k": int(b["k"]), "n": int(n)}

    def evidence(self, org, D, base, tau, scores=None):
        """FROZEN delta1 evidence E(T, D) — preregister section 2.
        E = sum over the current phase's matches knowable at D of
        [maps_T - played * p_hat_T], p_hat_T = sigmoid(beta_ref*rdiff)
        oriented to T, everything walk-forward. Returns (E, n_rows).
        Leak guard: every contributing match is ASSERTED dated < D.
        scores: (w_maps, l_maps) override — fixture support for the leak
        test ONLY; never a fitting input."""
        st = self.active_state(org, D, tau)
        if st is None:
            return 0.0, 0
        wm, lm = (self.w_maps, self.l_maps) if scores is None else scores
        mm = self.plan.orgs[org]["matches"]
        beta, rd, valid = base["beta"], base["rdiff"], base["valid"]
        rows = self.org_rows.get(org, {})
        E, ne = 0.0, 0
        for i in range(st["j"], st["v"]["nvis"]):
            d_i, mid_i, _lu = mm[i]
            assert d_i < D, f"LEAK: evidence match {mid_i} dated {d_i} >= {D}"
            hit = rows.get(int(mid_i))
            if hit is None:
                continue                    # corpus match absent from frame
            r, is_w = hit
            if not valid[r]:
                continue                    # no v6 expectation for this row
            pm = 1.0 / (1.0 + np.exp(-(beta * rd[r])))
            p_t = pm if is_w else 1.0 - pm
            maps_t = wm[r] if is_w else lm[r]
            E += maps_t - (wm[r] + lm[r]) * p_t
            ne += 1
        return float(E), ne

    # ── the member ──
    def run(self, base, variant, b, tau, team_filter=None, scores=None):
        """Overlay the base run. variant: 'delta1' (evidence-amplifier,
        direction = sign(E), sign(0)=0 => no adjustment) or 'delta2'
        (atlas intercept, unconditionally positive toward the changed
        team). Returns p_all (copy of base with ONLY affected rows
        recomputed), adj in map-logit units, affected_rows, detail."""
        assert variant in ("delta1", "delta2")
        assert np.isfinite(b) and b >= 0.0, f"b={b!r} must be >= 0"
        assert np.isfinite(tau) and tau > 0.0
        n_rows = len(self.frame)
        assert len(base["p_all"]) == n_rows and len(base["rdiff"]) == n_rows
        adj = np.zeros(n_rows)
        detail = []
        orgs = (list(team_filter) if team_filter is not None
                else list(self.plan.orgs.keys()))
        for org in orgs:
            rows = self.org_rows.get(org)
            if not rows or org not in self.plan.orgs:
                continue
            for mid, (r, is_w) in rows.items():
                if not base["valid"][r]:
                    continue                # no base price to adjust
                D = self.dates[r]
                st = self.active_state(org, D, tau)
                if st is None:
                    continue
                g = b * (1.0 - st["k"] / 5.0) * float(np.exp(-st["n"] / tau))
                if g == 0.0:
                    continue
                if variant == "delta2":
                    direction = 1.0
                else:
                    E, _ne = self.evidence(org, D, base, tau, scores=scores)
                    direction = float(np.sign(E))
                    if direction == 0.0:
                        continue            # no evidence -> no adjustment
                d = g * direction
                adj[r] += d if is_w else -d
                detail.append({"org": org, "row": int(r), "date": D,
                               "n": st["n"], "k": st["k"],
                               "delta_own_side": float(d)})
        affected = np.flatnonzero(adj)
        p_all = base["p_all"].copy()
        if affected.size:
            lg = base["beta"] * base["rdiff"][affected] + adj[affected]
            pm = 1.0 / (1.0 + np.exp(-lg))
            p_all[affected] = series_from_pm(pm, self.fmts[affected])
        return {"p_all": p_all, "adj": adj, "affected_rows": affected,
                "detail": detail, "rdiff": base["rdiff"],
                "beta": base["beta"], "variant": variant,
                "b": float(b), "tau": float(tau)}


# ── hybrid hook ────────────────────────────────────────────────────────────

def hybrid_run(plan, a, tau_solve, s, b, tau_delta, variant="delta2",
               n_min=3, cap=None, daily=False, overlay=None):
    """HYBRID: solve-side with small a (the a<=6.0 law applies unchanged)
    plus the delta overlay computed on the hybrid base's OWN rdiff/beta.
    a=s=b=0 nests to pure v6 exactly (fixture V1)."""
    base = solve_side_run(plan, a, tau_solve, s, n_min=n_min, cap=cap,
                          daily=daily)
    if b == 0.0:
        return {"base": base, "overlay": None, "p_all": base["p_all"]}
    ov = overlay if overlay is not None else Overlay(plan)
    out = ov.run(base, variant, b, tau_delta)
    return {"base": base, "overlay": out, "p_all": out["p_all"]}
