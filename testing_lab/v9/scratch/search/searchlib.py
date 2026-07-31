"""v9 SEARCH library — generalized overlay + transfer evaluator helpers.

Governed by v9/preregister.search.md (LOCKED before this file was written)
and stats/v9_transfer_protocol.json (executed verbatim). Extension of the
frozen family mechanics (scratch/family/v9lib.py): magnitude generalized to
g = b*(1-k/5)**gamma*exp(-n/tau) and direction generalized by (variant, m)
per preregister section 1a. Overlay.evidence and Overlay.active_state are
REUSED VERBATIM — the date-strict leak assert (d_i < D) executes on every
evidence row of every state build; no reimplementation of either exists
here. Fixture S1 proves (gamma=1, m=0) nests the frozen family bitwise.

Transfer-evaluator helpers (fit_beta / judged / fragility / mde_milli) are
copied with attribution from v9/scratch/design/autopsy.py so candidate
numbers are methodologically identical to the gate autopsy's.
"""
import hashlib
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize_scalar

TL = "/Users/benny_es1/PythonTest/testing_lab"
V8 = os.path.join(TL, "v8")
V9 = os.path.join(TL, "v9")
FAM = os.path.join(V9, "scratch", "family")
for _p in (TL, V8, FAM):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import referee  # noqa: E402
import v9lib  # noqa: E402
from v9lib import Overlay, build_plan, series_from_pm, horizon  # noqa: E402
import runner  # noqa: E402

PROTOCOL = json.load(open(os.path.join(V9, "stats",
                                       "v9_transfer_protocol.json")))

# ── frame + windows (protocol verbatim; sha asserted twice) ────────────────

def load_frame_checked():
    """runner.load_frame() asserts sha vs crn.json; re-assert vs the
    protocol blob too (preregister section 3)."""
    frame = runner.load_frame()
    sha = hashlib.sha256(
        open(runner.FRAME_PATH, "rb").read()).hexdigest()
    assert sha == PROTOCOL["frame"]["series_csv_sha256"], \
        f"FRAME SHA vs protocol MISMATCH {sha}"
    return frame


def windows(frame):
    d = frame.date.values
    W = {"FIT1": (d <= "2024-12-31"),
         "VAL1": (d > "2024-12-31") & (d <= "2025-12-31"),
         "FIT2": (d <= "2025-12-31"),
         "VAL2": (d > "2025-12-31") & (d <= "2026-07-28")}
    assert int(W["FIT1"].sum()) == PROTOCOL["windows"]["FIT1"]["n"]
    assert int(W["VAL1"].sum()) == PROTOCOL["windows"]["VAL1"]["n"]
    assert int(W["FIT2"].sum()) == PROTOCOL["windows"]["FIT2"]["n"]
    assert int(W["VAL2"].sum()) == PROTOCOL["windows"]["VAL2"]["n"]
    return W


# ── generalized overlay (search extension; preregister 1a) ─────────────────

VARIANTS = ("delta2", "delta1", "hybrid")


def direction_of(variant, m, E, ne):
    """Frozen direction rules. delta1(0) == the family's pure delta1."""
    if variant == "delta2":
        return 1.0
    if variant == "delta1":
        return float(np.sign(E)) if ne >= m else 0.0
    if variant == "hybrid":
        return 1.0 if ne < m else float(np.sign(E))
    raise AssertionError(f"unknown variant {variant!r}")


class OverlaySearch(Overlay):
    """Overlay with the generalized magnitude/direction. run_general mirrors
    the frozen Overlay.run loop (copied with attribution from v9lib.py) —
    active_state and evidence are the PARENT'S, untouched."""

    def state_table(self, base, tau):
        """Per-(org,row) active state via the REAL code path: one
        active_state + one evidence call per pricing (leak assert hot).
        Returns list of (row, is_w, n, k, E, ne)."""
        out = []
        for org in self.plan.orgs.keys():
            rows = self.org_rows.get(org)
            if not rows:
                continue
            for mid, (r, is_w) in rows.items():
                if not base["valid"][r]:
                    continue
                D = self.dates[r]
                st = self.active_state(org, D, tau)
                if st is None:
                    continue
                E, ne = self.evidence(org, D, base, tau)   # assert d_i < D hot
                out.append((int(r), bool(is_w), int(st["n"]), int(st["k"]),
                            float(E), int(ne)))
        return out

    def adj_from_state(self, state, b, tau, gamma, m, variant, n_rows):
        assert np.isfinite(b) and b >= 0.0, f"b={b!r} must be >= 0"
        assert np.isfinite(gamma) and gamma > 0.0
        assert variant in VARIANTS
        adj = np.zeros(n_rows)
        for r, is_w, n, k, E, ne in state:
            g = b * (1.0 - k / 5.0) ** gamma * float(np.exp(-n / tau))
            if g == 0.0:
                continue
            d = g * direction_of(variant, m, E, ne)
            if d == 0.0:
                continue
            adj[r] += d if is_w else -d
        return adj

    def run_general(self, base, variant, b, tau, gamma=1.0, m=0):
        """Full un-cached path (used by every LEDGERED evaluation and the
        ladder): the same per-(org,row) loop as the frozen Overlay.run,
        with the generalized g and direction."""
        assert variant in VARIANTS
        assert np.isfinite(b) and b >= 0.0, f"b={b!r} must be >= 0"
        assert np.isfinite(tau) and tau > 0.0
        assert np.isfinite(gamma) and gamma > 0.0
        n_rows = len(self.frame)
        assert len(base["p_all"]) == n_rows and len(base["rdiff"]) == n_rows
        adj = np.zeros(n_rows)
        for org in self.plan.orgs.keys():
            rows = self.org_rows.get(org)
            if not rows:
                continue
            for mid, (r, is_w) in rows.items():
                if not base["valid"][r]:
                    continue
                D = self.dates[r]
                st = self.active_state(org, D, tau)
                if st is None:
                    continue
                g = b * (1.0 - st["k"] / 5.0) ** gamma \
                    * float(np.exp(-st["n"] / tau))
                if g == 0.0:
                    continue
                E, ne = self.evidence(org, D, base, tau)   # leak assert hot
                d = g * direction_of(variant, m, E, ne)
                if d == 0.0:
                    continue
                adj[r] += d if is_w else -d
        affected = np.flatnonzero(adj)
        p_all = base["p_all"].copy()
        if affected.size:
            lg = base["beta"] * base["rdiff"][affected] + adj[affected]
            pm = 1.0 / (1.0 + np.exp(-lg))
            p_all[affected] = series_from_pm(pm, self.fmts[affected])
        return {"p_all": p_all, "adj": adj, "affected_rows": affected,
                "variant": variant, "b": float(b), "tau": float(tau),
                "gamma": float(gamma), "m": int(m)}


def p_from_adj(base, adj, beta, fmts):
    """Candidate p under a window-refit beta: closed form of
    sigmoid(beta*rdiff + adj) on adj!=0 rows, plain closed form elsewhere.
    (adj is frozen mechanics; beta governs the base map logit — preregister
    section 3.)"""
    valid = base["valid"]
    rd = base["rdiff"]
    p = np.full(len(rd), np.nan)
    p[valid] = runner.p_series_closed(beta, rd[valid], fmts[valid])
    aff = np.flatnonzero(adj)
    if aff.size:
        assert valid[aff].all()
        lg = beta * rd[aff] + adj[aff]
        pm = 1.0 / (1.0 + np.exp(-lg))
        p[aff] = series_from_pm(pm, fmts[aff])
    return p


# ── transfer-evaluator helpers (attribution: v9/scratch/design/autopsy.py) ──

SIG_W = 0.02207
ZPOW = 2.8016


def mde_milli(n):
    return round(ZPOW * SIG_W / np.sqrt(n) * 1000, 3)


def fit_beta(rdiff, mask, fmts, valid):
    m = mask & valid

    def nll(b):
        p = runner.p_series_closed(b, rdiff[m], fmts[m])
        return -np.mean(np.log(np.clip(p, 1e-9, 1.0)))

    r = minimize_scalar(nll, bounds=(0.001, 1.0), method="bounded",
                        options={"xatol": 1e-6})
    return float(r.x)


def judged(d, ev):
    iid = referee.paired_bootstrap_crn(d, mode="iid")
    blk = referee.paired_bootstrap_crn(d, mode="block_event", event_ids=ev)
    se_blk = (blk["ci_hi"] - blk["ci_lo"]) / 3.92
    return {"delta_milli": round(float(d.mean()) * 1000, 3),
            "n": int(len(d)),
            "iid_ci_milli": [round(iid["ci_lo"] * 1000, 2),
                             round(iid["ci_hi"] * 1000, 2)],
            "blk_ci_milli": [round(blk["ci_lo"] * 1000, 2),
                             round(blk["ci_hi"] * 1000, 2)],
            "p_better_iid": iid["p_better"], "p_better_blk": blk["p_better"],
            "se_blk_milli": round(se_blk * 1000, 3)}


def fragility(d, ev):
    n = len(d)
    k = int(np.ceil(0.05 * n))
    keep = np.argsort(d)[: n - k]          # drop the k largest contributions
    drop5 = float(d[keep].mean())
    jk = {}
    for e in dict.fromkeys(ev):
        m = ev != e
        jk[str(e)] = round(float(d[m].mean()) * 1000, 3)
    worst_e = min(jk, key=jk.get)
    return {"drop_top5_milli": round(drop5 * 1000, 3), "n_dropped": k,
            "jackknife_min_milli": jk[worst_e], "jackknife_min_event": worst_e,
            "jackknife_all": jk}


def transfer_eval(rd_cand, adj, rd_v6, frame, W, valid, label,
                  beta_v6_fit1, beta_v6_fit2):
    """ONE protocol transfer evaluation. rd_cand: candidate rdiff (solve
    side) or v6's rdiff (overlay). adj: map-logit adjustment vector (zeros
    for solve-side candidates). Returns the full clause-by-clause blob."""
    fmts = frame.fmt.values
    event_ids = frame.event_id.values
    r = {"label": label}
    base_like = {"rdiff": rd_cand, "valid": valid}

    def probs_plain(rd, beta):
        p = np.full(len(frame), np.nan)
        p[valid] = runner.p_series_closed(beta, rd[valid], fmts[valid])
        return p

    def probs_cand(beta):
        if adj is None:
            return probs_plain(rd_cand, beta)
        return p_from_adj(base_like, adj, beta, fmts)

    out_win = {}
    for tkey, fitw, valw, bv6 in (("T1_fit2324_val2025", "FIT1", "VAL1",
                                   beta_v6_fit1),
                                  ("T2_fit2325_val2026H1", "FIT2", "VAL2",
                                   beta_v6_fit2)):
        bc = fit_beta(rd_cand, W[fitw], fmts, valid)
        mv = W[valw] & valid
        pc = probs_cand(bc)[mv]
        pv = probs_plain(rd_v6, bv6)[mv]
        d = referee.delta_vector(pc, pv)
        r[tkey] = dict(judged(d, event_ids[mv]),
                       beta_cand=round(bc, 6), beta_v6=round(bv6, 6),
                       mde_milli=mde_milli(int(mv.sum())),
                       diagnostics=fragility(d, event_ids[mv]))
        out_win[valw] = (d, event_ids[mv], pv, pc, mv)
    d1, ev1, pv1, pc1, m1 = out_win["VAL1"]
    d2, ev2, pv2, pc2, m2 = out_win["VAL2"]
    dp = np.concatenate([d1, d2])
    evp = np.concatenate([ev1, ev2])
    r["pooled_validation"] = dict(judged(dp, evp), mde_milli=mde_milli(len(dp)),
                                  note="concat of era-legal window d vectors")
    frag = fragility(dp, evp)
    r["pooled_fragility"] = frag
    a1 = r["T1_fit2324_val2025"]["delta_milli"] >= \
        1.0 * r["T1_fit2324_val2025"]["se_blk_milli"]
    a2 = r["T2_fit2325_val2026H1"]["delta_milli"] >= \
        1.0 * r["T2_fit2325_val2026H1"]["se_blk_milli"]
    a3 = r["pooled_validation"]["blk_ci_milli"][0] > 0
    a4 = r["pooled_validation"]["delta_milli"] >= 1.773
    a5 = bool(frag["drop_top5_milli"] > 0 and frag["jackknife_min_milli"] > 0) \
        if (a1 and a2 and a3 and a4) else None
    r["advance_rule"] = {"A1_val1_ge_1se": bool(a1), "A2_val2_ge_1se": bool(a2),
                         "A3_pooled_ci_gt0": bool(a3),
                         "A4_pooled_ge_mde": bool(a4), "A5_fragility": a5,
                         "ADVANCE": bool(a1 and a2 and a3 and a4 and (a5 or False))}
    roi = referee.expected_roi_of_dll(
        r["pooled_validation"]["delta_milli"] / 1000.0,
        np.concatenate([pv1, pv2]))
    r["pooled_roi_translation"] = {
        "expected_roi_delta": roi["expected_roi_delta"],
        "delta_logit_equiv": roi["delta_logit_equiv"],
        "ladder_source": roi["ladder_source"]}
    # ladder-order key input (protocol survivors clause): VAL2 prob distance
    r["mean_abs_p_gap_val2"] = round(float(np.mean(np.abs(pc2 - pv2))), 5)
    return r
