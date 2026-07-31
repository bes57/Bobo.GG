"""SPEC RUN library — causal sub-vs-change classification + weight modifiers.

Governing doc: briefs/roster_spec_operator.md; preregistered as ADDENDUM 4 in
preregister.roster.md (locked 2026-07-29 01:37 BEFORE any run).

All classification is walk-forward: M_{i,T} uses matches dated < T only
(date-strict; same-day matches are never in each other's knowledge set).
State as of solve day D is a pure function of the org's match prefix dated
< D — provisional P2 states included — so every solve day reproduces exactly
the state that was knowable at that day. Nothing here ever reads a match
dated >= D when serving day D.

Tie-break note (disclosed, ADDENDUM 4 + fixtures): modal ties break by most
recent occurrence (spec rule), EXCEPT a count-1 challenger never displaces an
incumbent mode still present in the window — this preserves the spec's own
definitional invariant that "a one-off sub cannot outvote a trailing window",
which pure recency would violate in the tiny-prefix 1-vs-1 case. Ties at
count >= 2 (e.g. W=8 at 4-4) break by recency exactly as the spec says.

Phase established-mode rule (detector-consistent): a CLOSED phase's
established five is the mode the next boundary displaced (its 'old'); the
OPEN phase's is the detector's current mode. Sub games therefore score o < 1
against the lineup that was established in their own phase, never against
the team's current five (spec §2.5 reference-point rule).
"""
import os
import pickle
from bisect import bisect_left, bisect_right
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load_corpus():
    with open(os.path.join(HERE, "corpus.pkl"), "rb") as f:
        return pickle.load(f)


def _mode(entries, prev_mode=None):
    """entries: list of (idx, lineup_frozenset). Max count; ties by most
    recent occurrence; a count-1 challenger cannot displace a present
    incumbent (module docstring)."""
    if not entries:
        return None
    cnt = defaultdict(int)
    last = {}
    for pos, (idx, lu) in enumerate(entries):
        cnt[lu] += 1
        last[lu] = pos
    cmax = max(cnt.values())
    cands = [lu for lu, c in cnt.items() if c == cmax]
    winner = max(cands, key=lambda lu: last[lu]) if len(cands) > 1 else cands[0]
    if (prev_mode is not None and winner != prev_mode and cmax == 1
            and cnt.get(prev_mode, 0) >= 1):
        return prev_mode  # one match never outvotes the incumbent
    return winner


def _ov5(lu, mode):
    """o = min(|L ∩ M|, 5)/5; missing lineup or undefined mode ⇒ 1."""
    if lu is None or mode is None:
        return 1.0
    return min(len(lu & mode), 5) / 5.0


def _k5(new_mode, old_mode):
    if new_mode is None or old_mode is None:
        return 5
    return min(len(new_mode & old_mode), 5)


class SpecPlan:
    """Precomputed per-org, per-date-version classifier state for one policy.

    policy: 'p1' (window W), 'p2' (horizon m; phase-mode based), 'p3'
    (window W; o frozen at game date; no boundaries, no boost).
    c: chain merge — consecutive boundaries <= c matches apart fold into the
    FIRST's anchor, k end-to-end (c=0 never merges).

    Version i of an org covers matches dated <= dkeys[i] and is the state
    for any solve day D with dkeys[i] < D <= dkeys[i+1].
    """

    def __init__(self, policy, W=5, c=3, m=3, corpus=None):
        self.policy, self.W, self.c, self.m = policy, W, c, m
        self.corpus = corpus if corpus is not None else load_corpus()
        seq = self.corpus["team_match_seq"]
        lus = self.corpus["lineups"]
        self.orgs = {}
        for org, sq in seq.items():
            matches = [(d, mid, lus.get((org, mid))) for d, mid in sq]
            self.orgs[org] = self._build_org(matches)

    def tag(self):
        if self.policy == "p1":
            return f"p1w{self.W}c{self.c}"
        if self.policy == "p2":
            return f"p2m{self.m}c{self.c}"
        return f"p3w{self.W}"

    # ── per-org construction ────────────────────────────────────────────────
    def _build_org(self, matches):
        dates = [mm[0] for mm in matches]
        dkeys = sorted(set(dates))
        o_frozen = self._p3_frozen_o(matches) if self.policy == "p3" else None
        events, mode_by_dkey = ([], []) if self.policy != "p1" else \
            self._p1_events(matches, dkeys)
        versions = []
        for vi, dk in enumerate(dkeys):
            nvis = bisect_right(dates, dk)
            if self.policy == "p1":
                versions.append(self._p1_version(matches, nvis, events,
                                                 mode_by_dkey[vi], dk))
            elif self.policy == "p2":
                versions.append(self._p2_version(matches, nvis))
            else:
                versions.append(self._p3_version(matches, nvis, o_frozen))
        return {"dates": dates, "dkeys": dkeys, "matches": matches,
                "versions": versions, "events": events,
                "final_boundaries": (versions[-1]["boundaries"] if versions
                                     else [])}

    # ---- P1 ----
    def _p1_events(self, matches, dkeys):
        """Date-granular mode-shift detections on the trailing-W window of
        lineup-known matches. Returns (events, mode_by_dkey)."""
        events, mode_by_dkey = [], []
        known = []          # (idx, lu), lineup-known prefix
        prev_mode = None
        di = 0
        for dk in dkeys:
            while di < len(matches) and matches[di][0] <= dk:
                if matches[di][2] is not None:
                    known.append((di, matches[di][2]))
                di += 1
            win = known[-self.W:]
            mode = _mode(win, prev_mode=prev_mode)
            if mode is not None and prev_mode is not None and mode != prev_mode:
                j = min(idx for idx, lu in win if lu == mode)
                events.append({"det_date": dk, "j": j, "old": prev_mode,
                               "new": mode, "k": _k5(mode, prev_mode)})
            if mode is not None:
                prev_mode = mode
            mode_by_dkey.append(mode)
        return events, mode_by_dkey

    def _merge(self, bounds):
        """Chain merge (ADDENDUM 4 A3). Distance to the previous boundary
        event, merged or not."""
        merged = []
        for b in bounds:
            if merged and self.c > 0 and b["j"] - merged[-1]["last_j"] <= self.c:
                a = merged[-1]
                a["last_j"] = b["j"]
                if b.get("new") is not None and a.get("old") is not None:
                    a["k"] = _k5(b["new"], a["old"])   # end-to-end
                else:
                    a["k"] = min(a["k"], b["k"])       # p2 fallback, disclosed
                a["new"] = b.get("new", a.get("new"))
                a["provisional"] = a.get("provisional", False) and \
                    b.get("provisional", False)
            else:
                merged.append(dict(b, last_j=b["j"]))
        return merged

    def _p1_version(self, matches, nvis, events, cur_mode, dk):
        vis = [e for e in events if e["det_date"] <= dk]
        bounds = self._merge([{"j": e["j"], "k": e["k"], "old": e["old"],
                               "new": e["new"], "provisional": False}
                              for e in vis])
        bounds = [b for b in bounds if b["j"] < nvis]
        phase_modes = [b["old"] for b in bounds] + [cur_mode]
        return self._assemble(matches, nvis, bounds, phase_modes,
                              pending=False, mode=cur_mode)

    # ---- P2 ----
    def _p2_version(self, matches, nvis):
        """Re-run the provisional/retraction state machine on the prefix —
        the state knowable from matches dated < D, provisional included."""
        bounds = []            # {'j','k','provisional','M_old'}
        phase_start = 0
        phase_known = []       # (idx, lu) in current confirmed phase
        pending = None
        for t in range(nvis):
            d, mid, lu = matches[t]
            if lu is None:
                if pending is not None:
                    pending["count"] += 1
                    if pending["count"] >= self.m:
                        bounds.append({"j": pending["t"], "k": pending["k"],
                                       "provisional": False,
                                       "M_old": pending["M_old"]})
                        phase_start = pending["t"]
                        phase_known = [(i, matches[i][2])
                                       for i in range(phase_start, t + 1)
                                       if matches[i][2] is not None]
                        pending = None
                continue
            M = _mode(phase_known) if phase_known else None
            if pending is None:
                if M is not None and _ov5(lu, M) < 1.0:
                    pending = {"t": t, "k": min(len(lu & M), 5), "M_old": M,
                               "count": 0, "lu": lu}
                else:
                    phase_known.append((t, lu))
            else:
                if _ov5(lu, pending["M_old"]) >= 1.0:
                    phase_known.append((pending["t"], pending["lu"]))
                    phase_known.append((t, lu))
                    phase_known.sort()
                    pending = None          # retracted; dev stays a sub game
                else:
                    pending["count"] += 1
                    if pending["count"] >= self.m:
                        bounds.append({"j": pending["t"], "k": pending["k"],
                                       "provisional": False,
                                       "M_old": pending["M_old"]})
                        phase_start = pending["t"]
                        phase_known = [(i, matches[i][2])
                                       for i in range(phase_start, t + 1)
                                       if matches[i][2] is not None]
                        pending = None
        open_pending = pending is not None
        if open_pending:
            bounds.append({"j": pending["t"], "k": pending["k"],
                           "provisional": True, "M_old": pending["M_old"]})
        # phase established modes: closed phase -> M_old of the boundary that
        # ended it; open phase -> mode over its own known matches
        bounds = self._merge([{"j": b["j"], "k": b["k"], "old": b["M_old"],
                               "new": None, "provisional": b["provisional"]}
                              for b in bounds])
        starts = [0] + [b["j"] for b in bounds]
        phase_modes = [b["old"] for b in bounds]
        seg = [(i, matches[i][2]) for i in range(starts[-1], nvis)
               if matches[i][2] is not None]
        phase_modes.append(_mode(seg) if seg else None)
        return self._assemble(matches, nvis, bounds, phase_modes,
                              pending=open_pending,
                              mode=phase_modes[-1])

    # ---- P3 ----
    def _p3_frozen_o(self, matches):
        n = len(matches)
        o = np.ones(n)
        known = [(i, lu) for i, (d, m, lu) in enumerate(matches)
                 if lu is not None]
        pos_dates = {i: d for i, (d, m, lu) in enumerate(matches)}
        for i, (d, mid, lu) in enumerate(matches):
            if lu is None:
                continue
            win = [e for e in known if pos_dates[e[0]] < d][-self.W:]
            o[i] = _ov5(lu, _mode(win))
        return o

    def _p3_version(self, matches, nvis, o_frozen):
        known = [(i, lu) for i, (d, m, lu) in enumerate(matches[:nvis])
                 if lu is not None]
        return {"nvis": nvis, "o": o_frozen[:nvis].copy(),
                "n": np.full(nvis, -1, dtype=np.int32),
                "k": np.full(nvis, 5, dtype=np.int32),
                "phase_size": np.zeros(nvis, dtype=np.int32),
                "boundaries": [], "pending": False,
                "mode": _mode(known[-self.W:]) if known else None}

    # ---- shared assembly ----
    @staticmethod
    def _assemble(matches, nvis, bounds, phase_modes, pending, mode):
        o = np.ones(nvis)
        nn = np.full(nvis, -1, dtype=np.int32)
        kk = np.full(nvis, 5, dtype=np.int32)
        psize = np.zeros(nvis, dtype=np.int32)
        starts = [0] + [b["j"] for b in bounds]
        ends = [b["j"] for b in bounds] + [nvis]
        for pi, (a, bnd) in enumerate(zip(starts, ends)):
            bnd = min(bnd, nvis)
            if a >= bnd:
                continue
            pm = phase_modes[pi]
            for i in range(a, bnd):
                o[i] = _ov5(matches[i][2], pm)
                if pi >= 1:
                    nn[i] = i - a
                    kk[i] = bounds[pi - 1]["k"]
                psize[i] = bnd - a
        return {"nvis": nvis, "o": o, "n": nn, "k": kk, "phase_size": psize,
                "boundaries": [{"j": b["j"], "date": matches[b["j"]][0],
                                "k": int(b["k"]),
                                "provisional": bool(b["provisional"])}
                               for b in bounds if b["j"] < nvis],
                "pending": bool(pending), "mode": mode}

    # ── public as-of API ────────────────────────────────────────────────────
    def version_asof(self, org, D):
        """State knowable at solve day D (matches dated < D)."""
        oo = self.orgs.get(org)
        if oo is None:
            return None
        i = bisect_left(oo["dkeys"], D)     # play-dates strictly < D
        if i == 0:
            return None
        return oo["versions"][i - 1]

    def final(self, org):
        oo = self.orgs.get(org)
        return oo["versions"][-1] if oo and oo["versions"] else None

    def multipliers(self, ver, a, tau, s, n_min=3, cap=None, boost_only=False,
                    min_boundary_date=None):
        """Per-visible-match weight multiplier vector (len ver['nvis']).
        boost 1 + a(1-k/5)e^(-n/tau) on post-boundary games (per-game n);
        floor: phases with < n_min visible matches capped at `cap`;
        sub down-weight x[1 - s(1-o)] (skipped when boost_only).
        min_boundary_date: ablation windowing (A5) — only boundaries dated >=
        it activate."""
        nv = ver["nvis"]
        mult = np.ones(nv)
        if a > 0 and ver["boundaries"]:
            act = ver["n"] >= 0
            if min_boundary_date is not None:
                keep = np.zeros(nv, dtype=bool)
                bs = ver["boundaries"]
                for bi, b in enumerate(bs):
                    if b["date"] >= min_boundary_date:
                        nxt = bs[bi + 1]["j"] if bi + 1 < len(bs) else nv
                        keep[b["j"]:nxt] = True
                act = act & keep
            if act.any():
                boost = 1.0 + a * (1.0 - kkf(ver["k"][act]) / 5.0) * \
                    np.exp(-ver["n"][act] / tau)
                if cap is not None:
                    thin = ver["phase_size"][act] < n_min
                    boost = np.where(thin, np.minimum(boost, cap), boost)
                mult[act] = boost
        if s > 0 and not boost_only:
            mult = mult * (1.0 - s * (1.0 - ver["o"]))
        return mult


def kkf(k):
    return k.astype(float) if hasattr(k, "astype") else float(k)
