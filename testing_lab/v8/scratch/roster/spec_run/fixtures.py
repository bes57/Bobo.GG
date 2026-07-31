"""SPEC RUN fixtures F1-F4 — hard assertions BEFORE any train fitting.

Every check appends {name, expected, got, pass} to the assertion ledger;
ANY failure writes stats/roster_spec_fixtures.json with status=FAILED and
raises. Numbers produced after a failed fixture are void (ADDENDUM 4).
No holdout metric is computed anywhere here (ratings only).
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V8 = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
STATS = os.path.join(V8, "stats")
sys.path.insert(0, HERE)

from speclib import SpecPlan, _mode, _ov5, load_corpus  # noqa: E402

LEDGER = []
FIX = {"written_by": "agent:roster-g (SPEC RUN)",
       "preregistered": "preregister.roster.md ADDENDUM 4 (2026-07-29 01:37)",
       "order": "fixtures ran BEFORE any train fitting (logs/roster.log)",
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
    with open(os.path.join(STATS, "roster_spec_fixtures.json"), "w") as f:
        json.dump(FIX, f, indent=1)


# ════════════════════════════════════════════════════════════════════════════
# Synthetic mini-corpus (F2/F3): 16 teams, weekly rounds, deterministic.
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
        # circle-method pairings so everyone plays every round
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
    """Standalone walk-forward Massey using the SAME SpecPlan.multipliers
    code path as the real engine hook. Returns {day: ratings vector}."""
    teams = corpus["teams"]
    tidx = {t: i for i, t in enumerate(teams)}
    games = corpus["games"]
    seq = corpus["team_match_seq"]
    pos_of = {org: {m: i for i, (d, m) in enumerate(sq)}
              for org, sq in seq.items()}
    days = sorted({g["date_s"] for g in games})
    days = days[1:] + ["2024-12-31"]           # predict-from-history days
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


print("=== F2: synthetic revert ===", flush=True)
corp2, A2, N2 = mini_corpus(change_round=5, sustained=False)
dev_date, revert_date = "2024-02-08", "2024-02-15"   # rounds 5 and 6
for W in (3, 5, 8):
    pl = SpecPlan("p1", W=W, c=3, corpus=corp2)
    fb = pl.final("T00")["boundaries"]
    check(f"F2 p1w{W}: zero boundaries", [], fb)
pl = SpecPlan("p3", W=5, corpus=corp2)
check("F2 p3w5: frozen o flags exactly the one sub game",
      1, int((pl.final("T00")["o"] < 1).sum()))
p2 = SpecPlan("p2", m=3, c=3, corpus=corp2)
fin2 = p2.final("T00")
check("F2 p2: zero confirmed/open boundaries at corpus end", [],
      fin2["boundaries"])
mid_ver = p2.version_asof("T00", "2024-02-10")       # dev seen, revert not
check("F2 p2: provisional ACTIVE between dev and revert (knowable state)",
      [(5, True)], [(b["j"], b["provisional"]) for b in mid_ver["boundaries"]])
# solve level: P1-W5 boost-on equals boost-off EVERYWHERE (no boundary)
pl15 = SpecPlan("p1", W=5, c=3, corpus=corp2)
sa = mini_solve(corp2, pl15, a=2.0, tau=5.0, s=0.0)
sb = mini_solve(corp2, None, a=0.0, tau=5.0, s=0.0)
check("F2 p1w5 mini-solve: max|dR| before/during/after == 0.0 exactly",
      0.0, max_abs_diff(sa, sb))
# P2: identical outside the provisional window; different inside (by design)
sa2 = mini_solve(corp2, p2, a=2.0, tau=5.0, s=0.0)
check("F2 p2 mini-solve: identical for D <= dev_date", 0.0,
      max_abs_diff(sa2, sb, day_filter=lambda D: D <= dev_date))
check("F2 p2 mini-solve: FINAL timeline identical (D > revert_date)", 0.0,
      max_abs_diff(sa2, sb, day_filter=lambda D: D > revert_date))
check_true("F2 p2 mini-solve: provisional window shows the designed effect",
           max_abs_diff(sa2, sb,
                        day_filter=lambda D: dev_date < D <= revert_date) > 0)

print("=== F3: synthetic sustained change ===", flush=True)
corp3, A3, N3 = mini_corpus(change_round=5, sustained=True)
rd_date = {r: f"2024-{1 + r // 4:02d}-{1 + 7 * (r % 4):02d}"
           for r in range(12)}
LAG = {3: 6, 5: 7, 8: 8}          # detection at the 2nd/3rd/4th new match
for W, det_idx in LAG.items():
    pl = SpecPlan("p1", W=W, c=3, corpus=corp3)
    evs = [e for e in pl.orgs["T00"]["events"]]
    check(f"F3 p1w{W}: exactly one detection", 1, len(evs))
    check(f"F3 p1w{W}: boundary index j == 5 (first new-five match)",
          5, evs[0]["j"])
    check(f"F3 p1w{W}: detected at match index {det_idx}, never earlier",
          rd_date[det_idx], evs[0]["det_date"])
    check(f"F3 p1w{W}: k == 3", 3, evs[0]["k"])
    fb = pl.final("T00")["boundaries"]
    check(f"F3 p1w{W}: exactly one final boundary at j=5, k=3",
          [(5, 3, False)], [(b["j"], b["k"], b["provisional"]) for b in fb])
p23 = SpecPlan("p2", m=3, c=3, corpus=corp3)
fin3 = p23.final("T00")
check("F3 p2: one confirmed boundary at j=5, k=3",
      [(5, 3, False)], [(b["j"], b["k"], b["provisional"])
                        for b in fin3["boundaries"]])
early = p23.version_asof("T00", rd_date[6])
check("F3 p2: provisional (not confirmed) as of the day after the dev",
      [(5, True)], [(b["j"], b["provisional"]) for b in early["boundaries"]])
conf_day = rd_date[8]
post = p23.version_asof("T00", "2024-03-02")   # after round 8 (m=3 reached)
check("F3 p2: confirmed once 3 non-reverting matches passed",
      [(5, False)], [(b["j"], b["provisional"]) for b in post["boundaries"]])
# new-phase games are full weight, not penalized (A1): o == 1 after j
check("F3 p1w5: new-phase games o==1 (not penalized)", True,
      bool((SpecPlan("p1", W=5, c=3, corpus=corp3)
            .final("T00")["o"][5:] == 1.0).all()))
check("F3 p1w5: per-game n counts from the boundary (0 for the first)",
      list(range(7)), SpecPlan("p1", W=5, c=3, corpus=corp3)
      .final("T00")["n"][5:].tolist())
# solve level, P1-W5: identical through detection date, divergent after
pl35 = SpecPlan("p1", W=5, c=3, corpus=corp3)
sa3 = mini_solve(corp3, pl35, a=2.0, tau=5.0, s=0.0)
sb3 = mini_solve(corp3, None, a=0.0, tau=5.0, s=0.0)
det5 = rd_date[7]
check("F3 p1w5 mini-solve: ALL teams identical through detection date "
      "(no peeking)", 0.0,
      max_abs_diff(sa3, sb3, day_filter=lambda D: D <= det5))
check_true("F3 p1w5 mini-solve: featured team moves after detection",
           max_abs_diff(sa3, sb3, team_i=0,
                        day_filter=lambda D: D > det5) > 0)

print("=== F1: SEN, the named case (real corpus) ===", flush=True)
corpus = load_corpus()
SEN_SEQ = corpus["team_match_seq"]["SEN"]
pos_of_sen = {m: i for i, (d, m) in enumerate(SEN_SEQ)}
J_MARVED = pos_of_sen[706350]
lu = corpus["lineups"][("SEN", 706350)]
names = {u.rsplit("/", 1)[-1] for u in lu}
check("F1 data: marved fielded in m706350 (2026-07-16)", True,
      "marved" in names)
check("F1 data: johnqt back in m706360 (2026-07-25)", True,
      "johnqt" in {u.rsplit("/", 1)[-1]
                   for u in corpus["lineups"][("SEN", 706360)]})
WINDOW = "2026-05-01"
for W in (3, 5, 8):
    pl = SpecPlan("p1", W=W, c=3, corpus=corpus)
    fb = pl.final("SEN")["boundaries"]
    check(f"F1 p1w{W}: no boundary at the marved match", [],
          [b for b in fb if b["j"] == J_MARVED])
    check(f"F1 p1w{W}: zero boundaries in the case window (>= {WINDOW})", [],
          [b["date"] for b in fb if b["date"] >= WINDOW])
    if W == 5:
        check("F1 p3w5 (same window detector): no M-shift in window", [],
              [e["det_date"] for e in pl.orgs["SEN"]["events"]
               if e["det_date"] >= WINDOW])
        fo = pl.final("SEN")["o"]
        check("F1 p1w5: marved game IS a deviation (o=0.8)",
              0.8, float(fo[J_MARVED]))
p2r = SpecPlan("p2", m=3, c=3, corpus=corpus)
fin_sen = p2r.final("SEN")
check("F1 p2: zero surviving boundaries in window", [],
      [b["date"] for b in fin_sen["boundaries"] if b["date"] >= WINDOW])
mid = p2r.version_asof("SEN", "2026-07-20")
check("F1 p2: provisional boundary LIVE between dev and reversion",
      [(J_MARVED, True)],
      [(b["j"], b["provisional"]) for b in mid["boundaries"]
       if b["date"] >= WINDOW])
after = p2r.version_asof("SEN", "2026-07-26")
check("F1 p2: retracted once the reversion is visible", [],
      [b["j"] for b in after["boundaries"] if b["date"] >= WINDOW])

print("--- F1 solve-level: P2 probe, final-timeline identity ---", flush=True)
import runner  # noqa: E402  (engine import deferred until classifier passed)

ra = runner.run_config(p2r, a=2.0, tau=5.0, s=0.0, daily=True)
corpus_nodev = dict(corpus)
corpus_nodev["lineups"] = dict(corpus["lineups"])
corpus_nodev["lineups"][("SEN", 706350)] = corpus["lineups"][("SEN", 706360)]
p2_nodev = SpecPlan("p2", m=3, c=3, corpus=corpus_nodev)
rb = runner.run_config(p2_nodev, a=2.0, tau=5.0, s=0.0, daily=True)
ti_sen = ra["tidx"]["SEN"]
days = sorted(ra["daily_r"].keys())
diffs = {D: float(abs(ra["daily_r"][D][ti_sen] - rb["daily_r"][D][ti_sen]))
         for D in days}
pre = max((v for D, v in diffs.items() if D <= "2026-07-16"), default=0.0)
dur = max((v for D, v in diffs.items() if "2026-07-16" < D <= "2026-07-25"),
          default=0.0)
post = max((v for D, v in diffs.items() if D > "2026-07-25"), default=0.0)
check("F1 p2 solve: SEN identical to the no-boundary run BEFORE the dev",
      0.0, pre)
check_true("F1 p2 solve: provisional window moved SEN (designed behavior, "
           f"max {dur:.4f})", dur > 0)
# ── final-timeline identity: the spec's P2 requirement. The single-pass
# harness leaves a residue after retraction; prove the root cause is v6's
# region-prior recursion (prev-day solve seeds each day's region prior), not
# a classifier/weight bug: (i) classifier state and weight multipliers are
# bit-identical post-retraction; (ii) with the recursion off, the residue
# must vanish EXACTLY. Then P2 is declared NOT RUN for scoring (spec §2.5:
# "do not run P2 rather than running it wrong").
va = p2r.version_asof("SEN", "2026-07-26")
vb = p2_nodev.version_asof("SEN", "2026-07-26")
check("F1 p2 root-cause (i): post-retraction classifier boundaries identical",
      [(b["j"], b["k"], b["provisional"]) for b in vb["boundaries"]],
      [(b["j"], b["k"], b["provisional"]) for b in va["boundaries"]])
mu_a = p2r.multipliers(va, 2.0, 5.0, 0.0)
mu_b = p2_nodev.multipliers(vb, 2.0, 5.0, 0.0)
check("F1 p2 root-cause (i): post-retraction weight multipliers identical",
      0.0, float(np.abs(mu_a - mu_b).max()))
ra0 = runner.run_config(p2r, a=2.0, tau=5.0, s=0.0, daily=True,
                        cfg_override={"region_prior_ridge": 0.0})
rb0 = runner.run_config(p2_nodev, a=2.0, tau=5.0, s=0.0, daily=True,
                        cfg_override={"region_prior_ridge": 0.0})
post0 = max((float(abs(ra0["daily_r"][D][ti_sen] - rb0["daily_r"][D][ti_sen]))
             for D in ra0["daily_r"] if D > "2026-07-25"), default=0.0)
check("F1 p2 root-cause (ii): residue vanishes EXACTLY with the region-prior "
      "recursion off (diagnostic config, not v6)", 0.0, post0)
FIX["f1_sen_p2_probe"] = {
    "config": "P2 m=3 c=3, a=2, tau=5, s=0 (probe)",
    "max_dr_pre": pre, "max_dr_provisional_window": dur,
    "max_dr_after_retraction_v6prior": post,
    "max_dr_after_retraction_regionprior_off": post0}
FIX["P2_DECISION"] = {
    "status": "NOT RUN (as a scoring/fitting policy)",
    "reason": "The spec requires a retracted P2 boundary to leave a final "
              "rating timeline IDENTICAL to the no-boundary run. v6's "
              "region_prior_ridge (1.5) seeds each day's region prior with "
              "the PREVIOUS day's solved ratings, so the provisional week's "
              "solves leave a residue in the prior chain after retraction — "
              f"measured max {post:.2e} rating points on the SEN probe. "
              "Root cause proven: classifier state and weight multipliers "
              "are bit-identical post-retraction, and the residue is exactly "
              "0.0 with the recursion disabled. Exact P2 semantics would "
              "need a full-chain re-solve per solve day (O(days^2)); per "
              "spec §2.5, P2 is NOT RUN rather than run wrong. P2 classifier "
              "fixtures + census remain published (classifier-only).",
    "classifier_fixtures": "PASSED", "census": "included"}

print("=== F4: corpus census ===", flush=True)
m2e = {}
for g in corpus["games"]:
    m2e[g["match_id"]] = g["event_id"]
EWC_PAT = ("ewc", "rbhg", "evo", "ten_", "acl", "radiant", "fgc",
           "convergence", "super_champions", "china_champions_qual")
csv_tags = corpus["ev_class"]


def tier(eid):
    if csv_tags.get(eid) == "vct":
        return "vct"
    if csv_tags.get(eid) == "ewc":
        return "ewc_class"
    return "ewc_class" if any(p in eid for p in EWC_PAT) else "vct"


mis = [e for e, t in csv_tags.items() if t == "vct"
       and any(p in e for p in EWC_PAT)]
check("F4 tier map: no csv-vct event matches the ewc pattern", [], mis)

census = {"tier_rule": {"csv_event_class_first": True,
                        "fallback_pattern_ewc": list(EWC_PAT),
                        "note": "untagged corpus-addition events classed by "
                                "pattern; shanghai_masters/lcq etc resolve "
                                "to vct (mainline)."},
          "reference": {"vct_standin_rate_pct": 6.7,
                        "ewc_class_standin_rate_pct": 23.3,
                        "source": "lineups agent, 30d-calendar-modal "
                                  "definition (briefs/context.md)"},
          "policies": {}}
team_matches = defaultdict(int)     # (tier,) lineup-known team-matches
tm_by_year_tier = defaultdict(int)
for org, sq in corpus["team_match_seq"].items():
    for d, mid in sq:
        if (org, mid) in corpus["lineups"]:
            t = tier(m2e[mid])
            team_matches[t] += 1
            tm_by_year_tier[(d[:4], t)] += 1

POLICIES = ([("p1", W, c) for W in (3, 5, 8) for c in (0, 3, 5)]
            + [("p2", 3, c) for c in (0, 3, 5)] + [("p3", 5, None)])
plans_cache = {}
for pol, Wm, c in POLICIES:
    pl = SpecPlan(pol, W=Wm if pol != "p2" else 5,
                  m=Wm if pol == "p2" else 3, c=c if c is not None else 3,
                  corpus=corpus)
    plans_cache[(pol, Wm, c)] = pl
    by = defaultdict(int)
    nb_total = 0
    dev_total = defaultdict(int)
    rev_counts = defaultdict(int)
    for org in pl.orgs:
        fin = pl.final(org)
        if fin is None:
            continue
        mm = pl.orgs[org]["matches"]
        for b in fin["boundaries"]:
            t = tier(m2e[mm[b["j"]][1]])
            by[(b["date"][:4], t, "boundaries")] += 1
            nb_total += 1
        dev_idx = np.where(fin["o"] < 1.0)[0]
        for i in dev_idx:
            t = tier(m2e[mm[i][1]])
            by[(mm[i][0][:4], t, "deviations")] += 1
            dev_total[t] += 1
    entry = {"total_boundaries": nb_total,
             "by_year_tier": {f"{y}|{t}": {"boundaries": by.get((y, t, "boundaries"), 0),
                                           "deviations_without_boundary": by.get((y, t, "deviations"), 0),
                                           "team_matches": tm_by_year_tier.get((y, t), 0)}
                              for y in ("2023", "2024", "2025", "2026")
                              for t in ("vct", "ewc_class")},
             "dev_rate_pct": {t: round(100.0 * dev_total[t] / max(team_matches[t], 1), 2)
                              for t in ("vct", "ewc_class")},
             "boundary_rate_per_100_team_matches": {
                 t: round(100.0 * sum(by.get((y, t, "boundaries"), 0)
                                      for y in ("2023", "2024", "2025", "2026"))
                          / max(team_matches[t], 1), 2)
                 for t in ("vct", "ewc_class")}}
    b_ewc = sum(by.get((y, "ewc_class", "boundaries"), 0)
                for y in ("2023", "2024", "2025", "2026"))
    d_ewc = dev_total["ewc_class"]
    entry["ewc_misclassification_flag"] = bool(b_ewc > 0.6 * max(d_ewc, 1))
    entry["b_ewc"], entry["d_ewc"] = b_ewc, d_ewc
    entry["share_of_ewc_lineup_events_becoming_boundaries"] = round(
        b_ewc / max(b_ewc + d_ewc, 1), 3)
    census["policies"][f"{pol}" + (f"w{Wm}" if pol != "p2" else f"m{Wm}")
                       + (f"c{c}" if c is not None else "")] = entry
    print(f"  {pol} W/m={Wm} c={c}: boundaries={nb_total} "
          f"dev_rate vct={entry['dev_rate_pct']['vct']}% "
          f"ewc={entry['dev_rate_pct']['ewc_class']}% "
          f"misclass={entry['ewc_misclassification_flag']}", flush=True)

# P2 retraction ledger (reverted-deviation counts, spec census requirement):
# under P2 a deviation with o<1 in the FINAL version is exactly a deviation
# that did NOT survive as a boundary (retracted or never-confirmed sub) —
# already counted as deviations_without_boundary above. Open provisionals:
for c in (0, 3, 5):
    pl = plans_cache[("p2", 3, c)]
    n_prov_open = sum(1 for org in pl.orgs
                      for b in (pl.final(org)["boundaries"] if pl.final(org)
                                else []) if b["provisional"])
    census["policies"][f"p2m3c{c}"]["open_provisional_at_corpus_end"] = n_prov_open

p1w5 = census["policies"]["p1w5c3"]
check_true("F4 sanity: ewc_class deviation rate exceeds vct (P1-W5)",
           p1w5["dev_rate_pct"]["ewc_class"] > p1w5["dev_rate_pct"]["vct"],
           str(p1w5["dev_rate_pct"]))
# The 60% rule DISQUALIFIES policies from shipping (ADDENDUM 4: "the census
# must expose any policy tripping the rule" — exposure, then exclusion at
# selection time). Record the verdicts as census output:
tripped = {k: v["ewc_misclassification_flag"]
           for k, v in census["policies"].items()}
census["misclassification_rule"] = {
    "rule": "b_ewc > 0.6 * d_ewc  =>  policy cannot ship (preregistered)",
    "verdicts": tripped,
    "shippable_policies": sorted(k for k, v in tripped.items() if not v),
}
check_true("F4: misclassification rule exposed with at least one shippable "
           "policy remaining", any(not v for v in tripped.values()),
           str(tripped))

with open(os.path.join(STATS, "roster_spec_census.json"), "w") as f:
    json.dump(census, f, indent=1)
_flush("PASSED")
print(f"ALL FIXTURES PASSED ({FIX['n_pass']} assertions) — "
      "stats/roster_spec_fixtures.json + roster_spec_census.json written",
      flush=True)
