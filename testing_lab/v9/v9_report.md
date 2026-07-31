# v9 Lab report — answer first

**THE LADDER IS V6 ALONE.** 3,240 configurations searched → 231 eligible on
train → 5 frozen candidates evaluated one-shot on era transfer → **0
advanced**. The frozen prospective ladder (stats/v9_ladder.json) is pure v6,
β = 0.128512 (refit once on 2023-01-01..2026-07-28, never again). Nothing
ships; nothing on the public site regardless; VCTMM stays hands-off. The only
thing that can change this verdict is the prospective scoreboard (§4) or the
re-open triggers (§5). Page: /testing/report/v9_lab.

## §0 The answer
The family is measured dead across its whole span. Solve-side boosts
transfer negative from a=0.5 through a=28 (this program: −0.67m → −1.70m
pooled, monotone in a; the autopsy adds a ∈ {4.5, 6, 28} all negative). The
prediction layer's best in-era config (δ2, +5.47m on FIT1) transferred to
−5.38m pooled with the block CI entirely below zero — the third
train-mirage, and the red-flag rule had found the tell before nomination
(74.8% of its FIT1 gain in 10 rows). δ1 (pure evidence-sign) never even fit
in-era (best dF +0.81m < 1.0m floor, LOEO negative — nominated nothing).
Hybrid solve+δ skipped as preregistered: no advancing parents.

## §1 The motivation was real
LEV −14.7pp/match for 10 matches (Neon change), ENVY +24–32pp (chain),
atlas +4.4/+5.6/+4.8pp first-3-match outperformance. The misses are not
disputed; every mechanism tried pays more elsewhere than it earns on the
change windows. Case charts on the Roster page.

## §2 The validation design
Era-transfer selection (fit 2023-24 → validate 2025; fit 2023-25 → validate
2026H1; win BOTH), clauses A1–A5, measured null false-advance ≈ 2%, power
≈ 0.75–0.9. The gate autopsy: the a=28 ship would have been **BLOCKED on
every clause it touched** (VAL1 −5.08m, VAL2 −21.05m, pooled −12.21m CI
[−18.99, −5.58]; holdout said −11.60m, reproduced to the third decimal).
Win-BOTH-eras is load-bearing: damage concentrates in 2026H1 while 2025 is
flat, so a single pooled test would have looked merely inconclusive.

## §3 The search
Grid: 4 policies × 5 direction variants × 9 b × 6 τ × 3 γ = 3,240 configs
on FIT1 only (MDE 2.15m); eligibility dF ≥ 1.0m AND era_min > 0 AND LOEO
min > 0 → 231. Five one-shot ledgered transfer evaluations, all DIE (every
one failed A1–A4; A5 never reached): solve a=0.5/1.0/2.0 (−0.67/−0.91/
−1.70m pooled), δ2 (−5.38m [−9.29, −1.03]), hybrid(6) (−1.98m). Mechanics
fixture-proven first: 67/67 passed (bitwise v6 nesting, per-team isolation,
date-strict evidence, a ≤ 6 hard cap in code).

## §4 The prospective scoreboard (the real referee, live)
Standing evaluator: testing_lab/v9/score_prospective.py — idempotent,
read-only over data/, refits nothing; run manual or cron weekly. Implements
stats/v9_prospective_protocol.json verbatim: settled series > 2026-07-28 by
the frame recipe; checkpoints at scored n ∈ {100, 200, 400}; G1 p_better ≥
{.999, .995, .975} both CRN modes (α-spend union 0.031); G2 Δ ≥ MDE(n) =
{6.18, 4.37, 3.09}m; G3 v8 bucket bars verbatim; G4 team-bias ≤ v6 + 2pp;
G5 most conservative; kill if block ci_hi < 0; no promotion by 400 ⇒ NO
SHIP. Current state (stats/v9_prospective_scoreboard.json):
**accumulating (n=0)** — the live data has zero settled series dated
> 2026-07-28 yet; first read at n=100 (Stage 2 live, Champions Sep–Oct).
Arms: v6 baseline (β 0.128512) + six v8 reference arms scored for the
record only (B_continuity β 0.127363, C_coldstart β 0.131629,
F_v6_overreact β 0.125735, H_specrun β 0.094734 — all frozen once on
≤ 07/28 data; E_atlas non-model; D_phase_reset NOT_SCOREABLE_LIVE — its
h3-1b round-level base has no enrichment for CN/live corpus, disclosed).
Candidate set: EMPTY. Fixtures at init: live frame rebuild = frozen frame
2058/2058 rows, 0 column mismatches; protocol β refit reproduces the
ladder's 0.128512 exactly (gap 0.000, 2,044 freeze rows).

## §5 Ledger and integrity
Looks: 5 selection reads (one per candidate, ledgered), 5 preregistered
autopsy reads, exploratory 0/3, prospective 0/3. The design agent's
struck-draft process incident is quoted verbatim on the page (a prior
preregister draft briefly contained invented post-run numbers; caught and
struck within minutes, before any analysis ran). Do-not-retest: the roster
family at these definitions (solve-side a ∈ [0.5, 28]; δ2/δ1/hybrid
overlays at the searched grid; the spec-run config). Re-open triggers ONLY:
a genuinely new player-level data source, or a prospective surprise on the
§4 scoreboard. Standing: the three v9 laws; a ≤ 6.0 cap is code; market
data never a fitting target.

— agent:v9-finish, 2026-07-29. Artifacts: score_prospective.py,
stats/v9_prospective_scoreboard.json, stats/v9_page_{grid,autopsy}.json,
gen_v9_report.py (testing_lab/), out/reports/v9_lab.html, TestingLab.py
/testing/v9/stats/ route, nav patches (7 pages + hub + 2 generator
sources), logs/prospective.log, logs/finish.log, preregister.finish.md.




## 6. The plain-English version

The plain-English version


**What we were trying to fix.** After LEV added Neon, the model kept pricing them too low
(≈15 points too low per match, for
6 matches). After ENVY swapped players mid-season, it kept pricing
them too high (≈24 points per match). Both
misses are real and documented on the Roster page. The idea: give v6 a subsystem that reacts
faster right after a roster change, and tune everything between plain v6 and the full subsystem.


**What got tested.** 3,240 versions across two designs — one that
re-weights games inside the rating solve, and one that only nudges the changed team’s match
probability. Every version was judged the hard way: does a gain learned on 2023–24 still
show up in 2025 and 2026? Looking good where you were fit doesn’t count.


**What was worse.** Every rating-level version made the model worse, at every strength
tried — even tiny boosts — because re-weighting one team’s games bends every
other team’s rating through the shared solve. The most aggressive version a gate had
previously approved (a=28) was the worst of all
(-11.6m on the holdout, losing most exactly on
the full-rebuild matches it was built for).


**What looked better, but wasn’t.** The best probability-layer version looked
+5.5m better on the era it was tuned on
— but 75% of that gain came from
just ten matches, and on the eras it had never seen it was
-5.4m worse. That is overfitting,
and the referee caught it before it could be promoted.


**The conclusion.** v6 stays, unchanged — 0 of the
5 finalists earned promotion. Not because the roster problem
isn’t real, but because the fixes lose on roster-change matches themselves: the
rating-level versions additionally bleed into unchanged teams through the shared solve (part of
why they are worst), while the probability-layer versions — which by construction cannot
touch unchanged teams — still lose net across the corpus’s other roster
changes: the always-boost version helps LEV-shaped upgrades but actively hurts ENVY-shaped
downgrades, and the read-the-early-results version has no reliable direction signal at 1–5
matches. The case-by-case ledger is in §7. What ships instead is on the betting side: the
roster_flag tells the bot to quote smaller on fresh rosters. And the
question stays open the right way: from tonight onward, every settling match scores the frozen
candidates on data nobody has touched (§4). If reacting faster to roster changes genuinely
helps, the scoreboard will say so at its checkpoints — and the pass/fail rules for that
are already locked, so nobody can move the goalposts, including us.

## 7. Where exactly it lost — the case ledger

Where exactly it lost — the case ledger

locality (your test)Δ = 0.000
on every scored match with NO active roster phase (5 such
matches) — asserted, prediction-layer candidates cannot touch unchanged teams
tie-outPASS
per-match sums reproduce the ledgered totals (-5.38m pooled)
the catch: always on
1182
/1217
scored validation matches had an active roster phase — the winning config's
65-match horizon meant the subsystem effectively never turned off (5
phase-free matches total)
solve-side coupling (S_a1.0)-447m·rows
damage on the 2 truly-untouched VAL2 matches — the rating-level family's
disqualifier, quantified (tiny n because phases were ubiquitous)



### Who the adjustment helped and hurt (δ2, contribution to window mean)


Each affected match is classed by its DOMINANT active side’s walk-forward
evidence (E>0 improving = LEV-shape, E<0 degrading = ENVY-shape). δ2 lost on BOTH:
it hurt even where its always-up push agreed with the evidence’s direction — the
failure is magnitude and timing, not just direction. And with phases lasting 65 matches, the
subsystem was effectively always on, adjusting far outside the fresh-change windows the concept
was aimed at.


### When the adjustment hurt: by matches-since-change


The direct test of “react harder to the first results”: the damage
concentrates exactly where the push is biggest — the first 1–5 matches after a change
— while the nearly-decayed tail (14+ matches) was mildly positive in both eras. The early
post-change results are the noisiest part of the phase; leaning on them harder than v6 does is
where the money is lost.


### Worst 10 affected matches (δ2 vs v6)


| datematchactive phasesp_v6 → p_δ
ΔLLreason |
| 2025-11-16 | FPX beat NOVA 3-1 | FPX k=5/5 n=18; NOVA k=2/5 n=40.434 → 0.275-457m | net push 0.370 toward NOVA (k=2/5 kept, n_since=4, E=+1.98 maps vs exp over 4); NOVA lost 3-1 vs FPX — push backfired; L |
| 2026-04-08 | EF beat BBL 2-0 | BBL k=0/5 n=80.299 → 0.198-410m | net push 0.351 toward BBL (k=0/5 kept, n_since=8, E=+2.32 maps vs exp over 8); BBL lost 2-0 vs EF — push backfired; LEV- |
| 2025-11-13 | FPX beat WOL 2-1 | FPX k=5/5 n=16; WOL k=2/5 n=40.442 → 0.311-350m | net push 0.370 toward WOL (k=2/5 kept, n_since=4, E=-0.54 maps vs exp over 4); WOL lost 2-1 vs FPX — push backfired; ENV |
| 2025-03-31 | BME beat GE 2-0 | GE k=0/5 n=40.617 → 0.440-338m | net push 0.478 toward GE (k=0/5 kept, n_since=4, E=-0.23 maps vs exp over 4); GE lost 2-0 vs BME — push backfired; ENVY- |
| 2025-03-30 | 2G beat LOUD 2-0 | LOUD k=2/5 n=100.167 → 0.121-322m | net push 0.233 toward LOUD (k=2/5 kept, n_since=10, E=-1.42 maps vs exp over 10); LOUD lost 2-0 vs 2G — push backfired; |
| 2026-02-14 | G2 beat MIBR 3-0 | MIBR k=2/5 n=3; G2 k=4/5 n=150.550 → 0.407-302m | net push 0.308 toward MIBR (k=2/5 kept, n_since=3, E=+1.21 maps vs exp over 3); MIBR lost 3-0 vs G2 — push backfired; LE |
| 2025-01-18 | NS beat ZETA 2-0 | ZETA k=2/5 n=30.573 → 0.424-301m | net push 0.400 toward ZETA (k=2/5 kept, n_since=3, E=+0.08 maps vs exp over 3); ZETA lost 2-0 vs NS — push backfired; LE |
| 2025-01-17 | VIT beat KC 2-1 | KC k=1/5 n=3; VIT k=4/5 n=180.560 → 0.415-299m | net push 0.389 toward KC (k=1/5 kept, n_since=3, E=-0.31 maps vs exp over 3); KC lost 2-1 vs VIT — push backfired; ENVY- |
| 2025-01-29 | TH beat BBL 2-0 | BBL k=0/5 n=2; TH k=4/5 n=180.686 → 0.513-291m | net push 0.485 toward BBL (k=0/5 kept, n_since=2, E=+1.46 maps vs exp over 2); BBL lost 2-0 vs TH — push backfired; LEV- |
| 2026-04-15 | GX beat BBL 2-1 | BBL k=0/5 n=9; GX k=2/5 n=260.314 → 0.236-287m | net push 0.257 toward BBL (k=0/5 kept, n_since=9, E=+1.06 maps vs exp over 9); BBL lost 2-1 vs GX — push backfired; LEV- |


### Best 10 affected matches


| datematchactive phasesp_v6 → p_δ
ΔLLreason |
| 2026-04-19 | KRÜ beat NRG 2-0 | NRG k=4/5 n=17; KRÜ k=0/5 n=40.221 → 0.344+442m | net push 0.399 toward KRÜ (k=0/5 kept, n_since=4, E=-0.80 maps vs exp over 4); KRÜ won 2-0 vs NRG — push paid; ENVY-shap |
| 2026-02-06 | EG be

(full per-match detail: stats/v9_case_decomposition.json)
