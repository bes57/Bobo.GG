# THE SPEC RUN — v6 + mid-season roster subsystem (agent:roster-g, 2026-07-29)

**v6 does not gain a roster subsystem.** The activation gate FIRED on train
inner-CV (P1 W=8 c=5; shrunk â=28, τ̂=13, ŝ=0.7, λ̂=0; CV improvement
+6.42m vs its own SE 5.34m), but the single preregistered corpus-wide
holdout read killed it: **−11.595m vs v6** (iid CI [−18.99, −4.39], block
[−18.18, −5.20], p_better .001/.000, n=1217) against a **1.773m** MDE — the
preregistered falsifier fired, so **the deployed model is v6 exactly** and
the frozen prospective arm H_specrun carries the config only for
adjudication, expectations LOW. The out-of-sample failure of a fired gate
is itself the headline finding: mean>SE at K=5 inner-CV did not protect
against this family's train-era overfit (third instance of the 2023-24 →
2025-26 non-transfer after treatments (b) and (e)).
`stats/roster_spec_read.json` · preregister ADDENDUM 4 (+outcomes).

What DID hold, spec-conformant end to end (`roster_spec_fixtures.json`,
54/54 hard assertions BEFORE any run): the **causal sub-vs-change
classifier** — SEN/Marved (one match 2026-07-16, reverted 07-25, verified
from data) produces zero boundaries under every policy; synthetic
sustained-change detection lands at exactly the policy's lag (never
earlier — peeking guard at solve level, max|Δr| == 0.0 pre-detection);
a=s=0 is v6 **bit-identical by sha256 checksum**. **P2 = NOT RUN** as a
scoring policy, per the spec's own escape: v6's region-prior recursion
carries provisional residue (3.5e-4 rating pts) across retraction — root
cause proven (residue exactly 0.0 with the recursion off) — so the
single-pass harness cannot reproduce final-timeline identity honestly.
**Census** (`roster_spec_census.json`): at c=0, 69-74% of EWC-class lineup
events become boundaries (the spec's predicted misclassification spike,
exposed); the preregistered 60% rule disqualified c∈{0,3}; at c=5,
event-length stand-in excursions merge into **k=5 boost-inert boundaries**
— sub rejection at event scale falling out of the chain rule. **Case
charts** (`roster_spec_cases.json`): per-team ablations with pre-boundary
max|diff| **exactly 0.0** (global, 514/487/631 solve days) — ENVY 2026-02-06
k=4/5 + merged May-July chain k=3/5; LEV 2025-11-29 k=2/5; SEN: no
verticals, ablation ≡ v6. **Coupling** (own subsection of the read JSON):
under the corpus-wide run, never-changing teams move |Δr| p90 1.70 / max
8.03 rating points — the a=28 solve deformation is where the damage lives
(sub-heavy rows themselves: +0.56m, neutral). **Integration**:
`roster_integration.json` now carries the spec's roster_flag extension
(modal five, last-match deviation, matches-since-boundary, provisional bit;
73 teams, 22 pending as of 2026-07-27) for Tier-1 sizing only. Looks
402→403 (1 of ≤3 budget; contrast condition not met). Everything
EXPLORATORY on the spent holdout.

---

# Roster-change adaptation — operator-requested report (agent:roster)

Page: `testing_lab/out/reports/roster_adaptation.html` (via `gen_roster_report.py`).
Preregistered: `preregister.roster.md` + OPERATOR ADDENDUM (phase-reset filter),
appended after prereg lock but BEFORE any treatment computation — ordering
disclosed in `logs/roster.log`. **EXPLORATORY THROUGHOUT: the holdout is spent
(398 prior recorded looks; this report adds 4, grand total 402 —
`stats/roster_looks.json`). Nothing here is confirmatory or promotable.**

Frame: `data/frame_expanded/series.csv` (sha verified vs crn.json every script).
Baseline: stored v6 (β 0.1152, holdout LL 0.64216, n=1217), replayed
bit-identically for daily ratings. All CRN judging via `referee.py`; floors
quoted from `power_mde_expanded.json` (within 1.773m / cross 5.889m, n=1217)
and `power_mde.json` (post-change ≤3 bucket: 2.52m / 7.81m at frozen n=598).

## 1. ENVY 2026 S1→S2, forensically (descriptive)

A chained pair of one-out swaps, verified from lineups (never memory):

| date | event | out | in | kept | run |
|---|---|---|---|---|---|
| 2026-02-06 | 2026_kickoff | inspire | Demon1 | 4/5 | 6 |
| 2026-05-12 | 2026_ewc_qual | eggsterr | NightZ | 4/5 | 3 |
| 2026-07-17 | 2026_stage2 | p0ppin | Glyph | 4/5 | 3 (censored) |

Stage-1 five {demon1, eggsterr, keznit, p0ppin, rossy} → Stage-2 five
{demon1, glyph, keznit, nightz, rossy}: **S1→S2 overlap 0.6 (3/5 kept)**.
v6 (year-boundary continuity only) carried the rating through all of it:
**+24.3pp/match** overpricing over the 6 matches after the 2026-05-12 chain
start (pre-change bias +7.4pp), **+32.2pp/match** over the 3 Stage-2 matches so
far (censored). Rating stabilization ≈ 5 matches after the May chain start.

Named cases (operator amendment, data-verified): **LEV gains Neon**
2025-11-29, 2/5 kept (in blowz, spike, Neon; out tex, c0m, okeanos) — v6
**under**-priced LEV by **−14.7pp/match** for 6 matches (improvement case).
**ENVY loses inspire** 2026-02-06 — v6 **over**-priced by **+7.4pp/match**
(degradation case). Gallery of full rebuilds (0/5 kept): GE 2025-01-19
(+12.8pp), EDG 2025-11-29 (−1.8), FUR 2026-01-16 (−16.1), KRÜ 2026-01-16
(−8.9) — direction-agnostic at case level. `roster_case_envy.json`,
`roster_case_gallery.json`.

## 2. Population atlas (descriptive)

333 sustained changes 2023-26 (205 keep-4, 75 keep-3, 53 overhaul; 156
transient; CN churns most, 102). Adaptation curve (mean(won − p_v6), first 3
matches pooled, team-observations): **keep4 +4.37pp [0.65, 8.09] · keep3
+5.63pp [−0.55, 11.80] · overhaul +4.80pp [−2.99, 12.59]** vs stable reference
+0.73pp [−2.02, 3.47]. Post-change teams **outperform** the carried rating —
the operator's "new phase" intuition is directionally right in the data, and
skews toward improvement (teams change because they're underperforming).
`roster_population.json`.

## 3. Three preregistered EXPLORATORY reads (all negative; `roster_treatments.json`)

| read | spec (train-selected) | Δ overall | Δ post-change ≤3 | verdict |
|---|---|---|---|---|
| (b) change-point continuity | γ=2.0 replaces year mode | **−3.889m** iid [−8.46,+0.35], block [−7.77,−0.26] | −4.95m [−11.72,+1.55] | **falsifier FIRED** — dead on frame |
| (c) partial cold start | a0=1.0, M=6, ov≤0.6 blend to region mean | **−1.423m** [−3.54,+0.75] | −2.83m [−6.36,+0.58] | INSIDE NOISE FLOOR — no support |
| (d) phase-reset filter (operator) | g=0.5 on h3-1b; Δq=g(1−k/5)R at change | **−19.275m** [−29.33,−9.40]; **−7.524m vs own base** [−12.08,−3.34] | −23.79m [−38.13,−9.09] | **falsifier FIRED** — dead on frame |

| (e) v6 + post-change overreaction (operator-directed, reserve read) | a=2.0, τ=5.0; post-change games ×(1+a(1−ov)e^(−n/τ)) on v6 EXACTLY | **−0.395m** [−3.13,+2.30] | −0.19m [−4.38,+3.95] | INSIDE NOISE FLOOR — falsifier NOT fired; a null, not a kill |

ROI translations (symmetric first-order): −0.32pp / −0.12pp / −1.60pp / −0.03pp maker-ROI.
(b)'s +3.6m train gain did not transfer. **The operator's contrast prediction
is REVERSED**: (d)−(c) on improvement cases = **−38.32m [−68.25, −9.17]** —
the mean-blend beats the variance-spike on the rows the over-reaction was
designed for; single-map results are too noisy for a one-shot gain spike.
The learning-rate telemetry chart confirms the intended shape renders
(overhaul gain 0.294 vs stable 0.090 at m=0, decaying by ~m=5) — it just
doesn't pay on this frame. Look hygiene: grids train-only, holdout scrubbed;
reserve read unused (no train tie); 1b base + v6 baseline = stored reuses.

**Attribution acknowledgment (operator's design point, correct):** read (d)
confounded base-model replacement with the mechanism — the no-injection
state-space base is itself −11.75m vs v6 (stored), so (d) could not cleanly
falsify phase-reset-on-v6. **Read (e) is the clean attribution**: v6 exactly
(reproduction guard to 1e-12), one mechanism added. Its verdict: NULL inside
the floor — no detectable overreaction benefit at this n, but the slices the
mechanism targets lean positive (improvement cases +4.12m [−3.99,+12.36],
gated +2.57m, keep4 +2.69m; CIs all straddle 0). Adjudication moves to the
prospective arm F_v6_overreact (a=2.0, τ=5.0 frozen). Overlay windows carry
`v6_overreact_path` on the v6 base (pre-change solve coupling of the
corpus-wide rule: ENVY max |Δ| 0.735 / mean −0.03 over 19 pre-change days;
LEV max 0.271 / mean −0.04 over 10) and `base_path` (g=0 state-space base)
so the charts separate base-model difference from mechanism.

## 4. Wave-2 context (adversary demotions respected)

Ledger rows "Lineup-overlap roster reweighting" / "Roster-instability shrink"
were UNRESOLVED re-opens — answer on this frame: negative. Decay 5b-a lineup
continuity: redundant with year mode, inside floor — consistent with (b).
H3 5d HL table: demoted to hypothesis; quoted only as such. H3 cold(<10 maps)
+71.7m stays a **cold-start** lead, not a roster-change lead — (c) found
nothing moving toward priors on change rows. Compose S1: HOLD (evidence's
verdict); the original change-scoped-gate Read 3 was withdrawn UNREAD when the
operator's phase-reset replaced it.

## 5. Integration + prospective plan (`roster_integration.json`)

**Tier 1 (sizing-only): RECOMMENDED NOW** — snapshot `roster_flag`
{overlap_vs_prev, matches_since_change, run_len_alive, sustained_alive,
change_date} from the existing maps CSVs; playbook sketch: msc ≤ 2 & ov ≤ 0.6
→ quarter-size + widen a tick; msc ≤ 2 & ov = 0.8 → half-size. Zero fair-value
risk; justified by the atlas + cases. **Tier 2 (fair-value): NOT NOW** — no
treatment earned it; v6's year continuity stands. Player-identity mean shift:
design note only (no read spent). **Prospective test (preregistered)**: series
dated > 2026-07-28, arms = v6 / (b) γ=2.0 / (c) a0=1.0,M=6 / (d) g=0.5 / (e) a=2.0,τ=5.0
(expectations LOW, carried for adjudication) / atlas-replication check;
decision rule = ΔLL ≥ MDE80 at realized n AND p_better ≥ .95 both CRN modes
AND no G3 regression; MDE80 ≈ 6.2m at n=100 fresh series.
