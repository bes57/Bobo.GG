# Pre-registration — agent:power (Phase 0: power analysis)

Written 2026-07-28, BEFORE computing any result below. Inputs are frozen:
per-series probabilities from `out/v7_probs.npz` / `out/v7_probs2.npz` (v7 era,
1695 rows, holdout n=1007), aligned to `harness.load_series()` rows 0..1694 —
alignment already verified by reproducing all 24 published `ll_test` values to
5 dp and the two motivating seed-7 bootstraps to machine precision
(logs/power.log T1). No model constant is retuned anywhere in this phase; no
new model is fit. Estimating the NOISE of the test statistic on the holdout is
measurement of the instrument, not holdout tuning — the holdout is never used
to select a model here.

## 1. MDE estimator (overall)

Per candidate pair (candidate c vs champion v6), per-series loss difference
`d_i = l_v6,i − l_c,i` with `l = −log(p_winner)`, on the 1007 valid holdout
series. σ_d = sample SD (ddof=1).

- **Analytic MDE** (two-sided α=0.05, power 80%):
  `MDE_80 = (z_.975 + z_.80)·σ_d/√n = 2.8016·σ_d/√n`.
- **Two regimes, reported separately** (σ_d differs by construction):
  - *within-family* — variants sharing the v6 consistency core:
    v5_asym_W20L12, consist_16_10, consist_14_8, consist_12_8,
    v6_consist_20_12+form{3,5,8}.
  - *cross-family* — structurally different decay/shape:
    sym_{6,8,10,12,14,16,20,24}, surprise_{12_20,16_24}, boxexp_{c3_hl8,c5_hl10},
    power_t6_a15, sym_20+form{3,5,8}.
  Regime headline = median σ_d over the regime's pairs; full spread reported.
- **Simulation verification** on two representative pairs (consist_16_10 =
  within; sym_20 = cross): recenter d (subtract its mean), inject shift
  μ ∈ {0.5, 1.0, 1.5}×MDE_analytic, resample n=1007 iid, run the actual
  paired-bootstrap decision (percentile 95% CI excludes 0 ⇔ p_better outside
  [.025,.975]); 400 simulations × 2000 resamples, seeds from crn.json
  mc_seeds. Acceptance: simulated power at μ=MDE_analytic in [0.74, 0.86]
  (binomial noise at 400 sims). If outside, the simulated MDE (interpolated μ
  at 80% rejection) supersedes the analytic number and the discrepancy is
  reported.

## 2. MDE per bucket

Same estimator on bucket subsets of the holdout; bucket n as observed. Bucket
list (fixed now): format bo1/bo3/bo5/bo5_gf; stage groups/playoffs/
grand_final; international/domestic; EWC-class events; CN involved;
cross-region; domestic Americas/EMEA/Pacific/CN; favorite band [.5,.6),
[.6,.7), [.7,.8), [.8,.9), [.9,1] (band of max(p,1−p) under frozen v6 probs);
rating-gap bands <1.5, [1.5,4), [4,7), 7+ (|rd__v6_consist_20_12|);
post-break (either team's days since previous series > 45, computed over the
full series history; target = reproduce v6_profile n=135, definition recovery
documented if a variant is needed); elite-vs-floor and form-shift (frozen
arrays in the npz); roster-change recency from
`testing_lab/v8/data/lineup_features.csv` — if absent at finish time, that
bucket is reported PENDING DATA, not approximated.
EWC-class = holdout event_ids matching ewc/evolution-series (target =
reproduce v6_profile n=109; actual id list documented).
Per bucket: σ_d median within each regime → MDE per regime.

## 3. v7 ladder re-adjudication (all 24 configs)

For every config vs v6_consist_20_12: mean Δ (milli-LL/series), iid bootstrap
CI + P(better) under (a) legacy seed 7 (reproduction of record) and (b)
crn.json canonical seeds; block-by-event bootstrap CI (crn block seed, unit =
event_id, 18 events). **Verdict rule (fixed now): DISTINGUISHABLE iff
|mean Δ| ≥ MDE_80 of that pair; else INSIDE NOISE FLOOR.** Secondary flags:
iid CI excludes 0; block CI excludes 0. sym_20 and consist_16_10 reported
explicitly. v6 appears as its own reference row.

## 4. Ledger reclassification (published ledger; generator source is authoritative)

Note recorded now: the authoritative source `gen_final_model.py` REJECTED list
and the published HTML both contain **32** entries, not the 33 the brief
states; classified as-is and the discrepancy reported.

Status semantics (fixed now, applied to the IDEA):
- **REFUTED** — idea stays dead: (i) mechanism-rejected (bug, double-count,
  leak, structural fit-to-noise) — flagged `mechanism`, immune to magnitude; or
  (ii) recovered effect worse than champion by ≥ its MDE at time of test.
- **UNRESOLVED** — no mechanism ground and recovered |Δ| < MDE (verdict was
  inside the noise floor), or magnitude UNRECOVERABLE from out/*.json.
  UNRECOVERABLE is marked explicitly, never silently dropped.
- **CONFIRMED** — re-analysis shows the idea distinguishably BETTER (≥ MDE)
  than its era champion, i.e. the rejection was itself wrong. Expected rare.
MDE at time of test = 2.8016·σ_regime/√n_at_test, with σ_regime from §1
measured on the current frozen probs (within-family σ for solve-tweak ideas,
cross-family σ for structural ideas) and n_at_test recovered from the source
JSON where present (else current 1007, flagged `n_assumed`). Tie-magnitude
recovery searched across `testing_lab/out/*.json`; the source file for each
recovered number is recorded in the row.

## 5. Variance reduction (each reported as effective-n multiplier)

(a) **Pairing**: (var(l_c)+var(l_v6))/var(d) per pair (unpaired two-sample vs
paired variance of the mean difference at equal n).
(b) **Control variate**: x = l_v6 − mean(l_v6); OLS slope β̂ on the holdout;
variance ratio var(d)/var(d−β̂x) = 1/(1−ρ²). Caveat reported: for
candidate-vs-v6 the covariate is a component of d — the ratio is the design
number for FUTURE candidate-vs-candidate tests and CUPED-style adjustment; the
point estimate of mean Δ is unchanged in-sample.
(c) **Blocking by event**: DEFF = var_block(mean d)/var_iid(mean d) from the
two canonical bootstraps. DEFF > 1 means iid CIs understated σ (reported as an
inflation, not a reduction — this is the honest direction).
(d) **Multivariate CV**: residualize d on [l_v6, |rd_v6|, p_fav_v6];
incremental ratio over (b). No other designs.

## 6. What would change the program's conclusions (fixed before results)

- Cross-family MDE(overall) > 2.0 milli at n=1007 ⇒ every past near-tie
  rejection with |Δ| < 2 milli was a coin flip dressed as a verdict; the v7
  "no promotion" outcome stands only as "not distinguishable", not "worse".
- Within-family MDE < 0.5 milli ⇒ tightly-coupled variant verdicts
  (consist_16_10-class) were adequately powered and stand as stated.
- ≥ 1/3 of ledger entries flip UNRESOLVED ⇒ the do-not-retest ledger cannot
  gate future work without power annotations; the v8 page must carry them.
- Block-vs-iid DEFF > 1.3 ⇒ all historical iid CIs were too narrow; borderline
  verdicts weaken further and block CIs become the program standard.
- CV multiplier ≥ 1.5× ⇒ the CV-adjusted paired test becomes the recommended
  standard for referee.py (Phase 6).

## Randomness

All new numbers use crn.json (iid seed 20260728, block seed 20260729, mc_seeds
pool); legacy seed 7 used only to reproduce published records. n_boot = 4000
for CIs; simulations as in §1.

---

## POST-CORPUS ADDENDUM (2026-07-28, after agent:corpus landed; written BEFORE computing)

Production rating files are NOT rebuilt (`data/rating_timeline*.json` untouched);
per-series probabilities remain the frozen v7 arrays. Two expanded-corpus MDE
estimators, reported side by side:

**E1 — Mechanical scaling.** I count scoreable holdout series myself from the raw
corpus (`data/match_results.csv` per-map rows + org pairs from `data/maps/<event>.csv`
player rows + `data/match_dates.json`), mirroring harness rules exactly: both orgs in
ORG_REGIONS (junk-org filter), series score from per-map winners, winner maps wm > lm
and wm in {1,2,3} (forfeit/incomplete filter), de-dup MatchID, holdout = date >
2024-12-31. Prefranchise (v8/data/prefranchise/) is excluded; I verify no 2021-22
dates exist in match_dates.json. MDE_naive(regime) = 2.8016 * sigma_regime_old /
sqrt(n_new), i.e. the Phase-0 regime MDEs scaled by sqrt(1007/n_new). Calibration
targets the estimator must reproduce before being trusted: 2,068 total raw series,
~1,223 raw 2025+ series, ~+335 raw new series (corpus agent's counts).

**E2 — Composition-adjusted.** Partition (fixed now): intl = event_id contains
masters/champions/lock_in (harness rule); ewc_offseason = EWC/Evolution families plus
off-season invitationals (rbhg, ten_*, radiant_*, convergence, fgc, acl and any new
registry event of that archetype); vct_domestic = everything else (official-circuit
events incl. LCQ/champions quals/league). sigma_class per regime = median over the
regime's frozen v7 pairs of sd(d) on that class's OLD-holdout rows (computed from the
frozen npz probs only). sigma_adj^2 = sum_class w_class * sigma_class^2 with w_class =
class shares of the EXPANDED scoreable holdout. MDE_adj(regime) = 2.8016 * sigma_adj /
sqrt(n_new). Assumptions stated with the result: class sigmas transfer from old to new
series of the same class (new off-season events are mapped to the measured EWC-class
sigma — if anything an understatement of their noise); d-distributions stationary.

**Decision fixed now: the checkpoint quotes the composition-adjusted number** (the
conservative one); mechanical scaling is reported alongside as the optimistic bound.
No resampling is planned (point estimates from frozen vectors); if any resampling
becomes necessary it uses crn.json seeds. Failure accounting: every raw new series
that fails scoreability is counted with its reason (org not in ORG_REGIONS / no map
data / invalid series score / no date). n_holdout_old = 1007 (frozen npz era);
organic post-npz growth (timeline rows after 2026-07-23) is reported separately from
corpus additions, identified by event: corpus additions = events absent from the old
timeline frame.
