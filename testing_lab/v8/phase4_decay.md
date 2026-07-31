# Phase 4 — recency & asymmetry, tested properly (agent:decay, 2026-07-28)

Frame: `v8/data/frame_expanded/series.csv` (sha `ff772d41…d55142` verified vs
crn.json), n=2058, train 841 / holdout 1217. Engine corpus 5140 maps, 73 orgs,
637 pred days. Runner validated bit-exact against `engine.run` on v6
(max |Δrdiff| = 0.0). All boots CRN (iid seed 20260728, block 20260729,
n_boot 4000). Pre-registered in `preregister.decay.md` BEFORE runs. Verdict
rule (pre-committed): WIN/KILL needs |Δ| ≥ pair-empirical MDE₈₀
(2.8016·σ_d/√n) AND p_better ≥.95/≤.05 in BOTH boot modes; otherwise INSIDE
NOISE FLOOR. Family MDEs at n=1217: **within 1.773m, cross 5.889m**.

**v6 champion on the expanded frame: holdout LL 0.64216 (β 0.1152, n=1217).**
Continuity: restricted to the 1007 frozen-npz rows it scores 0.64085 vs the
published 0.64095 — the +2023-24 corpus barely moves the frozen-row surface.

## 5a — the near-ties, re-raced (CV-adjusted CIs)

| cand vs v6 | Δ (milli) | pair MDE raw→CV | iid CI raw | iid CI CV | p_iid / p_blk | verdict |
|---|---|---|---|---|---|---|
| consist_16_10 (within, fam 1.773) | **−0.53** | 1.67→1.66 | [−1.70,+0.61] | [−1.70,+0.60] | .19 / .13 | **INSIDE NOISE FLOOR** |
| sym_20 (cross, fam 5.889) | **−2.17** | 3.90→3.39 | [−4.84,+0.53] | [−4.59,+0.13] | .056 / .058 | **INSIDE NOISE FLOOR** |
| sym_24 (cross) | **−2.42** | 4.31→3.61 | [−5.42,+0.55] | [−5.04,+0.05] | .058 / .076 | **INSIDE NOISE FLOOR** |
| sym_16 (cross, control) | **−2.33** | 3.94→3.66 | [−5.11,+0.41] | [−4.95,+0.20] | .047 / .015 | **INSIDE NOISE FLOOR** |

The answer is **ties**, in those words. The control variate delivers its
predicted ~13-16% CI shrink on cross-family pairs (MDE 3.9→3.4m) and ~1%
within-family — real but not enough to resolve a −2.2m sym deficit (that
needs ~4× the holdout). Secondary lens, stated plainly: every sym config sits
−2.2 to −2.4m with p_better ≤ .076, and sym_16 is bare-significant for v6 in
both modes (block CI [−4.49,−0.22]) — the direction consistently favors
consistency conditioning; the magnitude is below the 80%-power bar.

## 5b — five new conditioning axes (train-grid selected, one verdict each)

| axis | symmetric? | selected | Δ vs v6 (pair MDE) | vs own sym control | verdict |
|---|---|---|---|---|---|
| a lineup continuity | YES | h24 γ=2 | −2.82m (8.29) | +1.68m vs sym_24_nc (9.74); −0.40m vs sym_24+yearcont | INSIDE NOISE FLOOR |
| b opp quality of anomaly | no (anomaly-conditioned) | m=1.67 | −0.85m (2.48) | — | INSIDE NOISE FLOOR |
| c anomaly margin | no | k=1.0 | −0.55m (2.89) | — | INSIDE NOISE FLOOR |
| d event-class fade (sym) | YES | h20 m_ewc=0.8 | −1.90m (3.84) | +0.27m vs sym_20 (0.87) | INSIDE NOISE FLOOR |
| d on top of v6 | addon symmetric | m_ewc=0.8 | **+0.24m** (0.93), p .75/.68 | — | INSIDE NOISE FLOOR |
| e patch/map-pool fade (sym) | YES | h24 γ_p=0.7 | −5.51m (8.03), p .022/.001 | −3.09m vs sym_24 | INSIDE NOISE FLOOR |
| e on top of v6 | addon symmetric | γ_p=0.7 | −5.46m (7.86), p .022/.002 | — | INSIDE NOISE FLOOR |

Readings, per pre-registered falsifiers:
- **a**: lineup continuity is real but redundant — it roughly recovers what
  year-boundary continuity already provides (+1.68m over the no-continuity
  control, −0.40m vs the year-continuity sym). It does not close the sym
  family's gap to v6. Falsifier ("no gain over own control") did NOT fire;
  the axis just isn't additive.
- **b, c**: train gains (+0.40m, +0.96m) failed to transfer (holdout −0.85m,
  −0.55m). Both axes dead on holdout at this power.
- **d**: the one positive-direction result. As an addon to v6 (+0.24m) it is
  positive in ALL SIX subpopulations (S1 +0.65, S2 +0.52, S3 +1.03, S4 +0.37,
  S5 +0.27) — the only config in the wave with that property — but every cell
  is deep inside its noise floor. Flag for re-test when the corpus grows;
  coordinates with agent:context 3a (solve-weight side).
- **e**: the sharpest train-holdout reversal of the wave (best train LL of ANY
  config, 0.64436 vs v6's 0.64823; then −5.5m on holdout, bare-significant
  negative both modes). The mechanical rotation-fade axis fits 2023-24-era
  noise. Falsifier fired; axis killed at γ≤0.7 (γ=0.85 was mild: −2.3m).

**Symmetric vs asymmetric, answered plainly (pre-committed wording):** all
symmetric axes sit inside the noise floor with Δ < 0 — *"unresolved either
way at n=1217: asymmetry is not demonstrably needed, nor demonstrably
better."* No vindication of the operator's objection, and no proof of the
asymmetry either; the uniform −2 to −2.8m lean of five independent
symmetric configs (plus sym_16's bare significance) is the evidence that
currently exists for keeping consistency conditioning.

## 5c — performance-based form (all on v6 rdiff, within-family)

Old-frame reference: b_form(wr,HL3) = −0.0872, −0.25m n.s. All fits
Nelder-Mead (tight tolerances after an initial near-flat-objective stall —
first pass at default tolerances left b_form at the x0 vertex for the
side/player features; documented in logs, refit properly).

| form definition | b_form (train) | train gain | holdout Δ vs v6 (MDE) | verdict |
|---|---|---|---|---|
| wr HL3 (replication) | −0.118 | 0.09m | −0.66m (1.23), p .06/.05 | INSIDE NOISE FLOOR |
| wr HL5 / HL8 | −0.130 / −0.308 | 0.04/0.06m | −0.31m / −0.27m | INSIDE NOISE FLOOR |
| rd-margin HL3 (selected) | −0.0059 | 0.08m | −0.82m (1.20), p .027/.009 | INSIDE NOISE FLOOR |
| rd-margin HL5 / HL8 | −0.0064 / −0.0143 | — | −0.43m / −0.40m | INSIDE NOISE FLOOR |
| side-conditional HL5 (coverage 1149/1217 holdout) | −0.566 | 0.25m | −0.98m (2.38) | INSIDE NOISE FLOOR |
| player R2.0 HL5 (coverage 1129/1217) | −2.681 | 2.09m | −5.24m (6.04), p .007/.004 | INSIDE NOISE FLOOR |
| combined (b_rd, b_side) | +0.006 / −0.795 | — | −0.59m (2.51) | INSIDE NOISE FLOOR |

Verdict: **performance-defined form adds nothing.** Every definition fits a
negative (mean-reverting) coefficient on train — replicating v7's sign — and
every one scores ≤ v6 on holdout; the richest feature (player R2.0) overfits
hardest (train +2.09m → holdout −5.24m, bare-significant negative). The
ratings already price form correctly by smoothing it.

## 5e — subpopulation panel (stats/decay_subpops.json, every config)

Holdout ns: S1 post-roster-change 731, S2 post-patch 730, S3 post-break 130,
S4 within-event day-2+ 1118, S5 quoted-band(20-55c, p_fav6 ≤ .80) 1189. No
config wins any subpop at bucket MDE. Notable cells (hypothesis fodder only):
sym_20/24 are +1.5/+2.1m vs v6 on S3 post-break (n=130, bucket MDE ~18m) —
consistency conditioning may be weakest right after long breaks. S1 mask
validated: my expanded-corpus matches_since_change recompute agrees 100%
(165/165) with the lineups agent's column on orgs untouched by corpus
additions; the 77% overall agreement is pure interleaving effect.

Instrument note: bucket tags use family-MDE scaling (pre-registered); for
rot_on_v6 and form_player5 the within-family 1.773m understates their actual
pair σ (empirical 7.9/6.0m — rating-perturbing addons behave cross-family),
so their "WORSE" bucket tags overstate resolution; the headline rule keeps
both INSIDE NOISE FLOOR with bare-significant negative direction.

## Both units

Largest candidate effect |Δ| = 5.5m ⇒ δ_logit ≈ 0.016 ⇒ expected maker-ROI
move ≈ ±0.6pp on the quote-margin ladder; the only positive effect (+0.24m)
translates to +0.0007 logit ≈ +0.03pp ROI. **No axis in this phase moves the
quoting surface.** (Ladder is one-sided from the operating point; negative
Δs quoted as the ROI equivalent of their magnitude, foregone.)

## Bottom line

v6 consist(20,12) survives Phase 4 unchanged. The re-race says "ties" at 2-4×
better resolution than the old frame. No symmetric axis dethrones the
asymmetry; none of the operator's two hypotheses (more recency,
outcome-symmetric decay) finds support above the noise floor, and the
consistent negative lean of every recency-increasing variant is the closest
thing to an answer the data gives. One thread worth keeping warm:
event-class-conditioned fade on top of v6 (+0.24m, positive in all six
subpops) — re-test at the next corpus expansion.

Artifacts: stats/decay_rerace.json, decay_axes.json, decay_form.json,
decay_subpops.json, decay_curves.json · logs/decay.log · scratch/decay/.
