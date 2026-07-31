# preregister.compose.md — Wave 3 stacks (agent:compose)

Written 2026-07-28, BEFORE any train fit or holdout scoring by this agent.
Frame: testing_lab/v8/data/frame_expanded/series.csv, sha256 verified against
crn.json `frame_expanded` at every entry point (abort on mismatch). Holdout =
date > 2024-12-31 (n=1217), train n=841. All resampling via referee
paired_bootstrap_crn on crn.json (iid + block_event). Baseline = v6 champion
replayed on the expanded frame (consist 20/12, PO 1.6, champ x2 exact-shape,
ridge 0.5 + region-prior 1.5, rd |.|^0.75*2.5, year continuity), train-fit
β=0.1152, holdout LL 0.64216, from scratch/bias_h3/model_probs.npz `p_v6`
(cross-checked against scratch/context/b0.npz to ≤1e-9 before use).

Components allowed (compose brief): 3e stand-in shrink (context), event-class
fade (decay axis d, on-v6 attachment), n_eff-gated 5d state-space (bias-h3).
Nothing else. No killed component resurrected; 1b (calendar noise) is NOT used.

## Exactly three stacks

### S1 "gate5d" — the named h3 shape, scored for the first time
- p_S1[i] = p_ss_5d[i] if neff_min_5d[i] < 12 else p_v6[i].
- neff_min_5d = min(R/v_w, R/v_l) from the 5d roster-typed SS filter's
  pre-match posterior variances (model_probs.npz vw_5d/vl_5d); R = h3's train
  Var(y) identification constant (11.2933; n_eff invariant to its value).
- Threshold 12 is the ONLY 5d hard gate that beats v6 on TRAIN
  (h3_ensemble.json: 0.646927 vs v6 0.648226, +1.30m train; neff<8/<5/<3 are
  all train-worse). Gate membership verification: gated counts must equal
  h3's published 330 train / 178 holdout; if not, recompute R exactly from
  the engine game corpus (train Var(y)); abort loudly if still unequal.
- Free parameters: none new. β_5d and the 5d q's are h3's train fits; β_v6 is
  the v6 train fit. (No solve constant changes ⇒ no β refit obligation.)
- Mechanism: SS posterior uncertainty helps exactly where data are thin;
  h3's cold buckets (either <10 prior maps +71.7m, <30 +28.0m, holdout,
  recorded as full-5d diagnostics, never as a scoring of this composite).
- Predicted sign +; predicted effect +0.5..+2.5m overall (train +1.30m,
  concentrated on 178/1217 rows). Predicted promotion verdict: HOLD (the
  effect should be real but below the cross-family bar).
- Falsifier: holdout ΔLL ≤ 0 overall, or ΔLL < 0 restricted to the 178 gated
  rows ⇒ the cold-row advantage was corpus-era-specific; component dead for
  stacking and said so.
- G1 MDE regime: cross-family 5.889m (h3 precedent for SS-vs-v6 composites);
  empirical pair-MDE (2.8016·SD(d)/√n, raw + CUPED-CV) quoted alongside.

### S2 "fade+shrink" — the two engine/surface survivors, jointly refit
- Ratings: rdiff_fade from scratch/decay/probs/eclass_on_v6_m0.8.npz — v6
  consist(20,12) with HL multiplier m_ewc=0.8 on ewc_offseason-class games
  (decay axis-d event set, frozen; m=0.8 was the train-selected multiplier in
  the wave-2 grid; the on-v6 attachment is the surviving shape, +0.24m,
  positive in all six subpops). m_ewc is NOT refit here (frozen constant;
  refitting it would require new solver runs outside the surviving artifact).
- Surface: z = β·rdiff_fade·exp(−k·X1), X1 = (1−integ_w)+(1−integ_l) (context
  3e frame_features.csv, 0 NaN), p = house series closed form by fmt.
- Joint train fit: (β, k) by Nelder-Mead (x0=(0.13,0), xatol 1e-5, fatol
  1e-9) on train rows with valid rdiff_fade — the exact 3e procedure, run on
  the fade base. This is the stack's β refit (scale-bound rule satisfied).
- Predicted sign +; predicted k ∈ +0.2..+0.5 (3e fit +0.347 on B0); predicted
  effect −0.2..+0.8m overall (components +0.24m and −0.40m alone; the case
  for the pair is the shared EWC-bucket mechanism: +3.46m and +0.52..+0.65m
  subpop gains), EWC full-class bucket predicted positive. Predicted
  promotion verdict: HOLD.
- Falsifiers: fitted k ≤ 0 on the fade base (mechanism does not survive
  composition — S2 still scored, but the component is declared incoherent);
  holdout ΔLL ≤ −1.773m (stack is worse than v6 beyond within-family noise
  ⇒ pair dead).
- G1 MDE regime: within-family 1.773m (v6-family variant; no SS content).

### S3 "full" — S2 base with the S1 gate on top
- p_S3[i] = p_ss_5d[i] if neff_min_5d[i] < 12 else p_S2'[i], where p_S2' uses
  S3's own (β,k): refit jointly on train to minimize the COMPOSITE train NLL
  (equivalently: the 3e fit restricted to non-gated train rows, since gated
  rows don't depend on (β,k)). Same frozen gate, same frozen m_ewc.
- Predicted sign +; predicted effect ≈ additive: +0.5..+3.0m. Predicted
  promotion verdict: HOLD (below the 5.889m cross bar).
- Falsifier: S3 ΔLL < min(S1, S2) − 0.5m (anti-synergy: stacking hurts), or
  ΔLL ≤ 0 (stack concept dead).
- G1 MDE regime: cross-family 5.889m (contains the SS component).

## Judging (identical for all three; ONE holdout scoring each, ever)
- referee.promotion_gate(candidate, v6, mde = regime MDE above, frame,
  rdiff_ref = v6 rdiff, games = engine corpus for elite/floor + form masks):
  G1 (mean ΔLL ≥ MDE AND p_better ≥ 0.95 in iid AND block CRN), G2 max|team
  bias| strictly < v6's (expected 0.1478, min_n 25), G3 pre-committed bucket
  floors. Verdict published clause by clause, whatever it is.
- Both units everywhere: milli-LL + expected_roi_of_dll on the quoting surface.
- CV: CUPED control variate on centered l_v6 (decay judge recipe, banked in
  stats/variance_reduction.json) — CV boots + CV pair-MDE reported next to
  raw in every CI. Point estimates unchanged by CV.
- Caterpillar (full per-team bias table) + full bucket panel per stack; the
  PRX/NRG unexplained residual (v6: −10.2/−9.1pp) reported per stack, not
  chased; if it persists it is named future work.
- Gated-subset detail for S1/S3 (n=178 rows): ΔLL on gated and non-gated
  splits (diagnostic accompanying the single scoring, not extra scorings).
- No iteration after seeing holdout numbers. Whatever comes out, publishes.
- Multiple-looks tally: every holdout scoring by every agent harvested from
  stats/*.json + logs/*.log, published in stats/compose_looks.json with
  these three additional looks counted, and the family-wise context stated
  next to every p-value quoted in phase_compose.md.

## Program verdict rule (fixed in advance)
If any stack passes G1∧G2∧G3 ⇒ "stack X clears the bar." Else if all three
fall inside their noise floors or below ⇒ "v6 stands because no preregistered
stack of the surviving components beats it at the promotion bar on the
expanded holdout." If results are mixed-sign / unresolvable ⇒ "we could not
tell," in those words.

## Outcomes (appended AFTER the one-shot scorings — same resolution win or lose)

Scored 2026-07-28 22:36 (one pass, sentinel scratch/compose/SCORED). Frame
sha verified; gate counts reproduced h3 exactly (330 train / 178 holdout).

- **S1 gate5d: +1.958m** — INSIDE NOISE FLOOR (cross MDE 5.889m; pair-MDE
  3.08m raw / 3.06m CV; iid CI −0.15..+4.21m, block −1.19..+4.94m; p_better
  0.966/0.888). Predicted +0.5..+2.5m, sign + → **prediction confirmed in
  range**; falsifier did NOT fire (overall > 0 and gated rows +13.39m > 0;
  non-gated identically 0). Promotion gate HOLD: G1 fail (sub-MDE, block
  p<0.95), G2 fail (max|bias| 0.1491 vs 0.1478 — worsened, prediction of a
  clean HOLD was right for the wrong clause count), G3 fail (huge-gap
  −8.86m). ROI +0.16pp at δ_logit +0.0043.
- **S2 fade+shrink: −0.140m** — INSIDE NOISE FLOOR (within MDE 1.773m; iid
  CI −1.89..+1.64m; p 0.43/0.47). Fitted k=+0.352, β=0.1252 → k-range
  prediction confirmed; effect prediction (−0.2..+0.8m) confirmed at the low
  end. EWC full-class +3.15m (predicted positive — confirmed). Neither
  falsifier fired (k>0; ΔLL > −1.773m). Gate HOLD (G1/G2/G3 all fail;
  G2 0.1548; G3 domestic EMEA −4.41m, favorite [0.7,0.8) −7.63m).
- **S3 full: −7.874m** — **LOSS** beyond the cross floor (iid CI
  −14.3..−1.5m, p 0.008/0.043; CV CI −12.9..−2.9m). **Anti-synergy
  falsifier FIRED** (S3 ≪ min(S1,S2) − 0.5m): the declared subset refit
  (non-gated train, n=497) drove k to 1.87 and produced the wave's best
  composite train NLL (0.64469) with the worst holdout (0.65004) — a train
  mirage by the program's own definition; non-gated rows −11.5m. The
  additive-effect prediction (+0.5..+3.0m) was **wrong**; recorded at full
  resolution. Gate HOLD (18 major bucket regressions).
- PRX/NRG residual persists under every stack (−10.1/−9.3, −10.1/−9.3,
  −11.9/−11.4 vs v6 −10.2/−9.1) → named future work.
- Looks tally published (stats/compose_looks.json): 163 primary looks
  program-wide (+227 sweep-checkpoint, +8 EM-iteration = 398 recorded
  holdout numbers); compose added exactly 3.

**Program verdict (preregistered wording): v6 stands because no
preregistered stack of the surviving components beats it at the promotion
bar on the expanded holdout.**
