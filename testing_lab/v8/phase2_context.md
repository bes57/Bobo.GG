# Phase 2 — Event context, incentives, preparation asymmetry (agent:context)

2026-07-28. Preregistered in `preregister.context.md` (predictions written
before any experimental run; outcomes appended there at full resolution).
Frame: `v8/data/frame_expanded/series.csv`, sha256 verified against
`crn.json` (`ff772d41…`); n=2058, train 841 (827 valid), holdout 1217.
Baseline **B0** = v6 champion replayed on the expanded frame (consistency
20/12 games-decay, stage-playoffs ×1.6, Champions ×2.0, β train-fit):
ll_holdout **0.64216**, β 0.1152. Judging: CRN paired bootstrap (iid +
event-block), pair-MDE **1.773m within-family** (5.889m cross;
`stats/power_mde_expanded.json`) — every candidate here is a v6-family
variant. Both units quoted: milli-LL and expected maker-ROI via
`referee.expected_roi_of_dll` (scale: 1.77m ≈ 0.15 ROI pts, 5.7m ≈ 0.48 ROI
pts; the ladder is one-sided so negative ΔLL clamps to 0.0 — magnitudes
quoted as |Δ|). Data: 335 corpus-addition matches lacked lineup features —
topped up in scratch with the lineups agent's definitions verbatim
(cross-check 20/20 exact; 0 gaps; 7/400 covered sides would shift
overlap_modal under full-corpus history, their rows kept canonical).

## Verdict table

| # | Mechanism | Verdict | Overall Δ (milli / ROI) |
|---|-----------|---------|--------------------------|
| 3a | Lineup-conditioned EWC solve weight | **DEAD** (train falsifier fired) | n/a — train picks w0=1.0 = B0 |
| 3a′ | Blanket EWC solve weight | **INSIDE NOISE FLOOR** | −0.05m / 0.00 |
| 3b-a | Footage/prep prediction term | **INSIDE NOISE FLOOR** | −0.28m / 0.00 (EWC bucket −10.9m) |
| 3b-b | Form-vs-exposure decomposition | **ANSWERED** (see quotable) | form5 alone −0.31m (floor) |
| 3b-adj | Deep-run → next-intl slump | **UNTESTABLE AT THIS N** | c=+0.021, CI [−0.088,+0.137] |
| 3c-A | Elimination solve weight | **INSIDE NOISE FLOOR** (dir. wrong) | −0.54m / 0.00 |
| 3c-B | Elimination variance shrink | **INSIDE NOISE FLOOR** | −0.37m / 0.00 |
| 3c′ | Dead rubbers | **UNTESTABLE** (declared, not approximated) | — |
| 3d | Learned event-class solve weights | **DEAD** | −5.74m / |Δ|≈0.48 ROI pts, iid CI [−10.0,−1.2]m |
| 3e | Stand-in confidence shrink (X1) | **INSIDE NOISE FLOOR**; bucket lead | −0.40m / 0.00; EWC class +3.46m |
| 3e′ | Prep-asymmetry shrink (X2) | **DEAD-FLAT** | −0.01m |

## The one-sentence decomposition answer (brief deliverable)

**With footage-exposure controls added, the v7 form coefficient collapses
from −0.130 to +0.012 at HL5 (109% absorbed — a sign flip to zero) and from
−0.118 to −0.049 at HL3 (58% absorbed): at HL5, v7's "form is
mean-reverting" was scouting/footage-exposure in disguise; at HL3 about
half of it was — and neither term survives the expanded holdout anyway
(form5 alone −0.31m, all variants inside the 1.773m floor; the old-frame
+1.3m form gain does not replicate).** Forest data:
`stats/context_exposure.json`; corr(Δform5, {Δmaps30, Δlog dso, Δlog dsi})
= {+0.114, −0.142, +0.169} on train.

## What the mechanisms said

- **3a seriousness (lineups)**: train NLL is monotone WORSE as stand-in
  games are down-weighted (0.64823 at w0=1 → 0.65052 at w0=0); the
  preregistered falsifier fired before holdout was ever touched. The
  blanket grid agrees from the other side: train argmin at the 1.2 UP-weight
  edge, holdout −0.05m. EWC-class games are not less informative per solve
  weight than VCT games, stand-ins or no.
- **3d learned weights**: the joint fit {vct_po 0.73, champions **0.001**,
  masters 1.88, ewc **1.018**} improves train by 4.2m and loses **−5.74m**
  on holdout (iid CI [−10.0,−1.2]m) — in-sample regime memorization; the
  champions collapse is even "identified" in-sample (86% of CRN
  argmin-bootstrap resamples at the grid floor) and still anti-validates.
  **Fitted EWC-class weight = 1.018, 95% CI [0.4, 3.4] ⊇ 1.0** — and the
  v6-conditioned 1-D sensitivity is monotone UP to the 3.4 grid edge
  (argmin-boot CI [0.6, 3.4]). The operator's "down-weight EWC" intuition
  is unsupported in every parameterization tried; if anything the in-sample
  pull is upward. Hand-set {1.6 playoffs, 2.0 Champions} stands.
- **3c stakes**: both tests fit the SAME direction on train — elimination
  matches are noisier (solve wants w_elim 0.7 at the grid edge; prediction
  layer fits a_elim −0.348, a 30% confidence shrink) — and both land inside
  the floor on holdout (−0.54m / −0.37m). Dead rubbers: declared
  untestable (standings reconstruction not derivable; approximation
  forbidden). No usable stakes signal at n=1217.
- **3b prep/footage**: the exposure term train-fits (+0.118 per 10 trailing
  maps, +0.073 rest, −0.087 intl-recency) and transfers nowhere (−0.28m,
  EWC bucket −10.9m). Its real value was the decomposition above.
- **3b-adjacency**: n=57 both-attended series over 7 Masters→next-intl
  pairs (incl. Toronto→Riyadh, London→EWC). c = +0.021, Wald
  [−0.081,+0.124], CRN boot [−0.088,+0.137] — sign opposite the fatigue
  story, CI spans zero. Published as untestable at this n.
- **3e context-conditional confidence**: the ONE mechanism with the right
  sign that survives its own falsifier — z′ = z·exp(−k·X), k_standin =
  +0.347: overall −0.40m (floor) but the EWC-class bucket improves
  **+3.46m** (0.67224→0.66878, n=291; legacy-2026 bucket 0.68633→0.68506,
  +1.26m, vs published old-frame 0.6918). The free class-dummy falsifier
  (k=+0.86) gains overall (+0.50m, floor) but makes the full EWC bucket
  −3.76m WORSE — its gain is β re-inflation on non-EWC rows, not in-bucket
  skill (though it does help the 2026-only legacy slice +3.34m; bucket-
  definition sensitivity flagged). Verdict: not promotable, but this is the
  cleanest Phase-5 lead Phase 2 produced.

## Synthesis

Event context carries almost no probability information the v6 pipeline
doesn't already have. Every solve-side lever (lineup integrity, blanket
class weights, stakes, learned weights) either fails in-sample or fails
walk-forward; the only mechanism-consistent residue is that **stand-in
lineups deserve a confidence shrink in the off-season bucket** — real
enough to improve the bucket it targets (+3.5m), too small to move the
1217-row holdout (−0.4m overall, floor 1.77m). And the seriousness prior
runs backwards: in-sample, EWC-class games want MORE solve weight, not
less; elimination games want LESS. Neither survives out-of-sample.

## Artifacts

`stats/context_weights.json` (fitted class weights + CIs + sensitivity,
chart-ready), `context_exposure.json` (forest data), `context_seriousness
.json`, `context_stakes.json`, `context_shrink.json`, `context_adjacency
.json`; `preregister.context.md` (predictions + appended outcomes);
`logs/context.log`; scratch (incl. `lineup_topup.csv`) in
`scratch/context/`. No writes outside declared paths; no network; no
holdout fitting; market data untouched.
