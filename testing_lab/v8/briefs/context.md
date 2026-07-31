# agent:context — Phase 2: event context, incentives, preparation asymmetry

Read briefs/wave2_common.md first; it is law. Scope: does event context —
seriousness, prep/footage asymmetry, stakes — carry probability information
the rating solve or prediction layer should use? Each mechanism tested
separately, pre-registered.

Motivating observable (agent:lineups): 30d-modal stand-in rate 6.7% VCT vs
23.3% EWC-class, concentrated in qualifiers/evo (CES 70.8%, quals 21.0%, EWC
main 6.25%). The seriousness effect, if real, lives in the qualifier tier.

## Experiments (preregister each: mechanism, sign, size, falsifier)
1. **3a Lineup-conditioned event weighting.** Solve-weight EWC-class games by
   lineup integrity (full modal five = weight 1, stand-in lineups down to w0)
   vs blanket EWC down-weight vs v6 baseline. Fit w0 (and any blanket w) on
   train only via the engine's w_custom; β refit per config. Judge on holdout.
2. **3b Footage exposure & prep.** Per team-match walk-forward features:
   official maps in trailing 14/30d, days since last official, days since last
   LAN (intl event), and the A−B differentials. Two deliverables:
   (a) prediction-layer term: logit adjustment fit on train, scored holdout;
   (b) THE DECOMPOSITION: refit b_form (v7's decayed-form term, run_v7_stage2
   machinery reproduced on the expanded frame) with and without exposure
   controls; report both coefficients side by side. If exposure absorbs the
   form penalty, v7's "form is mean-reverting" was scouting in disguise —
   say so plainly; if not, say that plainly.
3. **3b-adjacency** Deep-run-at-previous-international → next-international
   underperformance, controlling for rating: team-match level across all
   Masters→next-intl adjacencies in the corpus (incl. 2025 Toronto→Riyadh EWC,
   2026 London→EWC). Report the CI honestly; "untestable at this n" is an
   acceptable published answer.
4. **3c Stakes.** Derivable-only flags (preregister exactly which): stage from
   match_name (groups/playoffs/GF exists), elimination matches (lower-bracket
   / knockout naming), dead rubbers only where derivable from group standings
   reconstruction — if not cleanly derivable, declare the subset untestable
   rather than approximating. Test as solve weights and as prediction-layer
   variance (shrink-to-0.5) terms.
5. **3d Learned event-class solve weights.** Replace hand-set {playoffs 1.6,
   Champions 2.0} with jointly fitted per-class weights {vct_regular,
   vct_playoffs, champions, masters, ewc_offseason} on train (walk-forward
   inside train for the fit objective), CI via crn bootstrap. Report the
   fitted EWC-class weight — this quantifies the operator's intuition either
   direction. Compare holdout LL vs v6's hand-set weights.
6. **3e Context-conditional confidence, mechanism version.** ONE global
   shrinkage-toward-0.5 term: delta_logit = -k * X where X = observable
   (lineup delta, prep asymmetry, event class dummy ONLY via its observables),
   1-2 coefficients TOTAL, fit train, scored holdout, with the EWC-class
   bucket LL (0.6918 baseline) reported before/after. A free per-event-class
   parameter is the falsifier of the mechanism story — if only that works,
   report "no mechanism found."

## Outputs (yours alone)
- stats/context_weights.json (fitted class weights + CIs, chart-ready)
- stats/context_exposure.json (b_form ± exposure forest-plot data)
- stats/context_seriousness.json, context_stakes.json, context_shrink.json,
  context_adjacency.json
- phase2_context.md (prose: each mechanism, verdict, both units, MDE context)
- preregister.context.md, logs/context.log, scratch/context/

Done when: every experiment has preregistered prediction + measured outcome +
verdict (WIN / INSIDE NOISE FLOOR / DEAD), fitted EWC weight published with
CI, and the exposure-vs-form decomposition answered in one quotable sentence.
