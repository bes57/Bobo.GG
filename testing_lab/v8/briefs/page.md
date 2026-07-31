# agent:page — Wave 3: the v8 Lab page + written report + ledger revisions

Scope: publish the program, per the operator brief's §10, in the house style,
with the adversary's amendments applied. You are the reporter — you may read
everything under testing_lab/v8/ including phase*.md and adversary_report.md.

## Hard rules
1. **No hardcoded numbers in the HTML.** Every figure/table/chart reads from
   JSON under testing_lab/v8/stats/ (add derived JSONs there if a chart needs
   reshaping — one writer: you). Each chart gets a download link to its JSON.
   Charts render client-side (follow gen_report.py's inline-JSON + JS
   pattern; Chart.js is already the house library).
2. **The adversary's report publishes verbatim** (adversary_report.md) in a
   distinct callout section, including the parts that overturn program
   claims. No softening, no editorializing around it.
3. **Amended verdicts are the published verdicts.** The §0 headline set:
   - v6 stands; no stack cleared the gate (compose_gate.json), co-signed by
     the adversary.
   - CORRECTION over Phase 0's prose: four v7 configs (sym_6, sym_8,
     surprise_12_20, boxexp_c3) WERE distinguishable — as losses — under the
     program-standard 5.889m floor; the near-ties (sym_20, consist_16_10)
     remain unresolved (adversary CONFIRMED-FLAW; reflect it in the §1
     ladder re-plot and its caption).
   - Demoted by adversarial review (present as noise-floor leads, not wins):
     3e stand-in shrink's EWC bucket; event-class fade's subpop positivity;
     5d's 7.4-vs-24.8 half-life contrast; compose S1's gated gain.
   - The ONE surviving lead: H3 cold-start (<10 prior maps) +71.7m (n=39,
     floor 32.9m, drop-5% +53.1m, jackknife +66.0m) — with its caveats
     (4 events; parent model loses overall).
   - THE HOLDOUT IS SPENT: 398 recorded holdout numbers; future train-only
     claims are unfalsifiable on this frame. This is a banner-level
     statement on the page, and it drives the tripwires section's "what
     data next" list (prospective logger accumulation; the deferred 194
     prefranchise regional events; 2026 S2/Champions as they settle).
4. Site mechanics, house-conformant: discover how TestingLab.py serves
   reports (_REPORTS_DIR, /testing/report/<name>) and how the six existing
   pages' shared tab strip is built. Add the "v8 Lab" page behind the same
   password gate at the house URL shape, and add its tab to the six existing
   pages by STRING-PATCHING their HTML nav strips — do NOT regenerate the
   old pages (their input JSONs have drifted since build; the
   vm-snapshot-drift incident is the precedent). Write a generator script
   testing_lab/gen_v8_report.py in the style of the existing gen_*.py so
   the page is rebuildable.

## Sections (map to §10 of the operator brief; JSON sources)
§0 Verdict + stat cards (compose_*.json, power_mde_expanded.json,
   h2/h1/h3/h4_bias_caterpillar.json for max|bias|, autopsy_pnl.json).
§1 Power: MDE curve (power_mde*.json), v7 ladder re-plot with the noise
   floor shaded AND the adversary's four distinguishable-losses marked
   (v7_reclass.json + adversary_report.json), ledger reclass table
   (ledger_reclass.json), variance reduction (variance_reduction.json).
§2 Corpus: coverage timeline + diff table + verification
   (corpus_diff.json, corpus_blocks.json, verification_report.json),
   prefranchise row with its deferral note.
§3 Context: fitted class weights w/ CIs vs hand-set marks
   (context_weights.json), seriousness observables (lineups_coverage.json,
   modal5 data), exposure-vs-form forest (context_exposure.json),
   adjacency honesty panel (context_adjacency.json).
§4 Decay: w(g) overlay (decay_curves.json), re-race table with MDE header
   (decay_rerace.json), axes table (decay_axes.json), subpop small
   multiples (decay_subpops.json) — with the adversary's overlap caveat in
   the caption.
§5 Bias: caterpillar v6-vs-candidates (h*_bias_caterpillar.json,
   compose_stacks.json), one subsection per mechanism with its diagnostic
   (h1_censor_diag.json, h2_centrality.json, h3_process_noise.json +
   h3_neff.json, h4_dispersion_diag.json), reliability curves house-style.
§6 Buckets (compose_stacks.json buckets + decay_subpops.json), regressions
   explained in text beside any red bar.
§7 Autopsy (autopsy_*.json): equity waterfall, fill-conditional overlay,
   markouts, config-gap ladder, variance check — labeled clearly as the
   operator's implementation memo; one-week-sample caveat prominent
   (README rule 9 wording).
§8 Research log: preregistration-vs-outcome scatter — parse predicted vs
   realized from preregister.*.md outcome sections into a derived JSON;
   session timeline in the State-of-BenPom voice; adversary verbatim.
§9 Slate: upcoming matches with v6 snapshot prices (predict.py +
   data/upcoming_matches.json); NO v8 column — a note states no candidate
   was promoted; flag rule stays for the future.
§10 Tripwires + ledger: concrete thresholds/windows for every conclusion;
   do-not-retest additions from this program (H1 censoring premise, H2
   connectivity, H4 dispersion premise+link, learned event-class weights,
   patch fade, performance-form definitions, lineup-conditioned EWC
   down-weight, calendar-noise state-space, S3 anti-synergy) each with its
   kill evidence; the rejected→unresolved list from ledger_reclass.json as
   its own table. Emit stats/ledger_v8_updates.json.

## Also deliver
- v8_report.md at testing_lab/v8/ — the repo-mirror of the page (deliverable
  4), same amended verdicts, plain markdown.
- Visual conventions: house palette/typography, hover n's, 95% CIs, sparse
  bins merged (n<15), every table an n column, "inside the noise floor"
  shading wherever a comparison is unresolved. Where a result is not
  statistically resolved, the chart must show it.

## Outputs (yours alone)
- testing_lab/gen_v8_report.py + the generated page installed where
  TestingLab.py serves it + the route/tab wiring + six nav string-patches
- testing_lab/v8/v8_report.md, stats/ledger_v8_updates.json, any derived
  stats/page_*.json
- preregister.page.md (trivial — what you will and won't compute yourself:
  you compute NOTHING new statistically; you only reshape), logs/page.log

Done when: the page renders behind the gate with every §; numbers verified
spot-check-identical to their JSONs; nav updated on all six old pages
without regenerating them; v8_report.md mirrors; ledger JSON emitted.

Return ≤400 words: the page URL path, sections built, any JSON you had to
derive, nav-patch status, spot-check results.
