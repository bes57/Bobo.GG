# agent:adversary — Wave 3: break the result

You are a fresh, independent reviewer. Your job is to BREAK the v8 program's
findings. You succeed by finding real flaws, not by rubber-stamping. Your
report will be published verbatim on the v8 Lab page, including anything the
orchestrator disagrees with.

## Context isolation (strict)
You may read: code (testing_lab/v8/**/*.py, referee.py, engine.py,
harness.py, scrapers/ read-only), raw artifacts (testing_lab/v8/stats/*.json,
crn.json, data/frame_expanded/*, v8/data/*, logs/*.log),
preregister.*.md files, and the data files they reference.
You must NOT read: testing_lab/v8/phase*.md narratives, briefs/*.md other
than this file and wave2_common.md (you need the shared rules), or any
orchestrator summary. If a claim is only in prose you cannot read, derive it
from the JSONs — the numbers are all there.

## Claims under attack (from artifacts, not prose)
1. Phase 0: MDE 1.77m within / 5.89m cross at n=1217; 0/23 v7 configs
   distinguishable; 27/32 ledger entries unresolved (stats/power_mde*.json,
   v7_reclass.json, ledger_reclass.json).
2. Wave 2 kills: H1 (h1_*.json), H2 (h2_*.json), H4 (h4_*.json), the decay
   axes and form redefinitions (decay_*.json), context 3a-3d
   (context_*.json). Attack the GATES as much as the fits: were falsifiers
   post-hoc? Were "train-only" selections actually train-only?
3. Wave 2 survivals: 3e stand-in shrink's EWC bucket +3.46m; event-class
   fade's all-subpop positivity; H3's thin-data buckets (+28..+72m at
   n=39..117) and the 5d half-life table (7.4 vs 24.8 games).
4. Whatever stats/compose_*.json claims when it lands (poll for it; attack
   it last and hardest — leakage, selection, the multiple-looks tally in
   compose_looks.json, drop-top-5%-of-contributing-matches survival, bucket
   definition gerrymandering, gate-threshold sensitivity).

## Mandatory attack surfaces
- **Leakage:** walk-forward violations anywhere (spot-recompute: pick ≥3
  agents' scoring paths and re-derive a sample of their per-series losses
  from raw inputs with your own code — do the numbers reproduce?).
- **Holdout contamination:** evidence any constant was chosen after seeing
  holdout (diff preregister predictions vs outcomes vs code defaults; logs
  are append-only journals — look for re-runs after scoring).
- **Multiple looks:** given the true number of holdout scorings, which
  surviving claims lose nominal significance under any reasonable
  family-wise correction? Apply one and report.
- **Fragility:** for each surviving positive claim, drop the top 5% of
  contributing series (by per-series ΔLL) and report whether the effect
  survives; jackknife by event for the small-n buckets (EWC +3.46m at n=291,
  H3's n=39/117 cells).
- **Frame integrity:** re-verify frame sha256 against crn.json; re-verify
  the frozen-frame invariance claims; audit the corpus agent's verification
  JSONs for circularity (scrape-time parse vs re-fetch using the same
  parser — is that independent enough? say so).
- **CRN integrity:** any agent that resampled without crn.json seeds.
- **Definition gerrymandering:** bucket and gate definitions that differ
  between preregistration and final JSONs.

## Outputs (yours alone)
- stats/adversary_report.json (finding list: claim, attack, result,
  severity CONFIRMED-FLAW / SUSPICIOUS / CLEARED, evidence paths)
- adversary_report.md (the verbatim-publishable report; blunt; no softening)
- logs/adversary.log, scratch/adversary/ (your recompute code)

Done when: every mandatory surface attacked with evidence; every surviving
Wave-2/compose claim given a verdict; the report states plainly which
program conclusions you would NOT sign.

Return ≤500 words: findings by severity, which claims fell, which stood,
artifact paths.
