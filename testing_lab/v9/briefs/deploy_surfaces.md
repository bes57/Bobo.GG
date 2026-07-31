# agent:deploy-surfaces — every site probability becomes v6 (stage 2)

Stage 1 is live: data/rating_timeline*.json are v6 ratings and
data/site_model.json is the single source of truth
(beta 0.128512, xregion_offsets {Americas +2.0966, EMEA +1.9045,
Pacific +2.4482, CN 0}, region_priors, gf_upper_logit 0.25, b_pick 0.0937).
Reference math: trading_model/predict.py — series_probability /
map_probability are THE spec. Your job: wire every surface to the snapshot
and remove the old constants everywhere.

## Surfaces (from the deploy inventory)
1. **MapElo.py backend** (~L7242-7515, the past-match/upcoming closed-form
   path): _tl_beta 0.170 → snapshot beta (load site_model.json once,
   hot-reload by mtime like other data files); REPLACE the backend
   intl-gated exp/cn_dog offset logic with predict.py's xregion adjustment
   (adj = off[reg_a] − off[reg_b] on the map logit, same-region ⇒ 0,
   applied at ALL matches, no event gating — that is v6). GF logit stays
   0.25 via snapshot. Unknown/new orgs: region prior from snapshot.
2. **MapElo.py frontend JS** (~L8514-8551 SNAP_BETA/INTL_EXP_BONUS/
   CN_DOG_OFFSET and any other consumer, incl. the L9927 "keep hardcoded
   SNAP_BETA=0.22" comment area — audit every beta in the file): inject the
   snapshot values server-side into the template (the pages already
   template data in); replace the favExp/dogExp + CN-dog JS logic with the
   xregion-offset delta. Kill dead constants; leave a comment pointing at
   site_model.json.
3. **Veto-MC page**: per-map veto walkthrough machinery STAYS (product
   feature). The HEADLINE series probability it displays must equal the v6
   closed form on overall ratings (predict.py math incl. b_pick for
   map-level displays where a picked-map probability is shown:
   map_probability semantics). Ensure the MC's map-level inputs use overall
   rating + b_pick, not per-map splits stacked WITH pick bonus double-count
   — where per-map split ratings are displayed as content, they stay
   content; probabilities quoted as "win chance" come from the v6 surface.
4. **BobosHome.py** (~L360 hardcoded 0.17 in team-profile upcoming +
   anywhere else `0.17` appears as beta; grep the whole file): read
   site_model.json (import-light: json load with mtime cache), apply beta +
   xregion offsets + the same bo3/bo5 closed form. The alpha home's
   upcoming probabilities come through the hub payload — verify they are
   snapshot-driven after your MapElo change (they are computed backend-side;
   confirm and note where).
5. **Sweep for stragglers**: grep the repo (site code only: MapElo.py,
   BobosHome.py, EventLeaderboards.py, MatchDataExplorer.py, static/*.js,
   any gen that renders probabilities) for 0.17/0.170 beta usages,
   intl_exp/cn_dog references, and stale "production model" language on
   user-facing strings; update or kill each with a one-line note in your
   log. testing_lab/ and trading_model/ are NOT yours except the README
   note below.
6. **Docs**: trading_model/README.md + FINDINGS.md top: one-line update —
   "2026-07-30, operator decision: v6 deployed to the public site
   (site_model.json); site and bot now share the champion." Do not rewrite
   history elsewhere.

## Gates (hard, before declaring done)
A. **predict.py parity**: script testing_lab/v9/scratch/deploy/
   surface_parity.py — for 40 team pairs (sampled from current ratings,
   mixed regions + formats incl. bo5_gf both uppers + an unknown-org case):
   site backend probability == predict.py series_probability(m from
   site_model.json-equivalent inputs... note predict.py loads
   trading_model/model_snapshot.json — for parity, construct its `m` dict
   FROM data/site_model.json + current timeline ratings) to ≤1e-9. The
   frontend JS math: verify by executing the injected constants path in
   node/python-replica for 10 pairs to ≤1e-6.
B. **Visual**: restart Flask, then headless-verify (isolated Chrome
   profile) /mapelo/ (hub leaderboard shows v6 scale), /mapelo/modern/,
   / (alpha home upcoming %), a team profile, and the veto sim page — all
   200, charts render, probabilities present, no NaN/undefined; screenshot
   evidence saved to your scratch. Flask LEFT RUNNING (standing rule).
C. Rebuild anything the surfaces consume that you changed the shape of
   (none expected — snapshot is additive).

## Outputs (yours alone)
- MapElo.py, BobosHome.py edits (+ any static/*.js if implicated)
- trading_model/README.md + FINDINGS.md one-liners
- testing_lab/v9/stats/deploy_surface_parity.json (gate A evidence)
- testing_lab/v9/logs/deploy_surfaces.log (+ straggler sweep list)
- scratch: testing_lab/v9/scratch/deploy/ (parity script, screenshots)
No git. Return ≤300 words: parity numbers, surfaces changed (file:line
list), straggler sweep count, visual-verify results, anything left.
