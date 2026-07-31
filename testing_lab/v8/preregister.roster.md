# Pre-registration — agent:roster (written 2026-07-28, BEFORE any computation)

Scope: roster-change adaptation REPORT (briefs/roster.md). Epistemic frame: the
adversary established THE HOLDOUT IS SPENT (398 recorded looks). Nothing below
is confirmatory or promotable. Structure: (a) case forensics, (b) descriptive
population atlas, (c) ≤4 preregistered EXPLORATORY holdout reads, tallied in
stats/roster_looks.json (successor tally to stats/compose_looks.json).

Frame: testing_lab/v8/data/frame_expanded/series.csv, sha256 verified against
crn.json `frame_expanded` before every script touches it (abort on mismatch).
Holdout = date > 2024-12-31 (n=1217). Baseline = the stored bias_h3 v6 build
(`scratch/bias_h3/v6_baseline.npz`, β=0.1152, holdout LL 0.64216, n=1217) —
reused, NOT a new look. Any v6 re-run here (for daily rating trajectories) must
reproduce that stored `p_all` to ≤1e-12 max abs diff or the script aborts.

## 1. Change definition (frozen before computing)

Source of lineups: `engine.Engine().lineups` — (org, match_id) → frozenset of
ProfileURLs fielded (union over maps), the same loader the lineups agent and
bias_h3 mirrored. Org sequence sorted (date, match_id), date from the engine
game list; history = strictly earlier DATE (same-day matches are never history
— matches preregister.lineups.md).

- **overlap(A, B)** = |A ∩ B| / max(|A|, |B|, 5) — the engine's own lineup-mode
  formula (engine.py L279). For two clean 5-stacks this is k/5.
- **Change event** at org O's match m_i (i ≥ 1, both lineups defined):
  L_i ≠ L_{i-1} AND L_i ∉ {L_{i-2}, L_{i-3}} (rotation guard: a lineup the org
  fielded within its previous 3 matches re-appearing is rotation, not change).
- **Episode**: each change event opens an episode e with change_date = date(m_i),
  new five N_e = L_i, prev five P_e = L_{i-1}, **ov_e = overlap(N_e, P_e)**.
- **Run length** R_e = number of consecutive matches from m_i (inclusive)
  fielding exactly N_e.
- **Sustained (retrospective, for the atlas/gallery)**: R_e ≥ 3, OR the org's
  recorded sequence ends inside the run (censored=True). Else **transient**
  (stand-in / revert).
- **Sustained-at-day-D (walk-forward, for treatments)**: as of solve day D
  (using only matches dated < D): the run has already reached 3 matches, OR
  N_e is still the org's last-fielded lineup ("alive"). A reverted episode
  stops counting from the day the revert is visible. Fully walk-forward.
- **matches_since_change (msc)**: the lineups-agent definition (walk back
  while L equal), recomputed on the full corpus. MUST reproduce
  `scratch/bias_h3/lineup_topup.csv` msc on every common (org, match) row with
  0 mismatches, else abort (their definitions are the program's).
- **Magnitude classes** on ov_e: keep-4 [0.8, 1.0) · keep-3 [0.6, 0.8) ·
  overhaul (< 0.6). (Clean-set equivalents: 4/5, 3/5, ≤2/5 kept.)

## 2. Case forensics (descriptive; holdout rows labeled descriptive, not scored)

- **ENVY 2026 S1→S2** (operator-requested): exact fives by match through 2026
  with dates/events/opponents; the change event(s) between 2026_stage1 and
  2026_stage2 (who left / who joined, by ProfileURL slug), ov_e; v6 daily
  rating trajectory ±60d around the change (from the verified v6 replay with
  daily_out); per-match predicted p vs outcome before/after; stabilization
  time; carryover cost.
- **Stabilization time** (frozen): with r_pre = team rating on the last solve
  day before change_date and r_post = rating after the 10th post-change match
  (or last available), stabilization = smallest m ≤ 10 such that for every
  post-change match m' ≥ m in the window, |r(m') − r_post| ≤ 0.25·|r_pre −
  r_post|; ">10" if never; "n/a" if |r_pre − r_post| < 0.10 (no jump to adapt to).
- **Carryover cost** (descriptive, probability points): mean(p_v6(team) − won)
  over the first 6 post-change matches (and per-match table). Positive =
  change-blind model overpriced the team.
- **Gallery**: the 4 largest-magnitude sustained episodes in the corpus outside
  ENVY (lowest ov_e, tie → earlier date, org must have ≥5 pre and ≥5 post
  matches), same panels.

## 3. Population atlas (descriptive; no model selection)

Every sustained episode 2023–2026 (full corpus): counts by magnitude class ×
year × region; transient episodes counted separately. Centerpiece: for
team-observations at msc = m (m = 0..9, 10+; msc from the run's start episode,
runs from sustained episodes only), signed calibration bias
mean(won − p_v6(team)) ×100 and mean per-series LL, by magnitude class, CI =
±1.96·SE (normal approx; team-observations = 2 per series). Uses stored v6
p_all descriptively. Power context printed alongside: post-change ≤3 bucket
n=598, MDE80 2.52m within / 7.81m cross (stats/power_mde.json) — and even that
is exploratory-only on this spent frame.

## 4. EXPLORATORY treatments — ≤4 holdout reads, budget frozen

All walk-forward, params train-fit ONLY (train = date ≤ 2024-12-31), β refit
on train per config (engine's beta=None path or identical closed form), CRN
paired bootstrap (iid + block_event) vs the stored v6 baseline, both units
(milli-LL + referee.expected_roi_of_dll), per-bucket panel with post-change ×
magnitude cells. Every read labeled EXPLORATORY on page and in JSON metadata,
appended to stats/roster_looks.json. None of these can be called confirmatory
or promotable — the frame is spent.

**Read 1 — (b) graded change-point continuity.** roster_mode 'change' replaces
v6's year mode inside the otherwise-identical v6 config: weight of game g for
team T at solve day D gains a factor ∏ max(ov_e, 0.2)^γ over T's
sustained-at-D episodes with g.date < change_date ≤ D. γ ∈ {0.5, 1.0, 2.0}
train-fit (select min train LL; each with its own β). One holdout read at the
selected γ.
Mechanism: pre-change history misrepresents a rebuilt roster in proportion to
how much changed; year mode only sees Jan 1. Predicted sign +; predicted size
+0.3..+2.5m overall (likely INSIDE the 1.77m within floor), +1..+4m on the
post-change ≤3 bucket. Falsifier: overall Δ ≤ −1.77m ⇒ dead; train gain of the
selected γ < 0.5m over v6 ⇒ report as null-by-train (read still counted).

**Read 2 — (c) change-triggered partial cold start.** Post-solve, prediction-
time blend on the v6 solve: for team T on day D inside an active
sustained-at-D episode with ov_e ≤ 0.6 and n_since < M (n_since = T's matches
since change_date, dated < D): r' = (1−a)·r + a·region_mean(D),
a = a0·(1−ov_e)·(1 − n_since/M). region_mean = mean of day-D solved ratings
over the team's region (engine team_region_idx universe). a0 ∈ {0.3, 0.6,
1.0} × M ∈ {3, 6} train-fit (min train LL, β refit on adjusted rdiff). One
holdout read at the selected (a0, M).
Mechanism: H3's cold-start/uncertainty-spike result in point-estimate form —
after a big sustained change the old rating is too confident. Predicted sign +
on post-change ≤3 (+1..+5m there); overall +0..+1.5m. Falsifier: post-change
≤3 bucket Δ < 0 with CI excluding 0 ⇒ dead.

**Read 3 — (d) change-scoped 5d gate (cheap, from h3 scratch).** Row-mix:
p = stored p_ss_5d (scratch/bias_h3/model_probs.npz, 2058 rows) on gated rows,
stored v6 p_all elsewhere. Gate (walk-forward): either team is within its
first 3 post-change matches of a sustained-at-day episode with ov_e ≤ 0.6 as
of the series date. Zero new fitting. One holdout read.
Mechanism: compose S1's gain concentrated on thin/change rows (adversary: the
S1 surface was pre-screened; this is the roster-scoped restatement, offered as
exploratory only). Predicted gated-row Δ > 0, overall +0..+2m. Falsifier:
Δ ≤ 0 on the gated rows ⇒ the change-scoped gate does not transfer —
consistent with the adversary's S1 demotion.

**Read 4 — RESERVE.** Used only if Read 1's or Read 2's train selection is a
tie (train-LL gap between top two grid points < 0.1m): the runner-up grid
point of that one treatment may be read once. Otherwise unused and reported
unused.

MDE context quoted with every read: within-family 1.77m / cross-family 5.89m
at n=1217 (stats/power_mde_expanded.json checkpoint_quote); post-change ≤3
bucket 2.52m/7.81m at n=598 (stats/power_mde.json). Reads 1-2 are
within-family (v6 core kept); Read 3 mixes model families → cross floor on
the gated subset.

## 5. Wave-2 results integrated with the adversary's demotions (verbatim law)

- decay 5b-a lineup continuity: +1.68m vs its own no-continuity control,
  −2.82m vs v6, INSIDE NOISE FLOOR — "real but redundant with year mode".
- h3 5d change-adjacent HL 7.4 vs stable 24.8 games: DEMOTED — profile CIs
  overlap ([3.9,18.5] vs [5.3,300.2]), DL τ²=0, supporting WIN fragile;
  "a hypothesis to test on new data, not a finding".
- h3 cold(<10 prior maps) +71.7m n=39: the ONE adversary-robust lead (floor
  32.9m, drop-top-5% +53.1m, jackknife min +66.0m; caveats: 4 events, parent
  model −5.3m overall).
- compose S1 gated shape +1.958m: HOLD — "the evidence's verdict" (surface
  pre-screened, flips under drop-top-5%, block p .888).
- Ledger rows 6-7 (lineup reweighting; roster-instability shrink): REJECTED
  pre-v8 on year-boundary heuristics, reclassified UNRESOLVED by Phase 0
  (stats/ledger_reclass.json) — per-match lineups are a new data source; this
  report is the legitimate re-open.

## 6. Outputs (one writer: agent:roster)

stats/roster_case_envy.json · roster_case_gallery.json · roster_population.json
· roster_treatments.json · roster_integration.json · roster_looks.json;
testing_lab/gen_roster_report.py → testing_lab/out/reports/roster_adaptation.html
(house pattern, EXPLORATORY badges, /testing/v8/stats/ download links, shared
tab strip); testing_lab/v8/roster_adaptation.md mirror; logs/roster.log;
scratch/roster/ only. The prospective-validation plan (§5 of the brief) is
preregistered inside roster_integration.json + the report: frozen specs of
Reads 1-2, test population = series dated > 2026-07-28 as they settle, metric
= mean ΔLL vs v6 (walk-forward, β refit on data ≤ 2026-07-28), decision rule
= Δ ≥ MDE80 at realized bucket n AND p_better ≥ 0.95 in both CRN modes AND no
G3 major bucket regression; MDE80 at projected n from the stored bucket sigmas
(display arithmetic).

## OPERATOR ADDENDUM (appended 2026-07-28 23:47, AFTER the prereg above was
## locked but BEFORE any treatment was computed — ordering disclosed in
## logs/roster.log; no holdout read had been taken)

The operator specified the treatment shape verbatim: "after a roster change,
it operates as a new phase, using previous information as a reference point,
and then overreacting a bit to whether the roster change seems to make the
team now better/worse ... With differences based on the scale of roster
continuity." Formalized as the **PHASE-RESET FILTER**; it REPLACES Read 3
(the change-scoped 5d gate), which is withdrawn UNREAD.

**Read 3' — (d) phase-reset filter (mandatory, operator-specified).** Reuse
the bias_h3 state-space machinery (scratch/bias_h3/lib_h3.py). At every
sustained-at-day change episode of team T with overlap ov_e: the state MEAN is
kept unchanged (previous info = reference point); the state VARIANCE gets a
one-time injection Δq = g·(1 − ov_e) at the change point (walk-forward: on the
first filter update dated ≥ change_date). The elevated Kalman gain then makes
the first post-change results move T's rating unusually fast in WHICHEVER
direction they point (deliberate, direction-agnostic early over-reaction),
decaying back to normal gain as evidence accumulates. g ∈ {0.5, 1.0, 2.0, 4.0}
× base-q from the train-selected h3 core (single filter family) — g train-fit
by min train LL with β/scale handled exactly as bias_h3's judging did; ONE
holdout read at the selected g.
Predictions (operator's + mine, frozen before running): sign + overall
(+0.5..+3m); post-change ≤3 bucket +2..+8m; **contrast vs Read 2 (c)**: the
variance-spike beats the mean-blend on IMPROVEMENT cases — rows in the
post-change ≤3 bucket whose team's realized post-change results beat the
pre-change rating expectation — because the mean-blend pre-judges toward the
prior (an upgraded roster gets marked DOWN), while the variance-spike is
direction-agnostic. Canonical improvement case: LEV gaining Neon (2026);
canonical degradation case: ENVY losing inspire (2026-02-06) — both verified
from lineups data, never memory. Falsifiers: (d) Δ < 0 on the post-change ≤3
bucket with CI excluding 0 ⇒ the over-reaction shape is dead; (d) ≤ (c) on the
improvement-case subset ⇒ the operator's direction-agnostic intuition is not
supported over mean-blend.

Required additions (descriptive, no extra holdout reads):
- **Effective learning rate chart**: filter gain (rating movement per unit
  surprise) vs matches-since-change, by magnitude class — filter telemetry
  across the corpus at the train-selected g (goes in roster_population.json or
  its own key; renders the operator's "overreact a bit, scaled by continuity").
- **Gallery must include** LEV/Neon and ENVY/inspire with per-case overlays of
  v6's rating path vs the phase-reset filter's path through the change window.
- **Player-identity mean-shift** (incoming vs outgoing player quality moving
  the reference point directionally): DESIGN NOTE ONLY in
  roster_integration.json — no holdout read (ledger's crude player-carryover
  rejection is UNRESOLVED; Phase 3's player-rating idea unused).

Read budget unchanged: ≤4 total = Read 1 (b), Read 2 (c), Read 3' (d
phase-reset), Read 4 reserve (unchanged rule, now also usable for a g
tie-break under the same <0.1m train-tie condition).

## OPERATOR ADDENDUM 2 (appended 2026-07-29 00:58, BEFORE running treatment (e);
## ordering in logs/roster.log; uses the reserved 4th read by operator directive)

Operator: "it should be based in v6. You said v6 was the best, so let's start
from there." Design acknowledgment (goes in §5 of the report): read (d)
confounded base-model replacement with the mechanism — the no-injection
state-space base is itself −11.75m vs v6 on stored numbers (1b 0.653913 vs v6
0.64216), and the injection's marginal was −7.52m on top; (d) could NOT
cleanly falsify the mechanism-on-v6. Treatment (e) changes exactly one thing
on v6.

**Read 4 — (e) v6 + temporary post-change overreaction.** Engine config = the
stored v6 baseline EXACTLY (games-decay consist 20/12, rd^0.75×2.5, year
continuity 0.3 kept ON, ridge 0.5, region prior 1.5, PO ×1.6, champ ×2),
plus, in the per-day solve loop: for each team with a sustained-at-D episode
(frozen §1 walk-forward definition; most recent such episode; visible only
when change_date < D), its POST-change games (g_date ≥ change_date) get
weight multiplier
    m(D) = 1 + a·(1−ov_e)·exp(−n_since(D)/τ),
n_since(D) = the team's post-change matches with change_date ≤ date < D
(≥1 whenever visible — the change match itself counts). Pre-change games
UNTOUCHED (the discount side was (b)'s, already adjudicated; kept OFF so (e)
is pure overreaction). Composition: per-side factors folded into the
continuity vector as m² per side, so a game's weight gains ×m per young side
(×m_w·m_l if both sides young — rare, symmetric, preregistered).
**Corpus-wide** (all teams' episodes), the deployable rule — same as (d);
pre-change trajectory coupling through the global Massey solve is therefore
expected and will be reported honestly (base_check-style) rather than zeroed.
Grid (train-only, β refit per config, holdout scrubbed on non-selected):
a ∈ {0.5, 1, 2} × τ ∈ {2, 5}. ONE holdout read at the train argmin.
EXPLORATORY; tallied in roster_looks.json (401→402; this consumes the
reserve, by operator directive rather than the tie rule — disclosed).

Predictions (operator's + mine, frozen): sign POSITIVE — early new-roster
results are informative and the carried rating under-weights them; predicted
size +0..+2m overall, +1..+5m on the post-change ≤3 bucket (the atlas's
+4-6pp outperformance is the signal it should harvest). Falsifier (frozen):
post-change ≤3 bucket CI entirely worse than v6 (ci_hi < 0), OR overall
Δ ≤ −1.77m (within floor, wrong direction) ⇒ the overreaction mechanism is
dead ON V6 — the clean kill (d) could not deliver.

Deliverables: read4_e_v6_overreact appended to stats/roster_treatments.json
(same schema, both units, MDE context); v6_overreact_path [{d,r}] on the v6
base for the ENVY and LEV overlay windows in the two case JSONs + pre-change
coupling magnitude vs v6_path; frozen arm added to the prospective plan in
stats/roster_integration.json; md mirror §3/§5 updated; logs/roster.log.

## ADDENDUM 4 — THE SPEC RUN (agent:roster-g, written 2026-07-29 01:37,
## BEFORE any classifier run, fixture run, solve, or holdout computation.
## Governing document: briefs/roster_spec_operator.md — it wins every conflict.)

### §7 read-back (own words) + ambiguity flags

1. **What the subsystem does.** The model is v6 exactly, plus a local additive
   term: after a team's confirmed roster-change boundary, that team's
   post-boundary games carry extra solve weight ×[1 + a(1−k/5)e^(−n_g/τ)] on
   its side only (n_g = the GAME's own match count since the boundary, 0 for
   the first), so the new five's first results move the rating harder in
   whichever direction they point, scaled by how much changed, decaying as the
   phase matures; independently, any game fielded by a non-modal lineup is
   down-weighted ×[1 − s(1−o_g)]; pre-change history is never discounted — it
   is the reference point.
2. **Sub vs change, causally.** Everything derives from the modal five M_{i,T}
   over the trailing W matches dated < T. A one-off sub deviates (o<1) but
   cannot outvote a trailing window, so M never shifts and no boundary exists —
   the property falls out of the definition. A sustained change flips M only
   after the new five reaches a window majority (P1's confirmation lag); P2
   declares provisionally at the first deviation and retracts on reversion,
   with the harness reproducing the exact knowable state at every T; P3 has no
   boundaries and just weights each game by its contemporaneous o.
3. **Ablation gate.** Every case chart's dashed line is a dedicated run with
   the boost enabled for the featured team only; max|r_v6 − r_ablation| over
   every solve day strictly before that team's first in-window confirmed
   boundary must be EXACTLY 0.0, enforced by a raise-not-render assertion; a
   nonzero value is an implementation bug to fix, never a number to explain.
4. **Activation gate.** θ=(a,τ,s) contains a=s=0 with the model bit-identical
   to v6 (checksum-asserted); a is fit with shrinkage (λa², λ by inner-CV on
   train only) and the subsystem ships enabled only if the inner-CV improvement
   clears its own inner-CV SE; otherwise â=0 and the deployed model IS v6 —
   the deployment cannot behave worse than the champion it extends.

**Ambiguity flags (resolved BEFORE running, disclosed):**
- **A1 (o's reference under P1/P2).** §2.5 defines o against M contemporaneous
  with each game, yet requires that after a sustained change the new lineup's
  games are "full-weight and not penalized" — the first ~W/2 new-phase games
  predate the mode flip. Resolution: under P1/P2 o is measured against the
  modal five OF THE PHASE the game belongs to (phases per boundaries knowable
  at solve day D; phase mode = modal lineup over that phase's matches dated
  < D, ties by recency). Sub games inside a stable phase get o<1; new-phase
  games get o=1; pre-change games are judged against their own phase's mode —
  never the current five. Under P3 (no phases) o is frozen at game date
  against M_{i,date(g)} — exactly the spec's "carries less evidence" behavior.
- **A2 (boundary placement).** The boundary index is the first match at which
  the new modal lineup was fielded within the trailing window at detection
  (min index in window with L == new mode). For a clean change that is the
  change match itself. Detection date = date of the match whose inclusion
  flips the mode; the boost becomes visible to solves only on days D >
  detection date (knowable-state), applying retroactively to games from the
  boundary index with their fixed per-game n_g.
- **A3 (chain anchor).** Two boundaries ≤ c matches apart merge into ONE
  boundary anchored at the FIRST change's index; k re-measured end to end
  (new mode vs the mode before the first change) once the second is detected;
  n keeps counting from the anchor. c=0 ⇒ never merge. Walk-forward: as of D
  only visible detections merge.
- **A4 (s in ablation runs).** §4 defines the ablation as a>0 for the featured
  team, a=0 elsewhere — s is not mentioned, and any s>0 would move pre-boundary
  sub games and make the exact-zero gate unsatisfiable by design. Ablation
  runs therefore set s=0 globally and enable the BOOST only, for the featured
  team's in-window boundaries only; the corpus-wide scoring run (where s
  participates) is a separate run, never mixed (§5's own rule).
- **A5 (window-relative gate).** Every org in a 4-year corpus has pre-2026
  boundaries; "identical until the vertical" is satisfiable only relative to
  the chart window. Case ablations activate the featured team's boundaries
  dated ≥ window start; the zero-gate is computed over ALL solve days in the
  corpus strictly before the first activated boundary's date (stronger than
  in-window-only). SEN's window contains no boundary, so its ablation must be
  bit-identical to v6 on every solve day of the corpus.
- **A6 (a=0 "interior").** a ∈ [0,6] is sign-constrained by design (the
  mechanism is a boost); "interior" is implemented as: 0 is a selectable grid
  point and the shrinkage target, and the model at a=s=0 is v6 bit-identical.
  Negative a (anti-boost) is not part of the spec'd family.
- **A7 (P3's window).** o under P3 needs a window too: W=5 preregistered as
  P3's primary (census reports P3 mode-shift counts for W ∈ {3,5,8}).
- **A8 (unspecified constants).** P2 retraction horizon m=3 matches; per-team
  floor n_min=3 with cap C ∈ {1.5, 2.0, 3.0} train-fit; both preregistered
  here, not tuned post hoc.
- **A9 (SEN's real 2026 change).** Data verification (lineups only, no outcome
  looks): SEN's established five {cortezia, jerrwin, johnqt, jonahp, reduxx}
  since 2026-04-19 (victor→jerrwin, a REAL sustained 4/5 change confirmed by
  any sane classifier); marved deputized for johnqt in exactly ONE match,
  m706350 on 2026-07-16, reverted m706360 on 2026-07-25 — the spec's named
  case verified as stated. SEN case window starts 2026-05-01 so the jerrwin
  boundary (2026-04-19) predates it; in-window boundaries must be [].
- **A10 (missing lineups).** (org, match) slots without lineup rows (coverage
  gaps): excluded from modal windows, o:=1 (never a deviation, never a
  boundary trigger), still counted in n_g match counts. Counted and disclosed
  in the census.

### Policies (all walk-forward; matches ordered (date, match_id); same-day
### matches are never in each other's history; M uses matches dated < T)

- **P1-W (W ∈ {3,5,8}).** M = most frequent exact lineup-set over the trailing
  W lineup-known matches; ties by most recent occurrence. Boundary = M shifts.
  Deviation = o<1 vs the phase mode (A1). Boost + s-down-weight as spec'd.
- **P2 (m=3).** Phase mode = modal lineup over the current phase's matches.
  First deviation ⇒ provisional boundary at that match (boost active
  immediately, per-day knowable state); if any of the next m matches fields
  the phase mode exactly ⇒ retracted (re-solve = automatic in per-day
  recomputation); else confirmed, k = 5·o_dev vs the old phase mode. P2 runs
  ONLY because the harness recomputes every solve day from the dated prefix,
  provisional states included; if any fixture shows otherwise it is NOT RUN.
- **P3 (W=5).** No boundaries, no boost; every game weighted ×[1 − s(1−o_g)],
  o_g frozen at game date. Chart verticals, if P3 ships: M-shift events
  (display only; no model discontinuity).
- k = min(|new mode ∩ old mode|, 5) (P1/P3 display); P2 k = min(|L_dev ∩
  M_old|, 5). o = min(|L ∩ M_phase|, 5)/5.

### Grids and fitting (train = date ≤ 2024-12-31 ONLY; β refit per config;
### holdout numbers scrubbed unseen on every non-read run)

- a ∈ {0, 0.25, 0.5, 1, 1.5, 2, 3, 4.5, 6}; τ ∈ {2,3,5,8,13}; s ∈ {0, 0.2,
  0.4, 0.7, 1.0}; W ∈ {3,5,8}; c ∈ {0,3,5}; m=3; n_min=3; C ∈ {1.5,2,3}.
  If â_raw hits a=6, widen once to {8, 10} and say so.
- Per policy: (a×τ) cross-grid at s=0, c=3 → (â_raw, τ̂); s-profile at
  (â_raw, τ̂) → ŝ; if ŝ>0, a-profile re-run at (τ̂, ŝ); c ∈ {0,5} at the
  optimum (P1/P2); floor cap C fit last at θ̂. P3: s-profile only.
- **Shrinkage/inner-CV:** K=5 contiguous time folds over train rows.
  λ ∈ {0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2} (mean-NLL units per a²).
  Per fold: â_{−f}(λ) = argmin_a [meanNLL_{train∖f}(a; τ̂, ŝ) + λa²];
  CV(λ) = Σ_f n_f·meanNLL_f(â_{−f}(λ)); λ̂ = argmin. â = argmin_a
  [meanNLL_train(a) + λ̂a²]. β per config from the full-train fit (disclosed
  simplification; β is a single monotone link scalar). P3: same machinery on
  s (λs²) — the guarantee applies to its one parameter.
- **Activation gate (pre-committed bar):** Δ_f = meanNLL_f(v6) −
  meanNLL_f(model at â_{−f}(λ̂), τ̂, ŝ); gate fires iff mean_f(Δ_f) >
  SE_f(Δ_f) = sd(Δ_f)/√5 AND â > 0 (P3: ŝ_shrunk > 0). Not fired ⇒ deployed
  model IS v6 (a=s=0), recorded as a first-class outcome.
- **Policy selection (train-side only):** shipped policy = max mean_f(Δ_f)
  among policies whose gate fires; tie (<1e-4 nats) → P3 > P1-W5 > P1-W3 >
  P1-W8 > P2 (spec: P3 cleanest; P2 most complex). **Census disqualifier:**
  a policy whose confirmed boundaries exceed 60% of its deviations in the
  EWC-class tier is misclassifying subs and cannot ship regardless of train
  evidence.
- **Nesting assertion:** subclass run at a=s=0 must equal the plain-Engine v6
  run bit-identically — sha256 checksum on the rdiff array bytes AND on the
  stacked daily-ratings matrix; plus ≤1e-12 vs the stored v6 baseline p_all.

### Fixtures (hard assertions, coded and passing BEFORE any train fitting;
### published in stats/roster_spec_fixtures.json)

- **F1 SEN (real data).** P1 all W: no boundary at m706350; SEN window
  2026-05-01→end boundary list empty. P3-W5: no M shift in window. P2:
  provisional at m706350 (2026-07-16), retracted at m706360 (2026-07-25),
  zero confirmed. Solve-level: two full P2-probe solves (a=2, τ=5, s=0), one
  normal, one with the marved provisional suppressed — SEN daily ratings
  identical (==0.0) on every solve day D ≥ 2026-07-26 (final-timeline
  identity); any difference confined to D ∈ (2026-07-16, 2026-07-26).
- **F2 synthetic revert.** Fabricated 16-team mini-corpus (deterministic
  schedule + results), featured team deviates at match index 5 (2 players
  swapped), reverts at 6. Assert: 0 boundaries (P1 all W), P2 provisional@5
  retracted@6, P3 mode never shifts; mini-solve through the REAL modifier code
  path at (a=2, τ=5, s=0) equals its a=0 twin with max|Δr| == 0.0 on every
  day, before/during/after.
- **F3 synthetic sustained.** Same mini-corpus, change at index 5 (2 swapped,
  k=3) that stays. Assert exactly one boundary at index 5 per policy, detected
  at the policy's predicted lag and NEVER earlier (peeking guard): P1-W3 at
  the 2nd new-five match, P1-W5 at the 3rd, P1-W8 at the 4th (recency
  tiebreak), P2 provisional@5 confirmed after m=3 more, P3-W5 M-shift at the
  3rd. Mini-solve: featured team's ratings equal the a=0 twin (==0.0) through
  the detection date, diverge strictly after; all other teams identical
  pre-detection.
- **F4 corpus census (classifier-only, no solves).** Boundaries per policy ×
  season × tier (vct vs ewc_class per lineups.csv event_class; untagged
  offseason/EWC events ⇒ ewc_class, mapping disclosed), deviations,
  reverted-deviations, chain-merge boundary counts for c ∈ {0,3,5}
  corpus-wide, missing-lineup counts. Sanity assertions: ewc_class deviation
  rate > vct deviation rate (reference: 6.7% vct / 23.3% ewc-class on the 30d
  definition); the census must expose any policy tripping the 60%
  misclassification rule.

### Runs and reads (holdout budget ≤3, tallied in roster_looks.json)

1. **Read A (headline, corpus-wide).** The gate-selected configuration
   (policy + θ̂ + c + floor), all teams' boundaries active — the model that
   would ship. If NO gate fires: deployed model is v6 (Δ≡0 recorded, no read
   spent on it) and Read A is instead taken at the best policy's UNSHRUNK
   train-argmin config, labeled DOCUMENTATION-ONLY (what the gate declined).
   Slices (same vector, no extra reads): change-gated (either side within
   first 3 post-boundary matches), improvement/degradation cases
   (retrospective, prior report's definition), retention bands k=4 / k=3 /
   k≤2, sub-heavy rows (a side's match deviates, no confirmed boundary).
   CRN iid + block_event; both units; MDE 1.773m within quoted alongside.
2. **Reads B/C (conditional policy contrasts).** Only if the top-2 policies'
   CV improvements are within 1e-4 nats AND both gates fire: the runner-up
   corpus-wide, same schema. Otherwise NOT spent.
- **Per-team ablations (ENVY, LEV, SEN)** are chart runs, never quoted as
  performance. Config: shipped θ̂ if a gate fired, else the documentation
  config, disclosed either way. Windows: ENVY 2025-11-01→end, LEV
  2025-08-01→2026-03-01, SEN 2026-05-01→end. prechange_max_abs_diff must be
  EXACTLY 0.0 (A4/A5); the SEN chart must draw no vertical.
- **Coupling subsection:** corpus-wide vs v6 daily ratings; distribution
  (p50/p90/p99/max) of |Δr| over (team, day) for teams with zero corpus
  boundaries under the shipped policy, and separately over changing teams'
  strictly-pre-first-boundary days. Documented property of the joint solve;
  never contaminates the case charts.

### Predictions + falsifiers (frozen)

- Prediction: given (e)'s null (−0.395m) and the stricter causal classifier,
  I predict the activation gate does NOT fire for the boost policies
  (P(fire) ≈ 0.3); if it fires, holdout Δ ∈ [−1, +2]m — inside or near the
  1.773m floor — with the change-gated slice positive. P3's s: weak-negative
  prior (context 3a's integrity down-weight was −0.3..−1m on holdout);
  predict ŝ_shrunk = 0.
- Falsifiers: any fixture failure ⇒ STOP, fix implementation, re-run fixtures
  (numbers produced before the fix are void). Gate fires AND Read A ≤ −1.773m
  ⇒ the causal subsystem is dead on this frame (reported at full resolution).
  Ablation prechange diff ≠ 0.0 ⇒ implementation bug, fix before emitting.
  Census 60% rule ⇒ policy disqualified from shipping.
- Everything here is EXPLORATORY on a spent holdout (402 recorded looks);
  the frozen prospective arm H_specrun (policy + θ̂ + all constants) goes to
  roster_integration.json for adjudication on post-2026-07-28 series.

### Deliverables

stats/roster_spec_fixtures.json · roster_spec_census.json ·
roster_spec_cases.json · roster_spec_read.json · roster_looks.json (updated)
· roster_integration.json (roster_flag extension: modal five, last-match
deviation, matches since confirmed boundary, provisional bit; + H_specrun) ·
this addendum's outcomes at the same resolution · roster_adaptation.md new
top section (answer first) · logs/roster.log · scratch/roster/spec_run/ code.
HTML generator untouched (orchestrator-owned).

## Outcomes (appended AFTER runs — same resolution for failures)

All numbers: stats/roster_treatments.json; reads tallied in
stats/roster_looks.json (3 new EXPLORATORY reads, grand total 398→401; reserve
unused — no train tie). Baseline v6 holdout 0.64216 (n=1217).

- **Read 1 (b) change-point continuity.** Train selected γ=2.0 (train LL
  0.64463 vs v6 0.64823 — a +3.6m train gain). Holdout: **−3.889m** vs v6,
  iid CI [−8.46, +0.35], block CI [−7.77, −0.26] (excludes 0), p_better .037.
  Post-change ≤3: −4.95m [−11.72, +1.55]. Predicted +0.3..+2.5m — SIGN WRONG.
  **Falsifier FIRED** (overall ≤ −1.77m). Replacing year continuity with
  graded change-point continuity is dead on this frame: the train gain did
  not transfer (train-era rosters churn differently than 2025-26).
- **Read 2 (c) partial cold start.** Train selected a0=1.0, M=6 (train
  0.646955 vs 0.64823). Holdout: **−1.423m** [−3.54, +0.75] — INSIDE the
  1.77m within-family floor. Post-change ≤3: −2.83m [−6.36, +0.58] (bucket
  floor 2.52m; CI includes 0) — falsifier NOT fired (required CI excluding
  0), but predicted + sign did not materialize. No support; not definitively
  dead. Keep-4 cells ≈ 0; the damage concentrates in keep3 (−8.93m).
- **Read 3' (d) phase-reset filter (operator addendum).** Train selected
  g=0.5 (train 0.644307, better than base 1b's 0.644845). Holdout:
  **−19.275m** vs v6 [iid −29.33, −9.40], and **−7.524m vs its own
  no-injection 1b base** [−12.08, −3.34] — the injection itself hurts.
  Post-change ≤3: −23.79m [−38.13, −9.09] — **FALSIFIER FIRED** (CI excludes
  0 on the target bucket). Contrast prediction REVERSED: (d) − (c) on
  improvement cases = **−38.32m** [−68.25, −9.17] — the mean-blend BEATS the
  variance-spike on the very cases the over-reaction was designed for. The
  one-shot variance injection makes the filter over-trust noisy first
  results; the atlas's +4-6pp post-change outperformance is real but too
  small relative to single-map noise for a gain spike to capture.
- **Descriptive findings that stand:** post-change teams OUTPERFORM the
  carried v6 rating (+4.4pp keep4 [0.65,8.09] / +5.6pp keep3 / +4.8pp
  overhaul, first 3 matches pooled, vs +0.7pp stable reference) — the
  operator's "new phase" intuition is directionally right in the DATA even
  though all three model treatments failed to monetize it on this frame.
  ENVY S1→S2: 3/5 of the Stage-1 five kept via two chained swaps; v6
  overpriced ENVY by 24-32pp/match through the changes. LEV/Neon (ov 0.4):
  v6 underpriced by 14.7pp/match.
- Look hygiene held: non-selected grid points train-only (holdout scrubbed
  before recording); 1b base holdout was a stored reuse, not a new look.

### ADDENDUM 4 outcomes — THE SPEC RUN (2026-07-29, agent:roster-g; same
### resolution as successes)

- **Fixtures: 54/54 PASSED** (roster_spec_fixtures.json; 53 + the §3.5
  nesting checksum: EngineSpec at a=s=0 is sha256-bit-identical to plain v6
  on rdiff AND stacked daily ratings, and 0.0e0 vs the stored baseline).
  F1 SEN verified from data exactly as the spec asserted (marved one match
  m706350 2026-07-16, reverted m706360 2026-07-25; zero boundaries all
  policies; P2 provisional→retracted). F3 detection lags exact (W3@2nd new
  match, W5@3rd, W8@4th via the 4-4 recency tie, P2@m=3) with pre-detection
  solve identity exactly 0.0.
- **P2: NOT RUN as a scoring policy** (preregistered escape used honestly):
  final-timeline identity after retraction fails by 3.5e-4 rating points —
  root cause PROVEN to be v6's region-prior recursion (classifier state and
  multipliers bit-identical post-retraction; residue exactly 0.0 with the
  recursion off). Exact semantics need an O(days²) full-chain re-solve.
  Classifier fixtures + census for P2 still published.
- **Census** (roster_spec_census.json): the preregistered 60% rule
  disqualified c∈{0,3} for P1 and P2 (at c=0, 69-74% of EWC-class lineup
  events become boundaries — the spec's predicted spike, exposed). c=5
  passes (26-28%) because event-length stand-in excursions MERGE into k=5
  boost-inert boundaries — the intended sub rejection at event scale.
  Deviation rates ewc_class > vct everywhere (P1-W5-c3: 14.1% vs 5.4%,
  same regime as the 23.3%/6.7% reference). DEVIATION FROM PLAN (disclosed):
  the c-scan stage was voided — c fixed at 5 by the census rule.
- **Train stages** (106+8 configs, all train-only, holdout scrubbed):
  τ̂=13 (τ-grid EDGE, disclosed; no widen rule was preregistered for τ).
  a-profiles hit the a=6 edge → widened per spec §3 to {8,10}, then {14,20},
  then (W8) {28,40}: interior argmins a_raw = 8 (W3, ŝ=1.0), 10 (W5, ŝ=0.7),
  28 (W8, ŝ=0.7). P3's s-profile is monotonically worse (ŝ_raw=0) — P3 dead
  on train. W5 s-tie at 5 decimals between 0.7/1.0 → 0.7 by grid order.
- **Shrinkage + activation gate** (inner-CV, K=5 contiguous time folds):
  p1w3c5 λ̂=3e-5 â=4.5 CV+4.97m SE 2.89m FIRED; p1w5c5 λ̂=3e-5 â=4.5
  CV+4.88m SE 2.49m FIRED; p1w8c5 λ̂=0 â=28 CV+6.42m SE 5.34m FIRED
  (barely, ratio 1.20); p3w5 not fired (ŝ_shrunk=0). Preregistered selection
  (max CV improvement) ⇒ **p1w8c5, â=28, τ̂=13, ŝ=0.7**; floor cap Ĉ=1.5
  (train argmin), n_min=3.
- **THE READ (1 look, 402→403): −11.595m** vs v6, iid [−18.99, −4.39],
  block [−18.18, −5.20], p_better .001/.000, n=1217; MDE 1.773m.
  **FALSIFIER FIRED** (gate fired AND read ≤ −1.773m): the causal subsystem
  at the gate-selected config is DEAD on this frame; the deployed model IS
  v6. Slices: change-gated −6.17m [−20.59,+8.40] n=352; k=4 −4.75m; k=3
  +5.75m; k≤2 −18.65m [−49.58,+11.17]; improvement −4.76m; degradation
  −7.32m; sub-heavy +0.56m [−15.62,+15.43] (the s-component alone is not
  the damage). Coupling (own subsection, corpus-wide vs v6): never-changing
  teams (n=21) |Δr| p50 0.00 / p90 1.70 / max 8.03 rating points — the a=28
  reweighting deforms the whole joint solve; that is where the corpus-wide
  damage lives. Contrast reads: condition not met (CV gap 1.45m > 1e-4
  nats) — 0 spent; budget 1 of ≤3 used.
- **Prediction scorecard:** predicted P(gate fires) ≈ 0.3 — WRONG (fired);
  predicted holdout −1..+2m if fired — WRONG (−11.6m, worse); predicted
  ŝ_shrunk=0 — WRONG for P1 (0.7), RIGHT for P3. The spec §3.5 guarantee
  chain's weak link is now measured: mean>SE at K=5 with λ from the same
  inner-CV endorsed a train-overfit family (train-era 2023-24 roster
  dynamics do not transfer to 2025-26 — third instance of this pattern
  after (b) and (e)).
- **Ablations (chart runs only): pre-boundary max|diff| EXACTLY 0.0 ×3**
  (ENVY 514 days, LEV 487, SEN 631 — global over all teams, stronger than
  the spec's own-team requirement). ENVY: 2026-02-06 k=4 + one MERGED
  boundary 2026-05-12 k=3 (May+July chain, end-to-end — the chain rule
  working). LEV: 2025-11-29 k=2. SEN: zero verticals, ablation ≡ v6.
- **Integration:** roster_flag extension emitted for 73 teams (22
  provisional/deviating as of 2026-07-27); H_specrun FROZEN for the
  prospective arm with expectations LOW (falsifier already fired on the
  spent frame).

### ADDENDUM 2 outcome — Read 4 (e) v6 + post-change overreaction (2026-07-29)

Train selected a=2.0, τ=5.0 (grid edge on a; train LL 0.64582 vs v6 0.64823,
+2.4m train gain; no tie). Holdout: **−0.395m** vs v6, iid CI [−3.13, +2.30],
block [−2.32, +1.35], p_better .391/.337 — **INSIDE THE 1.77m NOISE FLOOR**.
Post-change ≤3: −0.19m [−4.38, +3.95]. **Falsifier NOT fired** (bucket ci_hi
+3.95 > 0; overall > −1.77m). Predicted positive sign did not materialize at
the point estimate, but unlike (b)/(d) this is a NULL, not a kill: the clean
mechanism-on-v6 shows no detectable effect at this n. Slice pattern points
weakly the operator's way where the mechanism aims — improvement cases
+4.12m [−3.99, +12.36], gated +2.57m [−6.85, +11.71], keep4 +2.69m
[−1.62, +6.95] — all CIs straddle 0; adjudicable only prospectively (arm
F_v6_overreact frozen). ROI symmetric reading −0.03pp. v6-reproduction guard
passed to 1e-12 before the grid; grid holdouts scrubbed; looks 401→402
(reserve consumed by operator directive, disclosed). Attribution caveat
written into read3's JSON spec: (d) confounded the −11.75m base-model swap
with the mechanism; (e) is the clean attribution. Pre-change solve coupling
of the corpus-wide rule (overlay windows, vs v6_path): ENVY max |Δ| 0.735,
mean −0.030 (n=19); LEV max 0.271, mean −0.042 (n=10).
