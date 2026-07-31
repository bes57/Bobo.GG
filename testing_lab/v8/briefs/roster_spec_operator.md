# BenPom v6 + mid-season roster subsystem: the spec
(OPERATOR-AUTHORED, 2026-07-29 — verbatim. If any experiment or chart
contradicts it, the experiment or chart is wrong, not the spec.)

## 1. What the thing IS

BenPom v6, unchanged, plus one subsystem that activates for a single team at the
moment that team changes its roster mid-season.

- The base model is v6. Not the H3 state-space core. Not a variant. v6, exactly
  as it is in the champion snapshot, with the same solve, the same games-counted
  consistency decay, the same everything.
- The subsystem is additive and local. Before a team's change, that team is
  scored by v6 and by v6+subsystem identically. After the change, the two
  diverge for that team.
- Nothing about this is a replacement model. It is v6 with an extra term that is
  zero almost everywhere.

## 2. What it does, conceptually

A roster change starts a new phase for that team.

- **The old rating is the reference point, not garbage.** Do not discard
  pre-change history. Do not blend the team toward a region prior. Do not
  shrink them toward the league mean. The team's prior level is the best
  starting estimate we have for the new lineup, and it stays.
- **React harder to the first results of the new phase.** The new five's early
  matches should move the rating more than a normal match does, in whichever
  direction those results point. LEV adding Neon should rise faster than v6
  lets it. ENVY losing a core player should fall faster than v6 lets it. Do not
  encode a direction: the mechanism is agnostic, and the early results decide.
- **The reaction decays.** As the new lineup accumulates matches, the extra
  responsiveness fades back to normal v6 behavior. After enough matches the new
  phase is just a phase.
- **Everything scales with how much changed.** A 4/5-retained one-player swap
  barely triggers. A 2/5 rebuild triggers hard. A 0/5 rebuild triggers hardest.

## 2.5 Subs are not changes, and you cannot tell them apart in real time

SEN fielded Marved for one match last week and were back to johnqt immediately
after. That is a substitution, not a roster change. The subsystem must not fire.
The team stays in its existing phase, its pre-existing rating history stays
intact, and nothing about that week is a boundary.

This is the hardest part of the spec, because at the moment the sub game is
played, a sub and a change are the same observation. You only learn which it was
from what happens next. Any classifier that looks at future matches to decide
whether a past match was a sub has leaked, and every number it produces is void.
Solve this causally or do not solve it.

### Definitions, all computed walk-forward

For team i at time T, using only matches dated < T:

- L_{i,g}: the five who actually played in game g.
- M_{i,T}: the team's **modal five**, the most frequent starting five across
  its trailing W matches. Ties broken by recency.
- o_{i,g} = |L_{i,g} ∩ M_{i,T}| / 5: how much of that game was played by
  the team's established lineup.

A **deviation** is any game with o < 1. A **sustained change** is a shift in
M_{i,T} itself. A one-off sub deviates but cannot outvote a trailing window,
so M never moves and no phase boundary is created. That property falls out of
the definition rather than needing a special case.

### The tension you must confront, not paper over

Modal-window classification is causal by construction, but it activates late: a
genuine change takes roughly W/2 matches to flip the mode, and the subsystem
exists precisely to react to the *first* matches of the new phase. A large W
buys correct sub rejection and forfeits the mechanism's purpose. A small W
activates fast and misclassifies every sub as a change.

Pre-register all three policies and report all three. Do not silently pick one.

- **P1, confirmation lag.** M over a trailing window, W ∈ {3,5,8} matches.
  Simple, causal, late. This is the baseline.
- **P2, provisional activation with retraction.** Declare a phase boundary on
  the first deviation, then retract it if the lineup reverts within m matches.
  Retraction re-solves history, which is legitimate for the rating timeline, but
  the walk-forward harness must reproduce exactly the state that was knowable at
  each T, including the provisional boundary that was later retracted. If your
  harness cannot do that, do not run P2 rather than running it wrong.
- **P3, no classification at all.** Skip the discrete boundary. Weight each
  game's contribution by o_{i,g} continuously, so a sub game simply carries
  less evidence about the team, and a sustained change re-bases M over time on
  its own. This is the cleanest formulation and may make the boundary concept
  unnecessary. Test it as a genuine competitor, not an afterthought.

### Sub games themselves, independent of which policy wins

Regardless of phase logic, a game played by a non-modal lineup is weaker
evidence about that team. SEN with Marved tells you less about SEN than SEN with
johnqt does. Down-weight it:

    w_{i,g} <- w_{i,g} · [1 - s(1 - o_{i,g})]

with s fit on train, s ∈ [0, 1]. This is orthogonal to the overreaction
boost in section 3: one says "trust this game less because the wrong players
played it," the other says "trust these games more because they are the new
lineup's first evidence." A game can trigger both, and after a sustained change
o is computed against the *new* modal five, so the new lineup's games are
full-weight and not penalized.

Do not use this to discount pre-change history. Section 2's reference-point rule
still holds: o is measured against the modal five contemporaneous with each
game, not against the team's current five.

### Chains

ENVY's 2026 sequence was two swaps months apart, not one event. Pre-register the
merge rule: two changes within c matches of each other are one phase boundary
with k measured end to end; beyond c they are separate boundaries and the
second re-triggers on a lineup that is already mid-adaptation. Test
c ∈ {0, 3, 5}. Report how many boundaries each setting produces corpus-wide
so the choice is visible rather than buried.

### Required test fixtures

Write these before running anything. Each is a hard assertion, not a chart.

1. **SEN, the named case.** One deviation, immediate reversion. Assert: zero
   phase boundaries under P1 and P3; under P2, a boundary that is retracted, and
   a final rating timeline identical to the no-boundary run. If SEN shows a
   surviving boundary, the classifier is broken.
2. **Synthetic revert.** A fabricated team with a single-match deviation at a
   known index. Assert zero boundaries and no rating deviation before, during,
   or after.
3. **Synthetic sustained change.** Same fabricated team, lineup changes and
   stays. Assert exactly one boundary, detected at the match index the policy's
   lag predicts. If P1 with W=5 detects it at match 1, the implementation is
   peeking.
4. **Corpus census.** Boundaries per policy per season, with the
   deviation-that-reverted count alongside. Sanity: the earlier lineup work found
   stand-in rates of 6.7% in VCT events and 23.3% in EWC-class. If a policy turns
   most of that 23.3% into phase boundaries, it is misclassifying subs, and the
   census will show it as an implausible spike in EWC-class boundaries.

### What the bot gets

At quote time the ambiguity is live and unresolved, which is exactly what the
bot needs told. Extend roster_flag per team with: current modal five, whether
the last match deviated from it, matches since the last confirmed boundary, and
a provisional bit when a deviation has occurred but not yet been confirmed or
retracted. A team in the provisional state is a sizing signal, not a fair-value
change. Quote it smaller until the classification settles.

## 3. The implementation

v6 already carries per-side game weights (the decay is per-side). Use that hook.

For team i and game g played after i's most recent confirmed boundary:

    w_{i,g} = w^v6_{i,g} · [1 + a(1 - k_i/5) e^{-n_{i,g}/τ}]

where k_i is the number of players retained at that boundary, measured against
the pre-boundary modal five, and n_{i,g} counts team i's matches since the
boundary date (0 for the first one).

Properties this must have, each asserted in code:

- a = 0 reduces exactly to v6. Bit-identical. Test it.
- Pre-change games are untouched. The reference point is preserved. This is
  deliberately NOT treatment (b), the pre-change discount, which already failed
  at -3.9 milli. Do not re-run (b) and call it this.
- k_i = 5 (no change) gives a boost factor of exactly 1. No trigger.
- The boost applies to the changed team's side of the game only, not the
  opponent's side.
- Fit a, τ, and s on train only. The previous grid selected a at its upper
  edge, so widen it: a ∈ [0, 6], τ ∈ {2,3,5,8,13} matches. If it picks the
  edge again, say so and widen again rather than reporting an edge fit.

## 3.5 The design guarantee: nested, shrunk, gated

Build this so that the shipped model contains v6 as a special case and defaults
to it absent evidence. Three requirements, all enforced in code:

**Nesting.** The parameter vector θ = (a, τ, s) has a = s = 0 as an interior
point of the search space, and at that point the model is v6 bit-identical.
Assert it with a checksum on the solved ratings, not by eyeball. The family
therefore cannot fit worse than v6 on the objective it is fit to; the worst
case is that it selects v6.

**Shrinkage toward a = 0.** Do not fit a by raw train argmin. Penalize it:

    â = argmin_a [ NLL_train(a) + λ a² ]

with λ set by inner-fold cross-validation on train only. This is what makes the
guarantee hold out of sample rather than only in sample: a weak or noisy signal
gets pulled to zero and the model degenerates to v6 on its own.

**Activation gate.** The subsystem ships enabled only if the train-side
evidence clears a pre-committed bar (inner-CV improvement exceeding its own
inner-CV standard error). If the bar is not cleared, a is set to 0 and the
deployed model IS v6. Record the gate decision as a first-class output, not a
footnote.

**Per-team floor.** For any team with fewer than n_min post-boundary matches,
the boost is capped so the effective weight cannot exceed the cap set on train.
A rebuild with two games played should not be able to swing the solve further
than the evidence supports.

Together these mean: the mechanism either adds something or it does nothing.
There is no configuration in which the deployed model behaves worse than the
champion it extends, because when the evidence is thin the deployed model is
the champion.

## 4. The chart requirement, which is where this keeps going wrong

Every case chart (ENVY, LEV, SEN, and any other named team) shows two lines and
only two lines:

1. **v6**, solid.
2. **v6 + subsystem, with the subsystem enabled for THIS TEAM ONLY**, dashed.

That second line is a per-team ablation run: a > 0 for the featured team,
a = 0 for all other teams. This is not optional and it is not a display trick.
It is the only run that answers the question the chart is captioned with.

**Hard gate, enforced in the generator, not in prose:** compute
max|r^v6_t - r^ablation_t| over every solve day strictly before the team's
first confirmed boundary. It must be exactly 0. Not "small," not "0.03 on
average," not "explained by network coupling." Zero. If it is not zero, raise
an exception and do not write the HTML. A nonzero value means the subsystem
leaked into a team it should not have touched, and that is a bug in the
implementation, not a footnote for the caption.

Also on the chart:

- Stepped lines. Horizontal between that team's matches, vertical jumps on
  their match dates. Already done, keep it.
- One red vertical per that team's own confirmed boundaries, labelled with k/5.
  A reverted deviation gets no vertical. If the SEN chart draws one, the
  section 2.5 gate has failed and the generator must raise rather than write.
- No state-space lines. No no-injection base lines. No phase-reset lines. That
  implementation is dead; it keeps its row in the results table as a record and
  appears nowhere else.
- Caption states in one sentence: same model, identical until the vertical,
  every point of separation after it is this team's own adaptation.

## 5. Scoring, which is a separate question from the chart

The chart uses per-team ablations. The accuracy numbers use the corpus-wide
run, where every team's boundaries are active simultaneously, because that is
the model that would actually ship. Report both and never mix them.

- **Corpus-wide holdout read**: the headline. Overall ΔLL vs v6 with CI, plus
  the pre-registered slices: change-gated rows, improvement cases, degradation
  cases, by retention band (k=4, k=3, k<=2), and sub-heavy rows (deviation
  without boundary).
- **Ablation runs**: charts and per-case forensics only. Never quote an
  ablation ΔLL as the model's performance.
- **Coupling, in its own subsection**: under the corpus-wide run, how much does
  a non-changing team's rating move because its opponents changed? Report the
  distribution. It is a real property of a joint solve and worth documenting.
  It is not allowed to contaminate the case charts.
- **Gate outcome**: state whether the activation gate in 3.5 fired, the fitted
  â after shrinkage, and the inner-CV margin that drove the decision.

## 6. Reporting

Open with the answer, not the journey. First paragraph: does v6 gain a roster
subsystem, what did the gate decide, what is the fitted â, and what is the
effect size with its CI next to the MDE so the number is read in context.

The holdout is spent. Label the reads exploratory and keep the looks tally
current. Keep the frozen prospective arm: when post-2026-07-28 series settle,
this gets re-scored on data nobody has touched, with the decision rule fixed in
advance. That is where a lean becomes a result.

Deliverables: the spec-conformance assertions from 2.5 and 3.5 as a passing
test report, the corpus census by policy, the case charts with their zero-gate
verified, the corpus-wide slice table, and the ledger entry.

## 7. Before you write code

State back, in four sentences: what the subsystem does, how a sub is
distinguished from a change without looking forward, what the ablation gate
asserts, and what the activation gate guarantees. If those four sentences do
not match sections 2, 2.5, 3.5, and 4, stop and re-read rather than starting.
