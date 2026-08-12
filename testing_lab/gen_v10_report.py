"""Build /testing/report/v10_lab — the year-isolation lab.

Reads testing_lab/v10/stats/*.json (written by run_yeariso.py,
run_concentration.py and the premise script) and writes one standalone
HTML document to testing_lab/out/reports/v10_lab.html.

Same house conventions as gen_v9_report.py: shared CSS, Chart.js 4.4.3 via
CDN, data inlined as __TOKEN__ -> json.dumps, download links back to
/testing/v10/stats/.

Run: python3 testing_lab/gen_v10_report.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
V10S = os.path.join(HERE, "v10", "stats")
RD = os.path.join(HERE, "out", "reports")
os.makedirs(RD, exist_ok=True)


def rj(name, d=V10S):
    with open(os.path.join(d, name)) as f:
        return json.load(f)


TR = rj("v10_transfer.json")
RO = rj("v10_roi.json")
CO = rj("v10_concentration.json")
PR = rj("v10_premise.json")

with open(os.path.join(HERE, "v10", "preregister.v10.md")) as f:
    PREREG = f.read()

C = {"s1": "#7c4dd6", "d5": "#c96a2a", "a1": "#3a90cc", "v6": "#6f6a7c",
     "gray": "#9a93a6", "good": "#1e7a4f", "bad": "#c0392b", "ink": "#16121d"}

# reuse the house stylesheet verbatim
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "genv9", os.path.join(HERE, "gen_v9_report.py"))
# gen_v9_report executes work at import time, so read its CSS as text instead
with open(os.path.join(HERE, "gen_v9_report.py")) as f:
    _src = f.read()
CSS = _src.split('CSS = """', 1)[1].split('"""', 1)[0]

NAV = """<div class="labtabs">
<a href="/testing/">Testing Lab</a>
<a href="/testing/report/state_of_benpom">State of BenPom</a>
<a href="/testing/playbook">Deployment Playbook</a>
<a href="/testing/report/favorites_lab">Favorites Lab</a>
<a href="/testing/report/final_model">Final Model</a>
<a href="/testing/report/v7_lab">v7 Lab</a>
<a href="/testing/report/v8_lab">v8 Lab</a>
<a href="/testing/report/roster_adaptation">Roster</a>
<a href="/testing/report/v9_lab">v9 Lab</a>
<a href="/testing/report/v10_lab" class="on">v10 Lab</a>
</div>"""


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def dl(name):
    return (f'<div class="dl"><a href="/testing/v10/stats/{name}" download>'
            f'&#8681; {name}</a></div>')


def sgn(x, nd=2):
    return f"{x:+.{nd}f}"


hard = TR["arms"]["A_iso_hard"]
none = TR["arms"]["A_iso_none"]
rb = PR["roster_boundaries"]
pw = PR["prior_year_weight"]

# ── chart payloads ───────────────────────────────────────────────────────────
page_transfer = {
    "arms": ["A_iso_hard", "A_iso_none"],
    "labels": ["Year isolation (the proposal)", "Full carryover (other endpoint)"],
    "t1": [hard["T1"]["delta_milli_ll"], none["T1"]["delta_milli_ll"]],
    "t2": [hard["T2"]["delta_milli_ll"], none["T2"]["delta_milli_ll"]],
    "floor": 1.773,
}
page_phase = {"labels": [r["phase"] for r in CO["by_season_phase"]],
              "delta": [r["delta_milli"] for r in CO["by_season_phase"]],
              "n": [r["n"] for r in CO["by_season_phase"]]}
page_month = {"labels": [r["month"] for r in CO["by_month"]],
              "delta": [r["delta_milli"] for r in CO["by_month"]],
              "n": [r["n"] for r in CO["by_month"]]}
page_roster = {"dist": rb["dist_kept_0_to_5"],
               "labels": ["0/5", "1/5", "2/5", "3/5", "4/5", "5/5"]}
page_weight = {"orgs": [t["org"] for t in pw["teams"]],
               "share": [round(100 * t["share"], 1) for t in pw["teams"]]}

page_roi = {"thresh": [r["threshold_cents"] for r in RO["by_threshold"]["t2h"]],
            "roi": [r["roi_pct"] for r in RO["by_threshold"]["t2h"]],
            "n": [r["n_bets"] for r in RO["by_threshold"]["t2h"]],
            "lo": [r["roi_ci95"][0] for r in RO["by_threshold"]["t2h"]],
            "hi": [r["roi_ci95"][1] for r in RO["by_threshold"]["t2h"]]}
json.dump(page_roi, open(os.path.join(V10S, "v10_page_roi.json"), "w"), indent=1)
json.dump(page_transfer, open(os.path.join(V10S, "v10_page_transfer.json"), "w"), indent=1)
json.dump(page_phase, open(os.path.join(V10S, "v10_page_phase.json"), "w"), indent=1)

worst = min(CO["by_season_phase"], key=lambda r: r["delta_milli"])
best = max(CO["by_season_phase"], key=lambda r: r["delta_milli"])

HTML = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>v10 Lab — Year Isolation</title>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@800&family=DM+Sans:wght@400;700&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
<style>{CSS}</style></head><body><div class="wrap">

<h1>v10 Lab — Year Isolation</h1>
<div class="tagline">Should each season be solved from scratch, ignoring previous years?</div>
{NAV}

<div class="banner" style="background:var(--badbg);border-color:#e8cfcf;border-left-color:var(--bad)">
<style>.banner b {{ color:var(--bad); }}</style>
<b>THE ANSWER — NO, AND BY A LOT.</b> Solving each year in isolation loses
<b>{sgn(hard['T1']['delta_milli_ll'])} milli-LL</b> on 2025 and
<b>{sgn(hard['T2']['delta_milli_ll'])}</b> on 2026H1 — both eras lost, both
several times the {page_transfer['floor']}m noise floor, in the wrong direction.
The damage is a cold-start problem: <b>{sgn(worst['delta_milli'])}m in
{worst['phase']}</b>, fading to roughly nothing by mid-season. But the
hypothesis is not silly — see §5, where isolation is actually
<b>{sgn(best['delta_milli'])}m better</b> late in the season, and §2, where the
premise turns out to be half right.
</div>

<section>
<h2><span class="n">1</span>What was asked, and what was tested</h2>
<p>The operator's hypothesis, verbatim: <em>"treat each year as a new year where
you don't look back at match data from previous years &mdash; they're often with
old rosters that shouldn't affect current ratings."</em></p>
<p>That is a precise, falsifiable claim about the solver, so it was preregistered
and run. The arm <code>A_iso_hard</code> sets the year-continuity factor to zero
for <b>every</b> team at <b>every</b> January boundary, and &mdash; critically
&mdash; also resets the region-prior chain, without which "isolation" leaks the
previous season straight back in (§3).</p>
<div class="cards">
  <div class="card"><div class="lbl">2025 (VAL1)</div>
    <div class="big bad">{sgn(hard['T1']['delta_milli_ll'])}m</div>
    <div class="sub">vs v6, n={hard['T1']['n_scored']}</div></div>
  <div class="card"><div class="lbl">2026H1 (VAL2)</div>
    <div class="big bad">{sgn(hard['T2']['delta_milli_ll'])}m</div>
    <div class="sub">vs v6, n={hard['T2']['n_scored']}</div></div>
  <div class="card"><div class="lbl">Advance rule</div>
    <div class="big">FAILED</div>
    <div class="sub">needs &ge;+1.773m on BOTH</div></div>
  <div class="card"><div class="lbl">p(better than v6)</div>
    <div class="big">{hard['T1']['bootstrap']['p_better']:.3f}</div>
    <div class="sub">paired bootstrap, VAL1</div></div>
</div>
</section>

<section>
<h2><span class="n">2</span>The premise, measured</h2>
<p>Before testing the cure, measure the disease. Two things had to be true for
the hypothesis to have force: rosters must actually turn over at year
boundaries, and old games must actually still matter.</p>
<p><b>Both are true.</b> Teams keep {rb['mean_kept']:.2f}/5 players across a January
boundary on average, and <b>{pw['mean_share']*100:.1f}%</b> of the average team's
current rating weight still comes from pre-2026 games.</p>
<div class="chartbox"><canvas id="c_roster"></canvas></div>
<p class="cap">How many of its five players a team keeps across a year boundary,
{rb['n']} team-years. <b>{rb['pct_half_or_less']}%</b> keep two or fewer &mdash;
those are the real rebuilds the hypothesis is aimed at. But
<b>{rb['pct_full']}%</b> keep all five, and for them last season is the same
team.</p>
{dl('v10_premise.json')}
<div class="callout">The premise is <b>half right</b>, and that half is exactly
why the blanket cure fails. A third of teams genuinely should forget last year.
A quarter genuinely should not. v6 already tells them apart &mdash; see §3.</div>
</section>

<section>
<h2><span class="n">3</span>What the champion already does (and a dead constant)</h2>
<p>v6 is <em>already</em> a year-isolation model &mdash; a conditional one. At each
boundary it computes</p>
<p class="mono">year_cont = min(|roster(last event Y-1) &cap; roster(first event Y)| / 5, 1)</p>
<p>so a team that kept 0/5 players <b>already has its prior-year games zeroed</b>,
and one that kept 5/5 keeps them in full. The proposal is the
<code>year_cont &equiv; 0</code> endpoint of a dial the champion already fits per
team from data.</p>
<div class="callout warn"><b>Correction to the record.</b>
<code>ROSTER_CONT = 0.3</code> in <span class="mono">BuildRatingTimeline.py:132</span>
is <b>dead code</b> &mdash; referenced only at its own definition, and the lab
engine's <code>"year"</code> mode ignores the persistence argument entirely. The
factor actually applied is the measured overlap, mean
<b>{rb['mean_continuity_factor']:.2f}</b>, not 0.3. Lab documents describing v6 as
"0.3 continuity" are wrong.</div>
<h3>Isolation has to cut three channels, not one</h3>
<table><thead><tr><th>#</th><th>Channel</th><th>Strength</th></tr></thead><tbody>
<tr><td>A</td><td>prior-year games still in the fit</td><td>~11% weight at 1 year, ~1% at 2</td></tr>
<tr><td>B</td><td><b>region-prior chain</b> (<code>_chain</code>, never reset)</td><td><b>dominant</b> &mdash; a team with no in-year data solves to 0.75 &times; last year's regional mean</td></tr>
<tr><td>C</td><td>consistency classifier carries last year's winrate</td><td>weak (picks HL 20 vs 12)</td></tr>
</tbody></table>
<p class="cap">Cutting only the games (A) and calling it isolation would have
been a null experiment: channel B would have carried the previous season back in
regardless. The tested arm cuts A and B.</p>
</section>

<section>
<h2><span class="n">4</span>The result &mdash; both eras, both lost</h2>
<div class="chartbox"><canvas id="c_transfer"></canvas></div>
<p class="cap">Era-transfer scores in milli log-loss versus v6; positive is
better than v6. Grey band is the &plusmn;{page_transfer['floor']}m within-family
noise floor. Year isolation is not close, and it loses on both eras &mdash; the
protocol requires winning both.</p>
{dl('v10_page_transfer.json')}
<h3>The other endpoint, for scale</h3>
<p>Running the dial the other way &mdash; <code>A_iso_none</code>, full carryover
with no boundary discount at all &mdash; costs only
{sgn(none['T1']['delta_milli_ll'])}m / {sgn(none['T2']['delta_milli_ll'])}m.
So v6's fitted middle beats both ends, and the <b>isolation end is roughly seven
times worse than the carryover end</b>. If the model is mis-set, it is not
mis-set in the direction of remembering too much.</p>
<div class="scroll"><table>
<thead><tr><th>Arm</th><th>2025 &Delta;</th><th>95% CI</th><th>2026H1 &Delta;</th><th>Verdict</th></tr></thead>
<tbody>
<tr><td class="mono">A_iso_hard</td><td class="bad">{sgn(hard['T1']['delta_milli_ll'])}m</td>
    <td class="mono">[{hard['T1']['bootstrap']['ci_lo']*1000:.1f}, {hard['T1']['bootstrap']['ci_hi']*1000:.1f}]</td>
    <td class="bad">{sgn(hard['T2']['delta_milli_ll'])}m</td>
    <td><span class="verd dead">DEAD</span></td></tr>
<tr><td class="mono">A_iso_none</td><td>{sgn(none['T1']['delta_milli_ll'])}m</td>
    <td class="mono">[{none['T1']['bootstrap']['ci_lo']*1000:.1f}, {none['T1']['bootstrap']['ci_hi']*1000:.1f}]</td>
    <td>{sgn(none['T2']['delta_milli_ll'])}m</td>
    <td><span class="verd hold">no advance</span></td></tr>
</tbody></table></div>
</section>

<section>
<h2><span class="n">5</span>Where it loses &mdash; and the one place it wins</h2>
<p>The preregistered prediction was that the damage would concentrate early in
each season. It does, overwhelmingly.</p>
<div class="chartbox"><canvas id="c_phase"></canvas></div>
<p class="cap">Log-loss difference by season phase. Negative = isolation worse.
The entire verdict is decided in {worst['phase']}
({sgn(worst['delta_milli'])}m, n={worst['n']}), when an isolated solve has almost
no games of its own yet.</p>
{dl('v10_concentration.json')}
<div class="callout good"><b>The interesting part.</b> By
{best['phase']} the sign flips: isolation is <b>{sgn(best['delta_milli'])}m
better</b> than v6. Once a season has enough of its own data, forgetting last
year genuinely helps &mdash; the operator's intuition is right, it is just
swamped by the cold-start cost of getting there. That is a real finding and it
points somewhere better than either endpoint (§6).</div>
<h3>Why the cold start is so expensive</h3>
<p>With no history, every team solves to a rating near zero and every match is a
coin flip. In January&ndash;February the isolated model has to price
<b>{CO['coinflip']['A_iso_hard']['janfeb_pct_within_0.5']}%</b> of series inside a
half-point rating gap, against <b>{CO['coinflip']['v6']['janfeb_pct_within_0.5']}%</b>
for v6 &mdash; it is guessing far more often, precisely when the schedule is
busiest.</p>
<div class="chartbox"><canvas id="c_month"></canvas></div>
<p class="cap">Same measurement by calendar month.</p>
</section>

<section>
<h2><span class="n">6</span>What this suggests instead</h2>
<p>The result is not "the idea was worthless" &mdash; it is "the idea is right
about staleness and wrong about the remedy." Three follow-ups are better
motivated by this data than another blanket setting:</p>
<ol>
<li><b>Asymmetric decay by roster state, already fitted.</b> v6's conditional
factor is the right shape; the late-season <b>{sgn(best['delta_milli'])}m</b> gain
suggests it may be too generous to old data <em>once the current season is
established</em>. A factor that decays with in-year games played, rather than a
constant per boundary, is untested.</li>
<li><b>A cold-start prior instead of a cold start.</b> Most of the loss is
January having nothing to say. Seeding a new season from the previous year's
<em>regional</em> level while still zeroing team-specific history would keep the
staleness fix and drop most of the cost.</li>
<li><b>Leave it alone.</b> The honest default. Both endpoints lose, v6's fitted
middle wins, and the corpus of prior labs already says memory was too short
rather than too long.</li>
</ol>
</section>

<section>
<h2><span class="n">7</span>Standing, caveats, and what this does NOT establish</h2>
<div class="callout bad"><b>This is selection-grade evidence only. It cannot
promote or demote anything.</b> The virgin confirmatory window holds
<b>53 settled series</b> against a first checkpoint of n=100, so no confirmatory
read was available and none was taken.</div>
<h3>Frame change &mdash; disclosed</h3>
<p>The corpus moved under this program: 26 off-circuit events were synced in, Red
Bull Home Ground was removed, and the EWC qualifiers were split into four
regional events. The frame is therefore <b>n={TR['frame_n']}</b>, not the v9
frame's 2058, and the protocol's &beta; fixture
(<span class="mono">&beta;(FIT1, v6) = 0.1152 &plusmn; 1e-3</span>) reads
<b>{TR['fixture_beta_fit1_v6']}</b> here. The v9 A1&ndash;A5 bar heights were
calibrated on the old frame and do not transfer exactly.</p>
<p>This matters for a marginal result. It does not rescue this one: the effect is
{abs(hard['T1']['delta_milli_ll'])/page_transfer['floor']:.1f}&times; the noise
floor in the losing direction on one era and
{abs(hard['T2']['delta_milli_ll'])/page_transfer['floor']:.1f}&times; on the other,
and the mechanism (§5) is legible and consistent across both.</p>
<h3>Other limits</h3>
<ul>
<li>Three arms, no hyperparameter search &mdash; nothing here was tuned, so
nothing here is overfit, but nor has the <em>space around</em> isolation been
explored.</li>
<li>The late-season sign flip (§5) is a slice of a losing arm, not a candidate.
It is a hypothesis for a future preregistration, not a result.</li>
<li>Promotion means promotion within the lab only. Nothing on the public site
regardless of outcome, and VCTMM remains hands-off.</li>
</ul>
</section>

<section>
<h2><span class="n">8</span>The model we are sticking with</h2>
<p>v10 is rejected, v9 rejected five candidates, v8 rejected the roster family.
The champion is unchanged, and this is it in full &mdash; every number below is
frozen and fitted only on 2023&ndash;2024 data.</p>
<div class="scroll"><table>
<thead><tr><th>Component</th><th>Setting</th><th>What it does</th></tr></thead><tbody>
<tr><td>Solver</td><td class="mono">walk-forward Massey</td><td>ratings for day D solved only from games before D</td></tr>
<tr><td>Decay</td><td class="mono">games, HL 20 / 12</td><td>by games played since, not calendar time; anomalous results age out faster</td></tr>
<tr><td>Consistency</td><td class="mono">WR_HALF_LIFE 16</td><td>classifies a result against the team's decayed winrate at the time</td></tr>
<tr><td>Margin</td><td class="mono">RD_POWER 0.75, RD_SCALE 2.5</td><td>round differential, compressed</td></tr>
<tr><td>Year boundary</td><td class="mono">min(overlap/5, 1)</td><td>measured roster carryover &mdash; mean {rb['mean_continuity_factor']:.2f}, NOT the dead 0.3 constant</td></tr>
<tr><td>Stakes</td><td class="mono">Champions &times;2.0, playoffs &times;1.6</td><td>bigger games count more</td></tr>
<tr><td>Ridge</td><td class="mono">0.5 + region prior 1.5</td><td>borrowed strength for thin schedules</td></tr>
<tr><td>&beta;</td><td class="mono">{RO['beta']:.4f}</td><td>rating gap &rarr; map probability, fit on FIT1 only</td></tr>
</tbody></table></div>
<div class="callout">The deployed site and trading model refit &beta; on the live
corpus each build; the {RO['beta']:.4f} above is the lab's FIT1-only refit used
for every score on this page, so nothing here is fitted on data it is scored on.</div>
</section>

<section>
<h2><span class="n">9</span>What it would have returned against Kalshi</h2>
<p>Reporting only. Market data is never a fitting target and never a selection
signal &mdash; this asks what the frozen model would have done against the book,
it does not tune anything toward it.</p>
<p><b>Window: {RO['window'][0]} to {RO['window'][1]}</b> &mdash;
{RO['n_matches']} settled tier-1 matches, median volume
{RO['median_volume']:,} contracts. That is roughly three months, <b>not a full
year</b>; the Kalshi VALORANT market does not go back further than this.</p>
<div class="cards">
  <div class="card"><div class="lbl">&ge;5c edge, T-2h</div>
    <div class="big">{RO['headline']['n_bets']} bets</div>
    <div class="sub">of {RO['n_matches']} matches</div></div>
  <div class="card"><div class="lbl">Hit rate</div>
    <div class="big">{RO['headline']['hit_rate']}%</div>
    <div class="sub">mostly longshots &mdash; see below</div></div>
  <div class="card"><div class="lbl">ROI</div>
    <div class="big good">+{RO['headline']['roi_pct']}%</div>
    <div class="sub">block CI [{RO['headline']['roi_ci95_block_by_day'][0]}, {RO['headline']['roi_ci95_block_by_day'][1]}]</div></div>
  <div class="card"><div class="lbl">Read this as</div>
    <div class="big warn">ONE BET</div>
    <div class="sub">not {RO['headline']['n_bets']} independent edges</div></div>
</div>
<div class="chartbox"><canvas id="c_roi"></canvas></div>
<p class="cap">ROI by minimum edge, buying at the T-2h price. It rises with the
threshold, which looks like a dose-response curve and is the single most
seductive thing on this page. &sect;9.1 is why you should not believe it.</p>
{dl('v10_page_roi.json')}

<h3>9.1 &mdash; Why I do not believe this number</h3>
<p>Three checks, all of which point the same way.</p>
<p><b>It is 91% one bet.</b> Of the {RO['headline']['n_bets']} qualifying wagers,
<b>{RO['by_side']['underdog']['n']} are on the underdog</b>
(ROI +{RO['by_side']['underdog']['roi_pct']}%) and only
{RO['by_side']['favourite']['n']} on the favourite
(ROI {RO['by_side']['favourite']['roi_pct']}%). This is not a model finding
edges in both directions; it is a single directional wager &mdash; "underdogs are
underpriced" &mdash; repeated {RO['by_side']['underdog']['n']} times in one
three-month window. A bootstrap over bets, or even over days, cannot tell you
whether that direction survives; it only tells you the window was consistent.</p>
<p><b>The model is worse at picking winners than the book it is beating.</b>
BenPom's favourite won {RO['discrimination']['benpom_fav_hit_pct']}% of the time
against Kalshi's {RO['discrimination']['kalshi_fav_hit_pct']}%, and the mean
probability each assigned to the eventual winner was
<b>{RO['discrimination']['mean_p_on_winner_benpom']}</b> for BenPom versus
<b>{RO['discrimination']['mean_p_on_winner_kalshi']}</b> for Kalshi. BenPom scores
the better log-loss here ({RO['logloss']['delta_milli']:+.1f}m) only because it is
<em>less confident</em>, and log-loss rewards hedging. Being less confident than a
market is exactly what makes a model disagree with it toward the underdog on
almost every match.</p>
<p><b>It contradicts the lab's own benchmark.</b> The earlier head-to-head
(<span class="mono">out/kalshi_compare.json</span>, n={RO['prior_lab_benchmark']['n']})
found the opposite: <b>{RO['prior_lab_benchmark']['finding']}</b>. Different
window and a smaller sample, but a straight sign flip on the same question is a
reason to hold the new number loosely, not to overwrite the old one with it.</p>
<div class="callout bad"><b>Do not size anything off this.</b> The honest reading
is: over one three-month window, taking the underdog wherever a
deliberately-hedged model disagreed with the book by 5c or more happened to pay.
That is a hypothesis about underdog pricing in this market, not a demonstrated
model edge, and the discrimination numbers argue against the flattering
interpretation. Dropping the sub-10c longshots leaves
ROI +{RO['ex_longshots']['roi_pct']}% on {RO['ex_longshots']['n']} bets, so it is
not purely a lottery-ticket artifact either &mdash; which makes it worth a
preregistered prospective test, and nothing more until then.</div>
</section>

<section>
<h2><span class="n">10</span>Preregistration (frozen before scoring)</h2>
<div class="disc"><div class="dhead">testing_lab/v10/preregister.v10.md</div>
<pre>{esc(PREREG)}</pre></div>
</section>

<script>
const PAL = __PAL__, TRF = __TRF__, PHA = __PHA__, MON = __MON__,
      ROS = __ROS__, WGT = __WGT__, ROI = __ROI__;
Chart.defaults.font.family = "'DM Sans',system-ui,sans-serif";
Chart.defaults.color = '#6b6478';
Chart.defaults.animation = false;
const floorBand = {{ id:'floorBand', beforeDatasetsDraw(c) {{
  const fb = c.options.plugins.floorBand; if (!fb || fb.lo === undefined) return;
  const ax = fb.axis === 'x' ? c.scales.x : c.scales.y, ctx = c.ctx, ca = c.chartArea;
  const p0 = ax.getPixelForValue(fb.lo), p1 = ax.getPixelForValue(fb.hi);
  ctx.save(); ctx.fillStyle = 'rgba(154,147,166,0.16)';
  if (fb.axis === 'x') ctx.fillRect(Math.min(p0,p1), ca.top, Math.abs(p1-p0), ca.bottom-ca.top);
  else ctx.fillRect(ca.left, Math.min(p0,p1), ca.right-ca.left, Math.abs(p1-p0));
  ctx.restore(); }} }};
Chart.register(floorBand);

new Chart(document.getElementById('c_transfer'), {{ type:'bar',
 data:{{ labels: TRF.labels, datasets:[
   {{ label:'2025 (VAL1)', data:TRF.t1, backgroundColor:PAL.s1, borderRadius:4, maxBarThickness:54 }},
   {{ label:'2026H1 (VAL2)', data:TRF.t2, backgroundColor:PAL.d5, borderRadius:4, maxBarThickness:54 }} ] }},
 options:{{ responsive:true, maintainAspectRatio:false,
  plugins:{{ legend:{{position:'bottom', labels:{{boxWidth:14, boxHeight:3}}}},
    floorBand:{{axis:'y', lo:-TRF.floor, hi:TRF.floor}},
    tooltip:{{callbacks:{{label:(t)=>t.dataset.label+': '+(t.parsed.y>=0?'+':'')+t.parsed.y.toFixed(2)+'m'}}}} }},
  scales:{{ x:{{grid:{{display:false}}}},
    y:{{grid:{{color:'#eceef2'}}, title:{{display:true, text:'Δ log-loss vs v6, milli (>0 = better)'}}}} }} }} }});

new Chart(document.getElementById('c_phase'), {{ type:'bar',
 data:{{ labels: PHA.labels, datasets:[{{ label:'Δ vs v6 (milli-LL)', data:PHA.delta,
   backgroundColor: PHA.delta.map(v=>v>=0?PAL.good:PAL.bad), borderRadius:4, maxBarThickness:60 }}] }},
 options:{{ responsive:true, maintainAspectRatio:false,
  plugins:{{ legend:{{display:false}}, floorBand:{{axis:'y', lo:-TRF.floor, hi:TRF.floor}},
    tooltip:{{callbacks:{{label:(t)=>(t.parsed.y>=0?'+':'')+t.parsed.y.toFixed(2)+'m  (n='+PHA.n[t.dataIndex]+')'}}}} }},
  scales:{{ x:{{grid:{{display:false}}}},
    y:{{grid:{{color:'#eceef2'}}, title:{{display:true, text:'Δ log-loss vs v6, milli'}}}} }} }} }});

new Chart(document.getElementById('c_month'), {{ type:'line',
 data:{{ labels: MON.labels, datasets:[{{ label:'Δ vs v6 (milli-LL)', data:MON.delta,
   borderColor:PAL.s1, backgroundColor:'rgba(124,77,214,.10)', fill:true, tension:.3,
   pointRadius:3, pointBackgroundColor:PAL.s1 }}] }},
 options:{{ responsive:true, maintainAspectRatio:false,
  plugins:{{ legend:{{display:false}}, floorBand:{{axis:'y', lo:-TRF.floor, hi:TRF.floor}},
    tooltip:{{callbacks:{{label:(t)=>(t.parsed.y>=0?'+':'')+t.parsed.y.toFixed(2)+'m  (n='+MON.n[t.dataIndex]+')'}}}} }},
  scales:{{ x:{{grid:{{display:false}}, title:{{display:true, text:'calendar month'}}}},
    y:{{grid:{{color:'#eceef2'}}, title:{{display:true, text:'Δ log-loss vs v6, milli'}}}} }} }} }});

new Chart(document.getElementById('c_roi'), {{ type:'bar',
 data:{{ labels: ROI.thresh.map(t=>'>='+t+'c'), datasets:[
   {{ label:'ROI %', data:ROI.roi, backgroundColor:PAL.a1, borderRadius:4, maxBarThickness:52 }} ] }},
 options:{{ responsive:true, maintainAspectRatio:false,
  plugins:{{ legend:{{display:false}},
    tooltip:{{callbacks:{{label:(t)=>'ROI '+t.parsed.y.toFixed(1)+'%  (n='+ROI.n[t.dataIndex]+' bets)'}}}} }},
  scales:{{ x:{{grid:{{display:false}}, title:{{display:true, text:'minimum edge vs the T-2h price'}}}},
    y:{{grid:{{color:'#eceef2'}}, title:{{display:true, text:'ROI %'}}}} }} }} }});

new Chart(document.getElementById('c_roster'), {{ type:'bar',
 data:{{ labels: ROS.labels, datasets:[{{ label:'team-years', data:ROS.dist,
   backgroundColor: ROS.labels.map((_,i)=> i<=2 ? PAL.d5 : PAL.a1), borderRadius:4, maxBarThickness:60 }}] }},
 options:{{ responsive:true, maintainAspectRatio:false,
  plugins:{{ legend:{{display:false}},
    tooltip:{{callbacks:{{label:(t)=>t.parsed.y+' team-years kept '+ROS.labels[t.dataIndex]}}}} }},
  scales:{{ x:{{grid:{{display:false}}, title:{{display:true, text:'players retained across the January boundary'}}}},
    y:{{grid:{{color:'#eceef2'}}, title:{{display:true, text:'team-years'}}}} }} }} }});
</script>
</div></body></html>"""

html = (HTML.replace("__PAL__", json.dumps(C))
            .replace("__TRF__", json.dumps(page_transfer))
            .replace("__PHA__", json.dumps(page_phase))
            .replace("__MON__", json.dumps(page_month))
            .replace("__ROS__", json.dumps(page_roster))
            .replace("__WGT__", json.dumps(page_weight))
            .replace("__ROI__", json.dumps(page_roi)))

out = os.path.join(RD, "v10_lab.html")
with open(out, "w") as f:
    f.write(html)
print(f"wrote {out}  ({len(html)} bytes)")
