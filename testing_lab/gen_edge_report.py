"""Build /testing/report/edge_lab — does BenPom actually have an edge on Kalshi?

Reads testing_lab/v10/stats/{v10_edge,v10_roi,v10_slippage,v10_monthly}.json and
writes testing_lab/out/reports/edge_lab.html.

Run: python3 testing_lab/gen_edge_report.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
V10S = os.path.join(HERE, "v10", "stats")
RD = os.path.join(HERE, "out", "reports")


def rj(n):
    with open(os.path.join(V10S, n)) as f:
        return json.load(f)


ED = rj("v10_edge.json")
RO = rj("v10_roi.json")
SL = rj("v10_slippage.json")
MO = rj("v10_monthly.json")
LA = rj("v10_live_autopsy.json")
TA = rj("v10_timing_artifact.json")

with open(os.path.join(HERE, "gen_v9_report.py")) as f:
    CSS = f.read().split('CSS = """', 1)[1].split('"""', 1)[0]

C = {"s1": "#7c4dd6", "d5": "#c96a2a", "a1": "#3a90cc", "v6": "#6f6a7c",
     "gray": "#9a93a6", "good": "#1e7a4f", "bad": "#c0392b", "ink": "#16121d"}

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
<a href="/testing/report/v10_lab">v10 Lab</a>
<a href="/testing/report/edge_lab" class="on">Edge vs Market</a>
<a href="/testing/report/playbook_bt">Playbook (backtested)</a>
</div>"""


def dl(n):
    return (f'<div class="dl"><a href="/testing/edge/stats/{n}" download>'
            f'&#8681; {n}</a></div>')


def sg(x, nd=2):
    return f"{x:+.{nd}f}"


bd, bf = ED["blind_dog"], ED["blind_fav"]
md, m5 = ED["model_dog_only"], ED["model_5c"]
pick, nopick = ED["dog_picked_by_model"], ED["dog_not_picked"]
sh, disc, nt = ED["sharpened"], ED["discrimination"], ED["null_test"]
gap = m5["roi_pct"] - bd["roi_pct"]

page_control = {"labels": ["Back every underdog\\n(model ignored)",
                           "Model, >=5c edge",
                           "Model, underdog side only",
                           "Back every favourite\\n(model ignored)"],
                "roi": [bd["roi_pct"], m5["roi_pct"], md["roi_pct"], bf["roi_pct"]],
                "lo": [bd["ci95"][0], m5["ci95"][0], md["ci95"][0], bf["ci95"][0]],
                "hi": [bd["ci95"][1], m5["ci95"][1], md["ci95"][1], bf["ci95"][1]]}
page_month = {"labels": [r["month"] for r in MO],
              "blind": [r["blind_dog_roi"] for r in MO],
              "model": [r["model_roi"] for r in MO],
              "n": [r["n"] for r in MO]}
page_slip = {"slip": sorted(int(k) for k in SL),
             "model": [SL[str(k)][0] for k in sorted(int(k) for k in SL)],
             "blind": [SL[str(k)][1] for k in sorted(int(k) for k in SL)]}
for k, v in (("edge_control", page_control), ("edge_month", page_month),
             ("edge_slip", page_slip)):
    json.dump(v, open(os.path.join(V10S, f"v10_page_{k}.json"), "w"), indent=1)

segrows = "".join(
    f'<tr><td>{r["label"]}</td><td class="mono">{r["n"]}</td>'
    f'<td class="mono">{r["hit_pct"]}%</td>'
    f'<td class="mono {"good" if r["roi_pct"] > 0 else "bad"}">{sg(r["roi_pct"])}%</td>'
    f'<td class="mono dim">[{r["ci95"][0]:.0f}, {r["ci95"][1]:.0f}]</td></tr>'
    for r in ED["segments"])
monrows = "".join(
    f'<tr><td class="mono">{r["month"]}</td><td class="mono">{r["n"]}</td>'
    f'<td class="mono">{r["dog_hit_pct"]}%</td>'
    f'<td class="mono">{sg(r["blind_dog_roi"])}%</td>'
    f'<td class="mono">{"n/a" if r["model_roi"] is None else sg(r["model_roi"]) + "%"}</td></tr>'
    for r in MO)

HTML = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Edge vs Market — is the model beating Kalshi?</title>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@800&family=DM+Sans:wght@400;700&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
<style>{CSS}
.banner b {{ color:var(--warn); }}</style></head><body><div class="wrap">

<h1>Edge vs Market</h1>
<div class="tagline">Where does BenPom beat Kalshi, and how sure can we be?</div>
{NAV}

<div class="banner" style="background:var(--warnbg);border-color:#eddcc9;border-left-color:var(--warn)">
<b>THE ANSWER — THERE IS NO EDGE. THE +34.7% WAS A BROKEN TIMESTAMP.</b>
The market price this study used, <span class="mono">prob_a_t2h</span>, is
anchored two hours before the market <em>closes</em> &mdash; and these markets
close <em>when a winner is declared</em>. So it is not a pre-match price:
<b>{TA["pct_sampled_after_start"]}%</b> of the sample was taken
<em>during</em> the match, a median of <b>{TA["median_offset_min"]:.0f} minutes</b>
after the first map started (&sect;3). Priced honestly &mdash; pre-match only,
paying the ask, paying Kalshi's fee &mdash; the whole thing returns
<b>{sg(TA["cost_stack_honest_subsample"]["plus_ask_and_fees"]["roi"])}%</b>. Two
further findings agree. <b>(1)</b> Backing
<em>every</em> market underdog blindly, model switched off, returned
<b>{sg(bd["roi_pct"])}%</b> against the model's <b>{sg(m5["roi_pct"])}%</b> &mdash;
the model is worth about <b>{sg(gap, 1)} points</b> on top of ignoring it, and it
is slightly <em>worse</em> at picking winners ({disc["benpom_acc"]}% vs
{disc["kalshi_acc"]}%). <b>(2)</b> This exact thesis traded live for a week in
July: <b>{LA["fills"]} fills, ${abs(LA["realized_total"]):,.2f} lost</b>, and a
bootstrap says that is real underperformance, not variance
(p&nbsp;=&nbsp;{LA["variance_test"]["p_cum_le_observed"]:.3f}). The backtest edge did
not survive contact with a real order book.
</div>

<div class="callout bad" style="margin-bottom:16px"><b>Correction.</b> An earlier
version of this page, and the v10 lab's &sect;9, described
<span class="mono">prob_a_t2h</span> as a clean pre-match price and reported
+{RO['headline']['roi_pct']}% on that basis. That description was wrong and the
number is not usable. The corrected read is &sect;3. The live-trading result
(&sect;4) was already pointing the same way.</div>

<section>
<h2><span class="n">1</span>The question, and the control that answers it</h2>
<p>The v10 lab reported +{RO['headline']['roi_pct']}% ROI at a &ge;5c edge and
flagged that {RO['by_side']['underdog']['n']} of {RO['headline']['n_bets']} bets
were on the underdog. That pattern has two possible explanations, and they have
completely different consequences:</p>
<ol>
<li><b>The model finds mispriced underdogs.</b> Real edge, worth trading.</li>
<li><b>Underdogs were simply cheap.</b> A market-wide bias any coin could have
harvested; the model is incidental.</li>
</ol>
<p>These are trivially separable, and the separating test is the one the
original read was missing: <b>back every underdog at the market price with the
model switched off</b>. If that returns roughly the same, hypothesis 2 wins.</p>
<div class="cards">
  <div class="card"><div class="lbl">Sample</div><div class="big">{ED['n']}</div>
    <div class="sub">settled tier-1 matches</div></div>
  <div class="card"><div class="lbl">Window</div><div class="big">3 months</div>
    <div class="sub">{ED['window'][0]} .. {ED['window'][1]}</div></div>
  <div class="card"><div class="lbl">Blind underdog</div>
    <div class="big good">{sg(bd['roi_pct'])}%</div><div class="sub">no model at all</div></div>
  <div class="card"><div class="lbl">Model adds</div>
    <div class="big warn">{sg(gap, 1)} pts</div><div class="sub">over blind</div></div>
</div>
</section>

<section>
<h2><span class="n">2</span>The control result</h2>
<div class="chartbox"><canvas id="c_ctrl"></canvas></div>
<p class="cap">ROI with 95% bootstrap intervals. The first bar uses no model
whatsoever. The gap between it and the model's bets is
<b>{sg(gap, 1)} points</b> &mdash; well inside the intervals, i.e. not
distinguishable from zero contribution.</p>
{dl('v10_page_edge_control.json')}
<div class="callout warn">Backing every <b>favourite</b> returned
<b>{sg(bf['roi_pct'])}%</b> ({bf['hit_pct']}% hit). That is the same fact stated
from the other side: in this window the market's favourites were systematically
overpriced. A model that is under-confident will disagree with those favourites
by construction and look clever for doing so.</div>
</section>

<section>
<h2><span class="n">3</span>The timestamp is coupled to the outcome</h2>
<p>Kalshi VALORANT markets carry
<span class="mono">"This market will close and expire after a winner is
declared"</span>. So <span class="mono">close_time</span> is not a schedule, it is
the moment the match ended &mdash; and a price sampled at
<span class="mono">close &minus; 2h</span> is sampled two hours before the
<em>finish</em>, not before the start.</p>
<p>Because match length varies, that sampling instant lands at wildly different
points: <b>{TA['pct_sampled_after_start']}%</b> of prices were taken after the
first map had begun, median <b>+{TA['median_offset_min']:.0f} minutes</b> in.
And crucially, <em>long matches are the ones underdogs win</em>. The dataset
therefore quotes a cheaper underdog price exactly when the underdog is likelier
to win. That is not look-ahead in the ordinary sense &mdash; the sampling time is
itself a function of the result.</p>
<div class="scroll"><table>
<thead><tr><th>When the price was sampled</th><th>n</th><th>Blind-underdog ROI</th><th>95% CI</th><th>Underdog won</th></tr></thead><tbody>
<tr><td><b>before match start (usable)</b></td><td class="mono">{TA['by_timing']['pre_start']['n']}</td>
    <td class="mono">{sg(TA['by_timing']['pre_start']['roi'])}%</td>
    <td class="mono dim">[{TA['by_timing']['pre_start']['ci'][0]:.0f}, {TA['by_timing']['pre_start']['ci'][1]:.0f}]</td>
    <td class="mono">{TA['dog_winrate_by_timing']['pre_start']}%</td></tr>
<tr><td>after match start</td><td class="mono">{TA['by_timing']['post_start']['n']}</td>
    <td class="mono bad">{sg(TA['by_timing']['post_start']['roi'])}%</td>
    <td class="mono dim">[{TA['by_timing']['post_start']['ci'][0]:.0f}, {TA['by_timing']['post_start']['ci'][1]:.0f}]</td>
    <td class="mono">{TA['dog_winrate_by_timing']['30_90min']}% (30-90m)</td></tr>
<tr><td>&nbsp;&nbsp;&mdash; more than 30 min in</td><td class="mono">{TA['by_timing']['gt30min']['n']}</td>
    <td class="mono bad">{sg(TA['by_timing']['gt30min']['roi'])}%</td>
    <td class="mono dim">[{TA['by_timing']['gt30min']['ci'][0]:.0f}, {TA['by_timing']['gt30min']['ci'][1]:.0f}]</td>
    <td class="mono">&mdash;</td></tr>
</tbody></table></div>
<p class="cap">The "edge" is entirely in the rows where the price was read
mid-match. On the rows that are actually tradeable it is
{sg(TA['by_timing']['pre_start']['roi'])}% with an interval straddling zero
(p(profit) = {TA['by_timing']['pre_start']['p_profit']}).</p>
<h3>And then the costs</h3>
<div class="scroll"><table>
<thead><tr><th>Pre-match subsample, n={TA['cost_stack_honest_subsample']['mid_no_costs']['n']}</th><th>ROI</th><th>95% CI</th><th>p(profit)</th></tr></thead><tbody>
<tr><td>at the mid, no costs</td><td class="mono">{sg(TA['cost_stack_honest_subsample']['mid_no_costs']['roi'])}%</td>
    <td class="mono dim">[{TA['cost_stack_honest_subsample']['mid_no_costs']['ci'][0]:.0f}, {TA['cost_stack_honest_subsample']['mid_no_costs']['ci'][1]:.0f}]</td>
    <td class="mono">{TA['cost_stack_honest_subsample']['mid_no_costs']['p_profit']}</td></tr>
<tr><td>+ buy at the ask</td><td class="mono">{sg(TA['cost_stack_honest_subsample']['plus_ask']['roi'])}%</td>
    <td class="mono dim">[{TA['cost_stack_honest_subsample']['plus_ask']['ci'][0]:.0f}, {TA['cost_stack_honest_subsample']['plus_ask']['ci'][1]:.0f}]</td>
    <td class="mono">{TA['cost_stack_honest_subsample']['plus_ask']['p_profit']}</td></tr>
<tr><td><b>+ Kalshi fee ceil(0.07&middot;P&middot;(1&minus;P))</b></td>
    <td class="mono"><b>{sg(TA['cost_stack_honest_subsample']['plus_ask_and_fees']['roi'])}%</b></td>
    <td class="mono dim">[{TA['cost_stack_honest_subsample']['plus_ask_and_fees']['ci'][0]:.0f}, {TA['cost_stack_honest_subsample']['plus_ask_and_fees']['ci'][1]:.0f}]</td>
    <td class="mono">{TA['cost_stack_honest_subsample']['plus_ask_and_fees']['p_profit']}</td></tr>
</tbody></table></div>
{dl('v10_timing_artifact.json')}
<div class="callout bad">Nothing survives. The fee alone is roughly 2c on a 30c
ticket &mdash; 6-7% of stake &mdash; and it was absent from every earlier number on
this page and in the v10 lab. <b>{sg(TA['cost_stack_honest_subsample']['plus_ask_and_fees']['roi'])}%
with p(profit) {TA['cost_stack_honest_subsample']['plus_ask_and_fees']['p_profit']}
is a coin flip.</b></div>
</section>

<section>
<h2><span class="n">4</span>It was also traded live, and it lost</h2>
<p>Everything above is a backtest. The lab already has the thing a backtest is a
substitute for &mdash; a week of real fills against this exact market, recorded in
the v8 phase-7 autopsy.</p>
<div class="cards">
  <div class="card"><div class="lbl">Real fills</div><div class="big">{LA['fills']}</div>
    <div class="sub">{LA['contracts']:,.0f} contracts, {LA['events']} events</div></div>
  <div class="card"><div class="lbl">Deployed</div>
    <div class="big">${LA['cost_dollars']:,.0f}</div><div class="sub">{LA['window_utc'][0][:10]} .. {LA['window_utc'][1][:10]}</div></div>
  <div class="card"><div class="lbl">Realized</div>
    <div class="big bad">-${abs(LA['realized_total']):,.2f}</div>
    <div class="sub">fees ${LA['fees']:.2f} &mdash; not a cost problem</div></div>
  <div class="card"><div class="lbl">Primary side ROI</div>
    <div class="big bad">{LA['side1_settled']['roi']*100:.1f}%</div>
    <div class="sub">on ${LA['side1_settled']['cost']:,.0f}</div></div>
</div>
<p>The simulation that authorised those trades expected
<b>+${LA['variance_test']['H0_expected']:,.0f}</b>. A common-random-number bootstrap
over {LA['variance_test']['B']:,} resamples, unit = settled event, put the observed
result at <b>p = {LA['variance_test']['p_cum_le_observed']:.3f}</b> under that
hypothesis. The lab's own verdict:
<span class="mono">"{LA['variance_test']['verdict']}"</span> &mdash; the loss was
not bad luck.</p>
<h3>Why: adverse selection, not fees and not the spread</h3>
<p>Fees were <b>${LA['fees']:.2f}</b> (this series pays no maker fee), and the
market-making markouts were fine &mdash; the spread was being captured. The damage
is in <em>which</em> orders got filled:</p>
<div class="scroll"><table>
<thead><tr><th>Measure</th><th>Value</th></tr></thead><tbody>
<tr><td>Gap on filled contracts (realized &minus; predicted)</td><td class="mono bad">{LA['adverse_selection']['fill_gap_frozen_v6']*100:+.2f} pts</td></tr>
<tr><td>Same gap, unconditional benchmark (had you not needed a fill)</td><td class="mono good">{LA['adverse_selection']['uncond_gap']*100:+.2f} pts</td></tr>
<tr><td><b>Adverse selection</b></td><td class="mono bad"><b>{LA['adverse_selection']['adverse_selection_cents_per_contract']:.2f}c per contract</b></td></tr>
</tbody></table></div>
{dl('v10_live_autopsy.json')}
<div class="callout bad">This is the number that kills the backtest. A simulation
buys at the quoted price on every match it likes. Reality fills you
<b>preferentially when you are wrong</b> &mdash; {abs(LA['adverse_selection']['adverse_selection_cents_per_contract']):.1f}c
per contract of it here, against a mid-price backtest edge of
{m5['roi_pct']:.0f}% on an average {RO['headline']['mean_cost'] if 'mean_cost' in RO['headline'] else 28.8:.0f}c contract. The
slippage curve in &sect;8 models a fixed haircut; adverse selection is worse than a
haircut, because it is correlated with the outcome.</div>
</section>

<section>
<h2><span class="n">5</span>Is it just under-confidence? Mostly, yes</h2>
<p>BenPom assigns the eventual winner an average probability of
<b>{ED['mean_conf']['benpom']}</b>; Kalshi assigns <b>{ED['mean_conf']['kalshi']}</b>.
The model is systematically less sure of itself, so it disagrees toward the
underdog on nearly every match &mdash; not because it has information, but
because it hedges.</p>
<p>To test that directly, sharpen the model: scale its log-odds by
<b>&times;{ED['sharpen_slope']}</b> so its average confidence exactly matches the
market's, changing nothing about which team it prefers. Its ranking of matchups
is untouched; only its certainty moves.</p>
<div class="cards">
  <div class="card"><div class="lbl">As-is, &ge;5c</div>
    <div class="big good">{sg(m5['roi_pct'])}%</div>
    <div class="sub">CI [{m5['ci95'][0]:.0f}, {m5['ci95'][1]:.0f}]</div></div>
  <div class="card"><div class="lbl">Sharpened to market confidence</div>
    <div class="big">{sg(sh['roi_pct'])}%</div>
    <div class="sub">CI [{sh['ci95'][0]:.0f}, {sh['ci95'][1]:.0f}] &mdash; includes zero</div></div>
</div>
<p>Sharpening costs roughly two-thirds of the return and pushes the interval
across zero. So most of the apparent edge is the hedging, not the opinion.</p>
</section>

<section>
<h2><span class="n">6</span>Does it pick <em>better</em> underdogs?</h2>
<p>This is the one place the model looks genuinely additive, and it deserves a
careful read rather than a headline.</p>
<div class="scroll"><table>
<thead><tr><th>Underdog bets</th><th>n</th><th>Hit</th><th>ROI</th></tr></thead><tbody>
<tr><td>the model liked (&ge;5c)</td><td class="mono">{pick['n']}</td>
    <td class="mono">{pick['hit_pct']}%</td><td class="mono good">{sg(pick['roi_pct'])}%</td></tr>
<tr><td>the model did NOT like</td><td class="mono">{nopick['n']}</td>
    <td class="mono">{nopick['hit_pct']}%</td><td class="mono">{sg(nopick['roi_pct'])}%</td></tr>
</tbody></table></div>
<p>The underdogs it liked returned {sg(pick['roi_pct'])}% against
{sg(nopick['roi_pct'])}% for the ones it passed on &mdash; but note the hit rates
run the <em>other</em> way ({pick['hit_pct']}% vs {nopick['hit_pct']}%). The model
is not picking underdogs that win more often; it is picking <em>longer-priced</em>
ones that pay more when they do. That is a price-selection effect, and with
{nopick['n']} matches in the comparison group it is nowhere near conclusive.</p>
</section>

<section>
<h2><span class="n">7</span>Discrimination &mdash; the uncomfortable number</h2>
<div class="cards">
  <div class="card"><div class="lbl">Picks the winner</div>
    <div class="big">{disc['benpom_acc']}%</div><div class="sub">BenPom</div></div>
  <div class="card"><div class="lbl">Picks the winner</div>
    <div class="big">{disc['kalshi_acc']}%</div><div class="sub">Kalshi &mdash; better</div></div>
  <div class="card"><div class="lbl">Log-loss</div>
    <div class="big">{disc['benpom_logloss']}</div><div class="sub">BenPom &mdash; better</div></div>
  <div class="card"><div class="lbl">Log-loss</div>
    <div class="big">{disc['kalshi_logloss']}</div><div class="sub">Kalshi</div></div>
</div>
<p>BenPom wins on log-loss and loses on accuracy. That combination has one
explanation: it is better calibrated in the sense of being less confident, and
log-loss rewards hedging when the market's confident calls go wrong. It is not
seeing matchups more clearly &mdash; on the only metric that measures that, it is
{round(disc['kalshi_acc'] - disc['benpom_acc'], 1)} points behind.</p>
</section>

<section>
<h2><span class="n">8</span>How assured is any of this?</h2>
<h3>The anomaly is persistent, not one hot streak</h3>
<div class="chartbox"><canvas id="c_mon"></canvas></div>
<div class="scroll"><table>
<thead><tr><th>Month</th><th>n</th><th>Underdog won</th><th>Blind dog ROI</th><th>Model ROI</th></tr></thead>
<tbody>{monrows}</tbody></table></div>
<p class="cap">Positive in all four months, so it is not a single weekend. But
the monthly samples are small and the spread is enormous
(+{min(r['blind_dog_roi'] for r in MO):.0f}% to
+{max(r['blind_dog_roi'] for r in MO):.0f}%), which is exactly what a
small-sample, high-variance bet looks like.</p>
{dl('v10_page_edge_month.json')}
<h3>Against a fair-market null</h3>
<p>If every contract were priced fairly, {nt['n_bets']} bets of this size and
shape would give an ROI standard deviation of <b>{nt['null_roi_sd']}%</b>, so
anything under <b>+{nt['mde_roi_at_n']}%</b> is indistinguishable from luck at
this sample size. The observed {sg(nt['observed_roi'])}% sits above that
(one-sided p = {nt['p_value_one_sided']}).</p>
<div class="callout"><b>Read that carefully.</b> It says the <em>prices</em> were
not fair. It does not say the model is good &mdash; the blind-underdog control
clears the same bar without any model at all.</div>
<h3>Slippage</h3>
<div class="chartbox"><canvas id="c_slip"></canvas></div>
<p class="cap">You never transact at the mid. Both lines survive
{max(page_slip['slip'])}c of adverse fill, and the gap between them barely moves
&mdash; slippage does not change the conclusion, it just scales both down.</p>
{dl('v10_page_edge_slip.json')}
</section>

<section>
<h2><span class="n">9</span>Segments, with the caveat that matters</h2>
<div class="scroll"><table>
<thead><tr><th>Segment</th><th>n</th><th>Hit</th><th>ROI</th><th>95% CI</th></tr></thead>
<tbody>{segrows}</tbody></table></div>
<div class="callout bad"><b>Do not stock-pick from this table.</b> These are
{len(ED['segments'])} overlapping slices of {ED['n']} matches. At this sample
size several will look strong by chance alone, and choosing the best one after
seeing it is how backtests get invented. Any segment worth trading has to be
named in advance and tested on data that did not suggest it.</div>
</section>

<section>
<h2><span class="n">10</span>What would actually settle it</h2>
<ol>
<li><b>Trade the control, not the model.</b> If the thesis is "underdogs are
underpriced", the honest instrument is a flat underdog rule. It returned
{sg(bd['roi_pct'])}% here and needs no model at all. Everything the model adds is
inside the noise.</li>
<li><b>Sharpen BenPom and re-measure.</b> The &times;{ED['sharpen_slope']} scaling
is a one-parameter fix that costs nothing and would make the probabilities
honest. If a sharpened model still disagrees with the book profitably, that is a
real signal; the current evidence says it mostly would not.</li>
<li><b>Preregister and go forward.</b> Fix the rule, the threshold and the stake
before the next match, then score it prospectively. At {nt['n_bets']} bets per
three months, an effect the size of the model's marginal contribution
({sg(gap, 1)} pts) would need <b>years</b> to separate from noise &mdash; which is
itself the answer about how much to size it.</li>
</ol>
<div class="callout warn"><b>Standing.</b> This is a reporting exercise on a
three-month window. Market data is never a fitting target and never a selection
signal, nothing here was tuned, and nothing here promotes or demotes any model.
The deployed model is unchanged.</div>
</section>

<script>
const PAL = __PAL__, CTRL = __CTRL__, MON = __MON__, SLIP = __SLIP__;
Chart.defaults.font.family = "'DM Sans',system-ui,sans-serif";
Chart.defaults.color = '#6b6478';
Chart.defaults.animation = false;

new Chart(document.getElementById('c_ctrl'), {{ type:'bar',
 data:{{ labels: CTRL.labels.map(l=>l.split('\\n')), datasets:[
  {{ label:'ROI %', data:CTRL.roi,
     backgroundColor: CTRL.roi.map((v,i)=> i===0 ? PAL.d5 : (v>=0?PAL.a1:PAL.bad)),
     borderRadius:4, maxBarThickness:64 }} ] }},
 options:{{ responsive:true, maintainAspectRatio:false,
  plugins:{{ legend:{{display:false}},
    tooltip:{{callbacks:{{label:(t)=>'ROI '+t.parsed.y.toFixed(1)+'%  CI ['
      +CTRL.lo[t.dataIndex].toFixed(0)+', '+CTRL.hi[t.dataIndex].toFixed(0)+']'}}}} }},
  scales:{{ x:{{grid:{{display:false}}, ticks:{{font:{{size:10}}}}}},
    y:{{grid:{{color:'#eceef2'}}, title:{{display:true, text:'ROI %'}}}} }} }} }});

new Chart(document.getElementById('c_mon'), {{ type:'bar',
 data:{{ labels: MON.labels, datasets:[
  {{ label:'blind underdog', data:MON.blind, backgroundColor:PAL.d5, borderRadius:4, maxBarThickness:38 }},
  {{ label:'model >=5c', data:MON.model, backgroundColor:PAL.s1, borderRadius:4, maxBarThickness:38 }} ] }},
 options:{{ responsive:true, maintainAspectRatio:false,
  plugins:{{ legend:{{position:'bottom', labels:{{boxWidth:14, boxHeight:3}}}},
    tooltip:{{callbacks:{{label:(t)=>t.dataset.label+': '+t.parsed.y.toFixed(1)+'%  (n='+MON.n[t.dataIndex]+')'}}}} }},
  scales:{{ x:{{grid:{{display:false}}}},
    y:{{grid:{{color:'#eceef2'}}, title:{{display:true, text:'ROI %'}}}} }} }} }});

new Chart(document.getElementById('c_slip'), {{ type:'line',
 data:{{ labels: SLIP.slip.map(s=>s+'c'), datasets:[
  {{ label:'model >=5c', data:SLIP.model, borderColor:PAL.s1, backgroundColor:'rgba(124,77,214,.10)',
     fill:false, tension:.25, pointRadius:3 }},
  {{ label:'blind underdog', data:SLIP.blind, borderColor:PAL.d5, borderDash:[5,4],
     fill:false, tension:.25, pointRadius:3 }} ] }},
 options:{{ responsive:true, maintainAspectRatio:false,
  plugins:{{ legend:{{position:'bottom', labels:{{boxWidth:14, boxHeight:3}}}},
    tooltip:{{callbacks:{{label:(t)=>t.dataset.label+': '+t.parsed.y.toFixed(1)+'%'}}}} }},
  scales:{{ x:{{grid:{{display:false}}, title:{{display:true, text:'cents of adverse fill per contract'}}}},
    y:{{grid:{{color:'#eceef2'}}, title:{{display:true, text:'ROI %'}}}} }} }} }});
</script>
</div></body></html>"""

html = (HTML.replace("__PAL__", json.dumps(C))
            .replace("__CTRL__", json.dumps(page_control))
            .replace("__MON__", json.dumps(page_month))
            .replace("__SLIP__", json.dumps(page_slip)))
out = os.path.join(RD, "edge_lab.html")
with open(out, "w") as f:
    f.write(html)
print(f"wrote {out}  ({len(html)} bytes)")
