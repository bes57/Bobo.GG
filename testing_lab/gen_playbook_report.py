"""Build /testing/report/playbook_bt — the simple, backtested playbook.

Reads testing_lab/v10/stats/v10_playbook.json (written by run_playbook.py, on
the corrected pre-match book from build_prematch.py) and writes
testing_lab/out/reports/playbook_bt.html.

Run: python3 testing_lab/gen_playbook_report.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
V10S = os.path.join(HERE, "v10", "stats")
RD = os.path.join(HERE, "out", "reports")

PB = json.load(open(os.path.join(V10S, "v10_playbook.json")))
LA = json.load(open(os.path.join(V10S, "v10_live_autopsy.json")))
with open(os.path.join(HERE, "gen_v9_report.py")) as f:
    CSS = f.read().split('CSS = """', 1)[1].split('"""', 1)[0]

C = {"s1": "#7c4dd6", "d5": "#c96a2a", "a1": "#3a90cc", "gray": "#9a93a6",
     "good": "#1e7a4f", "bad": "#c0392b"}

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
<a href="/testing/report/edge_lab">Edge vs Market</a>
<a href="/testing/report/playbook_bt" class="on">Playbook (backtested)</a>
</div>"""


def dl(n):
    return (f'<div class="dl"><a href="/testing/edge/stats/{n}" download>'
            f'&#8681; {n}</a></div>')


def sg(x, nd=2):
    return f"{x:+.{nd}f}"


T = PB["chosen_T"]
tr, te = PB["train_at_T"], PB["test_at_T"]
mk = PB["maker_bound"]["test"]
nofee, mid = PB["no_fee_test"], PB["mid_no_fee_test"]

page_curve = {"T": [r["T"] for r in PB["test_curve"]],
              "test": [r["roi"] for r in PB["test_curve"]],
              "train": [next((x["roi"] for x in PB["train_curve"] if x["T"] == r["T"]), None)
                        for r in PB["test_curve"]],
              "n": [r["n"] for r in PB["test_curve"]]}
page_stack = {"labels": ["at the mid,\\nno fee", "pay the ask", "+ Kalshi fee",
                         "maker bound\\n(optimistic)"],
              "roi": [mid["roi"], nofee["roi"], te["roi"], mk["roi"]]}
json.dump(page_curve, open(os.path.join(V10S, "v10_page_pb_curve.json"), "w"), indent=1)
json.dump(page_stack, open(os.path.join(V10S, "v10_page_pb_stack.json"), "w"), indent=1)

curve_rows = "".join(
    f'<tr><td class="mono">{r["T"]}c</td><td class="mono">{r["n"]}</td>'
    f'<td class="mono">{r["hit"]}%</td>'
    f'<td class="mono {"good" if r["roi"] > 0 else "bad"}">{sg(r["roi"])}%</td>'
    f'<td class="mono dim">[{r["ci"][0]:.0f}, {r["ci"][1]:.0f}]</td></tr>'
    for r in PB["test_curve"])

HTML = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Playbook (backtested) — one rule, tested honestly</title>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@800&family=DM+Sans:wght@400;700&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
<style>{CSS}
.banner b {{ color:var(--bad); }}
.rule {{ font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.05rem;
        background:var(--accbg); border-left:6px solid var(--acc); border-radius:0 14px 14px 0;
        padding:16px 20px; margin:14px 0; line-height:1.5; }}</style></head><body><div class="wrap">

<h1>Playbook (backtested)</h1>
<div class="tagline">One rule. Real pre-match prices, real costs, threshold chosen out-of-sample.</div>
{NAV}

<div class="banner" style="background:var(--badbg);border-color:#e8cfcf;border-left-color:var(--bad)">
<b>THE ANSWER — DO NOT TRADE IT.</b> The rule below, with its threshold picked on
the first half of the window and scored on the second, returns
<b>{sg(te['roi'])}%</b> out-of-sample
(p(profit)&nbsp;{te['p_profit']}). It is profitable before costs and unprofitable
after them. Even the <em>optimistic</em> maker version &mdash; you post the bid,
you always get filled, fees are zero &mdash; is {sg(mk['roi'])}% with an interval
from {mk['ci'][0]:.0f} to {mk['ci'][1]:.0f}. There is no threshold on the curve
that survives honest pricing.
</div>

<section>
<h2><span class="n">1</span>The rule</h2>
<div class="rule">Two hours before a match, buy the side whose BenPom
probability exceeds the Kalshi <em>ask</em> by at least {T} cents.
One contract. Hold to settlement.</div>
<p>That is the whole strategy. No sizing ladder, no windows, no regional
carve-outs &mdash; those all came from slicing a small sample and are how the
previous playbook got its numbers. One threshold, chosen once, on data it was
not scored on.</p>
<div class="cards">
  <div class="card"><div class="lbl">Matches</div><div class="big">{PB['n_matches']}</div>
    <div class="sub">{PB['n_legs']} sides evaluated</div></div>
  <div class="card"><div class="lbl">Threshold</div><div class="big">{T}c</div>
    <div class="sub">chosen on the train half only</div></div>
  <div class="card"><div class="lbl">Out-of-sample</div>
    <div class="big bad">{sg(te['roi'])}%</div><div class="sub">n={te['n']} bets</div></div>
  <div class="card"><div class="lbl">p(profitable)</div>
    <div class="big">{te['p_profit']}</div><div class="sub">a coin flip</div></div>
</div>
</section>

<section>
<h2><span class="n">2</span>What makes this backtest different</h2>
<p>Three defects broke every previous read on this question. All are fixed here.</p>
<div class="scroll"><table>
<thead><tr><th>Defect</th><th>Previous reads</th><th>Here</th></tr></thead><tbody>
<tr><td>Price timestamp</td><td class="bad">T&minus;2h from market <em>close</em>, and these markets close when a winner is declared &mdash; 74% of prices were sampled mid-match</td>
    <td class="good">T&minus;2h from the real scheduled start, rebuilt from raw candles; median offset &minus;120 min, nothing post-start</td></tr>
<tr><td>Execution</td><td class="bad">transacted at the mid</td>
    <td class="good">pays the ask, with the real book spread</td></tr>
<tr><td>Fees</td><td class="bad">none</td>
    <td class="good">Kalshi taker fee <span class="mono">ceil(0.07&middot;C&middot;P&middot;(1&minus;P))</span> on every ticket</td></tr>
<tr><td>Threshold choice</td><td class="bad">swept, best reported</td>
    <td class="good">chosen on the first half, scored on the second</td></tr>
</tbody></table></div>
<p class="cap">Both sides of every match are evaluated independently, so the rule
never needs to know who won in order to decide whether to bet.</p>
</section>

<section>
<h2><span class="n">3</span>Where the money goes</h2>
<div class="chartbox"><canvas id="c_stack"></canvas></div>
<p class="cap">Same {te['n']} bets, priced four ways. The rule clears a small
profit at an untransactable mid, loses it to the spread, and goes negative on the
fee. The fee is roughly 2c on a 30c ticket &mdash; 6&ndash;7% of stake.</p>
{dl('v10_page_pb_stack.json')}
<div class="callout">The maker bound is the only positive column, and it is
optimistic twice over: it assumes you are filled at your own bid on every bet you
want, and it charges no fee. Real maker fills are the opposite of free &mdash;
the live autopsy measured <b>{abs(LA['adverse_selection']['adverse_selection_cents_per_contract']):.1f}c
per contract</b> of adverse selection, which is larger than the entire
{sg(mk['roi'])}% it shows here.</div>
</section>

<section>
<h2><span class="n">4</span>The threshold curve &mdash; and why not to read it hopefully</h2>
<div class="chartbox"><canvas id="c_curve"></canvas></div>
<p class="cap">Out-of-sample ROI at every threshold, with the train half for
comparison. The train curve peaked at {T}c; the test curve does not agree, which
is the normal fate of a swept threshold on a few hundred bets.</p>
{dl('v10_page_pb_curve.json')}
<div class="scroll"><table>
<thead><tr><th>Threshold</th><th>Bets</th><th>Hit</th><th>ROI (test)</th><th>95% CI</th></tr></thead>
<tbody>{curve_rows}</tbody></table></div>
<div class="callout bad">Two cells are positive. Both sit on tiny samples with
intervals spanning fifty points or more, and picking either one <em>after</em>
seeing this table is precisely the error this page exists to avoid.</div>
</section>

<section>
<h2><span class="n">5</span>What would make this tradeable</h2>
<p>Not a different threshold &mdash; the curve has no hiding place. The three
things that would actually change the answer, in order of size:</p>
<ol>
<li><b>Stop paying the spread and the fee.</b> Quoting as a maker rather than
lifting the ask is worth roughly {mk['roi'] - te['roi']:.0f} points here. That is
the entire gap between a losing rule and a break-even one, which is why the
existing deployment playbook is a maker strategy. It is also where adverse
selection lives, so the gain is not free.</li>
<li><b>A model that discriminates better.</b> On this window BenPom picks the
winner slightly less often than the market does. A rule built on it is trying to
buy an information edge that the measurements do not show.</li>
<li><b>More data.</b> {PB['n_matches']} matches over three months is not enough to
resolve an effect of the size in question; the intervals here are ±30 points.
This market has not existed for long enough to answer the question yet.</li>
</ol>
<div class="callout warn"><b>Standing.</b> Reporting only. Market data is never a
fitting target or a selection signal, and nothing here changes the deployed
model or the trading configuration.</div>
</section>

<script>
const PAL = __PAL__, CURVE = __CURVE__, STACK = __STACK__;
Chart.defaults.font.family = "'DM Sans',system-ui,sans-serif";
Chart.defaults.color = '#6b6478';
Chart.defaults.animation = false;

new Chart(document.getElementById('c_stack'), {{ type:'bar',
 data:{{ labels: STACK.labels.map(l=>l.split('\\n')), datasets:[
  {{ data: STACK.roi, borderRadius:4, maxBarThickness:64,
     backgroundColor: STACK.roi.map((v,i)=> i===3 ? PAL.gray : (v>=0?PAL.a1:PAL.bad)) }} ] }},
 options:{{ responsive:true, maintainAspectRatio:false,
  plugins:{{ legend:{{display:false}},
   tooltip:{{callbacks:{{label:(t)=>'ROI '+t.parsed.y.toFixed(2)+'%'}}}} }},
  scales:{{ x:{{grid:{{display:false}}, ticks:{{font:{{size:10}}}}}},
   y:{{grid:{{color:'#eceef2'}}, title:{{display:true, text:'ROI %'}}}} }} }} }});

new Chart(document.getElementById('c_curve'), {{ type:'line',
 data:{{ labels: CURVE.T.map(t=>t+'c'), datasets:[
  {{ label:'out-of-sample (test half)', data:CURVE.test, borderColor:PAL.s1,
     backgroundColor:'rgba(124,77,214,.10)', fill:true, tension:.25, pointRadius:3 }},
  {{ label:'train half (where T was chosen)', data:CURVE.train, borderColor:PAL.gray,
     borderDash:[5,4], fill:false, tension:.25, pointRadius:2 }} ] }},
 options:{{ responsive:true, maintainAspectRatio:false,
  plugins:{{ legend:{{position:'bottom', labels:{{boxWidth:14, boxHeight:3}}}},
   tooltip:{{callbacks:{{label:(t)=>t.dataset.label+': '+t.parsed.y.toFixed(1)+'%'
     +(t.datasetIndex===0?'  (n='+CURVE.n[t.dataIndex]+')':'')}}}} }},
  scales:{{ x:{{grid:{{display:false}}, title:{{display:true, text:'minimum edge over the ask'}}}},
   y:{{grid:{{color:'#eceef2'}}, title:{{display:true, text:'ROI %'}}}} }} }} }});
</script>
</div></body></html>"""

html = (HTML.replace("__PAL__", json.dumps(C))
            .replace("__CURVE__", json.dumps(page_curve))
            .replace("__STACK__", json.dumps(page_stack)))
out = os.path.join(RD, "playbook_bt.html")
with open(out, "w") as f:
    f.write(html)
print(f"wrote {out}  ({len(html)} bytes)")
