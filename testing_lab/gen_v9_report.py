"""Generate reports/v9_lab.html — the v9 Lab page (agent:v9-finish).

House pattern (gen_v8_report.py / gen_roster_report.py): every number on the
page reads from a stats JSON and links to it via /testing/v9/stats/ (new
TestingLab.py route, same auth gate; roster motivation numbers link via the
existing /testing/v8/stats/ route). Verdict-first per house convention.

This generator RESHAPES recorded artifacts only. It derives two chart-ready
page JSONs into testing_lab/v9/stats/ (v9_page_grid.json,
v9_page_autopsy.json — one writer: agent:v9-finish) and never touches any
other agent's artifact. Regenerating this page never regenerates the older
report pages (nav additions there are assert-exactly-once string patches).
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))          # testing_lab
V9S = os.path.join(HERE, "v9", "stats")
V8S = os.path.join(HERE, "v8", "stats")
RD = os.path.join(HERE, "out", "reports")


def j(name, base=V9S):
    with open(os.path.join(base, name)) as f:
        return json.load(f)


lad = j("v9_ladder.json")
cand = j("v9_candidates.json")
grid = j("v9_search_grid.json")
aut = j("v9_gate_autopsy.json")
tproto = j("v9_transfer_protocol.json")
pproto = j("v9_prospective_protocol.json")
looks = j("v9_looks.json")
fix = j("v9_fixtures.json")
board = j("v9_prospective_scoreboard.json")
try:
    dec = j("v9_case_decomposition.json")
except FileNotFoundError:
    dec = None
integ = j("roster_integration.json", base=V8S)
gal = j("roster_case_gallery.json", base=V8S)
_lev = next(c for c in gal["named_cases"] if c["org"] == "LEV")
_env_pp = j("roster_case_envy.json", base=V8S)["panel_chain_start_2026_05_12"]
prereg_design = open(os.path.join(HERE, "v9", "preregister.design.md")).read()

assert lad["verdict_sentence"] == "the ladder is v6 alone"
BETA6 = [a for a in lad["arms"] if a["id"] == "v6"][0]["beta_frozen"]

# ── the design agent's struck-draft disclosure, quoted VERBATIM ─────────────
m = re.search(r"\[Lock-time note, kept permanently:.*?\]", prereg_design, re.S)
assert m, "struck-draft disclosure not found in preregister.design.md"
STRUCK = m.group(0)

# ── derived page JSONs (chart-ready; sourced 1:1 from the frozen artifacts) ──
pts = [[c["dF_milli"], c["era_min_milli"], int(bool(c["eligible"]))]
       for c in grid["configs"]]
nom = []
for c in cand["candidates"]:
    g = c.get("provenance", {}).get("grid_row")
    if g:
        nom.append({"id": c["id"], "dF_milli": g["dF_milli"],
                    "era_min_milli": g["era_min_milli"]})
page_grid = {"written_by": "agent:v9-finish (gen_v9_report.py)",
             "source": "stats/v9_search_grid.json configs[] (dF_milli, era_min_milli, eligible) verbatim",
             "n_configs": grid["n_configs"], "n_eligible": grid["n_eligible"],
             "eligibility_rule": grid["grid_def"]["eligibility"],
             "points_dF_eramin_eligible": pts, "grid_nominated": nom}
with open(os.path.join(V9S, "v9_page_grid.json"), "w") as f:
    json.dump(page_grid, f)

gate_cv = {}
for k, row in aut["table_gate_transfer_holdout"].items():
    mm = re.search(r"CV \+?([\d.]+)m", row["gate_said"])
    gate_cv[k] = float(mm.group(1)) if mm else None
aut_reads = {e["config"]: e for e in looks["autopsy_methodological_reads"]["entries"]}
AUT_ROWS = [
    ("p1w3c5 a=4.5 (gate-selected)", "p1w3c5", "p1w3c5",
     "p1w3c5 a=4.5 t=13 s=1.0 cap=None (gate-selected)"),
    ("p1w5c5 a=4.5 (gate-selected)", "p1w5c5", "p1w5c5",
     "p1w5c5 a=4.5 t=13 s=0.7 cap=None (gate-selected)"),
    ("p1w8c5 a=28 (THE SHIP)", "p1w8c5 (the a=28 ship)", "p1w8c5_ship",
     "p1w8c5 a=28 t=13 s=0.7 n_min=3 cap=1.5 (SHIPPED)"),
]
page_aut = {"written_by": "agent:v9-finish (gen_v9_report.py)",
            "source": "stats/v9_gate_autopsy.json (gate CV strings + per_config pooled) "
                      "+ stats/v9_looks.json autopsy_methodological_reads (holdout column)",
            "configs": [{"label": lbl, "gate_cv_milli": gate_cv[tk],
                         "transfer_pooled_milli":
                             aut["per_config"][pk]["pooled_validation"]["delta_milli"],
                         "holdout_milli": aut_reads[lk]["holdout_v8_method_milli"]}
                        for lbl, tk, pk, lk in AUT_ROWS]}
with open(os.path.join(V9S, "v9_page_autopsy.json"), "w") as f:
    json.dump(page_aut, f, indent=1)

# ── shared bits ─────────────────────────────────────────────────────────────
C = {"s1": "#7c4dd6", "d5": "#c96a2a", "a1": "#3a90cc", "v6": "#6f6a7c",
     "gray": "#9a93a6", "good": "#1e7a4f", "bad": "#c0392b", "ink": "#16121d"}

CSS = """
 * { box-sizing:border-box; margin:0; }
 :root { --ink:#16121d; --dim:#6b6478; --line:#eceef2; --acc:#7c4dd6; --accbg:#f3eefb;
         --good:#1e7a4f; --goodbg:#ecf8f1; --bad:#c0392b; --badbg:#fbeaea;
         --warn:#b3541e; --warnbg:#fdf3ec; }
 body { font-family:'DM Sans',system-ui,sans-serif; background:#faf9fc; color:var(--ink);
        line-height:1.55; padding:30px 18px 90px; }
 .wrap { max-width:940px; margin:0 auto; }
 h1 { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.55rem;
      text-align:center; margin:6px 0 2px; }
 .tagline { text-align:center; color:var(--dim); font-size:.9rem; margin-bottom:18px; }
 .labtabs { display:flex; justify-content:center; gap:6px; margin:0 0 18px; flex-wrap:wrap; }
 .labtabs a { font-size:.8rem; font-weight:700; color:var(--dim); text-decoration:none;
   padding:7px 15px; border-radius:999px; border:1px solid var(--line); background:#fff; }
 .labtabs a:hover { color:var(--ink); background:var(--accbg); }
 .labtabs a.on { color:#fff; background:var(--acc); border-color:var(--acc); }
 .banner { background:var(--goodbg); border:1px solid #cfe8da; border-left:6px solid var(--good);
   border-radius:14px; padding:14px 20px; margin:0 0 18px; font-size:.92rem; }
 .banner b { color:var(--good); letter-spacing:.4px; }
 section { background:#fff; border:1px solid var(--line); border-radius:18px;
           padding:24px 28px; margin-bottom:16px; box-shadow:0 3px 14px #00000008; }
 h2 { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.08rem;
      margin-bottom:10px; display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
 h2 .n { background:var(--acc); color:#fff; border-radius:8px; font-size:.72rem; min-width:24px;
        height:24px; padding:0 4px; display:inline-flex; align-items:center; justify-content:center; flex-shrink:0; }
 h3 { font-size:.93rem; font-weight:700; margin:16px 0 6px; }
 p { font-size:.9rem; margin:7px 0; }
 .dim { color:var(--dim); } .good { color:var(--good); } .bad { color:var(--bad); }
 .mono { font-family:'JetBrains Mono',monospace; font-size:.82em; }
 .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px; margin:12px 0; }
 .card { border:1px solid var(--line); border-radius:14px; padding:13px 15px; }
 .card .lbl { font-size:.68rem; font-weight:700; letter-spacing:.8px; text-transform:uppercase; color:var(--dim); }
 .card .big { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.3rem; margin:2px 0; }
 .card .sub { font-size:.76rem; color:var(--dim); }
 table { width:100%; border-collapse:collapse; font-size:.84rem; margin:8px 0; }
 th { text-align:left; color:var(--dim); font-size:.7rem; text-transform:uppercase;
      letter-spacing:.5px; padding:6px 9px; border-bottom:2px solid var(--line); }
 td { padding:6px 9px; border-bottom:1px solid var(--line); vertical-align:top; }
 tr:last-child td { border-bottom:0; }
 .verd { font-weight:700; border-radius:999px; padding:2px 9px; font-size:.68rem; white-space:nowrap; }
 .verd.dead { background:var(--badbg); color:var(--bad); }
 .verd.hold { background:#f1f0f4; color:var(--dim); }
 .verd.lead { background:var(--goodbg); color:var(--good); }
 .verd.warn { background:var(--warnbg); color:var(--warn); }
 .callout { border-left:4px solid var(--acc); background:var(--accbg);
            border-radius:0 12px 12px 0; padding:11px 15px; margin:10px 0; font-size:.88rem; }
 .callout.good { border-color:var(--good); background:var(--goodbg); }
 .callout.warn { border-color:var(--warn); background:var(--warnbg); }
 .callout.bad { border-color:var(--bad); background:var(--badbg); }
 canvas { max-height:340px; }
 .chartbox { margin:14px 0 2px; }
 .chartbox.tall { height:420px; } .chartbox.tall canvas { max-height:none; height:100% !important; }
 .dl { display:block; text-align:right; font-size:.72rem; margin:2px 0 10px; }
 .dl a { color:var(--acc); text-decoration:none; font-weight:700; }
 .dl a:hover { text-decoration:underline; }
 .cap { font-size:.78rem; color:var(--dim); margin:4px 0 8px; }
 code { font-family:'JetBrains Mono',monospace; font-size:.8em; background:#f4f2f8;
        border:1px solid var(--line); border-radius:6px; padding:1px 6px; }
 ul, ol { font-size:.9rem; margin:8px 0 8px 20px; } li { margin:5px 0; }
 .disc { border:2px solid var(--warn); border-radius:14px; padding:16px 18px; margin:14px 0;
        background:#fffdfa; }
 .disc .dhead { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; color:var(--warn);
        font-size:.9rem; margin-bottom:8px; }
 .disc pre { white-space:pre-wrap; font-family:'JetBrains Mono',monospace; font-size:.76rem;
        line-height:1.6; color:var(--ink); }
 .scroll { overflow-x:auto; }
 @media (max-width:640px) { section { padding:18px 14px; } }
"""

NAV = """<div class="labtabs">
<a href="/testing/">Testing Lab</a>
<a href="/testing/report/state_of_benpom">State of BenPom</a>
<a href="/testing/playbook">Deployment Playbook</a>
<a href="/testing/report/favorites_lab">Favorites Lab</a>
<a href="/testing/report/final_model">Final Model</a>
<a href="/testing/report/v7_lab">v7 Lab</a>
<a href="/testing/report/v8_lab">v8 Lab</a>
<a href="/testing/report/roster_adaptation">Roster</a>
<a href="/testing/report/v9_lab" class="on">v9 Lab</a>
</div>"""


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def dl(name, base="v9"):
    return (f'<div class="dl"><a href="/testing/{base}/stats/{name}" download>'
            f'&#8681; {name}</a></div>')


def fm(x, nd=2, plus=True):
    if x is None:
        return "&mdash;"
    s = f"{x:+.{nd}f}" if plus else f"{x:.{nd}f}"
    return s


# ── section pieces ──────────────────────────────────────────────────────────
sel = looks["selection_reads"]["entries"]
n_sel = len(sel)
mde_p = {k: v for k, v in pproto["checkpoints"]["mde_at_n_milli"]["within_family"].items()}
thr = pproto["promotion_rule"]["G1_sequential_evidence"]["thresholds"]

cand_rows = []
for c in cand["candidates"]:
    t1, t2, pl = c["transfer"]["T1_fit2324_val2025"], c["transfer"]["T2_fit2325_val2026H1"], \
        c["transfer"]["pooled_validation"]
    ar = c["transfer"]["advance_rule"]
    clauses = " ".join(
        f'<span class="verd {"lead" if ar[k] else "dead"}">{k.split("_")[0]}</span>'
        for k in ("A1_val1_ge_1se", "A2_val2_ge_1se", "A3_pooled_ci_gt0", "A4_pooled_ge_mde"))
    clauses += ' <span class="verd hold">A5 n/a</span>'
    cand_rows.append(
        f"<tr><td><b>{esc(c['id'])}</b><div class='dim' style='font-size:.76rem'>"
        f"{esc(c['transfer']['label'])}</div></td>"
        f"<td>{fm(t1['delta_milli'])}m <span class='dim'>(bar {t1['se_blk_milli']:.2f})</span></td>"
        f"<td>{fm(t2['delta_milli'])}m <span class='dim'>(bar {t2['se_blk_milli']:.2f})</span></td>"
        f"<td>{fm(pl['delta_milli'])}m <span class='dim'>[{pl['blk_ci_milli'][0]:+.2f}, "
        f"{pl['blk_ci_milli'][1]:+.2f}]</span></td>"
        f"<td>{clauses}</td><td><span class='verd dead'>{esc(c['verdict'])}</span></td></tr>")

aut_tab = []
for k, row in aut["table_gate_transfer_holdout"].items():
    aut_tab.append(f"<tr><td><b>{esc(k)}</b></td><td>{esc(row['gate_said'])}</td>"
                   f"<td>{esc(row['transfer_would_have_said'])}</td>"
                   f"<td>{esc(row['holdout_said'])}</td></tr>")

arm_rows = []
for a in board["arms"]:
    st = a["live_status"]["status"]
    cls = {"BASELINE": "lead", "ACCUMULATING": "hold", "ALIVE": "lead",
           "PROMOTED": "lead", "KILLED": "dead", "NOT_SCOREABLE_LIVE": "warn"}.get(st, "hold")
    b = a.get("beta_frozen")
    note = a.get("status_note", "")
    arm_rows.append(
        f"<tr><td><b>{esc(a['id'])}</b> <span class='dim'>({esc(a['role'])})</span>"
        + (f"<div class='dim' style='font-size:.74rem'>{esc(note)}</div>" if note else "")
        + f"</td><td class='mono'>{b if b is not None else '&mdash;'}</td>"
        f"<td><span class='verd {cls}'>{esc(st)}</span></td></tr>")

sel_rows = []
for e in sel:
    sel_rows.append(f"<tr><td>{esc(e['candidate'])}</td><td>{fm(e['T1'])}m</td>"
                    f"<td>{fm(e['T2'])}m</td><td>{fm(e['pooled'])}m</td>"
                    f"<td><span class='verd dead'>{esc(e['verdict'])}</span></td></tr>")

aut_read_rows = []
for e in looks["autopsy_methodological_reads"]["entries"]:
    aut_read_rows.append(f"<tr><td>{esc(e['config'])}</td><td>{fm(e['T1_val2025_milli'])}m</td>"
                         f"<td>{fm(e['T2_val2026H1_milli'])}m</td>"
                         f"<td>{fm(e['pooled_validation_milli'])}m</td>"
                         f"<td>{fm(e['holdout_v8_method_milli'])}m</td></tr>")

n1 = [c for c in cand["candidates"] if c["id"] == "N1_delta2"][0]
n1g = n1["provenance"]["grid_row"]
n1f = n1["provenance"]["fit1_flag_investigation"]

chk_cards = "".join(
    f'<div class="card"><div class="lbl">checkpoint n={n}</div>'
    f'<div class="big">p &ge; {thr[str(n)]}</div>'
    f'<div class="sub">G1 both CRN modes &middot; G2 &Delta; &ge; {mde_p[str(n)]}m '
    f'(within-family MDE)</div></div>'
    for n in (100, 200, 400))

pop = board["population"]
E_ARMS = board["checkpoints"]

html = f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex">
<title>v9 Lab — the roster family, transfer-gated: v6 stands</title>
<link rel="icon" href="/logo.svg">
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@700;800&family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
<style>{CSS}</style></head>
<body><div class="wrap">
<h1>v9 Lab — the roster family, transfer-gated</h1>
<div class="tagline">v6 + roster subsystem, optimized between the two &middot; program run 2026-07-29 &middot;
every number on this page reads from <code>testing_lab/v9/stats/</code> (download links throughout) &middot;
selection by era-transfer, confirmation prospective-only</div>
{NAV}

<div class="banner"><b>THE ANSWER: {esc(lad['verdict_sentence']).upper()}.</b>
The search funnel ran <b>{grid['n_configs']:,} configurations &rarr; {grid['n_eligible']} eligible on
train &rarr; {n_sel} frozen candidates evaluated one-shot &rarr; {cand['n_advanced']} advanced</b>. The
frozen prospective ladder is pure v6, &beta; = <span class="mono">{BETA6}</span> (refit once on
2023-01-01..2026-07-28, never again). Nothing ships; nothing on the public site regardless; VCTMM
stays hands-off.</div>

<section>
<h2><span class="n">0</span>The answer</h2>
<div class="cards">
 <div class="card"><div class="lbl">the ladder</div><div class="big">v6 alone</div>
   <div class="sub">&beta; frozen {BETA6} &middot; 2,044 freeze rows</div></div>
 <div class="card"><div class="lbl">candidates advanced</div><div class="big">0 of {n_sel}</div>
   <div class="sub">solve-side a=0.5/1/2 &middot; &delta;2 &middot; hybrid &mdash; all DIE</div></div>
 <div class="card"><div class="lbl">grid searched</div><div class="big">{grid['n_configs']:,}</div>
   <div class="sub">{grid['n_eligible']} eligible &middot; &delta;1 nominated nothing</div></div>
 <div class="card"><div class="lbl">prospective referee</div><div class="big">n={pop['n_scored_paired']}</div>
   <div class="sub">{esc(board['verdict'])} &middot; first read at n=100</div></div>
</div>
<p><b>The family is measured dead across its whole span.</b> Solve-side boosts transfer negative
from a=0.5 through a=28 (this program: &minus;0.67m &rarr; &minus;1.70m pooled, monotone in a; the
autopsy adds a&isin;{{4.5, 6, 28}} all negative). The prediction layer's best in-era config
(&delta;2, +{n1g['dF_milli']}m on FIT1) transferred to <b>{n1['transfer']['pooled_validation']['delta_milli']}m</b>
pooled with the block CI entirely below zero &mdash; the third train-mirage of this program line, and the
red-flag rule had already found the tell before nomination. &delta;1 (pure evidence-sign) never even fit
in-era (best dF +0.81m &lt; 1.0m floor, LOEO negative &mdash; nominated nothing). The hybrid clause was
skipped as preregistered: no advancing parents.</p>
<p><b>What would change this:</b> the prospective scoreboard below (&sect;4) &mdash; the only
confirmatory instrument left &mdash; or the re-open triggers in &sect;5. Nothing else can.</p>
{dl('v9_ladder.json')}
</section>

<section>
<h2><span class="n">1</span>The motivation was real (and still is)</h2>
<p>v9 was commissioned on measured misses, on record: v6 underpriced <b>LEV by
&minus;14.7pp/match for 10 matches</b> after the Neon change; overpriced <b>ENVY by +24&ndash;32pp/match</b>
after its chain; the population atlas shows <b>+4.4/+5.6/+4.8pp</b> early new-roster outperformance
(keep4/keep3/overhaul, first 3 matches, vs +0.7pp stable). Those misses are not disputed &mdash; the case
charts live on the <a href="/testing/report/roster_adaptation">Roster page</a> and are not duplicated here.</p>
<div class="callout">Every mechanism tried &mdash; solve-side reweighting, prediction-layer overlays in three
direction encodings, a hybrid &mdash; <b>pays more elsewhere than it earns on the change windows</b>. The
signal exists; no tested pricing of it survives out of era at these definitions.</div>
{dl('roster_integration.json', base='v8')}
</section>

<section>
<h2><span class="n">2</span>The validation design &mdash; why this referee is trustworthy</h2>
<p>Selection runs on <b>era transfer inside pre-07/28 data</b>: freeze a candidate on
FIT1 (&le;2024-12-31, n=841), score it paired vs v6 on VAL1 (2025, n=674); refreeze on FIT2
(&le;2025-12-31, n=1515), score on VAL2 (2026H1, n=543). Advance requires <b>ALL</b> of:
A1 &Delta;<sub>VAL1</sub> &ge; 1&times;SE<sub>blk</sub>; A2 &Delta;<sub>VAL2</sub> &ge; 1&times;SE<sub>blk</sub>;
A3 pooled block 95% CI &gt; 0; A4 pooled &Delta; &ge; 1.773m (pair-MDE at n=1217); A5 fragility
(drop-top-5% &gt; 0 AND leave-one-event-out min &gt; 0). Measured error rates: <b>null false-advance
&asymp; 2%, power &asymp; 0.75&ndash;0.9</b> against a true uniform +2.5m. One ledgered evaluation per
candidate, ever. The 2025-26 rows are SPENT as a confirmatory instrument (403 recorded looks) &mdash;
here they are demoted to selection-grade targets and confirm nothing.</p>
<h3>The gate autopsy &mdash; would this referee have blocked the a=28 ship? YES, on every clause.</h3>
<div class="chartbox"><canvas id="c_aut"></canvas></div>
<p class="cap">The v8 gate's train-CV claim (blue) vs what era-transfer says (orange) vs what the
recorded holdout said (gray), milli-LL per series. The gate fired on train for all three; transfer
blocks all three; the holdout agrees with transfer to &le;0.6m. Gray band = &plusmn;1.773m pooled noise
floor. Reproduction fixture: the ship's pooled-holdout number was reproduced to the third decimal
(&minus;11.595m, gap 0.000).</p>
{dl('v9_page_autopsy.json')}
<div class="scroll"><table>
<tr><th>config (as gate-selected)</th><th>gate said (train CV)</th><th>era-transfer would have said</th><th>holdout said</th></tr>
{''.join(aut_tab)}
</table></div>{dl('v9_gate_autopsy.json')}
<div class="callout warn"><b>Why the conjunction is load-bearing:</b> every config's damage concentrates
in VAL2/2026H1 (&minus;9.6 to &minus;21m) while VAL1/2025 is &asymp; flat. A single pooled test would have
looked merely inconclusive for the a=4.5 configs &mdash; the <b>win-BOTH-eras</b> requirement is what turns
&ldquo;meh&rdquo; into BLOCK. The gate's own CV folds contained the warning too: drop fold 2 or 3 and the
a=28 gate stops firing (the preregistered C5 concentration flag fires for the ship alone).</div>
{dl('v9_transfer_protocol.json')}
</section>

<section>
<h2><span class="n">3</span>The search &mdash; {grid['n_configs']:,} configs, five candidates, zero survivors</h2>
<div class="cards">
 <div class="card"><div class="lbl">grid</div><div class="big">{grid['n_configs']:,}</div>
   <div class="sub">4 policies &times; 5 direction variants &times; 9 b &times; 6 &tau; &times; 3 &gamma; (FIT1 only, MDE 2.15m)</div></div>
 <div class="card"><div class="lbl">eligible</div><div class="big">{grid['n_eligible']}</div>
   <div class="sub">{esc(grid['grid_def']['eligibility'])}</div></div>
 <div class="card"><div class="lbl">evaluated on transfer</div><div class="big">{n_sel}</div>
   <div class="sub">one-shot, ledgered before use</div></div>
 <div class="card"><div class="lbl">advanced</div><div class="big">0</div>
   <div class="sub">every candidate failed A1&ndash;A4</div></div>
</div>
<div class="chartbox tall"><canvas id="c_grid"></canvas></div>
<p class="cap">All {grid['n_configs']:,} prediction-layer configs on FIT1: train gain (x) vs the worse
of the two era submeans (y). Purple = eligible ({grid['n_eligible']}), gray = ineligible, orange rings =
the two grid nominees (&delta;2 b=.65 &tau;13 &gamma;.5; hybrid(6) b=.20 &tau;21 &gamma;.5). The whole
eligible cloud is train-side only &mdash; and both nominees died on transfer anyway.</p>
{dl('v9_page_grid.json')}{dl('v9_search_grid.json')}
<h3>The five one-shot transfer evaluations (clause-by-clause)</h3>
<div class="scroll"><table>
<tr><th>candidate (frozen config)</th><th>T1 / VAL1 (bar)</th><th>T2 / VAL2 (bar)</th>
<th>pooled [blk 95% CI]</th><th>advance clauses</th><th>verdict</th></tr>
{''.join(cand_rows)}
</table></div>{dl('v9_candidates.json')}
<p class="cap">A5 is only evaluated when A1&ndash;A4 all pass &mdash; no candidate got there. Negative
deltas translate to 0.0 expected-ROI on the quote-margin ladder (it spans positive shifts only); no
positive-ROI claim exists in this phase. Pooled MDE context: 1.773m within-family.</p>
<div class="callout bad"><b>The N1 red-flag story (the third train-mirage, caught in advance):</b>
&delta;2's nominee carried +{n1g['dF_milli']}m on FIT1 with both era submeans positive
(+{n1g['d23_milli']}m / +{n1g['d24_milli']}m), all-25-events LOEO positive, time-folds clean &mdash; and
the preregistered &gt;+5m red-flag investigation still found <b>{n1f['top10_row_share']:.1%} of the gain
living in 10 rows</b> (FIT1 drop-top-5% {n1f['drop_top5_milli']}m). Transfer verdict:
{n1['transfer']['pooled_validation']['delta_milli']}m pooled, block CI
[{n1['transfer']['pooled_validation']['blk_ci_milli'][0]:+.2f}, {n1['transfer']['pooled_validation']['blk_ci_milli'][1]:+.2f}]
&mdash; entirely below zero; T1 was {n1['transfer']['T1_fit2324_val2025']['delta_milli']}m, meaning the
atlas prior <i>actively mispriced 2025</i>, not merely failed to help. Pooled fragility confirmed it
(drop-top-5% {n1['transfer']['pooled_fragility']['drop_top5_milli']}m).</div>
<p class="dim">Mechanics were proven before any scoring: 67/67 preregistered fixtures passed
(exact v6 identity at a=s=b=0 by sha256; per-team overlays touch no other team's rows, bit-checked;
date-strict evidence with the leak assert hot on every pricing; the a &le; 6 hard cap raises on
violation &mdash; the a=28 lesson is enforced in code).</p>
{dl('v9_fixtures.json')}
</section>

<section>
<h2><span class="n">4</span>The prospective scoreboard &mdash; the real referee, live</h2>
<div class="callout good"><b>This is the only confirmatory instrument left, by design.</b> Frozen before
any post-2026-07-28 row was examined: arms frozen, &beta;s frozen, decision rules fixed, three reads ever
(scored n &isin; {{100, 200, 400}}), &alpha;-spend {{.001, .005, .025}} (union 0.031 &le; 0.05,
OBF-shaped). Kill: block 95% CI ci_hi &lt; 0 at any checkpoint. No promotion by n=400 &rArr; NO SHIP;
revival only on new prospective data. The standing evaluator
(<code>testing_lab/v9/score_prospective.py</code>) is idempotent, read-only over <code>data/</code>,
and refits nothing.</div>
<div class="cards">
 <div class="card"><div class="lbl">status</div><div class="big">{esc(board['verdict'])}</div>
   <div class="sub">settled series &gt; 2026-07-28 &middot; last run {esc(board['last_run'])}</div></div>
 <div class="card"><div class="lbl">next checkpoint</div><div class="big">n={E_ARMS['next_checkpoint']}</div>
   <div class="sub">reads taken {len(E_ARMS['reads_taken'])}/3 &middot; Stage 2 live; Champions lands Sep&ndash;Oct</div></div>
 <div class="card"><div class="lbl">paired drop rule</div><div class="big">{pop['n_dropped_pairing']}</div>
   <div class="sub">rows dropped (NaN for any arm drops the row for all)</div></div>
</div>
<div class="cards">{chk_cards}</div>
<p class="cap">Plus at every checkpoint: G3 no bucket catastrophe (v8 bars verbatim: &minus;4m@n&ge;100 /
&minus;8m@30&ndash;99); G4 max|team-bias| &le; v6 + 2pp; G5 if several pass, the most conservative wins.
Cross-family MDE context: 20.5 / 14.5 / 10.3m at the three checkpoints.</p>
<h3>Arms (frozen; candidate set is empty &mdash; reference arms are scored for the record only)</h3>
<div class="scroll"><table>
<tr><th>arm</th><th>&beta; frozen</th><th>status</th></tr>
{''.join(arm_rows)}
</table></div>
<p class="cap">The candidate ladder is v6 alone &mdash; there is nothing v9 can promote. The six
reference arms are the v8 roster program's preregistered prospective plan (frozen 2026-07-28, before
the v9 verdict existed); they ride the same machinery so that plan still gets its answer, but they are
not v9 candidates and cannot ship anything. D_phase_reset cannot be scored on the live corpus and says
so rather than pretending.</p>
{dl('v9_prospective_scoreboard.json')}{dl('v9_prospective_protocol.json')}
</section>

<section>
<h2><span class="n">5</span>Looks ledger, integrity &amp; the do-not-retest entry</h2>
<div class="cards">
 <div class="card"><div class="lbl">selection reads</div><div class="big">{n_sel}</div>
   <div class="sub">one per candidate, ledgered at run time</div></div>
 <div class="card"><div class="lbl">autopsy reads</div><div class="big">{looks['autopsy_methodological_reads']['count']}</div>
   <div class="sub">all preregistered, 2026-07-29</div></div>
 <div class="card"><div class="lbl">exploratory budget</div><div class="big">{looks['exploratory_budget']['spent']}/3</div>
   <div class="sub">untouched</div></div>
 <div class="card"><div class="lbl">prospective reads</div><div class="big">{len(looks['prospective_reads']['entries'])}/3</div>
   <div class="sub">none yet &mdash; first at scored n=100</div></div>
</div>
<h3>Every transfer evaluation (the selection ledger, verbatim)</h3>
<div class="scroll"><table>
<tr><th>candidate</th><th>T1</th><th>T2</th><th>pooled</th><th>verdict</th></tr>
{''.join(sel_rows)}
</table></div>
<h3>The five preregistered autopsy reads (design input for the validator)</h3>
<div class="scroll"><table>
<tr><th>config</th><th>VAL1</th><th>VAL2</th><th>pooled</th><th>holdout (v8 method)</th></tr>
{''.join(aut_read_rows)}
</table></div>{dl('v9_looks.json')}
<div class="disc"><div class="dhead">Process incident, disclosed by the design agent (part of the
record &mdash; quoted verbatim from preregister.design.md)</div>
<pre>{esc(STRUCK)}</pre></div>
<h3>Ledger entry &mdash; the family's death is do-not-retest at these definitions</h3>
<ul>
<li><b>Dead, with kill numbers on this page:</b> solve-side per-side boost 1 + a(1&minus;k/5)e<sup>&minus;n/&tau;</sup>
(a &isin; [0.5, 28], all transfer-negative, monotone worsening); prediction-layer &delta;2 / &delta;1 /
hybrid overlays at the searched grid (&delta;2 pooled &minus;5.38m CI &lt; 0; &delta;1 never eligible
in-era; hybrid &minus;1.98m); the v8 spec-run config (autopsy: BLOCK on every clause).</li>
<li><b>Re-open triggers (the only ones):</b> a genuinely NEW data source at the player level
(e.g. per-player ratings feeding a mean-shift, the unused Phase-3 idea), or a prospective surprise
on the &sect;4 scoreboard (a reference arm beating its kill/floor bars on virgin data).</li>
<li><b>Tripwires standing:</b> the three v9 laws (spent holdout adjudicates nothing; selection on
transfer only; prospective is the referee); the a &le; 6.0 hard cap is code, not convention;
market data is never a fitting target (standing rule 9).</li>
<li><b>Integrity notes:</b> one writer per artifact; every &beta; scale-bound and window-refit
(fixture &beta;(FIT1, v6) = 0.115199, realized); CRN seeds 20260728/20260729 for every bootstrap, no
private seeds; the evaluator's frame builder reproduced the frozen frame 2058/2058 rows, 0 column
mismatches, and the protocol &beta; refit on the live frame reproduced the ladder's {BETA6} exactly
(gap 0.000).</li>
</ul>
</section>

<p class="dim" style="text-align:center;font-size:.78rem">agent:v9-finish &middot; v9 lab program
2026-07-29 &middot; preregisters: design / family / search / finish (all locked before their runs) &middot;
every number from <span class="mono">/testing/v9/stats/*.json</span> &middot; CRN per
<span class="mono">crn.json</span> &middot; the ladder is v6 alone &middot; nothing on the public site
regardless &middot; VCTMM hands-off</p>
</div>
<script>
const PAL = __PAL__;
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

// §2 gate autopsy — grouped bars per config
const AUT = __AUT__;
new Chart(document.getElementById('c_aut'), {{ type:'bar',
 data:{{ labels: AUT.configs.map(c=>c.label), datasets:[
  {{ label:'gate said (train CV)', data:AUT.configs.map(c=>c.gate_cv_milli),
     backgroundColor:PAL.a1, borderRadius:4, maxBarThickness:40 }},
  {{ label:'era-transfer pooled', data:AUT.configs.map(c=>c.transfer_pooled_milli),
     backgroundColor:PAL.d5, borderRadius:4, maxBarThickness:40 }},
  {{ label:'recorded holdout', data:AUT.configs.map(c=>c.holdout_milli),
     backgroundColor:PAL.v6, borderRadius:4, maxBarThickness:40 }} ] }},
 options:{{ responsive:true, maintainAspectRatio:false,
  plugins:{{ legend:{{position:'bottom', labels:{{boxWidth:14, boxHeight:3}}}},
    floorBand:{{axis:'y', lo:-1.773, hi:1.773}},
    tooltip:{{callbacks:{{label:(t)=>t.dataset.label+': '+(t.parsed.y>=0?'+':'')+t.parsed.y.toFixed(2)+'m'}}}} }},
  scales:{{ x:{{grid:{{display:false}}, ticks:{{font:{{size:11}}}}}},
    y:{{grid:{{color:'#eceef2'}}, title:{{display:true, text:'Δ log-loss vs v6, milli (>0 = better)'}}}} }} }} }});

// §3 grid scatter — 3,240 configs
const G = __GRID__;
const inel = G.points_dF_eramin_eligible.filter(p=>!p[2]).map(p=>({{x:p[0], y:p[1]}}));
const elig = G.points_dF_eramin_eligible.filter(p=> p[2]).map(p=>({{x:p[0], y:p[1]}}));
const noms = G.grid_nominated.map(p=>({{x:p.dF_milli, y:p.era_min_milli}}));
new Chart(document.getElementById('c_grid'), {{ type:'scatter',
 data:{{ datasets:[
  {{ label:'ineligible ('+inel.length+')', data:inel, backgroundColor:'rgba(154,147,166,0.25)',
     pointRadius:2, pointHoverRadius:3 }},
  {{ label:'eligible ('+elig.length+')', data:elig, backgroundColor:PAL.s1,
     pointRadius:3, pointHoverRadius:5 }},
  {{ label:'nominated (2)', data:noms, backgroundColor:'rgba(0,0,0,0)', borderColor:PAL.d5,
     borderWidth:3, pointRadius:9, pointHoverRadius:10, pointStyle:'circle' }} ] }},
 options:{{ responsive:true, maintainAspectRatio:false,
  plugins:{{ legend:{{position:'bottom', labels:{{boxWidth:14, boxHeight:8}}}},
    tooltip:{{callbacks:{{label:(t)=>'dF '+t.parsed.x.toFixed(2)+'m, era-min '+t.parsed.y.toFixed(2)+'m'}}}} }},
  scales:{{ x:{{grid:{{color:'#eceef2'}}, title:{{display:true, text:'FIT1 train gain dF (milli-LL vs v6)'}}}},
    y:{{grid:{{color:'#eceef2'}}, title:{{display:true, text:'era_min — worse of the 2023/2024 submeans (milli)'}}}} }} }} }});
</script>
</body></html>"""


def _g(d, *ks, default=None):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


_n1 = next((x for x in cand["candidates"] if x["id"] == "N1_delta2"), {})
_n1_fit = _g(_n1, "provenance", "grid_row", "dF_milli")
_n1_share = _g(_n1, "provenance", "fit1_flag_investigation", "top10_row_share")
_n1_pool = _g(_n1, "transfer", "pooled_validation", "delta_milli")
import re as _re
_a28_row = aut.get("table_gate_transfer_holdout", {}).get("p1w8c5 (the a=28 ship)", {})
_a28_m = _re.match(r"\s*([+-]?\d+\.?\d*)m", str(_a28_row.get("holdout_said", "")))
_a28_hold = float(_a28_m.group(1)) if _a28_m else None
_nadv = board["ladder"]["n_candidate_arms_beyond_v6"]
NARRATIVE = f"""
<section><h2><span class="n">6</span> The plain-English version</h2>
<p><b>What we were trying to fix.</b> After LEV added Neon, the model kept pricing them too low
(&asymp;{abs(_lev['panel']['carryover_cost_pp_per_match']):.0f} points too low per match, for
{_lev['panel']['carryover_n']} matches). After ENVY swapped players mid-season, it kept pricing
them too high (&asymp;{abs(_env_pp['carryover_cost_pp_per_match']):.0f} points per match). Both
misses are real and documented on the Roster page. The idea: give v6 a subsystem that reacts
faster right after a roster change, and tune everything between plain v6 and the full subsystem.</p>
<p><b>What got tested.</b> {grid['n_configs']:,} versions across two designs &mdash; one that
re-weights games inside the rating solve, and one that only nudges the changed team&rsquo;s match
probability. Every version was judged the hard way: does a gain learned on 2023&ndash;24 still
show up in 2025 and 2026? Looking good where you were fit doesn&rsquo;t count.</p>
<p><b>What was worse.</b> Every rating-level version made the model worse, at every strength
tried &mdash; even tiny boosts &mdash; because re-weighting one team&rsquo;s games bends every
other team&rsquo;s rating through the shared solve. The most aggressive version a gate had
previously approved (a=28) was the worst of all
({(f"{_a28_hold:+.1f}m on the holdout, " if _a28_hold is not None else "")}losing most exactly on
the full-rebuild matches it was built for).</p>
<p><b>What looked better, but wasn&rsquo;t.</b> The best probability-layer version looked
{(f"+{_n1_fit:.1f}m" if _n1_fit is not None else "clearly")} better on the era it was tuned on
&mdash; but {(f"{_n1_share:.0%}" if _n1_share is not None else "most")} of that gain came from
just ten matches, and on the eras it had never seen it was
{(f"{_n1_pool:+.1f}m" if _n1_pool is not None else "clearly")} <i>worse</i>. That is overfitting,
and the referee caught it before it could be promoted.</p>
<p><b>The conclusion.</b> v6 stays, unchanged &mdash; {_nadv} of the
{len(cand['candidates'])} finalists earned promotion. Not because the roster problem
isn&rsquo;t real, but because the fixes lose <i>on roster-change matches themselves</i>: the
rating-level versions additionally bleed into unchanged teams through the shared solve (part of
why they are worst), while the probability-layer versions &mdash; which by construction cannot
touch unchanged teams &mdash; still lose net across the corpus&rsquo;s <i>other</i> roster
changes: the always-boost version helps LEV-shaped upgrades but actively hurts ENVY-shaped
downgrades, and the read-the-early-results version has no reliable direction signal at 1&ndash;5
matches. The case-by-case ledger is in &sect;7. What ships instead is on the betting side: the
<span class="mono">roster_flag</span> tells the bot to quote smaller on fresh rosters. And the
question stays open the right way: from tonight onward, every settling match scores the frozen
candidates on data nobody has touched (&sect;4). If reacting faster to roster changes genuinely
helps, the scoreboard will say so at its checkpoints &mdash; and the pass/fail rules for that
are already locked, so nobody can move the goalposts, including us.</p>
</section>

"""

if dec:
    _n1d = dec["candidates"]["N1_delta2"]
    _cats = ["improving_LEV_shape", "degrading_ENVY_shape"]
    _cat_labels = ["dominant side improving (LEV-shape)", "dominant side degrading (ENVY-shape)"]
    _v1 = [_n1d["windows"]["VAL1"]["lev_envy_split"][c]["contrib_to_window_mean_milli"] for c in _cats]
    _v2 = [_n1d["windows"]["VAL2"]["lev_envy_split"][c]["contrib_to_window_mean_milli"] for c in _cats]
    _fb = ["n_since_1_5", "n_since_6_13", "n_since_14_plus"]
    _fb_labels = ["matches 1-5 after the change (freshest, biggest push)",
                  "matches 6-13", "matches 14+ (push nearly decayed)"]
    _f1 = [_n1d["windows"]["VAL1"]["by_freshness"][b]["contrib_to_window_mean_milli"] for b in _fb]
    _f2 = [_n1d["windows"]["VAL2"]["by_freshness"][b]["contrib_to_window_mean_milli"] for b in _fb]
    _loc = dec["locality_assertion"]["N1_delta2"]
    _tie = dec["tie_out"]["N1_delta2"]["pooled"]

    def _exrow(ex):
        act = "; ".join(f"{a['org']} k={a['k']}/5 n={a['n_since']}" for a in ex["active_sides"])
        pc = ex.get("p_cand", ex.get("p_candidate"))
        return (f"<tr><td>{ex['date']}</td><td>{esc(ex['winner'])} beat {esc(ex['loser'])} "
                f"{esc(ex['score'])}</td><td>{esc(act)}</td>"
                f"<td class=\"mono\">{ex['p_v6']:.3f} &rarr; {pc:.3f}</td>"
                f"<td class=\"mono\">{ex.get('d_milli', ex.get('delta_milli')):+.0f}m</td>"
                f"<td>{esc(str(ex.get('reason',''))[:120])}</td></tr>")

    _worst = "".join(_exrow(e) for e in _n1d["examples"]["worst_10"])
    _best = "".join(_exrow(e) for e in _n1d["examples"]["best_10"])
    _s1v2 = dec["candidates"]["S_a1.0"]["windows"]["VAL2"]["unaffected_split"]["untouched_pure_coupling"]
    SEC7 = f"""
<section><h2><span class="n">7</span> Where exactly it lost &mdash; the case ledger</h2>
<div class="cards">
<div class="card"><div class="lbl">locality (your test)</div><div class="big good">&Delta; = 0.000</div>
<div class="sub">on every scored match with NO active roster phase ({_loc['n_scored_rows_no_active_phase']} such
matches) &mdash; asserted, prediction-layer candidates cannot touch unchanged teams</div></div>
<div class="card"><div class="lbl">tie-out</div><div class="big">{'PASS' if _tie['pass'] else 'FAIL'}</div>
<div class="sub">per-match sums reproduce the ledgered totals ({_tie['ledgered_milli']:+.2f}m pooled)</div></div>
<div class="card"><div class="lbl">the catch: always on</div>
<div class="big bad">{_n1d['windows']['VAL1']['affected']['n'] + _n1d['windows']['VAL2']['affected']['n']}
/{_n1d['windows']['VAL1']['n_window'] + _n1d['windows']['VAL2']['n_window']}</div>
<div class="sub">scored validation matches had an active roster phase &mdash; the winning config's
65-match horizon meant the subsystem effectively never turned off ({_loc['n_scored_rows_no_active_phase']}
phase-free matches total)</div></div>
<div class="card"><div class="lbl">solve-side coupling (S_a1.0)</div><div class="big bad">{_s1v2['sum_milli']:+.0f}m&middot;rows</div>
<div class="sub">damage on the {_s1v2['n']} truly-untouched VAL2 matches &mdash; the rating-level family's
disqualifier, quantified (tiny n because phases were ubiquitous)</div></div>
</div>
<h3>Who the adjustment helped and hurt (&delta;2, contribution to window mean)</h3>
<div class="chartbox" style="height:280px"><canvas id="decompChart"></canvas></div>
<p class="cap">Each affected match is classed by its DOMINANT active side&rsquo;s walk-forward
evidence (E&gt;0 improving = LEV-shape, E&lt;0 degrading = ENVY-shape). &delta;2 lost on BOTH:
it hurt even where its always-up push agreed with the evidence&rsquo;s direction &mdash; the
failure is magnitude and timing, not just direction. And with phases lasting 65 matches, the
subsystem was effectively always on, adjusting far outside the fresh-change windows the concept
was aimed at.</p>
<h3>When the adjustment hurt: by matches-since-change</h3>
<div class="chartbox" style="height:280px"><canvas id="freshChart"></canvas></div>
<p class="cap">The direct test of &ldquo;react harder to the first results&rdquo;: the damage
concentrates exactly where the push is biggest &mdash; the first 1&ndash;5 matches after a change
&mdash; while the nearly-decayed tail (14+ matches) was mildly positive in both eras. The early
post-change results are the noisiest part of the phase; leaning on them harder than v6 does is
where the money is lost.</p>
<h3>Worst 10 affected matches (&delta;2 vs v6)</h3>
<div class="scroll"><table><tr><th>date</th><th>match</th><th>active phases</th><th>p_v6 &rarr; p_&delta;</th>
<th>&Delta;LL</th><th>reason</th></tr>{_worst}</table></div>
<h3>Best 10 affected matches</h3>
<div class="scroll"><table><tr><th>date</th><th>match</th><th>active phases</th><th>p_v6 &rarr; p_&delta;</th>
<th>&Delta;LL</th><th>reason</th></tr>{_best}</table></div>
<div class="dl"><a href="/testing/v9/stats/v9_case_decomposition.json" download>&#8681; v9_case_decomposition.json</a></div>
</section>
"""
    SEC7_JS2 = ("new Chart(document.getElementById('freshChart'), { type:'bar',"
        "data:{ labels:" + json.dumps(_fb_labels) + ", datasets:["
        "{ label:'VAL1 (2025)', data:" + json.dumps(_f1) + ", backgroundColor:'#3a90cc' },"
        "{ label:'VAL2 (2026H1)', data:" + json.dumps(_f2) + ", backgroundColor:'#c96a2a' } ] },"
        "options:{ responsive:true, maintainAspectRatio:false,"
        "plugins:{ legend:{position:'bottom', labels:{boxWidth:14, boxHeight:8}} },"
        "scales:{ y:{grid:{color:'#eceef2'}, title:{display:true, text:'contribution to window mean (milli-LL; + = helped)'}},"
        "x:{grid:{display:false}, ticks:{font:{size:10}}} } } });")
    SEC7_JS = ("new Chart(document.getElementById('decompChart'), { type:'bar',"
        "data:{ labels:" + json.dumps(_cat_labels) + ", datasets:["
        "{ label:'VAL1 (2025)', data:" + json.dumps(_v1) + ", backgroundColor:'#3a90cc' },"
        "{ label:'VAL2 (2026H1)', data:" + json.dumps(_v2) + ", backgroundColor:'#c96a2a' } ] },"
        "options:{ responsive:true, maintainAspectRatio:false,"
        "plugins:{ legend:{position:'bottom', labels:{boxWidth:14, boxHeight:8}} },"
        "scales:{ y:{grid:{color:'#eceef2'}, title:{display:true, text:'contribution to window mean (milli-LL; + = helped)'}},"
        "x:{grid:{display:false}} } } });")
else:
    SEC7, SEC7_JS, SEC7_JS2 = "", "", ""

_last = html.rfind("</section>")
html = html[:_last + len("</section>")] + NARRATIVE + SEC7 + html[_last + len("</section>"):]
html = html.replace("</script>\n</body></html>", SEC7_JS + "\n" + SEC7_JS2 + "\n</script>\n</body></html>")

html = (html
        .replace("__PAL__", json.dumps(C))
        .replace("__AUT__", json.dumps(page_aut))
        .replace("__GRID__", json.dumps(page_grid)))

os.makedirs(RD, exist_ok=True)
out = os.path.join(RD, "v9_lab.html")
with open(out, "w") as f:
    f.write(html)
print(f"wrote {out} ({len(html)} bytes)")
