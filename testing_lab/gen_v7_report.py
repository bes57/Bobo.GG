"""Generate reports/v7_lab.html — the v7 research program: recency bias &
decay symmetry, run 2026-07-23/24 at the operator's request."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
RD = os.path.join(OUT, "reports")


def j(name):
    with open(os.path.join(OUT, name)) as f:
        return json.load(f)


st1 = j("v7_stage1.json")
st2 = j("v7_stage2.json")
st3 = j("v7_stage3.json")
res1, boots = st1["results"], st1["boots"]

NAV = """<div class="labtabs">
<a href="/testing/">Testing Lab</a>
<a href="/testing/report/state_of_benpom">State of BenPom</a>
<a href="/testing/playbook">Deployment Playbook</a>
<a href="/testing/report/favorites_lab">Favorites Lab</a>
<a href="/testing/report/final_model">Final Model</a>
<a href="/testing/report/v7_lab" class="on">v7 Lab</a><span class="brk"></span>
<a href="/testing/report/v8_lab">v8 Lab</a>
<a href="/testing/report/roster_adaptation">Roster</a>
<a href="/testing/report/v9_lab">v9 Lab</a>
<a href="/testing/report/v10_lab">v10 Lab</a>
<a href="/testing/report/edge_lab">Edge vs Market</a>
<a href="/testing/report/playbook_bt">Playbook (backtested)</a>
</div>"""

ORDER = ["v6_consist_20_12", "v5_asym_W20L12", "consist_16_10", "consist_14_8",
         "consist_12_8", "sym_24", "sym_20", "sym_16", "sym_14", "sym_12",
         "sym_10", "sym_8", "sym_6", "surprise_16_24", "surprise_12_20",
         "boxexp_c5_hl10", "boxexp_c3_hl8", "power_t6_a15"]
LABELS = {
    "v6_consist_20_12": "v6 — consistency (20/12) · CHAMPION",
    "v5_asym_W20L12": "v5 — asym wins 20 / losses 12",
    "consist_16_10": "consistency (16/10) — more recency",
    "consist_14_8": "consistency (14/8)",
    "consist_12_8": "consistency (12/8)",
    "surprise_12_20": "REVERSED: surprises persist (12/20)",
    "surprise_16_24": "REVERSED: surprises persist (16/24)",
    "boxexp_c3_hl8": "flat 3 games, then HL 8",
    "boxexp_c5_hl10": "flat 5 games, then HL 10",
    "power_t6_a15": "power-law tail (τ6, α1.5)",
}
grid_rows = ""
for name in ORDER:
    r = res1[name]
    lbl = LABELS.get(name, f"symmetric HL {name.split('_')[1]}")
    if name in boots:
        bt = boots[name]
        d = bt["mean_delta"] * 1000
        boot_s = f"{d:+.2f}m · p={bt['p_better']:.2f}"
        cls = "bad" if d < -3 else ("dim" if d < 0 else "good")
    else:
        boot_s, cls = "—", "good"
    hi = " style='background:var(--accbg)'" if name == "v6_consist_20_12" else ""
    grid_rows += (f"<tr{hi}><td>{lbl}</td>"
                  f"<td class='mono'>{r['ll_test']:.5f}</td>"
                  f"<td class='mono'>{r['ll_2026']:.5f}</td>"
                  f"<td class='mono'>{r['ll_formshift']:.5f}</td>"
                  f"<td class='mono'>{r['formshift_pred_mover']:.3f}</td>"
                  f"<td class='mono {cls}'>{boot_s}</td></tr>")

form_rows = ""
for name, r in st2.items():
    bt = r["boot_vs_v6"]
    form_rows += (f"<tr><td class='mono'>{name}</td>"
                  f"<td class='mono'><b>{r['b_form']:+.3f}</b></td>"
                  f"<td class='mono'>{r['ll_test']:.5f}</td>"
                  f"<td class='mono'>{r['ll_formshift']:.5f}</td>"
                  f"<td class='mono dim'>{bt['mean_delta']*1000:+.2f}m · "
                  f"p={bt['p_better']:.2f}</td></tr>")

slump_rows = ""
SLABELS = {"slump_fav_gap2_streak2": "gap ≥ 2.0, favorite on 2+ series losing streak",
           "slump_fav_gap2.5_streak2": "gap ≥ 2.5, favorite on 2+ series losing streak",
           "slump_fav_gap2_streak1": "gap ≥ 2.0, favorite on 1+ series losing streak"}
for key, lbl in SLABELS.items():
    r = st3[key]
    slump_rows += (f"<tr><td>{lbl}</td><td class='mono'>{r['n']}</td>"
                   f"<td class='mono'><b>{r['fav_emp_winrate']:.1%}</b></td>"
                   f"<td class='mono'>{r['v6']['fav_pred_mean']:.1%}</td>"
                   f"<td class='mono'>{r['sym_20']['fav_pred_mean']:.1%}</td>"
                   f"<td class='mono'>{r['v6']['ll']:.4f} / {r['sym_20']['ll']:.4f}</td></tr>")

slate_rows = ""
sl_v6 = {(r["a"], r["b"]): r for r in st3["slate"]["v6"]}
sl_sym = {(r["a"], r["b"]): r for r in st3["slate"]["sym_20"]}
sl_f = {(r["a"], r["b"]): r for r in st3["slate"]["v6+form5"]}
for key_ in sl_v6:
    a, b = key_
    r6, rs, rf = sl_v6[key_], sl_sym.get(key_), sl_f.get(key_)
    star = " ★" if key_ in (("NS", "GE"), ("SEN", "LOUD"), ("GE", "NS"), ("LOUD", "SEN")) else ""
    hi = " style='background:var(--goodbg)'" if star else ""
    slate_rows += (f"<tr{hi}><td>{a} vs {b}{star} <span class='dim'>({r6['date']})</span></td>"
                   f"<td class='mono'><b>{r6['p_a']:.1%}</b></td>"
                   f"<td class='mono'>{rs['p_a']:.1%}</td>"
                   f"<td class='mono'>{rf['p_a']:.1%}</td>"
                   f"<td class='mono dim'>{(rs['p_a']-r6['p_a'])*100:+.1f}</td></tr>")

fs = st3["form_states"]
form_state_rows = "".join(
    f"<tr><td class='mono'>{t}</td><td class='mono'>{v['wr16']:.3f}</td>"
    f"<td class='mono'>{v['wr5']:.3f}</td>"
    f"<td class='mono'>{v['wr5']-v['wr16']:+.3f}</td></tr>"
    for t, v in fs.items())

kal = st3["kalshi"]

html = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex">
<title>v7 Lab — recency &amp; symmetry</title>
<link rel="icon" href="/logo.svg">
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@700;800&family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
 * {{ box-sizing:border-box; margin:0; }}
 :root {{ --ink:#16121d; --dim:#6b6478; --line:#eceef2; --acc:#7c4dd6; --accbg:#f3eefb;
         --good:#1e7a4f; --goodbg:#ecf8f1; --bad:#c0392b; --badbg:#fbeaea;
         --warn:#b3541e; --warnbg:#fdf3ec; }}
 body {{ font-family:'DM Sans',system-ui,sans-serif; background:#faf9fc; color:var(--ink);
        line-height:1.55; padding:30px 18px 90px; }}
 .wrap {{ max-width:880px; margin:0 auto; }}
 h1 {{ font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.55rem;
      text-align:center; margin:6px 0 2px; }}
 .tagline {{ text-align:center; color:var(--dim); font-size:.9rem; margin-bottom:18px; }}
 .labtabs {{ display:flex; justify-content:center; gap:6px; margin:0 0 24px; flex-wrap:wrap; }}
 .labtabs a {{ font-size:.8rem; font-weight:700; color:var(--dim); text-decoration:none;
   padding:7px 15px; border-radius:999px; border:1px solid var(--line); background:#fff; white-space:nowrap; }}
 .labtabs a:hover {{ color:var(--ink); background:var(--accbg); }}
 .labtabs a.on {{ color:#fff; background:var(--acc); border-color:var(--acc); }}
 .labtabs .brk {{ flex-basis:100%; height:0; margin:0; }}
 section {{ background:#fff; border:1px solid var(--line); border-radius:18px;
           padding:24px 28px; margin-bottom:16px; box-shadow:0 3px 14px #00000008; }}
 h2 {{ font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.08rem;
      margin-bottom:10px; display:flex; align-items:center; gap:10px; }}
 h2 .n {{ background:var(--acc); color:#fff; border-radius:8px; font-size:.78rem; width:24px;
        height:24px; display:inline-flex; align-items:center; justify-content:center; flex-shrink:0; }}
 p {{ font-size:.9rem; margin:7px 0; }}
 .dim {{ color:var(--dim); }} .good {{ color:var(--good); }} .bad {{ color:var(--bad); }}
 .mono {{ font-family:'JetBrains Mono',monospace; font-size:.82em; }}
 table {{ width:100%; border-collapse:collapse; font-size:.85rem; margin:8px 0; }}
 th {{ text-align:left; color:var(--dim); font-size:.7rem; text-transform:uppercase;
      letter-spacing:.5px; padding:6px 9px; border-bottom:2px solid var(--line); }}
 td {{ padding:7px 9px; border-bottom:1px solid var(--line); }}
 tr:last-child td {{ border-bottom:0; }}
 .callout {{ border-left:4px solid var(--acc); background:var(--accbg); border-radius:0 12px 12px 0;
            padding:12px 16px; margin:10px 0; font-size:.89rem; }}
 .callout.good {{ border-color:var(--good); background:var(--goodbg); }}
 .callout.warn {{ border-color:var(--warn); background:var(--warnbg); }}
 ul {{ font-size:.9rem; margin:8px 0 8px 20px; }} li {{ margin:5px 0; }}
</style></head>
<body><div class="wrap">
<h1>v7 Lab — recency &amp; symmetry</h1>
<div class="tagline">Operator hypotheses (2026-07-23): "not enough recency bias" and "I
don't like the asymmetry." 18 decay shapes + fitted form features + case forensics +
market benchmark, all walk-forward · holdout = 2025–26 ({res1['v6_consist_20_12']['n_test']} series)</div>
{NAV}

<section>
<h2><span class="n">0</span>Verdict up front</h2>
<div class="callout"><b>v6 stands. Both hypotheses were tested to death and neither
survives contact with the data.</b> More recency loses monotonically — every step
shorter in memory is a step worse, in total and in the exact buckets built to catch a
recency edge. And the "asymmetry" turns out to be the model's implementation of a real,
measurable phenomenon: <u>recent form is mean-reverting noise, and the correct response
is to fade it</u>. Remove the asymmetry and the walk-forward fit immediately demands a
contrarian form-correction term to replace it.</div>
<p>The two complaints are actually one complaint, inverted: consistency-conditioned decay
is <i>anti</i>-recency by construction (a form change is "anomalous" vs the team's trailing
level, so it fades at HL 12 instead of persisting at HL 20). Disliking the asymmetry and
wanting more recency both amount to trusting recent form more. The data says: don't.</p>
</section>

<section>
<h2><span class="n">1</span>The recency axis — 18 decay shapes, walk-forward</h2>
<p>Champion stack held fixed (RD^0.75, playoff ×1.6, region ridge 1.5, year-boundary
roster); only the decay changes. β refit on train (≤2024) per config; scored on 2025–26.
"FS bucket" = the 121 holdout matches where a team's 5-map form diverged ≥15pts from its
16-map level — the exact matches a recency-biased model should win.</p>
<table>
<tr><th>decay</th><th>holdout LL</th><th>2026 LL</th><th>FS-bucket LL</th>
<th>P(mover)</th><th>vs v6 (boot)</th></tr>
{grid_rows}
</table>
<div class="callout warn"><b>The form-shift bucket empirical fact:</b> teams whose recent
form surged won only <b>47.1%</b> of these matches — recent form divergence carries
<i>zero</i> forward signal beyond the ratings (if anything it mean-reverts). Every model
that prices the "hot" team up (sym_6 gives them 55.2%) gets punished in exactly the bucket
that motivated it. v6 prices them at 48.2% — the closest to truth of all 18 configs.</div>
</section>

<section>
<h2><span class="n">2</span>The symmetry axis — what does asymmetry actually buy?</h2>
<p>Best symmetric config (HL 20) costs <b>−1.65 milli</b> vs v6 (p_better 0.14 — v6 ahead
but not at the promotion bar in reverse either). The reversed conditioning
("surprises are news → persist them, confirmations fade") is the pro-recency mirror image
of v6 — it loses <b>−7.4m / −5.0m</b>. The direction of v6's conditioning is not a
stylistic choice; it is the direction the data picks, decisively.</p>
<p>Then the sharper test: keep ratings symmetric (HL 20) and let a fitted probability-layer
term add back whatever recency the ratings miss:
z = β·Δrating + b_form·Δ(form₅ − form₁₆). If the model lacked recency, b_form would fit
positive.</p>
<table>
<tr><th>base + form horizon</th><th>fitted b_form</th><th>holdout LL</th>
<th>FS bucket</th><th>vs v6</th></tr>
{form_rows}
</table>
<div class="callout"><b>The fitted amount of extra recency is zero to negative.</b> On the
v6 base, b_form fits −0.02 to −0.09 (a whisper of form-fade, no holdout gain). On the
symmetric base it fits <b>−0.27 to −0.71</b>: strip the consistency conditioning and the
optimizer immediately rebuilds a fade-recent-form correction at the probability layer —
and still lands ~2m short of v6. The asymmetry <i>is</i> the calibrated form fade.</div>
</section>

<section>
<h2><span class="n">3</span>The NS-GE shape: slumping big favorites (case evidence)</h2>
<p>The trigger for this research: NS at 68% over GE while on a two-series losing streak
felt too high. Historical analogs — holdout matches where the rating favorite entered on a
losing streak:</p>
<table>
<tr><th>bucket</th><th>n</th><th>favorite won</th><th>v6 priced</th><th>sym_20 priced</th>
<th>LL v6 / sym</th></tr>
{slump_rows}
</table>
<div class="callout good"><b>Slumping favorites win at almost exactly the rate v6 charges
for them.</b> Gap ≥ 2 + 2-loss streak: won 68.8%, priced 69.8%. The 68% on NS is not a
model artifact — it is the historically calibrated price for this exact situation, and v6
scores better here than the symmetric alternative.</div>
<p class="dim">Current form states (decayed map winrate, 16-map vs 5-map horizon):</p>
<table>
<tr><th>team</th><th>wr₁₆ (level)</th><th>wr₅ (form)</th><th>form − level</th></tr>
{form_state_rows}
</table>
</section>

<section>
<h2><span class="n">4</span>Today's slate under the three finalists</h2>
<p>★ = the matches that prompted this research. "Δ sym" = how much the symmetric model
disagrees with v6 — this is the full size of the philosophical choice, match by match.</p>
<table>
<tr><th>match (P of first team)</th><th>v6</th><th>sym_20</th><th>v6+form5</th><th>Δ sym</th></tr>
{slate_rows}
</table>
</section>

<section>
<h2><span class="n">5</span>Market benchmark (sanctioned use: benchmark only)</h2>
<p>Kalshi 2026 overlap, pre-match T-2h prices, n={kal['n']}: market LL
<span class="mono">{kal['market_t2h']:.5f}</span> · v6
<span class="mono">{kal['v6']:.5f}</span> · sym_20
<span class="mono">{kal['sym_20']:.5f}</span> · v6+form5
<span class="mono">{kal['v6+form5']:.5f}</span>. A three-way statistical tie (bootstrap
CIs ±0.07). The market is not seeing a recency edge the model misses either — if hot form
carried signal, the market would monetize it against us on exactly these matches, and it
doesn't.</p>
</section>

<section>
<h2><span class="n">6</span>What would change this verdict (tripwires)</h2>
<ul>
<li><b>FS-bucket drift:</b> if the form-shift mover win rate climbs above 55% on a rolling
120-match window, recency is becoming real signal — rerun stage 2 first (the b_form fit
is the cheapest detector).</li>
<li><b>Slump-fav drift:</b> if slumping favorites (gap≥2, streak≥2) start winning 5+pts
below the v6 price on n≥50 fresh matches, the form fade is over-tuned — revisit
consistency HLs (16/10 was a near-tie at −0.4m).</li>
<li><b>If symmetry is wanted anyway</b> (as an operator preference, not an accuracy
claim): sym_20 is the honest price — about −1.7 milli of holdout log-loss and slightly
worse slump-favorite calibration. Nothing in this research supports paying it.</li>
</ul>
<p class="dim">Do-not-retest ledger additions: symmetric HL &lt; 14 (monotone worse);
surprise-persist decay both directions; form-delta probability feature at HL 3/5/8 (fits
≈0 on v6, negative on symmetric bases). All rejected walk-forward on 2025–26 holdout,
2026-07-24.</p>
</section>

<p class="dim" style="text-align:center">v7 Lab · research run 2026-07-23/24 · engine:
testing_lab walk-forward Massey replica · v6 remains the champion and the
trading_model snapshot is unchanged</p>
</div></body></html>"""

os.makedirs(RD, exist_ok=True)
with open(os.path.join(RD, "v7_lab.html"), "w") as f:
    f.write(html)
print("wrote", os.path.join(RD, "v7_lab.html"))
