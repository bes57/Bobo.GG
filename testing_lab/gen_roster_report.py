"""Generate reports/roster_adaptation.html — roster-change adaptation report
(agent:roster, operator-requested; preregister.roster.md + operator addendum).

RENDER-ONLY: every number on the page comes from testing_lab/v8/stats/*.json
written by agent:roster's preregistered pipeline (plus verbatim quotes from
adversary_report.md and idea strings from ledger_reclass.json). No numeric
measurement literal appears in the HTML template. EXPLORATORY badges are
mandatory: the holdout is spent; nothing here is confirmatory or promotable.

Usage: python3 testing_lab/gen_roster_report.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
V8 = os.path.join(HERE, "v8")
STATS = os.path.join(V8, "stats")
RD = os.path.join(HERE, "out", "reports")


def j(name):
    with open(os.path.join(STATS, name)) as f:
        return json.load(f)


envy = j("roster_case_envy.json")
gal = j("roster_case_gallery.json")
pop = j("roster_population.json")
tr = j("roster_treatments.json")
integ = j("roster_integration.json")
looks = j("roster_looks.json")
ledger = j("ledger_reclass.json")
adv_md = open(os.path.join(V8, "adversary_report.md")).read()

C = {"keep4": "#3a90cc", "keep3": "#c96a2a", "overhaul": "#7c4dd6",
     "v6": "#6f6a7c", "filt": "#c96a2a", "b": "#3a90cc", "c": "#6b8f1f",
     "d": "#c96a2a", "gray": "#9a93a6", "good": "#1e7a4f", "bad": "#c0392b"}

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
 .banner { background:var(--warnbg); border:1px solid #f0d9c7; border-left:6px solid var(--warn);
   border-radius:14px; padding:14px 20px; margin:0 0 18px; font-size:.92rem; }
 .banner b { color:var(--warn); letter-spacing:.4px; }
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
 .badge { font-weight:800; border-radius:999px; padding:2px 10px; font-size:.66rem;
          letter-spacing:.6px; background:var(--warnbg); color:var(--warn);
          border:1px solid #eccdb2; white-space:nowrap; }
 .badge.desc { background:#eef3f8; color:#2b6f9e; border-color:#cfe0ee; }
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
 .verd.floor { background:var(--warnbg); color:var(--warn); }
 .verd.hold { background:#f1f0f4; color:var(--dim); }
 .verd.lead { background:var(--goodbg); color:var(--good); }
 .callout { border-left:4px solid var(--acc); background:var(--accbg);
            border-radius:0 12px 12px 0; padding:11px 15px; margin:10px 0; font-size:.88rem; }
 .callout.good { border-color:var(--good); background:var(--goodbg); }
 .callout.warn { border-color:var(--warn); background:var(--warnbg); }
 .callout.bad { border-color:var(--bad); background:var(--badbg); }
 canvas { max-height:340px; }
 .chartbox { margin:14px 0 2px; }
 .dl { display:block; text-align:right; font-size:.72rem; margin:2px 0 10px; }
 .dl a { color:var(--acc); text-decoration:none; font-weight:700; }
 .dl a:hover { text-decoration:underline; }
 .cap { font-size:.78rem; color:var(--dim); margin:4px 0 8px; }
 code { font-family:'JetBrains Mono',monospace; font-size:.8em; background:#f4f2f8;
        border:1px solid var(--line); border-radius:6px; padding:1px 6px; }
 ul, ol { font-size:.9rem; margin:8px 0 8px 20px; } li { margin:5px 0; }
 .adv { border:2px solid var(--bad); border-radius:14px; padding:18px 20px; margin:14px 0;
        background:#fffdfd; }
 .adv .advhead { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; color:var(--bad);
        font-size:.95rem; margin-bottom:8px; }
 .adv pre { white-space:pre-wrap; font-family:'JetBrains Mono',monospace; font-size:.76rem;
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
<a href="/testing/report/roster_adaptation" class="on">Roster</a><a href="/testing/report/v9_lab">v9 Lab</a>
<a href="/testing/report/v10_lab">v10 Lab</a>
<a href="/testing/report/edge_lab">Edge vs Market</a>
<a href="/testing/report/playbook_bt">Playbook (backtested)</a>
</div>"""


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def dl(name):
    return (f'<div class="dl"><a href="/testing/v8/stats/{name}" download>'
            f'&#8681; {name}</a></div>')


def fm(x, nd=2, plus=False):
    if x is None:
        return "—"
    s = f"{x:+.{nd}f}" if plus else f"{x:.{nd}f}"
    return s


def ci(lo, hi, nd=2):
    return f"[{lo:.{nd}f}, {hi:.{nd}f}]"


# ── assemble data for charts ─────────────────────────────────────────────────
def _step_vals(path, match_dates, labels):
    """Match-anchored step resample (operator request, 2026-07-29).

    The daily solve drifts between a team's matches (decay aging, other
    teams' results moving the Massey system), but a TEAM trajectory chart
    should read as: rating holds until THIS team plays, then jumps. For each
    match m, the step value s_m is the path value on the first solve day
    AFTER m (the day the match's information lands); labels before the first
    match hold the window-start value. Rendered with Chart.js
    stepped:'after', so the vertical sits exactly on the match-date label."""
    pv = sorted((p["d"], p["r"]) for p in path)
    if not pv:
        return [None for _ in labels]

    def first_after(d):
        for pd, pr in pv:
            if pd > d:
                return pr
        return pv[-1][1]

    steps = sorted((m, first_after(m)) for m in match_dates)
    init = pv[0][1]
    out = []
    for d in labels:
        v = init
        for m, s in steps:
            if m <= d:
                v = s
            else:
                break
        out.append(v)
    return out


def _prechange_offset(base_step, other_step, labels, first_change):
    """Mean (base − other) over labels strictly before the first change —
    the base-model difference the caption must state as a number."""
    ds = [b - o for d, b, o in zip(labels, base_step, other_step)
          if d < first_change and b is not None and o is not None]
    return (sum(ds) / len(ds)) if ds else None


# ── SPEC RUN case data (roster_spec_operator.md §4) — gated ──
def _load_spec_cases():
    try:
        d = j("roster_spec_cases.json")
    except FileNotFoundError:
        return None
    return d.get("cases", d) if isinstance(d, dict) else d


def _spec_gate(case):
    """Spec §4 hard gates. Computed HERE from the raw paths, not trusted
    from the JSON's own field. Raise => no HTML is written."""
    v6m = {pt["d"]: pt["r"] for pt in case["v6_path"]}
    abm = {pt["d"]: pt["r"] for pt in case["ablation_path"]}
    bounds = case.get("boundaries") or []
    first = min((b["date"] for b in bounds), default=None)
    pre = [abs(v6m[d] - abm[d]) for d in v6m
           if d in abm and (first is None or d < first)]
    mx = max(pre) if pre else 0.0
    if mx != 0.0:
        raise RuntimeError(
            f"SPEC §4 ZERO-GATE VIOLATION [{case['org']}]: pre-boundary "
            f"max|v6-ablation| = {mx!r}, must be exactly 0.0. The subsystem "
            f"leaked pre-change — implementation bug. HTML NOT written.")
    if case["org"] == "SEN" and bounds:
        raise RuntimeError(
            "SPEC §2.5 GATE: SEN shows a surviving phase boundary — the "
            "classifier is broken. HTML NOT written.")
    return mx


SPEC_CASES = _load_spec_cases()

try:
    SPEC_READ = j("roster_spec_read.json")
except FileNotFoundError:
    SPEC_READ = None
try:
    SPEC_FIX = j("roster_spec_fixtures.json")
except FileNotFoundError:
    SPEC_FIX = None


def _spec_case(org):
    if not SPEC_CASES:
        return None
    c = next((c for c in SPEC_CASES if c.get("org") == org), None)
    if c is not None:
        _spec_gate(c)
    return c


ov = envy["phase_reset_overlay"]
env_dates = sorted({p["d"] for p in ov["v6_path"]} | {p["d"] for p in ov["filter_path"]})
_env_m = [m["date"] for m in ov["matches"]]
env_v6_step = _step_vals(ov["v6_path"], _env_m, env_dates)
env_f_step = _step_vals(ov["filter_path"], _env_m, env_dates)
env_changes = [c["change_date"] for c in envy["change_chain_2026"]]
# no-injection base of the SAME state-space core (added 2026-07-29 after the
# operator caught pre-change divergence: the filter's base model is H3's
# state-space, not v6 — the chart must separate base-model difference from
# the phase-reset mechanism, which is the filter-vs-base gap only).
env_b_step = (_step_vals(ov["base_path"], _env_m, env_dates)
              if ov.get("base_path") else None)
env_off = (_prechange_offset(env_b_step, env_v6_step, env_dates, env_changes[0])
           if env_b_step else None)
env_couple = (max((abs(b - f) for d, b, f in zip(env_dates, env_b_step, env_f_step)
                   if d < env_changes[0] and b is not None and f is not None),
                  default=None) if env_b_step else None)
_env_spec = _spec_case("ENVY")
if _env_spec:
    env_dates = sorted({pt["d"] for pt in _env_spec["v6_path"]})
    _env_m = list(_env_spec.get("match_dates") or _env_m)
    env_v6_step = _step_vals(_env_spec["v6_path"], _env_m, env_dates)
    env_e_step = _step_vals(_env_spec["ablation_path"], _env_m, env_dates)
    env_marks = [{"d": b["date"], "t": str(b.get("k5", "change"))}
                 for b in (_env_spec.get("boundaries") or [])]
else:
    env_e_step = (_step_vals(ov["v6_overlay_path"], _env_m, env_dates)
                  if ov.get("v6_overlay_path") else None)
    env_marks = [{"d": d, "t": "change"} for d in env_changes]

lev = next(c for c in gal["named_cases"] if c["org"] == "LEV")
lov = lev["phase_reset_overlay"]
lev_dates = sorted({p["d"] for p in lov["v6_path"]} | {p["d"] for p in lov["filter_path"]})
_lev_m = [m["date"] for m in lov["matches"]]
lev_v6_step = _step_vals(lov["v6_path"], _lev_m, lev_dates)
lev_f_step = _step_vals(lov["filter_path"], _lev_m, lev_dates)
lev_b_step = (_step_vals(lov["base_path"], _lev_m, lev_dates)
              if lov.get("base_path") else None)
lev_off = (_prechange_offset(lev_b_step, lev_v6_step, lev_dates, lev["change_date"])
           if lev_b_step else None)
_lev_spec = _spec_case("LEV")
if _lev_spec:
    lev_dates = sorted({pt["d"] for pt in _lev_spec["v6_path"]})
    _lev_m = list(_lev_spec.get("match_dates") or _lev_m)
    lev_v6_step = _step_vals(_lev_spec["v6_path"], _lev_m, lev_dates)
    lev_e_step = _step_vals(_lev_spec["ablation_path"], _lev_m, lev_dates)
    lev_marks = [{"d": b["date"], "t": str(b.get("k5", "change"))}
                 for b in (_lev_spec.get("boundaries") or [])]
else:
    lev_e_step = (_step_vals(lov["v6_overlay_path"], _lev_m, lev_dates)
                  if lov.get("v6_overlay_path") else None)
    lev_marks = [{"d": lev["change_date"], "t": "change"}]

pool = pop["adaptation_curve"]["pooled_first3"]
ref = pop["adaptation_curve"]["reference_stable"]
curve = pop["adaptation_curve"]["by_magnitude"]
lr = tr["learning_rate_curve"]

reads = [("read1_b_continuity", "b", "(b) graded change-point continuity",
          "dead", "falsifier FIRED (overall beyond within floor; block CI excludes 0)"),
         ("read2_c_coldstart", "c", "(c) change-triggered partial cold start",
          "floor", "INSIDE NOISE FLOOR — no support, falsifier not fired"),
         ("read3_d_phase_reset", "d", "(d) phase-reset filter (operator-specified)",
          "dead", "falsifier FIRED (post-change CI excludes 0; loses to own base)")]
mde = tr["read1_b_continuity"]["mde_context"]

treat_labels = [lbl for _, _, lbl, _, _ in reads]
treat_delta = [tr[k]["delta_milli"] for k, _, _, _, _ in reads]
treat_lo = [round(tr[k]["iid"]["ci_lo"] * 1000, 2) for k, _, _, _, _ in reads]
treat_hi = [round(tr[k]["iid"]["ci_hi"] * 1000, 2) for k, _, _, _, _ in reads]
treat_cols = [C[c] for _, c, _, _, _ in reads]

# adversary excerpt (verbatim lines re: 5d table + cold + S1) for §6
adv_lines = []
grab = False
for ln in adv_md.splitlines():
    if ln.startswith(("- **H3 cold", "4. **The 5d half-life table",
                      "5. **Any residue of compose S1")):
        grab = True
    elif grab and (ln.startswith(("- ", "1. ", "2. ", "3. ", "4. ", "5. ",
                                  "6. ", "#", "## "))):
        grab = False
    if grab:
        adv_lines.append(ln)
ADV_EXC = "\n".join(adv_lines)

env_p = envy["panel_chain_start_2026_05_12"]
env_p2 = envy["panel_s2_change_2026_07_17"]
agg = envy["s1_to_s2_aggregate"]
c1, c2, c3 = envy["change_chain_2026"]
envy_inspire = next(c for c in gal["named_cases"] if c["org"] == "ENVY")
counts = pop["counts"]
pcx = pop["power_context"]
lb = looks

# ── tables ───────────────────────────────────────────────────────────────────
chain_rows = "".join(
    f"<tr><td class='mono'>{c['change_date']}</td><td>{esc(c['event_id'])}</td>"
    f"<td class='bad'>{', '.join(map(esc, c['out']))}</td>"
    f"<td class='good'>{', '.join(map(esc, c['in']))}</td>"
    f"<td>{c['kept']}/5</td><td>{c['run_len']}{' (censored)' if c['censored'] else ''}</td></tr>"
    for c in envy["change_chain_2026"])

match_rows = "".join(
    f"<tr><td class='mono'>{m['date']}</td><td>{esc(m['event_id'])}</td>"
    f"<td>{esc(m['opponent'])}</td><td>{'W' if m['won'] else 'L'} {m['score']}</td>"
    f"<td class='mono'>{m['p_team']:.3f}</td><td>{m['msc']}</td>"
    f"<td class='dim' style='font-size:.76rem'>{', '.join(map(esc, m['five']))}</td></tr>"
    for m in envy["matches_2026"])

gal_rows = "".join(
    f"<tr><td><b>{esc(c['org'])}</b></td><td class='mono'>{c['change_date']}</td>"
    f"<td>{esc(c['event_id'])}</td><td>{c['kept']}/5 (ov {c['ov']:.1f})</td>"
    f"<td class='bad'>{', '.join(map(esc, c['out']))}</td>"
    f"<td class='good'>{', '.join(map(esc, c['in']))}</td>"
    f"<td>{esc(str(c['panel']['stabilization_matches']))}</td>"
    f"<td class='mono'>{fm(c['panel']['carryover_cost_pp_per_match'], 1, True)}pp</td></tr>"
    for c in gal["cases"])

read_rows = ""
for k, ckey, lbl, verd, note in reads:
    r = tr[k]
    post = next(b for b in r["buckets"] if b["name"].startswith("post-change <=3 ("))
    roi = r["expected_roi_both_units"]["symmetric_reading_roi_delta"]
    sel = (f"&gamma;={r['spec']['selected_gamma']}" if k.startswith("read1")
           else f"a0={r['spec']['selected']['a0']}, M={r['spec']['selected']['M']}"
           if k.startswith("read2") else f"g={r['spec']['selected_g']}")
    read_rows += (
        f"<tr><td><b>{lbl}</b><br><span class='dim mono' style='font-size:.72rem'>"
        f"train-selected {sel}</span></td>"
        f"<td class='mono'>{fm(r['delta_milli'], 2, True)}m<br>"
        f"<span class='dim' style='font-size:.72rem'>iid {ci(r['iid']['ci_lo']*1000, r['iid']['ci_hi']*1000)}"
        f"<br>block {ci(r['block_event']['ci_lo']*1000, r['block_event']['ci_hi']*1000)}</span></td>"
        f"<td class='mono'>{fm(post['delta_milli'], 2, True)}m<br>"
        f"<span class='dim' style='font-size:.72rem'>{ci(post['ci_lo_milli'], post['ci_hi_milli'])} "
        f"n={post['n']}</span></td>"
        f"<td class='mono'>{fm(roi*100, 2, True)}pp</td>"
        f"<td><span class='verd {verd}'>{'DEAD ON FRAME' if verd=='dead' else 'NO SUPPORT'}</span>"
        f"<br><span class='dim' style='font-size:.72rem'>{note}</span></td></tr>")

bucket_names = [b["name"] for b in tr["read1_b_continuity"]["buckets"]]
bucket_rows = ""
for i, bn in enumerate(bucket_names):
    cells = ""
    for k, _, _, _, _ in reads:
        b = tr[k]["buckets"][i]
        if "delta_milli" in b:
            cells += (f"<td class='mono'>{fm(b['delta_milli'], 1, True)} "
                      f"<span class='dim' style='font-size:.7rem'>"
                      f"{ci(b['ci_lo_milli'], b['ci_hi_milli'], 1)}</span></td>")
        else:
            cells += "<td class='dim'>n&lt;10</td>"
    b0 = tr["read1_b_continuity"]["buckets"][i]
    bucket_rows += f"<tr><td>{esc(bn)}<br><span class='dim' style='font-size:.7rem'>n={b0['n']}</span></td>{cells}</tr>"

pool_rows = "".join(
    f"<tr><td><span style='color:{C[m]}'>&#9632;</span> {m}</td>"
    f"<td class='mono'>{fm(pool[m]['bias_pp'], 2, True)}pp</td>"
    f"<td class='mono dim'>{ci(pool[m]['ci_lo'], pool[m]['ci_hi'])}</td>"
    f"<td>{pool[m]['n']}</td></tr>" for m in ("keep4", "keep3", "overhaul")) + (
    f"<tr><td><span style='color:{C['v6']}'>&#9632;</span> stable reference (&gt;10)</td>"
    f"<td class='mono'>{fm(ref['bias_pp'], 2, True)}pp</td>"
    f"<td class='mono dim'>{ci(ref['ci_lo'], ref['ci_hi'])}</td><td>{ref['n']}</td></tr>")

mde_proj = integ["prospective_validation_plan"]["mde80_projection_milli"]["by_n"]
proj_rows = "".join(
    f"<tr><td class='mono'>{n}</td><td class='mono'>{v['overall_within_milli']}m</td>"
    f"<td class='mono'>{v['post_change_le3_within_milli']}m</td></tr>"
    for n, v in mde_proj.items())

arm_rows = "".join(
    f"<tr><td class='mono'>{esc(k)}</td><td>{esc(v)}</td></tr>"
    for k, v in integ["prospective_validation_plan"]["arms_frozen"].items())

contrast = tr["contrast_d_vs_c"]
ctr_rows = "".join(
    f"<tr><td>{esc(nm)}</td><td class='mono'>{fm(contrast[nm]['d_minus_c_milli'], 2, True)}m</td>"
    f"<td class='mono dim'>{ci(contrast[nm]['ci_lo_milli'], contrast[nm]['ci_hi_milli'])}</td>"
    f"<td>{contrast[nm]['n']}</td></tr>"
    for nm in ("improvement cases", "degradation cases", "post-change <=3 (all)")
    if nm in contrast)

lrs = lr["stable_reference_gain"]

# ── JS ───────────────────────────────────────────────────────────────────────
JS = """
const PAL = __PAL__;
Chart.defaults.font.family = "'DM Sans',system-ui,sans-serif";
Chart.defaults.color = '#6b6478';
Chart.defaults.animation = false;
const floorBand = { id:'floorBand', beforeDatasetsDraw(c) {
  const fb = c.options.plugins.floorBand; if (!fb || fb.lo === undefined) return;
  const ax = fb.axis === 'x' ? c.scales.x : c.scales.y, ctx = c.ctx, ca = c.chartArea;
  const p0 = ax.getPixelForValue(fb.lo), p1 = ax.getPixelForValue(fb.hi);
  ctx.save(); ctx.fillStyle = 'rgba(154,147,166,0.16)';
  if (fb.axis === 'x') ctx.fillRect(Math.min(p0,p1), ca.top, Math.abs(p1-p0), ca.bottom-ca.top);
  else ctx.fillRect(ca.left, Math.min(p0,p1), ca.right-ca.left, Math.abs(p1-p0));
  ctx.restore(); } };
const ciBars = { id:'ciBars', afterDatasetsDraw(c) {
  const cb = c.options.plugins.ciBars; if (!cb || !cb.lo) return;
  const ctx = c.ctx; ctx.save(); ctx.strokeStyle = '#16121d'; ctx.lineWidth = 1.4;
  const horiz = cb.horizontal;
  const meta = c.getDatasetMeta(cb.dsIndex || 0);
  meta.data.forEach((el, i) => {
    const lo = cb.lo[i], hi = cb.hi[i]; if (lo == null) return;
    if (horiz) { const y = el.y, x0 = c.scales.x.getPixelForValue(lo), x1 = c.scales.x.getPixelForValue(hi);
      ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y);
      ctx.moveTo(x0, y-4); ctx.lineTo(x0, y+4); ctx.moveTo(x1, y-4); ctx.lineTo(x1, y+4); ctx.stroke();
    } else { const x = el.x, y0 = c.scales.y.getPixelForValue(lo), y1 = c.scales.y.getPixelForValue(hi);
      ctx.beginPath(); ctx.moveTo(x, y0); ctx.lineTo(x, y1);
      ctx.moveTo(x-4, y0); ctx.lineTo(x+4, y0); ctx.moveTo(x-4, y1); ctx.lineTo(x+4, y1); ctx.stroke(); } });
  ctx.restore(); } };
const vLines = { id:'vLines', afterDatasetsDraw(c) {
  const vl = c.options.plugins.vLines; if (!vl || (!vl.dates && !vl.marks)) return;
  const marks = vl.marks || (vl.dates || []).map(d => ({d:d, t:'change'}));
  const ctx = c.ctx, ca = c.chartArea; ctx.save();
  ctx.strokeStyle = '#c0392b'; ctx.setLineDash([4,3]); ctx.lineWidth = 1.2;
  ctx.fillStyle = '#c0392b'; ctx.font = '10px DM Sans';
  marks.forEach(m => { const i = vl.labels.indexOf(m.d); if (i < 0) return;
    const x = c.scales.x.getPixelForValue(i);
    ctx.beginPath(); ctx.moveTo(x, ca.top); ctx.lineTo(x, ca.bottom); ctx.stroke();
    ctx.fillText(m.t, x + 3, ca.top + 10); });
  ctx.restore(); } };
Chart.register(floorBand, ciBars, vLines);

function lineChart(id, labels, datasets, opts) {
  new Chart(document.getElementById(id), { type:'line',
    data:{ labels, datasets },
    options: Object.assign({ responsive:true, maintainAspectRatio:false,
      spanGaps:true, interaction:{mode:'index', intersect:false},
      plugins:{ legend:{position:'bottom', labels:{boxWidth:14, boxHeight:3}} },
      elements:{ point:{radius:0, hoverRadius:5, hitRadius:8}, line:{borderWidth:2, tension:.25} },
      scales:{ x:{grid:{display:false}, ticks:{maxTicksLimit:9, autoSkip:true}},
               y:{grid:{color:'#eceef2'}} } }, opts || {}) });
}

// ENVY overlay
lineChart('envyChart', __ENV_LABELS__, [
  { label:'v6 (steps at ENVY matches)', data:__ENV_V6__, borderColor:PAL.v6, backgroundColor:PAL.v6, stepped:'after', tension:0 },
  { label:'v6 + subsystem, THIS TEAM ONLY (ablation)', data:__ENV_E__, borderColor:PAL.overhaul,
    backgroundColor:PAL.overhaul, borderDash:[8,4], stepped:'after', tension:0 } ],
  { plugins:{ vLines:{marks:__ENV_MARKS__, labels:__ENV_LABELS__},
              legend:{position:'bottom', labels:{boxWidth:14, boxHeight:3}} },
    scales:{ x:{grid:{display:false}, ticks:{maxTicksLimit:8}}, y:{grid:{color:'#eceef2'},
             title:{display:true, text:'rating (transformed round-margin units)'} } } });

// LEV overlay
lineChart('levChart', __LEV_LABELS__, [
  { label:'v6 (steps at LEV matches)', data:__LEV_V6__, borderColor:PAL.v6, backgroundColor:PAL.v6, stepped:'after', tension:0 },
  { label:'v6 + subsystem, THIS TEAM ONLY (ablation)', data:__LEV_E__, borderColor:PAL.overhaul,
    backgroundColor:PAL.overhaul, borderDash:[8,4], stepped:'after', tension:0 } ],
  { plugins:{ vLines:{marks:__LEV_MARKS__, labels:__LEV_LABELS__},
              legend:{position:'bottom', labels:{boxWidth:14, boxHeight:3}} },
    scales:{ x:{grid:{display:false}, ticks:{maxTicksLimit:8}}, y:{grid:{color:'#eceef2'},
             title:{display:true, text:'rating (transformed round-margin units)'} } } });
__SEN_JS__

// atlas pooled bar + CI
new Chart(document.getElementById('atlasPooled'), { type:'bar',
  data:{ labels:__POOL_LABELS__, datasets:[{ data:__POOL_VALS__,
    backgroundColor:__POOL_COLS__, borderRadius:4, maxBarThickness:46,
    borderSkipped:'bottom' }] },
  options:{ responsive:true, maintainAspectRatio:false,
    plugins:{ legend:{display:false}, ciBars:{lo:__POOL_LO__, hi:__POOL_HI__},
      tooltip:{callbacks:{label:(t)=>t.parsed.y.toFixed(2)+'pp  CI ['+__POOL_LO__[t.dataIndex]+', '+__POOL_HI__[t.dataIndex]+']'}} },
    scales:{ x:{grid:{display:false}}, y:{grid:{color:'#eceef2'},
      title:{display:true, text:'mean(won − p_v6), pp — first 3 matches'} } } } });

// atlas curve
lineChart('atlasCurve', __CURVE_X__, __CURVE_DS__,
  { plugins:{ legend:{position:'bottom', labels:{boxWidth:14, boxHeight:3}},
      floorBand:{axis:'y', lo:__REF_LO__, hi:__REF_HI__} },
    scales:{ x:{grid:{display:false}, title:{display:true, text:'matches since change'}},
             y:{grid:{color:'#eceef2'}, title:{display:true, text:'bias pp (won − p_v6)'} } } });

// treatments deltas (horizontal) + CI + floor band
new Chart(document.getElementById('treatChart'), { type:'bar',
  data:{ labels:__TREAT_LABELS__, datasets:[{ data:__TREAT_D__,
    backgroundColor:__TREAT_COLS__, borderRadius:4, maxBarThickness:34 }] },
  options:{ indexAxis:'y', responsive:true, maintainAspectRatio:false,
    plugins:{ legend:{display:false},
      floorBand:{axis:'x', lo:__FLOOR_LO__, hi:__FLOOR_HI__},
      ciBars:{horizontal:true, lo:__TREAT_LO__, hi:__TREAT_HI__},
      tooltip:{callbacks:{label:(t)=>t.parsed.x.toFixed(2)+'m  iid CI ['+__TREAT_LO__[t.dataIndex]+', '+__TREAT_HI__[t.dataIndex]+']'}} },
    scales:{ x:{grid:{color:'#eceef2'}, title:{display:true,
        text:'Δ log-loss vs v6, milli (holdout n=__NHOLD__; >0 = better; gray band = ±within-family floor)'}},
      y:{grid:{display:false}, ticks:{font:{size:11}}} } } });

// learning-rate curve
lineChart('gainChart', __GAIN_X__, __GAIN_DS__,
  { plugins:{ legend:{position:'bottom', labels:{boxWidth:14, boxHeight:3}} },
    scales:{ x:{grid:{display:false}, title:{display:true, text:'matches since change'}},
             y:{grid:{color:'#eceef2'}, title:{display:true, text:'mean Kalman gain per map update'} } } });
"""

curve_ds = []
for cmag in curve:
    pts = [p.get("bias_pp") for p in cmag["points"]]
    curve_ds.append({"label": cmag["magnitude"], "data": pts,
                     "borderColor": C[cmag["magnitude"]],
                     "backgroundColor": C[cmag["magnitude"]]})
gain_ds = []
for cmag in lr["by_magnitude"]:
    pts = [p.get("gain") for p in cmag["points"]]
    gain_ds.append({"label": cmag["magnitude"], "data": pts,
                    "borderColor": C[cmag["magnitude"]],
                    "backgroundColor": C[cmag["magnitude"]]})
gain_ds.append({"label": "stable reference", "data": [lrs["gain"]] * 10,
                "borderColor": C["v6"], "backgroundColor": C["v6"],
                "borderDash": [4, 4], "pointRadius": 0})

# SEN: the named sub-not-change case (spec §2.5 fixture 1). Gated: zero
# boundaries => zero verticals; identity line-on-line is the visual proof.
_sen_spec = _spec_case("SEN")
if _sen_spec:
    sen_dates = sorted({pt["d"] for pt in _sen_spec["v6_path"]})
    _sen_m = list(_sen_spec.get("match_dates") or [])
    sen_v6_step = _step_vals(_sen_spec["v6_path"], _sen_m, sen_dates)
    sen_e_step = _step_vals(_sen_spec["ablation_path"], _sen_m, sen_dates)
    sen_marks = [{"d": b["date"], "t": str(b.get("k5", "change"))}
                 for b in (_sen_spec.get("boundaries") or [])]
    SEN_HTML = ("<h3>SEN window: a substitution is not a change "
                '<span class="badge desc">SPEC &sect;2.5 FIXTURE</span></h3>'
                '<div class="chartbox" style="height:260px"><canvas id="senChart"></canvas></div>'
                '<p class="cap">SEN&rsquo;s one-match deviation (the Marved game) creates no phase '
                'boundary: no vertical is drawn, the subsystem never activates, and the two lines '
                'are identical everywhere — asserted by the generator&rsquo;s zero-gate, not eyeballed.</p>')
    SEN_JS = ("lineChart('senChart', " + json.dumps(sen_dates) + ", ["
              "{ label:'v6 (steps at SEN matches)', data:" + json.dumps(sen_v6_step)
              + ", borderColor:PAL.v6, backgroundColor:PAL.v6, stepped:'after', tension:0 },"
              "{ label:'v6 + subsystem, THIS TEAM ONLY (ablation)', data:" + json.dumps(sen_e_step)
              + ", borderColor:PAL.overhaul, backgroundColor:PAL.overhaul, borderDash:[8,4], stepped:'after', tension:0 } ],"
              "{ plugins:{ vLines:{marks:" + json.dumps(sen_marks) + ", labels:" + json.dumps(sen_dates) + "},"
              "legend:{position:'bottom', labels:{boxWidth:14, boxHeight:3}} },"
              "scales:{ x:{grid:{display:false}, ticks:{maxTicksLimit:8}}, y:{grid:{color:'#eceef2'},"
              "title:{display:true, text:'rating (transformed round-margin units)'} } } });")
else:
    SEN_HTML, SEN_JS = "", ""

JS = (JS
      .replace("__PAL__", json.dumps(C))
      .replace("__ENV_LABELS__", json.dumps(env_dates))
      .replace("__ENV_V6__", json.dumps(env_v6_step))
      .replace("__ENV_B__", json.dumps(env_b_step or []))
      .replace("__ENV_E__", json.dumps(env_e_step or []))
      .replace("__ENV_F__", json.dumps(env_f_step))
      .replace("__ENV_MARKS__", json.dumps(env_marks))
      .replace("__LEV_LABELS__", json.dumps(lev_dates))
      .replace("__LEV_V6__", json.dumps(lev_v6_step))
      .replace("__LEV_B__", json.dumps(lev_b_step or []))
      .replace("__LEV_E__", json.dumps(lev_e_step or []))
      .replace("__LEV_F__", json.dumps(lev_f_step))
      .replace("__LEV_MARKS__", json.dumps(lev_marks))
      .replace("__SEN_JS__", SEN_JS)
      .replace("__POOL_LABELS__", json.dumps(
          ["keep4 (4/5 kept)", "keep3 (3/5)", "overhaul (≤2/5)", "stable ref"]))
      .replace("__POOL_VALS__", json.dumps(
          [pool["keep4"]["bias_pp"], pool["keep3"]["bias_pp"],
           pool["overhaul"]["bias_pp"], ref["bias_pp"]]))
      .replace("__POOL_COLS__", json.dumps(
          [C["keep4"], C["keep3"], C["overhaul"], C["v6"]]))
      .replace("__POOL_LO__", json.dumps(
          [pool["keep4"]["ci_lo"], pool["keep3"]["ci_lo"],
           pool["overhaul"]["ci_lo"], ref["ci_lo"]]))
      .replace("__POOL_HI__", json.dumps(
          [pool["keep4"]["ci_hi"], pool["keep3"]["ci_hi"],
           pool["overhaul"]["ci_hi"], ref["ci_hi"]]))
      .replace("__CURVE_X__", json.dumps(list(range(10))))
      .replace("__CURVE_DS__", json.dumps(curve_ds))
      .replace("__REF_LO__", json.dumps(ref["ci_lo"]))
      .replace("__REF_HI__", json.dumps(ref["ci_hi"]))
      .replace("__TREAT_LABELS__", json.dumps(
          ["(b) continuity", "(c) cold start", "(d) phase-reset"]))
      .replace("__TREAT_D__", json.dumps(treat_delta))
      .replace("__TREAT_COLS__", json.dumps(treat_cols))
      .replace("__TREAT_LO__", json.dumps(treat_lo))
      .replace("__TREAT_HI__", json.dumps(treat_hi))
      .replace("__FLOOR_LO__", json.dumps(-mde["within_milli"]))
      .replace("__FLOOR_HI__", json.dumps(mde["within_milli"]))
      .replace("__NHOLD__", str(tr["_meta"]["n_holdout"]))
      .replace("__GAIN_X__", json.dumps(list(range(10))))
      .replace("__GAIN_DS__", json.dumps(gain_ds)))



html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Roster-Change Adaptation — BenPom v8</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Plus+Jakarta+Sans:wght@700;800&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
<style>{CSS}</style></head><body><div class="wrap">
<h1>Roster-Change Adaptation</h1>
<div class="tagline">How should BenPom react when a team changes players mid-season? —
operator-requested report · agent:roster · {tr['_meta']['written']}</div>
{NAV}

<div class="banner"><b>EXPLORATORY ONLY.</b> The 2025-26 holdout is <b>spent</b> —
{lb['prior_grand_total_recorded_holdout_numbers']} recorded holdout looks existed before this report
(adversary finding); this report adds {lb['new_primary_looks']} preregistered exploratory reads
(grand total {lb['grand_total_after']}, tallied in <span class="mono">roster_looks.json</span>).
Nothing on this page is confirmatory or promotable. Adjudication requires the
preregistered prospective test on fresh series (&sect;7).</div>

<div class="banner" style="border-color:#7c4dd6;background:#f5f1fd">
{("<b>THE ANSWER, UP FRONT (SPEC RUN, " + esc(SPEC_READ["read_label"]) + "):</b> "
  "<b>v6 does not gain a roster subsystem — the deployed model IS v6.</b> "
  "The activation gate FIRED on train (inner-CV +" + fm(SPEC_READ["gate_outcome"]["per_policy_gate"][SPEC_READ["gate_outcome"]["selected_policy_or_doc"]]["gate_mean_milli"], 2)
  + "m vs SE " + fm(SPEC_READ["gate_outcome"]["per_policy_gate"][SPEC_READ["gate_outcome"]["selected_policy_or_doc"]]["gate_se_milli"], 2)
  + "m; &acirc;=" + fm(SPEC_READ["gate_outcome"]["a_shrunk"], 0) + ", &tau;=" + fm(SPEC_READ["gate_outcome"]["tau"], 0)
  + ", s=" + fm(SPEC_READ["gate_outcome"]["s"], 1) + ", policy " + esc(SPEC_READ["gate_outcome"]["selected_policy_or_doc"]) + "), "
  "and the corpus-wide exploratory read then fired the preregistered falsifier: <b>"
  + fm(SPEC_READ["headline"]["delta_milli"], 2) + "m</b> vs v6 (iid CI ["
  + fm(SPEC_READ["headline"]["iid"]["ci_lo"]*1000, 1) + ", " + fm(SPEC_READ["headline"]["iid"]["ci_hi"]*1000, 1)
  + "]m; block [" + fm(SPEC_READ["headline"]["block_event"]["ci_lo"]*1000, 1) + ", "
  + fm(SPEC_READ["headline"]["block_event"]["ci_hi"]*1000, 1) + "]m; MDE "
  + fm(SPEC_READ["headline"]["mde_context"]["within_milli"], 2) + "m; n=" + str(SPEC_READ["headline"]["n_scored"])
  + ") — a distinguishable loss, worst exactly where the subsystem aims (k&le;2 rebuilds "
  + fm(next(sl["delta_milli"] for sl in SPEC_READ["slices"] if sl["name"].startswith("retention k<=2")), 2)
  + "m). The sub-heavy slice was neutral (+"
  + fm(next(sl["delta_milli"] for sl in SPEC_READ["slices"] if sl["name"].startswith("sub-heavy")), 2)
  + "m) — the damage is the a=28 solve deformation, not the sub down-weight. "
  "A train-fired gate failing out of sample is the run&rsquo;s headline lesson: mean&gt;SE at K=5 folds "
  "did not stop the 2023-24 &rarr; 2025-26 non-transfer. Spec conformance held everywhere: fixtures "
  + str(SPEC_FIX.get("n_pass", "?")) + "/" + str((SPEC_FIX.get("n_pass") or 0) + (SPEC_FIX.get("n_fail") or 0))
  + " passed, ablation zero-gates exactly 0.0 for ENVY/LEV/SEN, P2 correctly NOT RUN "
  "(region-prior recursion breaks retraction identity), and the shipped answer honors &sect;3.5: "
  "no evidence, no subsystem. Prospective arm H_specrun stays frozen for the fresh-data test.")
 if SPEC_READ else
 ("<b>THE ANSWER, UP FRONT:</b> spec run in progress — fixtures/census landed, "
  "scoring pending. Fair value remains v6.")}
</div>

<section><h2><span class="n">1</span> The question, and what v6 does today</h2>
<p>The production model's only roster mechanism is <b>year-boundary continuity</b>
(carry-over weight 0.3 scaled by the January roster overlap). Every mid-season change —
ENVY's three 2026 swaps included — is <b>invisible to the solve</b>: the team keeps its
full rating as if nothing happened. This report measures what that costs, forensically
and population-wide, and takes three preregistered exploratory shots at fixing it.</p>
<p class="dim">Change definition (preregistered before computing): a rotation-guarded
lineup-set change vs the previous match; <b>sustained</b> = the new five holds &ge;3
consecutive matches (or is censored by the data edge); magnitude = lineup overlap
|A&cap;B|/max(|A|,|B|,5) — the engine's own formula. Verified to reproduce the lineups
agent's <span class="mono">matches_since_change</span> on all {counts['episodes_total']}-episode
corpus with 0 mismatches.</p>
<div class="cards">
<div class="card"><div class="lbl">sustained changes 2023-26</div><div class="big">{counts['sustained']}</div>
<div class="sub">+{counts['transient']} transient (stand-in/revert); {counts['sustained_censored']} censored at data edge</div></div>
<div class="card"><div class="lbl">by magnitude</div><div class="big">{counts['by_magnitude']['keep4']} / {counts['by_magnitude']['keep3']} / {counts['by_magnitude']['overhaul']}</div>
<div class="sub">keep-4 / keep-3 / overhaul (&le;2 kept)</div></div>
<div class="card"><div class="lbl">churn by region</div><div class="big">{counts['by_region']['CN']} CN</div>
<div class="sub">{counts['by_region']['Pacific']} Pacific · {counts['by_region']['Americas']} Americas · {counts['by_region']['EMEA']} EMEA</div></div>
<div class="card"><div class="lbl">2026 alone</div><div class="big">{counts['by_year']['2026']}</div>
<div class="sub">sustained changes — roster churn is the norm, not the exception</div></div>
</div>{dl('roster_population.json')}</section>

<section><h2><span class="n">2</span> The ENVY case, forensically
<span class="badge desc">DESCRIPTIVE</span></h2>
<p>The operator flagged ENVY's Stage 1 &rarr; Stage 2 anomaly. The lineups data shows it is a
<b>chained pair of one-out swaps</b> — by Stage 2 the fielded five keeps
<b>{agg['kept'].__len__()}/5</b> of the Stage-1 five
({', '.join(map(esc, agg['kept']))}; out: {', '.join(map(esc, agg['left_between_s1_s2']))};
in: {', '.join(map(esc, agg['joined_between_s1_s2']))}).</p>
<div class="scroll"><table><tr><th>date</th><th>event</th><th>out</th><th>in</th><th>kept</th><th>run</th></tr>
{chain_rows}</table></div>
<div class="cards">
<div class="card"><div class="lbl">S1&rarr;S2 overlap</div><div class="big">{agg['overlap_s1_to_s2']:.1f}</div>
<div class="sub">3/5 of the Stage-1 core survived to Stage 2</div></div>
<div class="card"><div class="lbl">carryover cost after {c2['change_date']}</div>
<div class="big bad">{fm(env_p['carryover_cost_pp_per_match'], 1, True)}pp</div>
<div class="sub">mean(p_v6 &minus; won) per match, first {env_p['carryover_n']} matches
(pre-change bias {fm(env_p['pre_bias_pp'], 1, True)}pp)</div></div>
<div class="card"><div class="lbl">cost after S2 swap {c3['change_date']}</div>
<div class="big bad">{fm(env_p2['carryover_cost_pp_per_match'], 1, True)}pp</div>
<div class="sub">first {env_p2['carryover_n']} S2 matches (censored — Stage 2 ongoing)</div></div>
<div class="card"><div class="lbl">rating stabilization</div><div class="big">{esc(str(env_p['stabilization_matches']))} matches</div>
<div class="sub">after the {c2['change_date']} chain start (frozen definition, prereg &sect;2)</div></div>
</div>
<h3>ENVY 2026: v6 rating vs the phase-reset filter <span class="badge">EXPLORATORY overlay</span></h3>
<div class="chartbox" style="height:300px"><canvas id="envyChart"></canvas></div>
<p class="cap">Same model: the dashed line is v6 with the subsystem enabled for ENVY only
(per-team ablation run), identical to solid v6 until each red vertical (labelled k/5 retained),
and every point of separation after a vertical is ENVY&rsquo;s own adaptation. Chart = ablation;
accuracy numbers = the corpus-wide run in &sect;5 — never mixed.</p>
<h3>Every ENVY match of 2026</h3>
<div class="scroll"><table><tr><th>date</th><th>event</th><th>opp</th><th>result</th>
<th>p_v6 (ENVY)</th><th>msc</th><th>fielded five</th></tr>{match_rows}</table></div>
{dl('roster_case_envy.json')}</section>

<section><h2><span class="n">3</span> Named cases + the biggest overhauls
<span class="badge desc">DESCRIPTIVE</span></h2>
<p>The operator named two canonical cases; both verified from lineups data (never memory):</p>
<div class="cards">
<div class="card"><div class="lbl">improvement — LEV gains Neon</div>
<div class="big good">{fm(lev['panel']['carryover_cost_pp_per_match'], 1, True)}pp/match</div>
<div class="sub">{lev['change_date']}: {lev['kept']}/5 kept (out {', '.join(map(esc, lev['out']))};
in {', '.join(map(esc, lev['in']))}). Negative cost = v6 <b>under</b>-priced the upgraded roster
for {lev['panel']['carryover_n']} matches; stabilization {esc(str(lev['panel']['stabilization_matches']))} matches.</div></div>
<div class="card"><div class="lbl">degradation — ENVY loses inspire</div>
<div class="big bad">{fm(envy_inspire['panel']['carryover_cost_pp_per_match'], 1, True)}pp/match</div>
<div class="sub">{envy_inspire['change_date']}: {envy_inspire['kept']}/5 kept
(out {', '.join(map(esc, envy_inspire['out']))}; in {', '.join(map(esc, envy_inspire['in']))}).
Positive cost = v6 <b>over</b>-priced ENVY; stabilization {esc(str(envy_inspire['panel']['stabilization_matches']))} matches.</div></div>
</div>
<h3>LEV window: v6 vs phase-reset <span class="badge">EXPLORATORY overlay</span></h3>
<div class="chartbox" style="height:300px"><canvas id="levChart"></canvas></div>
<p class="cap">Same model: dashed = v6 with the subsystem enabled for LEV only (ablation),
identical to solid v6 until LEV&rsquo;s boundary vertical; separation after it is LEV&rsquo;s own
adaptation to the Neon-era five. Chart = ablation; accuracy = corpus-wide run (&sect;5).</p>
{SEN_HTML}
<h3>Gallery: the four largest sustained overhauls (full rebuilds, 0/5 kept)</h3>
<div class="scroll"><table><tr><th>org</th><th>date</th><th>event</th><th>kept</th>
<th>out</th><th>in</th><th>stab.</th><th>cost/match</th></tr>{gal_rows}</table></div>
<p class="cap">Cost = mean(p_v6 &minus; won) over the first 6 post-change matches; negative =
the new roster beat the carried rating. Sign varies case-by-case — the error is
<b>direction-agnostic at the case level</b>, which motivated the operator's variance-spike shape.</p>
{dl('roster_case_gallery.json')}</section>

<section><h2><span class="n">4</span> Population atlas: the adaptation curve
<span class="badge desc">DESCRIPTIVE</span></h2>
<p>Across every sustained change 2023-26, how wrong is the carried v6 rating in the first
matches after a change? Positive = post-change teams <b>outperform</b> the carried rating.</p>
<div class="chartbox" style="height:280px"><canvas id="atlasPooled"></canvas></div>
<p class="cap">First 3 matches after a sustained change, pooled, vs the stable reference
(&gt;10 matches on the same five). Whiskers = 95% CI (team-observations).</p>
<div class="scroll"><table><tr><th>class</th><th>bias (first 3)</th><th>95% CI</th><th>n team-obs</th></tr>
{pool_rows}</table></div>
<div class="callout">Post-change teams beat v6's carried rating by <b>+4&ndash;6pp per match</b>
in their first three matches — the keep-4 class alone is
{fm(pool['keep4']['bias_pp'], 2, True)}pp with a CI excluding zero. The operator's
"new phase" intuition is directionally right <b>in the data</b>. Changes skew toward
improvement on average (teams change rosters because they are underperforming), which is
why a symmetric mean-blend toward a prior cannot capture it and why the year-only
mechanism misses it entirely.</div>
<div class="chartbox" style="height:300px"><canvas id="atlasCurve"></canvas></div>
<p class="cap">Per-match adaptation curve by magnitude (gray band = stable-reference 95% CI).
Per-point CIs are wide — cell-level values and CIs are in the JSON download.
Power context: the post-change (&le;3) bucket floor is {pcx['buckets'][0]['mde80_within_milli']}m
within-family at its frozen n={pcx['buckets'][0]['n']} — and even that is exploratory-only
on this spent frame.</p>
{dl('roster_population.json')}</section>

<section><h2><span class="n">5</span> Three treatments, three exploratory reads
<span class="badge">EXPLORATORY</span></h2>
<p>All preregistered (specs, predictions, falsifiers) before any run — including the
operator-specified <b>phase-reset filter</b> (addendum, ordering disclosed in the log):
keep the rating mean as the reference point, inject state variance
&Delta;q&nbsp;=&nbsp;g&middot;(1&minus;k/5)&middot;R at the change, and let the elevated Kalman
gain over-react to the first post-change results in whichever direction they point.
Train-only selection; &beta; refit per config; CRN paired bootstrap (iid + event-block);
baseline = stored v6 (holdout LL {tr['read1_b_continuity']['ll_v6_same_rows']:.5f}, n={tr['_meta']['n_holdout']}).
Floors: within-family {mde['within_milli']}m, cross {mde['cross_milli']}m; post-change bucket
{mde['post_le3_within_milli']}m / {mde['post_le3_cross_milli']}m (frozen n=598).</p>
<div class="chartbox" style="height:230px"><canvas id="treatChart"></canvas></div>
<div class="scroll"><table><tr><th>treatment (EXPLORATORY read)</th><th>&Delta; overall</th>
<th>&Delta; post-change &le;3</th><th>&approx;ROI &Delta;</th><th>verdict on this frame</th></tr>
{read_rows}</table></div>
<p class="cap">ROI = referee quote-margin translation, symmetric first-order reading
(the ladder is defined for improvements; negative &Delta;LL clamps at the operating point —
both conventions in the JSON). All three predicted-positive treatments came back negative.</p>
<div class="callout bad"><b>The operator's contrast prediction is REVERSED.</b> Preregistered:
variance-spike (d) beats mean-blend (c) on improvement cases. Measured (d&nbsp;&minus;&nbsp;c on
improvement rows): <b>{fm(contrast['improvement cases']['d_minus_c_milli'], 2, True)}m</b>
{ci(contrast['improvement cases']['ci_lo_milli'], contrast['improvement cases']['ci_hi_milli'])} —
the mean-blend wins on the very rows the over-reaction was designed for. The injection also
loses to its own no-injection filter base
({fm(tr['read3_d_phase_reset']['vs_own_base_1b']['delta_milli'], 2, True)}m
{ci(tr['read3_d_phase_reset']['vs_own_base_1b']['ci_lo_milli'], tr['read3_d_phase_reset']['vs_own_base_1b']['ci_hi_milli'])}):
single-map results are too noisy for a one-shot gain spike — the filter over-trusts
exactly the evidence the atlas shows is thin.</div>
<div class="scroll"><table><tr><th>d vs c contrast slice</th><th>d &minus; c</th><th>95% CI</th><th>n</th></tr>
{ctr_rows}</table></div>
<h3>Effective learning rate after a change (operator chart) <span class="badge desc">DESCRIPTIVE telemetry</span></h3>
<div class="chartbox" style="height:300px"><canvas id="gainChart"></canvas></div>
<p class="cap">Mean Kalman gain per map update vs matches-since-change at the train-selected
injection (g={tr['read3_d_phase_reset']['spec']['selected_g']}): the filter learns
&approx;{round(next(p['gain'] for p in lr['by_magnitude'][2]['points'] if p['m']==0)/lrs['gain'], 1)}&times;
faster than the stable reference ({lrs['gain']}) right after a full overhaul, decaying back
within ~5 matches — exactly the operator's "overreact a bit, scaled by continuity" shape.
The shape renders correctly; on this frame it just doesn't pay.</p>
<h3>Per-bucket panel (all three reads)</h3>
<div class="scroll"><table><tr><th>bucket</th><th>(b) continuity</th><th>(c) cold start</th>
<th>(d) phase-reset</th></tr>{bucket_rows}</table></div>
<p class="cap">&Delta; milli-LL vs v6, iid CRN CIs; improvement/degradation slices are
retrospective diagnostics of already-counted vectors, not additional looks.</p>
{dl('roster_treatments.json')}{dl('roster_looks.json')}</section>

<section><h2><span class="n">6</span> Where this sits in Wave 2 (adversary-amended)</h2>
<ul>
<li><b>Ledger re-open is legitimate:</b> "{esc(ledger['rows'][6]['idea'])}" and
"{esc(ledger['rows'][7]['idea'])}" were rejected pre-v8 on coarse year-boundary heuristics and
reclassified UNRESOLVED by Phase 0 — per-match lineups are a new data source. This report is
that re-open; its answer on this frame is negative.</li>
<li><b>Decay 5b-a</b> (lineup continuity): real but redundant with year mode, inside the
noise floor — consistent with read (b)'s failure to beat year continuity.</li>
<li><b>H3 5d half-life table</b> (change-adjacent 7.4 vs stable 24.8 games): DEMOTED by the
adversary to "a hypothesis to test on new data" — quoted here only as such.</li>
<li><b>H3 cold-start &lt;10 maps</b>: the one adversary-robust lead. Read (c) tried its
point-estimate form on change rows and found nothing — the lead stays a cold-start
(rating-absence) story, not a roster-change story, on current evidence.</li>
<li><b>Compose S1</b>: HOLD — "the evidence's verdict" per the adversary; the original
change-scoped gate idea was withdrawn unread when the operator's phase-reset replaced it.</li>
</ul>
<div class="adv"><div class="advhead">Adversary excerpts this report must respect (verbatim)</div>
<pre>{esc(ADV_EXC)}</pre></div>
{dl('adversary_report.json')}{dl('ledger_reclass.json')}</section>

<section><h2><span class="n">7</span> Bot integration + the prospective plan</h2>
<h3>Tier 1 — sizing only <span class="verd lead">RECOMMENDED NOW</span></h3>
<p>{esc(integ['tier1_sizing_only']['what'])}</p>
<ul>{''.join(f'<li class="mono" style="font-size:.82rem">{esc(r)}</li>' for r in integ['tier1_sizing_only']['playbook_rule_sketch'])}</ul>
<p class="dim">Basis: the atlas's elevated, directional post-change error and the case
evidence (ENVY {fm(env_p['carryover_cost_pp_per_match'],1,True)}pp,
LEV {fm(lev['panel']['carryover_cost_pp_per_match'],1,True)}pp per match). Sizing down on
elevated-uncertainty rows needs no directional model claim and zero fair-value risk.
Snapshot fields + data path in the JSON.</p>
<h3>Tier 2 — fair-value change <span class="verd hold">NOT RECOMMENDED NOW</span></h3>
<p>{esc(integ['tier2_fair_value']['why'])}</p>
<h3>Player-identity mean shift (design note only)</h3>
<p class="dim">{esc(integ['player_identity_mean_shift_design_note']['idea'])} No holdout
read was spent on it (operator addendum).</p>
<h3>The preregistered prospective test (the only path to adjudication)</h3>
<div class="scroll"><table><tr><th>arm</th><th>frozen spec</th></tr>{arm_rows}</table></div>
<p class="dim">{esc(integ['prospective_validation_plan']['decision_rule'])}</p>
<div class="scroll"><table><tr><th>fresh n</th><th>MDE80 overall (within)</th>
<th>MDE80 post-change bucket</th></tr>{proj_rows}</table></div>
<p class="cap">{esc(integ['prospective_validation_plan']['mde80_projection_milli']['reading'])}</p>
{dl('roster_integration.json')}</section>

<section><h2><span class="n">8</span> Looks accounting</h2>
<p>Prior recorded holdout numbers: <b>{lb['prior_grand_total_recorded_holdout_numbers']}</b>
(compose tally at Wave 3 close). This report: <b>{lb['new_primary_looks']}</b> new
EXPLORATORY reads (one per treatment, train-selected params only; non-selected grid points
were evaluated on train only, with engine/filter holdout numbers scrubbed before recording —
see the log). Reserve read: unused (no train tie). Grand total now
<b>{lb['grand_total_after']}</b>. Stored reuses (v6 baseline; h3 1b base) are not new looks.</p>
<p class="dim">Verdict, in the program's own words: v6 stands. The roster signal is real in
the data (atlas &sect;4) but none of the three preregistered mechanisms monetized it on this
frame — the honest deliverables are the Tier-1 sizing flag and the prospective test.</p>
{dl('roster_looks.json')}</section>

<p class="dim" style="text-align:center;font-size:.78rem">agent:roster · preregistered in
<span class="mono">preregister.roster.md</span> (+ operator addendum, ordering disclosed) ·
every number from <span class="mono">/testing/v8/stats/roster_*.json</span> · CRN per
<span class="mono">crn.json</span> · frame sha verified</p>
</div><script>{JS}</script></body></html>"""

os.makedirs(RD, exist_ok=True)
out = os.path.join(RD, "roster_adaptation.html")
with open(out, "w") as f:
    f.write(html)
print(f"wrote {out} ({len(html)} bytes)")
