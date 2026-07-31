"""Testing Lab — password-gated workbench for the BenPom probability-accuracy
project (model audit, backtests, Kalshi benchmarking).

Isolated Flask blueprint, registered at /testing/ (BobosHome.py). Access is
gated by a password ("TenZ") checked server-side; a salted-hash cookie keeps
the session for 30 days. The lab page currently hosts the master plan; as
experiment phases run, their result reports get rendered here too.

Routes:
  GET  /testing/       login form, or the lab if the auth cookie is valid
  POST /testing/auth   password check -> sets cookie, redirects back
"""
import glob
import hashlib
import json
import os
import re

from flask import Blueprint, Response, abort, redirect, request

testing_bp = Blueprint("testing_bp", __name__)

_ROOT = os.path.dirname(os.path.abspath(__file__))
_LAB_OUT = os.path.join(_ROOT, "testing_lab", "out")
_REPORTS_DIR = os.path.join(_LAB_OUT, "reports")
_STATUS_PATH = os.path.join(_LAB_OUT, "status.json")

_PASSWORD = "TenZ"
_COOKIE = "tl_auth"
# Bumping the salt suffix (v1 -> v2 -> …) invalidates every tl_auth cookie
# already in a browser, forcing a fresh login everywhere without changing the
# password. That's the "log me out of all devices" lever. Reset 2026-07-28.
_TOKEN = hashlib.sha256(f"{_PASSWORD}|bobo-testing-lab-v2".encode()).hexdigest()
_COOKIE_MAX_AGE = 30 * 24 * 3600


def _authed() -> bool:
    return request.cookies.get(_COOKIE, "") == _TOKEN


@testing_bp.route("/")
def lab():
    if not _authed():
        return Response(_login_html(err="err" in request.args), mimetype="text/html")
    return Response(_render_lab(), mimetype="text/html")


@testing_bp.route("/playbook")
def playbook():
    if not _authed():
        return redirect("/testing/")
    path = os.path.join(_REPORTS_DIR, "when_to_deploy.html")
    if not os.path.exists(path):
        abort(404)
    with open(path) as f:
        return Response(f.read(), mimetype="text/html")


@testing_bp.route("/report/<name>")
def report(name):
    if not _authed():
        return redirect("/testing/")
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", name):
        abort(404)
    path = os.path.join(_REPORTS_DIR, f"{name}.html")
    if not os.path.exists(path):
        abort(404)
    with open(path) as f:
        return Response(f.read(), mimetype="text/html")


# v8 Lab chart-data downloads: every figure on /testing/report/v8_lab reads
# from testing_lab/v8/stats/*.json and links back here (same auth gate).
_V8_STATS = os.path.join(_ROOT, "testing_lab", "v8", "stats")


@testing_bp.route("/v8/stats/<name>")
def v8_stats(name):
    if not _authed():
        return redirect("/testing/")
    if not re.fullmatch(r"[A-Za-z0-9_\-]+\.json", name):
        abort(404)
    path = os.path.join(_V8_STATS, name)
    if not os.path.exists(path):
        abort(404)
    with open(path) as f:
        return Response(f.read(), mimetype="application/json",
                        headers={"Content-Disposition": f"attachment; filename={name}"})


# v9 Lab chart-data downloads: every figure on /testing/report/v9_lab reads
# from testing_lab/v9/stats/*.json and links back here (same auth gate,
# same name regex as the v8 route above).
_V9_STATS = os.path.join(_ROOT, "testing_lab", "v9", "stats")


@testing_bp.route("/v9/stats/<name>")
def v9_stats(name):
    if not _authed():
        return redirect("/testing/")
    if not re.fullmatch(r"[A-Za-z0-9_\-]+\.json", name):
        abort(404)
    path = os.path.join(_V9_STATS, name)
    if not os.path.exists(path):
        abort(404)
    with open(path) as f:
        return Response(f.read(), mimetype="application/json",
                        headers={"Content-Disposition": f"attachment; filename={name}"})


def _render_lab():
    """Inject dynamic report links + status entries into the plan page."""
    html = _LAB_HTML
    # reports list
    items = []
    for p in sorted(glob.glob(os.path.join(_REPORTS_DIR, "*.html"))):
        name = os.path.splitext(os.path.basename(p))[0]
        title = name.replace("_", " ").title()
        try:
            with open(p) as f:
                head = f.read(2000)
            m = re.search(r"<title>([^<]+)</title>", head)
            if m:
                title = m.group(1)
        except Exception:
            pass
        items.append(f'<a class="repl" href="/testing/report/{name}">{title} &rarr;</a>')
    reports_html = ("".join(items) if items
                    else '<span class="dim" style="font-size:.85rem">No reports yet.</span>')
    html = html.replace("<!--REPORTS-->", reports_html)
    # status entries (newest first), prepended above the static ones
    rows = []
    try:
        with open(_STATUS_PATH) as f:
            for e in reversed(json.load(f)):
                rows.append('<div class="status-row"><div class="status-date">%s</div>'
                            '<div>%s</div></div>' % (e.get("date", ""), e.get("text", "")))
    except Exception:
        pass
    return html.replace("<!--STATUS-->", "".join(rows))


@testing_bp.route("/auth", methods=["POST"])
def auth():
    if request.form.get("password", "") == _PASSWORD:
        resp = redirect("/testing/")
        resp.set_cookie(_COOKIE, _TOKEN, max_age=_COOKIE_MAX_AGE,
                        httponly=True, samesite="Lax")
        return resp
    return redirect("/testing/?err=1")


# ────────────────────────────── login page ──────────────────────────────────
def _login_html(err: bool = False) -> str:
    msg = ('<div class="err">Wrong password.</div>' if err else "")
    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex">
<title>Testing — Bobo gg</title>
<link rel="icon" href="/logo.svg">
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@700;800&family=DM+Sans:wght@400;500;700&display=swap" rel="stylesheet">
<style>
  * {{ box-sizing: border-box; margin: 0; }}
  body {{ font-family:'DM Sans',system-ui,sans-serif; background:#faf9fc; color:#16121d;
         min-height:100vh; display:flex; align-items:center; justify-content:center; }}
  .card {{ background:#fff; border:1px solid #eceef2; border-radius:20px;
           box-shadow:0 14px 44px #0000000f; padding:38px 40px; width:min(380px,92vw);
           text-align:center; }}
  .card img {{ height:40px; margin-bottom:14px; }}
  h1 {{ font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.25rem;
        margin-bottom:4px; }}
  .sub {{ color:#6b6478; font-size:.85rem; margin-bottom:22px; }}
  /* Scoped to .card on purpose: the shared Alpha nav is injected into every
     page and builds its dropdown toggle as a <button>, so a bare `button`
     rule here leaks onto it — `width:100%` blew the "Historical VCT Tools"
     toggle up to ~1760px and pushed the last three nav links off the bar. */
  .card input[type=password] {{ width:100%; padding:11px 14px; border:1px solid #dcd6e6;
        border-radius:11px; font:inherit; font-size:.95rem; outline:none;
        transition:border .15s, box-shadow .15s; }}
  .card input[type=password]:focus {{ border-color:#7c4dd6; box-shadow:0 0 0 3px #7c4dd622; }}
  .card button {{ width:100%; margin-top:12px; padding:11px; border:0; border-radius:11px;
        background:#7c4dd6; color:#fff; font:inherit; font-weight:700; font-size:.95rem;
        cursor:pointer; transition:background .15s; }}
  .card button:hover {{ background:#6a3fc0; }}
  .err {{ color:#c0392b; font-size:.82rem; font-weight:700; margin-top:12px; }}
</style></head>
<body>
  <form class="card" method="POST" action="/testing/auth">
    <img src="/logo.svg" alt="Bobo">
    <h1>Testing Lab</h1>
    <div class="sub">This area is private. Enter the password to view.</div>
    <input type="password" name="password" placeholder="Password" autofocus autocomplete="off">
    <button type="submit">Enter</button>
    {msg}
  </form>
</body></html>"""


# ─────────────────────────────── lab page ───────────────────────────────────
_LAB_HTML = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex">
<title>Testing Lab — BenPom Accuracy Project</title>
<link rel="icon" href="/logo.svg">
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@700;800&family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
  * { box-sizing: border-box; margin: 0; }
  :root { --ink:#16121d; --dim:#6b6478; --line:#eceef2; --acc:#7c4dd6; --accbg:#f3eefb;
          --warn:#b3541e; --warnbg:#fdf3ec; --good:#1e7a4f; --goodbg:#ecf8f1; }
  body { font-family:'DM Sans',system-ui,sans-serif; background:#faf9fc; color:var(--ink);
         line-height:1.55; padding:34px 18px 90px; }
  .wrap { max-width: 880px; margin: 0 auto; }
  h1 { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.7rem;
       text-align:center; margin:6px 0 2px; }
  .tagline { text-align:center; color:var(--dim); font-size:.92rem; margin-bottom:30px; }
  .badge { display:inline-block; background:var(--accbg); color:var(--acc); font-weight:700;
           font-size:.72rem; border-radius:999px; padding:3px 11px; letter-spacing:.4px;
           text-transform:uppercase; }
  section { background:#fff; border:1px solid var(--line); border-radius:18px;
            padding:26px 30px; margin-bottom:18px; box-shadow:0 3px 14px #00000008; }
  h2 { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.08rem;
       margin-bottom:12px; display:flex; align-items:center; gap:10px; }
  h2 .n { background:var(--acc); color:#fff; border-radius:8px; font-size:.78rem;
          width:24px; height:24px; display:inline-flex; align-items:center;
          justify-content:center; flex-shrink:0; }
  h3 { font-size:.92rem; font-weight:700; margin:16px 0 6px; }
  p { margin: 8px 0; font-size:.9rem; }
  ul, ol { margin:8px 0 8px 22px; font-size:.9rem; }
  li { margin: 5px 0; }
  code { font-family:'JetBrains Mono',monospace; font-size:.8em; background:#f4f2f8;
         border:1px solid var(--line); border-radius:6px; padding:1px 6px; }
  .dim { color: var(--dim); }
  table { width:100%; border-collapse:collapse; font-size:.84rem; margin:10px 0; }
  th { text-align:left; color:var(--dim); font-size:.72rem; text-transform:uppercase;
       letter-spacing:.6px; padding:7px 10px; border-bottom:2px solid var(--line); }
  td { padding:8px 10px; border-bottom:1px solid var(--line); vertical-align:top; }
  tr:last-child td { border-bottom:0; }
  .sev { font-weight:700; border-radius:999px; padding:2px 10px; font-size:.72rem;
         white-space:nowrap; }
  .sev.hi   { background:#fbeaea; color:#c0392b; }
  .sev.med  { background:var(--warnbg); color:var(--warn); }
  .sev.lo   { background:#f1f0f4; color:var(--dim); }
  .tag { display:inline-block; background:var(--accbg); color:var(--acc); border-radius:6px;
         font-size:.72rem; font-weight:700; padding:1px 8px; margin-right:4px; }
  .callout { border-left:4px solid var(--acc); background:var(--accbg); border-radius:0 12px 12px 0;
             padding:12px 16px; margin:12px 0; font-size:.88rem; }
  .callout.warn { border-color:var(--warn); background:var(--warnbg); }
  .callout.good { border-color:var(--good); background:var(--goodbg); }
  .phase { display:flex; gap:14px; margin:14px 0; }
  .phase .pnum { flex-shrink:0; width:34px; height:34px; border-radius:11px; background:var(--accbg);
                 color:var(--acc); font-weight:800; display:flex; align-items:center;
                 justify-content:center; font-family:'Plus Jakarta Sans',sans-serif; }
  .phase .pbody { flex:1; }
  .phase .ptitle { font-weight:700; font-size:.92rem; }
  .phase .pdesc { color:var(--dim); font-size:.85rem; margin-top:2px; }
  .status-row { display:flex; gap:12px; padding:10px 0; border-bottom:1px solid var(--line);
                font-size:.87rem; }
  .status-row:last-child { border-bottom:0; }
  .status-date { flex-shrink:0; font-family:'JetBrains Mono',monospace; font-size:.76rem;
                 color:var(--dim); padding-top:2px; }
  .repl { display:block; padding:12px 16px; border:1px solid var(--line); border-radius:12px;
          font-weight:700; font-size:.9rem; color:var(--acc); text-decoration:none;
          transition:background .15s, border-color .15s; }
  .repl:hover { background:var(--accbg); border-color:var(--acc); }
  .labtabs { display:flex; justify-content:center; gap:6px; margin:0 0 26px; }
  .labtabs a { font-size:.8rem; font-weight:700; color:var(--dim); text-decoration:none;
    padding:7px 15px; border-radius:999px; border:1px solid var(--line); background:#fff; }
  .labtabs a:hover { color:var(--ink); background:var(--accbg); }
  .labtabs a.on { color:#fff; background:var(--acc); border-color:var(--acc); }
  @media (max-width:640px){ section{padding:20px 16px;} body{padding:22px 10px 60px;} }
</style></head>
<body>
<div class="wrap">
  <div style="text-align:center;margin-bottom:10px;"><span class="badge">Private · Phase 0</span></div>
  <h1>BenPom Accuracy Project</h1>
  <div class="tagline">Systematic audit &amp; optimization of the series-probability pipeline.
    Real money quotes on these numbers (VCTMM on Kalshi) — the bar is: measurably better, proven out-of-sample, no regressions.</div>

  <div class="labtabs">
    <a href="/testing/" class="on">Testing Lab</a>
    <a href="/testing/report/state_of_benpom">State of BenPom</a>
    <a href="/testing/playbook">Deployment Playbook</a>
    <a href="/testing/report/favorites_lab">Favorites Lab</a>
<a href="/testing/report/final_model">Final Model</a>
<a href="/testing/report/v7_lab">v7 Lab</a>
<a href="/testing/report/v8_lab">v8 Lab</a><a href="/testing/report/roster_adaptation">Roster</a><a href="/testing/report/v9_lab">v9 Lab</a>
  </div>

  <section>
    <h2><span class="n">1</span>Mission &amp; ground rules</h2>
    <p>Make BenPom's predicted win probabilities more accurate — better calibrated and sharper —
       using every data source available: the full VLR match corpus (2023–2026), Kalshi's settled
       market history, and the bot's own live fair-value/fill records. Both small fixes and deep
       overhauls are on the table, but every change must survive the same referee.</p>
    <div class="callout good"><b>Promotion rule:</b> a candidate model replaces production only if it
      beats the frozen production model on walk-forward log-loss with a paired-bootstrap
      p&nbsp;&lt;&nbsp;0.05, and no major bucket (format, event type, region pair, favorite band)
      gets meaningfully worse. Until then, production stays.</div>
    <div class="callout"><b>How we use Kalshi prices.</b> The market aggregates information BenPom
      can't see (roster news, stand-ins, sharp money) and may well be better calibrated — Phase 1
      measures exactly that. Three sanctioned uses: <b>(1) benchmark</b> — log-loss of Kalshi close
      vs BenPom on the overlap, stratified by liquidity; <b>(2) teacher</b> — regress
      market−BenPom residuals on candidate features (roster turnover, stand-ins, event type) to
      find what the market knows, then feed those <i>features</i> into BenPom from primary sources;
      <b>(3) blend</b> — a walk-forward logit blend of BenPom + market, likely our most accurate
      surface, published alongside pure BenPom and a candidate input for quote <i>sizing</i>. The
      fitted blend weight is itself the key diagnostic: it measures how much information BenPom
      adds beyond the market.</div>
    <div class="callout warn"><b>One boundary stays:</b> the fair value the bot <i>quotes from</i>
      remains structurally independent of the live Kalshi book. The maker edge is the divergence
      between our number and the book, and in books this thin our own quotes are often the price —
      recent closes partially reflect BenPom itself (echo-chamber risk), so the quoting surface
      can't be a function of them.</div>
  </section>

  <section>
    <h2><span class="n">2</span>The model as it stands (inventory)</h2>
    <p class="dim">Every number below verified against the live code (MapElo.py + the VCTMM vendored port, which is bit-exact by construction).</p>
    <ul>
      <li><b>Map ratings</b> — KenPom-style opponent-adjusted net rating per map
          (<code>BuildMapRatings.py</code>), with CN cluster-offset calibration (v10) and
          international-calibration regional offsets + individual bonuses applied to every match
          via <code>getGlobalRatingHUB</code>.</li>
      <li><b>Series probability</b> — 20,000-sim Monte Carlo per matchup: sample the veto
          (ban/pick model v2: recency half-life 6w, own-strength factor <code>(0.3+win)²</code>,
          +0.02 smoothing), then each played map is a Bernoulli with
          <code>P = σ(β·(gA − gB))</code>, <b>β = 0.170</b> shared by every surface.</li>
      <li><b>Match-level logit offsets</b>, applied to the <i>series</i> probability at
          internationals: intl-experience <b>+0.40</b> (signed toward the experienced side of the
          fav/dog pair), CN-underdog <b>+0.35</b> (to the non-CN favorite), grand-final
          upper-bracket edge expressed via the <code>bo5_gf</code> veto sequence in the MC
          (closed form uses <b>+0.25</b> logit instead).</li>
      <li><b>Closed-form cross-check</b> — overall-rating-only surface used as a divergence alarm,
          not for quoting.</li>
      <li><b>History</b>: β, offsets and the veto model were fit via walk-forward backtests
          (train/test split validated). Per-map MC for <i>past-match</i> evaluation was tried and
          rejected (worse Brier). Per-step/format veto conditioning tried and rejected.</li>
    </ul>
  </section>

  <section>
    <h2><span class="n">3</span>Suspected problems &amp; inconsistencies (initial audit)</h2>
    <p class="dim">Found by code reading on 2026-07-22 — each becomes a measurable hypothesis in the backlog. Severity = my prior on probability-accuracy impact, to be confirmed empirically.</p>
    <table>
      <tr><th style="width:52px">ID</th><th>Finding</th><th style="width:86px">Severity</th></tr>
      <tr><td><b>F1</b></td><td><b>Frontend/backend offset gating disagree.</b> The upcoming-card MC applies
        intl offsets at all 9 internationals (2024–2026); the backend past-match path gates them on the three
        2026 events only. Backtests therefore evaluate a <i>different model</i> than the one being traded.
        Decide which is intended, unify, and re-fit affected parameters.</td>
        <td><span class="sev hi">High</span></td></tr>
      <tr><td><b>F2</b></td><td><b>CN-underdog offset is discontinuous at 50%.</b> It fires only when the CN team
        is the <i>underdog</i>: a CN team moving from 50.1% favorite to 49.9% dog snaps the opponent from
        no adjustment to +0.35 logit (≈ +8–9 pts) — a cliff in the output surface right where markets are
        most sensitive. Should be a smooth function (e.g. scaled by |logit|-distance from even, or folded
        into ratings entirely).</td>
        <td><span class="sev hi">High</span></td></tr>
      <tr><td><b>F3</b></td><td><b><code>win_pct = 0</code> reads as 50%.</b> The JS falsy-check
        (<code>win_pct || 0.5</code>, faithfully ported) means a team genuinely 0-for-N on a map is treated
        as a coin flip on it — in veto strength factors and anywhere else this helper feeds. A real 0% map
        should read as ~0% (with shrinkage), not 50%.</td>
        <td><span class="sev med">Medium</span></td></tr>
      <tr><td><b>F4</b></td><td><b>Three stacked CN corrections.</b> Cluster offset (inside ratings) +
        intl-calibration regional offset (per map, every match) + CN-dog match-level offset. Each was fit
        at a different time against different samples — audit for double-counting on the 2026 intl sample,
        where all three fire at once.</td>
        <td><span class="sev med">Medium</span></td></tr>
      <tr><td><b>F5</b></td><td><b>Map-rating fallback asymmetry.</b> A team missing a rating on a picked map
        silently falls back to its overall rating while the opponent uses a real map rating — mixing two
        differently-calibrated scales inside one Bernoulli. Cold-start maps (Corrode) and newly promoted
        orgs hit this constantly.</td>
        <td><span class="sev med">Medium</span></td></tr>
      <tr><td><b>F6</b></td><td><b>Offsets ignore series format.</b> A +0.40 logit shift on the series
        probability moves a Bo3 and a Bo5 identically, while rating edges properly compound with length.
        Test per-format offsets (or apply at map level and let the format aggregate them).</td>
        <td><span class="sev med">Medium</span></td></tr>
      <tr><td><b>F7</b></td><td><b>Ratings are org-level, rosters aren't.</b> A 3-player offseason rebuild
        keeps the org's full rating history. Prediction error should spike right after big roster turnover —
        if confirmed, add roster-continuity decay (weight history by share of returning players).</td>
        <td><span class="sev hi">High</span></td></tr>
      <tr><td><b>F8</b></td><td><b>Veto model uses raw empirical rates.</b> +0.02 additive smoothing with no
        sample-size shrinkage: a team with 4 recent vetoes gets near-raw rates. Wrong vetoes put the wrong
        maps in the simulated series. (Per-step conditioning was already tried and rejected — this is
        about shrinkage, a different axis.)</td>
        <td><span class="sev lo">Low–Med</span></td></tr>
      <tr><td><b>F9</b></td><td><b>Margin construction unaudited.</b> How BuildMapRatings treats overtime
        rounds, blowout margins, and dead/stand-in maps needs a fresh read — margin-based ratings are only
        as good as their margin definition (OT margins are capped by rule; 13-1 vs 13-9 may deserve
        diminishing returns).</td>
        <td><span class="sev med">Medium</span></td></tr>
      <tr><td><b>F10</b></td><td><b>β is one global constant.</b> 0.170 maps rating gaps to map win prob
        everywhere: Bo3/Bo5, LAN/online, playoffs/groups, big-gap/small-gap. Check the logistic link's tails
        against reality (reliability curve at &gt;80% favorites) and test per-context β — carefully,
        this is the easiest place to overfit.</td>
        <td><span class="sev med">Medium</span></td></tr>
    </table>
  </section>

  <section>
    <h2><span class="n">4</span>Data assets</h2>
    <h3>Already on disk</h3>
    <ul>
      <li>Full VLR corpus 2023–2026: <code>match_results.csv</code>, per-map and per-series CSVs,
          <code>map_vetos.csv</code>, rating timelines per season, VLR-enriched round-level JSONs,
          Riot deep-stats pipeline output.</li>
      <li>VCTMM's SQLite: every fair value the bot computed, every quote, fill and settlement since
          it went live (July 2026) — a small but perfectly prospective prediction log.</li>
    </ul>
    <h3>To acquire (Phase 0)</h3>
    <ul>
      <li><b>Kalshi settled-market history</b> — probed today: <b>906 settled markets (~453 match
          events) back to 2026-05-15</b>, publicly readable, cursor-paginated. Pull markets +
          candlesticks (price paths) + trades; capture close price, T-2h mid, and volume. Filter to
          BenPom-covered orgs (the series also carries Game Changers / tier-2 markets).
          Store under <code>data/kalshi/</code> with a loader.</li>
      <li><b>Prediction snapshot logger</b> — from now on, persist every probability the site/bot
          computes (timestamp, model version, inputs hash) so future evaluation is prospective
          instead of reconstructed.</li>
      <li><b>Stretch:</b> VLR 2021–2022 pre-franchising history (different map pool/patches — only
          if diagnosis shows we're data-starved), archived sportsbook closing odds if a clean
          source exists.</li>
    </ul>
  </section>

  <section>
    <h2><span class="n">5</span>Evaluation protocol (the referee)</h2>
    <ul>
      <li><b>Strict walk-forward</b>: the prediction for a match at time T uses only data dated
          &lt; T — ratings snapshots, veto patterns, offsets, everything. No leakage, ever.</li>
      <li><b>Metrics</b>: log-loss (primary), Brier, reliability curves + expected calibration
          error, sharpness. Reported overall and per bucket: format, event type
          (intl / domestic / GF), region pair, CN involvement, favorite-probability band,
          roster-change recency, days-into-split.</li>
      <li><b>Benchmarks</b>: (1) frozen production model, (2) Kalshi closing price, (3) Kalshi
          T-2h mid — the bot's actual quoting horizon, (4) naive overall-rating logistic
          (floor). Beating (1) is required; closing the gap to (2) on the overlap window is the
          honest external yardstick.</li>
      <li><b>Significance</b>: paired bootstrap over matches on per-match log-loss deltas.
          Fits on 2023–2025, holdout 2026 (plus rolling-origin CV for parameter grids).</li>
      <li><b>Divergence review</b>: rank the overlap sample by |BenPom − Kalshi close|, manually
          taxonomize the biggest gaps (stand-in? roster news? dead rubber? model wrong?) — the
          taxonomy tells us which misses are modelable.</li>
    </ul>
  </section>

  <section>
    <h2><span class="n">6</span>Roadmap</h2>
    <div class="phase"><div class="pnum">0</div><div class="pbody">
      <div class="ptitle">Infrastructure <span class="tag">next up</span></div>
      <div class="pdesc">Kalshi history puller + local store; walk-forward harness extension
      (bucketed metrics, reliability plots, paired bootstrap); prediction snapshot logger;
      results render into this tab.</div></div></div>
    <div class="phase"><div class="pnum">1</div><div class="pbody">
      <div class="ptitle">Diagnosis</div>
      <div class="pdesc">Full 2024–2026 report card of the production model; Kalshi divergence
      review; measure whether the market beats BenPom (by liquidity band) and fit the optimal
      BenPom/market logit blend — the weight quantifies what BenPom adds; confirm/kill findings
      F1–F10 with measurements. Output: ranked list of real problems with effect sizes.</div></div></div>
    <div class="phase"><div class="pnum">2</div><div class="pbody">
      <div class="ptitle">Mechanical fixes &amp; refits</div>
      <div class="pdesc">Fix confirmed bugs (F1, F2, F3, F5), unify offset gating, re-fit β and
      offsets walk-forward, per-format tests. Cheap wins first.</div></div></div>
    <div class="phase"><div class="pnum">3</div><div class="pbody">
      <div class="ptitle">Structural experiments</div>
      <div class="pdesc">One at a time, each gated by the referee: roster-continuity decay (F7),
      map cold-start / pool rotation handling (F5 deep version), margin/OT audit (F9),
      intra-series correlation, link-function checks (F10).</div></div></div>
    <div class="phase"><div class="pnum">4</div><div class="pbody">
      <div class="ptitle">Veto &amp; CN layers</div>
      <div class="pdesc">Veto shrinkage (F8); CN triple-stack audit (F4) — possibly collapse three
      corrections into one owned by the ratings.</div></div></div>
    <div class="phase"><div class="pnum">5</div><div class="pbody">
      <div class="ptitle">Promotion &amp; parity</div>
      <div class="pdesc">Promotion rule from §1; then site swap, <code>vendor_sync.py</code> to
      VCTMM, <code>parity_check.py</code> green, STRATEGY.md updated. The bot never quotes on an
      unsynced model.</div></div></div>
  </section>

  <section>
    <h2><span class="n">7</span>Reports</h2>
    <div style="display:flex;flex-direction:column;gap:8px;">
      <!--REPORTS-->
    </div>
  </section>

  <section>
    <h2><span class="n">8</span>Status log</h2>
    <!--STATUS-->
    <div class="status-row"><div class="status-date">2026-07-22</div>
      <div>Kalshi policy revised after review: prices are a sanctioned signal (benchmark, teacher,
      blend surface + sizing input), not just a diagnostic — often likely better than BenPom, and
      Phase 1 will measure that directly. Only the bot's quoting fair value stays book-independent
      (edge + echo-chamber reasons, §1).</div></div>
    <div class="status-row"><div class="status-date">2026-07-22</div>
      <div>Lab created. Model inventory verified against code; initial audit produced findings
      F1–F10. Kalshi probe: 906 settled markets since 2026-05-15, public API confirmed.
      Next: Phase 0 infrastructure.</div></div>
  </section>
</div>
</body></html>"""
