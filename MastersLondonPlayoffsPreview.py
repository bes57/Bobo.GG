"""Article: Masters London Playoffs Preview.

Mirrors the Masters London Preview layout (label / h1 / byline / hero image /
sections TOC top-right / body). Body is a skeleton placeholder for now.
"""

import os
import json
from flask import Blueprint, render_template_string

article_masters_london_playoffs_bp = Blueprint("article_masters_london_playoffs", __name__)

_ROOT = os.path.dirname(os.path.abspath(__file__))
_PLAYOFF_TEAMS = ['TH', 'G2', 'PRX', 'VIT', 'FUT', 'LEV', 'EDG', 'XLG']
_TEAM_REGION = {'TH': 'EMEA', 'G2': 'Americas', 'PRX': 'Pacific', 'VIT': 'EMEA',
                'FUT': 'EMEA', 'LEV': 'Americas', 'EDG': 'China', 'XLG': 'China'}
# Established site region palette (Modern Hub region badges / over-underperformers
# article): Americas=orange, EMEA=green, Pacific=blue, China=pink.
_REGION_COLOR = {'EMEA': '#16a34a', 'Americas': '#ea580c',
                 'Pacific': '#2563eb', 'China': '#db2777'}


_WINDOW_START = '2026-05-13'


def _chart_payload():
    """Per-playoff-team BenPom timeline (windowed to the Masters London run) for
    the regional-shift chart. Read live from rating_timeline.json so it matches
    the Modern Hub. Each checkpoint where the team played a match also carries an
    `m` object (win/loss, opponent, score, rating delta, per-map breakdown) so the
    frontend can draw hover-able green/red match dots like the Modern Hub."""
    try:
        tl = json.load(open(os.path.join(_ROOT, 'data', 'rating_timeline.json')))
    except Exception:
        return {'series': {}, 'region': _TEAM_REGION, 'colors': _REGION_COLOR}
    cps = tl.get('checkpoints', [])
    mes = tl.get('match_events', [])
    pset = set(_PLAYOFF_TEAMS)

    # Index each playoff team's matches by (team, date), from that team's POV.
    match_by = {}
    for me in mes:
        d = me.get('date') or ''
        if d < _WINDOW_START:
            continue
        if me.get('event_id') != '2026_masters_london':
            continue   # dots are Masters London results only — this chart is the London shift
        w, l = me.get('winner'), me.get('loser')
        # Modern-Hub-shaped match object so the popup can reuse the Hub's
        # _matchTooltipHTML(m, won) builder verbatim.
        me_obj = {
            'winner': w, 'loser': l,
            'winner_delta': me.get('winner_delta', 0.0), 'loser_delta': me.get('loser_delta', 0.0),
            'winner_after': me.get('winner_after', 0.0), 'loser_after': me.get('loser_after', 0.0),
            'series_score': me.get('series_score', ''),
            'event_id': me.get('event_id', ''),
            'date': d,
            'maps': me.get('maps') or [],
        }
        for team, won in ((w, True), (l, False)):
            if team not in pset:
                continue
            match_by[(team, d)] = {'won': won, 'me': me_obj}

    series = {}
    for t in _PLAYOFF_TEAMS:
        pts = []
        for c in cps:
            d = c.get('date') or ''
            if d < _WINDOW_START:
                continue
            r = c.get('ratings') or {}
            if t not in r:
                continue
            pt = {'x': d, 'y': round(r[t], 3)}
            mm = match_by.get((t, d))
            if mm:
                pt['me'] = mm['me']
                pt['won'] = mm['won']
            pts.append(pt)
        if pts:
            series[t] = pts
    return {'series': series, 'region': _TEAM_REGION, 'colors': _REGION_COLOR}


# ── BenPom-prediction match cards (win% = upcoming-match formula; veto + per-map
#    breakdown = predicted/display, exactly like the Modern Hub upcoming card) ──
import math as _math

# v6 site model (data/site_model.json) — single source of truth for the
# bracket-card probabilities (β, cross-region offsets, gf_upper_logit,
# b_pick); reference math = trading_model/predict.py. The article PROSE is
# frozen, but these cards always recomputed from live ratings, so they track
# the deployed model.
_SITE_MODEL_PATH = os.path.join(_ROOT, 'data', 'site_model.json')

def _site_model():
    return json.load(open(_SITE_MODEL_PATH))

# Article region labels use 'China'; the model's offset keys use 'CN'.
_REGION_KEY = {'China': 'CN'}
_VETO_STEPS = {
    'bo3': [('A', 'ban'), ('B', 'ban'), ('A', 'pick'), ('B', 'pick'), ('A', 'ban'), ('B', 'ban')],
    'bo5': [('A', 'ban'), ('B', 'ban'), ('A', 'pick'), ('B', 'pick'), ('A', 'pick'), ('B', 'pick')],
    'bo5_gf': [('A', 'ban'), ('A', 'ban'), ('A', 'pick'), ('B', 'pick'), ('A', 'pick'), ('B', 'pick')],
}
# The 14 chalk matches (A = bracket-top team). GF: A = upper-bracket (UF winner).
_BRACKET_MATCHES = [
    ('uqf1', 'G2', 'XLG', 'bo3', None), ('uqf2', 'EDG', 'FUT', 'bo3', None),
    ('uqf3', 'PRX', 'LEV', 'bo3', None), ('uqf4', 'TH', 'VIT', 'bo3', None),
    ('usf1', 'G2', 'FUT', 'bo3', None), ('usf2', 'PRX', 'TH', 'bo3', None),
    ('uf', 'G2', 'TH', 'bo5', None), ('gf', 'TH', 'G2', 'bo5_gf', 'TH'),
    ('lr1a', 'XLG', 'EDG', 'bo3', None), ('lr1b', 'LEV', 'VIT', 'bo3', None),
    ('lr2a', 'FUT', 'VIT', 'bo3', None), ('lr2b', 'PRX', 'EDG', 'bo3', None),
    ('lr3', 'VIT', 'PRX', 'bo3', None), ('lf', 'G2', 'PRX', 'bo5', None),
]


def _sig(x):
    return 1.0 / (1.0 + _math.exp(-x))


def _logit(p):
    p = min(max(p, 1e-9), 1 - 1e-9)
    return _math.log(p / (1 - p))


def _series_p(p, fmt):
    if fmt in ('bo5', 'bo5_gf'):
        return p ** 3 * (1 + 3 * (1 - p) + 6 * (1 - p) ** 2)
    return p * p * (3 - 2 * p)


def _bracket_cards():
    try:
        vm = json.load(open(os.path.join(_ROOT, 'data', 'veto_model.json')))
        mr = json.load(open(os.path.join(_ROOT, 'data', 'map_ratings.json')))
        tl = json.load(open(os.path.join(_ROOT, 'data', 'rating_timeline.json')))
    except Exception:
        return {}
    snap = '2026_after_london'
    veto = (vm.get('teams') or {}).get(snap, {})
    pool = (vm.get('snap_pools') or {}).get(snap) or []
    mteams = mr['ratings']['2026']['snapshots']['after_london']['teams']
    ovr = max(tl['checkpoints'], key=lambda c: c['date'])['ratings']
    sm = _site_model()

    def map_winpct(org, m):  # opponent-agnostic strength on map, for veto scoring
        mm = (mteams.get(org, {}).get('maps') or {}).get(m)
        return mm.get('win_pct', 0.5) if mm else 0.5

    def ban_argmax(patt, opp, rem):
        sc = {m: ((patt or {}).get('bans', {}).get(m, 0) + 0.02) * (0.75 + map_winpct(opp, m)) for m in rem}
        return max(sc, key=sc.get)

    def pick_argmax(patt, own, rem):
        sc = {m: ((patt or {}).get('picks', {}).get(m, 0) + 0.02) * ((0.3 + map_winpct(own, m)) ** 2) for m in rem}
        return max(sc, key=sc.get)

    def xadj(a, b):
        # v6 cross-region adjustment (predict.py): off[reg_a] − off[reg_b],
        # 0 same-region. Replaces the old intl_exp/cn_dog logit shifts.
        ra = _REGION_KEY.get(_TEAM_REGION.get(a, ''), _TEAM_REGION.get(a, ''))
        rb = _REGION_KEY.get(_TEAM_REGION.get(b, ''), _TEAM_REGION.get(b, ''))
        if not ra or not rb or ra == rb:
            return 0.0
        off = sm.get('xregion_offsets') or {}
        return off.get(ra, 0.0) - off.get(rb, 0.0)

    def winprob_a(a, b, fmt, gfu):
        # v6 closed form (predict.py series_probability).
        p = _series_p(_sig(sm['beta'] * (ovr[a] - ovr[b] + xadj(a, b))), fmt)
        if gfu:
            p = _sig(_logit(p) + (sm['gf_upper_logit'] if gfu == a else -sm['gf_upper_logit']))
        return p

    out = {}
    for mid, a, b, fmt, gfu in _BRACKET_MATCHES:
        pa = winprob_a(a, b, fmt, gfu)
        rem = list(pool)
        seq = []
        for side, act in _VETO_STEPS[fmt]:
            patt = veto.get(a if side == 'A' else b)
            if act == 'ban':
                m = ban_argmax(patt, b if side == 'A' else a, rem)
            else:
                m = pick_argmax(patt, a if side == 'A' else b, rem)
            seq.append({'side': side, 'action': act, 'map': m})
            rem = [x for x in rem if x != m]
        if rem:
            seq.append({'side': '', 'action': 'dec', 'map': rem[0]})
        maps = []
        z_base = sm['beta'] * (ovr[a] - ovr[b] + xadj(a, b))
        for s in seq:
            if s['action'] in ('pick', 'dec'):
                m = s['map']
                # v6 map prob (predict.py map_probability): overall ratings +
                # xregion + the pick logit toward the picker (decider: none).
                z = z_base + (sm['b_pick'] if s['side'] == 'A' else
                              (-sm['b_pick'] if s['side'] == 'B' else 0.0))
                wp = _sig(z)
                maps.append({'map': m, 'wp_a': round(wp, 3), 'fate': s['action'] + s['side']})
        out[mid] = {'a': a, 'b': b, 'fmt': fmt, 'pa': round(pa, 3),
                    'winner': a if pa >= 0.5 else b, 'veto': seq, 'maps': maps}
    return out

PAGE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=820">
<title>Masters London: Playoffs Preview &mdash; Bobo's VCT Database</title>
<!-- Open Graph / Twitter link-preview cards -->
<meta property="og:type" content="article">
<meta property="og:site_name" content="Bobo gg">
<meta property="og:title" content="Masters London Playoffs Preview">
<meta property="og:description" content="A brief statistical glimpse into the final stage of Masters London.">
<meta property="og:url" content="https://bobo-gg.net/articles/masters-london-playoffs-preview/">
<meta property="og:image" content="https://bobo-gg.net/chronlondon.jpg">
<meta property="og:image:secure_url" content="https://bobo-gg.net/chronlondon.jpg">
<meta property="og:image:type" content="image/jpeg">
<meta property="og:image:width" content="2048">
<meta property="og:image:height" content="1366">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="Masters London Playoffs Preview">
<meta name="twitter:description" content="A brief statistical glimpse into the final stage of Masters London.">
<meta name="twitter:image" content="https://bobo-gg.net/chronlondon.jpg">
<link rel="icon" type="image/svg+xml" href="/favicon.svg">
<link rel="preload" as="image" href="/logos/TH.png"><link rel="preload" as="image" href="/logos/G2.png"><link rel="preload" as="image" href="/logos/PRX.png"><link rel="preload" as="image" href="/logos/VIT.png"><link rel="preload" as="image" href="/logos/FUT.png"><link rel="preload" as="image" href="/logos/LEV.png"><link rel="preload" as="image" href="/logos/EDG.png"><link rel="preload" as="image" href="/logos/XLG.png">
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@700;800&family=DM+Sans:wght@300;400;500;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="/static/base.css">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3.0.0/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
<style>
  html { -webkit-text-size-adjust:100%; text-size-adjust:100%; }
  .top-nav { padding:32px 32px 0; position:relative; z-index:1; }
  .home-logo { height:80px; width:auto; display:block; opacity:.85; transition:opacity .2s; }
  .home-logo:hover { opacity:1; }
  /* Anchored at the viewport's right edge, but its width is clamped to the
     space actually available beside the centered 860px article
     (50vw - 430 - 32 - 20 gutter), so at any resolution/zoom it narrows in
     place rather than ever reaching the text. Hidden when no room remains. */
  .toc { position:fixed; top:32px; right:32px; background:white; border-radius:16px; padding:20px 24px; box-shadow:0 4px 24px #0000000f; display:flex; flex-direction:column; gap:6px; z-index:100; width:max-content; max-width:min(240px, calc(50vw - 482px)); }
  .toc-title { font-family:'Plus Jakarta Sans',sans-serif; font-size:0.77rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; color:var(--soft); margin-bottom:4px; }
  .toc a { font-size:.78rem; color:var(--soft); text-decoration:none; font-weight:400; transition:color .15s; line-height:1.4; }
  .toc a:hover { color:var(--ink); }
  .toc a.active { color:var(--ink); font-weight:500; }
  .toc a.toc-sub { padding-left:16px; font-size:.74rem; border-left:2px solid #ede5f3; margin-left:4px; }
  .toc a.toc-sub.active { border-left-color:#7c3aed; color:var(--ink); font-weight:600; }
  .alpha-navbar ~ .toc { top:72px; }
  @media(max-width:1200px) { .toc { display:none; } }
  .page { position:relative; z-index:1; flex:1; display:flex; flex-direction:column; align-items:center; padding:60px 32px 80px; }
  .article { max-width:860px; width:100%; }
  .label { font-family:'Plus Jakarta Sans',sans-serif; font-size:0.77rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase; color:var(--soft); margin-bottom:16px; text-align:center; }
  h1 { font-family:'Plus Jakarta Sans',sans-serif; font-size:clamp(2.2rem,5vw,3.52rem); font-weight:800; letter-spacing:-1px; line-height:1.1; margin-bottom:16px; text-align:center; }
  .dek { font-size:1.05rem; font-weight:300; line-height:1.55; color:var(--ink); margin-bottom:24px; opacity:.85; }
  .byline { font-size:.82rem; color:var(--soft); font-weight:300; margin-bottom:48px; padding-bottom:32px; border-bottom:1px solid #e8e0ec; text-align:center; }
  .cover { width:100%; border-radius:16px; overflow:hidden; margin-bottom:12px; }
  .cover img { width:100%; height:auto; display:block; }
  .cover-caption { font-size:.75rem; color:var(--soft); font-weight:300; font-style:italic; margin-bottom:48px; text-align:center; }
  .content p { font-size:1rem; font-weight:300; line-height:1.8; color:var(--ink); margin-bottom:24px; }
  .section-bubble { display:inline-block; background:white; padding:16px 44px; border-radius:999px; box-shadow:0 6px 32px #0000001a; }
  .section-bubble-text { font-family:'Plus Jakarta Sans',sans-serif; font-size:2.09rem; font-weight:800; letter-spacing:.01em;
    background-image:linear-gradient(95deg,#f472b6 0%,#a855f7 55%,#7c3aed 100%);
    -webkit-background-clip:text; background-clip:text;
    -webkit-text-fill-color:transparent; color:transparent;
  }
  .section-bubble-wrap { text-align:center; margin:96px 0 32px; }
  .section-bubble-wrap.section-bubble-tight { margin-top:40px; }
  .section-note { font-size:.85rem; color:var(--soft); font-weight:400; font-style:italic; line-height:1.55; text-align:center; max-width:680px; margin:-12px auto 28px; padding:0 24px; }

  /* Two-team comparison panel (Gen.G vs PRX) */
  .comparison-chart { background:white; border-radius:16px; padding:24px 28px; box-shadow:0 4px 24px #0000000a; margin:24px 0 32px; }
  .comparison-header { display:grid; grid-template-columns:1fr 150px 1fr; gap:16px; padding-bottom:14px; margin-bottom:8px; border-bottom:2px solid #f0eaf4; align-items:end; }
  .comparison-team { display:flex; flex-direction:column; align-items:center; gap:6px; }
  .comparison-team img { width:38px; height:38px; object-fit:contain; }
  .comparison-team-name { font-family:'Plus Jakarta Sans',sans-serif; font-size:1.155rem; font-weight:800; }
  .comparison-team-sub { font-size:.7rem; color:var(--soft); font-weight:300; text-align:center; line-height:1.3; }
  .comparison-row { display:grid; grid-template-columns:1fr 150px 1fr; gap:16px; padding:14px 0; align-items:center; border-bottom:1px solid #f5eff8; }
  .comparison-row:last-child { border-bottom:none; }
  .comparison-value { text-align:center; font-weight:500; line-height:1.4; font-size:.95rem; }
  .comparison-value.num { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.43rem; font-variant-numeric:tabular-nums; }
  .comparison-label { font-family:'Plus Jakarta Sans',sans-serif; font-size:0.682rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; color:var(--soft); text-align:center; }

  /* Tier list overview (click-to-jump chips) */
  .tier-overview { background:white; border-radius:16px; padding:20px 24px; box-shadow:0 4px 24px #0000000a; margin:24px 0 18px; }
  .tier-overview-label { font-family:'Plus Jakarta Sans',sans-serif; font-size:0.77rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; color:var(--soft); margin-bottom:14px; text-align:center; }
  .tier-row { display:grid; grid-template-columns:54px 1fr; gap:14px; align-items:center; padding:10px 0; border-bottom:1px solid #f5eff8; }
  .tier-row:last-child { border-bottom:none; }
  .tier-badge { font-family:'Plus Jakarta Sans',sans-serif; font-size:1.54rem; font-weight:800; height:46px; border-radius:10px; display:flex; align-items:center; justify-content:center; color:white; }
  .tier-badge.s { background:#e74c3c; }
  .tier-badge.a { background:#e67e22; }
  .tier-badge.b { background:#d4a017; }
  .tier-badge.c { background:#27ae60; }
  .tier-badge.d { background:#4a7fbf; }
  .tier-teams { display:flex; flex-wrap:wrap; gap:8px; }
  .tier-chip { display:inline-flex; align-items:center; justify-content:center; width:74px; height:74px; padding:8px; border-radius:14px; background:#faf6fd; text-decoration:none; transition:background .15s, transform .15s; }
  .tier-chip:hover { background:#efe2f7; transform:translateY(-2px) scale(1.05); }
  .tier-chip img { width:100%; height:100%; object-fit:contain; }

  /* Tier-section banner header */
  .tier-section-header { font-family:'Plus Jakarta Sans',sans-serif; font-size:1.54rem; font-weight:800; margin:32px 0 16px; padding:10px 28px; border-radius:12px; color:white; display:inline-block; }
  .tier-section-header.s { background:#e74c3c; }
  .tier-section-header.a { background:#e67e22; }
  .tier-section-header.b { background:#d4a017; }
  .tier-section-header.c { background:#27ae60; }
  .tier-section-header.d { background:#4a7fbf; }

  /* Per-team heading + stat card + why-blocks */
  .team-heading { display:flex; align-items:center; gap:14px; margin:48px 0 20px; flex-wrap:wrap; }
  .team-heading img { width:42px; height:42px; object-fit:contain; flex-shrink:0; }
  .team-heading h2 { margin:0; font-family:'Plus Jakarta Sans',sans-serif; font-size:1.54rem; font-weight:800; letter-spacing:-.5px; }
  .tier-pill { font-family:'Plus Jakarta Sans',sans-serif; font-size:0.748rem; font-weight:800; letter-spacing:.08em; padding:4px 10px; border-radius:6px; color:white; }
  .tier-pill.s { background:#e74c3c; }
  .tier-pill.a { background:#e67e22; }
  .tier-pill.b { background:#d4a017; }
  .tier-pill.c { background:#27ae60; }
  .tier-pill.d { background:#4a7fbf; }
  .team-stat-card { background:white; border-radius:14px; padding:18px 22px; box-shadow:0 4px 24px #0000000a; margin:0 0 24px; }
  .team-stat-row { display:flex; gap:28px; flex-wrap:wrap; justify-content:center; }
  .team-stat-block { display:flex; flex-direction:column; gap:5px; align-items:center; text-align:center; min-width:170px; flex:1; }
  .team-stat-label { font-family:'Plus Jakarta Sans',sans-serif; font-size:0.792rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; color:var(--soft); }
  .team-stat-value { font-family:'Plus Jakarta Sans',sans-serif; font-size:1.1rem; font-weight:700; }
  .team-stat-value.big { font-size:1.45rem; font-variant-numeric:tabular-nums; }
  .team-stat-value.muted { color:var(--soft); font-style:italic; font-weight:400; }

  .roster-row { margin-top:18px; padding-top:16px; border-top:1px solid #f5eff8; }
  .roster-row-label { font-family:'Plus Jakarta Sans',sans-serif; font-size:0.792rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; color:var(--soft); margin-bottom:12px; text-align:center; }
  .roster-headshots { display:flex; gap:18px; justify-content:center; flex-wrap:wrap; }
  .roster-player { display:flex; flex-direction:column; align-items:center; gap:7px; width:80px; }
  .roster-headshot { width:66px; height:66px; border-radius:50%; object-fit:cover; object-position:top center; background:#f0ecf4; flex-shrink:0; }
  .roster-player-name { font-size:.78rem; font-weight:600; color:var(--ink); text-align:center; line-height:1.2; word-break:break-word; }
  .team-stat-value.pos { color:#1a6a4a; }
  .team-stat-value.neg { color:#a33247; }

  .why-block { margin:32px 0 18px; }
  .why-label { font-family:'Plus Jakarta Sans',sans-serif; font-size:0.77rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; margin-bottom:6px; display:block; }
  .why-label.win { color:#1a6a4a; }
  .why-label.lose { color:#a33247; }
  .why-label.watch { color:#7c3aed; }
  .why-block .placeholder { font-style:italic; color:var(--soft); font-weight:300; margin:0; }
  .content strong, .why-block strong { font-weight:800; color:var(--ink); }
  .why-block p { font-size:1rem; font-weight:300; line-height:1.8; color:var(--ink); margin:0 0 12px; }
  .why-block p:last-child { margin-bottom:0; }
  .content ul { list-style:none; margin:-4px 0 14px; display:flex; flex-direction:column; gap:6px; padding-left:0; }
  .content ul li { font-size:1rem; font-weight:300; line-height:1.7; padding-left:20px; position:relative; color:var(--ink); }
  .content ul li::before { content:'—'; position:absolute; left:0; color:var(--soft); }
  .content ol.numbered { list-style:none; margin:-4px 0 14px; padding-left:0; display:flex; flex-direction:column; gap:8px; counter-reset:nl; }
  .content ol.numbered li { font-size:1rem; font-weight:300; line-height:1.7; padding-left:30px; position:relative; color:var(--ink); counter-increment:nl; }
  .content ol.numbered li::before { content:counter(nl) '.'; position:absolute; left:0; color:#ED1C7C; font-weight:800; font-family:'Plus Jakarta Sans',sans-serif; }
  .team-lede { font-size:1rem; font-weight:300; line-height:1.8; color:var(--ink); margin:4px 0 24px; }
  .player-delta-chart { display:flex; flex-direction:column; gap:14px; margin:16px 0 4px; }
  .player-delta-event { background:white; border-radius:14px; padding:14px 18px; box-shadow:0 4px 24px #0000000a; }
  .player-delta-title { font-family:'Plus Jakarta Sans',sans-serif; font-size:0.858rem; font-weight:800; letter-spacing:.04em; color:var(--ink); padding-bottom:9px; border-bottom:1px solid #f0eaf4; margin-bottom:6px; }
  .player-delta-table { width:100%; border-collapse:collapse; font-size:.88rem; font-variant-numeric:tabular-nums; }
  .player-delta-table th { font-family:'Plus Jakarta Sans',sans-serif; font-size:0.66rem; font-weight:800; text-transform:uppercase; letter-spacing:.08em; color:var(--soft); padding:6px 4px; text-align:right; }
  .player-delta-table th:first-child { text-align:left; }
  .player-delta-table td { padding:7px 4px; text-align:right; border-bottom:1px solid #f5eff8; }
  .player-delta-table td:first-child { text-align:left; font-weight:600; }
  .player-delta-table tr:last-child td { border-bottom:none; }
  .player-delta-table .pos { color:#1a6a4a; font-weight:700; }
  .player-delta-table .neg { color:#a33247; font-weight:700; }
  .player-delta-table .flat { color:var(--soft); font-weight:600; }
  .player-delta-title { display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
  .delta-event-pill { font-family:'Plus Jakarta Sans',sans-serif; font-size:0.638rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; padding:3px 9px; border-radius:6px; }
  .delta-event-pill.won  { background:rgba(34,197,94,.18); color:#176a47; }
  .delta-event-pill.lost { background:rgba(220,38,38,.13); color:#a33247; }
  .data-table-wrap { background:white; border-radius:14px; padding:20px 28px; box-shadow:0 4px 24px #0000000a; margin:14px -105px 32px; overflow-x:auto; -webkit-overflow-scrolling:touch; }
  @media(max-width:1120px) { .data-table-wrap { margin-left:0; margin-right:0; } }
  .data-table-label { font-family:'Plus Jakarta Sans',sans-serif; font-size:1.21rem; font-weight:800; letter-spacing:.02em; text-transform:uppercase; color:var(--ink); text-align:center; margin-bottom:16px; }
  .data-table { width:100%; border-collapse:collapse; font-size:.96rem; font-weight:400; }
  .data-table th { font-family:'DM Sans',sans-serif; font-size:.95rem; font-weight:700; letter-spacing:.01em; color:var(--ink); padding:9px 12px; text-align:left; border-bottom:2px solid #f0eaf4; }
  .data-table td { padding:11px 12px; border-bottom:1px solid #f5eff8; }
  .data-table tr:last-child td { border-bottom:none; }
  .data-table .num { font-variant-numeric:tabular-nums; text-align:center; }
  .data-table td.num:not(.pos):not(.neg) { font-weight:800; font-size:1.08rem; }
  .data-table th.num { text-align:center; }
  .data-table .rank { color:var(--soft); font-weight:500; width:96px; text-align:center; }
  .data-table th.rank { text-align:center; white-space:nowrap; }
  .data-table .team { font-weight:600; }
  .data-table tr.highlight td { background:rgba(168,85,247,.07); }
  .data-table tr.highlight .team { color:#7c3aed; }
  .data-table .team-cell { display:flex; align-items:center; gap:10px; }
  .data-table .team-logo { width:25px; height:25px; object-fit:contain; flex-shrink:0; }

  /* Player-rating heatmap (7 matches × 6 players) */
  .rating-heatmap-wrap { background:white; border-radius:14px; padding:18px 20px 16px; box-shadow:0 4px 24px #0000000a; margin:14px 0 6px; overflow-x:auto; }
  .rating-heatmap-title { font-family:'Plus Jakarta Sans',sans-serif; font-size:0.858rem; font-weight:800; letter-spacing:.04em; color:var(--ink); padding-bottom:10px; border-bottom:1px solid #f0eaf4; margin-bottom:12px; }
  .rating-heatmap { width:100%; min-width:680px; border-collapse:separate; border-spacing:4px; font-variant-numeric:tabular-nums; }
  .rating-heatmap th { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:0.682rem; letter-spacing:.04em; color:var(--ink); padding:6px 4px; text-align:center; vertical-align:bottom; line-height:1.25; }
  .rating-heatmap th.player-col { text-align:left; font-size:.7rem; min-width:90px; padding-left:8px; color:var(--soft); text-transform:uppercase; letter-spacing:.08em; }
  .rating-heatmap th .ev-sub { display:block; font-size:.52rem; color:var(--soft); font-weight:700; margin-top:3px; text-transform:uppercase; letter-spacing:.06em; }
  .rating-heatmap th .wl { display:inline-block; padding:2px 7px; border-radius:5px; font-size:.5rem; margin-top:4px; letter-spacing:.06em; }
  .rating-heatmap th .wl.w { background:rgba(34,197,94,.2); color:#176a47; }
  .rating-heatmap th .wl.l { background:rgba(220,38,38,.14); color:#a33247; }
  .rating-heatmap td { padding:9px 4px; text-align:center; border-radius:6px; font-weight:700; font-size:.85rem; }
  .rating-heatmap td.name { text-align:left; font-family:'Plus Jakarta Sans',sans-serif; font-size:0.935rem; font-weight:800; padding-left:8px; color:var(--ink); background:transparent; border-radius:0; }
  .rating-heatmap tr.smth-row td.name { color:#ED1C7C; }
  .rating-heatmap tr.smth-row { box-shadow:inset 3px 0 0 #ED1C7C; }
  .rating-heatmap td.r1 { background:#fbe1e4; color:#9b2138; }
  .rating-heatmap td.r2 { background:#fceadf; color:#955a26; }
  .rating-heatmap td.r3 { background:#fbf6e0; color:#7a6814; }
  .rating-heatmap td.r4 { background:#e8f3df; color:#3d6b28; }
  .rating-heatmap td.r5 { background:#d2e9d5; color:#176a47; }
  .rating-heatmap td.dnp { background:#f7f3fa; color:#bbb1c7; font-style:italic; font-weight:500; font-size:.78rem; }
  .rating-heatmap td.lowest { outline:2px solid #ED1C7C; outline-offset:-2px; }
  .rating-heatmap-legend { font-size:.7rem; color:var(--soft); font-weight:400; margin-top:10px; text-align:center; font-style:italic; }
  .rating-heatmap-legend .swatch { display:inline-block; width:11px; height:11px; border-radius:3px; vertical-align:-1px; margin:0 4px 0 10px; }
  .rating-heatmap-legend .swatch.low { background:#fbe1e4; }
  .rating-heatmap-legend .swatch.high { background:#d2e9d5; }
  .rating-heatmap-legend .ring { display:inline-block; width:11px; height:11px; border-radius:3px; vertical-align:-1px; margin:0 4px 0 10px; border:2px solid #ED1C7C; }

  /* Top 15 players ranked list — vertical cards with descriptions */
  .player-rank-list { display:flex; flex-direction:column; gap:14px; margin:14px 0 24px; padding:0; list-style:none; }
  .player-rank-card { display:flex; align-items:center; gap:18px; background:white; border-radius:14px; padding:16px 20px; box-shadow:0 4px 24px #0000000a; }
  .player-rank-header { display:flex; flex-direction:column; align-items:center; gap:8px; width:96px; flex-shrink:0; padding-right:14px; border-right:1px solid #f0eaf4; }
  .player-rank-num { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.705rem; color:var(--soft); line-height:1; font-variant-numeric:tabular-nums; }
  .player-rank-img { width:64px; height:64px; border-radius:50%; object-fit:cover; object-position:top center; background:#f0ecf4; flex-shrink:0; }
  .player-rank-name { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:0.99rem; line-height:1.15; text-align:center; white-space:nowrap; }
  .player-rank-team { display:flex; align-items:center; gap:5px; font-family:'Plus Jakarta Sans',sans-serif; font-size:0.66rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase; color:var(--soft); }
  .player-rank-team img { width:14px; height:14px; object-fit:contain; }
  .content p.player-rank-desc { font-size:.95rem; font-weight:300; line-height:1.65; color:var(--ink); margin:0; flex:1; align-self:center; }
  @media(max-width:560px) { .player-rank-card { flex-direction:column; align-items:stretch; } .player-rank-header { flex-direction:row; width:auto; padding-right:0; border-right:none; border-bottom:1px solid #f0eaf4; padding-bottom:12px; gap:14px; justify-content:flex-start; align-self:auto; } }

  /* Masters London FI ranking list */
  .fi-rank-wrap { background:white; border-radius:14px; padding:18px 22px 16px; box-shadow:0 4px 24px #0000000a; margin:14px 0 4px; }
  .fi-rank-title { font-family:'Plus Jakarta Sans',sans-serif; font-size:0.858rem; font-weight:800; letter-spacing:.04em; color:var(--ink); padding-bottom:9px; border-bottom:1px solid #f0eaf4; margin-bottom:12px; }
  .fi-rank-list { columns:3; column-gap:28px; padding:0; list-style:none; margin:0; }
  .fi-rank-list li { display:flex; align-items:center; gap:8px; padding:6px 4px; font-size:.8rem; break-inside:avoid; border-bottom:1px solid #f5eff8; }
  .fi-rank-list li .rank { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:0.748rem; color:var(--soft); width:22px; flex-shrink:0; text-align:right; }
  .fi-rank-list li img { width:18px; height:18px; object-fit:contain; flex-shrink:0; }
  .fi-rank-list li .player { flex:1; font-weight:600; line-height:1.2; }
  .fi-rank-list li .org-tag { font-family:'Plus Jakarta Sans',sans-serif; font-size:0.605rem; font-weight:800; letter-spacing:.06em; color:var(--soft); margin-left:4px; vertical-align:1px; }
  .fi-rank-list li .pct { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:0.935rem; font-variant-numeric:tabular-nums; }
  .fi-rank-list li.smth-highlight { background:rgba(237,28,124,.1); border-radius:6px; padding-left:6px; }
  .fi-rank-list li.smth-highlight .player { color:#ED1C7C; }
  .fi-rank-list li.smth-highlight .pct { color:#ED1C7C; }
  @media(max-width:720px) { .fi-rank-list { columns:2; } }
  @media(max-width:480px) { .fi-rank-list { columns:1; } }

  /* Inline tweet embeds */
  .tweet-embeds { display:flex; flex-direction:column; gap:14px; align-items:center; margin:14px 0 24px; }
  .tweet-embeds blockquote.twitter-tweet { max-width:550px; width:100%; margin:0 !important; }

  /* YouTube embed (16:9) */
  .yt-embed { position:relative; width:100%; padding-top:56.25%; border-radius:12px; overflow:hidden; margin:14px 0 4px; box-shadow:0 4px 24px #0000000a; background:#000; }
  .yt-embed iframe { position:absolute; inset:0; width:100%; height:100%; border:0; }

  /* Primmie FI-share split chart */
  .primmie-chart { background:white; border-radius:14px; padding:18px 22px; box-shadow:0 4px 24px #0000000a; margin:14px 0 24px; }
  .primmie-title { font-family:'Plus Jakarta Sans',sans-serif; font-size:0.858rem; font-weight:800; letter-spacing:.04em; color:var(--ink); padding-bottom:10px; border-bottom:1px solid #f0eaf4; margin-bottom:14px; }
  .primmie-grid { display:grid; grid-template-columns:1fr 40px 1fr; align-items:center; gap:14px; }
  .primmie-col { display:flex; flex-direction:column; align-items:center; gap:6px; padding:14px 12px; border-radius:12px; }
  .primmie-col.high { background:rgba(34,197,94,.10); }
  .primmie-col.low  { background:rgba(220,38,38,.08); }
  .primmie-pct { font-family:'Plus Jakarta Sans',sans-serif; font-size:2.64rem; font-weight:800; font-variant-numeric:tabular-nums; line-height:1; }
  .primmie-col.high .primmie-pct { color:#176a47; }
  .primmie-col.low  .primmie-pct { color:#a33247; }
  .primmie-label { font-size:.78rem; color:var(--ink); font-weight:500; text-align:center; line-height:1.3; }
  .primmie-record { font-family:'Plus Jakarta Sans',sans-serif; font-size:0.858rem; font-weight:800; letter-spacing:.05em; color:var(--soft); }
  .primmie-divider { font-family:'Plus Jakarta Sans',sans-serif; font-size:0.88rem; font-weight:800; letter-spacing:.1em; color:var(--soft); text-align:center; }
  @media(max-width:560px) { .primmie-grid { grid-template-columns:1fr; } .primmie-divider { padding:6px 0; } }

  /* Featured player stat-line */
  .player-statline { background:white; border-radius:14px; padding:16px 20px; box-shadow:0 4px 24px #0000000a; margin:14px 0 24px; }
  .player-statline-head { display:flex; align-items:center; gap:14px; margin-bottom:14px; padding-bottom:10px; border-bottom:1px solid #f0eaf4; }
  .player-statline-head img.headshot { width:44px; height:44px; border-radius:50%; object-fit:cover; object-position:top center; background:#f0ecf4; flex-shrink:0; }
  .player-statline-name { font-family:'Plus Jakarta Sans',sans-serif; font-size:1.1rem; font-weight:800; color:var(--ink); line-height:1.2; }
  .player-statline-sub { font-size:.72rem; color:var(--soft); font-weight:400; margin-top:2px; }
  .player-statline-stats { display:grid; grid-template-columns:repeat(4, 1fr); gap:10px; }
  .player-statline-stat { display:flex; flex-direction:column; gap:2px; align-items:center; }
  .player-statline-stat-label { font-family:'DM Sans',sans-serif; font-size:.55rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase; color:var(--soft); }
  .player-statline-stat-value { font-family:'DM Sans',sans-serif; font-size:1.05rem; font-weight:800; font-variant-numeric:tabular-nums; color:var(--ink); }
  @media(max-width:560px) { .player-statline-stats { grid-template-columns:repeat(3, 1fr); row-gap:14px; } }

  /* FB+FD comparison panel */
  .fbfd-wrap { background:white; border-radius:14px; padding:18px 22px; box-shadow:0 4px 24px #0000000a; margin:14px 0 4px; }
  .fbfd-title { font-family:'Plus Jakarta Sans',sans-serif; font-size:0.858rem; font-weight:800; letter-spacing:.04em; color:var(--ink); padding-bottom:9px; border-bottom:1px solid #f0eaf4; margin-bottom:14px; }
  .fbfd-grid { display:grid; grid-template-columns:1fr 1fr; gap:18px; }
  .fbfd-col { display:flex; flex-direction:column; gap:8px; padding:12px 14px; border-radius:10px; background:#faf6fd; }
  .fbfd-col-header { font-family:'Plus Jakarta Sans',sans-serif; font-size:0.682rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; color:var(--soft); text-align:center; }
  .fbfd-col-sub { font-size:.7rem; color:var(--soft); font-weight:400; text-align:center; font-style:italic; margin-top:-4px; margin-bottom:4px; }
  .fbfd-row { display:flex; justify-content:space-between; align-items:baseline; padding:6px 8px; border-bottom:1px solid #ece5f3; }
  .fbfd-row:last-child { border-bottom:none; }
  .fbfd-row-label { font-size:.78rem; color:var(--ink); font-weight:500; }
  .fbfd-row-val { font-family:'Plus Jakarta Sans',sans-serif; font-size:1.43rem; font-weight:800; color:#ED1C7C; font-variant-numeric:tabular-nums; }
  .fbfd-baseline { font-size:.72rem; color:var(--soft); font-weight:300; font-style:italic; text-align:center; margin-top:10px; }
  @media(max-width:560px) { .fbfd-grid { grid-template-columns:1fr; } .rating-heatmap th.player-col { min-width:74px; } }

  .content em { font-style:italic; }
  .content a { color:var(--ink); font-weight:400; }
  .content a:hover { opacity:.7; }
  .content blockquote { margin:20px 0 24px; padding:14px 22px; border-left:4px solid #d4b8f4; background:#faf6fd; border-radius:0 12px 12px 0; }
  .content blockquote p { margin:0; font-size:1rem; font-weight:300; font-style:italic; line-height:1.75; color:var(--ink); }
  .rs-chart-card { background:white; border-radius:16px; padding:22px 26px 18px; box-shadow:0 4px 24px #0000000a; margin:14px -105px 32px; }
  @media(max-width:1120px) { .rs-chart-card { margin-left:0; margin-right:0; } }
  .rs-chart-title { font-family:'Plus Jakarta Sans',sans-serif; font-size:1.21rem; font-weight:800; text-transform:uppercase; letter-spacing:.02em; color:var(--ink); text-align:center; margin-bottom:10px; }
  .rs-legend { display:flex; justify-content:center; gap:20px; flex-wrap:wrap; margin-bottom:16px; }
  .rs-leg { display:inline-flex; align-items:center; gap:7px; font-family:'DM Sans',sans-serif; font-size:.8rem; font-weight:600; color:var(--ink); }
  .rs-sw { width:14px; height:14px; border-radius:4px; display:inline-block; flex-shrink:0; }
  .rs-chart-wrap { position:relative; height:460px; }

  /* Playoff bracket — proper round-to-round vertical centering like VLR */
  .br-wrap { background:white; border-radius:16px; padding:22px 24px; box-shadow:0 4px 24px #0000000a; margin:14px -105px 16px; overflow-x:auto; }
  @media(max-width:1120px) { .br-wrap { margin-left:0; margin-right:0; } }
  .br-side-label { font-family:'Plus Jakarta Sans',sans-serif; font-size:0.902rem; font-weight:800; letter-spacing:.06em; text-transform:uppercase; color:var(--ink); margin:4px 0 8px; }
  .br-side-label.lower { margin-top:24px; }
  .br-caption { font-size:.72rem; color:var(--soft); font-style:italic; margin:2px 0 10px; }
  .br-flow { display:flex; align-items:stretch; min-width:740px; }
  .br-col { display:flex; flex-direction:column; flex:1; min-width:150px; padding:0 8px; }
  .br-round-label { font-family:'Plus Jakarta Sans',sans-serif; font-size:0.638rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase; color:var(--soft); text-align:center; margin-bottom:8px; height:13px; }
  .br-col-body { flex:1; display:flex; flex-direction:column; justify-content:space-around; gap:10px; }
  .br-match { position:relative; background:#faf6fd; border:1px solid #f0e8f8; border-radius:9px; }
  .br-team { display:flex; align-items:center; gap:8px; font-size:.8rem; font-weight:600; color:var(--soft); padding:5px 9px; }
  .br-team + .br-team { border-top:1px solid #ece3f5; }
  .br-team img { width:17px; height:17px; object-fit:contain; flex-shrink:0; }
  .br-team.win { color:var(--ink); font-weight:800; box-shadow:inset 3px 0 0 #7c3aed; border-radius:8px; }
  .br-team.br-blank { min-height:17px; }
  .br-clickable { cursor:pointer; transition:box-shadow .15s; }
  .br-clickable:hover { box-shadow:0 3px 14px rgba(124,58,237,.22); }
  /* Match card popup (mirrors the Modern Hub upcoming card) */
  .mc-pop { position:absolute; left:0; top:0; z-index:250; width:300px; background:white; border-radius:14px; box-shadow:0 14px 46px #00000026; border:1px solid #efe7f6; padding:14px 16px; text-align:left; opacity:0; visibility:hidden; pointer-events:none; transform:translate(-50%,-100%) scale(.97); transform-origin:bottom center; transition:opacity .15s ease, transform .15s cubic-bezier(.34,1.56,.64,1), visibility .15s; }
  .mc-pop.open { opacity:1; visibility:visible; pointer-events:auto; transform:translate(-50%,-100%) scale(1); }
  .mc-pop::after { content:''; position:absolute; top:100%; left:50%; transform:translateX(-50%); border:8px solid transparent; border-top-color:white; }
  .mc-head { display:flex; align-items:center; gap:10px; margin-bottom:7px; }
  .mc-team { display:flex; flex-direction:column; align-items:center; gap:3px; width:46px; flex-shrink:0; }
  .mc-team img { width:30px; height:30px; object-fit:contain; }
  .mc-team-name { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:0.858rem; }
  .mc-prob { flex:1; min-width:0; }
  .mc-prob-pcts { display:flex; justify-content:space-between; gap:10px; font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.012rem; color:var(--soft); white-space:nowrap; }
  .mc-prob-pcts .fav { color:#16a34a; }
  .mc-bar { height:8px; border-radius:99px; background:#ece4f6; overflow:hidden; margin-top:4px; }
  .mc-bar-a { height:100%; background:#7c3aed; }
  .mc-fmt { text-align:center; font-size:.6rem; text-transform:uppercase; letter-spacing:.06em; color:var(--soft); font-weight:700; margin-bottom:6px; }
  .mc-sec-lbl { font-family:'Plus Jakarta Sans',sans-serif; font-size:0.616rem; font-weight:800; text-transform:uppercase; letter-spacing:.07em; color:var(--soft); margin:9px 0 5px; }
  .mc-veto { display:flex; flex-direction:column; gap:3px; }
  .mc-veto-row { display:flex; align-items:center; gap:8px; font-size:.74rem; font-weight:600; color:var(--ink); }
  .mc-veto-row img { width:28px; height:18px; object-fit:cover; border-radius:3px; flex-shrink:0; }
  .mc-veto-tag { margin-left:auto; font-family:'Plus Jakarta Sans',sans-serif; font-size:0.572rem; font-weight:800; text-transform:uppercase; letter-spacing:.03em; padding:2px 6px; border-radius:5px; white-space:nowrap; }
  .mc-tag-ban { background:#fde8e8; color:#a51d1d; } .mc-tag-pick { background:#e0effb; color:#1e5a9e; } .mc-tag-dec { background:#f0ecf4; color:#555; }
  .mc-maps { width:100%; border-collapse:collapse; font-size:.72rem; }
  .mc-maps td { padding:3px 5px; border-top:1px solid #f3eef9; }
  .mc-maps td:first-child { font-weight:600; }
  .mc-maps .mc-mw { text-align:right; font-variant-numeric:tabular-nums; font-weight:700; color:var(--soft); }
  .br-team .br-tag { margin-left:auto; font-family:'Plus Jakarta Sans',sans-serif; font-size:0.572rem; font-weight:800; letter-spacing:.05em; color:#7c3aed; }
  .br-team.champ { background:linear-gradient(90deg,rgba(124,58,237,.10),transparent); }
  .br-col:not(:last-child) .br-match::after { content:''; position:absolute; right:-9px; top:50%; width:9px; height:2px; background:#e2d6f2; }
  .br-col:not(:first-child) .br-match::before { content:''; position:absolute; left:-9px; top:50%; width:9px; height:2px; background:#e2d6f2; }

  /* Top vs bottom half — one graphic, a curly brace per cluster to its average */
  .half-graphic { background:white; border-radius:14px; padding:24px 30px; box-shadow:0 4px 24px #0000000a; margin:14px auto 26px; width:fit-content; max-width:100%; display:flex; flex-direction:column; gap:24px; }
  .hg-cluster { display:flex; align-items:stretch; gap:14px; }
  .hg-cluster-lbl { font-family:'DM Sans',sans-serif; font-size:.6rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase; color:var(--soft); margin-bottom:8px; }
  .hg-teams { display:flex; flex-direction:column; gap:7px; width:180px; flex-shrink:0; }
  .hg-row { display:flex; align-items:center; gap:9px; font-family:'DM Sans',sans-serif; font-weight:800; font-size:.95rem; color:var(--ink); }
  .hg-row img { width:26px; height:26px; object-fit:contain; flex-shrink:0; }
  .hg-rt { margin-left:auto; font-variant-numeric:tabular-nums; }
  .hg-brace { width:20px; flex-shrink:0; color:#c4b2de; }
  .hg-avg { display:flex; align-items:center; justify-content:center; min-width:104px; }
  .hg-avg-inner { position:relative; text-align:center; }
  .hg-avg-lbl { position:absolute; bottom:100%; left:0; right:0; margin-bottom:3px; white-space:nowrap; font-size:.64rem; color:var(--soft); font-weight:700; text-transform:uppercase; letter-spacing:.06em; }
  .hg-avg-num { font-family:'DM Sans',sans-serif; font-weight:800; font-size:2.3rem; font-variant-numeric:tabular-nums; line-height:1.05; }
  .hg-avg-num.strong { color:#16a34a; } .hg-avg-num.weak { color:#a33247; }
  .benpom-pick { background:#f5eefe; border:1px solid #e4d4fb; border-radius:12px; padding:14px 20px; margin:6px 0 26px; font-size:.95rem; font-weight:300; line-height:1.6; color:var(--ink); }
  .benpom-pick strong { font-weight:800; }
  .my-pred-head { font-family:'Plus Jakarta Sans',sans-serif; font-size:1.43rem; font-weight:800; letter-spacing:-.5px; color:var(--ink); margin:30px 0 4px; }
  /* Dot hover tooltip — copied from the Modern Hub (#dotTooltip). */
  #rsPopup{position:absolute;z-index:20;pointer-events:none;min-width:280px;max-width:380px;background:#1a0938;border:1px solid rgba(167,139,250,.28);border-radius:16px;padding:20px 24px;box-shadow:0 16px 60px rgba(0,0,0,.7);opacity:0;transform:translateY(8px);transition:opacity .18s ease,transform .18s ease}
  #rsPopup.visible{opacity:1;transform:translateY(0)}
  #rsPopup .popup-inner{text-align:center}
  #rsPopup .popup-event-label{font-size:.65rem;font-weight:600;color:rgba(167,139,250,.5);text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px}
  #rsPopup .popup-teams{display:flex;align-items:center;justify-content:center;gap:16px;margin-bottom:8px}
  #rsPopup .popup-team-block{display:flex;flex-direction:column;align-items:center;gap:5px;min-width:60px}
  #rsPopup .popup-logo{width:44px;height:44px;object-fit:contain}
  #rsPopup .popup-team-name{font-size:.7rem;color:rgba(232,213,245,.6);font-weight:500}
  #rsPopup .popup-score-block{display:flex;flex-direction:column;align-items:center;gap:3px}
  #rsPopup .popup-score{font-size:2.09rem;font-weight:800;font-family:'Plus Jakarta Sans',sans-serif;line-height:1}
  #rsPopup .popup-score.w{color:#4ade80}#rsPopup .popup-score.l{color:#f87171}
  #rsPopup .popup-vs-label{font-size:.65rem;color:rgba(232,213,245,.3)}
  #rsPopup .popup-date{color:rgba(232,213,245,.3);font-size:.68rem;margin-bottom:4px}
  #rsPopup .popup-delta{font-size:.85rem;font-weight:600;margin-bottom:14px}
  #rsPopup .popup-delta.pos{color:#4ade80}#rsPopup .popup-delta.neg{color:#f87171}
  #rsPopup .popup-maps-table{width:100%;border-collapse:collapse;margin-top:2px}
  #rsPopup .popup-maps-table th{font-size:.6rem;font-weight:600;color:rgba(167,139,250,.5);text-transform:uppercase;letter-spacing:.07em;padding:0 6px 6px;text-align:center}
  #rsPopup .popup-maps-table th:first-child{text-align:left}
  #rsPopup .popup-maps-table th:last-child{text-align:right}
  #rsPopup .popup-maps-table td{padding:5px 6px;font-size:.78rem;color:rgba(232,213,245,.8);border-top:1px solid rgba(255,255,255,.06)}
  #rsPopup .popup-map-name{font-weight:500;color:#e8d5f5}
  #rsPopup .popup-map-score{font-variant-numeric:tabular-nums;font-weight:600;text-align:center}
  #rsPopup .popup-map-score.w{color:#4ade80}#rsPopup .popup-map-score.l{color:#f87171}
  #rsPopup .popup-map-diff{text-align:right;font-size:.7rem;color:rgba(232,213,245,.4)}

  /* Clickable "BenPom" term with a popover bubble */
  .benpom-term { position:relative; color:#7c3aed; font-weight:600; cursor:pointer; border-bottom:1.5px dotted #7c3aed; }
  .benpom-pop { position:absolute; left:0; top:0; width:268px; background:white; border-radius:14px; box-shadow:0 10px 34px #00000026; padding:15px 17px; z-index:300; text-align:center; cursor:default; opacity:0; visibility:hidden; pointer-events:none; transform:translate(-50%,-100%) scale(.96); transform-origin:bottom center; transition:opacity .17s ease, transform .17s cubic-bezier(.34,1.56,.64,1), visibility .17s; }
  .benpom-pop.open { opacity:1; visibility:visible; pointer-events:auto; transform:translate(-50%,-100%) scale(1); }
  .benpom-pop::after { content:''; position:absolute; top:100%; left:50%; transform:translateX(-50%); border:8px solid transparent; border-top-color:white; }
  .benpom-pop-text { display:block; font-family:'DM Sans',sans-serif; font-size:.82rem; font-weight:300; line-height:1.55; color:var(--ink); margin-bottom:12px; }
  .benpom-pop-links { display:flex; flex-direction:column; gap:8px; }
  .benpom-pop-links a { display:block; font-family:'Plus Jakarta Sans',sans-serif; font-size:0.792rem; font-weight:800; letter-spacing:.03em; color:#7c3aed; text-decoration:none; padding:8px 10px; border-radius:9px; background:#f5eefe; transition:background .15s; }
  .benpom-pop-links a:hover { background:#e9dcfb; opacity:1; }
  .data-table .pos { color:#1a6a4a; font-weight:700; }
  .data-table .neg { color:#a33247; font-weight:700; }
  footer { position:relative; z-index:1; text-align:center; padding:24px; color:var(--soft); font-size:.75rem; font-weight:300; }
  @keyframes fadeUp{from{opacity:0;transform:translateY(24px)}to{opacity:1;transform:translateY(0)}}
  .page { animation:fadeUp .6s ease both; }
  /* Bracket: shrink internals so it fits the fixed-width mobile viewport
     with NO horizontal scroll (fires at width=820). */
  @media (max-width:1000px){
    .content p, .content ul li, .content ol.numbered li, .content blockquote p { font-size:1.3rem; }
    .br-flow { min-width:660px; }
    .br-col { min-width:120px; padding:0 6px; }
    .br-team { font-size:.72rem; padding:5px 7px; gap:6px; }
    .br-team img { width:14px; height:14px; }
    .br-round-label { font-size:.58rem; }
  }
  /* ── Mobile (phone) ── */
  @media (max-width:600px){
    .page { padding:24px 16px 56px; }
    .content p, .content ul li, .content ol.numbered li { font-size:.94rem; }
    .data-table-wrap { padding:16px 14px; }
    .data-table { font-size:.82rem; min-width:520px; }
    .data-table th, .data-table td { padding:7px 7px; }
    .data-table .rank { width:42px; }
    .data-table .team-logo { width:20px; height:20px; }
    .data-table-label { font-size:1.02rem; }
  }
</style>
</head>
<body>
<nav class="toc">
  <div class="toc-title">Sections</div>
  <a href="#introduction">Introduction</a>
  <a href="#benpommania">BenPomMania!</a>
  <a href="#bracket-predictions">Bracket-Based Predictions</a>
</nav>
<div class="top-nav">
  <a href="/"><img src="/logo.svg" alt="Home" class="home-logo"></a>
</div>
<div class="page">
  <div class="article">
    <div class="label">Research / Opinion</div>
    <h1>Masters London<br>Playoffs Preview</h1>
    <div class="byline">Bobo &mdash; June 2026</div>
    <div class="cover">
      <img src="/chronlondon.jpg" alt="Masters London playoffs">
    </div>
    <p class="cover-caption">Chronicle pictured with his Vitality teammates amidst their 2&ndash;0 path through the Swiss Stage; a win at London would mean Chronicle&rsquo;s 4th Masters title.</p>
    <div class="content">
      <div class="section-bubble-wrap section-bubble-tight"><span class="section-bubble" id="introduction"><span class="section-bubble-text">Introduction</span></span></div>

      <p>Masters London&rsquo;s Swiss Stage has finished, and it has been nothing if not historic and shocking. So far:</p>

      <ul>
        <li>The region that was thought of as the strongest heading into Masters London, Pacific, had both of their Swiss Stage teams fail to qualify to Playoffs.</li>
        <li>We saw the second-highest VLR rating in an international match of all time, with Keiko&rsquo;s 1.85 rating against XLG. He was 0.01 away from tying the record.</li>
      </ul>

      <div class="player-statline">
        <div class="player-statline-head">
          <img class="headshot" src="https://owcdn.net/img/697406c5ecbcd.png" alt="Keiko" onerror="this.style.visibility='hidden'">
          <div>
            <div class="player-statline-name">Keiko &middot; NRG</div>
            <div class="player-statline-sub">Masters London 2026, Swiss Stage Round 1 vs XLG &mdash; Series (2&ndash;0)</div>
          </div>
        </div>
        <div class="player-statline-stats">
          <div class="player-statline-stat"><div class="player-statline-stat-label">R2.0</div><div class="player-statline-stat-value">1.85</div></div>
          <div class="player-statline-stat"><div class="player-statline-stat-label">K / D / A</div><div class="player-statline-stat-value">60/29/8</div></div>
          <div class="player-statline-stat"><div class="player-statline-stat-label">KAST</div><div class="player-statline-stat-value">79%</div></div>
          <div class="player-statline-stat"><div class="player-statline-stat-label">FK / FD</div><div class="player-statline-stat-value">7 / 1</div></div>
        </div>
      </div>

      <ul>
        <li>Since franchising, only two CN teams have ever gotten past the Swiss/Groups Stage of a Masters: EDG at Tokyo and Wolves at Toronto. That number is now 3, as XLG (narrowly) made it into Playoffs.</li>
        <li>The two teams with the highest <span class="benpom-term" onclick="toggleBenPomPop(event, this)">BenPom</span> ratings coming into Swiss Stage (Full Sense and NRG) both failed to qualify for Playoffs.</li>
      </ul>

      <p>I could mention other items and discuss with greater depth, but that&rsquo;s not the point of this preview. Between the limited turnaround time between Swiss and Playoffs and the fact that I&rsquo;ve already done a complete overview of every team at Masters London (which you can read <a href="/articles/masters-london-preview/" target="_blank" rel="noopener">here</a>), this will be a succinct overview of Playoffs focusing mainly on what the numbers say. Still, with such upheaval, it&rsquo;ll be hard not to touch on a few narratives.</p>

      <div class="section-bubble-wrap"><span class="section-bubble" id="benpommania"><span class="section-bubble-text">BenPomMania!</span></span></div>

      <p>With the 8 Playoff teams for Masters London locked in, let&rsquo;s look at how they rank according to <span class="benpom-term" onclick="toggleBenPomPop(event, this)">BenPom</span>:</p>

      <div class="data-table-wrap">
        <div class="data-table-label">Masters London Playoff Teams &mdash; BenPom Rankings</div>
        <table class="data-table">
          <thead>
            <tr>
              <th class="rank">Global Rank</th>
              <th>Team</th>
              <th>Region</th>
              <th class="num">BenPom</th>
              <th class="num">&Delta; vs Pre-London</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td class="rank">1</td>
              <td class="team"><div class="team-cell"><img class="team-logo" src="/logos/TH.png" alt="TH" onerror="this.style.display='none'">Team Heretics</div></td>
              <td>EMEA</td>
              <td class="num">+3.65</td>
              <td class="num pos">+1.16</td>
            </tr>
            <tr>
              <td class="rank">3</td>
              <td class="team"><div class="team-cell"><img class="team-logo" src="/logos/G2.png" alt="G2" onerror="this.style.display='none'">G2 Esports</div></td>
              <td>Americas</td>
              <td class="num">+3.17</td>
              <td class="num neg">&minus;0.18</td>
            </tr>
            <tr>
              <td class="rank">4</td>
              <td class="team"><div class="team-cell"><img class="team-logo" src="/logos/PRX.png" alt="PRX" onerror="this.style.display='none'">Paper Rex</div></td>
              <td>Pacific</td>
              <td class="num">+3.16</td>
              <td class="num neg">&minus;0.64</td>
            </tr>
            <tr>
              <td class="rank">5</td>
              <td class="team"><div class="team-cell"><img class="team-logo" src="/logos/VIT.png" alt="VIT" onerror="this.style.display='none'">Team Vitality</div></td>
              <td>EMEA</td>
              <td class="num">+3.15</td>
              <td class="num pos">+1.28</td>
            </tr>
            <tr>
              <td class="rank">7</td>
              <td class="team"><div class="team-cell"><img class="team-logo" src="/logos/FUT.png" alt="FUT" onerror="this.style.display='none'">FUT Esports</div></td>
              <td>EMEA</td>
              <td class="num">+2.59</td>
              <td class="num pos">+1.98</td>
            </tr>
            <tr>
              <td class="rank">10</td>
              <td class="team"><div class="team-cell"><img class="team-logo" src="/logos/LEV.png" alt="LEV" onerror="this.style.display='none'">Leviatán</div></td>
              <td>Americas</td>
              <td class="num">+2.28</td>
              <td class="num neg">&minus;0.04</td>
            </tr>
            <tr>
              <td class="rank">19</td>
              <td class="team"><div class="team-cell"><img class="team-logo" src="/logos/EDG.png" alt="EDG" onerror="this.style.display='none'">EDward Gaming</div></td>
              <td>China</td>
              <td class="num">+0.77</td>
              <td class="num neg">&minus;0.18</td>
            </tr>
            <tr>
              <td class="rank">23</td>
              <td class="team"><div class="team-cell"><img class="team-logo" src="/logos/XLG.png" alt="XLG" onerror="this.style.display='none'">Xi Lai Gaming</div></td>
              <td>China</td>
              <td class="num">+0.38</td>
              <td class="num neg">&minus;0.20</td>
            </tr>
          </tbody>
        </table>
      </div>

      <p>For all of the talk about how Paper Rex were heavy favorites for Masters London, Swiss Stage has knocked them out of the number 1 <span class="benpom-term" onclick="toggleBenPomPop(event, this)">BenPom</span> spot. More specifically, Pacific&rsquo;s middling performance in Swiss Stage has knocked Paper Rex down.</p>

      <p>I know I said I would avoid the narratives, but it&rsquo;s really shocking how bad Pacific was. Coming into Swiss Stage, Full Sense looked confidently like the second-best team in Pacific with a 2.47 rating (the 7th best in the world). They were the most picked team to make it through Swiss flawlessly on Riot&rsquo;s pick&rsquo;ems. Then, not only did they go out 0&ndash;2, but they got STOMPED by the EMEA 3-seed, FUT (&minus;16 round differential over two maps). FS&rsquo;s other loss was closer, but it was against the other Pacific team, Global Esports. As for GE, they ended up getting eliminated by XLG, the third Chinese team ever to make it past Groups at a Masters event. Again, Pacific were historically bad in Swiss.</p>

      <p>The end result is not only PRX dropping out of the number 1 spot, but actually dropping two spots all the way to 3rd without playing a single match. I can&rsquo;t disagree with BenPom here.</p>

      <p>Importantly, for every action there is an equal and opposite reaction. In this way, BenPom doesn&rsquo;t have losers without winners. For all of Pacific&rsquo;s hullabaloo, EMEA is the region that shot up.</p>

      <p>While Full Sense were expected to be great and were horrible, FUT were expected to be horrible and were great. As a certain (incorrect) VCT analyst wrote:</p>

      <blockquote><p>[FUT] have lost 3 of their past 4 matches, they come from the second-worst region, their best map in Stage 1 (Bind) isn&rsquo;t in the map pool anymore, and their best player (s0pp) has only played Neon in Stage 1, who&rsquo;s getting nerfed into the ground&hellip; this team is hard to be optimistic about.</p></blockquote>

      <p>That&rsquo;s a quote from my <a href="/articles/masters-london-preview/" target="_blank" rel="noopener">Masters London Preview</a>; how wrong I was. FUT did the following:</p>

      <ol class="numbered">
        <li>Clobbered what was supposed to be Pacific&rsquo;s best team in the Swiss Stage &mdash; Full Sense &mdash; as discussed previously (re: 16-round differential over two maps).</li>
        <li>Narrowly lost to their EMEA counterpart Vitality.</li>
        <li>Narrowly beat what was supposed to be America&rsquo;s best team in the Swiss Stage &mdash; NRG.<span style="display:block;margin-top:6px"><strong>Note:</strong> <em>Emphasis on the &ldquo;Narrowly&rdquo;, as the map scores were as follows: 11&ndash;13, 14&ndash;12, 13&ndash;11.</em></span></li>
      </ol>

      <p>Vitality, on the other hand, clobbered DRG and then narrowly beat FUT.</p>

      <p>If the EMEA teams beat every non-EMEA team they played in Swiss Stage, no matter how strong/weak their opponents were, I can&rsquo;t disagree with Heretics (the EMEA 1-seed) rising up to the number 1 spot. Notably, BenPom predicts that Heretics will be a cut above the rest. They have a 3.65 rating while G2, PRX, and VIT are clustered with 3.17, 3.16, and 3.15 ratings respectively.</p>

      <p>The regional shift in BenPom Ratings is visualized below:</p>

      <div class="rs-chart-card">
        <div class="rs-chart-title">Playoff Teams &mdash; BenPom Through Masters London</div>
        <div class="rs-legend">
          <span class="rs-leg"><span class="rs-sw" style="background:#16a34a"></span>EMEA</span>
          <span class="rs-leg"><span class="rs-sw" style="background:#ea580c"></span>Americas</span>
          <span class="rs-leg"><span class="rs-sw" style="background:#2563eb"></span>Pacific</span>
          <span class="rs-leg"><span class="rs-sw" style="background:#db2777"></span>China</span>
        </div>
        <div class="rs-chart-wrap"><canvas id="regionShiftChart"></canvas><div id="rsPopup"><div class="popup-inner" id="rsPopupContent"></div></div></div>
      </div>

      <div class="section-bubble-wrap"><span class="section-bubble" id="bracket-predictions"><span class="section-bubble-text">Bracket-Based Predictions</span></span></div>

      <p>However, it&rsquo;s one thing to be the best team, and it&rsquo;s another to be in the best situation to win. Now that we have a bracket, we can take that into account. The Playoffs bracket looks as follows:</p>

      <div class="br-wrap">
        <div class="br-side-label">Upper Bracket</div>
        <div class="br-flow">
          <div class="br-col"><div class="br-round-label">Quarterfinals</div><div class="br-col-body"><div class="br-match"><div class="br-team"><img src="/logos/G2.png" onerror="this.style.display='none'"><span>G2</span></div><div class="br-team"><img src="/logos/XLG.png" onerror="this.style.display='none'"><span>XLG</span></div></div><div class="br-match"><div class="br-team"><img src="/logos/EDG.png" onerror="this.style.display='none'"><span>EDG</span></div><div class="br-team"><img src="/logos/FUT.png" onerror="this.style.display='none'"><span>FUT</span></div></div><div class="br-match"><div class="br-team"><img src="/logos/PRX.png" onerror="this.style.display='none'"><span>PRX</span></div><div class="br-team"><img src="/logos/LEV.png" onerror="this.style.display='none'"><span>LEV</span></div></div><div class="br-match"><div class="br-team"><img src="/logos/TH.png" onerror="this.style.display='none'"><span>TH</span></div><div class="br-team"><img src="/logos/VIT.png" onerror="this.style.display='none'"><span>VIT</span></div></div></div></div>
          <div class="br-col"><div class="br-round-label">Semifinals</div><div class="br-col-body"><div class="br-match"><div class="br-team br-blank">&nbsp;</div><div class="br-team br-blank">&nbsp;</div></div><div class="br-match"><div class="br-team br-blank">&nbsp;</div><div class="br-team br-blank">&nbsp;</div></div></div></div>
          <div class="br-col"><div class="br-round-label">Upper Final</div><div class="br-col-body"><div class="br-match"><div class="br-team br-blank">&nbsp;</div><div class="br-team br-blank">&nbsp;</div></div></div></div>
          <div class="br-col"><div class="br-round-label">Grand Final</div><div class="br-col-body"><div class="br-match"><div class="br-team br-blank">&nbsp;</div><div class="br-team br-blank">&nbsp;</div></div></div></div>
        </div>
        <div class="br-side-label lower">Lower Bracket</div>
        <div class="br-flow">
          <div class="br-col"><div class="br-round-label">Round 1</div><div class="br-col-body"><div class="br-match"><div class="br-team br-blank">&nbsp;</div><div class="br-team br-blank">&nbsp;</div></div><div class="br-match"><div class="br-team br-blank">&nbsp;</div><div class="br-team br-blank">&nbsp;</div></div></div></div>
          <div class="br-col"><div class="br-round-label">Round 2</div><div class="br-col-body"><div class="br-match"><div class="br-team br-blank">&nbsp;</div><div class="br-team br-blank">&nbsp;</div></div><div class="br-match"><div class="br-team br-blank">&nbsp;</div><div class="br-team br-blank">&nbsp;</div></div></div></div>
          <div class="br-col"><div class="br-round-label">Round 3</div><div class="br-col-body"><div class="br-match"><div class="br-team br-blank">&nbsp;</div><div class="br-team br-blank">&nbsp;</div></div></div></div>
          <div class="br-col"><div class="br-round-label">Lower Final</div><div class="br-col-body"><div class="br-match"><div class="br-team br-blank">&nbsp;</div><div class="br-team br-blank">&nbsp;</div></div></div></div>
        </div>
      </div>

      <p>Immediately, it&rsquo;s clear the bottom half is stronger than the top half. <span class="benpom-term" onclick="toggleBenPomPop(event, this)">BenPom</span> affirm this.</p>

      <div class="half-graphic">
        <div>
          <div class="hg-cluster-lbl">Top Half &mdash; Upper SF 1</div>
          <div class="hg-cluster">
            <div class="hg-teams">
              <div class="hg-row"><img src="/logos/G2.png" onerror="this.style.display='none'"><span>G2</span><span class="hg-rt">+3.17</span></div>
              <div class="hg-row"><img src="/logos/XLG.png" onerror="this.style.display='none'"><span>XLG</span><span class="hg-rt">+0.38</span></div>
              <div class="hg-row"><img src="/logos/EDG.png" onerror="this.style.display='none'"><span>EDG</span><span class="hg-rt">+0.77</span></div>
              <div class="hg-row"><img src="/logos/FUT.png" onerror="this.style.display='none'"><span>FUT</span><span class="hg-rt">+2.59</span></div>
            </div>
            <svg class="hg-brace" viewBox="0 0 20 100" preserveAspectRatio="none"><path d="M7,2 Q10,2 10,13 L10,41 Q10,50 17,50 Q10,50 10,59 L10,87 Q10,98 7,98" fill="none" stroke="currentColor" stroke-width="2" vector-effect="non-scaling-stroke"/></svg>
            <div class="hg-avg"><div class="hg-avg-inner"><div class="hg-avg-lbl">Avg BenPom</div><div class="hg-avg-num weak">+1.73</div></div></div>
          </div>
        </div>
        <div>
          <div class="hg-cluster-lbl">Bottom Half &mdash; Upper SF 2</div>
          <div class="hg-cluster">
            <div class="hg-teams">
              <div class="hg-row"><img src="/logos/PRX.png" onerror="this.style.display='none'"><span>PRX</span><span class="hg-rt">+3.16</span></div>
              <div class="hg-row"><img src="/logos/LEV.png" onerror="this.style.display='none'"><span>LEV</span><span class="hg-rt">+2.28</span></div>
              <div class="hg-row"><img src="/logos/TH.png" onerror="this.style.display='none'"><span>TH</span><span class="hg-rt">+3.65</span></div>
              <div class="hg-row"><img src="/logos/VIT.png" onerror="this.style.display='none'"><span>VIT</span><span class="hg-rt">+3.15</span></div>
            </div>
            <svg class="hg-brace" viewBox="0 0 20 100" preserveAspectRatio="none"><path d="M7,2 Q10,2 10,13 L10,41 Q10,50 17,50 Q10,50 10,59 L10,87 Q10,98 7,98" fill="none" stroke="currentColor" stroke-width="2" vector-effect="non-scaling-stroke"/></svg>
            <div class="hg-avg"><div class="hg-avg-inner"><div class="hg-avg-lbl">Avg BenPom</div><div class="hg-avg-num strong">+3.06</div></div></div>
          </div>
        </div>
      </div>

      <p>Using the current <span class="benpom-term" onclick="toggleBenPomPop(event, this)">BenPom</span> ratings, I simulated the bracket 50,000 times and got the following probabilities:</p>

      <div class="data-table-wrap">
        <div class="data-table-label">Masters London Playoffs &mdash; 50,000 Simulations</div>
        <table class="data-table">
          <thead><tr><th>Team</th><th>Region</th><th class="num">Win Masters London</th><th class="num">Top 3</th><th class="num">Out 0&ndash;2</th></tr></thead>
          <tbody>
            <tr><td class="team"><div class="team-cell"><img class="team-logo" src="/logos/G2.png" onerror="this.style.display='none'">G2 Esports</div></td><td>Americas</td><td class="num">21.3%</td><td class="num">58.0%</td><td class="num">8.3%</td></tr>
            <tr><td class="team"><div class="team-cell"><img class="team-logo" src="/logos/TH.png" onerror="this.style.display='none'">Team Heretics</div></td><td>EMEA</td><td class="num">20.8%</td><td class="num">48.0%</td><td class="num">20.6%</td></tr>
            <tr><td class="team"><div class="team-cell"><img class="team-logo" src="/logos/PRX.png" onerror="this.style.display='none'">Paper Rex</div></td><td>Pacific</td><td class="num">16.3%</td><td class="num">44.2%</td><td class="num">23.0%</td></tr>
            <tr><td class="team"><div class="team-cell"><img class="team-logo" src="/logos/VIT.png" onerror="this.style.display='none'">Team Vitality</div></td><td>EMEA</td><td class="num">15.3%</td><td class="num">41.3%</td><td class="num">24.6%</td></tr>
            <tr><td class="team"><div class="team-cell"><img class="team-logo" src="/logos/FUT.png" onerror="this.style.display='none'">FUT Esports</div></td><td>EMEA</td><td class="num">14.9%</td><td class="num">49.5%</td><td class="num">10.7%</td></tr>
            <tr><td class="team"><div class="team-cell"><img class="team-logo" src="/logos/LEV.png" onerror="this.style.display='none'">Leviat&aacute;n</div></td><td>Americas</td><td class="num">9.1%</td><td class="num">32.0%</td><td class="num">31.9%</td></tr>
            <tr><td class="team"><div class="team-cell"><img class="team-logo" src="/logos/EDG.png" onerror="this.style.display='none'">EDward Gaming</div></td><td>China</td><td class="num">1.4%</td><td class="num">14.7%</td><td class="num">37.5%</td></tr>
            <tr><td class="team"><div class="team-cell"><img class="team-logo" src="/logos/XLG.png" onerror="this.style.display='none'">Xi Lai Gaming</div></td><td>China</td><td class="num">1.0%</td><td class="num">12.4%</td><td class="num">43.5%</td></tr>
          </tbody>
        </table>
      </div>

      <p>Immediately, the effects of regional shifts and bracket disparity are visible. Despite being in the bottom half of Playoffs teams by <span class="benpom-term" onclick="toggleBenPomPop(event, this)">BenPom</span> rating, FUT are given a ~50% chance to finish top 3 because of their path through the top half of the bracket. Similarly, G2 have the greatest chance to win Masters London despite not entering the tournament as the favorites (re: PRX) or being the highest-rated team currently (re: Team Heretics). Both of those teams have to survive a gauntlet to even get to the upper finals.</p>

      <p>Also, BenPom&rsquo;s pessimism on China continues. EDG at 1.4% to win Masters London is probably the result I would disagree with the most, but it&rsquo;s not my job to alter the data to agree with me, instead it&rsquo;s just to react to it. This pessimism is despite XLG making it into playoffs (i.e. promoting confidence in the CN region). In fairness, DRG did get crushed by Vitality and XLG got crushed by NRG. Given EDG&rsquo;s similar power level to XLG based on domestic results, it makes sense to be doubtful of EDG&rsquo;s ability to play at the level of higher-seeded teams than Vitality and NRG. Call it bias, but I have enough faith in EDG&rsquo;s player quality to win Masters London more than 1.4% of the time. You can come back to this when I&rsquo;m inevitably wrong.</p>

      <p>On the topic of regional commentary, it&rsquo;s interesting that BenPom predicts that the Masters London winner will be from EMEA 51% of the time. Having 3 teams in a group of 8 teams helps, let alone those 3 teams being top-5 teams by BenPom.</p>

      <p>To finish off this quick preview, I&rsquo;ll let BenPom simulate the bracket out where it advances its favorite for each match. Click on any match to see the projected probability, veto, and individual map favorites. After that, I&rsquo;ll fill out a bracket with my prediction. Check back later to see who was more accurate!</p>

      <div class="my-pred-head">BenPom&rsquo;s Prediction</div>

      <div class="br-wrap">
        <div class="br-side-label">Upper Bracket</div>
        <div class="br-flow">
          <div class="br-col">
            <div class="br-round-label">Quarterfinals</div>
            <div class="br-col-body">
              <div class="br-match br-clickable" data-match="uqf1" onclick="showMatchCard(event, this)"><div class="br-team win"><img src="/logos/G2.png" onerror="this.style.display='none'"><span>G2</span></div><div class="br-team"><img src="/logos/XLG.png" onerror="this.style.display='none'"><span>XLG</span></div></div>
              <div class="br-match br-clickable" data-match="uqf2" onclick="showMatchCard(event, this)"><div class="br-team"><img src="/logos/EDG.png" onerror="this.style.display='none'"><span>EDG</span></div><div class="br-team win"><img src="/logos/FUT.png" onerror="this.style.display='none'"><span>FUT</span></div></div>
              <div class="br-match br-clickable" data-match="uqf3" onclick="showMatchCard(event, this)"><div class="br-team win"><img src="/logos/PRX.png" onerror="this.style.display='none'"><span>PRX</span></div><div class="br-team"><img src="/logos/LEV.png" onerror="this.style.display='none'"><span>LEV</span></div></div>
              <div class="br-match br-clickable" data-match="uqf4" onclick="showMatchCard(event, this)"><div class="br-team"><img src="/logos/TH.png" onerror="this.style.display='none'"><span>TH</span></div><div class="br-team win"><img src="/logos/VIT.png" onerror="this.style.display='none'"><span>VIT</span></div></div>
            </div>
          </div>
          <div class="br-col">
            <div class="br-round-label">Semifinals</div>
            <div class="br-col-body">
              <div class="br-match br-clickable" data-match="usf1" onclick="showMatchCard(event, this)"><div class="br-team win"><img src="/logos/G2.png" onerror="this.style.display='none'"><span>G2</span></div><div class="br-team"><img src="/logos/FUT.png" onerror="this.style.display='none'"><span>FUT</span></div></div>
              <div class="br-match br-clickable" data-match="usf2" onclick="showMatchCard(event, this)"><div class="br-team win"><img src="/logos/PRX.png" onerror="this.style.display='none'"><span>PRX</span></div><div class="br-team"><img src="/logos/VIT.png" onerror="this.style.display='none'"><span>VIT</span></div></div>
            </div>
          </div>
          <div class="br-col">
            <div class="br-round-label">Upper Final</div>
            <div class="br-col-body">
              <div class="br-match br-clickable" data-match="uf" onclick="showMatchCard(event, this)"><div class="br-team"><img src="/logos/G2.png" onerror="this.style.display='none'"><span>G2</span></div><div class="br-team win"><img src="/logos/PRX.png" onerror="this.style.display='none'"><span>PRX</span></div></div>
            </div>
          </div>
          <div class="br-col">
            <div class="br-round-label">Grand Final</div>
            <div class="br-col-body">
              <div class="br-match br-clickable" data-match="gf" onclick="showMatchCard(event, this)"><div class="br-team win champ"><img src="/logos/PRX.png" onerror="this.style.display='none'"><span>PRX</span><span class="br-tag">CHAMPION</span></div><div class="br-team"><img src="/logos/G2.png" onerror="this.style.display='none'"><span>G2</span></div></div>
            </div>
          </div>
        </div>
        <div class="br-side-label lower">Lower Bracket</div>
        <div class="br-flow">
          <div class="br-col">
            <div class="br-round-label">Round 1</div>
            <div class="br-col-body">
              <div class="br-match br-clickable" data-match="lr1a" onclick="showMatchCard(event, this)"><div class="br-team win"><img src="/logos/XLG.png" onerror="this.style.display='none'"><span>XLG</span></div><div class="br-team"><img src="/logos/EDG.png" onerror="this.style.display='none'"><span>EDG</span></div></div>
              <div class="br-match br-clickable" data-match="lr1b" onclick="showMatchCard(event, this)"><div class="br-team"><img src="/logos/LEV.png" onerror="this.style.display='none'"><span>LEV</span></div><div class="br-team win"><img src="/logos/TH.png" onerror="this.style.display='none'"><span>TH</span></div></div>
            </div>
          </div>
          <div class="br-col">
            <div class="br-round-label">Round 2</div>
            <div class="br-col-body">
              <div class="br-match br-clickable" data-match="lr2a" onclick="showMatchCard(event, this)"><div class="br-team win"><img src="/logos/VIT.png" onerror="this.style.display='none'"><span>VIT</span></div><div class="br-team"><img src="/logos/XLG.png" onerror="this.style.display='none'"><span>XLG</span></div></div>
              <div class="br-match br-clickable" data-match="lr2b" onclick="showMatchCard(event, this)"><div class="br-team"><img src="/logos/FUT.png" onerror="this.style.display='none'"><span>FUT</span></div><div class="br-team win"><img src="/logos/TH.png" onerror="this.style.display='none'"><span>TH</span></div></div>
            </div>
          </div>
          <div class="br-col">
            <div class="br-round-label">Round 3</div>
            <div class="br-col-body">
              <div class="br-match br-clickable" data-match="lr3" onclick="showMatchCard(event, this)"><div class="br-team win"><img src="/logos/VIT.png" onerror="this.style.display='none'"><span>VIT</span></div><div class="br-team"><img src="/logos/TH.png" onerror="this.style.display='none'"><span>TH</span></div></div>
            </div>
          </div>
          <div class="br-col">
            <div class="br-round-label">Lower Final</div>
            <div class="br-col-body">
              <div class="br-match br-clickable" data-match="lf" onclick="showMatchCard(event, this)"><div class="br-team win"><img src="/logos/G2.png" onerror="this.style.display='none'"><span>G2</span></div><div class="br-team"><img src="/logos/VIT.png" onerror="this.style.display='none'"><span>VIT</span></div></div>
            </div>
          </div>
        </div>
      </div>

      <div class="my-pred-head">My Prediction</div>
      <div class="br-wrap">
        <div class="br-side-label">Upper Bracket</div>
        <div class="br-flow">
          <div class="br-col">
            <div class="br-round-label">Quarterfinals</div>
            <div class="br-col-body">
              <div class="br-match"><div class="br-team win"><img src="/logos/G2.png" onerror="this.style.display='none'"><span>G2</span></div><div class="br-team"><img src="/logos/XLG.png" onerror="this.style.display='none'"><span>XLG</span></div></div>
              <div class="br-match"><div class="br-team win"><img src="/logos/EDG.png" onerror="this.style.display='none'"><span>EDG</span></div><div class="br-team"><img src="/logos/FUT.png" onerror="this.style.display='none'"><span>FUT</span></div></div>
              <div class="br-match"><div class="br-team win"><img src="/logos/PRX.png" onerror="this.style.display='none'"><span>PRX</span></div><div class="br-team"><img src="/logos/LEV.png" onerror="this.style.display='none'"><span>LEV</span></div></div>
              <div class="br-match"><div class="br-team win"><img src="/logos/TH.png" onerror="this.style.display='none'"><span>TH</span></div><div class="br-team"><img src="/logos/VIT.png" onerror="this.style.display='none'"><span>VIT</span></div></div>
            </div>
          </div>
          <div class="br-col">
            <div class="br-round-label">Semifinals</div>
            <div class="br-col-body">
              <div class="br-match"><div class="br-team"><img src="/logos/G2.png" onerror="this.style.display='none'"><span>G2</span></div><div class="br-team win"><img src="/logos/EDG.png" onerror="this.style.display='none'"><span>EDG</span></div></div>
              <div class="br-match"><div class="br-team"><img src="/logos/PRX.png" onerror="this.style.display='none'"><span>PRX</span></div><div class="br-team win"><img src="/logos/TH.png" onerror="this.style.display='none'"><span>TH</span></div></div>
            </div>
          </div>
          <div class="br-col">
            <div class="br-round-label">Upper Final</div>
            <div class="br-col-body">
              <div class="br-match"><div class="br-team"><img src="/logos/EDG.png" onerror="this.style.display='none'"><span>EDG</span></div><div class="br-team win"><img src="/logos/TH.png" onerror="this.style.display='none'"><span>TH</span></div></div>
            </div>
          </div>
          <div class="br-col">
            <div class="br-round-label">Grand Final</div>
            <div class="br-col-body">
              <div class="br-match"><div class="br-team win champ"><img src="/logos/TH.png" onerror="this.style.display='none'"><span>TH</span><span class="br-tag">CHAMPION</span></div><div class="br-team"><img src="/logos/PRX.png" onerror="this.style.display='none'"><span>PRX</span></div></div>
            </div>
          </div>
        </div>
        <div class="br-side-label lower">Lower Bracket</div>
        <div class="br-flow">
          <div class="br-col">
            <div class="br-round-label">Round 1</div>
            <div class="br-col-body">
              <div class="br-match"><div class="br-team"><img src="/logos/XLG.png" onerror="this.style.display='none'"><span>XLG</span></div><div class="br-team win"><img src="/logos/FUT.png" onerror="this.style.display='none'"><span>FUT</span></div></div>
              <div class="br-match"><div class="br-team"><img src="/logos/LEV.png" onerror="this.style.display='none'"><span>LEV</span></div><div class="br-team win"><img src="/logos/VIT.png" onerror="this.style.display='none'"><span>VIT</span></div></div>
            </div>
          </div>
          <div class="br-col">
            <div class="br-round-label">Round 2</div>
            <div class="br-col-body">
              <div class="br-match"><div class="br-team win"><img src="/logos/PRX.png" onerror="this.style.display='none'"><span>PRX</span></div><div class="br-team"><img src="/logos/FUT.png" onerror="this.style.display='none'"><span>FUT</span></div></div>
              <div class="br-match"><div class="br-team win"><img src="/logos/G2.png" onerror="this.style.display='none'"><span>G2</span></div><div class="br-team"><img src="/logos/VIT.png" onerror="this.style.display='none'"><span>VIT</span></div></div>
            </div>
          </div>
          <div class="br-col">
            <div class="br-round-label">Round 3</div>
            <div class="br-col-body">
              <div class="br-match"><div class="br-team win"><img src="/logos/PRX.png" onerror="this.style.display='none'"><span>PRX</span></div><div class="br-team"><img src="/logos/G2.png" onerror="this.style.display='none'"><span>G2</span></div></div>
            </div>
          </div>
          <div class="br-col">
            <div class="br-round-label">Lower Final</div>
            <div class="br-col-body">
              <div class="br-match"><div class="br-team"><img src="/logos/EDG.png" onerror="this.style.display='none'"><span>EDG</span></div><div class="br-team win"><img src="/logos/PRX.png" onerror="this.style.display='none'"><span>PRX</span></div></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>
<footer>
  Data sourced from VLR.gg
  <div style="margin-top:8px;">Like my work? Tips are appreciated! <a href="https://ko-fi.com/bobovct" target="_blank" rel="noopener" style="color:inherit;text-decoration:underline;">ko-fi.com/bobovct</a></div>
</footer>
<script>
(function() {
  var tocLinks = document.querySelectorAll('.toc a');
  var ids = Array.from(tocLinks).map(function(a) { return a.getAttribute('href').slice(1); });
  function onScroll() {
    var y = window.scrollY + 120;
    var active = ids[0];
    ids.forEach(function(id) {
      var el = document.getElementById(id);
      if (el && el.offsetTop <= y) active = id;
    });
    tocLinks.forEach(function(a) {
      a.classList.toggle('active', a.getAttribute('href') === '#' + active);
    });
  }
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();
})();
function toggleBenPomPop(ev, el) {
  ev.stopPropagation();
  var pop = document.getElementById('benpomSharedPop');
  if (!pop) return;
  var openHere = pop.classList.contains('open') && pop._anchor === el;
  pop.classList.remove('open');
  if (openHere) return;
  pop._anchor = el;
  var r = el.getBoundingClientRect();
  pop.style.left = (window.scrollX + r.left + r.width / 2) + 'px';
  pop.style.top  = (window.scrollY + r.top - 8) + 'px';
  pop.classList.add('open');
}
document.addEventListener('click', function(e) {
  if (!e.target.closest('.benpom-term') && !e.target.closest('#benpomSharedPop')) {
    var pop = document.getElementById('benpomSharedPop');
    if (pop) pop.classList.remove('open');
  }
});
</script>
<script>
// ── Regional-shift chart: playoff teams' BenPom over the Masters London run,
// lines colored by region, team logos at the current (latest) end. ───────────
(function() {
  var DATA = {{ chart_json|safe }};
  var canvas = document.getElementById('regionShiftChart');
  if (!canvas || !window.Chart || !DATA.series || !Object.keys(DATA.series).length) return;

  var ORDER = ['TH','G2','PRX','VIT','FUT','LEV','EDG','XLG'];
  var WIN = '#16a34a', LOSS = '#dc2626';
  var logos = {};
  // Redraw the chart the instant each logo finishes loading, so the end-of-line
  // logos pop in immediately instead of waiting for the next hover/resize redraw.
  ORDER.forEach(function(org){ if (DATA.series[org]) { var im = new Image(); im.onload = function(){ if (typeof chart !== 'undefined' && chart) chart.draw(); }; im.src = '/logos/'+org+'.png'; logos[org] = im; } });

  var datasets = ORDER.filter(function(o){ return DATA.series[o]; }).map(function(org){
    var color = DATA.colors[DATA.region[org]] || '#888';
    var pts = DATA.series[org];
    return {
      label: org, org: org,
      data: pts,
      borderColor: color, backgroundColor: color,
      borderWidth: 2.6,
      // Green/red dot only on checkpoints where this team played a match.
      pointRadius:      pts.map(function(p){ return p.me ? 4.5 : 0; }),
      pointHoverRadius: pts.map(function(p){ return p.me ? 6.5 : 0; }),
      pointBackgroundColor: pts.map(function(p){ return p.me ? (p.won ? WIN : LOSS) : color; }),
      pointBorderColor: '#fff', pointBorderWidth: 1.5,
      pointHitRadius:   pts.map(function(p){ return p.me ? 5 : 0; }),
      cubicInterpolationMode: 'monotone', tension: 0.25,
    };
  });

  // End-of-line team logos. The near-tied G2/PRX/VIT need vertical separation
  // to be legible, but nudging a logo away from its true rating is misleading —
  // so we mark each line's true endpoint with a colored dot and draw a thin
  // leader line to the (de-overlapped, cluster-centered) logo.
  var endLogos = {
    id: 'endLogos',
    afterDatasetsDraw: function(chart) {
      var ctx = chart.ctx, S = 20, GAP = S + 3, items = [];
      chart.data.datasets.forEach(function(ds, i){
        var meta = chart.getDatasetMeta(i);
        if (!meta || !meta.data || !meta.data.length) return;
        var last = meta.data[meta.data.length - 1];
        items.push({ org: ds.org, x: last.x, trueY: last.y, y: last.y, color: ds.borderColor });
      });
      if (!items.length) return;
      items.sort(function(a,b){ return a.trueY - b.trueY; });
      items.forEach(function(it){ it.y = it.trueY; });
      // Symmetric push-apart relaxation: only logos that actually collide get
      // separated, splitting the overlap equally so a tight group spreads
      // around its center. Isolated logos (FUT, LEV) keep their true y — a flat
      // leader — instead of being dragged along by a neighbouring cluster.
      for (var iter = 0; iter < 60; iter++) {
        var moved = false;
        for (var k = 0; k < items.length - 1; k++) {
          var d = (items[k+1].y - items[k].y) - GAP;
          if (d < -0.01) {
            var sh = (-d) / 2;
            items[k].y -= sh; items[k+1].y += sh;
            moved = true;
          }
        }
        if (!moved) break;
      }
      var logoCx = function(it){ return it.x + 10 + S/2; };
      // Leader lines from each line's true endpoint to its (de-overlapped) logo.
      items.forEach(function(it){
        ctx.save();
        ctx.strokeStyle = it.color; ctx.globalAlpha = 0.45; ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.moveTo(it.x, it.trueY); ctx.lineTo(logoCx(it) - S/2 - 1, it.y); ctx.stroke();
        ctx.restore();
      });
      // Logos on white discs for legibility.
      items.forEach(function(it){
        var img = logos[it.org];
        if (!img || !img.complete || !img.naturalWidth) return;
        var cx = logoCx(it);
        ctx.save();
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';
        ctx.beginPath(); ctx.arc(cx, it.y, S/2 + 2, 0, Math.PI*2); ctx.fillStyle = '#fff'; ctx.fill();
        ctx.drawImage(img, cx - S/2, it.y - S/2, S, S);
        ctx.restore();
      });
      // Store logo hit-boxes so the mousemove handler can highlight a line
      // when its end logo is hovered.
      chart._logoBoxes = items.map(function(it){
        return { org: it.org, x: logoCx(it) - S/2, y: it.y - S/2, w: S, h: S };
      });
    }
  };

  // Match-dot popup — copied from the Modern Hub (_matchTooltipHTML + showDotTooltip).
  var popEl = document.getElementById('rsPopup');
  var popContent = document.getElementById('rsPopupContent');
  var EVENT_LABELS = { '2026_masters_london': 'Masters London' };

  function _matchTooltipHTML(m, won) {
    const org  = won ? m.winner : m.loser;
    const opp  = won ? m.loser  : m.winner;
    const d    = won ? m.winner_delta : m.loser_delta;
    const rat  = won ? m.winner_after  : m.loser_after;
    const dStr = (d >= 0 ? '+' : '') + d.toFixed(2);
    const evt  = EVENT_LABELS[m.event_id] || m.event_id || '';
    const rawParts = (m.series_score || '0-0').split('-');
    const displayScore = won ? m.series_score : `${rawParts[1]}-${rawParts[0]}`;
    const mapsRows = (m.maps || []).map(mp => {
      const mapWon = mp.winner === org;
      const orgRd = mapWon ? mp.wr : mp.lr;
      const oppRd = mapWon ? mp.lr : mp.wr;
      const diff   = orgRd - oppRd;
      return `<tr>
        <td class="popup-map-name">${mp.map}</td>
        <td class="popup-map-score ${mapWon?'w':'l'}">${orgRd}</td>
        <td class="popup-map-score ${mapWon?'l':'w'}">${oppRd}</td>
        <td class="popup-map-diff">${diff >= 0 ? '+' : ''}${diff}</td>
      </tr>`;
    }).join('');
    return `
      ${evt ? `<div class="popup-event-label">${evt}</div>` : ''}
      <div class="popup-teams">
        <div class="popup-team-block">
          <img class="popup-logo" src="/static/logos/${org}.png" onerror="this.style.display='none'" alt="${org}">
          <span class="popup-team-name">${org}</span>
        </div>
        <div class="popup-score-block">
          <span class="popup-score ${won?'w':'l'}">${displayScore}</span>
          <span class="popup-vs-label">series</span>
        </div>
        <div class="popup-team-block">
          <img class="popup-logo" src="/static/logos/${opp}.png" onerror="this.style.display='none'" alt="${opp}">
          <span class="popup-team-name">${opp}</span>
        </div>
      </div>
      <div class="popup-date">${m.date}</div>
      <div class="popup-delta ${d>=0?'pos':'neg'}">BenPom ${rat.toFixed(2)} &nbsp;(${dStr})</div>
      ${mapsRows ? `<table class="popup-maps-table">
        <thead><tr><th>Map</th><th>${org}</th><th>${opp}</th><th>Diff</th></tr></thead>
        <tbody>${mapsRows}</tbody>
      </table>` : ''}`;
  }

  function rsTooltip(ctx) {
    var tt = ctx.tooltip;
    if (!popEl) return;
    if (!tt || tt.opacity === 0) { popEl.classList.remove('visible'); return; }
    var dp = tt.dataPoints && tt.dataPoints[0];
    var raw = dp && dp.raw;
    if (!raw || !raw.me) { popEl.classList.remove('visible'); return; }
    popContent.innerHTML = _matchTooltipHTML(raw.me, raw.won);
    popEl.style.visibility = 'hidden';
    popEl.classList.add('visible');
    var wrap = ctx.chart.canvas.parentNode;
    var ttW = popEl.offsetWidth || 300, ttH = popEl.offsetHeight || 300, gap = 20;
    var dotX = tt.caretX, dotY = tt.caretY;
    var left = dotX - ttW / 2, top = dotY - ttH - gap;
    if (top < 4)                            top  = dotY + gap;
    if (left < 4)                           left = 4;
    if (left + ttW > wrap.offsetWidth - 4)  left = wrap.offsetWidth - ttW - 4;
    popEl.style.left = left + 'px';
    popEl.style.top  = top  + 'px';
    popEl.style.visibility = '';
  }

  var chart = new Chart(canvas, {
    type: 'line',
    data: { datasets: datasets },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      // Render the backing store at a higher pixel ratio so the chart stays
      // crisp when the page is zoomed in (default DPR rasterizes once, then
      // browser zoom just magnifies that bitmap → blur).
      devicePixelRatio: Math.min(4, (window.devicePixelRatio || 1) * 2),
      layout: { padding: { right: 50 } },
      interaction: { mode: 'nearest', intersect: true, axis: 'xy' },
      scales: {
        x: {
          type: 'time', min: '2026-05-22', max: '2026-06-16',
          time: { unit: 'day', tooltipFormat: 'MMM d' },
          ticks: { maxRotation: 0, autoSkip: true, maxTicksLimit: 7, font: { size: 11 }, color: '#7a6e7e' },
          grid: { color: '#f0ecf4' }
        },
        y: {
          title: { display: true, text: 'BenPom Rating', color: '#7a6e7e', font: { size: 12, weight: 'bold' } },
          ticks: { font: { size: 11 }, color: '#7a6e7e' },
          grid: { color: '#f0ecf4' }
        }
      },
      plugins: {
        legend: { display: false },
        tooltip: { enabled: false, external: rsTooltip }
      }
    },
    plugins: [endLogos]
  });

  // Hovering a team's end logo highlights just that team's line (dims others).
  var baseColors = {};
  chart.data.datasets.forEach(function(ds){ baseColors[ds.org] = ds.borderColor; });
  var hoverOrg = null, pinnedOrg = null;
  function activeOrg(){ return pinnedOrg || hoverOrg; }
  function applyHighlight() {
    var active = activeOrg();
    chart.data.datasets.forEach(function(ds){
      var base = baseColors[ds.org];
      var dim = active && ds.org !== active;
      if (!active) { ds.borderColor = base; ds.borderWidth = 2.6; }
      else if (ds.org === active) { ds.borderColor = base; ds.borderWidth = 4; }
      else { ds.borderColor = base + '26'; ds.borderWidth = 2; }  // dim others
      // Only the focused team keeps its win/loss dots.
      ds.pointRadius    = ds.data.map(function(p){ return (!dim && p.me) ? 4.5 : 0; });
      ds.pointHitRadius = ds.data.map(function(p){ return (!dim && p.me) ? 5 : 0; });
    });
    chart.update('none');
  }
  function logoAt(e) {
    var rect = canvas.getBoundingClientRect();
    var sx = rect.width  ? (canvas.clientWidth  / rect.width)  : 1;
    var sy = rect.height ? (canvas.clientHeight / rect.height) : 1;
    var mx = (e.clientX - rect.left) * sx, my = (e.clientY - rect.top) * sy;
    var boxes = chart._logoBoxes || [];
    for (var i = 0; i < boxes.length; i++) {
      var b = boxes[i];
      if (mx >= b.x - 3 && mx <= b.x + b.w + 3 && my >= b.y - 3 && my <= b.y + b.h + 3) return b.org;
    }
    return null;
  }
  canvas.addEventListener('mousemove', function(e){
    var found = logoAt(e);
    canvas.style.cursor = found ? 'pointer' : 'default';
    if (found) {
      // Over a logo: never show a dot popup. Force-clear any active dot tooltip.
      popEl.classList.remove('visible');
      chart.setActiveElements([]);
      if (chart.tooltip) chart.tooltip.setActiveElements([], {x:0, y:0});
    }
    if (found !== hoverOrg) { hoverOrg = found; if (!pinnedOrg) applyHighlight(); }
  });
  canvas.addEventListener('mouseleave', function(){
    if (hoverOrg) { hoverOrg = null; if (!pinnedOrg) applyHighlight(); }
  });
  // Click a logo to PIN its highlight; click elsewhere to clear it.
  canvas.addEventListener('click', function(e){
    var found = logoAt(e);
    pinnedOrg = found ? (pinnedOrg === found ? null : found) : null;
    applyHighlight();
  });
  document.addEventListener('click', function(e){
    if (e.target !== canvas && pinnedOrg) { pinnedOrg = null; applyHighlight(); }
  });
})();
</script>
<div id="benpomSharedPop" class="benpom-pop"><span class="benpom-pop-text">BenPom is my personal rating model for VCT teams.</span><span class="benpom-pop-links"><a href="/mapelo/modern/" target="_blank" rel="noopener">Current BenPom Hub</a><a href="/mapelo/how-it-works/" target="_blank" rel="noopener">How BenPom works</a></span></div>
<div id="mcPop" class="mc-pop"></div>
<script>
// ===================================================================
// Modern-Hub prediction engine, ported from MapElo.py so the bracket
// match cards use the identical MC veto model AND the identical v6
// closed-form win probabilities (SITE_MODEL from data.site_model =
// data/site_model.json; reference math trading_model/predict.py) as
// the Hub's Upcoming Matches. Cards stay consistent with the live bars.
// ===================================================================
var MATCH_CARDS = {};
var _cardsReady = false;
var VETO_HUB={teams:{},snap_pools:{}}, ORG_REGIONS_HUB={}, SNAP_TEAMS={};
var SITE_MODEL={}, SNAP_BETA=0, XREGION_OFFSETS={}, GF_UPPER_LOGIT=0, B_PICK=0;
var SNAP_KEY='after_santiago';
function shiftSeriesProb(p,delta){ if(!delta) return p; var ps=Math.max(Math.min(p,1-1e-9),1e-9); return 1.0/(1.0+Math.exp(-(Math.log(ps/(1-ps))+delta))); }
function xregionAdjHUB(orgA,orgB){ var ra=(ORG_REGIONS_HUB||{})[orgA], rb=(ORG_REGIONS_HUB||{})[orgB]; if(!ra||!rb||ra===rb) return 0; return (XREGION_OFFSETS[ra]||0)-(XREGION_OFFSETS[rb]||0); }
function v6SeriesProbHUB(rA,rB,orgA,orgB,fmt,gfUpperOrg){ var p=1/(1+Math.exp(-SNAP_BETA*(rA-rB+xregionAdjHUB(orgA,orgB)))); var ps; if(fmt==='bo1') ps=p; else if(fmt==='bo5'||fmt==='bo5_gf'){ var q=1-p; ps=p*p*p*(1+3*q+6*q*q); } else ps=p*p*(3-2*p); if(fmt==='bo5_gf'&&(gfUpperOrg===orgA||gfUpperOrg===orgB)) ps=shiftSeriesProb(ps,gfUpperOrg===orgA?GF_UPPER_LOGIT:-GF_UPPER_LOGIT); return ps; }
var VETO_STEPS_HUB={
  bo1:[{side:'A',action:'ban'},{side:'B',action:'ban'},{side:'A',action:'ban'},{side:'B',action:'ban'},{side:'A',action:'ban'},{side:'B',action:'ban'}],
  bo3:[{side:'A',action:'ban'},{side:'B',action:'ban'},{side:'A',action:'pick'},{side:'B',action:'pick'},{side:'A',action:'ban'},{side:'B',action:'ban'}],
  bo5:[{side:'A',action:'ban'},{side:'B',action:'ban'},{side:'A',action:'pick'},{side:'B',action:'pick'},{side:'A',action:'pick'},{side:'B',action:'pick'}],
  bo5_gf:[{side:'A',action:'ban'},{side:'A',action:'ban'},{side:'A',action:'pick'},{side:'B',action:'pick'},{side:'A',action:'pick'},{side:'B',action:'pick'}]
};
var SERIES_THRESH_HUB={bo1:1,bo3:2,bo5:3,bo5_gf:3};
function getActivePoolHUB(snap){ var key='2026_'+snap; var cp=(VETO_HUB.computed_pools||{})[key]; if(cp&&cp.length>=7) return cp; return (VETO_HUB.snap_pools||{})[key]||null; }
function getBanProbsHUB(patt,oppTeam,rem){ var scores={}; rem.forEach(function(m){ var rate=(patt&&patt.bans&&patt.bans[m]!=null)?patt.bans[m]:0; var oppWin=(oppTeam&&oppTeam.maps&&oppTeam.maps[m])?(oppTeam.maps[m].win_pct||0.5):0.5; scores[m]=(rate+0.02)*(0.75+oppWin); }); var tot=rem.reduce(function(s,m){return s+scores[m];},0); if(tot===0) rem.forEach(function(m){scores[m]=1/rem.length;}); else rem.forEach(function(m){scores[m]/=tot;}); return scores; }
function getPickProbsHUB(patt,rem,ownTeam){ var scores={}; rem.forEach(function(m){ var rate=(patt&&patt.picks&&patt.picks[m]!=null)?patt.picks[m]:0; var base=rate+0.02; var ownWin=(ownTeam&&ownTeam.maps&&ownTeam.maps[m])?(ownTeam.maps[m].win_pct||0.5):0.5; var ownF=ownTeam?Math.pow(0.3+ownWin,2.0):1.0; scores[m]=base*ownF; }); var tot=rem.reduce(function(s,m){return s+scores[m];},0); if(tot===0) rem.forEach(function(m){scores[m]=1/rem.length;}); else rem.forEach(function(m){scores[m]/=tot;}); return scores; }
function sampleFromHUB(probs){ var r=Math.random(),cum=0,keys=Object.keys(probs); for(var i=0;i<keys.length;i++){cum+=probs[keys[i]];if(r<=cum) return keys[i];} return keys[keys.length-1]; }
function _seededRng(seed){ var s=seed>>>0; return function(){ s=(s+0x6D2B79F5)|0; var t=Math.imul(s^s>>>15,1|s); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296; }; }
function _matchSeed(){ var h=2166136261; for(var i=0;i<arguments.length;i++){ var s=String(arguments[i]); for(var j=0;j<s.length;j++){ h^=s.charCodeAt(j); h=Math.imul(h,16777619); } } return h>>>0; }
function _withSeededRand(seed,fn){ var orig=Math.random; Math.random=_seededRng(seed); try{ return fn(); } finally{ Math.random=orig; } }
function simulateVetoHUB(tA,tB,orgA,orgB,pool,snap,fmt){ var pA=((VETO_HUB.teams||{})['2026_'+snap]||{})[orgA]||null; var pB=((VETO_HUB.teams||{})['2026_'+snap]||{})[orgB]||null; var rem=pool.slice(),fate={}; (VETO_STEPS_HUB[fmt]||VETO_STEPS_HUB.bo3).forEach(function(step){ var patt=step.side==='A'?pA:pB, oppT=step.side==='A'?tB:tA, ownT=step.side==='A'?tA:tB; var m=step.action==='ban'?sampleFromHUB(getBanProbsHUB(patt,oppT,rem)):sampleFromHUB(getPickProbsHUB(patt,rem,ownT)); fate[m]=step.action+step.side; rem=rem.filter(function(x){return x!==m;}); }); if(rem.length) fate[rem[0]]='dec'; return fate; }
function topVetoHUB(tA,tB,orgA,orgB,pool,snap,fmt,K){ var pA=((VETO_HUB.teams||{})['2026_'+snap]||{})[orgA]||null; var pB=((VETO_HUB.teams||{})['2026_'+snap]||{})[orgB]||null; K=K||3; var steps=VETO_STEPS_HUB[fmt]||VETO_STEPS_HUB.bo3; var states=[{rem:pool.slice(),seq:[],prob:1.0}]; steps.forEach(function(step){ var next=[]; states.forEach(function(st){ var patt=step.side==='A'?pA:pB, oppT=step.side==='A'?tB:tA, ownT=step.side==='A'?tA:tB; var probs=step.action==='ban'?getBanProbsHUB(patt,oppT,st.rem):getPickProbsHUB(patt,st.rem,ownT); st.rem.forEach(function(m){ var p=probs[m]||0; if(p>0.005) next.push({rem:st.rem.filter(function(x){return x!==m;}),seq:st.seq.concat([{side:step.side,action:step.action,map:m}]),prob:st.prob*p}); }); }); next.sort(function(a,b){return b.prob-a.prob;}); states=next.slice(0,K*3); }); states.forEach(function(st){if(st.rem.length) st.seq.push({side:'',action:'dec',map:st.rem[0]});}); states.sort(function(a,b){return b.prob-a.prob;}); return states.slice(0,K); }
function _getTeamObjBC(org,lbTeams,liveMapStats){ var lb=lbTeams[org]; var overall=lb?lb.rating:0; var maps={}; var st=SNAP_TEAMS[org]; if(st&&st.maps){ Object.keys(st.maps).forEach(function(mp){ maps[mp]=Object.assign({},st.maps[mp]); }); } else if(lb){ (lb.all_maps||[]).forEach(function(m){ maps[m.map]={rating:m.rating,w:m.w,l:m.l,win_pct:m.w/Math.max(1,m.w+m.l)}; }); } var live=liveMapStats[org]; if(live){ Object.keys(live).forEach(function(mp){ var base=maps[mp]||{}; var ld=live[mp]||{}; maps[mp]={rating:base.rating, w:(ld.w!=null)?ld.w:base.w, l:(ld.l!=null)?ld.l:base.l, win_pct:(ld.win_pct!=null)?ld.win_pct:base.win_pct}; }); } if(!Object.keys(maps).length&&!overall) return null; return {overall_rating:overall, maps:maps}; }

// Run the full per-map MC for one matchup. Identical to the Hub's Upcoming
// computation, so each card's win % matches the live Upcoming bars.
function computeCard(ctx, orgA, orgB, fmt, mDate){
  var pool=ctx.pool, lbTeams=ctx.lbTeams, liveMapStats=ctx.liveMapStats, snapKey=ctx.snapKey, vetoSnapKey=ctx.vetoSnapKey;
  var tA=_getTeamObjBC(orgA,lbTeams,liveMapStats), tB=_getTeamObjBC(orgB,lbTeams,liveMapStats);
  var lbA=lbTeams[orgA], lbB=lbTeams[orgB];
  var ratingA=lbA?lbA.rating:(tA?(tA.overall_rating||0):0);
  var ratingB=lbB?lbB.rating:(tB?(tB.overall_rating||0):0);
  var nSims=20000;
  var mapWins={}, mapPlays={};
  pool.forEach(function(mp){ mapWins[mp]=0; mapPlays[mp]=0; });
  if(tA&&tB){
    // v6 map-level inputs: overall ratings + cross-region adjustment + the
    // pick logit (±B_PICK) by veto fate (predict.py map_probability) — the
    // MC is the veto/map-breakdown engine, not the headline win chance.
    var zBase=SNAP_BETA*((tA.overall_rating||ratingA)-(tB.overall_rating||ratingB)+xregionAdjHUB(orgA,orgB));
    _withSeededRand(_matchSeed(orgA,orgB,fmt,mDate),function(){
      for(var s=0;s<nSims;s++){
        var fm=simulateVetoHUB(tA,tB,orgA,orgB,pool,snapKey,fmt);
        pool.forEach(function(mp){
          var fc=fm[mp]||'banA';
          if(fc==='pickA'||fc==='pickB'||fc==='dec'){
            mapPlays[mp]++;
            var z=zBase+(fc==='pickA'?B_PICK:(fc==='pickB'?-B_PICK:0));
            if(Math.random()<1/(1+Math.exp(-z))){ mapWins[mp]++; }
          }
        });
      }
    });
  }
  // Headline win prob = the v6 closed form on overall ratings (predict.py
  // series_probability; for bo5_gf side A is the upper-bracket team).
  var pA=v6SeriesProbHUB(ratingA,ratingB,orgA,orgB,fmt,fmt==='bo5_gf'?orgA:'');
  var topSeqs=(tA&&tB&&pool.length)?topVetoHUB(tA,tB,orgA,orgB,pool,snapKey,fmt,1):[];
  var veto=topSeqs.length?topSeqs[0].seq.map(function(step){ return {side:step.side,action:step.action,map:step.map}; }):[];
  var maps=pool.filter(function(mp){return mapPlays[mp]>0;}).sort(function(a,b){return mapPlays[b]-mapPlays[a];})
               .map(function(mp){ return {map:mp, wp_a:mapWins[mp]/mapPlays[mp]}; });
  return {a:orgA,b:orgB,fmt:fmt,pa:pA,winner:(pA>=0.5?orgA:orgB),veto:veto,maps:maps};
}

function _W(c){ return c.winner; }
function _L(c){ return c.winner===c.a?c.b:c.a; }

// Re-seed the whole double-elimination bracket from the dynamic model: every
// match's advancer is the team the per-map MC favours (>50%). Winners feed
// forward so later-round matchups are themselves model-derived — guaranteeing
// the bracket and the click-through cards are perfectly self-consistent.
function simulateAndBuild(data){
  var lbTeams={}; (data.leaderboard.teams||[]).forEach(function(t){ lbTeams[t.org]=t; });
  var snapKey=SNAP_KEY;
  var pool=(VETO_HUB.current_pool&&VETO_HUB.current_pool.length>=7)?VETO_HUB.current_pool:getActivePoolHUB(snapKey);
  if(!pool||!pool.length){ var seen={}; Object.values(SNAP_TEAMS).forEach(function(t){ Object.keys(t.maps||{}).forEach(function(m){ seen[m]=1; }); }); pool=Object.keys(seen).sort(); }
  if(!pool||!pool.length) pool=['Ascent','Bind','Breeze','Fracture','Haven','Lotus','Pearl','Split'];
  var ctx={pool:pool, lbTeams:lbTeams, liveMapStats:VETO_HUB.live_map_stats||{}, snapKey:snapKey, vetoSnapKey:'2026_'+snapKey};
  // Inherit the real upcoming match date per unordered pair so the live first-
  // round matchups are byte-identical to the Hub's Upcoming bars.
  var upc={}; (data.upcoming||[]).forEach(function(m){ var x=m.org_a||m.team_a, y=m.org_b||m.team_b; if(x&&y) upc[[x,y].sort().join('|')]=m.date||''; });
  function dt(a,b){ return upc[[a,b].sort().join('|')]||'2026-06-14'; }
  function cc(a,b,fmt){ return computeCard(ctx,a,b,fmt,dt(a,b)); }
  var C={};
  // Upper bracket
  C.uqf1=cc('G2','XLG','bo3'); C.uqf2=cc('EDG','FUT','bo3'); C.uqf3=cc('PRX','LEV','bo3'); C.uqf4=cc('TH','VIT','bo3');
  C.usf1=cc(_W(C.uqf1),_W(C.uqf2),'bo3'); C.usf2=cc(_W(C.uqf3),_W(C.uqf4),'bo3');
  C.uf=cc(_W(C.usf1),_W(C.usf2),'bo5');
  // Lower bracket. Each LR1 winner stays on its own side into LR2; the USF
  // losers cross over (loser USF2 -> top LR2, loser USF1 -> bottom LR2), per
  // the standard VCT Masters double-elim format.
  C.lr1a=cc(_L(C.uqf1),_L(C.uqf2),'bo3'); C.lr1b=cc(_L(C.uqf3),_L(C.uqf4),'bo3');
  C.lr2a=cc(_L(C.usf2),_W(C.lr1a),'bo3'); C.lr2b=cc(_L(C.usf1),_W(C.lr1b),'bo3');
  C.lr3=cc(_W(C.lr2a),_W(C.lr2b),'bo3');
  C.lf=cc(_L(C.uf),_W(C.lr3),'bo5');
  // Grand Final — Upper Final winner is the upper seed (side A: both bans + 1st pick)
  C.gf=computeCard(ctx,_W(C.uf),_W(C.lf),'bo5_gf',dt(_W(C.uf),_W(C.lf)));
  MATCH_CARDS=C;
  return C;
}

// Paint the model-derived teams + winner highlight into the static bracket DOM.
function _setMatchDOM(id, a, b, winner, isChamp){
  var el=document.querySelector('.br-match[data-match="'+id+'"]'); if(!el) return;
  function th(t){
    var champ=isChamp && t===winner;
    return '<div class="br-team'+(t===winner?' win':'')+(champ?' champ':'')+'">'+
      '<img src="/logos/'+t+'.png" onerror="this.style.display=\\'none\\'"><span>'+t+'</span>'+
      (champ?'<span class="br-tag">CHAMPION</span>':'')+'</div>';
  }
  el.innerHTML=th(a)+th(b);
}
function renderPredictionBracket(C){
  ['uqf1','uqf2','uqf3','uqf4','usf1','usf2','uf','lr1a','lr1b','lr2a','lr2b','lr3','lf'].forEach(function(id){
    var c=C[id]; if(c) _setMatchDOM(id,c.a,c.b,c.winner,false);
  });
  if(C.gf) _setMatchDOM('gf',C.gf.a,C.gf.b,C.gf.winner,true);
}

function _initBracketCards(retries){
  fetch('/mapelo/modern/data').then(function(r){ return r.ok?r.json():null; }).then(function(data){
    if(data&&data.leaderboard&&data.leaderboard.teams&&data.leaderboard.teams.length){
      VETO_HUB=data.veto_model||{teams:{},snap_pools:{}};
      ORG_REGIONS_HUB=data.org_regions||{};
      SNAP_TEAMS=data.snap_teams||{};
      // v6 model constants from the hub payload (data/site_model.json)
      SITE_MODEL=data.site_model||{};
      SNAP_BETA=SITE_MODEL.beta||0;
      XREGION_OFFSETS=SITE_MODEL.xregion_offsets||{};
      GF_UPPER_LOGIT=SITE_MODEL.gf_upper_logit||0;
      B_PICK=SITE_MODEL.b_pick||0;
      SNAP_KEY=data.snap_key||'after_santiago';
      setTimeout(function(){ try{ var C=simulateAndBuild(data); renderPredictionBracket(C); _cardsReady=true; }catch(e){ console.error('bracket sim failed',e); } },0);
    } else if((retries||0)<150){
      setTimeout(function(){ _initBracketCards((retries||0)+1); },2000);
    }
  }).catch(function(){ if((retries||0)<150) setTimeout(function(){ _initBracketCards((retries||0)+1); },2000); });
}
_initBracketCards(0);

function _mcImg(m){ return '/maps/' + (m||'').toLowerCase() + '.jpg'; }
function showMatchCard(ev, el) {
  ev.stopPropagation();
  var d = MATCH_CARDS[el.dataset.match];
  var pop = document.getElementById('mcPop');
  if (!pop) return;
  if (!d) {
    var openCalc = pop.classList.contains('open') && pop._anchor === el;
    pop.classList.remove('open');
    if (openCalc) return;
    pop._anchor = el;
    pop.innerHTML = '<div class="mc-fmt" style="padding:14px 4px">Calculating prediction&hellip;</div>';
    var rc = el.getBoundingClientRect();
    pop.style.left = (window.scrollX + rc.left + rc.width/2) + 'px';
    pop.style.top = (window.scrollY + rc.top - 8) + 'px';
    pop.classList.add('open');
    return;
  }
  var openHere = pop.classList.contains('open') && pop._anchor === el;
  pop.classList.remove('open');
  if (openHere) return;
  pop._anchor = el;
  // Always show the predicted winner on the left.
  var winA = d.winner === d.a;
  var W = winA ? d.a : d.b, L = winA ? d.b : d.a;
  var pWn = (winA ? d.pa : 1 - d.pa) * 100;
  var pW = pWn.toFixed(1), pL = (100 - pWn).toFixed(1);
  var fmtLbl = d.fmt === 'bo5' ? 'Bo5' : (d.fmt === 'bo5_gf' ? 'Bo5 &middot; Grand Final' : 'Bo3');
  var veto = d.veto.map(function(v){
    var tag = v.action === 'dec' ? 'Decider' : ((v.side === 'A' ? d.a : d.b) + ' ' + (v.action === 'ban' ? 'ban' : 'pick'));
    var cls = v.action === 'ban' ? 'mc-tag-ban' : (v.action === 'pick' ? 'mc-tag-pick' : 'mc-tag-dec');
    return '<div class="mc-veto-row"><img src="' + _mcImg(v.map) + '" onerror="this.style.display=\\'none\\'"><span>' + v.map + '</span><span class="mc-veto-tag ' + cls + '">' + tag + '</span></div>';
  }).join('');
  var maps = d.maps.map(function(mp){
    var wWn = (winA ? mp.wp_a : 1 - mp.wp_a) * 100;
    var wW = wWn.toFixed(1), wL = (100 - wWn).toFixed(1);
    return '<tr><td>' + mp.map + '</td><td class="mc-mw">' + W + ' ' + wW + '%</td><td class="mc-mw">' + L + ' ' + wL + '%</td></tr>';
  }).join('');
  pop.innerHTML =
    '<div class="mc-head">' +
      '<div class="mc-team"><img src="/logos/' + W + '.png" onerror="this.style.visibility=\\'hidden\\'"><span class="mc-team-name">' + W + '</span></div>' +
      '<div class="mc-prob"><div class="mc-prob-pcts"><span class="fav">' + pW + '%</span><span>' + pL + '%</span></div>' +
        '<div class="mc-bar"><div class="mc-bar-a" style="width:' + pW + '%"></div></div></div>' +
      '<div class="mc-team"><img src="/logos/' + L + '.png" onerror="this.style.visibility=\\'hidden\\'"><span class="mc-team-name">' + L + '</span></div>' +
    '</div>' +
    '<div class="mc-fmt">' + fmtLbl + ' &middot; Series win prob.</div>' +
    '<div class="mc-sec-lbl">Predicted Veto</div><div class="mc-veto">' + veto + '</div>' +
    '<div class="mc-sec-lbl">Per-Map Win Probability</div><table class="mc-maps"><tbody>' + maps + '</tbody></table>';
  var r = el.getBoundingClientRect();
  var x = window.scrollX + r.left + r.width / 2;
  x = Math.max(window.scrollX + 156, Math.min(x, window.scrollX + document.documentElement.clientWidth - 156));
  pop.style.left = x + 'px';
  pop.style.top = (window.scrollY + r.top - 8) + 'px';
  pop.classList.add('open');
}
document.addEventListener('click', function(e){
  if (!e.target.closest('.br-clickable') && !e.target.closest('#mcPop')) {
    var p = document.getElementById('mcPop'); if (p) p.classList.remove('open');
  }
});
</script>
<script async src="https://platform.twitter.com/widgets.js"></script>
</body>
</html>
"""


@article_masters_london_playoffs_bp.route("/")
def index():
    # Match cards are computed CLIENT-SIDE from /mapelo/modern/data using the
    # exact same MC veto engine as the Modern Hub's Upcoming Matches, so the
    # win probabilities are identical. (No server-side closed-form injection.)
    return render_template_string(PAGE_HTML, chart_json=json.dumps(_chart_payload()))
