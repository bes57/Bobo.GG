"""
EMERGENCY ARTICLE: Previewing Champions Shanghai Grand Finals!

Structure mirrors Top10Shanghai.py / ChampionshipDNA.py: h1 / byline / cover /
content, with the fixed .toc rail + scroll-spy. Article prose is added by the
author; this file renders only the structural elements.
"""

import os, re, json
from flask import Blueprint

article_gf_preview_bp = Blueprint("article_gf_preview", __name__)

ROOT      = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, "data", "article_gf_preview.json")   # built by scrapers/BuildGFPreviewData.py

_MON = ["January", "February", "March", "April", "May", "June", "July",
        "August", "September", "October", "November", "December"]
_REGION_CLS = {"EMEA": "emea", "Americas": "americas", "Pacific": "pacific", "CN": "cn", "China": "cn"}


def _load():
    try:
        return json.load(open(DATA_PATH))
    except Exception:
        return {}


def _long_date(iso):
    try:
        y, m, d = iso.split("-")
        return f"{_MON[int(m) - 1]} {int(d)}, {y}"
    except Exception:
        return iso


def _toc_html():
    return '<a href="#intro">Intro</a>\n  <a href="#numbers">The Numbers</a>\n  <a href="#narratives">The Narratives</a>\n  <a href="#prediction">My Prediction</a>'


def _rankings_html(d):
    """Static top-5 BenPom power rankings card (frozen snapshot)."""
    rows = []
    finalists = {d.get("gf", {}).get("a"), d.get("gf", {}).get("b")}
    for t in d.get("top5", []):
        hl = " hl" if t["org"] in finalists else ""
        rc = _REGION_CLS.get(t.get("region", ""), "")
        rows.append(
            f'<tr class="rk-row{hl}"><td class="rk-n">{t["rank"]}</td>'
            f'<td class="rk-team"><img src="/logos/{t["org"]}.png" onerror="this.style.visibility=\'hidden\'" alt="">{t["org"]}</td>'
            f'<td><span class="reg {rc}">{t.get("region", "")}</span></td>'
            f'<td class="rk-rt">{t["rating"]:.2f}</td></tr>')
    return (
        '<div class="gfx rk">'
        '<div class="gfx-head"><span class="gfx-title">BenPom Power Rankings</span>'
        f'<span class="gfx-sub">Top 5 &middot; {_long_date(d.get("as_of", ""))}</span></div>'
        '<table class="rk-tbl"><thead><tr><th>#</th><th>Team</th><th>Region</th><th>Rating</th></tr></thead>'
        '<tbody>' + "".join(rows) + '</tbody></table></div>')


def _match_card_html(d):
    """Static BenPom grand-final card: series win prob, predicted veto, per-map odds."""
    g = d.get("gf") or {}
    if not g:
        return ""
    a, b, pa = g["a"], g["b"], g["pa"]
    pA, pB = f"{pa * 100:.1f}", f"{(1 - pa) * 100:.1f}"
    fav_a = pa >= 0.5
    veto_rows = []
    for v in g.get("veto", []):
        if v["action"] == "dec":
            tag, cls = "Decider", "mc-tag-dec"
        else:
            tag = f'{a if v["side"] == "A" else b} {v["action"]}'
            cls = "mc-tag-ban" if v["action"] == "ban" else "mc-tag-pick"
        veto_rows.append(
            f'<div class="mc-veto-row"><img src="/maps/{v["map"].lower()}.jpg" onerror="this.style.display=\'none\'" alt="">'
            f'<span>{v["map"]}</span><span class="mc-veto-tag {cls}">{tag}</span></div>')
    wp = {m["map"]: m["wp_a"] for m in g.get("maps", [])}
    map_rows = []
    for v in g.get("veto", []):
        if v["action"] in ("pick", "dec") and v["map"] in wp:
            wa = wp[v["map"]] * 100
            map_rows.append(
                f'<tr><td>{v["map"]}</td><td class="mc-mw{" fav" if wa >= 50 else ""}">{a} {wa:.1f}%</td>'
                f'<td class="mc-mw{" fav" if wa < 50 else ""}">{b} {100 - wa:.1f}%</td></tr>')
    return (
        '<div class="gfx mc">'
        '<div class="mc-head">'
        f'<div class="mc-team"><img src="/logos/{a}.png" onerror="this.style.visibility=\'hidden\'" alt=""><span class="mc-team-name">{a}</span><span class="mc-seed">Upper bracket</span></div>'
        f'<div class="mc-prob"><div class="mc-prob-pcts"><span class="{"fav" if fav_a else ""}">{pA}%</span><span class="{"" if fav_a else "fav"}">{pB}%</span></div>'
        f'<div class="mc-bar"><div class="mc-bar-a" style="width:{pA}%"></div></div></div>'
        f'<div class="mc-team"><img src="/logos/{b}.png" onerror="this.style.visibility=\'hidden\'" alt=""><span class="mc-team-name">{b}</span><span class="mc-seed">Lower bracket</span></div>'
        '</div>'
        '<div class="mc-fmt">Bo5 &middot; Grand Final &middot; Series win prob.</div>'
        '<div class="mc-sec-lbl">Predicted Veto</div><div class="mc-veto">' + "".join(veto_rows) + '</div>'
        '<div class="mc-sec-lbl">Per-Map Win Probability</div><table class="mc-maps"><tbody>' + "".join(map_rows) + '</tbody></table>'
        f'<div class="mc-note">BenPom snapshot &middot; {_long_date(d.get("as_of", ""))}</div>'
        '</div>')


def _optimal_card_html(d):
    """Static card: BenPom-optimal veto for LOUD by per-map ratings."""
    o = d.get("optimal") or {}
    g = d.get("gf") or {}
    if not o or not g:
        return ""
    a, b = g["a"], g["b"]
    rows = []
    for v in o.get("veto", []):
        if v["action"] == "dec":
            tag, cls = "Decider", "mc-tag-dec"
        else:
            tag = f'{a if v["side"] == "A" else b} {v["action"]}'
            cls = "mc-tag-ban" if v["action"] == "ban" else "mc-tag-pick"
        ra = f'{v["rating_a"]:.2f}' if v.get("rating_a") is not None else "&ndash;"
        rb = f'{v["rating_b"]:.2f}' if v.get("rating_b") is not None else "&ndash;"
        e = v.get("edge") or 0.0
        ecls = "pos" if e > 0 else ("neg" if e < 0 else "")
        rows.append(
            f'<tr><td class="ov-map"><img src="/maps/{v["map"].lower()}.jpg" onerror="this.style.display=\'none\'" alt="">{v["map"]}</td>'
            f'<td class="ov-rt">{ra}</td><td class="ov-rt">{rb}</td>'
            f'<td class="ov-edge {ecls}">{e:+.2f}</td>'
            f'<td class="ov-tag"><span class="mc-veto-tag {cls}">{tag}</span></td></tr>')
    return (
        '<div class="gfx ov">'
        f'<div class="gfx-head"><span class="gfx-title">Optimal Veto for {a}</span>'
        f'<span class="gfx-sub">BenPom map ratings &middot; {b} picks as predicted</span></div>'
        f'<table class="ov-tbl"><thead><tr><th>Map</th><th>{a}</th><th>{b}</th><th>Edge</th><th></th></tr></thead>'
        '<tbody>' + "".join(rows) + '</tbody></table>'
        f'<div class="mc-note">Bo5 &middot; Grand Final &middot; {_long_date(d.get("as_of", ""))}</div>'
        '</div>')


def _chronicle_html(d):
    """Static card: Vitality's Champions map record split by Chronicle's agent."""
    c = dict(d.get("chronicle") or {})
    if not c or not c.get("maps"):
        return ""
    # Playoffs only (group-stage maps dropped at the author's request).
    c["maps"] = [m for m in c["maps"] if m["stage"].startswith("Playoffs")]
    c["other_agents"] = sorted({m["agent_label"] for m in c["maps"] if m["agent"] != "neon"})
    def rounds(maps):
        rf = rl = 0
        for m in maps:
            a, b = m["score"].split("-")
            rf += int(a); rl += int(b)
        return rf, rl
    def tile(label, rec, maps, extra, cls):
        rf, rl = rounds(maps)
        diff = rf - rl
        per = f'{diff / len(maps):+.1f}' if maps else "&ndash;"
        return (f'<div class="cn-tile {cls}"><div class="cn-lbl">{label}</div>'
                f'<div class="cn-rec">{per}</div>'
                f'<div class="cn-pct"><span>avg round differential</span></div>'
                f'<div class="cn-sub">{len(maps)} maps &middot; {rf}&ndash;{rl} rounds{extra}</div></div>')
    neon_maps = [m for m in c["maps"] if m["agent"] == "neon"]
    other_maps = [m for m in c["maps"] if m["agent"] != "neon"]
    others = ", ".join(c.get("other_agents") or [])
    tiles = (tile(f'{c["player"]} on Neon', c["neon"], neon_maps, "", "neon")
             + tile(f'{c["player"]} not on Neon', c["other"], other_maps, f' &middot; {others}', "other"))
    rows = []
    for m in c["maps"]:
        res = f'<span class="cn-res {"w" if m["won"] else "l"}">{"W" if m["won"] else "L"} {m["score"]}</span>'
        ag = f'<span class="cn-ag{" neon" if m["agent"] == "neon" else ""}">{m["agent_label"]}</span>'
        stage = m["stage"].replace("Group Stage: ", "").replace("Playoffs: ", "")
        rows.append(f'<tr><td class="cn-stage">{stage}</td><td class="cn-map"><img src="/maps/{m["map"].lower()}.jpg" onerror="this.style.display=\'none\'" alt="">{m["map"]}</td>'
                    f'<td class="cn-opp"><img src="/logos/{m["opp"]}.png" onerror="this.style.visibility=\'hidden\'" alt="">{m["opp"]}</td><td>{ag}</td><td class="cn-r">{res}</td></tr>')
    return (
        '<div class="gfx cn">'
        f'<div class="gfx-head"><span class="gfx-title">{c["org"]} in the {c["event"]} Playoffs</span>'
        f'<span class="gfx-sub">Map record by {c["player"]}&rsquo;s agent</span></div>'
        f'<div class="cn-tiles">{tiles}</div>'
        '<table class="cn-tbl"><thead><tr><th>Stage</th><th>Map</th><th>Opp</th><th>Agent</th><th>Result</th></tr></thead>'
        '<tbody>' + "".join(rows) + '</tbody></table>'
        f'<div class="mc-note">Through the Lower Final &middot; {_long_date(d.get("as_of", ""))}</div>'
        '</div>')


def _intl_card(players, names, title, sub, note, cls=""):
    """One column per player: headshot, VCT-international count, titles, event list."""
    cols = []
    for name in names:
        pl = players.get(name)
        if not pl:
            continue
        evs = []
        for e in pl["events"]:
            pl_txt = (e.get("placement") or "").replace("Playoffs - ", "").replace("Main Event - ", "").replace("Group Stage - ", "").replace("Bracket Stage - ", "")
            win = " win" if re.search(r"\b1st\b", pl_txt) and not e.get("current") else ""
            cur = " cur" if e.get("current") else ""
            if e.get("current"):
                pl_txt = ""          # the ongoing event needs no placement label
            evs.append(f'<li class="ie-ev{win}{cur}"><span class="ie-yr">{e["year"]}</span><span class="ie-nm">{e["short"]}</span><span class="ie-pl">{pl_txt}</span></li>')
        titles = pl.get("titles", 0)
        head = pl.get("head") or ""
        cols.append(
            f'<div class="ie-col"><img class="ie-head" src="{head}" onerror="this.style.visibility=\'hidden\'" alt="">'
            f'<div class="ie-name">{name}</div><div class="ie-count">{pl["count"]}</div>'
            f'<div class="ie-lbl">international event{"" if pl["count"] == 1 else "s"}</div>'
            f'<div class="ie-sub">{titles} title{"" if titles == 1 else "s"}</div>'
            f'<ul class="ie-list">{"".join(evs)}</ul></div>')
    if not cols:
        return ""
    return (f'<div class="gfx ie {cls}"><div class="gfx-head"><span class="gfx-title">{title}</span>'
            f'<span class="gfx-sub">{sub}</span></div>'
            f'<div class="ie-cols">{"".join(cols)}</div>'
            f'<div class="mc-note">{note}</div></div>')


def _intl_html(d):
    ie = (d.get("intl_events") or {}).get("players") or {}
    return _intl_card(ie, ("Derke", "Chronicle", "Jamppi"), "Vitality&rsquo;s veterans",
                      "VCT international events since 2021",
                      "Masters, Champions and LOCK//IN &middot; includes Champions Shanghai")


def _loud_intl_html(d):
    ie = (d.get("intl_events") or {}).get("loud_players") or {}
    return _intl_card(ie, ("erde", "lukxo", "Darker", "tkzin", "DaviH"), "LOUD&rsquo;s roster",
                      "VCT international events since 2021",
                      "Masters, Champions and LOCK//IN &middot; includes Champions Shanghai", cls="five")

def _loud_po_html(d):
    """Static card: LOUD players' VLR rating in every Champions playoff series."""
    lp = d.get("loud_po") or {}
    if not lp or not lp.get("matches"):
        return ""
    ms = lp["matches"]
    head = "".join(f'<th><span class="lp-stage">{m["short"]}</span><span class="lp-opp"><img src="/logos/{m["opp"]}.png" onerror="this.style.visibility=\'hidden\'" alt="">vs {m["opp"]}</span>'
                   f'<span class="lp-res {"w" if m["won"] else "l"}">{"W" if m["won"] else "L"} {m["score"]}</span></th>' for m in ms)
    rows = []
    for pl in lp["players"]:
        star = pl["top2_count"] == len(ms)
        cells = []
        for m in ms:
            r = pl["ratings"].get(m["match_id"])
            top = pl["player"] in lp["top2"].get(m["match_id"], [])
            cells.append(f'<td class="lp-rt{" top" if top else ""}">{"" if r is None else f"{r:.2f}"}</td>')
        rows.append(f'<tr class="lp-row{" star" if star else ""}"><td class="lp-name">{pl["player"]}</td>{"".join(cells)}'
                    f'<td class="lp-avg">{pl["avg"]:.2f}</td></tr>')
    return ('<div class="gfx lp"><div class="gfx-head"><span class="gfx-title">LOUD in the Champions Shanghai playoffs</span>'
            '<span class="gfx-sub">VLR rating by series</span></div>'
            f'<table class="lp-tbl"><thead><tr><th></th>{head}<th>Avg</th></tr></thead><tbody>{"".join(rows)}</tbody></table>'
            '<div class="mc-note">Highlighted: the two highest-rated LOUD players in that series</div></div>')


PAGE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=820">
<title>EMERGENCY ARTICLE: Previewing Champions Shanghai Grand Finals! &mdash; Bobo's VCT Database</title>
<meta property="og:type" content="article">
<meta property="og:site_name" content="Bobo gg">
<meta property="og:title" content="EMERGENCY ARTICLE: Previewing Champions Shanghai Grand Finals!">
<meta property="og:description" content="With LOUD and Vitality locked in for the grand finals and 12 hours to go, let&#39;s discuss what to expect and lock in some predictions.">
<meta property="og:url" content="https://bobo-gg.net/articles/champions-shanghai-grand-finals-preview/">
<meta property="og:image" content="https://bobo-gg.net/static/gf-preview/cover.jpg">
<meta property="og:image:secure_url" content="https://bobo-gg.net/static/gf-preview/cover.jpg">
<meta property="og:image:type" content="image/jpeg">
<meta property="og:image:width" content="2200">
<meta property="og:image:height" content="1467">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="EMERGENCY ARTICLE: Previewing Champions Shanghai Grand Finals!">
<meta name="twitter:description" content="With LOUD and Vitality locked in for the grand finals and 12 hours to go, let&#39;s discuss what to expect and lock in some predictions.">
<meta name="twitter:image" content="https://bobo-gg.net/static/gf-preview/cover.jpg">
<link rel="icon" type="image/svg+xml" href="/favicon.svg">
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@700;800&family=DM+Sans:wght@300;400;500;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="/static/base.css">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<style>
  html { -webkit-text-size-adjust:100%; text-size-adjust:100%; }
  .page { position:relative; z-index:1; flex:1; display:flex; flex-direction:column; align-items:center; padding:60px 32px 80px; }
  .article { max-width:860px; width:100%; }
  h1 { font-family:'Plus Jakarta Sans',sans-serif; font-size:clamp(1.55rem,3.1vw,2.4rem); font-weight:800; letter-spacing:-1px; line-height:1.14; margin-bottom:24px; text-align:center; white-space:nowrap; }
  .byline { font-size:.82rem; color:var(--soft); font-weight:300; margin-bottom:48px; padding-bottom:32px; border-bottom:1px solid #e8e0ec; text-align:center; }
  .cover { width:100%; border-radius:16px; overflow:hidden; margin-bottom:12px; }
  .cover img { width:100%; height:auto; display:block; }
  .cover-caption { font-size:min(.71rem, calc(1.3vw - 0.6px)); white-space:nowrap; color:var(--soft); font-weight:300; font-style:italic; margin-bottom:48px; text-align:center; }
  .content p { font-size:1rem; font-weight:300; line-height:1.8; color:var(--ink); margin-bottom:24px; }
  .content b { font-weight:700; }
  .content h2 { font-family:'Plus Jakarta Sans',sans-serif; font-size:1.54rem; font-weight:800; letter-spacing:-0.5px; margin:48px 0 20px; }
  .content h3 { font-family:'Plus Jakarta Sans',sans-serif; font-size:1.15rem; font-weight:800; letter-spacing:-0.3px; margin:34px 0 14px; }
  .content h3 .h3-logo { width:28px; height:28px; object-fit:contain; vertical-align:-7px; margin-right:9px; }
  .content h2, .cover { scroll-margin-top:84px; }
  .content .secbreak { border:0; border-top:1px solid var(--ink); margin:52px 0 40px; }
  .content .xlink { color:#7c4dd6; font-weight:500; text-decoration:underline;
                    text-decoration-thickness:1px; text-underline-offset:2px; }
  .content .xlink:hover { color:#5b21b6; }
  .content em { font-style:italic; }
  /* Bubble: same box language as the Championship DNA takeaways. */
  .bubble { max-width:720px; margin:4px auto 28px; text-align:center; background:#fff;
            border:1.5px solid #e0d4ec; border-radius:22px; padding:22px 30px; box-shadow:0 6px 26px #0000000f; }
  .bubble em { display:block; font-style:italic; font-weight:500; font-size:.98rem; line-height:1.7; color:#3d1a6e; }
  .bubble em + em { margin-top:16px; }
  .content ol { margin:0 0 24px; padding-left:1.6em; }
  .content ol li { font-size:1rem; font-weight:300; line-height:1.8; color:var(--ink); margin-bottom:14px; padding-left:.3em; }
  /* ── static BenPom graphics ── */
  .gfx { background:white; border-radius:14px; box-shadow:0 4px 24px #0000000a; border:1px solid #efe7f6;
         padding:16px 18px; margin:6px auto 32px; width:fit-content; max-width:100%; box-sizing:border-box; }
  .gfx-head { display:flex; align-items:baseline; justify-content:space-between; gap:16px; margin-bottom:10px; }
  .gfx-title { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:.95rem; color:var(--ink); }
  .gfx-sub { font-size:.68rem; color:var(--soft); font-weight:500; white-space:nowrap; }
  .rk { min-width:400px; }
  .rk-tbl { width:100%; border-collapse:collapse; font-size:.86rem; }
  .rk-tbl th { font-family:'Plus Jakarta Sans',sans-serif; font-size:.6rem; font-weight:800; letter-spacing:.08em;
               text-transform:uppercase; color:var(--soft); text-align:left; padding:4px 8px 6px; border-bottom:1px solid #ece3f5; }
  .rk-tbl th:first-child, .rk-tbl th:last-child { text-align:right; }
  .rk-tbl td { padding:7px 8px; border-top:1px solid #f3eef9; vertical-align:middle; }
  .rk-row:first-child td { border-top:0; }
  .rk-n { text-align:right; font-variant-numeric:tabular-nums; color:var(--soft); font-weight:600; width:22px; }
  .rk-team { display:flex; align-items:center; gap:9px; font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; color:var(--ink); }
  .rk-team img { width:22px; height:22px; object-fit:contain; flex-shrink:0; }
  .rk-rt { text-align:right; font-variant-numeric:tabular-nums; font-weight:700; color:var(--ink); }
  .rk-row.hl td { background:rgba(124,58,237,.07); }
  .rk-row.hl td:first-child { box-shadow:inset 3px 0 0 #7c3aed; }
  .reg { display:inline-block; font-family:'Plus Jakarta Sans',sans-serif; font-size:.58rem; font-weight:800; letter-spacing:.05em;
         text-transform:uppercase; padding:2px 7px; border-radius:99px; color:white; }
  .reg.emea { background:#16a34a; } .reg.americas { background:#ea580c; } .reg.pacific { background:#2563eb; } .reg.cn { background:#db2777; }
  .mc { width:440px; }
  .mc-head { display:flex; align-items:center; gap:14px; margin-bottom:8px; }
  .mc-team { display:flex; flex-direction:column; align-items:center; gap:3px; width:72px; flex-shrink:0; }
  .mc-team img { width:40px; height:40px; object-fit:contain; }
  .mc-team-name { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:.95rem; color:var(--ink); }
  .mc-seed { font-size:.56rem; font-weight:700; letter-spacing:.05em; text-transform:uppercase; color:var(--soft); white-space:nowrap; }
  .mc-prob { flex:1; min-width:0; }
  .mc-prob-pcts { display:flex; justify-content:space-between; gap:10px; font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.25rem; color:var(--soft); white-space:nowrap; }
  .mc-prob-pcts .fav { color:#16a34a; }
  .mc-bar { height:10px; border-radius:99px; background:#ece4f6; overflow:hidden; margin-top:5px; }
  .mc-bar-a { height:100%; background:#7c3aed; }
  .mc-fmt { text-align:center; font-size:.62rem; text-transform:uppercase; letter-spacing:.06em; color:var(--soft); font-weight:700; margin-bottom:8px; }
  .mc-sec-lbl { font-family:'Plus Jakarta Sans',sans-serif; font-size:.64rem; font-weight:800; text-transform:uppercase; letter-spacing:.07em; color:var(--soft); margin:12px 0 6px; }
  .mc-veto { display:flex; flex-direction:column; gap:4px; }
  .mc-veto-row { display:flex; align-items:center; gap:10px; font-size:.82rem; font-weight:600; color:var(--ink); }
  .mc-veto-row img { width:34px; height:22px; object-fit:cover; border-radius:4px; flex-shrink:0; }
  .mc-veto-tag { margin-left:auto; font-family:'Plus Jakarta Sans',sans-serif; font-size:.6rem; font-weight:800; text-transform:uppercase; letter-spacing:.03em; padding:2px 7px; border-radius:5px; white-space:nowrap; }
  .mc-tag-ban { background:#fde8e8; color:#a51d1d; } .mc-tag-pick { background:#e0effb; color:#1e5a9e; } .mc-tag-dec { background:#f0ecf4; color:#555; }
  .mc-maps { width:100%; border-collapse:collapse; font-size:.8rem; }
  .mc-maps td { padding:4px 6px; border-top:1px solid #f3eef9; }
  .mc-maps td:first-child { font-weight:600; color:var(--ink); }
  .mc-maps .mc-mw { text-align:right; font-variant-numeric:tabular-nums; font-weight:700; color:var(--soft); }
  .mc-maps .mc-mw.fav { color:#16a34a; }
  .mc-note { margin-top:10px; font-size:.62rem; color:var(--soft); text-align:center; }
  .ov { width:480px; }
  .ov-tbl { width:100%; border-collapse:collapse; font-size:.82rem; }
  .ov-tbl th { font-family:'Plus Jakarta Sans',sans-serif; font-size:.6rem; font-weight:800; letter-spacing:.08em;
               text-transform:uppercase; color:var(--soft); text-align:right; padding:4px 6px 6px; border-bottom:1px solid #ece3f5; }
  .ov-tbl th:first-child { text-align:left; }
  .ov-tbl td { padding:5px 6px; border-top:1px solid #f3eef9; vertical-align:middle; }
  .ov-tbl tr:first-child td { border-top:0; }
  .ov-map { display:flex; align-items:center; gap:10px; font-weight:600; color:var(--ink); }
  .ov-map img { width:34px; height:22px; object-fit:cover; border-radius:4px; flex-shrink:0; }
  .ov-rt { text-align:right; font-variant-numeric:tabular-nums; font-weight:700; color:var(--soft); }
  .ov-edge { text-align:right; font-variant-numeric:tabular-nums; font-weight:800; color:var(--soft); }
  .ov-edge.pos { color:#16a34a; } .ov-edge.neg { color:#a51d1d; }
  .ov-tag { text-align:right; }
  .cn { width:520px; }
  .cn-tiles { display:flex; gap:12px; margin:4px 0 14px; }
  .cn-tile { flex:1; border-radius:12px; padding:12px 14px; background:#f7f3fb; border:1px solid #efe7f6; text-align:center; }
  .cn-tile.neon { background:#eef7ff; border-color:#d6e8fa; }
  .cn-lbl { font-family:'Plus Jakarta Sans',sans-serif; font-size:.62rem; font-weight:800; letter-spacing:.07em; text-transform:uppercase; color:var(--soft); }
  .cn-rec { font-family:'Plus Jakarta Sans',sans-serif; font-size:1.9rem; font-weight:800; letter-spacing:-1px; color:var(--ink); line-height:1.15; margin-top:4px; }
  .cn-pct { font-family:'Plus Jakarta Sans',sans-serif; font-size:.95rem; font-weight:800; color:#1e5a9e; }
  .cn-tile.other .cn-pct { color:#5b21b6; }
  .cn-pct span { font-family:'DM Sans',sans-serif; font-size:.66rem; font-weight:500; color:var(--soft); }
  .cn-sub { font-size:.64rem; color:var(--soft); margin-top:3px; }
  .cn-tbl { width:100%; border-collapse:collapse; font-size:.74rem; }
  .cn-tbl th { font-family:'Plus Jakarta Sans',sans-serif; font-size:.56rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase; color:var(--soft); text-align:left; padding:3px 5px 5px; border-bottom:1px solid #ece3f5; }
  .cn-tbl th:last-child { text-align:right; }
  .cn-tbl td { padding:3px 5px; border-top:1px solid #f3eef9; vertical-align:middle; white-space:nowrap; }
  .cn-tbl tr:first-child td { border-top:0; }
  .cn-stage { color:var(--soft); font-size:.68rem; }
  .cn-map { display:flex; align-items:center; gap:7px; font-weight:600; color:var(--ink); }
  .cn-map img { width:26px; height:17px; object-fit:cover; border-radius:3px; flex-shrink:0; }
  .cn-opp img { width:15px; height:15px; object-fit:contain; vertical-align:-3px; margin-right:5px; }
  .cn-ag { font-weight:600; color:var(--soft); }
  .cn-ag.neon { color:#1e5a9e; font-weight:800; }
  .cn-r { text-align:right; }
  .cn-res { font-family:'Plus Jakarta Sans',sans-serif; font-size:.62rem; font-weight:800; padding:2px 7px; border-radius:5px; }
  .cn-res.w { background:#e3f5ea; color:#15803d; } .cn-res.l { background:#fde8e8; color:#a51d1d; }
  .ie { width:790px; }
  .ie-cols { display:flex; gap:12px; align-items:flex-start; }
  .ie-col { flex:1; min-width:0; background:#f7f3fb; border:1px solid #efe7f6; border-radius:12px; padding:14px 12px 10px; text-align:center; }
  .ie-head { width:60px; height:60px; border-radius:50%; object-fit:cover; display:block; margin:0 auto 6px; background:#e9e1f2; }
  .ie-name { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:.95rem; color:var(--ink); }
  .ie-count { font-family:'Plus Jakarta Sans',sans-serif; font-size:2.3rem; font-weight:800; letter-spacing:-1.5px; color:#5b21b6; line-height:1.1; margin-top:2px; }
  .ie-lbl { font-family:'Plus Jakarta Sans',sans-serif; font-size:.6rem; font-weight:800; letter-spacing:.07em; text-transform:uppercase; color:var(--soft); }
  .ie-sub { font-size:.68rem; color:var(--soft); margin:2px 0 10px; }
  .ie-list { list-style:none; margin:0; padding:8px 0 0; border-top:1px solid #e9e1f2; text-align:left; }
  .ie-ev { display:flex; align-items:baseline; gap:6px; font-size:.7rem; line-height:1.55; color:var(--ink); white-space:nowrap; }
  .ie-yr { color:var(--soft); font-variant-numeric:tabular-nums; flex-shrink:0; }
  .ie-nm { font-weight:600; overflow:hidden; text-overflow:ellipsis; }
  .ie-pl { margin-left:auto; color:var(--soft); flex-shrink:0; }
  .ie-ev.win .ie-pl { color:#15803d; font-weight:800; }
  .ie-ev.cur { color:#5b21b6; } .ie-ev.cur .ie-nm, .ie-ev.cur .ie-pl { color:#5b21b6; font-weight:800; }
  .lp { width:560px; }
  .lp-tbl { width:100%; border-collapse:collapse; font-size:.86rem; }
  .lp-tbl th { padding:4px 8px 8px; border-bottom:1px solid #ece3f5; text-align:center; vertical-align:bottom; font-weight:400; }
  .lp-tbl th:first-child { text-align:left; }
  .lp-tbl th:last-child { font-family:'Plus Jakarta Sans',sans-serif; font-size:.6rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase; color:var(--soft); }
  .lp-stage { display:block; font-family:'Plus Jakarta Sans',sans-serif; font-size:.6rem; font-weight:800; letter-spacing:.07em; text-transform:uppercase; color:var(--soft); }
  .lp-opp { display:flex; align-items:center; justify-content:center; gap:5px; font-size:.78rem; font-weight:700; color:var(--ink); margin-top:3px; }
  .lp-opp img { width:16px; height:16px; object-fit:contain; }
  .lp-res { display:inline-block; margin-top:4px; font-family:'Plus Jakarta Sans',sans-serif; font-size:.58rem; font-weight:800; padding:1px 6px; border-radius:5px; }
  .lp-res.w { background:#e3f5ea; color:#15803d; } .lp-res.l { background:#fde8e8; color:#a51d1d; }
  .lp-tbl td { padding:7px 8px; border-top:1px solid #f3eef9; text-align:center; font-variant-numeric:tabular-nums; }
  .lp-row:first-child td { border-top:0; }
  .lp-name { text-align:left !important; font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; color:var(--ink); }
  .lp-rt { font-weight:600; color:var(--soft); }
  .lp-rt.top { color:#5b21b6; font-weight:800; background:rgba(124,58,237,.08); }
  .lp-avg { font-weight:700; color:var(--ink); }
  .lp-row.star .lp-name { color:#5b21b6; }
  .ie.five { width:820px; }
  .ie.five .ie-col { padding:12px 8px 8px; }
  .ie.five .ie-ev { font-size:.66rem; flex-wrap:wrap; row-gap:0; }
  .ie.five .ie-pl { margin-left:0; flex-basis:100%; padding-left:34px; }
  .ie.five .ie-count { font-size:2.1rem; }
  @media (max-width:560px) { .rk { min-width:0; width:100%; } .mc { width:100%; } .ov { width:100%; } .cn { width:100%; } .cn-tiles { flex-direction:column; } .ie { width:100%; } .ie-cols { flex-direction:column; } .lp { width:100%; } }
  .toc { position:fixed; top:32px; right:32px; background:#fff; border-radius:16px;
         padding:18px 20px; box-shadow:0 4px 24px #0000000f; display:flex;
         flex-direction:column; gap:6px; z-index:100; width:max-content;
         max-width:min(215px, calc(50vw - 480px)); }
  .toc-title { font-family:'Plus Jakarta Sans',sans-serif; font-size:.72rem; font-weight:800;
               letter-spacing:.1em; text-transform:uppercase; color:var(--soft); margin-bottom:4px;
               text-align:center; padding-left:.1em; }
  .toc a { font-size:.76rem; color:var(--soft); text-decoration:none; font-weight:400;
           transition:color .15s; line-height:1.4; text-align:center; }
  .toc a:hover { color:var(--ink); }
  .toc a.active { color:var(--ink); font-weight:500; }
  .alpha-navbar ~ .toc { top:72px; }
  @media (max-width:1500px) { .toc { display:none; } }
</style>
</head>
<body>
<nav class="toc">
  <div class="toc-title">Sections</div>
  __TOC__
</nav>
<div class="page">
  <div class="article">
    <h1>EMERGENCY ARTICLE:<br>Previewing Champions Shanghai Grand Finals!</h1>
    <div class="byline">Bobo &mdash; October 17, 2026</div>
    <div class="cover" id="intro">
      <img src="/static/gf-preview/cover.jpg" alt="Lukxo of LOUD">
    </div>
    <p class="cover-caption">lukxo, surrounded by his LOUD teammates, decompresses following their 2-1 victory over Vitality in the Upper Finals. The two teams will rematch in the grand finals.</p>
    <div class="content">
      <p>12 hours ago, Vitality upset 100 Thieves in a 3-1 lower finals match, earning their spot in the Champions Shanghai grand finals - the first grand final in Vitality’s VCT history. 8 hours from now, they will face LOUD in the grand final, which will also be their 3rd matchup within this event alone. The stage is set for the final match of Champions Shanghai, the final match of VCT 2026, and the final match of VCT Franchising as we know it. Fortunately, this matchup is postured to be an electric and titanic clash, a deserving finish.</p>
      <p>With the aforementioned 8 hours until this event (aka I’m writing with a deadline), I have little time to spare, so let’s dive into the match and finalize predictions!</p>

      <h2 id="numbers">The Numbers</h2>
      <p>For those who aren’t familiar, this website has a rating algorithm called <a class="xlink" href="/mapelo/modern/" target="_blank" rel="noopener">BenPom</a>, which automatically calculates power rankings as well as previews matches. You can explore it more in the BenPom tab, but for now let’s look at what it says for the grand final match.</p>
      __RANKINGS__
      <p>LOUD and Vitality are neck and neck. LOUD have a 7.72 rating, 2nd in the world, while Vitality have a 7.03 rating, 3rd in the world. Exciting!</p>
      <p>Now, using BenPom’s algorithm to predict the map pick/ban given LOUD’s advantage coming from the uppers, what’s the estimated win probability for the match?</p>
      __MATCHCARD__
      <p>However, the map pick/ban is anything but a foregone conclusion. In their first two matchups, the map pick/ban was different:</p>
      <div class="bubble">
        <em>In Group Stage:<br>VIT ban Sunset; LOUD ban Split; VIT pick Ascent; LOUD pick Summit; VIT ban Abyss; LOUD ban Haven; Lotus remains</em>
        <em>In Playoffs:<br>VIT ban Sunset; LOUD ban Ascent; VIT pick Lotus; LOUD pick Summit; VIT ban Abyss; LOUD ban Split; Haven remains</em>
      </div>
      <p>Two things to consider:</p>
      <ol>
        <li>What will LOUD ban? As you can see above, they switched from banning Split to banning Ascent against Vitality. Even against other teams recently, they’ve been jumping around. Before using their first ban on Ascent against Vitality in playoffs, they used their first ban on Lotus against NRG, Ascent against PRX, and then Split (as mentioned previously) against VIT in Groups.</li>
        <li>What to do about Lotus? LOUD have floated Lotus against Vitality in their two Bo3s at Champs, playing it once (and losing). Having this float allowed them to keep their better maps available (Summit, Sunset, Abyss) and ban one of Vitality’s better maps. As the tournament has gone on, though, this strategy has looked more dubious. Vitality won Lotus both times they’ve played it at Champs: against LOUD themselves, and against 100T, who were considered favorites coming into the tournament. Will they continue to float Lotus? I wouldn’t be surprised if they’re scared enough to ban it, and I probably would as well if I were in LOUD’s shoes.</li>
      </ol>
      <p>According to BenPom the optimal pick/ban from LOUD is:</p>
      __OPTIMAL__
      <p>Interestingly, the algorithm <em>predicts</em> that LOUD will open by banning Ascent/Split based on their recent matches, while the algorithm thinks the <em>optimal</em> opening ban from LOUD is Ascent/Lotus. These types of discrepancies are extremely interesting, and something that could make or break the grand final.</p>
      <p>My opinion is that, the more maps you can play where Chronicle is on Neon, the better. For instance:</p>
      __CHRONICLE__
      <p>It’s not the craziest difference in the world, but it’s there. His neon has looked easy to punish throughout the matches he’s played it. LOUD have had a day off to watch and counterstrat, so I hope they will take advantage of that fact (and Chronicle’s Neon).</p>

      <h2 id="narratives">The Narratives</h2>
      <p>Irrespective of the map pick/ban, this match has a lot of pull and drama going into it. In many ways, these teams couldn’t be more dissimilar.</p>

      <h3><img class="h3-logo" src="/logos/VIT.png" alt="">VITALITY:</h3>
      <p>Vitality are a historic organization in the whole of Esports, especially CS and Rocket League. On their roster, they have titans of VCT in Chronicle and Derke, two consensus top-10 players of all time. This organization has continually funneled money into picking up top talent, both old (Derke, Chronicle, Sayf, Less) and new (CyvOph and Sayonara). After two super teams and multiple roster shuffles, the organization has finally reached the apex of VCT at Champions Shanghai. Even Jamppi has been a face of the sport since 2021, reaching an international grand final after 5 years.</p>
      <p>Sayonara and Profek are two outliers on the roster; they are younger, support players who have risen to the moment on this historic, decorated roster - especially Sayonara. Both of them are at their second international VCT event ever (both from Vitality, this year). Meanwhile:</p>
      __INTL__

      <h3><img class="h3-logo" src="/logos/LOUD.png" alt="">LOUD:</h3>
      <p>The LOUD roster, however, is an opposite story. This is the first international VCT event for every single one of the players on Loud’s roster. Any of the players in the above Vitality graphic (Jamppi, Derke, and Chronicle) ALONE have more internationals attended than the ENTIRE LOUD roster.</p>
      __LOUD_INTL__
      <p>Since Aspas left LOUD in 2024, the historically dominant Valorant organization has accomplished little to nothing, only reaching Masters Madrid in 2024. Not only are the LOUD players young and hungry, but so are the fans. The Brazilian fanbase is massive and have been starved of any competitive rosters, so they’ve rallied around this LOUD team with fervor. While Vitality have had superteam after superteam, shuffling in high level talent from around the world, LOUD have had 0 expectations to succeed over the previous years, with this young South American squad shattering expectations.</p>

      <h2 id="prediction">My Prediction:</h2>
      <p>I have to stick with LOUD here. Every angle I look at, it’s hard not to favor them. In my <a class="xlink" href="/articles/championship-dna/" target="_blank" rel="noopener">historical trends article</a> I highlighted them as one of the three teams that history favors. By the numbers (BenPom) they’re favored, both in rating and in the projected probability. LOUD also have map pick/ban advantage, coming from uppers. EG remains the only team to win Champions coming from the lower bracket, back in 2023.</p>
      <p>The biggest caveat to LOUD is how heavily they rely on two players: Erde and Lukxo. In Champions Shanghai playoffs, Erde and Lukxo were the two highest rated players on their team every time, and both have had above a 1.25 rating every time.</p>
      __LOUDPO__
      <p>That is insane, but they’re also two rookies, in the biggest match of their lives, at their first international event ever. If one of the two falter, I can’t imagine the other LOUD players being able to replace that level of impact.</p>
      <p>My final prediction is LOUD 3-1 VIT.</p>
    </div>
  </div>
</div>
<script>
(function() {
  var tocLinks = document.querySelectorAll('.toc a');
  var ids = Array.from(tocLinks).map(function(a) { return a.getAttribute('href').slice(1); });
  function onScroll() {
    var scrollY = window.scrollY + 120;
    var active = ids[0];
    ids.forEach(function(id) {
      var el = document.getElementById(id);
      if (el && el.offsetTop <= scrollY) active = id;
    });
    tocLinks.forEach(function(a) {
      a.classList.toggle('active', a.getAttribute('href') === '#' + active);
    });
  }
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();
})();
</script>
</body>
</html>
"""


@article_gf_preview_bp.route("/")
def article_gf_preview():
    d = _load()
    return (PAGE_HTML
            .replace("__TOC__", _toc_html())
            .replace("__RANKINGS__", _rankings_html(d))
            .replace("__MATCHCARD__", _match_card_html(d))
            .replace("__OPTIMAL__", _optimal_card_html(d))
            .replace("__CHRONICLE__", _chronicle_html(d))
            .replace("__INTL__", _intl_html(d))
            .replace("__LOUD_INTL__", _loud_intl_html(d))
            .replace("__LOUDPO__", _loud_po_html(d)))
