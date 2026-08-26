"""Article: Championship DNA — Historical Trends To Note For Champions Shanghai

Scaffold only. Title + hero image; the body is intentionally EMPTY, waiting on
the author's copy. Nothing in here invents prose — no deck, no caption, no
section headings, no category label — so whatever lands in `.content` is the
author's words and only the author's words.

Layout mirrors AspasGreatestPrime / the Masters previews: centred 860px column,
label / h1 / byline / cover / content. Add the `.toc` nav and a `.label` when
the sections and category exist.
"""

import os
import json
from flask import Blueprint, render_template_string

article_championship_dna_bp = Blueprint("article_championship_dna", __name__)

_ROOT = os.path.dirname(os.path.abspath(__file__))
_LANDSCAPE = os.path.join(_ROOT, "data", "enriched", "side_landscape.json")
_STREAKS   = os.path.join(_ROOT, "data", "enriched", "momentum_streaks.json")
_STARS     = os.path.join(_ROOT, "data", "enriched", "star_power.json")

# Each international's "Before <event>" snapshot. Ranks are read through
# MapElo.benpom_snapshot_board so the article shows exactly what
# /mapelo/rankings/ shows — computing them any other way silently disagreed
# (see that helper's docstring).
_BEFORE_SNAP = [
    ("Masters Tokyo 2023",    "2023", "before_tokyo"),
    ("Champions 2023",        "2023", "before_champions"),
    ("Masters Madrid 2024",   "2024", "before_madrid"),
    ("Masters Shanghai 2024", "2024", "before_shanghai"),
    ("Champions 2024",        "2024", "before_champions"),
    ("Masters Bangkok 2025",  "2025", "before_bangkok"),
    ("Masters Toronto 2025",  "2025", "before_toronto"),
    ("Champions 2025",        "2025", "before_champions"),
    ("Masters Santiago 2026", "2026", "before_santiago"),
    ("Masters London 2026",   "2026", "before_london"),
]

# Roster turnover behind each title, as supplied by the author. Held as data
# rather than markup so the outs and ins can be styled apart from each other;
# the wording of every label and name is theirs.
_ROSTERS = [
    # Two cards, one roster: the same offseason changes carried FNATIC through
    # both titles, and merging them into one card hid that they won twice.
    ("FNC", 2023, "LOCK//IN", "FNATIC", [
        ("preseason", ["Mistic", "Enzo"], ["Chronicle", "Leo"]),
    ]),
    ("FNC", 2023, "Masters Tokyo", "FNATIC", [
        ("preseason", ["Mistic", "Enzo"], ["Chronicle", "Leo"]),
    ]),
    ("EG", 2023, "Champions LA", "EG", [
        ("preseason", ["Apoth", "Reformed"], ["BcJ", "Ethan"]),
        ("midseason", ["BcJ"], ["Demon1"]),
    ]),
    ("SEN", 2024, "Masters Madrid", "Sentinels", [
        ("preseason", ["Pancada", "Marved"], ["JohnQT", "Zellsis"]),
    ]),
    ("GEN", 2024, "Masters Shanghai", "Gen.G", [
        ("preseason", ["TS", "k1Ng", "Secret", "eKo", "GodDead"],
             ["t3xture", "Munchkin", "Karon", "Lakia"]),
    ]),
    ("EDG", 2024, "Champions Seoul", "EDG", [
        ("midseason", ["Haodong"], ["S1Mon"]),
    ]),
    ("T1", 2025, "Masters Bangkok", "T1", [
        ("preseason", ["Sayaplayer", "Rossy", "xccurate"], ["Meteor", "BuZz", "Sylvan"]),
    ]),
    ("PRX", 2025, "Masters Toronto", "PRX", [
        ("midseason", ["mindfreak"], ["PatMen"]),
    ]),
    ("NRG", 2025, "Champions Paris", "NRG", [
        ("preseason", ["crashies", "Victor"], ["Verno", "Mada"]),
        ("midseason", ["Verno", "FNS"], ["Brawk", "skuba"]),
    ]),
    ("NS", 2026, "Masters Santiago", "NS RedForce", [
        ("preseason", ["margaret", "Persia"], ["Rb", "Xross"]),
    ]),
    ("LEV", 2026, "Masters London", "Leviat\u00e1n", [
        ("preseason", ["C0M", "tex"], ["spikeziN", "blowz"]),
        ("midseason", ["PxS"], ["Neon"]),
    ]),
]


def _rosters_html():
    """Static markup: this never changes between requests, unlike the charts."""
    # Grouped by season, reusing the star cards' year header so the two
    # sections read as the same kind of list.
    TAGS = {"preseason": "(Preseason)", "midseason": "(Midseason)"}
    years, cards = [], {}
    for org, year, event, team, rows in _ROSTERS:
        lines = []
        for tag, gone, came in rows:
            # Outs and ins get a row each. Nine names on one line -- Gen.G's
            # preseason rebuild -- cannot fit a third of the figure at any
            # readable size, and splitting on the natural seam keeps every row
            # inside the card without shrinking the type to nothing.
            block = [f'<div class="ro-tag">{TAGS[tag]}</div>'] if tag else []
            if gone:
                block.append('<div class="ro-line">'
                             + "".join(f'<span class="ro-out">&minus; {n}</span>' for n in gone)
                             + "</div>")
            if came:
                block.append('<div class="ro-line">'
                             + "".join(f'<span class="ro-in">+ {n}</span>' for n in came)
                             + "</div>")
            lines.append('<div class="ro-change">' + "".join(block) + "</div>")
        if year not in cards:
            years.append(year); cards[year] = []
        cards[year].append(
            '<div class="ro">'
            '<div class="ro-head">'
            f'<div class="ro-evt">{event}</div>'
            f'<div class="ro-team"><img src="/logos/{org}.png" alt=""><span>{team}</span></div>'
            '</div>'
            + "".join(lines) + "</div>")
    return "\n".join(
        f'<div><div class="star-year"><span>{y}</span><i></i></div>'
        f'<div class="ro-row">{"".join(cards[y])}</div></div>'
        for y in years)


_ls_cache = (None, -1.0)


def _landscape():
    """Attack/defense win% for every international-attending team over the split
    before that international. Built by scrapers/BuildSideLandscape.py — see
    there for why it is precomputed rather than derived per request."""
    global _ls_cache
    try:
        # Keyed on both files: the streaks ride along in the same payload, so a
        # rebuild of either one has to invalidate the cache.
        stamp = (os.path.getmtime(_LANDSCAPE), os.path.getmtime(_STREAKS),
                 os.path.getmtime(_STARS))
    except OSError:
        return {"points": [], "internationals": []}
    if _ls_cache[0] is not None and _ls_cache[1] == stamp:
        return _ls_cache[0]
    try:
        with open(_LANDSCAPE) as f:
            data = json.load(f)
    except Exception:
        data = {"points": [], "internationals": []}
    data["winner_ranks"] = _winner_ranks(data.get("winners") or {})
    # Each side file is loaded on its own, so a missing or half-written one
    # blanks its own section instead of taking the others down with it.
    for path, keys in ((_STREAKS, (("streaks", "winners"), ("last5", "tally"),
                                   ("success", "success"))),
                       (_STARS,   (("stars", "stars"),))):
        try:
            with open(path) as f:
                blob = json.load(f)
        except Exception:
            blob = {}
        for dest, src in keys:
            data[dest] = blob.get(src) or []
    _ls_cache = (data, stamp)
    return data


def _winner_ranks(winners):
    """Each winner's BenPom rank in that event's Before-<event> board."""
    try:
        from MapElo import benpom_snapshot_board
    except Exception:
        return []
    out = []
    for label, year, snap in _BEFORE_SNAP:
        org = winners.get(label)
        if not org:
            continue
        board, cutoff = benpom_snapshot_board(year, snap)
        rank = next((i + 1 for i, (o, _) in enumerate(board) if o == org), None)
        if rank is None:
            continue
        out.append({"event": label, "org": org, "rank": rank,
                    "pool": len(board), "as_of": cutoff})
    return out


PAGE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=820">
<title>Championship DNA: Historical Trends To Note For Champions Shanghai &mdash; Bobo's VCT Database</title>
<!-- Open Graph / Twitter link-preview cards. The description is the author's
     articles-index blurb, reused verbatim — same job, same words. -->
<meta property="og:type" content="article">
<meta property="og:site_name" content="Bobo gg">
<meta property="og:title" content="Championship DNA: Historical Trends To Note For Champions Shanghai">
<meta property="og:description" content="Understanding the indicators of a championship team - by the numbers, by the rosters, by the regions, and other miscellaneous trends.">
<meta property="og:url" content="https://bobo-gg.net/articles/championship-dna/">
<meta property="og:image" content="https://bobo-gg.net/championshipdna.jpg">
<meta property="og:image:secure_url" content="https://bobo-gg.net/championshipdna.jpg">
<meta property="og:image:type" content="image/jpeg">
<meta property="og:image:width" content="2048">
<meta property="og:image:height" content="1404">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="Championship DNA: Historical Trends To Note For Champions Shanghai">
<meta name="twitter:description" content="Understanding the indicators of a championship team - by the numbers, by the rosters, by the regions, and other miscellaneous trends.">
<meta name="twitter:image" content="https://bobo-gg.net/championshipdna.jpg">
<link rel="icon" type="image/svg+xml" href="/favicon.svg">
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="/static/base.css">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<style>
  html { -webkit-text-size-adjust:100%; text-size-adjust:100%; }
  .page { position:relative; z-index:1; flex:1; display:flex; flex-direction:column; align-items:center; padding:60px 32px 80px; }
  .article { max-width:860px; width:100%; }
  .label { font-family:'Plus Jakarta Sans',sans-serif; font-size:0.77rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase; color:var(--soft); margin-bottom:16px; text-align:center; }
  /* This title is 66 characters — half again as long as the other articles'.
     At their 3.52rem it strands orphans, so the ceiling drops to 2.92rem
     and it sits on two even lines instead.
     text-wrap:balance rather than a hard <br>, so it stays even at any
     width instead of breaking in a fixed place that only suits one. */
  h1 { font-family:'Plus Jakarta Sans',sans-serif; font-size:clamp(1.55rem,3.1vw,2.25rem); font-weight:800; letter-spacing:-1px; line-height:1.14; margin-bottom:24px; text-align:center; }
  .nb { white-space:nowrap; }
  .deck { font-size:1.06rem; font-weight:300; color:var(--soft); line-height:1.6; text-align:center; max-width:640px; margin:-8px auto 22px; }
  .byline { font-size:.82rem; color:var(--soft); font-weight:300; margin-bottom:48px; padding-bottom:32px; border-bottom:1px solid #e8e0ec; text-align:center; }
  .cover { width:100%; border-radius:16px; overflow:hidden; margin-bottom:12px; }
  .cover img { width:100%; height:auto; display:block; }
  .cover-caption { font-size:.75rem; color:var(--soft); font-weight:300; font-style:italic; margin-bottom:48px; text-align:center; }
  .content p { font-size:1rem; font-weight:300; line-height:1.8; color:var(--ink); margin-bottom:24px; }
  .content h2 { font-family:'Plus Jakarta Sans',sans-serif; font-size:1.54rem; font-weight:800; letter-spacing:-0.5px; margin:48px 0 20px; }
  .content h2, .cover, .content .fig { scroll-margin-top:84px; }
  /* Paragraph anchors land tighter than the rest. The nav bar is ~42px tall, so
     84px left a ~42px gap under it — just enough for the tail of the previous
     line to peek out. 66px tucks that line behind the bar while still leaving
     the paragraph a comfortable ~17px of daylight. */
  .content p[id] { scroll-margin-top:66px; }
  /* Breaks out of the 860px text column. Ten event names cannot fit one line
     at any readable size inside 860px, and the scatter is dense enough that
     the extra width helps it too. Centred on the viewport, capped so it never
     runs to the screen edge. */
  /* Centred on the CONTENT column, not the viewport. Using 100vw here counted
     the scrollbar, so the wide charts sat a few px right of the body text and
     the images that sit between them — visible as everything looking slightly
     off-centre against everything else. */
  .fig { margin:34px 0 40px; width:min(1120px, calc(100% + 260px));
         margin-left:50%; transform:translateX(-50%); }
  .fig-wrap { position:relative; width:100%; aspect-ratio:1.25/1; background:#fff;
              border:1px solid #ece6f2; box-shadow:0 4px 24px #0000000a;
              overflow:hidden; }
  /* Scoped under .content: `.content p` is class+element, which outranks a bare
     .fig-note class, so an unscoped rule here silently lost every property to
     the body-paragraph style. */
  /* Centred in the bullet, and pulled back over the list indent so it lines up
     with the body column rather than sitting off to the right. */
  .inline-fig { margin:18px 0; }
  /* Only the in-bullet inset needs pulling back over the list indent. Applying
     it to every .inline-fig knocked the full-width reference image off centre. */
  li .inline-fig { margin-left:-22px; }
  .content .fig-credit { font-size:.68rem; font-weight:300; color:var(--soft);
                         text-align:center; margin-top:8px; word-break:break-all; }
  .content .fig-credit a { color:var(--soft); text-decoration:underline; }
  .content .fig-credit a:hover { color:#7c4dd6; }
  .inline-fig.wide img { display:block; width:100%; max-width:820px; height:auto; margin:0 auto;
                        border:1px solid #ece6f2; }
  .inline-fig-wrap { position:relative; width:100%; max-width:760px; aspect-ratio:1.15/1;
                     margin:0 auto; background:#fff; border:1px solid #ece6f2; overflow:hidden; }
  .content ul.notes { margin:0 0 24px; padding-left:22px; }
  .content ul.notes li { font-size:1rem; font-weight:300; line-height:1.8; color:var(--ink);
                         margin-bottom:12px; }
  /* Only as wide as the headline needs. Measured against the 1120px figure
     above it, that line renders ~871px, so 34px of padding each side puts
     the minimum at ~940px. 970px keeps ~30px of slack against font-
     rendering differences without running the full width of the figure. */
  .takeaway { max-width:970px; margin:24px auto 0; text-align:center;
              background:#fff; border:1.5px solid #e0d4ec; border-radius:22px;
              padding:26px 34px; box-shadow:0 6px 26px #0000000f; }
  /* Dash markers come from CSS, not the markup, so the copy stays clean. */
  .takeaway b::before, .takeaway span::before { content:'- '; white-space:pre; }
  .takeaway b, .takeaway span {
              display:block; font-family:'Plus Jakarta Sans',sans-serif; font-weight:800;
              font-style:italic; font-size:1.32rem; line-height:1.3; color:#3d1a6e;
              letter-spacing:-0.2px; }
  .takeaway span { margin-top:34px; }
  /* The rookie bubble carries a headline and a roster list rather than two
     short claims, so it takes the full figure width and drops the list to
     reading size. */
  .takeaway--roster { max-width:1120px; padding:26px 34px; }
  .takeaway--roster b { font-size:1.24rem; line-height:1.35; }
  .takeaway--roster span { font-size:1rem; font-weight:600; line-height:1.65; margin-top:16px; }
  /* The list is a sentence, not a second claim, so it takes no dash. */
  .takeaway--roster span::before { content:none; }
  /* Sections legend, top-right, matching every other article. Its width is
     clamped to the gutter beside the
     centred content — but this article's figures break out to 1120px, so the
     gutter is much narrower than on articles whose widest element is the 860px
     text column. Hidden entirely below 1500px, where there is no room at all. */
  .toc { position:fixed; top:32px; right:32px; background:#fff; border-radius:16px;
         padding:18px 20px; box-shadow:0 4px 24px #0000000f; display:flex;
         flex-direction:column; gap:6px; z-index:100; width:max-content;
         max-width:min(215px, calc(50vw - 585px)); }
  .toc-title { font-family:'Plus Jakarta Sans',sans-serif; font-size:.72rem; font-weight:800;
               letter-spacing:.1em; text-transform:uppercase; color:var(--soft); margin-bottom:4px; }
  .toc a { font-size:.76rem; color:var(--soft); text-decoration:none; font-weight:400;
           transition:color .15s; line-height:1.4; }
  .toc a:hover { color:var(--ink); }
  .toc a.sub { padding-left:12px; font-size:.72rem; }
  .alpha-navbar ~ .toc { top:72px; }
  @media (max-width:1500px) { .toc { display:none; } }
  /* Black, matching the body ink rather than the pale lilac the figure
     borders use -- these divide sections, so they should read as structure. */
  .content .secbreak { border:0; border-top:1px solid var(--ink); margin:46px 0 34px; }
  .content .xlink { color:#7c4dd6; font-weight:500; text-decoration:underline;
                    text-decoration-thickness:1px; text-underline-offset:2px; }
  .content .xlink:hover { color:#5b21b6; }
  .rank-wrap { aspect-ratio:2.1/1; }
  /* A rose chart is radial, so it wants a near-square frame rather than the
     2.1:1 the line and scatter charts use. */
  /* overflow:visible overrides .fig-wrap's clip: the bubble is allowed to
     hang off the white plate rather than being cut at its edge. */
  .polar-wrap { aspect-ratio:1.35/1; overflow:visible; }
  .streak-key { display:flex; justify-content:center; gap:20px; margin-bottom:10px;
                font-family:'DM Sans',sans-serif; font-size:.76rem; color:#7a6e7e; }
  .streak-key span { display:inline-flex; align-items:center; gap:7px; }
  .streak-key .sk { width:13px; height:13px; border-radius:3px; display:inline-block; }
  .streak-key .sk-top { background:#7c4dd6; }
  .streak-key .sk-rest { background:#e4dcf0; }
  /* HTML tooltip, used by both the rose and the stacked bars. Two reasons the
     canvas one could not do this job: it draws a single colour box per data
     item, so a logo per TEAM is impossible, and it re-picks its vertical
     alignment as the box grows -- which made the bubble sit at the bar's top on
     one bar and centred on the next. */
  .chart-tip { position:absolute; pointer-events:none; opacity:0; transition:opacity .14s linear;
               background:rgba(22,18,29,.94); color:#fff; border-radius:8px; padding:10px 12px;
               font-family:'DM Sans',sans-serif; font-size:.78rem; line-height:1.35;
               white-space:nowrap; z-index:5; }
  .chart-tip .rt-head { font-weight:700; margin-bottom:3px; }
  .chart-tip .rt-sub { color:#c9bfd6; margin-bottom:5px; }
  .chart-tip .rt-row { display:flex; align-items:center; gap:7px; }
  .chart-tip .rt-row + .rt-row { margin-top:3px; }
  .chart-tip .rt-row img { width:17px; height:17px; object-fit:contain; flex:0 0 17px; }
  .chart-tip .rt-place { color:#c9bfd6; }
  .tip-wrap { overflow:visible; }
  /* Star cards. A grid rather than a chart: the headshot is the point, and
     nine of them read better side by side than as labelled marks. */
  .stars { display:flex; flex-direction:column; gap:22px; }
  /* Grouped by season. Card widths stay uniform across groups, so a year with
     one winner gets a card the same size as a year with three. */
  .star-year { display:flex; align-items:center; gap:12px; margin-bottom:10px; }
  .star-year span { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:.82rem;
                    letter-spacing:.08em; color:#7c4dd6; }
  .star-year i { flex:1 1 auto; height:1px; background:#e8e0f2; }
  .star-row { display:grid; grid-template-columns:repeat(3, 1fr); gap:14px; }
  .star { background:#fff; border:1px solid #ece6f2; border-radius:16px; padding:16px 16px 14px;
          box-shadow:0 4px 20px #0000000a; display:flex; align-items:center; gap:14px;
          color:inherit; text-decoration:none;
          transition:transform .13s ease, box-shadow .13s ease, border-color .13s ease; }
  /* Only the linked ones lift -- a card that goes nowhere should not invite a
     click. Every card happens to have a leaderboard today, but the renderer
     still falls back to a plain div when one is missing. */
  a.star:hover { transform:translateY(-2px); border-color:#d9c9f0;
                 box-shadow:0 10px 26px #7c4dd61f; }
  /* No overflow:hidden here. The circle is clipped on the portrait itself, so
     the team badge can hang off the edge instead of being sliced by the
     parent's rounding. */
  .star-face { position:relative; flex:0 0 62px; width:62px; height:62px; border-radius:50%;
               background:#f3eefb; display:flex; align-items:center; justify-content:center; }
  .star-face > img { width:100%; height:100%; border-radius:50%;
                     object-fit:cover; object-position:top center; }
  /* Not every player has a headshot on file, so the initial stands in. */
  .star-face span { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.5rem;
                    color:#b9a9d4; }
  .star-face .star-org { position:absolute; right:-5px; bottom:-4px; width:26px; height:26px;
                         border-radius:50%; background:#fff; border:1px solid #ece6f2;
                         box-shadow:0 1px 5px #0000001a;
                         display:flex; align-items:center; justify-content:center; }
  .star-face .star-org img { width:18px; height:18px; object-fit:contain; }
  .star-body { min-width:0; flex:1 1 auto; }
  .star-name { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.02rem;
               color:#16121d; line-height:1.2; }
  .star-evt { font-size:.72rem; color:var(--soft); margin-top:2px; }
  .star-line { display:flex; align-items:baseline; gap:8px; margin-top:7px; }
  .star-rating { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.12rem;
                 color:#7c4dd6; line-height:1; }
  /* The rank is the headline number on the card -- a 1.14 rating means little
     on its own, "#3 of 50" is the part that says how good it was. */
  .star-rank { font-size:.72rem; color:var(--soft); }
  .star-rank b { font-family:'Plus Jakarta Sans',sans-serif; font-size:1.72rem; font-weight:800;
                 color:#3d1a6e; line-height:1; letter-spacing:-0.5px; margin-right:1px; }
  .star-split { font-size:.66rem; color:var(--soft); margin-top:4px; }
  .star-stat { font-size:.66rem; font-weight:700; color:#7c4dd6; letter-spacing:.04em;
               margin-left:-4px; }
  .star-note { font-size:.66rem; color:#b06a2c; margin-top:3px; }
  /* Says why the rank is 4 and not 5 when two players show the same figure. */
  .star-tie { font-style:italic; }
  /* .fig-note normally sits above a figure; this one sits under it. */
  .content .fig-note.below { margin:14px 0 0; }
  /* Roster turnover. Outs and ins are coloured rather than just signed, so the
     shape of a change reads before any name does. */
  /* Inside a .fig, so it breaks out to 1120px like the star cards instead of
     being held to the 860px text column. .fig supplies the outer margin. */
  .rosters { display:flex; flex-direction:column; gap:22px; }
  .ro-row { display:grid; grid-template-columns:repeat(3, 1fr); gap:14px; }
  .ro { background:#fff; border:1px solid #ece6f2; border-radius:16px; padding:16px 16px 18px;
        box-shadow:0 4px 20px #0000000a; min-width:0; text-align:center; }
  /* Tournament on top, team under it with its logo alongside. Centred, so each
     change block reads as a unit hanging off its own label. */
  .ro-head { margin-bottom:13px; padding-bottom:12px; border-bottom:1px solid #f2edf7; }
  .ro-evt { font-size:.71rem; font-weight:600; letter-spacing:.04em; color:var(--soft);
            line-height:1.2; text-transform:uppercase; }
  .ro-team { display:flex; align-items:center; justify-content:center; gap:8px; margin-top:4px; }
  .ro-team img { width:24px; height:24px; object-fit:contain; flex:0 0 24px; }
  .ro-team span { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1rem;
                  color:#16121d; line-height:1.2; }
  /* One row of outs, one row of ins, both centred under the label they belong
     to. flex-wrap stays on as a safety net -- nothing reaches it at these
     lengths, but a longer roster should push down rather than spill out. */
  .ro-change + .ro-change { margin-top:14px; padding-top:13px; border-top:1px dashed #efe9f5; }
  .ro-line { display:flex; flex-wrap:wrap; align-items:center; justify-content:center;
             gap:5px; margin-top:6px; }
  .ro-tag { display:inline-block; font-size:.63rem; font-weight:700; letter-spacing:.03em;
            color:var(--soft); background:#f4f0f8; border-radius:5px; padding:2px 6px;
            white-space:nowrap; margin-bottom:1px; }
  .ro-out, .ro-in { font-size:.71rem; font-weight:600; border-radius:6px; padding:2px 7px;
                    white-space:nowrap; flex:0 0 auto; }
  .ro-out { color:#a33b3b; background:#fbeeee; }
  .ro-in  { color:#2f7a54; background:#eaf6ef; }
  @media (max-width:900px) { .ro-row { grid-template-columns:repeat(2, 1fr); } }
  @media (max-width:620px) { .ro-row { grid-template-columns:1fr; } }
  .star--empty { background:#faf8fd; border-style:dashed; box-shadow:none; }
  .star-face--org { background:#fff; border:1px solid #ece6f2; }
  .star-face--org > img { width:60%; height:60%; border-radius:0; object-fit:contain; }
  .star-none { font-size:.76rem; font-style:italic; color:var(--soft); margin-top:8px; }
  @media (max-width:900px) { .star-row { grid-template-columns:repeat(2, 1fr); } }
  @media (max-width:620px) { .star-row { grid-template-columns:1fr; } }
  .content .pin { color:#7c4dd6; font-weight:500; text-decoration:underline;
                  text-decoration-thickness:1px; text-underline-offset:2px; cursor:pointer; }
  .content .pin:hover { color:#5b21b6; }
  .content .fig-note { font-size:.68rem; font-weight:300; color:var(--soft);
                       text-align:center; margin:0 0 14px; line-height:1.5; }
  .fig-filter { margin-bottom:14px; }
  .ls-win { display:flex; justify-content:center; margin-bottom:8px; }
  /* One line, always. The ten event names are far wider than the 860px column,
     so the row scrolls sideways rather than wrapping — same treatment as the
     site nav. Scrollbar hidden; it's still swipeable/shift-scrollable. */
  .ls-events { display:flex; flex-wrap:nowrap; gap:5px; justify-content:safe center;
               overflow-x:auto; scrollbar-width:none; padding-bottom:2px;
               --lsb-fs:.68rem; --lsb-px:10px; }
  .ls-events .lsb { font-size:var(--lsb-fs); padding:4px var(--lsb-px); }
  .ls-events::-webkit-scrollbar { display:none; }
  .ls-events .lsb { flex:0 0 auto; }
  .lsb { font-family:'DM Sans',sans-serif; font-size:.62rem; font-weight:700; color:var(--soft);
         background:#fff; border:1px solid #e8e0ec; border-radius:99px; padding:4px 9px; cursor:pointer;
         white-space:nowrap; transition:background .15s,border-color .15s,color .15s; }
  .lsb:hover { color:var(--ink); border-color:#c9b8d8; }
  .lsb.on { background:#7c4dd6; border-color:#7c4dd6; color:#fff; }
  .lsb-win { border-color:#e0c48a; color:#8a6a1a; font-size:.76rem; padding:7px 16px; }
  .lsb-state { display:inline-block; margin-left:7px; padding:1px 7px; border-radius:99px;
               font-size:.62rem; font-weight:800; letter-spacing:.05em;
               background:rgba(138,106,26,.14); color:#8a6a1a; }
  .lsb-win.on .lsb-state { background:rgba(255,255,255,.28); color:#fff; }
  .lsb-win.on { background:#d8a93a; border-color:#d8a93a; color:#fff; }
  @media (max-width:820px) {
    .page { padding:40px 18px 60px; }
  }
</style>
</head>
<body>
<nav class="toc">
  <div class="toc-title">Sections</div>
  <a href="#intro">Intro</a>
  <a href="#by-the-numbers">The Winners: By The Numbers</a>
  <a href="#sec-landscape" class="sub">Attack &amp; Defense</a>
  <a href="#sec-tiers" class="sub">Bands of Favoritism</a>
  <a href="#sec-benpom" class="sub">BenPom Rank</a>
  <a href="#sec-momentum" class="sub">Momentum</a>
  <a href="#sec-stars" class="sub">Star Power</a>
  <a href="#roster-composition">Roster Composition</a>
</nav>
<div class="page">
  <div class="article">
    <!-- Explicit break after the colon: the series name owns line 1, the
         subject owns line 2. The nowrap span keeps the event name whole if
         line 2 ever has to wrap on a narrow screen. Wording untouched. -->
    <h1>Championship DNA:<br>Historical Trends To Note For <span class="nb">Champions Shanghai</span></h1>
    <div class="byline">Bobo &mdash; August 2026</div>
    <div class="cover" id="intro">
      <img src="/championshipdna.jpg" alt="VCT champions lifting trophies">
    </div>
    <p class="cover-caption">International winners from each year of franchised VCT<br>FNATIC at Tokyo in 2023, Sentinels at Madrid in 2024, Paper Rex in 2025, and Leviatán at London in 2026</p>
    <div class="content">
      <p>Champions, the biggest event in the year for VCT, is right around the corner. This means that the entirety of the fanbase will be making Pick&rsquo;Ems, discussing their predictions online, casting bets, and constructing fantasy teams. Furthermore, Champions involves 4 teams from each region, as opposed to the 3 (and sometimes 2, in the past) from each region at Masters events. With such a wide pool of teams alongside a large, captive, and opinionated audience, there is no better time to look back on the (almost) 4 years of franchised VCT history! How can we use history to sort through these teams and see who is expected to fail and who are true favorites?</p>

      <p>This tradition of historical and analytical trends exists heavily in other sports and the results are often fascinating (and accurate). I am excited to borrow these ideas, frameworks, and visualizations I&rsquo;ve read over the years and bring them into the world of VCT! In this article, you will likely see a few references to college basketball/baseball analytics, so bear with me if that&rsquo;s unfamiliar. (Or you can skip them).</p>

      <h2 id="by-the-numbers">The Winners: By The Numbers</h2>

      <p>One of the simplest ways that a championship team is understood in any sport is by their offensive and defensive strength levels. Rely too heavily on one of these sides, and imbalance can often lead to failure. VCT is no different, except we&rsquo;re dealing with attack and defense rather than offense and defense. Here is a graph of every international-attending team, mapped by their attack win% and defense win% in the split prior <em>(e.g. Leviatan at London uses their numbers from Stage 1 of 2026)</em>.</p>

      <figure class="fig" id="sec-landscape">
        <p class="fig-note"><em>Note: Champions 2023 was not included, since there was no domestic split prior to the tournament</em></p>
        <div class="fig-filter">
          <div class="ls-win" id="lsWin"></div>
          <div class="ls-events" id="lsEvents"></div>
        </div>
        <div class="fig-wrap"><canvas id="sideLandscape"></canvas></div>
      </figure>

      <p>This is an awesome visualization that's fun to play around with! Some notes:</p>

      <ul class="notes">
        <li>We can see teams that did worse than they were expected to: for example, <a class="pin" data-org="LOUD" data-intl="Masters Tokyo 2023">LOUD at Tokyo</a>. This is a favorite example of mine, with a great narrative. 2023 LOUD was an amazing team with intense success before Masters Tokyo (2nd at LOCK//IN and then won Americas Stage 1) and after Masters Tokyo (3rd at Champions LA). They were a consensus top-2 favorite to win the event (Platchat put them above FNATIC, in fact, as favorites for Masters Tokyo). Their flop at Tokyo was shocking and historic - what happened? As I recall, Masters Tokyo was the start of a rift between Less/Saadhak and Aspas, a reminder that this game cannot be just broken down into numbers. Also, it’s a reminder that Valorant was, is, and will be extremely random.</li>
        <li>Speaking of random, we can also see which teams overshot their previous domestic performance! <a class="pin" data-org="T1" data-intl="Masters Bangkok 2025">T1 at Bangkok</a> is the most obvious one. Here are all of the teams coming into Masters Bangkok on this graph:
          <figure class="inline-fig">
            <div class="inline-fig-wrap"><canvas id="bangkokInset"></canvas></div>
          </figure>
          I mean seriously, how did they win this tournament?<br><br>
          <a class="pin" data-org="MIBR" data-intl="Champions 2025">MIBR at Champions Paris</a> and <a class="pin" data-org="WOL" data-intl="Masters Toronto 2025">Wolves at Masters Toronto</a> are also worth mentioning for this category.</li>
        <li>We can see that Chinese teams get consistently overrated by this visualization, due to the less competitive state of domestic CN Valorant (e.g. <a class="pin" data-org="FPX" data-intl="Masters Shanghai 2024">FPX at Shanghai</a> and <a class="pin" data-org="XLG" data-intl="Masters Santiago 2026">XLG at Santiago</a> are placed impressively on this graph - they also went 1-2 and 0-2 in their respective events)</li>
      </ul>

      <p id="sec-tiers">Lastly, this visualization was inspired by EvanMiya&rsquo;s March Madness Efficiency Landscape graph which includes tiers of favoritism to win the NCAA tournament.</p>

      <figure class="inline-fig wide">
        <img src="/evanmiya-landscape.jpg" alt="EvanMiya&rsquo;s March Madness Predicted Efficiency Landscape">
        <figcaption class="fig-credit">Credit: <a href="https://substack.com/home/post/p-191132130" target="_blank" rel="noopener">https://substack.com/home/post/p-191132130</a></figcaption>
      </figure>

      <p>In accordance with him, let&rsquo;s make our own bands of favoritism to win VCT tournaments based on this graph.</p>

      <figure class="fig">
        <div class="fig-filter">
          <div class="ls-win" id="tierWin"></div>
          <div class="ls-events" id="tierEvents"></div>
        </div>
        <div class="fig-wrap"><canvas id="tierLandscape"></canvas></div>
      </figure>

      <hr class="secbreak">

      <p id="sec-benpom">Another historical trend cited in NCAAM is the fact that every tournament winner in the 21st century has been in the Top 25 of <a class="xlink" href="https://kenpom.com/" target="_blank" rel="noopener">KenPom&rsquo;s rating system</a>.</p>

      <p>I have my own <a class="xlink" href="/mapelo/modern/" target="_blank" rel="noopener">BenPom</a> rating system for VCT - let&rsquo;s see what the trend is there:</p>

      <figure class="fig">
        <div class="fig-wrap rank-wrap"><canvas id="rankChart"></canvas></div>
        <div class="takeaway">
          <b>Every trophy winner since franchising was Top-15 by BenPom before the tournament</b>
          <span>70% were top-7</span>
        </div>
      </figure>

      <hr class="secbreak">

      <p id="sec-momentum">To go back to the point on teams that &ldquo;overshot their previous domestic performance&rdquo;, those teams often were playing better towards the end of their split. In other words, they had <em>momentum</em>. For instance, MIBR placed horribly on the Attack/Defense graph before Champions Paris, yet they finished 5th-6th and only lost to the top-3 teams at the tournament (narrowly). In their Americas Stage 2 split, they won 2 of their last 3 games but were on a 5-match losing streak before that.</p>

      <p>How important is momentum, then?</p>

      <figure class="fig">
        <p class="fig-note"><em>Note: LOCK//IN was not included, since there were no prior matches for any of those teams.</em></p>
        <div class="fig-wrap polar-wrap"><canvas id="last5Chart"></canvas></div>
      </figure>

      <p>Interesting, only one team (Gen.G) was able to win a trophy despite losing a majority of their past 5 matches. Momentum is clearly important - half of the winners had a 4-1 record or better before their events. It&rsquo;s also not the be-all-end-all, as 4 of the 10 winners lost 2 of their past 5 matches.</p>

      <p>More generally, can momentum be a strong predictor of international <em>success</em>, not just winning?</p>

      <figure class="fig">
        <div class="streak-key">
          <span><i class="sk sk-top"></i>Finished top 3</span>
          <span><i class="sk sk-rest"></i>Did not</span>
        </div>
        <div class="fig-wrap rank-wrap tip-wrap"><canvas id="successChart"></canvas></div>
      </figure>

      <p>The trend is there, but it&rsquo;s nothing more extreme than we would&rsquo;ve expected. Let&rsquo;s move on!</p>

      <hr class="secbreak">

      <p id="sec-stars">One last trope that surrounds champions in practically all sports is the notion of <em>star power</em>. It&rsquo;s an arbitrary concept, but the idea is that to be a champion, you have to have an elite player who can rise in the most important matches/moments. In VCT, our best comprehensive statistic for player quality is VLR-rating, so let&rsquo;s use that to look at each championship team&rsquo;s best player coming into the international tournament:</p>

      <figure class="fig">
        <div class="stars" id="starGrid"></div>
        <p class="fig-note below"><em>Note: For EDG at Champions Seoul, I just used ACS since VLR rating wasn&rsquo;t calculated in domestic CN splits until 2025</em></p>
      </figure>

      <p>This is also extremely interesting. 1/3 of the winning teams had the highest-rated player in the previous domestic split. Also, T1 at Bangkok are again an exception to the rule. Their highest-rated player (Buzz) was ranked 18th in Pacific Kickoff 2025. The second-lowest rank in this list was 6th for RB with Nongshim in 2026. This article is just turning into <em>&ldquo;T1 winning Bangkok is a gigantic statistical and historical anomaly&rdquo;</em>, but I digress.</p>

      <div class="takeaway">
        <b>8/9 eligible tournament winners had a top-6 rated player in the previous domestic split</b>
        <span>7/9 had a top-4 rated player</span>
      </div>

      <hr class="secbreak">

      <h2 id="roster-composition">Roster Composition</h2>

      <p>Now one of the most interesting trends I&rsquo;ve noticed in championship-winning teams is about their rosters. More specifically, their roster turnover rate. Every single championship-winning roster made roster changes from the previous year <em>OR</em> they made roster changes during the year:</p>

      <figure class="fig">
        <div class="rosters">__ROSTERS__</div>
      </figure>

      <p>There&rsquo;s a more interesting observation, though:</p>

      <div class="takeaway takeaway--roster">
        <b>Every international-winning roster since Masters Tokyo has featured a player in their rookie year.</b>
        <span>Demon1 on EG, JohnQT on Sen, Karon on Gen.G, Simon on EDG, Sylvan on T1, Patmen on PRX, Brawk/Skuba on NRG, Xross on NS, and spikziN/blowz/Neon on LEV.</span>
      </div>

      <p>This speaks to the continuous influx of top-level talent in VCT. As years go on and Valorant has been around for longer, there are new pros who grew up playing Valorant, the mechanical ceiling gets higher, and older talent generally fades out. It is proven that the best way to win a trophy in Valorant is by embracing new talent, not reshuffling older talent. Even if it means adding newer talent into a roster with veterans.</p>

      <p>This is bad news for a team that some would call the current Champions Shanghai favorites: NRG. Also PRX. We&rsquo;ve watched both of these teams get outgunned in the final stages of the two Masters events this year - by Nongshim at Masters Santiago and Leviatan at Masters London. I&rsquo;m not necessarily advocating for making roster changes on PRX and NRG, I&rsquo;m just pointing out a trend.</p>

      <p>What&rsquo;s even more damming is that every team that&rsquo;s won Champions specifically has made a mid-season roster change to add a rookie.</p>
    </div>
  </div>
</div>

<script>
const LS = __LANDSCAPE_JSON__;
const logoCache = {}, greyCache = {}, wCache = {};
const charts = [];
let redrawQueued = false;
function queueRedraw() {
  if (redrawQueued) return;
  redrawQueued = true;
  requestAnimationFrame(() => {
    redrawQueued = false;
    charts.forEach(c => { try { c.draw(); } catch (e) {} });
  });
}
// Point labels drop the "Masters" prefix: on a plot this dense the word is
// repeated forty times and carries nothing, since the city already identifies
// the event. Champions has no city in its name and is left alone. Only the
// DRAWN text changes -- the filter buttons, ring colours and tooltips all still
// key off the full name.
function shortIntl(s) { return s.replace(/^Masters /, ''); }

function logo(org) {
  if (!logoCache[org]) {
    const i = new Image();
    i.onload = queueRedraw;
    i.src = '/logos/' + org + '.png';
    logoCache[org] = i;
  }
  return logoCache[org];
}
// Greyscale baked once per logo. Doing it with ctx.filter per image per frame
// is a full filter pass 100+ times a frame, which is what made hovering crawl.
function grey(org) {
  const img = logo(org);
  if (!img.complete || !img.naturalWidth) return null;
  if (!greyCache[org]) {
    const c = document.createElement('canvas');
    c.width = img.naturalWidth; c.height = img.naturalHeight;
    const g = c.getContext('2d');
    g.filter = 'grayscale(1)';
    g.drawImage(img, 0, 0);
    greyCache[org] = c;
  }
  return greyCache[org];
}
function textW(ctx, t, font) {
  const k = font + '|' + t;
  if (wCache[k] === undefined) { ctx.font = font; wCache[k] = ctx.measureText(t).width; }
  return wCache[k];
}
const pct = v => (v * 100).toFixed(1) + '%';
const ringFor = intl => intl.indexOf('Champions') === 0 ? '#d8a93a' : '#7c4dd6';

Chart.Tooltip.positioners.aboveMark = function (items) {
  if (!items.length) return false;
  const e = items[0].element;
  return {x: e.x, y: e.y - 30};
};


// Shared HTML tooltip. Placement is fixed by the caller and only ever clamped
// vertically, so a given chart always puts its bubble in the same relation to
// the thing under the cursor. Horizontal overflow is deliberately left alone --
// the box is allowed off the white plate.
function makeTip(canvas) {
  const box = document.createElement('div');
  box.className = 'chart-tip';
  canvas.parentNode.appendChild(box);
  return {
    hide() { box.style.opacity = 0; },
    place(html, x, y, sideX, sideY) {
      box.innerHTML = html;
      box.style.opacity = 1;
      const W = box.offsetWidth, H = box.offsetHeight, g = 10;
      const h = canvas.parentNode.clientHeight;
      const L = sideX > 0 ? x + g : sideX < 0 ? x - g - W : x - W / 2;
      const T = sideY > 0 ? y + g : sideY < 0 ? y - g - H : y - H / 2;
      box.style.left = L + 'px';
      box.style.top  = Math.max(2, Math.min(T, h - H - 2)) + 'px';
    }
  };
}

const plate = {
  id: 'plate',
  beforeDraw(c) {
    const {ctx} = c;
    ctx.save(); ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, c.width, c.height); ctx.restore();
  }
};

const fifty = {
  id: 'fifty',
  beforeDatasetsDraw(c) {
    const {ctx, chartArea: a, scales: {x, y}} = c;
    ctx.save();
    ctx.strokeStyle = 'rgba(61,26,110,.38)'; ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(a.left, y.getPixelForValue(0.5)); ctx.lineTo(a.right, y.getPixelForValue(0.5));
    ctx.moveTo(x.getPixelForValue(0.5), a.top);  ctx.lineTo(x.getPixelForValue(0.5), a.bottom);
    ctx.stroke();
    ctx.restore();
  }
};

// Tiers of favouritism, EvanMiya-style. A band is a range of attack% + defense%,
// so every boundary is the same diagonal slope — a team can reach a tier by
// being strong on either side or balanced across both. Thresholds are set from
// the teams the author named: 1.168 clears FPX at Shanghai (1.160) while taking
// in G2 (1.176) and Vitality (1.177) at Bangkok; 1.08 sits just under NRG at
// Champions 2025 (1.086) and PRX at Toronto (1.084); 1.015 sits just above T1
// at Bangkok (1.008), who names the floor tier.
const BANDS = [
  {lo: 1.168, hi: 2.00,  label: 'Trophy Favorites',  fill: 'rgba(216,169,58,.20)', ink: '#8a6a1a', dx: 6,
   sub: 'A rare tier of strength to fall into. Half of the teams in this tier finished top-2 at their events'},
  {lo: 1.08,  hi: 1.168, label: 'Trophy Contenders', fill: 'rgba(124,77,214,.16)', ink: '#5b21b6',
   nudge: 0.035, dx: -18,
   sub: 'Most of these teams are strong enough to win. It’s the tier that contains the most trophy winners (and also the most entries).'},
  {lo: 1.015, hi: 1.08,  label: 'Trophy Believers',  fill: 'rgba(37,99,235,.14)',  ink: '#1d4ed8', dx: 8,
   sub: 'These teams are either middling on both attack/defense or have one side that is weak'},
  {lo: 0.00,  hi: 1.015, label: 'T1 Tier',           fill: 'rgba(220,38,38,.12)',  ink: '#b91c1c', dx: 2, dy: 6,
   sub: 'Apparently, you can win from this tier - see T1 at Bangkok'}
];

const bands = {
  id: 'bands',
  beforeDatasetsDraw(c) {
    if (!c.$bands) return;
    const {ctx, chartArea: a, scales: {x, y}} = c;
    const px = (vx, vy) => [x.getPixelForValue(vx), y.getPixelForValue(vy)];
    ctx.save();
    ctx.beginPath(); ctx.rect(a.left, a.top, a.right - a.left, a.bottom - a.top); ctx.clip();
    // Each band is the strip between two constant-sum diagonals. Drawn as a
    // parallelogram running well past the axes and clipped to the plot.
    c.$bands.forEach(b => {
      const p1 = px(0.20, b.hi - 0.20), p2 = px(0.90, b.hi - 0.90);
      const p3 = px(0.90, b.lo - 0.90), p4 = px(0.20, b.lo - 0.20);
      ctx.beginPath();
      ctx.moveTo(p1[0], p1[1]); ctx.lineTo(p2[0], p2[1]);
      ctx.lineTo(p3[0], p3[1]); ctx.lineTo(p4[0], p4[1]);
      ctx.closePath();
      ctx.fillStyle = b.fill; ctx.fill();
      if (b.lo > 0.01) {
        const l1 = px(0.20, b.lo - 0.20), l2 = px(0.90, b.lo - 0.90);
        ctx.beginPath(); ctx.moveTo(l1[0], l1[1]); ctx.lineTo(l2[0], l2[1]);
        ctx.setLineDash([6, 5]); ctx.lineWidth = 1.25;
        ctx.strokeStyle = 'rgba(61,26,110,.35)'; ctx.stroke(); ctx.setLineDash([]);
      }
    });
    ctx.restore();

    // Titles. Three constraints: inside the band, biased up and left, and clear
    // of everything already on the plot. Candidates run ALONG the band — a line
    // of constant attack+defense — so any position tried is by construction
    // still in that tier; the search just picks the first that collides with
    // nothing.
    ctx.save();
    const occupied = [];
    // What the marks will occupy, computed here because bands draw first.
    c.data.datasets[0].data.forEach(d => {
      const mx = x.getPixelForValue(d.x), my = y.getPixelForValue(d.y);
      const half = (d.p.won ? 30 : 25) / 2;
      const lw = textW(ctx, shortIntl(d.p.intl), "700 9px 'DM Sans',sans-serif");
      occupied.push({l: mx - Math.max(half, lw / 2), r: mx + Math.max(half, lw / 2),
                     t: my - half, b: my + half + 14});
    });
    const hits = r => occupied.some(q => r.l < q.r && r.r > q.l && r.t < q.b && r.b > q.t);

    c.$bands.forEach(b => {
      // Pixels per data unit, to reason about the block in the plot's own space.
      const ux = Math.abs(x.getPixelForValue(0.5) - x.getPixelForValue(0.4)) / 0.1;
      const uy = Math.abs(y.getPixelForValue(0.4) - y.getPixelForValue(0.5)) / 0.1;
      const openTop = b.hi >= 1.9, openBot = b.lo <= 0.01;

      // A band is a DIAGONAL strip, so a horizontal block of width W and height
      // H only fits inside it when the strip is thicker than W + H — its
      // top-right and bottom-left corners rest on opposite edges. Believers is
      // 0.065 thick and the full-size block is 0.083, which is why titles kept
      // ending up in a neighbouring tier. Try progressively smaller type and
      // narrower wraps until the block genuinely fits its own band.
      // Search every (scale, wrap, step) and keep the clear candidate with the
      // SMALLEST step — nearest the tier's top-left corner. Breaking at the
      // first clear hit instead meant a full-size block that had to slide a
      // long way beat a smaller one that fitted right at the corner, which is
      // how T1 Tier ended up at the bottom of its band.
      let placed = null, placedStep = 1e9, fallback = null;
      const SCALES = [1, 0.92, 0.85, 0.78, 0.72, 0.66];
      for (const sc of SCALES) {
        const tf = "800 " + (14 * sc).toFixed(1) + "px 'DM Sans',sans-serif";
        const sf = "500 " + (9.5 * sc).toFixed(1) + "px 'DM Sans',sans-serif";
        const pillH = 28 * sc, lineH = 12 * sc, pad = 22 * sc;
        const wPill = textW(ctx, b.label, tf) + pad;
        for (const maxW of [208, 168, 140, 118, 100]) {
          ctx.font = sf;
          const ls = []; let cur = '';
          (b.sub || '').split(' ').forEach(word => {
            const test = cur ? cur + ' ' + word : word;
            if (ctx.measureText(test).width > maxW * sc && cur) { ls.push(cur); cur = word; }
            else cur = test;
          });
          if (cur) ls.push(cur);
          const bw = Math.max(wPill, ls.length ? maxW * sc : 0);
          const bh = pillH + (ls.length ? ls.length * lineH + 4 : 0);
          const bwD = bw / ux, bhD = bh / uy;
          const thick = b.hi - b.lo;
          if (!openTop && !openBot && thick < bwD + bhD + 0.005) continue;

          // Walk from the band's left end rightward, a short way only.
          const xStart = Math.max(0.40, (openTop ? b.lo : b.lo) - 0.70) + (b.nudge || 0);
          for (let step = 0; step <= 10; step++) {
            const xl = xStart + step * 0.010;
            if (xl + bwD > 0.695) break;
            // Top edge sits under the upper boundary measured at the block's
            // RIGHT edge — the corner that breaches it first.
            let yt = openTop ? 0.695 : b.hi - (xl + bwD) - 0.003;
            yt = Math.min(yt, 0.695);
            const yb = yt - bhD;
            const botLimit = openBot ? 0.402 : (b.lo - xl) + 0.003;
            if (yb < botLimit) continue;
            const r = {l: x.getPixelForValue(xl), r: x.getPixelForValue(xl + bwD),
                       t: y.getPixelForValue(yt), b: y.getPixelForValue(yb)};
            if (r.t < a.top + 3 || r.b > a.bottom - 3) continue;
            const cand = {r, ls, bw, tf, sf, pillH, lineH, wPill};
            if (!fallback) fallback = cand;
            if (!hits(r) && step < placedStep) { placed = cand; placedStep = step; }
            if (placedStep === 0) break;
          }
          if (placedStep === 0) break;
        }
        if (placedStep === 0) break;
      }
      if (!placed) placed = fallback;
      if (!placed) return;

      const {r: best, ls: subLines, bw: blockW, tf, sf, pillH, lineH, wPill} = placed;
      // Hand nudges, in pixels. Each stays well inside the clearance measured
      // for that band: Contenders has ~50px of room toward its lower edge,
      // Believers ~13px toward its upper one, and T1 Tier moves DOWN, which
      // is away from its only boundary.
      const rx = best.l + (b.dx || 0), ry = best.t + (b.dy || 0);
      const px2 = rx + (blockW - wPill) / 2;
      ctx.font = tf; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillStyle = 'rgba(255,255,255,.93)';
      ctx.strokeStyle = 'rgba(61,26,110,.22)'; ctx.lineWidth = 1;
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(px2, ry, wPill, pillH, pillH / 2);
      else ctx.rect(px2, ry, wPill, pillH);
      ctx.fill(); ctx.stroke();
      ctx.fillStyle = b.ink;
      ctx.fillText(b.label, px2 + wPill / 2, ry + pillH / 2 + 0.5);

      if (subLines.length) {
        ctx.font = sf; ctx.textBaseline = 'top';
        ctx.fillStyle = 'rgba(61,26,110,.68)';
        subLines.forEach((ln, k2) => ctx.fillText(ln, rx + blockW / 2, ry + pillH + 4 + k2 * lineH));
        ctx.textBaseline = 'middle';
      }
      occupied.push(best);
    });
    ctx.restore();
  }
};

const marks = {
  id: 'marks',
  afterDatasetsDraw(c) {
    const {ctx} = c;
    const st = c.$state ? c.$state() : {hovered: null, pinned: null, winnersOn: false};
    const focusIdx = st.hovered !== null ? st.hovered : st.pinned;
    const meta = c.getDatasetMeta(0);
    const rank = i => {
      if (i === focusIdx) return 2;
      const p = c.data.datasets[0].data[i].p;
      const lit = focusIdx !== null ? false : (!st.winnersOn || p.won);
      return lit ? 1 : 0;
    };
    const order = meta.data.map((_, i) => i).sort((a, b) => rank(a) - rank(b));
    order.forEach(i => {
      const pt = meta.data[i];
      const p = c.data.datasets[0].data[i].p;
      const img = logo(p.org);
      const on  = focusIdx === i;
      const dim = focusIdx !== null ? !on : (st.winnersOn && !p.won);
      const S = (p.won ? 30 : 25) * (on ? 1.7 : 1);
      const src = dim ? grey(p.org) : img;
      ctx.save();
      ctx.globalAlpha = dim ? 0.45 : 1;
      if (src && (dim || (img.complete && img.naturalWidth))) {
        ctx.drawImage(src, pt.x - S / 2, pt.y - S / 2, S, S);
      }
      ctx.globalAlpha = 1;
      const font = (on ? "800 12px " : "700 9px ") + "'DM Sans',sans-serif";
      const lbl = shortIntl(p.intl);
      const w = textW(ctx, lbl, font);
      ctx.font = font;
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      const ly = pt.y + S / 2 + 3;
      ctx.lineWidth = on ? 3.5 : 2;
      ctx.lineJoin = 'round';
      ctx.strokeStyle = dim ? 'rgba(190,184,196,.5)' : ringFor(p.intl);
      ctx.strokeText(lbl, pt.x, ly);
      ctx.fillStyle = dim ? 'rgba(150,142,158,.55)' : '#ffffff';
      ctx.fillText(lbl, pt.x, ly);
      ctx.restore();
    });
  }
};

const AXIS = title => ({
  min: 0.40, max: 0.70,
  title: {display: true, text: title,
          font: {family: "'DM Sans',sans-serif", size: 11, weight: 600}, color: '#7a6e7e'},
  ticks: {callback: v => pct(v), stepSize: 0.05, font: {size: 10}, color: '#9a8fa4'},
  grid: {color: 'rgba(0,0,0,.05)'}
});

// One interactive landscape. Built twice — plain, then with tier bands — so its
// state lives in a closure rather than on the page, or the two would share a
// hover.
function buildLandscape({canvas, winBox, eventBox, bandDefs, winnersDefault = true}) {
  let hovered = null, pinned = null, active = 'All', winnersOn = winnersDefault, chart;
  const HOVER_PX = 20;

  const visible = () => LS.points.filter(p => active === 'All' || p.intl === active)
                                 .map(p => ({x: p.dfn, y: p.atk, p}));
  const hitRadii = data => data.map(d => (winnersOn && !d.p.won) ? 0 : 16);

  // `animate` is only true for an event-button press. Chart.js tweens element
  // positions by index, and the filtered sets barely overlap, so the logos
  // glide across the plot into the new layout while the ones that dropped out
  // shrink away -- which is the shuffle this is here for. Every other caller
  // (winners toggle, pin, first paint) passes nothing and updates instantly:
  // their data is unchanged, so an animation would just burn 550ms of frames.
  function draw(animate) {
    const data = visible();
    if (chart) {
      chart.data.datasets[0].data = data;
      chart.data.datasets[0].hitRadius = hitRadii(data);
      hovered = null; chart.update(animate ? undefined : 'none'); return;
    }
    chart = new Chart(document.getElementById(canvas), {
      type: 'scatter',
      data: {datasets: [{data, pointRadius: 0, pointHoverRadius: 0, hitRadius: hitRadii(data)}]},
      options: {
        responsive: true, maintainAspectRatio: false,
        // Hover redraws call chart.draw() straight, so they never touch this --
        // it costs nothing at rest and only runs on an event change.
        animation: {duration: 550, easing: 'easeOutQuart'},
        layout: {padding: {top: 14, right: 16, bottom: 4, left: 4}},
        interaction: {mode: 'nearest', intersect: true, axis: 'xy'},
        events: ['mousemove', 'mouseout', 'click', 'touchstart', 'touchmove'],
        onHover(e, els, c) {
          if (pinned !== null) { pinned = null; c.draw(); }
          let i = null;
          if (els.length) {
            const pt = c.getDatasetMeta(0).data[els[0].index];
            if (Math.hypot(e.x - pt.x, e.y - pt.y) <= HOVER_PX) i = els[0].index;
          }
          if (i !== hovered) { hovered = i; c.draw(); }
        },
        scales: {x: AXIS('Defense win%'), y: AXIS('Attack win%')},
        plugins: {
          legend: {display: false},
          tooltip: {
            displayColors: false, backgroundColor: 'rgba(22,18,29,.94)', padding: 10,
            position: 'aboveMark', yAlign: 'bottom', xAlign: 'center',
            animation: {duration: 140},
            animations: {numbers: {duration: 0}, opacity: {duration: 140, easing: 'linear'}},
            caretSize: 5,
            callbacks: {
              title: it => it[0].raw.p.org + ' — ' + it[0].raw.p.intl + (it[0].raw.p.won ? '  (Won)' : ''),
              label: it => {
                const p = it.raw.p;
                return ['from ' + p.prior,
                        'Attack  ' + pct(p.atk) + '  (' + p.atk_w + '/' + p.atk_n + ')',
                        'Defense ' + pct(p.dfn) + '  (' + p.def_w + '/' + p.def_n + ')'];
              }
            }
          }
        }
      },
      plugins: [plate, bands, fifty, marks]
    });
    chart.$state = () => ({hovered, pinned, winnersOn});
    if (bandDefs) chart.$bands = bandDefs;
    charts.push(chart);
    chart.canvas.addEventListener('mouseleave', () => {
      if (hovered !== null || pinned !== null) { hovered = null; pinned = null; chart.draw(); }
    });
  }

  const box = document.getElementById(eventBox);
  const evBtns = [];
  ['All'].concat(LS.internationals).forEach(name => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'lsb' + (name === 'All' ? ' on' : '');
    b.textContent = name;
    b.onclick = () => {
      active = name; pinned = null;
      if (chart) chart.tooltip.setActiveElements([], {});
      evBtns.forEach(c => c.classList.toggle('on', c === b));
      draw(true);
    };
    evBtns.push(b);
    box.appendChild(b);
  });
  const w = document.createElement('button');
  w.type = 'button';
  const setLabel = () => {
    w.innerHTML = 'Highlight winners <span class="lsb-state">' + (winnersOn ? 'ON' : 'OFF') + '</span>';
  };
  w.className = 'lsb lsb-win' + (winnersOn ? ' on' : '');
  setLabel();
  w.onclick = () => {
    winnersOn = !winnersOn; pinned = null;
    if (chart) chart.tooltip.setActiveElements([], {});
    w.classList.toggle('on', winnersOn);
    setLabel();
    draw();
  };
  document.getElementById(winBox).appendChild(w);

  // Fit the row to one line by measurement — ten event names are wider than the
  // column at any comfortable size, and a tuned constant breaks when an event
  // is added.
  function fitRow() {
    const steps = [[.68, 10], [.64, 9], [.60, 8], [.56, 7], [.52, 6], [.48, 5], [.44, 4], [.40, 4]];
    for (const [fs, px] of steps) {
      box.style.setProperty('--lsb-fs', fs + 'rem');
      box.style.setProperty('--lsb-px', px + 'px');
      if (box.scrollWidth <= box.clientWidth) return;
    }
  }
  fitRow();
  addEventListener('resize', fitRow);
  draw();

  return {
    pin(org, intl) {
      active = 'All';
      evBtns.forEach(c => c.classList.toggle('on', c.textContent === 'All'));
      if (winnersOn) { winnersOn = false; w.classList.remove('on'); setLabel(); }
      draw();
      const i = chart.data.datasets[0].data.findIndex(d => d.p.org === org && d.p.intl === intl);
      pinned = i >= 0 ? i : null;
      chart.draw();
      if (pinned !== null) {
        const el = chart.getDatasetMeta(0).data[pinned];
        chart.tooltip.setActiveElements([{datasetIndex: 0, index: pinned}], {x: el.x, y: el.y});
        chart.update();
      }
    }
  };
}

const mainLandscape = buildLandscape({canvas: 'sideLandscape', winBox: 'lsWin', eventBox: 'lsEvents'});
buildLandscape({canvas: 'tierLandscape', winBox: 'tierWin', eventBox: 'tierEvents',
                bandDefs: BANDS, winnersDefault: false});

// Static, non-interactive: the cluster the T1 sentence points at.
(function () {
  const el = document.getElementById('bangkokInset');
  if (!el) return;
  const pts = LS.points.filter(p => p.intl === 'Masters Bangkok 2025')
                       .map(p => ({x: p.dfn, y: p.atk, p}));
  const c = new Chart(el, {
    type: 'scatter',
    data: {datasets: [{data: pts, pointRadius: 0, pointHoverRadius: 0, hitRadius: 0}]},
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      layout: {padding: {top: 12, right: 14, bottom: 2, left: 2}},
      events: [],
      scales: {x: AXIS('Defense win%'), y: AXIS('Attack win%')},
      plugins: {legend: {display: false}, tooltip: {enabled: false}}
    },
    plugins: [plate, fifty, marks]
  });
  c.$state = () => ({hovered: null, pinned: null, winnersOn: false});
  charts.push(c);
})();

// Last-5 records of the winners, as a rose chart in the shape of the NCAA
// original: one wedge per possible record, radius = how many winners had it.
// Counted across events rather than inside the split -- a run into an
// international routinely spans a split, an off-season event and a previous
// international, and an event boundary does not reset momentum.
(function () {
  const el = document.getElementById('last5Chart');
  if (!el || !LS.last5 || !LS.last5.length) return;
  const rows = LS.last5;
  const top  = Math.max(...rows.map(r => r.n));
  // Darker the more common, so the shape reads before the labels do. Empty
  // buckets keep a faint ring so a missing wedge is visibly a zero.
  const ink = n => n === 0 ? 'rgba(124,77,214,.05)'
                           : 'rgba(124,77,214,' + (0.22 + 0.58 * (n / top)).toFixed(3) + ')';

  const tipBox = makeTip(el);

  // Anchored off the arc itself rather than tooltip.caretX/caretY: Chart.js
  // derives the caret from where IT would have put a canvas tooltip, sized to
  // its own box, which left this one tens of pixels off the wedge.
  function tip(ctx) {
    const t = ctx.tooltip;
    if (!t.opacity || !t.dataPoints || !t.dataPoints.length) { tipBox.hide(); return; }
    const idx = t.dataPoints[0].dataIndex;
    const r = rows[idx];
    let h = '<div class="rt-head">' + r.bucket + ' in their last 5</div>';
    h += '<div class="rt-sub">' + (r.n ? r.n + (r.n === 1 ? ' winner' : ' winners')
                                       : 'No winner came in on this record') + '</div>';
    h += r.who.map(w => '<div class="rt-row"><img src="/logos/' + w.org + '.png" alt="">'
                        + w.org + ' ' + w.label + '</div>').join('');
    const a = ctx.chart.getDatasetMeta(0).data[idx];
    const mid = (a.startAngle + a.endAngle) / 2;
    const dx = Math.cos(mid), dy = Math.sin(mid);
    // Out along the wedge's own bisector, just inside the tip. Empty wedges have
    // no tip, so they anchor on the ring the count sits on.
    const maxR = Math.max(...ctx.chart.getDatasetMeta(0).data.map(e => e.outerRadius || 0));
    const rad = r.n ? a.outerRadius * 0.92 : maxR * 0.13;
    tipBox.place(h, a.x + dx * rad, a.y + dy * rad,
                 dx > 0.3 ? 1 : dx < -0.3 ? -1 : 0,
                 dy > 0.3 ? 1 : dy < -0.3 ? -1 : 0);
  }

  // Count inside each wedge. Zeros have no wedge to sit in, so they park on a
  // fixed inner radius in a muted ink -- the label still says which bucket.
  const wedgeCounts = {
    id: 'wedgeCounts',
    afterDatasetsDraw(ch) {
      const {ctx} = ch;
      const arcs = ch.getDatasetMeta(0).data;
      const maxR = Math.max(...arcs.map(a => a.outerRadius || 0));
      arcs.forEach((a, i) => {
        const n = rows[i].n;
        const mid = (a.startAngle + a.endAngle) / 2;
        const rad = n ? a.outerRadius * 0.55 : maxR * 0.13;
        ctx.save();
        ctx.font = "800 " + (n ? 16 : 12) + "px 'DM Sans',sans-serif";
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        // Halo first: the wedge ramps from near-white to deep purple, and one
        // flat ink would vanish at one end of that ramp or the other.
        ctx.lineWidth = 3.5; ctx.lineJoin = 'round'; ctx.strokeStyle = '#fff';
        const x = a.x + Math.cos(mid) * rad, y = a.y + Math.sin(mid) * rad;
        ctx.strokeText(String(n), x, y);
        ctx.fillStyle = n ? '#3d1a6e' : '#b6acc0';
        ctx.fillText(String(n), x, y);
        ctx.restore();
      });
    }
  };

  const c = new Chart(el, {
    type: 'polarArea',
    data: {
      labels: rows.map(r => r.bucket),
      datasets: [{
        data: rows.map(r => r.n),
        backgroundColor: rows.map(r => ink(r.n)),
        borderColor: '#fff', borderWidth: 2
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      layout: {padding: {top: 6, right: 10, bottom: 6, left: 10}},
      scales: {
        r: {
          min: 0, max: top,
          ticks: {stepSize: 1, font: {size: 9.5}, color: '#9a8fa4',
                  backdropColor: 'rgba(255,255,255,.8)', z: 2},
          grid: {color: 'rgba(0,0,0,.07)'},
          angleLines: {color: 'rgba(0,0,0,.05)'},
          pointLabels: {
            display: true, centerPointLabels: true,
            font: {family: "'DM Sans',sans-serif", size: 13, weight: 700},
            color: '#5c5165'
          }
        }
      },
      plugins: {
        legend: {display: false},
        title: {
          display: true,
          text: 'Last 5-match records of international winners',
          color: '#3d1a6e', padding: {top: 2, bottom: 10},
          font: {family: "'Plus Jakarta Sans',sans-serif", size: 14, weight: 800}
        },
        tooltip: {
          enabled: false, external: tip
        }
      }
    },
    plugins: [plate, wedgeCounts]
  });
  charts.push(c);
})();

// Every team at every international, bucketed by its last-5 record and split by
// whether it finished top 3. Stacked counts rather than a bare rate, so the
// sample behind each bar stays visible -- 1-4 holds a single team, and a lone
// 0% would otherwise read as loudly as the buckets with forty.
(function () {
  const el = document.getElementById('successChart');
  if (!el || !LS.success || !LS.success.length) return;
  const rows = LS.success;
  const tallest = Math.max(...rows.map(r => r.n));

  const rateLabels = {
    id: 'rateLabels',
    afterDatasetsDraw(ch) {
      const {ctx} = ch;
      ch.getDatasetMeta(1).data.forEach((bar, i) => {
        const r = rows[i];
        if (!r.n) return;
        ctx.save();
        ctx.font = "800 17px 'DM Sans',sans-serif";
        ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
        ctx.fillStyle = '#3d1a6e';
        ctx.fillText(Math.round(100 * r.top3 / r.n) + '%', bar.x, bar.y - 17);
        ctx.font = "600 10px 'DM Sans',sans-serif";
        ctx.fillStyle = '#9a8fa4';
        ctx.fillText(r.top3 + ' of ' + r.n, bar.x, bar.y - 4);
        ctx.restore();
      });
    }
  };

  const tipBox = makeTip(el);
  const ord = p => p + (p === 1 ? 'st' : p === 2 ? 'nd' : 'rd');

  // One rule, every bar: box beside the bar, centred on the bar's own span.
  //
  // Levelling its top with the top of the stack was geometrically consistent but
  // did not look it -- on a tall bar the box sat up at the tip, on a short one it
  // dangled past the bar's bottom. Centring on the bar reads the same at every
  // height. Side comes from which half of the plot the bar is in, and only the
  // vertical clamp in makeTip can move it, and only to stay on canvas.
  function barTip(ctx) {
    const t = ctx.tooltip;
    if (!t.opacity || !t.dataPoints || !t.dataPoints.length) { tipBox.hide(); return; }
    const idx = t.dataPoints[0].dataIndex, r = rows[idx];
    let h = '<div class="rt-head">' + r.bucket + ' in their last 5</div>';
    h += '<div class="rt-sub">' + (r.n
        ? r.top3 + ' of ' + r.n + ' finished top 3 (' + Math.round(100 * r.top3 / r.n) + '%)'
        : 'No team arrived on this record') + '</div>';
    // Nothing placed means nothing to list, so name the teams instead -- those
    // are the small buckets. Capped, in case one is ever large and podium-less.
    const list = r.top3 ? r.who : r.all.slice(0, 8);
    h += list.map(w => '<div class="rt-row"><img src="/logos/' + w.org + '.png" alt="">'
                     + w.org + ' ' + w.label
                     + (w.place ? ' <span class="rt-place">(' + ord(w.place) + ')</span>' : '')
                     + '</div>').join('');
    if (!r.top3 && r.all.length > list.length) {
      h += '<div class="rt-row rt-place">+' + (r.all.length - list.length) + ' more</div>';
    }
    const bar = ctx.chart.getDatasetMeta(0).data[idx], ca = ctx.chart.chartArea;
    const topSeg = ctx.chart.getDatasetMeta(1).data[idx];
    const toRight = bar.x < (ca.left + ca.right) / 2;
    // Baseline off the scale, not off bar.base: an empty bucket has no drawn
    // bar to take a base from.
    const foot = ctx.chart.scales.y.getPixelForValue(0);
    const head = topSeg ? topSeg.y : bar.y;
    tipBox.place(h, bar.x + ((bar.width || 0) / 2) * (toRight ? 1 : -1),
                 (head + foot) / 2, toRight ? 1 : -1, 0);
  }

  const c = new Chart(el, {
    type: 'bar',
    data: {
      labels: rows.map(r => r.bucket),
      datasets: [
        {label: 'Finished top 3', data: rows.map(r => r.top3),
         backgroundColor: '#7c4dd6', maxBarThickness: 62, stack: 's'},
        {label: 'Did not', data: rows.map(r => r.n - r.top3),
         backgroundColor: '#e4dcf0', maxBarThickness: 62, stack: 's',
         borderRadius: {topLeft: 3, topRight: 3}}
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      layout: {padding: {top: 32, right: 16, bottom: 4, left: 4}},
      // intersect:true, or the bubble fires anywhere in the column -- including
      // the empty space above a short bar.
      interaction: {mode: 'index', intersect: true},
      scales: {
        x: {stacked: true, grid: {display: false},
            title: {display: true, text: 'Record over the last 5 matches',
                    font: {family: "'DM Sans',sans-serif", size: 11, weight: 600}, color: '#7a6e7e'},
            ticks: {font: {size: 11, weight: 700}, color: '#7a6e7e'}},
        y: {stacked: true, min: 0, max: Math.ceil((tallest + 6) / 10) * 10,
            title: {display: true, text: 'Teams',
                    font: {family: "'DM Sans',sans-serif", size: 11, weight: 600}, color: '#7a6e7e'},
            ticks: {stepSize: 10, font: {size: 10}, color: '#9a8fa4'},
            grid: {color: 'rgba(0,0,0,.05)'}}
      },
      plugins: {
        legend: {display: false},
        title: {
          display: true,
          text: 'Top-3 finishes by record over the last 5 matches',
          color: '#3d1a6e', padding: {top: 2, bottom: 14},
          font: {family: "'Plus Jakarta Sans',sans-serif", size: 14, weight: 800}
        },
        tooltip: {enabled: false, external: barTip}
      }
    },
    plugins: [plate, rateLabels]
  });
  charts.push(c);
})();

// Winners' BenPom rank going into the event they won. Category axis so the
// tournaments sit equidistant in chronological order regardless of the gaps
// between them, and the rank axis is reversed so #1 is at the top.
(function () {
  const el = document.getElementById('rankChart');
  if (!el || !LS.winner_ranks || !LS.winner_ranks.length) return;
  const rows = LS.winner_ranks;
  const worst = Math.max(...rows.map(r => r.rank));
  const axisMax = Math.max(25, Math.ceil(worst / 5) * 5);

  const rankMarks = {
    id: 'rankMarks',
    afterDatasetsDraw(c) {
      const {ctx} = c;
      c.getDatasetMeta(0).data.forEach((pt, i) => {
        const r = rows[i], img = logo(r.org);
        ctx.save();
        if (img.complete && img.naturalWidth) ctx.drawImage(img, pt.x - 13, pt.y - 13, 26, 26);
        ctx.font = "800 10px 'DM Sans',sans-serif";
        ctx.textAlign = 'center'; ctx.textBaseline = 'top';
        ctx.lineWidth = 2.5; ctx.lineJoin = 'round';
        ctx.strokeStyle = '#7c4dd6';
        ctx.strokeText('#' + r.rank, pt.x, pt.y + 15);
        ctx.fillStyle = '#fff';
        ctx.fillText('#' + r.rank, pt.x, pt.y + 15);
        ctx.restore();
      });
    }
  };

  const c = new Chart(el, {
    type: 'line',
    data: {
      labels: rows.map(r => r.event),
      datasets: [{
        data: rows.map(r => r.rank),
        borderColor: 'rgba(124,77,214,.45)', borderWidth: 2,
        pointRadius: 0, pointHoverRadius: 0, hitRadius: 18, tension: 0.25
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      layout: {padding: {top: 18, right: 16, bottom: 4, left: 4}},
      interaction: {mode: 'nearest', intersect: true, axis: 'x'},
      scales: {
        x: {ticks: {font: {size: 9.5}, color: '#9a8fa4', maxRotation: 40, minRotation: 40},
            grid: {color: 'rgba(0,0,0,.04)'}},
        y: {reverse: true, min: 1, max: axisMax,
            title: {display: true, text: 'BenPom rank',
                    font: {family: "'DM Sans',sans-serif", size: 11, weight: 600}, color: '#7a6e7e'},
            ticks: {stepSize: 5, font: {size: 10}, color: '#9a8fa4',
                    callback: v => '#' + v},
            grid: {color: 'rgba(0,0,0,.05)'}}
      },
      plugins: {
        legend: {display: false},
        title: {
          display: true,
          text: 'Every international winner\u2019s BenPom rank before the event',
          color: '#3d1a6e', padding: {top: 2, bottom: 14},
          font: {family: "'Plus Jakarta Sans',sans-serif", size: 14, weight: 800}
        },
        tooltip: {
          displayColors: false, backgroundColor: 'rgba(22,18,29,.94)', padding: 10,
          animation: {duration: 140},
          animations: {numbers: {duration: 0}, opacity: {duration: 140, easing: 'linear'}},
          callbacks: {
            title: it => rows[it[0].dataIndex].org + ' won ' + rows[it[0].dataIndex].event,
            label: it => {
              const r = rows[it.dataIndex];
              return ['BenPom #' + r.rank + ' of ' + r.pool + ' rated teams',
                      'as of ' + r.as_of];
            }
          }
        }
      }
    },
    plugins: [plate, rankMarks]
  });
  charts.push(c);
})();

// Every roster card to the height of the tallest, across all four seasons.
// CSS can only equalise within one grid, and each season is its own -- so a
// year whose winners all made a single change would sit shorter than the rest.
(function () {
  const cards = [].slice.call(document.querySelectorAll('.ro'));
  if (!cards.length) return;
  let queued = false;
  function level() {
    queued = false;
    cards.forEach(c => { c.style.minHeight = ''; });
    // Read every height before writing any, or each write forces a reflow.
    const tallest = Math.max.apply(null, cards.map(c => c.offsetHeight));
    cards.forEach(c => { c.style.minHeight = tallest + 'px'; });
  }
  function schedule() {
    if (queued) return;
    queued = true;
    requestAnimationFrame(level);
  }
  schedule();
  addEventListener('resize', schedule);
  // Web fonts land after first paint and change every measurement.
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(schedule);
})();

// Star cards. Plain DOM, not Chart.js -- these are portraits with numbers
// attached, and nothing here is plotted against an axis.
(function () {
  const grid = document.getElementById('starGrid');
  if (!grid || !LS.stars || !LS.stars.length) return;
  const esc = t => String(t).replace(/[&<>"]/g, c =>
    ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
  // A split name reads better without the season prefix repeated nine times.
  // Backslashes are doubled through this file: PAGE_HTML is a plain (non-raw)
  // triple-quote, so a single one in a JS regex is an invalid Python escape --
  // a warning today, an error in a later Python.
  const split = t => String(t).replace(/^(Champions Tour|VCT)\\s+/, '').replace(/^(\\d{4}):\\s*/, '$1 ');

  const card = s => {
    const org = '<span class="star-org"><img src="/logos/' + esc(s.org) + '.png" alt=""></span>';

    // Events with nothing to measure still get a card. A visible gap says more
    // than a quietly shorter row, and the reason is the interesting part.
    const open = s.url ? '<a class="star star--empty" href="' + esc(s.url)
                       + '" target="_blank" rel="noopener">' : '<div class="star star--empty">';
    const shut = s.url ? '</a>' : '</div>';
    if (s.kind === 'nodata' || s.kind === 'nostage') {
      return open
        + '<div class="star-face star-face--org"><img src="/logos/' + esc(s.org) + '.png" alt=""></div>'
        + '<div class="star-body">'
        +   '<div class="star-name">' + esc(s.intl) + '</div>'
        +   '<div class="star-evt">' + esc(s.org) + '</div>'
        +   '<div class="star-none">' + esc(s.note) + '</div>'
        + '</div>' + shut;
    }

    const face = s.head
      ? '<img src="' + esc(s.head) + '" alt="" loading="lazy">'
      : '<span>' + esc(s.player.slice(0, 1).toUpperCase()) + '</span>';
    // The ACS card is labelled, so 258 next to everyone else's 1.14 cannot be
    // read as a rating.
    const stat = s.kind === 'acs' ? ' <span class="star-stat">ACS</span>' : '';
    return (s.url ? '<a class="star" href="' + esc(s.url) + '" target="_blank" rel="noopener">'
                  : '<div class="star">')
      + '<div class="star-face">' + face + org + '</div>'
      + '<div class="star-body">'
      +   '<div class="star-name">' + esc(s.player) + '</div>'
      +   '<div class="star-evt">' + esc(s.org) + ' &middot; ' + esc(s.intl) + '</div>'
      +   '<div class="star-line"><span class="star-rating">' + s.val + '</span>' + stat
      +     '<span class="star-rank"><b>#' + s.rank + '</b> of ' + s.pool
      +       (s.tied ? ' <span class="star-tie">tied</span>' : '') + '</span></div>'
      +   '<div class="star-split">in ' + esc(split(s.prior)) + ' &middot; ' + s.rounds + ' rounds</div>'
      +   (s.note ? '<div class="star-note">' + esc(s.note) + '</div>' : '')
      + '</div>' + (s.url ? '</a>' : '</div>');
  };

  // Season comes off the event label's trailing year; the data is already in
  // chronological order, so first-seen order is chronological too.
  const years = [];
  LS.stars.forEach(s => {
    const y = (String(s.intl).match(/(\\d{4})\\s*$/) || [, '?'])[1];
    let g = years.find(v => v.year === y);
    if (!g) years.push(g = {year: y, rows: []});
    g.rows.push(s);
  });

  grid.innerHTML = years.map(g =>
    '<div><div class="star-year"><span>' + esc(g.year) + '</span><i></i></div>'
    + '<div class="star-row">' + g.rows.map(card).join('') + '</div></div>'
  ).join('');
})();

// In-text mentions drive the first chart.
document.querySelectorAll('.pin').forEach(a => {
  a.addEventListener('click', ev => {
    ev.preventDefault();
    mainLandscape.pin(a.dataset.org, a.dataset.intl);
    document.querySelector('.fig').scrollIntoView({behavior: 'smooth', block: 'center'});
  });
});
</script>
</body>
</html>
"""


@article_championship_dna_bp.route("/")
def article_championship_dna():
    # Straight substitution rather than a Jinja variable: the chart script is
    # full of JS object literals, and render_template_string would try to read
    # some of them as Jinja delimiters.
    return (PAGE_HTML
            .replace("__LANDSCAPE_JSON__", json.dumps(_landscape()))
            .replace("__ROSTERS__", _rosters_html()))
