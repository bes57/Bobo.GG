"""
The best, the worst, and the most surprising moves of the
offseason — opinion article.

Structure mirrors Top10Shanghai.py: h1 / byline / cover / content with the
fixed .toc rail + scroll-spy. The prose is the author's, reproduced verbatim;
this file only adds the two data visualisations the draft asks for:

  * most kills in a single VCT match, top 20 all time (dot plot)
  * Krostaly's 2026 event ratings, in order (line)

Both are computed from the CSVs on disk at request time (cached).
"""

import os, json
import pandas as pd
from flask import Blueprint

article_offseason_moves_bp = Blueprint("article_offseason_moves", __name__)

ROOT     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")


def _cap(name):
    name = str(name)
    return name[:1].upper() + name[1:]


_kills_cache = None

def _kills_points():
    """Kill totals for every player performance in a Bo5, franchised VCT only
    — the same event rule as the All-Time Highs page (CN-only and ratings_only
    events hidden, 2023 LCQ kept). A Bo5 is any series whose winner took 3
    maps (the SeriesFormat column has a few mislabels, the score doesn't)."""
    global _kills_cache
    if _kills_cache is not None:
        return _kills_cache
    from MoreTestingMaybeFiles import ALL_EVENTS
    pts = []
    try:
        mr = pd.read_csv(os.path.join(DATA_DIR, "match_results.csv"), dtype=str)
        mr = mr[mr["MapNum"] == "all"].set_index("MatchID")
        mname, mscore = mr["MatchName"].to_dict(), mr["Score"].to_dict()
        frames = []
        for e in ALL_EVENTS:
            if list(e["regions"].keys()) == ["CN"]:
                continue
            if e.get("ratings_only") and e["id"] != "2023_lcq":
                continue
            path = os.path.join(DATA_DIR, "series", f"{e['id']}.csv")
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path, dtype=str)
            df["_ev"] = e["label"]
            frames.append(df[["Player", "Org", "MatchID", "K", "_ev"]])
        s = pd.concat(frames, ignore_index=True)
        s["K"] = pd.to_numeric(s["K"], errors="coerce")
        s = s.dropna(subset=["K"])
        s = s[s["MatchID"].map(lambda m: str(mscore.get(m, "")).startswith("3-"))]
        for _, r in s.sort_values("K", ascending=False).iterrows():
            stage = str(mname.get(r["MatchID"], "")).split(": ")[-1]
            pts.append({
                "p": _cap(r["Player"]), "org": str(r["Org"]), "k": int(r["K"]),
                "ev": r["_ev"], "m": stage,
                "hl": False,
            })
        if pts:
            pts[0]["hl"] = True   # only the record itself is highlighted
    except Exception:
        pts = []
    _kills_cache = pts
    return pts


_soc_cache = None

def _krostaly_points():
    """Krostaly's event rating at each franchised 2026 event, oldest first."""
    global _soc_cache
    if _soc_cache is not None:
        return _soc_cache
    from MoreTestingMaybeFiles import ALL_EVENTS
    pts = []
    for e in sorted((e for e in ALL_EVENTS if e.get("year") == 2026 and e.get("start")),
                    key=lambda e: e["start"]):
        if e.get("ratings_only") or list(e["regions"].keys()) == ["CN"]:
            continue
        path = os.path.join(DATA_DIR, f"{e['id']}.csv")
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path, dtype=str)
        except Exception:
            continue
        if "R2.0" not in df.columns:
            continue
        r = df[df["Player"].str.lower() == "krostaly"]
        if r.empty:
            continue
        r = r.iloc[0]
        try:
            rating = float(r["R2.0"])
        except (TypeError, ValueError):
            continue
        pts.append({"ev": e["label"].replace("2026 ", "", 1), "org": str(r["Org"]),
                    "r": rating, "rnd": str(r.get("Rnd", ""))})
    _soc_cache = pts
    return pts


# The T1 reactions, in the author's order, drawn as a chain of thoughts.
_THOUGHTS = [
    "TenZ is coming back?",
    "TenZ is going to APAC?",
    "Crashies is playing with TenZ?",
    "Crashies is going to APAC?",
    "T1 is keeping Buzz, Stax, and Izu?",
    "Buzz, Stax, and Izu are comming in English?",
    "Sacy is coming back?",
    "Sacy is going to APAC?",
    "Sacy is a head coach?",
    "Wait, what role is TenZ even playing?",
]

_ARROW = ('<svg viewBox="0 0 40 24" aria-hidden="true"><path d="M3 12h30M24 4l10 8-10 8" fill="none" '
          'stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/></svg>')


def _thoughts_html():
    """Two-across snake: 1 -> 2, down, 4 <- 3, down, 5 -> 6 ... so the arrows
    lead the eye from each thought to the next in reading order."""
    rows = [_THOUGHTS[k:k + 2] for k in range(0, len(_THOUGHTS), 2)]
    out = ['      <div class="thoughts">']
    for r, pair in enumerate(rows):
        first = f'<div class="th">{pair[0]}</div>'
        second = f'<div class="th">{pair[1]}</div>' if len(pair) > 1 else '<div></div>'
        ltr = r % 2 == 0
        arrow = (f'<div class="th-arrow{"" if ltr else " th-left"}">{_ARROW}</div>'
                 if len(pair) > 1 else '<div></div>')
        out.append("        " + (first + arrow + second if ltr else second + arrow + first))
        if r < len(rows) - 1:
            down = f'<div class="th-arrow th-down">{_ARROW}</div>'
            # the turn happens under whichever side this row ended on
            out.append("        " + ('<div></div><div></div>' + down if ltr
                                     else down + '<div></div><div></div>'))
    out.append('      </div>')
    return "\n".join(out)


PAGE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=820">
<title>The best, the worst, and the most surprising moves of the offseason &mdash; Bobo's VCT Database</title>
<meta property="og:type" content="article">
<meta property="og:site_name" content="Bobo gg">
<meta property="og:title" content="The best, the worst, and the most surprising moves of the offseason">
<meta property="og:description" content="This signing of Erde and Lukxo may be the greatest 1-2 punch signing in the history of VCT">
<meta property="og:url" content="https://bobo-gg.net/articles/offseason-moves/">
<meta property="og:image" content="https://bobo-gg.net/static/offseason/cover.jpg">
<meta property="og:image:secure_url" content="https://bobo-gg.net/static/offseason/cover.jpg">
<meta property="og:image:type" content="image/jpeg">
<meta property="og:image:width" content="2200">
<meta property="og:image:height" content="1466">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="The best, the worst, and the most surprising moves of the offseason">
<meta name="twitter:description" content="This signing of Erde and Lukxo may be the greatest 1-2 punch signing in the history of VCT">
<meta name="twitter:image" content="https://bobo-gg.net/static/offseason/cover.jpg">
<link rel="icon" type="image/svg+xml" href="/favicon.svg">
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@700;800&family=DM+Sans:wght@300;400;500;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="/static/base.css">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<style>
  html { -webkit-text-size-adjust:100%; text-size-adjust:100%; }
  .page { position:relative; z-index:1; flex:1; display:flex; flex-direction:column; align-items:center; padding:60px 32px 80px; }
  .article { max-width:860px; width:100%; }
  h1 { font-family:'Plus Jakarta Sans',sans-serif; font-size:clamp(1.55rem,3.1vw,2.6rem); font-weight:800; letter-spacing:-1px; line-height:1.14; margin-bottom:24px; text-align:center; }
  .byline { font-size:.82rem; color:var(--soft); font-weight:300; margin-bottom:48px; padding-bottom:32px; border-bottom:1px solid #e8e0ec; text-align:center; }
  .cover { width:100%; border-radius:16px; overflow:hidden; margin-bottom:48px; }
  .cover img { width:100%; height:auto; display:block; }
  .content p { font-size:1rem; font-weight:300; line-height:1.8; color:var(--ink); margin-bottom:24px; }
  .content h2 { font-family:'Plus Jakarta Sans',sans-serif; font-size:1.9rem; font-weight:800; letter-spacing:-0.6px; color:#7c4dd6; margin:0 0 26px; }
  .content h3 { font-family:'Plus Jakarta Sans',sans-serif; font-size:1.5rem; font-weight:800; letter-spacing:-0.4px; line-height:1.2; margin:44px 0 20px; }
  .content h3 { display:flex; align-items:center; gap:12px; }
  .content h3 .h-logo { width:38px; height:38px; object-fit:contain; flex:0 0 auto; }
  .content h2 + h3 { margin-top:0; }
  .content h2, .content h3, .cover { scroll-margin-top:84px; }
  .content .secbreak { border:0; border-top:1px solid var(--ink); margin:52px 0 40px; }
  .content em { font-style:italic; }
  .content ul.notes { margin:0 0 24px; padding-left:22px; }
  .content ul.notes li { font-size:1rem; font-weight:300; line-height:1.8; color:var(--ink); margin-bottom:12px; }
  .content ul.notes li p { margin:0; }
  .photo { width:100%; border-radius:16px; overflow:hidden; border:1px solid #ece6f2;
           box-shadow:0 4px 24px #0000000a; background:#f3eefb; margin:0 0 30px; }
  .photo img { width:100%; height:auto; display:block; }
  /* Portrait graphics: narrower and centred so they don't run a full screen tall */
  .photo.portrait { max-width:480px; margin-left:auto; margin-right:auto; }
  .chart-wrap { background:white; border-radius:20px; padding:24px 26px 18px;
                box-shadow:0 4px 24px #0000000a; margin:28px 0 34px; }
  .chart-title { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.02rem;
                 letter-spacing:-0.2px; color:var(--ink); text-align:center; margin-bottom:10px; }
  .kills-box { position:relative; height:400px; }
  .soc-box { position:relative; height:280px; }
  /* Chain of reactions: thought bubbles joined by arrows, snaking two across */
  .thoughts { display:grid; grid-template-columns:1fr 56px 1fr; align-items:center;
              row-gap:6px; margin:30px 0 36px; }
  .th { background:#fff; border:1.5px solid #e0d4ec; border-radius:18px; padding:16px 20px;
        text-align:center; font-family:'Plus Jakarta Sans',sans-serif; font-weight:700;
        font-size:1rem; line-height:1.35; color:var(--ink); box-shadow:0 4px 18px #0000000a; }
  .th-arrow { display:flex; align-items:center; justify-content:center; color:#7c4dd6; }
  .th-arrow svg { width:34px; height:20px; display:block; }
  .th-arrow.th-left svg { transform:rotate(180deg); }
  .th-arrow.th-down { height:34px; }
  .th-arrow.th-down svg { transform:rotate(90deg); }
  /* Sections rail, matching the other articles */
  .toc { position:fixed; top:32px; right:32px; background:#fff; border-radius:16px;
         padding:18px 20px; box-shadow:0 4px 24px #0000000f; display:flex;
         flex-direction:column; gap:6px; z-index:100; width:max-content;
         max-width:min(215px, calc(50vw - 480px)); }
  .toc-title { font-family:'Plus Jakarta Sans',sans-serif; font-size:.72rem; font-weight:800;
               letter-spacing:.1em; text-transform:uppercase; color:var(--soft); margin-bottom:4px; }
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
  <a href="#best">The best</a>
  <a href="#worst">The worst</a>
  <a href="#surprising">The most surprising</a>
</nav>
<div class="page">
  <div class="article">
    <h1>The best, the worst, and the most surprising<br>moves of the offseason</h1>
    <div class="byline">Bobo &mdash; November 22, 2026</div>
    <div class="content">

      <h2 id="best">The best:</h2>

      <h3><img class="h-logo" src="/static/logos/LEV.png" alt="Leviatán">LEV sign Lukxo and Erde</h3>

      <div class="photo"><img src="/static/offseason/cover.jpg" alt="Lukxo and Erde" width="2200" height="1466"></div>

      <p>When it broke that Leviatán would be losing both Sato and kiNgg to G2 - the first blockbuster move of the offseason - it seemed like the story of this Leviatán core would be a quick rise and fall. They failed to qualify for the international events before and after Masters London, and now they had lost their star duelist and franchise IGL. In short, Leviatán’s brilliance seemed to be localized just within that short timeframe of Masters London, as the gap left by Sato and kiNgg would be impossible to fill.</p>

      <p>But Leviatán did just that. This signing of Erde and Lukxo may be the greatest 1-2 punch signing in the history of VCT - it is the closest thing we have ever seen to a team signing the two best players in the world (whether those two are actually the top 2 players in the world is up for debate).</p>

      <ul class="notes">
        <li>Lukxo and Erde were the <em>two-highest rated players at Champions Shanghai</em></li>
        <li>They are both (obviously) reigning Champions-winners</li>
        <li>Erde was both Champions MVP and LOUD’s IGL</li>
      </ul>

      <p>Put simply, the gap between Erde + Lukxo and then the next 3 players on LOUD is greater than any other international-winning roster in VCT history - they are the reason LOUD won Champions Shanghai. To get the two hard-carries who won the most recent international event is ridiculous.</p>

      <p>Individual signings, such as TenZ to Sentinels in 2021, Yay to Cloud9 in 2023, and Demon1 to NRG in 2024 were groundbreaking moves that shook up VCT, where a team signed a player that was considered the best in the world. This move by Leviatán is the same, except doubled. Ridiculous.</p>

      <h3><img class="h-logo" src="/static/logos/FNC.png" alt="FNATIC">FNATIC sign Valyn, Marteen, and Minny</h3>

      <div class="photo"><img src="/static/offseason/marteen-minny.jpg" alt="Marteen and Minny" loading="lazy" width="2200" height="1833"></div>

      <p>In offseason discussions, FNATIC’s moves have flown a bit under the radar; maybe because they happened so early on or maybe because people forget how good these players are - specifically Marteen.</p>

      <p>Put simply, if Lukxo and Erde aren’t the best players in the world, then Marteen is. Marteen went through ups and downs with a confusing GentleM8’s team in 2026, where he did his best.</p>

      <p>After coming out to a roaring start in 2026, M8’s came 2nd in EMEA Kickoff with Marteen:</p>

      <ul class="notes">
        <li>Having the second-highest rating of the event (1.27) and third-highest across all kickoffs in VCT</li>
        <li>Breaking the kill record by <em>15 KILLS</em>, notching 126 kills in the Middle Final against FNATIC to qualify for Masters Santiago. If it’s not clear how insane that number is, the graph below can serve as reference
          <div class="chart-wrap">
            <div class="chart-title">Kills in a VCT match all time</div>
            <div class="kills-box"><canvas id="killsDots"></canvas></div>
          </div>
          <p>The gap between him and ZmjjKK in 2nd is ridiculous - 15 kills as mentioned before. The gap between him and 3rd is 20. Who’s 3rd? Marteen, from the same event, with 106 kills in the Upper Final against BBL. What’s more, that is now the record for most kills in a lost match. He now holds <em>both</em> records.</p>
        </li>
        <li>Qualifying to Masters Santiago, where he broke the record for highest VLR-rating at a VCT Event in history, dropping a 1.41 rating <em>on average</em> while carrying his team into playoffs.</li>
      </ul>

      <p>After Santiago, M8’s struggled in VCT. Despite swapping around roles (the most confusing one being putting Minny on duelist), dropping Bipo, and changing strategies, they never made another VCT International on the year. However, they did reach EWC, where Marteen was - you guessed it - the highest-rated player and carried M8’s to playoffs again.</p>

      <p>Marteen’s 2026 may be the greatest prime in VCT history, and we only saw him at one true international.</p>

      <p>The point is: FNATIC did a damn good job in picking up Marteen. How this is not more hyped is beyond me.</p>

      <p>Furthermore, who did they supplement him with? They brought in a high-fragging player that he has known chemistry with in Minny, and one of the greatest IGL’s in the game in Valyn.</p>

      <p>In who they dropped, losing Alfajer is rough, but dropping Boaster was long overdue. He wasn’t able to pull off the same IGL-roster magic he had in the past, as FNATIC qualified to 0 international events in 2026 - including EWC. His poor shooting and statistics had no redeeming argument to make for them this year.</p>

      <p>In this FNATIC- M8’s roster shuffle, FNATIC objectively came out on top. In Kajaak they have the best player of 2025 and in Marteen they have the best player of 2026. With a high-fragging, cohesive core around them, they’re poised to be one of the best teams of 2027.</p>

      <h3><img class="h-logo" src="/static/logos/PRX.png" alt="Paper Rex">PRX sign Primmie and Patmen</h3>

      <div class="photo"><img src="/static/offseason/primmie.jpg" alt="Primmie" loading="lazy" width="2200" height="1467"></div>

      <p>Complacency, sloth, or whatever you want to call it, has been the downfall of many a team throughout VCT history. Failing to make roster changes and update your team with new energy and ideas is a historical fallacy. One of the greatest examples of this is PRX themselves, where after years of coming so close to success with their roster of Mindfreak, Jinggg, d4v41, and F0rsaken, they made the controversial decision to drop Mindfreak and bring in Patmen - leading to a revitalization of PRX and their first trophy. G2, who have also come so close to a trophy, have only trended downwards as time’s gone on, as they finally blew up their core this offseason.</p>

      <p>Here, I am proud of PRX for, despite being the best overall team for the whole of 2026 (2 international grandfinals, a Stage 1 trophy, attending all 3 internationals), being proactive and critical in their offseason.</p>

      <p>Something and Invy specifically, I think, have been incredible this season. Something continues to establish himself as the best mechanical player on this juggernaut of a team, and Invy has risen at every single international they’ve attended.</p>

      <p>Clearly, though, the underlying traits of PRX being erratic and unpredictable are still causing issues. Choking both grandfinals once again and looking slightly dysfunctional at the most recent Champions Shanghai, it’s clear that PRX have identified Jinggg as a core contributor of this identity. As such, he was kicked from the team. Then for D4v4i, whether it be because of his existence as a weak link in the team (this is debatable) or having to make space for Primmie, has also been removed.</p>

      <p>Swallowing their pride and resigning Patmen after dropping him the year before is admirable, and signing the best APAC player in Primmie is extraordinary. These moves are of course controversial, but I believe PRX have done an amazing job in not being complacent, removing the right players, and making great signings.</p>

      <p>Primmie and Something is a ridiculous mechanical duo and Patmen will slot easily into PRX with his past chemistry and success with the core.</p>

      <p>The roster upgrades may not be as great as what teams like Sentinels are experiencing, but the moves themselves are clearly calculated and impressive.</p>

      <hr class="secbreak">

      <h2 id="worst">The worst:</h2>

      <h3><img class="h-logo" src="/static/logos/FUT.png" alt="FUT Esports">FUT sign Aspas, Wo0t, and Hezacoil</h3>

      <div class="photo"><img src="/static/offseason/krostaly.jpg" alt="Krostaly" loading="lazy" width="2200" height="1467"></div>

      <p>FUT is a team that was proactive this offseason, notably after a surprisingly good year where they overshot expectations by a <em>large</em> margin. So why are they in the “worst moves” category, while PRX are in the “best moves” category? While PRX come across as calculated, with a clear direction in their moves, FUT come across as the opposite.</p>

      <p>It seems that FUT initially decided they wanted to overhaul their team (whether that be because of players leaving for higher offers or scouted players they preferred) and were continuing to be a Turkish-speaking team. The way in which this happened was confusing.</p>

      <p>To start, sociablEE and Krostaly weren’t their best players, but did they deserve to get dropped? Krostaly especially. Keep in mind, he came in as a rookie IGL to a team with little expectations and IGL’d them to two internationals, a 5th-6th finish at Masters London, and was performing better and better as the year went on.</p>

      <div class="chart-wrap">
        <div class="chart-title">Krostaly’s event ratings chronologically over the year</div>
        <div class="soc-box"><canvas id="socLine"></canvas></div>
      </div>

      <p>In any case, FUT dropped them and picked up two T2 players from Fireflux - hezacoil and Batujanx - to replace them. Picking up rookies is a historically-proven method to do well in VCT, but these two aren’t even the best on their T2 team (re: Kachow). I’m slightly pessimistic here and a bit confused - keep in mind Batujanx was last seen in T1 with APEKS, which says enough for anyone who remembers that team. Hezacoil seems good, though. I still don’t love this swap, especially with Krostaly for Batujanx.</p>

      <p>Then Xeus got dropped for Wo0t which is questionable in and of itself. Wo0t is a high ceiling player - he probably has a higher ceiling than Xeus - so it can be justified (even if Xeus has done quite well for FUT this year). I still think Wo0t is a bit overhyped, so I find this to be a sidegrade. It’s a big-name signing nonetheless. Moving on.</p>

      <p>Then, FUT notably lose their best player in S0pp while bringing in Aspas. This just raises so many questions.</p>

      <p>Quite frankly, this reads as FUT going all-in on the star power and allure of Aspas and throwing all team dynamics and logic to the wind. Obviously, the language barrier here is a massive red flag. I don’t care if Wo0t and Aspas go on stream and say that they’re communicating fine (they did), getting 5 players to speak English when:</p>

      <ul class="notes">
        <li>none of them speak it natively</li>
        <li>some of them have questionable levels of English (Aspas, Yetujey, and maybe Hezacoil?)</li>
        <li>and FUT were very clearly planning on being a Turkish-speaking team before realizing they could sign Aspas</li>
      </ul>

      <p>is a recipe for disaster. If we went over teams and players that did worse than expected (at least partially) because of language barriers, we would be here all day. A few that come to mind - Comeback on TH in 2026, Less on Vitality in 2025, Pancada on SEN in 2023, etc.</p>

      <p>Now, this isn’t an absolute rule - none of PRX’s players have ever spoken English as their first language - but it’s very clear that this will likely present some issues and FUT didn’t plan for this until later on in the offseason.</p>

      <p>Also, why lose your best player from a very successful year in S0pp? Yetujey is the one you keep?</p>

      <p>If you have the money to get Aspas, why not get better Turkish players? BBL players were being shopped around in free agency, and FUT didn’t sign them despite them performing well in T1 and outperforming those Fireflux players when they were all in T2.</p>

      <p>The roster moves from FUT this offseason are confusing at best, and stupid at worst. I don’t think FUT will necessarily be bad, but I don’t think they’ll be good or have better results than they had in 2026 - all while paying a much higher roster salary in 2027. Lumping together big names (who are both past their prime) and confusing T2 signings without really considering proven chemistry or language barriers is something I do not support.</p>

      <h3><img class="h-logo" src="/static/logos/NS.png" alt="Nongshim RedForce">NS sign UdoTan</h3>

      <div class="photo"><img src="/static/offseason/udotan.jpg" alt="UdoTan" loading="lazy" width="2200" height="1466"></div>

      <p>This one is simple, as it’s a straight downgrade. Nonghsim obviously had an amazing year in 2026, and Dambi was a key part of that. Though his performances tapered off a bit into the end of the year + Neon nerfs, he was still a highly-regarded player. However, sometimes players leave on their own accord or teams can’t match competing offers, so it’s unfair to just blame Nongshim for losing him.</p>

      <p>However, they’re replacing him with… UdoTan? One of the worse players from GE - a team that was worse than NS? UdoTan was not better than Patmen or Xavi8k this year. At Champions Shanghai, he dropped a 0.9 rating.</p>

      <p>I’m underwhelmed by this signing. I would much rather NS have signed a T2 Korean duelist or poached a KRX duelist.</p>

      <hr class="secbreak">

      <h2 id="surprising">The most surprising:</h2>

      <h3><img class="h-logo" src="/static/logos/T1.png" alt="T1">T1 sign Tenz and Crashies (+ Sacy as HC)</h3>

      <div class="photo portrait"><img src="/static/offseason/tenz.jpg" alt="TenZ" loading="lazy" width="1080" height="1350"></div>

      <p>Could there have been any other roster move in this category? This move raises a bajillion questions, but most reactions went something like this:</p>

__THOUGHTS__

      <p>These moves are just so utterly mind-boggling, it’s hard to know where to start - and where to stop.</p>

      <p>As for the roster’s actual success, I’ll keep it short and leave it at this:</p>

      <p>There are 1274819572129 reasons this team could fail:</p>

      <ul class="notes">
        <li>Language barriers</li>
        <li>Player forms</li>
        <li>Role issues</li>
        <li>Pressure/expectations</li>
        <li>First-time head coach</li>
        <li>Extremely competitive APAC region</li>
        <li>So on</li>
      </ul>

      <p>But at the end of the day, TenZ is playing again, and T1 will be more than happy to cash that check he brings in.</p>

    </div>
  </div>
</div>
<script>
var AXIS_FONT = { family: "'Plus Jakarta Sans',sans-serif", weight: '800', size: 11 };

// ── Most kills in a match: one dot per performance, ties stacked ────────────
(function() {
  var el = document.getElementById('killsDots');
  if (!el || typeof Chart === 'undefined') return;
  var RAW = __KILLS_JSON__;
  if (!RAW.length) return;
  // Histogram-style dot plot: one dot per performance, stacked up from the
  // axis at its kill count, so the shape of the whole field is visible.
  var groups = {};
  RAW.forEach(function(p) { (groups[p.k] = groups[p.k] || []).push(p); });
  var maxStack = 1;
  Object.keys(groups).forEach(function(k) { if (groups[k].length > maxStack) maxStack = groups[k].length; });
  var field = [], hl = [];
  Object.keys(groups).forEach(function(k) {
    var arr = groups[k];
    for (var i = 0; i < arr.length; i++) {
      var pt = { x: arr[i].k, y: i + 0.5, raw: arr[i] };
      (arr[i].hl ? hl : field).push(pt);
    }
  });
  var ks = RAW.map(function(p) { return p.k; });
  var lo = Math.floor((Math.min.apply(null, ks) - 1) / 10) * 10;
  var hi = Math.ceil((Math.max.apply(null, ks) + 2) / 10) * 10;
  // Name + total above each highlighted dot
  var hlLabels = { id: 'hlLabels', afterDatasetsDraw: function(chart) {
    var ctx = chart.ctx, meta = chart.getDatasetMeta(1);
    ctx.save();
    ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    meta.data.forEach(function(pt, i) {
      var r = hl[i].raw;
      ctx.font = "800 13px 'Plus Jakarta Sans',sans-serif"; ctx.fillStyle = '#7c4dd6';
      ctx.fillText(String(r.k), pt.x, pt.y - 26);
      ctx.font = "700 11px 'Plus Jakarta Sans',sans-serif"; ctx.fillStyle = '#1a1523';
      ctx.fillText(r.p, pt.x, pt.y - 11);
    });
    ctx.restore();
  } };
  function draw() {
    new Chart(el.getContext('2d'), {
      type: 'scatter',
      data: { datasets: [
        { data: field, backgroundColor: 'rgba(149,118,184,0.62)', borderColor: '#fff',
          borderWidth: 0.5, pointRadius: 3.4, pointHoverRadius: 6, pointHitRadius: 4 },
        { data: hl, backgroundColor: '#7c4dd6', borderColor: '#fff',
          borderWidth: 2, pointRadius: 8, pointHoverRadius: 10 }
      ] },
      plugins: [hlLabels],
      options: {
        responsive: true, maintainAspectRatio: false,
        interaction: { mode: 'nearest', intersect: true },
        layout: { padding: { top: 8, left: 4, right: 14 } },
        scales: {
          x: { min: lo, max: hi,
               title: { display: true, text: 'Number of kills', color: '#9b8fae', font: AXIS_FONT },
               grid: { color: '#f1ecf6' }, ticks: { color: '#9b8fae', font: { size: 11 }, stepSize: 10 } },
          y: { min: 0, max: maxStack + 2, display: false, grid: { display: false } }
        },
        plugins: {
          legend: { display: false },
          tooltip: { displayColors: false, padding: 10,
            titleFont: { family: "'Plus Jakarta Sans',sans-serif", weight: '800', size: 13 },
            callbacks: {
              title: function(items) { var r = items[0].raw.raw; return r.p + ' (' + r.org + ')'; },
              label: function(c) { var r = c.raw.raw;
                return [r.k + ' kills', r.ev + (r.m ? ' \u2014 ' + r.m : '')]; } } }
        }
      }
    });
  }
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(draw); else draw();
})();

// ── Krostaly's 2026 event ratings, in order ────────────────────────────────
(function() {
  var el = document.getElementById('socLine');
  if (!el || typeof Chart === 'undefined') return;
  var RAW = __SOC_JSON__;
  if (!RAW.length) return;
  var vals = RAW.map(function(p) { return p.r; });
  var lo = Math.floor((Math.min.apply(null, vals) - 0.05) * 10) / 10;
  var hi = Math.ceil((Math.max.apply(null, vals) + 0.05) * 10) / 10;
  // Draw each rating above its point
  var valueLabels = { id: 'valueLabels', afterDatasetsDraw: function(chart) {
    var ctx = chart.ctx, meta = chart.getDatasetMeta(0);
    ctx.save();
    ctx.font = "800 13px 'Plus Jakarta Sans',sans-serif";
    ctx.fillStyle = '#1a1523'; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    meta.data.forEach(function(pt, i) { ctx.fillText(RAW[i].r.toFixed(2), pt.x, pt.y - 11); });
    ctx.restore();
  } };
  function draw() {
    new Chart(el.getContext('2d'), {
      type: 'line',
      data: { labels: RAW.map(function(p) { return [p.ev, p.org]; }),
              datasets: [{ data: vals, borderColor: '#7c4dd6', borderWidth: 2.5, tension: 0,
                           pointBackgroundColor: '#7c4dd6', pointBorderColor: '#fff',
                           pointBorderWidth: 2, pointRadius: 7, pointHoverRadius: 9 }] },
      plugins: [valueLabels],
      options: {
        responsive: true, maintainAspectRatio: false,
        layout: { padding: { top: 26, left: 4, right: 18 } },
        scales: {
          x: { grid: { display: false }, offset: true,
               ticks: { color: '#9b8fae', font: { size: 11 } } },
          y: { min: lo, max: hi,
               title: { display: true, text: 'VLR rating', color: '#9b8fae', font: AXIS_FONT },
               grid: { color: '#f1ecf6' },
               ticks: { color: '#9b8fae', font: { size: 11 }, stepSize: 0.1,
                        callback: function(v) { return v.toFixed(2); } } }
        },
        plugins: {
          legend: { display: false },
          tooltip: { displayColors: false, padding: 10,
            callbacks: {
              title: function(items) { var r = RAW[items[0].dataIndex]; return r.ev + ' (' + r.org + ')'; },
              label: function(c) { var r = RAW[c.dataIndex];
                return r.r.toFixed(2) + ' rating' + (r.rnd ? ' over ' + r.rnd + ' rounds' : ''); } } }
        }
      }
    });
  }
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(draw); else draw();
})();

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


@article_offseason_moves_bp.route("/")
def article_offseason_moves():
    return (PAGE_HTML
            .replace("__THOUGHTS__", _thoughts_html())
            .replace("__KILLS_JSON__", json.dumps(_kills_points()))
            .replace("__SOC_JSON__", json.dumps(_krostaly_points())))
