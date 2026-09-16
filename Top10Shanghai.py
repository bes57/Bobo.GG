"""
Top 10 Players at Champions Shanghai — countdown listicle article.

Structure mirrors ChampionshipDNA.py: label / h1 / byline / cover / content,
with the fixed .toc rail + scroll-spy. Each ranked section is generated
server-side from data/2026_stage2.csv so the stat chips stay current with the
freshest split data on disk. Article prose is added by the author; this file
renders only the structural elements (rank headers, photos, stat strips).
"""

import os, json
import pandas as pd
from flask import Blueprint

article_top10_shanghai_bp = Blueprint("article_top10_shanghai", __name__)

ROOT     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")

# Rank 1 → 10. Rendered bottom-up (the page counts down from #10 to #1).
# csv: the Player value in data/2026_stage2.csv when it differs from the
# display name the article uses.
RANKING = [
    {"rank": 1,  "name": "Erde",     "csv": "Erde",      "org": "LOUD", "team": "LOUD"},
    {"rank": 2,  "name": "N4rrate",  "csv": "N4rrate",   "org": "KC",   "team": "Karmine Corp"},
    {"rank": 3,  "name": "Lukxo",    "csv": "Lukxo",     "org": "LOUD", "team": "LOUD"},
    {"rank": 4,  "name": "Cryo",     "csv": "Cryocells", "org": "100T", "team": "100 Thieves"},
    {"rank": 5,  "name": "Xavi8k",   "csv": "Xavi8k",    "org": "GE",   "team": "Global Esports"},
    {"rank": 6,  "name": "Asuna",    "csv": "Asuna",     "org": "100T", "team": "100 Thieves"},
    {"rank": 7,  "name": "Brawk",    "csv": "Brawk",     "org": "NRG",  "team": "NRG"},
    {"rank": 8,  "name": "Keiko",    "csv": "Keiko",     "org": "NRG",  "team": "NRG"},
    {"rank": 9,  "name": "Sayonara", "csv": "Sayonara",  "org": "VIT",  "team": "Team Vitality"},
    {"rank": 10, "name": "Slowly",   "csv": "Slowly",    "org": "TYL",  "team": "TYLOO"},
]

ORG_REGION = {"LOUD": "Americas", "KC": "EMEA", "100T": "Americas", "GE": "Pacific",
              "NRG": "Americas", "VIT": "EMEA", "TYL": "CN"}


def _stage2_stats():
    """{lower player name: {rating, acs, adr, kast, kd, rnd}} from the split CSV."""
    out = {}
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, "2026_stage2.csv"))
    except Exception:
        return out
    for _, r in df.iterrows():
        out[str(r.get("Player", "")).lower()] = {
            "rating": r.get("R2.0", ""), "acs": r.get("ACS", ""),
            "adr": r.get("ADR", ""), "kast": r.get("KAST", ""),
            "kd": r.get("K:D", ""), "rnd": r.get("Rnd", ""),
            "apr": r.get("APR", ""),
        }
    return out


_po_cache = None

def _po_ratings():
    """{lower player name: round-weighted R2.0 across 2026 Stage 2 playoff maps}."""
    global _po_cache
    if _po_cache is not None:
        return _po_cache
    out = {}
    try:
        mr = pd.read_csv(os.path.join(DATA_DIR, "match_results.csv"), dtype=str)
        maps = pd.read_csv(os.path.join(DATA_DIR, "maps", "2026_stage2.csv"), dtype=str)
        ev = mr[mr["MatchID"].isin(set(maps["MatchID"]))]
        po_ids = set(ev[ev["MatchName"].str.startswith("Playoffs", na=False)]["MatchID"])
        rounds = {}
        for _, r in ev[ev["MapNum"] != "all"].iterrows():
            try:
                a, b = str(r["Score"]).split("-")
                rounds[(r["MatchID"], r["MapNum"])] = int(a) + int(b)
            except Exception:
                pass
        po = maps[maps["MatchID"].isin(po_ids)].copy()
        po["_rnd"] = [rounds.get((m, n), 0) for m, n in zip(po["MatchID"], po["MapNum"])]
        po["_r"] = pd.to_numeric(po["R2.0"], errors="coerce")
        po = po[po["_r"].notna() & (po["_rnd"] > 0)]
        for nm, g in po.groupby(po["Player"].str.lower()):
            out[nm] = (g["_r"] * g["_rnd"]).sum() / g["_rnd"].sum()
    except Exception:
        pass
    _po_cache = out
    return out


def _gs_po_chart(player, title, stats):
    """Group stage vs playoffs t-chart for one player in 2026 Stage 2,
    round-weighted from the per-map data, with the +/- delta treatment."""
    import pandas as pd
    try:
        mr = pd.read_csv(os.path.join(DATA_DIR, "match_results.csv"), dtype=str)
        maps = pd.read_csv(os.path.join(DATA_DIR, "maps", "2026_stage2.csv"), dtype=str)
        ev = mr[mr["MatchID"].isin(set(maps["MatchID"]))]
        phase = {}
        for _, r in ev[ev["MapNum"] == "all"].iterrows():
            n = str(r["MatchName"])
            phase[r["MatchID"]] = ("po" if n.startswith("Playoffs")
                                   else "gs" if n.startswith("Group Stage") else None)
        rounds = {}
        for _, r in ev[ev["MapNum"] != "all"].iterrows():
            try:
                a, b = str(r["Score"]).split("-")
                rounds[(r["MatchID"], r["MapNum"])] = int(a) + int(b)
            except Exception:
                pass
        x = maps[maps["Player"].str.lower() == player].copy()
        x["_ph"] = x["MatchID"].map(phase)
        x["_rnd"] = [rounds.get((m, n), 0) for m, n in zip(x["MatchID"], x["MapNum"])]
        vals = {}
        for ph in ("gs", "po"):
            g = x[(x["_ph"] == ph) & (x["_rnd"] > 0)].copy()
            for c in ("R2.0", "ADR", "K", "D", "A", "FK", "FD"):
                g[c] = pd.to_numeric(g[c], errors="coerce")
            R = g["_rnd"].sum()
            fk, fd = g["FK"].sum(), g["FD"].sum()
            vals[ph] = {
                "Rating": round(float((g["R2.0"] * g["_rnd"]).sum() / R), 2),
                "KPR":    round(float(g["K"].sum() / R), 2),
                "DPR":    round(float(g["D"].sum() / R), 2),
                "APR":    round(float(g["A"].sum() / R), 2),
                "K/D":    round(float(g["K"].sum() / g["D"].sum()), 2),
                "ADR":    round(float((g["ADR"] * g["_rnd"]).sum() / R), 1),
                "FIWR":   round(float(100 * fk / (fk + fd)), 2) if fk + fd else 0.0,
                "FKPR":   round(float(fk / R), 2),
                "FDPR":   round(float(fd / R), 2),
            }
    except Exception:
        return ""
    pct = {"FIWR"}

    def _dspan(d, suf=""):
        cls = "sy-up" if d > 0 else ("sy-dn" if d < 0 else "")
        return f'<span class="{cls}">{d:+.2f}{suf}</span>'

    body = ""
    for stat in stats:
        if stat == "FKPR/FDPR":
            fk1, fk2 = vals["gs"]["FKPR"], vals["po"]["FKPR"]
            fd1, fd2 = vals["gs"]["FDPR"], vals["po"]["FDPR"]
            body += (f'<tr><td>FKPR/FDPR</td><td>{fk1:.2f}/{fd1:.2f}</td><td>{fk2:.2f}/{fd2:.2f}</td>'
                     f'<td class="sy-delta">{_dspan(round(fk2 - fk1, 2))}/{_dspan(round(fd2 - fd1, 2))}</td></tr>')
            continue
        v1, v2 = vals["gs"][stat], vals["po"][stat]
        d = round(v2 - v1, 2)
        cls = "sy-up" if d > 0 else ("sy-dn" if d < 0 else "")
        suf = "%" if stat in pct else ""
        body += (f'<tr><td>{stat}</td><td>{v1:.2f}{suf}</td><td>{v2:.2f}{suf}</td>'
                 f'<td class="sy-delta {cls}">{d:+.2f}{suf}</td></tr>')
    return (
        '<div class="chart-wrap">'
        f'<div class="chart-title">{title} &mdash; Group Stage vs. Playoffs (2026 Stage 2)</div>'
        '<table class="sy-tbl"><thead><tr><th></th><th>Group Stage</th><th>Playoffs</th><th>+/-</th></tr></thead>'
        '<tbody>' + body + '</tbody></table>'
        '</div>')


def _xavi_chart():
    return _gs_po_chart("xavi8k", "Xavi8k", ["Rating", "KPR", "DPR", "APR", "K/D"])


def _erde_gf_chart():
    """Round-by-round scores of the Americas Stage 2 Grand Final (LOUD vs
    100T), with the total-rounds row — the differential the copy points at."""
    import pandas as pd
    try:
        mr = pd.read_csv(os.path.join(DATA_DIR, "match_results.csv"), dtype=str)
        maps = pd.read_csv(os.path.join(DATA_DIR, "maps", "2026_stage2.csv"), dtype=str)
        loud_ids = set(maps[maps["Org"] == "LOUD"]["MatchID"])
        gf = mr[mr["MatchID"].isin(loud_ids)
                & mr["MatchName"].str.contains("Grand Final", na=False)]
        mid = gf["MatchID"].iloc[0]
        names = dict(zip(maps[maps["MatchID"] == mid]["MapNum"],
                         maps[maps["MatchID"] == mid]["MapName"]))
        rows, loud_t, opp_t = [], 0, 0
        for _, r in gf[gf["MapNum"] != "all"].iterrows():
            w, l = [int(v) for v in str(r["Score"]).split("-")]
            loud, opp = (w, l) if r["WinnerOrg"] == "LOUD" else (l, w)
            loud_t += loud
            opp_t += opp
            mapname = str(names.get(r["MapNum"], "")).replace("PICK", "")
            b1, b2 = ("<b>", "</b>") if loud > opp else ("", "")
            c1, c2 = ("<b>", "</b>") if opp > loud else ("", "")
            rows.append(f'<tr><td>{mapname}</td><td>{b1}{loud}{b2}</td><td>{c1}{opp}{c2}</td></tr>')
        rows.append(f'<tr class="sy-rwrow"><td>Total rounds</td><td><b>{loud_t}</b></td><td>{opp_t}</td></tr>')
    except Exception:
        return ""
    return (
        '<div class="chart-wrap">'
        '<div class="chart-title">Stage 2 Grand Finals &mdash; LOUD 2&ndash;3 100 Thieves</div>'
        '<table class="sy-tbl"><thead><tr><th></th><th>LOUD</th><th>100T</th></tr></thead>'
        '<tbody>' + "".join(rows) + '</tbody></table>'
        '</div>')


def _erde_chart():
    return _gs_po_chart("erde", "Erde", ["Rating", "K/D", "FIWR", "FKPR/FDPR", "APR"])


def _sections_html():
    stats = _stage2_stats()
    parts = []
    for p in sorted(RANKING, key=lambda x: -x["rank"]):   # 10 first, 1 last
        s = stats.get(p["csv"].lower(), {})
        po = _po_ratings().get(p["csv"].lower())
        chips = "".join(
            f'<div class="ps-chip"><span class="ps-v">{v}</span><span class="ps-l">{l}</span></div>'
            for l, v in [("Overall Stage 2 Rating", s.get("rating", "—")),
                         ("Stage 2 Playoffs Rating", f"{po:.2f}" if po is not None else "—"),
                         ("K/D", s.get("kd", "—")), ("APR", s.get("apr", "—"))]
        )
        parts.append(f'''
      <section class="plr" id="p{p["rank"]}">
        <div class="plr-head">
          <span class="plr-rank">{p["rank"]}</span>
          <div class="plr-id">
            <div class="plr-name">{p["name"]}</div>
            <div class="plr-team"><img src="/static/logos/{p["org"]}.png" alt="{p["team"]}">{p["team"]} &middot; {ORG_REGION[p["org"]]}</div>
          </div>
        </div>
        <div class="plr-photo"><img src="/static/top10/{p["name"].lower()}.jpg" alt="{p["name"]}" loading="lazy"></div>
        <div class="plr-stats">
          <div class="ps-row">{chips}</div>
        </div>{_PLAYER_COPY.get(p["rank"], "")}
      </section>''')
    return ('\n      <hr class="secbreak">\n'.join(parts))


_fiwr_cache = None

def _fiwr_points():
    """Every player-event with 100+ first interactions across all franchised
    events (CN-only and ratings-only events excluded, live split included),
    for the N4rrate beeswarm. FIWR = FK / (FK + FD)."""
    global _fiwr_cache
    if _fiwr_cache is not None:
        return _fiwr_cache
    from MoreTestingMaybeFiles import ALL_EVENTS
    pts = []
    for e in ALL_EVENTS:
        if list(e["regions"].keys()) == ["CN"] or e.get("ratings_only"):
            continue
        path = os.path.join(DATA_DIR, f"{e['id']}.csv")
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if not {"FK", "FD", "Player"}.issubset(df.columns):
            continue
        fk = pd.to_numeric(df["FK"], errors="coerce")
        fd = pd.to_numeric(df["FD"], errors="coerce")
        fi = fk + fd
        for i in df.index[fi >= 100]:
            pts.append({
                "p":   str(df.at[i, "Player"]),
                "org": str(df.at[i, "Org"]) if "Org" in df.columns else "",
                "ev":  e["label"],
                "x":   round(100 * float(fk[i]) / float(fi[i]), 2),
                "fi":  int(fi[i]),
                "n4":  str(df.at[i, "Player"]).lower() == "n4rrate" and e["id"] == "2026_stage2",
            })
            if pts[-1]["n4"] and "ProfileURL" in df.columns:
                try:
                    with open(os.path.join(DATA_DIR, "headshots.json")) as f:
                        pts[-1]["hs"] = json.load(f).get(str(df.at[i, "ProfileURL"]), "")
                except Exception:
                    pts[-1]["hs"] = ""
    _fiwr_cache = pts
    return pts


_keiko_n_cache = None

def _keiko_n():
    """Players in VCT history with a 1.10+ event rating at 3+ internationals."""
    global _keiko_n_cache
    if _keiko_n_cache is not None:
        return _keiko_n_cache
    import pandas as pd
    from MoreTestingMaybeFiles import ALL_EVENTS
    counts, names = {}, {}
    for e in ALL_EVENTS:
        if "International" not in e.get("regions", {}) or e.get("ratings_only"):
            continue
        path = os.path.join(DATA_DIR, f"{e['id']}.csv")
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "R2.0" not in df.columns:
            continue
        r = pd.to_numeric(df["R2.0"], errors="coerce")
        for u, nm in zip(df.loc[r >= 1.10, "ProfileURL"], df.loc[r >= 1.10, "Player"]):
            counts[u] = counts.get(u, 0) + 1
            names[u] = str(nm)
    qual = sorted((u for u, c in counts.items() if c >= 4), key=lambda u: -counts[u])
    disp = [names[u][0].upper() + names[u][1:] for u in qual]
    listed = ", ".join(disp[:-1]) + ", and " + disp[-1] if len(disp) > 1 else "".join(disp)
    _keiko_n_cache = (len(qual), listed)
    return _keiko_n_cache


def _sayo_chart():
    """Stage 1 vs Stage 2 mini-bar pairs for Sayonara (rating, K/D, ADR, CL%,
    FIPR), with Vitality's round win % per stage pinned at the top. Values come
    straight from the split CSVs so they track the freshest data on disk."""
    import pandas as pd
    rows = {}
    for eid in ("2026_stage1", "2026_stage2"):
        df = pd.read_csv(os.path.join(DATA_DIR, f"{eid}.csv"))
        r = df[df["Player"].str.lower() == "sayonara"]
        if r.empty:
            return ""
        r = r.iloc[0]
        fk, fd, rnd = float(r["FK"]), float(r["FD"]), float(r["Rnd"])
        rows[eid] = {
            "Rating": float(r["R2.0"]), "K/D": float(r["K:D"]), "ADR": float(r["ADR"]),
            "Clutch%": float(str(r["CL%"]).rstrip("%")), "FIPR": (fk + fd) / rnd,
        }
    # Round win % per stage
    try:
        from EventLeaderboards import _rgt_org_ratios
        rw = {eid: 100 * _rgt_org_ratios(eid).get("VIT", 0) for eid in rows}
    except Exception:
        rw = {eid: 0 for eid in rows}
    fmt = {"Rating": "{:.2f}", "K/D": "{:.2f}", "ADR": "{:.1f}", "Clutch%": "{:.0f}%", "FIPR": "{:.2f}"}
    dfmt = {"Rating": "{:+.2f}", "K/D": "{:+.2f}", "ADR": "{:+.1f}", "Clutch%": "{:+.0f}%", "FIPR": "{:+.2f}"}

    def delta_cell(d, text):
        cls = "sy-up" if d > 0 else ("sy-dn" if d < 0 else "")
        return f'<td class="sy-delta {cls}">{text}</td>'

    # delta of the displayed (rounded) values, so the column never disagrees
    # with what the two stage cells visibly show
    drw = round(rw["2026_stage2"], 1) - round(rw["2026_stage1"], 1)
    body = (
        f'<tr class="sy-rwrow"><td>Vitality Round Win%</td>'
        f'<td>{rw["2026_stage1"]:.1f}%</td><td>{rw["2026_stage2"]:.1f}%</td>'
        + delta_cell(drw, f"{drw:+.1f}%") + '</tr>'
        # Bracket results in the data: S1 lost the Grand Final; S2 lost Upper R1
        # then Lower R1 of the 8-team playoff (7th-8th).
        '<tr class="sy-rwrow"><td>Vitality Placement</td>'
        '<td>2nd</td><td>7th-8th</td><td class="sy-delta"></td></tr>')
    for stat in ("Rating", "K/D", "ADR", "Clutch%", "FIPR"):
        nd = {"Rating": 2, "K/D": 2, "ADR": 1, "Clutch%": 0, "FIPR": 2}[stat]
        v1 = round(rows["2026_stage1"][stat], nd)
        v2 = round(rows["2026_stage2"][stat], nd)
        body += (f'<tr><td>{stat}</td><td>{fmt[stat].format(v1)}</td><td>{fmt[stat].format(v2)}</td>'
                 + delta_cell(v2 - v1, dfmt[stat].format(v2 - v1)) + '</tr>')
    return (
        '<div class="chart-wrap">'
        '<div class="chart-title">Sayonara &mdash; 2026 Stage 1 vs. Stage 2</div>'
        '<table class="sy-tbl"><thead><tr><th></th><th>Stage 1</th><th>Stage 2</th><th>+/-</th></tr></thead>'
        '<tbody>' + body + '</tbody></table>'
        '</div>')


# Author copy per rank, inserted after the stat strip in that player's section.
_PLAYER_COPY = {
    1: """
        <p>The player that I expect to be the greatest player at Champions Shanghai is none other than Erde. Now, I expect people to disagree with this, but I know exactly what I saw in those playoffs - the greatest player in the world.</p>

        <p>Throughout the Stage 2 split, he was barely a player who got mentioned. In Stage 2 playoffs, he showcased a form that we haven’t seen from an IGL in VCT history, other than players like Valyn, F0rsaken (if you count him), Nats, and Munchkin.</p>

__ERDE_CHART__

        <p style="margin-top:34px">What an abso-fucking-lutely insane turnaround. It only gets crazier, though.</p>

        <ol class="notes">
          <li>Keep in mind that he did all of this while IGLing. Not only that, he IGL’d a team of 3 other rookies + a sophomore to an Americas grandfinals where they <em>barely</em> lost to 100 Thieves (Champs favorites). When I say barely, I mean that the round differential was actually in their favor in the Stage 2 grand finals.
__ERDE_GF__
          </li>
          <li>This massive jump in performance occurred against <em>harder</em> opponents. In their playoff run, they stomped Furia, beat MIBR (14th in the world by BenPom), beat G2 (8th in the world by BenPom), beat NRG (2nd in the world by BenPom), and fell short against 100 Thieves (1st in the world by BenPom). They played the top competition in the strongest region in the world when Erde was performing <b>better</b>.</li>
          <li>He entered VCT leaderboards:
            <ul class="notes">
              <li><a class="xlink" href="/highs/?direction=high&amp;stat=Kills&amp;format=bo5&amp;year=all&amp;context=regional" target="_blank" rel="noopener">10th most kills in a domestic Bo5</a></li>
              <li><a class="xlink" href="/highs/?direction=high&amp;stat=Kill%2FDeath%20Ratio&amp;format=map&amp;year=all&amp;context=regional" target="_blank" rel="noopener">6th-highest K/D in a domestic map</a></li>
            </ul>
          </li>
          <li>He was the highest-rated LOUD player for 4/5 of LOUD’s playoff matches, aside from the first where he was 2nd-highest.</li>
        </ol>

        <p>If you just look at Stage 2 playoffs, Erde:</p>

        <ul class="notes">
          <li>is tied for the 3rd-highest VLR-rating in VCT</li>
          <li>has the 4th-highest K/D in VCT</li>
          <li>has the 3rd-highest FIWR in VCT (30+ first interactions)
            <ul class="notes">
              <li><em>The two players above him are N4rrate and T3xture</em></li>
            </ul>
          </li>
        </ul>

        <p>Doing all of this while IGLing a young team, carrying every match, playing support agents, and finishing 2nd in the most competitive region makes him the best player in the world in my eyes. I expect him to continue being the best player in the world at Champions Shanghai.</p>""",
    3: """
        <p>Similarly to Cryo, Lukxo has been one of the most hyped talents stuck on poor teams for the past two years. The difference is that, while Cryo has a longer career with international experience but is coming into the tournament in moderate form, Lukxo has no international experience but is coming into the tournament in incredible form.</p>

        <p>To exemplify the point about Lukxo’s hype coming into this tournament, let’s look at one of the statistics I’ve created (available in my <a class="xlink" href="/vct/" target="_blank" rel="noopener">Leaderboards</a> page), called RGT (Rating Given Team). What it measures, in simplest terms, how impressive a player’s VLR-rating is given their team by normalizing said rating to the team’s round win% + the player’s role. For instance, being a duelist with a 1.05 rating on a team that wins 55% of its rounds is worth a 1.00 RGT and being a duelist with a 0.95 rating on a team that wins 45% of its rounds is worth a 1.01 RGT. Those two are about comparable. Players with high RGT are overperforming their team while players with low RGT are underperforming their team.</p>

        <p>Out of 2538 entries (combinations of players and splits) in the history of franchised VCT, Lukxo is one of five players to be in the top 25 for RGT at least twice. The others are Aspas, Primmie, Marteen, and Vo0kashu.</p>

        <p>Even without the RGT narrative, it’s been clear for a while how talented Lukxo is, consistently hitting jaw-dropping highlights and topping statistical leaderboards without any team success to show for it. Now, that’s changed.</p>

        <p>Lukxo is arriving at Champions Shanghai, making one of the most anticipated international debuts in VCT history - two years in the making. Not only that, he’s making his debut in form.</p>

        <ul class="notes">
          <li>He has the 2nd-highest K/D in VCT Stage 2 across all regions (1.31)</li>
          <li>He has the 10th-highest VLR-rating out of players attending Champions Shanghai (1.16)</li>
          <li>He has the 5th-highest ADR out of players attending Champions Shanghai (153.9)</li>
        </ul>

        <p>This resume is accompanied by extremely high usage and selflessness, making these stats more impressive:</p>

        <ul class="notes">
          <li>90th+ percentile in both FIPR (0.30) and FKPR (0.17)</li>
          <li>He has the 2nd-highest KPR in VCT Stage 2 across all regions (0.88)</li>
        </ul>

        <p>Between the consistent high-level form he’s shown for the past 2 years, his recent statistics in Stage 2, his showcased ability to play against top teams (barely losing to Champs favorites 100T), and the hype surrounding him, he has to be towards the top of this list. I expect him to validate those who believed in him on the stage he’s long deserved.</p>""",
    4: """
        <p>Of the players on this list, Cryo does not have the best numbers from Stage 2, there’s no doubt.</p>

        <p>However, here’s what we do know:</p>

        <ul class="notes">
          <li>The last international event Cryo attended was EWC 2026, two months ago. There, he was the highest-rated player for 100 Thieves as they won the entire event and he won MVP.</li>
          <li>If you don’t count EWC events, the last international event he was at was Masters Shanghai in 2024, where there were limited statistics recorded (i.e. no VLR-rating, KAST%, HS%, etc.) There, he had the highest ACS and K/D for 100 Thieves as they finished 4th.</li>
          <li>If you want to look past Masters Shanghai because of the limited statistics, the last international event he was at was… Champions Istanbul, before VCT franchising.</li>
        </ul>

        <p>In this way, Cryo is a bit of an enigma. He has been one of the most hyped talents in VCT for years now, but he’s had extremely few opportunities to showcase that talent.</p>

        <p>As FNS said a year ago:</p>

        <div class="quote">“Cryo will win an international tournament, the question is of when and with whom. If he’s available, you get him.”<div class="quote-att">- FNS</div></div>

        <p>In fact, he’s been saying that since 2024.</p>

        <p>Just now, at EWC 2026, we got to see him fully demonstrate said talent (re: winning the tournament/MVP). Watching him over the year, it’s clear how much he still exists in that highest tier of VCT pros. His aim has been unreal.</p>

        <p>With 100 Thieves entering Champions as the favorites, Cryo deserves a top spot on this list. It’s true that he hasn’t been in the best form recently, but when given the opportunity to perform at the highest level, he has never been one to falter.</p>

        <p>His ceiling is limitless. Namely, his ceiling is being the best player in the world.</p>""",
    5: """
        <p>My sole Pacific representative, Xavi8k stands alone.</p>

        <p>To quickly acknowledge this: I am sure that <em>at least</em> one player from Nongshim Redforce and/or Paper Rex will be a top-10 player at the tournament. However, as things stand currently, both teams’ players are ~uniformly distributed in their current form. It’s hard for me to confidently highlight one NS/PRX player over anyone in this top 10 list. Neither team have a clear standout player, (maybe) aside from Something on PRX.</p>

        <p>Xavi8k, on the other hand, I have no problem with highlighting. For the Pacific 1-seed, Global Esports, Xavi8k:</p>

        <ul class="notes">
          <li>Has the second-highest KPR on GE</li>
          <li>Has the lowest DPR on GE</li>
          <li>Has the highest KD on GE</li>
          <li>Has the highest APR on GE</li>
          <li>Has the highest VLR rating on GE</li>
          <li>Does all of the above while IGLing</li>
        </ul>

        <p>To reemphasize this: Xavi8k is outperforming his entire team in almost all major aspects while IGLing GE to a Stage 2 title.</p>

        <p>What’s more, his statistics marginally <em>improved</em> from the group stage to the playoffs, despite playing harder teams (on average).</p>

__XAVI_CHART__

        <p style="margin-top:34px">Unlike the situation with NS/PRX, GE have a very clear star in Xavi8k. A high-fragging, team-oriented, playoff-rising IGL who can lead his team to a 1-seed is an absurd occurrence. He doesn’t get the respect he deserves, but I’ll try to give him his credit.</p>""",
    6: """
        <p>Prior to Stage 1 in 2026, Asuna was one of the biggest failures in VCT history. In my opinion, he <em>was</em> the biggest failure. From his success in the pre-franchising era, he has been a story of shaky aim, overcomming, poor statistics, and 1 international attended in franchised VCT. For the amount of hype that he had (and has), this was uncanny.</p>

        <p>What changed? The short answer: Phoenix</p>

        <p>The long answer is this: it’s been a perfect storm of self-discipline, roster allignment, and game state.</p>

        <ol class="notes">
          <li>He’s been vocal in interviews about the fact that this was his first year with a regimented aim routine as well as consciously scaling back his overcomming. Despite scaling back his overcomming, his mid-rounding has been crucial for 100 Thieves, as AC d00mbr0s discussed:
            <div class="quote">“I just think his mid-rounding and mid-round understanding of the state of the map… and being able to do super fast decisions is on another world."<div class="quote-att">- 100 Thieves d00mbr0s</div></div>
          </li>
          <li>Furthermore, this new 100 Thieves roster is built perfectly for him to succeed. Timotino has taken over Asuna’s role in previous 100 Thieves rosters as entry bait. In fact, Timotino has the highest first interactions per round (FIPR) out of every player at Champions Shanghai with 0.38. This allows Asuna to play less selflessly and more selfishly, shining off of Timo’s created space. Vora’s calling is a significant upgrade over Zander’s from 2025. Bang’s return to the roster has re-unlocked the synergy he once had with him in 2024. This new roster showcases Asuna perfectly.</li>
          <li>Lastly, Phoenix is a big piece of the puzzle. Out of every Americas Stage 2 player, Asuna had the HIGHEST INDIVIDUAL AGENT PICK RATE with 95% for Phoenix. He is the purest one-trick in VCT. With the recent rework, Asuna has slotted perfectly onto this agent that allows him to create advantageous duels for himself, support his teammates, and run it down <em>without</em> playing a dive duelist. Being the best player on an agent that many people would consider overtuned is a gigantic plus.</li>
        </ol>

        <p>The result?</p>

        <ul class="notes">
          <li>Asuna was the highest-rated player (1.19) in Americas Stage 2 (which most would consider the most competitive region in VCT)</li>
          <li>Asuna is the third-highest rated player coming into Champions Shanghai.</li>
          <li>Asuna had the second-highest KAST% in VCT in Stage 2</li>
          <li>Asuna has the fourth-highest ADR out of players at Champions Shanghai</li>
          <li>Asuna has the 9th-highest KD out of players at Champions Shanghai</li>
          <li>100 Thieves won the Stage 2 grandfinals with Asuna winning MVP</li>
        </ul>

        <p>Beyond his wide-ranging statistical dominance, he looks like the best player in the best region. For many people, Asuna will be considered the best player at Champions Shanghai. Why not myself?</p>

        <p>A couple reasons:</p>

        <ul class="notes">
          <li>I believe Asuna looks worse upon ignoring his resume and using the eye test. Despite his improved aim (re: aim routine), he’s still incredibly shaky, sporting a 22% headshot rate. This puts him in the <em>4th percentile</em> across all regions’ Stage 2. His clutch% is 14%, which puts him just below average (45th percentile). His FIWR is 52.75%, which is a bit above average (34th percentile). These statistics which indicate his indiviudal duel/aim capabilities (somewhat) back up the eye test.</li>
          <li>To go back to a previous point, it’s very clear that this team builds plays/rounds around him. Playing secondary entry to the most selfless duelist at Champions Shanghai (Timotino) on the best team in VCT (Rank 1 in BenPom) is a hard role not to shine in.</li>
          <li>He does not perform as well at LANs. Granted, we’re operating off of limited sample size, but the trend is consistent.
            <ul class="notes">
              <li>At Masters Shanghai, we don’t have VLR-ratings, but he was one of two 100T players to have a sub-1.0 KD (Boostio was the other).</li>
              <li>At EWC 2026, which they just won, Asuna had a 1.0 rating (average at the event), which ranks him 4th out of the 5 100T players (above Timotino).</li>
              <li>At the 2026 Stage 2 playoffs in Brazil, Asuna had a 1.03 rating (about average, 4th out of the 5 100T players). In the group stage, he had a 1.30 rating (highest in the league).</li>
            </ul>
          </li>
        </ul>

        <p>He’s a great player and lots of people will rate him higher than this, but I find it hard to look past these above reasons. I do not trust Asuna to recreate the same level of production we saw in Americas Stage 2.</p>""",
    7: """
        <p>The reigning Champions MVP is coming into Champions Shanghai postured well to run the MVP trophy back.</p>

        <p>NRG have been up and down over the past month, going from Champions favorites to questionable to make it past groups. Now, some of that is due to a horrendous draw with NS and KC, but the degradation in overall team form has been obvious. Brawk, however, has been an exception.</p>

        <p>He is the second-highest rated player from Stage 2 in Americas and the highest-rated player if you only count the playoffs. His form has excelled recently, being the 1st, 1st, and 2nd highest rated player for NRG in their three playoff matches.</p>

        <p>The Odin has received 0 nerfs since Champions Paris. In conjunction with his recent form, it’d be foolish to doubt Brawk coming into Champions Shanghai.</p>""",
    8: """
        <p>In VCT, there are two players who exist in a tier of “if they qualify for an event, they need to be in top-10 lists for that event”. In my opinion, those players are Aspas and Alfajer.</p>

        <p>In a tier just below that, there are players like Riens, Keiko, Primmie, and Trent, where you should <em>probably</em> put them in top-10 lists.</p>

        <p>Granted, this is an extremely anecdotal statement, but most people know this is the case, consciously or unconsciously. We’ve come to know over the years that there are very few players in VCT history with better mechanics than Keiko. Watching his aim for a single match, let alone the past year, would make this clear.</p>

        <p>He’s also never recorded below a 1.0 rating at an international event. In fact, he only recorded below a 1.10 rating at an international tournament once, on a TL team that went 1-2 in groups at Champions Paris. That’s an insane fact, especially given that he’s attended 5 international events.</p>

        <p>In this year alone, he was the 6th-highest rated player at Masters Santiago, he was the 3rd-highest rated player at Masters London, and he broke the international match VLR-rating record with a 1.85 rating against XLG.</p>

        <p>Yes, he does have one of the lower ratings on this list (1.08 - which is still great) and an average headshot percentage (30%), but he still undoubtedly deserves to be in this top-10 list. He has demonstrated enough consistency and aim over the year (and his career) to be given this 8th-place slot.</p>

        <div class="takeaway"><b>Keiko has recorded a 1.10+ VLR rating at every international he&rsquo;s attended, aside from one (4/5)</b><span>Only __KEIKO_N__ players in franchised VCT history have recorded a 1.10+ rating at 4 or more international events (__KEIKO_NAMES__)</span></div>""",
    9: """
        <p>Anyone who watches Vitality knows how good Sayonara <em>can</em> look. His aim style is super clean and he plays intelligently off of the space that Derke/Jampii create. He’s young and has only looked better as his first year of VCT has gone on.</p>

        <p>Importantly, he underwhelmed at Masters London, but showed progression at EWC. His Stage 2 split was extremely good, putting up better stats than his Stage 1 split but on a marginally worse team (in my opinion).</p>

        <p>Out of the players attending Champs, Sayonara has the:</p>

        <ul class="notes">
          <li>Fourth-highest VLR rating (1.19)</li>
          <li>Third-highest K/D (1.31)</li>
          <li>Highest FIWR (67.44%)</li>
        </ul>

        <p>On one hand, the fact that he’s producing better stats while being on a worse version of Vitality makes a case for him to be a top-5 player. On the other hand, being on a worse team where his passive playstyle leaves him in disadvantageous situations can inflate these stats (K/D + VLR-rating). For instance:</p>

__SAYO_CHART__

        <p style="margin-top:34px">While his rating and K/D have gotten higher, he’s doing less damage, winning fewer clutches, and taking fewer first interactions. He’s taking duels at a lesser frequency, though it’s not clear whether his playstyle or his team is at fault.</p>

        <p>Still, that’s a small caveat to big numbers. We can’t gloss over the fact that he has the highest first-duel win rate and third-highest VLR rating out of every player at Champions. Sayonara is a rookie with insane mechanics, top-tier statistics, and an upwards trajectory. He’s continually outperformed his team domestically, and I hope to see <em>both</em> him and Vitality as a whole come into form at Champions Shanghai.</p>

        <p>I believe he will perform well, but I can’t say the same for Team Vitality.</p>""",
    10: """
        <p>Ranking VCT CN players highly in a list like this is a struggle. Historically, EDG players are the only CN players who consistently attend and, more importantly, show up at international events. With the mediocre to downright terrible form from EDG right now, though, it would be disingenuous to put any EDG players on this list.</p>

        <p>Thankfully for VCT CN, Tyloo have a player who is statistically too good to ignore.</p>

        <p>Out of the players attending Champs, Slowly has:</p>

        <ul class="notes">
          <li>The highest-VLR rating (1.22)</li>
          <li>The fourth-highest K/D (1.3)</li>
          <li>The second-highest ADR (161.7)</li>
          <li>The fourth-highest FIWR (61.35%)</li>
        </ul>

        <p>I’d be lying if I said I’ve watched him play many times, but I did watch him in the VCT CN Stage 2 grandfinals, and the eye test certainly checks out. His playstyle was selfless (re: second-highest ADR) and his mechanics looked just as good as anyone else on this list.</p>

        <p>If you have the time, I highly recommend checking out this frag movie to get a sense for Slowly's insane aim:</p>

        <div class="clip-card">
          <div class="yt-box"><iframe src="https://www.youtube.com/embed/t6twPVClJ_0?rel=0" title="Slowly frag movie" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></div>
        </div>

        <p style="margin-top:34px">If he notched these numbers in Americas, I would probably rank him top 3 in the world, but it’s hard to do that for a player from VCT CN.</p>

        <p>Here are two players who were statistically the best players in VCT CN before heading to their first international, where they dissapointed:</p>

        <ul class="notes">
          <li>XLG Rarga at Masters Toronto</li>
          <li>DRG Vo0kashu at Masters London</li>
        </ul>

        <p>I hope he doesn't join this list.</p>

        <p>Again, it’s hard for me to rate him higher, but it’s hard for me not to rate him whatsoever.</p>""",
    2: """
        <p>N4rrate is, in my opinion, <em>just barely</em> not the best player coming into Champions Shanghai. He is, however, the most valuable player. Karmine Corp aren’t lacking in player quality, but those great players run through N4rrate’s ability to open up each round with a first blood at a historic rate.</p>

        <p>Before discussing his first interactions, let’s just go over the other behemoth stats of N4rrate:</p>

        <ul class="notes">
          <li>Highest K/D in Stage 2 throughout every region (1.34)</li>
          <li>Second-highest VLR-rating in Stage 2 throughout every region <em>of players attending Champions Shanghai</em> (1.21)</li>
          <li>5th-highest KPR in Stage 2 throughout every region <em>of players attending Champions Shanghai</em> (0.85)</li>
        </ul>

        <p>Going back to his first interactions, N4rrate is coming into Champions Shanghai with a 65.55% FIWR (first-interaction win rate), the second-highest of all players attending the event. It’s worth nothing that the player above him (Sayonara with a 67.44% FIWR) has 36% of the amount of first interactions that N4rrate has.</p>

        <p>Having a 65.55% FIWR with 100+ first interactions is historically insane:</p>

        <div class="chart-wrap">
          <div class="chart-title">First-Interaction Win % &mdash; Every Player with 100+ First Interactions in a Single Split</div>
          <div class="strip-box"><canvas id="fiwrSwarm"></canvas><div class="hl-card" id="fiwrCard" style="display:none"></div></div>
        </div>

        <p style="margin-top:34px">He ranks 3rd all-time out of __FIWR_N__ entries, behind only Aspas and Lovers Rock. Insane!</p>

        <p>His ability to have such a high K/D and VLR-rating while being used for first interactions so heavily is ridiculous. What’s more is that, if you watch KC’s games, you’ll see that he creates and wins these high-volume first interactions on his own. A prime example of this is KC’s Lotus.</p>

        <p>Now you could take any Stage 2 match to prove that point, but let’s take the Upper Final against FUT because it was one of his more impressive cases, going 6/0 in FK/FD.</p>

        <div class="clip-card">
          <div class="clip-side">N4rrate's First Interactions vs. FUT on Lotus</div>
          <div class="clip-carousel">
            <button class="cc-arrow cc-prev" id="n4Prev" aria-label="Previous clip">&#8249;</button>
            <div class="cc-viewport">
              <div class="cc-track" id="n4Track">
            <figure class="clip"><video src="/static/top10/clips/n4_1.mp4#t=0.001" controls playsinline preload="metadata"></video><div class="clip-sub"><span class="clip-rd">Round 1</span> &mdash; Non-assisted, solo duel on C main</div></figure>
            <figure class="clip"><video src="/static/top10/clips/n4_2.mp4#t=0.001" controls playsinline preload="metadata"></video><div class="clip-sub"><span class="clip-rd">Round 2</span> &mdash; Non-assisted, solo duel on C main</div></figure>
            <figure class="clip"><video src="/static/top10/clips/n4_3.mp4#t=0.001" controls playsinline preload="metadata"></video><div class="clip-sub"><span class="clip-rd">Round 8</span> &mdash; Non-assisted, solo hold on A main</div></figure>
            <figure class="clip"><video src="/static/top10/clips/n4_4.mp4#t=0.001" controls playsinline preload="metadata"></video><div class="clip-sub"><span class="clip-rd">Round 10</span> &mdash; Non-assisted dry peak into a retake</div></figure>
            <figure class="clip"><video src="/static/top10/clips/n4_6.mp4#t=0.001" controls playsinline preload="metadata"></video><div class="clip-sub"><span class="clip-rd">Round 17</span> &mdash; Non-assisted op kill while solo-clearing B-main</div></figure>
            <figure class="clip"><video src="/static/top10/clips/n4_7.mp4#t=0.001" controls playsinline preload="metadata"></video><div class="clip-sub"><span class="clip-rd">Round 18</span> &mdash; Non-assisted op kill during team push into B-main</div></figure>
              </div>
            </div>
            <button class="cc-arrow cc-next" id="n4Next" aria-label="Next clip">&#8250;</button>
          </div>
          <div class="cc-dots" id="n4Dots"></div>
        </div>

        <p style="margin-top:34px">Granted, FUT did make some questionable decisions (e.g. Skye dry peaks into A lobby despite having teammates/util), but it doesn’t change the fact that N4rrate is taking (and winning) duels in an impactful manner. You can see in the examples how often he is getting these kills completely alone and the map control/contention he offers. It’s not just the numbers - his individual value is second-to-none.</p>

        <p>In KC’s playoffs run, here are the FK/FD statlines N4rrate notched:</p>

        <p>18/4(!), 8/3, and 14/9</p>

        <p>If he continues this into Champions, N4rrate will likely be the best player in the world. That's a big if, though.</p>""",
}


def _toc_html():
    links = ['<a href="#intro">Intro</a>']
    for p in sorted(RANKING, key=lambda x: -x["rank"]):
        # Number only — naming the player here would spoil the countdown.
        links.append(f'<a href="#p{p["rank"]}">#{p["rank"]}</a>')
    return "\n  ".join(links)


PAGE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=820">
<title>Top 10 Players at Champions Shanghai &mdash; Bobo's VCT Database</title>
<meta property="og:type" content="article">
<meta property="og:site_name" content="Bobo gg">
<meta property="og:title" content="Top 10 Players at Champions Shanghai">
<meta property="og:description" content="Counting down the ten best players heading into Champions Shanghai.">
<meta property="og:url" content="https://bobo-gg.net/articles/top10-champions-shanghai/">
<meta property="og:image" content="https://bobo-gg.net/static/top10/cover.jpg">
<meta property="og:image:secure_url" content="https://bobo-gg.net/static/top10/cover.jpg">
<meta property="og:image:type" content="image/jpeg">
<meta property="og:image:width" content="2200">
<meta property="og:image:height" content="1467">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="Top 10 Players at Champions Shanghai">
<meta name="twitter:description" content="Counting down the ten best players heading into Champions Shanghai.">
<meta name="twitter:image" content="https://bobo-gg.net/static/top10/cover.jpg">
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
  .cover { width:100%; border-radius:16px; overflow:hidden; margin-bottom:12px; }
  .cover img { width:100%; height:auto; display:block; }
  .cover-caption { font-size:.75rem; color:var(--soft); font-weight:300; font-style:italic; margin-bottom:48px; text-align:center; }
  .content p { font-size:1rem; font-weight:300; line-height:1.8; color:var(--ink); margin-bottom:24px; }
  .content p b { font-weight:700; }
  .content h2 { font-family:'Plus Jakarta Sans',sans-serif; font-size:1.54rem; font-weight:800; letter-spacing:-0.5px; margin:48px 0 20px; }
  .content h2, .cover, .plr { scroll-margin-top:84px; }
  .content .secbreak { border:0; border-top:1px solid var(--ink); margin:52px 0 40px; }
  .content .xlink { color:#7c4dd6; font-weight:500; text-decoration:underline;
                    text-decoration-thickness:1px; text-underline-offset:2px; }
  .content .xlink:hover { color:#5b21b6; }
  .content em { font-style:italic; }
  /* Breathing room between a player's stat strip and the start of their write-up */
  .plr-stats + p { margin-top:34px; }
  .content ul.notes, .content ol.notes { margin:0 0 24px; padding-left:22px; }
  .content ul.notes li, .content ol.notes li { font-size:1rem; font-weight:300; line-height:1.8;
                         color:var(--ink); margin-bottom:12px; }
  .chart-wrap { background:white; border-radius:20px; padding:24px 26px 18px;
                box-shadow:0 4px 24px #0000000a; margin:28px 0 8px; }
  /* Inside a list item the next point follows directly — give it real air */
  .content li .chart-wrap { margin-bottom:34px; }
  .chart-title { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.02rem;
                 letter-spacing:-0.2px; color:var(--ink); text-align:center; margin-bottom:10px; }
  .strip-box { position:relative; height:220px; }
  .hl-card { position:absolute; transform:translate(-50%,-100%); margin-top:-14px; background:white;
             border-radius:14px; padding:10px 12px 11px; box-shadow:0 8px 30px #00000022;
             text-align:center; min-width:120px; z-index:5; pointer-events:none;
             border:1.5px solid #d9c9f0; }
  .hl-card::after { content:''; position:absolute; left:calc(50% + var(--arrow-dx, 0px)); bottom:-7px;
                    transform:translateX(-50%) rotate(45deg); width:12px; height:12px; background:white;
                    border-right:1.5px solid #d9c9f0; border-bottom:1.5px solid #d9c9f0; }
  .hl-card img { width:46px; height:46px; border-radius:50%; object-fit:cover; display:block;
                 margin:0 auto 6px; border:2px solid #b79ae0; }
  .hl-card .hc-name { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:.92rem; color:var(--ink); }
  .hl-card .hc-stat { font-size:.72rem; color:#7c4dd6; font-weight:500; margin-top:1px; }
  .hl-card .hc-ev { font-size:.66rem; color:var(--soft); font-weight:300; margin-top:2px; }
  .clip-card { background:white; border-radius:20px; padding:18px 18px 16px;
               box-shadow:0 4px 24px #0000000a; margin:28px 0 8px; }
  .clip-side { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.02rem;
               letter-spacing:-0.2px; color:var(--ink); text-align:center; margin-bottom:12px; }
  /* Horizontal carousel: one full-width clip at a time, arrows to advance. */
  /* Quote bubble */
  .quote { max-width:680px; margin:26px auto; background:#fff; border:1.5px solid #e0d4ec;
           border-left:5px solid #7c4dd6; border-radius:16px; padding:20px 26px;
           font-style:italic; font-weight:300; font-size:1.05rem; line-height:1.7;
           color:var(--ink); }
  .quote-att { font-family:'Plus Jakarta Sans',sans-serif; font-style:normal; font-weight:800;
               font-size:.82rem; color:var(--soft); margin-top:12px; text-align:right; }
  /* Nested sub-bullets inside a notes list */
  .content ul.notes ul.notes, .content ol.notes ul.notes { margin:16px 0 0; }
  /* Takeaway bubble, same treatment as the DNA article's */
  .takeaway { max-width:640px; margin:34px auto 8px; text-align:center;
              background:#fff; border:1.5px solid #e0d4ec; border-radius:22px;
              padding:26px 34px; box-shadow:0 6px 26px #0000000f; }
  .takeaway b { display:block; font-family:'Plus Jakarta Sans',sans-serif; font-weight:800;
                font-style:italic; font-size:1.24rem; line-height:1.35; color:#3d1a6e;
                letter-spacing:-0.2px; }
  .takeaway span { display:block; font-family:'Plus Jakarta Sans',sans-serif; font-weight:800;
                   font-style:italic; font-size:1.02rem; line-height:1.35; color:#7c4dd6;
                   letter-spacing:-0.2px; margin-top:14px; }
  /* Sayonara stage-comparison t-chart */
  .sy-tbl { width:100%; max-width:560px; margin:0 auto; border-collapse:collapse; font-size:.95rem; }
  .sy-tbl th { font-family:'Plus Jakarta Sans',sans-serif; font-size:.7rem; font-weight:800;
               letter-spacing:.08em; text-transform:uppercase; color:var(--soft); text-align:center;
               padding:8px 10px; border-bottom:2px solid var(--line,#e8e0ec); }
  .sy-tbl td { padding:11px 10px; border-bottom:1px solid #f0edf5; text-align:center;
               font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; color:var(--ink); }
  .sy-tbl td:first-child { text-align:left; font-family:'DM Sans',sans-serif; font-weight:500;
                           color:var(--soft); }
  .sy-tbl tr:last-child td { border-bottom:0; }
  .sy-tbl td:nth-child(n+2) { border-left:1px solid #f0edf5; }
  .sy-rwrow td { background:#faf8fd; }
  .sy-tbl .sy-delta { font-size:.86rem; }
  .sy-tbl .sy-up { color:#1f9d55; }
  .sy-tbl .sy-dn { color:#d23b3b; }
  .yt-box { position:relative; width:100%; aspect-ratio:16/9; border-radius:10px; overflow:hidden; }
  .yt-box iframe { position:absolute; inset:0; width:100%; height:100%; }
  .clip-carousel { position:relative; }
  .cc-viewport { overflow:hidden; }
  .cc-track { display:flex; transition:transform .35s ease; }
  .clip { margin:0; flex:0 0 100%; min-width:0; }
  .clip video { width:100%; aspect-ratio:16/9; border-radius:10px; display:block; background:#000; }
  .clip figcaption { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:.78rem;
                     letter-spacing:.06em; color:#7c4dd6; margin-bottom:7px; text-align:center; }
  .clip-sub .clip-rd { color:#7c4dd6; }
  .clip-sub { font-family:'Plus Jakarta Sans',sans-serif; font-size:1.02rem; color:var(--ink);
              font-weight:700; text-align:center; margin-top:10px; }
  .cc-arrow { position:absolute; top:50%; transform:translateY(-50%); z-index:3;
              width:42px; height:42px; border-radius:50%; background:#fff;
              border:1px solid #e6dcf3; color:#7c4dd6; font-size:1.7rem; line-height:1;
              display:flex; align-items:center; justify-content:center; padding:0 0 3px;
              cursor:pointer; box-shadow:0 4px 16px #00000026; transition:all .13s ease; }
  .cc-arrow:hover { background:#f4eefc; border-color:#c9b2ea; }
  .cc-arrow:disabled { opacity:.35; cursor:default; }
  .cc-arrow:disabled:hover { background:#fff; border-color:#e6dcf3; }
  .cc-prev { left:12px; }
  .cc-next { right:12px; }
  .cc-dots { display:flex; justify-content:center; gap:7px; margin-top:12px; }
  .cc-dots button { width:8px; height:8px; border-radius:50%; border:0; padding:0;
                    background:#e4dcf0; cursor:pointer; transition:all .13s ease; }
  .cc-dots button.on { background:#7c4dd6; transform:scale(1.25); }
  /* Countdown sections */
  .plr-head { display:flex; align-items:center; gap:20px; margin:0 0 18px; }
  .plr-rank { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:3.4rem;
              line-height:1; color:#7c4dd6; letter-spacing:-2px; min-width:74px; text-align:center; }
  .plr-name { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.9rem;
              letter-spacing:-0.5px; line-height:1.15; }
  .plr-team { display:flex; align-items:center; gap:8px; font-size:.88rem; font-weight:500;
              color:var(--soft); margin-top:5px; }
  .plr-team img { width:22px; height:22px; object-fit:contain; }
  /* All the player shots are 3:2, so the frame holds its height before the
     lazy image arrives — no layout jump, and scroll-spy offsets stay stable. */
  .plr-photo { width:100%; aspect-ratio:3/2; border-radius:16px; overflow:hidden;
               border:1px solid #ece6f2; box-shadow:0 4px 24px #0000000a; background:#f3eefb; }
  .plr-photo img { width:100%; height:100%; object-fit:cover; display:block; }
  .plr-stats { margin:16px 0 8px; background:#fff; border:1px solid #ece6f2; border-radius:16px;
               padding:16px 20px 18px; box-shadow:0 4px 20px #0000000a; }
  .ps-title { font-family:'Plus Jakarta Sans',sans-serif; font-size:.68rem; font-weight:800;
              letter-spacing:.1em; text-transform:uppercase; color:var(--soft); text-align:center;
              margin-bottom:12px; }
  .ps-row { display:flex; justify-content:center; gap:12px; flex-wrap:wrap; }
  .ps-chip { flex:1 1 0; min-width:88px; max-width:210px; text-align:center; background:#faf8fd;
             border:1px solid #f0eaf7; border-radius:12px; padding:10px 8px 9px; }
  .ps-v { display:block; font-family:'Plus Jakarta Sans',sans-serif; font-weight:800;
          font-size:1.22rem; letter-spacing:-0.3px; color:var(--ink); }
  .ps-l { display:block; font-size:.66rem; font-weight:600; letter-spacing:.08em;
          text-transform:uppercase; color:var(--soft); margin-top:3px; white-space:nowrap; }
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
  __TOC__
</nav>
<div class="page">
  <div class="article">
    <h1>Top 10 Players at Champions <span style="white-space:nowrap">Shanghai</span></h1>
    <div class="byline">Bobo &mdash; September 16, 2026</div>
    <div class="cover" id="intro">
      <img src="/static/top10/cover.jpg" alt="Erde and Lukxo of LOUD">
    </div>
    <p class="cover-caption">LOUD enter Champions Shanghai on the back of star performances from Lukxo and Erde, both arriving at their first international with lofty expectations</p>
    <div class="content">
      <p>Champions Shanghai’s field is set, and is noticeably open. As I discussed in my <a class="xlink" href="/articles/championship-dna/" target="_blank" rel="noopener">historical prelude to Champions Shanghai</a>, most 1-seeds are unexpected teams that went on runs (e.g. GE, Tyloo, and KC), thus they are “question marks” per se. Meanwhile, plenty of the strongest teams of the year are coming into the tournament in poor form (e.g. Vitality, PRX, EDG). Do you default to teams with year-long consistency, or young teams that are catching momentum at the right time?</p>

      <p>This same conundrum applies to players, not just teams. How do you rank players when the best performers from playoffs are entirely different from the best players throughout the overall split? What about the swath of exciting players that are at their first VCT international event? (re: LOUD, TYLOO, KC, 100 Thieves)</p>

      <p>In this ranking, I made my best effort to evaluate the 10 best players coming into Champions Shanghai, predicting which players will impress us the most in the coming weeks.</p>

      <hr class="secbreak">
__SECTIONS__
    </div>
  </div>
</div>
<script>
(function() {
  var el = document.getElementById('fiwrSwarm');
  if (!el || typeof Chart === 'undefined') return;
  var RAW = __FIWR_JSON__;

  // Same organization as the aspas article's strip charts: dots that share a
  // value stack into a tight centered column; everything else hugs the line.
  var hl = null, field = [];
  RAW.forEach(function(p) {
    if (p.n4) hl = { x: p.x, y: 0, raw: p }; else field.push(p);
  });
  var groups = {};
  field.forEach(function(p) { var k = p.x.toFixed(1); (groups[k] = groups[k] || []).push(p); });
  var maxStack = 1;
  Object.keys(groups).forEach(function(k) { if (groups[k].length > maxStack) maxStack = groups[k].length; });
  var step = Math.min(0.16, 1.55 / maxStack);
  var others = [];
  Object.keys(groups).forEach(function(k) {
    var arr = groups[k];
    for (var i = 0; i < arr.length; i++) {
      others.push({ x: arr[i].x, y: (i - (arr.length - 1) / 2) * step, raw: arr[i] });
    }
  });

  var chart;
  function drawStrip() {
  chart = new Chart(el.getContext('2d'), {
    type: 'scatter',
    data: { datasets: [
      { label: 'field', data: others, backgroundColor: 'rgba(149,118,184,0.55)',
        borderColor: '#fff', borderWidth: 0.5, pointRadius: 3.5, pointHoverRadius: 5 },
      { label: 'n4rrate', data: hl ? [hl] : [], backgroundColor: '#7c4dd6',
        borderColor: '#fff', borderWidth: 2, pointRadius: 8, pointHoverRadius: 9 }
    ] },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'nearest', intersect: true },
      layout: { padding: { top: 105, left: 4, right: 12 } },
      scales: {
        x: { min: 30, max: 70,
             title: { display: true, text: 'First-Interaction Win %', color: '#9b8fae',
                      font: { family: "'Plus Jakarta Sans',sans-serif", weight: '800', size: 11 } },
             grid: { color: '#f1ecf6' },
             ticks: { color: '#9b8fae', font: { size: 11 },
                      callback: function(v){ return v + '%'; } } },
        y: { min: -1, max: 1, display: false, grid: { display: false } }
      },
      plugins: {
        legend: { display: false },
        tooltip: { mode: 'nearest', intersect: true, displayColors: false,
          callbacks: { label: function(c) { var r = c.raw.raw;
            return r.p + ' (' + r.org + ', ' + r.ev + ') \u2014 ' + r.x.toFixed(2) + '% on ' + r.fi + ' FIs'; } } }
      }
    }
  });

  var card = document.getElementById('fiwrCard');
  if (hl && card) {
    var h = hl.raw;
    card.innerHTML = (h.hs ? '<img src="' + h.hs + '" alt="">' : '') +
      '<div class="hc-name">N4rrate</div>' +
      '<div class="hc-stat">' + h.x.toFixed(2) + '% FIWR</div>' +
      '<div class="hc-ev">' + h.ev + '</div>';
    var place = function() {
      var meta = chart.getDatasetMeta(1);
      if (!meta.data || !meta.data[0]) return;
      var elpt = meta.data[0];
      card.style.top = elpt.y + 'px';
      card.style.display = 'block';
      var box = card.parentElement, half = card.offsetWidth / 2, pad = 6;
      var center = Math.max(half + pad, Math.min(elpt.x, box.clientWidth - half - pad));
      card.style.left = center + 'px';
      card.style.setProperty('--arrow-dx', (elpt.x - center) + 'px');
    };
    place();
    chart.options.animation = { onComplete: place };
    chart.update();
    window.addEventListener('resize', function() { setTimeout(place, 60); });
  }
  }
  if (document.fonts && document.fonts.ready) {
    document.fonts.ready.then(drawStrip);
  } else {
    drawStrip();
  }
})();

(function() {
  var track = document.getElementById('n4Track');
  if (!track) return;
  var slides = track.children.length;
  var prev = document.getElementById('n4Prev'), next = document.getElementById('n4Next');
  var dots = document.getElementById('n4Dots');
  var idx = 0;
  for (var i = 0; i < slides; i++) {
    var d = document.createElement('button');
    d.setAttribute('aria-label', 'Clip ' + (i + 1));
    (function(i){ d.onclick = function(){ go(i); }; })(i);
    dots.appendChild(d);
  }
  function go(n) {
    idx = Math.max(0, Math.min(slides - 1, n));
    track.style.transform = 'translateX(-' + (idx * 100) + '%)';
    prev.disabled = idx === 0;
    next.disabled = idx === slides - 1;
    Array.from(dots.children).forEach(function(b, i){ b.classList.toggle('on', i === idx); });
    Array.from(track.querySelectorAll('video')).forEach(function(v, i){ if (i !== idx) v.pause(); });
  }
  prev.onclick = function(){ go(idx - 1); };
  next.onclick = function(){ go(idx + 1); };
  go(0);
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


@article_top10_shanghai_bp.route("/")
def article_top10_shanghai():
    return (PAGE_HTML
            .replace("__TOC__", _toc_html())
            .replace("__SECTIONS__", _sections_html())
            .replace("__FIWR_JSON__", json.dumps(_fiwr_points()))
            .replace("__FIWR_N__", str(len(_fiwr_points())))
            .replace("__SAYO_CHART__", _sayo_chart())
            .replace("__KEIKO_N__", str(_keiko_n()[0]))
            .replace("__KEIKO_NAMES__", _keiko_n()[1])
            .replace("__XAVI_CHART__", _xavi_chart())
            .replace("__ERDE_CHART__", _erde_chart())
            .replace("__ERDE_GF__", _erde_gf_chart()))
