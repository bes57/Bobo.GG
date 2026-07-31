#!/usr/bin/env python3
"""Build TESTING-ONLY data patch from VLR.gg for missing matches.

Writes only under testing_lab/data_patch/. Caches raw HTML in html/.
"""
import csv
import os
import re
import sys
import time
from datetime import datetime, timedelta

sys.path.insert(0, '/Users/benny_es1/VCTMM')
from vctmm.benpom.teams import resolve_team_name  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
HTML = os.path.join(BASE, 'html')
TARGETS = os.path.join(BASE, '..', 'data_patch_targets.csv')

EVENTS = {
    2775: '2026_stage1_late',   # Pacific Stage 1
    2860: '2026_stage1_late',   # Americas Stage 1
    2863: '2026_stage1_late',   # EMEA Stage 1
    2864: '2026_stage1_late',   # China Stage 1
    2953: '2026_stage1_late',   # EWC 2026 Americas Qualifier (late-May block)
    2954: '2026_stage1_late',   # EWC 2026 EMEA Qualifier
    2955: '2026_stage1_late',   # EWC 2026 Pacific Qualifier
    2956: '2026_stage1_late',   # EWC 2026 China Qualifier
    2988: '2026_stage1_late',   # China Evolution Series 2026 Act 2 (late-May CN block)
    2952: '2026_ewc',           # Esports World Cup 2026
}

NAME_OVERRIDES = {
    'DRX': 'KRX',
    'JD Gaming': 'JDG',
}

CANON_MAPS = {'Ascent', 'Bind', 'Haven', 'Split', 'Lotus', 'Sunset', 'Icebox',
              'Abyss', 'Pearl', 'Fracture', 'Breeze', 'Corrode'}

_session = None


def sess():
    global _session
    if _session is None:
        from curl_cffi import requests as cffi_requests
        _session = cffi_requests.Session(impersonate='chrome')
    return _session


def fetch(url, cache_name):
    path = os.path.join(HTML, cache_name)
    if os.path.exists(path) and os.path.getsize(path) > 5000:
        return open(path, encoding='utf-8', errors='replace').read()
    r = sess().get(url, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f'{url} -> HTTP {r.status_code}')
    open(path, 'w', encoding='utf-8').write(r.text)
    time.sleep(1.5)
    return r.text


def resolve_org(raw_name):
    n = raw_name.strip()
    if n in NAME_OVERRIDES:
        return NAME_OVERRIDES[n]
    org = resolve_team_name(n)
    return org


# ---------------- event match-list parsing ----------------

def parse_event_matches(text):
    """Yield dicts: mid, slug, date (datetime.date from VLR label), names, scores, winner_idx."""
    out = []
    cur_date = None
    # tokenize by date labels and match anchors, in document order
    token_re = re.compile(
        r'<div class="wf-label mod-large">\s*([^<]+?)\s*</div>'
        r'|<a href="/(\d+)/([^"]+)" class="wf-module-item match-item[^"]*">(.*?)</a>',
        re.S)
    for m in token_re.finditer(text):
        if m.group(1):
            label = m.group(1).strip()
            try:
                cur_date = datetime.strptime(label, '%a, %B %d, %Y').date()
            except ValueError:
                cur_date = None
            continue
        mid, slug, body = m.group(2), m.group(3), m.group(4)
        teams = []
        for tm in re.finditer(
                r'<div class="match-item-vs-team\s*(mod-winner)?\s*">(.*?)'
                r'match-item-vs-team-score[^>]*>\s*([^<\s]+)', body, re.S):
            winner, tbody, score = tm.group(1), tm.group(2), tm.group(3)
            nm = re.search(r'<span class="flag[^"]*"></span>\s*([^<]+)', tbody)
            if not nm:
                continue
            name = re.sub(r'\s+', ' ', nm.group(1)).strip()
            teams.append({
                'name': name,
                'winner': bool(winner),
                'score': score.strip(),
            })
        if len(teams) == 2:
            out.append({'mid': mid, 'slug': slug, 'date': cur_date, 'teams': teams})
    return out


def load_targets():
    targets = []
    with open(TARGETS, newline='', encoding='utf-8') as f:
        for i, row in enumerate(csv.DictReader(f)):
            dt = datetime.strptime(row['date_utc'], '%Y-%m-%d %H:%M')
            targets.append({
                'idx': i,
                'ticker': row['event_ticker'],
                'dt': dt,
                'date': dt.date(),
                'org_a': row['org_a'],
                'org_b': row['org_b'],
                'raw_a': row['team_a_raw'],
                'raw_b': row['team_b_raw'],
                'winner': row['winner_org'],
            })
    return targets


def build_candidates():
    """Fetch all event match lists (default view + every series sub-list),
    resolve orgs, return deduped candidate VLR matches."""
    cands = []
    seen_mids = set()
    unresolved = set()
    for eid, tag in EVENTS.items():
        texts = []
        main = fetch(f'https://www.vlr.gg/event/matches/{eid}/?series_id=all&page=1',
                     f'evmatches_{eid}_p1.html')
        texts.append(main)
        # the default view truncates; fetch each series sub-list too
        sids = sorted(set(re.findall(r'series_id=(\d+)', main)))
        for sid in sids:
            texts.append(fetch(f'https://www.vlr.gg/event/matches/{eid}/?series_id={sid}',
                               f'evmatches_{eid}_s{sid}.html'))
        for text in texts:
            for m in parse_event_matches(text):
                if m['mid'] in seen_mids:
                    continue
                seen_mids.add(m['mid'])
                orgs = []
                for t in m['teams']:
                    org = resolve_org(t['name'])
                    if org is None:
                        unresolved.add(t['name'])
                    orgs.append(org)
                m['orgs'] = orgs
                m['event_id'] = eid
                m['event_tag'] = tag
                cands.append(m)
    return cands, unresolved


def match_targets(targets, cands):
    """One-to-one greedy assignment by (org pair, min date diff <=1 day, same block)."""
    pairs = []  # (datediff, target, cand)
    for t in targets:
        tp = frozenset([t['org_a'], t['org_b']])
        for c in cands:
            if None in c['orgs'] or c['date'] is None:
                continue
            if frozenset(c['orgs']) != tp:
                continue
            dd = abs((c['date'] - t['date']).days)
            if dd > 1:
                continue
            pairs.append((dd, t['idx'], c['mid'], t, c))
    pairs.sort(key=lambda x: (x[0], x[1]))
    used_t, used_c = set(), set()
    assign = {}
    for dd, tidx, mid, t, c in pairs:
        if tidx in used_t or mid in used_c:
            continue
        used_t.add(tidx)
        used_c.add(mid)
        assign[tidx] = c
    return assign


# ---------------- match page parsing (stage 2) ----------------

BLOCK_RE = re.compile(
    r'<div class="vm-stats-game(?: mod-active)?\s*" data-game-id="(\d+|all)"(.*?)'
    r'(?=<div class="vm-stats-game(?: mod-active)?\s*" data-game-id=|\Z)', re.S)
PLAYER_ROW_RE = re.compile(
    r'href="(/player/\d+/[^"]+)">\s*<div class="ovw-player-name text-of">([^<]+)</div>'
    r'\s*<div class="ovw-player-tag[^"]*">([^<]*)</div>')


def parse_match(text, mid):
    """Parse a VLR match page into header orgs, per-map results, lineups."""
    issues = []
    hdr = re.findall(
        r'match-header-link-name[^>]*>.*?<div class="wf-title-med[^"]*">\s*(.*?)\s*</div>',
        text, re.S)
    hdr = [re.sub(r'\s+', ' ', h).strip() for h in hdr[:2]]
    hdr_orgs = [resolve_org(h) for h in hdr]
    if len(hdr) != 2 or None in hdr_orgs:
        issues.append(f'header team resolution failed: {hdr} -> {hdr_orgs}')

    blocks = {}
    order = []
    for bm in BLOCK_RE.finditer(text):
        gid, body = bm.group(1), bm.group(2)
        if gid not in blocks:
            blocks[gid] = []
            order.append(gid)
        blocks[gid].append(body)

    maps = []
    for gid in order:
        if gid == 'all':
            continue
        # pick the block variant that has a map header (name + 2 scores)
        parsed = None
        for body in blocks[gid]:
            mn = re.search(r'<div class="map">\s*<div[^>]*>\s*<span[^>]*>\s*([A-Za-z]+)', body, re.S)
            scores = re.findall(r'<div class="score[^"]*"[^>]*>\s*(\d+)', body)
            names = re.findall(r'<div class="team-name">\s*(.*?)\s*</div>', body, re.S)
            if mn and len(scores) >= 2 and len(names) >= 2:
                parsed = {
                    'game_id': gid,
                    'map': mn.group(1).strip(),
                    'scores': [int(scores[0]), int(scores[1])],
                    'names': [re.sub(r'\s+', ' ', n).strip() for n in names[:2]],
                }
                break
        if parsed:
            maps.append(parsed)
        else:
            # unplayed decider maps render without scores; only flag if truly odd
            body = blocks[gid][0]
            if re.search(r'<div class="score[^"]*"[^>]*>\s*\d+', body):
                issues.append(f'map block {gid}: could not parse')
    # resolve orgs + winner per map
    out_maps = []
    for mp in maps:
        orgs = [resolve_org(n) for n in mp['names']]
        if None in orgs:
            issues.append(f"map {mp['map']}: unresolved team {mp['names']}")
            continue
        s1, s2 = mp['scores']
        if s1 == s2:
            issues.append(f"map {mp['map']}: tied score {s1}-{s2}")
            continue
        wi = 0 if s1 > s2 else 1
        out_maps.append({
            'map': mp['map'],
            'winner': orgs[wi], 'loser': orgs[1 - wi],
            'wr': mp['scores'][wi], 'lr': mp['scores'][1 - wi],
        })
        if mp['map'] not in CANON_MAPS:
            issues.append(f"non-canonical map name: {mp['map']}")

    # lineups from the game-id="all" overview (fall back to first map block)
    lineup_src = None
    for gid in (['all'] + [g for g in order if g != 'all']):
        for body in blocks.get(gid, []):
            tables = re.split(r'<div class="ovw-table">', body)[1:]
            if len(tables) >= 2 and PLAYER_ROW_RE.search(body):
                lineup_src = tables
                break
        if lineup_src:
            break
    lineups = {}  # org -> list of (url, name)
    if lineup_src and len(hdr_orgs) == 2 and None not in hdr_orgs:
        for i, tb in enumerate(lineup_src[:2]):
            rows = PLAYER_ROW_RE.findall(tb)
            org = hdr_orgs[i]
            lineups[org] = [(u, nm.strip(), tag.strip()) for u, nm, tag in rows]
            if len(rows) != 5:
                issues.append(f'{org}: {len(rows)} players listed (expected 5)')
            tags = {tag.strip() for _, _, tag in rows if tag.strip()}
            if len(tags) > 1:
                issues.append(f'{org}: mixed player tags {tags}')
    else:
        issues.append('no lineup tables found')

    return {'mid': mid, 'hdr': hdr, 'hdr_orgs': hdr_orgs,
            'maps': out_maps, 'lineups': lineups, 'issues': issues}


def stage2():
    import json
    assign = list(csv.DictReader(open(os.path.join(BASE, '_assign.csv'))))
    games, lineup_rows, series_rows, problems = [], [], [], []
    for i, a in enumerate(assign):
        mid = a['mid']
        url = f"https://www.vlr.gg/{mid}/{a['slug']}"
        text = fetch(url, f'match_{mid}.html')
        pm = parse_match(text, mid)
        # per-map rows
        for mp in pm['maps']:
            games.append({
                'match_id': mid, 'event_tag': a['event_tag'], 'date': a['date'],
                'map_name': mp['map'], 'winner': mp['winner'], 'loser': mp['loser'],
                'wr': mp['wr'], 'lr': mp['lr'],
            })
        # series row
        wins = {}
        for mp in pm['maps']:
            wins[mp['winner']] = wins.get(mp['winner'], 0) + 1
        if wins:
            worg = max(wins, key=wins.get)
            wmaps = wins.get(worg, 0)
            lmaps = sum(v for k, v in wins.items() if k != worg)
            fmt = {1: 'bo1', 2: 'bo3', 3: 'bo5'}.get(wmaps, f'bo?{wmaps}')
        else:
            worg, wmaps, lmaps, fmt = '', 0, 0, ''
            pm['issues'].append('no maps parsed')
        oa, ob = sorted([a['org_a'], a['org_b']])
        series_rows.append({
            'match_id': mid, 'date': a['date'], 'event_tag': a['event_tag'],
            'org_a': oa, 'org_b': ob, 'winner_org': worg,
            'series_score': f'{wmaps}-{lmaps}', 'fmt': fmt,
        })
        # winner cross-checks
        if worg != a['winner']:
            pm['issues'].append(
                f"WINNER MISMATCH: target={a['winner']} scraped={worg}")
        try:
            ls, rs = int(a['vlr_s1']), int(a['vlr_s2'])
            if sorted([wmaps, lmaps], reverse=True) != sorted([ls, rs], reverse=True):
                pm['issues'].append(
                    f'series score mismatch: maps say {wmaps}-{lmaps}, list says {ls}-{rs}')
        except ValueError:
            pass
        # lineups
        for org, rows in pm['lineups'].items():
            if org not in (a['org_a'], a['org_b']):
                pm['issues'].append(f'lineup org {org} not in target pair')
            for u, nm, tag in rows:
                lineup_rows.append({'org': org, 'match_id': mid,
                                    'ProfileURL': 'https://www.vlr.gg' + u})
        if pm['issues']:
            problems.append({'ticker': a['ticker'], 'mid': mid, 'issues': pm['issues']})
        print(f"[{i+1}/{len(assign)}] {mid} {a['org_a']}v{a['org_b']} maps={len(pm['maps'])} "
              f"{'ISSUES: ' + '; '.join(pm['issues']) if pm['issues'] else 'ok'}")

    with open(os.path.join(BASE, 'patch_games.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['match_id', 'event_tag', 'date', 'map_name',
                                          'winner', 'loser', 'wr', 'lr'])
        w.writeheader()
        w.writerows(games)
    with open(os.path.join(BASE, 'patch_lineups.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['org', 'match_id', 'ProfileURL'])
        w.writeheader()
        w.writerows(lineup_rows)
    with open(os.path.join(BASE, 'patch_series.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['match_id', 'date', 'event_tag', 'org_a', 'org_b',
                                          'winner_org', 'series_score', 'fmt'])
        w.writeheader()
        w.writerows(series_rows)
    json.dump(problems, open(os.path.join(BASE, '_problems.json'), 'w'), indent=1)
    print(f'\nWrote {len(games)} game rows, {len(lineup_rows)} lineup rows, '
          f'{len(series_rows)} series rows. Problem matches: {len(problems)}')


# ---------------- legacy stage-1 helpers ----------------

def parse_match_page(text, mid):
    """Return dict: team names+orgs, maps [(name, s1, s2)], players per side, series score."""
    res = {'mid': mid}
    # header team names
    hdr = re.findall(r'match-header-link-name[^>]*>.*?<div class="wf-title-med[^"]*">\s*(.*?)\s*</div>',
                     text, re.S)
    res['header_teams'] = [re.sub(r'\s+', ' ', h).strip() for h in hdr[:2]]
    # series score, e.g. <div class="match-header-vs-score"> ... spans
    sm = re.search(r'match-header-vs-score.*?<span[^>]*>\s*(\d+)\s*</span>\s*<span[^>]*>\s*:\s*</span>\s*<span[^>]*>\s*(\d+)\s*</span>', text, re.S)
    res['series'] = (int(sm.group(1)), int(sm.group(2))) if sm else None
    # per-map blocks: vm-stats-game with data-game-id != all
    maps = []
    for gm in re.finditer(r'<div class="vm-stats-game\s*[^"]*" data-game-id="(\d+)"(.*?)(?=<div class="vm-stats-game |\Z)', text, re.S):
        body = gm.group(2)
        mn = re.search(r'<div class="map">.*?<span[^>]*>\s*([A-Za-z]+)', body, re.S)
        scores = re.findall(r'<div class="score[^"]*"[^>]*>\s*(\d+)\s*</div>', body)
        team_names = re.findall(r'<div class="team-name">\s*(.*?)\s*</div>', body, re.S)
        if mn and len(scores) >= 2:
            maps.append({
                'game_id': gm.group(1),
                'map': mn.group(1).strip(),
                's1': int(scores[0]),
                's2': int(scores[1]),
                'names': [re.sub(r'\s+', ' ', n).strip() for n in team_names[:2]],
            })
    res['maps'] = maps
    # players: from the "all maps" stats (game-id=all preferred, else union of maps)
    # profile links appear as /player/<id>/<slug>
    players = {}  # side idx -> set of urls; need per-team association
    res['player_links'] = parse_players(text)
    return res


def parse_players(text):
    """Return list of (team_label_index, profile_url, player_name) using overview blocks.

    VLR (post-July-2026) uses div.ovw-* grids; player rows link /player/<id>/<slug>.
    We associate players to teams via the table/grid grouping per team.
    """
    out = []
    # Strategy: find the game-id="all" overview section; inside, two team blocks each
    # listing 5 players. Try new div layout first, fall back to legacy tables.
    allsec = re.search(r'data-game-id="all"(.*?)(?=<div class="vm-stats-game |\Z)', text, re.S)
    section = allsec.group(1) if allsec else text
    # New layout: team containers with class ovw-table or similar; just split section
    # into two halves by team header if present. Generic approach: find all player rows
    # in order; VLR always lists team1's 5 players then team2's 5 (per block).
    rows = re.findall(r'href="(/player/\d+/[^"]+)"[^>]*>\s*<div[^>]*>\s*([^<]+?)\s*</div>', section)
    if not rows:
        rows = [(u, '') for u in re.findall(r'href="(/player/\d+/[^"]+)"', section)]
    seen = []
    for u, nm in rows:
        if u not in [s[0] for s in seen]:
            seen.append((u, nm))
    return seen


def main():
    targets = load_targets()
    cands, unresolved = build_candidates()
    assign = match_targets(targets, cands)
    print(f'targets={len(targets)} candidates={len(cands)} matched={len(assign)}')
    if unresolved:
        print('UNRESOLVED NAMES:', sorted(unresolved))
    missed = [t for t in targets if t['idx'] not in assign]
    for t in missed:
        print('MISS:', t['ticker'], t['org_a'], t['org_b'], t['date'])
    # save assignment for next stage
    with open(os.path.join(BASE, '_assign.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['ticker', 'target_idx', 'date', 'org_a', 'org_b', 'winner',
                    'mid', 'slug', 'vlr_date', 'vlr_t1', 'vlr_t2', 'vlr_s1', 'vlr_s2',
                    'vlr_winner_org', 'event_tag'])
        for t in targets:
            c = assign.get(t['idx'])
            if not c:
                continue
            widx = 0 if c['teams'][0]['winner'] else (1 if c['teams'][1]['winner'] else -1)
            worg = c['orgs'][widx] if widx >= 0 else ''
            w.writerow([t['ticker'], t['idx'], t['date'], t['org_a'], t['org_b'], t['winner'],
                        c['mid'], c['slug'], c['date'], c['teams'][0]['name'], c['teams'][1]['name'],
                        c['teams'][0]['score'], c['teams'][1]['score'], worg, c['event_tag']])


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'stage2':
        stage2()
    else:
        main()
