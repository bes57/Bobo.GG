"""
Compare Kalshi pre-match prices vs our model's prediction for the same
matchup. Identify trends, calibration, and where we'd have had edge.
"""
import json, os, sys, math, datetime
import numpy as np
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scrapers'))

from ScrapeKalshi import TEAM_ALIAS

# ─── Load Kalshi data ───
with open(os.path.join(ROOT, 'data/kalshi_valorant.json')) as f:
    kalshi = json.load(f)
print(f'Loaded {len(kalshi)} Kalshi markets')

# ─── Load match snapshots ───
with open(os.path.join(ROOT, 'data/map_ratings.json')) as f:
    mr = json.load(f)
snaps = []
for y, yblock in mr['ratings'].items():
    for snap, sdata in yblock['snapshots'].items():
        rd = sdata.get('ref_date')
        if not rd: continue
        d = {t: v.get('overall_rating', 0.0) for t, v in sdata.get('teams', {}).items()}
        snaps.append((rd, d))
snaps.sort(key=lambda x: x[0])


def find_snap_for(date_str):
    """latest snapshot ref_date < date_str"""
    best = None
    for rd, d in snaps:
        if rd < date_str: best = d
        else: break
    return best


def predict_bo3(rA, rB, beta=0.25):
    p = 1/(1+math.exp(-beta*(rA-rB)))
    return p**2*(3-2*p)


def parse_team(name):
    return TEAM_ALIAS.get(name, None)


# ─── Match Kalshi rows to my model prediction ───
mr_csv = pd.read_csv(os.path.join(ROOT, 'data/match_results.csv'))
mr_csv = mr_csv[mr_csv['MapNum']=='all'].copy()
mr_csv['MatchID'] = mr_csv['MatchID'].astype(int)

with open(os.path.join(ROOT, 'data/match_dates.json')) as f:
    dates = json.load(f)

# Build (date, frozenset(team_a, team_b)) -> MatchID
match_id_lookup = {}
maps_files = [
    'data/maps/2026_stage1.csv', 'data/maps/2026_kickoff.csv',
    'data/maps/2026_masters_santiago.csv',
]
for path in maps_files:
    p = os.path.join(ROOT, path)
    if not os.path.exists(p): continue
    df = pd.read_csv(p, usecols=['MatchID','Org'])
    for mid, grp in df.groupby('MatchID'):
        orgs = list(grp['Org'].dropna().unique())
        if len(orgs) == 2:
            d = dates.get(str(int(mid))) or dates.get(int(mid))
            if d:
                key = (d, frozenset(sorted(orgs)))
                match_id_lookup[key] = int(mid)

print(f'Built {len(match_id_lookup)} (date,teams)->MatchID mappings\n')


rows = []
unmatched = []
for k in kalshi:
    a_kal, b_kal = k['team_a_kalshi'], k['team_b_kalshi']
    a_org, b_org = parse_team(a_kal), parse_team(b_kal)
    if not a_org or not b_org:
        unmatched.append((a_kal, b_kal, 'alias'))
        continue
    pa_yes = k['team_a_pre_yes']
    pb_yes = k['team_b_pre_yes']
    if pa_yes is None or pb_yes is None:
        unmatched.append((a_kal, b_kal, 'no_price'))
        continue
    if k['winner_kalshi'] is None:
        unmatched.append((a_kal, b_kal, 'no_winner'))
        continue

    # Normalize the Kalshi market price by removing overround
    # (yes_a + yes_b > 1 due to vig). Normalize to fair probabilities.
    raw_sum = pa_yes + pb_yes
    fair_a = pa_yes / raw_sum
    fair_b = pb_yes / raw_sum
    vig_bps = (raw_sum - 1.0) * 10000

    # Find the model rating snapshot for the match date
    rating_snap = find_snap_for(k['date'])
    if not rating_snap:
        unmatched.append((a_kal, b_kal, 'no_snap'))
        continue
    rA = rating_snap.get(a_org)
    rB = rating_snap.get(b_org)
    if rA is None or rB is None:
        unmatched.append((a_kal, b_kal, f'missing_rating: {a_org if rA is None else b_org}'))
        continue
    model_p_a = predict_bo3(rA, rB, beta=0.25)
    model_p_b = 1 - model_p_a

    # Find MatchID and verify outcome
    key = (k['date'], frozenset([a_org, b_org]))
    mid = match_id_lookup.get(key)
    actual_winner_org = None
    if mid:
        winner = mr_csv[mr_csv['MatchID']==mid]['WinnerOrg'].iloc[0] if (mr_csv['MatchID']==mid).any() else None
        actual_winner_org = winner

    # Winner per Kalshi (use alias to map back)
    kalshi_winner_org = parse_team(k['winner_kalshi'])

    rows.append({
        'date': k['date'],
        'event_ticker': k['event_ticker'],
        'team_a': a_org, 'team_b': b_org,
        'kalshi_a_raw': round(pa_yes, 4),
        'kalshi_b_raw': round(pb_yes, 4),
        'kalshi_a_fair': round(fair_a, 4),
        'kalshi_b_fair': round(fair_b, 4),
        'vig_bps': round(vig_bps, 1),
        'volume_total': round(k['team_a_volume_total']+k['team_b_volume_total'], 0),
        'model_p_a': round(model_p_a, 4),
        'model_p_b': round(model_p_b, 4),
        'a_won': 1 if kalshi_winner_org == a_org else 0,
        'kalshi_winner_org': kalshi_winner_org,
        'actual_winner_org': actual_winner_org,
        'edge_for_a': round(model_p_a - fair_a, 4),
        'match_id': mid,
    })

df = pd.DataFrame(rows)
print(f'Matched {len(df)} markets to (org_a, org_b, model rating)')
print(f'Unmatched: {len(unmatched)}')

# Show unmatched samples for diagnostic
from collections import Counter
reason_counts = Counter([u[2] for u in unmatched])
print(f'Unmatched reasons: {dict(reason_counts)}')
for u in unmatched[:8]:
    print(f'  - {u[0]} vs {u[1]} | {u[2]}')

# Save matched dataset
out_path = os.path.join(ROOT, 'data/kalshi_vs_model.csv')
df.to_csv(out_path, index=False)
print(f'\nSaved matched dataset to {out_path}')

# ────────── Analysis ──────────
print('\n' + '='*70)
print('━━━ ANALYSIS — Kalshi vs Model ━━━')
print('='*70)

# Overall stats
print(f'\nDataset: {len(df)} 2026 VCT matches (pre-match price ~4hr window VWAP)')
print(f'  Mean Kalshi vig: {df["vig_bps"].mean():.0f} bps (= {df["vig_bps"].mean()/100:.1f}pp overround)')
print(f'  Median volume: ${df["volume_total"].median():.0f}')

# Calibration: Kalshi vs Model
print('\n━━━ Calibration: Kalshi pre-match price vs actual outcome ━━━')
print('  (Does Kalshi price equal true win probability?)')
bins = [(0,0.3),(0.3,0.4),(0.4,0.5),(0.5,0.6),(0.6,0.7),(0.7,1.01)]
print(f'  {"Kalshi p(A)":<14}  {"n":>4}  {"actual A win%":>14}  {"diff":>7}')
for lo, hi in bins:
    sub = df[(df['kalshi_a_fair']>=lo)&(df['kalshi_a_fair']<hi)]
    if len(sub) < 5: continue
    p_mean = sub['kalshi_a_fair'].mean()
    actual = sub['a_won'].mean()
    print(f'  [{lo:.2f}, {hi:.2f})    {len(sub):>4}  {actual:>14.4f}  {actual-p_mean:>+7.4f}')

print('\n━━━ Calibration: Model prediction vs actual outcome ━━━')
print(f'  {"Model p(A)":<14}  {"n":>4}  {"actual A win%":>14}  {"diff":>7}')
for lo, hi in bins:
    sub = df[(df['model_p_a']>=lo)&(df['model_p_a']<hi)]
    if len(sub) < 5: continue
    p_mean = sub['model_p_a'].mean()
    actual = sub['a_won'].mean()
    print(f'  [{lo:.2f}, {hi:.2f})    {len(sub):>4}  {actual:>14.4f}  {actual-p_mean:>+7.4f}')

# Aggregate metrics
def brier(p, y): return float(np.mean((np.asarray(p)-np.asarray(y))**2))
def logloss(p, y):
    p = np.clip(np.asarray(p), 1e-9, 1-1e-9); y = np.asarray(y)
    return float(-np.mean(y*np.log(p) + (1-y)*np.log(1-p)))
def ece(p, y, n_bins=10):
    p = np.asarray(p); y = np.asarray(y)
    bins = np.linspace(0, 1, n_bins+1)
    idx = np.clip(np.digitize(p, bins)-1, 0, n_bins-1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any(): continue
        e += (m.sum()/len(p)) * abs(p[m].mean() - y[m].mean())
    return float(e)

print('\n━━━ Aggregate metrics ━━━')
print(f'  {"Source":<22}  {"Brier":>7}  {"LogLoss":>8}  {"ECE":>6}')
print(f'  {"Kalshi (vig-adj)":<22}  {brier(df["kalshi_a_fair"], df["a_won"]):.5f}  '
      f'{logloss(df["kalshi_a_fair"], df["a_won"]):.5f}  {ece(df["kalshi_a_fair"], df["a_won"]):.4f}')
print(f'  {"Kalshi (raw, with vig)":<22}  {brier(df["kalshi_a_raw"], df["a_won"]):.5f}  '
      f'{logloss(df["kalshi_a_raw"], df["a_won"]):.5f}  {ece(df["kalshi_a_raw"], df["a_won"]):.4f}')
print(f'  {"My model":<22}  {brier(df["model_p_a"], df["a_won"]):.5f}  '
      f'{logloss(df["model_p_a"], df["a_won"]):.5f}  {ece(df["model_p_a"], df["a_won"]):.4f}')
print(f'  {"Naive 0.5":<22}  {brier([0.5]*len(df), df["a_won"]):.5f}  '
      f'{logloss([0.5]*len(df), df["a_won"]):.5f}  -')

# Agreement
agree = (np.sign(df['model_p_a']-0.5) == np.sign(df['kalshi_a_fair']-0.5)).sum()
print(f'\n━━━ Agreement: Model and Kalshi agree on the favorite ━━━')
print(f'  Same favorite: {agree}/{len(df)} = {agree/len(df)*100:.1f}%')
disagree = df[np.sign(df['model_p_a']-0.5) != np.sign(df['kalshi_a_fair']-0.5)]
print(f'  Disagreements: {len(disagree)}')

# Edge distribution: model_p_a - kalshi_a_fair
print(f'\n━━━ Edge distribution (model_p_a − kalshi_fair_a) ━━━')
print(f'  mean: {df["edge_for_a"].mean():+.4f}')
print(f'  std:  {df["edge_for_a"].std():.4f}')
print(f'  p5/p25/p50/p75/p95: '
      f'{df["edge_for_a"].quantile(0.05):+.4f} / '
      f'{df["edge_for_a"].quantile(0.25):+.4f} / '
      f'{df["edge_for_a"].quantile(0.50):+.4f} / '
      f'{df["edge_for_a"].quantile(0.75):+.4f} / '
      f'{df["edge_for_a"].quantile(0.95):+.4f}')

# Edge-based betting simulation
print('\n━━━ Edge-based betting backtest ━━━')
print('  Strategy: 1u bet on side where model edge > threshold, at Kalshi yes price')
print(f'  {"thr":>5}  {"bets":>5}  {"win%":>6}  {"$staked":>9}  {"$pnl":>9}  {"ROI":>7}')
for thr in [0.03, 0.05, 0.07, 0.10, 0.15, 0.20]:
    # When edge_for_a > thr: bet on A (buy YES at kalshi_a_raw)
    # When edge_for_a < -thr: bet on B (buy YES at kalshi_b_raw)
    bets, wins, staked, pnl = 0, 0, 0.0, 0.0
    for _, r in df.iterrows():
        e = r['edge_for_a']
        if e > thr:
            price = r['kalshi_a_raw']
            won = (r['a_won'] == 1)
        elif e < -thr:
            price = r['kalshi_b_raw']
            won = (r['a_won'] == 0)
        else:
            continue
        bets += 1
        staked += price
        if won:
            wins += 1
            pnl += (1.0 - price)
        else:
            pnl -= price
    roi = pnl/staked*100 if staked else 0
    print(f'  {thr:>5.2f}  {bets:>5}  {wins/max(bets,1)*100:>5.1f}%  '
          f'${staked:>8.2f}  ${pnl:>+8.2f}  {roi:>+6.2f}%')

# Per-region: where does model do best/worst vs market?
print('\n━━━ Per-region: model edge vs Kalshi ━━━')
ORG_REGIONS = {
    'TL':'EMEA','FNC':'EMEA','NAVI':'EMEA','VIT':'EMEA','BBL':'EMEA','GX':'EMEA','KC':'EMEA','TH':'EMEA',
    'FUT':'EMEA','GIA':'EMEA','MKOI':'EMEA','M8':'EMEA','EF':'EMEA','PCF':'EMEA',
    'SEN':'Americas','G2':'Americas','MIBR':'Americas','NRG':'Americas','100T':'Americas','C9':'Americas',
    'EG':'Americas','KRÜ':'Americas','LEV':'Americas','FUR':'Americas','LOUD':'Americas','ENVY':'Americas',
    'PRX':'Pacific','DRX':'Pacific','T1':'Pacific','TLN':'Pacific','GEN':'Pacific','DFM':'Pacific',
    'ZETA':'Pacific','RRQ':'Pacific','TS':'Pacific','GE':'Pacific','NS':'Pacific','FS':'Pacific',
    'KRX':'Pacific','VL':'Pacific',
    'EDG':'CN','BLG':'CN','TE':'CN','DRG':'CN','ASE':'CN','AG':'CN','XLG':'CN','FPX':'CN',
    'JDG':'CN','NOVA':'CN','TEC':'CN','TYL':'CN','TYLOO':'CN','WOL':'CN',
}

df['region'] = df['team_a'].map(ORG_REGIONS)
print(f'  {"region":<10}  {"n":>4}  {"model_Brier":>11}  {"kalshi_Brier":>13}  {"edge:Brier_diff":>17}')
for rg in ['EMEA','Americas','Pacific','CN']:
    sub = df[df['region']==rg]
    if len(sub) < 5: continue
    mB = brier(sub['model_p_a'], sub['a_won'])
    kB = brier(sub['kalshi_a_fair'], sub['a_won'])
    print(f'  {rg:<10}  {len(sub):>4}  {mB:>11.5f}  {kB:>13.5f}  {(mB-kB):>+17.5f}')

# Biggest disagreements (model favored opposite of Kalshi)
print('\n━━━ Top 10 disagreements where model bet AGAINST Kalshi favorite ━━━')
print('  (winners are real edges, losers are real losses)')
df_dis = df[np.sign(df['model_p_a']-0.5) != np.sign(df['kalshi_a_fair']-0.5)].copy()
df_dis['gap'] = np.abs(df_dis['model_p_a'] - df_dis['kalshi_a_fair'])
df_dis = df_dis.sort_values('gap', ascending=False).head(15)
for _, r in df_dis.iterrows():
    model_pick = r['team_a'] if r['model_p_a'] > 0.5 else r['team_b']
    kalshi_pick = r['team_a'] if r['kalshi_a_fair'] > 0.5 else r['team_b']
    actual = r['team_a'] if r['a_won']==1 else r['team_b']
    win_for_model = 'WIN' if model_pick == actual else 'lose'
    print(f'  {r["date"]} {r["team_a"]:<6} vs {r["team_b"]:<6}  '
          f'model:{r["model_p_a"]:.2f} kalshi:{r["kalshi_a_fair"]:.2f}  '
          f'model_picks={model_pick:<6} kalshi_picks={kalshi_pick:<6} '
          f'actual={actual:<6} → {win_for_model}')

# How does volume correlate with calibration?
print('\n━━━ Calibration by Kalshi market volume ━━━')
df_high = df[df['volume_total'] > 50000]
df_mid  = df[(df['volume_total'] > 5000) & (df['volume_total'] <= 50000)]
df_low  = df[df['volume_total'] <= 5000]
for name, sub in [('high (>$50k)', df_high), ('mid ($5-50k)', df_mid), ('low (<$5k)', df_low)]:
    if len(sub) < 3: continue
    print(f'  {name:<15}  n={len(sub):>3}  '
          f'kalshi_Brier={brier(sub["kalshi_a_fair"], sub["a_won"]):.5f}  '
          f'model_Brier={brier(sub["model_p_a"], sub["a_won"]):.5f}  '
          f'ECE_kalshi={ece(sub["kalshi_a_fair"], sub["a_won"]):.4f}  '
          f'ECE_model={ece(sub["model_p_a"], sub["a_won"]):.4f}')
