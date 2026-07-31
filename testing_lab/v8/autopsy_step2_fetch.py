import sqlite3, json, time, urllib.request
DB="/Users/benny_es1/PythonTest/testing_lab/v8/data/vctmm_live.db"
c=sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
tickers=[r[0] for r in c.execute("SELECT market_ticker FROM markets ORDER BY market_ticker")]
out={}
for i in range(0,len(tickers),40):
    chunk=tickers[i:i+40]
    url="https://api.elections.kalshi.com/trade-api/v2/markets?tickers="+",".join(chunk)
    with urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"benpom-research/1.0"}),timeout=30) as r:
        d=json.load(r)
    for m in d.get("markets",[]):
        out[m["ticker"]]={"result":m.get("result"),"status":m.get("status"),"last_price":m.get("last_price"),"close_time":m.get("close_time"),"settlement_ts":m.get("settlement_ts"),"volume":m.get("volume"),"maker_fee":m.get("maker_fee"),"taker_fee":m.get("taker_fee")}
    time.sleep(0.6)
json.dump(out,open("/Users/benny_es1/PythonTest/testing_lab/v8/data/kalshi_markets_meta.json","w"),indent=1)
from collections import Counter
print(len(out),"markets fetched;",Counter(v["result"] or "open" for v in out.values()))
