import sqlite3, json, collections
DB="/Users/benny_es1/PythonTest/testing_lab/v8/data/vctmm_live.db"
c=sqlite3.connect(f"file:{DB}?mode=ro", uri=True); c.row_factory=sqlite3.Row
def band(p): lo=((p-1)//5)*5+1; return f"{lo:02d}-{lo+4:02d}"
out={"generated_utc":"2026-07-28","source":"vctmm_live.db snapshot (VM, VACUUM 2026-07-28)","dry_run_filter":"dry_run=0 only","price_convention":"NO-side cents, 5c bands","bands":{}}
B=collections.defaultdict(lambda: {"fills":{"1":{"n":0,"contracts":0.0,"dollars":0.0},"2":{"n":0,"contracts":0.0,"dollars":0.0}},"orders_placed":{"1":{"n":0,"contracts":0.0},"2":{"n":0,"contracts":0.0}}})
for r in c.execute("SELECT side_role,price_cents,qty FROM fills WHERE dry_run=0"):
    d=B[band(r["price_cents"])]["fills"][str(r["side_role"])]
    d["n"]+=1; d["contracts"]+=r["qty"]; d["dollars"]+=r["qty"]*r["price_cents"]/100.0
for r in c.execute("SELECT side_role,price_cents,qty FROM orders WHERE dry_run=0"):
    d=B[band(r["price_cents"])]["orders_placed"][str(r["side_role"])]
    d["n"]+=1; d["contracts"]+=r["qty"]
for k in sorted(B):
    b=B[k]
    for s in ("1","2"):
        f,o=b["fills"][s],b["orders_placed"][s]
        f["contracts"]=round(f["contracts"],1); f["dollars"]=round(f["dollars"],2); o["contracts"]=round(o["contracts"],1)
        b.setdefault("fill_per_order_contracts",{})[s]=round(f["contracts"]/o["contracts"],4) if o["contracts"] else None
    out["bands"][k]=b
tot=c.execute("SELECT COUNT(*),SUM(qty),SUM(qty*price_cents)/100.0 FROM fills WHERE dry_run=0").fetchone()
out["totals"]={"fills":tot[0],"contracts":round(tot[1],1),"cost_dollars":round(tot[2],2)}
out["notes"]=["side_role 1=accumulate (edge side), 2=hedge tier","orders_placed = rows in orders table (each row one posted order); quote_events place-count similar","fills 719 verified real (dry_run=0); zero dry-run fills in live db"]
json.dump(out,open("/Users/benny_es1/PythonTest/testing_lab/v8/stats/quote_density.json","w"),indent=1)
print(json.dumps(out["totals"]));print({k:(v["fills"]["1"]["n"],v["fills"]["2"]["n"]) for k,v in out["bands"].items()})
