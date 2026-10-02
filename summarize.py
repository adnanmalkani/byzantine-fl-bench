"""Collect results/*.json into one table: mean +/- std accuracy over seeds."""
import json
import sys
from pathlib import Path

import pandas as pd

rows = []
for p in Path(sys.argv[1] if len(sys.argv) > 1 else "results").glob("*.json"):
    r = json.loads(p.read_text())
    c = r["config"]
    rows.append({"dataset": c["dataset"], "alpha": c["alpha"], "attack": c["attack"],
                 "byz_frac": c["byz_frac"], "aggregator": c["aggregator"],
                 "acc": r.get("final_acc") if r["status"] == "ok" else None})

df = pd.DataFrame(rows)
table = (df.groupby(["dataset", "alpha", "attack", "byz_frac", "aggregator"])["acc"]
           .agg(["mean", "std", "count"]).round(4))
print(table.to_string())
table.to_csv("summary.csv")
print("\nsaved summary.csv")
