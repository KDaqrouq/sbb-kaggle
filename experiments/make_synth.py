"""Synthetic no-break series cut from inside single segments of the training series.

Every pre or post segment of at least MIN_LEN points holds no boundary, so splitting it at a
random point (30-70%, like the real boundaries) gives a series with no break. Features of
the four standard views go to <work>/extra_features_<view>.parquet and the source series of
each synthetic one to <work>/extra_map.csv (so CV keeps it in its source's fold).

    python solution/make_synth.py --work solution/work_syn
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "solution"))  # pipeline modules
import features  # noqa: E402

MIN_LEN = 256
ap = argparse.ArgumentParser()
ap.add_argument("--work", type=Path, required=True)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
t = pd.read_parquet("train_series.parquet").sort_values(["id", "time"])
parts, rows = [], []
for (sid, per), g in t.groupby(["id", "period"], sort=False):
    v = g.value.to_numpy()
    if len(v) < MIN_LEN:
        continue
    b = int(round(len(v) * rng.uniform(0.3, 0.7)))
    new = f"{sid}_s{per}"
    parts.append(pd.DataFrame({"id": new, "time": np.arange(len(v)), "value": v,
                               "period": (np.arange(len(v)) >= b).astype(np.int64)}))
    rows.append((new, sid, per, len(v)))
table = pd.concat(parts, ignore_index=True)
pd.DataFrame(rows, columns=["id", "source", "segment", "n"]).to_csv(a.work / "extra_map.csv", index=False)
print(len(rows), "synthetic series", flush=True)
views = features.compute_views(table)
for v, f in views.items():
    f.to_parquet(a.work / f"extra_features_{v}.parquet")
print("done", {v: f.shape for v, f in views.items()})
