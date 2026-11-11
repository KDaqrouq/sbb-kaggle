"""Nested cross-fitting of the H1/H0 break model (removes the second-order leakage).

Standard cross-fitting: series in fold j are scored by H1/H0 networks fine-tuned without fold j,
but those networks saw the labels of every other fold, including the outer validation fold k of
the tree model. Here, for each outer fold k and each inner fold j != k, H1/H0 are fine-tuned on
folds outside {k, j} and score fold j. Fold k itself keeps its standard scores (networks fine-tuned
without fold k), which is exactly what the test series get. The tree model for outer fold k then
trains on features that carry no information about fold k's labels.

Output: <out>/k<k>/nn_features_<view>.parquet (same columns as work_h10), for fastscreen's
NESTED=<out>/k{k} option.

    python solution/nested_break.py --nets solution/nets_all --std solution/work_h10 --out solution/work_h10n
Base networks: the fold-k no-break 16/64 network (trained without fold k; it still saw fold j's
no-break series, a smaller, label-free-ish leak that this script does not remove).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "solution"))  # pipeline modules
import nullbreak as B  # noqa: E402
import nullfast as NF  # noqa: E402
import nullnet as N  # noqa: E402
import nullnet2 as M  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--nets", type=Path, required=True)
ap.add_argument("--std", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--steps", type=int, default=6000)
ap.add_argument("--outer", default="0,1,2,3,4")
a = ap.parse_args()
dev = "cuda"
M.MIX, N.LAGS, M.HIST, M.BOTTLENECK = 16, 64, 0, 0
log = lambda m: print(m, flush=True)
ids, views = N.view_pairs(pd.read_parquet("train_series.parquet"))
folds = pd.read_csv("solution/work/folds.csv").set_index("id").fold.reindex(ids).to_numpy()
y = pd.read_csv("train_labels.csv").set_index("id").label.reindex(ids).to_numpy()
idx = pd.Index(ids, name="id")
for k in [int(s) for s in a.outer.split(",")]:
    od = a.out / f"k{k}"
    if (od / "done").exists():
        continue
    od.mkdir(parents=True, exist_ok=True)
    std = {v: pd.read_parquet(a.std / f"nn_features_{v}.parquet").reindex(idx) for v in views}
    for j in sorted(set(np.unique(folds)) - {k}):
        part = od / f"j{j}.npz"
        if part.exists():
            continue
        rows = np.where(folds == j)[0]
        sc = {}
        for lab in (1, 0):
            net = M.make_net().to(dev)
            net.load_state_dict(torch.load(a.nets / "work_nn2m16l64XL" / f"nullnet_fold{k}.pt", map_location=dev))
            tr = np.where((folds != k) & (folds != j) & (y == lab))[0]
            log(f"outer {k} inner {j}: H{lab} on {len(tr)} series")
            net = B.finetune(net, [views["orig"][i] for i in tr], 500 + 10 * k + j + 100 * lab, a.steps, dev, log)
            with torch.no_grad():
                sc[lab] = {v: NF.break_score_fast(net, [pairs[i] for i in rows], dev) for v, pairs in views.items()}
        np.savez(part, rows=rows, **{f"{lab}_{v}": sc[lab][v] for lab in (1, 0) for v in views})
    for v, pairs in views.items():
        df = std[v].copy()
        nQ = np.array([len(q) for _, q in pairs], float)
        for j in sorted(set(np.unique(folds)) - {k}):
            z = np.load(od / f"j{j}.npz")
            rows = z["rows"]
            r1, r0 = z[f"1_{v}"], z[f"0_{v}"]
            for c_, name in enumerate(B.NAMES):
                df.iloc[rows, df.columns.get_loc(f"lb_h1_{name}")] = r1[:, c_]
                df.iloc[rows, df.columns.get_loc(f"lb_h0_{name}")] = r0[:, c_]
                df.iloc[rows, df.columns.get_loc(f"lb_lr_{name}")] = r1[:, c_] - r0[:, c_]
            df.iloc[rows, df.columns.get_loc("lb_lr_sum")] = nQ[rows] * (r1[:, 0] - r0[:, 0])
        df.to_parquet(od / f"nn_features_{v}.parquet")
    (od / "done").write_text("ok")
    log(f"outer {k} written")
log("all done")
