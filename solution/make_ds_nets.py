"""Network features on the DESEASONALIZED series (seasonal.deseason): the existing fold networks
(8/32 and 16/64 no-break nets, 16/64 Gaussianized battery, H1/H0 and hidden-break models) scored
on each series with its pooled harmonic component removed. The networks were trained on all
series families, so a deseasonalized series is in-distribution for them (it looks like the
non-seasonal families), and the likelihood ratios are no longer dominated by the cycle.
Cross-fitted as usual (fold k series scored by fold k networks). Columns prefixed dsn_.

    python solution/make_ds_nets.py --nets solution/nets_all --hh solution/work_hh --out solution/work_dsn
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nullbreak as B  # noqa: E402
import nullfast as NF  # noqa: E402
import nullgauss as G  # noqa: E402
import nullnet as N  # noqa: E402
import nullnet2 as M  # noqa: E402
import seasonal as SE  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--nets", type=Path, required=True)
ap.add_argument("--hh", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--jobs", type=int, default=24)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
S = Path("solution")
dev = "cuda"
from joblib import Parallel, delayed  # noqa: E402

table = pd.read_parquet("train_series.parquet")
ids0 = table["id"].drop_duplicates().to_numpy()
pos = pd.Series(np.arange(len(ids0)), index=ids0).reindex(table["id"].to_numpy()).to_numpy()
parts = np.array_split(np.arange(len(ids0)), a.jobs * 4)
ds = pd.concat(Parallel(n_jobs=a.jobs)(delayed(SE.deseason)(table[np.isin(pos, p)]) for p in parts)).sort_index()
print("deseasonalized", flush=True)
ids, views = N.view_pairs(ds)
idx = pd.Index(ids, name="id")
folds = pd.read_csv(S / "work" / "folds.csv").set_index("id").fold.reindex(ids).to_numpy()
cols = {d: list(pd.read_parquet(S / d / "nn_features_orig.parquet").columns)
        for d in ("work_nn2m8l32XL", "work_nn2m16l64XL", "work_ng16")}
HON = {"q_q1", "p_p1", "nc_post_honest", "pq1_q2", "hon_post_pool", "p1_p2", "p1_p2q", "hon_pre_pool"}
c32 = [c for c in cols["work_nn2m8l32XL"] if "_pit" not in c and c.split("_", 1)[1] not in HON]
c64 = [c for c in cols["work_nn2m16l64XL"] if "_pit" not in c]
bnames = ([f"lb_h1_{c}" for c in B.NAMES] + [f"lb_h0_{c}" for c in B.NAMES] + [f"lb_lr_{c}" for c in B.NAMES]
          + ["lb_lr_sum"] + [f"hb_h_{c}" for c in B.NAMES] + [f"hb_lr_{c}" for c in B.NAMES] + ["hb_lr_sum"])
names = ["dsn_" + c for c in c32 + c64 + cols["work_ng16"] + bnames]
# gate: only series with a clear cycle get these features (NaN elsewhere), so the extra network
# passes at test time are limited to ~15-20% of the series
sea = pd.read_parquet(S / "work_seas" / "nn_features_orig.parquet").reindex(idx)
gate = ((np.expm1(sea["seas_narrow"].to_numpy()) > SE.GATE_NARROW) & (sea["seas_period"].to_numpy() < SE.GATE_PERIOD))
print("gated series:", int(gate.sum()), "of", len(gate), flush=True)
out = {v: np.full((len(ids), len(names)), np.nan) for v in views}


def load(path, mix, lags):
    M.MIX, N.LAGS, M.HIST, M.BOTTLENECK = mix, lags, 0, 0
    net = M.make_net().to(dev)
    net.load_state_dict(torch.load(path, map_location=dev, weights_only=True))
    return net.eval()


with torch.no_grad():
    for k in sorted(np.unique(folds)):
        rows = np.where((folds == k) & gate)[0]
        blk = {v: [] for v in views}
        net = load(a.nets / "work_nn2m8l32XL" / f"nullnet_fold{k}.pt", 8, 32)
        for v in views:
            blk[v].append(NF.features2_fast(net, [views[v][i] for i in rows], pit=False, dev=dev, honest=False))
        net = load(a.nets / "work_nn2m16l64XL" / f"nullnet_fold{k}.pt", 16, 64)
        for v in views:
            f, ns = NF.features2_fast(net, [views[v][i] for i in rows], pit=False, normal=True, dev=dev)
            blk[v] += [f, G.battery(ns)]
        h1 = load(a.nets / "work_h1" / f"break1_fold{k}.pt", 16, 64)
        h0 = load(a.nets / "work_h0" / f"break0_fold{k}.pt", 16, 64)
        hh = load(a.hh / f"breakh_fold{k}.pt", 16, 64)
        for v in views:
            pr = [views[v][i] for i in rows]
            r1, r0, rh = NF.break_score_fast(h1, pr, dev), NF.break_score_fast(h0, pr, dev), NF.break_score_fast(hh, pr, dev)
            n = np.array([len(q) for _, q in pr], float)
            blk[v] += [r1, r0, r1 - r0, (n * (r1[:, 0] - r0[:, 0]))[:, None], rh, rh - r0, (n * (rh[:, 0] - r0[:, 0]))[:, None]]
            out[v][rows] = np.hstack(blk[v])
        torch.cuda.empty_cache()
        print(f"fold {k} done", flush=True)
for v, arr in out.items():
    pd.DataFrame(arr, index=idx, columns=names).to_parquet(a.out / f"nn_features_{v}.parquet")
print("done", a.out)
