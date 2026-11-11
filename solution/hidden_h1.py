"""Hidden-break model: an H1 network fine-tuned only on HIDDEN breaks (positives whose BIC break
decomposition shows no level/trend/AR/scale change, work_hx/btype_all.csv), scored against the
existing H0 network. ~90% of the misordered (break, no-break) pairs involve hidden breaks, and the
general H1 network is dominated by visible breaks, so its post-segment density is a poor
alternative for them.
Cross-fitted like nullbreak.py (fold k scored by networks fine-tuned without fold k). Test-time
cost: one more decode of the post segment per series and view (~10 s per view on 2 cores).

    python solution/hidden_h1.py --nets solution/nets_all --out solution/work_hh [--rounds 4]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F  # noqa: E402
import nullbreak as B  # noqa: E402
import nullfast as NF  # noqa: E402
import nullnet as N  # noqa: E402
import nullnet2 as M  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--nets", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--steps", type=int, default=6000)
ap.add_argument("--rounds", type=int, default=F.CROP_ROUNDS)
ap.add_argument("--types", default="none", help="comma list of break types to fine-tune on")
ap.add_argument("--seed", type=int, default=700)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
dev = "cuda"
M.MIX, N.LAGS, M.HIST, M.BOTTLENECK = 16, 64, 0, 0
log = lambda m: print(m, flush=True)
first = F.CROP_ROUNDS
F.CROP_ROUNDS = a.rounds
ids, views = N.view_pairs(pd.read_parquet("train_series.parquet"))
F.CROP_ROUNDS = first
folds = pd.read_csv("solution/work/folds.csv").set_index("id").fold.reindex(ids).to_numpy()
y = pd.read_csv("train_labels.csv").set_index("id").label.reindex(ids).to_numpy()
bt = pd.read_csv("solution/work_hx/btype_all.csv").set_index("id").btype.reindex(ids).to_numpy()
types = set(a.types.split(","))
sel = (y == 1) & np.isin(bt, list(types))
res = {v: np.zeros((len(ids), 7)) for v in views}
for k in sorted(np.unique(folds)):
    net = M.make_net().to(dev)
    net.load_state_dict(torch.load(a.nets / "work_nn2m16l64XL" / f"nullnet_fold{k}.pt", map_location=dev))
    tr = np.where((folds != k) & sel)[0]
    log(f"fold {k}: fine-tuning on {len(tr)} series of types {sorted(types)}")
    net = B.finetune(net, [views["orig"][i] for i in tr], a.seed + k, a.steps, dev, log)
    torch.save(net.state_dict(), a.out / f"breakh_fold{k}.pt")
    h0 = M.make_net().to(dev)
    h0.load_state_dict(torch.load(a.nets / "work_h0" / f"break0_fold{k}.pt", map_location=dev))
    h0.eval()
    rows = np.where(folds == k)[0]
    with torch.no_grad():
        for v, pairs in views.items():
            pr = [pairs[i] for i in rows]
            rh, r0 = NF.break_score_fast(net, pr, dev), NF.break_score_fast(h0, pr, dev)
            nQ = np.array([len(q) for _, q in pr], float)
            res[v][rows] = np.column_stack([rh, rh - r0, nQ * (rh[:, 0] - r0[:, 0])])
    log(f"fold {k} scored")
names = [f"hb_h_{c}" for c in B.NAMES] + [f"hb_lr_{c}" for c in B.NAMES] + ["hb_lr_sum"]
for v, r in res.items():
    pd.DataFrame(r, index=pd.Index(ids, name="id"), columns=names).to_parquet(a.out / f"nn_features_{v}.parquet")
log(f"done {a.out}")
