"""Final XGBoost fits on all 24,000 training series (12 training views), same features, params and
views as the fastscreen_pv_x.py CV run of the configuration. Round count per seed = 1.1 x the mean
best iteration over that seed's 5 CV folds (read from its screen log).

    EXC2=... python solution/fit_final_xgb.py --work solution/work_hx --exclude REGEX \
        --add solution/work_nn2m8l32XL ... --seeds 0,1,2 --logs 'solution/queue/x_NAME_s{seed}.log' \
        --out solution/checkpoint_cpu_L1 [--device cuda]

Saves xgb_s<seed>.json plus xgb_meta.json (feature order, rounds). The models are predicted on CPU
at test time (predict_cpu.py), from float32 features, like in the CV run.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features  # noqa: E402

PARAMS = {"objective": "binary:logistic", "eval_metric": "auc", "tree_method": "hist",
          "eta": 0.03, "max_depth": 8, "min_child_weight": 4,
          "subsample": 0.8, "colsample_bytree": 0.3, "lambda": 2.0, "max_bin": 128}
ap = argparse.ArgumentParser()
ap.add_argument("--work", type=Path, required=True)
ap.add_argument("--exclude", default="")
ap.add_argument("--add", type=Path, action="append", default=[])
ap.add_argument("--seeds", required=True)
ap.add_argument("--logs", required=True, help="screen log pattern with {seed}")
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--device", default="cuda")
ap.add_argument("--set", default="")
a = ap.parse_args()
for kv in filter(None, a.set.split(",")):
    k_, v_ = kv.split("=")
    PARAMS[k_] = type(PARAMS[k_])(v_) if k_ in PARAMS else (int(v_) if v_.isdigit() else float(v_))
a.out.mkdir(parents=True, exist_ok=True)
names = list(features.VIEWS) + [f"crop{r}_{v}" for r in range(features.CROP_ROUNDS) for v in features.VIEWS]
views = {v: pd.read_parquet(a.work / f"train_features_{v}.parquet") for v in names}
cols = [c for c in views["orig"].columns if not (a.exclude and re.match(a.exclude, c))]
idx = views["orig"].index
for extra_dir in a.add:
    for v in names:
        e = pd.read_parquet(extra_dir / f"nn_features_{v}.parquet").reindex(views[v].index)
        views[v] = pd.concat([views[v], e], axis=1)
    cols += [c for c in e.columns if c not in cols]
x2 = os.environ.get("EXC2")
if x2:
    cols = [c for c in cols if not re.search(x2, c)]
print("features", len(cols), flush=True)
X = np.concatenate([views[v].reindex(idx)[cols].to_numpy(np.float32) for v in names])
labels = pd.read_csv("train_labels.csv")
y = labels.set_index(labels.columns[0])[labels.columns[1]].reindex(idx).to_numpy().astype(int)
Y = np.tile(y, len(names))
dtr = xgb.QuantileDMatrix(X, Y, max_bin=PARAMS["max_bin"])
rounds = {}
for seed in [int(s) for s in a.seeds.split(",")]:
    its = [int(m) for m in re.findall(r"^fold \d: [0-9.]+ it (\d+)", Path(a.logs.format(seed=seed)).read_text(), re.M)]
    assert len(its) == 5, (seed, its)
    n = int(round(1.1 * np.mean(np.array(its) + 1)))
    f = a.out / f"xgb_s{seed}.json"
    if not f.exists():
        m = xgb.train(dict(PARAMS, seed=seed, device=a.device), dtr, n)
        m.save_model(str(f))
    rounds[seed] = n
    print("seed", seed, "rounds", n, flush=True)
(a.out / "xgb_meta.json").write_text(json.dumps({"feature_names": cols, "rounds": rounds, "params": PARAMS,
                                                "exclude": a.exclude, "exc2": x2, "add": [str(d) for d in a.add]}, indent=1))
print("done", a.out)
