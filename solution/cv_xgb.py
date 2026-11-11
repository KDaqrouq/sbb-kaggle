"""Quick GPU screen of a feature set: XGBoost (gpu hist) with the same folds, views and
early-stopping split as train.py. Used only to rank feature ideas faster; real gains are
confirmed with the LightGBM screen in train.py.

    python solution/fastscreen.py --work solution/work_tr [--exclude REGEX] [--seed 0]
"""
import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features  # noqa: E402

PARAMS = {"objective": "binary:logistic", "eval_metric": "auc", "tree_method": "hist",
          "device": "cuda", "eta": 0.03, "max_depth": 8, "min_child_weight": 4,
          "subsample": 0.8, "colsample_bytree": 0.3, "lambda": 2.0, "max_bin": 128}
import os; PREDICT = tuple(os.environ.get("PRED", "orig,flip,rev,revflip").split(","))

ap = argparse.ArgumentParser()
ap.add_argument("--work", type=Path, required=True)
ap.add_argument("--exclude", default="")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tag", default="")
ap.add_argument("--extra", type=Path, default=None, help="dir from make_synth.py: extra no-break series")
ap.add_argument("--extra-weight", type=float, default=0.5)
ap.add_argument("--smooth", type=float, default=0.0, help="label smoothing of training targets")
ap.add_argument("--no-balance", action="store_true")
ap.add_argument("--set", default="", help="param overrides, e.g. max_depth=10,colsample_bytree=0.5")
ap.add_argument("--pseudo", type=Path, default=None, help="dir with test_features_<view>.parquet")
ap.add_argument("--pthr", type=float, default=0.15)
ap.add_argument("--nullres", action="store_true", help="add stats minus their expected no-break value given pre-segment nuisances")
ap.add_argument("--add", type=Path, action="append", default=[], help="dir with nn_features_<view>.parquet to append")
ap.add_argument("--folds-file", default="solution/work/folds.csv")
ap.add_argument("--frac", type=float, default=1.0, help="learning curve: use this share of training series")
ap.add_argument("--wfile", type=Path, default=None, help="csv id,w: per-series training weight multiplier")
ap.add_argument("--btype", type=Path, default=Path("solution/work_hx/btype_all.csv"), help="break types, for the hidden-break report")
ap.add_argument("--spec", action="store_true", help="also train a hidden-break specialist (no-break vs hidden breaks) and report blends")
a = ap.parse_args()
t0 = time.time()
for kv in filter(None, a.set.split(",")):
    k_, v_ = kv.split("=")
    PARAMS[k_] = type(PARAMS[k_])(v_) if k_ in PARAMS else (int(v_) if v_.lstrip("-").isdigit() else (float(v_) if v_.replace(".", "", 1).lstrip("-").isdigit() else v_))
names = list(features.VIEWS) + [f"crop{r}_{v}" for r in range(int(os.environ.get("CROPR", features.CROP_ROUNDS))) for v in features.VIEWS]
if os.environ.get("DECIM"):
    names += ["decim_orig", "decim_rev"]
views = {v: pd.read_parquet(a.work / f"train_features_{v}.parquet") for v in names}
cols = [c for c in views["orig"].columns if not (a.exclude and re.match(a.exclude, c))]
idx = views["orig"].index
for extra_dir in a.add:
    for v in names:
        e = pd.read_parquet(extra_dir / f"nn_features_{v}.parquet").reindex(views[v].index)
        views[v] = pd.concat([views[v], e], axis=1)
    cols += [c for c in e.columns if c not in cols]
_x2 = os.environ.get("EXC2")
if _x2:
    cols = [c for c in cols if not re.search(_x2, c)]
    print("after EXC2", len(cols), flush=True)
views = {v: f.reindex(idx)[cols].to_numpy(np.float32) for v, f in views.items()}
labels = pd.read_csv("train_labels.csv")
y = labels.set_index(labels.columns[0])[labels.columns[1]].reindex(idx).to_numpy().astype(int)
if a.extra is not None:
    emap = pd.read_csv(a.extra / "extra_map.csv")
    EV = {v: pd.read_parquet(a.extra / f"extra_features_{v}.parquet").reindex(emap.id)[cols].to_numpy(np.float32) for v in PREDICT}
    esrc = emap.source.to_numpy()
    elab = emap.label.to_numpy().astype(float) if "label" in emap else np.zeros(len(emap))
    nlen = (views_n := pd.read_parquet(a.work / "train_features_orig.parquet", columns=["n_total"]).reindex(idx).n_total.to_numpy())
    bins = np.array([256, 320, 400, 500, 600, 700, 800, 1025])
if a.pseudo is not None:
    TV = {v: pd.read_parquet(a.pseudo / f"test_features_{v}.parquet")[cols].to_numpy(np.float32) for v in PREDICT}
NR_STATS = ["chow_lr", "trendchow_lr", "rb_chow1", "rb_chow3", "d_chow_lr", "gchow_lr", "dgchow_lr",
            "wh8_bf_logf", "wh8_ks", "dw8_bf_logf", "dw8_ks", "wh3_ks", "wh3_bf_logf", "gpa_bf_logf",
            "dgpa_bf_logf", "tl8_lr", "tlg_lr", "arscan_raw2_lr", "arscan_raw5_lr", "var_lr", "mean_lr",
            "fchow2g_lr", "adfc_d", "kpssc_lr", "hw8_t2_log", "gw_bf_logf"]
NR_NUIS = ["n_total", "boundary_frac", "raw_pre_acf1", "raw_pre_kurt", "wh8_pre_acf1", "wh8_pre_kurt",
           "tl8_lnu_pre", "adfc_pre", "kpssc_pre", "ar_phi1_pre"]
if a.nullres:
    NR_STATS = [c for c in NR_STATS if c in cols]
    si = [cols.index(c) for c in NR_STATS]; ni = [cols.index(c) for c in NR_NUIS]
    sgnlog = lambda v: np.sign(v) * np.log1p(np.abs(v))
wser = pd.read_csv(a.wfile).set_index("id").w.reindex(idx).fillna(1.0).to_numpy() if a.wfile else None
hidden = None
if a.btype.exists():
    bt = pd.read_csv(a.btype).set_index("id").btype.reindex(idx).to_numpy()
    hidden = (bt == "none") & (y == 1)
spec = np.zeros(len(y))
folds = pd.read_csv(a.folds_file).set_index("id").fold.reindex(idx).to_numpy()
pred = np.zeros(len(y))
for k in sorted(np.unique(folds)):
    tr = np.where(folds != k)[0]
    if a.frac < 1:
        tr = np.random.default_rng(a.seed).choice(tr, int(len(tr) * a.frac), replace=False)
    fit, stop = train_test_split(tr, test_size=0.1, stratify=y[tr], random_state=a.seed)
    if a.nullres:
        # cross-fitted: a training negative's expected value comes from a regressor that
        # never saw that series (in-sample residuals would shrink and leak the label)
        neg = fit[y[fit] == 0]
        half = np.random.default_rng(a.seed + k).random(len(neg)) < 0.5
        # every other series gets a random owner too, so all residuals are equally noisy
        owner = (np.random.default_rng(a.seed + 100 + k).random(len(y)) < 0.5).astype(int)
        owner[neg[half]] = 0; owner[neg[~half]] = 1
        add = {v: [] for v in names}
        for j in si:
            regs = []
            for h in (0, 1):
                part = neg[(half if h == 0 else ~half)]
                Xn = np.concatenate([views[v][part][:, ni] for v in names])
                tn = sgnlog(np.concatenate([views[v][part][:, j] for v in names]))
                regs.append(xgb.train({"objective": "reg:squarederror", "tree_method": "hist", "device": "cuda",
                                       "eta": 0.1, "max_depth": 5, "subsample": 0.8, "seed": a.seed},
                                      xgb.DMatrix(Xn, tn), 150))
            for v in names:
                D = xgb.DMatrix(views[v][:, ni])
                p0, p1 = regs[0].predict(D), regs[1].predict(D)
                e = np.where(owner == 0, p1, p0)
                add[v].append(sgnlog(views[v][:, j]) - e)
        views_k = {v: np.column_stack([views[v]] + add[v]).astype(np.float32) for v in names}
    else:
        views_k = views
    if os.environ.get("NESTED"):
        # nested cross-fitting: this outer fold's training rows get features from networks
        # that never saw this fold (see nested_break.py); the fold's own rows keep standard ones
        nd = Path(os.environ["NESTED"].format(k=k))
        views_k = {}
        for v in names:
            nf = pd.read_parquet(nd / f"nn_features_{v}.parquet").reindex(idx)
            arr = views[v].copy()
            for c_ in nf.columns:
                if c_ in cols:
                    arr[:, cols.index(c_)] = nf[c_].to_numpy(np.float32)
            views_k[v] = arr
        print(f"  nested features from {nd}", flush=True)
    X = np.concatenate([views_k[v][fit] for v in names]); Y = np.tile(y[fit], len(names))
    W = np.ones(len(Y)) if wser is None else np.tile(wser[fit], len(names))
    if a.extra is not None:
        keep = np.isin(esrc, idx[fit])
        ex = np.concatenate([EV[v][keep] for v in PREDICT])
        # per series the same total weight as a real one (4 rows instead of 12)
        ew = np.full(len(ex), a.extra_weight * len(names) / len(PREDICT))
        # keep the class balance within each length bin: positives of that bin absorb the
        # added negative weight, so length does not become a cue
        elen = np.tile(emap.n.to_numpy()[keep], len(PREDICT))
        flen = np.tile(nlen[fit], len(names))
        eb = np.digitize(elen, bins); fb = np.digitize(flen, bins)
        for b_ in ([] if a.no_balance else np.unique(eb)):
            pos = (fb == b_) & (Y == 1)
            if pos.sum():
                W[pos] += ew[eb == b_].sum() / pos.sum()
        X = np.concatenate([X, ex]); Y = np.concatenate([Y, np.tile(elab[keep], len(PREDICT))]); W = np.concatenate([W, ew])
    if a.smooth:
        Y = Y * (1 - 2 * a.smooth) + a.smooth
    dtr = xgb.QuantileDMatrix(X, Y, weight=W, max_bin=PARAMS["max_bin"])
    dst = xgb.DMatrix(views_k["orig"][stop], y[stop])
    m = xgb.train(dict(PARAMS, seed=a.seed), dtr, 5000, evals=[(dst, "stop")],
                  early_stopping_rounds=200, verbose_eval=False)
    if a.pseudo is not None:
        rng0 = (0, m.best_iteration + 1)
        pt = np.mean([m.predict(xgb.DMatrix(TV[v]), iteration_range=rng0) for v in PREDICT], axis=0)
        sel = (pt < a.pthr) | (pt > 1 - a.pthr)
        px = np.concatenate([TV[v][sel] for v in PREDICT]); py = np.tile((pt[sel] > 0.5).astype(float), len(PREDICT))
        pw = np.full(len(py), len(names) / len(PREDICT))
        dtr = xgb.QuantileDMatrix(np.concatenate([X, px]), np.concatenate([Y, py]), weight=np.concatenate([W, pw]),
                                  max_bin=PARAMS["max_bin"])
        m = xgb.train(dict(PARAMS, seed=a.seed), dtr, 5000, evals=[(dst, "stop")],
                      early_stopping_rounds=200, verbose_eval=False)
        print(f"  pseudo: {sel.sum()} test series ({(pt[sel] > 0.5).mean():.2f} positive)", flush=True)
    te = np.where(folds == k)[0]
    rng = (0, m.best_iteration + 1)
    pred[te] = np.mean([m.predict(xgb.DMatrix(views_k[v][te]), iteration_range=rng) for v in PREDICT], axis=0)
    if a.spec:
        sf = fit[(y[fit] == 0) | hidden[fit]]; ss = stop[(y[stop] == 0) | hidden[stop]]
        ds = xgb.QuantileDMatrix(np.concatenate([views_k[v][sf] for v in names]), np.tile(y[sf], len(names)), max_bin=PARAMS["max_bin"])
        ms = xgb.train(dict(PARAMS, seed=a.seed), ds, 5000, evals=[(xgb.DMatrix(views_k["orig"][ss], y[ss]), "stop")],
                       early_stopping_rounds=200, verbose_eval=False)
        spec[te] = np.mean([ms.predict(xgb.DMatrix(views_k[v][te]), iteration_range=(0, ms.best_iteration + 1)) for v in PREDICT], axis=0)
        print(f"  spec it {ms.best_iteration}", flush=True)
    print(f"fold {k}: {roc_auc_score(y[te], pred[te]):.4f} it {m.best_iteration}", flush=True)
auc = roc_auc_score(y, pred)
if hidden is not None:
    hm = (y == 0) | hidden
    print(f"hidden-vs-nobreak AUC {roc_auc_score(y[hm], pred[hm]):.4f}", flush=True)
if a.spec:
    from scipy.special import logit
    print(f"spec: all {roc_auc_score(y, spec):.4f} hidden {roc_auc_score(y[hm], spec[hm]):.4f}", flush=True)
    lp, ls_ = logit(np.clip(pred, 1e-6, 1 - 1e-6)), logit(np.clip(spec, 1e-6, 1 - 1e-6))
    for wt in (0.1, 0.2, 0.3, 0.5):
        b = (1 - wt) * lp + wt * ls_
        print(f"  blend w={wt}: all {roc_auc_score(y, b):.4f} hidden {roc_auc_score(y[hm], b[hm]):.4f}", flush=True)
    pd.DataFrame({"id": idx, "spec": spec}).to_csv(a.work / f"oof_spec{a.tag}_s{a.seed}.csv", index=False)
pd.DataFrame({"id": idx, "xgb": pred, "label": y}).to_csv(a.work / f"oof_fast{a.tag}_s{a.seed}.csv", index=False)
print(f"FAST {a.work.name}{a.tag} {len(cols)}f frac {a.frac} seed {a.seed}: OOF AUC {auc:.4f} ({time.time() - t0:.0f}s)", flush=True)
