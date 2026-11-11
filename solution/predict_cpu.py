"""CPU inference: test_series.parquet -> predictions CSV (id, break_probability).

Built for a strict inference budget (2 CPU cores, 8 GB RAM, no network, 30 minutes): one fold's
networks, three label-preserving views, lean feature code. Fully deterministic: two cold runs give
identical probabilities.

    python solution/predict_cpu.py --config L3n --test test_series.parquet --out predictions.csv --threads 2

L3n (the final configuration; needs solution/checkpoint_cpu_L3n/): L3 plus the networks run on the
deseasonalized series of the clearly seasonal series (seasonal.gate). L3:
  handcrafted change statistics (features_lean.py) on the series and on its deseasonalized
  version (seasonal.py), seasonal change tests, the 8/32 and 16/64 no-break networks with the
  Gaussianized battery and the lean pipeline on the 16/64 normal scores (nullfast.py, nullgauss.py),
  the H1/H0 and hidden-break likelihood ratios; XGBoost seeds (and LightGBM seeds when the
  checkpoint holds them), averaged over the views orig, flip and revflip.
The other entries of CONFIGS are earlier or experimental configurations kept for reference; each
needs its own checkpoint_cpu_<config>/ folder, which this repository does not ship.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

CONFIGS = {
    "A": {"nets": ["work_nn2m8l32XL", "work_nn2m16l64XL", "work_nn2m16l64dt"], "ng16": True, "ngfull": True,
          "views": ["orig", "revflip"]},
    "B": {"nets": ["work_nn2m8l32XL", "work_nn2m16l64XL"], "ng16": True, "ngfull": False,
          "views": ["orig", "flip", "revflip"]},
    # L1: B minus the moment scans (hx8/hxg/hxd) and the networks' PIT features; lean feature code
    # (skips every block the model does not use, see features_lean.KEEP_*) and the
    # one-encode-per-context network scorer (nullfast)
    "L1": {"nets": ["work_nn2m8l32XL", "work_nn2m16l64XL"], "ng16": True, "ngfull": False,
           "views": ["orig", "flip", "revflip"], "lean": True, "pit": False, "lgb_weight": 0.5},
    # L2: L1 + hidden-break H1 network (hb_*), 8/32 network without its half-segment ("honest")
    # contexts (2.5x faster, CV-neutral); "ngl": lean full pipeline on the 16/64 normal scores
    "L2": {"nets": ["work_nn2m8l32XL", "work_nn2m16l64XL"], "ng16": True, "ngfull": False,
           "views": ["orig", "flip", "revflip"], "lean": True, "pit": False, "lgb_weight": 0.5,
           "honest32": False, "hh": True, "ngl": True},
    # L3: L2 + seasonal change tests (seas_*) + the lean handcrafted battery on the deseasonalized
    # series (ds_*), see seasonal.py
    "L3": {"nets": ["work_nn2m8l32XL", "work_nn2m16l64XL"], "ng16": True, "ngfull": False,
           "views": ["orig", "flip", "revflip"], "lean": True, "pit": False, "lgb_weight": 0.5,
           "honest32": False, "hh": True, "ngl": True, "seas": True, "ds": True},
}
# optional blocks on top of L3 (see shape.py and seasonal.difference): L3s = + shape-only tests,
# L3d = + battery on first differences, L3sd = both; L3n = + networks on the deseasonalized series
# for the clearly seasonal series only (dsn_*, NaN elsewhere, make_ds_nets.py); L4 = L3 + the lean battery on
# the detrended and deseasonalized series (dt_*, seasonal.detrend), L4r = L4 without the ds_* battery;
# L5 = L3 + the lean battery on the stable stretches next to the boundary (tr_*, seasonal.trim)
for _n, _x in (("L3s", {"shape": True}), ("L3d", {"dd": True}), ("L3sd", {"shape": True, "dd": True}),
               ("L3n", {"dsn": True}), ("L3sn", {"shape": True, "dsn": True}),
               ("L4", {"dt": True}), ("L4r", {"dt": True, "ds": False}), ("L5", {"tr": True})):
    CONFIGS[_n] = dict(CONFIGS["L3"], **_x)
HONEST = {"q_q1", "p_p1", "nc_post_honest", "pq1_q2", "hon_post_pool", "p1_p2", "p1_p2q", "hon_pre_pool"}
NETS = {"work_nn2m8l32XL": (8, 32, "none"), "work_nn2m12l48L": (12, 48, "none"),
        "work_nn2m16l64XL": (16, 64, "none"), "work_nn2m16l64dt": (16, 64, "detrend")}

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True, choices=sorted(CONFIGS))
ap.add_argument("--test", default="test_series.parquet")
ap.add_argument("--out", default="predictions.csv")
ap.add_argument("--fold", default="0", help="which CV folds' networks to use (comma list; predictions are averaged over folds and views)")
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--models", default=None, help="folder holding checkpoint_cpu_<config>/, networks/, v9cols/")
a = ap.parse_args()
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[v] = str(a.threads)

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import features as F  # noqa: E402
import nullbreak as B  # noqa: E402
import nullgauss as G  # noqa: E402
import nullgauss2 as G2  # noqa: E402
import nullnet as N  # noqa: E402
import nullnet2 as M  # noqa: E402
import features_lean as FL  # noqa: E402
import nullfast as NF  # noqa: E402

torch.set_num_threads(a.threads)
torch.use_deterministic_algorithms(True, warn_only=True)
cfg = CONFIGS[a.config]
ROOT = Path(a.models) if a.models else HERE
CK, NETS_DIR, COLS = ROOT / f"checkpoint_cpu_{a.config}", ROOT / "networks", ROOT / "v9cols"
dev = "cpu"
FOLDS = [int(f) for f in str(a.fold).split(",")]
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def load_net(path, mix, lags):
    M.MIX, N.LAGS, M.HIST, M.BOTTLENECK = mix, lags, 0, 0
    net = M.make_net().to(dev)
    net.load_state_dict(torch.load(path, map_location=dev, weights_only=True))
    return net.eval()


def NG_lean(scores, size=500):
    """nullgauss2.full_pipeline with the lean feature code (same values on the kept columns)."""
    out = np.zeros((len(scores), F.N_FEATURES))
    rng = np.random.default_rng(0)
    for s in range(0, len(scores), size):
        chunk = [(z + 1e-7 * rng.standard_normal(len(z)), b) for z, b in scores[s:s + size]]
        lens = np.array([len(z) for z, _ in chunk], np.int64)
        bounds = np.array([max(64, min(b, len(z) - 64)) for z, b in chunk], np.int64)
        starts = np.concatenate(([0], np.cumsum(lens)[:-1])).astype(np.int64)
        out[s:s + len(chunk)] = FL._all_features_lean(np.concatenate([z for z, _ in chunk]), starts, bounds, lens)
    return out


table = pd.read_parquet(a.test)
# handcrafted features for the selected views only (same construction as features.compute_views)
ids_l, starts, lengths, bnd, values = F._layout(table)
rv = np.empty_like(values)
for s, n in zip(starts, lengths):
    rv[s:s + n] = values[s:s + n][::-1]
arrays = {"orig": (values, bnd), "flip": (-values, bnd), "rev": (rv, lengths - bnd), "revflip": (-rv, lengths - bnd)}
allf = FL._all_features_lean if cfg.get("lean") else F._all_features
hand = {v: F._frame(allf(arrays[v][0], starts, arrays[v][1], lengths), ids_l) for v in cfg["views"]}
ids = hand[cfg["views"][0]].index
log(f"handcrafted features: {len(ids)} series, views {cfg['views']}")
vids, allviews = N.view_pairs(table, crops=False)
views = {v: allviews[v] for v in cfg["views"]}
colnames = {f: list(pd.read_parquet(COLS / f / "nn_features_orig.parquet").columns)
            for f in list(NETS) + ["work_ng16", "work_ngfull"]}
# LightGBM seeds (meta.json) and/or XGBoost seeds (xgb_meta.json); with both, the final
# probability is LGB_WEIGHT * mean(LightGBM) + (1 - LGB_WEIGHT) * mean(XGBoost)
meta = json.loads((CK / "meta.json").read_text()) if (CK / "meta.json").exists() else {"models": [], "feature_names": []}
cols = meta["feature_names"]
boosters = [lgb.Booster(model_file=str(CK / f"{m}.txt")) for m in meta["models"]]
xmeta = json.loads((CK / "xgb_meta.json").read_text()) if (CK / "xgb_meta.json").exists() else {"rounds": {}, "feature_names": []}
xcols = xmeta["feature_names"]
xboosters = []
if xmeta["rounds"]:
    import xgboost as xgb
    for s in xmeta["rounds"]:
        bst = xgb.Booster(model_file=str(CK / f"xgb_s{s}.json"))
        bst.set_param({"nthread": a.threads, "device": "cpu"})
        xboosters.append(bst)
LGB_WEIGHT = cfg.get("lgb_weight", 1.0) if boosters and xboosters else (1.0 if boosters else 0.0)

extra = {v: [] for v in views}


def _battery_on(tab, prefix):
    """Lean handcrafted battery on a transformed table, for the prediction views."""
    ids_t, starts_t, lengths_t, bnd_t, values_t = F._layout(tab)
    rvt = np.empty_like(values_t)
    for s, n in zip(starts_t, lengths_t):
        rvt[s:s + n] = values_t[s:s + n][::-1]
    arr = {"orig": (values_t, bnd_t), "flip": (-values_t, bnd_t), "rev": (rvt, lengths_t - bnd_t),
           "revflip": (-rvt, lengths_t - bnd_t)}
    res = {}
    for v in views:
        fr = F._frame(allf(arr[v][0], starts_t, arr[v][1], lengths_t), ids_t)
        fr.columns = [prefix + c for c in fr.columns]
        res[v] = fr.reindex(ids)
    return res


if cfg.get("shape"):
    import shape as SH
    for v in views:
        extra[v].append(pd.DataFrame(SH.all_stats(views[v]), index=pd.Index(vids), columns=SH.NAMES).reindex(ids))
    log("shape tests done")
tds = None
if cfg.get("seas") or cfg.get("ds") or cfg.get("dd") or cfg.get("dsn") or cfg.get("dt") or cfg.get("tr"):
    import seasonal as SE
    if cfg.get("seas"):
        for v in views:
            extra[v].append(pd.DataFrame(SE.all_stats(views[v]), index=pd.Index(vids), columns=SE.NAMES).reindex(ids))
        log("seasonal tests done")
    if cfg.get("ds"):
        tds = SE.deseason(table)
        ids_d, starts_d, lengths_d, bnd_d, values_d = F._layout(tds)
        rvd = np.empty_like(values_d)
        for s, n in zip(starts_d, lengths_d):
            rvd[s:s + n] = values_d[s:s + n][::-1]
        arr_d = {"orig": (values_d, bnd_d), "flip": (-values_d, bnd_d), "rev": (rvd, lengths_d - bnd_d),
                 "revflip": (-rvd, lengths_d - bnd_d)}
        for v in views:
            fr = F._frame(allf(arr_d[v][0], starts_d, arr_d[v][1], lengths_d), ids_d)
            fr.columns = ["ds_" + c for c in fr.columns]
            extra[v].append(fr.reindex(ids))
        log("deseasonalized battery done")
    if cfg.get("tr"):
        ttab, tkeep = SE.trim(table)
        trb = _battery_on(ttab, "tr_")
        for v in views:
            fr = trb[v]
            kk = tkeep.reindex(ids)
            if v in ("rev", "revflip"):
                fr["tr_keep_pre"], fr["tr_keep_post"] = kk["tr_keep_post"].to_numpy(), kk["tr_keep_pre"].to_numpy()
            else:
                fr["tr_keep_pre"], fr["tr_keep_post"] = kk["tr_keep_pre"].to_numpy(), kk["tr_keep_post"].to_numpy()
            extra[v].append(fr)
        log("stable-regime battery done")
    if cfg.get("dt"):
        dtb = _battery_on(SE.detrend(table), "dt_")
        for v in views:
            extra[v].append(dtb[v])
        log("detrended battery done")
    if cfg.get("dd"):
        dd = _battery_on(SE.difference(table), "dd_")
        for v in views:
            extra[v].append(dd[v])
        log("differenced battery done")
feats = {(k, v): [hand[v]] + extra[v] for k in FOLDS for v in views}


def add(v, values_, names):
    feats[(k, v)].append(pd.DataFrame(values_, index=pd.Index(vids), columns=names).reindex(ids))


for k in FOLDS:
    with torch.no_grad():
        for folder in cfg["nets"]:
            mix, lags, tf = NETS[folder]
            net = load_net(NETS_DIR / folder / f"nullnet_fold{k}.pt", mix, lags)
            for v in views:
                pr = [M.transform(P, Q, tf) for P, Q in views[v]]
                ns = None
                if cfg.get("lean"):
                    pit = cfg.get("pit", True)
                    honest = not (folder == "work_nn2m8l32XL" and cfg.get("honest32") is False)
                    names = [c for c in colnames[folder] if (pit or "_pit" not in c)
                             and (honest or c.split("_", 1)[1] not in HONEST)]
                    want_ns = folder == "work_nn2m16l64XL" and (cfg["ng16"] or cfg.get("ngl")) and tf == "none"
                    res = NF.features2_fast(net, pr, pit=pit, normal=want_ns, honest=honest)
                    if want_ns:
                        res, ns = res
                    add(v, res, names)
                else:
                    add(v, M.features2(net, pr, dev, True), colnames[folder])
                if folder == "work_nn2m16l64XL":
                    if cfg["ng16"]:
                        add(v, G.battery(ns if ns is not None else G.normal_scores(net, views[v], dev)), colnames["work_ng16"])
                    if cfg.get("ngl"):
                        add(v, NG_lean(ns), colnames["work_ngfull"])
                    if cfg["ngfull"]:
                        sc = G2.ensemble_scores([(net, mix, lags)], views[v], dev)
                        add(v, G2.full_pipeline(sc), colnames["work_ngfull"])
            log(f"fold {k}: {folder} done")
        h = {lab: load_net(NETS_DIR / f"work_h{lab}" / f"break{lab}_fold{k}.pt", 16, 64) for lab in (1, 0)}
        hb = load_net(NETS_DIR / "work_hh" / f"breakh_fold{k}.pt", 16, 64) if cfg.get("hh") else None
        for v in views:
            if cfg.get("lean"):
                r1, r0 = NF.break_score_fast(h[1], views[v]), NF.break_score_fast(h[0], views[v])
            else:
                r1, r0 = B.score(h[1], views[v], dev), B.score(h[0], views[v], dev)
            n = np.array([len(q) for _, q in views[v]], float)
            df = pd.DataFrame(index=pd.Index(vids))
            for j, c in enumerate(B.NAMES):
                df[f"lb_h1_{c}"] = r1[:, j]
                df[f"lb_h0_{c}"] = r0[:, j]
                df[f"lb_lr_{c}"] = r1[:, j] - r0[:, j]
            df["lb_lr_sum"] = n * (r1[:, 0] - r0[:, 0])
            feats[(k, v)].append(df.reindex(ids))
            if hb is not None:
                rh = NF.break_score_fast(hb, views[v])
                dh = pd.DataFrame(index=pd.Index(vids))
                for j, c in enumerate(B.NAMES):
                    dh[f"hb_h_{c}"] = rh[:, j]
                for j, c in enumerate(B.NAMES):
                    dh[f"hb_lr_{c}"] = rh[:, j] - r0[:, j]
                dh["hb_lr_sum"] = n * (rh[:, 0] - r0[:, 0])
                feats[(k, v)].append(dh.reindex(ids))
        log(f"fold {k}: break model done")

if cfg.get("dsn"):
    # networks on the deseasonalized series, only for the clearly seasonal series
    if tds is None:
        tds = SE.deseason(table)
    gmask = SE.gate(SE.all_stats(allviews["orig"]))
    gi = np.flatnonzero(gmask)
    dvids, dall = N.view_pairs(tds, crops=False)
    assert list(dvids) == list(vids)
    c32 = [c for c in colnames["work_nn2m8l32XL"] if "_pit" not in c and c.split("_", 1)[1] not in HONEST]
    c64 = [c for c in colnames["work_nn2m16l64XL"] if "_pit" not in c]
    bn = ([f"lb_h1_{c}" for c in B.NAMES] + [f"lb_h0_{c}" for c in B.NAMES] + [f"lb_lr_{c}" for c in B.NAMES] + ["lb_lr_sum"]
          + [f"hb_h_{c}" for c in B.NAMES] + [f"hb_lr_{c}" for c in B.NAMES] + ["hb_lr_sum"])
    dnames = ["dsn_" + c for c in c32 + c64 + colnames["work_ng16"] + bn]
    for k in FOLDS:
        with torch.no_grad():
            blk = {v: [] for v in views}
            net = load_net(NETS_DIR / "work_nn2m8l32XL" / f"nullnet_fold{k}.pt", 8, 32)
            for v in views:
                blk[v].append(NF.features2_fast(net, [dall[v][i] for i in gi], pit=False, honest=False))
            net = load_net(NETS_DIR / "work_nn2m16l64XL" / f"nullnet_fold{k}.pt", 16, 64)
            for v in views:
                f_, ns_ = NF.features2_fast(net, [dall[v][i] for i in gi], pit=False, normal=True)
                blk[v] += [f_, G.battery(ns_)]
            h1_ = load_net(NETS_DIR / "work_h1" / f"break1_fold{k}.pt", 16, 64)
            h0_ = load_net(NETS_DIR / "work_h0" / f"break0_fold{k}.pt", 16, 64)
            hh_ = load_net(NETS_DIR / "work_hh" / f"breakh_fold{k}.pt", 16, 64)
            for v in views:
                pr = [dall[v][i] for i in gi]
                r1, r0, rh = NF.break_score_fast(h1_, pr), NF.break_score_fast(h0_, pr), NF.break_score_fast(hh_, pr)
                nq = np.array([len(q) for _, q in pr], float)
                blk[v] += [r1, r0, r1 - r0, (nq * (r1[:, 0] - r0[:, 0]))[:, None], rh, rh - r0, (nq * (rh[:, 0] - r0[:, 0]))[:, None]]
                arr = np.full((len(vids), len(dnames)), np.nan)
                arr[gi] = np.hstack(blk[v])
                feats[(k, v)].append(pd.DataFrame(arr, index=pd.Index(vids), columns=dnames).reindex(ids))
    log(f"networks on deseasonalized series done ({len(gi)} series)")

preds, xpreds = [], []
for (k, v) in feats:
    full = pd.concat(feats[(k, v)], axis=1)
    if boosters:
        X = full[cols]
        assert np.isfinite(X[[c for c in cols if not c.startswith("dsn_")]].to_numpy()).all(), f"non-finite features in view {v}"
        preds += [b.predict(X, num_threads=a.threads) for b in boosters]
    if xboosters:
        Xx = full[xcols].to_numpy(np.float32)
        assert np.isfinite(full[[c for c in xcols if not c.startswith("dsn_")]].to_numpy(np.float32)).all(), f"non-finite features in view {v}"
        D = xgb.DMatrix(Xx)
        xpreds += [b.predict(D) for b in xboosters]
p = (LGB_WEIGHT * np.mean(preds, axis=0) if boosters else 0.0) +     ((1 - LGB_WEIGHT) * np.mean(xpreds, axis=0) if xboosters else 0.0)
log(f"models: {len(boosters)} LightGBM, {len(xboosters)} XGBoost, LightGBM weight {LGB_WEIGHT}")
sub = pd.DataFrame({"id": ids, "break_probability": p})
sub = sub.set_index("id").reindex(table["id"].drop_duplicates()).reset_index()
assert sub.break_probability.notna().all() and sub.break_probability.between(0, 1).all()
sub.to_csv(a.out, index=False, float_format="%.10f")
log(f"wrote {a.out} {sub.shape}")
