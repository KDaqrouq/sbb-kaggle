"""Train the KU-SBB model: features -> grouped CV -> final checkpoint.

    python solution/train.py --data . --checkpoint solution/checkpoint

Each training series is one row per feature view, and folds are assigned by series
id, so every view of a series sits in the same fold (whole-series grouping). The
fold assignment is saved and reused by every experiment. Every learned step
(early stopping, model fitting) uses training-fold series only.
"""
import argparse
import json
import re
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import data
import features
import keepawake
import models

FOLD_SEED = 2026
N_FOLDS = 5


def make_folds(labels, path):
    if path.exists():
        folds = pd.read_csv(path, dtype={"id": str}).set_index("id").fold
        if set(folds.index) == set(labels.index):
            return folds.reindex(labels.index)
    ordered = labels.sort_index()
    folds = pd.Series(-1, index=ordered.index, name="fold")
    splitter = StratifiedKFold(N_FOLDS, shuffle=True, random_state=FOLD_SEED)
    for k, (_, val) in enumerate(splitter.split(ordered.index, ordered.to_numpy())):
        folds.iloc[val] = k
    folds.rename_axis("id").reset_index().to_csv(path, index=False, lineterminator="\n")
    return folds.reindex(labels.index)


def train_features(data_dir, work, cached_only=False):
    """Feature views of the training series, cached as work/train_features_<view>.parquet.
    cached_only: use the cache as it is even if features.py has changed since (screens)."""
    names = list(features.VIEWS) + [f"crop{r}_{v}" for r in range(features.CROP_ROUNDS)
                                    for v in features.VIEWS] + (["decim_orig", "decim_rev"] if features.USE_DECIM_VIEWS else [])
    paths = {v: work / f"train_features_{v}.parquet" for v in names}
    if all(p.exists() for p in paths.values()):
        views = {v: pd.read_parquet(p) for v, p in paths.items()}
        if cached_only or all(list(f.columns) == features.FEATURE_NAMES for f in views.values()):
            return views
    table = data.load_series(Path(data_dir) / "train_series.parquet")
    views = features.compute_views(table)
    views.update(features.compute_crop_views(table))
    if features.USE_DECIM_VIEWS:
        views.update(features.compute_decim_views(table))
    for v, frame in views.items():
        frame.to_parquet(paths[v])
    return views


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("."))
    parser.add_argument("--checkpoint", type=Path, default=Path("solution/checkpoint"))
    parser.add_argument("--work", type=Path, default=Path("solution/work"))
    parser.add_argument("--models", default=",".join(models.DEFAULT_MODELS))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--params", type=Path, default=Path("solution/params.json"),
                        help="tuned hyperparameters from tune.py (defaults used if missing)")
    parser.add_argument("--skip-final", action="store_true", help="CV only, no checkpoint")
    parser.add_argument("--exclude", default="", help="regex of feature names to leave out (ablations)")
    parser.add_argument("--cached", action="store_true", help="use the feature cache in --work as it is")
    parser.add_argument("--monotone", default="", help="regex: LightGBM features constrained increasing")
    parser.add_argument("--train-views", default="", help="comma list of extra training-only view names in --work")
    parser.add_argument("--add", type=Path, action="append", default=[],
                        help="dir with nn_features_<view>.parquet to append (screens)")
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    keepawake.hold()
    started = time.time()

    data.verify_hashes(args.data)
    views = train_features(args.data, args.work, cached_only=args.cached)
    if args.exclude:
        keep = [c for c in views["orig"].columns if not re.match(args.exclude, c)]
        views = {v: frame[keep] for v, frame in views.items()}
        print(f"excluded {len(features.FEATURE_NAMES) - len(keep)} features matching {args.exclude!r}", flush=True)
    for name in filter(None, args.train_views.split(",")):
        views[name] = pd.read_parquet(args.work / f"train_features_{name}.parquet").reindex(views["orig"].index)[views["orig"].columns]
    for extra in args.add:
        views = {v: pd.concat([f, pd.read_parquet(extra / f"nn_features_{v}.parquet").reindex(f.index)], axis=1)
                 for v, f in views.items()}
    if __import__("os").environ.get("EXC2"):
        _x2 = __import__("os").environ["EXC2"]
        _k2 = [c for c in views["orig"].columns if not re.search(_x2, c)]
        views = {v: f[_k2] for v, f in views.items()}
        print("after EXC2", len(_k2), flush=True)
    used = list(views["orig"].columns)
    X = views["orig"]
    print(f"features: {len(views)} views x {X.shape} in {time.time() - started:.0f}s", flush=True)
    y = data.load_labels(Path(args.data) / "train_labels.csv", X.index.tolist()).reindex(X.index)
    folds = make_folds(y, args.work / "folds.csv")

    tuned = models.load_params(args.params)
    if args.monotone:
        mono = [1 if re.search(args.monotone, c) else 0 for c in used]
        tuned.setdefault("lgb", {})
        tuned["lgb"] = dict(tuned["lgb"], monotone_constraints=mono, monotone_constraints_method="advanced")
        print(f"monotone increasing: {sum(mono)} features", flush=True)
    print(f"tuned params: {sorted(tuned) or 'none, using defaults'}", flush=True)
    names = args.models.split(",")
    oof = pd.DataFrame(index=X.index)
    report = {"n_features": X.shape[1], "models": {}}
    # Each model's CV result is saved as soon as it finishes, so an interrupted run
    # resumes where it stopped instead of starting over.
    parts = args.work / "cv_parts"
    parts.mkdir(exist_ok=True)
    settings = json.dumps(tuned, sort_keys=True) + json.dumps(used)
    # the key covers the exact feature and view names, so different screens never share a result
    key = settings + "|" + ",".join(used) + "|" + ",".join(views) + "|" + "|".join(str(d) for d in args.add)
    tag = f"{len(used)}f_{len(views)}v_{zlib.crc32(key.encode()):08x}"
    for name in names:
        t0 = time.time()
        part = parts / f"{name}_{tag}.json"
        if part.exists():
            saved = json.loads(part.read_text(encoding="utf-8"))
            pred, iters = np.array(saved["pred"]), saved["iterations"]
            print(f"{name}: resumed from {part.name}", flush=True)
        else:
            pred, iters = models.cross_validate(name, views, y, folds, args.threads,
                                                params=tuned.get(models.kind_of(name)))
            part.write_text(json.dumps({"pred": pred.tolist(), "iterations": iters}), encoding="utf-8")
        oof[name] = pred
        fold_auc = [roc_auc_score(y[folds == k], pred[folds == k]) for k in range(N_FOLDS)]
        report["models"][name] = {"oof_auc": roc_auc_score(y, pred), "fold_auc": fold_auc,
                                  "fold_std": float(np.std(fold_auc)), "iterations": iters,
                                  "seconds": time.time() - t0}
        print(f"{name}: OOF AUC {report['models'][name]['oof_auc']:.4f} "
              f"folds {np.round(fold_auc, 4).tolist()} ({time.time() - t0:.0f}s)", flush=True)
    blend = models.blend(oof[names].to_numpy())
    report["blend_oof_auc"] = roc_auc_score(y, blend)
    report["blend_fold_auc"] = [roc_auc_score(y[folds == k], blend[folds == k]) for k in range(N_FOLDS)]
    print(f"blend: OOF AUC {report['blend_oof_auc']:.4f}", flush=True)
    oof["blend"] = blend
    oof["label"] = y
    oof["fold"] = folds
    oof.to_csv(args.work / "oof.csv", lineterminator="\n")

    if not args.skip_final:
        args.checkpoint.mkdir(parents=True, exist_ok=True)
        for name in names:
            done = args.checkpoint / (f"{name}.txt" if models.kind_of(name) == "lgb" else f"{name}.cbm")
            if done.exists() and (parts / f"{name}_{tag}.final").exists():
                print(f"{name}: final model already saved", flush=True)
                continue
            models.fit_final(name, views, y, report["models"][name]["iterations"], args.threads,
                             args.checkpoint, params=tuned.get(models.kind_of(name)))
            (parts / f"{name}_{tag}.final").write_text("ok", encoding="utf-8")
        meta = {"models": names, "feature_names": used, "views": list(models.PREDICT_VIEWS),
                "crop_views": {"rounds": features.CROP_ROUNDS, "min_fraction": features.CROP_MIN_FRACTION},
                "params": tuned,
                "cv": report, "train_seconds": time.time() - started}
        (args.checkpoint / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    report["total_seconds"] = time.time() - started
    (args.work / "cv_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"done in {report['total_seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
