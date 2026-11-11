"""Gradient-boosted tree models used by train.py, tune.py and predict.py.

Model names are "<kind>_s<seed>", e.g. lgb_s0 or cat_s1.

Augmentation: every series is seen in four label-preserving views (original,
sign-flipped, time-reversed, both; see features.compute_views), plus random crops
of those views for training only (features.compute_crop_views). Models train on
all views of the training series and predict by averaging the four standard views
(test-time augmentation). Views of one series always stay in the same fold.

Inside each outer training fold, 10% of the training *series* (all their views)
are held out for early stopping on their original view, so the outer validation
fold never influences fitting. The final model is refit on all training series
for 1.1x the mean early-stopped iteration count.

`params` arguments override the defaults below; tune.py searches them and saves
the winners to params.json, which train.py passes in.
"""
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_MODELS = ("lgb_s0", "lgb_s1", "lgb_s2")  # cat_s0 dropped: +0.0001 OOF for ~60% of training time

LGB_PARAMS = {"objective": "binary", "learning_rate": 0.02, "num_leaves": 31,
              "min_child_samples": 100, "feature_fraction": 0.3, "bagging_fraction": 0.8,
              "bagging_freq": 1, "lambda_l2": 20.0, "max_bin": 255, "verbose": -1,
              "deterministic": True, "force_col_wise": True}
CAT_PARAMS = {"depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 5.0, "loss_function": "Logloss",
              "eval_metric": "AUC", "verbose": False, "allow_writing_files": False}
MAX_ROUNDS = 6000
PATIENCE = 300


def _kind_seed(name):
    kind, seed = name.split("_s")
    return kind, int(seed)


def kind_of(name):
    return _kind_seed(name)[0]


def load_params(path):
    """Tuned overrides per model kind ({"lgb": {...}, "cat": {...}}), or {} if absent."""
    if path is None or not path.exists():
        return {}
    saved = json.loads(path.read_text(encoding="utf-8"))
    return {kind: saved[kind]["params"] for kind in ("lgb", "cat") if kind in saved}


def _subset(views, mask):
    return {v: frame[mask] for v, frame in views.items()}


def _stack(views, y):
    """All views of the given series as one training table."""
    X = pd.concat(list(views.values()), axis=0)
    return X, np.tile(np.asarray(y), len(views))


def _fit(kind, seed, views, y, threads, rounds=None, params=None):
    """Fit one model on all views. Without `rounds`, early-stop on 10% of the series."""
    params = params or {}
    if rounds is None:
        ids = y.index.to_numpy()
        fit_ids, stop_ids = train_test_split(ids, test_size=0.1, stratify=y.to_numpy(),
                                             random_state=seed)
        fit_mask = np.isin(ids, fit_ids)
        X_fit, y_fit = _stack(_subset(views, fit_mask), y[fit_mask])
        X_stop, y_stop = views["orig"][~fit_mask], y[~fit_mask]
    else:
        X_fit, y_fit = _stack(views, y)
    if kind == "lgb":
        import lightgbm as lgb
        lgb_params = dict(LGB_PARAMS, **params, seed=seed, num_threads=threads)
        train_set = lgb.Dataset(X_fit, y_fit)
        if rounds is None:
            stop_set = lgb.Dataset(X_stop, y_stop, reference=train_set)
            model = lgb.train(lgb_params, train_set, MAX_ROUNDS, valid_sets=[stop_set],
                              callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
            return model, model.best_iteration
        return lgb.train(lgb_params, train_set, rounds), rounds
    if kind == "cat":
        from catboost import CatBoostClassifier
        cat_params = dict(CAT_PARAMS, **params)
        if rounds is None:
            model = CatBoostClassifier(**cat_params, iterations=MAX_ROUNDS, random_seed=seed,
                                       thread_count=threads, od_type="Iter", od_wait=PATIENCE,
                                       use_best_model=True)
            model.fit(X_fit, y_fit, eval_set=(X_stop, y_stop))
            return model, model.get_best_iteration() + 1
        model = CatBoostClassifier(**cat_params, iterations=rounds, random_seed=seed,
                                   thread_count=threads)
        model.fit(X_fit, y_fit)
        return model, rounds
    raise ValueError("Unknown model kind: " + kind)


def _predict_one(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


PREDICT_VIEWS = tuple(__import__("os").environ.get("PRED", "orig,flip,rev,revflip").split(","))


def predict(model, views):
    """Test-time augmentation: mean prediction over the standard views (crop views are
    training-only)."""
    return np.mean([_predict_one(model, views[v]) for v in PREDICT_VIEWS if v in views], axis=0)


def cross_validate(name, views, y, folds, threads, params=None, on_fold=None):
    """Out-of-fold predictions. `on_fold(k, pred, done_mask)` may raise to stop early."""
    kind, seed = _kind_seed(name)
    n = len(y)
    pred = np.zeros(n)
    done = np.zeros(n, dtype=bool)
    iters = []
    for k in sorted(folds.unique()):
        train = (folds != k).to_numpy()
        model, best = _fit(kind, seed, _subset(views, train), y[train], threads, params=params)
        pred[~train] = predict(model, _subset(views, ~train))
        done |= ~train
        iters.append(int(best))
        if on_fold is not None:
            on_fold(k, pred, done)
    return pred, iters


def fit_final(name, views, y, iters, threads, directory, params=None):
    kind, seed = _kind_seed(name)
    rounds = max(50, int(round(1.1 * float(np.mean(iters)))))
    model, _ = _fit(kind, seed, views, y, threads, rounds=rounds, params=params)
    if kind == "lgb":
        model.save_model(str(directory / f"{name}.txt"))
    else:
        model.save_model(str(directory / f"{name}.cbm"))


def load(name, directory, threads):
    """Returns a function mapping feature views to test-time-averaged probabilities."""
    kind, _ = _kind_seed(name)
    if kind == "lgb":
        import lightgbm as lgb
        model = lgb.Booster(model_file=str(directory / f"{name}.txt"))
        one = lambda X: model.predict(X, num_threads=threads)
    else:
        from catboost import CatBoostClassifier
        model = CatBoostClassifier()
        model.load_model(str(directory / f"{name}.cbm"))
        one = lambda X: model.predict_proba(X, thread_count=threads)[:, 1]
    return lambda views: np.mean([one(frame) for frame in views.values()], axis=0)


def blend(columns):
    """Equal-weight mean of model probabilities (stays in [0, 1])."""
    return np.asarray(columns, dtype=float).mean(axis=1)
