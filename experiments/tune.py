"""Hyperparameter search for the KU-SBB tree models (Optuna TPE).

Selection run (writes the winners to params.json, which train.py then uses):
    python solution/tune.py --model lgb --trials 500 --threads 8
    python solution/tune.py --model cat --trials 200 --threads 8

Nested run (honest estimate of how well the *tuned* pipeline generalises):
    python solution/tune.py --model lgb --trials 150 --nested

Every trial is scored by 5-fold whole-series CV on the saved folds, using the
same early-stopping protocol as train.py. The selection run's best CV AUC is
optimistically biased because the folds also chose the parameters; the nested
run removes that bias by tuning on four folds and scoring the fifth, for each
fold in turn. Studies live in an SQLite file, so a search can be stopped and
resumed, and several worker processes (or machines sharing the file) can run
the same study in parallel.
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import sys
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "solution"))  # pipeline modules
import data  # noqa: E402
import keepawake
import models
import train

# multivariate TPE is flagged experimental in Optuna 4.1; it is stable for this use.
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)


def suggest(trial, kind):
    if kind == "lgb":
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 7, 255, log=True),
            "max_depth": trial.suggest_categorical("max_depth", [-1, 4, 6, 8, 10, 12]),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 400, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 0.9),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 50.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 100.0, log=True),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 2.0),
            "max_bin": trial.suggest_categorical("max_bin", [63, 127, 255, 511]),
            "extra_trees": trial.suggest_categorical("extra_trees", [False, True]),
            "path_smooth": trial.suggest_float("path_smooth", 0.0, 50.0),
        }
        return params
    if kind == "cat":
        return {
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 50.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 3.0),
            "border_count": trial.suggest_categorical("border_count", [64, 128, 254]),
            "rsm": trial.suggest_float("rsm", 0.1, 1.0),
        }
    raise ValueError("Unknown model kind: " + kind)


def objective(kind, views, y, folds, threads, seed):
    name = f"{kind}_s{seed}"

    def run(trial):
        params = suggest(trial, kind)

        def on_fold(k, pred, done):
            trial.report(roc_auc_score(y[done], pred[done]), step=int(k))
            if trial.should_prune():
                raise optuna.TrialPruned()

        pred, iters = models.cross_validate(name, views, y, folds, threads, params=params,
                                            on_fold=on_fold)
        trial.set_user_attr("iterations", iters)
        return roc_auc_score(y, pred)

    return run


def run_study(study_name, storage, kind, views, y, folds, args):
    # Each parallel worker gets its own sampler seed so they don't propose identical trials.
    sampler = optuna.samplers.TPESampler(seed=args.seed * 1000 + args.worker, multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=15, n_warmup_steps=1)
    # Workers started together race to create the SQLite schema; the losers retry.
    for attempt in range(20):
        try:
            study = optuna.create_study(study_name=study_name, storage=storage, direction="maximize",
                                        sampler=sampler, pruner=pruner, load_if_exists=True)
            break
        except Exception:
            if attempt == 19:
                raise
            time.sleep(1 + args.worker)
    # Warm start: a fresh study first tries the settings already saved in params.json.
    previous = models.load_params(args.params).get(kind)
    if previous and args.worker == 0 and not study.trials:
        study.enqueue_trial(previous)
    finished = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    if len(study.get_trials(states=finished)) < args.trials:
        # --trials is the study's total across all workers and restarts.
        cap = optuna.study.MaxTrialsCallback(args.trials, states=finished)
        study.optimize(objective(kind, views, y, folds, args.threads, args.seed), n_trials=args.trials,
                       timeout=args.timeout, gc_after_trial=True, callbacks=[cap])
    return study


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("."))
    parser.add_argument("--work", type=Path, default=Path("solution/work"))
    parser.add_argument("--model", choices=["lgb", "cat"], required=True)
    parser.add_argument("--trials", type=int, default=300, help="total trials per study")
    parser.add_argument("--timeout", type=float, default=None, help="seconds per study")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--worker", type=int, default=0, help="worker number when running in parallel")
    parser.add_argument("--nested", action="store_true")
    parser.add_argument("--storage", default=None, help="Optuna storage URL (default: SQLite in --work)")
    parser.add_argument("--params", type=Path, default=Path("solution/params.json"))
    args = parser.parse_args()
    keepawake.hold()
    args.work.mkdir(parents=True, exist_ok=True)
    storage = args.storage or f"sqlite:///{(args.work / 'optuna.db').as_posix()}"
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    started = time.time()

    data.verify_hashes(args.data)
    views = train.train_features(args.data, args.work)
    X = views["orig"]
    y = data.load_labels(args.data / "train_labels.csv", X.index.tolist()).reindex(X.index)
    folds = train.make_folds(y, args.work / "folds.csv")

    if not args.nested:
        study = run_study(f"{args.model}_aug_select", storage, args.model, views, y, folds, args)
        best = study.best_trial
        saved = json.loads(args.params.read_text(encoding="utf-8")) if args.params.exists() else {}
        saved[args.model] = {"params": best.params, "cv_auc": best.value,
                             "iterations": best.user_attrs.get("iterations"),
                             "trials": len(study.trials), "study": study.study_name}
        args.params.write_text(json.dumps(saved, indent=1), encoding="utf-8")
        study.trials_dataframe().to_csv(args.work / f"tune_{args.model}_trials.csv", index=False)
        print(f"{args.model}: best CV AUC {best.value:.4f} after {len(study.trials)} trials "
              f"({time.time() - started:.0f}s); saved to {args.params}")
        return

    # Nested: tune on four folds, score the untouched fifth, for each outer fold.
    name = f"{args.model}_s{args.seed}"
    oof = np.zeros(len(X))
    report = {"model": args.model, "outer": []}
    for k in sorted(folds.unique()):
        inner = (folds != k).to_numpy()
        study = run_study(f"{args.model}_aug_nested_outer{k}", storage, args.model, models._subset(views, inner),
                          y[inner], folds[inner], args)
        model, _ = models._fit(args.model, args.seed, models._subset(views, inner), y[inner], args.threads,
                               params=study.best_params)
        oof[~inner] = models.predict(model, models._subset(views, ~inner))
        auc = roc_auc_score(y[~inner], oof[~inner])
        report["outer"].append({"fold": int(k), "auc": auc, "inner_cv_auc": study.best_value,
                                "params": study.best_params})
        print(f"outer fold {k}: test AUC {auc:.4f} (inner CV {study.best_value:.4f})", flush=True)
    report["nested_oof_auc"] = roc_auc_score(y, oof)
    report["seconds"] = time.time() - started
    (args.work / f"nested_{args.model}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    pd.DataFrame({"id": X.index, f"{name}_nested": oof}).to_csv(
        args.work / f"nested_{args.model}_oof.csv", index=False)
    print(f"{args.model}: nested OOF AUC {report['nested_oof_auc']:.4f} ({report['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
