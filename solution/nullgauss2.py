"""Gaussianization v2: better normal scores, and the full feature pipeline on them.

--ens   Average the predictive CDFs of several no-break networks (an ensemble predictive
        distribution is better calibrated than any single network), then take normal
        scores; scores are aligned on the largest lag window. Then the nullgauss battery.
--full  Run the ENTIRE handcrafted feature pipeline (features._all_features: whitening,
        detrending, GARCH, tails, moments, scans, Chow tests ...) on the normal-score series,
        as if it were a raw series. Residual structure the network missed gets the same
        treatment raw series got.

    python solution/nullgauss2.py --out solution/work_nge --ens solution/work_nn2m16l64XL:16:64 solution/work_nn2m12l48L:12:48 solution/work_nn2m8l32XL:8:32
    python solution/nullgauss2.py --out solution/work_ngfull --ens solution/work_nn2m16l64XL:16:64 --full
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F  # noqa: E402
import nullgauss as G  # noqa: E402
import nullnet as N  # noqa: E402
import nullnet2 as M  # noqa: E402


def ensemble_scores(nets, pairs, dev, cond="pooled"):
    """nets: list of (net, mix, lags). -> list of (z, boundary) aligned on max lags."""
    Lmax = max(l for _, _, l in nets)
    us = []
    for net, mix, lags in nets:
        M.MIX, N.LAGS = mix, lags
        sc = G.normal_scores(net, pairs, dev, cond=cond)
        us.append([stats.norm.cdf(z[Lmax - lags:]) for z, _ in sc])
    out = []
    for i, (P, Q) in enumerate(pairs):
        u = np.mean([us[j][i] for j in range(len(nets))], axis=0)
        out.append((stats.norm.ppf(np.clip(u, 1e-6, 1 - 1e-6)), len(P) - Lmax))
    return out


def _run(chunk):
    lens = np.array([len(z) for z, _ in chunk], np.int64)
    bounds = np.array([max(64, min(b, len(z) - 64)) for z, b in chunk], np.int64)
    starts = np.concatenate(([0], np.cumsum(lens)[:-1])).astype(np.int64)
    return F._all_features(np.concatenate([z for z, _ in chunk]), starts, bounds, lens)


def full_pipeline(scores, size=500):
    """Chunked; a chunk that fails inside numba (e.g. a singular fit on a degenerate score
    series) is redone series by series, and a series that still fails gets zeros."""
    out = np.zeros((len(scores), F.N_FEATURES))
    rng = np.random.default_rng(0)
    for s in range(0, len(scores), size):
        chunk = [(z + 1e-7 * rng.standard_normal(len(z)), b) for z, b in scores[s:s + size]]
        try:
            out[s:s + len(chunk)] = _run(chunk)
        except Exception:
            for j, item in enumerate(chunk):
                try:
                    out[s + j] = _run([item])[0]
                except Exception:
                    pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ens", nargs="+", required=True, help="work_dir:mix:lags for each network")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--cond", default="pooled", choices=["pooled", "pre"])
    ap.add_argument("--transform", default="none", choices=["none", "detrend", "diff"])
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    specs = [(Path(s.split(":")[0]), int(s.split(":")[1]), int(s.split(":")[2])) for s in a.ens]
    ids, views = N.view_pairs(pd.read_parquet("train_series.parquet"))
    if a.transform != "none":
        views = {v: [M.transform(P, Q, a.transform) for P, Q in pairs] for v, pairs in views.items()}
    nf = specs[0][0] / "netfolds.csv"
    folds = pd.read_csv(nf if nf.exists() else "solution/work/folds.csv").set_index("id").fold.reindex(ids).to_numpy()
    width = F.N_FEATURES if a.full else 2 * G.PER
    feats = {v: np.zeros((len(ids), width), np.float32) for v in views}
    for k in sorted(np.unique(folds)):
        nets = []
        for work, mix, lags in specs:
            M.MIX, N.LAGS = mix, lags
            net = M.make_net().to(dev)
            net.load_state_dict(torch.load(work / f"nullnet_fold{k}.pt", map_location=dev))
            nets.append((net.eval(), mix, lags))
        rows = np.where(folds == k)[0]
        for v, pairs in views.items():
            sc = ensemble_scores(nets, [pairs[i] for i in rows], dev, cond=a.cond)
            feats[v][rows] = full_pipeline(sc) if a.full else G.battery(sc)
        print(f"fold {k} done", flush=True)
    prefix = a.prefix or ("ngfull_" if a.full else "nge_")
    cols = ([prefix + c for c in F.FEATURE_NAMES] if a.full
            else [f"{prefix}{s}_{j}" for s in ("z", "a") for j in range(G.PER)])
    for v, f in feats.items():
        pd.DataFrame(f, index=pd.Index(ids, name="id"), columns=cols).to_parquet(a.out / f"nn_features_{v}.parquet")
    print("done", a.out)


if __name__ == "__main__":
    main()
