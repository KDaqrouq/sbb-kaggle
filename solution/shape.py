"""Shape-only two-sample tests: is the post distribution a different SHAPE from the pre distribution
once each side is standardised by its own location and scale?

Hidden breaks (no level/trend/AR/scale change) show almost no variance change (AUC 0.51) but a
kurtosis change (0.555) and a KS distance between separately standardised sides (0.552): many of
them are changes of distribution shape at fixed mean and variance. The existing batteries
standardise both sides with the same location/scale, which mixes shape with level and scale.
Here each side is standardised on its own (mean/sd and median/MAD) and compared: KS, Cramer-von
Mises, energy distance, quantile profile, tail and centre mass, skewness, kurtosis and L-moment
ratios, on the raw values and on AR(8)-whitened innovations.

    python solution/shape.py --out solution/work_shape      # train features, 12 views
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F  # noqa: E402

QS = np.array([0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99])
TAILS = (1.0, 1.5, 2.0, 2.5)
_ONE = (["ks", "cvm", "energy", "ks_rob", "ks_abs", "dskew", "dskew_z", "dkurt", "dkurt_z", "dl3", "dl4", "centre25", "centre50"]
        + [f"tail{int(t * 10)}" for t in TAILS] + [f"tail{int(t * 10)}_z" for t in TAILS] + [f"q{int(q * 1000)}" for q in QS])
NAMES = [f"shp_{k}_{c}" for k in ("raw", "wh8") for c in _ONE]


def _lmom(s):
    """L-skewness and L-kurtosis of a sorted sample."""
    n = len(s)
    i = np.arange(n, dtype=np.float64)
    b0 = s.mean()
    b1 = (i * s).sum() / (n * (n - 1))
    b2 = (i * (i - 1) * s).sum() / (n * (n - 1) * (n - 2))
    b3 = (i * (i - 1) * (i - 2) * s).sum() / (n * (n - 1) * (n - 2) * (n - 3))
    l2 = 2 * b1 - b0
    l3 = 6 * b2 - 6 * b1 + b0
    l4 = 20 * b3 - 30 * b2 + 12 * b1 - b0
    return l3 / (l2 + 1e-12), l4 / (l2 + 1e-12)


def _ecdf_stats(a, c):
    """KS and Cramer-von Mises between two sorted samples."""
    allv = np.concatenate((a, c))
    fa = np.searchsorted(a, allv, "right") / len(a)
    fc = np.searchsorted(c, allv, "right") / len(c)
    d = fa - fc
    n, m = len(a), len(c)
    return np.abs(d).max(), (d * d).sum() * n * m / (n + m) ** 2


def _energy(a, c):
    """Energy distance between two sorted 1-D samples (O(n log n))."""
    def mean_abs_within(s):
        n = len(s)
        i = np.arange(n)
        return 2.0 * ((2 * i - n + 1) * s).sum() / (n * n)
    allv = np.concatenate((a, c))
    lab = np.concatenate((np.zeros(len(a)), np.ones(len(c))))
    o = np.argsort(allv, kind="mergesort")
    v, l = allv[o], lab[o]
    ca = np.cumsum(1 - l) / len(a)
    cc = np.cumsum(l) / len(c)
    cross = 2.0 * ((ca[:-1] - cc[:-1]) ** 2 * np.diff(v)).sum()   # = 2 E|X-Y| - E|X-X'| - E|Y-Y'|
    return cross


def _block(p, q):
    out = np.zeros(len(_ONE))
    if len(p) < 16 or len(q) < 16:
        return out
    a = np.sort((p - p.mean()) / (p.std() + 1e-12))
    c = np.sort((q - q.mean()) / (q.std() + 1e-12))
    n, m = len(a), len(c)
    ks, cvm = _ecdf_stats(a, c)
    mp, mq = np.median(p), np.median(q)
    ar = np.sort((p - mp) / (1.4826 * np.median(np.abs(p - mp)) + 1e-12))
    cr = np.sort((q - mq) / (1.4826 * np.median(np.abs(q - mq)) + 1e-12))
    ks_rob, _ = _ecdf_stats(ar, cr)
    ks_abs, _ = _ecdf_stats(np.sort(np.abs(a)), np.sort(np.abs(c)))
    sk_a, sk_c = (a ** 3).mean(), (c ** 3).mean()
    ku_a, ku_c = (a ** 4).mean() - 3.0, (c ** 4).mean() - 3.0
    l3a, l4a = _lmom(a)
    l3c, l4c = _lmom(c)
    k = 0
    for v in (ks * np.sqrt(n * m / (n + m)), cvm, _energy(a, c) * n * m / (n + m), ks_rob * np.sqrt(n * m / (n + m)),
              ks_abs * np.sqrt(n * m / (n + m)), sk_c - sk_a, (sk_c - sk_a) / np.sqrt(6.0 / n + 6.0 / m),
              ku_c - ku_a, (ku_c - ku_a) / np.sqrt(24.0 / n + 24.0 / m), l3c - l3a, l4c - l4a,
              (np.abs(c) < 0.25).mean() - (np.abs(a) < 0.25).mean(), (np.abs(c) < 0.5).mean() - (np.abs(a) < 0.5).mean()):
        out[k] = v
        k += 1
    ta = np.array([(np.abs(a) > t).mean() for t in TAILS])
    tc = np.array([(np.abs(c) > t).mean() for t in TAILS])
    out[k:k + 4] = tc - ta
    pool = (ta * n + tc * m) / (n + m)
    out[k + 4:k + 8] = (tc - ta) / np.sqrt(pool * (1 - pool) * (1.0 / n + 1.0 / m) + 1e-9)
    out[k + 8:] = np.quantile(c, QS) - np.quantile(a, QS)
    return out


def stats(P, Q):
    x = np.concatenate((P, Q)).astype(np.float64)
    b = len(P)
    pre, post = F._standardise(x[:b], x[b:])
    w8a, w8c = F._prewhiten(pre, post, 8)
    return np.concatenate((_block(pre, post), _block(np.asarray(w8a), np.asarray(w8c))))


def all_stats(pairs):
    return np.array([stats(P, Q) for P, Q in pairs])


def main():
    from joblib import Parallel, delayed
    import nullnet as N
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=24)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    ids, views = N.view_pairs(pd.read_parquet("train_series.parquet"))
    idx = pd.Index(ids, name="id")
    for v, pairs in views.items():
        ch = [pairs[i:i + 500] for i in range(0, len(pairs), 500)]
        r = np.vstack(Parallel(n_jobs=a.jobs)(delayed(all_stats)(c) for c in ch))
        pd.DataFrame(r, index=idx, columns=NAMES).to_parquet(a.out / f"nn_features_{v}.parquet")
        print("shape", v, flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
