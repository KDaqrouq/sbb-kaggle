"""Seasonality-aware features.

About 15-20% of the series carry a clean periodic component (periods ~12-31 points) that explains
most of their variance. Two-sample tests on such a series are dominated by the cycle, and a change
in the cycle itself (phase, amplitude, frequency) leaves the marginal distribution, the ACF and the
spectrum shape unchanged. Two feature blocks:

  stats(P, Q)      seasonal change tests for one (pre, post) pair: harmonic regression at the
                   series' dominant frequency (2 harmonics), Chow-type F test for a change of the
                   harmonic coefficients, phase and amplitude change, per-side frequency, and
                   extrapolation error of the pre-fit on post (and post-fit on pre).
  deseason(table)  the table with the pooled harmonic component removed from every series; the
                   whole handcrafted battery is rerun on it (columns ds_*), like the detrended and
                   whitened batteries.

Train features (CPU):  python solution/seasonal.py --out solution/work_seas --ds solution/work_ds
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

NAMES = ["seas_r2", "seas_narrow", "seas_period", "seas_f", "seas_amp", "seas_amp_lr", "seas_dph", "seas_dph_z",
         "seas_dfreq", "seas_xpost", "seas_xpre", "seas_r2_pre", "seas_r2_post", "seas_h2"]
MINC = 4  # the cycle must repeat at least this many times in the series
GATE_NARROW, GATE_PERIOD = 20.0, 80.0   # "clearly seasonal": periodogram peak / neighbours > 20, period < 80


def gate(stats_orig):
    """Boolean mask of clearly seasonal series from the orig-view seasonal stats (rows of stats())."""
    return (np.expm1(stats_orig[:, NAMES.index("seas_narrow")]) > GATE_NARROW) & (stats_orig[:, NAMES.index("seas_period")] < GATE_PERIOD)


def _harm(fr, t):
    w = 2 * np.pi * fr * t
    return np.column_stack((np.cos(w), np.sin(w), np.cos(2 * w), np.sin(2 * w)))


def _rss(X, y):
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ beta
    return beta, float(r @ r)


def fit_freq(x):
    """Dominant frequency (cycles per point) of the linearly detrended series, refined on a grid;
    narrowness of its periodogram peak; share of variance explained by 2 harmonics."""
    n = len(x)
    t = np.arange(n, dtype=np.float64)
    tc = t - t.mean()
    base = np.column_stack((np.ones(n), tc))
    v = x - base @ np.linalg.lstsq(base, x, rcond=None)[0]
    tot = float(v @ v) + 1e-12
    f = np.abs(np.fft.rfft(v * np.hanning(n))) ** 2
    if len(f) <= MINC + 2:
        return 0.25, 0.0, 0.0
    k = MINC + int(np.argmax(f[MINC:]))
    lo, hi = max(1, k - 12), min(len(f), k + 13)
    nb = np.concatenate((f[lo:max(lo, k - 3)], f[k + 4:hi]))
    narrow = float(f[k] / (np.median(nb) + 1e-12)) if len(nb) else 0.0
    best_r2, best_fr = -1.0, k / n
    for fr in np.linspace((k - 1) / n, (k + 1) / n, 21):
        _, rss = _rss(np.column_stack((base, _harm(fr, t))), x)
        r2 = 1.0 - rss / tot
        if r2 > best_r2:
            best_r2, best_fr = r2, fr
    return float(best_fr), narrow, float(best_r2)


def stats(P, Q):
    x = np.concatenate((P, Q)).astype(np.float64)
    n, b = len(x), len(P)
    fr, narrow, r2 = fit_freq(x)
    t = np.arange(n, dtype=np.float64)
    tc = t - t.mean()
    post = (t >= b).astype(np.float64)
    H = _harm(fr, t)
    D0 = np.column_stack((np.ones(n), tc, post))
    _, rss0 = _rss(np.column_stack((D0, H)), x)
    beta1, rss1 = _rss(np.column_stack((D0, H * (1 - post)[:, None], H * post[:, None])), x)
    F = ((rss0 - rss1) / 4.0) / (rss1 / max(n - 11, 1) + 1e-12)
    ca, sa, cc, sc = beta1[3], beta1[4], beta1[7], beta1[8]
    amp_a, amp_c = np.hypot(ca, sa), np.hypot(cc, sc)
    dph = abs(np.angle(np.exp(1j * (np.arctan2(sa, ca) - np.arctan2(sc, cc)))))
    sd_res = np.sqrt(rss1 / n) + 1e-12
    amp = 0.5 * (amp_a + amp_c)
    h2 = 0.5 * (np.hypot(beta1[5], beta1[6]) + np.hypot(beta1[9], beta1[10])) / sd_res

    def side(v, t_):
        Xs = np.column_stack((np.ones(len(v)), t_ - t_.mean(), _harm(fr, t_)))
        beta, rss = _rss(Xs, v)
        vv = v - v.mean()
        return beta, rss / len(v), 1.0 - rss / (float(vv @ vv) + 1e-12), t_.mean()

    ba, va, r2a, ma = side(x[:b], t[:b])
    bc, vc, r2c, mc = side(x[b:], t[b:])
    # extrapolate each side's harmonic fit to the other side (level and slope refitted there)
    def xpred(beta, v, t_):
        res = v - _harm(fr, t_) @ beta[2:]
        Xl = np.column_stack((np.ones(len(v)), t_ - t_.mean()))
        _, rss = _rss(Xl, res)
        return rss / len(v)

    xpost = np.log((xpred(ba, x[b:], t[b:]) + 1e-12) / (vc + 1e-12))
    xpre = np.log((xpred(bc, x[:b], t[:b]) + 1e-12) / (va + 1e-12))
    fa = fit_freq(x[:b])[0] if b >= 48 else fr
    fc = fit_freq(x[b:])[0] if n - b >= 48 else fr
    return np.array([r2, np.log1p(narrow), 1.0 / max(fr, 1e-6), np.log1p(max(F, 0.0)), amp / sd_res,
                     np.log((amp_c + 1e-9) / (amp_a + 1e-9)), dph, dph * amp / sd_res * np.sqrt(min(b, n - b) / 2.0),
                     abs(fa - fc) * min(b, n - b), xpost, xpre, r2a, r2c, h2])


def deseason_values(x, b):
    """x minus its pooled harmonic component (level, slope and a level step at b are kept)."""
    x = x.astype(np.float64)
    n = len(x)
    fr, _, _ = fit_freq(x)
    t = np.arange(n, dtype=np.float64)
    post = (t >= b).astype(np.float64)
    H = _harm(fr, t)
    X = np.column_stack((np.ones(n), t - t.mean(), post, H))
    beta = np.linalg.lstsq(X, x, rcond=None)[0]
    return x - H @ beta[3:]


def detrend_values(x, b, harmonics=True):
    """x minus its pooled linear trend (and, with harmonics=True, its pooled harmonic component);
    the level and a level step at b are kept. ~18% of the series carry a linear trend (r2 >= 0.02)
    and are the weakest group (AUC 0.70-0.79 against 0.88 for the others): with the trend left in,
    every pre/post contrast is dominated by it."""
    x = x.astype(np.float64)
    n = len(x)
    t = np.arange(n, dtype=np.float64)
    tc = t - t.mean()
    post = (t >= b).astype(np.float64)
    if harmonics:
        fr, _, _ = fit_freq(x)
        H = _harm(fr, t)
        X = np.column_stack((np.ones(n), post, tc, H))
    else:
        X = np.column_stack((np.ones(n), post, tc))
    beta = np.linalg.lstsq(X, x, rcond=None)[0]
    return x - X[:, 2:] @ beta[2:]


def detrend(table, harmonics=True):
    """Same table with the pooled linear trend (and cycle) removed from every series."""
    out = table.copy()
    vals = out["value"].to_numpy().copy()
    per = out["period"].to_numpy()
    ids = out["id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    ends = np.r_[starts[1:], len(ids)]
    for s, e in zip(starts, ends):
        vals[s:e] = detrend_values(vals[s:e], int((per[s:e] == 0).sum()), harmonics)
    out["value"] = vals
    return out


TRIM_CRIT, TRIM_MIN = 1.63, 64   # sup-CUSUM threshold (1% Kolmogorov point), least points kept per side


def _last_change(v):
    """Index of the strongest mean or scale change inside v by CUSUM (None if not significant)."""
    n = len(v)
    if n < 2 * TRIM_MIN:
        return None
    med = np.median(v)
    best, where = 0.0, None
    for u in (v, np.abs(v - med)):
        m = np.median(u)
        sd = 1.4826 * np.median(np.abs(u - m)) + 1e-12
        cs = np.cumsum(u - u.mean()) / (sd * np.sqrt(n))
        k = int(np.argmax(np.abs(cs[:-1])))
        if abs(cs[k]) > best:
            best, where = abs(cs[k]), k + 1
    return where if best > TRIM_CRIT else None


def trim_values(x, b):
    """(pre, post) cut down to the stable stretches next to the boundary: points of the pre segment
    before its last detected in-segment change, and points of the post segment after its first one,
    are dropped (up to two cuts per side, at least TRIM_MIN points kept). Series whose segments
    are themselves unstable are a weak group; the boundary comparison should use the regime that
    is in force at the boundary."""
    pre, post = x[:b].astype(np.float64), x[b:].astype(np.float64)
    for _ in range(2):
        j = _last_change(pre)
        if j is None or len(pre) - j < TRIM_MIN:
            break
        pre = pre[j:]
    for _ in range(2):
        r = post[::-1]
        j = _last_change(r)
        if j is None or len(r) - j < TRIM_MIN:
            break
        post = r[j:][::-1]
    return pre, post


def trim(table):
    """Table of the trimmed series (see trim_values), plus per-id kept fractions."""
    ids = table["id"].to_numpy()
    vals = table["value"].to_numpy()
    per = table["period"].to_numpy()
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    ends = np.r_[starts[1:], len(ids)]
    I, V, P, T, keep = [], [], [], [], []
    for s, e in zip(starts, ends):
        b = int((per[s:e] == 0).sum())
        pre, post = trim_values(vals[s:e], b)
        n = len(pre) + len(post)
        I.append(np.repeat(ids[s], n)); V.append(np.concatenate((pre, post)))
        P.append(np.r_[np.zeros(len(pre), np.int64), np.ones(len(post), np.int64)]); T.append(np.arange(n))
        keep.append((ids[s], len(pre) / b, len(post) / (e - s - b)))
    out = pd.DataFrame({"id": np.concatenate(I), "time": np.concatenate(T), "value": np.concatenate(V), "period": np.concatenate(P)})
    return out, pd.DataFrame(keep, columns=["id", "tr_keep_pre", "tr_keep_post"]).set_index("id")


def deseason(table):
    """Same table (id, time, value, period) with deseasonalized values."""
    out = table.copy()
    vals = out["value"].to_numpy().copy()
    per = out["period"].to_numpy()
    ids = out["id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    ends = np.r_[starts[1:], len(ids)]
    for s, e in zip(starts, ends):
        b = int((per[s:e] == 0).sum())
        vals[s:e] = deseason_values(vals[s:e], b)
    out["value"] = vals
    return out


def difference(table):
    """First differences of every series as a table (id, time, value, period): the n-1 increments,
    the increment across the boundary counted as the first post point. For random-walk-like series
    the increments are the stationary object, and the whole battery is rerun on them (dd_*)."""
    ids = table["id"].to_numpy()
    vals = table["value"].to_numpy()
    per = table["period"].to_numpy()
    keep = np.r_[False, ids[1:] == ids[:-1]]
    d = np.r_[0.0, np.diff(vals)]
    out = pd.DataFrame({"id": ids[keep], "value": d[keep], "period": per[keep]})
    out["time"] = out.groupby("id", sort=False).cumcount()
    return out[["id", "time", "value", "period"]]


def all_stats(pairs):
    return np.array([stats(P, Q) for P, Q in pairs])


def main():
    from joblib import Parallel, delayed
    import features as F
    import nullnet as N
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--ds", type=Path, default=None)
    ap.add_argument("--jobs", type=int, default=28)
    ap.add_argument("--dd", type=Path, default=None, help="only write the battery on first differences here")
    ap.add_argument("--tr", type=Path, default=None, help="only write the battery on the trimmed (stable-regime) series here")
    ap.add_argument("--dt", type=Path, default=None, help="only write the battery on the detrended + deseasonalized series here")
    a = ap.parse_args()
    table = pd.read_parquet("train_series.parquet")
    if a.tr is not None:
        from joblib import Parallel as _P, delayed as _d
        a.tr.mkdir(parents=True, exist_ok=True)
        ids0 = table["id"].drop_duplicates().to_numpy()
        pos = pd.Series(np.arange(len(ids0)), index=ids0).reindex(table["id"].to_numpy()).to_numpy()
        parts = np.array_split(np.arange(len(ids0)), a.jobs * 4)
        res = _P(n_jobs=a.jobs)(_d(trim)(table[np.isin(pos, q)]) for q in parts)
        tt = pd.concat([r[0] for r in res], ignore_index=True)
        keep = pd.concat([r[1] for r in res]).reindex(ids0)
        print("trimmed table", tt.shape, "kept pre/post mean", keep.mean().round(3).tolist(), "trimmed share", float(((keep < 1).any(axis=1)).mean()), flush=True)
        dv = F.compute_views(tt)
        dv.update(F.compute_crop_views(tt))
        swap = {"rev", "revflip"}
        for v, fr in dv.items():
            fr.columns = ["tr_" + c for c in fr.columns]
            fr = fr.reindex(pd.Index(ids0, name="id"))
            k = keep.copy()
            if v.split("_")[-1] in swap:
                k.columns = ["tr_keep_post", "tr_keep_pre"]
            fr[["tr_keep_pre", "tr_keep_post"]] = k[["tr_keep_pre", "tr_keep_post"]].to_numpy()
            fr.to_parquet(a.tr / f"nn_features_{v}.parquet")
        print("done tr", flush=True)
        return
    if a.dt is not None:
        from joblib import Parallel as _P, delayed as _d
        a.dt.mkdir(parents=True, exist_ok=True)
        ids0 = table["id"].drop_duplicates().to_numpy()
        pos = pd.Series(np.arange(len(ids0)), index=ids0).reindex(table["id"].to_numpy()).to_numpy()
        parts = np.array_split(np.arange(len(ids0)), a.jobs * 4)
        td = pd.concat(_P(n_jobs=a.jobs)(_d(detrend)(table[np.isin(pos, q)]) for q in parts)).sort_index()
        print("detrended table", td.shape, flush=True)
        dv = F.compute_views(td)
        dv.update(F.compute_crop_views(td))
        for v, fr in dv.items():
            fr.columns = ["dt_" + c for c in fr.columns]
            fr.reindex(pd.Index(ids0, name="id")).to_parquet(a.dt / f"nn_features_{v}.parquet")
        print("done dt", flush=True)
        return
    if a.dd is not None:
        a.dd.mkdir(parents=True, exist_ok=True)
        td = difference(table)
        ids0 = table["id"].drop_duplicates().to_numpy()
        dv = F.compute_views(td)
        dv.update(F.compute_crop_views(td))
        for v, fr in dv.items():
            fr.columns = ["dd_" + c for c in fr.columns]
            fr.reindex(pd.Index(ids0, name="id")).to_parquet(a.dd / f"nn_features_{v}.parquet")
        print("done dd", flush=True)
        return
    a.out.mkdir(parents=True, exist_ok=True)
    a.ds.mkdir(parents=True, exist_ok=True)
    ids, views = N.view_pairs(table)
    idx = pd.Index(ids, name="id")
    for v, pairs in views.items():
        ch = [pairs[i:i + 500] for i in range(0, len(pairs), 500)]
        r = np.vstack(Parallel(n_jobs=a.jobs)(delayed(all_stats)(c) for c in ch))
        pd.DataFrame(r, index=idx, columns=NAMES).to_parquet(a.out / f"nn_features_{v}.parquet")
        print("seasonal stats", v, flush=True)
    parts = np.array_split(np.arange(len(ids)), a.jobs * 4)
    gid = table["id"].to_numpy()
    pos = pd.Series(np.arange(len(ids)), index=ids).reindex(gid).to_numpy()
    chunks = [table[np.isin(pos, p)] for p in parts]
    ds = pd.concat(Parallel(n_jobs=a.jobs)(delayed(deseason)(c) for c in chunks)).sort_index()
    print("deseasonalized table", ds.shape, flush=True)
    dv = F.compute_views(ds)
    dv.update(F.compute_crop_views(ds))
    for v, fr in dv.items():
        fr.columns = ["ds_" + c for c in fr.columns]
        fr.reindex(idx).to_parquet(a.ds / f"nn_features_{v}.parquet")
    print("done", flush=True)


if __name__ == "__main__":
    main()
