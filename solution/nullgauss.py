"""Gaussianize each series with the no-break network, then run the classical test battery.

For every point t (after the first LAGS) the fold network gives its predictive CDF under the
no-break model with the whole series as the regime context (pooled = H0). The normal score
z_t = Phi^-1(F(x_t | past, pooled regime)) is ~iid N(0,1) when there is no break: dependence,
volatility clustering, heavy tails and bimodal shapes are all removed at once. The battery
that made whitening our biggest win is then run on z and on |z|:
  segment descriptors + two-sample tests (_transform_block, _two_sample), persistence along
  the post segment (_persistence) and boundary-vs-other-splits scans (_whitened_scan).
Cross-fitted like the network features (fold k series scored by the model without fold k).

    python solution/nullgauss.py --work solution/work_nn2m16l64XL --out solution/work_ng16 --mix 16 --lags 64
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numba import njit, prange
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F  # noqa: E402
import nullnet as N  # noqa: E402
import nullnet2 as M  # noqa: E402

PER = 30 + 11 + 23 + 4 * F.N_WSCAN          # transform block + two-sample + persistence + scan


@torch.no_grad()
def normal_scores(net, pairs, dev, batch=256, cond="pooled"):
    """-> list of (z array, boundary index in z)."""
    out = [None] * len(pairs)
    order = np.argsort([len(p) + len(q) for p, q in pairs])
    L = N.LAGS
    for s in range(0, len(pairs), batch):
        idx = order[s:s + batch]
        xs = [np.concatenate(pairs[i]) for i in idx]
        T = max(len(x) for x in xs) - L
        ctx, lag, tgt = [], [], []
        for j, x in enumerate(xs):
            nP = len(pairs[idx[j]][0])
            base = x if cond == "pooled" else x[:nP]          # pre-only: regime and scale from the pre segment
            mu, sc = N.med_scale(base)
            z = ((x - mu) / sc).astype(np.float32)
            ctx.append(z if cond == "pooled" else z[:nP])
            lg = np.zeros((T, L), np.float32)
            w = np.lib.stride_tricks.sliding_window_view(z[:-1], L)[:, ::-1]      # lags for t = L..n-1
            lg[:len(w)] = w
            yy = np.zeros(T, np.float32)
            yy[:len(z) - L] = z[L:]
            lag.append(lg); tgt.append(yy)
        c, cm = N.pad_left(ctx, N.CTX)
        h = net.encode(torch.from_numpy(c).to(dev), torch.from_numpy(cm).to(dev))
        lg = torch.from_numpy(np.stack(lag)).to(dev)
        hh = h.unsqueeze(1).expand(-1, lg.shape[1], -1)
        o = net.dec(torch.cat((lg, lg.abs(), hh), 2)).cpu().numpy()
        yy = np.stack(tgt)
        if M.MIX > 1:
            o = o.reshape(*o.shape[:2], M.MIX, 4)
            w = np.exp(o[..., 3] - o[..., 3].max(-1, keepdims=True))
            w /= w.sum(-1, keepdims=True)
            r = (yy[..., None] - o[..., 0]) / np.exp(np.clip(o[..., 1], -6, 4))
            nu = 2.2 + np.log1p(np.exp(o[..., 2]))
            u = (w * stats.t.cdf(r, nu)).sum(-1)
        else:
            r = (yy - o[..., 0]) / np.exp(np.clip(o[..., 1], -6, 4))
            u = stats.t.cdf(r, 2.2 + np.log1p(np.exp(o[..., 2])))
        zz = stats.norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
        for j, i in enumerate(idx):
            n = len(xs[j]) - L
            out[i] = (zz[j, :n].astype(np.float64), len(pairs[i][0]) - L)
    return out


@njit(cache=True)
def _battery(z, b, out, k):
    pre, post = z[:b].copy(), z[b:].copy()
    k = F._transform_block(pre, post, out, k)
    k = F._two_sample(pre, post, out, k)
    k = F._persistence(pre, post, out, k)
    k = F._whitened_scan(z, b, out, k)
    return k


@njit(parallel=True, cache=True)
def _all(values, starts, lens, bounds):
    m = starts.shape[0]
    out = np.zeros((m, 2 * PER))
    for i in prange(m):
        z = values[starts[i]:starts[i] + lens[i]].copy()
        b = bounds[i]
        k = _battery(z, b, out[i], 0)
        a = np.abs(z)
        a = (a - a[:b].mean()) / (a[:b].std() + 1e-9)
        _battery(a, b, out[i], k)
    return out


PER_X = 4 + 4 + 4 + 5 + 5 + 3 + 16


@njit(cache=True)
def _extra(z, b, out, k):
    pre, post = z[:b].copy(), z[b:].copy()
    k = F._moment_block(pre, post, out, k)
    k = F._moment_scan(pre, post, out, k)
    k = F._tail_block(pre, post, out, k)
    k = F._robust_chow(pre, post, 1, out, k)
    k = F._robust_chow(pre, post, 3, out, k)
    k = F._trend_chow(pre, post, out, k)
    k = F._chow_family(pre, post, out, k)
    return k


@njit(parallel=True, cache=True)
def _all_extra(values, starts, lens, bounds):
    m = starts.shape[0]
    out = np.zeros((m, PER_X))
    for i in prange(m):
        _extra(values[starts[i]:starts[i] + lens[i]].copy(), bounds[i], out[i], 0)
    return out


def battery_extra(scores):
    lens = np.array([len(z) for z, _ in scores], np.int64)
    bounds = np.array([max(32, min(b, len(z) - 32)) for z, b in scores], np.int64)
    starts = np.concatenate(([0], np.cumsum(lens)[:-1])).astype(np.int64)
    return _all_extra(np.concatenate([z for z, _ in scores]), starts, lens, bounds)


def battery(scores):
    lens = np.array([len(z) for z, _ in scores], np.int64)
    bounds = np.array([max(16, min(b, len(z) - 16)) for z, b in scores], np.int64)
    starts = np.concatenate(([0], np.cumsum(lens)[:-1])).astype(np.int64)
    return _all(np.concatenate([z for z, _ in scores]), starts, lens, bounds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mix", type=int, default=16)
    ap.add_argument("--lags", type=int, default=64)
    ap.add_argument("--cond", default="pooled", choices=["pooled", "pre"])
    ap.add_argument("--extra", action="store_true", help="only the extra tests (moments, tails, Chow) on z")
    a = ap.parse_args()
    M.MIX, N.LAGS = a.mix, a.lags
    a.out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    ids, views = N.view_pairs(pd.read_parquet("train_series.parquet"))
    nf = a.work / "netfolds.csv"
    folds = pd.read_csv(nf if nf.exists() else "solution/work/folds.csv").set_index("id").fold.reindex(ids).to_numpy()
    width = PER_X if a.extra else 2 * PER
    feats = {v: np.zeros((len(ids), width)) for v in views}
    for k in sorted(np.unique(folds)):
        net = M.make_net().to(dev)
        net.load_state_dict(torch.load(a.work / f"nullnet_fold{k}.pt", map_location=dev))
        net.eval()
        rows = np.where(folds == k)[0]
        for v, pairs in views.items():
            sc = normal_scores(net, [pairs[i] for i in rows], dev, cond=a.cond)
            feats[v][rows] = battery_extra(sc) if a.extra else battery(sc)
        print(f"fold {k} done", flush=True)
    cols = ([f"ngx{a.cond}{a.lags}_{j}" for j in range(PER_X)] if a.extra
            else [f"ng{a.cond}{a.lags}_{s}_{j}" for s in ("z", "a") for j in range(PER)])
    for v, f in feats.items():
        pd.DataFrame(f, index=pd.Index(ids, name="id"), columns=cols).to_parquet(a.out / f"nn_features_{v}.parquet")
    print("done", a.out)


if __name__ == "__main__":
    main()
