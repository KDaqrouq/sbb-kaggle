"""Faster CPU version of nullnet2.features2 (mixture networks only), same outputs.

features2 scores 15 (context, target range) tasks per series, encoding a context of up to
1,024 points for each task. Many tasks share a context (P, Q or the pooled series x), and a
decoder pass over the whole series under one context contains every target range scored
under that context. Here each distinct context is encoded once and decoded once over all
points; the task values (mean log density, PIT statistics) are read off as slices.
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import special

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nullnet as N  # noqa: E402
import nullnet2 as M  # noqa: E402


@torch.no_grad()
def _decode(net, ctxs, fulls, ranges, cdfr, batch=256, dev="cpu"):
    """For each (context, full series): z-normalise by the context's median/MAD, encode the
    context, decode every point of the full series. Returns per item: (logp per point,
    cdf per point, log scale) for points 0..len(full)-1."""
    res = [None] * len(ctxs)
    order = np.argsort([r[1] - r[0] for r in ranges])
    L = N.LAGS
    for s0 in range(0, len(order), batch):
        idx = order[s0:s0 + batch]
        T = max(1, max(ranges[i][1] - ranges[i][0] for i in idx))
        cz, lag, tgt, lsc = [], [], [], []
        for i in idx:
            mu, sc = N.med_scale(ctxs[i])
            cz.append(((ctxs[i] - mu) / sc).astype(np.float32))
            z = ((fulls[i] - mu) / sc).astype(np.float32)
            zp = np.concatenate((np.zeros(L, np.float32), z))
            a, b = ranges[i]
            m = b - a
            win = np.lib.stride_tricks.sliding_window_view(zp, L)[a:b][:, ::-1]
            lg = np.zeros((T, L), np.float32); lg[:m] = win
            yy = np.zeros(T, np.float32); yy[:m] = z[a:b]
            lag.append(lg); tgt.append(yy); lsc.append(math.log(sc))
        c, cm = N.pad_left(cz, N.CTX)
        h = net.encode(torch.from_numpy(c).to(dev), torch.from_numpy(cm).to(dev))
        lags = torch.from_numpy(np.stack(lag)).to(dev); y = torch.from_numpy(np.stack(tgt)).to(dev)
        mu, ls, nu, lw = net.params(h, lags)
        r = (y.unsqueeze(-1) - mu) / ls.exp()
        lp = (torch.lgamma((nu + 1) / 2) - torch.lgamma(nu / 2) - 0.5 * torch.log(nu * math.pi) - ls
              - (nu + 1) / 2 * torch.log1p(r * r / nu))
        lp = torch.logsumexp(lp + lw, -1).cpu().numpy()
        need = [j for j, i in enumerate(idx) if cdfr[i] is not None]
        cdf = {}
        if need:
            jj = np.array(need)
            w = lw[jj].exp().cpu().numpy()
            cd = (w * special.stdtr(nu[jj].cpu().numpy().astype(np.float64), r[jj].cpu().numpy().astype(np.float64))).sum(-1)
            for q, j in enumerate(need):
                cdf[j] = cd[q]
        for j, i in enumerate(idx):
            a, b = ranges[i]
            res[i] = (a, lp[j, :b - a].astype(np.float64), cdf[j][:b - a] if j in cdf else None, lsc[j])
    return res


def _mean_lp(item, a, b):
    off, lp, _, lsc = item
    n = max(0, min(b - a, off + len(lp) - a))
    if n <= 0:
        return -lsc
    return lp[a - off:a - off + n].mean() - lsc


def _pit(item, a, b):
    off, _, cdf, _ = item
    v = np.sort(cdf[a - off:b - off])
    n = len(v)
    if n < 5:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    ecdf = np.arange(1, n + 1) / n
    return (np.max(np.abs(ecdf - v)), v.mean() - 0.5, np.mean(np.abs(v - 0.5)) - 0.25,
            np.mean(np.abs(v - 0.5) > 0.45) - 0.1, np.mean(np.abs(v - 0.5) < 0.1) - 0.2)


def features2_fast(net, pairs, pit=True, normal=False, dev="cpu", honest=True):
    """normal=True also returns nullgauss.normal_scores(net, pairs) (pooled context), read off the
    pooled-context decode instead of a separate network pass: (features, normal scores)."""
    s = N.LAGS
    ctxs, fulls, ranges, cdfr = [], [], [], []
    for P, Q in pairs:
        x = np.concatenate((P, Q)); nP, nQ = len(P), len(Q); n = nP + nQ; hp, hq = nP // 2, nQ // 2
        e = max(s + 1, hp)
        spec = ((P, (min(s, nP), n), (nP, n)), (Q, (min(s, nP), n), (min(s, nP), nP)), (x, (min(s, nP), n), (min(s, nP), n)),
                (Q[:hq], (nP + hq, n), None), (P[:hp], (hp, nP), None), (x[:nP + hq], (nP + hq, n), None),
                (P[hp:], (s, min(e, n)), None), (x[hp:], (s, min(e, n)), None))
        if not honest:
            # honest=False: only the P, Q and pooled contexts (no half-segment contexts); the
            # half-segment ("honest") features are not computed
            spec = spec[:3]
        for j, (c, rg, cr) in enumerate(spec):
            ctxs.append(c); fulls.append(x); ranges.append(rg)
            cdfr.append(cr if (pit or (normal and j == 2)) else None)
    out = _decode(net, ctxs, fulls, ranges, cdfr, dev=dev)
    K = 8 if honest else 3
    if not honest:
        rows = []
        for i, (P, Q) in enumerate(pairs):
            nP, nQ = len(P), len(Q); n = nP + nQ; hq = nQ // 2
            dP, dQ, dX = out[3 * i:3 * i + 3]
            q_p, q_q = _mean_lp(dP, nP, n), _mean_lp(dQ, nP, n)
            p_p, p_q = _mean_lp(dP, s, nP), _mean_lp(dQ, s, nP)
            q_p2 = _mean_lp(dP, nP + hq, n)
            sym = nQ * (q_q - q_p) + nP * (p_p - p_q)
            a_a = _mean_lp(dX, s, n)
            lr_pool = (nP - s) * p_p + nQ * q_q - (nP + n - nP - s) * a_a
            rows.append([q_p, q_q, p_p, p_q, q_p2, q_q - q_p, p_p - p_q, sym, sym / n, a_a, lr_pool, lr_pool / n])
        if not normal:
            return np.array(rows)
        ns = []
        for i, (P, Q) in enumerate(pairs):
            off, _, cdf, _ = out[3 * i + 2]
            ns.append((special.ndtri(np.clip(cdf[s - off:], 1e-6, 1 - 1e-6)), len(P) - s))
        return np.array(rows), ns
    rows = []
    for i, (P, Q) in enumerate(pairs):
        nP, nQ = len(P), len(Q); n = nP + nQ; hp, hq = nP // 2, nQ // 2
        dP, dQ, dX, dQ1, dP1, dXq, dP2, dX2 = out[8 * i:8 * i + 8]
        q_p, q_q = _mean_lp(dP, nP, n), _mean_lp(dQ, nP, n)
        p_p, p_q = _mean_lp(dP, s, nP), _mean_lp(dQ, s, nP)
        q_q1, q_p2 = _mean_lp(dQ1, nP + hq, n), _mean_lp(dP, nP + hq, n)
        p_p1 = _mean_lp(dP1, hp, nP)
        sym = nQ * (q_q - q_p) + nP * (p_p - p_q)
        base = [q_p, q_q, p_p, p_q, q_q1, q_p2, p_p1, q_q1 - q_p2, q_q - q_p, p_p - p_q, sym, sym / n]
        a_a = _mean_lp(dX, s, n)
        pq1_q2 = _mean_lp(dXq, nP + hq, n)
        p1_p2 = _mean_lp(dP2, s, max(s + 1, hp))
        p1_p2q = _mean_lp(dX2, s, max(s + 1, hp))
        lr_pool = (nP - s) * p_p + nQ * q_q - (nP + n - nP - s) * a_a
        row = base + [a_a, lr_pool, lr_pool / n, pq1_q2, q_q1 - pq1_q2, p1_p2, p1_p2q, p1_p2 - p1_p2q]
        if pit:
            for item, a, b in ((dP, nP, n), (dX, nP, n), (dQ, s, nP), (dX, s, nP)):
                row += list(_pit(item, a, b))
        rows.append(row)
    if not normal:
        return np.array(rows)
    ns = []
    for i, (P, Q) in enumerate(pairs):
        off, _, cdf, _ = out[8 * i + 2]
        u = np.clip(cdf[s - off:], 1e-6, 1 - 1e-6)
        ns.append((special.ndtri(u), len(P) - s))
    return np.array(rows), ns


def break_score_fast(net, pairs, dev="cpu"):
    """Same as nullbreak.score: mean log density of the post segment given the pre regime, all /
    first 64 post points / the rest, from one decode of the post segment per series."""
    ctxs, fulls, ranges = [], [], []
    for P, Q in pairs:
        x = np.concatenate((P, Q)); nP, n = len(P), len(P) + len(Q)
        ctxs.append(P); fulls.append(x); ranges.append((nP, n))
    out = _decode(net, ctxs, fulls, ranges, [None] * len(pairs), dev=dev)
    r = np.zeros((len(pairs), 3))
    for i, (P, Q) in enumerate(pairs):
        nP, n = len(P), len(P) + len(Q)
        e = min(n, nP + 64)
        r[i] = (_mean_lp(out[i], nP, n), _mean_lp(out[i], nP, e), _mean_lp(out[i], e, max(e + 1, n)))
    nQ = np.array([len(q) for _, q in pairs], float)
    r[nQ <= 64, 2] = r[nQ <= 64, 0]
    return r
