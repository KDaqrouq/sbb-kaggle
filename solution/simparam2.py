"""Explicit-model simulator v2: breaks change only the parameters that actually changed.

Whole-series model: x_t = c + g*t/n + sum_j phi_j x_{t-j} + s_t * e_t, with e_t a GARCH(1,1)
filtered shock resampled from the series' own standardised residuals. For a break series,
BIC picks which of {level, trend, AR dynamics, innovation scale} switch at the boundary
(post dummies on the regression terms; a separate post innovation variance); a no-break
series uses the single pooled model. Replicas start from the real first p values and run
continuously through the boundary.

    python solution/simparam2.py --fold 0 --n 6000 --out solution/work_sim/simp2_f0.parquet
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

GA = np.array([0.0, 0.05, 0.1, 0.2])
GP = np.array([0.0, 0.6, 0.8, 0.9, 0.95])


def med_scale(x):
    m = np.median(x)
    s = np.median(np.abs(x - m)) * 1.4826
    return m, (s if s > 1e-9 else x.std() + 1e-9)


@njit(cache=True)
def garch_h(e, s2, a, b):
    om = s2 * (1 - a - b)
    h = np.empty(len(e))
    hv = s2
    for t in range(len(e)):
        if t:
            hv = om + a * e[t - 1] ** 2 + b * hv
        h[t] = max(hv, 1e-6 * s2)
    return h


def garch_fit(e):
    s2 = max(np.mean(e * e), 1e-9)
    best, ab = -0.5 * np.sum(np.log(np.full(len(e), s2)) + e * e / s2), (0.0, 0.0)
    for a in GA[1:]:
        for ps in GP[1:]:
            if ps <= a:
                continue
            h = garch_h(e, s2, a, ps - a)
            ll = -0.5 * np.sum(np.log(h) + e * e / h) - np.log(len(e))
            if ll > best:
                best, ab = ll, (a, ps - a)
    return s2, ab


def fit_series(x, b, label, pmax=8):
    n = len(x)
    p = max(1, min(pmax, min(b, n - b) // 8))
    t = np.arange(p, n) / n
    L = np.column_stack([x[p - j - 1:n - j - 1] for j in range(p)])
    D = (np.arange(p, n) >= b).astype(float)
    y = x[p:]
    blocks = {"level": D[:, None], "trend": (D * t)[:, None], "ar": L * D[:, None]}
    base = np.column_stack((np.ones(len(y)), t, L))
    chosen, best_bic = (), None
    combos = [()] if label == 0 else [c for r in range(4) for c in itertools.combinations(blocks, r)]
    for combo in combos:
        X = np.column_stack([base] + [blocks[k] for k in combo])
        c = np.linalg.lstsq(X, y, rcond=None)[0]
        r = y - X @ c
        bic = len(y) * np.log(np.mean(r * r)) + X.shape[1] * np.log(len(y))
        if best_bic is None or bic < best_bic:
            best_bic, chosen, coef, resid = bic, combo, c, r
    pre, post = resid[D == 0], resid[D == 1]
    s_pre, s_post = np.sqrt(np.mean(pre ** 2)), np.sqrt(np.mean(post ** 2))
    pooled = np.sqrt(np.mean(resid ** 2))
    var_change = False
    if label == 1:
        lr = len(resid) * np.log(pooled ** 2) - len(pre) * np.log(s_pre ** 2) - len(post) * np.log(s_post ** 2)
        var_change = lr > np.log(len(resid))
    scale = np.where(D == 1, s_post if var_change else pooled, s_pre if var_change else pooled)
    u = resid / scale
    s2, ab = garch_fit(u)
    z = u / np.sqrt(garch_h(u, s2, *ab))
    k = 2 + p
    full = dict(c=coef[0], g=coef[1], phi=coef[2:k], dc=0.0, dg=0.0, dphi=np.zeros(p))
    i = k
    for name in chosen:
        if name == "level":
            full["dc"] = coef[i]; i += 1
        elif name == "trend":
            full["dg"] = coef[i]; i += 1
        else:
            full["dphi"] = coef[i:i + p]; i += p
    zs = (z - z.mean()) / (z.std() + 1e-9)
    Dz = D == 1
    zpre, zpost = zs[~Dz], zs[Dz]
    if label == 1:   # break series: each side keeps its own shock distribution (shape changes)
        zpre, zpost = (zpre - zpre.mean()) / (zpre.std() + 1e-9), (zpost - zpost.mean()) / (zpost.std() + 1e-9)
    else:
        zpre = zpost = zs
    return dict(p=p, x0=x[:p].copy(), s_pre=s_pre if var_change else pooled, s_post=s_post if var_change else pooled,
                s2=s2, ab=ab, z=zs, zpre=zpre, zpost=zpost, changes=chosen + (("scale",) if var_change else ()),
                **full)


@njit(cache=True)
def simulate(n, b, p, x0, c, g, phi, dc, dg, dphi, s_pre, s_post, s2, a, bb, z):
    x = np.empty(n)
    x[:p] = x0
    om = s2 * (1 - a - bb)
    hv = s2
    eprev = 0.0
    for t in range(p, n):
        post = t >= b
        m = c + g * t / n + (dc + dg * t / n if post else 0.0)
        for j in range(p):
            m += (phi[j] + (dphi[j] if post else 0.0)) * x[t - j - 1]
        if t > p:
            hv = om + a * eprev * eprev + bb * hv
        e = np.sqrt(max(hv, 1e-6 * s2)) * z[t - p]
        eprev = e
        x[t] = m + (s_post if post else s_pre) * e
    return x


def replicate(P, Q, label, rng):
    mp, sp = med_scale(P)
    x = np.concatenate(((P - mp) / sp, (Q - mp) / sp))
    b = len(P)
    f = fit_series(x, b, label)
    z = np.concatenate((rng.choice(f["zpre"], b), rng.choice(f["zpost"], len(x) - b)))[f["p"]:]
    z = np.concatenate((z, np.zeros(f["p"])))
    out = simulate(len(x), b, f["p"], f["x0"], f["c"], f["g"], f["phi"], f["dc"], f["dg"], f["dphi"],
                   f["s_pre"], f["s_post"], f["s2"], f["ab"][0], f["ab"][1], z)
    if not np.all(np.isfinite(out)) or np.abs(out).max() > 1e6:      # explosive fit: fall back to the real series
        out = x
    return out[:b], out[b:], f["changes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    t = pd.read_parquet("train_series.parquet")
    y = pd.read_csv("train_labels.csv").set_index("id").label
    folds = pd.read_csv("solution/work/folds.csv").set_index("id").fold
    ids = [i for i in t.id.unique() if folds[i] != a.fold]
    rng = np.random.default_rng(0)
    pick = rng.choice(ids, a.n, replace=False)
    G = dict(tuple(t[t.id.isin(set(pick))].groupby("id")))
    rows, labs = [], []
    counts = {}
    for sid in pick:
        g = G[sid].sort_values("time")
        v = g.value.to_numpy()
        b = int((g.period == 0).sum())
        P, Q, ch = replicate(v[:b], v[b:], y[sid], rng)
        if y[sid] == 1:
            counts[ch] = counts.get(ch, 0) + 1
        x = np.concatenate((P, Q))
        rows.append(pd.DataFrame({"id": f"simp_{sid}", "time": np.arange(len(x)), "value": x,
                                  "period": (np.arange(len(x)) >= b).astype(np.int8)}))
        labs.append((f"simp_{sid}", sid, y[sid]))
    pd.concat(rows, ignore_index=True).to_parquet(a.out)
    pd.DataFrame(labs, columns=["id", "source", "label"]).to_csv(a.out.with_suffix(".labels.csv"), index=False)
    print("break types chosen for break series:")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {'+'.join(k) or '(none)'}: {v}")
    print("wrote", a.out, len(labs))


if __name__ == "__main__":
    main()
