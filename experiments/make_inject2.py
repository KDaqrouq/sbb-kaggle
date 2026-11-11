"""Injection probes: which kind of small break looks like the hidden real ones?

Into every real no-break series, inject at its boundary a break of one kind, working on
the series' own AR(8) innovations so all nuisance structure stays real:
  shape : post innovations e -> sign(e)|e|^g, rescaled to the same variance (tails change,
          mean and variance do not), g in [0.6, 0.85] or [1.2, 1.5]
  ar    : post rebuilt with the lag-1 AR coefficient moved by +-[0.1, 0.3] (variance matched)
  vol   : post innovations multiplied by a variance-1 stochastic-volatility factor
          (volatility clustering appears; overall variance unchanged)
If adding these as extra break examples helps the real CV, that kind matches hidden breaks.

    python solution/make_inject2.py --kind shape --work solution/work_inj_shape --nn solution/work_nn
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import lfilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "solution"))  # pipeline modules
import features  # noqa: E402
import nullnet as N  # noqa: E402

P_AR = 8


def ar_fit(x, p=P_AR):
    n = len(x)
    X = np.column_stack([np.ones(n - p)] + [x[p - j - 1:n - j - 1] for j in range(p)])
    c = np.linalg.lstsq(X, x[p:], rcond=None)[0]
    return c, x[p:] - X @ c


def rebuild(x, b, c, e_post, phi=None):
    """Rebuild x[b:] from innovations with coefficients (c0, phi), continuing from x[:b]."""
    p = P_AR
    phi = c[1:] if phi is None else phi
    out = x.copy()
    for t in range(b, len(x)):
        out[t] = c[0] + np.dot(phi, out[t - p:t][::-1]) + e_post[t - b]
    return out


def inject(x, b, kind, rng):
    c, e = ar_fit(x)
    ep = e[b - P_AR:].copy()                  # innovations of the post segment
    sd = ep.std()
    if kind == "shape":
        g = rng.uniform(0.6, 0.85) if rng.random() < 0.5 else rng.uniform(1.2, 1.5)
        z = ep / (sd + 1e-12)
        z2 = np.sign(z) * np.abs(z) ** g
        ep2 = (z2 - z2.mean()) / (z2.std() + 1e-12) * sd + ep.mean()
        return rebuild(x, b, c, ep2)
    if kind == "ar":
        phi = c[1:].copy()
        phi[0] += (1 if rng.random() < 0.5 else -1) * rng.uniform(0.1, 0.3)
        if np.max(np.abs(np.roots(np.concatenate(([1.0], -phi))))) >= 0.98:
            phi[0] -= 2 * (phi[0] - c[1])     # flip direction if it would be explosive
        if np.max(np.abs(np.roots(np.concatenate(([1.0], -phi))))) >= 0.98:
            return None
        # match the post marginal variance to the old one via the impulse responses
        def var_of(ph):
            ir = lfilter([1.0], np.concatenate(([1.0], -ph)), np.r_[1.0, np.zeros(400)])
            return np.sum(ir ** 2)
        ep2 = ep * np.sqrt(var_of(c[1:]) / var_of(phi))
        return rebuild(x, b, c, ep2, phi)
    if kind == "vol":
        n = len(ep)
        h = np.zeros(n)
        rho, s = rng.uniform(0.85, 0.97), rng.uniform(0.3, 0.6)
        for t in range(1, n):
            h[t] = rho * h[t - 1] + s * np.sqrt(1 - rho ** 2) * rng.standard_normal()
        f = np.exp(h)
        f /= np.sqrt(np.mean(f ** 2))
        return rebuild(x, b, c, ep * f)
    if kind == "donor":
        return None  # handled in main (needs the donor pool)
    raise ValueError(kind)


def donor_pool(t, y, rng, n=4000):
    """Breaks fitted on real VISIBLE break series: (dc, dg, dphi, log scale ratio)."""
    import simparam2 as S2
    out = []
    ids = [i for i in t.id.unique() if y[i] == 1]
    G = dict(tuple(t.groupby("id")))
    for sid in rng.choice(ids, n, replace=False):
        g = G[sid]
        v = g.value.to_numpy()
        b = int((g.period == 0).sum())
        m, s = S2.med_scale(v[:b])
        f = S2.fit_series((v - m) / s, b, 1)
        if not f["changes"]:
            continue
        dphi = np.zeros(P_AR)
        dphi[:len(f["dphi"])] = f["dphi"]
        out.append((f["dc"], f["dg"], dphi, np.log(f["s_post"] / f["s_pre"])))
    return out


def inject_donor(x, b, d, shrink):
    """Apply a donor break (shrunk) at the boundary of a no-break series, in pre robust-sd units."""
    m = np.median(x[:b])
    s = np.median(np.abs(x[:b] - m)) * 1.4826 + 1e-9
    z = (x - m) / s
    c, e = ar_fit(z)
    dc, dg, dphi, lsr = d
    phi = c[1:] + shrink * dphi
    if np.max(np.abs(np.roots(np.concatenate(([1.0], -phi))))) >= 0.98:
        phi = c[1:]
    n = len(z)
    out = z.copy()
    ep = e[b - P_AR:] * np.exp(shrink * lsr)
    for t in range(b, n):
        out[t] = c[0] + shrink * (dc + dg * t / n) + np.dot(phi, out[t - P_AR:t][::-1]) + ep[t - b]
    return out * s + m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--nn", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shrink-lo", type=float, default=0.2)
    ap.add_argument("--shrink-hi", type=float, default=0.6)
    a = ap.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    t = pd.read_parquet("train_series.parquet")
    y = pd.read_csv("train_labels.csv").set_index("id").label
    folds = pd.read_csv("solution/work/folds.csv").set_index("id").fold
    rows, meta = [], []
    pool = donor_pool(t, y, rng) if a.kind == "donor" else None
    if pool is not None:
        print(len(pool), "donor breaks", flush=True)
    for sid, g in t.groupby("id", sort=False):
        if y[sid] != 0:
            continue
        v = g.value.to_numpy().astype(float)
        b = int((g.period == 0).sum())
        if pool is not None:
            w = inject_donor(v, b, pool[rng.integers(len(pool))], rng.uniform(a.shrink_lo, a.shrink_hi))
        else:
            w = inject(v, b, a.kind, rng)
        if w is None or not np.all(np.isfinite(w)):
            continue
        new = f"inj_{sid}"
        rows.append(pd.DataFrame({"id": new, "time": np.arange(len(w)), "value": w, "period": g.period.to_numpy()}))
        meta.append((new, sid, 1, len(w), folds[sid]))
    table = pd.concat(rows, ignore_index=True)
    emap = pd.DataFrame(meta, columns=["id", "source", "label", "n", "fold"])
    emap.to_csv(a.work / "extra_map.csv", index=False)
    print(len(emap), "injected series", flush=True)
    hand = features.compute_views(table)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ids, views = N.view_pairs(table, crops=False)
    fold_of = emap.set_index("id").fold.reindex(ids).to_numpy()
    nnf = {v: np.zeros((len(ids), len(N.NAMES))) for v in views}
    for k in sorted(np.unique(fold_of)):
        net = N.Net().to(dev)
        net.load_state_dict(torch.load(a.nn / f"nullnet_fold{k}.pt", map_location=dev))
        net.eval()
        rk = np.where(fold_of == k)[0]
        for v, pairs in views.items():
            nnf[v][rk] = N.series_features(net, [pairs[i] for i in rk], dev)
    for v in views:
        nn_df = pd.DataFrame(nnf[v], index=pd.Index(ids, name="id"), columns=[f"nn_{c}" for c in N.NAMES])
        pd.concat([hand[v], nn_df.reindex(hand[v].index)], axis=1).to_parquet(a.work / f"extra_features_{v}.parquet")
    print("done", a.work)


if __name__ == "__main__":
    main()
