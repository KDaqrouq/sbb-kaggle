"""No-break network v2: pooled-context likelihood ratios (a learned Chow test) + variants.

v1 compared post under the regime inferred from the PRE segment alone with post under its
own regime. With short segments the pre-only regime estimate is noisy, which costs power.
A Chow test instead compares separate fits with ONE POOLED fit on all data (the no-break
model). v2 does the same with the network:
  * the encoder context can cover a whole series (CTX 1024), and training stretches include
    whole no-break series from the training folds (all genuinely break-free);
  * features: v1's, plus the in-sample pooled log-likelihood, the pooled LR (separate -
    pooled) and honest held-out versions.
Variants (each one experiment):
  --transform detrend|diff : the network models the series after removing one linear trend
                              through the whole series / after first differencing
  --mix K                  : K-component Student-t mixture output (richer shapes)
  --lags L                 : longer lag window
  --pit                    : PIT calibration features (post under pre / pooled regime, pre
                              under post / pooled regime): KS, location, dispersion, tails, core
  --reuse                  : load the fold models from --work instead of training
Cross-fitting as in v1 (fold k features from the model trained without fold k).

    python solution/nullnet2.py --work solution/work_nn2p --steps 8000 [variant flags]
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nullnet as N  # noqa: E402

N.CTX = 1024
MIX = 1
BASE2 = ["a_a", "lr_pool", "lr_pool_per_n", "pq1_q2", "hon_post_pool", "p1_p2", "p1_p2q", "hon_pre_pool"]
PIT_NAMES = [f"pit_{side}_{cond}_{s}" for side, conds in (("q", ("p", "a")), ("p", ("q", "a")))
             for cond in conds for s in ("ks", "loc", "disp", "tail", "core")]


class MixNet(N.Net):
    """Student-t mixture head (MIX components)."""

    def __init__(self):
        super().__init__()
        self.dec = nn.Sequential(nn.Linear(2 * N.LAGS + N.D, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(),
                                 nn.Linear(128, 4 * MIX))

    def params(self, h, lags):
        hh = h.unsqueeze(1).expand(-1, lags.shape[1], -1)
        o = self.dec(torch.cat((lags, lags.abs(), hh), 2)).view(*lags.shape[:2], MIX, 4)
        mu, ls, rn, w = o[..., 0], o[..., 1].clamp(-6, 4), o[..., 2], o[..., 3]
        return mu, ls, 2.2 + nn.functional.softplus(rn), w.log_softmax(-1)

    def logp(self, h, lags, y):
        mu, ls, nu, lw = self.params(h, lags)
        r = (y.unsqueeze(-1) - mu) / ls.exp()
        lp = (torch.lgamma((nu + 1) / 2) - torch.lgamma(nu / 2) - 0.5 * torch.log(nu * math.pi) - ls
              - (nu + 1) / 2 * torch.log1p(r * r / nu))
        return torch.logsumexp(lp + lw, -1)


HIST = 0          # >0: histogram density head with HIST bins on [-HR, HR] plus exponential tails
HR = 5.0


class HistNet(N.Net):
    """Fully flexible density: location/scale + HIST-bin histogram of the standardised
    shock r = (y - mu)/s on [-HR, HR], with exponential tails beyond (rate 1)."""

    def __init__(self):
        super().__init__()
        self.dec = nn.Sequential(nn.Linear(2 * N.LAGS + N.D, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(),
                                 nn.Linear(128, 2 + HIST + 2))

    def params(self, h, lags):
        hh = h.unsqueeze(1).expand(-1, lags.shape[1], -1)
        o = self.dec(torch.cat((lags, lags.abs(), hh), 2))
        return o[..., 0], o[..., 1].clamp(-6, 4), o[..., 2:].log_softmax(-1)

    def logp(self, h, lags, y):
        mu, ls, lw = self.params(h, lags)
        r = (y - mu) / ls.exp()
        width = 2 * HR / HIST
        k = ((r + HR) / width).floor().clamp(0, HIST - 1).long()
        inside = lw[..., :HIST].gather(-1, k.unsqueeze(-1)).squeeze(-1) - math.log(width)
        left = lw[..., HIST] - (-HR - r)
        right = lw[..., HIST + 1] - (r - HR)
        lp = torch.where(r < -HR, left, torch.where(r > HR, right, inside))
        return lp - ls


BOTTLENECK = 0   # >0: squeeze the regime embedding through this many dimensions


def make_net():
    if HIST:
        net = HistNet()
    else:
        net = MixNet() if MIX > 1 else N.Net()
    if BOTTLENECK:
        net.pool = nn.Sequential(nn.Linear(3 * N.D, BOTTLENECK), nn.Linear(BOTTLENECK, N.D))
    return net


def train_model(segments, seed, steps, dev, log):
    """nullnet.train_model with a configurable network."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    net = make_net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    B, TGT, L = 256, N.TGT, N.LAGS
    segs = [s for s in segments if len(s) >= max(96, L + 64)]
    t0, run = time.time(), 0.0
    for step in range(steps):
        ctx, lag, tgt, tm = [], [], [], []
        for i in rng.integers(0, len(segs), B):
            s = segs[i]
            if rng.random() < 0.5:
                s = -s
            if rng.random() < 0.5:
                s = s[::-1]
            n = len(s)
            m = int(rng.integers(max(L + 16, n // 4), n - 32))
            mu, sc = N.med_scale(s[:m])
            z = ((s - mu) / sc).astype(np.float32)
            ctx.append(z[:m])
            t = min(TGT, n - m)
            yy = np.zeros(TGT, np.float32); yy[:t] = z[m:m + t]
            lg = np.zeros((TGT, L), np.float32)
            win = np.lib.stride_tricks.sliding_window_view(z[m - L:m + t - 1 + 1], L)[:t][:, ::-1] if t else None
            if t:
                lg[:t] = win[:, :L]
            msk = np.zeros(TGT, np.float32); msk[:t] = 1
            lag.append(lg); tgt.append(yy); tm.append(msk)
        c, cm = N.pad_left(ctx, N.CTX)
        lp = net.logp(net.encode(torch.from_numpy(c).to(dev), torch.from_numpy(cm).to(dev)),
                      torch.from_numpy(np.stack(lag)).to(dev), torch.from_numpy(np.stack(tgt)).to(dev))
        tmt = torch.from_numpy(np.stack(tm)).to(dev)
        loss = -(lp * tmt).sum() / tmt.sum()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step else loss.item()
        if step % 2000 == 0 or step == steps - 1:
            log(f"  step {step} nll {run:.4f} ({time.time() - t0:.0f}s)")
    return net.eval()


@torch.no_grad()
def pit_stats(net, tasks, dev, batch=512):
    """For (cond, full, t0, t1): PIT of full[t0:t1] under the regime of cond (Student-t,
    single-component models only) -> ks, location, dispersion, tail rate, core rate."""
    out = np.zeros((len(tasks), 5))
    order = np.argsort([t[3] - t[2] for t in tasks])
    for s in range(0, len(tasks), batch):
        idx = order[s:s + batch]
        T = max(1, max(tasks[i][3] - tasks[i][2] for i in idx))
        ctx, lag, tgt, tm = [], [], [], []
        for i in idx:
            c, x, a, b = tasks[i]
            mu, sc = N.med_scale(c)
            ctx.append(((c - mu) / sc).astype(np.float32))
            z = ((x - mu) / sc).astype(np.float32)
            zp = np.concatenate((np.zeros(N.LAGS, np.float32), z))
            lg = np.zeros((T, N.LAGS), np.float32)
            win = np.lib.stride_tricks.sliding_window_view(zp, N.LAGS)[a:b][:, ::-1]
            nn_ = max(0, min(b - a, win.shape[0], len(z) - a))
            lg[:nn_] = win[:nn_]
            yy = np.zeros(T, np.float32); yy[:nn_] = z[a:a + nn_]
            lag.append(lg); tgt.append(yy); tm.append(nn_)
        c, cm = N.pad_left(ctx, N.CTX)
        h = net.encode(torch.from_numpy(c).to(dev), torch.from_numpy(cm).to(dev))
        lg = torch.from_numpy(np.stack(lag)).to(dev)
        hh = h.unsqueeze(1).expand(-1, lg.shape[1], -1)
        o = net.dec(torch.cat((lg, lg.abs(), hh), 2)).cpu().numpy()
        yy = np.stack(tgt)
        if HIST:
            w = np.exp(o[..., 2:] - o[..., 2:].max(-1, keepdims=True))
            w /= w.sum(-1, keepdims=True)
            r = (yy - o[..., 0]) / np.exp(np.clip(o[..., 1], -6, 4))
            width = 2 * HR / HIST
            cw = np.concatenate((np.zeros(w.shape[:-1] + (1,)), np.cumsum(w[..., :HIST], -1)), -1)
            pos = np.clip((r + HR) / width, 0, HIST)
            kk = np.clip(np.floor(pos).astype(int), 0, HIST - 1)
            inner = (np.take_along_axis(cw, kk[..., None], -1)[..., 0]
                     + np.take_along_axis(w[..., :HIST], kk[..., None], -1)[..., 0] * (pos - kk))
            wl, wr = w[..., HIST], w[..., HIST + 1]
            u = np.where(r < -HR, wl * np.exp(-(-HR - r)), np.where(r > HR, 1 - wr * np.exp(-(r - HR)), wl + inner))
        elif MIX > 1:      # mixture CDF: sum_k w_k T_k((y - mu_k) / s_k)
            o = o.reshape(*o.shape[:2], MIX, 4)
            w = np.exp(o[..., 3] - o[..., 3].max(-1, keepdims=True))
            w /= w.sum(-1, keepdims=True)
            r = (yy[..., None] - o[..., 0]) / np.exp(np.clip(o[..., 1], -6, 4))
            nu = 2.2 + np.log1p(np.exp(o[..., 2]))
            u = (w * stats.t.cdf(r, nu)).sum(-1)
        else:
            r = (yy - o[..., 0]) / np.exp(np.clip(o[..., 1], -6, 4))
            nu = 2.2 + np.log1p(np.exp(o[..., 2]))
            u = stats.t.cdf(r, nu)
        for j, i in enumerate(idx):
            v = np.sort(u[j, :tm[j]])
            n = len(v)
            if n < 5:
                continue
            ecdf = np.arange(1, n + 1) / n
            out[i] = (np.max(np.abs(ecdf - v)), v.mean() - 0.5, np.mean(np.abs(v - 0.5)) - 0.25,
                      np.mean(np.abs(v - 0.5) > 0.45) - 0.1, np.mean(np.abs(v - 0.5) < 0.1) - 0.2)
    return out


def transform(P, Q, how):
    if how == "detrend":
        x = np.concatenate((P, Q))
        t = np.linspace(-1, 1, len(x))
        r = x - np.polyval(np.polyfit(t, x, 1), t)
        return r[:len(P)], r[len(P):]
    if how == "diff":
        x = np.diff(np.concatenate((P, Q)))
        return x[:len(P) - 1], x[len(P) - 1:]
    return P, Q


def features2(net, pairs, dev, pit):
    base = N.series_features(net, pairs, dev)
    tasks, ptasks = [], []
    s = N.LAGS
    for P, Q in pairs:
        x = np.concatenate((P, Q))
        nP, nQ = len(P), len(Q)
        hp, hq = nP // 2, nQ // 2
        tasks += [(x, x, s, nP + nQ), (x[:nP + hq], x, nP + hq, nP + nQ), (P[hp:], x, s, max(s + 1, hp)),
                  (x[hp:], x, s, max(s + 1, hp))]
        if pit:
            ptasks += [(P, x, nP, nP + nQ), (x, x, nP, nP + nQ), (Q, x, s, nP), (x, x, s, nP)]
    r = N.score_tasks(net, tasks, dev).reshape(len(pairs), 4)
    a_a, pq1_q2, p1_p2, p1_p2q = r.T
    nP = np.array([len(p) for p, _ in pairs], float)
    nQ = np.array([len(q) for _, q in pairs], float)
    q_q, p_p, q_q1 = base[:, 1], base[:, 2], base[:, 4]
    lr_pool = (nP - s) * p_p + nQ * q_q - (nP + nQ - s) * a_a
    cols = [base, np.column_stack((a_a, lr_pool, lr_pool / (nP + nQ), pq1_q2, q_q1 - pq1_q2, p1_p2, p1_p2q,
                                   p1_p2 - p1_p2q))]
    if pit:
        cols.append(pit_stats(net, ptasks, dev).reshape(len(pairs), 20))
    return np.concatenate(cols, 1)


def main():
    global MIX
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-whole", action="store_true", help="train on segments only (as v1)")
    ap.add_argument("--transform", default="none", choices=["none", "detrend", "diff"])
    ap.add_argument("--mix", type=int, default=1)
    ap.add_argument("--lags", type=int, default=16)
    ap.add_argument("--pit", action="store_true")
    ap.add_argument("--hist", type=int, default=0)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--bottleneck", type=int, default=0)
    ap.add_argument("--with-test", action="store_true", help="also train on the (unlabelled) pre/post segments of the test series")
    ap.add_argument("--netfolds", type=int, default=5, choices=[5, 10])
    ap.add_argument("--reuse", action="store_true")
    a = ap.parse_args()
    global HIST, BOTTLENECK
    MIX, N.LAGS, HIST = a.mix, a.lags, a.hist
    N.D = a.dim
    BOTTLENECK = a.bottleneck
    a.work.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log = lambda m: print(m, flush=True)
    ids, views = N.view_pairs(pd.read_parquet("train_series.parquet"))
    views = {v: [transform(P, Q, a.transform) for P, Q in pairs] for v, pairs in views.items()}
    folds = pd.read_csv("solution/work/folds.csv").set_index("id").fold.reindex(ids).to_numpy()
    if a.netfolds == 10:
        # finer cross-fitting: each CV fold split in two, so every network trains on 90% of the
        # series; a series' features still never come from a network that saw it
        half = np.random.default_rng(12345).random(len(ids)) < 0.5
        folds = folds * 2 + half.astype(int)
        pd.DataFrame({"id": ids, "fold": folds}).to_csv(a.work / "netfolds.csv", index=False)
    y = pd.read_csv("train_labels.csv").set_index("id").label.reindex(ids).to_numpy()
    orig = views["orig"]
    names = N.NAMES + BASE2 + (PIT_NAMES if a.pit else [])
    feats = {v: np.zeros((len(ids), len(names))) for v in views}
    test_segs = []
    if a.with_test:
        # transductive, label-free: each test segment on its own is a no-break stretch
        _, tv = N.view_pairs(pd.read_parquet("test_series.parquet"), crops=False)
        test_segs = [M_transform_seg for P, Q in tv["orig"] for M_transform_seg in transform(P, Q, a.transform)]
        print(f"adding {len(test_segs)} unlabelled test segments", flush=True)
    for k in sorted(np.unique(folds)):
        if a.reuse:
            net = make_net().to(dev)
            net.load_state_dict(torch.load(a.work / f"nullnet_fold{k}.pt", map_location=dev))
            net.eval()
        else:
            tr = np.where(folds != k)[0]
            segs = [s for i in tr for s in orig[i]]
            if not a.no_whole:
                segs += [np.concatenate(orig[i]) for i in tr if y[i] == 0]
            segs += test_segs
            log(f"fold {k}: {len(segs)} training stretches")
            net = train_model(segs, a.seed + k, a.steps, dev, log)
            torch.save(net.state_dict(), a.work / f"nullnet_fold{k}.pt")
        rows = np.where(folds == k)[0]
        for v, pairs in views.items():
            feats[v][rows] = features2(net, [pairs[i] for i in rows], dev, a.pit)
        log(f"fold {k} features done")
    tag = "nn" + {"none": "", "detrend": "dt", "diff": "df"}[a.transform] + ("m" if a.mix > 1 else "") + (f"h{a.hist}" if a.hist else "") + (f"d{a.dim}" if a.dim != 64 else "") + (f"b{a.bottleneck}" if a.bottleneck else "") + ("t" if a.with_test else "") +         (f"l{a.lags}" if a.lags != 16 else "") + "_"
    for v, f in feats.items():
        pd.DataFrame(f, index=pd.Index(ids, name="id"), columns=[f"{tag}{c}" for c in names]).to_parquet(
            a.work / f"nn_features_{v}.parquet")
    log("done")


if __name__ == "__main__":
    main()
