"""Learned no-break model ("neural null") and the likelihood-ratio features built from it.

A small network is trained WITHOUT labels on stretches that contain no boundary (each pre
or post segment on its own). Given a conditioning stretch C it infers the regime (encoder
E), and a decoder gives a Student-t predictive density for the next point from its 16 lags
and E(C). For a series with pre P and post Q, the features compare how well Q is predicted
under the regime inferred from P versus from Q itself (and the reverse), which is a learned
Chow-type likelihood ratio that covers whatever dependence, volatility and tail structure
the generator uses.

Leakage control: the training series are split by the CV folds; the features of a series
in fold k come from the model trained on the other folds' segments, so no series is ever
scored by a model that saw it. Test series use the model trained on all training series.

    python solution/nullnet.py --work solution/work_nn            # 5 fold models + all-data model, features
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F  # noqa: E402

CTX = 384
LAGS = 16
TGT = 128
D = 64
NAMES = ["q_p", "q_q", "p_p", "p_q", "q_q1", "q_p2", "p_p1", "nc_post_honest", "nc_post", "nc_pre",
         "nc_sym", "nc_sym_per_n"]


def med_scale(x):
    m = np.median(x)
    s = np.median(np.abs(x - m)) * 1.4826
    if s < 1e-9:
        s = x.std() + 1e-9
    return m, s


# ------------------------------------------------------------------ views
def view_pairs(table, crops=True):
    """{view: list of (pre, post)} for the 4 standard views and (optionally) the crop
    views, built exactly like features.compute_views / compute_crop_views."""
    ids, starts, lengths, b, values = F._layout(table)
    rv = np.empty_like(values)
    for s, n in zip(starts, lengths):
        rv[s:s + n] = values[s:s + n][::-1]
    bases = {"orig": (values, b), "flip": (-values, b), "rev": (rv, lengths - b), "revflip": (-rv, lengths - b)}
    out = {}
    for view, (v, bb) in bases.items():
        out[view] = [(v[s:s + a], v[s + a:s + n]) for s, a, n in zip(starts, bb, lengths)]
    if crops:
        for r in range(F.CROP_ROUNDS):
            for k, (view, (v, bb)) in enumerate(bases.items()):
                rng = np.random.default_rng([F.CROP_SEED, r, k])
                n0 = bb
                n1 = lengths - bb
                kp = np.minimum(n0, np.maximum(F.CROP_MIN_LENGTH, (n0 * rng.uniform(F.CROP_MIN_FRACTION, 1.0, len(n0))).astype(np.int64)))
                kq = np.minimum(n1, np.maximum(F.CROP_MIN_LENGTH, (n1 * rng.uniform(F.CROP_MIN_FRACTION, 1.0, len(n1))).astype(np.int64)))
                out[f"crop{r}_{view}"] = [(v[s + a - p:s + a], v[s + a:s + a + q]) for s, a, p, q in zip(starts, n0, kp, kq)]
    return ids, out


# ------------------------------------------------------------------ model
class Block(nn.Module):
    def __init__(self, d, dil):
        super().__init__()
        self.c = nn.Conv1d(d, d, 3, dilation=dil, padding=dil)
        self.n = nn.GroupNorm(8, d)

    def forward(self, x):
        return x + torch.nn.functional.gelu(self.n(self.c(x)))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Conv1d(3, D, 5, padding=2)
        self.blocks = nn.Sequential(*[Block(D, 2 ** i) for i in range(5)])
        self.pool = nn.Linear(3 * D, D)
        self.dec = nn.Sequential(nn.Linear(2 * LAGS + D, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(),
                                 nn.Linear(128, 3))

    @staticmethod
    def chans(z):
        return torch.stack((z, z.abs(), (z * z).clamp(max=25.0)), 1)

    def encode(self, c, mask):
        """c, mask: (B, CTX), left padded."""
        h = self.blocks(self.inp(self.chans(c)))
        m = mask.unsqueeze(1)
        cnt = m.sum(2).clamp(min=1)
        mean = (h * m).sum(2) / cnt
        std = (((h - mean.unsqueeze(2)) ** 2 * m).sum(2) / cnt).clamp(min=1e-6).sqrt()
        last = h[:, :, -32:].mean(2)
        return self.pool(torch.cat((mean, std, last), 1))

    def logp(self, h, lags, y):
        """h (B, D); lags (B, T, LAGS); y (B, T) -> per-point Student-t log density."""
        hh = h.unsqueeze(1).expand(-1, y.shape[1], -1)
        o = self.dec(torch.cat((lags, lags.abs(), hh), 2))
        mu, ls, rn = o[..., 0], o[..., 1].clamp(-6, 4), o[..., 2]
        nu = 2.2 + torch.nn.functional.softplus(rn)
        r = (y - mu) / ls.exp()
        return (torch.lgamma((nu + 1) / 2) - torch.lgamma(nu / 2) - 0.5 * torch.log(nu * math.pi) - ls
                - (nu + 1) / 2 * torch.log1p(r * r / nu))


def pad_left(arrs, L):
    out = np.zeros((len(arrs), L), np.float32)
    mask = np.zeros((len(arrs), L), np.float32)
    for i, a in enumerate(arrs):
        a = a[-L:]
        out[i, L - len(a):] = a
        mask[i, L - len(a):] = 1
    return out, mask


# ------------------------------------------------------------------ training
def train_model(segments, seed, steps, dev, log=print):
    """segments: list of 1-D arrays without a boundary inside."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    net = Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    segs = [s for s in segments if len(s) >= 96]
    B = 256
    t0 = time.time()
    run = 0.0
    for step in range(steps):
        ctx, lag, tgt, tm = [], [], [], []
        for i in rng.integers(0, len(segs), B):
            s = segs[i]
            if rng.random() < 0.5:
                s = -s
            if rng.random() < 0.5:
                s = s[::-1]
            L = len(s)
            m = int(rng.integers(max(32, L // 4), L - 32))
            mu, sc = med_scale(s[:m])
            z = ((s - mu) / sc).astype(np.float32)
            ctx.append(z[:m])
            t = min(TGT, L - m)
            yy = np.zeros(TGT, np.float32)
            yy[:t] = z[m:m + t]
            lg = np.zeros((TGT, LAGS), np.float32)
            for j in range(t):
                w = z[m + j - LAGS:m + j][::-1]
                lg[j, :len(w)] = w
            msk = np.zeros(TGT, np.float32)
            msk[:t] = 1
            lag.append(lg); tgt.append(yy); tm.append(msk)
        c, cm = pad_left(ctx, CTX)
        c, cm = torch.from_numpy(c).to(dev), torch.from_numpy(cm).to(dev)
        lag = torch.from_numpy(np.stack(lag)).to(dev)
        tgt = torch.from_numpy(np.stack(tgt)).to(dev)
        tm = torch.from_numpy(np.stack(tm)).to(dev)
        lp = net.logp(net.encode(c, cm), lag, tgt)
        loss = -(lp * tm).sum() / tm.sum()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step else loss.item()
        if step % 1000 == 0 or step == steps - 1:
            log(f"  step {step} nll {run:.4f} ({time.time() - t0:.0f}s)")
    return net.eval()


# ------------------------------------------------------------------ scoring
@torch.no_grad()
def score_tasks(net, tasks, dev, batch=512):
    """tasks: list of (cond_raw, full_raw, t0, t1): mean log density (original units) of
    full_raw[t0:t1] given the regime inferred from cond_raw and each point's own lags."""
    res = np.zeros(len(tasks))
    order = np.argsort([t[3] - t[2] for t in tasks])
    for s in range(0, len(tasks), batch):
        idx = order[s:s + batch]
        T = max(1, max(tasks[i][3] - tasks[i][2] for i in idx))
        ctx, lag, tgt, tm, lsc = [], [], [], [], []
        for i in idx:
            c, x, a, b = tasks[i]
            mu, sc = med_scale(c)
            ctx.append(((c - mu) / sc).astype(np.float32))
            z = ((x - mu) / sc).astype(np.float32)
            n = b - a
            zp = np.concatenate((np.zeros(LAGS, np.float32), z))
            # lags of target point t (index a+j in z) = z[a+j-1 ... a+j-LAGS]
            win = np.lib.stride_tricks.sliding_window_view(zp, LAGS)[a:b][:, ::-1]
            n = max(0, min(n, win.shape[0], len(z) - a))   # empty or past-the-end target ranges (tiny segments)
            win = win[:n]
            lg = np.zeros((T, LAGS), np.float32)
            lg[:n] = win
            yy = np.zeros(T, np.float32)
            yy[:n] = z[a:b]
            m = np.zeros(T, np.float32)
            m[:n] = 1
            lag.append(lg); tgt.append(yy); tm.append(m); lsc.append(math.log(sc))
        c, cm = pad_left(ctx, CTX)
        h = net.encode(torch.from_numpy(c).to(dev), torch.from_numpy(cm).to(dev))
        lp = net.logp(h, torch.from_numpy(np.stack(lag)).to(dev), torch.from_numpy(np.stack(tgt)).to(dev))
        tm = torch.from_numpy(np.stack(tm)).to(dev)
        mean = ((lp * tm).sum(1) / tm.sum(1).clamp(min=1)).cpu().numpy()
        res[idx] = mean - np.array(lsc)
    return res


def series_features(net, pairs, dev):
    tasks = []
    for P, Q in pairs:
        x = np.concatenate((P, Q))
        nP, nQ = len(P), len(Q)
        hp, hq = nP // 2, nQ // 2
        s = LAGS  # first points of the series have no full lag window
        tasks += [(P, x, nP, nP + nQ),            # q_p
                  (Q, x, nP, nP + nQ),            # q_q
                  (P, x, s, nP),                  # p_p
                  (Q, x, s, nP),                  # p_q
                  (Q[:hq], x, nP + hq, nP + nQ),  # q_q1
                  (P, x, nP + hq, nP + nQ),       # q_p2
                  (P[:hp], x, hp, nP)]            # p_p1
    r = score_tasks(net, tasks, dev).reshape(len(pairs), 7)
    q_p, q_q, p_p, p_q, q_q1, q_p2, p_p1 = r.T
    nP = np.array([len(p) for p, _ in pairs], float)
    nQ = np.array([len(q) for _, q in pairs], float)
    sym = nQ * (q_q - q_p) + nP * (p_p - p_q)
    return np.column_stack((q_p, q_q, p_p, p_q, q_q1, q_p2, p_p1, q_q1 - q_p2, q_q - q_p, p_p - p_q,
                            sym, sym / (nP + nQ)))


def main():
    global D, LAGS, CTX
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds-only", action="store_true")
    ap.add_argument("--dim", type=int, default=D)
    ap.add_argument("--lags", type=int, default=LAGS)
    ap.add_argument("--ctx", type=int, default=CTX)
    a = ap.parse_args()
    D, LAGS, CTX = a.dim, a.lags, a.ctx
    a.work.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log = lambda m: print(m, flush=True)
    table = pd.read_parquet("train_series.parquet")
    ids, views = view_pairs(table)
    folds = pd.read_csv("solution/work/folds.csv").set_index("id").fold.reindex(ids).to_numpy()
    orig = views["orig"]
    feats = {v: np.zeros((len(ids), len(NAMES))) for v in views}
    for k in sorted(np.unique(folds)):
        segs = [s for i in np.where(folds != k)[0] for s in orig[i]]
        log(f"fold {k}: {len(segs)} training segments")
        net = train_model(segs, a.seed + k, a.steps, dev, log)
        torch.save(net.state_dict(), a.work / f"nullnet_fold{k}.pt")
        rows = np.where(folds == k)[0]
        for v, pairs in views.items():
            feats[v][rows] = series_features(net, [pairs[i] for i in rows], dev)
        log(f"fold {k} features done")
    for v, f in feats.items():
        pd.DataFrame(f, index=pd.Index(ids, name="id"), columns=[f"nn_{c}" for c in NAMES]).to_parquet(
            a.work / f"nn_features_{v}.parquet")
    if not a.folds_only:
        net = train_model([s for p in orig for s in p], a.seed + 99, a.steps, dev, log)
        torch.save(net.state_dict(), a.work / "nullnet_all.pt")
    log("done")


if __name__ == "__main__":
    main()
