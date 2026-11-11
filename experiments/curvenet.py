"""Bet: a learned model over the networks' PER-POINT evidence curves (the trees only see averages).

Per series and view, channels over time (points LAGS..n-1, all from decodes the inference route
already makes):
  0 d1  = log p_H1(x_t | pre) - log p_H0(x_t | pre)        (post points; 0 on pre)
  1 dh  = log p_Hhidden(x_t | pre) - log p_H0(x_t | pre)   (post points; 0 on pre)
  2 a   = log p_null(x_t | pre regime)  (16/64 no-break net, pre context, all points)
  3 x   = log p_null(x_t | whole series) (pooled context, all points)
  4 a-x
  5 post indicator
A small dilated CNN with masked mean/max pooling over pre and over post outputs a break logit.
Cross-fitted twice like everything else: curves of a series come from its own fold's networks;
the curve model for outer fold k is trained on the other folds and scores fold k (all views).

    python solution/curvenet.py --nets solution/nets_all --hh solution/work_hh --out solution/work_cvm
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
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "solution"))  # pipeline modules
import nullfast as NF  # noqa: E402
import nullnet as N  # noqa: E402
import nullnet2 as M  # noqa: E402

C = 6


def curves(nets, pairs, dev):
    """nets: dict h1, h0, hh, nn (16/64). -> list of float32 arrays (C, n-L)."""
    L = N.LAGS
    ctxP, fulls, rpost, rall = [], [], [], []
    for P, Q in pairs:
        x = np.concatenate((P, Q)); nP, n = len(P), len(x)
        ctxP.append(P); fulls.append(x); rpost.append((nP, n)); rall.append((min(L, nP), n))
    none = [None] * len(pairs)
    o1 = NF._decode(nets["h1"], ctxP, fulls, rpost, none, dev=dev)
    o0 = NF._decode(nets["h0"], ctxP, fulls, rpost, none, dev=dev)
    oh = NF._decode(nets["hh"], ctxP, fulls, rpost, none, dev=dev)
    oa = NF._decode(nets["nn"], ctxP, fulls, rall, none, dev=dev)
    ox = NF._decode(nets["nn"], fulls, fulls, rall, none, dev=dev)
    out = []
    for i, (P, Q) in enumerate(pairs):
        nP, n = len(P), len(P) + len(Q)
        T = n - L
        c = np.zeros((C, T), np.float32)
        s = L  # time index of column 0
        lp1, lp0, lph = o1[i][1], o0[i][1], oh[i][1]
        a0 = max(nP, s)
        c[0, a0 - s:] = (lp1 - lp0)[a0 - nP:]
        c[1, a0 - s:] = (lph - lp0)[a0 - nP:]
        offa, lpa, _, lsa = oa[i]
        offx, lpx, _, lsx = ox[i]
        c[2] = (lpa - lsa)[s - offa:]
        c[3] = (lpx - lsx)[s - offx:]
        c[4] = c[2] - c[3]
        c[5, max(0, nP - s):] = 1
        out.append(np.clip(c, -20, 20))
    return out


class CurveNet(nn.Module):
    def __init__(self, d=48):
        super().__init__()
        self.inp = nn.Conv1d(C, d, 5, padding=2)
        self.blocks = nn.ModuleList([nn.Conv1d(d, d, 3, padding=r, dilation=r) for r in (1, 2, 4, 8, 16)])
        self.norms = nn.ModuleList([nn.GroupNorm(8, d) for _ in range(5)])
        self.head = nn.Sequential(nn.Linear(4 * d + 2, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x, valid):
        """x (B, C, T); valid (B, T) 1 on real points."""
        h = self.inp(x)
        for c, g in zip(self.blocks, self.norms):
            h = h + nn.functional.gelu(g(c(h))) * valid.unsqueeze(1)
        post = x[:, 5] * valid
        pre = (1 - x[:, 5]) * valid
        feats = []
        for m in (pre, post):
            cnt = m.sum(1, keepdim=True).clamp(min=1)
            feats.append((h * m.unsqueeze(1)).sum(2) / cnt)
            feats.append((h + (m.unsqueeze(1) - 1) * 1e4).max(2).values)
        ln = torch.stack((torch.log1p(pre.sum(1)), torch.log1p(post.sum(1))), 1)
        return self.head(torch.cat(feats + [ln], 1)).squeeze(1)


def batches(items, idx, bs, rng=None):
    idx = np.array(idx)
    if rng is not None:
        idx = idx[rng.permutation(len(idx))]
        # bucket by length within chunks of 50 batches for speed
        chunks = [idx[i:i + bs * 50] for i in range(0, len(idx), bs * 50)]
        idx = np.concatenate([c[np.argsort([items[j].shape[1] for j in c])] for c in chunks])
        order = [idx[i:i + bs] for i in range(0, len(idx), bs)]
        rng.shuffle(order)
    else:
        idx = idx[np.argsort([items[j].shape[1] for j in idx])]
        order = [idx[i:i + bs] for i in range(0, len(idx), bs)]
    for b in order:
        T = max(items[j].shape[1] for j in b)
        x = np.zeros((len(b), C, T), np.float32); v = np.zeros((len(b), T), np.float32)
        for r, j in enumerate(b):
            t = items[j].shape[1]; x[r, :, :t] = items[j]; v[r, :t] = 1
        yield b, torch.from_numpy(x), torch.from_numpy(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nets", type=Path, required=True)
    ap.add_argument("--hh", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=6)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)
    ids, views = N.view_pairs(pd.read_parquet("train_series.parquet"))
    vnames = list(views)
    folds = pd.read_csv("solution/work/folds.csv").set_index("id").fold.reindex(ids).to_numpy()
    y = pd.read_csv("train_labels.csv").set_index("id").label.reindex(ids).to_numpy().astype(np.float32)
    nS = len(ids)
    cache = a.out / "curves.npz"
    items = [None] * (nS * len(vnames))       # item index = vi * nS + series
    M.MIX, N.LAGS, M.HIST, M.BOTTLENECK = 16, 64, 0, 0

    def load(path):
        net = M.make_net().to(dev)
        net.load_state_dict(torch.load(path, map_location=dev, weights_only=True))
        return net.eval()

    with torch.no_grad():
        for k in sorted(np.unique(folds)):
            nets = {"h1": load(a.nets / "work_h1" / f"break1_fold{k}.pt"), "h0": load(a.nets / "work_h0" / f"break0_fold{k}.pt"),
                    "hh": load(a.hh / f"breakh_fold{k}.pt"), "nn": load(a.nets / "work_nn2m16l64XL" / f"nullnet_fold{k}.pt")}
            rows = np.where(folds == k)[0]
            for vi, v in enumerate(vnames):
                cs = curves(nets, [views[v][i] for i in rows], dev)
                for r, c in zip(rows, cs):
                    items[vi * nS + r] = c.astype(np.float16)
            log(f"curves fold {k}")
    pred = np.zeros((len(vnames), nS), np.float32)
    test_views = [vnames.index(v) for v in ("orig", "flip", "revflip")]
    for k in sorted(np.unique(folds)):
        tr_s = np.where(folds != k)[0]
        rng = np.random.default_rng(k)
        val_s = rng.choice(tr_s, len(tr_s) // 10, replace=False)
        fit_s = np.setdiff1d(tr_s, val_s)
        fit = np.concatenate([vi * nS + fit_s for vi in range(len(vnames))])
        val = np.concatenate([vi * nS + val_s for vi in test_views])
        torch.manual_seed(k)
        net = CurveNet().to(dev)
        steps = a.epochs * math.ceil(len(fit) / 256)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.1)
        best, best_state = -1, None
        for ep in range(a.epochs):
            net.train()
            for b, x, v in batches(items, fit, 256, rng):
                yy = torch.from_numpy(y[b % nS]).to(dev)
                loss = nn.functional.binary_cross_entropy_with_logits(net(x.to(dev).float(), v.to(dev)), yy)
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step()
            net.eval(); pv = np.zeros(len(val))
            with torch.no_grad():
                pos = {j: q for q, j in enumerate(val)}
                for b, x, v in batches(items, val, 512):
                    o = net(x.to(dev).float(), v.to(dev)).cpu().numpy()
                    for j, oo in zip(b, o):
                        pv[pos[j]] = oo
            auc = roc_auc_score(np.tile(y[val_s], len(test_views)), pv)
            log(f"fold {k} epoch {ep}: val AUC {auc:.4f}")
            if auc > best:
                best, best_state = auc, {kk: vv.clone() for kk, vv in net.state_dict().items()}
        net.load_state_dict(best_state); net.eval()
        torch.save(best_state, a.out / f"curvenet_fold{k}.pt")
        rows = np.where(folds == k)[0]
        allidx = np.concatenate([vi * nS + rows for vi in range(len(vnames))])
        with torch.no_grad():
            for b, x, v in batches(items, allidx, 512):
                o = net(x.to(dev).float(), v.to(dev)).cpu().numpy()
                pred[b // nS, b % nS] = o
        log(f"fold {k}: OOF AUC (3-view mean) {roc_auc_score(y[rows], pred[test_views][:, rows].mean(0)):.4f}")
    p3 = pred[test_views].mean(0)
    log(f"curve model OOF AUC (orig/flip/revflip mean): {roc_auc_score(y, p3):.4f}")
    for vi, v in enumerate(vnames):
        pd.DataFrame({"cvm_logit": pred[vi]}, index=pd.Index(ids, name="id")).to_parquet(a.out / f"nn_features_{v}.parquet")
    log("done")


if __name__ == "__main__":
    main()
