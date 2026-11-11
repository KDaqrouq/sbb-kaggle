"""End-to-end deep break classifier on the raw series (GPU).

Siamese design: one encoder (dilated residual CNN + attention pooling) reads the pre and the
post segment separately, each standardised by the PRE segment's median/MAD (so level and
scale changes stay visible); the head compares the two embeddings. A third pass encodes the
pre segment's own two halves as a no-break reference. Training uses fresh random
augmentations every epoch: sign flip, time reversal and random crops of each side.

    python solution/deepnet.py --fold 0 --epochs 40          # one CV fold (quick check)
    python solution/deepnet.py --epochs 40                   # all folds -> OOF
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
import torch.nn.functional as Fn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "solution"))  # pipeline modules
import features as F  # noqa: E402

MAXSEG = 720


def med_scale(x):
    m = np.median(x)
    s = np.median(np.abs(x - m)) * 1.4826
    return m, (s if s > 1e-9 else x.std() + 1e-9)


class Res(nn.Module):
    def __init__(self, d, dil):
        super().__init__()
        self.c1 = nn.Conv1d(d, d, 5, padding=2 * dil, dilation=dil)
        self.c2 = nn.Conv1d(d, d, 1)
        self.n = nn.GroupNorm(8, d)

    def forward(self, x, m):
        return (x + self.c2(Fn.gelu(self.n(self.c1(x))))) * m


class Enc(nn.Module):
    def __init__(self, d, extra=0):
        super().__init__()
        self.extra = extra
        self.inp = nn.Conv1d(4 + extra, d, 7, padding=3)
        self.blocks = nn.ModuleList([Res(d, 2 ** (i % 6)) for i in range(10)])
        self.att = nn.Conv1d(d, 4, 1)

    def forward(self, x, m):
        """x (B, L) or (B, 2, L) [values, innovations], m (B, L) -> (B, 6d)."""
        e = None
        if x.dim() == 3:
            x, e = x[:, 0], x[:, 1]
        dx = torch.cat((torch.zeros_like(x[:, :1]), x[:, 1:] - x[:, :-1]), 1) * m
        ch = [x.clamp(-20, 20), x.abs().clamp(max=20), (x * x).clamp(max=25), dx.clamp(-20, 20)]
        if e is not None:
            ch += [e.clamp(-20, 20), e.abs().clamp(max=20)]
        inp = torch.stack(ch, 1)
        mm = m.unsqueeze(1)
        h = self.inp(inp) * mm
        for b in self.blocks:
            h = b(h, mm)
        a = self.att(h).masked_fill(mm == 0, -1e4).softmax(2)            # 4 attention heads
        pooled = torch.einsum("bkl,bdl->bkd", a, h).flatten(1)            # (B, 4d)
        cnt = mm.sum(2).clamp(min=1)
        mean = h.sum(2) / cnt
        std = ((((h - mean.unsqueeze(2)) ** 2) * mm).sum(2) / cnt).clamp(min=1e-6).sqrt()
        return torch.cat((pooled, mean, std), 1)


class Net(nn.Module):
    def __init__(self, d=96, extra=0):
        super().__init__()
        self.enc = Enc(d, extra)
        e = 6 * d
        self.head = nn.Sequential(nn.Linear(4 * e + 2, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 64), nn.GELU(),
                                  nn.Linear(64, 1))

    def forward(self, P, mP, Q, mQ, R1, m1, R2, m2, lens):
        eP, eQ = self.enc(P, mP), self.enc(Q, mQ)
        r1, r2 = self.enc(R1, m1), self.enc(R2, m2)
        return self.head(torch.cat((eQ - eP, eQ * eP, (eQ - eP).abs() - (r2 - r1).abs(), r2 - r1, lens), 1)).squeeze(1)


WHITEN = False


def innovations(p, q, order=8):
    """Pooled AR(order) innovations of the two (already standardised) segments, each
    filtered on its own; scaled by the pre innovations' robust sd."""
    def design(x):
        n = len(x)
        X = np.column_stack([np.ones(n - order)] + [x[order - j - 1:n - j - 1] for j in range(order)])
        return X, x[order:]
    Xa, ya = design(p)
    Xb, yb = design(q)
    c = np.linalg.lstsq(np.vstack((Xa, Xb)), np.concatenate((ya, yb)), rcond=None)[0]
    ea = np.concatenate((np.zeros(order), ya - Xa @ c))
    eb = np.concatenate((np.zeros(order), yb - Xb @ c))
    sc = np.median(np.abs(ea[order:])) * 1.4826 + 1e-9
    return ea / sc, eb / sc


def batchify(pairs, dev, rng=None, crop=False):
    Ps, Qs, R1s, R2s, lens = [], [], [], [], []
    for P, Q in pairs:
        if rng is not None:
            if rng.random() < 0.5:
                P, Q = -P, -Q
            if rng.random() < 0.5:
                P, Q = Q[::-1].copy(), P[::-1].copy()
            if crop and rng.random() < 0.6:
                kp = max(64, int(len(P) * rng.uniform(0.4, 1.0)))
                kq = max(64, int(len(Q) * rng.uniform(0.4, 1.0)))
                P, Q = P[-kp:], Q[:kq]
        mu, sc = med_scale(P)
        p, q = (P - mu) / sc, (Q - mu) / sc
        h = len(p) // 2
        if WHITEN:
            ea, eb = innovations(p, q)
            p2, q2 = np.stack((p, ea)), np.stack((q, eb))
            Ps.append(p2[:, -MAXSEG:]); Qs.append(q2[:, :MAXSEG]); R1s.append(p2[:, :h][:, -MAXSEG:]); R2s.append(p2[:, h:][:, :MAXSEG])
        else:
            Ps.append(p[-MAXSEG:]); Qs.append(q[:MAXSEG]); R1s.append(p[:h][-MAXSEG:]); R2s.append(p[h:][:MAXSEG])
        lens.append((math.log(len(P)) - 5.8, math.log(len(Q)) - 5.8))

    def pad(arrs, right):
        L = max(a.shape[-1] for a in arrs)
        lead = arrs[0].shape[:-1]
        X = np.zeros((len(arrs),) + lead + (L,), np.float32)
        M = np.zeros((len(arrs), L), np.float32)
        for i, a in enumerate(arrs):
            n = a.shape[-1]
            if right:              # post side: aligned at the boundary on the left
                X[i, ..., :n] = a; M[i, :n] = 1
            else:                  # pre side: aligned at the boundary on the right
                X[i, ..., L - n:] = a; M[i, L - n:] = 1
        return torch.from_numpy(X).to(dev), torch.from_numpy(M).to(dev)

    P, mP = pad(Ps, False)
    Q, mQ = pad(Qs, True)
    R1, m1 = pad(R1s, False)
    R2, m2 = pad(R2s, True)
    return P, mP, Q, mQ, R1, m1, R2, m2, torch.tensor(lens, dtype=torch.float32, device=dev)


def predict(net, pairs, dev):
    net.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(pairs), 256):
            chunk = pairs[s:s + 256]
            views = [chunk, [(-P, -Q) for P, Q in chunk], [(Q[::-1].copy(), P[::-1].copy()) for P, Q in chunk],
                     [(-Q[::-1].copy(), -P[::-1].copy()) for P, Q in chunk]]
            out.append(np.mean([torch.sigmoid(net(*batchify(v, dev))).cpu().numpy() for v in views], axis=0))
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=-1)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--whiten", action="store_true", help="add pooled AR(8) innovation channels")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", type=Path, default=Path("solution/work_deep"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    global WHITEN
    WHITEN = a.whiten
    dev = "cuda"
    torch.manual_seed(a.seed)
    t = pd.read_parquet("train_series.parquet")
    ids, starts, lengths, b, values = F._layout(t)
    pairs = [(values[s:s + k].copy(), values[s + k:s + n].copy()) for s, k, n in zip(starts, b, lengths)]
    y = pd.read_csv("train_labels.csv").set_index("id").label.reindex(ids).to_numpy().astype(np.float32)
    folds = pd.read_csv("solution/work/folds.csv").set_index("id").fold.reindex(ids).to_numpy()
    pred = np.full(len(ids), np.nan)
    t0 = time.time()
    for k in ([a.fold] if a.fold >= 0 else sorted(np.unique(folds))):
        rng = np.random.default_rng(a.seed + k)
        tr = np.where(folds != k)[0]
        rng.shuffle(tr)
        stop, fit = tr[:len(tr) // 10], tr[len(tr) // 10:]
        net = Net(a.dim, 2 if a.whiten else 0).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.05)
        steps = a.epochs * ((len(fit) + 127) // 128)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
        best, state = -1, None
        for ep in range(a.epochs):
            net.train()
            order = rng.permutation(fit)
            for s in range(0, len(order), 128):
                idx = order[s:s + 128]
                logit = net(*batchify([pairs[i] for i in idx], dev, rng, crop=True))
                loss = Fn.binary_cross_entropy_with_logits(logit, torch.from_numpy(y[idx]).to(dev))
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                sched.step()
            auc = roc_auc_score(y[stop], predict(net, [pairs[i] for i in stop], dev))
            if auc > best:
                best, state = auc, {kk: vv.detach().clone() for kk, vv in net.state_dict().items()}
            print(f"fold {k} ep {ep} loss {loss.item():.4f} stop-auc {auc:.4f} best {best:.4f} ({time.time() - t0:.0f}s)", flush=True)
        net.load_state_dict(state)
        te = np.where(folds == k)[0]
        pred[te] = predict(net, [pairs[i] for i in te], dev)
        torch.save(state, a.out / f"deep{a.tag}_fold{k}_s{a.seed}.pt")
        print(f"FOLD {k}: {roc_auc_score(y[te], pred[te]):.4f}", flush=True)
    ok = ~np.isnan(pred)
    pd.DataFrame({"id": np.asarray(ids)[ok], "deep": pred[ok], "label": y[ok]}).to_csv(
        a.out / f"oof_deep{a.tag}_f{a.fold}_s{a.seed}.csv", index=False)
    print(f"DEEP AUC {roc_auc_score(y[ok], pred[ok]):.4f} on {ok.sum()} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
