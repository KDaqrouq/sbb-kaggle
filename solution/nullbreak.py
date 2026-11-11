"""Learned break model: H1 and H0 networks for "post segment given pre segment".

The no-break networks test the post segment against an unrestricted alternative. The most
powerful test (Neyman-Pearson) instead compares two explicit models of the post segment
given the pre regime:
  H1 network: fine-tuned only on BREAK series (labels 1) -> p(post | pre, break)
  H0 network: fine-tuned only on NO-BREAK series        -> p(post | pre, no break)
Both start from the fold's trained no-break network (nullnet2.py) and are fine-tuned on
(pre context, post chunk) pairs of the training folds, with sign flips and time reversal
(reversed post-then-pre is also a valid pair). log p_H1 - log p_H0 over the post segment
is a learned likelihood ratio against the ACTUAL distribution of break types and sizes.
Cross-fitted: series in fold k are scored by networks fine-tuned without fold k.

Train one side (run H1 and H0 on different GPUs), then score that side:
    python solution/nullbreak.py --base solution/work_nn2m16l64XL --mix 16 --lags 64 --label 1 --out solution/work_h1
    python solution/nullbreak.py --base solution/work_nn2m16l64XL --mix 16 --lags 64 --label 0 --out solution/work_h0
Combine (LR features for the trees):
    python solution/nullbreak.py --combine solution/work_h1 solution/work_h0 --out solution/work_h10
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nullnet as N  # noqa: E402
import nullnet2 as M  # noqa: E402

NAMES = ["ll_all", "ll_early", "ll_late"]


def finetune(net, pairs, seed, steps, dev, log, lr=5e-4):
    """pairs: list of (P, Q). Samples a post chunk (up to TGT points) with the pre segment as
    the regime context; lags may reach back into P."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    B, TGT, L = 256, N.TGT, N.LAGS
    ok = [i for i, (P, Q) in enumerate(pairs) if len(P) >= L + 16 and len(Q) >= 16]
    t0, run = time.time(), 0.0
    net.train()
    for step in range(steps):
        ctx, lag, tgt, tm = [], [], [], []
        for i in rng.choice(ok, B):
            P, Q = pairs[i]
            if rng.random() < 0.5:
                P, Q = -P, -Q
            if rng.random() < 0.5:
                P, Q = Q[::-1].copy(), P[::-1].copy()
                if len(P) < L + 16:
                    P, Q = pairs[i]
            x = np.concatenate((P, Q))
            mu, sc = N.med_scale(P)
            z = ((x - mu) / sc).astype(np.float32)
            nP = len(P)
            ctx.append(z[:nP])
            # post chunk: start anywhere in the post segment (early chunks matter most)
            a = nP + int(rng.integers(0, max(1, len(Q) - 16)))
            t = min(TGT, len(x) - a)
            yy = np.zeros(TGT, np.float32); yy[:t] = z[a:a + t]
            lg = np.zeros((TGT, L), np.float32)
            lg[:t] = np.lib.stride_tricks.sliding_window_view(z[a - L:a + t - 1], L)[:t][:, ::-1]
            m = np.zeros(TGT, np.float32); m[:t] = 1
            lag.append(lg); tgt.append(yy); tm.append(m)
        c, cm = N.pad_left(ctx, N.CTX)
        lp = net.logp(net.encode(torch.from_numpy(c).to(dev), torch.from_numpy(cm).to(dev)),
                      torch.from_numpy(np.stack(lag)).to(dev), torch.from_numpy(np.stack(tgt)).to(dev))
        tmt = torch.from_numpy(np.stack(tm)).to(dev)
        loss = -(lp * tmt).sum() / tmt.sum()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step else loss.item()
        if step % 1000 == 0 or step == steps - 1:
            log(f"  step {step} nll {run:.4f} ({time.time() - t0:.0f}s)")
    return net.eval()


def score(net, pairs, dev):
    """Mean log density (pre units) of the post segment given the pre regime: all, first 64
    post points, the rest."""
    tasks = []
    for P, Q in pairs:
        x = np.concatenate((P, Q))
        nP, n = len(P), len(P) + len(Q)
        e = min(n, nP + 64)
        tasks += [(P, x, nP, n), (P, x, nP, e), (P, x, e, max(e + 1, n))]
    r = N.score_tasks(net, tasks, dev).reshape(len(pairs), 3)
    nQ = np.array([len(q) for _, q in pairs], float)
    r[nQ <= 64, 2] = r[nQ <= 64, 0]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path)
    ap.add_argument("--mix", type=int, default=16)
    ap.add_argument("--lags", type=int, default=64)
    ap.add_argument("--label", type=int, choices=[0, 1])
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--combine", nargs=2, type=Path)
    ap.add_argument("--prefix", default="lb_")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    views_names = ["orig", "flip", "rev", "revflip"] + [f"crop{r}_{v}" for r in range(2) for v in ("orig", "flip", "rev", "revflip")]
    if a.combine:
        for v in views_names:
            h1 = pd.read_parquet(a.combine[0] / f"ll_{v}.parquet")
            h0 = pd.read_parquet(a.combine[1] / f"ll_{v}.parquet").reindex(h1.index)
            n = h1["nQ"].to_numpy()
            out = pd.DataFrame(index=h1.index)
            for c in NAMES:
                out[f"{a.prefix}h1_{c}"] = h1[c]
                out[f"{a.prefix}h0_{c}"] = h0[c]
                out[f"{a.prefix}lr_{c}"] = h1[c] - h0[c]
            out[f"{a.prefix}lr_sum"] = n * (h1["ll_all"] - h0["ll_all"])
            out.to_parquet(a.out / f"nn_features_{v}.parquet")
        print("combined", a.out)
        return
    M.MIX, N.LAGS = a.mix, a.lags
    dev = "cuda"
    log = lambda m: print(m, flush=True)
    ids, views = N.view_pairs(pd.read_parquet("train_series.parquet"))
    folds = pd.read_csv("solution/work/folds.csv").set_index("id").fold.reindex(ids).to_numpy()
    y = pd.read_csv("train_labels.csv").set_index("id").label.reindex(ids).to_numpy()
    res = {v: np.zeros((len(ids), 3)) for v in views}
    for k in sorted(np.unique(folds)):
        net = M.make_net().to(dev)
        net.load_state_dict(torch.load(a.base / f"nullnet_fold{k}.pt", map_location=dev))
        tr = np.where((folds != k) & (y == a.label))[0]
        log(f"fold {k}: fine-tuning on {len(tr)} series with label {a.label}")
        net = finetune(net, [views["orig"][i] for i in tr], 500 + k, a.steps, dev, log)
        torch.save(net.state_dict(), a.out / f"break{a.label}_fold{k}.pt")
        rows = np.where(folds == k)[0]
        for v, pairs in views.items():
            res[v][rows] = score(net, [pairs[i] for i in rows], dev)
        log(f"fold {k} scored")
    for v, r in res.items():
        nQ = np.array([len(q) for _, q in views[v]], float)
        df = pd.DataFrame(r, index=pd.Index(ids, name="id"), columns=NAMES)
        df["nQ"] = nQ
        df.to_parquet(a.out / f"ll_{v}.parquet")
    log(f"done {a.out}")


if __name__ == "__main__":
    main()
