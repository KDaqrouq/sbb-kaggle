"""Series-level parametric bootstrap with the learned no-break network as the simulator.

For each real training series the fold network gives a regime fingerprint for its pre
segment (hP) and for its post segment (hQ). A replica is generated point by point: the pre
part from hP, then the post part from hQ, keeping the real series' lengths, its post/pre
level and scale offsets, and its label. Breaks therefore have the real size and type
distribution, and no-break replicas carry the same estimation noise as real ones.

Used by deepsim.py (fresh replicas on the fly) and for the real-vs-synthetic check:
    python solution/simgen.py --fold 0 --n 6000 --out solution/work_sim/sim_f0.parquet
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "solution"))  # pipeline modules
import nullnet as N  # noqa: E402

BURN = 64


class Simulator:
    def __init__(self, net, pairs, dev):
        """pairs: list of (P, Q) raw arrays (the series this simulator may copy)."""
        self.net, self.dev = net, dev
        self.pairs = pairs
        ctxP, ctxQ, st = [], [], []
        for P, Q in pairs:
            mp, sp = N.med_scale(P)
            mq, sq = N.med_scale(Q)
            ctxP.append(((P - mp) / sp).astype(np.float32))
            ctxQ.append(((Q - mq) / sq).astype(np.float32))
            st.append((mp, sp, mq, sq))
        self.st = np.array(st)
        self.seed = np.stack([c[:N.LAGS] for c in ctxP])          # first lags of the real pre segment
        with torch.no_grad():
            self.hP = self._enc(ctxP)
            self.hQ = self._enc([c[:N.CTX][::-1].copy() for c in ctxQ])   # as in nullfeat3: ends at the boundary

    def _enc(self, segs):
        out = []
        for s in range(0, len(segs), 1024):
            c, m = N.pad_left(segs[s:s + 1024], N.CTX)
            out.append(self.net.encode(torch.from_numpy(c).to(self.dev), torch.from_numpy(m).to(self.dev)))
        return torch.cat(out)

    @torch.no_grad()
    def _run(self, h, lags, T):
        """Generate T points; lags (B, LAGS) most recent first; returns (B, T)."""
        out = torch.empty((h.shape[0], T), device=self.dev)
        lg = lags.clone()
        for t in range(T):
            o = self.net.dec(torch.cat((lg, lg.abs(), h), 1))
            mu, ls, rn = o[:, 0], o[:, 1].clamp(-6, 4), o[:, 2]
            nu = 2.2 + torch.nn.functional.softplus(rn)
            x = mu + ls.exp() * torch.distributions.StudentT(nu).sample()
            x = x.clamp(-50, 50)
            out[:, t] = x
            lg = torch.cat((x.unsqueeze(1), lg[:, :-1]), 1)
        return out

    def sample(self, idx, rng):
        """Replicas of series idx -> list of (P, Q) arrays in the pre segment's units."""
        idx = np.asarray(idx)
        nP = np.array([len(self.pairs[i][0]) for i in idx])
        nQ = np.array([len(self.pairs[i][1]) for i in idx])
        st = self.st[idx]
        seed = torch.from_numpy(self.seed[idx][:, ::-1].copy()).to(self.dev)
        h = self.hP[idx]
        pre = self._run(h, seed, BURN + nP.max())
        # post: lags = last generated pre points of each series, converted into post units
        Tq = nQ.max()
        lag = torch.stack([pre[i, BURN + nP[i] - N.LAGS:BURN + nP[i]].flip(0) for i in range(len(idx))])
        mp, sp, mq, sq = (torch.tensor(st[:, j], dtype=torch.float32, device=self.dev).unsqueeze(1) for j in range(4))
        lag_q = (lag * sp + mp - mq) / sq
        post = self._run(self.hQ[idx], lag_q, Tq)
        post_pre_units = (post * sq + mq - mp) / sp
        pre, post_pre_units = pre.cpu().numpy(), post_pre_units.cpu().numpy()
        return [(pre[i, BURN:BURN + nP[i]].astype(np.float64), post_pre_units[i, :nQ[i]].astype(np.float64))
                for i in range(len(idx))]


def load_fold(work, k, dev):
    net = N.Net().to(dev)
    net.load_state_dict(torch.load(work / f"nullnet_fold{k}.pt", map_location=dev))
    return net.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--work", type=Path, default=Path("solution/work_nn"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    ids, views = N.view_pairs(pd.read_parquet("train_series.parquet"), crops=False)
    folds = pd.read_csv("solution/work/folds.csv").set_index("id").fold.reindex(ids).to_numpy()
    y = pd.read_csv("train_labels.csv").set_index("id").label.reindex(ids).to_numpy()
    tr = np.where(folds != a.fold)[0]
    sim = Simulator(load_fold(a.work, a.fold, dev), [views["orig"][i] for i in tr], dev)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(tr), a.n, replace=False)
    rows = []
    for s in range(0, len(pick), 1000):
        reps = sim.sample(pick[s:s + 1000], rng)
        for j, (P, Q) in zip(pick[s:s + 1000], reps):
            sid = f"sim_{ids[tr[j]]}"
            x = np.concatenate((P, Q))
            rows.append(pd.DataFrame({"id": sid, "time": np.arange(len(x)), "value": x,
                                      "period": (np.arange(len(x)) >= len(P)).astype(np.int8)}))
    tab = pd.concat(rows, ignore_index=True)
    tab.to_parquet(a.out)
    pd.DataFrame({"id": [f"sim_{ids[tr[j]]}" for j in pick], "source": [ids[tr[j]] for j in pick],
                  "label": y[tr[pick]]}).to_csv(a.out.with_suffix(".labels.csv"), index=False)
    print("wrote", a.out, tab.id.nunique())


if __name__ == "__main__":
    main()
