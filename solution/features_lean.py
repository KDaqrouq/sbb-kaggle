"""Lean version of features._series_features for fast CPU inference.

Same feature values and the same column layout as features.FEATURE_NAMES, but blocks whose
columns the model does not use are skipped (their columns are left at 0). The kept blocks run
the identical code from features.py, so their values match exactly.

Which blocks are skipped is fixed in KEEP below (decided by the CPU-cost / CV ablation; L1 config:
only the moment scans hx8/hxg/hxd are dropped on top of the never-used families).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F  # noqa: E402

# --- which optional blocks to compute (True = compute) -----------------------------------
KEEP_ARSCAN = True
KEEP_RB = True
KEEP_HIWH = True        # whitening orders 40/80 (wh40, wh80, whpre40, wh40a)
KEEP_WSCAN = True       # whitened split scans ws8, ws8a, dws8
KEEP_GARCH = True       # GARCH-filtered battery on AR8 innovations (garch, gw, gwa, gpa, gws, gwsa, gchow)
KEEP_DGARCH = True      # same on detrended innovations (dg*)
KEEP_ADF = True         # ADF/KPSS persistence change
KEEP_HW = True          # moment Hotelling hw8/hgz/hdw
KEEP_HX = False         # moment scans hx8/hxg/hxd (40% of the handcrafted cost)
# always skipped: cw8, cw8a, hpa, tchow/ttchow/dchow, fchow (excluded from every model)

NAMES = F.FEATURE_NAMES


def _size(prefixes):
    return sum(1 for n in NAMES if n.split("_")[0] in prefixes)


S_ARSCAN = _size({"arscan"})
S_RB = _size({"rb"})
S_WH41 = 41
S_WSCAN1 = 28
S_GARCH = _size({"garch", "gw", "gwa", "gpa", "gws", "gwsa", "gchow"})
S_DGARCH = _size({"dg", "dgw", "dgwa", "dgpa", "dgws", "dgwsa", "dgchow"})
S_CHOWFAM = _size({"tchow1", "tchow5", "tchow8", "ttchow", "dchow"})
S_FCHOW = _size({"fchow2g", "fchow2t", "fchow5g", "fchow5t"})
S_ADF = _size({"adfc", "kpssc", "kimc", "adft", "kpsst", "kimt"})
S_HW1 = 4
S_HX1 = 4
_HIWH_SKIP = (40, 80)  # orders whose kind-0/1/2 blocks are dropped when KEEP_HIWH is False


@njit(cache=True)
def _series_features_lean(values, b):
    out = np.zeros(F.N_FEATURES)
    pre, post = F._standardise(values[:b], values[b:])
    n0 = pre.shape[0]
    n1 = post.shape[0]

    dpre, dpost = F._standardise(np.diff(pre), np.diff(post))
    apre, apost = F._standardise(np.abs(pre), np.abs(post))
    adpre, adpost = F._standardise(np.abs(np.diff(pre)), np.abs(np.diff(post)))
    cs = np.cumsum(np.concatenate((pre, post)))
    cpre, cpost = F._standardise(cs[:b], cs[b:])

    k = 0
    k = F._transform_block(pre, post, out, k)
    k = F._transform_block(dpre, dpost, out, k)
    k = F._transform_block(apre, apost, out, k)
    k = F._transform_block(adpre, adpost, out, k)
    k = F._transform_block(cpre, cpost, out, k)
    k = F._two_sample(pre, post, out, k)
    k = F._two_sample(dpre, dpost, out, k)
    k = F._two_sample(apre, apost, out, k)
    out[k] = F._mmd(pre, post)
    out[k + 1] = F._mmd(dpre, dpost)
    k += 2
    k = F._chow(pre, post, out, k)
    k = F._persistence(pre, post, out, k)
    for w in F.WINDOWS:
        wl = min(w, n0, n1)
        a = pre[n0 - wl:]
        c = post[:wl]
        out[k] = c.mean() - a.mean()
        out[k + 1] = np.median(c) - np.median(a)
        out[k + 2] = F._lratio(c.std(), a.std())
        k += 3
    wpre = np.minimum(np.maximum(pre, -F.WINSOR), F.WINSOR)
    wpost = np.minimum(np.maximum(post, -F.WINSOR), F.WINSOR)
    out[k] = wpost.mean() - wpre.mean()
    out[k + 1] = F._lratio(wpost.std(), wpre.std())
    _, dsc = F._med_scale(np.diff(pre))
    out[k + 2] = F._spike_rate(pre, 4.0 * dsc)
    out[k + 3] = F._spike_rate(post, 4.0 * dsc)
    k += 4
    x = np.concatenate((pre, post))
    k = F._cusum(x, b, out, k)
    xm = x - np.median(x)
    k = F._cusum(xm * xm, b, out, k)
    k = F._scan(x, b, out, k)
    if KEEP_ARSCAN:
        ax = np.abs(x)
        for t, p in F.AR_SCANS:
            k = F._ar_scan(ax if t == "abs" else x, b, p, out, k)
    else:
        k += S_ARSCAN
    k = F._cross_predict(pre, post, out, k)
    k = F._cross_predict(post, pre, out, k)
    k = F._whole_context(pre, post, out, k)
    k = F._diff_block(pre, post, out, k)
    if KEEP_RB:
        for p in F.ROBUST_ORDERS:
            k = F._robust_chow(pre, post, p, out, k)
    else:
        k += S_RB
    wpre, wpost = F._prewhiten(pre, post, 3)
    k = F._transform_block(wpre, wpost, out, k)
    k = F._two_sample(wpre, wpost, out, k)
    apre, apost = F._standardise(np.abs(wpre), np.abs(wpost))
    k = F._transform_block(apre, apost, out, k)
    k = F._two_sample(apre, apost, out, k)
    w8pre, w8post = F._prewhiten(pre, post, 8)
    k = F._transform_block(w8pre, w8post, out, k)
    k = F._two_sample(w8pre, w8post, out, k)
    dpre_w, dpost_w = F._prewhiten(np.diff(pre), np.diff(post), 3)
    k = F._transform_block(dpre_w, dpost_w, out, k)
    k = F._two_sample(dpre_w, dpost_w, out, k)
    ppre, ppost = F._prewhiten_pre(pre, post, 5)
    k = F._transform_block(ppre, ppost, out, k)
    k = F._two_sample(ppre, ppost, out, k)
    w20pre, w20post = F._prewhiten(pre, post, 20)
    k = F._transform_block(w20pre, w20post, out, k)
    k = F._two_sample(w20pre, w20post, out, k)
    for p_, kind in ((40, 0), (20, 1), (20, 2), (10, 3), (80, 0), (40, 1), (40, 2), (20, 4)):
        if (not KEEP_HIWH) and (p_ == 40 or p_ == 80):
            k += S_WH41
            continue
        if kind == 0:
            qa, qb = F._prewhiten(pre, post, p_)
        elif kind == 1:
            qa, qb = F._prewhiten_pre(pre, post, p_)
        elif kind == 2:
            ra, rb = F._prewhiten(pre, post, p_)
            qa, qb = F._standardise(np.abs(ra), np.abs(rb))
        elif kind == 4:
            qb, qa = F._prewhiten_pre(post, pre, p_)
        else:
            qa, qb = F._prewhiten(np.diff(pre), np.diff(post), p_)
        k = F._transform_block(qa, qb, out, k)
        k = F._two_sample(qa, qb, out, k)
    k = F._persistence(w8pre, w8post, out, k)
    va, vb = F._standardise(np.abs(w8pre), np.abs(w8post))
    k = F._persistence(va, vb, out, k)
    if KEEP_WSCAN:
        k = F._whitened_scan(np.concatenate((w8pre, w8post)), w8pre.shape[0], out, k)
        ab = np.abs(np.concatenate((w8pre, w8post)))
        k = F._whitened_scan(ab, w8pre.shape[0], out, k)
    else:
        k += 2 * S_WSCAN1
    tpre, tpost, tstat = F._detrend_pooled(pre, post)
    dpre8, dpost8 = F._prewhiten(tpre, tpost, 8)
    k = F._transform_block(dpre8, dpost8, out, k)
    k = F._two_sample(dpre8, dpost8, out, k)
    qa, qb = F._standardise(np.abs(dpre8), np.abs(dpost8))
    k = F._transform_block(qa, qb, out, k)
    k = F._two_sample(qa, qb, out, k)
    if KEEP_WSCAN:
        k = F._whitened_scan(np.concatenate((dpre8, dpost8)), dpre8.shape[0], out, k)
    else:
        k += S_WSCAN1
    out[k] = tstat
    k += 1
    k = F._trend_chow(pre, post, out, k)
    k += 3 * S_WH41                      # cw8, cw8a, hpa (never used)
    if KEEP_GARCH:
        k = F._garch_block(w8pre, w8post, out, k)
    else:
        k += S_GARCH
    k += S_CHOWFAM                       # tchow/ttchow/dchow (never used)
    k = F._tail_block(w8pre, w8post, out, k)
    e = np.concatenate((w8pre, w8post))
    ga, gb, gs2, gll, gll0 = F._garch_fit(e)
    hz = np.empty(e.shape[0])
    F._garch_ll(e, gs2, ga, gb, hz)
    z = e / np.sqrt(hz)
    k = F._tail_block(z[:w8pre.shape[0]].copy(), z[w8pre.shape[0]:].copy(), out, k)
    if KEEP_DGARCH:
        k = F._garch_block(dpre8, dpost8, out, k)
    else:
        k += S_DGARCH
    k = F._tail_block(dpre8, dpost8, out, k)
    e2 = np.concatenate((dpre8, dpost8))
    ga, gb, gs2, gll, gll0 = F._garch_fit(e2)
    hz2 = np.empty(e2.shape[0])
    F._garch_ll(e2, gs2, ga, gb, hz2)
    z2 = e2 / np.sqrt(hz2)
    k = F._tail_block(z2[:dpre8.shape[0]].copy(), z2[dpre8.shape[0]:].copy(), out, k)
    k += S_FCHOW                         # fchow (never used)
    if KEEP_ADF:
        k = F._persistence_change(pre, post, out, k)
    else:
        k += S_ADF
    if KEEP_HW or KEEP_HX:
        e3 = np.concatenate((w8pre, w8post))
        ga, gb, gs2, gll, gll0 = F._garch_fit(e3)
        hz3 = np.empty(e3.shape[0])
        F._garch_ll(e3, gs2, ga, gb, hz3)
        z3 = e3 / np.sqrt(hz3)
    else:
        z3 = np.zeros(1)
    if KEEP_HW:
        k = F._moment_block(w8pre, w8post, out, k)
        k = F._moment_block(z3[:w8pre.shape[0]].copy(), z3[w8pre.shape[0]:].copy(), out, k)
        k = F._moment_block(dpre8, dpost8, out, k)
    else:
        k += 3 * S_HW1
    if KEEP_HX:
        k = F._moment_scan(w8pre, w8post, out, k)
        k = F._moment_scan(z3[:w8pre.shape[0]].copy(), z3[w8pre.shape[0]:].copy(), out, k)
        k = F._moment_scan(dpre8, dpost8, out, k)
    else:
        k += 3 * S_HX1
    n = n0 + n1
    out[k] = n0
    out[k + 1] = n1
    out[k + 2] = n
    out[k + 3] = n0 / n
    return out


@njit(parallel=True, cache=True)
def _all_features_lean(values, starts, boundaries, lengths):
    m = starts.shape[0]
    out = np.empty((m, F.N_FEATURES))
    for i in prange(m):
        seg = values[starts[i]:starts[i] + lengths[i]]
        out[i] = _series_features_lean(seg, boundaries[i])
    return out


def views_lean(table, views):
    ids, starts, lengths, bnd, values = F._layout(table)
    rv = np.empty_like(values)
    for s, n in zip(starts, lengths):
        rv[s:s + n] = values[s:s + n][::-1]
    arrays = {"orig": (values, bnd), "flip": (-values, bnd), "rev": (rv, lengths - bnd), "revflip": (-rv, lengths - bnd)}
    return {v: F._frame(_all_features_lean(arrays[v][0], starts, arrays[v][1], lengths), ids) for v in views}
