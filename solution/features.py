"""Before/after comparison features for KU-SBB.

Every feature is computed from one series on its own (no statistics are shared
across series), so feature extraction cannot leak information between CV folds.
Each series is first standardised by the robust location/scale of its own
pre-boundary segment, which makes all features scale-free.

Feature families (see README for the reasoning):
  * segment descriptors on five transformed copies (raw, first differences,
    absolute deviation, absolute differences, cumulative sum), turned into
    post-minus-pre differences and log post/pre ratios;
  * classical two-sample tests and distribution distances (Welch t,
    Mann-Whitney, KS, Cramer-von Mises, Wasserstein, energy, Brown-Forsythe,
    Gaussian-kernel MMD with median bandwidth);
  * AR(2) Chow-type likelihood ratios at the known boundary;
  * persistence: pre vs thirds / late half of post, early vs late post, and the
    pre segment against itself as a no-break reference;
  * local windows of +-7, 15, 30, 100 points around the boundary;
  * outlier-robust (winsorised) contrasts, spike rates and CUSUM statistics.
"""
import math

import numpy as np
import pandas as pd
from numba import njit, prange

EPS = 1e-9
TRIM = 0.1
WINDOWS = (7, 15, 30, 100)
MMD_POINTS = 128
MMD_BANDWIDTH_POINTS = 64
WINSOR = 3.0
SCAN_POINTS = 33
WSCAN_POINTS = 15
HIGHPASS_WINDOW = 25
WSCAN_STATS = ["ks", "wass", "energy", "cvm", "bf", "abst", "mw"]
N_WSCAN = len(WSCAN_STATS)
SCAN_STATS = ["mean", "logvar", "acf1", "absdev"]
N_SCAN = len(SCAN_STATS)
AR_SCANS = (("raw", 1), ("raw", 2), ("raw", 5), ("abs", 2))
CROSS_AR_ORDER = 5
WHOLE_STATS = ["std", "iqr", "ptp", "kurt", "skew", "acf1", "zcr", "absdiff"]
N_WHOLE = len(WHOLE_STATS)
ROBUST_ORDERS = (1, 3)
T_NU = 4.0

DESC_NAMES = ["mean", "median", "trim_mean", "q05", "q25", "q75", "q95", "std", "mad", "iqr",
              "skew", "kurt", "acf1", "acf2", "acf3", "acf4", "acf5", "resid_std", "slope",
              "zcr", "max_abs_z", "out_frac", "vr5", "vr20"]
N_DESC = len(DESC_NAMES)
LOCATION = ["mean", "median", "trim_mean", "q05", "q25", "q75", "q95"]
SCALE = ["std", "mad", "iqr", "resid_std"]
UNITLESS = ["skew", "kurt", "acf1", "acf2", "acf3", "acf4", "acf5", "zcr", "max_abs_z", "out_frac",
            "vr5", "vr20"]
CONTEXT = ["acf1", "kurt"]
TRANSFORMS = ["raw", "diff", "absdev", "absdiff", "cumsum"]
TEST_TRANSFORMS = ["raw", "diff", "absdev"]
TEST_NAMES = ["welch_t", "abs_t", "mw_z", "abs_mw_z", "ks", "ks_scaled", "ks_logp", "cvm",
              "wasserstein", "energy", "bf_logf"]

_LOC_IDX = np.array([DESC_NAMES.index(k) for k in LOCATION])
_SCALE_IDX = np.array([DESC_NAMES.index(k) for k in SCALE])
_UNITLESS_IDX = np.array([DESC_NAMES.index(k) for k in UNITLESS])
_CONTEXT_IDX = np.array([DESC_NAMES.index(k) for k in CONTEXT])
_SLOPE_IDX = DESC_NAMES.index("slope")


def feature_names():
    names = []
    for t in TRANSFORMS:
        names += [f"{t}_d_{k}" for k in LOCATION]
        names += [f"{t}_ad_mean", f"{t}_ad_median"]
        names += [f"{t}_lr_{k}" for k in SCALE]
        names += [f"{t}_d_{k}" for k in UNITLESS]
        names += [f"{t}_d_slope"]
        names += [f"{t}_{side}_{k}" for k in CONTEXT for side in ("pre", "post")]
    for t in TEST_TRANSFORMS:
        names += [f"{t}_{k}" for k in TEST_NAMES]
    names += ["mmd_raw", "mmd_diff"]
    names += ["chow_lr", "chow_lr_per_n", "mean_lr", "var_lr", "ar_phi1_pre", "ar_phi1_d",
              "ar_phi2_d"]
    for j in range(3):
        names += [f"third{j}_median", f"third{j}_lmad", f"third{j}_ks"]
    names += ["thirds_min_abs_shift", "thirds_max_abs_shift", "thirds_min_abs_lmad",
              "thirds_max_abs_lmad"]
    names += ["late_median", "late_lmad", "late_ks", "late_acf1_d"]
    names += ["post_halves_median_d", "post_halves_lmad", "post_halves_ks"]
    names += ["pre_halves_median_d", "pre_halves_lmad", "pre_halves_ks"]
    for w in WINDOWS:
        names += [f"win{w}_mean_d", f"win{w}_median_d", f"win{w}_lstd"]
    names += ["wins_mean_d", "wins_lstd", "spike_rate_pre", "spike_rate_post"]
    names += ["cusum_at_b", "cusum_max", "cusum_argmax_dist", "vcusum_at_b", "vcusum_max",
              "vcusum_argmax_dist"]
    for s in SCAN_STATS:
        names += [f"scan_{s}_at_b", f"scan_{s}_pct", f"scan_{s}_z", f"scan_{s}_ratio_max"]
    for t, p in AR_SCANS:
        c = f"arscan_{t}{p}"
        names += [f"{c}_lr", f"{c}_lr_per_n", f"{c}_coef_f", f"{c}_pct", f"{c}_z", f"{c}_ratio_max"]
    for direction in ("pre2post", "post2pre"):
        names += [f"x_{direction}_{k}" for k in ("lmse", "lmse_vs_self", "mean", "kurt", "ac1")]
    for s in WHOLE_STATS:
        names += [f"wh_{s}_whole", f"wh_{s}_contrib_l", f"wh_{s}_rtw_l", f"wh_{s}_rtw_r"]
    # the same strongest tests on the differenced series (for random-walk-like processes)
    names += ["d_chow_lr", "d_chow_lr_per_n", "d_mean_lr", "d_var_lr", "d_ar_phi1_pre", "d_ar_phi1_d", "d_ar_phi2_d"]
    for p in (1, 2):
        c = f"d_arscan{p}"
        names += [f"{c}_lr", f"{c}_lr_per_n", f"{c}_coef_f", f"{c}_pct", f"{c}_z", f"{c}_ratio_max"]
    for direction in ("pre2post", "post2pre"):
        names += [f"d_x_{direction}_{k}" for k in ("lmse", "lmse_vs_self", "mean", "kurt", "ac1")]
    names += ["d_cusum_at_b", "d_cusum_max", "d_cusum_argmax_dist"]
    for s in SCAN_STATS:
        names += [f"d_scan_{s}_at_b", f"d_scan_{s}_pct", f"d_scan_{s}_z", f"d_scan_{s}_ratio_max"]
    names += ["jump_z", "jump_z_rank"]
    for p in ROBUST_ORDERS:
        names += [f"rb_chow{p}", f"rb_chow{p}_n", f"rb_lscale{p}", f"rb_coef{p}", f"rb_mean{p}"]
    # classical two-sample battery on innovations of a pooled AR(3) fit (prewhitened)
    t = "wh3"
    names += [f"{t}_d_{k}" for k in LOCATION] + [f"{t}_ad_mean", f"{t}_ad_median"]
    names += [f"{t}_lr_{k}" for k in SCALE] + [f"{t}_d_{k}" for k in UNITLESS] + [f"{t}_d_slope"]
    names += [f"{t}_{side}_{k}" for k in CONTEXT for side in ("pre", "post")]
    names += [f"{t}_{k}" for k in TEST_NAMES]
    for t in ("wh3a", "wh8", "whd3", "whpre5", "wh20", "wh40", "whpre20", "wh20a", "whd10",
              "wh80", "whpre40", "wh40a", "whpost20"):
        names += [f"{t}_d_{k}" for k in LOCATION] + [f"{t}_ad_mean", f"{t}_ad_median"]
        names += [f"{t}_lr_{k}" for k in SCALE] + [f"{t}_d_{k}" for k in UNITLESS] + [f"{t}_d_slope"]
        names += [f"{t}_{side}_{k}" for k in CONTEXT for side in ("pre", "post")]
        names += [f"{t}_{k}" for k in TEST_NAMES]
    for t in ("wh8p", "wh8ap"):
        for j in range(3):
            names += [f"{t}_third{j}_median", f"{t}_third{j}_lmad", f"{t}_third{j}_ks"]
        names += [f"{t}_thirds_min_abs_shift", f"{t}_thirds_max_abs_shift", f"{t}_thirds_min_abs_lmad",
                  f"{t}_thirds_max_abs_lmad"]
        names += [f"{t}_late_median", f"{t}_late_lmad", f"{t}_late_ks", f"{t}_late_acf1_d"]
        names += [f"{t}_post_halves_median_d", f"{t}_post_halves_lmad", f"{t}_post_halves_ks"]
        names += [f"{t}_pre_halves_median_d", f"{t}_pre_halves_lmad", f"{t}_pre_halves_ks"]
    for t in ("ws8", "ws8a"):
        for st in WSCAN_STATS:
            names += [f"{t}_{st}_at_b", f"{t}_{st}_pct", f"{t}_{st}_z", f"{t}_{st}_ratio_mean"]
    # the same battery after removing one linear trend fitted to the whole series
    for t in ("dw8", "dw8a"):
        names += [f"{t}_d_{k}" for k in LOCATION] + [f"{t}_ad_mean", f"{t}_ad_median"]
        names += [f"{t}_lr_{k}" for k in SCALE] + [f"{t}_d_{k}" for k in UNITLESS] + [f"{t}_d_slope"]
        names += [f"{t}_{side}_{k}" for k in CONTEXT for side in ("pre", "post")]
        names += [f"{t}_{k}" for k in TEST_NAMES]
    for st in WSCAN_STATS:
        names += [f"dws8_{st}_at_b", f"dws8_{st}_pct", f"dws8_{st}_z", f"dws8_{st}_ratio_mean"]
    names += ["trend_t", "trendchow_lr", "trendchow_lr_per_n", "trend_slope_d"]
    # smooth (cubic) common trend removed, then whitened; and a high-pass copy for scale tests
    for t in ("cw8", "cw8a", "hpa"):
        names += [f"{t}_d_{k}" for k in LOCATION] + [f"{t}_ad_mean", f"{t}_ad_median"]
        names += [f"{t}_lr_{k}" for k in SCALE] + [f"{t}_d_{k}" for k in UNITLESS] + [f"{t}_d_slope"]
        names += [f"{t}_{side}_{k}" for k in CONTEXT for side in ("pre", "post")]
        names += [f"{t}_{k}" for k in TEST_NAMES]
    # volatility clustering removed: whitened innovations divided by a GARCH(1,1) volatility
    names += ["garch_alpha", "garch_persist", "garch_llgain"]
    for t in ("gw", "gwa", "gpa"):
        names += [f"{t}_d_{k}" for k in LOCATION] + [f"{t}_ad_mean", f"{t}_ad_median"]
        names += [f"{t}_lr_{k}" for k in SCALE] + [f"{t}_d_{k}" for k in UNITLESS] + [f"{t}_d_slope"]
        names += [f"{t}_{side}_{k}" for k in CONTEXT for side in ("pre", "post")]
        names += [f"{t}_{k}" for k in TEST_NAMES]
    for t in ("gws", "gwsa"):
        for st in WSCAN_STATS:
            names += [f"{t}_{st}_at_b", f"{t}_{st}_pct", f"{t}_{st}_z", f"{t}_{st}_ratio_mean"]
    names += ["gchow_lr", "gchow_lr_per_n", "gchow_alpha_d", "gchow_persist_d", "gchow_lvar"]
    # richer Chow likelihood ratios: AR(p) + trend at more orders, Student-t errors, and the
    # AR(2)+trend LR split into which part changed
    for p in (1, 5, 8):
        names += [f"tchow{p}_lr", f"tchow{p}_lr_per_n"]
    names += ["ttchow_lr", "ttchow_lr_per_n", "ttchow_lscale"]
    names += ["dchow_coef", "dchow_var", "dchow_level", "dchow_level_trend", "dchow_ar",
              "dchow_level_t", "dchow_trend_t"]
    # tail-shape change: Student-t degrees of freedom profiled separately per segment (each
    # segment keeps its own scale, so only the tail weight differs)
    for t in ("tl8", "tlg"):
        names += [f"{t}_lr", f"{t}_lr_per_n", f"{t}_lnu_d", f"{t}_lnu_pre"]
    # the GARCH filter and tail test again, on the detrended (dw8) innovations
    names += ["dg_garch_alpha", "dg_garch_persist", "dg_garch_llgain"]
    for t in ("dgw", "dgwa", "dgpa"):
        names += [f"{t}_d_{k}" for k in LOCATION] + [f"{t}_ad_mean", f"{t}_ad_median"]
        names += [f"{t}_lr_{k}" for k in SCALE] + [f"{t}_d_{k}" for k in UNITLESS] + [f"{t}_d_slope"]
        names += [f"{t}_{side}_{k}" for k in CONTEXT for side in ("pre", "post")]
        names += [f"{t}_{k}" for k in TEST_NAMES]
    for t in ("dgws", "dgwsa"):
        for st in WSCAN_STATS:
            names += [f"{t}_{st}_at_b", f"{t}_{st}_pct", f"{t}_{st}_z", f"{t}_{st}_ratio_mean"]
    names += ["dgchow_lr", "dgchow_lr_per_n", "dgchow_alpha_d", "dgchow_persist_d", "dgchow_lvar"]
    for t in ("tld8", "tldg"):
        names += [f"{t}_lr", f"{t}_lr_per_n", f"{t}_lnu_d", f"{t}_lnu_pre"]
    # full-model Chow: AR(p) + trend mean with GARCH(1,1) variance (Gaussian and t5 errors),
    # pooled vs separate fits of everything
    for p in (2, 5):
        for d in ("g", "t"):
            names += [f"fchow{p}{d}_lr", f"fchow{p}{d}_lr_per_n"]
    # persistence change (I(0) <-> I(1)): ADF t per segment, KPSS per segment, Kim ratio
    for d in ("c", "t"):
        names += [f"adf{d}_pre", f"adf{d}_post", f"adf{d}_d"]
        names += [f"kpss{d}_pre", f"kpss{d}_post", f"kpss{d}_lr"]
        names += [f"kim{d}_lr"]
    # joint moment-change test (Hotelling T2 on 8 moment functions of the innovations) and
    # jump-rate change (exceedances of 3 and 4 robust sd)
    for t in ("hw8", "hgz", "hdw"):
        names += [f"{t}_t2", f"{t}_t2_log", f"{t}_jump3_lr", f"{t}_jump4_lr"]
    # richer joint moment test (14 moments) at the boundary, and compared with other splits
    for t in ("hx8", "hxg", "hxd"):
        names += [f"{t}_t2_log", f"{t}_pct", f"{t}_z", f"{t}_ratio"]
    names += ["n_pre", "n_post", "n_total", "boundary_frac"]
    return names


FEATURE_NAMES = feature_names()
N_FEATURES = len(FEATURE_NAMES)


# ---------------------------------------------------------------- basic stats
@njit(cache=True)
def _q(s, q):
    """Linear-interpolation quantile of an already sorted array."""
    pos = q * (s.shape[0] - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, s.shape[0] - 1)
    w = pos - lo
    return s[lo] * (1.0 - w) + s[hi] * w


@njit(cache=True)
def _med_scale(a):
    """Median and robust scale (1.4826 * MAD, falling back to std, then 1)."""
    s = np.sort(a)
    med = _q(s, 0.5)
    mad = _q(np.sort(np.abs(a - med)), 0.5) * 1.4826
    if mad < EPS:
        mad = a.std()
    if mad < EPS:
        mad = 1.0
    return med, mad


@njit(cache=True)
def _acf(d, k, ss):
    if ss <= EPS or k >= d.shape[0]:
        return 0.0
    acc = 0.0
    for i in range(k, d.shape[0]):
        acc += d[i] * d[i - k]
    return acc / ss


@njit(cache=True)
def _vr(d, k, var):
    """Variance ratio: var of overlapping k-sums over k * var."""
    n = d.shape[0]
    if n <= k + 1 or var <= EPS:
        return 1.0
    c = np.zeros(n + 1)
    for i in range(n):
        c[i + 1] = c[i] + d[i]
    m = n - k + 1
    sums = np.empty(m)
    for i in range(m):
        sums[i] = c[i + k] - c[i]
    return sums.var() / (k * var)


@njit(cache=True)
def _describe(a):
    n = a.shape[0]
    out = np.zeros(N_DESC)
    s = np.sort(a)
    mean = a.mean()
    d = a - mean
    ss = (d * d).sum()
    var = ss / n
    std = math.sqrt(var)
    med = _q(s, 0.5)
    k = int(TRIM * n)
    mad = _q(np.sort(np.abs(a - med)), 0.5) * 1.4826
    out[0] = mean
    out[1] = med
    out[2] = s[k:n - k].mean()
    out[3] = _q(s, 0.05)
    out[4] = _q(s, 0.25)
    out[5] = _q(s, 0.75)
    out[6] = _q(s, 0.95)
    out[7] = std
    out[8] = mad
    out[9] = out[5] - out[4]
    if std > EPS:
        out[10] = (d ** 3).mean() / std ** 3
        out[11] = (d ** 4).mean() / var ** 2 - 3.0
    for lag in range(1, 6):
        out[11 + lag] = _acf(d, lag, ss)
    phi = out[12]
    r = 0.0
    for i in range(1, n):
        e = d[i] - phi * d[i - 1]
        r += e * e
    out[17] = math.sqrt(r / max(n - 1, 1))
    tbar = (n - 1) / 2.0
    num = 0.0
    den = 0.0
    for i in range(n):
        num += (i - tbar) * a[i]
        den += (i - tbar) ** 2
    out[18] = num / den if den > 0 else 0.0
    zc = 0
    for i in range(1, n):
        if (a[i] - med) * (a[i - 1] - med) < 0:
            zc += 1
    out[19] = zc / max(n - 1, 1)
    sc = mad if mad > EPS else (std if std > EPS else 1.0)
    mx = 0.0
    cnt = 0
    for i in range(n):
        z = abs(a[i] - med) / sc
        if z > mx:
            mx = z
        if z > 3.0:
            cnt += 1
    out[20] = mx
    out[21] = cnt / n
    out[22] = _vr(d, 5, var)
    out[23] = _vr(d, 20, var)
    return out


@njit(cache=True)
def _lratio(a, b):
    return math.log((a + EPS) / (b + EPS))


# ---------------------------------------------------------- two-sample tests
@njit(cache=True)
def _ecdf_stats(a, b):
    """KS, Cramer-von Mises, Wasserstein, energy distance and Mann-Whitney U.

    One merge pass over the two sorted samples.
    """
    sa = np.sort(a)
    sb = np.sort(b)
    na = sa.shape[0]
    nb = sb.shape[0]
    i = 0
    j = 0
    ks = 0.0
    cvm = 0.0
    wass = 0.0
    energy = 0.0
    u = 0.0  # number of (pre, post) pairs with pre < post
    prev = min(sa[0], sb[0])
    while i < na or j < nb:
        if j >= nb or (i < na and sa[i] < sb[j]):
            x = sa[i]
        elif i >= na or sb[j] < sa[i]:
            x = sb[j]
        else:
            x = sa[i]
        fa = i / na
        fb = j / nb
        gap = x - prev
        wass += abs(fa - fb) * gap
        energy += (fa - fb) ** 2 * gap
        while i < na and sa[i] == x:
            i += 1
        while j < nb and sb[j] == x:
            j += 1
        prev = x
        fa = i / na
        fb = j / nb
        diff = abs(fa - fb)
        if diff > ks:
            ks = diff
        cvm += (fa - fb) ** 2
    # Mann-Whitney U via a separate counting merge (continuous data, ties split).
    i = 0
    for j in range(nb):
        while i < na and sa[i] < sb[j]:
            i += 1
        k = i
        ties = 0
        while k < na and sa[k] == sb[j]:
            ties += 1
            k += 1
        u += i + 0.5 * ties
    cvm *= na * nb / (na + nb) ** 2
    return ks, cvm, wass, 2.0 * energy, u


@njit(cache=True)
def _ks_logp(d, na, nb):
    """log10 of the asymptotic two-sided KS p-value (Kolmogorov distribution)."""
    en = math.sqrt(na * nb / (na + nb))
    lam = (en + 0.12 + 0.11 / en) * d
    if lam < 0.2:
        return 0.0
    if lam > 3.0:
        return (math.log(2.0) - 2.0 * lam * lam) / math.log(10.0)
    p = 0.0
    for k in range(1, 101):
        p += 2.0 * (-1.0) ** (k - 1) * math.exp(-2.0 * k * k * lam * lam)
    p = min(max(p, 1e-300), 1.0)
    return math.log10(p)


@njit(cache=True)
def _brown_forsythe(a, b):
    """log of the Brown-Forsythe (median-centred Levene) F statistic."""
    za = np.abs(a - np.median(a))
    zb = np.abs(b - np.median(b))
    na = za.shape[0]
    nb = zb.shape[0]
    ma = za.mean()
    mb = zb.mean()
    m = (za.sum() + zb.sum()) / (na + nb)
    between = na * (ma - m) ** 2 + nb * (mb - m) ** 2
    within = ((za - ma) ** 2).sum() + ((zb - mb) ** 2).sum()
    f = between / (within / (na + nb - 2) + EPS)
    return math.log(f + 1e-6)


@njit(cache=True)
def _two_sample(a, b, out, k):
    na = a.shape[0]
    nb = b.shape[0]
    va = a.var() * na / (na - 1)
    vb = b.var() * nb / (nb - 1)
    t = (b.mean() - a.mean()) / math.sqrt(va / na + vb / nb + EPS)
    ks, cvm, wass, energy, u = _ecdf_stats(a, b)
    mu = na * nb / 2.0
    sd = math.sqrt(na * nb * (na + nb + 1) / 12.0)
    mwz = (u - mu) / sd
    out[k] = t
    out[k + 1] = abs(t)
    out[k + 2] = mwz
    out[k + 3] = abs(mwz)
    out[k + 4] = ks
    out[k + 5] = ks * math.sqrt(na * nb / (na + nb))
    out[k + 6] = _ks_logp(ks, na, nb)
    out[k + 7] = cvm
    out[k + 8] = wass
    out[k + 9] = energy
    out[k + 10] = _brown_forsythe(a, b)
    return k + 11


@njit(cache=True)
def _even_subsample(a, m):
    n = a.shape[0]
    if n <= m:
        return a.copy()
    out = np.empty(m)
    for i in range(m):
        out[i] = a[(i * (n - 1)) // (m - 1)]
    return out


@njit(cache=True)
def _mmd(a, b):
    """Biased MMD^2 with a Gaussian kernel, median-heuristic bandwidth.

    Deterministic evenly spaced subsamples keep the cost bounded.
    """
    x = _even_subsample(a, MMD_POINTS)
    y = _even_subsample(b, MMD_POINTS)
    bx = _even_subsample(x, MMD_BANDWIDTH_POINTS)
    by = _even_subsample(y, MMD_BANDWIDTH_POINTS)
    pooled = np.concatenate((bx, by))
    m = pooled.shape[0]
    dist = np.empty(m * (m - 1) // 2)
    c = 0
    for i in range(m):
        for j in range(i + 1, m):
            dist[c] = abs(pooled[i] - pooled[j])
            c += 1
    h = np.median(dist)
    if h < EPS:
        h = 1.0
    g = 1.0 / (2.0 * h * h)
    kxx = 0.0
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            kxx += math.exp(-g * (x[i] - x[j]) ** 2)
    kyy = 0.0
    for i in range(y.shape[0]):
        for j in range(y.shape[0]):
            kyy += math.exp(-g * (y[i] - y[j]) ** 2)
    kxy = 0.0
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            kxy += math.exp(-g * (x[i] - y[j]) ** 2)
    nx = x.shape[0]
    ny = y.shape[0]
    return kxx / (nx * nx) + kyy / (ny * ny) - 2.0 * kxy / (nx * ny)


# ------------------------------------------------------------ AR(2) Chow test
@njit(cache=True)
def _ar2_fit(a):
    """OLS of a[t] on (1, a[t-1], a[t-2]). Returns (XtX, Xty, yty, n, coef, rss)."""
    n = a.shape[0] - 2
    xtx = np.zeros((3, 3))
    xty = np.zeros(3)
    yty = 0.0
    row = np.empty(3)
    for t in range(2, a.shape[0]):
        row[0] = 1.0
        row[1] = a[t - 1]
        row[2] = a[t - 2]
        y = a[t]
        for p in range(3):
            xty[p] += row[p] * y
            for q in range(3):
                xtx[p, q] += row[p] * row[q]
        yty += y * y
    return xtx, xty, yty, n


@njit(cache=True)
def _solve_rss(xtx, xty, yty):
    reg = xtx + np.eye(3) * 1e-8
    coef = np.linalg.solve(reg, xty)
    rss = yty - 2.0 * (coef * xty).sum() + (coef * (xtx @ coef)).sum()
    return coef, max(rss, EPS)


@njit(cache=True)
def _chow(a, b, out, k):
    xa, ya, yya, na = _ar2_fit(a)
    xb, yb, yyb, nb = _ar2_fit(b)
    ca, ra = _solve_rss(xa, ya, yya)
    cb, rb = _solve_rss(xb, yb, yyb)
    cp, rp = _solve_rss(xa + xb, ya + yb, yya + yyb)
    n = na + nb
    lr = n * math.log(rp / n) - na * math.log(ra / na) - nb * math.log(rb / nb)
    out[k] = lr
    out[k + 1] = lr / n
    # mean-shift LR (common variance) and variance-change LR (separate means).
    ma = a.mean()
    mb = b.mean()
    m = (a.sum() + b.sum()) / (a.shape[0] + b.shape[0])
    ssa = ((a - ma) ** 2).sum()
    ssb = ((b - mb) ** 2).sum()
    sst = ((a - m) ** 2).sum() + ((b - m) ** 2).sum()
    nt = a.shape[0] + b.shape[0]
    out[k + 2] = nt * math.log((sst + EPS) / (ssa + ssb + EPS))
    va = ssa / a.shape[0] + EPS
    vb = ssb / b.shape[0] + EPS
    vp = (ssa + ssb) / nt + EPS
    out[k + 3] = nt * math.log(vp) - a.shape[0] * math.log(va) - b.shape[0] * math.log(vb)
    out[k + 4] = ca[1]
    out[k + 5] = cb[1] - ca[1]
    out[k + 6] = cb[2] - ca[2]
    return k + 7


# ------------------------------------------------------------ small contrasts
@njit(cache=True)
def _ks_only(a, b):
    ks, _, _, _, _ = _ecdf_stats(a, b)
    return ks


@njit(cache=True)
def _robust_pair(a, b):
    """Median shift and log robust-scale ratio of b relative to a."""
    ma, sa = _med_scale(a)
    mb, sb = _med_scale(b)
    return mb - ma, math.log(sb / sa)


@njit(cache=True)
def _spike_rate(a, thr):
    """Rate of isolated spikes: a big jump immediately reversed."""
    n = a.shape[0]
    cnt = 0
    for i in range(1, n - 1):
        d1 = a[i] - a[i - 1]
        d2 = a[i + 1] - a[i]
        if abs(d1) > thr and abs(d2) > thr and d1 * d2 < 0:
            cnt += 1
    return cnt / max(n - 2, 1)


@njit(cache=True)
def _cusum(x, b, out, k):
    n = x.shape[0]
    m = x.mean()
    sd = x.std()
    if sd < EPS:
        sd = 1.0
    c = 0.0
    best = 0.0
    arg = 0
    at_b = 0.0
    for i in range(n):
        c += x[i] - m
        v = abs(c) / (sd * math.sqrt(n))
        if v > best:
            best = v
            arg = i + 1
        if i + 1 == b:
            at_b = v
    out[k] = at_b
    out[k + 1] = best
    out[k + 2] = abs(arg - b) / n
    return k + 3


# ------------------------------------------------------------- split scanning
@njit(cache=True)
def _split_stats(c1, c2, c11, ca, l, m, r, out):
    """Change statistics for splitting [l, r) at m, from prefix sums.

    c1/c2: sums of x and x^2, c11: sums of x[t]*x[t-1] (t >= 1), ca: sums of |x - median|.
    Each statistic is scaled by sqrt(n_left * n_right / n) so values are comparable
    across split positions.
    """
    nl = m - l
    nr = r - m
    w = math.sqrt(nl * nr / (nl + nr))
    ml = (c1[m] - c1[l]) / nl
    mr = (c1[r] - c1[m]) / nr
    vl = max((c2[m] - c2[l]) / nl - ml * ml, EPS)
    vr = max((c2[r] - c2[m]) / nr - mr * mr, EPS)
    vp = ((c2[r] - c2[l]) / (nl + nr)) - ((c1[r] - c1[l]) / (nl + nr)) ** 2
    out[0] = abs(mr - ml) / math.sqrt(max(vp, EPS)) * w
    out[1] = abs(math.log(vr / vl)) * w
    al = ((c11[m] - c11[l + 1]) / (nl - 1) - ml * ml) / vl
    ar = ((c11[r] - c11[m + 1]) / (nr - 1) - mr * mr) / vr
    out[2] = abs(ar - al) * w
    dl = (ca[m] - ca[l]) / nl
    dr = (ca[r] - ca[m]) / nr
    out[3] = abs(math.log((dr + EPS) / (dl + EPS))) * w


@njit(cache=True)
def _scan(x, b, out, k):
    """How exceptional is the known boundary compared with other split points?

    Processes with volatility clustering or slow wander produce large before/after
    differences at many split points; a real break should stand out at b.
    """
    n = x.shape[0]
    med = np.median(x)
    c1 = np.zeros(n + 1)
    c2 = np.zeros(n + 1)
    c11 = np.zeros(n + 1)
    ca = np.zeros(n + 1)
    for i in range(n):
        c1[i + 1] = c1[i] + x[i]
        c2[i + 1] = c2[i] + x[i] * x[i]
        c11[i + 1] = c11[i] + (x[i] * x[i - 1] if i > 0 else 0.0)
        ca[i + 1] = ca[i] + abs(x[i] - med)
    ns = N_SCAN
    at_b = np.empty(ns)
    _split_stats(c1, c2, c11, ca, 0, b, n, at_b)
    grid = np.empty((SCAN_POINTS, ns))
    row = np.empty(ns)
    for j in range(SCAN_POINTS):
        m = int(round(n * (0.1 + 0.8 * j / (SCAN_POINTS - 1))))
        m = min(max(m, 8), n - 8)
        _split_stats(c1, c2, c11, ca, 0, m, n, row)
        grid[j] = row
    for s in range(ns):
        col = grid[:, s]
        out[k] = at_b[s]
        out[k + 1] = (col < at_b[s]).sum() / SCAN_POINTS
        sd = col.std()
        out[k + 2] = (at_b[s] - col.mean()) / (sd + 1e-6)
        out[k + 3] = at_b[s] / (col.max() + EPS)
        k += 4
    return k


@njit(cache=True)
def _ar_split_rss(cp, p, l, m, r):
    """RSS of AR(p)+intercept fits on rows [l, m) and [m, r), and of the pooled fit."""
    q = p + 1
    sl = cp[m] - cp[l]
    sr = cp[r] - cp[m]
    out = np.empty(3)
    for side in range(3):
        s = sl if side == 0 else (sr if side == 1 else sl + sr)
        xtx = s[:q, :q] + np.eye(q) * 1e-8
        xty = s[:q, q].copy()
        coef = np.linalg.solve(xtx, xty)
        rss = s[q, q] - 2.0 * (coef * xty).sum() + (coef * (s[:q, :q] @ coef)).sum()
        out[side] = max(rss, EPS)
    return out[0], out[1], out[2]


@njit(cache=True)
def _ar_scan(x, b, p, out, k):
    """AR(p) Chow likelihood ratio at the boundary, and how it ranks among other splits."""
    n = x.shape[0]
    q = p + 2
    rows = n - p
    cp = np.zeros((rows + 1, q, q))
    z = np.empty(q)
    for t in range(p, n):
        z[0] = 1.0
        for j in range(1, p + 1):
            z[j] = x[t - j]
        z[q - 1] = x[t]
        i = t - p
        for a in range(q):
            for c in range(q):
                cp[i + 1, a, c] = cp[i, a, c] + z[a] * z[c]
    nb = b - p
    ra, rb, rp = _ar_split_rss(cp, p, 0, nb, rows)
    na_ = nb
    nb_ = rows - nb
    lr_b = rows * math.log(rp / rows) - na_ * math.log(ra / na_) - nb_ * math.log(rb / nb_)
    dof = rows - 2 * (p + 1)
    out[k] = lr_b
    out[k + 1] = lr_b / rows
    out[k + 2] = ((rp - ra - rb) / (p + 1)) / ((ra + rb) / dof)
    grid = np.empty(SCAN_POINTS)
    for j in range(SCAN_POINTS):
        m = int(round(rows * (0.1 + 0.8 * j / (SCAN_POINTS - 1))))
        m = min(max(m, 3 * q), rows - 3 * q)
        sa, sb, sp = _ar_split_rss(cp, p, 0, m, rows)
        grid[j] = rows * math.log(sp / rows) - m * math.log(sa / m) - (rows - m) * math.log(sb / (rows - m))
    out[k + 3] = (grid < lr_b).sum() / SCAN_POINTS
    out[k + 4] = (lr_b - grid.mean()) / (grid.std() + 1e-6)
    out[k + 5] = lr_b / (grid.max() + EPS)
    return k + 6


# ------------------------------------- ideas from the ADIA 5th-place solution
@njit(cache=True)
def _ar_design(x, p):
    m = x.shape[0] - p
    X = np.empty((m, p + 1))
    for i in range(m):
        X[i, 0] = 1.0
        for j in range(p):
            X[i, j + 1] = x[p + i - j - 1]
    return X, x[p:].copy()


@njit(cache=True)
def _ar_residuals(x, p):
    X, y = _ar_design(x, p)
    coef = np.linalg.solve(X.T @ X + np.eye(p + 1) * 1e-8, X.T @ y)
    return coef, y - X @ coef


@njit(cache=True)
def _excess_kurt(r):
    d = r - r.mean()
    v = (d * d).mean()
    return (d ** 4).mean() / (v * v) - 3.0 if v > EPS else 0.0


@njit(cache=True)
def _cross_predict(src, dst, out, k):
    """Fit AR(5) on one segment and use it to predict the other one step ahead.

    If the process is unchanged, the source model predicts the destination about as
    well as the destination's own AR fit; a persistent change makes it worse.
    """
    p = CROSS_AR_ORDER
    coef, r_in = _ar_residuals(src, p)
    X, y = _ar_design(dst, p)
    r_out = y - X @ coef
    _, r_self = _ar_residuals(dst, p)
    v = r_in.var() + EPS
    mse = (r_out * r_out).mean()
    out[k] = math.log(mse / v)
    out[k + 1] = math.log(mse / (r_self.var() + EPS))
    out[k + 2] = r_out.mean() / math.sqrt(v)
    out[k + 3] = _excess_kurt(r_out) - _excess_kurt(r_in)
    d = r_out - r_out.mean()
    out[k + 4] = _acf(d, 1, (d * d).sum())
    return k + 5


@njit(cache=True)
def _wstats(x):
    """std, IQR, range, excess kurtosis, skew, lag-1 ACF, zero-crossing rate, mean |diff|."""
    out = np.zeros(N_WHOLE)
    n = x.shape[0]
    s = np.sort(x)
    d = x - x.mean()
    ss = (d * d).sum()
    var = ss / n
    sd = math.sqrt(var)
    out[0] = sd
    out[1] = _q(s, 0.75) - _q(s, 0.25)
    out[2] = s[n - 1] - s[0]
    if sd > EPS:
        out[3] = (d ** 4).mean() / var ** 2 - 3.0
        out[4] = (d ** 3).mean() / sd ** 3
    out[5] = _acf(d, 1, ss)
    zc = 0
    ad = 0.0
    for i in range(1, n):
        if (d[i] > 0) != (d[i - 1] > 0):
            zc += 1
        ad += abs(x[i] - x[i - 1])
    out[6] = zc / (n - 1)
    out[7] = ad / (n - 1)
    return out


@njit(cache=True)
def _whole_context(pre, post, out, k):
    """Each segment's statistic relative to the whole series: whole value, the pre
    segment's share of the two, and pre/whole and post/whole ratios."""
    l = _wstats(pre)
    r = _wstats(post)
    h = _wstats(np.concatenate((pre, post)))
    for i in range(l.shape[0]):
        out[k] = h[i]
        out[k + 1] = l[i] / (abs(l[i]) + abs(r[i]) + EPS)
        out[k + 2] = l[i] / (h[i] + EPS)
        out[k + 3] = r[i] / (h[i] + EPS)
        k += 4
    return k


# ------------------------------- differenced-series tests (random-walk-like)
@njit(cache=True)
def _diff_block(pre, post, out, k):
    """Chow, split-scan, cross-prediction and CUSUM tests on the increments.

    For near-unit-root processes the level-based tests are dominated by the wandering
    level; on the increments a drift change is a mean shift, a volatility change is a
    variance change, and a level jump at the boundary is one large increment.
    """
    dpre, dpost = _standardise(np.diff(pre), np.diff(post))
    k = _chow(dpre, dpost, out, k)
    d = np.concatenate((dpre, dpost))
    nb = dpre.shape[0]
    k = _ar_scan(d, nb, 1, out, k)
    k = _ar_scan(d, nb, 2, out, k)
    k = _cross_predict(dpre, dpost, out, k)
    k = _cross_predict(dpost, dpre, out, k)
    k = _cusum(d, nb, out, k)
    k = _scan(d, nb, out, k)
    # the increment across the boundary, against the pre increments' robust scale
    _, sc = _med_scale(np.diff(pre))
    jump = abs(post[0] - pre[pre.shape[0] - 1]) / sc
    out[k] = jump
    inc = np.abs(np.diff(pre)) / sc
    out[k + 1] = (inc < jump).sum() / inc.shape[0]
    return k + 2


# ------------------------------- outlier-robust Chow test (Student-t errors)
@njit(cache=True)
def _t_ar_fit(X, y, nu):
    """AR regression with Student-t errors by iteratively reweighted least squares.
    Returns log-likelihood, coefficients and scale^2."""
    n, q = X.shape
    w = np.ones(n)
    coef = np.zeros(q)
    s2 = 1.0
    r = y.copy()
    for _ in range(8):
        Xw = X * w.reshape(-1, 1)
        coef = np.linalg.solve(X.T @ Xw + np.eye(q) * 1e-8, Xw.T @ y)
        r = y - X @ coef
        s2 = max((w * r * r).sum() / n, 1e-12)
        w = (nu + 1.0) / (nu + r * r / s2)
    c = math.lgamma((nu + 1.0) / 2.0) - math.lgamma(nu / 2.0) - 0.5 * math.log(nu * math.pi * s2)
    ll = 0.0
    for i in range(n):
        ll += c - (nu + 1.0) / 2.0 * math.log1p(r[i] * r[i] / (nu * s2))
    return ll, coef, s2


@njit(cache=True)
def _robust_chow(pre, post, p, out, k):
    """Chow-type likelihood ratio for an AR(p) model with Student-t errors: a few
    outliers or a short burst barely move it, unlike the Gaussian version."""
    Xa, ya = _ar_design(pre, p)
    Xb, yb = _ar_design(post, p)
    la, ca, sa = _t_ar_fit(Xa, ya, T_NU)
    lb, cb, sb = _t_ar_fit(Xb, yb, T_NU)
    lp, _, _ = _t_ar_fit(np.concatenate((Xa, Xb)), np.concatenate((ya, yb)), T_NU)
    n = ya.shape[0] + yb.shape[0]
    lr = 2.0 * (la + lb - lp)
    out[k] = lr
    out[k + 1] = lr / n
    out[k + 2] = abs(0.5 * math.log(sb / sa))
    d = 0.0
    for j in range(1, p + 1):
        d += abs(cb[j] - ca[j])
    out[k + 3] = d
    out[k + 4] = abs(cb[0] / (1.0 - cb[1:].sum() + 1e-6) - ca[0] / (1.0 - ca[1:].sum() + 1e-6)) / math.sqrt(sa)
    return k + 5


# --------------------------------------------- prewhitening with a pooled AR fit
@njit(cache=True)
def _prewhiten(pre, post, p):
    """Innovations of an AR(p) fitted to both segments jointly (the no-break model),
    each segment filtered on its own so nothing crosses the boundary. Classical iid
    tests on these are valid for dependent processes, unlike tests on the levels.
    The order is capped at a third of the shorter segment."""
    p = max(1, min(p, min(pre.shape[0], post.shape[0]) // 3))
    Xa, ya = _ar_design(pre, p)
    Xb, yb = _ar_design(post, p)
    X = np.concatenate((Xa, Xb))
    y = np.concatenate((ya, yb))
    coef = np.linalg.solve(X.T @ X + np.eye(p + 1) * 1e-8, X.T @ y)
    ra = ya - Xa @ coef
    rb = yb - Xb @ coef
    return _standardise(ra, rb)


@njit(cache=True)
def _prewhiten_pre(pre, post, p):
    """Innovations of an AR(p) fitted on the pre segment only, applied to both segments."""
    p = max(1, min(p, min(pre.shape[0], post.shape[0]) // 3))
    Xa, ya = _ar_design(pre, p)
    Xb, yb = _ar_design(post, p)
    coef = np.linalg.solve(Xa.T @ Xa + np.eye(p + 1) * 1e-8, Xa.T @ ya)
    return _standardise(ya - Xa @ coef, yb - Xb @ coef)


# ------------------------------------------------------------- persistence
@njit(cache=True)
def _persistence(pre, post, out, k):
    """Pre vs each third and the late half of post (a lasting change moves every third),
    early vs late post, and the two pre halves as a no-break reference."""
    n0 = pre.shape[0]
    n1 = post.shape[0]
    shifts = np.empty(3)
    lmads = np.empty(3)
    for j in range(3):
        seg = post[(j * n1) // 3:((j + 1) * n1) // 3]
        s, l = _robust_pair(pre, seg)
        shifts[j] = s
        lmads[j] = l
        out[k] = s
        out[k + 1] = l
        out[k + 2] = _ks_only(pre, seg)
        k += 3
    out[k] = np.abs(shifts).min()
    out[k + 1] = np.abs(shifts).max()
    out[k + 2] = np.abs(lmads).min()
    out[k + 3] = np.abs(lmads).max()
    k += 4
    late = post[n1 // 2:]
    s, l = _robust_pair(pre, late)
    out[k] = s
    out[k + 1] = l
    out[k + 2] = _ks_only(pre, late)
    dl = late - late.mean()
    dp = pre - pre.mean()
    out[k + 3] = _acf(dl, 1, (dl * dl).sum()) - _acf(dp, 1, (dp * dp).sum())
    k += 4
    early = post[:n1 // 2]
    s, l = _robust_pair(early, late)
    out[k] = s
    out[k + 1] = l
    out[k + 2] = _ks_only(early, late)
    k += 3
    p1 = pre[:n0 // 2]
    p2 = pre[n0 // 2:]
    s, l = _robust_pair(p1, p2)
    out[k] = s
    out[k + 1] = l
    out[k + 2] = _ks_only(p1, p2)
    k += 3
    return k


# ------------------------------- whitened tests normalised by other split points
@njit(cache=True)
def _split_tests(e, m, out):
    a = e[:m]
    b = e[m:]
    na = a.shape[0]
    nb = b.shape[0]
    ks, cvm, wass, energy, u = _ecdf_stats(a, b)
    scale = math.sqrt(na * nb / (na + nb))
    out[0] = ks * scale
    out[1] = wass * scale
    out[2] = energy * scale
    out[3] = cvm
    out[4] = _brown_forsythe(a, b)
    va = a.var() * na / (na - 1)
    vb = b.var() * nb / (nb - 1)
    out[5] = abs(b.mean() - a.mean()) / math.sqrt(va / na + vb / nb + EPS)
    mu = na * nb / 2.0
    out[6] = abs(u - mu) / math.sqrt(na * nb * (na + nb + 1) / 12.0)


@njit(cache=True)
def _whitened_scan(e, b, out, k):
    """Whitened-innovation tests at the known boundary compared with the same tests at
    other split points of the same series. Slow wander that AR whitening leaves behind
    inflates every split; a real break should stand out at the boundary."""
    n = e.shape[0]
    at_b = np.empty(N_WSCAN)
    _split_tests(e, b, at_b)
    grid = np.empty((WSCAN_POINTS, N_WSCAN))
    row = np.empty(N_WSCAN)
    for j in range(WSCAN_POINTS):
        m = int(round(n * (0.15 + 0.7 * j / (WSCAN_POINTS - 1))))
        m = min(max(m, 16), n - 16)
        _split_tests(e, m, row)
        grid[j] = row
    for s in range(N_WSCAN):
        col = grid[:, s]
        out[k] = at_b[s]
        out[k + 1] = (col < at_b[s]).sum() / WSCAN_POINTS
        out[k + 2] = (at_b[s] - col.mean()) / (col.std() + 1e-6)
        out[k + 3] = at_b[s] / (abs(col.mean()) + EPS)
        k += 4
    return k


# ------------------------------------------------------ deterministic trends
@njit(cache=True)
def _detrend_pooled(pre, post):
    """Remove one straight line fitted to the whole series (the no-break model); also
    return the t statistic of its slope."""
    x = np.concatenate((pre, post))
    n = x.shape[0]
    tm = (n - 1) / 2.0
    sxx = 0.0
    sxy = 0.0
    xm = x.mean()
    for i in range(n):
        sxx += (i - tm) ** 2
        sxy += (i - tm) * (x[i] - xm)
    b = sxy / sxx
    r = np.empty(n)
    for i in range(n):
        r[i] = x[i] - xm - b * (i - tm)
    se = math.sqrt(max((r * r).sum() / (n - 2), 1e-12) / sxx)
    nb = pre.shape[0]
    return r[:nb].copy(), r[nb:].copy(), b / se


@njit(cache=True)
def _ols_rss(X, y):
    q = X.shape[1]
    c = np.linalg.solve(X.T @ X + np.eye(q) * 1e-8, X.T @ y)
    r = y - X @ c
    return max((r * r).sum(), 1e-12), c


@njit(cache=True)
def _trend_chow(pre, post, out, k):
    """Chow likelihood ratio for AR(2) plus a linear time trend: pooled vs separate fits.
    Unlike the plain AR Chow test, a common trend through both segments is not a break."""
    n0 = pre.shape[0]
    x = np.concatenate((pre, post))
    n = x.shape[0]
    rows = n - 2
    X = np.empty((rows, 4))
    y = np.empty(rows)
    for i in range(rows):
        t = i + 2
        X[i, 0] = 1.0
        X[i, 1] = t / n
        X[i, 2] = x[t - 1]
        X[i, 3] = x[t - 2]
        y[i] = x[t]
    cut = n0 - 2
    rp, _ = _ols_rss(X, y)
    ra, ca = _ols_rss(X[:cut], y[:cut])
    rb, cb = _ols_rss(X[cut:], y[cut:])
    na = cut
    nb = rows - cut
    lr = rows * math.log(rp / rows) - na * math.log(ra / na) - nb * math.log(rb / nb)
    out[k] = lr
    out[k + 1] = lr / rows
    out[k + 2] = cb[1] - ca[1]
    return k + 3


@njit(cache=True)
def _trend_design(x, p, n0, extra):
    """Rows t = p..n-1 of [1, t/n, x[t-1..t-p]] plus, if extra, [post dummy, post dummy * t/n]."""
    n = x.shape[0]
    rows = n - p
    X = np.empty((rows, 2 + p + (2 if extra else 0)))
    y = np.empty(rows)
    for i in range(rows):
        t = i + p
        X[i, 0] = 1.0
        X[i, 1] = t / n
        for j in range(p):
            X[i, 2 + j] = x[t - j - 1]
        if extra:
            dm = 1.0 if t >= n0 else 0.0
            X[i, 2 + p] = dm
            X[i, 3 + p] = dm * t / n
        y[i] = x[t]
    return X, y


@njit(cache=True)
def _gauss_chow(X, y, cut):
    rows = y.shape[0]
    rp, _ = _ols_rss(X, y)
    ra, _ = _ols_rss(X[:cut], y[:cut])
    rb, _ = _ols_rss(X[cut:], y[cut:])
    return rows * math.log(rp / rows) - cut * math.log(ra / cut) - (rows - cut) * math.log(rb / (rows - cut))


@njit(cache=True)
def _chow_family(pre, post, out, k):
    n0 = pre.shape[0]
    x = np.concatenate((pre, post))
    cap = max(1, min(pre.shape[0], post.shape[0]) // 4)
    for p in (1, 5, 8):
        q = min(p, cap)
        X, y = _trend_design(x, q, n0, False)
        lr = _gauss_chow(X, y, n0 - q)
        out[k] = lr
        out[k + 1] = lr / y.shape[0]
        k += 2
    # Student-t errors, AR(2) + trend
    X, y = _trend_design(x, 2, n0, False)
    cut = n0 - 2
    rows = y.shape[0]
    la, _, sa = _t_ar_fit(X[:cut], y[:cut], T_NU)
    lb, _, sb = _t_ar_fit(X[cut:], y[cut:], T_NU)
    lp, _, _ = _t_ar_fit(X, y, T_NU)
    lr = 2.0 * (la + lb - lp)
    out[k] = lr
    out[k + 1] = lr / rows
    out[k + 2] = 0.5 * math.log(sb / sa)
    k += 3
    # decomposition of the Gaussian AR(2)+trend LR
    rp, _ = _ols_rss(X, y)
    ra, _ = _ols_rss(X[:cut], y[:cut])
    rb, _ = _ols_rss(X[cut:], y[cut:])
    nb = rows - cut
    coef_part = rows * math.log(rp / (ra + rb))            # coefficients change, common variance
    full = rows * math.log(rp / rows) - cut * math.log(ra / cut) - nb * math.log(rb / nb)
    out[k] = coef_part
    out[k + 1] = full - coef_part                           # variance change
    Xe, _ = _trend_design(x, 2, n0, True)
    r_lvl, c_lvl = _ols_rss(Xe[:, :5], y)                   # + level shift
    r_lt, c_lt = _ols_rss(Xe, y)                            # + level and trend shift
    out[k + 2] = rows * math.log(rp / r_lvl)
    out[k + 3] = rows * math.log(rp / r_lt)
    out[k + 4] = rows * math.log(r_lt / (ra + rb))          # what remains: AR dynamics
    s2 = r_lt / max(rows - 6, 1)
    XtX = Xe.T @ Xe + np.eye(6) * 1e-8
    inv = np.linalg.inv(XtX)
    out[k + 5] = c_lt[4] / math.sqrt(max(s2 * inv[4, 4], 1e-12))
    out[k + 6] = c_lt[5] / math.sqrt(max(s2 * inv[5, 5], 1e-12))
    return k + 7


@njit(cache=True)
def _detrend_poly(pre, post, deg):
    """Remove one polynomial (degree deg) fitted to the whole series: smooth wander shared
    by both segments is not a break."""
    x = np.concatenate((pre, post))
    n = x.shape[0]
    X = np.empty((n, deg + 1))
    for i in range(n):
        u = 2.0 * i / (n - 1) - 1.0
        v = 1.0
        for d in range(deg + 1):
            X[i, d] = v
            v *= u
    _, c = _ols_rss(X, x)
    r = x - X @ c
    nb = pre.shape[0]
    return r[:nb].copy(), r[nb:].copy()


@njit(cache=True)
def _highpass(x, w):
    """Subtract a centred moving average (window w) computed within the segment only."""
    n = x.shape[0]
    h = w // 2
    out = np.empty(n)
    c = np.zeros(n + 1)
    for i in range(n):
        c[i + 1] = c[i] + x[i]
    for i in range(n):
        lo = max(0, i - h)
        hi = min(n, i + h + 1)
        out[i] = x[i] - (c[hi] - c[lo]) / (hi - lo)
    return out


# ------------------------------------------------------ volatility clustering
_G_ALPHA = np.array([0.03, 0.06, 0.1, 0.15, 0.2, 0.3])
_G_PERSIST = np.array([0.5, 0.7, 0.8, 0.9, 0.95, 0.98])


@njit(cache=True)
def _garch_ll(e, s2, a, b, h):
    """Gaussian log-likelihood of GARCH(1,1) with variance targeting (omega = s2 (1-a-b));
    fills h with the conditional variances."""
    om = s2 * (1.0 - a - b)
    hv = s2
    ll = 0.0
    for t in range(e.shape[0]):
        if t > 0:
            hv = om + a * e[t - 1] * e[t - 1] + b * hv
        hv = max(hv, 1e-6 * s2)
        h[t] = hv
        ll -= 0.5 * (math.log(hv) + e[t] * e[t] / hv)
    return ll


@njit(cache=True)
def _garch_fit(e):
    """Grid MLE of GARCH(1,1) on e. Returns alpha, beta, s2, loglik, iid loglik."""
    s2 = max((e * e).mean(), 1e-9)
    h = np.empty(e.shape[0])
    ll0 = _garch_ll(e, s2, 0.0, 0.0, h)
    best = ll0
    ba = 0.0
    bb = 0.0
    for a in _G_ALPHA:
        for ps in _G_PERSIST:
            if ps <= a:
                continue
            ll = _garch_ll(e, s2, a, ps - a, h)
            if ll > best:
                best = ll
                ba = a
                bb = ps - a
    return ba, bb, s2, best, ll0


_NU_GRID = np.array([2.2, 3.0, 4.0, 6.0, 10.0, 20.0, 60.0])


@njit(cache=True)
def _t_scale_ll(r, nu):
    """Log-likelihood of centred data r under Student-t(nu) with its scale fitted (IRLS)."""
    n = r.shape[0]
    s2 = max((r * r).mean(), 1e-12)
    for _ in range(12):
        acc = 0.0
        for i in range(n):
            acc += (nu + 1.0) / (nu + r[i] * r[i] / s2) * r[i] * r[i]
        s2 = max(acc / n, 1e-12)
    c = math.lgamma((nu + 1.0) / 2.0) - math.lgamma(nu / 2.0) - 0.5 * math.log(nu * math.pi * s2)
    ll = 0.0
    for i in range(n):
        ll += c - (nu + 1.0) / 2.0 * math.log1p(r[i] * r[i] / (nu * s2))
    return ll


@njit(cache=True)
def _tail_block(a, b, out, k):
    ra = a - np.median(a)
    rb = b - np.median(b)
    m = _NU_GRID.shape[0]
    la = np.empty(m)
    lb = np.empty(m)
    for j in range(m):
        la[j] = _t_scale_ll(ra, _NU_GRID[j])
        lb[j] = _t_scale_ll(rb, _NU_GRID[j])
    ja = np.argmax(la)
    jb = np.argmax(lb)
    lr = 2.0 * (la[ja] + lb[jb] - (la + lb).max())
    out[k] = lr
    out[k + 1] = lr / (a.shape[0] + b.shape[0])
    out[k + 2] = math.log(_NU_GRID[jb] / _NU_GRID[ja])
    out[k + 3] = math.log(_NU_GRID[ja])
    return k + 4


@njit(cache=True)
def _garch_ll_any(e, s2, a, b, tnu):
    """GARCH(1,1) log-likelihood with variance targeting; Gaussian if tnu <= 0, else
    unit-variance Student-t(tnu) errors."""
    om = s2 * (1.0 - a - b)
    hv = s2
    ll = 0.0
    if tnu > 0:
        c = math.lgamma((tnu + 1.0) / 2.0) - math.lgamma(tnu / 2.0) - 0.5 * math.log((tnu - 2.0) * math.pi)
    for t in range(e.shape[0]):
        if t > 0:
            hv = om + a * e[t - 1] * e[t - 1] + b * hv
        hv = max(hv, 1e-6 * s2)
        if tnu > 0:
            ll += c - 0.5 * math.log(hv) - (tnu + 1.0) / 2.0 * math.log1p(e[t] * e[t] / ((tnu - 2.0) * hv))
        else:
            ll -= 0.5 * (math.log(hv) + e[t] * e[t] / hv)
    return ll


@njit(cache=True)
def _garch_best(e, tnu):
    s2 = max((e * e).mean(), 1e-9)
    best = _garch_ll_any(e, s2, 0.0, 0.0, tnu)
    for a in _G_ALPHA:
        for ps in _G_PERSIST:
            if ps <= a:
                continue
            ll = _garch_ll_any(e, s2, a, ps - a, tnu)
            if ll > best:
                best = ll
    return best


@njit(cache=True)
def _full_chow(pre, post, out, k):
    n0 = pre.shape[0]
    x = np.concatenate((pre, post))
    cap = max(1, min(pre.shape[0], post.shape[0]) // 4)
    for p in (2, 5):
        q = min(p, cap)
        X, y = _trend_design(x, q, n0, False)
        cut = n0 - q
        _, cp = _ols_rss(X, y)
        _, ca = _ols_rss(X[:cut], y[:cut])
        _, cb = _ols_rss(X[cut:], y[cut:])
        rp = y - X @ cp
        ra = y[:cut] - X[:cut] @ ca
        rb = y[cut:] - X[cut:] @ cb
        for tnu in (0.0, 5.0):
            lr = 2.0 * (_garch_best(ra, tnu) + _garch_best(rb, tnu) - _garch_best(rp, tnu))
            out[k] = lr
            out[k + 1] = lr / y.shape[0]
            k += 2
    return k


@njit(cache=True)
def _adf_t(x, trend, lags):
    """ADF t statistic of rho in dx_t = a (+ b t) + rho x_{t-1} + sum c_j dx_{t-j}."""
    n = x.shape[0]
    dx = np.diff(x)
    rows = n - 1 - lags
    q = 2 + (1 if trend else 0) + lags
    X = np.empty((rows, q))
    y = np.empty(rows)
    for i in range(rows):
        t = i + lags            # index into dx
        X[i, 0] = 1.0
        X[i, 1] = x[t]
        c = 2
        if trend:
            X[i, 2] = t / n
            c = 3
        for j in range(lags):
            X[i, c + j] = dx[t - j - 1]
        y[i] = dx[t]
    A = X.T @ X + np.eye(q) * 1e-8
    coef = np.linalg.solve(A, X.T @ y)
    r = y - X @ coef
    s2 = (r * r).sum() / max(rows - q, 1)
    inv = np.linalg.inv(A)
    return coef[1] / math.sqrt(max(s2 * inv[1, 1], 1e-12))


@njit(cache=True)
def _kpss_parts(x, trend):
    """Partial-sum numerator (sum S_t^2 / n^2) and Bartlett long-run variance of residuals
    from a mean (or mean + trend) fit."""
    n = x.shape[0]
    if trend:
        X = np.empty((n, 2))
        for i in range(n):
            X[i, 0] = 1.0
            X[i, 1] = i / n
        _, c = _ols_rss(X, x)
        e = x - X @ c
    else:
        e = x - x.mean()
    S = 0.0
    num = 0.0
    for i in range(n):
        S += e[i]
        num += S * S
    num /= n * n
    L = int(4.0 * (n / 100.0) ** 0.25)
    lrv = (e * e).sum() / n
    for l in range(1, L + 1):
        w = 1.0 - l / (L + 1.0)
        lrv += 2.0 * w * (e[l:] * e[:-l]).sum() / n
    return num, max(lrv, 1e-9)


@njit(cache=True)
def _persistence_change(pre, post, out, k):
    lags = 4
    for trend in (False, True):
        if min(pre.shape[0], post.shape[0]) > 3 * lags + 10:
            a = _adf_t(pre, trend, lags)
            b = _adf_t(post, trend, lags)
        else:
            a = 0.0
            b = 0.0
        out[k] = a
        out[k + 1] = b
        out[k + 2] = b - a
        na, la = _kpss_parts(pre, trend)
        nb, lb = _kpss_parts(post, trend)
        out[k + 3] = na / la
        out[k + 4] = nb / lb
        out[k + 5] = math.log((nb / lb + 1e-6) / (na / la + 1e-6))
        out[k + 6] = math.log((nb + 1e-9) / (na + 1e-9))   # Kim ratio (variance of partial sums)
        k += 7
    return k


@njit(cache=True)
def _moments_rows(z):
    n = z.shape[0]
    M = np.empty((n - 2, 8))
    for i in range(2, n):
        a = z[i]
        b = z[i - 1]
        c = z[i - 2]
        a2 = min(a * a, 25.0)
        M[i - 2, 0] = a
        M[i - 2, 1] = a2
        M[i - 2, 2] = abs(a)
        M[i - 2, 3] = a * b
        M[i - 2, 4] = a * c
        M[i - 2, 5] = a2 * min(b * b, 25.0)
        M[i - 2, 6] = max(-25.0, min(a * a * a, 25.0))
        M[i - 2, 7] = min(a2 * a2, 100.0)
    return M


@njit(cache=True)
def _binom_ll(k, n):
    if k <= 0 or k >= n:
        return 0.0
    p = k / n
    return k * math.log(p) + (n - k) * math.log(1.0 - p)


@njit(cache=True)
def _moment_block(a, b, out, k):
    """Hotelling T2 for a change in the joint mean of 8 moment functions (level, variance,
    |.|, lag-1/2 autocovariance, volatility clustering, skew, kurtosis) between segments."""
    Ma = _moments_rows(a)
    Mb = _moments_rows(b)
    na = Ma.shape[0]
    nb = Mb.shape[0]
    ma = np.zeros(8)
    mb = np.zeros(8)
    for j in range(8):
        ma[j] = Ma[:, j].mean()
        mb[j] = Mb[:, j].mean()
    S = np.zeros((8, 8))
    for i in range(na):
        d = Ma[i] - ma
        S += np.outer(d, d)
    for i in range(nb):
        d = Mb[i] - mb
        S += np.outer(d, d)
    S /= (na + nb - 2)
    d = mb - ma
    t2 = (na * nb) / (na + nb) * (d @ np.linalg.solve(S + np.eye(8) * 1e-6, d))
    out[k] = t2
    out[k + 1] = math.log1p(max(t2, 0.0))
    k += 2
    za = np.abs(a - np.median(a))
    zb = np.abs(b - np.median(b))
    sc = _q(np.sort(np.concatenate((za, zb))), 0.5) * 1.4826 + 1e-9
    for thr in (3.0, 4.0):
        ka = 0
        kb = 0
        for v in za:
            if v > thr * sc:
                ka += 1
        for v in zb:
            if v > thr * sc:
                kb += 1
        n1 = za.shape[0]
        n2 = zb.shape[0]
        out[k] = 2.0 * (_binom_ll(ka, n1) + _binom_ll(kb, n2) - _binom_ll(ka + kb, n1 + n2))
        k += 1
    return k


@njit(cache=True)
def _moments14(z):
    n = z.shape[0]
    M = np.empty((n - 3, 14))
    for i in range(3, n):
        a = z[i]
        b = z[i - 1]
        c = z[i - 2]
        d = z[i - 3]
        a2 = min(a * a, 25.0)
        b2 = min(b * b, 25.0)
        r = i - 3
        M[r, 0] = a
        M[r, 1] = a2
        M[r, 2] = abs(a)
        M[r, 3] = a * b
        M[r, 4] = a * c
        M[r, 5] = a * d
        M[r, 6] = a2 * b2
        M[r, 7] = max(-25.0, min(a * a * a, 25.0))
        M[r, 8] = min(a2 * a2, 100.0)
        M[r, 9] = abs(a) * abs(b)
        M[r, 10] = 1.0 if a * b > 0 else 0.0
        M[r, 11] = max(-25.0, min(a2 * b, 25.0))
        M[r, 12] = 1.0 if abs(a) > 2.0 else 0.0
        M[r, 13] = abs(a - b)
    return M


@njit(cache=True)
def _t2_split(M, m):
    q = M.shape[1]
    n = M.shape[0]
    ma = np.zeros(q)
    mb = np.zeros(q)
    for i in range(m):
        ma += M[i]
    for i in range(m, n):
        mb += M[i]
    ma /= m
    mb /= (n - m)
    S = np.zeros((q, q))
    for i in range(n):
        d = M[i] - (ma if i < m else mb)
        S += np.outer(d, d)
    S /= (n - 2)
    dd = mb - ma
    return (m * (n - m)) / n * (dd @ np.linalg.solve(S + np.eye(q) * 1e-6, dd))


@njit(cache=True)
def _moment_scan(a, b, out, k):
    z = np.concatenate((a, b))
    M = _moments14(z)
    n = M.shape[0]
    mb = a.shape[0] - 3
    tb = _t2_split(M, mb)
    grid = np.empty(WSCAN_POINTS)
    for j in range(WSCAN_POINTS):
        m = int(round(n * (0.15 + 0.7 * j / (WSCAN_POINTS - 1))))
        m = min(max(m, 20), n - 20)
        grid[j] = _t2_split(M, m)
    out[k] = math.log1p(max(tb, 0.0))
    out[k + 1] = (grid < tb).sum() / WSCAN_POINTS
    out[k + 2] = (tb - grid.mean()) / (grid.std() + 1e-6)
    out[k + 3] = tb / (grid.mean() + EPS)
    return k + 4


@njit(cache=True)
def _garch_block(epre, epost, out, k):
    e = np.concatenate((epre, epost))
    nb = epre.shape[0]
    n = e.shape[0]
    a, b, s2, ll, ll0 = _garch_fit(e)
    out[k] = a
    out[k + 1] = a + b
    out[k + 2] = (ll - ll0) / n
    k += 3
    h = np.empty(n)
    _garch_ll(e, s2, a, b, h)
    z = e / np.sqrt(h)
    za, zb = _standardise(z[:nb].copy(), z[nb:].copy())
    k = _transform_block(za, zb, out, k)
    k = _two_sample(za, zb, out, k)
    qa, qb = _standardise(np.abs(za), np.abs(zb))
    k = _transform_block(qa, qb, out, k)
    k = _two_sample(qa, qb, out, k)
    # parameters fitted on the pre segment only, then run through the whole series
    pa, pb, ps2, pll, pll0 = _garch_fit(epre)
    _garch_ll(e, ps2, pa, pb, h)
    zp = e / np.sqrt(h)
    qa, qb = _standardise(np.abs(zp[:nb]), np.abs(zp[nb:]))
    k = _transform_block(qa, qb, out, k)
    k = _two_sample(qa, qb, out, k)
    k = _whitened_scan(np.concatenate((za, zb)), nb, out, k)
    k = _whitened_scan(np.abs(np.concatenate((za, zb))), nb, out, k)
    qa_, qb_, qs2, qll, qll0 = _garch_fit(epost)
    lr = 2.0 * (pll + qll - ll)
    out[k] = lr
    out[k + 1] = lr / n
    out[k + 2] = qa_ - pa
    out[k + 3] = (qa_ + qb_) - (pa + pb)
    out[k + 4] = math.log(qs2 / ps2)
    return k + 5


# --------------------------------------------------------------- per series
@njit(cache=True)
def _standardise(pre, post):
    med, sc = _med_scale(pre)
    return (pre - med) / sc, (post - med) / sc


@njit(cache=True)
def _transform_block(pre, post, out, k):
    p = _describe(pre)
    q = _describe(post)
    for idx in _LOC_IDX:
        out[k] = q[idx] - p[idx]
        k += 1
    out[k] = abs(q[0] - p[0])
    out[k + 1] = abs(q[1] - p[1])
    k += 2
    for idx in _SCALE_IDX:
        out[k] = _lratio(q[idx], p[idx])
        k += 1
    for idx in _UNITLESS_IDX:
        out[k] = q[idx] - p[idx]
        k += 1
    n = pre.shape[0] + post.shape[0]
    out[k] = (q[_SLOPE_IDX] - p[_SLOPE_IDX]) * n
    k += 1
    for idx in _CONTEXT_IDX:
        out[k] = p[idx]
        out[k + 1] = q[idx]
        k += 2
    return k


@njit(cache=True)
def _series_features(values, b):
    out = np.zeros(N_FEATURES)
    pre, post = _standardise(values[:b], values[b:])
    n0 = pre.shape[0]
    n1 = post.shape[0]
    n = n0 + n1

    dpre, dpost = _standardise(np.diff(pre), np.diff(post))
    apre, apost = _standardise(np.abs(pre), np.abs(post))
    adpre, adpost = _standardise(np.abs(np.diff(pre)), np.abs(np.diff(post)))
    cs = np.cumsum(np.concatenate((pre, post)))
    cpre, cpost = _standardise(cs[:b], cs[b:])

    k = 0
    k = _transform_block(pre, post, out, k)
    k = _transform_block(dpre, dpost, out, k)
    k = _transform_block(apre, apost, out, k)
    k = _transform_block(adpre, adpost, out, k)
    k = _transform_block(cpre, cpost, out, k)

    k = _two_sample(pre, post, out, k)
    k = _two_sample(dpre, dpost, out, k)
    k = _two_sample(apre, apost, out, k)

    out[k] = _mmd(pre, post)
    out[k + 1] = _mmd(dpre, dpost)
    k += 2

    k = _chow(pre, post, out, k)

    k = _persistence(pre, post, out, k)

    # Local windows around the boundary.
    for w in WINDOWS:
        wl = min(w, n0, n1)
        a = pre[n0 - wl:]
        c = post[:wl]
        out[k] = c.mean() - a.mean()
        out[k + 1] = np.median(c) - np.median(a)
        out[k + 2] = _lratio(c.std(), a.std())
        k += 3

    # Outlier-robust contrasts: winsorise at +-WINSOR pre-robust-sd.
    wpre = np.minimum(np.maximum(pre, -WINSOR), WINSOR)
    wpost = np.minimum(np.maximum(post, -WINSOR), WINSOR)
    out[k] = wpost.mean() - wpre.mean()
    out[k + 1] = _lratio(wpost.std(), wpre.std())
    _, dsc = _med_scale(np.diff(pre))
    out[k + 2] = _spike_rate(pre, 4.0 * dsc)
    out[k + 3] = _spike_rate(post, 4.0 * dsc)
    k += 4

    x = np.concatenate((pre, post))
    k = _cusum(x, b, out, k)
    xm = x - np.median(x)
    k = _cusum(xm * xm, b, out, k)
    k = _scan(x, b, out, k)
    ax = np.abs(x)
    for t, p in AR_SCANS:
        k = _ar_scan(ax if t == "abs" else x, b, p, out, k)
    k = _cross_predict(pre, post, out, k)
    k = _cross_predict(post, pre, out, k)
    k = _whole_context(pre, post, out, k)
    k = _diff_block(pre, post, out, k)
    for p in ROBUST_ORDERS:
        k = _robust_chow(pre, post, p, out, k)
    wpre, wpost = _prewhiten(pre, post, 3)
    k = _transform_block(wpre, wpost, out, k)
    k = _two_sample(wpre, wpost, out, k)
    apre, apost = _standardise(np.abs(wpre), np.abs(wpost))
    k = _transform_block(apre, apost, out, k)
    k = _two_sample(apre, apost, out, k)
    w8pre, w8post = _prewhiten(pre, post, 8)
    k = _transform_block(w8pre, w8post, out, k)
    k = _two_sample(w8pre, w8post, out, k)
    dpre_w, dpost_w = _prewhiten(np.diff(pre), np.diff(post), 3)
    k = _transform_block(dpre_w, dpost_w, out, k)
    k = _two_sample(dpre_w, dpost_w, out, k)
    ppre, ppost = _prewhiten_pre(pre, post, 5)
    k = _transform_block(ppre, ppost, out, k)
    k = _two_sample(ppre, ppost, out, k)
    w20pre, w20post = _prewhiten(pre, post, 20)
    k = _transform_block(w20pre, w20post, out, k)
    k = _two_sample(w20pre, w20post, out, k)
    for p_, kind in ((40, 0), (20, 1), (20, 2), (10, 3), (80, 0), (40, 1), (40, 2), (20, 4)):
        if kind == 0:
            qa, qb = _prewhiten(pre, post, p_)
        elif kind == 1:
            qa, qb = _prewhiten_pre(pre, post, p_)
        elif kind == 2:
            ra, rb = _prewhiten(pre, post, p_)
            qa, qb = _standardise(np.abs(ra), np.abs(rb))
        elif kind == 4:
            qb, qa = _prewhiten_pre(post, pre, p_)

        else:
            qa, qb = _prewhiten(np.diff(pre), np.diff(post), p_)
        k = _transform_block(qa, qb, out, k)
        k = _two_sample(qa, qb, out, k)

    # persistence of the innovation changes (level and volatility) along post
    k = _persistence(w8pre, w8post, out, k)
    va, vb = _standardise(np.abs(w8pre), np.abs(w8post))
    k = _persistence(va, vb, out, k)
    # boundary vs other split points on the whitened innovations (self-normalisation)
    k = _whitened_scan(np.concatenate((w8pre, w8post)), w8pre.shape[0], out, k)
    ab = np.abs(np.concatenate((w8pre, w8post)))
    k = _whitened_scan(ab, w8pre.shape[0], out, k)
    # common linear trend removed first (a trend with no break is not a break)
    tpre, tpost, tstat = _detrend_pooled(pre, post)
    dpre8, dpost8 = _prewhiten(tpre, tpost, 8)
    k = _transform_block(dpre8, dpost8, out, k)
    k = _two_sample(dpre8, dpost8, out, k)
    qa, qb = _standardise(np.abs(dpre8), np.abs(dpost8))
    k = _transform_block(qa, qb, out, k)
    k = _two_sample(qa, qb, out, k)
    k = _whitened_scan(np.concatenate((dpre8, dpost8)), dpre8.shape[0], out, k)
    out[k] = tstat
    k += 1
    k = _trend_chow(pre, post, out, k)
    cpre, cpost = _detrend_poly(pre, post, 3)
    cpre8, cpost8 = _prewhiten(cpre, cpost, 8)
    k = _transform_block(cpre8, cpost8, out, k)
    k = _two_sample(cpre8, cpost8, out, k)
    qa, qb = _standardise(np.abs(cpre8), np.abs(cpost8))
    k = _transform_block(qa, qb, out, k)
    k = _two_sample(qa, qb, out, k)
    hpre = _highpass(pre, HIGHPASS_WINDOW)
    hpost = _highpass(post, HIGHPASS_WINDOW)
    hpre3, hpost3 = _prewhiten(hpre, hpost, 3)
    qa, qb = _standardise(np.abs(hpre3), np.abs(hpost3))
    k = _transform_block(qa, qb, out, k)
    k = _two_sample(qa, qb, out, k)
    k = _garch_block(w8pre, w8post, out, k)
    k = _chow_family(pre, post, out, k)
    k = _tail_block(w8pre, w8post, out, k)
    e = np.concatenate((w8pre, w8post))
    ga, gb, gs2, gll, gll0 = _garch_fit(e)
    hz = np.empty(e.shape[0])
    _garch_ll(e, gs2, ga, gb, hz)
    z = e / np.sqrt(hz)
    k = _tail_block(z[:w8pre.shape[0]].copy(), z[w8pre.shape[0]:].copy(), out, k)
    k = _garch_block(dpre8, dpost8, out, k)
    k = _tail_block(dpre8, dpost8, out, k)
    e = np.concatenate((dpre8, dpost8))
    ga, gb, gs2, gll, gll0 = _garch_fit(e)
    hz = np.empty(e.shape[0])
    _garch_ll(e, gs2, ga, gb, hz)
    z = e / np.sqrt(hz)
    k = _tail_block(z[:dpre8.shape[0]].copy(), z[dpre8.shape[0]:].copy(), out, k)
    k = _full_chow(pre, post, out, k)
    k = _persistence_change(pre, post, out, k)
    k = _moment_block(w8pre, w8post, out, k)
    e = np.concatenate((w8pre, w8post))
    ga, gb, gs2, gll, gll0 = _garch_fit(e)
    hz = np.empty(e.shape[0])
    _garch_ll(e, gs2, ga, gb, hz)
    z = e / np.sqrt(hz)
    k = _moment_block(z[:w8pre.shape[0]].copy(), z[w8pre.shape[0]:].copy(), out, k)
    k = _moment_block(dpre8, dpost8, out, k)
    k = _moment_scan(w8pre, w8post, out, k)
    k = _moment_scan(z[:w8pre.shape[0]].copy(), z[w8pre.shape[0]:].copy(), out, k)
    k = _moment_scan(dpre8, dpost8, out, k)

    out[k] = n0
    out[k + 1] = n1
    out[k + 2] = n
    out[k + 3] = n0 / n
    return out


@njit(parallel=True, cache=True)
def _all_features(values, starts, boundaries, lengths):
    m = starts.shape[0]
    out = np.empty((m, N_FEATURES))
    for i in prange(m):
        seg = values[starts[i]:starts[i] + lengths[i]]
        out[i] = _series_features(seg, boundaries[i])
    return out


VIEWS = ("orig", "flip", "rev", "revflip")


def compute_views(table):
    """Features of four label-preserving views of every series.

    flip negates the values; rev reverses time, so the old post segment becomes the
    new pre segment. A persistent change stays a persistent change under both, and a
    series without one stays without one. Returns {view: DataFrame}, same index.
    """
    ids, starts, lengths, boundaries, values = _layout(table)
    reversed_values = np.empty_like(values)
    for s, n in zip(starts, lengths):
        reversed_values[s:s + n] = values[s:s + n][::-1]
    reversed_boundaries = lengths - boundaries
    arrays = {"orig": (values, boundaries), "flip": (-values, boundaries),
              "rev": (reversed_values, reversed_boundaries),
              "revflip": (-reversed_values, reversed_boundaries)}
    return {view: _frame(_all_features(v, starts, b, lengths), ids)
            for view, (v, b) in arrays.items()}


CROP_ROUNDS = 2
USE_DECIM_VIEWS = False  # v5/v6 used them; dropped after leaderboard drops
CROP_MIN_FRACTION = 0.4
CROP_MIN_LENGTH = 64
CROP_SEED = 2026


def compute_crop_views(table, rounds=CROP_ROUNDS, lo=CROP_MIN_FRACTION, seed=CROP_SEED):
    """Training-only views: random crops of every augmented view.

    Each crop keeps a random tail of the pre segment and a random head of the post
    segment (each between `lo` and 100% of its length, at least 64 points). The label
    is unchanged: a persistent change is still visible right after the boundary, and a
    series without one stays without one. Seeds are fixed, so the views are reproducible.
    """
    ids, starts, lengths, boundaries, values = _layout(table)
    reversed_values = np.empty_like(values)
    for s, n in zip(starts, lengths):
        reversed_values[s:s + n] = values[s:s + n][::-1]
    bases = {"orig": (values, boundaries), "flip": (-values, boundaries),
             "rev": (reversed_values, lengths - boundaries),
             "revflip": (-reversed_values, lengths - boundaries)}
    views = {}
    for r in range(rounds):
        for k, (view, (v, b)) in enumerate(bases.items()):
            rng = np.random.default_rng([seed, r, k])
            n0 = b
            n1 = lengths - b
            keep_pre = np.minimum(n0, np.maximum(CROP_MIN_LENGTH, (n0 * rng.uniform(lo, 1.0, len(n0))).astype(np.int64)))
            keep_post = np.minimum(n1, np.maximum(CROP_MIN_LENGTH, (n1 * rng.uniform(lo, 1.0, len(n1))).astype(np.int64)))
            pieces = [v[s + a0 - kp: s + a0 + kq] for s, a0, kp, kq in zip(starts, n0, keep_pre, keep_post)]
            new_lengths = (keep_pre + keep_post).astype(np.int64)
            new_starts = np.concatenate(([0], np.cumsum(new_lengths)[:-1])).astype(np.int64)
            out = _all_features(np.concatenate(pieces), new_starts, keep_pre.astype(np.int64), new_lengths)
            views[f"crop{r}_{view}"] = _frame(out, ids)
    return views


def compute_decim_views(table):
    """Training-only views: every second point (a time-scale change), forward and
    time-reversed. Kept only where both segments still have at least 32 points."""
    ids, starts, lengths, boundaries, values = _layout(table)
    views = {}
    for name, rev in (("decim_orig", False), ("decim_rev", True)):
        pieces, bounds = [], []
        for s, n, b in zip(starts, lengths, boundaries):
            x = values[s:s + n]
            bb = b
            if rev:
                x = x[::-1]
                bb = n - b
            pieces.append(x[::2])
            bounds.append(min(max((bb + 1) // 2, 32), (n + 1) // 2 - 32))
        new_lengths = np.array([len(z) for z in pieces], np.int64)
        new_starts = np.concatenate(([0], np.cumsum(new_lengths)[:-1])).astype(np.int64)
        out = _all_features(np.concatenate(pieces), new_starts, np.array(bounds, np.int64), new_lengths)
        views[name] = _frame(out, ids)
    return views


def compute_features(table):
    """One feature row per series id (original view only), in order of first appearance."""
    ids, starts, lengths, boundaries, values = _layout(table)
    return _frame(_all_features(values, starts, boundaries, lengths), ids)


def _frame(out, series_ids):
    frame = pd.DataFrame(out, index=pd.Index(series_ids, name="id"), columns=FEATURE_NAMES)
    if not np.isfinite(frame.to_numpy()).all():
        bad = frame.columns[~np.isfinite(frame.to_numpy()).all(axis=0)].tolist()
        raise ValueError(f"Nonfinite features: {bad[:10]}")
    return frame


def _layout(table):
    """Series offsets in a validated table (contiguous time 0..n-1, one 0->1 boundary)."""
    ids = table["id"].to_numpy()
    change = np.flatnonzero(ids[1:] != ids[:-1]) + 1
    starts = np.concatenate(([0], change)).astype(np.int64)
    lengths = np.diff(np.concatenate((starts, [len(ids)]))).astype(np.int64)
    series_ids = ids[starts]
    if len(set(series_ids)) != len(series_ids):
        raise ValueError("Rows of each series must be contiguous")
    period = table["period"].to_numpy()
    values = table["value"].to_numpy(dtype=np.float64)
    boundaries = np.empty(len(starts), dtype=np.int64)
    for i, (s, n) in enumerate(zip(starts, lengths)):
        boundaries[i] = int(np.argmax(period[s:s + n] == 1))
    return series_ids, starts, lengths, boundaries, values
