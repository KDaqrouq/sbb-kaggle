# Methodology

This document explains every block of the final model (configuration **L3n**), why it is there,
how it was validated, and where the model still fails. Version names (v1 to v9b, L1 to L3n) refer
to the progression chart in the main README.

## 1. The problem

The training set has 24,000 univariate series (12,000 with a break, 12,000 without), and the test
set has 4,000. Series are 256 to about 1,024 points long. A `period` column marks which points
are before (0) and after (1) the boundary, so the break location never has to be searched for. The
only question is whether the generating process changed **persistently** at that point. Noise,
outliers and temporary disturbances can make an unchanged series look different on the two sides,
and those are labelled 0.

Things checked at the start, because they shape everything after:

- Series length and boundary position carry no label information on their own, so there is no
  leakage to exploit and nothing to correct for.
- Breaks go both ways (the mean can go up or down, the variance can grow or shrink), so features
  measure the size of a change, not its sign.
- Breaks affect level, spread, dependence (autocorrelation) and distribution shape. A model has to
  look at all four.
- No two series share opening values or increments, so there are no shared generator lineages and
  each series can be treated as an independent sample in validation.
- The series come from several families: plain noise (about half), seasonal (about 15%), trending,
  random-walk-like, heavy-tailed, volatility-clustering and two-peaked.

With the boundary known, this is a **two-sample problem on dependent data**: are the pre and post
segments draws from the same process? Most design choices follow from one question: *what would
this series look like after the boundary if nothing had changed?*

## 2. Validation

| Learned component | How it is isolated from the series it scores |
|---|---|
| Handcrafted and seasonal features | Each fit (AR, GARCH, trend, harmonic regression) uses only the series being described. Nothing is pooled across series, so there is no fitted state to leak. |
| No-break networks | Trained only on boundary-free stretches of training-fold series. Features for fold k come from networks trained without fold k (cross-fitting). |
| Gaussianized features | Computed with the same cross-fitted fold network. |
| Break models (H1, H0, hidden) | Fine-tuned on labelled training-fold series only, and cross-fitted the same way. |
| Trees | Trained on training folds. Early stopping on a stratified 10% split of the training folds, never on the validation fold. |
| Final models | Trained on all 24,000 series, with the round count fixed in advance at 1.1 × the mean early-stopped round from CV. |

**Folds.** 5 folds by whole series, stratified by label, seed 2026 (`solution/folds.csv`). All
augmented views of a series stay in its fold.

**Second-order cross-fitting effect.** When the trees for validation fold k train, the network
features of their training rows (fold j ≠ k) come from networks whose training data included fold
k. Information about fold k can reach the trees indirectly through the network weights. This was
measured two ways:

- a model with no learned features at all (handcrafted and seasonal blocks only) reaches
  **0.8600**, so most of the score has no cross-fitting dependence;
- fully nested fitting of the break model (fine-tuned without both folds) lowered fold-0 AUC by
  **0.0012** (`experiments/nested_break.py`).

**Selection discipline.** Well over a hundred variants were evaluated on the same folds. That
risk was controlled four ways:

- *Two-stage screening:* a fast GPU XGBoost screen on the same folds and views (noise about
  ±0.0005 per seed, `experiments/fastscreen.py`), then confirmation with more seeds.
- *Coarse decisions:* whole feature families were kept or dropped. There was no importance-based
  feature selection.
- *Frozen hyperparameters:* tree settings were tuned once and not re-tuned on every feature set.
- *Fresh-split check:* key comparisons were rerun on a new fold assignment (seed 2027). Scores
  moved by about 0.001 and the late gains reproduced.

| Comparison (XGBoost screen) | Original folds | Fresh folds |
|---|---|---|
| Handcrafted features only | 0.8490 | 0.8478 |
| v9 features (two seeds) | | 0.8596 / 0.8597 |
| v9 + learned break model (two seeds) | | 0.8617 / 0.8616 |
| Gain from the break model | +0.0013 to +0.0017 | +0.0020 |

## 3. Features

### 3.1 Augmentation

Two transformations keep the label: negating the series, and reversing time (a break is still a
break, with pre and post swapped). Every training series appears in 4 such views, plus 8 random
crops that keep 40 to 100% of each side (at least 64 points). That gives 12 training rows per
series. Prediction averages the original, flipped and reversed-flipped views.

- Symmetry views (with tuned parameters): 0.8186 → 0.8279.
- Crops: +0.0056 CV and +0.005 on the leaderboard.
- Decimated views improved CV slightly but preceded two leaderboard drops, so they were removed.

### 3.2 Handcrafted change statistics

`features.py` (about 70 numba kernels) and `features_lean.py` (the same values, skipping blocks
the final model does not use). Each series is first standardised by the median and MAD of its own
pre segment. The same battery runs on several copies of the series: raw, first difference,
absolute deviations, absolute differences and cumulative sum.

- **Segment descriptors and contrasts:** mean, median, trimmed mean, quantiles, std, MAD, IQR,
  skewness, kurtosis, autocorrelations and variance ratios, as post-minus-pre differences and log
  ratios.
- **Two-sample tests:** Welch t, Brown–Forsythe, Kolmogorov–Smirnov, Mann–Whitney, energy
  distance, Gaussian-kernel MMD with median bandwidth.
- **Regression tests:** AR Chow tests of several orders, a Student-t Chow test fitted by
  iteratively reweighted least squares (robust to outliers), trend Chow tests.
- **AR cross-prediction:** fit AR(5) on one side and predict the other side one step ahead, in
  both directions, compared with each side's own fit. An unchanged process is predicted equally
  well by either model. (The idea came from public ADIA Lab Structural Break Challenge write-ups.)
- **Scans:** CUSUM, plus split-point scans that compare a statistic at the true boundary with the
  same statistic at 33 other split points. A series that naturally wanders scores high
  everywhere, not only at the boundary, so each test becomes self-calibrating.
- **Persistence:** pre against each third and the late half of post, early post against late
  post, and the two halves of pre as a no-break reference. ADF and KPSS statistics on each side.
  These separate a lasting change from a temporary disturbance, which is what the label asks.

### 3.3 Prewhitening

Two-sample tests assume independent observations. With autocorrelated data they find differences
that are not there. Fitting an AR(p) model to the **pooled** series (the no-break hypothesis) and
running the battery on its innovations restores their calibration: under no break the innovations
are close to independent, and under a break a change in level, variance or dependence shows up in
them. Orders p ∈ {3, 8, 20, 40, 80} (capped at a third of the shorter segment) were used, plus
pre-side-only fits and innovations of the differenced series.

**This step alone took CV from 0.8366 to 0.8453, the largest single gain of the project.**

The extensions remove the next sources of miscalibration. Linear detrending before whitening
handles trending series. GARCH(1,1) filtering removes volatility clustering, which otherwise makes
variances differ across the boundary by chance. Student-t tail-shape likelihood ratios capture
changes in tail weight. Together these reached 0.8527 (v8).

### 3.4 Learned no-break model

An AR or GARCH model is a parametric guess at the no-break process. `nullnet2.py` replaces the
guess with a learned model:

- a **dilated 1-D convolutional encoder** (residual blocks with growing dilation, reading the
  standardised values, their absolute values and their squares) summarises a context window of up
  to 1,024 points into a regime embedding;
- a **decoder MLP** takes the embedding and the previous L values and outputs a K-component
  **Student-t mixture** for the next value;
- it is trained by maximum likelihood (AdamW, one-cycle schedule) on random context/target splits
  of boundary-free stretches, with sign flips and time reversal.

Features are log-likelihood ratios of each segment under the regime inferred from the pre segment,
the post segment and the pooled series. This is a **learned Chow test**, where the network supplies
the dependence, volatility and tail behaviour. Two sizes are used in the final model (K/L = 8/32
and 16/64). Larger networks, histogram output heads and 10-fold cross-fitting gave nothing more.
`nullfast.py` scores each context once and reuses it across features, which made the networks
affordable on 2 CPU cores.

### 3.5 Gaussianization

The network gives a full predictive CDF F, so every point can be mapped to

  z_t = Φ⁻¹( F(x_t | x_{t−1}, …, x_{t−L}, pooled regime) ).

Under no break the z_t are approximately i.i.d. N(0, 1). This generalises prewhitening: it removes
dependence, heteroskedasticity and non-Gaussian shape in one step, after which classical tests are
close to their textbook conditions. A 184-feature battery runs on z and |z|, and a lean version of
the whole handcrafted pipeline runs on the 16/64 network's normal scores (`nullgauss.py`,
`nullgauss2.py`). Networks plus Gaussianization took CV from 0.8527 to 0.8613 (v9).

### 3.6 Learned break models

By the Neyman–Pearson lemma, the most powerful test between two simple hypotheses is their
likelihood ratio. The no-break networks give one side. The other side, the distribution of the
post segment given the pre segment *when a break occurred*, can be learned from labels.
`nullbreak.py` fine-tunes two copies of each fold's 16/64 network on (pre context, post segment)
pairs: **H1** on breaks only and **H0** on non-breaks only. Their log-likelihood difference over the
whole post segment and its early and late parts is a likelihood ratio against the breaks that
actually occur. It added +0.0015 CV and +0.0031 on the leaderboard (v9b).

`hidden_h1.py` fine-tunes a third copy only on *hidden* breaks (breaks with no level, trend, AR
or scale change, from the decomposition in section 5), scored against H0 (+0.0003).

### 3.7 Seasonality

About 15% of series carry a clean cycle (periods 12 to 31) that explains most of their variance.
A change of its phase, amplitude or frequency leaves the marginal distribution and the spectrum
unchanged, and the cycle swamps every other test. `seasonal.py` adds:

- **seasonal change tests:** harmonic regression at the series' dominant frequency on a
  continuous time axis, a Chow-type F test on the harmonic coefficients, phase and amplitude
  change, a frequency change, and the error of extrapolating one side's cycle to the other;
- the **whole handcrafted battery on the deseasonalized series** (pooled cycle removed), the same
  idea as prewhitening and detrending.

One-seed CV went from 0.8614 to 0.8652, seasonal series from 0.882 to 0.897, and hidden breaks
overall from 0.742 to 0.750 (L3). Finally, for the clearly seasonal series only (periodogram peak
over 20 times its neighbours, period under 80; about 19%), `make_ds_nets.py` reruns the networks,
the Gaussianized battery and the break models on the deseasonalized series: +0.0006 over 7 seeds,
with every seed higher (L3n).

### 3.8 Model

The feature set (about 5,200 columns) is large, heterogeneous in scale and redundant by design.
Gradient-boosted trees handle that without scaling or selection, and they find the interactions
between a statistic and its nuisance context (a Chow statistic means less in a near-unit-root
series).

The final model is **XGBoost** (histogram trees, depth 8, learning rate 0.03, column sampling 0.3)
over 7 seeds, with probabilities averaged over seeds and views. Earlier versions used LightGBM
with parameters from a 500-trial Optuna search (`solution/params.json`,
`experiments/tune.py`). By L3, blending LightGBM in no longer helped.

## 4. Inference under a compute budget

Inference had to run on 2 CPU cores with 8 GB of RAM and no network, in under 30 minutes, and be
deterministic. That constraint shaped the final model:

- v9b used 5 folds × 4 views of network features and needed about 5 hours on 2 cores, so it was
  cut back to fold-0 networks and three views;
- `features_lean.py` skips every handcrafted block the model does not use, and `nullfast.py`
  encodes each context once;
- the networks on deseasonalized series run only on the ~19% of series that are clearly seasonal.

Result: 26.4 minutes cold (about 5 of them numba compilation), 5.6 GB, and identical
probabilities across two cold runs.

## 5. Error and robustness analysis

### By break type

Each break series was described with a small parametric model (level, linear trend, AR
coefficients, scale), and BIC decided which parameters change at the boundary. This uses labels
and is for analysis only. Each group is scored against all 12,000 negatives (out-of-fold v9b
predictions):

| Detected change | n | AUC |
|---|---|---|
| Level | 1,864 | 0.986 |
| Scale | 3,067 | 0.972 |
| Level + scale | 437 | 0.997 |
| Trend | 389 | 0.945 |
| Trend + scale | 122 | 0.982 |
| Level + trend | 205 | 0.863 |
| AR coefficients (alone or combined) | 154 | 0.978 |
| None of the above ("hidden") | 5,762 | 0.744 |

The final model raises the hidden group to about 0.75. Attempts aimed at it directly:

- 16 new statistic families (spectral shape in 8 bands, autocorrelations to lag 20, ACF of squares
  and absolute values, leverage, KS on standardised segments, turning points, boundary jumps)
  moved hidden-group AUC from 0.7406 to 0.7415;
- upweighting hidden breaks ×2 or ×3 raised them to 0.748 / 0.754 but lowered overall AUC, which
  just traded easy-break errors for hard ones;
- a specialist model on hidden breaks reached 0.768 on that group and added nothing in a blend.

Most hidden breaks look like real but small changes, below what can be resolved at these lengths.

### By series characteristics (v9b)

| Group | n | AUC | AUC, hidden breaks only |
|---|---|---|---|
| Length < 400 | 4,472 | 0.809 | 0.682 |
| Length 400 to 600 | 6,301 | 0.843 | 0.717 |
| Length 600 to 850 | 7,787 | 0.876 | 0.758 |
| Length ≥ 850 | 5,440 | 0.899 | 0.793 |
| Pre lag-1 autocorrelation > 0.9 | 1,968 | 0.708 | 0.608 |
| Pre lag-1 autocorrelation 0.5 to 0.9 | 3,008 | 0.886 | 0.814 |
| Pre lag-1 autocorrelation ≤ 0.5 | 19,024 | 0.870 | 0.741 |
| Pre excess kurtosis ≥ 3 | 1,273 | 0.827 | 0.716 |

AUC rises steadily with length, as test power should. In near-unit-root series a level shift
looks like ordinary wandering. Heavy tails make single outliers mimic changes, which is what the
Student-t Chow tests and Gaussianization target.

### Stability and calibration (v9b)

- Fold AUCs 0.861, 0.856, 0.861, 0.867 and 0.870.
- On random subsets of 4,000 series (the test-set size), the standard deviation of AUC is 0.0052.
  Leaderboard differences under about 0.005 are noise, so final selection rested on CV.
- Probabilities are well calibrated: in every decile the mean prediction is within 0.03 of the
  observed break rate (Brier score 0.144).

### Cross-validation versus leaderboard

The public score sits slightly below CV for every version. This was checked for leakage or a
pipeline bug:

- **Pipeline consistency:** the upper quantiles of handcrafted statistics (no trained model)
  shift from train to test by the same factor as the network features. A test-time network bug
  would show in the network features alone.
- **Test difficulty:** the test set has fewer very confident breaks. Because CV probabilities are
  calibrated, the expected AUC of a set can be estimated from its predictions alone. That gives
  0.852 for the test set (and 0.8636 for train, against a true 0.8628). The public score was above
  that.
- **Train/test shift:** an adversarial classifier separates train from test at AUC 0.59, a mild
  shift. Density-ratio reweighting did not improve the leaderboard.
- **Transfer of gains:** improvements over 0.003 in CV carried over to the leaderboard.

## 6. Limitations

- **Hidden breaks** (48% of breaks) score about 0.75. This looks close to a detection limit, but a
  missed mechanism can't be ruled out.
- **Short and near-unit-root series** are clearly harder (0.81 and 0.71).
- **Cross-fitting is not fully nested.** The measured effect is small (−0.0012 for the break
  model), and full nesting of the no-break networks would take about 40 GPU hours.
- **Break-type labels are approximate.** The decomposition is a BIC choice on noisy data.
- **Inference time depends on the CPU.** The 30-minute budget is met on fast cores. An older
  2.0 GHz server CPU took 34 minutes on a lighter configuration.
- **One synthetic generator.** The learned null and break models describe this generator. On real
  data the networks would need retraining, and the classical features are the more transferable
  part.
