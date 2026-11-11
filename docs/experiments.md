# Experiment log

Every idea was scored on the same whole-series folds. Early on, a change was kept only if it
lifted out-of-fold AUC by at least 0.003. Later, once a fast GPU screen and multi-seed
confirmation made smaller gains measurable (seed noise about ±0.0005), the bar dropped to a
consistent gain across seeds.

## Version history

| Version | Change | CV AUC | Public LB |
|---|---|---|---|
| baseline | Starter before/after features + logistic regression | 0.708 | – |
| v1 | 281 before/after statistics and tests, LightGBM + CatBoost | 0.8186 | – |
| v1-tuned | + Optuna-tuned parameters and sign/time symmetry views | 0.8243 | – |
| v2 | + AR cross-prediction and whole-series context ratios | 0.8279 | 0.83118 |
| v3 | + random crop views | 0.8335 | 0.83657 |
| v5 | + differenced-series battery, decimated views | 0.8353 | 0.8337 |
| v6 | + Student-t Chow tests | 0.8366 | 0.83484 |
| v7 | + prewhitened batteries, decimated views removed | 0.8453 | 0.8440 |
| v8 | + detrending, GARCH filter, tail LRs, whitened scans, persistence | 0.8527 | 0.84969 |
| – | + four learned no-break networks | 0.8588 | – |
| v9 | + Gaussianized batteries, full pipeline on normal scores | 0.8613 | 0.85787 |
| v9b | + learned break model (H1/H0) | 0.8626 | 0.86096 |
| B | CPU-sized cut of v9b (fold-0 networks, 3 views) | 0.8619 | 0.85971 |
| L1 | lean feature code, fast network scorer, LightGBM + XGBoost | 0.8628 | 0.85979 |
| L2 | + hidden-break network, lean pipeline on normal scores | 0.8632 | – |
| L3 | + seasonal change tests, battery on the deseasonalized series; XGBoost only | 0.8661 | 0.86416 |
| **L3n** | **+ networks on the deseasonalized series (seasonal series only)** | **0.8669** | **0.86539** |

v9b needed about 5 hours of inference on 2 CPU cores. B, L1 and later are the versions that fit a
30-minute, 2-core budget.

## Kept

| Idea | Effect |
|---|---|
| Sign-flip and time-reversal views, averaged at test time | 0.8171 → 0.8219 (LightGBM) |
| AR(5) cross-prediction + whole-series context ratios (from ADIA Lab 5th-place write-up) | +0.0045 |
| Hyperparameter search (500 LightGBM and 203 CatBoost trials, `experiments/tune.py`) | +0.003 to +0.005 |
| Random crop views (8 per series) | +0.0056 CV, +0.005 LB |
| Differenced-series battery (aimed at random-walk-like series) | +0.001 overall, +0.022 on that group |
| Student-t Chow tests | +0.0013 |
| Prewhitened batteries | 0.8366 → 0.8453 |
| Detrending, GARCH(1,1) filtering, tail-shape LRs, whitened scans | → 0.8527 |
| Learned no-break networks (four sizes, later two) | +0.0061 |
| Gaussianization | +0.0025 |
| Learned break model H1/H0 | +0.0015 CV, +0.0031 LB |
| Hidden-break network | +0.0003 |
| Seasonal tests + deseasonalized battery | +0.0038 (one seed) |
| Networks on the deseasonalized seasonal series | +0.0006 (7 seeds, every seed higher) |

## Not kept

### Features

| Idea | Result |
|---|---|
| Spectral band features | +0.0005 |
| Random-convolution (MiniRocket-style) features | no gain |
| Entropy and complexity (permutation, sample, SVD, Higuchi, DFA, Hjorth, Lempel-Ziv) | +0.0007 |
| Extra tests (Anderson–Darling, Bartlett, Fligner, Jensen–Shannon, Bhattacharyya, Jarque–Bera) | 0.000 |
| Multi-scale windows on rank / abs / rolling transforms (ADIA 2nd place) | −0.002 |
| Pairwise interactions of the top 25 features, selected inside each training fold (ADIA 5th place) | 0.000 |
| Raw-scale features incl. index of dispersion | +0.0003 (no scale signal in this data) |
| AR(1)-whitened boundary innovations, level-shift test, jump-retained ratios | 0.000 overall, −0.006 on random-walk series |
| ADF unit-root profile along post, de-spiked (Hampel) battery | ≤ +0.0004 |
| 16 statistic families aimed at hidden breaks | hidden-group AUC 0.7406 → 0.7415 |
| Shape-only two-sample tests (`solution/shape.py`) | +0.0003 |
| Battery on first differences, on detrended series, on stable stretches next to the boundary | flat or negative |
| Per-point evidence curves with a small CNN on top (`experiments/curvenet.py`) | no gain |

### Views and data

| Idea | Result |
|---|---|
| Decimated (subsampled) training views | +0.0008 CV, but preceded two leaderboard drops; removed |
| 4 crop rounds instead of 2, crop or decimated test-time views | no gain |
| Synthetic no-break series cut from inside segments (`experiments/make_synth.py`) | worse |
| Network-simulated replicas (`experiments/simgen.py`) | real vs synthetic separable at 0.74 to 0.87 AUC; worse |
| Breaks injected into real no-break series (`experiments/make_inject2.py`) | no gain |
| Negative re-splits, monotone warps, block bootstrap | no gain or worse |

### Models

| Idea | Result |
|---|---|
| Stacking: XGBoost, random forest, extra trees, logistic under a meta-model (ADIA 1st place) | +0.001 |
| CatBoost in the blend | +0.0001 for 60% more training time |
| 1D CNN on the raw series, 5-fold (`experiments/deepnet.py`) | 0.799 alone, +0.0007 in a blend |
| Hybrid network (raw series + tree features) | 0.821 alone, no gain in a blend |
| Bigger networks, more mixture components, histogram heads, 10-fold cross-fitting | within noise |
| Attention-encoder no-break network, crop-trained break models, bidirectional break model | within noise |
| Averaging 5 folds of networks at test time | within noise (fold feature sets correlate at 0.998) |
| Upweighting hidden breaks ×2 / ×3 | hidden 0.748 / 0.754, overall lower |
| Specialist model on hidden breaks | 0.768 on that group, no gain in a blend |
| Second hyperparameter search on a later feature set | +0.0002 |
| Blending LightGBM into the final XGBoost | +0.00006 |
| Density-ratio reweighting toward the test distribution | LB 0.86035 vs 0.86096 |

## Lessons

1. The most productive question was always "what would a no-break series look like here?"
   Prewhitening, Gaussianization and the network likelihood ratios are successively better
   answers. Most of the gain came from making classical tests valid for dependent, heavy-tailed
   data, not from inventing new tests.
2. Neural networks were far more useful as learned null models feeding a tree model than as
   classifiers.
3. The break-type decomposition was the most informative error analysis. It showed where the
   remaining error sits, and so where to stop spending time.
4. Error analysis by series family found the seasonal blind spot late. Grouping errors by
   generator family earlier would have found it sooner.
5. Validation discipline paid for itself: with every learned component cross-fitted, gains of
   0.001 to 0.002 could be trusted, and the leaderboard tracked them.
