# Archived experiments

Exploratory scripts kept as a record of what was tried. They import the pipeline modules from
`../solution/` but were written against earlier versions of the feature caches, so treat them as
reference code rather than maintained tools. Results are in [`../docs/experiments.md`](../docs/experiments.md).

| Script | What it tested | Outcome |
|---|---|---|
| `tune.py` | Optuna TPE search for LightGBM and CatBoost on the fixed folds, with a nested mode for an unbiased estimate of the tuned pipeline | Kept: produced `solution/params.json` |
| `fastscreen.py` | Fast GPU XGBoost screen with the same folds, views and early stopping, used to rank ideas before multi-seed confirmation | Kept as the screening tool |
| `nested_break.py` | Fully nested cross-fitting of the H1/H0 break model, to measure the second-order leak | Fold-0 AUC −0.0012 |
| `deepnet.py` | Siamese dilated CNN with attention pooling on the raw series | 0.799 alone, +0.0007 in a blend |
| `curvenet.py` | Small CNN over the networks' per-point log-likelihood curves | No gain |
| `make_synth.py` | Synthetic no-break series cut from inside single segments | Lowered CV |
| `simgen.py` | Series-level bootstrap with the no-break network as simulator | Synthetic series separable from real; lowered CV |
| `make_inject2.py` | Injecting small shape, AR and volatility breaks into real no-break series | No gain |
