"""v9 final fits: LightGBM seeds on all 24,000 training series (12 views) with the handcrafted
features (minus the dropped families) plus any number of added feature folders (network,
Gaussianized-battery and normal-score-pipeline features), using the round counts from the
saved 5-fold CV of exactly the same configuration (1.1 x mean best iteration).

    python solution/fit_final_v9.py --work solution/work_hx --parts 80de1ca0 --out solution/checkpoint_v9 \
        --add solution/work_nn2m8l32XL --add solution/work_nn2m12l48L ... --threads 24
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
import models  # noqa: E402
import train  # noqa: E402

EXCLUDE = r"(cw8|cw8a|hpa|tchow\d|ttchow|dchow|fchow\d.)_"
ap = argparse.ArgumentParser()
ap.add_argument("--work", type=Path, required=True)
ap.add_argument("--add", type=Path, action="append", default=[])
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--parts", required=True, help="cv_parts key of the CV run of this configuration")
ap.add_argument("--threads", type=int, default=24)
ap.add_argument("--models", default="lgb_s0,lgb_s1,lgb_s2")
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
views = train.train_features(".", a.work, cached_only=True)
keep = [c for c in views["orig"].columns if not re.match(EXCLUDE, c)]
views = {v: f[keep] for v, f in views.items()}
for extra in a.add:
    views = {v: pd.concat([f, pd.read_parquet(extra / f"nn_features_{v}.parquet").reindex(f.index)], axis=1)
             for v, f in views.items()}
if __import__("os").environ.get("EXC2"):
    _x2 = __import__("os").environ["EXC2"]
    _k2 = [c for c in views["orig"].columns if not re.search(_x2, c)]
    views = {v: f[_k2] for v, f in views.items()}
    print("after EXC2", len(_k2), flush=True)
used = list(views["orig"].columns)
y = data.load_labels(Path("train_labels.csv"), views["orig"].index.tolist()).reindex(views["orig"].index)
params = models.load_params(Path("solution/params.json"))
names = a.models.split(",")
for name in names:
    if (a.out / f"{name}.txt").exists():
        print(name, "already fitted", flush=True)
        continue
    part = json.loads((a.work / "cv_parts" / f"{name}_{len(used)}f_12v_{a.parts}.json").read_text())
    models.fit_final(name, views, y, part["iterations"], a.threads, a.out, params=params.get("lgb"))
    print(name, "fitted", part["iterations"], flush=True)
(a.out / "meta.json").write_text(json.dumps({"models": names, "feature_names": used, "exclude": EXCLUDE,
                                             "add": [str(d) for d in a.add]}, indent=1))
print("done", a.out)
