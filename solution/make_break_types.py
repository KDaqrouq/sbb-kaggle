"""Break-type label per training series, used only to pick the series the hidden-break network
(hidden_h1.py) is fine-tuned on and for the error analysis in the report.

Each series is fitted (simparam2.fit_series) with a small parametric model (level, linear trend,
AR coefficients, scale, GARCH innovations); BIC decides which of level / trend / AR / scale differ
across the boundary. "none" = no parametric change selected ("hidden" break when the label is 1).

    python solution/make_break_types.py        # writes solution/work_hx/btype_all.csv (id, btype)
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simparam2 as S2  # noqa: E402

t = pd.read_parquet("train_series.parquet")
rows = []
for sid, g in t.groupby("id", sort=False):
    v = g.value.to_numpy()
    b = int((g.period == 0).sum())
    mp, sp = S2.med_scale(v[:b])
    f = S2.fit_series((v - mp) / sp, b, 1)
    rows.append((sid, "+".join(f["changes"]) or "none"))
out = Path("solution/work_hx")
out.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows, columns=["id", "btype"]).to_csv(out / "btype_all.csv", index=False)
print("done", out / "btype_all.csv")
