"""Loading and validation of the KU-SBB inputs.

`series_ids`, `check_labels` and `checked_submission` are taken from the
organisers' starter.py (KU-SBB participant bundle, Khalifa University AI Club)
with only cosmetic changes; the original file sits next to this folder.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

INPUT_FILES = ("train_series.parquet", "train_labels.csv", "test_series.parquet",
               "sample_submission.csv", "starter.py", "README.md")


def verify_hashes(data):
    """Check the competition inputs against manifest.json (from starter.py)."""
    manifest = json.loads((Path(data) / "manifest.json").read_text(encoding="utf-8"))
    hashes = manifest.get("files_sha256", {})
    for name in INPUT_FILES:
        digest = hashlib.sha256((Path(data) / name).read_bytes()).hexdigest()
        if digest != hashes.get(name):
            raise ValueError("Changed input: " + name)


def series_ids(table):
    """Validate the observation table and return its ids (from starter.py)."""
    if table.empty or set(table.columns) != {"id", "time", "value", "period"} or len(table.columns) != 4:
        raise ValueError("Wrong observation columns")
    if not table.id.map(lambda x: isinstance(x, str) and bool(x)).all():
        raise ValueError("Nonempty string ids required")
    if not pd.api.types.is_numeric_dtype(table.value) or pd.api.types.is_bool_dtype(table.value) or pd.api.types.is_complex_dtype(table.value):
        raise ValueError("Real numeric observations required")
    if not np.isfinite(table.value.to_numpy(dtype=float)).all():
        raise ValueError("Finite observations required")
    for column in ("time", "period"):
        if not pd.api.types.is_integer_dtype(table[column]) or pd.api.types.is_bool_dtype(table[column]) or table[column].isna().any():
            raise ValueError("Integer time and period required")
    if table.duplicated(["id", "time"]).any():
        raise ValueError("Duplicate observation key")
    for _, group in table.groupby("id", sort=False):
        periods = group.period.to_numpy()
        if not np.array_equal(group.time, np.arange(len(group))):
            raise ValueError("Time must start at zero and be consecutive")
        if set(periods) != {0, 1} or periods[0] != 0 or periods[-1] != 1 or np.count_nonzero(np.diff(periods)) != 1:
            raise ValueError("Exactly one known 0-to-1 boundary required")
        if min(np.count_nonzero(periods == 0), np.count_nonzero(periods == 1)) < 32:
            raise ValueError("At least 32 observations per period required")
    return table.id.drop_duplicates().tolist()


def check_labels(table, ids):
    """One integer 0/1 label per training id (from starter.py)."""
    if set(table.columns) != {"id", "label"} or len(table.columns) != 2 or table.id.duplicated().any():
        raise ValueError("One label per id required")
    if set(table.id) != set(ids) or not pd.api.types.is_integer_dtype(table.label) or pd.api.types.is_bool_dtype(table.label) or not table.label.isin([0, 1]).all():
        raise ValueError("Labels must match ids and be integer 0/1")


def checked_submission(predicted, sample_table):
    """Align scores to the sample ids and check them (from starter.py)."""
    if list(sample_table.columns) != ["id", "break_probability"] or sample_table.id.duplicated().any():
        raise ValueError("Invalid sample submission")
    if not sample_table.id.map(lambda x: isinstance(x, str) and bool(x)).all():
        raise ValueError("Nonempty expected string ids required")
    if predicted.index.has_duplicates or set(predicted.index) != set(sample_table.id):
        raise ValueError("Prediction IDs differ from sample")
    values = predicted.reindex(sample_table.id).to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Finite scores in [0,1] required")
    return pd.DataFrame({"id": sample_table.id.to_numpy(), "break_probability": values})


def load_series(path):
    table = pd.read_parquet(path)
    series_ids(table)
    return table


def load_labels(path, ids):
    labels = pd.read_csv(path, dtype={"id": str})
    check_labels(labels, ids)
    return labels.set_index("id").label
