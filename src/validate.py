"""Submission format check. Run before every Kaggle submit."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"


def find_file(name: str) -> Path:
    hits = sorted(DATA.rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {DATA}")
    return hits[0]


def validate_submission(path, ref_path=None) -> pd.DataFrame:
    """Check that a submission matches test.csv in columns, row count and id order,
    has no missing values, and all scores lie in [0, 5]. Raises AssertionError on failure.
    test.csv is the reference because sample_submission.csv lists different file names
    (only 25 of its 204 ids exist in test/; see docs/decisions.md)."""
    sample = pd.read_csv(ref_path or find_file("test.csv"))
    sub = pd.read_csv(path)
    assert list(sub.columns) == list(sample.columns), f"columns {list(sub.columns)} != {list(sample.columns)}"
    assert len(sub) == len(sample), f"{len(sub)} rows, expected {len(sample)}"
    id_col, score_col = sample.columns[0], sample.columns[-1]
    assert (sub[id_col].astype(str).values == sample[id_col].astype(str).values).all(), "id order differs from sample"
    assert sub.isna().sum().sum() == 0, "missing values"
    s = pd.to_numeric(sub[score_col], errors="raise").to_numpy()
    assert np.isfinite(s).all(), "non-finite scores"
    assert s.min() >= 0 and s.max() <= 5, f"scores outside [0,5]: {s.min()}..{s.max()}"
    print(f"OK: {path} ({len(sub)} rows, {score_col} range {s.min():.3f}..{s.max():.3f})")
    return sub


if __name__ == "__main__":
    validate_submission(sys.argv[1])
