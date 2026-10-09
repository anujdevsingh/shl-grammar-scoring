"""Fixed CV folds and metrics, shared by every experiment.
Folds are rebuilt deterministically from the seed; the per-file fold table is written to cache/ (gitignored)."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import RepeatedStratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "Dataset_Final"
FOLDS = ROOT / "cache" / "folds.csv"
N_SPLITS, N_REPEATS, SEED = 5, 3, 42


def load_train():
    return pd.read_csv(DATA / "train.csv")


def make_folds():
    """One column per repeat (r0, r1, r2) holding the fold id 0-4 of each train file.
    Stratified on the label (0.5 steps; rare 1.0/1.5 are merged with 2.0 so every bin has >= 5 clips)."""
    tr = load_train()
    strat = (tr.label.clip(lower=1.0).replace({1.0: 2.0, 1.5: 2.0}) * 2).astype(int)
    strat[tr.label == 0] = 0
    out = tr[["filename"]].copy()
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    for i, (_, va) in enumerate(rskf.split(out, strat)):
        r, f = divmod(i, N_SPLITS)
        out.loc[out.index[va], f"r{r}"] = f
    out[[f"r{r}" for r in range(N_REPEATS)]] = out[[f"r{r}" for r in range(N_REPEATS)]].astype(int)
    FOLDS.parent.mkdir(exist_ok=True)
    out.to_csv(FOLDS, index=False)
    return out


def get_folds():
    return pd.read_csv(FOLDS) if FOLDS.exists() else make_folds()


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def pearson(y, p):
    return float(pearsonr(y, p)[0]) if np.std(p) > 0 else 0.0


def cv_predict(fit_predict, X, y, folds=None):
    """Run fit_predict(X_tr, y_tr, X_va) over every fold of every repeat.
    Returns OOF predictions (n_repeats x n) plus mean/std of per-repeat RMSE and Pearson."""
    folds = get_folds() if folds is None else folds
    reps = [c for c in folds.columns if c.startswith("r")]
    oof = np.zeros((len(reps), len(y)))
    for ri, c in enumerate(reps):
        for f in range(N_SPLITS):
            va = folds[c].to_numpy() == f
            oof[ri, va] = fit_predict(X[~va], y[~va], X[va])
    r = [rmse(y, o) for o in oof]
    p = [pearson(y, o) for o in oof]
    return oof, dict(rmse=np.mean(r), rmse_sd=np.std(r), pearson=np.mean(p), pearson_sd=np.std(p))


if __name__ == "__main__":
    f = make_folds()
    tr = load_train().merge(f, on="filename")
    print(tr.groupby("r0").label.agg(["count", "mean"]).round(3).to_string())
