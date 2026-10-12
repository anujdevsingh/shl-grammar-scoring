"""Speaker-grouped validation (D19): no voice appears in both the training and the validation part of a fold.

- folds: cache/folds_speaker.csv (column g0 = fold, speaker = voice cluster), made from ECAPA voice vectors.
- Ridge alpha is chosen by an INNER speaker-grouped CV. RidgeCV's built-in leave-one-out choice is leaky here: the
  left-out clip's voice is still in the training rows, so it prefers too little regularisation (voice memory).
- metric(): the leaderboard formula found in D19's source, 0.6 * RMSE + 0.4 * (1 - Pearson); lower is better.
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cv import ROOT, get_folds

ALPHAS = np.logspace(0, 6, 25)


def grouped_folds():
    """(fold, speaker) arrays in the order of cache/folds.csv."""
    f = pd.read_csv(ROOT / "cache" / "folds_speaker.csv").set_index("filename").loc[get_folds().filename]
    return f.g0.to_numpy(), f.speaker.to_numpy()


def metric(y, p):
    r = float(np.corrcoef(y, p)[0, 1])
    rm = float(np.sqrt(np.mean((y - p) ** 2)))
    return 0.6 * rm + 0.4 * (1 - r), rm, r


def _pipe(alpha):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha))


def grouped_ridge_fp(Xtr, ytr, Xva, gtr, alphas=ALPHAS):
    """Ridge with alpha picked by 4-fold speaker-grouped CV on the training rows only."""
    best, best_a = np.inf, alphas[0]
    for a in alphas:
        err = 0.0
        for i, v in GroupKFold(4).split(Xtr, ytr, gtr):
            err += ((_pipe(a).fit(Xtr[i], ytr[i]).predict(Xtr[v]) - ytr[v]) ** 2).sum()
        if err < best:
            best, best_a = err, a
    return _pipe(best_a).fit(Xtr, ytr).predict(Xva), best_a


def grouped_oof(X, y, fold, spk, use, Xte=None):
    """OOF predictions on grouped folds (training on `use` rows) and, if Xte is given, the mean of the 5 fold models'
    test predictions. Returns oof, test, list of chosen alphas."""
    oof, tes, al = np.zeros(len(y)), [], []
    for k in range(5):
        tr = (fold != k) & use
        va = fold == k
        Xo = X[va] if Xte is None else np.vstack([X[va], Xte])
        p, a = grouped_ridge_fp(X[tr], y[tr], Xo, spk[tr])
        oof[va] = p[:va.sum()]
        if Xte is not None:
            tes.append(p[va.sum():])
        al.append(a)
    return oof, (np.mean(tes, 0) if tes else None), al
