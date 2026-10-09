"""M3: transcript branch (fluency + grammar features) and the stack with the audio branch.

Base models (same fixed folds as everything else):
- text_ridge: standardise -> RidgeCV on the fluency/grammar features (missing values filled with the train median).
- text_lgbm:  LightGBM with strong regularisation (shallow trees, many samples per leaf, subsampling).
- audio:      the M2 audio-embedding Ridge (Whisper-large-v3 + WavLM-large, all layers).
Stack: a linear regression with non-negative weights on the base models' out-of-fold predictions.
It is evaluated with the same folds (meta-model fitted on the training part of each fold only), so the stack
score is honest. Final test prediction: base models fitted on all train, combined with the stack weights.

Usage: python src/train_text.py [--exp e003]
"""
import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from cv import DATA, N_SPLITS, ROOT, cv_predict, get_folds, pearson, rmse  # noqa: E402
from train_emb import ALPHAS, fit_predict as audio_fit_predict  # noqa: E402
from validate import validate_submission  # noqa: E402


def ridge_fp(Xtr, ytr, Xva):
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25)))
    return m.fit(Xtr, ytr).predict(Xva)


LGB = dict(n_estimators=600, learning_rate=0.02, num_leaves=8, max_depth=4, min_child_samples=20,
           subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, verbose=-1, random_state=0)


def lgbm_fp(Xtr, ytr, Xva):
    return lgb.LGBMRegressor(**LGB).fit(Xtr, ytr).predict(Xva)


def stack_cv(P, y, folds):
    """P: (n_models, n_repeats, n) OOF predictions. Fit non-negative linear meta-model per fold."""
    reps = [c for c in folds.columns if c.startswith("r")]
    out = np.zeros(P.shape[1:])
    for ri, c in enumerate(reps):
        Z = P[:, ri].T
        for f in range(N_SPLITS):
            va = folds[c].to_numpy() == f
            meta = LinearRegression(positive=True).fit(Z[~va], y[~va])
            out[ri, va] = meta.predict(Z[va])
    return out


def summary(name, oof, y):
    r = [rmse(y, o) for o in oof]
    p = [pearson(y, o) for o in oof]
    print(f"{name:12s} RMSE {np.mean(r):.4f} ±{np.std(r):.4f}  Pearson {np.mean(p):.4f} ±{np.std(p):.4f}", flush=True)
    return dict(model=name, rmse=np.mean(r), rmse_sd=np.std(r), pearson=np.mean(p), pearson_sd=np.std(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="e003")
    ap.add_argument("--feat", default=str(ROOT / "cache" / "feat_text.parquet"))
    a = ap.parse_args()
    folds = get_folds()
    tr, te = pd.read_csv(DATA / "train.csv"), pd.read_csv(DATA / "test.csv")
    y = tr.set_index("filename").loc[folds.filename, "label"].to_numpy()

    F = pd.read_parquet(a.feat)
    # Length-dependent counts are dropped: test clips are shorter than train (median 45 s vs 60 s),
    # so raw duration / word / sentence counts would shift between train and test.
    LENGTH = {"duration", "n_words", "n_sents"}
    cols = [c for c in F.columns if c not in ("split", "filename") and c not in LENGTH]
    Ftr = F[F.split == "train"].set_index("filename").loc[folds.filename, cols].to_numpy(float)
    Fte = F[F.split == "test"].set_index("filename").loc[te.filename, cols].to_numpy(float)

    emb = ROOT / "cache" / "emb"
    order = pd.read_csv(emb / "emb_order.csv")
    idx = {(s, f): i for i, (f, s) in enumerate(zip(order.filename, order.split))}
    A = np.hstack([np.load(emb / f"emb_{t}.npy").astype(np.float32)[:, 1:].mean(1) for t in ["whisper", "wavlm"]])
    Atr, Ate = A[[idx[("train", f)] for f in folds.filename]], A[[idx[("test", f)] for f in te.filename]]

    bases = {"text_ridge": (ridge_fp, Ftr, Fte), "text_lgbm": (lgbm_fp, Ftr, Fte), "audio": (audio_fit_predict, Atr, Ate)}
    oofs, rows = {}, []
    for name, (fp, Xtr, _) in bases.items():
        oofs[name], _ = cv_predict(fp, Xtr, y, folds)
        rows.append(summary(name, oofs[name], y))

    names = list(bases)
    P = np.stack([oofs[n] for n in names])
    st = stack_cv(P, y, folds)
    rows.append(summary("stack", st, y))
    for drop in names:  # how much each branch adds
        keep = [i for i, n in enumerate(names) if n != drop]
        rows.append(summary(f"stack-{drop}", stack_cv(P[keep], y, folds), y))

    # final: meta weights on all OOF (averaged over repeats), base models refit on all train
    meta = LinearRegression(positive=True).fit(P.mean(1).T, y)
    w = dict(zip(names, meta.coef_.round(3)))
    print("stack weights:", w, "intercept %.3f" % meta.intercept_)
    full_tr = np.stack([fp(Xtr, y, Xtr) for fp, Xtr, _ in bases.values()]).T
    full_te = np.stack([fp(Xtr, y, Xte) for fp, Xtr, Xte in bases.values()]).T
    train_rmse = rmse(y, meta.predict(full_tr))
    pred = np.clip(meta.predict(full_te), 0, 5)
    out = ROOT / "outputs" / f"submission_{a.exp}.csv"
    pd.DataFrame({"filename": te.filename, "label": pred}).to_csv(out, index=False)
    validate_submission(out)
    best = rows[len(names)]
    print(f"\nSTACK: CV RMSE {best['rmse']:.4f}  CV Pearson {best['pearson']:.4f}  in-sample train RMSE {train_rmse:.4f}")
    print(f"test prediction mean {pred.mean():.3f} std {pred.std():.3f}")

    # feature importance (gain) of the LightGBM text model on all train, for the report
    imp = pd.Series(lgb.LGBMRegressor(**LGB).fit(Ftr, y).booster_.feature_importance("gain"), index=cols)
    print("\ntop text features (LightGBM gain):\n" + (imp / imp.sum()).sort_values(ascending=False).head(12).round(3).to_string())
    corr = pd.Series({c: pearson(y, np.nan_to_num(Ftr[:, i], nan=np.nanmedian(Ftr[:, i]))) for i, c in enumerate(cols)})
    print("\nstrongest single-feature Pearson with label:\n" + corr.reindex(corr.abs().sort_values(ascending=False).index).head(12).round(3).to_string())

    pd.DataFrame(rows).to_csv(ROOT / "cache" / f"{a.exp}_results.csv", index=False)
    np.save(ROOT / "cache" / f"oof_{a.exp}.npy", P)
    json.dump(dict(exp=a.exp, weights={k: float(v) for k, v in w.items()}, cv_rmse=best["rmse"],
                   cv_pearson=best["pearson"], train_rmse=train_rmse), open(ROOT / "cache" / f"{a.exp}_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
