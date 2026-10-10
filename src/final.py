"""Final engine: stack every branch on the same folds and write the final submission.

Base models (out-of-fold predictions on repeat r0 of the fixed folds; fine-tuning was run on r0 only):
  frozen     Ridge on frozen Whisper-large-v3 + WavLM-large embeddings      (src/train_emb.py, e002)
  finetuned  WavLM-large fine-tuned on our scores, 5 fold models             (src/finetune.py)
  text       Ridge on fluency + grammar transcript features                  (src/train_text.py, e003)
  judge      Ridge on the Qwen2-Audio rubric-judge scores, if available      (src/judge.py)
Stack: non-negative linear regression on the base predictions, evaluated fold by fold (fit on 4 folds, score the 5th).
Training RMSE (in-sample): every refittable base model is refit on all 769 clips and predicts them; the fine-tuned
component has no all-data model, so its out-of-fold predictions are used there (stated wherever reported).

Writes outputs/submission_<exp>.csv and cache/final_<exp>.npz / .json for the notebook.
Usage: python src/final.py [--exp final] [--no-judge] [--drop name,name]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from cv import DATA, N_SPLITS, ROOT, get_folds, pearson, rmse  # noqa: E402
from train_emb import fit_predict as emb_fp  # noqa: E402
from train_text import ridge_fp  # noqa: E402
from validate import validate_submission  # noqa: E402


def oof_and_full(fp, Xtr, y, Xte, fold):
    """Out-of-fold predictions on r0, plus in-sample train and test predictions from a model fit on all train."""
    oof = np.zeros(len(y))
    for k in range(N_SPLITS):
        va = fold == k
        oof[va] = fp(Xtr[~va], y[~va], Xtr[va])
    return oof, fp(Xtr, y, Xtr), fp(Xtr, y, Xte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="final")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--drop", default="")
    a = ap.parse_args()
    folds = get_folds()
    tr, te = pd.read_csv(DATA / "train.csv"), pd.read_csv(DATA / "test.csv")
    y = tr.set_index("filename").loc[folds.filename, "label"].to_numpy()
    fold = folds.r0.to_numpy()
    base = {}  # name -> (oof, in-sample train, test)

    # frozen audio embeddings
    emb = ROOT / "cache" / "emb"
    order = pd.read_csv(emb / "emb_order.csv")
    idx = {(s, f): i for i, (f, s) in enumerate(zip(order.filename, order.split))}
    A = np.hstack([np.load(emb / f"emb_{t}.npy").astype(np.float32)[:, 1:].mean(1) for t in ["whisper", "wavlm"]])
    base["frozen"] = oof_and_full(emb_fp, A[[idx[("train", f)] for f in folds.filename]], y,
                                  A[[idx[("test", f)] for f in te.filename]], fold)

    # fine-tuned WavLM (5 fold models): OOF from each fold's model; test = mean of the 5 models
    ft = ROOT / "cache" / "ft"
    o = pd.concat([pd.read_csv(ft / f"oof_fold{k}.csv") for k in range(N_SPLITS)]).set_index("filename")
    t = np.mean([pd.read_csv(ft / f"test_fold{k}.csv").set_index("filename").loc[te.filename, "pred"].to_numpy()
                 for k in range(N_SPLITS)], 0)
    p = o.loc[folds.filename, "pred"].to_numpy()
    base["finetuned"] = (p, p, t)  # no all-data model: in-sample slot uses OOF

    # transcript features
    F = pd.read_parquet(ROOT / "cache" / "feat_text.parquet")
    cols = [c for c in F.columns if c not in ("split", "filename", "duration", "n_words", "n_sents")]
    base["text"] = oof_and_full(ridge_fp, F[F.split == "train"].set_index("filename").loc[folds.filename, cols].to_numpy(float), y,
                                F[F.split == "test"].set_index("filename").loc[te.filename, cols].to_numpy(float), fold)

    # audio-LLM judge
    jp = ROOT / "cache" / "judge" / "judge.csv"
    if jp.exists() and not a.no_judge:
        J = pd.read_csv(jp)
        jc = [c for c in J.columns if c.startswith("judge_")]
        base["judge"] = oof_and_full(ridge_fp, J[J.split == "train"].set_index("filename").loc[folds.filename, jc].to_numpy(float), y,
                                     J[J.split == "test"].set_index("filename").loc[te.filename, jc].to_numpy(float), fold)
        print("judge raw score Pearson with label: " + ", ".join(
            f"{c} {pearson(y, J[J.split == 'train'].set_index('filename').loc[folds.filename, c]):.3f}"
            for c in ["judge_rubric", "judge_short"]))
    for d in filter(None, a.drop.split(",")):
        base.pop(d)

    names = list(base)
    P = np.c_[[base[n][0] for n in names]].T
    st = np.zeros(len(y))
    for k in range(N_SPLITS):
        va = fold == k
        st[va] = LinearRegression(positive=True).fit(P[~va], y[~va]).predict(P[va])
    meta = LinearRegression(positive=True).fit(P, y)
    st = np.clip(st, 0, 5)
    train_in = np.clip(meta.predict(np.c_[[base[n][1] for n in names]].T), 0, 5)
    test = np.clip(meta.predict(np.c_[[base[n][2] for n in names]].T), 0, 5)

    base_rmse = rmse(y, np.full_like(y, y.mean()))
    rows = [dict(model="mean baseline", cv_rmse=base_rmse, cv_pearson=0.0, train_rmse=base_rmse)]
    for n in names:
        rows.append(dict(model=n, cv_rmse=rmse(y, base[n][0]), cv_pearson=pearson(y, base[n][0]),
                         train_rmse=rmse(y, base[n][1]) if n != "finetuned" else np.nan))
    rows.append(dict(model="STACK (final)", cv_rmse=rmse(y, st), cv_pearson=pearson(y, st), train_rmse=rmse(y, train_in)))
    res = pd.DataFrame(rows)
    fold_rmse = [rmse(y[fold == k], st[fold == k]) for k in range(N_SPLITS)]
    print(res.round(4).to_string(index=False))
    print("stack per-fold RMSE:", np.round(fold_rmse, 4))
    w = dict(zip(names, map(float, meta.coef_)))
    print("stack weights:", {k: round(v, 3) for k, v in w.items()}, "intercept %.3f" % meta.intercept_)

    out = ROOT / "outputs" / f"submission_{a.exp}.csv"
    pd.DataFrame({"filename": te.filename, "label": test}).to_csv(out, index=False)
    validate_submission(out)
    np.savez(ROOT / "cache" / f"final_{a.exp}.npz", y=y, fold=fold, stack_oof=st, train_in=train_in, test=test,
             names=np.array(names), P=P)
    res.to_csv(ROOT / "cache" / f"final_{a.exp}_table.csv", index=False)
    json.dump(dict(exp=a.exp, weights=w, intercept=float(meta.intercept_), cv_rmse=rmse(y, st), cv_pearson=pearson(y, st),
                   train_rmse=rmse(y, train_in), fold_rmse=fold_rmse), open(ROOT / "cache" / f"final_{a.exp}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
