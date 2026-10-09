"""Evaluate the fine-tuned WavLM (Stage G) and stack it with the frozen-embedding Ridge (e002).

Fine-tuning ran on repeat r0 of the fixed folds only (5 folds x 1), so every comparison here uses r0:
- frozen e002 Ridge, out-of-fold on r0
- fine-tuned WavLM, out-of-fold on r0 (final epoch of each fold; nothing was chosen on the validation fold)
- stack: non-negative linear blend, itself evaluated fold-by-fold on r0 (fitted on the other 4 folds only)
Test prediction of the fine-tuned model = mean of the 5 fold models.

Usage: python src/eval_finetune.py --ft cache/ft [--exp e006]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).parent))
from cv import DATA, N_SPLITS, ROOT, get_folds, pearson, rmse  # noqa: E402
from validate import validate_submission  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft", default=str(ROOT / "cache" / "ft"))
    ap.add_argument("--exp", default="e006")
    a = ap.parse_args()
    ft = Path(a.ft)
    folds = get_folds()
    tr, te = pd.read_csv(DATA / "train.csv"), pd.read_csv(DATA / "test.csv")
    y = tr.set_index("filename").loc[folds.filename, "label"].to_numpy()
    fold = folds.r0.to_numpy()

    logs = pd.concat([pd.read_csv(ft / f"log_fold{k}.csv") for k in range(N_SPLITS)])
    print("per-epoch validation RMSE (rows = epoch, cols = fold):")
    print(logs.pivot(index="epoch", columns="fold", values="val_rmse").round(3).to_string())

    oof_ft = pd.concat([pd.read_csv(ft / f"oof_fold{k}.csv") for k in range(N_SPLITS)]).set_index("filename")
    p_ft = oof_ft.loc[folds.filename, "pred"].to_numpy()
    p_fr = np.load(ROOT / "cache" / "oof_e002_A.npy")[0]  # frozen Ridge, repeat r0
    t_ft = np.mean([pd.read_csv(ft / f"test_fold{k}.csv").set_index("filename").loc[te.filename, "pred"].to_numpy()
                    for k in range(N_SPLITS)], axis=0)
    t_fr = pd.read_csv(ROOT / "outputs" / "submission_e002.csv").set_index("filename").loc[te.filename, "label"].to_numpy()

    def fold_rmse(p):
        return np.std([rmse(y[fold == k], p[fold == k]) for k in range(N_SPLITS)])

    P = np.c_[p_fr, p_ft]
    st = np.zeros(len(y))
    for k in range(N_SPLITS):
        va = fold == k
        st[va] = LinearRegression(positive=True).fit(P[~va], y[~va]).predict(P[va])
    meta = LinearRegression(positive=True).fit(P, y)

    rows = [("frozen e002 (r0)", p_fr), ("fine-tuned WavLM", p_ft), ("stack frozen+fine-tuned", st)]
    print("\nCV on repeat r0 (5 folds):")
    for name, p in rows:
        print(f"  {name:26s} RMSE {rmse(y, p):.4f} (fold sd {fold_rmse(p):.4f})  Pearson {pearson(y, p):.4f}")
    print(f"stack weights: frozen {meta.coef_[0]:.3f}  fine-tuned {meta.coef_[1]:.3f}  intercept {meta.intercept_:.3f}")
    gain = rmse(y, p_fr) - rmse(y, st)
    print(f"gain of stack over frozen: {gain:.4f}")

    pred = np.clip(meta.predict(np.c_[t_fr, t_ft]), 0, 5)
    out = ROOT / "outputs" / f"submission_{a.exp}.csv"
    pd.DataFrame({"filename": te.filename, "label": pred}).to_csv(out, index=False)
    validate_submission(out)
    print(f"test prediction mean {pred.mean():.3f} std {pred.std():.3f}; "
          f"corr(frozen, fine-tuned) on test {np.corrcoef(t_fr, t_ft)[0, 1]:.3f}")
    pd.DataFrame({"filename": te.filename, "label": np.clip(t_ft, 0, 5)}).to_csv(
        ROOT / "outputs" / f"submission_{a.exp}_ftonly.csv", index=False)


if __name__ == "__main__":
    main()
