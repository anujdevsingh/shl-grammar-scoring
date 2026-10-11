"""Pseudo-labels for semi-supervised training of the Whisper adapter (Stage J), built without leakage.

For each CV fold k, the teacher may not have seen fold k's labels:
- held-out fold k clips: the e011 stack's out-of-fold predictions (base models trained on the other folds,
  stack weights fitted on the other folds).
- test clips: fold-k base models only (Ridge refit on the other folds; fine-tuned models' fold-k checkpoints),
  combined with stack weights fitted on the other folds' out-of-fold predictions.
The student then trains on real labels (other folds) + these pseudo-labels, and is scored on fold k's real labels.
Output: cache/pseudo/pseudo_fold{k}.csv with split, filename, label.

Usage: python src/pseudo.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).parent))
from cv import DATA, N_SPLITS, ROOT, get_folds  # noqa: E402
from train_emb import fit_predict as emb_fp  # noqa: E402
from train_text import ridge_fp  # noqa: E402

FOLD_DIRS = {"finetuned": "ft", "audiollm": "alm", "adapter_whisper": "adapter_whisper", "adapter_wavlm": "adapter_wavlm"}


def main():
    d = np.load(ROOT / "cache" / "final_e011.npz", allow_pickle=True)
    y, fold, P, st, names = d["y"], d["fold"], d["P"], d["stack_oof"], list(d["names"])
    folds = get_folds()
    te = pd.read_csv(DATA / "test.csv")
    use = y > 0
    emb = ROOT / "cache" / "emb"
    order = pd.read_csv(emb / "emb_order.csv")
    idx = {(s, f): i for i, (f, s) in enumerate(zip(order.filename, order.split))}
    A = np.hstack([np.load(emb / f"emb_{t}.npy").astype(np.float32)[:, 1:].mean(1) for t in ["whisper", "wavlm"]])
    Atr, Ate = A[[idx[("train", f)] for f in folds.filename]], A[[idx[("test", f)] for f in te.filename]]
    F = pd.read_parquet(ROOT / "cache" / "feat_text.parquet")
    cols = [c for c in F.columns if c not in ("split", "filename", "duration", "n_words", "n_sents")]
    Ttr = F[F.split == "train"].set_index("filename").loc[folds.filename, cols].to_numpy(float)
    Tte = F[F.split == "test"].set_index("filename").loc[te.filename, cols].to_numpy(float)
    out = ROOT / "cache" / "pseudo"
    out.mkdir(exist_ok=True)
    for k in range(N_SPLITS):
        tr_k = (fold != k) & use
        test_base = []
        for n in names:
            if n == "frozen":
                test_base.append(emb_fp(Atr[tr_k], y[tr_k], Ate))
            elif n == "text":
                test_base.append(ridge_fp(Ttr[tr_k], y[tr_k], Tte))
            else:
                t = pd.read_csv(ROOT / "cache" / FOLD_DIRS[n] / f"test_fold{k}.csv").set_index("filename")
                test_base.append(t.loc[te.filename, "pred"].to_numpy())
        meta = LinearRegression(positive=True).fit(P[tr_k], y[tr_k])
        t_lab = np.clip(meta.predict(np.c_[test_base].T), 1, 5)
        v = fold == k
        v_lab = np.clip(st[v], 1, 5)
        df = pd.concat([pd.DataFrame({"split": "train", "filename": folds.filename[v].to_numpy(), "label": v_lab}),
                        pd.DataFrame({"split": "test", "filename": te.filename, "label": t_lab})])
        df = df[~((df.split == "train") & np.r_[y[v] == 0, np.zeros(len(te), bool)])]  # skip invalid (label-0) audio
        df.to_csv(out / f"pseudo_fold{k}.csv", index=False)
        print(f"fold {k}: {len(df)} pseudo-labelled clips (held-out {int((df.split == 'train').sum())}, test {len(te)}), "
              f"test label mean {t_lab.mean():.2f}")


if __name__ == "__main__":
    main()
