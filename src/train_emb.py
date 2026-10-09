"""M2: audio-embedding branch. Ridge on layer-group embeddings, repeated CV, test predictions.

Layer groups (fixed in advance, not tuned per layer, to avoid overfitting the choice on 769 clips):
early / middle / late thirds of the encoder, and all layers averaged. Layer 0 (input embedding) is skipped.
Inside each training fold: standardise -> RidgeCV (alpha chosen by efficient leave-one-out on that fold only).

Usage: python src/train_emb.py [--emb cache/emb] [--exp e001]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from cv import DATA, ROOT, cv_predict, get_folds, pearson, rmse  # noqa: E402
from validate import validate_submission  # noqa: E402

ALPHAS = np.logspace(0, 6, 25)


def layer_groups(n_layers):
    """n_layers counts hidden states incl. layer 0. Returns {name: list of layer ids}."""
    L = np.arange(1, n_layers)
    thirds = np.array_split(L, 3)
    return {"early": list(thirds[0]), "middle": list(thirds[1]), "late": list(thirds[2]), "all": list(L)}


def fit_predict(Xtr, ytr, Xva):
    m = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
    return m.fit(Xtr, ytr).predict(Xva)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default=str(ROOT / "cache" / "emb"))
    ap.add_argument("--exp", default="e001")
    a = ap.parse_args()
    emb = Path(a.emb)
    order = pd.read_csv(emb / "emb_order.csv")
    tr = pd.read_csv(DATA / "train.csv")
    te = pd.read_csv(DATA / "test.csv")
    folds = get_folds()
    # align rows: train rows in folds order, test rows in test.csv order
    idx = {(s, f): i for i, (f, s) in enumerate(zip(order.filename, order.split))}
    itr = [idx[("train", f)] for f in folds.filename]
    ite = [idx[("test", f)] for f in te.filename]
    y = tr.set_index("filename").loc[folds.filename, "label"].to_numpy()

    feats, results = {}, []
    tags = [t for t in ["whisper", "wavlm"] if (emb / f"emb_{t}.npy").exists()]
    for tag in tags:
        E = np.load(emb / f"emb_{tag}.npy").astype(np.float32)  # (n, layers, 2d)
        for g, layers in layer_groups(E.shape[1]).items():
            X = E[:, layers].mean(1)
            feats[f"{tag}_{g}"] = X
            _, m = cv_predict(fit_predict, X[itr], y, folds)
            results.append(dict(features=f"{tag}_{g}", layers=f"{layers[0]}-{layers[-1]}", **m))
            print(f"{tag:8s} {g:7s} layers {layers[0]:2d}-{layers[-1]:2d}  "
                  f"RMSE {m['rmse']:.4f} ±{m['rmse_sd']:.4f}  Pearson {m['pearson']:.4f} ±{m['pearson_sd']:.4f}", flush=True)
    res = pd.DataFrame(results)
    combo = None
    if len(tags) == 2:
        # Combination: concatenate the best Whisper group and the best WavLM group.
        bw = res[res.features.str.startswith("whisper")].sort_values("rmse").features.iloc[0]
        bl = res[res.features.str.startswith("wavlm")].sort_values("rmse").features.iloc[0]
        combo = np.hstack([feats[bw], feats[bl]])
        _, m = cv_predict(fit_predict, combo[itr], y, folds)
        res.loc[len(res)] = dict(features=f"{bw}+{bl}", layers="concat", **m)
        print(f"concat {bw}+{bl}: RMSE {m['rmse']:.4f} ±{m['rmse_sd']:.4f}  Pearson {m['pearson']:.4f} ±{m['pearson_sd']:.4f}")

    best = res.sort_values("rmse").iloc[0]
    X = combo if "+" in best.features else feats[best.features]
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS)).fit(X[itr], y)
    train_rmse = rmse(y, model.predict(X[itr]))
    print(f"\nBEST {best.features}: CV RMSE {best.rmse:.4f}  CV Pearson {best.pearson:.4f}  "
          f"in-sample train RMSE {train_rmse:.4f}  alpha {model[-1].alpha_:.0f}")

    pred = np.clip(model.predict(X[ite]), 0, 5)
    out = ROOT / "outputs" / f"submission_{a.exp}.csv"
    pd.DataFrame({"filename": te.filename, "label": pred}).to_csv(out, index=False)
    validate_submission(out)
    print(f"test prediction mean {pred.mean():.3f} std {pred.std():.3f} (train label mean {y.mean():.3f} std {y.std():.3f})")
    res.to_csv(ROOT / "cache" / f"{a.exp}_results.csv", index=False)
    json.dump(dict(exp=a.exp, best=best.features, cv_rmse=best.rmse, cv_pearson=best.pearson,
                   train_rmse=train_rmse, alpha=float(model[-1].alpha_)),
              open(ROOT / "cache" / f"{a.exp}_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
