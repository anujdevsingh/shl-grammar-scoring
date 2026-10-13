"""Final engine v2, chosen on SPEAKER-GROUPED folds (D19): every voice is either in training or in validation.

Branches (Ridge, alpha picked by inner speaker-grouped CV, trained on scores 1-5):
  audio     frozen Whisper-large-v3 layers 29-32 + WavLM-large layers 17-24 (mean + std over time)
  enc_text  frozen DeBERTa-v3-large + RoBERTa-large transcript embeddings + the 42 transcript features
  qwen7b    frozen Qwen2.5-7B-Instruct hidden states (layers 16/19/22, mean over transcript tokens)
Stack: non-negative linear regression on the grouped out-of-fold predictions. Test prediction of each branch = mean
of its 5 fold models. Then same-voice smoothing: clips whose ECAPA voice vectors match (cosine > 0.7) are pulled
halfway to their voice's mean prediction. Finally clip to [0, 5]. (The test set has no invalid audio, D12.)

Usage: python src/final_grouped.py [--exp e016] [--smooth 0.5] [--fold_models cache/adapter5_run/adapter5_whisper]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).parent))
from cv import DATA, ROOT, get_folds  # noqa: E402
from grouped import grouped_folds, grouped_oof, metric  # noqa: E402
from validate import validate_submission  # noqa: E402


def rows_of(split_arr, file_arr, keys):
    idx = {(s, f): i for i, (s, f) in enumerate(zip(split_arr, file_arr))}
    return [idx[k] for k in keys]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="e016")
    ap.add_argument("--smooth", type=float, default=0.5)
    ap.add_argument("--fold_models", default="", help="comma list of folders with grouped-fold oof_fold{k}.csv / "
                    "test_fold{k}.csv (e.g. cache/adapter5_run/adapter5_whisper); test = mean of the 5 fold models")
    a = ap.parse_args()
    folds, te = get_folds(), pd.read_csv(DATA / "test.csv")
    ktr = [("train", f) for f in folds.filename]
    kte = [("test", f) for f in te.filename]
    y = pd.read_csv(DATA / "train.csv").set_index("filename").loc[folds.filename, "label"].to_numpy()
    use = y > 0
    fold, spk = grouped_folds()

    eo = pd.read_csv(ROOT / "cache" / "emb" / "emb_order.csv")
    W = np.load(ROOT / "cache" / "emb" / "emb_whisper.npy", mmap_mode="r")
    V = np.load(ROOT / "cache" / "emb" / "emb_wavlm.npy", mmap_mode="r")

    def audio(keys):
        r = rows_of(eo.split, eo.filename, keys)
        return np.hstack([W[r][:, 29:].astype(np.float32).mean(1), V[r][:, 17:].astype(np.float32).mean(1)])

    to = pd.read_csv(ROOT / "cache" / "text_emb" / "order.csv")
    D = np.load(ROOT / "cache" / "text_emb" / "deberta-v3-large.npy")
    R = np.load(ROOT / "cache" / "text_emb" / "roberta-large.npy")
    F = pd.read_parquet(ROOT / "cache" / "feat_text.parquet")
    cols = [c for c in F.columns if c not in ("split", "filename", "duration", "n_words", "n_sents")]
    Fi = F.set_index(["split", "filename"])

    def enc_text(keys):
        r = rows_of(to.split, to.filename, keys)
        return np.hstack([D[r], R[r], Fi.loc[keys, cols].to_numpy(float)])

    Q = np.load(ROOT / "cache" / "llm_text_run" / "llm_text" / "Qwen2.5-7B-Instruct.npz")
    ql = [list(Q["layers"]).index(v) for v in (16, 19, 22)]

    def qwen7b(keys):
        return Q["mean"][rows_of(Q["split"], Q["filename"], keys)][:, ql].astype(np.float32).mean(1)

    base = {}
    for name, fn in [("audio", audio), ("enc_text", enc_text), ("qwen7b", qwen7b)]:
        oof, test, al = grouped_oof(fn(ktr), y, fold, spk, use, fn(kte))
        base[name] = (oof, test)
        s, rm, r = metric(y[use], oof[use])
        print(f"{name:9s} grouped CV (1-5): RMSE {rm:.4f} r {r:.4f} metric {s:.4f}  alphas {np.round(al).astype(int).tolist()}")

    for d in filter(None, a.fold_models.split(",")):
        d = ROOT / d
        o = pd.concat([pd.read_csv(d / f"oof_fold{k}.csv") for k in range(5)]).set_index("filename")
        t = np.mean([pd.read_csv(d / f"test_fold{k}.csv").set_index("filename").loc[te.filename, "pred"].to_numpy()
                     for k in range(5)], 0)
        base[d.name] = (o.loc[folds.filename, "pred"].to_numpy(), t)
        s, rm, r = metric(y[use], base[d.name][0][use])
        print(f"{d.name:9s} grouped CV (1-5): RMSE {rm:.4f} r {r:.4f} metric {s:.4f}  (fold models)")

    names = list(base)
    P = np.c_[[base[n][0] for n in names]].T
    Pt = np.c_[[base[n][1] for n in names]].T
    st = np.zeros(len(y))
    for k in range(5):
        va = fold == k
        st[va] = LinearRegression(positive=True).fit(P[~va & use], y[~va & use]).predict(P[va])
    meta = LinearRegression(positive=True).fit(P[use], y[use])
    test = meta.predict(Pt)

    # same-voice smoothing: train side uses the grouped-fold voice ids, test side clusters the test voices
    E = np.load(ROOT / "cache" / "speaker_emb.npy")
    so = pd.read_csv(ROOT / "cache" / "speaker_order.csv")
    Et = E[rows_of(so.split, so.filename, kte)]
    tv = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                 distance_threshold=0.3).fit_predict(Et)

    def smooth(p, g):
        q = p.copy()
        for c in np.unique(g):
            i = g == c
            if i.sum() > 1:
                q[i] = (1 - a.smooth) * p[i] + a.smooth * p[i].mean()
        return q

    st_s, test_s = np.clip(smooth(st, spk), 0, 5), np.clip(smooth(test, tv), 0, 5)
    print(f"test clips sharing a voice with another test clip: {int(sum((tv == c).sum() > 1 for c in tv))}/216")
    for lab, p in [("stack", np.clip(st, 0, 5)), (f"stack + smoothing {a.smooth}", st_s)]:
        s, rm, r = metric(y[use], p[use])
        fr = [metric(y[use & (fold == k)], p[use & (fold == k)])[0] for k in range(5)]
        print(f"{lab:24s} grouped CV (1-5): RMSE {rm:.4f} r {r:.4f} metric {s:.4f}  folds {np.round(fr, 4).tolist()}")
    w = dict(zip(names, map(float, meta.coef_)))
    print("stack weights:", {k: round(v, 3) for k, v in w.items()}, "intercept %.3f" % meta.intercept_)

    out = ROOT / "outputs" / f"submission_{a.exp}.csv"
    pd.DataFrame({"filename": te.filename, "label": test_s}).to_csv(out, index=False)
    validate_submission(out)
    s, rm, r = metric(y[use], st_s[use])
    np.savez(ROOT / "cache" / f"final_{a.exp}.npz", y=y, fold=fold, spk=spk, stack_oof=st_s, test=test_s,
             names=np.array(names), P=P, Pt=Pt)
    json.dump(dict(exp=a.exp, weights=w, intercept=float(meta.intercept_), smooth=a.smooth, grouped_metric=s,
                   grouped_rmse=rm, grouped_pearson=r), open(ROOT / "cache" / f"final_{a.exp}.json", "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
