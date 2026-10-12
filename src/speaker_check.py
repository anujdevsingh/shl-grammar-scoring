"""Speaker check: how often does a voice repeat, across train and test, and do repeated voices share scores?

Reads cache/speaker_emb.npy (src/speaker.py). Prints only aggregate numbers.
Usage: python src/speaker_check.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

sys.path.insert(0, str(Path(__file__).parent))
from cv import DATA, ROOT  # noqa: E402


def main():
    E = np.load(ROOT / "cache" / "speaker_emb.npy")
    order = pd.read_csv(ROOT / "cache" / "speaker_order.csv")
    lab = pd.read_csv(DATA / "train.csv").set_index("filename").label
    y = np.array([lab[f] if s == "train" else np.nan for s, f in zip(order.split, order.filename)])
    tr, te = (order.split == "train").to_numpy(), (order.split == "test").to_numpy()
    S = E @ E.T
    np.fill_diagonal(S, -1)
    nn = S.max(1)
    print("nearest-neighbour cosine similarity, percentiles 10/25/50/75/90:")
    for name, m in [("train", tr), ("test", te)]:
        print(f"  {name}: " + " ".join(f"{v:.2f}" for v in np.percentile(nn[m], [10, 25, 50, 75, 90])))
    print("  histogram of NN similarity (all clips):",
          np.histogram(nn, bins=[-1, .2, .3, .4, .5, .6, .7, .8, .9, 1])[0].tolist(), "bins <.2,.3,...,.9,1")
    # test clips: is their nearest neighbour a train clip?
    St = S[te][:, tr].max(1)
    print("test -> nearest TRAIN clip similarity percentiles 10/50/90:", np.round(np.percentile(St, [10, 50, 90]), 2))
    nz = tr & (y > 0)
    for t in (0.5, 0.6, 0.7):
        cl = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                     distance_threshold=1 - t).fit_predict(E)
        c = pd.DataFrame(dict(c=cl, tr=tr, te=te, y=y))
        g = c.groupby("c").agg(n=("c", "size"), ntr=("tr", "sum"), nte=("te", "sum"))
        multi = g[g.n > 1]
        mixed = g[(g.ntr > 0) & (g.nte > 0)]
        te_with_train = c[te & c.c.isin(mixed.index)].shape[0]
        # label agreement inside clusters (train clips scored 1-5 only)
        cz = c[nz]
        within = cz.groupby("c").y.transform(lambda v: v.std(ddof=0) if len(v) > 1 else np.nan)
        loo = cz.groupby("c").y.transform(lambda v: (v.sum() - v) / (len(v) - 1) if len(v) > 1 else np.nan)
        ok = loo.notna()
        r = np.corrcoef(cz.y[ok], loo[ok])[0, 1] if ok.sum() > 2 else np.nan
        print(f"\nthreshold cos>{t}: {len(g)} clusters for {len(c)} clips; clusters with >1 clip: {len(multi)} "
              f"(holding {multi.n.sum()} clips); largest {g.n.max()}")
        print(f"  mixed train+test clusters: {len(mixed)}; test clips sharing a cluster with train: {te_with_train}/{te.sum()}")
        print(f"  train 1-5 clips with another same-cluster train clip: {ok.sum()}; mean within-cluster label std "
              f"{np.nanmean(within[ok]):.3f} (overall std {cz.y.std(ddof=0):.3f}); corr(label, mean of same-cluster "
              f"others) {r:.3f}")


if __name__ == "__main__":
    main()
