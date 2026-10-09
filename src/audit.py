"""M1 data audit: CSV checks, label distribution, audio properties, mean baseline.
Writes only aggregate numbers to stdout and a per-file table to cache/ (gitignored)."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "raw" / "Dataset_Final"
CACHE = ROOT / "cache"
CACHE.mkdir(exist_ok=True)

train = pd.read_csv(D / "train.csv")
test = pd.read_csv(D / "test.csv")
sample = pd.read_csv(D / "sample_submission.csv")

print("== CSVs")
for name, df in [("train", train), ("test", test), ("sample", sample)]:
    print(f"{name}: {df.shape}, columns={list(df.columns)}, dtypes={dict(df.dtypes.astype(str))}, "
          f"dup filenames={df.filename.duplicated().sum()}")
print("sample ids in test.csv:", sample.filename.isin(test.filename).sum(), "/", len(sample))
print("test ids in sample:", test.filename.isin(sample.filename).sum(), "/", len(test))
print("sample order == test order (on shared ids):",
      list(test.filename[test.filename.isin(sample.filename)]) == list(sample.filename))
print("train/test filename overlap:", len(set(train.filename) & set(test.filename)))

print("\n== Labels (train)")
y = train.label.astype(float)
print(y.value_counts().sort_index().to_string())
print(f"n={len(y)} mean={y.mean():.4f} std(pop)={y.std(ddof=0):.4f} min={y.min()} max={y.max()} skew={y.skew():.3f}")
print(f"Mean baseline: RMSE={np.sqrt(((y - y.mean()) ** 2).mean()):.4f}, Pearson=0 (constant prediction)")

print("\n== Audio")
rows = []
for split, df in [("train", train), ("test", test)]:
    for fn in df.filename:
        p = D / split / fn
        info = sf.info(p)
        x, sr = sf.read(p, dtype="float32", always_2d=True)
        m = x.mean(axis=1)
        frame = sr // 50  # 20 ms frames
        n = len(m) // frame
        rms = np.sqrt((m[: n * frame].reshape(n, frame) ** 2).mean(axis=1) + 1e-12)
        db = 20 * np.log10(rms)
        rows.append(dict(
            split=split, filename=fn, sr=sr, channels=info.channels, subtype=info.subtype,
            duration=len(m) / sr, peak=float(np.abs(x).max()),
            clip_frac=float((np.abs(x) >= 0.999).mean()),
            rms_db=float(20 * np.log10(np.sqrt((m ** 2).mean()) + 1e-12)),
            silence_ratio=float((db < db.max() - 40).mean()),
            md5=hashlib.md5(p.read_bytes()).hexdigest(),
        ))
    print(f"{split} on disk: {len(list((D / split).glob('*.wav')))} wav, listed in csv: {len(df)}")
a = pd.DataFrame(rows)
a.to_csv(CACHE / "audit_audio.csv", index=False)

for split, g in a.groupby("split"):
    print(f"\n-- {split}")
    print("sample rates:", g.sr.value_counts().to_dict(), "| channels:", g.channels.value_counts().to_dict(),
          "| subtypes:", g.subtype.value_counts().to_dict())
    print(g[["duration", "peak", "clip_frac", "rms_db", "silence_ratio"]].describe(
        percentiles=[.05, .5, .95]).round(3).to_string())

dups = a[a.md5.duplicated(keep=False)].sort_values("md5")
print(f"\nIdentical audio files (by md5): {len(dups)} files in {dups.md5.nunique()} groups")
if len(dups):
    print("splits involved:", dups.groupby("md5").split.apply(lambda s: "+".join(sorted(s))).value_counts().to_dict())

tr = a[a.split == "train"].merge(train, on="filename")
print("\nPearson(label, duration)=%.3f  (label, silence_ratio)=%.3f  (label, rms_db)=%.3f" % (
    tr.label.corr(tr.duration), tr.label.corr(tr.silence_ratio), tr.label.corr(tr.rms_db)))
nums = train.filename.str.extract(r"(\d+)")[0].astype(int)
print("Pearson(label, file number)=%.3f  (checks for ordering leaks)" % y.corr(nums))
for name, df in [("train", train), ("test", test), ("sample", sample)]:
    n = df.filename.str.extract(r"(\d+)")[0].astype(int)
    print(f"file number range {name}: {n.min()}-{n.max()}")
