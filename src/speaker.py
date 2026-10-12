"""Speaker embeddings: does the same voice appear in several clips (train and test)?

ECAPA-TDNN speaker-verification model (SpeechBrain, trained on VoxCeleb, Apache-2.0) turns each clip into a
192-dim voice vector. Clips from the same person have high cosine similarity.
Output: cache/speaker_emb.npy (L2-normalised) + cache/speaker_order.csv (split, filename).

Usage: python src/speaker.py [--data DIR --out DIR]   (runs on CPU; on Kaggle via kaggle/build_speaker.sh)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
from cv import DATA, ROOT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--out", default=str(ROOT / "cache"))
    a = ap.parse_args()
    data, out_dir = Path(a.data), Path(a.out)
    from speechbrain.inference.speaker import EncoderClassifier
    local = out_dir / "ecapa_model"  # use a local copy of the weights if one exists
    src = str(local) if (local / "hyperparams.yaml").exists() else "speechbrain/spkrec-ecapa-voxceleb"
    model = EncoderClassifier.from_hparams(source=src, savedir=str(local), run_opts={"device": "cpu"})
    rows = [("train", f) for f in pd.read_csv(data / "train.csv").filename] + \
           [("test", f) for f in pd.read_csv(data / "test.csv").filename]
    out = np.zeros((len(rows), 192), np.float32)
    for i, (split, f) in enumerate(rows):
        x, sr = sf.read(data / split / f, dtype="float32", always_2d=True)
        x = x.mean(1)
        x = x / (np.sqrt(np.mean(x ** 2)) + 1e-8) * 0.1  # RMS normalisation, as elsewhere
        with torch.no_grad():
            e = model.encode_batch(torch.from_numpy(x)[None]).squeeze().numpy()
        out[i] = e / np.linalg.norm(e)
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(rows)}", flush=True)
    np.save(out_dir / "speaker_emb.npy", out)
    pd.DataFrame(rows, columns=["split", "filename"]).to_csv(out_dir / "speaker_order.csv", index=False)


if __name__ == "__main__":
    main()
