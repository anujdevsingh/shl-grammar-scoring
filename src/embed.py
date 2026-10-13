"""Stage A: layer-wise audio embeddings (Whisper encoder, WavLM).

For every clip and every hidden layer we keep the mean and the std over time,
so each model gives an array of shape (n_clips, n_layers, 2 * hidden_size).
- Whisper's encoder takes at most 30 s, so clips are cut into 30 s windows.
  Only frames that hold real audio are pooled (the padding is ignored), and
  windows are combined weighted by their length.
- WavLM takes the whole clip at once.
Runs on Kaggle GPU (heavy models) or locally with small models for testing.
Saves float16 .npy + the file order, every CKPT clips, and resumes from a checkpoint.
Multi-GPU: run one process per GPU with --shard i --nshards N (each takes a contiguous block of files),
then once with --merge to join the blocks in order.

Crop mode (augmentation, D21): --crops K embeds K random crops of every TRAIN clip, each 40-50 s long (test-like
length; test clips have median 45 s), and keeps only the average of the chosen layers (--layer_avg) to stay small.
Output emb_<tag>_crops.npy (n_train*K, 2*d) + crops_order.csv (filename, k, start_s, length_s).

Usage: python embed.py --data DIR --out DIR [--whisper openai/whisper-large-v3] [--wavlm microsoft/wavlm-large]
                       [--limit N] [--shard I --nshards N | --merge]
                       [--crops 3 --crop_min 40 --crop_max 50 --layer_avg whisper:29-32,wavlm:17-24]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel, WavLMModel, WhisperFeatureExtractor, WhisperModel

SR = 16000
WIN = 30 * SR  # Whisper window: 30 s
CKPT = 50


def device_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def load(path, start=0.0, length=None):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    assert sr == SR, f"{path}: sample rate {sr}"
    x = x.mean(axis=1)
    if length is not None:  # crop first, then normalise the crop's own loudness
        x = x[int(start * SR):int((start + length) * SR)]
    # Loudness (RMS) normalisation to about -20 dBFS: quiet test clips get the same level as train.
    # RMS, not peak, so one loud click does not decide the scale of the whole clip.
    x = x * (0.1 / (np.sqrt(np.mean(x ** 2)) + 1e-8))
    return np.clip(x, -1.0, 1.0)


def mean_std(h):
    """h: (layers, time, dim) -> (layers, 2*dim)"""
    return torch.cat([h.mean(1), h.std(1)], dim=-1)


class WhisperEmbedder:
    def __init__(self, name, dev, dt):
        self.fe = WhisperFeatureExtractor.from_pretrained(name)
        self.enc = WhisperModel.from_pretrained(name, torch_dtype=dt).get_encoder().to(dev).eval()
        self.dev, self.dt = dev, dt

    @torch.no_grad()
    def __call__(self, x):
        wins = [x[i:i + WIN] for i in range(0, len(x), WIN)]
        wins = [w for w in wins if len(w) >= SR] or [x]  # drop tail windows shorter than 1 s
        feats = self.fe(wins, sampling_rate=SR, return_tensors="pt").input_features.to(self.dev, self.dt)
        hs = self.enc(feats, output_hidden_states=True).hidden_states  # each (n_win, 1500, d)
        h = torch.stack(hs, 1).float()  # (n_win, layers, 1500, d)
        pooled, weights = [], []
        for k, w in enumerate(wins):
            n = max(1, min(1500, int(round(len(w) / SR * 50))))  # 50 encoder frames per second
            pooled.append(mean_std(h[k, :, :n]))
            weights.append(n)
        wts = torch.tensor(weights, dtype=torch.float32, device=h.device)
        return (torch.stack(pooled) * wts[:, None, None]).sum(0) / wts.sum()


class WavLMEmbedder:
    def __init__(self, name, dev, dt):
        self.fe = AutoFeatureExtractor.from_pretrained(name)
        self.m = WavLMModel.from_pretrained(name, torch_dtype=dt).to(dev).eval()
        self.dev, self.dt = dev, dt

    @torch.no_grad()
    def __call__(self, x):
        inp = self.fe(x, sampling_rate=SR, return_tensors="pt").input_values.to(self.dev, self.dt)
        hs = self.m(inp, output_hidden_states=True).hidden_states  # each (1, T, d)
        return mean_std(torch.stack(hs, 0)[:, 0].float())


class W2VBertEmbedder:
    """w2v-BERT 2.0 (Meta, MIT): Conformer encoder on 80-dim filterbanks, takes the whole clip."""

    def __init__(self, name, dev, dt):
        self.fe = AutoFeatureExtractor.from_pretrained(name)
        self.m = Wav2Vec2BertModel.from_pretrained(name, torch_dtype=dt).to(dev).eval()
        self.dev, self.dt = dev, dt

    @torch.no_grad()
    def __call__(self, x):
        inp = self.fe(x, sampling_rate=SR, return_tensors="pt").input_features.to(self.dev, self.dt)
        hs = self.m(inp, output_hidden_states=True).hidden_states  # each (1, T, d)
        return mean_std(torch.stack(hs, 0)[:, 0].float())


def run_crops(tag, embedder, items, out, suffix, layers):
    """items: list of (path, start_s, length_s). Saves (len(items), 2*d) averaged over `layers` (inclusive range)."""
    lo, hi = layers
    t0, res = time.time(), []
    for i, (path, st, ln) in enumerate(items):
        e = embedder(load(path, st, ln))[lo:hi + 1].mean(0)
        res.append(e.cpu().numpy().astype(np.float16))
        if (i + 1) % 200 == 0:
            print(f"[{tag} crops] {i + 1}/{len(items)}  {(time.time() - t0) / (i + 1):.2f} s/crop", flush=True)
    np.save(out / f"emb_{tag}_crops{suffix}.npy", np.stack(res))
    print(f"[{tag} crops] saved {len(res)} in {time.time() - t0:.0f} s", flush=True)


def run(tag, embedder, files, out, suffix=""):
    npy, ckpt = out / f"emb_{tag}{suffix}.npy", out / f"emb_{tag}{suffix}.partial.npy"
    if npy.exists():
        print(f"[{tag}] done already, skipping")
        return
    done = list(np.load(ckpt)) if ckpt.exists() else []
    print(f"[{tag}] {len(files)} files, resuming at {len(done)}")
    t0 = time.time()
    for i in range(len(done), len(files)):
        done.append(embedder(load(files[i])).cpu().numpy().astype(np.float16))
        if (i + 1) % CKPT == 0:
            np.save(ckpt, np.stack(done))
            print(f"[{tag}] {i + 1}/{len(files)}  {(time.time() - t0) / (i + 1 - 0):.2f} s/file", flush=True)
    arr = np.stack(done)
    np.save(npy, arr)
    ckpt.unlink(missing_ok=True)
    print(f"[{tag}] saved {arr.shape} in {time.time() - t0:.0f} s; finite={np.isfinite(arr).all()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder holding train.csv, test.csv, train/, test/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--whisper", default="openai/whisper-large-v3")
    ap.add_argument("--wavlm", default="microsoft/wavlm-large")
    ap.add_argument("--w2vbert", default="none", help="e.g. facebook/w2v-bert-2.0")
    ap.add_argument("--limit", type=int, default=0, help="only the first N train + N test files (smoke test)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--merge", action="store_true", help="join shard outputs into one file per model")
    ap.add_argument("--crops", type=int, default=0, help="crop mode: K random crops per train clip")
    ap.add_argument("--crop_min", type=float, default=40.0)
    ap.add_argument("--crop_max", type=float, default=50.0)
    ap.add_argument("--layer_avg", default="whisper:29-32,wavlm:17-24", help="crop mode: layers to average per model")
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tr, te = pd.read_csv(data / "train.csv"), pd.read_csv(data / "test.csv")
    if a.limit:
        tr, te = tr.head(a.limit), te.head(a.limit)
    order = pd.concat([tr[["filename"]].assign(split="train"), te[["filename"]].assign(split="test")])
    order.to_csv(out / "emb_order.csv", index=False)
    files = [str(data / s / f) for f, s in zip(order.filename, order.split)]
    models = [(t, c, n) for t, c, n in [("whisper", WhisperEmbedder, a.whisper), ("wavlm", WavLMEmbedder, a.wavlm),
                                          ("w2vbert", W2VBertEmbedder, a.w2vbert)]
              if n.lower() != "none"]
    if a.crops:
        rng = np.random.default_rng(0)
        rows = []
        for f in tr.filename:
            dur = sf.info(str(data / "train" / f)).duration
            for k in range(a.crops):
                ln = min(dur, rng.uniform(a.crop_min, a.crop_max))
                rows.append((f, k, round(rng.uniform(0, max(0.0, dur - ln)), 3), round(ln, 3)))
        co = pd.DataFrame(rows, columns=["filename", "k", "start_s", "length_s"])
        co.to_csv(out / "crops_order.csv", index=False)
        lay = {m.split(":")[0]: tuple(int(v) for v in m.split(":")[1].split("-")) for m in a.layer_avg.split(",")}
        if a.merge:
            for tag, _, _ in models:
                arr = np.concatenate([np.load(out / f"emb_{tag}_crops.s{i}.npy") for i in range(a.nshards)])
                assert len(arr) == len(co), (tag, arr.shape)
                np.save(out / f"emb_{tag}_crops.npy", arr)
                [(out / f"emb_{tag}_crops.s{i}.npy").unlink() for i in range(a.nshards)]
                print(f"[{tag} crops] merged {arr.shape}; finite={np.isfinite(arr).all()}")
            return
        block = np.array_split(np.arange(len(co)), a.nshards)[a.shard]
        items = [(str(data / "train" / co.filename[i]), co.start_s[i], co.length_s[i]) for i in block]
        dev, dt = device_dtype()
        print("crop mode | device", dev, "| shard", a.shard, "of", a.nshards, "| crops", len(items), flush=True)
        for tag, cls, name in models:
            emb = cls(name, dev, dt)
            run_crops(tag, emb, items, out, f".s{a.shard}" if a.nshards > 1 else "", lay[tag])
            del emb
            if dev == "cuda":
                torch.cuda.empty_cache()
        return
    if a.merge:
        for tag, _, name in models:
            parts = [out / f"emb_{tag}.s{i}.npy" for i in range(a.nshards)]
            arr = np.concatenate([np.load(p) for p in parts])
            assert len(arr) == len(files), (tag, arr.shape, len(files))
            np.save(out / f"emb_{tag}.npy", arr)
            for p in parts:
                p.unlink()
            json.dump({"model": name}, open(out / f"emb_{tag}.json", "w"))
            print(f"[{tag}] merged {arr.shape}; finite={np.isfinite(arr).all()}")
        return
    block = np.array_split(np.arange(len(files)), a.nshards)[a.shard]
    files = [files[i] for i in block]
    suffix = f".s{a.shard}" if a.nshards > 1 else ""
    dev, dt = device_dtype()
    print("device", dev, dt, "| shard", a.shard, "of", a.nshards, "| files", len(files))
    for tag, cls, name in models:
        emb = cls(name, dev, dt)
        run(tag, emb, files, out, suffix)
        if a.nshards == 1:
            json.dump({"model": name}, open(out / f"emb_{tag}.json", "w"))
        del emb
        if dev == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
