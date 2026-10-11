"""Stage I: adapter grader on a frozen speech encoder (the Speak & Improve 2025 winning recipe, adapted).

Idea, in one line: keep the big encoder frozen, but instead of averaging its output over time first (what the
Ridge branch does), train a small network that transforms EVERY frame, then average. Short events such as a broken
sentence or a restart can then shape the score before they are averaged away.

Two steps:
  extract  run the frozen encoder once per clip; keep 4 layers; average frames into 80 ms steps; save as float16.
           Whisper-large-v3 (30 s windows, padding dropped) or WavLM-large (whole clip).
  train    for each fold and seed: per-layer adapters (LayerNorm -> Linear -> GELU -> Linear, 128 dims),
           concatenated -> Linear -> one small Transformer layer -> mean + std over valid frames -> MLP -> score.
           About 1M trainable parameters. Training uses random 20 s crops and random frame masking as augmentation,
           trains on scores 1-5 only (the test set has no invalid label-0 clips), and runs a FIXED 15 epochs.
           The seeds are averaged. Output in the same per-fold format as finetune.py.

Usage: python adapter.py extract --data DIR --out DIR --encoder whisper|wavlm [--model NAME] [--limit N]
       python adapter.py train --feats DIR --data DIR --folds folds.csv --out DIR --folds_to_run 0,1 [--seeds 3]
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn

SR, POOL = 16000, 4  # encoders give 50 frames/s; averaging 4 frames -> 12.5 steps/s (80 ms)
LAYERS = {"whisper": [16, 22, 27, 32], "wavlm": [12, 16, 20, 24]}  # middle-to-top layers (early layers were weakest)


def load(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    assert sr == SR
    x = x.mean(axis=1)
    x = x * (0.1 / (np.sqrt(np.mean(x ** 2)) + 1e-8))  # same RMS normalisation as the embeddings
    return np.clip(x, -1, 1)


def pool_frames(h, n):
    """h: (layers, T, d) -> average over blocks of POOL frames, keeping only the first n valid frames."""
    h = h[:, :n]
    t = (h.shape[1] // POOL) * POOL
    return h[:, :t].reshape(h.shape[0], t // POOL, POOL, h.shape[2]).mean(2)


@torch.no_grad()
def extract(a):
    from transformers import AutoFeatureExtractor, WavLMModel, WhisperFeatureExtractor, WhisperModel
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dt = torch.float16 if dev == "cuda" else torch.float32
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tr, te = pd.read_csv(data / "train.csv"), pd.read_csv(data / "test.csv")
    if a.limit:
        tr, te = tr.head(a.limit), te.head(a.limit)
    items = [("train", f) for f in tr.filename] + [("test", f) for f in te.filename]
    layers = LAYERS[a.encoder] if not a.layers else [int(v) for v in a.layers.split(",")]
    if a.encoder == "whisper":
        fe = WhisperFeatureExtractor.from_pretrained(a.model)
        enc = WhisperModel.from_pretrained(a.model, torch_dtype=dt).get_encoder().to(dev).eval()
    else:
        fe = AutoFeatureExtractor.from_pretrained(a.model)
        enc = WavLMModel.from_pretrained(a.model, torch_dtype=dt).to(dev).eval()
    # Frames are streamed to a raw float16 file (about 7 GB for Whisper-large on all clips), never held in memory.
    lens, t0, shape = [], time.time(), None
    fh = open(out / f"frames_{a.encoder}.bin", "wb")
    for i, (split, fn) in enumerate(items, 1):
        x = load(data / split / fn)
        if a.encoder == "whisper":
            wins = [x[j:j + 30 * SR] for j in range(0, len(x), 30 * SR)]
            wins = [w for w in wins if len(w) >= SR] or [x]
            inp = fe(wins, sampling_rate=SR, return_tensors="pt").input_features.to(dev, dt)
            hs = enc(inp, output_hidden_states=True).hidden_states
            h = torch.stack([hs[l] for l in layers], 0).float()  # (L, n_win, 1500, d)
            parts = [pool_frames(h[:, k], max(1, int(len(w) / SR * 50))) for k, w in enumerate(wins)]
            h = torch.cat(parts, 1)
        else:
            inp = fe(x, sampling_rate=SR, return_tensors="pt").input_values.to(dev, dt)
            hs = enc(inp, output_hidden_states=True).hidden_states
            h = pool_frames(torch.stack([hs[l][0] for l in layers], 0).float(), 10 ** 9)
        h = h.cpu().numpy().astype(np.float16).transpose(1, 0, 2)  # (T, L, d)
        fh.write(np.ascontiguousarray(h).tobytes())
        lens.append(h.shape[0]); shape = h.shape[1:]
        if i % 100 == 0:
            print(f"[{a.encoder}] {i}/{len(items)}  {(time.time() - t0) / i:.2f} s/clip", flush=True)
    fh.close()
    pd.DataFrame({"split": [s for s, _ in items], "filename": [f for _, f in items], "len": lens,
                  "L": shape[0], "d": shape[1]}).to_csv(out / f"frames_{a.encoder}_index.csv", index=False)
    print(f"[{a.encoder}] saved ({sum(lens)}, {shape[0]}, {shape[1]}) "
          f"({sum(lens) * shape[0] * shape[1] * 2 / 1e9:.1f} GB) in {time.time() - t0:.0f} s", flush=True)


class AdapterGrader(nn.Module):
    def __init__(self, n_layers, d_in, init_bias, d=128):
        super().__init__()
        self.adapters = nn.ModuleList(nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d), nn.GELU(), nn.Linear(d, d))
                                      for _ in range(n_layers))
        self.proj = nn.Linear(n_layers * d, d)
        self.tf = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(0.1), nn.Linear(d, 1))
        nn.init.constant_(self.head[-1].bias, init_bias)

    def forward(self, x, mask):  # x: (B, T, L, d_in); mask: (B, T) True = real frame
        h = self.proj(torch.cat([ad(x[:, :, i]) for i, ad in enumerate(self.adapters)], -1))
        h = self.tf(h, src_key_padding_mask=~mask)
        m = mask.unsqueeze(-1).float()
        n = m.sum(1).clamp(min=1)
        mu = (h * m).sum(1) / n
        sd = (((h - mu.unsqueeze(1)) ** 2 * m).sum(1) / n).clamp(min=1e-6).sqrt()
        return self.head(torch.cat([mu, sd], -1)).squeeze(-1)


def train(a):
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    feats, out, data = Path(a.feats), Path(a.out), Path(a.data)
    out.mkdir(parents=True, exist_ok=True)
    ix = pd.read_csv(feats / f"frames_{a.encoder}_index.csv")
    flat = np.memmap(feats / f"frames_{a.encoder}.bin", dtype=np.float16, mode="r",
                     shape=(int(ix.len.sum()), int(ix.L[0]), int(ix.d[0])))
    starts = np.r_[0, np.cumsum(ix.len.to_numpy())[:-1]]
    seq = {(s, f): (st, n) for s, f, st, n in zip(ix.split, ix.filename, starts, ix.len)}
    get = lambda key: np.asarray(flat[seq[key][0]:seq[key][0] + seq[key][1]], dtype=np.float32)
    tr = pd.read_csv(data / "train.csv").merge(pd.read_csv(a.folds)[["filename", "r0"]], on="filename")
    tr = tr[[("train", f) in seq for f in tr.filename]]
    te = pd.read_csv(data / "test.csv")
    te = te[[("test", f) in seq for f in te.filename]]
    L, d_in = flat.shape[1], flat.shape[2]
    crop = int(20 * SR / 320 / POOL)  # 20 s in 80 ms steps

    def batch(keys, training):
        xs = [get(k) for k in keys]
        if training:
            out_ = []
            for x in xs:
                if len(x) > crop:
                    s = np.random.randint(0, len(x) - crop + 1)
                    x = x[s:s + crop]
                x = x.copy()
                for _ in range(2):  # mask two random spans of up to 1 s
                    w = np.random.randint(0, 13)
                    s = np.random.randint(0, max(1, len(x) - w))
                    x[s:s + w] = 0
                out_.append(x)
            xs = out_
        T = max(len(x) for x in xs)
        X = np.zeros((len(xs), T, L, d_in), np.float32)
        M = np.zeros((len(xs), T), bool)
        for i, x in enumerate(xs):
            X[i, :len(x)], M[i, :len(x)] = x, True
        return torch.tensor(X, device=dev), torch.tensor(M, device=dev)

    @torch.no_grad()
    def predict(model, keys):
        model.eval()
        p = []
        for i in range(0, len(keys), 16):
            X, M = batch(keys[i:i + 16], False)
            p.append(model(X, M).float().cpu().numpy())
        model.train()
        return np.concatenate(p)

    logs = []
    for k in [int(v) for v in a.folds_to_run.split(",")]:
        trn = tr[(tr.r0 != k) & (tr.label > 0)].reset_index(drop=True)  # learn the 1-5 scale only
        val = tr[tr.r0 == k].reset_index(drop=True)
        vkeys, tkeys = [("train", f) for f in val.filename], [("test", f) for f in te.filename]
        pv_all, pt_all = [], []
        for seed in range(a.seeds):
            torch.manual_seed(seed); np.random.seed(seed)
            model = AdapterGrader(L, d_in, float(trn.label.mean())).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
            steps = a.epochs * math.ceil(len(trn) / a.bs)
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
            t0 = time.time()
            for ep in range(a.epochs):
                order = trn.sample(frac=1, random_state=seed * 100 + ep)
                for i in range(0, len(order), a.bs):
                    rows = order.iloc[i:i + a.bs]
                    X, M = batch([("train", f) for f in rows.filename], True)
                    loss = nn.functional.mse_loss(model(X, M), torch.tensor(rows.label.to_numpy(np.float32), device=dev))
                    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            pv = predict(model, vkeys)
            pv_all.append(pv); pt_all.append(predict(model, tkeys))
            nz = val.label.to_numpy() > 0
            r = float(np.sqrt(np.mean((pv[nz] - val.label.to_numpy()[nz]) ** 2)))
            logs.append(dict(encoder=a.encoder, fold=k, seed=seed, val_rmse_1to5=r, minutes=(time.time() - t0) / 60))
            print(f"[{a.encoder}] fold {k} seed {seed}: val RMSE (1-5 clips) {r:.4f}  ({(time.time() - t0) / 60:.1f} min)",
                  flush=True)
        pv, pt = np.mean(pv_all, 0), np.mean(pt_all, 0)
        nz = val.label.to_numpy() > 0
        print(f"[{a.encoder}] fold {k} seed-average: val RMSE (1-5 clips) "
              f"{np.sqrt(np.mean((pv[nz] - val.label.to_numpy()[nz]) ** 2)):.4f}", flush=True)
        pd.DataFrame({"filename": val.filename, "pred": pv}).to_csv(out / f"oof_fold{k}.csv", index=False)
        pd.DataFrame({"filename": te.filename, "pred": pt}).to_csv(out / f"test_fold{k}.csv", index=False)
        pd.DataFrame(logs).to_csv(out / f"log_{a.encoder}_{a.folds_to_run.replace(',', '')}.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "train"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder", choices=["whisper", "wavlm"], required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--layers", default=None, help="override layer ids, e.g. 2,3,4 for a tiny test model")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--feats", default=None)
    ap.add_argument("--folds", default=None)
    ap.add_argument("--folds_to_run", default="0,1,2,3,4")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    a = ap.parse_args()
    a.model = a.model or {"whisper": "openai/whisper-large-v3", "wavlm": "microsoft/wavlm-large"}[a.encoder]
    extract(a) if a.cmd == "extract" else train(a)


if __name__ == "__main__":
    main()
