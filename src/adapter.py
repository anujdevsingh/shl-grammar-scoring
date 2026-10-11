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
           Short-clip options (round 2): --crop_min/--crop_max variable crop lengths, --dur_feat (clip length as an
           input to the head), --tta (average of full-clip and 20 s window scores).
           Semi-supervised option (round 3): --pseudo_dir adds leakage-free pseudo-labelled clips (src/pseudo.py)
           with loss weight --pseudo_w.
           Robustness option (round 4): --cmvn mean|meanvar normalises each clip's frames over time, removing constant
           recording-condition offsets (classic cepstral mean/variance normalisation applied to encoder features).

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
    def __init__(self, n_layers, d_in, init_bias, d=128, dur_feat=False):
        super().__init__()
        self.dur_feat = dur_feat
        self.adapters = nn.ModuleList(nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d), nn.GELU(), nn.Linear(d, d))
                                      for _ in range(n_layers))
        self.proj = nn.Linear(n_layers * d, d)
        self.tf = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(2 * d + int(dur_feat), d), nn.GELU(), nn.Dropout(0.1), nn.Linear(d, 1))
        nn.init.constant_(self.head[-1].bias, init_bias)

    def forward(self, x, mask, dur=None):  # x: (B, T, L, d_in); mask: (B, T) True = real frame; dur: (B,) seconds
        h = self.proj(torch.cat([ad(x[:, :, i]) for i, ad in enumerate(self.adapters)], -1))
        h = self.tf(h, src_key_padding_mask=~mask)
        m = mask.unsqueeze(-1).float()
        n = m.sum(1).clamp(min=1)
        mu = (h * m).sum(1) / n
        sd = (((h - mu.unsqueeze(1)) ** 2 * m).sum(1) / n).clamp(min=1e-6).sqrt()
        z = torch.cat([mu, sd] + ([torch.log(dur).unsqueeze(-1) - 3.9] if self.dur_feat else []), -1)  # log(50 s) ~ 3.9
        return self.head(z).squeeze(-1)


def train(a):
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    feats, out, data = Path(a.feats), Path(a.out), Path(a.data)
    out.mkdir(parents=True, exist_ok=True)
    ix = pd.read_csv(feats / f"frames_{a.encoder}_index.csv")
    flat = np.memmap(feats / f"frames_{a.encoder}.bin", dtype=np.float16, mode="r",
                     shape=(int(ix.len.sum()), int(ix.L[0]), int(ix.d[0])))
    starts = np.r_[0, np.cumsum(ix.len.to_numpy())[:-1]]
    seq = {(s, f): (st, n) for s, f, st, n in zip(ix.split, ix.filename, starts, ix.len)}
    def get(key):
        x = np.asarray(flat[seq[key][0]:seq[key][0] + seq[key][1]], dtype=np.float32)  # (T, L, d)
        if a.cmvn != "none":  # per-clip normalisation: removes a constant device/room "colour" from each recording
            x = x - x.mean(0, keepdims=True)
            if a.cmvn == "meanvar":
                x = x / (x.std(0, keepdims=True) + 1e-5)
        return x
    tr = pd.read_csv(data / "train.csv").merge(pd.read_csv(a.folds)[["filename", "r0"]], on="filename")
    tr = tr[[("train", f) in seq for f in tr.filename]]
    te = pd.read_csv(data / "test.csv")
    te = te[[("test", f) in seq for f in te.filename]]
    L, d_in = flat.shape[1], flat.shape[2]
    step_s = 320 * POOL / SR  # seconds per cached step (0.08 s)
    crop_lo, crop_hi = int(a.crop_min / step_s), int(a.crop_max / step_s)
    clip_dur = {key: n * step_s for key, (_, n) in seq.items()}  # true length of each recording, in seconds

    def batch(keys, training):
        xs = [get(k) for k in keys]
        if training:
            out_ = []
            for x in xs:
                crop = np.random.randint(crop_lo, crop_hi + 1)  # variable-length crops (8-40 s by default)
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

    def dur_t(keys):
        return torch.tensor([clip_dur[k] for k in keys], dtype=torch.float32, device=dev)

    @torch.no_grad()
    def predict(model, keys):
        """Full clip; with --tta also the mean over 20 s windows (hop 10 s), averaged with the full-clip score."""
        model.eval()
        p = []
        for i in range(0, len(keys), 16):
            ks = keys[i:i + 16]
            X, M = batch(ks, False)
            p.append(model(X, M, dur_t(ks)).float().cpu().numpy())
        p = np.concatenate(p)
        if a.tta:
            w, hop = int(20 / step_s), int(10 / step_s)
            tta = []
            for k in keys:
                x = get(k)
                wins = [x[s:s + w] for s in range(0, max(1, len(x) - w + 1), hop)] or [x]
                T = max(len(v) for v in wins)
                X = np.zeros((len(wins), T, L, d_in), np.float32); M = np.zeros((len(wins), T), bool)
                for j, v in enumerate(wins):
                    X[j, :len(v)], M[j, :len(v)] = v, True
                d = torch.full((len(wins),), clip_dur[k], device=dev)
                tta.append(float(model(torch.tensor(X, device=dev), torch.tensor(M, device=dev), d).mean()))
            p = (p + np.array(tta)) / 2
        model.train()
        return p

    logs = []
    for k in [int(v) for v in a.folds_to_run.split(",")]:
        trn = tr[(tr.r0 != k) & (tr.label > 0)].assign(split="train", w=1.0).reset_index(drop=True)  # 1-5 scale only
        if a.pseudo_dir:  # semi-supervised: add pseudo-labelled held-out + test clips (no fold-k labels used)
            ps = pd.read_csv(Path(a.pseudo_dir) / f"pseudo_fold{k}.csv").assign(w=a.pseudo_w)
            ps = ps[[(s_, f_) in seq for s_, f_ in zip(ps.split, ps.filename)]]
            trn = pd.concat([trn[["split", "filename", "label", "w"]], ps], ignore_index=True)
            print(f"[{a.encoder}] fold {k}: {int((trn.w == 1).sum())} labelled + {len(ps)} pseudo-labelled clips "
                  f"(weight {a.pseudo_w})", flush=True)
        val = tr[tr.r0 == k].reset_index(drop=True)
        vkeys, tkeys = [("train", f) for f in val.filename], [("test", f) for f in te.filename]
        pv_all, pt_all = [], []
        for seed in range(a.seeds):
            torch.manual_seed(seed); np.random.seed(seed)
            model = AdapterGrader(L, d_in, float(trn.label.mean()), dur_feat=a.dur_feat).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
            steps = a.epochs * math.ceil(len(trn) / a.bs)
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
            t0 = time.time()
            for ep in range(a.epochs):
                order = trn.sample(frac=1, random_state=seed * 100 + ep)
                for i in range(0, len(order), a.bs):
                    rows = order.iloc[i:i + a.bs]
                    ks = list(zip(rows.split, rows.filename))
                    X, M = batch(ks, True)
                    err = (model(X, M, dur_t(ks)) - torch.tensor(rows.label.to_numpy(np.float32), device=dev)) ** 2
                    wt = torch.tensor(rows.w.to_numpy(np.float32), device=dev)
                    loss = (wt * err).sum() / wt.sum()
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
    ap.add_argument("--crop_min", type=float, default=20.0, help="training crop length range in seconds")
    ap.add_argument("--crop_max", type=float, default=20.0)
    ap.add_argument("--dur_feat", action="store_true", help="give the clip's true length to the head")
    ap.add_argument("--tta", action="store_true", help="average full-clip and 20 s window scores at prediction")
    ap.add_argument("--pseudo_dir", default=None, help="folder with pseudo_fold{k}.csv (split, filename, label)")
    ap.add_argument("--pseudo_w", type=float, default=0.5, help="loss weight of pseudo-labelled clips")
    ap.add_argument("--cmvn", choices=["none", "mean", "meanvar"], default="none",
                    help="per-clip normalisation of the frames over time (round 4: robustness to recording conditions)")
    a = ap.parse_args()
    a.model = a.model or {"whisper": "openai/whisper-large-v3", "wavlm": "microsoft/wavlm-large"}[a.encoder]
    extract(a) if a.cmd == "extract" else train(a)


if __name__ == "__main__":
    main()
