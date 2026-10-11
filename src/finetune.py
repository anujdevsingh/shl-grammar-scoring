"""Stage G: fine-tune WavLM-large to predict the grammar score directly from audio.

Why: frozen embeddings + Ridge plateaued at CV RMSE 0.524. Training the encoder on our own labels lets it
learn which parts of the speech matter for this rubric (the strongest Speak & Improve 2025 system did this).

Model:  WavLM-large (MIT) -> learned weighted sum of all hidden layers -> mean over time -> small MLP -> score.
        The convolutional front-end stays frozen; the transformer layers are trained with a small learning rate,
        the head with a larger one.
Train:  random 20 s crops (data augmentation and fits T4 memory), MSE loss, fp16, AdamW, cosine schedule,
        a FIXED number of epochs (no early stopping on the validation fold, so out-of-fold scores are not tuned on validation data).
Eval:   the whole clip.
CV:     the same fixed folds as every other model (repeat r0). One process per fold; folds are spread over GPUs.
Output per fold: oof_fold{k}.csv (validation predictions), test_fold{k}.csv, log_fold{k}.csv (per-epoch metrics).

Usage: python finetune.py --data DIR --folds folds.csv --out DIR --fold K [--model microsoft/wavlm-large]
                          [--epochs 8] [--limit N]  (limit: smoke test on N train clips)
"""
import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
from transformers import WavLMModel

SR = 16000


def load(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    assert sr == SR
    x = x.mean(axis=1)
    x = x * (0.1 / (np.sqrt(np.mean(x ** 2)) + 1e-8))  # same RMS normalisation as the embeddings
    x = np.clip(x, -1, 1)
    return (x - x.mean()) / (x.std() + 1e-7)  # zero mean, unit variance, as WavLM expects


class Scorer(nn.Module):
    def __init__(self, name, init_bias):
        super().__init__()
        self.enc = WavLMModel.from_pretrained(name)
        self.enc.feature_extractor._freeze_parameters()
        self.enc.config.layerdrop = 0.0
        n = self.enc.config.num_hidden_layers + 1
        self.layer_w = nn.Parameter(torch.zeros(n))
        d = self.enc.config.hidden_size
        self.head = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 1))
        nn.init.constant_(self.head[-1].bias, init_bias)

    def forward(self, x):
        hs = self.enc(x, output_hidden_states=True).hidden_states
        h = (torch.stack(hs, 0) * torch.softmax(self.layer_w, 0)[:, None, None, None]).sum(0)
        return self.head(h.mean(1)).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--folds", required=True, help="csv with filename and fold id column r0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--model", default="microsoft/wavlm-large")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--crop", type=float, default=20.0)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr_enc", type=float, default=2e-5)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    amp = dev == "cuda"
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    tr = pd.read_csv(data / "train.csv").merge(pd.read_csv(a.folds)[["filename", "r0"]], on="filename")
    te = pd.read_csv(data / "test.csv")
    if a.limit:
        tr, te = tr.head(a.limit), te.head(4)
    trn, val = tr[tr.r0 != a.fold].reset_index(drop=True), tr[tr.r0 == a.fold].reset_index(drop=True)
    print(f"fold {a.fold}: train {len(trn)} val {len(val)} test {len(te)} | device {dev}", flush=True)
    audio = {("train", f): load(data / "train" / f) for f in tr.filename}
    audio |= {("test", f): load(data / "test" / f) for f in te.filename}

    model = Scorer(a.model, float(trn.label.mean())).to(dev)
    model.enc.gradient_checkpointing_enable()
    enc_params = [p for n, p in model.named_parameters() if n.startswith("enc.") and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": a.lr_enc}, {"params": head_params, "lr": a.lr_head}],
                            weight_decay=0.01)
    steps = a.epochs * math.ceil(len(trn) / (a.bs * a.accum))
    warm = max(1, int(0.1 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    crop = int(a.crop * SR)

    def batch_train(rows):
        xs = []
        for f in rows.filename:
            x = audio[("train", f)]
            if len(x) > crop:
                s = np.random.randint(0, len(x) - crop + 1)
                x = x[s:s + crop]
            else:
                x = np.pad(x, (0, crop - len(x)))
            xs.append(x)
        return torch.tensor(np.stack(xs), device=dev), torch.tensor(rows.label.to_numpy(np.float32), device=dev)

    @torch.no_grad()
    def predict(split, files):
        model.eval()
        p = []
        for f in files:
            x = torch.tensor(audio[(split, f)][None], device=dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                p.append(float(model(x).float()))
        model.train()
        return np.array(p)

    log, t0 = [], time.time()
    model.train()
    for ep in range(a.epochs):
        order = trn.sample(frac=1, random_state=a.seed + ep).reset_index(drop=True)
        losses = []
        for i in range(0, len(order), a.bs):
            x, y = batch_train(order.iloc[i:i + a.bs])
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss = nn.functional.mse_loss(model(x).float(), y) / a.accum
            scaler.scale(loss).backward()
            losses.append(float(loss.detach()) * a.accum)
            if (i // a.bs + 1) % a.accum == 0 or i + a.bs >= len(order):
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
        pv = predict("train", val.filename)
        rm = float(np.sqrt(np.mean((pv - val.label.to_numpy()) ** 2)))
        pr = float(np.corrcoef(pv, val.label)[0, 1]) if pv.std() > 0 else 0.0
        log.append(dict(fold=a.fold, epoch=ep + 1, train_mse=np.mean(losses), val_rmse=rm, val_pearson=pr,
                        minutes=(time.time() - t0) / 60))
        print(f"fold {a.fold} epoch {ep + 1}/{a.epochs}: train MSE {np.mean(losses):.4f}  "
              f"val RMSE {rm:.4f}  Pearson {pr:.4f}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
        pd.DataFrame(log).to_csv(out / f"log_fold{a.fold}.csv", index=False)

    # final-epoch predictions (fixed epochs: the validation fold never chose anything)
    pd.DataFrame({"filename": val.filename, "pred": pv}).to_csv(out / f"oof_fold{a.fold}.csv", index=False)
    pd.DataFrame({"filename": te.filename, "pred": predict("test", te.filename)}).to_csv(
        out / f"test_fold{a.fold}.csv", index=False)
    print(f"fold {a.fold}: done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
