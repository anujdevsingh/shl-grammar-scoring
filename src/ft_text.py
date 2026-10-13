"""Stage C4: fine-tune DeBERTa-v3-large on the transcripts to predict the grammar score (speaker-grouped folds).

Why: on speaker-grouped CV the frozen text encoders already match the audio models (D20), and text cannot memorise a
voice. Fine-tuning lets the encoder learn what in the words separates a 3 from a 4 (the Speak & Improve baseline is a
fine-tuned BERT text grader).
How:
- Input: the faster-whisper transcript (segments joined), max 512 tokens (no transcript is cut).
- Model: DeBERTa-v3-large (MIT) + mean pooling over tokens + a linear head; MSE loss.
- AdamW, lr 1e-5 (encoder) / 1e-3 (head), linear warm-up 10% then linear decay, batch 8, a FIXED 4 epochs
  (no early stopping on the validation fold), fp16, several seeds averaged.
- Trained on scores 1-5; scored on the held-out fold of cache/folds_grouped_r0.csv (no voice in both parts).
Output per fold: oof_fold{k}.csv, test_fold{k}.csv (filename, pred), log_fold{k}.csv.

Usage: python ft_text.py --texts texts.json --data DIR --folds folds.csv --out DIR --fold K [--seeds 3]
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class Grader(nn.Module):
    def __init__(self, name, init_bias):
        super().__init__()
        from transformers import AutoModel
        self.enc = AutoModel.from_pretrained(name, torch_dtype=torch.float32)  # fp32 master weights; fp16 only inside autocast
        self.head = nn.Linear(self.enc.config.hidden_size, 1)
        nn.init.constant_(self.head.bias, init_bias)

    def forward(self, ids, mask):
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).float()
        return self.head((h.float() * m).sum(1) / m.sum(1)).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--folds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--model", default="microsoft/deberta-v3-large")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N train / val / test rows")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    txt = {(r["split"], r["filename"]): r["text"] for r in json.load(open(a.texts))}
    tr = pd.read_csv(Path(a.data) / "train.csv").merge(pd.read_csv(a.folds)[["filename", "r0"]], on="filename")
    te = pd.read_csv(Path(a.data) / "test.csv")
    trn = tr[(tr.r0 != a.fold) & (tr.label > 0)].reset_index(drop=True)
    val = tr[tr.r0 == a.fold].reset_index(drop=True)
    if a.limit:
        trn, val, te = trn.head(a.limit), val.head(a.limit), te.head(a.limit)
    tok = AutoTokenizer.from_pretrained(a.model)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp = dev == "cuda"

    def enc(split, files):
        b = tok([txt[(split, f)] for f in files], padding=True, truncation=True, max_length=512, return_tensors="pt")
        return b["input_ids"], b["attention_mask"]

    @torch.no_grad()
    def predict(model, split, files):
        model.eval()
        ps = []
        for i in range(0, len(files), 16):
            ids, m = enc(split, files[i:i + 16])
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                ps.append(model(ids.to(dev), m.to(dev)).float().cpu().numpy())
        model.train()
        return np.concatenate(ps)

    pv, pt, log, t0 = [], [], [], time.time()
    for seed in range(a.seeds):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        model = Grader(a.model, float(trn.label.mean())).to(dev)
        opt = torch.optim.AdamW([{"params": model.enc.parameters(), "lr": a.lr},
                                 {"params": model.head.parameters(), "lr": a.lr_head}], weight_decay=0.01)
        steps = a.epochs * math.ceil(len(trn) / a.bs)
        warm = max(1, int(0.1 * steps))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / (steps - warm))))
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        model.train()
        for ep in range(a.epochs):
            order = np.random.permutation(len(trn))
            losses = []
            for i in range(0, len(order), a.bs):
                b = trn.iloc[order[i:i + a.bs]]
                ids, m = enc("train", list(b.filename))
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    p = model(ids.to(dev), m.to(dev))
                loss = ((p.float() - torch.tensor(b.label.to_numpy(), dtype=torch.float32, device=dev)) ** 2).mean()
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sched.step()
                losses.append(float(loss))
            v = predict(model, "train", list(val.filename))
            nz = val.label.to_numpy() > 0
            rm = float(np.sqrt(np.mean((v[nz] - val.label.to_numpy()[nz]) ** 2)))
            log.append(dict(fold=a.fold, seed=seed, epoch=ep + 1, train_mse=np.mean(losses), val_rmse_1to5=rm,
                            minutes=(time.time() - t0) / 60))
            print(f"fold {a.fold} seed {seed} epoch {ep + 1}: train MSE {np.mean(losses):.4f}  val RMSE (1-5) {rm:.4f}  "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        pv.append(v)
        pt.append(predict(model, "test", list(te.filename)))
        del model, opt
        torch.cuda.empty_cache()
    pd.DataFrame(log).to_csv(out / f"log_fold{a.fold}.csv", index=False)
    pd.DataFrame({"filename": val.filename, "pred": np.mean(pv, 0)}).to_csv(out / f"oof_fold{a.fold}.csv", index=False)
    pd.DataFrame({"filename": te.filename, "pred": np.mean(pt, 0)}).to_csv(out / f"test_fold{a.fold}.csv", index=False)
    print(f"fold {a.fold}: done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
