"""Stage C5: fine-tune Qwen2.5-7B-Instruct on the transcripts with QLoRA to predict the grammar score.

Why: on new voices the text view is as strong as audio (D20), fine-tuning DeBERTa beat frozen DeBERTa by 0.02 (D21),
and frozen Qwen2.5-7B already scores 0.440 with a plain Ridge. A 7B model knows far more English; fine-tuning should
teach it this rubric.
How (QLoRA regression, the same recipe as the audio LLM in src/ft_alm.py, text only):
- The 7B model is loaded in 4-bit (NF4) and frozen; LoRA adapters (rank 16) on every attention and MLP projection,
  plus a small regression head, are trained.
- Input: a rubric instruction + the transcript. The head reads the mean of the last layer's hidden states over all
  tokens (frozen tests showed mean pooling beats the last token).
- MSE loss, AdamW (LoRA 1e-4, head 1e-3), warm-up then cosine, batch 4 x 2 accumulation, a FIXED 2 epochs.
- Speaker-grouped folds, trained on scores 1-5. Output per fold: oof_fold{k}.csv, test_fold{k}.csv, log_fold{k}.csv.

Usage: python ft_llm_text.py --texts texts.json --data DIR --folds folds.csv --out DIR --fold K [--tiny]
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

PROMPT = ("Below is a transcript of a person speaking English. Rate the grammar of the speaker from 1 to 5 "
          "(sentence structure, syntax, errors, use of complex grammar).\n\nTranscript: ")


def build(name, tiny):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "right"
    if tiny:  # random tiny model with the same architecture: tests the code path on a laptop
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(name)
        cfg.update(dict(hidden_size=64, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                        intermediate_size=128))
        model = AutoModelForCausalLM.from_config(cfg).float()
    else:
        from peft import prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(name, quantization_config=q, device_map={"": 0}, dtype=torch.float16)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return tok, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--folds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr_lora", type=float, default=1e-4)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tiny", action="store_true")
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    txt = {(r["split"], r["filename"]): r["text"] for r in json.load(open(a.texts))}
    tr = pd.read_csv(Path(a.data) / "train.csv").merge(pd.read_csv(a.folds)[["filename", "r0"]], on="filename")
    te = pd.read_csv(Path(a.data) / "test.csv")
    trn = tr[(tr.r0 != a.fold) & (tr.label > 0)].reset_index(drop=True)
    val = tr[tr.r0 == a.fold].reset_index(drop=True)
    if a.limit:
        trn, val, te = trn.head(a.limit), val.head(a.limit), te.head(a.limit)
    print(f"fold {a.fold}: train {len(trn)} val {len(val)} test {len(te)}", flush=True)
    tok, model = build(a.model, a.tiny)
    dev = next(p.device for p in model.parameters())
    amp = dev.type == "cuda"
    d = model.config.hidden_size
    head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 1)).to(dev)
    nn.init.constant_(head[-1].bias, float(trn.label.mean()))

    def score(split, files):
        b = tok([PROMPT + txt[(split, f)] + "\n\nGrammar score (1-5):" for f in files], padding=True, truncation=True,
                max_length=768, return_tensors="pt").to(dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            h = model(**b, output_hidden_states=True).hidden_states[-1]
        m = b["attention_mask"].unsqueeze(-1).float()
        return head(((h.float() * m).sum(1) / m.sum(1))).squeeze(-1)

    lora_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": lora_params, "lr": a.lr_lora}, {"params": head.parameters(), "lr": a.lr_head}],
                            weight_decay=0.01)
    steps = a.epochs * math.ceil(len(trn) / (a.bs * a.accum))
    warm = max(1, int(0.1 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    @torch.no_grad()
    def predict(split, files):
        model.eval(); head.eval()
        p = np.concatenate([score(split, files[i:i + 8]).float().cpu().numpy() for i in range(0, len(files), 8)])
        model.train(); head.train()
        return p

    log, t0 = [], time.time()
    model.train(); head.train()
    for ep in range(a.epochs):
        order = trn.sample(frac=1, random_state=a.seed + ep).reset_index(drop=True)
        losses = []
        for j, i in enumerate(range(0, len(order), a.bs)):
            b = order.iloc[i:i + a.bs]
            y = torch.tensor(b.label.to_numpy(), dtype=torch.float32, device=dev)
            loss = ((score("train", list(b.filename)).float() - y) ** 2).mean() / a.accum
            scaler.scale(loss).backward()
            losses.append(float(loss.detach()) * a.accum)
            if (j + 1) % a.accum == 0 or i + a.bs >= len(order):
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(lora_params + list(head.parameters()), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
        pv = predict("train", list(val.filename))
        nz = val.label.to_numpy() > 0
        rm = float(np.sqrt(np.mean((pv[nz] - val.label.to_numpy()[nz]) ** 2)))
        log.append(dict(fold=a.fold, epoch=ep + 1, train_mse=np.mean(losses), val_rmse_1to5=rm, minutes=(time.time() - t0) / 60))
        print(f"fold {a.fold} epoch {ep + 1}/{a.epochs}: train MSE {np.mean(losses):.4f}  val RMSE (1-5) {rm:.4f}  "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    pd.DataFrame(log).to_csv(out / f"log_fold{a.fold}.csv", index=False)
    pd.DataFrame({"filename": val.filename, "pred": pv}).to_csv(out / f"oof_fold{a.fold}.csv", index=False)
    pd.DataFrame({"filename": te.filename, "pred": predict("test", list(te.filename))}).to_csv(
        out / f"test_fold{a.fold}.csv", index=False)
    print(f"fold {a.fold}: done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
