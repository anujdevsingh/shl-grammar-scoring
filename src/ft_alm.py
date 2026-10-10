"""Stage H: fine-tune an audio LLM (Qwen2-Audio-7B-Instruct, Apache-2.0) to predict the grammar score.

Why: zero-shot, the audio LLM judge was weak (Pearson 0.43). Published work found the same model jumps from
Pearson 0.64 zero-shot to 0.95 after fine-tuning on proficiency scores (Interspeech 2025, arXiv 2505.21148).
How (QLoRA regression, explainable in one line: "the LLM listens to the clip, a small head reads its final state"):
- The 7B model is loaded in 4-bit (NF4); only small LoRA adapters (rank 16) on the language model's attention and
  MLP layers are trained, plus a small regression head. The audio encoder stays frozen.
- Input = the rubric prompt + one 30 s window of audio (Qwen2-Audio's limit). Training uses a random 30 s window
  of each clip; evaluation averages the windows of the whole clip.
- The head reads the hidden state of the last prompt token (where the model would start its answer) -> score.
- MSE loss, AdamW, cosine schedule, a FIXED number of epochs (no early stopping on the validation fold).
Output per fold, same format as finetune.py: oof_fold{k}.csv, test_fold{k}.csv, log_fold{k}.csv.

Usage: python ft_alm.py --data DIR --folds folds.csv --out DIR --fold K [--model PATH] [--epochs 3] [--limit N] [--tiny]
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

SR, WIN = 16000, 30 * 16000
PROMPT = ("You are an expert English examiner. Listen to the speaker and rate ONLY their spoken grammar "
          "(sentence structure, syntax, errors, use of complex grammar), from 0 to 5.")


def load_audio(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    assert sr == SR
    return x.mean(axis=1)


def windows(x):
    w = [x[i:i + WIN] for i in range(0, len(x), WIN)]
    return [v for v in w if len(v) >= 5 * SR] or [x]


def build(name, tiny):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
    proc = AutoProcessor.from_pretrained(name)
    if tiny:  # random tiny model with the same architecture: tests the code path on a laptop
        from transformers import Qwen2AudioConfig
        cfg = Qwen2AudioConfig.from_pretrained(name)
        cfg.audio_config.update(dict(d_model=64, encoder_layers=2, encoder_attention_heads=2, encoder_ffn_dim=128))
        cfg.text_config.update(dict(hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                                    num_key_value_heads=2, intermediate_size=128))
        model = Qwen2AudioForConditionalGeneration(cfg).float()
    else:
        from peft import prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.float16)
        model = Qwen2AudioForConditionalGeneration.from_pretrained(name, quantization_config=q, device_map={"": 0},
                                                                   torch_dtype=torch.float16)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return proc, model


class Head(nn.Module):
    def __init__(self, d, init_bias):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 1))
        nn.init.constant_(self.net[-1].bias, init_bias)

    def forward(self, h):
        return self.net(h.float()).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--folds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--model", default="Qwen/Qwen2-Audio-7B-Instruct")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr_lora", type=float, default=1e-4)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tiny", action="store_true")
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    data = Path(a.data)
    tr = pd.read_csv(data / "train.csv").merge(pd.read_csv(a.folds)[["filename", "r0"]], on="filename")
    te = pd.read_csv(data / "test.csv")
    if a.limit:
        tr, te = tr.head(a.limit), te.head(3)
    trn, val = tr[tr.r0 != a.fold].reset_index(drop=True), tr[tr.r0 == a.fold].reset_index(drop=True)
    print(f"fold {a.fold}: train {len(trn)} val {len(val)} test {len(te)}", flush=True)
    audio = {("train", f): load_audio(data / "train" / f) for f in tr.filename}
    audio |= {("test", f): load_audio(data / "test" / f) for f in te.filename}

    proc, model = build(a.model, a.tiny)
    dev = next(p.device for p in model.parameters())
    amp = dev.type == "cuda"
    d = model.config.text_config.hidden_size
    head = Head(d, float(trn.label.mean())).to(dev)
    audio_dtype = next(p.dtype for n, p in model.named_parameters() if "audio_tower" in n and "conv1" in n)
    conv = [{"role": "user", "content": [{"type": "audio", "audio_url": "clip"}, {"type": "text", "text": PROMPT}]}]
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)

    def score(x):
        inp = proc(text=text, audio=[x], sampling_rate=SR, return_tensors="pt").to(dev)
        inp["input_features"] = inp["input_features"].to(audio_dtype)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            hs = model(**inp, output_hidden_states=True).hidden_states[-1]
        return head(hs[:, -1])

    lora_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": lora_params, "lr": a.lr_lora}, {"params": head.parameters(), "lr": a.lr_head}],
                            weight_decay=0.01)
    steps = a.epochs * math.ceil(len(trn) / a.accum)
    warm = max(1, int(0.1 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    @torch.no_grad()
    def predict(split, files):
        model.eval(); head.eval()
        p = [float(np.mean([float(score(w)) for w in windows(audio[(split, f)])])) for f in files]
        model.train(); head.train()
        return np.array(p)

    log, t0 = [], time.time()
    model.train(); head.train()
    for ep in range(a.epochs):
        order = trn.sample(frac=1, random_state=a.seed + ep).reset_index(drop=True)
        losses = []
        for i, r in order.iterrows():
            x = audio[("train", r.filename)]
            if len(x) > WIN:
                s = np.random.randint(0, len(x) - WIN + 1)
                x = x[s:s + WIN]
            loss = (score(x) - torch.tensor([r.label], device=dev)).pow(2).mean() / a.accum
            scaler.scale(loss).backward()
            losses.append(float(loss.detach()) * a.accum)
            if (i + 1) % a.accum == 0 or i + 1 == len(order):
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(lora_params + list(head.parameters()), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
            if (i + 1) % 100 == 0:
                print(f"fold {a.fold} epoch {ep + 1}: {i + 1}/{len(order)} clips, train MSE {np.mean(losses):.4f} "
                      f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        pv = predict("train", val.filename)
        rm = float(np.sqrt(np.mean((pv - val.label.to_numpy()) ** 2)))
        pr = float(np.corrcoef(pv, val.label)[0, 1]) if pv.std() > 0 else 0.0
        log.append(dict(fold=a.fold, epoch=ep + 1, train_mse=np.mean(losses), val_rmse=rm, val_pearson=pr,
                        minutes=(time.time() - t0) / 60))
        print(f"fold {a.fold} epoch {ep + 1}/{a.epochs}: train MSE {np.mean(losses):.4f}  val RMSE {rm:.4f}  "
              f"Pearson {pr:.4f}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
        pd.DataFrame(log).to_csv(out / f"log_fold{a.fold}.csv", index=False)
        pd.DataFrame({"filename": val.filename, "pred": pv}).to_csv(out / f"oof_fold{a.fold}.csv", index=False)
    pd.DataFrame({"filename": te.filename, "pred": predict("test", te.filename)}).to_csv(
        out / f"test_fold{a.fold}.csv", index=False)
    print(f"fold {a.fold}: done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
