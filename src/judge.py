"""Stage F: audio-LLM grammar judge (Qwen2-Audio-7B-Instruct, Apache-2.0), run locally on Kaggle GPUs.

The model *listens* to the clip and is asked to rate the speaker's grammar with the task's own rubric.
Because it hears the audio, it is not limited by Whisper cleaning up the transcript.
No text is generated: we read the probabilities the model gives to the answers "1".."5" for the next token,
and keep the probability-weighted (expected) score. That gives a smooth score instead of one digit.
- Qwen2-Audio's audio encoder takes up to 30 s, so each clip is cut into 30 s windows and the windows are averaged.
- Two prompt wordings (full rubric, short instruction) are scored separately; both become features.
- Zero-shot: the model never sees our labels, so its scores can be used in CV without leakage.
- Loaded in 4-bit (NF4) so one copy fits on each T4; one worker per GPU.
Output: judge.s{shard}.csv with filename, split, and per prompt: expected score + p1..p5.

Usage: python judge.py --data DIR --out DIR [--model Qwen/Qwen2-Audio-7B-Instruct] [--limit N] [--shard I --nshards N]
       [--tiny]  (smoke test with a tiny random model, same code path)
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

SR = 16000
WIN = 30 * SR
RUBRIC = """Grammar score rubric:
1 = Struggles with sentence structure and syntax; limited control of simple structures and memorized patterns.
2 = Limited understanding of structure and syntax; simple structures with consistent basic mistakes; may leave sentences incomplete.
3 = Decent grasp of sentence structure but makes grammatical errors, or decent grammar but errors in syntax and structure.
4 = Strong understanding of structure and syntax; consistently good control; occasional minor errors that do not cause misunderstanding.
5 = High grammatical accuracy and control of complex grammar; seldom makes noticeable mistakes; handles complex structures."""
PROMPTS = {
    "rubric": "You are an expert English examiner. Listen to the speaker and rate ONLY their spoken grammar "
              "(not accent, pronunciation or ideas).\n" + RUBRIC + "\nAnswer with a single digit from 1 to 5.",
    "short": "Rate the grammatical accuracy and complexity of this person's spoken English on a scale from "
             "1 (very poor) to 5 (excellent). Answer with a single digit from 1 to 5.",
}


def load_model(name, tiny):
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
    proc = AutoProcessor.from_pretrained(name)
    if tiny:  # same architecture, random tiny weights: tests the code path without a 7B download
        from transformers import Qwen2AudioConfig
        cfg = Qwen2AudioConfig.from_pretrained(name)
        cfg.audio_config.update(dict(d_model=64, encoder_layers=2, encoder_attention_heads=2, encoder_ffn_dim=128))
        cfg.text_config.update(dict(hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                                    num_key_value_heads=2, intermediate_size=128))
        return proc, Qwen2AudioForConditionalGeneration(cfg).float().eval()
    from transformers import BitsAndBytesConfig
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    m = Qwen2AudioForConditionalGeneration.from_pretrained(name, quantization_config=q, device_map={"": 0},
                                                           torch_dtype=torch.float16)
    return proc, m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2-Audio-7B-Instruct")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--tiny", action="store_true")
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tr, te = pd.read_csv(data / "train.csv"), pd.read_csv(data / "test.csv")
    if a.limit:
        tr, te = tr.head(a.limit), te.head(a.limit)
    items = [("train", f) for f in tr.filename] + [("test", f) for f in te.filename]
    items = [items[i] for i in np.array_split(np.arange(len(items)), a.nshards)[a.shard]]
    path = out / f"judge.s{a.shard}.csv"
    done = set(pd.read_csv(path).filename) if path.exists() else set()
    todo = [it for it in items if it[1] not in done]
    print(f"shard {a.shard}: {len(items)} clips, {len(todo)} to do", flush=True)

    proc, model = load_model(a.model, a.tiny)
    dev = next(model.parameters()).device
    # dtype of the audio encoder's first conv (its attribute path differs between transformers versions)
    audio_dtype = next(p.dtype for n, p in model.named_parameters() if "audio_tower" in n and "conv1" in n)
    digit_ids = [proc.tokenizer.convert_tokens_to_ids(str(d)) for d in range(1, 6)]
    assert all(i is not None and i >= 0 for i in digit_ids), digit_ids
    texts = {}
    for k, instr in PROMPTS.items():
        conv = [{"role": "user", "content": [{"type": "audio", "audio_url": "clip"}, {"type": "text", "text": instr}]}]
        texts[k] = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)

    t0, rows = time.time(), []
    for n, (split, fn) in enumerate(todo, 1):
        x, sr = sf.read(str(data / split / fn), dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        wins = [x[i:i + WIN] for i in range(0, len(x), WIN)]
        wins = [w for w in wins if len(w) >= 5 * SR] or [x]  # drop tails shorter than 5 s
        row = dict(split=split, filename=fn)
        for k, text in texts.items():
            probs = []
            for w in wins:
                inp = proc(text=text, audio=[w], sampling_rate=SR, return_tensors="pt").to(dev)
                inp["input_features"] = inp["input_features"].to(audio_dtype)
                with torch.no_grad():
                    logits = model(**inp).logits[0, -1].float()
                probs.append(torch.softmax(logits[digit_ids], 0).cpu().numpy())
            p = np.mean(probs, 0)
            row[f"judge_{k}"] = float((p * np.arange(1, 6)).sum())
            row |= {f"judge_{k}_p{d}": float(p[d - 1]) for d in range(1, 6)}
        rows.append(row)
        if n % 25 == 0 or n == len(todo):
            old = pd.read_csv(path) if path.exists() else None
            pd.concat([old, pd.DataFrame(rows)]).to_csv(path, index=False)
            rows = []
            print(f"shard {a.shard}: {n}/{len(todo)}  {(time.time() - t0) / n:.2f} s/clip", flush=True)
    print(f"shard {a.shard}: finished in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
