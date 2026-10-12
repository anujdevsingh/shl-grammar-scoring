"""Stage C3: hidden states of an open LLM reading the transcript (frozen, no training).

Why: D19 showed the text view generalises to new voices, and frozen DeBERTa already matches the audio models there.
A 4-7B LLM has seen far more English and should "know" what learner grammar looks like.
How: the LLM reads a rubric instruction followed by the transcript. For selected layers we keep
  - the mean of the transcript tokens' hidden states, and
  - the hidden state of the last token (where the model would start its answer).
A Ridge model on top is then chosen and scored with speaker-grouped CV, locally.
Input: --texts JSON list of {split, filename, text}. Output: <out>/<name>.npz with mean (N, L, d), last (N, L, d),
layers, split, filename (float16).

Usage: python llm_text_emb.py --texts texts.json --out DIR --model Qwen/Qwen2.5-7B-Instruct [--4bit]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

PROMPT = ("Below is a transcript of a person speaking English. Rate the grammar of the speaker from 1 to 5 "
          "(sentence structure, syntax, errors, use of complex grammar).\n\nTranscript: ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--4bit", dest="q4", action="store_true")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    recs = json.load(open(a.texts))
    tok = AutoTokenizer.from_pretrained(a.model)
    kw = dict(torch_dtype=torch.float16, device_map={"": 0})
    if a.q4:
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(a.model, **kw).eval()
    n = model.config.num_hidden_layers
    layers = sorted(set(list(range(n // 4, n + 1, max(1, n // 8))) + [n]))
    pre = tok(PROMPT, add_special_tokens=False)["input_ids"]
    M, Lst = [], []
    for i, r in enumerate(recs):
        ids = pre + tok(r["text"] + "\n\nGrammar score (1-5):", add_special_tokens=False)["input_ids"]
        x = torch.tensor([ids[:1500]], device="cuda")
        with torch.no_grad():
            hs = model(input_ids=x, output_hidden_states=True).hidden_states
        M.append(np.stack([hs[l][0, len(pre):].float().mean(0).cpu().numpy() for l in layers]).astype(np.float16))
        Lst.append(np.stack([hs[l][0, -1].float().cpu().numpy() for l in layers]).astype(np.float16))
        if (i + 1) % 100 == 0:
            print(f"{a.model}: {i + 1}/{len(recs)}", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{a.model.split('/')[-1]}.npz", mean=np.stack(M), last=np.stack(Lst), layers=np.array(layers),
             split=np.array([r["split"] for r in recs]), filename=np.array([r["filename"] for r in recs]))
    print("saved", a.model, np.stack(M).shape, flush=True)


if __name__ == "__main__":
    main()
