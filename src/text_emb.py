"""Stage C2: frozen text-encoder embeddings of the transcripts.

Why: a transcript does not carry the speaker's voice, so a text model cannot "recognise" a speaker. After D19 (voices
leak across random folds; test voices are all new) the text view matters more. The hand-made features (D6) are only
part of what is in the words, so we also try general text encoders.
How: each transcript (segments joined) goes through a frozen encoder; we average the token states (mean pooling,
padding masked) of every layer, then average the layers in the upper half of the network. Max 512 tokens (the
transcripts are 60-180 words, so nothing is cut).
Output: cache/text_emb/<name>.npy (rows in cache/text_emb/order.csv). Runs locally (Apple MPS or CPU).

Usage: python src/text_emb.py [--models microsoft/deberta-v3-large,FacebookAI/roberta-large]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from cv import ROOT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="microsoft/deberta-v3-large,FacebookAI/roberta-large")
    a = ap.parse_args()
    from transformers import AutoModel, AutoTokenizer
    recs = [json.loads(line) for line in open(ROOT / "cache" / "transcripts.jsonl")]
    texts = [" ".join(s["text"].strip() for s in r["segments"]).strip() or "." for r in recs]
    out = ROOT / "cache" / "text_emb"
    out.mkdir(exist_ok=True)
    pd.DataFrame({"split": [r["split"] for r in recs], "filename": [r["filename"] for r in recs]}).to_csv(
        out / "order.csv", index=False)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    for name in a.models.split(","):
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModel.from_pretrained(name, output_hidden_states=True).to(dev).eval()
        embs = []
        for i in range(0, len(texts), 16):
            b = tok(texts[i:i + 16], padding=True, truncation=True, max_length=512, return_tensors="pt").to(dev)
            with torch.no_grad():
                hs = model(**b).hidden_states  # (layers+1) x (B, T, d)
            mask = b["attention_mask"].unsqueeze(-1).float()
            n = len(hs) - 1
            layers = [(h * mask).sum(1) / mask.sum(1) for h in hs[1 + n // 2:]]  # upper half
            embs.append(torch.stack(layers).mean(0).float().cpu().numpy())
        E = np.concatenate(embs)
        np.save(out / f"{name.split('/')[-1]}.npy", E)
        print(name, E.shape, flush=True)


if __name__ == "__main__":
    main()
