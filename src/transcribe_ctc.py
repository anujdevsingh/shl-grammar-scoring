"""Stage B2: a second, LITERAL transcript with a CTC speech recogniser and no language model.

Why: Whisper is a sequence-to-sequence model with a strong built-in language model, so it often writes what the
speaker meant ("he went") instead of what they said ("he go"). A CTC model without a language model spells out the
sounds frame by frame, so learner errors, fillers and restarts survive. The fine-tuned text grader (D21) can then
see the errors the raters scored.
How: wav2vec2-large-960h-lv60-self (Meta, Apache-2.0), greedy CTC decoding, 20 s chunks, RMS-normalised audio.
Output: <out>/ctc_transcripts.s<shard>.jsonl with split, filename, text (lower case, no punctuation).

Usage: python transcribe_ctc.py --data DIR --out DIR [--shard I --nshards N]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

SR, CHUNK = 16000, 20 * 16000
MODEL = "facebook/wav2vec2-large-960h-lv60-self"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [("train", f) for f in pd.read_csv(data / "train.csv").filename] + \
           [("test", f) for f in pd.read_csv(data / "test.csv").filename]
    if a.limit:
        rows = rows[:a.limit]
    rows = [rows[i] for i in np.array_split(np.arange(len(rows)), a.nshards)[a.shard]]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = Wav2Vec2Processor.from_pretrained(MODEL)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL, dtype=torch.float32).to(dev).eval()
    t0 = time.time()
    with open(out / f"ctc_transcripts.s{a.shard}.jsonl", "w") as fh:
        for i, (split, f) in enumerate(rows):
            x, sr = sf.read(data / split / f, dtype="float32", always_2d=True)
            x = x.mean(1)
            x = x * (0.1 / (np.sqrt(np.mean(x ** 2)) + 1e-8))
            words = []
            for s in range(0, len(x), CHUNK):
                c = x[s:s + CHUNK]
                if len(c) < SR // 2:
                    continue
                inp = proc(c, sampling_rate=SR, return_tensors="pt").input_values.to(dev)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                    ids = model(inp).logits.argmax(-1)[0]
                words.append(proc.decode(ids).lower())
            fh.write(json.dumps(dict(split=split, filename=f, text=" ".join(w for w in words if w).strip())) + "\n")
            if (i + 1) % 100 == 0:
                print(f"shard {a.shard}: {i + 1}/{len(rows)}  {(time.time() - t0) / (i + 1):.2f} s/clip", flush=True)
    print(f"shard {a.shard}: done {len(rows)} clips in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
