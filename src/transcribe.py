"""Stage B: transcripts with word timestamps (faster-whisper).

Settings and why:
- beam_size=5, temperature=0: the most likely transcript, repeatable.
- condition_on_previous_text=False: stops Whisper from repeating itself in loops.
- word_timestamps=True: needed for pause and speech-rate features.
- initial_prompt full of fillers and restarts: nudges Whisper to write what was said
  ("um", "I, I went") instead of a cleaned-up version, because the errors are what we measure.
- vad_filter=False: pauses are signal; we do not cut them away.
Output: one JSON line per clip (segments with confidence, words with times and probabilities).
Resumable: clips already in the output file (or listed in --skip) are skipped. Multi-GPU: --shard i --nshards N.
Robust: faster-whisper's word alignment can crash on a segment with no alignable words. A failing clip is retried
without the prompt, then without word timestamps (word times are then spread evenly inside each segment).
The field "mode" records which attempt worked, so these clips can be checked.

Usage: python transcribe.py --data DIR --out DIR [--model large-v3] [--device cuda] [--limit N] [--shard I --nshards N]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

PROMPT = ("Umm, let me think like, hmm... Okay, here's what I'm, like, thinking. "
          "So, uh, I, I went to the, the market and, um, I buyed some... I mean, I bought some things.")


def transcribe_one(model, audio, kw):
    segs, info = model.transcribe(audio, language="en", beam_size=5, temperature=0.0,
                                  condition_on_previous_text=False, vad_filter=False, **kw)
    segs = list(segs)
    words = []
    for sg in segs:
        if sg.words:
            words += [dict(w=w.word, s=w.start, e=w.end, p=w.probability) for w in sg.words]
        elif not kw.get("word_timestamps") and sg.text.strip():
            ws = sg.text.split()  # no alignment: spread the words evenly over the segment
            step = (sg.end - sg.start) / len(ws)
            words += [dict(w=" " + w, s=sg.start + i * step, e=sg.start + (i + 1) * step, p=float(np.exp(sg.avg_logprob)))
                      for i, w in enumerate(ws)]
    return dict(duration=info.duration,
                segments=[dict(start=sg.start, end=sg.end, text=sg.text, avg_logprob=sg.avg_logprob,
                               no_speech_prob=sg.no_speech_prob, compression_ratio=sg.compression_ratio) for sg in segs],
                words=words)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compute", default=None, help="default: float16 on cuda, int8 on cpu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--skip", default=None, help="text file of split/filename lines already done elsewhere")
    a = ap.parse_args()
    from faster_whisper import WhisperModel

    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tr, te = pd.read_csv(data / "train.csv"), pd.read_csv(data / "test.csv")
    if a.limit:
        tr, te = tr.head(a.limit), te.head(a.limit)
    items = [("train", f) for f in tr.filename] + [("test", f) for f in te.filename]
    if a.skip:
        skip = {tuple(l.strip().split("/")) for l in open(a.skip) if l.strip()}
        items = [it for it in items if it not in skip]
    items = [items[i] for i in np.array_split(np.arange(len(items)), a.nshards)[a.shard]]
    path = out / (f"transcripts.s{a.shard}.jsonl" if a.nshards > 1 else "transcripts.jsonl")
    done = set()
    if path.exists():
        done = {(r["split"], r["filename"]) for r in map(json.loads, path.open())}
    todo = [it for it in items if it not in done]
    print(f"shard {a.shard}/{a.nshards}: {len(items)} clips, {len(todo)} to do", flush=True)

    compute = a.compute or ("float16" if a.device == "cuda" else "int8")
    model = WhisperModel(a.model, device=a.device, compute_type=compute)
    t0 = time.time()
    with path.open("a") as fh:
        for k, (split, fn) in enumerate(todo, 1):
            # Load the audio ourselves (16 kHz mono float32) instead of letting faster-whisper decode
            # the file: its decoder depends on the PyAV version, which differs between machines.
            x, sr = sf.read(str(data / split / fn), dtype="float32", always_2d=True)
            assert sr == 16000, (fn, sr)
            rec = None
            for mode, kw in [("full", dict(initial_prompt=PROMPT, word_timestamps=True)),
                             ("no_prompt", dict(word_timestamps=True)),
                             ("no_word_times", dict(initial_prompt=PROMPT, word_timestamps=False))]:
                try:
                    rec = transcribe_one(model, x.mean(axis=1), kw)
                    break
                except Exception as err:  # noqa: BLE001 - log and try the next, simpler setting
                    print(f"shard {a.shard}: {fn} failed with {mode}: {type(err).__name__}: {err}", flush=True)
            if rec is None:
                rec = dict(duration=len(x) / sr, segments=[], words=[])
                mode = "failed"
            rec = dict(split=split, filename=fn, mode=mode) | rec
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if k % 25 == 0:
                print(f"shard {a.shard}: {k}/{len(todo)}  {(time.time() - t0) / k:.2f} s/clip", flush=True)
    print(f"shard {a.shard}: finished in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
