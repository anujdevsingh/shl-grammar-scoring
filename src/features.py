"""Stage C + D: fluency and grammar features from the transcripts.

C. Fluency (from word timestamps). Everything is per second or per word, because test clips are shorter.
D. Grammar (from the text):
   - GPT-2 mean token log-probability: ungrammatical word sequences are less likely under a language model.
   - spaCy syntax: sentence length, dependency depth, subordinate clauses, sentences without a verb, POS mix,
     simple agreement checks (he/she/it + base-form verb; a/an before vowel sound).
   - LanguageTool grammar-error rate per 100 words, if Java is available (optional).
Output: cache/feat_text.parquet, one row per clip (split, filename, features).

Usage: python src/features.py [--tr cache/transcripts.jsonl] [--no-lt]
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FILLERS = {"um", "umm", "uh", "uhh", "er", "erm", "ah", "hmm", "hm", "mm", "eh"}
PAUSE, LONG = 0.25, 1.0  # seconds
SUBORD = {"mark", "advcl", "ccomp", "xcomp", "relcl", "acl", "csubj"}


def norm(w):
    return re.sub(r"[^a-z']", "", w.lower())


def fluency(r):
    W = r["words"]
    dur = max(r["duration"], 1e-6)
    toks = [norm(w["w"]) for w in W]
    f = dict(duration=dur, n_words=len(W), words_per_sec=len(W) / dur)
    if len(W) < 2:
        return f | dict(no_words=1.0)
    s, e = np.array([w["s"] for w in W]), np.array([w["e"] for w in W])
    p = np.array([w["p"] for w in W])
    gaps = s[1:] - e[:-1]
    speak = np.clip(e - s, 0, None).sum()
    pauses = gaps[gaps > PAUSE]
    runs = np.diff(np.flatnonzero(np.r_[True, gaps > PAUSE, True]))
    segs = r["segments"]
    f |= dict(
        no_words=0.0,
        articulation_rate=len(W) / max(speak, 1e-6),
        phonation_ratio=speak / dur,
        pauses_per_min=60 * len(pauses) / dur,
        long_pauses_per_min=60 * (gaps > LONG).sum() / dur,
        mean_pause=pauses.mean() if len(pauses) else 0.0,
        max_pause=gaps.max(),
        lead_silence=s[0] / dur,
        trail_silence=max(0.0, dur - e[-1]) / dur,
        mean_run_words=runs.mean(),
        filler_rate=sum(t in FILLERS for t in toks) / len(W),
        repeat_rate=sum(a == b and a != "" for a, b in zip(toks[:-1], toks[1:])) / len(W),
        word_prob_mean=p.mean(),
        word_prob_low=(p < 0.5).mean(),
        seg_logprob=np.mean([x["avg_logprob"] for x in segs]),
        seg_nospeech=np.mean([x["no_speech_prob"] for x in segs]),
        seg_compression_max=max(x["compression_ratio"] for x in segs),
        mattr=mattr([t for t in toks if t and t not in FILLERS]),
    )
    return f


def mattr(toks, w=25):
    """Moving-average type-token ratio: vocabulary variety that does not depend on text length."""
    if len(toks) < w:
        return len(set(toks)) / max(len(toks), 1)
    return float(np.mean([len(set(toks[i:i + w])) / w for i in range(len(toks) - w + 1)]))


def clean_text(r):
    """Transcript without fillers, used for the grammar features."""
    words = [w["w"].strip() for w in r["words"] if norm(w["w"]) not in FILLERS]
    return re.sub(r"\s+", " ", " ".join(words)).strip()


def gpt2_scores(texts):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained("gpt2-medium")
    lm = AutoModelForCausalLM.from_pretrained("gpt2-medium").to(dev).eval()
    out = []
    for t in texts:
        ids = tok(tok.bos_token + t, return_tensors="pt").input_ids[:, :1024].to(dev)
        if ids.shape[1] < 3:
            out.append(dict(lm_logprob=np.nan, lm_logprob_p10=np.nan))
            continue
        with torch.no_grad():
            lp = torch.log_softmax(lm(ids).logits[0, :-1].float(), -1).gather(1, ids[0, 1:, None])[:, 0].cpu().numpy()
        out.append(dict(lm_logprob=lp.mean(), lm_logprob_p10=np.percentile(lp, 10)))
    return out


def syntax(nlp, texts):
    out = []
    for doc in nlp.pipe(texts, batch_size=32):
        sents = [s for s in doc.sents if len(s) > 1]
        n = max(len([t for t in doc if not t.is_punct]), 1)

        def depth(t):
            d = 0
            while t.head.i != t.i and d < 100:  # compare positions: spaCy makes a new Token object on each access
                t, d = t.head, d + 1
            return d
        pos = pd.Series([t.pos_ for t in doc if not t.is_punct]).value_counts()
        agree = sum(1 for t in doc if t.lower_ in {"he", "she", "it"} and t.dep_ == "nsubj" and t.head.tag_ == "VBP")
        an_err = sum(1 for a, b in zip(doc[:-1], doc[1:])
                     if (a.lower_ == "a" and b.text[:1].lower() in "aeiou") or (a.lower_ == "an" and b.text[:1].lower() not in "aeiouh"))
        out.append(dict(
            n_sents=len(sents),
            sent_len=np.mean([len(s) for s in sents]) if sents else 0.0,
            dep_depth_mean=np.mean([depth(t) for t in doc]) if len(doc) else 0.0,
            dep_depth_max=max([depth(t) for t in doc], default=0),
            subord_per_sent=sum(t.dep_ in SUBORD for t in doc) / max(len(sents), 1),
            noverb_sent_frac=np.mean([not any(t.pos_ in {"VERB", "AUX"} for t in s) for s in sents]) if sents else 1.0,
            agree_err_rate=agree / n,
            article_err_rate=an_err / n,
            past_ratio=sum(t.tag_ in {"VBD", "VBN"} for t in doc) / max(sum(t.pos_ in {"VERB", "AUX"} for t in doc), 1),
            **{f"pos_{k.lower()}": pos.get(k, 0) / n for k in
               ["NOUN", "VERB", "AUX", "ADJ", "ADV", "PRON", "DET", "ADP", "CCONJ", "SCONJ"]},
        ))
    return out


def languagetool(texts):
    import language_tool_python
    tool = language_tool_python.LanguageTool("en-US")
    out = []
    for t in texts:
        m = tool.check(t) if t else []
        n = max(len(t.split()), 1)
        cats = pd.Series([x.category for x in m]).value_counts()
        out.append(dict(lt_rate=100 * len(m) / n, lt_grammar=100 * cats.get("GRAMMAR", 0) / n,
                        lt_typos=100 * cats.get("TYPOS", 0) / n))
    tool.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tr", default=str(ROOT / "cache" / "transcripts.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "cache" / "feat_text.parquet"))
    ap.add_argument("--no-lt", action="store_true")
    a = ap.parse_args()
    import spacy
    recs = [json.loads(l) for l in open(a.tr)]
    texts = [clean_text(r) for r in recs]
    rows = [dict(split=r["split"], filename=r["filename"]) | fluency(r) for r in recs]
    print("fluency done", flush=True)
    for row, s in zip(rows, syntax(spacy.load("en_core_web_md"), texts)):
        row |= s
    print("syntax done", flush=True)
    for row, g in zip(rows, gpt2_scores(texts)):
        row |= g
    print("gpt2 done", flush=True)
    if not a.no_lt:
        for row, l in zip(rows, languagetool(texts)):
            row |= l
        print("languagetool done", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(a.out, index=False)
    print(f"saved {df.shape} -> {a.out}")


if __name__ == "__main__":
    main()
