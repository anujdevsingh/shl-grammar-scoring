# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Spoken Grammar Scoring Engine
# **SHL Hiring Assessment 2026 · Anuj Dev Singh**
#
# **Task.** Given a 45–60 s recording of someone speaking English, predict the grammar score (0–5, steps of 0.5)
# that a human rater gave using SHL's rubric. 769 scored training clips, 216 test clips. Metrics: RMSE and Pearson.
#
# **Approach in one paragraph.** A rater *hears* the speech, so the engine scores the audio directly, with the
# transcript as a second view. (1) Frozen speech encoders (Whisper-large-v3 and WavLM-large) turn each clip into a
# fixed-length vector, and a Ridge regression maps it to a score. (2) WavLM-large is also fine-tuned end to end
# on the scores. (3) Whisper transcripts give fluency and grammar features (pauses, speech rate, fillers,
# LanguageTool errors, GPT-2 likelihood, syntax depth). (4) Optionally, an open audio-LLM (Qwen2-Audio) listens
# to each clip and rates it with the rubric. A non-negative linear stack combines the branches. Every number
# below comes from the same fixed cross-validation folds.
#
# **Contents.** 1 Data · 2 Preprocessing · 3 Features and models · 4 Results (training RMSE, CV RMSE, Pearson)
# · 5 Visualisations · 6 Error analysis and limitations · 7 Reproduce
#
# *Note: the competition data is for competition use only. This notebook shows aggregate numbers and plots only.
# It never prints audio, transcripts or per-file rows.*

# %%
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
from cv import DATA, pearson, rmse  # noqa: E402

CACHE = ROOT / "cache"
FINAL = "final"  # experiment name written by src/final.py
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
BLUE, ORANGE, GREY = "#2f6db5", "#e07b39", "#8c8c8c"

# %% [markdown]
# ## 1. Data
# Labels are skewed toward higher scores. 37 clips are scored **0**, below the rubric's 1. A check with Whisper showed
# that these are mostly invalid recordings: high no-speech probability, low ASR confidence, often not confidently
# English. The test clips are **shorter** (median 45 s vs 60 s) and slightly quieter, so every count feature is
# normalised per second or per word.

# %%
train = pd.read_csv(DATA / "train.csv")
audit = pd.read_csv(CACHE / "audit_audio.csv")
y_all = train.label
print(f"train clips {len(train)}, test clips {len(pd.read_csv(DATA / 'test.csv'))}")
print(f"label mean {y_all.mean():.3f}, std {y_all.std(ddof=0):.3f}, clips labelled 0: {(y_all == 0).sum()}")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
vc = y_all.value_counts().sort_index()
ax[0].bar(vc.index, vc.values, width=0.4, color=BLUE)
ax[0].set(title="Training label distribution", xlabel="grammar score", ylabel="clips")
for s, c in [("train", BLUE), ("test", ORANGE)]:
    ax[1].hist(audit[audit.split == s].duration, bins=30, alpha=0.6, color=c, label=s, density=True)
ax[1].set(title="Clip duration: test clips are shorter", xlabel="seconds", ylabel="density")
ax[1].legend(frameon=False)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Preprocessing
# - All files are already 16 kHz mono 16-bit, with no clipping and no duplicate files, so no resampling is needed.
# - **Loudness (RMS) normalisation** to about −20 dBFS, so quiet test clips match train. RMS rather than peak, so one
#   click cannot set the scale.
# - **No silence trimming.** Pauses and hesitations are part of what raters hear.
# - Whisper's encoder reads at most 30 s, so clips are cut into 30 s windows and the pooled vectors are averaged,
#   weighted by window length. Padding frames are ignored.
#
# ## 3. Features and models
# | Branch | What it captures | Model |
# |---|---|---|
# | **Frozen audio** | Whisper-large-v3 encoder + WavLM-large; mean and std over time of every layer, averaged per model | Standardise → RidgeCV |
# | **Fine-tuned audio** | WavLM-large trained on our scores (20 s random crops, 8 fixed epochs, learned layer weights + small MLP head) | 5 fold models |
# | **Transcript** | faster-whisper large-v3 transcripts (prompted to keep "um", restarts). 42 features: pauses, speech rate, fillers, ASR confidence, vocabulary variety, LanguageTool errors / 100 words, GPT-2 likelihood, syntax depth, clauses | Standardise → RidgeCV |
# | **Audio-LLM judge** (optional) | Qwen2-Audio-7B listens and rates grammar with the rubric; expected score from the probabilities of "1"–"5" | RidgeCV on judge scores |
# | **Stack** | how much to trust each branch | Non-negative linear regression on out-of-fold predictions |
#
# **Validation.** Stratified 5-fold CV with folds fixed once and reused for every model. The frozen models used
# 3 repeats. Fine-tuning ran on repeat 0 only (it costs about 80 GPU-minutes per fold), so the final comparison uses
# repeat 0. The stack is itself cross-validated: its weights are fitted on 4 folds and scored on the 5th.
#
# Example of what the transcript features measure, on a **made-up** sentence (not competition data):

# %%
import spacy  # noqa: E402

nlp = spacy.load("en_core_web_md")
doc = nlp("Yesterday he go to the market and, um, I buyed a apple because it was cheap.")
print("tokens:", [t.text for t in doc])
print("he + base-form verb (agreement error):",
      [(t.text, t.head.text) for t in doc if t.lower_ == "he" and t.dep_ == "nsubj" and t.head.tag_ == "VBP"])
print("'a' before a vowel sound:", [(a.text, b.text) for a, b in zip(doc[:-1], doc[1:]) if a.lower_ == "a" and b.text[0] in "aeiou"])
print("subordinate clauses:", [t.text for t in doc if t.dep_ in {"advcl", "ccomp", "relcl", "mark"}])

# %% [markdown]
# ## 4. Results
# **Training RMSE (in-sample)** = every model refit on all 769 clips and scored on those same clips. It is optimistic
# by design. **CV RMSE / Pearson** = each clip predicted by a model that never saw it. This is the honest estimate.
# The fine-tuned model has no all-data version (its test prediction averages the 5 fold models), so inside the
# stack's training RMSE its out-of-fold predictions are used.

# %%
table = pd.read_csv(CACHE / f"final_{FINAL}_table.csv")
summ = json.load(open(CACHE / f"final_{FINAL}.json"))
fr = pd.read_csv(CACHE / "e002_results.csv")
display_table = table.rename(columns={"model": "Model", "cv_rmse": "CV RMSE", "cv_pearson": "CV Pearson",
                                      "train_rmse": "Training RMSE (in-sample)"})
print(display_table.round(4).to_string(index=False))
print(f"\nFINAL ENGINE: training RMSE (in-sample) {summ['train_rmse']:.4f} | CV RMSE {summ['cv_rmse']:.4f} | "
      f"CV Pearson {summ['cv_pearson']:.4f}")
print("per-fold CV RMSE:", np.round(summ["fold_rmse"], 4))
print("stack weights:", {k: round(v, 3) for k, v in summ["weights"].items()})

# %% [markdown]
# **Public leaderboard (RMSE) for the submissions made along the way:**
#
# | Submission | What | CV RMSE | Public LB |
# |---|---|---|---|
# | e000 | predict the train mean | 1.238 | 1.0050 |
# | e001 | Whisper-tiny frozen + Ridge | 0.568 | 0.4529 |
# | e002 | Whisper-large-v3 + WavLM-large frozen + Ridge | 0.524 | 0.4165 |
# | e006 | e002 stacked with fine-tuned WavLM | 0.511 | 0.3985 |
#
# The leaderboard moved in the same direction as CV every time, so the CV setup can be trusted for decisions.

# %% [markdown]
# ## 5. Visualisations

# %%
d = np.load(CACHE / f"final_{FINAL}.npz", allow_pickle=True)
y, st = d["y"], d["stack_oof"]
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
jit = np.random.default_rng(0).normal(0, 0.06, len(y))
ax[0].scatter(y + jit, st, s=8, alpha=0.45, color=BLUE)
ax[0].plot([0, 5], [0, 5], color=GREY, lw=1, ls="--")
ax[0].set(title=f"Out-of-fold prediction vs true (Pearson {pearson(y, st):.3f})", xlabel="true score", ylabel="predicted")
ax[1].scatter(st, st - y, s=8, alpha=0.45, color=BLUE)
ax[1].axhline(0, color=GREY, lw=1, ls="--")
ax[1].set(title="Residuals", xlabel="predicted", ylabel="predicted − true")
g = pd.DataFrame(dict(y=y, e2=(st - y) ** 2)).groupby("y")
by = g.e2.mean().pow(0.5)
ax[2].bar(by.index, by.values, width=0.35, color=ORANGE)
for xv, n in g.size().items():
    ax[2].text(xv, by[xv] + 0.02, str(n), ha="center", fontsize=7, color=GREY)
ax[2].set(title="CV RMSE by true score (n above bars)", xlabel="true score", ylabel="RMSE")
plt.tight_layout()
plt.show()

# %% [markdown]
# Which layers matter? For both encoders, averaging all layers beats any third alone, and the early layers are the
# weakest. So the score depends on higher-level information about words and structure, not only on raw sound.
# Newer is not automatically better: frozen w2v-BERT 2.0 (2023, 4.5 M hours of pre-training) was the weakest encoder.

# %%
lg = pd.read_csv(CACHE / "e005_results.csv")
lg = lg[lg.layers != "concat"].copy()
lg["model"], lg["group"] = lg.features.str.split("_").str[0], lg.features.str.split("_").str[1]
piv = lg.pivot(index="group", columns="model", values="rmse").loc[["early", "middle", "late", "all"]]
ax = piv.plot.bar(figsize=(7, 3.2), color=[BLUE, ORANGE, GREY], rot=0)
ax.set(title="Frozen encoders: CV RMSE by layer group (lower is better)", ylabel="CV RMSE", xlabel="", ylim=(0.5, 0.65))
ax.legend(frameon=False, ncol=3)
plt.tight_layout()
plt.show()

# %% [markdown]
# Fine-tuning curves: validation RMSE per epoch for each fold. The number of epochs (8) was fixed in advance. No
# epoch was picked by looking at these curves, so the out-of-fold score stays honest.

# %%
logs = pd.concat([pd.read_csv(CACHE / "ft" / f"log_fold{k}.csv") for k in range(5)])
fig, ax = plt.subplots(figsize=(7, 3.2))
for k, gk in logs.groupby("fold"):
    ax.plot(gk.epoch, gk.val_rmse, marker="o", ms=3, lw=1.2, label=f"fold {k}")
ax.axhline(0.5205, color=GREY, ls="--", lw=1)
ax.text(8, 0.53, "frozen Ridge", color=GREY, fontsize=8, ha="right")
ax.set(title="Fine-tuned WavLM: validation RMSE per epoch", xlabel="epoch", ylabel="RMSE", ylim=(0.45, 1.0))
ax.legend(frameon=False, ncol=5, fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# What the transcript features say: the strongest single signals are Whisper's own **word confidence** and
# **vocabulary variety**. The direct grammar measures (LanguageTool error rate, GPT-2 likelihood, syntax) are weaker.
# Whisper rewrites disfluent, ungrammatical speech into clean text, so many errors never reach the transcript.
# This is why the audio branches carry most of the weight.

# %%
F = pd.read_parquet(CACHE / "feat_text.parquet")
F = F[F.split == "train"].merge(train, on="filename")
cols = [c for c in F.columns if c not in ("split", "filename", "label", "duration", "n_words", "n_sents")]
corr = pd.Series({c: np.corrcoef(F[c], F.label)[0, 1] for c in cols if F[c].std() > 0})
top = corr.reindex(corr.abs().sort_values().index[-14:])
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(top.index, top.values, color=[BLUE if v > 0 else ORANGE for v in top.values])
ax.axvline(0, color=GREY, lw=0.8)
ax.set(title="Transcript features: Pearson correlation with the grammar score", xlabel="Pearson r")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Error analysis and limitations
# - **Squeezing toward the middle.** Low scores are predicted too high and 5s too low (see "RMSE by true score").
#   Scores 1.0–1.5 have only 4 training clips, so the model has almost no examples of the bottom of the scale.
# - **Label-0 clips are easy.** Most are invalid audio and are predicted close to 0. A separate detector was tried:
#   it gained 0.004 CV RMSE, which is within noise, so it was dropped.
# - **What ASR cannot see.** Whisper cleans up grammar errors, so transcript-based grammar features are weak. The
#   audio-only branches do not have this problem, but they also respond to fluency and confidence, not grammar alone.
# - **Small data.** 615 training clips per fold. Fine-tuning alone (CV 0.550) did *not* beat frozen features +
#   Ridge (0.521). It helps only as a second opinion in the stack.
# - **Train/test shift.** Test clips are shorter and quieter. Count features are per second, raw duration and
#   word counts are excluded, and audio is loudness-normalised. The leaderboard (lower RMSE than CV) suggests the
#   test labels are less spread out than train.
# - **Tried, no gain (kept for transparency):** SVR instead of Ridge (0.523 vs 0.524), frozen w2v-BERT 2.0 added to
#   the stack (0.520), and a label-0 detector (0.520).
#
# ## 7. Reproduce
# ```
# python src/audit.py                       # data checks
# python src/cv.py                          # fixed folds -> cache/folds.csv
# kaggle/build_embed.sh; kaggle kernels push -p kaggle/embed --accelerator NvidiaTeslaT4      # frozen embeddings
# kaggle/build_transcribe.sh; kaggle kernels push -p kaggle/transcribe --accelerator NvidiaTeslaT4
# kaggle/build_finetune.sh; kaggle kernels push -p kaggle/finetune --accelerator NvidiaTeslaT4
# kaggle/build_judge.sh; kaggle kernels push -p kaggle/judge --accelerator NvidiaTeslaT4      # optional
# kaggle kernels output <kernel> -p cache/<dir>   # emb, ft, judge; transcripts -> cache/transcripts.jsonl
# python src/features.py                    # transcript features (needs Java for LanguageTool)
# python src/train_emb.py; python src/train_text.py
# python src/final.py                       # stack + outputs/submission_final.csv
# ```
# Heavy stages run on Kaggle T4 ×2 (one worker per GPU) and write checkpoints, so a dropped session can resume.
# This notebook only reads the cached results in `cache/`.
