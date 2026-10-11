# Method: how the grammar scoring engine works

## 1. The problem in one paragraph
Each input is a 45-60 s recording of someone speaking English. A human rater gave it a grammar score from 0 to 5 (steps of 0.5) using a rubric that looks at sentence structure, syntax, how often errors happen, and whether complex grammar is used. We have 769 scored clips to learn from and 216 test clips to score. Results are judged by RMSE and Pearson correlation.

## 2. What the research says
| Finding | Source | What it means for us |
|---|---|---|
| Pooled embeddings from a Whisper encoder, with no transcript, beat a Whisper-ASR + BERT text baseline for speaking-proficiency scoring (RMSE 0.384 vs 0.440). The model mostly used *how* people speak, not *what* they say. | Speak & Improve Challenge 2025 (Aalto, "One Whisper to Grade Them All") | Whisper-encoder audio embeddings are the strongest single starting point. Build them first. |
| wav2vec 2.0 audio-only scoring can match transcript-based systems, and combining the audio and transcript systems gives clear gains. | Bannò & Matassoni (Linguaskill, SLaTE 2023); "Proficiency assessment of L2 spoken English using wav2vec 2.0" | Build an audio branch and a transcript branch, and stack them. |
| Intermediate layers of speech models often carry more useful information than the last layer. | Wu & Chen 2024 (layer-wise features for speech assessment); Liang et al. 2025 (layer selection for MOS) | Extract several layers and pick by CV, rather than only using the last one. |
| Standard Whisper turns speech into clean text and drops fillers, stutters and false starts. Verbatim models exist, but the main one (CrisperWhisper) is CC-BY-NC, which is not allowed here. | CrisperWhisper paper (Interspeech 2024) | Use Whisper with a verbatim-style prompt, and rely on the audio branch for what ASR hides. |
| ASR + grammar-error tools (LanguageTool) + gradient boosting is the usual design for this kind of task; reported Pearson is about 0.78 under CV. | Public write-ups of similar grammar-scoring projects | A reasonable level for the transcript branch alone. The stack should beat it. |
| The Speak & Improve baseline is ASR + BERT text graders. | Speak & Improve 2025 task paper | Text embeddings of the transcript are a sensible extra branch. |

Why grammar is scored from audio at all. Raters hear the speech, not a transcript. Fluent speakers who pause less, restart less, and form complete sentences also tend to have better grammar. So audio features predict the grammar score well, even though grammar is "a text thing". The transcript adds the direct grammar evidence: errors, sentence types, complexity.

## 3. Our data (from the M1 audit)
- 769 train clips, mean score 3.31, std 1.24. 37 clips are scored 0, below the rubric's 1. They look like non-speech or off-task audio.
- Test clips are shorter (median 45 s against 60 s) and quieter. So all count features are made per second.
- A constant prediction scores RMSE 1.238 on train CV and 1.005 on the public leaderboard. So the test labels are less spread out than train, probably with fewer 0s. Predictions must not be over-spread, and clipping or calibration matters.

## 4. The engine (final design)

```
audio (16 kHz mono)
 ├─ A. Audio branch ──── Whisper-large-v3 encoder, layer-wise mean+std pooling ─┐
 │                      WavLM-large, layer-wise mean+std pooling ─────────────┤-> Ridge / SVR
 ├─ B. ASR ───────────── faster-whisper large-v3 (verbatim prompt, word times)
 │     ├─ C. Fluency ── speech rate, pauses, fillers, repetitions (per second) ┐
 │     ├─ D. Grammar ── LanguageTool error rate, GPT-2 perplexity, spaCy syntax ┤-> LightGBM / Ridge
 │     ├─ E. Text emb ─ DeBERTa-v3 mean-pooled transcript embedding ─────────────-> Ridge
 │     └─ F. LLM judge  Qwen2.5-7B-Instruct reads the rubric + transcript; ──────-> one feature
 │                       expected score from the 1-5 digit probabilities
 └─ Stack: non-negative Ridge on out-of-fold predictions -> clip to [0, 5] -> score
```

A. Audio embeddings (M2, first real submission). Whisper's encoder only takes 30 s at a time, so each clip is cut into 30 s windows. For each window we average (mean) and measure spread (std) over time, for several layers, then average the windows. That gives one fixed-length vector per clip and per layer. WavLM takes the whole clip. A Ridge (or SVR) model on standardized features maps vectors to scores. Ridge is used because 769 samples and about 2,500 features would overfit an unregularized model, and Ridge's penalty is tuned by CV.

B. Transcripts. faster-whisper large-v3 with beam 5, temperature 0, no conditioning on earlier text (this stops loops), word timestamps, and a prompt full of "um, uh, I, I mean" so it keeps disfluencies. We also keep its confidence scores, which flag mumbled or unclear speech.

C. Fluency. From word timestamps: words per second, number and length of pauses, long pauses, fillers, repeated words, and restarts. Everything is per second or per word, because test clips are shorter.

D. Grammar. LanguageTool counts grammar mistakes per 100 words. GPT-2 perplexity measures how unusual the word sequences are, and ungrammatical text scores high. spaCy measures sentence complexity: clause depth, subordinate clauses, and sentences without a verb.

E. Text embeddings. A frozen DeBERTa-v3 vector of the transcript. It captures vocabulary and structure in a general way.

F. LLM judge (optional). A local, open model rates the transcript using the exact rubric. We read its probabilities for the digits 1-5 and take the weighted average, which gives a smooth score rather than a single digit. It is one feature among many, never the whole model. No hosted APIs are used.

Stack. Each branch gives out-of-fold predictions on the same folds. A Ridge with non-negative weights learns how much to trust each branch. Showing each branch's score next to the stack makes the gain from combining visible.

Validation. Repeated stratified 5-fold CV (3 repeats, fixed folds saved to disk). Report RMSE, Pearson, and in-sample training RMSE. Leaderboard scores are logged but not chased.

## 5. Why this and not something else
- Not end-to-end fine-tuning first: 769 clips is little data. Frozen features + Ridge are fast, stable, and easy to explain. Fine-tuning is optional at the end, if CV plateaus.
- Not transcript-only: ASR cleans the speech and hides errors, and raters also hear fluency. The research shows audio-only can match or beat transcript-only.
- Not audio-only: combining both gives clear gains in the research, and the transcript branch adds explainable grammar evidence for the report.
- Not hosted LLM APIs: sending competition data out would break the data rule.

## 6. Licenses (all allow commercial use)
| Component | License |
|---|---|
| Whisper large-v3 (and faster-whisper) | MIT |
| WavLM-large | MIT |
| DeBERTa-v3 | MIT |
| GPT-2 | MIT |
| spaCy + English models | MIT |
| LanguageTool | LGPL-2.1 |
| Qwen2.5-7B-Instruct | Apache-2.0 |
| scikit-learn / LightGBM | BSD / MIT |
| CrisperWhisper | CC-BY-NC: not used |

## 7. Second review and changes made
The design was checked by a second model (TypeSafe Jev, jev-1.13.0). Only this document was sent: no audio, transcripts or per-file data. It agreed with the core choices: audio-first 0.84, two-branch stack 0.82, frozen over fine-tuning 0.83, CV scheme 0.78 (probability of "yes"). It flagged two gaps:

1. Too many parts to explain (0.33). Change: the engine has a fixed core of three branches: A (audio embeddings), C+D (fluency + grammar features) and the stack. E (text embeddings) and F (LLM judge) are add-ons. They stay only if they improve repeated-CV RMSE by more than the fold-to-fold noise. Anything that does not earn its place is dropped and listed in the report as "tried, no gain".
2. Label-0 clips and calibration (0.82 risk, top gap). Change: a dedicated step, done in M2 and M3:
   - Find out what the 37 label-0 clips are. Check whether they contain speech, using ASR no-speech probability and word count.
   - Compare by CV: (a) train on all clips; (b) train on scores 1-5 only, plus a separate "no valid speech" detector that outputs 0.
   - Calibrate the final predictions with a linear map fitted on out-of-fold predictions. Clip to the label range.
   - Use at most a few leaderboard submissions to check whether the test set has 0-type clips (the constant baseline suggests few). Never tune many knobs on the leaderboard.

### Second review, round 2 (full context: plan, milestones, risks, research claims, revised method)
Jev agreed most strongly with combining audio and transcript (0.91), rule compliance (0.85) and per-second features (0.79). Overall confidence was 2.69/4. Changes made:
- Layer choice: picking 1 of 33 layers on the same CV would overfit (0.57). We compare 4 fixed layer groups only (early / middle / late / all layers averaged) and report all four.
- Loudness: peak normalisation was judged weak (0.46). Clips are now RMS-normalised to about -20 dBFS, so one loud click cannot set the scale.
- Caveat on the evidence: the Whisper-encoder result (0.53) comes from overall proficiency scoring, not grammar alone. We keep it first because it is cheap, and the CV result decides.
- Label-0 check first: Jev's top suggestion (0.61). This runs locally in parallel with M2, because the embeddings are needed for all clips either way.

## Sources
- Speak & Improve Challenge 2025: Tasks and Baseline Systems. https://arxiv.org/abs/2412.11985
- One Whisper to Grade Them All (Aalto, SLaTE 2025). https://arxiv.org/abs/2507.17918
- Bannò & Matassoni, wav2vec 2.0 for Linguaskill. https://arxiv.org/abs/2211.08849
- Proficiency assessment of L2 spoken English using wav2vec 2.0. https://arxiv.org/abs/2210.13168
- Layer selection from SSL models for MOS prediction. https://arxiv.org/abs/2508.08962
- CrisperWhisper. https://arxiv.org/abs/2408.16589
- Spoken Grammar Assessment Using LLM. https://arxiv.org/abs/2410.01579
