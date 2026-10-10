# Spoken Grammar Scoring Engine (SHL Hiring Assessment 2026)

Predicts a grammar score from 0 to 5 for a 45–60 second recording of spoken English. The score is the one a human rater would give using SHL's rubric.

Author: Anuj Dev Singh

## Approach
A rater *hears* the speech, so the engine scores the audio directly and uses the transcript as a second view:

1. **Frozen audio embeddings.** Whisper-large-v3 encoder and WavLM-large. For every layer, take the mean and standard deviation over time, then average the layers. A Ridge regression maps the result to a score.
2. **Fine-tuned audio model.** WavLM-large trained end to end on the scores: random 20 s crops, 8 fixed epochs, learned layer weights and a small MLP head.
3. **Transcript features.** faster-whisper large-v3, prompted to keep fillers and restarts. From the transcript, 42 per-second / per-word features: pauses, speech rate, fillers, ASR confidence, vocabulary variety, LanguageTool errors per 100 words, GPT-2 likelihood, and spaCy syntax depth and clauses. These go into a Ridge regression.
4. **Stack.** A non-negative linear regression on the out-of-fold predictions of 1–3. Final predictions are clipped to [0, 5].

Every model uses the same fixed, label-stratified 5-fold CV. Ideas that did not beat the fold-to-fold noise were dropped and are listed in the notebook: SVR, w2v-BERT 2.0, a label-0 detector, and a zero-shot audio-LLM judge.

## Results
| Model | CV RMSE | CV Pearson | Training RMSE (in-sample) |
|---|---|---|---|
| Mean baseline | 1.238 | 0.000 | 1.238 |
| Frozen Whisper + WavLM + Ridge | 0.521 | 0.907 | 0.249 |
| Fine-tuned WavLM | 0.550 | 0.896 | – |
| Transcript features + Ridge | 0.842 | 0.735 | 0.789 |
| **Final stack** | **0.500** | **0.915** | **0.348** |

The in-sample training RMSE of the stack uses the fine-tuned model's out-of-fold predictions, because no all-data fine-tuned model exists. Its test prediction averages the 5 fold models.

| Kaggle submission | Public leaderboard RMSE |
|---|---|
| Mean baseline | 1.0050 |
| Whisper-tiny + Ridge | 0.4529 |
| Whisper-large-v3 + WavLM-large + Ridge | 0.4165 |
| + fine-tuned WavLM | 0.3985 |
| + transcript features (final) | 0.3865 |

## Repository layout
```
notebooks/   shl_grammar_scoring.ipynb: the report (approach, results, plots, error analysis)
src/         audit, CV folds, embeddings, transcription, features, fine-tuning, stacking, submission check
kaggle/      scripts that build the Kaggle GPU jobs (one worker per T4 GPU)
docs/        method.md (research basis) and decisions.md (why each choice was made, with measured numbers)
experiments/ log.csv: every experiment with CV and leaderboard scores
```
The competition audio, labels, transcripts and every derived feature are **not** included: the competition data is for competition use only.

## Reproduce
1. `pip install -r requirements.txt` (Python 3.11). LanguageTool also needs Java.
2. Download the data with the Kaggle CLI into `data/raw/` (needs a Kaggle account that joined the competition).
3. `python src/audit.py` and `python src/cv.py` (fixed folds).
4. GPU stages on Kaggle (T4 ×2): `kaggle/build_embed.sh`, `kaggle/build_transcribe.sh` and `kaggle/build_finetune.sh`, then `kaggle kernels push -p kaggle/<stage> --accelerator NvidiaTeslaT4`. Download each output into `cache/`.
5. `python src/features.py`, then `python src/final.py --no-judge`. This writes `outputs/submission_final.csv`.
6. Run `notebooks/shl_grammar_scoring.ipynb`. It reads only the cached results.

## Models and licences
Whisper-large-v3 (MIT), WavLM-large (MIT), faster-whisper (MIT), GPT-2 (MIT), spaCy (MIT), LanguageTool (LGPL-2.1), scikit-learn (BSD), LightGBM (MIT). Also tried: w2v-BERT 2.0 (MIT) and Qwen2-Audio-7B-Instruct (Apache-2.0).
