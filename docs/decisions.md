# Design decisions

Each entry: what was chosen, why, what else was considered, and what it gained (measured numbers only).

## D1. Environment
- **What:** Python 3.11 virtual environment (`.venv`), packages in `requirements.txt`. Light work (audit, CV, modeling) runs on the Mac (Apple M5, 16 GB RAM). Heavy jobs (Whisper large-v3, WavLM embeddings) run on Kaggle GPU.
- **Why:** 16 GB RAM is enough for tabular models and small embedding matrices, but large speech models on 985 files are much faster on a GPU.

## D2. Data stays local
- **What:** `data/`, `cache/`, `outputs/` and all derived files (transcripts, embeddings, features, labels) are gitignored and never uploaded.
- **Why:** Competition rules say the data is for competition use only. Derived data is treated the same way to be safe.

## D3. Data audit findings (M1)
- **Labels:** 769 train clips, scores 0 to 5 in steps of 0.5. Mean 3.31, std 1.24, skewed toward high scores. 37 clips are labelled **0**, although the rubric starts at 1. These 37 clips have almost no pauses and are louder than the rest, so they may be non-speech or off-task audio. Scores 1.0 and 1.5 are very rare (4 clips), so the model will see few examples at the low end.
- **Mean baseline:** always predicting 3.31 gives RMSE 1.238 and Pearson 0. Every model is compared to this.
- **Audio:** all files are already 16 kHz, mono, 16-bit, with no clipping and no duplicate files. So no resampling is needed.
- **Train/test shift:** test clips are shorter (median 45 s against 60 s in train) and quieter (median loudness -26 dB against -24 dB). One test clip is only 6.5 s. Features that grow with clip length (word counts, pause counts) must be normalised per second.
- **File numbers:** in train, the number in the file name correlates with the label (Pearson -0.55), because the files come in blocks with different average scores. Test files are numbered 0-215 separately, so this pattern cannot carry over. It is **not** used as a feature: it would not be a grammar signal.
- **Submission ids:** `sample_submission.csv` has 204 rows, and only 25 of its file names exist in `test/`. `test.csv` lists exactly the 216 test files. Submissions use the 216 ids and order from `test.csv`; Confirmed: Kaggle accepted the 216-row file from `test.csv` (submission e000, public score 1.0050).
- **Competition slug:** `shl-hiring-assessment-2026`. The slug in the handoff notes (`shl-intern-hiring-assessment`) returns 403.

## D4. What the label-0 clips are
- **Check:** Whisper-small (local) ran on the 37 label-0 clips and on 37 random other train clips. Only aggregate numbers were looked at.
- **Result:**
  - Label-0 clips have a higher no-speech probability (mean 0.63 vs 0.35) and lower ASR confidence (mean log-prob -0.98 vs -0.39).
  - 21 of 37 are not confidently detected as English, against 5 of 37 for other clips.
  - 9 of 37 have fewer than 20 words, against 0 for other clips.
- **Meaning:** most label-0 clips are invalid recordings: noise, music, unclear audio or non-English speech, rather than very poor grammar. They can likely be detected from ASR confidence and audio features. So a "valid speech" detector plus a 1-5 grammar model is a realistic design. It is compared by CV against a single model trained on all clips.

## D5. Audio embedding results and the label-0 decision (M2)
- **Large models vs tiny:** Whisper-large-v3 + WavLM-large reach CV RMSE 0.524 and Pearson 0.906. Whisper-tiny alone gave 0.568 / 0.889.
- **WavLM vs Whisper:** WavLM alone does better than Whisper alone (0.538 vs 0.554 RMSE), and concatenating both is better than either.
- **Layer groups:** for both models, "all layers averaged" beats any single third, and the early layers are clearly the weakest. So the score depends on higher-level information (words and structure), not only on sound.
- **Label-0 clips are not a problem for this model:** it already predicts them close to 0 (mean prediction 0.03, RMSE 0.28 on these clips). They cause only 1.4% of the squared error. A separate detector improved CV RMSE by only 0.004, which is smaller than the fold-to-fold noise (0.008). So we keep the single, simpler model. Jev's concern was valid to check, and the measurement shows it does not need extra machinery.
- **Where the error is:** the model squeezes predictions toward the middle. Clips scored 2.0 are predicted 2.59 on average, and clips scored 5.0 are predicted 4.72. Most of the remaining error is in grading clips within 1-5. This is where transcript and grammar features (M3) should help.
