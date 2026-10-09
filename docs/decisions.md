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
- **Submission ids:** `sample_submission.csv` has 204 rows, and only 25 of its file names exist in `test/`. `test.csv` lists exactly the 216 test files. Submissions use the 216 ids and order from `test.csv`; the first format-check submission confirms whether Kaggle accepts this.
- **Competition slug:** `shl-hiring-assessment-2026`. The slug in the handoff notes (`shl-intern-hiring-assessment`) returns 403.
