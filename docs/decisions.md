# Design decisions

Each entry: what was chosen, why, what else was considered, and what it gained (measured numbers only).

## D1. Environment
- **What:** Python 3.11 virtual environment (`.venv`), packages in `requirements.txt`. Light work (audit, CV, modeling) runs on the Mac (Apple M5, 16 GB RAM). Heavy jobs (Whisper large-v3, WavLM embeddings) run on Kaggle GPU.
- **Why:** 16 GB RAM is enough for tabular models and small embedding matrices, but large speech models on 985 files are much faster on a GPU.

## D2. Data stays local
- **What:** `data/`, `cache/`, `outputs/` and all derived files (transcripts, embeddings, features, labels) are gitignored and never uploaded.
- **Why:** Competition rules say the data is for competition use only. Derived data is treated the same way to be safe.
