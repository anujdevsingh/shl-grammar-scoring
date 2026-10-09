# Spoken Grammar Scoring (SHL Hiring Assessment 2026)

Predicts a grammar score from 0 to 5 for a 45-60 second audio clip of speech.

Author: Anuj Dev Singh

## Approach (to be completed with measured results)
- Audio preprocessing: mono, 16 kHz.
- Features: pretrained audio embeddings (Whisper encoder, WavLM), transcript-based grammar and fluency features, optional text embeddings.
- Models: Ridge / SVR / gradient boosting, combined with a simple stack.
- Validation: repeated stratified K-fold with fixed folds. Metrics: RMSE and Pearson correlation.

## Results
| Model | CV RMSE | CV Pearson |
|---|---|---|
| Mean baseline | | |
| Audio embeddings + Ridge | | |
| Grammar/fluency features + GBM | | |
| Stack | | |

Training RMSE (in-sample): _fill in from the final notebook_.

## Repository layout
```
notebooks/   final notebook with the report
src/         feature extraction and modeling code
docs/        design decisions
experiments/ experiment log and CV fold ids
```
The competition audio, transcripts and derived features are not included, because the competition data is for competition use only.

## Reproduce
1. Install requirements: `pip install -r requirements.txt`.
2. Download the data with the Kaggle CLI into `data/` (needs a Kaggle account that joined the competition).
3. Run the stages in `src/` in the order listed in the notebook, then run the notebook.
