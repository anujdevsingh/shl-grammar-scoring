# Spoken Grammar Scoring Engine (SHL Hiring Assessment 2026)

Predicts a grammar score from 0 to 5 for a 45–60 second recording of spoken English. The score is the one a human rater would give using SHL's rubric.

Author: Anuj Dev Singh

## Approach
A rater *hears* the speech, so the engine scores the audio directly and uses the transcript as a second view. Five branches each predict a score:

1. **Frozen encoders + Ridge.** Whisper-large-v3 encoder and WavLM-large. For every layer, take the mean and standard deviation over time, then average the layers. A Ridge regression maps the result to a score.
2. **Fine-tuned WavLM-large,** trained end to end on the scores: random 20 s crops, 8 fixed epochs.
3. **Fine-tuned audio LLM.** Qwen2-Audio-7B in 4-bit, with LoRA adapters and a regression head (QLoRA), fed the rubric prompt and 30 s of audio. 3 fixed epochs.
4. **Adapter grader.**
   - Small per-layer adapters and one Transformer layer on frozen Whisper-large-v3 frames (about 1M trainable parameters). This is the recipe of the winning Speak & Improve 2025 system's best single model, and it is our best single model.
   - It is trained with **leakage-free pseudo-labels**: for each fold, a teacher that never saw that fold's labels labels the held-out clips and the test clips.
5. **Transcript features.**
   - faster-whisper large-v3, prompted to keep fillers and restarts.
   - 42 per-second and per-word features: pauses, speech rate, fillers, ASR confidence, vocabulary variety, LanguageTool errors per 100 words, GPT-2 likelihood, spaCy syntax depth and clauses.
   - These go into a Ridge regression.

**Stack.** A non-negative linear regression on the out-of-fold predictions, clipped to [0, 5].

**No invalid clips in test.** The 37 train clips scored 0 are invalid recordings, and the test set has none. So the models learn from scores 1–5, and a frozen detector trained on all clips keeps any invalid-looking clip near 0.

**Validation.**
- Every model uses the same fixed, label-stratified 5-fold CV.
- Anything new is kept only if it is better on at least 4 of 5 folds and the bootstrap 95% interval of the gain is above 0.
- `docs/decisions.md` records every choice and every measured attempt, including the ones that failed.

## Results
| Model | CV RMSE | CV Pearson | Training RMSE (in-sample) | Public LB RMSE |
|---|---|---|---|---|
| Mean baseline | 1.238 | 0.000 | 1.238 | 1.0050 |
| **e009**: frozen + fine-tuned WavLM + fine-tuned audio LLM + transcript | 0.492 | 0.918 | 0.370 | **0.3655** |
| **e013**: e009 branches (trained on scores 1–5) + pseudo-labelled adapter grader | **0.470** | **0.925** | 0.443 | 0.3757 |

These are the two final submissions.
- **CV:** e013 is better than e009 on all 5 folds and on the train clips that look most like test.
- **Public leaderboard:** e009 is better. Every submission with the adapter scored better in CV but worse on the public leaderboard.
- **Why both are kept:** the public split is only about 70 clips, and a label-free check shows that the adapter's predictions drift on the test recordings. So the two picks hedge both views. The notebook (section 6) has the full story.

**Training RMSE.** The fine-tuned and adapter branches have no all-data model: their test prediction averages the 5 fold models. So their out-of-fold predictions are used inside the stack's training RMSE. That is why e013's in-sample figure is close to its CV figure.

| Kaggle submission | CV RMSE | Public LB RMSE |
|---|---|---|
| Mean baseline | 1.238 | 1.0050 |
| Whisper-tiny + Ridge | 0.568 | 0.4529 |
| Whisper-large-v3 + WavLM-large + Ridge | 0.524 | 0.4165 |
| + fine-tuned WavLM | 0.511 | 0.3985 |
| + transcript features | 0.500 | 0.3865 |
| + fine-tuned audio LLM (**e009**, final) | 0.492 | **0.3655** |
| + adapter grader | 0.476 | 0.3742 |
| adapter round 2 (short-clip aware) | 0.474 | 0.3790 |
| adapter + pseudo-labels (**e013**, final) | **0.470** | 0.3757 |

## Repository layout
```
notebooks/   shl_grammar_scoring.ipynb: the report (approach, results, plots, CV vs leaderboard, error analysis)
             shl_grammar_scoring.py: the same notebook as a plain-text (jupytext) source
src/         audit, CV folds, embeddings, transcription, features, fine-tuning, audio LLM, adapter grader,
             pseudo-labels, final stack, submission check
kaggle/      scripts that build the Kaggle GPU jobs (one worker per T4 GPU)
docs/        method.md (research basis) and decisions.md (why each choice was made, with measured numbers)
experiments/ log.csv: every experiment with CV and leaderboard scores
```
The competition audio, labels, transcripts and every derived feature are **not** included: the competition data is for competition use only.

## Reproduce
1. `pip install -r requirements.txt` (Python 3.11). LanguageTool also needs Java.
2. Download the data with the Kaggle CLI into `data/raw/` (needs a Kaggle account that joined the competition).
3. `python src/audit.py` and `python src/cv.py` (fixed folds).
4. Run the GPU stages on Kaggle (T4 ×2). For each one, run its builder, then `kaggle kernels push -p kaggle/<stage> --accelerator NvidiaTeslaT4`, then download its output into `cache/`. The builders, in order:
   - `kaggle/build_embed.sh`
   - `kaggle/build_transcribe.sh`
   - `kaggle/build_finetune.sh`
   - `kaggle/build_ft_alm.sh`
   - `kaggle/build_adapter.sh`
5. `python src/features.py`.
6. Build the teacher, make the pseudo-labels and train the pseudo-labelled adapter:
   - `python src/final.py --exp e011 --no-judge --nz --drop adapter2_whisper,adapter3_whisper,adapter4mean_whisper,adapter4meanvar_whisper`
   - `python src/pseudo.py`
   - `kaggle/build_adapter3.sh`, then push it and download its output as in step 4.
7. Build the two final submissions:
   - `python src/final.py --exp e013 --no-judge --nz --drop adapter_whisper,adapter2_whisper,adapter4mean_whisper,adapter4meanvar_whisper`
   - `python src/final.py --exp e009 --no-judge --drop adapter_whisper,adapter_wavlm,adapter2_whisper,adapter3_whisper,adapter4mean_whisper,adapter4meanvar_whisper`

   Both reproduce the submitted files exactly from the cached branch outputs.
8. Run `notebooks/shl_grammar_scoring.ipynb`. It reads only the cached results.

## Models and licences
- **Used:** Whisper-large-v3 (MIT), WavLM-large (MIT), Qwen2-Audio-7B-Instruct (Apache-2.0), faster-whisper (MIT), GPT-2 (MIT), spaCy (MIT), LanguageTool (LGPL-2.1), PEFT (Apache-2.0), scikit-learn (BSD), PyTorch (BSD).
- **Also tried:** w2v-BERT 2.0 (MIT) and LightGBM (MIT).
- **No external labelled data** was used.
