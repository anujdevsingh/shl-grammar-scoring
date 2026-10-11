# Spoken Grammar Scoring Engine (SHL Hiring Assessment 2026)

This project predicts a grammar score from 0 to 5 for a 45-60 second recording of spoken English, the same score a human rater would give using SHL's rubric.

Author: Anuj Dev Singh

## Approach
A rater listens to the speech, so I score the audio directly and use the transcript as a second source of information. Five models each predict a score:

1. Frozen encoders + Ridge. I run the Whisper-large-v3 encoder and WavLM-large over each clip, take the mean and standard deviation over time for every layer, average the layers, and fit a Ridge regression on top.
2. Fine-tuned WavLM-large, trained end to end on the scores (random 20 s crops, 8 epochs).
3. Fine-tuned audio LLM. Qwen2-Audio-7B loaded in 4-bit, with LoRA adapters and a regression head (QLoRA). It gets the rubric as a prompt plus 30 s of audio, and trains for 3 epochs.
4. Adapter grader. Small per-layer adapters and one Transformer layer on top of frozen Whisper-large-v3 frames, about 1M trainable parameters. This follows the best single model of the winning Speak & Improve 2025 system, and it is also my best single model. I train it with pseudo-labels: for each fold, a teacher that never saw that fold's labels scores the held-out clips and the test clips, so no label leaks into validation.
5. Transcript features. faster-whisper large-v3 transcripts (prompted so it keeps fillers and restarts), turned into 42 per-second and per-word features: pauses, speech rate, fillers, ASR confidence, vocabulary variety, LanguageTool errors per 100 words, GPT-2 likelihood, and spaCy syntax depth and clauses. These go into a Ridge regression.

A non-negative linear regression on the out-of-fold predictions combines the five, and the result is clipped to [0, 5].

The 37 training clips scored 0 turned out to be invalid recordings, and the test set has none of them. So the models learn from the clips scored 1-5, and a separate frozen model trained on all clips keeps anything that looks invalid near 0.

Every model uses the same fixed, label-stratified 5-fold CV. I only kept a change if it improved at least 4 of 5 folds and the bootstrap 95% interval of the gain stayed above 0. `docs/decisions.md` explains each choice with the measured numbers, including the ideas that did not work.

## Results
| Model | CV RMSE | CV Pearson | Training RMSE (in-sample) | Public LB RMSE |
|---|---|---|---|---|
| Mean baseline | 1.238 | 0.000 | 1.238 | 1.0050 |
| e009: frozen + fine-tuned WavLM + fine-tuned audio LLM + transcript | 0.492 | 0.918 | 0.370 | 0.3655 |
| e013: the e009 models (trained on scores 1-5) + pseudo-labelled adapter grader | 0.470 | 0.925 | 0.443 | 0.3757 |

These are my two final submissions. e013 is better in cross-validation: on all 5 folds, and also on the training clips that look most like the test set. e009 is better on the public leaderboard. In fact every submission that included the adapter did better in CV and worse on the public leaderboard. The public split is only about 70 clips, but a label-free check also showed the adapter's predictions drifting on the test recordings, so I kept one submission of each kind. Section 6 of the notebook goes through this in detail.

A note on training RMSE: the fine-tuned and adapter models have no version trained on all the data (their test prediction is the average of the 5 fold models), so their out-of-fold predictions stand in when computing the stack's training RMSE. That is why e013's in-sample number is close to its CV number.

| Kaggle submission | CV RMSE | Public LB RMSE |
|---|---|---|
| Mean baseline | 1.238 | 1.0050 |
| Whisper-tiny + Ridge | 0.568 | 0.4529 |
| Whisper-large-v3 + WavLM-large + Ridge | 0.524 | 0.4165 |
| + fine-tuned WavLM | 0.511 | 0.3985 |
| + transcript features | 0.500 | 0.3865 |
| + fine-tuned audio LLM (e009, final) | 0.492 | 0.3655 |
| + adapter grader | 0.476 | 0.3742 |
| adapter round 2 (short-clip aware) | 0.474 | 0.3790 |
| adapter + pseudo-labels (e013, final) | 0.470 | 0.3757 |

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
The competition audio, labels, transcripts and anything derived from them are not in this repo, because the competition data is for competition use only.

## Reproduce
1. `pip install -r requirements.txt` (Python 3.11). LanguageTool also needs Java.
2. Download the data with the Kaggle CLI into `data/raw/` (you need a Kaggle account that has joined the competition).
3. Run `python src/audit.py` and `python src/cv.py` (this fixes the folds).
4. Run the GPU stages on Kaggle (T4 x2). For each stage, run its builder, push it with `kaggle kernels push -p kaggle/<stage> --accelerator NvidiaTeslaT4`, and download its output into `cache/`. In order: `kaggle/build_embed.sh`, `kaggle/build_transcribe.sh`, `kaggle/build_finetune.sh`, `kaggle/build_ft_alm.sh`, `kaggle/build_adapter.sh`.
5. Run `python src/features.py`.
6. Build the teacher, make the pseudo-labels, and train the pseudo-labelled adapter:
   - `python src/final.py --exp e011 --no-judge --nz --drop adapter2_whisper,adapter3_whisper,adapter4mean_whisper,adapter4meanvar_whisper`
   - `python src/pseudo.py`
   - `kaggle/build_adapter3.sh`, then push it and download the output as in step 4.
7. Build the two final submissions:
   - `python src/final.py --exp e013 --no-judge --nz --drop adapter_whisper,adapter2_whisper,adapter4mean_whisper,adapter4meanvar_whisper`
   - `python src/final.py --exp e009 --no-judge --drop adapter_whisper,adapter_wavlm,adapter2_whisper,adapter3_whisper,adapter4mean_whisper,adapter4meanvar_whisper`

   Both give back the submitted files exactly from the cached model outputs.
8. Run `notebooks/shl_grammar_scoring.ipynb`. It only reads the cached results.

## Models and licences
Used: Whisper-large-v3 (MIT), WavLM-large (MIT), Qwen2-Audio-7B-Instruct (Apache-2.0), faster-whisper (MIT), GPT-2 (MIT), spaCy (MIT), LanguageTool (LGPL-2.1), PEFT (Apache-2.0), scikit-learn (BSD), PyTorch (BSD). Also tried: w2v-BERT 2.0 (MIT) and LightGBM (MIT). No external labelled data was used.
