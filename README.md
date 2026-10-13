# Spoken Grammar Scoring Engine (SHL Hiring Assessment 2026)

This project predicts a grammar score from 0 to 5 for a 45-60 second recording of spoken English, the same score a human rater would give using SHL's rubric.

Author: Anuj Dev Singh

## The key finding
The same speakers appear in many training clips, but every test speaker is new. With ordinary random cross-validation, one speaker's clips end up in both training and validation, so models can partly score a clip by recognising the voice. That made CV look better than it really was and pushed me towards models that memorise voices. I measured it with speaker-verification embeddings (ECAPA-TDNN): within one voice the score barely changes, and my random folds split 187 of 208 repeated voices. Re-scoring the frozen audio model on speaker-grouped folds moved its RMSE from 0.529 to 0.622.

From then on I chose every model, every setting and the stack weights on speaker-grouped folds, where each voice is either in training or in validation. The first engine built that way became my best public score (0.3507, up from 0.3655).

## Final engine
Five models each predict a score, and a non-negative linear stack combines them:

1. Audio Ridge: frozen Whisper-large-v3 (layers 29-32) and WavLM-large (layers 17-24), mean and standard deviation over time.
2. DeBERTa-v3-large fine-tuned on the Whisper transcripts (4 fixed epochs, 3 seeds).
3. Frozen Qwen2.5-7B-Instruct reading the transcript, with Ridge on its hidden states.
4. Adapter grader: small per-layer adapters and one Transformer layer on frozen Whisper frames.
5. Frozen DeBERTa-v3 / RoBERTa transcript embeddings plus 42 fluency and grammar features (weight 0 in the final stack).

The models train on the 732 clips scored 1-5, because the 37 clips scored 0 are invalid recordings and the test set has none. Ridge regularisation is chosen with an inner speaker-grouped CV. Test clips that share a voice are pulled halfway towards their voice's mean prediction.

## Results
Speaker-grouped CV on the 1-5 clips. "Score" is the leaderboard formula 0.6 x RMSE + 0.4 x (1 - Pearson); lower is better.

| Model | CV RMSE | CV Pearson | Score |
|---|---|---|---|
| Mean baseline | 1.014 | 0 | 1.008 |
| Audio Ridge | 0.579 | 0.821 | 0.419 |
| Fine-tuned DeBERTa | 0.577 | 0.823 | 0.417 |
| Qwen-7B transcript states | 0.603 | 0.805 | 0.440 |
| Adapter grader | 0.592 | 0.814 | 0.430 |
| **e016 stack** (models 1, 3, 5) | 0.525 | 0.856 | 0.373 |
| **e019 stack** (all five) | **0.517** | **0.861** | **0.366** |

Training RMSE (in-sample, 1-5 clips): e016 0.367, e019 0.447. The fine-tuned DeBERTa and the adapter have no all-data model, so their out-of-fold predictions stand in there, which is why e019's in-sample number is closer to its CV.

| Kaggle submission | Folds used to choose it | Public LB |
|---|---|---|
| Mean baseline | - | 1.0050 |
| Frozen Whisper + WavLM + Ridge | random | 0.4165 |
| + fine-tuned WavLM | random | 0.3985 |
| + transcript features | random | 0.3865 |
| + fine-tuned audio LLM (Qwen2-Audio) | random | 0.3655 |
| + adapter grader (and two variants) | random | 0.3742-0.3790 |
| **e016**: speaker-grouped engine | grouped | **0.3507** |
| **e019**: e016 + fine-tuned DeBERTa + adapter | grouped | 0.3532 |

My two final submissions are e019 (better on all 5 grouped folds) and e016 (best public score). The 0.0025 gap between them on the public leaderboard comes from about 70 clips and is within noise.

Tried and dropped on grouped CV: fine-tuned WavLM, length-matched crop augmentation, a second literal transcript (wav2vec2 CTC), and Qwen2.5-7B fine-tuned on transcripts. None improved the stack. `docs/decisions.md` has the numbers for every attempt.

## Repository layout
```
notebooks/   shl_grammar_scoring.ipynb: the report (approach, speaker finding, results, plots, what failed)
             shl_grammar_scoring.py: the same notebook as a plain-text (jupytext) source
src/         audit, folds, speaker embeddings and grouped CV, embeddings, transcription, features, text models,
             fine-tuning scripts, adapter grader, final stacks, submission check
kaggle/      scripts that build the Kaggle GPU jobs (one worker per T4 GPU)
docs/        method.md (research basis) and decisions.md (every choice with measured numbers)
experiments/ log.csv: every experiment with CV and leaderboard scores
```
The competition audio, labels, transcripts and anything derived from them are not in this repo, because the competition data is for competition use only.

## Reproduce
1. `pip install -r requirements.txt` (Python 3.11). LanguageTool needs Java.
2. Download the data with the Kaggle CLI into `data/raw/`.
3. `python src/audit.py`, `python src/cv.py`, `python src/speaker.py`, `python src/speaker_check.py`.
4. GPU stages on Kaggle (T4 x2): run each builder, push it with `kaggle kernels push -p kaggle/<stage> --accelerator NvidiaTeslaT4`, and download its output into `cache/`. Builders: `build_embed.sh`, `build_transcribe.sh`, `build_llm_text.sh`, `build_adapter5.sh`, `build_ft_text.sh`.
5. `python src/features.py` and `python src/text_emb.py`.
6. `python src/final_grouped.py --exp e016` and `python src/final_grouped.py --exp e019 --fold_models cache/ft_text_run/ft_text,cache/adapter5_run/adapter5_whisper`. Both rebuild the submitted files exactly.
7. Run `notebooks/shl_grammar_scoring.ipynb`. It only reads cached results.

## Models and licences
Whisper-large-v3 (MIT), WavLM-large (MIT), DeBERTa-v3-large (MIT), RoBERTa-large (MIT), Qwen2.5-7B-Instruct (Apache-2.0), ECAPA-TDNN / SpeechBrain (Apache-2.0), faster-whisper (MIT), GPT-2 (MIT), spaCy (MIT), LanguageTool (LGPL-2.1), PEFT (Apache-2.0), scikit-learn (BSD), PyTorch (BSD). Also tried: Qwen2-Audio-7B (Apache-2.0), Qwen3-4B (Apache-2.0), wav2vec2-large-960h (Apache-2.0), w2v-BERT 2.0 (MIT). No external labelled data was used.
