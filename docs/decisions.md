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

## D6. Transcript features (M3): measured, small gain
- **Transcripts:** faster-whisper large-v3 on all 985 clips. One clip hit a word-alignment bug and was recovered by the retry without the prompt.
- **Features:** 42 fluency and grammar features (pauses, speech rate, fillers, ASR confidence, vocabulary variety, GPT-2 likelihood, spaCy syntax, LanguageTool errors per 100 words).
  - Raw length counts (duration, word count, sentence count) are excluded, because test clips are shorter.
- **Alone:** CV RMSE 0.84 and Pearson 0.735, much weaker than the audio embeddings (0.524 / 0.906).
  - The strongest single features are ASR word confidence (Pearson 0.68 with the label) and vocabulary variety (0.37).
  - The direct grammar measures (LanguageTool rate, GPT-2 likelihood, syntax depth) are weaker.
  - Likely reason: Whisper cleans up speech, so many errors never reach the text.
- **Stacked with audio:** CV RMSE 0.5197 vs 0.5237. A gain of 0.004 is smaller than the fold-to-fold noise (0.007), so it is **not counted as an improvement** and was not submitted. The features stay in the report because they explain *what* the score depends on.

## D7. Newer encoder: w2v-BERT 2.0 (frozen)
- **Setup:** w2v-BERT 2.0 (Meta, MIT, pre-trained on 4.5M hours) embedded on Kaggle T4 x2 in 11.6 min.
- **Alone:** CV RMSE 0.592 / Pearson 0.878. That is clearly **weaker** than WavLM-large (0.538) and Whisper-large-v3 (0.554) when frozen. Being newer and trained on more data did not make it better for this task.
- **Combined:** added to Whisper + WavLM, it gives 0.520 vs 0.524. That is within noise, so it was not submitted.
- **Lesson:** frozen encoders have plateaued at about 0.52. The remaining options are fine-tuning (running) and a model that can judge grammar directly.

## D8. Fine-tuning WavLM-large (Stage G)
- **Setup:** WavLM-large trained on our scores. Random 20 s crops, 8 epochs (fixed, never picked on the validation fold), a small learning rate for the encoder and a larger one for the head. 5 folds on Kaggle T4 x2 took 81 min per fold and 4.1 h of wall time.
- **Alone:** CV RMSE 0.550 / Pearson 0.896 on repeat r0. That is *worse* than the frozen Whisper+WavLM Ridge on the same folds (0.521 / 0.907). With only 615 training clips per fold, end-to-end training does not beat a regularised linear model on frozen features. This matches the original plan's warning that fine-tuning is costly for a modest gain on small data.
- **Stacked:** the fine-tuned model makes different mistakes (test correlation with the frozen model 0.94), so blending helps. The stack gives **0.511 vs 0.521**:
  - it improves all 5 folds;
  - the bootstrap 95% interval of the gain is [0.0015, 0.017], so the gain is small but real;
  - a plain 50/50 average gives 0.513, so the gain does not depend on the fitted weights.
- **Limits:** only one repeat (r0) was run, because of GPU cost. No in-sample training RMSE exists for the fine-tuned part, because there is no model trained on all clips; the test prediction averages the 5 fold models.

## D9. Audio-LLM judge (Qwen2-Audio-7B): tried, dropped
- **Setup:** Qwen2-Audio-7B-Instruct (Apache-2.0), 4-bit, one copy per T4. It listened to each clip (30 s windows) and was asked to rate grammar with the exact rubric. We read the probabilities of "1"-"5" and took the expected score. Two prompt wordings. 42 min on T4 x2.
- **Result:** its scores correlate only **0.43** with the labels, much weaker than every trained branch. Added to the stack it changes CV RMSE by 0.0007 (better in 3/5 folds, bootstrap interval centred on 0), so it is **dropped**.
- **Why it is weak (likely):** a general audio model, not trained on our rubric, gives coarse ratings, and its probabilities cluster around a few values. Our trained models learn the raters' actual scale from 769 examples. The judge is in the report as "tried, no gain".

## D10. Final engine
- **Branches:** frozen Whisper-large-v3 + WavLM-large embeddings with Ridge (weight 0.57), fine-tuned WavLM-large (0.35), transcript fluency/grammar features with Ridge (0.14), combined by a non-negative linear stack. Predictions are clipped to [0, 5].
- **Numbers:** CV RMSE 0.500 and Pearson 0.915 on repeat r0. In-sample training RMSE 0.348 (the fine-tuned part uses its out-of-fold predictions).
- **Why these three:** each one was kept only because it improved CV in most folds. Everything else (SVR, w2v-BERT, label-0 detector, LLM judge) was measured and dropped.

## D11. Fine-tuning an audio LLM (Qwen2-Audio-7B, QLoRA): kept
- **Why:** zero-shot, the audio LLM was weak (D9). Published work shows the same model improves sharply once fine-tuned on proficiency scores (Pearson 0.64 -> 0.95, arXiv 2505.21148).
- **Setup:** the 7B model in 4-bit, with LoRA adapters (rank 16, 40M trainable parameters, 0.5% of the model) on the language model and a small regression head reading the last prompt token. Random 30 s windows, 3 fixed epochs, one fold per T4. About 2.5 h per fold.
- **Test first:** it ran on folds 0-1 only, with the rule set in advance: continue only if it improves the stack in both folds. It did (+0.010, +0.008), so the remaining 3 folds were run.
- **Alone:** CV RMSE 0.597 / Pearson 0.877. Weaker than the frozen (0.521) and fine-tuned WavLM (0.550) models. With only 615 clips per fold, the 7B model overfits after 2-3 epochs (train MSE 0.28 vs val about 0.35).
- **Stacked:** it makes different mistakes, so the stack improves from **0.5003 to 0.4916**. It is better in 5/5 folds, and the bootstrap 95% interval of the gain is [0.0018, 0.016], so the gain is real. Stack weights: frozen 0.46, fine-tuned WavLM 0.28, audio LLM 0.25, text 0.08.

## D12. Test set has no invalid clips; train for scores 1-5
- **Evidence:**
  - Our all-clips model predicts the 37 label-0 training clips at about 0.1, but its lowest test prediction is 1.82.
  - The leaderboard RMSE of the constant baseline (1.005) matches the spread of train labels *without* the zeros (std 1.01), not with them (1.24).
  - A rule "keep the score if the all-clips model says below 1.0" catches 37/37 label-0 train clips and 0/216 test clips.
- **Change:** the Ridge models, the adapters and the stack weights are trained on scores 1-5 only. CV is reported on 1-5 clips as the test-like number, and the rule stays as a safety net.

## D13. Adapter grader on frozen encoder frames (Speak & Improve 2025 winner's recipe): biggest single gain
- **Why:** the winning S&I 2025 system's best single model was a frozen Whisper encoder with small trainable per-frame adapters (RMSE 0.363 vs their BERT text grader 0.396). Our Ridge branch averages over time *before* learning anything, so short events such as a broken sentence are averaged away.
- **Setup:**
  - Whisper-large-v3 layers 16/22/27/32, frames averaged into 80 ms steps and cached once.
  - Per-layer adapters (LayerNorm, Linear 1280->128, GELU, Linear), concatenated, then one small Transformer layer, then mean + std over time, then an MLP. About 1M trainable parameters.
  - Random 20 s crops and masked spans as augmentation. 15 fixed epochs, 5 folds x 3 seeds averaged. 42 min in total on T4 x2.
- **Result:** Whisper adapter **0.503** on 1-5 clips, our best single model (frozen Ridge 0.529, fine-tuned WavLM about 0.55). The WavLM adapter scored 0.537 and the stack gives it weight 0.
- **Stack (e011):** 0.4857 vs e009 0.5011 on 1-5 clips. Better in 5/5 folds, bootstrap 95% interval of the gain [0.005, 0.025].

## D14. e011: better CV, worse public leaderboard. What we did about it
- **What happened:** e011 improved CV on 1-5 clips (0.5011 -> 0.4857, 5/5 folds), but its public leaderboard score is 0.3742 vs 0.3655 for e009. This is the first time CV and the leaderboard disagreed.
- **Checks:**
  - **Mean shift:** none (test mean 3.296 vs 3.310).
  - **Clip length:** on train clips under 50 s, which are like most test clips, e011 is still better (0.549 vs 0.562).
  - **Sampling noise:** if e011 is truly better, a random public subset of 65-108 test clips would make it look this much worse 3-8% of the time.
- **Reading:** most likely noise in a small public split, but a real test-set difference cannot be ruled out.
- **Decision:** do not tune on the public leaderboard (it uses roughly a third of 216 clips; CV uses 769). The 2 final submissions hedge both views: **e011** (best CV) and **e009** (best public score).

## D15. Short-clip-aware adapter (round 2): no real gain, not submitted
- **Hypothesis:** short clips (< 50 s; 69% of test) are harder (RMSE about 0.58 vs 0.47) because the adapter only saw fixed 20 s crops.
- **Change:** variable 8-40 s training crops, clip length as an input, averaging full-clip and 20 s window scores at prediction, and 5 seeds.
- **Result:**
  - Alone it is slightly better overall (0.498 vs 0.503), but all of that gain is on long clips.
  - Short clips are unchanged (0.586 vs 0.585). So short clips are not hard because of a length mismatch; they are probably harder recordings or speakers.
  - In the stack it replaces round 1, but the gain (0.0024, bootstrap interval includes 0) is within noise, and the test predictions correlate 0.999 with e011.
- **Decision:** keep e011 as the final engine. The round-2 code stays in the repo as a documented, measured attempt.

## D16. Classical tools for train/test shift: adversarial validation and importance weighting
- **Adversarial validation:** a logistic regression on the audio embeddings separates 1-5 train clips from test clips with **AUC 0.836**, so the test set is clearly different. Clip length explains only part of it (correlation -0.45 with test-likeness).
- **Re-scoring on test-like train clips:** on the 25% of train clips that look most like test, e011 still beats e009 (0.506 vs 0.525). The same holds on the top 50% (0.492 vs 0.511). So the CV preference for e011 survives a test-like view.
- **Importance weighting (p(test|x)/p(train|x), clipped):**
  - The effective sample size is only 49 of 732 clips, so very few train clips resemble test.
  - A stack fitted with these weights is worse on the test-like 25% (0.514 vs 0.506) and on all clips. Not adopted.
- **Simple average of the 4 strongest audio models:** worse than the fitted stack on all three views (0.494 / 0.543 / 0.521). Not adopted.
- **TabPFN (tabular foundation model):** versions 2.5 and later are non-commercial, so they are not allowed under the competition's licence rule. Not used.
- **Reading:** the e011-vs-e009 public-LB disagreement is not explained by anything our features can see. Most likely it is public-subset noise, or a shift outside the audio embeddings. Final selection stays e011 + e009 as a hedge.

## D17. Pseudo-labelling (semi-supervised self-training): small but real gain
- **Why:** the test recordings differ from train (adversarial AUC 0.84). Pseudo-labelling lets the model also learn from unlabelled clips that look like the ones it must score.
- **Leakage-free setup (src/pseudo.py):**
  - For fold k, the teacher is built only from models that never saw fold k's labels: the fold-k models, plus stack weights fitted on the other folds.
  - The teacher labels fold k's clips (treated as unlabelled) and the 216 test clips.
  - The student is the Whisper adapter with round-1 settings. It trains on the real labels plus these pseudo-labels at half weight and is scored on fold k's real labels. CV on fold k and prediction on test therefore follow the same recipe.
- **Result:**
  - The student alone is **better than its own teacher**: 0.481 vs the e011 stack's 0.486 on 1-5 clips, and better than the round-1 adapter (0.503) in all 5 folds.
  - In the stack (e013): **0.4792 vs 0.4857**. Better in 5/5 folds, bootstrap 95% interval of the gain [0.002, 0.011]. Also better on test-like clips (0.501 vs 0.506) and short clips (0.545 vs 0.549).
- **Caveat:** the gain is modest and the test predictions correlate 0.998 with e011, so the leaderboard can only move a little. The stack now relies mostly on the student (weight 0.77), which itself learned from all the other branches.

## D18. Is the leaderboard drop adapter-specific? Disagreement test and second review
- **Label-free test:** for each model, compare how far its predictions are from the other models' average, on test vs on train (OOF).
  - Fine-tuned WavLM (0.95) and audio-LLM (0.90) behave the same on test as on train.
  - The round-1 Whisper adapter drifts away from the others on test by **1.27x**, the WavLM adapter by 1.14x, and the pseudo-labelled adapter by 1.10x.
  - So the adapter reacts to something in the test recordings. This fits its worse public scores despite better CV.
- **Second review (Jev, full context):**
  - Adapter-specific cause, not just noise: 0.76.
  - A more robust adapter is plausible and worth one test: 0.75.
  - The disagreement ratio is a valid diagnostic: 0.73.
  - Stop probing the public leaderboard: 0.77.
  - The e009 + e013 hedge is sound: 0.62.
  - Best first fix: per-clip normalisation (0.55) over domain-adversarial training (0.41).
  - Chance the fix beats e009's public 0.3655: "unlikely" (1.47/4).
- **Round 4:** per-clip normalisation of the encoder frames (classic cepstral mean/variance normalisation) on top of the pseudo-labelled adapter, in two variants: mean only and mean + variance.
- **Pass rule, set in advance and judged without the leaderboard:** CV on 1-5 clips no worse than e013 (0.479), test-like CV no worse than e013 (0.501), and disagreement ratio clearly closer to 1.0 than 1.10. If it passes, it replaces e013 as the second final pick.
- **Round 4 result: failed, not submitted.**
  - Stack CV on 1-5 clips: mean + variance 0.4916, mean only 0.4937, vs e013 0.4792. Worse in 5 of 5 folds; the bootstrap 95% interval of the gain is fully below 0.
  - Test-like CV: 0.5099 and 0.5101 vs e013 0.5006. Worse on the clips most like the test set too.
  - Disagreement ratio (vs the average of fine-tuned WavLM and audio-LLM): 1.05 and 1.03 vs e013 1.07. Only slightly closer to 1.0.
  - Plain words: removing each recording's constant "colour" also removed useful signal, and it barely changed how differently the adapter treats the test set. So the colour was not the main cause of the drift.
  - Decision: stop work on the adapter. Final picks stay e009 (best public score) + e013 (best CV).
