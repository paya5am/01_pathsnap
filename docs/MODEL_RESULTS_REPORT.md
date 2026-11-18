# PathSnap — Models tested, their results, and why they differ
Every number comes from committed notebooks: 03_models_A–D and 04 (comparison), 05 (selection), 06 (tuning), 07 (HMM),
08 (M8 review). Protocol for all models: the same 16 features, 5-fold GroupKFold by trip on the 80 training trips (trips 10+16
grouped, seed 42), refit on all 80 trips, evaluated once on the 10 student-test trips. "Top-1" = share of labelled GPS points
whose highest-probability candidate is the reference arc (a candidate miss counts as wrong).
Baselines to beat (student-test Top-1): nearest arc 0.533 · nearest arc pointing the car's way 0.835.

## 1. Classifier results (M4)
| Model (owner) | Family | CV Top-1 (± fold std) | CV macro F1 | CV log loss | Calibration error (ECE, OOF) | Student-test Top-1 | Test macro F1 | Test log loss | Train-fold Top-1 (gap) | Inference ms / 1000 points |
|---|---|---|---|---|---|---|---|---|---|---|
| hist_gb (B) | Boosting | **0.8520 ± 0.019** | 0.9103 | 0.1252 | **0.0049** | 0.8617 | 0.9183 | 0.1112 | 0.8875 (+0.036) | 9.6 |
| random_forest (B) | Bagging | 0.8512 ± 0.018 | **0.9105** | **0.1237** | 0.0055 | **0.8676** | **0.9189** | **0.1107** | 0.9120 (+0.061) | 32 |
| mlp (D) | Neural net | 0.8498 ± **0.018** | 0.9063 | 0.1313 | 0.0086 | 0.8657 | 0.9167 | 0.1167 | 0.8712 (+0.022) | 3.6 |
| logreg (A) | Linear | 0.8438 ± 0.022 | 0.8887 | 0.1780 | 0.0188 | 0.8504 | 0.8947 | 0.1620 | 0.8443 (+0.001) | 0.4 |
| extra_trees (C) | Randomised trees | 0.8427 ± 0.020 | 0.9010 | 0.1345 | 0.0136 | 0.8592 | 0.9098 | 0.1224 | 0.8785 (+0.036) | 38 |
| knn (D) | Instance-based | 0.8330 ± 0.021 | 0.8860 | 0.1842 | 0.0098 | 0.8401 | 0.8931 | 0.1690 | 0.9924 (+0.159) | 1128 |
| gaussian_nb (A) | Probabilistic | 0.8138 ± 0.022 | 0.8055 | 0.6718 | 0.1382 | 0.8031 | 0.7984 | 0.6951 | 0.8180 (+0.004) | 0.9 |
| lightgbm (C), shared 16 features | Boosting (leaf-wise) | run on Colab (03_models_C) | | | | | | | | |

Statistical comparison (notebook 05, 80 training trips, paired by trip, 2,000 bootstrap resamples): hist_gb, random_forest and
mlp are **tied** (95% CIs of their deficit vs the best include 0; Wilcoxon p 0.98 and 0.087). logreg, extra_trees, knn and
gaussian_nb are **significantly worse** (p < 0.001).

## 2. Why each model did better or worse (connected to our evidence)
**What the task rewards.** Choosing the right arc needs *interactions*: "a small heading difference matters only when the car is
moving", "being 5 m further away matters when the nearest arc is 2 m away, not when it is 40 m away", "a candidate touching the
point only at its end is suspicious if the next fix is also near it". The distance features are also strongly correlated
(dist, dist_rel, dist_ratio, dist_rank, win_dist_mean), and the class is imbalanced (≈ 20% of rows are correct).

- **HistGradientBoosting (selected): best accuracy (tied) and best calibration.** Boosting adds trees that fix the previous
  trees' mistakes, so interaction rules are learned automatically; histogram binning handles 0.9 M rows in seconds. It optimises
  log loss directly, so its probabilities are the most honest (ECE 0.005), which matters because they become HMM emissions.
  Early stopping keeps the train/validation gap small (+0.036).
- **Random Forest: same accuracy, slightly better log loss, but heavy.** Averaging many deep trees captures the same interactions
  and is stable across folds. Fully grown trees (min 20 rows per leaf) partly memorise points (train-fold Top-1 0.912 vs 0.851
  validation: mild overfitting). The model is 154 MB, 5× slower per trip and ~18× slower to tune, which is why it lost on the
  secondary criteria (notebook 05), not on accuracy.
- **MLP: tied on accuracy, weaker probabilities.** Smooth non-linear combinations of scaled features work well (best fold
  stability); early stopping prevented overfitting (+0.022). Its probabilities are less calibrated (ECE 0.009) and sigmoid
  outputs are over-confident on easy rows.
- **Logistic Regression: underfits.** One linear boundary on the 16 features cannot express "heading matters only when moving"
  without hand-made interaction terms. Evidence: train and validation Top-1 are equal (0.844 vs 0.844), i.e. high bias, not
  overfitting. It is still the fastest model (0.4 ms / 1000 points) and beats the rule baseline.
- **Extra Trees: random thresholds blur fine distance boundaries.** Thresholds are drawn at random instead of optimised, which
  lowers variance but cannot place the split at, for example, "dist_rel < 2 m" precisely. Accuracy −0.9 points vs the forest.
- **KNN: memorises and suffers from distance problems.** With distance weighting, every training point predicts itself (train
  Top-1 0.99 vs 0.83 validation: severe overfitting). All 16 scaled features weigh equally in the Euclidean distance, including
  weak context features, and it was trained on a 150 k-row subsample to keep prediction time practical (still 1.1 s per 1000
  points).
- **Gaussian Naive Bayes: wrong assumptions.** It assumes features are independent and Gaussian within each class. Our distance
  features are strongly correlated (dist vs win_dist_mean r = 0.91, dist vs dist_rel 0.76) and skewed, so the same evidence is counted several times, which makes it
  over-confident (precision 0.57, recall 0.89, log loss 0.68, ECE 0.138). It is the only model below the 0.835 rule baseline.

**What limits all of them (error analysis, notebook 04).** The most common error is the **twin**: right road, wrong direction
(6.0–9.2% of points). 85–88% of twin errors (71% for naive Bayes) happen when the car is stopped or crawling (heading unknown; median speed 1.9–2.2 m/s),
which is only 14.6% of the points. Top-1 when moving is 0.88–0.94, when stopped 0.35–0.45 for every model. A single stopped GPS
point lies equally on both directions of a two-way road, so no per-point classifier can fix this: it needs the sequence (HMM).

## 3. Tuning (M6): Bayesian optimisation of hist_gb
| Configuration | CV Top-1 | CV log loss | CV fold std | Student-test Top-1 (reference only) |
|---|---|---|---|---|
| Untuned | 0.8520 | 0.1252 | 0.019 | 0.8617 |
| Best CV Top-1 (trial 25) | 0.8533 | 0.1258 | 0.020 | 0.8641 |
| Best log loss among Top-1-tied (trial 3; deployed) | 0.8516 | **0.1212** | **0.015** | 0.8668 |
30 trials of Gaussian-process Bayesian optimisation (150 fits, 65 min). The Top-1 gain is +0.0014, about 1/14 of the fold
spread: the defaults were already near-optimal (consistent with C's independent Optuna search on LightGBM). Tuning did produce a
better-calibrated, more stable configuration, which is what the HMM needs.

## 4. Classifier + HMM (M7) — the system result
| | Train (out-of-fold) Top-1 | Student-test Top-1 | Student-test route continuity (reachable) | Student-test route-arc F1 |
|---|---|---|---|---|
| Classifier only (deployed hist_gb) | 0.8521 | 0.8668 | 0.9820 | 0.9111 |
| **Classifier + HMM/Viterbi** | **0.9246** | **0.9344** | **0.9997** | **0.9666** |
Why the HMM helps so much: it resolves exactly the errors no classifier can. When stopped, Top-1 goes 0.43 → 0.73 (the car keeps
the direction it arrived in); twin errors halve (6.4% → 3.0%). Impossible jumps between unconnected roads are ruled out, so the
route becomes drivable end to end. All 10 student-test trips improve; 76 of 80 training trips improve (details: M7 report).

## 5. The alternative LightGBM pipeline (C) — why a better classifier did not give a better system (M8, notebook 08)
| Same 182,124 training points, approved labels | Classifier Top-1 | After our HMM (pooled) |
|---|---|---|
| Deployed hist_gb, 16 features | 0.8590 | 0.9314 |
| hist_gb, 16 + C's 5 features | — (all points: 0.8832) | 0.9283 |
| C's LightGBM, 15 features | 0.8957 | 0.9373 (better in 32/80 trips, worse in 45/80, p 0.77) |
C's features `dist_prev3`, `dist_next3` (where the fixes 3 s before/after lie relative to this arc) and `proj_t` (position along
the arc) give the classifier +3 points, because they let it see the neighbouring points. That is the same information the HMM
already uses, so after decoding the advantage disappears and the per-trip results are mixed. We kept the verified pipeline and
report the finding. Lesson: `proj_t` scored 0.508 on its own and was dropped in M3. Dropping features by single-feature AUC misses
features that only help in combination.

## 6. Summary in plain words
Eight "judges" were trained to pick the right road for each GPS dot. The tree-boosting, forest and small neural-network judges
were equally good (~85–87%); simpler ones were worse for clear reasons: a straight-line rule can't express "direction matters only
when moving", naive Bayes double-counts overlapping clues, KNN memorises. All of them fail mainly when the car is stopped, because
a parked dot sits on both directions of a street. Adding the HMM (which looks at the whole trip) fixes most of those cases:
93.4% correct on unseen trips and a drivable route.
