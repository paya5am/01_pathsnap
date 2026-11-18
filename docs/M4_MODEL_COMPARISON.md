# M4 — Model training & comparison: summary (all numbers from results/*.json)
Protocol: 16 M3 features; 5-fold GroupKFold by trip on 80 training trips (trips 10+16 grouped); final fit on all training
trips; evaluated once on the 10 student-test trips. Seed 42. Candidate miss rate (same for every model): train 0.0077,
student-test 0.0030. Reference-label route continuity (ceiling, student-test): 0.9854.
Rule baseline to beat (student-test Top-1): "nearest arc pointing the car's way" = 0.835.

| Model (owner) | Family | CV Top-1 (mean ± std) | CV macro F1 | CV log loss | Test Top-1 | Test macro F1 | Test log loss | Test continuity | Test route-arc F1 | Train-fold Top-1 (gap) | Train s | Infer ms/1000 pts |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| logreg (A) | Linear | 0.8438 ± 0.0215 | 0.8887 | 0.1780 | 0.8504 | 0.8947 | 0.1620 | 0.9251 | 0.8854 | 0.8443 (+0.001) | 1.5 | 0.4 |
| gaussian_nb (A) | Probabilistic | 0.8138 ± 0.0218 | 0.8055 | 0.6718 | 0.8031 | 0.7984 | 0.6951 | 0.9054 | 0.8270 | 0.8180 (+0.004) | 0.2 | 0.9 |
| random_forest (B) | Bagging | 0.8512 ± 0.0184 | **0.9105** | **0.1237** | **0.8676** | **0.9189** | **0.1107** | 0.9273 | 0.9021 | 0.9120 (+0.061) | 186.3 | 32.4 |
| hist_gb (B) | Boosting | **0.8520 ± 0.0194** | 0.9103 | 0.1252 | 0.8617 | 0.9183 | 0.1112 | 0.9292 | **0.9077** | 0.8875 (+0.036) | 6.1 | 9.6 |
| extra_trees (C) | Randomised trees | 0.8427 ± 0.0196 | 0.9010 | 0.1345 | 0.8592 | 0.9098 | 0.1224 | 0.9177 | 0.8822 | 0.8785 (+0.036) | 48.0 | 38.0 |
| lightgbm (C) | Boosting (leaf-wise) | Colab only (see note) | | | | | | | | | | |
| mlp (D) | Neural net | 0.8498 ± **0.0175** | 0.9063 | 0.1313 | 0.8657 | 0.9167 | 0.1167 | **0.9337** | 0.9073 | 0.8712 (+0.022) | 41.2 | 3.6 |
| knn (D) | Instance-based | 0.8330 ± 0.0206 | 0.8860 | 0.1842 | 0.8401 | 0.8931 | 0.1690 | 0.9169 | 0.8697 | 0.9924 (+0.159) | 0.0 | 1127.7 |

NOTE lightgbm: the lightgbm package is not installable in Claude's sandbox. Its notebook (03_models_C) is ready and must be
run by nisarg-ravi on Colab; 04_model_comparison picks it up automatically. Until then 7 models are compared (≥ 5 required).

## Observations (evidence-based)
1. **Top three are statistically tied on CV Top-1:** hist_gb 0.8520, random_forest 0.8512, mlp 0.8498. The differences
   (≤ 0.0022) are ~10× smaller than the fold-to-fold std (0.018–0.019). Separating them needs other criteria (M5).
2. **Probability quality (matters for HMM emissions):** best CV/test log loss: random_forest (0.1237 / 0.1107) ≈ hist_gb
   (0.1252 / 0.1112) < mlp (0.1313 / 0.1167). GaussianNB is badly calibrated (0.67): its independence assumption fails on
   correlated distance features, so it over-predicts "correct" (precision 0.57, recall 0.89).
3. **Over/under-fitting:** logreg train ≈ validation (gap +0.001) at a lower level → high bias / underfits (linear boundary).
   KNN memorises its training points (train Top-1 0.99, train log loss ≈ 0) → overfits (gap +0.16). Random forest gap +0.061
   (fully grown trees, min_samples_leaf 20) → mild overfitting. hist_gb +0.036 and mlp +0.022 (both early-stopped) → small gaps.
4. **Cost:** inference per 1000 GPS points: logreg 0.4 ms, mlp 3.6 ms, hist_gb 9.6 ms, random_forest 32 ms, extra_trees 38 ms,
   knn 1128 ms. Training: hist_gb 6 s vs random_forest 186 s.
5. **Every model except Gaussian NB beats the 0.835 rule baseline** on student-test (KNN only barely: 0.840).
6. **Error analysis (04 notebook §3–4):** the largest error type for every model is the TWIN (right road, wrong direction,
   ~6–7% of points). 85–88% of twin errors happen when the car is stopped/slow (heading unknown; median 1.9 m/s), which is
   only 14.6% of points. Top-1 when moving ≈ 0.93–0.94; when stopped ≈ 0.35–0.45 for all models. A single stopped dot
   cannot reveal direction; the sequence can → the main job for the HMM in M7.
7. Route continuity of classifier-only predictions is 0.905–0.934 vs 0.985 for the reference labels → independent
   per-point choices produce disconnected jumps (expected; motivates M7).

## In plain words
We trained 8 different kinds of "judges" (7 here, LightGBM on Colab) to look at each GPS dot's shortlist of roads and
pick one. The best ones (gradient boosting, random forest, small neural network) pick the right road about 86–87% of the
time on trips they never saw, beating the simple "closest road pointing the right way" rule (83.5%).
They're nearly perfect when the car is moving (94%). They struggle when the car is stopped (40%), because a parked dot looks
the same on both directions of a street. Fixing that needs knowing where the car came from, which is what the HMM
step (Milestone 7) adds.
