# M5 — Model selection: summary (numbers from results/*.csv, selection.json)
**Selected model: HistGradientBoosting (`hist_gb`, owner B).** Decided on the 80 TRAINING trips only (out-of-fold
predictions from M4's trip-grouped CV). Student-test was not used to decide.

## Step 1: primary metric (point-level Top-1, OOF), paired by trip
| Model | OOF Top-1 | Deficit vs best | 95% trip-bootstrap CI | Wilcoxon p | Verdict |
|---|---|---|---|---|---|
| hist_gb | 0.8517 | — | — | — | best |
| random_forest | 0.8511 | 0.0006 | [−0.0020, 0.0037] | 0.98 | tied |
| mlp | 0.8496 | 0.0021 | [−0.0007, 0.0050] | 0.087 | tied |
| logreg | 0.8433 | 0.0085 | [0.0040, 0.0127] | <0.001 | significantly worse |
| extra_trees | 0.8427 | 0.0090 | [0.0062, 0.0125] | <0.001 | significantly worse |
| knn | 0.8334 | 0.0183 | [0.0142, 0.0234] | <0.001 | significantly worse |
| gaussian_nb | 0.8174 | 0.0343 | [0.0249, 0.0469] | <0.001 | significantly worse |
→ Tied set: hist_gb, random_forest, mlp (accuracy alone can't separate them).

## Step 2: secondary criteria that matter downstream (tied set)
| Criterion (why it matters) | hist_gb | random_forest | mlp |
|---|---|---|---|
| OOF log loss (HMM emissions in M7) | 0.1254 | **0.1239** | 0.1313 |
| Calibration error ECE (probabilities must mean what they say) | **0.0049** | 0.0055 | 0.0086 |
| Fold std of Top-1 (stability) | 0.0194 | 0.0184 | **0.0175** |
| Inference / one median trip (API on Laptop 1) | 19 ms | 99 ms | **4 ms** |
| 5-fold CV cost → Bayesian tuning trials per hour (M6) | **42 s → ~87/h** | 744 s → ~5/h | 170 s → ~21/h |
| Model size | 0.4 MB | 154 MB | 0.1 MB |
Strict count of wins: hist_gb 2, mlp 2, rf 1 → tie broken by log loss → hist_gb.
Tolerant count (within 5% of best = shared win): hist_gb 3, rf 2, mlp 2 → hist_gb. **Same decision under both rules.**

## Why hist_gb (technical)
- Joint-best accuracy (statistically tied with RF/MLP; best point estimate).
- Best-calibrated and near-best log loss → its probabilities can be used directly as HMM emission probabilities in M7.
- Only model in the tied set that makes Bayesian optimisation practical: ~87 five-fold evaluations per hour vs ~5 for RF.
- Small (0.4 MB) and fast enough for the API (19 ms per trip); RF is 154 MB and 5× slower.
- Overfitting under control: train-fold vs validation Top-1 gap +0.036 with early stopping (RF +0.061).
- Weakness (shared by all models): stopped/slow points (OOF Top-1 0.49 vs 0.94 moving); left to the HMM.

## In plain words
Three "judges" scored about the same, so we checked what each one needs for the next steps. Gradient boosting gives
the most trustworthy confidence numbers (the HMM needs those), it's small and quick, and it's the only one fast enough to
tune properly in the time we have. The random forest is as accurate but about 360× bigger and 18× slower to tune. The neural
net is fastest to run but its confidence numbers are less reliable.

## Caveat
LightGBM (C) has not been run yet (Colab only). After nisarg-ravi runs 03_models_C on Colab, re-run this notebook. If
LightGBM enters the tied set and wins the criteria, the selection must be revisited before M6.
