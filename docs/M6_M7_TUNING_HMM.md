# M6 (Bayesian optimisation) + M7 (HMM / Viterbi) — summary
All numbers come from m6_results/ and m7_results/. Every decision was made on the 80 TRAINING trips; student-test was
used once for the final comparisons.

## About the LightGBM results reported by the team (Optuna) — superseded by the M8 review (docs/M8_EXPORT_REPORT.md)
Not directly comparable with ours: built on "temporary features" and a different training set (909,018 vs our 928,517
labelled rows) → different candidates/labels, so its 0.879 Top-1 can't be set against hist_gb's 0.852. The repo was
still empty, so we couldn't verify it. Decision: keep hist_gb (D21) and re-run LightGBM through 03_models_C on the shared
M3 dataset; the HMM takes any model's probabilities, so a later swap is cheap.
Its two conclusions agree with ours: (1) tuning gains are smaller than the fold noise; (2) log loss matters because the
HMM uses the probabilities. We adopted (2) when choosing the configuration that feeds the HMM.

## M6 — Bayesian hyperparameter optimisation of hist_gb (notebook 06)
- Method: Gaussian-process surrogate (Matérn-5/2, ARD) + Expected Improvement, 10 Sobol starts + 20 GP-EI trials = 30
  trials × 5 trip-grouped folds = 150 fits, 64.6 min. Implemented in pathsnap/bayesopt.py (scikit-learn/scipy only, so it
  runs identically on Colab; Optuna isn't preinstalled there and isn't available in Claude's sandbox). Stopped at 30 of the
  planned 40: no improvement after trial 8 until a marginal one at trial 25 (convergence plot); time budget.
- Search space: learning_rate 0.02–0.3 (log), max_leaf_nodes 15–255 (log), min_samples_leaf 10–500 (log),
  l2_regularization 1e-4–10 (log), max_features 0.4–1.0. Fixed: max_iter 600 + early stopping, max_bins 255.
- Objective: mean point-level Top-1 over the same 5 trip-grouped folds as M4 (log loss and macro F1 logged per trial).

| Configuration | CV Top-1 | CV std | CV log loss | CV macro F1 | Test Top-1* | Test log loss* | Inference ms/1000 pts |
|---|---|---|---|---|---|---|---|
| Untuned (M4) | 0.8520 | 0.0194 | 0.1252 | 0.9103 | 0.8617 | 0.1112 | 9.6 |
| Best CV Top-1 (trial 25: lr 0.244, 16 leaves, min_leaf 84, l2 1e-4, max_feat 0.72) | 0.8533 | 0.0203 | 0.1258 | 0.9099 | 0.8641 | 0.1159 | 8.0 |
| **Best log loss among Top-1-tied (trial 3: lr 0.027, 46 leaves, min_leaf 38, l2 0.79, max_feat 0.42)** | 0.8516 | **0.0150** | **0.1212** | **0.9123** | 0.8668 | 0.1089 | 49.5 |
*student-test: reference only.
- Honest conclusion: the best-Top-1 trial beats untuned by +0.0014 (2 of 5 folds), about 1/14 of the fold std, with slightly
  worse log loss → winner's curse; the defaults were already near-optimal. Tuning did produce a better-CALIBRATED, more
  stable configuration (trial 3: log loss −3%, fold std 0.015 vs 0.019).

## Choice of emission model (M7 §1b, training trips, out-of-fold)
| Candidate | OOF Top-1 classifier | OOF Top-1 after HMM | CV log loss |
|---|---|---|---|
| hist_gb_tuned (trial 25) | 0.8527 | 0.9250 | 0.1258 |
| hist_gb (untuned) | 0.8517 | 0.9243 | 0.1252 |
| hist_gb_tuned_ll (trial 3) | 0.8521 | 0.9241 | **0.1212** |
All three are within 0.001 after HMM → rule (fixed in the notebook beforehand): lowest log loss → **hist_gb_tuned_ll**.

## M7 — HMM + Viterbi (notebook 07, pathsnap/hmm.py)
- States: the candidate arcs of each GPS point (~5). Emission: log P from hist_gb_tuned_ll.
- Transition: −|d_route − d_gps| / β (Newson & Krumm 2009); d_route on the DIRECTED graph (Dijkstra) from the projection
  on arc a → end of a → shortest path → start of b → projection on b; same arc: forward distance, backward ≤ 15 m allowed
  (jitter); unreachable within 3·d_gps + 200 m = impossible.
- Gaps / breaks: chain cut if Δt > break_s or nothing reachable; segments decoded independently. Points with no candidates → no prediction.
- Decoder tuned on training trips (OOF): grid β ∈ {0.25…80} × break_s ∈ {30, 60, 120, 300} × γ ∈ {0, 0.5, 1} = 120 settings.
  Best β = 3 m, break_s = 300 s, γ = 0 (optimum inside the grid after extending it; the first grid had it on the edge).
  Plateau: β 2–5 within 0.0005.

| Student-test (10 trips) | Top-1 | Continuity (strict) | Continuity (reachable) | Route-arc F1 |
|---|---|---|---|---|
| Classifier only | 0.8668 | 0.9316 | 0.9820 | 0.9111 |
| **Classifier + HMM** | **0.9344** | **0.9830** | **0.9997** | **0.9666** |
| Change | +0.0675 | +0.0514 | +0.0177 | +0.0555 |
Training trips (OOF): 0.8521 → 0.9246. All 10 student-test trips improve; 76 of 80 training trips improve, 4 get worse.

Where it helps (student-test): Top-1 when stopped/slow 0.427 → 0.728; when moving 0.942 → 0.970; twin errors 6.4% → 3.0%.
1,631 points fixed, 339 broken.
Where it doesn't: training trips 10 (−5.6 pts, parked 45% of the time in dense Moscow streets: the decoder commits to a wrong
chain through jitter), 57 (−1.7), 5 (−0.4), 97 (−0.1): slow points (2–3 m/s) where the decoder holds one direction while
the reference switches (U-turn or label noise).
Limitation: when GPS drifts > 20 m beyond the nearest road (trip 43, divided motorway) the true arc isn't a candidate and
nothing is reachable → the chain is cut (136 cuts in trip 43, which still gains +3.5 pts). Improvement for later: skip
outlier points and bridge over them instead of cutting.
Self-checks passed: decoding never reads the route; every prediction is one of its point's candidates; segment/cut
statistics reported (student-test: 15 segments, 0 points without prediction).
Speed: decoding all 10 student-test trips takes 0.26 s (after ~0.5 s per trip to build its lattice) → suitable for the API.

## In plain words
Tuning: we let a smart search try 30 different settings for the model. It barely beat the default settings, which tells
us the defaults were already good. One setting gave more trustworthy confidence numbers, so we used that one.
HMM: instead of judging each GPS dot alone, the decoder looks at the whole trip and asks "which chain of road pieces could a
car really have driven, given the dots?". Jumps that a car can't make are ruled out, and a parked car keeps the direction it
arrived in. Correct road pieces go from 87% to 93% on unseen trips, and the route is now drivable from start to end.
