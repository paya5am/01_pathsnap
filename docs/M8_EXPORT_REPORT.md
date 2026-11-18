# M8 — Model export: report
Owner of the export: B (GeniePlunder, Laptop 1). Review of the alternative pipeline: C (nisarg-ravi), notebook 08.

## 1. What was decided before exporting (decisions D31, D32)
Member C delivered an independent LightGBM pipeline (alt_lightgbm_pipeline/: own stand-in labels and 15 features, Optuna-tuned
LightGBM, own HMM; reported student-test Top-1 0.888 classifier / 0.908 with HMM on its own labels). Those numbers use different
labels, so they were re-measured against the APPROVED M3 labels on the 80 training trips (out-of-fold), notebook 08:

| Classifier only, same 182,124 points, approved labels | OOF Top-1 |
|---|---|
| C LightGBM (untuned / tuned) | 0.8965 / 0.8957 |
| Deployed hist_gb_tuned_ll (16 features) | 0.8590 |
C better in 53/80 trips, Wilcoxon p = 1.5e-5. C's stand-in labels agree with the approved labels on 95.6% of shared points.

Ablation with the SAME model (hist_gb, M6 trial-3 params) and folds — which part explains the gap?
| Features | OOF Top-1 (all labelled points) | OOF log loss |
|---|---|---|
| Our 16 (approved) | 0.8521 | 0.1215 |
| Our 16 + C's 5 extra (proj_t, dist_prev3, dist_next3, head_diff_raw, head_diff×moving) | 0.8832 | 0.1113 |
| C's 15 rebuilt in our pipeline | 0.8813 | 0.1164 |
→ The features explain most of the gap, not LightGBM.

After HMM decoding (system-level objective; same intersection of points; β tuned per source on train OOF):
| Emissions → our HMM | HMM Top-1 (pooled) | vs deployed per trip |
|---|---|---|
| Deployed (16 features) | 0.9314 | — |
| Our 16 + C's 5 | 0.9283 | better 54/80, worse 18/80 (mean +0.0009, p 0.006) |
| C LightGBM tuned | 0.9373 | better 32/80, worse 45/80 (p 0.77) |
Pooled and per-trip results point in opposite directions and none is a consistent improvement. The deployed pipeline is
also the only one verified end-to-end through the API with scikit-learn only (LightGBM would need a second feature pipeline
built on stand-in labels and an extra runtime package that could not be verified in the development sandbox).
**Decision: export hist_gb_tuned_ll on the 16 approved features.** C's finding is reported: +3.1 to +3.7 classifier points from
temporal-context features, absorbed by the HMM because they encode the same sequence information.

## 2. Exported artifacts (artifacts/)
| File | Content |
|---|---|
| model.joblib (1.5 MB) | HistGradientBoostingClassifier: learning_rate 0.0267, max_leaf_nodes 46, min_samples_leaf 38, l2 0.789, max_features 0.423, 600 trees, early stopping; fitted on all 928,517 labelled candidate rows of the 80 training trips |
| manifest.json | features (16, order checked by the API), params, pathsnap / scikit-learn / Python versions, training-row count + SHA-1 fingerprint (cb33df31…) |
| hmm_config.json | decoder β = 3 m, break_s = 300 s, γ = 0 (+ the M7 evidence) |
Export script: scripts/export_model.py (--data-root rebuilds the training rows from raw trips; identical rows/fingerprint to the
M3 dataset). Run it ON LAPTOP 1 so the scikit-learn version matches (joblib files are version-sensitive).

## 3. Verification
Exported model + API service on the 10 student-test trips: HMM Top-1 0.9344, classifier-only 0.8668 — identical to notebook 07.
