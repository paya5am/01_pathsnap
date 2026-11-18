# artifacts/ — what the API loads (the M8 export contract)
| File | Required | Produced by | Content |
|---|---|---|---|
| `model.joblib` | yes (else PLACEHOLDER) | `scripts/export_model.py` | fitted HistGradientBoostingClassifier (hist_gb_tuned_ll, D24), trained on all 80 training trips |
| `manifest.json` | yes | `scripts/export_model.py` | model name, params, the 16 FEATURES (must equal `pathsnap.features.FEATURES`), pathsnap / scikit-learn / Python versions, training-row count + fingerprint |
| `hmm_config.json` | yes (else default decoder) | notebook 07 (copied by the export script) | decoder β, break_s, γ (+ the M7 evidence) |

Behaviour of the API:
- `model.joblib` missing → PLACEHOLDER scorer (rule baseline), flagged in /health, every response, the UI (red banner) and the logs.
- `hmm_config.json` missing → M7 defaults β = 3, break_s = 300, γ = 0, flagged "defaults".
- Feature list mismatch → the API refuses to start. scikit-learn version mismatch → loud warning (install the version in the manifest).
- Demo day: start with `PATHSNAP_ALLOW_PLACEHOLDER=0`, so the API refuses to start if the real model is missing.

Recommended: run the export ON LAPTOP 1 itself (~2 min), so the scikit-learn version matches automatically:
```
python scripts/export_model.py --data-root "<...>\Student folder\Student folder" --params scripts\inputs\best_params.json --hmm-config scripts\inputs\hmm_config.json --out artifacts
```
Verified in development: exported model + API reproduce the M7 student-test result exactly (HMM 0.9344, classifier-only 0.8668).
