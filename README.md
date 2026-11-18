# PathSnap — distributed map-matching and route reconstruction
PES University UE24CS352A Machine Learning hackathon · team: kumarsaavan07-jpg (A), GeniePlunder (B), nisarg-ravi (C), paya5am (D)

PathSnap matches noisy GPS observations to a road network: for every GPS point it generates candidate road arcs, scores them
with a classifier (HistGradientBoosting, 16 engineered features), and decodes the most plausible drivable route with an HMM
(Viterbi; transitions from directed shortest-path distances). A FastAPI backend (Laptop 1) runs the full pipeline and computes
accuracy against server-side reference data; a Streamlit UI (Laptop 2) lets the evaluator pick a trip and see the result.

## Results (all reproducible from the notebooks below)
| Stage | Partition | Top-1 accuracy |
|---|---|---|
| Rule baseline: nearest arc pointing the car's way | student-test | 0.835 |
| Classifier only (hist_gb, tuned) | 5-fold trip-grouped CV (80 train trips) | 0.852 |
| Classifier only | student-test (10 trips) | 0.867 |
| **Classifier + HMM/Viterbi** | train, out-of-fold | **0.925** |
| **Classifier + HMM/Viterbi** | **student-test** | **0.934** (route continuity 0.9997) |

## Repository map
| Path | What | Owner |
|---|---|---|
| `pathsnap/` | shared package (io, geo, network, candidates, labels, track, features, modeling, model_zoo, bayesopt, hmm, pipeline) | A, B, C |
| `notebooks/01_eda` | dataset exploration, validity checks, candidate rule, Folium maps | A |
| `notebooks/02_preprocessing_features` | labels, candidates, features, dataset | A (+B) |
| `notebooks/03_models_A/B/C/D` | 8 models, identical protocol | A / B / C / D |
| `notebooks/04_model_comparison`, `05_model_selection` | comparison, error analysis, statistical selection | B |
| `notebooks/06_bayes_opt`, `07_hmm_viterbi`, `08_m8_lightgbm_feature_review` | Bayesian optimisation, HMM, M8 review | C |
| `alt_lightgbm_pipeline/` | C's independent LightGBM + Optuna + HMM pipeline (self-contained, own package) | C |
| `scripts/export_model.py`, `artifacts/` | M8 export + exported model | B |
| `api/` | FastAPI backend (Laptop 1) | B |
| `ui/` | Streamlit UI (Laptop 2) | D |
| `tests/`, `scripts/smoke_test_api.py` | integration tests (M11) | D |
| `docs/` | milestone summaries, feature explanation, M8 + M11 reports | all |
| `experiments/` | candidate-radius analysis + committed results of notebook runs | A, all |

## Run
- Notebooks: Google Colab. Put `Student folder.zip` in `MyDrive/PathSnap/` (shared team folder); open a notebook from GitHub
  in Colab → Runtime → Run all. Order: 01 → 02 → 03_A/B/C/D → 04 → 05 → 06 → 07 → 08.
- Two-laptop deployment: see `DEPLOYMENT.md` (Laptop 1: `python -m api.main`; Laptop 2: `streamlit run ui/app.py`).
- Data is never committed (`.gitignore`). Ground truth stays on Laptop 1.

## Disclosure
An AI assistant (Claude) helped write code, run experiments and draft documentation; every result in this repository is
produced by the committed code and was re-run and checked by the team member who owns it.
