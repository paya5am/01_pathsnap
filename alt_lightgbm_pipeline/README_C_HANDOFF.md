# Member C (nisarg-ravi): M4 LightGBM, M6 Bayesian HPO, M7 HMM/Viterbi, M8 export

## What's in this bundle
| Path | What | Goes to repo as |
|---|---|---|
| `notebooks/05_bayes_opt.ipynb` | M4 + M6 notebook (clean, no outputs) | `notebooks/05_bayes_opt.ipynb`, after YOU run it on Colab |
| `notebooks/06_hmm_viterbi.ipynb` | M7 + M8 notebook (clean) | `notebooks/06_hmm_viterbi.ipynb`, after YOU run it on Colab |
| `notebooks_executed_reference/` | The same notebooks as I ran them, with outputs | Not committed. Use them only to compare against your Colab run |
| `pathsnap/hmm.py`, `cv.py`, `metrics.py`, `inference.py` | Your modules | `pathsnap/` |
| `pathsnap/io.py`, `network.py`, `candidates.py`, `pipeline.py` | Shared modules the notebooks need | `pathsnap/`. Agree with A who commits them |
| `pathsnap/labels.py`, `features.py` | **Stand-ins** for A's and B's M3 work | Replace with A's and B's versions, then re-run both notebooks |
| `scripts/build_dataset.py`, `train_lgbm.py`, `run_hmm.py`, `export_artifacts.py` | The code behind every number | `scripts/` |
| `experiments/results/`, `artifacts/`, `reports/` | Outputs from my run | Regenerate them on Colab and commit your own run |
| `docs/C_M2_prep_decoder_edge_cases.md` | Step 1 notes | Not committed (viva prep) |

## Running on Colab
1. Put `Student folder.zip` at `MyDrive/PathSnap/`.
2. Push this code to `paya5am/01_pathsnap` from your account. The notebooks clone that repo.
3. Run `05_bayes_opt.ipynb` (about 45 min, mostly Optuna), then `06_hmm_viterbi.ipynb` (about 15 min).
4. To run locally instead, keep `data/train` and `data/student_test` in the repo folder. The notebooks skip any step whose output already exists, so delete `experiments/` to force a full re-run.

## Results from my run (seed 42)
LightGBM, labelled points only, GroupKFold(5) by trip with trips 10 and 16 kept together:

| | CV log loss | CV Top-1 | Student-test Top-1 | Student-test F1 |
|---|---|---|---|---|
| Untuned | 0.1153 ± 0.010 | 0.879 | 0.883 | 0.877 |
| Tuned (Optuna TPE, 40 trials) | 0.1135 ± 0.011 | 0.880 | 0.888 | 0.880 |

HMM decoder settings were tuned on train out-of-fold probabilities only. Best: β = 40 m, unreachable penalty = 2, gap = 60 s.

| Student-test (10 trips) | Top-1 | Route continuity | Sequence F1 (stitched) |
|---|---|---|---|
| Classifier only | 0.888 | 0.966 | 0.777 |
| Classifier + HMM | **0.908** | **0.994** | **0.884** |

The HMM improved 9 of the 10 student-test trips. Trip 7 dropped slightly (0.971 to 0.962).
Speed: the largest trip (15,783 points) takes 1.4 s end to end.

## Viva points
- **Why log loss is the HPO objective.** The HMM uses the probabilities as emissions, so they must be well calibrated. Top-1 is reported alongside.
- **Tuning gain is small.** It is within the variation between CV folds. Say so honestly. Better features (B's M3) matter more than hyperparameters.
- **Why β is large and the penalty small.** The reference and the graph have breaks: 10 trips have route arcs outside the largest connected component, and 11 trips have route breaks. A strict connectivity rule would then force the decoder onto wrong arcs. The grid shows both extremes do worse (penalty 25 gives 0.880, penalty 0.5 gives 0.891).
- **Two bugs I found and fixed during tuning.** (1) The path cache could return paths longer than the current search limit. (2) A whole-number setting made the transition table an integer array, which rounded every transition score. The final grid was re-run after both fixes, and the grid and final runs now agree exactly (0.8952).
- **Leakage guard.** `predict_trip_folder()` loads the trip with `with_route=False`. Notebook 06 deletes the `.route` file from a copy of a trip and predicts on it successfully.
