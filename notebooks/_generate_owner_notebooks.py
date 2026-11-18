"""Generate the four owner notebooks (03_models_<owner>.py, cell format) from one template, so every model is run
through the same harness. Converted to .ipynb with to_ipynb.py."""
SETUP = r'''# %%
# ---- Setup (Colab or local) ----------------------------------------------------------------------
import os, sys, json, zipfile, time
import numpy as np, pandas as pd, matplotlib.pyplot as plt
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE = "/content/drive/MyDrive/PathSnap"
    ZIP_PATH, EXTRACT_DIR = f"{DRIVE}/Student folder.zip", "/content/pathsnap_data"
    DS_DIR = f"{DRIVE}/m3_outputs/dataset"          # written by notebook 02
    OUT_DIR = f"{DRIVE}/m4_outputs"
    REPO = "/content/01_pathsnap"
    if not os.path.isdir(REPO):
        os.system(f"git clone -q https://github.com/paya5am/01_pathsnap {REPO}")
    PKG_PARENT = REPO if os.path.isdir(f"{REPO}/pathsnap") else DRIVE
else:
    ZIP_PATH = os.environ.get("PATHSNAP_ZIP", "Student folder.zip")
    EXTRACT_DIR = os.environ.get("PATHSNAP_EXTRACT", "pathsnap_data")
    DS_DIR = os.environ.get("PATHSNAP_DS", "m3_outputs/dataset")
    OUT_DIR = os.environ.get("PATHSNAP_M4", "m4_outputs")
    PKG_PARENT = os.environ.get("PATHSNAP_PKG", ".")
sys.path.insert(0, PKG_PARENT)
if not os.path.isdir(EXTRACT_DIR):
    with zipfile.ZipFile(ZIP_PATH) as z: z.extractall(EXTRACT_DIR)
DATA_ROOT = next(dp for dp, dn, _ in os.walk(EXTRACT_DIR) if "train" in dn and "student_test" in dn)
from pathsnap.modeling import load_dataset, arc_tables, run_experiment, candidate_miss_rate, reference_continuity
from pathsnap.model_zoo import ZOO, history_of
from pathsnap.features import FEATURES
os.makedirs(OUT_DIR, exist_ok=True)
try:
    display
except NameError:
    display = print
pd.set_option("display.width", 200)
F, P = load_dataset(DS_DIR)
ARCS = arc_tables(DATA_ROOT, [tuple(x) for x in F[["partition", "trip_id"]].drop_duplicates().itertuples(index=False)])
print(f"{len(F):,} candidate rows | train trips {F[F.partition=='train'].trip_id.nunique()} | student-test trips {F[F.partition=='student_test'].trip_id.nunique()}")
print("labelled train rows:", int(((F.partition == 'train') & (F.label >= 0)).sum()), "| features:", FEATURES)
print("candidate miss rate (same for every model): train %.4f, student-test %.4f" % (candidate_miss_rate(P, "train"), candidate_miss_rate(P, "student_test")))
'''
PROTOCOL = r'''# %% [markdown]
# ## Experimental protocol (identical for every model — `pathsnap/modeling.py`)
# * **Data:** M3 dataset; one row per (GPS point, candidate arc); X = the 16 features, y = `label` (rows of unlabelled
#   points, label = −1, are never used for fitting or scoring).
# * **CV:** 5-fold `GroupKFold` on the 80 training trips, groups = `cv_group` (= trip; trips 10 & 16 merged, D12).
#   No trip is ever split across folds. Any scaler is inside the pipeline → fitted on the training folds only.
# * **Final fit:** all 80 training trips → evaluated once on the 10 student-test trips (development comparison only).
# * **Metrics:** row-level accuracy / precision / recall / F1 / **macro F1** (threshold 0.5) / confusion matrix /
#   **log loss**; point-level **Top-1 accuracy** (pick the highest-probability candidate per GPS point; denominator =
#   labelled points, so a candidate miss counts as wrong); **route continuity** (consecutive points whose predicted arcs
#   are the same or directly connected); sequence-level **route-arc F1** (set of predicted vs reference arcs per trip);
#   training time; inference time.
# * **Overfitting check:** each fold also reports Top-1 / macro F1 / log loss on (a sample of) its own training rows.
# * **Seed:** 42 everywhere.
'''
SHOW = r'''
def show(name):
    r = json.load(open(f"{OUT_DIR}/{name}_results.json")); folds = pd.read_csv(f"{OUT_DIR}/{name}_folds.csv")
    display(folds[["fold", "val_top1_accuracy", "val_row_macro_f1", "val_log_loss", "val_route_continuity",
                   "train_top1", "train_macro_f1", "train_log_loss", "fit_s"]].round(4))
    st = r["student_test"]
    print(f"CV Top-1 {r['cv']['val_top1_accuracy']['mean']:.4f} ± {r['cv']['val_top1_accuracy']['std']:.4f} | "
          f"CV macro-F1 {r['cv']['val_row_macro_f1']['mean']:.4f} | CV log loss {r['cv']['val_log_loss']['mean']:.4f}")
    print(f"Student-test: Top-1 {st['top1_accuracy']:.4f} | macro-F1 {st['row_macro_f1']:.4f} | log loss {st['log_loss']:.4f} | "
          f"precision {st['row_precision']:.4f} | recall {st['row_recall']:.4f} | continuity {st['route_continuity']:.4f} | "
          f"route-arc F1 {st['route_arc_f1']:.4f}")
    print(f"Train time {r['train_time_s']:.1f}s | inference {r['inference_ms_per_1000_points']:.1f} ms per 1000 GPS points")
    print(f"Overfitting check — train-fold Top-1 {r['cv']['train_top1']['mean']:.4f} vs validation {r['cv']['val_top1_accuracy']['mean']:.4f} "
          f"(gap {r['cv']['train_top1']['mean'] - r['cv']['val_top1_accuracy']['mean']:+.4f}); log loss train {r['cv']['train_log_loss']['mean']:.4f} vs val {r['cv']['val_log_loss']['mean']:.4f}")
    ncol = 2 if r.get("history") else 1
    fig, ax = plt.subplots(1, ncol, figsize=(5.5 * ncol, 4)); ax = np.atleast_1d(ax)
    cm = np.array(st["confusion_matrix"]); ax[0].imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2): ax[0].text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="k")
    ax[0].set(xticks=[0, 1], yticks=[0, 1], xticklabels=["pred wrong", "pred correct"], yticklabels=["wrong arc", "true arc"],
              title=f"{name}: student-test confusion matrix (rows)")
    if r.get("history"):
        h = r["history"]; a2 = None
        for k, v in h.items():
            if "accuracy" in k:
                a2 = a2 or ax[1].twinx(); a2.plot(v, color="tab:orange", label=k); a2.set_ylabel(k); a2.legend(loc="upper right")
            else:
                ax[1].plot(v, label=k)
        ax[1].set(xlabel="epoch / iteration", title=f"{name}: recorded training history"); ax[1].legend(loc="center right")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/{name}_diagnostics.png", dpi=120); plt.show()
'''
OWNERS = {
 "A": dict(who="A — kumarsaavan07-jpg", models=["logreg", "gaussian_nb"], notes={
   "logreg": """### Model 1 — Logistic Regression (linear / probabilistic)
**Why consider it:** the strongest features (relative distance, heading agreement, distance from the arc end) are close
to monotone in the probability of being the true arc, so a linear decision boundary on scaled features may already
work; it outputs calibrated-ish probabilities natively, which the HMM needs as emissions.
**Expected limits:** cannot model interactions (e.g. "heading only matters when moving") unless engineered.
**Initial hyperparameters:** C = 1.0 (L2), lbfgs, max_iter 1000, StandardScaler. **Objective:** binary cross-entropy (log loss) + L2.""",
   "gaussian_nb": """### Model 2 — Gaussian Naive Bayes (probabilistic)
**Why consider it:** very fast baseline that turns per-feature class-conditional densities into probabilities.
**Expected limits:** assumes features are independent and Gaussian given the class; ours are skewed (distances),
discrete (rank, degrees) and strongly correlated (dist / dist_rel / win_dist_mean), so probabilities will be over-confident.
**Initial hyperparameters:** var_smoothing = 1e-9, empirical class priors. **Objective:** maximum likelihood of per-class Gaussians."""}),
 "B": dict(who="B — GeniePlunder", models=["random_forest", "hist_gb"], notes={
   "random_forest": """### Model 3 — Random Forest (ensemble / bagging)
**Why consider it:** captures non-linear interactions (heading × moving, distance × candidate density) without scaling;
averaging many decorrelated trees is stable across trips.
**Expected limits:** large model on ~1M rows, slower inference; leaf-frequency probabilities can be less sharp than boosting.
**Initial hyperparameters:** 150 trees, max_features = sqrt, min_samples_leaf = 20 (limits memorising ~740k rows), no depth limit.
**Objective:** Gini impurity per split.""",
   "hist_gb": """### Model 4 — Histogram Gradient Boosting (boosting)
**Why consider it:** sequential error correction typically gives the best accuracy on tabular data; histogram binning
makes it fast on ~1M rows; optimises log loss directly, which suits HMM emissions.
**Expected limits:** more hyperparameters; can overfit without early stopping.
**Initial hyperparameters:** learning_rate 0.1, max_iter 300 with early stopping (10% internal validation, patience 20),
max_leaf_nodes 31, min_samples_leaf 20. **Objective:** binary log loss. Note: the internal early-stopping split is by row,
inside the training folds only (never touches the CV validation trips)."""}),
 "C": dict(who="C — nisarg-ravi", models=["extra_trees", "lightgbm"], notes={
   "extra_trees": """### Model 5 — Extra Trees (randomised trees)
**Why consider it:** like a random forest but with random split thresholds → lower variance, faster training; a useful
contrast to see whether bagging-style ensembles benefit from extra randomisation here.
**Expected limits:** random thresholds can underfit fine distance boundaries.
**Initial hyperparameters:** 150 trees, max_features = sqrt, min_samples_leaf = 20, no bootstrap. **Objective:** Gini impurity.""",
   "lightgbm": """### Model 6 — LightGBM (boosting, leaf-wise)
**Why consider it:** leaf-wise gradient boosting, very fast on large tabular data, strong probability estimates.
Comparing it with HistGB (level-wise-ish, sklearn) tests whether the boosting implementation matters.
**Expected limits:** leaf-wise growth can overfit small groups without min_child_samples.
**Initial hyperparameters:** n_estimators 500, learning_rate 0.05, num_leaves 31, min_child_samples 20, subsample 0.8
(freq 1), colsample_bytree 0.8. **Objective:** binary log loss.
⚠️ Requires the `lightgbm` package (preinstalled on Colab)."""}),
 "D": dict(who="D — paya5am", models=["mlp", "knn"], notes={
   "mlp": """### Model 7 — Multi-layer Perceptron (neural network)
**Why consider it:** learns smooth non-linear combinations of the scaled features; produces probabilities via a sigmoid
output trained on log loss; gives an epoch-wise training history.
**Expected limits:** needs scaling and early stopping; less interpretable; sensitive to architecture.
**Initial hyperparameters:** hidden layers (64, 32), ReLU, Adam lr 1e-3, batch 1024, L2 alpha 1e-4, max 60 epochs,
early stopping on a 10% internal split (patience 8). **Objective:** binary cross-entropy + L2.""",
   "knn": """### Model 8 — k-Nearest Neighbours (instance-based)
**Why consider it:** "this (point, arc) situation looks like these labelled situations" — no training, local decisions.
**Expected limits:** prediction cost grows with the training set; needs scaling; all features weigh equally in the
distance. **Adaptation:** fitted on a seeded 150,000-row subsample of the training rows (whole GPS points) to keep
prediction time practical; probabilities = distance-weighted share of positive neighbours.
**Initial hyperparameters:** k = 50, distance weighting, Euclidean metric on StandardScaled features."""}),
}
LGBM = r'''# %%
# LightGBM is not part of model_zoo (it needs an extra package); defined here
try:
    import lightgbm as lgb
except ImportError:
    os.system("pip -q install lightgbm"); import lightgbm as lgb
LGBM_CONFIG = {"n_estimators": 500, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 20,
               "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8}
make_lgbm = lambda: lgb.LGBMClassifier(**LGBM_CONFIG, random_state=42, verbose=-1)
res, model = run_experiment("lightgbm", "Boosting (leaf-wise)", make_lgbm, F, P, ARCS, LGBM_CONFIG, OUT_DIR)
show("lightgbm")
'''
for o, d in OWNERS.items():
    cells = [f"""# %% [markdown]
# # PathSnap — Notebook 03_{o}: model experiments
# **Milestone 4 · Owner: {d['who']}** · Models: {', '.join(d['models'])}
#
# Partitions: 80 train trips (grouped CV + final fit) · 10 student-test trips (evaluation only). Evaluator trips: never used.
# Input: the M3 dataset written by notebook 02 (`MyDrive/PathSnap/m3_outputs/dataset/`). Results → `MyDrive/PathSnap/m4_outputs/`.
""", SETUP, PROTOCOL, "# %%" + SHOW]
    for mname in d["models"]:
        cells.append("# %% [markdown]\n" + "\n".join("# " + l for l in d["notes"][mname].splitlines()))
        if mname == "lightgbm":
            cells.append(LGBM)
        else:
            cells.append(f'''# %%
z = ZOO["{mname}"]; print("config:", z["config"])
res, model = run_experiment("{mname}", z["family"], z["make"], F, P, ARCS, z["config"], OUT_DIR,
                            train_subsample=z.get("train_subsample"), history_fn=lambda m: history_of("{mname}", m))
show("{mname}")
''')
    cells.append("""# %% [markdown]
# ## Your write-up (fill in YOUR OWN words from the numbers above — required for every model, Deliverables §3.2)
# 1. Why did you consider this algorithm suitable for PathSnap?  2. Strengths and limitations *for this task* (cite the numbers).
# 5. Important hyperparameters and initial values.  6. Loss / objective.  7–8. Student-test Top-1, macro F1, log loss.
# 9. Over- or under-fitting? Evidence: train-fold vs validation Top-1 / log loss gap, fold spread, training curves (if any).
""")
    open(f"03_models_{o}.py", "w").write("\n".join(cells))
print("written")
