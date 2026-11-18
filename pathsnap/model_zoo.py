"""The M4 candidate models: initial configurations (before any tuning). One entry per model.
Each owner's notebook trains its own two models through modeling.run_experiment.
Initial values are library defaults or common starting points, adjusted only for data size (see notes)."""
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from .config import SEED

ZOO = {
    # ---------------- A: kumarsaavan07-jpg ----------------
    "logreg": dict(owner="A", family="Linear / probabilistic",
                   config={"scaler": "StandardScaler", "C": 1.0, "penalty": "l2", "solver": "lbfgs", "max_iter": 1000},
                   make=lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=SEED))),
    "gaussian_nb": dict(owner="A", family="Linear / probabilistic",
                        config={"var_smoothing": 1e-9, "priors": "empirical"},
                        make=lambda: GaussianNB(var_smoothing=1e-9)),
    # ---------------- B: GeniePlunder ----------------
    "random_forest": dict(owner="B", family="Ensemble (bagging)",
                          config={"n_estimators": 150, "max_features": "sqrt", "min_samples_leaf": 20, "max_depth": None, "bootstrap": True},
                          make=lambda: RandomForestClassifier(n_estimators=150, max_features="sqrt", min_samples_leaf=20,
                                                              n_jobs=-1, random_state=SEED)),
    "hist_gb": dict(owner="B", family="Boosting",
                    config={"max_iter": 300, "learning_rate": 0.1, "max_leaf_nodes": 31, "min_samples_leaf": 20,
                            "early_stopping": True, "validation_fraction": 0.1, "n_iter_no_change": 20},
                    make=lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=31,
                                                                min_samples_leaf=20, early_stopping=True, validation_fraction=0.1,
                                                                n_iter_no_change=20, random_state=SEED)),
    # ---------------- C: nisarg-ravi ----------------
    "extra_trees": dict(owner="C", family="Tree-based (randomised)",
                        config={"n_estimators": 150, "max_features": "sqrt", "min_samples_leaf": 20, "bootstrap": False},
                        make=lambda: ExtraTreesClassifier(n_estimators=150, max_features="sqrt", min_samples_leaf=20,
                                                          n_jobs=-1, random_state=SEED)),
    # lightgbm is defined in the owner's notebook (needs the lightgbm package, preinstalled on Colab)
    # ---------------- D: paya5am ----------------
    "mlp": dict(owner="D", family="Neural network",
                config={"scaler": "StandardScaler", "hidden_layer_sizes": [64, 32], "activation": "relu", "alpha": 1e-4,
                        "batch_size": 1024, "learning_rate_init": 1e-3, "max_iter": 60, "early_stopping": True,
                        "validation_fraction": 0.1, "n_iter_no_change": 8},
                make=lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=1024,
                                                                           learning_rate_init=1e-3, max_iter=60, early_stopping=True,
                                                                           validation_fraction=0.1, n_iter_no_change=8,
                                                                           random_state=SEED))),
    "knn": dict(owner="D", family="Instance-based",
                config={"scaler": "StandardScaler", "n_neighbors": 50, "weights": "distance", "metric": "euclidean",
                        "train_subsample_rows": 150000},
                make=lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=50, weights="distance", n_jobs=-1)),
                train_subsample=150_000),
}


def history_of(name, model):
    """Epoch/iteration-wise training history, ONLY for models that produce one (no artificial curves)."""
    est = model[-1] if hasattr(model, "steps") else model
    if name == "mlp":
        return {"train_loss": list(map(float, est.loss_curve_)), "val_accuracy": list(map(float, est.validation_scores_))}
    if name == "hist_gb":
        return {"train_neg_log_loss": list(map(float, est.train_score_)), "val_neg_log_loss": list(map(float, est.validation_score_))}
    return None
