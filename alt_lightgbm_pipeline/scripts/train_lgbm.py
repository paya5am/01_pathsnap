"""M4 + M6 (Member C): LightGBM untuned baseline, Optuna Bayesian tuning with
trip-grouped CV on the 80 training trips, refit, student-test comparison.

usage: python scripts/train_lgbm.py <dataset_dir> <out_dir> [n_trials]
Training rows = candidates of LABELLED points only (decision D6).
"""
import json
import os
import sys
import time

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap.cv import SEED, trip_folds  # noqa: E402
from pathsnap.metrics import point_metrics, row_metrics  # noqa: E402

UNTUNED = dict(random_state=SEED, n_jobs=2, verbose=-1)   # LightGBM defaults


def load(path):
    d = np.load(path)
    m = d["point_labelled"]
    return {k: d[k][m] for k in ["X", "y", "trip", "point", "arc"]} | {"features": list(d["features"])}


def evaluate(model, D):
    t0 = time.time()
    p = model.predict_proba(D["X"])[:, 1]
    inf = time.time() - t0
    r = row_metrics(D["y"], p)
    r |= point_metrics(D["trip"], D["point"], D["y"], p, np.ones(len(p), bool))
    r["inference_s"] = inf
    r["inference_us_per_row"] = 1e6 * inf / len(p)
    return r, p


def cv(params, D, trial=None):
    folds, oof = [], np.zeros(len(D["y"]))
    for k, (tr, va) in enumerate(trip_folds(D["trip"])):
        t0 = time.time()
        m = lgb.LGBMClassifier(**params).fit(D["X"][tr], D["y"][tr])
        fit_s = time.time() - t0
        sub = {key: D[key][va] for key in ["X", "y", "trip", "point"]}
        r, p = evaluate(m, sub)
        oof[va] = p
        r["fit_s"] = fit_s; r["fold"] = k
        folds.append(r)
        if trial is not None:
            trial.report(np.mean([f["log_loss"] for f in folds]), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return folds, oof


def summarise(folds):
    keys = ["accuracy", "precision", "recall", "f1", "log_loss", "top1", "miss_rate", "fit_s"]
    return {f"cv_{k}_mean": float(np.mean([f[k] for f in folds])) for k in keys} | \
           {f"cv_{k}_std": float(np.std([f[k] for f in folds])) for k in keys}


def main(ds, out, n_trials=40):
    os.makedirs(out, exist_ok=True)
    TR = load(os.path.join(ds, "candidates_train.npz"))
    ST = load(os.path.join(ds, "candidates_student_test.npz"))
    print("train rows", len(TR["y"]), "pos rate", TR["y"].mean(), "student-test rows", len(ST["y"]))
    results = {}

    # ---------- untuned baseline ----------
    folds, oof_untuned = cv(UNTUNED, TR)
    t0 = time.time(); base = lgb.LGBMClassifier(**UNTUNED).fit(TR["X"], TR["y"]); fit_all = time.time() - t0
    st, _ = evaluate(base, ST)
    results["untuned"] = dict(params=UNTUNED, **summarise(folds), folds=folds,
                              student_test=st, fit_all_s=fit_all)
    print("untuned CV top1 %.4f logloss %.4f | ST top1 %.4f" %
          (results["untuned"]["cv_top1_mean"], results["untuned"]["cv_log_loss_mean"], st["top1"]))

    # ---------- Bayesian optimisation (TPE) ----------
    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 800, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 500, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0), subsample_freq=1,
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            random_state=SEED, n_jobs=2, verbose=-1)
        folds, _ = cv(params, TR, trial)
        s = summarise(folds)
        for k in ["cv_top1_mean", "cv_f1_mean", "cv_fit_s_mean", "cv_log_loss_std"]:
            trial.set_user_attr(k, s[k])
        return s["cv_log_loss_mean"]

    study = optuna.create_study(direction="minimize", study_name="lgbm_pathsnap",
                                sampler=optuna.samplers.TPESampler(seed=SEED),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    hpo_s = time.time() - t0
    study.trials_dataframe().to_csv(os.path.join(out, "optuna_trials.csv"), index=False)
    best = dict(study.best_params, subsample_freq=1, random_state=SEED, n_jobs=2, verbose=-1)

    # ---------- refit tuned model ----------
    folds, oof_tuned = cv(best, TR)
    t0 = time.time(); tuned = lgb.LGBMClassifier(**best).fit(TR["X"], TR["y"]); fit_all = time.time() - t0
    st, p_st = evaluate(tuned, ST)
    results["tuned"] = dict(params=best, **summarise(folds), folds=folds, student_test=st,
                            fit_all_s=fit_all, hpo_wall_s=hpo_s, n_trials=n_trials,
                            n_pruned=sum(t.state.name == "PRUNED" for t in study.trials))
    print("tuned   CV top1 %.4f logloss %.4f | ST top1 %.4f | HPO %.0fs" %
          (results["tuned"]["cv_top1_mean"], results["tuned"]["cv_log_loss_mean"], st["top1"], hpo_s))

    imp = pd.DataFrame({"feature": TR["features"], "gain": tuned.booster_.feature_importance("gain")})
    imp.sort_values("gain", ascending=False).to_csv(os.path.join(out, "feature_importance.csv"), index=False)
    joblib.dump(tuned, os.path.join(out, "lgbm_tuned.joblib"))
    joblib.dump(base, os.path.join(out, "lgbm_untuned.joblib"))
    np.savez_compressed(os.path.join(out, "oof_probs_train.npz"), tuned=oof_tuned, untuned=oof_untuned,
                        trip=TR["trip"], point=TR["point"], arc=TR["arc"], y=TR["y"])
    with open(os.path.join(out, "lgbm_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 40)
