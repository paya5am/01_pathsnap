"""M4 model-experiment harness: identical data split, metrics and timing for every model.

Primary selection metric = point-level Top-1 accuracy under trip-grouped 5-fold CV on the 80 training trips
(the quantity the map-matcher is judged on), with log loss as the tie-breaker because the selected model's
probabilities become HMM emission probabilities in M7. Student-test is used for development comparison only.
"""
import os, json, time
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, log_loss,
                             confusion_matrix)
from .features import FEATURES
from .config import SEED

KEYS = ["partition", "trip_id", "pt", "arc"]


# ----------------------------------------------------------------------------------------------- data
def load_dataset(ds_dir):
    def _load(name):
        for ext, fn in ((".parquet", pd.read_parquet), (".pkl", pd.read_pickle)):
            p = os.path.join(ds_dir, name + ext)
            if os.path.isfile(p):
                return fn(p)
        raise FileNotFoundError(f"{name} not found in {ds_dir} — run notebook 02 first")
    F = _load("candidates_features"); P = _load("points")
    F = F.sort_values(["partition", "trip_id", "pt", "dist_rank"], kind="mergesort").reset_index(drop=True)
    return F, P


def arc_tables(data_root, trips):
    """{(partition, trip_id): arcs array} — used only to check whether consecutive predicted arcs are connected."""
    from .io import trip_dir
    out = {}
    for part, tid in trips:
        base = os.path.join(trip_dir(data_root, part, tid), f"{tid:08d}")
        out[(part, tid)] = pd.read_csv(base + ".arcs", sep=r"\s+", header=None).values.astype(np.int64)
    return out


# ----------------------------------------------------------------------------------------------- metrics
def _point_argmax(df, proba):
    """For each (partition, trip, pt) return the row index with the highest probability."""
    key = df["trip_id"].values.astype(np.int64) * 1_000_000 + df["pt"].values
    # rows of one point are contiguous (sorted in load_dataset); block id per row
    starts = np.r_[0, np.where(np.diff(key) != 0)[0] + 1]
    blk = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(key)]))
    # sort by (block, -proba): the first row of each block is that point's argmax (ties → nearest, stable order)
    o = np.lexsort((-np.asarray(proba), blk))
    first = np.r_[True, blk[o][1:] != blk[o][:-1]]
    return o[first]


def point_predictions(df, proba):
    """One row per GPS point with the predicted arc and (if labelled) whether it is correct."""
    best = _point_argmax(df, proba)
    out = df.iloc[best][["partition", "trip_id", "pt", "arc", "label", "label_status"]].copy()
    out["proba"] = np.asarray(proba)[best]
    return out.rename(columns={"arc": "pred_arc", "label": "pred_is_ref"}).reset_index(drop=True)


def route_continuity(pp, arcs_by_trip):
    """Share of consecutive GPS points whose predicted arcs are the same arc or directly connected
    (end node of the first = start node of the second). Computed over ALL points (prediction needs no labels)."""
    ok = tot = 0
    for (part, tid), g in pp.groupby(["partition", "trip_id"]):
        g = g.sort_values("pt"); a = g.pred_arc.values; ar = arcs_by_trip[(part, tid)]
        consec = np.diff(g.pt.values) == 1
        same = a[1:] == a[:-1]; conn = ar[a[:-1], 1] == ar[a[1:], 0]
        ok += int(((same | conn) & consec).sum()); tot += int(consec.sum())
    return ok / max(tot, 1)


def route_arc_f1(pp, points, arcs_by_trip):
    """Sequence-level: per trip, compare the SET of predicted arcs with the set of reference arcs of labelled points.
    Returns mean precision, recall, F1 over trips."""
    rows = []
    for (part, tid), g in pp.groupby(["partition", "trip_id"]):
        ref = set(points[(points.partition == part) & (points.trip_id == tid) & (points.label_status == 0)].ref_arc)
        pred = set(g[g.label_status == 0].pred_arc)
        tp = len(ref & pred); p = tp / max(len(pred), 1); r = tp / max(len(ref), 1)
        rows.append((p, r, 2 * p * r / max(p + r, 1e-9)))
    a = np.array(rows)
    return {"route_arc_precision": a[:, 0].mean(), "route_arc_recall": a[:, 1].mean(), "route_arc_f1": a[:, 2].mean()}


def all_metrics(df, proba, points, arcs_by_trip, threshold=0.5):
    """df: candidate rows of the evaluated trips (all rows, labelled or not); proba: P(correct) per row."""
    proba = np.clip(np.asarray(proba, float), 1e-7, 1 - 1e-7)
    lab = df["label"].values >= 0
    y = df["label"].values[lab]; p = proba[lab]; yhat = (p >= threshold).astype(int)
    pp = point_predictions(df, proba)
    labelled_pts = pp[pp.label_status == 0]
    n_lab_points = int(((points.label_status == 0) & points.set_index(["partition", "trip_id"]).index.isin(
        pp.set_index(["partition", "trip_id"]).index.unique())).sum())
    m = {
        "row_accuracy": accuracy_score(y, yhat),
        "row_precision": precision_score(y, yhat, zero_division=0),
        "row_recall": recall_score(y, yhat, zero_division=0),
        "row_f1": f1_score(y, yhat, zero_division=0),
        "row_macro_f1": f1_score(y, yhat, average="macro", zero_division=0),
        "log_loss": log_loss(y, p, labels=[0, 1]),
        "top1_accuracy": labelled_pts.pred_is_ref.sum() / max(n_lab_points, 1),
        "route_continuity": route_continuity(pp, arcs_by_trip),
        **route_arc_f1(pp, points, arcs_by_trip),
        "confusion_matrix": confusion_matrix(y, yhat, labels=[0, 1]).tolist(),
        "n_rows": int(lab.sum()), "n_labelled_points": n_lab_points,
    }
    return m, pp


def candidate_miss_rate(points, partition):
    q = points[(points.partition == partition) & (points.label_status == 0)]
    return 1 - q.ref_in_cands.mean()


def reference_continuity(points, arcs_by_trip, partition):
    """Continuity of the reference labels themselves (ceiling for comparison)."""
    q = points[(points.partition == partition) & (points.label_status == 0)].rename(columns={"ref_arc": "pred_arc"})
    return route_continuity(q, arcs_by_trip)


# ----------------------------------------------------------------------------------------------- experiment
def run_experiment(name, family, make_model, F, P, arcs_by_trip, config, out_dir, n_splits=5,
                   train_subsample=None, history_fn=None, train_eval_max_rows=300_000):
    """Grouped CV on train trips → refit on all train → evaluate on student-test. Saves JSON + point predictions.
    make_model(): returns an unfitted sklearn-style estimator with predict_proba.
    train_subsample: optional max number of TRAIN rows used for fitting (e.g. KNN), sampled by whole points, seeded."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)
    TR = F[F.partition == "train"].reset_index(drop=True)
    TE = F[F.partition == "student_test"].reset_index(drop=True)
    gkf = GroupKFold(n_splits=n_splits)
    fold_rows, oof = [], np.zeros(len(TR))
    trip_groups = TR.cv_group.values

    def fit_rows(df):
        d = df[df.label >= 0]
        if train_subsample and len(d) > train_subsample:
            keys = d[["trip_id", "pt"]].drop_duplicates()
            frac = train_subsample / len(d)
            keep = keys.sample(frac=frac, random_state=SEED)
            d = d.merge(keep, on=["trip_id", "pt"])
        return d

    for k, (tr_idx, va_idx) in enumerate(gkf.split(TR, groups=trip_groups)):
        dtr = fit_rows(TR.iloc[tr_idx]); dva = TR.iloc[va_idx].reset_index(drop=True)
        model = make_model()
        t0 = time.time(); model.fit(dtr[FEATURES].values, dtr.label.values); fit_s = time.time() - t0
        t0 = time.time(); pva = model.predict_proba(dva[FEATURES].values)[:, 1]; pred_s = time.time() - t0
        oof[va_idx] = pva
        mva, _ = all_metrics(dva, pva, P, arcs_by_trip)
        # training-fold score (overfitting check) on at most train_eval_max_rows rows, whole points
        dtr_eval = dtr
        if len(dtr_eval) > train_eval_max_rows:
            keys = dtr_eval[["trip_id", "pt"]].drop_duplicates().sample(frac=train_eval_max_rows / len(dtr_eval), random_state=SEED)
            dtr_eval = dtr_eval.merge(keys, on=["trip_id", "pt"])
        dtr_eval = dtr_eval.sort_values(["trip_id", "pt", "dist_rank"]).reset_index(drop=True)
        ptr = model.predict_proba(dtr_eval[FEATURES].values)[:, 1]
        ytr = dtr_eval.label.values
        pptr = point_predictions(dtr_eval, ptr)
        fold_rows.append({"fold": k, "val_trips": int(dva.trip_id.nunique()), "fit_s": fit_s, "predict_s": pred_s,
                          "train_top1": pptr.pred_is_ref.mean(),
                          "train_macro_f1": f1_score(ytr, (ptr >= .5).astype(int), average="macro"),
                          "train_log_loss": log_loss(ytr, np.clip(ptr, 1e-7, 1 - 1e-7), labels=[0, 1]),
                          **{f"val_{a}": b for a, b in mva.items() if a != "confusion_matrix"}})
        print(f"  [{name}] fold {k}: val top1 {mva['top1_accuracy']:.4f}  macroF1 {mva['row_macro_f1']:.4f}  "
              f"logloss {mva['log_loss']:.4f} | train top1 {fold_rows[-1]['train_top1']:.4f} | fit {fit_s:.1f}s", flush=True)
    folds = pd.DataFrame(fold_rows)
    # final model on all training trips → student-test
    dall = fit_rows(TR)
    model = make_model()
    t0 = time.time(); model.fit(dall[FEATURES].values, dall.label.values); train_s = time.time() - t0
    t0 = time.time(); pte = model.predict_proba(TE[FEATURES].values)[:, 1]; infer_s = time.time() - t0
    mte, pp_te = all_metrics(TE, pte, P, arcs_by_trip)
    n_te_points = TE[["trip_id", "pt"]].drop_duplicates().shape[0]
    res = {"model": name, "family": family, "config": config, "features": FEATURES, "n_splits": n_splits,
           "train_rows_used_final": int(len(dall)), "train_subsample": train_subsample,
           "cv": {c: {"mean": float(folds[c].mean()), "std": float(folds[c].std())} for c in folds.columns if c not in ("fold",)},
           "student_test": {k_: (v if k_ == "confusion_matrix" else float(v)) for k_, v in mte.items()},
           "train_time_s": train_s, "inference_time_s_student_test": infer_s,
           "inference_ms_per_1000_points": infer_s / n_te_points * 1000 * 1000,
           "history": history_fn(model) if history_fn else None}
    folds.to_csv(os.path.join(out_dir, f"{name}_folds.csv"), index=False)
    pp_te.to_csv(os.path.join(out_dir, f"{name}_student_test_point_preds.csv"), index=False)
    pd.DataFrame({"trip_id": TR.trip_id, "pt": TR.pt, "arc": TR.arc, "oof_proba": oof}).to_csv(
        os.path.join(out_dir, f"{name}_oof.csv.gz"), index=False)
    pd.DataFrame({"trip_id": TE.trip_id, "pt": TE.pt, "arc": TE.arc, "proba": pte}).to_csv(
        os.path.join(out_dir, f"{name}_student_test_proba.csv.gz"), index=False)
    json.dump(res, open(os.path.join(out_dir, f"{name}_results.json"), "w"), indent=1, default=float)
    print(f"[{name}] CV top1 {res['cv']['val_top1_accuracy']['mean']:.4f}±{res['cv']['val_top1_accuracy']['std']:.4f} | "
          f"student-test top1 {mte['top1_accuracy']:.4f} macroF1 {mte['row_macro_f1']:.4f} logloss {mte['log_loss']:.4f} "
          f"continuity {mte['route_continuity']:.4f} | train {train_s:.1f}s infer {infer_s:.2f}s", flush=True)
    return res, model
