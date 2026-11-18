"""M7 (Member C): tune decoder parameters on TRAIN (out-of-fold classifier probabilities),
then compare classifier-only vs classifier+HMM on the 10 student-test trips.

usage: python scripts/run_hmm.py <data_root> <results_dir>
Needs results_dir/lgbm_results.json and lgbm_tuned.joblib from train_lgbm.py.
"""
import itertools
import json
import os
import sys
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap import hmm, io, metrics, pipeline  # noqa: E402
from pathsnap.cv import trip_folds  # noqa: E402

GRID = dict(beta=[2, 5, 10, 20, 40, 80, 160], unreachable=[0.5, 1, 2, 5, 10, 25])


def oof_all_rows(root, params):
    """Out-of-fold probabilities for EVERY candidate row of every train trip
    (fit on labelled rows of the other folds, predict all rows of the held-out trips)."""
    preps = {}
    for tid in io.list_trips(root, "train"):
        preps[tid] = pipeline.prepare_trip(io.load_trip(root, "train", tid))
    trips = np.array(sorted(preps))
    # fold assignment at trip level, same rule as training CV
    fold_of = {}
    for k, (_, va) in enumerate(trip_folds(trips)):
        for t in trips[va]:
            fold_of[int(t)] = k
    probs = {}
    for k in range(5):
        tr_trips = [t for t in trips if fold_of[int(t)] != k]
        X = np.concatenate([preps[t]["X"][(preps[t]["label"] >= 0)[preps[t]["cand"]["point"]]] for t in tr_trips])
        y = np.concatenate([preps[t]["y"][(preps[t]["label"] >= 0)[preps[t]["cand"]["point"]]] for t in tr_trips])
        m = lgb.LGBMClassifier(**params).fit(X, y)
        for t in trips:
            if fold_of[int(t)] == k:
                probs[int(t)] = m.predict_proba(preps[t]["X"])[:, 1]
    return preps, probs


def score_trip(prep, arcs, route):
    lab = prep["label"]; L = lab >= 0
    seq = metrics.sequence_metrics(arcs[L], route, prep["net"].canon)
    st = hmm.stitch_route(arcs[L], prep["t"][L], prep["net"], prep["px"][L], prep["py"][L])
    seq |= {"stitched_" + k: v for k, v in
            metrics.sequence_metrics(np.array(st), route, prep["net"].canon).items()}
    return dict(top1=float((arcs[L] == lab[L]).mean()) if L.any() else np.nan, n_labelled=int(L.sum()),
                continuity=metrics.route_continuity(arcs, prep["t"], prep["net"]), **seq)


def main(root, res_dir):
    R = json.load(open(os.path.join(res_dir, "lgbm_results.json")))
    params = R["tuned"]["params"]
    t0 = time.time()
    preps, probs = oof_all_rows(root, params)
    print("OOF done %.0fs" % (time.time() - t0), flush=True)

    # ---------- grid search of decoder parameters on train ----------
    rows = []
    cfgs = [dict(beta=b, unreachable=u) for b, u in itertools.product(GRID["beta"], GRID["unreachable"])]
    for tid, prep in preps.items():
        lab = prep["label"]; L = lab >= 0
        if not L.any():
            continue
        for cfg in cfgs:
            arcs, _ = hmm.viterbi(prep, probs[tid], cfg)
            rows.append(dict(trip_id=tid, **cfg, correct=int((arcs[L] == lab[L]).sum()), n=int(L.sum())))
        prep["net"]._sp_cache.clear()
    g = pd.DataFrame(rows)
    grid = g.groupby(["beta", "unreachable"])[["correct", "n"]].sum().reset_index()
    grid["train_oof_top1"] = grid.correct / grid.n
    grid.to_csv(os.path.join(res_dir, "hmm_grid_train.csv"), index=False)
    b = grid.sort_values("train_oof_top1", ascending=False).iloc[0]
    best_cfg = dict(hmm.DEFAULT_CFG, beta=float(b.beta), unreachable=float(b.unreachable))
    print("best decoder cfg", best_cfg, "train OOF top1", b.train_oof_top1, flush=True)

    # gap threshold sensitivity (train)
    gap_rows = []
    for gap in [30, 60, 120, 300]:
        c = n = 0
        for tid, prep in preps.items():
            lab = prep["label"]; L = lab >= 0
            if not L.any():
                continue
            arcs, _ = hmm.viterbi(prep, probs[tid], dict(best_cfg, gap_s=gap))
            c += int((arcs[L] == lab[L]).sum()); n += int(L.sum())
        gap_rows.append(dict(gap_s=gap, train_oof_top1=c / n))
    pd.DataFrame(gap_rows).to_csv(os.path.join(res_dir, "hmm_gap_sensitivity_train.csv"), index=False)
    best_gap = max(gap_rows, key=lambda r: r["train_oof_top1"])["gap_s"]
    best_cfg["gap_s"] = float(best_gap)

    # train OOF comparison per trip (for the report)
    per = []
    for tid, prep in preps.items():
        trip = io.load_trip(root, "train", tid)
        top = metrics.argmax_per_point(prep["cand"]["point"], probs[tid], len(prep["px"]))
        clf_arcs = prep["cand"]["arc"][top]
        hmm_arcs, info = hmm.viterbi(prep, probs[tid], best_cfg)
        per.append(dict(partition="train_oof", trip_id=tid, method="classifier", **score_trip(prep, clf_arcs, trip.route)))
        per.append(dict(partition="train_oof", trip_id=tid, method="classifier+hmm", **score_trip(prep, hmm_arcs, trip.route), **info))
    del preps

    # ---------- student-test: final tuned model (fit on all train) ----------
    model = joblib.load(os.path.join(res_dir, "lgbm_tuned.joblib"))
    for tid in io.list_trips(root, "student_test"):
        trip = io.load_trip(root, "student_test", tid)
        t0 = time.time(); prep = pipeline.prepare_trip(trip); t_prep = time.time() - t0
        t0 = time.time(); p = model.predict_proba(prep["X"])[:, 1]; t_clf = time.time() - t0
        top = metrics.argmax_per_point(prep["cand"]["point"], p, len(prep["px"]))
        clf_arcs = prep["cand"]["arc"][top]
        t0 = time.time(); hmm_arcs, info = hmm.viterbi(prep, p, best_cfg); t_hmm = time.time() - t0
        base = dict(partition="student_test", trip_id=tid, prep_s=t_prep, clf_s=t_clf)
        per.append(dict(base, method="classifier", **score_trip(prep, clf_arcs, trip.route)))
        per.append(dict(base, method="classifier+hmm", hmm_s=t_hmm, **score_trip(prep, hmm_arcs, trip.route), **info))
    P = pd.DataFrame(per)
    P.to_csv(os.path.join(res_dir, "hmm_vs_classifier_per_trip.csv"), index=False)

    def agg(df):
        w = df.n_labelled
        return pd.Series(dict(top1_point_weighted=(df.top1 * w).sum() / w.sum(),
                              top1_trip_mean=df.top1.mean(), continuity_mean=df.continuity.mean(),
                              seq_precision=df.seq_precision.mean(), seq_recall=df.seq_recall.mean(),
                              seq_f1=df.seq_f1.mean(), stitched_seq_precision=df.stitched_seq_precision.mean(),
                              stitched_seq_recall=df.stitched_seq_recall.mean(),
                              stitched_seq_f1=df.stitched_seq_f1.mean(), n_points=w.sum()))
    S = P.groupby(["partition", "method"]).apply(agg).reset_index()
    S.to_csv(os.path.join(res_dir, "hmm_vs_classifier_summary.csv"), index=False)
    print(S.to_string())
    with open(os.path.join(res_dir, "hmm_config.json"), "w") as f:
        json.dump(best_cfg, f, indent=2)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
