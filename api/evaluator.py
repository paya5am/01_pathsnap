"""Server-side evaluation: compares predictions with the internal reference route. Runs AFTER prediction and only
on Laptop 1. Reference arcs/geometry are never sent to the UI and never reach the model; only summary numbers do.
Labels are derived exactly as in training (heading- and order-aware alignment, pathsnap.labels.align_route)."""
import numpy as np
from pathsnap.io import read_trip
from pathsnap.track import motion_features
from pathsnap.labels import align_route


def evaluate_trip(trip_dir, result):
    raw = read_trip(str(trip_dir), with_route=True)
    if raw.route is None:
        return {"available": False, "reason": "no reference route for this trip"}
    lat = result["lat"]
    motion = motion_features(lat.P, lat.t)
    lab = align_route(lat.net, lat.P, motion, raw.route)
    ok = lab["label_status"] == 0; ref = lab["ref_arc"]
    pred, clf = result["pred"], result["clf_pred"]
    ref_set = set(ref[ok].tolist()); pred_set = set(pred[ok & (pred >= 0)].tolist())
    tp = len(ref_set & pred_set); pr = tp / max(len(pred_set), 1); rc = tp / max(len(ref_set), 1)
    return {"available": True,
            "labelled_points": int(ok.sum()), "unlabelled_points": int((~ok).sum()),
            "top1_accuracy_hmm": float(np.mean(pred[ok] == ref[ok])) if ok.any() else None,
            "top1_accuracy_classifier_only": float(np.mean(clf[ok] == ref[ok])) if ok.any() else None,
            "route_arc_precision": pr, "route_arc_recall": rc, "route_arc_f1": 2 * pr * rc / max(pr + rc, 1e-9)}
