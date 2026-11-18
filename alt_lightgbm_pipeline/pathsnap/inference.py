"""Inference entry points for the API (Member D calls these; Member C owns them).

    arts = load_artifacts("artifacts")
    pred = predict_trip_folder("data/eval/00000015", arts)      # never reads .route
    score = evaluate_prediction(pred, "data/eval/00000015")     # server side only

The prediction step loads the trip WITHOUT its route, so ground truth can never leak
into the model. evaluate_prediction() reads the route separately, after prediction.
"""
from __future__ import annotations

import json
import os
import time

import joblib
import numpy as np

from . import candidates, features, hmm, labels, metrics
from .io import load_trip_folder
from .network import build_network, make_projection


def load_artifacts(art_dir: str) -> dict:
    with open(os.path.join(art_dir, "hmm_config.json")) as f:
        hmm_cfg = json.load(f)
    with open(os.path.join(art_dir, "preprocessing_config.json")) as f:
        pre = json.load(f)
    if pre["features"] != features.FEATURES:
        raise ValueError("feature list in artifacts does not match pathsnap.features.FEATURES")
    return dict(model=joblib.load(os.path.join(art_dir, "model.joblib")), hmm_cfg=hmm_cfg, pre=pre)


def _prepare_unlabelled(trip, pre):
    proj = make_projection(trip.track)
    px, py = proj(trip.track[:, 0], trip.track[:, 1])
    t = trip.track[:, 2].astype(float)
    net = build_network(trip.nodes, trip.arcs, proj)
    idx = candidates.ArcIndex(net, pre["candidate_spacing_m"])
    cand = candidates.generate(px, py, idx, pre["candidate_delta_m"], pre["candidate_cap_m"])
    heading, speed = labels.motion(px, py, t)
    X = features.build(cand, px, py, heading, speed, net)
    return dict(px=px, py=py, t=t, heading=heading, speed=speed, net=net, cand=cand, X=X)


def predict_trip_folder(folder: str, arts: dict, use_hmm: bool = True) -> dict:
    timings = {}
    t0 = time.time()
    trip = load_trip_folder(folder, with_route=False)
    prep = _prepare_unlabelled(trip, arts["pre"])
    timings["preprocess_s"] = time.time() - t0
    t0 = time.time()
    p = arts["model"].predict_proba(prep["X"])[:, 1]
    timings["classifier_s"] = time.time() - t0
    top = metrics.argmax_per_point(prep["cand"]["point"], p, len(prep["px"]))
    clf_arcs = prep["cand"]["arc"][top]
    arcs = clf_arcs
    info = {}
    if use_hmm:
        t0 = time.time()
        arcs, info = hmm.viterbi(prep, p, arts["hmm_cfg"])
        timings["hmm_s"] = time.time() - t0
    net = prep["net"]
    route = hmm.stitch_route(arcs, prep["t"], net, prep["px"], prep["py"], arts["hmm_cfg"]["gap_s"])
    nodes = trip.nodes
    seg = [[[float(nodes[net.frm[a], 1]), float(nodes[net.frm[a], 0])],
            [float(nodes[net.to[a], 1]), float(nodes[net.to[a], 0])]] for a in route]
    return dict(trip_id=trip.trip_id, n_points=len(arcs), point_arcs=arcs.tolist(),
                classifier_point_arcs=clf_arcs.tolist(), route_arcs=route,
                route_latlon_segments=seg, gps_latlon=trip.track[:, [1, 0]].tolist(),
                decoder_info=info, timings=timings, _prep=prep)


def evaluate_prediction(pred: dict, folder: str) -> dict:
    """Server-side scoring against the reference route (labels derived as in training)."""
    trip = load_trip_folder(folder, with_route=True)
    prep = pred["_prep"]
    lab, _ = labels.align(prep["px"], prep["py"], prep["heading"], prep["speed"], prep["net"], trip.route)
    L = lab >= 0
    arcs = np.asarray(pred["point_arcs"]); carcs = np.asarray(pred["classifier_point_arcs"])
    hit = np.zeros(len(lab), bool)
    c = prep["cand"]
    hit[c["point"][c["arc"] == lab[c["point"]]]] = True
    out = dict(
        n_points=int(len(lab)), n_labelled=int(L.sum()),
        accuracy_hmm=float((arcs[L] == lab[L]).mean()) if L.any() else None,
        accuracy_classifier=float((carcs[L] == lab[L]).mean()) if L.any() else None,
        candidate_miss_rate=float(1 - hit[L].mean()) if L.any() else None,
        route_continuity=metrics.route_continuity(arcs, prep["t"], prep["net"]),
    )
    st = hmm.stitch_route(arcs[L], prep["t"][L], prep["net"], prep["px"][L], prep["py"][L])
    out |= metrics.sequence_metrics(np.array(st), trip.route, prep["net"].canon)
    return out
