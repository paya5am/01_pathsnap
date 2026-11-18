"""End-to-end per-trip processing shared by notebooks, the API and the UI.

prepare_trip()  : raw trip -> projected geometry, candidates, features (+ labels if route given)
predict_trip()  : prepared trip + model + HMM config -> predicted arc per GPS point
The route (ground truth) is only used by labels/metrics, never by prediction.
"""
from __future__ import annotations

import numpy as np

from . import candidates, features, labels
from .io import Trip
from .network import build_network, make_projection


def prepare_trip(trip: Trip, with_labels: bool = True) -> dict:
    proj = make_projection(trip.track)
    px, py = proj(trip.track[:, 0], trip.track[:, 1])
    t = trip.track[:, 2].astype(float)
    net = build_network(trip.nodes, trip.arcs, proj)
    index = candidates.ArcIndex(net)
    cand = candidates.generate(px, py, index)
    heading, speed = labels.motion(px, py, t)
    X = features.build(cand, px, py, heading, speed, net)
    out = dict(trip_id=trip.trip_id, px=px, py=py, t=t, heading=heading, speed=speed,
               net=net, cand=cand, X=X)
    if with_labels and trip.route is not None:
        lab, pos = labels.align(px, py, heading, speed, net, trip.route)
        out["label"] = lab                       # canonical arc id per point, -1 = unlabelled
        out["route_pos"] = pos
        out["y"] = (cand["arc"] == lab[cand["point"]]).astype(np.int8)
    return out


def predict_trip(prep: dict, model, hmm_cfg: dict | None = None) -> dict:
    """Classifier scores every candidate; optionally decode with the HMM."""
    from . import hmm
    p = model.predict_proba(prep["X"])[:, 1]
    cand = prep["cand"]
    from .metrics import argmax_per_point
    top1 = argmax_per_point(cand["point"], p, len(prep["px"]))
    res = dict(prob=p, classifier_arc=cand["arc"][top1])
    if hmm_cfg is not None:
        res["hmm_arc"], res["hmm_info"] = hmm.viterbi(prep, p, hmm_cfg)
    return res
