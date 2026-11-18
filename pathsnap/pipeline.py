"""One call that turns a trip folder into candidate rows (+ labels when a route is available)."""
import numpy as np
import pandas as pd
from .config import CV_GROUP_MERGE
from .io import read_trip
from .network import build_network
from .candidates import generate_candidates
from .track import motion_features
from .features import candidate_features
from .labels import align_route


def process_trip(trip_dir, with_labels=True):
    raw = read_trip(trip_dir, with_route=with_labels)
    tr = raw.track
    lon0, lat0 = tr.lon.mean(), tr.lat.mean()
    from .geo import make_projection
    P = make_projection(lon0, lat0)(tr.lon.values, tr.lat.values)
    net = build_network(raw.nodes, raw.arcs, lon0, lat0, aoi_xy=P)
    t = tr.t.values
    motion = motion_features(P, t)
    cands, d_nearest = generate_candidates(net, P)
    feats = candidate_features(net, P, t, motion, cands, d_nearest)
    feats.insert(0, "trip_id", raw.trip_id)
    points = pd.DataFrame({"trip_id": raw.trip_id, "pt": np.arange(len(P)), "t": t,
                           "d_nearest": d_nearest, "moving": motion["moving"],
                           "n_cands": np.bincount(cands.pt.values, minlength=len(P))})
    if with_labels and raw.route is not None:
        lab = align_route(net, P, motion, raw.route)
        for k, v in lab.items():
            points[k] = v
        ref = lab["ref_arc"][feats.pt.values]
        status = lab["label_status"][feats.pt.values].astype(np.int8)
        # label: 1 = this candidate is the reference arc, 0 = it is not, -1 = point unlabelled (never train/evaluate on it)
        feats["label"] = np.where(status == 0, (feats.arc.values == ref).astype(np.int8), -1).astype(np.int8)
        feats["label_status"] = status
        has_pos = feats[feats.label == 1].groupby("pt").size() > 0
        points["ref_in_cands"] = points.pt.map(has_pos).fillna(0).astype(bool)
    group = CV_GROUP_MERGE.get(raw.trip_id, raw.trip_id)
    feats["cv_group"] = group; points["cv_group"] = group
    return feats, points, {"raw": raw, "net": net, "P": P, "motion": motion}
