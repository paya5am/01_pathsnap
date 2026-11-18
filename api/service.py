"""Prediction pipeline used by the API. It NEVER reads the reference route: read_trip(..., with_route=False).

trip folder → projection → network + candidates → 16 features → classifier P(correct) → HMM lattice → Viterbi
→ predicted arc per GPS point + connected reconstructed route (with shortest-path fill between decoded arcs).
"""
import time
import numpy as np
from pathsnap.io import read_trip
from pathsnap.geo import make_projection
from pathsnap.network import build_network
from pathsnap.candidates import generate_candidates
from pathsnap.track import motion_features
from pathsnap.features import candidate_features, FEATURES
from pathsnap.hmm import TripLattice, viterbi, reconstruct_route


def predict_trip(trip_dir, artifacts):
    tm = {}
    t0 = time.perf_counter()
    raw = read_trip(str(trip_dir), with_route=False)           # ground truth is never loaded here
    tr = raw.track
    lon0, lat0 = tr.lon.mean(), tr.lat.mean()
    P = make_projection(lon0, lat0)(tr.lon.values, tr.lat.values)
    net = build_network(raw.nodes, raw.arcs, lon0, lat0, aoi_xy=P)
    t = tr.t.values
    motion = motion_features(P, t)
    cands, d_nearest = generate_candidates(net, P)
    feats = candidate_features(net, P, t, motion, cands, d_nearest)
    tm["preprocess_s"] = time.perf_counter() - t0
    t1 = time.perf_counter()
    proba = artifacts.model.predict_proba(feats[FEATURES].values)[:, 1] if len(feats) else np.zeros(0)
    tm["classifier_s"] = time.perf_counter() - t1
    t2 = time.perf_counter()
    lookup = dict(zip(zip(feats.pt.values.tolist(), feats.arc.values.tolist()), proba.tolist()))
    lat = TripLattice.from_parts(raw.trip_id, net, P, t, tr[["lon", "lat"]].values, raw.nodes[["lon", "lat"]].values, cands, lookup)
    pred, seg = viterbi(lat, **artifacts.decoder)
    clf_pred = np.array([a[np.argmax(lp)] if len(a) else -1 for a, lp in zip(lat.cand_arc, lat.logp)], np.int64)
    route = reconstruct_route(lat, pred)
    tm["hmm_s"] = time.perf_counter() - t2
    tm["total_s"] = time.perf_counter() - t0
    return {"trip_id": raw.trip_id, "lat": lat, "pred": pred, "clf_pred": clf_pred, "segments": seg, "route": route,
            "n_points": len(P), "n_candidate_rows": int(len(feats)), "timings": tm}


def route_geometry(lat, arc_ids):
    """[[lat, lon], [lat, lon]] per arc, for drawing."""
    nd, A = lat.nodes_lonlat, lat.net.arcs
    return [[[float(nd[A[a, 0], 1]), float(nd[A[a, 0], 0])], [float(nd[A[a, 1], 1]), float(nd[A[a, 1], 0])]] for a in arc_ids]


def continuity(lat, pred):
    """strict = consecutive predicted arcs identical or touching; reachable = drivable within the decoder's limit."""
    A = lat.net.arcs
    v = (pred[:-1] >= 0) & (pred[1:] >= 0) & (lat.dt[:-1] <= 60)
    a, b = pred[:-1][v], pred[1:][v]
    strict = ((a == b) | (A[a, 1] == A[b, 0])).mean() if len(a) else float("nan")
    reach = []
    for i in np.where(v)[0]:
        M = lat.cost[i]
        if M is None: reach.append(False); continue
        x = np.where(lat.cand_arc[i] == pred[i])[0][0]; y = np.where(lat.cand_arc[i + 1] == pred[i + 1])[0][0]
        reach.append(bool(np.isfinite(M[x, y])))
    return float(strict), float(np.mean(reach)) if reach else float("nan")
