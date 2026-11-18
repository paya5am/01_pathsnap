"""Per (GPS point, candidate arc) features.

STAND-IN for Member B's M3 features.py: a minimal set tied to M2 findings, so the
classifier -> HMM pipeline can be built and tested now. B owns the final list.
"""
from __future__ import annotations

import numpy as np

from .labels import ang_diff_deg, MOVING
from .network import Network, point_segment

FEATURES = [
    "dist", "dist_rel", "rank", "n_cands", "d_near",
    "head_diff", "head_cos", "speed", "moving",
    "arc_len_log", "two_way", "off_frac",
    "dist_prev3", "dist_next3", "head_diff_x_moving",
]


def build(cand: dict, px, py, heading, speed, net: Network) -> np.ndarray:
    P, A, d = cand["point"], cand["arc"], cand["dist"]
    T = len(px)
    d_near = cand["d_near"][P]
    # rank of each candidate by distance within its point
    order = np.lexsort((d, P))
    rank = np.empty(len(P), float)
    starts = np.r_[0, np.nonzero(np.diff(P[order]))[0] + 1]
    cnt = np.diff(np.r_[starts, len(P)])
    rank[order] = np.arange(len(P)) - np.repeat(starts, cnt)
    n_c = np.bincount(P, minlength=T)[P].astype(float)
    hd = ang_diff_deg(heading[P], net.bearing[A])
    mv = (speed[P] >= MOVING).astype(float)
    L = net.length[A]
    ip = np.clip(P - 3, 0, T - 1); inx = np.clip(P + 3, 0, T - 1)
    dp, _ = point_segment(px[ip], py[ip], net, A)
    dn, _ = point_segment(px[inx], py[inx], net, A)
    X = np.c_[d, d - d_near, rank, n_c, d_near,
              hd, np.cos(np.radians(hd)), speed[P], mv,
              np.log1p(L), net.two_way[A].astype(float), cand["offset"] / np.maximum(L, 1e-6),
              dp, dn, hd * mv]
    return X.astype(np.float32)
