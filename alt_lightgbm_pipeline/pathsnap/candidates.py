"""Candidate generation (decision D7).

Candidates of GPS point i = every usable arc whose point-to-segment distance is
<= min(d_nearest + DELTA, CAP). If the nearest arc is itself beyond CAP, the
nearest arc is still kept, so no point ever has an empty candidate set.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .network import Network, point_segment

DELTA = 20.0
CAP = 150.0
SPACING = 10.0  # densification step for the search index (m)


class ArcIndex:
    def __init__(self, net: Network, spacing: float = SPACING):
        ids = np.nonzero(net.valid)[0]
        n = np.maximum(1, np.ceil(net.length[ids] / spacing).astype(int))
        rep = np.repeat(ids, n + 1)
        # fractions 0..1 along each arc
        starts = np.repeat(np.cumsum(np.concatenate([[0], n[:-1] + 1])), n + 1)
        k = np.arange(len(rep)) - starts
        f = k / np.repeat(n, n + 1)
        sx = net.ax[rep] + f * (net.bx[rep] - net.ax[rep])
        sy = net.ay[rep] + f * (net.by[rep] - net.ay[rep])
        self.tree = cKDTree(np.c_[sx, sy])
        self.sample_arc = rep
        self.spacing = spacing
        self.net = net


def generate(px: np.ndarray, py: np.ndarray, index: ArcIndex,
             delta: float = DELTA, cap: float = CAP):
    """Returns dict of flat arrays: point, arc, dist, offset (one row per candidate)."""
    net = index.net
    pts = np.c_[px, py]
    d1, s1 = index.tree.query(pts, k=1)
    a1 = index.sample_arc[s1]
    d_est, _ = point_segment(px, py, net, a1)           # exact distance to that arc >= d_near
    radius = np.minimum(d_est + delta, np.maximum(cap, d_est)) + index.spacing / 2 + 0.5
    hits = index.tree.query_ball_point(pts, r=radius)
    P, A = [], []
    for i, h in enumerate(hits):
        arcs = np.unique(index.sample_arc[h]) if len(h) else np.array([a1[i]])
        P.append(np.full(len(arcs), i)); A.append(arcs)
    P = np.concatenate(P); A = np.concatenate(A)
    d, off = point_segment(px[P], py[P], net, A)
    # exact nearest per point
    d_near = np.full(len(px), np.inf)
    np.minimum.at(d_near, P, d)
    thr = np.minimum(d_near + delta, np.maximum(cap, d_near))
    keep = d <= thr[P] + 1e-9
    return dict(point=P[keep], arc=A[keep], dist=d[keep], offset=off[keep], d_near=d_near)
