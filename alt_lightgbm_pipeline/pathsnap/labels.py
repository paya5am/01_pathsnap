"""GPS motion (heading/speed) and per-point reference labels (decisions D2, D6, D9).

STAND-IN for Member A's M3 labels.py: same interface, replace when A's version lands.

The .route is an ordered arc sequence, not one label per point. We align points
to route POSITIONS with a monotone dynamic programme (time order can only move
forward along the route; revisited arcs are separate positions), using a cost
of distance + heading mismatch. Points with no route arc within MAX_DIST, or
whose aligned arc disagrees in heading while moving, are labelled -1 (unlabelled):
excluded from training and from accuracy, and counted.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .network import Network, point_segment

MAX_DIST = 50.0       # m
W_HEAD = 0.25         # m of cost per degree of heading mismatch (90 deg = 22.5 m)
MOVING = 2.0          # m/s: below this heading is unreliable
HALF_WIN = 3          # points either side for heading/speed
C_SKIP = 50.0         # cost of leaving a point unlabelled (= MAX_DIST)
BEAM = 300.0          # prune DP states this much worse than the best


def motion(px, py, t, half=HALF_WIN):
    n = len(px)
    i0 = np.clip(np.arange(n) - half, 0, n - 1)
    i1 = np.clip(np.arange(n) + half, 0, n - 1)
    dx, dy = px[i1] - px[i0], py[i1] - py[i0]
    dt = np.maximum(t[i1] - t[i0], 1e-6)
    speed = np.hypot(dx, dy) / dt
    heading = np.arctan2(dy, dx)
    return heading, np.clip(speed, 0, 60.0)


def ang_diff_deg(a, b):
    d = np.abs((a - b + np.pi) % (2 * np.pi) - np.pi)
    return np.degrees(d)


def align(px, py, heading, speed, net: Network, route: np.ndarray):
    """Return label arc (canonical id or -1) per point and the aligned route position."""
    T = len(px)
    R = len(route)
    labels = np.full(T, -1, np.int64)
    pos = np.full(T, -1, np.int64)
    if R == 0:
        return labels, pos
    # densify route positions for a radius query
    L = net.length[route]
    n = np.maximum(1, np.ceil(L / 10.0).astype(int))
    rep = np.repeat(np.arange(R), n + 1)
    starts = np.repeat(np.cumsum(np.concatenate([[0], n[:-1] + 1])), n + 1)
    f = (np.arange(len(rep)) - starts) / np.repeat(n, n + 1)
    a = route[rep]
    sx = net.ax[a] + f * (net.bx[a] - net.ax[a]); sy = net.ay[a] + f * (net.by[a] - net.ay[a])
    tree = cKDTree(np.c_[sx, sy])
    hits = tree.query_ball_point(np.c_[px, py], r=MAX_DIST + 5.5)

    moving = speed >= MOVING
    # Sparse monotone DP with a "skip" option (point left unlabelled at cost C_SKIP),
    # so one odd point can never drag the whole alignment backwards. Beam-pruned.
    K = np.zeros(0, np.int64); V = np.zeros(0)
    steps = []          # per processed point: (i, positions, prev_index, labelled_flag)
    for i in range(T):
        if not hits[i]:
            continue
        ks = np.unique(rep[hits[i]])
        arcs = route[ks]
        d, _ = point_segment(np.full(len(ks), px[i]), np.full(len(ks), py[i]), net, arcs)
        ok = d <= MAX_DIST
        if not ok.any():
            continue
        ks, arcs, d = ks[ok], arcs[ok], d[ok]
        c = d.copy()
        if moving[i]:
            c += W_HEAD * ang_diff_deg(heading[i], net.bearing[arcs])
        if len(K) == 0:
            nk, nv, npr, nl = ks, c, np.full(len(ks), -1), np.ones(len(ks), bool)
        else:
            pm = np.minimum.accumulate(V)
            am = np.zeros(len(V), np.int64)
            for j in range(1, len(V)):
                am[j] = am[j - 1] if pm[j - 1] <= V[j] else j
            idx = np.searchsorted(K, ks, side="right") - 1
            f = idx >= 0
            lk, lv, lp = ks[f], c[f] + pm[idx[f]], am[idx[f]]
            # merge labelled states with carried (skipped) states
            allk = np.concatenate([K, lk]); allv = np.concatenate([V + C_SKIP, lv])
            allp = np.concatenate([np.arange(len(K)), lp])
            alll = np.concatenate([np.zeros(len(K), bool), np.ones(len(lk), bool)])
            o = np.lexsort((allv, allk))
            allk, allv, allp, alll = allk[o], allv[o], allp[o], alll[o]
            first = np.r_[True, allk[1:] != allk[:-1]]
            nk, nv, npr, nl = allk[first], allv[first], allp[first], alll[first]
        keep = nv <= nv.min() + BEAM
        # remap is not needed for prev pointers (they index the previous set), only filter
        nk, nv, npr, nl = nk[keep], nv[keep], npr[keep], nl[keep]
        steps.append((i, nk, npr, nl))
        K, V = nk, nv
    if not steps:
        return labels, pos
    j = int(np.argmin(V))
    for s in range(len(steps) - 1, -1, -1):
        i, nk, npr, nl = steps[s]
        if nl[j]:
            pos[i] = nk[j]
        j = int(npr[j])
        if j < 0:
            break
    # final checks
    lab_arc = np.where(pos >= 0, route[np.maximum(pos, 0)], -1)
    has = pos >= 0
    d, _ = point_segment(px[has], py[has], net, lab_arc[has])
    hd = ang_diff_deg(heading[has], net.bearing[lab_arc[has]])
    bad = (d > MAX_DIST) | (moving[has] & (hd > 90))
    idx = np.nonzero(has)[0]
    lab = net.canon[lab_arc[has]]
    lab[bad] = -1
    labels[idx] = lab
    pos[idx[bad]] = -1
    return labels, pos
