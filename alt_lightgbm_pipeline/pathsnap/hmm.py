"""HMM + Viterbi route reconstruction (Milestone 7, Member C).

Hidden state at GPS point t  = one candidate arc of that point (candidate row -> arc id
                               correspondence is kept, so the decoded state IS an arc id).
Emission  log P(z_t | arc)   = log of the classifier probability, normalised over the
                               point's candidates (so every point's emissions sum to 1).
Transition log P(arc_j | arc_i) = -|d_route - d_gps| / beta   (Newson & Krumm 2009)
        d_gps   = straight-line distance between the two GPS points
        d_route = driving distance along the DIRECTED network from the projection of
                  z_t on arc_i to the projection of z_{t+1} on arc_j (bounded Dijkstra).
Edge cases (from the M2 findings):
  * gap > gap_s seconds (trips 78, 90, ...): chain is split; the new piece starts fresh.
  * no path within the search limit (one-way, outside largest SCC, reference breaks):
    a fixed penalty `unreachable` instead of -inf, so the chain never dies.
  * a point whose candidate set is empty cannot happen (rule D7 keeps the nearest arc).
  * small backward moves on the same arc (parked jitter) are allowed with tolerance.
Viterbi: delta_t(j) = e_t(j) + max_i [delta_{t-1}(i) + a(i, j)], backpointers, then backtrack.
"""
from __future__ import annotations

import numpy as np

from .metrics import argmax_per_point  # noqa: F401  (re-exported for pipeline)

DEFAULT_CFG = dict(beta=5.0, gap_s=60.0, unreachable=25.0, same_arc_back_tol=5.0,
                   limit_factor=2.0, limit_extra=100.0, limit_max=3000.0, emission_floor=1e-4)


def _groups(point, n_points):
    order = np.argsort(point, kind="stable")
    counts = np.bincount(point, minlength=n_points)
    starts = np.r_[0, np.cumsum(counts)]
    return order, starts


def viterbi(prep: dict, prob: np.ndarray, cfg: dict | None = None):
    c = {k: float(v) for k, v in dict(DEFAULT_CFG, **(cfg or {})).items()}
    net, cand = prep["net"], prep["cand"]
    px, py, t = prep["px"], prep["py"], prep["t"]
    T = len(px)
    order, starts = _groups(cand["point"], T)
    arc_all, off_all = cand["arc"], cand["offset"]
    L = net.length
    beta = c["beta"]

    def state(tt):
        rows = order[starts[tt]:starts[tt + 1]]
        p = np.maximum(prob[rows], c["emission_floor"])
        return rows, arc_all[rows], off_all[rows], np.log(p / p.sum())

    rows0, arcs_prev, offs_prev, e = state(0)
    delta = e.copy()
    back = [None] * T
    rows_of = [None] * T
    rows_of[0] = rows0
    n_breaks = n_unreach_steps = 0
    for tt in range(1, T):
        rows, arcs, offs, e = state(tt)
        rows_of[tt] = rows
        dt = t[tt] - t[tt - 1]
        if dt > c["gap_s"]:
            back[tt] = np.full(len(rows), int(np.argmax(delta)))
            delta = e + delta.max()
            n_breaks += 1
            arcs_prev, offs_prev = arcs, offs
            continue
        dg = float(np.hypot(px[tt] - px[tt - 1], py[tt] - py[tt - 1]))
        limit = min(c["limit_max"], c["limit_factor"] * dg + c["limit_extra"])
        A = np.full((len(arcs_prev), len(arcs)), -float(c["unreachable"]), dtype=float)
        for i, (ai, oi) in enumerate(zip(arcs_prev, offs_prev)):
            rest = L[ai] - oi
            dist = None
            for j, (aj, oj) in enumerate(zip(arcs, offs)):
                if aj == ai and oj >= oi - c["same_arc_back_tol"]:
                    dr = abs(oj - oi)
                else:
                    if dist is None:
                        dist = net.dists_from(int(net.to[ai]), limit)
                    sp = dist.get(int(net.frm[aj]))
                    if sp is None or sp > limit:   # cache may hold a larger search
                        continue
                    dr = rest + sp + oj
                A[i, j] = -abs(dr - dg) / beta
        if (A <= -c["unreachable"]).all():
            n_unreach_steps += 1
        s = delta[:, None] + A
        bp = np.argmax(s, axis=0)
        delta = s[bp, np.arange(len(arcs))] + e
        back[tt] = bp
        arcs_prev, offs_prev = arcs, offs
    # backtrack
    out = np.empty(T, np.int64)
    j = int(np.argmax(delta))
    for tt in range(T - 1, -1, -1):
        out[tt] = arc_all[rows_of[tt][j]]
        if tt > 0:
            j = int(back[tt][j])
    return out, dict(n_breaks=n_breaks, n_unreachable_steps=n_unreach_steps)


def stitch_route(arcs_per_point: np.ndarray, t: np.ndarray, net, px=None, py=None,
                 gap_s: float = 60.0, factor: float = 2.0, extra: float = 50.0) -> list:
    """Turn the per-point decoded arcs into one drivable arc sequence: consecutive
    distinct arcs are joined with the shortest directed path between them (the arcs the
    vehicle passed between two GPS fixes). A join is only made if that path is no longer
    than factor x GPS distance + extra (a long detour means the decoded arc is wrong,
    so the pieces are left unjoined). Pieces across a big time gap are never joined."""
    from .network import shortest_path_arcs
    route = []
    prev = None
    for k, (a, tt) in enumerate(zip(arcs_per_point, t)):
        a = int(a)
        if prev is None:
            route.append(a)
        elif a != prev:
            if tt - t[k - 1] <= gap_s:
                dg = float(np.hypot(px[k] - px[k - 1], py[k] - py[k - 1])) if px is not None else 0.0
                mid = shortest_path_arcs(net, int(net.to[prev]), int(net.frm[a]), factor * dg + extra)
                if mid:
                    route.extend(int(x) for x in mid if net.frm[x] != net.to[x])
            route.append(a)
        prev = a
    return route
