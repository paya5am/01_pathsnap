"""Per-GPS-point reference labels derived from the ordered .route (decisions D6, D9).

The .route is an ordered list of arcs, not one label per GPS point. We align points to route positions with a
monotone dynamic programme (like DTW): point i is matched to route position j(i) with j non-decreasing in time,
minimising   sum_i  min(d(i, j), CAP) + PENALTY * [point i is moving AND arc j points the other way].
  * order-aware  -> a point can't be matched to a route arc the vehicle only reaches later/earlier
  * heading-aware -> on out-and-back trips (trip 11) the outbound and return arcs share geometry; heading picks the right one
Afterwards a point is LABELLED only if its matched arc is within LABEL_MAX_DIST_M and it is not inside a sustained
(>= HEADING_RUN_MIN points) run where the matched arc points against the vehicle's motion;
otherwise it is UNLABELLED (parked off-network, beyond the route's end, local reference errors such as trip 66).
"""
import numpy as np
from .config import ALIGN_DIST_CAP_M, ALIGN_HEADING_PENALTY_M, LABEL_MAX_DIST_M, HEADING_RUN_MIN
from .geo import seg_project, angle_diff_deg

STATUS = {0: "labelled", 1: "unlabelled_far", 2: "unlabelled_heading"}


def align_route(net, P, motion, route, chunk=1000, dist_cap=ALIGN_DIST_CAP_M, heading_penalty=ALIGN_HEADING_PENALTY_M,
                label_max_dist=LABEL_MAX_DIST_M, heading_run_min=HEADING_RUN_MIN):
    n, m = len(P), len(route)
    A, B = net.nxy[net.arcs[route, 0]], net.nxy[net.arcs[route, 1]]
    rb = net.bearing[route]
    back = np.empty((n, m), np.int32)          # back-pointer: best previous route position
    prev = None
    dist_keep = np.empty((n, m), np.float32)   # needed to report the matched distance
    head_bad = np.zeros((n, m), bool)
    for s in range(0, n, chunk):
        sl = slice(s, min(s + chunk, n))
        d, _ = seg_project(P[sl, None], A[None], B[None])
        bad = motion["moving"][sl, None] & (angle_diff_deg(motion["heading"][sl, None], rb[None]) >= 90)
        cost = np.minimum(d, dist_cap) + heading_penalty * bad
        dist_keep[sl] = d; head_bad[sl] = bad
        for r in range(cost.shape[0]):
            i = s + r
            if prev is None:
                cur = cost[r].astype(float); back[i] = np.arange(m)
            else:
                # best previous position k <= j : running minimum with argmin
                run_idx = np.arange(m)
                best = np.minimum.accumulate(prev)
                # argmin of prefix minimum: index where the running minimum was last updated
                upd = np.r_[True, prev[1:] < best[:-1]]
                run_idx = np.maximum.accumulate(np.where(upd, run_idx, 0))
                cur = cost[r] + best
                back[i] = run_idx
            prev = cur
    # backtrack
    j = np.empty(n, np.int64); j[-1] = int(np.argmin(prev))
    for i in range(n - 1, 0, -1):
        j[i - 1] = back[i, j[i]]
    idx = np.arange(n)
    mdist = dist_keep[idx, j].astype(float); mbad = head_bad[idx, j]
    # sustained heading mismatch only (short runs = heading noise at turns, see notebook 02)
    sustained = np.zeros(n, bool)
    idx_bad = np.where(mbad)[0]
    if len(idx_bad):
        for r in np.split(idx_bad, np.where(np.diff(idx_bad) > 1)[0] + 1):
            if len(r) >= heading_run_min:
                sustained[r] = True
    status = np.where(mdist > label_max_dist, 1, np.where(sustained, 2, 0))
    arc = net.canonical[route[j]]
    return {"route_pos": j, "ref_arc": arc, "ref_dist": mdist, "label_status": status}
