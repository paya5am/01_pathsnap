"""Per-GPS-point motion descriptors computed from the track only (no road network, no labels)."""
import numpy as np
from .config import HEADING_LAG, MOVING_SPEED_MS, SPEED_CLIP_MS, GAP_S
from .geo import bearing_deg, angle_diff_deg


def motion_features(P, t):
    """P: (n,2) metres, t: (n,) seconds. Returns dict of per-point arrays."""
    n = len(P); L = HEADING_LAG
    heading = np.full(n, np.nan); speed = np.full(n, np.nan)
    if n > 2 * L:
        v = P[2 * L:] - P[:-2 * L]
        dt = np.maximum(t[2 * L:] - t[:-2 * L], 1e-6)
        sp = np.linalg.norm(v, axis=1) / dt
        speed[L:-L] = sp
        heading[L:-L] = bearing_deg(v)
    # edges: copy nearest defined value
    if n > 2 * L:
        speed[:L], speed[-L:] = speed[L], speed[-L - 1]
        heading[:L], heading[-L:] = heading[L], heading[-L - 1]
    else:
        speed[:] = 0.0
    # gap-aware: a heading spanning a long gap is meaningless
    dt_prev = np.r_[np.nan, np.diff(t)]; dt_next = np.r_[np.diff(t), np.nan]
    span_gap = np.zeros(n, bool)
    for k in range(1, L + 1):
        span_gap |= np.r_[np.zeros(k, bool), (t[k:] - t[:-k]) > GAP_S * k]
        span_gap |= np.r_[(t[k:] - t[:-k]) > GAP_S * k, np.zeros(k, bool)]
    moving = (speed >= MOVING_SPEED_MS) & ~span_gap & np.isfinite(heading)
    heading_known = moving.copy()
    # turn rate: change of heading between the two halves of the window (deg)
    turn = np.full(n, np.nan)
    if n > 4 * L:
        h1 = bearing_deg(P[2 * L:-2 * L] - P[:-4 * L]); h2 = bearing_deg(P[4 * L:] - P[2 * L:-2 * L])
        turn[2 * L:-2 * L] = angle_diff_deg(h1, h2)
    return {"heading": heading, "heading_known": heading_known, "moving": moving,
            "speed": np.clip(np.nan_to_num(speed, nan=0.0), 0, SPEED_CLIP_MS),
            "turn": np.nan_to_num(turn, nan=0.0),
            "dt_prev": np.nan_to_num(dt_prev, nan=0.0), "dt_next": np.nan_to_num(dt_next, nan=0.0)}
