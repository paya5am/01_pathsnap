"""Feature engineering for (GPS point, candidate arc) pairs.
Every feature uses ONLY the GPS track and the road network — never the reference route — so the identical code
runs at evaluation time on the hidden trips. Features are deterministic (nothing is fitted here); scaling is
fitted later inside each model's pipeline on training trips only."""
import numpy as np
import pandas as pd
from .config import WINDOW_S
from .geo import seg_project, angle_diff_deg

# Final model features (16). Dropped after the train-only analysis in notebook 02 §6:
#   heading_diff (duplicate of cos_heading, r = -1.00), win_dist_max (r = 0.91 with win_dist_mean),
#   win_n, proj_t, dt_prev, dt_next (no signal alone, separation ≈ 0.50; gaps are handled by the HMM in M7).
FEATURES = [
    "dist", "dist_rel", "dist_ratio", "dist_rank", "n_cands", "d_nearest",
    "cos_heading", "heading_known", "speed", "turn",
    "arc_len_log", "dist_to_end_m", "two_way", "from_out_deg", "to_out_deg",
    "win_dist_mean",
]

# Extra candidate features adopted from member C's pipeline (M8 review): kept as columns in every dataset; which list the
# model uses is recorded in the exported manifest. Definitions mirror C's stand-in features.py.
C_EXTRA = ["proj_t", "dist_prev3", "dist_next3", "head_diff_raw", "head_diff_x_moving2"]
ALL_COLUMNS = FEATURES + C_EXTRA


def candidate_features(net, P, t, motion, cands, d_nearest, window_s=WINDOW_S, max_k=2):
    c = cands
    pt, arc = c.pt.values, c.arc.values
    f = pd.DataFrame({"pt": pt, "arc": arc})
    f["dist"] = c.dist.values
    f["dist_rel"] = c.dist.values - d_nearest[pt]                       # 0 for the nearest arc
    f["dist_ratio"] = (c.dist.values + 1.0) / (d_nearest[pt] + 1.0)
    f["dist_rank"] = c.groupby("pt").cumcount().values                    # cands are sorted by (pt, dist)
    f["n_cands"] = c.groupby("pt").pt.transform("size").values
    f["d_nearest"] = d_nearest[pt]
    # direction agreement between vehicle and arc (the key feature for two-way roads, D8)
    hk = motion["heading_known"][pt]
    hd = angle_diff_deg(motion["heading"][pt], net.bearing[arc])
    f["cos_heading"] = np.where(hk, np.cos(np.radians(hd)), 0.0)          # +1 same way, -1 opposite, 0 unknown
    f["heading_known"] = hk.astype(np.int8)
    f["speed"] = motion["speed"][pt]
    f["turn"] = motion["turn"][pt]
    L = net.length[arc]
    f["arc_len_log"] = np.log1p(L)
    f["dist_to_end_m"] = np.minimum(c.t.values, 1 - c.t.values) * L      # near a junction → ambiguous
    f["two_way"] = net.two_way[arc].astype(np.int8)
    # --- C's extra features ---
    f["proj_t"] = c.t.values                                             # position of the projection along the arc (0..1)
    n_ = len(P)
    for name, k in (("dist_prev3", -3), ("dist_next3", 3)):              # where the fix 3 samples earlier / later lies
        q = np.clip(pt + k, 0, n_ - 1)
        d_k, _ = seg_project(P[q], net.nxy[net.arcs[arc, 0]], net.nxy[net.arcs[arc, 1]])
        f[name] = d_k
    hd_raw = angle_diff_deg(np.nan_to_num(motion["heading"][pt], nan=0.0), net.bearing[arc])   # even when slow
    f["head_diff_raw"] = hd_raw
    f["head_diff_x_moving2"] = hd_raw * (motion["speed"][pt] >= 2.0)
    f["from_out_deg"] = net.out_deg[net.arcs[arc, 0]]
    f["to_out_deg"] = net.out_deg[net.arcs[arc, 1]]
    # temporal context: how close do the NEIGHBOURING GPS points (within ±window_s, gap-aware) pass to this arc?
    A, B = net.nxy[net.arcs[arc, 0]], net.nxy[net.arcs[arc, 1]]
    s_sum = np.zeros(len(f)); s_n = np.zeros(len(f))
    n = len(P)
    for k in range(-max_k, max_k + 1):
        if k == 0:
            continue
        q = pt + k
        ok = (q >= 0) & (q < n)
        q = np.clip(q, 0, n - 1)
        ok &= np.abs(t[q] - t[pt]) <= window_s + 1e-6
        if not ok.any():
            continue
        d, _ = seg_project(P[q[ok]], A[ok], B[ok])
        s_sum[ok] += d; s_n[ok] += 1
    f["win_dist_mean"] = np.where(s_n > 0, (s_sum + c.dist.values) / (s_n + 1), c.dist.values)
    return f
