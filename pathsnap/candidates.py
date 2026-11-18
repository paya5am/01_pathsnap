"""Candidate generation (D7): for each GPS point, every candidate arc within min(d_nearest + DELTA, CAP)."""
import numpy as np
import pandas as pd
from .config import CAND_DELTA_M, CAND_CAP_M, DENSIFY_STEP_M
from .geo import seg_project


def generate_candidates(net, P, delta=CAND_DELTA_M, cap=CAND_CAP_M, chunk=2000):
    """P: (n,2) projected GPS points. Returns DataFrame [pt, arc, dist, t] sorted by (pt, dist),
    and per-point array d_nearest (inf if no arc within cap)."""
    n = len(P)
    d_nearest = np.full(n, np.inf)
    out = []
    if net.tree is None:
        return pd.DataFrame(columns=["pt", "arc", "dist", "t"]), d_nearest
    arcs, nxy = net.arcs, net.nxy
    for s in range(0, n, chunk):
        m_ = min(chunk, n - s)
        hits = net.tree.query_ball_point(P[s:s + m_], r=cap + DENSIFY_STEP_M / 2)
        cnt = np.fromiter((len(h) for h in hits), int, m_)
        if cnt.sum() == 0:
            continue
        pi = np.repeat(np.arange(m_), cnt)
        ai = net.tree_arc[np.concatenate([np.asarray(h, np.int64) for h in hits if len(h)])]
        key = np.unique(pi.astype(np.int64) * 10_000_000 + ai)          # unique (point, arc) pairs
        pi, ai = key // 10_000_000, key % 10_000_000
        d, t = seg_project(P[s + pi], nxy[arcs[ai, 0]], nxy[arcs[ai, 1]])
        ok = d <= cap
        pi, ai, d, t = pi[ok], ai[ok], d[ok], t[ok]
        mn = np.full(m_, np.inf); np.minimum.at(mn, pi, d)
        d_nearest[s:s + m_] = mn
        keep = d <= np.minimum(mn[pi] + delta, cap)
        out.append(pd.DataFrame({"pt": pi[keep] + s, "arc": ai[keep], "dist": d[keep], "t": t[keep]}))
    cands = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["pt", "arc", "dist", "t"])
    cands = cands.sort_values(["pt", "dist"], kind="mergesort").reset_index(drop=True)
    return cands, d_nearest
