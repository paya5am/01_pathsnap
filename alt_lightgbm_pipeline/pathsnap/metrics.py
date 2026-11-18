"""Metrics from Models & Training Guidelines §5.

Top-1 and candidate miss rate are PER GPS POINT (only labelled points count).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score)


def argmax_per_point(point: np.ndarray, score: np.ndarray, n_points: int) -> np.ndarray:
    """Row index of the best-scoring candidate for each point (rows grouped by point)."""
    o = np.lexsort((-score, point))
    first = np.r_[True, point[o][1:] != point[o][:-1]]
    best = np.full(n_points, -1, np.int64)
    best[point[o][first]] = o[first]
    return best


def row_metrics(y, p, thr=0.5) -> dict:
    yh = (p >= thr).astype(int)
    return dict(accuracy=accuracy_score(y, yh), precision=precision_score(y, yh, zero_division=0),
                recall=recall_score(y, yh, zero_division=0), f1=f1_score(y, yh, zero_division=0),
                log_loss=log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]),
                confusion=confusion_matrix(y, yh, labels=[0, 1]).tolist())


def point_metrics(group: np.ndarray, point: np.ndarray, y: np.ndarray, score: np.ndarray,
                  point_labelled: np.ndarray) -> dict:
    """Top-1 accuracy and candidate miss rate over labelled points.
    group = trip id per row; a point key is (trip, point)."""
    key = group.astype(np.int64) * 10_000_000 + point
    uk, inv = np.unique(key, return_inverse=True)
    best = argmax_per_point(inv, score, len(uk))
    has_pos = np.zeros(len(uk), bool); has_pos[inv[y == 1]] = True
    lab = np.zeros(len(uk), bool); lab[inv[point_labelled]] = True
    correct = y[best] == 1
    return dict(top1=float(correct[lab].mean()), miss_rate=float(1 - has_pos[lab].mean()),
                top1_given_covered=float(correct[lab & has_pos].mean()), n_points=int(lab.sum()))


def route_continuity(arcs_seq: np.ndarray, t: np.ndarray, net, gap_s: float = 60.0,
                     slack: float = 50.0) -> float:
    """Share of consecutive point pairs (no big gap) whose predicted arcs are connected:
    same arc, or arc j reachable from the end of arc i within max(slack, 3 x GPS step) m."""
    ok = tot = 0
    for k in range(len(arcs_seq) - 1):
        a, b = arcs_seq[k], arcs_seq[k + 1]
        if a < 0 or b < 0 or t[k + 1] - t[k] > gap_s:
            continue
        tot += 1
        if a == b:
            ok += 1; continue
        lim = slack
        dist = net.dists_from(int(net.to[a]), lim)
        if dist.get(int(net.frm[b]), 1e18) <= lim:   # cache may hold a larger search
            ok += 1
    return ok / tot if tot else float("nan")


def collapse(seq):
    seq = [int(a) for a in seq if a >= 0]
    out = []
    for a in seq:
        if not out or out[-1] != a:
            out.append(a)
    return out


def sequence_metrics(pred_seq, ref_route, canon) -> dict:
    """Route-level precision/recall/F1 over arcs, using the LCS of the collapsed predicted
    arc sequence and the reference route (duplicate twins mapped to one id)."""
    p = collapse(pred_seq)
    r = collapse(canon[np.asarray(ref_route)])
    if not p or not r:
        return dict(seq_precision=0.0, seq_recall=0.0, seq_f1=0.0)
    # LCS via DP on bands would be faster; route lengths here are <= a few thousand
    import numpy as _np
    n, m = len(p), len(r)
    prev = _np.zeros(m + 1, _np.int32)
    rr = _np.asarray(r)
    for i in range(n):
        cur = _np.zeros(m + 1, _np.int32)
        eq = (rr == p[i])
        # cur[j+1] = prev[j]+1 if eq else max(prev[j+1], cur[j])
        diag = prev[:-1] + eq
        best = _np.maximum(prev[1:], diag)
        cur[1:] = _np.maximum.accumulate(best)  # valid since LCS rows are non-decreasing
        prev = cur
    lcs = int(prev[-1])
    P, R = lcs / n, lcs / m
    return dict(seq_precision=P, seq_recall=R, seq_f1=2 * P * R / (P + R) if P + R else 0.0)
