"""M7: HMM + Viterbi route reconstruction.

Hidden states  : at GPS point t, the candidate arcs of t (same candidate rule as training, D7). Candidate ↔ arc id is kept.
Emissions      : log P(candidate is the true arc | features) from the selected classifier (hist_gb), times weight `lam`.
Transitions    : Newson & Krumm (2009): a vehicle's route distance between two consecutive matches should be close to the
                 straight-line distance between the two GPS fixes.  log T(a→b) = −|d_route(a,b) − d_gps| / beta_t,
                 beta_t = beta · max(1, dt)^gap_exp  (gaps allow more slack).
                 d_route is computed on the DIRECTED road graph (Dijkstra): from the projection on a, to the end of a,
                 shortest path to the start of b, then to the projection on b. Same arc: forward distance; a small backward
                 move (≤ back_tol m, GPS jitter while crawling) is allowed. Unreachable within a search limit → impossible.
Gaps / breaks  : if dt > break_s, or no candidate of t+1 is reachable from any surviving state of t, the chain is cut and a
                 new segment starts (decoded independently). Points with no candidates get no prediction (-1).
Decoding       : Viterbi in log space per segment: delta_t+1(b) = lam·log e_t+1(b) + max_a [delta_t(a) + log T(a→b)].
"""
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from .io import read_trip
from .geo import make_projection
from .network import build_network
from .candidates import generate_candidates

LIMIT_BUCKETS = (250.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0)


class RouteGraph:
    """Directed graph of the trip network (edge = arc length; duplicates → shortest; zero-length kept as 1 mm)."""
    def __init__(self, net):
        a = net.arcs; L = np.maximum(net.length, 1e-3)
        keep = a[:, 0] != a[:, 1]
        df = pd.DataFrame({"u": a[keep, 0], "v": a[keep, 1], "w": L[keep]}).groupby(["u", "v"], as_index=False).w.min()
        n = len(net.nxy)
        self.G = coo_matrix((df.w.values, (df.u.values, df.v.values)), shape=(n, n)).tocsr()
        self.cache = {}

    def dist_from(self, node, limit):
        b = next((x for x in LIMIT_BUCKETS if x >= limit), LIMIT_BUCKETS[-1])
        key = (int(node), b)
        if key not in self.cache:
            d = dijkstra(self.G, directed=True, indices=int(node), limit=b)
            idx = np.where(np.isfinite(d))[0]
            self.cache[key] = dict(zip(idx.tolist(), d[idx].tolist()))
        return self.cache[key]

    def path(self, s, t, limit=10000.0):
        """Node path s→t (for drawing / filling the reconstructed route)."""
        d, pred = dijkstra(self.G, directed=True, indices=int(s), limit=limit, return_predecessors=True)
        if not np.isfinite(d[t]): return None
        out = [int(t)]
        while out[-1] != s:
            out.append(int(pred[out[-1]]))
        return out[::-1]


class TripLattice:
    """Everything the decoder needs for one trip, precomputed once (independent of beta / lam / break_s).
    TripLattice(trip_dir, proba_lookup)      — reads the trip itself (notebooks)
    TripLattice.from_parts(...)              — reuses network/candidates already computed (API: no double work)"""
    def __init__(self, trip_dir, proba_lookup, back_tol=15.0, max_dt_for_transitions=600.0):
        raw = read_trip(trip_dir, with_route=False); tr = raw.track
        lon0, lat0 = tr.lon.mean(), tr.lat.mean()
        P = make_projection(lon0, lat0)(tr.lon.values, tr.lat.values)
        net = build_network(raw.nodes, raw.arcs, lon0, lat0, aoi_xy=P)
        cands, _ = generate_candidates(net, P)
        self._setup(raw.trip_id, net, P, tr.t.values, tr[["lon", "lat"]].values, raw.nodes[["lon", "lat"]].values,
                    cands, proba_lookup, back_tol, max_dt_for_transitions)

    @classmethod
    def from_parts(cls, trip_id, net, P, t, lonlat, nodes_lonlat, cands, proba_lookup, back_tol=15.0, max_dt_for_transitions=600.0):
        obj = cls.__new__(cls)
        obj._setup(trip_id, net, P, t, lonlat, nodes_lonlat, cands, proba_lookup, back_tol, max_dt_for_transitions)
        return obj

    def _setup(self, trip_id, net, P, t, lonlat, nodes_lonlat, cands, proba_lookup, back_tol, max_dt_for_transitions):
        self.trip_id, self.net, self.P, self.t = trip_id, net, P, np.asarray(t, float)
        self.lonlat = np.asarray(lonlat); self.nodes_lonlat = np.asarray(nodes_lonlat)
        self.graph = RouteGraph(net)
        n = len(P)
        starts = np.searchsorted(cands.pt.values, np.arange(n + 1))
        self.cand_arc = [cands.arc.values[starts[i]:starts[i + 1]] for i in range(n)]
        self.cand_t = [cands.t.values[starts[i]:starts[i + 1]] for i in range(n)]
        # emission probabilities from the classifier (lookup: (pt, arc) -> proba)
        self.logp = []
        for i in range(n):
            p = np.array([proba_lookup.get((i, int(a)), 1e-6) for a in self.cand_arc[i]], float)
            self.logp.append(np.log(np.clip(p, 1e-6, 1.0)))
        self.dt = np.r_[np.diff(self.t), np.inf]
        self.dgps = np.r_[np.linalg.norm(np.diff(P, axis=0), axis=1), np.inf]
        # |d_route - d_gps| for every consecutive pair (inf = impossible / not computed)
        self.cost = []
        Lr = net.length; A = net.arcs
        for i in range(n - 1):
            ka, kb = self.cand_arc[i], self.cand_arc[i + 1]
            if len(ka) == 0 or len(kb) == 0 or self.dt[i] > max_dt_for_transitions:
                self.cost.append(None); continue
            limit = 3.0 * self.dgps[i] + 200.0
            M = np.full((len(ka), len(kb)), np.inf, np.float32)
            for x, (a, ta) in enumerate(zip(ka, self.cand_t[i])):
                rem = (1.0 - ta) * Lr[a]
                reach = self.graph.dist_from(A[a, 1], limit)
                for y, (b, tb) in enumerate(zip(kb, self.cand_t[i + 1])):
                    if a == b:
                        fwd = (tb - ta) * Lr[a]
                        if fwd >= -back_tol:
                            M[x, y] = abs(abs(fwd) - self.dgps[i]); continue
                    sp = reach.get(int(A[b, 0]))
                    if sp is not None:
                        M[x, y] = abs(rem + sp + tb * Lr[b] - self.dgps[i])
            self.cost.append(M)


def viterbi(lat, beta=10.0, lam=1.0, break_s=60.0, gap_exp=0.5):
    """Returns (pred_arc per point, -1 where no candidates; segment id per point)."""
    n = len(lat.cand_arc); pred = np.full(n, -1, np.int64); seg = np.full(n, -1, np.int64)
    s_id = -1; delta = None; back = []; seg_pts = []

    def flush():
        if not seg_pts: return
        j = int(np.argmax(delta))
        for k in range(len(seg_pts) - 1, -1, -1):
            pred[seg_pts[k]] = lat.cand_arc[seg_pts[k]][j]
            if k > 0: j = back[k][j]

    for i in range(n):
        if len(lat.cand_arc[i]) == 0:
            flush(); delta, back, seg_pts = None, [], []; continue
        e = lam * lat.logp[i]
        M = lat.cost[i - 1] if (i > 0 and delta is not None and lat.dt[i - 1] <= break_s) else None
        if M is not None:
            bt = beta * max(1.0, lat.dt[i - 1]) ** gap_exp
            sc = delta[:, None] - M / bt                       # (Ka, Kb)
            best = sc.max(0)
            if np.isfinite(best).any():
                arg = sc.argmax(0); nd = e + best
                nd[~np.isfinite(best)] = -np.inf             # unreachable candidates die
                delta = nd; back.append(arg); seg_pts.append(i); seg[i] = s_id; continue
        flush()                                              # break: close previous segment, start a new one
        s_id += 1; delta = e.copy(); back = [None]; seg_pts = [i]; seg[i] = s_id
    flush()
    return pred, seg


def reconstruct_route(lat, pred):
    """Connected arc sequence: decoded arcs, with shortest-path arcs filled between consecutive different arcs."""
    A = lat.net.arcs
    uv_to_arc = {(int(u), int(v)): int(k) for k, (u, v) in enumerate(A) if lat.net.canonical[k] == k}
    route, prev = [], None
    for a in pred:
        if a < 0: prev = None; continue
        a = int(a)
        if prev is None or a == prev:
            if not route or route[-1] != a: route.append(a)
            prev = a; continue
        if A[prev, 1] != A[a, 0]:
            nodes = lat.graph.path(A[prev, 1], A[a, 0], limit=5000.0)
            if nodes:
                for u, v in zip(nodes[:-1], nodes[1:]):
                    k = uv_to_arc.get((u, v))
                    if k is not None and (not route or route[-1] != k): route.append(k)
        if not route or route[-1] != a: route.append(a)
        prev = a
    return route
