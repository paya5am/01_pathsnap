"""Metric projection, arc geometry and the directed road graph for one trip.

Decisions used here (CONTEXT_LOG): D5 local equirectangular projection,
D11 zero-length arcs dropped from candidates, duplicate twins mapped to one id.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np

R_EARTH = 6371008.8


@dataclass
class Projection:
    lon0: float
    lat0: float

    def __call__(self, lon, lat):
        k = np.pi / 180.0
        x = (np.asarray(lon) - self.lon0) * k * R_EARTH * np.cos(self.lat0 * k)
        y = (np.asarray(lat) - self.lat0) * k * R_EARTH
        return x, y


def make_projection(track: np.ndarray) -> Projection:
    """Centred on the trip's mean GPS position (only the track, never the route)."""
    return Projection(float(track[:, 0].mean()), float(track[:, 1].mean()))


@dataclass
class Network:
    nx: np.ndarray; ny: np.ndarray          # node coords (m)
    frm: np.ndarray; to: np.ndarray         # arc endpoints
    ax: np.ndarray; ay: np.ndarray          # arc start (m)
    bx: np.ndarray; by: np.ndarray          # arc end (m)
    length: np.ndarray                      # arc length (m)
    bearing: np.ndarray                     # arc bearing, radians, atan2(dy, dx)
    canon: np.ndarray                       # canonical id (duplicate twins -> smallest id)
    valid: np.ndarray                       # usable as candidate (canonical & non-zero length)
    two_way: np.ndarray                     # reverse arc exists
    indptr: list = field(repr=False, default=None)    # CSR over nodes (outgoing arcs)
    out_arcs: list = field(repr=False, default=None)
    _sp_cache: dict = field(repr=False, default_factory=dict)

    # ---------- shortest paths (bounded Dijkstra, node -> node) ----------
    def dists_from(self, src: int, limit: float) -> dict:
        """Network distances (m) from node src to every node within `limit` metres.
        Results are cached per source with the largest limit seen so far."""
        hit = self._sp_cache.get(src)
        if hit is not None and hit[0] >= limit:
            return hit[1]
        dist = {src: 0.0}
        heap = [(0.0, src)]
        indptr, out_arcs, to, L = self.indptr, self.out_arcs, self.to_l, self.len_l
        done = set()
        while heap:
            d, u = heapq.heappop(heap)
            if u in done:
                continue
            done.add(u)
            for k in range(indptr[u], indptr[u + 1]):
                a = out_arcs[k]
                nd = d + L[a]
                if nd > limit:
                    continue
                v = to[a]
                if nd < dist.get(v, 1e18):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        if len(self._sp_cache) > 200000:
            self._sp_cache.clear()
        self._sp_cache[src] = (limit, dist)
        return dist


def build_network(nodes: np.ndarray, arcs: np.ndarray, proj: Projection) -> Network:
    nx, ny = proj(nodes[:, 0], nodes[:, 1])
    frm = arcs[:, 0].astype(np.int64)
    to = arcs[:, 1].astype(np.int64)
    ax, ay, bx, by = nx[frm], ny[frm], nx[to], ny[to]
    length = np.hypot(bx - ax, by - ay)
    bearing = np.arctan2(by - ay, bx - ax)

    # duplicate twins: same (from, to) pair -> smallest arc id
    key = frm * (len(nodes) + 1) + to
    order = np.lexsort((np.arange(len(key)), key))
    ks = key[order]
    first = np.ones(len(ks), bool)
    first[1:] = ks[1:] != ks[:-1]
    grp = np.cumsum(first) - 1
    canon_sorted = order[first][grp]
    canon = np.empty(len(key), np.int64)
    canon[order] = canon_sorted

    valid = (canon == np.arange(len(canon))) & (length > 0.01) & (frm != to)
    rev_key = to * (len(nodes) + 1) + frm
    two_way = np.isin(rev_key, key)

    # CSR of outgoing usable arcs (zero-length arcs kept for connectivity: they cost 0)
    use = (canon == np.arange(len(canon))) & (frm != to)
    ids = np.nonzero(use)[0]
    o = ids[np.argsort(frm[ids], kind="stable")]
    counts = np.bincount(frm[o], minlength=len(nodes))
    indptr = np.concatenate([[0], np.cumsum(counts)])

    net = Network(nx, ny, frm, to, ax, ay, bx, by, length, bearing, canon, valid, two_way)
    net.indptr = indptr.tolist()
    net.out_arcs = o.tolist()
    net.to_l = to.tolist()
    net.len_l = length.tolist()
    return net


def point_segment(px, py, net: Network, arc_ids):
    """Distance from points to arcs plus the projection offset along the arc (m)."""
    ax, ay = net.ax[arc_ids], net.ay[arc_ids]
    dx, dy = net.bx[arc_ids] - ax, net.by[arc_ids] - ay
    L2 = dx * dx + dy * dy
    u = np.where(L2 > 0, ((px - ax) * dx + (py - ay) * dy) / np.where(L2 > 0, L2, 1), 0.0)
    u = np.clip(u, 0.0, 1.0)
    qx, qy = ax + u * dx, ay + u * dy
    d = np.hypot(px - qx, py - qy)
    return d, u * np.sqrt(L2)


def shortest_path_arcs(net: Network, src: int, dst: int, limit: float):
    """Arc ids of the shortest directed path src -> dst (None if longer than limit)."""
    if src == dst:
        return []
    dist = {src: 0.0}; pred = {}
    heap = [(0.0, src)]
    done = set()
    while heap:
        d, u = heapq.heappop(heap)
        if u == dst:
            break
        if u in done:
            continue
        done.add(u)
        for k in range(net.indptr[u], net.indptr[u + 1]):
            a = net.out_arcs[k]
            nd = d + net.len_l[a]
            if nd > limit:
                continue
            v = net.to_l[a]
            if nd < dist.get(v, 1e18):
                dist[v] = nd; pred[v] = a
                heapq.heappush(heap, (nd, v))
    if dst not in pred:
        return None
    path = []
    v = dst
    while v != src:
        a = pred[v]; path.append(a); v = net.frm[a]
    return path[::-1]
