"""Projected road network for one trip + spatial index for candidate search."""
from dataclasses import dataclass, field
import numpy as np
from scipy.spatial import cKDTree
from .config import MIN_ARC_LEN_M, DENSIFY_STEP_M, CAND_CAP_M
from .geo import make_projection, bearing_deg


@dataclass
class Network:
    to_xy: object
    nxy: np.ndarray            # (n_nodes, 2) metres
    arcs: np.ndarray           # (n_arcs, 2) original arc table
    length: np.ndarray         # (n_arcs,) metres
    bearing: np.ndarray        # (n_arcs,) degrees
    canonical: np.ndarray      # (n_arcs,) arc id that represents this (u,v) pair (lowest id among exact duplicates)
    cand_arcs: np.ndarray      # arc ids that may be candidates: canonical and length >= MIN_ARC_LEN_M
    two_way: np.ndarray        # (n_arcs,) bool: reverse (v,u) also exists as a candidate arc
    out_deg: np.ndarray        # (n_nodes,) number of candidate arcs leaving the node
    in_deg: np.ndarray         # (n_nodes,) number of candidate arcs entering the node
    tree: cKDTree = None       # KD-tree over densified candidate arcs (built for an area of interest)
    tree_arc: np.ndarray = field(default=None)   # arc id of every densified point


def build_network(nodes_lonlat, arcs, lon0, lat0, aoi_xy=None, margin=CAND_CAP_M + 50):
    """nodes_lonlat: DataFrame(lon, lat). arcs: (n,2). (lon0, lat0): projection centre (use the GPS-track mean).
    aoi_xy: points (e.g. the projected GPS track); only arcs whose bounding box is within `margin` of
    the AOI bounding box are indexed (the networks have up to 307k arcs, most far from the track)."""
    to_xy = make_projection(lon0, lat0)
    nxy = to_xy(nodes_lonlat.lon.values, nodes_lonlat.lat.values)
    A, B = nxy[arcs[:, 0]], nxy[arcs[:, 1]]
    length = np.linalg.norm(B - A, axis=1)
    bearing = bearing_deg(B - A)
    # canonical id: lowest arc id among identical (u, v) pairs
    order = np.lexsort((np.arange(len(arcs)), arcs[:, 1], arcs[:, 0]))
    s = arcs[order]
    new_grp = np.r_[True, (s[1:] != s[:-1]).any(1)]
    grp_first = order[np.maximum.accumulate(np.where(new_grp, np.arange(len(s)), 0))]
    canonical = np.empty(len(arcs), np.int64); canonical[order] = grp_first
    is_cand = (canonical == np.arange(len(arcs))) & (length >= MIN_ARC_LEN_M)
    cand_arcs = np.where(is_cand)[0]
    n_nodes = len(nxy)
    key = arcs[:, 0] * n_nodes + arcs[:, 1]; rkey = arcs[:, 1] * n_nodes + arcs[:, 0]
    two_way = np.isin(rkey, key[cand_arcs]) & is_cand
    out_deg = np.bincount(arcs[cand_arcs, 0], minlength=n_nodes)
    in_deg = np.bincount(arcs[cand_arcs, 1], minlength=n_nodes)
    net = Network(to_xy, nxy, arcs, length, bearing, canonical, cand_arcs, two_way, out_deg, in_deg)
    build_index(net, aoi_xy, margin)
    return net


def build_index(net: Network, aoi_xy=None, margin=CAND_CAP_M + 50, step=DENSIFY_STEP_M):
    idx = net.cand_arcs
    A, B = net.nxy[net.arcs[idx, 0]], net.nxy[net.arcs[idx, 1]]
    if aoi_xy is not None:
        lo, hi = aoi_xy.min(0) - margin, aoi_xy.max(0) + margin
        m = (np.maximum(A, B) >= lo).all(1) & (np.minimum(A, B) <= hi).all(1)
        idx, A, B = idx[m], A[m], B[m]
    L = np.linalg.norm(B - A, axis=1)
    k = np.maximum(1, np.ceil(L / step).astype(int))
    rep = np.repeat(np.arange(len(idx)), k + 1)
    t = np.concatenate([np.linspace(0, 1, kk + 1) for kk in k]) if len(k) else np.zeros(0)
    pts = A[rep] + t[:, None] * (B - A)[rep]
    net.tree = cKDTree(pts) if len(pts) else None
    net.tree_arc = idx[rep]
    return net
