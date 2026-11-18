"""Reading one trip folder. Format verified in notebook 01 (§2, §3, V1, V2)."""
import os
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd


@dataclass
class RawTrip:
    trip_id: int
    track: pd.DataFrame            # lon, lat, t
    nodes: pd.DataFrame            # lon, lat   (row index = node id)
    arcs: np.ndarray               # (n_arcs, 2) from_node, to_node (row index = arc id)
    route: Optional[np.ndarray]    # ordered reference arc ids, or None (evaluation must not need it)


def _rd(path, names):
    return pd.read_csv(path, sep=r"\s+", header=None, names=names)


def read_trip(trip_dir: str, with_route: bool = True) -> RawTrip:
    """trip_dir = folder like .../train/00000001 containing 00000001.track/.nodes/.arcs/.route."""
    tid = os.path.basename(os.path.normpath(trip_dir))
    base = os.path.join(trip_dir, tid)
    track = _rd(base + ".track", ["lon", "lat", "t"]).astype(float)
    nodes = _rd(base + ".nodes", ["lon", "lat"]).astype(float)
    arcs = _rd(base + ".arcs", ["u", "v"]).values.astype(np.int64)
    route = None
    if with_route and os.path.isfile(base + ".route"):
        route = _rd(base + ".route", ["arc"]).arc.values.astype(np.int64)
    return RawTrip(int(tid), track, nodes, arcs, route)


def list_trips(data_root: str, partition: str):
    d = os.path.join(data_root, partition)
    return sorted(int(x) for x in os.listdir(d) if os.path.isdir(os.path.join(d, x)))


def trip_dir(data_root: str, partition: str, trip_id: int) -> str:
    return os.path.join(data_root, partition, f"{trip_id:08d}")
