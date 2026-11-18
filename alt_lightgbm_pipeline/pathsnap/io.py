"""Reading PathSnap trip folders.

Each trip folder <root>/<partition>/<id>/ holds <id>.track, .nodes, .arcs and
(for train / student_test) .route. Files have no header and are whitespace separated.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Trip:
    trip_id: int
    track: np.ndarray          # (T, 3) lon, lat, t_sec
    nodes: np.ndarray          # (N, 2) lon, lat ; row index = node id
    arcs: np.ndarray           # (A, 2) from_node, to_node ; row index = arc id
    route: Optional[np.ndarray]  # (R,) ordered reference arc ids, None if absent


def _load(path: str, ncols: int, dtype) -> np.ndarray:
    arr = np.loadtxt(path, dtype=dtype, ndmin=2)
    if arr.size == 0:
        return np.zeros((0, ncols), dtype=dtype)
    return arr.reshape(-1, ncols)


def trip_folder(root: str, partition: str, trip_id: int) -> str:
    return os.path.join(root, partition, f"{trip_id:08d}")


def list_trips(root: str, partition: str) -> list[int]:
    d = os.path.join(root, partition)
    return sorted(int(x) for x in os.listdir(d) if x.isdigit())


def load_trip_folder(folder: str, with_route: bool = True) -> Trip:
    tid = os.path.basename(os.path.normpath(folder))
    p = lambda ext: os.path.join(folder, f"{tid}.{ext}")
    track = _load(p("track"), 3, float)
    nodes = _load(p("nodes"), 2, float)
    arcs = _load(p("arcs"), 2, np.int64)
    route = None
    if with_route and os.path.exists(p("route")):
        route = np.loadtxt(p("route"), dtype=np.int64, ndmin=1)
    return Trip(int(tid), track, nodes, arcs, route)


def load_trip(root: str, partition: str, trip_id: int, with_route: bool = True) -> Trip:
    return load_trip_folder(trip_folder(root, partition, trip_id), with_route)
