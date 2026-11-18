"""Geometry helpers: per-trip metric projection, bearings, point-to-segment distances."""
import math
import numpy as np
from .config import R_EARTH


def make_projection(lon0: float, lat0: float):
    """Local equirectangular projection centred on (lon0, lat0) → metres. Validated in notebook 01 §4."""
    c = math.cos(math.radians(lat0))

    def to_xy(lon, lat):
        lon = np.asarray(lon, float); lat = np.asarray(lat, float)
        return np.column_stack([np.radians(lon - lon0) * R_EARTH * c, np.radians(lat - lat0) * R_EARTH])
    return to_xy


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))


def bearing_deg(v):
    """Compass bearing of vectors v (..., 2) in degrees: 0 = north, 90 = east."""
    v = np.asarray(v, float)
    return np.degrees(np.arctan2(v[..., 0], v[..., 1])) % 360


def angle_diff_deg(a, b):
    """Smallest absolute difference between two bearings (0..180)."""
    return np.abs((np.asarray(a) - np.asarray(b) + 180) % 360 - 180)


def seg_project(P, A, B):
    """Project points P onto segments A→B (broadcasting). Returns (distance, t in [0,1])."""
    AB = B - A; AP = P - A
    t = np.clip((AP * AB).sum(-1) / np.maximum((AB ** 2).sum(-1), 1e-12), 0.0, 1.0)
    d = np.linalg.norm(AP - t[..., None] * AB, axis=-1)
    return d, t
