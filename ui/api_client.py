"""Thin HTTP client for the PathSnap API (Laptop 1). The UI only ever talks to the API: no model, no data here."""
import requests


class ApiError(Exception):
    pass


def _get(base, path, timeout):
    try:
        r = requests.get(base.rstrip("/") + path, timeout=timeout)
    except requests.RequestException as e:
        raise ApiError(f"Cannot reach the API at {base}: {e}") from e
    if r.status_code != 200:
        raise ApiError(f"{path} returned HTTP {r.status_code}: {r.text[:300]}")
    return r.json()


def health(base, timeout=5):
    return _get(base, "/health", timeout)


def trips(base, timeout=10):
    return _get(base, "/trips", timeout)["trips"]


def predict(base, trip_id, timeout=120):
    try:
        r = requests.post(base.rstrip("/") + "/predict", json={"trip_id": str(trip_id)}, timeout=timeout)
    except requests.RequestException as e:
        raise ApiError(f"Cannot reach the API at {base}: {e}") from e
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise ApiError(f"/predict returned HTTP {r.status_code}: {str(detail)[:300]}")
    return r.json()
