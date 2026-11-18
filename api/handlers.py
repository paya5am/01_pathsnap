"""Plain-Python request handlers (no web framework). api/main.py exposes them with FastAPI; tests can call them directly."""
import csv
import logging
import time
from datetime import datetime
from api import settings
from api.service import predict_trip, route_geometry, continuity
from api.evaluator import evaluate_trip

log = logging.getLogger("pathsnap.api")


def list_eval_trips():
    d = settings.EVAL_DIR
    if not d.is_dir():
        return []
    return [p.name for p in sorted(d.iterdir()) if p.is_dir() and p.name.isdigit() and (p / f"{p.name}.track").is_file()]


def health_payload(artifacts):
    return {"status": "ok", "eval_dir": str(settings.EVAL_DIR), "n_eval_trips": len(list_eval_trips()), **artifacts.public_info()}


def trips_payload():
    out = []
    for name in list_eval_trips():
        with open(settings.EVAL_DIR / name / f"{name}.track") as fh:
            n = sum(1 for line in fh if line.strip())
        out.append({"trip_id": name, "trip_number": int(name), "gps_points": n})
    return {"trips": out, "count": len(out)}


def predict_payload(trip_id, artifacts):
    """Raises LookupError if trip_id is not an evaluation trip."""
    names = list_eval_trips()
    valid = {t: t for t in names}; valid.update({str(int(t)): t for t in names})
    key = str(trip_id).strip()
    if key not in valid:
        raise LookupError(f"trip '{trip_id}' is not one of the evaluation trips: {[int(t) for t in names]}")
    folder = settings.EVAL_DIR / valid[key]
    a = artifacts
    log.info("predict: trip %s | model %s (%s) | decoder %s", valid[key], a.model_name, a.model_status, a.decoder)
    res = predict_trip(folder, a)                       # prediction: route file never read
    t_eval = time.perf_counter()
    metrics = evaluate_trip(folder, res)                # evaluation: server-side only, after prediction
    eval_s = time.perf_counter() - t_eval
    strict, reach = continuity(res["lat"], res["pred"])
    lat = res["lat"]
    out = {
        "trip_id": valid[key], "model_status": a.model_status, "model_name": a.model_name,
        "decoder": a.decoder, "decoder_status": a.decoder_status,
        "n_points": int(res["n_points"]), "n_candidate_rows": int(res["n_candidate_rows"]),
        "n_segments": int(res["segments"].max() + 1) if len(res["segments"]) else 0,
        "points_without_prediction": int((res["pred"] < 0).sum()),
        "gps": [[float(y), float(x)] for x, y in lat.lonlat],                       # [lat, lon]
        "predicted_route": route_geometry(lat, res["route"]),                       # connected route (HMM + path fill)
        "predicted_arc_per_point": [int(x) for x in res["pred"]],
        "metrics": {**metrics, "route_continuity_strict": strict, "route_continuity_reachable": reach},
        "timings_s": {**{k: float(v) for k, v in res["timings"].items()}, "evaluation_s": float(eval_s)},
        "warnings": list(a.warnings),
    }
    m = metrics
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else "n/a"
    log.info("RESULT trip %s | points %d | Top-1 HMM %s | classifier-only %s | route-arc F1 %s | continuity %.4f | %.2f s | model %s",
             valid[key], res["n_points"], fmt(m.get("top1_accuracy_hmm")), fmt(m.get("top1_accuracy_classifier_only")),
             fmt(m.get("route_arc_f1")), strict, res["timings"]["total_s"], a.model_status)
    _append_csv(out)
    return out


def _append_csv(out):
    settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
    f = settings.LOG_DIR / "requests.csv"
    new = not f.is_file()
    m = out["metrics"]
    with open(f, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["time", "trip_id", "model_status", "model_name", "points", "top1_hmm", "top1_classifier",
                        "route_arc_f1", "continuity_strict", "total_s"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), out["trip_id"], out["model_status"], out["model_name"],
                    out["n_points"], m.get("top1_accuracy_hmm"), m.get("top1_accuracy_classifier_only"), m.get("route_arc_f1"),
                    m.get("route_continuity_strict"), round(out["timings_s"]["total_s"], 3)])
