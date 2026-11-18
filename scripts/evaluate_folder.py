"""Final-evaluation run of the DEPLOYED pipeline on every trip folder in a directory (e.g. the evaluator trips).
Usage (repo root, Laptop 1):
    python scripts/evaluate_folder.py --eval "<folder containing 00000015, 00000017, ...>" --artifacts artifacts --out reports/evaluator_run
Writes per_trip.csv + summary.md. INTEGRITY: this is evaluation only. Nothing (model, features, decoder) may be changed
because of these numbers; they are reported as they come out."""
import argparse, json, sys, time
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from api.artifacts import load_artifacts
from api.service import predict_trip, continuity
from api.evaluator import evaluate_trip

ap = argparse.ArgumentParser(); ap.add_argument("--eval", required=True); ap.add_argument("--artifacts", default="artifacts")
ap.add_argument("--out", default="reports/evaluator_run"); a = ap.parse_args()
art = load_artifacts(Path(a.artifacts), allow_placeholder=False)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
rows = []
for d in sorted(p for p in Path(a.eval).iterdir() if p.is_dir() and p.name.isdigit()):
    t0 = time.perf_counter(); r = predict_trip(d, art); m = evaluate_trip(d, r); el = time.perf_counter() - t0
    s, rc = continuity(r["lat"], r["pred"])
    rows.append({"trip_id": int(d.name), "gps_points": r["n_points"], "labelled_points": m.get("labelled_points"),
                 "unlabelled_points": m.get("unlabelled_points"), "top1_hmm": m.get("top1_accuracy_hmm"),
                 "top1_classifier_only": m.get("top1_accuracy_classifier_only"), "route_arc_f1": m.get("route_arc_f1"),
                 "continuity_strict": s, "continuity_reachable": rc, "segments": int(r["segments"].max() + 1),
                 "points_without_prediction": int((r["pred"] < 0).sum()), "seconds": round(el, 2), "reference_available": m["available"]})
    print(rows[-1], flush=True)
T = pd.DataFrame(rows); T.to_csv(out / "per_trip.csv", index=False)
L = T[T.reference_available]
pool = lambda c: float((L[c] * L.labelled_points).sum() / L.labelled_points.sum()) if len(L) else float("nan")
summ = {"run_at": datetime.now().isoformat(timespec="seconds"), "model": art.model_name, "model_status": art.model_status,
        "decoder": art.decoder, "trips": int(len(T)), "gps_points": int(T.gps_points.sum()),
        "pooled_top1_hmm": pool("top1_hmm"), "pooled_top1_classifier_only": pool("top1_classifier_only"),
        "mean_route_arc_f1": float(L.route_arc_f1.mean()) if len(L) else None,
        "mean_continuity_reachable": float(T.continuity_reachable.mean()), "max_seconds_per_trip": float(T.seconds.max())}
json.dump(summ, open(out / "summary.json", "w"), indent=1)
md = ["# Evaluation run: deployed pipeline", "", f"Folder: `{a.eval}` · {summ['run_at']} · model {art.model_name} ({art.model_status}) · decoder {art.decoder}", "",
      f"**Pooled Top-1: HMM {summ['pooled_top1_hmm']:.4f} · classifier-only {summ['pooled_top1_classifier_only']:.4f}** · mean route-arc F1 "
      f"{summ['mean_route_arc_f1']:.4f} · mean reachable continuity {summ['mean_continuity_reachable']:.4f} · slowest trip {summ['max_seconds_per_trip']:.2f} s", "",
      T.round(4).to_markdown(index=False) if hasattr(T, "to_markdown") else T.round(4).to_string(index=False), "",
      "Integrity: evaluation only; no model, feature or decoder setting was changed after seeing these numbers."]
(out / "summary.md").write_text("\n".join(md)); print("\n".join(md[:5])); print("written:", out)
