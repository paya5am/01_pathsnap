"""Offline test of the backend logic WITHOUT FastAPI/HTTP: runs predict + evaluate on every trip in an eval folder.
Usage (repo root):  python scripts/test_service_offline.py "<folder with trip folders>" [artifacts_dir]"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api.artifacts import load_artifacts
from api.service import predict_trip, continuity
from api.evaluator import evaluate_trip
eval_dir = Path(sys.argv[1]); art = load_artifacts(Path(sys.argv[2]) if len(sys.argv) > 2 else Path("artifacts"))
print("model:", art.model_name, art.model_status, "| decoder:", art.decoder, art.decoder_status)
tot_ok = tot_lab = tot_clf = 0
for d in sorted(p for p in eval_dir.iterdir() if p.is_dir()):
    r = predict_trip(d, art); m = evaluate_trip(d, r); s, rc = continuity(r["lat"], r["pred"])
    tot_lab += m["labelled_points"]; tot_ok += round(m["top1_accuracy_hmm"] * m["labelled_points"]); tot_clf += round(m["top1_accuracy_classifier_only"] * m["labelled_points"])
    print(f"trip {d.name}: {r['n_points']:5d} pts | HMM {m['top1_accuracy_hmm']:.4f} | clf {m['top1_accuracy_classifier_only']:.4f} | "
          f"route-arc F1 {m['route_arc_f1']:.4f} | continuity {s:.4f}/{rc:.4f} | {r['timings']['total_s']:.2f}s {json.dumps({k: round(v, 2) for k, v in r['timings'].items()})}")
print(f"POOLED Top-1: HMM {tot_ok/tot_lab:.4f} | classifier-only {tot_clf/tot_lab:.4f}")
