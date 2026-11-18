"""Smoke test for a RUNNING API (run on either laptop):  python scripts/smoke_test_api.py http://<laptop1-ip>:8000
Checks /health, /trips, /predict on every trip, a bad trip id, and that no reference geometry is in the response."""
import sys, time, requests
base = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")
h = requests.get(base + "/health", timeout=10).json()
print(f"health: model={h['model_name']} ({h['model_status']}) decoder={h['decoder']} ({h['decoder_status']}) trips={h['n_eval_trips']}")
for w in h.get("warnings", []): print("  WARNING:", w)
ts = requests.get(base + "/trips", timeout=10).json()["trips"]
print("trips:", [t["trip_number"] for t in ts])
forbidden = {"route", "reference_route", "ref_arc", "reference"}
for t in ts:
    t0 = time.time(); r = requests.post(base + "/predict", json={"trip_id": t["trip_id"]}, timeout=300)
    assert r.status_code == 200, r.text
    j = r.json(); m = j["metrics"]
    assert not (forbidden & set(j)), "response contains reference data!"
    assert len(j["gps"]) == j["n_points"] == len(j["predicted_arc_per_point"]) and len(j["predicted_route"]) > 0
    print(f"  trip {t['trip_number']:3d}: {j['n_points']:5d} pts | Top-1 HMM {m.get('top1_accuracy_hmm')} | clf {m.get('top1_accuracy_classifier_only')} "
          f"| continuity {m['route_continuity_reachable']:.4f} | {time.time()-t0:.2f}s round trip")
bad = requests.post(base + "/predict", json={"trip_id": "99999"}, timeout=10)
assert bad.status_code == 404, bad.status_code
print("bad trip id → 404 OK\nALL CHECKS PASSED")
