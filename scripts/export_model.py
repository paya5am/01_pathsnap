"""M8 export: produce the artifacts the API loads (contract: artifacts/README.md).

Fits the final hist_gb configuration on ALL 80 training trips (labelled rows of the M3 dataset) and writes
  artifacts/model.joblib      the fitted HistGradientBoostingClassifier
  artifacts/manifest.json     features, versions, params, training-data fingerprint
  artifacts/hmm_config.json   decoder parameters (copied from notebook 07's output)
Run on Colab (same scikit-learn version as the API laptop!) or locally:
  python scripts/export_model.py --dataset <m3_outputs/dataset> --params <m6_outputs/best_params.json>
                                 --params-key best_logloss_tied_params --hmm-config <m7_outputs/hmm_config.json> --out artifacts
"""
import argparse, hashlib, json, platform, sys, time
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib, numpy as np, sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from pathsnap import __version__ as PV
from pathsnap.features import FEATURES
from pathsnap.modeling import load_dataset
from pathsnap.config import SEED

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", help="M3 dataset folder (candidates_features + points); OR use --data-root")
ap.add_argument("--data-root", help="folder containing train/ (raw trips): training rows are rebuilt with pathsnap (~1 min)")
ap.add_argument("--params", required=True)
ap.add_argument("--params-key", default="best_logloss_tied_params"); ap.add_argument("--hmm-config", required=True)
ap.add_argument("--out", default="artifacts"); ap.add_argument("--name", default="hist_gb_tuned_ll")
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
bp = json.load(open(a.params)); params = bp[a.params_key]
fixed = dict(max_iter=600, early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, max_bins=255, random_state=SEED)
if a.dataset:
    F, P = load_dataset(a.dataset)
    tr = F[(F.partition == "train") & (F.label >= 0)]
elif a.data_root:
    import pandas as pd
    from pathsnap.pipeline import process_trip
    from pathsnap.io import list_trips, trip_dir
    rows = [process_trip(trip_dir(a.data_root, "train", t))[0] for t in list_trips(a.data_root, "train")]
    tr = pd.concat(rows, ignore_index=True); tr = tr[tr.label >= 0]
    print(f"rebuilt {len(tr):,} labelled training rows from {a.data_root}/train")
else:
    sys.exit("give --dataset or --data-root")
X, y = tr[FEATURES].values, tr.label.values
fp = hashlib.sha1(np.ascontiguousarray(X[:: max(1, len(X) // 20000)]).tobytes()).hexdigest()
t0 = time.time(); model = HistGradientBoostingClassifier(**params, **fixed).fit(X, y); fit_s = time.time() - t0
joblib.dump(model, out / "model.joblib", compress=3)
hmm = json.load(open(a.hmm_config))
if hmm.get("emission_model") and hmm["emission_model"] != a.name:
    print(f"WARNING: hmm_config was tuned for '{hmm['emission_model']}' but you are exporting '{a.name}'")
json.dump(hmm, open(out / "hmm_config.json", "w"), indent=1)
manifest = {"model_name": a.name, "model_file": "model.joblib", "model_class": type(model).__name__,
            "params": params, "fixed_params": {k: v for k, v in fixed.items()}, "n_trees": int(model.n_iter_),
            "features": FEATURES, "pathsnap_version": PV, "sklearn_version": sklearn.__version__,
            "python_version": platform.python_version(), "trained_on": "all 80 training trips (labelled rows of the M3 dataset)",
            "training_rows": int(len(X)), "positive_share": float(y.mean()), "training_rows_sha1": fp, "fit_seconds": round(fit_s, 1),
            "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
json.dump(manifest, open(out / "manifest.json", "w"), indent=1)
print(json.dumps({k: v for k, v in manifest.items() if k != "features"}, indent=1))
print("wrote", sorted(p.name for p in out.iterdir()))
