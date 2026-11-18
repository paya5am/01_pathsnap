"""M8 (Member C): package the final model, preprocessing config and HMM config.

usage: python scripts/export_artifacts.py <results_dir> <artifacts_dir> [<data_root> to self-test]
"""
import hashlib
import json
import os
import platform
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap import candidates, features, labels  # noqa: E402


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main(res, art, data_root=None):
    import lightgbm, numpy, scipy, sklearn
    os.makedirs(art, exist_ok=True)
    shutil.copy(os.path.join(res, "lgbm_tuned.joblib"), os.path.join(art, "model.joblib"))
    shutil.copy(os.path.join(res, "hmm_config.json"), os.path.join(art, "hmm_config.json"))
    pre = dict(projection="local equirectangular centred on trip mean GPS position (D5)",
               candidate_delta_m=candidates.DELTA, candidate_cap_m=candidates.CAP,
               candidate_spacing_m=candidates.SPACING, heading_half_window=labels.HALF_WIN,
               moving_speed_mps=labels.MOVING, features=features.FEATURES,
               notes="Zero-length arcs and self-loops are never candidates; duplicate twins -> smallest id (D11).")
    json.dump(pre, open(os.path.join(art, "preprocessing_config.json"), "w"), indent=2)
    R = json.load(open(os.path.join(res, "lgbm_results.json")))
    manifest = dict(
        model="LightGBM (Optuna-tuned)", params=R["tuned"]["params"],
        trained_on="80 train trips, candidates of labelled points only",
        cv="GroupKFold(5) by trip, trips 10+16 together, seed 42",
        student_test_top1_classifier=R["tuned"]["student_test"]["top1"],
        files={f: sha(os.path.join(art, f)) for f in ["model.joblib", "hmm_config.json", "preprocessing_config.json"]},
        versions=dict(python=platform.python_version(), lightgbm=lightgbm.__version__, numpy=numpy.__version__,
                      scipy=scipy.__version__, sklearn=sklearn.__version__))
    json.dump(manifest, open(os.path.join(art, "manifest.json"), "w"), indent=2)
    if data_root:   # self-test: load artifacts and run one student-test trip end to end
        from pathsnap.inference import evaluate_prediction, load_artifacts, predict_trip_folder
        arts = load_artifacts(art)
        folder = os.path.join(data_root, "student_test", "00000000")
        pred = predict_trip_folder(folder, arts)
        print("self-test trip 0:", evaluate_prediction(pred, folder), pred["timings"])


if __name__ == "__main__":
    main(*sys.argv[1:])
