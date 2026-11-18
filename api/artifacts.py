"""Load the exported model + HMM configuration (contract: artifacts/README.md).

If artifacts/model.joblib is missing, a clearly-labelled PLACEHOLDER scorer is used so the API and UI can be developed
and demonstrated end-to-end before the trained model is delivered. Every response and /health report which one is
active. Dropping the real files into artifacts/ and restarting the API switches over; no code change is needed.
"""
import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

from pathsnap.features import FEATURES
from pathsnap import __version__ as PATHSNAP_VERSION

log = logging.getLogger("pathsnap.api")

DEFAULT_DECODER = {"beta": 3.0, "break_s": 300.0, "gap_exp": 0.0}   # M7 tuned values (D26), used if hmm_config.json is missing


class PlaceholderModel:
    """NOT A TRAINED MODEL. Hand-written score mirroring the M3 rule baseline (nearest arc pointing the car's way),
    turned into a probability. Exists only so the pipeline runs before the real model is exported."""
    name = "PLACEHOLDER (rule: nearest arc pointing the car's way)"

    def predict_proba(self, X):
        f = {k: X[:, i] for i, k in enumerate(FEATURES)}
        z = 2.0 - 0.35 * f["dist_rel"] - 0.05 * f["dist"] + 2.5 * f["cos_heading"] * f["heading_known"]
        p = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1 - p, p])


@dataclass
class Artifacts:
    model: object
    model_status: str                 # "trained" | "placeholder"
    model_name: str
    features: list
    decoder: dict
    decoder_status: str               # "tuned" | "defaults"
    manifest: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def public_info(self):
        return {"model_status": self.model_status, "model_name": self.model_name, "decoder": self.decoder,
                "decoder_status": self.decoder_status, "features": self.features, "pathsnap_version": PATHSNAP_VERSION,
                "manifest": {k: v for k, v in self.manifest.items() if k != "training_rows_sha1"}, "warnings": self.warnings}


def load_artifacts(artifact_dir: Path, allow_placeholder=True) -> Artifacts:
    artifact_dir = Path(artifact_dir)
    warn = []
    manifest = {}
    mf = artifact_dir / "manifest.json"
    if mf.is_file():
        manifest = json.loads(mf.read_text())
    model_path = artifact_dir / manifest.get("model_file", "model.joblib")
    if model_path.is_file():
        import joblib, sklearn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = joblib.load(model_path)
            warn += [str(x.message) for x in w if "version" in str(x.message).lower()]
        status, name = "trained", manifest.get("model_name", model_path.name)
        trained_with = manifest.get("sklearn_version")
        if trained_with and trained_with != sklearn.__version__:
            warn.append(f"scikit-learn version mismatch: model saved with {trained_with}, running {sklearn.__version__}. "
                        f"Install the same version: pip install scikit-learn=={trained_with}")
        feats = manifest.get("features", FEATURES)
        if list(feats) != list(FEATURES):
            raise RuntimeError(f"Feature list in manifest {feats} differs from pathsnap.FEATURES {FEATURES}. "
                               "The model was trained with a different feature set; re-export it with this pathsnap version.")
        if manifest.get("pathsnap_version") and manifest["pathsnap_version"].split(".")[:2] != PATHSNAP_VERSION.split(".")[:2]:
            warn.append(f"pathsnap version mismatch: exported with {manifest['pathsnap_version']}, running {PATHSNAP_VERSION}")
    else:
        if not allow_placeholder:
            raise RuntimeError(f"No model at {model_path} and placeholders are disabled (PATHSNAP_ALLOW_PLACEHOLDER=0).")
        model, status, name = PlaceholderModel(), "placeholder", PlaceholderModel.name
        warn.append(f"No trained model found at {model_path}: using the PLACEHOLDER scorer. Results are NOT the final model's.")
    hc = artifact_dir / "hmm_config.json"
    if hc.is_file():
        cfg = json.loads(hc.read_text())
        decoder = {k: float(cfg["decoder"][k]) for k in ("beta", "break_s", "gap_exp")}
        dstatus = "tuned"
    else:
        decoder, dstatus = dict(DEFAULT_DECODER), "defaults"
        warn.append(f"No hmm_config.json in {artifact_dir}: using M7 default decoder parameters {DEFAULT_DECODER}.")
    for w_ in warn:
        log.warning(w_)
    log.info("model: %s (%s) | decoder: %s (%s)", name, status, decoder, dstatus)
    return Artifacts(model, status, name, list(FEATURES), decoder, dstatus, manifest, warn)
