"""Backend settings (Laptop 1). Override with environment variables; defaults suit running from the repo root."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Folder that directly contains the evaluation trip folders (00000015/, 00000017/, ...), each with
# .track/.nodes/.arcs and (server-side only) .route. Until demo day, point it at the student_test folder.
EVAL_DIR = Path(os.environ.get("PATHSNAP_EVAL_DIR", REPO_ROOT / "data" / "eval"))
# Exported model + configuration (see artifacts/README.md for the contract).
ARTIFACT_DIR = Path(os.environ.get("PATHSNAP_ARTIFACTS", REPO_ROOT / "artifacts"))
LOG_DIR = Path(os.environ.get("PATHSNAP_LOG_DIR", REPO_ROOT / "logs"))
HOST = os.environ.get("PATHSNAP_HOST", "0.0.0.0")      # 0.0.0.0 = reachable from Laptop 2 on the same network
PORT = int(os.environ.get("PATHSNAP_PORT", "8000"))
# Safety: the API refuses to serve a placeholder model unless explicitly allowed (development only).
ALLOW_PLACEHOLDER = os.environ.get("PATHSNAP_ALLOW_PLACEHOLDER", "1") == "1"
