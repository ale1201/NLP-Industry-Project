"""Central configuration. Everything is overridable by environment variable."""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Trained classifier. Default is the combined model; other variants
# (_real, _svm, _nb, _synthetic) live in the same folder.
SEMANTIC_MODEL_PATH = os.getenv(
    "PID_SEMANTIC_MODEL",
    str(BASE_DIR / "models" / "semantic" / "semantic_model_combined.joblib"),
)

MAX_UPLOAD_MB = float(os.getenv("PID_MAX_UPLOAD_MB", "20"))
MAX_PAGES = int(os.getenv("PID_MAX_PAGES", "200"))

# Verdict cut-offs on the fused risk score.
SUSPICIOUS_AT = float(os.getenv("PID_SUSPICIOUS_AT", "0.35"))
INJECTED_AT = float(os.getenv("PID_INJECTED_AT", "0.65"))

# Min classifier score for hidden text to count as an attack.
HIDDEN_GATE = float(os.getenv("PID_HIDDEN_GATE", "0.5"))
# Residual risk kept for hidden-but-benign text, so it still shows up.
BENIGN_HIDDEN_DAMPING = float(os.getenv("PID_BENIGN_HIDDEN_DAMPING", "0.25"))

CORS_ORIGINS = os.getenv(
    "PID_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
).split(",")
