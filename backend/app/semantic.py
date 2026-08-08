"""TF-IDF injection classifier loaded from a joblib pipeline.

Default model is models/semantic/semantic_model_combined.joblib; override with
PID_SEMANTIC_MODEL to use one of the other variants (_real, _svm, _nb,
_synthetic).
"""
from __future__ import annotations

import threading
from pathlib import Path

from . import config

_lock = threading.Lock()
_bundle: dict | None = None


class SemanticModelUnavailable(RuntimeError):
    """Model file is missing or failed to load."""


def _fix_pickle(pipe):
    # Pipelines were pickled with sklearn 1.9; 1.7 reads LogisticRegression.
    # multi_class in predict_proba but 1.9 no longer sets it. Restore it.
    clf = pipe.named_steps.get("clf") if hasattr(pipe, "named_steps") else None
    if clf is not None and not hasattr(clf, "multi_class"):
        clf.multi_class = "auto"
    return pipe


def _load() -> dict:
    global _bundle
    with _lock:
        if _bundle is not None:
            return _bundle

        path = Path(config.SEMANTIC_MODEL_PATH)
        if not path.exists():
            raise SemanticModelUnavailable(
                f"No model file at {path}. Set PID_SEMANTIC_MODEL to a .joblib pipeline."
            )

        import joblib

        try:
            pipe = _fix_pickle(joblib.load(path))
        except Exception as exc:
            raise SemanticModelUnavailable(f"Could not load {path}: {exc}") from exc

        _bundle = {"pipe": pipe, "name": f"semantic:{path.stem}", "path": str(path)}
        return _bundle


def available() -> bool:
    try:
        _load()
        return True
    except SemanticModelUnavailable:
        return False


def info() -> dict:
    return {"model_name": _load()["name"]}


def score_text(text: str) -> tuple[float, list[dict]]:
    """Return P(injection) for the text plus a one-item chunk list."""
    if not text or not text.strip():
        return 0.0, []

    pipe = _load()["pipe"]
    prob = round(float(pipe.predict_proba([text])[0][1]), 4)
    return prob, [{"index": 0, "score": prob, "preview": " ".join(text.split())[:400]}]
