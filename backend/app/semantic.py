"""TF-IDF injection classifiers loaded from joblib pipelines.

Two separate models are exposed, `logreg` and `svm` (see config.MODELS). Each
is its own pipeline; only one is used per request and they are never combined.
"""
from __future__ import annotations

import threading

from . import config

_lock = threading.Lock()
_bundles: dict[str, dict] = {}


class SemanticModelUnavailable(RuntimeError):
    """Model key is unknown, or its file is missing or failed to load."""


def _fix_pickle(pipe):
    # Pipelines were pickled with sklearn 1.9; 1.7 reads LogisticRegression.
    # multi_class in predict_proba but 1.9 no longer sets it. Restore it.
    clf = pipe.named_steps.get("clf") if hasattr(pipe, "named_steps") else None
    if clf is not None and not hasattr(clf, "multi_class"):
        clf.multi_class = "auto"
    return pipe


def _resolve(model: str | None) -> str:
    return model or config.DEFAULT_MODEL


def _load(model: str | None = None) -> dict:
    key = _resolve(model)
    with _lock:
        if key in _bundles:
            return _bundles[key]

        spec = config.MODELS.get(key)
        if spec is None:
            raise SemanticModelUnavailable(
                f"Unknown model {key!r}; choose one of {', '.join(config.MODELS)}."
            )

        from pathlib import Path

        path = Path(spec["path"])
        if not path.exists():
            raise SemanticModelUnavailable(f"No model file for {key!r} at {path}.")

        import joblib

        try:
            pipe = _fix_pickle(joblib.load(path))
        except Exception as exc:
            raise SemanticModelUnavailable(f"Could not load {path}: {exc}") from exc

        _bundles[key] = {"pipe": pipe, "key": key, "label": spec["label"], "path": str(path)}
        return _bundles[key]


def available(model: str | None = None) -> bool:
    try:
        _load(model)
        return True
    except SemanticModelUnavailable:
        return False


def list_models() -> list[dict]:
    """Model keys the server can actually serve, with labels."""
    return [
        {"id": key, "label": spec["label"]}
        for key, spec in config.MODELS.items()
        if available(key)
    ]


def info(model: str | None = None) -> dict:
    b = _load(model)
    return {"model": b["key"], "model_name": b["label"]}


def score_text(text: str, model: str | None = None) -> tuple[float, list[dict]]:
    """Return P(injection) for the text under the chosen model, plus one chunk."""
    if not text or not text.strip():
        return 0.0, []

    pipe = _load(model)["pipe"]
    prob = round(float(pipe.predict_proba([text])[0][1]), 4)
    return prob, [{"index": 0, "score": prob, "preview": " ".join(text.split())[:400]}]
