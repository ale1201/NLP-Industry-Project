"""
Structural signals + semantic classifier (hybrid, combined mode) -- the project's
best-performing detector.

Backend-ready wrapper around detector.py's --mode combined: flags a PDF only if
hidden text is found AND the recovered text is judged instruction-like by the
trained semantic model. See results/three_model_comparison_v2.json and report.md
§5b for how the underlying semantic model compares to its synthetic-only and
real-only variants.

Defaults to semantic_model_combined.joblib (trained on synthetic + real deepset
data together -- the model that matched or tied the best score on every benchmark
in report.md §5b). Pass model_path= to swap in semantic_model.joblib (synthetic-only)
or semantic_model_real.joblib (real-only) for comparison.

Public interface (stable -- safe to import from an API layer):
    predict(pdf_path: str, model_path: str = "semantic_model_combined.joblib") -> dict
        {
          "model": "hybrid_combined",
          "semantic_model": str,     # which .joblib was used
          "label": 0 | 1,
          "score": float,            # semantic P(injection) if hidden text was found, else 0.0
          "signals": [str, ...],
          "hidden_text": str,
        }

CLI usage
    python model_hybrid_combined.py path/to/file.pdf
    python model_hybrid_combined.py path/to/file.pdf semantic_model.joblib
"""

import sys
from detector import scan_pdf, decide, load_semantic_model, _MODEL

_LOADED_PATH = None


def _ensure_model(model_path: str):
    global _LOADED_PATH
    if _LOADED_PATH != model_path:
        load_semantic_model(model_path)
        _LOADED_PATH = model_path


def predict(pdf_path: str, model_path: str = "semantic_model_combined.joblib") -> dict:
    _ensure_model(model_path)
    result = scan_pdf(pdf_path)
    flagged = decide(result, mode="combined")

    score = 0.0
    if result["has_hidden"] and result["hidden_text"]:
        import detector as _detector_module
        if _detector_module._MODEL is not None:
            score = float(_detector_module._MODEL.predict_proba([result["hidden_text"]])[0][1])
        elif result["injection_like"]:
            score = 1.0

    return {
        "model": "hybrid_combined",
        "semantic_model": model_path,
        "label": int(flagged),
        "score": score,
        "signals": result["signals"],
        "hidden_text": result["hidden_text"],
    }


if __name__ == "__main__":
    import json
    if len(sys.argv) not in (2, 3):
        print("usage: python model_hybrid_combined.py path/to/file.pdf [model.joblib]")
        sys.exit(1)
    model_arg = sys.argv[2] if len(sys.argv) == 3 else "semantic_model_combined.joblib"
    print(json.dumps(predict(sys.argv[1], model_arg), indent=2))
