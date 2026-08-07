"""
Structural rule-based detector (fixed thresholds).

Backend-ready wrapper around detector.py's --mode structural: flags a PDF if ANY
hidden text is found via fixed thresholds (contrast, tiny font, off-page, render
mode 3, transparency). No semantics involved -- higher recall, much lower precision
than the hybrid model (see results/dataset_check_structural.json).

Public interface (stable -- safe to import from an API layer):
    predict(pdf_path: str) -> dict
        {
          "model": "structural_rule",
          "label": 0 | 1,
          "score": float,          # 1.0 if flagged, else 0.0 (binary, not a probability)
          "signals": [str, ...],   # which structural signals fired, e.g. ["low_contrast", "tiny_font"]
          "hidden_text": str,      # recovered hidden text, if any
        }

CLI usage
    python model_structural_rule.py path/to/file.pdf
"""

import sys
from detector import scan_pdf, decide


def predict(pdf_path: str) -> dict:
    result = scan_pdf(pdf_path)
    flagged = decide(result, mode="structural")
    return {
        "model": "structural_rule",
        "label": int(flagged),
        "score": 1.0 if flagged else 0.0,
        "signals": result["signals"],
        "hidden_text": result["hidden_text"],
    }


if __name__ == "__main__":
    import json
    if len(sys.argv) != 2:
        print("usage: python model_structural_rule.py path/to/file.pdf")
        sys.exit(1)
    print(json.dumps(predict(sys.argv[1]), indent=2))
