"""Combine the classifier score and the structural signals into a verdict.

The classifier scores the whole document text and, separately, any hidden text
the structural layer recovered. A structural finding only counts as an attack
when its hidden text also scores as injection, so legitimate hidden text
(accessibility tags and the like) does not trigger a false alarm.
"""
from __future__ import annotations

import time

from . import config, pdf_forensics, semantic

# Contribution of each signal severity to the structural score.
SEVERITY_WEIGHT = {"high": 0.55, "medium": 0.25, "low": 0.08}


def _structural_risk(signals: list[dict]) -> float:
    # Noisy-OR: weak signals still accumulate, none pins the score at 1.0.
    inv = 1.0
    for s in signals:
        inv *= 1.0 - SEVERITY_WEIGHT.get(s["severity"], 0.0)
    return 1.0 - inv


def _verdict(risk: float) -> str:
    if risk >= config.INJECTED_AT:
        return "injected"
    if risk >= config.SUSPICIOUS_AT:
        return "suspicious"
    return "clean"


def analyse_pdf(raw: bytes, filename: str) -> dict:
    started = time.perf_counter()
    return analyse_parsed(pdf_forensics.analyse(raw), filename, started)


def analyse_parsed(parsed: dict, filename: str, started: float | None = None) -> dict:
    # Takes an already-parsed document so a corpus can be parsed once and
    # scored many times; parsing is the expensive part.
    started = time.perf_counter() if started is None else started

    text = parsed["text"]
    signals = parsed["signals"]
    hidden = parsed.get("hidden_text", "")

    doc_score, chunks = semantic.score_text(text)
    hidden_score = semantic.score_text(hidden)[0] if hidden.strip() else 0.0
    struct = _structural_risk(signals)

    if struct > 0.0 and hidden.strip():
        if hidden_score >= config.HIDDEN_GATE:
            hidden_component = 1.0 - (1.0 - struct) * (1.0 - hidden_score)
        else:
            # Hidden but benign: keep the signals in the response, damp the risk.
            hidden_component = struct * config.BENIGN_HIDDEN_DAMPING
    else:
        hidden_component = struct

    # doc_score gives a visible injection (no hidden text) its own path.
    risk = 1.0 - (1.0 - hidden_component) * (1.0 - doc_score)

    info = semantic.info()
    return {
        "filename": filename,
        "pages": parsed["pages"],
        "verdict": _verdict(risk),
        "risk": round(risk, 4),
        "model_score": round(doc_score, 4),
        "hidden_score": round(hidden_score, 4),
        "hidden_text": hidden[:1500],
        "model_name": info["model_name"],
        "signals": signals,
        "top_chunks": chunks[:5],
        "char_count": len(text),
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }
