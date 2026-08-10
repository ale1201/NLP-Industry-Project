"""Response models. These define the contract the React app codes against."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Verdict = Literal["clean", "suspicious", "injected"]
Severity = Literal["low", "medium", "high"]


class Signal(BaseModel):
    """One forensic finding from the PDF structure (not from the classifier)."""

    id: str
    title: str
    severity: Severity
    detail: str
    page: int | None = Field(None, description="1-indexed page, null if document-wide")
    evidence: str | None = None


class Chunk(BaseModel):
    """A scored window of document text."""

    index: int
    score: float
    preview: str


class AnalyzeResponse(BaseModel):
    filename: str
    pages: int
    verdict: Verdict
    risk: float = Field(..., description="Fused risk in [0,1] driving the verdict")
    model_score: float = Field(..., description="Classifier score on the document text")
    hidden_score: float = Field(0.0, description="Classifier score on the hidden text")
    hidden_text: str = Field("", description="Text hidden from a human reader")
    model: str = Field(..., description="Model key that scored this document")
    model_name: str
    signals: list[Signal]
    top_chunks: list[Chunk]
    char_count: int
    elapsed_ms: int


class ModelInfo(BaseModel):
    id: str
    label: str


class HealthResponse(BaseModel):
    status: str
    models: list[ModelInfo] = Field(default_factory=list, description="Selectable models")
    default_model: str | None = None
