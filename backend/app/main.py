# Run with: uvicorn app.main:app --reload --port 8000
from __future__ import annotations

import fitz
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from . import analysis, config, semantic
from .schemas import AnalyzeResponse, HealthResponse

app = FastAPI(
    title="PDF Prompt-Injection Detector",
    description="Classifies uploaded PDFs for prompt-injection content.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in config.CORS_ORIGINS if o.strip()],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    # Warms the default model so the first upload isn't slow.
    models = semantic.list_models()
    if not models:
        return HealthResponse(status="degraded: no models loaded", models=[])
    return HealthResponse(
        status="ok",
        models=models,
        default_model=config.DEFAULT_MODEL,
    )


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(...),
    model: str = Form(config.DEFAULT_MODEL),
) -> AnalyzeResponse:
    raw = await file.read()

    if model not in config.MODELS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown model {model!r}; choose one of {', '.join(config.MODELS)}.",
        )

    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    size_mb = len(raw) / (1024 * 1024)
    if size_mb > config.MAX_UPLOAD_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File is {size_mb:.1f} MB; the limit is {config.MAX_UPLOAD_MB:.0f} MB.",
        )

    if not raw.lstrip()[:5].startswith(b"%PDF"):
        raise HTTPException(
            status_code=415,
            detail="That does not look like a PDF (missing %PDF header).",
        )

    try:
        result = analysis.analyse_pdf(raw, file.filename or "upload.pdf", model)
    except fitz.FileDataError as exc:
        raise HTTPException(
            status_code=422, detail=f"The PDF could not be parsed: {exc}"
        ) from exc
    except semantic.SemanticModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return AnalyzeResponse(**result)
