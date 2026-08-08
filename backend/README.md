# Backend — PDF Prompt-Injection Detector

FastAPI service that flags prompt-injection content in uploaded PDFs. It scores
the extracted text with the TF-IDF classifier in `models/semantic/` and inspects
the PDF structure for text hidden from a human reader (white-on-white, tiny
fonts, off-page glyphs, invisible render modes, unicode tricks, etc.).

## Run

    cd backend
    pip install -r requirements.txt
    uvicorn app.main:app --port 8000

Open http://127.0.0.1:8000/docs for the API.

By default it loads `models/semantic/semantic_model_combined.joblib`. Point
`PID_SEMANTIC_MODEL` at another file to use a different variant.

## Frontend

    cd frontend
    npm install
    npm run dev        # http://localhost:5173

## Tests

    cd backend
    python tests/make_fixtures.py
    pytest tests/
