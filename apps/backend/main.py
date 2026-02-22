import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import (
    ThreatAnalysisRequest,
    ThreatClassification,
    AlertRecord,
    IngestResponse,
    AnalyticsSummary,
    EscalationPrediction,
)

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="BlackBox API",
    description="Threat detection RAG pipeline with analytics",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/")
async def read_root():
    return {"message": "BlackBox API is running"}


@app.get("/api/health")
async def health_check():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

@app.post("/api/ingest", response_model=IngestResponse)
async def ingest_legal_codes():
    """Parse Title 18 XML, embed, and store in Actian VectorAI DB."""
    from rag.ingest_legal_codes import run_ingestion

    try:
        result = await run_ingestion()
        return IngestResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


# ---------------------------------------------------------------------------
# Threat Analysis (core RAG pipeline)
# ---------------------------------------------------------------------------

@app.post("/api/analyze", response_model=dict)
async def analyze_transcript(request: ThreatAnalysisRequest):
    """
    Analyze a conversation transcript for threats.
    Embeds the transcript, searches legal codes in Actian,
    classifies via Groq LLM, and writes alert to Supabase if threat detected.
    """
    from rag.pipeline import analyze_transcript as run_analysis

    try:
        result = await run_analysis(
            transcript=request.transcript,
            user_id=request.user_id,
            thread_id=request.thread_id,
            guardian_id=request.guardian_id,
        )
        classification: ThreatClassification = result["classification"]
        alert: AlertRecord | None = result["alert"]

        return {
            "classification": classification.model_dump(),
            "alert": alert.model_dump() if alert else None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

@app.get("/api/analytics/summary/{guardian_id}", response_model=AnalyticsSummary)
async def analytics_summary(guardian_id: str, days: int = 30):
    """Descriptive analytics: severity breakdown, category breakdown, timeline."""
    from analytics.summary import get_guardian_summary

    try:
        return await get_guardian_summary(guardian_id, days=days)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analytics query failed: {str(e)}")


@app.post("/api/predict/{user_id}", response_model=list[EscalationPrediction])
async def predict_escalation(user_id: str):
    """Predictive analytics: identify behavioral escalation patterns."""
    from analytics.predict import predict_escalation as run_prediction

    try:
        return await run_prediction(user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
