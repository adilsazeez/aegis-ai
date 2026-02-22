from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ThreatAnalysisRequest(BaseModel):
    transcript: str = Field(..., description="Conversation transcript to analyze")
    user_id: str = Field(..., description="ID of the user being protected")
    thread_id: Optional[str] = Field(None, description="Conversation session ID")
    guardian_id: Optional[str] = Field(None, description="Assigned guardian ID")


class ThreatClassification(BaseModel):
    is_threat: bool = Field(..., description="Whether a threat was detected")
    severity: Optional[str] = Field(None, description="'moderate' or 'severe'")
    offense_title: Optional[str] = Field(None, description="One-line offense description")
    section_reference: Optional[str] = Field(None, description="Legal code reference")
    category: Optional[str] = Field(None, description="Threat category")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)


class AlertRecord(BaseModel):
    id: Optional[str] = None
    user_id: str
    thread_id: Optional[str] = None
    guardian_id: Optional[str] = None
    severity: str
    offense_title: str
    section_reference: str
    category: str
    confidence_score: float
    transcript_snippet: Optional[str] = None
    created_at: Optional[datetime] = None


class AnalyticsSummary(BaseModel):
    total_alerts: int
    severity_breakdown: dict[str, int]
    category_breakdown: dict[str, int]
    timeline: list[dict]


class EscalationPrediction(BaseModel):
    pattern: str = Field(..., description="Identified behavioral pattern")
    description: str = Field(..., description="Why this pattern is dangerous")
    prediction: str = Field(..., description="What is likely to happen next")
    risk_level: int = Field(..., ge=1, le=10, description="Risk score 1-10")
    recommendation: str = Field(..., description="Action recommendation for guardian")


class IngestResponse(BaseModel):
    chunks_ingested: int
    collection_name: str
    message: str
