import json
import logging

from config import get_settings, get_groq_client
from models import EscalationPrediction
from analytics.summary import get_user_alert_history

logger = logging.getLogger(__name__)

ESCALATION_SYSTEM_PROMPT = """You are a behavioral threat analyst for a personal safety application.
Your job: analyze a user's alert history and identify escalation patterns.

You will receive a chronological list of threat alerts detected for a specific user.
Each alert has: severity, category, offense_title, section_reference, confidence_score, timestamp.

Your analysis should:
1. Identify behavioral PATTERNS (e.g., escalating severity, category migration from verbal
   threats to physical threats, increasing frequency).
2. Predict what is LIKELY to happen next based on these patterns.
3. Assign a risk_level from 1-10 (10 = highest danger).
4. Provide a concrete RECOMMENDATION for the guardian.

If the alert history is too small (< 3 alerts) or shows no pattern, still provide your best
assessment but note the limited data.

Respond with a JSON array of pattern objects. Each object must match:
{
  "pattern": "short pattern name",
  "description": "why this pattern is dangerous",
  "prediction": "what is likely to happen next",
  "risk_level": integer 1-10,
  "recommendation": "specific action for the guardian"
}

Return between 1 and 3 patterns. Focus on the most significant ones."""


async def predict_escalation(user_id: str) -> list[EscalationPrediction]:
    """
    Fetch a user's alert history and use Groq LLM to identify
    behavioral escalation patterns and predictions.
    """
    settings = get_settings()
    groq = get_groq_client()

    alerts = await get_user_alert_history(user_id, limit=50)

    if not alerts:
        return [
            EscalationPrediction(
                pattern="No data",
                description="No alerts have been recorded for this user yet.",
                prediction="Cannot predict without historical data.",
                risk_level=1,
                recommendation="Continue monitoring. The system will analyze patterns as alerts accumulate.",
            )
        ]

    alert_summary = "\n".join(
        f"- [{a['created_at']}] severity={a['severity']}, category={a['category']}, "
        f"offense=\"{a['offense_title']}\", section={a['section_reference']}, "
        f"confidence={a['confidence_score']}"
        for a in reversed(alerts)  # chronological order for the LLM
    )

    user_prompt = f"""ALERT HISTORY FOR USER (oldest first, {len(alerts)} total alerts):
{alert_summary}

Analyze this history for escalation patterns. Respond with JSON array only."""

    response = await groq.chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=[
            {"role": "system", "content": ESCALATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
        max_tokens=1024,
    )

    raw_output = response.choices[0].message.content
    parsed = json.loads(raw_output)

    # Groq may wrap the array in an object like {"patterns": [...]}
    if isinstance(parsed, dict):
        for key in ("patterns", "predictions", "results", "data"):
            if key in parsed and isinstance(parsed[key], list):
                parsed = parsed[key]
                break

    if isinstance(parsed, list):
        return [EscalationPrediction(**p) for p in parsed]

    # Single object response
    return [EscalationPrediction(**parsed)]
