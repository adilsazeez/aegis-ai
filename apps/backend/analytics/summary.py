from collections import Counter
from datetime import datetime, timedelta, timezone

from config import get_supabase_client
from models import AnalyticsSummary


async def get_guardian_summary(
    guardian_id: str,
    days: int = 30,
) -> AnalyticsSummary:
    """
    Build a descriptive analytics summary of all alerts for a guardian's
    protected users over the given time window.

    Returns severity breakdown, category breakdown, and daily timeline.
    """
    supabase = get_supabase_client()
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    result = (
        supabase.table("alerts")
        .select("severity, category, created_at")
        .eq("guardian_id", guardian_id)
        .gte("created_at", since)
        .order("created_at", desc=False)
        .execute()
    )

    alerts = result.data

    severity_counts = Counter(a["severity"] for a in alerts)
    category_counts = Counter(a["category"] for a in alerts)

    daily: dict[str, dict[str, int]] = {}
    for alert in alerts:
        day = alert["created_at"][:10]
        if day not in daily:
            daily[day] = {"moderate": 0, "severe": 0}
        daily[day][alert["severity"]] += 1

    timeline = [
        {"date": day, "moderate": counts["moderate"], "severe": counts["severe"]}
        for day, counts in sorted(daily.items())
    ]

    return AnalyticsSummary(
        total_alerts=len(alerts),
        severity_breakdown=dict(severity_counts),
        category_breakdown=dict(category_counts),
        timeline=timeline,
    )


async def get_user_alert_history(
    user_id: str,
    limit: int = 50,
) -> list[dict]:
    """
    Fetch recent alerts for a specific user, ordered newest-first.
    Used by the escalation prediction module.
    """
    supabase = get_supabase_client()

    result = (
        supabase.table("alerts")
        .select("severity, category, offense_title, section_reference, confidence_score, created_at")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )

    return result.data
