# risk_analysis.py
import os
import json
import asyncio
from typing import Optional, Dict, Any, List

from dotenv import load_dotenv
load_dotenv()

from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are a risk triage assistant for user interactions.
Return ONLY JSON (no markdown, no extra text) with exactly these keys:
- level: one of ["low","medium","high","critical"]
- score: number 0 to 100
- reason: short string

Scoring guide:
0–39 low, 40–69 medium, 70–84 high, 85–100 critical
Be conservative: if unsure, choose the LOWER score.
"""

RAG_SYSTEM_PROMPT = """You are a safety and criminal law analysis assistant protecting a user's physical safety. You receive:
1) A transcript of what the user said (voice or typed text).
2) Optionally, written_context: background the user wrote when starting the session. Consider it alongside the transcript.
3) Relevant prosocial-dialog examples (context/response snippets) for safety guidance.
4) Excerpts from Title 18 U.S. Criminal Code (18 USC sections).

Return ONLY a single JSON object (no markdown, no extra text) with exactly these keys:
- threat_category: string, e.g. "none", "self_harm", "violence", "harassment", "stalking", "intimidation", "other"
- severity: one of "low", "medium", "high", "critical"
- label: one of "low", "medium", "high", "critical" (must match severity)
- score: integer 0 to 100
- reason: short string explaining the safety assessment
- narrative_analysis: 1–3 sentences summarizing the situation and risk
- relevant_law: if applicable, one line citing 18 USC section(s), e.g. "18 USC § 1111"; otherwise empty string

SCORING GUIDE — follow these ranges strictly:
  0-15: Completely benign, everyday conversation, no safety relevance at all.
 16-39: Mild concern or ambiguous phrasing but no clear threat.
 40-59: Moderate concern — user expresses fear, asks for help, mentions feeling unsafe or in danger,
        OR uses words like "danger", "help me", "emergency", "scared", "unsafe", "threatened".
        These are NOT zero — a person calling out for help is at least moderately at risk.
 60-79: High concern — explicit mention of stalking, harassment, abuse, being followed,
        direct threats received, domestic violence situations.
 80-100: Critical — explicit violence (kill, murder, weapon), self-harm (suicide, end my life),
         or imminent physical danger.

IMPORTANT:
- If the user says "danger", "help", "please help me", "I'm scared", "emergency", or similar distress language,
  that is ALWAYS at least score 40-59 (medium). NEVER score distress calls as 0.
- If the user describes being followed, stalked, harassed, or threatened, score at least 60.
- If violence or self-harm is mentioned, score at least 80.
- When the transcript is short (a few words), infer intent from the written_context too.
- Use the prosocial examples to understand conversational safety patterns.
- Use Title 18 only to cite possible criminal relevance when applicable.
"""

def _sanitize_result(obj: dict) -> dict:
    level = obj.get("level", "low")
    score = float(obj.get("score", 0.0))
    reason = obj.get("reason", "")
    if level not in {"low", "medium", "high", "critical"}:
        level = "low"
    if score < 0: score = 0.0
    if score > 100: score = 100.0
    if not isinstance(reason, str):
        reason = str(reason)
    return {"level": level, "score": score, "reason": reason}


def _sanitize_rag_result(obj: dict) -> dict:
    level = (obj.get("label") or obj.get("level") or "low").lower()
    severity = (obj.get("severity") or level).lower()
    for v in (level, severity):
        if v not in {"low", "medium", "high", "critical"}:
            level = "low" if level == v else level
            severity = "low" if severity == v else severity
    score = float(obj.get("score", 0.0))
    if score < 0: score = 0.0
    if score > 100: score = 100.0
    return {
        "threat_category": str(obj.get("threat_category") or "none")[:200],
        "severity": severity,
        "label": level,
        "score": score,
        "reason": str(obj.get("reason") or "")[:500],
        "narrative_analysis": str(obj.get("narrative_analysis") or "")[:2000],
        "relevant_law": str(obj.get("relevant_law") or "")[:500],
    }


def _apply_keyword_floor(result: dict, transcript: str, written_context: Optional[str] = None) -> dict:
    """
    Graduated safety net: ensures keywords cannot be scored below their minimum floor.
    Works in tiers so different severity keywords get different minimum scores.
    """
    text = f"{written_context or ''} {transcript or ''}".lower()
    current_score = float(result.get("score", 0.0))

    # Tiers: (min_score, label, severity, threat_category, keywords)
    tiers = [
        (85, "critical", "critical", "violence",
         ["kill", "murder", "shoot", "stab", "bomb", "weapon", "gun"]),
        (75, "high", "high", "self_harm",
         ["suicide", "self harm", "self-harm", "end my life", "kill myself", "want to die"]),
        (65, "high", "high", "harassment",
         ["stalk", "stalking", "stalked", "harass", "harassment", "abuse", "abused",
          "blackmail", "threatened me", "domestic violence", "being followed"]),
        (45, "medium", "medium", "other",
         ["danger", "in danger", "help me", "emergency", "please help",
          "unsafe", "scared", "frightened", "terrified", "threatened",
          "i'm not safe", "someone is following"]),
    ]

    matched_floor = None
    matched_tier = None
    for floor_score, floor_label, floor_severity, floor_cat, keywords in tiers:
        if any(k in text for k in keywords):
            matched_floor = floor_score
            matched_tier = (floor_label, floor_severity, floor_cat)
            break  # highest-priority tier wins

    if matched_floor is None:
        return result

    if current_score < matched_floor:
        result["score"] = float(matched_floor)
        result["label"] = matched_tier[0]
        result["severity"] = matched_tier[1]
        if (result.get("threat_category") or "none") == "none":
            result["threat_category"] = matched_tier[2]
        prior_reason = (result.get("reason") or "").strip()
        extra = f"Safety floor applied (keyword match → min {matched_floor})."
        result["reason"] = f"{prior_reason} {extra}".strip()
    return result


def _local_fallback_analysis(transcript: str, written_context: Optional[str] = None) -> dict:
    """
    Local rule-based fallback when Groq is rate-limited.
    Uses graduated scoring matching the keyword floor tiers.
    """
    text = f"{written_context or ''} {transcript or ''}".lower()
    score = 10.0
    threat_category = "none"
    severity = "low"
    label = "low"
    reason = "Low immediate risk detected from local fallback analysis."
    narrative = "Fallback analysis used because the LLM API is currently rate-limited."
    relevant_law = ""

    if any(k in text for k in ["kill", "murder", "shoot", "stab", "bomb", "weapon", "gun"]):
        score = 90.0
        threat_category = "violence"
        severity = label = "critical"
        reason = "Violence-related terms detected; urgent review recommended."
        narrative = "Critical safety concern: violence-related language identified. Immediate review recommended."
    elif any(k in text for k in ["suicide", "self harm", "self-harm", "end my life", "kill myself", "want to die"]):
        score = 80.0
        threat_category = "self_harm"
        severity = label = "high"
        reason = "Self-harm related terms detected; urgent support recommended."
        narrative = "High safety concern: self-harm language identified. Urgent support and resources should be provided."
    elif any(k in text for k in ["stalk", "stalking", "stalked", "harass", "harassment", "abuse", "abused",
                                  "blackmail", "threatened me", "domestic violence", "being followed"]):
        score = 70.0
        threat_category = "harassment"
        severity = label = "high"
        reason = "Harassment, stalking, or abuse language detected."
        narrative = "High concern: language consistent with harassment, stalking, or abuse. Situation warrants monitoring."
    elif any(k in text for k in ["threat", "threaten", "intimidate", "intimidation"]):
        score = 60.0
        threat_category = "intimidation"
        severity = label = "medium"
        reason = "Threat or intimidation language detected."
    elif any(k in text for k in ["danger", "in danger", "help me", "emergency", "please help",
                                  "scared", "frightened", "terrified", "threatened", "unsafe"]):
        score = 50.0
        threat_category = "other"
        severity = label = "medium"
        reason = "Distress or safety-concern language detected."
        narrative = "Moderate concern: user expresses distress or fear. Monitoring recommended."
    elif any(k in text for k in ["alone", "night", "followed", "uncomfortable", "worried"]):
        score = 30.0
        threat_category = "other"
        severity = label = "low"
        reason = "Mild personal safety concern detected."

    return _sanitize_rag_result(
        {
            "threat_category": threat_category,
            "severity": severity,
            "label": label,
            "score": score,
            "reason": reason,
            "narrative_analysis": narrative,
            "relevant_law": relevant_law,
        }
    )

async def assess_danger(transcript: str, location: Optional[dict] = None) -> dict:
    """
    Returns: {"level": str, "score": float, "reason": str}
    """
    payload = {"transcript": transcript, "location": location}

    def _call_groq_sync() -> dict:
        # NOTE: Groq client is sync; run in a thread to avoid blocking.
        resp = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload)},
            ],
            temperature=0,
        )

        content = resp.choices[0].message.content.strip()

        # Some models may wrap JSON in text; try to extract the JSON object.
        # Fast heuristic: find first "{" and last "}"
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            content = content[start : end + 1]

        return json.loads(content)

    try:
        # Optional timeout so you never hang:
        result = await asyncio.wait_for(asyncio.to_thread(_call_groq_sync), timeout=8)
        return _sanitize_result(result)

    except Exception as e:
        return {
            "level": "low",
            "score": 0.0,
            "reason": f"Assessment unavailable (Groq error: {type(e).__name__}).",
        }


async def assess_danger_with_rag(
    transcript: str,
    prosocial_chunks: List[dict],
    title18_context: str,
    written_context: Optional[str] = None,
    location: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    One Groq call: safety + criminal analysis using transcript, optional written context, RAG chunks, and Title 18.
    Returns dict with: threat_category, severity, label, score, reason, narrative_analysis, relevant_law.
    """
    prosocial_text = "\n\n".join(
        (c.get("context") or "") + " → " + (c.get("response") or "") for c in prosocial_chunks[:10]
    )
    if not prosocial_text:
        prosocial_text = "(No similar prosocial examples retrieved.)"
    payload = {
        "transcript": transcript,
        "location": location,
        "prosocial_examples": prosocial_text[:8000],
        "title18_excerpts": title18_context[:12000],
    }
    if written_context:
        payload["written_context"] = written_context[:2000]  # user-provided text when starting the session
    user_content = json.dumps(payload)

    def _call_groq_sync() -> dict:
        resp = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            messages=[
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0,
        )
        content = (resp.choices[0].message.content or "").strip()
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            content = content[start : end + 1]
        return json.loads(content)

    async def _call_with_retry() -> dict:
        last_err = None
        timeout_seconds = 22  # generous so text-after-voice has time (Groq can be slow back-to-back)
        for attempt in range(3):  # initial + 2 retries
            try:
                return await asyncio.wait_for(asyncio.to_thread(_call_groq_sync), timeout=timeout_seconds)
            except Exception as e:
                last_err = e
                is_rate_limit = "RateLimit" in type(e).__name__ or "429" in str(e)
                is_timeout = isinstance(e, asyncio.TimeoutError) or "Timeout" in type(e).__name__
                if is_rate_limit and attempt < 2:
                    await asyncio.sleep(8)  # wait 8s before next try (Groq limits often per minute)
                    continue
                if is_timeout and attempt < 2:
                    await asyncio.sleep(4)  # brief pause then retry (helps when text follows voice)
                    continue
                raise
        raise last_err

    try:
        result = await _call_with_retry()
        return _apply_keyword_floor(
            _sanitize_rag_result(result), transcript=transcript, written_context=written_context
        )
    except Exception as e:
        is_rate_limit = "RateLimit" in type(e).__name__ or "429" in str(e) or "rate" in str(e).lower()
        is_timeout = isinstance(e, asyncio.TimeoutError) or "Timeout" in type(e).__name__
        if is_rate_limit or is_timeout:
            # Keep UX functional: use local fallback so score/reason reflect content, not "unavailable"
            return _apply_keyword_floor(
                _local_fallback_analysis(transcript=transcript, written_context=written_context),
                transcript=transcript,
                written_context=written_context,
            )
        return _apply_keyword_floor(_sanitize_rag_result({
            "threat_category": "none",
            "severity": "low",
            "label": "low",
            "score": 0.0,
            "reason": f"Analysis unavailable ({type(e).__name__}).",
            "narrative_analysis": "",
            "relevant_law": "",
        }), transcript=transcript, written_context=written_context)