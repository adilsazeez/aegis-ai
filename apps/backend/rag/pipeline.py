import json
import logging

from config import get_settings, get_groq_client, get_supabase_client, get_actian_client
from models import ThreatClassification, AlertRecord
from rag.embeddings import embed_text

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a threat classification system for a personal safety application.
Your job: analyze a conversation transcript and determine if it contains a credible threat
based on US Federal Criminal Code (Title 18).

You will receive:
1. A conversation transcript
2. Relevant legal code sections retrieved from Title 18

Classification rules:
- PRIORITIZE RECALL: it is far better to flag a borderline case (false positive) than to miss
  a real threat (false negative). When in doubt, flag it.
- "moderate" severity: veiled threats, intimidation, coercive language, passive-aggressive threats,
  sarcasm that implies harm, boundary-testing behavior.
- "severe" severity: explicit threats of violence, kidnapping, stalking, sexual abuse, terrorism,
  extortion with bodily harm, or any language that indicates imminent danger.
- If there is NO threat detected, set is_threat to false and leave other fields null.

You MUST respond with valid JSON matching this exact schema:
{
  "is_threat": boolean,
  "severity": "moderate" | "severe" | null,
  "offense_title": "one-line description of the offense" | null,
  "section_reference": "18 U.S.C. Section XXX" | null,
  "category": "category name from the legal code" | null,
  "confidence_score": float between 0.0 and 1.0
}"""


async def retrieve_legal_context(transcript: str) -> list[dict]:
    """
    Embed the transcript and search Actian VectorAI DB for
    the most relevant Title 18 legal code sections.
    """
    settings = get_settings()
    client = await get_actian_client()

    query_vector = embed_text(transcript)

    results = await client.search(
        settings.LEGAL_CODES_COLLECTION,
        query_vector,
        top_k=settings.VECTOR_SEARCH_TOP_K,
        with_payload=True,
    )

    legal_context = []
    for result in results:
        if result.payload:
            legal_context.append({
                "title": result.payload["title"],
                "category": result.payload["category"],
                "chapter": result.payload["chapter"],
                "content": result.payload["content"],
                "relevance_score": result.score,
            })

    return legal_context


def _build_user_prompt(transcript: str, legal_context: list[dict]) -> str:
    """Construct the user message with transcript and retrieved legal context."""
    context_block = "\n\n".join(
        f"[{i+1}] {ctx['title']} (Chapter: {ctx['chapter']}, Category: {ctx['category']})\n{ctx['content']}"
        for i, ctx in enumerate(legal_context)
    )

    return f"""CONVERSATION TRANSCRIPT:
{transcript}

RELEVANT LEGAL CODE SECTIONS (from Title 18, US Federal Criminal Code):
{context_block}

Analyze the transcript for threats. Respond with JSON only."""


async def classify_threat(transcript: str) -> ThreatClassification:
    """
    Full RAG classification: retrieve legal context from Actian,
    then use Groq LLM to classify the threat.
    """
    settings = get_settings()
    groq = get_groq_client()

    legal_context = await retrieve_legal_context(transcript)
    user_prompt = _build_user_prompt(transcript, legal_context)

    response = await groq.chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
        max_tokens=512,
    )

    raw_output = response.choices[0].message.content
    parsed = json.loads(raw_output)

    return ThreatClassification(**parsed)


async def write_alert_to_supabase(
    classification: ThreatClassification,
    user_id: str,
    thread_id: str | None = None,
    guardian_id: str | None = None,
    transcript_snippet: str | None = None,
) -> AlertRecord:
    """Write a threat alert record to the Supabase alerts table."""
    supabase = get_supabase_client()

    snippet = transcript_snippet
    if snippet is None and classification.is_threat:
        snippet = "(transcript available on request)"

    record = {
        "user_id": user_id,
        "thread_id": thread_id,
        "guardian_id": guardian_id,
        "severity": classification.severity,
        "offense_title": classification.offense_title,
        "section_reference": classification.section_reference,
        "category": classification.category,
        "confidence_score": classification.confidence_score,
        "transcript_snippet": snippet,
    }

    result = supabase.table("alerts").insert(record).execute()
    inserted = result.data[0]

    return AlertRecord(**inserted)


async def analyze_transcript(
    transcript: str,
    user_id: str,
    thread_id: str | None = None,
    guardian_id: str | None = None,
) -> dict:
    """
    End-to-end pipeline: classify transcript, and if a threat is detected,
    persist the alert to Supabase for the guardian dashboard.

    Returns both the classification and the alert record (if created).
    """
    classification = await classify_threat(transcript)

    alert = None
    if classification.is_threat:
        snippet = transcript[:200] if len(transcript) > 200 else transcript
        try:
            alert = await write_alert_to_supabase(
                classification=classification,
                user_id=user_id,
                thread_id=thread_id,
                guardian_id=guardian_id,
                transcript_snippet=snippet,
            )
            logger.info("Alert created for user %s: %s", user_id, classification.severity)
        except Exception as e:
            logger.error("Failed to write alert to Supabase: %s", str(e))
            raise

    return {
        "classification": classification,
        "alert": alert,
    }
