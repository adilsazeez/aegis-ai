import os
import ssl
import json
import asyncio
from dotenv import load_dotenv

load_dotenv()

# Fix SSL certificate verification on macOS (AssemblyAI streaming connection)
try:
    import certifi
    _cafile = certifi.where()
    ssl.create_default_https_context = lambda: ssl.create_default_context(cafile=_cafile)
    os.environ.setdefault("SSL_CERT_FILE", _cafile)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", _cafile)
except Exception:
    pass

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from supabase.lib.client_options import ClientOptions
import assemblyai as aai
from auth_utils import get_current_user
from fastapi import Depends
from pydantic import BaseModel

# Setup Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("WARNING: SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not found in environment variables.")
    supabase: Client = None
else:
    # Explicitly set the schema to 'public'
    supabase: Client = create_client(
        SUPABASE_URL,
        SUPABASE_KEY,
        options=ClientOptions(schema="public")
    )

# Setup AssemblyAI
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
aai.settings.api_key = ASSEMBLYAI_API_KEY

app = FastAPI()


class ThreadCreate(BaseModel):
    initial_context: str = ""


# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Loosen for debugging
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def read_root():
    return {"message": "Hello from FastAPI backend!"}


@app.get('/settings')
async def get_settings():
    return {"settings": {}}


@app.get("/api/health")
async def health_check():
    return {"status": "ok"}


@app.websocket("/ws/{thread_id}")
async def monitor_audio(websocket: WebSocket, thread_id: str):
    print(f"WS CONNECTION ATTEMPT: thread_id={thread_id}")
    await websocket.accept()
    print(f"WS ACCEPTED: {thread_id}")

    from assemblyai.streaming.v3 import (
        StreamingClient,
        StreamingClientOptions,
        StreamingEvents,
        StreamingParameters,
        BeginEvent,
        TurnEvent,
        StreamingError,
        TerminationEvent,
    )
    import queue
    import threading

    import time as _time

    audio_queue = queue.Queue()
    loop = asyncio.get_running_loop()
    session_location = {"lat": None, "lon": None}
    last_analysis_time = {"t": 0.0}
    session_transcript_parts: list = []  # accumulate all turns for richer context
    ANALYSIS_COOLDOWN = 15  # seconds between Groq calls per session
    analysis_lock = asyncio.Lock()  # only one RAG+Groq run at a time (voice or chat) to avoid timeouts

    def on_turn(client, event: TurnEvent):
        if not event.transcript: return
        sentence = event.transcript
        is_final = event.end_of_turn
        lat, lon = session_location["lat"], session_location["lon"]

        async def process():
            try:
                if is_final and supabase:
                    session_transcript_parts.append(sentence)

                    log_res = supabase.table("logs").insert({
                        "thread_id": thread_id,
                        "content": sentence,
                        "latitude": lat,
                        "longitude": lon,
                    }).execute()
                    log_id = log_res.data[0]["id"] if log_res.data else None

                    now = _time.time()
                    since_last = now - last_analysis_time["t"]
                    # Check urgency across accumulated transcript, not just current fragment
                    accumulated = " ".join(session_transcript_parts).lower()
                    lower_sentence = (sentence or "").lower()
                    urgent_terms = ["danger", "in danger", "help me", "emergency", "please help",
                                    "unsafe", "kill", "murder", "stalk", "harass", "abuse",
                                    "scared", "threatened", "suicide"]
                    is_urgent = any(t in lower_sentence for t in urgent_terms)
                    if since_last < ANALYSIS_COOLDOWN and not is_urgent:
                        print(f"[RAG] Skipping analysis (cooldown {ANALYSIS_COOLDOWN - since_last:.0f}s remaining)")
                        await websocket.send_json(
                            {"transcript": sentence, "is_final": is_final, "risk": 0, "action": "Analyzing..."})
                        return
                    if since_last < ANALYSIS_COOLDOWN and is_urgent:
                        print("[RAG] Urgent wording detected; bypassing cooldown")

                    last_analysis_time["t"] = now

                    async with analysis_lock:
                        # Send accumulated transcript (last ~500 chars) so the LLM has full context
                        full_transcript = " ".join(session_transcript_parts)
                        if len(full_transcript) > 500:
                            full_transcript = full_transcript[-500:]
                        print(f"[RAG] Running analysis for: {full_transcript[:80]}...")

                        thread_row = supabase.table("threads").select("user_id, initial_context").eq("id", thread_id).limit(1).execute()
                        row0 = thread_row.data[0] if thread_row.data else {}
                        user_id = row0.get("user_id")
                        written_context = (row0.get("initial_context") or "").strip()

                        from rag_search import search_prosocial_chunks
                        from title18_loader import get_title18_context
                        from risk_analysis import assess_danger_with_rag

                        chunks = await search_prosocial_chunks(full_transcript)
                        print(f"[RAG] Got {len(chunks)} prosocial chunks")
                        title18 = get_title18_context()
                        analysis = await assess_danger_with_rag(
                            transcript=full_transcript,
                            prosocial_chunks=chunks,
                            title18_context=title18,
                            written_context=written_context if written_context else None,
                            location={"lat": lat, "lon": lon} if (lat is not None and lon is not None) else None,
                        )
                        print(f"[RAG] Analysis result: score={analysis.get('score')}, label={analysis.get('label')}, reason={analysis.get('reason', '')[:60]}")
                        if log_id and user_id:
                            try:
                                score_val = analysis.get("score", 0)
                                supabase.table("risk_analyses").insert({
                                    "thread_id": thread_id,
                                    "log_id": log_id,
                                    "user_id": user_id,
                                    "threat_category": analysis.get("threat_category"),
                                    "severity": analysis.get("severity"),
                                    "label": analysis.get("label"),
                                    "score": int(round(float(score_val))) if score_val is not None else 0,
                                    "reason": analysis.get("reason"),
                                    "narrative_analysis": analysis.get("narrative_analysis"),
                                    "relevant_law": analysis.get("relevant_law"),
                                }).execute()
                                print(f"[RAG] Inserted into risk_analyses")
                            except Exception as insert_err:
                                print(f"risk_analyses insert skipped: {insert_err}")
                        risk = analysis.get("score", 0)
                        action = analysis.get("reason") or "Analyzing..."
                else:
                    risk = 0
                    action = "Listening..."
                await websocket.send_json(
                    {"transcript": sentence, "is_final": is_final, "risk": risk, "action": action})
            except Exception as e:
                print(f"WS SEND ERROR: {e}")
                try:
                    await websocket.send_json(
                        {"transcript": sentence, "is_final": is_final, "risk": 0, "action": "Analysis unavailable."})
                except Exception:
                    pass

        asyncio.run_coroutine_threadsafe(process(), loop)

    client = StreamingClient(
        options=StreamingClientOptions(api_key=aai.settings.api_key)
    )
    client.on(StreamingEvents.Turn, on_turn)
    client.on(StreamingEvents.Error, lambda c, e: print(f"AAI Error: {e}"))

    client.connect(StreamingParameters(sample_rate=16000))

    def audio_generator():
        while True:
            chunk = audio_queue.get()
            if chunk is None: break
            yield chunk

    threading.Thread(target=lambda: client.stream(audio_generator()), daemon=True).start()

    try:
        while True:
            data = await websocket.receive()
            if data.get("type") == "websocket.disconnect":
                print(f"WS DISCONNECT MESSAGE RECEIVED: {thread_id}")
                break
            
            if "bytes" in data:
                audio_queue.put(data["bytes"])
            elif "text" in data:
                msg = json.loads(data["text"])
                if msg.get("type") == "location":
                    session_location["lat"], session_location["lon"] = msg.get("lat"), msg.get("lon")
                elif msg.get("type") == "chat" and msg.get("text"):
                    chat_text = msg["text"].strip()
                    if chat_text and supabase:
                        session_transcript_parts.append(chat_text)
                        print(f"[CHAT-RAG] Received typed text: {chat_text[:80]}...")
                        async with analysis_lock:
                            try:
                                log_res = supabase.table("logs").insert({
                                    "thread_id": thread_id,
                                    "content": chat_text,
                                }).execute()
                                log_id = log_res.data[0]["id"] if log_res.data else None

                                thread_row = supabase.table("threads").select("user_id, initial_context").eq("id", thread_id).limit(1).execute()
                                row0 = thread_row.data[0] if thread_row.data else {}
                                user_id = row0.get("user_id")
                                written_context = (row0.get("initial_context") or "").strip()

                                from rag_search import search_prosocial_chunks
                                from title18_loader import get_title18_context
                                from risk_analysis import assess_danger_with_rag

                                full_transcript = " ".join(session_transcript_parts)
                                if len(full_transcript) > 500:
                                    full_transcript = full_transcript[-500:]

                                chunks = await search_prosocial_chunks(full_transcript)
                                print(f"[CHAT-RAG] Got {len(chunks)} prosocial chunks")
                                title18 = get_title18_context()
                                analysis = await assess_danger_with_rag(
                                    transcript=full_transcript,
                                    prosocial_chunks=chunks,
                                    title18_context=title18,
                                    written_context=written_context if written_context else None,
                                )
                                print(f"[CHAT-RAG] Result: score={analysis.get('score')}, label={analysis.get('label')}, reason={analysis.get('reason', '')[:60]}")

                                if log_id and user_id:
                                    score_val = analysis.get("score", 0)
                                    supabase.table("risk_analyses").insert({
                                        "thread_id": thread_id,
                                        "log_id": log_id,
                                        "user_id": user_id,
                                        "threat_category": analysis.get("threat_category"),
                                        "severity": analysis.get("severity"),
                                        "label": analysis.get("label"),
                                        "score": int(round(float(score_val))) if score_val is not None else 0,
                                        "reason": analysis.get("reason"),
                                        "narrative_analysis": analysis.get("narrative_analysis"),
                                        "relevant_law": analysis.get("relevant_law"),
                                    }).execute()
                                    print(f"[CHAT-RAG] Inserted into risk_analyses")

                                risk = analysis.get("score", 0)
                                action = analysis.get("reason") or "Analyzed typed input."
                                await websocket.send_json({
                                    "risk": risk,
                                    "action": action,
                                })
                            except Exception as chat_err:
                                print(f"[CHAT-RAG] Error: {chat_err}")
                                await websocket.send_json({
                                    "risk": 0,
                                    "action": "Text analysis failed.",
                                })
    except WebSocketDisconnect:
        print(f"Disconnected: {thread_id}")
    finally:
        audio_queue.put(None)


@app.get("/api/profile")
async def get_profile(user=Depends(get_current_user)):
    if not supabase:
        return {"id": user.id, "email": user.email, "is_enrolled": False}

    try:
        res = supabase.table("profiles").select("*").eq("id", user.id).execute()
        if res.data:
            return res.data[0]
        return {"id": user.id, "email": user.email, "is_enrolled": False}
    except Exception as e:
        print(f"Profile error: {e}")
        return {"id": user.id, "email": user.email, "is_enrolled": False}


@app.get("/api/profile/role")
async def get_profile_role(user=Depends(get_current_user)):
    if not supabase:
        return {"account_role": "both"}
    try:
        res = supabase.table("profiles").select("account_role").eq("id", user.id).execute()
        if res.data:
            role = res.data[0].get("account_role", "both")
            return {"account_role": role}
        return {"account_role": "both"}
    except Exception as e:
        print(f"Role fetch error: {e}")
        return {"account_role": "both"}


class RoleUpdate(BaseModel):
    account_role: str
    is_enrolled: bool = None


@app.post("/api/profile/role")
async def update_profile_role(data: RoleUpdate, user=Depends(get_current_user)):
    if not supabase:
        return {"message": "Supabase not configured"}

    role = data.account_role

    if role not in ["guardian", "both"]:
        return {"error": "Invalid role. Must be 'guardian' or 'both'"}, 400

    try:
        update_data = {"account_role": role}
        if data.is_enrolled is not None:
            update_data["is_enrolled"] = data.is_enrolled
            
        supabase.table("profiles").update(update_data).eq("id", user.id).execute()
        return {"message": f"Account role updated to {role}"}
    except Exception as e:
        print(f"Role update error: {e}")
        return {"error": str(e)}, 400


@app.get("/api/history")
async def get_history(user=Depends(get_current_user)):
    if not supabase:
        return []

    try:
        # Fetch threads with their logs
        threads_res = supabase.table("threads").select("*, logs(*)").eq("user_id", user.id).order("created_at",
                                                                                                  desc=True).execute()
        return threads_res.data
    except Exception as e:
        print(f"History error: {e}")
        return []


@app.get("/api/guarding")
async def get_guarding(user=Depends(get_current_user)):
    """
    Returns the guarding relationships where this user is the guardian.
    Includes the profile of the person who added them.
    """
    if not supabase:
        return []

    try:
        # Fetch relationships where this user is the guardian
        res = supabase.table("guardians").select("*, profiles:user_id(*)").eq("guardian_id", user.id).execute()
        return res.data
    except Exception as e:
        print(f"Guarding error: {e}")
        return []


@app.post("/api/guardians/accept/{relationship_id}")
async def accept_guardian(relationship_id: str, user=Depends(get_current_user)):
    if not supabase:
        return {"message": "Supabase not configured"}
    try:
        # Update status to active only if this user is the guardian for this relationship
        supabase.table("guardians").update({"status": "active"}).eq("id", relationship_id).eq("guardian_id",
                                                                                              user.id).execute()
        return {"message": "Guardian request accepted"}
    except Exception as e:
        print(f"Accept error: {e}")
        return {"error": str(e)}, 400


@app.get("/api/my-guardians")
async def get_my_guardians(user=Depends(get_current_user)):
    if not supabase:
        return []
    try:
        # Fetch guardians added by the current user
        res = supabase.table("guardians").select("*").eq("user_id", user.id).execute()
        return res.data
    except Exception as e:
        print(f"My Guardians error: {e}")
        return []


@app.get("/api/notifications")
async def get_notifications(user=Depends(get_current_user)):
    if not supabase:
        return []
    try:
        res = supabase.table("notifications").select("*").eq("user_id", user.id).order("created_at",
                                                                                       desc=True).execute()
        return res.data
    except Exception as e:
        print(f"Notifications error: {e}")
        return []


@app.post("/api/notifications/read/{notification_id}")
async def mark_notification_read(notification_id: str, user=Depends(get_current_user)):
    if not supabase:
        return {"message": "Supabase not configured"}
    try:
        supabase.table("notifications").update({"is_read": True}).eq("id", notification_id).eq("user_id",
                                                                                               user.id).execute()
        return {"message": "Notification marked as read"}
    except Exception as e:
        print(f"Mark read error: {e}")
        return {"error": str(e)}, 400


@app.post("/api/enroll-voice")
async def enroll_voice(file: UploadFile = File(...), user=Depends(get_current_user)):
    """
    Accepts a .wav file, generates a voice fingerprint embedding,
    and updates the user's profile.
    """
    if not supabase:
        return {"message": "Supabase not configured"}

    try:
        # Save uploaded file temporarily for processing
        temp_path = f"temp_{user.id}.wav"
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # For now, we simulate the embedding generation (384-dimensional vector)
        import numpy as np
        embedding = np.random.uniform(-1, 1, 384).tolist()

        # Update user profile in Supabase
        # First ensure the profile exists
        supabase.table("profiles").upsert({
            "id": user.id,
            "email": user.email,
            "voice_fingerprint": embedding,
            "is_enrolled": True
        }).execute()

        return {"message": "Voice enrolled successfully"}
    except Exception as e:
        print(f"Enrollment error: {e}")
        return {"error": str(e)}, 400
    finally:
        if os.path.exists("temp_enrollment.wav"):
            os.remove("temp_enrollment.wav")


@app.post("/api/guardians/add")
async def add_guardian(
        guardian_email: str = Body(..., embed=True),
        guardian_phone: str = Body(None, embed=True),
        user=Depends(get_current_user)
):
    """
    Adds a guardian to the user's circle.
    If the email exists in profiles, links them.
    Otherwise, marks as a guest guardian.
    """
    if not supabase:
        return {"message": "Supabase not configured"}

    try:
        # Check if guardian exists as a user
        res = supabase.table("profiles").select("id").eq("email", guardian_email).execute()
        guardian_id = res.data[0]["id"] if res.data else None

        # Insert into guardians table
        supabase.table("guardians").insert({
            "user_id": user.id,
            "guardian_id": guardian_id,
            "guardian_email": guardian_email,
            "guardian_phone": guardian_phone,
            "status": "pending"
        }).execute()

        # If guardian is a registered user, send them a notification
        if guardian_id:
            try:
                # Fetch adder's name or email for the notification message
                res_adder = supabase.table("profiles").select("full_name, email").eq("id", user.id).execute()
                adder_name = res_adder.data[0].get("full_name") or res_adder.data[0].get(
                    "email") if res_adder.data else user.email

                supabase.table("notifications").insert({
                    "user_id": guardian_id,
                    "type": "guardian_added",
                    "title": "New Guarding Request",
                    "message": f"{adder_name} has added you as their guardian. You can now monitor their safety sessions.",
                    "link": "/guardians",
                    "is_read": False
                }).execute()
            except Exception as notif_err:
                print(f"Failed to send guardian notification: {notif_err}")

        return {
            "message": "Guardian added",
            "is_guest": guardian_id is None,
            "status": "pending"
        }
    except Exception as e:
        print(f"Add guardian error: {e}")
        return {"error": str(e)}, 400


@app.delete("/api/guardians/{relationship_id}")
async def delete_guardian(relationship_id: str, user=Depends(get_current_user)):
    if not supabase:
        return {"message": "Supabase not configured"}
    try:
        # Allow deletion if user is either the one who added (user_id) or the guardian (guardian_id)
        res = supabase.table("guardians").delete().eq("id", relationship_id).or_(f"user_id.eq.{user.id},guardian_id.eq.{user.id}").execute()
        return {"message": "Guardian relationship removed"}
    except Exception as e:
        print(f"Delete guardian error: {e}")
        return {"error": str(e)}, 400


@app.post("/api/threads")
async def create_thread(data: ThreadCreate, user=Depends(get_current_user)):
    print(f"CREATE_THREAD: Received data={data} for user={user.id}")
    if not supabase:
        print("CREATE_THREAD: Supabase not configured, using fallback")
        import uuid
        return {"id": str(uuid.uuid4()), "message": "Development mode (no Supabase)"}

    try:
        initial_context = data.initial_context
        print(f"CREATE_THREAD: Attempting insert into threads table (context: {initial_context})")
        response = supabase.table("threads").insert({
            "user_id": user.id,
            "initial_context": initial_context
        }).execute()

        if response.data:
            thread_row = response.data[0]
            thread_id = thread_row.get("id")
            print(f"CREATE_THREAD: Success! Thread ID: {thread_id}")

            # Run RAG analysis on written context if non-empty
            if initial_context and initial_context.strip():
                try:
                    from rag_search import search_prosocial_chunks
                    from title18_loader import get_title18_context
                    from risk_analysis import assess_danger_with_rag

                    print(f"[TEXT-RAG] Running analysis on written context: {initial_context[:80]}...")
                    log_res = supabase.table("logs").insert({
                        "thread_id": thread_id,
                        "content": initial_context,
                    }).execute()
                    log_id = log_res.data[0]["id"] if log_res.data else None

                    chunks = await search_prosocial_chunks(initial_context)
                    print(f"[TEXT-RAG] Got {len(chunks)} prosocial chunks")
                    title18 = get_title18_context()
                    analysis = await assess_danger_with_rag(
                        transcript=initial_context,
                        prosocial_chunks=chunks,
                        title18_context=title18,
                        written_context=initial_context,
                    )
                    print(f"[TEXT-RAG] Result: score={analysis.get('score')}, label={analysis.get('label')}, reason={analysis.get('reason', '')[:60]}")
                    if log_id:
                        score_val = analysis.get("score", 0)
                        supabase.table("risk_analyses").insert({
                            "thread_id": thread_id,
                            "log_id": log_id,
                            "user_id": user.id,
                            "threat_category": analysis.get("threat_category"),
                            "severity": analysis.get("severity"),
                            "label": analysis.get("label"),
                            "score": int(round(float(score_val))) if score_val is not None else 0,
                            "reason": analysis.get("reason"),
                            "narrative_analysis": analysis.get("narrative_analysis"),
                            "relevant_law": analysis.get("relevant_law"),
                        }).execute()
                        print(f"[TEXT-RAG] Inserted into risk_analyses")
                except Exception as text_rag_err:
                    print(f"[TEXT-RAG] Text analysis skipped: {text_rag_err}")

            return thread_row

        print("CREATE_THREAD: Insert returned no data, using fallback mock ID")
        return {"id": "mock-thread-id"}
    except Exception as e:
        print(f"CREATE_THREAD ERROR: {e}")
        try:
            print("CREATE_THREAD: Attempting fallback insert without initial_context")
            response = supabase.table("threads").insert({
                "user_id": user.id
            }).execute()
            if response.data:
                print(f"CREATE_THREAD: Fallback success! Thread ID: {response.data[0].get('id')}")
                return response.data[0]
        except Exception as e2:
            print(f"CREATE_THREAD FALLBACK ERROR: {e2}")

        import uuid
        mock_id = str(uuid.uuid4())
        print(f"CREATE_THREAD: Returning mock UUID: {mock_id}")
        return {"id": mock_id, "error": str(e)}


class AnalyzeTextRequest(BaseModel):
    thread_id: str
    text: str


@app.post("/api/analyze-text")
async def analyze_text(data: AnalyzeTextRequest, user=Depends(get_current_user)):
    """
    Run RAG + Groq analysis on typed text and insert the result into risk_analyses.
    Works independently from the voice/WebSocket path.
    """
    if not supabase:
        return {"error": "Supabase not configured"}

    text = (data.text or "").strip()
    if not text:
        return {"error": "Empty text, nothing to analyze"}

    thread_id = data.thread_id
    print(f"[ANALYZE-TEXT] thread={thread_id} text={text[:80]}...")

    try:
        from rag_search import search_prosocial_chunks
        from title18_loader import get_title18_context
        from risk_analysis import assess_danger_with_rag

        # Fetch written context from the thread for additional LLM context
        written_context = ""
        try:
            thread_res = supabase.table("threads").select("initial_context").eq("id", thread_id).execute()
            if thread_res.data:
                written_context = thread_res.data[0].get("initial_context") or ""
        except Exception:
            pass

        log_res = supabase.table("logs").insert({
            "thread_id": thread_id,
            "content": text,
        }).execute()
        log_id = log_res.data[0]["id"] if log_res.data else None

        chunks = await search_prosocial_chunks(text)
        print(f"[ANALYZE-TEXT] Got {len(chunks)} prosocial chunks")
        title18 = get_title18_context()

        # Combine both the typed text and the original written context
        full_context = f"{written_context}\n{text}".strip() if written_context else text

        analysis = await assess_danger_with_rag(
            transcript=text,
            prosocial_chunks=chunks,
            title18_context=title18,
            written_context=full_context,
        )
        print(f"[ANALYZE-TEXT] Result: score={analysis.get('score')}, label={analysis.get('label')}, reason={analysis.get('reason', '')[:60]}")

        if log_id:
            score_val = analysis.get("score", 0)
            supabase.table("risk_analyses").insert({
                "thread_id": thread_id,
                "log_id": log_id,
                "user_id": user.id,
                "threat_category": analysis.get("threat_category"),
                "severity": analysis.get("severity"),
                "label": analysis.get("label"),
                "score": int(round(float(score_val))) if score_val is not None else 0,
                "reason": analysis.get("reason"),
                "narrative_analysis": analysis.get("narrative_analysis"),
                "relevant_law": analysis.get("relevant_law"),
            }).execute()
            print(f"[ANALYZE-TEXT] Inserted into risk_analyses")

        return {
            "status": "ok",
            "analysis": analysis,
            "log_id": log_id,
        }
    except Exception as e:
        print(f"[ANALYZE-TEXT] Error: {e}")
        return {"error": str(e)}

