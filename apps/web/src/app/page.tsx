"use client";

import { useState, useRef, useEffect } from "react";
import { createClient } from "@/utils/supabase/client";
import { useRouter } from "next/navigation";
import { AuthUser as User } from "@supabase/supabase-js";

export default function Home() {
  const [activeTab, setActiveTab] = useState<"black-box" | "history" | "guardians" | "notifications" | "profile">("black-box");
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [threadId, setThreadId] = useState<string | null>(null);
  const [risk, setRisk] = useState<number>(0);
  const [action, setAction] = useState<string>("Shadow is idle.");
  const [transcripts, setTranscripts] = useState<string[]>([]);
  const [currentTranscript, setCurrentTranscript] = useState<string>("");
  const [status, setStatus] = useState<"idle" | "connecting" | "active" | "error">("idle");
  const [location, setLocation] = useState<{ lat: number, lon: number } | null>(null);
  const [manualInput, setManualInput] = useState("");
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Onboarding, History, Guarding & Notifications state
  const [profile, setProfile] = useState<{ is_enrolled: boolean, account_role?: string, email?: string, full_name?: string } | null>(null);
  const [showOnboarding, setShowOnboarding] = useState(false);
  const [onboardingStep, setOnboardingStep] = useState<0 | 1 | 2>(0);
  const [showAddGuardianModal, setShowAddGuardianModal] = useState(false);
  const [showInitiationModal, setShowInitiationModal] = useState(false);
  const [sessionContext, setSessionContext] = useState("");
  const [monitoringMode, setMonitoringMode] = useState<"audio" | "text" | "both">("both");

  const [history, setHistory] = useState<any[]>([]);
  const [guarding, setGuarding] = useState<any[]>([]);
  const [myGuardians, setMyGuardians] = useState<any[]>([]);
  const [notifications, setNotifications] = useState<any[]>([]);
  const [isRecording, setIsRecording] = useState(false);
  const [guardianEmail, setGuardianEmail] = useState("");
  const [guardianPhone, setGuardianPhone] = useState("");
  const [isSubmittingGuardian, setIsSubmittingGuardian] = useState(false);

  const supabase = createClient();
  const router = useRouter();

  const ws = useRef<WebSocket | null>(null);
  const audioContext = useRef<AudioContext | null>(null);
  const audioWorklet = useRef<ScriptProcessorNode | null>(null);
  const audioStream = useRef<MediaStream | null>(null);
  const locationInterval = useRef<NodeJS.Timeout | null>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);

  const fetchBaseData = async () => {
    try {
      setIsLoading(true);
      const { data: { session } } = await supabase.auth.getSession();
      if (!session) return;
      const headers = { "Authorization": `Bearer ${session?.access_token}` };

      const [profRes, histRes, guardRes, myGuardRes, notifRes] = await Promise.all([
        fetch("http://localhost:8000/api/profile", { headers }),
        fetch("http://localhost:8000/api/history", { headers }),
        fetch("http://localhost:8000/api/guarding", { headers }),
        fetch("http://localhost:8000/api/my-guardians", { headers }),
        fetch("http://localhost:8000/api/notifications", { headers })
      ]);

      const [profData, histData, guardData, myGuardData, notifData] = await Promise.all([
        profRes.json(), histRes.json(), guardRes.json(), myGuardRes.json(), notifRes.json()
      ]);

      setProfile(profData);
      if (profData && !profData.is_enrolled) {
        setShowOnboarding(true);
        setOnboardingStep(0);
      }
      if (profData?.account_role === "guardian") {
        setActiveTab("guardians");
      }

      setHistory(histData || []);
      setGuarding(guardData || []);
      setMyGuardians(myGuardData || []);
      setNotifications(notifData || []);
    } catch (err) {
      console.error("Base data fetch failed:", err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    const init = async () => {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) {
        router.push("/login");
        return;
      }
      setUser(user);
      await fetchBaseData();
    };
    init();
  }, [router, supabase]);

  const handleSignOut = async () => {
    await supabase.auth.signOut();
    router.push("/login");
  };

  const markNotificationRead = async (id: string) => {
    const { data: { session } } = await supabase.auth.getSession();
    await fetch(`http://localhost:8000/api/notifications/read/${id}`, {
      method: "POST",
      headers: { "Authorization": `Bearer ${session?.access_token}` }
    });
    setNotifications(prev => prev.map(n => n.id === id ? { ...n, is_read: true } : n));
  };

  const handleAcceptGuardian = async (id: string) => {
    const { data: { session } } = await supabase.auth.getSession();
    await fetch(`http://localhost:8000/api/guardians/accept/${id}`, {
      method: "POST",
      headers: { "Authorization": `Bearer ${session?.access_token}` }
    });
    await fetchBaseData();
  };

  const handleRemoveGuardian = async (id: string) => {
    try {
      const { data: { session } } = await supabase.auth.getSession();
      if (!session) return;

      const res = await fetch(`http://localhost:8000/api/guardians/${id}`, {
        method: "DELETE",
        headers: { "Authorization": `Bearer ${session?.access_token}` }
      });

      if (res.ok) {
        await fetchBaseData();
      }
    } catch (err) {
      console.error("Failed to remove guardian:", err);
    }
  };

  const handleUpdateRole = async (isGuardian: boolean) => {
    try {
      const { data: { session } } = await supabase.auth.getSession();
      if (!session) return;
      const role = isGuardian ? "guardian" : "both";
      const res = await fetch("http://localhost:8000/api/profile/role", {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${session?.access_token}`,
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ 
          account_role: role,
          is_enrolled: role === "guardian" ? true : undefined
        })
      });

      if (res.ok) {
        setProfile(prev => prev ? { ...prev, account_role: role, is_enrolled: role === "guardian" ? true : prev.is_enrolled } : null);
        if (role === "guardian") {
          setActiveTab("guardians");
          setShowOnboarding(false);
        } else {
          setOnboardingStep(1);
        }
      }
    } catch (err) {
      console.error("Failed to update role:", err);
    }
  };

  // Auto-scroll chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [transcripts, currentTranscript]);

  const sendManualChat = (e: React.FormEvent) => {
    e.preventDefault();
    if (!manualInput.trim() || ws.current?.readyState !== WebSocket.OPEN) return;
    setAction("Analyzing typed message...");
    ws.current.send(JSON.stringify({ type: "chat", text: manualInput.trim() }));
    setTranscripts(prev => [...prev, manualInput.trim()].slice(-20));
    setManualInput("");
  };

  useEffect(() => {
    if (isMonitoring && status === "active" && ws.current?.readyState === WebSocket.OPEN) {
      locationInterval.current = setInterval(() => {
        navigator.geolocation.getCurrentPosition(
          (pos) => {
            const loc = { lat: pos.coords.latitude, lon: pos.coords.longitude };
            setLocation(loc);
            if (ws.current?.readyState === WebSocket.OPEN) {
              ws.current.send(JSON.stringify({ type: "location", ...loc }));
            }
          },
          (err) => console.error("Location access failed:", err),
          { enableHighAccuracy: true }
        );
      }, 3000);
    } else if (locationInterval.current) {
      clearInterval(locationInterval.current);
    }
    return () => { if (locationInterval.current) clearInterval(locationInterval.current); };
  }, [isMonitoring, status]);

  const startMonitoring = async () => {
    // Clear old session state
    setTranscripts([]);
    setCurrentTranscript("");
    setRisk(0);
    setAction("Shadow is idle.");

    setStatus("connecting");
    setShowInitiationModal(false);
    console.log("Starting monitoring session...");

    try {
      const { data: { session } } = await supabase.auth.getSession();
      if (!session) {
        console.error("No active session found");
        setStatus("error");
        return;
      }

      console.log("Creating session thread...");
      const res = await fetch("http://localhost:8000/api/threads", {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${session?.access_token}`,
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ initial_context: sessionContext || "" })
      });

      if (!res.ok) {
        const errText = await res.text();
        console.error("Thread creation failed with status:", res.status, errText);
        setStatus("error");
        return;
      }

      const data = await res.json();
      console.log("Thread created successfully:", data);

      if (!data || !data.id) {
        console.error("No thread ID in response:", data);
        setStatus("error");
        return;
      }

      setThreadId(data.id);

      const wsUrl = `ws://127.0.0.1:8000/ws/${data.id}`;
      console.log("Attempting WebSocket connection to:", wsUrl);

      ws.current = new WebSocket(wsUrl);

      ws.current.onopen = () => {
        console.log("WebSocket connected successfully!");
        setStatus("active");
        setIsMonitoring(true);
        if (monitoringMode === "audio" || monitoringMode === "both") {
          console.log("Starting audio recording...");
          startRecording();
        }
      };

      ws.current.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          setRisk(msg.risk || 0);
          setAction(msg.action || "Shadow is monitoring...");
          if (msg.transcript) {
            if (msg.is_final) {
              setTranscripts((prev) => [...prev, msg.transcript].slice(-20));
              setCurrentTranscript("");
            } else {
              setCurrentTranscript(msg.transcript);
            }
          }
        } catch (e) {
          console.error("Failed to parse WebSocket message:", event.data, e);
        }
      };

      ws.current.onclose = (event) => {
        console.log("WebSocket closed:", event.code, event.reason);
        stopMonitoring();
      };

      ws.current.onerror = (err) => {
        console.error("WebSocket error observed:", err);
        setStatus("error");
      };

    } catch (err) {
      console.error("Detailed monitoring initiation error:", err);
      setStatus("error");
    }
  };

  const startRecording = async () => {
    try {
      audioStream.current = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioContext.current = new AudioContext({ sampleRate: 16000 });
      const source = audioContext.current.createMediaStreamSource(audioStream.current);
      audioWorklet.current = audioContext.current.createScriptProcessor(4096, 1, 1);
      audioWorklet.current.onaudioprocess = (event) => {
        if (ws.current?.readyState !== WebSocket.OPEN) return;
        const inputData = event.inputBuffer.getChannelData(0);
        const pcmData = new Int16Array(inputData.length);
        for (let i = 0; i < inputData.length; i++) {
          const s = Math.max(-1, Math.min(1, inputData[i]));
          pcmData[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }
        ws.current.send(pcmData.buffer);
      };
      source.connect(audioWorklet.current);
      audioWorklet.current.connect(audioContext.current.destination);
    } catch (err) {
      console.error("Mic access error:", err);
      setStatus("error");
    }
  };

  const stopMonitoring = () => {
    console.log("Stopping monitoring session...");
    setIsMonitoring(false);
    setStatus("idle");
    setCurrentTranscript("");
    audioWorklet.current?.disconnect();
    audioContext.current?.close();
    audioStream.current?.getTracks().forEach(track => track.stop());
    ws.current?.close();
  };

  const handleEnrollVoice = async () => {
    setIsRecording(true);
    setTimeout(async () => {
      try {
        const { data: { session } } = await supabase.auth.getSession();
        if (!session) return;
        const formData = new FormData();
        formData.append("file", new Blob(["dummy"], { type: "audio/wav" }), "voice.wav");
        await fetch("http://localhost:8000/api/enroll-voice", {
          method: "POST",
          headers: { "Authorization": `Bearer ${session?.access_token}` },
          body: formData
        });
        setIsRecording(false);
        setOnboardingStep(2);
      } catch (err) {
        console.error("Voice enrollment failed:", err);
        setIsRecording(false);
      }
    }, 2000);
  };

  const handleAddGuardian = async (e: React.FormEvent) => {
    e.preventDefault();
    if (isSubmittingGuardian) return;
    setIsSubmittingGuardian(true);
    try {
      const { data: { session } } = await supabase.auth.getSession();
      if (!session) return;
      await fetch("http://localhost:8000/api/guardians/add", {
        method: "POST",
        headers: { "Authorization": `Bearer ${session?.access_token}`, "Content-Type": "application/json" },
        body: JSON.stringify({ guardian_email: guardianEmail, guardian_phone: guardianPhone })
      });

      // Refresh data
      await fetchBaseData();

      // Clear form and close modal
      setGuardianEmail("");
      setGuardianPhone("");
      setShowAddGuardianModal(false);

      // If it was onboarding, finish it
      if (showOnboarding) {
        setProfile(prev => prev ? { ...prev, is_enrolled: true } : null);
        setTimeout(() => {
          setShowOnboarding(false);
          setActiveTab("black-box");
          setIsMonitoring(false);
          setStatus("idle");
        }, 500);
      }
    } catch (err) {
      console.error("Add guardian failed:", err);
    } finally {
      setIsSubmittingGuardian(false);
    }
  };

  const unreadCount = notifications.filter(n => !n.is_read).length;

  return (
    <div className="flex min-h-screen flex-col bg-[#070F1A] text-[#E5E7EB] font-sans selection:bg-[#14B8A6]/30 overflow-x-hidden relative">
      {/* Background Halo */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[1000px] h-[1000px] bg-[#14B8A6]/5 rounded-full blur-[120px]" />
        <div className="absolute -top-[10%] -left-[10%] w-[40%] h-[40%] bg-[#14B8A6]/3 rounded-full blur-[80px]" />
        <div className="absolute -bottom-[10%] -right-[10%] w-[40%] h-[40%] bg-[#14B8A6]/3 rounded-full blur-[80px]" />
      </div>

      <nav className="fixed top-6 left-1/2 -translate-x-1/2 z-50 px-2 py-2 bg-[#0F172A]/80 backdrop-blur-xl border border-[#0F766E]/30 rounded-full shadow-[0_24px_60px_rgba(0,0,0,0.5)] flex items-center gap-1">
        {isLoading ? (
          <div className="flex items-center gap-2 px-4 py-2">
            <div className="w-20 h-6 bg-[#111827] rounded-full animate-shimmer" />
            <div className="w-20 h-6 bg-[#111827] rounded-full animate-shimmer" />
            <div className="w-20 h-6 bg-[#111827] rounded-full animate-shimmer" />
            <div className="w-20 h-6 bg-[#111827] rounded-full animate-shimmer" />
          </div>
        ) : (
          <>
            {profile?.account_role !== "guardian" && (
              <>
                <button
                  onClick={() => setActiveTab("black-box")}
                  className={`px-6 py-2 rounded-full text-sm font-semibold transition-all border ${activeTab === "black-box" ? "bg-[#14B8A6] text-[#0B1120] border-[#14B8A6] shadow-[0_0_20px_rgba(20,184,166,0.3)]" : "text-[#9CA3AF] border-[#0F766E]/50 hover:text-[#E5E7EB] hover:border-[#14B8A6]/50"}`}
                >
                  black-box
                </button>
                <button
                  onClick={() => setActiveTab("history")}
                  className={`px-6 py-2 rounded-full text-sm font-semibold transition-all border ${activeTab === "history" ? "bg-[#14B8A6] text-[#0B1120] border-[#14B8A6] shadow-[0_0_20px_rgba(20,184,166,0.3)]" : "text-[#9CA3AF] border-[#0F766E]/50 hover:text-[#E5E7EB] hover:border-[#14B8A6]/50"}`}
                >
                  history
                </button>
              </>
            )}
            <button
              onClick={() => setActiveTab("guardians")}
              className={`px-6 py-2 rounded-full text-sm font-semibold transition-all border ${activeTab === "guardians" ? "bg-[#14B8A6] text-[#0B1120] border-[#14B8A6] shadow-[0_0_20px_rgba(20,184,166,0.3)]" : "text-[#9CA3AF] border-[#0F766E]/50 hover:text-[#E5E7EB] hover:border-[#14B8A6]/50"}`}
            >
              guardians
            </button>
            <button
              onClick={() => setActiveTab("notifications")}
              className={`px-6 py-2 rounded-full text-sm font-semibold transition-all relative border ${activeTab === "notifications" ? "bg-[#14B8A6] text-[#0B1120] border-[#14B8A6] shadow-[0_0_20px_rgba(20,184,166,0.3)]" : "text-[#9CA3AF] border-[#0F766E]/50 hover:text-[#E5E7EB] hover:border-[#14B8A6]/50"}`}
            >
              notifications
              {unreadCount > 0 && <span className="absolute -top-1 -right-1 flex h-4 w-4 items-center justify-center rounded-full bg-[#FF8559] text-[8px] font-bold text-white shadow-lg">{unreadCount}</span>}
            </button>
            <button
              onClick={() => setActiveTab("profile")}
              className={`px-6 py-2 rounded-full text-sm font-semibold transition-all border ${activeTab === "profile" ? "bg-[#14B8A6] text-[#0B1120] border-[#14B8A6] shadow-[0_0_20px_rgba(20,184,166,0.3)]" : "text-[#9CA3AF] border-[#0F766E]/50 hover:text-[#E5E7EB] hover:border-[#14B8A6]/50"}`}
            >
              profile
            </button>
          </>
        )}
      </nav>

      {isLoading && (
        <main className="flex-1 flex flex-col pt-32 px-4 max-w-4xl mx-auto w-full">
          <div className="w-48 h-10 bg-zinc-900 rounded-2xl mb-8 animate-shimmer" />
          <div className="space-y-4">
            <div className="h-32 bg-zinc-900/50 rounded-3xl animate-shimmer" />
            <div className="h-32 bg-zinc-900/50 rounded-3xl animate-shimmer" />
            <div className="h-32 bg-zinc-900/50 rounded-3xl animate-shimmer" />
          </div>
        </main>
      )}

      {!isLoading && activeTab === "black-box" && (
        <main className="flex-1 flex flex-col pt-24 pb-12 px-4 max-w-4xl mx-auto w-full relative z-10">
          {!isMonitoring ? (
            <div className="flex-1 flex flex-col items-center justify-center text-center">
              <div className="mb-8 p-4 rounded-2xl bg-[#0F172A]/50 border border-[#0F766E]/30 max-w-lg shadow-[0_24px_60px_rgba(0,0,0,0.5)]">
                <h2 className="text-xl font-bold mb-4 flex items-center justify-center gap-3">
                  <span className="w-1.5 h-6 bg-[#14B8A6] rounded-full" />
                  The Safety Shadow
                </h2>
                <p className="text-[#9CA3AF] text-sm leading-relaxed max-w-[520px]">Conversational guardian for high-stakes events. Real-time risk evaluation as you speak.</p>
              </div>
              <button
                onClick={() => {
                  setSessionContext("");
                  setMonitoringMode("both");
                  setShowInitiationModal(true);
                }}
                disabled={status === 'connecting'}
                className="group px-10 py-5 bg-[#14B8A6] text-[#0B1120] font-black rounded-full hover:bg-[#22C9B7] hover:scale-105 hover:shadow-[0_0_24px_rgba(20,184,166,0.5)] active:scale-95 active:bg-[#0F766E] transition-all flex items-center gap-3 uppercase tracking-widest"
              >
                INITIATE BLACK-BOX
                <svg className="w-4 h-4 text-[#0B1120] group-hover:translate-x-1 transition-transform" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><path d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>
              </button>
              {status === "error" && (
                <p className="mt-4 text-[#FF8559] text-xs font-bold uppercase tracking-widest animate-pulse">Connection Failed. Please check console for details.</p>
              )}
            </div>
          ) : (
            <div className="flex-1 flex flex-col">
              <div className="flex items-center justify-between mb-8 p-6 bg-zinc-900/30 border border-zinc-800 rounded-2xl backdrop-blur-sm">
                <div>
                  <h3 className={`text-xl font-bold ${risk > 75 ? 'text-red-500' : 'text-zinc-100'}`}>{action}</h3>
                  {sessionContext && <p className="text-xs text-zinc-500 mt-1 italic line-clamp-1">Context: {sessionContext}</p>}
                </div>
                <div className="text-right ml-6">
                  <span className="text-4xl font-black">{risk.toFixed(0)}%</span>
                  <div className="w-24 bg-zinc-800 h-1 mt-2 rounded-full overflow-hidden">
                    <div className={`h-full transition-all duration-1000 ${risk > 75 ? 'bg-red-500' : 'bg-green-500'}`} style={{ width: `${risk}%` }} />
                  </div>
                </div>
              </div>
              <div className="flex-1 space-y-4 overflow-y-auto mb-4 pr-2 custom-scrollbar">
                {transcripts.map((t, i) => (<div key={i} className="flex flex-col items-end gap-1 ml-auto max-w-[85%]"><div className="px-4 py-3 rounded-2xl rounded-tr-none bg-white/5 border border-zinc-800/50 text-zinc-200 text-sm leading-relaxed">{t}</div></div>))}
                {currentTranscript && <div className="flex flex-col items-end gap-1 ml-auto max-w-[85%] animate-pulse"><div className="px-4 py-3 rounded-2xl rounded-tr-none bg-zinc-900/50 border border-zinc-800/30 text-zinc-400 text-sm italic">{currentTranscript}...</div></div>}
                <div ref={chatEndRef} />
              </div>
              <form onSubmit={sendManualChat} className="mb-6 relative">
                <input type="text" value={manualInput} onChange={(e) => setManualInput(e.target.value)} placeholder="Send silent context..." className="w-full bg-zinc-900/50 border border-zinc-800 rounded-xl px-4 py-3 text-sm focus:outline-none focus:border-white/20 transition-all pr-12" />
                <button type="submit" className="absolute right-2 top-1/2 -translate-y-1/2 p-2 text-zinc-500 hover:text-white"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg></button>
              </form>
              <footer className="sticky bottom-0 bg-[#070F1A]/80 backdrop-blur-md pt-4 border-t border-[#0F766E]/20 flex items-center justify-between">
                <button onClick={stopMonitoring} className="px-6 py-3 bg-red-950/20 border border-red-500/20 text-red-500 text-sm font-bold rounded-xl hover:bg-red-500 hover:text-white transition-all">Terminate Session</button>
                <div className="flex items-center gap-2">
                  <span className="text-[10px] font-black uppercase text-[#0F766E]/60 tracking-widest">Monitoring Mode:</span>
                  <span className="text-[10px] font-black uppercase text-[#14B8A6] tracking-widest bg-[#14B8A6]/10 border border-[#14B8A6]/20 px-3 py-1 rounded-full uppercase">{monitoringMode}</span>
                </div>
              </footer>
            </div>
          )}
        </main>
      )}

      {showOnboarding && (
        <div className="fixed inset-0 z-[100] bg-[#070F1A]/90 backdrop-blur-md flex items-center justify-center p-4">
          <div className="max-w-md w-full bg-[#0F172A] border border-[#0F766E]/30 rounded-[32px] p-8 shadow-2xl animate-in zoom-in-95 duration-300">
            {onboardingStep === 0 ? (
              <div className="text-center">
                <h2 className="text-2xl font-black mb-2 text-[#E5E7EB]">Choose Your Role</h2>
                <p className="text-[#9CA3AF] text-sm mb-8">How will you be using Black Box today?</p>
                <div className="grid grid-cols-1 gap-4">
                  <button 
                    onClick={() => handleUpdateRole(false)}
                    className="group p-6 bg-[#070F1A] border border-[#0F766E]/20 rounded-2xl text-left hover:border-[#14B8A6]/50 transition-all hover:bg-[#070F1A]/80 shadow-lg"
                  >
                    <h3 className="font-bold text-[#E5E7EB] mb-1">I need Protection</h3>
                    <p className="text-xs text-[#9CA3AF]">I want my guardians to monitor me during high-stakes events.</p>
                  </button>
                  <button 
                    onClick={() => handleUpdateRole(true)}
                    className="group p-6 bg-[#070F1A] border border-[#0F766E]/20 rounded-2xl text-left hover:border-[#14B8A6]/50 transition-all hover:bg-[#070F1A]/80 shadow-lg"
                  >
                    <h3 className="font-bold text-[#E5E7EB] mb-1">I am a Guardian</h3>
                    <p className="text-xs text-[#9CA3AF]">I am here only to watch over others and respond to alerts.</p>
                  </button>
                </div>
              </div>
            ) : onboardingStep === 1 ? (
              <div className="text-center">
                <h2 className="text-2xl font-black mb-4 text-[#E5E7EB]">Voice Enrollment</h2>
                <div className="bg-[#070F1A] p-6 rounded-2xl border border-[#0F766E]/20 italic text-lg mb-8 text-[#9CA3AF] leading-relaxed">
                  &quot;The quick brown fox jumps over the lazy dog. My shadow is my guardian, keeping me safe in the dark.&quot;
                </div>
                <button 
                  onClick={handleEnrollVoice} 
                  disabled={isRecording} 
                  className={`w-full py-4 rounded-full font-bold transition-all shadow-lg ${isRecording ? 'bg-[#FF8559] animate-pulse text-white' : 'bg-[#14B8A6] text-[#0B1120] hover:scale-105 hover:bg-[#22C9B7]'}`}
                >
                  {isRecording ? "RECORDING..." : "START RECORDING"}
                </button>
              </div>
            ) : (
              <div>
                <h2 className="text-2xl font-black mb-6 text-center text-[#E5E7EB]">Guardian Setup</h2>
                <form onSubmit={handleAddGuardian} className="space-y-4">
                  <input type="email" required placeholder="Guardian Email" value={guardianEmail} onChange={(e) => setGuardianEmail(e.target.value)} className="w-full bg-[#070F1A] border border-[#0F766E]/20 rounded-xl px-4 py-3 text-sm focus:border-[#14B8A6] outline-none text-[#E5E7EB] placeholder:text-zinc-700 transition-all" />
                  <input type="tel" placeholder="Guardian Phone" value={guardianPhone} onChange={(e) => setGuardianPhone(e.target.value)} className="w-full bg-[#070F1A] border border-[#0F766E]/20 rounded-xl px-4 py-3 text-sm focus:border-[#14B8A6] outline-none text-[#E5E7EB] placeholder:text-zinc-700 transition-all" />
                  <button 
                    disabled={isSubmittingGuardian} 
                    className="w-full py-4 bg-[#14B8A6] text-[#0B1120] rounded-full font-black hover:bg-[#22C9B7] hover:scale-105 transition-all flex items-center justify-center gap-2 mt-4 shadow-lg shadow-[#14B8A6]/20 disabled:opacity-50"
                  >
                    {isSubmittingGuardian ? (
                      <>
                        <div className="h-4 w-4 border-2 border-[#0B1120]/20 border-t-[#0B1120] rounded-full animate-spin" />
                        PROCESSING...
                      </>
                    ) : (
                      "COMPLETE SETUP"
                    )}
                  </button>
                </form>
              </div>
            )}
          </div>
        </div>
      )}

      {showInitiationModal && (
        <div className="fixed inset-0 z-[100] bg-[#070F1A]/80 backdrop-blur-sm flex items-center justify-center p-4" onClick={() => setShowInitiationModal(false)}>
          <div className="max-w-lg w-full bg-[#0F172A] border border-[#0F766E]/30 rounded-[32px] p-8 shadow-[0_32px_80px_rgba(0,0,0,0.6)] active:scale-[0.99] transition-all" onClick={e => e.stopPropagation()}>
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-2xl font-black flex items-center gap-3">
                <span className="w-1.5 h-6 bg-[#14B8A6] rounded-full" />
                Session Setup
              </h2>
              <button onClick={() => setShowInitiationModal(false)} className="w-10 h-10 rounded-full bg-[#070F1A] flex items-center justify-center text-[#9CA3AF] hover:text-[#E5E7EB] transition-colors border border-[#0F766E]/20">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
              </button>
            </div>

            <div className="space-y-6">
              <div>
                <label className="block text-[10px] font-black uppercase tracking-[0.2em] text-[#0F766E] mb-3">SITUATION CONTEXT</label>
                <textarea
                  placeholder="e.g. Walking home late at night..."
                  value={sessionContext}
                  onChange={(e) => setSessionContext(e.target.value)}
                  className="w-full bg-[#070F1A] border border-[#0F766E]/20 rounded-2xl p-5 text-sm focus:border-[#14B8A6] outline-none h-32 resize-none leading-relaxed transition-all placeholder:text-zinc-800"
                />
              </div>

              <div>
                <label className="block text-[10px] font-black uppercase tracking-[0.2em] text-[#0F766E] mb-3">MONITORING MODE</label>
                <div className="grid grid-cols-3 gap-3">
                  {(['audio', 'text', 'both'] as const).map((mode) => (
                    <button
                      key={mode}
                      onClick={() => setMonitoringMode(mode)}
                      className={`py-4 rounded-2xl text-[10px] font-black uppercase tracking-widest transition-all border ${monitoringMode === mode
                        ? 'bg-[#14B8A6] text-[#0B1120] border-[#14B8A6]'
                        : 'bg-[#070F1A] text-[#9CA3AF] border-[#0F766E]/20 hover:border-[#0F766E]/50'
                        }`}
                    >
                      {mode}
                    </button>
                  ))}
                </div>
              </div>

              <div className="pt-4 space-y-3">
                <button
                  onClick={startMonitoring}
                  disabled={status === 'connecting'}
                  className="w-full py-5 bg-[#14B8A6] text-[#0B1120] font-black rounded-full hover:bg-[#22C9B7] hover:scale-[1.02] active:scale-[0.98] transition-all uppercase tracking-[0.15em] shadow-lg shadow-[#14B8A6]/20"
                >
                  {status === 'connecting' ? 'CONNECTING...' : 'START BLACK-BOX'}
                </button>
                <button
                  onClick={() => setShowInitiationModal(false)}
                  className="w-full py-3 text-[#9CA3AF] font-bold hover:text-[#E5E7EB] transition-all uppercase text-[10px] tracking-widest"
                >
                  CANCEL
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {showAddGuardianModal && (
        <div className="fixed inset-0 z-[100] bg-[#070F1A]/80 backdrop-blur-sm flex items-center justify-center p-4" onClick={() => setShowAddGuardianModal(false)}>
          <div className="max-w-md w-full bg-[#0F172A] border border-[#0F766E]/30 rounded-[32px] p-8 shadow-[0_32px_80px_rgba(0,0,0,0.6)] animate-in fade-in zoom-in duration-300" onClick={e => e.stopPropagation()}>
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-2xl font-black flex items-center gap-3">
                <span className="w-1.5 h-6 bg-[#14B8A6] rounded-full" />
                Add Guardian
              </h2>
              <button onClick={() => setShowAddGuardianModal(false)} className="w-10 h-10 rounded-full bg-[#070F1A] flex items-center justify-center text-[#9CA3AF] hover:text-[#E5E7EB] transition-colors border border-[#0F766E]/20">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
              </button>
            </div>
            <p className="text-[#9CA3AF] text-sm mb-8 leading-relaxed">They will receive an alert if your risk levels spike during a session.</p>
            <form onSubmit={handleAddGuardian} className="space-y-4">
              <input type="email" required placeholder="Guardian Email" value={guardianEmail} onChange={(e) => setGuardianEmail(e.target.value)} className="w-full bg-[#070F1A] border border-[#0F766E]/20 rounded-xl px-4 py-3 text-sm focus:border-[#14B8A6] outline-none text-[#E5E7EB] transition-all" />
              <input type="tel" placeholder="Guardian Phone" value={guardianPhone} onChange={(e) => setGuardianPhone(e.target.value)} className="w-full bg-[#070F1A] border border-[#0F766E]/20 rounded-xl px-4 py-3 text-sm focus:border-[#14B8A6] outline-none text-[#E5E7EB] transition-all" />
              <button 
                disabled={isSubmittingGuardian} 
                className="w-full py-4 bg-[#14B8A6] text-[#0B1120] rounded-full font-black hover:bg-[#22C9B7] hover:scale-105 transition-all flex items-center justify-center gap-2 shadow-lg shadow-[#14B8A6]/20 disabled:opacity-50"
              >
                {isSubmittingGuardian ? (
                  <>
                    <div className="h-4 w-4 border-2 border-[#0B1120]/20 border-t-[#0B1120] rounded-full animate-spin" />
                    ADDING...
                  </>
                ) : (
                  "ADD GUARDIAN"
                )}
              </button>
            </form>
          </div>
        </div>
      )}


      {/* History View */}
      {!isLoading && activeTab === "history" && (
        <main className="flex-1 flex flex-col pt-24 px-4 max-w-4xl mx-auto w-full relative z-10">
          <div className="flex justify-between items-center mb-8">
            <h2 className="text-2xl font-black flex items-center gap-3">
              <span className="w-1.5 h-6 bg-[#14B8A6] rounded-full" />
              Session Vault
            </h2>
          </div>
          <div className="space-y-4">
            {history.length === 0 ? <div className="p-12 border-2 border-dashed border-[#0F766E]/20 rounded-[32px] text-center text-[#9CA3AF] italic bg-[#0F172A]/30">No historical traces found.</div> :
              history.map((h, i) => (
                <div key={i} className="bg-[#0F172A]/50 border border-[#0F766E]/20 rounded-[24px] p-6 transition-all hover:bg-[#0F172A] hover:border-[#14B8A6]/30 shadow-lg group">
                  <div className="flex justify-between items-start mb-4">
                    <code className="text-[10px] text-[#14B8A6] font-mono bg-[#14B8A6]/10 px-2 py-0.5 rounded-md">{h.id}</code>
                    <span className="text-[10px] font-bold text-[#9CA3AF] uppercase tracking-widest">{new Date(h.created_at).toLocaleString()}</span>
                  </div>
                  {h.initial_context && (
                    <div className="mb-4 p-4 bg-[#070F1A] border border-[#0F766E]/10 rounded-2xl text-xs text-[#9CA3AF] italic leading-relaxed">
                      &quot;{h.initial_context}&quot;
                    </div>
                  )}
                  <div className="space-y-3">
                    {h.logs?.slice(0, 3).map((l: any, i: number) => (
                      <div key={i} className="text-sm text-[#E5E7EB] line-clamp-1 border-l-2 border-[#14B8A6]/20 pl-4 py-1">
                        <span className="text-[10px] font-black mr-3 text-[#14B8A6] uppercase tracking-wider">{l.speaker_label || 'USER'}:</span>
                        {l.content}
                      </div>
                    ))}
                  </div>
                </div>
              ))}
          </div>
        </main>
      )}

      {/* Guardians View */}
      {!isLoading && activeTab === "guardians" && (
        <main className="flex-1 flex flex-col pt-24 px-4 max-w-4xl mx-auto w-full pb-20 relative z-10">
          <div className={`grid grid-cols-1 ${profile?.account_role !== "guardian" ? 'lg:grid-cols-2' : ''} gap-12`}>
            {/* My Protectors */}
            {profile?.account_role !== "guardian" && (
              <div>
                <div className="flex justify-between items-center mb-6">
                  <h2 className="text-2xl font-black flex items-center gap-3">
                    <span className="w-1.5 h-6 bg-[#14B8A6] rounded-full" />
                    My Protectors
                  </h2>
                  <button onClick={() => setShowAddGuardianModal(true)} className="w-10 h-10 rounded-full bg-[#14B8A6] text-[#0B1120] flex items-center justify-center hover:bg-[#22C9B7] hover:scale-105 transition-all shadow-lg shadow-[#14B8A6]/20">
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" /></svg>
                  </button>
                </div>
                <p className="text-[#9CA3AF] text-sm mb-8 leading-relaxed">These are the people notified during your high-risk sessions.</p>
                <div className="space-y-3">
                  {myGuardians.length === 0 ? <div className="p-12 border border-dashed border-[#0F766E]/20 rounded-3xl text-center text-[#9CA3AF] text-sm bg-[#0F172A]/30">Nobody is protecting you yet.</div> :
                    myGuardians.map((g, i) => (
                      <div key={i} className="bg-[#0F172A]/50 border border-[#0F766E]/20 rounded-2xl p-4 flex items-center justify-between group hover:border-[#14B8A6]/30 transition-all shadow-md">
                        <div className="flex items-center gap-4">
                          <div className="h-12 w-12 rounded-2xl bg-[#070F1A] border border-[#0F766E]/20 flex items-center justify-center font-black text-[#14B8A6] uppercase">{g.guardian_email?.charAt(0)}</div>
                          <div>
                            <h4 className="text-sm font-bold text-[#E5E7EB]">{g.guardian_email}</h4>
                            <span className={`text-[10px] font-black uppercase tracking-[0.2em] ${g.status === 'active' ? 'text-[#14B8A6]' : 'text-[#9CA3AF]'}`}>{g.status}</span>
                          </div>
                        </div>
                        <button
                          onClick={() => handleRemoveGuardian(g.id)}
                          className="p-3 text-[#9CA3AF] hover:text-[#FF8559] hover:bg-[#FF8559]/10 rounded-xl transition-all opacity-0 group-hover:opacity-100"
                        >
                          <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 6h18m-2 0v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6m3 0V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2m-6 9 2 2 4-4" /></svg>
                        </button>
                      </div>
                    ))}
                </div>
              </div>
            )}

            {/* Watching Over */}
            <div>
              <h2 className="text-2xl font-black mb-6 flex items-center gap-3">
                <span className="w-1.5 h-6 bg-[#FF8559] rounded-full" />
                Watching Over
              </h2>
              <p className="text-[#9CA3AF] text-sm mb-8 leading-relaxed">Users who have added you as their safety contact.</p>
              <div className="grid grid-cols-1 gap-4">
                {guarding.length === 0 ? <div className="p-12 border-2 border-dashed border-[#0F766E]/20 rounded-[32px] text-center text-[#9CA3AF] italic bg-[#0F172A]/30">You aren&apos;t guarding anyone yet.</div> :
                  guarding.map((rel, i) => {
                    const p = rel.profiles;
                    if (!p) return null;
                    return (
                      <div key={rel.id} className="bg-[#0F172A]/50 border border-[#0F766E]/20 rounded-[24px] p-6 flex items-center justify-between group transition-all hover:border-[#14B8A6]/30 shadow-lg">
                        <div className="flex items-center gap-5">
                          <div className="h-14 w-14 rounded-2xl bg-[#070F1A] border border-[#0F766E]/20 flex items-center justify-center font-black text-[#9CA3AF] group-hover:text-[#14B8A6] uppercase transition-colors">{p.full_name?.charAt(0) || p.email.charAt(0)}</div>
                          <div>
                            <h4 className="font-bold text-[#E5E7EB] text-lg">{p.full_name || "Anonymous User"}</h4>
                            <p className="text-[10px] text-[#9CA3AF] font-mono tracking-wider opacity-60">{p.email}</p>
                            <span className={`text-[10px] font-black uppercase tracking-[0.2em] mt-2 block ${rel.status === 'active' ? 'text-[#14B8A6]' : 'text-[#FF8559]'}`}>{rel.status}</span>
                          </div>
                        </div>
                        <div className="text-right flex items-center gap-4">
                          {rel.status === 'pending' ? (
                            <button onClick={() => handleAcceptGuardian(rel.id)} className="px-5 py-2.5 bg-[#14B8A6] text-[#0B1120] text-[10px] font-black rounded-full hover:bg-[#22C9B7] hover:scale-105 transition-all uppercase tracking-widest shadow-lg">Accept Request</button>
                          ) : (
                            <button className="text-[10px] font-black text-[#14B8A6] hover:text-[#22C9B7] uppercase transition-all tracking-widest border border-[#14B8A6]/20 bg-[#14B8A6]/5 px-4 py-2 rounded-full">Live Status</button>
                          )}
                          <button
                            onClick={() => handleRemoveGuardian(rel.id)}
                            className="p-3 text-[#9CA3AF] hover:text-[#FF8559] hover:bg-[#FF8559]/10 rounded-xl transition-all opacity-0 group-hover:opacity-100"
                          >
                            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 6h18m-2 0v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6m3 0V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2m-6 9 2 2 4-4" /></svg>
                          </button>
                        </div>
                      </div>
                    );
                  })}
              </div>
            </div>
          </div>
        </main>
      )}

      {/* Notifications View */}
      {!isLoading && activeTab === "notifications" && (
        <main className="flex-1 flex flex-col pt-24 px-4 max-w-2xl mx-auto w-full relative z-10">
          <div className="flex justify-between items-center mb-8">
            <h2 className="text-2xl font-black flex items-center gap-3">
              <span className="w-1.5 h-6 bg-[#FF8559] rounded-full" />
              Alerts & Notifications
            </h2>
          </div>
          <div className="space-y-4">
            {notifications.length === 0 ? (
              <div className="p-12 border-2 border-dashed border-[#0F766E]/20 rounded-[32px] text-center text-[#9CA3AF] italic bg-[#0F172A]/30">
                No active signals found.
              </div>
            ) : (
              notifications.map((n, i) => (
                <div key={i} onClick={() => !n.is_read && markNotificationRead(n.id)} className={`bg-[#0F172A]/50 border ${n.is_read ? 'border-[#0F766E]/20' : 'border-[#14B8A6]/40'} rounded-2xl p-6 transition-all hover:bg-[#0F172A] cursor-pointer relative shadow-lg group`}>
                  {!n.is_read && <div className="absolute top-6 right-6 h-2 w-2 rounded-full bg-[#14B8A6] shadow-[0_0_8px_rgba(20,184,166,0.5)]" />}
                  <div className="mb-3 flex items-center gap-3">
                    <span className={`text-[10px] font-black uppercase tracking-widest px-2 py-0.5 rounded ${n.type === 'risk_alert' ? 'bg-[#FF8559]/20 text-[#FF8559]' : 'bg-[#14B8A6]/20 text-[#14B8A6]'}`}>{n.type.replace('_', ' ')}</span>
                    <span className="text-[10px] text-[#9CA3AF] font-mono opacity-60">{new Date(n.created_at).toLocaleString()}</span>
                  </div>
                  <h4 className={`font-bold text-lg ${n.is_read ? 'text-[#9CA3AF]' : 'text-[#E5E7EB]'}`}>{n.title}</h4>
                  <p className="text-sm text-[#9CA3AF] mt-2 leading-relaxed">{n.message}</p>
                </div>
              ))
            )}
          </div>
        </main>
      )}

      {/* Profile View */}
      {!isLoading && activeTab === "profile" && (
        <main className="flex-1 flex flex-col pt-24 px-4 max-w-4xl mx-auto w-full relative z-10">
          <h2 className="text-2xl font-black mb-8 flex items-center gap-3">
            <span className="w-1.5 h-6 bg-[#14B8A6] rounded-full" />
            Account Command
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="bg-[#0F172A]/50 border border-[#0F766E]/20 rounded-[32px] p-8 shadow-xl">
              <label className="block text-[10px] font-black uppercase tracking-[0.2em] text-[#0F766E] mb-6">PROFILE INFORMATION</label>
              <div className="space-y-6">
                <div>
                  <p className="text-[10px] text-[#9CA3AF] uppercase font-black mb-1">Full Name</p>
                  <p className="text-[#E5E7EB] font-bold text-lg">{profile?.full_name || "Tracer User"}</p>
                </div>
                <div>
                  <p className="text-[10px] text-[#9CA3AF] uppercase font-black mb-1">Identity</p>
                  <p className="text-[#E5E7EB] font-bold text-lg">{profile?.email || user?.email}</p>
                </div>
              </div>
            </div>

            <div className="bg-[#0F172A]/50 border border-[#0F766E]/20 rounded-[32px] p-8 shadow-xl">
              <label className="block text-[10px] font-black uppercase tracking-[0.2em] text-[#0F766E] mb-6">PREFERENCES</label>
              <div className="space-y-8">
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="font-bold text-[#E5E7EB]">Guardian mode</h4>
                    <p className="text-xs text-[#9CA3AF]">Focus solely on helping others.</p>
                  </div>
                  <button
                    onClick={() => handleUpdateRole(profile?.account_role !== "guardian")}
                    className={`w-14 h-8 rounded-full transition-all relative ${profile?.account_role === 'guardian' ? 'bg-[#14B8A6]' : 'bg-[#070F1A] border border-[#0F766E]/30'}`}
                  >
                    <div className={`absolute top-1 w-6 h-6 rounded-full bg-white transition-all shadow-md ${profile?.account_role === 'guardian' ? 'left-7' : 'left-1'}`} />
                  </button>
                </div>

                <div className="pt-4 border-t border-[#0F766E]/10">
                  <button onClick={handleSignOut} className="w-full py-4 bg-[#FF8559]/5 border border-[#FF8559]/20 text-[#FF8559] rounded-2xl font-black text-[10px] uppercase tracking-widest hover:bg-[#FF8559] hover:text-white transition-all">
                    Sign Out
                  </button>
                </div>
              </div>
            </div>
          </div>
        </main>
      )}

      <style jsx global>{`
        .custom-scrollbar::-webkit-scrollbar { width: 4px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: #27272a; border-radius: 2px; }
      `}</style>
    </div>
  );
}
