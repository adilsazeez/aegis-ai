-- Create the alerts table for threat detection results.
-- Run this in Supabase Dashboard > SQL Editor.

CREATE TABLE IF NOT EXISTS public.alerts (
    id          UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id     UUID NOT NULL,
    thread_id   UUID,
    guardian_id UUID,
    severity    TEXT NOT NULL CHECK (severity IN ('moderate', 'severe')),
    offense_title    TEXT NOT NULL,
    section_reference TEXT NOT NULL,
    category    TEXT NOT NULL,
    confidence_score FLOAT NOT NULL CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
    transcript_snippet TEXT,
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- Index for guardian dashboard queries (most frequent access pattern)
CREATE INDEX IF NOT EXISTS idx_alerts_guardian_id ON public.alerts(guardian_id);

-- Index for per-user alert history (analytics + escalation prediction)
CREATE INDEX IF NOT EXISTS idx_alerts_user_id ON public.alerts(user_id);

-- Index for time-range queries (timeline charts)
CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON public.alerts(created_at DESC);

-- Enable Row Level Security
ALTER TABLE public.alerts ENABLE ROW LEVEL SECURITY;

-- Policy: service role can do everything (our FastAPI backend uses service role key)
CREATE POLICY "Service role full access" ON public.alerts
    FOR ALL
    USING (true)
    WITH CHECK (true);

-- Enable Realtime so the guardian frontend gets instant push notifications
ALTER PUBLICATION supabase_realtime ADD TABLE public.alerts;
