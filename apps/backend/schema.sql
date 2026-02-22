-- 1. Create Profiles Table with Voice Biometrics
CREATE TABLE IF NOT EXISTS public.profiles (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    email TEXT UNIQUE NOT NULL,
    full_name TEXT,
    voice_fingerprint VECTOR(384), -- Requires pgvector extension enabled in Supabase
    is_enrolled BOOLEAN DEFAULT false,
    account_role TEXT DEFAULT 'both', -- 'guardian_only', 'both'
    created_at TIMESTAMPTZ DEFAULT now()
);

-- 2. Create Guardians Table (The Relationship Bridge)
CREATE TABLE IF NOT EXISTS public.guardians (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES public.profiles(id) ON DELETE CASCADE,
    guardian_id UUID REFERENCES public.profiles(id) ON DELETE SET NULL,
    guardian_email TEXT NOT NULL,
    guardian_phone TEXT,
    status TEXT DEFAULT 'pending', -- 'pending', 'active'
    created_at TIMESTAMPTZ DEFAULT now()
);

-- 3. Create Threads Table
CREATE TABLE IF NOT EXISTS public.threads (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES public.profiles(id) ON DELETE CASCADE,
    initial_context TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- 4. Create Logs Table
CREATE TABLE IF NOT EXISTS public.logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    thread_id UUID REFERENCES public.threads(id) ON DELETE CASCADE,
    content TEXT,
    speaker_label TEXT,
    is_primary_user BOOLEAN DEFAULT true,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- 5. Analyses table (one row per turn: safety + criminal analysis from RAG + Groq)
CREATE TABLE IF NOT EXISTS public.analyses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    thread_id UUID REFERENCES public.threads(id) ON DELETE CASCADE,
    log_id UUID REFERENCES public.logs(id) ON DELETE SET NULL,
    user_id UUID REFERENCES public.profiles(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT now(),
    threat_category TEXT,
    severity TEXT,
    label TEXT,
    score DOUBLE PRECISION,
    reason TEXT,
    narrative_analysis TEXT,
    relevant_law TEXT
);

-- 6. Create Notifications Table
CREATE TABLE IF NOT EXISTS public.notifications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    type TEXT NOT NULL, -- 'system', 'risk_alert', 'guardian_added'
    title TEXT NOT NULL,
    message TEXT,
    is_read BOOLEAN DEFAULT false,
    link TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Enable RLS
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.guardians ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.threads ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.analyses ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.notifications ENABLE ROW LEVEL SECURITY;

-- Policies for Notifications
CREATE POLICY "Users can view their own notifications" 
ON public.notifications FOR SELECT 
USING (auth.uid() = user_id);

CREATE POLICY "Users can update their own notifications" 
ON public.notifications FOR UPDATE 
USING (auth.uid() = user_id);

-- RLS Policies
CREATE POLICY "Users can view their own profile" ON public.profiles FOR SELECT USING (auth.uid() = id);
CREATE POLICY "Users can update their own profile" ON public.profiles FOR UPDATE USING (auth.uid() = id);
CREATE POLICY "Users can manage their own guardians" ON public.guardians FOR ALL USING (auth.uid() = user_id);
CREATE POLICY "Users can manage their own threads" ON public.threads FOR ALL USING (auth.uid() = user_id);
CREATE POLICY "Users can view logs for their own threads" ON public.logs FOR SELECT USING (
  EXISTS (
    SELECT 1 FROM public.threads 
    WHERE public.threads.id = public.logs.thread_id 
    AND public.threads.user_id = auth.uid()
  )
);

CREATE POLICY "Users can view their own analyses" ON public.analyses FOR SELECT USING (auth.uid() = user_id);

-- 4. Automate Profile Creation on Signup
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO public.profiles (id, email, full_name)
  VALUES (
    NEW.id, 
    NEW.email, 
    COALESCE(NEW.raw_user_meta_data->>'full_name', NEW.raw_user_meta_data->>'name', '')
  );
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Trigger to call the function after a new user is created in auth.users
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();
