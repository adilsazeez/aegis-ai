# RAG pipeline setup

## Task 1: Actian VectorAI DB (you do this)

### 1.1 Install the Python client

From the backend directory:

```bash
cd apps/backend
pip install "actiancortex @ https://github.com/hackmamba-io/actian-vectorAI-db-beta/raw/main/actiancortex-0.1.0b1-py3-none-any.whl"
```

If that fails (e.g. corporate firewall), clone the repo, then install the wheel locally:

```bash
git clone https://github.com/hackmamba-io/actian-vectorAI-db-beta.git
pip install ./actian-vectorAI-db-beta/actiancortex-0.1.0b1-py3-none-any.whl
```

### 1.2 Start the database

You already have a container `vectoraidb`. Start it so the app can reach Actian at `localhost:50051`:

```bash
docker start vectoraidb
```

If you don’t have the image yet, from the [Actian beta repo](https://github.com/hackmamba-io/actian-vectorAI-db-beta) you can run:

```bash
docker compose up -d
```

Ensure `.env` has:

```env
ACTIAN_HOST=localhost:50051
```

---

## Supabase: who creates the `analyses` table?

- **You** (backend owner): Create the table **once** in **your** Supabase project (the one in your backend `.env`). The FastAPI app writes to `analyses`, so the table must exist in that project.
- **Your friend** (visuals/dashboard): Does **not** create tables. They only **read** from the same Supabase (e.g. `analyses` and `logs`) for their UI. They use the same project’s URL and anon/key with RLS so users see only their own data.

### Create the table (you, in Supabase)

In the Supabase Dashboard → SQL Editor, run the following (or run your full `schema.sql` if you prefer):

```sql
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

ALTER TABLE public.analyses ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view their own analyses"
ON public.analyses FOR SELECT USING (auth.uid() = user_id);
```

---

## After Task 1 and Supabase

1. Ingest Prosocial into Actian (one-time):

   ```bash
   python scripts/ingest_prosocial_to_actian.py
   ```

2. Start the backend and use the app; the WebSocket will run RAG + Groq and write rows to `analyses`.
