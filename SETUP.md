# Setup Guide

Complete local setup instructions for the AI Business Decision Simulation System.

---

## Prerequisites

- Python 3.11+ (3.12 or 3.14 work)
- Node.js 18+
- Git
- A free [Supabase](https://supabase.com) account
- A free [Groq](https://console.groq.com) API key
- (Optional) Docker and Docker Compose

---

## Step 1: Supabase Project Setup

1. Go to [supabase.com](https://supabase.com) and create a free account
2. Click **New Project**, pick a name and set a database password
3. Wait for the project to finish provisioning (~30 seconds)

### Run the Database Schema

4. Go to **SQL Editor** in the left sidebar
5. Click **New Query**
6. Open `supabase/schema.sql` from this repo, copy the entire contents, paste into the editor
7. Click **Run** — this creates all tables, indexes, vector extensions, and RLS policies

### Get Your Keys

8. Go to **Project Settings** > **API** (left sidebar)
9. Copy these values:
   - **Project URL** (e.g., `https://xyz.supabase.co`)
   - **anon public** key (starts with `eyJ...`)
   - **service_role** key (starts with `eyJ...`, keep this secret)
10. Go to **Project Settings** > **API** > **JWT Settings**
    - Copy the **JWT Secret**

---

## Step 2: Backend Setup

```bash
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt

# Note: sentence-transformers will download ~90MB model on first use
```

### Configure Environment

```bash
cp .env.example .env
```

Edit `backend/.env` with your actual keys:

```env
GROQ_API_KEY=gsk_your_key_here
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
SUPABASE_SERVICE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
SUPABASE_JWT_SECRET=your-jwt-secret-from-supabase-dashboard
```

### Start the Backend

```bash
uvicorn app.main:app --reload --port 8000
```

Verify: open http://localhost:8000/docs to see all API routes.

---

## Step 3: Frontend Setup

```bash
cd frontend

# Install dependencies
npm install
```

### Configure Environment

```bash
cp .env.example .env
```

Edit `frontend/.env`:

```env
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### Start the Frontend

```bash
npm run dev
```

Open http://localhost:5173

---

## Step 4: First Run

1. Open http://localhost:5173 — you'll be redirected to the **Login** page
2. Click **Sign Up**, enter an email and password (Supabase sends a confirmation email)
3. After confirming your email, sign in
4. You'll land on the **Chat** page
5. Type `train all models` to train the ML models on sample data (~10 seconds)
6. Try asking: "What happens if I raise prices by 25%?"
7. Go to **Knowledge Center** (sidebar) to upload PDF/DOCX/TXT documents for RAG

---

## Step 5: Docker Setup (Alternative)

If you prefer Docker over manual setup:

```bash
# From the root directory
cp .env.example .env
# Edit .env with your keys (same as above)

docker-compose up --build
```

- Frontend: http://localhost:5173
- Backend: http://localhost:8000
- Swagger Docs: http://localhost:8000/docs

---

## Environment Variables Reference

### Backend (`backend/.env`)

| Variable | Required | Where to Get It |
|----------|----------|-----------------|
| `GROQ_API_KEY` | Yes (for chat/LLM) | [console.groq.com](https://console.groq.com) > API Keys |
| `SUPABASE_URL` | Yes | Supabase Dashboard > Settings > API > Project URL |
| `SUPABASE_ANON_KEY` | Yes | Supabase Dashboard > Settings > API > anon public |
| `SUPABASE_SERVICE_KEY` | Yes | Supabase Dashboard > Settings > API > service_role |
| `SUPABASE_JWT_SECRET` | Yes | Supabase Dashboard > Settings > API > JWT Settings > JWT Secret |

### Frontend (`frontend/.env`)

| Variable | Required | Where to Get It |
|----------|----------|-----------------|
| `VITE_SUPABASE_URL` | Yes | Same as backend's SUPABASE_URL |
| `VITE_SUPABASE_ANON_KEY` | Yes | Same as backend's SUPABASE_ANON_KEY |

---

## Troubleshooting

### "Module not found: sentence_transformers"
```bash
pip install sentence-transformers
```
First embedding call downloads the model (~90MB). Requires internet.

### "libomp not found" (macOS, LightGBM error)
```bash
brew install libomp
```

### Port 8000 already in use
```bash
lsof -ti:8000 | xargs kill -9
```

### "Invalid token" on API calls
- Check that `SUPABASE_JWT_SECRET` in backend `.env` matches the one in Supabase dashboard
- Make sure the frontend `VITE_SUPABASE_URL` matches backend `SUPABASE_URL`

### CORS errors in browser
- Backend `cors_origins` in `config.py` must include your frontend URL (`http://localhost:5173`)

### "No insight generated. Train models first."
- Type `train all models` in the chat, or call `POST /api/train/all`
- This trains on sample data included in `backend/app/data/`

---

## Project Structure

```
betchProject/
??? README.md              # Architecture and design decisions
??? SETUP.md               # This file (setup instructions)
??? docker-compose.yml     # One-command deployment
??? supabase/
?   ??? schema.sql         # Database schema (run in Supabase SQL Editor)
??? backend/
?   ??? requirements.txt   # Python dependencies
?   ??? app/
?   ?   ??? main.py        # FastAPI entry point (26 routes)
?   ?   ??? config.py      # Settings (env vars)
?   ?   ??? auth/          # JWT middleware
?   ?   ??? db/            # Supabase client + CRUD layers
?   ?   ??? rag/           # RAG pipeline (embeddings, parser, retriever)
?   ?   ??? models/        # ML models (LightGBM, DistilBERT)
?   ?   ??? engine/        # Causal simulation engine
?   ?   ??? llm/           # LLM orchestration + prompts
?   ?   ??? routes/        # API endpoints
?   ??? tests/             # 46 API tests
??? frontend/
    ??? package.json
    ??? src/
        ??? App.tsx         # Router (Login, Chat, Knowledge)
        ??? contexts/       # Auth state
        ??? pages/          # Login, Chat, KnowledgeCenter
        ??? components/     # ProtectedRoute, etc.
        ??? api/            # Typed API client + Supabase
```
