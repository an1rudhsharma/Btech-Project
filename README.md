# AI Business Decision Simulation System

An AI-driven business simulator with user authentication, persistent chat, RAG-powered knowledge retrieval, and causally-connected ML models. Users upload documents, train models, and ask natural language questions — the system responds with data-driven insights using real simulation results.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         FRONTEND (React + TypeScript)                     │
│                                                                          │
│  ┌──────────┐   ┌──────────────┐   ┌─────────────────────────┐         │
│  │  Login   │   │  Chat (SSE   │   │   Knowledge Center      │         │
│  │  Page    │   │  Streaming)  │   │   (Upload PDFs, etc.)   │         │
│  └────┬─────┘   └──────┬───────┘   └───────────┬─────────────┘         │
│       │                 │                       │                        │
│       └─────────────────┼───────────────────────┘                        │
│                         │  Axios + JWT Auth Header                        │
└─────────────────────────┼────────────────────────────────────────────────┘
                          │
┌─────────────────────────┼────────────────────────────────────────────────┐
│                    BACKEND (FastAPI + Python)                             │
│                         │                                                │
│  ┌──────────────────────▼────────────────────────────────────┐          │
│  │              Auth Middleware (JWT Verification)             │          │
│  └──────────────────────┬────────────────────────────────────┘          │
│                         │                                                │
│  ┌──────────┐   ┌──────▼───────┐   ┌─────────────────────┐            │
│  │ Sessions │   │  Streaming   │   │   Knowledge Upload   │            │
│  │  CRUD    │   │  Chat + RAG  │   │   (Parse + Embed)    │            │
│  └──────────┘   └──────┬───────┘   └─────────────────────┘            │
│                         │                                                │
│  ┌──────────────────────▼────────────────────────────────────┐          │
│  │              RAG Retriever (pgvector similarity search)     │          │
│  │              + Text-to-Pandas (for CSV queries)             │          │
│  └──────────────────────┬────────────────────────────────────┘          │
│                         │                                                │
│  ┌──────────────────────▼────────────────────────────────────┐          │
│  │              Causal Simulation Engine                       │          │
│  │  Pricing → Marketing → Sentiment → Churn (DAG order)      │          │
│  │  + SHAP explainability + DiCE counterfactuals              │          │
│  └──────────────────────┬────────────────────────────────────┘          │
│                         │                                                │
│  ┌──────────────────────▼────────────────────────────────────┐          │
│  │              Groq LLM (Llama 3.3 70B, SSE streaming)       │          │
│  │              Chain-of-Thought insight generation            │          │
│  └───────────────────────────────────────────────────────────┘          │
└──────────────────────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────┼────────────────────────────────────────────────┐
│                    SUPABASE (Cloud Database)                              │
│                         │                                                │
│  ┌──────────┐   ┌──────▼───────┐   ┌─────────────────────┐            │
│  │  Auth    │   │  Postgres    │   │   pgvector           │            │
│  │  (JWT)   │   │  (Sessions,  │   │   (Document chunks   │            │
│  │          │   │   Messages)  │   │    + embeddings)     │            │
│  └──────────┘   └──────────────┘   └─────────────────────┘            │
│                                                                          │
│              Row Level Security: User A cannot see User B's data          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## What This System Does

1. **User Authentication** — Secure login/signup via Supabase Auth. Each user has isolated data.
2. **Chat Persistence** — Conversations survive browser refreshes, stored per-user in Postgres.
3. **Knowledge Base (RAG)** — Upload PDFs, DOCX, TXT files. The AI references them in every answer.
4. **ML Simulation** — 4 causally-connected models predict business outcomes:
   - Churn (customer attrition probability)
   - Sentiment (customer perception from text)
   - Marketing (conversion rate from ad spend)
   - Pricing (demand and elasticity from price changes)
5. **Streaming Responses** — Tokens appear word-by-word via SSE, like ChatGPT.
6. **Explainability** — SHAP tells you WHY; DiCE-ML tells you WHAT TO CHANGE.

---

## Design Decisions and WHY

### Authentication: Supabase (not Firebase, not custom JWT)

| Factor | Supabase | Firebase | Custom JWT |
|--------|----------|----------|------------|
| Cost | Free (50K users) | Free (limited) | Free (self-hosted) |
| Database included | Postgres (relational + vector) | Firestore (NoSQL only) | Must build separately |
| Vector search | pgvector built-in | Not available | Must add Pinecone/Chroma |
| Row Level Security | Native Postgres RLS | Firestore rules (different paradigm) | Application-level only |
| Complexity | One service for everything | Need separate DB + vector store | Must implement token refresh, hashing, etc. |

**Decision:** Supabase gives auth + Postgres + pgvector under one free service. Firebase would require 3 separate services. Custom JWT is too much boilerplate for a project demo.

---

### Vector Database: pgvector in Supabase (not Pinecone, not ChromaDB)

| Factor | pgvector | Pinecone | ChromaDB |
|--------|----------|----------|----------|
| Cost | Free (inside Supabase) | Free tier (1 index) | Free (local) |
| Setup | Zero (just `CREATE EXTENSION`) | Need API key + separate service | Separate process |
| Auth integration | Same DB as users/sessions | None | None |
| Scale needed | Thousands of chunks | Millions+ | Local only |
| Multi-user isolation | RLS policies | Application-level filtering | No multi-user support |

**Decision:** At project scale (thousands of document chunks, not millions), pgvector performs identically to Pinecone. Being inside the same Postgres as sessions/messages means zero extra infrastructure and RLS automatically scopes searches to the current user.

---

### Embedding Model: sentence-transformers/all-MiniLM-L6-v2 (not OpenAI, not HuggingFace API)

| Factor | all-MiniLM-L6-v2 | OpenAI text-embedding-3-small | HuggingFace API |
|--------|-------------------|-------------------------------|-----------------|
| Cost | Free (runs locally) | $0.02 per 1M tokens | Free tier (rate limited) |
| Latency | ~1ms per sentence (cached) | 200-500ms (network round trip) | 1-5 seconds (queue) |
| Privacy | Data never leaves server | Sent to OpenAI servers | Sent to HuggingFace |
| Dimensions | 384 | 1536 | Varies |
| Quality | Good (not best) | Excellent | Varies |

**Decision:** At project scale, the quality difference between 384-dim MiniLM and 1536-dim OpenAI embeddings is negligible. MiniLM runs locally with zero cost, zero latency, and zero privacy concerns. The model loads once on startup (~90MB) and stays in memory.

**Tradeoff:** English-only. Slightly worse at nuanced semantic similarity than OpenAI. Acceptable for business analytics with clear, structured queries.

---

### Structured Data: Text-to-Pandas (not RAG for CSVs)

RAG (embedding + similarity search) works beautifully for unstructured text like PDFs and reports. But for CSV/Excel data, it fails at quantitative queries:

- "What was total revenue in March?" requires summing ALL March rows — vector search only returns top-5 similar chunks
- Embedding "Row 1: price=100, demand=500" loses the mathematical relationships
- LLMs cannot reliably do arithmetic on text-embedded numbers

**Decision:** Dual pipeline:
- PDFs/DOCX/TXT → RAG (chunk, embed, similarity search)
- CSV/Excel → Text-to-Pandas (LLM generates pandas code, execute in sandbox)

The LLM receives the column schema + sample rows, generates a pandas expression (e.g., `df[df['month']=='March']['revenue'].sum()`), which executes in a restricted sandbox (no file I/O, no network, pandas/numpy only).

---

### Chat Streaming: SSE (not WebSockets, not polling)

| Factor | SSE | WebSockets | Long Polling |
|--------|-----|-----------|--------------|
| Direction | Server → Client (unidirectional) | Bidirectional | Client → Server |
| Complexity | Low (standard HTTP) | Medium (upgrade handshake) | High (repeated requests) |
| Reconnection | Automatic | Manual | N/A |
| Use case match | Token streaming (server pushes) | Real-time collaboration | Legacy compatibility |

**Decision:** Token streaming is purely server-to-client (the LLM generates, we push tokens). No bidirectional communication needed. SSE is simpler than WebSockets, auto-reconnects on network failures, and works with standard HTTP infrastructure (no special proxy config).

---

### Message Persistence: Two-Phase Save (not atomic)

**Problem:** If we save both user message + assistant response after generation completes, and the user refreshes during the 2-10 second generation time, their question is lost.

**Solution:**
1. Save user message immediately (Phase 1) — survives any refresh
2. Stream the response via SSE
3. Save assistant message after generation completes (Phase 2)

If the user refreshes mid-generation, they see their question + a "generating..." state. When they refresh again after completion, the full response appears.

---

### ML Model Choices

| Model | Algorithm | Why This One |
|-------|-----------|--------------|
| Churn | LightGBM Classifier | 5-10x faster than XGBoost. Native SHAP support. Handles missing values. |
| Sentiment | DistilBERT | 97% of BERT accuracy at 60% size. No GPU needed. Pre-trained. |
| Marketing | LightGBM Regressor | Same benefits as churn. Continuous output (conversion rate). |
| Pricing | LightGBM Regressor + elasticity formula | Combines ML prediction with economic theory (price elasticity = %Δdemand / %Δprice). |

**Why not Neural Networks for tabular data?** LightGBM consistently outperforms deep learning on structured/tabular data (< 10K features). Neural nets excel at images, text, and sequential data — not business spreadsheets.

**Why Causal DAG instead of parallel models?** A price increase doesn't just affect demand — it cascades: price↑ → sentiment↓ → churn↑. Running models independently misses compound effects. The DAG ensures each model receives upstream signals.

---

### ML-to-RAG Bridge (not separate silos)

**Problem:** If a user trains a Churn model and then asks "What drives churn?", the RAG system has no knowledge of the model's results — they live in separate silos.

**Solution:** After every model training, the system automatically:
1. Generates a text summary (accuracy, top features, SHAP values)
2. Embeds it into pgvector

Now when the user asks "What drives churn?", the RAG retriever finds the model's own analysis and the LLM cites specific feature importances.

---

### Security: Dual Supabase Client Strategy

**Problem:** If the backend uses Supabase's `service_key` (which bypasses Row Level Security), a bug could leak User A's data to User B.

**Solution:** Two clients:
- **Per-request client** (initialized with user's JWT): Used for all READS. RLS is active — even a missing WHERE clause can't leak data.
- **Admin client** (service_key): Used ONLY for internal writes (saving assistant messages). Always includes the verified `user_id` from the JWT — never from user input.

This gives defense-in-depth: RLS protects reads at the database level, explicit user_id protects writes at the application level.

---

## Causal Propagation Flow

```
Price Change ──→ Pricing Model ──→ demand, sentiment_impact
                                          │
Marketing Change ──→ Marketing Model ──→ engagement, conversion
                                          │
                              Sentiment Model ──→ sentiment_score
                                          │
                                 Churn Model ──→ churn_probability
```

A price increase doesn't just affect demand — it cascades through sentiment into churn. The system captures these real business dynamics that individual models miss.

---

## API Reference (26 routes)

### Authentication Required

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/sessions` | GET | List user's chat sessions |
| `/api/sessions` | POST | Create new session |
| `/api/sessions/{id}` | GET/PATCH/DELETE | Manage a session |
| `/api/sessions/{id}/messages` | GET | Get session messages |
| `/api/chat/stream` | POST | Streaming chat with RAG (SSE) |
| `/api/knowledge` | GET | List uploaded documents |
| `/api/knowledge/upload` | POST | Upload document to knowledge base |
| `/api/knowledge/{id}` | DELETE | Delete a document |

### Public (Backward-Compatible)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | Model training status |
| `/api/simulate` | POST | Run causal simulation |
| `/api/simulate/compare` | POST | Baseline vs scenario |
| `/api/counterfactual` | POST | DiCE counterfactuals |
| `/api/chat` | POST | Single-shot chat (non-streaming) |
| `/api/upload` | POST | Upload dataset for ML training |
| `/api/train` | POST | Train specific model |
| `/api/train/all` | POST | Train all models |

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Frontend | React 18 + TypeScript + Tailwind CSS | UI with streaming chat |
| Auth | Supabase Auth + JWT | User management, session tokens |
| Database | Supabase Postgres | Chat sessions, messages, documents |
| Vector DB | pgvector (in Supabase) | Document embeddings for RAG |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Local, free, 384-dim vectors |
| ML Models | LightGBM + DistilBERT | Business prediction |
| Explainability | SHAP + DiCE-ML | Feature attribution + counterfactuals |
| LLM | Groq (Llama 3.3 70B) | Natural language insight generation |
| Streaming | Server-Sent Events (SSE) | Real-time token delivery |
| Deployment | Docker Compose | One-command setup |

---

## Setup

See **[SETUP.md](SETUP.md)** for complete installation instructions.

Quick version:
1. Create Supabase project, run `supabase/schema.sql`
2. Add keys to `backend/.env` and `frontend/.env`
3. `pip install -r requirements.txt` (backend)
4. `npm install` (frontend)
5. Start both servers, sign up, and start chatting
