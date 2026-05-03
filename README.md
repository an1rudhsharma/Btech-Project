# AI Business Decision Simulation System

An AI-driven business decision simulator that predicts the cascading impact of pricing, marketing, feature, and customer decisions using causally-connected machine learning models, counterfactual explanations, and LLM-powered natural language insights.

---

## What This System Does

Businesses make decisions (change price, adjust marketing budget, add features) without understanding how these changes propagate through customer behavior. This system solves that by:

1. **Churn Prediction** - Predicts customer attrition probability based on pricing, features, usage, and propagated sentiment signals
2. **Sentiment Analysis** - Analyzes how customer perception shifts when business parameters change
3. **Marketing Impact** - Forecasts click-through rate, conversions, and engagement under varying budgets
4. **Pricing Impact** - Predicts demand, revenue, and price elasticity from pricing decisions

The key innovation: these models are **not independent**. They execute in a causal dependency graph where each model's output feeds as input to downstream models, creating a true simulation engine.

### Causal Propagation Flow

```
Price Change → Pricing Model → demand, sentiment_impact
                                          ↓
Marketing Change → Marketing Model → engagement, conversion
                                          ↓
                              Sentiment Model → sentiment_score
                                          ↓
                                 Churn Model → churn_probability
```

A price increase doesn't just affect demand - it cascades through sentiment into churn. The system captures these real business dynamics.

---

## Why These Design Decisions

### Why LightGBM (not RandomForest, XGBoost, or Neural Networks)

- **Speed**: 5-10x faster training than XGBoost, critical for interactive simulation
- **Accuracy on tabular data**: Consistently outperforms alternatives on structured business data
- **Native SHAP support**: TreeExplainer works directly with LightGBM for instant feature attribution
- **DiCE-ML compatibility**: sklearn backend works seamlessly for counterfactual generation
- **Handles missing values natively**: Important for auto column detection on arbitrary datasets

**Tradeoff**: Slightly less interpretable than linear models, but SHAP compensates. Neural networks would give marginal accuracy gains but destroy explainability.

### Why DistilBERT (not TF-IDF+LogReg, full BERT, or DeBERTa)

- **97% of BERT accuracy at 60% the size**: Best accuracy-to-resource ratio
- **No GPU required for inference**: Runs on CPU during simulation
- **Pre-trained**: Works immediately without labeled data (zero-shot capable)
- **Understands context**: Unlike bag-of-words, handles negation ("not good") and sarcasm

**Tradeoff**: DeBERTa gives ~2% better accuracy but requires GPU and 3x memory. For a simulation system where sentiment is one input signal, DistilBERT is sufficient.

### Why Causal DAG (not parallel model execution)

- **Real businesses have feedback loops**: Price affects sentiment affects churn
- **Parallel execution misses compound effects**: A 10% price increase + 20% marketing cut has multiplicative, not additive, impact on churn
- **Traceable propagation**: Every prediction shows exactly how signals flowed through the system

**Tradeoff**: Sequential execution is slower than parallel (~4x). Acceptable because simulation latency is <500ms total. A full Bayesian Network would be theoretically purer but computationally expensive and harder to explain in a viva.

### Why DiCE-ML Counterfactuals (not just SHAP)

- **SHAP is backward-looking**: "Churn is high because price is high" (explains the past)
- **DiCE is forward-looking**: "Reduce price by $5 to prevent churn" (prescribes action)
- **Actionable recommendations**: Business users need to know WHAT TO DO, not just why something happened
- **Diversity**: Generates multiple alternative paths, not just one

**Tradeoff**: DiCE's random method is faster but less diverse than genetic algorithm method. Random is sufficient for real-time demo; genetic can be added for batch analysis.

### Why Groq LLM (not OpenAI, local Ollama, or no LLM)

- **Free tier**: Sufficient for demo and development
- **Fast**: ~500 tokens/sec with Llama 3.1 70B (10x faster than GPT-4)
- **Chain-of-Thought capable**: Follows structured reasoning prompts accurately
- **No self-hosting required**: Unlike Ollama, no 8GB RAM requirement

**Tradeoff**: Dependent on external API. Groq's free tier has rate limits. Fallback to simpler template-based responses if API is unavailable.

### Why Chain-of-Thought Prompting (not direct LLM queries)

- **Prevents hallucination**: LLM must reason through SHAP values → baseline comparison → counterfactuals before making claims
- **Every claim is traceable**: No invented numbers, all data comes from model outputs
- **Structured output**: Consistent format (Key Drivers → Causal Chain → Impact → Recommendation)

**Tradeoff**: More tokens per response (higher cost at scale). Acceptable because accuracy and trustworthiness are paramount for a "business advisor" system.

---

## How It Works (Technical Architecture)

### Backend (Python/FastAPI)

```
backend/
├── app/
│   ├── main.py                 # FastAPI entry + CORS
│   ├── config.py               # Environment settings
│   ├── models/
│   │   ├── base.py             # Abstract base with SHAP integration
│   │   ├── churn.py            # LightGBM classifier
│   │   ├── sentiment.py        # DistilBERT pipeline
│   │   ├── marketing.py        # LightGBM regressor
│   │   └── pricing.py          # LightGBM regressor + elasticity
│   ├── engine/
│   │   ├── causal_graph.py     # DAG definition + topological sort
│   │   ├── simulation.py       # Sequential execution engine
│   │   ├── counterfactual.py   # DiCE-ML wrapper
│   │   ├── column_detector.py  # Auto-detect columns from any CSV
│   │   └── feature_builder.py  # Safe feature engineering
│   ├── llm/
│   │   ├── orchestrator.py     # Intent parsing + full pipeline
│   │   ├── insights.py         # CoT insight generation
│   │   └── prompts.py          # Structured prompt templates
│   ├── routes/
│   │   ├── simulate.py         # Simulation endpoints
│   │   ├── counterfactual.py   # What-if endpoints
│   │   ├── upload.py           # Dataset management
│   │   ├── train.py            # Model training
│   │   └── chat.py             # Natural language interface
│   └── data/                   # Sample datasets
```

### Frontend (React/TypeScript/Tailwind)

```
frontend/
├── src/
│   ├── App.tsx                 # Router + navigation
│   ├── pages/
│   │   ├── Home.tsx            # Dashboard + model status
│   │   ├── Simulate.tsx        # Sliders + real-time predictions + charts
│   │   ├── Counterfactuals.tsx # DiCE explorer
│   │   ├── Chat.tsx            # Natural language interface
│   │   └── Upload.tsx          # Dataset management
│   └── api/
│       └── client.ts           # Typed API client
```

### Data Flow

1. User adjusts sliders or asks a question
2. Frontend sends parameters to `/api/simulate` or `/api/chat`
3. Backend parses intent (LLM for chat, direct for sliders)
4. Causal engine executes models in topological order:
   - Pricing → outputs demand, price_sentiment_impact
   - Marketing → outputs conversion, marketing_effect
   - Sentiment → inputs price_impact + marketing_effect → outputs sentiment_score
   - Churn → inputs all upstream signals → outputs churn_probability
5. SHAP explains WHY (top feature drivers)
6. DiCE-ML generates WHAT TO CHANGE (counterfactuals)
7. LLM synthesizes everything into natural language insight (CoT)
8. Frontend renders predictions, charts, propagation trace, and recommendations

---

## Local Setup Instructions

### Prerequisites

- Python 3.11+ (3.12 or 3.13 also work)
- Node.js 18+
- Git
- (Optional) Docker & Docker Compose

### Quick Start with Docker

```bash
git clone <repo-url>
cd betchProject
cp .env.example .env
# Edit .env and add your GROQ_API_KEY (get free at https://console.groq.com)
docker-compose up --build
```

- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Docs (Swagger): http://localhost:8000/docs

### Manual Setup (Without Docker)

#### Backend

```bash
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env: add GROQ_API_KEY (get free at https://console.groq.com)

# Generate sample data (already included, but regenerate if needed)
python -c "from app.generate_data import create_all_samples; from pathlib import Path; create_all_samples(Path('app/data'))"

# Start server
uvicorn app.main:app --reload --port 8000
```

#### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Start dev server (proxies /api to backend)
npm run dev
```

Open http://localhost:5173

### First Run

1. Open the Dashboard (http://localhost:5173)
2. Click **"Train All Models"** - this trains all 4 models on sample data (~10 seconds)
3. Go to **Simulate** tab - adjust sliders and see predictions
4. Try **What-If** tab - find counterfactual recommendations
5. Use **Chat** tab - ask natural language questions (requires GROQ_API_KEY)

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | For chat feature | Free API key from [console.groq.com](https://console.groq.com) |
| `MODEL_DIR` | No (default: ./trained_models) | Where trained models are persisted |

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | Model training status |
| `/api/simulate` | POST | Run causal simulation |
| `/api/simulate/compare` | POST | Baseline vs scenario comparison |
| `/api/simulate/batch` | POST | Multiple scenarios at once |
| `/api/counterfactual` | POST | Generate DiCE counterfactuals |
| `/api/chat` | POST | Natural language query (full pipeline) |
| `/api/chat/report` | POST | Generate business report |
| `/api/upload` | POST | Upload dataset (CSV/Excel) |
| `/api/upload/datasets` | GET | List available datasets |
| `/api/train` | POST | Train specific model |
| `/api/train/all` | POST | Train all models on sample data |

---

## Sample Datasets Included

| Dataset | Rows | Purpose | Key Columns |
|---------|------|---------|-------------|
| `sample_churn.csv` | 5000 | Churn prediction | price, marketing_spend, usage, tenure, satisfaction, churn |
| `sample_marketing.csv` | 4000 | Marketing impact | marketing_spend, impressions, clicks, price, conversion_rate |
| `sample_pricing.csv` | 3000 | Pricing/demand | price, marketing_spend, usage, demand |
| `sample_sentiment.csv` | 2000 | Sentiment analysis | text, sentiment |

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Backend Framework | FastAPI | Async, auto-docs, type-safe |
| ML Models | LightGBM | Fast, accurate on tabular data, SHAP-compatible |
| NLP | DistilBERT (HuggingFace) | Pre-trained, no GPU needed |
| Explainability | SHAP + DiCE-ML | Why + What-to-change |
| LLM | Groq (Llama 3.1 70B) | Free, fast, CoT-capable |
| Frontend | React + TypeScript + Tailwind | Modern, type-safe, responsive |
| Charts | Recharts | Composable React charting |
| Deployment | Docker Compose | One-command local setup |
