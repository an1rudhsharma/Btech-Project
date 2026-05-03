from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.routes import simulate, upload, train, chat, counterfactual

app = FastAPI(
    title=settings.app_name,
    description="AI-driven Business Decision Simulation System with causal model propagation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(simulate.router, prefix="/api", tags=["Simulation"])
app.include_router(upload.router, prefix="/api", tags=["Upload"])
app.include_router(train.router, prefix="/api", tags=["Training"])
app.include_router(chat.router, prefix="/api", tags=["Chat"])
app.include_router(counterfactual.router, prefix="/api", tags=["Counterfactual"])


@app.get("/api/status")
async def get_status():
    """Check which models are trained and ready."""
    from app.engine.simulation import simulation_engine
    return {
        "models": simulation_engine.get_model_status(),
        "app": settings.app_name,
    }
