from pydantic_settings import BaseSettings
from pathlib import Path
import os


class Settings(BaseSettings):
    app_name: str = "AI Business Decision Simulator"
    groq_api_key: str = ""
    model_dir: Path = Path("./trained_models")
    data_dir: Path = Path(os.path.dirname(__file__)) / "data"
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]
    max_upload_size_mb: int = 200
    llm_model: str = "llama-3.3-70b-versatile"

    # Supabase
    supabase_url: str = ""
    supabase_anon_key: str = ""
    supabase_service_key: str = ""
    supabase_jwt_secret: str = ""

    class Config:
        env_file = ".env"


settings = Settings()
settings.model_dir.mkdir(parents=True, exist_ok=True)
