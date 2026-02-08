from __future__ import annotations

from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ── Application ──────────────────────────────────────────────────────
    APP_NAME: str = "Edge AI Betting"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True
    ENVIRONMENT: str = "development"

    # ── Database ─────────────────────────────────────────────────────────
    DATABASE_URL: str = "sqlite+aiosqlite:///./edge_ai.db"

    # ── Redis ────────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_ENABLED: bool = False

    # ── Authentication / JWT ─────────────────────────────────────────────
    SECRET_KEY: str = "dev-secret-key-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours
    ALGORITHM: str = "HS256"

    # ── External API Keys ────────────────────────────────────────────────
    ANTHROPIC_API_KEY: Optional[str] = None
    ANTHROPIC_MODEL: str = "claude-sonnet-4-5-20250929"
    ANTHROPIC_MAX_TOKENS: int = 8192

    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-3.5-turbo"

    ODDS_API_KEY: Optional[str] = None
    ODDS_API_BASE_URL: str = "https://api.the-odds-api.com/v4"

    # Optional OpenAI fallback
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"

    # ── Sports Data ──────────────────────────────────────────────────────
    ESPN_API_BASE_URL: str = "https://site.api.espn.com/apis/site/v2/sports"
    DEFAULT_ENABLED_SPORTS: str = "basketball,football,baseball,hockey"

    # ── Model / ML Defaults ──────────────────────────────────────────────
    DEFAULT_MODEL_NAME: str = "xgb_v1"
    MIN_EV_THRESHOLD: float = 0.02  # 2 % minimum edge
    DEFAULT_KELLY_FRACTION: float = 0.25  # quarter-Kelly
    MAX_BET_PERCENT: float = 0.05  # 5 % of bankroll

    # ── Rate Limiting ────────────────────────────────────────────────────
    RATE_LIMIT_PER_MINUTE: int = 30

    # ── CORS ─────────────────────────────────────────────────────────────
    CORS_ORIGINS: str = "*"

    # ── Logging ──────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }


# Module-level singleton so other modules can do `from app.config import settings`
settings = Settings()
