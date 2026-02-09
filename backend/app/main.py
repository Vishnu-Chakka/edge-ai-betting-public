from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import create_all_tables

logger = logging.getLogger(__name__)


# ── Lifespan ─────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup / shutdown lifecycle for the FastAPI application."""
    logger.info("Starting Edge AI Betting backend ...")
    await create_all_tables()
    logger.info("Database tables created (if not existing).")
    yield
    logger.info("Shutting down Edge AI Betting backend ...")


# ── App factory ──────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-powered sports betting analysis and recommendations",
    lifespan=lifespan,
)

# ── Middleware ────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──────────────────────────────────────────────────────────────────

from app.api.routes.chat import router as chat_router  # noqa: E402
from app.api.routes.games import router as games_router  # noqa: E402
from app.api.routes.picks import router as picks_router  # noqa: E402
from app.api.routes.user import router as user_router  # noqa: E402
from app.api.routes.wallet import router as wallet_router  # noqa: E402

app.include_router(chat_router, prefix="/api/chat", tags=["chat"])
app.include_router(picks_router, prefix="/api/picks", tags=["picks"])
app.include_router(games_router, prefix="/api/games", tags=["games"])
app.include_router(user_router, prefix="/api/users", tags=["users"])
app.include_router(wallet_router, prefix="/api/wallet", tags=["wallet"])


# ── Root endpoints ───────────────────────────────────────────────────────────


@app.get("/")
async def root() -> dict:
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
    }


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
