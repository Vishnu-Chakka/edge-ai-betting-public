from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    # SQLite doesn't support pool_size / max_overflow, but the async
    # engine ignores unsupported kwargs when using aiosqlite.
    future=True,
)

async_session_factory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def create_all_tables() -> None:
    """Import all ORM models and issue CREATE TABLE IF NOT EXISTS."""
    from app.models.schemas import Base  # noqa: F811

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_async_session() -> AsyncSession:
    """Return a fresh async session (non-generator helper)."""
    return async_session_factory()
