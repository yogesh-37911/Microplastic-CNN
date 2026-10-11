"""
Microplastic Vision – async database engine and session factory.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""
    pass


_engine = None
_session_factory = None


def _get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            settings.database_url,
            echo=settings.debug,
            future=True,
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=_get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_factory


async def get_db() -> AsyncSession:  # type: ignore[misc]
    """FastAPI dependency – yields an async session."""
    factory = get_session_factory()
    async with factory() as session:
        yield session


async def init_db() -> None:
    """Create all tables (call once at startup)."""
    engine = _get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Lightweight forward-compatible SQLite migration for additive fields.
        # SQLAlchemy create_all does not alter existing local databases.
        if engine.dialect.name == "sqlite":
            from sqlalchemy import inspect

            def add_missing_columns(sync_conn):
                inspector = inspect(sync_conn)
                tables = inspector.get_table_names()
                if "training_runs" in tables:
                    run_columns = {col["name"] for col in inspector.get_columns("training_runs")}
                    if "test_metrics" not in run_columns:
                        sync_conn.exec_driver_sql("ALTER TABLE training_runs ADD COLUMN test_metrics JSON")
                if "analyses" in tables:
                    analysis_columns = {col["name"] for col in inspector.get_columns("analyses")}
                    if "warnings" not in analysis_columns:
                        sync_conn.exec_driver_sql("ALTER TABLE analyses ADD COLUMN warnings JSON")

            await conn.run_sync(add_missing_columns)
