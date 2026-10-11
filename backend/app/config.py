"""
Microplastic Vision – application configuration.

All settings are read from environment variables or a .env file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


_IS_VERCEL = os.getenv("VERCEL") == "1"
_VERCEL_CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "data" / "checkpoints"


class Settings(BaseSettings):
    # ── Application ──────────────────────────────────────────────
    app_name: str = "Microplastic Vision"
    app_env: str = "development"
    debug: bool = True
    secret_key: str = "change-me-in-production"

    # ── Server ───────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000

    # ── Database ─────────────────────────────────────────────────
    database_url: str = (
        "sqlite+aiosqlite:////tmp/microplastic_vision.db"
        if _IS_VERCEL else "sqlite+aiosqlite:///./data/microplastic_vision.db"
    )

    # ── Storage ──────────────────────────────────────────────────
    upload_dir: str = "/tmp/microplastic_vision/uploads" if _IS_VERCEL else "data/uploads"
    dataset_dir: str = "/tmp/microplastic_vision/datasets" if _IS_VERCEL else "data/datasets"
    model_dir: str = "/tmp/microplastic_vision/models" if _IS_VERCEL else "data/models"
    checkpoint_dir: str = str(_VERCEL_CHECKPOINT_DIR) if _IS_VERCEL else "data/checkpoints"

    # ── Upload limits ────────────────────────────────────────────
    max_upload_size_mb: int = 4 if _IS_VERCEL else 50
    allowed_extensions: str = "jpg,jpeg,png,tiff,tif,bmp"

    # ── ML defaults ──────────────────────────────────────────────
    default_image_size: int = 224
    default_batch_size: int = 16
    default_epochs: int = 25
    default_learning_rate: float = 0.001
    confidence_threshold: float = 0.5

    # ── CORS ─────────────────────────────────────────────────────
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    # ── Demo mode ────────────────────────────────────────────────
    demo_mode: bool = True

    # ── Optional YOLO ────────────────────────────────────────────
    yolo_enabled: bool = False
    yolo_weights_path: str = ""

    # ── Derived helpers ──────────────────────────────────────────
    @property
    def allowed_extensions_list(self) -> List[str]:
        return [e.strip().lower() for e in self.allowed_extensions.split(",") if e.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def ensure_dirs(self) -> None:
        """Create required data directories."""
        for d in (self.upload_dir, self.dataset_dir, self.model_dir, self.checkpoint_dir):
            Path(d).mkdir(parents=True, exist_ok=True)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# Singleton
_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.ensure_dirs()
    return _settings
