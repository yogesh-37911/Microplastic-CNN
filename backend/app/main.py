"""
Microplastic Vision – FastAPI application entry point.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select, func

from app.config import get_settings
from app.database import init_db, get_db, get_session_factory
from app.models import Analysis, TrainingRun
from app.routes.analysis import router as analysis_router
from app.routes.datasets import router as datasets_router
from app.schemas import HealthResponse, StatusResponse
from app.ml.inference import get_inference_service
from app.ml.detector import is_yolo_available

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    settings = get_settings()
    settings.ensure_dirs()
    await init_db()
    # Pre-load inference service (will auto-detect checkpoints)
    get_inference_service()
    logger.info(
        "Microplastic Vision started (demo_mode=%s, device=%s)",
        settings.demo_mode,
        "cuda" if __import__("torch").cuda.is_available() else "cpu",
    )
    yield
    logger.info("Microplastic Vision shutting down.")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description=(
            "Image-based machine learning for preliminary microplastic screening. "
            "Results are model predictions for visual screening only – confirmed "
            "polymer identification requires FTIR, Raman, or other chemical evidence."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(analysis_router)
    app.include_router(datasets_router)

    # ── Health & Status ─────────────────────────────────────────

    @app.get("/api/health", response_model=HealthResponse, tags=["system"])
    async def health():
        return HealthResponse()

    @app.get("/api/status", response_model=StatusResponse, tags=["system"])
    async def status():
        service = get_inference_service()
        factory = get_session_factory()
        async with factory() as db:
            total_analyses = (await db.execute(select(func.count(Analysis.id)))).scalar() or 0
            total_runs = (await db.execute(select(func.count(TrainingRun.id)))).scalar() or 0

        return StatusResponse(
            app_name=settings.app_name,
            demo_mode=not service.is_loaded,
            model_loaded=service.is_loaded,
            model_ready=service.is_ready,
            model_scope="Exploratory microfibre screening on microscope filter crops; not validated for stock photos, illustrations, or general plastic-product images.",
            active_model=service.model_name,
            yolo_available=is_yolo_available(),
            confidence_threshold=settings.confidence_threshold,
            total_analyses=total_analyses,
            total_runs=total_runs,
        )

    # ── Optional YOLO detect endpoint ───────────────────────────

    @app.post("/api/detect", tags=["detection"])
    async def detect_particles(
        file: __import__("fastapi").UploadFile = __import__("fastapi").File(...),
    ):
        """Run YOLO particle detection (optional – requires configured detector)."""
        if not is_yolo_available():
            return {
                "available": False,
                "message": "YOLO detection is not configured.",
                "detections": [],
                "count": 0,
            }

        from PIL import Image as PILImage
        import io

        content = await file.read()
        try:
            img = PILImage.open(io.BytesIO(content))
            img.load()
        except Exception:
            from fastapi import HTTPException
            raise HTTPException(status_code=400, detail="Cannot decode image.")

        from app.ml.detector import detect
        return detect(img)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
