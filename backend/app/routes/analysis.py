"""
Microplastic Vision – image analysis routes.
"""

from __future__ import annotations

import logging
import mimetypes
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, Query
from fastapi.responses import Response
from PIL import Image
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_db
from app.models import Analysis
from app.schemas import AnalysisResponse, AnalysisListResponse, BatchAnalysisResponse
from app.ml.inference import get_inference_service
from app.ml.detector import detect as yolo_detect, is_yolo_available
from app.storage import upload_image, download_image, delete_image, is_supabase_path

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["analysis"])


def _validate_upload(file: UploadFile) -> None:
    """Validate file extension and size."""
    settings = get_settings()

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    ext = Path(file.filename).suffix.lower().lstrip(".")
    if ext not in settings.allowed_extensions_list:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '.{ext}'. "
                   f"Allowed: {settings.allowed_extensions_list}",
        )


async def _save_upload(file: UploadFile) -> tuple[str, str, str]:
    """Store an upload and return (persistent path, type, local inference path)."""
    settings = get_settings()
    ext = Path(file.filename).suffix.lower()
    storage_name = f"{uuid.uuid4().hex}{ext}"
    storage_path = Path(settings.upload_dir) / storage_name
    storage_path.parent.mkdir(parents=True, exist_ok=True)

    content = await file.read()
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum size of {settings.max_upload_size_mb} MB.",
        )

    storage_path.write_bytes(content)
    if settings.supabase_url and settings.supabase_secret_key:
        try:
            persistent_path = await upload_image(
                storage_name,
                content,
                file.content_type or mimetypes.guess_type(storage_name)[0] or "application/octet-stream",
            )
        except Exception as exc:
            storage_path.unlink(missing_ok=True)
            logger.exception("Could not store image in Supabase Storage")
            raise HTTPException(status_code=502, detail="Image storage is temporarily unavailable.") from exc
        return persistent_path, ext.lstrip("."), str(storage_path)
    return str(storage_path), ext.lstrip("."), str(storage_path)


def _open_and_validate_image(storage_path: str) -> Image.Image:
    """Open image and verify it's actually decodable."""
    try:
        img = Image.open(storage_path)
        img.verify()
        # Re-open after verify (verify can close the image)
        img = Image.open(storage_path)
        img.load()
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot decode image: {e}")


async def _run_analysis(file: UploadFile, db: AsyncSession) -> Analysis:
    """Core analysis logic shared by single and batch endpoints."""
    _validate_upload(file)
    storage_path, file_type, local_path = await _save_upload(file)

    remote_storage = is_supabase_path(storage_path)
    try:
        img = _open_and_validate_image(local_path)
        w, h = img.size

        # Run inference
        service = get_inference_service()
        result = service.predict(img)

        # Optional YOLO
        detections = None
        detection_count = None
        if is_yolo_available():
            det_result = yolo_detect(img)
            if det_result.get("available"):
                detections = det_result.get("detections", [])
                detection_count = det_result.get("count", 0)
    except Exception:
        if remote_storage:
            try:
                await delete_image(storage_path)
            except Exception:
                logger.exception("Could not clean up failed Supabase upload")
            Path(local_path).unlink(missing_ok=True)
        raise
    finally:
        if remote_storage:
            Path(local_path).unlink(missing_ok=True)

    analysis = Analysis(
        id=str(uuid.uuid4()),
        original_filename=file.filename,
        storage_path=storage_path,
        file_type=file_type,
        image_width=w,
        image_height=h,
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        model_name=result.get("model_name"),
        model_version=result.get("model_version"),
        status=result.get("status", "completed"),
        is_demo=result.get("is_demo", False),
        warnings=result.get("warnings", []),
        preprocessing_config=result.get("preprocessing_config"),
        detections=detections,
        detection_count=detection_count,
    )

    db.add(analysis)
    await db.commit()
    await db.refresh(analysis)
    return analysis


# ── Single image ────────────────────────────────────────────────


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Upload and classify a single image."""
    analysis = await _run_analysis(file, db)
    return AnalysisResponse(**analysis.to_dict())


# ── Batch ───────────────────────────────────────────────────────


@router.post("/analyze/batch", response_model=BatchAnalysisResponse)
async def analyze_batch(
    files: List[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Upload and classify multiple images."""
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 images per batch.")

    results: List[AnalysisResponse] = []
    errors = 0

    for file in files:
        try:
            analysis = await _run_analysis(file, db)
            results.append(AnalysisResponse(**analysis.to_dict()))
        except HTTPException:
            errors += 1
        except Exception as e:
            logger.error("Batch analysis error for %s: %s", file.filename, e)
            errors += 1

    return BatchAnalysisResponse(
        results=results,
        total_processed=len(results),
        total_errors=errors,
    )


# ── History ─────────────────────────────────────────────────────


@router.get("/analyses", response_model=AnalysisListResponse)
async def list_analyses(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    """List analysis history with pagination."""
    offset = (page - 1) * per_page

    total_q = await db.execute(select(func.count(Analysis.id)))
    total = total_q.scalar() or 0

    q = await db.execute(
        select(Analysis)
        .order_by(Analysis.created_at.desc())
        .offset(offset)
        .limit(per_page)
    )
    items = [AnalysisResponse(**a.to_dict()) for a in q.scalars().all()]

    return AnalysisListResponse(items=items, total=total, page=page, per_page=per_page)


@router.get("/analyses/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis(analysis_id: str, db: AsyncSession = Depends(get_db)):
    """Retrieve a single analysis by ID."""
    q = await db.execute(select(Analysis).where(Analysis.id == analysis_id))
    analysis = q.scalar_one_or_none()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    return AnalysisResponse(**analysis.to_dict())


def _remove_upload_if_owned(storage_path: str) -> None:
    """Remove an analysis upload only when it resolves inside the upload folder."""
    settings = get_settings()
    upload_root = Path(settings.upload_dir).resolve()
    stored_path = Path(storage_path).resolve()
    if upload_root in stored_path.parents and stored_path.is_file():
        try:
            stored_path.unlink()
        except OSError as exc:
            logger.warning("Could not remove analysis upload %s: %s", stored_path, exc)


async def _delete_stored_upload(storage_path: str) -> None:
    if is_supabase_path(storage_path):
        try:
            await delete_image(storage_path)
        except Exception:
            logger.exception("Could not remove image from Supabase Storage")
        return
    _remove_upload_if_owned(storage_path)


@router.delete("/analyses/{analysis_id}")
async def delete_analysis(analysis_id: str, db: AsyncSession = Depends(get_db)):
    """Delete one analysis and its unreferenced uploaded image."""
    q = await db.execute(select(Analysis).where(Analysis.id == analysis_id))
    analysis = q.scalar_one_or_none()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    storage_path = analysis.storage_path
    await db.delete(analysis)
    await db.commit()
    remaining = await db.execute(select(func.count(Analysis.id)).where(Analysis.storage_path == storage_path))
    if (remaining.scalar() or 0) == 0:
        await _delete_stored_upload(storage_path)
    return {"deleted": 1}


@router.delete("/analyses")
async def clear_analyses(db: AsyncSession = Depends(get_db)):
    """Clear analysis history and remove the corresponding uploaded images."""
    q = await db.execute(select(Analysis))
    analyses = q.scalars().all()
    storage_paths = {analysis.storage_path for analysis in analyses}
    for analysis in analyses:
        await db.delete(analysis)
    await db.commit()
    for storage_path in storage_paths:
        await _delete_stored_upload(storage_path)
    return {"deleted": len(analyses)}


# ── Uploaded image serving ──────────────────────────────────────


@router.get("/uploads/{filename}")
async def serve_upload(filename: str, db: AsyncSession = Depends(get_db)):
    """Serve an uploaded image file."""
    from fastapi.responses import FileResponse

    settings = get_settings()
    # Sanitize filename to prevent path traversal
    safe_name = Path(filename).name
    # Supabase objects use UUID filenames; resolve them through the private
    # bucket with the backend-only key so no storage secret reaches the browser.
    if settings.supabase_url and settings.supabase_secret_key:
        result = await db.execute(
            select(Analysis.storage_path)
            .where(Analysis.storage_path.endswith(f"/{safe_name}"))
            .limit(1)
        )
        stored_path = result.scalar_one_or_none()
        if not stored_path or not is_supabase_path(stored_path):
            raise HTTPException(status_code=404, detail="File not found.")
        try:
            content = await download_image(stored_path)
        except Exception as exc:
            logger.exception("Could not read image from Supabase Storage")
            raise HTTPException(status_code=502, detail="Image storage is temporarily unavailable.") from exc
        media_type = mimetypes.guess_type(safe_name)[0] or "application/octet-stream"
        return Response(content=content, media_type=media_type)

    file_path = Path(settings.upload_dir) / safe_name

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")

    return FileResponse(str(file_path))
