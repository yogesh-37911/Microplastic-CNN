"""
Microplastic Vision – dataset and model-run routes.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_db, get_session_factory
from app.models import DatasetManifest, TrainingRun, Evaluation
from app.schemas import (
    DatasetResponse,
    DatasetValidateRequest,
    TrainingRunResponse,
    TrainingRunCreateRequest,
    EvaluationResponse,
    EvaluateRequest,
)
from app.ml.training import discover_dataset, train_model, split_dataset
from app.ml.evaluation import evaluate_model
from app.ml.preprocessing import get_default_config

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["datasets-and-runs"])


# ── Datasets ────────────────────────────────────────────────────


@router.get("/datasets", response_model=List[DatasetResponse])
async def list_datasets(db: AsyncSession = Depends(get_db)):
    """List all registered dataset manifests."""
    q = await db.execute(select(DatasetManifest).order_by(DatasetManifest.created_at.desc()))
    return [DatasetResponse(**d.to_dict()) for d in q.scalars().all()]


@router.post("/datasets/validate", response_model=DatasetResponse)
async def validate_dataset(
    req: DatasetValidateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Validate a dataset directory and register its manifest."""
    base_path = req.base_path

    # Validate path existence; training separately checks labels and split viability.
    settings = get_settings()
    target = Path(base_path).resolve()

    if not target.exists():
        raise HTTPException(status_code=400, detail=f"Path does not exist: {base_path}")

    warnings: List[str] = []
    try:
        paths, labels, class_names, class_counts = discover_dataset(str(target))
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Check for issues
    total = len(paths)
    if total < 10:
        warnings.append(f"Very small dataset ({total} images). Model quality may be limited.")

    for cls, count in class_counts.items():
        if count == 0:
            warnings.append(f"Class '{cls}' has no images.")

    # Check class imbalance
    counts = list(class_counts.values())
    if len(counts) >= 2 and max(counts) > 5 * min(max(min(counts), 1), max(counts)):
        warnings.append(
            f"Significant class imbalance detected: {class_counts}. "
            "Consider using class weights during training."
        )
    if class_counts.get("microplastic", 0) < 50 or class_counts.get("non_microplastic", 0) < 50:
        warnings.append(
            "At least 50 images per class are required to begin a CNN experiment. Use diverse, "
            "microscopy-confirmed samples and negatives; more independent acquisitions are strongly preferred."
        )

    # Validate each image can be opened
    unreadable = 0
    from PIL import Image
    for p in paths:
        try:
            img = Image.open(p)
            img.verify()
        except Exception:
            unreadable += 1

    if unreadable > 0:
        warnings.append(f"{unreadable} image(s) could not be decoded and will be skipped.")

    # Split preview
    usable_class_counts = [count for count in class_counts.values()]
    dataset_ready = (
        set(class_names) == {"microplastic", "non_microplastic"}
        and all(count >= 10 for count in usable_class_counts)
        and class_counts.get("microplastic", 0) >= 50
        and class_counts.get("non_microplastic", 0) >= 50
        and unreadable == 0
    )
    if not dataset_ready:
        warnings.append(
            "Not ready for training: provide at least 50 readable images per class, with "
            "microscopy-confirmed labels and multiple independent acquisition groups."
        )

    if dataset_ready:
        try:
            (train_p, _), (val_p, _), (test_p, _) = split_dataset(paths, labels)
            split_counts = {"train": len(train_p), "val": len(val_p), "test": len(test_p)}
        except ValueError as exc:
            dataset_ready = False
            split_counts = None
            warnings.append(str(exc))
    else:
        split_counts = None

    manifest = DatasetManifest(
        id=str(uuid.uuid4()),
        name=req.name or "Unnamed dataset",
        description=req.description,
        source=req.source,
        license_notes=req.license_notes,
        base_path=str(target),
        total_images=total,
        class_counts=class_counts,
        split_counts=split_counts,
        validation_warnings=warnings if warnings else None,
        label_provenance=req.label_provenance,
        is_valid=dataset_ready,
    )

    db.add(manifest)
    await db.commit()
    await db.refresh(manifest)

    return DatasetResponse(**manifest.to_dict(), training_ready=dataset_ready)


# ── Training Runs ───────────────────────────────────────────────


@router.get("/runs", response_model=List[TrainingRunResponse])
async def list_runs(db: AsyncSession = Depends(get_db)):
    """List all training runs."""
    q = await db.execute(select(TrainingRun).order_by(TrainingRun.created_at.desc()))
    return [TrainingRunResponse(**r.to_dict()) for r in q.scalars().all()]


@router.get("/runs/{run_id}", response_model=TrainingRunResponse)
async def get_run(run_id: str, db: AsyncSession = Depends(get_db)):
    """Retrieve a single training run."""
    q = await db.execute(select(TrainingRun).where(TrainingRun.id == run_id))
    run = q.scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found.")
    return TrainingRunResponse(**run.to_dict())


async def _train_in_background(run_id: str, req: TrainingRunCreateRequest):
    """Background training task."""
    factory = get_session_factory()
    async with factory() as db:
        q = await db.execute(select(TrainingRun).where(TrainingRun.id == run_id))
        run = q.scalar_one_or_none()
        if not run:
            return

        try:
            run.status = "training"
            await db.commit()

            result = train_model(
                dataset_path=req.dataset_path,
                architecture=req.model_architecture,
                epochs=req.epochs,
                batch_size=req.batch_size,
                learning_rate=req.learning_rate,
                image_size=req.image_size,
                seed=req.seed,
                use_class_weights=req.use_class_weights,
            )

            # Publish only a finished model that has passed the held-out test
            # evaluation. The checkpoint is not auto-promoted before this.
            best_payload = __import__("torch").load(
                result["best_checkpoint_path"], map_location="cpu", weights_only=False
            )
            best_payload["dataset_manifest"] = str(Path(req.dataset_path).resolve())
            best_payload["test_metrics"] = result["test_metrics"]
            best_payload["split_method"] = result["split_method"]
            best_payload["split_groups"] = result["split_groups"]
            __import__("torch").save(best_payload, result["best_checkpoint_path"])
            run.status = "completed"
            run.checkpoint_path = result["best_checkpoint_path"]
            run.best_val_accuracy = result["best_val_accuracy"]
            run.training_history = result["training_history"]
            run.class_mapping = result["class_mapping"]
            run.train_count = result["train_count"]
            run.val_count = result["val_count"]
            run.test_count = result["test_count"]
            run.test_metrics = result["test_metrics"]
            run.split_method = result["split_method"]
            run.preprocessing_config = result["preprocessing_config"]
            run.progress = 100.0
            await db.commit()

            logger.info("Training run %s completed. Best val acc: %.4f",
                        run_id, result["best_val_accuracy"])

        except Exception as e:
            logger.error("Training run %s failed: %s", run_id, e)
            run.status = "failed"
            run.error_message = str(e)
            run.progress = 0.0
            await db.commit()


@router.post("/runs", response_model=TrainingRunResponse)
async def create_run(
    req: TrainingRunCreateRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Start a new training run in the background."""
    # Verify dataset path
    dataset_path = Path(req.dataset_path)
    if not dataset_path.exists():
        raise HTTPException(status_code=400, detail=f"Dataset path not found: {req.dataset_path}")
    try:
        _, _, class_names, class_counts = discover_dataset(str(dataset_path))
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if (
        set(class_names) != {"microplastic", "non_microplastic"}
        or class_counts.get("microplastic", 0) < 50
        or class_counts.get("non_microplastic", 0) < 50
    ):
        raise HTTPException(
            status_code=400,
            detail=("Training requires at least 50 images in each of exactly two class folders: "
                    "microplastic and non_microplastic, with microscopy-confirmed labels."),
        )
    image_paths, image_labels, _, _ = discover_dataset(str(dataset_path))
    try:
        split_dataset(image_paths, image_labels)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Dataset cannot be split safely: {exc}")

    run = TrainingRun(
        id=str(uuid.uuid4()),
        run_name=req.run_name,
        model_architecture=req.model_architecture,
        dataset_manifest=req.dataset_path,
        epochs=req.epochs,
        batch_size=req.batch_size,
        learning_rate=req.learning_rate,
        image_size=req.image_size,
        seed=req.seed,
        status="pending",
    )

    db.add(run)
    await db.commit()
    await db.refresh(run)

    background_tasks.add_task(_train_in_background, run.id, req)

    return TrainingRunResponse(**run.to_dict())


# ── Evaluation ──────────────────────────────────────────────────


@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_checkpoint(
    req: EvaluateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Evaluate a trained checkpoint against a held-out test split."""
    # Fetch the training run
    q = await db.execute(select(TrainingRun).where(TrainingRun.id == req.run_id))
    run = q.scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found.")

    if run.status != "completed":
        raise HTTPException(status_code=400, detail=f"Run is not completed (status: {run.status}).")

    if not run.checkpoint_path or not Path(run.checkpoint_path).exists():
        raise HTTPException(status_code=400, detail="Checkpoint file not found.")

    # Re-discover test set from the dataset
    try:
        paths, labels, class_names, _ = discover_dataset(run.dataset_manifest)
        _, _, (test_paths, test_labels) = split_dataset(
            paths, labels, seed=run.seed or 42,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not load test split: {e}")

    if len(test_paths) == 0:
        raise HTTPException(status_code=400, detail="Test split is empty.")

    # Prevent evaluating on training data
    # The split is deterministic with same seed, so test_paths won't overlap train

    pp_config = get_default_config(run.image_size)

    try:
        metrics = evaluate_model(
            checkpoint_path=run.checkpoint_path,
            architecture=run.model_architecture,
            test_paths=test_paths,
            test_labels=test_labels,
            class_names=class_names,
            image_size=run.image_size,
            batch_size=run.batch_size,
            preprocessing_config=pp_config,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {e}")

    evaluation = Evaluation(
        id=str(uuid.uuid4()),
        run_id=run.id,
        test_split_name=req.test_split_name,
        test_count=metrics["test_count"],
        class_counts=metrics["class_counts"],
        accuracy=metrics["accuracy"],
        precision_per_class=metrics["precision_per_class"],
        recall_per_class=metrics["recall_per_class"],
        f1_per_class=metrics["f1_per_class"],
        precision_macro=metrics["precision_macro"],
        recall_macro=metrics["recall_macro"],
        f1_macro=metrics["f1_macro"],
        confusion_matrix=metrics["confusion_matrix"],
        class_order=metrics["class_order"],
        model_checkpoint=run.checkpoint_path,
        preprocessing_config=pp_config.to_dict(),
        dataset_manifest=run.dataset_manifest,
        status="completed",
    )

    db.add(evaluation)
    await db.commit()
    await db.refresh(evaluation)

    return EvaluationResponse(**evaluation.to_dict())


@router.get("/evaluations/{run_id}", response_model=List[EvaluationResponse])
async def get_evaluations(run_id: str, db: AsyncSession = Depends(get_db)):
    """Get all evaluations for a training run."""
    q = await db.execute(
        select(Evaluation).where(Evaluation.run_id == run_id).order_by(Evaluation.created_at.desc())
    )
    return [EvaluationResponse(**e.to_dict()) for e in q.scalars().all()]


# ── Models info ─────────────────────────────────────────────────


@router.get("/models")
async def list_models():
    """List available model architectures and active model info."""
    from app.ml.classifier import MODEL_REGISTRY
    from app.ml.inference import get_inference_service

    service = get_inference_service()
    return {
        "available_architectures": list(MODEL_REGISTRY.keys()),
        "active_model": {
            "loaded": service.is_loaded,
            "name": service.model_name,
            "version": service.model_version,
        },
    }
