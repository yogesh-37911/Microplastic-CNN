"""
Microplastic Vision – SQLAlchemy ORM models.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text, JSON,
    ForeignKey, Enum as SAEnum,
)
from sqlalchemy.orm import relationship

from app.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── Analysis ────────────────────────────────────────────────────


class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(String, primary_key=True, default=_uuid)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    # Image metadata
    original_filename = Column(String, nullable=False)
    storage_path = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    image_width = Column(Integer, nullable=True)
    image_height = Column(Integer, nullable=True)

    # Prediction
    predicted_class = Column(String, nullable=True)
    confidence = Column(Float, nullable=True)
    model_name = Column(String, nullable=True)
    model_version = Column(String, nullable=True)

    # Status
    status = Column(String, default="pending", nullable=False)  # pending, completed, error, demo
    error_message = Column(Text, nullable=True)
    is_demo = Column(Boolean, default=False, nullable=False)
    warnings = Column(JSON, nullable=True)

    # Preprocessing config used
    preprocessing_config = Column(JSON, nullable=True)

    # Optional detections (YOLO)
    detections = Column(JSON, nullable=True)  # list of {box, class, confidence}
    detection_count = Column(Integer, nullable=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "original_filename": self.original_filename,
            "storage_path": self.storage_path,
            "file_type": self.file_type,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "status": self.status,
            "error_message": self.error_message,
            "is_demo": self.is_demo,
            "warnings": self.warnings or [],
            "preprocessing_config": self.preprocessing_config,
            "detections": self.detections,
            "detection_count": self.detection_count,
        }


# ── Training Run ────────────────────────────────────────────────


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id = Column(String, primary_key=True, default=_uuid)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False)

    # Run metadata
    run_name = Column(String, nullable=False)
    status = Column(String, default="pending", nullable=False)  # pending, training, completed, failed
    model_architecture = Column(String, nullable=False)  # e.g. "resnet18", "simple_cnn"
    dataset_manifest = Column(String, nullable=True)
    dataset_source = Column(String, nullable=True)

    # Hyperparameters
    epochs = Column(Integer, nullable=False, default=25)
    batch_size = Column(Integer, nullable=False, default=16)
    learning_rate = Column(Float, nullable=False, default=0.001)
    image_size = Column(Integer, nullable=False, default=224)
    class_weights = Column(JSON, nullable=True)
    augmentation_config = Column(JSON, nullable=True)
    preprocessing_config = Column(JSON, nullable=True)
    seed = Column(Integer, nullable=True, default=42)

    # Results
    checkpoint_path = Column(String, nullable=True)
    best_val_accuracy = Column(Float, nullable=True)
    training_history = Column(JSON, nullable=True)  # list of {epoch, train_loss, val_loss, val_acc}
    class_mapping = Column(JSON, nullable=True)  # {0: "microplastic", 1: "non_microplastic"}

    # Split info
    train_count = Column(Integer, nullable=True)
    val_count = Column(Integer, nullable=True)
    test_count = Column(Integer, nullable=True)
    split_method = Column(String, nullable=True)
    test_metrics = Column(JSON, nullable=True)

    error_message = Column(Text, nullable=True)
    progress = Column(Float, default=0.0)  # 0-100

    # Evaluation relationship
    evaluations = relationship("Evaluation", back_populates="run", cascade="all, delete-orphan")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "run_name": self.run_name,
            "status": self.status,
            "model_architecture": self.model_architecture,
            "dataset_manifest": self.dataset_manifest,
            "dataset_source": self.dataset_source,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "image_size": self.image_size,
            "class_weights": self.class_weights,
            "augmentation_config": self.augmentation_config,
            "preprocessing_config": self.preprocessing_config,
            "seed": self.seed,
            "checkpoint_path": self.checkpoint_path,
            "best_val_accuracy": self.best_val_accuracy,
            "training_history": self.training_history,
            "class_mapping": self.class_mapping,
            "train_count": self.train_count,
            "val_count": self.val_count,
            "test_count": self.test_count,
            "split_method": self.split_method,
            "test_metrics": self.test_metrics,
            "error_message": self.error_message,
            "progress": self.progress,
        }


# ── Evaluation ──────────────────────────────────────────────────


class Evaluation(Base):
    __tablename__ = "evaluations"

    id = Column(String, primary_key=True, default=_uuid)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    run_id = Column(String, ForeignKey("training_runs.id"), nullable=False)
    run = relationship("TrainingRun", back_populates="evaluations")

    # Test-set info
    test_split_name = Column(String, nullable=False, default="test")
    test_count = Column(Integer, nullable=True)
    class_counts = Column(JSON, nullable=True)  # {class_name: count}

    # Metrics
    accuracy = Column(Float, nullable=True)
    precision_per_class = Column(JSON, nullable=True)
    recall_per_class = Column(JSON, nullable=True)
    f1_per_class = Column(JSON, nullable=True)
    precision_macro = Column(Float, nullable=True)
    recall_macro = Column(Float, nullable=True)
    f1_macro = Column(Float, nullable=True)
    confusion_matrix = Column(JSON, nullable=True)  # [[TP, FP], [FN, TN]]
    class_order = Column(JSON, nullable=True)  # [class0, class1, ...]

    # Provenance
    model_checkpoint = Column(String, nullable=True)
    preprocessing_config = Column(JSON, nullable=True)
    dataset_manifest = Column(String, nullable=True)

    status = Column(String, default="pending")  # pending, completed, error
    error_message = Column(Text, nullable=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "run_id": self.run_id,
            "test_split_name": self.test_split_name,
            "test_count": self.test_count,
            "class_counts": self.class_counts,
            "accuracy": self.accuracy,
            "precision_per_class": self.precision_per_class,
            "recall_per_class": self.recall_per_class,
            "f1_per_class": self.f1_per_class,
            "precision_macro": self.precision_macro,
            "recall_macro": self.recall_macro,
            "f1_macro": self.f1_macro,
            "confusion_matrix": self.confusion_matrix,
            "class_order": self.class_order,
            "model_checkpoint": self.model_checkpoint,
            "preprocessing_config": self.preprocessing_config,
            "dataset_manifest": self.dataset_manifest,
            "status": self.status,
            "error_message": self.error_message,
        }


# ── Dataset Manifest ────────────────────────────────────────────


class DatasetManifest(Base):
    __tablename__ = "dataset_manifests"

    id = Column(String, primary_key=True, default=_uuid)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    source = Column(String, nullable=True)
    license_notes = Column(Text, nullable=True)
    base_path = Column(String, nullable=False)

    total_images = Column(Integer, default=0)
    class_counts = Column(JSON, nullable=True)  # {class_name: count}
    split_counts = Column(JSON, nullable=True)  # {split_name: count}
    validation_warnings = Column(JSON, nullable=True)
    label_provenance = Column(String, nullable=True)

    is_valid = Column(Boolean, default=False)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "name": self.name,
            "description": self.description,
            "source": self.source,
            "license_notes": self.license_notes,
            "base_path": self.base_path,
            "total_images": self.total_images,
            "class_counts": self.class_counts,
            "split_counts": self.split_counts,
            "validation_warnings": self.validation_warnings,
            "label_provenance": self.label_provenance,
            "is_valid": self.is_valid,
        }
