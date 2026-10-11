"""
Microplastic Vision – Pydantic request / response schemas.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Health / Status ─────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"


class StatusResponse(BaseModel):
    app_name: str
    demo_mode: bool
    model_loaded: bool
    model_ready: bool = False
    model_scope: Optional[str] = None
    active_model: Optional[str] = None
    yolo_available: bool
    confidence_threshold: float
    total_analyses: int = 0
    total_runs: int = 0


# ── Analysis ────────────────────────────────────────────────────


class AnalysisResponse(BaseModel):
    id: str
    created_at: Optional[str] = None
    original_filename: str
    storage_path: str
    file_type: str
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    predicted_class: Optional[str] = None
    confidence: Optional[float] = None
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    status: str
    error_message: Optional[str] = None
    is_demo: bool = False
    warnings: List[str] = Field(default_factory=list)
    preprocessing_config: Optional[Dict[str, Any]] = None
    detections: Optional[List[Dict[str, Any]]] = None
    detection_count: Optional[int] = None
    confidence_note: str = Field(
        default="Confidence reflects model certainty on visual features, "
                "not the probability that a particle is chemically a polymer."
    )
    screening_disclaimer: str = Field(
        default="This is a preliminary image-based screening result. "
                "Confirmed polymer identification requires FTIR, Raman, "
                "or other chemical/spectroscopic evidence."
    )


class AnalysisListResponse(BaseModel):
    items: List[AnalysisResponse]
    total: int
    page: int
    per_page: int


class BatchAnalysisResponse(BaseModel):
    results: List[AnalysisResponse]
    total_processed: int
    total_errors: int


# ── Dataset ─────────────────────────────────────────────────────


class DatasetResponse(BaseModel):
    id: str
    created_at: Optional[str] = None
    name: str
    description: Optional[str] = None
    source: Optional[str] = None
    license_notes: Optional[str] = None
    base_path: str
    total_images: int
    class_counts: Optional[Dict[str, int]] = None
    split_counts: Optional[Dict[str, int]] = None
    validation_warnings: Optional[List[str]] = None
    label_provenance: Optional[str] = None
    is_valid: bool
    training_ready: Optional[bool] = None


class DatasetValidateRequest(BaseModel):
    base_path: str
    name: Optional[str] = "Unnamed dataset"
    description: Optional[str] = None
    source: Optional[str] = None
    license_notes: Optional[str] = None
    label_provenance: Optional[str] = None


# ── Training Run ────────────────────────────────────────────────


class TrainingRunResponse(BaseModel):
    id: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    run_name: str
    status: str
    model_architecture: str
    dataset_manifest: Optional[str] = None
    dataset_source: Optional[str] = None
    epochs: int
    batch_size: int
    learning_rate: float
    image_size: int
    class_weights: Optional[Dict[str, float]] = None
    augmentation_config: Optional[Dict[str, Any]] = None
    preprocessing_config: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None
    checkpoint_path: Optional[str] = None
    best_val_accuracy: Optional[float] = None
    training_history: Optional[List[Dict[str, Any]]] = None
    class_mapping: Optional[Dict[str, str]] = None
    train_count: Optional[int] = None
    val_count: Optional[int] = None
    test_count: Optional[int] = None
    split_method: Optional[str] = None
    test_metrics: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    progress: float = 0.0


class TrainingRunCreateRequest(BaseModel):
    run_name: str
    model_architecture: str = "mobilenet_v3_small"
    dataset_path: str
    epochs: int = Field(default=10, ge=1, le=100)
    batch_size: int = Field(default=32, ge=1, le=256)
    learning_rate: float = Field(default=0.0001, gt=0, le=0.1)
    image_size: int = 224
    seed: int = 42
    use_class_weights: bool = False


# ── Evaluation ──────────────────────────────────────────────────


class EvaluationResponse(BaseModel):
    id: str
    created_at: Optional[str] = None
    run_id: str
    test_split_name: str
    test_count: Optional[int] = None
    class_counts: Optional[Dict[str, int]] = None
    accuracy: Optional[float] = None
    precision_per_class: Optional[Dict[str, float]] = None
    recall_per_class: Optional[Dict[str, float]] = None
    f1_per_class: Optional[Dict[str, float]] = None
    precision_macro: Optional[float] = None
    recall_macro: Optional[float] = None
    f1_macro: Optional[float] = None
    confusion_matrix: Optional[List[List[int]]] = None
    class_order: Optional[List[str]] = None
    model_checkpoint: Optional[str] = None
    preprocessing_config: Optional[Dict[str, Any]] = None
    dataset_manifest: Optional[str] = None
    status: str
    error_message: Optional[str] = None
    evaluation_note: str = Field(
        default="Metrics are specific to the test data used and may not "
                "generalise to images from other microscopes, environments, "
                "or sample preparation methods."
    )


class EvaluateRequest(BaseModel):
    run_id: str
    test_split_name: str = "test"
