"""
Microplastic Vision – model evaluation.

Calculates metrics from an explicitly identified held-out test set.
Never uses training data for final metrics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from app.ml.classifier import create_model, load_checkpoint
from app.ml.preprocessing import (
    PreprocessingConfig,
    get_default_config,
    get_validation_transform,
)
from app.ml.training import ImageFolderDataset

logger = logging.getLogger(__name__)


def evaluate_model(
    checkpoint_path: str,
    architecture: str,
    test_paths: List[str],
    test_labels: List[int],
    class_names: List[str],
    image_size: int = 224,
    batch_size: int = 16,
    preprocessing_config: Optional[PreprocessingConfig] = None,
) -> Dict[str, Any]:
    """
    Evaluate a checkpoint on a held-out test set.

    Returns a dict with all metrics, confusion matrix, and provenance.
    """
    if len(test_paths) == 0:
        raise ValueError("Test set is empty – cannot evaluate.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if preprocessing_config is None:
        preprocessing_config = get_default_config(image_size)

    transform = get_validation_transform(preprocessing_config)
    test_ds = ImageFolderDataset(test_paths, test_labels, class_names, transform)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    num_classes = len(class_names)
    model = create_model(architecture, num_classes=num_classes, pretrained=False)
    load_checkpoint(model, checkpoint_path, device=device)
    model.to(device)
    model.eval()

    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for batch_imgs, batch_labels in test_loader:
            batch_imgs = batch_imgs.to(device)
            outputs = model(batch_imgs)
            preds = outputs.argmax(dim=1).cpu().numpy().tolist()
            all_preds.extend(preds)
            all_labels.extend(batch_labels.numpy().tolist())

    all_preds_np = np.array(all_preds)
    all_labels_np = np.array(all_labels)

    # Metrics
    acc = float(accuracy_score(all_labels_np, all_preds_np))

    prec_per_class = precision_score(
        all_labels_np, all_preds_np, average=None, zero_division=0,
    ).tolist()
    rec_per_class = recall_score(
        all_labels_np, all_preds_np, average=None, zero_division=0,
    ).tolist()
    f1_per_class = f1_score(
        all_labels_np, all_preds_np, average=None, zero_division=0,
    ).tolist()

    prec_macro = float(precision_score(all_labels_np, all_preds_np, average="macro", zero_division=0))
    rec_macro = float(recall_score(all_labels_np, all_preds_np, average="macro", zero_division=0))
    f1_macro = float(f1_score(all_labels_np, all_preds_np, average="macro", zero_division=0))

    cm = confusion_matrix(all_labels_np, all_preds_np, labels=list(range(num_classes)))

    # Class counts in test set
    class_counts = {}
    for i, name in enumerate(class_names):
        class_counts[name] = int((all_labels_np == i).sum())

    return {
        "accuracy": round(acc, 4),
        "precision_per_class": {class_names[i]: round(v, 4) for i, v in enumerate(prec_per_class)},
        "recall_per_class": {class_names[i]: round(v, 4) for i, v in enumerate(rec_per_class)},
        "f1_per_class": {class_names[i]: round(v, 4) for i, v in enumerate(f1_per_class)},
        "precision_macro": round(prec_macro, 4),
        "recall_macro": round(rec_macro, 4),
        "f1_macro": round(f1_macro, 4),
        "confusion_matrix": cm.tolist(),
        "class_order": class_names,
        "test_count": len(test_paths),
        "class_counts": class_counts,
        "checkpoint_path": checkpoint_path,
        "architecture": architecture,
        "preprocessing_config": preprocessing_config.to_dict(),
    }
