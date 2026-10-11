"""
Microplastic Vision – training pipeline.

Trains a CNN classifier with configurable architecture, augmentation,
and class-weight handling.  Saves checkpoints with full provenance metadata.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import cv2
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from torchvision import transforms

from app.config import get_settings
from app.ml.classifier import create_model
from app.ml.preprocessing import (
    PreprocessingConfig,
    get_default_config,
    get_training_transform,
    get_validation_transform,
)

logger = logging.getLogger(__name__)


# ── Dataset ─────────────────────────────────────────────────────


class ImageFolderDataset(Dataset):
    """
    Loads images from a directory structure:
        base_path/
            microplastic/
            non_microplastic/
    """

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        class_names: List[str],
        transform: Optional[transforms.Compose] = None,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.class_names = class_names
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        raw = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if raw is None:
            raise ValueError(f"OpenCV could not decode image: {img_path}")
        img = Image.fromarray(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
        if self.transform:
            img = self.transform(img)
        return img, label


def discover_dataset(base_path: str) -> Tuple[List[str], List[int], List[str], Dict[str, int]]:
    """
    Discover images from a folder-per-class layout.

    Returns (paths, labels, class_names, class_counts).
    """
    base = Path(base_path)
    if not base.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {base_path}")

    allowed_ext = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}
    class_dirs = sorted([d for d in base.iterdir() if d.is_dir()])

    if len(class_dirs) < 2:
        raise ValueError(
            f"Expected at least 2 class directories in {base_path}, "
            f"found {len(class_dirs)}: {[d.name for d in class_dirs]}"
        )

    class_names = [d.name for d in class_dirs]
    paths: List[str] = []
    labels: List[int] = []
    class_counts: Dict[str, int] = {}

    for class_idx, class_dir in enumerate(class_dirs):
        count = 0
        for img_file in sorted(class_dir.iterdir()):
            if img_file.suffix.lower() in allowed_ext and img_file.is_file():
                paths.append(str(img_file))
                labels.append(class_idx)
                count += 1
        class_counts[class_dir.name] = count

    if len(paths) == 0:
        raise ValueError(f"No valid images found in {base_path}")

    return paths, labels, class_names, class_counts


def _source_group(path: str) -> str:
    """Infer a source/acquisition group from known dataset filename patterns."""
    name = Path(path).name
    patterns = (
        (r"^mfca_(?:pos|neg)_(\d+)", "mfca_source"),
        (r"^(hf_wwtp_mp|hf_wwtp_substrate)_(\d+)", "wwtp"),
        (r"^(beni_mp|beni_nmp)_(\d+)", "beni"),
        (r"^(hf_IMG_\d+)", "hf_image"),
        (r"^([a-z]+_[0-9a-f]{8})_\d+", "derived_sample"),
        (r"^(mp_sample|nmp_sample|mp_synthetic|nmp_synthetic|neg_macro_surface|neg_poster_splatter|neg_sketch_portrait)_(\d+)", "synthetic_id"),
        (r"^(real_(?:illustration_nmp|macro_mp|nmp_user_ood)_[0-9a-f]{8})_\d+", "derived_capture"),
        (r"^(real_whatsapp_microplastic_\d{8})_\d+", "derived_capture"),
    )
    import re
    for pattern, prefix in patterns:
        match = re.match(pattern, name, flags=re.IGNORECASE)
        if match:
            if prefix in {"wwtp", "beni"}:
                return f"{prefix}:{match.group(2)}"
            if prefix == "derived_sample":
                return f"derived:{match.group(1).lower()}"
            if prefix == "synthetic_id":
                return f"synthetic:{match.group(1).lower()}:{match.group(2)}"
            if prefix == "derived_capture":
                return f"capture:{match.group(1).lower()}"
            return f"{prefix}:{match.group(1).lower()}"
    # Generic source identifiers stay together even when file suffixes differ.
    return f"file:{Path(path).stem}"


def split_dataset(
    paths: List[str],
    labels: List[int],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[
    Tuple[List[str], List[int]],
    Tuple[List[str], List[int]],
    Tuple[List[str], List[int]],
]:
    """
    Stratified acquisition-grouped train/validation/test split.
    Returns ((train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)).
    """
    if len(paths) != len(labels):
        raise ValueError("Paths and labels must have the same length.")
    counts = np.bincount(np.asarray(labels, dtype=int))
    if len(counts) < 2 or np.any(counts < 3):
        raise ValueError("Each class needs at least 3 labeled images for train/validation/test splits.")
    indices = np.arange(len(paths))
    groups = np.asarray([_source_group(path) for path in paths])
    if len(set(groups)) < 5:
        raise ValueError("At least five independent source/acquisition groups are required for leakage-resistant splits.")
    test_ratio = 1.0 - train_ratio - val_ratio
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1 or not 0 < test_ratio < 1:
        raise ValueError("Train, validation, and test ratios must each be positive and sum to 1.")
    from sklearn.model_selection import GroupShuffleSplit
    label_arr = np.asarray(labels)

    def grouped_split(pool: np.ndarray, fraction: float, split_seed: int):
        pool_groups = groups[pool]
        for offset in range(200):
            splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=split_seed + offset)
            left_rel, right_rel = next(splitter.split(pool, label_arr[pool], pool_groups))
            left, right = pool[left_rel], pool[right_rel]
            if set(label_arr[left]) == set(np.unique(label_arr)) and set(label_arr[right]) == set(np.unique(label_arr)):
                return left, right
        raise ValueError("Could not create class-balanced splits while keeping source groups together. Add independent acquisitions for each class.")

    train_idx, remainder_idx = grouped_split(indices, 1.0 - train_ratio, seed)
    val_idx, test_idx = grouped_split(remainder_idx, test_ratio / (val_ratio + test_ratio), seed + 100)
    if min(len(train_idx), len(val_idx), len(test_idx)) == 0:
        raise ValueError("Each train, validation, and test split must contain images.")

    def gather(idx):
        return [paths[i] for i in idx], [labels[i] for i in idx]

    return gather(train_idx), gather(val_idx), gather(test_idx)


# ── Training ────────────────────────────────────────────────────


def train_model(
    dataset_path: str,
    architecture: str = "mobilenet_v3_small",
    epochs: int = 25,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    image_size: int = 224,
    seed: int = 42,
    use_class_weights: bool = False,
    freeze_backbone: bool = False,
    checkpoint_dir: Optional[str] = None,
    progress_callback: Optional[callable] = None,
) -> Dict[str, Any]:
    """
    Full training run.  Returns a metadata dict with paths, metrics, and history.
    """
    settings = get_settings()
    if checkpoint_dir is None:
        checkpoint_dir = settings.checkpoint_dir

    run_id = str(uuid.uuid4())[:8]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    # Seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Discover and split dataset
    paths, labels, class_names, class_counts = discover_dataset(dataset_path)
    required_classes = {"microplastic", "non_microplastic"}
    if set(class_names) != required_classes:
        raise ValueError("Dataset must contain exactly 'microplastic' and 'non_microplastic' class folders.")
    if (
        class_counts.get("microplastic", 0) < 50
        or class_counts.get("non_microplastic", 0) < 50
    ):
        raise ValueError(
            "At least 50 images per class are required to begin training; "
            f"found {class_counts}. Use diverse, microscopy-relevant, correctly labeled images."
        )
    for image_path in paths:
        try:
            with Image.open(image_path) as candidate:
                candidate.verify()
        except Exception as exc:
            raise ValueError(f"Unreadable training image '{image_path}': {exc}") from exc
    (train_p, train_l), (val_p, val_l), (test_p, test_l) = split_dataset(
        paths, labels, seed=seed,
    )
    train_groups = {_source_group(path) for path in train_p}
    val_groups = {_source_group(path) for path in val_p}
    test_groups = {_source_group(path) for path in test_p}
    if train_groups & val_groups or train_groups & test_groups or val_groups & test_groups:
        raise RuntimeError("Source-group leakage detected between dataset splits.")

    logger.info(
        "Dataset: %d total (%d train, %d val, %d test) – classes: %s",
        len(paths), len(train_p), len(val_p), len(test_p), class_names,
    )

    # Preprocessing config
    pp_config = get_default_config(image_size)
    train_transform = get_training_transform(pp_config)
    val_transform = get_validation_transform(pp_config)

    train_ds = ImageFolderDataset(train_p, train_l, class_names, train_transform)
    val_ds = ImageFolderDataset(val_p, val_l, class_names, val_transform)

    # Optional class-weight sampling
    sampler = None
    if use_class_weights and len(train_l) > 0:
        label_arr = np.array(train_l)
        counts = np.bincount(label_arr, minlength=len(class_names))
        weights = 1.0 / (counts + 1e-6)
        sample_weights = weights[label_arr]
        sampler = WeightedRandomSampler(
            torch.DoubleTensor(sample_weights), len(sample_weights),
        )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=(sampler is None), sampler=sampler, num_workers=0,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    num_classes = len(class_names)
    model = create_model(architecture, num_classes=num_classes)
    if freeze_backbone and architecture != "simple_cnn":
        for parameter in model.parameters():
            parameter.requires_grad = False
        head = getattr(model, "fc", None)
        if head is None:
            head = getattr(model, "classifier", None)
        if head is None:
            raise ValueError(f"Cannot identify classifier head for frozen-backbone training: {architecture}")
        for parameter in head.parameters():
            parameter.requires_grad = True
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    # Training loop
    history: List[Dict[str, Any]] = []
    best_val_score = -1.0
    best_val_acc = 0.0
    best_ckpt_path = ""

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_imgs, batch_labels in train_loader:
            batch_imgs = batch_imgs.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_imgs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_imgs.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == batch_labels).sum().item()
            train_total += batch_labels.size(0)

        train_loss /= max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds: List[int] = []
        val_targets: List[int] = []

        with torch.no_grad():
            for batch_imgs, batch_labels in val_loader:
                batch_imgs = batch_imgs.to(device)
                batch_labels = batch_labels.to(device)
                outputs = model(batch_imgs)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item() * batch_imgs.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == batch_labels).sum().item()
                val_total += batch_labels.size(0)
                val_preds.extend(preds.cpu().tolist())
                val_targets.extend(batch_labels.cpu().tolist())

        val_loss /= max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)
        from sklearn.metrics import f1_score
        val_macro_f1 = float(f1_score(val_targets, val_preds, average="macro", zero_division=0))

        epoch_data = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4),
            "val_macro_f1": round(val_macro_f1, 4),
        }
        history.append(epoch_data)

        logger.info(
            "Epoch %d/%d – train_loss: %.4f  train_acc: %.4f  val_loss: %.4f  val_acc: %.4f",
            epoch, epochs, train_loss, train_acc, val_loss, val_acc,
        )

        # Save best checkpoint
        if val_macro_f1 > best_val_score:
            best_val_score = val_macro_f1
            best_val_acc = val_acc
            ckpt_name = f"{architecture}_{run_id}_best.pth"
            best_ckpt_path = str(Path(checkpoint_dir) / ckpt_name)
            torch.save({
                "model_state_dict": model.state_dict(),
                "architecture": architecture,
                "freeze_backbone": freeze_backbone,
                "num_classes": num_classes,
                "class_mapping": {i: n for i, n in enumerate(class_names)},
                "class_names": class_names,
                "epoch": epoch,
                "val_accuracy": val_acc,
                "val_macro_f1": val_macro_f1,
                "run_id": run_id,
                "preprocessing_config": pp_config.to_dict(),
                "image_size": image_size,
                "dataset_manifest": str(Path(dataset_path).resolve()),
                "test_metrics": {"status": "pending", "count": len(test_p)},
            }, best_ckpt_path)

        if progress_callback:
            progress_callback(epoch / epochs * 100)

    # Restore the validation-selected model before measuring the held-out test set.
    best_checkpoint = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.eval()
    test_dataset = ImageFolderDataset(test_p, test_l, class_names, val_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_targets: List[int] = []
    test_preds: List[int] = []
    with torch.no_grad():
        for batch_imgs, batch_labels in test_loader:
            outputs = model(batch_imgs.to(device))
            test_preds.extend(outputs.argmax(dim=1).cpu().tolist())
            test_targets.extend(batch_labels.tolist())
    test_metrics = {
        "accuracy": float(accuracy_score(test_targets, test_preds)),
        "macro_f1": float(f1_score(test_targets, test_preds, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(test_targets, test_preds, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(test_targets, test_preds, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(test_targets, test_preds, labels=list(range(num_classes))).tolist(),
        "count": len(test_targets),
    }
    final_name = f"{architecture}_{run_id}_final.pth"
    final_path = str(Path(checkpoint_dir) / final_name)
    torch.save({
        "model_state_dict": model.state_dict(),
        "architecture": architecture,
        "freeze_backbone": freeze_backbone,
        "num_classes": num_classes,
        "class_mapping": {i: n for i, n in enumerate(class_names)},
        "class_names": class_names,
        "epoch": epochs,
        "val_accuracy": val_acc if val_total > 0 else None,
        "run_id": run_id,
        "preprocessing_config": pp_config.to_dict(),
        "image_size": image_size,
        "dataset_manifest": str(Path(dataset_path).resolve()),
        "test_metrics": test_metrics,
        "split_method": "source-grouped 70/15/15",
        "split_groups": {
            "train": sorted(train_groups),
            "validation": sorted(val_groups),
            "test": sorted(test_groups),
        },
    }, final_path)

    return {
        "run_id": run_id,
        "architecture": architecture,
        "class_names": class_names,
        "class_counts": class_counts,
        "class_mapping": {i: n for i, n in enumerate(class_names)},
        "train_count": len(train_p),
        "val_count": len(val_p),
        "test_count": len(test_p),
        "test_labels": test_l,
        "test_paths": test_p,
        "test_metrics": test_metrics,
        "split_method": "source-grouped 70/15/15",
        "split_groups": {
            "train": sorted(train_groups),
            "validation": sorted(val_groups),
            "test": sorted(test_groups),
        },
        "epochs": epochs,
        "best_val_accuracy": round(best_val_acc, 4),
        "best_checkpoint_path": best_ckpt_path,
        "final_checkpoint_path": final_path,
        "training_history": history,
        "preprocessing_config": pp_config.to_dict(),
        "image_size": image_size,
        "seed": seed,
        "device": str(device),
    }
