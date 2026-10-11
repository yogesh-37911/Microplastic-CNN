"""
Microplastic Vision – image preprocessing pipeline.

Every operation is recorded so the exact preprocessing can be reproduced
during evaluation and inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple

import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms


@dataclass
class PreprocessingConfig:
    """Immutable record of all preprocessing settings."""
    target_size: Tuple[int, int] = (224, 224)
    color_mode: str = "RGB"
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    apply_noise_reduction: bool = False
    noise_reduction_radius: int = 1
    apply_contrast_enhancement: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def get_default_config(image_size: int = 224) -> PreprocessingConfig:
    return PreprocessingConfig(target_size=(image_size, image_size))


def validate_image(img: Image.Image) -> list[str]:
    """Return a list of warnings; empty means OK."""
    warnings = []
    w, h = img.size
    if w < 16 or h < 16:
        warnings.append(f"Image very small ({w}x{h}); results may be unreliable.")
    if w == 0 or h == 0:
        warnings.append("Image has zero dimension.")
    ratio = max(w, h) / max(min(w, h), 1)
    if ratio > 10:
        warnings.append(f"Extreme aspect ratio ({ratio:.1f}:1); image may be distorted after resize.")
    # Check if image is essentially blank
    arr = np.array(img.convert("L"))
    if arr.std() < 2.0:
        warnings.append("Image appears nearly blank or uniform.")
    return warnings


def assess_microscopy_domain(img: Image.Image) -> tuple[bool, list[str]]:
    """Flag low-quality inputs and report visual hints; heuristics cannot prove provenance."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    original_width, original_height = img.size
    arr = np.asarray(img.resize((224, 224), Image.Resampling.LANCZOS), dtype=np.float32)
    gray = arr.mean(axis=2)
    warnings: list[str] = []
    if min(original_width, original_height) < 64:
        warnings.append("Resolution is too low for microscopy screening (minimum 64 px on each side).")
    if float(gray.std()) < 2.0:
        warnings.append("Image is nearly uniform or blank.")

    # JPEG/PNG artwork often loses exact flat pixels during compression, so use
    # local gradients as well as quantization. Thin, strong edges separated by
    # smooth color fields are common in cartoons and poster graphics.
    flat_fraction = float(np.mean(np.isclose(gray, np.rint(gray), atol=1e-6)))
    gray_std = float(gray.std())
    # Check overlapping, aspect-preserving tiles as well as the whole image so
    # a poster with letterboxed margins cannot hide its illustrated content.
    scale = min(1.0, 896 / max(original_width, original_height)) if min(original_width, original_height) >= 224 else 224 / min(original_width, original_height)
    resized_size = (max(224, round(original_width * scale)), max(224, round(original_height * scale)))
    tiled = img.resize(resized_size, Image.Resampling.LANCZOS)

    def starts(length: int) -> list[int]:
        positions = list(range(0, max(length - 224, 0) + 1, 112))
        positions.append(max(length - 224, 0))
        return sorted(set(positions))

    vector_like = False
    for top in starts(tiled.height):
        for left in starts(tiled.width):
            tile = np.asarray(tiled.crop((left, top, left + 224, top + 224)), dtype=np.float32)
            tile_gray = tile.mean(axis=2)
            gx = np.abs(np.diff(tile_gray, axis=1, prepend=tile_gray[:, :1]))
            gy = np.abs(np.diff(tile_gray, axis=0, prepend=tile_gray[:1, :]))
            tile_gradient = gx + gy
            medium_edge_fraction = float(np.mean(tile_gradient > 16))
            strong_edge_fraction = float(np.mean(tile_gradient > 32))
            if (
                float(tile_gray.std()) > 18
                and medium_edge_fraction < 0.32
                and strong_edge_fraction > 0.12
            ):
                vector_like = True
                break
        if vector_like:
            break
    if (gray_std > 8 and flat_fraction > 0.995) or vector_like:
        warnings.append(
            "The image style differs from the labeled microscope filter crops used for training. "
            "The classifier cannot determine whether this image contains plastic."
        )

    # Darkfield optical images can be dark, so brightness alone is never used
    # as a rejection criterion.
    # This visual cue cannot establish that an image is not microscopy or lacks
    # plastic. It only marks image styles outside this model's labeled scope.
    out_of_domain = any(
        "too low" in warning
        or "uniform or blank" in warning
        or "image style differs" in warning
        for warning in warnings
    )
    return not out_of_domain, warnings


def preprocess_pil(
    img: Image.Image,
    config: Optional[PreprocessingConfig] = None,
) -> Image.Image:
    """Apply non-destructive preprocessing to a PIL Image; return a new Image."""
    if config is None:
        config = get_default_config()

    # Use the same explicit OpenCV BGR/RGB path for training and inference.
    img = ImageOps.exif_transpose(img).convert("RGB")
    bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), mode="RGB")

    # Optional noise reduction
    if config.apply_noise_reduction:
        img = img.filter(ImageFilter.MedianFilter(size=config.noise_reduction_radius * 2 + 1))

    # Optional contrast enhancement
    if config.apply_contrast_enhancement:
        img = ImageOps.autocontrast(img)

    # Resize
    img = img.resize(config.target_size, Image.LANCZOS)

    return img


def get_inference_transform(config: Optional[PreprocessingConfig] = None) -> transforms.Compose:
    """Return a torchvision transform chain for model inference."""
    if config is None:
        config = get_default_config()
    return transforms.Compose([
        transforms.Resize(config.target_size, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(config.normalize_mean), std=list(config.normalize_std)),
    ])


def get_training_transform(config: Optional[PreprocessingConfig] = None) -> transforms.Compose:
    """Training transform with data augmentation."""
    if config is None:
        config = get_default_config()
    return transforms.Compose([
        transforms.Resize(config.target_size, antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=180),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(config.normalize_mean), std=list(config.normalize_std)),
    ])


def get_validation_transform(config: Optional[PreprocessingConfig] = None) -> transforms.Compose:
    """Validation/test transform – same as inference (no augmentation)."""
    return get_inference_transform(config)
