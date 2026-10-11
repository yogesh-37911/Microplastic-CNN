"""
Microplastic Vision – inference service.

Loads a trained checkpoint and runs classification on preprocessed images.
Handles demo mode when no real model is available.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

from app.config import get_settings
from app.ml.classifier import create_model, load_checkpoint
from app.ml.preprocessing import (
    PreprocessingConfig,
    get_default_config,
    get_inference_transform,
    preprocess_pil,
    validate_image,
    assess_microscopy_domain,
)

logger = logging.getLogger(__name__)

# Default class mapping
DEFAULT_CLASS_MAPPING: Dict[int, str] = {
    0: "microplastic",
    1: "non_microplastic",
}


class InferenceService:
    """Singleton-style service that holds the loaded model."""

    def __init__(self):
        self._model: Optional[torch.nn.Module] = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._class_mapping: Dict[int, str] = dict(DEFAULT_CLASS_MAPPING)
        self._model_name: Optional[str] = None
        self._model_version: Optional[str] = None
        self._model_warnings: List[str] = []
        self._model_requires_validation = False
        self._preprocessing_config: PreprocessingConfig = get_default_config()
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def is_ready(self) -> bool:
        return self._is_loaded and not self._model_requires_validation

    @property
    def model_name(self) -> Optional[str]:
        return self._model_name

    @property
    def model_version(self) -> Optional[str]:
        return self._model_version

    @property
    def preprocessing_config(self) -> PreprocessingConfig:
        return self._preprocessing_config

    def load_model(
        self,
        architecture: str,
        checkpoint_path: str,
        class_mapping: Optional[Dict[int, str]] = None,
        preprocessing_config: Optional[PreprocessingConfig] = None,
        model_version: Optional[str] = None,
    ) -> None:
        """Load a trained model checkpoint."""
        num_classes = len(class_mapping) if class_mapping else 2
        model = create_model(architecture, num_classes=num_classes, pretrained=False)
        meta = load_checkpoint(model, checkpoint_path, device=self._device)

        model.to(self._device)
        model.eval()

        self._model = model
        self._model_name = architecture
        self._model_version = model_version or meta.get("run_id", "unknown")
        self._is_loaded = True
        self._model_warnings = []
        self._model_requires_validation = True
        metrics = meta.get("test_metrics")
        if isinstance(metrics, dict) and isinstance(metrics.get("macro_f1"), (int, float)):
            score = float(metrics["macro_f1"])
            if score >= 0.75:
                self._model_requires_validation = False
            else:
                self._model_warnings.append(
                    f"Held-out macro-F1 is {score:.1%}, below the 75% readiness threshold. "
                    "No class prediction will be issued until the model is validated on stronger microscopy data."
                )
        else:
            self._model_warnings.append(
                "This checkpoint has no complete held-out macro-F1 score. No class prediction will be issued until it is evaluated."
            )
        if not self._model_requires_validation:
            self._model_warnings.append(
                "Validation applies to microfibre crops from microscope filter images only. "
                "Stock photos, illustrations, product scenes, and other particle types have not been validated."
            )

        if class_mapping:
            self._class_mapping = class_mapping
        elif "class_mapping" in meta:
            self._class_mapping = {int(k): v for k, v in meta["class_mapping"].items()}

        if preprocessing_config:
            self._preprocessing_config = preprocessing_config
        elif "preprocessing_config" in meta:
            cfg = meta["preprocessing_config"]
            self._preprocessing_config = PreprocessingConfig(**cfg)

        logger.info(
            "Model loaded: %s (version %s) on %s",
            self._model_name, self._model_version, self._device,
        )

    def predict(self, img: Image.Image) -> Dict:
        """
        Classify a single PIL Image.

        Returns dict with: predicted_class, confidence, class_probabilities,
        warnings, preprocessing_config, model_name, model_version, is_demo.
        """
        settings = get_settings()
        warnings = validate_image(img)

        in_domain, domain_warnings = assess_microscopy_domain(img)
        warnings.extend(domain_warnings)
        warnings.extend(self._model_warnings)
        # Never manufacture a class prediction in the absence of a trained model.
        if not self._is_loaded:
            warnings.append("No trained model is loaded; classification was not performed.")
            return self._abstain("model_unavailable", warnings, is_demo=True)
        if self._model_requires_validation:
            return self._abstain("model_requires_validation", warnings)
        if not in_domain:
            return self._abstain("out_of_domain", warnings)

        transform = get_inference_transform(self._preprocessing_config)
        processed = preprocess_pil(img, self._preprocessing_config)
        tensor = transform(processed).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = self._model(tensor)
            probs = F.softmax(logits, dim=1)[0]

        class_idx = int(probs.argmax())
        raw_confidence = float(probs[class_idx])
        predicted_class = self._class_mapping.get(class_idx, f"class_{class_idx}")

        confidence = round(raw_confidence, 4)

        # Add caution for low confidence
        threshold = settings.confidence_threshold
        status = "completed"
        if confidence < threshold:
            warnings.append(
                f"Confidence ({confidence:.2%}) is below the configured threshold "
                f"({threshold:.2%}); no class label was returned. Review recommended."
            )
            predicted_class = None
            status = "uncertain"

        class_probs = {
            self._class_mapping.get(i, f"class_{i}"): round(float(probs[i]), 4)
            for i in range(len(probs))
        }
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "class_probabilities": class_probs,
            "warnings": warnings,
            "preprocessing_config": self._preprocessing_config.to_dict(),
            "model_name": self._model_name,
            "model_version": self._model_version,
            "is_demo": False,
            "status": status,
        }

    def _abstain(self, status: str, warnings: List[str], is_demo: bool = False) -> Dict:
        return {
            "predicted_class": None,
            "confidence": None,
            "class_probabilities": {},
            "warnings": warnings,
            "preprocessing_config": self._preprocessing_config.to_dict(),
            "model_name": self._model_name,
            "model_version": self._model_version,
            "is_demo": is_demo,
            "status": status,
        }


# Module-level singleton
_inference_service: Optional[InferenceService] = None


def get_inference_service() -> InferenceService:
    global _inference_service
    if _inference_service is None:
        _inference_service = InferenceService()
        # Try auto-loading if a checkpoint exists
        _try_autoload(_inference_service)
    return _inference_service


def _try_autoload(service: InferenceService) -> None:
    """Attempt to load the best checkpoint from the configured directory."""
    settings = get_settings()
    ckpt_dir = Path(settings.checkpoint_dir)
    if not ckpt_dir.exists():
        return
    checkpoints = sorted(ckpt_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not checkpoints:
        return

    # Only automatically activate a checkpoint if it carries provenance from
    # the standard training pipeline. Legacy generated checkpoints can still
    # be explicitly loaded by an operator after review.
    for candidate in checkpoints:
        try:
            meta = torch.load(candidate, map_location="cpu", weights_only=False)
            metrics = meta.get("test_metrics")
            if (
                not meta.get("dataset_manifest")
                or not isinstance(metrics, dict)
                or not isinstance(metrics.get("accuracy"), (int, float))
                or not meta.get("split_groups")
                or meta.get("split_method") != "source-grouped 70/15/15"
            ):
                logger.warning("Ignoring checkpoint without training/test provenance: %s", candidate)
                continue
            arch = meta.get("architecture") or ("resnet18" if "resnet" in candidate.stem.lower() else "simple_cnn")
            service.load_model(architecture=arch, checkpoint_path=str(candidate))
            logger.info("Auto-loaded provenance-checked checkpoint: %s (%s)", candidate, arch)
            return
        except Exception as e:
            logger.warning("Could not load checkpoint %s: %s", candidate, e)
