"""
Microplastic Vision – optional YOLO detection module.

The application runs without this module.  When Ultralytics is installed
and a trained detector weight file is configured, detection endpoints
become available.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from app.config import get_settings

logger = logging.getLogger(__name__)


def is_yolo_available() -> bool:
    """Check whether Ultralytics is installed and weights are configured."""
    settings = get_settings()
    if not settings.yolo_enabled:
        return False
    if not settings.yolo_weights_path or not Path(settings.yolo_weights_path).exists():
        return False
    try:
        import ultralytics  # noqa: F401
        return True
    except ImportError:
        logger.info("Ultralytics not installed – YOLO detection unavailable.")
        return False


def detect(img: Image.Image) -> Dict[str, Any]:
    """
    Run YOLO detection on a PIL image.

    Returns a dict with detections (list of {box, class, confidence}),
    count, and model metadata.
    """
    if not is_yolo_available():
        return {
            "available": False,
            "message": "YOLO detection is not configured. Install ultralytics "
                       "and provide a trained detector weight file.",
            "detections": [],
            "count": 0,
        }

    settings = get_settings()

    try:
        from ultralytics import YOLO

        model = YOLO(settings.yolo_weights_path)
        results = model.predict(img, verbose=False)

        detections: List[Dict[str, Any]] = []
        for result in results:
            for box in result.boxes:
                det = {
                    "box": box.xyxy[0].tolist(),
                    "confidence": round(float(box.conf[0]), 4),
                    "class_id": int(box.cls[0]),
                    "class_name": result.names[int(box.cls[0])] if result.names else None,
                }
                detections.append(det)

        return {
            "available": True,
            "detections": detections,
            "count": len(detections),
            "model_weights": settings.yolo_weights_path,
        }

    except Exception as e:
        logger.error("YOLO detection error: %s", e)
        return {
            "available": True,
            "detections": [],
            "count": 0,
            "error": str(e),
        }
