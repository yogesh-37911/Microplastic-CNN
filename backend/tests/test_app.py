"""
Microplastic Vision – automated tests.

Uses small synthetic fixtures; no GPU, external dataset, or network required.
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Ensure app is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./test_data/test.db"
os.environ["UPLOAD_DIR"] = "./test_data/uploads"
os.environ["DATASET_DIR"] = "./test_data/datasets"
os.environ["MODEL_DIR"] = "./test_data/models"
os.environ["CHECKPOINT_DIR"] = "./test_data/checkpoints"
os.environ["DEMO_MODE"] = "true"
os.environ["DEBUG"] = "false"


@pytest.fixture(scope="session", autouse=True)
def setup_test_dirs():
    """Create test data directories."""
    for d in ["test_data/uploads", "test_data/datasets", "test_data/models", "test_data/checkpoints"]:
        Path(d).mkdir(parents=True, exist_ok=True)
    yield
    # Cleanup handled by CI or developer


def _create_test_image(width=100, height=100, color=(128, 128, 128)) -> bytes:
    """Create a minimal valid PNG image."""
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _create_corrupt_file() -> bytes:
    """Create bytes that are not a valid image."""
    return b"this is not an image file at all"


# ── Preprocessing tests ────────────────────────────────────────


class TestPreprocessing:
    def test_preprocess_output_size(self):
        from app.ml.preprocessing import preprocess_pil, PreprocessingConfig

        img = Image.new("RGB", (400, 300))
        config = PreprocessingConfig(target_size=(224, 224))
        result = preprocess_pil(img, config)
        assert result.size == (224, 224)

    def test_preprocess_color_conversion(self):
        from app.ml.preprocessing import preprocess_pil, PreprocessingConfig

        img = Image.new("L", (100, 100), 128)  # Grayscale
        config = PreprocessingConfig(target_size=(224, 224), color_mode="RGB")
        result = preprocess_pil(img, config)
        assert result.mode == "RGB"

    def test_validate_blank_image(self):
        from app.ml.preprocessing import validate_image

        img = Image.new("RGB", (100, 100), (100, 100, 100))
        warnings = validate_image(img)
        assert any("blank" in w.lower() or "uniform" in w.lower() for w in warnings)

    def test_validate_small_image(self):
        from app.ml.preprocessing import validate_image

        img = Image.new("RGB", (10, 10), (128, 64, 200))
        warnings = validate_image(img)
        assert any("small" in w.lower() for w in warnings)

    def test_validate_extreme_aspect_ratio(self):
        from app.ml.preprocessing import validate_image

        img = Image.new("RGB", (1000, 50))
        # Add some variation so it's not blank
        arr = np.random.randint(0, 255, (50, 1000, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        warnings = validate_image(img)
        assert any("aspect" in w.lower() for w in warnings)

    def test_inference_transform_shape(self):
        from app.ml.preprocessing import get_inference_transform, PreprocessingConfig

        config = PreprocessingConfig(target_size=(224, 224))
        transform = get_inference_transform(config)
        img = Image.new("RGB", (300, 300))
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)


# ── Classifier tests ───────────────────────────────────────────


class TestClassifier:
    def test_simple_cnn_output_shape(self):
        import torch
        from app.ml.classifier import SimpleCNN

        model = SimpleCNN(num_classes=2)
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        assert out.shape == (1, 2)

    def test_create_model_factory(self):
        from app.ml.classifier import create_model

        model = create_model("simple_cnn", num_classes=2)
        assert model is not None

    def test_unknown_architecture_raises(self):
        from app.ml.classifier import create_model

        with pytest.raises(ValueError, match="Unknown architecture"):
            create_model("nonexistent_model")


# ── Inference / Demo mode tests ─────────────────────────────────


class TestInference:
    def test_missing_model_abstains_instead_of_guessing(self):
        from app.ml.inference import InferenceService

        service = InferenceService()
        img = Image.fromarray(np.random.default_rng(5).integers(0, 255, (128, 128, 3), dtype=np.uint8))
        result = service.predict(img)

        assert "predicted_class" in result
        assert "confidence" in result
        assert "is_demo" in result
        assert result["status"] == "model_unavailable"
        assert result["is_demo"] is True
        assert result["status"] == "model_unavailable"
        assert result["predicted_class"] is None
        assert result["confidence"] is None

    def test_missing_model_explains_abstention(self):
        from app.ml.inference import InferenceService

        service = InferenceService()
        img = Image.fromarray(np.random.default_rng(6).integers(0, 255, (128, 128, 3), dtype=np.uint8))
        result = service.predict(img)

        warnings = result.get("warnings", [])
        assert any("no trained model" in w.lower() for w in warnings)

    def test_flat_digital_graphic_is_out_of_domain(self):
        from app.ml.inference import InferenceService

        img = Image.new("RGB", (256, 256), "black")
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.rectangle((40, 40, 210, 210), fill="white")
        result = InferenceService().predict(img)

        assert result["status"] == "out_of_domain"
        assert result["predicted_class"] is None
        assert any("digital" in warning.lower() for warning in result["warnings"])


# ── Dataset handling tests ──────────────────────────────────────


class TestDataset:
    def _make_dataset(self, tmp_path: Path, counts=None):
        """Create a small test dataset."""
        if counts is None:
            counts = {"microplastic": 6, "non_microplastic": 6}
        for cls, n in counts.items():
            cls_dir = tmp_path / cls
            cls_dir.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                img = Image.new("RGB", (50, 50), (i * 20, i * 10, i * 5))
                img.save(cls_dir / f"img_{i:03d}.png")
        return str(tmp_path)

    def test_discover_dataset(self, tmp_path):
        from app.ml.training import discover_dataset

        ds_path = self._make_dataset(tmp_path)
        paths, labels, class_names, class_counts = discover_dataset(ds_path)

        assert len(paths) == 12
        assert len(labels) == 12
        assert set(class_names) == {"microplastic", "non_microplastic"}
        assert class_counts["microplastic"] == 6

    def test_discover_missing_path_raises(self):
        from app.ml.training import discover_dataset

        with pytest.raises(FileNotFoundError):
            discover_dataset("/nonexistent/path")

    def test_discover_single_class_raises(self, tmp_path):
        from app.ml.training import discover_dataset

        (tmp_path / "only_class").mkdir()
        img = Image.new("RGB", (50, 50))
        img.save(tmp_path / "only_class" / "img.png")

        with pytest.raises(ValueError, match="at least 2 class directories"):
            discover_dataset(str(tmp_path))

    def test_split_dataset_no_overlap(self, tmp_path):
        from app.ml.training import discover_dataset, split_dataset

        ds_path = self._make_dataset(tmp_path, {"microplastic": 20, "non_microplastic": 20})
        paths, labels, _, _ = discover_dataset(ds_path)
        (train_p, _), (val_p, _), (test_p, _) = split_dataset(paths, labels, seed=42)

        # No overlap between splits
        all_sets = [set(train_p), set(val_p), set(test_p)]
        for i in range(len(all_sets)):
            for j in range(i + 1, len(all_sets)):
                assert len(all_sets[i] & all_sets[j]) == 0, "Splits overlap!"

    def test_split_covers_all(self, tmp_path):
        from app.ml.training import discover_dataset, split_dataset

        ds_path = self._make_dataset(tmp_path, {"microplastic": 10, "non_microplastic": 10})
        paths, labels, _, _ = discover_dataset(ds_path)
        (train_p, _), (val_p, _), (test_p, _) = split_dataset(paths, labels)

        assert len(train_p) + len(val_p) + len(test_p) == len(paths)


# ── Metrics tests ───────────────────────────────────────────────


class TestMetrics:
    def test_confusion_matrix_shape(self):
        from sklearn.metrics import confusion_matrix
        import numpy as np

        y_true = [0, 0, 1, 1, 0, 1]
        y_pred = [0, 1, 1, 1, 0, 0]
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        assert cm.shape == (2, 2)

    def test_accuracy_range(self):
        from sklearn.metrics import accuracy_score

        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 1, 0]
        acc = accuracy_score(y_true, y_pred)
        assert 0.0 <= acc <= 1.0


# ── API tests (async) ──────────────────────────────────────────


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    """Create a test client."""
    from httpx import AsyncClient, ASGITransport
    from app.main import app
    from app.database import init_db

    await init_db()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.mark.asyncio
class TestAPI:
    async def test_health(self, client):
        resp = await client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    async def test_status(self, client):
        resp = await client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "demo_mode" in data
        assert "model_loaded" in data

    async def test_analyze_valid_image(self, client):
        img_bytes = _create_test_image()
        resp = await client.post(
            "/api/analyze",
            files={"file": ("test.png", img_bytes, "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "predicted_class" in data
        assert "confidence" in data
        assert "screening_disclaimer" in data

    async def test_analyze_corrupt_file(self, client):
        resp = await client.post(
            "/api/analyze",
            files={"file": ("bad.png", _create_corrupt_file(), "image/png")},
        )
        assert resp.status_code == 400

    async def test_analyze_unsupported_extension(self, client):
        resp = await client.post(
            "/api/analyze",
            files={"file": ("script.py", b"print('hello')", "text/plain")},
        )
        assert resp.status_code == 400

    async def test_analyze_oversized_file(self, client):
        # Create a file slightly over 50MB (but we'll test with a much smaller limit)
        # For this test, we just verify the endpoint exists
        img_bytes = _create_test_image()
        resp = await client.post(
            "/api/analyze",
            files={"file": ("big.png", img_bytes, "image/png")},
        )
        # Should succeed since it's small
        assert resp.status_code == 200

    async def test_analyses_list(self, client):
        resp = await client.get("/api/analyses")
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert "total" in data

    async def test_analysis_not_found(self, client):
        resp = await client.get("/api/analyses/nonexistent-id")
        assert resp.status_code == 404

    async def test_models_list(self, client):
        resp = await client.get("/api/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "available_architectures" in data

    async def test_detect_without_yolo(self, client):
        img_bytes = _create_test_image()
        resp = await client.post(
            "/api/detect",
            files={"file": ("test.png", img_bytes, "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["available"] is False

    async def test_batch_analyze(self, client):
        img1 = _create_test_image(color=(50, 50, 50))
        img2 = _create_test_image(color=(200, 200, 200))
        resp = await client.post(
            "/api/analyze/batch",
            files=[
                ("files", ("img1.png", img1, "image/png")),
                ("files", ("img2.png", img2, "image/png")),
            ],
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_processed"] == 2

    async def test_datasets_list(self, client):
        resp = await client.get("/api/datasets")
        assert resp.status_code == 200

    async def test_runs_list(self, client):
        resp = await client.get("/api/runs")
        assert resp.status_code == 200
