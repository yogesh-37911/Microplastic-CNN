"""
Microplastic Vision – Production-Grade Transfer Learning & Dataset Generator.

Trains a robust ResNet-18 classifier using:
1. Real ImageNet pretrained features (general vision representations).
2. Diverse synthetic microplastic morphologies (fibers, fragments, beads, films).
3. Comprehensive natural and macro hard negatives (sand, plant matter, fabrics, everyday textures).
4. Label smoothing and weight decay to prevent logit explosion and overconfidence.
5. Calibrated confidence output with strict screening disclaimers.
"""

from __future__ import annotations

import math
import os
import random
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import torchvision.models as models

# Ensure backend root is in sys.path
sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from app.config import get_settings
from app.ml.preprocessing import get_default_config, get_training_transform, get_validation_transform
from app.ml.training import ImageFolderDataset, split_dataset


# ════════════════════════════════════════════════════════════════════
# 1. DATASET GENERATOR WITH HARD NEGATIVES
# ════════════════════════════════════════════════════════════════════

def create_microscope_background(size: int, rng: np.random.RandomState, is_darkfield: bool = False) -> Image.Image:
    """Generate realistic microscope field background with vignetting and sensor noise."""
    if is_darkfield:
        base_color = rng.randint(18, 38)
        arr = np.full((size, size, 3), [base_color, base_color + 4, base_color + 8], dtype=np.float32)
    else:
        base_color = rng.randint(215, 238)
        tint = rng.randint(-5, 5, 3)
        arr = np.full((size, size, 3), [base_color + tint[0], base_color + tint[1], base_color + tint[2]], dtype=np.float32)

    # Vignetting (falloff towards lens corners)
    y, x = np.ogrid[:size, :size]
    center = size / 2.0
    dist = np.sqrt((x - center) ** 2 + (y - center) ** 2) / (size * 0.707)
    vignette = 1.0 - 0.15 * (dist ** 2)
    arr *= vignette[:, :, np.newaxis]

    # Sensor noise
    noise = rng.normal(0, 3.5, (size, size, 3))
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def generate_microplastic_sample(size: int, seed: int) -> Image.Image:
    """Generate synthetic microplastic particle (fiber, fragment, bead, film)."""
    rng = np.random.RandomState(seed)
    is_darkfield = rng.choice([True, False], p=[0.35, 0.65])
    img = create_microscope_background(size, rng, is_darkfield)
    draw = ImageDraw.Draw(img, "RGBA")

    mp_type = rng.choice(["fiber", "fragment", "bead", "film"])

    plastic_palettes = [
        (30, 110, 235),   # Royal Blue
        (225, 35, 45),    # Bright Red
        (20, 175, 95),    # Vibrant Green
        (235, 175, 25),   # Bright Yellow
        (35, 35, 40),     # Black / Dark Grey
        (170, 70, 220),   # Purple
        (240, 110, 180),  # Bright Pink
        (210, 230, 245),  # Translucent clear PE
    ]
    color_rgb = plastic_palettes[rng.randint(0, len(plastic_palettes))]

    if mp_type == "fiber":
        num_points = rng.randint(6, 12)
        cx, cy = rng.randint(40, size - 40), rng.randint(40, size - 40)
        thickness = rng.randint(3, 7)
        points = [(cx, cy)]
        angle = rng.uniform(0, 2 * math.pi)

        for _ in range(num_points):
            angle += rng.uniform(-0.6, 0.6)
            step = rng.uniform(12, 24)
            nx = points[-1][0] + step * math.cos(angle)
            ny = points[-1][1] + step * math.sin(angle)
            nx = max(10, min(size - 10, nx))
            ny = max(10, min(size - 10, ny))
            points.append((nx, ny))

        draw.line(points, fill=color_rgb + (230,), width=thickness, joint="curve")
        highlight_color = tuple(min(255, c + 90) for c in color_rgb)
        draw.line(points, fill=highlight_color + (140,), width=max(1, thickness // 2), joint="curve")

    elif mp_type == "fragment":
        cx, cy = rng.randint(50, size - 50), rng.randint(50, size - 50)
        n_vertices = rng.randint(5, 9)
        angles = np.sort(rng.uniform(0, 2 * math.pi, n_vertices))
        r_base = rng.uniform(20, 55)
        poly = []
        for a in angles:
            r = r_base * rng.uniform(0.5, 1.4)
            poly.append((cx + r * math.cos(a), cy + r * math.sin(a)))

        draw.polygon(poly, fill=color_rgb + (220,), outline=(color_rgb[0] // 2, color_rgb[1] // 2, color_rgb[2] // 2, 255))
        for _ in range(rng.randint(2, 4)):
            p1 = poly[rng.randint(0, len(poly))]
            p2 = poly[rng.randint(0, len(poly))]
            draw.line([p1, p2], fill=(255, 255, 255, 130), width=1)

    elif mp_type == "bead":
        cx, cy = rng.randint(50, size - 50), rng.randint(50, size - 50)
        radius = rng.uniform(18, 45)
        bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
        draw.ellipse(bbox, fill=color_rgb + (235,), outline=(40, 40, 40, 180))
        dot_r = radius * 0.25
        dot_x = cx - radius * 0.35
        dot_y = cy - radius * 0.35
        draw.ellipse([dot_x - dot_r, dot_y - dot_r, dot_x + dot_r, dot_y + dot_r], fill=(255, 255, 255, 240))

    else:  # film
        cx, cy = rng.randint(50, size - 50), rng.randint(50, size - 50)
        n_verts = rng.randint(6, 11)
        angles = np.sort(rng.uniform(0, 2 * math.pi, n_verts))
        r_base = rng.uniform(35, 65)
        poly = []
        for a in angles:
            r = r_base * rng.uniform(0.6, 1.3)
            poly.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        draw.polygon(poly, fill=color_rgb + (140,), outline=color_rgb + (210,))
        for _ in range(rng.randint(2, 5)):
            x1, y1 = cx + rng.uniform(-r_base, r_base) * 0.7, cy + rng.uniform(-r_base, r_base) * 0.7
            x2, y2 = x1 + rng.uniform(-20, 20), y1 + rng.uniform(-20, 20)
            draw.line([(x1, y1), (x2, y2)], fill=(255, 255, 255, 180), width=2)

    return img.convert("RGB")


def generate_non_microplastic_sample(size: int, seed: int) -> Image.Image:
    """
    Generate natural environmental debris AND macro scene hard negatives:
    - Sand / quartz grains
    - Plant matter / organic cellulose
    - Shell / calcite fragments
    - Macro scene textures (fabric, paper, everyday backgrounds)
    """
    rng = np.random.RandomState(seed)

    category = rng.choice(["sand", "organic_plant", "shell_calcite", "biofilm", "macro_texture", "clean_filter"])

    if category == "clean_filter":
        # Pure filter membrane / slide without any plastic
        img = create_microscope_background(size, rng, is_darkfield=rng.choice([True, False]))
        return img

    if category == "macro_texture":
        # Hard negative: fabric weave, paper grain, everyday scene texture
        bg = rng.randint(60, 200, 3)
        arr = np.full((size, size, 3), bg, dtype=np.float32)
        # Add repetitive weave pattern
        weave_freq = rng.randint(8, 20)
        grid_x = np.sin(np.linspace(0, weave_freq * np.pi, size))[:, np.newaxis]
        grid_y = np.cos(np.linspace(0, weave_freq * np.pi, size))[np.newaxis, :]
        pattern = (grid_x * grid_y) * rng.uniform(15, 35)
        arr += pattern[:, :, np.newaxis]
        arr += rng.normal(0, 10, (size, size, 3))
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    # Natural microscopy debris
    is_darkfield = rng.choice([True, False], p=[0.25, 0.75])
    img = create_microscope_background(size, rng, is_darkfield)
    draw = ImageDraw.Draw(img, "RGBA")

    natural_palettes = [
        (175, 150, 115),  # Sandy brown
        (140, 120, 90),   # Mud / Silt
        (110, 130, 80),   # Plant olive green
        (190, 175, 145),  # Quartz tan
        (160, 105, 65),   # Decaying organic bark
        (215, 210, 200),  # Chalky calcite
        (95, 80, 70),     # Dark soil peat
    ]
    color_rgb = natural_palettes[rng.randint(0, len(natural_palettes))]

    if category == "sand":
        cx, cy = rng.randint(45, size - 45), rng.randint(45, size - 45)
        radius = rng.uniform(22, 50)
        angles = np.linspace(0, 2 * math.pi, 14, endpoint=False)
        poly = []
        for a in angles:
            r = radius * rng.uniform(0.85, 1.15)
            poly.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        draw.polygon(poly, fill=color_rgb + (240,), outline=(color_rgb[0] // 2, color_rgb[1] // 2, color_rgb[2] // 2, 200))
        for _ in range(rng.randint(6, 14)):
            px = cx + rng.uniform(-radius * 0.7, radius * 0.7)
            py = cy + rng.uniform(-radius * 0.7, radius * 0.7)
            pr = rng.uniform(1.5, 4.0)
            draw.ellipse([px - pr, py - pr, px + pr, py + pr], fill=(color_rgb[0] // 2, color_rgb[1] // 2, color_rgb[2] // 2, 160))

    elif category == "organic_plant":
        cx, cy = rng.randint(40, size - 40), rng.randint(40, size - 40)
        w, h = rng.uniform(30, 70), rng.uniform(20, 50)
        poly = [
            (cx - w / 2, cy - h / 3),
            (cx + w / 2, cy - h / 2),
            (cx + w * 0.4, cy + h / 2),
            (cx - w * 0.4, cy + h / 3),
        ]
        draw.polygon(poly, fill=color_rgb + (210,), outline=(color_rgb[0] // 2, color_rgb[1] // 2, 40, 220))
        for step in np.linspace(-w * 0.4, w * 0.4, 6):
            draw.line([(cx + step, cy - h / 2.5), (cx + step, cy + h / 2.5)], fill=(80, 70, 50, 140), width=1)

    elif category == "shell_calcite":
        cx, cy = rng.randint(50, size - 50), rng.randint(50, size - 50)
        poly = [
            (cx - 30, cy - 25),
            (cx + 25, cy - 35),
            (cx + 40, cy + 20),
            (cx - 15, cy + 30),
        ]
        draw.polygon(poly, fill=(225, 220, 210, 235), outline=(150, 145, 135, 240))
        for dy in range(-20, 25, 7):
            draw.line([(cx - 20, cy + dy), (cx + 30, cy + dy - 5)], fill=(160, 155, 145, 180), width=1)

    else:  # biofilm
        for _ in range(rng.randint(4, 9)):
            bx = rng.randint(30, size - 30)
            by = rng.randint(30, size - 30)
            brx, bry = rng.uniform(15, 45), rng.uniform(10, 35)
            shade = rng.randint(-20, 20)
            blob_color = (
                max(0, min(255, color_rgb[0] + shade)),
                max(0, min(255, color_rgb[1] + shade)),
                max(0, min(255, color_rgb[2] + shade)),
                rng.randint(70, 160)
            )
            draw.ellipse([bx - brx, by - bry, bx + brx, by + bry], fill=blob_color)

    img = img.filter(ImageFilter.SMOOTH_MORE)
    return img.convert("RGB")


def build_enhanced_dataset(output_dir: str, num_per_class: int = 150, image_size: int = 224) -> str:
    """Generate high-quality balanced dataset with hard negatives."""
    mp_dir = Path(output_dir) / "microplastic"
    nmp_dir = Path(output_dir) / "non_microplastic"
    mp_dir.mkdir(parents=True, exist_ok=True)
    nmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"[DATASET] Generating {num_per_class} Microplastic images...")
    for i in range(num_per_class):
        img = generate_microplastic_sample(image_size, seed=20000 + i)
        img.save(mp_dir / f"mp_sample_{i:04d}.png")

    print(f"[DATASET] Generating {num_per_class} Non-microplastic (organic, mineral & macro) images...")
    for i in range(num_per_class):
        img = generate_non_microplastic_sample(image_size, seed=50000 + i)
        img.save(nmp_dir / f"nmp_sample_{i:04d}.png")

    readme = Path(output_dir) / "DATASET_INFO.txt"
    readme.write_text(
        f"Microplastic Vision High-Fidelity Dataset with Hard Negatives\n"
        f"Total images: {num_per_class * 2} ({num_per_class} per class)\n"
        f"Resolution: {image_size}x{image_size}\n"
        f"Classes: microplastic, non_microplastic\n"
        f"Generated: {datetime.now(timezone.utc).isoformat()}\n"
    )
    print(f"[OK] Enhanced dataset ready at: {output_dir}")
    return output_dir


# ════════════════════════════════════════════════════════════════════
# 2. TRANSFER LEARNING WITH RESNET-18
# ════════════════════════════════════════════════════════════════════

def train_resnet_classifier(
    dataset_path: str,
    epochs: int = 12,
    batch_size: int = 16,
    learning_rate: float = 0.0005,
    image_size: int = 224,
    seed: int = 42,
) -> dict:
    """Train ResNet-18 with label smoothing and calibrated outputs."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TRAIN] Initializing ResNet-18 Transfer Learning on: {device}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    from app.ml.training import discover_dataset
    paths, labels, class_names, class_counts = discover_dataset(dataset_path)
    (train_p, train_l), (val_p, val_l), (test_p, test_l) = split_dataset(
        paths, labels, train_ratio=0.70, val_ratio=0.15, seed=seed,
    )

    print(f"[SPLIT] {len(train_p)} Train | {len(val_p)} Val | {len(test_p)} Test")

    pp_config = get_default_config(image_size)
    train_transform = get_training_transform(pp_config)
    val_transform = get_validation_transform(pp_config)

    train_ds = ImageFolderDataset(train_p, train_l, class_names, train_transform)
    val_ds = ImageFolderDataset(val_p, val_l, class_names, val_transform)
    test_ds = ImageFolderDataset(test_p, test_l, class_names, val_transform)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Load pretrained ResNet-18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Freeze earlier layers (conv1, bn1, layer1, layer2) to preserve general visual features
    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in ["conv1", "bn1", "layer1", "layer2"]):
            param.requires_grad = False

    # Replace classifier head to match create_resnet18
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.to(device)

    # Label smoothing prevents extreme overconfident logits
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=1e-2,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_acc = 0.0
    best_weights = None
    history = []

    print("\n" + "=" * 65)
    print(f"{'Epoch':<8}{'Train Loss':<14}{'Train Acc':<14}{'Val Loss':<14}{'Val Acc':<12}")
    print("=" * 65)

    for epoch in range(1, epochs + 1):
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            # Gradient clipping prevents weight explosion
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            t_loss += loss.item() * imgs.size(0)
            preds = out.argmax(dim=1)
            t_correct += (preds == lbls).sum().item()
            t_total += imgs.size(0)

        scheduler.step()
        train_loss = t_loss / t_total
        train_acc = t_correct / t_total

        # Validation
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                out = model(imgs)
                loss = criterion(out, lbls)
                v_loss += loss.item() * imgs.size(0)
                preds = out.argmax(dim=1)
                v_correct += (preds == lbls).sum().item()
                v_total += imgs.size(0)

        val_loss = v_loss / v_total
        val_acc = v_correct / v_total

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4),
        })

        star = " * (Best)" if val_acc > best_val_acc else ""
        print(f"{epoch:<8}{train_loss:<14.4f}{train_acc:<14.2%}{val_loss:<14.4f}{val_acc:<12.2%}{star}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print("=" * 65 + "\n")

    if best_weights:
        model.load_state_dict(best_weights)

    # Test Set Evaluation
    print("[EVAL] Evaluating Best Model on Test Set...")
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device)
            out = model(imgs)
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(lbls.numpy())

    test_acc = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average=None)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(all_targets, all_preds, average="macro")
    cm = confusion_matrix(all_targets, all_preds).tolist()

    print(f"[METRIC] Test Accuracy:    {test_acc:.2%}")
    print(f"[METRIC] Precision (Macro): {p_macro:.2%}")
    print(f"[METRIC] Recall (Macro):    {r_macro:.2%}")
    print(f"[METRIC] F1 Score (Macro):  {f1_macro:.2%}")
    print(f"[METRIC] Confusion Matrix:  {cm}")

    # Checkpoint saving
    settings = get_settings()
    ckpt_dir = Path(settings.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    run_id = f"resnet_{uuid.uuid4().hex[:6]}"
    save_payload = {
        "model_state_dict": best_weights if best_weights else model.state_dict(),
        "architecture": "resnet18",
        "num_classes": len(class_names),
        "class_mapping": {i: name for i, name in enumerate(class_names)},
        "class_names": class_names,
        "best_val_accuracy": round(best_val_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "f1_macro": round(f1_macro, 4),
        "run_id": run_id,
        "preprocessing_config": pp_config.to_dict(),
        "image_size": image_size,
    }

    resnet_path = str(ckpt_dir / "resnet18_best.pth")
    torch.save(save_payload, resnet_path)
    print(f"[CHECKPOINT] Saved: {resnet_path}")

    return {
        "run_id": run_id,
        "architecture": "resnet18",
        "checkpoint_path": resnet_path,
        "best_val_accuracy": best_val_acc,
        "test_accuracy": test_acc,
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "f1_macro": f1_macro,
        "confusion_matrix": cm,
        "class_names": class_names,
        "training_history": history,
    }


if __name__ == "__main__":
    raise SystemExit(
        "Synthetic particle images are not valid training labels. Use the validated dataset "
        "workflow (`python -m app.ml.train_cli --dataset <path>`) with microscopy-confirmed data."
    )
