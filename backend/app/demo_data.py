"""
Microplastic Vision – demo data generator.

Creates small synthetic demo images so the app can run without
downloading any external dataset.  Clearly labels all demo content.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def generate_demo_dataset(
    output_dir: str = "data/datasets/demo",
    num_per_class: int = 20,
    image_size: int = 224,
    seed: int = 42,
) -> str:
    """
    Generate a small synthetic demo dataset.

    Creates two class directories:
      - microplastic/   – dark images with small bright particle-like blobs
      - non_microplastic/ – lighter images with organic-looking textures

    Returns the output directory path.
    """
    random.seed(seed)
    np.random.seed(seed)

    mp_dir = Path(output_dir) / "microplastic"
    nmp_dir = Path(output_dir) / "non_microplastic"
    mp_dir.mkdir(parents=True, exist_ok=True)
    nmp_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_per_class):
        # Microplastic-like: dark background with bright particles
        img = _generate_microplastic_image(image_size, seed + i)
        img.save(mp_dir / f"demo_mp_{i:04d}.png")

        # Non-microplastic-like: lighter organic textures
        img = _generate_non_microplastic_image(image_size, seed + i + 1000)
        img.save(nmp_dir / f"demo_nmp_{i:04d}.png")

    # Write a README
    readme_path = Path(output_dir) / "README.txt"
    readme_path.write_text(
        "DEMO DATASET\n"
        "=============\n\n"
        "This dataset contains synthetic (generated) images for demonstration\n"
        "purposes only.  These images do NOT depict real microplastics or real\n"
        "environmental samples.\n\n"
        "Any model trained on this data is for functional testing only and\n"
        "MUST NOT be used for scientific conclusions.\n\n"
        f"Generated: {num_per_class} images per class\n"
        f"Image size: {image_size}x{image_size}\n"
        f"Seed: {seed}\n"
    )

    return output_dir


def _generate_microplastic_image(size: int, seed: int) -> Image.Image:
    """Dark background with bright particle-like spots."""
    rng = np.random.RandomState(seed)

    # Dark blue-gray background
    bg_r = rng.randint(15, 40)
    bg_g = rng.randint(20, 50)
    bg_b = rng.randint(30, 60)
    arr = np.full((size, size, 3), [bg_r, bg_g, bg_b], dtype=np.uint8)

    # Add subtle noise
    noise = rng.randint(-8, 8, (size, size, 3), dtype=np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)

    # Draw 2-6 bright particle-like shapes
    num_particles = rng.randint(2, 7)
    for _ in range(num_particles):
        cx = rng.randint(20, size - 20)
        cy = rng.randint(20, size - 20)
        rx = rng.randint(3, 15)
        ry = rng.randint(3, 15)

        shape_type = rng.choice(["ellipse", "irregular"])
        color = (
            rng.randint(180, 255),
            rng.randint(180, 255),
            rng.randint(180, 255),
        )

        if shape_type == "ellipse":
            draw.ellipse(
                [cx - rx, cy - ry, cx + rx, cy + ry],
                fill=color, outline=color,
            )
        else:
            # Irregular polygon
            n_points = rng.randint(5, 9)
            angles = np.sort(rng.uniform(0, 2 * np.pi, n_points))
            points = []
            for angle in angles:
                r = rng.uniform(rx * 0.5, rx * 1.5)
                px = cx + r * np.cos(angle)
                py = cy + r * np.sin(angle)
                points.append((px, py))
            draw.polygon(points, fill=color, outline=color)

    return img


def _generate_non_microplastic_image(size: int, seed: int) -> Image.Image:
    """Lighter organic-looking texture (sand/sediment)."""
    rng = np.random.RandomState(seed)

    # Warm beige/sand background
    bg_r = rng.randint(160, 210)
    bg_g = rng.randint(150, 195)
    bg_b = rng.randint(120, 170)
    arr = np.full((size, size, 3), [bg_r, bg_g, bg_b], dtype=np.uint8)

    # Stronger organic noise
    noise = rng.randint(-25, 25, (size, size, 3), dtype=np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)

    # Add some organic blobs (larger, more diffuse)
    num_blobs = rng.randint(3, 8)
    for _ in range(num_blobs):
        cx = rng.randint(15, size - 15)
        cy = rng.randint(15, size - 15)
        rx = rng.randint(8, 30)
        ry = rng.randint(8, 30)
        shade = rng.randint(-30, 30)
        color = (
            max(0, min(255, bg_r + shade)),
            max(0, min(255, bg_g + shade - 5)),
            max(0, min(255, bg_b + shade - 10)),
        )
        draw.ellipse(
            [cx - rx, cy - ry, cx + rx, cy + ry],
            fill=color,
        )

    return img


if __name__ == "__main__":
    path = generate_demo_dataset()
    print(f"Demo dataset generated at: {path}")
