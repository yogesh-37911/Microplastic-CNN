"""Create a provenance-labeled binary crop dataset from the public CA-filter set.

Positive labels come from manually annotated microfibre polygons. Negative labels
mean only that the crop contains no polygon annotated in its source image; they
are not chemically verified plastic-free samples.
"""

from __future__ import annotations

import io
import json
import random
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image


REPO_ID = "femartip/microfibres_CA_filter"
PARQUET = "data/train-00000-of-00001-d73d2a63ae96996e.parquet"
OUT = Path("data/datasets/microfibres_ca_crops_v1")
CROP_SIZE = 224


def _polygon_points(mask: dict) -> list[list[tuple[float, float]]]:
    raw = (mask or {}).get("polygons", {}).get("points", [])
    polygons: list[list[tuple[float, float]]] = []
    for polygon in raw:
        points = [(float(p[0]), float(p[1])) for p in polygon if len(p) >= 2]
        if len(points) >= 3:
            polygons.append(points)
    return polygons


def _square_crop(image: Image.Image, cx: float, cy: float, side: int) -> Image.Image:
    width, height = image.size
    side = min(max(1, int(side)), width, height)
    left = min(max(0, int(round(cx - side / 2))), width - side)
    top = min(max(0, int(round(cy - side / 2))), height - side)
    return image.crop((left, top, left + side, top + side)).resize(
        (CROP_SIZE, CROP_SIZE), Image.Resampling.LANCZOS
    )


def _negative_crop(image: Image.Image, polygons: list[list[tuple[float, float]]], seed: int):
    width, height = image.size
    if width < CROP_SIZE or height < CROP_SIZE:
        return None
    # Bounding boxes provide a fast conservative exclusion buffer around labels.
    excluded = [
        (min(x for x, _ in poly) - 8, min(y for _, y in poly) - 8,
         max(x for x, _ in poly) + 8, max(y for _, y in poly) + 8)
        for poly in polygons
    ]
    candidates = [
        (x, y)
        for y in range(0, height - CROP_SIZE + 1, 112)
        for x in range(0, width - CROP_SIZE + 1, 112)
    ]
    if width > CROP_SIZE:
        candidates.append((width - CROP_SIZE, max(0, (height - CROP_SIZE) // 2)))
    if height > CROP_SIZE:
        candidates.append((max(0, (width - CROP_SIZE) // 2), height - CROP_SIZE))
    rng = random.Random(seed)
    rng.shuffle(candidates)
    for left, top in candidates:
        right, bottom = left + CROP_SIZE, top + CROP_SIZE
        overlaps_annotation = any(
            not (right <= x1 or left >= x2 or bottom <= y1 or top >= y2)
            for x1, y1, x2, y2 in excluded
        )
        if not overlaps_annotation:
            return image.crop((left, top, left + CROP_SIZE, top + CROP_SIZE)).resize(
                (CROP_SIZE, CROP_SIZE), Image.Resampling.LANCZOS
            )
    return None


def main() -> None:
    parquet_path = hf_hub_download(REPO_ID, PARQUET, repo_type="dataset")
    rows = pd.read_parquet(parquet_path)
    positive_dir = OUT / "microplastic"
    negative_dir = OUT / "non_microplastic"
    positive_dir.mkdir(parents=True, exist_ok=True)
    negative_dir.mkdir(parents=True, exist_ok=True)

    positives = negatives = skipped = 0
    for source_id, row in rows.iterrows():
        try:
            image = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
            polygons = _polygon_points(row["mask"])
            if not polygons:
                skipped += 1
                continue

            # Train on the largest annotated fibre in each original image.
            polygon = max(polygons, key=lambda pts: (max(x for x, _ in pts) - min(x for x, _ in pts)) * (max(y for _, y in pts) - min(y for _, y in pts)))
            xs, ys = [p[0] for p in polygon], [p[1] for p in polygon]
            bw, bh = max(xs) - min(xs), max(ys) - min(ys)
            side = min(max(CROP_SIZE, int(max(bw, bh) * 2.0 + 48)), 512)
            positive = _square_crop(image, (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, side)
            negative = _negative_crop(image, polygons, seed=int(source_id))

            positive.save(positive_dir / f"mfca_pos_{source_id:04d}_fiber.jpg", quality=95)
            positives += 1
            if negative is not None:
                negative.save(negative_dir / f"mfca_neg_{source_id:04d}_background.jpg", quality=95)
                negatives += 1
        except Exception as exc:
            skipped += 1
            print(f"Skipping source row {source_id}: {exc}")

    manifest = {
        "dataset_name": "Microfibres on Cellulose Acetate Filters",
        "repository": f"https://huggingface.co/datasets/{REPO_ID}",
        "license": "CC BY 4.0",
        "source_images": len(rows),
        "positive_crops": positives,
        "negative_crops": negatives,
        "skipped_rows": skipped,
        "label_method": {
            "microplastic": "Crop centered on the largest manually annotated microfibre polygon in a source image.",
            "non_microplastic": "Crop with no annotated microfibre polygon (17 px exclusion buffer) in that source image; not chemically confirmed plastic-free.",
        },
        "split_group": "original source image row; positive and negative crops from one source stay together",
        "limitations": [
            "The data represents microfibres on cellulose acetate filters, not all microplastic morphologies.",
            "The source comes from a specific wastewater sampling and microscopy pipeline.",
            "Groups are source-image rows; the dataset does not establish independent validation across filters, sessions, or laboratories.",
            "Background crops are annotation-negative only; unannotated particles may remain.",
            "This is suitable for exploratory CNN training, not deployment or general accuracy claims.",
        ],
    }
    (OUT / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
