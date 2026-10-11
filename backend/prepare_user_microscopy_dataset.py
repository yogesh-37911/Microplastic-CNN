"""Combine the labeled CA-filter crops with a user-provided microscope image set."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image, ImageOps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images",
        type=Path,
        default=Path.home() / "Pictures" / "Screenshots",
        help="Folder containing the WhatsApp microscope JPEGs",
    )
    parser.add_argument(
        "--base-dataset",
        type=Path,
        default=Path("data/datasets/microfibres_ca_crops_v1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/datasets/microfibres_ca_user_20261011"),
    )
    args = parser.parse_args()

    images_dir = args.images.expanduser().resolve()
    base_dataset = args.base_dataset.resolve()
    output = args.output.resolve()
    if not images_dir.is_dir():
        parser.error(f"Image folder does not exist: {images_dir}")
    if not base_dataset.is_dir():
        parser.error(f"Base dataset does not exist: {base_dataset}")
    if output.exists():
        parser.error(f"Output already exists; choose a new output folder: {output}")

    sources = sorted(images_dir.glob("WhatsApp Image 2026-10-11 at 6.21*.jpeg"))
    if not sources:
        parser.error(f"No matching microscope JPEGs found in {images_dir}")

    shutil.copytree(base_dataset, output)
    positive_dir = output / "microplastic"
    positive_dir.mkdir(parents=True, exist_ok=True)
    accepted: list[str] = []
    for index, source in enumerate(sources, start=1):
        try:
            with Image.open(source) as image:
                ImageOps.exif_transpose(image).verify()
        except Exception as exc:
            shutil.rmtree(output)
            parser.error(f"Unreadable image {source}: {exc}")
        destination = positive_dir / f"real_whatsapp_microplastic_20261011_{index:03d}.jpeg"
        shutil.copy2(source, destination)
        accepted.append(source.name)

    manifest = {
        "dataset_name": "CA-filter crops plus user microscope images",
        "base_dataset": str(base_dataset),
        "user_image_directory": str(images_dir),
        "user_images_added": len(accepted),
        "user_image_files": accepted,
        "user_label_method": (
            "User-identified microplastic microscopy images; polymer identity and label "
            "have not been independently chemically verified."
        ),
        "user_images_split_group": "one acquisition group for all images in this upload set",
        "base_label_limitations": (
            "CA-filter negatives are crops with no annotated fibre, not chemically confirmed "
            "plastic-free samples."
        ),
    }
    (output / "combined_dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Prepared {len(accepted)} user images in {output}")
    print(f"Training dataset: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
