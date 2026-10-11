"""
Ingest real microplastics and background from Hugging Face WWTP dataset,
plus synthetic artwork/graphic/portrait hard negatives to prevent false positives.
"""
import io
import math
import os
import random
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
from huggingface_hub import hf_hub_download

def ingest_hf_wwtp(max_slides: int = 100):
    print("[HF] Downloading a training split from andreusv/Microplastic-Fragments-WWTP-Dataset...")
    parquet_path = hf_hub_download(
        repo_id="andreusv/Microplastic-Fragments-WWTP-Dataset",
        filename="data/train-00000-of-00001.parquet",
        repo_type="dataset",
    )
    df = pd.read_parquet(parquet_path)
    print(f"[HF] Parquet loaded with {len(df)} microscope slide records.")

    mp_dest = Path("data/datasets/wwtp_microscopy/microplastic")
    nmp_dest = Path("data/datasets/wwtp_microscopy/non_microplastic")
    mp_dest.mkdir(parents=True, exist_ok=True)
    nmp_dest.mkdir(parents=True, exist_ok=True)

    extracted_mp = 0
    extracted_bg = 0

    # Extract crops from the first 25 high-res microscope images
    for idx in range(min(max_slides, len(df))):
        row = df.iloc[idx]
        img_bytes = row["image"]["bytes"]
        full_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        w, h = full_img.size

        bboxes = row["objects"]["bbox"]
        # Save microplastic crops with context
        for b_idx, box in enumerate(bboxes):
            bx, by, bw, bh = box
            # Add padding around the particle
            pad = max(15, int(max(bw, bh) * 0.25))
            x1 = max(0, int(bx - pad))
            y1 = max(0, int(by - pad))
            x2 = min(w, int(bx + bw + pad))
            y2 = min(h, int(by + bh + pad))

            if (x2 - x1) >= 25 and (y2 - y1) >= 25:
                crop = full_img.crop((x1, y1, x2, y2))
                crop = crop.resize((224, 224), Image.Resampling.BICUBIC)
                crop.save(mp_dest / f"hf_wwtp_mp_{idx:03d}_{b_idx:02d}.png")
                extracted_mp += 1

        # Extract 2 clean substrate patches without particles
        saved_background = 0
        for s in range(12):
            # Pick a corner patch or patch far from bounding boxes
            sx = (s * 1500 + 100) % (w - 250)
            sy = (s * 800 + 150) % (h - 250)
            # Keep background crops away from annotated particles.
            if any(not (sx + 224 < bx or sx > bx + bw or sy + 224 < by or sy > by + bh) for bx, by, bw, bh in bboxes):
                continue
            bg_crop = full_img.crop((sx, sy, sx + 224, sy + 224))
            bg_crop.save(nmp_dest / f"hf_wwtp_substrate_{idx:03d}_{s}.png")
            extracted_bg += 1
            saved_background += 1
            if saved_background >= 4:
                break

    print(f"[HF] Extracted {extracted_mp} real microplastic fragments and {extracted_bg} clean substrate patches.")

def add_hard_negative_artwork_and_scenes(output_dir="data/datasets/ood_graphics"):
    """Generate explicitly out-of-domain graphics; never use as microscopy negatives."""
    nmp_dest = Path(output_dir)
    nmp_dest.mkdir(parents=True, exist_ok=True)
    
    print("[OOD DATA] Generating artwork, sketches, and everyday scene examples...")
    rng = random.Random(777)
    
    # 1. Anime / sketch style face & eyes negatives (the exact failure case the user reported)
    for i in range(40):
        img = Image.new("RGB", (224, 224), color=(rng.randint(210, 245), rng.randint(210, 245), rng.randint(210, 245)))
        draw = ImageDraw.Draw(img)
        # Draw eyes and hair strands
        eye_y = 110 + rng.randint(-15, 15)
        # Eyes
        draw.ellipse([50, eye_y - 20, 95, eye_y + 15], fill=(230, 235, 240), outline=(30, 30, 35), width=2)
        draw.ellipse([129, eye_y - 20, 174, eye_y + 15], fill=(230, 235, 240), outline=(30, 30, 35), width=2)
        # Irises
        iris_color = (rng.randint(40, 100), rng.randint(60, 120), rng.randint(90, 160))
        draw.ellipse([65, eye_y - 12, 85, eye_y + 10], fill=iris_color)
        draw.ellipse([144, eye_y - 12, 164, eye_y + 10], fill=iris_color)
        # Hair strands (black/dark fine lines)
        for _ in range(rng.randint(15, 30)):
            hx1 = rng.randint(20, 204)
            hy1 = rng.randint(0, 40)
            hx2 = hx1 + rng.randint(-30, 30)
            hy2 = rng.randint(70, 180)
            draw.line([(hx1, hy1), (hx2, hy2)], fill=(25, 25, 30), width=rng.randint(1, 3))
        img.save(nmp_dest / f"neg_sketch_portrait_{i:03d}.png")

    # 2. Graphic posters, typography, ink splatters (like the Toji poster)
    for i in range(30):
        img = Image.new("RGB", (224, 224), color=(rng.randint(240, 255), rng.randint(240, 255), rng.randint(240, 255)))
        draw = ImageDraw.Draw(img)
        # Dark silhouette or large block
        draw.polygon([(0, 60), (140, 0), (200, 180), (40, 224)], fill=(20, 20, 25))
        # Text-like rectangles
        for ty in range(80, 190, 25):
            draw.rectangle([150, ty, 210, ty + 12], fill=(20, 20, 25))
        # Ink spray
        for _ in range(40):
            sx = rng.randint(0, 224)
            sy = rng.randint(0, 224)
            sr = rng.uniform(0.5, 3.0)
            draw.ellipse([sx - sr, sy - sr, sx + sr, sy + sr], fill=(15, 15, 20))
        img.save(nmp_dest / f"neg_poster_splatter_{i:03d}.png")

    # 3. Macro everyday objects (fabrics, wood grain, desk surface, finger skin)
    for i in range(40):
        skin_or_desk = rng.choice([
            (210, 160, 130), # skin tone
            (140, 95, 60),   # wood
            (80, 90, 110),   # fabric denim
            (190, 190, 195), # metal / paper
        ])
        arr = np.ones((224, 224, 3), dtype=np.float32) * np.array(skin_or_desk)
        # Add smooth gradients and macro blur
        grad = np.linspace(-30, 30, 224).reshape(1, 224, 1)
        arr += grad
        arr += np.random.normal(0, 8, (224, 224, 3))
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        img.save(nmp_dest / f"neg_macro_surface_{i:03d}.png")

    print("[OOD DATA] Generated 110 examples for domain-gate evaluation only.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest-wwtp", action="store_true", help="Download/crop the WWTP dataset")
    parser.add_argument("--generate-ood", action="store_true", help="Generate synthetic OOD graphics for gate evaluation only")
    args = parser.parse_args()
    if args.ingest_wwtp:
        ingest_hf_wwtp()
    if args.generate_ood:
        add_hard_negative_artwork_and_scenes()
    if not args.ingest_wwtp and not args.generate_ood:
        parser.error("Choose --ingest-wwtp or --generate-ood; synthetic OOD graphics are never training negatives.")
