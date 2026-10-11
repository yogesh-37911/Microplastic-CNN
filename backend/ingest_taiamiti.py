"""
Ingest annotated fluorescent microscopy samples from taiamiti/microplastic-beni-segmentation
using the configured Hugging Face Access Token.
"""
import io
import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from huggingface_hub import HfApi, hf_hub_download
import numpy as np
from PIL import Image

load_dotenv()
load_dotenv(find_dotenv(usecwd=True))

HF_TOKEN = os.getenv("HF_TOKEN")
REPO_ID = "taiamiti/microplastic-beni-segmentation"

def check_access() -> bool:
    api = HfApi(token=HF_TOKEN)
    try:
        # Check if we can download a ground truth file
        if not HF_TOKEN:
            print("[NOTICE] Set HF_TOKEN in the environment or backend .env after requesting dataset access.")
            return False
        files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
        gt_files = [f for f in files if f.startswith("fields/ground_truth/") and f.endswith(".png")]
        if not gt_files:
            return False
        test_file = gt_files[0]
        hf_hub_download(repo_id=REPO_ID, filename=test_file, repo_type="dataset", token=HF_TOKEN)
        return True
    except Exception as e:
        print(f"[ACCESS CHECK] Status: {e}")
        return False

def ingest_beni_samples(max_samples: int = 100):
    print(f"[TAIAMITI INGEST] Checking gated access for {REPO_ID} with provided HF_TOKEN...")
    api = HfApi(token=HF_TOKEN)
    try:
        files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    except Exception as e:
        print(f"[ERROR] Could not list repository: {e}")
        return False

    gt_files = [f for f in files if f.startswith("fields/ground_truth/") and f.endswith(".png")]
    print(f"[TAIAMITI INGEST] Found {len(gt_files)} annotated ground truth masks.")

    mp_dest = Path("data/datasets/beni_microscopy/microplastic")
    nmp_dest = Path("data/datasets/beni_microscopy/non_microplastic")
    mp_dest.mkdir(parents=True, exist_ok=True)
    nmp_dest.mkdir(parents=True, exist_ok=True)

    success_count = 0
    for idx, mask_rel_path in enumerate(gt_files[:max_samples]):
        # Match data file: replace fields/ground_truth/ground_truth_0 with data/data_0 and .png with .jpg
        filename = Path(mask_rel_path).name
        stem = Path(mask_rel_path).stem
        img_rel_path = f"data/data_0/{stem}.jpg"

        try:
            mask_local = hf_hub_download(repo_id=REPO_ID, filename=mask_rel_path, repo_type="dataset", token=HF_TOKEN)
            img_local = hf_hub_download(repo_id=REPO_ID, filename=img_rel_path, repo_type="dataset", token=HF_TOKEN)

            mask = Image.open(mask_local)
            raw = Image.open(img_local).convert("RGB")
            mask_arr = np.array(mask)

            # If mask has positive segmentation pixels, it's a microplastic sample
            if mask_arr.shape[:2] != np.asarray(raw).shape[:2]:
                print(f"[WARNING] Image/mask shape mismatch for {mask_rel_path}; skipping to avoid incorrect labels.")
                continue
            if (mask_arr > 0).sum() > 50:
                raw.save(mp_dest / f"beni_mp_{idx:04d}.jpg", quality=95)
                success_count += 1
            else:
                # Negative background sample from bivalve slide
                raw.save(nmp_dest / f"beni_nmp_{idx:04d}.jpg", quality=95)
                success_count += 1

            if (idx + 1) % 10 == 0:
                print(f"[{idx + 1}/{min(max_samples, len(gt_files))}] Ingested BENI samples...")

        except Exception as e:
            print(f"[ERROR] Failed to download {mask_rel_path}: {e}")
            break

    print(f"[COMPLETE] Successfully ingested {success_count} BENI bivalve microscopy samples!")
    if success_count > 0:
        print("[NOTICE] Dataset ingestion completed. Review labels/provenance and validate the dataset before training.")
    return True

if __name__ == "__main__":
    if check_access():
        print("[SUCCESS] Access verified! Ingesting dataset...")
        ingest_beni_samples(max_samples=150)
    else:
        print("[NOTICE] Please visit https://huggingface.co/datasets/taiamiti/microplastic-beni-segmentation and click 'Request Access' or 'Access repository' while logged in.")

