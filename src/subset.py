import os
import json
import random
import shutil
from pathlib import Path

# Configurable paths
NUM_SAMPLES = 2000
BASE_DIR = Path("data/raw")
OUT_DIR = Path("data/processed/subset_2000")

IMG_DIR = BASE_DIR / "spdocvqa_images"
OCR_DIR = BASE_DIR / "spdocvqa_ocr"
QAS_FILE = BASE_DIR / "spdocvqa_qas/train_v1.0_withQT.json"

IMG_OUT = OUT_DIR / "images"
OCR_OUT = OUT_DIR / "ocr"
QAS_OUT = OUT_DIR / "qas"

# Create output dirs
IMG_OUT.mkdir(parents=True, exist_ok=True)
OCR_OUT.mkdir(parents=True, exist_ok=True)
QAS_OUT.mkdir(parents=True, exist_ok=True)

# Load annotations
with open(QAS_FILE, "r") as f:
    full_data = json.load(f)

all_entries = full_data["data"]
random.shuffle(all_entries)

subset = []
count = 0

for entry in all_entries:
    image_name = entry["image"].split("/")[-1]
    ocr_name = image_name.replace(".jpg", ".json").replace(".png", ".json")

    img_path = IMG_DIR / image_name
    ocr_path = OCR_DIR / ocr_name

    if img_path.exists() and ocr_path.exists():
        shutil.copy(img_path, IMG_OUT / image_name)
        shutil.copy(ocr_path, OCR_OUT / ocr_name)
        subset.append(entry)
        count += 1

    if count == NUM_SAMPLES:
        break

# Save subset annotation
with open(QAS_OUT / "train_subset.json", "w") as f:
    json.dump({"version": "1.0", "data": subset}, f, indent=2)

print(f"✅ Extracted {len(subset)} samples to: {OUT_DIR}")
