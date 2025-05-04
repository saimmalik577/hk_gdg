import os
import json

# Paths
base_dir = "data/processed/subset_2000"
img_dir = os.path.join(base_dir, "images")
ocr_dir = os.path.join(base_dir, "ocr")
qa_path = os.path.join(base_dir, "qas", "train_subset.json")

# Read annotation data
with open(qa_path, "r") as f:
    annotations = json.load(f)

# Extract cleaned image IDs (remove "documents/" if present)
qa_ids = set(entry["image"].split("/")[-1].split(".")[0] for entry in annotations["data"])

# Load available image and OCR filenames
img_ids = set(os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith(".png"))
ocr_ids = set(os.path.splitext(f)[0] for f in os.listdir(ocr_dir) if f.endswith(".json"))

# Check matches
matching_ids = qa_ids & img_ids & ocr_ids

print(f"✅ Total samples in annotations: {len(qa_ids)}")
print(f"🖼️  Matching images: {len(img_ids & qa_ids)}")
print(f"📝 Matching OCRs: {len(ocr_ids & qa_ids)}")
print(f"🧩 Fully matched entries (all 3): {len(matching_ids)}")

missing_imgs = qa_ids - img_ids
missing_ocr = qa_ids - ocr_ids

if missing_imgs:
    print(f"⚠️ Missing images for {len(missing_imgs)} samples. Examples: {list(missing_imgs)[:5]}")
if missing_ocr:
    print(f"⚠️ Missing OCRs for {len(missing_ocr)} samples. Examples: {list(missing_ocr)[:5]}")
