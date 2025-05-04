import os
import json
from tqdm import tqdm
import torch
from transformers import LayoutLMv3Processor
from PIL import Image

# Constants
DATA_PATH = "data/prepared/train_data.pt"
OUTPUT_PATH = "data/prepared/tokenized_data.pt"

# Load data
samples = torch.load(DATA_PATH)

# Load processor with apply_ocr=False so we can pass our own bounding boxes
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

tokenized = []

print("🧠 Tokenizing and preparing training data...")

for sample in tqdm(samples):
    try:
        image = Image.open(sample["image_path"]).convert("RGB")

        words = []
        boxes = []

        for line in sample["ocr_lines"]:
            line_text = line["text"]
            line_words = line.get("words", [])

            for word in line_words:
                words.append(word["text"])
                boxes.append(word["boundingBox"])

        if len(words) == 0 or len(boxes) == 0:
            continue

        # Flatten boxes to 4-point format expected by LayoutLMv3
        norm_boxes = []
        for box in boxes:
            # boundingBox = [x0, y0, x1, y1, x2, y2, x3, y3]
            x_coords = box[::2]
            y_coords = box[1::2]
            x_min = min(x_coords)
            y_min = min(y_coords)
            x_max = max(x_coords)
            y_max = max(y_coords)
            norm_boxes.append([x_min, y_min, x_max, y_max])

        encoding = processor(
            images=image,
            text=words,
            boxes=norm_boxes,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        )

        # Append necessary fields
        tokenized.append({
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "bbox": encoding["bbox"].squeeze(0),
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "question": sample["question"],
            "answers": sample["answers"]
        })

    except Exception as e:
        print(f"⚠️ Skipping due to error: {e}")
        continue

# Save tokenized data
torch.save(tokenized, OUTPUT_PATH)
print(f"\n✅ Tokenized {len(tokenized)} samples saved to {OUTPUT_PATH}")
