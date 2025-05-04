import json
import os

ocr_sample = "data/processed/subset_2000/ocr/krkb0228_1.json"  # replace with any OCR file path you have

with open(ocr_sample, "r") as f:
    data = json.load(f)

print(json.dumps(data, indent=2)[:1000])  # print the first 1000 characters to inspect
