import torch
from pathlib import Path

data = torch.load("data/prepared/train_data.pt")

all_exist = True
for i, sample in enumerate(data):
    img_path = Path(sample["image_path"])
    if not img_path.exists():
        print(f"❌ Missing image: {img_path}")
        all_exist = False
        break

if all_exist:
    print("✅ All image paths are valid.")
