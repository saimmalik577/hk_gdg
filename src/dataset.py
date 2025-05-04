import torch
from torch.utils.data import Dataset

class SmartDocDataset(Dataset):
    def __init__(self, data_path):
        self.samples = torch.load(data_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        # HuggingFace Trainer expects keys: input_ids, attention_mask, bbox, pixel_values, labels (optional)
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "bbox": torch.tensor(item["bbox"], dtype=torch.long),
            "pixel_values": item["pixel_values"],  # already a tensor
            "labels": torch.tensor(item["labels"], dtype=torch.long)  # token-level answer indices
        }
