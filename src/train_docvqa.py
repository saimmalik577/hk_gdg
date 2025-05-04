# src/train_docvqa.py

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    AutoModelForTokenClassification, # Correct model head for token labels
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import logging # Added logging

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- 1. Define your Custom Dataset Class (Copy/Paste or Import) ---
class SmartDocDataset(Dataset):
    def __init__(self, data_path):
        logging.info(f"Attempting to load data from {data_path}...")
        self.samples = torch.load(data_path)
        logging.info(f"Successfully loaded {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        # The data in tokenized_data.pt should already contain tensors
        # after the latest preprocess.py script.
        # If they are *not* tensors, we might need to adjust, but let's assume they are.
        try:
            # Check if keys exist and are tensors, else convert (more robust)
            input_ids = item["input_ids"]
            attention_mask = item["attention_mask"]
            bbox = item["bbox"]
            pixel_values = item["pixel_values"]
            labels = item["labels"]

            # Ensure correct types - Processor output should be tensors already,
            # but adding checks doesn't hurt.
            if not isinstance(input_ids, torch.Tensor): input_ids = torch.tensor(input_ids, dtype=torch.long)
            if not isinstance(attention_mask, torch.Tensor): attention_mask = torch.tensor(attention_mask, dtype=torch.long)
            if not isinstance(bbox, torch.Tensor): bbox = torch.tensor(bbox, dtype=torch.long)
            # pixel_values should definitely be a tensor from the processor
            if not isinstance(pixel_values, torch.Tensor):
                 logging.warning(f"Sample {idx}: pixel_values is not a tensor (Type: {type(pixel_values)}). Attempting conversion.")
                 # This might fail depending on the actual type
                 pixel_values = torch.tensor(pixel_values, dtype=torch.float) # Assuming float
            if not isinstance(labels, torch.Tensor): labels = torch.tensor(labels, dtype=torch.long)

            return {
                "input_ids": input_ids.to(torch.long),
                "attention_mask": attention_mask.to(torch.long),
                "bbox": bbox.to(torch.long),
                "pixel_values": pixel_values, # Keep original dtype unless conversion needed
                "labels": labels.to(torch.long)
            }
        except KeyError as e:
            logging.error(f"KeyError: Missing key {e} in sample index {idx}. Sample keys: {item.keys()}")
            # You might want to raise the error or return None/empty dict depending on handling strategy
            raise e # Re-raise the error to stop execution and see the problem
        except Exception as e:
            logging.error(f"Error processing sample index {idx}: {e}", exc_info=True)
            raise e
# --- End of Dataset Class Definition ---


# --- 2. Configuration ---
MODEL_CHECKPOINT = "microsoft/layoutlmv3-base" # Or layoutlmv3-large
TRAIN_DATA_PATH = "data/prepared/tokenized_data.pt" # Verified path to your data
OUTPUT_DIR = "./models/layoutlmv3_finetuned_smartdoc_qa" # Where the fine-tuned model will be saved

# !!! IMPORTANT: Match this to the number of unique token labels in your data (e.g., O, B-ANSWER, I-ANSWER -> 3)
NUM_LABELS = 3 # <--- Verify this number based on your tokenization/labeling step

# --- 3. Load the Pre-trained Model with Token Classification Head ---
logging.info(f"Loading model from {MODEL_CHECKPOINT} with {NUM_LABELS} labels...")
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=NUM_LABELS,
    # ignore_mismatched_sizes=True # Add if loading a checkpoint with different head size, but usually not needed for fine-tuning
)
logging.info("Model loaded successfully.")

# --- 4. Define Training Arguments ---
# Using the version compatible with older transformers (no evaluation_strategy)
logging.info("Defining training arguments...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,  # Keep low based on potential memory limits
    gradient_accumulation_steps=4,  # Effective batch size = 4 * 4 = 16
    learning_rate=5e-5,
    num_train_epochs=15,            # Adjust as needed based on convergence
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=50,               # Log training loss every 50 steps
    save_strategy="epoch",          # Save a checkpoint every epoch
    # Removed evaluation_strategy, load_best_model_at_end
    report_to="none",
    remove_unused_columns=False     # Important for LayoutLMv3
)
logging.info("Training arguments defined.")

# --- 5. Prepare Dataset and Data Collator ---
logging.info(f"Instantiating dataset from {TRAIN_DATA_PATH}...")
train_dataset = SmartDocDataset(TRAIN_DATA_PATH) # Instantiate your custom dataset

logging.info("Loading processor for data collator...")
processor = AutoProcessor.from_pretrained(MODEL_CHECKPOINT, apply_ocr=False) # Needed for tokenizer padding ID
data_collator = DataCollatorWithPadding(processor.tokenizer, padding="longest") # Pad batches dynamically
logging.info("Data collator initialized.")

# --- 6. Initialize the Trainer ---
logging.info("Initializing the Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)
logging.info("Trainer initialized.")

# --- 7. Start Fine-tuning ---
logging.info("\nStarting fine-tuning...")
try:
    trainer.train()
    logging.info("✅ Fine-tuning complete!")
except Exception as e:
    logging.error(f"🚨 An error occurred during training: {e}", exc_info=True)


# --- 8. Optional: Explicitly save the final model ---
# Although Trainer saves checkpoints, you can save the final state explicitly if desired
# logging.info(f"Saving final model to {OUTPUT_DIR}/final_model...")
# trainer.save_model(OUTPUT_DIR + "/final_model")
# logging.info("Final model saved.")