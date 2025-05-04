#!/usr/bin/env python3
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, LayoutLMv3ForTokenClassification
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from tqdm.auto import tqdm
import logging
import numpy as np
import sys
from PIL import ImageFile # Added just in case

# Allow loading truncated images (useful for some datasets)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- !!! HARDCODED CONFIGURATION !!! ---
# --- Ensure these paths are correct for your project structure ---

# <<<--- MODIFIED THIS LINE --->>>
MODEL_PATH = Path("./models/layoutlmv3_finetuned_smartdoc_qa/checkpoint-1500") # <<<--- Pointing to Epoch 1 checkpoint

TEST_DATA_PATH = Path("data/prepared/tokenized_test_data_500.pt")
EVAL_BATCH_SIZE = 4 # Adjust based on your Mac's memory (4 or 8 is usually safe)
# --- End of Hardcoded Configuration ---


# --- Constants (Should match preprocessing) ---
LABEL_MAP = {"O": 0, "B-ANSWER": 1, "I-ANSWER": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
IGNORED_LABEL_ID = -100 # Standard Hugging Face practice


# --- Dataset Class ---
class TokenizedDataset(Dataset):
    """Simple Dataset wrapper for the pre-tokenized data."""
    def __init__(self, tokenized_data):
        self.data = tokenized_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# --- Evaluation Function ---
def evaluate_model(model_path: Path, test_data_path: Path, batch_size: int):
    """Loads a fine-tuned model and evaluates it on the provided tokenized test data."""
    logging.info(f"--- Starting Evaluation ---")
    logging.info(f"Using Hardcoded Model Path: {model_path}") # Will now show checkpoint-125
    logging.info(f"Using Hardcoded Test Data Path: {test_data_path}")
    logging.info(f"Using Hardcoded Batch Size: {batch_size}")

    # --- 1. Setup Device ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logging.info(f"Using device: {device}")

    # --- 2. Load Model ---
    try:
        logging.info("Loading model...")
        model = LayoutLMv3ForTokenClassification.from_pretrained(model_path)
        model.to(device)
        model.eval()
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"❌ Failed to load model from {model_path}: {e}", exc_info=True)
        return

    # --- 3. Load Test Data ---
    try:
        logging.info(f"Loading test data from {test_data_path}...")
        tokenized_test_samples = torch.load(test_data_path, map_location='cpu')
        if not tokenized_test_samples:
            logging.error("❌ Loaded test data is empty.")
            return
        logging.info(f"Loaded {len(tokenized_test_samples)} test samples.")
        test_dataset = TokenizedDataset(tokenized_test_samples)
    except Exception as e:
        logging.error(f"❌ Failed to load or process test data: {e}", exc_info=True)
        return

    # --- 4. Create DataLoader ---
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    logging.info(f"Created DataLoader with batch size {batch_size}.")

    # --- 5. Run Inference ---
    all_predictions_ids = []
    all_true_labels_ids = []

    logging.info("Running inference on the test set...")
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            try:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                bbox = batch['bbox'].to(device)
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)
            except KeyError as e:
                logging.error(f"❌ Missing key in batch data: {e}. Check the content of {test_data_path}.")
                return
            except Exception as e:
                 logging.error(f"❌ Error moving batch to device: {e}", exc_info=True)
                 return

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
                pixel_values=pixel_values
            )
            predictions = torch.argmax(outputs.logits, dim=2)
            all_predictions_ids.extend(predictions.cpu().numpy())
            all_true_labels_ids.extend(labels.cpu().numpy())

    logging.info("Inference complete.")

    # --- 6. Post-process for Seqeval ---
    logging.info("Post-processing predictions for seqeval...")
    true_labels_seqeval = []
    predictions_seqeval = []

    for true_sample_ids, pred_sample_ids in zip(all_true_labels_ids, all_predictions_ids):
        true_labels = []
        pred_labels = []
        for true_id, pred_id in zip(true_sample_ids, pred_sample_ids):
            if true_id != IGNORED_LABEL_ID:
                true_labels.append(ID_TO_LABEL.get(true_id, "O"))
                pred_labels.append(ID_TO_LABEL.get(pred_id, "O"))
        if true_labels:
            true_labels_seqeval.append(true_labels)
            predictions_seqeval.append(pred_labels)

    logging.info(f"Formatted {len(true_labels_seqeval)} samples for seqeval.")

    # --- 7. Calculate and Print Metrics ---
    if not true_labels_seqeval or not predictions_seqeval:
         logging.warning("⚠️ No valid labels found after filtering. Cannot calculate metrics.")
         return

    logging.info("\n--- Evaluation Results ---")
    try:
        if not isinstance(true_labels_seqeval[0], list):
             logging.error("❌ Internal error: true_labels_seqeval is not a list of lists.")
             return
        if not isinstance(predictions_seqeval[0], list):
             logging.error("❌ Internal error: predictions_seqeval is not a list of lists.")
             return

        report = classification_report(true_labels_seqeval, predictions_seqeval, digits=4)
        print("\nClassification Report (seqeval):\n")
        print(report)

        overall_precision = precision_score(true_labels_seqeval, predictions_seqeval)
        overall_recall = recall_score(true_labels_seqeval, predictions_seqeval)
        overall_f1 = f1_score(true_labels_seqeval, predictions_seqeval)

        print("\nOverall Metrics (Micro Average):")
        print(f"  Precision: {overall_precision:.4f}")
        print(f"  Recall:    {overall_recall:.4f}")
        print(f"  F1-Score:  {overall_f1:.4f}")

    except Exception as e:
        logging.error(f"❌ Error calculating or printing metrics using seqeval: {e}", exc_info=True)

    logging.info("--- Evaluation Finished ---")


# --- Main execution block ---
if __name__ == "__main__":
    # --- Validate Hardcoded Paths Before Running ---
    paths_ok = True
    # Use the updated MODEL_PATH variable
    if not MODEL_PATH.is_dir():
        logging.error(f"❌ Configured Model Path does not exist or is not a directory: {MODEL_PATH}")
        paths_ok = False
    if not TEST_DATA_PATH.is_file():
        logging.error(f"❌ Configured Test Data Path does not exist or is not a file: {TEST_DATA_PATH}")
        paths_ok = False

    if paths_ok:
        evaluate_model(
            model_path=MODEL_PATH, # Pass the updated path
            test_data_path=TEST_DATA_PATH,
            batch_size=EVAL_BATCH_SIZE
        )
    else:
        logging.error("Please fix the hardcoded paths in the script before running.")
        sys.exit(1)