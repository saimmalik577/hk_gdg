#!/usr/bin/env python3
import os
import json
import random
import shutil
from pathlib import Path
import argparse
import logging

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Default Configuration ---
DEFAULT_NUM_TEST_SAMPLES = 500
BASE_RAW_DIR = Path("data/raw") # Root directory for raw data
TRAIN_SUBSET_DIR = Path("data/processed/subset_2000") # Directory of the EXISTING training subset
DEFAULT_OUT_DIR_TEST = Path(f"data/processed/subset_test_{DEFAULT_NUM_TEST_SAMPLES}") # Default output for this script

# --- Raw Data Paths (Derived from BASE_RAW_DIR) ---
RAW_IMG_DIR = BASE_RAW_DIR / "spdocvqa_images"
RAW_OCR_DIR = BASE_RAW_DIR / "spdocvqa_ocr"
RAW_QAS_FILE = BASE_RAW_DIR / "spdocvqa_qas/train_v1.0_withQT.json" # The original full annotation file

# --- Training Subset Info Path (Used to identify and exclude training samples) ---
TRAIN_SUBSET_QAS_FILE = TRAIN_SUBSET_DIR / "qas/train_subset.json"

def create_test_subset(num_samples: int, output_dir: Path):
    """
    Creates a test subset from the raw data, ensuring no overlap with the
    samples listed in the specified training subset annotation file.

    Args:
        num_samples: The target number of samples for the test set.
        output_dir: The directory where the test subset (images, ocr, qas) will be saved.
    """
    logging.info(f"Starting test subset creation: Target={num_samples} samples, Output={output_dir}")

    # --- 1. Validate Inputs and Setup Output Directories ---
    if not BASE_RAW_DIR.is_dir():
        logging.error(f"❌ Raw data directory not found: {BASE_RAW_DIR}")
        return
    if not RAW_IMG_DIR.is_dir():
        logging.error(f"❌ Raw images directory not found: {RAW_IMG_DIR}")
        return
    if not RAW_OCR_DIR.is_dir():
        logging.error(f"❌ Raw OCR directory not found: {RAW_OCR_DIR}")
        return
    if not RAW_QAS_FILE.is_file():
        logging.error(f"❌ Raw Q&A annotation file not found: {RAW_QAS_FILE}")
        return
    if not TRAIN_SUBSET_QAS_FILE.is_file():
        logging.error(f"❌ Training subset Q&A file not found: {TRAIN_SUBSET_QAS_FILE}")
        logging.error("   This file is needed to exclude training samples.")
        return

    test_img_out = output_dir / "images"
    test_ocr_out = output_dir / "ocr"
    test_qas_out = output_dir / "qas"
    test_qas_filename = test_qas_out / f"test_subset_{num_samples}.json"

    try:
        test_img_out.mkdir(parents=True, exist_ok=True)
        test_ocr_out.mkdir(parents=True, exist_ok=True)
        test_qas_out.mkdir(parents=True, exist_ok=True)
        logging.info(f"Ensured output directories exist: {output_dir}")
    except OSError as e:
        logging.error(f"❌ Failed to create output directories: {e}")
        return

    # --- 2. Identify Training Samples ---
    training_image_names = set()
    try:
        with open(TRAIN_SUBSET_QAS_FILE, "r", encoding='utf-8') as f:
            train_subset_data = json.load(f)
        for entry in train_subset_data.get("data", []):
            image_name = Path(entry.get("image", "")).name # Get filename part
            if image_name:
                training_image_names.add(image_name)
        logging.info(f"Identified {len(training_image_names)} unique image names from the training subset: {TRAIN_SUBSET_QAS_FILE}")
    except Exception as e:
        logging.error(f"❌ Failed to read or parse training subset annotation file: {e}")
        return

    # --- 3. Load Full Raw Annotations ---
    try:
        with open(RAW_QAS_FILE, "r", encoding='utf-8') as f:
            full_data = json.load(f)
        all_entries = full_data.get("data", [])
        if not all_entries:
            logging.error("❌ No data entries found in the raw annotation file.")
            return
        logging.info(f"Loaded {len(all_entries)} total entries from raw annotations: {RAW_QAS_FILE}")
    except Exception as e:
        logging.error(f"❌ Failed to read or parse raw annotation file: {e}")
        return

    # Shuffle for random selection
    random.seed(42) # Use a fixed seed for reproducibility if desired
    random.shuffle(all_entries)

    # --- 4. Select Non-Training Samples for Test Set ---
    test_subset_annotations = []
    selected_count = 0
    skipped_training = 0
    skipped_missing_files = 0
    skipped_invalid_entry = 0
    processed_count = 0

    logging.info("Scanning raw data to find suitable test samples...")
    for entry in all_entries:
        processed_count += 1
        if selected_count >= num_samples:
            logging.info(f"Target number of test samples ({num_samples}) reached.")
            break # We have enough samples

        # Basic validation of the entry
        image_path_str = entry.get("image")
        if not image_path_str or not isinstance(image_path_str, str):
            skipped_invalid_entry += 1
            continue # Skip entries without a valid image path string

        image_name = Path(image_path_str).name
        if not image_name:
             skipped_invalid_entry += 1
             continue # Skip if name extraction failed

        # Check if this sample was used for training
        if image_name in training_image_names:
            skipped_training += 1
            continue

        # Determine corresponding OCR filename
        ocr_name = image_name.replace(".jpg", ".json").replace(".png", ".json")
        if ocr_name == image_name: # Basic check if replacement happened
             logging.warning(f"Potential issue: OCR name is same as image name for {image_name}. Skipping.")
             skipped_invalid_entry += 1
             continue

        # Construct full paths for raw image and OCR files
        raw_img_path = RAW_IMG_DIR / image_name
        raw_ocr_path = RAW_OCR_DIR / ocr_name

        # Check if corresponding raw files exist
        img_exists = raw_img_path.is_file()
        ocr_exists = raw_ocr_path.is_file()

        if img_exists and ocr_exists:
            # Copy files to the test subset directory
            try:
                shutil.copy2(raw_img_path, test_img_out / image_name) # copy2 preserves metadata
                shutil.copy2(raw_ocr_path, test_ocr_out / ocr_name)
                test_subset_annotations.append(entry)
                selected_count += 1
                if selected_count % 100 == 0: # Log progress periodically
                     logging.info(f"Selected {selected_count}/{num_samples} samples...")
            except Exception as e:
                 logging.warning(f"⚠️ Could not copy files for {image_name}. Error: {e}")
                 # Attempt to clean up potentially partially copied files
                 (test_img_out / image_name).unlink(missing_ok=True)
                 (test_ocr_out / ocr_name).unlink(missing_ok=True)
                 skipped_missing_files +=1 # Count as skipped due to file issues
        else:
            skipped_missing_files += 1
            # Optional: More verbose logging for missing files if needed
            # logging.debug(f"Skipping {image_name}: Missing image ({img_exists}) or OCR ({ocr_exists})")


    logging.info(f"Finished scanning {processed_count} raw entries.")
    logging.info(f"\n--- Subset Creation Summary ---")
    logging.info(f"Target samples:          {num_samples}")
    logging.info(f"Successfully selected:   {selected_count}")
    logging.info(f"Skipped (in train set):  {skipped_training}")
    logging.info(f"Skipped (missing files): {skipped_missing_files}")
    logging.info(f"Skipped (invalid entry): {skipped_invalid_entry}")


    if selected_count == 0:
        logging.error("❌ No valid test samples could be selected. Check paths and data integrity.")
        return
    elif selected_count < num_samples:
        logging.warning(f"⚠️ Warning: Could only find {selected_count} valid samples for the test set (requested {num_samples}).")

    # --- 5. Save Test Subset Annotation File ---
    output_qas_data = {
        "version": full_data.get("version", "1.0"), # Preserve original version if available
        "split": "test", # Add a split identifier
        "num_samples": selected_count,
        "data": test_subset_annotations
    }
    try:
        with open(test_qas_filename, "w", encoding='utf-8') as f:
            json.dump(output_qas_data, f, indent=2)
        logging.info(f"✅ Successfully created test subset annotation file: {test_qas_filename}")
    except Exception as e:
        logging.error(f"❌ Failed to save test subset annotation file: {e}")
        return

    logging.info(f"--- Test Subset Location ---")
    logging.info(f"   Images: {test_img_out}")
    logging.info(f"   OCR:    {test_ocr_out}")
    logging.info(f"   QAs:    {test_qas_filename}")
    logging.info("--- Script Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a test subset for DocVQA from raw data, ensuring no overlap with a specified training subset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults in help message
    )
    parser.add_argument(
        "-n", "--num_samples",
        type=int,
        default=DEFAULT_NUM_TEST_SAMPLES,
        help="Number of samples desired for the test set."
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        default=None, # Default is dynamically generated
        help="Path to the output directory for the test subset. If None, defaults to data/processed/subset_test_<num_samples>."
    )
    parser.add_argument(
        "--train_subset_dir",
        type=str,
        default=str(TRAIN_SUBSET_DIR), # Convert Path object to string for default
        help="Path to the directory containing the training subset (used to find train_subset.json)."
    )
    parser.add_argument(
        "--raw_data_dir",
        type=str,
        default=str(BASE_RAW_DIR),
        help="Path to the root directory containing raw 'spdocvqa_images', 'spdocvqa_ocr', 'spdocvqa_qas'."
    )

    args = parser.parse_args()

    # --- Update Configuration based on Args ---
    BASE_RAW_DIR = Path(args.raw_data_dir)
    TRAIN_SUBSET_DIR = Path(args.train_subset_dir)

    # Re-derive dependent paths based on potentially updated base dirs
    RAW_IMG_DIR = BASE_RAW_DIR / "spdocvqa_images"
    RAW_OCR_DIR = BASE_RAW_DIR / "spdocvqa_ocr"
    RAW_QAS_FILE = BASE_RAW_DIR / "spdocvqa_qas/train_v1.0_withQT.json"
    TRAIN_SUBSET_QAS_FILE = TRAIN_SUBSET_DIR / "qas/train_subset.json"

    # Determine the final output directory
    output_directory = Path(args.output_dir) if args.output_dir else Path(f"data/processed/subset_test_{args.num_samples}")

    # --- Run the main function ---
    create_test_subset(num_samples=args.num_samples, output_dir=output_directory)