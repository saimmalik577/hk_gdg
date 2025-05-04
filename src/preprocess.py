import json
import os
from tqdm import tqdm
import torch
from transformers import AutoProcessor, BatchEncoding # Import BatchEncoding for type checking if needed
from PIL import Image
import logging

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Constants ---
BASE_DIR = "data/processed/subset_2000"
IMAGE_DIR = os.path.join(BASE_DIR, "images")
OCR_DIR = os.path.join(BASE_DIR, "ocr")
ANNOTATION_FILE = os.path.join(BASE_DIR, "qas", "train_subset.json")
SAVE_PATH = "data/prepared/tokenized_data.pt" # Output path for the final tokenized data
MODEL_CHECKPOINT = "microsoft/layoutlmv3-base"

# --- Helper Functions ---
def normalize_box(box, width, height):
    """Normalizes bounding box coordinates [xmin, ymin, xmax, ymax] to the 0-1000 range."""
    if width == 0 or height == 0: return [0, 0, 0, 0]
    return [
        int(1000 * (box[0] / width)), int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)), int(1000 * (box[3] / height)),
    ]

def convert_8point_to_4point_bbox(bbox_8points):
    """Converts 8-point list to [xmin, ymin, xmax, ymax]."""
    if not bbox_8points or len(bbox_8points) != 8: return None
    try:
        xs = [bbox_8points[i] for i in range(0, 8, 2)]
        ys = [bbox_8points[i] for i in range(1, 8, 2)]
        xmin, ymin, xmax, ymax = min(xs), min(ys), max(xs), max(ys)
        if xmin >= xmax or ymin >= ymax: return None
        return [xmin, ymin, xmax, ymax]
    except Exception: return None

# --- Main Processing ---
try:
    logging.info(f"Loading annotations from {ANNOTATION_FILE}...")
    with open(ANNOTATION_FILE, "r") as f: annotations = json.load(f)["data"]
    logging.info(f"Loaded {len(annotations)} annotations.")

    logging.info(f"Loading processor: {MODEL_CHECKPOINT}...")
    processor = AutoProcessor.from_pretrained(MODEL_CHECKPOINT, apply_ocr=False)
    logging.info("Processor loaded.")

    processed_samples = []
    skipped_counts = {"image_missing": 0, "ocr_missing": 0, "ocr_parse_error": 0, "word_extraction_fail": 0, "processing_error": 0, "answer_not_found": 0}
    label_map = {"O": 0, "B-ANSWER": 1, "I-ANSWER": 2}

    logging.info("Starting sample processing...")
    for ann in tqdm(annotations[:2000], desc="Processing Samples"):
        img_filename = ann["image"].replace("documents/", "")
        image_path = os.path.join(IMAGE_DIR, img_filename)
        ocr_path = os.path.join(OCR_DIR, img_filename.replace(".png", ".json"))

        try:
            if not os.path.exists(image_path): skipped_counts["image_missing"] += 1; continue
            if not os.path.exists(ocr_path): skipped_counts["ocr_missing"] += 1; continue

            image = Image.open(image_path).convert("RGB")
            width, height = image.size

            with open(ocr_path, "r") as f: ocr_data = json.load(f)

            words = []
            unnormalized_boxes = []
            if "recognitionResults" not in ocr_data or not ocr_data["recognitionResults"]:
                skipped_counts["ocr_parse_error"] += 1; continue
            lines = ocr_data["recognitionResults"][0].get("lines", [])
            if not lines:
                skipped_counts["ocr_parse_error"] += 1; continue

            word_found_in_file = False
            for line in lines:
                for word_info in line.get("words", []):
                    word_text = word_info.get("text")
                    bbox_8points = word_info.get("boundingBox")
                    if word_text and bbox_8points:
                        bbox_4points = convert_8point_to_4point_bbox(bbox_8points)
                        if bbox_4points:
                            words.append(word_text)
                            unnormalized_boxes.append(bbox_4points)
                            word_found_in_file = True

            if not word_found_in_file:
                skipped_counts["word_extraction_fail"] += 1; continue

            boxes = [normalize_box(box, width, height) for box in unnormalized_boxes]

            # --- Process with LayoutLMv3Processor ---
            question = ann["question"]
            encoding_result = processor( # Store result in a temporary variable
                images=image,
                text=question,
                text_pair=words,
                boxes=boxes,
                max_length=512,
                padding="max_length",
                truncation="longest_first",
                return_tensors="pt"
            )

            # --- FIX 3: Get word_ids BEFORE squeezing/rebuilding dict ---
            try:
                 # Get word_ids from the original BatchEncoding object
                 word_ids_list = encoding_result.word_ids(batch_index=0)
            except AttributeError:
                 # Log if word_ids method is missing (shouldn't happen with correct processor output)
                 logging.error(f"Processor output for {img_filename} is type {type(encoding_result)}, missing 'word_ids' method. Skipping.")
                 skipped_counts["processing_error"] += 1
                 continue

            # Now create the plain dict with squeezed tensors
            # Use squeeze(0) to remove the batch dimension specifically
            encoding = {k: v.squeeze(0) for k, v in encoding_result.items()}
            # --- End of FIX 3 ---

            # --- Generate Token Labels (IOB Scheme) ---
            labels = [label_map["O"]] * len(encoding["input_ids"]) # Use length from the dict
            answer_texts = ann.get("answers", [])
            if not answer_texts: processed_samples.append(encoding); continue
            first_answer = answer_texts[0].strip()
            if not first_answer: processed_samples.append(encoding); continue

            # Use the extracted word_ids_list
            word_ids = word_ids_list # Use the list we got earlier

            answer_found_in_tokens = False
            # Iterate through tokens corresponding to the context (words)
            for token_idx, word_id in enumerate(word_ids):
                 if word_id is None: continue
                 current_word_idx = word_id
                 current_reconstructed_text = ""
                 related_token_indices = []
                 temp_word_idx = current_word_idx
                 # Find the first token corresponding to this word_id to avoid re-processing
                 first_token_for_word = min([i for i, wid in enumerate(word_ids) if wid == current_word_idx])
                 if token_idx > first_token_for_word: continue

                 while temp_word_idx is not None and temp_word_idx < len(words):
                     tokens_for_current_word = [i for i, wid in enumerate(word_ids) if wid == temp_word_idx]
                     current_reconstructed_text += words[temp_word_idx] + " "
                     related_token_indices.extend(tokens_for_current_word)
                     normalized_reconstructed = current_reconstructed_text.strip().lower()
                     normalized_answer = first_answer.lower()
                     if normalized_reconstructed == normalized_answer:
                         first_token_of_answer = True
                         unique_related_token_indices = sorted(list(set(related_token_indices)))
                         for rel_token_idx in unique_related_token_indices:
                             if rel_token_idx < len(labels) and word_ids[rel_token_idx] is not None:
                                 labels[rel_token_idx] = label_map["B-ANSWER"] if first_token_of_answer else label_map["I-ANSWER"]
                                 first_token_of_answer = False
                         answer_found_in_tokens = True; break
                     if len(normalized_reconstructed) > len(normalized_answer) + 10: break
                     temp_word_idx += 1
                 if answer_found_in_tokens: break

            if not answer_found_in_tokens:
                skipped_counts["answer_not_found"] += 1
                # continue # Decide if skipping

            # --- Finalize Sample ---
            # 'encoding' is now the dictionary with squeezed tensors
            encoding["labels"] = torch.tensor(labels, dtype=torch.long)
            if "pixel_values" not in encoding or not isinstance(encoding["pixel_values"], torch.Tensor):
                 logging.error(f"pixel_values missing or not a tensor for {img_filename}. Skipping.")
                 skipped_counts["processing_error"] += 1
                 continue

            processed_samples.append(encoding) # Append the dictionary

        except Exception as e:
            skipped_counts["processing_error"] += 1
            logging.error(f"Error processing {img_filename}: {e}", exc_info=False) # Set exc_info=True for full trace
            continue

    # --- Save Processed Data ---
    if processed_samples:
        logging.info(f"\nSaving {len(processed_samples)} processed samples to {SAVE_PATH}...")
        torch.save(processed_samples, SAVE_PATH)
        logging.info("✅ Save complete.")
    else:
        logging.warning("\n⛔ No valid samples were processed and saved. Check logs for errors.")

    # --- Summary ---
    logging.info("\n--- Processing Summary ---")
    logging.info(f"Total Annotations Attempted: {len(annotations[:2000])}")
    logging.info(f"Successfully Processed: {len(processed_samples)}")
    logging.info("Skipped Samples Breakdown:")
    for reason, count in skipped_counts.items():
        if count > 0: logging.info(f"  - {reason}: {count}")

except Exception as e:
     logging.error(f"An critical error occurred during the main process setup: {e}", exc_info=True)