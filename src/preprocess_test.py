#!/usr/bin/env python3
import json
import os
import torch
import argparse
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoProcessor, BatchEncoding
from PIL import Image, ImageFile
import logging

# Allow loading truncated images (useful for some datasets)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
DEFAULT_MODEL_CHECKPOINT = "microsoft/layoutlmv3-base"
DEFAULT_MAX_LENGTH = 512
LABEL_MAP = {"O": 0, "B-ANSWER": 1, "I-ANSWER": 2}
IGNORED_LABEL_ID = -100 # Standard Hugging Face practice

# --- Default Paths (Modify if your structure differs) ---
DEFAULT_SUBSET_DIR = Path("data/processed/subset_test_500")
DEFAULT_OUTPUT_PATH = Path("data/prepared/tokenized_test_data_500.pt")

# --- Helper Functions (Copied *directly* from your training preprocess script) ---

def normalize_box(box, width, height):
    """Normalizes bounding box coordinates [xmin, ymin, xmax, ymax] to the 0-1000 range."""
    if width <= 0 or height <= 0:
        return [0, 0, 0, 0]
    return [
        max(0, min(1000, int(1000 * (box[0] / width)))),
        max(0, min(1000, int(1000 * (box[1] / height)))),
        max(0, min(1000, int(1000 * (box[2] / width)))),
        max(0, min(1000, int(1000 * (box[3] / height)))),
    ]

def convert_8point_to_4point_bbox(bbox_8points):
    """Converts 8-point list [x0, y0, x1, y1, ...] to [xmin, ymin, xmax, ymax]."""
    if not bbox_8points or len(bbox_8points) != 8: return None
    try:
        xs = [float(bbox_8points[i]) for i in range(0, 8, 2)]
        ys = [float(bbox_8points[i]) for i in range(1, 8, 2)]
        xmin, ymin, xmax, ymax = min(xs), min(ys), max(xs), max(ys)
        if xmin >= xmax or ymin >= ymax: return None
        return [xmin, ymin, xmax, ymax]
    except (ValueError, TypeError): return None

def generate_token_labels(encoding_result: BatchEncoding, words: list, answer_texts: list, label_map: dict, ignored_label_id: int):
    """
    Generates IOB token labels based on the first answer, matching the logic
    from the provided training preprocessing script. Handles subword tokenization.
    """
    try:
        word_ids = encoding_result.word_ids(batch_index=0)
        input_ids_length = len(encoding_result["input_ids"][0])
    except Exception as e:
        logging.error(f"Failed to get word_ids from processor output: {e}")
        return None, False

    labels = [label_map["O"]] * input_ids_length
    answer_found_in_tokens = False

    if answer_texts:
        first_answer = answer_texts[0].strip()
        if first_answer:
            normalized_answer = first_answer.lower()

            for current_word_idx in range(len(words)):
                current_reconstructed_text = ""
                span_word_indices = []

                for temp_word_idx in range(current_word_idx, len(words)):
                    current_reconstructed_text += words[temp_word_idx] + " "
                    span_word_indices.append(temp_word_idx)
                    normalized_reconstructed = current_reconstructed_text.strip().lower()

                    if normalized_reconstructed == normalized_answer:
                        first_token_of_answer = True
                        for token_idx, word_id in enumerate(word_ids):
                            if word_id in span_word_indices:
                                if token_idx < input_ids_length:
                                     labels[token_idx] = label_map["B-ANSWER"] if first_token_of_answer else label_map["I-ANSWER"]
                                     first_token_of_answer = False
                        answer_found_in_tokens = True
                        break

                    if len(normalized_reconstructed) > len(normalized_answer) + 15 and \
                       not normalized_answer.startswith(normalized_reconstructed):
                        break
                if answer_found_in_tokens: break

    for token_idx, word_id in enumerate(word_ids):
        if word_id is None:
             if token_idx < input_ids_length:
                labels[token_idx] = ignored_label_id

    return torch.tensor(labels, dtype=torch.long), answer_found_in_tokens


# --- Main Processing Function ---
def preprocess_subset(subset_dir: Path, output_path: Path, model_checkpoint: str, max_length: int):
    """
    Processes a dataset subset (images, OCR, Q&A) into tokenized format with labels,
    mirroring the training data preprocessing script.
    """
    logging.info(f"--- Starting Preprocessing for Evaluation ---")
    logging.info(f"Input Subset Directory: {subset_dir}")
    logging.info(f"Output File: {output_path}")
    logging.info(f"Model Processor: {model_checkpoint}")

    # --- Ensure input directory exists ---
    if not subset_dir.is_dir():
        logging.error(f"❌ Input subset directory not found: {subset_dir}")
        return

    image_dir = subset_dir / "images"
    ocr_dir = subset_dir / "ocr"
    qas_dir = subset_dir / "qas"

    # --- 1. Find and Load Annotations ---
    try:
        annotation_files = list(qas_dir.glob("*.json"))
        if not annotation_files:
            logging.error(f"❌ No annotation JSON file found in {qas_dir}")
            return
        annotation_file = annotation_files[0]
        logging.info(f"Loading annotations from {annotation_file}...")
        with open(annotation_file, "r", encoding='utf-8') as f:
            annotations = json.load(f).get("data", [])
        logging.info(f"Loaded {len(annotations)} annotations.")
        if not annotations: logging.warning("Annotation file empty."); return
    except Exception as e:
        logging.error(f"❌ Failed to load annotation file {annotation_file}: {e}"); return

    # --- 2. Load Processor ---
    try:
        logging.info(f"Loading processor: {model_checkpoint}...")
        processor = AutoProcessor.from_pretrained(model_checkpoint, apply_ocr=False)
        logging.info("Processor loaded successfully.")
    except Exception as e:
        logging.error(f"❌ Failed to load processor '{model_checkpoint}': {e}"); return

    # --- 3. Process Each Sample ---
    processed_samples = []
    skipped_counts = {"image_missing": 0, "ocr_missing": 0, "ocr_parse_error": 0, "word_extraction_fail": 0, "processing_error": 0, "label_gen_error": 0, "answer_not_found": 0}

    logging.info("Starting sample processing loop...")
    for ann in tqdm(annotations, desc="Processing Samples"):
        img_filename = Path(ann.get("image", "")).name
        if not img_filename:
             logging.warning("Skipping annotation with missing 'image' field.")
             skipped_counts["processing_error"] += 1; continue

        image_path = image_dir / img_filename
        ocr_filename = img_filename.replace(".png", ".json").replace(".jpg", ".json")
        ocr_path = ocr_dir / ocr_filename

        try:
            if not image_path.is_file(): skipped_counts["image_missing"] += 1; continue
            if not ocr_path.is_file(): skipped_counts["ocr_missing"] += 1; continue

            image = Image.open(image_path).convert("RGB")
            width, height = image.size
            if width <= 0 or height <= 0: skipped_counts["processing_error"] += 1; continue

            with open(ocr_path, "r", encoding='utf-8') as f: ocr_data = json.load(f)

            words = []
            unnormalized_boxes_4pt = []
            word_found_in_file = False

            if "recognitionResults" in ocr_data and ocr_data["recognitionResults"]:
                lines = ocr_data["recognitionResults"][0].get("lines", [])
                for line in lines:
                    for word_info in line.get("words", []):
                        word_text = word_info.get("text")
                        bbox_8points = word_info.get("boundingBox")
                        if word_text and bbox_8points:
                            bbox_4points = convert_8point_to_4point_bbox(bbox_8points)
                            if bbox_4points:
                                words.append(word_text)
                                unnormalized_boxes_4pt.append(bbox_4points)
                                word_found_in_file = True
            elif "fullTextAnnotation" in ocr_data:
                 logging.debug(f"Using 'fullTextAnnotation' structure for {img_filename}")
                 page = ocr_data.get('fullTextAnnotation', {}).get('pages', [{}])[0]
                 for block in page.get('blocks', []):
                    for paragraph in block.get('paragraphs', []):
                        for word_info in paragraph.get('words', []):
                            word_text = "".join([symbol.get('text', '') for symbol in word_info.get('symbols', [])])
                            vertices = word_info.get('boundingBox', {}).get('vertices', [])
                            if word_text and len(vertices) == 4:
                                bbox_8points_list = []
                                valid_verts = True
                                for vert in vertices:
                                    x, y = vert.get('x'), vert.get('y')
                                    if x is None or y is None: valid_verts=False; break
                                    bbox_8points_list.extend([x, y])
                                if valid_verts:
                                    bbox_4points = convert_8point_to_4point_bbox(bbox_8points_list)
                                    if bbox_4points:
                                        words.append(word_text)
                                        unnormalized_boxes_4pt.append(bbox_4points)
                                        word_found_in_file = True

            if not word_found_in_file:
                skipped_counts["word_extraction_fail"] += 1; continue

            boxes = [normalize_box(box, width, height) for box in unnormalized_boxes_4pt]

            question = ann.get("question", "")
            if not words or not boxes or len(words) != len(boxes):
                 skipped_counts["processing_error"] += 1; continue

            encoding_result = processor(
                images=image,
                text=question,
                text_pair=words,
                boxes=boxes,
                max_length=max_length,
                padding="max_length",
                truncation="longest_first",
                return_tensors="pt"
            )

            answer_texts = ann.get("answers", [])
            labels_tensor, answer_found = generate_token_labels(
                encoding_result, words, answer_texts, LABEL_MAP, IGNORED_LABEL_ID
            )

            if labels_tensor is None:
                skipped_counts["label_gen_error"] += 1; continue

            if not answer_found and answer_texts and answer_texts[0].strip():
                 skipped_counts["answer_not_found"] += 1

            encoding_dict = {k: v.squeeze(0) for k, v in encoding_result.items()}
            encoding_dict["labels"] = labels_tensor

            required_keys = ["input_ids", "attention_mask", "bbox", "pixel_values", "labels"]
            if not all(key in encoding_dict for key in required_keys):
                 logging.error(f"Missing keys after processing {img_filename}. Skipping.")
                 skipped_counts["processing_error"] += 1; continue
            if not isinstance(encoding_dict["pixel_values"], torch.Tensor):
                 logging.error(f"pixel_values invalid for {img_filename}. Skipping.")
                 skipped_counts["processing_error"] += 1; continue

            processed_samples.append(encoding_dict)

        except FileNotFoundError as fnf_err:
             logging.error(f"File not found error: {fnf_err}")
             if 'image_path' in locals() and str(image_path) in str(fnf_err): skipped_counts["image_missing"] += 1
             elif 'ocr_path' in locals() and str(ocr_path) in str(fnf_err): skipped_counts["ocr_missing"] += 1
             else: skipped_counts["processing_error"] += 1
        except json.JSONDecodeError as json_err:
             logging.error(f"Error decoding JSON {ocr_path}: {json_err}")
             skipped_counts["ocr_parse_error"] += 1
        except Image.UnidentifiedImageError:
             logging.error(f"Cannot identify image file {image_path}")
             skipped_counts["processing_error"] += 1
        except Exception as e:
            logging.error(f"Error processing {img_filename}: {e}", exc_info=False)
            skipped_counts["processing_error"] += 1
            continue

    # --- 4. Save Processed Data ---
    if processed_samples:
        logging.info(f"\nSaving {len(processed_samples)} processed test samples to {output_path}...")
        # Ensure parent directory exists
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(processed_samples, output_path)
            logging.info("✅ Save complete.")
        except Exception as save_err:
             logging.error(f"❌ Failed to save output file {output_path}: {save_err}")
    else:
        logging.warning("\n⛔ No valid test samples were processed and saved. Check logs.")

    # --- 5. Summary ---
    logging.info("\n--- Test Data Processing Summary ---")
    logging.info(f"Total Annotations Attempted: {len(annotations)}")
    logging.info(f"Successfully Processed:      {len(processed_samples)}")
    logging.info("Skipped Samples Breakdown:")
    for reason, count in skipped_counts.items():
        if count > 0: logging.info(f"  - {reason}: {count}")

    logging.info("--- Script Finished ---")

# --- Command Line Interface ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess and tokenize a test subset for LayoutLMv3 Document QA, mirroring the training process.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--subset_dir",
        type=Path, # Use Path type directly
        # Set the default value using the constant defined above
        default=DEFAULT_SUBSET_DIR,
        help="Path to the test subset directory containing images/, ocr/, qas/."
    )
    parser.add_argument(
        "--output_path",
        type=Path, # Use Path type directly
        # Set the default value using the constant defined above
        default=DEFAULT_OUTPUT_PATH,
        help="Path to save the final tokenized data file (.pt)."
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default=DEFAULT_MODEL_CHECKPOINT,
        help="Hugging Face model identifier for the processor."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="Maximum sequence length for tokenization."
    )

    args = parser.parse_args()

    # Call the main function with Path objects from args
    preprocess_subset(
        subset_dir=args.subset_dir,
        output_path=args.output_path,
        model_checkpoint=args.model_checkpoint,
        max_length=args.max_length
    )