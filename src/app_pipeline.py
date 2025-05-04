import os
import io
import torch
from pathlib import Path
from PIL import Image, ImageDraw, ImageFile
from transformers import AutoProcessor, LayoutLMv3ForTokenClassification
from google.cloud import documentai # Google Cloud Client Library
from pdf2image import convert_from_path # To handle PDFs
import logging
import numpy as np
import time

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
MODEL_CHECKPOINT_PATH = Path("./models/layoutlmv3_finetuned_smartdoc_qa/checkpoint-1500") # Best checkpoint
PROCESSOR_SOURCE = "microsoft/layoutlmv3-base" # Base model identifier for processor
CREDENTIALS_FILE_PATH = Path(".keys/edtech-vllm-app-b6ae35847a58.json")

# Determine device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
logging.info(f"Using device: {DEVICE}")

# --- Google Document AI Config (Hardcoded) ---
GCP_PROJECT_ID = "edtech-vllm-app"
GCP_LOCATION = "eu"
GCP_PROCESSOR_ID = "67bd191d8754344a".strip() # Form Parser ID

logging.info(f"Using GCP Project ID: {GCP_PROJECT_ID}")
logging.info(f"Using GCP Location: {GCP_LOCATION}")
logging.info(f"Using GCP Processor ID: {GCP_PROCESSOR_ID}")

# --- Constants ---
LABEL_MAP = {"O": 0, "B-ANSWER": 1, "I-ANSWER": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
IGNORED_LABEL_ID = -100

# --- Load Model and Processor Globally ---
MODEL_LOADED = False
try:
    logging.info(f"Loading processor from: {PROCESSOR_SOURCE}")
    processor = AutoProcessor.from_pretrained(PROCESSOR_SOURCE, apply_ocr=False, trust_remote_code=True)
    logging.info(f"Loading model from: {MODEL_CHECKPOINT_PATH}")
    model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_CHECKPOINT_PATH)
    model.to(DEVICE)
    model.eval()
    logging.info(f"Model and processor loaded successfully on device: {DEVICE}")
    MODEL_LOADED = True
except Exception as e:
    logging.error(f"❌ Critical error loading model or processor: {e}", exc_info=True)
    processor = None; model = None

# --- Helper Functions ---

def normalize_box(box, width, height):
    if width <= 0 or height <= 0: return [0, 0, 0, 0]
    return [
        max(0, min(1000, int(1000 * (box[0] / width)))),
        max(0, min(1000, int(1000 * (box[1] / height)))),
        max(0, min(1000, int(1000 * (box[2] / width)))),
        max(0, min(1000, int(1000 * (box[3] / height)))),
    ]

def docai_normalized_vertices_to_absolute_bbox(vertices, page_width, page_height):
    if not vertices or page_width <= 0 or page_height <= 0: return None
    xs = [v.x * page_width for v in vertices if v.x is not None]
    ys = [v.y * page_height for v in vertices if v.y is not None]
    if not xs or not ys: return None
    try:
        xmin, ymin = min(xs), min(ys); xmax, ymax = max(xs), max(ys)
        if xmin >= xmax or ymin >= ymax: return None
        return [xmin, ymin, xmax, ymax]
    except ValueError: return None

def process_document_google_docai(project_id: str, location: str, processor_id: str, file_path: str, mime_type: str) -> documentai.Document | None:
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        abs_credentials_path = CREDENTIALS_FILE_PATH.resolve()
        if abs_credentials_path.is_file():
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(abs_credentials_path)
            logging.info(f"Set GOOGLE_APPLICATION_CREDENTIALS programmatically to: {abs_credentials_path}")
        else:
            logging.error(f"Credential file not found: {abs_credentials_path}"); return None
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"): logging.error("GOOGLE_APPLICATION_CREDENTIALS could not be set."); return None
    if not all([project_id, location, processor_id]): logging.error("GCP Config missing."); return None
    try:
        opts = {}
        if location != "us": opts["api_endpoint"] = f"{location}-documentai.googleapis.com"
        client = documentai.DocumentProcessorServiceClient(client_options=opts)
        name = client.processor_path(project_id, location, processor_id)
        with open(file_path, "rb") as doc_file: content = doc_file.read()
        if not content: logging.error(f"File empty: {file_path}"); return None
        raw_document = documentai.RawDocument(content=content, mime_type=mime_type)
        request = documentai.ProcessRequest(name=name, raw_document=raw_document)
        logging.info(f"Sending document ({mime_type}) to DocAI: {name}"); start_time = time.time()
        result = client.process_document(request=request)
        end_time = time.time(); logging.info(f"DocAI processing took {end_time - start_time:.2f} sec.")
        return result.document
    except Exception as e: logging.error(f"❌ Error during DocAI API call: {e}", exc_info=True); return None

# --- !!! UPDATED OCR EXTRACTION FUNCTION using page.tokens !!! ---
def extract_ocr_data_from_docai(document: documentai.Document):
    words, unnormalized_boxes_4pt = [], []
    page_width, page_height = 0, 0
    if not document or not document.pages:
        logging.warning("DocAI response invalid/empty."); return words, unnormalized_boxes_4pt, 0, 0
    page = document.pages[0]
    page_width, page_height = page.dimension.width, page.dimension.height
    full_text = document.text
    if page_width <= 0 or page_height <= 0 or not full_text:
        logging.warning(f"Invalid page dimensions({page_width}x{page_height}) or missing text."); return words, unnormalized_boxes_4pt, 0, 0
    logging.info(f"Extracting OCR from Page 1 ({page_width}x{page_height}) - Iterating via Tokens")
    for token in page.tokens:
        layout = token.layout
        if layout and layout.bounding_poly and layout.bounding_poly.normalized_vertices:
            bbox_4pt_abs = docai_normalized_vertices_to_absolute_bbox(layout.bounding_poly.normalized_vertices, page_width, page_height)
            token_text = ""
            if layout.text_anchor and layout.text_anchor.text_segments:
                try:
                    start_index = int(layout.text_anchor.text_segments[0].start_index or 0)
                    end_index = int(layout.text_anchor.text_segments[0].end_index or 0)
                    if 0 <= start_index <= end_index <= len(full_text):
                         token_text = full_text[start_index:end_index]
                    else: logging.warning(f"Invalid text anchor: [{start_index}:{end_index}]")
                except (ValueError, IndexError, TypeError) as e: logging.warning(f"Error processing text anchor: {e}")
            if token_text and bbox_4pt_abs:
                words.append(token_text); unnormalized_boxes_4pt.append(bbox_4pt_abs)
    if not words: logging.warning("No tokens/words extracted using token iteration.")
    logging.info(f"Extracted {len(words)} tokens (words) from Page 1 OCR.")
    return words, unnormalized_boxes_4pt, page_width, page_height
# --- !!! END OF UPDATED FUNCTION !!! ---

def extract_answer_details(predictions_ids, encoding, words, unnormalized_boxes_4pt):
    answer_indices = []
    attention_mask = encoding["attention_mask"].squeeze().tolist()
    for idx, pred_id in enumerate(predictions_ids):
        if attention_mask[idx] == 1:
            pred_label = ID_TO_LABEL.get(pred_id)
            if pred_label in ["B-ANSWER", "I-ANSWER"]: answer_indices.append(idx)
    if not answer_indices: return "No answer found.", None
    try: token_to_word_map = encoding.word_ids(0)
    except Exception as e: logging.error(f"Error getting word_ids: {e}"); return "Error mapping tokens.", None
    answer_word_indices_set = set()
    for token_idx in answer_indices:
        if token_idx >= len(token_to_word_map): continue
        word_idx = token_to_word_map[token_idx]
        if word_idx is not None and word_idx < len(words): answer_word_indices_set.add(word_idx)
    if not answer_word_indices_set: return "Answer tokens mapped, but not to valid words.", None
    sorted_word_indices = sorted(list(answer_word_indices_set))
    answer_words_list, answer_bboxes_list = [], []
    for word_idx in sorted_word_indices:
        if word_idx < len(words) and word_idx < len(unnormalized_boxes_4pt):
             answer_words_list.append(words[word_idx]); answer_bboxes_list.append(unnormalized_boxes_4pt[word_idx])
        else: logging.warning(f"Word index {word_idx} out of bounds.")
    if not answer_words_list or not answer_bboxes_list: return "No valid words/boxes for answer indices.", None
    min_x=min(b[0] for b in answer_bboxes_list); min_y=min(b[1] for b in answer_bboxes_list)
    max_x=max(b[2] for b in answer_bboxes_list); max_y=max(b[3] for b in answer_bboxes_list)
    final_answer_box = [min_x, min_y, max_x, max_y]
    answer_text = " ".join(answer_words_list)
    logging.info(f"Extracted Answer: '{answer_text}'"); logging.info(f"Extracted BBox: {final_answer_box}")
    return answer_text, final_answer_box

# --- Core Pipeline Function ---
def run_pipeline(doc_file_path: str, question: str):
    if not MODEL_LOADED: return "Error: Model/Processor not loaded.", None, None, None
    logging.info(f"--- Running Pipeline ---"); logging.info(f"Document: {doc_file_path}"); logging.info(f"Question: {question}")
    file_path = Path(doc_file_path)
    if not file_path.is_file(): return f"Error: File not found: {doc_file_path}", None, None, None
    file_extension = file_path.suffix.lower()
    page_image_pil, mime_type = None, None
    try: # 1. Load/Convert Doc
        if file_extension == ".pdf":
            logging.info("Converting PDF..."); images = convert_from_path(file_path, first_page=1, last_page=1, fmt='png', dpi=200)
            if not images: raise ValueError("pdf2image failed."); page_image_pil = images[0].convert("RGB"); mime_type = "application/pdf"; logging.info("PDF OK.")
        elif file_extension in [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"]:
            logging.info("Loading image..."); page_image_pil = Image.open(file_path).convert("RGB")
            img_format = page_image_pil.format or file_extension.lstrip('.').upper()
            if img_format=="JPG": img_format="JPEG" # Normalize JPG->JPEG
            mime_type = Image.MIME.get(img_format, f"image/{img_format.lower()}")
            logging.info(f"Image OK (Format: {img_format}, MIME: {mime_type}).")
        else: return f"Error: Unsupported type '{file_extension}'.", None, None, None
    except Exception as e: logging.error(f"Error loading doc: {e}", exc_info=True); return f"Error processing file: {e}", None, None, None

    # 2. DocAI OCR
    docai_document = process_document_google_docai(GCP_PROJECT_ID, GCP_LOCATION, GCP_PROCESSOR_ID, str(file_path), mime_type)
    if not docai_document: return "Error: Google Document AI failed.", None, None, page_image_pil

    # 3. Extract OCR (Uses updated function)
    words, unnormalized_boxes_4pt, _, _ = extract_ocr_data_from_docai(docai_document)
    if not words or not unnormalized_boxes_4pt: return "Warning: No words/boxes extracted via OCR.", None, None, page_image_pil

    # 4. Get Image Dims
    pil_page_width, pil_page_height = page_image_pil.size
    if pil_page_width <= 0 or pil_page_height <= 0: return "Error: Invalid PIL dimensions.", None, None, page_image_pil

    # 5. Normalize Boxes
    normalized_boxes = [normalize_box(box, pil_page_width, pil_page_height) for box in unnormalized_boxes_4pt]

    # 6. Prepare Model Input
    try:
        encoding = processor(images=page_image_pil, text=question, text_pair=words, boxes=normalized_boxes, max_length=512, padding="max_length", truncation="longest_first", return_tensors="pt")
        prepared_batch = {k: v.to(DEVICE) for k, v in encoding.items()}
    except Exception as e: logging.error(f"Error during encoding: {e}", exc_info=True); return f"Error preparing input: {e}", None, None, page_image_pil

    # 7. Run Inference
    try:
        logging.info("Running inference..."); start_time = time.time()
        with torch.no_grad(): outputs = model(**prepared_batch)
        end_time = time.time(); logging.info(f"Inference took {end_time - start_time:.2f} sec.")
        predictions = torch.argmax(outputs.logits, dim=2); predictions_ids = predictions.squeeze().cpu().tolist()
    except Exception as e: logging.error(f"Error during inference: {e}", exc_info=True); return f"Error during inference: {e}", None, None, page_image_pil

    # 8. Extract Answer
    answer_text, answer_bbox = extract_answer_details(predictions_ids, encoding, words, unnormalized_boxes_4pt)

    # 9. Return
    status = "Success" if answer_text != "No answer found." and answer_bbox is not None else "Completed - No answer found."
    return status, answer_text, answer_bbox, page_image_pil


# --- Example Usage ---
if __name__ == "__main__":
    logging.info("Testing pipeline directly...")
    test_file = "assets/sample_doc.jpg" # <<<--- CHANGE TO YOUR TEST FILE NAME
    test_question = "What is the equity owner type?" # <<<--- CHANGE TO A RELEVANT QUESTION
    test_file_path = Path(test_file)
    if MODEL_LOADED and test_file_path.exists():
        if not CREDENTIALS_FILE_PATH.resolve().is_file():
            logging.error(f"Credentials file specified ({CREDENTIALS_FILE_PATH}) not found.")
        else:
            status, ans_text, ans_box, img = run_pipeline(str(test_file_path), test_question)
            print("\n--- Pipeline Test Results ---")
            print(f"Status: {status}"); print(f"Answer Text: {ans_text}"); print(f"Answer BBox: {ans_box}")
            if img and ans_box:
                draw = ImageDraw.Draw(img); draw.rectangle([int(c) for c in ans_box], outline="red", width=3)
                img.save("test_output_with_box.png"); print("Saved output: test_output_with_box.png")
            elif img: img.save("test_output_no_box.png"); print("Saved output: test_output_no_box.png")
    elif not MODEL_LOADED: print("Model not loaded.");
    else: print(f"Test file '{test_file}' not found in CWD ({Path.cwd()}).")