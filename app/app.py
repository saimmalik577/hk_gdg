# app.py
import os
import io # Added import for io.BytesIO
import torch
from pathlib import Path
from PIL import Image, ImageDraw, ImageFile, UnidentifiedImageError, ImageFont
from transformers import AutoProcessor, LayoutLMv3ForTokenClassification
from google.cloud import documentai # Google Cloud Client Library
# Make sure pdf2image and its dependency (poppler) are installed
try:
    from pdf2image import convert_from_path, pdfinfo_from_path
    PDF2IMAGE_INSTALLED = True
except ImportError:
    PDF2IMAGE_INSTALLED = False
    print("WARNING: pdf2image not found. PDF processing will be disabled. Install with 'pip install pdf2image' and ensure poppler is installed.")

import logging
import numpy as np
import time
import gradio as gr # Import Gradio

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
MODEL_CHECKPOINT_PATH = Path("./models/layoutlmv3_finetuned_smartdoc_qa/checkpoint-1500")
PROCESSOR_SOURCE = "microsoft/layoutlmv3-base"
CREDENTIALS_FILE_PATH = Path(".keys/edtech-vllm-app-b6ae35847a58.json")

# Determine device
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = torch.device("mps")
    logging.info("MPS device detected and available.")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
logging.info(f"Using device: {DEVICE}")

# --- Google Document AI Config ---
GCP_PROJECT_ID = "edtech-vllm-app"
GCP_LOCATION = "eu"
GCP_PROCESSOR_ID = "67bd191d8754344a".strip()

logging.info(f"Using GCP Project ID: {GCP_PROJECT_ID}, Location: {GCP_LOCATION}, Processor ID: {GCP_PROCESSOR_ID}")

# --- Constants ---
LABEL_MAP = {"O": 0, "B-ANSWER": 1, "I-ANSWER": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
IGNORED_LABEL_ID = -100
MAX_SEQ_LENGTH = 512

# --- Load Model and Processor Globally ---
def load_model_and_processor():
    """Loads the model and processor once."""
    try:
        if not MODEL_CHECKPOINT_PATH.exists():
            logging.error(f"❌ Model checkpoint directory not found: {MODEL_CHECKPOINT_PATH}")
            return None, None, False
        logging.info(f"Loading processor from: {PROCESSOR_SOURCE}")
        processor = AutoProcessor.from_pretrained(PROCESSOR_SOURCE, apply_ocr=False, trust_remote_code=True)
        logging.info(f"Loading model from: {MODEL_CHECKPOINT_PATH}")
        model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_CHECKPOINT_PATH)
        model.to(DEVICE)
        model.eval()
        logging.info(f"✅ Model and processor loaded successfully on device: {DEVICE}")
        return model, processor, True
    except Exception as e:
        logging.error(f"❌ Critical error loading model or processor: {e}", exc_info=True)
        return None, None, False

model, processor, MODEL_LOADED = load_model_and_processor()


# --- Helper Functions ---
def normalize_box(box, width, height):
    """Normalizes box [xmin, ymin, xmax, ymax] to 0-1000 range, clamping values."""
    if width <= 0 or height <= 0: return [0, 0, 0, 0]
    return [
        max(0, min(1000, int(1000 * (box[0] / width)))),
        max(0, min(1000, int(1000 * (box[1] / height)))),
        max(0, min(1000, int(1000 * (box[2] / width)))),
        max(0, min(1000, int(1000 * (box[3] / height)))),
    ]

def docai_normalized_vertices_to_absolute_bbox(vertices, page_width, page_height):
    """Converts DocAI normalized vertices to absolute pixel [xmin, ymin, xmax, ymax]."""
    if not vertices or page_width <= 0 or page_height <= 0: return None
    xs = [v.x * page_width for v in vertices if v.x is not None]
    ys = [v.y * page_height for v in vertices if v.y is not None]
    if not xs or not ys: return None
    try:
        xmin, ymin = min(xs), min(ys)
        xmax, ymax = max(xs), max(ys)
        if xmin > xmax or ymin > ymax:
             logging.warning(f"Degenerate box calculated (min>max): {[xmin, ymin, xmax, ymax]}")
             return None
        return [xmin, ymin, xmax, ymax]
    except ValueError: return None

def set_gcp_credentials():
    """Sets the GOOGLE_APPLICATION_CREDENTIALS environment variable if not already set."""
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        abs_credentials_path = CREDENTIALS_FILE_PATH.resolve()
        if abs_credentials_path.is_file():
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(abs_credentials_path)
            return True
        else:
            logging.error(f"❌ GCP Credential file not found: {abs_credentials_path}")
            return False
    return True

def process_document_google_docai(project_id: str, location: str, processor_id: str, file_path: str, mime_type: str) -> documentai.Document | None:
    """Sends a document to Google Document AI and returns the processed Document object."""
    if not set_gcp_credentials(): return None
    if not all([project_id, location, processor_id]): logging.error("GCP Config missing."); return None
    try:
        opts = {}
        api_endpoint = f"{location}-documentai.googleapis.com"
        if location == "us": api_endpoint = "documentai.googleapis.com"
        opts["api_endpoint"] = api_endpoint
        logging.info(f"Using DocAI API endpoint: {api_endpoint}")
        client = documentai.DocumentProcessorServiceClient(client_options=opts)
        name = client.processor_path(project_id, location, processor_id)
        with open(file_path, "rb") as doc_file: content = doc_file.read()
        if not content: logging.error(f"File content is empty: {file_path}"); return None
        raw_document = documentai.RawDocument(content=content, mime_type=mime_type)
        request = documentai.ProcessRequest(name=name, raw_document=raw_document)
        logging.info(f"Sending document ({mime_type}) to DocAI Processor: {processor_id}"); start_time = time.time()
        result = client.process_document(request=request)
        end_time = time.time(); logging.info(f"✅ DocAI processing finished in {end_time - start_time:.2f} sec.")
        return result.document
    except Exception as e:
        logging.error(f"❌ Error during DocAI API call: {e}", exc_info=True)
        if "permission denied" in str(e).lower(): logging.error("Hint: Check GCP credentials/IAM permissions.")
        elif "could not find processor" in str(e).lower(): logging.error("Hint: Check Processor ID/Location.")
        return None

def extract_ocr_data_from_docai(document: documentai.Document):
    """Extracts words and UNNORMALIZED [xmin, ymin, xmax, ymax] boxes from DocAI Document."""
    words, unnormalized_boxes_4pt = [], []
    if not document or not document.pages:
        logging.warning("DocAI response document or pages missing.")
        return words, unnormalized_boxes_4pt
    page = document.pages[0] # Process first page only
    page_width, page_height = page.dimension.width, page.dimension.height
    full_text = document.text
    if page_width <= 0 or page_height <= 0:
        logging.warning(f"Invalid page dimensions ({page_width}x{page_height}).")
        return words, unnormalized_boxes_4pt
    logging.info(f"Extracting OCR from Page 1 ({page_width}x{page_height}) using page.tokens")
    for token in page.tokens:
        layout = token.layout
        if layout and layout.bounding_poly and layout.bounding_poly.normalized_vertices:
            bbox_4pt_abs = docai_normalized_vertices_to_absolute_bbox(layout.bounding_poly.normalized_vertices, page_width, page_height)
            token_text = ""
            if layout.text_anchor and layout.text_anchor.text_segments:
                try:
                    segment = layout.text_anchor.text_segments[0]
                    start_index = int(segment.start_index or 0); end_index = int(segment.end_index or 0)
                    if 0 <= start_index <= end_index <= len(full_text): token_text = full_text[start_index:end_index]
                    else: logging.warning(f"Invalid text anchor indices: [{start_index}:{end_index}]")
                except Exception as e: logging.warning(f"Error processing text anchor: {e}")
            if token_text and bbox_4pt_abs:
                words.append(token_text); unnormalized_boxes_4pt.append(bbox_4pt_abs)
    if not words: logging.warning("⚠️ No words/tokens extracted.")
    logging.info(f"Extracted {len(words)} tokens (words) from Page 1 OCR.")
    return words, unnormalized_boxes_4pt

def extract_answer_details(predictions_ids, encoding, words, unnormalized_boxes_4pt):
    """Extracts answer text and calculates the final bbox from token predictions."""
    answer_indices = []
    attention_mask = encoding["attention_mask"].squeeze().cpu().tolist()
    if isinstance(predictions_ids, torch.Tensor): predictions_ids = predictions_ids.squeeze().cpu().tolist()
    if not isinstance(predictions_ids, list) or (predictions_ids and not isinstance(predictions_ids[0], int)):
        logging.error(f"Unexpected format for predictions_ids: {type(predictions_ids)}")
        return "Error processing model predictions.", None
    for idx, pred_id in enumerate(predictions_ids):
        if idx >= len(attention_mask): break
        if attention_mask[idx] == 1:
             label = ID_TO_LABEL.get(pred_id, "O")
             if label in ["B-ANSWER", "I-ANSWER"]: answer_indices.append(idx)
    if not answer_indices: return "No answer found.", None
    try:
        if hasattr(encoding, 'word_ids') and callable(encoding.word_ids): token_to_word_map = encoding.word_ids(batch_index=0)
        else: logging.error("Encoding missing 'word_ids'."); return "Error mapping tokens (encoding issue).", None
        if token_to_word_map is None: logging.error("word_ids returned None."); return "Error mapping tokens (None result).", None
    except Exception as e: logging.error(f"Error getting word_ids: {e}", exc_info=True); return "Error mapping tokens.", None
    answer_word_indices_set = set()
    for token_idx in answer_indices:
        if token_idx >= len(token_to_word_map): continue
        word_idx = token_to_word_map[token_idx]
        if word_idx is not None and 0 <= word_idx < len(words): answer_word_indices_set.add(word_idx)
    if not answer_word_indices_set: return "Answer found, but couldn't map to original words.", None
    sorted_word_indices = sorted(list(answer_word_indices_set))
    answer_words_list, answer_bboxes_list = [], []
    for word_idx in sorted_word_indices:
        if word_idx < len(words) and word_idx < len(unnormalized_boxes_4pt):
            answer_words_list.append(words[word_idx]); answer_bboxes_list.append(unnormalized_boxes_4pt[word_idx])
        else: logging.warning(f"Reconstruction index error: word_idx {word_idx}.")
    if not answer_words_list or not answer_bboxes_list: return "Error reconstructing answer.", None
    try:
        min_x = min(b[0] for b in answer_bboxes_list); min_y = min(b[1] for b in answer_bboxes_list)
        max_x = max(b[2] for b in answer_bboxes_list); max_y = max(b[3] for b in answer_bboxes_list)
        final_answer_box_pixels = [min_x, min_y, max_x, max_y]
    except ValueError: logging.error("ValueError calculating final bbox."); return "Error calculating final box.", None
    answer_text = " ".join(answer_words_list)
    logging.info(f"✅ Extracted Answer: '{answer_text}'")
    logging.info(f"✅ Extracted BBox (Pixels): {final_answer_box_pixels}")
    return answer_text, final_answer_box_pixels

def draw_box_on_image(image: Image.Image | None, box: list | None, color="red", width=3):
    """Draws a bounding box on a *copy* of the image."""
    if image is None: return None
    img_copy = image.copy()
    if box is None or not isinstance(box, (list, tuple)) or len(box) != 4: return img_copy
    draw = ImageDraw.Draw(img_copy)
    try:
        draw_box = [float(c) for c in box]
        if draw_box[0] >= draw_box[2] or draw_box[1] >= draw_box[3]: return img_copy # Skip degenerate
        draw.rectangle(draw_box, outline=color, width=width)
    except Exception as e: logging.error(f"Failed to draw rectangle {box}: {e}", exc_info=True)
    return img_copy

def create_placeholder_image(width=400, height=100, message="Image Unavailable\n(Processing Error)"):
     """Creates a placeholder PIL image with text."""
     try:
         img = Image.new('RGB', (width, height), color=(211, 211, 211)) # Light gray
         draw = ImageDraw.Draw(img)
         try:
             font = ImageFont.load_default() # Safest fallback font
             text_bbox = draw.textbbox((0, 0), message, font=font)
             text_width = text_bbox[2] - text_bbox[0]; text_height = text_bbox[3] - text_bbox[1]
             text_x = max(0, (width - text_width) / 2); text_y = max(0, (height - text_height) / 2)
             draw.text((text_x, text_y), message, fill=(0,0,0), font=font, align="center")
         except Exception as font_err:
             logging.warning(f"Could not use default font: {font_err}")
             draw.text((10, 10), message, fill=(0,0,0)) # Backup
         return img
     except Exception as placeholder_err:
          logging.error(f"Failed to create placeholder image: {placeholder_err}")
          return None

# --- Gradio Core Logic Function ---
def predict_answer(doc_file_obj, question_str):
    """Main function called by Gradio to process inputs and return outputs."""
    start_overall_time = time.time()
    if not MODEL_LOADED: return "❌ Error: Model not loaded. Check server logs.", create_placeholder_image(message="Model Load Error")
    if doc_file_obj is None: return "❌ Error: Please upload a document.", create_placeholder_image(message="No Document Uploaded")
    if not question_str or not question_str.strip(): return "❌ Error: Please enter a question.", create_placeholder_image(message="No Question Entered")

    doc_file_path = doc_file_obj.name
    logging.info(f"--- Gradio Request ---")
    logging.info(f"Processing File: {doc_file_path}")
    logging.info(f"Question: '{question_str}'")

    page_image_pil = None
    final_answer_text = "Processing failed."
    final_output_image = None
    final_answer_box_pixels = None
    original_doc_path_for_docai = doc_file_path

    try:
        # --- 2a. Load/Convert Document & Determine MIME type ---
        file_path = Path(doc_file_path)
        file_extension = file_path.suffix.lower()
        mime_type = None
        logging.info(f"Loading/Converting document: {file_path}")
        if file_extension == ".pdf":
            if not PDF2IMAGE_INSTALLED: raise ImportError("PDF processing requires pdf2image and poppler.")
            images = convert_from_path(file_path, first_page=1, last_page=1, fmt='png', dpi=200)
            if not images: raise ValueError("pdf2image failed.")
            page_image_pil = images[0].convert("RGB")
            mime_type = "application/pdf"
            logging.info("✅ PDF converted; will send original PDF to DocAI.")
        elif file_extension in [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"]:
            try:
                page_image_pil = Image.open(file_path).convert("RGB")
                img_format = page_image_pil.format or file_extension.lstrip('.').upper()
                if img_format=="JPG": img_format="JPEG"
                mime_map = {"JPEG": "image/jpeg", "PNG": "image/png", "TIFF": "image/tiff", "BMP": "image/bmp", "GIF": "image/gif"}
                mime_type = mime_map.get(img_format, f"image/{img_format.lower()}")
                if mime_type == f"image/{img_format.lower()}": logging.warning(f"Using potentially non-standard MIME type: {mime_type}")
                logging.info(f"✅ Image loaded (Format: {img_format}, MIME: {mime_type}).")
            except UnidentifiedImageError: raise ValueError(f"Cannot identify image: {file_path}")
            except Exception as img_err: raise ValueError(f"Error opening image: {img_err}")
        else: raise ValueError(f"Unsupported type: {file_extension}.")

        if page_image_pil is None: raise ValueError("Failed to load/convert doc to image.")
        final_output_image = page_image_pil.copy()

        # --- 2b. Document AI OCR ---
        if not mime_type: raise ValueError("MIME type determination failed.")
        docai_document = process_document_google_docai(GCP_PROJECT_ID, GCP_LOCATION, GCP_PROCESSOR_ID, original_doc_path_for_docai, mime_type)
        if not docai_document: raise ConnectionError("DocAI processing failed.")

        # --- 2c. Extract OCR Words and Absolute Boxes ---
        words, unnormalized_boxes_4pt = extract_ocr_data_from_docai(docai_document)
        if not words or not unnormalized_boxes_4pt: raise ValueError("No text/boxes extracted by OCR.")

        # --- 2d. Normalize Boxes & Prepare Model Input ---
        pil_page_width, pil_page_height = page_image_pil.size
        if pil_page_width <= 0 or pil_page_height <= 0: raise ValueError("Invalid image dimensions.")
        normalized_boxes_1000 = [normalize_box(box, pil_page_width, pil_page_height) for box in unnormalized_boxes_4pt]

        logging.info(f"Encoding {len(words)} words for the model...")
        encoding = processor(images=page_image_pil, text=question_str, text_pair=words, boxes=normalized_boxes_1000, max_length=MAX_SEQ_LENGTH, padding="max_length", truncation="longest_first", return_tensors="pt")
        logging.info(f"Input shape: {encoding['input_ids'].shape}")

        # --- 2e. Model Inference ---
        logging.info("Running model inference...")
        start_inference_time = time.time()
        inputs = {k: v.to(DEVICE) for k, v in encoding.items()}
        with torch.no_grad(): outputs = model(**inputs)
        predictions_ids = outputs.logits.argmax(-1)
        end_inference_time = time.time(); logging.info(f"✅ Inference finished in {end_inference_time - start_inference_time:.2f} sec.")

        # --- 2f. Extract Answer & Draw Box ---
        logging.info("Extracting answer details...")
        final_answer_text, final_answer_box_pixels = extract_answer_details(predictions_ids, encoding, words, unnormalized_boxes_4pt)
        logging.info("Drawing bounding box...")
        final_output_image = draw_box_on_image(final_output_image, final_answer_box_pixels, color="red", width=3)
        if final_output_image is None: final_output_image = page_image_pil.copy() # Fallback

    # --- Error Handling ---
    except ImportError as e: final_answer_text, final_output_image = f"❌ Error: Library missing. {e}.", page_image_pil.copy() if page_image_pil else create_placeholder_image(message=f"Import Error:\n{e}")
    except FileNotFoundError as e: final_answer_text, final_output_image = f"❌ Error: Input file not found.", create_placeholder_image(message=f"File Not Found:\n{Path(doc_file_path).name}")
    except UnidentifiedImageError as e: final_answer_text, final_output_image = f"❌ Error: Cannot identify image.", create_placeholder_image(message=f"Bad Image Format:\n{Path(doc_file_path).name}")
    except ValueError as e: final_answer_text, final_output_image = f"❌ Error: {e}", final_output_image if final_output_image else page_image_pil.copy() if page_image_pil else create_placeholder_image(message=f"Processing Error:\n{e}")
    except ConnectionError as e: final_answer_text, final_output_image = f"❌ Error: DocAI connection failed.", final_output_image if final_output_image else page_image_pil.copy() if page_image_pil else create_placeholder_image(message=f"DocAI Error:\n{e}")
    except Exception as e: final_answer_text, final_output_image = f"❌ Unexpected error: {e}.", final_output_image if final_output_image else page_image_pil.copy() if page_image_pil else create_placeholder_image(message=f"Unexpected Error:\n{e}")
    finally: logging.error(f"{final_answer_text}", exc_info=True) if "❌ Error" in final_answer_text else None # Log errors

    # --- Timing and Return ---
    end_overall_time = time.time(); processing_time = end_overall_time - start_overall_time
    logging.info(f"--- Request Finished ({processing_time:.2f} sec) ---"); logging.info(f"Answer: '{final_answer_text[:100]}...'")
    if final_output_image is None: final_output_image = create_placeholder_image(message=final_answer_text) # Ensure image return
    return final_answer_text, final_output_image


# --- Build Gradio Interface using Blocks ---

# Define content for the informational tabs
about_text = """
## 🚀 Project: EdTech Smart Document Assistant (MVP)

**Team:** [** NIGHT SHADOW **]

**Goal:** To create a Minimum Viable Product demonstrating how AI can help students and educators quickly find specific information within educational documents.

**Problem:** Navigating long textbooks, research papers, or lecture notes to find answers to specific questions can be time-consuming and inefficient for both learning and teaching preparation.

**Solution:** This tool leverages a state-of-the-art Document Question Answering model (LayoutLMv3) combined with powerful OCR (Google Document AI) to understand the layout and text of a document. Users can upload a document and ask a natural language question. The AI then pinpoints the answer within the text and visually highlights its location on the document image.

**Impact:** This enhances study efficiency, aids research, and assists educators in quickly extracting key information.
"""

how_it_works_text = """
## 🛠️ How It Works: Technology & Workflow

This application combines several powerful technologies:

1.  **Document Input:** Accepts various formats (PDF, PNG, JPG, etc.). PDFs are converted to images (first page only) using `pdf2image`.
2.  **OCR & Layout Analysis:** The uploaded document (or original PDF) is sent to **Google Cloud Document AI** (specifically, a Form Parser or similar OCR processor). Document AI extracts text (`words`) and their precise locations (`bounding boxes`) on the page.
3.  **LayoutLMv3 Model:**
    *   We use a `microsoft/layoutlmv3-base` model, **fine-tuned** on a document question-answering dataset in this case DOCVQA. This fine-tuning teaches the model to identify specific answer spans within a document context based on a question.
    *   The **Processor** prepares the inputs: the question, the extracted words, their normalized bounding boxes (scaled to 0-1000), and the document image itself. LayoutLMv3 uses multimodal embeddings, considering text, layout, and visual features.
4.  **Inference:** The prepared inputs are fed to the fine-tuned LayoutLMv3 model running on the available device (CPU, CUDA GPU, or MPS). The model outputs predictions for each token, indicating whether it's part of the answer (`B-ANSWER`, `I-ANSWER`) or not (`O`).
5.  **Answer Extraction:** We identify the tokens predicted as the answer, map them back to the original words provided by OCR, and reconstruct the answer text. The bounding boxes of these answer words are combined to create a final bounding box encompassing the full answer span.
6.  **Visualization:** The final bounding box (in pixel coordinates) is drawn onto a copy of the input image using the Pillow library.
7.  **Interface:** **Gradio** provides the interactive web user interface, allowing easy file uploads, question input, and display of the text answer and highlighted image.

**Code:** [**https://github.com/saimmalik577/hk_gdg.git**]
"""

# --- Define Example(s) ---
# Assumes app.py runs from GDG_HACKATHON root
example_list = [
    ["assets/sample_doc.jpg", "What is the equity owner name?"]
    # Add more examples relevant to EdTech if available:
    # ["example_docs/science_paper.png", "What method was used for analysis?"]
]
# Validate example file paths
valid_examples = []
for ex in example_list:
    example_file_path = Path(ex[0])
    if example_file_path.is_file():
        valid_examples.append(ex)
    else:
        logging.warning(f"Example file not found, skipping: {example_file_path}")
        print(f"Warning: Example file not found, skipping: {example_file_path}")


# --- Create Gradio Blocks Interface ---
if MODEL_LOADED:
    with gr.Blocks(theme=gr.themes.Soft(), title="EdTech Smart Document Assistant") as demo:
        gr.Markdown("# 🧠 EdTech Smart Document Assistant (MVP)")

        with gr.Tabs():
            # --- Tab 1: Main Application ---
            with gr.TabItem("Assistant"):
                gr.Markdown("**Upload a document, ask a question, and get the answer highlighted!**")
                with gr.Row():
                    with gr.Column(scale=1):
                        doc_input = gr.File(label="Upload Document (PDF/Image)", type="filepath")
                        question_input = gr.Textbox(label="Question", placeholder="e.g., What is the main conclusion?")
                        submit_btn = gr.Button("Find Answer", variant="primary")
                        gr.Examples(
                            examples=valid_examples,
                            inputs=[doc_input, question_input],
                            label="Examples (Click to run)"
                        )
                    with gr.Column(scale=2):
                        answer_output = gr.Textbox(label="Answer", interactive=False)
                        image_output = gr.Image(label="Document with Answer Highlighted", type="pil", interactive=False)

                # Wire the button click to the prediction function
                submit_btn.click(
                    fn=predict_answer,
                    inputs=[doc_input, question_input],
                    outputs=[answer_output, image_output]
                )

            # --- Tab 2: About This Project ---
            with gr.TabItem("About"):
                gr.Markdown(about_text)

            # --- Tab 3: How It Works ---
            with gr.TabItem("How It Works"):
                gr.Markdown(how_it_works_text)

else:
    # Fallback interface if model loading failed
    with gr.Blocks(theme=gr.themes.Soft(), title="Error - App Startup Failed") as demo:
         gr.Markdown("# ❌ Application Startup Failed")
         gr.Markdown("The LayoutLMv3 model or its processor could not be loaded. Please check the console logs for error messages regarding model paths, dependencies, or memory issues.")
         gr.Textbox(label="Status", value="Model loading failed. Application cannot run. See console logs.", interactive=False)
         gr.Image(label="Status Image", value=create_placeholder_image(message="Model Loading Failed")) # Show placeholder

# --- Launch the App ---
if __name__ == "__main__":
    logging.info("Starting Gradio App for EdTech MVP...")
    # Launch the Gradio app
    # Set server_name="0.0.0.0" to allow access from network
    demo.launch(server_name="0.0.0.0", server_port=7860)
    # demo.launch() # Basic local launch