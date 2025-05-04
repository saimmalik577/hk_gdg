# EdTech Smart Document Assistant (MVP) - GDG Hackathon Submission

**Team:** EdTech Smart Document Assistant (MVP)  *(You can choose a more creative team name if you like!)*

## 🚀 Project Goal

To create a Minimum Viable Product demonstrating how AI (LayoutLMv3 fine-tuned for Document Question Answering) can help students and educators quickly find and locate specific information within educational documents, reducing study time and improving research efficiency.

## ✨ Features

*   **Versatile Input:** Accepts PDF (first page only), PNG, JPG, TIFF, BMP, GIF formats.
*   **Accurate OCR:** Leverages Google Cloud Document AI for high-fidelity text extraction and layout parsing.
*   **Intelligent QA:** Employs a fine-tuned LayoutLMv3 model (multimodal: text, layout, image) to understand questions and document context.
*   **Precise Extraction:** Identifies the specific answer text within the document.
*   **Visual Localization:** Highlights the exact location of the answer with a bounding box on the document image for easy reference.
*   **Interactive UI:** Provides a user-friendly web interface built with Gradio, including informational tabs and examples.

## 🛠️ Technology Stack

*   **Core Model:** Fine-tuned `microsoft/layoutlmv3-base`
*   **OCR Service:** Google Cloud Document AI API (Form Parser Processor recommended, but OCR Processor works)
*   **ML Frameworks:** PyTorch, Hugging Face Transformers
*   **Web UI:** Gradio
*   **PDF Handling:** `pdf2image` (requires `poppler` system dependency)
*   **Language:** Python 3

## 📋 Setup Instructions

Follow these steps carefully to set up and run the project locally:

1.  **Clone Repository:**
    ```bash
    git clone https://github.com/saimmalik577/hk_gdg.git
    cd hk_gdg # Navigate into the cloned repository directory
    ```

2.  **Create & Activate Virtual Environment (Recommended):**
    ```bash
    python3 -m venv venv # Use python3 explicitly if needed
    source venv/bin/activate  # Linux/macOS
    # .\venv\Scripts\activate  # Windows Powershell
    # venv\Scripts\activate.bat # Windows CMD
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Google Cloud Platform & Document AI Setup (Required):**
    *   **Create/Select GCP Project:** You need a Google Cloud Platform project. [Create one here](https://console.cloud.google.com/projectcreate) if needed.
    *   **Enable Document AI API:** Navigate to the "APIs & Services" > "Library" section in your GCP Console. Search for "Document AI API" and click **Enable**. You might also need to enable Billing for your project if you haven't already.
    *   **Create a Document AI Processor:** Go to the Document AI section in the GCP Console. Click "Create Processor". Choose a suitable processor type (e.g., "Form Parser" or "Document OCR") and select the region (`eu` used in this code). Note the **Processor ID** generated after creation. *(While the app uses the ID `67bd191d8754344a`, judges should ideally create their own processor in their project).*
    *   **Create Service Account Key:** Go to "IAM & Admin" > "Service Accounts". Click "Create Service Account", give it a name (e.g., `doc-ai-runner`), and grant it the **"Document AI API User"** role (or a broader role like Editor if necessary). After creating the service account, go to its "Keys" tab, click "Add Key" > "Create new key", choose **JSON**, and click "Create". A JSON key file will be downloaded.
    *   **Place Credentials File:** Create a folder named `.keys` in the project root directory (`hk_gdg/.keys/`). Move the downloaded JSON key file into this folder and rename it to **`edtech-vllm-app-b6ae35847a58.json`**. (Alternatively, update the `CREDENTIALS_FILE_PATH` variable in `app.py` to match your key file's name and location).
    *   *Note: The `.keys` directory is correctly ignored by `.gitignore`.*

5.  **Model Checkpoint (Required):**
    *   The fine-tuned LayoutLMv3 model (`checkpoint-1500`) is crucial but too large for Git.
    *   Download the required model checkpoint archive (`checkpoint-1500.zip`) from:
        **[ <<< PASTE YOUR PUBLIC DOWNLOAD LINK FOR checkpoint-1500.zip HERE >>> ]**
    *   Create the directory structure: `mkdir -p models/layoutlmv3_finetuned_smartdoc_qa/`
    *   Unzip the downloaded `checkpoint-1500.zip` **directly inside** the `models/layoutlmv3_finetuned_smartdoc_qa/` directory.
    *   Verify the final path: `<project_root>/models/layoutlmv3_finetuned_smartdoc_qa/checkpoint-1500/` (this folder should contain `pytorch_model.bin`, `config.json`, etc.).

6.  **Dataset & Preprocessed Data (For Reference):**
    *   **Training Dataset:** This model was fine-tuned using the publicly available **DocVQA v1.0 dataset**. Its focus on VQA over document images with complex layouts made it ideal for our EdTech application.
    *   **Data Preprocessing Pipeline:** The scripts in the `src/` directory detail our process. Key steps included:
        *   Aligning dataset annotations (questions, contexts, answers) with model input requirements.
        *   Tokenizing text and calculating layout bounding boxes compatible with LayoutLMv3.
        *   Converting character-level answer spans into token-level classification labels (B-ANSWER, I-ANSWER, O).
    *   **Subset Creation:** During development, scripts like `src/create_test_subset.py` were used to generate smaller data subsets. This enabled faster fine-tuning iterations, debugging, and hyperparameter testing.
    *   **Preprocessed/Tokenized Data (Optional Download):** For judges interested in examining the exact format of the data fed into the model *after* preprocessing and tokenization, we have made the contents of the `data/prepared/` directory available. These are typically PyTorch tensor files (`.pt`) containing token IDs, attention masks, bounding boxes, labels, etc., ready for model training/evaluation.
        *   Download the preprocessed data archive (`prepared_data.zip` or similar name) from:
            **[ <<< PASTE YOUR PUBLIC DOWNLOAD LINK FOR THE ZIPPED 'prepared' FOLDER HERE >>> ]**
        *   To inspect, unzip the archive. You can place the contents into the `<project_root>/data/prepared/` directory (this path is ignored by Git). Loading these `.pt` files typically requires PyTorch (`torch.load(...)`).
    *   *Note: Neither the raw DocVQA dataset nor the preprocessed data linked above are required to run the application demo using the provided fine-tuned model checkpoint.*
        
7.  **PDF Prerequisite (Optional but Recommended):**
    *   For the app to process **PDF files**, the `poppler` utility library must be installed system-wide.
    *   **macOS:** `brew install poppler`
    *   **Debian/Ubuntu:** `sudo apt-get update && sudo apt-get install -y poppler-utils`
    *   **Windows:** Download builds from [Poppler for Windows Releases](https://github.com/oschwartz10612/poppler-windows/releases/), unzip, and add the `bin` directory within the unzipped folder to your system's **PATH** environment variable. Alternatively, use `conda install -c conda-forge poppler`.

## ▶️ How to Run the Application

1.  Ensure all **Setup Instructions** are completed (dependencies, credentials, model).
2.  Activate your virtual environment (`source venv/bin/activate` or similar).
3.  Navigate to the project root directory (`hk_gdg`) in your terminal.
4.  Launch the Gradio web server:
    ```bash
    python app.py
    ```
5.  Open your web browser to the local URL provided (usually `http://127.0.0.1:7860` or `http://0.0.0.0:7860`).
6.  **Interact with the Web UI:**
    *   The **"Assistant"** tab is the main application interface.
    *   Use the "Upload Document" button or drag-and-drop a file (PDF/Image).
    *   Type your question about the document into the "Question" box.
    *   Click the "Find Answer" button.
    *   Results (text answer and highlighted image) will appear on the right.
    *   Try the pre-loaded example by clicking on it below the inputs.
    *   Explore the "About" and "How It Works" tabs for more project context.

## ℹ️ Further Information & Potential Improvements

*   **Challenges:** Key challenges included meticulous data preprocessing to align OCR results with dataset annotations and managing compute resources for fine-tuning large models. We wanted to train the model on hand written annotation too, to improve the accuracy but unfortunately fell short of time.
*   **Future Work:** Enhancements could include multi-page PDF support, handling unanswerable questions, integrating chat history/follow-up questions, and exploring different Document AI processor types for specialized documents.

## 👥 Team Members

*   Saim AD Malik
*   Basma Salim
*   Christian