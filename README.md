# EdTech Smart Document Assistant (MVP) - GDG Hackathon Submission

**Team:** [EdTech Smart Document Assistant (MVP)]

## 🚀 Project Goal

To create a Minimum Viable Product demonstrating how AI (LayoutLMv3 fine-tuned for Document Question Answering) can help students and educators quickly find and locate specific information within educational documents.

## ✨ Features

*   Accepts PDF, PNG, JPG, and other image formats (PDFs processed first page only).
*   Uses Google Cloud Document AI for robust OCR and layout detection.
*   Employs a fine-tuned LayoutLMv3 model to understand the question and document content.
*   Identifies the answer text within the document.
*   Highlights the location of the answer with a bounding box on the document image.
*   Provides an interactive web interface using Gradio.

## 🛠️ Technology Stack

*   **Model:** Fine-tuned `microsoft/layoutlmv3-base`
*   **OCR:** Google Cloud Document AI API
*   **Frameworks:** PyTorch, Hugging Face Transformers
*   **UI:** Gradio
*   **PDF Handling:** `pdf2image` (requires `poppler`)
*   **Core Language:** Python 3

## 📋 Setup Instructions

1.  **Clone Repository:**
    ```bash
    git clone <https://github.com/saimmalik577/hk_gdg.git>
    cd gdg_hackathon # hk_gdg
    ```

2.  **Create Virtual Environment & Activate (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # .\venv\Scripts\activate  # Windows Powershell
    # venv\Scripts\activate.bat # Windows CMD
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **GCP Credentials (Required):**
    *   You need a Google Cloud Platform project with the **Document AI API enabled**.
    *   Create or use an existing Service Account and download its JSON key file.
    *   Create a folder named `.keys` in the project root directory.
    *   Place the downloaded service account key file inside this folder and ensure its name is exactly `edtech-vllm-app-b6ae35847a58.json` (or update the `CREDENTIALS_FILE_PATH` variable in `app.py`).
    *   *Note: The `.keys` directory is included in `.gitignore` and will not be committed.*

5.  **Model Checkpoint (Required):**
    *   The fine-tuned LayoutLMv3 model checkpoint is essential for the application's functionality but is too large for Git.
    *   Download the model checkpoint archive (`checkpoint-1500.zip`) from:
        **[ <<< PASTE YOUR PUBLIC DOWNLOAD LINK FOR THE MODEL CHECKPOINT ZIP >>> ]**
    *   Unzip the downloaded file. This should result in a folder named `checkpoint-1500`.
    *   Create the necessary directory structure if it doesn't exist: `mkdir -p models/layoutlmv3_finetuned_smartdoc_qa/`
    *   Move the unzipped `checkpoint-1500` folder into this path. The final structure **must** be: `<project_root>/models/layoutlmv3_finetuned_smartdoc_qa/checkpoint-1500/`

6.  **Dataset (Used for Fine-tuning):**
    *   **(Option 1: If you used a standard public dataset like SQuAD, DocVQA, etc.)**
        This model was fine-tuned using the [**Specify Dataset Name and Version, e.g., DocVQA v1.0 training set**]. No separate data download is required for running the app if you only need to reference the dataset source.
    *   **(Option 2: If you used a custom or modified dataset)**
        The specific dataset used for fine-tuning is available for review.
        *   Download the dataset archive (`[your_dataset_name].zip`) from:
            **[ <<< PASTE YOUR PUBLIC DOWNLOAD LINK FOR THE DATASET ZIP >>> ]**
        *   Unzip the downloaded file.
        *   Place the contents in the appropriate location within the `data/` directory (e.g., `data/processed/`, `data/custom/` - specify the exact path where the unzipped files should go based on your training scripts, if relevant). *Example: Place the JSON files in `data/raw/spdocvqa_qas/`.*

7.  **PDF Prerequisite (Optional but Recommended):**
    *   To process PDF files, the `poppler` utility library must be installed on your system.
    *   **macOS:** `brew install poppler`
    *   **Debian/Ubuntu:** `sudo apt-get update && sudo apt-get install -y poppler-utils`
    *   **Windows:** Download from [official Poppler builds](https://github.com/oschwartz10612/poppler-windows/releases/) or use a package manager like `conda install -c conda-forge poppler`. Ensure the `poppler/bin` directory is added to your system's PATH.

## ▶️ How to Run

1.  Ensure you have completed all steps in the **Setup Instructions**.
2.  Make sure your virtual environment is activated.
3.  Run the Gradio application from the project root directory:
    ```bash
    python app.py
    ```
4.  Open your web browser and navigate to the URL provided in the console (usually `http://127.0.0.1:7860` or `http://0.0.0.0:7860`).

5.  Use the interface:
    *   Go to the "Assistant" tab.
    *   Upload a document (PDF or Image).
    *   Type a question related to the document content.
    *   Click "Find Answer".
    *   View the extracted text answer and the highlighted bounding box on the image.
    *   You can also try the provided example by clicking on it.

## ℹ️ Further Information

*   Explore the "About" tab for the project's motivation and goals.
*   Explore the "How It Works" tab for details on the technical workflow.

## 👥 Team Members

*   [Saim AD Malik]
*   [Basma Salim]
*   ...