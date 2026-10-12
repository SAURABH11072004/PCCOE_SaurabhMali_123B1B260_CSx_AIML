# Installation and Execution Guide — AUTOSAR HLD Document Analysis Assistant

## 1. System Requirements
- **Operating System**: Windows 10/11 (x64)
- **Python Version**: Python 3.10 to 3.13 (Python 3.13 verified)
- **RAM**: Minimum 8 GB (16 GB recommended for local embedding caching)
- **Disk Space**: ~2 GB free disk space for dependencies and embedding weights

---

## 2. Environment Setup

### 2.1 Clone / Extract Project
Open PowerShell and navigate to the project directory:
```powershell
cd "PCCOE_StudentName_PRN_CSx_AIML\Code"
```

### 2.2 Create and Activate Virtual Environment
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2.3 Upgrade pip and Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Environment & Secrets Configuration

### 3.1 Copy Environment Template
```powershell
cp .env.example .env
```

### 3.2 Configure `.env`
Open `.env` in a text editor:
```dotenv
GOOGLE_API_KEY=AIzaSyYourGeminiAPIKeyHere
GEMINI_MODEL=gemini-1.5-flash
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHROMA_PERSIST_DIR=./chroma_db
CHUNK_SIZE=500
CHUNK_OVERLAP=100
RETRIEVAL_TOP_K=4
```

> [!IMPORTANT]
> - Never commit your `.env` file to Git.
> - If `GOOGLE_API_KEY` is omitted, the application operates in **Retrieval-Only Mode**, preserving full search and citation capabilities without making external API calls.

### 3.3 Test Gemini API Connectivity
Verify your configuration using the safe connectivity script:
```powershell
python scripts/check_gemini_connection.py
```

---

## 4. Running the Application

Launch the Streamlit user interface:
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 5. Running Automated Tests and Evaluation

### 5.1 Run Automated pytest Suite
```powershell
pytest tests/ -v
```

### 5.2 Run Evaluation Benchmark
```powershell
python -m evaluation.run_evaluation
```
Benchmark outputs and markdown summaries will be written to `Evaluation_Results/`.

---

## 6. Optional: Installing Tesseract OCR (Windows)

Tesseract OCR enables text extraction from scanned or image-only PDF pages. Text-based PDFs work without Tesseract.

### 6.1 Download and Install
1. Download the Windows installer from UB-Mannheim:
   **<https://github.com/UB-Mannheim/tesseract/wiki>**
2. Run the installer. The default installation path is:
   `C:\Program Files\Tesseract-OCR\`
3. During installation, select **Additional language data** if your documents contain non-English text.

### 6.2 Verify Installation
Open a new PowerShell terminal and run:
```powershell
tesseract --version
```
Expected output shows the Tesseract version (e.g., `tesseract 5.x.x`).

### 6.3 Configure Custom Path (Optional)
If Tesseract was installed to a non-default location, add this line to your `.env` file:
```dotenv
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

The application checks for Tesseract in this order:
1. `TESSERACT_CMD` environment variable
2. System `PATH`
3. Common Windows installation directories (`Program Files`, `Program Files (x86)`, `LOCALAPPDATA`)

### 6.4 OCR Behaviour Without Tesseract
- Text-based PDFs continue working normally — no functionality is lost.
- If a scanned PDF is uploaded and Tesseract is unavailable, the application will:
  - Index any digital text pages normally.
  - Display a contextual warning in the Documents section identifying the affected pages.
  - Never crash or produce corrupted results.

> [!NOTE]
> OCR extracts machine-readable text from scanned images. It does not reconstruct architectural diagram semantics, signal-flow connections, or graphical relationships. See `Documentation/Known_Limitations.md` for full details.
