# AUTOSAR HLD Document Analysis Assistant

## Live Demo

Try the deployed application: [AUTOSAR HLD Assistant](https://hld-assistant-saurabhmali.streamlit.app/)

The live app requires a valid Gemini API key for AI-generated answers; retrieval features may still be available if Gemini authentication is not configured correctly.


An AI-powered document analysis assistant designed to extract, organize, retrieve, and validate architectural knowledge from AUTOSAR High-Level Design (HLD) documents using Retrieval-Augmented Generation (RAG).

## Overview

AUTOSAR High-Level Design documents contain information about software components, interfaces, ports, signals, dependencies, and functional flows. These documents can be lengthy and difficult to analyze manually.

The **AUTOSAR HLD Document Analysis Assistant** simplifies this process by combining document ingestion, text chunking, semantic retrieval, and AI-assisted question answering. It helps users locate relevant technical information and review architectural details with supporting document citations.

## Key Features

- **PDF Document Ingestion:** Extracts text from AUTOSAR HLD PDF documents.
- **Text Chunking:** Divides document content into manageable chunks for retrieval.
- **Semantic Search:** Retrieves relevant content using sentence embeddings and a vector database.
- **AI-Powered Question Answering:** Uses a RAG pipeline to generate context-grounded responses.
- **Source Citations:** Provides document references to help users verify generated answers.
- **Document Comparison:** Supports comparison of technical documents.
- **Entity Extraction:** Identifies architectural entities and relevant technical information.
- **Issue Detection:** Helps identify potential inconsistencies or issues in document content.
- **Evaluation Framework:** Measures retrieval performance, citation quality, factual matching, and abstention behavior.
- **Optional OCR:** Supports text extraction from scanned PDFs when Tesseract OCR is configured.

## System Architecture

The application follows a Retrieval-Augmented Generation workflow.

```text
AUTOSAR HLD PDF
       |
       v
Document Ingestion
       |
       v
Text Extraction and Chunking
       |
       v
Embedding Generation
       |
       v
Chroma Vector Database
       |
       v
User Query
       |
       v
Semantic Retrieval
       |
       v
Retrieved Context + AI Model
       |
       v
Grounded Answer with Citations
```

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| User Interface | Streamlit |
| Language Model | Google Gemini API |
| Embedding Model | all-MiniLM-L6-v2 |
| Vector Database | ChromaDB |
| Document Processing | PDF text extraction utilities |
| OCR | Tesseract OCR (optional) |
| Testing | pytest |
| Evaluation | Custom retrieval and citation evaluation framework |

## Repository Structure

```text
PCCOE_SaurabhMali_123B1B260_CSx_AIML/
│
├── Code/
│   ├── app.py
│   ├── requirements.txt
│   ├── .env.example
│   ├── .gitignore
│   ├── src/
│   ├── evaluation/
│   ├── tests/
│   ├── scripts/
│   └── Evaluation_Results/
│
├── Declarations/
│   ├── Faculty_Approval_Checklist.md
│   └── Student_Declaration_Template.md
│
├── Documentation/
│   ├── Installation_Guide.md
│   ├── Known_Limitations.md
│   ├── System_Architecture.md
│   ├── Technical_Report.md
│   ├── Technical_Report.pdf
│   ├── Testing_Report.md
│   ├── User_Guide.md
│   └── Video_Demonstration_Script.md
│
├── Evaluation_Results/
│   ├── Evaluation_Report.md
│   ├── metrics.json
│   ├── answer_results.json
│   ├── citation_validation.json
│   ├── evaluation_questions.json
│   └── retrieval_results.json
│
├── Input_Data/
│   ├── README.md
│   ├── evaluation_dataset.json
│   ├── synthetic_autosar_hld.pdf
│   └── synthetic_autosar_hld_v2.pdf
│
├── Model_Prompts_Config/
│   ├── .env.example
│   ├── evaluation_methodology.md
│   ├── extraction_prompt.md
│   ├── model_configuration.md
│   └── rag_system_prompt.md
│
├── Synopsis/
│   └── synopsis.pdf
│
├── Video/
│   └── README.md
│
└── PCCOE_SaurabhMali_123B1B260_CSx_AIML.zip
```

## Prerequisites

Before running the application, install:

- Python 3.10 or a compatible version supported by the project's dependencies.
- pip package manager.
- Git.
- A Google Gemini API key for AI-powered response generation.
- Tesseract OCR only if scanned-document OCR is required.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/SAURABH11072004/PCCOE_SaurabhMali_123B1B260_CSx_AIML.git
cd PCCOE_SaurabhMali_123B1B260_CSx_AIML
```

### 2. Navigate to the application directory

```bash
cd Code
```

### 3. Create a virtual environment

**Windows:**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux/macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Configure environment variables

Create a local `.env` file based on `.env.example`, then configure the API key using the variable name expected by `src/config.py` and the Gemini client.

Example configuration:

```env
GEMINI_API_KEY=YOUR_GEMINI_API_KEY_HERE
GEMINI_MODEL=gemini-3.5-flash-lite

EMBEDDING_MODEL=all-MiniLM-L6-v2
CHROMA_PERSIST_DIR=./chroma_db
CHUNK_SIZE=500
CHUNK_OVERLAP=100
RETRIEVAL_TOP_K=4
```

Obtain your API key from [Google AI Studio](https://aistudio.google.com/).

**Security:** Never commit your actual API key or `.env` file to GitHub. Keep credentials in environment variables or your deployment platform's secrets manager.

> Note: The committed example file currently uses `GOOGLE_API_KEY=GEMINI_API_KEY`. Confirm the exact variable name expected by your application before configuring the environment; the placeholder above may need to match the code.

### 6. Run the application

From the `Code/` directory:

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

```text
http://localhost:8501
```

## How to Use

1. Launch the Streamlit application.
2. Upload or select an AUTOSAR HLD PDF supported by the application.
3. Allow the document ingestion and indexing process to complete.
4. Enter a question about components, interfaces, ports, signals, dependencies, or functional flows.
5. Review the generated answer and its source citations.
6. Verify important technical statements against the cited document pages.
7. Use document comparison and issue-analysis features when available in the application.
8. Consult the evaluation reports to understand the system's measured performance.

## Evaluation

The project includes a reproducible evaluation framework for assessing retrieval and answer-grounding behavior.

The latest recorded benchmark results are:

| Evaluation Metric | Result |
|---|---:|
| Evaluation Questions | 20 |
| Retrieval Recall@4 | 100% |
| Citation Validity | 100% |
| Citation Coverage | 100% |
| Abstention Accuracy | 100% |
| Factual Match Rate | 82.8% |
| Average Query Latency | 0.051 seconds |
| Document Ingestion Time | 34.02 seconds |

These figures are results from the recorded synthetic evaluation dataset and configuration. They should not be interpreted as guaranteed performance on real-world AUTOSAR documents or as evidence of production readiness.

See `Evaluation_Results/Evaluation_Report.md` and the associated JSON files for evaluation details.

## Testing

The repository contains pytest-based tests for modules such as chunking, ingestion, vector storage, document comparison, retrieval, export utilities, and issue detection.

Run the test suite from the `Code/` directory:

```bash
pytest
```

Review test output to determine whether any tests require API access, additional configuration, or optional system dependencies.

## Configuration

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | API credential, if this is the variable expected by the application |
| `GEMINI_MODEL` | Configures the Gemini model |
| `EMBEDDING_MODEL` | Selects the embedding model |
| `CHROMA_PERSIST_DIR` | Sets the vector database persistence directory |
| `CHUNK_SIZE` | Controls text chunk size |
| `CHUNK_OVERLAP` | Controls overlap between adjacent chunks |
| `RETRIEVAL_TOP_K` | Sets the number of retrieved chunks |
| `TESSERACT_CMD` | Optional path to the Tesseract executable |

The actual defaults and supported variable names are determined by the application configuration in `Code/src/config.py`.

## Known Limitations

- Generated answers may be incomplete or incorrect and must be checked against the source document.
- Retrieval performance depends on document quality, chunking, embeddings, and the selected retrieval settings.
- OCR requires the external Tesseract executable to be installed and configured.
- Text-based retrieval does not necessarily capture the complete semantics of complex architectural diagrams, connectors, or visual relationships.
- Results on synthetic evaluation data may not generalize to all real-world AUTOSAR HLD documents.
- API availability, rate limits, network connectivity, and model changes may affect response generation.

For additional details, see `Documentation/Known_Limitations.md`.

## Future Enhancements

Potential improvements include:

- More robust extraction of architectural entities and relationships.
- Improved handling of tables, scanned PDFs, and diagrams.
- Expanded evaluation on diverse, real-world HLD documents.
- Stronger validation of generated answers and citations.
- More detailed traceability between components, interfaces, signals, and dependencies.
- Deployment with managed secrets, monitoring, and appropriate access controls.

These are potential enhancements, not claims that they have already been implemented.

## Responsible Use

This tool is intended to assist with document analysis and engineering knowledge retrieval. It should not replace expert architectural review, formal AUTOSAR conformance checks, safety analysis, or applicable certification processes.

Always validate important technical conclusions against authoritative project documentation.

## Documentation

Additional project material is available in:

- `Synopsis/` — Project synopsis.
- `Documentation/` — Installation, user, architecture, testing, and technical reports.
- `Input_Data/` — Sample documents and evaluation dataset.
- `Model_Prompts_Config/` — Prompts and model configuration documentation.
- `Evaluation_Results/` — Recorded evaluation results.
- `Video/` — Video demonstration information.
- `Declarations/` — Declaration and approval templates.

## License

No license is specified in this repository. Contact the project owner before reusing or redistributing the code or project materials.

## Author

**Saurabh Digambar Mali**  
B.Tech Computer Engineering  
Pimpri Chinchwad College of Engineering (PCCOE), Pune

GitHub: [SAURABH11072004](https://github.com/SAURABH11072004)

---

*Developed as an applied AI/ML capstone project focused on improving access to architectural knowledge in AUTOSAR High-Level Design documents.*
