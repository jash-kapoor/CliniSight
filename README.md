# CliniSight: Medical Laboratory Report Explainer

CliniSight is an educational laboratory report explanation system that ingests photographed or scanned diagnostic blood test reports, performs dual-pass vision and OCR extraction, recomputes clinical reference flags deterministically in Python, provides a human-in-the-loop verification interface, and generates grounded clinical insights and conversational explanations with strict medical safety guardrails.

---

## Architecture and Pipeline

```
Patient Report Upload (PDF / JPG / PNG / HEIC)
   |
   v
[Stage A: Ingestion & OpenCV Preprocessing]
   * Aspect-ratio scaling, blur scoring, deskew, and contrast enhancement.
   |
   +---> [Stage B: Multimodal Vision Extraction & RapidOCR Cross-Check]
   |        * Schema-constrained extraction with Gemini Vision API.
   |        * Independent RapidOCR line and token extraction.
   |
   +---> [Stage C: Completeness & Token Audit]
   |        * Second-pass inventory comparison and OCR token coverage calculation.
   |
   +---> [Stage D: Deterministic Normalization & Flagging]
   |        * Regex interval parsing, Lakh scale alignment, explicit critical limits.
   |
   v
[Stage E: Human-In-The-Loop Verification Screen]
   * Two-column UI displaying page image with row highlighting and editable values.
   * User verifies or amends parameters before indexing.
   |
   v
[Storage, Indexing & Insight Engine]
   * Isolated per-report namespace indexing in Chroma / FAISS / Pinecone.
   * Rule-based physiological clustering (Lipid, Glycemic, Renal, Anemia patterns).
   * Priority stratification (High, Medium, Low Concern).
   * Contextual diet recommendations and doctor visit question generation.
   |
   v
[Grounded Explanatory Chat: Dr. Mira]
   * Sliding-window multi-turn memory buffer with session isolation.
   * Deterministic acute symptom detection and emergency escalation protocol.
   * Prompt-constrained generation with page-level biological evidence citations.
```

---

## Key Features

- **Universal Document Ingestion:** Processes photographed smartphone images, scanned documents, and digital PDFs.
- **Dual-Pass Extraction & OCR Cross-Check:** Audits vision model outputs against an independent local RapidOCR baseline to detect missed table rows or dropped tokens.
- **Deterministic Python Flagging:** Eliminates LLM arithmetic hallucination by calculating low, normal, high, and critical reference flags mathematically in Python.
- **Human Verification Touchpoint:** Empowers the patient to review and verify every extracted parameter before clinical synthesis.
- **Isolated Per-Report Vector Spaces:** Enforces strict multi-tenant namespace partitioning to prevent cross-report data contamination.
- **Clinical Insight Synthesis:** Formulates personalized overviews, physiological pattern clusters, dietary guidance, and physician consultation questions.
- **Multi-Turn Grounded Chat:** Interactive Q&A grounded exclusively in verified laboratory findings, with acute emergency symptom escalation.
- **Complete Privacy Purge:** Provides one-click cryptographic deletion of all stored vector namespaces, parsed JSON records, and rendered page images.

---

## Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Backend Framework** | Python 3.11+, FastAPI, Pydantic v2, Uvicorn |
| **Computer Vision & OCR** | OpenCV (cv2), RapidOCR (ONNX runtime), pypdfium2, pillow-heif |
| **AI Models & Embeddings** | Google Gemini Multimodal Vision, Gemini Chat, Gemini Text Embeddings (`gemini-embedding-001`) |
| **Vector Storage** | ChromaDB (local persistence default), FAISS, Pinecone (cloud managed) |
| **Frontend UI** | Streamlit, custom design system tokens, Material icons |
| **Testing Suite** | Python standard library `unittest`, recorded golden-fixture regression suites |

---

## Repository Structure

```
CliniSight/
├── .gitignore                      # Git exclusion rules for secrets and private data
├── README.md                       # Project overview and setup documentation
├── pyproject.toml                  # Project packaging and metadata
├── client/                         # Streamlit frontend application
│   ├── app.py                      # Main client application entry point
│   ├── config.py                   # Client settings and endpoint definitions
│   ├── requirements.txt            # Client Python dependencies
│   ├── theme.py                    # Design tokens and visual style rules
│   ├── components/                 # UI section modules
│   │   ├── chatUI.py               # Grounded chat interface with pinned input
│   │   ├── dashboard.py            # Clinical overview dashboard and priority findings
│   │   ├── diet_precautions.py     # Diet guidance and lifestyle precautions
│   │   ├── disclaimer.py           # Safety disclaimers and advisory banners
│   │   ├── doctor_questions.py     # Doctor visit preparation questions
│   │   ├── history_download.py     # Conversation export and summary download
│   │   ├── upload.py               # File upload handler
│   │   └── verification.py         # Stage E human verification interface
│   └── utils/                      # Client helper utilities
│       ├── api.py                  # HTTP client connector for backend routes
│       ├── compat.py               # Version compatibility wrappers
│       └── emoji.py                # Plain-text sanitization utility
├── samples/                        # Synthetic sample data for testing
│   └── sample_blood_report.pdf     # Synthetic laboratory report with fictional values
├── server/                         # FastAPI backend service
│   ├── .env.example                # Template for environment variables
│   ├── config.py                   # Server configuration models
│   ├── logger.py                   # Structured logging utility
│   ├── main.py                     # FastAPI application entry point
│   ├── requirements.txt            # Server Python dependencies
│   ├── middlewares/                # Server middlewares
│   │   └── exception_handlers.py   # Global HTTP exception handling
│   ├── modules/                    # Core business logic and NLP engines
│   │   ├── chat_memory.py          # Multi-turn sliding memory manager
│   │   ├── display_helpers.py      # Bounding box and badge formatting helpers
│   │   ├── embeddings.py           # Gemini dense text embeddings integration
│   │   ├── image_preprocessor.py   # Stage A OpenCV preprocessing and deskew
│   │   ├── insights.py             # Insight generation, clusters, and precautions
│   │   ├── lab_indexer.py          # Clinical chunking and namespace indexing
│   │   ├── lab_parser.py           # Stages B-D vision extraction and deterministic flagging
│   │   ├── llm.py                  # Dr. Mira system prompts and output cleaning
│   │   ├── ocr_engine.py           # Stage B RapidOCR engine runtime
│   │   ├── query_handlers.py       # Grounded chat queries and emergency detection
│   │   ├── settings.py             # Environment paths and operational thresholds
│   │   └── vectorstore.py          # Unified vector store abstraction (Chroma/FAISS/Pinecone)
│   └── routes/                     # API route endpoints
│       ├── ask_question.py         # Conversational Q&A route
│       └── upload_report.py        # Report upload, verification, and summary routes
└── tests/                          # Automated test suites
    ├── fixtures/                   # Anonymized golden test fixtures
    ├── test_chat_rag.py            # Dialogue memory and emergency escalation tests
    ├── test_client_components.py   # Client component unit tests
    ├── test_lab_parser.py          # Parser, flagger, and clustering tests
    ├── test_stage_ab.py            # Image preprocessing and OCR tests
    └── test_stage_e.py             # Verification endpoint and integration tests
```

---

## Quick Setup and Execution

### 1. Backend Service (FastAPI)

```bash
# Navigate to the server directory
cd server

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit server/.env to provide your GOOGLE_API_KEY

# Start the FastAPI server
uvicorn main:app --reload --port 8000
```

### 2. Client Application (Streamlit)

```bash
# In a separate terminal, navigate to the client directory
cd client

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate

# Install dependencies (pinned to Streamlit 1.65.0)
pip install -r requirements.txt

# Start the Streamlit application
streamlit run app.py --server.port 8501
```

Access the web interface at `http://localhost:8501`. A synthetic test document is available under `samples/sample_blood_report.pdf` to test upload and verification immediately.

---

## Running Automated Tests

Run the test suite from the repository root:

```bash
# Run all tests using unittest discovery
PYTHONPATH=.:server python3 -m unittest discover -s tests -p "test_*.py"
```

---

## Acknowledgements and Origin

This project originated from the open-source [medicalAssistant](https://github.com/snsupratim/medicalAssistant) repository by Supratim Nag.

- **Inherited Architecture:** Initial FastAPI server scaffold, logger setup, exception handling middleware, and base dependency wiring.
- **Rebuilt Components:** The conversational query handlers, LLM persona prompts (`server/modules/llm.py`), client user interface (`client/app.py`, `client/components/chatUI.py`, `client/components/upload.py`), and client API connector were substantially redesigned and rewritten.
- **Original Additions in CliniSight:** The multi-stage laboratory report processing pipeline (Stages A through E: OpenCV preprocessing, RapidOCR cross-checking, Pass 2 completeness auditing, deterministic unit and scale normalization, Lakh scale conversion, physiological cluster detection, the two-column human verification interface, per-report isolated vector namespaces, and comprehensive automated test suites) were newly designed and implemented for CliniSight.
- Commit history was created retroactively; development happened iteratively in phases.

---

## Safety and Educational Notice

CliniSight is designed solely as an educational tool to assist patients in understanding laboratory blood test parameters and preparing for informed discussions with their healthcare providers. It does not provide medical diagnoses, prescribe treatments, or substitute for licensed clinical consultation. Always consult a qualified physician regarding any medical condition or laboratory finding.
