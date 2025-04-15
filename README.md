# Project: MVP Evaluation Pipeline Backend
## Version: 0.1 (Initial MVP)
## Date: 2025-03-29

## 1. Project Overview

This project is a backend service (built with FastAPI) designed to automatically process text snippets (simulating performance report entries) to identify mentioned employees and predefined skill expectations they have potentially met. It uses semantic similarity via sentence embeddings and pgvector for matching expectations and records confirmed achievements in a PostgreSQL database.

## 2. Current Status

* **MVP v0.1 Complete:** The core end-to-end backend functionality as defined in the initial MVP scope has been implemented and tested.
* The service can receive text, identify a known employee in a sentence, find the semantically closest matching expectation, check if the match is below a similarity threshold, and record the achievement in the database.
* GPU acceleration (CUDA) via PyTorch is enabled and verified for ML model inference.

## 3. Project Structure & File Descriptions

<pre>
MVP_EvaluationPipeline/
├── venv/                     # Python virtual environment
├── app/                      # Main application source code module
│   ├── __init__.py
│   ├── api/                  # API endpoint definitions (FastAPI Routers)
│   │   ├── __init__.py
│   │   ├── endpoints/        # Specific endpoint files
│   │   │   ├── __init__.py
│   │   │   └── evaluation.py # Contains `/process_snippet` & `/test/employees` endpoints
│   │   └── router.py         # Aggregates endpoint routers
│   ├── core/                 # Core business logic & configuration
│   │   ├── __init__.py
│   │   ├── pipeline.py       # Main achievement processing pipeline logic
│   │   └── config.py         # Settings management (loads from .env)
│   ├── db/                   # Database interaction & schemas
│   │   ├── __init__.py
│   │   ├── database.py       # Database connection functions
│   │   └── schemas.py        # Pydantic models for API request/response validation
│   └── models/               # ML model loading logic
│       ├── __init__.py
│       └── loader.py         # Functions to load NER & Sentence Transformer models on startup
├── scripts/                  # Utility / helper scripts
│   ├── __init__.py
│   ├── generate_embeddings.py # Script to calculate and store embeddings for Expectations
│   ├── download_nltk_data.py  # Script to pre-download NLTK data ('punkt', 'punkt_tab')
│   ├── download_ner_model.py  # Script to pre-download NER model ('dslim/bert-base-NER')
│   ├── check_gpu.py           # Script to test PyTorch CUDA availability
│   └── test_nltk_punkt.py     # Script to isolate NLTK sentence tokenization test
├── .env                      # Environment variables (DB credentials, threshold) - 
├── .gitignore                # Files/folders for Git to ignore (venv, .env, __pycache__, etc.)
├── main.py                   # FastAPI application entry point (creates app, includes router, runs lifespan)
├── requirements.txt          # Python package dependencies
└── README.MD         # This file
</pre>

## 4. Core Functionality Implemented (MVP v0.1)

1.  **API Endpoint (`POST /evaluation/process_snippet`):** Accepts JSON payload `{"text": "..."}`.
2.  **Sentence Segmentation:** Input text is split into sentences using `nltk.sent_tokenize`.
3.  **NER & Employee Lookup:**
    * Each sentence processed by NER model (`dslim/bert-base-NER`) loaded via `transformers.pipeline`.
    * Identifies PERSON entities.
    * *MVP Simplification:* Processes only sentences with exactly one PERSON found.
    * Looks up the identified name in the `Employees` PostgreSQL table.
4.  **Semantic Matching:**
    * If a known employee is found, the sentence embedding is generated using `sentence-transformers` (`all-MiniLM-L6-v2`).
    * A PostgreSQL query using `pgvector`'s cosine distance operator (`<=>`) finds the closest `Expectation` based on pre-computed embeddings stored in the `Expectations` table.
5.  **Threshold Check:** The distance from the semantic match is compared against `settings.SIMILARITY_THRESHOLD`.
6.  **Database Recording:**
    * If the distance is below the threshold, a new record is inserted into the `EmployeeAchievements` table.
    * Stores `employee_id`, `expectation_id`, `date_achieved` (current date from DB), and the `evidence_snippet` (the sentence).

## 5. Key Technologies Used

* Python 3.13
* FastAPI: Web framework for the API.
* Uvicorn: ASGI server to run FastAPI.
* PostgreSQL: Relational database.
* pgvector: PostgreSQL extension for vector similarity search.
* psycopg2-binary: Python driver for PostgreSQL.
* sentence-transformers: For generating text embeddings (`all-MiniLM-L6-v2`).
* transformers: For NER pipeline (`dslim/bert-base-NER`).
* nltk: For sentence tokenization (`punkt`).
* PyTorch: Backend ML framework, configured with CUDA support for GPU acceleration.
* pydantic / pydantic-settings: For data validation and settings management.
* python-dotenv: For loading environment variables from `.env`.

## 6. How to Run

1.  **Prerequisites:** Python 3.13+, PostgreSQL installed with `pgvector` extension enabled. Git (optional).
2.  **Setup:**
    * Clone the repository (if applicable).
    * Navigate to the project root directory.
    * Create and activate a Python virtual environment:
        ```bash
        python -m venv venv
        .\venv\Scripts\activate # Windows
        # source venv/bin/activate # Linux/macOS
        ```
    * Install dependencies: `pip install -r requirements.txt`
3.  **Environment:** Create a `.env` file in the project root based on `.env.example` (if provided) or add necessary variables (DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, SIMILARITY_THRESHOLD).
4.  **Database:**
    * Ensure PostgreSQL server is running.
    * Connect to Postgres and create the database (e.g., `CREATE DATABASE mvp_eval_db;`).
    * Connect to the new database and enable pgvector (`CREATE EXTENSION vector;`).
    * Create the tables using SQL commands provided separately (or via a migration tool if added later).
    * Populate `Employees` and `Expectations` (text) tables with initial data.
5.  **Prepare ML Resources:**
    * Generate expectation embeddings: `python scripts/generate_embeddings.py`
    * Download NLTK data: `python scripts/download_nltk_data.py`
    * Download NER model: `python scripts/download_ner_model.py`
6.  **Run API Server:**
    ```bash
    uvicorn main:app --reload
    ```
7.  **Test:** Access `http://127.0.0.1:8000/docs` in a browser or use tools like `curl`/Postman to send `POST` requests to `http://127.0.0.1:8000/evaluation/process_snippet` with a JSON body like `{"text": "..."}`.

## 7. Known Limitations / MVP Simplifications

* Only processes single text snippets via API, no file ingestion (PDF, DOCX).
* Only processes sentences containing exactly one recognized employee found in the DB. Sentences with zero, multiple, or unknown persons are skipped.
* No coreference resolution (doesn't understand "he/she/they").
* Basic sentence segmentation (`nltk`), may struggle with complex structures.
* No handling of negation or complex sentiment that might contradict an achievement.
* Uses general pre-trained ML models; not fine-tuned for specific company language or edge cases.
* Records achievements automatically based on threshold; no human review step implemented *yet*.
* Minimal error handling in some pipeline steps.
* Limited testing (manual API calls).

## 8. Potential Next Steps / Enhancements

* **Accuracy & Validation:**
    * **Implement Human Review Workflow:** Present potential matches to a user for confirmation before saving (HIGH PRIORITY).
    * Tune `SIMILARITY_THRESHOLD`.
    * Implement Negation/Sentiment Checking.
    * Define strategy for sentences matching multiple expectations.
* **Input & Context:**
    * Add parsing for real report formats (DOCX, PDF).
    * Implement Coreference Resolution.
    * Improve sentence segmentation robustness.
    * Refine logic for handling multiple/zero/unknown employees per sentence.
* **Model Improvement:**
    * **Fine-tune NER model:** If pre-trained model struggles with company-specific names/formats (requires labeled data, separate process).
    * **Fine-tune Sentence Transformer:** (Advanced) If semantic matching needs improvement for specific expectation nuances.
* **Production Readiness:**
    * Add comprehensive unit and integration tests.
    * Improve DB connection management (e.g., connection pooling).
    * Enhance logging and error handling.
    * Containerize (Docker).
    * Implement API authentication/authorization.
