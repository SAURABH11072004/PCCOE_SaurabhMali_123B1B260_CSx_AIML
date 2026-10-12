# AUTOSAR HLD Document Analysis Assistant — Source Code

## Quick Start
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env        # then add your GOOGLE_API_KEY
streamlit run app.py
```

## Run Tests
```powershell
pytest tests/ -v
```

## Run Evaluation
```powershell
python -m evaluation.run_evaluation
```

## Notes
- `.env.example` with placeholder credentials is in `Model_Prompts_Config/`.
- Embedding model (`all-MiniLM-L6-v2`) downloads automatically on first use.
- ChromaDB data directory is created at runtime (`./chroma_db/`).
