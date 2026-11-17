# Syllabus Planner

Streamlit app: upload course syllabi (PDF) -> extract deadlines, exams and weights with an LLM -> combined timeline with crunch-week warnings -> .ics calendar export.

I'm a second-year CS student building this for a hackathon application. I must be able to explain every line, so prioritize clarity over cleverness.

## Stack
- Python 3.13, virtual environment in `.venv`
- Streamlit (UI), pdfplumber (PDF text), pydantic (validation)
- Gemini API (free tier) via the `google-genai` SDK
- `ics` (calendar export), pytest (tests), python-dotenv (secrets)

## Project structure
- `app.py`: Streamlit UI (entry point)
- `planner/models.py`: `Event` model (course, title, type, due_date, weight_percent, confidence)
- `planner/pdf_reader.py`: PDF -> raw text
- `planner/llm.py`: `call_llm(prompt) -> str`. The ONLY file that knows about Gemini.
- `planner/extractor.py`: builds the prompt, calls `llm.call_llm`, validates output into `Event` objects
- `planner/workload.py`: weekly weight totals and crunch-week flags
- `planner/calendar_export.py`: events -> .ics
- `tests/`: pytest tests and `fixtures/` (sample syllabi plus hand-labelled expected events)

## Rules
- Propose a plan before any large change, and wait for my approval.
- Keep the app runnable after every step. Make small, reviewable changes.
- Explain non-obvious code and design choices briefly.
- Small functions, type hints, one responsibility per module.
- All LLM calls go through `planner/llm.py`.
- Cache every LLM response to disk (keyed by a hash of the input text) so re-runs don't spend free-tier quota.
- Handle HTTP 429 rate-limit errors with retry and backoff, and pause briefly between calls.
- Never trust raw model output: validate it with pydantic and handle bad or missing dates and weights.
- Before using the Gemini SDK or choosing a model, look up the current docs and a currently free model. Don't guess names from memory.
- Write a pytest test for each new module where practical.

## Secrets and safety
- The API key lives in `.env` as `GEMINI_API_KEY`. Never print it, hardcode it, or commit it.
- `.env`, `.venv/` and `__pycache__/` must stay in `.gitignore`.
- Only use my own course syllabi as test data. No personal data.

## Commands
- Activate venv: `.venv\Scripts\activate`
- Install: `pip install -r requirements.txt`
- Run app: `streamlit run app.py`
- Test: `pytest`