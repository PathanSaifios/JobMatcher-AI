# JobMatch AI - Hybrid Job Recommendation System

JobMatch AI is an end-to-end AI/ML project that recommends jobs to candidates using a hybrid recommendation engine. It combines content-based matching, collaborative filtering, explainable recommendations, a FastAPI backend, and a Streamlit dashboard.

## Features

| Feature | Detail |
|---|---|
| Content-based matching | Sentence-BERT embeddings when available, with a TF-IDF fallback for lightweight local runs |
| Collaborative filtering | PyTorch matrix factorization trained on simulated candidate-job interactions |
| Hybrid blending | Tunable content/collaborative weight with cold-start fallback to content scores |
| Explainability | Matched skills, missing skills, experience alignment, title alignment, and reasoning text |
| REST API | FastAPI endpoints for recommendations, skill extraction, jobs, and health checks |
| Dashboard | Streamlit app with login, resume upload, profile editing, recommendations, platform search links, and analytics |
| Data pipeline | Synthetic job catalog generation, preprocessing, candidate simulation, interaction simulation, and model training |

## Project Structure

```text
AIML Project/
|-- api/
|   |-- main.py
|   |-- routes.py
|   `-- schemas.py
|-- app/
|   `-- streamlit_app.py
|-- data/
|   |-- download_data.py
|   |-- raw/
|   `-- processed/
|-- models_store/
|-- src/
|   |-- config.py
|   |-- data/
|   |-- evaluation/
|   |-- inference/
|   |-- models/
|   `-- training/
|-- tests/
|-- Dockerfile
|-- requirements.txt
`-- README.md
```

## Quick Start

1. Create and activate a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

3. Build the local data and model artifacts.

```powershell
python data\download_data.py
python src\training\train_hybrid.py
```

4. Run the Streamlit dashboard.

```powershell
.\.venv\Scripts\streamlit.exe run app\streamlit_app.py
```

Open http://localhost:8501 in your browser.

Default local login:

```text
Email: student@jobmatch.ai
Password: jobmatch123
```

You can override these in environment variables:

```text
JOBMATCH_EMAIL
JOBMATCH_PASSWORD
JOBMATCH_MONGODB_URI
JOBMATCH_DB_NAME
```

## FastAPI Backend

Run the API server:

```powershell
.\.venv\Scripts\uvicorn.exe api.main:app --reload --port 8000
```

Useful URLs:

```text
http://localhost:8000/
http://localhost:8000/health
http://localhost:8000/docs
```

Main API endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/v1/recommend` | Return ranked job recommendations |
| POST | `/api/v1/extract-skills` | Extract known skills from text |
| GET | `/api/v1/jobs` | List processed job postings |
| GET | `/health` | Health check |

## VS Code

Open this folder in VS Code, select the project interpreter at `.venv\Scripts\python.exe`,
then run one of the launch configurations:

```text
Streamlit Dashboard
FastAPI Server
Train ML Pipeline
Generate Dataset
Run All Tests
```

You can also use Terminal > Run Task for:

```text
Run Streamlit Dashboard
Run FastAPI Server
Run Tests
Train Hybrid Model
```

## Deployment Notes

The Streamlit dashboard is the main user interface and should be deployed on a service
that supports a long-running Streamlit process, such as Streamlit Community Cloud or a
Docker-capable host.

Vercel's Python runtime is designed for ASGI/WSGI apps and serverless functions. This
project includes a compatible FastAPI app at `api.main:app`, so Vercel is a reasonable
target for the API layer. The Streamlit UI itself is not a native Vercel app unless it is
rebuilt as a supported web frontend.

## Testing

Run the full test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Current coverage includes skill extraction, preprocessing, model scoring, hybrid ranking behavior, and API health/skill endpoints.

## Training Pipeline

For a full rebuild from generated raw data:

```powershell
python data\download_data.py
python src\training\train_hybrid.py
```

The pipeline creates:

```text
data/raw/linkedin_jobs.csv
data/processed/jobs_cleaned.csv
data/processed/candidate_profiles.csv
data/processed/interactions.csv
models_store/cf_model.pt
models_store/content_embeddings.npz
```

Generated data, trained model artifacts, local users, virtual environments, and `.env` files are ignored by Git.

## Docker

```powershell
docker build -t jobmatch-ai .
docker run -p 8501:8501 jobmatch-ai
```

The Docker image builds the synthetic dataset and trains the model during the image build, then starts the Streamlit dashboard by default.

## Tech Stack

- Python
- pandas and NumPy
- scikit-learn
- sentence-transformers
- PyTorch
- FastAPI and Uvicorn
- Streamlit
- pymongo
- pypdf and python-docx
- pytest
