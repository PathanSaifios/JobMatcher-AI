import os
from pathlib import Path

# Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_STORE_DIR = BASE_DIR / "models_store"

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_STORE_DIR.mkdir(parents=True, exist_ok=True)

# File Paths
RAW_JOBS_CSV = RAW_DATA_DIR / "linkedin_jobs.csv"
PROCESSED_JOBS_CSV = PROCESSED_DATA_DIR / "jobs_cleaned.csv"
CANDIDATES_CSV = PROCESSED_DATA_DIR / "candidate_profiles.csv"
INTERACTIONS_CSV = PROCESSED_DATA_DIR / "interactions.csv"

# Model Paths
CF_MODEL_WEIGHTS = MODEL_STORE_DIR / "cf_model.pt"
CONTENT_EMBEDDINGS_NPZ = MODEL_STORE_DIR / "content_embeddings.npz"

# Hyperparameters
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CF_EMBEDDING_DIM = 16
CF_EPOCHS = 15
CF_BATCH_SIZE = 64
CF_LEARNING_RATE = 0.005
HYBRID_ALPHA = 0.5  # Weight for content-based score (1 - alpha for collaborative)

# App Configs
API_PORT = 8000
API_HOST = "0.0.0.0"
