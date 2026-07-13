import pandas as pd
from pathlib import Path
from src.config import PROCESSED_JOBS_CSV
from src.models.hybrid import HybridRecommender

# Cache recommender instance
_hybrid_recommender = None

def get_recommender_instance() -> HybridRecommender:
    """Singleton pattern to load and cache the hybrid recommendation engine."""
    global _hybrid_recommender
    if _hybrid_recommender is None:
        print("Initializing and loading Hybrid Recommendation engine...")
        _hybrid_recommender = HybridRecommender(alpha=0.5, use_sbert=True)
        # Attempt to load precomputed weights and embeddings
        success = _hybrid_recommender.load_precomputed_models()
        if not success:
            print("Precomputed models/embeddings not found. Running complete pipeline fit...")
            # If not found, load data and fit on the fly
            from src.config import INTERACTIONS_CSV
            try:
                jobs_df = pd.read_csv(PROCESSED_JOBS_CSV)
                interactions_df = pd.read_csv(INTERACTIONS_CSV)
                _hybrid_recommender.fit(jobs_df, interactions_df)
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"Data files missing. Please run 'python src/training/train_hybrid.py' first. Error: {e}"
                )
    return _hybrid_recommender

def recommend_jobs(candidate_profile: dict, top_k: int = 5) -> pd.DataFrame:
    """Wrapper function to recommend jobs for a candidate.
    
    Args:
        candidate_profile (dict): A dictionary representing the candidate:
            - candidate_id (str, optional)
            - target_title (str)
            - experience_level (str)
            - skills (str - comma separated)
            - resume_text (str)
        top_k (int): Number of jobs to return.
        
    Returns:
        pd.DataFrame: Top jobs with hybrid, content, and collaborative scores.
    """
    recommender = get_recommender_instance()
    jobs_df = pd.read_csv(PROCESSED_JOBS_CSV)
    
    # Generate recommendations
    recs = recommender.recommend(candidate_profile, jobs_df, top_k=top_k)
    return recs

if __name__ == "__main__":
    test_cand = {
        "candidate_id": "C001",
        "target_title": "Python Developer",
        "experience_level": "Mid",
        "skills": "Python, Django, SQL",
        "resume_text": "Web developer with experience in python backends."
    }
    try:
        results = recommend_jobs(test_cand, top_k=3)
        print("Inference Test Output:\n", results[["job_id", "title", "company", "hybrid_score"]])
    except Exception as e:
        print(f"Inference Test skipped: {e}")
