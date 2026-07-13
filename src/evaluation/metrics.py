import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def precision_at_k(recommended_ids: list, actual_interacted_ids: list, k: int) -> float:
    """Computes Precision@K: portion of recommended items that were actually interacted with."""
    if not recommended_ids or not actual_interacted_ids:
        return 0.0
    
    recs = recommended_ids[:k]
    intersect = set(recs).intersection(set(actual_interacted_ids))
    return len(intersect) / len(recs)

def recall_at_k(recommended_ids: list, actual_interacted_ids: list, k: int) -> float:
    """Computes Recall@K: portion of actual interacted items captured in recommendations."""
    if not recommended_ids or not actual_interacted_ids:
        return 0.0
    
    recs = recommended_ids[:k]
    intersect = set(recs).intersection(set(actual_interacted_ids))
    return len(intersect) / len(actual_interacted_ids)

def catalog_coverage(all_recommendations: list[list], total_jobs_in_catalog: int) -> float:
    """Computes catalog coverage: percentage of total jobs that are recommended at least once."""
    unique_recommended = set()
    for recs in all_recommendations:
        unique_recommended.update(recs)
        
    return len(unique_recommended) / total_jobs_in_catalog if total_jobs_in_catalog > 0 else 0.0

def recommendation_diversity(recommended_jobs_df: pd.DataFrame) -> float:
    """Computes recommendation diversity: average pairwise cosine distance (1 - similarity) of recommended jobs."""
    if len(recommended_jobs_df) <= 1:
        return 0.0
        
    # Combine title and description to represent jobs
    texts = (recommended_jobs_df["title"] + " " + recommended_jobs_df["description"]).fillna("").tolist()
    
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(texts)
    
    sim_matrix = cosine_similarity(tfidf)
    
    # Extract upper triangular matrix excluding diagonal
    n = len(recommended_jobs_df)
    upper_tri_indices = np.triu_indices(n, k=1)
    similarities = sim_matrix[upper_tri_indices]
    
    # Diversity = 1 - similarity
    distances = 1.0 - similarities
    return float(np.mean(distances))

def evaluate_recommender(hybrid_recommender, candidates_df: pd.DataFrame, jobs_df: pd.DataFrame, interactions_df: pd.DataFrame, k: int = 5) -> dict:
    """Runs a full evaluation over candidates with interaction history."""
    precisions = []
    recalls = []
    all_recs = []
    
    # Filter interactions to only positive ones (rating >= 3.0 / view, bookmark, apply)
    positive_interactions = interactions_df[interactions_df['rating'] >= 3.0]
    
    # Only evaluate candidates who have at least one positive interaction
    evaluated_candidates = positive_interactions['candidate_id'].unique()
    
    for cand_id in evaluated_candidates:
        cand_row = candidates_df[candidates_df['candidate_id'] == cand_id].iloc[0]
        actual_jobs = positive_interactions[positive_interactions['candidate_id'] == cand_id]['job_id'].tolist()
        
        # Candidate dictionary
        cand_profile = {
            "candidate_id": cand_row["candidate_id"],
            "target_title": cand_row["target_title"],
            "experience_level": cand_row["experience_level"],
            "skills": cand_row["skills"],
            "resume_text": cand_row["resume_text"]
        }
        
        # Get recommendations
        recs = hybrid_recommender.recommend(cand_profile, jobs_df, top_k=k)
        rec_ids = recs["job_id"].tolist()
        
        all_recs.append(rec_ids)
        precisions.append(precision_at_k(rec_ids, actual_jobs, k))
        recalls.append(recall_at_k(rec_ids, actual_jobs, k))
        
    avg_precision = np.mean(precisions) if precisions else 0.0
    avg_recall = np.mean(recalls) if recalls else 0.0
    coverage = catalog_coverage(all_recs, len(jobs_df))
    
    # Sample diversity for one set of recommendations (e.g. first candidate)
    sample_div = 0.0
    if evaluated_candidates.size > 0:
        first_cand = candidates_df[candidates_df['candidate_id'] == evaluated_candidates[0]].iloc[0].to_dict()
        recs_sample = hybrid_recommender.recommend(first_cand, jobs_df, top_k=k)
        sample_div = recommendation_diversity(recs_sample)
        
    return {
        f"Precision@{k}": avg_precision,
        f"Recall@{k}": avg_recall,
        "Catalog Coverage": coverage,
        "Recommendation Diversity": sample_div
    }
