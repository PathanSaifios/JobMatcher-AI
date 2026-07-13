import pandas as pd
import numpy as np
import re
from src.config import HYBRID_ALPHA
from src.models.content_based import ContentBasedRecommender
from src.models.collaborative import CollaborativeRecommender

try:
    from src.data.skill_extraction import extract_skills
except Exception:
    extract_skills = None

class HybridRecommender:
    def __init__(self, alpha: float = HYBRID_ALPHA, use_sbert: bool = True):
        self.alpha = alpha
        self.content_model = ContentBasedRecommender(use_sbert=use_sbert)
        self.collab_model = CollaborativeRecommender()
        self.is_fitted = False

    @staticmethod
    def _split_skills(value) -> set[str]:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return set()
        return {
            skill.strip().lower()
            for skill in str(value).split(",")
            if skill.strip()
        }

    @staticmethod
    def _tokenize_title(value) -> set[str]:
        stopwords = {"a", "an", "and", "for", "in", "of", "the", "to", "with"}
        return {
            token
            for token in re.findall(r"[a-z0-9+#.]+", str(value).lower())
            if token not in stopwords and len(token) > 1
        }

    @staticmethod
    def _experience_score(candidate_level: str, job_level: str) -> float:
        order = {"Junior": 0, "Mid": 1, "Senior": 2, "Lead": 3}
        cand_idx = order.get(str(candidate_level), 1)
        job_idx = order.get(str(job_level), 1)
        distance = abs(cand_idx - job_idx)

        if distance == 0:
            return 1.0
        if distance == 1:
            return 0.65
        return 0.3

    def _structured_match_scores(self, candidate_profile: dict, jobs_df: pd.DataFrame) -> np.ndarray:
        """Score direct skill, title, and experience alignment for clearer ranking."""
        candidate_skills = self._split_skills(candidate_profile.get("skills", ""))
        resume_text = str(candidate_profile.get("resume_text", ""))
        if extract_skills is not None and resume_text:
            candidate_skills.update(skill.lower() for skill in extract_skills(resume_text))

        candidate_title_tokens = self._tokenize_title(candidate_profile.get("target_title", ""))
        candidate_level = candidate_profile.get("experience_level", "Mid")

        scores = []
        for _, row in jobs_df.iterrows():
            job_skills = self._split_skills(row.get("skills", ""))
            if job_skills:
                skill_score = len(candidate_skills.intersection(job_skills)) / len(job_skills)
            else:
                skill_score = 0.35

            job_title_tokens = self._tokenize_title(row.get("title", ""))
            if candidate_title_tokens and job_title_tokens:
                title_score = len(candidate_title_tokens.intersection(job_title_tokens)) / max(
                    1, min(len(candidate_title_tokens), len(job_title_tokens))
                )
            else:
                title_score = 0.0

            exp_score = self._experience_score(candidate_level, row.get("experience_level", "Mid"))
            structured_score = (0.55 * skill_score) + (0.30 * title_score) + (0.15 * exp_score)
            scores.append(float(np.clip(structured_score, 0.0, 1.0)))

        return np.array(scores)

    def fit(self, jobs_df: pd.DataFrame, interactions_df: pd.DataFrame):
        """Fits both Content-Based and Collaborative Filtering models."""
        print("Fitting Hybrid Recommender components...")
        self.content_model.fit_jobs(jobs_df)
        self.collab_model.train_model(interactions_df)
        self.is_fitted = True
        print("Hybrid Recommender fitted successfully.")

    def load_precomputed_models(self) -> bool:
        """Loads precomputed S-BERT embeddings and trained Collaborative model weights."""
        success_content = self.content_model.load_embeddings()
        success_collab = self.collab_model.load_model()
        self.is_fitted = success_content and success_collab
        return self.is_fitted

    def recommend(self, candidate_profile: dict, jobs_df: pd.DataFrame, top_k: int = 10, alpha: float | None = None) -> pd.DataFrame:
        """Generates top-K hybrid recommendations for a candidate."""
        alpha_value = self.alpha if alpha is None else float(np.clip(alpha, 0.0, 1.0))

        # 1. Compute content scores and reinforce them with explicit profile alignment.
        semantic_scores = self.content_model.predict_similarities(candidate_profile, jobs_df)
        semantic_scores = np.clip(semantic_scores, 0, 1)
        structured_scores = self._structured_match_scores(candidate_profile, jobs_df)
        cb_scores = np.clip((0.72 * semantic_scores) + (0.28 * structured_scores), 0, 1)
        
        # 2. Compute Collaborative scores
        candidate_id = candidate_profile.get("candidate_id", "COLD_START_USER")
        cf_raw_scores = []
        
        for job_id in jobs_df["job_id"]:
            # Predict rating (typically in range [1, 5])
            pred_rating = self.collab_model.predict_rating(candidate_id, job_id)
            cf_raw_scores.append(pred_rating)
            
        cf_raw_scores = np.array(cf_raw_scores)
        
        # Scale Collaborative scores from [1, 5] range to [0, 1]
        # Min rating is 1.0, Max is 5.0
        cf_scores = (cf_raw_scores - 1.0) / 4.0
        cf_scores = np.clip(cf_scores, 0, 1)
        
        # 3. Dynamic Hybrid Blending
        # Handle User Cold-Start: If the candidate has no interaction history, ignore CF and use CB
        is_cold_start_user = candidate_id not in self.collab_model.user_to_idx
        
        hybrid_scores = []
        collaborative_used = []
        for idx, job_id in enumerate(jobs_df["job_id"]):
            cb_s = cb_scores[idx]
            cf_s = cf_scores[idx]
            
            # Check if Job is Cold-Start (no one has interacted with it)
            is_cold_start_job = job_id not in self.collab_model.item_to_idx
            
            if is_cold_start_user or is_cold_start_job:
                # Fall back 100% on content-based similarity
                score = cb_s
                collaborative_used.append(False)
            else:
                # Blend scores
                score = (alpha_value * cb_s) + ((1 - alpha_value) * cf_s)
                collaborative_used.append(True)
                
            hybrid_scores.append(score)
            
        # 4. Compile and Sort Results
        recs_df = jobs_df.copy()
        recs_df["semantic_score"] = semantic_scores
        recs_df["structured_score"] = structured_scores
        recs_df["content_score"] = cb_scores
        recs_df["collaborative_score"] = cf_scores
        recs_df["collaborative_used"] = collaborative_used
        recs_df["hybrid_score"] = hybrid_scores
        recs_df["score_confidence"] = np.clip((0.7 * recs_df["hybrid_score"]) + (0.3 * structured_scores), 0, 1)
        
        # Sort by hybrid score descending
        recs_df = recs_df.sort_values(by=["hybrid_score", "structured_score"], ascending=False)
        
        return recs_df.head(top_k)

if __name__ == "__main__":
    # Test stub
    jobs = pd.DataFrame([
        {"job_id": 1001, "title": "Software Engineer", "company": "TechA", "description": "Python, SQL", "skills": "Python, SQL", "experience_level": "Mid"},
        {"job_id": 1002, "title": "Frontend dev", "company": "DesignB", "description": "React, HTML", "skills": "React, HTML", "experience_level": "Junior"}
    ])
    interactions = pd.DataFrame([
        {"candidate_id": "C001", "job_id": 1001, "rating": 5.0},
        {"candidate_id": "C001", "job_id": 1002, "rating": 2.0}
    ])
    
    hybrid = HybridRecommender(alpha=0.5, use_sbert=False)
    hybrid.fit(jobs, interactions)
    
    cand = {"candidate_id": "C001", "target_title": "Python Developer", "experience_level": "Mid", "skills": "Python, SQL", "resume_text": "Experienced Python coder."}
    recs = hybrid.recommend(cand, jobs, top_k=2)
    print("Recommendations:\n", recs[["job_id", "title", "hybrid_score", "content_score", "collaborative_score"]])
