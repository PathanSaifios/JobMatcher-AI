import numpy as np
import pandas as pd
from pathlib import Path
from src.config import EMBEDDING_MODEL_NAME, CONTENT_EMBEDDINGS_NPZ

# Global check for sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False
    print("Warning: sentence-transformers not found. Falling back to TF-IDF for content-based matching.")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    def __init__(self, use_sbert: bool = True):
        self.use_sbert = use_sbert and HAS_SBERT
        self.model = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.job_embeddings = None
        self.job_ids = None

        if self.use_sbert:
            try:
                print(f"Loading Sentence-BERT model: {EMBEDDING_MODEL_NAME}...")
                self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
                print("Sentence-BERT loaded successfully.")
            except Exception as e:
                print(f"Error loading Sentence-BERT: {e}. Falling back to TF-IDF.")
                self.use_sbert = False

        if not self.use_sbert:
            print("Using TF-IDF Vectorizer for Content-Based matching.")
            self.tfidf_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))

    def _prepare_job_text(self, job_row) -> str:
        """Combine job fields into a single text representation."""
        title = str(job_row.get("title", ""))
        company = str(job_row.get("company", ""))
        desc = str(job_row.get("description", ""))
        skills = str(job_row.get("skills", ""))
        level = str(job_row.get("experience_level", ""))
        
        return f"{title} at {company}. Role Level: {level}. Required Skills: {skills}. Description: {desc}"

    def fit_jobs(self, jobs_df: pd.DataFrame):
        """Fit content representations for all jobs and precompute embeddings."""
        print("Computing job embeddings...")
        self.job_ids = jobs_df["job_id"].tolist()
        texts = jobs_df.apply(self._prepare_job_text, axis=1).tolist()
        
        if self.use_sbert:
            self.job_embeddings = self.model.encode(
                texts, 
                show_progress_bar=True, 
                convert_to_numpy=True
            )
            # Save to disk
            np.savez(CONTENT_EMBEDDINGS_NPZ, embeddings=self.job_embeddings, job_ids=np.array(self.job_ids))
            print(f"Computed and saved S-BERT embeddings for {len(jobs_df)} jobs.")
        else:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            print(f"Computed TF-IDF matrix of shape {self.tfidf_matrix.shape} for {len(jobs_df)} jobs.")

    def load_embeddings(self):
        """Attempts to load precomputed embeddings from disk."""
        if self.use_sbert and Path(CONTENT_EMBEDDINGS_NPZ).exists():
            data = np.load(CONTENT_EMBEDDINGS_NPZ)
            self.job_embeddings = data["embeddings"]
            self.job_ids = data["job_ids"].tolist()
            print("Loaded S-BERT job embeddings from disk.")
            return True
        return False

    def _align_scores_to_jobs(self, scores: np.ndarray, jobs_df: pd.DataFrame) -> np.ndarray:
        """Return scores in the same order and length as the requested jobs."""
        if self.job_ids is None or "job_id" not in jobs_df.columns:
            return scores

        requested_ids = [str(job_id) for job_id in jobs_df["job_id"].tolist()]
        fitted_ids = [str(job_id) for job_id in self.job_ids]

        if requested_ids == fitted_ids:
            return scores

        score_by_job_id = {
            job_id: float(score)
            for job_id, score in zip(fitted_ids, scores)
        }
        return np.array([score_by_job_id.get(job_id, 0.0) for job_id in requested_ids])

    def predict_similarities(self, candidate_profile: dict, jobs_df: pd.DataFrame) -> np.ndarray:
        """Compute cosine similarity scores between a candidate profile and all jobs."""
        candidate_text = (
            f"Target: {candidate_profile.get('target_title', '')}. "
            f"Experience: {candidate_profile.get('experience_level', '')}. "
            f"Skills: {candidate_profile.get('skills', '')}. "
            f"Resume Summary: {candidate_profile.get('resume_text', '')}"
        )
        
        if self.use_sbert:
            # If embeddings are not loaded/computed, compute them on the fly
            if self.job_embeddings is None:
                if not self.load_embeddings():
                    self.fit_jobs(jobs_df)
                    
            cand_embedding = self.model.encode([candidate_text], convert_to_numpy=True)
            similarities = cosine_similarity(cand_embedding, self.job_embeddings).flatten()
        else:
            # TF-IDF matching
            if self.tfidf_matrix is None:
                self.fit_jobs(jobs_df)
                
            cand_tfidf = self.tfidf_vectorizer.transform([candidate_text])
            similarities = cosine_similarity(cand_tfidf, self.tfidf_matrix).flatten()
            
        return self._align_scores_to_jobs(similarities, jobs_df)

if __name__ == "__main__":
    # Test ContentBasedRecommender
    jobs_data = pd.DataFrame([
        {"job_id": 101, "title": "Python Engineer", "company": "Tech", "skills": "Python, SQL, Git", "description": "Writing backend web code."},
        {"job_id": 102, "title": "React Frontend Developer", "company": "Design", "skills": "React, JavaScript, CSS", "description": "Building interactive web UIs."}
    ])
    
    cand = {"target_title": "Python Developer", "experience_level": "Mid", "skills": "Python, Django", "resume_text": "Experienced web developer using python Django."}
    
    recommender = ContentBasedRecommender(use_sbert=False) # Test with TF-IDF fallback
    recommender.fit_jobs(jobs_data)
    scores = recommender.predict_similarities(cand, jobs_data)
    print("TF-IDF match scores:", scores)
