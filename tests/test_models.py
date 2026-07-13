import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np
from src.models.content_based import ContentBasedRecommender
from src.models.collaborative import CollaborativeRecommender
from src.models.hybrid import HybridRecommender

JOBS = pd.DataFrame([
    {"job_id": 101, "title": "Python Engineer", "company": "TechA",
     "description": "Python Django AWS developer needed.", "skills": "Python, Django, AWS",
     "experience_level": "Mid", "location": "Remote", "salary": "$100k"},
    {"job_id": 102, "title": "React Frontend Dev", "company": "DesignB",
     "description": "React TypeScript frontend developer.", "skills": "React, TypeScript",
     "experience_level": "Junior", "location": "NY", "salary": "$90k"},
    {"job_id": 103, "title": "Data Scientist", "company": "DataCo",
     "description": "Machine learning Python Pandas data scientist.", "skills": "Python, Machine Learning, Pandas",
     "experience_level": "Senior", "location": "SF", "salary": "$140k"},
])

INTERACTIONS = pd.DataFrame([
    {"candidate_id": "C001", "job_id": 101, "rating": 5.0},
    {"candidate_id": "C001", "job_id": 102, "rating": 2.0},
    {"candidate_id": "C002", "job_id": 103, "rating": 4.0},
    {"candidate_id": "C002", "job_id": 101, "rating": 3.0},
])

CAND = {
    "candidate_id": "C001",
    "target_title": "Python Developer",
    "experience_level": "Mid",
    "skills": "Python, Django",
    "resume_text": "Experienced Python backend developer."
}


# ── Content-Based Tests ──────────────────────────────────────────────────────
class TestContentBased:
    def setup_method(self):
        self.model = ContentBasedRecommender(use_sbert=False)
        self.model.fit_jobs(JOBS)

    def test_scores_shape(self):
        scores = self.model.predict_similarities(CAND, JOBS)
        assert len(scores) == len(JOBS)

    def test_scores_in_range(self):
        scores = self.model.predict_similarities(CAND, JOBS)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.01)   # allow tiny float tolerance

    def test_python_job_ranks_higher_than_react(self):
        scores = self.model.predict_similarities(CAND, JOBS)
        idx_python = JOBS[JOBS["job_id"] == 101].index[0]
        idx_react  = JOBS[JOBS["job_id"] == 102].index[0]
        # Python job should score higher for a Python candidate
        assert scores[idx_python] >= scores[idx_react]


# ── Collaborative Filtering Tests ────────────────────────────────────────────
class TestCollaborative:
    def setup_method(self):
        self.model = CollaborativeRecommender(embedding_dim=4)
        self.model.train_model(INTERACTIONS)

    def test_prediction_numeric(self):
        pred = self.model.predict_rating("C001", 101)
        assert isinstance(pred, float)

    def test_cold_start_returns_global_mean(self):
        pred = self.model.predict_rating("UNKNOWN_USER", 9999)
        assert 0.0 < pred < 10.0  # not NaN, not extreme

    def test_trained_flag(self):
        assert self.model.is_trained is True


# ── Hybrid Tests ─────────────────────────────────────────────────────────────
class TestHybrid:
    def setup_method(self):
        self.hybrid = HybridRecommender(alpha=0.5, use_sbert=False)
        self.hybrid.fit(JOBS, INTERACTIONS)

    def test_returns_dataframe(self):
        recs = self.hybrid.recommend(CAND, JOBS, top_k=2)
        assert isinstance(recs, pd.DataFrame)

    def test_top_k_respected(self):
        recs = self.hybrid.recommend(CAND, JOBS, top_k=2)
        assert len(recs) <= 2

    def test_hybrid_score_column_exists(self):
        recs = self.hybrid.recommend(CAND, JOBS, top_k=3)
        assert "hybrid_score" in recs.columns

    def test_scores_in_valid_range(self):
        recs = self.hybrid.recommend(CAND, JOBS, top_k=3)
        assert recs["hybrid_score"].between(0.0, 1.01).all()

    def test_sorted_descending(self):
        recs = self.hybrid.recommend(CAND, JOBS, top_k=3)
        scores = recs["hybrid_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_cold_start_user_still_works(self):
        cold_cand = {"candidate_id": "NEW_USER", "target_title": "Engineer",
                     "experience_level": "Mid", "skills": "Python", "resume_text": "New user."}
        recs = self.hybrid.recommend(cold_cand, JOBS, top_k=2)
        assert len(recs) >= 1
