import pandas as pd
from fastapi import APIRouter, HTTPException
from typing import List
from api.schemas import CandidateProfile, RecommendationResponse, SkillExtractionRequest, SkillExtractionResponse, JobPosting
from src.inference.recommend import recommend_jobs
from src.inference.explain import explain_recommendation
from src.data.skill_extraction import extract_skills
from src.config import PROCESSED_JOBS_CSV

router = APIRouter()

@router.post("/recommend", response_model=List[RecommendationResponse])
def get_recommendations(profile: CandidateProfile, top_k: int = 5):
    """Generates hybrid job recommendations based on candidate profile and simulates explanation details."""
    try:
        candidate_dict = profile.model_dump()
        
        # 1. Generate recommendations
        recs_df = recommend_jobs(candidate_dict, top_k=top_k)
        
        # 2. Enrich recommendations with explanatory details
        response_list = []
        for _, row in recs_df.iterrows():
            job_dict = row.to_dict()
            explanation = explain_recommendation(candidate_dict, job_dict)
            
            response_list.append(
                RecommendationResponse(
                    job_id=int(row["job_id"]),
                    title=str(row["title"]),
                    company=str(row["company"]),
                    location=str(row["location"]),
                    salary=str(row["salary"]),
                    skills=str(row["skills"]),
                    description=str(row["description"]),
                    content_score=float(row["content_score"]),
                    collaborative_score=float(row["collaborative_score"]),
                    hybrid_score=float(row["hybrid_score"]),
                    explanation=explanation
                )
            )
            
        return response_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation engine error: {str(e)}")

@router.post("/extract-skills", response_model=SkillExtractionResponse)
def get_extracted_skills(request: SkillExtractionRequest):
    """Extracts technical and soft skills from raw input text."""
    try:
        skills = extract_skills(request.text)
        return SkillExtractionResponse(extracted_skills=skills)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Skill extraction error: {str(e)}")

@router.get("/jobs", response_model=List[JobPosting])
def list_jobs(limit: int = 50):
    """Retrieves all available job listings in the system."""
    try:
        if not pd.io.common.file_exists(PROCESSED_JOBS_CSV):
            raise HTTPException(status_code=404, detail="Processed jobs catalog not found. Please train models first.")
            
        df = pd.read_csv(PROCESSED_JOBS_CSV)
        # Limit the results
        df = df.head(limit)
        
        jobs_list = []
        for _, row in df.iterrows():
            jobs_list.append(
                JobPosting(
                    job_id=int(row["job_id"]),
                    title=str(row["title"]),
                    company=str(row["company"]),
                    location=str(row["location"]),
                    description=str(row["description"]),
                    salary=str(row["salary"]),
                    experience_level=str(row["experience_level"]),
                    skills=str(row["skills"])
                )
            )
        return jobs_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing jobs: {str(e)}")
