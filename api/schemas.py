from pydantic import BaseModel, Field
from typing import List, Optional

class CandidateProfile(BaseModel):
    candidate_id: Optional[str] = Field(default="COLD_START_USER", description="Unique identifier for the candidate if they exist in the interaction logs.")
    target_title: str = Field(..., description="Target job title (e.g. 'Software Engineer').")
    experience_level: str = Field(default="Mid", description="Experience level: Junior, Mid, Senior, Lead.")
    skills: str = Field(..., description="Comma-separated list of candidate skills.")
    resume_text: str = Field(..., description="Raw text of the resume or candidate biography.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "candidate_id": "C001",
                "target_title": "Python Developer",
                "experience_level": "Mid",
                "skills": "Python, Django, SQL, Git, AWS",
                "resume_text": "Experienced Python Backend Developer with 3+ years of experience working with FastAPI and Django. Skilled in postgresql database management, dockerizing applications, and deploying in AWS."
            }
        }
    }

class SkillExtractionRequest(BaseModel):
    text: str = Field(..., description="The raw description or resume text from which to extract skills.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "We are seeking a Senior React Developer who knows TypeScript, Redux, and Node.js. Experience with AWS and Docker is preferred."
            }
        }
    }

class SkillExtractionResponse(BaseModel):
    extracted_skills: List[str]

class JobPosting(BaseModel):
    job_id: int
    title: str
    company: str
    location: str
    description: str
    salary: str
    experience_level: str
    skills: str

class RecommendationExplanation(BaseModel):
    matched_skills: List[str]
    missing_skills: List[str]
    skill_match_percentage: float
    experience_level_match: bool
    title_alignment: bool
    explanation: str
    match_strength: str
    match_color: str

class RecommendationResponse(BaseModel):
    job_id: int
    title: str
    company: str
    location: str
    salary: str
    skills: str
    description: str
    content_score: float
    collaborative_score: float
    hybrid_score: float
    explanation: RecommendationExplanation
