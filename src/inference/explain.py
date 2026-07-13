import re

from src.data.skill_extraction import extract_skills


def _split_skills(value) -> list[str]:
    return [skill.strip() for skill in str(value or "").split(",") if skill.strip()]


def _tokenize_title(value) -> set[str]:
    stopwords = {"a", "an", "and", "for", "in", "of", "the", "to", "with"}
    return {
        token
        for token in re.findall(r"[a-z0-9+#.]+", str(value).lower())
        if token not in stopwords and len(token) > 1
    }


def explain_recommendation(candidate_profile: dict, job_row: dict) -> dict:
    """Generates a detailed explanation for why a job was recommended to a candidate.
    
    Args:
        candidate_profile (dict): Candidate profile dictionary containing skills and target_title.
        job_row (dict): A row/dictionary representing the recommended job.
        
    Returns:
        dict: Detailed explanation metrics, including matched skills, missing skills, 
              match percentage, and textual reason.
    """
    # 1. Skills matching, including skills that appear only inside pasted resume text.
    candidate_skill_names = _split_skills(candidate_profile.get("skills", ""))
    resume_text = str(candidate_profile.get("resume_text", ""))
    if resume_text:
        candidate_skill_names.extend(extract_skills(resume_text))

    cand_skills = {skill.lower() for skill in candidate_skill_names}
    job_skills = {skill.lower() for skill in _split_skills(job_row.get("skills", ""))}
    
    # Original capitalization mappings
    job_skills_orig = _split_skills(job_row.get("skills", ""))
    job_skills_map = {s.lower(): s for s in job_skills_orig}
    
    matched_skills_lower = cand_skills.intersection(job_skills)
    missing_skills_lower = job_skills.difference(cand_skills)
    
    matched_skills = [job_skills_map[s] for s in matched_skills_lower if s in job_skills_map]
    missing_skills = [job_skills_map[s] for s in missing_skills_lower if s in job_skills_map]
    
    # Calculate skill match percentage
    num_job_skills = len(job_skills)
    skill_match_percentage = (len(matched_skills) / num_job_skills * 100.0) if num_job_skills > 0 else 100.0
    
    # 2. Experience Level Match
    cand_exp = candidate_profile.get("experience_level", "Mid")
    job_exp = job_row.get("experience_level", "Mid")
    exp_match = cand_exp == job_exp
    
    # 3. Title match
    cand_title_words = _tokenize_title(candidate_profile.get("target_title", ""))
    job_title_words = _tokenize_title(job_row.get("title", ""))
    title_overlap = len(cand_title_words.intersection(job_title_words)) > 0
    
    # 4. Generate Explanatory Reason
    reasons = []
    
    # Core skill reason
    if len(matched_skills) > 0:
        reasons.append(f"you have core skills required for this role: {', '.join(matched_skills[:4])}")
        if len(matched_skills) > 4:
            reasons.append(f"plus {len(matched_skills) - 4} other matching skills")
    else:
        reasons.append("your overall profile text is close to the role requirements")
        
    # Experience match reason
    if exp_match:
        reasons.append(f"the experience level matches your profile ({cand_exp})")
    else:
        reasons.append(f"this is a {job_exp} level role while your profile indicates {cand_exp}")
        
    # Title match reason
    if title_overlap:
        reasons.append("the job title aligns directly with your career objective")
        
    # Collaborative filtering reason
    collab_score = float(job_row.get("collaborative_score", 0.0))
    collab_used = bool(job_row.get("collaborative_used", False))
    if collab_used and collab_score > 0.6:
        reasons.append("this job is trending highly among other candidates with similar skill profiles")
    elif not collab_used:
        reasons.append("the ranking uses content-based matching because this profile is new to the interaction model")
        
    explanation_text = "Recommended because " + "; ".join(reasons) + "."
    
    # Match Strengths
    hybrid_score = float(job_row.get("hybrid_score", 0.0))
    if skill_match_percentage >= 70 and hybrid_score >= 0.45:
        match_strength = "Strong Match"
        match_color = "green"
    elif skill_match_percentage >= 40 or hybrid_score >= 0.35:
        match_strength = "Good Match"
        match_color = "blue"
    else:
        match_strength = "Potential Match"
        match_color = "orange"
        
    return {
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "skill_match_percentage": round(skill_match_percentage, 1),
        "experience_level_match": exp_match,
        "title_alignment": title_overlap,
        "explanation": explanation_text,
        "match_strength": match_strength,
        "match_color": match_color
    }

if __name__ == "__main__":
    cand = {"skills": "Python, Django, AWS, Git", "experience_level": "Mid", "target_title": "Python Backend Developer"}
    job = {"skills": "Python, AWS, PostgreSQL, Docker", "experience_level": "Mid", "title": "Senior Python Backend Developer", "collaborative_score": 0.8}
    
    print("Explanation:", explain_recommendation(cand, job))
