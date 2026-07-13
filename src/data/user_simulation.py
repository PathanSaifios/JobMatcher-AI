import pandas as pd
import numpy as np
import random
from pathlib import Path
from src.config import PROCESSED_JOBS_CSV, CANDIDATES_CSV, INTERACTIONS_CSV
from src.data.skill_extraction import SKILL_PATTERNS

# Seed for reproducibility
random.seed(42)
np.random.seed(42)

def generate_candidates(num_candidates=80):
    print(f"Simulating {num_candidates} candidate profiles...")
    
    # Load processed jobs to see what skills/titles we have
    jobs_df = pd.read_csv(PROCESSED_JOBS_CSV)
    
    # Extract unique skills from jobs
    all_job_skills = set()
    for s_list in jobs_df['skills'].dropna():
        all_job_skills.update([s.strip() for s in s_list.split(',')])
    all_job_skills = list(all_job_skills)
    
    # Define template candidates
    profiles_templates = [
        ("Frontend Engineer", ["React", "JavaScript", "TypeScript", "HTML", "CSS", "Tailwind CSS", "Figma"], "Junior"),
        ("Senior Frontend Engineer", ["React", "TypeScript", "Next.js", "Redux", "GraphQL", "Git", "Webpack"], "Senior"),
        ("Python Backend Developer", ["Python", "Django", "FastAPI", "SQL", "PostgreSQL", "Docker", "Git"], "Mid"),
        ("Backend Engineer", ["Go", "Node.js", "MongoDB", "Redis", "Docker", "REST API", "CI/CD"], "Senior"),
        ("Data Scientist", ["Python", "Pandas", "NumPy", "Scikit-Learn", "SQL", "Tableau", "Machine Learning"], "Mid"),
        ("Machine Learning Engineer", ["Python", "PyTorch", "TensorFlow", "Docker", "AWS", "Scikit-Learn", "Machine Learning"], "Senior"),
        ("Data Analyst", ["SQL", "Excel", "Tableau", "PowerBI", "Python", "Data Visualization"], "Junior"),
        ("DevOps Engineer", ["AWS", "Docker", "Kubernetes", "Terraform", "CI/CD", "Linux", "Python"], "Senior"),
        ("Product Manager", ["Agile", "Scrum", "Jira", "Product Roadmap", "Communication", "A/B Testing"], "Mid"),
        ("UX/UI Designer", ["Figma", "Adobe XD", "Wireframing", "User Research", "Prototyping", "Design Systems"], "Mid"),
        ("Cybersecurity Analyst", ["Network Security", "Linux", "SIEM", "Firewalls", "Wireshark"], "Mid")
    ]
    
    candidate_names = [
        "Liam Johnson", "Noah Smith", "Oliver Jones", "Elijah Miller", "James Davis", 
        "William Rodriguez", "Benjamin Martinez", "Lucas Hernandez", "Henry Lopez", "Alexander Gonzalez",
        "Emma Wilson", "Olivia Anderson", "Ava Thomas", "Isabella Taylor", "Sophia Moore", 
        "Charlotte Jackson", "Amelia Martin", "Mia Lee", "Harper Perez", "Evelyn Thompson",
        "Mason White", "Logan Harris", "Ethan Sanchez", "Jacob Clark", "Michael Ramirez", 
        "Daniel Lewis", "Henry Robinson", "Jackson Walker", "Sebastian Young", "Aiden Allen",
        "Matthew King", "Samuel Wright", "David Scott", "Joseph Torres", "Carter Nguyen", 
        "Owen Hill", "Wyatt Flores", "John Green", "Jack Adams", "Luke Nelson"
    ]
    
    candidates = []
    for c_id in range(1, num_candidates + 1):
        name = candidate_names[(c_id - 1) % len(candidate_names)] + f" {c_id // len(candidate_names) + 1}"
        
        # Pick template
        title_base, core_skills, exp = random.choice(profiles_templates)
        
        # Experience level tweak
        level = random.choice(["Junior", "Mid", "Senior", "Lead"])
        title = f"{level} {title_base}" if level != "Mid" else title_base
        
        # Mix in some other random skills
        skills = list(core_skills)
        if random.random() > 0.4:
            skills.append(random.choice(all_job_skills))
        if random.random() > 0.6:
            skills.append(random.choice(all_job_skills))
        skills = list(set(skills))
        
        resume = (
            f"Experienced {title} with a demonstrated history of working in the technology sector. "
            f"Skilled in {', '.join(skills[:-1])} and {skills[-1]}. "
            f"Passionate about software architecture, clean code, and working in agile teams to build scalable products."
        )
        
        candidates.append({
            "candidate_id": f"C{c_id:03d}",
            "name": name,
            "target_title": title,
            "experience_level": level,
            "skills": ", ".join(skills),
            "resume_text": resume
        })
        
    df = pd.DataFrame(candidates)
    df.to_csv(CANDIDATES_CSV, index=False)
    print(f"Generated {len(df)} candidate profiles and saved to {CANDIDATES_CSV}")
    return df

def simulate_interactions(candidates_df, num_interactions=1200):
    print(f"Simulating {num_interactions} user interactions...")
    jobs_df = pd.read_csv(PROCESSED_JOBS_CSV)
    
    interactions = []
    
    # Help guide simulation: high overlap => higher rating
    # We loop to create a diverse rating matrix
    for _ in range(num_interactions):
        cand = candidates_df.sample(1).iloc[0]
        job = jobs_df.sample(1).iloc[0]
        
        c_skills = set([s.strip().lower() for s in cand['skills'].split(',')])
        j_skills = set([s.strip().lower() for s in job['skills'].split(',')])
        
        # Overlap fraction
        overlap = len(c_skills.intersection(j_skills))
        max_possible = len(j_skills)
        overlap_ratio = overlap / max_possible if max_possible > 0 else 0
        
        # Title similarity bonus
        title_overlap = 0
        cand_words = set(cand['target_title'].lower().split())
        job_words = set(job['title'].lower().split())
        if len(cand_words.intersection(job_words)) > 0:
            title_overlap = 1
            
        # Experience level match
        exp_match = 1 if cand['experience_level'] == job['experience_level'] else 0
        
        # Calculate matching probability score (between 0 and 1)
        base_match = (overlap_ratio * 0.5) + (title_overlap * 0.3) + (exp_match * 0.2)
        
        # Choose action
        # Actions: skip (rating 1), view (rating 2), bookmark (rating 3), apply (rating 5)
        # Higher match score leads to higher probability of bookmark/apply
        rand = random.random()
        
        if base_match > 0.6:
            # High match: Mostly 4 (bookmark) or 5 (apply)
            rating = 5 if rand < 0.6 else (4 if rand < 0.9 else 3)
        elif base_match > 0.3:
            # Medium match: Mostly 3 (view-interested) or 4 (bookmark) or 2 (view-uninterested)
            rating = 4 if rand < 0.3 else (3 if rand < 0.7 else 2)
        else:
            # Low match: Mostly 1 (skip) or 2 (view)
            rating = 2 if rand < 0.4 else 1
            
        action_map = {
            1: "skip",
            2: "view",
            3: "view",
            4: "bookmark",
            5: "apply"
        }
        
        interactions.append({
            "candidate_id": cand['candidate_id'],
            "job_id": job['job_id'],
            "interaction_type": action_map[rating],
            "rating": float(rating)
        })
        
    df = pd.DataFrame(interactions)
    # Deduplicate candidate_id + job_id to have unique ratings, keep max rating
    df = df.sort_values('rating', ascending=False).drop_duplicates(subset=['candidate_id', 'job_id']).sort_index()
    
    df.to_csv(INTERACTIONS_CSV, index=False)
    print(f"Generated {len(df)} unique candidate-job interactions. Saved to {INTERACTIONS_CSV}")
    return df

def generate_all_simulation_data():
    candidates_df = generate_candidates()
    simulate_interactions(candidates_df)

if __name__ == "__main__":
    generate_all_simulation_data()
