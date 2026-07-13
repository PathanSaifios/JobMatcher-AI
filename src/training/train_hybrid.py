import pandas as pd
from pathlib import Path
from data.download_data import main as download_main
from src.config import RAW_JOBS_CSV, PROCESSED_JOBS_CSV, CANDIDATES_CSV, INTERACTIONS_CSV
from src.data.preprocessing import preprocess_jobs
from src.data.skill_extraction import extract_skills
from src.data.user_simulation import generate_all_simulation_data
from src.models.hybrid import HybridRecommender

def run_pipeline():
    print("==================================================")
    print("   Starting Hybrid Job Recommender ML Pipeline   ")
    print("==================================================")
    
    # 1. Download/Generate Raw Data
    if not Path(RAW_JOBS_CSV).exists():
        download_main()
    else:
        print(f"Raw jobs file found at {RAW_JOBS_CSV}. Skipping download.")
        
    # 2. Preprocess Jobs
    preprocess_jobs()
    
    # 3. Dynamic Skill Extraction (NLP Pipeline)
    print("Extracting skills from job descriptions...")
    jobs_df = pd.read_csv(PROCESSED_JOBS_CSV)
    
    extracted_skills_list = []
    for idx, row in jobs_df.iterrows():
        desc = row['description']
        # Extract skills using NLP regex matcher
        skills = extract_skills(desc)
        # Fallback to existing skills if none matched
        if not skills and pd.notna(row.get('skills')):
            skills = [s.strip() for s in str(row['skills']).split(',')]
        extracted_skills_list.append(", ".join(skills))
        
    jobs_df['skills'] = extracted_skills_list
    jobs_df.to_csv(PROCESSED_JOBS_CSV, index=False)
    print(f"Skills extracted and updated in {PROCESSED_JOBS_CSV}")
    
    # 4. User and Interaction Simulation
    print("Running user and interaction simulation...")
    generate_all_simulation_data()
    
    # 5. Fit & Train Hybrid Models
    print("Training Hybrid Recommendation Model...")
    jobs_df = pd.read_csv(PROCESSED_JOBS_CSV)
    interactions_df = pd.read_csv(INTERACTIONS_CSV)
    
    # We will use Sentence-BERT for embeddings.
    # Note: If sentence-transformers is not installed, it will automatically fallback to TF-IDF.
    hybrid_model = HybridRecommender(alpha=0.5, use_sbert=True)
    hybrid_model.fit(jobs_df, interactions_df)
    
    print("==================================================")
    print("   ML Pipeline Execution Completed Successfully   ")
    print("==================================================")

if __name__ == "__main__":
    run_pipeline()
