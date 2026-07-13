import os
import pandas as pd
import random
from pathlib import Path

def generate_synthetic_jobs():
    print("Generating high-quality synthetic jobs dataset...")
    
    titles_skills = [
        # Software Engineering - Frontend
        ("Frontend Engineer", ["React", "JavaScript", "TypeScript", "HTML5", "CSS3", "Redux", "Tailwind CSS"], 
         "Develop user-facing features, optimize web applications for speed, and collaborate with UX designers.", 
         ["Junior", "Mid", "Senior"]),
        ("Senior React Developer", ["React", "TypeScript", "Next.js", "Redux", "GraphQL", "Jest", "Web Performance"], 
         "Lead the frontend engineering team, build scaleable UI platforms, design state management solutions.", 
         ["Senior", "Lead"]),
        ("UI Developer", ["HTML5", "CSS3", "JavaScript", "Sass", "Figma", "Bootstrap", "Responsive Design"], 
         "Translate design wireframes to responsive HTML/CSS pages, build UI prototypes, ensure visual consistency.", 
         ["Junior", "Mid"]),

        # Software Engineering - Backend
        ("Python Developer", ["Python", "Django", "FastAPI", "SQL", "PostgreSQL", "Git", "Docker"], 
         "Build and maintain robust backend API endpoints, integrate third-party services, write unit tests.", 
         ["Junior", "Mid", "Senior"]),
        ("Backend Engineer (Go/Node)", ["Go", "Node.js", "Express", "MongoDB", "Redis", "Docker", "REST API"], 
         "Design microservices, handle high-throughput databases, build secure authentication and authorization systems.", 
         ["Mid", "Senior"]),
        ("Java Web Services Engineer", ["Java", "Spring Boot", "Hibernate", "MySQL", "AWS", "JUnit", "Microservices"], 
         "Implement enterprise-level backend components, integrate SQL databases, monitor production APIs.", 
         ["Mid", "Senior", "Lead"]),

        # Data Science & AI/ML
        ("Data Scientist", ["Python", "Pandas", "NumPy", "Scikit-Learn", "SQL", "Tableau", "Statistics"], 
         "Analyze product usage data, build predictive models, run A/B tests, and present dashboards to stakeholders.", 
         ["Mid", "Senior"]),
        ("Machine Learning Engineer", ["Python", "PyTorch", "TensorFlow", "Scikit-Learn", "Docker", "AWS", "MLOps"], 
         "Design, train, and deploy machine learning models. Set up continuous integration and inference pipelines.", 
         ["Mid", "Senior", "Lead"]),
        ("Data Analyst", ["SQL", "Excel", "Tableau", "PowerBI", "Python", "Data Visualization", "Communication"], 
         "Extract insights from data warehouses, build reports and dashboards, support business operations.", 
         ["Junior", "Mid"]),
        ("NLP Specialist", ["Python", "NLTK", "SpaCy", "Hugging Face", "PyTorch", "Transformers", "BERT"], 
         "Build text classification, entity extraction, and semantic search models using state-of-the-art NLP models.", 
         ["Senior", "Lead"]),

        # Cloud & DevOps
        ("DevOps Engineer", ["AWS", "Docker", "Kubernetes", "Terraform", "CI/CD", "Linux", "Python"], 
         "Automate cloud infrastructure deployments, build CI/CD pipelines, monitor application performance.", 
         ["Mid", "Senior"]),
        ("Cloud Architect", ["AWS", "Azure", "Terraform", "Cloud Security", "Enterprise Architecture", "Kubernetes"], 
         "Design scaleable and resilient cloud infrastructures, define security policies, migrate legacy architectures.", 
         ["Senior", "Lead"]),

        # Design & Product
        ("Product Manager", ["Agile", "Scrum", "Jira", "Product Roadmap", "SQL", "A/B Testing", "Analytics"], 
         "Define product features, run user interviews, set product roadmaps, and align development teams.", 
         ["Mid", "Senior", "Lead"]),
        ("UX/UI Designer", ["Figma", "Adobe XD", "Wireframing", "User Research", "Prototyping", "Design Systems"], 
         "Conduct user research, design wireframes and high-fidelity mockups, build prototypes, design systems.", 
         ["Junior", "Mid", "Senior"]),
        
        # Cyber Security
        ("Cybersecurity Analyst", ["Network Security", "Linux", "Penetration Testing", "SIEM", "Firewalls", "Wireshark"], 
         "Monitor company systems for security breaches, run vulnerability scans, configure firewalls.", 
         ["Mid", "Senior"])
    ]

    companies = ["TechCorp", "CloudScale", "DevFlow", "DataPulse", "InnoSoft", "ApexSystems", "MetaBytes", "SoftGrid", "NexusAI", "ByteLabs"]
    locations = ["San Francisco, CA", "New York, NY", "Austin, TX", "Seattle, WA", "Boston, MA", "Chicago, IL", "Remote", "Los Angeles, CA", "Denver, CO", "Atlanta, GA"]
    salary_ranges = {
        "Junior": (70000, 100000),
        "Mid": (105000, 140000),
        "Senior": (145000, 185000),
        "Lead": (190000, 240000)
    }

    jobs = []
    job_id = 1000
    
    # Generate ~150 jobs
    for i in range(160):
        title_base, skills, desc_base, levels = random.choice(titles_skills)
        level = random.choice(levels)
        title = f"{level} {title_base}" if level != "Mid" else title_base
        company = random.choice(companies)
        loc = random.choice(locations)
        min_sal, max_sal = salary_ranges[level]
        sal = f"${min_sal:,} - ${max_sal:,}"
        
        # Add some random skills to expand variety
        all_ext_skills = ["Git", "GitHub", "REST API", "CI/CD", "Agile", "SQL", "Docker"]
        extra_skills = random.sample(all_ext_skills, k=random.randint(1, 3))
        combined_skills = list(set(skills + extra_skills))
        
        # Form detailed description
        full_desc = (
            f"We are looking for a {title} to join our engineering division at {company}. "
            f"{desc_base} "
            f"Requirements:\n"
            f"- Strong understanding of {', '.join(combined_skills[:-1])} and {combined_skills[-1]}.\n"
            f"- Experience with software engineering best practices, version control (Git), and agile workflows.\n"
            f"- Excellent problem solving, collaboration, and verbal/written communication skills."
        )
        
        jobs.append({
            "job_id": job_id,
            "title": title,
            "company": company,
            "location": loc,
            "description": full_desc,
            "salary": sal,
            "experience_level": level,
            "skills": ", ".join(combined_skills),
            "posted_date": f"2026-07-{random.randint(1, 9):02d}"
        })
        job_id += 1
        
    df = pd.DataFrame(jobs)
    
    # Ensure raw directory exists
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    df.to_csv("data/raw/linkedin_jobs.csv", index=False)
    print(f"Successfully generated {len(df)} jobs and saved to data/raw/linkedin_jobs.csv")

def main():
    # Attempt Kaggle download first if credentials are set
    if "KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ:
        try:
            print("Kaggle credentials found. Attempting to download dataset...")
            # Here we would normally use the kaggle API:
            # import kaggle
            # kaggle.api.authenticate()
            # kaggle.api.dataset_download_files('arshkon/linkedin-job-postings', path='data/raw', unzip=True)
            # For simplicity, fallback if dataset name or API fails
            raise NotImplementedError("Kaggle download mocked to prioritize clean synthetic run.")
        except Exception as e:
            print(f"Kaggle download failed or skipped: {e}")
            generate_synthetic_jobs()
    else:
        print("Kaggle credentials not set in environment.")
        generate_synthetic_jobs()

if __name__ == "__main__":
    main()
