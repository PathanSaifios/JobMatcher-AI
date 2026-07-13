import re

# Comprehensive list of standard industry skills mapped to their matching regex/patterns
SKILL_PATTERNS = {
    # Programming Languages
    "Python": r"\bpython\b",
    "JavaScript": r"\bjavascript\b|\bjs\b",
    "TypeScript": r"\btypescript\b|\bts\b",
    "Java": r"\bjava\b(?! script|script)",
    "C++": r"\bc\+\+\b",
    "C#": r"\bc\#\b|\bcsharp\b",
    "Go": r"\bgo\b|\bgolang\b",
    "Ruby": r"\bruby\b",
    "PHP": r"\bphp\b",
    "Swift": r"\bswift\b",
    "Kotlin": r"\bkotlin\b",
    "Rust": r"\brust\b",
    "Scala": r"\bscala\b",
    "HTML": r"\bhtml5?\b",
    "CSS": r"\bcss3?\b",
    "Sass": r"\bsass\b|\bscss\b",
    "SQL": r"\bsql\b",

    # Frameworks & Libraries
    "React": r"\breact\b|\breact\.js\b|\breactjs\b",
    "Angular": r"\bangular\b|\bangular\.js\b",
    "Vue": r"\bvue\b|\bvue\.js\b|\bvuejs\b",
    "Node.js": r"\bnode\b|\bnode\.js\b",
    "Express": r"\bexpress\b|\bexpress\.js\b",
    "Django": r"\bdjango\b",
    "Flask": r"\bflask\b",
    "FastAPI": r"\bfastapi\b",
    "Spring Boot": r"\bspring boot\b|\bspring\b",
    "Hibernate": r"\bhibernate\b",
    "Next.js": r"\bnext\.js\b|\bnextjs\b",
    "Redux": r"\bredux\b",
    "GraphQL": r"\bgraphql\b",
    "Tailwind CSS": r"\btailwind\b",
    "Bootstrap": r"\bbootstrap\b",
    "PyTorch": r"\bpytorch\b",
    "TensorFlow": r"\btensorflow\b",
    "Keras": r"\bkeras\b",
    "Scikit-Learn": r"\bscikit-learn\b|\bsklearn\b",
    "Pandas": r"\bpandas\b",
    "NumPy": r"\bnumpy\b",
    "SpaCy": r"\bspacy\b",
    "NLTK": r"\bnltk\b",
    "Hugging Face": r"\bhugging face\b|\bhuggingface\b",
    "Transformers": r"\btransformers\b",
    "BERT": r"\bbert\b",

    # Cloud & DevOps
    "AWS": r"\baws\b|\bamazon web services\b",
    "Google Cloud": r"\bgcp\b|\bgoogle cloud\b",
    "Azure": r"\bazure\b",
    "Docker": r"\bdocker\b",
    "Kubernetes": r"\bkubernetes\b|\bk8s\b",
    "Terraform": r"\bterraform\b",
    "Ansible": r"\bansible\b",
    "Jenkins": r"\bjenkins\b",
    "CI/CD": r"\bci/cd\b|\bcontinuous integration\b",
    "Git": r"\bgit\b",
    "GitHub": r"\bgithub\b",
    "GitLab": r"\bgitlab\b",
    "Linux": r"\blinux\b",

    # Databases
    "PostgreSQL": r"\bpostgresql\b|\bpostgres\b",
    "MySQL": r"\bmysql\b",
    "MongoDB": r"\bmongodb\b|\bmongo\b",
    "NoSQL": r"\bnosql\b",
    "Redis": r"\bredis\b",
    "SQLite": r"\bsqlite\b",
    "Elasticsearch": r"\belasticsearch\b",

    # Concepts & Methodologies
    "Machine Learning": r"\bmachine learning\b|\bml\b",
    "Deep Learning": r"\bdeep learning\b|\bdl\b",
    "NLP": r"\bnlp\b|\bnatural language processing\b",
    "Computer Vision": r"\bcomputer vision\b|\bcv\b",
    "Data Analysis": r"\bdata analysis\b",
    "A/B Testing": r"\ba/b testing\b",
    "Data Visualization": r"\bdata visualization\b",
    "Tableau": r"\btableau\b",
    "PowerBI": r"\bpowerbi\b|\bpower bi\b",
    "Microservices": r"\bmicroservices\b|\bmicroservice\b",
    "REST API": r"\brest api\b|\brestful api\b",
    "Agile": r"\bagile\b",
    "Scrum": r"\bscrum\b",
    "Jira": r"\bjira\b",
    "Product Roadmap": r"\bproduct roadmap\b|\broadmap\b",
    "Figma": r"\bfigma\b",
    "UX/UI": r"\bux/ui\b|\bux\b|\bui\b|\buser experience\b|\buser interface\b",
    "Wireframing": r"\bwireframing\b|\bwireframe\b",
    "Prototyping": r"\bprototyping\b|\bprototype\b",
    "Design Systems": r"\bdesign system\b|\bdesign systems\b",
    
    # Cyber Security
    "Network Security": r"\bnetwork security\b",
    "Penetration Testing": r"\bpenetration testing\b|\bpentesting\b",
    "SIEM": r"\bsiem\b",
    "Firewalls": r"\bfirewalls?\b",
    "Wireshark": r"\bwireshark\b"
}

def extract_skills(text: str) -> list[str]:
    """Analyzes text and returns a list of matched skills."""
    if not isinstance(text, str):
        return []
    
    text_lower = text.lower()
    matched_skills = []
    
    for skill_name, pattern in SKILL_PATTERNS.items():
        if re.search(pattern, text_lower):
            matched_skills.append(skill_name)
            
    return sorted(matched_skills)

if __name__ == "__main__":
    test_text = "Looking for a Python Developer experienced in Django, React, AWS, Docker and SQL. Machine Learning experience is a plus."
    print("Extracted Skills:", extract_skills(test_text))
