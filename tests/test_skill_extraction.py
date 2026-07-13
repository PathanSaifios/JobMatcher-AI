import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.skill_extraction import extract_skills

def test_extract_python():
    skills = extract_skills("We need a strong Python developer.")
    assert "Python" in skills

def test_extract_react():
    skills = extract_skills("Building a React.js application with TypeScript.")
    assert "React" in skills
    assert "TypeScript" in skills

def test_extract_aws_docker():
    skills = extract_skills("Deploy on AWS using Docker and Kubernetes.")
    assert "AWS" in skills
    assert "Docker" in skills
    assert "Kubernetes" in skills

def test_extract_empty():
    skills = extract_skills("")
    assert skills == []

def test_extract_none():
    skills = extract_skills(None)
    assert skills == []

def test_returns_sorted_list():
    skills = extract_skills("Python and React developer needed.")
    assert skills == sorted(skills)

def test_no_false_positives():
    skills = extract_skills("The weather is sunny today.")
    assert len(skills) == 0
