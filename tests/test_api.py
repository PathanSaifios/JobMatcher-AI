import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient

def get_client():
    from api.main import app
    return TestClient(app)

def test_root_health():
    client = get_client()
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert "message" in data

def test_health_endpoint():
    client = get_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"

def test_extract_skills_endpoint():
    client = get_client()
    payload = {"text": "Looking for a Python developer with experience in Django and AWS."}
    resp = client.post("/api/v1/extract-skills", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "extracted_skills" in data
    assert isinstance(data["extracted_skills"], list)
    assert "Python" in data["extracted_skills"]

def test_extract_skills_empty_text():
    client = get_client()
    resp = client.post("/api/v1/extract-skills", json={"text": ""})
    assert resp.status_code == 200
    assert resp.json()["extracted_skills"] == []
