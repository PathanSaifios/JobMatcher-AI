from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.data.skill_extraction import extract_skills


class SkillExtractionRequest(BaseModel):
    text: str


class DemoRecommendationRequest(BaseModel):
    target_title: str = "Frontend Engineer"
    location: str = "Bengaluru, India or Remote"
    skills: list[str] = []
    resume_text: str = ""


app = FastAPI(
    title="JobMatcher AI",
    description=(
        "Vercel-hosted API preview for JobMatcher AI. The full Streamlit dashboard "
        "runs locally, on Streamlit Cloud, or on Docker-capable hosting."
    ),
    version="1.0.0",
)


@app.get("/", response_class=HTMLResponse)
def landing_page() -> str:
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>JobMatcher AI</title>
  <style>
    :root {
      color-scheme: light dark;
      --ink: #0a0a10;
      --muted: #555b67;
      --panel: rgba(255, 250, 244, 0.88);
      --edge: rgba(12, 12, 18, 0.28);
      --accent: #8b5cf6;
      --mint: #22c991;
    }
    * { box-sizing: border-box; }
    body {
      min-height: 100vh;
      margin: 0;
      display: grid;
      place-items: center;
      padding: 28px;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: linear-gradient(120deg, #f6efe7 0%, #fffaf4 56%, #f1e7ff 100%);
    }
    main {
      width: min(980px, 100%);
      border: 1.5px solid var(--edge);
      border-radius: 24px;
      padding: clamp(26px, 5vw, 58px);
      background: linear-gradient(145deg, rgba(255,255,255,0.96), var(--panel));
      box-shadow: 0 24px 70px rgba(45,35,24,0.16), inset 0 1px 0 rgba(255,255,255,0.54);
      backdrop-filter: blur(22px) saturate(150%);
    }
    .mark {
      width: 62px;
      height: 62px;
      display: grid;
      place-items: center;
      border-radius: 18px;
      color: white;
      font-weight: 900;
      background: linear-gradient(135deg, #050505, var(--accent), var(--mint));
    }
    h1 {
      margin: 24px 0 10px;
      font-size: clamp(2.4rem, 6vw, 5.1rem);
      line-height: 0.95;
      letter-spacing: 0;
    }
    p {
      max-width: 760px;
      color: var(--muted);
      line-height: 1.7;
      font-size: 1.04rem;
    }
    nav {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 30px;
    }
    a {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 44px;
      padding: 0 16px;
      border-radius: 14px;
      border: 1px solid var(--edge);
      color: #0a0a10;
      text-decoration: none;
      font-weight: 800;
      background: rgba(255,255,255,0.78);
    }
    a.primary {
      color: white;
      background: linear-gradient(135deg, #050505, var(--accent));
    }
  </style>
</head>
<body>
  <main>
    <div class="mark">JM</div>
    <h1>JobMatcher AI is live.</h1>
    <p>
      This Vercel deployment exposes the API preview for health checks, skill extraction,
      and demo recommendations. The complete Streamlit dashboard remains available through
      local/Docker/Streamlit hosting because Streamlit is a long-running interactive app.
    </p>
    <nav>
      <a class="primary" href="/docs">Open API docs</a>
      <a href="/health">Health check</a>
      <a href="/api/v1/demo-recommendations">Demo recommendations</a>
    </nav>
  </main>
</body>
</html>
"""


@app.get("/health")
def health_check() -> dict:
    return {"status": "healthy", "service": "JobMatcher AI API preview"}


@app.post("/api/v1/extract-skills")
def get_extracted_skills(request: SkillExtractionRequest) -> dict:
    return {"extracted_skills": extract_skills(request.text)}


@app.get("/api/v1/demo-recommendations")
def demo_recommendations() -> list[dict]:
    return [
        {
            "rank": 1,
            "title": "Junior Frontend Engineer",
            "company": "DataPulse",
            "location": "Bengaluru, India or Remote",
            "fit": 84,
            "matched_skills": ["React", "JavaScript", "CSS", "HTML", "Git"],
        },
        {
            "rank": 2,
            "title": "Python Backend Developer",
            "company": "CloudScale",
            "location": "Hyderabad, India or Remote",
            "fit": 79,
            "matched_skills": ["Python", "FastAPI", "SQL", "Docker", "Git"],
        },
    ]


@app.post("/api/v1/demo-recommendations")
def custom_demo_recommendations(request: DemoRecommendationRequest) -> list[dict]:
    skills = request.skills or extract_skills(request.resume_text)
    return [
        {
            "rank": 1,
            "title": request.target_title,
            "company": "JobMatcher Demo",
            "location": request.location,
            "fit": min(95, 62 + len(skills) * 4),
            "matched_skills": skills[:8],
            "note": "Demo Vercel response. Run the Streamlit dashboard for the full hybrid recommender.",
        }
    ]
