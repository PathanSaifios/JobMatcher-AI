from html import escape
from urllib.parse import quote_plus

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.data.skill_extraction import extract_skills


PLATFORMS = ["LinkedIn", "Naukri", "Indeed", "Glassdoor", "Foundit", "Google jobs"]

JOB_CATALOG = [
    {
        "title": "Junior Frontend Engineer",
        "company": "DataPulse",
        "location": "Hyderabad, India",
        "experience_level": "Junior",
        "salary": "$70,000 - $100,000",
        "skills": ["React", "JavaScript", "CSS", "HTML", "Git", "UX/UI"],
        "description": "Build responsive web interfaces with React, JavaScript, CSS, HTML, and design-system patterns.",
    },
    {
        "title": "Frontend Developer",
        "company": "CloudScale",
        "location": "Bengaluru, India",
        "experience_level": "Junior",
        "salary": "$65,000 - $95,000",
        "skills": ["React", "TypeScript", "Tailwind CSS", "Redux", "Git", "Figma"],
        "description": "Create polished product dashboards and reusable frontend components.",
    },
    {
        "title": "Python Backend Developer",
        "company": "NexaAPI",
        "location": "Remote",
        "experience_level": "Mid",
        "salary": "$80,000 - $120,000",
        "skills": ["Python", "FastAPI", "SQL", "PostgreSQL", "Docker", "Git"],
        "description": "Design REST APIs, database models, and production services using Python and FastAPI.",
    },
    {
        "title": "Machine Learning Engineer",
        "company": "InsightWorks",
        "location": "Bengaluru, India",
        "experience_level": "Senior",
        "salary": "$110,000 - $155,000",
        "skills": ["Python", "Machine Learning", "PyTorch", "TensorFlow", "Scikit-Learn", "AWS"],
        "description": "Build ML pipelines, ranking models, NLP prototypes, and cloud model services.",
    },
    {
        "title": "DevOps Engineer",
        "company": "InfraNest",
        "location": "Pune, India",
        "experience_level": "Senior",
        "salary": "$95,000 - $135,000",
        "skills": ["AWS", "Docker", "Kubernetes", "Terraform", "CI/CD", "Linux"],
        "description": "Automate cloud infrastructure, deployments, observability, and CI/CD pipelines.",
    },
    {
        "title": "Data Analyst",
        "company": "MetricLoop",
        "location": "Mumbai, India",
        "experience_level": "Junior",
        "salary": "$55,000 - $85,000",
        "skills": ["Python", "SQL", "Pandas", "NumPy", "PowerBI", "Data Visualization"],
        "description": "Analyze business datasets, build reports, and explain product performance trends.",
    },
    {
        "title": "Full Stack Developer",
        "company": "OrbitApps",
        "location": "Remote",
        "experience_level": "Mid",
        "salary": "$85,000 - $125,000",
        "skills": ["React", "Node.js", "FastAPI", "PostgreSQL", "Docker", "Git"],
        "description": "Own full-stack product features from frontend interactions to backend APIs.",
    },
    {
        "title": "UI Engineer",
        "company": "GlassForge",
        "location": "Delhi, India",
        "experience_level": "Junior",
        "salary": "$60,000 - $92,000",
        "skills": ["HTML", "CSS", "JavaScript", "Tailwind CSS", "Figma", "UX/UI"],
        "description": "Convert high-quality designs into responsive, accessible, production-ready interfaces.",
    },
]


class SkillExtractionRequest(BaseModel):
    text: str


class RecommendationRequest(BaseModel):
    full_name: str = "Job seeker"
    email: str = ""
    target_title: str = "Frontend Engineer"
    location: str = "Bengaluru, India or Remote"
    experience_level: str = "Junior"
    skills: list[str] = Field(default_factory=list)
    resume_text: str = ""
    platforms: list[str] = Field(default_factory=lambda: ["LinkedIn", "Naukri", "Indeed"])
    top_k: int = 6


app = FastAPI(
    title="JobMatcher AI",
    description="Vercel-compatible JobMatcher AI web app and recommendation API.",
    version="1.1.0",
)


def normalize_words(value: str) -> set[str]:
    ignored = {"and", "or", "the", "for", "in", "of", "to", "with", "near", "remote"}
    words = "".join(char.lower() if char.isalnum() else " " for char in str(value)).split()
    return {word for word in words if word not in ignored and len(word) > 1}


def platform_search_url(platform: str, title: str, location: str, skills: list[str]) -> str:
    query = " ".join([title, *skills[:4], location]).strip()
    encoded = quote_plus(query)
    platform_key = platform.lower()
    if platform_key == "linkedin":
        return f"https://www.linkedin.com/jobs/search/?keywords={encoded}&location={quote_plus(location)}"
    if platform_key == "naukri":
        return f"https://www.naukri.com/{quote_plus(title).replace('+', '-')}-jobs?k={encoded}"
    if platform_key == "indeed":
        return f"https://www.indeed.com/jobs?q={encoded}&l={quote_plus(location)}"
    if platform_key == "glassdoor":
        return f"https://www.glassdoor.com/Job/jobs.htm?sc.keyword={encoded}"
    if platform_key == "foundit":
        return f"https://www.foundit.in/srp/results?query={encoded}&locations={quote_plus(location)}"
    return f"https://www.google.com/search?q={encoded}+jobs"


def score_job(request: RecommendationRequest, job: dict) -> dict:
    resume_skills = extract_skills(request.resume_text)
    candidate_skills = {skill.lower() for skill in [*request.skills, *resume_skills] if skill.strip()}
    job_skills = {skill.lower() for skill in job["skills"]}

    skill_hits = sorted(candidate_skills.intersection(job_skills))
    skill_score = len(skill_hits) / max(1, len(job_skills))

    target_words = normalize_words(request.target_title)
    job_words = normalize_words(job["title"])
    title_score = len(target_words.intersection(job_words)) / max(1, min(len(target_words), len(job_words)))

    preferred_place = normalize_words(request.location)
    job_place = normalize_words(job["location"])
    remote_ok = "remote" in str(request.location).lower() or "remote" in str(job["location"]).lower()
    place_score = 0.45
    if preferred_place.intersection(job_place):
        place_score = 1.0
    elif remote_ok:
        place_score = 0.78

    level_score = 1.0 if request.experience_level.lower() == job["experience_level"].lower() else 0.56
    raw_score = (0.48 * skill_score) + (0.24 * title_score) + (0.18 * place_score) + (0.10 * level_score)
    fit = int(max(28, min(96, round(raw_score * 100))))

    platforms = [platform for platform in request.platforms if platform in PLATFORMS] or ["LinkedIn"]
    apply_links = [
        {
            "platform": platform,
            "url": platform_search_url(platform, job["title"], request.location or job["location"], job["skills"]),
        }
        for platform in platforms
    ]

    return {
        "title": job["title"],
        "company": job["company"],
        "location": job["location"],
        "experience_level": job["experience_level"],
        "salary": job["salary"],
        "description": job["description"],
        "fit": fit,
        "matched_skills": [skill.title() for skill in skill_hits],
        "missing_skills": [skill for skill in job["skills"] if skill.lower() not in candidate_skills][:5],
        "apply_links": apply_links,
        "reason": (
            f"Recommended for {request.full_name or 'this candidate'} because the role matches "
            f"{len(skill_hits)} skill signals, the target title alignment, and the selected place preference."
        ),
    }


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
      --ink: #0b0c12;
      --muted: #566070;
      --panel: rgba(255, 250, 244, 0.9);
      --panel-strong: rgba(255, 255, 255, 0.96);
      --edge: rgba(12, 12, 18, 0.28);
      --accent: #8b5cf6;
      --mint: #22c991;
      --danger: #d9485f;
      --amber: #d49a26;
    }
    * { box-sizing: border-box; }
    body {
      min-height: 100vh;
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        linear-gradient(rgba(255,255,255,0.2) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.18) 1px, transparent 1px),
        linear-gradient(120deg, #f6efe7 0%, #fffaf4 58%, #f1e7ff 100%);
      background-size: 48px 48px, 48px 48px, auto;
    }
    body.dark {
      --ink: #f8f3ea;
      --muted: #c8beb2;
      --panel: rgba(22,21,23,0.86);
      --panel-strong: rgba(31,29,33,0.94);
      --edge: rgba(255,255,255,0.23);
      background:
        linear-gradient(rgba(255,255,255,0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.04) 1px, transparent 1px),
        linear-gradient(120deg, #070707 0%, #121013 55%, #1d1728 100%);
      background-size: 48px 48px, 48px 48px, auto;
    }
    .shell {
      display: grid;
      grid-template-columns: 300px minmax(0, 1fr);
      min-height: 100vh;
    }
    aside {
      position: sticky;
      top: 0;
      height: 100vh;
      overflow: auto;
      padding: 24px 18px;
      background: #030303;
      color: white;
      border-right: 1px solid rgba(255,255,255,0.12);
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 26px;
    }
    .mark {
      width: 52px;
      height: 52px;
      display: grid;
      place-items: center;
      border-radius: 16px;
      color: white;
      font-weight: 900;
      background: linear-gradient(135deg, #050505, var(--accent), var(--mint));
    }
    .brand strong { display: block; }
    .brand small { color: rgba(255,255,255,0.68); }
    .side-card {
      border: 1px solid rgba(255,255,255,0.18);
      border-radius: 18px;
      padding: 14px;
      background: rgba(255,255,255,0.08);
      margin-bottom: 14px;
    }
    .side-card span {
      display: block;
      color: rgba(255,255,255,0.66);
      font-size: 0.76rem;
      font-weight: 800;
      text-transform: uppercase;
    }
    .side-card strong, .side-card small {
      display: block;
      overflow-wrap: anywhere;
    }
    .mini-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-top: 12px;
    }
    .mini-grid div {
      border: 1px solid rgba(255,255,255,0.14);
      border-radius: 14px;
      padding: 10px;
      background: rgba(255,255,255,0.08);
    }
    main {
      padding: clamp(18px, 4vw, 46px);
    }
    .hero, .card, .result {
      border: 1.5px solid var(--edge);
      border-radius: 22px;
      background: linear-gradient(145deg, var(--panel-strong), var(--panel));
      box-shadow: 0 24px 70px rgba(45,35,24,0.14), inset 0 1px 0 rgba(255,255,255,0.42);
      backdrop-filter: blur(22px) saturate(150%);
    }
    .hero {
      padding: clamp(24px, 5vw, 50px);
      margin-bottom: 20px;
    }
    h1 {
      margin: 0 0 12px;
      font-size: clamp(2.1rem, 5vw, 4.7rem);
      line-height: 0.96;
      letter-spacing: 0;
    }
    h2, h3 { margin: 0 0 12px; }
    p { color: var(--muted); line-height: 1.7; }
    label {
      display: block;
      margin: 14px 0 7px;
      color: var(--muted);
      font-size: 0.84rem;
      font-weight: 800;
    }
    input, textarea, select {
      width: 100%;
      border: 1.5px solid var(--edge);
      border-radius: 14px;
      padding: 12px 13px;
      color: var(--ink);
      background: var(--panel-strong);
      font: inherit;
    }
    textarea { min-height: 150px; resize: vertical; }
    .grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(320px, 0.58fr);
      gap: 18px;
      align-items: start;
    }
    .card { padding: 20px; }
    .row {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }
    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }
    .chip {
      border: 1px solid var(--edge);
      border-radius: 999px;
      padding: 8px 10px;
      color: var(--ink);
      background: rgba(255,255,255,0.48);
      font-weight: 800;
      font-size: 0.82rem;
      cursor: pointer;
      user-select: none;
    }
    .chip input { display: none; }
    .chip:has(input:checked) {
      color: white;
      border-color: transparent;
      background: linear-gradient(135deg, #050505, var(--accent));
    }
    button, .apply {
      border: 0;
      border-radius: 14px;
      min-height: 46px;
      padding: 0 16px;
      color: white;
      background: linear-gradient(135deg, #050505, var(--accent));
      font-weight: 900;
      cursor: pointer;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      justify-content: center;
    }
    button.secondary {
      color: #0b0c12;
      border: 1px solid rgba(255,255,255,0.26);
      background: rgba(255,255,255,0.9);
    }
    .actions {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 16px;
    }
    .results {
      display: grid;
      gap: 16px;
      margin-top: 20px;
    }
    .result {
      padding: 20px;
    }
    .result-head {
      display: grid;
      grid-template-columns: auto 1fr auto;
      gap: 14px;
      align-items: start;
    }
    .rank, .score {
      min-width: 54px;
      min-height: 54px;
      display: grid;
      place-items: center;
      border-radius: 16px;
      color: white;
      background: linear-gradient(135deg, #050505, var(--accent), var(--mint));
      font-weight: 900;
    }
    .score { min-width: 76px; }
    .meta { color: var(--muted); }
    .badge {
      display: inline-flex;
      border-radius: 999px;
      padding: 7px 10px;
      font-size: 0.8rem;
      font-weight: 800;
      border: 1px solid var(--edge);
      background: rgba(255,255,255,0.38);
    }
    .match { color: #047267; background: rgba(34,201,145,0.16); }
    .gap { color: #a1403f; background: rgba(217,72,95,0.13); }
    .apply-link {
      border: 1px solid var(--edge);
      border-radius: 12px;
      padding: 10px 12px;
      color: var(--ink);
      text-decoration: none;
      font-weight: 850;
      background: rgba(255,255,255,0.50);
    }
    .hidden { display: none !important; }
    .status { color: var(--muted); margin-top: 10px; }
    @media (max-width: 900px) {
      .shell { grid-template-columns: 1fr; }
      aside { position: static; height: auto; }
      .grid, .row, .result-head { grid-template-columns: 1fr; }
      .score { width: 100%; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <aside>
      <div class="brand">
        <div class="mark">JM</div>
        <div>
          <strong>JobMatcher AI</strong>
          <small>Vercel web matcher</small>
        </div>
      </div>
      <div class="side-card" id="accountCard">
        <span>Signed in</span>
        <strong id="sideName">Guest user</strong>
        <small id="sideEmail">guest@jobmatcher.ai</small>
        <div class="mini-grid">
          <div><strong id="sideCandidate">guest</strong><small>Candidate ID</small></div>
          <div><strong id="sideMode">Balanced</strong><small>Mode</small></div>
        </div>
      </div>
      <button class="secondary" id="themeBtn" type="button" style="width:100%">Toggle dark mode</button>
      <button class="secondary" id="logoutBtn" type="button" style="width:100%; margin-top:10px">Log out</button>
    </aside>
    <main>
      <section class="hero">
        <h1 id="welcomeTitle">Welcome to JobMatcher AI.</h1>
        <p>
          Create a quick session with your email, paste or upload resume text, choose a place,
          and get ranked job recommendations with direct apply links for LinkedIn, Naukri, Indeed, and more.
        </p>
      </section>

      <div class="grid">
        <section class="card" id="loginCard">
          <h2>Start matching</h2>
          <div class="row">
            <div>
              <label for="fullName">Full name</label>
              <input id="fullName" placeholder="Pathan Saif Khan" />
            </div>
            <div>
              <label for="email">Email</label>
              <input id="email" type="email" placeholder="you@example.com" />
            </div>
          </div>
          <label for="password">Password</label>
          <input id="password" type="password" placeholder="Create or enter password" />
          <div class="actions">
            <button type="button" id="saveAccount">Continue with email</button>
          </div>
          <p class="status">This Vercel edition stores the session only in your browser.</p>
        </section>

        <section class="card">
          <h2>Recommendation options</h2>
          <label for="topK">Results to show</label>
          <select id="topK">
            <option>3</option>
            <option selected>6</option>
            <option>8</option>
          </select>
          <label>Apply platforms</label>
          <div class="chips" id="platforms"></div>
        </section>
      </div>

      <section class="card" style="margin-top:18px">
        <h2>Candidate profile</h2>
        <div class="row">
          <div>
            <label for="targetTitle">Target job title</label>
            <input id="targetTitle" value="Junior Frontend Engineer" />
          </div>
          <div>
            <label for="location">Preferred place</label>
            <input id="location" value="Bengaluru, India or Remote" />
          </div>
        </div>
        <div class="row">
          <div>
            <label for="level">Experience level</label>
            <select id="level">
              <option selected>Junior</option>
              <option>Mid</option>
              <option>Senior</option>
              <option>Lead</option>
            </select>
          </div>
          <div>
            <label for="skills">Skills, comma separated</label>
            <input id="skills" value="React, JavaScript, CSS, HTML, Git" />
          </div>
        </div>
        <label for="resumeFile">Upload resume text file</label>
        <input id="resumeFile" type="file" accept=".txt,.md,.csv" />
        <label for="resumeText">Resume or bio</label>
        <textarea id="resumeText">Junior frontend developer focused on React, JavaScript, HTML, CSS, Git, UX/UI, and responsive dashboards.</textarea>
        <div class="actions">
          <button type="button" id="runMatcher">Find matching jobs</button>
          <button class="secondary" type="button" id="fillFrontend">Frontend sample</button>
          <button class="secondary" type="button" id="fillBackend">Backend sample</button>
        </div>
        <p class="status" id="statusText">Ready to rank jobs.</p>
      </section>

      <section class="results" id="results"></section>
    </main>
  </div>

  <script>
    const platformOptions = ["LinkedIn", "Naukri", "Indeed", "Glassdoor", "Foundit", "Google jobs"];
    const state = {
      account: JSON.parse(localStorage.getItem("jobmatcherAccount") || "{}"),
      dark: localStorage.getItem("jobmatcherDark") === "true"
    };

    function candidateId(email) {
      return (email || "guest").split("@")[0].toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "") || "guest";
    }

    function updateAccountView() {
      const name = state.account.fullName || "Guest user";
      const email = state.account.email || "guest@jobmatcher.ai";
      document.getElementById("sideName").textContent = name;
      document.getElementById("sideEmail").textContent = email;
      document.getElementById("sideCandidate").textContent = candidateId(email);
      document.getElementById("welcomeTitle").textContent = `Welcome, ${name}.`;
      document.getElementById("fullName").value = state.account.fullName || "";
      document.getElementById("email").value = state.account.email || "";
      document.getElementById("loginCard").classList.toggle("hidden", Boolean(state.account.email));
    }

    function setupPlatforms() {
      const wrap = document.getElementById("platforms");
      wrap.innerHTML = platformOptions.map((name, index) => `
        <label class="chip"><input type="checkbox" value="${name}" ${index < 3 ? "checked" : ""}>${name}</label>
      `).join("");
    }

    function selectedPlatforms() {
      return [...document.querySelectorAll("#platforms input:checked")].map(input => input.value);
    }

    function splitSkills(value) {
      return value.split(",").map(skill => skill.trim()).filter(Boolean);
    }

    function renderResults(items) {
      const results = document.getElementById("results");
      if (!items.length) {
        results.innerHTML = `<div class="result"><h3>No recommendations found</h3><p>Try adding more resume skills or broadening your location.</p></div>`;
        return;
      }
      results.innerHTML = items.map((job, index) => `
        <article class="result">
          <div class="result-head">
            <div class="rank">#${index + 1}</div>
            <div>
              <h3>${job.title}</h3>
              <div class="meta">${job.company} | ${job.location} | ${job.experience_level} level | ${job.salary}</div>
              <div class="chips">
                ${job.matched_skills.map(skill => `<span class="badge match">Match: ${skill}</span>`).join("")}
                ${job.missing_skills.map(skill => `<span class="badge gap">Gap: ${skill}</span>`).join("")}
              </div>
              <p>${job.reason}</p>
              <div class="actions">
                ${job.apply_links.map((link, linkIndex) => `<a class="${linkIndex === 0 ? "apply" : "apply-link"}" target="_blank" rel="noopener" href="${link.url}">${linkIndex === 0 ? "Apply on " : ""}${link.platform}</a>`).join("")}
              </div>
            </div>
            <div class="score">${job.fit}%<br><small>fit</small></div>
          </div>
        </article>
      `).join("");
    }

    async function runMatcher() {
      document.getElementById("statusText").textContent = "Ranking roles...";
      const response = await fetch("/api/v1/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          full_name: state.account.fullName || document.getElementById("fullName").value || "Job seeker",
          email: state.account.email || document.getElementById("email").value,
          target_title: document.getElementById("targetTitle").value,
          location: document.getElementById("location").value,
          experience_level: document.getElementById("level").value,
          skills: splitSkills(document.getElementById("skills").value),
          resume_text: document.getElementById("resumeText").value,
          platforms: selectedPlatforms(),
          top_k: Number(document.getElementById("topK").value)
        })
      });
      const data = await response.json();
      renderResults(data.recommendations || []);
      document.getElementById("statusText").textContent = `Found ${(data.recommendations || []).length} ranked jobs.`;
    }

    document.getElementById("saveAccount").addEventListener("click", () => {
      state.account = {
        fullName: document.getElementById("fullName").value || "Job seeker",
        email: document.getElementById("email").value || "guest@jobmatcher.ai"
      };
      localStorage.setItem("jobmatcherAccount", JSON.stringify(state.account));
      updateAccountView();
    });

    document.getElementById("logoutBtn").addEventListener("click", () => {
      state.account = {};
      localStorage.removeItem("jobmatcherAccount");
      updateAccountView();
    });

    document.getElementById("themeBtn").addEventListener("click", () => {
      state.dark = !state.dark;
      localStorage.setItem("jobmatcherDark", String(state.dark));
      document.body.classList.toggle("dark", state.dark);
    });

    document.getElementById("runMatcher").addEventListener("click", runMatcher);
    document.getElementById("fillFrontend").addEventListener("click", () => {
      document.getElementById("targetTitle").value = "Junior Frontend Engineer";
      document.getElementById("level").value = "Junior";
      document.getElementById("skills").value = "React, JavaScript, CSS, HTML, Tailwind CSS, Git, UX/UI";
      document.getElementById("resumeText").value = "Junior frontend developer with React, JavaScript, HTML, CSS, Tailwind CSS, Git, Figma, UX/UI, and responsive dashboard experience.";
    });
    document.getElementById("fillBackend").addEventListener("click", () => {
      document.getElementById("targetTitle").value = "Python Backend Developer";
      document.getElementById("level").value = "Mid";
      document.getElementById("skills").value = "Python, FastAPI, SQL, PostgreSQL, Docker, Git";
      document.getElementById("resumeText").value = "Backend developer with Python, FastAPI, SQL, PostgreSQL, Docker, REST API, and Git experience.";
    });
    document.getElementById("resumeFile").addEventListener("change", async event => {
      const file = event.target.files[0];
      if (!file) return;
      document.getElementById("resumeText").value = await file.text();
      document.getElementById("statusText").textContent = `Loaded ${file.name}.`;
    });

    setupPlatforms();
    document.body.classList.toggle("dark", state.dark);
    updateAccountView();
    runMatcher();
  </script>
</body>
</html>
"""


@app.get("/health")
def health_check() -> dict:
    return {"status": "healthy", "service": "JobMatcher AI Vercel app"}


@app.get("/api/v1/jobs")
def list_jobs() -> list[dict]:
    return JOB_CATALOG


@app.post("/api/v1/extract-skills")
def get_extracted_skills(request: SkillExtractionRequest) -> dict:
    return {"extracted_skills": extract_skills(request.text)}


@app.post("/api/v1/recommend")
def recommend_jobs(request: RecommendationRequest) -> dict:
    ranked = sorted((score_job(request, job) for job in JOB_CATALOG), key=lambda item: item["fit"], reverse=True)
    return {
        "candidate": {
            "full_name": escape(request.full_name),
            "email": escape(request.email),
        },
        "recommendations": ranked[: max(1, min(request.top_k, 10))],
    }


@app.get("/api/v1/demo-recommendations")
def demo_recommendations() -> list[dict]:
    request = RecommendationRequest()
    return recommend_jobs(request)["recommendations"][:2]
