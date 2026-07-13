import hmac
import hashlib
import json
import os
import re
import secrets
import sys
import time
from datetime import datetime, timezone
from html import escape
from io import BytesIO
from pathlib import Path
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import streamlit as st


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
ROOT_PATH = Path(ROOT)
LOGO_PATH = ROOT_PATH / "app" / "assets" / "jobmatch_logo.svg"
LOGO_ICON_PATH = ROOT_PATH / "app" / "assets" / "jobmatch_icon.svg"

from src.config import DATA_DIR, PROCESSED_JOBS_CSV
from src.data.skill_extraction import SKILL_PATTERNS, extract_skills
from src.inference.explain import explain_recommendation


st.set_page_config(
    page_title="JobMatcher AI",
    page_icon=str(LOGO_ICON_PATH) if LOGO_ICON_PATH.exists() else ":material/travel_explore:",
    layout="wide",
    initial_sidebar_state="expanded",
)


EXP_LEVELS = ["Junior", "Mid", "Senior", "Lead"]
SKILL_ALL = sorted(SKILL_PATTERNS.keys())
LOGIN_EMAIL = os.getenv("JOBMATCH_EMAIL", "student@jobmatch.ai").strip().lower()
LEGACY_LOGIN_USER = os.getenv("JOBMATCH_USERNAME", "student").strip().lower()
LOGIN_PASSWORD = os.getenv("JOBMATCH_PASSWORD", "jobmatch123")
USER_STORE_PATH = DATA_DIR / "app_users.json"
PLATFORMS = ["LinkedIn", "Naukri", "Indeed", "Glassdoor", "Foundit", "Google jobs"]
DEFAULT_LOCATION = "Bengaluru, India or Remote"
RECOMMENDATION_MODES = {
    "Balanced": 0.62,
    "Skill-first": 0.82,
    "History-first": 0.38,
}

PROFILE_EXAMPLES = {
    "ML engineer": {
        "target_title": "Machine Learning Engineer",
        "experience_level": "Senior",
        "skills": ["Python", "PyTorch", "TensorFlow", "Scikit-Learn", "AWS", "Docker", "Machine Learning"],
        "resume_text": (
            "Senior machine learning engineer with 6 years building NLP and recommender systems. "
            "Experienced with Python, PyTorch, TensorFlow, Scikit-Learn, Docker, Kubernetes, and AWS."
        ),
    },
    "Backend developer": {
        "target_title": "Python Backend Developer",
        "experience_level": "Mid",
        "skills": ["Python", "Django", "FastAPI", "SQL", "PostgreSQL", "Docker", "Git"],
        "resume_text": (
            "Backend developer with 3 years building REST APIs in Django and FastAPI. "
            "Strong PostgreSQL, SQL, Docker, Git, and production debugging experience."
        ),
    },
    "Frontend developer": {
        "target_title": "Frontend Engineer",
        "experience_level": "Junior",
        "skills": ["React", "JavaScript", "TypeScript", "HTML", "CSS", "Tailwind CSS", "Git"],
        "resume_text": (
            "Junior frontend developer focused on React, TypeScript, HTML, CSS, and Tailwind CSS. "
            "Built accessible SPA projects and polished portfolio interfaces."
        ),
    },
    "DevOps engineer": {
        "target_title": "DevOps Engineer",
        "experience_level": "Senior",
        "skills": ["AWS", "Docker", "Kubernetes", "Terraform", "CI/CD", "Linux", "Python"],
        "resume_text": (
            "DevOps engineer with 5 years automating AWS infrastructure, Kubernetes deployments, "
            "Terraform modules, Linux operations, and CI/CD pipelines."
        ),
    },
}


def init_state() -> None:
    st.session_state.setdefault("authenticated", False)
    st.session_state.setdefault("profile_name", "")
    st.session_state.setdefault("profile_email", "")
    st.session_state.setdefault("active_view", "Recommend")
    st.session_state.setdefault("auth_mode", "Sign in")
    st.session_state.setdefault("ui_theme", "Light")
    st.session_state.setdefault("recommendation_mode", "Balanced")
    st.session_state.setdefault("target_title", PROFILE_EXAMPLES["Backend developer"]["target_title"])
    st.session_state.setdefault("experience_level", PROFILE_EXAMPLES["Backend developer"]["experience_level"])
    st.session_state.setdefault("selected_skills", PROFILE_EXAMPLES["Backend developer"]["skills"])
    st.session_state.setdefault("resume_text", PROFILE_EXAMPLES["Backend developer"]["resume_text"])
    st.session_state.setdefault("resume_file_name", "")
    st.session_state.setdefault("location", DEFAULT_LOCATION)
    st.session_state.setdefault("selected_platforms", ["LinkedIn", "Naukri", "Indeed"])
    st.session_state.setdefault("candidate_id", "COLD_START_USER")
    st.session_state.setdefault("recommendation_results", None)
    st.session_state.setdefault("last_candidate_profile", None)
    st.session_state.setdefault("last_profile_example", None)

    if st.session_state.get("recommendation_mode") not in RECOMMENDATION_MODES:
        st.session_state.recommendation_mode = "Balanced"
    if st.session_state.get("ui_theme") not in {"Light", "Dark"}:
        st.session_state.ui_theme = "Light"
    if st.session_state.get("selected_skills") is None:
        st.session_state.selected_skills = []
    if st.session_state.get("selected_platforms") is None:
        st.session_state.selected_platforms = ["LinkedIn", "Naukri", "Indeed"]
    if st.session_state.get("resume_text") is None:
        st.session_state.resume_text = ""
    if st.session_state.get("location") is None:
        st.session_state.location = DEFAULT_LOCATION
    if st.session_state.get("candidate_id") is None:
        st.session_state.candidate_id = "COLD_START_USER"


def apply_theme() -> None:
    ui_theme = st.session_state.get("ui_theme", "Light")
    is_dark = ui_theme == "Dark"
    palette = (
        {
            "ink": "#f8f3ea",
            "muted": "#c8beb2",
            "soft": "rgba(255,255,255,0.08)",
            "line": "rgba(255,255,255,0.22)",
            "line_strong": "rgba(255,255,255,0.34)",
            "glass": "rgba(22,21,23,0.84)",
            "glass_strong": "rgba(33,31,35,0.92)",
            "navy": "#050505",
            "teal": "#38d99f",
            "blue": "#a978ff",
            "rose": "#ff7f9d",
            "amber": "#f3c86a",
            "shadow": "0 24px 70px rgba(0,0,0,0.34)",
            "app_bg": "linear-gradient(120deg, #070707 0%, #121013 50%, #1d1728 100%)",
            "panel": "rgba(22,21,23,0.86)",
            "panel_strong": "rgba(31,29,33,0.94)",
            "edge": "rgba(255,255,255,0.24)",
            "sidebar": "#030303",
            "sidebar_panel": "rgba(255,255,255,0.08)",
            "canvas": "#121013",
        }
        if is_dark
        else {
            "ink": "#0a0a10",
            "muted": "#555b67",
            "soft": "#f4ece5",
            "line": "rgba(12,12,18,0.24)",
            "line_strong": "rgba(12,12,18,0.40)",
            "glass": "rgba(255,249,242,0.88)",
            "glass_strong": "rgba(255,253,249,0.96)",
            "navy": "#050505",
            "teal": "#22c991",
            "blue": "#8b5cf6",
            "rose": "#ef6f86",
            "amber": "#e8bc5d",
            "shadow": "0 22px 58px rgba(45,35,24,0.12)",
            "app_bg": "linear-gradient(120deg, #f6efe7 0%, #fffaf4 56%, #f1e7ff 100%)",
            "panel": "rgba(255,250,244,0.88)",
            "panel_strong": "rgba(255,255,255,0.96)",
            "edge": "rgba(12,12,18,0.30)",
            "sidebar": "#030303",
            "sidebar_panel": "rgba(255,255,255,0.08)",
            "canvas": "#fff7ef",
        }
    )
    theme_overrides = f"""
<style>
:root {{
    --ink: {palette["ink"]};
    --muted: {palette["muted"]};
    --soft: {palette["soft"]};
    --line: {palette["line"]};
    --line-strong: {palette["line_strong"]};
    --glass: {palette["glass"]};
    --glass-strong: {palette["glass_strong"]};
    --navy: {palette["navy"]};
    --teal: {palette["teal"]};
    --blue: {palette["blue"]};
    --rose: {palette["rose"]};
    --amber: {palette["amber"]};
    --shadow: {palette["shadow"]};
    --app-bg: {palette["app_bg"]};
    --panel: {palette["panel"]};
    --panel-strong: {palette["panel_strong"]};
    --edge: {palette["edge"]};
    --sidebar-bg: {palette["sidebar"]};
    --sidebar-panel: {palette["sidebar_panel"]};
    --canvas: {palette["canvas"]};
}}

.stApp {{
    background: var(--app-bg) !important;
    color: var(--ink) !important;
}}
.stApp::before {{
    background-image:
        linear-gradient(rgba(255,255,255,0.16) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.11) 1px, transparent 1px) !important;
    background-size: 48px 48px, 48px 48px !important;
    opacity: {"0.18" if is_dark else "0.48"};
}}
.main .block-container {{
    max-width: 1340px;
    padding-top: 1.15rem;
}}
[data-testid="stSidebar"] {{
    background: var(--sidebar-bg) !important;
    border-right: 2px solid rgba(255,255,255,0.10) !important;
    box-shadow: 18px 0 60px rgba(0,0,0,0.14) !important;
    backdrop-filter: none !important;
}}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
    color: #f8fafc !important;
}}

.app-banner, .welcome-panel, .login-panel, .result-card, .empty-state,
.insight-card, .platform-card, .glass-card, .profile-preview {{
    border: 1.5px solid var(--edge) !important;
    border-radius: 18px !important;
    background:
        linear-gradient(145deg, var(--panel-strong), var(--panel)),
        var(--canvas) !important;
    box-shadow: var(--shadow), inset 0 1px 0 rgba(255,255,255,0.42) !important;
    backdrop-filter: blur(22px) saturate(150%) !important;
}}
[data-testid="stVerticalBlockBorderWrapper"] {{
    border: 1.5px solid var(--edge) !important;
    border-radius: 16px !important;
    background: var(--panel-strong) !important;
    box-shadow: 0 16px 42px rgba(0,0,0,0.08), inset 0 1px 0 rgba(255,255,255,0.30) !important;
    backdrop-filter: blur(20px) saturate(140%) !important;
}}
.app-banner {{
    color: var(--ink) !important;
    border-width: 2px !important;
}}
.app-banner::after, .login-panel::before {{
    display: none !important;
}}
.banner-copy, .welcome-copy, .login-copy, .muted, .job-meta,
.result-reason, .apply-hint, .platform-card span, .auth-card-intro span {{
    color: var(--muted) !important;
}}
.banner-title, .welcome-title, .login-title, .brand-title, .job-title,
.auth-card-intro strong, .platform-card strong, .insight-card strong,
.welcome-stat strong, .login-stat strong {{
    color: var(--ink) !important;
}}
.eyebrow, .welcome-kicker, .login-panel .eyebrow {{
    color: var(--blue) !important;
}}
.brand-mark {{
    border-radius: 14px !important;
    border: 1px solid var(--edge);
    background: linear-gradient(135deg, #050505, var(--blue) 58%, var(--teal)) !important;
}}
.welcome-stat, .login-stat {{
    border: 1px solid var(--edge) !important;
    border-radius: 14px !important;
    background: var(--soft) !important;
}}
.welcome-stat strong, .sidebar-mini-grid b {{
    overflow-wrap: anywhere;
}}
.result-card {{
    margin: 18px 0 16px !important;
}}
.result-card:hover {{
    transform: translateY(-2px);
    border-color: var(--blue) !important;
    box-shadow: 0 24px 62px rgba(139,92,246,0.16), inset 0 1px 0 rgba(255,255,255,0.54) !important;
}}
.rank-badge, .score-chip {{
    border-radius: 13px !important;
    background: linear-gradient(135deg, #050505, var(--blue) 56%, var(--teal)) !important;
}}
.result-company-pill, .apply-chip {{
    color: var(--ink) !important;
    background: var(--soft) !important;
    border: 1px solid var(--edge) !important;
}}
.apply-button {{
    border-radius: 14px !important;
    background: linear-gradient(135deg, #050505, var(--blue)) !important;
    box-shadow: 0 14px 30px rgba(139,92,246,0.22) !important;
}}
.apply-chip {{
    border-radius: 12px !important;
}}
.platform-card a {{
    border-radius: 12px !important;
    background: linear-gradient(135deg, #050505, var(--blue)) !important;
}}
.score-bar-track {{
    background: rgba(140,140,150,0.18) !important;
}}

.sidebar-account, .sidebar-panel {{
    border: 1px solid rgba(255,255,255,0.20) !important;
    border-radius: 18px !important;
    background: var(--sidebar-panel) !important;
    color: #f8fafc !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.14) !important;
}}
.sidebar-account strong, .sidebar-account small,
.sidebar-account span, .sidebar-panel span {{
    color: #f8fafc !important;
}}
.sidebar-account small, .sidebar-mini-grid em {{
    color: rgba(248,250,252,0.72) !important;
}}
.sidebar-mini-grid div {{
    border: 1px solid rgba(255,255,255,0.16);
    border-radius: 14px;
    background: rgba(255,255,255,0.10) !important;
}}
.theme-switch-title {{
    margin: 14px 0 8px;
    color: #f8fafc;
    font-size: 0.78rem;
    font-weight: 850;
    text-transform: uppercase;
}}
.theme-current {{
    margin: 8px 0 14px;
    padding: 10px 12px;
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.18);
    background: rgba(255,255,255,0.08);
    color: rgba(248,250,252,0.78);
    font-size: 0.82rem;
}}
.theme-current strong {{
    color: #ffffff;
}}
[data-testid="stSidebar"] .st-key-theme_light_btn button,
[data-testid="stSidebar"] .st-key-theme_dark_btn button {{
    min-height: 42px;
    border-radius: 14px !important;
    border: 1px solid rgba(255,255,255,0.24) !important;
    box-shadow: none !important;
}}
[data-testid="stSidebar"] .st-key-theme_light_btn button[kind="primary"],
[data-testid="stSidebar"] .st-key-theme_dark_btn button[kind="primary"] {{
    background: linear-gradient(135deg, var(--blue), var(--teal)) !important;
    color: white !important;
}}
[data-testid="stSidebar"] .st-key-theme_light_btn button[kind="secondary"],
[data-testid="stSidebar"] .st-key-theme_dark_btn button[kind="secondary"] {{
    background: rgba(255,255,255,0.92) !important;
    color: #0a0a10 !important;
}}

.stButton > button, .stFormSubmitButton > button {{
    border-radius: 14px !important;
    border: 1.5px solid var(--edge) !important;
    box-shadow: 0 12px 28px rgba(0,0,0,0.10) !important;
}}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {{
    background: linear-gradient(135deg, #050505, var(--blue)) !important;
    color: white !important;
}}
[data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea,
[data-testid="stSelectbox"] div, [data-baseweb="select"] > div,
[data-testid="stFileUploaderDropzone"] {{
    border-radius: 12px !important;
    border-color: var(--edge) !important;
    background: var(--panel-strong) !important;
    color: var(--ink) !important;
}}

@media (max-width: 700px) {{
    .main .block-container {{
        padding-left: 0.85rem;
        padding-right: 0.85rem;
    }}
    .apply-actions a {{
        width: 100%;
    }}
    .sidebar-mini-grid {{
        grid-template-columns: 1fr;
    }}
}}
</style>
"""
    st.markdown(
        """
<style>
:root {
    --ink: #102033;
    --muted: #53657c;
    --soft: #eef6fb;
    --line: rgba(255,255,255,0.54);
    --line-strong: rgba(255,255,255,0.78);
    --glass: rgba(255,255,255,0.54);
    --glass-strong: rgba(255,255,255,0.72);
    --navy: #102033;
    --teal: #0ea899;
    --blue: #2358d7;
    --rose: #ff7f9d;
    --amber: #f8b84e;
    --shadow: 0 24px 70px rgba(39, 54, 84, 0.16);
}

.stApp {
    background:
        linear-gradient(120deg, rgba(237,253,255,0.96), rgba(242,247,255,0.92) 36%, rgba(255,246,249,0.92) 72%, rgba(255,249,239,0.94)),
        linear-gradient(180deg, #fbfdff 0%, #f8fbff 48%, #fff8f3 100%);
    color: var(--ink);
}

.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background-image:
        linear-gradient(115deg, rgba(255,255,255,0.38), rgba(255,255,255,0) 34%),
        linear-gradient(rgba(255,255,255,0.24) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.18) 1px, transparent 1px);
    background-size: auto, 56px 56px, 56px 56px;
    mask-image: linear-gradient(to bottom, rgba(0,0,0,0.8), transparent 84%);
}

#MainMenu, footer { visibility: hidden; }

[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, rgba(16,32,51,0.88), rgba(18,48,68,0.78)),
        rgba(16,32,51,0.78);
    border-right: 1px solid rgba(255,255,255,0.24);
    box-shadow: 14px 0 42px rgba(16,32,51,0.16);
    backdrop-filter: blur(28px) saturate(150%);
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #f8fafc !important;
}
[data-testid="stSidebar"] button,
[data-testid="stSidebar"] button *,
[data-testid="stSidebar"] [role="radiogroup"] label,
[data-testid="stSidebar"] [role="radiogroup"] label *,
[data-testid="stSidebar"] [data-testid="stSegmentedControl"] *,
[data-testid="stSidebar"] [data-baseweb="select"] * {
    color: #102033 !important;
}
[data-testid="stSidebar"] [data-baseweb="slider"] * { color: #f8fafc !important; }
[data-testid="stSidebar"] .stButton > button {
    color: #102033 !important;
}

.main .block-container {
    max-width: 1360px;
    padding-top: 1.4rem;
    padding-bottom: 2.5rem;
    padding-left: 2.1rem;
    padding-right: 2.1rem;
}

.app-banner, .login-panel, .result-card, .empty-state, .insight-card, .platform-card, .glass-card,
.sidebar-account, .sidebar-panel, .profile-preview {
    border: 1px solid var(--line);
    border-radius: 24px;
    background:
        linear-gradient(145deg, rgba(255,255,255,0.84), rgba(255,255,255,0.42)),
        var(--glass);
    box-shadow: var(--shadow), inset 0 1px 0 rgba(255,255,255,0.86);
    backdrop-filter: blur(30px) saturate(175%);
}
[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid rgba(255,255,255,0.68) !important;
    border-radius: 24px !important;
    background:
        linear-gradient(145deg, rgba(255,255,255,0.76), rgba(255,255,255,0.42)),
        rgba(255,255,255,0.48) !important;
    box-shadow: 0 22px 64px rgba(39,54,84,0.12), inset 0 1px 0 rgba(255,255,255,0.88);
    backdrop-filter: blur(28px) saturate(170%);
}

.brand-lockup {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 22px;
}
.brand-mark {
    width: 58px;
    height: 58px;
    border-radius: 18px;
    display: grid;
    place-items: center;
    color: white;
    font-size: 1.6rem;
    font-weight: 900;
    background: linear-gradient(135deg, var(--navy), var(--blue) 48%, var(--teal));
    box-shadow: 0 16px 40px rgba(35,88,215,0.28);
}
.brand-title {
    margin: 0;
    font-size: 1rem;
    font-weight: 850;
    color: var(--ink);
}
.brand-subtitle {
    margin: 2px 0 0;
    color: var(--muted);
    font-size: 0.86rem;
}

.app-banner {
    position: relative;
    overflow: hidden;
    padding: 30px 34px;
    margin-bottom: 20px;
    color: white;
    background:
        linear-gradient(135deg, rgba(16,32,51,0.94), rgba(35,88,215,0.86) 48%, rgba(14,168,153,0.84)),
        rgba(16,32,51,0.88);
}
.app-banner::after {
    content: "";
    position: absolute;
    inset: auto 0 0 0;
    height: 2px;
    background: linear-gradient(90deg, rgba(255,255,255,0.08), rgba(255,255,255,0.8), rgba(255,255,255,0.1));
}
.welcome-panel {
    position: relative;
    overflow: hidden;
    padding: 30px;
    margin-bottom: 22px;
    border-radius: 28px;
    border: 1px solid rgba(255,255,255,0.70);
    background:
        linear-gradient(145deg, rgba(255,255,255,0.78), rgba(255,255,255,0.36)),
        linear-gradient(120deg, rgba(35,88,215,0.14), rgba(14,168,153,0.12), rgba(255,127,157,0.12));
    box-shadow: 0 26px 80px rgba(39,54,84,0.16), inset 0 1px 0 rgba(255,255,255,0.92);
    backdrop-filter: blur(34px) saturate(180%);
}
.welcome-panel::after {
    content: "";
    position: absolute;
    inset: auto 24px 0 24px;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.86), transparent);
}
.welcome-kicker {
    margin: 0 0 8px;
    color: #2358d7;
    font-size: 0.78rem;
    font-weight: 850;
    letter-spacing: 0;
    text-transform: uppercase;
}
.welcome-title {
    margin: 0;
    color: var(--ink);
    font-size: 2.55rem;
    line-height: 1.04;
    font-weight: 900;
    letter-spacing: 0;
}
.welcome-copy {
    max-width: 850px;
    margin: 12px 0 0;
    color: var(--muted);
    line-height: 1.62;
    font-size: 1rem;
}
.welcome-stat-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
    margin-top: 22px;
}
.welcome-stat {
    padding: 14px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.72);
    background: rgba(255,255,255,0.45);
}
.welcome-stat strong {
    display: block;
    color: var(--ink);
    font-size: 1.15rem;
}
.welcome-stat span {
    color: var(--muted);
    font-size: 0.8rem;
}
.eyebrow {
    margin: 0 0 8px 0;
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 0;
    text-transform: uppercase;
    opacity: 0.82;
}
.banner-title {
    margin: 0;
    font-size: 3rem;
    line-height: 1.05;
    font-weight: 850;
    letter-spacing: 0;
}
.banner-copy {
    max-width: 820px;
    margin: 12px 0 0;
    font-size: 1.02rem;
    line-height: 1.65;
    color: rgba(255,255,255,0.88);
}

.login-panel {
    padding: 34px;
    min-height: 500px;
    position: relative;
    overflow: hidden;
    background:
        linear-gradient(145deg, rgba(255,255,255,0.86), rgba(255,255,255,0.42)),
        linear-gradient(125deg, rgba(184,255,243,0.26), rgba(199,213,255,0.24), rgba(255,218,229,0.22));
}
.login-panel::before {
    content: "";
    position: absolute;
    right: -90px;
    top: -80px;
    width: 260px;
    height: 260px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(35,88,215,0.20), rgba(14,168,153,0.08), transparent 66%);
}
.login-panel > * {
    position: relative;
    z-index: 1;
}
.login-title {
    margin: 0;
    font-size: 3rem;
    line-height: 1.06;
    color: var(--ink);
    font-weight: 850;
    letter-spacing: 0;
}
.login-copy, .muted {
    color: var(--muted);
    line-height: 1.65;
}
.login-copy {
    font-size: 1.02rem;
    max-width: 720px;
}
.login-stat-row {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
    margin-top: 26px;
}
.login-stat {
    padding: 16px;
    border-radius: 18px;
    background: rgba(255,255,255,0.54);
    border: 1px solid rgba(255,255,255,0.72);
}
.login-stat strong {
    display: block;
    color: var(--ink);
    font-size: 1.35rem;
}
.login-stat span {
    color: var(--muted);
    font-size: 0.82rem;
}
.auth-card-intro {
    padding: 16px 0 8px;
}
.auth-card-intro strong {
    display: block;
    color: var(--ink);
    font-size: 1.28rem;
    font-weight: 900;
}
.auth-card-intro span {
    display: block;
    margin-top: 5px;
    color: var(--muted);
    line-height: 1.45;
}

.section-label {
    margin: 28px 0 12px;
    color: var(--ink);
    font-size: 1.18rem;
    font-weight: 800;
}

.result-card {
    position: relative;
    padding: 22px;
    margin: 14px 0 12px;
    transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
}
.result-card:hover {
    transform: translateY(-2px);
    border-color: rgba(255,255,255,0.88);
    box-shadow: 0 28px 84px rgba(35,88,215,0.16), inset 0 1px 0 rgba(255,255,255,0.94);
}
.result-card-link {
    display: inline;
    color: inherit !important;
    text-decoration: none !important;
}
.result-card-link:hover {
    text-decoration: none !important;
}
.result-head {
    display: grid;
    grid-template-columns: auto 1fr auto;
    gap: 16px;
    align-items: start;
}
.result-topline {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
    margin: 0 0 6px;
}
.result-company-pill {
    display: inline-flex;
    align-items: center;
    padding: 4px 9px;
    border-radius: 999px;
    color: #1d4ed8;
    background: rgba(35,88,215,0.10);
    border: 1px solid rgba(35,88,215,0.16);
    font-size: 0.74rem;
    font-weight: 800;
}
.rank-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 42px;
    height: 42px;
    border-radius: 15px;
    background: linear-gradient(135deg, var(--navy), var(--blue), var(--teal));
    color: white;
    font-weight: 850;
}
.job-title {
    color: var(--ink);
    font-size: 1.22rem;
    font-weight: 850;
    line-height: 1.25;
    margin: 0 0 5px;
}
.result-title-link {
    color: var(--ink) !important;
    text-decoration: none !important;
}
.result-title-link:hover {
    color: #2358d7 !important;
}
.job-meta {
    color: var(--muted);
    font-size: 0.9rem;
    line-height: 1.5;
}
.score-chip {
    min-width: 92px;
    text-align: center;
    padding: 9px 10px;
    border-radius: 18px;
    color: white;
    background: linear-gradient(135deg, var(--teal), var(--blue));
    font-weight: 850;
}
.score-chip small {
    display: block;
    margin-top: 2px;
    font-size: 0.7rem;
    opacity: 0.84;
    font-weight: 700;
}
.badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: 7px;
    margin-top: 13px;
}
.badge {
    border-radius: 999px;
    padding: 5px 10px;
    font-size: 0.78rem;
    font-weight: 750;
    border: 1px solid transparent;
}
.badge-match {
    color: #047267;
    background: rgba(14,168,153,0.14);
    border-color: rgba(14,168,153,0.28);
}
.badge-gap {
    color: #a1403f;
    background: rgba(232,93,93,0.12);
    border-color: rgba(232,93,93,0.26);
}
.badge-info {
    color: #2438a9;
    background: rgba(68,83,211,0.12);
    border-color: rgba(68,83,211,0.24);
}
.badge-place {
    color: #725012;
    background: rgba(248,184,78,0.18);
    border-color: rgba(248,184,78,0.35);
}
.result-reason {
    max-width: 900px;
    margin: 12px 0 0;
    color: #3f5369;
    font-size: 0.9rem;
    line-height: 1.55;
}
.apply-actions {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 9px;
    margin-top: 16px;
}
.apply-button {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 7px;
    padding: 10px 14px;
    border-radius: 999px;
    color: white !important;
    background: linear-gradient(135deg, #0ea899, #2358d7);
    text-decoration: none !important;
    font-weight: 850;
    box-shadow: 0 14px 28px rgba(35,88,215,0.18);
}
.apply-button:hover {
    filter: brightness(1.04);
}
.apply-chip {
    display: inline-flex;
    padding: 8px 11px;
    border-radius: 999px;
    color: #173154 !important;
    background: rgba(255,255,255,0.56);
    border: 1px solid rgba(255,255,255,0.78);
    text-decoration: none !important;
    font-size: 0.78rem;
    font-weight: 800;
}
.apply-chip:hover {
    background: rgba(255,255,255,0.84);
    border-color: rgba(35,88,215,0.28);
}
.apply-hint {
    color: var(--muted);
    font-size: 0.8rem;
    margin-top: 8px;
}
.platform-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 12px;
    margin: 14px 0 4px;
}
.platform-card {
    padding: 16px;
}
.platform-card strong {
    display: block;
    color: var(--ink);
    font-size: 1.02rem;
    margin-bottom: 4px;
}
.platform-card span {
    display: block;
    color: var(--muted);
    font-size: 0.84rem;
    min-height: 38px;
}
.platform-card a {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    margin-top: 12px;
    padding: 9px 12px;
    border-radius: 999px;
    color: white !important;
    text-decoration: none;
    font-weight: 800;
    background: linear-gradient(135deg, var(--blue), var(--teal));
}

.score-bar-label {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    margin: 10px 0 5px;
    color: var(--ink);
    font-size: 0.86rem;
    font-weight: 750;
}
.score-bar-track {
    height: 9px;
    border-radius: 999px;
    background: rgba(232,238,245,0.86);
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: inherit;
}
.empty-state, .insight-card {
    padding: 18px;
}
.profile-preview {
    padding: 20px;
}
.insight-card strong {
    display: block;
    font-size: 1.6rem;
    color: var(--ink);
}
.insight-card span {
    color: var(--muted);
    font-size: 0.88rem;
}

.sidebar-account, .sidebar-panel {
    padding: 14px;
    margin: 10px 0 14px;
    color: #f8fafc;
    background:
        linear-gradient(145deg, rgba(255,255,255,0.16), rgba(255,255,255,0.06)),
        rgba(255,255,255,0.08);
    border-color: rgba(255,255,255,0.22);
    box-shadow: 0 18px 42px rgba(0,0,0,0.18), inset 0 1px 0 rgba(255,255,255,0.24);
}
.sidebar-account span, .sidebar-panel span {
    display: block;
    color: rgba(248,250,252,0.72);
    font-size: 0.76rem;
    font-weight: 800;
    text-transform: uppercase;
}
.sidebar-account strong {
    display: block;
    margin-top: 4px;
    font-size: 1rem;
    line-height: 1.25;
}
.sidebar-account small {
    display: block;
    margin-top: 4px;
    color: rgba(248,250,252,0.72);
    line-height: 1.35;
    word-break: break-word;
}
.sidebar-mini-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-top: 10px;
}
.sidebar-mini-grid div {
    padding: 8px;
    border-radius: 13px;
    background: rgba(255,255,255,0.10);
}
.sidebar-mini-grid b {
    display: block;
    font-size: 0.95rem;
}
.sidebar-mini-grid em {
    color: rgba(248,250,252,0.70);
    font-size: 0.72rem;
    font-style: normal;
}

.stButton > button, .stFormSubmitButton > button {
    border-radius: 999px !important;
    font-weight: 800 !important;
    border: 1px solid rgba(255,255,255,0.72) !important;
    box-shadow: 0 12px 30px rgba(35,88,215,0.16) !important;
}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--teal), var(--blue)) !important;
    border-color: rgba(255,255,255,0.4) !important;
}
[data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea,
[data-testid="stSelectbox"] div, [data-baseweb="select"] > div {
    border-radius: 14px !important;
}
[data-testid="stFileUploaderDropzone"] {
    border-radius: 20px !important;
    border-color: rgba(35,88,215,0.28) !important;
    background: rgba(255,255,255,0.52) !important;
}
[data-testid="stMetric"] {
    padding: 4px 0;
}
[data-testid="stExpander"] {
    border: 1px solid rgba(255,255,255,0.58) !important;
    border-radius: 18px !important;
    background: rgba(255,255,255,0.36) !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.72);
}

@media (max-width: 900px) {
    .main .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
        padding-top: 0.8rem;
    }
    .app-banner, .login-panel, .result-card, .profile-preview { padding: 18px; }
    .login-stat-row { grid-template-columns: 1fr; }
    .result-head { grid-template-columns: 1fr; }
    .score-chip { width: 100%; }
    .banner-title, .login-title, .welcome-title { font-size: 2.1rem; }
    .welcome-stat-grid { grid-template-columns: 1fr; }
    .platform-grid { grid-template-columns: 1fr; }
}
</style>
        """ + theme_overrides,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_jobs() -> pd.DataFrame | None:
    if Path(PROCESSED_JOBS_CSV).exists():
        return pd.read_csv(PROCESSED_JOBS_CSV)
    return None


def run_pipeline_cached():
    from data.download_data import main as dl_main
    from src.config import INTERACTIONS_CSV, RAW_JOBS_CSV
    from src.data.preprocessing import preprocess_jobs
    from src.data.user_simulation import generate_all_simulation_data
    from src.models.hybrid import HybridRecommender

    if not Path(RAW_JOBS_CSV).exists():
        dl_main()

    preprocess_jobs()
    jobs_df = pd.read_csv(PROCESSED_JOBS_CSV)
    job_skills = []
    for _, row in jobs_df.iterrows():
        skills = extract_skills(str(row.get("description", "")))
        if not skills and pd.notna(row.get("skills")):
            skills = [skill.strip() for skill in str(row.get("skills", "")).split(",") if skill.strip()]
        job_skills.append(", ".join(skills))

    jobs_df["skills"] = job_skills
    jobs_df.to_csv(PROCESSED_JOBS_CSV, index=False)
    generate_all_simulation_data()

    interactions_df = pd.read_csv(INTERACTIONS_CSV)
    jobs_df = pd.read_csv(PROCESSED_JOBS_CSV)
    hybrid = HybridRecommender(alpha=0.5, use_sbert=False)
    hybrid.fit(jobs_df, interactions_df)
    return hybrid, jobs_df


@st.cache_resource(show_spinner=False)
def get_hybrid_model():
    return run_pipeline_cached()


@st.cache_resource(show_spinner=False)
def get_mongo_users_collection():
    uri = os.getenv("JOBMATCH_MONGODB_URI", "").strip()
    if not uri:
        return None

    try:
        from pymongo import MongoClient
    except Exception:
        return None

    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=2500)
        client.admin.command("ping")
        db_name = os.getenv("JOBMATCH_DB_NAME", "jobmatch_ai")
        collection = client[db_name]["users"]
        collection.create_index("email", unique=True)
        return collection
    except Exception:
        return None


def load_local_users() -> dict:
    if not USER_STORE_PATH.exists():
        return {}
    try:
        with USER_STORE_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_local_users(users: dict) -> None:
    USER_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with USER_STORE_PATH.open("w", encoding="utf-8") as handle:
        json.dump(users, handle, indent=2)


def hash_password(password: str, salt_hex: str | None = None) -> tuple[str, str]:
    salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120_000)
    return salt.hex(), digest.hex()


def verify_password(password: str, user_record: dict) -> bool:
    salt = user_record.get("salt", "")
    expected = user_record.get("password_hash", "")
    if not salt or not expected:
        return False
    _, actual = hash_password(password, salt)
    return hmac.compare_digest(actual, expected)


def normalize_email(value: str) -> str:
    return str(value or "").strip().lower()


def is_valid_email(value: str) -> bool:
    return re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", normalize_email(value)) is not None


def display_name_from_email(email: str) -> str:
    local_part = normalize_email(email).split("@")[0]
    cleaned = re.sub(r"[^a-zA-Z0-9]+", " ", local_part).strip()
    return cleaned.title() or "Job seeker"


def email_to_candidate_id(email: str) -> str:
    local_part = normalize_email(email).split("@")[0]
    candidate_id = re.sub(r"[^a-z0-9]+", "_", local_part).strip("_")
    return candidate_id or "COLD_START_USER"


def get_user_record(identifier: str) -> dict | None:
    normalized = normalize_email(identifier)
    collection = get_mongo_users_collection()
    if collection is not None:
        user = collection.find_one({"email": normalized}, {"_id": False})
        if user:
            return user
        user = collection.find_one({"username": normalized}, {"_id": False})
        if user:
            return user

    users = load_local_users()
    direct = users.get(normalized)
    if direct:
        return direct

    for record in users.values():
        if normalize_email(record.get("email", "")) == normalized:
            return record
        if normalize_email(record.get("username", "")) == normalized:
            return record
    return None


def register_user(full_name: str, email: str, password: str) -> tuple[bool, str, dict | None]:
    normalized_email = normalize_email(email)
    display_name = str(full_name or "").strip() or display_name_from_email(normalized_email)

    if not is_valid_email(normalized_email):
        return False, "Enter a valid email address.", None
    if len(display_name) < 2:
        return False, "Enter your name.", None
    if len(password) < 6:
        return False, "Password must be at least 6 characters.", None
    if get_user_record(normalized_email) is not None or normalized_email == LOGIN_EMAIL:
        return False, "An account with this email already exists.", None

    salt, password_hash = hash_password(password)
    record = {
        "username": display_name,
        "display_name": display_name,
        "email": normalized_email,
        "candidate_id": email_to_candidate_id(normalized_email),
        "salt": salt,
        "password_hash": password_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    collection = get_mongo_users_collection()
    if collection is not None:
        try:
            collection.insert_one(record)
            return True, "Account created with MongoDB storage.", record
        except Exception as exc:
            return False, f"MongoDB account save failed: {exc}", None

    users = load_local_users()
    users[normalized_email] = record
    save_local_users(users)
    return True, "Account created locally.", record


def authenticate_user(email: str, password: str) -> dict | None:
    normalized = normalize_email(email)
    demo_identifiers = {LOGIN_EMAIL, LEGACY_LOGIN_USER}
    if normalized in demo_identifiers and hmac.compare_digest(password, LOGIN_PASSWORD):
        return {
            "display_name": "Demo student",
            "username": "Demo student",
            "email": LOGIN_EMAIL,
            "candidate_id": "C001",
        }

    record = get_user_record(normalized)
    return record if record and verify_password(password, record) else None


def sign_in_record(record: dict) -> None:
    profile_email = normalize_email(record.get("email", "")) or LOGIN_EMAIL
    st.session_state.authenticated = True
    st.session_state.profile_name = (
        record.get("display_name")
        or record.get("username")
        or display_name_from_email(profile_email)
    )
    st.session_state.profile_email = profile_email
    st.session_state.candidate_id = record.get("candidate_id") or email_to_candidate_id(profile_email)


def decode_text_bytes(data: bytes) -> str:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def extract_resume_file_text(uploaded_file) -> tuple[str, str]:
    if uploaded_file is None:
        return "", ""

    filename = uploaded_file.name
    suffix = Path(filename).suffix.lower()
    data = uploaded_file.getvalue()

    if suffix in {".txt", ".md"}:
        return decode_text_bytes(data), filename

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader

            reader = PdfReader(BytesIO(data))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            return text.strip(), filename
        except Exception as exc:
            return f"Could not read PDF text: {exc}", filename

    if suffix == ".docx":
        try:
            from docx import Document

            document = Document(BytesIO(data))
            text = "\n".join(paragraph.text for paragraph in document.paragraphs)
            return text.strip(), filename
        except Exception as exc:
            return f"Could not read DOCX text: {exc}", filename

    return decode_text_bytes(data), filename


def infer_title_from_resume(text: str, fallback: str) -> str:
    text_lower = text.lower()
    title_patterns = [
        ("Machine Learning Engineer", ["machine learning engineer", "ml engineer", "deep learning"]),
        ("Data Scientist", ["data scientist", "data science", "predictive model"]),
        ("Python Backend Developer", ["backend developer", "fastapi", "django", "api developer"]),
        ("Frontend Engineer", ["frontend", "react", "typescript", "ui developer"]),
        ("DevOps Engineer", ["devops", "kubernetes", "terraform", "ci/cd"]),
        ("Cybersecurity Analyst", ["security analyst", "penetration", "siem", "firewall"]),
    ]
    for title, markers in title_patterns:
        if any(marker in text_lower for marker in markers):
            return title
    return fallback or "Software Engineer"


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    return slug or "jobs"


def platform_search_url(platform: str, title: str, location: str, skills: list[str]) -> str:
    query = " ".join([title, *skills[:3]]).strip()
    encoded_query = quote_plus(query)
    encoded_location = quote_plus(location or DEFAULT_LOCATION)

    if platform == "LinkedIn":
        return f"https://www.linkedin.com/jobs/search/?keywords={encoded_query}&location={encoded_location}"
    if platform == "Naukri":
        return f"https://www.naukri.com/{slugify(query)}-jobs?k={encoded_query}&l={encoded_location}"
    if platform == "Indeed":
        return f"https://www.indeed.com/jobs?q={encoded_query}&l={encoded_location}"
    if platform == "Glassdoor":
        return f"https://www.glassdoor.co.in/Job/jobs.htm?sc.keyword={encoded_query}"
    if platform == "Foundit":
        return f"https://www.foundit.in/srp/results?query={encoded_query}&locations={encoded_location}"
    return f"https://www.google.com/search?q={encoded_query}+jobs+{encoded_location}"


def build_platform_matches(recs: pd.DataFrame, candidate_profile: dict) -> pd.DataFrame:
    if recs is None or recs.empty:
        return pd.DataFrame()

    location = candidate_profile.get("location") or st.session_state.get("location", DEFAULT_LOCATION)
    platforms = st.session_state.get("selected_platforms") or ["LinkedIn", "Naukri", "Indeed"]
    rows = []

    for _, row in recs.head(5).iterrows():
        job_skills = [skill.strip() for skill in str(row.get("skills", "")).split(",") if skill.strip()]
        title = str(row.get("title") or candidate_profile.get("target_title") or "Software Engineer")
        for platform in platforms:
            rows.append(
                {
                    "platform": platform,
                    "matched_role": title,
                    "company_signal": str(row.get("company", "Multiple companies")),
                    "fit": percent(row.get("hybrid_score", 0)),
                    "location": location,
                    "open_search": platform_search_url(platform, title, location, job_skills),
                }
            )

    return pd.DataFrame(rows)


def selected_platforms() -> list[str]:
    platforms = st.session_state.get("selected_platforms") or ["LinkedIn", "Naukri", "Indeed"]
    return [str(platform) for platform in platforms if str(platform).strip()] or ["LinkedIn"]


def job_apply_links(job_row: dict | pd.Series, candidate_profile: dict) -> list[dict]:
    title = str(job_row.get("title") or candidate_profile.get("target_title") or "Software Engineer")
    location = str(candidate_profile.get("location") or job_row.get("location") or DEFAULT_LOCATION)
    skills = [skill.strip() for skill in str(job_row.get("skills", "")).split(",") if skill.strip()]
    return [
        {
            "platform": platform,
            "url": platform_search_url(platform, title, location, skills),
        }
        for platform in selected_platforms()
    ]


def credential_id(email: str, candidate_id: str) -> str:
    raw = f"{normalize_email(email)}:{candidate_id}".encode("utf-8")
    digest = hashlib.sha1(raw).hexdigest()[:8].upper()
    return f"JM-{digest}"


def location_tokens(value: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", str(value or "").lower())
    ignored = {"and", "or", "the", "near", "in", "jobs", "job", "role", "roles"}
    return {word for word in words if word not in ignored and len(word) > 1}


def location_preference_score(job_location: str, preferred_location: str, include_remote: bool = True) -> float:
    preferred_tokens = location_tokens(preferred_location)
    job_tokens = location_tokens(job_location)
    if not preferred_tokens:
        return 0.75

    wants_remote = "remote" in preferred_tokens
    is_remote = "remote" in job_tokens
    if wants_remote and is_remote:
        return 1.0
    if include_remote and is_remote:
        return 0.86

    overlap = preferred_tokens.intersection(job_tokens)
    if overlap:
        return min(1.0, 0.58 + (0.14 * len(overlap)))

    if include_remote:
        return 0.28
    return 0.12


def apply_location_preference(recs: pd.DataFrame, candidate_profile: dict, include_remote: bool) -> pd.DataFrame:
    preferred_location = candidate_profile.get("location", DEFAULT_LOCATION)
    adjusted = recs.copy()
    adjusted["location_match_score"] = adjusted["location"].apply(
        lambda value: location_preference_score(value, preferred_location, include_remote)
    )
    adjusted["preferred_location"] = preferred_location
    adjusted["hybrid_score"] = np.clip(
        (0.86 * adjusted["hybrid_score"].astype(float)) + (0.14 * adjusted["location_match_score"].astype(float)),
        0,
        1,
    )
    structured_component = (
        adjusted["structured_score"].astype(float)
        if "structured_score" in adjusted
        else pd.Series(0.0, index=adjusted.index)
    )
    adjusted["score_confidence"] = np.clip(
        (0.62 * adjusted["hybrid_score"].astype(float))
        + (0.24 * structured_component)
        + (0.14 * adjusted["location_match_score"].astype(float)),
        0,
        1,
    )
    return adjusted.sort_values(
        by=["hybrid_score", "location_match_score", "structured_score"],
        ascending=False,
    )


def clamp01(value) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def percent(value) -> int:
    return int(round(clamp01(value) * 100))


def safe_escape(value, fallback: str = "") -> str:
    if value is None:
        value = fallback
    return escape(str(value))


def score_bar(label: str, value, color: str) -> None:
    pct = percent(value)
    st.markdown(
        f"""
<div class="score-bar-label"><span>{safe_escape(label)}</span><strong>{pct}%</strong></div>
<div class="score-bar-track"><div class="score-bar-fill" style="width:{pct}%; background:{color};"></div></div>
        """,
        unsafe_allow_html=True,
    )


def badge_html(values, css_class: str, prefix: str, limit: int) -> str:
    items = [value for value in values if str(value).strip()]
    if not items:
        return ""
    return "".join(
        f'<span class="badge {css_class}">{escape(prefix)}{escape(str(value))}</span>'
        for value in items[:limit]
    )


def render_banner(title: str, copy: str, eyebrow: str = "Hybrid recommender") -> None:
    st.markdown(
        f"""
<section class="app-banner">
    <p class="eyebrow">{safe_escape(eyebrow)}</p>
    <h1 class="banner-title">{safe_escape(title)}</h1>
    <p class="banner-copy">{safe_escape(copy)}</p>
</section>
        """,
        unsafe_allow_html=True,
    )


def render_welcome_header() -> None:
    profile_name = st.session_state.get("profile_name") or "Job seeker"
    profile_email = st.session_state.get("profile_email") or LOGIN_EMAIL
    candidate_id = st.session_state.get("candidate_id") or "COLD_START_USER"
    cred_id = credential_id(profile_email, candidate_id)
    st.markdown(
        f"""
<section class="welcome-panel">
    <p class="welcome-kicker">JobMatcher-AI workspace</p>
    <h1 class="welcome-title">Welcome, {safe_escape(profile_name)}.</h1>
    <p class="welcome-copy">
        JobMatcher-AI is ready to rank jobs for your resume. Upload or paste your profile,
        select your preferred place, then open the best matching role directly on your selected job platform.
    </p>
    <div class="welcome-stat-grid">
        <div class="welcome-stat"><strong>{safe_escape(candidate_id)}</strong><span>Candidate ID</span></div>
        <div class="welcome-stat"><strong>{safe_escape(cred_id)}</strong><span>Credential ID</span></div>
        <div class="welcome-stat"><strong>{safe_escape(st.session_state.get("recommendation_mode"), "Balanced")}</strong><span>Recommendation mode</span></div>
    </div>
</section>
        """,
        unsafe_allow_html=True,
    )


def render_login() -> None:
    left, right = st.columns([1.08, 0.92], gap="large", vertical_alignment="center")

    with left:
        st.markdown(
            """
<section class="login-panel">
    <div class="brand-lockup">
        <div class="brand-mark">JM</div>
        <div>
            <p class="brand-title">JobMatcher AI</p>
            <p class="brand-subtitle">Hybrid resume-to-role intelligence</p>
        </div>
    </div>
    <p class="eyebrow" style="color:#2358d7">Job matcher</p>
    <h1 class="login-title">Find roles that actually match your resume.</h1>
    <p class="login-copy">
        Upload your resume, extract skills automatically, choose your preferred place, and let the
        hybrid recommender rank jobs by skills, experience, title alignment, history signals, and location fit.
    </p>
    <div class="login-stat-row">
        <div class="login-stat"><strong>Parse</strong><span>PDF, DOCX, TXT, and Markdown resumes</span></div>
        <div class="login-stat"><strong>Rank</strong><span>Hybrid AI scores with explainable matches</span></div>
        <div class="login-stat"><strong>Apply</strong><span>Location-aware searches on major platforms</span></div>
    </div>
</section>
            """,
            unsafe_allow_html=True,
        )

    with right:
        with st.container(border=True):
            st.markdown(
                """
<div class="auth-card-intro">
    <strong>Start matching in seconds</strong>
    <span>Use your email to keep a clean profile session, upload your resume, and apply through the job portal you choose.</span>
</div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(f"Demo login: {LOGIN_EMAIL} / {LOGIN_PASSWORD}")
            auth_mode = st.segmented_control(
                "Account mode",
                ["Sign in", "Create account"],
                key="auth_mode",
                width="stretch",
            )

            if auth_mode == "Create account":
                with st.form("create_account_form", border=False):
                    full_name = st.text_input("Full name", placeholder="Pathan Saif Khan")
                    email = st.text_input("Email", placeholder="you@example.com")
                    new_password = st.text_input("Create password", type="password")
                    confirm_password = st.text_input("Confirm password", type="password")
                    submitted = st.form_submit_button(
                        "Create account with email",
                        type="primary",
                        icon=":material/person_add:",
                        width="stretch",
                    )

                if submitted:
                    if new_password != confirm_password:
                        st.error("Passwords do not match.", icon=":material/error:")
                    else:
                        ok, message, record = register_user(full_name, email, new_password)
                        if ok:
                            sign_in_record(record or {})
                            st.toast(message, icon=":material/check_circle:")
                            st.rerun()
                        else:
                            st.error(message, icon=":material/error:")
            else:
                with st.form("login_form", border=False):
                    email = st.text_input("Email", placeholder=LOGIN_EMAIL)
                    password = st.text_input("Password", type="password", placeholder=LOGIN_PASSWORD)
                    submitted = st.form_submit_button(
                        "Sign in",
                        type="primary",
                        icon=":material/login:",
                        width="stretch",
                    )

                if submitted:
                    record = authenticate_user(email, password)
                    if record:
                        sign_in_record(record)
                        st.toast("Signed in", icon=":material/check_circle:")
                        st.rerun()
                    else:
                        st.error("Invalid email or password.", icon=":material/error:")

            storage_label = "MongoDB ready" if get_mongo_users_collection() is not None else "Local account store"
            st.caption(f"Storage: {storage_label}. Accounts are keyed by email.")


def populate_example(name: str) -> None:
    example = PROFILE_EXAMPLES[name]
    st.session_state.target_title = example["target_title"]
    st.session_state.experience_level = example["experience_level"]
    st.session_state.selected_skills = example["skills"]
    st.session_state.resume_text = example["resume_text"]


def sidebar_controls():
    with st.sidebar:
        if LOGO_PATH.exists():
            st.logo(str(LOGO_PATH), icon_image=str(LOGO_ICON_PATH), size="large")

        current_theme = st.session_state.get("ui_theme", "Light")
        st.markdown('<div class="theme-switch-title">Appearance</div>', unsafe_allow_html=True)
        theme_cols = st.columns(2)
        with theme_cols[0]:
            if st.button(
                "Light",
                icon=":material/light_mode:",
                type="primary" if current_theme == "Light" else "secondary",
                key="theme_light_btn",
                width="stretch",
            ):
                st.session_state.ui_theme = "Light"
                st.rerun()
        with theme_cols[1]:
            if st.button(
                "Dark",
                icon=":material/dark_mode:",
                type="primary" if current_theme == "Dark" else "secondary",
                key="theme_dark_btn",
                width="stretch",
            ):
                st.session_state.ui_theme = "Dark"
                st.rerun()
        st.markdown(
            f'<div class="theme-current">Theme mode: <strong>{safe_escape(current_theme)}</strong></div>',
            unsafe_allow_html=True,
        )

        profile_name = st.session_state.profile_name or "Job seeker"
        profile_email = st.session_state.profile_email or LOGIN_EMAIL
        candidate_label = st.session_state.get("candidate_id") or "COLD_START_USER"
        cred_label = credential_id(profile_email, candidate_label)
        st.markdown(
            f"""
<div class="sidebar-account">
    <span>Signed in</span>
    <strong>{safe_escape(profile_name, "Job seeker")}</strong>
    <small>{safe_escape(profile_email, LOGIN_EMAIL)}</small>
    <div class="sidebar-mini-grid">
        <div><b>{safe_escape(candidate_label)}</b><em>Candidate ID</em></div>
        <div><b>{safe_escape(cred_label)}</b><em>Credential ID</em></div>
        <div><b>{safe_escape(st.session_state.get("recommendation_mode"), "Balanced")}</b><em>Mode</em></div>
        <div><b>{safe_escape(selected_platforms()[0])}</b><em>Apply site</em></div>
    </div>
</div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Log out", icon=":material/logout:", width="stretch"):
            st.session_state.authenticated = False
            st.session_state.profile_name = ""
            st.session_state.profile_email = ""
            st.session_state.recommendation_results = None
            st.rerun()

        st.session_state.active_view = st.segmented_control(
            "View",
            ["Recommend", "Analytics", "About"],
            key="main_view_selector",
        ) or st.session_state.active_view

        st.markdown("### Recommendation options")
        top_k = st.slider("Results to show", 3, 15, 6)
        recommendation_mode = st.segmented_control(
            "Ranking mode",
            list(RECOMMENDATION_MODES.keys()),
            key="recommendation_mode",
            help="Balanced blends content and history. Skill-first favors resume fit. History-first favors collaborative signals when available.",
        )
        alpha = RECOMMENDATION_MODES.get(recommendation_mode or "Balanced", RECOMMENDATION_MODES["Balanced"])
        strict_level = st.toggle("Filter by experience level", value=True)
        include_remote = st.toggle(
            "Include remote roles",
            value=True,
            help="Remote roles receive a location boost when your selected place has fewer exact catalog matches.",
        )

        st.markdown("### Candidate signal")
        candidate_id = st.text_input(
            "Candidate ID",
            key="candidate_id",
            help="Use C001, C002, etc. to test collaborative history. New IDs use content matching.",
        )
        st.caption(f"Content weight used by this mode: {alpha:.2f}")

    return top_k, alpha, strict_level, include_remote, candidate_id


def build_candidate_profile(candidate_id: str) -> dict:
    selected_skills = st.session_state.get("selected_skills") or []
    skills = ", ".join(str(skill) for skill in selected_skills if str(skill).strip())
    return {
        "candidate_id": str(candidate_id or "").strip() or "COLD_START_USER",
        "target_title": str(st.session_state.get("target_title") or "").strip(),
        "experience_level": st.session_state.get("experience_level") or "Mid",
        "location": str(st.session_state.get("location") or DEFAULT_LOCATION).strip() or DEFAULT_LOCATION,
        "skills": skills,
        "resume_text": str(st.session_state.get("resume_text") or "").strip(),
    }


def render_profile_form(top_k: int, alpha: float, strict_level: bool, include_remote: bool, candidate_id: str) -> None:
    st.markdown('<div class="section-label">Upload resume and build a candidate profile</div>', unsafe_allow_html=True)

    picked = st.pills(
        "Profile starters",
        list(PROFILE_EXAMPLES.keys()),
        selection_mode="single",
        label_visibility="collapsed",
    )
    if picked and picked != st.session_state.last_profile_example:
        populate_example(picked)
        st.session_state.last_profile_example = picked
        st.rerun()

    form_col, preview_col = st.columns([1.05, 0.95], gap="large")

    with form_col:
        uploaded_resume = st.file_uploader(
            "Upload resume",
            type=["pdf", "docx", "txt", "md"],
            max_upload_size=8,
            help="Upload a PDF, DOCX, TXT, or Markdown resume. Extracted text will prefill the profile.",
        )
        if uploaded_resume is not None and uploaded_resume.name != st.session_state.resume_file_name:
            extracted_text, file_name = extract_resume_file_text(uploaded_resume)
            if extracted_text.startswith("Could not read"):
                st.error(extracted_text, icon=":material/error:")
            elif extracted_text.strip():
                extracted_skills = extract_skills(extracted_text)
                merged_skills = list(dict.fromkeys((st.session_state.get("selected_skills") or []) + extracted_skills))
                st.session_state.resume_text = extracted_text
                st.session_state.resume_file_name = file_name
                st.session_state.selected_skills = merged_skills
                st.session_state.target_title = infer_title_from_resume(extracted_text, st.session_state.target_title)
                st.toast(f"Resume parsed: {file_name}", icon=":material/upload_file:")

        with st.form("recommendation_form", border=True):
            st.text_input("Target job title", key="target_title", placeholder="Machine Learning Engineer")
            st.text_input("Preferred place", key="location", placeholder="Bengaluru, India or Remote")
            st.selectbox("Experience level", EXP_LEVELS, key="experience_level")
            st.multiselect(
                "Skills",
                options=SKILL_ALL,
                key="selected_skills",
                accept_new_options=True,
                placeholder="Choose skills or type your own",
            )
            st.multiselect(
                "Job platforms",
                options=PLATFORMS,
                key="selected_platforms",
                placeholder="Choose platforms for live searches",
            )
            st.text_area(
                "Resume or bio",
                key="resume_text",
                height=210,
                placeholder="Paste your resume text or write a short professional summary.",
            )
            submitted = st.form_submit_button(
                "Find matching jobs",
                type="primary",
                icon=":material/search:",
                width="stretch",
            )

    with preview_col:
        resume_text = str(st.session_state.get("resume_text") or "")
        extracted = extract_skills(resume_text)
        preview_skills = list(dict.fromkeys((st.session_state.get("selected_skills") or []) + extracted))
        platform_badges = st.session_state.get("selected_platforms") or ["LinkedIn", "Naukri", "Indeed"]
        st.markdown(
            f"""
<div class="profile-preview">
    <div class="job-title">{safe_escape(st.session_state.target_title or "Target role")}</div>
    <div class="job-meta">Level: {safe_escape(st.session_state.experience_level, "Mid")} | Location: {safe_escape(st.session_state.location, DEFAULT_LOCATION)} | Candidate: {safe_escape(candidate_id or "COLD_START_USER")}</div>
    <div class="badge-row">
        {badge_html(preview_skills, "badge-info", "", 12)}
    </div>
    <div class="badge-row">
        {badge_html(platform_badges, "badge-match", "Search: ", 8)}
    </div>
    <p class="muted" style="margin-bottom:0">{safe_escape(str(st.session_state.resume_text or "")[:220])}{'...' if len(str(st.session_state.resume_text or "")) > 220 else ''}</p>
</div>
            """,
            unsafe_allow_html=True,
        )

    if submitted:
        candidate_profile = build_candidate_profile(candidate_id)
        if not candidate_profile["target_title"]:
            st.warning("Please enter a target job title.", icon=":material/warning:")
            return
        if not candidate_profile["skills"] and not candidate_profile["resume_text"]:
            st.warning("Add at least one skill or paste resume text.", icon=":material/warning:")
            return
        run_recommendations(candidate_profile, top_k, alpha, strict_level, include_remote)


def run_recommendations(candidate_profile: dict, top_k: int, alpha: float, strict_level: bool, include_remote: bool) -> None:
    with st.status("Preparing recommendation engine", expanded=False) as status:
        try:
            hybrid_model, jobs_df = get_hybrid_model()
            status.update(label="Ranking jobs", state="running")
            time.sleep(0.15)
        except Exception as exc:
            status.update(label="Pipeline failed", state="error")
            st.error(f"Pipeline error: {exc}", icon=":material/error:")
            return

        filter_df = jobs_df.copy()
        if strict_level and "experience_level" in filter_df.columns:
            level_mask = filter_df["experience_level"].astype(str).str.lower() == candidate_profile["experience_level"].lower()
            level_df = filter_df[level_mask]
            if not level_df.empty:
                filter_df = level_df
            else:
                st.info("No exact experience-level jobs were found, so all levels are ranked.", icon=":material/info:")

        recs = hybrid_model.recommend(candidate_profile, filter_df, top_k=len(filter_df), alpha=alpha)
        recs = apply_location_preference(recs, candidate_profile, include_remote).head(top_k)
        st.session_state.recommendation_results = recs
        st.session_state.last_candidate_profile = candidate_profile
        status.update(label="Recommendations ready", state="complete")


def render_recommendations() -> None:
    recs = st.session_state.recommendation_results
    candidate_profile = st.session_state.last_candidate_profile

    if recs is None or candidate_profile is None:
        st.markdown(
            """
<div class="empty-state">
    <div class="job-title">No recommendations yet</div>
    <div class="job-meta">Fill the profile form and run the matcher to see ranked jobs here.</div>
</div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown(f'<div class="section-label">Top {len(recs)} recommendations</div>', unsafe_allow_html=True)
    best_score = percent(recs.iloc[0].get("hybrid_score", 0)) if len(recs) else 0
    avg_confidence = percent(recs["score_confidence"].mean()) if "score_confidence" in recs else 0
    avg_place = percent(recs["location_match_score"].mean()) if "location_match_score" in recs else 0
    cf_used = int(recs.get("collaborative_used", pd.Series(dtype=bool)).sum())
    preferred_location = candidate_profile.get("location", DEFAULT_LOCATION)

    metric_cols = st.columns(4)
    with metric_cols[0].container(border=True):
        st.metric("Best fit", f"{best_score}%")
    with metric_cols[1].container(border=True):
        st.metric("Average confidence", f"{avg_confidence}%")
    with metric_cols[2].container(border=True):
        st.metric("Place fit", f"{avg_place}%")
    with metric_cols[3].container(border=True):
        st.metric("Collaborative signals", f"{cf_used}/{len(recs)}")

    render_platform_matches(recs, candidate_profile)

    for rank, (_, row) in enumerate(recs.iterrows(), start=1):
        job = row.to_dict()
        explanation = explain_recommendation(candidate_profile, job)
        matched = explanation["matched_skills"]
        missing = explanation["missing_skills"]
        score = percent(row.get("hybrid_score", 0))
        company_text = str(row.get("company", "Company not listed"))
        location_text = str(row.get("location", "Remote"))
        company = escape(company_text)
        location = escape(location_text)
        place_fit = percent(row.get("location_match_score", 0))
        salary = escape(str(row.get("salary", "Not disclosed")))
        level = escape(str(row.get("experience_level", "Level not listed")))
        apply_links = job_apply_links(job, candidate_profile)
        primary_apply = apply_links[0]
        apply_actions = "".join(
            (
                f'<a class="apply-button" href="{safe_escape(link["url"])}" target="_blank" rel="noopener">Apply on {safe_escape(link["platform"])}</a>'
                if idx == 0
                else f'<a class="apply-chip" href="{safe_escape(link["url"])}" target="_blank" rel="noopener">{safe_escape(link["platform"])}</a>'
            )
            for idx, link in enumerate(apply_links[:4])
        )

        st.markdown(
            f"""
<article class="result-card">
    <div class="result-head">
        <div class="rank-badge">#{rank}</div>
        <div>
            <div class="result-topline">
                <span class="result-company-pill">{company}</span>
                <span class="result-company-pill">{location}</span>
            </div>
            <div class="job-title"><a class="result-title-link" href="{safe_escape(primary_apply["url"])}" target="_blank" rel="noopener">{safe_escape(row.get("title", "Untitled role"))}</a></div>
            <div class="job-meta">{company} | {location} | {level} level | {salary}</div>
            <div class="badge-row">
                {badge_html(matched, "badge-match", "Match: ", 8)}
                {badge_html(missing, "badge-gap", "Gap: ", 5)}
                {badge_html([f"Place fit {place_fit}% near {preferred_location}"], "badge-place", "", 1)}
            </div>
            <p class="result-reason">{safe_escape(explanation["explanation"])}</p>
            <div class="apply-actions">
                {apply_actions}
            </div>
            <div class="apply-hint">Choose LinkedIn, Naukri, Indeed, or your selected platform to open the application search for {safe_escape(company_text)}.</div>
        </div>
        <div class="score-chip">{score}%<small>fit</small></div>
    </div>
</article>
            """,
            unsafe_allow_html=True,
        )


def render_platform_matches(recs: pd.DataFrame, candidate_profile: dict) -> None:
    platform_df = build_platform_matches(recs, candidate_profile)
    if platform_df.empty:
        return

    st.markdown('<div class="section-label">Best platform searches from your resume match</div>', unsafe_allow_html=True)
    top_cards = platform_df.sort_values("fit", ascending=False).head(6)
    card_html = '<div class="platform-grid">'
    for _, row in top_cards.iterrows():
        card_html += f"""
<div class="platform-card">
    <strong>{escape(row["platform"])}</strong>
    <span>{escape(row["matched_role"])} near {escape(row["location"])}<br>{escape(str(row["fit"]))}% recommender fit</span>
    <a href="{escape(row["open_search"])}" target="_blank" rel="noopener">Open jobs</a>
</div>
"""
    card_html += "</div>"
    st.markdown(card_html, unsafe_allow_html=True)

    with st.expander("All generated platform searches", icon=":material/open_in_new:"):
        st.dataframe(
            platform_df,
            hide_index=True,
            width="stretch",
            column_config={
                "platform": st.column_config.TextColumn("Platform", pinned=True),
                "matched_role": st.column_config.TextColumn("Matched role"),
                "company_signal": st.column_config.TextColumn("Catalog signal"),
                "fit": st.column_config.ProgressColumn("Fit", min_value=0, max_value=100),
                "location": st.column_config.TextColumn("Location"),
                "open_search": st.column_config.LinkColumn("Open search", display_text="Open"),
            },
        )


def render_recommend_page(top_k: int, alpha: float, strict_level: bool, include_remote: bool, candidate_id: str) -> None:
    render_welcome_header()
    render_profile_form(top_k, alpha, strict_level, include_remote, candidate_id)
    render_recommendations()


def render_analytics_page() -> None:
    render_banner(
        "Catalog analytics",
        "Review the job catalog used by the recommender, including levels, companies, locations, and common skills.",
        eyebrow="Dataset view",
    )
    jobs_df = load_jobs()
    if jobs_df is None:
        st.info("The processed job catalog has not been built yet. Run a recommendation once to generate it.", icon=":material/info:")
        return

    cols = st.columns(4)
    metrics = [
        ("Jobs", len(jobs_df)),
        ("Companies", jobs_df["company"].nunique() if "company" in jobs_df else 0),
        ("Locations", jobs_df["location"].nunique() if "location" in jobs_df else 0),
        ("Levels", jobs_df["experience_level"].nunique() if "experience_level" in jobs_df else 0),
    ]
    for col, (label, value) in zip(cols, metrics):
        with col.container(border=True):
            st.metric(label, value)

    chart_cols = st.columns(2, gap="large")
    with chart_cols[0]:
        st.subheader("Jobs by experience level", anchor=False)
        if "experience_level" in jobs_df:
            st.bar_chart(jobs_df["experience_level"].value_counts())

    with chart_cols[1]:
        st.subheader("Top companies", anchor=False)
        if "company" in jobs_df:
            st.bar_chart(jobs_df["company"].value_counts().head(10))

    chart_cols = st.columns(2, gap="large")
    with chart_cols[0]:
        st.subheader("Top locations", anchor=False)
        if "location" in jobs_df:
            st.bar_chart(jobs_df["location"].value_counts().head(10))

    with chart_cols[1]:
        st.subheader("In-demand skills", anchor=False)
        if "skills" in jobs_df:
            skill_counts = {}
            for skills in jobs_df["skills"].dropna():
                for skill in [item.strip() for item in str(skills).split(",") if item.strip()]:
                    skill_counts[skill] = skill_counts.get(skill, 0) + 1
            top_skills = pd.Series(skill_counts).sort_values(ascending=False).head(15)
            st.bar_chart(top_skills)

    with st.expander("Preview job catalog", icon=":material/table_chart:"):
        display_cols = [col for col in ["title", "company", "location", "experience_level", "skills"] if col in jobs_df]
        st.dataframe(jobs_df[display_cols].head(40), width="stretch", hide_index=True)


def render_about_page() -> None:
    render_banner(
        "How JobMatch AI works",
        "The project combines resume parsing, text similarity, structured profile matching, and collaborative filtering when interaction history exists.",
        eyebrow="Model architecture",
    )

    cols = st.columns(3, gap="large")
    cards = [
        ("Resume parsing", "PDF, DOCX, TXT, and Markdown uploads extract profile text and technical skills."),
        ("Hybrid ranking", "Skill overlap, title alignment, experience level, and content similarity drive the fit score."),
        ("Platform launch", "Top ranked roles generate targeted job searches on LinkedIn, Naukri, Indeed, and more."),
    ]
    for col, (title, body) in zip(cols, cards):
        with col:
            st.markdown(
                f"""
<div class="insight-card">
    <strong>{escape(title)}</strong>
    <span>{escape(body)}</span>
</div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-label">Project modules</div>', unsafe_allow_html=True)
    st.code(
        """app/streamlit_app.py        Streamlit login and dashboard
data/app_users.json          Local hashed account store
src/models/content_based.py   Text similarity scoring
src/models/collaborative.py   Interaction-based ratings
src/models/hybrid.py          Final rank blending
src/inference/explain.py      Match explanation details
api/                          FastAPI endpoints""",
        language="text",
    )


init_state()
apply_theme()

if not st.session_state.authenticated:
    render_login()
    st.stop()

top_k_value, alpha_value, strict_level_value, include_remote_value, candidate_id_value = sidebar_controls()

if st.session_state.active_view == "Analytics":
    render_analytics_page()
elif st.session_state.active_view == "About":
    render_about_page()
else:
    render_recommend_page(top_k_value, alpha_value, strict_level_value, include_remote_value, candidate_id_value)
