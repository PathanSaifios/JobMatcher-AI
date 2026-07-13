from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:
    from api.routes import router
except ModuleNotFoundError as exc:
    if exc.name in {"pandas", "numpy", "torch", "sklearn", "sentence_transformers"}:
        from api.index import app
    else:
        raise
else:
    app = FastAPI(
        title="Hybrid Job Recommender System",
        description="An AI/ML-powered hybrid recommendation engine combining Sentence-BERT content-based filtering and PyTorch collaborative filtering.",
        version="1.0.0"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api/v1")

    @app.get("/")
    def root():
        return {
            "message": "Hybrid Job Recommender API is running.",
            "docs": "/docs",
            "endpoints": {
                "recommend": "/api/v1/recommend",
                "extract_skills": "/api/v1/extract-skills",
                "jobs": "/api/v1/jobs"
            }
        }

    @app.get("/health")
    def health_check():
        return {"status": "healthy"}
