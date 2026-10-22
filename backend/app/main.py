"""
NOVA-X Backend Entrypoint
---------------------------
Run with:
    uvicorn app.main:app --reload --port 8000
(from inside the backend/ directory, with the venv activated)
"""
from contextlib import asynccontextmanager
import logging
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database.database import init_db
from app.api import detect, dashboard, unknown_objects, clusters, models_api, analytics, evaluation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("nova_x")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Initializing %s database at %s", app.title, settings.DATABASE_URL)
    init_db()
    logger.info("NOVA-X backend ready. Known classes: %s", settings.KNOWN_CLASSES)
    yield


app = FastAPI(
    title="NOVA-X API",
    description="Novelty-Aware Open-World Visual Intelligence — "
                 "known-object detection, OOD/novelty detection, open-vocabulary "
                 "recognition, and continual learning.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detect.router)
app.include_router(dashboard.router)
app.include_router(unknown_objects.router)
app.include_router(clusters.router)
app.include_router(models_api.router)
app.include_router(analytics.router)
app.include_router(evaluation.router)


@app.get("/api/health")
def health():
    return {"status": "ok", "app": settings.APP_NAME}
