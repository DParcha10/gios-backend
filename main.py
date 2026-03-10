"""
GIOS Environmental Monitoring Platform — FastAPI Application Entry Point

启动命令:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Swagger docs → http://localhost:8000/docs
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.utils.cache import close_cache, get_cache

# ── Route imports ───────────────────────────────────────────────────────────
from app.api.routes.health import router as health_router
from app.api.routes.data import router as data_router
from app.api.routes.analysis import router as analysis_router
from app.api.routes.timeseries import router as timeseries_router

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("gios")


# ── Application Lifespan ───────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks."""
    logger.info("Starting %s …", settings.app_name)
    get_cache()  # warm up the disk cache
    yield
    # Shutdown
    from app.services.integration import data_integration_service
    await data_integration_service.close()
    close_cache()
    logger.info("Shutdown complete.")


# ── FastAPI App ─────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.app_name,
    description=(
        "Backend API for the GIOS Environmental Monitoring Platform. "
        "Provides satellite imagery acquisition (Landsat & Sentinel-2), "
        "spectral index computation, time-series analysis, zonal statistics, "
        "algal-bloom detection, and multi-source data integration."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# ── CORS ────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routers ───────────────────────────────────────────────────────

app.include_router(health_router)
app.include_router(data_router)
app.include_router(analysis_router)
app.include_router(timeseries_router)


# ── Root redirect ───────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return {
        "name": settings.app_name,
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }
