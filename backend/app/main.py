"""
main.py
-------
FastAPI application entry point for the
Ensemble-Based Hybrid Disaster Prediction System.

Startup sequence
----------------
1. Load ML model artefacts via model_loader.load_models().
2. Initialise MySQL database via database.init_db()
   (creates tables if they do not yet exist).

Routers mounted
---------------
  predict_router         →  POST /predict
                              GET  /models/status
  health_router          →  GET  /health
  history_router         →  GET  /history
                              GET  /history/{id}
                              DELETE /history
  explainability_router  →  GET  /explainability

CORS
----
  allow_origins=["*"]  — permits the React dev-server on any port.

Database
--------
  MySQL via PyMySQL + SQLAlchemy ORM.
  Connection URL: mysql+pymysql://<user>:<pass>@<host>:<port>/disaster_prediction_db
  Configure via DATABASE_URL environment variable or individual DB_* vars.
  Tables are defined in models.py and created via Base.metadata.create_all().

Pre-requisite
-------------
  The MySQL database must be created before starting the API:

      CREATE DATABASE disaster_prediction_db
          CHARACTER SET utf8mb4
          COLLATE utf8mb4_unicode_ci;
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database import init_db, DATABASE_URL
from model_loader import load_models
from predict_router import router as predict_router
from health_router import router as health_router
from history_router import router as history_router
from explainability_router import router as explainability_router

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("backend.main")
_DB_ENGINE_LABEL = "SQLite" if DATABASE_URL.startswith("sqlite") else "MySQL"


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan handler.

    Startup
    -------
    - Load ML artefacts     (warns on failure, never raises)
    - Initialise MySQL DB   (creates tables; warns on failure, never raises)

    Shutdown
    --------
    - MySQL connections are managed per-request via get_db(); no teardown needed.
      SQLAlchemy's connection pool handles cleanup automatically.
    """
    # ── startup ────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Ensemble-Based Hybrid Disaster Prediction System — API")
    logger.info("=" * 60)
    logger.info("Database engine : %s", _DB_ENGINE_LABEL)

    # 1. ML models
    try:
        load_models()
        logger.info("ML model artefacts loaded successfully.")
    except FileNotFoundError as exc:
        logger.warning(
            "ML models not found: %s\n"
            "→ Run 'python ml_pipeline/train_pipeline.py' to generate artefacts.\n"
            "→ /predict will return HTTP 503 until models are available.",
            exc,
        )
    except Exception as exc:
        logger.error("Unexpected error loading models: %s", exc)

    # 2. Database — create tables
    try:
        init_db()
        logger.info("%s database ready — tables verified.", _DB_ENGINE_LABEL)
    except Exception as exc:
        logger.error(
            "Database initialisation failed: %s\n"
            "→ Ensure your configured database is reachable.\n"
            "→ Prediction history will NOT be persisted until DB is available.",
            exc,
        )

    logger.info("API startup complete.  Listening for requests.")
    logger.info("=" * 60)

    yield  # ← application runs here

    # ── shutdown ───────────────────────────────────────────────────────────
    logger.info("Shutting down API … connection pool disposed.")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Ensemble-Based Hybrid Disaster Prediction System",
    description=(
        "Stacking-Based Ensemble Disaster Risk Assessment Framework.\n\n"
        "Provides real-time disaster risk prediction using a heterogeneous "
        "ensemble of Random Forest, Gradient Boosting, and SVM base learners "
        "combined via a Logistic Regression meta-learner. "
        "Risk is quantified through the Disaster Risk Index (DRI).\n\n"
        "**Database:** MySQL 8+ (via PyMySQL + SQLAlchemy ORM)  \n"
        "**Models:** Loaded from `ml_pipeline/models/*.pkl`"
    ),
    version="v1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS — allow React dev server (any origin, any method, any header)
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(predict_router)        # POST /predict, GET /models/status
app.include_router(health_router)         # GET  /health
app.include_router(history_router)        # GET  /history, GET /history/{id}, DELETE /history
app.include_router(explainability_router) # GET  /explainability

# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------

@app.get("/", tags=["Root"], summary="API root")
async def root() -> dict:
    """Return a simple confirmation that the API is reachable."""
    return {
        "message":       "Ensemble Disaster Prediction API Running",
        "model_version": "v1.0.0",
        "database":      _DB_ENGINE_LABEL,
        "docs":          "/docs",
    }


# ---------------------------------------------------------------------------
# Dev entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
