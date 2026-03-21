"""
health_router.py
----------------
GET /health  — Liveness + readiness probe.

Response shape
--------------
{
  "status":       "healthy" | "degraded",
  "api_version":  "v1.0.0",
  "db_connected": true | false
}

Status rules
------------
  healthy  — ML models are loaded (predictions available)
  degraded — Models not loaded (POST /predict will return 503)

Database
--------
  db_connected reflects whether MySQL responds to SELECT 1.
  Returns False if MySQL is unreachable, credentials are wrong,
  or the database does not exist.

Used by
-------
  - Frontend Header (30-second polling → 🟢 / 🔴)
  - APIStatus page
  - Docker / Kubernetes liveness probes
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

from database import get_db_engine_label, is_db_connected
from model_loader import is_models_loaded
from schemas import HealthResponse

logger = logging.getLogger("backend.health_router")
router = APIRouter(tags=["Health"])

_API_VERSION = "v1.0.0"


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="API liveness and readiness probe",
)
def health_check() -> HealthResponse:
    """
    Return the current liveness / readiness status of the API.

    - **status**       : 'healthy' when ML models are loaded; 'degraded' otherwise.
    - **api_version**  : Semantic version string frozen at deployment time.
    - **db_connected** : True when MySQL responds to a trivial SELECT 1 query.
    """
    models_ready: bool = is_models_loaded()
    db_ready: bool     = is_db_connected()

    # healthy  = able to serve predictions (models loaded)
    # degraded = models absent; inference will return 503
    status = "healthy" if models_ready else "degraded"

    logger.debug(
        "Health check → %s  (models_loaded=%s  db_connected=%s)",
        status, models_ready, db_ready,
    )

    return HealthResponse(
        status=status,
        api_version=_API_VERSION,
        db_connected=db_ready,
        db_engine=get_db_engine_label(),
    )
