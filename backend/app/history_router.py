"""
history_router.py
-----------------
GET    /history        — Paginated, filterable prediction history from MySQL.
GET    /history/{id}   — Single prediction record by integer primary key.
DELETE /history        — Clear all prediction history (admin / dev use).

All records are sourced from the MySQL `prediction_records` table via
SQLAlchemy ORM.  If the database is offline the endpoints return 503.

Query parameters for GET /history
----------------------------------
  page           int  ≥ 1    (default 1)
  page_size      int  1–100  (default 20)
  disaster_type  "Flood" | "Earthquake"           (optional filter)
  risk_level     "LOW"  | "MODERATE" | "HIGH"     (optional filter)
  date_from      ISO-8601 date string  e.g. "2024-01-01"  (optional)
  date_to        ISO-8601 date string  e.g. "2024-12-31"  (optional)

Response shape (GET /history)
------------------------------
{
  "total": 42,
  "page": 1,
  "page_size": 20,
  "records": [
    {
      "id": "1",
      "timestamp": "2024-01-15T10:30:00Z",
      "disaster_type": "Flood",
      "location": "Chennai",
      "region_code": "Zone-4",
      "predicted_disaster": "Flood",
      "probability": 0.87,
      "dri": 0.82,
      "risk_level": "HIGH",
      "confidence": 94.5,
      "model_probabilities": { "random_forest": 0.89, ... },
      "input": { "rainfall": 245.8, ... }
    },
    ...
  ]
}
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, delete
from sqlalchemy.orm import Session

from database import get_db, is_db_connected
from models import PredictionRecord
from schemas import HistoryRecord, HistoryResponse

logger = logging.getLogger("backend.history_router")
router = APIRouter(prefix="/history", tags=["History"])


# ---------------------------------------------------------------------------
# Guards and helpers
# ---------------------------------------------------------------------------

def _db_guard() -> None:
    """Raise HTTP 503 if the MySQL database is not reachable."""
    if not is_db_connected():
        raise HTTPException(
            status_code=503,
            detail=(
                "Database not connected. "
                "Ensure MySQL is running and the database exists, then restart the API."
            ),
        )


def _parse_date(value: Optional[str], field_name: str) -> Optional[datetime]:
    """Parse an ISO-8601 date string to a naive UTC datetime or raise HTTP 400."""
    if value is None:
        return None
    try:
        dt = datetime.fromisoformat(value)
        # Store as naive UTC (MySQL DATETIME has no tz storage)
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        return dt
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid date format for '{field_name}'. Use YYYY-MM-DD or ISO-8601.",
        )


# ---------------------------------------------------------------------------
# GET /history
# ---------------------------------------------------------------------------

@router.get(
    "",
    response_model=HistoryResponse,
    summary="Retrieve paginated prediction history",
)
def get_history(
    page: int       = Query(1,  ge=1,          description="Page number (1-based)"),
    page_size: int  = Query(20, ge=1, le=100,  description="Results per page"),
    disaster_type: Optional[str] = Query(
        None, description="Filter: 'Flood' or 'Earthquake'"
    ),
    risk_level: Optional[str] = Query(
        None, description="Filter: 'LOW', 'MODERATE', or 'HIGH'"
    ),
    date_from: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    date_to:   Optional[str] = Query(None, description="End date   (YYYY-MM-DD)"),
    db: Session = Depends(get_db),
) -> HistoryResponse:
    """
    Return a paginated, optionally filtered list of past predictions
    stored in the MySQL database.

    Records are ordered newest-first (timestamp DESC).
    Each record contains the full input feature set, all model probabilities,
    the DRI, risk level, and confidence score.
    """
    _db_guard()

    # ── Build base query ───────────────────────────────────────────────────
    query = db.query(PredictionRecord)

    # ── Filter: disaster_type ──────────────────────────────────────────────
    if disaster_type:
        normalized = disaster_type.strip().capitalize()
        if normalized not in ("Flood", "Earthquake"):
            raise HTTPException(
                status_code=400,
                detail="disaster_type filter must be 'Flood' or 'Earthquake'.",
            )
        query = query.filter(PredictionRecord.disaster_type == normalized)

    # ── Filter: risk_level ────────────────────────────────────────────────
    if risk_level:
        normalized_risk = risk_level.strip().upper()
        if normalized_risk not in ("LOW", "MODERATE", "HIGH"):
            raise HTTPException(
                status_code=400,
                detail="risk_level filter must be 'LOW', 'MODERATE', or 'HIGH'.",
            )
        query = query.filter(PredictionRecord.risk_level == normalized_risk)

    # ── Filter: date range ────────────────────────────────────────────────
    dt_from = _parse_date(date_from, "date_from")
    dt_to   = _parse_date(date_to,   "date_to")

    if dt_from:
        query = query.filter(PredictionRecord.timestamp >= dt_from)
    if dt_to:
        end_of_day = dt_to.replace(hour=23, minute=59, second=59)
        query = query.filter(PredictionRecord.timestamp <= end_of_day)

    # ── Total count ───────────────────────────────────────────────────────
    try:
        total: int = query.count()
    except Exception as exc:
        logger.exception("MySQL count error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Database error: {exc}") from exc

    # ── Fetch paginated page ──────────────────────────────────────────────
    skip = (page - 1) * page_size
    try:
        rows = (
            query
            .order_by(PredictionRecord.timestamp.desc())   # newest first
            .offset(skip)
            .limit(page_size)
            .all()
        )
    except Exception as exc:
        logger.exception("MySQL query error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Database error: {exc}") from exc

    # ── Serialise ORM rows → Pydantic HistoryRecord ───────────────────────
    records = [HistoryRecord(**row.to_api()) for row in rows]

    logger.info(
        "GET /history  page=%d  page_size=%d  total=%d  returned=%d",
        page, page_size, total, len(records),
    )

    return HistoryResponse(
        total=total,
        page=page,
        page_size=page_size,
        records=records,
    )


# ---------------------------------------------------------------------------
# GET /history/{id}
# ---------------------------------------------------------------------------

@router.get(
    "/{record_id}",
    response_model=HistoryRecord,
    summary="Retrieve a single prediction record by ID",
)
def get_history_record(
    record_id: int,
    db: Session = Depends(get_db),
) -> HistoryRecord:
    """Fetch one prediction record by its auto-incremented integer primary key."""
    _db_guard()

    try:
        row = (
            db.query(PredictionRecord)
            .filter(PredictionRecord.id == record_id)
            .first()
        )
    except Exception as exc:
        logger.exception("MySQL find error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Database error: {exc}") from exc

    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Prediction record with id='{record_id}' not found.",
        )

    return HistoryRecord(**row.to_api())


# ---------------------------------------------------------------------------
# DELETE /history  (admin / development)
# ---------------------------------------------------------------------------

@router.delete(
    "",
    summary="Clear all prediction history",
    response_description="Number of deleted records",
)
def clear_history(db: Session = Depends(get_db)) -> dict:
    """
    Permanently delete all prediction records from the MySQL database.
    Intended for development and testing only.
    """
    _db_guard()

    try:
        result = db.execute(delete(PredictionRecord))
        db.commit()
        deleted: int = result.rowcount  # type: ignore[assignment]
        logger.info("Cleared %d prediction records from MySQL.", deleted)
        return {"deleted": deleted, "message": f"{deleted} record(s) permanently removed."}
    except Exception as exc:
        db.rollback()
        logger.exception("MySQL delete error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Database error: {exc}") from exc
