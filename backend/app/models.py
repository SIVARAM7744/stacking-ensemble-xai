"""
models.py
---------
SQLAlchemy ORM model for the Ensemble-Based Hybrid Disaster Prediction System.

Table: prediction_records
-------------------------
One row is inserted for every successful POST /predict call.
The row captures the full prediction context:
  - metadata       : timestamp (UTC, auto-generated), disaster_type, location
  - input features : all environmental sensors relevant to the prediction type
  - output         : DRI, risk_level, probability, confidence, per-model probs

MySQL column types
------------------
  Integer  → INT / BIGINT (auto-increment PK)
  Float    → DOUBLE (8-byte)
  String   → VARCHAR with explicit length (required by MySQL)
  DateTime → DATETIME (UTC enforced in Python; MySQL has no tz storage)

Indexes
-------
  Clustered primary key    : id
  Single-column indexes    : timestamp, disaster_type, risk_level, dri
  Composite indexes        : (disaster_type, risk_level), (timestamp, disaster_type)

Usage
-----
  # In predict_router.py
  record = PredictionRecord.from_prediction(
      disaster_type="Flood",
      location="Chennai",
      region_code="Zone-4",
      input_features={"rainfall": 245.8, ...},
      output={
          "predicted_disaster": "Flood",
          "probability": 0.87,
          "dri": 0.82,
          "risk_level": "HIGH",
          "confidence": 94.5,
          "model_probabilities": {"random_forest": 0.89, ...},
      },
  )
  db.add(record)
  db.commit()
  db.refresh(record)   # populates auto-generated id + timestamp

  # In history_router.py — serialise to API dict
  api_dict = record.to_api()
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import DateTime, Float, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from database import Base


# ---------------------------------------------------------------------------
# UTC timestamp helper
# ---------------------------------------------------------------------------

def _utc_now() -> datetime:
    """
    Return the current UTC timestamp as a *naive* datetime.

    MySQL DATETIME does not store timezone information.
    We store UTC and interpret all values as UTC on read.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ---------------------------------------------------------------------------
# ORM Model
# ---------------------------------------------------------------------------

class PredictionRecord(Base):
    """
    SQLAlchemy ORM model mapped to the `prediction_records` table in MySQL.

    Every successful POST /predict inserts one row via PredictionRecord.from_prediction().
    The timestamp column is automatically populated with the current UTC time on insert.
    """

    __tablename__ = "prediction_records"

    # ── Composite table-level indexes ──────────────────────────────────────
    __table_args__ = (
        Index("ix_pr_type_risk", "disaster_type", "risk_level"),
        Index("ix_pr_ts_type",   "timestamp",     "disaster_type"),
    )

    # ── Primary key ────────────────────────────────────────────────────────
    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        index=True,
        comment="Auto-increment primary key",
    )

    # ── Metadata ───────────────────────────────────────────────────────────
    timestamp: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=_utc_now,           # ← auto-generated on every INSERT
        index=True,
        comment="UTC timestamp of prediction request (auto-set on insert)",
    )
    disaster_type: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        index=True,
        comment="Flood | Earthquake",
    )
    location: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="Optional free-text location submitted with request",
    )
    region_code: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        comment="Optional region / zone code",
    )

    # ── Input features (nullable — only relevant subset is populated) ──────
    rainfall: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Rainfall mm [Flood only]"
    )
    temperature: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Temperature °C"
    )
    humidity: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Relative humidity %"
    )
    soil_moisture: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Soil moisture % [Flood only]"
    )
    wind_speed: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Wind speed km/h"
    )
    atmospheric_pressure: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Atmospheric pressure hPa"
    )
    seismic_activity: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Seismic activity Richter scale [Earthquake only]"
    )
    previous_disaster: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="0 = No previous disaster  |  1 = Yes",
    )

    # ── Prediction output ──────────────────────────────────────────────────
    predicted_disaster: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="Flood | Earthquake | No Risk",
    )
    probability: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Stacking meta-learner P(positive class)  [0–1]",
    )
    dri: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        index=True,
        comment="Disaster Risk Index: 0.6 * stacking_prob + 0.4 * avg_base_prob",
    )
    risk_level: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        index=True,
        comment="LOW (≤0.33) | MODERATE (≤0.66) | HIGH (>0.66)",
    )
    confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Prediction confidence % derived from model agreement and decisiveness",
    )

    # ── Per-model probabilities (flat columns — avoids JSON type) ──────────
    rf_probability: Mapped[float] = mapped_column(
        Float, nullable=False, comment="Random Forest P(positive)"
    )
    gb_probability: Mapped[float] = mapped_column(
        Float, nullable=False, comment="Gradient Boosting P(positive)"
    )
    svm_probability: Mapped[float] = mapped_column(
        Float, nullable=False, comment="SVM P(positive)"
    )
    stacking_probability: Mapped[float] = mapped_column(
        Float, nullable=False, comment="Stacking meta-learner P(positive)"
    )

    # -----------------------------------------------------------------------
    # Class-level factory
    # -----------------------------------------------------------------------

    @classmethod
    def from_prediction(
        cls,
        *,
        disaster_type: str,
        location: Optional[str],
        region_code: Optional[str],
        input_features: Dict[str, Any],
        output: Dict[str, Any],
    ) -> "PredictionRecord":
        """
        Construct a PredictionRecord from the structured dicts produced by
        predict_router.py.  Ready for db.add() + db.commit().

        The `timestamp` column is NOT set here — it is set automatically by
        the column `default=_utc_now` when SQLAlchemy flushes the INSERT.

        Parameters
        ----------
        disaster_type   : "Flood" or "Earthquake"
        location        : Free-text location string (may be None)
        region_code     : Zone / region code (may be None)
        input_features  : Dict of raw feature values keyed by column name
        output          : Dict with keys:
                            predicted_disaster, probability, dri, risk_level,
                            confidence, model_probabilities (nested dict)
        """
        model_probs: Dict[str, float] = output.get("model_probabilities", {})

        return cls(
            # metadata
            disaster_type=disaster_type,
            location=location,
            region_code=region_code,
            # input features — only present fields are non-None
            rainfall=input_features.get("rainfall"),
            temperature=input_features.get("temperature"),
            humidity=input_features.get("humidity"),
            soil_moisture=input_features.get("soil_moisture"),
            wind_speed=input_features.get("wind_speed"),
            atmospheric_pressure=input_features.get("atmospheric_pressure"),
            seismic_activity=input_features.get("seismic_activity"),
            previous_disaster=int(input_features.get("previous_disaster", 0)),
            # prediction output
            predicted_disaster=str(output["predicted_disaster"]),
            probability=float(output["probability"]),
            dri=float(output["dri"]),
            risk_level=str(output["risk_level"]),
            confidence=float(output["confidence"]),
            # per-model probabilities
            rf_probability=float(model_probs.get("random_forest", 0.0)),
            gb_probability=float(model_probs.get("gradient_boosting", 0.0)),
            svm_probability=float(model_probs.get("svm", 0.0)),
            stacking_probability=float(model_probs.get("stacking", 0.0)),
        )

    # -----------------------------------------------------------------------
    # Serialisation → API response dict
    # -----------------------------------------------------------------------

    def to_api(self) -> Dict[str, Any]:
        """
        Return a JSON-serialisable dict for GET /history API responses.

        - id        : str(int PK) — consistent with HistoryRecord.id: Optional[str]
        - timestamp : ISO-8601 string with 'Z' suffix (UTC marker)
        - input     : nested dict of all feature columns
        - model_probabilities : nested dict of per-model probabilities
        """
        return {
            "id": str(self.id),
            "timestamp": (
                self.timestamp.isoformat() + "Z"   # 'Z' = UTC
                if self.timestamp
                else None
            ),
            "disaster_type": self.disaster_type,
            "location":      self.location,
            "region_code":   self.region_code,
            # grouped input features
            "input": {
                "rainfall":             self.rainfall,
                "temperature":          self.temperature,
                "humidity":             self.humidity,
                "soil_moisture":        self.soil_moisture,
                "wind_speed":           self.wind_speed,
                "atmospheric_pressure": self.atmospheric_pressure,
                "seismic_activity":     self.seismic_activity,
                "previous_disaster":    self.previous_disaster,
            },
            # prediction output
            "predicted_disaster": self.predicted_disaster,
            "probability":        self.probability,
            "dri":                self.dri,
            "risk_level":         self.risk_level,
            "confidence":         self.confidence,
            # per-model probabilities (nested)
            "model_probabilities": {
                "random_forest":     self.rf_probability,
                "gradient_boosting": self.gb_probability,
                "svm":               self.svm_probability,
                "stacking":          self.stacking_probability,
            },
        }

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"PredictionRecord("
            f"id={self.id}, "
            f"type={self.disaster_type!r}, "
            f"risk={self.risk_level!r}, "
            f"dri={self.dri:.4f}, "
            f"ts={self.timestamp!r})"
        )
