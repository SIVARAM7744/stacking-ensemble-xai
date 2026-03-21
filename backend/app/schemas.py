"""
schemas.py
----------
Pydantic v2 request / response schemas for the Ensemble-Based Hybrid
Disaster Prediction System backend.

Schemas
-------
  PredictionRequest    — POST /predict            (input)
  ModelProbabilities   — nested inside responses
  PredictionResponse   — POST /predict            (output)
  HistoryRecord        — GET /history             (single row)
  HistoryResponse      — GET /history             (paginated list)
  HealthResponse       — GET /health
  ModelsStatusResponse — GET /models/status
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SchemaModel(BaseModel):
    model_config = {"protected_namespaces": ()}


# ===========================================================================
# POST /predict — Request
# ===========================================================================

class PredictionRequest(SchemaModel):
    """
    Input schema for the ensemble prediction endpoint.
    Disaster-type controls which optional fields are required at the
    router level; Pydantic only enforces the field types here.
    """

    disaster_type: str = Field(
        ...,
        description="'Flood' or 'Earthquake'",
        examples=["Flood"],
    )

    # ── Flood-only ──────────────────────────────────────────────────────────
    rainfall: Optional[float] = Field(
        None,
        ge=0,
        description="Rainfall in mm  [Flood only, ≥ 0]",
    )
    soil_moisture: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Soil moisture %  [Flood only, 0-100]",
    )

    # ── Earthquake-only ─────────────────────────────────────────────────────
    seismic_activity: Optional[float] = Field(
        None,
        ge=0,
        le=10,
        description="Seismic activity on Richter scale  [Earthquake only, 0-10]",
    )

    # ── Shared environmental features ────────────────────────────────────────
    temperature: Optional[float] = Field(
        None,
        description="Temperature in °C",
    )
    humidity: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Relative humidity %  [0-100]",
    )
    wind_speed: Optional[float] = Field(
        None,
        ge=0,
        description="Wind speed in km/h  [≥ 0]",
    )
    atmospheric_pressure: Optional[float] = Field(
        None,
        ge=800,
        le=1100,
        description="Atmospheric pressure in hPa  [800-1100]",
    )

    # ── Categorical feature ──────────────────────────────────────────────────
    previous_disaster: int = Field(
        ...,
        ge=0,
        le=1,
        description="Previous disaster occurrence: 0 = No, 1 = Yes",
    )

    # ── Optional metadata ────────────────────────────────────────────────────
    location: Optional[str] = Field(None, description="Location name")
    region_code: Optional[str] = Field(None, description="Region code (optional)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "disaster_type": "Flood",
                    "rainfall": 245.8,
                    "temperature": 28.5,
                    "humidity": 87.0,
                    "soil_moisture": 72.0,
                    "wind_speed": 35.2,
                    "atmospheric_pressure": 1008.5,
                    "previous_disaster": 1,
                    "location": "Chennai",
                    "region_code": "Zone-4",
                }
            ]
        }
    }


# ===========================================================================
# POST /predict — Response
# ===========================================================================

class ModelProbabilities(SchemaModel):
    """Individual per-model probability scores (0–1)."""

    random_forest: float
    gradient_boosting: float
    svm: float
    stacking: float


class PredictionResponse(SchemaModel):
    """
    Output schema returned by POST /predict.
    All numeric values are pre-rounded by the router.
    """

    predicted_disaster: str = Field(
        ...,
        description="Predicted disaster type or 'No Risk'",
    )
    probability: float = Field(
        ...,
        ge=0,
        le=1,
        description="Stacking meta-learner probability for the positive class",
    )
    dri: float = Field(
        ...,
        ge=0,
        le=1,
        description="Disaster Risk Index: 0.6 × stacking + 0.4 × avg_base",
    )
    risk_level: str = Field(
        ...,
        description="LOW | MODERATE | HIGH",
    )
    confidence: float = Field(
        ...,
        ge=0,
        le=100,
        description="Prediction confidence % derived from model agreement",
    )
    model_probabilities: ModelProbabilities

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "predicted_disaster": "Flood",
                    "probability": 0.87,
                    "dri": 0.82,
                    "risk_level": "HIGH",
                    "confidence": 94.5,
                    "model_probabilities": {
                        "random_forest": 0.89,
                        "gradient_boosting": 0.91,
                        "svm": 0.85,
                        "stacking": 0.87,
                    },
                }
            ]
        }
    }


# ===========================================================================
# GET /history — Records
# ===========================================================================

class HistoryRecord(SchemaModel):
    """
    Single row returned by GET /history or GET /history/{id}.
    id is a string representation of the MySQL integer primary key.
    """

    id: Optional[str] = None                          # str(int PK)
    timestamp: Optional[str] = None                   # ISO-8601 string
    disaster_type: str
    location: Optional[str] = None
    region_code: Optional[str] = None
    input: Dict[str, Any] = Field(default_factory=dict)

    # Prediction output
    predicted_disaster: Optional[str] = None
    probability: Optional[float] = None
    dri: Optional[float] = None
    risk_level: Optional[str] = None
    confidence: Optional[float] = None
    model_probabilities: Dict[str, float] = Field(default_factory=dict)


class HistoryResponse(SchemaModel):
    """Paginated list of prediction history records."""

    total: int = Field(..., description="Total number of matching rows in MySQL")
    page: int
    page_size: int
    records: List[HistoryRecord]


# ===========================================================================
# GET /health
# ===========================================================================

class HealthResponse(SchemaModel):
    """Response from GET /health."""

    status: str                # "healthy" | "degraded"
    api_version: str           # e.g. "v1.0.0"
    db_connected: bool         # True when MySQL SELECT 1 succeeds
    db_engine: str             # "SQLite" | "MySQL"


# ===========================================================================
# GET /models/status
# ===========================================================================

class ModelsStatusResponse(SchemaModel):
    """Response from GET /models/status."""

    status: str                            # "loaded" | "not_loaded" | "error"
    models: List[str] = Field(default_factory=list)   # list of loaded artefact keys
    message: str


# ===========================================================================
# POST /predict/from-raw - External API payload mapping
# ===========================================================================

class RawDataPredictionRequest(SchemaModel):
    """
    Input schema for raw external payload conversion.

    This endpoint accepts provider payloads (weather/seismic APIs), maps them
    to the internal PredictionRequest schema, then runs standard /predict flow.
    """

    disaster_type: str = Field(..., description="'Flood' or 'Earthquake'")
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    location: Optional[str] = None
    region_code: Optional[str] = None
    previous_disaster: int = Field(0, ge=0, le=1)
    weather_payload: Dict[str, Any] = Field(default_factory=dict)
    seismic_payload: Dict[str, Any] = Field(default_factory=dict)


class RawDataPredictionResponse(SchemaModel):
    """Prediction response with mapped feature transparency."""

    mapped_features: Dict[str, Any] = Field(default_factory=dict)
    prediction: PredictionResponse


# ===========================================================================
# POST /predict/live - Lat/Lon real-time provider integration
# ===========================================================================

class LivePredictionRequest(SchemaModel):
    disaster_type: str = Field(..., description="'Flood' or 'Earthquake'")
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    location: Optional[str] = None
    region_code: Optional[str] = None
    previous_disaster: int = Field(0, ge=0, le=1)
    earthquake_radius_km: float = Field(300.0, gt=0, le=1000)
    earthquake_min_magnitude: float = Field(1.0, ge=0, le=10)
    strict_live_sources: bool = Field(
        False,
        description="If true, fail when live provider fetch fails. If false, use fallback values.",
    )


class LivePredictionResponse(SchemaModel):
    source: Dict[str, Any] = Field(default_factory=dict)
    mapped_features: Dict[str, Any] = Field(default_factory=dict)
    prediction: PredictionResponse


# ===========================================================================
# GET /explainability
# ===========================================================================

class FeatureImportanceItem(SchemaModel):
    """A single feature's global importance and signed SHAP-style contribution."""

    feature: str = Field(..., description="Feature name (e.g. 'rainfall')")
    importance_pct: float = Field(
        ...,
        description="Relative importance as a percentage (0–100). All items sum to 100.",
    )
    shap_contribution: float = Field(
        ...,
        description=(
            "Signed SHAP-style contribution. "
            "Positive = increases risk, Negative = decreases risk."
        ),
    )


class ExplainabilityResponse(SchemaModel):
    """Response from GET /explainability."""

    available: bool = Field(
        ...,
        description="True when feature_importance.json exists and was parsed.",
    )
    source: str = Field(
        ...,
        description="Filesystem path from which the data was loaded.",
    )
    feature_importance: List[FeatureImportanceItem] = Field(
        default_factory=list,
        description="Features sorted by importance descending.",
    )
    top_feature: Optional[str] = Field(
        None,
        description="Feature with the highest importance score.",
    )
    total_features: int = Field(0, description="Number of features in the list.")
    note: str = Field("", description="Contextual note about explainability method.")
