"""
predict_router.py
-----------------
Prediction endpoints:
- POST /predict
- POST /predict/from-raw
- GET  /models/status
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import httpx
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from database import get_db
from model_loader import get_loaded_model_types, get_models, is_models_loaded
from models import PredictionRecord
from schemas import (
    ModelProbabilities,
    ModelsStatusResponse,
    PredictionRequest,
    PredictionResponse,
    LivePredictionRequest,
    LivePredictionResponse,
    RawDataPredictionRequest,
    RawDataPredictionResponse,
)

logger = logging.getLogger("backend.predict_router")
router = APIRouter(tags=["Prediction"])

FLOOD_FEATURES: List[str] = [
    "rainfall",
    "temperature",
    "humidity",
    "soil_moisture",
    "wind_speed",
    "atmospheric_pressure",
    "previous_disaster",
]

EARTHQUAKE_FEATURES: List[str] = [
    "seismic_activity",
    "temperature",
    "humidity",
    "wind_speed",
    "atmospheric_pressure",
    "previous_disaster",
]

_OPTIONAL_DEFAULTS: Dict[str, float] = {
    "temperature": 25.0,
    "humidity": 60.0,
    "wind_speed": 10.0,
    "atmospheric_pressure": 1013.0,
}


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _get_risk_level(dri: float) -> str:
    if dri <= 0.33:
        return "LOW"
    if dri <= 0.66:
        return "MODERATE"
    return "HIGH"


def _calculate_confidence(probabilities: Dict[str, float]) -> float:
    values = list(probabilities.values())
    mean_p = float(np.mean(values))
    std_p = float(np.std(values))
    agreement = max(0.0, 1.0 - std_p * 2.0)
    decisiveness = abs(mean_p - 0.5) * 2.0
    confidence = (agreement * 0.6 + decisiveness * 0.4) * 100.0
    return round(max(0.0, min(100.0, confidence)), 1)


def _build_feature_vector(request: PredictionRequest, features: List[str]) -> pd.DataFrame:
    req_dict: Dict[str, Any] = request.model_dump()
    row: Dict[str, float] = {}
    for col in features:
        value = req_dict.get(col)
        if value is None:
            value = _OPTIONAL_DEFAULTS.get(col, 0.0)
        row[col] = float(value)
    return pd.DataFrame([row], columns=features)


def _pick_value(payload: Dict[str, Any], keys: List[str], default: float = 0.0) -> float:
    lowered = {str(k).strip().lower(): v for k, v in payload.items()}
    for key in keys:
        raw = lowered.get(key.lower())
        if raw is None:
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return float(default)


def _clamp_float(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _sanitize_mapped_features(mapped: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure mapped values respect PredictionRequest schema bounds.
    Prevents 422 errors from provider payload quirks.
    """
    out = dict(mapped)

    if out.get("humidity") is not None:
        out["humidity"] = _clamp_float(out["humidity"], 0.0, 100.0)
    if out.get("soil_moisture") is not None:
        out["soil_moisture"] = _clamp_float(out["soil_moisture"], 0.0, 100.0)
    if out.get("seismic_activity") is not None:
        out["seismic_activity"] = _clamp_float(out["seismic_activity"], 0.0, 10.0)
    if out.get("atmospheric_pressure") is not None:
        out["atmospheric_pressure"] = _clamp_float(out["atmospheric_pressure"], 800.0, 1100.0)
    if out.get("wind_speed") is not None:
        out["wind_speed"] = max(0.0, float(out["wind_speed"]))
    if out.get("rainfall") is not None:
        out["rainfall"] = max(0.0, float(out["rainfall"]))

    prev = out.get("previous_disaster", 0)
    out["previous_disaster"] = 1 if int(prev) == 1 else 0

    return out


def _map_raw_to_prediction_request(raw: RawDataPredictionRequest) -> tuple[PredictionRequest, Dict[str, Any]]:
    disaster_type = raw.disaster_type.strip().capitalize()
    weather = raw.weather_payload or {}
    seismic = raw.seismic_payload or {}

    base = {
        "disaster_type": disaster_type,
        "temperature": _pick_value(weather, ["temperature", "temp", "temperature_c", "temp_c"], 25.0),
        "humidity": _pick_value(weather, ["humidity", "relative_humidity"], 60.0),
        "wind_speed": _pick_value(weather, ["wind_speed", "wind_kph", "wind_mps"], 10.0),
        "atmospheric_pressure": _pick_value(
            weather,
            ["pressure", "pressure_hpa", "atmospheric_pressure"],
            1013.0,
        ),
        "previous_disaster": raw.previous_disaster,
        "location": raw.location,
        "region_code": raw.region_code,
    }

    if disaster_type == "Flood":
        mapped = {
            **base,
            "rainfall": _pick_value(
                weather,
                ["rainfall", "rain", "precip_mm", "rainfall_mm", "total_precipitation"],
                0.0,
            ),
            "soil_moisture": _pick_value(
                weather,
                ["soil_moisture", "soil_moisture_pct", "soil_moisture_percent"],
                40.0,
            ),
            "seismic_activity": None,
        }
    else:
        mapped = {
            **base,
            "seismic_activity": _pick_value(
                seismic,
                ["seismic_activity", "magnitude", "richter", "quake_magnitude"],
                0.0,
            ),
            "rainfall": None,
            "soil_moisture": None,
        }

    mapped = _sanitize_mapped_features(mapped)
    return PredictionRequest(**mapped), mapped


def _fetch_open_meteo_snapshot(latitude: float, longitude: float) -> Dict[str, Any]:
    """
    Fetch a single weather snapshot from Open-Meteo.

    Uses current weather + hourly precipitation/soil moisture.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m",
        "hourly": "precipitation,soil_moisture_0_to_1cm",
        "timezone": "auto",
        "forecast_days": 1,
    }
    with httpx.Client(timeout=12.0) as client:
        res = client.get(url, params=params)
        res.raise_for_status()
        data = res.json()

    current = data.get("current", {})
    hourly = data.get("hourly", {})
    precip = 0.0
    soil_moisture_pct = 40.0

    precip_values = hourly.get("precipitation") or []
    if precip_values:
        try:
            precip = float(max(precip_values))
        except Exception:
            precip = 0.0

    sm_values = hourly.get("soil_moisture_0_to_1cm") or []
    if sm_values:
        try:
            soil_moisture_pct = float(sm_values[-1]) * 100.0
        except Exception:
            soil_moisture_pct = 40.0

    return {
        "temperature": current.get("temperature_2m"),
        "humidity": current.get("relative_humidity_2m"),
        "pressure_hpa": current.get("surface_pressure"),
        "wind_kph": current.get("wind_speed_10m"),
        "rainfall_mm": precip,
        "soil_moisture_pct": soil_moisture_pct,
    }


def _fetch_usgs_recent_magnitude(
    latitude: float,
    longitude: float,
    radius_km: float,
    min_magnitude: float,
) -> Dict[str, Any]:
    """Fetch latest nearby earthquake event from USGS and extract magnitude."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=30)
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format": "geojson",
        "latitude": latitude,
        "longitude": longitude,
        "maxradiuskm": radius_km,
        "minmagnitude": min_magnitude,
        "starttime": start.date().isoformat(),
        "endtime": end.date().isoformat(),
        "orderby": "time",
        "limit": 1,
    }
    with httpx.Client(timeout=12.0) as client:
        res = client.get(url, params=params)
        res.raise_for_status()
        data = res.json()

    features = data.get("features") or []
    if not features:
        return {"magnitude": 0.0, "event_count": 0}

    props = features[0].get("properties", {})
    magnitude = props.get("mag")
    try:
        magnitude = float(magnitude)
    except Exception:
        magnitude = 0.0

    return {"magnitude": magnitude, "event_count": len(features)}


def _fallback_weather_snapshot(latitude: float, longitude: float) -> Dict[str, Any]:
    """
    Deterministic fallback weather snapshot for offline/degraded mode.

    Keeps API functional when external providers are unavailable.
    """
    abs_lat = abs(latitude)
    abs_lon = abs(longitude)
    rainfall = _clamp(15.0 + (abs_lat % 25) * 1.4 - (abs_lon % 10) * 0.6, 0.0, 350.0)
    temperature = _clamp(34.0 - abs_lat * 0.25, -10.0, 48.0)
    humidity = _clamp(50.0 + ((abs_lat + abs_lon) % 35), 15.0, 98.0)
    pressure_hpa = _clamp(1014.0 - (abs_lat % 8), 930.0, 1050.0)
    wind_kph = _clamp(6.0 + (abs_lon % 22) * 0.35, 0.0, 140.0)
    soil_moisture_pct = _clamp(30.0 + rainfall * 0.45, 5.0, 95.0)
    return {
        "temperature": round(temperature, 2),
        "humidity": round(humidity, 2),
        "pressure_hpa": round(pressure_hpa, 2),
        "wind_kph": round(wind_kph, 2),
        "rainfall_mm": round(rainfall, 2),
        "soil_moisture_pct": round(soil_moisture_pct, 2),
    }


def _fallback_seismic_snapshot(
    latitude: float,
    longitude: float,
    min_magnitude: float,
) -> Dict[str, Any]:
    """
    Deterministic fallback seismic snapshot for offline/degraded mode.
    """
    baseline = (abs(latitude) + abs(longitude)) / 120.0
    magnitude = _clamp(min_magnitude * 0.8 + baseline, 0.0, 8.8)
    return {"magnitude": round(magnitude, 2), "event_count": 0}


def _persist_record(
    db: Session,
    *,
    disaster_type: str,
    request: PredictionRequest,
    features: List[str],
    response: PredictionResponse,
    model_probs: ModelProbabilities,
) -> None:
    req_dict = request.model_dump()
    record = PredictionRecord.from_prediction(
        disaster_type=disaster_type,
        location=req_dict.get("location"),
        region_code=req_dict.get("region_code"),
        input_features={k: req_dict.get(k) for k in features},
        output={
            "predicted_disaster": response.predicted_disaster,
            "probability": response.probability,
            "dri": response.dri,
            "risk_level": response.risk_level,
            "confidence": response.confidence,
            "model_probabilities": {
                "random_forest": model_probs.random_forest,
                "gradient_boosting": model_probs.gradient_boosting,
                "svm": model_probs.svm,
                "stacking": model_probs.stacking,
            },
        },
    )
    db.add(record)
    db.commit()
    db.refresh(record)


@router.post("/predict", response_model=PredictionResponse, summary="Run ensemble disaster risk prediction")
def predict_disaster(
    request: PredictionRequest,
    db: Session = Depends(get_db),
) -> PredictionResponse:
    if not is_models_loaded():
        raise HTTPException(status_code=503, detail="ML models are not loaded.")

    disaster_type = request.disaster_type.strip().capitalize()
    if disaster_type not in ("Flood", "Earthquake"):
        raise HTTPException(status_code=400, detail="disaster_type must be 'Flood' or 'Earthquake'.")

    try:
        models = get_models(disaster_type)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if disaster_type == "Flood":
        features = FLOOD_FEATURES
        if request.rainfall is None:
            raise HTTPException(status_code=400, detail="'rainfall' is required for Flood prediction.")
        if request.soil_moisture is None:
            raise HTTPException(status_code=400, detail="'soil_moisture' is required for Flood prediction.")
    else:
        features = EARTHQUAKE_FEATURES
        if request.seismic_activity is None:
            raise HTTPException(status_code=400, detail="'seismic_activity' is required for Earthquake prediction.")

    try:
        feature_df = _build_feature_vector(request, features)
        X_processed: np.ndarray = models["preprocessing"].transform(feature_df)
    except Exception as exc:
        logger.exception("Preprocessing failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Preprocessing pipeline error: {exc}") from exc

    try:
        rf_prob = float(models["rf_model"].predict_proba(X_processed)[0][1])
        gb_prob = float(models["gb_model"].predict_proba(X_processed)[0][1])
        svm_prob = float(models["svm_model"].predict_proba(X_processed)[0][1])
        stacking_prob = float(models["stacking_model"].predict_proba(X_processed)[0][1])
    except Exception as exc:
        logger.exception("Model inference failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Model inference error: {exc}") from exc

    avg_base = (rf_prob + gb_prob + svm_prob) / 3.0
    dri = 0.6 * stacking_prob + 0.4 * avg_base
    risk_level = _get_risk_level(dri)
    predicted_disaster = disaster_type if stacking_prob > 0.5 else "No Risk"

    raw_probs = {
        "random_forest": rf_prob,
        "gradient_boosting": gb_prob,
        "svm": svm_prob,
        "stacking": stacking_prob,
    }
    confidence = _calculate_confidence(raw_probs)

    model_probs = ModelProbabilities(
        random_forest=round(rf_prob, 4),
        gradient_boosting=round(gb_prob, 4),
        svm=round(svm_prob, 4),
        stacking=round(stacking_prob, 4),
    )
    response = PredictionResponse(
        predicted_disaster=predicted_disaster,
        probability=round(stacking_prob, 4),
        dri=round(dri, 4),
        risk_level=risk_level,
        confidence=confidence,
        model_probabilities=model_probs,
    )

    try:
        _persist_record(
            db,
            disaster_type=disaster_type,
            request=request,
            features=features,
            response=response,
            model_probs=model_probs,
        )
    except Exception as exc:
        logger.warning("Failed to persist prediction (inference returned): %s", exc)

    return response


@router.post(
    "/predict/from-raw",
    response_model=RawDataPredictionResponse,
    summary="Map external weather/seismic payloads and run prediction",
)
def predict_from_raw_data(
    request: RawDataPredictionRequest,
    db: Session = Depends(get_db),
) -> RawDataPredictionResponse:
    disaster_type = request.disaster_type.strip().capitalize()
    if disaster_type not in ("Flood", "Earthquake"):
        raise HTTPException(status_code=400, detail="disaster_type must be 'Flood' or 'Earthquake'.")

    mapped_request, mapped_features = _map_raw_to_prediction_request(request)
    prediction = predict_disaster(mapped_request, db)
    return RawDataPredictionResponse(mapped_features=mapped_features, prediction=prediction)


@router.post(
    "/predict/live",
    response_model=LivePredictionResponse,
    summary="Fetch live external data by lat/lon and run prediction",
)
def predict_from_live_sources(
    request: LivePredictionRequest,
    db: Session = Depends(get_db),
) -> LivePredictionResponse:
    disaster_type = request.disaster_type.strip().capitalize()
    if disaster_type not in ("Flood", "Earthquake"):
        raise HTTPException(status_code=400, detail="disaster_type must be 'Flood' or 'Earthquake'.")

    warnings: List[str] = []
    weather_mode = "live"
    seismic_mode = "live"

    try:
        weather_data = _fetch_open_meteo_snapshot(request.latitude, request.longitude)
    except Exception as exc:
        if request.strict_live_sources:
            raise HTTPException(status_code=502, detail=f"Weather provider fetch failed: {exc}") from exc
        warnings.append(f"Weather provider unavailable, fallback used: {exc}")
        weather_mode = "fallback"
        weather_data = _fallback_weather_snapshot(request.latitude, request.longitude)

    seismic_data: Dict[str, Any] = {}
    if disaster_type == "Earthquake":
        try:
            seismic_data = _fetch_usgs_recent_magnitude(
                request.latitude,
                request.longitude,
                request.earthquake_radius_km,
                request.earthquake_min_magnitude,
            )
        except Exception as exc:
            if request.strict_live_sources:
                raise HTTPException(status_code=502, detail=f"Seismic provider fetch failed: {exc}") from exc
            warnings.append(f"Seismic provider unavailable, fallback used: {exc}")
            seismic_mode = "fallback"
            seismic_data = _fallback_seismic_snapshot(
                request.latitude,
                request.longitude,
                request.earthquake_min_magnitude,
            )

    raw_request = RawDataPredictionRequest(
        disaster_type=disaster_type,
        latitude=request.latitude,
        longitude=request.longitude,
        location=request.location,
        region_code=request.region_code,
        previous_disaster=request.previous_disaster,
        weather_payload=weather_data,
        seismic_payload=seismic_data,
    )
    mapped_request, mapped_features = _map_raw_to_prediction_request(raw_request)
    prediction = predict_disaster(mapped_request, db)

    return LivePredictionResponse(
        source={
            "weather_provider": "open-meteo",
            "seismic_provider": "usgs" if disaster_type == "Earthquake" else None,
            "weather_mode": weather_mode,
            "seismic_mode": seismic_mode if disaster_type == "Earthquake" else None,
            "warnings": warnings,
            "latitude": request.latitude,
            "longitude": request.longitude,
        },
        mapped_features=mapped_features,
        prediction=prediction,
    )


@router.get("/models/status", response_model=ModelsStatusResponse, summary="Model artefact loading status")
def get_model_status() -> ModelsStatusResponse:
    if not is_models_loaded():
        return ModelsStatusResponse(
            status="not_loaded",
            models=[],
            message="Models have not been loaded. Run the training pipeline first.",
        )
    try:
        loaded_types = get_loaded_model_types()
        return ModelsStatusResponse(
            status="loaded",
            models=loaded_types,
            message=f"Model artefacts loaded for disaster types: {', '.join(loaded_types)}.",
        )
    except RuntimeError as exc:
        return ModelsStatusResponse(status="error", models=[], message=str(exc))
