from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict

from fastapi import APIRouter, HTTPException, Query

from model_loader import get_model_dir
from schemas import ExplainabilityResponse, FeatureImportanceItem

logger = logging.getLogger("backend.explainability_router")
router = APIRouter(tags=["Explainability"])

_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_FEATURE_IMPORTANCE_PATH = _ROOT / "ml_pipeline" / "models" / "feature_importance.json"
_CUSTOM_PATH = os.environ.get("FEATURE_IMPORTANCE_PATH")
_NEGATIVE_RISK_FEATURES = {"atmospheric_pressure"}


def _normalize_disaster_type(disaster_type: str) -> str:
    value = disaster_type.strip().capitalize()
    if value not in {"Flood", "Earthquake"}:
        raise HTTPException(status_code=400, detail="disaster_type must be 'Flood' or 'Earthquake'.")
    return value


def _resolve_feature_importance_path(disaster_type: str) -> Path:
    if _CUSTOM_PATH:
        return Path(_CUSTOM_PATH)

    models_dir = get_model_dir(disaster_type)
    candidate = models_dir / "feature_importance.json"
    if candidate.exists():
        return candidate

    return _DEFAULT_FEATURE_IMPORTANCE_PATH


def _load_feature_importance(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(
            f"feature_importance.json not found at {path}. "
            "Run the ML training pipeline to generate it."
        )

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if "feature_importance" not in data:
        raise ValueError(
            f"'feature_importance' key missing in {path}. "
            "Re-run the training pipeline to regenerate the file."
        )

    return data


def _build_response(data: Dict[str, object], path: Path, disaster_type: str) -> ExplainabilityResponse:
    raw = data["feature_importance"]
    if not isinstance(raw, dict):
        raise ValueError(f"'feature_importance' must be an object in {path}.")

    if not raw:
        return ExplainabilityResponse(
            available=True,
            source=str(path),
            feature_importance=[],
            top_feature=None,
            total_features=0,
            note=f"{disaster_type} feature importance file is empty. Retrain the pipeline.",
        )

    values = [float(v) for v in raw.values()]
    total = sum(values)
    if total <= 0:
        raise ValueError(f"feature_importance values must sum to a positive number in {path}.")

    normalised = {str(k): (float(v) / total) * 100.0 for k, v in raw.items()}
    sorted_items = sorted(normalised.items(), key=lambda item: item[1], reverse=True)

    items = [
        FeatureImportanceItem(
            feature=feature,
            importance_pct=round(pct, 2),
            shap_contribution=round((round(pct, 2) / 100.0) * (-1.0 if feature in _NEGATIVE_RISK_FEATURES else 1.0), 4),
        )
        for feature, pct in sorted_items
    ]

    has_shap_values = "shap_values" in data
    note = (
        f"{disaster_type} SHAP values loaded from training artefact."
        if has_shap_values
        else (
            f"{disaster_type} SHAP-style contributions approximated from feature importance. "
            "Sign reflects known feature-risk correlation direction."
        )
    )

    return ExplainabilityResponse(
        available=True,
        source=str(path),
        feature_importance=items,
        top_feature=items[0].feature if items else None,
        total_features=len(items),
        note=note,
    )


@router.get(
    "/explainability",
    response_model=ExplainabilityResponse,
    summary="Disaster-type aware feature importance and SHAP-style contributions",
)
def get_explainability(
    disaster_type: str = Query("Flood", description="Flood or Earthquake"),
) -> ExplainabilityResponse:
    resolved_type = _normalize_disaster_type(disaster_type)

    try:
        path = _resolve_feature_importance_path(resolved_type)
        data = _load_feature_importance(path)
        response = _build_response(data, path, resolved_type)
        logger.info(
            "Explainability data served for %s: %d features, top='%s', source='%s'",
            resolved_type,
            response.total_features,
            response.top_feature,
            path,
        )
        return response
    except FileNotFoundError as exc:
        logger.warning("feature_importance.json not found for %s: %s", resolved_type, exc)
        return ExplainabilityResponse(
            available=False,
            source=str(_resolve_feature_importance_path(resolved_type)),
            feature_importance=[],
            top_feature=None,
            total_features=0,
            note=(
                f"{resolved_type} feature importance data is not available. "
                "Run the ML training pipeline to generate it."
            ),
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (ValueError, json.JSONDecodeError) as exc:
        logger.error("Failed to parse feature importance for %s: %s", resolved_type, exc)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse feature importance data: {exc}",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error in /explainability for %s: %s", resolved_type, exc)
        raise HTTPException(status_code=500, detail=f"Internal server error: {exc}") from exc
