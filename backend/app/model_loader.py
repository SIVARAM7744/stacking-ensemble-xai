"""
model_loader.py
---------------
Load and expose disaster-specific model artefacts.

Primary behavior:
- Flood models are loaded from a Flood models directory.
- Earthquake models are loaded from an Earthquake models directory.
- If both are unavailable, fallback to the legacy generic MODELS_DIR.
"""

from __future__ import annotations

import logging
import os
import re
import importlib
import sys
import types
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

logger = logging.getLogger("backend.model_loader")

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
_BACKEND_DIR = _THIS_DIR.parent

load_dotenv(_BACKEND_DIR / ".env", override=False)
load_dotenv(_PROJECT_ROOT / ".env", override=False)

_DEFAULT_MODELS_DIR = _PROJECT_ROOT / "ml_pipeline" / "models"
_DEFAULT_FLOOD_MODELS_DIR = _PROJECT_ROOT / "ml_pipeline" / "models_tuned_v4_flood"
_DEFAULT_EARTHQUAKE_MODELS_DIR = _PROJECT_ROOT / "ml_pipeline" / "models_tuned_v3_earthquake"

MODELS_DIR: Path = Path(os.getenv("MODELS_DIR", str(_DEFAULT_MODELS_DIR)))
FLOOD_MODELS_DIR: Path = Path(os.getenv("FLOOD_MODELS_DIR", str(_DEFAULT_FLOOD_MODELS_DIR)))
EARTHQUAKE_MODELS_DIR: Path = Path(
    os.getenv("EARTHQUAKE_MODELS_DIR", str(_DEFAULT_EARTHQUAKE_MODELS_DIR))
)

_ARTEFACT_FILES: Dict[str, str] = {
    "preprocessing": "preprocessing.pkl",
    "rf_model": "rf_model.pkl",
    "gb_model": "gb_model.pkl",
    "svm_model": "svm_model.pkl",
    "stacking_model": "stacking_model.pkl",
}

_models_by_type: Dict[str, Dict[str, Any]] = {}
_model_dirs_by_type: Dict[str, Path] = {}


def _install_pickle_compat_shims() -> None:
    """
    Install import aliases for older/newer sklearn pickle module paths.
    This prevents ModuleNotFoundError during joblib.load for version-mismatched artefacts.
    """
    if "_loss" in sys.modules:
        return
    target = None
    for mod_name in ("sklearn._loss.loss", "sklearn._loss"):
        try:
            target = importlib.import_module(mod_name)
            break
        except Exception:
            continue
    if target is None:
        return

    # Emulate both module forms that may appear in pickles.
    pkg = types.ModuleType("_loss")
    for name in dir(target):
        try:
            setattr(pkg, name, getattr(target, name))
        except Exception:
            pass
    pkg._loss = target

    sys.modules["_loss"] = pkg
    sys.modules["_loss._loss"] = target

    # numpy RNG pickle compatibility (newer pickles may pass class objects)
    try:
        from numpy.random import _pickle as np_pickle  # type: ignore

        if not getattr(np_pickle, "_compat_ctor_patched", False):
            _orig_ctor = np_pickle.__bit_generator_ctor

            def _compat_ctor(bitgen_name: Any = "MT19937") -> Any:
                name = bitgen_name
                if isinstance(bitgen_name, type):
                    name = bitgen_name.__name__
                elif hasattr(bitgen_name, "__name__"):
                    name = getattr(bitgen_name, "__name__")
                return _orig_ctor(name)

            np_pickle.__bit_generator_ctor = _compat_ctor  # type: ignore[assignment]
            np_pickle._compat_ctor_patched = True  # type: ignore[attr-defined]
    except Exception:
        pass

    # numpy module path aliases (newer pickle paths -> current numpy paths)
    try:
        numpy_core = importlib.import_module("numpy.core")
        sys.modules.setdefault("numpy._core", numpy_core)

        for old_mod, new_mod in (
            ("numpy._core.numeric", "numpy.core.numeric"),
            ("numpy._core.multiarray", "numpy.core.multiarray"),
            ("numpy._core.umath", "numpy.core.umath"),
            ("numpy._core._multiarray_umath", "numpy.core._multiarray_umath"),
        ):
            if old_mod not in sys.modules:
                try:
                    sys.modules[old_mod] = importlib.import_module(new_mod)
                except Exception:
                    pass
    except Exception:
        pass


def _has_all_artefacts(models_dir: Path) -> bool:
    return models_dir.exists() and all((models_dir / name).exists() for name in _ARTEFACT_FILES.values())


def _iter_model_dirs(base: Path) -> Iterable[Path]:
    if not base.exists():
        return []
    return [p for p in base.iterdir() if p.is_dir()]


def _tuned_version(folder_name: str, suffix: str) -> int:
    # e.g. models_tuned_v4_flood -> 4
    m = re.match(rf"^models_tuned_v(\d+)_{suffix}$", folder_name.strip().lower())
    return int(m.group(1)) if m else -1


def _discover_disaster_dir(disaster_type: str) -> Optional[Path]:
    """
    Find the best available artefact directory for a disaster type.
    Priority:
    1. Highest version models_tuned_v*_{type}
    2. models_{type}
    3. generic models
    """
    suffix = "flood" if disaster_type == "Flood" else "earthquake"
    ml_root = _PROJECT_ROOT / "ml_pipeline"
    candidates = [p for p in _iter_model_dirs(ml_root) if p.name.lower().endswith(suffix)]

    tuned = sorted(
        [p for p in candidates if _tuned_version(p.name, suffix) >= 0 and _has_all_artefacts(p)],
        key=lambda p: _tuned_version(p.name, suffix),
        reverse=True,
    )
    if tuned:
        return tuned[0]

    named = ml_root / f"models_{suffix}"
    if _has_all_artefacts(named):
        return named

    if _has_all_artefacts(MODELS_DIR):
        return MODELS_DIR

    return None


class PreprocessingAdapter:
    """Adapter for preprocessing artefacts saved as plain dict payloads."""

    def __init__(self, payload: Dict[str, Any]) -> None:
        self.imputer = payload.get("imputer")
        self.scaler = payload.get("scaler")
        self.feature_columns = payload.get("feature_columns") or []

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.imputer is None or self.scaler is None:
            raise RuntimeError("Invalid preprocessing artefact: imputer/scaler missing.")
        if not self.feature_columns:
            raise RuntimeError("Invalid preprocessing artefact: feature_columns missing.")

        aligned = X.copy()
        for col in self.feature_columns:
            if col not in aligned.columns:
                aligned[col] = np.nan
        aligned = aligned[self.feature_columns]
        aligned = aligned.apply(pd.to_numeric, errors="coerce")

        imputed = self.imputer.transform(aligned)
        scaled = self.scaler.transform(imputed)
        return pd.DataFrame(scaled, columns=self.feature_columns, index=aligned.index)


def _normalize_loaded_artefact(key: str, artefact: Any) -> Any:
    if key == "preprocessing":
        if hasattr(artefact, "transform"):
            return artefact
        if isinstance(artefact, dict):
            return PreprocessingAdapter(artefact)
        raise RuntimeError("Unsupported preprocessing artefact format.")

    if key == "stacking_model":
        if hasattr(artefact, "predict_proba"):
            return artefact
        if isinstance(artefact, dict) and "stacking_model" in artefact:
            return artefact["stacking_model"]
        raise RuntimeError("Unsupported stacking_model artefact format.")

    return artefact


def _load_models_from_dir(models_dir: Path) -> Dict[str, Any]:
    logger.info("Loading model artefacts from: %s", models_dir)
    _install_pickle_compat_shims()

    if not models_dir.exists():
        raise FileNotFoundError(
            f"Model artifacts not found. Please run training first. Expected directory: {models_dir}"
        )

    missing: list[str] = []
    for filename in _ARTEFACT_FILES.values():
        if not (models_dir / filename).exists():
            missing.append(filename)
    if missing:
        raise FileNotFoundError(
            f"Model artifacts missing in {models_dir}: {', '.join(missing)}"
        )

    loaded: Dict[str, Any] = {}
    for key, filename in _ARTEFACT_FILES.items():
        raw = joblib.load(models_dir / filename)
        loaded[key] = _normalize_loaded_artefact(key, raw)
    return loaded


def load_models() -> Dict[str, Dict[str, Any]]:
    """Load model sets for Flood and Earthquake."""
    global _models_by_type, _model_dirs_by_type

    loaded_by_type: Dict[str, Dict[str, Any]] = {}
    loaded_dirs_by_type: Dict[str, Path] = {}
    warnings: list[str] = []

    flood_dir = FLOOD_MODELS_DIR
    if not _has_all_artefacts(flood_dir):
        discovered = _discover_disaster_dir("Flood")
        if discovered:
            warnings.append(f"Flood configured dir unavailable; auto-discovered: {discovered}")
            flood_dir = discovered
    try:
        loaded_by_type["Flood"] = _load_models_from_dir(flood_dir)
        loaded_dirs_by_type["Flood"] = flood_dir
    except Exception as exc:
        warnings.append(f"Flood load failed: {exc}")

    eq_dir = EARTHQUAKE_MODELS_DIR
    if not _has_all_artefacts(eq_dir):
        discovered = _discover_disaster_dir("Earthquake")
        if discovered:
            warnings.append(f"Earthquake configured dir unavailable; auto-discovered: {discovered}")
            eq_dir = discovered
    try:
        loaded_by_type["Earthquake"] = _load_models_from_dir(eq_dir)
        loaded_dirs_by_type["Earthquake"] = eq_dir
    except Exception as exc:
        warnings.append(f"Earthquake load failed: {exc}")

    if "Flood" not in loaded_by_type or "Earthquake" not in loaded_by_type:
        if _has_all_artefacts(MODELS_DIR):
            logger.warning("Using fallback generic MODELS_DIR=%s for missing disaster model sets", MODELS_DIR)
            generic = _load_models_from_dir(MODELS_DIR)
            loaded_by_type.setdefault("Flood", generic)
            loaded_by_type.setdefault("Earthquake", generic)
            loaded_dirs_by_type.setdefault("Flood", MODELS_DIR)
            loaded_dirs_by_type.setdefault("Earthquake", MODELS_DIR)

    if not loaded_by_type:
        raise FileNotFoundError(
            "No valid model artefacts found for Flood/Earthquake. "
            "Expected tuned folders (models_tuned_v*), models_flood/models_earthquake, or generic models."
        )

    _models_by_type = loaded_by_type
    _model_dirs_by_type = loaded_dirs_by_type

    if warnings:
        logger.warning("Model load warnings: %s", " | ".join(warnings))

    logger.info("Loaded model sets for: %s", ", ".join(sorted(_models_by_type.keys())))
    return _models_by_type


def get_models(disaster_type: str) -> Dict[str, Any]:
    """Return loaded models for one disaster type."""
    if not _models_by_type:
        raise RuntimeError("Models are not loaded.")

    key = disaster_type.strip().capitalize()
    models = _models_by_type.get(key)
    if models is None:
        raise RuntimeError(
            f"Models for disaster_type='{disaster_type}' are not loaded. "
            f"Available: {', '.join(sorted(_models_by_type.keys()))}"
        )
    return models


def is_models_loaded() -> bool:
    return bool(_models_by_type)


def get_loaded_model_types() -> list[str]:
    return sorted(_models_by_type.keys())


def get_model_dir(disaster_type: str) -> Path:
    """Return the resolved artefact directory for one loaded disaster type."""
    key = disaster_type.strip().capitalize()
    models_dir = _model_dirs_by_type.get(key)
    if models_dir is None:
        discovered = _discover_disaster_dir(key)
        if discovered is not None:
            return discovered
        raise RuntimeError(
            f"Model directory for disaster_type='{disaster_type}' is not loaded. "
            f"Available: {', '.join(sorted(_model_dirs_by_type.keys()))}"
        )
    return models_dir
