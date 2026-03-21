"""
Configuration settings for the Ensemble-Based Hybrid Disaster Prediction System
"""

import os
from pathlib import Path

# =============================================================================
# DIRECTORY PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# =============================================================================
# MODEL ARTIFACTS
# =============================================================================
PREPROCESSING_PATH = MODELS_DIR / "preprocessing.pkl"
RF_MODEL_PATH = MODELS_DIR / "rf_model.pkl"
GB_MODEL_PATH = MODELS_DIR / "gb_model.pkl"
SVM_MODEL_PATH = MODELS_DIR / "svm_model.pkl"
STACKING_MODEL_PATH = MODELS_DIR / "stacking_model.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"
FEATURE_IMPORTANCE_PATH = MODELS_DIR / "feature_importance.json"

# =============================================================================
# DATASET COLUMNS
# =============================================================================
# Common features for all disaster types
COMMON_FEATURES = [
    'temperature',
    'humidity',
    'wind_speed',
    'atmospheric_pressure',
    'previous_disaster'
]

# Flood-specific features
FLOOD_FEATURES = COMMON_FEATURES + [
    'rainfall',
    'soil_moisture'
]

# Earthquake-specific features
EARTHQUAKE_FEATURES = COMMON_FEATURES + [
    'seismic_activity'
]

# Target column
TARGET_COLUMN = 'target'

# Disaster type column
DISASTER_TYPE_COLUMN = 'disaster_type'

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42,
    # Keep single-process for restricted Windows environments.
    'n_jobs': 1
}

GRADIENT_BOOSTING_PARAMS = {
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.1,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42
}

SVM_PARAMS = {
    'C': 1.0,
    'kernel': 'rbf',
    'gamma': 'scale',
    'probability': True,
    'random_state': 42
}

META_LEARNER_PARAMS = {
    'C': 1.0,
    'solver': 'lbfgs',
    'max_iter': 1000,
    'random_state': 42
}

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# =============================================================================
# DRI CONFIGURATION
# =============================================================================
DRI_WEIGHTS = {
    'stacked_probability_weight': 0.6,
    'average_base_probability_weight': 0.4
}

# Meta-learner base model weights (for display purposes)
META_LEARNER_WEIGHTS = {
    'rf': 0.32,
    'gb': 0.38,
    'svm': 0.30
}

# Risk level thresholds
RISK_THRESHOLDS = {
    'low_max': 0.33,
    'moderate_max': 0.66
}

# =============================================================================
# VALIDATION RANGES
# =============================================================================
FEATURE_VALIDATION = {
    'rainfall': {'min': 0, 'max': float('inf')},
    'humidity': {'min': 0, 'max': 100},
    'soil_moisture': {'min': 0, 'max': 100},
    'wind_speed': {'min': 0, 'max': float('inf')},
    'seismic_activity': {'min': 0, 'max': 10},
    'atmospheric_pressure': {'min': 800, 'max': 1100},
    'temperature': {'min': -50, 'max': 60},
    'previous_disaster': {'min': 0, 'max': 1}
}
