# Ensemble-Based Hybrid Disaster Prediction System
## ML Training Pipeline

A production-ready Python ML training pipeline for disaster risk prediction using ensemble learning with stacking meta-optimization.

---

## 🎯 Overview

This pipeline implements a complete machine learning workflow for predicting disaster risk (Flood/Earthquake) using:

- **Base Models**: Random Forest, Gradient Boosting, SVM
- **Meta-Learner**: Stacking Classifier with Logistic Regression
- **Risk Metric**: Disaster Risk Index (DRI)
- **Explainability**: Feature importance and SHAP-style explanations

---

## 📁 Project Structure

```
ml_pipeline/
├── config.py              # Configuration settings
├── data_generator.py      # Synthetic data generation
├── preprocessing.py       # Step 1: Data preprocessing
├── base_models.py         # Step 2: Base model training
├── stacking_model.py      # Step 3: Stacking meta-learner
├── dri_calculator.py      # Step 4: DRI computation
├── explainability.py      # Step 5: Model explainability
├── train_pipeline.py      # Main training orchestrator
├── inference.py           # Production inference module
├── evaluation_plots.py    # Visualization utilities
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── data/                  # Dataset directory
│   └── disaster_dataset.csv
└── models/                # Saved model artifacts
    ├── preprocessing.pkl
    ├── rf_model.pkl
    ├── gb_model.pkl
    ├── svm_model.pkl
    ├── stacking_model.pkl
    ├── metrics.json
    ├── feature_importance.json
    └── plots/
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd ml_pipeline
pip install -r requirements.txt
```

### 2. Run Training Pipeline

```bash
python train_pipeline.py
```

This will:
- Generate synthetic disaster data (if no dataset provided)
- Preprocess data (imputation, normalization)
- Train all base models
- Build stacking meta-learner
- Compute DRI statistics
- Generate explainability reports
- Save all artifacts to `/models/`

### 3. Run Inference

```python
from inference import DisasterPredictionModel

# Load trained model
model = DisasterPredictionModel.load()

# Make prediction
result = model.predict({
    'rainfall': 245.8,
    'temperature': 28.5,
    'humidity': 87,
    'soil_moisture': 72,
    'wind_speed': 45.2,
    'atmospheric_pressure': 1002,
    'previous_disaster': 1
}, disaster_type='Flood', region='Chennai - Zone 4')

print(f"DRI: {result.dri:.4f}")
print(f"Risk Level: {result.risk_level}")
```

---

## 📊 Pipeline Steps

### Step 1: Preprocessing Pipeline
- **Missing Value Handling**: Median imputation
- **Normalization**: Min-Max scaling (0-1)
- **Feature Selection**: Based on disaster type
- **Train-Test Split**: 80/20 stratified

### Step 2: Base Models
| Model | Default Parameters |
|-------|-------------------|
| Random Forest | n_estimators=100, max_depth=10 |
| Gradient Boosting | n_estimators=100, learning_rate=0.1 |
| SVM | kernel='rbf', C=1.0, probability=True |

### Step 3: Stacking Meta-Learner
- **Base Learners**: RF, GB, SVM
- **Meta Learner**: Logistic Regression
- **Cross-Validation**: 5-fold

### Step 4: Disaster Risk Index (DRI)
```
DRI = 0.6 × stacked_probability + 0.4 × average_base_probability
```

**Risk Categories:**
- Low: 0.00 - 0.33
- Moderate: 0.34 - 0.66
- High: 0.67 - 1.00

### Step 5: Explainability
- Global feature importance from Random Forest
- SHAP-style local explanations
- Risk factor identification

### Step 6: Save Artifacts
All models and metrics saved to `/models/` directory.

---

## 📈 Dataset Features

### Flood Prediction
| Feature | Type | Range |
|---------|------|-------|
| rainfall | float | ≥ 0 mm |
| temperature | float | °C |
| humidity | float | 0-100 % |
| soil_moisture | float | 0-100 % |
| wind_speed | float | ≥ 0 km/h |
| atmospheric_pressure | float | 800-1100 hPa |
| previous_disaster | int | 0 or 1 |

### Earthquake Prediction
| Feature | Type | Range |
|---------|------|-------|
| seismic_activity | float | 0-10 Richter |
| temperature | float | °C |
| humidity | float | 0-100 % |
| wind_speed | float | ≥ 0 km/h |
| atmospheric_pressure | float | 800-1100 hPa |
| previous_disaster | int | 0 or 1 |

---

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Model hyperparameters
RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    ...
}

# DRI weights
DRI_WEIGHTS = {
    'stacked_probability_weight': 0.6,
    'average_base_probability_weight': 0.4
}

# Risk thresholds
RISK_THRESHOLDS = {
    'low_max': 0.33,
    'moderate_max': 0.66
}
```

---

## 📋 Command Line Options

```bash
python train_pipeline.py --help

Options:
  --data-path PATH       Path to dataset CSV file
  --disaster-type TYPE   Flood, Earthquake, or all (default: all)
  --no-generate          Don't generate synthetic data if missing
```

---

## 📊 Output Metrics

After training, `metrics.json` contains:

```json
{
  "training_timestamp": "2024-01-15T14:32:05",
  "model_version": "v1.0.0",
  "base_models": {
    "random_forest": {"accuracy": 91.2, "precision": 89.5, ...},
    "gradient_boosting": {"accuracy": 93.1, ...},
    "svm": {"accuracy": 89.4, ...}
  },
  "stacking_model": {
    "accuracy": 95.2,
    "precision": 93.8,
    "recall": 96.1,
    "f1_score": 94.9,
    "roc_auc": 97.3
  },
  "dri_statistics": {
    "mean_dri": 0.52,
    "std_dri": 0.18,
    "risk_distribution": {"LOW": 35, "MODERATE": 42, "HIGH": 23}
  }
}
```

---

## 🔬 Explainability Output

Feature importance example:
```
Feature                   Importance
─────────────────────────────────────
rainfall                  45.2%
soil_moisture             28.7%
humidity                  14.1%
temperature               7.3%
wind_speed                3.2%
atmospheric_pressure      1.5%
```

SHAP-style contribution:
```
rainfall         → +0.42 (↑ Risk)
soil_moisture    → +0.28 (↑ Risk)
humidity         → +0.12 (↑ Risk)
temperature      → -0.05 (↓ Risk)
```

---

## 📦 Model Artifacts

| File | Description | Size |
|------|-------------|------|
| preprocessing.pkl | Scaler + Imputer | ~5 KB |
| rf_model.pkl | Random Forest | ~45 MB |
| gb_model.pkl | Gradient Boosting | ~38 MB |
| svm_model.pkl | Support Vector Machine | ~12 MB |
| stacking_model.pkl | Stacking Ensemble | ~100 MB |
| metrics.json | Evaluation metrics | ~2 KB |
| feature_importance.json | Feature scores | ~1 KB |

---

## 🔗 Integration

### FastAPI Example

```python
from fastapi import FastAPI
from inference import DisasterPredictionModel

app = FastAPI()
model = DisasterPredictionModel.load()

@app.post("/predict")
async def predict(data: dict):
    result = model.predict(data, disaster_type='Flood')
    return result.to_dict()
```

---

## 📝 License

This project is part of an academic research prototype for disaster prediction using ensemble machine learning.

---

## 👥 Authors

Ensemble-Based Hybrid Disaster Prediction System
Final Year AI/ML Engineering Research Project
