"""
Inference Module for Ensemble-Based Hybrid Disaster Prediction System

Provides reusable inference-ready structure for real-time predictions.

Usage:
    from inference import DisasterPredictionModel
    
    model = DisasterPredictionModel.load()
    result = model.predict(input_data)
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from datetime import datetime

import config
from preprocessing import DisasterPreprocessor
from dri_calculator import DRICalculator, DRIPrediction, RiskLevel


@dataclass
class PredictionResult:
    """
    Complete prediction result from the ensemble model.
    """
    timestamp: str
    disaster_type: str
    predicted_class: int
    probability: float
    dri: float
    risk_level: str
    confidence: float
    region: Optional[str]
    base_model_probabilities: Dict[str, float]
    meta_learner_weights: Dict[str, float]
    dominant_features: List[str]
    explanation: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'timestamp': self.timestamp,
            'disaster_type': self.disaster_type,
            'predicted_class': self.predicted_class,
            'probability': round(self.probability * 100, 2),
            'dri': round(self.dri, 4),
            'risk_level': self.risk_level,
            'confidence': round(self.confidence, 2),
            'region': self.region,
            'base_model_probabilities': {
                k: round(v * 100, 2) for k, v in self.base_model_probabilities.items()
            },
            'meta_learner_weights': self.meta_learner_weights,
            'dominant_features': self.dominant_features,
            'explanation': self.explanation
        }


class DisasterPredictionModel:
    """
    Production-ready inference model for disaster prediction.
    Loads all trained components and provides prediction interface.
    """
    
    def __init__(self):
        self.preprocessor = None
        self.rf_model = None
        self.gb_model = None
        self.svm_model = None
        self.stacking_model = None
        self.dri_calculator = None
        self.feature_importance = None
        self.is_loaded = False
        self.model_version = "v1.0.0"
    
    @classmethod
    def load(cls, models_dir: str = None) -> 'DisasterPredictionModel':
        """
        Load all model components from disk.
        
        Parameters:
        -----------
        models_dir : str, optional
            Directory containing model artifacts
            
        Returns:
        --------
        DisasterPredictionModel
            Loaded model ready for inference
        """
        model = cls()
        
        if models_dir:
            base_path = Path(models_dir)
        else:
            base_path = config.MODELS_DIR
        
        print("Loading Disaster Prediction Model...")
        
        # Load preprocessor
        model.preprocessor = DisasterPreprocessor.load(
            base_path / "preprocessing.pkl"
        )
        print("  ✓ Preprocessor loaded")
        
        # Load base models
        model.rf_model = joblib.load(base_path / "rf_model.pkl")
        print("  ✓ Random Forest loaded")
        
        model.gb_model = joblib.load(base_path / "gb_model.pkl")
        print("  ✓ Gradient Boosting loaded")
        
        model.svm_model = joblib.load(base_path / "svm_model.pkl")
        print("  ✓ SVM loaded")
        
        # Load stacking model
        stacking_data = joblib.load(base_path / "stacking_model.pkl")
        model.stacking_model = stacking_data['stacking_model']
        print("  ✓ Stacking Meta-Learner loaded")
        
        # Initialize DRI calculator
        model.dri_calculator = DRICalculator()
        
        # Load feature importance if available
        try:
            import json
            with open(base_path / "feature_importance.json", 'r') as f:
                model.feature_importance = json.load(f)
            print("  ✓ Feature importance loaded")
        except:
            model.feature_importance = None
        
        model.is_loaded = True
        print("Model ready for inference!\n")
        
        return model
    
    def _preprocess_input(
        self,
        input_data: Union[Dict, pd.DataFrame],
        disaster_type: str = 'Flood'
    ) -> pd.DataFrame:
        """
        Preprocess input data for prediction.
        
        Parameters:
        -----------
        input_data : Union[Dict, pd.DataFrame]
            Input features as dict or DataFrame
        disaster_type : str
            Type of disaster for feature selection
            
        Returns:
        --------
        pd.DataFrame
            Preprocessed features
        """
        # Convert dict to DataFrame
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
        
        # Transform using preprocessor
        X = self.preprocessor.transform(df)
        
        return X
    
    def _get_base_model_predictions(
        self,
        X: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Get predictions from each base model.
        """
        return {
            'rf': float(self.rf_model.predict_proba(X)[0, 1]),
            'gb': float(self.gb_model.predict_proba(X)[0, 1]),
            'svm': float(self.svm_model.predict_proba(X)[0, 1])
        }
    
    def _generate_explanation(
        self,
        input_data: Dict,
        disaster_type: str,
        dominant_features: List[str],
        risk_level: str
    ) -> str:
        """
        Generate human-readable explanation for the prediction.
        """
        if disaster_type == 'Flood':
            rainfall = input_data.get('rainfall', 'N/A')
            humidity = input_data.get('humidity', 'N/A')
            soil_moisture = input_data.get('soil_moisture', 'N/A')
            
            if risk_level == 'HIGH':
                return (
                    f"High flood risk detected. Rainfall at {rainfall}mm combined with "
                    f"soil moisture at {soil_moisture}% indicates saturated conditions. "
                    f"Humidity levels ({humidity}%) suggest continued precipitation probability. "
                    f"The ensemble model identifies {dominant_features[0] if dominant_features else 'environmental factors'} "
                    f"as the primary risk contributor."
                )
            elif risk_level == 'MODERATE':
                return (
                    f"Moderate flood risk. Environmental conditions show elevated values in "
                    f"{', '.join(dominant_features[:2]) if dominant_features else 'key indicators'}. "
                    f"Continued monitoring recommended."
                )
            else:
                return (
                    f"Low flood risk. Current conditions within normal parameters. "
                    f"No immediate action required."
                )
        else:  # Earthquake
            seismic = input_data.get('seismic_activity', 'N/A')
            pressure = input_data.get('atmospheric_pressure', 'N/A')
            
            if risk_level == 'HIGH':
                return (
                    f"High seismic risk detected. Activity recorded at {seismic} on Richter scale. "
                    f"Atmospheric pressure at {pressure} hPa. "
                    f"The ensemble model emphasizes {dominant_features[0] if dominant_features else 'seismic indicators'} "
                    f"as the dominant risk factor."
                )
            elif risk_level == 'MODERATE':
                return (
                    f"Moderate seismic risk. Activity levels require monitoring. "
                    f"Primary contributors: {', '.join(dominant_features[:2]) if dominant_features else 'geological factors'}."
                )
            else:
                return (
                    f"Low seismic risk. Current readings within normal baseline. "
                    f"Standard monitoring protocols apply."
                )
    
    def predict(
        self,
        input_data: Union[Dict, pd.DataFrame],
        disaster_type: str = 'Flood',
        region: str = None
    ) -> PredictionResult:
        """
        Make a disaster risk prediction.
        
        Parameters:
        -----------
        input_data : Union[Dict, pd.DataFrame]
            Input environmental features
        disaster_type : str
            Type of disaster ('Flood' or 'Earthquake')
        region : str, optional
            Geographic region for the prediction
            
        Returns:
        --------
        PredictionResult
            Complete prediction with DRI and explanation
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call DisasterPredictionModel.load() first.")
        
        # Store original input for explanation
        original_input = input_data if isinstance(input_data, dict) else input_data.iloc[0].to_dict()
        
        # Preprocess
        X = self._preprocess_input(input_data, disaster_type)
        
        # Get base model predictions
        base_probs = self._get_base_model_predictions(X)
        
        # Get stacking model prediction
        stacked_prob = float(self.stacking_model.predict_proba(X)[0, 1])
        predicted_class = int(self.stacking_model.predict(X)[0])
        
        # Compute DRI
        dri_result = self.dri_calculator.compute_dri(stacked_prob, base_probs)
        
        # Get dominant features
        if self.feature_importance:
            dominant_features = list(self.feature_importance.get('feature_importance', {}).keys())[:3]
        else:
            dominant_features = list(X.columns[:3])
        
        # Generate explanation
        explanation = self._generate_explanation(
            original_input,
            disaster_type,
            dominant_features,
            dri_result.risk_level.value
        )
        
        return PredictionResult(
            timestamp=datetime.now().isoformat(),
            disaster_type=disaster_type,
            predicted_class=predicted_class,
            probability=stacked_prob,
            dri=dri_result.dri,
            risk_level=dri_result.risk_level.value,
            confidence=dri_result.confidence,
            region=region,
            base_model_probabilities=base_probs,
            meta_learner_weights=config.META_LEARNER_WEIGHTS,
            dominant_features=dominant_features,
            explanation=explanation
        )
    
    def predict_batch(
        self,
        input_data: pd.DataFrame,
        disaster_type: str = 'Flood'
    ) -> pd.DataFrame:
        """
        Make predictions for multiple samples.
        
        Parameters:
        -----------
        input_data : pd.DataFrame
            Input features for multiple samples
        disaster_type : str
            Type of disaster
            
        Returns:
        --------
        pd.DataFrame
            Predictions for all samples
        """
        results = []
        
        for idx in range(len(input_data)):
            row = input_data.iloc[[idx]]
            result = self.predict(row, disaster_type)
            result_dict = result.to_dict()
            result_dict['sample_index'] = idx
            results.append(result_dict)
        
        return pd.DataFrame(results)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and status.
        
        Returns:
        --------
        Dict[str, Any]
            Model information
        """
        return {
            'model_version': self.model_version,
            'is_loaded': self.is_loaded,
            'components': {
                'preprocessor': self.preprocessor is not None,
                'random_forest': self.rf_model is not None,
                'gradient_boosting': self.gb_model is not None,
                'svm': self.svm_model is not None,
                'stacking_model': self.stacking_model is not None,
                'feature_importance': self.feature_importance is not None
            },
            'feature_columns': self.preprocessor.feature_columns if self.preprocessor else None,
            'dri_weights': {
                'stacked': config.DRI_WEIGHTS['stacked_probability_weight'],
                'base': config.DRI_WEIGHTS['average_base_probability_weight']
            },
            'risk_thresholds': config.RISK_THRESHOLDS
        }


# Convenience function for quick inference
def predict_disaster_risk(
    input_data: Dict,
    disaster_type: str = 'Flood',
    region: str = None,
    model: DisasterPredictionModel = None
) -> Dict[str, Any]:
    """
    Quick inference function for disaster risk prediction.
    
    Parameters:
    -----------
    input_data : Dict
        Environmental features
    disaster_type : str
        Type of disaster
    region : str, optional
        Geographic region
    model : DisasterPredictionModel, optional
        Pre-loaded model (loads if not provided)
        
    Returns:
    --------
    Dict[str, Any]
        Prediction result as dictionary
    """
    if model is None:
        model = DisasterPredictionModel.load()
    
    result = model.predict(input_data, disaster_type, region)
    return result.to_dict()


if __name__ == "__main__":
    # Example usage
    print("="*60)
    print("INFERENCE EXAMPLE")
    print("="*60)
    
    # Sample flood input
    flood_input = {
        'rainfall': 245.8,
        'temperature': 28.5,
        'humidity': 87,
        'soil_moisture': 72,
        'wind_speed': 45.2,
        'atmospheric_pressure': 1002,
        'previous_disaster': 1
    }
    
    # Sample earthquake input
    earthquake_input = {
        'seismic_activity': 5.2,
        'temperature': 26.3,
        'humidity': 65,
        'wind_speed': 12.5,
        'atmospheric_pressure': 1015,
        'previous_disaster': 0
    }
    
    try:
        # Load model
        model = DisasterPredictionModel.load()
        
        # Flood prediction
        print("\n--- Flood Prediction ---")
        flood_result = model.predict(flood_input, 'Flood', 'Chennai - Zone 4')
        print(f"DRI: {flood_result.dri:.4f}")
        print(f"Risk Level: {flood_result.risk_level}")
        print(f"Confidence: {flood_result.confidence:.1f}%")
        print(f"Explanation: {flood_result.explanation}")
        
        # Earthquake prediction
        print("\n--- Earthquake Prediction ---")
        eq_result = model.predict(earthquake_input, 'Earthquake', 'Delhi - Zone 2')
        print(f"DRI: {eq_result.dri:.4f}")
        print(f"Risk Level: {eq_result.risk_level}")
        print(f"Confidence: {eq_result.confidence:.1f}%")
        print(f"Explanation: {eq_result.explanation}")
        
    except FileNotFoundError:
        print("\nModel not found. Please run train_pipeline.py first.")
        print("Usage: python train_pipeline.py")
