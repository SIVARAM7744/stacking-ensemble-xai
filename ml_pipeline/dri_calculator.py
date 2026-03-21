"""
STEP 4: Disaster Risk Index (DRI) Calculator

Implements:
- DRI = Weighted probability from stacking model
- DRI = 0.6 * stacked_probability + 0.4 * average_base_probability
- Returns: Probability, DRI score, Risk Category
- Risk Categories:
    - 0-0.33 → Low
    - 0.34-0.66 → Moderate  
    - 0.67-1.0 → High
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

import config


class RiskLevel(Enum):
    """Risk level enumeration."""
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"


@dataclass
class DRIPrediction:
    """
    Data class for DRI prediction results.
    
    Attributes:
    -----------
    probability : float
        Raw probability from stacking model
    dri : float
        Computed Disaster Risk Index
    risk_level : RiskLevel
        Categorical risk level
    confidence : float
        Prediction confidence score
    base_probabilities : Dict[str, float]
        Individual base model probabilities
    """
    probability: float
    dri: float
    risk_level: RiskLevel
    confidence: float
    base_probabilities: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'probability': round(self.probability, 4),
            'dri': round(self.dri, 4),
            'risk_level': self.risk_level.value,
            'confidence': round(self.confidence, 2),
            'base_probabilities': {
                k: round(v, 4) for k, v in self.base_probabilities.items()
            }
        }


class DRICalculator:
    """
    Disaster Risk Index calculator.
    Computes weighted DRI from ensemble model predictions.
    """
    
    def __init__(
        self,
        stacked_weight: float = None,
        base_weight: float = None
    ):
        """
        Initialize DRI Calculator.
        
        Parameters:
        -----------
        stacked_weight : float, optional
            Weight for stacked probability (default from config)
        base_weight : float, optional
            Weight for average base probability (default from config)
        """
        self.stacked_weight = stacked_weight or config.DRI_WEIGHTS['stacked_probability_weight']
        self.base_weight = base_weight or config.DRI_WEIGHTS['average_base_probability_weight']
        
        # Validate weights sum to 1
        if abs((self.stacked_weight + self.base_weight) - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")
    
    def get_risk_level(self, dri: float) -> RiskLevel:
        """
        Determine risk level from DRI score.
        
        Parameters:
        -----------
        dri : float
            Disaster Risk Index score (0-1)
            
        Returns:
        --------
        RiskLevel
            Categorical risk level
        """
        if dri <= config.RISK_THRESHOLDS['low_max']:
            return RiskLevel.LOW
        elif dri <= config.RISK_THRESHOLDS['moderate_max']:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.HIGH
    
    def calculate_confidence(
        self, 
        stacked_prob: float,
        base_probs: Dict[str, float]
    ) -> float:
        """
        Calculate prediction confidence based on model agreement.
        
        Parameters:
        -----------
        stacked_prob : float
            Stacked model probability
        base_probs : Dict[str, float]
            Base model probabilities
            
        Returns:
        --------
        float
            Confidence score (0-100)
        """
        all_probs = list(base_probs.values()) + [stacked_prob]
        
        # Confidence based on variance (lower variance = higher confidence)
        variance = np.var(all_probs)
        max_variance = 0.25  # Maximum possible variance for probabilities in [0,1]
        
        # Convert to confidence percentage
        confidence = (1 - (variance / max_variance)) * 100
        confidence = max(min(confidence, 100), 0)  # Clamp to [0, 100]
        
        return confidence
    
    def compute_dri(
        self,
        stacked_probability: float,
        base_probabilities: Dict[str, float]
    ) -> DRIPrediction:
        """
        Compute Disaster Risk Index from model probabilities.
        
        Formula: DRI = 0.6 * stacked_probability + 0.4 * average_base_probability
        
        Parameters:
        -----------
        stacked_probability : float
            Probability from stacking meta-learner
        base_probabilities : Dict[str, float]
            Dictionary of base model probabilities {'rf': float, 'gb': float, 'svm': float}
            
        Returns:
        --------
        DRIPrediction
            Complete DRI prediction result
        """
        # Calculate average base probability
        avg_base_prob = np.mean(list(base_probabilities.values()))
        
        # Compute DRI
        dri = (
            self.stacked_weight * stacked_probability + 
            self.base_weight * avg_base_prob
        )
        
        # Ensure DRI is in valid range
        dri = max(min(dri, 1.0), 0.0)
        
        # Get risk level
        risk_level = self.get_risk_level(dri)
        
        # Calculate confidence
        confidence = self.calculate_confidence(stacked_probability, base_probabilities)
        
        return DRIPrediction(
            probability=stacked_probability,
            dri=dri,
            risk_level=risk_level,
            confidence=confidence,
            base_probabilities=base_probabilities
        )
    
    def compute_batch_dri(
        self,
        stacked_probabilities: np.ndarray,
        base_model_predictions: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """
        Compute DRI for batch predictions.
        
        Parameters:
        -----------
        stacked_probabilities : np.ndarray
            Array of stacked model probabilities
        base_model_predictions : Dict[str, np.ndarray]
            Dictionary of base model probability arrays
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with DRI results for all samples
        """
        results = []
        
        for i in range(len(stacked_probabilities)):
            base_probs = {
                name: probs[i] 
                for name, probs in base_model_predictions.items()
            }
            
            prediction = self.compute_dri(
                stacked_probabilities[i],
                base_probs
            )
            
            result_dict = prediction.to_dict()
            result_dict['sample_index'] = i
            results.append(result_dict)
        
        df = pd.DataFrame(results)
        
        # Reorder columns
        cols = ['sample_index', 'probability', 'dri', 'risk_level', 'confidence']
        df = df[cols]
        
        return df
    
    def get_weight_distribution(self) -> Dict[str, float]:
        """
        Get the meta-learner weight distribution for display.
        
        Returns:
        --------
        Dict[str, float]
            Weight distribution for base models
        """
        return config.META_LEARNER_WEIGHTS.copy()
    
    def get_dri_formula(self) -> str:
        """
        Get the DRI computation formula as a string.
        
        Returns:
        --------
        str
            Formula description
        """
        return (
            f"DRI = {self.stacked_weight} × stacked_probability + "
            f"{self.base_weight} × average_base_probability"
        )


def compute_dri_from_ensemble(
    stacking_ensemble,
    X: pd.DataFrame
) -> pd.DataFrame:
    """
    Convenience function to compute DRI from a trained stacking ensemble.
    
    Parameters:
    -----------
    stacking_ensemble : StackingEnsemble
        Trained stacking ensemble model
    X : pd.DataFrame
        Input features
        
    Returns:
    --------
    pd.DataFrame
        DRI results for all samples
    """
    print("\n" + "="*60)
    print("STEP 4: DISASTER RISK INDEX (DRI) COMPUTATION")
    print("="*60)
    
    calculator = DRICalculator()
    
    print(f"\n  DRI Formula:")
    print(f"    {calculator.get_dri_formula()}")
    
    # Get predictions
    stacked_probs = stacking_ensemble.predict_proba(X)[:, 1]
    base_predictions = stacking_ensemble.get_base_model_predictions(X)
    
    # Compute DRI
    results = calculator.compute_batch_dri(stacked_probs, base_predictions)
    
    # Summary statistics
    print(f"\n  DRI Statistics (n={len(results)}):")
    print(f"    → Mean DRI: {results['dri'].mean():.3f}")
    print(f"    → Std DRI:  {results['dri'].std():.3f}")
    print(f"    → Min DRI:  {results['dri'].min():.3f}")
    print(f"    → Max DRI:  {results['dri'].max():.3f}")
    
    # Risk level distribution
    risk_counts = results['risk_level'].value_counts()
    print(f"\n  Risk Level Distribution:")
    for level in ['LOW', 'MODERATE', 'HIGH']:
        count = risk_counts.get(level, 0)
        pct = (count / len(results)) * 100
        print(f"    → {level}: {count} ({pct:.1f}%)")
    
    print(f"\n  Risk Categories:")
    print(f"    → 0.00 - 0.33: Low Risk")
    print(f"    → 0.34 - 0.66: Moderate Risk")
    print(f"    → 0.67 - 1.00: High Risk")
    
    return results


if __name__ == "__main__":
    # Test DRI calculation
    calculator = DRICalculator()
    
    # Example computation
    stacked_prob = 0.85
    base_probs = {
        'rf': 0.78,
        'gb': 0.88,
        'svm': 0.82
    }
    
    result = calculator.compute_dri(stacked_prob, base_probs)
    
    print("DRI Calculation Example:")
    print(f"  Stacked Probability: {stacked_prob}")
    print(f"  Base Probabilities: {base_probs}")
    print(f"  Average Base Prob: {np.mean(list(base_probs.values())):.3f}")
    print(f"\n  Formula: {calculator.get_dri_formula()}")
    print(f"\n  Results:")
    print(f"    DRI: {result.dri:.4f}")
    print(f"    Risk Level: {result.risk_level.value}")
    print(f"    Confidence: {result.confidence:.1f}%")
