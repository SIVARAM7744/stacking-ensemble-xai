"""
STEP 5: Explainability Module for Ensemble-Based Hybrid Disaster Prediction System

Implements:
- Global feature importance from Random Forest
- SHAP explainer for stacked model (or meta learner)
- Returns:
    - Feature importance dict
    - SHAP contribution per instance
"""

import numpy as np
import pandas as pd
import json
from typing import Dict, List, Optional, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

import config

# Try to import SHAP, with fallback
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Using fallback explainability methods.")


class ExplainabilityEngine:
    """
    Explainability engine for ensemble disaster prediction model.
    Provides feature importance and SHAP-style explanations.
    """
    
    def __init__(self):
        self.feature_importance = None
        self.shap_explainer = None
        self.shap_values = None
        self.feature_names = None
    
    def compute_feature_importance(
        self,
        rf_model,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compute global feature importance from Random Forest.
        
        Parameters:
        -----------
        rf_model : RandomForestClassifier
            Trained Random Forest model
        feature_names : List[str], optional
            List of feature names
            
        Returns:
        --------
        Dict[str, float]
            Feature importance dictionary (sorted by importance)
        """
        print("\n  Computing Global Feature Importance...")
        
        # Get feature names
        if feature_names is None:
            if hasattr(rf_model, 'feature_names_in_'):
                feature_names = rf_model.feature_names_in_
            else:
                feature_names = [f"feature_{i}" for i in range(rf_model.n_features_in_)]
        
        self.feature_names = list(feature_names)
        
        # Get importance scores
        importance_scores = rf_model.feature_importances_
        
        # Create dictionary and sort
        self.feature_importance = dict(zip(self.feature_names, importance_scores))
        self.feature_importance = dict(
            sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        )
        
        # Convert to percentages
        total = sum(self.feature_importance.values())
        importance_pct = {
            k: round((v / total) * 100, 1) 
            for k, v in self.feature_importance.items()
        }
        
        print("    ✓ Feature Importance (from Random Forest):")
        for feature, pct in importance_pct.items():
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"      {feature:<25} {bar} {pct:>5.1f}%")
        
        return importance_pct
    
    def initialize_shap_explainer(
        self,
        model,
        X_train: pd.DataFrame,
        model_type: str = 'tree'
    ):
        """
        Initialize SHAP explainer for the model.
        
        Parameters:
        -----------
        model : Any
            Trained model (can be RF, GB, or stacking model)
        X_train : pd.DataFrame
            Training data for background distribution
        model_type : str
            Type of model ('tree', 'linear', 'kernel')
        """
        print("\n  Initializing SHAP Explainer...")
        
        if not SHAP_AVAILABLE:
            print("    ⚠ SHAP not available. Using coefficient-based approximation.")
            self.shap_explainer = None
            return
        
        try:
            if model_type == 'tree':
                # Use TreeExplainer for tree-based models
                self.shap_explainer = shap.TreeExplainer(model)
                print("    ✓ TreeExplainer initialized")
            elif model_type == 'linear':
                # Use LinearExplainer for linear models
                self.shap_explainer = shap.LinearExplainer(model, X_train)
                print("    ✓ LinearExplainer initialized")
            else:
                # Use KernelExplainer for any model (slower)
                # Sample background data for efficiency
                background = shap.sample(X_train, min(100, len(X_train)))
                self.shap_explainer = shap.KernelExplainer(model.predict_proba, background)
                print("    ✓ KernelExplainer initialized")
        except Exception as e:
            print(f"    ⚠ SHAP initialization failed: {str(e)}")
            print("    ⚠ Using fallback explanation method.")
            self.shap_explainer = None
    
    def compute_shap_values(
        self,
        X: pd.DataFrame,
        max_samples: int = 100
    ) -> np.ndarray:
        """
        Compute SHAP values for input samples.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
        max_samples : int
            Maximum number of samples to explain (for efficiency)
            
        Returns:
        --------
        np.ndarray
            SHAP values array
        """
        if not SHAP_AVAILABLE or self.shap_explainer is None:
            # Fallback: Use feature importance as proxy
            return self._compute_fallback_contributions(X)
        
        print(f"\n  Computing SHAP values for {min(len(X), max_samples)} samples...")
        
        # Limit samples for efficiency
        if len(X) > max_samples:
            X = X.sample(n=max_samples, random_state=42)
        
        try:
            self.shap_values = self.shap_explainer.shap_values(X)
            
            # Handle multi-class output
            if isinstance(self.shap_values, list):
                # Use positive class SHAP values
                self.shap_values = self.shap_values[1]
            
            print("    ✓ SHAP values computed successfully")
            
        except Exception as e:
            print(f"    ⚠ SHAP computation failed: {str(e)}")
            self.shap_values = self._compute_fallback_contributions(X)
        
        return self.shap_values
    
    def _compute_fallback_contributions(
        self,
        X: pd.DataFrame
    ) -> np.ndarray:
        """
        Compute fallback feature contributions when SHAP is not available.
        Uses feature importance weighted by normalized feature values.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
            
        Returns:
        --------
        np.ndarray
            Approximated contribution values
        """
        print("    Using fallback contribution calculation...")
        
        if self.feature_importance is None:
            # Uniform importance if not available
            n_features = X.shape[1]
            importance = np.ones(n_features) / n_features
        else:
            # Normalize importance to sum to 1
            importance = np.array([
                self.feature_importance.get(col, 0) 
                for col in X.columns
            ])
            importance = importance / importance.sum()
        
        # Compute contributions as importance * (value - 0.5)
        # Centering at 0.5 since data is MinMax normalized to [0, 1]
        X_centered = X.values - 0.5
        contributions = X_centered * importance
        
        return contributions
    
    def get_instance_explanation(
        self,
        X_instance: pd.DataFrame,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Get SHAP-style explanation for a single instance.
        
        Parameters:
        -----------
        X_instance : pd.DataFrame
            Single instance features (1 row)
        top_k : int
            Number of top contributing features to return
            
        Returns:
        --------
        Dict[str, Any]
            Instance explanation with feature contributions
        """
        # Ensure single row
        if len(X_instance) > 1:
            X_instance = X_instance.iloc[[0]]
        
        # Compute SHAP values for this instance
        if SHAP_AVAILABLE and self.shap_explainer is not None:
            try:
                shap_vals = self.shap_explainer.shap_values(X_instance)
                if isinstance(shap_vals, list):
                    # Binary classification often returns [class0, class1]
                    shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]

                shap_arr = np.asarray(shap_vals)
                # Common shapes:
                #   (1, n_features)            -> single sample
                #   (1, n_features, n_classes) -> class-wise values
                #   (n_features,)               -> already flattened
                if shap_arr.ndim == 3:
                    # Use positive class (index 1) when available, else class 0.
                    class_idx = 1 if shap_arr.shape[2] > 1 else 0
                    contributions = shap_arr[0, :, class_idx]
                elif shap_arr.ndim == 2:
                    contributions = shap_arr[0]
                else:
                    contributions = shap_arr.reshape(-1)
            except Exception:
                contributions = self._compute_fallback_contributions(X_instance)[0]
        else:
            contributions = self._compute_fallback_contributions(X_instance)[0]
        
        # Create contribution dictionary
        feature_names = X_instance.columns.tolist()
        contribution_dict = dict(zip(feature_names, contributions))
        
        # Sort by absolute contribution
        sorted_contributions = dict(
            sorted(contribution_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        )
        
        # Get top contributors
        top_features = dict(list(sorted_contributions.items())[:top_k])
        
        # Classify as increasing or decreasing risk
        positive_contributors = {
            k: v for k, v in sorted_contributions.items() if v > 0
        }
        negative_contributors = {
            k: v for k, v in sorted_contributions.items() if v < 0
        }
        
        return {
            'all_contributions': {k: round(v, 4) for k, v in sorted_contributions.items()},
            'top_contributors': {k: round(v, 4) for k, v in top_features.items()},
            'risk_increasing': list(positive_contributors.keys()),
            'risk_decreasing': list(negative_contributors.keys()),
            'dominant_feature': list(sorted_contributions.keys())[0] if sorted_contributions else None
        }
    
    def get_global_explanation(self) -> Dict[str, Any]:
        """
        Get global model explanation summary.
        
        Returns:
        --------
        Dict[str, Any]
            Global explanation with feature importance and statistics
        """
        if self.feature_importance is None:
            raise ValueError("Feature importance not computed. Call compute_feature_importance first.")
        
        # Convert to percentages
        total = sum(self.feature_importance.values())
        importance_pct = {
            k: round((v / total) * 100, 2) 
            for k, v in self.feature_importance.items()
        }
        
        return {
            'feature_importance': importance_pct,
            'top_feature': list(self.feature_importance.keys())[0],
            'feature_count': len(self.feature_importance),
            'importance_threshold_50pct': self._get_features_for_threshold(50),
            'importance_threshold_80pct': self._get_features_for_threshold(80)
        }
    
    def _get_features_for_threshold(self, threshold_pct: float) -> List[str]:
        """
        Get features that account for threshold% of total importance.
        """
        if self.feature_importance is None:
            return []
        
        total = sum(self.feature_importance.values())
        cumsum = 0
        features = []
        
        for feature, importance in self.feature_importance.items():
            cumsum += importance
            features.append(feature)
            if (cumsum / total) * 100 >= threshold_pct:
                break
        
        return features
    
    def save_importance(self, path: Optional[str] = None):
        """
        Save feature importance to JSON file.
        
        Parameters:
        -----------
        path : str, optional
            Path to save the file
        """
        if path is None:
            path = config.FEATURE_IMPORTANCE_PATH
        
        # Convert numpy values to float
        importance_data = {
            'feature_importance': {
                k: float(v) for k, v in self.feature_importance.items()
            },
            'feature_names': self.feature_names
        }
        
        with open(path, 'w') as f:
            json.dump(importance_data, f, indent=2)
        
        print(f"    ✓ Feature importance saved to: {path}")


def compute_explainability(
    rf_model,
    stacking_model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple[ExplainabilityEngine, Dict[str, Any]]:
    """
    Compute full explainability analysis.
    
    Parameters:
    -----------
    rf_model : RandomForestClassifier
        Trained Random Forest model
    stacking_model : StackingClassifier
        Trained stacking model
    X_train : pd.DataFrame
        Training features
    X_test : pd.DataFrame
        Testing features
        
    Returns:
    --------
    Tuple containing:
        - ExplainabilityEngine: Configured engine
        - Dict: Explainability results
    """
    print("\n" + "="*60)
    print("STEP 5: EXPLAINABILITY MODULE")
    print("="*60)
    
    engine = ExplainabilityEngine()
    
    # Compute feature importance from RF
    importance_pct = engine.compute_feature_importance(rf_model)
    
    # Initialize SHAP explainer using RF (tree-based)
    engine.initialize_shap_explainer(rf_model, X_train, model_type='tree')
    
    # Compute SHAP values for test set
    shap_values = engine.compute_shap_values(X_test, max_samples=50)
    
    # Get global explanation
    global_explanation = engine.get_global_explanation()
    
    # Get sample instance explanation
    print("\n  Sample Instance Explanation:")
    sample_explanation = engine.get_instance_explanation(X_test.iloc[[0]])
    print(f"    Dominant feature: {sample_explanation['dominant_feature']}")
    print(f"    Risk increasing: {sample_explanation['risk_increasing'][:3]}")
    print(f"    Risk decreasing: {sample_explanation['risk_decreasing'][:2]}")
    
    # SHAP-style contribution display
    print("\n  SHAP-Style Contributions (Sample Instance):")
    for feature, value in list(sample_explanation['top_contributors'].items())[:4]:
        direction = "→ +" if value > 0 else "→ "
        color = "↑ Risk" if value > 0 else "↓ Risk"
        print(f"    {feature:<25} {direction}{value:.4f} ({color})")
    
    print("\n  Note: SHAP-style local explanation computed for current inference instance.")
    
    # Save feature importance
    engine.save_importance()
    
    return engine, {
        'feature_importance': importance_pct,
        'global_explanation': global_explanation,
        'sample_explanation': sample_explanation
    }


if __name__ == "__main__":
    # Test explainability
    from data_generator import generate_unified_dataset
    from preprocessing import prepare_data
    from base_models import train_base_models
    from stacking_model import train_stacking_model
    
    df = generate_unified_dataset()
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(df)
    trainer, base_metrics = train_base_models(X_train, y_train, X_test, y_test)
    ensemble, stack_metrics = train_stacking_model(X_train, y_train, X_test, y_test)
    
    engine, results = compute_explainability(
        trainer.rf_model,
        ensemble.stacking_model,
        X_train,
        X_test
    )
