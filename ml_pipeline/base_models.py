"""
STEP 2: Base Models Training for Ensemble-Based Hybrid Disaster Prediction System

Implements:
- RandomForestClassifier
- GradientBoostingClassifier
- SVC (probability=True)
- Evaluation metrics: Accuracy, Precision, Recall, F1, ROC-AUC
- Save evaluation metrics
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)
import joblib
import json
from typing import Dict, Tuple, Any, Optional

import config


class BaseModelTrainer:
    """
    Trainer class for base models in the ensemble.
    """
    
    def __init__(self):
        self.rf_model = None
        self.gb_model = None
        self.svm_model = None
        self.metrics = {}
    
    def train_random_forest(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series
    ) -> RandomForestClassifier:
        """
        Train Random Forest classifier.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training labels
            
        Returns:
        --------
        RandomForestClassifier
            Trained model
        """
        print("\n  Training Random Forest...")
        
        self.rf_model = RandomForestClassifier(**config.RANDOM_FOREST_PARAMS)
        self.rf_model.fit(X_train, y_train)
        
        print(f"    ✓ Random Forest trained with {config.RANDOM_FOREST_PARAMS['n_estimators']} estimators")
        
        return self.rf_model
    
    def train_gradient_boosting(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series
    ) -> GradientBoostingClassifier:
        """
        Train Gradient Boosting classifier.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training labels
            
        Returns:
        --------
        GradientBoostingClassifier
            Trained model
        """
        print("\n  Training Gradient Boosting...")
        
        self.gb_model = GradientBoostingClassifier(**config.GRADIENT_BOOSTING_PARAMS)
        self.gb_model.fit(X_train, y_train)
        
        print(f"    ✓ Gradient Boosting trained with learning_rate={config.GRADIENT_BOOSTING_PARAMS['learning_rate']}")
        
        return self.gb_model
    
    def train_svm(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series
    ) -> SVC:
        """
        Train SVM classifier with probability=True.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training labels
            
        Returns:
        --------
        SVC
            Trained model
        """
        print("\n  Training SVM...")
        
        self.svm_model = SVC(**config.SVM_PARAMS)
        self.svm_model.fit(X_train, y_train)
        
        print(f"    ✓ SVM trained with kernel='{config.SVM_PARAMS['kernel']}', probability=True")
        
        return self.svm_model
    
    def evaluate_model(
        self, 
        model: Any, 
        X_test: pd.DataFrame, 
        y_test: pd.Series,
        model_name: str
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Parameters:
        -----------
        model : Any
            Trained model
        X_test : pd.DataFrame
            Testing features
        y_test : pd.Series
            Testing labels
        model_name : str
            Name of the model
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of evaluation metrics
        """
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': round(accuracy_score(y_test, y_pred) * 100, 2),
            'precision': round(precision_score(y_test, y_pred) * 100, 2),
            'recall': round(recall_score(y_test, y_pred) * 100, 2),
            'f1_score': round(f1_score(y_test, y_pred) * 100, 2),
            'roc_auc': round(roc_auc_score(y_test, y_pred_proba) * 100, 2)
        }
        
        return metrics
    
    def train_all_base_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, Dict[str, float]]:
        """
        Train all base models and evaluate them.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training labels
        X_test : pd.DataFrame
            Testing features
        y_test : pd.Series
            Testing labels
            
        Returns:
        --------
        Dict[str, Dict[str, float]]
            Dictionary of metrics for each model
        """
        print("\n" + "="*60)
        print("STEP 2: BASE MODELS TRAINING")
        print("="*60)
        
        # Train models
        self.train_random_forest(X_train, y_train)
        self.train_gradient_boosting(X_train, y_train)
        self.train_svm(X_train, y_train)
        
        # Evaluate models
        print("\n  Evaluating Base Models...")
        
        self.metrics['random_forest'] = self.evaluate_model(
            self.rf_model, X_test, y_test, 'Random Forest'
        )
        self.metrics['gradient_boosting'] = self.evaluate_model(
            self.gb_model, X_test, y_test, 'Gradient Boosting'
        )
        self.metrics['svm'] = self.evaluate_model(
            self.svm_model, X_test, y_test, 'SVM'
        )
        
        # Print results
        print("\n  ┌─────────────────────┬──────────┬───────────┬──────────┬──────────┬─────────┐")
        print("  │ Model               │ Accuracy │ Precision │ Recall   │ F1-Score │ ROC-AUC │")
        print("  ├─────────────────────┼──────────┼───────────┼──────────┼──────────┼─────────┤")
        
        for model_name, metrics in self.metrics.items():
            display_name = model_name.replace('_', ' ').title()
            print(f"  │ {display_name:<19} │ {metrics['accuracy']:>6.1f}%  │ {metrics['precision']:>7.1f}%  │ {metrics['recall']:>6.1f}%  │ {metrics['f1_score']:>6.1f}%  │ {metrics['roc_auc']:>5.1f}%  │")
        
        print("  └─────────────────────┴──────────┴───────────┴──────────┴──────────┴─────────┘")
        
        return self.metrics
    
    def save_models(self):
        """
        Save all trained base models.
        """
        if self.rf_model is not None:
            joblib.dump(self.rf_model, config.RF_MODEL_PATH)
            print(f"  ✓ Random Forest saved to: {config.RF_MODEL_PATH}")
        
        if self.gb_model is not None:
            joblib.dump(self.gb_model, config.GB_MODEL_PATH)
            print(f"  ✓ Gradient Boosting saved to: {config.GB_MODEL_PATH}")
        
        if self.svm_model is not None:
            joblib.dump(self.svm_model, config.SVM_MODEL_PATH)
            print(f"  ✓ SVM saved to: {config.SVM_MODEL_PATH}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from Random Forest model.
        
        Returns:
        --------
        Dict[str, float]
            Dictionary of feature names and their importance
        """
        if self.rf_model is None:
            raise ValueError("Random Forest model not trained yet.")
        
        importance = self.rf_model.feature_importances_
        feature_names = self.rf_model.feature_names_in_
        
        importance_dict = dict(zip(feature_names, importance))
        importance_dict = dict(sorted(
            importance_dict.items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
        
        return importance_dict
    
    @classmethod
    def load_models(cls) -> 'BaseModelTrainer':
        """
        Load saved base models.
        
        Returns:
        --------
        BaseModelTrainer
            Trainer with loaded models
        """
        trainer = cls()
        
        if config.RF_MODEL_PATH.exists():
            trainer.rf_model = joblib.load(config.RF_MODEL_PATH)
        
        if config.GB_MODEL_PATH.exists():
            trainer.gb_model = joblib.load(config.GB_MODEL_PATH)
        
        if config.SVM_MODEL_PATH.exists():
            trainer.svm_model = joblib.load(config.SVM_MODEL_PATH)
        
        return trainer


def train_base_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[BaseModelTrainer, Dict[str, Dict[str, float]]]:
    """
    Train all base models and return trainer and metrics.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    X_test : pd.DataFrame
        Testing features
    y_test : pd.Series
        Testing labels
        
    Returns:
    --------
    Tuple containing:
        - BaseModelTrainer: Trained model container
        - Dict: Metrics for all models
    """
    trainer = BaseModelTrainer()
    metrics = trainer.train_all_base_models(X_train, y_train, X_test, y_test)
    trainer.save_models()
    
    return trainer, metrics


if __name__ == "__main__":
    # Test with sample data
    from data_generator import generate_unified_dataset
    from preprocessing import prepare_data
    
    df = generate_unified_dataset()
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(df)
    
    trainer, metrics = train_base_models(X_train, y_train, X_test, y_test)
    
    print("\nFeature Importance (from Random Forest):")
    importance = trainer.get_feature_importance()
    for feature, score in importance.items():
        print(f"  {feature}: {score:.4f}")
