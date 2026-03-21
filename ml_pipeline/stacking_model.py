"""
STEP 3: Stacking Meta-Learner for Ensemble-Based Hybrid Disaster Prediction System

Implements:
- sklearn.ensemble.StackingClassifier
- Base learners: RF, GB, SVM
- Meta learner: LogisticRegression
- 5-fold cross-validation inside stacking
- Training and evaluation
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    StackingClassifier, 
    RandomForestClassifier, 
    GradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)
import joblib
from typing import Dict, Tuple, Any, Optional

import config


class StackingEnsemble:
    """
    Stacking ensemble classifier with RF, GB, SVM as base learners
    and Logistic Regression as meta-learner.
    """
    
    def __init__(self):
        self.stacking_model = None
        self.base_learners = None
        self.meta_learner = None
        self.cv_scores = None
        self.metrics = None
    
    def _create_base_learners(self) -> list:
        """
        Create base learners for stacking.
        
        Returns:
        --------
        list
            List of (name, estimator) tuples
        """
        base_learners = [
            ('rf', RandomForestClassifier(**config.RANDOM_FOREST_PARAMS)),
            ('gb', GradientBoostingClassifier(**config.GRADIENT_BOOSTING_PARAMS)),
            ('svm', SVC(**config.SVM_PARAMS))
        ]
        
        return base_learners
    
    def _create_meta_learner(self) -> LogisticRegression:
        """
        Create meta-learner (final estimator).
        
        Returns:
        --------
        LogisticRegression
            Configured logistic regression model
        """
        return LogisticRegression(**config.META_LEARNER_PARAMS)
    
    def build_stacking_classifier(self) -> StackingClassifier:
        """
        Build the stacking classifier.
        
        Returns:
        --------
        StackingClassifier
            Configured stacking classifier
        """
        self.base_learners = self._create_base_learners()
        self.meta_learner = self._create_meta_learner()
        
        self.stacking_model = StackingClassifier(
            estimators=self.base_learners,
            final_estimator=self.meta_learner,
            cv=config.CV_FOLDS,  # 5-fold cross-validation
            stack_method='predict_proba',
            passthrough=False,
            # Keep single-process for restricted Windows environments.
            n_jobs=1
        )
        
        return self.stacking_model
    
    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series
    ) -> StackingClassifier:
        """
        Train the stacking ensemble.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training labels
            
        Returns:
        --------
        StackingClassifier
            Trained stacking model
        """
        print("\n" + "="*60)
        print("STEP 3: STACKING META-LEARNER TRAINING")
        print("="*60)
        
        print("\n  Building Stacking Classifier...")
        print(f"    → Base Learners: RF, GB, SVM")
        print(f"    → Meta Learner: Logistic Regression")
        print(f"    → Cross-Validation: {config.CV_FOLDS}-fold")
        
        # Build the model
        self.build_stacking_classifier()
        
        # Train
        print("\n  Training Stacking Ensemble...")
        self.stacking_model.fit(X_train, y_train)
        
        # Perform cross-validation scoring
        print("\n  Performing Cross-Validation...")
        self.cv_scores = cross_val_score(
            self.stacking_model, X_train, y_train, 
            cv=config.CV_FOLDS, scoring='accuracy'
        )
        
        print(f"    → CV Scores: {[f'{s:.3f}' for s in self.cv_scores]}")
        print(f"    → Mean CV Accuracy: {self.cv_scores.mean():.2%} (±{self.cv_scores.std():.2%})")
        
        print("\n  ✓ Stacking Meta-Learner trained successfully")
        
        return self.stacking_model
    
    def evaluate(
        self, 
        X_test: pd.DataFrame, 
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate the stacking model.
        
        Parameters:
        -----------
        X_test : pd.DataFrame
            Testing features
        y_test : pd.Series
            Testing labels
            
        Returns:
        --------
        Dict[str, float]
            Evaluation metrics
        """
        if self.stacking_model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        y_pred = self.stacking_model.predict(X_test)
        y_pred_proba = self.stacking_model.predict_proba(X_test)[:, 1]
        
        self.metrics = {
            'accuracy': round(accuracy_score(y_test, y_pred) * 100, 2),
            'precision': round(precision_score(y_test, y_pred) * 100, 2),
            'recall': round(recall_score(y_test, y_pred) * 100, 2),
            'f1_score': round(f1_score(y_test, y_pred) * 100, 2),
            'roc_auc': round(roc_auc_score(y_test, y_pred_proba) * 100, 2)
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        self.metrics['confusion_matrix'] = {
            'tn': int(cm[0, 0]),
            'fp': int(cm[0, 1]),
            'fn': int(cm[1, 0]),
            'tp': int(cm[1, 1])
        }
        
        print("\n  Stacking Meta-Learner Evaluation:")
        print(f"    → Accuracy:  {self.metrics['accuracy']:.1f}%")
        print(f"    → Precision: {self.metrics['precision']:.1f}%")
        print(f"    → Recall:    {self.metrics['recall']:.1f}%")
        print(f"    → F1-Score:  {self.metrics['f1_score']:.1f}%")
        print(f"    → ROC-AUC:   {self.metrics['roc_auc']:.1f}%")
        
        print(f"\n  Confusion Matrix:")
        print(f"    → True Negatives:  {self.metrics['confusion_matrix']['tn']}")
        print(f"    → False Positives: {self.metrics['confusion_matrix']['fp']}")
        print(f"    → False Negatives: {self.metrics['confusion_matrix']['fn']}")
        print(f"    → True Positives:  {self.metrics['confusion_matrix']['tp']}")
        
        print(f"\n  Note: Stacking meta-learner optimizes decision boundary")
        print(f"        across heterogeneous base learners.")
        
        return self.metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the stacking model.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
            
        Returns:
        --------
        np.ndarray
            Predicted classes
        """
        if self.stacking_model is None:
            raise ValueError("Model not trained yet.")
        
        return self.stacking_model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get probability predictions from the stacking model.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
            
        Returns:
        --------
        np.ndarray
            Probability predictions
        """
        if self.stacking_model is None:
            raise ValueError("Model not trained yet.")
        
        return self.stacking_model.predict_proba(X)
    
    def get_base_model_predictions(
        self, 
        X: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        """
        Get predictions from each base model.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
            
        Returns:
        --------
        Dict[str, np.ndarray]
            Dictionary of model predictions
        """
        if self.stacking_model is None:
            raise ValueError("Model not trained yet.")
        
        predictions = {}
        
        for name, estimator in self.stacking_model.named_estimators_.items():
            predictions[name] = estimator.predict_proba(X)[:, 1]
        
        return predictions
    
    def save(self, path: Optional[str] = None):
        """
        Save the stacking model.
        
        Parameters:
        -----------
        path : str, optional
            Path to save the model
        """
        if path is None:
            path = config.STACKING_MODEL_PATH
        
        model_data = {
            'stacking_model': self.stacking_model,
            'cv_scores': self.cv_scores,
            'metrics': self.metrics
        }
        
        joblib.dump(model_data, path)
        print(f"\n  ✓ Stacking model saved to: {path}")
    
    @classmethod
    def load(cls, path: Optional[str] = None) -> 'StackingEnsemble':
        """
        Load a saved stacking model.
        
        Parameters:
        -----------
        path : str, optional
            Path to load the model from
            
        Returns:
        --------
        StackingEnsemble
            Loaded model instance
        """
        if path is None:
            path = config.STACKING_MODEL_PATH
        
        model_data = joblib.load(path)
        
        ensemble = cls()
        ensemble.stacking_model = model_data['stacking_model']
        ensemble.cv_scores = model_data.get('cv_scores')
        ensemble.metrics = model_data.get('metrics')
        
        return ensemble


def train_stacking_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[StackingEnsemble, Dict[str, float]]:
    """
    Train and evaluate the stacking ensemble.
    
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
        - StackingEnsemble: Trained ensemble
        - Dict: Evaluation metrics
    """
    ensemble = StackingEnsemble()
    ensemble.train(X_train, y_train)
    metrics = ensemble.evaluate(X_test, y_test)
    ensemble.save()
    
    return ensemble, metrics


if __name__ == "__main__":
    # Test with sample data
    from data_generator import generate_unified_dataset
    from preprocessing import prepare_data
    
    df = generate_unified_dataset()
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(df)
    
    ensemble, metrics = train_stacking_model(X_train, y_train, X_test, y_test)
    
    # Test predictions
    print("\nSample predictions:")
    sample_proba = ensemble.predict_proba(X_test[:5])
    print(f"Probabilities: {sample_proba[:, 1]}")
