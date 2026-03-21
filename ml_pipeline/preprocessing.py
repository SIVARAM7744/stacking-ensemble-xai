"""
STEP 1: Preprocessing Pipeline for Ensemble-Based Hybrid Disaster Prediction System

Implements:
- Missing value handling (median imputation)
- MinMaxScaler normalization
- Feature selection based on disaster type
- Train-test split (80/20)
- Stratified split on target
- Save preprocessing pipeline using joblib
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
from typing import Tuple, Dict, List, Optional, Any

import config


class DisasterPreprocessor:
    """
    Preprocessing pipeline for disaster prediction data.
    Handles missing values, normalization, and feature selection.
    """
    
    def __init__(self, disaster_type: str = 'all'):
        """
        Initialize the preprocessor.
        
        Parameters:
        -----------
        disaster_type : str
            Type of disaster ('Flood', 'Earthquake', or 'all')
        """
        self.disaster_type = disaster_type
        self.imputer = None
        self.scaler = None
        self.feature_columns = None
        self.is_fitted = False
        self.dataset_version = "v1.0"
        
    def get_feature_columns(self, disaster_type: str) -> List[str]:
        """
        Get feature columns based on disaster type.
        
        Parameters:
        -----------
        disaster_type : str
            Type of disaster
            
        Returns:
        --------
        List[str]
            List of feature column names
        """
        if disaster_type == 'Flood':
            return config.FLOOD_FEATURES.copy()
        elif disaster_type == 'Earthquake':
            return config.EARTHQUAKE_FEATURES.copy()
        else:
            # All features for unified model
            return list(set(config.FLOOD_FEATURES + config.EARTHQUAKE_FEATURES))
    
    def handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values using median imputation.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with imputed values
        """
        if self.imputer is None:
            self.imputer = SimpleImputer(strategy='median')
            X_imputed = self.imputer.fit_transform(X)
        else:
            X_imputed = self.imputer.transform(X)
        
        return pd.DataFrame(X_imputed, columns=X.columns, index=X.index)
    
    def normalize_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Min-Max normalization to features.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
            
        Returns:
        --------
        pd.DataFrame
            Normalized DataFrame
        """
        if self.scaler is None:
            self.scaler = MinMaxScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    def fit_transform(
        self, 
        X: pd.DataFrame, 
        y: Optional[pd.Series] = None,
        disaster_type: str = 'all'
    ) -> pd.DataFrame:
        """
        Fit the preprocessor and transform the data.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
        y : pd.Series, optional
            Target values (not used in transformation)
        disaster_type : str
            Type of disaster for feature selection
            
        Returns:
        --------
        pd.DataFrame
            Preprocessed features
        """
        self.disaster_type = disaster_type
        self.feature_columns = self.get_feature_columns(disaster_type)
        
        # Filter to relevant columns
        available_cols = [c for c in self.feature_columns if c in X.columns]
        X_filtered = X[available_cols].copy()
        
        # Step 1: Handle missing values
        X_imputed = self.handle_missing_values(X_filtered)
        
        # Step 2: Normalize features
        X_normalized = self.normalize_features(X_imputed)
        
        self.is_fitted = True
        
        return X_normalized
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted preprocessor.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
            
        Returns:
        --------
        pd.DataFrame
            Preprocessed features
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor has not been fitted. Call fit_transform first.")
        
        # Filter to relevant columns
        available_cols = [c for c in self.feature_columns if c in X.columns]
        X_filtered = X[available_cols].copy()
        
        # Apply transformations
        X_imputed = self.imputer.transform(X_filtered)
        X_imputed = pd.DataFrame(X_imputed, columns=available_cols, index=X.index)
        
        X_normalized = self.scaler.transform(X_imputed)
        X_normalized = pd.DataFrame(X_normalized, columns=available_cols, index=X.index)
        
        return X_normalized
    
    def get_processing_status(self) -> Dict[str, str]:
        """
        Get current processing status for display.
        
        Returns:
        --------
        Dict[str, str]
            Processing status information
        """
        return {
            'Missing Value Handling': 'Completed' if self.imputer is not None else 'Pending',
            'Min-Max Normalization': 'Active' if self.scaler is not None else 'Pending',
            'Feature Transformation': 'Applied' if self.is_fitted else 'Pending',
            'Dataset Version': self.dataset_version
        }
    
    def save(self, path: Optional[str] = None):
        """
        Save the preprocessing pipeline.
        
        Parameters:
        -----------
        path : str, optional
            Path to save the pipeline
        """
        if path is None:
            path = config.PREPROCESSING_PATH
        
        pipeline_data = {
            'imputer': self.imputer,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'disaster_type': self.disaster_type,
            'is_fitted': self.is_fitted,
            'dataset_version': self.dataset_version
        }
        
        joblib.dump(pipeline_data, path)
        print(f"✓ Preprocessing pipeline saved to: {path}")
    
    @classmethod
    def load(cls, path: Optional[str] = None) -> 'DisasterPreprocessor':
        """
        Load a saved preprocessing pipeline.
        
        Parameters:
        -----------
        path : str, optional
            Path to load the pipeline from
            
        Returns:
        --------
        DisasterPreprocessor
            Loaded preprocessor instance
        """
        if path is None:
            path = config.PREPROCESSING_PATH
        
        pipeline_data = joblib.load(path)
        
        preprocessor = cls()
        preprocessor.imputer = pipeline_data['imputer']
        preprocessor.scaler = pipeline_data['scaler']
        preprocessor.feature_columns = pipeline_data['feature_columns']
        preprocessor.disaster_type = pipeline_data['disaster_type']
        preprocessor.is_fitted = pipeline_data['is_fitted']
        preprocessor.dataset_version = pipeline_data.get('dataset_version', 'v1.0')
        
        return preprocessor


def prepare_data(
    df: pd.DataFrame,
    disaster_type: str = 'all',
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, DisasterPreprocessor]:
    """
    Prepare data for training with preprocessing pipeline.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw dataset
    disaster_type : str
        Type of disaster ('Flood', 'Earthquake', or 'all')
    test_size : float
        Proportion of data for testing
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    Tuple containing:
        - X_train: Training features
        - X_test: Testing features
        - y_train: Training labels
        - y_test: Testing labels
        - preprocessor: Fitted preprocessor
    """
    print("\n" + "="*60)
    print("STEP 1: PREPROCESSING PIPELINE")
    print("="*60)
    
    # Filter by disaster type if specified
    if disaster_type != 'all':
        df = df[df[config.DISASTER_TYPE_COLUMN] == disaster_type].copy()
        print(f"→ Filtered to {disaster_type} samples: {len(df)}")
    else:
        print(f"→ Using all samples: {len(df)}")
    
    # Separate features and target
    X = df.drop(columns=[config.TARGET_COLUMN, config.DISASTER_TYPE_COLUMN], errors='ignore')
    y = df[config.TARGET_COLUMN]
    
    # Check for missing values
    missing_count = X.isnull().sum().sum()
    print(f"→ Missing values detected: {missing_count}")
    
    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    
    print(f"→ Train-test split (80/20): Train={len(X_train)}, Test={len(X_test)}")
    print(f"→ Stratified split - Train positive rate: {y_train.mean():.2%}")
    print(f"→ Stratified split - Test positive rate: {y_test.mean():.2%}")
    
    # Initialize and fit preprocessor
    preprocessor = DisasterPreprocessor(disaster_type)
    
    # Fit and transform training data
    X_train_processed = preprocessor.fit_transform(X_train, y_train, disaster_type)
    
    # Transform test data
    X_test_processed = preprocessor.transform(X_test)
    
    print(f"→ Feature columns selected: {preprocessor.feature_columns}")
    print(f"→ Missing Value Handling: Median imputation applied")
    print(f"→ Min-Max Normalization: Applied (range 0-1)")
    
    # Save preprocessing pipeline
    preprocessor.save()
    
    # Get processing status
    status = preprocessor.get_processing_status()
    print("\n→ Processing Status:")
    for key, value in status.items():
        print(f"   • {key}: {value}")
    
    print(f"\n→ Note: Input data undergoes Min-Max normalization and feature")
    print(f"        transformation prior to ensemble inference.")
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor


if __name__ == "__main__":
    # Test with sample data
    from data_generator import generate_unified_dataset
    
    df = generate_unified_dataset()
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(df)
    
    print(f"\nFinal shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")
