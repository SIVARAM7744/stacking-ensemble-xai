"""
Main Training Pipeline for Ensemble-Based Hybrid Disaster Prediction System

Complete ML pipeline that orchestrates:
- STEP 1: Preprocessing (missing values, normalization, feature selection)
- STEP 2: Base Models Training (RF, GB, SVM)
- STEP 3: Stacking Meta-Learner
- STEP 4: DRI Computation
- STEP 5: Explainability
- STEP 6: Save Artifacts

Usage:
    python train_pipeline.py [--data-path PATH] [--disaster-type TYPE]
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# Import pipeline modules
import config
from data_generator import generate_unified_dataset, save_sample_dataset
from preprocessing import prepare_data, DisasterPreprocessor
from base_models import train_base_models, BaseModelTrainer
from stacking_model import train_stacking_model, StackingEnsemble
from dri_calculator import compute_dri_from_ensemble, DRICalculator
from explainability import compute_explainability, ExplainabilityEngine


def print_header():
    """Print pipeline header."""
    print("\n" + "="*70)
    print("   ENSEMBLE-BASED HYBRID DISASTER PREDICTION SYSTEM")
    print("   ML Training Pipeline v1.0")
    print("="*70)
    print(f"   Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Output directory: {config.MODELS_DIR}")
    print("="*70)


def print_footer(metrics):
    """Print pipeline summary footer."""
    print("\n" + "="*70)
    print("   TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"\n   Final Model Performance (Stacking Meta-Learner):")
    print(f"   ├─ Accuracy:  {metrics['stacking']['accuracy']:.1f}%")
    print(f"   ├─ Precision: {metrics['stacking']['precision']:.1f}%")
    print(f"   ├─ Recall:    {metrics['stacking']['recall']:.1f}%")
    print(f"   ├─ F1-Score:  {metrics['stacking']['f1_score']:.1f}%")
    print(f"   └─ ROC-AUC:   {metrics['stacking']['roc_auc']:.1f}%")
    print(f"\n   Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")


def save_all_metrics(
    base_metrics: dict,
    stacking_metrics: dict,
    dri_stats: dict,
    explainability_results: dict
):
    """
    Save all metrics to JSON file.
    
    Parameters:
    -----------
    base_metrics : dict
        Metrics from base models
    stacking_metrics : dict
        Metrics from stacking model
    dri_stats : dict
        DRI statistics
    explainability_results : dict
        Explainability results
    """
    print("\n" + "="*60)
    print("STEP 6: SAVING ARTIFACTS")
    print("="*60)
    
    # Combine all metrics
    all_metrics = {
        'training_timestamp': datetime.now().isoformat(),
        'model_version': 'v1.0.0',
        'base_models': base_metrics,
        'stacking_model': stacking_metrics,
        'dri_configuration': {
            'stacked_weight': config.DRI_WEIGHTS['stacked_probability_weight'],
            'base_weight': config.DRI_WEIGHTS['average_base_probability_weight'],
            'risk_thresholds': config.RISK_THRESHOLDS
        },
        'dri_statistics': dri_stats,
        'feature_importance': explainability_results.get('feature_importance', {}),
        'preprocessing': {
            'missing_value_handling': 'median_imputation',
            'normalization': 'min_max_scaler',
            'train_test_split': '80/20',
            'stratified': True
        }
    }
    
    # Convert any numpy types to Python types
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj
    
    all_metrics = convert_numpy(all_metrics)
    
    # Save to JSON
    with open(config.METRICS_PATH, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"\n  Saved Artifacts:")
    print(f"  ├─ {config.PREPROCESSING_PATH.name}")
    print(f"  ├─ {config.RF_MODEL_PATH.name}")
    print(f"  ├─ {config.GB_MODEL_PATH.name}")
    print(f"  ├─ {config.SVM_MODEL_PATH.name}")
    print(f"  ├─ {config.STACKING_MODEL_PATH.name}")
    print(f"  ├─ {config.METRICS_PATH.name}")
    print(f"  └─ {config.FEATURE_IMPORTANCE_PATH.name}")
    
    print(f"\n  Output folder: {config.MODELS_DIR}/")
    print(f"\n  ✓ All artifacts saved successfully")


def run_training_pipeline(
    data_path: str = None,
    disaster_type: str = 'all',
    generate_data: bool = True
) -> dict:
    """
    Run the complete training pipeline.
    
    Parameters:
    -----------
    data_path : str, optional
        Path to the dataset CSV file
    disaster_type : str
        Type of disaster to train for ('Flood', 'Earthquake', or 'all')
    generate_data : bool
        Whether to generate synthetic data if no data_path provided
        
    Returns:
    --------
    dict
        Complete metrics and results
    """
    print_header()
    
    # =========================================================================
    # LOAD OR GENERATE DATA
    # =========================================================================
    print("\n  Loading Dataset...")
    
    if data_path and Path(data_path).exists():
        df = pd.read_csv(data_path)
        print(f"  ✓ Loaded dataset from: {data_path}")
    elif generate_data:
        print("  → No dataset found. Generating synthetic data...")
        df = save_sample_dataset()
    else:
        raise FileNotFoundError(f"Dataset not found at: {data_path}")
    
    print(f"  → Dataset shape: {df.shape}")
    print(f"  → Target distribution: {df[config.TARGET_COLUMN].value_counts().to_dict()}")
    
    # =========================================================================
    # STEP 1: PREPROCESSING
    # =========================================================================
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(
        df, 
        disaster_type=disaster_type,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE
    )
    
    # =========================================================================
    # STEP 2: BASE MODELS TRAINING
    # =========================================================================
    trainer, base_metrics = train_base_models(X_train, y_train, X_test, y_test)
    
    # =========================================================================
    # STEP 3: STACKING META-LEARNER
    # =========================================================================
    ensemble, stacking_metrics = train_stacking_model(X_train, y_train, X_test, y_test)
    
    # =========================================================================
    # STEP 4: DRI COMPUTATION
    # =========================================================================
    dri_results = compute_dri_from_ensemble(ensemble, X_test)
    dri_stats = {
        'mean_dri': float(dri_results['dri'].mean()),
        'std_dri': float(dri_results['dri'].std()),
        'min_dri': float(dri_results['dri'].min()),
        'max_dri': float(dri_results['dri'].max()),
        'risk_distribution': dri_results['risk_level'].value_counts().to_dict()
    }
    
    # =========================================================================
    # STEP 5: EXPLAINABILITY
    # =========================================================================
    explainer, explainability_results = compute_explainability(
        trainer.rf_model,
        ensemble.stacking_model,
        X_train,
        X_test
    )
    
    # =========================================================================
    # STEP 6: SAVE ALL ARTIFACTS
    # =========================================================================
    save_all_metrics(
        base_metrics,
        stacking_metrics,
        dri_stats,
        explainability_results
    )
    
    # Combine all metrics for return
    all_metrics = {
        'base_models': base_metrics,
        'stacking': stacking_metrics,
        'dri': dri_stats,
        'explainability': explainability_results
    }
    
    print_footer(all_metrics)
    
    return all_metrics


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Train Ensemble-Based Hybrid Disaster Prediction System'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Path to the dataset CSV file'
    )
    
    parser.add_argument(
        '--disaster-type',
        type=str,
        default='all',
        choices=['Flood', 'Earthquake', 'all'],
        help='Type of disaster to train for'
    )
    
    parser.add_argument(
        '--no-generate',
        action='store_true',
        help='Do not generate synthetic data if dataset not found'
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    metrics = run_training_pipeline(
        data_path=args.data_path,
        disaster_type=args.disaster_type,
        generate_data=not args.no_generate
    )
    
    return metrics


if __name__ == "__main__":
    main()
