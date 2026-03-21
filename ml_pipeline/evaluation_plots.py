"""
Evaluation Plots Module for Ensemble-Based Hybrid Disaster Prediction System

Generates visualization plots for model evaluation:
- Confusion Matrix
- ROC Curve
- Feature Importance Bar Chart
- Model Comparison Chart
- Risk Distribution Histogram
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from typing import Dict, List, Optional, Any
from pathlib import Path

import config


class EvaluationPlotter:
    """
    Generate evaluation plots for the disaster prediction model.
    """
    
    def __init__(self, output_dir: str = None):
        """
        Initialize the plotter.
        
        Parameters:
        -----------
        output_dir : str, optional
            Directory to save plots
        """
        self.output_dir = Path(output_dir) if output_dir else config.MODELS_DIR / "plots"
        self.output_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        self.colors = {
            'primary': '#1E3A8A',
            'secondary': '#3B82F6',
            'success': '#15803D',
            'warning': '#CA8A04',
            'danger': '#B91C1C',
            'light': '#F3F4F6'
        }
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = 'Stacking Meta-Learner',
        save: bool = True
    ) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        model_name : str
            Name of the model
        save : bool
            Whether to save the plot
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create heatmap
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Count', rotation=-90, va="bottom")
        
        # Set labels
        classes = ['No Risk (0)', 'Risk (1)']
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=classes,
            yticklabels=classes,
            xlabel='Predicted Label',
            ylabel='True Label',
            title=f'Confusion Matrix - {model_name}'
        )
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontsize=14)
        
        # Add metrics
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics_text = f'Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}'
        fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=10)
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / 'confusion_matrix.png'
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"  ✓ Confusion matrix saved to: {path}")
        
        return fig
    
    def plot_roc_curves(
        self,
        y_true: np.ndarray,
        model_predictions: Dict[str, np.ndarray],
        save: bool = True
    ) -> plt.Figure:
        """
        Plot ROC curves for multiple models.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        model_predictions : Dict[str, np.ndarray]
            Dictionary of model names to predicted probabilities
        save : bool
            Whether to save the plot
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = [self.colors['primary'], self.colors['success'], 
                  self.colors['warning'], self.colors['danger']]
        
        for idx, (model_name, y_pred_proba) in enumerate(model_predictions.items()):
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            ax.plot(
                fpr, tpr,
                color=colors[idx % len(colors)],
                lw=2,
                label=f'{model_name} (AUC = {roc_auc:.3f})'
            )
        
        # Diagonal line
        ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random (AUC = 0.500)')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve Comparison')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / 'roc_curves.png'
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"  ✓ ROC curves saved to: {path}")
        
        return fig
    
    def plot_feature_importance(
        self,
        feature_importance: Dict[str, float],
        top_k: int = 10,
        save: bool = True
    ) -> plt.Figure:
        """
        Plot horizontal bar chart of feature importance.
        
        Parameters:
        -----------
        feature_importance : Dict[str, float]
            Feature importance scores
        top_k : int
            Number of top features to display
        save : bool
            Whether to save the plot
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure
        """
        # Sort and take top k
        sorted_features = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_k]
        )
        
        features = list(sorted_features.keys())
        importance = list(sorted_features.values())
        
        # Reverse for horizontal bar chart (highest at top)
        features = features[::-1]
        importance = importance[::-1]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.barh(features, importance, color=self.colors['primary'], height=0.6)
        
        # Add value labels
        for bar, imp in zip(bars, importance):
            width = bar.get_width()
            ax.annotate(f'{imp:.1f}%',
                       xy=(width, bar.get_y() + bar.get_height() / 2),
                       xytext=(5, 0),
                       textcoords="offset points",
                       ha='left', va='center',
                       fontsize=10)
        
        ax.set_xlabel('Importance (%)')
        ax.set_title('Feature Importance (Random Forest)')
        ax.set_xlim(0, max(importance) * 1.15)
        ax.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / 'feature_importance.png'
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"  ✓ Feature importance saved to: {path}")
        
        return fig
    
    def plot_model_comparison(
        self,
        metrics: Dict[str, Dict[str, float]],
        save: bool = True
    ) -> plt.Figure:
        """
        Plot model performance comparison.
        
        Parameters:
        -----------
        metrics : Dict[str, Dict[str, float]]
            Dictionary of model metrics
        save : bool
            Whether to save the plot
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure
        """
        models = list(metrics.keys())
        metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        x = np.arange(len(models))
        width = 0.15
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = ['#1E3A8A', '#3B82F6', '#60A5FA', '#93C5FD', '#BFDBFE']
        
        for i, metric in enumerate(metric_names):
            values = [metrics[model].get(metric, 0) for model in models]
            bars = ax.bar(x + i * width, values, width, label=metric.replace('_', ' ').title(),
                         color=colors[i])
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Score (%)')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in models], rotation=15)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.set_ylim(0, 105)
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / 'model_comparison.png'
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"  ✓ Model comparison saved to: {path}")
        
        return fig
    
    def plot_dri_distribution(
        self,
        dri_values: np.ndarray,
        save: bool = True
    ) -> plt.Figure:
        """
        Plot DRI distribution histogram.
        
        Parameters:
        -----------
        dri_values : np.ndarray
            Array of DRI values
        save : bool
            Whether to save the plot
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create histogram
        bins = np.linspace(0, 1, 21)
        n, bins_out, patches = ax.hist(dri_values, bins=bins, edgecolor='white', linewidth=0.5)
        
        # Color bins by risk level
        for i, (patch, b) in enumerate(zip(patches, bins_out[:-1])):
            if b < 0.33:
                patch.set_facecolor(self.colors['success'])
            elif b < 0.66:
                patch.set_facecolor(self.colors['warning'])
            else:
                patch.set_facecolor(self.colors['danger'])
        
        # Add threshold lines
        ax.axvline(x=0.33, color='gray', linestyle='--', linewidth=1.5, label='Low/Moderate Threshold')
        ax.axvline(x=0.66, color='gray', linestyle='-.', linewidth=1.5, label='Moderate/High Threshold')
        
        # Add legend for risk levels
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=self.colors['success'], label='Low Risk (0-0.33)'),
            Patch(facecolor=self.colors['warning'], label='Moderate Risk (0.34-0.66)'),
            Patch(facecolor=self.colors['danger'], label='High Risk (0.67-1.00)')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        ax.set_xlabel('Disaster Risk Index (DRI)')
        ax.set_ylabel('Frequency')
        ax.set_title('DRI Distribution')
        ax.set_xlim(0, 1)
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add statistics
        stats_text = f'Mean: {np.mean(dri_values):.3f} | Std: {np.std(dri_values):.3f}'
        ax.text(0.5, 0.95, stats_text, transform=ax.transAxes, ha='center',
               fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / 'dri_distribution.png'
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"  ✓ DRI distribution saved to: {path}")
        
        return fig
    
    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        model_name: str = 'Stacking Meta-Learner',
        save: bool = True
    ) -> plt.Figure:
        """
        Plot precision-recall curve.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred_proba : np.ndarray
            Predicted probabilities
        model_name : str
            Name of the model
        save : bool
            Whether to save the plot
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(recall, precision, color=self.colors['primary'], lw=2)
        ax.fill_between(recall, precision, alpha=0.2, color=self.colors['primary'])
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall Curve - {model_name}')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.grid(True, alpha=0.3)
        
        # Calculate and display average precision
        from sklearn.metrics import average_precision_score
        ap = average_precision_score(y_true, y_pred_proba)
        ax.text(0.6, 0.1, f'Average Precision: {ap:.3f}',
               transform=ax.transAxes, fontsize=11,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / 'precision_recall_curve.png'
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"  ✓ Precision-recall curve saved to: {path}")
        
        return fig
    
    def generate_all_plots(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_predictions: Dict[str, np.ndarray],
        metrics: Dict[str, Dict[str, float]],
        feature_importance: Dict[str, float],
        dri_values: np.ndarray
    ):
        """
        Generate all evaluation plots.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        model_predictions : Dict[str, np.ndarray]
            Dictionary of model predictions
        metrics : Dict[str, Dict[str, float]]
            Dictionary of model metrics
        feature_importance : Dict[str, float]
            Feature importance scores
        dri_values : np.ndarray
            Array of DRI values
        """
        print("\n  Generating evaluation plots...")
        
        # Get stacking predictions for PR curve
        stacking_proba = model_predictions.get('Stacking', model_predictions.get('stacking', None))
        
        self.plot_confusion_matrix(y_true, y_pred)
        self.plot_roc_curves(y_true, model_predictions)
        self.plot_feature_importance(feature_importance)
        self.plot_model_comparison(metrics)
        self.plot_dri_distribution(dri_values)
        
        if stacking_proba is not None:
            self.plot_precision_recall_curve(y_true, stacking_proba)
        
        print(f"\n  ✓ All plots saved to: {self.output_dir}/")
        
        plt.close('all')


if __name__ == "__main__":
    # Example usage
    print("Generating sample plots...")
    
    plotter = EvaluationPlotter()
    
    # Sample data
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 200)
    y_pred = np.random.randint(0, 2, 200)
    
    model_preds = {
        'Random Forest': np.random.random(200),
        'Gradient Boosting': np.random.random(200),
        'SVM': np.random.random(200),
        'Stacking': np.random.random(200)
    }
    
    metrics = {
        'random_forest': {'accuracy': 91, 'precision': 89, 'recall': 92, 'f1_score': 90, 'roc_auc': 94},
        'gradient_boosting': {'accuracy': 93, 'precision': 91, 'recall': 94, 'f1_score': 92, 'roc_auc': 96},
        'svm': {'accuracy': 89, 'precision': 87, 'recall': 90, 'f1_score': 88, 'roc_auc': 92},
        'stacking': {'accuracy': 95, 'precision': 93, 'recall': 96, 'f1_score': 94, 'roc_auc': 97}
    }
    
    feature_importance = {
        'rainfall': 45,
        'soil_moisture': 30,
        'humidity': 15,
        'temperature': 10
    }
    
    dri_values = np.random.beta(2, 3, 200)
    
    plotter.generate_all_plots(
        y_true, y_pred, model_preds, metrics, feature_importance, dri_values
    )
