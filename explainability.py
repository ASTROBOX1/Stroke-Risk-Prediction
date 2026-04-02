"""
===============================================================================
 SHAP Explainability Module for Stroke Prediction
 Company: Healthcare Analytics Division
 Author: Data Science Team
 Date: 2026-04-03
 
 Provides model-agnostic explainability using SHAP (SHapley Additive exPlanations)
 Supports:
   - Global feature importance
   - Local explanations for individual predictions
   - Dependence plots
   - Force plots for clinical interpretability
===============================================================================
"""

import numpy as np
import pandas as pd
import json
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import logging

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

import joblib
from utils import resolve_path

logger = logging.getLogger(__name__)


@dataclass
class ExplanationResult:
    """Container for SHAP explanation results."""
    patient_id: Optional[str]
    prediction: int
    probability: float
    base_value: float
    shap_values: Dict[str, float]
    top_features: List[Tuple[str, float]]
    risk_factors: List[str]
    protective_factors: List[str]
    explanation_text: str


class StrokeExplainer:
    """
    SHAP-based explainer for stroke prediction models.
    
    Usage:
        explainer = StrokeExplainer(model_path, X_train_sample)
        explanation = explainer.explain_prediction(patient_data, patient_id="P123")
    """
    
    def __init__(
        self, 
        model_path: str,
        X_background: pd.DataFrame,
        max_background_size: int = 100
    ):
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required. Install with: pip install shap")
        
        self.model_path = resolve_path(model_path)
        self.model = joblib.load(self.model_path)
        
        # Use a sample of background data for SHAP
        if len(X_background) > max_background_size:
            X_background = X_background.sample(max_background_size, random_state=42)
        self.X_background = X_background
        
        # Initialize SHAP explainer
        self._init_explainer()
        
        logger.info(f"StrokeExplainer initialized with {len(X_background)} background samples")
    
    def _init_explainer(self):
        """Initialize the appropriate SHAP explainer based on model type."""
        classifier = self.model.named_steps.get('classifier')
        
        if classifier is None:
            raise ValueError("Model pipeline doesn't have a 'classifier' step")
        
        # Tree-based models use TreeExplainer (faster)
        tree_based = [
            'XGBClassifier', 'RandomForestClassifier', 
            'HistGradientBoostingClassifier', 'GradientBoostingClassifier',
            'BalancedRandomForestClassifier'
        ]
        
        model_type = type(classifier).__name__
        
        if model_type in tree_based or hasattr(classifier, 'feature_importances_'):
            self.explainer = shap.TreeExplainer(classifier)
            logger.info(f"Using TreeExplainer for {model_type}")
        else:
            # Use KernelExplainer for other models (slower but model-agnostic)
            self.explainer = shap.KernelExplainer(
                classifier.predict_proba, 
                self.X_background
            )
            logger.info(f"Using KernelExplainer for {model_type}")
    
    def explain_prediction(
        self, 
        patient_data: Dict[str, Any],
        patient_id: Optional[str] = None
    ) -> ExplanationResult:
        """
        Generate a comprehensive explanation for a single prediction.
        
        Args:
            patient_data: Dictionary with patient features
            patient_id: Optional patient identifier
            
        Returns:
            ExplanationResult with detailed explanation
        """
        # Convert to DataFrame
        df = pd.DataFrame([patient_data])
        
        # Get prediction
        prediction = int(self.model.predict(df)[0])
        probability = float(self.model.predict_proba(df)[0, 1])
        
        # Compute SHAP values
        shap_values = self.explainer.shap_values(df)
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            # Multi-class: take class 1 (stroke)
            shap_vals = shap_values[1][0]
            base_value = self.explainer.expected_value[1]
        elif len(shap_values.shape) == 2:
            # Binary classification with 2D output
            shap_vals = shap_values[0]
            base_value = self.explainer.expected_value
        else:
            shap_vals = shap_values[0]
            base_value = self.explainer.expected_value
        
        # Get feature names
        feature_names = self._get_feature_names()
        
        # Create SHAP value dictionary
        shap_dict = dict(zip(feature_names, shap_vals))
        
        # Identify top features
        sorted_features = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        top_features = sorted_features[:5]
        
        # Identify risk and protective factors
        risk_factors = [f for f, v in shap_dict.items() if v > 0.01]
        protective_factors = [f for f, v in shap_dict.items() if v < -0.01]
        
        # Generate natural language explanation
        explanation_text = self._generate_explanation_text(
            prediction, probability, top_features, risk_factors, protective_factors
        )
        
        return ExplanationResult(
            patient_id=patient_id,
            prediction=prediction,
            probability=probability,
            base_value=base_value,
            shap_values=shap_dict,
            top_features=top_features,
            risk_factors=risk_factors,
            protective_factors=protective_factors,
            explanation_text=explanation_text
        )
    
    def _get_feature_names(self) -> List[str]:
        """Extract feature names from the preprocessor."""
        try:
            preprocessor = self.model.named_steps['preprocessor']
            ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
            
            from ml_pipeline import CATEGORICAL_COLS, NUMERICAL_COLS
            cat_features = list(ohe.get_feature_names_out(CATEGORICAL_COLS))
            return NUMERICAL_COLS + cat_features
        except Exception as e:
            logger.warning(f"Could not extract feature names: {e}")
            return [f"feature_{i}" for i in range(len(self.X_background.columns))]
    
    def _generate_explanation_text(
        self,
        prediction: int,
        probability: float,
        top_features: List[Tuple[str, float]],
        risk_factors: List[str],
        protective_factors: List[str]
    ) -> str:
        """Generate a human-readable explanation."""
        
        risk_level = "HIGH" if probability > 0.4 else "MODERATE" if probability > 0.15 else "LOW"
        
        if prediction == 1:
            text = f"This patient has a **{risk_level}** predicted stroke risk ({probability:.1%}).\n\n"
            text += "**Key Risk Factors:**\n"
            for feature, impact in top_features[:3]:
                if impact > 0:
                    text += f"• {self._format_feature_name(feature)} increases risk\n"
        else:
            text = f"This patient has a **{risk_level}** predicted stroke risk ({probability:.1%}).\n\n"
            text += "**Protective Factors:**\n"
            for feature, impact in top_features[:3]:
                if impact < 0:
                    text += f"• {self._format_feature_name(feature)} decreases risk\n"
        
        if risk_factors:
            text += f"\n**Additional risk factors:** {', '.join(risk_factors[:3])}"
        
        return text
    
    def _format_feature_name(self, feature: str) -> str:
        """Format feature name for display."""
        # Handle one-hot encoded features
        if '_' in feature:
            parts = feature.split('_')
            if len(parts) >= 2:
                return f"{parts[0]} = {' '.join(parts[1:])}"
        return feature.replace('_', ' ').title()
    
    def get_global_importance(self, X_sample: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Compute global feature importance using SHAP.
        
        Args:
            X_sample: Sample data for computing importance. If None, uses background data.
            
        Returns:
            Dictionary of feature names to importance scores
        """
        if X_sample is None:
            X_sample = self.X_background
        
        # Compute SHAP values for all samples
        shap_values = self.explainer.shap_values(X_sample)
        
        # Handle different output formats
        if isinstance(shap_values, list):
            shap_vals = np.abs(shap_values[1])
        else:
            shap_vals = np.abs(shap_values)
        
        # Mean absolute SHAP value per feature
        importance = shap_vals.mean(axis=0)
        
        feature_names = self._get_feature_names()
        importance_dict = dict(zip(feature_names, [float(v) for v in importance]))
        
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    def create_force_plot_data(
        self, 
        patient_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create data for a SHAP force plot (for visualization).
        
        Returns:
            Dictionary with data needed for force plot visualization
        """
        df = pd.DataFrame([patient_data])
        shap_values = self.explainer.shap_values(df)
        
        if isinstance(shap_values, list):
            shap_vals = shap_values[1][0]
            base_value = self.explainer.expected_value[1]
        else:
            shap_vals = shap_values[0] if len(shap_values.shape) > 1 else shap_values
            base_value = self.explainer.expected_value
        
        feature_names = self._get_feature_names()
        
        return {
            'base_value': float(base_value),
            'shap_values': [float(v) for v in shap_vals],
            'feature_names': feature_names,
            'feature_values': df.iloc[0].to_dict(),
            'prediction': float(self.model.predict_proba(df)[0, 1])
        }


def load_explainer(
    model_path: str = "models/best_stroke_model.joblib",
    data_path: str = "data/healthcare-dataset-stroke-data.csv",
    sample_size: int = 100
) -> StrokeExplainer:
    """
    Factory function to load a StrokeExplainer with training data as background.
    
    Args:
        model_path: Path to the trained model
        data_path: Path to the training data
        sample_size: Number of samples to use as background
        
    Returns:
        Initialized StrokeExplainer
    """
    # Load background data
    data_path = resolve_path(data_path)
    df = pd.read_csv(data_path)
    
    # Preprocess (simplified version - match training)
    df = df[df['gender'] != 'Other']
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
    if 'stroke' in df.columns:
        df = df.drop('stroke', axis=1)
    
    # Use sample as background
    X_background = df.sample(min(sample_size, len(df)), random_state=42)
    
    return StrokeExplainer(model_path, X_background)


# ─────────────────────────────────────────────────────────────
# API Integration Helper
# ─────────────────────────────────────────────────────────────

def explain_prediction_api(
    model,
    patient_data: Dict[str, Any],
    X_background: pd.DataFrame,
    patient_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate explanation for API response.
    
    This is a lightweight version that doesn't require loading SHAP
    if it's not available - falls back to feature importance.
    """
    
    # Get prediction
    df = pd.DataFrame([patient_data])
    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0, 1])
    
    result = {
        'patient_id': patient_id,
        'prediction': prediction,
        'probability': round(probability, 4),
        'risk_level': 'HIGH' if probability > 0.4 else 'MODERATE' if probability > 0.15 else 'LOW'
    }
    
    if SHAP_AVAILABLE:
        try:
            explainer = StrokeExplainer(model, X_background)
            explanation = explainer.explain_prediction(patient_data, patient_id)
            
            result.update({
                'explanation': {
                    'text': explanation.explanation_text,
                    'top_features': [
                        {'feature': f, 'impact': round(float(i), 4)}
                        for f, i in explanation.top_features
                    ],
                    'risk_factors': explanation.risk_factors[:5],
                    'protective_factors': explanation.protective_factors[:5]
                }
            })
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")
            result['explanation'] = {'error': 'Explanation unavailable'}
    else:
        # Fallback: use model's feature importance
        try:
            classifier = model.named_steps['classifier']
            if hasattr(classifier, 'feature_importances_'):
                importances = classifier.feature_importances_
            elif hasattr(classifier, 'coef_'):
                importances = np.abs(classifier.coef_[0])
            else:
                importances = None
            
            if importances is not None:
                from ml_pipeline import NUMERICAL_COLS, CATEGORICAL_COLS
                feature_names = NUMERICAL_COLS + CATEGORICAL_COLS
                importance_dict = dict(zip(feature_names, importances[:len(feature_names)]))
                top_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:5]
                
                result['explanation'] = {
                    'top_features': [
                        {'feature': f, 'importance': round(float(i), 4)}
                        for f, i in top_features
                    ],
                    'note': 'SHAP not available, showing feature importance instead'
                }
        except Exception as e:
            result['explanation'] = {'error': 'Feature importance unavailable'}
    
    return result
