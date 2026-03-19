"""
===============================================================================
 Stroke Prediction — Model Training Pipeline
 Company: Healthcare Analytics Division
 Author: Data Science Team
 Date: 2026-03-15
 Description: Train, evaluate, and export ML models for stroke risk prediction
===============================================================================
"""

import pandas as pd
import numpy as np
import json
import os
import sys
import logging
import warnings
from datetime import datetime
from typing import Tuple, Dict, Any, Optional

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, precision_score, recall_score, f1_score,
    roc_curve, precision_recall_curve
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
import joblib
from utils import resolve_path

warnings.filterwarnings('ignore')

TRAIN_LOG_PATH = resolve_path('logs/train_model.log')
TRAIN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(TRAIN_LOG_PATH, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# 1. Data Loading
# ─────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load and return the dataset.

    Args:
        filepath: Path to the CSV data file

    Returns:
        Loaded DataFrame

    Raises:
        FileNotFoundError: If file doesn't exist
        pd.errors.ParserError: If CSV is malformed
    """
    try:
        resolved_path = resolve_path(filepath)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Data file not found: {resolved_path}")

        logger.info(f"Loading data from {resolved_path}...")
        df = pd.read_csv(resolved_path)
        logger.info(f"✅ Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
        return df
    except FileNotFoundError as e:
        logger.error(str(e))
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading data: {str(e)}")
        raise


# ─────────────────────────────────────────────────────────────
# 2. Preprocessing
# ─────────────────────────────────────────────────────────────

CATEGORICAL_COLS = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
NUMERICAL_COLS = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    """
    Clean data and build a ColumnTransformer preprocessing pipeline.

    Args:
        df: Raw DataFrame

    Returns:
        Tuple of (X, y, preprocessor)

    Raises:
        ValueError: If required columns are missing
    """
    try:
        logger.info("Preprocessing data...")

        # Validate required columns
        required_cols = CATEGORICAL_COLS + NUMERICAL_COLS + ['stroke']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        df = df.copy()

        # Remove rare 'Other' gender category
        initial_rows = len(df)
        df = df[df['gender'] != 'Other']
        removed_rows = initial_rows - len(df)
        if removed_rows > 0:
            logger.info(f"Removed {removed_rows} rows with gender='Other'")

        # Drop ID column
        if 'id' in df.columns:
            df = df.drop('id', axis=1)
            logger.info("Dropped 'id' column")

        X = df.drop('stroke', axis=1)
        y = df['stroke']

        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(transformers=[
            ('num', numerical_transformer, NUMERICAL_COLS),
            ('cat', categorical_transformer, CATEGORICAL_COLS)
        ])

        logger.info(f"✅ Preprocessing complete. Shape: {X.shape}")
        return X, y, preprocessor

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        raise


# ─────────────────────────────────────────────────────────────
# 3. Model Definitions
# ─────────────────────────────────────────────────────────────

def build_models(preprocessor: ColumnTransformer) -> Dict[str, ImbPipeline]:
    """
    Build and return a dictionary of named model pipelines.

    Args:
        preprocessor: ColumnTransformer for data preprocessing

    Returns:
        Dictionary of model pipelines

    Raises:
        Exception: If model initialization fails
    """
    try:
        logger.info("Building model pipelines...")

        models = {
            'Random Forest': ImbPipeline(steps=[
                ('preprocessor', preprocessor),
                ('smote', SMOTE(random_state=42)),
                ('classifier', RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    random_state=42,
                    class_weight='balanced'
                ))
            ]),
            'XGBoost': ImbPipeline(steps=[
                ('preprocessor', preprocessor),
                ('smote', SMOTE(random_state=42)),
                ('classifier', XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric='logloss'
                ))
            ]),
            'Logistic Regression': ImbPipeline(steps=[
                ('preprocessor', preprocessor),
                ('smote', SMOTE(random_state=42)),
                ('classifier', LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                    class_weight='balanced'
                ))
            ]),
        }

        logger.info(f"✅ Built {len(models)} model pipelines")
        return models

    except Exception as e:
        logger.error(f"Error building models: {str(e)}")
        raise


# ─────────────────────────────────────────────────────────────
# 4. Training
# ─────────────────────────────────────────────────────────────

def train_models(models: Dict[str, ImbPipeline], X_train: pd.DataFrame,
                 y_train: pd.Series) -> Dict[str, ImbPipeline]:
    """
    Fit all models and return trained versions.

    Args:
        models: Dictionary of model pipelines to train
        X_train: Training feature data
        y_train: Training target data

    Returns:
        Dictionary of trained models

    Raises:
        Exception: If training fails for any model
    """
    trained = {}
    try:
        for name, pipeline in models.items():
            try:
                logger.info(f"Training {name}...")
                pipeline.fit(X_train, y_train)
                trained[name] = pipeline
                logger.info(f"✅ {name} trained successfully")
            except Exception as e:
                logger.error(f"Error training {name}: {str(e)}")
                raise

        return trained
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise


# ─────────────────────────────────────────────────────────────
# 5. Evaluation
# ─────────────────────────────────────────────────────────────

def evaluate_models(trained_models: Dict[str, ImbPipeline], X_test: pd.DataFrame,
                    y_test: pd.Series) -> Tuple[Dict[str, Dict[str, Any]], str, ImbPipeline]:
    """
    Evaluate all trained models and return metrics + best model.

    Args:
        trained_models: Dictionary of trained models
        X_test: Test feature data
        y_test: Test target data

    Returns:
        Tuple of (metrics_dict, best_model_name, best_model)

    Raises:
        Exception: If evaluation fails
    """
    try:
        logger.info("Evaluating models...")
        all_metrics = {}

        for name, model in trained_models.items():
            try:
                preds = model.predict(X_test)
                proba = model.predict_proba(X_test)[:, 1]

                acc = accuracy_score(y_test, preds)
                prec = precision_score(y_test, preds, zero_division=0)
                rec = recall_score(y_test, preds, zero_division=0)
                f1 = f1_score(y_test, preds, zero_division=0)
                auc = roc_auc_score(y_test, proba)
                cm = confusion_matrix(y_test, preds).tolist()
                report = classification_report(y_test, preds, output_dict=True)

                # ROC & PR curves
                fpr, tpr, _ = roc_curve(y_test, proba)
                pr_precision, pr_recall, _ = precision_recall_curve(y_test, proba)

                all_metrics[name] = {
                    'accuracy': round(acc, 4),
                    'precision': round(prec, 4),
                    'recall': round(rec, 4),
                    'f1_score': round(f1, 4),
                    'auc_roc': round(auc, 4),
                    'confusion_matrix': cm,
                    'classification_report': {
                        k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                             for kk, vv in v.items()} if isinstance(v, dict) else round(v, 4)
                        for k, v in report.items()
                    },
                    'roc_curve': {
                        'fpr': fpr.tolist(),
                        'tpr': tpr.tolist()
                    },
                    'pr_curve': {
                        'precision': pr_precision.tolist(),
                        'recall': pr_recall.tolist()
                    }
                }

                logger.info(f"  ── {name} ──")
                logger.info(f"     Accuracy : {acc:.4f}")
                logger.info(f"     Precision: {prec:.4f}")
                logger.info(f"     Recall   : {rec:.4f}")
                logger.info(f"     F1-Score : {f1:.4f}")
                logger.info(f"     AUC-ROC  : {auc:.4f}")

            except Exception as e:
                logger.error(f"Error evaluating {name}: {str(e)}")
                raise

        # Best model = highest AUC-ROC
        best_name = max(all_metrics, key=lambda k: all_metrics[k]['auc_roc'])
        logger.info(f"🏆 Best model: {best_name} (AUC = {all_metrics[best_name]['auc_roc']:.4f})")

        return all_metrics, best_name, trained_models[best_name]

    except Exception as e:
        logger.error(f"Error during model evaluation: {str(e)}")
        raise


# ─────────────────────────────────────────────────────────────
# 6. Feature Importance
# ─────────────────────────────────────────────────────────────

def extract_feature_importance(model: ImbPipeline, preprocessor: ColumnTransformer,
                              X: pd.DataFrame) -> Dict[str, float]:
    """
    Extract feature names and importances from the best model.

    Args:
        model: Trained model pipeline
        preprocessor: ColumnTransformer used for preprocessing
        X: Training feature data

    Returns:
        Dictionary of feature names to importance scores

    Raises:
        Exception: If feature extraction fails
    """
    try:
        logger.info("Extracting feature importance...")

        # Get feature names after preprocessing
        cat_features = []
        num_features = list(NUMERICAL_COLS)

        try:
            ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
            cat_features = list(ohe.get_feature_names_out(CATEGORICAL_COLS))
        except Exception as e:
            logger.warning(f"Could not extract categorical feature names: {str(e)}")
            cat_features = list(CATEGORICAL_COLS)

        all_features = num_features + cat_features

        # Try to get importances from the classifier step
        clf = model.named_steps.get('classifier') or model[-1]

        if hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
        elif hasattr(clf, 'coef_'):
            importances = np.abs(clf.coef_[0])
        else:
            logger.warning("Model does not have feature importance attribute")
            return {}

        # Match lengths
        if len(importances) == len(all_features):
            imp_dict = dict(zip(all_features, [round(float(v), 6) for v in importances]))
            imp_dict = dict(sorted(imp_dict.items(), key=lambda x: x[1], reverse=True))
            logger.info(f"✅ Extracted importances for {len(imp_dict)} features")
            return imp_dict
        else:
            logger.warning(f"Feature count mismatch: {len(importances)} vs {len(all_features)}")
            return {f"feature_{i}": round(float(v), 6) for i, v in enumerate(importances)}

    except Exception as e:
        logger.error(f"Error extracting feature importance: {str(e)}")
        raise


# ─────────────────────────────────────────────────────────────
# 7. Save Artifacts
# ─────────────────────────────────────────────────────────────

def save_artifacts(model_dir: str, best_model: ImbPipeline, best_name: str,
                  all_metrics: Dict[str, Dict[str, Any]],
                  feature_importance: Dict[str, float]) -> None:
    """
    Persist model, metrics, and feature importance.

    Args:
        model_dir: Directory to save artifacts
        best_model: Best trained model
        best_name: Name of best model
        all_metrics: Dictionary of all model metrics
        feature_importance: Feature importance dictionary

    Raises:
        Exception: If saving fails
    """
    try:
        resolved_model_dir = resolve_path(model_dir)
        resolved_model_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Creating model directory: {resolved_model_dir}")

        # Model
        try:
            model_path = resolved_model_dir / 'best_stroke_model.joblib'
            joblib.dump(best_model, model_path)
            logger.info(f"✅ Model saved → {model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

        # Metrics
        try:
            metrics_payload = {
                'generated_at': datetime.now().isoformat(),
                'best_model': best_name,
                'models': all_metrics
            }
            metrics_path = resolved_model_dir / 'metrics.json'
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics_payload, f, indent=2)
            logger.info(f"✅ Metrics saved → {metrics_path}")
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")
            raise

        # Feature Importance
        try:
            fi_path = resolved_model_dir / 'feature_importance.json'
            with open(fi_path, 'w', encoding='utf-8') as f:
                json.dump(feature_importance, f, indent=2)
            logger.info(f"✅ Feature importance saved → {fi_path}")
        except Exception as e:
            logger.error(f"Error saving feature importance: {str(e)}")
            raise

    except Exception as e:
        logger.error(f"Error saving artifacts: {str(e)}")
        raise


# ─────────────────────────────────────────────────────────────
# 8. Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    """
    Run the complete ML training pipeline.

    Raises:
        Exception: If any step in the pipeline fails
    """
    try:
        logger.info("="*70)
        logger.info("Starting ML Training Pipeline")
        logger.info("="*70)

        DATA_PATH = "data/healthcare-dataset-stroke-data.csv"
        MODEL_DIR = "models"

        # Load data
        df = load_data(DATA_PATH)

        # Preprocess
        X, y, preprocessor = preprocess_data(df)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        logger.info(f"Train: {len(X_train):,} samples | Test: {len(X_test):,} samples")
        logger.info(f"Stroke cases in train: {int(y_train.sum())} / {len(y_train)} ({y_train.mean()*100:.1f}%)")

        # Build, train, and evaluate models
        models = build_models(preprocessor)
        trained = train_models(models, X_train, y_train)
        all_metrics, best_name, best_model = evaluate_models(trained, X_test, y_test)

        # Extract feature importance
        preprocessor_clone = preprocessor
        preprocessor_clone.fit(X_train)
        fi = extract_feature_importance(best_model, preprocessor_clone, X_train)

        # Save artifacts
        save_artifacts(MODEL_DIR, best_model, best_name, all_metrics, fi)

        logger.info("="*70)
        logger.info("✅ Pipeline complete successfully!")
        logger.info("="*70)

    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error in pipeline: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
