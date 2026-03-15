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
import warnings
from datetime import datetime

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

warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────
# 1. Data Loading
# ─────────────────────────────────────────────────────────────

def load_data(filepath):
    """Load and return the dataset."""
    print(f"📂  Loading data from {filepath} ...")
    df = pd.read_csv(filepath)
    print(f"    ✅  Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────────────────────
# 2. Preprocessing
# ─────────────────────────────────────────────────────────────

CATEGORICAL_COLS = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
NUMERICAL_COLS   = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']


def preprocess_data(df):
    """Clean data and build a ColumnTransformer preprocessing pipeline."""
    print("🔧  Preprocessing data ...")

    # Remove rare 'Other' gender category
    df = df[df['gender'] != 'Other'].copy()

    # Drop ID column
    if 'id' in df.columns:
        df = df.drop('id', axis=1)

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

    return X, y, preprocessor


# ─────────────────────────────────────────────────────────────
# 3. Model Definitions
# ─────────────────────────────────────────────────────────────

def build_models(preprocessor):
    """Return a dict of named model pipelines."""
    return {
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


# ─────────────────────────────────────────────────────────────
# 4. Training
# ─────────────────────────────────────────────────────────────

def train_models(models, X_train, y_train):
    """Fit every model and return them."""
    trained = {}
    for name, pipeline in models.items():
        print(f"🏋️  Training {name} ...")
        pipeline.fit(X_train, y_train)
        trained[name] = pipeline
    return trained


# ─────────────────────────────────────────────────────────────
# 5. Evaluation
# ─────────────────────────────────────────────────────────────

def evaluate_models(trained_models, X_test, y_test):
    """Evaluate models and return metrics dict + best model."""
    print("\n📊  Evaluating models ...\n")
    all_metrics = {}

    for name, model in trained_models.items():
        preds  = model.predict(X_test)
        proba  = model.predict_proba(X_test)[:, 1]

        acc       = accuracy_score(y_test, preds)
        prec      = precision_score(y_test, preds, zero_division=0)
        rec       = recall_score(y_test, preds, zero_division=0)
        f1        = f1_score(y_test, preds, zero_division=0)
        auc       = roc_auc_score(y_test, proba)
        cm        = confusion_matrix(y_test, preds).tolist()
        report    = classification_report(y_test, preds, output_dict=True)

        # ROC & PR curves
        fpr, tpr, _               = roc_curve(y_test, proba)
        pr_precision, pr_recall, _ = precision_recall_curve(y_test, proba)

        all_metrics[name] = {
            'accuracy':   round(acc, 4),
            'precision':  round(prec, 4),
            'recall':     round(rec, 4),
            'f1_score':   round(f1, 4),
            'auc_roc':    round(auc, 4),
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
                'recall':    pr_recall.tolist()
            }
        }

        print(f"  ── {name} ──")
        print(f"     Accuracy : {acc:.4f}")
        print(f"     Precision: {prec:.4f}")
        print(f"     Recall   : {rec:.4f}")
        print(f"     F1-Score : {f1:.4f}")
        print(f"     AUC-ROC  : {auc:.4f}")
        print()

    # Best model = highest AUC-ROC
    best_name = max(all_metrics, key=lambda k: all_metrics[k]['auc_roc'])
    print(f"🏆  Best model: {best_name} (AUC = {all_metrics[best_name]['auc_roc']:.4f})")

    return all_metrics, best_name, trained_models[best_name]


# ─────────────────────────────────────────────────────────────
# 6. Feature Importance
# ─────────────────────────────────────────────────────────────

def extract_feature_importance(model, preprocessor, X):
    """Extract feature names and importances from the best model."""
    # Get feature names after preprocessing
    cat_features = []
    num_features = list(NUMERICAL_COLS)
    try:
        ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_features = list(ohe.get_feature_names_out(CATEGORICAL_COLS))
    except Exception:
        cat_features = list(CATEGORICAL_COLS)

    all_features = num_features + cat_features

    # Try to get importances from the classifier step
    clf = model.named_steps.get('classifier') or model[-1]

    if hasattr(clf, 'feature_importances_'):
        importances = clf.feature_importances_
    elif hasattr(clf, 'coef_'):
        importances = np.abs(clf.coef_[0])
    else:
        return {}

    # Match lengths
    if len(importances) == len(all_features):
        imp_dict = dict(zip(all_features, [round(float(v), 6) for v in importances]))
        imp_dict = dict(sorted(imp_dict.items(), key=lambda x: x[1], reverse=True))
        return imp_dict
    else:
        return {f"feature_{i}": round(float(v), 6) for i, v in enumerate(importances)}


# ─────────────────────────────────────────────────────────────
# 7. Save Artifacts
# ─────────────────────────────────────────────────────────────

def save_artifacts(model_dir, best_model, best_name, all_metrics, feature_importance):
    """Persist model, metrics, and feature importance."""
    os.makedirs(model_dir, exist_ok=True)

    # Model
    model_path = os.path.join(model_dir, 'best_stroke_model.joblib')
    joblib.dump(best_model, model_path)
    print(f"\n💾  Model saved → {model_path}")

    # Metrics
    metrics_payload = {
        'generated_at': datetime.now().isoformat(),
        'best_model': best_name,
        'models': all_metrics
    }
    metrics_path = os.path.join(model_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"💾  Metrics saved → {metrics_path}")

    # Feature Importance
    fi_path = os.path.join(model_dir, 'feature_importance.json')
    with open(fi_path, 'w') as f:
        json.dump(feature_importance, f, indent=2)
    print(f"💾  Feature importance saved → {fi_path}")


# ─────────────────────────────────────────────────────────────
# 8. Main
# ─────────────────────────────────────────────────────────────

def main():
    DATA_PATH  = "data/healthcare-dataset-stroke-data.csv"
    MODEL_DIR  = "models"

    # Pipeline
    df = load_data(DATA_PATH)
    X, y, preprocessor = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n📋  Train: {len(X_train):,} samples  |  Test: {len(X_test):,} samples")
    print(f"    Stroke cases in train: {int(y_train.sum())} / {len(y_train)} "
          f"({y_train.mean()*100:.1f}%)")

    models = build_models(preprocessor)
    trained = train_models(models, X_train, y_train)
    all_metrics, best_name, best_model = evaluate_models(trained, X_test, y_test)

    # Fit preprocessor alone to extract feature names
    preprocessor_clone = preprocessor
    preprocessor_clone.fit(X_train)
    fi = extract_feature_importance(best_model, preprocessor_clone, X_train)

    save_artifacts(MODEL_DIR, best_model, best_name, all_metrics, fi)
    print("\n✅  Pipeline complete!")


if __name__ == "__main__":
    main()
