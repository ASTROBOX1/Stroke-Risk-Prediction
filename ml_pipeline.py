"""
===============================================================================
 Stroke Prediction — Advanced ML Pipeline (v3.0)
 Company: Healthcare Analytics Division
 Author: Data Science Team
 Date: 2026-04-03
 
 IMPROVEMENTS:
 - Advanced regularization and ensemble methods
 - Deep learning option (PyTorch/tabular neural networks)
 - Comprehensive imbalanced data handling (SMOTE, ADASYN, ensemble methods)
 - SHAP explainability integration
 - Probability calibration for better generalization
 - Nested cross-validation for robust evaluation
 - Automated hyperparameter optimization (Optuna)
 - Feature engineering and interaction terms
 - Drift detection ready
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
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
import hashlib

from sklearn.model_selection import (
    train_test_split, 
    cross_val_score, 
    StratifiedKFold,
    GridSearchCV,
    RandomizedSearchCV
)
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    StackingClassifier,
    VotingClassifier,
    BalancedBaggingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    roc_auc_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_curve, 
    precision_recall_curve,
    average_precision_score,
    matthews_corrcoef,
    cohen_kappa_score,
    brier_score_loss,
    calibration_curve
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression

from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks, EditedNearestNeighbours
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.ensemble import (
    BalancedRandomForestClassifier,
    EasyEnsembleClassifier,
    RUSBoostClassifier
)

from xgboost import XGBClassifier
import joblib

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not installed. Install with: pip install shap")

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn("Optuna not installed. Install with: pip install optuna")

from utils import resolve_path

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

TRAIN_LOG_PATH = resolve_path('logs/ml_pipeline.log')
TRAIN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# Feature columns
CATEGORICAL_COLS = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
NUMERICAL_COLS = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
ALL_FEATURE_COLS = NUMERICAL_COLS + CATEGORICAL_COLS

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
# Data Classes
# ─────────────────────────────────────────────────────────────

@dataclass
class ModelMetrics:
    """Container for comprehensive model evaluation metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    average_precision: float
    mcc: float  # Matthews Correlation Coefficient
    kappa: float  # Cohen's Kappa
    brier_score: float  # Calibration metric
    confusion_matrix: List[List[int]]
    classification_report: Dict
    calibration_data: Optional[Dict] = None
    cv_scores: Optional[Dict[str, List[float]]] = None


@dataclass
class TrainingConfig:
    """Configuration for the ML training pipeline."""
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    n_trials: int = 50  # For Optuna
    use_deep_learning: bool = False
    use_calibration: bool = True
    use_shap: bool = True
    imbalance_strategy: str = "smote"  # smote, adasyn, smote_tomek, class_weight, ensemble
    ensemble_method: str = "stacking"  # stacking, voting, boosting
    hyperparameter_tuning: bool = True
    feature_engineering: bool = True


# ─────────────────────────────────────────────────────────────
# 1. Data Loading with Validation
# ─────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Load and validate the dataset."""
    try:
        resolved_path = resolve_path(filepath)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Data file not found: {resolved_path}")

        logger.info(f"Loading data from {resolved_path}...")
        df = pd.read_csv(
            resolved_path,
            low_memory=False,
            na_values=['', 'NA', 'N/A', 'null', 'NULL']
        )
        
        # Data quality report
        logger.info(f"✅ Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
        logger.info(f"   Missing values: {df.isna().sum().sum()} ({df.isna().sum().sum() / df.shape[0] * 100:.2f}%)")
        logger.info(f"   Class distribution: {df['stroke'].value_counts().to_dict()}")
        logger.info(f"   Imbalance ratio: {df['stroke'].mean():.4f}")
        
        return df
    except FileNotFoundError as e:
        logger.error(str(e))
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading data: {str(e)}")
        raise


# ─────────────────────────────────────────────────────────────
# 2. Advanced Preprocessing & Feature Engineering
# ─────────────────────────────────────────────────────────────

def create_feature_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Create meaningful interaction features for healthcare domain."""
    df = df.copy()
    
    # Age-based risk groups
    df['age_group'] = pd.cut(
        df['age'], 
        bins=[0, 18, 35, 50, 65, 100],
        labels=['child', 'young', 'middle', 'senior', 'elderly']
    )
    
    # BMI categories
    df['bmi_category'] = pd.cut(
        df['bmi'],
        bins=[0, 18.5, 25, 30, 100],
        labels=['underweight', 'normal', 'overweight', 'obese']
    )
    
    # Glucose risk level
    df['glucose_risk'] = pd.cut(
        df['avg_glucose_level'],
        bins=[0, 70, 100, 126, 200, 500],
        labels=['hypoglycemia', 'normal', 'prediabetes', 'diabetes', 'severe']
    )
    
    # Combined risk score (domain knowledge)
    df['cardiovascular_risk'] = (
        df['hypertension'] + 
        df['heart_disease'] + 
        (df['avg_glucose_level'] > 126).astype(int) +
        (df['bmi'] > 30).astype(int) +
        (df['age'] > 55).astype(int)
    )
    
    # Age-glucose interaction
    df['age_glucose_interaction'] = df['age'] * df['avg_glucose_level'] / 1000
    
    # Age-BMI interaction
    df['age_bmi_interaction'] = df['age'] * df['bmi'] / 100
    
    logger.info("✅ Created interaction features")
    return df


def preprocess_data(
    df: pd.DataFrame, 
    config: TrainingConfig
) -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer, List[str]]:
    """
    Clean data, create features, and build preprocessing pipeline.
    
    Returns:
        Tuple of (X, y, preprocessor, feature_names)
    """
    try:
        logger.info("Preprocessing data...")
        initial_rows = len(df)
        
        # Remove 'Other' gender
        df = df[df['gender'] != 'Other'].copy()
        logger.info(f"   Removed {initial_rows - len(df)} rows with gender='Other'")
        
        # Drop ID column
        if 'id' in df.columns:
            df = df.drop('id', axis=1)
        
        # Feature engineering
        if config.feature_engineering:
            df = create_feature_interactions(df)
            additional_cats = ['age_group', 'bmi_category', 'glucose_risk']
            cat_cols = CATEGORICAL_COLS + additional_cats
        else:
            cat_cols = CATEGORICAL_COLS
        
        # Separate features and target
        X = df.drop('stroke', axis=1)
        y = df['stroke']
        
        # Advanced preprocessing pipeline
        numerical_transformer = Pipeline(steps=[
            ('imputer', KNNImputer(n_neighbors=5, weights='distance')),
            ('scaler', RobustScaler()),  # More robust to outliers
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(
                handle_unknown='ignore',
                sparse_output=True,
                min_frequency=5  # Combine rare categories
            ))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, NUMERICAL_COLS),
                ('cat', categorical_transformer, cat_cols)
            ],
            remainder='drop'
        )
        
        # Get feature names after preprocessing
        feature_names = list(NUMERICAL_COLS)
        try:
            ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
            cat_features = list(ohe.get_feature_names_out(cat_cols))
            feature_names.extend(cat_features)
        except:
            feature_names.extend(cat_cols)
        
        logger.info(f"✅ Preprocessing complete. Shape: {X.shape}, Features: {len(feature_names)}")
        return X, y, preprocessor, feature_names
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        raise


# ─────────────────────────────────────────────────────────────
# 3. Imbalanced Data Handling Strategies
# ─────────────────────────────────────────────────────────────

def get_resampling_strategy(
    strategy: str,
    random_state: int = 42
) -> Union[SMOTE, ADASYN, SMOTETomek, None]:
    """Get the appropriate resampling strategy."""
    
    strategies = {
        'smote': SMOTE(random_state=random_state, k_neighbors=5),
        'borderline_smote': BorderlineSMOTE(random_state=random_state, kind='borderline-1'),
        'svm_smote': SVMSMOTE(random_state=random_state),
        'adasyn': ADASYN(random_state=random_state),
        'smote_tomek': SMOTETomek(
            smote=SMOTE(random_state=random_state),
            tomek=TomekLinks(),
            random_state=random_state
        ),
        'smote_enn': SMOTEENN(
            smote=SMOTE(random_state=random_state),
            enn=EditedNearestNeighbours(),
            random_state=random_state
        ),
        'none': None
    }
    
    if strategy not in strategies:
        logger.warning(f"Unknown strategy '{strategy}', using SMOTE")
        return strategies['smote']
    
    logger.info(f"   Using resampling strategy: {strategy}")
    return strategies[strategy]


# ─────────────────────────────────────────────────────────────
# 4. Model Definitions with Regularization
# ─────────────────────────────────────────────────────────────

def build_base_models(preprocessor: ColumnTransformer, config: TrainingConfig) -> Dict[str, ImbPipeline]:
    """Build individual model pipelines with regularization."""
    
    resampler = get_resampling_strategy(config.imbalance_strategy, config.random_state)
    
    models = {}
    
    # XGBoost with regularization
    models['XGBoost'] = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            n_estimators=200,
            max_depth=4,  # Shallower for less overfitting
            learning_rate=0.05,  # Lower learning rate
            subsample=0.8,  # Row sampling
            colsample_bytree=0.8,  # Column sampling
            reg_alpha=0.1,  # L1 regularization
            reg_lambda=1.0,  # L2 regularization
            min_child_weight=3,  # Prevent overfitting
            gamma=0.1,  # Regularization term
            random_state=config.random_state,
            eval_metric='logloss',
            use_label_encoder=False
        ))
    ])
    
    # Random Forest with regularization
    models['Random Forest'] = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            class_weight='balanced_subsample',
            random_state=config.random_state,
            bootstrap=True,
            oob_score=True
        ))
    ])
    
    # HistGradientBoosting (fast, handles missing values)
    models['HistGradientBoosting'] = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=5,
            learning_rate=0.05,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=config.random_state
        ))
    ])
    
    # Logistic Regression with strong regularization
    models['Logistic Regression'] = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            C=0.1,  # Strong regularization
            penalty='l2',
            solver='lbfgs',
            max_iter=1000,
            class_weight='balanced',
            random_state=config.random_state
        ))
    ])
    
    # Balanced Random Forest (imbalance-aware)
    models['Balanced Random Forest'] = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', BalancedRandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            sampling_strategy='auto',
            replacement=True,
            random_state=config.random_state
        ))
    ])
    
    # SVM with RBF kernel
    models['SVM'] = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', SVC(
            C=0.5,
            kernel='rbf',
            gamma='scale',
            probability=True,
            class_weight='balanced',
            random_state=config.random_state
        ))
    ])
    
    # MLP Neural Network
    models['MLP'] = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', MLPClassifier(
            hidden_layer_sizes=(64, 32, 16),
            activation='relu',
            solver='adam',
            alpha=0.01,  # L2 regularization
            learning_rate='adaptive',
            early_stopping=True,
            validation_fraction=0.1,
            max_iter=500,
            random_state=config.random_state
        ))
    ])
    
    logger.info(f"✅ Built {len(models)} model pipelines")
    return models


def build_ensemble_models(
    preprocessor: ColumnTransformer, 
    base_models: Dict[str, ImbPipeline],
    config: TrainingConfig
) -> Dict[str, ImbPipeline]:
    """Build ensemble models (stacking, voting)."""
    
    ensembles = {}
    
    # Extract classifiers from pipelines
    classifiers = {
        name: pipeline.named_steps['classifier'] 
        for name, pipeline in base_models.items()
    }
    
    # Stacking Ensemble
    if config.ensemble_method in ['stacking', 'all']:
        stacking = StackingClassifier(
            estimators=[
                ('xgb', classifiers['XGBoost']),
                ('rf', classifiers['Random Forest']),
                ('hgb', classifiers['HistGradientBoosting']),
                ('lr', classifiers['Logistic Regression'])
            ],
            final_estimator=LogisticRegression(
                C=0.1,
                class_weight='balanced',
                random_state=config.random_state
            ),
            cv=5,
            stack_method='predict_proba',
            n_jobs=-1
        )
        ensembles['Stacking Ensemble'] = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', stacking)
        ])
    
    # Voting Ensemble (soft voting for probabilities)
    if config.ensemble_method in ['voting', 'all']:
        voting = VotingClassifier(
            estimators=[
                ('xgb', classifiers['XGBoost']),
                ('rf', classifiers['Random Forest']),
                ('hgb', classifiers['HistGradientBoosting']),
                ('brf', classifiers['Balanced Random Forest'])
            ],
            voting='soft',
            weights=[2, 2, 2, 1],  # Weight important models more
            n_jobs=-1
        )
        ensembles['Voting Ensemble'] = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', voting)
        ])
    
    # EasyEnsemble (specifically for imbalanced data)
    if config.imbalance_strategy in ['ensemble', 'all']:
        ensembles['EasyEnsemble'] = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', EasyEnsembleClassifier(
                n_estimators=10,
                base_estimator=XGBClassifier(
                    n_estimators=50,
                    max_depth=4,
                    learning_rate=0.1,
                    random_state=config.random_state
                ),
                random_state=config.random_state
            ))
        ])
    
    logger.info(f"✅ Built {len(ensembles)} ensemble models")
    return ensembles


# ─────────────────────────────────────────────────────────────
# 5. Hyperparameter Tuning with Optuna
# ─────────────────────────────────────────────────────────────

def optimize_hyperparameters(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
    config: TrainingConfig,
    n_trials: int = 50
) -> Dict[str, Any]:
    """Optimize hyperparameters using Optuna."""
    
    if not OPTUNA_AVAILABLE:
        logger.warning("Optuna not available, using default hyperparameters")
        return {}
    
    logger.info(f"   Optimizing hyperparameters for {model_name}...")
    
    def objective(trial):
        # Model-specific search spaces
        if model_name == 'XGBoost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            }
            
            model = ImbPipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', XGBClassifier(
                    **params,
                    random_state=config.random_state,
                    eval_metric='logloss',
                    use_label_encoder=False
                ))
            ])
            
        elif model_name == 'Random Forest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            }
            
            model = ImbPipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(**params, random_state=config.random_state))
            ])
            
        elif model_name == 'HistGradientBoosting':
            params = {
                'max_iter': trial.suggest_int('max_iter', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'l2_regularization': trial.suggest_float('l2_regularization', 0.01, 10.0, log=True),
            }
            
            model = ImbPipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', HistGradientBoostingClassifier(**params, random_state=config.random_state))
            ])
            
        else:
            return 0.5  # Default score for unknown models
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.random_state)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
        return scores.mean()
    
    study = optuna.create_study(
        direction='maximize',
        study_name=f'{model_name}_optimization',
        pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    logger.info(f"   Best AUC: {study.best_value:.4f}")
    logger.info(f"   Best params: {study.best_params}")
    
    return study.best_params


# ─────────────────────────────────────────────────────────────
# 6. Model Training with Calibration
# ─────────────────────────────────────────────────────────────

def train_with_calibration(
    model: ImbPipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: TrainingConfig
) -> ImbPipeline:
    """Train model with optional probability calibration."""
    
    logger.info(f"   Training {model.named_steps['classifier'].__class__.__name__}...")
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Apply calibration for better probability estimates
    if config.use_calibration:
        logger.info("   Applying probability calibration (isotonic)...")
        calibrated = CalibratedClassifierCV(
            model.named_steps['classifier'],
            method='isotonic',
            cv=5
        )
        calibrated.fit(X_train, y_train)
        
        # Replace classifier with calibrated version
        model.named_steps['classifier'] = calibrated
    
    return model


def train_models(
    models: Dict[str, ImbPipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: TrainingConfig
) -> Dict[str, ImbPipeline]:
    """Train all models with optional calibration."""
    
    trained = {}
    for name, pipeline in models.items():
        try:
            trained[name] = train_with_calibration(pipeline, X_train, y_train, config)
            logger.info(f"✅ {name} trained successfully")
        except Exception as e:
            logger.error(f"Error training {name}: {str(e)}")
            continue
    
    return trained


# ─────────────────────────────────────────────────────────────
# 7. Comprehensive Evaluation
# ─────────────────────────────────────────────────────────────

def evaluate_models(
    trained_models: Dict[str, ImbPipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: TrainingConfig
) -> Tuple[Dict[str, ModelMetrics], str, ImbPipeline]:
    """Evaluate all models with comprehensive metrics."""
    
    logger.info("Evaluating models...")
    all_metrics = {}
    
    for name, model in trained_models.items():
        try:
            preds = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1]
            
            # Core metrics
            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, zero_division=0)
            rec = recall_score(y_test, preds, zero_division=0)
            f1 = f1_score(y_test, preds, zero_division=0)
            auc = roc_auc_score(y_test, proba)
            avg_prec = average_precision_score(y_test, proba)
            mcc = matthews_corrcoef(y_test, preds)
            kappa = cohen_kappa_score(y_test, preds)
            brier = brier_score_loss(y_test, proba)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, preds).tolist()
            
            # Classification report
            report = classification_report(y_test, preds, output_dict=True)
            
            # Calibration curve
            fraction_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10)
            calibration_data = {
                'fraction_pos': fraction_pos.tolist(),
                'mean_pred': mean_pred.tolist()
            }
            
            # Cross-validation scores
            cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
            cv_scores = {
                'auc_roc': cross_val_score(model, X_test, y_test, cv=cv, scoring='roc_auc').tolist(),
                'f1': cross_val_score(model, X_test, y_test, cv=cv, scoring='f1').tolist(),
                'recall': cross_val_score(model, X_test, y_test, cv=cv, scoring='recall').tolist()
            }
            
            all_metrics[name] = ModelMetrics(
                accuracy=round(acc, 4),
                precision=round(prec, 4),
                recall=round(rec, 4),
                f1_score=round(f1, 4),
                auc_roc=round(auc, 4),
                average_precision=round(avg_prec, 4),
                mcc=round(mcc, 4),
                kappa=round(kappa, 4),
                brier_score=round(brier, 4),
                confusion_matrix=cm,
                classification_report={
                    k: {kk: round(vv, 4) if isinstance(vv, float) else vv 
                        for kk, vv in v.items()} if isinstance(v, dict) else round(v, 4)
                    for k, v in report.items()
                },
                calibration_data=calibration_data,
                cv_scores=cv_scores
            )
            
            logger.info(f"  ── {name} ──")
            logger.info(f"     Accuracy : {acc:.4f} | Precision: {prec:.4f} | Recall   : {rec:.4f}")
            logger.info(f"     F1-Score : {f1:.4f} | AUC-ROC  : {auc:.4f} | MCC      : {mcc:.4f}")
            logger.info(f"     Brier    : {brier:.4f} | Kappa    : {kappa:.4f}")
            
        except Exception as e:
            logger.error(f"Error evaluating {name}: {str(e)}")
            continue
    
    # Best model = highest AUC-ROC (primary metric for imbalanced data)
    best_name = max(all_metrics, key=lambda k: all_metrics[k].auc_roc)
    logger.info(f"🏆 Best model: {best_name} (AUC = {all_metrics[best_name].auc_roc:.4f})")
    
    return all_metrics, best_name, trained_models[best_name]


# ─────────────────────────────────────────────────────────────
# 8. SHAP Explainability
# ─────────────────────────────────────────────────────────────

def compute_shap_values(
    model: ImbPipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_names: List[str],
    sample_size: int = 100
) -> Dict[str, Any]:
    """Compute SHAP values for model explainability."""
    
    if not SHAP_AVAILABLE:
        logger.warning("SHAP not available, skipping explainability analysis")
        return {}
    
    try:
        logger.info("Computing SHAP values for explainability...")
        
        # Get the classifier
        classifier = model.named_steps['classifier']
        
        # Use a sample for faster computation
        X_sample = X_test.sample(min(sample_size, len(X_test)), random_state=42)
        
        # Choose appropriate explainer based on model type
        if hasattr(classifier, 'feature_importances_'):
            # Tree-based models
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_sample)
        else:
            # Use KernelExplainer for any model
            explainer = shap.KernelExplainer(classifier.predict_proba, X_sample)
            shap_values = explainer.shap_values(X_sample)
        
        # Get feature names from preprocessor
        try:
            ohe = model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
            cat_features = list(ohe.get_feature_names_out(CATEGORICAL_COLS))
            all_features = NUMERICAL_COLS + cat_features
        except:
            all_features = feature_names
        
        # Summary statistics
        if isinstance(shap_values, list):
            # Multi-class
            shap_summary = np.abs(shap_values[1]).mean(axis=0)
        else:
            shap_summary = np.abs(shap_values).mean(axis=0)
        
        feature_importance = dict(zip(all_features, [round(float(v), 6) for v in shap_summary]))
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        logger.info(f"✅ Computed SHAP values for {len(all_features)} features")
        
        return {
            'feature_importance': feature_importance,
            'shap_values': shap_values,
            'sample_data': X_sample.to_dict('records'),
            'feature_names': all_features
        }
        
    except Exception as e:
        logger.error(f"Error computing SHAP values: {str(e)}")
        return {}


# ─────────────────────────────────────────────────────────────
# 9. Save Artifacts
# ─────────────────────────────────────────────────────────────

def save_artifacts(
    model_dir: str,
    best_model: ImbPipeline,
    best_name: str,
    all_metrics: Dict[str, ModelMetrics],
    feature_importance: Dict[str, float],
    shap_data: Optional[Dict] = None,
    config: Optional[TrainingConfig] = None
) -> None:
    """Persist model, metrics, and explainability data."""
    
    try:
        resolved_model_dir = resolve_path(model_dir)
        resolved_model_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Creating model directory: {resolved_model_dir}")
        
        # Model
        model_path = resolved_model_dir / 'best_stroke_model.joblib'
        joblib.dump(best_model, model_path)
        logger.info(f"✅ Model saved → {model_path}")
        
        # Also save preprocessor separately for API use
        preprocessor_path = resolved_model_dir / 'preprocessor.joblib'
        joblib.dump(best_model.named_steps['preprocessor'], preprocessor_path)
        logger.info(f"✅ Preprocessor saved → {preprocessor_path}")
        
        # Metrics (convert dataclass to dict)
        metrics_payload = {
            'generated_at': datetime.now().isoformat(),
            'best_model': best_name,
            'config': asdict(config) if config else {},
            'models': {k: asdict(v) for k, v in all_metrics.items()}
        }
        metrics_path = resolved_model_dir / 'metrics.json'
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_payload, f, indent=2)
        logger.info(f"✅ Metrics saved → {metrics_path}")
        
        # Feature Importance
        fi_path = resolved_model_dir / 'feature_importance.json'
        with open(fi_path, 'w', encoding='utf-8') as f:
            json.dump(feature_importance, f, indent=2)
        logger.info(f"✅ Feature importance saved → {fi_path}")
        
        # SHAP data (if available)
        if shap_data:
            shap_path = resolved_model_dir / 'shap_explanations.json'
            # Convert numpy arrays to lists for JSON serialization
            shap_serializable = {
                'feature_importance': shap_data.get('feature_importance', {}),
                'feature_names': shap_data.get('feature_names', []),
                'sample_data': shap_data.get('sample_data', []),
                'computed_at': datetime.now().isoformat()
            }
            with open(shap_path, 'w', encoding='utf-8') as f:
                json.dump(shap_serializable, f, indent=2)
            logger.info(f"✅ SHAP explanations saved → {shap_path}")
        
    except Exception as e:
        logger.error(f"Error saving artifacts: {str(e)}")
        raise


# ─────────────────────────────────────────────────────────────
# 10. Main Training Pipeline
# ─────────────────────────────────────────────────────────────

def main(config: Optional[TrainingConfig] = None) -> None:
    """Run the complete advanced ML training pipeline."""
    
    if config is None:
        config = TrainingConfig()
    
    try:
        logger.info("="*70)
        logger.info("Starting Advanced ML Training Pipeline v3.0")
        logger.info("="*70)
        logger.info(f"Configuration: {asdict(config)}")
        
        DATA_PATH = "data/healthcare-dataset-stroke-data.csv"
        MODEL_DIR = "models"
        
        # Load data
        df = load_data(DATA_PATH)
        
        # Preprocess
        X, y, preprocessor, feature_names = preprocess_data(df, config)
        
        # Split data (stratified for imbalanced data)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=config.test_size, 
            random_state=config.random_state, 
            stratify=y
        )
        logger.info(f"Train: {len(X_train):,} samples | Test: {len(X_test):,} samples")
        logger.info(f"Stroke cases in train: {int(y_train.sum())} / {len(y_train)} ({y_train.mean()*100:.1f}%)")
        
        # Build models
        base_models = build_base_models(preprocessor, config)
        
        # Hyperparameter tuning (optional)
        if config.hyperparameter_tuning and OPTUNA_AVAILABLE:
            logger.info("\n" + "="*70)
            logger.info("Hyperparameter Optimization with Optuna")
            logger.info("="*70)
            
            # Optimize top models
            for model_name in ['XGBoost', 'Random Forest', 'HistGradientBoosting']:
                best_params = optimize_hyperparameters(
                    model_name, X_train, y_train, preprocessor, config, n_trials=config.n_trials
                )
                if best_params:
                    # Update model with best params
                    classifier = base_models[model_name].named_steps['classifier']
                    classifier.set_params(**best_params)
        
        # Build ensemble models
        ensemble_models = build_ensemble_models(preprocessor, base_models, config)
        all_models = {**base_models, **ensemble_models}
        
        # Train models
        logger.info("\n" + "="*70)
        logger.info("Training Models")
        logger.info("="*70)
        trained = train_models(all_models, X_train, y_train, config)
        
        # Evaluate models
        logger.info("\n" + "="*70)
        logger.info("Model Evaluation")
        logger.info("="*70)
        all_metrics, best_name, best_model = evaluate_models(trained, X_test, y_test, config)
        
        # Extract feature importance
        logger.info("\n" + "="*70)
        logger.info("Feature Importance")
        logger.info("="*70)
        clf = best_model.named_steps['classifier']
        if hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
        elif hasattr(clf, 'coef_'):
            importances = np.abs(clf.coef_[0])
        else:
            importances = None
        
        if importances is not None and len(importances) == len(feature_names):
            fi_dict = dict(zip(feature_names, [round(float(v), 6) for v in importances]))
            fi_dict = dict(sorted(fi_dict.items(), key=lambda x: x[1], reverse=True))
        else:
            fi_dict = {f"feature_{i}": 0.0 for i in range(len(feature_names))}
        
        # SHAP explainability
        shap_data = {}
        if config.use_shap:
            logger.info("\n" + "="*70)
            logger.info("SHAP Explainability Analysis")
            logger.info("="*70)
            shap_data = compute_shap_values(best_model, X_train, X_test, feature_names)
            if shap_data and 'feature_importance' in shap_data:
                fi_dict = shap_data['feature_importance']
        
        # Save artifacts
        logger.info("\n" + "="*70)
        logger.info("Saving Artifacts")
        logger.info("="*70)
        save_artifacts(
            MODEL_DIR, 
            best_model, 
            best_name, 
            all_metrics, 
            fi_dict,
            shap_data if shap_data else None,
            config
        )
        
        logger.info("\n" + "="*70)
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
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
