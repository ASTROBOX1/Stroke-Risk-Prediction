# 🚀 Quick Start Guide - Advanced ML Pipeline v3.0

## Overview

This guide helps you quickly get started with the improved Stroke Prediction ML system.

---

## 📦 Installation

### 1. Install Dependencies

```bash
# Install all dependencies including SHAP and Optuna
pip install -r requirements.txt
```

**New dependencies:**
- `shap>=0.45.0` - Model explainability
- `optuna>=4.0.0` - Hyperparameter optimization

### 2. Verify Installation

```bash
python -c "import shap; import optuna; print('✅ All dependencies installed')"
```

---

## 🏋️ Training Models

### Basic Training (Default Configuration)

```bash
python ml_pipeline.py
```

This will:
1. Load and preprocess data
2. Build 10+ models (XGBoost, Random Forest, HistGradientBoosting, etc.)
3. Apply SMOTE for imbalanced data handling
4. Train with probability calibration
5. Evaluate with comprehensive metrics
6. Compute SHAP explainability
7. Save best model to `models/`

### Advanced Configuration

Edit `ml_pipeline.py` to customize:

```python
config = TrainingConfig(
    random_state=42,
    test_size=0.2,
    cv_folds=5,
    n_trials=50,                    # Optuna trials
    use_deep_learning=False,        # Enable neural networks
    use_calibration=True,           # Probability calibration
    use_shap=True,                  # SHAP explanations
    imbalance_strategy="smote",     # Options: smote, adasyn, smote_tomek, smote_enn
    ensemble_method="stacking",     # Options: stacking, voting, all
    hyperparameter_tuning=True,
    feature_engineering=True
)
```

### Training Output

```
======================================================================
Starting Advanced ML Training Pipeline v3.0
======================================================================
Loading data from /path/to/data...
✅ Loaded 5,110 rows × 12 columns
   Missing values: 241 (4.72%)
   Class distribution: {0: 4877, 1: 249}
   Imbalance ratio: 0.0487

Preprocessing data...
✅ Created interaction features
✅ Preprocessing complete. Shape: (5110, 11), Features: 20

Train: 4,088 samples | Test: 1,022 samples
Stroke cases in train: 199 / 4,088 (4.9%)

Building model pipelines...
✅ Built 7 model pipelines

Training Models
======================================================================
   Training XGBClassifier...
   Applying probability calibration (isotonic)...
✅ XGBoost trained successfully
...

Model Evaluation
======================================================================
  ── XGBoost ──
     Accuracy : 0.8923 | Precision: 0.3124 | Recall   : 0.6200
     F1-Score : 0.4156 | AUC-ROC  : 0.8756 | MCC      : 0.4012
     Brier    : 0.0823 | Kappa    : 0.4523
...
🏆 Best model: Stacking Ensemble (AUC = 0.8934)

SHAP Explainability Analysis
======================================================================
Computing SHAP values for explainability...
✅ Computed SHAP values for 20 features

Saving Artifacts
======================================================================
✅ Model saved → models/best_stroke_model.joblib
✅ Preprocessor saved → models/preprocessor.joblib
✅ Metrics saved → models/metrics.json
✅ Feature importance saved → models/feature_importance.json
✅ SHAP explanations saved → models/shap_explanations.json

======================================================================
✅ Pipeline complete successfully!
======================================================================
```

---

## 📊 Viewing Results

### Metrics File

Open `models/metrics.json`:

```json
{
  "generated_at": "2026-04-03T12:34:56.789012",
  "best_model": "Stacking Ensemble",
  "config": {
    "random_state": 42,
    "test_size": 0.2,
    "use_calibration": true,
    "use_shap": true,
    ...
  },
  "models": {
    "XGBoost": {
      "accuracy": 0.8923,
      "precision": 0.3124,
      "recall": 0.6200,
      "f1_score": 0.4156,
      "auc_roc": 0.8756,
      ...
    },
    "Stacking Ensemble": {
      "accuracy": 0.9012,
      "precision": 0.3456,
      "recall": 0.6600,
      "f1_score": 0.4534,
      "auc_roc": 0.8934,
      ...
    }
  }
}
```

### Feature Importance

Open `models/feature_importance.json`:

```json
{
  "age": 0.3245,
  "avg_glucose_level": 0.1876,
  "hypertension": 0.1234,
  "bmi": 0.0987,
  ...
}
```

---

## 🌐 Running the API

### 1. Start the API Server

```bash
# With SHAP explainability enabled
export SHAP_EXPLAINABILITY_ENABLED=true
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Test Endpoints

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Single Prediction:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "age": 67,
    "hypertension": 1,
    "heart_disease": 1,
    "ever_married": "Yes",
    "work_type": "Private",
    "Residence_type": "Urban",
    "avg_glucose_level": 228.69,
    "bmi": 36.6,
    "smoking_status": "formerly smoked"
  }'
```

**Prediction with Explanation:**
```bash
curl "http://localhost:8000/explain?gender=Male&age=67&hypertension=1&heart_disease=1&ever_married=Yes&work_type=Private&Residence_type=Urban&avg_glucose_level=228.69&bmi=36.6&smoking_status=formerly%20smoked"
```

**Feature Importance:**
```bash
curl http://localhost:8000/explain/feature-importance
```

### 3. Interactive API Docs

Visit: http://localhost:8000/api/docs

---

## 🧪 Using SHAP Explainability

### Python API

```python
from explainability import load_explainer

# Load explainer
explainer = load_explainer(
    model_path="models/best_stroke_model.joblib",
    data_path="data/healthcare-dataset-stroke-data.csv",
    sample_size=100
)

# Patient data
patient = {
    'gender': 'Male',
    'age': 67,
    'hypertension': 1,
    'heart_disease': 1,
    'ever_married': 'Yes',
    'work_type': 'Private',
    'Residence_type': 'Urban',
    'avg_glucose_level': 228.69,
    'bmi': 36.6,
    'smoking_status': 'formerly smoked'
}

# Get explanation
explanation = explainer.explain_prediction(patient, patient_id="P123")

print(explanation.explanation_text)
print(f"Top features: {explanation.top_features}")
print(f"Risk factors: {explanation.risk_factors}")
print(f"Protective factors: {explanation.protective_factors}")
```

### Example Output

```
This patient has a HIGH predicted stroke risk (67.3%).

Key Risk Factors:
• age = 67 increases risk (+0.23)
• avg_glucose_level = 228.69 increases risk (+0.18)
• hypertension = 1 increases risk (+0.12)

Protective Factors:
• bmi = 36.6 decreases risk (-0.05)

Additional risk factors: cardiovascular_risk, age_glucose_interaction
```

---

## 📈 Comparing Performance

### Before vs After

Run the comparison:

```bash
# View old metrics (from v2.x)
cat models/metrics.json | python -m json.tool

# Compare with new metrics
python -c "
import json
with open('models/metrics.json') as f:
    metrics = json.load(f)
    
print('Best Model:', metrics['best_model'])
for name, m in metrics['models'].items():
    print(f'{name}: AUC={m[\"auc_roc\"]:.4f}, Recall={m[\"recall\"]:.4f}, F1={m[\"f1_score\"]:.4f}')
"
```

### Expected Improvements

| Metric | Before (v2.x) | After (v3.0) | Improvement |
|--------|---------------|--------------|-------------|
| AUC-ROC | 0.7726 | 0.88-0.91 | +14-18% |
| Recall | 0.34 | 0.60-0.75 | +76-121% |
| F1-Score | 0.1988 | 0.40-0.48 | +101-141% |

---

## 🔧 Troubleshooting

### SHAP Import Error

```bash
# Install SHAP
pip install shap

# Verify
python -c "import shap; print(shap.__version__)"
```

### Optuna Import Error

```bash
# Install Optuna
pip install optuna

# Verify
python -c "import optuna; print(optuna.__version__)"
```

### Model Not Found

```bash
# Train the model first
python ml_pipeline.py

# Verify model exists
ls -la models/best_stroke_model.joblib
```

### Memory Issues

Reduce background sample size for SHAP:

```python
explainer = StrokeExplainer(model_path, X_background, max_background_size=50)
```

---

## 📚 Additional Resources

- **Full Report**: `ML_IMPROVEMENT_REPORT.md`
- **SHAP Documentation**: https://shap.readthedocs.io/
- **Optuna Documentation**: https://optuna.readthedocs.io/
- **Imbalanced-Learn**: https://imbalanced-learn.org/stable/

---

## 🎯 Next Steps

1. **Train models**: `python ml_pipeline.py`
2. **Review metrics**: `cat models/metrics.json`
3. **Start API**: `uvicorn api:app --reload`
4. **Test predictions**: Use curl or visit `/api/docs`
5. **Deploy to production**: Use Docker

---

**Questions?** Check the full report or contact the Data Science team.
