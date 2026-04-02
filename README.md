<div align="center">

# 🧠 Stroke Risk Prediction & Analytics Platform

### AI-Powered Healthcare Decision Support System

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-006400?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

<br>

An end-to-end **Machine Learning** platform for predicting stroke risk in patients using clinical and demographic data. Features a professional multi-page analytics dashboard, automated model training pipeline, and real-time risk assessment engine.

<br>

[🚀 Get Started](#-quick-start) · [📊 Dashboard](#-interactive-dashboard) · [🤖 Models](#-machine-learning-pipeline) · [📖 Documentation](#-project-structure)

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [Interactive Dashboard](#-interactive-dashboard)
- [Dataset](#-dataset)
- [Results & Performance](#-results--performance)
- [Business Impact](#-business-impact)
- [Future Roadmap](#-future-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔭 Overview

Stroke is the **2nd leading cause of death** globally, responsible for approximately 11% of total deaths ([WHO](https://www.who.int/)). Early identification of at-risk patients can dramatically improve outcomes through preventive interventions.

This project delivers a **production-grade analytics platform** that:

- 🧪 Trains and evaluates multiple ML models on healthcare data
- 📊 Provides an interactive, multi-page dashboard for clinical exploration
- ⚕️ Offers real-time stroke risk prediction for individual patients
- 📋 Generates automated executive reports with actionable recommendations

---

## ✨ Key Features



## 📌 Featured Projects

| Project | Key Focus | Status |
| :--- | :--- | :--- |
| 🧠 [**Stroke Risk Prediction**](https://github.com/ASTROBOX1/Stroke-Risk-Prediction) | Classification model to predict stroke risk with data preprocessing. | `Completed` |
| 🔍 [**Mini RAG System**](https://github.com/ASTROBOX1/mini-rag) | AI-powered document retrieval and generation using LangChain. | `Production` |
| 💎 [**Diamond Price Prediction**](https://github.com/ASTROBOX1/Diamond-Price-Prediction-ML) | Regression model with extensive Feature Engineering. | `Completed` |


---

## 🛠 Tech Stack

<table>
<tr>
<td align="center"><b>Category</b></td>
<td align="center"><b>Technologies</b></td>
</tr>
<tr>
<td><b>Language</b></td>
<td>Python 3.9+</td>
</tr>
<tr>
<td><b>ML Framework</b></td>
<td>scikit-learn · XGBoost · imbalanced-learn (SMOTE)</td>
</tr>
<tr>
<td><b>Dashboard</b></td>
<td>Streamlit · Plotly</td>
</tr>
<tr>
<td><b>Data Processing</b></td>
<td>Pandas · NumPy · SciPy</td>
</tr>
<tr>
<td><b>Visualization</b></td>
<td>Plotly · Matplotlib · Seaborn</td>
</tr>
<tr>
<td><b>Model Persistence</b></td>
<td>Joblib · JSON (metrics & feature importance)</td>
</tr>
<tr>
<td><b>Notebook</b></td>
<td>Jupyter (EDA & experimental analysis)</td>
</tr>
</table>

---

## ✨ Version 2.1 Improvements & Best Practices

### Code Quality Enhancements

| Enhancement | Description | Status |
|-------------|-------------|--------|
| **Modular Architecture** | Refactored into separate modules (pages, utils, API) for maintainability | ✅ |
| **Type Hints** | Added comprehensive type hints to all functions | ✅ |
| **Logging System** | Structured logging with file & console handlers | ✅ |
| **Error Handling** | Try-catch blocks with meaningful error messages | ✅ |
| **Configuration Management** | YAML config file for environment-specific settings | ✅ |
| **Input Validation** | Comprehensive validation for user inputs and data | ✅ |

### Production-Ready Features

| Feature | Description | Status |
|---------|-------------|--------|
| **REST API** | FastAPI endpoints for programmatic access | ✅ |
| **Docker Support** | Dockerfile & docker-compose for containerized deployment | ✅ |
| **Unit Tests** | Pytest test suite with 15+ test cases | ✅ |
| **CI/CD Pipeline** | GitHub Actions for automated testing & code quality | ✅ |
| **Environment Variables** | .env.example template for configuration | ✅ |
| **Comprehensive Logging** | Application-wide logging to file and console | ✅ |

---

```
stroke-prediction/
│
├── 📄 app.py                   # Streamlit multi-page dashboard (refactored)
├── 📄 train_model.py           # ML training pipeline (enhanced with logging & type hints)
├── 📄 api.py                   # FastAPI REST API for predictions
├── 📄 utils.py                 # Shared utilities, logging, validation
├── 📄 config.yaml              # Configuration management
├── 📄 requirements.txt         # Python dependencies
├── 📄 Dockerfile               # Docker containerization
├── 📄 docker-compose.yml       # Docker Compose orchestration
├── 📄 .env.example             # Environment variables template
│
├── 📂 pages/                   # Streamlit page modules
│   ├── __init__.py
│   ├── overview.py             # Executive Overview page
│   ├── explorer.py             # Data Explorer page
│   ├── eda.py                  # Exploratory Analysis page
│   ├── model_performance.py    # Model Performance page
│   ├── predictor.py            # Stroke Risk Predictor page
│   └── report.py               # Report Summary page
│
├── 📂 data/
│   └── healthcare-dataset-stroke-data.csv   # Source dataset (~5,110 records)
│
├── 📂 models/
│   ├── best_stroke_model.joblib             # Serialized best model
│   ├── metrics.json                         # Evaluation metrics for all models
│   └── feature_importance.json              # Feature importance scores
│
├── 📂 logs/                    # Application logs
│   ├── app.log                 # Streamlit app logs
│   └── train_model.log         # Training pipeline logs
│
├── 📂 tests/                   # Unit tests
│   ├── __init__.py
│   └── test_utils.py           # Tests for utilities and validation
│
├── 📂 .github/workflows/       # CI/CD Pipelines
│   └── tests.yml               # GitHub Actions workflow
│
└── 📄 README.md                # Project documentation (you are here)
```

---

## 🚀 Quick Start

### Prerequisites

- Python **3.9** or higher
- pip package manager

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/ASTROBOX1/stroke-prediction.git
cd stroke-prediction
```

### 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Train the Models

```bash
python train_model.py
```

This will:
- Load and preprocess the dataset
- Train **3 ML models** (Random Forest, XGBoost, Logistic Regression)
- Apply **SMOTE** oversampling to handle class imbalance
- Evaluate all models and select the best one (by AUC-ROC)
- Export the model, metrics, and feature importance to `models/`

### 5️⃣ Launch the Dashboard

```bash
streamlit run app.py
```

The dashboard will open at **http://localhost:8501** 🎉

### 6️⃣ (Optional) Launch the REST API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

API documentation will be available at **http://localhost:8000/api/docs**

### 7️⃣ (Optional) Run with Docker

```bash
# Build and run both dashboard and API
docker-compose up --build
```

This will:
- Start the Streamlit dashboard on port 8501
- Start the FastAPI on port 8000
- Set up shared network for inter-service communication

### 8️⃣ (Optional) Run Unit Tests

```bash
pytest tests/ -v --cov
```

---

## 🌐 REST API Usage

The FastAPI provides programmatic access to stroke predictions:

### Health Check
```bash
curl http://localhost:8000/health
```

### Single Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "age": 50,
    "hypertension": 0,
    "heart_disease": 0,
    "ever_married": "Yes",
    "work_type": "Private",
    "Residence_type": "Urban",
    "avg_glucose_level": 100.0,
    "bmi": 28.5,
    "smoking_status": "never smoked"
  }'
```

### Batch Predictions
```bash
curl -X POST http://localhost:8000/batch-predict \
  -H "Content-Type: application/json" \
  -d '[
    {"gender": "Male", "age": 50, ...},
    {"gender": "Female", "age": 45, ...}
  ]'
```

### API Documentation
- **Interactive Docs**: http://localhost:8000/api/docs (Swagger UI)
- **Alternative Docs**: http://localhost:8000/api/redoc (ReDoc)

## 🤖 Machine Learning Pipeline

### Pipeline Architecture

```
Raw Data ──► Preprocessing ──► SMOTE Oversampling ──► Model Training ──► Evaluation ──► Export
                │                                           │
                ├── Median Imputation (BMI)                 ├── Random Forest
                ├── Standard Scaling (numerical)            ├── XGBoost
                └── One-Hot Encoding (categorical)          └── Logistic Regression
```

### Models Trained

| Model | Algorithm | Key Hyperparameters |
|-------|-----------|---------------------|
| **Random Forest** | Ensemble (Bagging) | 200 trees, max depth 10, balanced class weights |
| **XGBoost** | Gradient Boosting | 200 estimators, max depth 6, LR 0.1 |
| **Logistic Regression** | Linear Model | max iter 1000, balanced class weights |

### Class Imbalance Handling

The dataset exhibits severe class imbalance (~4.9% positive class). We address this with:

- **SMOTE** (Synthetic Minority Over-sampling Technique) in the training pipeline
- **Balanced class weights** in Random Forest and Logistic Regression
- **AUC-ROC** as the primary evaluation metric (robust to imbalance)
- **Stratified splitting** to preserve class ratios in train/test sets

### Feature Engineering

| Category | Features |
|----------|----------|
| **Numerical** | `age`, `hypertension`, `heart_disease`, `avg_glucose_level`, `bmi` |
| **Categorical** | `gender`, `ever_married`, `work_type`, `Residence_type`, `smoking_status` |

---

## 📊 Interactive Dashboard

The Streamlit dashboard consists of **6 interconnected pages**:

### 🏠 Executive Overview
> KPI cards displaying total patients, stroke rate, average age, BMI, hypertension rate, and heart disease rate. Includes age distribution histograms, stroke rate by work type, glucose vs BMI scatter plots, and demographic pie charts.

### 🔍 Data Explorer
> Interactive filtering by age range, gender, stroke status, and work type. Real-time metric recalculation with filtered data table and CSV download capability.

### 📊 Exploratory Analysis
> Tabbed distributions (Age, Glucose, BMI) with overlaid histograms and box plots. Includes categorical feature analysis, correlation heatmap, and feature-stroke correlation rankings.

### 🤖 Model Performance
> Side-by-side model comparison cards, ROC and Precision-Recall curves, confusion matrices, feature importance visualization, and grouped bar chart comparison.

### ⚕️ Stroke Risk Predictor
> Input patient demographics, medical history, and lifestyle factors. Real-time risk gauge visualization with color-coded risk levels (Low / Moderate / High) and clinical recommendations.

### 📋 Report Summary
> Auto-generated executive report with dataset overview, key risk factor findings, model results table, business recommendations, and documented limitations.

---

## 📦 Dataset

| Property | Value |
|----------|-------|
| **Source** | [Kaggle — Stroke Prediction Dataset](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset) |
| **Records** | ~5,110 patients |
| **Features** | 11 clinical & demographic attributes |
| **Target** | `stroke` (binary: 0 = No, 1 = Yes) |
| **Imbalance** | ~4.9% positive class |

### Feature Dictionary

| Feature | Type | Description |
|---------|------|-------------|
| `gender` | Categorical | Male / Female |
| `age` | Numerical | Patient age |
| `hypertension` | Binary | 0 = No, 1 = Yes |
| `heart_disease` | Binary | 0 = No, 1 = Yes |
| `ever_married` | Categorical | Yes / No |
| `work_type` | Categorical | Private, Self-employed, Govt_job, children, Never_worked |
| `Residence_type` | Categorical | Urban / Rural |
| `avg_glucose_level` | Numerical | Average blood glucose level (mg/dL) |
| `bmi` | Numerical | Body Mass Index |
| `smoking_status` | Categorical | formerly smoked, never smoked, smokes, Unknown |
| `stroke` | Binary (Target) | 0 = No stroke, 1 = Stroke |

---

## 📈 Results & Performance

### Key Findings

- 🎯 **Age** is the strongest predictor — stroke risk increases dramatically after age 60
- 🩸 **Hypertension** patients show ~2× higher stroke rate
- ❤️ **Heart disease** patients show ~3× higher stroke rate
- 🍬 **High glucose** (>125 mg/dL) strongly correlates with stroke
- ⚖️ **BMI** has a moderate but statistically significant impact

### Model Selection Criteria

The best model is selected based on **AUC-ROC** — the most appropriate metric for imbalanced medical datasets, as it measures the model's ability to discriminate between stroke and non-stroke patients across all classification thresholds.

---

## 💼 Business Impact

| Recommendation | Description |
|----------------|-------------|
| 🎯 **Priority Screening** | Focus on patients aged 60+ with hypertension or heart disease |
| 📊 **Glucose Monitoring** | Implement regular monitoring for pre-diabetic levels (100–125 mg/dL) |
| ⚖️ **Clinical Deployment** | Deploy model for real-time risk scoring in clinical workflows |
| 🔄 **Continuous Learning** | Retrain quarterly with new patient data; monitor for concept drift |
| 📋 **EHR Integration** | Integrate risk score into electronic health records for rapid pre-screening |

---

## 🗺 Future Roadmap

### Version 2.1 ✅ (Complete)
- [x] Modular architecture with separate page components
- [x] Comprehensive logging system
- [x] Type hints throughout codebase
- [x] Configuration management (YAML)
- [x] Input validation framework
- [x] FastAPI REST API for predictions
- [x] Docker & docker-compose support
- [x] Unit tests with pytest
- [x] GitHub Actions CI/CD pipeline
- [x] Environment variable management

### Future Enhancements (v2.2+)
- [ ] Deep Learning models (Neural Networks, TabNet)
- [ ] SHAP/LIME explainability for individual predictions
- [ ] Model versioning system with timestamp tracking
- [ ] Time-series analysis for recurring patient visits
- [ ] Automated alerting system for high-risk patients
- [ ] Multi-language dashboard support
- [ ] PostgreSQL database integration
- [ ] Real-time data streaming capabilities
- [ ] Integration with hospital EHR systems
- [ ] Mobile app support

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

### Built with ❤️ by the Data Science Team

**Healthcare Analytics Division © 2026**

<br>

⭐ Star this repo if you found it useful!

</div>
