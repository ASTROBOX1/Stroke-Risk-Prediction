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

| Feature | Description |
|---------|-------------|
| 🏠 **Executive Overview** | KPI cards, demographic distributions, and cross-variable analysis at a glance |
| 🔍 **Data Explorer** | Interactive filters by age, gender, stroke status, and work type with CSV export |
| 📊 **Exploratory Analysis** | In-depth EDA with histograms, box plots, categorical breakdowns, and correlation heatmaps |
| 🤖 **Model Performance** | ROC/PR curves, confusion matrices, feature importance, and model comparison charts |
| ⚕️ **Risk Predictor** | Real-time stroke risk assessment with visual gauge and clinical recommendations |
| 📋 **Report Summary** | Automated executive report with findings, model results, and business recommendations |

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

## 📁 Project Structure

```
stroke-prediction/
│
├── 📄 app.py                  # Streamlit multi-page dashboard (1,095 lines)
├── 📄 train_model.py          # ML training & evaluation pipeline
├── 📄 requirements.txt        # Python dependencies
├── 📄 README.md               # Project documentation (you are here)
│
├── 📂 data/
│   └── healthcare-dataset-stroke-data.csv   # Source dataset (~5,110 records)
│
├── 📂 models/
│   ├── best_stroke_model.joblib             # Serialized best model
│   ├── metrics.json                         # Evaluation metrics for all models
│   └── feature_importance.json              # Feature importance scores
│
└── 📂 notebooks/
    └── stroke_analysis.ipynb                # Jupyter EDA notebook
```

---

## 🚀 Quick Start

### Prerequisites

- Python **3.9** or higher
- pip package manager

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/stroke-prediction.git
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

---

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

- [ ] Deep Learning models (Neural Networks, TabNet)
- [ ] SHAP/LIME explainability for individual predictions
- [ ] API endpoint (FastAPI) for integration with hospital systems
- [ ] Time-series analysis for recurring patient visits
- [ ] Automated alerting system for high-risk patients
- [ ] Multi-language dashboard support
- [ ] Docker containerization for deployment
- [ ] Integration with real hospital EHR systems

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
