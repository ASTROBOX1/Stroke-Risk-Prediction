"""
Report Summary page - Automated executive report.
"""

import logging
import os
from typing import Dict, Any, Optional
import pandas as pd
import streamlit as st
from utils import section_divider, bootstrap_standalone_page


def show(df: pd.DataFrame, metrics: Optional[Dict], metrics_loaded: bool,
         config: Dict[str, Any], logger: logging.Logger) -> None:
    """Display Report Summary page."""
    try:
        st.markdown("# 📋 Executive Report")
        st.markdown("*Automated summary report of the stroke prediction analysis.*")
        section_divider()

        # Header
        st.markdown("""
        <div class="report-card" style="background: linear-gradient(135deg, #0F172A, #1E293B); color:white; border:none;">
            <h2 style="color:white; margin:0;">Stroke Risk Prediction — Analytics Report</h2>
            <p style="opacity:0.8; margin:5px 0 0;">Healthcare Analytics Division • Data Science Team • March 2026</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Dataset Overview
        st.markdown("### 1. Dataset Overview")

        st.markdown(f"""
        <div class="report-card">
            <table style="width:100%; border-collapse:collapse;">
                <tr><td style="padding:8px; border-bottom:1px solid #eee;"><strong>Total Records</strong></td><td style="text-align:right; padding:8px; border-bottom:1px solid #eee;">{len(df):,}</td></tr>
                <tr><td style="padding:8px; border-bottom:1px solid #eee;"><strong>Features</strong></td><td style="text-align:right; padding:8px; border-bottom:1px solid #eee;">{df.shape[1]}</td></tr>
                <tr><td style="padding:8px; border-bottom:1px solid #eee;"><strong>Stroke Cases</strong></td><td style="text-align:right; padding:8px; border-bottom:1px solid #eee;">{int(df['stroke'].sum())} ({df['stroke'].mean()*100:.1f}%)</td></tr>
                <tr><td style="padding:8px; border-bottom:1px solid #eee;"><strong>Average Age</strong></td><td style="text-align:right; padding:8px; border-bottom:1px solid #eee;">{df['age'].mean():.1f} years</td></tr>
                <tr><td style="padding:8px; border-bottom:1px solid #eee;"><strong>Average BMI</strong></td><td style="text-align:right; padding:8px; border-bottom:1px solid #eee;">{df['bmi'].mean():.1f}</td></tr>
                <tr><td style="padding:8px;"><strong>Avg Glucose Level</strong></td><td style="text-align:right; padding:8px;">{df['avg_glucose_level'].mean():.1f} mg/dL</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

        # Key Findings
        st.markdown("### 2. Key Findings")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="report-card">
                <h4>🎯 Risk Factor Analysis</h4>
                <ul>
                    <li><strong>Age</strong> is the strongest predictor — risk jumps dramatically after age 60</li>
                    <li><strong>Hypertension</strong> patients have ~2x higher stroke rate</li>
                    <li><strong>Heart disease</strong> patients have ~3x higher stroke rate</li>
                    <li><strong>High glucose</strong> (>125 mg/dL) strongly correlates with stroke</li>
                    <li><strong>BMI</strong> has moderate but significant impact</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="report-card">
                <h4>⚖️ Data Challenges</h4>
                <ul>
                    <li><strong>Class imbalance:</strong> Only ~4.9% positive cases</li>
                    <li><strong>Missing data:</strong> BMI has missing values (imputed with median)</li>
                    <li><strong>Mitigation:</strong> SMOTE oversampling applied to training data</li>
                    <li><strong>Evaluation:</strong> AUC-ROC used as primary metric (robust to imbalance)</li>
                    <li><strong>Validation:</strong> Stratified train/test split preserves class ratio</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        # Model Results
        st.markdown("### 3. Model Results")

        if metrics_loaded:
            results_data = []
            for name in metrics['models']:
                m = metrics['models'][name]
                results_data.append({
                    'Model': name,
                    'Accuracy': f"{m['accuracy']:.4f}",
                    'Precision': f"{m['precision']:.4f}",
                    'Recall': f"{m['recall']:.4f}",
                    'F1-Score': f"{m['f1_score']:.4f}",
                    'AUC-ROC': f"{m['auc_roc']:.4f}"
                })

            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df.set_index('Model'), use_container_width=True)

            st.info(f"**🏆 Selected Model: {metrics['best_model']}** — Selected for highest AUC-ROC score, indicating best overall discrimination between stroke and non-stroke patients.")

        # Recommendations
        st.markdown("### 4. Business Recommendations")

        recs = [
            ("🎯", "Priority Screening", "Focus on patients aged 60+ with hypertension or heart disease — they represent the highest risk group"),
            ("📊", "Glucose Monitoring", "Implement regular glucose monitoring for patients with pre-diabetic levels (100-125 mg/dL)"),
            ("⚖️", "Model Deployment", "Deploy Logistic Regression model for interpretable, real-time stroke risk scoring in clinical workflows"),
            ("🔄", "Continuous Learning", "Retrain model quarterly with new patient data. Monitor for concept drift and feature distribution changes"),
            ("📋", "Risk Score Integration", "Integrate composite risk score into patient intake forms and electronic health records for rapid pre-screening"),
        ]

        for icon, title, desc in recs:
            st.markdown(f"""
            <div class="report-card">
                <h4>{icon} {title}</h4>
                <p style="margin:0; color:#4A5568;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

        # Limitations
        st.markdown("### 5. Limitations & Next Steps")

        st.markdown("""
        <div class="report-card" style="border-left: 4px solid #F39C12;">
            <h4>⚠️ Current Limitations</h4>
            <ul style="color:#4A5568;">
                <li>Dataset is limited to ~5,000 records — more data would improve generalization</li>
                <li>Missing BMI values were imputed, which may introduce bias</li>
                <li>No geographic, genetic, or medication data available</li>
                <li>Model performance on real-world clinical data may differ from training metrics</li>
            </ul>
            <h4 style="margin-top:16px;">🚀 Recommended Next Steps</h4>
            <ul style="color:#4A5568;">
                <li>Collect additional patient data from partner hospitals</li>
                <li>Integrate lab test results and medication history</li>
                <li>Develop time-series analysis for recurring patient visits</li>
                <li>Build automated alerting system for high-risk patients</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        section_divider()

        st.markdown("""
        <div style="text-align:center; color:#9CA3AF; font-size:0.85em;">
            Report generated automatically by the Stroke Prediction Analytics Pipeline<br>
            Healthcare Analytics Division • Data Science Team • March 2026
        </div>
        """, unsafe_allow_html=True)

        logger.info("Report Summary page displayed successfully")

    except Exception as e:
        logger.error(f"Error in Report page: {str(e)}")
        st.error(f"❌ Error in Report: {str(e)}")


if os.getenv("STROKE_APP_EMBEDDED_PAGE") != "1":
    _config, _logger, _app_data = bootstrap_standalone_page("Executive Report")
    show(
        _app_data['df'],
        _app_data['metrics'],
        _app_data['metrics_loaded'],
        _config,
        _logger
    )
