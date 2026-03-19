"""
Stroke Risk Predictor page - Real-time patient risk assessment.
"""

import logging
import os
from typing import Dict, Any, Optional
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from utils import section_divider, validate_input, bootstrap_standalone_page


def show(model: Optional[Any], model_loaded: bool, df: pd.DataFrame,
         config: Dict[str, Any], logger: logging.Logger) -> None:
    """Display Stroke Risk Predictor page."""
    try:
        st.markdown("# ⚕️ Stroke Risk Predictor")
        st.markdown("*Enter patient details to assess stroke risk using our trained ML model.*")
        section_divider()

        if not model_loaded:
            st.error("⚠️ Model not loaded. Please run `python train_model.py` first.")
            logger.warning("Model not loaded for prediction")
            st.stop()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 👤 Personal Details")
            gender = st.selectbox("Gender", ['Male', 'Female'], key='pred_gender')
            age = st.number_input("Age", min_value=0, max_value=120, value=50, key='pred_age')
            ever_married = st.selectbox("Ever Married", ['Yes', 'No'], key='pred_married')
            work_type = st.selectbox("Work Type", ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'],
                                    key='pred_work')
            residence_type = st.selectbox("Residence Type", ['Urban', 'Rural'], key='pred_residence')

        with col2:
            st.markdown("#### 🩺 Medical History")
            hypertension = st.selectbox("Hypertension", ['No', 'Yes'], key='pred_hyp')
            heart_disease = st.selectbox("Heart Disease", ['No', 'Yes'], key='pred_hd')
            avg_glucose_level = st.number_input("Avg Glucose Level (mg/dL)", min_value=0.0, max_value=300.0,
                                               value=100.0, key='pred_gluc')

        with col3:
            st.markdown("#### 🏃 Lifestyle")
            bmi = st.number_input("BMI", min_value=10.0, max_value=100.0, value=28.0, key='pred_bmi')
            smoking_status = st.selectbox("Smoking Status", ['formerly smoked', 'never smoked', 'smokes', 'Unknown'],
                                         key='pred_smoke')

        section_divider()

        predict_col, gauge_col = st.columns([1, 2])

        with predict_col:
            predict_button = st.button("🔬 Analyze Stroke Risk", type="primary", use_container_width=True)

        if predict_button:
            # Prepare input data
            input_data = {
                'gender': gender,
                'age': age,
                'hypertension': 1 if hypertension == 'Yes' else 0,
                'heart_disease': 1 if heart_disease == 'Yes' else 0,
                'ever_married': ever_married,
                'work_type': work_type,
                'Residence_type': residence_type,
                'avg_glucose_level': avg_glucose_level,
                'bmi': bmi,
                'smoking_status': smoking_status
            }

            # Validate input
            is_valid, error_msg = validate_input(input_data, config, logger)
            if not is_valid:
                st.error(f"❌ Input validation failed: {error_msg}")
                return

            try:
                input_df = pd.DataFrame([input_data])
                prediction = model.predict(input_df)[0]
                probability = model.predict_proba(input_df)[0][1] * 100

                logger.info(f"Prediction made: {probability:.2f}% stroke probability")

                with gauge_col:
                    # Risk gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=probability,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Stroke Risk Probability", 'font': {'size': 20}},
                        number={'suffix': '%', 'font': {'size': 40}},
                        gauge={
                            'axis': {'range': [0, 100], 'tickwidth': 2},
                            'bar': {'color': "#2C3E50"},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'steps': [
                                {'range': [0, 15], 'color': '#2ECC71'},
                                {'range': [15, 40], 'color': '#F39C12'},
                                {'range': [40, 100], 'color': '#E74C3C'}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.8,
                                'value': probability
                            }
                        }
                    ))
                    fig.update_layout(height=300, margin=dict(t=40, b=0, l=30, r=30))
                    st.plotly_chart(fig, use_container_width=True)

                # Risk assessment
                if prediction == 1 or probability > 40:
                    css = "risk-high"
                    icon = "🚨"
                    level = "HIGH RISK"
                    msg = "Immediate medical consultation recommended. The model identifies significant stroke risk factors in this patient profile."
                elif probability > 15:
                    css = "risk-moderate"
                    icon = "⚠️"
                    level = "MODERATE RISK"
                    msg = "Regular monitoring advised. Some risk factors are present. Consider lifestyle modifications and regular check-ups."
                else:
                    css = "risk-low"
                    icon = "✅"
                    level = "LOW RISK"
                    msg = "Current health profile suggests low stroke risk. Continue maintaining a healthy lifestyle."

                st.markdown(f"""
                <div class="{css}">
                    <h2 style="margin:0; color:white;">{icon} {level}</h2>
                    <p style="font-size:1.3em; margin:10px 0; color:white;">{probability:.2f}% probability of stroke</p>
                    <p style="opacity:0.9; color:white;">{msg}</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Patient profile summary
                with st.expander("📋 Patient Profile Summary"):
                    prof_col1, prof_col2 = st.columns(2)
                    with prof_col1:
                        st.markdown("**Demographics**")
                        st.write(f"- Gender: {gender}")
                        st.write(f"- Age: {age}")
                        st.write(f"- Married: {ever_married}")
                        st.write(f"- Work: {work_type}")
                        st.write(f"- Residence: {residence_type}")
                    with prof_col2:
                        st.markdown("**Health Metrics**")
                        st.write(f"- Hypertension: {hypertension}")
                        st.write(f"- Heart Disease: {heart_disease}")
                        st.write(f"- Glucose: {avg_glucose_level} mg/dL")
                        st.write(f"- BMI: {bmi}")
                        st.write(f"- Smoking: {smoking_status}")

            except Exception as e:
                logger.error(f"Error during prediction: {str(e)}")
                st.error(f"❌ Error during prediction: {str(e)}")

        logger.info("Stroke Risk Predictor page displayed successfully")

    except Exception as e:
        logger.error(f"Error in Predictor page: {str(e)}")
        st.error(f"❌ Error in Predictor: {str(e)}")


if os.getenv("STROKE_APP_EMBEDDED_PAGE") != "1":
    _config, _logger, _app_data = bootstrap_standalone_page("Stroke Risk Predictor")
    show(
        _app_data['model'],
        _app_data['model_loaded'],
        _app_data['df'],
        _config,
        _logger
    )
