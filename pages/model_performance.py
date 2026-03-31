"""
Model Performance page - Detailed evaluation of trained models.
"""

import logging
import os
from typing import Dict, Any, Optional
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from utils import section_divider, PLOTLY_TEMPLATE, MODEL_COLORS, bootstrap_standalone_page


def show(metrics: Optional[Dict], fi_data: Optional[Dict], model_loaded: bool,
         metrics_loaded: bool, fi_loaded: bool, config: Dict[str, Any],
         logger: logging.Logger) -> None:
    """Display Model Performance page."""
    try:
        st.markdown("# 🤖 Model Performance")
        st.markdown("*Detailed evaluation of trained machine learning models.*")
        section_divider()

        if not metrics_loaded:
            st.error("⚠️ Metrics not found. Please run `python train_model.py` first.")
            logger.warning("Metrics not loaded")
            st.stop()

        # Best model banner
        best = metrics['best_model']
        best_auc = metrics['models'][best]['auc_roc']
        st.success(f"🏆 **Best Model: {best}** — AUC-ROC: {best_auc:.4f}")

        st.markdown("<br>", unsafe_allow_html=True)

        # Metrics comparison
        model_names = list(metrics['models'].keys())

        mc1, mc2, mc3 = st.columns(3)
        for i, (col, name) in enumerate(zip([mc1, mc2, mc3], model_names)):
            m = metrics['models'][name]
            with col:
                css = "kpi-card-green" if name == best else "kpi-card-blue"
                st.markdown(f"""
                <div class="kpi-card {css}" style="text-align:left; padding:20px;">
                    <h3 style="color:white; margin:0 0 16px;">{name} {'🏆' if name == best else ''}</h3>
                    <table style="width:100%; color:white;">
                        <tr><td>Accuracy</td><td style="text-align:right; font-weight:700;">{m['accuracy']:.4f}</td></tr>
                        <tr><td>Precision</td><td style="text-align:right; font-weight:700;">{m['precision']:.4f}</td></tr>
                        <tr><td>Recall</td><td style="text-align:right; font-weight:700;">{m['recall']:.4f}</td></tr>
                        <tr><td>F1-Score</td><td style="text-align:right; font-weight:700;">{m['f1_score']:.4f}</td></tr>
                        <tr><td>AUC-ROC</td><td style="text-align:right; font-weight:700;">{m['auc_roc']:.4f}</td></tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ROC Curves
        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure()
            for i, name in enumerate(model_names):
                m = metrics['models'][name]
                fig.add_trace(go.Scatter(
                    x=m['roc_curve']['fpr'], y=m['roc_curve']['tpr'],
                    mode='lines', name=f"{name} (AUC={m['auc_roc']:.4f})",
                    line=dict(color=MODEL_COLORS[i], width=3)
                ))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode='lines',
                name='Random', line=dict(color='gray', dash='dash', width=1)
            ))
            fig.update_layout(
                title='ROC Curves',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                template=PLOTLY_TEMPLATE,
                height=450,
                legend=dict(x=0.5, y=0.05)
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = go.Figure()
            for i, name in enumerate(model_names):
                m = metrics['models'][name]
                fig.add_trace(go.Scatter(
                    x=m['pr_curve']['recall'], y=m['pr_curve']['precision'],
                    mode='lines', name=name,
                    line=dict(color=MODEL_COLORS[i], width=3)
                ))
            fig.update_layout(
                title='Precision-Recall Curves',
                xaxis_title='Recall',
                yaxis_title='Precision',
                template=PLOTLY_TEMPLATE,
                height=450,
                legend=dict(x=0.5, y=0.98)
            )
            st.plotly_chart(fig, use_container_width=True)

        # Confusion Matrices
        st.markdown("### 📊 Confusion Matrices")
        cm_cols = st.columns(3)

        for i, (col, name) in enumerate(zip(cm_cols, model_names)):
            with col:
                cm = np.array(metrics['models'][name]['confusion_matrix'])

                fig = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=['Predicted No', 'Predicted Yes'],
                    y=['Actual No', 'Actual Yes'],
                    colorscale='Blues',
                    text=cm,
                    texttemplate='%{text}',
                    textfont={"size": 20},
                    showscale=False
                ))
                fig.update_layout(
                    title=f'{name}',
                    template=PLOTLY_TEMPLATE,
                    height=350,
                    margin=dict(t=50, b=50)
                )
                st.plotly_chart(fig, use_container_width=True)

        # Feature Importance
        if fi_loaded:
            section_divider()
            st.markdown("### 🎯 Feature Importance")

            fi_sorted = dict(sorted(fi_data.items(), key=lambda x: x[1], reverse=False))

            fig = go.Figure(go.Bar(
                y=list(fi_sorted.keys()),
                x=list(fi_sorted.values()),
                orientation='h',
                marker=dict(
                    color=list(fi_sorted.values()),
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='Importance')
                ),
                text=[f'{v:.4f}' for v in fi_sorted.values()],
                textposition='outside'
            ))
            fig.update_layout(
                title=f'Feature Importance — {best}',
                xaxis_title='Importance Score',
                template=PLOTLY_TEMPLATE,
                height=max(400, len(fi_sorted) * 28),
                margin=dict(t=50, l=250, r=80)
            )
            st.plotly_chart(fig, use_container_width=True)

        # Model comparison
        section_divider()
        st.markdown("### 📊 Model Comparison")

        metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']

        fig = go.Figure()
        for i, name in enumerate(model_names):
            m = metrics['models'][name]
            values = [m[mn] for mn in metric_names]
            fig.add_trace(go.Bar(
                name=name,
                x=metric_labels,
                y=values,
                marker_color=MODEL_COLORS[i],
                opacity=0.85,
                text=[f'{v:.3f}' for v in values],
                textposition='outside'
            ))

        fig.update_layout(
            title='Model Performance Comparison',
            yaxis_title='Score',
            barmode='group',
            template=PLOTLY_TEMPLATE,
            height=450,
            yaxis=dict(range=[0, 1.15]),
            legend=dict(x=0.7, y=0.98)
        )
        st.plotly_chart(fig, use_container_width=True)

        logger.info("Model Performance page displayed successfully")

    except Exception as e:
        logger.error(f"Error in Model Performance page: {str(e)}")
        st.error(f"❌ Error in Model Performance: {str(e)}")


if os.getenv("STROKE_APP_EMBEDDED_PAGE") != "1":
    _config, _logger, _app_data = bootstrap_standalone_page("Model Performance")
    show(
        _app_data['metrics'],
        _app_data['fi_data'],
        _app_data['model_loaded'],
        _app_data['metrics_loaded'],
        _app_data['fi_loaded'],
        _config,
        _logger
    )
