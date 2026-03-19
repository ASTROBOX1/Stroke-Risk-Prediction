"""
Executive Overview page - Dashboard with KPIs and key visualizations.
"""

import logging
import os
from typing import Dict, Any
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from utils import kpi_card, section_divider, COLORS, PLOTLY_TEMPLATE, bootstrap_standalone_page


def show(df: pd.DataFrame, config: Dict[str, Any], logger: logging.Logger) -> None:
    """Display Executive Overview page."""
    try:
        st.markdown("# 🏠 Executive Overview")
        st.markdown("*Real-time analytics dashboard for stroke risk assessment and patient demographics.*")
        section_divider()

        # KPI Row
        c1, c2, c3, c4, c5, c6 = st.columns(6)

        stroke_rate = df['stroke'].mean() * 100
        avg_age = df['age'].mean()
        avg_bmi = df['bmi'].mean()
        avg_glucose = df['avg_glucose_level'].mean()
        hyp_rate = df['hypertension'].mean() * 100
        hd_rate = df['heart_disease'].mean() * 100

        with c1:
            kpi_card("👥", f"{len(df):,}", "Total Patients", "kpi-card-blue")
        with c2:
            kpi_card("⚠️", f"{stroke_rate:.1f}%", "Stroke Rate", "kpi-card-red")
        with c3:
            kpi_card("🎂", f"{avg_age:.0f}", "Avg Age", "kpi-card-purple")
        with c4:
            kpi_card("⚖️", f"{avg_bmi:.1f}", "Avg BMI", "kpi-card-orange")
        with c5:
            kpi_card("🩸", f"{hyp_rate:.1f}%", "Hypertension", "kpi-card-dark")
        with c6:
            kpi_card("❤️", f"{hd_rate:.1f}%", "Heart Disease", "kpi-card-dark")

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts Row 1
        col1, col2 = st.columns(2)

        with col1:
            # Age distribution
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df[df['stroke']==0]['age'], name='No Stroke',
                marker_color=COLORS['no_stroke'], opacity=0.7, nbinsx=30
            ))
            fig.add_trace(go.Histogram(
                x=df[df['stroke']==1]['age'], name='Stroke',
                marker_color=COLORS['stroke'], opacity=0.8, nbinsx=30
            ))
            fig.update_layout(
                title='Age Distribution by Stroke Status',
                xaxis_title='Age', yaxis_title='Count',
                barmode='overlay', template=PLOTLY_TEMPLATE,
                legend=dict(x=0.02, y=0.98),
                margin=dict(t=50, b=40, l=50, r=20),
                height=380
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Stroke by work type
            ct = pd.crosstab(df['work_type'], df['stroke'], normalize='index') * 100
            ct.columns = ['No Stroke %', 'Stroke %']
            ct = ct.sort_values('Stroke %', ascending=True)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=ct.index, x=ct['No Stroke %'], name='No Stroke',
                marker_color=COLORS['no_stroke'], orientation='h', opacity=0.85
            ))
            fig.add_trace(go.Bar(
                y=ct.index, x=ct['Stroke %'], name='Stroke',
                marker_color=COLORS['stroke'], orientation='h', opacity=0.85
            ))
            fig.update_layout(
                title='Stroke Rate by Work Type',
                xaxis_title='Percentage (%)', barmode='stack',
                template=PLOTLY_TEMPLATE,
                legend=dict(x=0.7, y=0.05),
                margin=dict(t=50, b=40, l=100, r=20),
                height=380
            )
            st.plotly_chart(fig, use_container_width=True)

        # Charts Row 2
        col1, col2 = st.columns(2)

        with col1:
            # Glucose vs BMI scatter
            sample = df.sample(min(1000, len(df)), random_state=42)
            fig = px.scatter(
                sample, x='avg_glucose_level', y='bmi', color='stroke',
                color_discrete_map={0: COLORS['no_stroke'], 1: COLORS['stroke']},
                labels={'avg_glucose_level': 'Avg Glucose Level', 'bmi': 'BMI', 'stroke': 'Stroke'},
                title='Glucose Level vs BMI',
                opacity=0.6,
                category_orders={'stroke': [0, 1]}
            )
            fig.update_layout(
                template=PLOTLY_TEMPLATE,
                margin=dict(t=50, b=40, l=50, r=20),
                height=380,
                legend_title_text='Stroke'
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Gender & marriage breakdown
            fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'pie'}, {'type': 'pie'}]],
                                subplot_titles=('Gender Distribution', 'Marital Status'))

            gender_counts = df['gender'].value_counts()
            fig.add_trace(go.Pie(
                labels=gender_counts.index, values=gender_counts.values,
                marker=dict(colors=['#667EEA', '#F472B6']),
                hole=0.45, textinfo='label+percent'
            ), row=1, col=1)

            married_counts = df['ever_married'].value_counts()
            fig.add_trace(go.Pie(
                labels=married_counts.index, values=married_counts.values,
                marker=dict(colors=['#2ECC71', '#F39C12']),
                hole=0.45, textinfo='label+percent'
            ), row=1, col=2)

            fig.update_layout(
                template=PLOTLY_TEMPLATE,
                margin=dict(t=50, b=20, l=20, r=20),
                height=380,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

        logger.info("Executive Overview page displayed successfully")

    except Exception as e:
        logger.error(f"Error in Executive Overview page: {str(e)}")
        st.error(f"❌ Error displaying Executive Overview: {str(e)}")


if os.getenv("STROKE_APP_EMBEDDED_PAGE") != "1":
    _config, _logger, _app_data = bootstrap_standalone_page("Executive Overview")
    show(_app_data['df'], _config, _logger)
