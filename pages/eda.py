"""
Exploratory Data Analysis (EDA) page - Comprehensive visual analysis.
"""

import logging
import os
from typing import Dict, Any
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from utils import section_divider, COLORS, PLOTLY_TEMPLATE, bootstrap_standalone_page


def show(df: pd.DataFrame, config: Dict[str, Any], logger: logging.Logger) -> None:
    """Display Exploratory Data Analysis page."""
    try:
        st.markdown("# 📊 Exploratory Data Analysis")
        st.markdown("*Comprehensive visual analysis of patient demographics and stroke risk factors.*")
        section_divider()

        # Numerical Distributions
        st.markdown("### 📈 Numerical Feature Distributions")

        num_cols = ['age', 'avg_glucose_level', 'bmi']
        tabs = st.tabs(["Age", "Glucose Level", "BMI"])

        for tab, col in zip(tabs, num_cols):
            with tab:
                c1, c2 = st.columns([2, 1])
                with c1:
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=df[df['stroke']==0][col], name='No Stroke',
                        marker_color=COLORS['no_stroke'], opacity=0.6, nbinsx=40
                    ))
                    fig.add_trace(go.Histogram(
                        x=df[df['stroke']==1][col], name='Stroke',
                        marker_color=COLORS['stroke'], opacity=0.8, nbinsx=40
                    ))
                    fig.add_vline(x=df[col].mean(), line_dash="dash", line_color="#E74C3C",
                                  annotation_text=f"Mean: {df[col].mean():.1f}")
                    fig.add_vline(x=df[col].median(), line_dash="dot", line_color="#2ECC71",
                                  annotation_text=f"Median: {df[col].median():.1f}")
                    fig.update_layout(
                        title=f'{col} Distribution by Stroke Status',
                        barmode='overlay', template=PLOTLY_TEMPLATE,
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with c2:
                    fig = go.Figure()
                    fig.add_trace(go.Box(
                        y=df[df['stroke']==0][col], name='No Stroke',
                        marker_color=COLORS['no_stroke']
                    ))
                    fig.add_trace(go.Box(
                        y=df[df['stroke']==1][col], name='Stroke',
                        marker_color=COLORS['stroke']
                    ))
                    fig.update_layout(
                        title=f'{col} Box Plot',
                        template=PLOTLY_TEMPLATE,
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)

        section_divider()

        # Categorical Analysis
        st.markdown("### 📊 Categorical Features vs Stroke")

        cat_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status',
                       'hypertension', 'heart_disease']

        col1, col2 = st.columns(2)

        for i, feat in enumerate(cat_features):
            with col1 if i % 2 == 0 else col2:
                ct = pd.crosstab(df[feat], df['stroke'], normalize='index') * 100
                ct.columns = ['No Stroke', 'Stroke']
                ct = ct.sort_values('Stroke', ascending=True)

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=ct.index.astype(str), y=ct['No Stroke'], name='No Stroke',
                    marker_color=COLORS['no_stroke'], opacity=0.85
                ))
                fig.add_trace(go.Bar(
                    x=ct.index.astype(str), y=ct['Stroke'], name='Stroke',
                    marker_color=COLORS['stroke'], opacity=0.85
                ))
                fig.update_layout(
                    title=f'Stroke Rate by {feat}',
                    barmode='stack',
                    yaxis_title='Percentage (%)',
                    template=PLOTLY_TEMPLATE,
                    height=350,
                    margin=dict(t=50, b=60)
                )
                st.plotly_chart(fig, use_container_width=True)

        section_divider()

        # Correlation Analysis
        st.markdown("### 🔗 Correlation Analysis")

        df_corr = df.copy()
        df_corr['gender'] = df_corr['gender'].map({'Male': 1, 'Female': 0})
        df_corr['ever_married'] = df_corr['ever_married'].map({'Yes': 1, 'No': 0})
        df_corr['Residence_type'] = df_corr['Residence_type'].map({'Urban': 1, 'Rural': 0})
        df_corr = pd.get_dummies(df_corr, columns=['work_type', 'smoking_status'], drop_first=True)

        corr = df_corr.corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale='RdBu_r',
            zmid=0,
            text=np.round(corr.values, 2),
            texttemplate='%{text}',
            textfont={"size": 8},
        ))
        fig.update_layout(
            title='Feature Correlation Matrix',
            template=PLOTLY_TEMPLATE,
            height=600,
            margin=dict(t=50, b=50, l=100, r=50)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Correlation with stroke
        stroke_corr = corr['stroke'].drop('stroke').sort_values(ascending=True)
        fig = go.Figure(go.Bar(
            y=stroke_corr.index,
            x=stroke_corr.values,
            orientation='h',
            marker=dict(
                color=stroke_corr.values,
                colorscale='RdBu_r',
                cmid=0,
                showscale=True,
                colorbar=dict(title='Correlation')
            ),
            text=[f'{v:.3f}' for v in stroke_corr.values],
            textposition='outside'
        ))
        fig.update_layout(
            title='Feature Correlation with Stroke',
            xaxis_title='Pearson Correlation',
            template=PLOTLY_TEMPLATE,
            height=500,
            margin=dict(t=50, l=200)
        )
        st.plotly_chart(fig, use_container_width=True)

        logger.info("EDA page displayed successfully")

    except Exception as e:
        logger.error(f"Error in EDA page: {str(e)}")
        st.error(f"❌ Error in EDA: {str(e)}")


if os.getenv("STROKE_APP_EMBEDDED_PAGE") != "1":
    _config, _logger, _app_data = bootstrap_standalone_page("Exploratory Analysis")
    show(_app_data['df'], _config, _logger)
