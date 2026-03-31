"""
Data Explorer page - Filter and explore the dataset interactively.
"""

import logging
import os
from typing import Dict, Any
import pandas as pd
import streamlit as st
from utils import kpi_card, section_divider, bootstrap_standalone_page


def show(df: pd.DataFrame, config: Dict[str, Any], logger: logging.Logger) -> None:
    """Display Data Explorer page."""
    try:
        st.markdown("# 🔍 Data Explorer")
        st.markdown("*Filter, explore, and download the dataset interactively.*")
        section_divider()

        # Filters
        with st.expander("🎛️ Filters", expanded=True):
            fc1, fc2, fc3, fc4 = st.columns(4)

            with fc1:
                age_range = st.slider("Age Range", 0, 120, (0, 120))
            with fc2:
                gender_filter = st.multiselect("Gender", df['gender'].unique().tolist(),
                                             default=df['gender'].unique().tolist())
            with fc3:
                stroke_filter = st.multiselect("Stroke", [0, 1], default=[0, 1])
            with fc4:
                work_filter = st.multiselect("Work Type", df['work_type'].unique().tolist(),
                                           default=df['work_type'].unique().tolist())

        # Apply filters
        mask = (
            (df['age'] >= age_range[0]) & (df['age'] <= age_range[1]) &
            (df['gender'].isin(gender_filter)) &
            (df['stroke'].isin(stroke_filter)) &
            (df['work_type'].isin(work_filter))
        )
        df_filtered = df[mask]

        # Stats
        st.markdown(f"**Showing {len(df_filtered):,} of {len(df):,} records** ({len(df_filtered)/len(df)*100:.1f}%)")

        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            kpi_card("📊", f"{len(df_filtered):,}", "Records", "kpi-card-blue")
        with mc2:
            kpi_card("⚠️", f"{df_filtered['stroke'].mean()*100:.1f}%", "Stroke Rate", "kpi-card-red")
        with mc3:
            kpi_card("🎂", f"{df_filtered['age'].mean():.0f}", "Avg Age", "kpi-card-purple")
        with mc4:
            kpi_card("⚖️", f"{df_filtered['bmi'].mean():.1f}", "Avg BMI", "kpi-card-orange")

        st.markdown("<br>", unsafe_allow_html=True)

        # Data table
        st.dataframe(
            df_filtered,
            use_container_width=True,
            height=400,
            column_config={
                "stroke": st.column_config.NumberColumn("Stroke", help="1 = stroke, 0 = no stroke"),
                "bmi": st.column_config.NumberColumn("BMI", format="%.1f"),
                "avg_glucose_level": st.column_config.NumberColumn("Glucose", format="%.1f"),
            }
        )

        # Download
        csv = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Filtered Data (CSV)",
            csv, "stroke_data_filtered.csv", "text/csv",
            use_container_width=True
        )

        logger.info(f"Data Explorer: Filtered to {len(df_filtered)} records")

    except Exception as e:
        logger.error(f"Error in Data Explorer page: {str(e)}")
        st.error(f"❌ Error in Data Explorer: {str(e)}")


if os.getenv("STROKE_APP_EMBEDDED_PAGE") != "1":
    _config, _logger, _app_data = bootstrap_standalone_page("Data Explorer")
    show(_app_data['df'], _config, _logger)
