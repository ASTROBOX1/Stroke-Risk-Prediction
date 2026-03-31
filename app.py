"""
═══════════════════════════════════════════════════════════════════════════════
 🧠 Stroke Prediction — Professional Analytics Dashboard
 Company: Healthcare Analytics Division
 Version: 2.1 (Refactored)
 Description: Multi-page interactive dashboard with EDA, model performance,
              prediction engine, and automated reporting.
═══════════════════════════════════════════════════════════════════════════════
"""

import logging
import importlib
import os
import sys
from typing import Any
import streamlit as st

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (
    load_config,
    setup_logging,
    CSS_STYLES,
    load_dashboard_state,
)

# ═══════════════════════════════════════════════════════════
# INITIALIZATION
# ═══════════════════════════════════════════════════════════

# Load configuration
CONFIG = load_config("config.yaml")

# Setup logging
LOGGER = setup_logging(CONFIG)
LOGGER.info("Streamlit app started")

# Page configuration
st.set_page_config(
    page_title=CONFIG['streamlit']['page_title'],
    page_icon=CONFIG['streamlit']['page_icon'],
    layout=CONFIG['streamlit']['layout'],
    initial_sidebar_state=CONFIG['streamlit']['initial_sidebar_state']
)
st.set_option("client.showSidebarNavigation", False)

# Apply CSS styling
st.markdown(CSS_STYLES, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════

def render_embedded_page(module_name: str, *args: Any) -> None:
    """Import page modules without triggering their standalone bootstrap."""
    os.environ["STROKE_APP_EMBEDDED_PAGE"] = "1"
    try:
        module = importlib.import_module(module_name)
    finally:
        os.environ.pop("STROKE_APP_EMBEDDED_PAGE", None)

    module.show(*args)


# Initialize all data
app_data = load_dashboard_state(CONFIG, LOGGER)
df_raw = app_data['df_raw']
df = app_data['df']
model = app_data['model']
model_loaded = app_data['model_loaded']
metrics = app_data['metrics']
metrics_loaded = app_data['metrics_loaded']
fi_data = app_data['fi_data']
fi_loaded = app_data['fi_loaded']

# ═══════════════════════════════════════════════════════════
# PAGE NAVIGATION
# ═══════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    # 🧠 Stroke Analytics
    ### Healthcare AI Platform
    ---
    """)

    page = st.radio(
        "📑 Navigation",
        [
            "🏠 Executive Overview",
            "🔍 Data Explorer",
            "📊 Exploratory Analysis",
            "🤖 Model Performance",
            "⚕️ Stroke Risk Predictor",
            "📋 Report Summary"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown(f"""
    **Dataset Info**
    - 📊 Records: {len(df_raw):,}
    - 📅 Report: March 2026
    - 🏥 Source: Healthcare DB
    """)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; opacity:0.6; font-size:0.8em; color: #94A3B8;">
    Healthcare Analytics Division<br>
    Data Science Team © 2026
    </div>
    """, unsafe_allow_html=True)

# Import and display pages
if page == "🏠 Executive Overview":
    render_embedded_page("pages.overview", df, CONFIG, LOGGER)

elif page == "🔍 Data Explorer":
    render_embedded_page("pages.explorer", df, CONFIG, LOGGER)

elif page == "📊 Exploratory Analysis":
    render_embedded_page("pages.eda", df, CONFIG, LOGGER)

elif page == "🤖 Model Performance":
    render_embedded_page(
        "pages.model_performance",
        metrics,
        fi_data,
        model_loaded,
        metrics_loaded,
        fi_loaded,
        CONFIG,
        LOGGER
    )

elif page == "⚕️ Stroke Risk Predictor":
    render_embedded_page("pages.predictor", model, model_loaded, df, CONFIG, LOGGER)

elif page == "📋 Report Summary":
    render_embedded_page("pages.report", df, metrics, metrics_loaded, CONFIG, LOGGER)

LOGGER.info(f"User navigated to: {page}")
