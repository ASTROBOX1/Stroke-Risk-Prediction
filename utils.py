"""
Utility functions for the Stroke Prediction Analytics Dashboard.
Includes logging setup, config loading, and helper functions.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union, Tuple
import yaml
import json
import streamlit as st
import pandas as pd
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent


# ═══════════════════════════════════════════════════════════
# LOGGING CONFIGURATION
# ═══════════════════════════════════════════════════════════

def resolve_path(path: Union[str, os.PathLike[str]]) -> Path:
    """Resolve project paths relative to the repository root."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate

def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """
    Configure logging with file and console handlers.

    Args:
        config: Configuration dictionary from config.yaml

    Returns:
        Configured logger instance
    """
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = resolve_path(log_config.get('file', 'logs/app.log'))

    # Create logs directory if it doesn't exist
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    logger.propagate = False

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    # File handler
    try:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not create file handler: {e}")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)

    return logger


# ═══════════════════════════════════════════════════════════
# CONFIGURATION LOADING
# ═══════════════════════════════════════════════════════════

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config.yaml

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    resolved_config_path = resolve_path(config_path)
    if not resolved_config_path.exists():
        raise FileNotFoundError(f"Config file not found: {resolved_config_path}")

    with resolved_config_path.open('r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


# ═══════════════════════════════════════════════════════════
# DATA LOADING WITH ERROR HANDLING
# ═══════════════════════════════════════════════════════════

# Get logger instance
_logger = logging.getLogger(__name__)


@st.cache_data
def load_data(filepath: str) -> Optional[pd.DataFrame]:
    """
    Load data with error handling and logging.

    Args:
        filepath: Path to CSV file

    Returns:
        Loaded DataFrame or None if error
    """
    try:
        resolved_path = resolve_path(filepath)
        if not resolved_path.exists():
            _logger.error(f"Data file not found: {resolved_path}")
            st.error(f"❌ Data file not found: {resolved_path}")
            return None

        df = pd.read_csv(resolved_path)
        _logger.info(f"Loaded {len(df)} rows × {df.shape[1]} columns from {resolved_path}")
        return df
    except Exception as e:
        _logger.error(f"Error loading data: {str(e)}")
        st.error(f"❌ Error loading data: {str(e)}")
        return None


@st.cache_resource
def load_model(model_path: str):
    """
    Load trained model with error handling.

    Args:
        model_path: Path to joblib model file

    Returns:
        Loaded model or None if error
    """
    try:
        resolved_path = resolve_path(model_path)
        if not resolved_path.exists():
            _logger.error(f"Model file not found: {resolved_path}")
            return None

        model = joblib.load(resolved_path)
        _logger.info(f"Loaded model from {resolved_path}")
        return model
    except Exception as e:
        _logger.error(f"Error loading model: {str(e)}")
        return None


@st.cache_data
def load_json_file(filepath: str) -> Optional[Dict]:
    """
    Load JSON file with error handling.

    Args:
        filepath: Path to JSON file

    Returns:
        Loaded JSON data or None if error
    """
    try:
        resolved_path = resolve_path(filepath)
        if not resolved_path.exists():
            _logger.warning(f"JSON file not found: {resolved_path}")
            return None

        with resolved_path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        _logger.info(f"Loaded JSON from {resolved_path}")
        return data
    except Exception as e:
        _logger.error(f"Error loading JSON from {resolved_path}: {str(e)}")
        return None


# ═══════════════════════════════════════════════════════════
# DATA PREPROCESSING
# ═══════════════════════════════════════════════════════════

def preprocess_data(
    df: pd.DataFrame,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Preprocess data for analysis.

    Args:
        df: Raw DataFrame

    Returns:
        Preprocessed DataFrame
    """
    logger = logger or _logger
    df = df.copy()

    # Drop ID column if exists
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
        logger.info("Dropped 'id' column")

    # Remove 'Other' gender
    if 'gender' in df.columns:
        initial_rows = len(df)
        df = df[df['gender'] != 'Other'].copy()
        removed = initial_rows - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} rows with gender='Other'")

    # Impute BMI
    if 'bmi' in df.columns:
        initial_nulls = df['bmi'].isna().sum()
        bmi_median = df['bmi'].median()
        if initial_nulls > 0 and pd.isna(bmi_median):
            logger.warning("BMI column contains only missing values; skipping median imputation")
        else:
            df['bmi'] = df['bmi'].fillna(bmi_median)
            logger.info(f"Imputed {initial_nulls} BMI values with median: {bmi_median:.2f}")

    logger.info(f"Preprocessing complete. Final shape: {df.shape}")
    return df


def load_dashboard_state(
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """Load the shared dashboard assets needed by pages and the main app."""
    logger = logger or _logger

    df_raw = load_data(config['paths']['data'])
    if df_raw is None:
        st.stop()

    df = preprocess_data(df_raw, logger)

    model_path = Path(config['paths']['models_dir']) / config['paths']['model_file']
    metrics_path = Path(config['paths']['models_dir']) / config['paths']['metrics_file']
    fi_path = Path(config['paths']['models_dir']) / config['paths']['feature_importance_file']

    model = load_model(str(model_path))
    metrics = load_json_file(str(metrics_path))
    fi_data = load_json_file(str(fi_path))

    return {
        'df_raw': df_raw,
        'df': df,
        'model': model,
        'model_loaded': model is not None,
        'metrics': metrics,
        'metrics_loaded': metrics is not None,
        'fi_data': fi_data,
        'fi_loaded': fi_data is not None,
    }


def bootstrap_standalone_page(
    page_title_suffix: str
) -> Tuple[Dict[str, Any], logging.Logger, Dict[str, Any]]:
    """Prepare a page module so it can run directly as a Streamlit page."""
    config = load_config("config.yaml")
    logger = setup_logging(config)

    st.set_page_config(
        page_title=f"{config['streamlit']['page_title']} • {page_title_suffix}",
        page_icon=config['streamlit']['page_icon'],
        layout=config['streamlit']['layout'],
        initial_sidebar_state=config['streamlit']['initial_sidebar_state']
    )
    st.set_option("client.showSidebarNavigation", False)
    st.markdown(CSS_STYLES, unsafe_allow_html=True)

    logger.info(f"Standalone page bootstrapped: {page_title_suffix}")
    app_data = load_dashboard_state(config, logger)
    return config, logger, app_data


# ═══════════════════════════════════════════════════════════
# INPUT VALIDATION
# ═══════════════════════════════════════════════════════════

def validate_input(
    data: Dict[str, Any],
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None
) -> Tuple[bool, str]:
    """Validate user input for stroke prediction.

    Args:
        data: Input data dictionary
        config: Configuration dictionary
        logger: Optional logger for recording validation messages

    Returns:
        Tuple of (is_valid, error_message)
    """
    validation = config.get('validation', {})
    logger = logger or _logger

    # Validate age
    age = data.get('age', 0)
    age_min = validation.get('age_min', 0)
    age_max = validation.get('age_max', 120)

    if not (age_min <= age <= age_max):
        msg = f"Age must be between {age_min} and {age_max}"
        logger.warning(f"Input validation failed: {msg}")
        return False, msg

    # Validate BMI
    bmi = data.get('bmi', 0)
    bmi_min = validation.get('bmi_min', 10.0)
    bmi_max = validation.get('bmi_max', 100.0)

    if not (bmi_min <= bmi <= bmi_max):
        msg = f"BMI must be between {bmi_min} and {bmi_max}"
        logger.warning(f"Input validation failed: {msg}")
        return False, msg

    # Validate glucose
    glucose = data.get('avg_glucose_level', 0)
    glucose_min = validation.get('glucose_min', 0.0)
    glucose_max = validation.get('glucose_max', 300.0)

    if not (glucose_min <= glucose <= glucose_max):
        msg = f"Glucose must be between {glucose_min} and {glucose_max}"
        logger.warning(f"Input validation failed: {msg}")
        return False, msg

    logger.info("Input validation passed")
    return True, ""


# ═══════════════════════════════════════════════════════════
# COLOR & STYLING CONSTANTS
# ═══════════════════════════════════════════════════════════

COLORS = {
    'primary':     '#667EEA',
    'secondary':   '#764BA2',
    'blue':        '#2E86AB',
    'green':       '#2ECC71',
    'red':         '#E74C3C',
    'orange':      '#F39C12',
    'purple':      '#9B59B6',
    'dark':        '#2C3E50',
    'no_stroke':   '#2ECC71',
    'stroke':      '#E74C3C',
}

PLOTLY_TEMPLATE = 'plotly_white'
MODEL_COLORS = ['#2E86AB', '#A23B72', '#F39C12']

CSS_STYLES = """
<style>
    /* ── Typography ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown li,
    [data-testid="stSidebar"] .stMarkdown label,
    [data-testid="stSidebar"] .stSelectbox label {
        color: #E2E8F0 !important;
    }

    /* ── KPI Cards ── */
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 24px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4);
    }
    .kpi-card-blue {
        background: linear-gradient(135deg, #2E86AB 0%, #1B5E7F 100%);
        box-shadow: 0 10px 30px rgba(46, 134, 171, 0.3);
    }
    .kpi-card-green {
        background: linear-gradient(135deg, #2ECC71 0%, #1A9A54 100%);
        box-shadow: 0 10px 30px rgba(46, 204, 113, 0.3);
    }
    .kpi-card-red {
        background: linear-gradient(135deg, #E74C3C 0%, #C0392B 100%);
        box-shadow: 0 10px 30px rgba(231, 76, 60, 0.3);
    }
    .kpi-card-orange {
        background: linear-gradient(135deg, #F39C12 0%, #D68910 100%);
        box-shadow: 0 10px 30px rgba(243, 156, 18, 0.3);
    }
    .kpi-card-purple {
        background: linear-gradient(135deg, #9B59B6 0%, #7D3C98 100%);
        box-shadow: 0 10px 30px rgba(155, 89, 182, 0.3);
    }
    .kpi-card-dark {
        background: linear-gradient(135deg, #2C3E50 0%, #1A252F 100%);
        box-shadow: 0 10px 30px rgba(44, 62, 80, 0.3);
    }
    .kpi-value {
        font-size: 2.2em;
        font-weight: 800;
        margin: 8px 0 4px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .kpi-label {
        font-size: 0.9em;
        font-weight: 500;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .kpi-icon {
        font-size: 1.8em;
        margin-bottom: 4px;
    }

    /* ── Dividers ── */
    .section-divider {
        border: none;
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2, #667eea);
        border-radius: 2px;
        margin: 30px 0;
    }

    /* ── Report Card ── */
    .report-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 20px 24px;
        margin: 12px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .report-card h4 {
        color: #1E293B;
        margin-top: 0;
    }

    /* ── Risk Gauge ── */
    .risk-high {
        background: linear-gradient(135deg, #E74C3C, #C0392B);
        padding: 30px;
        border-radius: 16px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(231, 76, 60, 0.4);
    }
    .risk-moderate {
        background: linear-gradient(135deg, #F39C12, #D68910);
        padding: 30px;
        border-radius: 16px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(243, 156, 18, 0.4);
    }
    .risk-low {
        background: linear-gradient(135deg, #2ECC71, #1A9A54);
        padding: 30px;
        border-radius: 16px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(46, 204, 113, 0.4);
    }

    /* ── Hide default footer ── */
    footer {visibility: hidden;}

    /* ── Main area ── */
    .block-container {
        padding-top: 2rem;
    }
</style>
"""