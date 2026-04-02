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

    # File handler with log rotation
    try:
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_file, 
            encoding='utf-8',
            maxBytes=10*1024*1024,  # 10 MB per file
            backupCount=5  # Keep 5 backup files
        )
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


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate that required configuration keys exist.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If required keys are missing
    """
    required_sections = ['paths', 'logging', 'validation', 'streamlit']
    missing_sections = [key for key in required_sections if key not in config]
    
    if missing_sections:
        raise ValueError(f"Missing required config sections: {missing_sections}")
    
    # Validate paths section
    if 'data_dir' not in config['paths'] or 'models_dir' not in config['paths']:
        raise ValueError("Config 'paths' section must contain 'data_dir' and 'models_dir'")
    
    # Validate logging section
    if 'level' not in config['logging']:
        raise ValueError("Config 'logging' section must contain 'level'")


# ═══════════════════════════════════════════════════════════
# DATA LOADING WITH ERROR HANDLING
# ═══════════════════════════════════════════════════════════

# Get logger instance
_logger = logging.getLogger(__name__)


@st.cache_data(ttl=3600)  # Cache for 1 hour to prevent stale data
def load_data(filepath: str) -> Optional[pd.DataFrame]:
    """
    Load data with error handling, validation, and logging.

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

        # Load with optimized settings
        df = pd.read_csv(
            resolved_path,
            low_memory=False,  # Avoid mixed type warnings
            na_values=['', 'NA', 'N/A', 'null', 'NULL']  # Standardize NaN values
        )
        
        # Basic validation
        if df.empty:
            _logger.warning(f"Loaded empty DataFrame from {resolved_path}")
            st.warning("⚠️ Data file is empty")
            return None
            
        _logger.info(f"Loaded {len(df):,} rows × {df.shape[1]} columns from {resolved_path}")
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
# COLOR & STYLING CONSTANTS & UI COMPONENTS
# ═══════════════════════════════════════════════════════════

def kpi_card(icon: str, value: str, label: str, css_class: str = ""):
    st.markdown(f"""
    <div class="kpi-card {css_class}">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

def section_divider():
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

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
/* ═══════════════════════════════════════════════════════════
   MODERN UI DESIGN SYSTEM - Stroke Risk Prediction Platform
   ═══════════════════════════════════════════════════════════ */

/* Import Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ===== ROOT VARIABLES ===== */
:root {
    --primary: #667eea;
    --primary-dark: #5a67d8;
    --secondary: #764ba2;
    --accent: #f093fb;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
    --info: #3b82f6;
    
    --bg-primary: #0f172a;
    --bg-secondary: #1e293b;
    --bg-card: #1e293b;
    --bg-glass: rgba(30, 41, 59, 0.7);
    
    --text-primary: #f1f5f9;
    --text-secondary: #cbd5e1;
    --text-muted: #94a3b8;
    
    --border: rgba(148, 163, 184, 0.1);
    --shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3), 0 10px 10px -5px rgba(0, 0, 0, 0.2);
    --shadow-lg: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
    
    --gradient-primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --gradient-accent: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    --gradient-success: linear-gradient(135deg, #10b981 0%, #059669 100%);
    --gradient-blue: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
}

/* ===== GLOBAL STYLES ===== */
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* ===== TYPOGRAPHY ===== */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    color: var(--text-primary) !important;
}

h1 {
    font-size: 2.5rem !important;
    background: var(--gradient-primary);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem !important;
}

p, span, div, label {
    color: var(--text-secondary) !important;
    line-height: 1.6 !important;
}

/* ===== SIDEBAR ===== */
[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border);
    backdrop-filter: blur(10px);
}

[data-testid="stSidebar"] .css-1d391kg, 
[data-testid="stSidebar"] .st-emotion-cache-1d391kg {
    background: transparent !important;
}

/* Sidebar navigation buttons */
[data-testid="stSidebar"] .stRadio > div {
    gap: 0.5rem;
}

[data-testid="stSidebar"] .stRadio label {
    background: rgba(148, 163, 184, 0.05) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
    margin: 4px 0 !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    cursor: pointer !important;
}

[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(102, 126, 234, 0.1) !important;
    border-color: var(--primary) !important;
    transform: translateX(4px) !important;
}

/* ===== KPI CARDS - ENHANCED ===== */
.kpi-card {
    background: var(--bg-glass);
    backdrop-filter: blur(20px);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    box-shadow: var(--shadow);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.kpi-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: var(--gradient-primary);
    opacity: 0;
    transition: opacity 0.3s ease;
}

.kpi-card:hover {
    transform: translateY(-6px) scale(1.02);
    box-shadow: var(--shadow-lg);
}

.kpi-card:hover::before {
    opacity: 1;
}

.kpi-icon {
    font-size: 2.5rem;
    margin-bottom: 12px;
    filter: drop-shadow(0 4px 6px rgba(0, 0, 0, 0.3));
}

.kpi-value {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin: 8px 0;
    background: var(--gradient-primary);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.kpi-label {
    font-size: 0.875rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
}

/* KPI Card Variants */
.kpi-card-blue { border-top: 3px solid #3b82f6; }
.kpi-card-red { border-top: 3px solid #ef4444; }
.kpi-card-green { border-top: 3px solid #10b981; }
.kpi-card-purple { border-top: 3px solid #8b5cf6; }
.kpi-card-orange { border-top: 3px solid #f59e0b; }
.kpi-card-dark { border-top: 3px solid #64748b; }

/* ===== BUTTONS ===== */
.stButton button {
    background: var(--gradient-primary) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 32px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 4px 14px rgba(102, 126, 234, 0.4) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.stButton button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
}

/* ===== METRICS ===== */
[data-testid="stMetric"] {
    background: var(--bg-glass);
    backdrop-filter: blur(20px);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px;
    box-shadow: var(--shadow);
}

[data-testid="stMetric"] label {
    font-size: 0.875rem !important;
    font-weight: 600 !important;
    color: var(--text-muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 2rem !important;
    font-weight: 800 !important;
    background: var(--gradient-primary);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* ===== SECTION DIVIDER ===== */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 2rem 0;
}

/* ===== ANIMATIONS ===== */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.fade-in {
    animation: fadeIn 0.6s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ===== HIDE STREAMLIT BRANDING ===== */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
/* Keep header visible so sidebar toggle button works */
header[data-testid="stHeader"] {background: transparent !important;}
[data-testid="collapsedControl"] {display: block !important; visibility: visible !important;}

</style>
"""