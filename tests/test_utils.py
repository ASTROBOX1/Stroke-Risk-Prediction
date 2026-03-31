"""
Utility functions for Stroke Prediction Analytics Platform
Includes config loading, validation, and preprocessing
"""

import os
import yaml
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Union


# ═══════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════

COLORS = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "stroke": "#d62728",
    "no_stroke": "#2ca02c"
}

PLOTLY_TEMPLATE = "plotly_white"


# ═══════════════════════════════════════════════════════════
# PATH RESOLUTION
# ═══════════════════════════════════════════════════════════

def resolve_path(path: Union[str, os.PathLike]) -> Path:
    """
    Resolve relative paths safely from any working directory.
    """
    path = Path(path)

    if path.is_absolute():
        return path

    # resolve relative to project root
    project_root = Path(__file__).resolve().parent.parent
    return project_root / path


# ═══════════════════════════════════════════════════════════
# CONFIG LOADING
# ═══════════════════════════════════════════════════════════

def load_config(config_path: Union[str, os.PathLike]) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    """
    config_path = resolve_path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config


# ═══════════════════════════════════════════════════════════
# INPUT VALIDATION
# ═══════════════════════════════════════════════════════════

def validate_input(data: Dict[str, Any],
                   config: Dict[str, Any],
                   logger: logging.Logger) -> Tuple[bool, str]:
    """
    Validate user input for stroke prediction.
    """

    validation = config["validation"]

    # Age validation
    age = data.get("age")
    if age is None or not (validation["age_min"] <= age <= validation["age_max"]):
        return False, "Age value is out of allowed range."

    # BMI validation
    bmi = data.get("bmi")
    if bmi is None or not (validation["bmi_min"] <= bmi <= validation["bmi_max"]):
        return False, "BMI value is out of allowed range."

    # Glucose validation
    glucose = data.get("avg_glucose_level")
    if glucose is None or not (validation["glucose_min"] <= glucose <= validation["glucose_max"]):
        return False, "Glucose value is out of allowed range."

    return True, ""


# ═══════════════════════════════════════════════════════════
# DATA PREPROCESSING
# ═══════════════════════════════════════════════════════════

def preprocess_data(df: pd.DataFrame,
                    logger: logging.Logger) -> pd.DataFrame:
    """
    Clean and preprocess stroke dataset.
    """

    df = df.copy()

    # Remove "Other" gender rows
    if "gender" in df.columns:
        df = df[df["gender"] != "Other"]

    # Remove ID column
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Fill missing BMI with median
    if "bmi" in df.columns:
        df["bmi"] = df["bmi"].fillna(df["bmi"].median())

    # Remove duplicates
    df = df.drop_duplicates()

    return df.reset_index(drop=True)
