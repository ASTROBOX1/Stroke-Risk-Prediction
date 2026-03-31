"""
Unit tests for the utility functions in utils.py.
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path so utils imports work
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pytest
import pandas as pd
import numpy as np
import logging
from utils import resolve_path, validate_input, preprocess_data

def test_resolve_path():
    path = resolve_path("hello.txt")
    assert path.name == "hello.txt"

def test_validate_input():
    config = {
        "validation": {
            "age_min": 0,
            "age_max": 120,
            "bmi_min": 10,
            "bmi_max": 100,
            "glucose_min": 0,
            "glucose_max": 300
        }
    }
    data = {
        "age": 45,
        "bmi": 25,
        "avg_glucose_level": 100
    }
    logger = logging.getLogger("test")
    valid, msg = validate_input(data, config, logger)
    assert valid is True
    assert msg == ""

    # Test invalid age
    invalid_data = data.copy()
    invalid_data["age"] = 150
    valid, msg = validate_input(invalid_data, config, logger)
    assert valid is False
    assert "Age must be between" in msg

def test_preprocess_data():
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "gender": ["Male", "Other", "Female"],
        "bmi": [22.0, np.nan, 28.0]
    })
    logger = logging.getLogger("test")
    processed = preprocess_data(df, logger)
    
    assert "id" not in processed.columns
    assert "Other" not in processed["gender"].values
    assert processed["bmi"].isnull().sum() == 0
    assert len(processed) == 2

