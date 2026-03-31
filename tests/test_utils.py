"""
Unit tests for Stroke Prediction Analytics Platform
Tests for data loading, preprocessing, and API endpoints
"""

import pytest
import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path
import sys
from fastapi.testclient import TestClient

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api import app
from utils import (
    load_config,
    validate_input,
    preprocess_data,
    COLORS,
    PLOTLY_TEMPLATE
)

# Configure logging
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════

@pytest.fixture
def sample_config():
    """Fixture for sample configuration."""
    return {
        'validation': {
            'age_min': 0,
            'age_max': 120,
            'bmi_min': 10.0,
            'bmi_max': 100.0,
            'glucose_min': 0.0,
            'glucose_max': 300.0
        },
        'paths': {
            'data': 'data/healthcare-dataset-stroke-data.csv',
            'models_dir': 'models',
            'model_file': 'best_stroke_model.joblib'
        }
    }


@pytest.fixture
def sample_patient_input():
    """Fixture for valid patient input."""
    return {
        'gender': 'Male',
        'age': 50,
        'hypertension': 0,
        'heart_disease': 0,
        'ever_married': 'Yes',
        'work_type': 'Private',
        'Residence_type': 'Urban',
        'avg_glucose_level': 100.0,
        'bmi': 28.5,
        'smoking_status': 'never smoked'
    }


@pytest.fixture
def sample_dataframe():
    """Fixture for sample healthcare data."""
    return pd.DataFrame({
        'id': range(1, 11),
        'gender': ['Male', 'Female'] * 5,
        'age': [25, 35, 45, 55, 65, 75, 85, 45, 50, 60],
        'hypertension': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        'heart_disease': [0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        'ever_married': ['Yes', 'No'] * 5,
        'work_type': ['Private', 'Self-employed', 'Govt_job', 'Private', 'Self-employed'] * 2,
        'Residence_type': ['Urban', 'Rural'] * 5,
        'avg_glucose_level': [100.0, 120.0, 95.0, 110.0, 105.0, 115.0, 100.0, 90.0, 125.0, 105.0],
        'bmi': [25.0, 28.5, 30.0, 27.0, 26.5, 29.0, 31.0, 24.0, 32.0, 28.0],
        'smoking_status': [
            'never smoked', 'formerly smoked', 'smokes', 'Unknown', 'never smoked',
            'formerly smoked', 'smokes', 'Unknown', 'never smoked', 'formerly smoked'
        ],
        'stroke': [0, 0, 1, 0, 0, 1, 1, 0, 1, 0]
    })


# ═══════════════════════════════════════════════════════════
# TESTS - UTILS
# ═══════════════════════════════════════════════════════════

class TestConfigLoading:
    """Tests for configuration loading."""

    def test_load_config_success(self, monkeypatch):
        """Test loading config file successfully from a nested working directory."""
        monkeypatch.chdir(str(Path(__file__).resolve().parent))
        config = load_config('config.yaml')
        assert config is not None
        assert 'paths' in config
        assert 'validation' in config

    def test_load_config_missing_file(self):
        """Test error handling for missing config file."""
        with pytest.raises(FileNotFoundError):
            load_config('nonexistent_config.yaml')


class TestInputValidation:
    """Tests for input validation."""

    def test_valid_input(self, sample_patient_input, sample_config):
        """Test validation with valid input."""
        is_valid, msg = validate_input(sample_patient_input, sample_config, logger)
        assert is_valid is True
        assert msg == ""

    def test_invalid_age_too_high(self, sample_patient_input, sample_config):
        """Test validation with age too high."""
        sample_patient_input['age'] = 150
        is_valid, msg = validate_input(sample_patient_input, sample_config, logger)
        assert is_valid is False
        assert "Age" in msg

    def test_invalid_age_negative(self, sample_patient_input, sample_config):
        """Test validation with negative age."""
        sample_patient_input['age'] = -5
        is_valid, msg = validate_input(sample_patient_input, sample_config, logger)
        assert is_valid is False

    def test_invalid_bmi_too_low(self, sample_patient_input, sample_config):
        """Test validation with BMI too low."""
        sample_patient_input['bmi'] = 5.0
        is_valid, msg = validate_input(sample_patient_input, sample_config, logger)
        assert is_valid is False
        assert "BMI" in msg

    def test_invalid_glucose_too_high(self, sample_patient_input, sample_config):
        """Test validation with glucose too high."""
        sample_patient_input['avg_glucose_level'] = 400.0
        is_valid, msg = validate_input(sample_patient_input, sample_config, logger)
        assert is_valid is False
        assert "Glucose" in msg


# ═══════════════════════════════════════════════════════════
# TESTS - DATA PREPROCESSING
# ═══════════════════════════════════════════════════════════

class TestDataPreprocessing:
    """Tests for data preprocessing."""

    def test_preprocess_data_success(self, sample_dataframe):
        """Test successful data preprocessing."""
        df_processed = preprocess_data(sample_dataframe, logger)
        assert df_processed is not None
        assert len(df_processed) > 0
        assert 'id' not in df_processed.columns

    def test_preprocess_data_no_bmi_nans(self, sample_dataframe):
        """Test that NaN BMI values are imputed."""
        sample_dataframe.loc[0, 'bmi'] = np.nan
        df_processed = preprocess_data(sample_dataframe, logger)
        assert df_processed['bmi'].isna().sum() == 0

    def test_preprocess_data_removes_other_gender(self):
        """Test that 'Other' gender is removed."""
        df = pd.DataFrame({
            'gender': ['Male', 'Female', 'Other', 'Male'],
            'age': [30, 40, 50, 35],
            'hypertension': [0, 1, 0, 1],
            'heart_disease': [0, 0, 1, 0],
            'ever_married': ['Yes', 'No', 'Yes', 'No'],
            'work_type': ['Private', 'Private', 'Private', 'Private'],
            'Residence_type': ['Urban', 'Rural', 'Urban', 'Rural'],
            'avg_glucose_level': [100.0, 120.0, 95.0, 110.0],
            'bmi': [25.0, 28.5, 30.0, 27.0],
            'smoking_status': ['never smoked', 'never smoked', 'never smoked', 'never smoked'],
            'stroke': [0, 0, 1, 0]
        })
        df_processed = preprocess_data(df, logger)
        assert 'Other' not in df_processed['gender'].values
        assert len(df_processed) == 3


# ═══════════════════════════════════════════════════════════
# TESTS - CONSTANTS & STYLING
# ═══════════════════════════════════════════════════════════

class TestConstants:
    """Tests for constants and styling."""

    def test_colors_defined(self):
        """Test that color palette is defined."""
        assert 'primary' in COLORS
        assert 'secondary' in COLORS
        assert 'stroke' in COLORS
        assert 'no_stroke' in COLORS

    def test_plotly_template_valid(self):
        """Test that Plotly template is valid."""
        assert PLOTLY_TEMPLATE in ['plotly', 'plotly_white', 'plotly_dark', 'ggplot2', 'seaborn']


class DummyModel:
    """Simple model stub for API endpoint tests."""

    def __init__(self, probability: float):
        self.probability = probability

    def predict(self, _: pd.DataFrame) -> np.ndarray:
        return np.array([int(self.probability >= 0.5)])

    def predict_proba(self, _: pd.DataFrame) -> np.ndarray:
        return np.array([[1 - self.probability, self.probability]])


class TestApiEndpoints:
    """Basic smoke tests for the FastAPI surface."""

    def test_health_endpoint_degraded_when_model_missing(self, monkeypatch):
        """Health endpoint should report degraded when no model is loaded."""
        client = TestClient(app)
        monkeypatch.setattr('api.model', None)
        response = client.get('/health')

        assert response.status_code == 200
        payload = response.json()
        assert payload['status'] == 'degraded'
        assert payload['model_loaded'] is False

    def test_predict_endpoint_returns_risk_assessment(self, sample_patient_input, monkeypatch):
        """Prediction endpoint should return a consistent response payload."""
        client = TestClient(app)
        monkeypatch.setattr('api.model', DummyModel(probability=0.72))
        response = client.post('/predict', json=sample_patient_input)

        assert response.status_code == 200
        payload = response.json()
        assert payload['prediction'] == 1
        assert payload['risk_level'] == 'HIGH'
        assert payload['probability'] == pytest.approx(0.72)
        assert payload['confidence'] == pytest.approx(0.72)
        assert len(payload['recommendations']) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])