"""
ML Engine module for ML Prediction Engine.

This module contains the core ML components:
- Feature Selector for loading and validating features from Auto Feature Selection
- LightGBM Classifier for binary win/loss prediction
- Isotonic Calibration for probability calibration
- Quantile Regression for R_multiple distribution prediction
- Conformal Prediction for interval prediction with coverage guarantee
- Prediction Pipeline for orchestrating all components
"""

__version__ = '1.0.0'

# Import feature selector
from .feature_selector import (
    FeatureSelector,
    load_features_from_config,
    save_features_to_config
)

# Import LightGBM classifier
from .lgbm_classifier import (
    LGBMClassifierWrapper,
    train_classifier
)

# Import Isotonic Calibrator
from .calibration_isotonic import (
    IsotonicCalibrator,
    calibrate_probabilities
)

# Import Quantile Regressor
from .lgbm_quantile_regressor import (
    QuantileRegressorEnsemble,
    train_quantile_ensemble
)

# Import Conformal Engine
from .conformal_engine import (
    ConformalEngine,
    fit_conformal_engine
)

# Import Prediction Pipeline
from .pipeline_prediction import (
    PredictionPipeline,
    create_pipeline
)

# Import Model Trainer
from .model_trainer import (
    ModelTrainer,
    TrainingProgress,
    train_models
)

__all__ = [
    'FeatureSelector',
    'load_features_from_config',
    'save_features_to_config',
    'LGBMClassifierWrapper',
    'train_classifier',
    'IsotonicCalibrator',
    'calibrate_probabilities',
    'QuantileRegressorEnsemble',
    'train_quantile_ensemble',
    'ConformalEngine',
    'fit_conformal_engine',
    'PredictionPipeline',
    'create_pipeline',
    'ModelTrainer',
    'TrainingProgress',
    'train_models',
]
