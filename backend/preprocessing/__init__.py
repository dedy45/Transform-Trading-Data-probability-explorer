"""
Preprocessing module for ML Prediction Engine.

This module contains data preprocessing utilities:
- Schema validation for data integrity
- Data cleaning (missing values, outliers)
- Time-series aware train/calib/test splitting
"""

from .schema_validation import (
    SchemaValidator,
    validate_schema,
    validate_feature_completeness
)

from .cleaning import (
    DataCleaner,
    clean_data
)

from .split_timeseries import (
    TimeSeriesSplitter,
    split_timeseries,
    validate_split_ratios
)

__version__ = '1.0.0'

__all__ = [
    # Schema validation
    'SchemaValidator',
    'validate_schema',
    'validate_feature_completeness',
    
    # Data cleaning
    'DataCleaner',
    'clean_data',
    
    # Time-series splitting
    'TimeSeriesSplitter',
    'split_timeseries',
    'validate_split_ratios',
]
