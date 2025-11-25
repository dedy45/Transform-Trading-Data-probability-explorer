"""
Schema Validation Module

This module provides data schema validation utilities for the ML Prediction Engine.
It validates column presence, data types, value ranges, and data integrity.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class SchemaValidator:
    """
    Validates data schema and ensures data integrity.
    """
    
    def __init__(self, required_features: List[str], target_column: str = 'R_multiple'):
        """
        Initialize schema validator.
        
        Parameters
        ----------
        required_features : List[str]
            List of required feature column names
        target_column : str, optional
            Name of target column (default: 'R_multiple')
        """
        self.required_features = required_features
        self.target_column = target_column
        
    def validate_schema(self, df: pd.DataFrame, check_target: bool = True) -> Tuple[bool, List[str]]:
        """
        Validate data schema.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data to validate
        check_target : bool, optional
            Whether to check for target column (default: True)
            
        Returns
        -------
        Tuple[bool, List[str]]
            (is_valid, list_of_errors)
        """
        errors = []
        
        # Check if DataFrame is empty
        if df.empty:
            errors.append("DataFrame is empty")
            return False, errors
        
        # Check for required features
        missing_features = [f for f in self.required_features if f not in df.columns]
        if missing_features:
            errors.append(f"Missing required features: {missing_features}")
        
        # Check for target column if needed
        if check_target and self.target_column not in df.columns:
            errors.append(f"Missing target column: {self.target_column}")
        
        # Check data types for numeric features
        for feature in self.required_features:
            if feature in df.columns:
                if not pd.api.types.is_numeric_dtype(df[feature]):
                    errors.append(f"Feature '{feature}' is not numeric (dtype: {df[feature].dtype})")
        
        # Check target column data type
        if check_target and self.target_column in df.columns:
            if not pd.api.types.is_numeric_dtype(df[self.target_column]):
                errors.append(f"Target column '{self.target_column}' is not numeric (dtype: {df[self.target_column].dtype})")
        
        is_valid = len(errors) == 0
        
        if is_valid:
            logger.info(f"Schema validation passed for {len(df)} rows")
        else:
            logger.warning(f"Schema validation failed with {len(errors)} errors")
            
        return is_valid, errors
    
    def validate_feature_completeness(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate that all selected features are present and report missing value statistics.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data to validate
            
        Returns
        -------
        Tuple[bool, Dict[str, Any]]
            (all_features_present, missing_value_report)
        """
        report = {
            'total_rows': len(df),
            'missing_features': [],
            'missing_value_stats': {}
        }
        
        # Check for missing features
        for feature in self.required_features:
            if feature not in df.columns:
                report['missing_features'].append(feature)
            else:
                # Calculate missing value percentage
                missing_count = df[feature].isna().sum()
                missing_pct = (missing_count / len(df)) * 100
                report['missing_value_stats'][feature] = {
                    'missing_count': int(missing_count),
                    'missing_percentage': round(missing_pct, 2)
                }
        
        all_present = len(report['missing_features']) == 0
        
        if all_present:
            logger.info(f"All {len(self.required_features)} required features are present")
        else:
            logger.error(f"Missing features: {report['missing_features']}")
            
        return all_present, report
    
    def validate_value_ranges(self, df: pd.DataFrame, 
                             feature_ranges: Optional[Dict[str, Tuple[float, float]]] = None) -> Tuple[bool, List[str]]:
        """
        Validate that feature values are within expected ranges.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data to validate
        feature_ranges : Dict[str, Tuple[float, float]], optional
            Dictionary mapping feature names to (min, max) tuples
            
        Returns
        -------
        Tuple[bool, List[str]]
            (all_in_range, list_of_warnings)
        """
        warnings = []
        
        if feature_ranges is None:
            return True, warnings
        
        for feature, (min_val, max_val) in feature_ranges.items():
            if feature in df.columns:
                out_of_range = df[(df[feature] < min_val) | (df[feature] > max_val)]
                if len(out_of_range) > 0:
                    pct = (len(out_of_range) / len(df)) * 100
                    warnings.append(
                        f"Feature '{feature}': {len(out_of_range)} values ({pct:.2f}%) "
                        f"outside range [{min_val}, {max_val}]"
                    )
        
        all_in_range = len(warnings) == 0
        
        if not all_in_range:
            logger.warning(f"Value range validation found {len(warnings)} issues")
            
        return all_in_range, warnings


def validate_schema(df: pd.DataFrame, 
                    required_features: List[str],
                    target_column: str = 'R_multiple',
                    check_target: bool = True) -> Tuple[bool, List[str]]:
    """
    Convenience function to validate data schema.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data to validate
    required_features : List[str]
        List of required feature column names
    target_column : str, optional
        Name of target column (default: 'R_multiple')
    check_target : bool, optional
        Whether to check for target column (default: True)
        
    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, list_of_errors)
        
    Examples
    --------
    >>> is_valid, errors = validate_schema(df, ['feature1', 'feature2'])
    >>> if not is_valid:
    ...     print(f"Validation errors: {errors}")
    """
    validator = SchemaValidator(required_features, target_column)
    return validator.validate_schema(df, check_target)


def validate_feature_completeness(df: pd.DataFrame, 
                                  required_features: List[str]) -> Tuple[bool, Dict[str, Any]]:
    """
    Convenience function to validate feature completeness.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data to validate
    required_features : List[str]
        List of required feature column names
        
    Returns
    -------
    Tuple[bool, Dict[str, Any]]
        (all_features_present, missing_value_report)
        
    Examples
    --------
    >>> all_present, report = validate_feature_completeness(df, ['feature1', 'feature2'])
    >>> print(f"Missing values: {report['missing_value_stats']}")
    """
    validator = SchemaValidator(required_features)
    return validator.validate_feature_completeness(df)
