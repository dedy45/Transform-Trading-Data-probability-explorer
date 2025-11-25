"""
Data Cleaning Module

This module provides data cleaning utilities for the ML Prediction Engine.
It handles missing values, outliers, and data quality issues.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Literal
import logging

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Handles missing values and outliers in data.
    """
    
    def __init__(self, 
                 handle_missing: Literal['median', 'mean', 'drop', 'forward_fill'] = 'median',
                 clip_r_multiple: bool = True,
                 r_min: float = -5.0,
                 r_max: float = 10.0):
        """
        Initialize data cleaner.
        
        Parameters
        ----------
        handle_missing : {'median', 'mean', 'drop', 'forward_fill'}
            Strategy for handling missing values (default: 'median')
        clip_r_multiple : bool
            Whether to clip R_multiple values (default: True)
        r_min : float
            Minimum R_multiple value (default: -5.0)
        r_max : float
            Maximum R_multiple value (default: 10.0)
        """
        self.handle_missing = handle_missing
        self.clip_r_multiple = clip_r_multiple
        self.r_min = r_min
        self.r_max = r_max
        self.imputation_values = {}
        
    def fit(self, df: pd.DataFrame, feature_columns: List[str]) -> 'DataCleaner':
        """
        Fit the cleaner on training data to learn imputation values.
        
        Parameters
        ----------
        df : pd.DataFrame
            Training data
        feature_columns : List[str]
            List of feature columns to compute imputation values for
            
        Returns
        -------
        DataCleaner
            Self for method chaining
        """
        if self.handle_missing == 'median':
            self.imputation_values = {col: df[col].median() for col in feature_columns if col in df.columns}
        elif self.handle_missing == 'mean':
            self.imputation_values = {col: df[col].mean() for col in feature_columns if col in df.columns}
        else:
            self.imputation_values = {}
        
        logger.info(f"Fitted cleaner with {len(self.imputation_values)} imputation values")
        return self
    
    def clean_data(self, df: pd.DataFrame, 
                   feature_columns: List[str],
                   target_column: str = 'R_multiple',
                   inplace: bool = False) -> pd.DataFrame:
        """
        Clean data by handling missing values and outliers.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data to clean
        feature_columns : List[str]
            List of feature columns to clean
        target_column : str, optional
            Name of target column (default: 'R_multiple')
        inplace : bool, optional
            Whether to modify DataFrame in place (default: False)
            
        Returns
        -------
        pd.DataFrame
            Cleaned data
        """
        if not inplace:
            df = df.copy()
        
        # Handle missing values in features
        df = self._handle_missing_values(df, feature_columns)
        
        # Clip R_multiple if requested
        if self.clip_r_multiple and target_column in df.columns:
            df = self._clip_r_multiple(df, target_column)
        
        logger.info(f"Cleaned {len(df)} rows")
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
        """
        Handle missing values according to strategy.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data with potential missing values
        feature_columns : List[str]
            List of feature columns to handle
            
        Returns
        -------
        pd.DataFrame
            Data with missing values handled
        """
        initial_missing = df[feature_columns].isna().sum().sum()
        
        if self.handle_missing == 'drop':
            df = df.dropna(subset=feature_columns)
            logger.info(f"Dropped rows with missing values. Remaining: {len(df)} rows")
            
        elif self.handle_missing == 'forward_fill':
            df[feature_columns] = df[feature_columns].fillna(method='ffill')
            # Handle any remaining NaN at the beginning
            df[feature_columns] = df[feature_columns].fillna(method='bfill')
            logger.info(f"Forward filled missing values")
            
        elif self.handle_missing in ['median', 'mean']:
            # Use fitted imputation values if available, otherwise compute on the fly
            for col in feature_columns:
                if col in df.columns:
                    if col in self.imputation_values:
                        fill_value = self.imputation_values[col]
                    else:
                        # Compute on the fly if not fitted
                        if self.handle_missing == 'median':
                            fill_value = df[col].median()
                        else:
                            fill_value = df[col].mean()
                    
                    missing_count = df[col].isna().sum()
                    if missing_count > 0:
                        df[col] = df[col].fillna(fill_value)
                        logger.debug(f"Imputed {missing_count} missing values in '{col}' with {fill_value:.4f}")
        
        final_missing = df[feature_columns].isna().sum().sum()
        logger.info(f"Missing value handling: {initial_missing} -> {final_missing} NaN values")
        
        return df
    
    def _clip_r_multiple(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """
        Clip R_multiple values to specified range.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data with R_multiple column
        target_column : str
            Name of R_multiple column
            
        Returns
        -------
        pd.DataFrame
            Data with clipped R_multiple values
        """
        if target_column not in df.columns:
            logger.warning(f"Target column '{target_column}' not found, skipping clipping")
            return df
        
        original_values = df[target_column].copy()
        df[target_column] = df[target_column].clip(lower=self.r_min, upper=self.r_max)
        
        n_clipped = (original_values != df[target_column]).sum()
        if n_clipped > 0:
            pct_clipped = (n_clipped / len(df)) * 100
            logger.info(f"Clipped {n_clipped} ({pct_clipped:.2f}%) R_multiple values to [{self.r_min}, {self.r_max}]")
        
        return df
    
    def get_data_quality_report(self, df: pd.DataFrame, feature_columns: List[str]) -> Dict[str, Any]:
        """
        Generate data quality report.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data to analyze
        feature_columns : List[str]
            List of feature columns to analyze
            
        Returns
        -------
        Dict[str, Any]
            Data quality report
        """
        report = {
            'total_rows': len(df),
            'total_features': len(feature_columns),
            'missing_values': {},
            'outliers': {},
            'data_types': {}
        }
        
        for col in feature_columns:
            if col in df.columns:
                # Missing values
                missing_count = df[col].isna().sum()
                missing_pct = (missing_count / len(df)) * 100
                report['missing_values'][col] = {
                    'count': int(missing_count),
                    'percentage': round(missing_pct, 2)
                }
                
                # Data type
                report['data_types'][col] = str(df[col].dtype)
                
                # Outliers (using IQR method)
                if pd.api.types.is_numeric_dtype(df[col]):
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                    outlier_pct = (outlier_count / len(df)) * 100
                    report['outliers'][col] = {
                        'count': int(outlier_count),
                        'percentage': round(outlier_pct, 2),
                        'bounds': (round(lower_bound, 4), round(upper_bound, 4))
                    }
        
        return report


def clean_data(df: pd.DataFrame,
               feature_columns: List[str],
               target_column: str = 'R_multiple',
               handle_missing: Literal['median', 'mean', 'drop', 'forward_fill'] = 'median',
               clip_r_multiple: bool = True,
               r_min: float = -5.0,
               r_max: float = 10.0,
               inplace: bool = False) -> pd.DataFrame:
    """
    Convenience function to clean data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data to clean
    feature_columns : List[str]
        List of feature columns to clean
    target_column : str, optional
        Name of target column (default: 'R_multiple')
    handle_missing : {'median', 'mean', 'drop', 'forward_fill'}
        Strategy for handling missing values (default: 'median')
    clip_r_multiple : bool
        Whether to clip R_multiple values (default: True)
    r_min : float
        Minimum R_multiple value (default: -5.0)
    r_max : float
        Maximum R_multiple value (default: 10.0)
    inplace : bool, optional
        Whether to modify DataFrame in place (default: False)
        
    Returns
    -------
    pd.DataFrame
        Cleaned data
        
    Examples
    --------
    >>> cleaned_df = clean_data(
    ...     df,
    ...     feature_columns=['feature1', 'feature2'],
    ...     handle_missing='median',
    ...     clip_r_multiple=True
    ... )
    """
    cleaner = DataCleaner(
        handle_missing=handle_missing,
        clip_r_multiple=clip_r_multiple,
        r_min=r_min,
        r_max=r_max
    )
    
    # Fit on the data (for median/mean imputation)
    if handle_missing in ['median', 'mean']:
        cleaner.fit(df, feature_columns)
    
    return cleaner.clean_data(df, feature_columns, target_column, inplace)
