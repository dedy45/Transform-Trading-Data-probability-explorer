"""
Time-Series Data Splitting Module

This module provides time-series aware data splitting utilities for the ML Prediction Engine.
It ensures no data leakage by maintaining temporal order.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Literal
import logging

logger = logging.getLogger(__name__)


class TimeSeriesSplitter:
    """
    Splits data into train/calibration/test sets with time-series awareness.
    """
    
    def __init__(self,
                 train_ratio: float = 0.60,
                 calib_ratio: float = 0.20,
                 test_ratio: float = 0.20,
                 time_column: Optional[str] = None,
                 random_state: Optional[int] = None):
        """
        Initialize time-series splitter.
        
        Parameters
        ----------
        train_ratio : float
            Proportion of data for training (default: 0.60)
        calib_ratio : float
            Proportion of data for calibration (default: 0.20)
        test_ratio : float
            Proportion of data for testing (default: 0.20)
        time_column : str, optional
            Name of time column for sorting (if None, assumes data is already sorted)
        random_state : int, optional
            Random state for reproducibility (used only for random split)
        """
        # Validate ratios
        total_ratio = train_ratio + calib_ratio + test_ratio
        if not np.isclose(total_ratio, 1.0):
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
        
        if train_ratio <= 0 or calib_ratio <= 0 or test_ratio <= 0:
            raise ValueError("All ratios must be positive")
        
        self.train_ratio = train_ratio
        self.calib_ratio = calib_ratio
        self.test_ratio = test_ratio
        self.time_column = time_column
        self.random_state = random_state
        
    def split(self, df: pd.DataFrame, 
              method: Literal['time_series', 'random'] = 'time_series') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/calibration/test sets.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data to split
        method : {'time_series', 'random'}
            Split method (default: 'time_series')
            
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            (train_df, calib_df, test_df)
        """
        if df.empty:
            raise ValueError("Cannot split empty DataFrame")
        
        if method == 'time_series':
            return self._split_time_series(df)
        elif method == 'random':
            return self._split_random(df)
        else:
            raise ValueError(f"Unknown split method: {method}")
    
    def _split_time_series(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data maintaining temporal order (no data leakage).
        
        Parameters
        ----------
        df : pd.DataFrame
            Data to split
            
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            (train_df, calib_df, test_df)
        """
        # Sort by time column if specified
        if self.time_column is not None:
            if self.time_column not in df.columns:
                raise ValueError(f"Time column '{self.time_column}' not found in DataFrame")
            df = df.sort_values(by=self.time_column).reset_index(drop=True)
        
        n = len(df)
        
        # Calculate split indices
        train_end = int(n * self.train_ratio)
        calib_end = train_end + int(n * self.calib_ratio)
        
        # Split data
        train_df = df.iloc[:train_end].copy()
        calib_df = df.iloc[train_end:calib_end].copy()
        test_df = df.iloc[calib_end:].copy()
        
        logger.info(f"Time-series split: train={len(train_df)}, calib={len(calib_df)}, test={len(test_df)}")
        
        # Validate split
        self._validate_split(train_df, calib_df, test_df, n)
        
        return train_df, calib_df, test_df
    
    def _split_random(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data randomly (use only if temporal order is not important).
        
        Parameters
        ----------
        df : pd.DataFrame
            Data to split
            
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            (train_df, calib_df, test_df)
        """
        # Shuffle data
        df_shuffled = df.sample(frac=1.0, random_state=self.random_state).reset_index(drop=True)
        
        n = len(df_shuffled)
        
        # Calculate split indices
        train_end = int(n * self.train_ratio)
        calib_end = train_end + int(n * self.calib_ratio)
        
        # Split data
        train_df = df_shuffled.iloc[:train_end].copy()
        calib_df = df_shuffled.iloc[train_end:calib_end].copy()
        test_df = df_shuffled.iloc[calib_end:].copy()
        
        logger.info(f"Random split: train={len(train_df)}, calib={len(calib_df)}, test={len(test_df)}")
        
        # Validate split
        self._validate_split(train_df, calib_df, test_df, n)
        
        return train_df, calib_df, test_df
    
    def _validate_split(self, train_df: pd.DataFrame, calib_df: pd.DataFrame, 
                       test_df: pd.DataFrame, original_n: int) -> None:
        """
        Validate that split was performed correctly.
        
        Parameters
        ----------
        train_df : pd.DataFrame
            Training set
        calib_df : pd.DataFrame
            Calibration set
        test_df : pd.DataFrame
            Test set
        original_n : int
            Original number of rows
        """
        # Check that all rows are accounted for
        total_rows = len(train_df) + len(calib_df) + len(test_df)
        if total_rows != original_n:
            logger.warning(f"Split validation: expected {original_n} rows, got {total_rows}")
        
        # Check that sets are non-empty
        if len(train_df) == 0:
            raise ValueError("Training set is empty")
        if len(calib_df) == 0:
            raise ValueError("Calibration set is empty")
        if len(test_df) == 0:
            raise ValueError("Test set is empty")
        
        # Check actual ratios
        actual_train_ratio = len(train_df) / original_n
        actual_calib_ratio = len(calib_df) / original_n
        actual_test_ratio = len(test_df) / original_n
        
        logger.debug(f"Actual ratios: train={actual_train_ratio:.3f}, "
                    f"calib={actual_calib_ratio:.3f}, test={actual_test_ratio:.3f}")
    
    def get_split_info(self, df: pd.DataFrame) -> dict:
        """
        Get information about how data would be split without actually splitting.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data to analyze
            
        Returns
        -------
        dict
            Split information
        """
        n = len(df)
        train_end = int(n * self.train_ratio)
        calib_end = train_end + int(n * self.calib_ratio)
        
        info = {
            'total_rows': n,
            'train_rows': train_end,
            'calib_rows': calib_end - train_end,
            'test_rows': n - calib_end,
            'train_ratio': self.train_ratio,
            'calib_ratio': self.calib_ratio,
            'test_ratio': self.test_ratio,
            'time_column': self.time_column
        }
        
        return info


def split_timeseries(df: pd.DataFrame,
                     train_ratio: float = 0.60,
                     calib_ratio: float = 0.20,
                     test_ratio: float = 0.20,
                     time_column: Optional[str] = None,
                     method: Literal['time_series', 'random'] = 'time_series',
                     random_state: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to split data into train/calibration/test sets.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data to split
    train_ratio : float
        Proportion of data for training (default: 0.60)
    calib_ratio : float
        Proportion of data for calibration (default: 0.20)
    test_ratio : float
        Proportion of data for testing (default: 0.20)
    time_column : str, optional
        Name of time column for sorting (if None, assumes data is already sorted)
    method : {'time_series', 'random'}
        Split method (default: 'time_series')
    random_state : int, optional
        Random state for reproducibility (used only for random split)
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_df, calib_df, test_df)
        
    Examples
    --------
    >>> train_df, calib_df, test_df = split_timeseries(
    ...     df,
    ...     train_ratio=0.60,
    ...     calib_ratio=0.20,
    ...     test_ratio=0.20,
    ...     time_column='timestamp'
    ... )
    """
    splitter = TimeSeriesSplitter(
        train_ratio=train_ratio,
        calib_ratio=calib_ratio,
        test_ratio=test_ratio,
        time_column=time_column,
        random_state=random_state
    )
    
    return splitter.split(df, method=method)


def validate_split_ratios(train_ratio: float, calib_ratio: float, test_ratio: float) -> bool:
    """
    Validate that split ratios are valid.
    
    Parameters
    ----------
    train_ratio : float
        Training ratio
    calib_ratio : float
        Calibration ratio
    test_ratio : float
        Test ratio
        
    Returns
    -------
    bool
        True if ratios are valid
        
    Raises
    ------
    ValueError
        If ratios are invalid
    """
    total = train_ratio + calib_ratio + test_ratio
    
    if not np.isclose(total, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {total}")
    
    if train_ratio <= 0 or calib_ratio <= 0 or test_ratio <= 0:
        raise ValueError("All ratios must be positive")
    
    return True
