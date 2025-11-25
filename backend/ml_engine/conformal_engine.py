"""
Conformal Prediction for Interval Prediction with Coverage Guarantee

This module implements conformal prediction to provide prediction intervals
with statistical coverage guarantees. It adjusts quantile predictions to
ensure that the actual coverage matches the target coverage.

**Feature: ml-prediction-engine**
**Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5**
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Optional, Dict, Union, Tuple


class ConformalEngine:
    """
    Conformal prediction for interval with coverage guarantee.
    
    This class uses conformal prediction to adjust prediction intervals
    from quantile regression, ensuring that the actual coverage matches
    the target coverage (e.g., 90% of actual values fall within the interval).
    
    The method computes nonconformity scores on a calibration set and uses
    these to widen the prediction intervals appropriately.
    
    Attributes
    ----------
    coverage : float
        Target coverage probability (e.g., 0.9 for 90%)
    nonconformity_quantile : float
        The quantile of nonconformity scores used for adjustment
    is_fitted : bool
        Whether the engine has been fitted
    metadata : dict
        Metadata from the fitting process
    
    Examples
    --------
    >>> engine = ConformalEngine(coverage=0.9)
    >>> metadata = engine.fit(y_pred_p10, y_pred_p90, y_true)
    >>> y_p10_conf, y_p90_conf = engine.adjust_intervals(y_pred_p10_test, y_pred_p90_test)
    >>> actual_coverage = engine.validate_coverage(y_p10_conf, y_p90_conf, y_true_test)
    >>> engine.save('models/conformal_meta.json')
    """
    
    def __init__(self, coverage: float = 0.9):
        """
        Initialize conformal engine.
        
        Parameters
        ----------
        coverage : float, default=0.9
            Target coverage probability (0-1). For example, 0.9 means
            90% of actual values should fall within the prediction interval.
        
        Raises
        ------
        ValueError
            If coverage is not in range (0, 1)
        """
        if not 0 < coverage < 1:
            raise ValueError(
                f"Coverage must be in range (0, 1), got {coverage}"
            )
        
        self.coverage = coverage
        self.nonconformity_quantile = None
        self.is_fitted = False
        self.metadata = {}
        
        # Store calibration data for validation
        self._nonconformity_scores = None
    
    def fit(
        self,
        y_pred_p10: np.ndarray,
        y_pred_p90: np.ndarray,
        y_true: np.ndarray
    ) -> Dict:
        """
        Fit conformal engine on calibration set.
        
        Computes nonconformity scores which measure how far the actual
        values are from the predicted intervals. The nonconformity score
        for each sample is:
        
        r_i = max(y_pred_p10_i - y_true_i, y_true_i - y_pred_p90_i, 0)
        
        This measures how much the interval needs to be widened to cover
        the actual value. A score of 0 means the value is already covered.
        
        Parameters
        ----------
        y_pred_p10 : np.ndarray
            Predicted P10 (lower bound) from quantile model
        y_pred_p90 : np.ndarray
            Predicted P90 (upper bound) from quantile model
        y_true : np.ndarray
            True R_multiple values
        
        Returns
        -------
        dict
            Metadata including:
            - nonconformity_quantile: The quantile value used for adjustment
            - coverage: Target coverage
            - n_samples: Number of calibration samples
            - mean_nonconformity: Mean nonconformity score
            - median_nonconformity: Median nonconformity score
        
        Raises
        ------
        ValueError
            If arrays are empty or have different lengths
        """
        # Validate inputs
        y_pred_p10 = np.asarray(y_pred_p10).flatten()
        y_pred_p90 = np.asarray(y_pred_p90).flatten()
        y_true = np.asarray(y_true).flatten()
        
        if len(y_pred_p10) == 0:
            raise ValueError("y_pred_p10 cannot be empty")
        
        if not (len(y_pred_p10) == len(y_pred_p90) == len(y_true)):
            raise ValueError(
                f"All arrays must have same length: "
                f"p10={len(y_pred_p10)}, p90={len(y_pred_p90)}, "
                f"true={len(y_true)}"
            )
        
        # Compute nonconformity scores
        # r_i = max(p10 - y_true, y_true - p90, 0)
        # This measures how much we need to widen the interval
        lower_violation = y_pred_p10 - y_true  # Positive if y_true < p10
        upper_violation = y_true - y_pred_p90  # Positive if y_true > p90
        
        nonconformity_scores = np.maximum(
            np.maximum(lower_violation, upper_violation),
            0.0
        )
        
        # Store nonconformity scores
        self._nonconformity_scores = nonconformity_scores
        
        # Compute the quantile of nonconformity scores
        # We use (n+1) * coverage / n to get the appropriate quantile
        # This ensures coverage guarantee even for finite samples
        n = len(nonconformity_scores)
        quantile_level = np.ceil((n + 1) * self.coverage) / n
        quantile_level = min(quantile_level, 1.0)  # Cap at 1.0
        
        self.nonconformity_quantile = float(
            np.quantile(nonconformity_scores, quantile_level)
        )
        
        # Store metadata
        self.metadata = {
            'nonconformity_quantile': self.nonconformity_quantile,
            'coverage': self.coverage,
            'n_samples': int(n),
            'mean_nonconformity': float(np.mean(nonconformity_scores)),
            'median_nonconformity': float(np.median(nonconformity_scores)),
            'max_nonconformity': float(np.max(nonconformity_scores)),
            'quantile_level': float(quantile_level)
        }
        
        self.is_fitted = True
        
        return self.metadata
    
    def adjust_intervals(
        self,
        y_pred_p10: np.ndarray,
        y_pred_p90: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Adjust intervals with margin from nonconformity quantile.
        
        The intervals are widened by subtracting the nonconformity quantile
        from the lower bound and adding it to the upper bound:
        
        y_pred_p10_conf = y_pred_p10 - nonconformity_quantile
        y_pred_p90_conf = y_pred_p90 + nonconformity_quantile
        
        This ensures that the adjusted intervals have the target coverage.
        
        Parameters
        ----------
        y_pred_p10 : np.ndarray
            Raw P10 predictions from quantile model
        y_pred_p90 : np.ndarray
            Raw P90 predictions from quantile model
        
        Returns
        -------
        tuple
            (y_pred_p10_conf, y_pred_p90_conf) - Adjusted prediction intervals
        
        Raises
        ------
        ValueError
            If engine is not fitted
            If arrays are empty or have different lengths
        """
        if not self.is_fitted:
            raise ValueError(
                "Engine must be fitted before adjusting intervals. Call fit() first."
            )
        
        # Validate inputs
        y_pred_p10 = np.asarray(y_pred_p10).flatten()
        y_pred_p90 = np.asarray(y_pred_p90).flatten()
        
        if len(y_pred_p10) == 0:
            raise ValueError("y_pred_p10 cannot be empty")
        
        if len(y_pred_p10) != len(y_pred_p90):
            raise ValueError(
                f"Arrays must have same length: "
                f"p10={len(y_pred_p10)}, p90={len(y_pred_p90)}"
            )
        
        # Adjust intervals by adding/subtracting nonconformity quantile
        y_pred_p10_conf = y_pred_p10 - self.nonconformity_quantile
        y_pred_p90_conf = y_pred_p90 + self.nonconformity_quantile
        
        return y_pred_p10_conf, y_pred_p90_conf
    
    def validate_coverage(
        self,
        y_pred_p10_conf: np.ndarray,
        y_pred_p90_conf: np.ndarray,
        y_true: np.ndarray
    ) -> float:
        """
        Validate actual coverage on test set.
        
        Computes the percentage of actual values that fall within the
        conformal prediction intervals.
        
        Parameters
        ----------
        y_pred_p10_conf : np.ndarray
            Conformal adjusted P10 predictions
        y_pred_p90_conf : np.ndarray
            Conformal adjusted P90 predictions
        y_true : np.ndarray
            True R_multiple values
        
        Returns
        -------
        float
            Actual coverage percentage (0-1)
        
        Raises
        ------
        ValueError
            If arrays are empty or have different lengths
        """
        # Validate inputs
        y_pred_p10_conf = np.asarray(y_pred_p10_conf).flatten()
        y_pred_p90_conf = np.asarray(y_pred_p90_conf).flatten()
        y_true = np.asarray(y_true).flatten()
        
        if len(y_pred_p10_conf) == 0:
            raise ValueError("y_pred_p10_conf cannot be empty")
        
        if not (len(y_pred_p10_conf) == len(y_pred_p90_conf) == len(y_true)):
            raise ValueError(
                f"All arrays must have same length: "
                f"p10={len(y_pred_p10_conf)}, p90={len(y_pred_p90_conf)}, "
                f"true={len(y_true)}"
            )
        
        # Check if actual values fall within intervals
        covered = (y_true >= y_pred_p10_conf) & (y_true <= y_pred_p90_conf)
        
        actual_coverage = float(np.mean(covered))
        
        return actual_coverage
    
    def get_interval_widths(
        self,
        y_pred_p10_conf: np.ndarray,
        y_pred_p90_conf: np.ndarray
    ) -> np.ndarray:
        """
        Calculate interval widths.
        
        Parameters
        ----------
        y_pred_p10_conf : np.ndarray
            Conformal adjusted P10 predictions
        y_pred_p90_conf : np.ndarray
            Conformal adjusted P90 predictions
        
        Returns
        -------
        np.ndarray
            Interval widths (P90_conf - P10_conf)
        
        Raises
        ------
        ValueError
            If arrays have different lengths
        """
        y_pred_p10_conf = np.asarray(y_pred_p10_conf).flatten()
        y_pred_p90_conf = np.asarray(y_pred_p90_conf).flatten()
        
        if len(y_pred_p10_conf) != len(y_pred_p90_conf):
            raise ValueError(
                f"Arrays must have same length: "
                f"p10={len(y_pred_p10_conf)}, p90={len(y_pred_p90_conf)}"
            )
        
        return y_pred_p90_conf - y_pred_p10_conf
    
    def save(self, path: Union[str, Path]):
        """
        Save metadata to JSON file.
        
        Parameters
        ----------
        path : str or Path
            Path to save the metadata
        
        Raises
        ------
        ValueError
            If engine is not fitted
        """
        if not self.is_fitted:
            raise ValueError(
                "Cannot save unfitted engine. Call fit() first."
            )
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for JSON serialization
        save_data = {
            'coverage': self.coverage,
            'nonconformity_quantile': self.nonconformity_quantile,
            'is_fitted': self.is_fitted,
            'metadata': self.metadata
        }
        
        # Save to JSON
        with open(path, 'w') as f:
            json.dump(save_data, f, indent=2)
    
    def load(self, path: Union[str, Path]):
        """
        Load metadata from JSON file.
        
        Parameters
        ----------
        path : str or Path
            Path to load the metadata from
        
        Raises
        ------
        FileNotFoundError
            If metadata file doesn't exist
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Metadata file not found: {path}")
        
        # Load from JSON
        with open(path, 'r') as f:
            load_data = json.load(f)
        
        self.coverage = load_data['coverage']
        self.nonconformity_quantile = load_data['nonconformity_quantile']
        self.is_fitted = load_data['is_fitted']
        self.metadata = load_data['metadata']
    
    def get_engine_info(self) -> Dict:
        """
        Get information about the conformal engine.
        
        Returns
        -------
        dict
            Engine information including coverage and metadata
        """
        info = {
            'is_fitted': self.is_fitted,
            'coverage': self.coverage,
            'nonconformity_quantile': self.nonconformity_quantile
        }
        
        if self.is_fitted:
            info['metadata'] = self.metadata
        
        return info


def fit_conformal_engine(
    y_pred_p10: np.ndarray,
    y_pred_p90: np.ndarray,
    y_true: np.ndarray,
    coverage: float = 0.9,
    save_path: Optional[Union[str, Path]] = None
) -> Tuple[ConformalEngine, Dict]:
    """
    Convenience function to fit a conformal engine.
    
    Parameters
    ----------
    y_pred_p10 : np.ndarray
        Predicted P10 from quantile model
    y_pred_p90 : np.ndarray
        Predicted P90 from quantile model
    y_true : np.ndarray
        True R_multiple values
    coverage : float, default=0.9
        Target coverage
    save_path : str or Path, optional
        Path to save the engine metadata
    
    Returns
    -------
    tuple
        (engine, metadata) - Fitted engine and metadata
    """
    engine = ConformalEngine(coverage=coverage)
    metadata = engine.fit(y_pred_p10, y_pred_p90, y_true)
    
    if save_path is not None:
        engine.save(save_path)
    
    return engine, metadata
