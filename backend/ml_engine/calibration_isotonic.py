"""
Isotonic Calibration for Probability Predictions

This module implements isotonic regression for calibrating probability predictions
from binary classifiers. Isotonic calibration ensures that predicted probabilities
are monotonically related to actual outcomes.

**Feature: ml-prediction-engine**
**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**
"""

import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
from pathlib import Path
from typing import Optional, Dict, Union
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

# Import calibration utilities from existing module
import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.calibration import (
    compute_reliability_diagram,
    compute_brier_score,
    compute_ece
)


class IsotonicCalibrator:
    """
    Isotonic regression for probability calibration.
    
    This class uses isotonic regression to calibrate raw probabilities from a
    classifier, ensuring that higher predicted probabilities correspond to
    higher actual frequencies of positive outcomes.
    
    Attributes
    ----------
    calibrator : IsotonicRegression
        The underlying sklearn isotonic regression model
    is_fitted : bool
        Whether the calibrator has been fitted
    calibration_metrics : dict
        Metrics from the calibration process
    
    Examples
    --------
    >>> calibrator = IsotonicCalibrator()
    >>> metrics = calibrator.fit(raw_probs, y_true)
    >>> calibrated_probs = calibrator.transform(raw_probs_test)
    >>> calibrator.save('models/calibrator.pkl')
    """
    
    def __init__(self):
        """
        Initialize isotonic calibrator.
        
        The calibrator uses isotonic regression with 'clip' out_of_bounds
        handling to ensure output probabilities stay in [0, 1].
        """
        self.calibrator = IsotonicRegression(
            out_of_bounds='clip',
            y_min=0.0,
            y_max=1.0
        )
        self.is_fitted = False
        self.calibration_metrics = {}
        
        # Store calibration data for plotting
        self._raw_probs_calib = None
        self._y_true_calib = None
    
    def fit(
        self,
        raw_probs: np.ndarray,
        y_true: np.ndarray
    ) -> Dict[str, float]:
        """
        Fit calibrator on calibration set.
        
        The calibrator learns a monotonic mapping from raw probabilities to
        calibrated probabilities that better match observed frequencies.
        
        Parameters
        ----------
        raw_probs : np.ndarray
            Raw probabilities from classifier (0-1)
        y_true : np.ndarray
            True binary labels (0/1)
        
        Returns
        -------
        dict
            Calibration metrics including:
            - brier_before: Brier score before calibration
            - brier_after: Brier score after calibration
            - brier_improvement: Reduction in Brier score
            - ece_before: Expected Calibration Error before
            - ece_after: Expected Calibration Error after
            - ece_improvement: Reduction in ECE
            - n_samples: Number of calibration samples
        
        Raises
        ------
        ValueError
            If arrays are empty or have different lengths
            If raw_probs contains values outside [0, 1]
            If y_true contains values other than 0 or 1
        """
        # Validate inputs
        raw_probs = np.asarray(raw_probs).flatten()
        y_true = np.asarray(y_true).flatten()
        
        if len(raw_probs) == 0:
            raise ValueError("raw_probs cannot be empty")
        
        if len(raw_probs) != len(y_true):
            raise ValueError(
                f"raw_probs and y_true must have same length: "
                f"{len(raw_probs)} != {len(y_true)}"
            )
        
        if np.any(raw_probs < 0) or np.any(raw_probs > 1):
            raise ValueError("raw_probs must be in range [0, 1]")
        
        if not np.all(np.isin(y_true, [0, 1])):
            raise ValueError("y_true must contain only 0 or 1")
        
        # Store calibration data for later use
        self._raw_probs_calib = raw_probs.copy()
        self._y_true_calib = y_true.copy()
        
        # Calculate metrics before calibration
        brier_before = compute_brier_score(raw_probs, y_true)
        ece_before = compute_ece(raw_probs, y_true, n_bins=10)
        
        # Fit isotonic regression
        self.calibrator.fit(raw_probs, y_true)
        self.is_fitted = True
        
        # Calculate metrics after calibration
        calibrated_probs = self.calibrator.transform(raw_probs)
        brier_after = compute_brier_score(calibrated_probs, y_true)
        ece_after = compute_ece(calibrated_probs, y_true, n_bins=10)
        
        # Store metrics
        self.calibration_metrics = {
            'brier_before': float(brier_before),
            'brier_after': float(brier_after),
            'brier_improvement': float(brier_before - brier_after),
            'ece_before': float(ece_before),
            'ece_after': float(ece_after),
            'ece_improvement': float(ece_before - ece_after),
            'n_samples': len(raw_probs)
        }
        
        return self.calibration_metrics
    
    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        """
        Transform raw probabilities to calibrated probabilities.
        
        Parameters
        ----------
        raw_probs : np.ndarray
            Raw probabilities from classifier (0-1)
        
        Returns
        -------
        np.ndarray
            Calibrated probabilities (0-1)
        
        Raises
        ------
        ValueError
            If calibrator is not fitted
            If raw_probs contains values outside [0, 1]
        """
        if not self.is_fitted:
            raise ValueError(
                "Calibrator must be fitted before transform. Call fit() first."
            )
        
        # Validate input
        raw_probs = np.asarray(raw_probs).flatten()
        
        if len(raw_probs) == 0:
            raise ValueError("raw_probs cannot be empty")
        
        if np.any(raw_probs < 0) or np.any(raw_probs > 1):
            raise ValueError("raw_probs must be in range [0, 1]")
        
        # Transform probabilities
        calibrated_probs = self.calibrator.transform(raw_probs)
        
        # Ensure output is in [0, 1] (should be guaranteed by out_of_bounds='clip')
        calibrated_probs = np.clip(calibrated_probs, 0.0, 1.0)
        
        return calibrated_probs
    
    def fit_transform(
        self,
        raw_probs: np.ndarray,
        y_true: np.ndarray
    ) -> np.ndarray:
        """
        Fit calibrator and transform in one step.
        
        Parameters
        ----------
        raw_probs : np.ndarray
            Raw probabilities from classifier (0-1)
        y_true : np.ndarray
            True binary labels (0/1)
        
        Returns
        -------
        np.ndarray
            Calibrated probabilities (0-1)
        """
        self.fit(raw_probs, y_true)
        return self.transform(raw_probs)
    
    def plot_reliability_diagram(
        self,
        raw_probs: Optional[np.ndarray] = None,
        y_true: Optional[np.ndarray] = None,
        n_bins: int = 10,
        show_calibrated: bool = True
    ) -> go.Figure:
        """
        Create reliability diagram for visualizing calibration.
        
        A reliability diagram plots mean predicted probability vs observed
        frequency. Perfect calibration shows points on the diagonal line.
        
        Parameters
        ----------
        raw_probs : np.ndarray, optional
            Raw probabilities to plot. If None, uses calibration set.
        y_true : np.ndarray, optional
            True labels. If None, uses calibration set.
        n_bins : int, default=10
            Number of bins for reliability diagram
        show_calibrated : bool, default=True
            Whether to show calibrated probabilities alongside raw
        
        Returns
        -------
        plotly.graph_objects.Figure
            Reliability diagram figure
        
        Raises
        ------
        ValueError
            If calibrator is not fitted and no data provided
        """
        # Use calibration data if not provided
        if raw_probs is None or y_true is None:
            if not self.is_fitted or self._raw_probs_calib is None:
                raise ValueError(
                    "Must provide raw_probs and y_true, or fit calibrator first"
                )
            raw_probs = self._raw_probs_calib
            y_true = self._y_true_calib
        
        raw_probs = np.asarray(raw_probs).flatten()
        y_true = np.asarray(y_true).flatten()
        
        # Compute reliability diagram for raw probabilities
        reliability_raw = compute_reliability_diagram(
            raw_probs, y_true, n_bins=n_bins
        )
        
        # Create figure
        fig = go.Figure()
        
        # Add perfect calibration line (diagonal)
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Perfect Calibration',
            line=dict(color='gray', dash='dash', width=2),
            showlegend=True
        ))
        
        # Add raw probabilities
        mask_raw = ~np.isnan(reliability_raw['mean_predicted'])
        fig.add_trace(go.Scatter(
            x=reliability_raw['mean_predicted'][mask_raw],
            y=reliability_raw['observed_frequency'][mask_raw],
            mode='markers+lines',
            name='Raw Probabilities',
            marker=dict(size=10, color='red'),
            line=dict(color='red', width=2),
            showlegend=True
        ))
        
        # Add calibrated probabilities if requested
        if show_calibrated and self.is_fitted:
            calibrated_probs = self.transform(raw_probs)
            reliability_cal = compute_reliability_diagram(
                calibrated_probs, y_true, n_bins=n_bins
            )
            
            mask_cal = ~np.isnan(reliability_cal['mean_predicted'])
            fig.add_trace(go.Scatter(
                x=reliability_cal['mean_predicted'][mask_cal],
                y=reliability_cal['observed_frequency'][mask_cal],
                mode='markers+lines',
                name='Calibrated Probabilities',
                marker=dict(size=10, color='green'),
                line=dict(color='green', width=2),
                showlegend=True
            ))
        
        # Update layout
        fig.update_layout(
            title='Reliability Diagram - Calibration Assessment',
            xaxis_title='Mean Predicted Probability',
            yaxis_title='Observed Frequency',
            xaxis=dict(range=[0, 1], constrain='domain'),
            yaxis=dict(range=[0, 1], scaleanchor='x', scaleratio=1),
            width=600,
            height=600,
            showlegend=True,
            legend=dict(x=0.02, y=0.98),
            hovermode='closest'
        )
        
        # Add metrics as annotation
        if self.is_fitted and self.calibration_metrics:
            metrics_text = (
                f"Brier Score: {self.calibration_metrics['brier_before']:.4f} → "
                f"{self.calibration_metrics['brier_after']:.4f}<br>"
                f"ECE: {self.calibration_metrics['ece_before']:.4f} → "
                f"{self.calibration_metrics['ece_after']:.4f}"
            )
            
            fig.add_annotation(
                text=metrics_text,
                xref='paper', yref='paper',
                x=0.98, y=0.02,
                xanchor='right', yanchor='bottom',
                showarrow=False,
                bgcolor='white',
                bordercolor='black',
                borderwidth=1,
                font=dict(size=10)
            )
        
        return fig
    
    def save(self, path: Union[str, Path]):
        """
        Save calibrator to file using joblib.
        
        Parameters
        ----------
        path : str or Path
            Path to save the calibrator
        
        Raises
        ------
        ValueError
            If calibrator is not fitted
        """
        if not self.is_fitted:
            raise ValueError(
                "Cannot save unfitted calibrator. Call fit() first."
            )
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save calibrator and metadata
        calibrator_data = {
            'calibrator': self.calibrator,
            'is_fitted': self.is_fitted,
            'calibration_metrics': self.calibration_metrics,
            'raw_probs_calib': self._raw_probs_calib,
            'y_true_calib': self._y_true_calib
        }
        
        joblib.dump(calibrator_data, path)
    
    def load(self, path: Union[str, Path]):
        """
        Load calibrator from file.
        
        Parameters
        ----------
        path : str or Path
            Path to load the calibrator from
        
        Raises
        ------
        FileNotFoundError
            If calibrator file doesn't exist
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Calibrator file not found: {path}")
        
        # Load calibrator and metadata
        calibrator_data = joblib.load(path)
        
        self.calibrator = calibrator_data['calibrator']
        self.is_fitted = calibrator_data['is_fitted']
        self.calibration_metrics = calibrator_data['calibration_metrics']
        self._raw_probs_calib = calibrator_data.get('raw_probs_calib')
        self._y_true_calib = calibrator_data.get('y_true_calib')
    
    def get_calibrator_info(self) -> Dict:
        """
        Get information about the calibrator.
        
        Returns
        -------
        dict
            Calibrator information including metrics
        """
        info = {
            'is_fitted': self.is_fitted,
            'out_of_bounds': 'clip',
            'y_min': 0.0,
            'y_max': 1.0
        }
        
        if self.is_fitted:
            info['calibration_metrics'] = self.calibration_metrics
            if self._raw_probs_calib is not None:
                info['n_calibration_samples'] = len(self._raw_probs_calib)
        
        return info


def calibrate_probabilities(
    raw_probs: np.ndarray,
    y_true: np.ndarray,
    save_path: Optional[Union[str, Path]] = None
) -> tuple:
    """
    Convenience function to calibrate probabilities.
    
    Parameters
    ----------
    raw_probs : np.ndarray
        Raw probabilities from classifier
    y_true : np.ndarray
        True binary labels
    save_path : str or Path, optional
        Path to save the calibrator
    
    Returns
    -------
    tuple
        (calibrator, calibrated_probs, metrics)
    """
    calibrator = IsotonicCalibrator()
    metrics = calibrator.fit(raw_probs, y_true)
    calibrated_probs = calibrator.transform(raw_probs)
    
    if save_path is not None:
        calibrator.save(save_path)
    
    return calibrator, calibrated_probs, metrics
