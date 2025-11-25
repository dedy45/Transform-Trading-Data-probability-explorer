"""
Calibration Models for Probability Predictions

This module provides functions for calibrating probability predictions and
assessing their reliability through various metrics.

Functions:
- create_calibration_bins: Create bins for calibration analysis
- compute_reliability_diagram: Compute reliability diagram data
- compute_brier_score: Calculate Brier score for probability accuracy
- compute_ece: Calculate Expected Calibration Error
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union


def create_calibration_bins(
    predicted_probs: np.ndarray,
    n_bins: int = 10,
    strategy: str = 'quantile'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create bins for calibration analysis.
    
    Parameters
    ----------
    predicted_probs : np.ndarray
        Array of predicted probabilities (0-1)
    n_bins : int, default=10
        Number of bins to create
    strategy : str, default='quantile'
        Binning strategy: 'quantile' or 'uniform'
        - 'quantile': Equal number of samples per bin
        - 'uniform': Equal width bins
    
    Returns
    -------
    bin_edges : np.ndarray
        Array of bin edges (length n_bins + 1)
    bin_indices : np.ndarray
        Array of bin indices for each prediction
    
    Raises
    ------
    ValueError
        If strategy is not 'quantile' or 'uniform'
        If n_bins < 2
        If predicted_probs contains values outside [0, 1]
    """
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")
    
    if strategy not in ['quantile', 'uniform']:
        raise ValueError(f"strategy must be 'quantile' or 'uniform', got {strategy}")
    
    # Validate probability range
    if np.any(predicted_probs < 0) or np.any(predicted_probs > 1):
        raise ValueError("predicted_probs must be in range [0, 1]")
    
    if len(predicted_probs) == 0:
        raise ValueError("predicted_probs cannot be empty")
    
    if strategy == 'quantile':
        # Create bins with equal number of samples
        bin_edges = np.quantile(
            predicted_probs,
            np.linspace(0, 1, n_bins + 1)
        )
        # Ensure edges are unique and properly ordered
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < n_bins + 1:
            # Fall back to uniform if quantiles produce too few unique edges
            bin_edges = np.linspace(0, 1, n_bins + 1)
    else:  # uniform
        # Create bins with equal width
        bin_edges = np.linspace(0, 1, n_bins + 1)
    
    # Assign each prediction to a bin
    bin_indices = np.digitize(predicted_probs, bin_edges[1:-1])
    
    return bin_edges, bin_indices


def compute_reliability_diagram(
    predicted_probs: np.ndarray,
    actual_outcomes: np.ndarray,
    n_bins: int = 10,
    strategy: str = 'quantile'
) -> Dict[str, Union[List, np.ndarray]]:
    """
    Compute reliability diagram data for calibration assessment.
    
    A reliability diagram plots mean predicted probability vs observed frequency
    for each bin. Perfect calibration would show points on the diagonal line.
    
    Parameters
    ----------
    predicted_probs : np.ndarray
        Array of predicted probabilities (0-1)
    actual_outcomes : np.ndarray
        Array of actual binary outcomes (0 or 1)
    n_bins : int, default=10
        Number of bins for calibration
    strategy : str, default='quantile'
        Binning strategy: 'quantile' or 'uniform'
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'bin_edges': Array of bin edges
        - 'mean_predicted': Mean predicted probability per bin
        - 'observed_frequency': Observed frequency per bin
        - 'n_samples': Number of samples per bin
        - 'bin_centers': Center point of each bin
    
    Raises
    ------
    ValueError
        If arrays have different lengths
        If actual_outcomes contains values other than 0 or 1
    """
    predicted_probs = np.asarray(predicted_probs)
    actual_outcomes = np.asarray(actual_outcomes)
    
    if len(predicted_probs) != len(actual_outcomes):
        raise ValueError(
            f"predicted_probs and actual_outcomes must have same length, "
            f"got {len(predicted_probs)} and {len(actual_outcomes)}"
        )
    
    if not np.all(np.isin(actual_outcomes, [0, 1])):
        raise ValueError("actual_outcomes must contain only 0 or 1")
    
    # Create bins
    bin_edges, bin_indices = create_calibration_bins(
        predicted_probs, n_bins, strategy
    )
    
    # Calculate statistics per bin
    n_bins_actual = len(bin_edges) - 1
    mean_predicted = np.zeros(n_bins_actual)
    observed_frequency = np.zeros(n_bins_actual)
    n_samples = np.zeros(n_bins_actual, dtype=int)
    
    for i in range(n_bins_actual):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            mean_predicted[i] = np.mean(predicted_probs[mask])
            observed_frequency[i] = np.mean(actual_outcomes[mask])
            n_samples[i] = np.sum(mask)
        else:
            # Empty bin - use NaN
            mean_predicted[i] = np.nan
            observed_frequency[i] = np.nan
            n_samples[i] = 0
    
    # Calculate bin centers for plotting
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    return {
        'bin_edges': bin_edges,
        'mean_predicted': mean_predicted,
        'observed_frequency': observed_frequency,
        'n_samples': n_samples,
        'bin_centers': bin_centers
    }


def compute_brier_score(
    predicted_probs: np.ndarray,
    actual_outcomes: np.ndarray
) -> float:
    """
    Calculate Brier score for probability accuracy.
    
    Brier score measures the mean squared difference between predicted
    probabilities and actual outcomes. Lower is better, with 0 being perfect.
    
    Brier Score = (1/N) * Σ(predicted_prob - actual_outcome)²
    
    Parameters
    ----------
    predicted_probs : np.ndarray
        Array of predicted probabilities (0-1)
    actual_outcomes : np.ndarray
        Array of actual binary outcomes (0 or 1)
    
    Returns
    -------
    float
        Brier score in range [0, 1], where 0 is perfect
    
    Raises
    ------
    ValueError
        If arrays have different lengths
        If actual_outcomes contains values other than 0 or 1
        If predicted_probs contains values outside [0, 1]
    """
    predicted_probs = np.asarray(predicted_probs)
    actual_outcomes = np.asarray(actual_outcomes)
    
    if len(predicted_probs) != len(actual_outcomes):
        raise ValueError(
            f"predicted_probs and actual_outcomes must have same length, "
            f"got {len(predicted_probs)} and {len(actual_outcomes)}"
        )
    
    if not np.all(np.isin(actual_outcomes, [0, 1])):
        raise ValueError("actual_outcomes must contain only 0 or 1")
    
    if np.any(predicted_probs < 0) or np.any(predicted_probs > 1):
        raise ValueError("predicted_probs must be in range [0, 1]")
    
    if len(predicted_probs) == 0:
        raise ValueError("Arrays cannot be empty")
    
    # Calculate Brier score
    brier_score = np.mean((predicted_probs - actual_outcomes) ** 2)
    
    return float(brier_score)


def compute_ece(
    predicted_probs: np.ndarray,
    actual_outcomes: np.ndarray,
    n_bins: int = 10,
    strategy: str = 'quantile'
) -> float:
    """
    Calculate Expected Calibration Error (ECE).
    
    ECE is the weighted average of absolute differences between mean predicted
    probability and observed frequency across bins.
    
    ECE = Σ (n_bin / n_total) * |mean_predicted - observed_frequency|
    
    Parameters
    ----------
    predicted_probs : np.ndarray
        Array of predicted probabilities (0-1)
    actual_outcomes : np.ndarray
        Array of actual binary outcomes (0 or 1)
    n_bins : int, default=10
        Number of bins for calibration
    strategy : str, default='quantile'
        Binning strategy: 'quantile' or 'uniform'
    
    Returns
    -------
    float
        Expected Calibration Error in range [0, 1], where 0 is perfect
    
    Raises
    ------
    ValueError
        If arrays have different lengths
        If actual_outcomes contains values other than 0 or 1
    """
    predicted_probs = np.asarray(predicted_probs)
    actual_outcomes = np.asarray(actual_outcomes)
    
    if len(predicted_probs) == 0:
        raise ValueError("Arrays cannot be empty")
    
    # Get reliability diagram data
    reliability = compute_reliability_diagram(
        predicted_probs, actual_outcomes, n_bins, strategy
    )
    
    mean_predicted = reliability['mean_predicted']
    observed_frequency = reliability['observed_frequency']
    n_samples = reliability['n_samples']
    
    # Calculate ECE (weighted average of absolute deviations)
    total_samples = len(predicted_probs)
    ece = 0.0
    
    for i in range(len(mean_predicted)):
        if n_samples[i] > 0 and not np.isnan(mean_predicted[i]):
            weight = n_samples[i] / total_samples
            deviation = abs(mean_predicted[i] - observed_frequency[i])
            ece += weight * deviation
    
    return float(ece)
