"""
Probability Calculator Module

This module implements 1D and 2D probability calculations with binning,
confidence intervals, and comprehensive metrics per bin/cell.

Handles both numeric and categorical features with automatic binning strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple
from backend.models.confidence_intervals import beta_posterior_ci
from backend.utils.performance_optimizer import cached_calculation


@cached_calculation()
def compute_1d_probability(
    df: pd.DataFrame,
    target: str,
    feature: str,
    conf_level: float = 0.95,
    bins: Union[int, List[float]] = 10,
    min_samples_per_bin: int = 5
) -> pd.DataFrame:
    """
    Compute 1D probability distribution with binning and confidence intervals.
    
    For each bin of the feature, calculates:
    - Probability estimate (p_est)
    - Confidence interval (ci_lower, ci_upper)
    - Mean R-multiple (mean_R)
    - P(R >= 1) (p_hit_1R)
    - P(R >= 2) (p_hit_2R)
    - Sample size (n)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with features and target
    target : str
        Target column name (e.g., 'y_win', 'y_hit_1R')
    feature : str
        Feature column name to bin
    conf_level : float
        Confidence level for intervals (0.8 to 0.99)
    bins : int or list
        Number of bins for numeric features, or list of bin edges
        For categorical features, this parameter is ignored
    min_samples_per_bin : int
        Minimum samples required per bin (bins with fewer samples marked as unreliable)
        
    Returns:
    --------
    pd.DataFrame with columns:
        - bin_index: Bin number
        - bin_left: Left edge of bin (for numeric) or category value
        - bin_right: Right edge of bin (for numeric) or None
        - label: String label for bin
        - n: Sample size in bin
        - successes: Number of successes in bin
        - p_est: Probability estimate
        - ci_lower: Lower confidence bound
        - ci_upper: Upper confidence bound
        - mean_R: Mean R-multiple in bin
        - p_hit_1R: Probability of R >= 1
        - p_hit_2R: Probability of R >= 2
        - is_reliable: Boolean indicating if sample size >= min_samples_per_bin
        
    Raises:
    -------
    ValueError: If target or feature column missing, or invalid parameters
    
    Examples:
    ---------
    >>> result = compute_1d_probability(df, 'y_win', 'trend_strength_tf', bins=10)
    >>> print(result[['label', 'n', 'p_est', 'ci_lower', 'ci_upper']])
    """
    # Validate inputs
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame")
    if feature not in df.columns:
        raise ValueError(f"Feature column '{feature}' not found in DataFrame")
    if not 0 < conf_level < 1:
        raise ValueError(f"Confidence level must be between 0 and 1, got {conf_level}")
    if min_samples_per_bin < 1:
        raise ValueError(f"min_samples_per_bin must be >= 1, got {min_samples_per_bin}")
    
    # Remove rows with missing values
    df_clean = df[[feature, target]].dropna()
    
    if len(df_clean) == 0:
        raise ValueError("No valid data after removing missing values")
    
    # Check if feature is numeric or categorical
    is_numeric = pd.api.types.is_numeric_dtype(df_clean[feature])
    
    if is_numeric:
        # Numeric feature: use binning
        df_clean = df_clean.copy()
        try:
            df_clean['bin'], bin_edges = pd.cut(
                df_clean[feature],
                bins=bins,
                retbins=True,
                duplicates='drop'
            )
        except ValueError as e:
            # Handle case where all values are the same
            raise ValueError(f"Cannot create bins for feature '{feature}': {e}")
    else:
        # Categorical feature: use unique values as bins
        df_clean = df_clean.copy()
        df_clean['bin'] = df_clean[feature]
        bin_edges = None
    
    # Group by bin and calculate statistics
    results = []
    
    for bin_idx, (bin_val, group) in enumerate(df_clean.groupby('bin', observed=True)):
        n = len(group)
        successes = int(group[target].sum())
        
        # Calculate probability with confidence interval
        if n >= min_samples_per_bin:
            ci_result = beta_posterior_ci(successes, n, conf_level=conf_level)
            p_est = ci_result['p_mean']
            ci_lower = ci_result['ci_lower']
            ci_upper = ci_result['ci_upper']
            is_reliable = True
        else:
            # Mark as unreliable but still calculate
            p_est = successes / n if n > 0 else np.nan
            ci_lower = np.nan
            ci_upper = np.nan
            is_reliable = False
        
        # Calculate additional metrics if R_multiple column exists
        if 'R_multiple' in df.columns:
            r_values = df.loc[group.index, 'R_multiple'].dropna()
            mean_R = float(r_values.mean()) if len(r_values) > 0 else np.nan
            p_hit_1R = float((r_values >= 1).mean()) if len(r_values) > 0 else np.nan
            p_hit_2R = float((r_values >= 2).mean()) if len(r_values) > 0 else np.nan
        else:
            mean_R = np.nan
            p_hit_1R = np.nan
            p_hit_2R = np.nan
        
        # Create bin label
        if is_numeric and hasattr(bin_val, 'left'):
            bin_left = float(bin_val.left)
            bin_right = float(bin_val.right)
            label = f"[{bin_left:.2f}, {bin_right:.2f})"
        else:
            bin_left = str(bin_val)
            bin_right = None
            label = str(bin_val)
        
        results.append({
            'bin_index': bin_idx,
            'bin_left': bin_left,
            'bin_right': bin_right,
            'label': label,
            'n': n,
            'successes': successes,
            'p_est': p_est,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'mean_R': mean_R,
            'p_hit_1R': p_hit_1R,
            'p_hit_2R': p_hit_2R,
            'is_reliable': is_reliable
        })
    
    return pd.DataFrame(results)


@cached_calculation()
def compute_2d_probability(
    df: pd.DataFrame,
    target: str,
    feature_x: str,
    feature_y: str,
    conf_level: float = 0.95,
    bins_x: Union[int, List[float]] = 10,
    bins_y: Union[int, List[float]] = 10,
    min_samples_per_cell: int = 5
) -> pd.DataFrame:
    """
    Compute 2D probability distribution for heatmap visualization.
    
    For each cell (bin_x, bin_y), calculates:
    - Probability estimate (p_est)
    - Confidence interval (ci_lower, ci_upper)
    - Mean R-multiple (mean_R)
    - P(R >= 1) (p_hit_1R)
    - P(R >= 2) (p_hit_2R)
    - Sample size (n)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with features and target
    target : str
        Target column name (e.g., 'y_win', 'y_hit_1R')
    feature_x : str
        First feature column name (X-axis)
    feature_y : str
        Second feature column name (Y-axis)
    conf_level : float
        Confidence level for intervals (0.8 to 0.99)
    bins_x : int or list
        Number of bins for feature_x (or list of bin edges)
    bins_y : int or list
        Number of bins for feature_y (or list of bin edges)
    min_samples_per_cell : int
        Minimum samples required per cell
        
    Returns:
    --------
    pd.DataFrame with columns:
        - bin_x: X-axis bin index
        - bin_y: Y-axis bin index
        - x_left: Left edge of X bin
        - x_right: Right edge of X bin
        - y_left: Left edge of Y bin
        - y_right: Right edge of Y bin
        - label_x: String label for X bin
        - label_y: String label for Y bin
        - n: Sample size in cell
        - successes: Number of successes in cell
        - p_est: Probability estimate
        - ci_lower: Lower confidence bound
        - ci_upper: Upper confidence bound
        - mean_R: Mean R-multiple in cell
        - p_hit_1R: Probability of R >= 1
        - p_hit_2R: Probability of R >= 2
        - is_reliable: Boolean indicating if sample size >= min_samples_per_cell
        
    Raises:
    -------
    ValueError: If columns missing or invalid parameters
    
    Examples:
    ---------
    >>> result = compute_2d_probability(
    ...     df, 'y_win', 'trend_strength_tf', 'volatility_regime',
    ...     bins_x=10, bins_y=3
    ... )
    >>> # Create heatmap from result
    >>> pivot = result.pivot(index='bin_y', columns='bin_x', values='p_est')
    """
    # Validate inputs
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame")
    if feature_x not in df.columns:
        raise ValueError(f"Feature X column '{feature_x}' not found in DataFrame")
    if feature_y not in df.columns:
        raise ValueError(f"Feature Y column '{feature_y}' not found in DataFrame")
    if not 0 < conf_level < 1:
        raise ValueError(f"Confidence level must be between 0 and 1, got {conf_level}")
    if min_samples_per_cell < 1:
        raise ValueError(f"min_samples_per_cell must be >= 1, got {min_samples_per_cell}")
    
    # Remove rows with missing values
    df_clean = df[[feature_x, feature_y, target]].dropna()
    
    if len(df_clean) == 0:
        raise ValueError("No valid data after removing missing values")
    
    # Check if features are numeric or categorical
    is_numeric_x = pd.api.types.is_numeric_dtype(df_clean[feature_x])
    is_numeric_y = pd.api.types.is_numeric_dtype(df_clean[feature_y])
    
    df_clean = df_clean.copy()
    
    # Bin feature X
    if is_numeric_x:
        try:
            df_clean['bin_x'], bin_edges_x = pd.cut(
                df_clean[feature_x],
                bins=bins_x,
                retbins=True,
                duplicates='drop'
            )
        except ValueError as e:
            raise ValueError(f"Cannot create bins for feature_x '{feature_x}': {e}")
    else:
        df_clean['bin_x'] = df_clean[feature_x]
        bin_edges_x = None
    
    # Bin feature Y
    if is_numeric_y:
        try:
            df_clean['bin_y'], bin_edges_y = pd.cut(
                df_clean[feature_y],
                bins=bins_y,
                retbins=True,
                duplicates='drop'
            )
        except ValueError as e:
            raise ValueError(f"Cannot create bins for feature_y '{feature_y}': {e}")
    else:
        df_clean['bin_y'] = df_clean[feature_y]
        bin_edges_y = None
    
    # Group by both bins and calculate statistics
    results = []
    
    for (bin_x_val, bin_y_val), group in df_clean.groupby(['bin_x', 'bin_y'], observed=True):
        n = len(group)
        successes = int(group[target].sum())
        
        # Calculate probability with confidence interval
        if n >= min_samples_per_cell:
            ci_result = beta_posterior_ci(successes, n, conf_level=conf_level)
            p_est = ci_result['p_mean']
            ci_lower = ci_result['ci_lower']
            ci_upper = ci_result['ci_upper']
            is_reliable = True
        else:
            # Mark as unreliable but still calculate
            p_est = successes / n if n > 0 else np.nan
            ci_lower = np.nan
            ci_upper = np.nan
            is_reliable = False
        
        # Calculate additional metrics if R_multiple column exists
        if 'R_multiple' in df.columns:
            r_values = df.loc[group.index, 'R_multiple'].dropna()
            mean_R = float(r_values.mean()) if len(r_values) > 0 else np.nan
            p_hit_1R = float((r_values >= 1).mean()) if len(r_values) > 0 else np.nan
            p_hit_2R = float((r_values >= 2).mean()) if len(r_values) > 0 else np.nan
        else:
            mean_R = np.nan
            p_hit_1R = np.nan
            p_hit_2R = np.nan
        
        # Create bin labels and extract edges
        if is_numeric_x and hasattr(bin_x_val, 'left'):
            x_left = float(bin_x_val.left)
            x_right = float(bin_x_val.right)
            label_x = f"[{x_left:.2f}, {x_right:.2f})"
        else:
            x_left = str(bin_x_val)
            x_right = None
            label_x = str(bin_x_val)
        
        if is_numeric_y and hasattr(bin_y_val, 'left'):
            y_left = float(bin_y_val.left)
            y_right = float(bin_y_val.right)
            label_y = f"[{y_left:.2f}, {y_right:.2f})"
        else:
            y_left = str(bin_y_val)
            y_right = None
            label_y = str(bin_y_val)
        
        # Get bin indices
        bin_x_idx = df_clean[df_clean['bin_x'] == bin_x_val].index[0]
        bin_y_idx = df_clean[df_clean['bin_y'] == bin_y_val].index[0]
        
        results.append({
            'bin_x': bin_x_idx,
            'bin_y': bin_y_idx,
            'x_left': x_left,
            'x_right': x_right,
            'y_left': y_left,
            'y_right': y_right,
            'label_x': label_x,
            'label_y': label_y,
            'n': n,
            'successes': successes,
            'p_est': p_est,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'mean_R': mean_R,
            'p_hit_1R': p_hit_1R,
            'p_hit_2R': p_hit_2R,
            'is_reliable': is_reliable
        })
    
    return pd.DataFrame(results)


def get_probability_summary(prob_df: pd.DataFrame) -> Dict[str, any]:
    """
    Get summary statistics from probability calculation results.
    
    Parameters:
    -----------
    prob_df : pd.DataFrame
        Result from compute_1d_probability or compute_2d_probability
        
    Returns:
    --------
    dict with summary statistics:
        - total_bins: Total number of bins/cells
        - reliable_bins: Number of bins with sufficient samples
        - total_samples: Total sample size across all bins
        - mean_probability: Mean probability across bins
        - min_probability: Minimum probability
        - max_probability: Maximum probability
        - probability_range: Max - Min probability
    """
    if len(prob_df) == 0:
        return {
            'total_bins': 0,
            'reliable_bins': 0,
            'total_samples': 0,
            'mean_probability': np.nan,
            'min_probability': np.nan,
            'max_probability': np.nan,
            'probability_range': np.nan
        }
    
    reliable_df = prob_df[prob_df['is_reliable']]
    
    return {
        'total_bins': len(prob_df),
        'reliable_bins': len(reliable_df),
        'total_samples': int(prob_df['n'].sum()),
        'mean_probability': float(reliable_df['p_est'].mean()) if len(reliable_df) > 0 else np.nan,
        'min_probability': float(reliable_df['p_est'].min()) if len(reliable_df) > 0 else np.nan,
        'max_probability': float(reliable_df['p_est'].max()) if len(reliable_df) > 0 else np.nan,
        'probability_range': float(reliable_df['p_est'].max() - reliable_df['p_est'].min()) if len(reliable_df) > 0 else np.nan
    }
