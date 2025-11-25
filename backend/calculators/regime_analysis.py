"""
Regime Analysis Module

This module provides functions for analyzing trading probabilities and performance
across different market regimes (trending/ranging/volatile, etc.).

Requirements: 10.1, 10.2, 10.3, 10.4, 10.5
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from backend.models.confidence_intervals import beta_posterior_ci


def compute_regime_probabilities(
    df: pd.DataFrame,
    regime_column: str,
    target_column: str = 'trade_success',
    conf_level: float = 0.95,
    min_samples: int = 5
) -> pd.DataFrame:
    """
    Calculate P(Win) per regime with confidence intervals.
    
    Args:
        df: DataFrame containing trade data
        regime_column: Name of the column containing regime classifications
        target_column: Name of the target column (1 for win, 0 for loss)
        conf_level: Confidence level for intervals (default: 0.95)
        min_samples: Minimum samples required per regime (default: 5)
    
    Returns:
        DataFrame with columns:
        - regime: Regime identifier
        - n_trades: Number of trades in that regime
        - n_wins: Number of winning trades
        - win_rate: Win rate (proportion of wins)
        - ci_lower: Lower bound of confidence interval
        - ci_upper: Upper bound of confidence interval
        - reliable: Boolean indicating if sample size >= min_samples
    
    Validates: Requirements 10.1
    """
    if df.empty:
        return pd.DataFrame(columns=['regime', 'n_trades', 'n_wins', 'win_rate',
                                     'ci_lower', 'ci_upper', 'reliable'])
    
    # Check if regime column exists
    if regime_column not in df.columns:
        raise ValueError(f"Regime column '{regime_column}' not found in DataFrame")
    
    # Check if target column exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    
    # Group by regime
    results = []
    unique_regimes = sorted(df[regime_column].dropna().unique())
    
    for regime in unique_regimes:
        regime_df = df[df[regime_column] == regime]
        n_trades = len(regime_df)
        
        if n_trades == 0:
            continue
        
        n_wins = regime_df[target_column].sum()
        win_rate = n_wins / n_trades
        
        # Calculate confidence interval using Beta-Binomial
        ci_result = beta_posterior_ci(
            successes=int(n_wins),
            total=n_trades,
            conf_level=conf_level
        )
        
        results.append({
            'regime': regime,
            'n_trades': n_trades,
            'n_wins': int(n_wins),
            'win_rate': win_rate,
            'ci_lower': ci_result['ci_lower'],
            'ci_upper': ci_result['ci_upper'],
            'reliable': n_trades >= min_samples
        })
    
    result_df = pd.DataFrame(results)
    return result_df


def compute_regime_threshold_probs(
    df: pd.DataFrame,
    regime_column: str,
    r_column: str = 'R_multiple',
    thresholds: Optional[List[float]] = None,
    conf_level: float = 0.95,
    min_samples: int = 5
) -> pd.DataFrame:
    """
    Calculate P(R >= threshold) per regime for multiple thresholds.
    
    Args:
        df: DataFrame containing trade data
        regime_column: Name of the column containing regime classifications
        r_column: Name of the R-multiple column
        thresholds: List of R-multiple thresholds (default: [1.0, 2.0])
        conf_level: Confidence level for intervals (default: 0.95)
        min_samples: Minimum samples required per regime (default: 5)
    
    Returns:
        DataFrame with columns:
        - regime: Regime identifier
        - n_trades: Number of trades in that regime
        - p_r_gte_1: Probability of R >= 1
        - p_r_gte_1_ci_lower: Lower CI for P(R >= 1)
        - p_r_gte_1_ci_upper: Upper CI for P(R >= 1)
        - p_r_gte_2: Probability of R >= 2
        - p_r_gte_2_ci_lower: Lower CI for P(R >= 2)
        - p_r_gte_2_ci_upper: Upper CI for P(R >= 2)
        - reliable: Boolean indicating if sample size >= min_samples
    
    Validates: Requirements 10.2
    """
    if thresholds is None:
        thresholds = [1.0, 2.0]
    
    if df.empty:
        columns = ['regime', 'n_trades', 'reliable']
        for threshold in thresholds:
            threshold_str = str(threshold).replace('.', '_')
            columns.extend([
                f'p_r_gte_{threshold_str}',
                f'p_r_gte_{threshold_str}_ci_lower',
                f'p_r_gte_{threshold_str}_ci_upper'
            ])
        return pd.DataFrame(columns=columns)
    
    # Check if columns exist
    if regime_column not in df.columns:
        raise ValueError(f"Regime column '{regime_column}' not found in DataFrame")
    if r_column not in df.columns:
        raise ValueError(f"R-multiple column '{r_column}' not found in DataFrame")
    
    # Group by regime
    results = []
    unique_regimes = sorted(df[regime_column].dropna().unique())
    
    for regime in unique_regimes:
        regime_df = df[df[regime_column] == regime]
        n_trades = len(regime_df)
        
        if n_trades == 0:
            continue
        
        result_row = {
            'regime': regime,
            'n_trades': n_trades,
            'reliable': n_trades >= min_samples
        }
        
        # Calculate probability for each threshold
        for threshold in thresholds:
            n_above_threshold = (regime_df[r_column] >= threshold).sum()
            prob = n_above_threshold / n_trades
            
            # Calculate confidence interval
            ci_result = beta_posterior_ci(
                successes=int(n_above_threshold),
                total=n_trades,
                conf_level=conf_level
            )
            
            threshold_str = str(threshold).replace('.', '_')
            result_row[f'p_r_gte_{threshold_str}'] = prob
            result_row[f'p_r_gte_{threshold_str}_ci_lower'] = ci_result['ci_lower']
            result_row[f'p_r_gte_{threshold_str}_ci_upper'] = ci_result['ci_upper']
        
        results.append(result_row)
    
    result_df = pd.DataFrame(results)
    return result_df


def create_regime_comparison_table(
    df: pd.DataFrame,
    regime_column: str,
    target_column: str = 'trade_success',
    r_column: str = 'R_multiple',
    conf_level: float = 0.95,
    min_samples: int = 5
) -> pd.DataFrame:
    """
    Create comprehensive comparison table with win rate, mean R, and sample size per regime.
    
    Args:
        df: DataFrame containing trade data
        regime_column: Name of the column containing regime classifications
        target_column: Name of the target column (1 for win, 0 for loss)
        r_column: Name of the R-multiple column
        conf_level: Confidence level for intervals (default: 0.95)
        min_samples: Minimum samples required per regime (default: 5)
    
    Returns:
        DataFrame with columns:
        - regime: Regime identifier
        - n_trades: Number of trades
        - win_rate: Win rate (proportion of wins)
        - win_rate_ci_lower: Lower CI for win rate
        - win_rate_ci_upper: Upper CI for win rate
        - mean_r: Mean R-multiple
        - median_r: Median R-multiple
        - std_r: Standard deviation of R-multiple
        - reliable: Boolean indicating if sample size >= min_samples
    
    Validates: Requirements 10.3
    """
    if df.empty:
        return pd.DataFrame(columns=['regime', 'n_trades', 'win_rate', 'win_rate_ci_lower',
                                     'win_rate_ci_upper', 'mean_r', 'median_r', 'std_r', 'reliable'])
    
    # Check if columns exist
    if regime_column not in df.columns:
        raise ValueError(f"Regime column '{regime_column}' not found in DataFrame")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    if r_column not in df.columns:
        raise ValueError(f"R-multiple column '{r_column}' not found in DataFrame")
    
    # Group by regime
    results = []
    unique_regimes = sorted(df[regime_column].dropna().unique())
    
    for regime in unique_regimes:
        regime_df = df[df[regime_column] == regime]
        n_trades = len(regime_df)
        
        if n_trades == 0:
            continue
        
        # Calculate win rate
        n_wins = regime_df[target_column].sum()
        win_rate = n_wins / n_trades
        
        # Calculate confidence interval for win rate
        ci_result = beta_posterior_ci(
            successes=int(n_wins),
            total=n_trades,
            conf_level=conf_level
        )
        
        # Calculate R-multiple statistics
        mean_r = regime_df[r_column].mean()
        median_r = regime_df[r_column].median()
        std_r = regime_df[r_column].std()
        
        results.append({
            'regime': regime,
            'n_trades': n_trades,
            'win_rate': win_rate,
            'win_rate_ci_lower': ci_result['ci_lower'],
            'win_rate_ci_upper': ci_result['ci_upper'],
            'mean_r': mean_r,
            'median_r': median_r,
            'std_r': std_r,
            'reliable': n_trades >= min_samples
        })
    
    result_df = pd.DataFrame(results)
    return result_df


def filter_by_regime(
    df: pd.DataFrame,
    regime_column: str,
    selected_regimes: Union[List, int, str]
) -> pd.DataFrame:
    """
    Filter trades to include only selected regimes.
    
    Args:
        df: DataFrame containing trade data
        regime_column: Name of the column containing regime classifications
        selected_regimes: Single regime or list of regimes to include
    
    Returns:
        Filtered DataFrame containing only trades from selected regimes
    
    Validates: Requirements 10.4
    """
    if df.empty:
        return df.copy()
    
    # Check if regime column exists
    if regime_column not in df.columns:
        raise ValueError(f"Regime column '{regime_column}' not found in DataFrame")
    
    # Convert single regime to list
    if not isinstance(selected_regimes, list):
        selected_regimes = [selected_regimes]
    
    # Filter by selected regimes
    filtered_df = df[df[regime_column].isin(selected_regimes)].copy()
    
    return filtered_df


def compute_regime_transition_matrix(
    df: pd.DataFrame,
    regime_column: str,
    timestamp_column: str = 'Timestamp',
    conf_level: float = 0.95
) -> Dict:
    """
    Calculate transition probability matrix between regimes.
    
    Args:
        df: DataFrame containing trade data with regime information
        regime_column: Name of the column containing regime classifications
        timestamp_column: Name of the timestamp column for ordering
        conf_level: Confidence level for intervals (default: 0.95)
    
    Returns:
        Dictionary containing:
        - transition_matrix: DataFrame with transition probabilities
          (rows = from_regime, columns = to_regime)
        - transition_counts: DataFrame with transition counts
        - confidence_intervals: Dict of dicts with CI for each transition
        - regimes: List of unique regime values
    
    Validates: Requirements 10.5
    """
    if df.empty:
        return {
            'transition_matrix': pd.DataFrame(),
            'transition_counts': pd.DataFrame(),
            'confidence_intervals': {},
            'regimes': []
        }
    
    # Check if columns exist
    if regime_column not in df.columns:
        raise ValueError(f"Regime column '{regime_column}' not found in DataFrame")
    if timestamp_column not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_column}' not found in DataFrame")
    
    # Sort by timestamp to get correct sequence
    df_sorted = df.sort_values(timestamp_column).copy()
    
    # Get regime sequence
    regime_sequence = df_sorted[regime_column].dropna().values
    
    if len(regime_sequence) < 2:
        # Not enough data for transitions
        return {
            'transition_matrix': pd.DataFrame(),
            'transition_counts': pd.DataFrame(),
            'confidence_intervals': {},
            'regimes': list(df_sorted[regime_column].dropna().unique())
        }
    
    # Get unique regimes
    unique_regimes = sorted(df_sorted[regime_column].dropna().unique())
    
    # Initialize transition count matrix
    n_regimes = len(unique_regimes)
    regime_to_idx = {regime: idx for idx, regime in enumerate(unique_regimes)}
    transition_counts = np.zeros((n_regimes, n_regimes), dtype=int)
    
    # Count transitions
    for i in range(len(regime_sequence) - 1):
        from_regime = regime_sequence[i]
        to_regime = regime_sequence[i + 1]
        
        from_idx = regime_to_idx[from_regime]
        to_idx = regime_to_idx[to_regime]
        
        transition_counts[from_idx, to_idx] += 1
    
    # Calculate transition probabilities
    transition_probs = np.zeros((n_regimes, n_regimes), dtype=float)
    confidence_intervals = {}
    
    for from_idx, from_regime in enumerate(unique_regimes):
        total_from = transition_counts[from_idx, :].sum()
        
        if total_from == 0:
            # No transitions from this regime
            transition_probs[from_idx, :] = np.nan
            continue
        
        confidence_intervals[from_regime] = {}
        
        for to_idx, to_regime in enumerate(unique_regimes):
            count = transition_counts[from_idx, to_idx]
            prob = count / total_from
            transition_probs[from_idx, to_idx] = prob
            
            # Calculate confidence interval
            ci_result = beta_posterior_ci(
                successes=int(count),
                total=int(total_from),
                conf_level=conf_level
            )
            
            confidence_intervals[from_regime][to_regime] = {
                'probability': prob,
                'ci_lower': ci_result['ci_lower'],
                'ci_upper': ci_result['ci_upper'],
                'count': int(count),
                'total': int(total_from)
            }
    
    # Create DataFrames
    transition_matrix_df = pd.DataFrame(
        transition_probs,
        index=unique_regimes,
        columns=unique_regimes
    )
    transition_matrix_df.index.name = 'from_regime'
    transition_matrix_df.columns.name = 'to_regime'
    
    transition_counts_df = pd.DataFrame(
        transition_counts,
        index=unique_regimes,
        columns=unique_regimes
    )
    transition_counts_df.index.name = 'from_regime'
    transition_counts_df.columns.name = 'to_regime'
    
    return {
        'transition_matrix': transition_matrix_df,
        'transition_counts': transition_counts_df,
        'confidence_intervals': confidence_intervals,
        'regimes': unique_regimes
    }
