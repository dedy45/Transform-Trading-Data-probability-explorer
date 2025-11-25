"""
Expectancy and R-Multiple Analyzer

This module provides functions for calculating trading expectancy, R-multiple distributions,
and related performance metrics.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional


def compute_expectancy_R(df: pd.DataFrame, r_column: str = 'R_multiple') -> Dict[str, float]:
    """
    Calculate global expectancy in R-multiple, dollar, and percentage terms.
    
    Expectancy is the expected profit per trade, calculated as the mean R-multiple.
    
    Args:
        df: DataFrame containing trade data
        r_column: Name of the R-multiple column (default: 'R_multiple')
    
    Returns:
        Dictionary containing:
        - expectancy_R: Mean R-multiple (expected R per trade)
        - expectancy_dollar: Mean profit in dollars (if 'net_profit' column exists)
        - expectancy_percent: Mean profit as percentage (if 'risk_percent' column exists)
        - win_rate: Proportion of winning trades
        - avg_win_R: Average R for winning trades
        - avg_loss_R: Average R for losing trades
        - total_trades: Number of trades
    
    Validates: Requirements 3.1
    """
    if df.empty:
        return {
            'expectancy_R': np.nan,
            'expectancy_dollar': np.nan,
            'expectancy_percent': np.nan,
            'win_rate': np.nan,
            'avg_win_R': np.nan,
            'avg_loss_R': np.nan,
            'total_trades': 0
        }
    
    # Calculate expectancy in R-multiple
    expectancy_R = df[r_column].dropna().mean() if not df[r_column].dropna().empty else np.nan
    
    # Calculate win rate
    winners = df[df[r_column] > 0]
    losers = df[df[r_column] < 0]
    win_rate = len(winners) / len(df) if len(df) > 0 else 0.0
    
    # Calculate average win and loss
    avg_win_R = winners[r_column].dropna().mean() if len(winners) > 0 else np.nan
    avg_loss_R = losers[r_column].dropna().mean() if len(losers) > 0 else np.nan
    
    result = {
        'expectancy_R': expectancy_R,
        'win_rate': win_rate,
        'avg_win_R': avg_win_R,
        'avg_loss_R': avg_loss_R,
        'total_trades': len(df)
    }
    
    # Calculate expectancy in dollars if available
    if 'net_profit' in df.columns:
        result['expectancy_dollar'] = df['net_profit'].dropna().mean() if not df['net_profit'].dropna().empty else np.nan
    else:
        result['expectancy_dollar'] = np.nan
    
    # Calculate expectancy in percentage if available
    if 'risk_percent' in df.columns and 'net_profit' in df.columns:
        # Expectancy as percentage of risk
        ratio = (df['net_profit'] / df['risk_percent']).replace([np.inf, -np.inf], np.nan).dropna()
        result['expectancy_percent'] = ratio.mean() if not ratio.empty else np.nan
    else:
        result['expectancy_percent'] = np.nan
    
    return result


def compute_expectancy_by_group(
    df: pd.DataFrame,
    group_column: str,
    r_column: str = 'R_multiple'
) -> pd.DataFrame:
    """
    Calculate expectancy metrics grouped by a specified column.
    
    Args:
        df: DataFrame containing trade data
        group_column: Column name to group by (e.g., 'session', 'trend_regime')
        r_column: Name of the R-multiple column (default: 'R_multiple')
    
    Returns:
        DataFrame with columns:
        - group_value: The grouping value
        - expectancy_R: Mean R-multiple for the group
        - win_rate: Win rate for the group
        - avg_R: Average R-multiple for the group
        - sample_size: Number of trades in the group
    
    Validates: Requirements 3.2
    """
    if df.empty:
        return pd.DataFrame(columns=['group_value', 'expectancy_R', 'win_rate', 'avg_R', 'sample_size'])
    
    grouped = df.groupby(group_column, observed=True)
    
    results = []
    for group_value, group_df in grouped:
        expectancy_metrics = compute_expectancy_R(group_df, r_column)
        
        results.append({
            'group_value': group_value,
            'expectancy_R': expectancy_metrics['expectancy_R'],
            'win_rate': expectancy_metrics['win_rate'],
            'avg_R': expectancy_metrics['expectancy_R'],  # Same as expectancy_R
            'sample_size': expectancy_metrics['total_trades']
        })
    
    result_df = pd.DataFrame(results)
    
    # Sort by expectancy descending
    result_df = result_df.sort_values('expectancy_R', ascending=False).reset_index(drop=True)
    
    return result_df


def compute_r_percentiles(
    df: pd.DataFrame,
    r_column: str = 'R_multiple',
    percentiles: Optional[List[float]] = None
) -> Dict[str, float]:
    """
    Calculate percentiles of R-multiple distribution.
    
    Args:
        df: DataFrame containing trade data
        r_column: Name of the R-multiple column (default: 'R_multiple')
        percentiles: List of percentiles to calculate (default: [25, 50, 75, 90, 95])
    
    Returns:
        Dictionary with percentile values:
        - p25, p50, p75, p90, p95: Percentile values
        - min_R: Minimum R-multiple
        - max_R: Maximum R-multiple
        - mean_R: Mean R-multiple
        - std_R: Standard deviation of R-multiple
    
    Validates: Requirements 3.3
    """
    if percentiles is None:
        percentiles = [25, 50, 75, 90, 95]
    
    if df.empty:
        result = {f'p{int(p)}': np.nan for p in percentiles}
        result.update({
            'min_R': np.nan,
            'max_R': np.nan,
            'mean_R': np.nan,
            'std_R': np.nan
        })
        return result
    
    r_values = df[r_column].dropna()
    
    if len(r_values) == 0:
        result = {f'p{int(p)}': np.nan for p in percentiles}
        result.update({
            'min_R': np.nan,
            'max_R': np.nan,
            'mean_R': np.nan,
            'std_R': np.nan
        })
        return result
    
    # Calculate percentiles
    result = {}
    for p in percentiles:
        result[f'p{int(p)}'] = np.percentile(r_values, p)
    
    # Add summary statistics
    result['min_R'] = r_values.min()
    result['max_R'] = r_values.max()
    result['mean_R'] = r_values.mean()
    result['std_R'] = r_values.std()
    
    return result


def compute_r_threshold_probabilities(
    df: pd.DataFrame,
    r_column: str = 'R_multiple',
    thresholds: Optional[List[float]] = None
) -> Dict[str, float]:
    """
    Calculate probabilities of exceeding R-multiple thresholds.
    
    Args:
        df: DataFrame containing trade data
        r_column: Name of the R-multiple column (default: 'R_multiple')
        thresholds: List of R thresholds (default: [1, 2, 3])
    
    Returns:
        Dictionary with probability values:
        - p_r_gt_1: P(R > 1)
        - p_r_gt_2: P(R > 2)
        - p_r_gt_3: P(R > 3)
        - p_r_gte_1: P(R >= 1)
        - p_r_gte_2: P(R >= 2)
        - p_r_gte_3: P(R >= 3)
    
    Validates: Requirements 3.4
    """
    if thresholds is None:
        thresholds = [1, 2, 3]
    
    if df.empty:
        result = {}
        for t in thresholds:
            result[f'p_r_gt_{int(t)}'] = np.nan
            result[f'p_r_gte_{int(t)}'] = np.nan
        return result
    
    r_values = df[r_column].dropna()
    
    if len(r_values) == 0:
        result = {}
        for t in thresholds:
            result[f'p_r_gt_{int(t)}'] = np.nan
            result[f'p_r_gte_{int(t)}'] = np.nan
        return result
    
    result = {}
    for t in thresholds:
        # P(R > threshold)
        result[f'p_r_gt_{int(t)}'] = (r_values > t).mean()
        # P(R >= threshold)
        result[f'p_r_gte_{int(t)}'] = (r_values >= t).mean()
    
    return result


def compute_expected_r_per_bin(
    df: pd.DataFrame,
    bin_column: str,
    r_column: str = 'R_multiple'
) -> pd.DataFrame:
    """
    Calculate expected R-multiple per bin/group.
    
    This is essentially the same as compute_expectancy_by_group but with
    a focus on expected R values.
    
    Args:
        df: DataFrame containing trade data
        bin_column: Column name defining bins (e.g., 'trend_strength_bin')
        r_column: Name of the R-multiple column (default: 'R_multiple')
    
    Returns:
        DataFrame with columns:
        - bin_value: The bin identifier
        - expected_R: Mean R-multiple for the bin
        - sample_size: Number of trades in the bin
        - std_R: Standard deviation of R in the bin
        - min_R: Minimum R in the bin
        - max_R: Maximum R in the bin
    
    Validates: Requirements 3.5
    """
    if df.empty:
        return pd.DataFrame(columns=['bin_value', 'expected_R', 'sample_size', 'std_R', 'min_R', 'max_R'])
    
    grouped = df.groupby(bin_column, observed=True)
    
    results = []
    for bin_value, group_df in grouped:
        r_values = group_df[r_column].dropna()
        
        if len(r_values) > 0:
            results.append({
                'bin_value': bin_value,
                'expected_R': r_values.mean(),
                'sample_size': len(r_values),
                'std_R': r_values.std(),
                'min_R': r_values.min(),
                'max_R': r_values.max()
            })
    
    result_df = pd.DataFrame(results)
    
    # Sort by bin_value
    result_df = result_df.sort_values('bin_value').reset_index(drop=True)
    
    return result_df
