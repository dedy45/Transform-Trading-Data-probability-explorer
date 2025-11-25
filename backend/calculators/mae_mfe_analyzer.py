"""
MAE/MFE Analyzer

This module provides functions for analyzing Maximum Adverse Excursion (MAE)
and Maximum Favorable Excursion (MFE) patterns to optimize stop loss and
take profit levels.

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats


def analyze_mae_patterns(
    df: pd.DataFrame,
    mae_column: str = 'MAE_R',
    r_column: str = 'R_multiple'
) -> Dict[str, Dict[str, float]]:
    """
    Analyze MAE patterns separately for winning and losing trades.
    
    MAE (Maximum Adverse Excursion) is the maximum drawdown during a trade.
    This function calculates MAE distributions for winners vs losers to help
    optimize stop loss placement.
    
    Args:
        df: DataFrame containing trade data
        mae_column: Name of the MAE column (default: 'MAE_R')
        r_column: Name of the R-multiple column (default: 'R_multiple')
    
    Returns:
        Dictionary containing:
        - winners: Dict with MAE statistics for winning trades
            - mean_mae: Average MAE for winners
            - median_mae: Median MAE for winners
            - std_mae: Standard deviation of MAE for winners
            - min_mae: Minimum MAE for winners
            - max_mae: Maximum MAE for winners
            - count: Number of winning trades
        - losers: Dict with MAE statistics for losing trades
            - mean_mae: Average MAE for losers
            - median_mae: Median MAE for losers
            - std_mae: Standard deviation of MAE for losers
            - min_mae: Minimum MAE for losers
            - max_mae: Maximum MAE for losers
            - count: Number of losing trades
    
    Validates: Requirements 4.1
    """
    if df.empty:
        empty_stats = {
            'mean_mae': np.nan,
            'median_mae': np.nan,
            'std_mae': np.nan,
            'min_mae': np.nan,
            'max_mae': np.nan,
            'count': 0
        }
        return {'winners': empty_stats.copy(), 'losers': empty_stats.copy()}
    
    # Separate winners and losers
    winners = df[df[r_column] > 0].copy()
    losers = df[df[r_column] <= 0].copy()
    
    def calculate_mae_stats(trades_df):
        """Helper function to calculate MAE statistics"""
        if trades_df.empty or mae_column not in trades_df.columns:
            return {
                'mean_mae': np.nan,
                'median_mae': np.nan,
                'std_mae': np.nan,
                'min_mae': np.nan,
                'max_mae': np.nan,
                'count': 0
            }
        
        mae_values = trades_df[mae_column].dropna()
        
        if len(mae_values) == 0:
            return {
                'mean_mae': np.nan,
                'median_mae': np.nan,
                'std_mae': np.nan,
                'min_mae': np.nan,
                'max_mae': np.nan,
                'count': 0
            }
        
        return {
            'mean_mae': mae_values.mean(),
            'median_mae': mae_values.median(),
            'std_mae': mae_values.std(),
            'min_mae': mae_values.min(),
            'max_mae': mae_values.max(),
            'count': len(mae_values)
        }
    
    return {
        'winners': calculate_mae_stats(winners),
        'losers': calculate_mae_stats(losers)
    }


def analyze_mfe_patterns(
    df: pd.DataFrame,
    mfe_column: str = 'MFE_R',
    r_column: str = 'R_multiple'
) -> Dict[str, float]:
    """
    Analyze MFE patterns and calculate correlation with final R-multiple.
    
    MFE (Maximum Favorable Excursion) is the maximum profit during a trade.
    This function calculates the correlation between MFE and final R to understand
    how much of the potential profit is typically captured.
    
    Args:
        df: DataFrame containing trade data
        mfe_column: Name of the MFE column (default: 'MFE_R')
        r_column: Name of the R-multiple column (default: 'R_multiple')
    
    Returns:
        Dictionary containing:
        - correlation: Pearson correlation between MFE and final R
        - p_value: P-value for the correlation
        - mean_mfe: Average MFE across all trades
        - median_mfe: Median MFE across all trades
        - std_mfe: Standard deviation of MFE
        - mean_r: Average final R-multiple
        - count: Number of trades analyzed
    
    Validates: Requirements 4.2
    """
    if df.empty:
        return {
            'correlation': np.nan,
            'p_value': np.nan,
            'mean_mfe': np.nan,
            'median_mfe': np.nan,
            'std_mfe': np.nan,
            'mean_r': np.nan,
            'count': 0
        }
    
    # Get valid data points (both MFE and R must be present)
    valid_df = df[[mfe_column, r_column]].dropna()
    
    if len(valid_df) < 2:
        return {
            'correlation': np.nan,
            'p_value': np.nan,
            'mean_mfe': np.nan,
            'median_mfe': np.nan,
            'std_mfe': np.nan,
            'mean_r': np.nan,
            'count': len(valid_df)
        }
    
    mfe_values = valid_df[mfe_column]
    r_values = valid_df[r_column]
    
    # Calculate Pearson correlation
    correlation, p_value = stats.pearsonr(mfe_values, r_values)
    
    return {
        'correlation': correlation,
        'p_value': p_value,
        'mean_mfe': mfe_values.mean(),
        'median_mfe': mfe_values.median(),
        'std_mfe': mfe_values.std(),
        'mean_r': r_values.mean(),
        'count': len(valid_df)
    }


def calculate_profit_left(
    df: pd.DataFrame,
    mfe_column: str = 'MFE_R',
    r_column: str = 'R_multiple'
) -> Dict[str, float]:
    """
    Calculate average profit left on the table for winning trades.
    
    Profit left = MFE - Final_R, representing the difference between
    the maximum profit achieved during the trade and the actual exit profit.
    
    Args:
        df: DataFrame containing trade data
        mfe_column: Name of the MFE column (default: 'MFE_R')
        r_column: Name of the R-multiple column (default: 'R_multiple')
    
    Returns:
        Dictionary containing:
        - mean_profit_left: Average (MFE - Final_R) for winners
        - median_profit_left: Median profit left for winners
        - std_profit_left: Standard deviation of profit left
        - min_profit_left: Minimum profit left (should be >= 0 for winners)
        - max_profit_left: Maximum profit left
        - capture_ratio: Average (Final_R / MFE) for winners
        - count: Number of winning trades analyzed
    
    Validates: Requirements 4.3
    """
    if df.empty:
        return {
            'mean_profit_left': np.nan,
            'median_profit_left': np.nan,
            'std_profit_left': np.nan,
            'min_profit_left': np.nan,
            'max_profit_left': np.nan,
            'capture_ratio': np.nan,
            'count': 0
        }
    
    # Filter for winning trades only
    winners = df[df[r_column] > 0].copy()
    
    if winners.empty:
        return {
            'mean_profit_left': np.nan,
            'median_profit_left': np.nan,
            'std_profit_left': np.nan,
            'min_profit_left': np.nan,
            'max_profit_left': np.nan,
            'capture_ratio': np.nan,
            'count': 0
        }
    
    # Get valid data points
    valid_winners = winners[[mfe_column, r_column]].dropna()
    
    if valid_winners.empty:
        return {
            'mean_profit_left': np.nan,
            'median_profit_left': np.nan,
            'std_profit_left': np.nan,
            'min_profit_left': np.nan,
            'max_profit_left': np.nan,
            'capture_ratio': np.nan,
            'count': 0
        }
    
    # Calculate profit left
    profit_left = valid_winners[mfe_column] - valid_winners[r_column]
    
    # Calculate capture ratio (what percentage of MFE was captured)
    # Avoid division by zero and very small numbers
    capture_ratios = []
    for _, row in valid_winners.iterrows():
        if row[mfe_column] > 1e-6:  # Avoid division by very small numbers
            ratio = row[r_column] / row[mfe_column]
            # Cap ratio to avoid overflow issues
            if not np.isinf(ratio) and not np.isnan(ratio):
                capture_ratios.append(ratio)
    
    mean_capture_ratio = np.mean(capture_ratios) if capture_ratios else np.nan
    
    return {
        'mean_profit_left': profit_left.mean(),
        'median_profit_left': profit_left.median(),
        'std_profit_left': profit_left.std(),
        'min_profit_left': profit_left.min(),
        'max_profit_left': profit_left.max(),
        'capture_ratio': mean_capture_ratio,
        'count': len(valid_winners)
    }


def optimize_sl_level(
    df: pd.DataFrame,
    sl_levels: Optional[List[float]] = None,
    mae_column: str = 'MAE_R',
    r_column: str = 'R_multiple'
) -> pd.DataFrame:
    """
    Optimize stop loss level by analyzing impact on winners and losers.
    
    For each SL level, calculate:
    - Percentage of winners that would be stopped out
    - Percentage of losers that would be avoided
    - Net impact on expectancy
    
    Args:
        df: DataFrame containing trade data
        sl_levels: List of SL levels to test (in R units, e.g., [0.5, 0.75, 1.0, 1.5])
                  Default: [0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        mae_column: Name of the MAE column (default: 'MAE_R')
        r_column: Name of the R-multiple column (default: 'R_multiple')
    
    Returns:
        DataFrame with columns:
        - sl_level: The stop loss level tested
        - pct_winners_stopped: Percentage of winners that would hit this SL
        - pct_losers_avoided: Percentage of losers that would be stopped before final loss
        - winners_stopped_count: Number of winners stopped
        - losers_avoided_count: Number of losers avoided
        - net_benefit: (losers_avoided - winners_stopped) as percentage
        - remaining_trades: Number of trades that would remain
        - new_expectancy: Expected R if this SL level was used
    
    Validates: Requirements 4.4
    """
    if sl_levels is None:
        sl_levels = [0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    
    if df.empty:
        return pd.DataFrame(columns=[
            'sl_level', 'pct_winners_stopped', 'pct_losers_avoided',
            'winners_stopped_count', 'losers_avoided_count', 'net_benefit',
            'remaining_trades', 'new_expectancy'
        ])
    
    # Get valid data
    valid_df = df[[mae_column, r_column]].dropna()
    
    if valid_df.empty:
        return pd.DataFrame(columns=[
            'sl_level', 'pct_winners_stopped', 'pct_losers_avoided',
            'winners_stopped_count', 'losers_avoided_count', 'net_benefit',
            'remaining_trades', 'new_expectancy'
        ])
    
    winners = valid_df[valid_df[r_column] > 0]
    losers = valid_df[valid_df[r_column] <= 0]
    
    total_winners = len(winners)
    total_losers = len(losers)
    
    results = []
    
    for sl_level in sl_levels:
        # Winners stopped: those whose MAE exceeded the SL level
        winners_stopped = winners[winners[mae_column] >= sl_level]
        winners_stopped_count = len(winners_stopped)
        pct_winners_stopped = (winners_stopped_count / total_winners * 100) if total_winners > 0 else 0.0
        
        # Losers avoided: those whose MAE exceeded the SL level
        # (they would have been stopped before reaching their final loss)
        losers_avoided = losers[losers[mae_column] >= sl_level]
        losers_avoided_count = len(losers_avoided)
        pct_losers_avoided = (losers_avoided_count / total_losers * 100) if total_losers > 0 else 0.0
        
        # Calculate new expectancy if this SL was applied
        # Winners that hit SL would exit at -sl_level
        # Losers that hit SL would exit at -sl_level
        # Others keep their original R
        new_r_values = []
        
        for _, row in valid_df.iterrows():
            if row[mae_column] >= sl_level:
                # Trade hit the SL
                new_r_values.append(-sl_level)
            else:
                # Trade didn't hit SL, keep original R
                new_r_values.append(row[r_column])
        
        new_expectancy = np.mean(new_r_values) if new_r_values else np.nan
        
        # Net benefit: more losers avoided is good, more winners stopped is bad
        net_benefit = pct_losers_avoided - pct_winners_stopped
        
        results.append({
            'sl_level': sl_level,
            'pct_winners_stopped': pct_winners_stopped,
            'pct_losers_avoided': pct_losers_avoided,
            'winners_stopped_count': winners_stopped_count,
            'losers_avoided_count': losers_avoided_count,
            'net_benefit': net_benefit,
            'remaining_trades': len(new_r_values),
            'new_expectancy': new_expectancy
        })
    
    return pd.DataFrame(results)


def optimize_tp_level(
    df: pd.DataFrame,
    tp_levels: Optional[List[float]] = None,
    mfe_column: str = 'MFE_R',
    r_column: str = 'R_multiple'
) -> pd.DataFrame:
    """
    Optimize take profit level by analyzing MFE capture percentage.
    
    For each TP level, calculate:
    - Percentage of MFE captured
    - Percentage of trades that would hit the TP
    - Impact on expectancy
    
    Args:
        df: DataFrame containing trade data
        tp_levels: List of TP levels to test (in R units, e.g., [1.0, 1.5, 2.0, 3.0])
                  Default: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
        mfe_column: Name of the MFE column (default: 'MFE_R')
        r_column: Name of the R-multiple column (default: 'R_multiple')
    
    Returns:
        DataFrame with columns:
        - tp_level: The take profit level tested
        - pct_trades_hit_tp: Percentage of trades that reached this TP level
        - trades_hit_tp_count: Number of trades that hit TP
        - avg_mfe_capture_pct: Average percentage of MFE captured
        - new_expectancy: Expected R if this TP level was used
        - expectancy_change: Change in expectancy vs original
    
    Validates: Requirements 4.5
    """
    if tp_levels is None:
        tp_levels = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    
    if df.empty:
        return pd.DataFrame(columns=[
            'tp_level', 'pct_trades_hit_tp', 'trades_hit_tp_count',
            'avg_mfe_capture_pct', 'new_expectancy', 'expectancy_change'
        ])
    
    # Get valid data
    valid_df = df[[mfe_column, r_column]].dropna()
    
    if valid_df.empty:
        return pd.DataFrame(columns=[
            'tp_level', 'pct_trades_hit_tp', 'trades_hit_tp_count',
            'avg_mfe_capture_pct', 'new_expectancy', 'expectancy_change'
        ])
    
    # Calculate original expectancy
    original_expectancy = valid_df[r_column].mean()
    
    results = []
    
    for tp_level in tp_levels:
        # Trades that hit TP: those whose MFE reached the TP level
        trades_hit_tp = valid_df[valid_df[mfe_column] >= tp_level]
        trades_hit_tp_count = len(trades_hit_tp)
        pct_trades_hit_tp = (trades_hit_tp_count / len(valid_df) * 100) if len(valid_df) > 0 else 0.0
        
        # Calculate new R values if this TP was applied
        new_r_values = []
        mfe_capture_percentages = []
        
        for _, row in valid_df.iterrows():
            if row[mfe_column] >= tp_level:
                # Trade hit the TP, exit at tp_level
                new_r = tp_level
                new_r_values.append(new_r)
                # Calculate what percentage of MFE was captured
                if row[mfe_column] > 1e-6:  # Avoid division by very small numbers
                    capture_pct = (tp_level / row[mfe_column]) * 100
                    # Cap to avoid overflow
                    if not np.isinf(capture_pct) and not np.isnan(capture_pct):
                        mfe_capture_percentages.append(capture_pct)
            else:
                # Trade didn't hit TP, keep original R
                new_r_values.append(row[r_column])
                # If MFE > 0, calculate capture percentage
                if row[mfe_column] > 1e-6:  # Avoid division by very small numbers
                    capture_pct = (row[r_column] / row[mfe_column]) * 100
                    # Cap to avoid overflow
                    if not np.isinf(capture_pct) and not np.isnan(capture_pct):
                        mfe_capture_percentages.append(capture_pct)
        
        new_expectancy = np.mean(new_r_values) if new_r_values else np.nan
        expectancy_change = new_expectancy - original_expectancy if not np.isnan(new_expectancy) else np.nan
        avg_mfe_capture_pct = np.mean(mfe_capture_percentages) if mfe_capture_percentages else np.nan
        
        results.append({
            'tp_level': tp_level,
            'pct_trades_hit_tp': pct_trades_hit_tp,
            'trades_hit_tp_count': trades_hit_tp_count,
            'avg_mfe_capture_pct': avg_mfe_capture_pct,
            'new_expectancy': new_expectancy,
            'expectancy_change': expectancy_change
        })
    
    return pd.DataFrame(results)
