"""
Composite Score Engine

This module provides functions for calculating composite probability scores
that combine multiple trading signals into a single actionable score.

Requirements: 14.1, 14.2, 14.3, 14.4, 14.5
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


# Default component weights (must sum to 1.0)
DEFAULT_WEIGHTS = {
    'win_rate': 0.30,
    'expected_r': 0.25,
    'structure_quality': 0.15,
    'time_based': 0.10,
    'correlation': 0.10,
    'entry_quality': 0.10
}


def calculate_component_scores(
    df: pd.DataFrame,
    prob_columns: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Calculate individual component scores (0-100) for each trade.
    
    Args:
        df: DataFrame containing trade data with probability features
        prob_columns: Dictionary mapping component names to column names
                     If None, uses default column mappings
    
    Returns:
        DataFrame with original data plus 6 component score columns:
        - score_win_rate: Win rate component (0-100)
        - score_expected_r: Expected R component (0-100)
        - score_structure_quality: Structure quality component (0-100)
        - score_time_based: Time-based component (0-100)
        - score_correlation: Correlation component (0-100)
        - score_entry_quality: Entry quality component (0-100)
    
    Validates: Requirements 14.1, 14.2
    """
    if df.empty:
        return df
    
    result_df = df.copy()
    
    # Default column mappings
    if prob_columns is None:
        prob_columns = {
            'win_rate': 'prob_global_win',
            'expected_r': 'prob_global_hit_1R',
            'structure_quality': 'prob_entropy_win',
            'time_based': 'prob_session_win',
            'correlation': 'prob_trend_dir_alignment_win',
            'entry_quality': 'prob_sr_zone_win'
        }
    
    # Calculate each component score (convert probability 0-1 to score 0-100)
    for component, col_name in prob_columns.items():
        score_col = f'score_{component}'
        
        if col_name in df.columns:
            # Convert probability (0-1) to score (0-100)
            result_df[score_col] = df[col_name].fillna(0.5) * 100
            # Clip to valid range
            result_df[score_col] = result_df[score_col].clip(0, 100)
        else:
            # If column doesn't exist, use neutral score of 50
            result_df[score_col] = 50.0
    
    return result_df


def calculate_composite_score(
    df: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
    prob_columns: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Calculate composite probability score (0-100) for each trade.
    
    The composite score is a weighted combination of 6 components:
    - Win rate (default 30%)
    - Expected R (default 25%)
    - Structure quality (default 15%)
    - Time-based (default 10%)
    - Correlation (default 10%)
    - Entry quality (default 10%)
    
    Args:
        df: DataFrame containing trade data with probability features
        weights: Dictionary of component weights (must sum to 1.0)
                If None, uses DEFAULT_WEIGHTS
        prob_columns: Dictionary mapping component names to column names
                     If None, uses default column mappings
    
    Returns:
        DataFrame with original data plus:
        - score_win_rate, score_expected_r, etc.: Individual component scores
        - composite_score: Weighted composite score (0-100)
        - score_breakdown: Dictionary with component contributions
    
    Validates: Requirements 14.1, 14.2
    """
    if df.empty:
        result_df = df.copy()
        result_df['composite_score'] = np.nan
        return result_df
    
    # Use default weights if not provided
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()
    
    # Validate weights sum to 1.0
    weight_sum = sum(weights.values())
    if not np.isclose(weight_sum, 1.0, atol=1e-6):
        raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")
    
    # Calculate component scores
    result_df = calculate_component_scores(df, prob_columns)
    
    # Calculate weighted composite score
    composite = np.zeros(len(result_df))
    
    for component, weight in weights.items():
        score_col = f'score_{component}'
        if score_col in result_df.columns:
            composite += result_df[score_col].values * weight
    
    result_df['composite_score'] = composite
    
    # Ensure composite score is in valid range
    result_df['composite_score'] = result_df['composite_score'].clip(0, 100)
    
    return result_df


def backtest_score_threshold(
    df: pd.DataFrame,
    thresholds: Optional[List[float]] = None,
    r_column: str = 'R_multiple',
    score_column: str = 'composite_score'
) -> pd.DataFrame:
    """
    Backtest multiple score thresholds to evaluate their performance.
    
    Args:
        df: DataFrame containing trade data with composite scores
        thresholds: List of score thresholds to test (default: [50, 60, 70, 80])
        r_column: Name of the R-multiple column (default: 'R_multiple')
        score_column: Name of the score column (default: 'composite_score')
    
    Returns:
        DataFrame with columns:
        - threshold: Score threshold value
        - win_rate: Win rate for trades with score >= threshold
        - expectancy: Expectancy (mean R) for trades with score >= threshold
        - trade_frequency: Number of trades with score >= threshold
        - trade_frequency_pct: Percentage of total trades
        - avg_score: Average score of filtered trades
    
    Validates: Requirements 14.3
    """
    if thresholds is None:
        thresholds = [50, 60, 70, 80]
    
    if df.empty or score_column not in df.columns:
        return pd.DataFrame(columns=[
            'threshold', 'win_rate', 'expectancy', 'trade_frequency',
            'trade_frequency_pct', 'avg_score'
        ])
    
    results = []
    total_trades = len(df)
    
    for threshold in thresholds:
        # Filter trades by threshold
        filtered_df = df[df[score_column] >= threshold].copy()
        
        if len(filtered_df) == 0:
            results.append({
                'threshold': threshold,
                'win_rate': np.nan,
                'expectancy': np.nan,
                'trade_frequency': 0,
                'trade_frequency_pct': 0.0,
                'avg_score': np.nan
            })
            continue
        
        # Calculate metrics for filtered trades
        if r_column in filtered_df.columns:
            winners = filtered_df[filtered_df[r_column] > 0]
            win_rate = len(winners) / len(filtered_df) if len(filtered_df) > 0 else 0.0
            expectancy = filtered_df[r_column].mean()
        else:
            win_rate = np.nan
            expectancy = np.nan
        
        results.append({
            'threshold': threshold,
            'win_rate': win_rate,
            'expectancy': expectancy,
            'trade_frequency': len(filtered_df),
            'trade_frequency_pct': (len(filtered_df) / total_trades) * 100,
            'avg_score': filtered_df[score_column].mean()
        })
    
    result_df = pd.DataFrame(results)
    return result_df


def filter_by_score(
    df: pd.DataFrame,
    threshold: float,
    score_column: str = 'composite_score'
) -> pd.DataFrame:
    """
    Filter trades by composite score threshold.
    
    Args:
        df: DataFrame containing trade data with composite scores
        threshold: Minimum score threshold (0-100)
        score_column: Name of the score column (default: 'composite_score')
    
    Returns:
        DataFrame containing only trades with score >= threshold
    
    Validates: Requirements 14.4
    """
    if df.empty or score_column not in df.columns:
        return df
    
    # Validate threshold
    if not 0 <= threshold <= 100:
        raise ValueError(f"Threshold must be between 0 and 100, got {threshold}")
    
    # Filter by threshold
    filtered_df = df[df[score_column] >= threshold].copy()
    
    return filtered_df


def classify_recommendation(
    score: float
) -> str:
    """
    Classify a composite score into a recommendation label.
    
    Classification rules:
    - [80-100]: STRONG BUY
    - [60-80): BUY
    - [40-60): NEUTRAL
    - [0-40): AVOID
    
    Args:
        score: Composite score (0-100)
    
    Returns:
        Recommendation label: 'STRONG BUY', 'BUY', 'NEUTRAL', or 'AVOID'
    
    Validates: Requirements 14.5
    """
    if pd.isna(score):
        return 'NEUTRAL'
    
    if score >= 80:
        return 'STRONG BUY'
    elif score >= 60:
        return 'BUY'
    elif score >= 40:
        return 'NEUTRAL'
    else:
        return 'AVOID'


def add_recommendation_labels(
    df: pd.DataFrame,
    score_column: str = 'composite_score'
) -> pd.DataFrame:
    """
    Add recommendation labels to DataFrame based on composite scores.
    
    Args:
        df: DataFrame containing trade data with composite scores
        score_column: Name of the score column (default: 'composite_score')
    
    Returns:
        DataFrame with added 'recommendation' column
    
    Validates: Requirements 14.5
    """
    if df.empty:
        result_df = df.copy()
        result_df['recommendation'] = None
        return result_df
    
    result_df = df.copy()
    
    if score_column in result_df.columns:
        result_df['recommendation'] = result_df[score_column].apply(classify_recommendation)
    else:
        result_df['recommendation'] = 'NEUTRAL'
    
    return result_df


def get_score_statistics(
    df: pd.DataFrame,
    score_column: str = 'composite_score'
) -> Dict[str, float]:
    """
    Calculate summary statistics for composite scores.
    
    Args:
        df: DataFrame containing trade data with composite scores
        score_column: Name of the score column (default: 'composite_score')
    
    Returns:
        Dictionary with statistics:
        - mean_score: Mean composite score
        - median_score: Median composite score
        - std_score: Standard deviation of scores
        - min_score: Minimum score
        - max_score: Maximum score
        - pct_strong_buy: Percentage of STRONG BUY recommendations
        - pct_buy: Percentage of BUY recommendations
        - pct_neutral: Percentage of NEUTRAL recommendations
        - pct_avoid: Percentage of AVOID recommendations
    """
    if df.empty or score_column not in df.columns:
        return {
            'mean_score': np.nan,
            'median_score': np.nan,
            'std_score': np.nan,
            'min_score': np.nan,
            'max_score': np.nan,
            'pct_strong_buy': 0.0,
            'pct_buy': 0.0,
            'pct_neutral': 0.0,
            'pct_avoid': 0.0
        }
    
    scores = df[score_column].dropna()
    
    if len(scores) == 0:
        return {
            'mean_score': np.nan,
            'median_score': np.nan,
            'std_score': np.nan,
            'min_score': np.nan,
            'max_score': np.nan,
            'pct_strong_buy': 0.0,
            'pct_buy': 0.0,
            'pct_neutral': 0.0,
            'pct_avoid': 0.0
        }
    
    # Calculate score statistics
    stats = {
        'mean_score': scores.mean(),
        'median_score': scores.median(),
        'std_score': scores.std(),
        'min_score': scores.min(),
        'max_score': scores.max()
    }
    
    # Calculate recommendation percentages
    total = len(scores)
    stats['pct_strong_buy'] = ((scores >= 80).sum() / total) * 100
    stats['pct_buy'] = (((scores >= 60) & (scores < 80)).sum() / total) * 100
    stats['pct_neutral'] = (((scores >= 40) & (scores < 60)).sum() / total) * 100
    stats['pct_avoid'] = ((scores < 40).sum() / total) * 100
    
    return stats
