"""
Time-Based Probability Analyzer

This module provides functions for calculating win rates and probabilities
based on time-related factors such as hour of day, day of week, trading session,
and news proximity.

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from backend.models.confidence_intervals import beta_posterior_ci


def compute_hourly_winrate(
    df: pd.DataFrame,
    timestamp_column: str = 'Timestamp',
    target_column: str = 'trade_success',
    conf_level: float = 0.95,
    min_samples: int = 5
) -> pd.DataFrame:
    """
    Calculate win rate per hour of day with confidence intervals.
    
    Args:
        df: DataFrame containing trade data
        timestamp_column: Name of the timestamp column
        target_column: Name of the target column (1 for win, 0 for loss)
        conf_level: Confidence level for intervals (default: 0.95)
        min_samples: Minimum samples required per hour (default: 5)
    
    Returns:
        DataFrame with columns:
        - hour: Hour of day (0-23)
        - n_trades: Number of trades in that hour
        - n_wins: Number of winning trades
        - win_rate: Win rate (proportion of wins)
        - ci_lower: Lower bound of confidence interval
        - ci_upper: Upper bound of confidence interval
        - reliable: Boolean indicating if sample size >= min_samples
    
    Validates: Requirements 7.1
    """
    if df.empty:
        return pd.DataFrame(columns=['hour', 'n_trades', 'n_wins', 'win_rate', 
                                     'ci_lower', 'ci_upper', 'reliable'])
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_column]):
        df = df.copy()
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    
    # Extract hour from timestamp
    df_with_hour = df.copy()
    df_with_hour['hour'] = df_with_hour[timestamp_column].dt.hour
    
    # Group by hour
    results = []
    for hour in range(24):
        hour_df = df_with_hour[df_with_hour['hour'] == hour]
        n_trades = len(hour_df)
        
        if n_trades == 0:
            results.append({
                'hour': hour,
                'n_trades': 0,
                'n_wins': 0,
                'win_rate': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'reliable': False
            })
            continue
        
        n_wins = hour_df[target_column].sum()
        win_rate = n_wins / n_trades
        
        # Calculate confidence interval using Beta-Binomial
        ci_result = beta_posterior_ci(
            successes=int(n_wins),
            total=n_trades,
            conf_level=conf_level
        )
        
        results.append({
            'hour': hour,
            'n_trades': n_trades,
            'n_wins': int(n_wins),
            'win_rate': win_rate,
            'ci_lower': ci_result['ci_lower'],
            'ci_upper': ci_result['ci_upper'],
            'reliable': n_trades >= min_samples
        })
    
    result_df = pd.DataFrame(results)
    return result_df


def compute_daily_winrate(
    df: pd.DataFrame,
    timestamp_column: str = 'Timestamp',
    target_column: str = 'trade_success',
    conf_level: float = 0.95,
    min_samples: int = 5
) -> pd.DataFrame:
    """
    Calculate win rate per day of week with confidence intervals.
    
    Args:
        df: DataFrame containing trade data
        timestamp_column: Name of the timestamp column
        target_column: Name of the target column (1 for win, 0 for loss)
        conf_level: Confidence level for intervals (default: 0.95)
        min_samples: Minimum samples required per day (default: 5)
    
    Returns:
        DataFrame with columns:
        - day_of_week: Day of week (0=Monday, 6=Sunday)
        - day_name: Name of the day
        - n_trades: Number of trades on that day
        - n_wins: Number of winning trades
        - win_rate: Win rate (proportion of wins)
        - ci_lower: Lower bound of confidence interval
        - ci_upper: Upper bound of confidence interval
        - reliable: Boolean indicating if sample size >= min_samples
    
    Validates: Requirements 7.2
    """
    if df.empty:
        return pd.DataFrame(columns=['day_of_week', 'day_name', 'n_trades', 'n_wins', 
                                     'win_rate', 'ci_lower', 'ci_upper', 'reliable'])
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_column]):
        df = df.copy()
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    
    # Extract day of week from timestamp
    df_with_day = df.copy()
    df_with_day['day_of_week'] = df_with_day[timestamp_column].dt.dayofweek
    
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Group by day of week
    results = []
    for day in range(7):
        day_df = df_with_day[df_with_day['day_of_week'] == day]
        n_trades = len(day_df)
        
        if n_trades == 0:
            results.append({
                'day_of_week': day,
                'day_name': day_names[day],
                'n_trades': 0,
                'n_wins': 0,
                'win_rate': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'reliable': False
            })
            continue
        
        n_wins = day_df[target_column].sum()
        win_rate = n_wins / n_trades
        
        # Calculate confidence interval using Beta-Binomial
        ci_result = beta_posterior_ci(
            successes=int(n_wins),
            total=n_trades,
            conf_level=conf_level
        )
        
        results.append({
            'day_of_week': day,
            'day_name': day_names[day],
            'n_trades': n_trades,
            'n_wins': int(n_wins),
            'win_rate': win_rate,
            'ci_lower': ci_result['ci_lower'],
            'ci_upper': ci_result['ci_upper'],
            'reliable': n_trades >= min_samples
        })
    
    result_df = pd.DataFrame(results)
    return result_df


def compute_session_winrate(
    df: pd.DataFrame,
    session_column: str = 'session',
    target_column: str = 'trade_success',
    conf_level: float = 0.95,
    min_samples: int = 5,
    session_mapping: Optional[Dict[int, str]] = None
) -> pd.DataFrame:
    """
    Calculate win rate per trading session with confidence intervals.
    
    Args:
        df: DataFrame containing trade data
        session_column: Name of the session column
        target_column: Name of the target column (1 for win, 0 for loss)
        conf_level: Confidence level for intervals (default: 0.95)
        min_samples: Minimum samples required per session (default: 5)
        session_mapping: Optional mapping from session codes to names
                        (default: {0: 'ASIA', 1: 'EUROPE', 2: 'US', 3: 'OVERLAP'})
    
    Returns:
        DataFrame with columns:
        - session_code: Session code (0, 1, 2, 3)
        - session_name: Session name (ASIA, EUROPE, US, OVERLAP)
        - n_trades: Number of trades in that session
        - n_wins: Number of winning trades
        - win_rate: Win rate (proportion of wins)
        - ci_lower: Lower bound of confidence interval
        - ci_upper: Upper bound of confidence interval
        - reliable: Boolean indicating if sample size >= min_samples
    
    Validates: Requirements 7.3
    """
    if session_mapping is None:
        session_mapping = {0: 'ASIA', 1: 'EUROPE', 2: 'US', 3: 'OVERLAP'}
    
    if df.empty:
        return pd.DataFrame(columns=['session_code', 'session_name', 'n_trades', 'n_wins',
                                     'win_rate', 'ci_lower', 'ci_upper', 'reliable'])
    
    # Check if session column exists
    if session_column not in df.columns:
        raise ValueError(f"Session column '{session_column}' not found in DataFrame")
    
    # Group by session
    results = []
    unique_sessions = sorted(df[session_column].dropna().unique())
    
    for session_code in unique_sessions:
        session_df = df[df[session_column] == session_code]
        n_trades = len(session_df)
        
        if n_trades == 0:
            continue
        
        n_wins = session_df[target_column].sum()
        win_rate = n_wins / n_trades
        
        # Calculate confidence interval using Beta-Binomial
        ci_result = beta_posterior_ci(
            successes=int(n_wins),
            total=n_trades,
            conf_level=conf_level
        )
        
        session_name = session_mapping.get(session_code, f'SESSION_{session_code}')
        
        results.append({
            'session_code': session_code,
            'session_name': session_name,
            'n_trades': n_trades,
            'n_wins': int(n_wins),
            'win_rate': win_rate,
            'ci_lower': ci_result['ci_lower'],
            'ci_upper': ci_result['ci_upper'],
            'reliable': n_trades >= min_samples
        })
    
    result_df = pd.DataFrame(results)
    return result_df


def compute_news_proximity_winrate(
    df: pd.DataFrame,
    news_column: str = 'minutes_to_next_high_impact_news',
    target_column: str = 'trade_success',
    conf_level: float = 0.95,
    min_samples: int = 5,
    bins: Optional[List[Tuple[float, float]]] = None
) -> pd.DataFrame:
    """
    Calculate win rate based on proximity to news events.
    
    Args:
        df: DataFrame containing trade data
        news_column: Name of the column with minutes to next news event
        target_column: Name of the target column (1 for win, 0 for loss)
        conf_level: Confidence level for intervals (default: 0.95)
        min_samples: Minimum samples required per bin (default: 5)
        bins: Optional list of (min, max) tuples defining time bins
              (default: [(0, 30), (30, 60), (60, 120), (120, 240), (240, float('inf'))])
    
    Returns:
        DataFrame with columns:
        - bin_label: Label for the time bin
        - min_minutes: Minimum minutes in the bin
        - max_minutes: Maximum minutes in the bin
        - n_trades: Number of trades in that bin
        - n_wins: Number of winning trades
        - win_rate: Win rate (proportion of wins)
        - ci_lower: Lower bound of confidence interval
        - ci_upper: Upper bound of confidence interval
        - reliable: Boolean indicating if sample size >= min_samples
    
    Validates: Requirements 7.4
    """
    if bins is None:
        bins = [(0, 30), (30, 60), (60, 120), (120, 240), (240, float('inf'))]
    
    if df.empty:
        return pd.DataFrame(columns=['bin_label', 'min_minutes', 'max_minutes', 'n_trades',
                                     'n_wins', 'win_rate', 'ci_lower', 'ci_upper', 'reliable'])
    
    # Check if news column exists
    if news_column not in df.columns:
        raise ValueError(f"News column '{news_column}' not found in DataFrame")
    
    results = []
    for min_min, max_min in bins:
        # Create bin label
        if max_min == float('inf'):
            bin_label = f'{int(min_min)}+ min'
        else:
            bin_label = f'{int(min_min)}-{int(max_min)} min'
        
        # Filter trades in this bin
        if max_min == float('inf'):
            bin_df = df[df[news_column] >= min_min]
        else:
            bin_df = df[(df[news_column] >= min_min) & (df[news_column] < max_min)]
        
        n_trades = len(bin_df)
        
        if n_trades == 0:
            results.append({
                'bin_label': bin_label,
                'min_minutes': min_min,
                'max_minutes': max_min,
                'n_trades': 0,
                'n_wins': 0,
                'win_rate': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'reliable': False
            })
            continue
        
        n_wins = bin_df[target_column].sum()
        win_rate = n_wins / n_trades
        
        # Calculate confidence interval using Beta-Binomial
        ci_result = beta_posterior_ci(
            successes=int(n_wins),
            total=n_trades,
            conf_level=conf_level
        )
        
        results.append({
            'bin_label': bin_label,
            'min_minutes': min_min,
            'max_minutes': max_min,
            'n_trades': n_trades,
            'n_wins': int(n_wins),
            'win_rate': win_rate,
            'ci_lower': ci_result['ci_lower'],
            'ci_upper': ci_result['ci_upper'],
            'reliable': n_trades >= min_samples
        })
    
    result_df = pd.DataFrame(results)
    return result_df


def find_optimal_time_windows(
    df: pd.DataFrame,
    timestamp_column: str = 'Timestamp',
    target_column: str = 'trade_success',
    min_win_rate: float = 0.6,
    min_samples: int = 20,
    window_type: str = 'hour'
) -> pd.DataFrame:
    """
    Find optimal trading time windows based on win rate and sample size criteria.
    
    Args:
        df: DataFrame containing trade data
        timestamp_column: Name of the timestamp column
        target_column: Name of the target column (1 for win, 0 for loss)
        min_win_rate: Minimum win rate threshold (default: 0.6)
        min_samples: Minimum sample size threshold (default: 20)
        window_type: Type of window ('hour' or 'day')
    
    Returns:
        DataFrame with optimal time windows that meet both criteria:
        - For 'hour': hour, n_trades, win_rate, ci_lower, ci_upper
        - For 'day': day_of_week, day_name, n_trades, win_rate, ci_lower, ci_upper
        Sorted by win_rate descending
    
    Validates: Requirements 7.5
    """
    if df.empty:
        if window_type == 'hour':
            return pd.DataFrame(columns=['hour', 'n_trades', 'win_rate', 'ci_lower', 'ci_upper'])
        else:
            return pd.DataFrame(columns=['day_of_week', 'day_name', 'n_trades', 
                                        'win_rate', 'ci_lower', 'ci_upper'])
    
    if window_type == 'hour':
        # Get hourly win rates
        hourly_df = compute_hourly_winrate(df, timestamp_column, target_column)
        
        # Filter by criteria
        optimal = hourly_df[
            (hourly_df['win_rate'] >= min_win_rate) &
            (hourly_df['n_trades'] >= min_samples) &
            (hourly_df['reliable'] == True)
        ].copy()
        
        # Sort by win rate descending
        optimal = optimal.sort_values('win_rate', ascending=False).reset_index(drop=True)
        
        # Return only relevant columns
        return optimal[['hour', 'n_trades', 'win_rate', 'ci_lower', 'ci_upper']]
    
    elif window_type == 'day':
        # Get daily win rates
        daily_df = compute_daily_winrate(df, timestamp_column, target_column)
        
        # Filter by criteria
        optimal = daily_df[
            (daily_df['win_rate'] >= min_win_rate) &
            (daily_df['n_trades'] >= min_samples) &
            (daily_df['reliable'] == True)
        ].copy()
        
        # Sort by win rate descending
        optimal = optimal.sort_values('win_rate', ascending=False).reset_index(drop=True)
        
        # Return only relevant columns
        return optimal[['day_of_week', 'day_name', 'n_trades', 'win_rate', 'ci_lower', 'ci_upper']]
    
    else:
        raise ValueError(f"Invalid window_type: {window_type}. Must be 'hour' or 'day'")
