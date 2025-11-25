"""
Data Preprocessor Module

This module handles loading and preprocessing of Feature CSV and Trade CSV files.
It validates column presence, merges datasets, filters valid trades, and creates target columns.
"""

import pandas as pd
from typing import List, Tuple
import os


# Feature CSV required columns (50+ columns)
FEATURE_COLUMNS = [
    'timestamp',
]

# Trade CSV required columns (40+ columns)
TRADE_COLUMNS = [
    'Ticket_id',
    'Symbol',
    'Timestamp',
    'Type',
    'OpenPrice',
    'Volume',
    'Timeframe',
    'UseFibo50Filter',
    'FiboBasePrice',
    'FiboRange',
    'MagicNumber',
    'StrategyType',
    'ConsecutiveSLCount',
    'TPHitsToday',
    'SLHitsToday',
    'SessionHour',
    'SessionMinute',
    'SessionDayOfWeek',
    'MFEPips',
    'MAEPips',
    'ClosePrice',
    'ExitReason',
    'MaxSLTP',
    'entry_time',
    'exit_time',
    'entry_session',
    'entry_price',
    'sl_price',
    'tp_price',
    'sl_distance',
    'money_risk',
    'risk_percent',
    'gross_profit',
    'net_profit',
    'holding_bars',
    'holding_minutes',
    'R_multiple',
    'trade_success',
    'MAE_R',
    'MFE_R',
    'max_drawdown_k',
    'max_runup_k',
    'K_bars',
    'future_return_k',
    'equity_at_entry',
    'equity_after_trade',
]


def load_feature_csv(path: str, optimize_memory: bool = True) -> pd.DataFrame:
    """
    Load Feature CSV file with validation and optimization.
    
    Args:
        path: Path to Feature CSV file
        optimize_memory: Whether to optimize memory usage (default: True)
        
    Returns:
        DataFrame with feature data
        
    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If required columns are missing
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature CSV file not found: {path}")
    
    try:
        from backend.utils.performance_optimizer import optimize_csv_loading
        df = optimize_csv_loading(path, sep=';')
    except Exception:
        df = None
    
    # Fallback: try comma separator using pandas if needed
    if df is None or ('timestamp' not in df.columns and optimize_memory):
        try:
            df = pd.read_csv(path, sep=',')
        except Exception:
            # Last resort: try semicolon with pandas
            df = pd.read_csv(path, sep=';')
    
    # Minimal validation: require timestamp column only
    ts_candidates = ['timestamp', 'Timestamp', 'time', 'datetime']
    if not any(col in df.columns for col in ts_candidates):
        raise ValueError("Feature CSV missing 'timestamp' column (or equivalent)")
    
    # Normalize to 'timestamp'
    if 'timestamp' not in df.columns:
        for c in ts_candidates:
            if c in df.columns:
                df = df.rename(columns={c: 'timestamp'})
                break
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df


def load_trade_csv(path: str, optimize_memory: bool = True) -> pd.DataFrame:
    """
    Load Trade CSV file with validation and optimization.
    
    Args:
        path: Path to Trade CSV file
        optimize_memory: Whether to optimize memory usage (default: True)
        
    Returns:
        DataFrame with trade data
        
    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If required columns are missing
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Trade CSV file not found: {path}")
    
    try:
        # Use optimized loading for large files
        from backend.utils.performance_optimizer import optimize_csv_loading
        df = optimize_csv_loading(path, sep='\t')
    except Exception as e:
        raise ValueError(f"Failed to read Trade CSV with separator '\\t': {e}")
    
    # Validate required columns
    missing_cols = [col for col in TRADE_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Trade CSV missing {len(missing_cols)} required columns: {missing_cols[:10]}..."
            if len(missing_cols) > 10 else
            f"Trade CSV missing required columns: {missing_cols}"
        )
    
    # Convert timestamp columns to datetime
    datetime_cols = ['Timestamp', 'entry_time', 'exit_time']
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    
    return df


def merge_datasets(features_df: pd.DataFrame, trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge feature and trade datasets on timestamp using left join.
    
    Args:
        features_df: DataFrame with feature data
        trades_df: DataFrame with trade data
        
    Returns:
        Merged DataFrame preserving all trade records
    """
    # Ensure timestamp columns exist
    if 'timestamp' not in features_df.columns:
        raise ValueError("Feature DataFrame missing 'timestamp' column")
    
    # Use Timestamp or entry_time from trades
    trade_time_col = 'Timestamp' if 'Timestamp' in trades_df.columns else 'entry_time'
    if trade_time_col not in trades_df.columns:
        raise ValueError(f"Trade DataFrame missing '{trade_time_col}' column")
    
    # Perform left join to preserve all trade records
    merged_df = trades_df.merge(
        features_df,
        left_on=trade_time_col,
        right_on='timestamp',
        how='left'
    )
    
    return merged_df


def filter_valid_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter dataset to keep only valid trades.
    
    Valid trades have:
    - Volume > 0
    - trade_success in {0, 1}
    
    Args:
        df: DataFrame with merged data
        
    Returns:
        Filtered DataFrame with only valid trades
    """
    if 'Volume' not in df.columns:
        raise ValueError("DataFrame missing 'Volume' column")
    if 'trade_success' not in df.columns:
        raise ValueError("DataFrame missing 'trade_success' column")
    
    # Filter Volume > 0
    valid_df = df[df['Volume'] > 0].copy()
    
    # Filter trade_success in {0, 1}
    valid_df = valid_df[valid_df['trade_success'].isin([0, 1])]
    
    return valid_df


def create_target_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create target columns for analysis.
    
    Creates:
    - y_win: 1 if trade_success == 1, else 0
    - y_hit_1R: 1 if R_multiple >= 1, else 0
    - y_hit_2R: 1 if R_multiple >= 2, else 0
    - y_future_win_k: 1 if future_return_k > 0, else 0
    
    Args:
        df: DataFrame with trade data
        
    Returns:
        DataFrame with added target columns
    """
    df = df.copy()
    
    # y_win: based on trade_success
    if 'trade_success' in df.columns:
        df['y_win'] = df['trade_success'].astype(int)
    else:
        raise ValueError("DataFrame missing 'trade_success' column")
    
    # y_hit_1R: R_multiple >= 1
    if 'R_multiple' in df.columns:
        df['y_hit_1R'] = (df['R_multiple'] >= 1).astype(int)
    else:
        raise ValueError("DataFrame missing 'R_multiple' column")
    
    # y_hit_2R: R_multiple >= 2
    df['y_hit_2R'] = (df['R_multiple'] >= 2).astype(int)
    
    # y_future_win_k: future_return_k > 0
    if 'future_return_k' in df.columns:
        df['y_future_win_k'] = (df['future_return_k'] > 0).astype(int)
    else:
        raise ValueError("DataFrame missing 'future_return_k' column")
    
    return df


def preprocess_data(feature_path: str, trade_path: str) -> pd.DataFrame:
    """
    Complete preprocessing pipeline.
    
    Args:
        feature_path: Path to Feature CSV
        trade_path: Path to Trade CSV
        
    Returns:
        Preprocessed DataFrame ready for analysis
    """
    # Load data
    features_df = load_feature_csv(feature_path)
    trades_df = load_trade_csv(trade_path)
    
    # Merge datasets
    merged_df = merge_datasets(features_df, trades_df)
    
    # Filter valid trades
    valid_df = filter_valid_trades(merged_df)
    
    # Create target columns
    final_df = create_target_columns(valid_df)
    
    return final_df
