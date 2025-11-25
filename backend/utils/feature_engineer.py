"""
Probability Feature Engineering Module

This module builds 18 probability features using target encoding based on historical data.
These features represent conditional probabilities that can be used for ML or EA systems.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


# List of 18 probability features to be created
PROBABILITY_FEATURES = [
    'prob_global_win',              # Global win rate
    'prob_global_hit_1R',           # Global P(R >= 1)
    'prob_session_win',             # Win rate per session
    'prob_session_hit_1R',          # P(R >= 1) per session
    'prob_trend_strength_win',      # Win rate per trend strength bin
    'prob_trend_strength_hit_1R',   # P(R >= 1) per trend strength bin
    'prob_trend_dir_alignment_win', # Win rate when trend direction aligns
    'prob_vol_regime_win',          # Win rate per volatility regime
    'prob_vol_regime_hit_1R',       # P(R >= 1) per volatility regime
    'prob_sr_zone_win',             # Win rate near support/resistance
    'prob_sr_zone_hit_1R',          # P(R >= 1) near support/resistance
    'prob_entropy_win',             # Win rate per entropy bin
    'prob_hurst_win',               # Win rate per Hurst exponent bin
    'prob_regime_cluster_win',      # Win rate per regime cluster
    'prob_streak_loss_win',         # Win rate given loss streak
    'prob_dd_state_win',            # Win rate given drawdown state
    'prob_trend_vol_cross_win',     # 2D: trend × volatility
    'prob_session_sr_cross_win',    # 2D: session × SR zone
]


def build_probability_features(df: pd.DataFrame, 
                               n_bins: int = 10,
                               min_samples: int = 5) -> pd.DataFrame:
    """
    Build all 18 probability features using target encoding.
    
    This function calculates conditional probabilities for various market conditions
    and adds them as new columns to the dataset.
    
    Args:
        df: DataFrame with merged feature and trade data
        n_bins: Number of bins for continuous features
        min_samples: Minimum samples required for reliable probability estimate
        
    Returns:
        DataFrame with 18 additional probability feature columns
        
    Raises:
        ValueError: If required columns are missing
    """
    df = df.copy()
    
    # Validate required columns
    required_cols = ['y_win', 'y_hit_1R']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required target columns: {missing}")
    
    # 1. Global features (constant across all rows)
    df = _add_global_features(df)
    
    # 2. Session features
    df = _add_session_features(df, min_samples)
    
    # 3. Trend features
    df = _add_trend_features(df, n_bins, min_samples)
    
    # 4. Volatility regime features
    df = _add_volatility_features(df, min_samples)
    
    # 5. Support/Resistance zone features
    df = _add_sr_zone_features(df, n_bins, min_samples)
    
    # 6. Entropy, Hurst, and Regime cluster features
    df = _add_structure_features(df, n_bins, min_samples)
    
    # 7. Behavioral features (streak, drawdown)
    df = _add_behavioral_features(df, min_samples)
    
    # 8. Cross features (2D probabilities)
    df = _add_cross_features(df, n_bins, min_samples)
    
    # Remove intermediate columns
    intermediate_cols = [
        'trend_strength_bin', 'trend_aligned', 'sr_zone_bin', 'dist_to_sr',
        'entropy_bin', 'hurst_bin', 'regime_cluster', 'streak_loss_bin',
        'dd_state', 'trend_vol_cross', 'session_sr_cross'
    ]
    df = df.drop(columns=[col for col in intermediate_cols if col in df.columns])
    
    return df


def _add_global_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add global probability features (constant across all rows).
    
    Features:
    - prob_global_win: Overall win rate
    - prob_global_hit_1R: Overall P(R >= 1)
    """
    # Calculate global probabilities
    global_win_rate = df['y_win'].mean()
    global_hit_1r = df['y_hit_1R'].mean()
    
    # Assign to all rows
    df['prob_global_win'] = global_win_rate
    df['prob_global_hit_1R'] = global_hit_1r
    
    return df


def _add_session_features(df: pd.DataFrame, min_samples: int) -> pd.DataFrame:
    """
    Add session-based probability features.
    
    Features:
    - prob_session_win: Win rate per trading session
    - prob_session_hit_1R: P(R >= 1) per trading session
    """
    if 'session' not in df.columns:
        # If session column missing, use global values
        df['prob_session_win'] = df['prob_global_win']
        df['prob_session_hit_1R'] = df['prob_global_hit_1R']
        return df
    
    # Calculate probabilities per session
    session_probs = df.groupby('session').agg({
        'y_win': ['mean', 'count'],
        'y_hit_1R': 'mean'
    })
    
    # Flatten column names
    session_probs.columns = ['win_rate', 'count', 'hit_1r']
    
    # Filter sessions with insufficient samples
    session_probs.loc[session_probs['count'] < min_samples, ['win_rate', 'hit_1r']] = np.nan
    
    # Map back to original dataframe
    df['prob_session_win'] = df['session'].map(session_probs['win_rate'])
    df['prob_session_hit_1R'] = df['session'].map(session_probs['hit_1r'])
    
    # Fill NaN with global values
    df['prob_session_win'] = df['prob_session_win'].fillna(df['prob_global_win'])
    df['prob_session_hit_1R'] = df['prob_session_hit_1R'].fillna(df['prob_global_hit_1R'])
    
    return df


def _add_trend_features(df: pd.DataFrame, n_bins: int, min_samples: int) -> pd.DataFrame:
    """
    Add trend-based probability features.
    
    Features:
    - prob_trend_strength_win: Win rate per trend strength bin
    - prob_trend_strength_hit_1R: P(R >= 1) per trend strength bin
    - prob_trend_dir_alignment_win: Win rate when trend direction aligns with trade
    """
    # Trend strength features
    if 'trend_strength_tf' in df.columns:
        df['trend_strength_bin'] = pd.qcut(df['trend_strength_tf'], q=n_bins, 
                                           labels=False, duplicates='drop')
        
        strength_probs = df.groupby('trend_strength_bin').agg({
            'y_win': ['mean', 'count'],
            'y_hit_1R': 'mean'
        })
        strength_probs.columns = ['win_rate', 'count', 'hit_1r']
        strength_probs.loc[strength_probs['count'] < min_samples, ['win_rate', 'hit_1r']] = np.nan
        
        df['prob_trend_strength_win'] = df['trend_strength_bin'].map(strength_probs['win_rate'])
        df['prob_trend_strength_hit_1R'] = df['trend_strength_bin'].map(strength_probs['hit_1r'])
        
        df['prob_trend_strength_win'] = df['prob_trend_strength_win'].fillna(df['prob_global_win'])
        df['prob_trend_strength_hit_1R'] = df['prob_trend_strength_hit_1R'].fillna(df['prob_global_hit_1R'])
    else:
        df['prob_trend_strength_win'] = df['prob_global_win']
        df['prob_trend_strength_hit_1R'] = df['prob_global_hit_1R']
    
    # Trend direction alignment
    if 'trend_tf_dir' in df.columns and 'Type' in df.columns:
        # Create alignment indicator: 1 if trend direction matches trade type
        df['trend_aligned'] = 0
        df.loc[(df['trend_tf_dir'] == 1) & (df['Type'] == 'BUY'), 'trend_aligned'] = 1
        df.loc[(df['trend_tf_dir'] == -1) & (df['Type'] == 'SELL'), 'trend_aligned'] = 1
        
        alignment_probs = df.groupby('trend_aligned').agg({
            'y_win': ['mean', 'count']
        })
        alignment_probs.columns = ['win_rate', 'count']
        alignment_probs.loc[alignment_probs['count'] < min_samples, 'win_rate'] = np.nan
        
        df['prob_trend_dir_alignment_win'] = df['trend_aligned'].map(alignment_probs['win_rate'])
        df['prob_trend_dir_alignment_win'] = df['prob_trend_dir_alignment_win'].fillna(df['prob_global_win'])
    else:
        df['prob_trend_dir_alignment_win'] = df['prob_global_win']
    
    return df


def _add_volatility_features(df: pd.DataFrame, min_samples: int) -> pd.DataFrame:
    """
    Add volatility regime probability features.
    
    Features:
    - prob_vol_regime_win: Win rate per volatility regime
    - prob_vol_regime_hit_1R: P(R >= 1) per volatility regime
    """
    if 'volatility_regime' not in df.columns:
        df['prob_vol_regime_win'] = df['prob_global_win']
        df['prob_vol_regime_hit_1R'] = df['prob_global_hit_1R']
        return df
    
    vol_probs = df.groupby('volatility_regime').agg({
        'y_win': ['mean', 'count'],
        'y_hit_1R': 'mean'
    })
    vol_probs.columns = ['win_rate', 'count', 'hit_1r']
    vol_probs.loc[vol_probs['count'] < min_samples, ['win_rate', 'hit_1r']] = np.nan
    
    df['prob_vol_regime_win'] = df['volatility_regime'].map(vol_probs['win_rate'])
    df['prob_vol_regime_hit_1R'] = df['volatility_regime'].map(vol_probs['hit_1r'])
    
    df['prob_vol_regime_win'] = df['prob_vol_regime_win'].fillna(df['prob_global_win'])
    df['prob_vol_regime_hit_1R'] = df['prob_vol_regime_hit_1R'].fillna(df['prob_global_hit_1R'])
    
    return df


def _add_sr_zone_features(df: pd.DataFrame, n_bins: int, min_samples: int) -> pd.DataFrame:
    """
    Add support/resistance zone probability features.
    
    Features:
    - prob_sr_zone_win: Win rate based on distance to SR levels
    - prob_sr_zone_hit_1R: P(R >= 1) based on distance to SR levels
    """
    # Use distance to day high/low as proxy for SR zones
    if 'dist_to_day_high_pips' in df.columns and 'dist_to_day_low_pips' in df.columns:
        # Calculate minimum distance to nearest SR level
        df['dist_to_sr'] = df[['dist_to_day_high_pips', 'dist_to_day_low_pips']].abs().min(axis=1)
        
        df['sr_zone_bin'] = pd.qcut(df['dist_to_sr'], q=n_bins, 
                                    labels=False, duplicates='drop')
        
        sr_probs = df.groupby('sr_zone_bin').agg({
            'y_win': ['mean', 'count'],
            'y_hit_1R': 'mean'
        })
        sr_probs.columns = ['win_rate', 'count', 'hit_1r']
        sr_probs.loc[sr_probs['count'] < min_samples, ['win_rate', 'hit_1r']] = np.nan
        
        df['prob_sr_zone_win'] = df['sr_zone_bin'].map(sr_probs['win_rate'])
        df['prob_sr_zone_hit_1R'] = df['sr_zone_bin'].map(sr_probs['hit_1r'])
        
        df['prob_sr_zone_win'] = df['prob_sr_zone_win'].fillna(df['prob_global_win'])
        df['prob_sr_zone_hit_1R'] = df['prob_sr_zone_hit_1R'].fillna(df['prob_global_hit_1R'])
    else:
        df['prob_sr_zone_win'] = df['prob_global_win']
        df['prob_sr_zone_hit_1R'] = df['prob_global_hit_1R']
    
    return df


def _add_structure_features(df: pd.DataFrame, n_bins: int, min_samples: int) -> pd.DataFrame:
    """
    Add market structure probability features.
    
    Features:
    - prob_entropy_win: Win rate per entropy bin
    - prob_hurst_win: Win rate per Hurst exponent bin
    - prob_regime_cluster_win: Win rate per regime cluster
    """
    # Entropy feature
    if 'ap_entropy_m1_2h' in df.columns:
        df['entropy_bin'] = pd.qcut(df['ap_entropy_m1_2h'], q=n_bins, 
                                    labels=False, duplicates='drop')
        
        entropy_probs = df.groupby('entropy_bin').agg({
            'y_win': ['mean', 'count']
        })
        entropy_probs.columns = ['win_rate', 'count']
        entropy_probs.loc[entropy_probs['count'] < min_samples, 'win_rate'] = np.nan
        
        df['prob_entropy_win'] = df['entropy_bin'].map(entropy_probs['win_rate'])
        df['prob_entropy_win'] = df['prob_entropy_win'].fillna(df['prob_global_win'])
    else:
        df['prob_entropy_win'] = df['prob_global_win']
    
    # Hurst exponent feature
    if 'hurst_m5_2d' in df.columns:
        df['hurst_bin'] = pd.qcut(df['hurst_m5_2d'], q=n_bins, 
                                  labels=False, duplicates='drop')
        
        hurst_probs = df.groupby('hurst_bin').agg({
            'y_win': ['mean', 'count']
        })
        hurst_probs.columns = ['win_rate', 'count']
        hurst_probs.loc[hurst_probs['count'] < min_samples, 'win_rate'] = np.nan
        
        df['prob_hurst_win'] = df['hurst_bin'].map(hurst_probs['win_rate'])
        df['prob_hurst_win'] = df['prob_hurst_win'].fillna(df['prob_global_win'])
    else:
        df['prob_hurst_win'] = df['prob_global_win']
    
    # Regime cluster (combination of trend_regime and volatility_regime)
    if 'trend_regime' in df.columns and 'volatility_regime' in df.columns:
        df['regime_cluster'] = (df['trend_regime'].astype(str) + '_' + 
                               df['volatility_regime'].astype(str))
        
        cluster_probs = df.groupby('regime_cluster').agg({
            'y_win': ['mean', 'count']
        })
        cluster_probs.columns = ['win_rate', 'count']
        cluster_probs.loc[cluster_probs['count'] < min_samples, 'win_rate'] = np.nan
        
        df['prob_regime_cluster_win'] = df['regime_cluster'].map(cluster_probs['win_rate'])
        df['prob_regime_cluster_win'] = df['prob_regime_cluster_win'].fillna(df['prob_global_win'])
    else:
        df['prob_regime_cluster_win'] = df['prob_global_win']
    
    return df


def _add_behavioral_features(df: pd.DataFrame, min_samples: int) -> pd.DataFrame:
    """
    Add behavioral probability features.
    
    Features:
    - prob_streak_loss_win: Win rate given current loss streak
    - prob_dd_state_win: Win rate given current drawdown state
    """
    # Loss streak feature
    if 'streak_loss' in df.columns:
        # Bin loss streaks (0, 1, 2, 3+)
        df['streak_loss_bin'] = df['streak_loss'].clip(upper=3)
        
        streak_probs = df.groupby('streak_loss_bin').agg({
            'y_win': ['mean', 'count']
        })
        streak_probs.columns = ['win_rate', 'count']
        streak_probs.loc[streak_probs['count'] < min_samples, 'win_rate'] = np.nan
        
        df['prob_streak_loss_win'] = df['streak_loss_bin'].map(streak_probs['win_rate'])
        df['prob_streak_loss_win'] = df['prob_streak_loss_win'].fillna(df['prob_global_win'])
    else:
        df['prob_streak_loss_win'] = df['prob_global_win']
    
    # Drawdown state feature
    if 'current_drawdown_from_equity_high' in df.columns:
        # Bin drawdown into states (0-5%, 5-10%, 10-15%, 15%+)
        df['dd_state'] = pd.cut(df['current_drawdown_from_equity_high'].abs(), 
                               bins=[0, 5, 10, 15, 100],
                               labels=[0, 1, 2, 3])
        
        dd_probs = df.groupby('dd_state', observed=True).agg({
            'y_win': ['mean', 'count']
        })
        dd_probs.columns = ['win_rate', 'count']
        dd_probs.loc[dd_probs['count'] < min_samples, 'win_rate'] = np.nan
        
        df['prob_dd_state_win'] = df['dd_state'].map(dd_probs['win_rate']).astype(float)
        df['prob_dd_state_win'] = df['prob_dd_state_win'].fillna(df['prob_global_win'])
    else:
        df['prob_dd_state_win'] = df['prob_global_win']
    
    return df


def _add_cross_features(df: pd.DataFrame, n_bins: int, min_samples: int) -> pd.DataFrame:
    """
    Add 2D cross-feature probabilities.
    
    Features:
    - prob_trend_vol_cross_win: Win rate for trend × volatility combinations
    - prob_session_sr_cross_win: Win rate for session × SR zone combinations
    """
    # Trend × Volatility cross feature
    if 'trend_regime' in df.columns and 'volatility_regime' in df.columns:
        df['trend_vol_cross'] = (df['trend_regime'].astype(str) + '_' + 
                                df['volatility_regime'].astype(str))
        
        cross_probs = df.groupby('trend_vol_cross').agg({
            'y_win': ['mean', 'count']
        })
        cross_probs.columns = ['win_rate', 'count']
        cross_probs.loc[cross_probs['count'] < min_samples, 'win_rate'] = np.nan
        
        df['prob_trend_vol_cross_win'] = df['trend_vol_cross'].map(cross_probs['win_rate'])
        df['prob_trend_vol_cross_win'] = df['prob_trend_vol_cross_win'].fillna(df['prob_global_win'])
    else:
        df['prob_trend_vol_cross_win'] = df['prob_global_win']
    
    # Session × SR zone cross feature
    if 'session' in df.columns and 'sr_zone_bin' in df.columns:
        df['session_sr_cross'] = (df['session'].astype(str) + '_' + 
                                 df['sr_zone_bin'].astype(str))
        
        cross_probs = df.groupby('session_sr_cross').agg({
            'y_win': ['mean', 'count']
        })
        cross_probs.columns = ['win_rate', 'count']
        cross_probs.loc[cross_probs['count'] < min_samples, 'win_rate'] = np.nan
        
        df['prob_session_sr_cross_win'] = df['session_sr_cross'].map(cross_probs['win_rate'])
        df['prob_session_sr_cross_win'] = df['prob_session_sr_cross_win'].fillna(df['prob_global_win'])
    else:
        df['prob_session_sr_cross_win'] = df['prob_global_win']
    
    return df


def export_to_csv(df: pd.DataFrame, output_path: str) -> None:
    """
    Export dataset with probability features to CSV.
    
    Args:
        df: DataFrame with probability features
        output_path: Path to save CSV file
        
    Raises:
        ValueError: If probability features are missing
    """
    # Validate that probability features exist
    missing_features = [feat for feat in PROBABILITY_FEATURES if feat not in df.columns]
    if missing_features:
        raise ValueError(f"Missing probability features: {missing_features}")
    
    # Export to CSV
    df.to_csv(output_path, index=False)
    print(f"Dataset with {len(PROBABILITY_FEATURES)} probability features exported to: {output_path}")


def get_probability_feature_summary(df: pd.DataFrame) -> Dict:
    """
    Get summary statistics for probability features.
    
    Args:
        df: DataFrame with probability features
        
    Returns:
        Dictionary with summary statistics for each probability feature
    """
    summary = {}
    
    for feature in PROBABILITY_FEATURES:
        if feature in df.columns:
            summary[feature] = {
                'mean': df[feature].mean(),
                'std': df[feature].std(),
                'min': df[feature].min(),
                'max': df[feature].max(),
                'null_count': df[feature].isnull().sum()
            }
    
    return summary
