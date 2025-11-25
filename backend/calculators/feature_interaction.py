"""
Feature Interaction Analyzer Module

This module provides functions for analyzing interaction effects between features
and identifying synergistic or interfering feature combinations.

Requirements: 17.1, 17.2, 17.3, 17.5
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from itertools import combinations


def calculate_interaction_effect(
    df: pd.DataFrame,
    feature_a: str,
    feature_b: str,
    target_column: str = 'trade_success',
    bins_a: int = 3,
    bins_b: int = 3
) -> Dict:
    """
    Calculate interaction effect between two features.
    
    Interaction effect is computed as:
    effect(A,B) - effect(A) - effect(B)
    
    Where effect is measured as the difference in win rate from baseline.
    
    Args:
        df: DataFrame containing trade data
        feature_a: Name of first feature
        feature_b: Name of second feature
        target_column: Name of the target column (1 for win, 0 for loss)
        bins_a: Number of bins for feature A (default: 3)
        bins_b: Number of bins for feature B (default: 3)
    
    Returns:
        Dictionary containing:
        - interaction_effect: The interaction effect value
        - main_effect_a: Main effect of feature A
        - main_effect_b: Main effect of feature B
        - joint_effect: Joint effect of A and B together
        - baseline_rate: Overall win rate
        - n_samples: Number of samples used
    
    Validates: Requirements 17.1
    """
    if df.empty:
        return {
            'interaction_effect': np.nan,
            'main_effect_a': np.nan,
            'main_effect_b': np.nan,
            'joint_effect': np.nan,
            'baseline_rate': np.nan,
            'n_samples': 0
        }
    
    # Check if columns exist
    if feature_a not in df.columns:
        raise ValueError(f"Feature '{feature_a}' not found in DataFrame")
    if feature_b not in df.columns:
        raise ValueError(f"Feature '{feature_b}' not found in DataFrame")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    
    # Calculate baseline win rate
    baseline_rate = df[target_column].mean()
    
    # Bin the features if they are numeric
    df_copy = df.copy()
    
    # Bin feature A
    if pd.api.types.is_numeric_dtype(df_copy[feature_a]):
        df_copy[f'{feature_a}_binned'] = pd.qcut(
            df_copy[feature_a], 
            q=bins_a, 
            labels=False, 
            duplicates='drop'
        )
    else:
        df_copy[f'{feature_a}_binned'] = df_copy[feature_a]
    
    # Bin feature B
    if pd.api.types.is_numeric_dtype(df_copy[feature_b]):
        df_copy[f'{feature_b}_binned'] = pd.qcut(
            df_copy[feature_b], 
            q=bins_b, 
            labels=False, 
            duplicates='drop'
        )
    else:
        df_copy[f'{feature_b}_binned'] = df_copy[feature_b]
    
    # Remove rows with NaN in binned columns
    df_copy = df_copy.dropna(subset=[f'{feature_a}_binned', f'{feature_b}_binned'])
    
    if df_copy.empty:
        return {
            'interaction_effect': np.nan,
            'main_effect_a': np.nan,
            'main_effect_b': np.nan,
            'joint_effect': np.nan,
            'baseline_rate': baseline_rate,
            'n_samples': 0
        }
    
    # Calculate main effect of A (average effect across all levels of B)
    main_effect_a_values = []
    for a_val in df_copy[f'{feature_a}_binned'].unique():
        subset = df_copy[df_copy[f'{feature_a}_binned'] == a_val]
        if len(subset) > 0:
            effect = subset[target_column].mean() - baseline_rate
            main_effect_a_values.append(effect)
    main_effect_a = np.mean(main_effect_a_values) if main_effect_a_values else 0.0
    
    # Calculate main effect of B (average effect across all levels of A)
    main_effect_b_values = []
    for b_val in df_copy[f'{feature_b}_binned'].unique():
        subset = df_copy[df_copy[f'{feature_b}_binned'] == b_val]
        if len(subset) > 0:
            effect = subset[target_column].mean() - baseline_rate
            main_effect_b_values.append(effect)
    main_effect_b = np.mean(main_effect_b_values) if main_effect_b_values else 0.0
    
    # Calculate joint effect (best combination of A and B)
    joint_effects = []
    for a_val in df_copy[f'{feature_a}_binned'].unique():
        for b_val in df_copy[f'{feature_b}_binned'].unique():
            subset = df_copy[
                (df_copy[f'{feature_a}_binned'] == a_val) & 
                (df_copy[f'{feature_b}_binned'] == b_val)
            ]
            if len(subset) > 0:
                effect = subset[target_column].mean() - baseline_rate
                joint_effects.append(effect)
    
    joint_effect = np.mean(joint_effects) if joint_effects else 0.0
    
    # Calculate interaction effect
    interaction_effect = joint_effect - main_effect_a - main_effect_b
    
    return {
        'interaction_effect': interaction_effect,
        'main_effect_a': main_effect_a,
        'main_effect_b': main_effect_b,
        'joint_effect': joint_effect,
        'baseline_rate': baseline_rate,
        'n_samples': len(df_copy)
    }


def decompose_effects(
    df: pd.DataFrame,
    feature_a: str,
    feature_b: str,
    target_column: str = 'trade_success',
    bins_a: int = 3,
    bins_b: int = 3
) -> Dict:
    """
    Decompose effects into main effects and interaction effect.
    
    Args:
        df: DataFrame containing trade data
        feature_a: Name of first feature
        feature_b: Name of second feature
        target_column: Name of the target column (1 for win, 0 for loss)
        bins_a: Number of bins for feature A (default: 3)
        bins_b: Number of bins for feature B (default: 3)
    
    Returns:
        Dictionary containing:
        - main_effect_A: Main effect of feature A
        - main_effect_B: Main effect of feature B
        - interaction_effect: Interaction effect between A and B
        - total_effect: Sum of all effects
        - baseline_rate: Overall win rate
        - feature_a: Name of feature A
        - feature_b: Name of feature B
    
    Validates: Requirements 17.2
    """
    result = calculate_interaction_effect(
        df=df,
        feature_a=feature_a,
        feature_b=feature_b,
        target_column=target_column,
        bins_a=bins_a,
        bins_b=bins_b
    )
    
    return {
        'main_effect_A': result['main_effect_a'],
        'main_effect_B': result['main_effect_b'],
        'interaction_effect': result['interaction_effect'],
        'total_effect': result['joint_effect'],
        'baseline_rate': result['baseline_rate'],
        'feature_a': feature_a,
        'feature_b': feature_b,
        'n_samples': result['n_samples']
    }


def find_top_interactions(
    df: pd.DataFrame,
    features: List[str],
    target_column: str = 'trade_success',
    top_n: int = 10,
    bins: int = 3,
    min_samples: int = 30
) -> pd.DataFrame:
    """
    Find top feature interactions ranked by interaction strength.
    
    Tests all pairwise combinations of features and ranks them by
    absolute interaction effect.
    
    Args:
        df: DataFrame containing trade data
        features: List of feature names to test
        target_column: Name of the target column (1 for win, 0 for loss)
        top_n: Number of top interactions to return (default: 10)
        bins: Number of bins for numeric features (default: 3)
        min_samples: Minimum samples required (default: 30)
    
    Returns:
        DataFrame with columns:
        - feature_a: First feature name
        - feature_b: Second feature name
        - interaction_effect: Interaction effect value
        - abs_interaction: Absolute interaction effect (for ranking)
        - main_effect_a: Main effect of feature A
        - main_effect_b: Main effect of feature B
        - interaction_type: 'synergistic' or 'interfering'
        - n_samples: Number of samples
        
        Sorted by abs_interaction descending.
    
    Validates: Requirements 17.3
    """
    if df.empty or len(features) < 2:
        return pd.DataFrame(columns=[
            'feature_a', 'feature_b', 'interaction_effect', 'abs_interaction',
            'main_effect_a', 'main_effect_b', 'interaction_type', 'n_samples'
        ])
    
    # Filter to only include features that exist in the DataFrame
    valid_features = [f for f in features if f in df.columns]
    
    if len(valid_features) < 2:
        return pd.DataFrame(columns=[
            'feature_a', 'feature_b', 'interaction_effect', 'abs_interaction',
            'main_effect_a', 'main_effect_b', 'interaction_type', 'n_samples'
        ])
    
    results = []
    
    # Test all pairwise combinations
    for feature_a, feature_b in combinations(valid_features, 2):
        try:
            result = calculate_interaction_effect(
                df=df,
                feature_a=feature_a,
                feature_b=feature_b,
                target_column=target_column,
                bins_a=bins,
                bins_b=bins
            )
            
            # Only include if we have enough samples
            if result['n_samples'] >= min_samples:
                interaction_type = classify_interaction(result['interaction_effect'])
                
                results.append({
                    'feature_a': feature_a,
                    'feature_b': feature_b,
                    'interaction_effect': result['interaction_effect'],
                    'abs_interaction': abs(result['interaction_effect']),
                    'main_effect_a': result['main_effect_a'],
                    'main_effect_b': result['main_effect_b'],
                    'interaction_type': interaction_type,
                    'n_samples': result['n_samples']
                })
        except Exception:
            # Skip pairs that cause errors (e.g., insufficient data)
            continue
    
    if not results:
        return pd.DataFrame(columns=[
            'feature_a', 'feature_b', 'interaction_effect', 'abs_interaction',
            'main_effect_a', 'main_effect_b', 'interaction_type', 'n_samples'
        ])
    
    # Create DataFrame and sort by absolute interaction strength
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('abs_interaction', ascending=False)
    
    # Return top N
    return result_df.head(top_n).reset_index(drop=True)


def classify_interaction(interaction_effect: float) -> str:
    """
    Classify interaction as synergistic or interfering.
    
    Args:
        interaction_effect: The interaction effect value
    
    Returns:
        'synergistic' if effect > 0, 'interfering' if effect < 0, 'neutral' if effect == 0
    
    Validates: Requirements 17.5
    """
    if np.isnan(interaction_effect):
        return 'unknown'
    elif interaction_effect > 0:
        return 'synergistic'
    elif interaction_effect < 0:
        return 'interfering'
    else:
        return 'neutral'


def create_interaction_matrix(
    df: pd.DataFrame,
    features: List[str],
    target_column: str = 'trade_success',
    bins: int = 3
) -> pd.DataFrame:
    """
    Create a matrix of interaction effects for all feature pairs.
    
    The diagonal contains main effects, off-diagonal contains interaction effects.
    
    Args:
        df: DataFrame containing trade data
        features: List of feature names to analyze
        target_column: Name of the target column (1 for win, 0 for loss)
        bins: Number of bins for numeric features (default: 3)
    
    Returns:
        DataFrame with features as both rows and columns, containing:
        - Diagonal: Main effects
        - Off-diagonal: Interaction effects
    
    Validates: Requirements 17.4 (mentioned in requirements but not in task)
    """
    if df.empty or len(features) < 2:
        return pd.DataFrame()
    
    # Filter to only include features that exist in the DataFrame
    valid_features = [f for f in features if f in df.columns]
    
    if len(valid_features) < 2:
        return pd.DataFrame()
    
    # Initialize matrix
    matrix = pd.DataFrame(
        np.zeros((len(valid_features), len(valid_features))),
        index=valid_features,
        columns=valid_features
    )
    
    # Calculate baseline
    baseline_rate = df[target_column].mean()
    
    # Fill diagonal with main effects
    for i, feature in enumerate(valid_features):
        try:
            # Calculate main effect
            if pd.api.types.is_numeric_dtype(df[feature]):
                df_copy = df.copy()
                df_copy[f'{feature}_binned'] = pd.qcut(
                    df_copy[feature], 
                    q=bins, 
                    labels=False, 
                    duplicates='drop'
                )
                effects = []
                for val in df_copy[f'{feature}_binned'].dropna().unique():
                    subset = df_copy[df_copy[f'{feature}_binned'] == val]
                    if len(subset) > 0:
                        effect = subset[target_column].mean() - baseline_rate
                        effects.append(effect)
                matrix.iloc[i, i] = np.mean(effects) if effects else 0.0
            else:
                effects = []
                for val in df[feature].dropna().unique():
                    subset = df[df[feature] == val]
                    if len(subset) > 0:
                        effect = subset[target_column].mean() - baseline_rate
                        effects.append(effect)
                matrix.iloc[i, i] = np.mean(effects) if effects else 0.0
        except Exception:
            matrix.iloc[i, i] = 0.0
    
    # Fill off-diagonal with interaction effects
    for i, feature_a in enumerate(valid_features):
        for j, feature_b in enumerate(valid_features):
            if i != j:
                try:
                    result = calculate_interaction_effect(
                        df=df,
                        feature_a=feature_a,
                        feature_b=feature_b,
                        target_column=target_column,
                        bins_a=bins,
                        bins_b=bins
                    )
                    matrix.iloc[i, j] = result['interaction_effect']
                except Exception:
                    matrix.iloc[i, j] = 0.0
    
    return matrix
