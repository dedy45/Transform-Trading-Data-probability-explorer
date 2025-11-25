"""
What-If Scenarios Integration for ML Prediction Engine

This module provides integration between the ML Prediction Engine and What-If Scenarios,
allowing users to create scenarios based on ML predictions.

**Feature: ml-prediction-engine**
**Validates: Requirements 15.4**
"""

import pandas as pd
from typing import Dict, List, Optional, Union
from pathlib import Path


def create_ml_prediction_scenario(
    ml_predictions: pd.DataFrame,
    scenario_name: str = "ML Prediction Filter",
    filter_by_quality: Optional[List[str]] = None,
    filter_by_prob_min: Optional[float] = None,
    filter_by_recommendation: bool = True
) -> Dict:
    """
    Create a What-If scenario based on ML predictions.
    
    This function creates a scenario configuration that can be added to the
    What-If Scenarios page. The scenario filters trades based on ML prediction
    quality, probability, and recommendation.
    
    Parameters
    ----------
    ml_predictions : pd.DataFrame
        DataFrame with ML predictions containing columns:
        - prob_win_calibrated: Calibrated win probability
        - quality_label: Setup quality (A+/A/B/C)
        - recommendation: Trade recommendation (TRADE/SKIP)
        - R_P50_raw: Expected R-multiple
    scenario_name : str, optional
        Name for the scenario (default: "ML Prediction Filter")
    filter_by_quality : list of str, optional
        List of quality labels to keep (e.g., ['A+', 'A'])
        If None, defaults to ['A+', 'A']
    filter_by_prob_min : float, optional
        Minimum probability threshold (0-1)
        If None, defaults to 0.55
    filter_by_recommendation : bool, optional
        If True, only keep trades with TRADE recommendation (default: True)
    
    Returns
    -------
    dict
        Scenario configuration dictionary with keys:
        - name: Scenario name
        - type: 'ml_prediction'
        - params: Dictionary of parameters
        - description: Human-readable description
        - expected_impact: Expected impact on metrics
    
    Examples
    --------
    >>> # Create scenario for A+ and A quality setups
    >>> scenario = create_ml_prediction_scenario(
    ...     ml_predictions,
    ...     scenario_name="High Quality ML Filter",
    ...     filter_by_quality=['A+', 'A'],
    ...     filter_by_prob_min=0.60
    ... )
    >>> 
    >>> # Create scenario for only A+ setups
    >>> scenario = create_ml_prediction_scenario(
    ...     ml_predictions,
    ...     scenario_name="Excellent Only",
    ...     filter_by_quality=['A+'],
    ...     filter_by_prob_min=0.65
    ... )
    """
    # Set defaults
    if filter_by_quality is None:
        filter_by_quality = ['A+', 'A']
    
    if filter_by_prob_min is None:
        filter_by_prob_min = 0.55
    
    # Validate inputs
    if not isinstance(ml_predictions, pd.DataFrame):
        raise ValueError("ml_predictions must be a pandas DataFrame")
    
    required_cols = ['prob_win_calibrated', 'quality_label', 'recommendation']
    missing_cols = [col for col in required_cols if col not in ml_predictions.columns]
    if missing_cols:
        raise ValueError(f"ml_predictions missing required columns: {missing_cols}")
    
    # Calculate expected impact
    total_trades = len(ml_predictions)
    
    # Apply filters to see impact
    mask = pd.Series([True] * len(ml_predictions))
    
    if filter_by_quality:
        mask &= ml_predictions['quality_label'].isin(filter_by_quality)
    
    if filter_by_prob_min is not None:
        mask &= ml_predictions['prob_win_calibrated'] >= filter_by_prob_min
    
    if filter_by_recommendation:
        mask &= ml_predictions['recommendation'] == 'TRADE'
    
    filtered_count = mask.sum()
    filter_rate = (filtered_count / total_trades * 100) if total_trades > 0 else 0
    
    # Calculate average metrics for filtered trades
    filtered_preds = ml_predictions[mask.values]  # Use .values to avoid index alignment issues
    avg_prob = filtered_preds['prob_win_calibrated'].mean() if len(filtered_preds) > 0 else 0
    avg_expected_r = filtered_preds['R_P50_raw'].mean() if 'R_P50_raw' in filtered_preds.columns and len(filtered_preds) > 0 else 0
    
    # Build description
    quality_str = ", ".join(filter_by_quality) if filter_by_quality else "All"
    prob_str = f"{filter_by_prob_min:.0%}" if filter_by_prob_min else "Any"
    rec_str = "TRADE only" if filter_by_recommendation else "All"
    
    description = (
        f"Filter trades using ML predictions: "
        f"Quality={quality_str}, Min Prob={prob_str}, Recommendation={rec_str}"
    )
    
    # Build expected impact summary
    expected_impact = {
        'trades_kept': filtered_count,
        'trades_filtered': total_trades - filtered_count,
        'filter_rate': filter_rate,
        'avg_prob_win': avg_prob,
        'avg_expected_r': avg_expected_r,
        'description': (
            f"Keeps {filtered_count}/{total_trades} trades ({filter_rate:.1f}%). "
            f"Avg prob: {avg_prob:.1%}, Avg expected R: {avg_expected_r:.2f}"
        )
    }
    
    # Build scenario configuration
    scenario = {
        'name': scenario_name,
        'type': 'ml_prediction',
        'params': {
            'filter_by_quality': filter_by_quality,
            'filter_by_prob_min': filter_by_prob_min,
            'filter_by_recommendation': filter_by_recommendation
        },
        'description': description,
        'expected_impact': expected_impact
    }
    
    return scenario


def create_ml_scenario_presets(ml_predictions: pd.DataFrame) -> List[Dict]:
    """
    Create a set of preset ML prediction scenarios.
    
    This function creates several common ML prediction scenarios that users
    can quickly add to their What-If analysis.
    
    Parameters
    ----------
    ml_predictions : pd.DataFrame
        DataFrame with ML predictions
    
    Returns
    -------
    list of dict
        List of scenario configurations
    
    Examples
    --------
    >>> presets = create_ml_scenario_presets(ml_predictions)
    >>> for preset in presets:
    ...     print(preset['name'], preset['expected_impact']['description'])
    """
    presets = []
    
    # Preset 1: Conservative (A+ only)
    try:
        presets.append(create_ml_prediction_scenario(
            ml_predictions,
            scenario_name="ML Conservative (A+ only)",
            filter_by_quality=['A+'],
            filter_by_prob_min=0.65,
            filter_by_recommendation=True
        ))
    except Exception as e:
        print(f"Warning: Could not create conservative preset: {e}")
    
    # Preset 2: Balanced (A+ and A)
    try:
        presets.append(create_ml_prediction_scenario(
            ml_predictions,
            scenario_name="ML Balanced (A+/A)",
            filter_by_quality=['A+', 'A'],
            filter_by_prob_min=0.55,
            filter_by_recommendation=True
        ))
    except Exception as e:
        print(f"Warning: Could not create balanced preset: {e}")
    
    # Preset 3: Aggressive (A+/A/B)
    try:
        presets.append(create_ml_prediction_scenario(
            ml_predictions,
            scenario_name="ML Aggressive (A+/A/B)",
            filter_by_quality=['A+', 'A', 'B'],
            filter_by_prob_min=0.50,
            filter_by_recommendation=True
        ))
    except Exception as e:
        print(f"Warning: Could not create aggressive preset: {e}")
    
    # Preset 4: High Probability (60%+)
    try:
        presets.append(create_ml_prediction_scenario(
            ml_predictions,
            scenario_name="ML High Probability (60%+)",
            filter_by_quality=None,  # Any quality
            filter_by_prob_min=0.60,
            filter_by_recommendation=True
        ))
    except Exception as e:
        print(f"Warning: Could not create high probability preset: {e}")
    
    return presets


def get_ml_scenario_summary(
    ml_predictions: pd.DataFrame,
    scenario_params: Dict
) -> Dict:
    """
    Get a summary of what an ML scenario would do.
    
    This function calculates the impact of applying an ML prediction scenario
    without actually modifying the trades.
    
    Parameters
    ----------
    ml_predictions : pd.DataFrame
        DataFrame with ML predictions
    scenario_params : dict
        Scenario parameters with keys:
        - filter_by_quality: List of quality labels
        - filter_by_prob_min: Minimum probability
        - filter_by_recommendation: Boolean
    
    Returns
    -------
    dict
        Summary with keys:
        - total_trades: Total number of trades
        - trades_kept: Number of trades that pass filters
        - trades_filtered: Number of trades filtered out
        - filter_rate: Percentage of trades kept
        - quality_distribution: Distribution of quality labels in kept trades
        - avg_prob_win: Average probability of kept trades
        - avg_expected_r: Average expected R of kept trades
    """
    total_trades = len(ml_predictions)
    
    # Apply filters
    mask = pd.Series([True] * len(ml_predictions))
    
    filter_by_quality = scenario_params.get('filter_by_quality')
    filter_by_prob_min = scenario_params.get('filter_by_prob_min')
    filter_by_recommendation = scenario_params.get('filter_by_recommendation', True)
    
    if filter_by_quality:
        mask &= ml_predictions['quality_label'].isin(filter_by_quality)
    
    if filter_by_prob_min is not None:
        mask &= ml_predictions['prob_win_calibrated'] >= filter_by_prob_min
    
    if filter_by_recommendation:
        mask &= ml_predictions['recommendation'] == 'TRADE'
    
    filtered_preds = ml_predictions[mask.values]  # Use .values to avoid index alignment issues
    trades_kept = len(filtered_preds)
    trades_filtered = total_trades - trades_kept
    filter_rate = (trades_kept / total_trades * 100) if total_trades > 0 else 0
    
    # Quality distribution
    quality_dist = filtered_preds['quality_label'].value_counts().to_dict() if len(filtered_preds) > 0 else {}
    
    # Average metrics
    avg_prob = filtered_preds['prob_win_calibrated'].mean() if len(filtered_preds) > 0 else 0
    avg_expected_r = filtered_preds['R_P50_raw'].mean() if 'R_P50_raw' in filtered_preds.columns and len(filtered_preds) > 0 else 0
    
    summary = {
        'total_trades': total_trades,
        'trades_kept': trades_kept,
        'trades_filtered': trades_filtered,
        'filter_rate': filter_rate,
        'quality_distribution': quality_dist,
        'avg_prob_win': avg_prob,
        'avg_expected_r': avg_expected_r
    }
    
    return summary
