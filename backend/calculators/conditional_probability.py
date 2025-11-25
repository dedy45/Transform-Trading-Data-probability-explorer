"""
Conditional Probability Engine Module

This module implements conditional probability calculations for multi-dimensional
analysis, including lift ratios, greedy condition building, and filtering.

Enables finding optimal combinations of conditions that maximize target probability.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple, Any
from backend.models.confidence_intervals import beta_posterior_ci


def calculate_conditional_probability(
    df: pd.DataFrame,
    target: str,
    conditions: Dict[str, Any],
    conf_level: float = 0.95
) -> Dict[str, Any]:
    """
    Calculate P(target | conditions) for a set of conditions.
    
    Filters the dataset based on conditions and calculates the probability
    of the target being True given those conditions.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with features and target
    target : str
        Target column name (e.g., 'y_win', 'y_hit_1R')
    conditions : dict
        Dictionary of conditions to apply. Format:
        - For equality: {'feature': value}
        - For range: {'feature': (min_val, max_val)}
        - For categorical list: {'feature': [val1, val2, ...]}
    conf_level : float
        Confidence level for interval (0.8 to 0.99)
        
    Returns:
    --------
    dict with:
        - probability: P(target | conditions)
        - ci_lower: Lower confidence bound
        - ci_upper: Upper confidence bound
        - n_samples: Number of samples matching conditions
        - n_successes: Number of successes in filtered data
        - conditions: Echo of input conditions
        
    Raises:
    -------
    ValueError: If target column missing or invalid parameters
    
    Examples:
    ---------
    >>> # Single condition
    >>> result = calculate_conditional_probability(
    ...     df, 'y_win', {'session': 1}
    ... )
    >>> 
    >>> # Multiple conditions with range
    >>> result = calculate_conditional_probability(
    ...     df, 'y_win', 
    ...     {'session': 1, 'trend_strength_tf': (0.5, 1.0)}
    ... )
    """
    # Validate inputs
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame")
    if not 0 < conf_level < 1:
        raise ValueError(f"Confidence level must be between 0 and 1, got {conf_level}")
    if not conditions:
        raise ValueError("At least one condition must be provided")
    
    # Start with full dataset
    filtered_df = df.copy()
    
    # Apply each condition
    for feature, condition_value in conditions.items():
        if feature not in df.columns:
            raise ValueError(f"Feature column '{feature}' not found in DataFrame")
        
        # Handle different condition types
        if isinstance(condition_value, tuple) and len(condition_value) == 2:
            # Range condition: (min, max)
            min_val, max_val = condition_value
            filtered_df = filtered_df[
                (filtered_df[feature] >= min_val) & 
                (filtered_df[feature] <= max_val)
            ]
        elif isinstance(condition_value, list):
            # List of values (categorical)
            filtered_df = filtered_df[filtered_df[feature].isin(condition_value)]
        else:
            # Equality condition
            filtered_df = filtered_df[filtered_df[feature] == condition_value]
    
    # Calculate probability
    n_samples = len(filtered_df)
    n_successes = int(filtered_df[target].sum()) if n_samples > 0 else 0
    
    if n_samples == 0:
        return {
            'probability': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'n_samples': 0,
            'n_successes': 0,
            'conditions': conditions
        }
    
    # Calculate confidence interval
    ci_result = beta_posterior_ci(n_successes, n_samples, conf_level=conf_level)
    
    return {
        'probability': ci_result['p_mean'],
        'ci_lower': ci_result['ci_lower'],
        'ci_upper': ci_result['ci_upper'],
        'n_samples': n_samples,
        'n_successes': n_successes,
        'conditions': conditions
    }


def calculate_lift_ratio(
    conditional_prob: float,
    base_rate: float
) -> float:
    """
    Calculate lift ratio: conditional_probability / base_rate.
    
    Lift ratio indicates how much better the conditional probability is
    compared to the base rate. Values > 1 indicate improvement.
    
    Parameters:
    -----------
    conditional_prob : float
        Conditional probability P(target | conditions)
    base_rate : float
        Base rate P(target) without conditions
        
    Returns:
    --------
    float: Lift ratio (conditional_prob / base_rate)
    
    Raises:
    -------
    ValueError: If base_rate is 0 or negative
    
    Examples:
    ---------
    >>> lift = calculate_lift_ratio(0.65, 0.52)
    >>> print(f"Lift: {lift:.2f}x")  # Output: Lift: 1.25x
    """
    if base_rate <= 0:
        raise ValueError(f"Base rate must be positive, got {base_rate}")
    
    return conditional_prob / base_rate


def sequential_condition_builder(
    df: pd.DataFrame,
    target: str,
    candidate_features: List[str],
    max_conditions: int = 5,
    min_samples: int = 30,
    min_lift: float = 1.1,
    conf_level: float = 0.95
) -> List[Dict[str, Any]]:
    """
    Build optimal condition combinations using greedy algorithm.
    
    Iteratively adds conditions that maximize probability improvement,
    stopping when no improvement is found or max_conditions reached.
    
    Algorithm:
    1. Calculate base rate P(target)
    2. For each candidate feature, find best threshold/value
    3. Add condition that gives highest probability
    4. Repeat with remaining features until stopping criterion
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with features and target
    target : str
        Target column name
    candidate_features : list of str
        Features to consider for conditions
    max_conditions : int
        Maximum number of conditions to add
    min_samples : int
        Minimum samples required to consider a condition
    min_lift : float
        Minimum lift ratio required to add a condition
    conf_level : float
        Confidence level for intervals
        
    Returns:
    --------
    list of dict, each containing:
        - step: Step number (0 = base rate, 1+ = after adding conditions)
        - conditions: Dict of conditions at this step
        - probability: P(target | conditions)
        - ci_lower: Lower confidence bound
        - ci_upper: Upper confidence bound
        - n_samples: Sample size
        - lift: Lift ratio vs base rate
        - improvement: Probability improvement vs previous step
        
    Examples:
    ---------
    >>> results = sequential_condition_builder(
    ...     df, 'y_win',
    ...     ['session', 'trend_strength_tf', 'volatility_regime'],
    ...     max_conditions=3
    ... )
    >>> for step in results:
    ...     print(f"Step {step['step']}: P={step['probability']:.3f}, "
    ...           f"Lift={step['lift']:.2f}x")
    """
    # Validate inputs
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame")
    if not candidate_features:
        raise ValueError("At least one candidate feature must be provided")
    if max_conditions < 1:
        raise ValueError(f"max_conditions must be >= 1, got {max_conditions}")
    if min_samples < 1:
        raise ValueError(f"min_samples must be >= 1, got {min_samples}")
    if min_lift <= 0:
        raise ValueError(f"min_lift must be positive, got {min_lift}")
    
    # Calculate base rate
    base_rate = float(df[target].mean())
    n_total = len(df)
    n_successes_total = int(df[target].sum())
    
    ci_base = beta_posterior_ci(n_successes_total, n_total, conf_level=conf_level)
    
    results = [{
        'step': 0,
        'conditions': {},
        'probability': base_rate,
        'ci_lower': ci_base['ci_lower'],
        'ci_upper': ci_base['ci_upper'],
        'n_samples': n_total,
        'lift': 1.0,
        'improvement': 0.0
    }]
    
    # Track current conditions and available features
    current_conditions = {}
    available_features = set(candidate_features)
    current_prob = base_rate
    
    # Greedy search
    for step in range(1, max_conditions + 1):
        if not available_features:
            break
        
        best_feature = None
        best_condition_value = None
        best_prob = current_prob
        best_result = None
        
        # Try each available feature
        for feature in available_features:
            if feature not in df.columns:
                continue
            
            # Determine if feature is numeric or categorical
            is_numeric = pd.api.types.is_numeric_dtype(df[feature])
            
            if is_numeric:
                # Try different quantile thresholds
                quantiles = [0.25, 0.33, 0.5, 0.67, 0.75]
                thresholds = df[feature].quantile(quantiles).unique()
                
                for threshold in thresholds:
                    # Try both >= and <=
                    for operator in ['>=', '<=']:
                        test_conditions = current_conditions.copy()
                        
                        if operator == '>=':
                            test_conditions[feature] = (threshold, df[feature].max())
                        else:
                            test_conditions[feature] = (df[feature].min(), threshold)
                        
                        try:
                            result = calculate_conditional_probability(
                                df, target, test_conditions, conf_level
                            )
                            
                            if (result['n_samples'] >= min_samples and 
                                result['probability'] > best_prob):
                                best_prob = result['probability']
                                best_feature = feature
                                best_condition_value = test_conditions[feature]
                                best_result = result
                        except:
                            continue
            else:
                # Categorical: try each unique value
                unique_values = df[feature].unique()
                
                for value in unique_values:
                    test_conditions = current_conditions.copy()
                    test_conditions[feature] = value
                    
                    try:
                        result = calculate_conditional_probability(
                            df, target, test_conditions, conf_level
                        )
                        
                        if (result['n_samples'] >= min_samples and 
                            result['probability'] > best_prob):
                            best_prob = result['probability']
                            best_feature = feature
                            best_condition_value = value
                            best_result = result
                    except:
                        continue
        
        # Check if we found an improvement
        if best_feature is None:
            break
        
        # Calculate lift
        lift = calculate_lift_ratio(best_prob, base_rate)
        
        # Check if lift meets minimum threshold
        if lift < min_lift:
            break
        
        # Add the best condition
        current_conditions[best_feature] = best_condition_value
        available_features.remove(best_feature)
        improvement = best_prob - current_prob
        current_prob = best_prob
        
        results.append({
            'step': step,
            'conditions': current_conditions.copy(),
            'probability': best_result['probability'],
            'ci_lower': best_result['ci_lower'],
            'ci_upper': best_result['ci_upper'],
            'n_samples': best_result['n_samples'],
            'lift': lift,
            'improvement': improvement
        })
    
    return results


def filter_by_thresholds(
    results: List[Dict[str, Any]],
    min_samples: int,
    min_lift: float
) -> List[Dict[str, Any]]:
    """
    Filter condition results by minimum sample size and lift thresholds.
    
    Parameters:
    -----------
    results : list of dict
        Results from sequential_condition_builder or similar
    min_samples : int
        Minimum sample size threshold
    min_lift : float
        Minimum lift ratio threshold
        
    Returns:
    --------
    list of dict: Filtered results meeting both thresholds
    
    Examples:
    ---------
    >>> filtered = filter_by_thresholds(results, min_samples=50, min_lift=1.2)
    """
    if min_samples < 1:
        raise ValueError(f"min_samples must be >= 1, got {min_samples}")
    if min_lift <= 0:
        raise ValueError(f"min_lift must be positive, got {min_lift}")
    
    filtered = []
    for result in results:
        if result['n_samples'] >= min_samples and result['lift'] >= min_lift:
            filtered.append(result)
    
    return filtered


def sort_by_probability_and_significance(
    results: List[Dict[str, Any]],
    ascending: bool = False
) -> List[Dict[str, Any]]:
    """
    Sort results by probability (primary) and statistical significance (secondary).
    
    Statistical significance is measured by confidence interval width
    (narrower = more significant).
    
    Parameters:
    -----------
    results : list of dict
        Results to sort
    ascending : bool
        If True, sort ascending (lowest probability first)
        If False, sort descending (highest probability first)
        
    Returns:
    --------
    list of dict: Sorted results
    
    Examples:
    ---------
    >>> sorted_results = sort_by_probability_and_significance(results)
    >>> # Highest probability combinations first
    """
    if not results:
        return []
    
    # Calculate CI width for each result (for secondary sort)
    for result in results:
        ci_width = result['ci_upper'] - result['ci_lower']
        result['_ci_width'] = ci_width
    
    # Sort by probability (primary), then by CI width (secondary, ascending)
    sorted_results = sorted(
        results,
        key=lambda x: (x['probability'], -x['_ci_width']),
        reverse=not ascending
    )
    
    # Remove temporary field
    for result in sorted_results:
        del result['_ci_width']
    
    return sorted_results


def find_top_combinations(
    df: pd.DataFrame,
    target: str,
    features: List[str],
    top_n: int = 10,
    min_samples: int = 30,
    min_lift: float = 1.1,
    conf_level: float = 0.95
) -> pd.DataFrame:
    """
    Find top N condition combinations with highest probabilities.
    
    Uses sequential_condition_builder for each starting feature,
    then aggregates and ranks all combinations found.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with features and target
    target : str
        Target column name
    features : list of str
        Features to consider
    top_n : int
        Number of top combinations to return
    min_samples : int
        Minimum samples per combination
    min_lift : float
        Minimum lift ratio
    conf_level : float
        Confidence level for intervals
        
    Returns:
    --------
    pd.DataFrame with columns:
        - rank: Rank (1 = best)
        - probability: P(target | conditions)
        - lift: Lift ratio vs base rate
        - n_samples: Sample size
        - ci_lower: Lower confidence bound
        - ci_upper: Upper confidence bound
        - conditions: String representation of conditions
        - n_conditions: Number of conditions
        
    Examples:
    ---------
    >>> top_combos = find_top_combinations(
    ...     df, 'y_win',
    ...     ['session', 'trend_strength_tf', 'volatility_regime'],
    ...     top_n=5
    ... )
    """
    all_results = []
    
    # Run sequential builder starting from each feature
    for start_feature in features:
        other_features = [f for f in features if f != start_feature]
        candidate_features = [start_feature] + other_features
        
        try:
            results = sequential_condition_builder(
                df, target, candidate_features,
                max_conditions=len(features),
                min_samples=min_samples,
                min_lift=min_lift,
                conf_level=conf_level
            )
            
            # Skip base rate (step 0)
            all_results.extend([r for r in results if r['step'] > 0])
        except:
            continue
    
    if not all_results:
        return pd.DataFrame()
    
    # Remove duplicates (same conditions)
    unique_results = []
    seen_conditions = set()
    
    for result in all_results:
        # Create hashable representation of conditions
        cond_str = str(sorted(result['conditions'].items()))
        if cond_str not in seen_conditions:
            seen_conditions.add(cond_str)
            unique_results.append(result)
    
    # Sort by probability and significance
    sorted_results = sort_by_probability_and_significance(unique_results)
    
    # Take top N
    top_results = sorted_results[:top_n]
    
    # Convert to DataFrame
    df_results = pd.DataFrame([
        {
            'rank': i + 1,
            'probability': r['probability'],
            'lift': r['lift'],
            'n_samples': r['n_samples'],
            'ci_lower': r['ci_lower'],
            'ci_upper': r['ci_upper'],
            'conditions': str(r['conditions']),
            'n_conditions': len(r['conditions'])
        }
        for i, r in enumerate(top_results)
    ])
    
    return df_results
