"""
Bayesian Probability Calculator for Trading Probability Explorer

This module provides Bayesian inference methods for probability estimation,
particularly useful for small sample sizes where frequentist methods may be unreliable.

Key Features:
- Bayesian win rate estimation with Beta posterior
- Adaptive probability tracking with rolling windows
- Bayesian comparison between conditions
- Certainty calculation based on posterior distribution width

All methods use Beta-Binomial conjugate prior for efficient computation.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple
from backend.models.confidence_intervals import beta_posterior_ci


def bayesian_win_rate(successes: int, total: int, alpha0: float = 1.0, beta0: float = 1.0,
                      conf_level: float = 0.95) -> Dict[str, float]:
    """
    Calculate Bayesian win rate estimate using Beta posterior distribution.
    
    This method is particularly useful for small sample sizes where frequentist
    methods may give unreliable estimates. The Beta distribution serves as a
    conjugate prior for binomial data, making computation efficient.
    
    Parameters:
    -----------
    successes : int
        Number of winning trades
    total : int
        Total number of trades
    alpha0 : float
        Prior alpha parameter (default 1.0 for uniform prior)
        - alpha0=1, beta0=1: Uniform prior (non-informative)
        - alpha0=0.5, beta0=0.5: Jeffreys prior (scale-invariant)
        - Higher values: More informative prior
    beta0 : float
        Prior beta parameter (default 1.0 for uniform prior)
    conf_level : float
        Confidence level for credible interval (0.8 to 0.99)
        
    Returns:
    --------
    dict with keys:
        - 'posterior_mean': Posterior mean probability (best point estimate)
        - 'posterior_mode': Posterior mode (MAP estimate)
        - 'posterior_median': Posterior median
        - 'credible_lower': Lower bound of credible interval
        - 'credible_upper': Upper bound of credible interval
        - 'posterior_std': Standard deviation of posterior
        - 'certainty': Certainty score (0-1, higher = more certain)
        - 'n_successes': Number of successes
        - 'n_total': Total sample size
        - 'alpha_posterior': Posterior alpha parameter
        - 'beta_posterior': Posterior beta parameter
        
    Examples:
    ---------
    >>> # Small sample: 7 wins out of 10 trades
    >>> result = bayesian_win_rate(7, 10)
    >>> print(f"Win rate: {result['posterior_mean']:.1%}")
    >>> print(f"95% Credible interval: [{result['credible_lower']:.1%}, {result['credible_upper']:.1%}]")
    >>> print(f"Certainty: {result['certainty']:.2f}")
    
    >>> # With informative prior (e.g., historical win rate of 60%)
    >>> result = bayesian_win_rate(7, 10, alpha0=60, beta0=40)
    >>> print(f"Win rate (with prior): {result['posterior_mean']:.1%}")
    """
    # Validate inputs
    if total < 0:
        raise ValueError(f"Total must be non-negative, got {total}")
    
    if successes < 0 or successes > total:
        raise ValueError(f"Successes must be between 0 and total, got {successes} out of {total}")
    
    if alpha0 <= 0 or beta0 <= 0:
        raise ValueError(f"Prior parameters must be positive, got alpha0={alpha0}, beta0={beta0}")
    
    if not 0 < conf_level < 1:
        raise ValueError(f"Confidence level must be between 0 and 1, got {conf_level}")
    
    # Calculate posterior parameters
    failures = total - successes
    alpha_post = alpha0 + successes
    beta_post = beta0 + failures
    
    # Posterior mean (expected value)
    posterior_mean = alpha_post / (alpha_post + beta_post)
    
    # Posterior mode (MAP estimate)
    if alpha_post > 1 and beta_post > 1:
        posterior_mode = (alpha_post - 1) / (alpha_post + beta_post - 2)
    else:
        posterior_mode = posterior_mean
    
    # Posterior median
    posterior_median = stats.beta.median(alpha_post, beta_post)
    
    # Posterior standard deviation
    posterior_std = np.sqrt(
        (alpha_post * beta_post) / 
        ((alpha_post + beta_post) ** 2 * (alpha_post + beta_post + 1))
    )
    
    # Credible interval (equal-tailed)
    alpha = 1 - conf_level
    credible_lower = stats.beta.ppf(alpha / 2, alpha_post, beta_post)
    credible_upper = stats.beta.ppf(1 - alpha / 2, alpha_post, beta_post)
    
    # Calculate certainty score (inverse of relative uncertainty)
    # Certainty is high when posterior std is low relative to the mean
    # Scale to [0, 1] where 1 is most certain
    if posterior_mean > 0:
        coefficient_of_variation = posterior_std / posterior_mean
        # Use exponential decay: certainty = exp(-k * CV)
        certainty = np.exp(-2 * coefficient_of_variation)
    else:
        certainty = 0.0
    
    # Alternative certainty: based on credible interval width
    # Narrower interval = higher certainty
    interval_width = credible_upper - credible_lower
    certainty_alt = 1.0 - interval_width
    
    # Use average of both certainty measures
    certainty_final = (certainty + certainty_alt) / 2
    
    return {
        'posterior_mean': posterior_mean,
        'posterior_mode': posterior_mode,
        'posterior_median': posterior_median,
        'credible_lower': credible_lower,
        'credible_upper': credible_upper,
        'posterior_std': posterior_std,
        'certainty': certainty_final,
        'n_successes': successes,
        'n_total': total,
        'alpha_posterior': alpha_post,
        'beta_posterior': beta_post
    }


def adaptive_probability_tracker(df: pd.DataFrame, target_col: str = 'trade_success',
                                 window_size: int = 50, alpha0: float = 1.0, 
                                 beta0: float = 1.0, conf_level: float = 0.95) -> pd.DataFrame:
    """
    Track probability evolution using rolling Bayesian updates.
    
    This function computes a rolling Bayesian probability estimate, updating
    the posterior as new trades arrive. Useful for detecting regime changes
    or tracking strategy performance over time.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with trade data (must be sorted by time)
    target_col : str
        Column name for binary outcome (0/1)
    window_size : int
        Number of recent trades to include in rolling window
    alpha0 : float
        Prior alpha parameter
    beta0 : float
        Prior beta parameter
    conf_level : float
        Confidence level for credible intervals
        
    Returns:
    --------
    pd.DataFrame with columns:
        - 'trade_index': Trade sequence number
        - 'posterior_mean': Rolling Bayesian probability estimate
        - 'credible_lower': Lower bound of credible interval
        - 'credible_upper': Upper bound of credible interval
        - 'certainty': Certainty score
        - 'n_window': Number of trades in current window
        
    Examples:
    ---------
    >>> df = pd.DataFrame({'trade_success': [1, 0, 1, 1, 0, 1, 1, 1, 0, 1]})
    >>> tracker = adaptive_probability_tracker(df, window_size=5)
    >>> print(tracker[['trade_index', 'posterior_mean', 'certainty']])
    """
    # Validate inputs
    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found in DataFrame")
    
    if window_size < 1:
        raise ValueError(f"Window size must be at least 1, got {window_size}")
    
    if len(df) == 0:
        return pd.DataFrame()
    
    # Initialize results
    results = []
    
    # Iterate through each trade
    for i in range(len(df)):
        # Define window: last window_size trades up to current trade
        start_idx = max(0, i - window_size + 1)
        end_idx = i + 1
        window_data = df.iloc[start_idx:end_idx]
        
        # Calculate successes and total in window
        successes = int(window_data[target_col].sum())
        total = len(window_data)
        
        # Calculate Bayesian estimate for this window
        bayesian_result = bayesian_win_rate(
            successes=successes,
            total=total,
            alpha0=alpha0,
            beta0=beta0,
            conf_level=conf_level
        )
        
        # Store results
        results.append({
            'trade_index': i,
            'posterior_mean': bayesian_result['posterior_mean'],
            'credible_lower': bayesian_result['credible_lower'],
            'credible_upper': bayesian_result['credible_upper'],
            'certainty': bayesian_result['certainty'],
            'n_window': total,
            'n_successes_window': successes
        })
    
    return pd.DataFrame(results)


def bayesian_comparison(df: pd.DataFrame, target_col: str, condition_a: Dict, 
                       condition_b: Dict, alpha0: float = 1.0, beta0: float = 1.0,
                       n_samples: int = 100000) -> Dict[str, float]:
    """
    Compare two conditions using Bayesian hypothesis testing.
    
    This function calculates the probability that condition A has a higher
    win rate than condition B by sampling from their posterior distributions.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with trade data and features
    target_col : str
        Column name for binary outcome (0/1)
    condition_a : dict
        Dictionary of column:value pairs defining condition A
        Example: {'session': 1, 'trend_tf_dir': 1}
    condition_b : dict
        Dictionary of column:value pairs defining condition B
        Example: {'session': 0, 'trend_tf_dir': -1}
    alpha0 : float
        Prior alpha parameter
    beta0 : float
        Prior beta parameter
    n_samples : int
        Number of Monte Carlo samples for comparison (default 100000)
        
    Returns:
    --------
    dict with keys:
        - 'prob_a_better': Probability that A has higher win rate than B
        - 'prob_b_better': Probability that B has higher win rate than A
        - 'prob_equal': Probability that A and B are equal (approximately)
        - 'mean_diff': Expected difference in win rates (A - B)
        - 'a_posterior_mean': Posterior mean for condition A
        - 'b_posterior_mean': Posterior mean for condition B
        - 'a_n': Sample size for condition A
        - 'b_n': Sample size for condition B
        - 'a_successes': Number of successes for condition A
        - 'b_successes': Number of successes for condition B
        - 'effect_size': Cohen's h effect size
        
    Examples:
    ---------
    >>> # Compare Europe session vs Asia session
    >>> result = bayesian_comparison(
    ...     df, 'trade_success',
    ...     condition_a={'session': 1},  # Europe
    ...     condition_b={'session': 0}   # Asia
    ... )
    >>> print(f"P(Europe better than Asia): {result['prob_a_better']:.1%}")
    >>> print(f"Expected difference: {result['mean_diff']:.1%}")
    """
    # Validate inputs
    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found in DataFrame")
    
    # Filter data for condition A
    mask_a = pd.Series([True] * len(df))
    for col, val in condition_a.items():
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
        mask_a &= (df[col] == val)
    
    df_a = df[mask_a]
    
    # Filter data for condition B
    mask_b = pd.Series([True] * len(df))
    for col, val in condition_b.items():
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
        mask_b &= (df[col] == val)
    
    df_b = df[mask_b]
    
    # Check if we have data for both conditions
    if len(df_a) == 0:
        raise ValueError(f"No data found for condition A: {condition_a}")
    if len(df_b) == 0:
        raise ValueError(f"No data found for condition B: {condition_b}")
    
    # Calculate successes and totals
    successes_a = int(df_a[target_col].sum())
    total_a = len(df_a)
    successes_b = int(df_b[target_col].sum())
    total_b = len(df_b)
    
    # Calculate posterior parameters
    alpha_post_a = alpha0 + successes_a
    beta_post_a = beta0 + (total_a - successes_a)
    alpha_post_b = alpha0 + successes_b
    beta_post_b = beta0 + (total_b - successes_b)
    
    # Posterior means
    mean_a = alpha_post_a / (alpha_post_a + beta_post_a)
    mean_b = alpha_post_b / (alpha_post_b + beta_post_b)
    
    # Sample from posterior distributions
    samples_a = np.random.beta(alpha_post_a, beta_post_a, n_samples)
    samples_b = np.random.beta(alpha_post_b, beta_post_b, n_samples)
    
    # Calculate probabilities
    prob_a_better = np.mean(samples_a > samples_b)
    prob_b_better = np.mean(samples_b > samples_a)
    prob_equal = np.mean(np.abs(samples_a - samples_b) < 0.01)  # Within 1%
    
    # Calculate expected difference
    mean_diff = mean_a - mean_b
    
    # Calculate effect size (Cohen's h for proportions)
    # h = 2 * (arcsin(sqrt(p1)) - arcsin(sqrt(p2)))
    effect_size = 2 * (np.arcsin(np.sqrt(mean_a)) - np.arcsin(np.sqrt(mean_b)))
    
    return {
        'prob_a_better': prob_a_better,
        'prob_b_better': prob_b_better,
        'prob_equal': prob_equal,
        'mean_diff': mean_diff,
        'a_posterior_mean': mean_a,
        'b_posterior_mean': mean_b,
        'a_n': total_a,
        'b_n': total_b,
        'a_successes': successes_a,
        'b_successes': successes_b,
        'effect_size': effect_size
    }


def calculate_certainty(posterior_mean: float, posterior_std: float, 
                       credible_lower: float, credible_upper: float) -> float:
    """
    Calculate certainty score from posterior distribution characteristics.
    
    Certainty is a measure of how confident we are in the probability estimate.
    It combines information from the posterior standard deviation and the
    width of the credible interval.
    
    Parameters:
    -----------
    posterior_mean : float
        Mean of the posterior distribution
    posterior_std : float
        Standard deviation of the posterior distribution
    credible_lower : float
        Lower bound of credible interval
    credible_upper : float
        Upper bound of credible interval
        
    Returns:
    --------
    float
        Certainty score in range [0, 1]
        - 1.0: Very certain (narrow posterior, large sample)
        - 0.5: Moderate certainty
        - 0.0: Very uncertain (wide posterior, small sample)
        
    Notes:
    ------
    Certainty decreases as:
    - Posterior standard deviation increases
    - Credible interval width increases
    - Sample size decreases (implicitly through wider intervals)
    
    Examples:
    ---------
    >>> # High certainty: narrow interval
    >>> certainty = calculate_certainty(0.65, 0.05, 0.60, 0.70)
    >>> print(f"Certainty: {certainty:.2f}")  # ~0.90
    
    >>> # Low certainty: wide interval
    >>> certainty = calculate_certainty(0.50, 0.20, 0.20, 0.80)
    >>> print(f"Certainty: {certainty:.2f}")  # ~0.20
    """
    # Validate inputs
    if not 0 <= posterior_mean <= 1:
        raise ValueError(f"Posterior mean must be in [0, 1], got {posterior_mean}")
    
    if posterior_std < 0:
        raise ValueError(f"Posterior std must be non-negative, got {posterior_std}")
    
    if not 0 <= credible_lower <= credible_upper <= 1:
        raise ValueError(f"Invalid credible interval: [{credible_lower}, {credible_upper}]")
    
    # Method 1: Based on coefficient of variation
    # Lower CV = higher certainty
    if posterior_mean > 0:
        cv = posterior_std / posterior_mean
        certainty_cv = np.exp(-2 * cv)  # Exponential decay
    else:
        certainty_cv = 0.0
    
    # Method 2: Based on credible interval width
    # Narrower interval = higher certainty
    interval_width = credible_upper - credible_lower
    certainty_width = 1.0 - interval_width
    
    # Method 3: Based on relative precision
    # Precision = 1 / variance
    # Higher precision = higher certainty
    if posterior_std > 0:
        precision = 1.0 / (posterior_std ** 2)
        # Normalize to [0, 1] using sigmoid-like function
        certainty_precision = 1.0 / (1.0 + np.exp(-0.5 * (precision - 10)))
    else:
        certainty_precision = 1.0
    
    # Combine all three methods (weighted average)
    certainty = (
        0.4 * certainty_cv +
        0.4 * certainty_width +
        0.2 * certainty_precision
    )
    
    # Ensure result is in [0, 1]
    certainty = np.clip(certainty, 0.0, 1.0)
    
    return certainty


def bayesian_win_rate_by_group(df: pd.DataFrame, target_col: str, group_col: str,
                               alpha0: float = 1.0, beta0: float = 1.0,
                               conf_level: float = 0.95, min_samples: int = 5) -> pd.DataFrame:
    """
    Calculate Bayesian win rate for each group in a categorical variable.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with trade data
    target_col : str
        Column name for binary outcome (0/1)
    group_col : str
        Column name for grouping variable
    alpha0 : float
        Prior alpha parameter
    beta0 : float
        Prior beta parameter
    conf_level : float
        Confidence level for credible intervals
    min_samples : int
        Minimum sample size to include group (default 5)
        
    Returns:
    --------
    pd.DataFrame with columns:
        - group_col: Group identifier
        - 'n': Sample size
        - 'n_successes': Number of successes
        - 'posterior_mean': Posterior mean probability
        - 'credible_lower': Lower bound of credible interval
        - 'credible_upper': Upper bound of credible interval
        - 'certainty': Certainty score
        - 'reliable': Boolean indicating if sample size >= min_samples
        
    Examples:
    ---------
    >>> # Win rate by session
    >>> results = bayesian_win_rate_by_group(df, 'trade_success', 'session')
    >>> print(results[['session', 'posterior_mean', 'certainty']])
    """
    # Validate inputs
    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found in DataFrame")
    
    if group_col not in df.columns:
        raise ValueError(f"Column '{group_col}' not found in DataFrame")
    
    # Group by the specified column
    grouped = df.groupby(group_col)
    
    results = []
    
    for group_value, group_df in grouped:
        n_total = len(group_df)
        n_successes = int(group_df[target_col].sum())
        
        # Calculate Bayesian estimate
        bayesian_result = bayesian_win_rate(
            successes=n_successes,
            total=n_total,
            alpha0=alpha0,
            beta0=beta0,
            conf_level=conf_level
        )
        
        results.append({
            group_col: group_value,
            'n': n_total,
            'n_successes': n_successes,
            'posterior_mean': bayesian_result['posterior_mean'],
            'credible_lower': bayesian_result['credible_lower'],
            'credible_upper': bayesian_result['credible_upper'],
            'certainty': bayesian_result['certainty'],
            'reliable': n_total >= min_samples
        })
    
    return pd.DataFrame(results)
