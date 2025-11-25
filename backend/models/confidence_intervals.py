"""
Confidence Interval Calculators for Trading Probability Explorer

This module provides three methods for calculating confidence intervals:
1. Wilson Score Interval (binomial_normal_ci) - for binomial proportions
2. Beta Posterior Interval (beta_posterior_ci) - Bayesian approach with Beta distribution
3. Bootstrap Interval (bootstrap_ci) - non-parametric robust method

All methods return confidence intervals in the range [0, 1] for probability estimates.
"""

import numpy as np
from scipy import stats
from typing import Dict, Callable, List


def binomial_normal_ci(successes: int, total: int, conf_level: float = 0.95) -> Dict[str, float]:
    """
    Calculate confidence interval using Wilson Score method.
    
    The Wilson Score interval is more accurate than the normal approximation,
    especially for small sample sizes or extreme probabilities.
    
    Parameters:
    -----------
    successes : int
        Number of successful outcomes (wins)
    total : int
        Total number of trials (trades)
    conf_level : float
        Confidence level (0.8 to 0.99), default 0.95
        
    Returns:
    --------
    dict with keys:
        - 'p_est': Point estimate of probability
        - 'ci_lower': Lower bound of confidence interval
        - 'ci_upper': Upper bound of confidence interval
        - 'n': Sample size
        - 'method': 'wilson_score'
        
    Raises:
    -------
    ValueError: If conf_level not in valid range or total <= 0
    
    Examples:
    ---------
    >>> result = binomial_normal_ci(70, 100, 0.95)
    >>> print(f"Win rate: {result['p_est']:.2%} [{result['ci_lower']:.2%}, {result['ci_upper']:.2%}]")
    """
    # Validate inputs
    if not 0 < conf_level < 1:
        raise ValueError(f"Confidence level must be between 0 and 1, got {conf_level}")
    
    if total <= 0:
        raise ValueError(f"Total must be positive, got {total}")
    
    if successes < 0 or successes > total:
        raise ValueError(f"Successes must be between 0 and total, got {successes} out of {total}")
    
    # Handle edge case: no data
    if total == 0:
        return {
            'p_est': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'n': 0,
            'method': 'wilson_score'
        }
    
    # Calculate point estimate
    p = successes / total
    
    # Get z-score for confidence level
    alpha = 1 - conf_level
    z = stats.norm.ppf(1 - alpha / 2)
    z_squared = z ** 2
    
    # Wilson Score interval formula
    denominator = 1 + z_squared / total
    center = (p + z_squared / (2 * total)) / denominator
    margin = z * np.sqrt((p * (1 - p) / total + z_squared / (4 * total ** 2))) / denominator
    
    ci_lower = max(0.0, center - margin)
    ci_upper = min(1.0, center + margin)
    
    return {
        'p_est': p,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n': total,
        'method': 'wilson_score'
    }


def beta_posterior_ci(successes: int, total: int, alpha0: float = 1.0, beta0: float = 1.0, 
                      conf_level: float = 0.95) -> Dict[str, float]:
    """
    Calculate Bayesian confidence interval using Beta posterior distribution.
    
    This method uses Bayesian inference with a Beta prior to estimate probability.
    The Beta distribution is the conjugate prior for the binomial likelihood,
    making the posterior also Beta distributed.
    
    Parameters:
    -----------
    successes : int
        Number of successful outcomes (wins)
    total : int
        Total number of trials (trades)
    alpha0 : float
        Prior alpha parameter (default 1.0 for uniform prior)
    beta0 : float
        Prior beta parameter (default 1.0 for uniform prior)
    conf_level : float
        Confidence level (0.8 to 0.99), default 0.95
        
    Returns:
    --------
    dict with keys:
        - 'p_mean': Posterior mean (point estimate)
        - 'p_mode': Posterior mode (MAP estimate)
        - 'ci_lower': Lower bound of credible interval
        - 'ci_upper': Upper bound of credible interval
        - 'posterior_std': Standard deviation of posterior
        - 'n': Sample size
        - 'method': 'beta_posterior'
        
    Raises:
    -------
    ValueError: If conf_level not in valid range or parameters invalid
    
    Notes:
    ------
    - alpha0=1, beta0=1 gives uniform prior (non-informative)
    - alpha0=0.5, beta0=0.5 gives Jeffreys prior (scale-invariant)
    - Larger alpha0, beta0 values give more informative priors
    
    Examples:
    ---------
    >>> result = beta_posterior_ci(7, 10, conf_level=0.95)
    >>> print(f"Posterior mean: {result['p_mean']:.2%}")
    >>> print(f"95% Credible interval: [{result['ci_lower']:.2%}, {result['ci_upper']:.2%}]")
    """
    # Validate inputs
    if not 0 < conf_level < 1:
        raise ValueError(f"Confidence level must be between 0 and 1, got {conf_level}")
    
    if total < 0:
        raise ValueError(f"Total must be non-negative, got {total}")
    
    if successes < 0 or successes > total:
        raise ValueError(f"Successes must be between 0 and total, got {successes} out of {total}")
    
    if alpha0 <= 0 or beta0 <= 0:
        raise ValueError(f"Prior parameters must be positive, got alpha0={alpha0}, beta0={beta0}")
    
    # Calculate posterior parameters
    failures = total - successes
    alpha_post = alpha0 + successes
    beta_post = beta0 + failures
    
    # Posterior mean
    p_mean = alpha_post / (alpha_post + beta_post)
    
    # Posterior mode (MAP estimate)
    if alpha_post > 1 and beta_post > 1:
        p_mode = (alpha_post - 1) / (alpha_post + beta_post - 2)
    else:
        p_mode = p_mean  # Use mean if mode undefined
    
    # Posterior standard deviation
    posterior_std = np.sqrt(
        (alpha_post * beta_post) / 
        ((alpha_post + beta_post) ** 2 * (alpha_post + beta_post + 1))
    )
    
    # Credible interval (equal-tailed)
    alpha = 1 - conf_level
    ci_lower = stats.beta.ppf(alpha / 2, alpha_post, beta_post)
    ci_upper = stats.beta.ppf(1 - alpha / 2, alpha_post, beta_post)
    
    return {
        'p_mean': p_mean,
        'p_mode': p_mode,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'posterior_std': posterior_std,
        'n': total,
        'method': 'beta_posterior'
    }


def bootstrap_ci(data: np.ndarray, statistic_func: Callable[[np.ndarray], float],
                 conf_level: float = 0.95, n_bootstrap: int = 10000, 
                 random_seed: int = None) -> Dict[str, float]:
    """
    Calculate confidence interval using bootstrap resampling method.
    
    Bootstrap is a non-parametric method that doesn't assume any distribution.
    It works by resampling the data with replacement and calculating the
    statistic on each resample.
    
    Parameters:
    -----------
    data : np.ndarray
        Array of data values (e.g., R-multiples, binary outcomes)
    statistic_func : Callable
        Function that takes data array and returns a scalar statistic
        Example: lambda x: np.mean(x) for mean
    conf_level : float
        Confidence level (0.8 to 0.99), default 0.95
    n_bootstrap : int
        Number of bootstrap iterations (default 10000)
    random_seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    dict with keys:
        - 'point_estimate': Statistic calculated on original data
        - 'ci_lower': Lower bound of confidence interval
        - 'ci_upper': Upper bound of confidence interval
        - 'bootstrap_std': Standard deviation of bootstrap distribution
        - 'n': Sample size
        - 'n_bootstrap': Number of bootstrap iterations
        - 'method': 'bootstrap'
        
    Raises:
    -------
    ValueError: If conf_level not in valid range or data is empty
    
    Examples:
    ---------
    >>> data = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 1])  # Binary outcomes
    >>> result = bootstrap_ci(data, lambda x: np.mean(x), conf_level=0.95)
    >>> print(f"Win rate: {result['point_estimate']:.2%}")
    >>> print(f"95% CI: [{result['ci_lower']:.2%}, {result['ci_upper']:.2%}]")
    
    >>> # For R-multiples
    >>> r_data = np.array([1.5, -1.0, 2.3, 0.8, -0.5, 1.2])
    >>> result = bootstrap_ci(r_data, lambda x: np.mean(x), conf_level=0.95)
    >>> print(f"Mean R: {result['point_estimate']:.2f}")
    """
    # Validate inputs
    if not 0 < conf_level < 1:
        raise ValueError(f"Confidence level must be between 0 and 1, got {conf_level}")
    
    if len(data) == 0:
        raise ValueError("Data array cannot be empty")
    
    if n_bootstrap < 100:
        raise ValueError(f"n_bootstrap should be at least 100, got {n_bootstrap}")
    
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Calculate point estimate on original data
    point_estimate = statistic_func(data)
    
    # Perform bootstrap resampling
    n = len(data)
    bootstrap_estimates = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        # Resample with replacement
        resample = np.random.choice(data, size=n, replace=True)
        bootstrap_estimates[i] = statistic_func(resample)
    
    # Calculate confidence interval using percentile method
    alpha = 1 - conf_level
    ci_lower = np.percentile(bootstrap_estimates, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2))
    
    # Calculate bootstrap standard deviation
    bootstrap_std = np.std(bootstrap_estimates, ddof=1)
    
    return {
        'point_estimate': point_estimate,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'bootstrap_std': bootstrap_std,
        'n': n,
        'n_bootstrap': n_bootstrap,
        'method': 'bootstrap'
    }


# Convenience functions for common use cases

def win_rate_ci(wins: int, total: int, conf_level: float = 0.95, 
                method: str = 'wilson') -> Dict[str, float]:
    """
    Calculate confidence interval for win rate.
    
    Parameters:
    -----------
    wins : int
        Number of winning trades
    total : int
        Total number of trades
    conf_level : float
        Confidence level (default 0.95)
    method : str
        Method to use: 'wilson', 'beta', or 'bootstrap'
        
    Returns:
    --------
    dict with confidence interval
    """
    if method == 'wilson':
        return binomial_normal_ci(wins, total, conf_level)
    elif method == 'beta':
        return beta_posterior_ci(wins, total, conf_level=conf_level)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'wilson' or 'beta'")


def expectancy_ci(r_multiples: np.ndarray, conf_level: float = 0.95) -> Dict[str, float]:
    """
    Calculate confidence interval for expectancy (mean R-multiple).
    
    Parameters:
    -----------
    r_multiples : np.ndarray
        Array of R-multiple values
    conf_level : float
        Confidence level (default 0.95)
        
    Returns:
    --------
    dict with confidence interval for mean R
    """
    return bootstrap_ci(r_multiples, lambda x: np.mean(x), conf_level)
