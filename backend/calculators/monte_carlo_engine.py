"""
Monte Carlo Simulation Engine

This module provides Monte Carlo simulation capabilities for trading strategy analysis,
including equity curve generation, risk of ruin calculation, and Kelly Criterion optimization.

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple


def monte_carlo_simulation(
    df: pd.DataFrame,
    n_simulations: int = 1000,
    initial_equity: float = 10000.0,
    risk_per_trade: float = 0.01,
    r_column: str = 'R_multiple',
    max_trades_per_sim: Optional[int] = None,
    random_seed: Optional[int] = None
) -> Dict[str, any]:
    """
    Run Monte Carlo simulation by randomly sampling historical trades.
    
    Generates multiple equity curves by sampling trades with replacement,
    applying position sizing, and calculating performance metrics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Historical trades data with R_multiple column
    n_simulations : int
        Number of simulation runs (default: 1000)
    initial_equity : float
        Starting equity for each simulation (default: 10000)
    risk_per_trade : float
        Risk per trade as fraction of equity (default: 0.01 = 1%)
    r_column : str
        Column name for R-multiple values (default: 'R_multiple')
    max_trades_per_sim : int, optional
        Maximum trades per simulation (default: same as historical data length)
    random_seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    dict containing:
        - equity_curves: List of equity curves (each is list of equity values)
        - final_equity_distribution: List of final equity values
        - max_drawdown_distribution: List of maximum drawdown percentages
        - prob_ruin: Probability of ruin (equity drops below threshold)
        - prob_reach_target: Probability of reaching target (e.g., 2x initial)
        - median_final_equity: Median final equity across simulations
        - percentile_5_equity: 5th percentile final equity
        - percentile_95_equity: 95th percentile final equity
        - percentile_95_dd: 95th percentile maximum drawdown
        - n_simulations: Number of simulations run
        - initial_equity: Starting equity
        - risk_per_trade: Risk per trade used
        
    Validates: Requirements 5.1, 5.2
    
    Examples:
    ---------
    >>> result = monte_carlo_simulation(df, n_simulations=1000, risk_per_trade=0.01)
    >>> print(f"Median final equity: ${result['median_final_equity']:.2f}")
    >>> print(f"Probability of ruin: {result['prob_ruin']:.2%}")
    """
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if r_column not in df.columns:
        raise ValueError(f"R-multiple column '{r_column}' not found in DataFrame")
    
    if n_simulations < 1:
        raise ValueError(f"n_simulations must be >= 1, got {n_simulations}")
    
    if initial_equity <= 0:
        raise ValueError(f"initial_equity must be > 0, got {initial_equity}")
    
    if not 0 < risk_per_trade <= 1:
        raise ValueError(f"risk_per_trade must be between 0 and 1, got {risk_per_trade}")
    
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Get R-multiple values
    r_values = df[r_column].dropna().values
    
    if len(r_values) == 0:
        raise ValueError("No valid R-multiple values found")
    
    # Determine number of trades per simulation
    if max_trades_per_sim is None:
        max_trades_per_sim = len(r_values)
    
    # Run simulations
    equity_curves = []
    final_equities = []
    max_drawdowns = []
    
    ruin_threshold = initial_equity * 0.5  # 50% drawdown = ruin
    target_equity = initial_equity * 2.0   # 2x initial = target
    
    n_ruin = 0
    n_reach_target = 0
    
    for sim_idx in range(n_simulations):
        # Sample trades with replacement
        sampled_r = np.random.choice(r_values, size=max_trades_per_sim, replace=True)
        
        # Generate equity curve
        equity_curve = [initial_equity]
        equity = initial_equity
        peak_equity = initial_equity
        max_dd = 0.0
        
        for r in sampled_r:
            # Calculate position size based on risk
            risk_amount = equity * risk_per_trade
            
            # Calculate profit/loss
            pnl = risk_amount * r
            
            # Update equity
            equity = equity + pnl
            equity_curve.append(equity)
            
            # Track peak and drawdown
            if equity > peak_equity:
                peak_equity = equity
            
            current_dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
            if current_dd > max_dd:
                max_dd = current_dd
            
            # Check for ruin
            if equity <= ruin_threshold:
                n_ruin += 1
                break
        
        # Check if reached target
        if equity >= target_equity:
            n_reach_target += 1
        
        equity_curves.append(equity_curve)
        final_equities.append(equity)
        max_drawdowns.append(max_dd * 100)  # Convert to percentage
    
    # Calculate statistics
    final_equities_array = np.array(final_equities)
    max_drawdowns_array = np.array(max_drawdowns)
    
    result = {
        'equity_curves': equity_curves,
        'final_equity_distribution': final_equities,
        'max_drawdown_distribution': max_drawdowns,
        'prob_ruin': n_ruin / n_simulations,
        'prob_reach_target': n_reach_target / n_simulations,
        'median_final_equity': float(np.median(final_equities_array)),
        'percentile_5_equity': float(np.percentile(final_equities_array, 5)),
        'percentile_95_equity': float(np.percentile(final_equities_array, 95)),
        'percentile_95_dd': float(np.percentile(max_drawdowns_array, 95)),
        'n_simulations': n_simulations,
        'initial_equity': initial_equity,
        'risk_per_trade': risk_per_trade
    }
    
    return result


def calculate_percentile_bands(
    equity_curves: List[List[float]],
    percentiles: Optional[List[float]] = None
) -> Dict[str, List[float]]:
    """
    Calculate percentile bands for equity curve fan chart.
    
    For each time step, calculates specified percentiles across all simulations.
    
    Parameters:
    -----------
    equity_curves : list of lists
        List of equity curves from Monte Carlo simulation
    percentiles : list of float, optional
        Percentiles to calculate (default: [5, 25, 50, 75, 95])
        
    Returns:
    --------
    dict with percentile bands:
        - p5: 5th percentile equity at each time step
        - p25: 25th percentile equity at each time step
        - p50: 50th percentile (median) equity at each time step
        - p75: 75th percentile equity at each time step
        - p95: 95th percentile equity at each time step
        - time_steps: Number of time steps
        
    Validates: Requirements 5.3
    
    Examples:
    ---------
    >>> mc_result = monte_carlo_simulation(df, n_simulations=1000)
    >>> bands = calculate_percentile_bands(mc_result['equity_curves'])
    >>> # Plot fan chart with bands
    """
    if not equity_curves:
        raise ValueError("equity_curves list is empty")
    
    if percentiles is None:
        percentiles = [5, 25, 50, 75, 95]
    
    # Find the minimum length across all curves (in case some terminated early)
    min_length = min(len(curve) for curve in equity_curves)
    
    # Truncate all curves to minimum length
    truncated_curves = [curve[:min_length] for curve in equity_curves]
    
    # Convert to numpy array for easier percentile calculation
    curves_array = np.array(truncated_curves)  # Shape: (n_simulations, time_steps)
    
    # Calculate percentiles at each time step
    result = {}
    for p in percentiles:
        percentile_values = np.percentile(curves_array, p, axis=0)
        result[f'p{int(p)}'] = percentile_values.tolist()
    
    result['time_steps'] = min_length
    
    return result


def kelly_criterion_calculator(
    df: pd.DataFrame,
    r_column: str = 'R_multiple'
) -> Dict[str, float]:
    """
    Calculate Kelly Criterion for optimal position sizing.
    
    Kelly Criterion formula: f* = (p * b - q) / b
    where:
    - p = win rate
    - q = loss rate (1 - p)
    - b = average win / average loss ratio
    
    Parameters:
    -----------
    df : pd.DataFrame
        Historical trades data with R_multiple column
    r_column : str
        Column name for R-multiple values (default: 'R_multiple')
        
    Returns:
    --------
    dict containing:
        - full_kelly: Full Kelly percentage (can be > 1)
        - half_kelly: Half Kelly percentage (full_kelly / 2)
        - quarter_kelly: Quarter Kelly percentage (full_kelly / 4)
        - win_rate: Win rate used in calculation
        - avg_win: Average win R-multiple
        - avg_loss: Average loss R-multiple (absolute value)
        - win_loss_ratio: Ratio of avg_win to avg_loss
        
    Validates: Requirements 5.5
    
    Examples:
    ---------
    >>> kelly = kelly_criterion_calculator(df)
    >>> print(f"Full Kelly: {kelly['full_kelly']:.2%}")
    >>> print(f"Half Kelly: {kelly['half_kelly']:.2%}")
    """
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if r_column not in df.columns:
        raise ValueError(f"R-multiple column '{r_column}' not found in DataFrame")
    
    r_values = df[r_column].dropna()
    
    if len(r_values) == 0:
        raise ValueError("No valid R-multiple values found")
    
    # Separate winners and losers
    winners = r_values[r_values > 0]
    losers = r_values[r_values < 0]
    
    if len(winners) == 0 or len(losers) == 0:
        # Cannot calculate Kelly if all wins or all losses
        return {
            'full_kelly': np.nan,
            'half_kelly': np.nan,
            'quarter_kelly': np.nan,
            'win_rate': len(winners) / len(r_values) if len(r_values) > 0 else np.nan,
            'avg_win': float(winners.mean()) if len(winners) > 0 else np.nan,
            'avg_loss': float(abs(losers.mean())) if len(losers) > 0 else np.nan,
            'win_loss_ratio': np.nan
        }
    
    # Calculate components
    win_rate = len(winners) / len(r_values)
    loss_rate = 1 - win_rate
    avg_win = float(winners.mean())
    avg_loss = float(abs(losers.mean()))
    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else np.nan
    
    # Kelly formula: f* = (p * b - q) / b
    # where b = win/loss ratio
    if not np.isnan(win_loss_ratio) and win_loss_ratio > 0:
        full_kelly = (win_rate * win_loss_ratio - loss_rate) / win_loss_ratio
        
        # Ensure Kelly is between 0 and 1
        full_kelly = max(0.0, min(1.0, full_kelly))
    else:
        full_kelly = 0.0
    
    return {
        'full_kelly': full_kelly,
        'half_kelly': full_kelly / 2,
        'quarter_kelly': full_kelly / 4,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'win_loss_ratio': win_loss_ratio
    }


def compare_risk_scenarios(
    df: pd.DataFrame,
    risk_levels: Optional[List[float]] = None,
    n_simulations: int = 1000,
    initial_equity: float = 10000.0,
    r_column: str = 'R_multiple',
    random_seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Compare multiple risk per trade scenarios using Monte Carlo simulation.
    
    Runs Monte Carlo simulations for different risk levels and compares results.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Historical trades data with R_multiple column
    risk_levels : list of float, optional
        Risk per trade levels to compare (default: [0.005, 0.01, 0.015, 0.02])
    n_simulations : int
        Number of simulations per risk level (default: 1000)
    initial_equity : float
        Starting equity for each simulation (default: 10000)
    r_column : str
        Column name for R-multiple values (default: 'R_multiple')
    random_seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame with columns:
        - risk_per_trade: Risk level
        - median_final_equity: Median final equity
        - percentile_5_equity: 5th percentile final equity
        - percentile_95_equity: 95th percentile final equity
        - prob_ruin: Probability of ruin
        - prob_reach_target: Probability of reaching 2x initial equity
        - percentile_95_dd: 95th percentile maximum drawdown
        - expected_return: Expected return percentage
        
    Validates: Requirements 5.4
    
    Examples:
    ---------
    >>> comparison = compare_risk_scenarios(df, risk_levels=[0.005, 0.01, 0.02])
    >>> print(comparison[['risk_per_trade', 'median_final_equity', 'prob_ruin']])
    """
    if risk_levels is None:
        risk_levels = [0.005, 0.01, 0.015, 0.02]  # 0.5%, 1%, 1.5%, 2%
    
    results = []
    
    for risk_level in risk_levels:
        # Run Monte Carlo simulation for this risk level
        mc_result = monte_carlo_simulation(
            df=df,
            n_simulations=n_simulations,
            initial_equity=initial_equity,
            risk_per_trade=risk_level,
            r_column=r_column,
            random_seed=random_seed
        )
        
        # Calculate expected return
        expected_return = ((mc_result['median_final_equity'] - initial_equity) / initial_equity) * 100
        
        results.append({
            'risk_per_trade': risk_level,
            'risk_percent': risk_level * 100,
            'median_final_equity': mc_result['median_final_equity'],
            'percentile_5_equity': mc_result['percentile_5_equity'],
            'percentile_95_equity': mc_result['percentile_95_equity'],
            'prob_ruin': mc_result['prob_ruin'],
            'prob_reach_target': mc_result['prob_reach_target'],
            'percentile_95_dd': mc_result['percentile_95_dd'],
            'expected_return': expected_return
        })
    
    result_df = pd.DataFrame(results)
    
    # Sort by risk level
    result_df = result_df.sort_values('risk_per_trade').reset_index(drop=True)
    
    return result_df


def generate_equity_curve_with_position_sizing(
    r_values: np.ndarray,
    initial_equity: float = 10000.0,
    risk_per_trade: float = 0.01,
    compounding: bool = True
) -> Tuple[List[float], float]:
    """
    Generate a single equity curve with position sizing.
    
    Helper function to generate equity curve from R-multiple sequence.
    
    Parameters:
    -----------
    r_values : np.ndarray
        Sequence of R-multiple values
    initial_equity : float
        Starting equity (default: 10000)
    risk_per_trade : float
        Risk per trade as fraction of equity (default: 0.01)
    compounding : bool
        If True, risk is calculated on current equity (compounding)
        If False, risk is calculated on initial equity (fixed)
        
    Returns:
    --------
    tuple of (equity_curve, max_drawdown):
        - equity_curve: List of equity values at each step
        - max_drawdown: Maximum drawdown percentage
        
    Examples:
    ---------
    >>> r_sequence = np.array([1.5, -1.0, 2.0, -0.5, 1.0])
    >>> curve, max_dd = generate_equity_curve_with_position_sizing(
    ...     r_sequence, initial_equity=10000, risk_per_trade=0.01
    ... )
    """
    equity_curve = [initial_equity]
    equity = initial_equity
    peak_equity = initial_equity
    max_dd = 0.0
    
    for r in r_values:
        # Calculate position size
        if compounding:
            risk_amount = equity * risk_per_trade
        else:
            risk_amount = initial_equity * risk_per_trade
        
        # Calculate profit/loss
        pnl = risk_amount * r
        
        # Update equity
        equity = equity + pnl
        equity_curve.append(equity)
        
        # Track peak and drawdown
        if equity > peak_equity:
            peak_equity = equity
        
        current_dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
        if current_dd > max_dd:
            max_dd = current_dd
    
    return equity_curve, max_dd * 100  # Return DD as percentage
