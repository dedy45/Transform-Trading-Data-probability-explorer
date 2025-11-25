"""
Sequential Analysis Module (Markov Chains & Streak Analysis)

This module provides functions for analyzing sequential patterns in trading outcomes,
including Markov chain transition probabilities, win/loss streak distributions,
and conditional probabilities based on recent trading history.

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from backend.models.confidence_intervals import beta_posterior_ci


def compute_first_order_markov(
    df: pd.DataFrame,
    target_column: str = 'trade_success',
    conf_level: float = 0.95
) -> Dict:
    """
    Calculate first-order Markov transition matrix for win/loss sequences.
    
    Computes transition probabilities:
    - P(Win | Win): Probability of winning after a win
    - P(Loss | Win): Probability of losing after a win
    - P(Win | Loss): Probability of winning after a loss
    - P(Loss | Loss): Probability of losing after a loss
    
    Args:
        df: DataFrame containing trade data (must be sorted by time)
        target_column: Name of the target column (1 for win, 0 for loss)
        conf_level: Confidence level for intervals (default: 0.95)
    
    Returns:
        Dictionary containing:
        - probs: Dict with transition probabilities
        - counts: Dict with transition counts
        - ci: Dict with confidence intervals for each transition
        - n_transitions: Total number of transitions analyzed
    
    Validates: Requirements 8.1
    """
    if df.empty or len(df) < 2:
        return {
            'probs': {
                'P_win_given_win': np.nan,
                'P_loss_given_win': np.nan,
                'P_win_given_loss': np.nan,
                'P_loss_given_loss': np.nan
            },
            'counts': {
                'win_to_win': 0,
                'win_to_loss': 0,
                'loss_to_win': 0,
                'loss_to_loss': 0
            },
            'ci': {},
            'n_transitions': 0
        }
    
    # Get outcomes as array
    outcomes = df[target_column].values
    
    # Count transitions
    win_to_win = 0
    win_to_loss = 0
    loss_to_win = 0
    loss_to_loss = 0
    
    for i in range(len(outcomes) - 1):
        current = outcomes[i]
        next_outcome = outcomes[i + 1]
        
        if current == 1:  # Current is win
            if next_outcome == 1:
                win_to_win += 1
            else:
                win_to_loss += 1
        else:  # Current is loss
            if next_outcome == 1:
                loss_to_win += 1
            else:
                loss_to_loss += 1
    
    # Calculate probabilities
    total_after_win = win_to_win + win_to_loss
    total_after_loss = loss_to_win + loss_to_loss
    
    # P(Win | Win) and P(Loss | Win)
    if total_after_win > 0:
        p_win_given_win = win_to_win / total_after_win
        p_loss_given_win = win_to_loss / total_after_win
        
        # Calculate CI for P(Win | Win)
        ci_win_win = beta_posterior_ci(
            successes=win_to_win,
            total=total_after_win,
            conf_level=conf_level
        )
        
        # Calculate CI for P(Loss | Win)
        ci_loss_win = beta_posterior_ci(
            successes=win_to_loss,
            total=total_after_win,
            conf_level=conf_level
        )
    else:
        p_win_given_win = np.nan
        p_loss_given_win = np.nan
        ci_win_win = {'ci_lower': np.nan, 'ci_upper': np.nan}
        ci_loss_win = {'ci_lower': np.nan, 'ci_upper': np.nan}
    
    # P(Win | Loss) and P(Loss | Loss)
    if total_after_loss > 0:
        p_win_given_loss = loss_to_win / total_after_loss
        p_loss_given_loss = loss_to_loss / total_after_loss
        
        # Calculate CI for P(Win | Loss)
        ci_win_loss = beta_posterior_ci(
            successes=loss_to_win,
            total=total_after_loss,
            conf_level=conf_level
        )
        
        # Calculate CI for P(Loss | Loss)
        ci_loss_loss = beta_posterior_ci(
            successes=loss_to_loss,
            total=total_after_loss,
            conf_level=conf_level
        )
    else:
        p_win_given_loss = np.nan
        p_loss_given_loss = np.nan
        ci_win_loss = {'ci_lower': np.nan, 'ci_upper': np.nan}
        ci_loss_loss = {'ci_lower': np.nan, 'ci_upper': np.nan}
    
    return {
        'probs': {
            'P_win_given_win': p_win_given_win,
            'P_loss_given_win': p_loss_given_win,
            'P_win_given_loss': p_win_given_loss,
            'P_loss_given_loss': p_loss_given_loss
        },
        'counts': {
            'win_to_win': win_to_win,
            'win_to_loss': win_to_loss,
            'loss_to_win': loss_to_win,
            'loss_to_loss': loss_to_loss
        },
        'ci': {
            'P_win_given_win': ci_win_win,
            'P_loss_given_win': ci_loss_win,
            'P_win_given_loss': ci_win_loss,
            'P_loss_given_loss': ci_loss_loss
        },
        'n_transitions': len(outcomes) - 1
    }


def compute_transition_ci(
    successes: int,
    total: int,
    conf_level: float = 0.95
) -> Dict:
    """
    Calculate confidence interval for a single transition probability.
    
    Args:
        successes: Number of times the transition occurred
        total: Total number of opportunities for this transition
        conf_level: Confidence level for interval (default: 0.95)
    
    Returns:
        Dictionary containing:
        - p_est: Point estimate of probability
        - ci_lower: Lower bound of confidence interval
        - ci_upper: Upper bound of confidence interval
        - n: Total number of observations
    
    Validates: Requirements 8.2
    """
    if total == 0:
        return {
            'p_est': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'n': 0
        }
    
    p_est = successes / total
    
    ci_result = beta_posterior_ci(
        successes=successes,
        total=total,
        conf_level=conf_level
    )
    
    return {
        'p_est': p_est,
        'ci_lower': ci_result['ci_lower'],
        'ci_upper': ci_result['ci_upper'],
        'n': total
    }


def compute_streak_distribution(
    df: pd.DataFrame,
    target_column: str = 'trade_success'
) -> Dict:
    """
    Calculate distribution of consecutive win and loss streaks.
    
    Args:
        df: DataFrame containing trade data (must be sorted by time)
        target_column: Name of the target column (1 for win, 0 for loss)
    
    Returns:
        Dictionary containing:
        - win_streaks: List of win streak lengths
        - loss_streaks: List of loss streak lengths
        - win_streak_distribution: Dict mapping streak length to count
        - loss_streak_distribution: Dict mapping streak length to count
        - max_win_streak: Maximum consecutive wins
        - max_loss_streak: Maximum consecutive losses
        - avg_win_streak: Average win streak length
        - avg_loss_streak: Average loss streak length
    
    Validates: Requirements 8.3
    """
    if df.empty:
        return {
            'win_streaks': [],
            'loss_streaks': [],
            'win_streak_distribution': {},
            'loss_streak_distribution': {},
            'max_win_streak': 0,
            'max_loss_streak': 0,
            'avg_win_streak': 0.0,
            'avg_loss_streak': 0.0
        }
    
    outcomes = df[target_column].values
    
    win_streaks = []
    loss_streaks = []
    
    current_streak = 1
    current_outcome = outcomes[0]
    
    for i in range(1, len(outcomes)):
        if outcomes[i] == current_outcome:
            # Continue current streak
            current_streak += 1
        else:
            # Streak ended, record it
            if current_outcome == 1:
                win_streaks.append(current_streak)
            else:
                loss_streaks.append(current_streak)
            
            # Start new streak
            current_outcome = outcomes[i]
            current_streak = 1
    
    # Record the final streak
    if current_outcome == 1:
        win_streaks.append(current_streak)
    else:
        loss_streaks.append(current_streak)
    
    # Create distribution dictionaries
    win_streak_dist = {}
    for streak in win_streaks:
        win_streak_dist[streak] = win_streak_dist.get(streak, 0) + 1
    
    loss_streak_dist = {}
    for streak in loss_streaks:
        loss_streak_dist[streak] = loss_streak_dist.get(streak, 0) + 1
    
    # Calculate statistics
    max_win_streak = max(win_streaks) if win_streaks else 0
    max_loss_streak = max(loss_streaks) if loss_streaks else 0
    avg_win_streak = np.mean(win_streaks) if win_streaks else 0.0
    avg_loss_streak = np.mean(loss_streaks) if loss_streaks else 0.0
    
    return {
        'win_streaks': win_streaks,
        'loss_streaks': loss_streaks,
        'win_streak_distribution': win_streak_dist,
        'loss_streak_distribution': loss_streak_dist,
        'max_win_streak': max_win_streak,
        'max_loss_streak': max_loss_streak,
        'avg_win_streak': float(avg_win_streak),
        'avg_loss_streak': float(avg_loss_streak)
    }


def compute_winrate_given_loss_streak(
    df: pd.DataFrame,
    target_column: str = 'trade_success',
    max_streak: int = 10,
    conf_level: float = 0.95,
    min_samples: int = 5
) -> pd.DataFrame:
    """
    Calculate P(Win | loss_streak = k) for various loss streak lengths.
    
    Args:
        df: DataFrame containing trade data (must be sorted by time)
        target_column: Name of the target column (1 for win, 0 for loss)
        max_streak: Maximum loss streak length to analyze (default: 10)
        conf_level: Confidence level for intervals (default: 0.95)
        min_samples: Minimum samples required for reliable estimate (default: 5)
    
    Returns:
        DataFrame with columns:
        - loss_streak_length: Length of preceding loss streak (0 to max_streak)
        - n_opportunities: Number of times this streak length occurred
        - n_wins: Number of wins after this streak length
        - win_rate: P(Win | loss_streak = k)
        - ci_lower: Lower bound of confidence interval
        - ci_upper: Upper bound of confidence interval
        - reliable: Boolean indicating if sample size >= min_samples
    
    Validates: Requirements 8.4
    """
    if df.empty or len(df) < 2:
        # Return empty rows for all streak lengths
        results = []
        for k in range(max_streak + 1):
            results.append({
                'loss_streak_length': k,
                'n_opportunities': 0,
                'n_wins': 0,
                'win_rate': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'reliable': False
            })
        return pd.DataFrame(results)
    
    outcomes = df[target_column].values
    
    # Track loss streak before each trade
    results = []
    
    for k in range(max_streak + 1):
        n_opportunities = 0
        n_wins = 0
        
        # Scan through outcomes looking for loss streaks of length k
        current_loss_streak = 0
        
        for i in range(len(outcomes)):
            if i == 0:
                # First trade has no history
                if outcomes[i] == 0:
                    current_loss_streak = 1
                else:
                    current_loss_streak = 0
                continue
            
            # Check if previous trade was a loss
            if outcomes[i - 1] == 0:
                current_loss_streak += 1
            else:
                current_loss_streak = 0
            
            # If we have exactly k consecutive losses before this trade
            if current_loss_streak == k:
                n_opportunities += 1
                if outcomes[i] == 1:
                    n_wins += 1
        
        # Calculate win rate and CI
        if n_opportunities > 0:
            win_rate = n_wins / n_opportunities
            
            ci_result = beta_posterior_ci(
                successes=n_wins,
                total=n_opportunities,
                conf_level=conf_level
            )
            
            results.append({
                'loss_streak_length': k,
                'n_opportunities': n_opportunities,
                'n_wins': n_wins,
                'win_rate': win_rate,
                'ci_lower': ci_result['ci_lower'],
                'ci_upper': ci_result['ci_upper'],
                'reliable': n_opportunities >= min_samples
            })
        else:
            results.append({
                'loss_streak_length': k,
                'n_opportunities': 0,
                'n_wins': 0,
                'win_rate': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'reliable': False
            })
    
    result_df = pd.DataFrame(results)
    return result_df


def find_max_streaks(
    df: pd.DataFrame,
    target_column: str = 'trade_success'
) -> Dict:
    """
    Find maximum win and loss streaks in the trading history.
    
    Args:
        df: DataFrame containing trade data (must be sorted by time)
        target_column: Name of the target column (1 for win, 0 for loss)
    
    Returns:
        Dictionary containing:
        - max_win_streak: Maximum consecutive wins
        - max_loss_streak: Maximum consecutive losses
        - max_win_streak_start_idx: Index where max win streak started
        - max_loss_streak_start_idx: Index where max loss streak started
        - current_streak: Current streak (positive for wins, negative for losses)
        - current_streak_length: Absolute length of current streak
    
    Validates: Requirements 8.5
    """
    if df.empty:
        return {
            'max_win_streak': 0,
            'max_loss_streak': 0,
            'max_win_streak_start_idx': None,
            'max_loss_streak_start_idx': None,
            'current_streak': 0,
            'current_streak_length': 0
        }
    
    outcomes = df[target_column].values
    
    max_win_streak = 0
    max_loss_streak = 0
    max_win_streak_start_idx = None
    max_loss_streak_start_idx = None
    
    current_streak_length = 1
    current_outcome = outcomes[0]
    current_streak_start_idx = 0
    
    for i in range(1, len(outcomes)):
        if outcomes[i] == current_outcome:
            # Continue current streak
            current_streak_length += 1
        else:
            # Streak ended, check if it's a record
            if current_outcome == 1 and current_streak_length > max_win_streak:
                max_win_streak = current_streak_length
                max_win_streak_start_idx = current_streak_start_idx
            elif current_outcome == 0 and current_streak_length > max_loss_streak:
                max_loss_streak = current_streak_length
                max_loss_streak_start_idx = current_streak_start_idx
            
            # Start new streak
            current_outcome = outcomes[i]
            current_streak_length = 1
            current_streak_start_idx = i
    
    # Check the final streak
    if current_outcome == 1 and current_streak_length > max_win_streak:
        max_win_streak = current_streak_length
        max_win_streak_start_idx = current_streak_start_idx
    elif current_outcome == 0 and current_streak_length > max_loss_streak:
        max_loss_streak = current_streak_length
        max_loss_streak_start_idx = current_streak_start_idx
    
    # Determine current streak (positive for wins, negative for losses)
    if current_outcome == 1:
        current_streak = current_streak_length
    else:
        current_streak = -current_streak_length
    
    return {
        'max_win_streak': max_win_streak,
        'max_loss_streak': max_loss_streak,
        'max_win_streak_start_idx': max_win_streak_start_idx,
        'max_loss_streak_start_idx': max_loss_streak_start_idx,
        'current_streak': current_streak,
        'current_streak_length': current_streak_length
    }
