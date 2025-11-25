"""
Trade Analysis Dashboard Backend Module

This module provides comprehensive trade analysis functions for the dashboard,
including summary metrics, equity curves, R-multiple analysis, MAE/MFE analysis,
time-based performance, trade type analysis, consecutive analysis, and risk metrics.

Requirements: 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any


def calculate_summary_metrics(trades_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate 6 summary metrics for dashboard cards.
    
    Args:
        trades_df: DataFrame with trade data including R_multiple, trade_success, 
                   gross_profit, net_profit columns
    
    Returns:
        Dictionary with:
        - total_trades: int
        - win_rate: float (0-1)
        - avg_r: float
        - expectancy: float (in dollars)
        - max_drawdown: float (as percentage, negative)
        - profit_factor: float
    
    Requirements: 0.2
    """
    if trades_df.empty:
        return {
            'total_trades': 0,
            'win_rate': 0.0,
            'avg_r': 0.0,
            'expectancy': 0.0,
            'max_drawdown': 0.0,
            'profit_factor': 0.0
        }
    
    total_trades = len(trades_df)
    
    # Win rate
    wins = trades_df['trade_success'].sum()
    win_rate = wins / total_trades if total_trades > 0 else 0.0
    
    # Average R-multiple
    avg_r = trades_df['R_multiple'].mean()
    
    # Expectancy (in dollars)
    expectancy = trades_df['net_profit'].mean()
    
    # Maximum drawdown
    if 'equity_after_trade' in trades_df.columns:
        equity_curve = trades_df['equity_after_trade'].values
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - running_max) / running_max
        max_drawdown = drawdowns.min() * 100  # Convert to percentage
    else:
        # Calculate from cumulative profit
        cumulative_profit = trades_df['net_profit'].cumsum()
        running_max = np.maximum.accumulate(cumulative_profit)
        drawdowns = cumulative_profit - running_max
        initial_equity = 10000  # Assume initial equity
        max_drawdown = (drawdowns.min() / initial_equity) * 100
    
    # Profit factor
    gross_wins = trades_df[trades_df['net_profit'] > 0]['net_profit'].sum()
    gross_losses = abs(trades_df[trades_df['net_profit'] < 0]['net_profit'].sum())
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_r': avg_r,
        'expectancy': expectancy,
        'max_drawdown': max_drawdown,
        'profit_factor': profit_factor
    }


def calculate_equity_curve(trades_df: pd.DataFrame, 
                          initial_equity: float = 10000) -> Dict[str, Any]:
    """
    Calculate equity curve with drawdown detection.
    
    Args:
        trades_df: DataFrame with trade data including timestamps and net_profit
        initial_equity: Starting equity amount
    
    Returns:
        Dictionary with:
        - timestamps: List of datetime
        - equity_values: List of equity at each trade
        - drawdown_periods: List of tuples (start_idx, end_idx, depth)
        - cumulative_profit: List of cumulative profit
    
    Requirements: 0.3
    """
    if trades_df.empty:
        return {
            'timestamps': [],
            'equity_values': [],
            'drawdown_periods': [],
            'cumulative_profit': []
        }
    
    # Sort by timestamp
    df_sorted = trades_df.sort_values('exit_time' if 'exit_time' in trades_df.columns else 'Timestamp')
    
    # Calculate equity curve
    cumulative_profit = df_sorted['net_profit'].cumsum()
    equity_values = initial_equity + cumulative_profit
    
    # Detect drawdown periods
    running_max = np.maximum.accumulate(equity_values)
    drawdowns = equity_values - running_max
    
    # Find drawdown periods (where drawdown < 0)
    in_drawdown = drawdowns < -0.01  # Small threshold to avoid noise
    drawdown_periods = []
    
    if in_drawdown.any():
        # Find start and end of each drawdown period
        drawdown_changes = np.diff(np.concatenate([[False], in_drawdown, [False]]).astype(int))
        starts = np.where(drawdown_changes == 1)[0]
        ends = np.where(drawdown_changes == -1)[0]
        
        for start, end in zip(starts, ends):
            # Skip single-point drawdowns
            if end - start <= 1:
                continue
            depth = drawdowns[start:end].min()
            depth_pct = (depth / running_max[start:end].max()) * 100
            drawdown_periods.append({
                'start_idx': int(start),
                'end_idx': int(end - 1),
                'depth': float(depth),
                'depth_pct': float(depth_pct)
            })
    
    timestamps = df_sorted['exit_time' if 'exit_time' in df_sorted.columns else 'Timestamp'].tolist()
    
    return {
        'timestamps': timestamps,
        'equity_values': equity_values.tolist(),
        'drawdown_periods': drawdown_periods,
        'cumulative_profit': cumulative_profit.tolist()
    }


def calculate_r_distribution(trades_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate R-multiple distribution with statistics.
    
    Args:
        trades_df: DataFrame with R_multiple column
    
    Returns:
        Dictionary with:
        - histogram_data: Dict with bins and counts
        - statistics: Dict with mean, median, percentiles
        - threshold_probs: Dict with P(R>1), P(R>2), P(R>3)
        - best_r: float
        - worst_r: float
    
    Requirements: 0.4
    """
    if trades_df.empty or 'R_multiple' not in trades_df.columns:
        return {
            'histogram_data': {'bins': [], 'counts': []},
            'statistics': {},
            'threshold_probs': {},
            'best_r': 0.0,
            'worst_r': 0.0
        }
    
    r_values = trades_df['R_multiple'].dropna()
    
    if len(r_values) == 0:
        return {
            'histogram_data': {'bins': [], 'counts': []},
            'statistics': {},
            'threshold_probs': {},
            'best_r': 0.0,
            'worst_r': 0.0
        }
    
    # Histogram data
    counts, bin_edges = np.histogram(r_values, bins=30)
    
    # Statistics
    statistics = {
        'mean': float(r_values.mean()),
        'median': float(r_values.median()),
        'std': float(r_values.std()),
        'p25': float(r_values.quantile(0.25)),
        'p50': float(r_values.quantile(0.50)),
        'p75': float(r_values.quantile(0.75)),
        'p90': float(r_values.quantile(0.90)),
        'p95': float(r_values.quantile(0.95))
    }
    
    # Threshold probabilities
    total = len(r_values)
    threshold_probs = {
        'p_r_gt_1': float((r_values > 1).sum() / total),
        'p_r_gt_2': float((r_values > 2).sum() / total),
        'p_r_gt_3': float((r_values > 3).sum() / total)
    }
    
    return {
        'histogram_data': {
            'bins': bin_edges.tolist(),
            'counts': counts.tolist()
        },
        'statistics': statistics,
        'threshold_probs': threshold_probs,
        'best_r': float(r_values.max()),
        'worst_r': float(r_values.min())
    }


def calculate_mae_mfe_analysis(trades_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate MAE/MFE analysis for winners and losers.
    
    Args:
        trades_df: DataFrame with MAE_R, MFE_R, R_multiple, trade_success columns
    
    Returns:
        Dictionary with:
        - scatter_data: Dict with mae, mfe, r_multiple, is_winner arrays
        - winners_stats: Dict with avg_mae, avg_mfe, mfe_to_r_ratio
        - losers_stats: Dict with avg_mae, avg_mfe
        - mae_distribution: Dict with winners and losers histograms
        - correlation: float (MFE vs R correlation)
    
    Requirements: 0.5
    """
    if trades_df.empty:
        return {
            'scatter_data': {},
            'winners_stats': {},
            'losers_stats': {},
            'mae_distribution': {},
            'correlation': 0.0
        }
    
    # Filter valid data
    valid_df = trades_df.dropna(subset=['MAE_R', 'MFE_R', 'R_multiple', 'trade_success'])
    
    if valid_df.empty:
        return {
            'scatter_data': {},
            'winners_stats': {},
            'losers_stats': {},
            'mae_distribution': {},
            'correlation': 0.0
        }
    
    # Scatter data
    scatter_data = {
        'mae': valid_df['MAE_R'].tolist(),
        'mfe': valid_df['MFE_R'].tolist(),
        'r_multiple': valid_df['R_multiple'].tolist(),
        'is_winner': valid_df['trade_success'].tolist()
    }
    
    # Winners and losers
    winners = valid_df[valid_df['trade_success'] == 1]
    losers = valid_df[valid_df['trade_success'] == 0]
    
    # Winners stats
    if not winners.empty:
        winners_stats = {
            'avg_mae': float(winners['MAE_R'].mean()),
            'avg_mfe': float(winners['MFE_R'].mean()),
            'mfe_to_r_ratio': float(winners['MFE_R'].mean() / winners['R_multiple'].mean()) 
                             if winners['R_multiple'].mean() != 0 else 0.0,
            'profit_left': float((winners['MFE_R'] - winners['R_multiple']).mean())
        }
    else:
        winners_stats = {
            'avg_mae': 0.0,
            'avg_mfe': 0.0,
            'mfe_to_r_ratio': 0.0,
            'profit_left': 0.0
        }
    
    # Losers stats
    if not losers.empty:
        losers_stats = {
            'avg_mae': float(losers['MAE_R'].mean()),
            'avg_mfe': float(losers['MFE_R'].mean())
        }
    else:
        losers_stats = {
            'avg_mae': 0.0,
            'avg_mfe': 0.0
        }
    
    # MAE distribution
    mae_distribution = {}
    if not winners.empty:
        w_counts, w_bins = np.histogram(winners['MAE_R'], bins=20)
        mae_distribution['winners'] = {
            'bins': w_bins.tolist(),
            'counts': w_counts.tolist()
        }
    if not losers.empty:
        l_counts, l_bins = np.histogram(losers['MAE_R'], bins=20)
        mae_distribution['losers'] = {
            'bins': l_bins.tolist(),
            'counts': l_counts.tolist()
        }
    
    # Correlation
    correlation = float(valid_df[['MFE_R', 'R_multiple']].corr().iloc[0, 1])
    
    return {
        'scatter_data': scatter_data,
        'winners_stats': winners_stats,
        'losers_stats': losers_stats,
        'mae_distribution': mae_distribution,
        'correlation': correlation
    }



def calculate_time_based_performance(trades_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate time-based performance (hourly, daily, weekly, monthly, session).
    
    Args:
        trades_df: DataFrame with exit_time/Timestamp, trade_success, R_multiple columns
    
    Returns:
        Dictionary with:
        - hourly: Dict with hour -> {win_rate, avg_r, count}
        - daily: Dict with day_of_week -> {win_rate, avg_r, count}
        - weekly: Dict with year-week -> {win_rate, avg_r, count, total_profit}
        - monthly: Dict with year-month -> {win_rate, avg_r, count, total_profit}
        - session: Dict with session -> {win_rate, avg_r, count, total_profit}
          Sessions detected from hour:
          - TOKYO/ASIA: 00:00 - 09:00
          - LONDON/EUROPE: 08:00 - 17:00
          - US: 13:00 - 22:00
          - SYDNEY: 22:00 - 07:00
    
    Requirements: 0.6
    """
    if trades_df.empty:
        return {
            'hourly': {},
            'daily': {},
            'session': {}
        }
    
    # Determine timestamp column
    time_col = 'exit_time' if 'exit_time' in trades_df.columns else 'Timestamp'
    
    # Ensure datetime
    df = trades_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    
    # Extract time components
    df['hour'] = df[time_col].dt.hour
    df['day_of_week'] = df[time_col].dt.dayofweek  # 0=Monday, 6=Sunday
    
    # Hourly performance
    hourly = {}
    for hour in range(24):
        hour_trades = df[df['hour'] == hour]
        if not hour_trades.empty:
            hourly[hour] = {
                'win_rate': float(hour_trades['trade_success'].mean()),
                'avg_r': float(hour_trades['R_multiple'].mean()),
                'count': int(len(hour_trades))
            }
    
    # Daily performance
    daily = {}
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for day in range(7):
        day_trades = df[df['day_of_week'] == day]
        if not day_trades.empty:
            daily[day_names[day]] = {
                'win_rate': float(day_trades['trade_success'].mean()),
                'avg_r': float(day_trades['R_multiple'].mean()),
                'count': int(len(day_trades))
            }
    
    # Weekly performance
    df['week'] = df[time_col].dt.isocalendar().week
    df['year'] = df[time_col].dt.year
    df['year_week'] = df['year'].astype(str) + '-W' + df['week'].astype(str).str.zfill(2)
    
    weekly = {}
    for year_week in df['year_week'].unique():
        week_trades = df[df['year_week'] == year_week]
        if not week_trades.empty:
            weekly[year_week] = {
                'win_rate': float(week_trades['trade_success'].mean()),
                'avg_r': float(week_trades['R_multiple'].mean()),
                'count': int(len(week_trades)),
                'total_profit': float(week_trades['net_profit'].sum()) if 'net_profit' in week_trades.columns else 0.0
            }
    
    # Monthly performance
    df['month'] = df[time_col].dt.to_period('M').astype(str)
    
    monthly = {}
    for month in df['month'].unique():
        month_trades = df[df['month'] == month]
        if not month_trades.empty:
            monthly[month] = {
                'win_rate': float(month_trades['trade_success'].mean()),
                'avg_r': float(month_trades['R_multiple'].mean()),
                'count': int(len(month_trades)),
                'total_profit': float(month_trades['net_profit'].sum()) if 'net_profit' in month_trades.columns else 0.0
            }
    
    # Session performance
    # Define trading sessions based on hour (UTC/Server time)
    # Tokyo/Asia: 00:00 - 09:00
    # London/Europe: 08:00 - 17:00
    # US: 13:00 - 22:00
    # Sydney: 22:00 - 07:00 (next day)
    
    def get_trading_session(hour):
        """
        Determine trading session based on hour.
        Multiple sessions can overlap.
        Returns list of active sessions.
        """
        sessions = []
        
        # Tokyo/Asia: 00:00 - 09:00
        if 0 <= hour < 9:
            sessions.append('TOKYO/ASIA')
        
        # London/Europe: 08:00 - 17:00
        if 8 <= hour < 17:
            sessions.append('LONDON/EUROPE')
        
        # US: 13:00 - 22:00
        if 13 <= hour < 22:
            sessions.append('US')
        
        # Sydney: 22:00 - 07:00 (22:00-23:59 and 00:00-06:59)
        if hour >= 22 or hour < 7:
            sessions.append('SYDNEY')
        
        # Determine primary session (first in list)
        return sessions[0] if sessions else 'OTHER'
    
    # Apply session detection
    df['trading_session'] = df['hour'].apply(get_trading_session)
    
    # Calculate session performance
    session = {}
    for sess_name in ['TOKYO/ASIA', 'LONDON/EUROPE', 'US', 'SYDNEY']:
        sess_trades = df[df['trading_session'] == sess_name]
        if not sess_trades.empty:
            session[sess_name] = {
                'win_rate': float(sess_trades['trade_success'].mean()),
                'avg_r': float(sess_trades['R_multiple'].mean()),
                'count': int(len(sess_trades)),
                'total_profit': float(sess_trades['net_profit'].sum()) if 'net_profit' in sess_trades.columns else 0.0
            }
    
    # Also check if there's an existing session column
    if 'session' in df.columns or 'entry_session' in df.columns:
        session_col = 'session' if 'session' in df.columns else 'entry_session'
        session_map = {0: 'ASIA', 1: 'EUROPE', 2: 'US', 3: 'OVERLAP'}
        
        for sess_id, sess_name in session_map.items():
            sess_trades = df[df[session_col] == sess_id]
            if not sess_trades.empty and sess_name not in session:
                session[sess_name] = {
                    'win_rate': float(sess_trades['trade_success'].mean()),
                    'avg_r': float(sess_trades['R_multiple'].mean()),
                    'count': int(len(sess_trades)),
                    'total_profit': float(sess_trades['net_profit'].sum()) if 'net_profit' in sess_trades.columns else 0.0
                }
    
    return {
        'hourly': hourly,
        'daily': daily,
        'weekly': weekly,
        'monthly': monthly,
        'session': session
    }



def calculate_trade_type_analysis(trades_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate trade type analysis (BUY vs SELL, exit reasons).
    
    Args:
        trades_df: DataFrame with Type, ExitReason, trade_success, R_multiple columns
    
    Returns:
        Dictionary with:
        - by_direction: Dict with BUY and SELL stats
        - by_exit_reason: Dict with exit reason -> stats
        - direction_distribution: Dict with counts
    
    Requirements: 0.7
    """
    if trades_df.empty:
        return {
            'by_direction': {},
            'by_exit_reason': {},
            'direction_distribution': {}
        }
    
    # By direction (BUY vs SELL)
    by_direction = {}
    if 'Type' in trades_df.columns:
        for direction in ['BUY', 'SELL']:
            dir_trades = trades_df[trades_df['Type'] == direction]
            if not dir_trades.empty:
                by_direction[direction] = {
                    'count': int(len(dir_trades)),
                    'win_rate': float(dir_trades['trade_success'].mean()),
                    'avg_r': float(dir_trades['R_multiple'].mean()),
                    'total_profit': float(dir_trades['net_profit'].sum()) if 'net_profit' in dir_trades.columns else 0.0
                }
    
    # By exit reason
    by_exit_reason = {}
    if 'ExitReason' in trades_df.columns:
        for reason in trades_df['ExitReason'].unique():
            if pd.notna(reason):
                reason_trades = trades_df[trades_df['ExitReason'] == reason]
                if not reason_trades.empty:
                    by_exit_reason[str(reason)] = {
                        'count': int(len(reason_trades)),
                        'win_rate': float(reason_trades['trade_success'].mean()),
                        'avg_r': float(reason_trades['R_multiple'].mean())
                    }
    
    # Direction distribution
    direction_distribution = {}
    if 'Type' in trades_df.columns:
        direction_counts = trades_df['Type'].value_counts()
        direction_distribution = {
            str(k): int(v) for k, v in direction_counts.items()
        }
    
    return {
        'by_direction': by_direction,
        'by_exit_reason': by_exit_reason,
        'direction_distribution': direction_distribution
    }



def calculate_consecutive_analysis(trades_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate consecutive trades analysis (streaks, cumulative).
    
    Args:
        trades_df: DataFrame with trade_success, R_multiple, exit_time/Timestamp columns
    
    Returns:
        Dictionary with:
        - streaks: Dict with win_streaks and loss_streaks distributions
        - max_win_streak: int
        - max_loss_streak: int
        - cumulative_by_streak: Dict with streak position -> avg performance
        - streak_recovery: Dict with recovery stats after loss streaks
    
    Requirements: 0.8
    """
    if trades_df.empty or 'trade_success' not in trades_df.columns:
        return {
            'streaks': {},
            'max_win_streak': 0,
            'max_loss_streak': 0,
            'cumulative_by_streak': {},
            'streak_recovery': {}
        }
    
    # Sort by time
    time_col = 'exit_time' if 'exit_time' in trades_df.columns else 'Timestamp'
    df = trades_df.sort_values(time_col).copy()
    
    # Calculate streaks
    outcomes = df['trade_success'].values
    
    # Find consecutive wins and losses
    win_streaks = []
    loss_streaks = []
    current_streak = 1
    current_type = outcomes[0]
    
    for i in range(1, len(outcomes)):
        if outcomes[i] == current_type:
            current_streak += 1
        else:
            if current_type == 1:
                win_streaks.append(current_streak)
            else:
                loss_streaks.append(current_streak)
            current_streak = 1
            current_type = outcomes[i]
    
    # Add last streak
    if current_type == 1:
        win_streaks.append(current_streak)
    else:
        loss_streaks.append(current_streak)
    
    # Streak distributions
    streaks = {
        'win_streaks': win_streaks,
        'loss_streaks': loss_streaks,
        'win_streak_distribution': {},
        'loss_streak_distribution': {}
    }
    
    if win_streaks:
        for length in set(win_streaks):
            streaks['win_streak_distribution'][int(length)] = win_streaks.count(length)
    
    if loss_streaks:
        for length in set(loss_streaks):
            streaks['loss_streak_distribution'][int(length)] = loss_streaks.count(length)
    
    max_win_streak = max(win_streaks) if win_streaks else 0
    max_loss_streak = max(loss_streaks) if loss_streaks else 0
    
    # Cumulative performance by position in sequence
    df['cumulative_r'] = df['R_multiple'].cumsum()
    cumulative_by_streak = {
        'positions': list(range(len(df))),
        'cumulative_r': df['cumulative_r'].tolist()
    }
    
    # Streak recovery analysis
    streak_recovery = {}
    if loss_streaks:
        # Find trades after loss streaks
        df['prev_outcome'] = df['trade_success'].shift(1)
        df['prev_2_outcome'] = df['trade_success'].shift(2)
        
        # After 1 loss
        after_1_loss = df[df['prev_outcome'] == 0]
        if not after_1_loss.empty:
            streak_recovery['after_1_loss'] = {
                'win_rate': float(after_1_loss['trade_success'].mean()),
                'avg_r': float(after_1_loss['R_multiple'].mean()),
                'count': int(len(after_1_loss))
            }
        
        # After 2 consecutive losses
        after_2_loss = df[(df['prev_outcome'] == 0) & (df['prev_2_outcome'] == 0)]
        if not after_2_loss.empty:
            streak_recovery['after_2_loss'] = {
                'win_rate': float(after_2_loss['trade_success'].mean()),
                'avg_r': float(after_2_loss['R_multiple'].mean()),
                'count': int(len(after_2_loss))
            }
    
    return {
        'streaks': streaks,
        'max_win_streak': int(max_win_streak),
        'max_loss_streak': int(max_loss_streak),
        'cumulative_by_streak': cumulative_by_streak,
        'streak_recovery': streak_recovery
    }



def calculate_risk_metrics(trades_df: pd.DataFrame, 
                          initial_equity: float = 10000,
                          risk_free_rate: float = 0.02) -> Dict[str, Any]:
    """
    Calculate comprehensive risk metrics (11+ metrics).
    
    Args:
        trades_df: DataFrame with trade data
        initial_equity: Starting equity
        risk_free_rate: Annual risk-free rate for Sharpe calculation
    
    Returns:
        Dictionary with 11+ risk metrics:
        - sharpe_ratio: float
        - sortino_ratio: float
        - calmar_ratio: float
        - max_drawdown_pct: float
        - max_drawdown_duration: int (days)
        - recovery_factor: float
        - profit_to_max_dd_ratio: float
        - win_loss_ratio: float
        - avg_win_to_avg_loss: float
        - largest_win: float
        - largest_loss: float
        - consecutive_wins_max: int
        - consecutive_losses_max: int
        - percent_profitable: float
    
    Requirements: 0.9
    """
    if trades_df.empty:
        return {
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'max_drawdown_pct': 0.0,
            'max_drawdown_duration': 0,
            'recovery_factor': 0.0,
            'profit_to_max_dd_ratio': 0.0,
            'win_loss_ratio': 0.0,
            'avg_win_to_avg_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'consecutive_wins_max': 0,
            'consecutive_losses_max': 0,
            'percent_profitable': 0.0
        }
    
    # Sort by time
    time_col = 'exit_time' if 'exit_time' in trades_df.columns else 'Timestamp'
    df = trades_df.sort_values(time_col).copy()
    
    # Returns
    returns = df['R_multiple'].values
    
    # Sharpe Ratio
    if len(returns) > 1 and returns.std() > 0:
        sharpe_ratio = (returns.mean() - risk_free_rate / 252) / returns.std() * np.sqrt(252)
    else:
        sharpe_ratio = 0.0
    
    # Sortino Ratio (using downside deviation)
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 1 and downside_returns.std() > 0:
        sortino_ratio = (returns.mean() - risk_free_rate / 252) / downside_returns.std() * np.sqrt(252)
    else:
        sortino_ratio = 0.0
    
    # Equity curve for drawdown calculations
    # Use same method as calculate_summary_metrics for consistency
    if 'equity_after_trade' in df.columns:
        # Method 1: Use equity_after_trade column if available
        equity_curve = df['equity_after_trade'].values
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns_pct = (equity_curve - running_max) / running_max
        max_drawdown_pct = float(drawdowns_pct.min() * 100)
        # Calculate dollar drawdowns for recovery factor
        drawdowns_dollar = equity_curve - running_max
    else:
        # Method 2: Calculate from cumulative profit
        cumulative_profit = df['net_profit'].cumsum()
        running_max = np.maximum.accumulate(cumulative_profit)
        drawdowns_dollar = cumulative_profit - running_max
        max_drawdown_pct = float((drawdowns_dollar.min() / initial_equity) * 100)
        drawdowns_pct = drawdowns_dollar / initial_equity
    
    # Max drawdown duration
    in_drawdown = drawdowns_pct < -0.001
    if in_drawdown.any():
        drawdown_changes = np.diff(np.concatenate([[False], in_drawdown, [False]]).astype(int))
        starts = np.where(drawdown_changes == 1)[0]
        ends = np.where(drawdown_changes == -1)[0]
        
        if len(starts) > 0 and len(ends) > 0:
            durations = []
            for start, end in zip(starts, ends):
                if start < len(df) and end <= len(df):
                    start_time = df.iloc[start][time_col]
                    end_time = df.iloc[min(end, len(df)-1)][time_col]
                    if pd.notna(start_time) and pd.notna(end_time):
                        duration = (pd.to_datetime(end_time) - pd.to_datetime(start_time)).days
                        durations.append(duration)
            max_drawdown_duration = max(durations) if durations else 0
        else:
            max_drawdown_duration = 0
    else:
        max_drawdown_duration = 0
    
    # Calmar Ratio
    net_profit = df['net_profit'].sum()
    total_return = net_profit / initial_equity
    if abs(max_drawdown_pct) > 0:
        calmar_ratio = (total_return * 100) / abs(max_drawdown_pct)
    else:
        calmar_ratio = 0.0
    
    # Recovery Factor
    if abs(drawdowns_dollar.min()) > 0:
        recovery_factor = net_profit / abs(drawdowns_dollar.min())
    else:
        recovery_factor = float('inf') if net_profit > 0 else 0.0
    
    # Profit to Max DD Ratio
    if abs(max_drawdown_pct) > 0:
        profit_to_max_dd_ratio = (net_profit / initial_equity * 100) / abs(max_drawdown_pct)
    else:
        profit_to_max_dd_ratio = 0.0
    
    # Win/Loss metrics
    winners = df[df['trade_success'] == 1]
    losers = df[df['trade_success'] == 0]
    
    win_count = len(winners)
    loss_count = len(losers)
    win_loss_ratio = win_count / loss_count if loss_count > 0 else float('inf')
    
    avg_win = winners['net_profit'].mean() if not winners.empty else 0.0
    avg_loss = abs(losers['net_profit'].mean()) if not losers.empty else 0.0
    avg_win_to_avg_loss = avg_win / avg_loss if avg_loss > 0 else 0.0
    
    largest_win = df['net_profit'].max()
    largest_loss = df['net_profit'].min()
    
    # Consecutive wins/losses
    consecutive_analysis = calculate_consecutive_analysis(df)
    consecutive_wins_max = consecutive_analysis['max_win_streak']
    consecutive_losses_max = consecutive_analysis['max_loss_streak']
    
    percent_profitable = (win_count / len(df)) * 100
    
    return {
        'sharpe_ratio': float(sharpe_ratio),
        'sortino_ratio': float(sortino_ratio),
        'calmar_ratio': float(calmar_ratio),
        'max_drawdown_pct': float(max_drawdown_pct),
        'max_drawdown_duration': int(max_drawdown_duration),
        'recovery_factor': float(recovery_factor) if recovery_factor != float('inf') else 999.99,
        'profit_to_max_dd_ratio': float(profit_to_max_dd_ratio),
        'win_loss_ratio': float(win_loss_ratio) if win_loss_ratio != float('inf') else 999.99,
        'avg_win_to_avg_loss': float(avg_win_to_avg_loss),
        'largest_win': float(largest_win),
        'largest_loss': float(largest_loss),
        'consecutive_wins_max': int(consecutive_wins_max),
        'consecutive_losses_max': int(consecutive_losses_max),
        'percent_profitable': float(percent_profitable)
    }
