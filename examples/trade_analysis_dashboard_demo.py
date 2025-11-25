"""
Demo script for Trade Analysis Dashboard backend functions.

This script demonstrates how to use the trade analysis dashboard functions
to analyze trading performance.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.calculators.trade_analysis_dashboard import (
    calculate_summary_metrics,
    calculate_equity_curve,
    calculate_r_distribution,
    calculate_mae_mfe_analysis,
    calculate_time_based_performance,
    calculate_trade_type_analysis,
    calculate_consecutive_analysis,
    calculate_risk_metrics
)


def generate_sample_trades(n=200):
    """Generate sample trade data for demonstration."""
    np.random.seed(42)
    
    base_time = datetime(2024, 1, 1)
    timestamps = [base_time + timedelta(hours=i*6) for i in range(n)]
    
    # Generate realistic trade data with some patterns
    r_multiples = np.random.normal(0.3, 1.8, n)
    trade_success = (r_multiples > 0).astype(int)
    
    df = pd.DataFrame({
        'Timestamp': timestamps,
        'exit_time': timestamps,
        'entry_time': [t - timedelta(hours=2) for t in timestamps],
        'Type': np.random.choice(['BUY', 'SELL'], n, p=[0.55, 0.45]),
        'R_multiple': r_multiples,
        'trade_success': trade_success,
        'net_profit': r_multiples * 100,
        'gross_profit': r_multiples * 105,
        'MAE_R': np.abs(np.random.uniform(0, 1.5, n)),
        'MFE_R': np.abs(np.random.uniform(0, 4, n)),
        'ExitReason': np.random.choice(['TP', 'SL', 'Manual', 'TimeExit'], n, p=[0.4, 0.35, 0.15, 0.1]),
        'session': np.random.choice([0, 1, 2, 3], n),
        'entry_session': np.random.choice(['ASIA', 'EUROPE', 'US'], n),
        'equity_after_trade': 10000 + np.cumsum(r_multiples * 100),
        'holding_minutes': np.random.randint(30, 480, n)
    })
    
    return df


def main():
    """Run the demo."""
    print("=" * 80)
    print("TRADE ANALYSIS DASHBOARD DEMO")
    print("=" * 80)
    print()
    
    # Generate sample data
    print("Generating sample trade data...")
    trades_df = generate_sample_trades(200)
    print(f"Generated {len(trades_df)} trades")
    print()
    
    # 1. Summary Metrics
    print("-" * 80)
    print("1. SUMMARY METRICS")
    print("-" * 80)
    summary = calculate_summary_metrics(trades_df)
    print(f"Total Trades:    {summary['total_trades']}")
    print(f"Win Rate:        {summary['win_rate']:.2%}")
    print(f"Avg R-Multiple:  {summary['avg_r']:.2f}")
    print(f"Expectancy:      ${summary['expectancy']:.2f}")
    print(f"Max Drawdown:    {summary['max_drawdown']:.2f}%")
    print(f"Profit Factor:   {summary['profit_factor']:.2f}")
    print()
    
    # 2. Equity Curve
    print("-" * 80)
    print("2. EQUITY CURVE")
    print("-" * 80)
    equity = calculate_equity_curve(trades_df, initial_equity=10000)
    print(f"Starting Equity: ${equity['equity_values'][0]:.2f}")
    print(f"Ending Equity:   ${equity['equity_values'][-1]:.2f}")
    print(f"Total Profit:    ${equity['cumulative_profit'][-1]:.2f}")
    print(f"Drawdown Periods: {len(equity['drawdown_periods'])}")
    if equity['drawdown_periods']:
        worst_dd = min(equity['drawdown_periods'], key=lambda x: x['depth_pct'])
        print(f"Worst Drawdown:  {worst_dd['depth_pct']:.2f}% (trades {worst_dd['start_idx']}-{worst_dd['end_idx']})")
    print()
    
    # 3. R-Multiple Distribution
    print("-" * 80)
    print("3. R-MULTIPLE DISTRIBUTION")
    print("-" * 80)
    r_dist = calculate_r_distribution(trades_df)
    stats = r_dist['statistics']
    probs = r_dist['threshold_probs']
    print(f"Mean R:          {stats['mean']:.2f}")
    print(f"Median R:        {stats['median']:.2f}")
    print(f"Std Dev:         {stats['std']:.2f}")
    print(f"25th Percentile: {stats['p25']:.2f}")
    print(f"75th Percentile: {stats['p75']:.2f}")
    print(f"P(R > 1):        {probs['p_r_gt_1']:.2%}")
    print(f"P(R > 2):        {probs['p_r_gt_2']:.2%}")
    print(f"P(R > 3):        {probs['p_r_gt_3']:.2%}")
    print(f"Best R:          {r_dist['best_r']:.2f}")
    print(f"Worst R:         {r_dist['worst_r']:.2f}")
    print()
    
    # 4. MAE/MFE Analysis
    print("-" * 80)
    print("4. MAE/MFE ANALYSIS")
    print("-" * 80)
    mae_mfe = calculate_mae_mfe_analysis(trades_df)
    winners = mae_mfe['winners_stats']
    losers = mae_mfe['losers_stats']
    print("Winners:")
    print(f"  Avg MAE:       {winners['avg_mae']:.2f}R")
    print(f"  Avg MFE:       {winners['avg_mfe']:.2f}R")
    print(f"  MFE/R Ratio:   {winners['mfe_to_r_ratio']:.2f}x")
    print(f"  Profit Left:   {winners['profit_left']:.2f}R")
    print("Losers:")
    print(f"  Avg MAE:       {losers['avg_mae']:.2f}R")
    print(f"  Avg MFE:       {losers['avg_mfe']:.2f}R")
    print(f"MFE-R Correlation: {mae_mfe['correlation']:.3f}")
    print()
    
    # 5. Time-Based Performance
    print("-" * 80)
    print("5. TIME-BASED PERFORMANCE")
    print("-" * 80)
    time_perf = calculate_time_based_performance(trades_df)
    
    print("Best Hours (by win rate):")
    if time_perf['hourly']:
        sorted_hours = sorted(time_perf['hourly'].items(), 
                            key=lambda x: x[1]['win_rate'], reverse=True)[:3]
        for hour, stats in sorted_hours:
            print(f"  Hour {hour:02d}: {stats['win_rate']:.1%} win rate, "
                  f"{stats['avg_r']:.2f} avg R ({stats['count']} trades)")
    
    print("\nBest Days (by win rate):")
    if time_perf['daily']:
        sorted_days = sorted(time_perf['daily'].items(), 
                           key=lambda x: x[1]['win_rate'], reverse=True)[:3]
        for day, stats in sorted_days:
            print(f"  {day}: {stats['win_rate']:.1%} win rate, "
                  f"{stats['avg_r']:.2f} avg R ({stats['count']} trades)")
    
    print("\nSession Performance:")
    if time_perf['session']:
        for session, stats in time_perf['session'].items():
            print(f"  {session}: {stats['win_rate']:.1%} win rate, "
                  f"{stats['avg_r']:.2f} avg R ({stats['count']} trades)")
    print()
    
    # 6. Trade Type Analysis
    print("-" * 80)
    print("6. TRADE TYPE ANALYSIS")
    print("-" * 80)
    trade_type = calculate_trade_type_analysis(trades_df)
    
    print("By Direction:")
    for direction, stats in trade_type['by_direction'].items():
        print(f"  {direction}: {stats['count']} trades, {stats['win_rate']:.1%} win rate, "
              f"{stats['avg_r']:.2f} avg R, ${stats['total_profit']:.2f} profit")
    
    print("\nTop Exit Reasons:")
    if trade_type['by_exit_reason']:
        sorted_reasons = sorted(trade_type['by_exit_reason'].items(), 
                              key=lambda x: x[1]['count'], reverse=True)[:3]
        for reason, stats in sorted_reasons:
            print(f"  {reason}: {stats['count']} trades, {stats['win_rate']:.1%} win rate, "
                  f"{stats['avg_r']:.2f} avg R")
    print()
    
    # 7. Consecutive Analysis
    print("-" * 80)
    print("7. CONSECUTIVE TRADES ANALYSIS")
    print("-" * 80)
    consecutive = calculate_consecutive_analysis(trades_df)
    print(f"Max Win Streak:  {consecutive['max_win_streak']} trades")
    print(f"Max Loss Streak: {consecutive['max_loss_streak']} trades")
    
    if consecutive['streak_recovery']:
        print("\nStreak Recovery:")
        for streak_type, stats in consecutive['streak_recovery'].items():
            print(f"  {streak_type}: {stats['win_rate']:.1%} win rate, "
                  f"{stats['avg_r']:.2f} avg R ({stats['count']} trades)")
    print()
    
    # 8. Risk Metrics
    print("-" * 80)
    print("8. COMPREHENSIVE RISK METRICS")
    print("-" * 80)
    risk = calculate_risk_metrics(trades_df, initial_equity=10000)
    print(f"Sharpe Ratio:           {risk['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio:          {risk['sortino_ratio']:.2f}")
    print(f"Calmar Ratio:           {risk['calmar_ratio']:.2f}")
    print(f"Max Drawdown:           {risk['max_drawdown_pct']:.2f}%")
    print(f"Max DD Duration:        {risk['max_drawdown_duration']} days")
    print(f"Recovery Factor:        {risk['recovery_factor']:.2f}")
    print(f"Profit/MaxDD Ratio:     {risk['profit_to_max_dd_ratio']:.2f}")
    print(f"Win/Loss Ratio:         {risk['win_loss_ratio']:.2f}")
    print(f"Avg Win/Avg Loss:       {risk['avg_win_to_avg_loss']:.2f}")
    print(f"Largest Win:            ${risk['largest_win']:.2f}")
    print(f"Largest Loss:           ${risk['largest_loss']:.2f}")
    print(f"Max Consecutive Wins:   {risk['consecutive_wins_max']}")
    print(f"Max Consecutive Losses: {risk['consecutive_losses_max']}")
    print(f"Percent Profitable:     {risk['percent_profitable']:.1f}%")
    print()
    
    print("=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
