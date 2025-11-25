"""
Demo script for Expectancy Calculator

This script demonstrates the usage of the expectancy and R-multiple analyzer functions.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from backend.calculators.expectancy_calculator import (
    compute_expectancy_R,
    compute_expectancy_by_group,
    compute_r_percentiles,
    compute_r_threshold_probabilities,
    compute_expected_r_per_bin
)


def main():
    print("=" * 70)
    print("Expectancy and R-Multiple Analyzer Demo")
    print("=" * 70)
    print()
    
    # Generate sample trade data
    np.random.seed(42)
    n_trades = 200
    
    # Simulate realistic R-multiple distribution
    # 60% win rate with average win of 1.5R and average loss of -0.8R
    wins = np.random.exponential(1.5, int(n_trades * 0.6))
    losses = -np.random.exponential(0.8, int(n_trades * 0.4))
    r_multiples = np.concatenate([wins, losses])
    np.random.shuffle(r_multiples)
    
    # Create sample dataframe
    df = pd.DataFrame({
        'R_multiple': r_multiples[:n_trades],
        'net_profit': r_multiples[:n_trades] * 100,  # $100 risk per trade
        'risk_percent': np.ones(n_trades) * 1.0,  # 1% risk per trade
        'session': np.random.choice(['ASIA', 'EUROPE', 'US'], n_trades),
        'trend_strength_bin': np.random.choice([0, 1, 2, 3], n_trades)
    })
    
    # 1. Compute global expectancy
    print("1. GLOBAL EXPECTANCY")
    print("-" * 70)
    expectancy = compute_expectancy_R(df)
    print(f"Total Trades: {expectancy['total_trades']}")
    print(f"Win Rate: {expectancy['win_rate']:.2%}")
    print(f"Expectancy (R): {expectancy['expectancy_R']:.3f}R")
    print(f"Expectancy ($): ${expectancy['expectancy_dollar']:.2f}")
    print(f"Average Win: {expectancy['avg_win_R']:.3f}R")
    print(f"Average Loss: {expectancy['avg_loss_R']:.3f}R")
    print()
    
    # 2. Compute expectancy by session
    print("2. EXPECTANCY BY SESSION")
    print("-" * 70)
    session_expectancy = compute_expectancy_by_group(df, 'session')
    print(session_expectancy.to_string(index=False))
    print()
    
    # 3. Compute R-multiple percentiles
    print("3. R-MULTIPLE DISTRIBUTION PERCENTILES")
    print("-" * 70)
    percentiles = compute_r_percentiles(df)
    print(f"Min R: {percentiles['min_R']:.3f}R")
    print(f"25th Percentile: {percentiles['p25']:.3f}R")
    print(f"50th Percentile (Median): {percentiles['p50']:.3f}R")
    print(f"75th Percentile: {percentiles['p75']:.3f}R")
    print(f"90th Percentile: {percentiles['p90']:.3f}R")
    print(f"95th Percentile: {percentiles['p95']:.3f}R")
    print(f"Max R: {percentiles['max_R']:.3f}R")
    print(f"Mean R: {percentiles['mean_R']:.3f}R")
    print(f"Std Dev: {percentiles['std_R']:.3f}R")
    print()
    
    # 4. Compute threshold probabilities
    print("4. R-MULTIPLE THRESHOLD PROBABILITIES")
    print("-" * 70)
    thresholds = compute_r_threshold_probabilities(df)
    print(f"P(R > 1): {thresholds['p_r_gt_1']:.2%}")
    print(f"P(R >= 1): {thresholds['p_r_gte_1']:.2%}")
    print(f"P(R > 2): {thresholds['p_r_gt_2']:.2%}")
    print(f"P(R >= 2): {thresholds['p_r_gte_2']:.2%}")
    print(f"P(R > 3): {thresholds['p_r_gt_3']:.2%}")
    print(f"P(R >= 3): {thresholds['p_r_gte_3']:.2%}")
    print()
    
    # 5. Compute expected R per bin
    print("5. EXPECTED R BY TREND STRENGTH BIN")
    print("-" * 70)
    bin_expectancy = compute_expected_r_per_bin(df, 'trend_strength_bin')
    print(bin_expectancy.to_string(index=False))
    print()
    
    print("=" * 70)
    print("Demo Complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
