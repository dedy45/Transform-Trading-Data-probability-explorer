"""
Demo script for probability calculator functionality.

This script demonstrates how to use the 1D and 2D probability calculators
with sample data.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.calculators.probability_calculator import (
    compute_1d_probability,
    compute_2d_probability,
    get_probability_summary
)

# Set random seed for reproducibility
np.random.seed(42)

# Create sample data
n_samples = 500

# Generate features
trend_strength = np.random.uniform(0, 1, n_samples)
volatility_regime = np.random.choice([0, 1, 2], n_samples)  # Low, Medium, High
session = np.random.choice(['ASIA', 'EUROPE', 'US'], n_samples)

# Generate target with some correlation to features
# Higher trend strength -> higher win probability
win_prob = 0.4 + 0.3 * trend_strength
y_win = (np.random.random(n_samples) < win_prob).astype(int)

# Generate R-multiples
R_multiple = np.random.normal(0.5, 1.5, n_samples)
R_multiple = np.where(y_win == 1, np.abs(R_multiple), -np.abs(R_multiple))

# Create DataFrame
df = pd.DataFrame({
    'y_win': y_win,
    'trend_strength': trend_strength,
    'volatility_regime': volatility_regime,
    'session': session,
    'R_multiple': R_multiple
})

print("=" * 80)
print("PROBABILITY CALCULATOR DEMO")
print("=" * 80)
print(f"\nDataset: {len(df)} samples")
print(f"Overall win rate: {df['y_win'].mean():.2%}")
print(f"Mean R-multiple: {df['R_multiple'].mean():.2f}")

# Example 1: 1D Probability with numeric feature
print("\n" + "=" * 80)
print("EXAMPLE 1: 1D Probability - Trend Strength")
print("=" * 80)

result_1d = compute_1d_probability(
    df,
    target='y_win',
    feature='trend_strength',
    bins=10,
    conf_level=0.95,
    min_samples_per_bin=10
)

print("\nResults:")
print(result_1d[['label', 'n', 'p_est', 'ci_lower', 'ci_upper', 'mean_R', 'is_reliable']].to_string(index=False))

summary_1d = get_probability_summary(result_1d)
print(f"\nSummary:")
print(f"  Total bins: {summary_1d['total_bins']}")
print(f"  Reliable bins: {summary_1d['reliable_bins']}")
print(f"  Mean probability: {summary_1d['mean_probability']:.2%}")
print(f"  Probability range: {summary_1d['probability_range']:.2%}")

# Example 2: 1D Probability with categorical feature
print("\n" + "=" * 80)
print("EXAMPLE 2: 1D Probability - Session (Categorical)")
print("=" * 80)

result_1d_cat = compute_1d_probability(
    df,
    target='y_win',
    feature='session',
    conf_level=0.95,
    min_samples_per_bin=20
)

print("\nResults:")
print(result_1d_cat[['label', 'n', 'p_est', 'ci_lower', 'ci_upper', 'mean_R', 'is_reliable']].to_string(index=False))

# Example 3: 2D Probability
print("\n" + "=" * 80)
print("EXAMPLE 3: 2D Probability - Trend Strength × Volatility Regime")
print("=" * 80)

result_2d = compute_2d_probability(
    df,
    target='y_win',
    feature_x='trend_strength',
    feature_y='volatility_regime',
    bins_x=5,
    bins_y=3,
    conf_level=0.95,
    min_samples_per_cell=10
)

print("\nResults (first 10 cells):")
print(result_2d[['label_x', 'label_y', 'n', 'p_est', 'ci_lower', 'ci_upper', 'is_reliable']].head(10).to_string(index=False))

summary_2d = get_probability_summary(result_2d)
print(f"\nSummary:")
print(f"  Total cells: {summary_2d['total_bins']}")
print(f"  Reliable cells: {summary_2d['reliable_bins']}")
print(f"  Mean probability: {summary_2d['mean_probability']:.2%}")

# Example 4: Create pivot table for heatmap visualization
print("\n" + "=" * 80)
print("EXAMPLE 4: Pivot Table for Heatmap")
print("=" * 80)

# Create a simpler 2D result for visualization
result_2d_simple = compute_2d_probability(
    df,
    target='y_win',
    feature_x='trend_strength',
    feature_y='volatility_regime',
    bins_x=3,
    bins_y=3,
    min_samples_per_cell=5
)

# Create pivot table
pivot_p_est = result_2d_simple.pivot_table(
    index='label_y',
    columns='label_x',
    values='p_est',
    aggfunc='first'
)

pivot_n = result_2d_simple.pivot_table(
    index='label_y',
    columns='label_x',
    values='n',
    aggfunc='first'
)

print("\nWin Probability Heatmap:")
print(pivot_p_est.to_string(float_format=lambda x: f"{x:.2%}" if not pd.isna(x) else "N/A"))

print("\nSample Size Heatmap:")
print(pivot_n.to_string(float_format=lambda x: f"{int(x)}" if not pd.isna(x) else "N/A"))

print("\n" + "=" * 80)
print("DEMO COMPLETE")
print("=" * 80)
