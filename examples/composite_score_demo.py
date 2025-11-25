"""
Composite Score Engine Demo

This script demonstrates the usage of the composite score engine for
combining multiple trading signals into a single actionable score.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.calculators.composite_score import (
    calculate_composite_score,
    backtest_score_threshold,
    filter_by_score,
    add_recommendation_labels,
    get_score_statistics
)


def generate_sample_data(n_trades=500):
    """Generate sample trade data with probability features"""
    np.random.seed(42)
    
    # Generate probability features (0-1)
    data = {
        'timestamp': pd.date_range('2024-01-01', periods=n_trades, freq='1H'),
        'prob_global_win': np.random.beta(6, 4, n_trades),  # Skewed toward higher probabilities
        'prob_global_hit_1R': np.random.beta(5, 5, n_trades),  # More balanced
        'prob_entropy_win': np.random.beta(4, 6, n_trades),  # Skewed toward lower
        'prob_session_win': np.random.beta(5, 5, n_trades),
        'prob_trend_dir_alignment_win': np.random.beta(6, 4, n_trades),
        'prob_sr_zone_win': np.random.beta(5, 5, n_trades),
    }
    
    df = pd.DataFrame(data)
    
    # Generate R-multiple based on probabilities (higher prob = better R)
    avg_prob = (df['prob_global_win'] + df['prob_global_hit_1R']) / 2
    df['R_multiple'] = np.random.normal(
        loc=avg_prob * 2 - 0.5,  # Mean R increases with probability
        scale=1.5,
        size=n_trades
    )
    
    return df


def demo_basic_scoring():
    """Demo 1: Basic composite score calculation"""
    print("=" * 80)
    print("DEMO 1: Basic Composite Score Calculation")
    print("=" * 80)
    
    # Generate sample data
    df = generate_sample_data(100)
    
    # Calculate composite scores
    df_with_scores = calculate_composite_score(df)
    
    # Display results
    print("\nSample of trades with composite scores:")
    print(df_with_scores[[
        'prob_global_win', 'prob_global_hit_1R', 
        'composite_score', 'R_multiple'
    ]].head(10).to_string(index=False))
    
    print("\nComponent scores for first trade:")
    first_trade = df_with_scores.iloc[0]
    print(f"  Win Rate Score:        {first_trade['score_win_rate']:.1f}")
    print(f"  Expected R Score:      {first_trade['score_expected_r']:.1f}")
    print(f"  Structure Quality:     {first_trade['score_structure_quality']:.1f}")
    print(f"  Time-Based Score:      {first_trade['score_time_based']:.1f}")
    print(f"  Correlation Score:     {first_trade['score_correlation']:.1f}")
    print(f"  Entry Quality Score:   {first_trade['score_entry_quality']:.1f}")
    print(f"  → Composite Score:     {first_trade['composite_score']:.1f}")
    
    return df_with_scores


def demo_threshold_backtesting(df_with_scores):
    """Demo 2: Backtest multiple score thresholds"""
    print("\n" + "=" * 80)
    print("DEMO 2: Threshold Backtesting")
    print("=" * 80)
    
    # Backtest thresholds
    thresholds = [40, 50, 60, 70, 80]
    backtest_results = backtest_score_threshold(df_with_scores, thresholds=thresholds)
    
    print("\nBacktest Results:")
    print(backtest_results.to_string(index=False))
    
    print("\nKey Insights:")
    for _, row in backtest_results.iterrows():
        if row['trade_frequency'] > 0:
            print(f"  Threshold {row['threshold']:.0f}: "
                  f"Win Rate {row['win_rate']:.1%}, "
                  f"Expectancy {row['expectancy']:.2f}R, "
                  f"{row['trade_frequency']:.0f} trades ({row['trade_frequency_pct']:.1f}%)")


def demo_filtering(df_with_scores):
    """Demo 3: Filter trades by score threshold"""
    print("\n" + "=" * 80)
    print("DEMO 3: Trade Filtering by Score")
    print("=" * 80)
    
    threshold = 65
    
    # Filter trades
    filtered_df = filter_by_score(df_with_scores, threshold=threshold)
    
    # Calculate metrics for original and filtered
    original_win_rate = (df_with_scores['R_multiple'] > 0).mean()
    original_expectancy = df_with_scores['R_multiple'].mean()
    
    filtered_win_rate = (filtered_df['R_multiple'] > 0).mean()
    filtered_expectancy = filtered_df['R_multiple'].mean()
    
    print(f"\nFiltering with threshold: {threshold}")
    print(f"\nOriginal Dataset:")
    print(f"  Trades:      {len(df_with_scores)}")
    print(f"  Win Rate:    {original_win_rate:.1%}")
    print(f"  Expectancy:  {original_expectancy:.2f}R")
    
    print(f"\nFiltered Dataset (score >= {threshold}):")
    print(f"  Trades:      {len(filtered_df)}")
    print(f"  Win Rate:    {filtered_win_rate:.1%}")
    print(f"  Expectancy:  {filtered_expectancy:.2f}R")
    
    print(f"\nImprovement:")
    print(f"  Win Rate:    {(filtered_win_rate - original_win_rate):.1%} points")
    print(f"  Expectancy:  {(filtered_expectancy - original_expectancy):.2f}R")
    print(f"  Trade Count: {len(filtered_df) / len(df_with_scores):.1%} of original")


def demo_recommendations(df_with_scores):
    """Demo 4: Recommendation classification"""
    print("\n" + "=" * 80)
    print("DEMO 4: Recommendation Classification")
    print("=" * 80)
    
    # Add recommendation labels
    df_with_recommendations = add_recommendation_labels(df_with_scores)
    
    # Count recommendations
    recommendation_counts = df_with_recommendations['recommendation'].value_counts()
    
    print("\nRecommendation Distribution:")
    for rec, count in recommendation_counts.items():
        pct = (count / len(df_with_recommendations)) * 100
        print(f"  {rec:12s}: {count:3d} trades ({pct:5.1f}%)")
    
    # Show performance by recommendation
    print("\nPerformance by Recommendation:")
    for rec in ['STRONG BUY', 'BUY', 'NEUTRAL', 'AVOID']:
        rec_trades = df_with_recommendations[df_with_recommendations['recommendation'] == rec]
        if len(rec_trades) > 0:
            win_rate = (rec_trades['R_multiple'] > 0).mean()
            expectancy = rec_trades['R_multiple'].mean()
            avg_score = rec_trades['composite_score'].mean()
            print(f"  {rec:12s}: Win Rate {win_rate:.1%}, "
                  f"Expectancy {expectancy:.2f}R, "
                  f"Avg Score {avg_score:.1f}")


def demo_custom_weights():
    """Demo 5: Custom component weights"""
    print("\n" + "=" * 80)
    print("DEMO 5: Custom Component Weights")
    print("=" * 80)
    
    # Generate sample data
    df = generate_sample_data(100)
    
    # Default weights
    df_default = calculate_composite_score(df)
    
    # Custom weights - emphasize win rate and expected R
    custom_weights = {
        'win_rate': 0.40,
        'expected_r': 0.35,
        'structure_quality': 0.10,
        'time_based': 0.05,
        'correlation': 0.05,
        'entry_quality': 0.05
    }
    df_custom = calculate_composite_score(df, weights=custom_weights)
    
    print("\nDefault Weights:")
    print("  Win Rate: 30%, Expected R: 25%, Structure: 15%")
    print("  Time: 10%, Correlation: 10%, Entry: 10%")
    print(f"  Mean Score: {df_default['composite_score'].mean():.1f}")
    
    print("\nCustom Weights (Emphasize Win Rate & Expected R):")
    print("  Win Rate: 40%, Expected R: 35%, Structure: 10%")
    print("  Time: 5%, Correlation: 5%, Entry: 5%")
    print(f"  Mean Score: {df_custom['composite_score'].mean():.1f}")
    
    # Compare top 10 trades
    print("\nTop 10 Trades Comparison:")
    print("Rank | Default Score | Custom Score | Difference")
    print("-" * 50)
    
    df_default_sorted = df_default.sort_values('composite_score', ascending=False).head(10)
    df_custom_sorted = df_custom.sort_values('composite_score', ascending=False).head(10)
    
    for i in range(10):
        default_score = df_default_sorted.iloc[i]['composite_score']
        custom_score = df_custom_sorted.iloc[i]['composite_score']
        diff = custom_score - default_score
        print(f"{i+1:4d} | {default_score:13.1f} | {custom_score:12.1f} | {diff:+10.1f}")


def demo_statistics(df_with_scores):
    """Demo 6: Score statistics"""
    print("\n" + "=" * 80)
    print("DEMO 6: Score Statistics")
    print("=" * 80)
    
    # Get statistics
    stats = get_score_statistics(df_with_scores)
    
    print("\nComposite Score Statistics:")
    print(f"  Mean:     {stats['mean_score']:.1f}")
    print(f"  Median:   {stats['median_score']:.1f}")
    print(f"  Std Dev:  {stats['std_score']:.1f}")
    print(f"  Min:      {stats['min_score']:.1f}")
    print(f"  Max:      {stats['max_score']:.1f}")
    
    print("\nRecommendation Distribution:")
    print(f"  STRONG BUY: {stats['pct_strong_buy']:5.1f}%")
    print(f"  BUY:        {stats['pct_buy']:5.1f}%")
    print(f"  NEUTRAL:    {stats['pct_neutral']:5.1f}%")
    print(f"  AVOID:      {stats['pct_avoid']:5.1f}%")


def main():
    """Run all demos"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "COMPOSITE SCORE ENGINE DEMO" + " " * 31 + "║")
    print("╚" + "=" * 78 + "╝")
    
    # Run demos
    df_with_scores = demo_basic_scoring()
    demo_threshold_backtesting(df_with_scores)
    demo_filtering(df_with_scores)
    demo_recommendations(df_with_scores)
    demo_custom_weights()
    demo_statistics(df_with_scores)
    
    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("  1. Composite scores combine multiple signals into a single metric")
    print("  2. Higher scores correlate with better win rates and expectancy")
    print("  3. Threshold backtesting helps find optimal filtering levels")
    print("  4. Custom weights allow strategy-specific optimization")
    print("  5. Recommendation labels provide clear action guidance")
    print("\n")


if __name__ == '__main__':
    main()
