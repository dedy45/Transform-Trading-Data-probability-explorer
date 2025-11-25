"""
Demo: Bayesian Probability Calculator

This script demonstrates the usage of Bayesian probability calculators
for trading analysis, particularly useful for small sample sizes.

Features demonstrated:
1. Bayesian win rate estimation with Beta posterior
2. Adaptive probability tracking with rolling windows
3. Bayesian comparison between trading conditions
4. Certainty calculation from posterior distributions
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from backend.calculators.bayesian_probability import (
    bayesian_win_rate,
    adaptive_probability_tracker,
    bayesian_comparison,
    calculate_certainty,
    bayesian_win_rate_by_group
)


def demo_bayesian_win_rate():
    """Demonstrate Bayesian win rate estimation"""
    print("=" * 80)
    print("DEMO 1: Bayesian Win Rate Estimation")
    print("=" * 80)
    
    # Example 1: Small sample with uniform prior
    print("\n1. Small sample (7 wins out of 10 trades) with uniform prior:")
    result = bayesian_win_rate(successes=7, total=10, alpha0=1.0, beta0=1.0)
    
    print(f"   Posterior Mean: {result['posterior_mean']:.1%}")
    print(f"   Posterior Mode: {result['posterior_mode']:.1%}")
    print(f"   95% Credible Interval: [{result['credible_lower']:.1%}, {result['credible_upper']:.1%}]")
    print(f"   Posterior Std: {result['posterior_std']:.3f}")
    print(f"   Certainty Score: {result['certainty']:.2f}")
    
    # Example 2: Same data with informative prior (historical 60% win rate)
    print("\n2. Same data with informative prior (historical 60% win rate):")
    result_prior = bayesian_win_rate(successes=7, total=10, alpha0=60, beta0=40)
    
    print(f"   Posterior Mean: {result_prior['posterior_mean']:.1%}")
    print(f"   95% Credible Interval: [{result_prior['credible_lower']:.1%}, {result_prior['credible_upper']:.1%}]")
    print(f"   Certainty Score: {result_prior['certainty']:.2f}")
    print(f"   Note: Prior pulls estimate toward 60%, narrower interval")
    
    # Example 3: Large sample
    print("\n3. Large sample (600 wins out of 1000 trades):")
    result_large = bayesian_win_rate(successes=600, total=1000)
    
    print(f"   Posterior Mean: {result_large['posterior_mean']:.1%}")
    print(f"   95% Credible Interval: [{result_large['credible_lower']:.1%}, {result_large['credible_upper']:.1%}]")
    print(f"   Certainty Score: {result_large['certainty']:.2f}")
    print(f"   Note: Large sample gives high certainty and narrow interval")


def demo_adaptive_tracker():
    """Demonstrate adaptive probability tracking"""
    print("\n" + "=" * 80)
    print("DEMO 2: Adaptive Probability Tracking")
    print("=" * 80)
    
    # Generate synthetic trade sequence with regime change
    np.random.seed(42)
    
    # First 50 trades: 40% win rate
    trades_1 = np.random.binomial(1, 0.4, 50)
    
    # Next 50 trades: 70% win rate (regime change)
    trades_2 = np.random.binomial(1, 0.7, 50)
    
    # Combine
    all_trades = np.concatenate([trades_1, trades_2])
    
    df = pd.DataFrame({'trade_success': all_trades})
    
    # Track with 20-trade rolling window
    print("\nTracking win rate with 20-trade rolling window:")
    tracker = adaptive_probability_tracker(df, window_size=20)
    
    # Show key points
    print(f"\nTrade 20 (early period):")
    print(f"   Win Rate: {tracker.iloc[19]['posterior_mean']:.1%}")
    print(f"   Certainty: {tracker.iloc[19]['certainty']:.2f}")
    
    print(f"\nTrade 50 (end of low win rate period):")
    print(f"   Win Rate: {tracker.iloc[49]['posterior_mean']:.1%}")
    print(f"   Certainty: {tracker.iloc[49]['certainty']:.2f}")
    
    print(f"\nTrade 70 (after regime change):")
    print(f"   Win Rate: {tracker.iloc[69]['posterior_mean']:.1%}")
    print(f"   Certainty: {tracker.iloc[69]['certainty']:.2f}")
    
    print(f"\nTrade 100 (end of high win rate period):")
    print(f"   Win Rate: {tracker.iloc[99]['posterior_mean']:.1%}")
    print(f"   Certainty: {tracker.iloc[99]['certainty']:.2f}")
    
    print("\nNote: Tracker detects regime change around trade 60-70")
    
    # Optional: Plot if matplotlib available
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(tracker['trade_index'], tracker['posterior_mean'], label='Posterior Mean', linewidth=2)
        plt.fill_between(tracker['trade_index'], 
                        tracker['credible_lower'], 
                        tracker['credible_upper'],
                        alpha=0.3, label='95% Credible Interval')
        plt.axhline(y=0.4, color='r', linestyle='--', label='True Win Rate (Period 1)')
        plt.axhline(y=0.7, color='g', linestyle='--', label='True Win Rate (Period 2)')
        plt.axvline(x=50, color='k', linestyle=':', alpha=0.5, label='Regime Change')
        plt.xlabel('Trade Number')
        plt.ylabel('Win Rate')
        plt.title('Adaptive Bayesian Probability Tracking')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('bayesian_tracker_demo.png', dpi=150)
        print("\nPlot saved as 'bayesian_tracker_demo.png'")
    except:
        print("\n(Plotting skipped - matplotlib not available)")


def demo_bayesian_comparison():
    """Demonstrate Bayesian comparison between conditions"""
    print("\n" + "=" * 80)
    print("DEMO 3: Bayesian Comparison Between Conditions")
    print("=" * 80)
    
    # Generate synthetic data for two trading sessions
    np.random.seed(42)
    
    # Europe session: 65% win rate, 80 trades
    europe_trades = np.random.binomial(1, 0.65, 80)
    
    # Asia session: 45% win rate, 60 trades
    asia_trades = np.random.binomial(1, 0.45, 60)
    
    # Create DataFrame
    df = pd.DataFrame({
        'trade_success': np.concatenate([europe_trades, asia_trades]),
        'session': ['EUROPE'] * 80 + ['ASIA'] * 60
    })
    
    print("\nComparing Europe session vs Asia session:")
    print(f"Europe: {europe_trades.sum()} wins out of {len(europe_trades)} trades ({europe_trades.mean():.1%})")
    print(f"Asia: {asia_trades.sum()} wins out of {len(asia_trades)} trades ({asia_trades.mean():.1%})")
    
    # Perform Bayesian comparison
    result = bayesian_comparison(
        df=df,
        target_col='trade_success',
        condition_a={'session': 'EUROPE'},
        condition_b={'session': 'ASIA'},
        n_samples=100000
    )
    
    print(f"\nBayesian Comparison Results:")
    print(f"   P(Europe better than Asia): {result['prob_a_better']:.1%}")
    print(f"   P(Asia better than Europe): {result['prob_b_better']:.1%}")
    print(f"   P(Approximately equal): {result['prob_equal']:.1%}")
    print(f"   Expected difference: {result['mean_diff']:.1%}")
    print(f"   Effect size (Cohen's h): {result['effect_size']:.3f}")
    
    print(f"\nPosterior Estimates:")
    print(f"   Europe posterior mean: {result['a_posterior_mean']:.1%}")
    print(f"   Asia posterior mean: {result['b_posterior_mean']:.1%}")
    
    # Interpretation
    if result['prob_a_better'] > 0.95:
        print("\n✓ Strong evidence that Europe session is better")
    elif result['prob_a_better'] > 0.80:
        print("\n✓ Moderate evidence that Europe session is better")
    elif result['prob_a_better'] > 0.50:
        print("\n~ Weak evidence that Europe session is better")
    else:
        print("\n~ No clear winner")


def demo_certainty_calculation():
    """Demonstrate certainty calculation"""
    print("\n" + "=" * 80)
    print("DEMO 4: Certainty Calculation")
    print("=" * 80)
    
    print("\nCertainty decreases as posterior distribution widens:")
    
    # Scenario 1: High certainty (large sample, narrow distribution)
    print("\n1. Large sample (500 trades, 60% win rate):")
    result_large = bayesian_win_rate(successes=300, total=500)
    print(f"   Posterior Mean: {result_large['posterior_mean']:.1%}")
    print(f"   Posterior Std: {result_large['posterior_std']:.4f}")
    print(f"   Credible Interval Width: {result_large['credible_upper'] - result_large['credible_lower']:.3f}")
    print(f"   Certainty: {result_large['certainty']:.2f} (HIGH)")
    
    # Scenario 2: Medium certainty (medium sample)
    print("\n2. Medium sample (50 trades, 60% win rate):")
    result_medium = bayesian_win_rate(successes=30, total=50)
    print(f"   Posterior Mean: {result_medium['posterior_mean']:.1%}")
    print(f"   Posterior Std: {result_medium['posterior_std']:.4f}")
    print(f"   Credible Interval Width: {result_medium['credible_upper'] - result_medium['credible_lower']:.3f}")
    print(f"   Certainty: {result_medium['certainty']:.2f} (MEDIUM)")
    
    # Scenario 3: Low certainty (small sample)
    print("\n3. Small sample (10 trades, 60% win rate):")
    result_small = bayesian_win_rate(successes=6, total=10)
    print(f"   Posterior Mean: {result_small['posterior_mean']:.1%}")
    print(f"   Posterior Std: {result_small['posterior_std']:.4f}")
    print(f"   Credible Interval Width: {result_small['credible_upper'] - result_small['credible_lower']:.3f}")
    print(f"   Certainty: {result_small['certainty']:.2f} (LOW)")
    
    print("\nNote: Certainty score helps identify when estimates are reliable")


def demo_win_rate_by_group():
    """Demonstrate win rate calculation by group"""
    print("\n" + "=" * 80)
    print("DEMO 5: Win Rate by Group")
    print("=" * 80)
    
    # Generate synthetic data for different sessions
    np.random.seed(42)
    
    data = []
    sessions = ['ASIA', 'EUROPE', 'US']
    win_rates = [0.45, 0.65, 0.55]
    n_trades = [60, 80, 70]
    
    for session, wr, n in zip(sessions, win_rates, n_trades):
        trades = np.random.binomial(1, wr, n)
        for trade in trades:
            data.append({'trade_success': trade, 'session': session})
    
    df = pd.DataFrame(data)
    
    print("\nCalculating Bayesian win rate for each trading session:")
    results = bayesian_win_rate_by_group(df, 'trade_success', 'session', min_samples=30)
    
    print("\nResults:")
    print(results[['session', 'n', 'n_successes', 'posterior_mean', 'certainty', 'reliable']].to_string(index=False))
    
    print("\nInterpretation:")
    for _, row in results.iterrows():
        print(f"   {row['session']}: {row['posterior_mean']:.1%} win rate "
              f"(certainty: {row['certainty']:.2f}, "
              f"{'reliable' if row['reliable'] else 'unreliable'})")


def main():
    """Run all demos"""
    print("\n" + "=" * 80)
    print("BAYESIAN PROBABILITY CALCULATOR DEMO")
    print("=" * 80)
    print("\nThis demo shows how to use Bayesian methods for probability estimation")
    print("in trading analysis, particularly useful for small sample sizes.")
    
    demo_bayesian_win_rate()
    demo_adaptive_tracker()
    demo_bayesian_comparison()
    demo_certainty_calculation()
    demo_win_rate_by_group()
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("1. Bayesian methods provide more reliable estimates for small samples")
    print("2. Credible intervals quantify uncertainty in probability estimates")
    print("3. Adaptive tracking can detect regime changes in win rate")
    print("4. Bayesian comparison gives probability that one condition is better")
    print("5. Certainty scores help identify when estimates are reliable")
    print("\nFor more information, see:")
    print("- backend/calculators/bayesian_probability.py")
    print("- backend/calculators/test_bayesian_probability.py")


if __name__ == '__main__':
    main()
