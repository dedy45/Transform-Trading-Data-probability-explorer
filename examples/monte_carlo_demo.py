"""
Monte Carlo Simulation Engine Demo

This script demonstrates how to use the Monte Carlo simulation engine
for trading strategy analysis.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from backend.calculators.monte_carlo_engine import (
    monte_carlo_simulation,
    calculate_percentile_bands,
    kelly_criterion_calculator,
    compare_risk_scenarios
)


def generate_sample_trades(n_trades=200):
    """Generate sample trade data for demonstration."""
    np.random.seed(42)
    
    # Generate R-multiples with realistic distribution
    # 55% win rate, average win 1.8R, average loss -0.9R
    wins = np.random.normal(1.8, 0.8, int(n_trades * 0.55))
    losses = np.random.normal(-0.9, 0.3, int(n_trades * 0.45))
    
    r_multiples = np.concatenate([wins, losses])
    np.random.shuffle(r_multiples)
    
    return pd.DataFrame({
        'R_multiple': r_multiples,
        'trade_success': (r_multiples > 0).astype(int)
    })


def demo_basic_monte_carlo():
    """Demonstrate basic Monte Carlo simulation."""
    print("=" * 70)
    print("DEMO 1: Basic Monte Carlo Simulation")
    print("=" * 70)
    
    # Generate sample data
    df = generate_sample_trades(200)
    
    print(f"\nHistorical Performance:")
    print(f"  Total Trades: {len(df)}")
    print(f"  Win Rate: {df['trade_success'].mean():.1%}")
    print(f"  Mean R: {df['R_multiple'].mean():.2f}")
    print(f"  Expectancy: {df['R_multiple'].mean():.2f}R per trade")
    
    # Run Monte Carlo simulation
    print(f"\nRunning Monte Carlo simulation (1000 iterations)...")
    result = monte_carlo_simulation(
        df=df,
        n_simulations=1000,
        initial_equity=10000.0,
        risk_per_trade=0.01,  # 1% risk per trade
        random_seed=42
    )
    
    print(f"\nMonte Carlo Results:")
    print(f"  Initial Equity: ${result['initial_equity']:,.2f}")
    print(f"  Risk per Trade: {result['risk_per_trade']:.1%}")
    print(f"  Median Final Equity: ${result['median_final_equity']:,.2f}")
    print(f"  5th Percentile: ${result['percentile_5_equity']:,.2f}")
    print(f"  95th Percentile: ${result['percentile_95_equity']:,.2f}")
    print(f"  Probability of Ruin: {result['prob_ruin']:.1%}")
    print(f"  Probability of 2x: {result['prob_reach_target']:.1%}")
    print(f"  95th Percentile Max DD: {result['percentile_95_dd']:.1f}%")
    
    # Calculate percentile bands for fan chart
    bands = calculate_percentile_bands(result['equity_curves'])
    print(f"\nPercentile Bands (for fan chart):")
    print(f"  Time Steps: {bands['time_steps']}")
    print(f"  Final Equity Bands:")
    print(f"    5th: ${bands['p5'][-1]:,.2f}")
    print(f"    25th: ${bands['p25'][-1]:,.2f}")
    print(f"    50th: ${bands['p50'][-1]:,.2f}")
    print(f"    75th: ${bands['p75'][-1]:,.2f}")
    print(f"    95th: ${bands['p95'][-1]:,.2f}")


def demo_kelly_criterion():
    """Demonstrate Kelly Criterion calculation."""
    print("\n" + "=" * 70)
    print("DEMO 2: Kelly Criterion Calculation")
    print("=" * 70)
    
    # Generate sample data
    df = generate_sample_trades(200)
    
    # Calculate Kelly Criterion
    kelly = kelly_criterion_calculator(df)
    
    print(f"\nKelly Criterion Results:")
    print(f"  Win Rate: {kelly['win_rate']:.1%}")
    print(f"  Average Win: {kelly['avg_win']:.2f}R")
    print(f"  Average Loss: {kelly['avg_loss']:.2f}R")
    print(f"  Win/Loss Ratio: {kelly['win_loss_ratio']:.2f}")
    print(f"\nOptimal Position Sizing:")
    print(f"  Full Kelly: {kelly['full_kelly']:.1%}")
    print(f"  Half Kelly: {kelly['half_kelly']:.1%} (recommended)")
    print(f"  Quarter Kelly: {kelly['quarter_kelly']:.1%} (conservative)")
    
    print(f"\nInterpretation:")
    if kelly['full_kelly'] > 0.05:
        print(f"  ✓ Strong positive expectancy detected")
        print(f"  ✓ Recommended risk: {kelly['half_kelly']:.1%} per trade")
    elif kelly['full_kelly'] > 0.02:
        print(f"  ⚠ Moderate positive expectancy")
        print(f"  ⚠ Recommended risk: {kelly['quarter_kelly']:.1%} per trade")
    else:
        print(f"  ✗ Weak or negative expectancy")
        print(f"  ✗ Consider strategy improvement before trading")


def demo_risk_comparison():
    """Demonstrate risk scenario comparison."""
    print("\n" + "=" * 70)
    print("DEMO 3: Risk Scenario Comparison")
    print("=" * 70)
    
    # Generate sample data
    df = generate_sample_trades(200)
    
    # Compare different risk levels
    print(f"\nComparing risk levels: 0.5%, 1%, 1.5%, 2%")
    print(f"Running 500 simulations per risk level...")
    
    comparison = compare_risk_scenarios(
        df=df,
        risk_levels=[0.005, 0.01, 0.015, 0.02],
        n_simulations=500,
        initial_equity=10000.0,
        random_seed=42
    )
    
    print(f"\nRisk Comparison Results:")
    print(f"\n{'Risk':<8} {'Median $':<12} {'5th %ile':<12} {'95th %ile':<12} {'P(Ruin)':<10} {'P(2x)':<10} {'Max DD':<10}")
    print("-" * 80)
    
    for _, row in comparison.iterrows():
        print(f"{row['risk_percent']:>6.1f}%  "
              f"${row['median_final_equity']:>10,.0f}  "
              f"${row['percentile_5_equity']:>10,.0f}  "
              f"${row['percentile_95_equity']:>10,.0f}  "
              f"{row['prob_ruin']:>8.1%}  "
              f"{row['prob_reach_target']:>8.1%}  "
              f"{row['percentile_95_dd']:>8.1f}%")
    
    print(f"\nKey Insights:")
    best_risk = comparison.loc[comparison['median_final_equity'].idxmax()]
    print(f"  • Highest median equity at {best_risk['risk_percent']:.1f}% risk")
    
    safest_risk = comparison.loc[comparison['prob_ruin'].idxmin()]
    print(f"  • Lowest ruin probability at {safest_risk['risk_percent']:.1f}% risk")
    
    print(f"  • Higher risk = higher potential returns but also higher drawdowns")
    print(f"  • Consider your risk tolerance and account size when choosing")


def demo_equity_curve_analysis():
    """Demonstrate equity curve generation and analysis."""
    print("\n" + "=" * 70)
    print("DEMO 4: Equity Curve Analysis")
    print("=" * 70)
    
    # Generate sample data
    df = generate_sample_trades(100)
    
    # Run simulation with fewer trades to show curve details
    result = monte_carlo_simulation(
        df=df,
        n_simulations=10,  # Just 10 for demonstration
        initial_equity=10000.0,
        risk_per_trade=0.01,
        max_trades_per_sim=50,  # Limit to 50 trades
        random_seed=42
    )
    
    print(f"\nGenerated {len(result['equity_curves'])} equity curves")
    print(f"Each curve has {len(result['equity_curves'][0])} data points")
    
    # Analyze first curve
    curve = result['equity_curves'][0]
    print(f"\nFirst Equity Curve Analysis:")
    print(f"  Starting Equity: ${curve[0]:,.2f}")
    print(f"  Final Equity: ${curve[-1]:,.2f}")
    print(f"  Return: {((curve[-1] - curve[0]) / curve[0]) * 100:+.1f}%")
    
    # Find peak and drawdown
    peak = max(curve)
    peak_idx = curve.index(peak)
    final_dd = ((peak - curve[-1]) / peak) * 100 if peak > 0 else 0
    
    print(f"  Peak Equity: ${peak:,.2f} (at trade {peak_idx})")
    print(f"  Drawdown from Peak: {final_dd:.1f}%")
    
    # Show distribution of final equities
    final_equities = result['final_equity_distribution']
    print(f"\nFinal Equity Distribution (10 simulations):")
    print(f"  Min: ${min(final_equities):,.2f}")
    print(f"  Max: ${max(final_equities):,.2f}")
    print(f"  Range: ${max(final_equities) - min(final_equities):,.2f}")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("MONTE CARLO SIMULATION ENGINE - DEMONSTRATION")
    print("=" * 70)
    
    demo_basic_monte_carlo()
    demo_kelly_criterion()
    demo_risk_comparison()
    demo_equity_curve_analysis()
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nNext Steps:")
    print("  1. Load your own trade data using pandas")
    print("  2. Run Monte Carlo simulations with different parameters")
    print("  3. Use Kelly Criterion to optimize position sizing")
    print("  4. Compare risk scenarios to find optimal risk level")
    print("  5. Visualize equity curves and percentile bands")
    print("\nFor more information, see:")
    print("  - backend/calculators/monte_carlo_engine.py")
    print("  - backend/calculators/test_monte_carlo_engine.py")
    print()


if __name__ == '__main__':
    main()
