"""
Regime Analysis Demo

This script demonstrates the usage of the regime analysis module
for analyzing trading performance across different market regimes.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from backend.calculators.regime_analysis import (
    compute_regime_probabilities,
    compute_regime_threshold_probs,
    create_regime_comparison_table,
    filter_by_regime,
    compute_regime_transition_matrix
)


def generate_sample_data(n_trades=200):
    """Generate sample trade data with regime information"""
    np.random.seed(42)
    
    # Generate regime sequence (0=ranging, 1=trending, 2=volatile)
    regimes = np.random.choice([0, 1, 2], size=n_trades, p=[0.4, 0.4, 0.2])
    
    # Generate timestamps
    timestamps = pd.date_range('2024-01-01', periods=n_trades, freq='1H')
    
    # Generate R-multiples based on regime
    # Trending regime has better performance
    r_multiples = []
    trade_success = []
    
    for regime in regimes:
        if regime == 0:  # Ranging - lower performance
            r = np.random.normal(0.3, 1.5)
        elif regime == 1:  # Trending - better performance
            r = np.random.normal(1.0, 1.8)
        else:  # Volatile - mixed performance
            r = np.random.normal(0.5, 2.5)
        
        r_multiples.append(r)
        trade_success.append(1 if r > 0 else 0)
    
    df = pd.DataFrame({
        'Timestamp': timestamps,
        'regime': regimes,
        'R_multiple': r_multiples,
        'trade_success': trade_success
    })
    
    return df


def main():
    print("=" * 80)
    print("REGIME ANALYSIS DEMO")
    print("=" * 80)
    print()
    
    # Generate sample data
    print("Generating sample trade data...")
    trades_df = generate_sample_data(n_trades=200)
    print(f"Generated {len(trades_df)} trades")
    print(f"Regimes: 0=Ranging, 1=Trending, 2=Volatile")
    print()
    
    # 1. Compute regime probabilities
    print("-" * 80)
    print("1. WIN RATES BY REGIME")
    print("-" * 80)
    regime_probs = compute_regime_probabilities(
        df=trades_df,
        regime_column='regime',
        target_column='trade_success',
        conf_level=0.95,
        min_samples=5
    )
    
    print(regime_probs.to_string(index=False))
    print()
    
    # 2. Compute regime threshold probabilities
    print("-" * 80)
    print("2. R-MULTIPLE THRESHOLD PROBABILITIES BY REGIME")
    print("-" * 80)
    threshold_probs = compute_regime_threshold_probs(
        df=trades_df,
        regime_column='regime',
        r_column='R_multiple',
        thresholds=[1.0, 2.0],
        conf_level=0.95,
        min_samples=5
    )
    
    print(threshold_probs.to_string(index=False))
    print()
    
    # 3. Create comprehensive comparison table
    print("-" * 80)
    print("3. COMPREHENSIVE REGIME COMPARISON")
    print("-" * 80)
    comparison = create_regime_comparison_table(
        df=trades_df,
        regime_column='regime',
        target_column='trade_success',
        r_column='R_multiple',
        conf_level=0.95,
        min_samples=5
    )
    
    print(comparison.to_string(index=False))
    print()
    
    # 4. Filter by regime
    print("-" * 80)
    print("4. FILTERING BY REGIME")
    print("-" * 80)
    
    # Find best performing regime
    best_regime = comparison.loc[comparison['win_rate'].idxmax(), 'regime']
    print(f"Best performing regime: {best_regime}")
    print(f"Win rate: {comparison.loc[comparison['regime'] == best_regime, 'win_rate'].values[0]:.2%}")
    print()
    
    # Filter for best regime
    filtered_trades = filter_by_regime(
        df=trades_df,
        regime_column='regime',
        selected_regimes=best_regime
    )
    print(f"Filtered to {len(filtered_trades)} trades in regime {best_regime}")
    print(f"Original dataset: {len(trades_df)} trades")
    print(f"Percentage retained: {len(filtered_trades) / len(trades_df) * 100:.1f}%")
    print()
    
    # Filter for multiple regimes
    selected_regimes = [0, 1]  # Ranging and Trending
    filtered_trades_multi = filter_by_regime(
        df=trades_df,
        regime_column='regime',
        selected_regimes=selected_regimes
    )
    print(f"Filtered to regimes {selected_regimes}: {len(filtered_trades_multi)} trades")
    print()
    
    # 5. Compute regime transition matrix
    print("-" * 80)
    print("5. REGIME TRANSITION MATRIX")
    print("-" * 80)
    transitions = compute_regime_transition_matrix(
        df=trades_df,
        regime_column='regime',
        timestamp_column='Timestamp',
        conf_level=0.95
    )
    
    print("Transition Probabilities:")
    print(transitions['transition_matrix'].round(3))
    print()
    
    print("Transition Counts:")
    print(transitions['transition_counts'])
    print()
    
    # Show specific transition with confidence interval
    print("Example: Transition from Regime 0 to Regime 1")
    if 0 in transitions['confidence_intervals'] and 1 in transitions['confidence_intervals'][0]:
        ci_info = transitions['confidence_intervals'][0][1]
        print(f"  Probability: {ci_info['probability']:.3f}")
        print(f"  95% CI: [{ci_info['ci_lower']:.3f}, {ci_info['ci_upper']:.3f}]")
        print(f"  Count: {ci_info['count']} out of {ci_info['total']} transitions")
    print()
    
    # 6. Insights and recommendations
    print("-" * 80)
    print("6. INSIGHTS AND RECOMMENDATIONS")
    print("-" * 80)
    
    # Find regimes with reliable data
    reliable_regimes = comparison[comparison['reliable'] == True]
    
    if len(reliable_regimes) > 0:
        print("Regimes with sufficient data (reliable=True):")
        for _, row in reliable_regimes.iterrows():
            print(f"  Regime {row['regime']}: {row['n_trades']} trades, "
                  f"Win Rate: {row['win_rate']:.2%}, Mean R: {row['mean_r']:.2f}")
        print()
        
        # Recommend best regime
        best_row = reliable_regimes.loc[reliable_regimes['mean_r'].idxmax()]
        print(f"Recommended regime for trading: {best_row['regime']}")
        print(f"  Reason: Highest mean R-multiple ({best_row['mean_r']:.2f})")
        print(f"  Win Rate: {best_row['win_rate']:.2%}")
        print(f"  Sample Size: {best_row['n_trades']} trades")
    else:
        print("Warning: No regimes have sufficient sample size for reliable analysis")
    
    print()
    print("=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
