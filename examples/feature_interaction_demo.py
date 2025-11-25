"""
Feature Interaction Analyzer Demo

This script demonstrates how to use the feature interaction analyzer
to find synergistic and interfering feature combinations.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from backend.calculators.feature_interaction import (
    calculate_interaction_effect,
    decompose_effects,
    find_top_interactions,
    classify_interaction,
    create_interaction_matrix
)


def generate_sample_data(n_trades=500):
    """Generate sample trading data with known interactions."""
    np.random.seed(42)
    
    # Generate features
    trend_strength = np.random.uniform(0, 1, n_trades)
    volatility = np.random.choice([0, 1, 2], n_trades)  # low, medium, high
    session = np.random.choice(['ASIA', 'EUROPE', 'US'], n_trades)
    entropy = np.random.uniform(0, 1, n_trades)
    hurst = np.random.uniform(0.3, 0.7, n_trades)
    
    # Create trade success with known interactions
    # Base probability
    base_prob = 0.5
    
    # Main effects
    trend_effect = trend_strength * 0.2
    vol_effect = (volatility == 1) * 0.1  # Medium volatility is best
    
    # Interaction: trend + medium volatility is synergistic
    interaction_effect = ((trend_strength > 0.6) & (volatility == 1)).astype(float) * 0.15
    
    # Calculate win probability
    win_prob = base_prob + trend_effect + vol_effect + interaction_effect
    win_prob = np.clip(win_prob, 0, 1)
    
    # Generate outcomes
    trade_success = (np.random.random(n_trades) < win_prob).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'trade_success': trade_success,
        'trend_strength': trend_strength,
        'volatility_regime': volatility,
        'session': session,
        'entropy': entropy,
        'hurst_exponent': hurst,
        'R_multiple': np.random.normal(0.5, 1.5, n_trades)
    })
    
    return df


def demo_basic_interaction():
    """Demo: Calculate interaction between two features."""
    print("=" * 70)
    print("DEMO 1: Basic Interaction Effect Calculation")
    print("=" * 70)
    
    df = generate_sample_data(500)
    
    result = calculate_interaction_effect(
        df=df,
        feature_a='trend_strength',
        feature_b='volatility_regime',
        target_column='trade_success',
        bins_a=3,
        bins_b=3
    )
    
    print(f"\nAnalyzing: trend_strength × volatility_regime")
    print(f"Baseline Win Rate: {result['baseline_rate']:.2%}")
    print(f"Main Effect (trend_strength): {result['main_effect_a']:+.4f}")
    print(f"Main Effect (volatility_regime): {result['main_effect_b']:+.4f}")
    print(f"Joint Effect: {result['joint_effect']:+.4f}")
    print(f"Interaction Effect: {result['interaction_effect']:+.4f}")
    print(f"Samples: {result['n_samples']}")
    
    interaction_type = classify_interaction(result['interaction_effect'])
    print(f"\nInteraction Type: {interaction_type.upper()}")
    
    if interaction_type == 'synergistic':
        print("✓ These features work better together than expected!")
    elif interaction_type == 'interfering':
        print("✗ These features interfere with each other.")
    else:
        print("○ These features work independently.")
    
    print()


def demo_decompose_effects():
    """Demo: Decompose effects into components."""
    print("=" * 70)
    print("DEMO 2: Effect Decomposition")
    print("=" * 70)
    
    df = generate_sample_data(500)
    
    result = decompose_effects(
        df=df,
        feature_a='trend_strength',
        feature_b='session',
        target_column='trade_success'
    )
    
    print(f"\nDecomposing effects for: {result['feature_a']} × {result['feature_b']}")
    print(f"Baseline Win Rate: {result['baseline_rate']:.2%}")
    print(f"\nEffect Breakdown:")
    print(f"  Main Effect A ({result['feature_a']}): {result['main_effect_A']:+.4f}")
    print(f"  Main Effect B ({result['feature_b']}): {result['main_effect_B']:+.4f}")
    print(f"  Interaction Effect: {result['interaction_effect']:+.4f}")
    print(f"  ─────────────────────────────────────")
    print(f"  Total Effect: {result['total_effect']:+.4f}")
    
    print()


def demo_top_interactions():
    """Demo: Find top feature interactions."""
    print("=" * 70)
    print("DEMO 3: Top Feature Interactions")
    print("=" * 70)
    
    df = generate_sample_data(500)
    
    features = [
        'trend_strength',
        'volatility_regime',
        'session',
        'entropy',
        'hurst_exponent'
    ]
    
    print(f"\nTesting all pairwise combinations of {len(features)} features...")
    print(f"Total pairs to test: {len(features) * (len(features) - 1) // 2}")
    
    top_interactions = find_top_interactions(
        df=df,
        features=features,
        target_column='trade_success',
        top_n=10,
        min_samples=30
    )
    
    print(f"\nTop {len(top_interactions)} Interactions (ranked by strength):")
    print("─" * 100)
    print(f"{'Rank':<6}{'Feature A':<20}{'Feature B':<20}{'Effect':<12}{'Type':<15}{'Samples':<10}")
    print("─" * 100)
    
    for idx, row in top_interactions.iterrows():
        rank = idx + 1
        effect_str = f"{row['interaction_effect']:+.4f}"
        print(f"{rank:<6}{row['feature_a']:<20}{row['feature_b']:<20}"
              f"{effect_str:<12}{row['interaction_type']:<15}{row['n_samples']:<10}")
    
    print()
    
    # Highlight synergistic combinations
    synergistic = top_interactions[top_interactions['interaction_type'] == 'synergistic']
    if len(synergistic) > 0:
        print(f"\n✓ Found {len(synergistic)} synergistic combinations:")
        for idx, row in synergistic.iterrows():
            print(f"  • {row['feature_a']} + {row['feature_b']}: "
                  f"{row['interaction_effect']:+.4f}")
    
    # Highlight interfering combinations
    interfering = top_interactions[top_interactions['interaction_type'] == 'interfering']
    if len(interfering) > 0:
        print(f"\n✗ Found {len(interfering)} interfering combinations:")
        for idx, row in interfering.iterrows():
            print(f"  • {row['feature_a']} + {row['feature_b']}: "
                  f"{row['interaction_effect']:+.4f}")
    
    print()


def demo_interaction_matrix():
    """Demo: Create interaction matrix."""
    print("=" * 70)
    print("DEMO 4: Interaction Matrix")
    print("=" * 70)
    
    df = generate_sample_data(500)
    
    features = [
        'trend_strength',
        'volatility_regime',
        'entropy',
        'hurst_exponent'
    ]
    
    print(f"\nCreating interaction matrix for {len(features)} features...")
    
    matrix = create_interaction_matrix(
        df=df,
        features=features,
        target_column='trade_success',
        bins=3
    )
    
    print("\nInteraction Matrix:")
    print("(Diagonal = main effects, Off-diagonal = interaction effects)")
    print()
    print(matrix.round(4))
    
    print("\nInterpretation:")
    print("  • Diagonal values: Main effect of each feature")
    print("  • Off-diagonal values: Interaction between feature pairs")
    print("  • Positive values: Synergistic (features work well together)")
    print("  • Negative values: Interfering (features conflict)")
    
    print()


def demo_practical_application():
    """Demo: Practical application for strategy optimization."""
    print("=" * 70)
    print("DEMO 5: Practical Application - Strategy Optimization")
    print("=" * 70)
    
    df = generate_sample_data(500)
    
    print("\nScenario: You want to optimize your trading strategy by finding")
    print("the best feature combinations for entry signals.")
    
    # Step 1: Find top interactions
    features = ['trend_strength', 'volatility_regime', 'session', 'entropy']
    
    print(f"\nStep 1: Testing {len(features)} features for interactions...")
    top_interactions = find_top_interactions(
        df=df,
        features=features,
        top_n=5,
        min_samples=30
    )
    
    # Step 2: Identify best combination
    best = top_interactions.iloc[0]
    print(f"\nStep 2: Best combination found:")
    print(f"  Features: {best['feature_a']} + {best['feature_b']}")
    print(f"  Interaction Effect: {best['interaction_effect']:+.4f}")
    print(f"  Type: {best['interaction_type']}")
    
    # Step 3: Analyze the combination in detail
    print(f"\nStep 3: Detailed analysis of best combination...")
    result = decompose_effects(
        df=df,
        feature_a=best['feature_a'],
        feature_b=best['feature_b']
    )
    
    print(f"  Baseline Win Rate: {result['baseline_rate']:.2%}")
    print(f"  With {best['feature_a']} only: "
          f"{result['baseline_rate'] + result['main_effect_A']:.2%}")
    print(f"  With {best['feature_b']} only: "
          f"{result['baseline_rate'] + result['main_effect_B']:.2%}")
    print(f"  With both features: "
          f"{result['baseline_rate'] + result['total_effect']:.2%}")
    
    # Step 4: Recommendation
    print(f"\nStep 4: Recommendation:")
    if best['interaction_type'] == 'synergistic':
        boost = result['total_effect'] * 100
        print(f"  ✓ Use both {best['feature_a']} and {best['feature_b']} together")
        print(f"  ✓ Expected win rate boost: {boost:+.2f} percentage points")
        print(f"  ✓ This combination shows strong synergy!")
    else:
        print(f"  ○ Consider using features independently")
    
    print()


def main():
    """Run all demos."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "FEATURE INTERACTION ANALYZER DEMO" + " " * 20 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    # Run demos
    demo_basic_interaction()
    demo_decompose_effects()
    demo_top_interactions()
    demo_interaction_matrix()
    demo_practical_application()
    
    print("=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. Interaction effects reveal how features work together")
    print("  2. Synergistic combinations boost performance beyond individual effects")
    print("  3. Interfering combinations should be avoided or used separately")
    print("  4. Use find_top_interactions() to discover the best feature pairs")
    print("  5. Validate findings with sufficient sample sizes")
    print()


if __name__ == '__main__':
    main()
