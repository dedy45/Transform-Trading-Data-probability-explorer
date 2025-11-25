"""
Conditional Probability Engine Demo

This script demonstrates the usage of the conditional probability engine
for finding optimal trading conditions.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from backend.calculators.conditional_probability import (
    calculate_conditional_probability,
    calculate_lift_ratio,
    sequential_condition_builder,
    filter_by_thresholds,
    sort_by_probability_and_significance,
    find_top_combinations
)


def create_sample_data(n_samples=500):
    """Create sample trading data for demonstration."""
    np.random.seed(42)
    
    df = pd.DataFrame({
        'y_win': np.random.binomial(1, 0.52, n_samples),
        'y_hit_1R': np.random.binomial(1, 0.45, n_samples),
        'session': np.random.choice([0, 1, 2, 3], n_samples),  # ASIA, EUROPE, US, OVERLAP
        'trend_strength': np.random.uniform(0, 1, n_samples),
        'volatility_regime': np.random.choice([0, 1, 2], n_samples),  # LOW, MED, HIGH
        'trend_dir': np.random.choice([-1, 0, 1], n_samples),
        'R_multiple': np.random.normal(0.5, 1.5, n_samples)
    })
    
    # Make some correlations for more interesting results
    # Higher trend strength -> higher win rate
    mask = df['trend_strength'] > 0.7
    df.loc[mask, 'y_win'] = np.random.binomial(1, 0.65, mask.sum())
    
    # Europe session -> higher win rate
    mask = df['session'] == 1
    df.loc[mask, 'y_win'] = np.random.binomial(1, 0.60, mask.sum())
    
    return df


def demo_basic_conditional_probability():
    """Demo 1: Basic conditional probability calculation."""
    print("=" * 70)
    print("DEMO 1: Basic Conditional Probability")
    print("=" * 70)
    
    df = create_sample_data()
    
    # Calculate base rate
    base_rate = df['y_win'].mean()
    print(f"\nBase Win Rate: {base_rate:.3f}")
    
    # Calculate P(Win | Session = EUROPE)
    result = calculate_conditional_probability(
        df, 'y_win', {'session': 1}
    )
    
    print(f"\nP(Win | Session = EUROPE):")
    print(f"  Probability: {result['probability']:.3f}")
    print(f"  95% CI: [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]")
    print(f"  Sample Size: {result['n_samples']}")
    print(f"  Successes: {result['n_successes']}")
    
    # Calculate lift
    lift = calculate_lift_ratio(result['probability'], base_rate)
    print(f"  Lift Ratio: {lift:.2f}x")
    
    # Multiple conditions
    result2 = calculate_conditional_probability(
        df, 'y_win',
        {'session': 1, 'trend_strength': (0.5, 1.0)}
    )
    
    print(f"\nP(Win | Session = EUROPE AND Trend Strength > 0.5):")
    print(f"  Probability: {result2['probability']:.3f}")
    print(f"  95% CI: [{result2['ci_lower']:.3f}, {result2['ci_upper']:.3f}]")
    print(f"  Sample Size: {result2['n_samples']}")
    
    lift2 = calculate_lift_ratio(result2['probability'], base_rate)
    print(f"  Lift Ratio: {lift2:.2f}x")


def demo_sequential_condition_builder():
    """Demo 2: Sequential condition builder (greedy algorithm)."""
    print("\n" + "=" * 70)
    print("DEMO 2: Sequential Condition Builder (Greedy Algorithm)")
    print("=" * 70)
    
    df = create_sample_data()
    
    features = ['session', 'trend_strength', 'volatility_regime']
    
    results = sequential_condition_builder(
        df, 'y_win', features,
        max_conditions=3,
        min_samples=20,
        min_lift=1.05
    )
    
    print(f"\nGreedy Search Results:")
    print(f"{'Step':<6} {'Probability':<12} {'Lift':<8} {'N':<6} {'Conditions'}")
    print("-" * 70)
    
    for result in results:
        cond_str = str(result['conditions']) if result['conditions'] else "Base Rate"
        print(f"{result['step']:<6} {result['probability']:<12.3f} "
              f"{result['lift']:<8.2f} {result['n_samples']:<6} {cond_str}")


def demo_filtering_and_sorting():
    """Demo 3: Filtering and sorting results."""
    print("\n" + "=" * 70)
    print("DEMO 3: Filtering and Sorting Results")
    print("=" * 70)
    
    df = create_sample_data()
    
    features = ['session', 'trend_strength', 'volatility_regime']
    
    results = sequential_condition_builder(
        df, 'y_win', features,
        max_conditions=3,
        min_samples=10,
        min_lift=1.0
    )
    
    print(f"\nOriginal Results: {len(results)} steps")
    
    # Filter by thresholds
    filtered = filter_by_thresholds(results, min_samples=30, min_lift=1.1)
    print(f"After Filtering (n>=30, lift>=1.1): {len(filtered)} steps")
    
    # Sort by probability
    sorted_results = sort_by_probability_and_significance(filtered)
    
    print(f"\nTop Results (sorted by probability):")
    print(f"{'Rank':<6} {'Probability':<12} {'Lift':<8} {'N':<6} {'CI Width':<10}")
    print("-" * 70)
    
    for i, result in enumerate(sorted_results[:5], 1):
        ci_width = result['ci_upper'] - result['ci_lower']
        print(f"{i:<6} {result['probability']:<12.3f} "
              f"{result['lift']:<8.2f} {result['n_samples']:<6} {ci_width:<10.3f}")


def demo_find_top_combinations():
    """Demo 4: Find top N combinations."""
    print("\n" + "=" * 70)
    print("DEMO 4: Find Top Combinations")
    print("=" * 70)
    
    df = create_sample_data(n_samples=1000)  # Larger dataset
    
    features = ['session', 'trend_strength', 'volatility_regime', 'trend_dir']
    
    top_combos = find_top_combinations(
        df, 'y_win', features,
        top_n=5,
        min_samples=30,
        min_lift=1.1
    )
    
    print(f"\nTop 5 Condition Combinations:")
    print(top_combos.to_string(index=False))
    
    if len(top_combos) > 0:
        print(f"\nBest Combination:")
        best = top_combos.iloc[0]
        print(f"  Probability: {best['probability']:.3f}")
        print(f"  Lift: {best['lift']:.2f}x")
        print(f"  Sample Size: {best['n_samples']}")
        print(f"  Conditions: {best['conditions']}")


def demo_practical_use_case():
    """Demo 5: Practical use case - finding optimal trading windows."""
    print("\n" + "=" * 70)
    print("DEMO 5: Practical Use Case - Optimal Trading Windows")
    print("=" * 70)
    
    df = create_sample_data(n_samples=1000)
    
    # Find best conditions for hitting 1R target
    print("\nFinding optimal conditions for P(Hit 1R)...")
    
    features = ['session', 'trend_strength', 'volatility_regime']
    
    results = sequential_condition_builder(
        df, 'y_hit_1R', features,
        max_conditions=3,
        min_samples=50,
        min_lift=1.15
    )
    
    if len(results) > 1:
        best = results[-1]  # Last step has most conditions
        
        print(f"\nOptimal Trading Setup:")
        print(f"  Target: Hit 1R")
        print(f"  Probability: {best['probability']:.3f}")
        print(f"  Lift vs Base Rate: {best['lift']:.2f}x")
        print(f"  Sample Size: {best['n_samples']}")
        print(f"  Conditions:")
        for feature, value in best['conditions'].items():
            print(f"    - {feature}: {value}")
        
        # Calculate expected value improvement
        base_rate = results[0]['probability']
        improvement = best['probability'] - base_rate
        print(f"\n  Improvement: +{improvement:.3f} ({improvement/base_rate*100:.1f}%)")
    else:
        print("\nNo significant improvements found with current thresholds.")


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("CONDITIONAL PROBABILITY ENGINE DEMONSTRATION")
    print("=" * 70)
    
    demo_basic_conditional_probability()
    demo_sequential_condition_builder()
    demo_filtering_and_sorting()
    demo_find_top_combinations()
    demo_practical_use_case()
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
