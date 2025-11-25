"""
Calibration Models Demo

This script demonstrates how to use the calibration models to assess
the reliability of probability predictions.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from backend.models.calibration import (
    create_calibration_bins,
    compute_reliability_diagram,
    compute_brier_score,
    compute_ece
)


def main():
    print("=" * 70)
    print("CALIBRATION MODELS DEMO")
    print("=" * 70)
    print()
    
    # Generate synthetic probability predictions
    np.random.seed(42)
    n_samples = 1000
    
    # Create three scenarios:
    # 1. Well-calibrated predictions
    # 2. Overconfident predictions
    # 3. Underconfident predictions
    
    print("Generating synthetic probability predictions...")
    print()
    
    # Scenario 1: Well-calibrated
    true_probs = np.random.uniform(0, 1, n_samples)
    actual_outcomes_1 = (np.random.random(n_samples) < true_probs).astype(int)
    predicted_probs_1 = true_probs + np.random.normal(0, 0.05, n_samples)
    predicted_probs_1 = np.clip(predicted_probs_1, 0, 1)
    
    # Scenario 2: Overconfident (predictions too extreme)
    predicted_probs_2 = np.where(true_probs > 0.5, 
                                  true_probs + 0.2, 
                                  true_probs - 0.2)
    predicted_probs_2 = np.clip(predicted_probs_2, 0, 1)
    actual_outcomes_2 = (np.random.random(n_samples) < true_probs).astype(int)
    
    # Scenario 3: Underconfident (predictions too moderate)
    predicted_probs_3 = 0.5 + 0.3 * (true_probs - 0.5)
    actual_outcomes_3 = (np.random.random(n_samples) < true_probs).astype(int)
    
    # Analyze each scenario
    scenarios = [
        ("Well-Calibrated", predicted_probs_1, actual_outcomes_1),
        ("Overconfident", predicted_probs_2, actual_outcomes_2),
        ("Underconfident", predicted_probs_3, actual_outcomes_3)
    ]
    
    for scenario_name, pred_probs, actual_out in scenarios:
        print("=" * 70)
        print(f"SCENARIO: {scenario_name}")
        print("=" * 70)
        print()
        
        # 1. Compute Brier Score
        brier = compute_brier_score(pred_probs, actual_out)
        print(f"Brier Score: {brier:.4f}")
        print(f"  (Lower is better, 0 = perfect, 1 = worst)")
        print()
        
        # 2. Compute Expected Calibration Error
        ece = compute_ece(pred_probs, actual_out, n_bins=10, strategy='quantile')
        print(f"Expected Calibration Error (ECE): {ece:.4f}")
        print(f"  (Lower is better, 0 = perfect calibration)")
        print()
        
        # 3. Compute Reliability Diagram
        reliability = compute_reliability_diagram(
            pred_probs, actual_out, n_bins=10, strategy='quantile'
        )
        
        print("Reliability Diagram (10 bins):")
        print("-" * 70)
        print(f"{'Bin':<5} {'Mean Pred':<12} {'Obs Freq':<12} {'Deviation':<12} {'N Samples':<10}")
        print("-" * 70)
        
        for i in range(len(reliability['mean_predicted'])):
            if reliability['n_samples'][i] > 0:
                mean_pred = reliability['mean_predicted'][i]
                obs_freq = reliability['observed_frequency'][i]
                deviation = abs(mean_pred - obs_freq)
                n_samp = reliability['n_samples'][i]
                
                print(f"{i+1:<5} {mean_pred:<12.4f} {obs_freq:<12.4f} "
                      f"{deviation:<12.4f} {n_samp:<10}")
        
        print("-" * 70)
        print()
        
        # 4. Show calibration quality assessment
        if ece < 0.05:
            quality = "Excellent"
        elif ece < 0.10:
            quality = "Good"
        elif ece < 0.15:
            quality = "Fair"
        else:
            quality = "Poor"
        
        print(f"Calibration Quality: {quality}")
        print()
    
    # Demonstrate binning strategies
    print("=" * 70)
    print("BINNING STRATEGIES COMPARISON")
    print("=" * 70)
    print()
    
    test_probs = predicted_probs_1[:100]
    
    for strategy in ['uniform', 'quantile']:
        print(f"Strategy: {strategy}")
        print("-" * 70)
        
        bin_edges, bin_indices = create_calibration_bins(
            test_probs, n_bins=5, strategy=strategy
        )
        
        print(f"Bin Edges: {bin_edges}")
        print(f"Samples per bin:")
        
        for i in range(len(bin_edges) - 1):
            count = np.sum(bin_indices == i)
            print(f"  Bin {i+1}: {count} samples")
        
        print()
    
    print("=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
