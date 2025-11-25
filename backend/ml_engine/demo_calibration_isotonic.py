"""
Demo: Isotonic Calibration for Probability Predictions

This script demonstrates how to use the IsotonicCalibrator to calibrate
probability predictions from a binary classifier.

Usage:
    python backend/ml_engine/demo_calibration_isotonic.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

from calibration_isotonic import IsotonicCalibrator, calibrate_probabilities


def generate_synthetic_data(n_samples=1000, seed=42):
    """
    Generate synthetic calibration data.
    
    Creates poorly calibrated probabilities that need calibration.
    """
    np.random.seed(seed)
    
    # Generate true probabilities
    true_probs = np.random.beta(2, 2, n_samples)
    
    # Generate outcomes based on true probabilities
    y_true = np.random.binomial(1, true_probs)
    
    # Generate poorly calibrated predictions (compressed range)
    # Raw predictions are too conservative (compressed toward 0.5)
    raw_probs = 0.3 + 0.4 * true_probs + np.random.normal(0, 0.05, n_samples)
    raw_probs = np.clip(raw_probs, 0.01, 0.99)
    
    return raw_probs, y_true


def demo_basic_calibration():
    """Demonstrate basic calibration workflow."""
    print("=" * 70)
    print("DEMO 1: Basic Calibration Workflow")
    print("=" * 70)
    
    # Generate data
    print("\n1. Generating synthetic calibration data...")
    raw_probs, y_true = generate_synthetic_data(n_samples=500)
    print(f"   Generated {len(raw_probs)} samples")
    print(f"   Raw probability range: [{raw_probs.min():.3f}, {raw_probs.max():.3f}]")
    print(f"   Actual win rate: {y_true.mean():.3f}")
    
    # Create and fit calibrator
    print("\n2. Fitting isotonic calibrator...")
    calibrator = IsotonicCalibrator()
    metrics = calibrator.fit(raw_probs, y_true)
    
    print("\n3. Calibration Metrics:")
    print(f"   Brier Score:")
    print(f"     Before: {metrics['brier_before']:.4f}")
    print(f"     After:  {metrics['brier_after']:.4f}")
    print(f"     Improvement: {metrics['brier_improvement']:.4f}")
    print(f"   Expected Calibration Error (ECE):")
    print(f"     Before: {metrics['ece_before']:.4f}")
    print(f"     After:  {metrics['ece_after']:.4f}")
    print(f"     Improvement: {metrics['ece_improvement']:.4f}")
    
    # Transform probabilities
    print("\n4. Transforming probabilities...")
    test_probs = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
    calibrated = calibrator.transform(test_probs)
    
    print("\n   Example transformations:")
    print("   Raw Prob  →  Calibrated Prob")
    print("   " + "-" * 30)
    for raw, cal in zip(test_probs, calibrated):
        print(f"   {raw:.2f}      →  {cal:.3f}")
    
    return calibrator


def demo_save_load():
    """Demonstrate saving and loading calibrator."""
    print("\n" + "=" * 70)
    print("DEMO 2: Save and Load Calibrator")
    print("=" * 70)
    
    # Generate and fit calibrator
    print("\n1. Training calibrator...")
    raw_probs, y_true = generate_synthetic_data(n_samples=500)
    calibrator = IsotonicCalibrator()
    calibrator.fit(raw_probs, y_true)
    
    # Save calibrator
    save_path = Path('data_processed/models/demo_calibrator.pkl')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n2. Saving calibrator to: {save_path}")
    calibrator.save(save_path)
    print("   ✓ Saved successfully")
    
    # Load calibrator
    print("\n3. Loading calibrator...")
    calibrator2 = IsotonicCalibrator()
    calibrator2.load(save_path)
    print("   ✓ Loaded successfully")
    
    # Verify predictions match
    test_probs = np.array([0.3, 0.5, 0.7])
    pred1 = calibrator.transform(test_probs)
    pred2 = calibrator2.transform(test_probs)
    
    print("\n4. Verifying predictions match:")
    print(f"   Original predictions: {pred1}")
    print(f"   Loaded predictions:   {pred2}")
    print(f"   Match: {np.allclose(pred1, pred2)}")


def demo_reliability_diagram():
    """Demonstrate reliability diagram plotting."""
    print("\n" + "=" * 70)
    print("DEMO 3: Reliability Diagram")
    print("=" * 70)
    
    # Generate and fit calibrator
    print("\n1. Training calibrator...")
    raw_probs, y_true = generate_synthetic_data(n_samples=500)
    calibrator = IsotonicCalibrator()
    calibrator.fit(raw_probs, y_true)
    
    # Create reliability diagram
    print("\n2. Creating reliability diagram...")
    fig = calibrator.plot_reliability_diagram(n_bins=10)
    
    print("   ✓ Reliability diagram created")
    print("   The diagram shows:")
    print("     - Perfect calibration line (diagonal)")
    print("     - Raw probabilities (red)")
    print("     - Calibrated probabilities (green)")
    print("\n   Note: In a well-calibrated model, points should be close to the diagonal.")
    
    # Save plot
    output_path = Path('output/calibration_reliability_diagram.html')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    print(f"\n   Plot saved to: {output_path}")


def demo_convenience_function():
    """Demonstrate convenience function."""
    print("\n" + "=" * 70)
    print("DEMO 4: Convenience Function")
    print("=" * 70)
    
    # Generate data
    print("\n1. Generating data...")
    raw_probs, y_true = generate_synthetic_data(n_samples=500)
    
    # Use convenience function
    print("\n2. Using calibrate_probabilities() convenience function...")
    calibrator, calibrated_probs, metrics = calibrate_probabilities(
        raw_probs,
        y_true,
        save_path='data_processed/models/demo_calibrator_convenience.pkl'
    )
    
    print("\n3. Results:")
    print(f"   Calibrator fitted: {calibrator.is_fitted}")
    print(f"   Calibrated {len(calibrated_probs)} probabilities")
    print(f"   Brier improvement: {metrics['brier_improvement']:.4f}")
    print(f"   ECE improvement: {metrics['ece_improvement']:.4f}")
    print("   ✓ Model saved automatically")


def demo_real_world_example():
    """Demonstrate real-world usage with classifier output."""
    print("\n" + "=" * 70)
    print("DEMO 5: Real-World Example - Calibrating Classifier Output")
    print("=" * 70)
    
    print("\n1. Simulating classifier predictions...")
    print("   (In real usage, these would come from LGBMClassifier)")
    
    # Simulate classifier output on calibration set
    np.random.seed(42)
    n_calib = 300
    
    # Classifier tends to be overconfident
    raw_probs_calib = np.random.beta(2, 2, n_calib)
    raw_probs_calib = np.clip(raw_probs_calib * 1.2, 0.01, 0.99)  # Overconfident
    
    # True outcomes
    true_probs = raw_probs_calib * 0.8  # Actual win rate is lower
    y_calib = np.random.binomial(1, true_probs)
    
    print(f"   Calibration set: {n_calib} samples")
    print(f"   Mean raw probability: {raw_probs_calib.mean():.3f}")
    print(f"   Actual win rate: {y_calib.mean():.3f}")
    print(f"   → Classifier is overconfident!")
    
    # Fit calibrator
    print("\n2. Fitting calibrator to fix overconfidence...")
    calibrator = IsotonicCalibrator()
    metrics = calibrator.fit(raw_probs_calib, y_calib)
    
    print(f"\n3. Calibration improved predictions:")
    print(f"   Brier score: {metrics['brier_before']:.4f} → {metrics['brier_after']:.4f}")
    print(f"   ECE: {metrics['ece_before']:.4f} → {metrics['ece_after']:.4f}")
    
    # Apply to test set
    print("\n4. Applying calibration to test set...")
    n_test = 100
    raw_probs_test = np.random.beta(2, 2, n_test)
    raw_probs_test = np.clip(raw_probs_test * 1.2, 0.01, 0.99)
    
    calibrated_test = calibrator.transform(raw_probs_test)
    
    print(f"   Test set: {n_test} samples")
    print(f"   Mean raw probability: {raw_probs_test.mean():.3f}")
    print(f"   Mean calibrated probability: {calibrated_test.mean():.3f}")
    print(f"   → Calibrated probabilities are more realistic")
    
    # Show examples
    print("\n5. Example predictions:")
    print("   Raw Prob  →  Calibrated Prob  |  Interpretation")
    print("   " + "-" * 60)
    examples = [(0.9, "Very confident"), (0.7, "Confident"), (0.5, "Uncertain")]
    for raw_p, desc in examples:
        cal_p = calibrator.transform(np.array([raw_p]))[0]
        print(f"   {raw_p:.2f}      →  {cal_p:.3f}           |  {desc}")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("ISOTONIC CALIBRATION DEMO")
    print("=" * 70)
    print("\nThis demo shows how to use IsotonicCalibrator to calibrate")
    print("probability predictions from binary classifiers.")
    
    # Run demos
    demo_basic_calibration()
    demo_save_load()
    demo_reliability_diagram()
    demo_convenience_function()
    demo_real_world_example()
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. Isotonic calibration improves probability reliability")
    print("  2. Brier score and ECE measure calibration quality")
    print("  3. Calibrated probabilities better match observed frequencies")
    print("  4. Use reliability diagrams to visualize calibration")
    print("  5. Save/load calibrators for production use")
    print("\nNext Steps:")
    print("  - Integrate with LGBMClassifier (Task 4)")
    print("  - Use in PredictionPipeline (Task 8)")
    print("  - Create frontend visualization (Task 14)")


if __name__ == '__main__':
    main()
