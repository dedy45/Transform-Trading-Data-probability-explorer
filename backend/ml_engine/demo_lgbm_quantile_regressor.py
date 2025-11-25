"""
Demo: LightGBM Quantile Regressor Ensemble

This demo shows how to use the QuantileRegressorEnsemble to predict
R_multiple distribution (P10, P50, P90) for trading setups.

**Feature: ml-prediction-engine**
"""

import numpy as np
import pandas as pd
from pathlib import Path

from lgbm_quantile_regressor import (
    QuantileRegressorEnsemble,
    train_quantile_ensemble
)


def create_sample_data(n_samples=500):
    """Create sample trading data for demonstration."""
    np.random.seed(42)
    
    # Create features
    X = pd.DataFrame({
        'trend_strength': np.random.uniform(0, 1, n_samples),
        'volatility': np.random.uniform(0.5, 3.0, n_samples),
        'support_distance': np.random.uniform(-2, 2, n_samples),
        'momentum': np.random.uniform(-1, 1, n_samples),
        'time_of_day': np.random.randint(0, 24, n_samples)
    })
    
    # Create target (R_multiple) with some relationship to features
    y = (
        X['trend_strength'] * 3.0 +
        X['momentum'] * 2.0 -
        X['volatility'] * 0.5 +
        np.random.normal(0, 1.5, n_samples)
    )
    
    # Clip to realistic R_multiple range
    y = np.clip(y, -5, 10)
    y = pd.Series(y, name='R_multiple')
    
    return X, y


def demo_basic_usage():
    """Demo 1: Basic usage of quantile ensemble."""
    print("=" * 60)
    print("Demo 1: Basic Quantile Regression")
    print("=" * 60)
    
    # Create sample data
    X, y = create_sample_data(n_samples=500)
    
    # Split data
    split_idx = int(len(X) * 0.7)
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {list(X.columns)}")
    
    # Train ensemble
    print("\nTraining quantile ensemble...")
    ensemble = QuantileRegressorEnsemble(n_estimators=50, learning_rate=0.1)
    metrics = ensemble.fit(X_train, y_train, X_test, y_test)
    
    print("\nTraining Metrics:")
    print(f"  MAE P10 (train): {metrics['mae_p10_train']:.4f}")
    print(f"  MAE P50 (train): {metrics['mae_p50_train']:.4f}")
    print(f"  MAE P90 (train): {metrics['mae_p90_train']:.4f}")
    print(f"  MAE P10 (test):  {metrics['mae_p10_val']:.4f}")
    print(f"  MAE P50 (test):  {metrics['mae_p50_val']:.4f}")
    print(f"  MAE P90 (test):  {metrics['mae_p90_val']:.4f}")
    
    # Make predictions
    predictions = ensemble.predict(X_test.head(5))
    
    print("\nSample Predictions (first 5 test samples):")
    print("-" * 60)
    for i in range(5):
        print(f"Sample {i+1}:")
        print(f"  P10: {predictions['p10'][i]:6.2f}  (10th percentile)")
        print(f"  P50: {predictions['p50'][i]:6.2f}  (median)")
        print(f"  P90: {predictions['p90'][i]:6.2f}  (90th percentile)")
        print(f"  Actual: {y_test.iloc[i]:6.2f}")
        print()


def demo_skewness_calculation():
    """Demo 2: Calculate distribution skewness."""
    print("\n" + "=" * 60)
    print("Demo 2: Distribution Skewness")
    print("=" * 60)
    
    # Create sample data
    X, y = create_sample_data(n_samples=300)
    
    # Train ensemble
    ensemble = QuantileRegressorEnsemble(n_estimators=50)
    ensemble.fit(X, y)
    
    # Calculate skewness
    skewness = ensemble.calculate_skewness(X=X)
    
    print("\nSkewness Interpretation:")
    print("  Positive skew: Upside potential (P90-P50 > P50-P10)")
    print("  Negative skew: Downside risk (P90-P50 < P50-P10)")
    print("  Zero skew: Symmetric distribution")
    
    print(f"\nSkewness Statistics:")
    print(f"  Mean skewness: {np.mean(skewness):.4f}")
    print(f"  Median skewness: {np.median(skewness):.4f}")
    print(f"  Positive skew samples: {np.sum(skewness > 0)} ({np.sum(skewness > 0)/len(skewness)*100:.1f}%)")
    print(f"  Negative skew samples: {np.sum(skewness < 0)} ({np.sum(skewness < 0)/len(skewness)*100:.1f}%)")
    
    # Show examples
    predictions = ensemble.predict(X.head(3))
    skew_samples = skewness[:3]
    
    print("\nExample Predictions with Skewness:")
    print("-" * 60)
    for i in range(3):
        p10 = predictions['p10'][i]
        p50 = predictions['p50'][i]
        p90 = predictions['p90'][i]
        skew = skew_samples[i]
        
        print(f"Sample {i+1}:")
        print(f"  P10={p10:6.2f}, P50={p50:6.2f}, P90={p90:6.2f}")
        print(f"  Lower spread: {p50-p10:6.2f}")
        print(f"  Upper spread: {p90-p50:6.2f}")
        print(f"  Skewness: {skew:6.2f} ({'upside' if skew > 0 else 'downside'} bias)")
        print()


def demo_feature_importance():
    """Demo 3: Feature importance analysis."""
    print("\n" + "=" * 60)
    print("Demo 3: Feature Importance")
    print("=" * 60)
    
    # Create sample data
    X, y = create_sample_data(n_samples=300)
    
    # Train ensemble
    ensemble = QuantileRegressorEnsemble(n_estimators=50)
    ensemble.fit(X, y)
    
    # Get feature importance for each quantile
    print("\nFeature Importance by Quantile:")
    print("-" * 60)
    
    for quantile in ['p10', 'p50', 'p90']:
        importance = ensemble.get_feature_importance(quantile=quantile)
        print(f"\n{quantile.upper()} Model:")
        for _, row in importance.iterrows():
            print(f"  {row['rank']}. {row['feature']:20s} {row['importance']:8.2f}")


def demo_save_load():
    """Demo 4: Save and load models."""
    print("\n" + "=" * 60)
    print("Demo 4: Model Persistence")
    print("=" * 60)
    
    # Create sample data
    X, y = create_sample_data(n_samples=300)
    
    # Train and save
    print("\nTraining and saving models...")
    ensemble1, metrics = train_quantile_ensemble(
        X, y,
        n_estimators=50,
        save_path_prefix=Path('data_processed/models/demo_quantile')
    )
    
    print(f"Models saved to: data_processed/models/")
    print(f"  - demo_quantile_p10.pkl")
    print(f"  - demo_quantile_p50.pkl")
    print(f"  - demo_quantile_p90.pkl")
    print(f"  - demo_quantile_meta.pkl")
    
    # Make predictions with original
    pred1 = ensemble1.predict(X.head(3))
    
    # Load and predict
    print("\nLoading models...")
    ensemble2 = QuantileRegressorEnsemble()
    ensemble2.load(Path('data_processed/models/demo_quantile'))
    
    pred2 = ensemble2.predict(X.head(3))
    
    # Verify predictions match
    print("\nVerifying predictions match:")
    for quantile in ['p10', 'p50', 'p90']:
        diff = np.abs(pred1[quantile] - pred2[quantile])
        print(f"  {quantile.upper()}: max difference = {np.max(diff):.10f}")
    
    print("\n✓ Models loaded successfully!")


def demo_quantile_ordering():
    """Demo 5: Verify quantile ordering property."""
    print("\n" + "=" * 60)
    print("Demo 5: Quantile Ordering Property")
    print("=" * 60)
    
    # Create sample data
    X, y = create_sample_data(n_samples=300)
    
    # Train ensemble
    ensemble = QuantileRegressorEnsemble(n_estimators=50)
    ensemble.fit(X, y)
    
    # Make predictions
    predictions = ensemble.predict(X)
    
    # Check ordering
    p10 = predictions['p10']
    p50 = predictions['p50']
    p90 = predictions['p90']
    
    violations_p10_p50 = np.sum(p10 > p50)
    violations_p50_p90 = np.sum(p50 > p90)
    
    print(f"\nQuantile Ordering Check (P10 <= P50 <= P90):")
    print(f"  Total samples: {len(X)}")
    print(f"  P10 > P50 violations: {violations_p10_p50} ({violations_p10_p50/len(X)*100:.1f}%)")
    print(f"  P50 > P90 violations: {violations_p50_p90} ({violations_p50_p90/len(X)*100:.1f}%)")
    
    if violations_p10_p50 + violations_p50_p90 == 0:
        print("\n✓ Perfect quantile ordering!")
    else:
        print(f"\n⚠ Some violations (expected for independent models)")
        print("  Note: Models are trained independently, so strict ordering")
        print("  is not guaranteed for every sample.")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("LightGBM Quantile Regressor Ensemble Demo")
    print("=" * 60)
    
    # Run all demos
    demo_basic_usage()
    demo_skewness_calculation()
    demo_feature_importance()
    demo_save_load()
    demo_quantile_ordering()
    
    print("\n" + "=" * 60)
    print("All demos completed successfully!")
    print("=" * 60)
