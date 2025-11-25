"""
Demo: Prediction Pipeline Usage

This script demonstrates how to use the PredictionPipeline to orchestrate
all 4 ML components for end-to-end predictions.

**Feature: ml-prediction-engine**
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.ml_engine.pipeline_prediction import PredictionPipeline, create_pipeline
from backend.ml_engine.lgbm_classifier import LGBMClassifierWrapper
from backend.ml_engine.calibration_isotonic import IsotonicCalibrator
from backend.ml_engine.lgbm_quantile_regressor import QuantileRegressorEnsemble
from backend.ml_engine.conformal_engine import ConformalEngine


def create_sample_data(n_samples=1000):
    """Create sample trading data."""
    np.random.seed(42)
    
    # Create features
    X = pd.DataFrame({
        'trend_strength': np.random.uniform(0, 1, n_samples),
        'swing_position': np.random.uniform(0, 1, n_samples),
        'volatility': np.random.uniform(0, 2, n_samples),
        'support_distance': np.random.uniform(-10, 10, n_samples),
        'momentum': np.random.uniform(-1, 1, n_samples)
    })
    
    # Create targets (simplified relationships)
    y_win = (
        (X['trend_strength'] > 0.5) & 
        (X['momentum'] > 0) & 
        (X['support_distance'] > 0)
    ).astype(int)
    
    y_R = (
        X['trend_strength'] * 2 + 
        X['momentum'] * 1.5 + 
        np.random.randn(n_samples) * 0.5
    )
    
    return X, y_win, y_R


def train_and_save_models(model_dir='data_processed/models'):
    """Train all 4 components and save to disk."""
    print("=" * 60)
    print("STEP 1: Training All Components")
    print("=" * 60)
    
    # Create sample data
    X, y_win, y_R = create_sample_data(1000)
    
    # Split data
    n_train = 600
    n_calib = 200
    
    X_train = X.iloc[:n_train]
    y_win_train = y_win.iloc[:n_train]
    y_R_train = y_R.iloc[:n_train]
    
    X_calib = X.iloc[n_train:n_train+n_calib]
    y_win_calib = y_win.iloc[n_train:n_train+n_calib]
    y_R_calib = y_R.iloc[n_train:n_train+n_calib]
    
    X_test = X.iloc[n_train+n_calib:]
    y_win_test = y_win.iloc[n_train+n_calib:]
    y_R_test = y_R.iloc[n_train+n_calib:]
    
    # Create model directory
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Train Classifier
    print("\n1. Training LightGBM Classifier...")
    classifier = LGBMClassifierWrapper(n_estimators=50, learning_rate=0.05)
    metrics = classifier.fit(X_train, y_win_train, X_calib, y_win_calib, use_cv=False)
    print(f"   AUC Train: {metrics['auc_train']:.4f}")
    print(f"   AUC Val: {metrics['auc_val']:.4f}")
    classifier.save(model_dir / 'lgbm_classifier.pkl')
    print("   ✓ Saved to lgbm_classifier.pkl")
    
    # 2. Train Calibrator
    print("\n2. Training Isotonic Calibrator...")
    raw_probs = classifier.predict_proba(X_calib)
    calibrator = IsotonicCalibrator()
    cal_metrics = calibrator.fit(raw_probs, y_win_calib.values)
    print(f"   Brier Before: {cal_metrics['brier_before']:.4f}")
    print(f"   Brier After: {cal_metrics['brier_after']:.4f}")
    print(f"   Improvement: {cal_metrics['brier_improvement']:.4f}")
    calibrator.save(model_dir / 'isotonic_calibrator.pkl')
    print("   ✓ Saved to isotonic_calibrator.pkl")
    
    # 3. Train Quantile Ensemble
    print("\n3. Training Quantile Regression Ensemble...")
    quantile_ensemble = QuantileRegressorEnsemble(n_estimators=50, learning_rate=0.05)
    q_metrics = quantile_ensemble.fit(X_train, y_R_train, X_calib, y_R_calib)
    print(f"   MAE P10: {q_metrics['mae_p10_val']:.4f}")
    print(f"   MAE P50: {q_metrics['mae_p50_val']:.4f}")
    print(f"   MAE P90: {q_metrics['mae_p90_val']:.4f}")
    quantile_ensemble.save(model_dir / 'lgbm_quantile')
    print("   ✓ Saved to lgbm_quantile_*.pkl")
    
    # 4. Train Conformal Engine
    print("\n4. Training Conformal Prediction Engine...")
    quantile_preds = quantile_ensemble.predict(X_calib)
    conformal_engine = ConformalEngine(coverage=0.9)
    conf_metrics = conformal_engine.fit(
        quantile_preds['p10'],
        quantile_preds['p90'],
        y_R_calib.values
    )
    print(f"   Target Coverage: {conf_metrics['coverage']:.2%}")
    print(f"   Nonconformity Quantile: {conf_metrics['nonconformity_quantile']:.4f}")
    conformal_engine.save(model_dir / 'conformal_meta.json')
    print("   ✓ Saved to conformal_meta.json")
    
    print("\n✓ All models trained and saved successfully!")
    
    return model_dir, X_test, y_win_test, y_R_test


def demo_single_prediction(pipeline, X_test):
    """Demonstrate single sample prediction."""
    print("\n" + "=" * 60)
    print("STEP 2: Single Sample Prediction")
    print("=" * 60)
    
    # Get a sample
    sample = X_test.iloc[0].to_dict()
    
    print("\nInput Features:")
    for feature, value in sample.items():
        print(f"  {feature}: {value:.4f}")
    
    # Make prediction
    result = pipeline.predict_for_sample(sample)
    
    print("\nPrediction Results:")
    print(f"  Raw Probability: {result['prob_win_raw']:.4f}")
    print(f"  Calibrated Probability: {result['prob_win_calibrated']:.4f}")
    print(f"  R_multiple P10: {result['R_P10_raw']:.4f}")
    print(f"  R_multiple P50: {result['R_P50_raw']:.4f}")
    print(f"  R_multiple P90: {result['R_P90_raw']:.4f}")
    print(f"  Conformal P10: {result['R_P10_conf']:.4f}")
    print(f"  Conformal P90: {result['R_P90_conf']:.4f}")
    print(f"  Skewness: {result['skewness']:.4f}")
    print(f"  Quality: {result['quality_label']}")
    print(f"  Recommendation: {result['recommendation']}")
    print(f"  Execution Time: {result['execution_time_ms']:.2f} ms")
    
    print("\nComponent Execution Times:")
    for component, time_ms in result['component_times_ms'].items():
        print(f"  {component}: {time_ms:.2f} ms")


def demo_batch_prediction(pipeline, X_test):
    """Demonstrate batch prediction."""
    print("\n" + "=" * 60)
    print("STEP 3: Batch Prediction")
    print("=" * 60)
    
    # Predict on test set
    print(f"\nPredicting on {len(X_test)} samples...")
    results = pipeline.predict_for_batch(X_test)
    
    print(f"✓ Batch prediction completed")
    print(f"\nResults shape: {results.shape}")
    print(f"Columns: {list(results.columns)}")
    
    # Show summary statistics
    print("\nSummary Statistics:")
    print(f"  Mean Calibrated Probability: {results['prob_win_calibrated'].mean():.4f}")
    print(f"  Mean R_P50: {results['R_P50_raw'].mean():.4f}")
    print(f"  Mean Skewness: {results['skewness'].mean():.4f}")
    
    # Show quality distribution
    print("\nQuality Distribution:")
    quality_counts = results['quality_label'].value_counts()
    for quality, count in quality_counts.items():
        pct = count / len(results) * 100
        print(f"  {quality}: {count} ({pct:.1f}%)")
    
    # Show recommendation distribution
    print("\nRecommendation Distribution:")
    rec_counts = results['recommendation'].value_counts()
    for rec, count in rec_counts.items():
        pct = count / len(results) * 100
        print(f"  {rec}: {count} ({pct:.1f}%)")
    
    # Show top 5 setups
    print("\nTop 5 Setups (by calibrated probability):")
    top_5 = results.nlargest(5, 'prob_win_calibrated')
    for idx, row in top_5.iterrows():
        print(f"  Sample {idx}: Prob={row['prob_win_calibrated']:.4f}, "
              f"R_P50={row['R_P50_raw']:.4f}, Quality={row['quality_label']}")


def demo_pipeline_info(pipeline):
    """Demonstrate getting pipeline information."""
    print("\n" + "=" * 60)
    print("STEP 4: Pipeline Information")
    print("=" * 60)
    
    info = pipeline.get_pipeline_info()
    
    print(f"\nPipeline Status:")
    print(f"  Loaded: {info['is_loaded']}")
    print(f"  Number of Features: {info['n_features']}")
    print(f"  Feature Names: {info['feature_names']}")
    
    print(f"\nComponent Status:")
    for component, status in info['components'].items():
        status_str = "✓ Fitted" if status else "✗ Not Fitted"
        print(f"  {component}: {status_str}")


def main():
    """Run the complete demo."""
    print("\n" + "=" * 60)
    print("PREDICTION PIPELINE DEMO")
    print("=" * 60)
    
    # Step 1: Train and save models
    model_dir, X_test, y_win_test, y_R_test = train_and_save_models()
    
    # Load pipeline
    print("\n" + "=" * 60)
    print("Loading Pipeline")
    print("=" * 60)
    print(f"\nLoading models from: {model_dir}")
    
    pipeline = create_pipeline(model_dir)
    print("✓ Pipeline loaded successfully")
    
    # Step 2: Single prediction
    demo_single_prediction(pipeline, X_test)
    
    # Step 3: Batch prediction
    demo_batch_prediction(pipeline, X_test)
    
    # Step 4: Pipeline info
    demo_pipeline_info(pipeline)
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("  1. Pipeline orchestrates 4 ML components seamlessly")
    print("  2. Single predictions complete in < 100ms")
    print("  3. Batch predictions are efficient and vectorized")
    print("  4. Setup categorization provides actionable recommendations")
    print("  5. All predictions include uncertainty quantification")


if __name__ == '__main__':
    main()
