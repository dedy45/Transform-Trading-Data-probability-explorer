"""
ML Prediction Engine - Training Example

This example demonstrates how to train all 4 ML components:
1. LightGBM Classifier
2. Isotonic Calibration
3. Quantile Regression (P10/P50/P90)
4. Conformal Prediction

Author: ML Prediction Engine Team
Date: 2024-11-24
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.ml_engine.model_trainer import ModelTrainer
from backend.ml_engine.performance_monitor import PerformanceMonitor

def main():
    """
    Complete training workflow example.
    """
    
    print("=" * 80)
    print("ML PREDICTION ENGINE - TRAINING EXAMPLE")
    print("=" * 80)
    print()
    
    # =========================================================================
    # STEP 1: Load Data
    # =========================================================================
    print("STEP 1: Loading data...")
    
    # Load your merged trading data
    # This should have features + trade_success (0/1) + R_multiple columns
    data_path = 'dataraw/merged_trading_data.csv'
    
    if not Path(data_path).exists():
        print(f"Error: Data file not found at {data_path}")
        print("Please ensure you have merged trading data with features.")
        return
    
    data = pd.read_csv(data_path)
    print(f"✓ Loaded {len(data)} samples")
    print(f"  Columns: {list(data.columns[:10])}...")
    print()
    
    # =========================================================================
    # STEP 2: Verify Data Quality
    # =========================================================================
    print("STEP 2: Verifying data quality...")
    
    # Check for required columns
    required_cols = ['trade_success', 'R_multiple']
    missing_cols = [col for col in required_cols if col not in data.columns]
    
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        return
    
    # Check data size
    if len(data) < 1000:
        print(f"Warning: Only {len(data)} samples. Recommended: 5000+")
    
    # Check missing values
    missing_pct = (data.isnull().sum() / len(data) * 100)
    high_missing = missing_pct[missing_pct > 20]
    
    if len(high_missing) > 0:
        print(f"Warning: High missing values in: {list(high_missing.index)}")
    
    # Check target distribution
    win_rate = data['trade_success'].mean()
    print(f"✓ Win rate: {win_rate:.1%}")
    print(f"✓ Mean R_multiple: {data['R_multiple'].mean():.2f}R")
    print()
    
    # =========================================================================
    # STEP 3: Select Features
    # =========================================================================
    print("STEP 3: Selecting features...")
    
    # Option 1: Manually specify features
    feature_list = [
        'trend_strength_tf',
        'swing_position',
        'volatility_regime',
        'support_distance',
        'momentum_score',
        'time_of_day',
        'spread_ratio',
        'volume_profile'
    ]
    
    # Option 2: Auto-select from Auto Feature Selection results
    # feature_list = None  # Will read from config
    
    # Verify features exist
    missing_features = [f for f in feature_list if f not in data.columns]
    if missing_features:
        print(f"Error: Missing features: {missing_features}")
        return
    
    print(f"✓ Using {len(feature_list)} features:")
    for i, feat in enumerate(feature_list, 1):
        print(f"  {i}. {feat}")
    print()
    
    # =========================================================================
    # STEP 4: Configure Training Parameters
    # =========================================================================
    print("STEP 4: Configuring training parameters...")
    
    config = {
        'cv_folds': 5,           # 5-fold cross-validation
        'train_ratio': 0.60,     # 60% training
        'calib_ratio': 0.20,     # 20% calibration
        'test_ratio': 0.20,      # 20% test
        'random_state': 42       # For reproducibility
    }
    
    print(f"✓ Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # =========================================================================
    # STEP 5: Initialize Trainer
    # =========================================================================
    print("STEP 5: Initializing trainer...")
    
    trainer = ModelTrainer()
    print("✓ Trainer initialized")
    print()
    
    # =========================================================================
    # STEP 6: Train All Components
    # =========================================================================
    print("STEP 6: Training all 4 components...")
    print("This may take 1-2 minutes...")
    print()
    
    try:
        metrics = trainer.train_all_components(
            data=data,
            target_win='trade_success',
            target_r='R_multiple',
            feature_list=feature_list,
            save_dir='data_processed/models',
            **config
        )
        
        print("✓ Training complete!")
        print()
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # =========================================================================
    # STEP 7: Review Training Metrics
    # =========================================================================
    print("STEP 7: Reviewing training metrics...")
    print()
    
    # Classifier metrics
    print("CLASSIFIER METRICS:")
    print(f"  AUC (train): {metrics['classifier']['auc_train']:.3f}")
    print(f"  AUC (val):   {metrics['classifier']['auc_val']:.3f}")
    print(f"  Brier (val): {metrics['classifier']['brier_score_val']:.3f}")
    
    if metrics['classifier']['auc_val'] < 0.55:
        print("  ⚠️  Warning: AUC < 0.55 (barely better than random)")
    elif metrics['classifier']['auc_val'] > 0.70:
        print("  ✓ Excellent AUC!")
    else:
        print("  ✓ Decent AUC")
    print()
    
    # Calibration metrics
    print("CALIBRATION METRICS:")
    print(f"  Brier before: {metrics['calibration']['brier_before']:.3f}")
    print(f"  Brier after:  {metrics['calibration']['brier_after']:.3f}")
    print(f"  Improvement:  {metrics['calibration']['brier_before'] - metrics['calibration']['brier_after']:.3f}")
    print(f"  ECE after:    {metrics['calibration']['ece_after']:.3f}")
    
    if metrics['calibration']['ece_after'] < 0.05:
        print("  ✓ Excellent calibration!")
    elif metrics['calibration']['ece_after'] < 0.10:
        print("  ✓ Good calibration")
    else:
        print("  ⚠️  Warning: Poor calibration (ECE > 0.10)")
    print()
    
    # Quantile metrics
    print("QUANTILE REGRESSION METRICS:")
    print(f"  MAE P10 (val): {metrics['quantile']['mae_p10_val']:.3f}R")
    print(f"  MAE P50 (val): {metrics['quantile']['mae_p50_val']:.3f}R")
    print(f"  MAE P90 (val): {metrics['quantile']['mae_p90_val']:.3f}R")
    
    avg_mae = np.mean([
        metrics['quantile']['mae_p10_val'],
        metrics['quantile']['mae_p50_val'],
        metrics['quantile']['mae_p90_val']
    ])
    
    if avg_mae < 0.4:
        print("  ✓ Excellent quantile predictions!")
    elif avg_mae < 0.6:
        print("  ✓ Good quantile predictions")
    else:
        print("  ⚠️  Warning: High MAE (> 0.6R)")
    print()
    
    # Conformal metrics
    print("CONFORMAL PREDICTION METRICS:")
    print(f"  Target coverage:  {metrics['conformal']['target_coverage']:.1%}")
    print(f"  Actual coverage:  {metrics['conformal']['actual_coverage_test']:.1%}")
    print(f"  Avg interval:     {metrics['conformal']['avg_interval_width']:.2f}R")
    
    coverage_diff = abs(
        metrics['conformal']['actual_coverage_test'] - 
        metrics['conformal']['target_coverage']
    )
    
    if coverage_diff < 0.05:
        print("  ✓ Excellent coverage!")
    elif coverage_diff < 0.10:
        print("  ✓ Good coverage")
    else:
        print("  ⚠️  Warning: Coverage far from target")
    print()
    
    # Metadata
    print("TRAINING METADATA:")
    print(f"  Training samples:   {metrics['metadata']['n_train']}")
    print(f"  Calibration samples: {metrics['metadata']['n_calib']}")
    print(f"  Test samples:       {metrics['metadata']['n_test']}")
    print(f"  Features:           {metrics['metadata']['n_features']}")
    print(f"  Training time:      {metrics['metadata']['training_time_seconds']:.1f}s")
    print()
    
    # =========================================================================
    # STEP 8: Save Metrics
    # =========================================================================
    print("STEP 8: Saving metrics...")
    
    import json
    metrics_path = 'data_processed/models/training_metrics.json'
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✓ Metrics saved to {metrics_path}")
    print()
    
    # =========================================================================
    # STEP 9: Recommendations
    # =========================================================================
    print("STEP 9: Recommendations...")
    print()
    
    recommendations = []
    
    # Check AUC
    if metrics['classifier']['auc_val'] < 0.55:
        recommendations.append(
            "• Low AUC: Try different features or more data"
        )
    
    # Check calibration
    if metrics['calibration']['ece_after'] > 0.10:
        recommendations.append(
            "• Poor calibration: Increase calibration set size"
        )
    
    # Check coverage
    if coverage_diff > 0.10:
        recommendations.append(
            "• Coverage off target: Adjust conformal coverage parameter"
        )
    
    # Check overfitting
    auc_gap = metrics['classifier']['auc_train'] - metrics['classifier']['auc_val']
    if auc_gap > 0.10:
        recommendations.append(
            "• Possible overfitting: Reduce model complexity or add regularization"
        )
    
    if recommendations:
        print("RECOMMENDATIONS:")
        for rec in recommendations:
            print(rec)
    else:
        print("✓ All metrics look good! Models are ready for production.")
    
    print()
    
    # =========================================================================
    # STEP 10: Next Steps
    # =========================================================================
    print("=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Review metrics above")
    print("2. Run prediction example: python examples/ml_prediction_prediction_example.py")
    print("3. Test in dashboard: Navigate to ML Prediction Engine (Tab 8)")
    print("4. Monitor performance over time")
    print()
    print("Models saved to: data_processed/models/")
    print("  - lgbm_classifier.pkl")
    print("  - isotonic_calibrator.pkl")
    print("  - lgbm_quantile_p10.pkl")
    print("  - lgbm_quantile_p50.pkl")
    print("  - lgbm_quantile_p90.pkl")
    print("  - conformal_meta.json")
    print("  - feature_list.json")
    print("  - training_metrics.json")
    print()

if __name__ == '__main__':
    main()
