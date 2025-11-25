"""
Demo: Model Trainer

This script demonstrates how to use the ModelTrainer to train all 4 ML components.

**Feature: ml-prediction-engine**
**Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5**
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.ml_engine.model_trainer import ModelTrainer, train_models


def create_demo_data(n_samples=5000):
    """
    Create demo trading data for training.
    
    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    
    Returns
    -------
    pd.DataFrame
        Demo trading data
    """
    np.random.seed(42)
    
    # Generate features
    data = pd.DataFrame({
        'trend_strength_tf': np.random.uniform(0, 1, n_samples),
        'swing_position': np.random.uniform(0, 1, n_samples),
        'volatility_regime': np.random.randint(0, 3, n_samples),
        'support_distance': np.random.uniform(-50, 50, n_samples),
        'momentum_score': np.random.uniform(-1, 1, n_samples),
        'time_of_day': np.random.randint(0, 24, n_samples),
        'spread_ratio': np.random.uniform(0.5, 2.0, n_samples),
        'volume_profile': np.random.uniform(0, 1, n_samples)
    })
    
    # Generate target: R_multiple (with some correlation to features)
    data['R_multiple'] = (
        data['trend_strength_tf'] * 2 +
        data['momentum_score'] * 1.5 +
        np.random.randn(n_samples) * 0.5
    )
    
    # Generate win/loss (binary)
    data['trade_success'] = (data['R_multiple'] > 0).astype(int)
    
    print(f"Created demo data with {n_samples} samples")
    print(f"Win rate: {data['trade_success'].mean():.2%}")
    print(f"Mean R_multiple: {data['R_multiple'].mean():.2f}")
    
    return data


def progress_callback(progress_dict):
    """
    Callback function to display training progress.
    
    Parameters
    ----------
    progress_dict : dict
        Progress information
    """
    component = progress_dict['current_component']
    step = progress_dict['current_step']
    progress = progress_dict['progress_percentage']
    status = progress_dict['status']
    
    print(f"[{status.upper()}] {component}: {step} ({progress:.1f}%)")
    
    if status == 'failed':
        print(f"ERROR: {progress_dict['error_message']}")


def demo_basic_training():
    """
    Demo 1: Basic training workflow.
    """
    print("\n" + "="*80)
    print("DEMO 1: Basic Training Workflow")
    print("="*80)
    
    # Create demo data
    data = create_demo_data(n_samples=3000)
    
    # Define features
    feature_columns = [
        'trend_strength_tf',
        'swing_position',
        'volatility_regime',
        'support_distance',
        'momentum_score',
        'time_of_day',
        'spread_ratio',
        'volume_profile'
    ]
    
    # Train models using convenience function
    print("\nStarting training...")
    metrics = train_models(
        data=data,
        feature_columns=feature_columns,
        target_column='R_multiple',
        win_column='trade_success',
        config_path='config/ml_prediction_config.yaml',
        overwrite=True,
        progress_callback=progress_callback
    )
    
    # Display results
    print("\n" + "-"*80)
    print("TRAINING RESULTS")
    print("-"*80)
    
    print("\n1. Classifier Metrics:")
    print(f"   - AUC (train): {metrics['classifier']['auc_train']:.4f}")
    if 'auc_val' in metrics['classifier']:
        print(f"   - AUC (val): {metrics['classifier']['auc_val']:.4f}")
    print(f"   - Brier Score (train): {metrics['classifier']['brier_train']:.4f}")
    
    print("\n2. Calibration Metrics:")
    print(f"   - Brier Score (before): {metrics['calibration']['brier_before']:.4f}")
    print(f"   - Brier Score (after): {metrics['calibration']['brier_after']:.4f}")
    print(f"   - Improvement: {metrics['calibration']['brier_improvement']:.4f}")
    print(f"   - ECE (before): {metrics['calibration']['ece_before']:.4f}")
    print(f"   - ECE (after): {metrics['calibration']['ece_after']:.4f}")
    
    print("\n3. Quantile Regression Metrics:")
    print(f"   - MAE P10 (train): {metrics['quantile']['mae_p10_train']:.4f}")
    print(f"   - MAE P50 (train): {metrics['quantile']['mae_p50_train']:.4f}")
    print(f"   - MAE P90 (train): {metrics['quantile']['mae_p90_train']:.4f}")
    
    print("\n4. Conformal Prediction Metrics:")
    print(f"   - Target Coverage: {metrics['conformal']['coverage']:.2%}")
    print(f"   - Nonconformity Quantile: {metrics['conformal']['nonconformity_quantile']:.4f}")
    
    print("\n5. Dataset Split:")
    print(f"   - Training samples: {metrics['metadata']['n_train']}")
    print(f"   - Calibration samples: {metrics['metadata']['n_calib']}")
    print(f"   - Test samples: {metrics['metadata']['n_test']}")
    print(f"   - Features: {metrics['metadata']['n_features']}")
    
    print("\n✓ Training completed successfully!")


def demo_training_with_custom_config():
    """
    Demo 2: Training with custom configuration.
    """
    print("\n" + "="*80)
    print("DEMO 2: Training with Custom Configuration")
    print("="*80)
    
    # Create demo data
    data = create_demo_data(n_samples=2000)
    
    # Define features (subset)
    feature_columns = [
        'trend_strength_tf',
        'momentum_score',
        'support_distance',
        'volatility_regime',
        'time_of_day'
    ]
    
    # Initialize trainer
    trainer = ModelTrainer(
        config_path='config/ml_prediction_config.yaml',
        progress_callback=progress_callback
    )
    
    print("\nStarting training with custom features...")
    metrics = trainer.train_all_components(
        data=data,
        feature_columns=feature_columns,
        target_column='R_multiple',
        win_column='trade_success',
        overwrite=True
    )
    
    print("\n✓ Training completed with custom configuration!")
    print(f"   - Features used: {len(feature_columns)}")
    print(f"   - Classifier AUC: {metrics['classifier']['auc_train']:.4f}")


def demo_training_status():
    """
    Demo 3: Monitoring training status.
    """
    print("\n" + "="*80)
    print("DEMO 3: Monitoring Training Status")
    print("="*80)
    
    # Create demo data
    data = create_demo_data(n_samples=2000)
    
    feature_columns = [
        'trend_strength_tf',
        'momentum_score',
        'support_distance'
    ]
    
    # Initialize trainer
    trainer = ModelTrainer(config_path='config/ml_prediction_config.yaml')
    
    # Check initial status
    print("\nInitial training status:")
    status = trainer.get_training_status()
    print(f"   - Status: {status['status']}")
    print(f"   - Progress: {status['progress_percentage']:.1f}%")
    
    # Train models
    print("\nStarting training...")
    trainer.train_all_components(
        data=data,
        feature_columns=feature_columns,
        target_column='R_multiple',
        win_column='trade_success',
        overwrite=True
    )
    
    # Check final status
    print("\nFinal training status:")
    status = trainer.get_training_status()
    print(f"   - Status: {status['status']}")
    print(f"   - Progress: {status['progress_percentage']:.1f}%")
    print(f"   - Component Status:")
    for component, comp_status in status['component_status'].items():
        print(f"     • {component}: {comp_status}")
    
    print("\n✓ Training status monitoring completed!")


def demo_model_files():
    """
    Demo 4: Verify saved model files.
    """
    print("\n" + "="*80)
    print("DEMO 4: Verify Saved Model Files")
    print("="*80)
    
    model_dir = Path('data_processed/models')
    
    # List of expected model files
    expected_files = [
        'lgbm_classifier.pkl',
        'isotonic_calibrator.pkl',
        'lgbm_quantile_p10.pkl',
        'lgbm_quantile_p50.pkl',
        'lgbm_quantile_p90.pkl',
        'lgbm_quantile_meta.pkl',
        'conformal_meta.json',
        'feature_list.json',
        'training_metrics.json'
    ]
    
    print(f"\nChecking model directory: {model_dir}")
    print("\nModel files:")
    
    for filename in expected_files:
        filepath = model_dir / filename
        if filepath.exists():
            size_kb = filepath.stat().st_size / 1024
            print(f"   ✓ {filename} ({size_kb:.1f} KB)")
        else:
            print(f"   ✗ {filename} (not found)")
    
    # Load and display training metrics
    metrics_file = model_dir / 'training_metrics.json'
    if metrics_file.exists():
        import json
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        print("\nTraining timestamp:")
        print(f"   {metrics['metadata']['training_timestamp']}")
    
    print("\n✓ Model file verification completed!")


def main():
    """Run all demos."""
    print("\n" + "="*80)
    print("MODEL TRAINER DEMO")
    print("="*80)
    print("\nThis demo shows how to use the ModelTrainer to train all ML components.")
    
    try:
        # Run demos
        demo_basic_training()
        demo_training_with_custom_config()
        demo_training_status()
        demo_model_files()
        
        print("\n" + "="*80)
        print("ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
