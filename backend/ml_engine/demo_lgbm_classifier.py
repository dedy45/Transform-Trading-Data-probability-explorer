"""
Demo script for LightGBM Classifier

This script demonstrates how to use the LGBMClassifierWrapper for
binary win/loss prediction in trading setups.

**Feature: ml-prediction-engine**
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from backend.ml_engine import LGBMClassifierWrapper, train_classifier


def create_sample_data(n_samples=500):
    """
    Create sample trading data for demonstration.
    
    Returns
    -------
    tuple
        (X, y) - Features and binary target
    """
    np.random.seed(42)
    
    # Create features
    X = pd.DataFrame({
        'trend_strength': np.random.uniform(0, 1, n_samples),
        'volatility': np.random.uniform(0, 1, n_samples),
        'momentum': np.random.uniform(-1, 1, n_samples),
        'support_distance': np.random.uniform(0, 100, n_samples),
        'volume_profile': np.random.uniform(0, 1, n_samples),
        'time_of_day': np.random.randint(0, 24, n_samples),
        'spread_ratio': np.random.uniform(0.5, 2.0, n_samples)
    })
    
    # Create target with some correlation to features
    # Win probability increases with trend_strength and momentum
    win_score = (
        X['trend_strength'] * 2 +
        X['momentum'] * 1.5 +
        X['volatility'] * 0.5 +
        np.random.randn(n_samples) * 0.5
    )
    
    y = pd.Series((win_score > 0.5).astype(int), name='trade_success')
    
    return X, y


def demo_basic_training():
    """Demonstrate basic classifier training."""
    print("=" * 60)
    print("Demo 1: Basic Training")
    print("=" * 60)
    
    # Create sample data
    X, y = create_sample_data(n_samples=500)
    
    print(f"\nDataset shape: {X.shape}")
    print(f"Win rate: {y.mean():.2%}")
    print(f"\nFeatures: {list(X.columns)}")
    
    # Split data
    split_idx = int(len(X) * 0.7)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Train classifier
    print("\nTraining classifier...")
    classifier = LGBMClassifierWrapper(
        n_estimators=50,
        learning_rate=0.05,
        max_depth=5
    )
    
    metrics = classifier.fit(
        X_train, y_train,
        X_val, y_val,
        use_cv=True,
        n_splits=5
    )
    
    # Display metrics
    print("\nTraining Metrics:")
    print(f"  AUC (train): {metrics['auc_train']:.4f}")
    print(f"  AUC (val):   {metrics['auc_val']:.4f}")
    print(f"  AUC (CV):    {metrics['auc_cv_mean']:.4f} ± {metrics['auc_cv_std']:.4f}")
    print(f"  Brier (train): {metrics['brier_train']:.4f}")
    print(f"  Brier (val):   {metrics['brier_val']:.4f}")
    
    return classifier, X_val, y_val


def demo_prediction(classifier, X_test):
    """Demonstrate making predictions."""
    print("\n" + "=" * 60)
    print("Demo 2: Making Predictions")
    print("=" * 60)
    
    # Predict probabilities
    probs = classifier.predict_proba(X_test)
    
    print(f"\nPredicted probabilities for {len(X_test)} samples:")
    print(f"  Min:  {probs.min():.4f}")
    print(f"  Mean: {probs.mean():.4f}")
    print(f"  Max:  {probs.max():.4f}")
    
    # Predict binary labels
    predictions = classifier.predict(X_test, threshold=0.5)
    
    print(f"\nBinary predictions (threshold=0.5):")
    print(f"  Predicted wins: {predictions.sum()} ({predictions.mean():.2%})")
    
    # Show sample predictions
    print("\nSample predictions:")
    sample_df = pd.DataFrame({
        'prob_win': probs[:5],
        'prediction': predictions[:5]
    })
    print(sample_df.to_string(index=False))


def demo_feature_importance(classifier):
    """Demonstrate feature importance analysis."""
    print("\n" + "=" * 60)
    print("Demo 3: Feature Importance")
    print("=" * 60)
    
    # Get feature importance
    importance = classifier.get_feature_importance()
    
    print("\nFeature Importance (all features):")
    print(importance.to_string(index=False))
    
    # Get top 5 features
    top_5 = classifier.get_feature_importance(top_n=5)
    
    print("\nTop 5 Most Important Features:")
    print(top_5.to_string(index=False))


def demo_save_load():
    """Demonstrate saving and loading models."""
    print("\n" + "=" * 60)
    print("Demo 4: Save and Load Model")
    print("=" * 60)
    
    # Create and train a classifier
    X, y = create_sample_data(n_samples=300)
    
    classifier = LGBMClassifierWrapper(n_estimators=20)
    classifier.fit(X, y, use_cv=False)
    
    # Get predictions before save
    probs_before = classifier.predict_proba(X[:10])
    
    # Save model
    save_path = Path("data_processed/models/demo_classifier.pkl")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    classifier.save(save_path)
    print(f"\nModel saved to: {save_path}")
    
    # Load model
    loaded_classifier = LGBMClassifierWrapper()
    loaded_classifier.load(save_path)
    print(f"Model loaded from: {save_path}")
    
    # Get predictions after load
    probs_after = loaded_classifier.predict_proba(X[:10])
    
    # Verify predictions match
    match = np.allclose(probs_before, probs_after)
    print(f"\nPredictions match after load: {match}")
    
    if match:
        print("✓ Save/load successful - predictions are identical")
    else:
        print("✗ Warning: predictions differ after load")
    
    # Show model info
    info = loaded_classifier.get_model_info()
    print(f"\nModel Info:")
    print(f"  Fitted: {info['is_fitted']}")
    print(f"  Features: {info['n_features']}")
    print(f"  Hyperparameters: {info['hyperparameters']}")


def demo_convenience_function():
    """Demonstrate using the convenience function."""
    print("\n" + "=" * 60)
    print("Demo 5: Convenience Function")
    print("=" * 60)
    
    # Create data
    X, y = create_sample_data(n_samples=300)
    
    split_idx = int(len(X) * 0.7)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Train using convenience function
    print("\nTraining with convenience function...")
    
    save_path = Path("data_processed/models/demo_classifier_convenience.pkl")
    
    classifier, metrics = train_classifier(
        X_train, y_train,
        X_val, y_val,
        n_estimators=30,
        learning_rate=0.05,
        use_cv=True,
        n_splits=3,
        save_path=save_path
    )
    
    print(f"\nTraining complete!")
    print(f"  AUC (train): {metrics['auc_train']:.4f}")
    print(f"  AUC (val):   {metrics['auc_val']:.4f}")
    print(f"  Model saved to: {save_path}")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("LightGBM Classifier Demo")
    print("=" * 60)
    
    # Demo 1: Basic training
    classifier, X_val, y_val = demo_basic_training()
    
    # Demo 2: Making predictions
    demo_prediction(classifier, X_val)
    
    # Demo 3: Feature importance
    demo_feature_importance(classifier)
    
    # Demo 4: Save and load
    demo_save_load()
    
    # Demo 5: Convenience function
    demo_convenience_function()
    
    print("\n" + "=" * 60)
    print("All demos completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
