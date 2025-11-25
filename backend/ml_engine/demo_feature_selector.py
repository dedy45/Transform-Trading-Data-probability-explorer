"""
Demo script for Feature Selector

Demonstrates how to use the Feature Selector with Auto Feature Selection.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from backend.ml_engine.feature_selector import FeatureSelector
from backend.calculators.auto_feature_selector import run_auto_feature_selection


def create_sample_data():
    """Create sample trading data for demonstration"""
    np.random.seed(42)
    n_samples = 300
    
    data = {
        # Target
        'trade_success': np.random.randint(0, 2, n_samples),
        
        # Good predictive features
        'trend_strength': np.random.rand(n_samples),
        'volatility': np.random.rand(n_samples) * 2,
        'momentum': np.random.randn(n_samples),
        'support_distance': np.random.rand(n_samples) * 100,
        'volume_profile': np.random.rand(n_samples),
        'rsi': np.random.rand(n_samples) * 100,
        'macd': np.random.randn(n_samples),
        'time_of_day': np.random.randint(0, 24, n_samples),
        'spread': np.random.rand(n_samples) * 2,
        'swing_position': np.random.rand(n_samples),
        
        # Trade metadata (will be filtered)
        'Ticket_id': range(n_samples),
        'Symbol': ['GOLD'] * n_samples,
        'OpenPrice': np.random.rand(n_samples) * 2000 + 1800,
        
        # Trade results (will be filtered - data leakage)
        'R_multiple': np.random.randn(n_samples) * 2,
        'gross_profit': np.random.randn(n_samples) * 100,
    }
    
    return pd.DataFrame(data)


def main():
    """Run feature selector demo"""
    print("="*70)
    print("FEATURE SELECTOR DEMO")
    print("="*70)
    
    # Step 1: Create sample data
    print("\n1. Creating sample trading data...")
    df = create_sample_data()
    print(f"   ✓ Created {len(df)} samples with {len(df.columns)} columns")
    
    # Step 2: Run Auto Feature Selection
    print("\n2. Running Auto Feature Selection...")
    results = run_auto_feature_selection(
        df=df,
        target_col='trade_success',
        mode='quick',
        n_features=8
    )
    print(f"   ✓ Auto Feature Selection complete")
    print(f"   ✓ Selected {len(results['selected_features'])} features")
    
    # Step 3: Initialize Feature Selector
    print("\n3. Initializing Feature Selector...")
    selector = FeatureSelector(min_features=5, max_features=8)
    print(f"   ✓ Feature Selector initialized (min={selector.min_features}, max={selector.max_features})")
    
    # Step 4: Load selected features
    print("\n4. Loading selected features (composite_score > 0.6)...")
    features = selector.load_selected_features(
        results=results,
        composite_score_threshold=0.6
    )
    print(f"   ✓ Loaded {len(features)} features:")
    for i, feature in enumerate(features, 1):
        score = selector.feature_scores.get(feature, 0)
        print(f"      {i}. {feature:20s} (score: {score:.3f})")
    
    # Step 5: Validate features
    print("\n5. Validating features in dataset...")
    present, missing = selector.validate_features(df)
    print(f"   ✓ Present: {len(present)} features")
    print(f"   ✓ Missing: {len(missing)} features")
    
    # Step 6: Get feature info
    print("\n6. Getting feature information...")
    info = selector.get_feature_info()
    print(f"   ✓ Number of features: {info['n_features']}")
    print(f"   ✓ Min score: {info['min_score']:.3f}")
    print(f"   ✓ Max score: {info['max_score']:.3f}")
    print(f"   ✓ Avg score: {info['avg_score']:.3f}")
    
    # Step 7: Filter dataset
    print("\n7. Filtering dataset to selected features...")
    filtered_df = selector.filter_dataset(df, handle_missing='ignore')
    print(f"   ✓ Original shape: {df.shape}")
    print(f"   ✓ Filtered shape: {filtered_df.shape}")
    print(f"   ✓ Columns: {list(filtered_df.columns)}")
    
    # Step 8: Save feature list
    print("\n8. Saving feature list...")
    output_path = 'data_processed/models/feature_list_demo.json'
    selector.save_feature_list(output_path)
    print(f"   ✓ Saved to: {output_path}")
    
    # Step 9: Load from saved file
    print("\n9. Loading from saved file...")
    # Use min_features that matches what was saved
    selector2 = FeatureSelector(min_features=1, max_features=10)
    features2 = selector2.load_selected_features(feature_list_path=output_path)
    print(f"   ✓ Loaded {len(features2)} features from file")
    print(f"   ✓ Features match: {features == features2}")
    
    print("\n" + "="*70)
    print("DEMO COMPLETE!")
    print("="*70)
    print("\nThe Feature Selector is ready for use in the ML Prediction Engine.")
    print("It will be used by:")
    print("  - LightGBM Classifier")
    print("  - Isotonic Calibration")
    print("  - Quantile Regression")
    print("  - Conformal Prediction")
    print("  - Prediction Pipeline")


if __name__ == '__main__':
    main()
