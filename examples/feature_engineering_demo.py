"""
Feature Engineering Demo

This script demonstrates how to use the probability feature engineering module
to create 18 probability features from trading data.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.utils.feature_engineer import (
    build_probability_features,
    export_to_csv,
    get_probability_feature_summary,
    PROBABILITY_FEATURES
)


def create_sample_data(n_rows=1000):
    """Create sample trading data for demonstration"""
    np.random.seed(42)
    
    # Create sample data with required columns
    data = {
        # Target variables
        'y_win': np.random.choice([0, 1], n_rows, p=[0.4, 0.6]),
        'y_hit_1R': np.random.choice([0, 1], n_rows, p=[0.5, 0.5]),
        'y_hit_2R': np.random.choice([0, 1], n_rows, p=[0.7, 0.3]),
        'R_multiple': np.random.normal(0.5, 2, n_rows),
        
        # Session features
        'session': np.random.choice([0, 1, 2, 3], n_rows),  # ASIA, EUROPE, US, OVERLAP
        
        # Trend features
        'trend_strength_tf': np.random.uniform(0, 1, n_rows),
        'trend_tf_dir': np.random.choice([-1, 0, 1], n_rows),
        'Type': np.random.choice(['BUY', 'SELL'], n_rows),
        'trend_regime': np.random.choice([0, 1], n_rows),  # 0=ranging, 1=trending
        
        # Volatility features
        'volatility_regime': np.random.choice([0, 1, 2], n_rows),  # low, medium, high
        
        # Support/Resistance features
        'dist_to_day_high_pips': np.random.uniform(0, 100, n_rows),
        'dist_to_day_low_pips': np.random.uniform(0, 100, n_rows),
        
        # Structure features
        'ap_entropy_m1_2h': np.random.uniform(0, 2, n_rows),
        'hurst_m5_2d': np.random.uniform(0, 1, n_rows),
        
        # Behavioral features
        'streak_loss': np.random.choice([0, 1, 2, 3, 4, 5], n_rows, p=[0.4, 0.25, 0.15, 0.1, 0.05, 0.05]),
        'current_drawdown_from_equity_high': np.random.uniform(-20, 0, n_rows),
    }
    
    return pd.DataFrame(data)


def main():
    print("=" * 80)
    print("PROBABILITY FEATURE ENGINEERING DEMO")
    print("=" * 80)
    print()
    
    # Step 1: Create sample data
    print("Step 1: Creating sample trading data...")
    df = create_sample_data(n_rows=1000)
    print(f"Created dataset with {len(df)} rows and {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    print()
    
    # Step 2: Build probability features
    print("Step 2: Building 18 probability features...")
    df_with_features = build_probability_features(
        df,
        n_bins=10,  # Number of bins for continuous features
        min_samples=5  # Minimum samples per bin for reliable estimates
    )
    print(f"Added {len(PROBABILITY_FEATURES)} probability features")
    print()
    
    # Step 3: Display feature summary
    print("Step 3: Probability Feature Summary")
    print("-" * 80)
    summary = get_probability_feature_summary(df_with_features)
    
    for feature, stats in summary.items():
        print(f"\n{feature}:")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Std:  {stats['std']:.4f}")
        print(f"  Min:  {stats['min']:.4f}")
        print(f"  Max:  {stats['max']:.4f}")
        print(f"  Null: {stats['null_count']}")
    
    print()
    print("-" * 80)
    
    # Step 4: Show sample rows
    print("\nStep 4: Sample rows with probability features")
    print("-" * 80)
    
    # Select a few probability features to display
    display_features = [
        'y_win',
        'prob_global_win',
        'prob_session_win',
        'prob_trend_strength_win',
        'prob_vol_regime_win',
        'prob_trend_vol_cross_win'
    ]
    
    print(df_with_features[display_features].head(10).to_string())
    print()
    
    # Step 5: Export to CSV
    print("Step 5: Exporting dataset with probability features...")
    output_path = "output_with_probability_features.csv"
    export_to_csv(df_with_features, output_path)
    print(f"Dataset exported to: {output_path}")
    print()
    
    # Step 6: Demonstrate feature usage
    print("Step 6: Example usage - Filter high probability trades")
    print("-" * 80)
    
    # Filter trades with high global win probability
    high_prob_threshold = 0.6
    high_prob_trades = df_with_features[
        df_with_features['prob_global_win'] > high_prob_threshold
    ]
    
    print(f"Total trades: {len(df_with_features)}")
    print(f"High probability trades (prob > {high_prob_threshold}): {len(high_prob_trades)}")
    print(f"Win rate of high prob trades: {high_prob_trades['y_win'].mean():.2%}")
    print(f"Win rate of all trades: {df_with_features['y_win'].mean():.2%}")
    print()
    
    # Filter trades with favorable conditions (multiple features)
    favorable_trades = df_with_features[
        (df_with_features['prob_session_win'] > 0.55) &
        (df_with_features['prob_trend_strength_win'] > 0.55) &
        (df_with_features['prob_vol_regime_win'] > 0.55)
    ]
    
    print(f"Favorable condition trades: {len(favorable_trades)}")
    if len(favorable_trades) > 0:
        print(f"Win rate of favorable trades: {favorable_trades['y_win'].mean():.2%}")
        print(f"Average R-multiple: {favorable_trades['R_multiple'].mean():.2f}")
    print()
    
    print("=" * 80)
    print("DEMO COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print()
    print("Key Takeaways:")
    print("1. 18 probability features were successfully created")
    print("2. Features represent conditional probabilities for various market conditions")
    print("3. Features can be used for:")
    print("   - Trade filtering (select high probability setups)")
    print("   - Machine learning (as input features)")
    print("   - Expert Advisors (as decision signals)")
    print("   - Risk management (adjust position size based on probability)")
    print()


if __name__ == '__main__':
    main()
