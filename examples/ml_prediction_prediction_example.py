"""
ML Prediction Engine - Prediction Example

This example demonstrates how to make predictions using trained models:
1. Single prediction
2. Batch prediction
3. Filtering and ranking
4. Exporting results

Author: ML Prediction Engine Team
Date: 2024-11-24
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.ml_engine.pipeline_prediction import PredictionPipeline
from backend.ml_engine.export_utils import ExportUtils

def print_prediction_result(result, title="Prediction Result"):
    """Pretty print a prediction result."""
    print(f"\n{title}")
    print("=" * 60)
    print(f"Win Probability:  {result['prob_win_calibrated']:.1%}")
    print(f"Expected R:       {result['R_P50_raw']:.2f}R")
    print(f"P10-P90 Interval: [{result['R_P10_conf']:.2f}R, {result['R_P90_conf']:.2f}R]")
    print(f"Skewness:         {result.get('skewness', 0):.2f}")
    print(f"Quality:          {result['quality_label']}")
    print(f"Recommendation:   {result['recommendation']}")
    print(f"Execution Time:   {result['execution_time_ms']:.1f}ms")
    print("=" * 60)

def main():
    """
    Complete prediction workflow example.
    """
    
    print("=" * 80)
    print("ML PREDICTION ENGINE - PREDICTION EXAMPLE")
    print("=" * 80)
    print()
    
    # =========================================================================
    # STEP 1: Load Pipeline
    # =========================================================================
    print("STEP 1: Loading prediction pipeline...")
    
    pipeline = PredictionPipeline()
    
    try:
        success = pipeline.load_models('data_processed/models')
        if not success:
            print("Error: Failed to load models")
            print("Please train models first: python examples/ml_prediction_training_example.py")
            return
        
        print("✓ Pipeline loaded successfully")
        print(f"  Features: {len(pipeline.feature_names)}")
        print(f"  Feature names: {pipeline.feature_names}")
        print()
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train models first: python examples/ml_prediction_training_example.py")
        return
    
    # =========================================================================
    # STEP 2: Single Prediction - Example 1 (High Quality Setup)
    # =========================================================================
    print("STEP 2: Single prediction - High quality setup...")
    
    # Example of a strong setup
    features_high_quality = {
        'trend_strength_tf': 0.85,    # Strong trend
        'swing_position': 0.75,       # Good position
        'volatility_regime': 1,       # Normal volatility
        'support_distance': 12.5,     # Near support
        'momentum_score': 0.65,       # Strong momentum
        'time_of_day': 10,            # Good trading hour
        'spread_ratio': 0.7,          # Low spread
        'volume_profile': 1.5         # High volume
    }
    
    result_high = pipeline.predict_for_sample(features_high_quality)
    print_prediction_result(result_high, "HIGH QUALITY SETUP")
    
    # =========================================================================
    # STEP 3: Single Prediction - Example 2 (Low Quality Setup)
    # =========================================================================
    print("\nSTEP 3: Single prediction - Low quality setup...")
    
    # Example of a weak setup
    features_low_quality = {
        'trend_strength_tf': 0.35,    # Weak trend
        'swing_position': 0.25,       # Poor position
        'volatility_regime': 2,       # High volatility
        'support_distance': 45.0,     # Far from support
        'momentum_score': -0.15,      # Negative momentum
        'time_of_day': 3,             # Poor trading hour
        'spread_ratio': 1.8,          # High spread
        'volume_profile': 0.4         # Low volume
    }
    
    result_low = pipeline.predict_for_sample(features_low_quality)
    print_prediction_result(result_low, "LOW QUALITY SETUP")
    
    # =========================================================================
    # STEP 4: Batch Prediction
    # =========================================================================
    print("\nSTEP 4: Batch prediction...")
    
    # Create sample batch data
    np.random.seed(42)
    n_samples = 100
    
    batch_data = pd.DataFrame({
        'setup_id': range(1, n_samples + 1),
        'trend_strength_tf': np.random.uniform(0.2, 0.9, n_samples),
        'swing_position': np.random.uniform(0.1, 0.8, n_samples),
        'volatility_regime': np.random.choice([0, 1, 2], n_samples),
        'support_distance': np.random.uniform(5, 50, n_samples),
        'momentum_score': np.random.uniform(-0.5, 0.8, n_samples),
        'time_of_day': np.random.randint(0, 24, n_samples),
        'spread_ratio': np.random.uniform(0.5, 2.0, n_samples),
        'volume_profile': np.random.uniform(0.3, 2.0, n_samples)
    })
    
    print(f"Predicting for {len(batch_data)} setups...")
    
    start_time = time.time()
    results_df = pipeline.predict_for_batch(batch_data)
    elapsed_time = time.time() - start_time
    
    print(f"✓ Batch prediction complete!")
    print(f"  Total time: {elapsed_time:.2f}s")
    print(f"  Time per sample: {elapsed_time / len(batch_data) * 1000:.1f}ms")
    print()
    
    # =========================================================================
    # STEP 5: Analyze Batch Results
    # =========================================================================
    print("STEP 5: Analyzing batch results...")
    print()
    
    # Quality distribution
    print("QUALITY DISTRIBUTION:")
    quality_counts = results_df['quality_label'].value_counts()
    for quality in ['A+', 'A', 'B', 'C']:
        count = quality_counts.get(quality, 0)
        pct = count / len(results_df) * 100
        print(f"  {quality}: {count:3d} ({pct:5.1f}%)")
    print()
    
    # Recommendation distribution
    print("RECOMMENDATION DISTRIBUTION:")
    rec_counts = results_df['recommendation'].value_counts()
    for rec in ['TRADE', 'SKIP']:
        count = rec_counts.get(rec, 0)
        pct = count / len(results_df) * 100
        print(f"  {rec}: {count:3d} ({pct:5.1f}%)")
    print()
    
    # Statistics
    print("PREDICTION STATISTICS:")
    print(f"  Mean Win Prob:  {results_df['prob_win_calibrated'].mean():.1%}")
    print(f"  Mean Expected R: {results_df['R_P50_raw'].mean():.2f}R")
    print(f"  Mean Interval:   {(results_df['R_P90_conf'] - results_df['R_P10_conf']).mean():.2f}R")
    print()
    
    # =========================================================================
    # STEP 6: Filter High-Quality Setups
    # =========================================================================
    print("STEP 6: Filtering high-quality setups...")
    
    # Filter A+ and A setups
    high_quality = results_df[results_df['quality_label'].isin(['A+', 'A'])]
    
    print(f"✓ Found {len(high_quality)} high-quality setups (A+ or A)")
    print()
    
    if len(high_quality) > 0:
        print("TOP 5 HIGH-QUALITY SETUPS:")
        print("-" * 80)
        
        # Sort by expected R
        top_5 = high_quality.nlargest(5, 'R_P50_raw')
        
        for idx, row in top_5.iterrows():
            print(f"Setup #{row['setup_id']}:")
            print(f"  Win Prob: {row['prob_win_calibrated']:.1%}")
            print(f"  Expected R: {row['R_P50_raw']:.2f}R")
            print(f"  Interval: [{row['R_P10_conf']:.2f}R, {row['R_P90_conf']:.2f}R]")
            print(f"  Quality: {row['quality_label']}")
            print()
    
    # =========================================================================
    # STEP 7: Advanced Filtering
    # =========================================================================
    print("STEP 7: Advanced filtering...")
    print()
    
    # Filter 1: High probability (> 60%)
    high_prob = results_df[results_df['prob_win_calibrated'] > 0.60]
    print(f"Setups with >60% win probability: {len(high_prob)}")
    
    # Filter 2: High expected R (> 1.5R)
    high_r = results_df[results_df['R_P50_raw'] > 1.5]
    print(f"Setups with >1.5R expected return: {len(high_r)}")
    
    # Filter 3: Positive skewness (upside potential)
    if 'skewness' in results_df.columns:
        positive_skew = results_df[results_df['skewness'] > 1.0]
        print(f"Setups with positive skewness: {len(positive_skew)}")
    
    # Filter 4: Combined criteria
    combined = results_df[
        (results_df['prob_win_calibrated'] > 0.60) &
        (results_df['R_P50_raw'] > 1.0) &
        (results_df['quality_label'].isin(['A+', 'A']))
    ]
    print(f"Setups meeting all criteria: {len(combined)}")
    print()
    
    # =========================================================================
    # STEP 8: Ranking Strategies
    # =========================================================================
    print("STEP 8: Ranking strategies...")
    print()
    
    # Strategy 1: Rank by expected R
    ranked_by_r = results_df.nlargest(10, 'R_P50_raw')
    print("TOP 10 BY EXPECTED R:")
    print(ranked_by_r[['setup_id', 'prob_win_calibrated', 'R_P50_raw', 'quality_label']].to_string(index=False))
    print()
    
    # Strategy 2: Rank by win probability
    ranked_by_prob = results_df.nlargest(10, 'prob_win_calibrated')
    print("TOP 10 BY WIN PROBABILITY:")
    print(ranked_by_prob[['setup_id', 'prob_win_calibrated', 'R_P50_raw', 'quality_label']].to_string(index=False))
    print()
    
    # Strategy 3: Rank by risk-adjusted return (Sharpe-like)
    results_df['interval_width'] = results_df['R_P90_conf'] - results_df['R_P10_conf']
    results_df['sharpe'] = results_df['R_P50_raw'] / results_df['interval_width']
    
    ranked_by_sharpe = results_df.nlargest(10, 'sharpe')
    print("TOP 10 BY RISK-ADJUSTED RETURN:")
    print(ranked_by_sharpe[['setup_id', 'prob_win_calibrated', 'R_P50_raw', 'sharpe', 'quality_label']].to_string(index=False))
    print()
    
    # =========================================================================
    # STEP 9: Export Results
    # =========================================================================
    print("STEP 9: Exporting results...")
    
    exporter = ExportUtils()
    
    # Export all predictions
    output_dir = Path('exports')
    output_dir.mkdir(exist_ok=True)
    
    csv_path = output_dir / 'ml_predictions_example.csv'
    exporter.export_predictions_to_csv(
        predictions_df=results_df,
        filepath=str(csv_path)
    )
    print(f"✓ All predictions exported to: {csv_path}")
    
    # Export high-quality only
    if len(high_quality) > 0:
        hq_path = output_dir / 'ml_predictions_high_quality.csv'
        exporter.export_predictions_to_csv(
            predictions_df=high_quality,
            filepath=str(hq_path)
        )
        print(f"✓ High-quality setups exported to: {hq_path}")
    
    # Export to JSON
    json_path = output_dir / 'ml_predictions_example.json'
    exporter.export_predictions_to_json(
        predictions_df=results_df,
        filepath=str(json_path)
    )
    print(f"✓ JSON export saved to: {json_path}")
    print()
    
    # =========================================================================
    # STEP 10: Performance Analysis
    # =========================================================================
    print("STEP 10: Performance analysis...")
    print()
    
    # Execution time analysis
    if 'execution_time_ms' in results_df.columns:
        print("EXECUTION TIME STATISTICS:")
        print(f"  Mean:   {results_df['execution_time_ms'].mean():.1f}ms")
        print(f"  Median: {results_df['execution_time_ms'].median():.1f}ms")
        print(f"  Min:    {results_df['execution_time_ms'].min():.1f}ms")
        print(f"  Max:    {results_df['execution_time_ms'].max():.1f}ms")
        
        if results_df['execution_time_ms'].mean() > 100:
            print("  ⚠️  Warning: Average prediction time > 100ms")
        else:
            print("  ✓ Performance within target")
        print()
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 80)
    print("PREDICTION COMPLETE!")
    print("=" * 80)
    print()
    print("Summary:")
    print(f"  Total setups predicted: {len(results_df)}")
    print(f"  High-quality setups (A+/A): {len(high_quality)}")
    print(f"  TRADE recommendations: {len(results_df[results_df['recommendation'] == 'TRADE'])}")
    print(f"  Mean win probability: {results_df['prob_win_calibrated'].mean():.1%}")
    print(f"  Mean expected R: {results_df['R_P50_raw'].mean():.2f}R")
    print()
    print("Next steps:")
    print("1. Review exported CSV files in exports/ folder")
    print("2. Run interpretation example: python examples/ml_prediction_interpretation_example.py")
    print("3. Test in dashboard: Navigate to ML Prediction Engine (Tab 8)")
    print("4. Integrate with your trading system")
    print()

if __name__ == '__main__':
    main()
