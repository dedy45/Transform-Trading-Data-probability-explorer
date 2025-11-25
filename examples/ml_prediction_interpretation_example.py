"""
ML Prediction Engine - Interpretation Example

This example demonstrates how to interpret ML predictions:
1. Understanding calibration (reliability diagrams)
2. Analyzing feature importance
3. Interpreting quantile predictions
4. Understanding conformal intervals
5. Setup quality categorization

Author: ML Prediction Engine Team
Date: 2024-11-24
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.ml_engine.pipeline_prediction import PredictionPipeline
from backend.ml_engine.lgbm_classifier import LGBMClassifierWrapper
from backend.ml_engine.calibration_isotonic import IsotonicCalibrator

def main():
    """
    Complete interpretation workflow example.
    """
    
    print("=" * 80)
    print("ML PREDICTION ENGINE - INTERPRETATION EXAMPLE")
    print("=" * 80)
    print()
    
    # =========================================================================
    # STEP 1: Load Models and Metrics
    # =========================================================================
    print("STEP 1: Loading models and training metrics...")
    
    pipeline = PredictionPipeline()
    
    try:
        pipeline.load_models('data_processed/models')
        print("✓ Models loaded")
    except FileNotFoundError:
        print("Error: Models not found. Please train first.")
        return
    
    # Load training metrics
    metrics_path = 'data_processed/models/training_metrics.json'
    if Path(metrics_path).exists():
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        print("✓ Training metrics loaded")
    else:
        print("Warning: Training metrics not found")
        metrics = None
    
    print()
    
    # =========================================================================
    # STEP 2: Understanding Calibration
    # =========================================================================
    print("STEP 2: Understanding calibration...")
    print()
    
    if metrics:
        print("CALIBRATION QUALITY:")
        print("-" * 60)
        
        brier_before = metrics['calibration']['brier_before']
        brier_after = metrics['calibration']['brier_after']
        ece_after = metrics['calibration']['ece_after']
        
        print(f"Brier Score Before Calibration: {brier_before:.4f}")
        print(f"Brier Score After Calibration:  {brier_after:.4f}")
        print(f"Improvement:                    {brier_before - brier_after:.4f}")
        print()
        
        print("What does this mean?")
        print("• Brier score measures probability accuracy (0 = perfect, 1 = worst)")
        print(f"• Before calibration: {brier_before:.4f}")
        print(f"• After calibration:  {brier_after:.4f}")
        
        if brier_after < 0.20:
            print("• ✓ Excellent calibration! Probabilities are reliable.")
        elif brier_after < 0.25:
            print("• ✓ Good calibration. Probabilities are reasonably reliable.")
        else:
            print("• ⚠️  Poor calibration. Probabilities may not match reality.")
        
        print()
        print(f"Expected Calibration Error (ECE): {ece_after:.4f}")
        print("What does this mean?")
        print("• ECE measures average calibration error across probability bins")
        
        if ece_after < 0.05:
            print("• ✓ Excellent! Predictions are well-calibrated.")
        elif ece_after < 0.10:
            print("• ✓ Good calibration.")
        else:
            print("• ⚠️  Poor calibration. Consider retraining with more data.")
        
        print()
        print("INTERPRETATION GUIDE:")
        print("When the model predicts 70% win probability:")
        if ece_after < 0.05:
            print("• You can expect ~70% of such setups to actually win")
            print("• The calibration is trustworthy")
        else:
            print("• The actual win rate may differ from 70%")
            print("• Use with caution and validate on your data")
        
        print()
    
    # =========================================================================
    # STEP 3: Feature Importance Analysis
    # =========================================================================
    print("STEP 3: Analyzing feature importance...")
    print()
    
    # Load classifier to get feature importance
    classifier = LGBMClassifierWrapper()
    try:
        classifier.load('data_processed/models/lgbm_classifier.pkl')
        importance_df = classifier.get_feature_importance()
        
        print("FEATURE IMPORTANCE (Top 10):")
        print("-" * 60)
        print(importance_df.head(10).to_string(index=False))
        print()
        
        print("INTERPRETATION:")
        top_feature = importance_df.iloc[0]
        print(f"• Most important feature: {top_feature['feature']}")
        print(f"  - This feature has the highest impact on predictions")
        print(f"  - Importance score: {top_feature['importance']:.1f}")
        print()
        
        # Analyze feature groups
        print("FEATURE CATEGORIES:")
        
        # Categorize features (example)
        trend_features = [f for f in importance_df['feature'] if 'trend' in f.lower()]
        momentum_features = [f for f in importance_df['feature'] if 'momentum' in f.lower()]
        volatility_features = [f for f in importance_df['feature'] if 'volatility' in f.lower()]
        
        if trend_features:
            trend_importance = importance_df[importance_df['feature'].isin(trend_features)]['importance'].sum()
            print(f"• Trend features: {len(trend_features)} features, {trend_importance:.1f} total importance")
        
        if momentum_features:
            momentum_importance = importance_df[importance_df['feature'].isin(momentum_features)]['importance'].sum()
            print(f"• Momentum features: {len(momentum_features)} features, {momentum_importance:.1f} total importance")
        
        if volatility_features:
            vol_importance = importance_df[importance_df['feature'].isin(volatility_features)]['importance'].sum()
            print(f"• Volatility features: {len(volatility_features)} features, {vol_importance:.1f} total importance")
        
        print()
        
    except Exception as e:
        print(f"Could not load feature importance: {e}")
        print()
    
    # =========================================================================
    # STEP 4: Understanding Quantile Predictions
    # =========================================================================
    print("STEP 4: Understanding quantile predictions...")
    print()
    
    # Example predictions
    example_predictions = [
        {
            'name': 'Conservative Setup',
            'P10': -0.2, 'P50': 0.8, 'P90': 1.5,
            'P10_conf': -0.4, 'P90_conf': 1.7
        },
        {
            'name': 'Aggressive Setup',
            'P10': -0.8, 'P50': 1.5, 'P90': 4.2,
            'P10_conf': -1.0, 'P90_conf': 4.5
        },
        {
            'name': 'Asymmetric Setup',
            'P10': -0.3, 'P50': 1.0, 'P90': 3.5,
            'P10_conf': -0.5, 'P90_conf': 3.8
        }
    ]
    
    for pred in example_predictions:
        print(f"{pred['name'].upper()}:")
        print("-" * 60)
        print(f"P10 (10th percentile): {pred['P10']:.2f}R")
        print(f"P50 (median):          {pred['P50']:.2f}R")
        print(f"P90 (90th percentile): {pred['P90']:.2f}R")
        print()
        
        print("What does this mean?")
        print(f"• 10% chance outcome is worse than {pred['P10']:.2f}R")
        print(f"• 50% chance outcome is around {pred['P50']:.2f}R (expected)")
        print(f"• 10% chance outcome is better than {pred['P90']:.2f}R")
        print(f"• 80% of outcomes fall between {pred['P10']:.2f}R and {pred['P90']:.2f}R")
        print()
        
        # Calculate skewness
        skewness = (pred['P90'] - pred['P50']) / (pred['P50'] - pred['P10'])
        print(f"Skewness: {skewness:.2f}")
        
        if skewness > 1.5:
            print("• ✓ Highly positive skew - significant upside potential!")
            print("• Best-case scenario much better than worst-case is bad")
        elif skewness > 1.0:
            print("• ✓ Positive skew - some upside potential")
        elif skewness < 0.5:
            print("• ⚠️  Negative skew - more downside risk than upside")
        else:
            print("• Symmetric distribution - balanced risk/reward")
        
        print()
        
        # Conformal intervals
        print("CONFORMAL PREDICTION INTERVALS:")
        print(f"Adjusted P10: {pred['P10_conf']:.2f}R")
        print(f"Adjusted P90: {pred['P90_conf']:.2f}R")
        print(f"Interval width: {pred['P90_conf'] - pred['P10_conf']:.2f}R")
        print()
        
        print("What does this mean?")
        print("• Conformal prediction adjusts intervals for coverage guarantee")
        print("• With 90% coverage, 90% of actual outcomes fall in this interval")
        print(f"• There's a 90% chance outcome is between {pred['P10_conf']:.2f}R and {pred['P90_conf']:.2f}R")
        print()
        print()
    
    # =========================================================================
    # STEP 5: Setup Quality Categorization
    # =========================================================================
    print("STEP 5: Understanding setup quality categories...")
    print()
    
    # Example setups
    example_setups = [
        {'prob': 0.70, 'R_P50': 1.8, 'expected_quality': 'A+'},
        {'prob': 0.60, 'R_P50': 1.2, 'expected_quality': 'A'},
        {'prob': 0.50, 'R_P50': 0.8, 'expected_quality': 'B'},
        {'prob': 0.40, 'R_P50': 0.3, 'expected_quality': 'C'},
    ]
    
    print("QUALITY CATEGORY EXAMPLES:")
    print("-" * 60)
    
    for setup in example_setups:
        quality, recommendation = pipeline.categorize_setup(
            prob_win=setup['prob'],
            R_P50=setup['R_P50']
        )
        
        print(f"Setup: {setup['prob']:.0%} win prob, {setup['R_P50']:.1f}R expected")
        print(f"  Quality: {quality}")
        print(f"  Recommendation: {recommendation}")
        print(f"  Expected: {setup['expected_quality']}")
        
        if quality == setup['expected_quality']:
            print("  ✓ Matches expected category")
        else:
            print(f"  ⚠️  Got {quality}, expected {setup['expected_quality']}")
        
        print()
    
    print("QUALITY CATEGORY GUIDE:")
    print("-" * 60)
    print("A+ (Excellent):")
    print("  • Win probability > 65%")
    print("  • Expected R > 1.5R")
    print("  • Action: STRONG TRADE")
    print("  • These are your best setups")
    print()
    
    print("A (Good):")
    print("  • Win probability > 55%")
    print("  • Expected R > 1.0R")
    print("  • Action: TRADE")
    print("  • Solid edge, worth taking")
    print()
    
    print("B (Fair):")
    print("  • Win probability > 45%")
    print("  • Expected R > 0.5R")
    print("  • Action: SKIP")
    print("  • Marginal edge, not worth the risk")
    print()
    
    print("C (Poor):")
    print("  • Everything else")
    print("  • Action: SKIP")
    print("  • No edge or negative expectancy")
    print()
    
    # =========================================================================
    # STEP 6: Practical Decision Making
    # =========================================================================
    print("STEP 6: Practical decision making guide...")
    print()
    
    print("DECISION FRAMEWORK:")
    print("=" * 60)
    print()
    
    print("1. CHECK WIN PROBABILITY:")
    print("   • > 65%: Strong edge, consider trading")
    print("   • 55-65%: Decent edge, trade with caution")
    print("   • 45-55%: Marginal, probably skip")
    print("   • < 45%: No edge, definitely skip")
    print()
    
    print("2. CHECK EXPECTED R:")
    print("   • > 1.5R: Excellent risk/reward")
    print("   • 1.0-1.5R: Good risk/reward")
    print("   • 0.5-1.0R: Fair risk/reward")
    print("   • < 0.5R: Poor risk/reward")
    print()
    
    print("3. CHECK INTERVAL WIDTH:")
    print("   • Narrow interval (< 2R): High confidence")
    print("   • Medium interval (2-4R): Moderate confidence")
    print("   • Wide interval (> 4R): Low confidence, high uncertainty")
    print()
    
    print("4. CHECK SKEWNESS:")
    print("   • Positive (> 1): Upside potential, asymmetric gains")
    print("   • Neutral (≈ 1): Symmetric distribution")
    print("   • Negative (< 1): Downside risk, asymmetric losses")
    print()
    
    print("5. COMBINE ALL FACTORS:")
    print("   • Quality A+ or A + Narrow interval + Positive skew = BEST")
    print("   • Quality A + Wide interval = Trade with reduced size")
    print("   • Quality B or C = SKIP regardless of other factors")
    print()
    
    # =========================================================================
    # STEP 7: Common Pitfalls
    # =========================================================================
    print("STEP 7: Common pitfalls to avoid...")
    print()
    
    print("COMMON MISTAKES:")
    print("=" * 60)
    print()
    
    print("❌ Using raw probabilities instead of calibrated")
    print("   • Always use prob_win_calibrated, not prob_win_raw")
    print("   • Raw probabilities are not calibrated to empirical rates")
    print()
    
    print("❌ Ignoring interval width")
    print("   • Wide intervals mean high uncertainty")
    print("   • Don't trade large size with wide intervals")
    print()
    
    print("❌ Focusing only on P50")
    print("   • P50 is expected value, but not the only outcome")
    print("   • Check P10 for worst-case scenario")
    print("   • Check P90 for best-case scenario")
    print()
    
    print("❌ Trading B or C quality setups")
    print("   • These have no edge or negative expectancy")
    print("   • Stick to A+ and A only")
    print()
    
    print("❌ Not monitoring model performance")
    print("   • Models degrade over time")
    print("   • Check metrics regularly")
    print("   • Retrain when performance drops")
    print()
    
    # =========================================================================
    # STEP 8: Best Practices
    # =========================================================================
    print("STEP 8: Best practices...")
    print()
    
    print("BEST PRACTICES:")
    print("=" * 60)
    print()
    
    print("✓ Always use calibrated probabilities")
    print("✓ Always use conformal intervals (P10_conf, P90_conf)")
    print("✓ Filter by quality (A+ or A only)")
    print("✓ Consider interval width for position sizing")
    print("✓ Monitor model performance regularly")
    print("✓ Retrain monthly or when degradation detected")
    print("✓ Validate predictions on out-of-sample data")
    print("✓ Keep a trading journal to track actual vs predicted")
    print("✓ Start with small position sizes until validated")
    print("✓ Have a fallback to rule-based system")
    print()
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 80)
    print("INTERPRETATION GUIDE COMPLETE!")
    print("=" * 80)
    print()
    print("Key Takeaways:")
    print("1. Calibrated probabilities match empirical win rates")
    print("2. Quantiles (P10/P50/P90) show distribution, not just expected value")
    print("3. Conformal intervals provide coverage guarantees")
    print("4. Quality categories (A+/A/B/C) simplify decision making")
    print("5. Always consider multiple factors: prob, R, interval, skewness")
    print()
    print("Next steps:")
    print("1. Practice with real data")
    print("2. Validate predictions against actual outcomes")
    print("3. Adjust thresholds based on your risk tolerance")
    print("4. Monitor and retrain regularly")
    print()

if __name__ == '__main__':
    main()
