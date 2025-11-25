"""
Demo: Setup Categorization

This script demonstrates the setup categorization functionality of the
ML Prediction Engine.

**Feature: ml-prediction-engine**
**Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5**
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.ml_engine.pipeline_prediction import PredictionPipeline


def demo_basic_categorization():
    """Demonstrate basic categorization functionality."""
    print("=" * 70)
    print("DEMO 1: Basic Setup Categorization")
    print("=" * 70)
    
    # Create pipeline (no models needed for categorization demo)
    pipeline = PredictionPipeline()
    
    # Test different quality levels
    test_cases = [
        {
            'name': 'Excellent Setup (A+)',
            'prob_win': 0.70,
            'R_P50': 2.0,
            'R_P10': 0.5,
            'R_P90': 3.5
        },
        {
            'name': 'Good Setup (A)',
            'prob_win': 0.60,
            'R_P50': 1.2,
            'R_P10': 0.3,
            'R_P90': 2.0
        },
        {
            'name': 'Fair Setup (B)',
            'prob_win': 0.50,
            'R_P50': 0.7,
            'R_P10': 0.1,
            'R_P90': 1.5
        },
        {
            'name': 'Poor Setup (C)',
            'prob_win': 0.40,
            'R_P50': 0.3,
            'R_P10': -0.5,
            'R_P90': 1.0
        }
    ]
    
    for case in test_cases:
        quality, recommendation = pipeline.categorize_setup(
            case['prob_win'],
            case['R_P50'],
            case['R_P10'],
            case['R_P90']
        )
        
        color = pipeline.get_quality_color(quality)
        
        print(f"\n{case['name']}:")
        print(f"  Probability: {case['prob_win']:.2f}")
        print(f"  R_P50: {case['R_P50']:.2f}")
        print(f"  → Quality: {quality}")
        print(f"  → Recommendation: {recommendation}")
        print(f"  → Color: {color}")


def demo_threshold_values():
    """Demonstrate threshold value retrieval."""
    print("\n" + "=" * 70)
    print("DEMO 2: Threshold Values")
    print("=" * 70)
    
    pipeline = PredictionPipeline()
    thresholds = pipeline.get_threshold_values()
    
    print("\nCurrent Thresholds:")
    for quality, values in thresholds.items():
        print(f"\n{quality}:")
        print(f"  Probability >= {values['prob_win_min']:.2f}")
        print(f"  R_P50 >= {values['R_P50_min']:.2f}")


def demo_color_coding():
    """Demonstrate color coding for all quality levels."""
    print("\n" + "=" * 70)
    print("DEMO 3: Color Coding")
    print("=" * 70)
    
    pipeline = PredictionPipeline()
    
    quality_labels = ['A+', 'A', 'B', 'C', 'ERROR']
    
    print("\nQuality Label Colors:")
    for label in quality_labels:
        color = pipeline.get_quality_color(label)
        print(f"  {label:6s} → {color}")


def demo_boundary_conditions():
    """Demonstrate behavior at threshold boundaries."""
    print("\n" + "=" * 70)
    print("DEMO 4: Boundary Conditions")
    print("=" * 70)
    
    pipeline = PredictionPipeline()
    thresholds = pipeline.get_threshold_values()
    
    # Test at A+ threshold
    prob_threshold = thresholds['A+']['prob_win_min']
    R_threshold = thresholds['A+']['R_P50_min']
    
    print(f"\nA+ Threshold: prob_win > {prob_threshold}, R_P50 > {R_threshold}")
    
    # Just below threshold
    quality, rec = pipeline.categorize_setup(
        prob_threshold - 0.01,
        R_threshold - 0.01,
        0.0, 0.0
    )
    print(f"  Just below: {quality} ({rec})")
    
    # At threshold
    quality, rec = pipeline.categorize_setup(
        prob_threshold,
        R_threshold,
        0.0, 0.0
    )
    print(f"  At threshold: {quality} ({rec})")
    
    # Just above threshold
    quality, rec = pipeline.categorize_setup(
        prob_threshold + 0.01,
        R_threshold + 0.01,
        0.0, 0.0
    )
    print(f"  Just above: {quality} ({rec})")


def demo_batch_categorization():
    """Demonstrate batch categorization."""
    print("\n" + "=" * 70)
    print("DEMO 5: Batch Categorization")
    print("=" * 70)
    
    pipeline = PredictionPipeline()
    
    # Create sample data
    np.random.seed(42)
    n_samples = 10
    
    # Generate random probabilities and R_P50 values
    prob_wins = np.random.uniform(0.3, 0.8, n_samples)
    R_P50s = np.random.uniform(-0.5, 3.0, n_samples)
    
    # Categorize all samples
    results = []
    for i in range(n_samples):
        quality, recommendation = pipeline.categorize_setup(
            prob_wins[i],
            R_P50s[i],
            0.0, 0.0
        )
        
        results.append({
            'Setup': i + 1,
            'Prob_Win': prob_wins[i],
            'R_P50': R_P50s[i],
            'Quality': quality,
            'Recommendation': recommendation
        })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    print("\nBatch Categorization Results:")
    print(df.to_string(index=False))
    
    # Summary statistics
    print("\nSummary:")
    print(f"  Total setups: {len(df)}")
    print(f"  A+ setups: {(df['Quality'] == 'A+').sum()}")
    print(f"  A setups: {(df['Quality'] == 'A').sum()}")
    print(f"  B setups: {(df['Quality'] == 'B').sum()}")
    print(f"  C setups: {(df['Quality'] == 'C').sum()}")
    print(f"  TRADE recommendations: {(df['Recommendation'] == 'TRADE').sum()}")
    print(f"  SKIP recommendations: {(df['Recommendation'] == 'SKIP').sum()}")


def demo_custom_thresholds():
    """Demonstrate using custom thresholds."""
    print("\n" + "=" * 70)
    print("DEMO 6: Custom Thresholds")
    print("=" * 70)
    
    import yaml
    import tempfile
    
    # Create custom config
    custom_config = {
        'thresholds': {
            'quality_A_plus': {'prob_win_min': 0.70, 'R_P50_min': 2.0},
            'quality_A': {'prob_win_min': 0.60, 'R_P50_min': 1.2},
            'quality_B': {'prob_win_min': 0.50, 'R_P50_min': 0.6}
        },
        'display': {
            'color_scheme': {
                'A_plus': '#006400',
                'A': '#32CD32',
                'B': '#FFD700',
                'C': '#DC143C'
            }
        }
    }
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(custom_config, f)
        config_path = f.name
    
    # Create pipeline with custom config
    pipeline = PredictionPipeline(config_path=config_path)
    
    print("\nCustom Thresholds:")
    thresholds = pipeline.get_threshold_values()
    for quality, values in thresholds.items():
        print(f"  {quality}: prob_win > {values['prob_win_min']:.2f}, "
              f"R_P50 > {values['R_P50_min']:.2f}")
    
    # Test with same values as default config
    test_prob = 0.65
    test_R = 1.5
    
    print(f"\nTest case: prob_win={test_prob}, R_P50={test_R}")
    
    # Default config
    default_pipeline = PredictionPipeline()
    quality_default, rec_default = default_pipeline.categorize_setup(
        test_prob, test_R, 0.0, 0.0
    )
    print(f"  Default config: {quality_default} ({rec_default})")
    
    # Custom config
    quality_custom, rec_custom = pipeline.categorize_setup(
        test_prob, test_R, 0.0, 0.0
    )
    print(f"  Custom config: {quality_custom} ({rec_custom})")
    
    # Cleanup
    Path(config_path).unlink()


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("SETUP CATEGORIZATION DEMO")
    print("ML Prediction Engine - Task 9")
    print("=" * 70)
    
    demo_basic_categorization()
    demo_threshold_values()
    demo_color_coding()
    demo_boundary_conditions()
    demo_batch_categorization()
    demo_custom_thresholds()
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nAll categorization features demonstrated successfully!")
    print("\nKey Features:")
    print("  ✓ Quality categorization (A+/A/B/C)")
    print("  ✓ Trade recommendations (TRADE/SKIP)")
    print("  ✓ Color coding for UI")
    print("  ✓ Configurable thresholds")
    print("  ✓ Batch processing")
    print("  ✓ Boundary condition handling")


if __name__ == '__main__':
    main()
