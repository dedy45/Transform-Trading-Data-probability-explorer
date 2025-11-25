"""
Demo script for R Distribution Fan Chart Component

This script demonstrates the usage of r_distribution_fan_chart.py
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from frontend.components.r_distribution_fan_chart import (
    create_fan_chart,
    create_comparison_histogram,
    create_distribution_analysis_section,
    calculate_coverage,
    calculate_skewness,
    interpret_skewness,
    create_empty_fan_chart,
    create_empty_histogram
)


def demo_basic_fan_chart():
    """Demo: Basic fan chart with predicted quantiles"""
    print("\n" + "="*60)
    print("DEMO 1: Basic Fan Chart")
    print("="*60)
    
    # Generate sample predictions
    n_samples = 50
    np.random.seed(42)
    
    # Simulate predictions with positive skew
    y_pred_p10 = np.random.uniform(-0.5, 0.5, n_samples)
    y_pred_p50 = np.random.uniform(0.8, 1.5, n_samples)
    y_pred_p90 = np.random.uniform(2.0, 3.5, n_samples)
    
    # Create fan chart
    fig = create_fan_chart(y_pred_p10, y_pred_p50, y_pred_p90)
    
    print(f"✓ Created fan chart with {n_samples} samples")
    print(f"  - P10 range: [{y_pred_p10.min():.2f}, {y_pred_p10.max():.2f}]")
    print(f"  - P50 range: [{y_pred_p50.min():.2f}, {y_pred_p50.max():.2f}]")
    print(f"  - P90 range: [{y_pred_p90.min():.2f}, {y_pred_p90.max():.2f}]")
    
    # Save figure
    fig.write_html('output/demo_fan_chart_basic.html')
    print(f"✓ Saved to output/demo_fan_chart_basic.html")
    
    return fig


def demo_fan_chart_with_actuals():
    """Demo: Fan chart with actual R_multiple overlay"""
    print("\n" + "="*60)
    print("DEMO 2: Fan Chart with Actual Values")
    print("="*60)
    
    # Generate sample predictions
    n_samples = 50
    np.random.seed(42)
    
    y_pred_p10 = np.random.uniform(-0.5, 0.5, n_samples)
    y_pred_p50 = np.random.uniform(0.8, 1.5, n_samples)
    y_pred_p90 = np.random.uniform(2.0, 3.5, n_samples)
    
    # Generate actual values (some within interval, some outside)
    y_true = y_pred_p50 + np.random.normal(0, 0.5, n_samples)
    
    # Create fan chart with actuals
    fig = create_fan_chart(y_pred_p10, y_pred_p50, y_pred_p90, y_true=y_true)
    
    # Calculate coverage
    coverage = calculate_coverage(y_true, y_pred_p10, y_pred_p90)
    
    print(f"✓ Created fan chart with actual values overlay")
    print(f"  - Coverage: {coverage:.1f}%")
    print(f"  - Actual range: [{y_true.min():.2f}, {y_true.max():.2f}]")
    
    # Count points within/outside interval
    within = np.sum((y_true >= y_pred_p10) & (y_true <= y_pred_p90))
    outside = n_samples - within
    print(f"  - Points within interval: {within} (green)")
    print(f"  - Points outside interval: {outside} (red)")
    
    # Save figure
    fig.write_html('output/demo_fan_chart_with_actuals.html')
    print(f"✓ Saved to output/demo_fan_chart_with_actuals.html")
    
    return fig


def demo_skewness_calculation():
    """Demo: Skewness calculation and interpretation"""
    print("\n" + "="*60)
    print("DEMO 3: Skewness Calculation")
    print("="*60)
    
    # Test different skewness scenarios
    scenarios = [
        {
            'name': 'Strong Upside Potential',
            'p10': np.array([0.0, 0.5, 1.0]),
            'p50': np.array([1.0, 1.5, 2.0]),
            'p90': np.array([3.0, 4.0, 5.0])
        },
        {
            'name': 'Balanced Distribution',
            'p10': np.array([0.0, 0.5, 1.0]),
            'p50': np.array([1.0, 1.5, 2.0]),
            'p90': np.array([2.0, 2.5, 3.0])
        },
        {
            'name': 'Strong Downside Risk',
            'p10': np.array([-2.0, -1.5, -1.0]),
            'p50': np.array([0.0, 0.5, 1.0]),
            'p90': np.array([0.5, 1.0, 1.5])
        }
    ]
    
    for scenario in scenarios:
        skewness = calculate_skewness(
            scenario['p10'], 
            scenario['p50'], 
            scenario['p90']
        )
        interpretation = interpret_skewness(skewness)
        
        print(f"\n{scenario['name']}:")
        print(f"  - Skewness: {skewness:.2f}")
        print(f"  - Interpretation: {interpretation}")


def demo_comparison_histogram():
    """Demo: Comparison histogram of predicted vs actual"""
    print("\n" + "="*60)
    print("DEMO 4: Comparison Histogram")
    print("="*60)
    
    # Generate sample data
    n_samples = 500
    np.random.seed(42)
    
    # Predicted distribution (slightly optimistic)
    y_pred_p50 = np.random.normal(1.2, 0.8, n_samples)
    
    # Actual distribution (slightly worse than predicted)
    y_true = np.random.normal(1.0, 0.9, n_samples)
    
    # Create histogram
    fig = create_comparison_histogram(y_pred_p50, y_true)
    
    print(f"✓ Created comparison histogram")
    print(f"  - Predicted mean: {y_pred_p50.mean():.2f}")
    print(f"  - Actual mean: {y_true.mean():.2f}")
    print(f"  - Predicted std: {y_pred_p50.std():.2f}")
    print(f"  - Actual std: {y_true.std():.2f}")
    
    # Save figure
    fig.write_html('output/demo_comparison_histogram.html')
    print(f"✓ Saved to output/demo_comparison_histogram.html")
    
    return fig


def demo_complete_analysis():
    """Demo: Complete distribution analysis section"""
    print("\n" + "="*60)
    print("DEMO 5: Complete Distribution Analysis")
    print("="*60)
    
    # Generate sample predictions
    n_samples = 100
    np.random.seed(42)
    
    y_pred_p10 = np.random.uniform(-0.5, 0.5, n_samples)
    y_pred_p50 = np.random.uniform(0.8, 1.5, n_samples)
    y_pred_p90 = np.random.uniform(2.0, 3.5, n_samples)
    y_true = y_pred_p50 + np.random.normal(0, 0.5, n_samples)
    
    # Create predictions dictionary
    predictions_dict = {
        'R_P10_conf': y_pred_p10,
        'R_P50_raw': y_pred_p50,
        'R_P90_conf': y_pred_p90,
        'R_actual': y_true
    }
    
    # Create complete analysis
    result = create_distribution_analysis_section(predictions_dict)
    
    print(f"✓ Created complete distribution analysis")
    print(f"\nMetrics:")
    print(f"  - Coverage: {result['coverage']:.1f}%")
    print(f"  - Skewness: {result['skewness']:.2f}")
    print(f"  - Interpretation: {result['skewness_interpretation']}")
    
    # Save figures
    result['fan_chart'].write_html('output/demo_complete_fan_chart.html')
    result['histogram'].write_html('output/demo_complete_histogram.html')
    print(f"\n✓ Saved fan chart to output/demo_complete_fan_chart.html")
    print(f"✓ Saved histogram to output/demo_complete_histogram.html")
    
    return result


def demo_empty_states():
    """Demo: Empty state visualizations"""
    print("\n" + "="*60)
    print("DEMO 6: Empty State Visualizations")
    print("="*60)
    
    # Create empty fan chart
    fig_fan = create_empty_fan_chart()
    fig_fan.write_html('output/demo_empty_fan_chart.html')
    print(f"✓ Created empty fan chart")
    print(f"  Saved to output/demo_empty_fan_chart.html")
    
    # Create empty histogram
    fig_hist = create_empty_histogram()
    fig_hist.write_html('output/demo_empty_histogram.html')
    print(f"✓ Created empty histogram")
    print(f"  Saved to output/demo_empty_histogram.html")


def main():
    """Run all demos"""
    print("\n" + "="*60)
    print("R DISTRIBUTION FAN CHART COMPONENT DEMO")
    print("="*60)
    
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    try:
        # Run demos
        demo_basic_fan_chart()
        demo_fan_chart_with_actuals()
        demo_skewness_calculation()
        demo_comparison_histogram()
        demo_complete_analysis()
        demo_empty_states()
        
        print("\n" + "="*60)
        print("ALL DEMOS COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nGenerated files in output/ directory:")
        print("  - demo_fan_chart_basic.html")
        print("  - demo_fan_chart_with_actuals.html")
        print("  - demo_comparison_histogram.html")
        print("  - demo_complete_fan_chart.html")
        print("  - demo_complete_histogram.html")
        print("  - demo_empty_fan_chart.html")
        print("  - demo_empty_histogram.html")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
