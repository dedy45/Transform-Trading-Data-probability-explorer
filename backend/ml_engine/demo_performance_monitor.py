"""
Demo script for PerformanceMonitor

This script demonstrates how to use the PerformanceMonitor class to:
1. Calculate performance metrics
2. Detect performance degradation
3. Track metrics over time
4. Export metrics history

**Feature: ml-prediction-engine**
"""

import numpy as np
import pandas as pd
from pathlib import Path

from backend.ml_engine.performance_monitor import (
    PerformanceMonitor,
    monitor_model_performance
)


def generate_sample_data(n=1000, seed=42):
    """Generate sample predictions for demonstration."""
    np.random.seed(seed)
    
    # Generate true values
    y_true_win = np.random.randint(0, 2, n)
    y_true_r = np.random.randn(n) * 2
    
    # Generate predictions (with some correlation to true values)
    y_pred_prob = np.clip(y_true_win * 0.7 + np.random.randn(n) * 0.2, 0, 1)
    y_pred_p10 = y_true_r - 1 + np.random.randn(n) * 0.3
    y_pred_p50 = y_true_r + np.random.randn(n) * 0.2
    y_pred_p90 = y_true_r + 1 + np.random.randn(n) * 0.3
    
    # Generate conformal adjusted intervals (wider)
    y_pred_p10_conf = y_pred_p10 - 0.3
    y_pred_p90_conf = y_pred_p90 + 0.3
    
    return {
        'y_true_win': y_true_win,
        'y_pred_prob': y_pred_prob,
        'y_true_r': y_true_r,
        'y_pred_p10': y_pred_p10,
        'y_pred_p50': y_pred_p50,
        'y_pred_p90': y_pred_p90,
        'y_pred_p10_conf': y_pred_p10_conf,
        'y_pred_p90_conf': y_pred_p90_conf
    }


def demo_basic_metrics():
    """Demo 1: Calculate basic performance metrics."""
    print("=" * 80)
    print("DEMO 1: Calculate Basic Performance Metrics")
    print("=" * 80)
    
    # Generate sample data
    data = generate_sample_data(n=500)
    
    # Create monitor
    monitor = PerformanceMonitor()
    
    # Calculate metrics
    metrics = monitor.calculate_metrics(
        y_true_win=data['y_true_win'],
        y_pred_prob=data['y_pred_prob'],
        y_true_r=data['y_true_r'],
        y_pred_p10=data['y_pred_p10'],
        y_pred_p50=data['y_pred_p50'],
        y_pred_p90=data['y_pred_p90'],
        y_pred_p10_conf=data['y_pred_p10_conf'],
        y_pred_p90_conf=data['y_pred_p90_conf']
    )
    
    # Display metrics
    print("\nPerformance Metrics:")
    print(f"  AUC:                {metrics['auc']:.4f}")
    print(f"  Brier Score:        {metrics['brier_score']:.4f}")
    print(f"  MAE P10:            {metrics['mae_p10']:.4f}")
    print(f"  MAE P50:            {metrics['mae_p50']:.4f}")
    print(f"  MAE P90:            {metrics['mae_p90']:.4f}")
    print(f"  Coverage (Raw):     {metrics['coverage_raw']:.4f}")
    print(f"  Coverage (Conf):    {metrics['coverage_conf']:.4f}")
    print(f"  N Samples:          {metrics['n_samples']}")
    print(f"  Timestamp:          {metrics['timestamp']}")
    
    print("\nInterpretation:")
    print(f"  - AUC {metrics['auc']:.4f}: {'Good' if metrics['auc'] > 0.7 else 'Fair' if metrics['auc'] > 0.6 else 'Poor'}")
    print(f"  - Brier {metrics['brier_score']:.4f}: {'Good' if metrics['brier_score'] < 0.20 else 'Fair' if metrics['brier_score'] < 0.25 else 'Poor'}")
    print(f"  - Coverage {metrics['coverage_conf']:.4f}: {'Good' if abs(metrics['coverage_conf'] - 0.90) < 0.05 else 'Needs adjustment'}")


def demo_degradation_detection():
    """Demo 2: Detect performance degradation."""
    print("\n" + "=" * 80)
    print("DEMO 2: Detect Performance Degradation")
    print("=" * 80)
    
    # Generate baseline data (good performance)
    baseline_data = generate_sample_data(n=500, seed=42)
    
    # Calculate baseline metrics
    monitor = PerformanceMonitor(degradation_threshold=0.10)
    
    baseline_metrics = monitor.calculate_metrics(
        y_true_win=baseline_data['y_true_win'],
        y_pred_prob=baseline_data['y_pred_prob'],
        y_true_r=baseline_data['y_true_r'],
        y_pred_p10=baseline_data['y_pred_p10'],
        y_pred_p50=baseline_data['y_pred_p50'],
        y_pred_p90=baseline_data['y_pred_p90']
    )
    
    print("\nBaseline Metrics:")
    print(f"  AUC:          {baseline_metrics['auc']:.4f}")
    print(f"  Brier Score:  {baseline_metrics['brier_score']:.4f}")
    print(f"  MAE P50:      {baseline_metrics['mae_p50']:.4f}")
    
    # Set as baseline
    monitor.set_baseline_metrics(baseline_metrics)
    
    # Generate degraded data (worse performance)
    degraded_data = generate_sample_data(n=500, seed=123)
    # Make predictions worse
    degraded_data['y_pred_prob'] = np.clip(
        degraded_data['y_pred_prob'] + np.random.randn(500) * 0.3,
        0, 1
    )
    
    current_metrics = monitor.calculate_metrics(
        y_true_win=degraded_data['y_true_win'],
        y_pred_prob=degraded_data['y_pred_prob'],
        y_true_r=degraded_data['y_true_r'],
        y_pred_p10=degraded_data['y_pred_p10'],
        y_pred_p50=degraded_data['y_pred_p50'],
        y_pred_p90=degraded_data['y_pred_p90']
    )
    
    print("\nCurrent Metrics:")
    print(f"  AUC:          {current_metrics['auc']:.4f}")
    print(f"  Brier Score:  {current_metrics['brier_score']:.4f}")
    print(f"  MAE P50:      {current_metrics['mae_p50']:.4f}")
    
    # Detect degradation
    degradation = monitor.detect_degradation(current_metrics)
    
    print("\nDegradation Detection:")
    print(f"  Is Degraded:       {degradation['is_degraded']}")
    print(f"  Degraded Metrics:  {degradation['degraded_metrics']}")
    
    if degradation['is_degraded']:
        print(f"\n  [!] ALERT: {degradation['alert_message']}")
        print("\n  Degradation Details:")
        for metric, details in degradation['degradation_details'].items():
            print(f"    {metric}:")
            print(f"      Baseline: {details['baseline']:.4f}")
            print(f"      Current:  {details['current']:.4f}")
            if 'degradation_pct' in details:
                print(f"      Degradation: {details['degradation_pct']:.2f}%")
            elif 'deviation' in details:
                print(f"      Deviation: {details['deviation']:.4f}")


def demo_metrics_tracking():
    """Demo 3: Track metrics over time."""
    print("\n" + "=" * 80)
    print("DEMO 3: Track Metrics Over Time")
    print("=" * 80)
    
    # Create monitor with history tracking
    history_path = Path('data_processed/models/metrics_history_demo.json')
    monitor = PerformanceMonitor(metrics_history_path=str(history_path))
    
    print(f"\nTracking metrics to: {history_path}")
    
    # Simulate multiple evaluation periods
    print("\nSimulating 10 evaluation periods...")
    for i in range(10):
        # Generate data with gradually degrading performance
        data = generate_sample_data(n=300, seed=42 + i)
        
        # Add noise that increases over time (simulating drift)
        noise_level = 0.1 + i * 0.02
        data['y_pred_prob'] = np.clip(
            data['y_pred_prob'] + np.random.randn(300) * noise_level,
            0, 1
        )
        
        # Calculate metrics
        metrics = monitor.calculate_metrics(
            y_true_win=data['y_true_win'],
            y_pred_prob=data['y_pred_prob'],
            y_true_r=data['y_true_r'],
            y_pred_p10=data['y_pred_p10'],
            y_pred_p50=data['y_pred_p50'],
            y_pred_p90=data['y_pred_p90'],
            timestamp=f"2025-11-{24+i:02d}T10:00:00"
        )
        
        # Add to history
        monitor.add_metrics_record(metrics, save_to_file=True)
        
        print(f"  Period {i+1}: AUC={metrics['auc']:.4f}, Brier={metrics['brier_score']:.4f}")
    
    # Get metrics history
    history_df = monitor.get_metrics_history()
    
    print(f"\nMetrics History Summary:")
    print(f"  Total Records: {len(history_df)}")
    print(f"\n  AUC Statistics:")
    print(f"    Mean:  {history_df['auc'].mean():.4f}")
    print(f"    Std:   {history_df['auc'].std():.4f}")
    print(f"    Min:   {history_df['auc'].min():.4f}")
    print(f"    Max:   {history_df['auc'].max():.4f}")
    
    # Calculate rolling metrics
    rolling = monitor.calculate_rolling_metrics(window_size=5)
    
    if not rolling.empty:
        print(f"\n  Rolling Metrics (window=5):")
        print(f"    Latest AUC Mean: {rolling['auc_mean'].iloc[-1]:.4f}")
        print(f"    Latest AUC Std:  {rolling['auc_std'].iloc[-1]:.4f}")


def demo_export_metrics():
    """Demo 4: Export metrics history."""
    print("\n" + "=" * 80)
    print("DEMO 4: Export Metrics History")
    print("=" * 80)
    
    # Create monitor and add some records
    monitor = PerformanceMonitor()
    
    print("\nGenerating metrics records...")
    for i in range(5):
        data = generate_sample_data(n=200, seed=42 + i)
        
        metrics = monitor.calculate_metrics(
            y_true_win=data['y_true_win'],
            y_pred_prob=data['y_pred_prob'],
            y_true_r=data['y_true_r'],
            y_pred_p10=data['y_pred_p10'],
            y_pred_p50=data['y_pred_p50'],
            y_pred_p90=data['y_pred_p90']
        )
        
        monitor.add_metrics_record(metrics, save_to_file=False)
    
    # Export to CSV
    csv_path = Path('data_processed/models/metrics_export_demo.csv')
    monitor.export_metrics_history(str(csv_path), format='csv')
    print(f"\n[OK] Exported to CSV: {csv_path}")
    
    # Export to JSON
    json_path = Path('data_processed/models/metrics_export_demo.json')
    monitor.export_metrics_history(str(json_path), format='json')
    print(f"[OK] Exported to JSON: {json_path}")
    
    # Show CSV preview
    df = pd.read_csv(csv_path)
    print(f"\nCSV Preview (first 3 rows):")
    print(df.head(3).to_string())


def demo_convenience_function():
    """Demo 5: Use convenience function."""
    print("\n" + "=" * 80)
    print("DEMO 5: Convenience Function")
    print("=" * 80)
    
    # Generate data
    data = generate_sample_data(n=400)
    
    # Define baseline
    baseline = {
        'auc': 0.75,
        'brier_score': 0.20,
        'mae_p50': 0.50,
        'coverage_raw': 0.85
    }
    
    print("\nUsing convenience function with baseline...")
    
    # Calculate metrics and detect degradation in one call
    metrics, degradation = monitor_model_performance(
        y_true_win=data['y_true_win'],
        y_pred_prob=data['y_pred_prob'],
        y_true_r=data['y_true_r'],
        y_pred_p10=data['y_pred_p10'],
        y_pred_p50=data['y_pred_p50'],
        y_pred_p90=data['y_pred_p90'],
        baseline_metrics=baseline,
        degradation_threshold=0.10
    )
    
    print("\nResults:")
    print(f"  Current AUC:       {metrics['auc']:.4f}")
    print(f"  Baseline AUC:      {baseline['auc']:.4f}")
    print(f"  Is Degraded:       {degradation['is_degraded']}")
    
    if degradation['is_degraded']:
        print(f"\n  [!] {degradation['alert_message']}")


def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("PERFORMANCE MONITOR DEMO")
    print("=" * 80)
    
    # Run demos
    demo_basic_metrics()
    demo_degradation_detection()
    demo_metrics_tracking()
    demo_export_metrics()
    demo_convenience_function()
    
    print("\n" + "=" * 80)
    print("All demos completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
