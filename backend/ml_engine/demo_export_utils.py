"""
Demo script for Export Utilities

This script demonstrates the export functionality for ML predictions,
reports, models, and configurations.
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.ml_engine.export_utils import ExportManager


def create_sample_predictions():
    """Create sample predictions DataFrame for testing."""
    np.random.seed(42)
    
    n_samples = 100
    
    # Generate sample predictions
    data = {
        'prob_win_raw': np.random.uniform(0.4, 0.8, n_samples),
        'prob_win_calibrated': np.random.uniform(0.4, 0.8, n_samples),
        'R_P10_raw': np.random.uniform(-1, 0.5, n_samples),
        'R_P50_raw': np.random.uniform(0.5, 2.0, n_samples),
        'R_P90_raw': np.random.uniform(2.0, 4.0, n_samples),
        'R_P10_conf': np.random.uniform(-1.5, 0.3, n_samples),
        'R_P90_conf': np.random.uniform(2.2, 4.5, n_samples),
        'skewness': np.random.uniform(0.8, 1.5, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Add quality labels
    quality_labels = []
    recommendations = []
    
    for _, row in df.iterrows():
        prob = row['prob_win_calibrated']
        r_p50 = row['R_P50_raw']
        
        if prob > 0.65 and r_p50 > 1.5:
            quality_labels.append('A+')
            recommendations.append('TRADE')
        elif prob > 0.55 and r_p50 > 1.0:
            quality_labels.append('A')
            recommendations.append('TRADE')
        elif prob > 0.45 and r_p50 > 0.5:
            quality_labels.append('B')
            recommendations.append('SKIP')
        else:
            quality_labels.append('C')
            recommendations.append('SKIP')
    
    df['quality_label'] = quality_labels
    df['recommendation'] = recommendations
    
    # Add some sample features
    df['feature_1'] = np.random.uniform(0, 1, n_samples)
    df['feature_2'] = np.random.uniform(-1, 1, n_samples)
    df['feature_3'] = np.random.uniform(0, 100, n_samples)
    
    return df


def create_sample_metrics():
    """Create sample metrics dictionary for testing."""
    return {
        'classifier': {
            'auc_train': 0.78,
            'auc_val': 0.75,
            'brier_score_train': 0.18,
            'brier_score_val': 0.20
        },
        'calibration': {
            'brier_before': 0.20,
            'brier_after': 0.18,
            'ece_before': 0.08,
            'ece_after': 0.03
        },
        'quantile': {
            'mae_p10_train': 0.35,
            'mae_p10_val': 0.38,
            'mae_p50_train': 0.42,
            'mae_p50_val': 0.45,
            'mae_p90_train': 0.48,
            'mae_p90_val': 0.52
        },
        'conformal': {
            'target_coverage': 0.90,
            'actual_coverage_calib': 0.91,
            'actual_coverage_test': 0.89,
            'avg_interval_width': 3.2
        }
    }


def create_sample_config():
    """Create sample configuration dictionary for testing."""
    return {
        'features': {
            'selected': [
                'trend_strength_tf',
                'swing_position',
                'volatility_regime',
                'support_distance',
                'momentum_score'
            ],
            'scaling': 'standard',
            'handle_missing': 'median'
        },
        'thresholds': {
            'quality_A_plus': {
                'prob_win_min': 0.65,
                'R_P50_min': 1.5
            },
            'quality_A': {
                'prob_win_min': 0.55,
                'R_P50_min': 1.0
            },
            'quality_B': {
                'prob_win_min': 0.45,
                'R_P50_min': 0.5
            }
        },
        'model_hyperparameters': {
            'classifier': {
                'n_estimators': 100,
                'learning_rate': 0.05,
                'max_depth': 5
            }
        }
    }


def demo_export_predictions():
    """Demo: Export predictions to CSV."""
    print("\n" + "="*60)
    print("Demo 1: Export Predictions to CSV")
    print("="*60)
    
    # Create sample data
    predictions_df = create_sample_predictions()
    print(f"Created sample predictions: {len(predictions_df)} rows")
    
    # Initialize exporter
    exporter = ExportManager(output_dir='output/exports')
    
    # Export with all features
    print("\nExporting predictions with all features...")
    path1 = exporter.export_predictions_to_csv(
        predictions_df,
        filename='demo_predictions_full.csv',
        include_features=True
    )
    print(f"✓ Exported to: {path1}")
    
    # Export predictions only (no features)
    print("\nExporting predictions only (no features)...")
    path2 = exporter.export_predictions_to_csv(
        predictions_df,
        filename='demo_predictions_only.csv',
        include_features=False
    )
    print(f"✓ Exported to: {path2}")
    
    return path1, path2


def demo_export_report():
    """Demo: Export report to HTML/PDF."""
    print("\n" + "="*60)
    print("Demo 2: Export Report to HTML")
    print("="*60)
    
    # Create sample data
    predictions_df = create_sample_predictions()
    metrics = create_sample_metrics()
    
    print(f"Created sample data: {len(predictions_df)} predictions")
    print(f"Metrics: {len(metrics)} categories")
    
    # Initialize exporter
    exporter = ExportManager(output_dir='output/exports')
    
    # Export report
    print("\nExporting report to HTML...")
    path = exporter.export_report_to_pdf(
        predictions_df,
        metrics,
        filename='demo_report.html',
        include_charts=True
    )
    print(f"✓ Exported to: {path}")
    print("  Note: Open this file in a browser to view the report")
    print("  To convert to PDF, use: wkhtmltopdf demo_report.html demo_report.pdf")
    
    return path


def demo_export_config():
    """Demo: Export configuration to YAML."""
    print("\n" + "="*60)
    print("Demo 3: Export Configuration to YAML")
    print("="*60)
    
    # Create sample config
    config = create_sample_config()
    print(f"Created sample config with {len(config)} sections")
    
    # Initialize exporter
    exporter = ExportManager(output_dir='output/exports')
    
    # Export config
    print("\nExporting configuration...")
    path = exporter.export_config_to_yaml(
        config,
        filename='demo_config_backup.yaml',
        add_timestamp=True
    )
    print(f"✓ Exported to: {path}")
    
    # Verify by reading back
    print("\nVerifying export by reading back...")
    with open(path, 'r') as f:
        loaded_config = yaml.safe_load(f)
    
    print(f"✓ Successfully loaded config with {len(loaded_config)} sections")
    if 'export_metadata' in loaded_config:
        print(f"  Export timestamp: {loaded_config['export_metadata']['export_timestamp']}")
    
    return path


def demo_export_model_to_onnx():
    """Demo: Export model to ONNX (if model exists)."""
    print("\n" + "="*60)
    print("Demo 4: Export Model to ONNX")
    print("="*60)
    
    # Check if trained model exists
    model_path = Path('data_processed/models/lgbm_classifier.pkl')
    
    if not model_path.exists():
        print(f"⚠ Model not found at {model_path}")
        print("  Skipping ONNX export demo")
        print("  Train a model first using demo_model_trainer.py")
        return None
    
    print(f"Found model at: {model_path}")
    
    # Initialize exporter
    exporter = ExportManager(output_dir='output/exports')
    
    try:
        # Export to ONNX
        print("\nExporting model to ONNX format...")
        path = exporter.export_model_to_onnx(
            model_path,
            output_filename='demo_classifier.onnx',
            model_type='classifier'
        )
        print(f"✓ Exported to: {path}")
        print("  This ONNX model can be deployed to EA or other platforms")
        
        return path
        
    except ImportError as e:
        print(f"⚠ {e}")
        print("  Install required packages: pip install onnxmltools skl2onnx")
        return None
    except Exception as e:
        print(f"✗ Export failed: {e}")
        return None


def demo_export_summary():
    """Demo: Create export summary."""
    print("\n" + "="*60)
    print("Demo 5: Create Export Summary")
    print("="*60)
    
    # Create sample exports
    predictions_df = create_sample_predictions()
    config = create_sample_config()
    
    exporter = ExportManager(output_dir='output/exports')
    
    # Perform multiple exports
    exports = {}
    
    print("\nPerforming multiple exports...")
    
    exports['predictions'] = exporter.export_predictions_to_csv(
        predictions_df,
        filename='summary_predictions.csv'
    )
    print(f"✓ Exported predictions")
    
    exports['config'] = exporter.export_config_to_yaml(
        config,
        filename='summary_config.yaml'
    )
    print(f"✓ Exported config")
    
    # Create summary
    print("\nCreating export summary...")
    summary_path = exporter.create_export_summary(
        exports,
        filename='export_summary.txt'
    )
    print(f"✓ Created summary at: {summary_path}")
    
    # Display summary
    print("\nSummary contents:")
    print("-" * 60)
    with open(summary_path, 'r') as f:
        print(f.read())
    
    return summary_path


def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("ML Prediction Engine - Export Utilities Demo")
    print("="*60)
    
    try:
        # Demo 1: Export predictions
        demo_export_predictions()
        
        # Demo 2: Export report
        demo_export_report()
        
        # Demo 3: Export config
        demo_export_config()
        
        # Demo 4: Export model to ONNX
        demo_export_model_to_onnx()
        
        # Demo 5: Export summary
        demo_export_summary()
        
        print("\n" + "="*60)
        print("All demos completed successfully!")
        print("="*60)
        print("\nExported files are in: output/exports/")
        print("\nNext steps:")
        print("  1. Check the exported CSV files")
        print("  2. Open the HTML report in a browser")
        print("  3. Review the config backup")
        print("  4. (Optional) Convert HTML to PDF using wkhtmltopdf")
        print("  5. (Optional) Install onnxmltools for ONNX export")
        
    except Exception as e:
        print(f"\n✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
