"""
Export Utilities - ML Prediction Engine

This module provides export functionality for ML predictions, reports, models,
and configurations.

**Feature: ml-prediction-engine**
**Validates: Requirements 16.1, 16.2, 16.3, 16.4, 16.5**
"""

import pandas as pd
import numpy as np
import yaml
import json
import joblib
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Union, List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExportManager:
    """
    Manager for exporting ML predictions, reports, models, and configurations.
    
    This class provides methods to export:
    - Predictions to CSV
    - Reports to PDF (with metrics and charts)
    - Models to ONNX format
    - Configuration to YAML
    
    Attributes
    ----------
    output_dir : Path
        Base directory for exports
    
    Examples
    --------
    >>> exporter = ExportManager(output_dir='output/exports')
    >>> exporter.export_predictions_to_csv(predictions_df, 'predictions_2024.csv')
    >>> exporter.export_config_to_yaml(config_dict, 'backup_config.yaml')
    """
    
    def __init__(self, output_dir: Union[str, Path] = 'output/exports'):
        """
        Initialize export manager.
        
        Parameters
        ----------
        output_dir : str or Path
            Base directory for exports. Created if doesn't exist.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ExportManager initialized with output_dir: {self.output_dir}")
    
    def export_predictions_to_csv(
        self,
        predictions_df: pd.DataFrame,
        filename: Optional[str] = None,
        include_timestamp: bool = True,
        include_features: bool = True
    ) -> Path:
        """
        Export predictions to CSV with all prediction columns.
        
        Parameters
        ----------
        predictions_df : pd.DataFrame
            DataFrame with predictions (from pipeline.predict_for_batch())
        filename : str, optional
            Output filename. If None, generates timestamp-based name.
        include_timestamp : bool, default=True
            Whether to add timestamp column
        include_features : bool, default=True
            Whether to include original feature columns
        
        Returns
        -------
        Path
            Path to exported CSV file
        
        Raises
        ------
        ValueError
            If predictions_df is empty or missing required columns
        
        Examples
        --------
        >>> exporter = ExportManager()
        >>> path = exporter.export_predictions_to_csv(predictions_df)
        >>> print(f"Exported to: {path}")
        """
        if predictions_df is None or len(predictions_df) == 0:
            raise ValueError("predictions_df cannot be None or empty")
        
        # Required prediction columns
        required_cols = [
            'prob_win_raw', 'prob_win_calibrated',
            'R_P10_raw', 'R_P50_raw', 'R_P90_raw',
            'R_P10_conf', 'R_P90_conf',
            'skewness', 'quality_label', 'recommendation'
        ]
        
        # Check for required columns
        missing_cols = [col for col in required_cols if col not in predictions_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Create export DataFrame
        export_df = predictions_df.copy()
        
        # Add timestamp if requested
        if include_timestamp and 'export_timestamp' not in export_df.columns:
            export_df.insert(0, 'export_timestamp', datetime.now().isoformat())
        
        # Select columns to export
        if include_features:
            # Export all columns
            cols_to_export = export_df.columns.tolist()
        else:
            # Export only prediction columns
            cols_to_export = ['export_timestamp'] + required_cols if include_timestamp else required_cols
            cols_to_export = [col for col in cols_to_export if col in export_df.columns]
        
        export_df = export_df[cols_to_export]
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'ml_predictions_{timestamp}.csv'
        
        # Ensure .csv extension
        if not filename.endswith('.csv'):
            filename += '.csv'
        
        # Export path
        export_path = self.output_dir / filename
        
        # Export to CSV
        export_df.to_csv(export_path, index=False)
        
        logger.info(f"Exported {len(export_df)} predictions to {export_path}")
        
        return export_path
    
    def export_report_to_pdf(
        self,
        predictions_df: pd.DataFrame,
        metrics: Dict,
        filename: Optional[str] = None,
        include_charts: bool = True
    ) -> Path:
        """
        Export report to PDF with summary metrics and charts.
        
        Note: This is a simplified implementation that exports to HTML first,
        then can be converted to PDF using external tools (wkhtmltopdf, weasyprint).
        For now, we export to HTML as PDF generation requires additional dependencies.
        
        Parameters
        ----------
        predictions_df : pd.DataFrame
            DataFrame with predictions
        metrics : dict
            Dictionary with training/validation metrics
        filename : str, optional
            Output filename. If None, generates timestamp-based name.
        include_charts : bool, default=True
            Whether to include charts in report
        
        Returns
        -------
        Path
            Path to exported HTML report file
        
        Examples
        --------
        >>> exporter = ExportManager()
        >>> metrics = {'auc': 0.75, 'brier_score': 0.18}
        >>> path = exporter.export_report_to_pdf(predictions_df, metrics)
        """
        if predictions_df is None or len(predictions_df) == 0:
            raise ValueError("predictions_df cannot be None or empty")
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'ml_report_{timestamp}.html'
        
        # Ensure .html extension (we export to HTML for now)
        if not filename.endswith('.html'):
            filename = filename.replace('.pdf', '.html')
            if not filename.endswith('.html'):
                filename += '.html'
        
        # Export path
        export_path = self.output_dir / filename
        
        # Generate HTML report
        html_content = self._generate_html_report(predictions_df, metrics, include_charts)
        
        # Write to file
        with open(export_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Exported report to {export_path}")
        logger.info("Note: Report exported as HTML. Use wkhtmltopdf or weasyprint to convert to PDF.")
        
        return export_path
    
    def _generate_html_report(
        self,
        predictions_df: pd.DataFrame,
        metrics: Dict,
        include_charts: bool
    ) -> str:
        """
        Generate HTML report content.
        
        Parameters
        ----------
        predictions_df : pd.DataFrame
            Predictions data
        metrics : dict
            Metrics dictionary
        include_charts : bool
            Whether to include charts
        
        Returns
        -------
        str
            HTML content
        """
        # Calculate summary statistics
        n_predictions = len(predictions_df)
        
        # Quality distribution
        quality_counts = predictions_df['quality_label'].value_counts().to_dict()
        
        # Average metrics
        avg_prob_win = predictions_df['prob_win_calibrated'].mean()
        avg_R_P50 = predictions_df['R_P50_raw'].mean()
        
        # Recommendation distribution
        recommendation_counts = predictions_df['recommendation'].value_counts().to_dict()
        
        # Generate HTML
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>ML Prediction Engine Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 8px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 6px;
            border-left: 4px solid #3498db;
        }}
        .metric-label {{
            font-size: 14px;
            color: #7f8c8d;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .quality-A-plus {{
            color: #006400;
            font-weight: bold;
        }}
        .quality-A {{
            color: #32CD32;
            font-weight: bold;
        }}
        .quality-B {{
            color: #FFD700;
            font-weight: bold;
        }}
        .quality-C {{
            color: #DC143C;
            font-weight: bold;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #7f8c8d;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>ML Prediction Engine Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Summary Statistics</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Total Predictions</div>
                <div class="metric-value">{n_predictions}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Avg Win Probability</div>
                <div class="metric-value">{avg_prob_win:.3f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Avg Expected R</div>
                <div class="metric-value">{avg_R_P50:.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">TRADE Recommendations</div>
                <div class="metric-value">{recommendation_counts.get('TRADE', 0)}</div>
            </div>
        </div>
        
        <h2>Quality Distribution</h2>
        <table>
            <tr>
                <th>Quality Label</th>
                <th>Count</th>
                <th>Percentage</th>
            </tr>
"""
        
        # Add quality distribution rows
        for quality in ['A+', 'A', 'B', 'C']:
            count = quality_counts.get(quality, 0)
            percentage = (count / n_predictions * 100) if n_predictions > 0 else 0
            quality_class = f"quality-{quality.replace('+', '-plus')}"
            html += f"""
            <tr>
                <td class="{quality_class}">{quality}</td>
                <td>{count}</td>
                <td>{percentage:.1f}%</td>
            </tr>
"""
        
        html += """
        </table>
        
        <h2>Model Performance Metrics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
"""
        
        # Add metrics rows
        if metrics:
            for key, value in metrics.items():
                if isinstance(value, dict):
                    # Nested metrics
                    for sub_key, sub_value in value.items():
                        display_key = f"{key}.{sub_key}"
                        if isinstance(sub_value, (int, float)):
                            formatted_value = f"{sub_value:.4f}" if isinstance(sub_value, float) else str(sub_value)
                            html += f"""
            <tr>
                <td>{display_key}</td>
                <td>{formatted_value}</td>
            </tr>
"""
                elif isinstance(value, (int, float)):
                    formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
                    html += f"""
            <tr>
                <td>{key}</td>
                <td>{formatted_value}</td>
            </tr>
"""
        
        html += """
        </table>
        
        <h2>Prediction Sample (First 10 rows)</h2>
        <table>
            <tr>
                <th>Prob Win</th>
                <th>R P50</th>
                <th>R P10-P90</th>
                <th>Quality</th>
                <th>Recommendation</th>
            </tr>
"""
        
        # Add sample predictions
        sample_df = predictions_df.head(10)
        for _, row in sample_df.iterrows():
            quality_class = f"quality-{row['quality_label'].replace('+', '-plus')}"
            html += f"""
            <tr>
                <td>{row['prob_win_calibrated']:.3f}</td>
                <td>{row['R_P50_raw']:.2f}</td>
                <td>[{row['R_P10_conf']:.2f}, {row['R_P90_conf']:.2f}]</td>
                <td class="{quality_class}">{row['quality_label']}</td>
                <td>{row['recommendation']}</td>
            </tr>
"""
        
        html += """
        </table>
        
        <div class="footer">
            <p>ML Prediction Engine - Trading Probability Explorer</p>
            <p>This report was automatically generated. For PDF conversion, use wkhtmltopdf or weasyprint.</p>
        </div>
    </div>
</body>
</html>
"""
        
        return html
    
    def export_model_to_onnx(
        self,
        model_path: Union[str, Path],
        output_filename: Optional[str] = None,
        model_type: str = 'classifier'
    ) -> Path:
        """
        Export LightGBM model to ONNX format for deployment to EA.
        
        Note: This requires onnxmltools and skl2onnx packages.
        If not available, this method will raise an ImportError with instructions.
        
        Parameters
        ----------
        model_path : str or Path
            Path to trained LightGBM model (.pkl file)
        output_filename : str, optional
            Output filename. If None, generates name based on model_type.
        model_type : str, default='classifier'
            Type of model ('classifier', 'quantile_p10', 'quantile_p50', 'quantile_p90')
        
        Returns
        -------
        Path
            Path to exported ONNX file
        
        Raises
        ------
        ImportError
            If onnxmltools or skl2onnx are not installed
        FileNotFoundError
            If model_path doesn't exist
        
        Examples
        --------
        >>> exporter = ExportManager()
        >>> onnx_path = exporter.export_model_to_onnx(
        ...     'data_processed/models/lgbm_classifier.pkl',
        ...     model_type='classifier'
        ... )
        """
        try:
            import onnxmltools
            from onnxmltools.convert import convert_lightgbm
            from skl2onnx.common.data_types import FloatTensorType
        except ImportError as e:
            raise ImportError(
                "ONNX export requires onnxmltools and skl2onnx packages. "
                "Install with: pip install onnxmltools skl2onnx"
            ) from e
        
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        model_data = joblib.load(model_path)
        
        # Extract LightGBM model (handle different wrapper formats)
        if hasattr(model_data, 'model'):
            lgbm_model = model_data.model
        elif hasattr(model_data, '_model'):
            lgbm_model = model_data._model
        elif isinstance(model_data, dict) and 'model' in model_data:
            lgbm_model = model_data['model']
        else:
            lgbm_model = model_data
        
        # Get feature names and count
        if hasattr(model_data, 'feature_names'):
            feature_names = model_data.feature_names
        elif hasattr(model_data, 'feature_name_'):
            feature_names = model_data.feature_name_
        elif hasattr(lgbm_model, 'feature_name_'):
            feature_names = lgbm_model.feature_name_
        else:
            # Default feature names
            n_features = lgbm_model.num_feature()
            feature_names = [f'feature_{i}' for i in range(n_features)]
        
        n_features = len(feature_names)
        
        # Define initial types for ONNX conversion
        initial_types = [('input', FloatTensorType([None, n_features]))]
        
        # Convert to ONNX
        logger.info(f"Converting {model_type} model to ONNX format")
        onnx_model = convert_lightgbm(
            lgbm_model,
            initial_types=initial_types,
            target_opset=12
        )
        
        # Generate filename if not provided
        if output_filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f'lgbm_{model_type}_{timestamp}.onnx'
        
        # Ensure .onnx extension
        if not output_filename.endswith('.onnx'):
            output_filename += '.onnx'
        
        # Export path
        export_path = self.output_dir / output_filename
        
        # Save ONNX model
        with open(export_path, 'wb') as f:
            f.write(onnx_model.SerializeToString())
        
        logger.info(f"Exported ONNX model to {export_path}")
        logger.info(f"Model has {n_features} features: {feature_names}")
        
        return export_path
    
    def export_config_to_yaml(
        self,
        config: Dict,
        filename: Optional[str] = None,
        add_timestamp: bool = True
    ) -> Path:
        """
        Export configuration to YAML for backup.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary to export
        filename : str, optional
            Output filename. If None, generates timestamp-based name.
        add_timestamp : bool, default=True
            Whether to add export timestamp to config
        
        Returns
        -------
        Path
            Path to exported YAML file
        
        Examples
        --------
        >>> exporter = ExportManager()
        >>> config = {'features': ['f1', 'f2'], 'thresholds': {'A+': 0.65}}
        >>> path = exporter.export_config_to_yaml(config)
        """
        if config is None:
            raise ValueError("config cannot be None")
        
        # Create export config
        export_config = config.copy()
        
        # Add timestamp if requested
        if add_timestamp:
            export_config['export_metadata'] = {
                'export_timestamp': datetime.now().isoformat(),
                'export_version': '1.0'
            }
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'ml_config_backup_{timestamp}.yaml'
        
        # Ensure .yaml extension
        if not filename.endswith('.yaml') and not filename.endswith('.yml'):
            filename += '.yaml'
        
        # Export path
        export_path = self.output_dir / filename
        
        # Export to YAML
        with open(export_path, 'w') as f:
            yaml.dump(export_config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Exported configuration to {export_path}")
        
        return export_path
    
    def export_feature_importance(
        self,
        feature_importance_df: pd.DataFrame,
        filename: Optional[str] = None
    ) -> Path:
        """
        Export feature importance to CSV.
        
        Parameters
        ----------
        feature_importance_df : pd.DataFrame
            DataFrame with feature importance (columns: feature, importance, rank)
        filename : str, optional
            Output filename
        
        Returns
        -------
        Path
            Path to exported CSV file
        """
        if feature_importance_df is None or len(feature_importance_df) == 0:
            raise ValueError("feature_importance_df cannot be None or empty")
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'feature_importance_{timestamp}.csv'
        
        # Ensure .csv extension
        if not filename.endswith('.csv'):
            filename += '.csv'
        
        # Export path
        export_path = self.output_dir / filename
        
        # Export to CSV
        feature_importance_df.to_csv(export_path, index=False)
        
        logger.info(f"Exported feature importance to {export_path}")
        
        return export_path
    
    def export_metrics_history(
        self,
        metrics_history: Union[Dict, List[Dict]],
        filename: Optional[str] = None,
        format: str = 'json'
    ) -> Path:
        """
        Export metrics history to JSON or CSV.
        
        Parameters
        ----------
        metrics_history : dict or list of dict
            Metrics history data
        filename : str, optional
            Output filename
        format : str, default='json'
            Export format ('json' or 'csv')
        
        Returns
        -------
        Path
            Path to exported file
        """
        if metrics_history is None:
            raise ValueError("metrics_history cannot be None")
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'metrics_history_{timestamp}.{format}'
        
        # Ensure correct extension
        if not filename.endswith(f'.{format}'):
            filename += f'.{format}'
        
        # Export path
        export_path = self.output_dir / filename
        
        # Export based on format
        if format == 'json':
            with open(export_path, 'w') as f:
                json.dump(metrics_history, f, indent=2)
        elif format == 'csv':
            # Convert to DataFrame if needed
            if isinstance(metrics_history, dict):
                df = pd.DataFrame([metrics_history])
            else:
                df = pd.DataFrame(metrics_history)
            df.to_csv(export_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'csv'.")
        
        logger.info(f"Exported metrics history to {export_path}")
        
        return export_path
    
    def create_export_summary(
        self,
        exports: Dict[str, Path],
        filename: Optional[str] = None
    ) -> Path:
        """
        Create a summary file listing all exports.
        
        Parameters
        ----------
        exports : dict
            Dictionary mapping export type to file path
        filename : str, optional
            Output filename
        
        Returns
        -------
        Path
            Path to summary file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'export_summary_{timestamp}.txt'
        
        export_path = self.output_dir / filename
        
        with open(export_path, 'w') as f:
            f.write("ML Prediction Engine - Export Summary\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for export_type, path in exports.items():
                f.write(f"{export_type}:\n")
                f.write(f"  Path: {path}\n")
                if path.exists():
                    size_kb = path.stat().st_size / 1024
                    f.write(f"  Size: {size_kb:.2f} KB\n")
                f.write("\n")
        
        logger.info(f"Created export summary at {export_path}")
        
        return export_path


# Convenience functions

def export_predictions_to_csv(
    predictions_df: pd.DataFrame,
    output_path: Union[str, Path],
    **kwargs
) -> Path:
    """
    Convenience function to export predictions to CSV.
    
    Parameters
    ----------
    predictions_df : pd.DataFrame
        Predictions DataFrame
    output_path : str or Path
        Output file path
    **kwargs
        Additional arguments passed to ExportManager.export_predictions_to_csv()
    
    Returns
    -------
    Path
        Path to exported file
    """
    output_path = Path(output_path)
    output_dir = output_path.parent
    filename = output_path.name
    
    exporter = ExportManager(output_dir=output_dir)
    return exporter.export_predictions_to_csv(predictions_df, filename=filename, **kwargs)


def export_config_to_yaml(
    config: Dict,
    output_path: Union[str, Path],
    **kwargs
) -> Path:
    """
    Convenience function to export config to YAML.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    output_path : str or Path
        Output file path
    **kwargs
        Additional arguments passed to ExportManager.export_config_to_yaml()
    
    Returns
    -------
    Path
        Path to exported file
    """
    output_path = Path(output_path)
    output_dir = output_path.parent
    filename = output_path.name
    
    exporter = ExportManager(output_dir=output_dir)
    return exporter.export_config_to_yaml(config, filename=filename, **kwargs)


def export_model_to_onnx(
    model_path: Union[str, Path],
    output_path: Union[str, Path],
    **kwargs
) -> Path:
    """
    Convenience function to export model to ONNX.
    
    Parameters
    ----------
    model_path : str or Path
        Path to trained model
    output_path : str or Path
        Output file path
    **kwargs
        Additional arguments passed to ExportManager.export_model_to_onnx()
    
    Returns
    -------
    Path
        Path to exported file
    """
    output_path = Path(output_path)
    output_dir = output_path.parent
    filename = output_path.name
    
    exporter = ExportManager(output_dir=output_dir)
    return exporter.export_model_to_onnx(model_path, output_filename=filename, **kwargs)
