"""
Performance Monitor for ML Prediction Engine

This module implements the PerformanceMonitor class for tracking and monitoring
model performance metrics over time.

**Feature: ml-prediction-engine**
**Validates: Requirements 14.1, 14.2, 14.3, 14.4, 14.5**
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging
from sklearn.metrics import roc_auc_score, brier_score_loss, mean_absolute_error

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    Monitor and track ML model performance metrics over time.
    
    This class provides functionality to:
    - Calculate performance metrics (AUC, Brier score, MAE, coverage)
    - Detect performance degradation
    - Track metrics over time with rolling windows
    - Export metrics history
    
    Attributes
    ----------
    baseline_metrics : dict
        Baseline metrics for comparison
    metrics_history : list
        Historical metrics records
    degradation_threshold : float
        Threshold for degradation detection (default 0.10 = 10%)
    
    Examples
    --------
    >>> monitor = PerformanceMonitor()
    >>> metrics = monitor.calculate_metrics(
    ...     y_true_win=y_win,
    ...     y_pred_prob=probs,
    ...     y_true_r=y_r,
    ...     y_pred_p10=p10,
    ...     y_pred_p50=p50,
    ...     y_pred_p90=p90
    ... )
    >>> print(f"AUC: {metrics['auc']:.4f}")
    """
    
    def __init__(
        self,
        baseline_metrics: Optional[Dict] = None,
        degradation_threshold: float = 0.10,
        metrics_history_path: Optional[str] = None
    ):
        """
        Initialize performance monitor.
        
        Parameters
        ----------
        baseline_metrics : dict, optional
            Baseline metrics for comparison
        degradation_threshold : float, default=0.10
            Threshold for degradation detection (10%)
        metrics_history_path : str, optional
            Path to load/save metrics history
        """
        self.baseline_metrics = baseline_metrics or {}
        self.degradation_threshold = degradation_threshold
        self.metrics_history = []
        self.metrics_history_path = metrics_history_path
        
        # Load existing history if path provided
        if metrics_history_path is not None:
            self._load_metrics_history()
        
        logger.info(
            f"PerformanceMonitor initialized with degradation_threshold={degradation_threshold}"
        )
    
    def calculate_metrics(
        self,
        y_true_win: np.ndarray,
        y_pred_prob: np.ndarray,
        y_true_r: np.ndarray,
        y_pred_p10: np.ndarray,
        y_pred_p50: np.ndarray,
        y_pred_p90: np.ndarray,
        y_pred_p10_conf: Optional[np.ndarray] = None,
        y_pred_p90_conf: Optional[np.ndarray] = None,
        timestamp: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics.
        
        Parameters
        ----------
        y_true_win : np.ndarray
            True binary win/loss labels (0/1)
        y_pred_prob : np.ndarray
            Predicted probabilities (0-1)
        y_true_r : np.ndarray
            True R_multiple values
        y_pred_p10 : np.ndarray
            Predicted P10 quantile
        y_pred_p50 : np.ndarray
            Predicted P50 quantile
        y_pred_p90 : np.ndarray
            Predicted P90 quantile
        y_pred_p10_conf : np.ndarray, optional
            Conformal adjusted P10
        y_pred_p90_conf : np.ndarray, optional
            Conformal adjusted P90
        timestamp : str, optional
            Timestamp for this metrics record
        
        Returns
        -------
        dict
            Performance metrics including:
            - auc: Area under ROC curve
            - brier_score: Brier score for probability predictions
            - mae_p10: MAE for P10 predictions
            - mae_p50: MAE for P50 predictions
            - mae_p90: MAE for P90 predictions
            - coverage_raw: Coverage for raw P10-P90 interval
            - coverage_conf: Coverage for conformal adjusted interval (if provided)
            - timestamp: Timestamp of calculation
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        # Calculate AUC
        try:
            auc = roc_auc_score(y_true_win, y_pred_prob)
        except Exception as e:
            logger.warning(f"Failed to calculate AUC: {e}")
            auc = np.nan
        
        # Calculate Brier score
        try:
            brier = brier_score_loss(y_true_win, y_pred_prob)
        except Exception as e:
            logger.warning(f"Failed to calculate Brier score: {e}")
            brier = np.nan
        
        # Calculate MAE per quantile
        mae_p10 = mean_absolute_error(y_true_r, y_pred_p10)
        mae_p50 = mean_absolute_error(y_true_r, y_pred_p50)
        mae_p90 = mean_absolute_error(y_true_r, y_pred_p90)
        
        # Calculate coverage for raw interval
        coverage_raw = self._calculate_coverage(y_true_r, y_pred_p10, y_pred_p90)
        
        # Calculate coverage for conformal interval if provided
        coverage_conf = None
        if y_pred_p10_conf is not None and y_pred_p90_conf is not None:
            coverage_conf = self._calculate_coverage(
                y_true_r, y_pred_p10_conf, y_pred_p90_conf
            )
        
        # Compile metrics
        metrics = {
            'auc': float(auc),
            'brier_score': float(brier),
            'mae_p10': float(mae_p10),
            'mae_p50': float(mae_p50),
            'mae_p90': float(mae_p90),
            'coverage_raw': float(coverage_raw),
            'coverage_conf': float(coverage_conf) if coverage_conf is not None else None,
            'timestamp': timestamp,
            'n_samples': len(y_true_win)
        }
        
        logger.info(
            f"Calculated metrics: AUC={auc:.4f}, Brier={brier:.4f}, "
            f"Coverage_raw={coverage_raw:.4f}"
        )
        
        return metrics
    
    def _calculate_coverage(
        self,
        y_true: np.ndarray,
        y_pred_lower: np.ndarray,
        y_pred_upper: np.ndarray
    ) -> float:
        """
        Calculate coverage: percentage of true values within predicted interval.
        
        Parameters
        ----------
        y_true : np.ndarray
            True values
        y_pred_lower : np.ndarray
            Lower bound predictions
        y_pred_upper : np.ndarray
            Upper bound predictions
        
        Returns
        -------
        float
            Coverage percentage (0-1)
        """
        within_interval = (y_true >= y_pred_lower) & (y_true <= y_pred_upper)
        coverage = np.mean(within_interval)
        return coverage
    
    def detect_degradation(
        self,
        current_metrics: Dict[str, float],
        baseline_metrics: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Detect performance degradation by comparing to baseline.
        
        Degradation is detected when:
        - AUC decreases by > threshold
        - Brier score increases by > threshold
        - MAE increases by > threshold
        - Coverage deviates from target by > threshold
        
        Parameters
        ----------
        current_metrics : dict
            Current performance metrics
        baseline_metrics : dict, optional
            Baseline metrics for comparison (uses self.baseline_metrics if not provided)
        
        Returns
        -------
        dict
            Degradation report with:
            - is_degraded: bool, whether degradation detected
            - degraded_metrics: list of metric names that degraded
            - degradation_details: dict with details per metric
            - alert_message: str, alert message if degraded
        """
        if baseline_metrics is None:
            baseline_metrics = self.baseline_metrics
        
        if not baseline_metrics:
            logger.warning("No baseline metrics available for degradation detection")
            return {
                'is_degraded': False,
                'degraded_metrics': [],
                'degradation_details': {},
                'alert_message': None
            }
        
        degraded_metrics = []
        degradation_details = {}
        
        # Check AUC (lower is worse)
        if 'auc' in baseline_metrics and 'auc' in current_metrics:
            baseline_auc = baseline_metrics['auc']
            current_auc = current_metrics['auc']
            
            if not np.isnan(baseline_auc) and not np.isnan(current_auc):
                degradation = (baseline_auc - current_auc) / baseline_auc
                
                if degradation > self.degradation_threshold:
                    degraded_metrics.append('auc')
                    degradation_details['auc'] = {
                        'baseline': baseline_auc,
                        'current': current_auc,
                        'degradation_pct': degradation * 100
                    }
        
        # Check Brier score (higher is worse)
        if 'brier_score' in baseline_metrics and 'brier_score' in current_metrics:
            baseline_brier = baseline_metrics['brier_score']
            current_brier = current_metrics['brier_score']
            
            if not np.isnan(baseline_brier) and not np.isnan(current_brier):
                degradation = (current_brier - baseline_brier) / baseline_brier
                
                if degradation > self.degradation_threshold:
                    degraded_metrics.append('brier_score')
                    degradation_details['brier_score'] = {
                        'baseline': baseline_brier,
                        'current': current_brier,
                        'degradation_pct': degradation * 100
                    }
        
        # Check MAE per quantile (higher is worse)
        for quantile in ['p10', 'p50', 'p90']:
            mae_key = f'mae_{quantile}'
            if mae_key in baseline_metrics and mae_key in current_metrics:
                baseline_mae = baseline_metrics[mae_key]
                current_mae = current_metrics[mae_key]
                
                degradation = (current_mae - baseline_mae) / baseline_mae
                
                if degradation > self.degradation_threshold:
                    degraded_metrics.append(mae_key)
                    degradation_details[mae_key] = {
                        'baseline': baseline_mae,
                        'current': current_mae,
                        'degradation_pct': degradation * 100
                    }
        
        # Check coverage (deviation from target)
        for coverage_key in ['coverage_raw', 'coverage_conf']:
            if coverage_key in baseline_metrics and coverage_key in current_metrics:
                baseline_cov = baseline_metrics[coverage_key]
                current_cov = current_metrics[coverage_key]
                
                if baseline_cov is not None and current_cov is not None:
                    # Coverage deviation (absolute)
                    deviation = abs(current_cov - baseline_cov)
                    
                    if deviation > self.degradation_threshold:
                        degraded_metrics.append(coverage_key)
                        degradation_details[coverage_key] = {
                            'baseline': baseline_cov,
                            'current': current_cov,
                            'deviation': deviation
                        }
        
        # Generate alert message
        is_degraded = len(degraded_metrics) > 0
        alert_message = None
        
        if is_degraded:
            alert_message = (
                f"Model performance degraded! "
                f"Degraded metrics: {', '.join(degraded_metrics)}. "
                f"Consider retraining the model."
            )
            logger.warning(alert_message)
        
        return {
            'is_degraded': is_degraded,
            'degraded_metrics': degraded_metrics,
            'degradation_details': degradation_details,
            'alert_message': alert_message
        }
    
    def add_metrics_record(
        self,
        metrics: Dict[str, Any],
        save_to_file: bool = True
    ):
        """
        Add metrics record to history.
        
        Parameters
        ----------
        metrics : dict
            Metrics dictionary to add
        save_to_file : bool, default=True
            Whether to save updated history to file
        """
        self.metrics_history.append(metrics)
        
        if save_to_file and self.metrics_history_path is not None:
            self._save_metrics_history()
        
        logger.info(f"Added metrics record to history (total: {len(self.metrics_history)})")
    
    def get_metrics_history(
        self,
        metric_names: Optional[List[str]] = None,
        window_size: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get metrics history as DataFrame.
        
        Parameters
        ----------
        metric_names : list of str, optional
            Specific metrics to include (all if None)
        window_size : int, optional
            Only return last N records (all if None)
        
        Returns
        -------
        pd.DataFrame
            Metrics history with timestamp index
        """
        if not self.metrics_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.metrics_history)
        
        # Set timestamp as index
        if 'timestamp' in df.columns:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
                df = df.set_index('timestamp')
            except Exception as e:
                logger.warning(f"Failed to parse timestamps: {e}")
        
        # Filter metrics if specified
        if metric_names is not None:
            available_cols = [col for col in metric_names if col in df.columns]
            df = df[available_cols]
        
        # Apply window if specified
        if window_size is not None and len(df) > window_size:
            df = df.tail(window_size)
        
        return df
    
    def calculate_rolling_metrics(
        self,
        window_size: int = 10
    ) -> pd.DataFrame:
        """
        Calculate rolling window statistics for metrics.
        
        Parameters
        ----------
        window_size : int, default=10
            Size of rolling window
        
        Returns
        -------
        pd.DataFrame
            Rolling statistics (mean, std) for each metric
        """
        df = self.get_metrics_history()
        
        if df.empty or len(df) < window_size:
            logger.warning(
                f"Insufficient data for rolling window: {len(df)} < {window_size}"
            )
            return pd.DataFrame()
        
        # Select only numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_numeric = df[numeric_cols]
        
        if df_numeric.empty:
            logger.warning("No numeric columns found for rolling metrics")
            return pd.DataFrame()
        
        # Calculate rolling mean and std
        rolling_mean = df_numeric.rolling(window=window_size).mean()
        rolling_std = df_numeric.rolling(window=window_size).std()
        
        # Combine into single DataFrame
        result = pd.DataFrame()
        for col in numeric_cols:
            result[f'{col}_mean'] = rolling_mean[col]
            result[f'{col}_std'] = rolling_std[col]
        
        return result
    
    def export_metrics_history(
        self,
        output_path: str,
        format: str = 'csv'
    ):
        """
        Export metrics history to file.
        
        Parameters
        ----------
        output_path : str
            Path to output file
        format : str, default='csv'
            Output format ('csv' or 'json')
        """
        df = self.get_metrics_history()
        
        if df.empty:
            logger.warning("No metrics history to export")
            return
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'csv':
            df.to_csv(output_path)
            logger.info(f"Exported metrics history to CSV: {output_path}")
        elif format == 'json':
            df.to_json(output_path, orient='records', date_format='iso', indent=2)
            logger.info(f"Exported metrics history to JSON: {output_path}")
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def set_baseline_metrics(
        self,
        metrics: Dict[str, float]
    ):
        """
        Set baseline metrics for degradation detection.
        
        Parameters
        ----------
        metrics : dict
            Baseline metrics dictionary
        """
        self.baseline_metrics = metrics
        logger.info("Updated baseline metrics")
    
    def _load_metrics_history(self):
        """Load metrics history from file."""
        if self.metrics_history_path is None:
            return
        
        history_path = Path(self.metrics_history_path)
        
        if not history_path.exists():
            logger.info(f"No existing metrics history at {history_path}")
            return
        
        try:
            with open(history_path, 'r') as f:
                self.metrics_history = json.load(f)
            
            logger.info(
                f"Loaded {len(self.metrics_history)} metrics records from {history_path}"
            )
        except Exception as e:
            logger.error(f"Failed to load metrics history: {e}")
            self.metrics_history = []
    
    def _save_metrics_history(self):
        """Save metrics history to file."""
        if self.metrics_history_path is None:
            return
        
        history_path = Path(self.metrics_history_path)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(history_path, 'w') as f:
                json.dump(self.metrics_history, f, indent=2, default=str)
            
            logger.info(f"Saved metrics history to {history_path}")
        except Exception as e:
            logger.error(f"Failed to save metrics history: {e}")


def monitor_model_performance(
    y_true_win: np.ndarray,
    y_pred_prob: np.ndarray,
    y_true_r: np.ndarray,
    y_pred_p10: np.ndarray,
    y_pred_p50: np.ndarray,
    y_pred_p90: np.ndarray,
    baseline_metrics: Optional[Dict] = None,
    degradation_threshold: float = 0.10
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Convenience function to calculate metrics and detect degradation.
    
    Parameters
    ----------
    y_true_win : np.ndarray
        True binary win/loss labels
    y_pred_prob : np.ndarray
        Predicted probabilities
    y_true_r : np.ndarray
        True R_multiple values
    y_pred_p10 : np.ndarray
        Predicted P10 quantile
    y_pred_p50 : np.ndarray
        Predicted P50 quantile
    y_pred_p90 : np.ndarray
        Predicted P90 quantile
    baseline_metrics : dict, optional
        Baseline metrics for comparison
    degradation_threshold : float, default=0.10
        Threshold for degradation detection
    
    Returns
    -------
    tuple
        (metrics, degradation_report)
    
    Examples
    --------
    >>> metrics, degradation = monitor_model_performance(
    ...     y_true_win=y_win,
    ...     y_pred_prob=probs,
    ...     y_true_r=y_r,
    ...     y_pred_p10=p10,
    ...     y_pred_p50=p50,
    ...     y_pred_p90=p90,
    ...     baseline_metrics=baseline
    ... )
    >>> if degradation['is_degraded']:
    ...     print(degradation['alert_message'])
    """
    monitor = PerformanceMonitor(
        baseline_metrics=baseline_metrics,
        degradation_threshold=degradation_threshold
    )
    
    metrics = monitor.calculate_metrics(
        y_true_win=y_true_win,
        y_pred_prob=y_pred_prob,
        y_true_r=y_true_r,
        y_pred_p10=y_pred_p10,
        y_pred_p50=y_pred_p50,
        y_pred_p90=y_pred_p90
    )
    
    degradation_report = monitor.detect_degradation(metrics)
    
    return metrics, degradation_report
