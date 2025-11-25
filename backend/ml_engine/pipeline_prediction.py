"""
Prediction Pipeline - Orchestration of 4 ML Components

This module implements the prediction pipeline that orchestrates the 4 ML
components (LightGBM Classifier, Isotonic Calibration, Quantile Regression,
Conformal Prediction) to provide end-to-end predictions.

**Feature: ml-prediction-engine**
**Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5**
"""

import numpy as np
import pandas as pd
import time
import yaml
from pathlib import Path
from typing import Optional, Dict, Union, Tuple, List
import logging
from functools import lru_cache
import threading

# Import ML components
from .lgbm_classifier import LGBMClassifierWrapper
from .calibration_isotonic import IsotonicCalibrator
from .lgbm_quantile_regressor import QuantileRegressorEnsemble
from .conformal_engine import ConformalEngine
from .error_handling import (
    ErrorHandler,
    ModelNotFoundError,
    FeatureMismatchError,
    validate_model_files
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Global model cache for singleton pattern
_model_cache = {}
_cache_lock = threading.Lock()


class PredictionPipeline:
    """
    Orchestration of 4 ML components for end-to-end prediction.
    
    This class loads and coordinates the 4 trained ML components to provide
    complete predictions including calibrated probabilities, quantile predictions,
    and conformal intervals.
    
    Attributes
    ----------
    classifier : LGBMClassifierWrapper
        Binary win/loss classifier
    calibrator : IsotonicCalibrator
        Probability calibrator
    quantile_ensemble : QuantileRegressorEnsemble
        Quantile regression ensemble (P10, P50, P90)
    conformal_engine : ConformalEngine
        Conformal prediction engine
    feature_names : list
        Names of features expected for prediction
    is_loaded : bool
        Whether all models have been loaded
    config : dict
        Configuration loaded from YAML
    
    Examples
    --------
    >>> pipeline = PredictionPipeline()
    >>> pipeline.load_models('data_processed/models')
    >>> result = pipeline.predict_for_sample({'feature1': 0.5, 'feature2': 1.2})
    >>> print(result['prob_win_calibrated'], result['R_P50_raw'])
    >>> 
    >>> # Batch prediction
    >>> results_df = pipeline.predict_for_batch(features_df)
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None, use_cache: bool = True):
        """
        Initialize prediction pipeline.
        
        Parameters
        ----------
        config_path : str or Path, optional
            Path to configuration YAML file. If None, uses default config.
        use_cache : bool, default=True
            Whether to use global model cache for performance
        """
        # Initialize components (lazy loading - will be loaded on demand)
        self.classifier = None
        self.calibrator = None
        self.quantile_ensemble = None
        self.conformal_engine = None
        
        self.feature_names = None
        self.is_loaded = False
        self.use_cache = use_cache
        self._model_dir = None
        
        # Initialize error handler
        self.error_handler = ErrorHandler(
            log_errors=True,
            raise_on_critical=False,
            memory_threshold_pct=90.0
        )
        
        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / 'config' / 'ml_prediction_config.yaml'
        
        self.config = self._load_config(config_path)
        
        # Extract thresholds from config
        self.thresholds = self.config.get('thresholds', {})
        
        # Performance tracking
        self._prediction_count = 0
        self._total_prediction_time = 0.0
    
    def _load_config(self, config_path: Union[str, Path]) -> Dict:
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}. Using defaults.")
            return self._get_default_config()
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'thresholds': {
                'quality_A_plus': {'prob_win_min': 0.65, 'R_P50_min': 1.5},
                'quality_A': {'prob_win_min': 0.55, 'R_P50_min': 1.0},
                'quality_B': {'prob_win_min': 0.45, 'R_P50_min': 0.5}
            },
            'paths': {
                'model_dir': 'data_processed/models',
                'classifier_model': 'lgbm_classifier.pkl',
                'calibrator_model': 'isotonic_calibrator.pkl',
                'quantile_p10_model': 'lgbm_quantile_p10.pkl',
                'quantile_p50_model': 'lgbm_quantile_p50.pkl',
                'quantile_p90_model': 'lgbm_quantile_p90.pkl',
                'conformal_meta': 'conformal_meta.json'
            }
        }
    
    def load_models(self, model_dir: Union[str, Path], force_reload: bool = False) -> bool:
        """
        Load all 4 components from model directory with caching support.
        
        Uses global cache to avoid reloading models multiple times across
        different pipeline instances. This significantly improves performance
        when multiple pipelines are created.
        
        Parameters
        ----------
        model_dir : str or Path
            Path to folder containing trained models
        force_reload : bool, default=False
            Force reload models even if cached
        
        Returns
        -------
        bool
            True if all models loaded successfully, False otherwise
        
        Raises
        ------
        ModelNotFoundError
            If any required model file is not found
        FeatureMismatchError
            If models have inconsistent feature names
        """
        model_dir = Path(model_dir)
        self._model_dir = model_dir
        
        # Check if model directory exists
        if not model_dir.exists():
            error_info = self.error_handler.handle_model_not_found(model_dir)
            raise ModelNotFoundError(error_info['error_message'])
        
        # Get model paths from config
        paths = self.config.get('paths', {})
        
        # Validate all required model files exist
        required_files = [
            paths.get('classifier_model', 'lgbm_classifier.pkl'),
            paths.get('calibrator_model', 'isotonic_calibrator.pkl'),
            paths.get('quantile_p10_model', 'lgbm_quantile_p10.pkl'),
            paths.get('quantile_p50_model', 'lgbm_quantile_p50.pkl'),
            paths.get('quantile_p90_model', 'lgbm_quantile_p90.pkl'),
            paths.get('conformal_meta', 'conformal_meta.json')
        ]
        
        all_exist, missing_files = validate_model_files(model_dir, required_files)
        
        if not all_exist:
            error_msg = (
                f"Missing model files: {missing_files}\n"
                f"Please train models first using the 'Train Models' button."
            )
            logger.error(error_msg)
            raise ModelNotFoundError(error_msg)
        
        # Generate cache key based on model directory
        cache_key = str(model_dir.resolve())
        
        # Check global cache first (if enabled and not forcing reload)
        if self.use_cache and not force_reload and cache_key in _model_cache:
            with _cache_lock:
                if cache_key in _model_cache:
                    logger.info(f"Loading models from cache for {model_dir}")
                    cached_models = _model_cache[cache_key]
                    
                    # Use cached models (shallow copy to share the same model objects)
                    self.classifier = cached_models['classifier']
                    self.calibrator = cached_models['calibrator']
                    self.quantile_ensemble = cached_models['quantile_ensemble']
                    self.conformal_engine = cached_models['conformal_engine']
                    self.feature_names = cached_models['feature_names']
                    
                    self.is_loaded = True
                    logger.info("Models loaded from cache successfully")
                    return True
        
        try:
            # Initialize components for loading
            self.classifier = LGBMClassifierWrapper()
            self.calibrator = IsotonicCalibrator()
            self.quantile_ensemble = QuantileRegressorEnsemble()
            self.conformal_engine = ConformalEngine()
            
            # Load classifier
            classifier_path = model_dir / paths.get('classifier_model', 'lgbm_classifier.pkl')
            logger.info(f"Loading classifier from {classifier_path}")
            
            exists, error_msg = self.error_handler.check_model_file_exists(classifier_path)
            if not exists:
                raise ModelNotFoundError(error_msg)
            
            try:
                self.classifier.load(classifier_path)
            except Exception as e:
                error_info = self.error_handler.handle_model_load_error(classifier_path, e)
                raise ModelNotFoundError(error_info['error_message']) from e
            
            # Load calibrator
            calibrator_path = model_dir / paths.get('calibrator_model', 'isotonic_calibrator.pkl')
            logger.info(f"Loading calibrator from {calibrator_path}")
            
            exists, error_msg = self.error_handler.check_model_file_exists(calibrator_path)
            if not exists:
                raise ModelNotFoundError(error_msg)
            
            try:
                self.calibrator.load(calibrator_path)
            except Exception as e:
                error_info = self.error_handler.handle_model_load_error(calibrator_path, e)
                raise ModelNotFoundError(error_info['error_message']) from e
            
            # Load quantile ensemble
            quantile_prefix = model_dir / 'lgbm_quantile'
            logger.info(f"Loading quantile ensemble from {quantile_prefix}")
            
            try:
                self.quantile_ensemble.load(quantile_prefix)
            except Exception as e:
                error_info = self.error_handler.handle_model_load_error(quantile_prefix, e)
                raise ModelNotFoundError(error_info['error_message']) from e
            
            # Load conformal engine
            conformal_path = model_dir / paths.get('conformal_meta', 'conformal_meta.json')
            logger.info(f"Loading conformal engine from {conformal_path}")
            
            exists, error_msg = self.error_handler.check_model_file_exists(conformal_path)
            if not exists:
                raise ModelNotFoundError(error_msg)
            
            try:
                self.conformal_engine.load(conformal_path)
            except Exception as e:
                error_info = self.error_handler.handle_model_load_error(conformal_path, e)
                raise ModelNotFoundError(error_info['error_message']) from e
            
            # Validate feature consistency
            self.feature_names = self.classifier.feature_names
            
            if self.quantile_ensemble.feature_names != self.feature_names:
                error_msg = (
                    f"Feature mismatch between classifier and quantile ensemble.\n"
                    f"Classifier features: {self.feature_names}\n"
                    f"Quantile features: {self.quantile_ensemble.feature_names}"
                )
                logger.error(error_msg)
                raise FeatureMismatchError(error_msg)
            
            self.is_loaded = True
            
            # Cache models for future use
            if self.use_cache:
                with _cache_lock:
                    _model_cache[cache_key] = {
                        'classifier': self.classifier,
                        'calibrator': self.calibrator,
                        'quantile_ensemble': self.quantile_ensemble,
                        'conformal_engine': self.conformal_engine,
                        'feature_names': self.feature_names
                    }
                    logger.info(f"Models cached for {model_dir}")
            
            logger.info("All models loaded successfully")
            
            return True
            
        except (ModelNotFoundError, FeatureMismatchError):
            self.is_loaded = False
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading models: {e}")
            self.is_loaded = False
            raise
    
    def predict_for_sample(
        self,
        feature_row: Union[Dict, pd.Series, pd.DataFrame]
    ) -> Dict:
        """
        Predict for single sample.
        
        Runs all 4 components sequentially:
        1. Classifier → raw probability
        2. Calibrator → calibrated probability
        3. Quantile ensemble → P10, P50, P90
        4. Conformal engine → adjusted intervals
        
        Parameters
        ----------
        feature_row : dict, pd.Series, or pd.DataFrame
            Features for 1 sample. If DataFrame, uses first row.
        
        Returns
        -------
        dict
            Prediction results with keys:
            - prob_win_raw: Raw probability from classifier
            - prob_win_calibrated: Calibrated probability
            - R_P10_raw: Raw P10 prediction
            - R_P50_raw: Raw P50 prediction (median)
            - R_P90_raw: Raw P90 prediction
            - R_P10_conf: Conformal adjusted P10
            - R_P90_conf: Conformal adjusted P90
            - skewness: Distribution skewness
            - quality_label: Setup quality (A+/A/B/C)
            - recommendation: Trade recommendation (TRADE/SKIP)
            - execution_time_ms: Total execution time
            - component_times_ms: Execution time per component
        
        Raises
        ------
        ValueError
            If models are not loaded
            If features are missing or invalid
        """
        if not self.is_loaded:
            raise ValueError(
                "Models must be loaded before prediction. Call load_models() first."
            )
        
        # Start timing
        start_time = time.time()
        component_times = {}
        
        try:
            # Convert input to DataFrame
            if isinstance(feature_row, dict):
                X = pd.DataFrame([feature_row])
            elif isinstance(feature_row, pd.Series):
                X = pd.DataFrame([feature_row])
            elif isinstance(feature_row, pd.DataFrame):
                X = feature_row.iloc[[0]]
            else:
                raise ValueError(
                    f"feature_row must be dict, Series, or DataFrame, got {type(feature_row)}"
                )
            
            # Validate features are present
            is_valid, validation_report = self.error_handler.validate_features(
                X, self.feature_names
            )
            
            if not is_valid:
                # Try to handle missing features
                logger.warning(f"Missing features detected: {validation_report['missing_features']}")
                X = self.error_handler.handle_missing_features(
                    X, self.feature_names, imputation_values=None
                )
            
            # Ensure correct feature order
            X = X[self.feature_names]
            
            # Component 1: Classifier
            t0 = time.time()
            prob_win_raw = self.classifier.predict_proba(X)[0]
            component_times['classifier_ms'] = (time.time() - t0) * 1000
            
            # Component 2: Calibrator
            t0 = time.time()
            prob_win_calibrated = self.calibrator.transform(np.array([prob_win_raw]))[0]
            component_times['calibrator_ms'] = (time.time() - t0) * 1000
            
            # Component 3: Quantile Ensemble
            t0 = time.time()
            quantile_preds = self.quantile_ensemble.predict(X)
            R_P10_raw = quantile_preds['p10'][0]
            R_P50_raw = quantile_preds['p50'][0]
            R_P90_raw = quantile_preds['p90'][0]
            component_times['quantile_ms'] = (time.time() - t0) * 1000
            
            # Component 4: Conformal Engine
            t0 = time.time()
            R_P10_conf, R_P90_conf = self.conformal_engine.adjust_intervals(
                np.array([R_P10_raw]),
                np.array([R_P90_raw])
            )
            R_P10_conf = R_P10_conf[0]
            R_P90_conf = R_P90_conf[0]
            component_times['conformal_ms'] = (time.time() - t0) * 1000
            
            # Calculate skewness
            skewness = self.quantile_ensemble.calculate_skewness(quantile_preds)[0]
            
            # Categorize setup
            quality_label, recommendation = self.categorize_setup(
                prob_win_calibrated, R_P50_raw, R_P10_conf, R_P90_conf
            )
            
            # Total execution time
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Update performance tracking
            self._prediction_count += 1
            self._total_prediction_time += (execution_time_ms / 1000)
            
            # Build result
            result = {
                'prob_win_raw': float(prob_win_raw),
                'prob_win_calibrated': float(prob_win_calibrated),
                'R_P10_raw': float(R_P10_raw),
                'R_P50_raw': float(R_P50_raw),
                'R_P90_raw': float(R_P90_raw),
                'R_P10_conf': float(R_P10_conf),
                'R_P90_conf': float(R_P90_conf),
                'skewness': float(skewness),
                'quality_label': quality_label,
                'recommendation': recommendation,
                'execution_time_ms': float(execution_time_ms),
                'component_times_ms': component_times
            }
            
            return result
            
        except MemoryError as e:
            # Handle memory errors specifically
            error_info = self.error_handler.handle_memory_error(e, batch_size=1)
            logger.error(f"Memory error: {error_info['error_message']}")
            return self._get_nan_result(error_info['error_message'])
        except Exception as e:
            # Handle all other prediction errors
            logger.error(f"Prediction failed: {e}")
            return self.error_handler.handle_prediction_error(e, return_nan=True)
    
    def predict_for_batch(
        self,
        feature_df: pd.DataFrame,
        chunk_size: Optional[int] = None,
        show_progress: bool = False
    ) -> pd.DataFrame:
        """
        Predict for multiple samples with automatic chunking for large batches.
        
        Automatically chunks large batches (> 100K samples) to avoid memory issues.
        Uses vectorized operations for maximum performance.
        
        Parameters
        ----------
        feature_df : pd.DataFrame
            Features for N samples
        chunk_size : int, optional
            Number of samples per chunk. If None, automatically determined
            based on batch size (10K for < 100K samples, 5K for larger)
        show_progress : bool, default=False
            Whether to log progress for each chunk
        
        Returns
        -------
        pd.DataFrame
            Original features + prediction columns:
            - prob_win_raw
            - prob_win_calibrated
            - R_P10_raw
            - R_P50_raw
            - R_P90_raw
            - R_P10_conf
            - R_P90_conf
            - skewness
            - quality_label
            - recommendation
        
        Raises
        ------
        ValueError
            If models are not loaded
            If features are missing or invalid
        """
        if not self.is_loaded:
            raise ValueError(
                "Models must be loaded before prediction. Call load_models() first."
            )
        
        if len(feature_df) == 0:
            raise ValueError("feature_df cannot be empty")
        
        # Determine if chunking is needed
        n_samples = len(feature_df)
        
        # Auto-determine chunk size if not provided
        if chunk_size is None:
            if n_samples > 100000:
                chunk_size = 5000
                logger.info(f"Large batch detected ({n_samples} samples), using chunk size {chunk_size}")
            elif n_samples > 10000:
                chunk_size = 10000
            else:
                # No chunking needed for small batches
                chunk_size = n_samples
        
        # If batch is small enough, process directly
        if n_samples <= chunk_size:
            return self._predict_batch_vectorized(feature_df)
        
        # Process in chunks for large batches
        logger.info(f"Processing {n_samples} samples in chunks of {chunk_size}")
        
        results_list = []
        n_chunks = (n_samples + chunk_size - 1) // chunk_size
        
        for i in range(0, n_samples, chunk_size):
            chunk_idx = i // chunk_size + 1
            end_idx = min(i + chunk_size, n_samples)
            chunk = feature_df.iloc[i:end_idx]
            
            if show_progress:
                logger.info(f"Processing chunk {chunk_idx}/{n_chunks} ({len(chunk)} samples)")
            
            try:
                chunk_results = self._predict_batch_vectorized(chunk)
                results_list.append(chunk_results)
            except MemoryError as e:
                # If even chunked processing fails, suggest smaller chunks
                error_info = self.error_handler.handle_memory_error(e, batch_size=len(chunk))
                logger.error(f"Memory error in chunk {chunk_idx}: {error_info['error_message']}")
                logger.error(f"Suggestion: {error_info['suggestion']}")
                raise MemoryError(
                    f"Memory error processing chunk {chunk_idx}. "
                    f"Try reducing chunk_size below {chunk_size}"
                ) from e
        
        # Combine all chunks
        logger.info(f"Combining {len(results_list)} chunks")
        results = pd.concat(results_list, ignore_index=True)
        
        logger.info(f"Batch prediction completed for {len(results)} samples")
        
        return results
    
    def _predict_batch_vectorized(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Internal method for vectorized batch prediction without chunking.
        
        This method performs all operations in vectorized form for maximum
        performance. It's used internally by predict_for_batch.
        
        Parameters
        ----------
        feature_df : pd.DataFrame
            Features for N samples (should be reasonably sized)
        
        Returns
        -------
        pd.DataFrame
            Results with prediction columns
        """
        try:
            # Check memory before processing
            data_size_mb = feature_df.memory_usage(deep=True).sum() / (1024 * 1024)
            is_safe, memory_info = self.error_handler.check_memory_usage(data_size_mb)
            
            if not is_safe:
                logger.warning(
                    f"Memory warning: {memory_info.get('error_message', 'High memory usage')}"
                )
                logger.warning(memory_info.get('suggestion', ''))
            
            # Validate features are present
            is_valid, validation_report = self.error_handler.validate_features(
                feature_df, self.feature_names
            )
            
            if not is_valid:
                # Try to handle missing features
                logger.warning(f"Missing features detected: {validation_report['missing_features']}")
                feature_df = self.error_handler.handle_missing_features(
                    feature_df, self.feature_names, imputation_values=None
                )
            
            # Ensure correct feature order (vectorized)
            X = feature_df[self.feature_names].copy()
            
            # Component 1: Classifier (vectorized)
            prob_win_raw = self.classifier.predict_proba(X)
            
            # Component 2: Calibrator (vectorized)
            prob_win_calibrated = self.calibrator.transform(prob_win_raw)
            
            # Component 3: Quantile Ensemble (vectorized)
            quantile_preds = self.quantile_ensemble.predict(X)
            R_P10_raw = quantile_preds['p10']
            R_P50_raw = quantile_preds['p50']
            R_P90_raw = quantile_preds['p90']
            
            # Component 4: Conformal Engine (vectorized)
            R_P10_conf, R_P90_conf = self.conformal_engine.adjust_intervals(
                R_P10_raw, R_P90_raw
            )
            
            # Calculate skewness (vectorized)
            skewness = self.quantile_ensemble.calculate_skewness(quantile_preds)
            
            # Categorize all setups (vectorized using numpy)
            quality_labels, recommendations = self._categorize_batch_vectorized(
                prob_win_calibrated, R_P50_raw, R_P10_conf, R_P90_conf
            )
            
            # Build results DataFrame (vectorized assignment)
            results = feature_df.copy()
            results['prob_win_raw'] = prob_win_raw
            results['prob_win_calibrated'] = prob_win_calibrated
            results['R_P10_raw'] = R_P10_raw
            results['R_P50_raw'] = R_P50_raw
            results['R_P90_raw'] = R_P90_raw
            results['R_P10_conf'] = R_P10_conf
            results['R_P90_conf'] = R_P90_conf
            results['skewness'] = skewness
            results['quality_label'] = quality_labels
            results['recommendation'] = recommendations
            
            return results
            
        except MemoryError as e:
            # Handle memory errors with batch size suggestion
            error_info = self.error_handler.handle_memory_error(e, batch_size=len(feature_df))
            logger.error(f"Memory error: {error_info['error_message']}")
            logger.error(f"Suggestion: {error_info['suggestion']}")
            raise MemoryError(error_info['error_message']) from e
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise
    
    def _categorize_batch_vectorized(
        self,
        prob_win: np.ndarray,
        R_P50: np.ndarray,
        R_P10: np.ndarray,
        R_P90: np.ndarray
    ) -> Tuple[List[str], List[str]]:
        """
        Vectorized batch categorization for performance.
        
        Uses numpy boolean indexing for fast categorization of large batches.
        
        Parameters
        ----------
        prob_win : np.ndarray
            Calibrated win probabilities
        R_P50 : np.ndarray
            Median R_multiple predictions
        R_P10 : np.ndarray
            P10 conformal predictions
        R_P90 : np.ndarray
            P90 conformal predictions
        
        Returns
        -------
        tuple
            (quality_labels, recommendations) as lists
        """
        # Get thresholds from config
        A_plus_thresholds = self.thresholds.get('quality_A_plus', {})
        A_thresholds = self.thresholds.get('quality_A', {})
        B_thresholds = self.thresholds.get('quality_B', {})
        
        prob_A_plus = A_plus_thresholds.get('prob_win_min', 0.65)
        R_A_plus = A_plus_thresholds.get('R_P50_min', 1.5)
        
        prob_A = A_thresholds.get('prob_win_min', 0.55)
        R_A = A_thresholds.get('R_P50_min', 1.0)
        
        prob_B = B_thresholds.get('prob_win_min', 0.45)
        R_B = B_thresholds.get('R_P50_min', 0.5)
        
        # Initialize arrays
        n = len(prob_win)
        quality_labels = np.full(n, 'C', dtype=object)
        recommendations = np.full(n, 'SKIP', dtype=object)
        
        # Vectorized categorization using boolean masks
        # A+ category
        mask_A_plus = (prob_win > prob_A_plus) & (R_P50 > R_A_plus)
        quality_labels[mask_A_plus] = 'A+'
        recommendations[mask_A_plus] = 'TRADE'
        
        # A category (only if not already A+)
        mask_A = (~mask_A_plus) & (prob_win > prob_A) & (R_P50 > R_A)
        quality_labels[mask_A] = 'A'
        recommendations[mask_A] = 'TRADE'
        
        # B category (only if not already A+ or A)
        mask_B = (~mask_A_plus) & (~mask_A) & (prob_win > prob_B) & (R_P50 > R_B)
        quality_labels[mask_B] = 'B'
        # recommendations already 'SKIP' for B
        
        # C category is default (already set)
        
        return quality_labels.tolist(), recommendations.tolist()
    
    def categorize_setup(
        self,
        prob_win: float,
        R_P50: float,
        R_P10: float,
        R_P90: float
    ) -> Tuple[str, str]:
        """
        Categorize setup quality based on thresholds.
        
        Rules (configurable via config):
        - A+: prob_win > 0.65 AND R_P50 > 1.5
        - A:  prob_win > 0.55 AND R_P50 > 1.0
        - B:  prob_win > 0.45 AND R_P50 > 0.5
        - C:  otherwise
        
        Recommendation:
        - TRADE: for A+ and A
        - SKIP: for B and C
        
        Parameters
        ----------
        prob_win : float
            Calibrated win probability
        R_P50 : float
            Median R_multiple prediction
        R_P10 : float
            P10 conformal prediction (not used in current rules)
        R_P90 : float
            P90 conformal prediction (not used in current rules)
        
        Returns
        -------
        tuple
            (quality_label, recommendation)
            quality_label: str in {'A+', 'A', 'B', 'C'}
            recommendation: str in {'TRADE', 'SKIP'}
        """
        # Get thresholds from config
        A_plus_thresholds = self.thresholds.get('quality_A_plus', {})
        A_thresholds = self.thresholds.get('quality_A', {})
        B_thresholds = self.thresholds.get('quality_B', {})
        
        prob_A_plus = A_plus_thresholds.get('prob_win_min', 0.65)
        R_A_plus = A_plus_thresholds.get('R_P50_min', 1.5)
        
        prob_A = A_thresholds.get('prob_win_min', 0.55)
        R_A = A_thresholds.get('R_P50_min', 1.0)
        
        prob_B = B_thresholds.get('prob_win_min', 0.45)
        R_B = B_thresholds.get('R_P50_min', 0.5)
        
        # Categorize
        if prob_win > prob_A_plus and R_P50 > R_A_plus:
            quality_label = 'A+'
            recommendation = 'TRADE'
        elif prob_win > prob_A and R_P50 > R_A:
            quality_label = 'A'
            recommendation = 'TRADE'
        elif prob_win > prob_B and R_P50 > R_B:
            quality_label = 'B'
            recommendation = 'SKIP'
        else:
            quality_label = 'C'
            recommendation = 'SKIP'
        
        return quality_label, recommendation
    
    def get_quality_color(self, quality_label: str) -> str:
        """
        Get color code for quality label.
        
        Parameters
        ----------
        quality_label : str
            Quality label ('A+', 'A', 'B', 'C')
        
        Returns
        -------
        str
            Hex color code
        """
        # Get color scheme from config
        color_scheme = self.config.get('display', {}).get('color_scheme', {})
        
        # Default colors
        default_colors = {
            'A+': '#006400',  # dark green
            'A': '#32CD32',   # green
            'B': '#FFD700',   # yellow
            'C': '#DC143C',   # red
            'ERROR': '#808080'  # gray
        }
        
        # Map quality label to config key
        label_to_key = {
            'A+': 'A_plus',
            'A': 'A',
            'B': 'B',
            'C': 'C',
            'ERROR': 'ERROR'
        }
        
        config_key = label_to_key.get(quality_label, quality_label)
        
        # Get color from config or use default
        color = color_scheme.get(config_key, default_colors.get(quality_label, '#808080'))
        
        return color
    
    def get_threshold_values(self) -> Dict:
        """
        Get current threshold values from config.
        
        Returns
        -------
        dict
            Dictionary with threshold values for each quality level
        """
        A_plus_thresholds = self.thresholds.get('quality_A_plus', {})
        A_thresholds = self.thresholds.get('quality_A', {})
        B_thresholds = self.thresholds.get('quality_B', {})
        
        return {
            'A+': {
                'prob_win_min': A_plus_thresholds.get('prob_win_min', 0.65),
                'R_P50_min': A_plus_thresholds.get('R_P50_min', 1.5)
            },
            'A': {
                'prob_win_min': A_thresholds.get('prob_win_min', 0.55),
                'R_P50_min': A_thresholds.get('R_P50_min', 1.0)
            },
            'B': {
                'prob_win_min': B_thresholds.get('prob_win_min', 0.45),
                'R_P50_min': B_thresholds.get('R_P50_min', 0.5)
            }
        }
    
    def _get_nan_result(self, error_msg: str = "") -> Dict:
        """
        Get result dict with NaN values for failed predictions.
        
        Parameters
        ----------
        error_msg : str
            Error message to include
        
        Returns
        -------
        dict
            Result dict with NaN values
        """
        return {
            'prob_win_raw': np.nan,
            'prob_win_calibrated': np.nan,
            'R_P10_raw': np.nan,
            'R_P50_raw': np.nan,
            'R_P90_raw': np.nan,
            'R_P10_conf': np.nan,
            'R_P90_conf': np.nan,
            'skewness': np.nan,
            'quality_label': 'ERROR',
            'recommendation': 'SKIP',
            'execution_time_ms': 0.0,
            'component_times_ms': {},
            'error': error_msg
        }
    
    def get_pipeline_info(self) -> Dict:
        """
        Get information about the pipeline.
        
        Returns
        -------
        dict
            Pipeline information including loaded status and feature names
        """
        info = {
            'is_loaded': self.is_loaded,
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names,
            'use_cache': self.use_cache,
            'components': {
                'classifier': self.classifier.is_fitted if self.classifier else False,
                'calibrator': self.calibrator.is_fitted if self.calibrator else False,
                'quantile_ensemble': self.quantile_ensemble.is_fitted if self.quantile_ensemble else False,
                'conformal_engine': self.conformal_engine.is_fitted if self.conformal_engine else False
            },
            'performance': {
                'prediction_count': self._prediction_count,
                'total_time_seconds': self._total_prediction_time,
                'avg_time_ms': (self._total_prediction_time * 1000 / self._prediction_count) 
                               if self._prediction_count > 0 else 0
            }
        }
        
        return info
    
    def get_performance_stats(self) -> Dict:
        """
        Get detailed performance statistics.
        
        Returns
        -------
        dict
            Performance statistics including:
            - prediction_count: Total number of predictions made
            - total_time_seconds: Total time spent on predictions
            - avg_time_ms: Average prediction time in milliseconds
            - cache_enabled: Whether model caching is enabled
            - cache_stats: Global cache statistics
        """
        stats = {
            'prediction_count': self._prediction_count,
            'total_time_seconds': self._total_prediction_time,
            'avg_time_ms': (self._total_prediction_time * 1000 / self._prediction_count) 
                           if self._prediction_count > 0 else 0,
            'cache_enabled': self.use_cache,
            'cache_stats': {
                'cached_models': len(_model_cache),
                'cache_keys': list(_model_cache.keys())
            }
        }
        
        return stats
    
    def reset_performance_stats(self):
        """Reset performance tracking statistics."""
        self._prediction_count = 0
        self._total_prediction_time = 0.0
        logger.info("Performance statistics reset")
    
    @staticmethod
    def clear_model_cache():
        """
        Clear the global model cache.
        
        This forces all pipelines to reload models from disk on next use.
        Useful for freeing memory or forcing model updates.
        """
        with _cache_lock:
            _model_cache.clear()
            logger.info("Global model cache cleared")
    
    @staticmethod
    def get_cache_info() -> Dict:
        """
        Get information about the global model cache.
        
        Returns
        -------
        dict
            Cache information including number of cached models and keys
        """
        with _cache_lock:
            return {
                'cached_models': len(_model_cache),
                'cache_keys': list(_model_cache.keys()),
                'memory_estimate_mb': len(_model_cache) * 50  # Rough estimate: 50MB per model set
            }


def create_pipeline(
    model_dir: Union[str, Path],
    config_path: Optional[Union[str, Path]] = None
) -> PredictionPipeline:
    """
    Convenience function to create and load a pipeline.
    
    Parameters
    ----------
    model_dir : str or Path
        Path to model directory
    config_path : str or Path, optional
        Path to configuration file
    
    Returns
    -------
    PredictionPipeline
        Loaded pipeline ready for prediction
    """
    pipeline = PredictionPipeline(config_path=config_path)
    pipeline.load_models(model_dir)
    
    return pipeline
