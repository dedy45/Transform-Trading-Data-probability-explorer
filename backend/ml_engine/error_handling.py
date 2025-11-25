"""
Error Handling and Validation Module

This module provides comprehensive error handling and validation utilities
for the ML Prediction Engine. It handles model loading errors, prediction
failures, data validation errors, and memory errors.

**Feature: ml-prediction-engine**
**Validates: Requirements 18.1, 18.2, 18.3, 18.4, 18.5**
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
import traceback
import psutil
import sys

logger = logging.getLogger(__name__)


class MLEngineError(Exception):
    """Base exception for ML Engine errors."""
    pass


class ModelNotFoundError(MLEngineError):
    """Raised when model file is not found."""
    pass


class ModelLoadError(MLEngineError):
    """Raised when model loading fails."""
    pass


class FeatureMismatchError(MLEngineError):
    """Raised when features don't match expected schema."""
    pass


class DataValidationError(MLEngineError):
    """Raised when data validation fails."""
    pass


class PredictionError(MLEngineError):
    """Raised when prediction fails."""
    pass


class MemoryError(MLEngineError):
    """Raised when memory limit is exceeded."""
    pass


class ErrorHandler:
    """
    Centralized error handling for ML Prediction Engine.
    
    This class provides methods to handle various error scenarios gracefully,
    including model loading errors, prediction failures, data validation errors,
    and memory errors.
    
    Examples
    --------
    >>> handler = ErrorHandler()
    >>> try:
    ...     # Some operation
    ...     pass
    ... except Exception as e:
    ...     result = handler.handle_prediction_error(e, sample_id=123)
    """
    
    def __init__(self, 
                 log_errors: bool = True,
                 raise_on_critical: bool = False,
                 memory_threshold_pct: float = 90.0):
        """
        Initialize error handler.
        
        Parameters
        ----------
        log_errors : bool
            Whether to log errors (default: True)
        raise_on_critical : bool
            Whether to raise exceptions on critical errors (default: False)
        memory_threshold_pct : float
            Memory usage threshold percentage for warnings (default: 90.0)
        """
        self.log_errors = log_errors
        self.raise_on_critical = raise_on_critical
        self.memory_threshold_pct = memory_threshold_pct
        self.error_count = 0
        self.error_log = []
    
    def check_model_file_exists(self, model_path: Union[str, Path]) -> Tuple[bool, Optional[str]]:
        """
        Check if model file exists.
        
        Parameters
        ----------
        model_path : str or Path
            Path to model file
        
        Returns
        -------
        Tuple[bool, Optional[str]]
            (exists, error_message)
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            error_msg = (
                f"Model file not found: {model_path}\n"
                f"Please train models first using the 'Train Models' button."
            )
            if self.log_errors:
                logger.error(error_msg)
            return False, error_msg
        
        return True, None
    
    def handle_model_not_found(self, model_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Handle model file not found error.
        
        Parameters
        ----------
        model_path : str or Path
            Path to missing model file
        
        Returns
        -------
        Dict[str, Any]
            Error information dictionary
        
        Raises
        ------
        ModelNotFoundError
            If raise_on_critical is True
        """
        error_msg = (
            f"Model not found: {model_path}. "
            f"Please train models first."
        )
        
        error_info = {
            'error_type': 'ModelNotFoundError',
            'error_message': error_msg,
            'model_path': str(model_path),
            'suggestion': 'Click the "Train Models" button to train new models.',
            'recoverable': False
        }
        
        self._log_error(error_info)
        
        if self.raise_on_critical:
            raise ModelNotFoundError(error_msg)
        
        return error_info
    
    def handle_model_load_error(self, 
                                model_path: Union[str, Path],
                                exception: Exception) -> Dict[str, Any]:
        """
        Handle model loading error.
        
        Parameters
        ----------
        model_path : str or Path
            Path to model file
        exception : Exception
            The exception that occurred
        
        Returns
        -------
        Dict[str, Any]
            Error information dictionary
        """
        error_msg = (
            f"Failed to load model from {model_path}: {str(exception)}\n"
            f"This may be due to version mismatch or corrupted file."
        )
        
        error_info = {
            'error_type': 'ModelLoadError',
            'error_message': error_msg,
            'model_path': str(model_path),
            'exception': str(exception),
            'traceback': traceback.format_exc(),
            'suggestion': 'Try retraining the models with the current library versions.',
            'recoverable': False
        }
        
        self._log_error(error_info)
        
        if self.raise_on_critical:
            raise ModelLoadError(error_msg) from exception
        
        return error_info
    
    def validate_features(self,
                         data: pd.DataFrame,
                         required_features: List[str]) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate that all required features are present in data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data
        required_features : List[str]
            List of required feature names
        
        Returns
        -------
        Tuple[bool, Dict[str, Any]]
            (is_valid, validation_report)
        """
        missing_features = [f for f in required_features if f not in data.columns]
        
        if missing_features:
            error_msg = f"Missing features: {missing_features}"
            
            validation_report = {
                'is_valid': False,
                'error_type': 'FeatureMismatchError',
                'error_message': error_msg,
                'missing_features': missing_features,
                'available_features': list(data.columns),
                'required_features': required_features,
                'suggestion': (
                    'Ensure your data contains all required features. '
                    'You may need to update the feature selection or retrain models.'
                ),
                'recoverable': True,
                'recovery_action': 'impute_missing_features'
            }
            
            self._log_error(validation_report)
            
            return False, validation_report
        
        return True, {'is_valid': True, 'message': 'All features present'}
    
    def handle_missing_features(self,
                                data: pd.DataFrame,
                                required_features: List[str],
                                imputation_values: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        Handle missing features by imputing with default values.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data
        required_features : List[str]
            List of required feature names
        imputation_values : Dict[str, float], optional
            Dictionary of feature names to imputation values
        
        Returns
        -------
        pd.DataFrame
            Data with missing features imputed
        """
        data = data.copy()
        missing_features = [f for f in required_features if f not in data.columns]
        
        if not missing_features:
            return data
        
        logger.warning(f"Imputing {len(missing_features)} missing features")
        
        for feature in missing_features:
            # Use provided imputation value or default to 0
            impute_value = imputation_values.get(feature, 0.0) if imputation_values else 0.0
            data[feature] = impute_value
            logger.debug(f"Imputed feature '{feature}' with value {impute_value}")
        
        return data
    
    def handle_prediction_error(self,
                               exception: Exception,
                               sample_id: Optional[Any] = None,
                               return_nan: bool = True) -> Dict[str, Any]:
        """
        Handle prediction error gracefully.
        
        Parameters
        ----------
        exception : Exception
            The exception that occurred
        sample_id : Any, optional
            Identifier for the sample that failed
        return_nan : bool
            Whether to return NaN result (default: True)
        
        Returns
        -------
        Dict[str, Any]
            Error information or NaN result
        """
        error_msg = f"Prediction failed: {str(exception)}"
        if sample_id is not None:
            error_msg = f"Prediction failed for sample {sample_id}: {str(exception)}"
        
        error_info = {
            'error_type': 'PredictionError',
            'error_message': error_msg,
            'sample_id': sample_id,
            'exception': str(exception),
            'traceback': traceback.format_exc(),
            'recoverable': True
        }
        
        self._log_error(error_info)
        
        if return_nan:
            return self._get_nan_result(error_msg)
        
        return error_info
    
    def validate_data_schema(self,
                            data: pd.DataFrame,
                            required_features: List[str],
                            check_types: bool = True) -> Tuple[bool, List[str]]:
        """
        Validate data schema including column presence and types.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data
        required_features : List[str]
            List of required feature names
        check_types : bool
            Whether to check data types (default: True)
        
        Returns
        -------
        Tuple[bool, List[str]]
            (is_valid, list_of_errors)
        """
        errors = []
        
        # Check if DataFrame is empty
        if data.empty:
            errors.append("DataFrame is empty")
            return False, errors
        
        # Check for required features
        missing_features = [f for f in required_features if f not in data.columns]
        if missing_features:
            errors.append(f"Missing required features: {missing_features}")
        
        # Check data types
        if check_types:
            for feature in required_features:
                if feature in data.columns:
                    if not pd.api.types.is_numeric_dtype(data[feature]):
                        errors.append(
                            f"Feature '{feature}' is not numeric (dtype: {data[feature].dtype})"
                        )
        
        # Check for all NaN columns
        for feature in required_features:
            if feature in data.columns:
                if data[feature].isna().all():
                    errors.append(f"Feature '{feature}' contains only NaN values")
        
        is_valid = len(errors) == 0
        
        if not is_valid:
            logger.error(f"Data schema validation failed with {len(errors)} errors")
            for error in errors:
                logger.error(f"  - {error}")
        
        return is_valid, errors
    
    def check_memory_usage(self,
                          data_size_mb: Optional[float] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Check memory usage and warn if threshold exceeded.
        
        Parameters
        ----------
        data_size_mb : float, optional
            Size of data being processed in MB
        
        Returns
        -------
        Tuple[bool, Dict[str, Any]]
            (is_safe, memory_info)
        """
        try:
            # Get memory info
            memory = psutil.virtual_memory()
            memory_pct = memory.percent
            available_mb = memory.available / (1024 * 1024)
            
            memory_info = {
                'memory_percent': memory_pct,
                'available_mb': available_mb,
                'total_mb': memory.total / (1024 * 1024),
                'used_mb': memory.used / (1024 * 1024)
            }
            
            # Check if memory usage is too high
            if memory_pct > self.memory_threshold_pct:
                error_msg = (
                    f"Memory usage is high: {memory_pct:.1f}% "
                    f"(available: {available_mb:.1f} MB)"
                )
                
                memory_info['error_type'] = 'MemoryWarning'
                memory_info['error_message'] = error_msg
                memory_info['suggestion'] = (
                    'Consider reducing batch size or closing other applications.'
                )
                
                logger.warning(error_msg)
                
                return False, memory_info
            
            # Check if data size exceeds available memory
            if data_size_mb is not None and data_size_mb > available_mb * 0.5:
                error_msg = (
                    f"Data size ({data_size_mb:.1f} MB) may exceed available memory "
                    f"({available_mb:.1f} MB)"
                )
                
                memory_info['error_type'] = 'MemoryWarning'
                memory_info['error_message'] = error_msg
                memory_info['data_size_mb'] = data_size_mb
                memory_info['suggestion'] = (
                    'Consider processing data in smaller batches.'
                )
                
                logger.warning(error_msg)
                
                return False, memory_info
            
            return True, memory_info
            
        except Exception as e:
            logger.warning(f"Failed to check memory usage: {e}")
            return True, {'error': str(e)}
    
    def handle_memory_error(self,
                           exception: Exception,
                           batch_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Handle memory error with suggestions.
        
        Parameters
        ----------
        exception : Exception
            The memory error exception
        batch_size : int, optional
            Current batch size
        
        Returns
        -------
        Dict[str, Any]
            Error information with suggestions
        """
        error_msg = f"Memory error occurred: {str(exception)}"
        
        # Calculate suggested batch size
        suggested_batch_size = None
        if batch_size is not None:
            suggested_batch_size = max(1, batch_size // 2)
        
        error_info = {
            'error_type': 'MemoryError',
            'error_message': error_msg,
            'exception': str(exception),
            'current_batch_size': batch_size,
            'suggested_batch_size': suggested_batch_size,
            'suggestion': (
                f"Memory limit exceeded. "
                f"Try reducing batch size to {suggested_batch_size} or smaller. "
                f"Alternatively, close other applications to free up memory."
            ),
            'recoverable': True,
            'recovery_action': 'reduce_batch_size'
        }
        
        self._log_error(error_info)
        
        return error_info
    
    def _log_error(self, error_info: Dict[str, Any]):
        """Log error information."""
        if self.log_errors:
            error_type = error_info.get('error_type', 'UnknownError')
            error_msg = error_info.get('error_message', 'No message')
            logger.error(f"[{error_type}] {error_msg}")
        
        self.error_count += 1
        self.error_log.append(error_info)
    
    def _get_nan_result(self, error_msg: str = "") -> Dict[str, Any]:
        """
        Get result dict with NaN values for failed predictions.
        
        Parameters
        ----------
        error_msg : str
            Error message to include
        
        Returns
        -------
        Dict[str, Any]
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
    
    def get_error_summary(self) -> Dict[str, Any]:
        """
        Get summary of all errors encountered.
        
        Returns
        -------
        Dict[str, Any]
            Error summary
        """
        error_types = {}
        for error in self.error_log:
            error_type = error.get('error_type', 'Unknown')
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            'total_errors': self.error_count,
            'error_types': error_types,
            'recent_errors': self.error_log[-10:]  # Last 10 errors
        }
    
    def clear_error_log(self):
        """Clear error log and reset counter."""
        self.error_count = 0
        self.error_log = []


def validate_model_files(model_dir: Union[str, Path],
                        required_files: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate that all required model files exist.
    
    Parameters
    ----------
    model_dir : str or Path
        Path to model directory
    required_files : List[str]
        List of required model file names
    
    Returns
    -------
    Tuple[bool, List[str]]
        (all_exist, list_of_missing_files)
    
    Examples
    --------
    >>> all_exist, missing = validate_model_files(
    ...     'data_processed/models',
    ...     ['lgbm_classifier.pkl', 'isotonic_calibrator.pkl']
    ... )
    >>> if not all_exist:
    ...     print(f"Missing files: {missing}")
    """
    model_dir = Path(model_dir)
    missing_files = []
    
    for filename in required_files:
        file_path = model_dir / filename
        if not file_path.exists():
            missing_files.append(filename)
    
    all_exist = len(missing_files) == 0
    
    return all_exist, missing_files


def safe_predict(predict_func,
                *args,
                error_handler: Optional[ErrorHandler] = None,
                return_nan_on_error: bool = True,
                **kwargs) -> Any:
    """
    Safely execute prediction function with error handling.
    
    Parameters
    ----------
    predict_func : callable
        Prediction function to execute
    *args
        Positional arguments for predict_func
    error_handler : ErrorHandler, optional
        Error handler instance
    return_nan_on_error : bool
        Whether to return NaN result on error (default: True)
    **kwargs
        Keyword arguments for predict_func
    
    Returns
    -------
    Any
        Prediction result or NaN result on error
    
    Examples
    --------
    >>> result = safe_predict(
    ...     pipeline.predict_for_sample,
    ...     feature_row,
    ...     error_handler=handler
    ... )
    """
    if error_handler is None:
        error_handler = ErrorHandler()
    
    try:
        return predict_func(*args, **kwargs)
    except Exception as e:
        return error_handler.handle_prediction_error(
            e,
            return_nan=return_nan_on_error
        )
