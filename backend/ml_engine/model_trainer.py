"""
Model Trainer for ML Prediction Engine

This module implements the ModelTrainer class that orchestrates the training
of all 4 ML components (Classifier, Calibration, Quantile Regression, Conformal Prediction).

**Feature: ml-prediction-engine**
**Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5**
"""

import pandas as pd
import numpy as np
import json
import yaml
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any
from datetime import datetime
import logging
import traceback

# Import ML components
from backend.ml_engine.lgbm_classifier import LGBMClassifierWrapper
from backend.ml_engine.calibration_isotonic import IsotonicCalibrator
from backend.ml_engine.lgbm_quantile_regressor import QuantileRegressorEnsemble
from backend.ml_engine.conformal_engine import ConformalEngine

# Import preprocessing utilities
from backend.preprocessing.schema_validation import SchemaValidator
from backend.preprocessing.cleaning import DataCleaner
from backend.preprocessing.split_timeseries import TimeSeriesSplitter

logger = logging.getLogger(__name__)


class TrainingProgress:
    """
    Tracks training progress for UI display.
    """
    
    def __init__(self):
        self.current_component = None
        self.current_step = None
        self.progress_percentage = 0.0
        self.status = 'not_started'  # not_started, in_progress, completed, failed
        self.error_message = None
        self.component_status = {
            'preprocessing': 'pending',
            'classifier': 'pending',
            'calibration': 'pending',
            'quantile': 'pending',
            'conformal': 'pending'
        }
    
    def update(self, component: str, step: str, progress: float, status: str = 'in_progress'):
        """Update progress."""
        self.current_component = component
        self.current_step = step
        self.progress_percentage = progress
        self.status = status
        self.component_status[component] = status
        
        logger.info(f"Training progress: {component} - {step} ({progress:.1f}%)")
    
    def set_error(self, component: str, error_message: str):
        """Set error state."""
        self.status = 'failed'
        self.error_message = error_message
        self.component_status[component] = 'failed'
        
        logger.error(f"Training failed at {component}: {error_message}")
    
    def complete(self):
        """Mark training as completed."""
        self.status = 'completed'
        self.progress_percentage = 100.0
        
        logger.info("Training completed successfully")
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'current_component': self.current_component,
            'current_step': self.current_step,
            'progress_percentage': self.progress_percentage,
            'status': self.status,
            'error_message': self.error_message,
            'component_status': self.component_status
        }


class ModelTrainer:
    """
    Orchestrates training of all 4 ML components.
    
    This class manages the complete training workflow:
    1. Load and validate data
    2. Preprocess data (clean, split)
    3. Train classifier
    4. Train calibrator
    5. Train quantile ensemble
    6. Fit conformal engine
    7. Save all models and metrics
    
    Attributes
    ----------
    config : dict
        Configuration dictionary loaded from YAML
    model_dir : Path
        Directory for saving models
    progress : TrainingProgress
        Training progress tracker
    backup_dir : Path
        Directory for backing up existing models
    
    Examples
    --------
    >>> trainer = ModelTrainer(config_path='config/ml_prediction_config.yaml')
    >>> metrics = trainer.train_all_components(
    ...     data=df,
    ...     feature_columns=['feature1', 'feature2'],
    ...     target_column='R_multiple',
    ...     win_column='trade_success'
    ... )
    >>> print(f"Training completed with AUC: {metrics['classifier']['auc_val']}")
    """
    
    def __init__(
        self,
        config_path: str = 'config/ml_prediction_config.yaml',
        model_dir: Optional[str] = None,
        progress_callback: Optional[Callable[[Dict], None]] = None
    ):
        """
        Initialize model trainer.
        
        Parameters
        ----------
        config_path : str, default='config/ml_prediction_config.yaml'
            Path to configuration file
        model_dir : str, optional
            Directory for saving models (overrides config)
        progress_callback : callable, optional
            Callback function for progress updates (receives progress dict)
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Set model directory
        if model_dir is not None:
            self.model_dir = Path(model_dir)
        else:
            self.model_dir = Path(self.config['paths']['model_dir'])
        
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize progress tracker
        self.progress = TrainingProgress()
        self.progress_callback = progress_callback
        
        # Backup directory for rollback
        self.backup_dir = self.model_dir / 'backup'
        
        # Initialize components (will be populated during training)
        self.classifier = None
        self.calibrator = None
        self.quantile_ensemble = None
        self.conformal_engine = None
        
        # Training data splits
        self.train_df = None
        self.calib_df = None
        self.test_df = None
        
        # Feature columns
        self.feature_columns = None
        
        logger.info(f"ModelTrainer initialized with model_dir: {self.model_dir}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
    
    def _update_progress(self, component: str, step: str, progress: float, status: str = 'in_progress'):
        """Update progress and call callback if provided."""
        self.progress.update(component, step, progress, status)
        
        if self.progress_callback is not None:
            self.progress_callback(self.progress.to_dict())
    
    def _backup_existing_models(self):
        """Backup existing models for rollback."""
        if not self.model_dir.exists():
            return
        
        # Create backup directory
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # List of model files to backup
        model_files = [
            self.config['paths']['classifier_model'],
            self.config['paths']['calibrator_model'],
            self.config['paths']['quantile_p10_model'],
            self.config['paths']['quantile_p50_model'],
            self.config['paths']['quantile_p90_model'],
            self.config['paths']['conformal_meta'],
            self.config['paths']['feature_list'],
            self.config['paths']['training_metrics']
        ]
        
        # Backup each file if it exists
        backed_up_count = 0
        for filename in model_files:
            src = self.model_dir / filename
            if src.exists():
                dst = self.backup_dir / filename
                shutil.copy2(src, dst)
                backed_up_count += 1
        
        if backed_up_count > 0:
            logger.info(f"Backed up {backed_up_count} existing model files to {self.backup_dir}")
    
    def _rollback_models(self):
        """Rollback to backed up models."""
        if not self.backup_dir.exists():
            logger.warning("No backup directory found, cannot rollback")
            return
        
        # Restore each backed up file
        restored_count = 0
        for backup_file in self.backup_dir.iterdir():
            if backup_file.is_file():
                dst = self.model_dir / backup_file.name
                shutil.copy2(backup_file, dst)
                restored_count += 1
        
        if restored_count > 0:
            logger.info(f"Rolled back {restored_count} model files from backup")
    
    def _cleanup_backup(self):
        """Remove backup directory after successful training."""
        if self.backup_dir.exists():
            shutil.rmtree(self.backup_dir)
            logger.info("Cleaned up backup directory")
    
    def train_all_components(
        self,
        data: pd.DataFrame,
        feature_columns: List[str],
        target_column: str = 'R_multiple',
        win_column: str = 'trade_success',
        overwrite: bool = False
    ) -> Dict[str, Any]:
        """
        Train all 4 ML components in sequence.
        
        This is the main training workflow that:
        1. Validates and preprocesses data
        2. Splits data into train/calib/test
        3. Trains classifier
        4. Trains calibrator
        5. Trains quantile ensemble
        6. Fits conformal engine
        7. Saves all models and metrics
        
        Parameters
        ----------
        data : pd.DataFrame
            Complete dataset with features and targets
        feature_columns : List[str]
            List of feature column names to use
        target_column : str, default='R_multiple'
            Name of target column (R_multiple values)
        win_column : str, default='trade_success'
            Name of binary win/loss column (0/1)
        overwrite : bool, default=False
            Whether to overwrite existing models without backup
        
        Returns
        -------
        dict
            Complete training metrics from all components
        
        Raises
        ------
        ValueError
            If data validation fails or insufficient data
        RuntimeError
            If training fails for any component
        """
        try:
            # Start training
            self._update_progress('preprocessing', 'Starting training', 0.0, 'in_progress')
            
            # Backup existing models if not overwriting
            if not overwrite:
                self._backup_existing_models()
            
            # Store feature columns
            self.feature_columns = feature_columns
            
            # Step 1: Validate and preprocess data (0-20%)
            self._update_progress('preprocessing', 'Validating data', 5.0)
            train_df, calib_df, test_df = self._preprocess_data(
                data, feature_columns, target_column, win_column
            )
            
            # Store splits
            self.train_df = train_df
            self.calib_df = calib_df
            self.test_df = test_df
            
            self._update_progress('preprocessing', 'Data preprocessing completed', 20.0, 'completed')
            
            # Step 2: Train classifier (20-40%)
            self._update_progress('classifier', 'Training LightGBM classifier', 25.0)
            classifier_metrics = self._train_classifier(
                train_df, calib_df, feature_columns, win_column
            )
            self._update_progress('classifier', 'Classifier training completed', 40.0, 'completed')
            
            # Step 3: Train calibrator (40-55%)
            self._update_progress('calibration', 'Training isotonic calibrator', 45.0)
            calibration_metrics = self._train_calibrator(
                calib_df, feature_columns, win_column
            )
            self._update_progress('calibration', 'Calibration training completed', 55.0, 'completed')
            
            # Step 4: Train quantile ensemble (55-75%)
            self._update_progress('quantile', 'Training quantile regressors', 60.0)
            quantile_metrics = self._train_quantile_ensemble(
                train_df, calib_df, feature_columns, target_column
            )
            self._update_progress('quantile', 'Quantile training completed', 75.0, 'completed')
            
            # Step 5: Fit conformal engine (75-90%)
            self._update_progress('conformal', 'Fitting conformal engine', 80.0)
            conformal_metrics = self._fit_conformal_engine(
                calib_df, feature_columns, target_column
            )
            self._update_progress('conformal', 'Conformal fitting completed', 90.0, 'completed')
            
            # Step 6: Save models and metrics (90-100%)
            self._update_progress('conformal', 'Saving models and metrics', 95.0)
            self._save_all_models(feature_columns)
            
            # Compile all metrics
            all_metrics = {
                'classifier': classifier_metrics,
                'calibration': calibration_metrics,
                'quantile': quantile_metrics,
                'conformal': conformal_metrics,
                'metadata': {
                    'n_train': len(train_df),
                    'n_calib': len(calib_df),
                    'n_test': len(test_df),
                    'n_features': len(feature_columns),
                    'feature_columns': feature_columns,
                    'training_timestamp': datetime.now().isoformat(),
                    'config': self.config
                }
            }
            
            # Save training metrics
            self._save_training_metrics(all_metrics)
            
            # Cleanup backup on success
            if not overwrite:
                self._cleanup_backup()
            
            # Mark as completed
            self.progress.complete()
            if self.progress_callback is not None:
                self.progress_callback(self.progress.to_dict())
            
            logger.info("All components trained successfully")
            return all_metrics
            
        except Exception as e:
            # Handle training failure
            error_msg = f"Training failed: {str(e)}\n{traceback.format_exc()}"
            self.progress.set_error(self.progress.current_component or 'unknown', error_msg)
            
            if self.progress_callback is not None:
                self.progress_callback(self.progress.to_dict())
            
            # Rollback to previous models if backup exists
            if not overwrite:
                logger.error("Training failed, attempting rollback...")
                self._rollback_models()
            
            raise RuntimeError(error_msg)
    
    def _preprocess_data(
        self,
        data: pd.DataFrame,
        feature_columns: List[str],
        target_column: str,
        win_column: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Preprocess data: validate, clean, and split.
        
        Parameters
        ----------
        data : pd.DataFrame
            Raw data
        feature_columns : List[str]
            Feature columns
        target_column : str
            Target column name
        win_column : str
            Win/loss column name
        
        Returns
        -------
        tuple
            (train_df, calib_df, test_df)
        """
        # Validate schema
        validator = SchemaValidator(feature_columns, target_column)
        is_valid, errors = validator.validate_schema(data, check_target=True)
        
        if not is_valid:
            raise ValueError(f"Data validation failed: {errors}")
        
        # Check for win column
        if win_column not in data.columns:
            raise ValueError(f"Win column '{win_column}' not found in data")
        
        # Check minimum samples
        min_samples = self.config['preprocessing']['min_samples_train']
        if len(data) < min_samples:
            raise ValueError(
                f"Insufficient data: need at least {min_samples} samples, got {len(data)}"
            )
        
        # Clean data
        cleaner = DataCleaner(
            handle_missing=self.config['features']['handle_missing'],
            clip_r_multiple=True,
            r_min=self.config['preprocessing']['r_multiple_clip_min'],
            r_max=self.config['preprocessing']['r_multiple_clip_max']
        )
        
        # Fit cleaner on full data
        cleaner.fit(data, feature_columns)
        
        # Clean data
        data_clean = cleaner.clean_data(
            data,
            feature_columns,
            target_column,
            inplace=False
        )
        
        logger.info(f"Data cleaned: {len(data)} -> {len(data_clean)} rows")
        
        # Split data
        splitter = TimeSeriesSplitter(
            train_ratio=self.config['data_split']['train_ratio'],
            calib_ratio=self.config['data_split']['calib_ratio'],
            test_ratio=self.config['data_split']['test_ratio'],
            time_column=None,  # Assume data is already sorted
            random_state=self.config['data_split'].get('random_state', 42)
        )
        
        train_df, calib_df, test_df = splitter.split(
            data_clean,
            method=self.config['data_split']['split_method']
        )
        
        logger.info(
            f"Data split: train={len(train_df)}, calib={len(calib_df)}, test={len(test_df)}"
        )
        
        return train_df, calib_df, test_df
    
    def _train_classifier(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_columns: List[str],
        win_column: str
    ) -> Dict[str, float]:
        """Train LightGBM classifier."""
        X_train = train_df[feature_columns]
        y_train = train_df[win_column]
        X_val = val_df[feature_columns]
        y_val = val_df[win_column]
        
        # Get hyperparameters from config
        hyper = self.config['model_hyperparameters']['classifier']
        
        # Initialize and train classifier
        self.classifier = LGBMClassifierWrapper(
            n_estimators=hyper['n_estimators'],
            learning_rate=hyper['learning_rate'],
            max_depth=hyper['max_depth'],
            min_child_samples=hyper['min_child_samples'],
            subsample=hyper.get('subsample', 0.8),
            colsample_bytree=hyper.get('colsample_bytree', 0.8),
            random_state=hyper['random_state']
        )
        
        metrics = self.classifier.fit(
            X_train, y_train,
            X_val, y_val,
            use_cv=True,
            n_splits=self.config['preprocessing']['cv_folds']
        )
        
        logger.info(f"Classifier trained: AUC={metrics.get('auc_val', metrics['auc_train']):.4f}")
        return metrics
    
    def _train_calibrator(
        self,
        calib_df: pd.DataFrame,
        feature_columns: List[str],
        win_column: str
    ) -> Dict[str, float]:
        """Train isotonic calibrator."""
        if self.classifier is None:
            raise RuntimeError("Classifier must be trained before calibrator")
        
        X_calib = calib_df[feature_columns]
        y_calib = calib_df[win_column]
        
        # Get raw probabilities from classifier
        raw_probs = self.classifier.predict_proba(X_calib)
        
        # Initialize and train calibrator
        self.calibrator = IsotonicCalibrator()
        metrics = self.calibrator.fit(raw_probs, y_calib.values)
        
        logger.info(
            f"Calibrator trained: Brier improvement={metrics['brier_improvement']:.4f}"
        )
        return metrics
    
    def _train_quantile_ensemble(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_columns: List[str],
        target_column: str
    ) -> Dict[str, float]:
        """Train quantile regression ensemble."""
        X_train = train_df[feature_columns]
        y_train = train_df[target_column]
        X_val = val_df[feature_columns]
        y_val = val_df[target_column]
        
        # Get hyperparameters from config
        hyper = self.config['model_hyperparameters']['quantile']
        
        # Initialize and train ensemble
        self.quantile_ensemble = QuantileRegressorEnsemble(
            n_estimators=hyper['n_estimators'],
            learning_rate=hyper['learning_rate'],
            max_depth=hyper['max_depth'],
            min_child_samples=hyper['min_child_samples'],
            random_state=hyper['random_state']
        )
        
        metrics = self.quantile_ensemble.fit(X_train, y_train, X_val, y_val)
        
        logger.info(
            f"Quantile ensemble trained: MAE_P50={metrics.get('mae_p50_val', metrics['mae_p50_train']):.4f}"
        )
        return metrics
    
    def _fit_conformal_engine(
        self,
        calib_df: pd.DataFrame,
        feature_columns: List[str],
        target_column: str
    ) -> Dict:
        """Fit conformal prediction engine."""
        if self.quantile_ensemble is None:
            raise RuntimeError("Quantile ensemble must be trained before conformal engine")
        
        X_calib = calib_df[feature_columns]
        y_calib = calib_df[target_column]
        
        # Get quantile predictions
        quantile_preds = self.quantile_ensemble.predict(X_calib)
        y_pred_p10 = quantile_preds['p10']
        y_pred_p90 = quantile_preds['p90']
        
        # Initialize and fit conformal engine
        coverage = self.config['conformal']['coverage']
        self.conformal_engine = ConformalEngine(coverage=coverage)
        
        metadata = self.conformal_engine.fit(y_pred_p10, y_pred_p90, y_calib.values)
        
        logger.info(
            f"Conformal engine fitted: coverage={coverage}, "
            f"nonconformity_quantile={metadata['nonconformity_quantile']:.4f}"
        )
        return metadata
    
    def _save_all_models(self, feature_columns: List[str]):
        """Save all trained models."""
        # Save classifier
        classifier_path = self.model_dir / self.config['paths']['classifier_model']
        self.classifier.save(classifier_path)
        logger.info(f"Saved classifier to {classifier_path}")
        
        # Save calibrator
        calibrator_path = self.model_dir / self.config['paths']['calibrator_model']
        self.calibrator.save(calibrator_path)
        logger.info(f"Saved calibrator to {calibrator_path}")
        
        # Save quantile ensemble
        quantile_prefix = self.model_dir / 'lgbm_quantile'
        self.quantile_ensemble.save(quantile_prefix)
        logger.info(f"Saved quantile ensemble to {quantile_prefix}_*.pkl")
        
        # Save conformal engine
        conformal_path = self.model_dir / self.config['paths']['conformal_meta']
        self.conformal_engine.save(conformal_path)
        logger.info(f"Saved conformal engine to {conformal_path}")
        
        # Save feature list
        feature_list_path = self.model_dir / self.config['paths']['feature_list']
        with open(feature_list_path, 'w') as f:
            json.dump({'features': feature_columns}, f, indent=2)
        logger.info(f"Saved feature list to {feature_list_path}")
    
    def _save_training_metrics(self, metrics: Dict):
        """Save training metrics to JSON."""
        metrics_path = self.model_dir / self.config['paths']['training_metrics']
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        logger.info(f"Saved training metrics to {metrics_path}")
    
    def get_training_status(self) -> Dict:
        """
        Get current training status.
        
        Returns
        -------
        dict
            Training status dictionary
        """
        return self.progress.to_dict()


def train_models(
    data: pd.DataFrame,
    feature_columns: List[str],
    target_column: str = 'R_multiple',
    win_column: str = 'trade_success',
    config_path: str = 'config/ml_prediction_config.yaml',
    model_dir: Optional[str] = None,
    overwrite: bool = False,
    progress_callback: Optional[Callable[[Dict], None]] = None
) -> Dict[str, Any]:
    """
    Convenience function to train all models.
    
    Parameters
    ----------
    data : pd.DataFrame
        Complete dataset
    feature_columns : List[str]
        Feature column names
    target_column : str, default='R_multiple'
        Target column name
    win_column : str, default='trade_success'
        Win/loss column name
    config_path : str, default='config/ml_prediction_config.yaml'
        Path to configuration file
    model_dir : str, optional
        Directory for saving models
    overwrite : bool, default=False
        Whether to overwrite existing models
    progress_callback : callable, optional
        Callback for progress updates
    
    Returns
    -------
    dict
        Training metrics from all components
    
    Examples
    --------
    >>> metrics = train_models(
    ...     data=df,
    ...     feature_columns=['feature1', 'feature2'],
    ...     target_column='R_multiple',
    ...     win_column='trade_success'
    ... )
    >>> print(f"Training completed with AUC: {metrics['classifier']['auc_val']}")
    """
    trainer = ModelTrainer(
        config_path=config_path,
        model_dir=model_dir,
        progress_callback=progress_callback
    )
    
    return trainer.train_all_components(
        data=data,
        feature_columns=feature_columns,
        target_column=target_column,
        win_column=win_column,
        overwrite=overwrite
    )
