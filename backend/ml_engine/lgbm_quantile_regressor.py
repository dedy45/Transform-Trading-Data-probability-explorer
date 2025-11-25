"""
LightGBM Quantile Regression for R_multiple Distribution Prediction

This module implements an ensemble of quantile regressors to predict the
distribution of R_multiple outcomes (P10, P50, P90) for trading setups.

**Feature: ml-prediction-engine**
**Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5**
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Optional, Dict, Union, Tuple
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb


class QuantileRegressorEnsemble:
    """
    Ensemble of 3 quantile regressors (P10, P50, P90).
    
    This class trains independent quantile regression models to predict
    different quantiles of the R_multiple distribution, providing a
    comprehensive view of potential outcomes.
    
    Attributes
    ----------
    models : dict
        Dictionary containing the three quantile models (p10, p50, p90)
    feature_names : list
        Names of features used for training
    training_metrics : dict
        Metrics from the training process
    is_fitted : bool
        Whether the models have been fitted
    
    Examples
    --------
    >>> ensemble = QuantileRegressorEnsemble(n_estimators=100)
    >>> metrics = ensemble.fit(X_train, y_train, X_val, y_val)
    >>> predictions = ensemble.predict(X_test)
    >>> print(predictions['p10'], predictions['p50'], predictions['p90'])
    >>> ensemble.save('models/quantile')
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = 5,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize quantile regressor ensemble.
        
        Parameters
        ----------
        n_estimators : int, default=100
            Number of boosting iterations
        learning_rate : float, default=0.05
            Learning rate for boosting
        max_depth : int, default=5
            Maximum tree depth
        min_child_samples : int, default=20
            Minimum number of samples in a leaf
        subsample : float, default=0.8
            Subsample ratio of training data
        colsample_bytree : float, default=0.8
            Subsample ratio of features
        random_state : int, default=42
            Random seed for reproducibility
        **kwargs : dict
            Additional parameters for LGBMRegressor
        """
        # Create three independent quantile models
        self.models = {
            'p10': lgb.LGBMRegressor(
                objective='quantile',
                alpha=0.1,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_child_samples=min_child_samples,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                random_state=random_state,
                verbose=-1,
                **kwargs
            ),
            'p50': lgb.LGBMRegressor(
                objective='quantile',
                alpha=0.5,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_child_samples=min_child_samples,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                random_state=random_state,
                verbose=-1,
                **kwargs
            ),
            'p90': lgb.LGBMRegressor(
                objective='quantile',
                alpha=0.9,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_child_samples=min_child_samples,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                random_state=random_state,
                verbose=-1,
                **kwargs
            )
        }
        
        self.feature_names = None
        self.training_metrics = {}
        self.is_fitted = False
        
        # Store hyperparameters
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Train 3 quantile models independently.
        
        Each model is trained to predict a different quantile of the
        R_multiple distribution: P10 (10th percentile), P50 (median),
        and P90 (90th percentile).
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target (R_multiple values)
        X_val : pd.DataFrame, optional
            Validation features
        y_val : pd.Series, optional
            Validation target
        
        Returns
        -------
        dict
            Training metrics including:
            - mae_p10_train: MAE for P10 model on training set
            - mae_p50_train: MAE for P50 model on training set
            - mae_p90_train: MAE for P90 model on training set
            - mae_p10_val: MAE for P10 model on validation set (if provided)
            - mae_p50_val: MAE for P50 model on validation set (if provided)
            - mae_p90_val: MAE for P90 model on validation set (if provided)
            - n_samples_train: Number of training samples
            - n_samples_val: Number of validation samples (if provided)
        
        Raises
        ------
        ValueError
            If X_train is empty or y_train has wrong shape
        """
        # Validate inputs
        if len(X_train) == 0:
            raise ValueError("X_train cannot be empty")
        
        if len(X_train) != len(y_train):
            raise ValueError(
                f"X_train and y_train must have same length: "
                f"{len(X_train)} != {len(y_train)}"
            )
        
        # Store feature names
        self.feature_names = list(X_train.columns)
        
        # Train each quantile model independently
        self.training_metrics = {
            'n_samples_train': len(X_train),
            'n_features': len(self.feature_names)
        }
        
        for quantile_name, model in self.models.items():
            # Prepare evaluation set
            eval_set = [(X_train, y_train)]
            if X_val is not None and y_val is not None:
                eval_set.append((X_val, y_val))
            
            # Train model
            model.fit(
                X_train,
                y_train,
                eval_set=eval_set if len(eval_set) > 1 else None
            )
            
            # Calculate training MAE
            y_train_pred = model.predict(X_train)
            mae_train = mean_absolute_error(y_train, y_train_pred)
            self.training_metrics[f'mae_{quantile_name}_train'] = float(mae_train)
            
            # Calculate validation MAE if available
            if X_val is not None and y_val is not None:
                y_val_pred = model.predict(X_val)
                mae_val = mean_absolute_error(y_val, y_val_pred)
                self.training_metrics[f'mae_{quantile_name}_val'] = float(mae_val)
        
        # Add validation sample count if available
        if X_val is not None:
            self.training_metrics['n_samples_val'] = len(X_val)
        
        self.is_fitted = True
        
        return self.training_metrics
    
    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Predict P10, P50, P90 for each sample.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features for prediction
        
        Returns
        -------
        dict
            Dictionary with keys 'p10', 'p50', 'p90' and values as np.ndarray
            containing predictions for each quantile
        
        Raises
        ------
        ValueError
            If models are not fitted or features don't match
        """
        if not self.is_fitted:
            raise ValueError(
                "Models must be fitted before prediction. Call fit() first."
            )
        
        # Validate features
        if list(X.columns) != self.feature_names:
            missing = set(self.feature_names) - set(X.columns)
            extra = set(X.columns) - set(self.feature_names)
            
            error_msg = "Feature mismatch:\n"
            if missing:
                error_msg += f"  Missing features: {missing}\n"
            if extra:
                error_msg += f"  Extra features: {extra}\n"
            
            raise ValueError(error_msg)
        
        # Predict with each model
        predictions = {}
        for quantile_name, model in self.models.items():
            predictions[quantile_name] = model.predict(X)
        
        return predictions
    
    def calculate_skewness(
        self,
        predictions: Optional[Dict[str, np.ndarray]] = None,
        X: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """
        Calculate skewness of the predicted distribution.
        
        Skewness is calculated as:
        skew = (P90 - P50) / (P50 - P10)
        
        Positive skew indicates upside potential (right tail is longer).
        Negative skew indicates downside risk (left tail is longer).
        
        Parameters
        ----------
        predictions : dict, optional
            Dictionary with 'p10', 'p50', 'p90' predictions.
            If None, X must be provided to generate predictions.
        X : pd.DataFrame, optional
            Features to generate predictions if predictions not provided
        
        Returns
        -------
        np.ndarray
            Skewness values for each sample
        
        Raises
        ------
        ValueError
            If neither predictions nor X is provided
            If predictions don't contain required keys
        """
        # Get predictions if not provided
        if predictions is None:
            if X is None:
                raise ValueError(
                    "Must provide either predictions or X to calculate skewness"
                )
            predictions = self.predict(X)
        
        # Validate predictions
        required_keys = {'p10', 'p50', 'p90'}
        if not required_keys.issubset(predictions.keys()):
            raise ValueError(
                f"Predictions must contain keys: {required_keys}"
            )
        
        p10 = predictions['p10']
        p50 = predictions['p50']
        p90 = predictions['p90']
        
        # Calculate skewness
        # Handle division by zero: if P50 == P10, set skewness to 0
        denominator = p50 - p10
        numerator = p90 - p50
        
        # Avoid division by zero
        skewness = np.zeros_like(p50)
        mask = np.abs(denominator) > 1e-10
        skewness[mask] = numerator[mask] / denominator[mask]
        
        return skewness
    
    def get_feature_importance(
        self,
        quantile: str = 'p50',
        top_n: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get feature importance from a specific quantile model.
        
        Parameters
        ----------
        quantile : str, default='p50'
            Which quantile model to get importance from ('p10', 'p50', or 'p90')
        top_n : int, optional
            Return only top N features. If None, return all.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: feature, importance, rank
        
        Raises
        ------
        ValueError
            If models are not fitted or invalid quantile specified
        """
        if not self.is_fitted:
            raise ValueError(
                "Models must be fitted before getting feature importance"
            )
        
        if quantile not in self.models:
            raise ValueError(
                f"Invalid quantile: {quantile}. Must be one of {list(self.models.keys())}"
            )
        
        model = self.models[quantile]
        importance_values = model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_values
        })
        
        # Sort by importance and add rank
        importance_df = importance_df.sort_values(
            'importance',
            ascending=False
        ).reset_index(drop=True)
        
        importance_df['rank'] = range(1, len(importance_df) + 1)
        
        if top_n is not None:
            return importance_df.head(top_n)
        
        return importance_df
    
    def save(self, path_prefix: Union[str, Path]):
        """
        Save 3 models to files.
        
        Models are saved as:
        - {path_prefix}_p10.pkl
        - {path_prefix}_p50.pkl
        - {path_prefix}_p90.pkl
        
        Parameters
        ----------
        path_prefix : str or Path
            Path prefix for saving models (without extension)
        
        Raises
        ------
        ValueError
            If models are not fitted
        """
        if not self.is_fitted:
            raise ValueError(
                "Cannot save unfitted models. Call fit() first."
            )
        
        path_prefix = Path(path_prefix)
        path_prefix.parent.mkdir(parents=True, exist_ok=True)
        
        # Save each model with metadata
        for quantile_name, model in self.models.items():
            model_path = path_prefix.parent / f"{path_prefix.name}_{quantile_name}.pkl"
            
            model_data = {
                'model': model,
                'quantile': quantile_name,
                'alpha': model.alpha,
                'feature_names': self.feature_names,
                'training_metrics': self.training_metrics,
                'hyperparameters': {
                    'n_estimators': self.n_estimators,
                    'learning_rate': self.learning_rate,
                    'max_depth': self.max_depth,
                    'random_state': self.random_state
                }
            }
            
            joblib.dump(model_data, model_path)
        
        # Also save ensemble metadata
        ensemble_meta_path = path_prefix.parent / f"{path_prefix.name}_meta.pkl"
        ensemble_meta = {
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics,
            'is_fitted': self.is_fitted,
            'hyperparameters': {
                'n_estimators': self.n_estimators,
                'learning_rate': self.learning_rate,
                'max_depth': self.max_depth,
                'random_state': self.random_state
            }
        }
        joblib.dump(ensemble_meta, ensemble_meta_path)
    
    def load(self, path_prefix: Union[str, Path]):
        """
        Load 3 models from files.
        
        Loads models from:
        - {path_prefix}_p10.pkl
        - {path_prefix}_p50.pkl
        - {path_prefix}_p90.pkl
        
        Parameters
        ----------
        path_prefix : str or Path
            Path prefix for loading models (without extension)
        
        Raises
        ------
        FileNotFoundError
            If any model file doesn't exist
        """
        path_prefix = Path(path_prefix)
        
        # Load each model
        for quantile_name in ['p10', 'p50', 'p90']:
            model_path = path_prefix.parent / f"{path_prefix.name}_{quantile_name}.pkl"
            
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model file not found: {model_path}"
                )
            
            model_data = joblib.load(model_path)
            self.models[quantile_name] = model_data['model']
        
        # Load ensemble metadata
        ensemble_meta_path = path_prefix.parent / f"{path_prefix.name}_meta.pkl"
        if ensemble_meta_path.exists():
            ensemble_meta = joblib.load(ensemble_meta_path)
            self.feature_names = ensemble_meta['feature_names']
            self.training_metrics = ensemble_meta['training_metrics']
            self.is_fitted = ensemble_meta['is_fitted']
            
            # Load hyperparameters if available
            if 'hyperparameters' in ensemble_meta:
                hyper = ensemble_meta['hyperparameters']
                self.n_estimators = hyper.get('n_estimators', 100)
                self.learning_rate = hyper.get('learning_rate', 0.05)
                self.max_depth = hyper.get('max_depth', 5)
                self.random_state = hyper.get('random_state', 42)
        else:
            # Fallback: load from first model
            model_path = path_prefix.parent / f"{path_prefix.name}_p10.pkl"
            model_data = joblib.load(model_path)
            self.feature_names = model_data['feature_names']
            self.training_metrics = model_data.get('training_metrics', {})
            self.is_fitted = True
    
    def get_ensemble_info(self) -> Dict:
        """
        Get information about the ensemble.
        
        Returns
        -------
        dict
            Ensemble information including hyperparameters and training metrics
        """
        info = {
            'is_fitted': self.is_fitted,
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names,
            'quantiles': list(self.models.keys()),
            'hyperparameters': {
                'n_estimators': self.n_estimators,
                'learning_rate': self.learning_rate,
                'max_depth': self.max_depth,
                'random_state': self.random_state
            }
        }
        
        if self.is_fitted:
            info['training_metrics'] = self.training_metrics
        
        return info


def train_quantile_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    n_estimators: int = 100,
    learning_rate: float = 0.05,
    max_depth: int = 5,
    save_path_prefix: Optional[Union[str, Path]] = None
) -> Tuple[QuantileRegressorEnsemble, Dict[str, float]]:
    """
    Convenience function to train a quantile ensemble.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target (R_multiple)
    X_val : pd.DataFrame, optional
        Validation features
    y_val : pd.Series, optional
        Validation target
    n_estimators : int, default=100
        Number of boosting iterations
    learning_rate : float, default=0.05
        Learning rate
    max_depth : int, default=5
        Maximum tree depth
    save_path_prefix : str or Path, optional
        Path prefix to save the trained models
    
    Returns
    -------
    tuple
        (ensemble, metrics) - Trained ensemble and training metrics
    """
    ensemble = QuantileRegressorEnsemble(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth
    )
    
    metrics = ensemble.fit(X_train, y_train, X_val, y_val)
    
    if save_path_prefix is not None:
        ensemble.save(save_path_prefix)
    
    return ensemble, metrics
