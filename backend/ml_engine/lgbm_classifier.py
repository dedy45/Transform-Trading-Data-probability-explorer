"""
LightGBM Binary Classifier for Win/Loss Prediction

This module implements a wrapper around LightGBM classifier for predicting
binary win/loss outcomes in trading setups.

**Feature: ml-prediction-engine**
**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5**
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Optional, Dict, Tuple, Union
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, brier_score_loss
import lightgbm as lgb


class LGBMClassifierWrapper:
    """
    Wrapper for LightGBM binary classifier with time-series cross-validation.
    
    This class provides a clean interface for training, predicting, and persisting
    a LightGBM classifier for binary win/loss prediction.
    
    Attributes
    ----------
    model : lgb.LGBMClassifier
        The underlying LightGBM classifier
    feature_names : list
        Names of features used for training
    feature_importance : pd.DataFrame
        Feature importance scores from the trained model
    training_metrics : dict
        Metrics from the training process
    is_fitted : bool
        Whether the model has been fitted
    
    Examples
    --------
    >>> classifier = LGBMClassifierWrapper(n_estimators=100, learning_rate=0.05)
    >>> metrics = classifier.fit(X_train, y_train, X_val, y_val)
    >>> probs = classifier.predict_proba(X_test)
    >>> classifier.save('models/classifier.pkl')
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
        Initialize LightGBM classifier wrapper.
        
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
            Additional parameters for LGBMClassifier
        """
        self.model = lgb.LGBMClassifier(
            objective='binary',
            metric='auc',
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            verbose=-1,  # Suppress warnings
            **kwargs
        )
        
        self.feature_names = None
        self.feature_importance = None
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
        y_val: Optional[pd.Series] = None,
        use_cv: bool = True,
        n_splits: int = 5
    ) -> Dict[str, float]:
        """
        Train classifier with optional time-series cross-validation.
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target (0/1 for loss/win)
        X_val : pd.DataFrame, optional
            Validation features
        y_val : pd.Series, optional
            Validation target
        use_cv : bool, default=True
            Whether to use time-series cross-validation
        n_splits : int, default=5
            Number of CV folds
        
        Returns
        -------
        dict
            Training metrics including:
            - auc_train: AUC on training set
            - auc_val: AUC on validation set (if provided)
            - auc_cv_mean: Mean AUC across CV folds (if use_cv=True)
            - auc_cv_std: Std AUC across CV folds (if use_cv=True)
            - brier_train: Brier score on training set
            - brier_val: Brier score on validation set (if provided)
        
        Raises
        ------
        ValueError
            If X_train is empty or y_train has wrong shape
        """
        # Validate inputs
        if len(X_train) == 0:
            raise ValueError("X_train cannot be empty")
        
        if len(X_train) != len(y_train):
            raise ValueError(f"X_train and y_train must have same length: {len(X_train)} != {len(y_train)}")
        
        # Store feature names
        self.feature_names = list(X_train.columns)
        
        # Time-series cross-validation
        cv_scores = []
        if use_cv and len(X_train) >= 100:  # Only do CV if enough data
            tscv = TimeSeriesSplit(n_splits=n_splits)
            
            for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
                X_fold_train = X_train.iloc[train_idx]
                y_fold_train = y_train.iloc[train_idx]
                X_fold_val = X_train.iloc[val_idx]
                y_fold_val = y_train.iloc[val_idx]
                
                # Train fold model
                fold_model = lgb.LGBMClassifier(
                    objective='binary',
                    metric='auc',
                    n_estimators=self.n_estimators,
                    learning_rate=self.learning_rate,
                    max_depth=self.max_depth,
                    random_state=self.random_state,
                    verbose=-1
                )
                
                fold_model.fit(X_fold_train, y_fold_train)
                
                # Evaluate fold
                y_fold_pred = fold_model.predict_proba(X_fold_val)[:, 1]
                fold_auc = roc_auc_score(y_fold_val, y_fold_pred)
                cv_scores.append(fold_auc)
        
        # Train final model on full training set
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set if len(eval_set) > 1 else None,
            eval_metric='auc'
        )
        
        self.is_fitted = True
        
        # Calculate training metrics
        y_train_pred = self.model.predict_proba(X_train)[:, 1]
        auc_train = roc_auc_score(y_train, y_train_pred)
        brier_train = brier_score_loss(y_train, y_train_pred)
        
        self.training_metrics = {
            'auc_train': auc_train,
            'brier_train': brier_train,
            'n_features': len(self.feature_names),
            'n_samples_train': len(X_train)
        }
        
        # Add CV metrics if available
        if cv_scores:
            self.training_metrics['auc_cv_mean'] = np.mean(cv_scores)
            self.training_metrics['auc_cv_std'] = np.std(cv_scores)
            self.training_metrics['auc_cv_scores'] = cv_scores
        
        # Add validation metrics if available
        if X_val is not None and y_val is not None:
            y_val_pred = self.model.predict_proba(X_val)[:, 1]
            auc_val = roc_auc_score(y_val, y_val_pred)
            brier_val = brier_score_loss(y_val, y_val_pred)
            
            self.training_metrics['auc_val'] = auc_val
            self.training_metrics['brier_val'] = brier_val
            self.training_metrics['n_samples_val'] = len(X_val)
        
        # Extract feature importance
        self._extract_feature_importance()
        
        return self.training_metrics
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict raw probability (before calibration).
        
        Parameters
        ----------
        X : pd.DataFrame
            Features for prediction
        
        Returns
        -------
        np.ndarray
            Raw probabilities (0-1) for positive class
        
        Raises
        ------
        ValueError
            If model is not fitted or features don't match
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")
        
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
        
        # Predict probabilities for positive class (win)
        probs = self.model.predict_proba(X)[:, 1]
        
        return probs
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary class labels.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features for prediction
        threshold : float, default=0.5
            Probability threshold for positive class
        
        Returns
        -------
        np.ndarray
            Binary predictions (0/1)
        """
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
    
    def get_feature_importance(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """
        Get feature importance from trained model.
        
        Parameters
        ----------
        top_n : int, optional
            Return only top N features. If None, return all.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: feature, importance, rank
            Sorted by importance (descending)
        
        Raises
        ------
        ValueError
            If model is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        if self.feature_importance is None:
            self._extract_feature_importance()
        
        if top_n is not None:
            return self.feature_importance.head(top_n)
        
        return self.feature_importance
    
    def _extract_feature_importance(self):
        """Extract and store feature importance from model."""
        if not self.is_fitted:
            return
        
        importance_values = self.model.feature_importances_
        
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_values
        })
        
        # Sort by importance and add rank
        self.feature_importance = self.feature_importance.sort_values(
            'importance',
            ascending=False
        ).reset_index(drop=True)
        
        self.feature_importance['rank'] = range(1, len(self.feature_importance) + 1)
    
    def save(self, path: Union[str, Path]):
        """
        Save model to file using joblib.
        
        Parameters
        ----------
        path : str or Path
            Path to save the model
        
        Raises
        ------
        ValueError
            If model is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model. Call fit() first.")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model and metadata
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'training_metrics': self.training_metrics,
            'is_fitted': self.is_fitted,
            'hyperparameters': {
                'n_estimators': self.n_estimators,
                'learning_rate': self.learning_rate,
                'max_depth': self.max_depth,
                'random_state': self.random_state
            }
        }
        
        joblib.dump(model_data, path)
    
    def load(self, path: Union[str, Path]):
        """
        Load model from file.
        
        Parameters
        ----------
        path : str or Path
            Path to load the model from
        
        Raises
        ------
        FileNotFoundError
            If model file doesn't exist
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # Load model and metadata
        model_data = joblib.load(path)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.feature_importance = model_data['feature_importance']
        self.training_metrics = model_data['training_metrics']
        self.is_fitted = model_data['is_fitted']
        
        # Load hyperparameters if available
        if 'hyperparameters' in model_data:
            hyper = model_data['hyperparameters']
            self.n_estimators = hyper.get('n_estimators', 100)
            self.learning_rate = hyper.get('learning_rate', 0.05)
            self.max_depth = hyper.get('max_depth', 5)
            self.random_state = hyper.get('random_state', 42)
    
    def get_model_info(self) -> Dict:
        """
        Get information about the model.
        
        Returns
        -------
        dict
            Model information including hyperparameters and training metrics
        """
        info = {
            'is_fitted': self.is_fitted,
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names,
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


def train_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    n_estimators: int = 100,
    learning_rate: float = 0.05,
    max_depth: int = 5,
    use_cv: bool = True,
    n_splits: int = 5,
    save_path: Optional[Union[str, Path]] = None
) -> Tuple[LGBMClassifierWrapper, Dict[str, float]]:
    """
    Convenience function to train a classifier.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
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
    use_cv : bool, default=True
        Whether to use cross-validation
    n_splits : int, default=5
        Number of CV folds
    save_path : str or Path, optional
        Path to save the trained model
    
    Returns
    -------
    tuple
        (classifier, metrics) - Trained classifier and training metrics
    """
    classifier = LGBMClassifierWrapper(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth
    )
    
    metrics = classifier.fit(
        X_train,
        y_train,
        X_val,
        y_val,
        use_cv=use_cv,
        n_splits=n_splits
    )
    
    if save_path is not None:
        classifier.save(save_path)
    
    return classifier, metrics
