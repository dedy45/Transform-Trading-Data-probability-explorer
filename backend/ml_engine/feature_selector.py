"""
Feature Selector Integration

Integrates with Auto Feature Selection to load selected features for ML models.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import os
import json
from pathlib import Path


class FeatureSelector:
    """
    Load and validate features from Auto Feature Selection results.
    """
    
    def __init__(self, min_features: int = 5, max_features: int = 15):
        """
        Initialize FeatureSelector.
        
        Parameters
        ----------
        min_features : int
            Minimum number of features required (default: 5)
        max_features : int
            Maximum number of features to select (default: 15)
        """
        self.min_features = min_features
        self.max_features = max_features
        self.selected_features = []
        self.feature_scores = {}
        
    def load_selected_features(
        self,
        results: Optional[Dict] = None,
        feature_list_path: Optional[str] = None,
        composite_score_threshold: float = 0.6
    ) -> List[str]:
        """
        Load top 5-8 features with composite_score > threshold.
        
        This function can load features from:
        1. Auto Feature Selection results dict (in-memory)
        2. Saved feature list JSON file
        
        Parameters
        ----------
        results : Dict, optional
            Results dict from Auto Feature Selection (contains 'combined_ranking')
        feature_list_path : str, optional
            Path to saved feature_list.json file
        composite_score_threshold : float
            Minimum composite score for feature selection (default: 0.6)
            
        Returns
        -------
        List[str]
            List of selected feature names (5-8 features)
            
        Raises
        ------
        ValueError
            If no valid features found or insufficient features
        FileNotFoundError
            If feature_list_path provided but file not found
        """
        # Option 1: Load from results dict
        if results is not None:
            if 'combined_ranking' not in results:
                raise ValueError(
                    "Results dict must contain 'combined_ranking' key. "
                    "Expected output from run_auto_feature_selection()."
                )
            
            combined_ranking = results['combined_ranking']
            
            # Filter by composite score threshold
            filtered = combined_ranking[
                combined_ranking['composite_score'] > composite_score_threshold
            ].copy()
            
            if len(filtered) == 0:
                raise ValueError(
                    f"No features found with composite_score > {composite_score_threshold}. "
                    f"Try lowering the threshold or check Auto Feature Selection results."
                )
            
            # Select top features (between min and max)
            n_features = min(max(len(filtered), self.min_features), self.max_features)
            selected = filtered.head(n_features)
            
            self.selected_features = selected['feature'].tolist()
            self.feature_scores = dict(zip(
                selected['feature'],
                selected['composite_score']
            ))
            
            return self.selected_features
        
        # Option 2: Load from saved JSON file
        elif feature_list_path is not None:
            if not os.path.exists(feature_list_path):
                raise FileNotFoundError(
                    f"Feature list file not found: {feature_list_path}"
                )
            
            with open(feature_list_path, 'r') as f:
                data = json.load(f)
            
            if 'features' not in data:
                raise ValueError(
                    "Feature list JSON must contain 'features' key. "
                    "Expected format: {'features': [...], 'scores': {...}}"
                )
            
            self.selected_features = data['features']
            self.feature_scores = data.get('scores', {})
            
            # Validate number of features
            if len(self.selected_features) < self.min_features:
                raise ValueError(
                    f"Insufficient features in saved list: {len(self.selected_features)} "
                    f"(minimum required: {self.min_features})"
                )
            
            # Truncate if too many
            if len(self.selected_features) > self.max_features:
                self.selected_features = self.selected_features[:self.max_features]
            
            return self.selected_features
        
        else:
            raise ValueError(
                "Must provide either 'results' dict or 'feature_list_path'. "
                "Both cannot be None."
            )
    
    def validate_features(
        self,
        dataset: pd.DataFrame,
        raise_on_missing: bool = False
    ) -> Tuple[List[str], List[str]]:
        """
        Check existence of selected features in dataset.
        
        Parameters
        ----------
        dataset : pd.DataFrame
            Dataset to validate against
        raise_on_missing : bool
            If True, raise ValueError when features are missing (default: False)
            
        Returns
        -------
        Tuple[List[str], List[str]]
            (present_features, missing_features)
            
        Raises
        ------
        ValueError
            If raise_on_missing=True and features are missing
        """
        if not self.selected_features:
            raise ValueError(
                "No features loaded. Call load_selected_features() first."
            )
        
        dataset_columns = set(dataset.columns)
        
        present_features = []
        missing_features = []
        
        for feature in self.selected_features:
            if feature in dataset_columns:
                present_features.append(feature)
            else:
                missing_features.append(feature)
        
        if missing_features and raise_on_missing:
            raise ValueError(
                f"Missing features in dataset: {missing_features}\n"
                f"Dataset has {len(dataset.columns)} columns.\n"
                f"Expected features: {self.selected_features}"
            )
        
        return present_features, missing_features
    
    def get_feature_info(self) -> Dict:
        """
        Get information about selected features.
        
        Returns
        -------
        Dict
            Information about selected features including scores
        """
        if not self.selected_features:
            return {
                'n_features': 0,
                'features': [],
                'scores': {},
                'status': 'No features loaded'
            }
        
        return {
            'n_features': len(self.selected_features),
            'features': self.selected_features,
            'scores': self.feature_scores,
            'min_score': min(self.feature_scores.values()) if self.feature_scores else None,
            'max_score': max(self.feature_scores.values()) if self.feature_scores else None,
            'avg_score': np.mean(list(self.feature_scores.values())) if self.feature_scores else None,
            'status': 'Features loaded successfully'
        }
    
    def save_feature_list(self, output_path: str) -> None:
        """
        Save selected features to JSON file.
        
        Parameters
        ----------
        output_path : str
            Path to save feature list JSON
        """
        if not self.selected_features:
            raise ValueError(
                "No features to save. Call load_selected_features() first."
            )
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        data = {
            'features': self.selected_features,
            'scores': self.feature_scores,
            'n_features': len(self.selected_features),
            'min_features': self.min_features,
            'max_features': self.max_features
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def filter_dataset(
        self,
        dataset: pd.DataFrame,
        handle_missing: str = 'raise'
    ) -> pd.DataFrame:
        """
        Filter dataset to only include selected features.
        
        Parameters
        ----------
        dataset : pd.DataFrame
            Input dataset
        handle_missing : str
            How to handle missing features:
            - 'raise': Raise error if features missing
            - 'warn': Print warning and use available features
            - 'ignore': Silently use available features
            
        Returns
        -------
        pd.DataFrame
            Filtered dataset with only selected features
            
        Raises
        ------
        ValueError
            If handle_missing='raise' and features are missing
        """
        present_features, missing_features = self.validate_features(
            dataset,
            raise_on_missing=(handle_missing == 'raise')
        )
        
        if missing_features:
            if handle_missing == 'warn':
                print(
                    f"Warning: {len(missing_features)} features missing from dataset: "
                    f"{missing_features[:5]}{'...' if len(missing_features) > 5 else ''}"
                )
                print(f"Using {len(present_features)} available features.")
        
        if not present_features:
            raise ValueError(
                "No selected features found in dataset. "
                "Check that dataset contains the expected columns."
            )
        
        return dataset[present_features].copy()


def load_features_from_config(config_path: str = 'config/ml_prediction_config.yaml') -> List[str]:
    """
    Load selected features from ML prediction config file.
    
    Parameters
    ----------
    config_path : str
        Path to ML prediction config YAML file
        
    Returns
    -------
    List[str]
        List of selected feature names
        
    Raises
    ------
    FileNotFoundError
        If config file not found
    ValueError
        If config file doesn't contain features section
    """
    import yaml
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if 'features' not in config or 'selected' not in config['features']:
        raise ValueError(
            "Config file must contain 'features.selected' section. "
            f"Check format of {config_path}"
        )
    
    return config['features']['selected']


def save_features_to_config(
    features: List[str],
    config_path: str = 'config/ml_prediction_config.yaml'
) -> None:
    """
    Save selected features to ML prediction config file.
    
    Parameters
    ----------
    features : List[str]
        List of feature names to save
    config_path : str
        Path to ML prediction config YAML file
    """
    import yaml
    
    # Load existing config or create new
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # Update features section
    if 'features' not in config:
        config['features'] = {}
    
    config['features']['selected'] = features
    
    # Create directory if needed
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    # Save config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
