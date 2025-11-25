"""
Interactive Filtering System for Trading Probability Explorer

This module provides a comprehensive filtering system for trade data analysis.
It allows users to apply multiple filters dynamically and see real-time updates
of metrics and visualizations.

Author: Trading Probability Explorer Team
Date: 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Callable, Any, Optional, Tuple
from datetime import datetime, time
import json
import os


class InteractiveFilter:
    """
    Interactive filtering system for trade data
    
    This class manages multiple filters that can be applied to trade data,
    providing real-time updates and filter management capabilities.
    
    Attributes:
        original_trades (pd.DataFrame): Original unfiltered trade data
        filtered_trades (pd.DataFrame): Currently filtered trade data
        active_filters (Dict): Dictionary of active filters with their configurations
    """
    
    def __init__(self, trades_df: pd.DataFrame):
        """
        Initialize the InteractiveFilter with trade data
        
        Args:
            trades_df: DataFrame containing trade data
        """
        if trades_df is None or len(trades_df) == 0:
            raise ValueError("trades_df cannot be None or empty")
        
        self.original_trades = trades_df.copy()
        self.filtered_trades = trades_df.copy()
        self.active_filters = {}
        self._filter_presets = {}
        self._load_default_presets()
    
    def add_filter(self, filter_name: str, filter_func: Callable, filter_params: Dict[str, Any]) -> None:
        """
        Add a filter to the active filter set
        
        Args:
            filter_name: Unique name for the filter
            filter_func: Function that takes DataFrame and params, returns filtered DataFrame
            filter_params: Dictionary of parameters for the filter function
        """
        if not callable(filter_func):
            raise ValueError("filter_func must be callable")
        
        self.active_filters[filter_name] = {
            'func': filter_func,
            'params': filter_params
        }
        self._apply_all_filters()
    
    def remove_filter(self, filter_name: str) -> None:
        """
        Remove a filter from active set
        
        Args:
            filter_name: Name of the filter to remove
        """
        if filter_name in self.active_filters:
            del self.active_filters[filter_name]
            self._apply_all_filters()
    
    def update_filter(self, filter_name: str, new_params: Dict[str, Any]) -> None:
        """
        Update parameters of existing filter
        
        Args:
            filter_name: Name of the filter to update
            new_params: New parameters for the filter
        """
        if filter_name in self.active_filters:
            self.active_filters[filter_name]['params'] = new_params
            self._apply_all_filters()
    
    def clear_all_filters(self) -> None:
        """Clear all active filters and reset to original data"""
        self.active_filters = {}
        self.filtered_trades = self.original_trades.copy()
    
    def _apply_all_filters(self) -> None:
        """Apply all active filters sequentially"""
        self.filtered_trades = self.original_trades.copy()
        for filter_name, filter_config in self.active_filters.items():
            try:
                self.filtered_trades = filter_config['func'](
                    self.filtered_trades, 
                    **filter_config['params']
                )
            except Exception as e:
                print(f"Error applying filter '{filter_name}': {str(e)}")
                # Continue with other filters
    
    def get_filter_summary(self) -> Dict[str, Any]:
        """
        Get summary of active filters and their impact
        
        Returns:
            Dictionary containing filter statistics
        """
        original_count = len(self.original_trades)
        filtered_count = len(self.filtered_trades)
        removed_count = original_count - filtered_count
        
        return {
            'original_count': original_count,
            'filtered_count': filtered_count,
            'removed_count': removed_count,
            'removal_percentage': (removed_count / original_count * 100) if original_count > 0 else 0,
            'active_filters': list(self.active_filters.keys()),
            'filter_count': len(self.active_filters)
        }
    
    def get_filtered_data(self) -> pd.DataFrame:
        """
        Get the currently filtered DataFrame
        
        Returns:
            Filtered DataFrame
        """
        return self.filtered_trades.copy()
    
    # ==================== FILTER FUNCTIONS ====================
    
    @staticmethod
    def date_range_filter(df: pd.DataFrame, start_date: Optional[str] = None, 
                         end_date: Optional[str] = None, 
                         date_column: str = 'entry_time') -> pd.DataFrame:
        """
        Filter trades by date range
        
        Args:
            df: DataFrame to filter
            start_date: Start date (YYYY-MM-DD format or None for no lower bound)
            end_date: End date (YYYY-MM-DD format or None for no upper bound)
            date_column: Name of the date column to filter on
        
        Returns:
            Filtered DataFrame
        """
        if date_column not in df.columns:
            return df
        
        result = df.copy()
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(result[date_column]):
            result[date_column] = pd.to_datetime(result[date_column])
        
        if start_date is not None:
            start_dt = pd.to_datetime(start_date)
            result = result[result[date_column] >= start_dt]
        
        if end_date is not None:
            end_dt = pd.to_datetime(end_date)
            result = result[result[date_column] <= end_dt]
        
        return result
    
    @staticmethod
    def session_filter(df: pd.DataFrame, sessions: List[str], 
                      session_column: str = 'entry_session') -> pd.DataFrame:
        """
        Filter trades by trading session
        
        Args:
            df: DataFrame to filter
            sessions: List of sessions to include (e.g., ['ASIA', 'EUROPE', 'US', 'OVERLAP'])
            session_column: Name of the session column
        
        Returns:
            Filtered DataFrame
        """
        if session_column not in df.columns or not sessions:
            return df
        
        return df[df[session_column].isin(sessions)].copy()
    
    @staticmethod
    def probability_range_filter(df: pd.DataFrame, min_prob: float = 0.0, 
                                 max_prob: float = 1.0, 
                                 prob_column: str = 'prob_global_win') -> pd.DataFrame:
        """
        Filter trades by probability range
        
        Args:
            df: DataFrame to filter
            min_prob: Minimum probability (0.0 to 1.0)
            max_prob: Maximum probability (0.0 to 1.0)
            prob_column: Name of the probability column
        
        Returns:
            Filtered DataFrame
        """
        if prob_column not in df.columns:
            return df
        
        return df[(df[prob_column] >= min_prob) & (df[prob_column] <= max_prob)].copy()
    
    @staticmethod
    def composite_score_filter(df: pd.DataFrame, min_score: float = 0.0, 
                               score_column: str = 'composite_score') -> pd.DataFrame:
        """
        Filter trades by minimum composite score
        
        Args:
            df: DataFrame to filter
            min_score: Minimum composite score (0 to 100)
            score_column: Name of the composite score column
        
        Returns:
            Filtered DataFrame
        """
        if score_column not in df.columns:
            return df
        
        return df[df[score_column] >= min_score].copy()
    
    @staticmethod
    def market_condition_filters(df: pd.DataFrame, 
                                 trend_regimes: Optional[List[int]] = None,
                                 volatility_regimes: Optional[List[int]] = None,
                                 risk_regimes: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Filter trades by market conditions
        
        Args:
            df: DataFrame to filter
            trend_regimes: List of trend regimes to include (0=ranging, 1=trending)
            volatility_regimes: List of volatility regimes (0=low, 1=medium, 2=high)
            risk_regimes: List of risk regimes (0=risk-on, 1=risk-off)
        
        Returns:
            Filtered DataFrame
        """
        result = df.copy()
        
        if trend_regimes is not None and 'trend_regime' in result.columns:
            result = result[result['trend_regime'].isin(trend_regimes)]
        
        if volatility_regimes is not None and 'volatility_regime' in result.columns:
            result = result[result['volatility_regime'].isin(volatility_regimes)]
        
        if risk_regimes is not None and 'risk_regime_global' in result.columns:
            result = result[result['risk_regime_global'].isin(risk_regimes)]
        
        return result
    
    @staticmethod
    def technical_filters(df: pd.DataFrame,
                         trend_strength_range: Optional[Tuple[float, float]] = None,
                         atr_range: Optional[Tuple[float, float]] = None,
                         entropy_range: Optional[Tuple[float, float]] = None,
                         hurst_range: Optional[Tuple[float, float]] = None) -> pd.DataFrame:
        """
        Filter trades by technical indicators
        
        Args:
            df: DataFrame to filter
            trend_strength_range: (min, max) for trend strength
            atr_range: (min, max) for ATR
            entropy_range: (min, max) for entropy
            hurst_range: (min, max) for Hurst exponent
        
        Returns:
            Filtered DataFrame
        """
        result = df.copy()
        
        if trend_strength_range is not None and 'trend_strength_tf' in result.columns:
            min_val, max_val = trend_strength_range
            result = result[(result['trend_strength_tf'] >= min_val) & 
                          (result['trend_strength_tf'] <= max_val)]
        
        if atr_range is not None and 'atr_tf_14' in result.columns:
            min_val, max_val = atr_range
            result = result[(result['atr_tf_14'] >= min_val) & 
                          (result['atr_tf_14'] <= max_val)]
        
        if entropy_range is not None and 'ap_entropy_m1_2h' in result.columns:
            min_val, max_val = entropy_range
            result = result[(result['ap_entropy_m1_2h'] >= min_val) & 
                          (result['ap_entropy_m1_2h'] <= max_val)]
        
        if hurst_range is not None and 'hurst_m5_2d' in result.columns:
            min_val, max_val = hurst_range
            result = result[(result['hurst_m5_2d'] >= min_val) & 
                          (result['hurst_m5_2d'] <= max_val)]
        
        return result
    
    @staticmethod
    def time_of_day_filter(df: pd.DataFrame,
                          hour_range: Optional[Tuple[int, int]] = None,
                          days_of_week: Optional[List[int]] = None,
                          time_column: str = 'entry_time') -> pd.DataFrame:
        """
        Filter trades by time of day and day of week
        
        Args:
            df: DataFrame to filter
            hour_range: (start_hour, end_hour) in 24-hour format (0-23)
            days_of_week: List of days to include (0=Monday, 6=Sunday)
            time_column: Name of the timestamp column
        
        Returns:
            Filtered DataFrame
        """
        if time_column not in df.columns:
            return df
        
        result = df.copy()
        
        # Ensure datetime type
        if not pd.api.types.is_datetime64_any_dtype(result[time_column]):
            result[time_column] = pd.to_datetime(result[time_column])
        
        if hour_range is not None:
            start_hour, end_hour = hour_range
            hours = result[time_column].dt.hour
            if start_hour <= end_hour:
                result = result[(hours >= start_hour) & (hours <= end_hour)]
            else:
                # Handle overnight range (e.g., 22-2)
                result = result[(hours >= start_hour) | (hours <= end_hour)]
        
        if days_of_week is not None:
            result = result[result[time_column].dt.dayofweek.isin(days_of_week)]
        
        return result
    
    @staticmethod
    def performance_filters(df: pd.DataFrame,
                           r_multiple_range: Optional[Tuple[float, float]] = None,
                           holding_time_range: Optional[Tuple[int, int]] = None,
                           profit_loss_range: Optional[Tuple[float, float]] = None) -> pd.DataFrame:
        """
        Filter trades by performance metrics
        
        Args:
            df: DataFrame to filter
            r_multiple_range: (min, max) for R-multiple
            holding_time_range: (min, max) for holding time in minutes
            profit_loss_range: (min, max) for profit/loss in dollars
        
        Returns:
            Filtered DataFrame
        """
        result = df.copy()
        
        if r_multiple_range is not None and 'R_multiple' in result.columns:
            min_val, max_val = r_multiple_range
            result = result[(result['R_multiple'] >= min_val) & 
                          (result['R_multiple'] <= max_val)]
        
        if holding_time_range is not None and 'holding_minutes' in result.columns:
            min_val, max_val = holding_time_range
            result = result[(result['holding_minutes'] >= min_val) & 
                          (result['holding_minutes'] <= max_val)]
        
        if profit_loss_range is not None and 'net_profit' in result.columns:
            min_val, max_val = profit_loss_range
            result = result[(result['net_profit'] >= min_val) & 
                          (result['net_profit'] <= max_val)]
        
        return result
    
    # ==================== PRESET MANAGEMENT ====================
    
    def _load_default_presets(self) -> None:
        """Load default filter presets"""
        self._filter_presets = {
            'high_probability': {
                'description': 'High probability trades only',
                'filters': {
                    'probability_filter': {
                        'func': 'probability_range_filter',
                        'params': {'min_prob': 0.65, 'max_prob': 1.0}
                    },
                    'score_filter': {
                        'func': 'composite_score_filter',
                        'params': {'min_score': 70}
                    }
                }
            },
            'trending_markets': {
                'description': 'Trending markets only',
                'filters': {
                    'trend_regime_filter': {
                        'func': 'market_condition_filters',
                        'params': {'trend_regimes': [1]}
                    },
                    'trend_strength_filter': {
                        'func': 'technical_filters',
                        'params': {'trend_strength_range': (0.3, 1.0)}
                    }
                }
            },
            'low_volatility': {
                'description': 'Low volatility conditions',
                'filters': {
                    'vol_regime_filter': {
                        'func': 'market_condition_filters',
                        'params': {'volatility_regimes': [0]}
                    }
                }
            },
            'europe_session': {
                'description': 'Europe session trades',
                'filters': {
                    'session_filter': {
                        'func': 'session_filter',
                        'params': {'sessions': ['EUROPE']}
                    },
                    'time_filter': {
                        'func': 'time_of_day_filter',
                        'params': {'hour_range': (8, 16)}
                    }
                }
            },
            'high_quality': {
                'description': 'High quality setups',
                'filters': {
                    'score_filter': {
                        'func': 'composite_score_filter',
                        'params': {'min_score': 80}
                    },
                    'probability_filter': {
                        'func': 'probability_range_filter',
                        'params': {'min_prob': 0.7, 'max_prob': 1.0}
                    }
                }
            },
            'conservative': {
                'description': 'Conservative trades',
                'filters': {
                    'r_filter': {
                        'func': 'performance_filters',
                        'params': {'r_multiple_range': (0, 10)}
                    },
                    'holding_time_filter': {
                        'func': 'performance_filters',
                        'params': {'holding_time_range': (0, 240)}
                    }
                }
            },
            'winners_only': {
                'description': 'Winning trades only',
                'filters': {
                    'r_filter': {
                        'func': 'performance_filters',
                        'params': {'r_multiple_range': (0, 100)}
                    }
                }
            },
            'losers_only': {
                'description': 'Losing trades only',
                'filters': {
                    'r_filter': {
                        'func': 'performance_filters',
                        'params': {'r_multiple_range': (-100, 0)}
                    }
                }
            }
        }
    
    def save_filter_preset(self, preset_name: str, description: str = "") -> None:
        """
        Save current filter configuration as preset
        
        Args:
            preset_name: Name for the preset
            description: Optional description of the preset
        """
        if not preset_name:
            raise ValueError("preset_name cannot be empty")
        
        # Convert active filters to serializable format
        preset_config = {
            'description': description,
            'filters': {}
        }
        
        for filter_name, filter_config in self.active_filters.items():
            # Get function name
            func_name = filter_config['func'].__name__
            preset_config['filters'][filter_name] = {
                'func': func_name,
                'params': filter_config['params']
            }
        
        self._filter_presets[preset_name] = preset_config
    
    def load_filter_preset(self, preset_name: str) -> None:
        """
        Load saved filter preset
        
        Args:
            preset_name: Name of the preset to load
        """
        if preset_name not in self._filter_presets:
            raise ValueError(f"Preset '{preset_name}' not found")
        
        # Clear existing filters
        self.clear_all_filters()
        
        # Load preset filters
        preset = self._filter_presets[preset_name]
        for filter_name, filter_config in preset['filters'].items():
            func_name = filter_config['func']
            params = filter_config['params']
            
            # Get the actual function
            filter_func = getattr(InteractiveFilter, func_name, None)
            if filter_func is None:
                print(f"Warning: Filter function '{func_name}' not found")
                continue
            
            self.add_filter(filter_name, filter_func, params)
    
    def get_available_presets(self) -> Dict[str, str]:
        """
        Get list of available presets with descriptions
        
        Returns:
            Dictionary mapping preset names to descriptions
        """
        return {name: config.get('description', '') 
                for name, config in self._filter_presets.items()}
    
    def export_preset_to_file(self, preset_name: str, filepath: str) -> None:
        """
        Export a preset to a JSON file
        
        Args:
            preset_name: Name of the preset to export
            filepath: Path to save the JSON file
        """
        if preset_name not in self._filter_presets:
            raise ValueError(f"Preset '{preset_name}' not found")
        
        with open(filepath, 'w') as f:
            json.dump(self._filter_presets[preset_name], f, indent=2)
    
    def import_preset_from_file(self, preset_name: str, filepath: str) -> None:
        """
        Import a preset from a JSON file
        
        Args:
            preset_name: Name to give the imported preset
            filepath: Path to the JSON file
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, 'r') as f:
            preset_config = json.load(f)
        
        self._filter_presets[preset_name] = preset_config
