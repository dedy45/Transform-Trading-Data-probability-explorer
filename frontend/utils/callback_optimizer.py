"""
Callback Optimizer Module

This module provides utilities for optimizing Dash callbacks,
including debouncing, throttling, and efficient data updates.
"""

import time
from functools import wraps
from typing import Callable, Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class CallbackThrottler:
    """
    Throttle callback execution to prevent excessive updates.
    
    Ensures callback is not executed more than once per time period.
    """
    
    def __init__(self, min_interval: float = 0.5):
        """
        Initialize throttler.
        
        Parameters
        ----------
        min_interval : float
            Minimum time between callback executions (seconds)
        """
        self.min_interval = min_interval
        self.last_execution = {}
    
    def __call__(self, func: Callable) -> Callable:
        """Throttle function execution."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_id = id(func)
            current_time = time.time()
            
            # Check if enough time has passed
            if func_id in self.last_execution:
                elapsed = current_time - self.last_execution[func_id]
                if elapsed < self.min_interval:
                    logger.debug(f"Throttled {func.__name__} (elapsed: {elapsed:.3f}s)")
                    from dash.exceptions import PreventUpdate
                    raise PreventUpdate
            
            # Execute and update timestamp
            self.last_execution[func_id] = current_time
            return func(*args, **kwargs)
        
        return wrapper


class CallbackDebouncer:
    """
    Debounce callback execution to wait for user to finish input.
    
    Delays callback execution until no new calls for specified time.
    """
    
    def __init__(self, wait_time: float = 0.5):
        """
        Initialize debouncer.
        
        Parameters
        ----------
        wait_time : float
            Time to wait after last call before executing (seconds)
        """
        self.wait_time = wait_time
        self.pending_calls = {}
    
    def __call__(self, func: Callable) -> Callable:
        """Debounce function execution."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_id = id(func)
            current_time = time.time()
            
            # Store pending call
            self.pending_calls[func_id] = {
                'time': current_time,
                'args': args,
                'kwargs': kwargs
            }
            
            # Check if we should execute
            time.sleep(self.wait_time)
            
            # Only execute if this is still the latest call
            if func_id in self.pending_calls:
                pending = self.pending_calls[func_id]
                if pending['time'] == current_time:
                    del self.pending_calls[func_id]
                    return func(*args, **kwargs)
            
            from dash.exceptions import PreventUpdate
            raise PreventUpdate
        
        return wrapper


def optimize_figure_update(figure: Dict[str, Any]) -> Dict[str, Any]:
    """
    Optimize Plotly figure for faster rendering.
    
    Parameters
    ----------
    figure : dict
        Plotly figure dictionary
        
    Returns
    -------
    dict
        Optimized figure
    """
    # Reduce number of points if too many
    if 'data' in figure:
        for trace in figure['data']:
            if 'x' in trace and len(trace['x']) > 10000:
                # Downsample to 10000 points
                step = len(trace['x']) // 10000
                trace['x'] = trace['x'][::step]
                if 'y' in trace:
                    trace['y'] = trace['y'][::step]
                logger.info(f"Downsampled trace to {len(trace['x'])} points")
    
    # Optimize layout
    if 'layout' in figure:
        # Disable hover for large datasets
        if 'hovermode' not in figure['layout']:
            figure['layout']['hovermode'] = 'closest'
    
    return figure


def batch_callback_updates(updates: list) -> list:
    """
    Batch multiple callback updates for efficiency.
    
    Parameters
    ----------
    updates : list
        List of update dictionaries
        
    Returns
    -------
    list
        Batched updates
    """
    # Group updates by component
    grouped = {}
    for update in updates:
        component_id = update.get('component_id')
        if component_id not in grouped:
            grouped[component_id] = []
        grouped[component_id].append(update)
    
    # Merge updates for same component
    batched = []
    for component_id, component_updates in grouped.items():
        if len(component_updates) == 1:
            batched.append(component_updates[0])
        else:
            # Merge multiple updates
            merged = component_updates[0].copy()
            for update in component_updates[1:]:
                merged.update(update)
            batched.append(merged)
    
    return batched


def lazy_load_data(data_loader: Callable, cache_key: str) -> Any:
    """
    Lazy load data with caching.
    
    Parameters
    ----------
    data_loader : Callable
        Function to load data
    cache_key : str
        Cache key for storing result
        
    Returns
    -------
    Any
        Loaded data
    """
    from backend.utils.performance_optimizer import _df_cache
    
    # Check cache
    cached_data = _df_cache.get(cache_key)
    if cached_data is not None:
        logger.debug(f"Loaded {cache_key} from cache")
        return cached_data
    
    # Load and cache
    logger.debug(f"Loading {cache_key}...")
    data = data_loader()
    _df_cache.set(cache_key, data)
    
    return data


def incremental_update(
    current_data: Any,
    new_data: Any,
    merge_key: Optional[str] = None
) -> Any:
    """
    Incrementally update data instead of full replacement.
    
    Parameters
    ----------
    current_data : Any
        Current data
    new_data : Any
        New data to merge
    merge_key : str, optional
        Key to use for merging DataFrames
        
    Returns
    -------
    Any
        Updated data
    """
    import pandas as pd
    
    if isinstance(current_data, pd.DataFrame) and isinstance(new_data, pd.DataFrame):
        if merge_key:
            # Merge on key
            return pd.concat([current_data, new_data]).drop_duplicates(
                subset=[merge_key], keep='last'
            )
        else:
            # Append
            return pd.concat([current_data, new_data], ignore_index=True)
    
    # For other types, just return new data
    return new_data


def optimize_table_data(df, max_rows: int = 1000):
    """
    Optimize DataFrame for table display.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to optimize
    max_rows : int
        Maximum rows to display
        
    Returns
    -------
    dict
        Optimized table data
    """
    import pandas as pd
    
    # Limit rows
    if len(df) > max_rows:
        df_display = df.head(max_rows)
        logger.info(f"Limited table to {max_rows} rows (total: {len(df)})")
    else:
        df_display = df
    
    # Convert to records for Dash table
    records = df_display.to_dict('records')
    
    # Round numeric columns
    for record in records:
        for key, value in record.items():
            if isinstance(value, float):
                record[key] = round(value, 4)
    
    return records


def create_loading_state(component_id: str, loading: bool = True) -> Dict[str, Any]:
    """
    Create loading state for component.
    
    Parameters
    ----------
    component_id : str
        Component ID
    loading : bool
        Whether component is loading
        
    Returns
    -------
    dict
        Loading state
    """
    return {
        'component_id': component_id,
        'loading': loading,
        'timestamp': time.time()
    }


def validate_callback_inputs(*inputs) -> bool:
    """
    Validate callback inputs to prevent errors.
    
    Parameters
    ----------
    *inputs
        Callback input values
        
    Returns
    -------
    bool
        True if all inputs are valid
    """
    for inp in inputs:
        if inp is None:
            return False
        if isinstance(inp, (list, dict)) and len(inp) == 0:
            return False
    
    return True


def handle_callback_error(func: Callable) -> Callable:
    """
    Decorator to handle callback errors gracefully.
    
    Parameters
    ----------
    func : Callable
        Callback function
        
    Returns
    -------
    Callable
        Wrapped function with error handling
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in callback {func.__name__}: {e}")
            
            # Return error state
            from dash import html
            return html.Div([
                html.H5("Error", className="text-danger"),
                html.P(f"An error occurred: {str(e)}", className="text-muted")
            ])
    
    return wrapper


# Global throttler and debouncer instances
throttle = CallbackThrottler(min_interval=0.5)
debounce = CallbackDebouncer(wait_time=0.5)
