"""
Performance Optimizer Module

This module provides caching, profiling, and optimization utilities
for the Trading Probability Explorer application.

Features:
- LRU caching for expensive calculations
- Data chunking for large datasets
- Optimized pandas operations
- Memory-efficient data processing
- Performance profiling utilities
"""

import pandas as pd
import numpy as np
from functools import lru_cache, wraps
import time
import hashlib
import pickle
from typing import Callable, Any, Dict, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceCache:
    """
    Advanced caching system for expensive calculations.
    
    Uses LRU cache with custom key generation for pandas DataFrames.
    """
    
    def __init__(self, maxsize: int = 128):
        """
        Initialize performance cache.
        
        Parameters
        ----------
        maxsize : int
            Maximum number of cached results
        """
        self.maxsize = maxsize
        self._cache = {}
        self._access_times = {}
        self._hit_count = 0
        self._miss_count = 0
    
    def _generate_key(self, *args, **kwargs) -> str:
        """
        Generate cache key from arguments.
        
        Handles DataFrames by hashing their content.
        """
        key_parts = []
        
        for arg in args:
            if isinstance(arg, pd.DataFrame):
                # Hash DataFrame content
                key_parts.append(self._hash_dataframe(arg))
            elif isinstance(arg, (list, tuple, dict)):
                key_parts.append(str(sorted(str(arg))))
            else:
                key_parts.append(str(arg))
        
        for k, v in sorted(kwargs.items()):
            if isinstance(v, pd.DataFrame):
                key_parts.append(f"{k}={self._hash_dataframe(v)}")
            else:
                key_parts.append(f"{k}={v}")
        
        return hashlib.md5("_".join(key_parts).encode()).hexdigest()
    
    def _hash_dataframe(self, df: pd.DataFrame) -> str:
        """Hash DataFrame for cache key generation."""
        # Use shape, columns, and sample of data for hash
        hash_input = f"{df.shape}_{list(df.columns)}_{df.head(5).values.tobytes()}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        if key in self._cache:
            self._hit_count += 1
            self._access_times[key] = time.time()
            return self._cache[key]
        self._miss_count += 1
        return None
    
    def set(self, key: str, value: Any):
        """Set cached value with LRU eviction."""
        if len(self._cache) >= self.maxsize:
            # Evict least recently used
            lru_key = min(self._access_times, key=self._access_times.get)
            del self._cache[lru_key]
            del self._access_times[lru_key]
        
        self._cache[key] = value
        self._access_times[key] = time.time()
    
    def clear(self):
        """Clear all cached values."""
        self._cache.clear()
        self._access_times.clear()
        self._hit_count = 0
        self._miss_count = 0
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hit_count + self._miss_count
        hit_rate = self._hit_count / total if total > 0 else 0
        
        return {
            'size': len(self._cache),
            'maxsize': self.maxsize,
            'hits': self._hit_count,
            'misses': self._miss_count,
            'hit_rate': hit_rate
        }


# Global cache instance
_global_cache = PerformanceCache(maxsize=256)


def cached_calculation(cache: Optional[PerformanceCache] = None):
    """
    Decorator for caching expensive calculations.
    
    Parameters
    ----------
    cache : PerformanceCache, optional
        Cache instance to use. If None, uses global cache.
    
    Examples
    --------
    >>> @cached_calculation()
    ... def expensive_function(df, param):
    ...     # Expensive calculation
    ...     return result
    """
    if cache is None:
        cache = _global_cache
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key = cache._generate_key(func.__name__, *args, **kwargs)
            
            # Check cache
            cached_result = cache.get(key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Calculate and cache
            logger.debug(f"Cache miss for {func.__name__}, calculating...")
            result = func(*args, **kwargs)
            cache.set(key, result)
            
            return result
        
        return wrapper
    return decorator


def profile_performance(func: Callable) -> Callable:
    """
    Decorator to profile function performance.
    
    Logs execution time and memory usage.
    
    Examples
    --------
    >>> @profile_performance
    ... def my_function(df):
    ...     return df.groupby('col').mean()
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        import psutil
        import os
        
        # Get initial memory
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Time execution
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Get final memory
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_delta = mem_after - mem_before
        
        # Log results
        logger.info(
            f"Performance: {func.__name__} | "
            f"Time: {end_time - start_time:.3f}s | "
            f"Memory: {mem_delta:+.2f}MB"
        )
        
        return result
    
    return wrapper


def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by downcasting numeric types.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to optimize
        
    Returns
    -------
    pd.DataFrame
        Optimized DataFrame with reduced memory footprint
        
    Examples
    --------
    >>> df_optimized = optimize_dataframe_memory(df)
    >>> # Memory usage reduced by 50-70% typically
    """
    df = df.copy()
    
    # Get initial memory usage
    mem_before = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
    
    # Optimize numeric columns
    for col in df.select_dtypes(include=['int']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    # Convert object columns to category if beneficial
    for col in df.select_dtypes(include=['object']).columns:
        num_unique = df[col].nunique()
        num_total = len(df[col])
        
        # Convert to category if less than 50% unique values
        if num_unique / num_total < 0.5:
            df[col] = df[col].astype('category')
    
    # Get final memory usage
    mem_after = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
    reduction = (1 - mem_after / mem_before) * 100
    
    logger.info(
        f"Memory optimization: {mem_before:.2f}MB → {mem_after:.2f}MB "
        f"({reduction:.1f}% reduction)"
    )
    
    return df


def chunk_dataframe(df: pd.DataFrame, chunk_size: int = 10000) -> list:
    """
    Split DataFrame into chunks for processing large datasets.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to chunk
    chunk_size : int
        Number of rows per chunk
        
    Returns
    -------
    list of pd.DataFrame
        List of DataFrame chunks
        
    Examples
    --------
    >>> chunks = chunk_dataframe(large_df, chunk_size=10000)
    >>> results = [process_chunk(chunk) for chunk in chunks]
    >>> final_result = pd.concat(results)
    """
    num_chunks = len(df) // chunk_size + (1 if len(df) % chunk_size else 0)
    chunks = []
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(df))
        chunks.append(df.iloc[start_idx:end_idx])
    
    logger.info(f"Split DataFrame into {num_chunks} chunks of ~{chunk_size} rows")
    
    return chunks


def parallel_apply(df: pd.DataFrame, func: Callable, n_jobs: int = -1) -> pd.DataFrame:
    """
    Apply function to DataFrame in parallel.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to process
    func : Callable
        Function to apply to each chunk
    n_jobs : int
        Number of parallel jobs (-1 for all CPUs)
        
    Returns
    -------
    pd.DataFrame
        Processed DataFrame
        
    Examples
    --------
    >>> def process_row(row):
    ...     return row['col1'] * row['col2']
    >>> result = parallel_apply(df, process_row)
    """
    from multiprocessing import Pool, cpu_count
    
    if n_jobs == -1:
        n_jobs = cpu_count()
    
    # Split into chunks
    chunk_size = len(df) // n_jobs + 1
    chunks = chunk_dataframe(df, chunk_size)
    
    # Process in parallel
    with Pool(n_jobs) as pool:
        results = pool.map(func, chunks)
    
    # Combine results
    return pd.concat(results, ignore_index=True)


def optimize_groupby(df: pd.DataFrame, groupby_cols: list, agg_dict: dict) -> pd.DataFrame:
    """
    Optimized groupby operation for large datasets.
    
    Uses categorical dtypes and efficient aggregation.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to group
    groupby_cols : list
        Columns to group by
    agg_dict : dict
        Aggregation dictionary
        
    Returns
    -------
    pd.DataFrame
        Grouped DataFrame
        
    Examples
    --------
    >>> result = optimize_groupby(
    ...     df,
    ...     ['category', 'subcategory'],
    ...     {'value': ['mean', 'std'], 'count': 'sum'}
    ... )
    """
    df = df.copy()
    
    # Convert groupby columns to category for faster grouping
    for col in groupby_cols:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category')
    
    # Perform groupby
    result = df.groupby(groupby_cols, observed=True).agg(agg_dict)
    
    return result


def batch_probability_calculation(
    df: pd.DataFrame,
    target: str,
    features: list,
    calc_func: Callable,
    batch_size: int = 5
) -> Dict[str, Any]:
    """
    Calculate probabilities for multiple features in batches.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset
    target : str
        Target column
    features : list
        List of feature columns
    calc_func : Callable
        Probability calculation function
    batch_size : int
        Number of features to process at once
        
    Returns
    -------
    dict
        Results for each feature
        
    Examples
    --------
    >>> from backend.calculators.probability_calculator import compute_1d_probability
    >>> results = batch_probability_calculation(
    ...     df, 'y_win', feature_list, compute_1d_probability
    ... )
    """
    results = {}
    
    for i in range(0, len(features), batch_size):
        batch = features[i:i + batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}: {batch}")
        
        for feature in batch:
            try:
                results[feature] = calc_func(df, target, feature)
            except Exception as e:
                logger.error(f"Error processing {feature}: {e}")
                results[feature] = None
    
    return results


class DataFrameCache:
    """
    Specialized cache for DataFrame operations.
    
    Stores DataFrames efficiently using pickle and tracks memory usage.
    """
    
    def __init__(self, max_memory_mb: int = 500):
        """
        Initialize DataFrame cache.
        
        Parameters
        ----------
        max_memory_mb : int
            Maximum memory to use for cache (MB)
        """
        self.max_memory_mb = max_memory_mb
        self._cache = {}
        self._memory_usage = {}
    
    def set(self, key: str, df: pd.DataFrame):
        """Cache DataFrame."""
        # Calculate memory usage
        mem_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        # Check if we need to evict
        total_mem = sum(self._memory_usage.values())
        if total_mem + mem_mb > self.max_memory_mb:
            # Evict largest item
            if self._memory_usage:
                largest_key = max(self._memory_usage, key=self._memory_usage.get)
                del self._cache[largest_key]
                del self._memory_usage[largest_key]
                logger.info(f"Evicted {largest_key} from cache")
        
        # Store DataFrame
        self._cache[key] = df
        self._memory_usage[key] = mem_mb
        logger.debug(f"Cached {key} ({mem_mb:.2f}MB)")
    
    def get(self, key: str) -> Optional[pd.DataFrame]:
        """Get cached DataFrame."""
        return self._cache.get(key)
    
    def clear(self):
        """Clear cache."""
        self._cache.clear()
        self._memory_usage.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'items': len(self._cache),
            'total_memory_mb': sum(self._memory_usage.values()),
            'max_memory_mb': self.max_memory_mb,
            'keys': list(self._cache.keys())
        }


# Global DataFrame cache
_df_cache = DataFrameCache(max_memory_mb=500)


def get_cache_stats() -> Dict[str, Any]:
    """
    Get statistics for all caches.
    
    Returns
    -------
    dict
        Cache statistics
    """
    return {
        'calculation_cache': _global_cache.stats(),
        'dataframe_cache': _df_cache.stats()
    }


def clear_all_caches():
    """Clear all caches."""
    _global_cache.clear()
    _df_cache.clear()
    logger.info("All caches cleared")


def optimize_csv_loading(path: str, sep: str = ',', **kwargs) -> pd.DataFrame:
    """
    Optimized CSV loading with automatic dtype inference and chunking.
    
    Parameters
    ----------
    path : str
        Path to CSV file
    sep : str
        Separator character
    **kwargs
        Additional arguments for pd.read_csv
        
    Returns
    -------
    pd.DataFrame
        Loaded and optimized DataFrame
        
    Examples
    --------
    >>> df = optimize_csv_loading('large_file.csv', sep=';')
    """
    # First pass: infer dtypes from sample
    sample_df = pd.read_csv(path, sep=sep, nrows=1000, **kwargs)
    
    # Build dtype dict
    dtypes = {}
    for col in sample_df.columns:
        if sample_df[col].dtype == 'object':
            # Check if it's actually numeric
            try:
                pd.to_numeric(sample_df[col])
                dtypes[col] = 'float32'
            except:
                # Keep as object, will convert to category later
                pass
        elif sample_df[col].dtype == 'int64':
            dtypes[col] = 'int32'
        elif sample_df[col].dtype == 'float64':
            dtypes[col] = 'float32'
    
    # Load full file with optimized dtypes
    df = pd.read_csv(path, sep=sep, dtype=dtypes, **kwargs)
    
    # Further optimize
    df = optimize_dataframe_memory(df)
    
    return df
