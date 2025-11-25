"""
Server-Side Data Cache

Persistent data storage on server side to prevent data loss on page refresh.
Data persists until server is stopped.
"""

import pandas as pd
from typing import Dict, Optional, Any
from datetime import datetime
import threading


class ServerDataCache:
    """
    Server-side data cache that persists across page refreshes.
    Data is stored in memory on the server and only cleared when server stops.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._cache: Dict[str, Any] = {}
        self._metadata: Dict[str, Dict] = {}
        self._initialized = True
        print("[OK] Server-side data cache initialized")
    
    def store_data(self, key: str, data: pd.DataFrame, metadata: Optional[Dict] = None):
        """
        Store DataFrame in server cache.
        
        Args:
            key: Unique identifier for the data
            data: DataFrame to store
            metadata: Optional metadata about the data
        """
        self._cache[key] = data.copy()
        self._metadata[key] = {
            'timestamp': datetime.now().isoformat(),
            'shape': data.shape,
            'columns': list(data.columns),
            'memory_mb': data.memory_usage(deep=True).sum() / 1024 / 1024,
            **(metadata or {})
        }
        print(f"[OK] Stored '{key}' in server cache: {data.shape[0]} rows, {data.shape[1]} cols")
    
    def get_data(self, key: str) -> Optional[pd.DataFrame]:
        """
        Retrieve DataFrame from server cache.
        
        Args:
            key: Unique identifier for the data
            
        Returns:
            DataFrame if exists, None otherwise
        """
        if key in self._cache:
            print(f"[OK] Retrieved '{key}' from server cache")
            return self._cache[key].copy()
        return None
    
    def has_data(self, key: str) -> bool:
        """Check if data exists in cache."""
        return key in self._cache
    
    def get_metadata(self, key: str) -> Optional[Dict]:
        """Get metadata for cached data."""
        return self._metadata.get(key)
    
    def list_cached_keys(self) -> list:
        """List all cached data keys."""
        return list(self._cache.keys())
    
    def get_cache_summary(self) -> Dict:
        """Get summary of all cached data."""
        summary = {}
        for key in self._cache.keys():
            meta = self._metadata.get(key, {})
            summary[key] = {
                'rows': meta.get('shape', (0, 0))[0],
                'cols': meta.get('shape', (0, 0))[1],
                'memory_mb': meta.get('memory_mb', 0),
                'timestamp': meta.get('timestamp', 'unknown')
            }
        return summary
    
    def clear_data(self, key: str):
        """Clear specific data from cache."""
        if key in self._cache:
            del self._cache[key]
            if key in self._metadata:
                del self._metadata[key]
            print(f"[OK] Cleared '{key}' from server cache")
    
    def clear_all(self):
        """Clear all data from cache."""
        self._cache.clear()
        self._metadata.clear()
        print("[OK] Cleared all data from server cache")
    
    def get_total_memory_mb(self) -> float:
        """Get total memory usage of cache in MB."""
        return sum(meta.get('memory_mb', 0) for meta in self._metadata.values())


# Global cache instance
_cache = ServerDataCache()


def get_cache() -> ServerDataCache:
    """Get the global cache instance."""
    return _cache


# Convenience functions
def store_feature_data(df: pd.DataFrame):
    """Store feature CSV data."""
    get_cache().store_data('feature_data', df, {'type': 'feature_csv'})


def store_trade_data(df: pd.DataFrame):
    """Store trade CSV data."""
    get_cache().store_data('trade_data', df, {'type': 'trade_csv'})


def store_merged_data(df: pd.DataFrame):
    """Store merged dataset."""
    get_cache().store_data('merged_data', df, {'type': 'merged'})


def get_feature_data() -> Optional[pd.DataFrame]:
    """Get feature CSV data."""
    return get_cache().get_data('feature_data')


def get_trade_data() -> Optional[pd.DataFrame]:
    """Get trade CSV data."""
    return get_cache().get_data('trade_data')


def get_merged_data() -> Optional[pd.DataFrame]:
    """Get merged dataset."""
    return get_cache().get_data('merged_data')


def has_cached_data() -> bool:
    """Check if any data is cached."""
    cache = get_cache()
    return cache.has_data('merged_data') or cache.has_data('trade_data')


def clear_all_data():
    """Clear all cached data."""
    get_cache().clear_all()


def get_cache_info() -> Dict:
    """Get information about cached data."""
    cache = get_cache()
    return {
        'has_data': has_cached_data(),
        'cached_keys': cache.list_cached_keys(),
        'summary': cache.get_cache_summary(),
        'total_memory_mb': cache.get_total_memory_mb()
    }


print("[OK] Server-side data cache module loaded")
