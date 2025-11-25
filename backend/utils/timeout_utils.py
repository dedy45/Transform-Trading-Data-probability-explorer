"""
Timeout Utilities

Provides timeout context managers and decorators for long-running operations.
"""

import signal
import functools
import threading
from contextlib import contextmanager
from typing import Callable, Any


class TimeoutError(Exception):
    """Custom timeout exception"""
    pass


@contextmanager
def timeout_context(seconds: int):
    """
    Context manager for timeout protection.
    
    Works on Unix-like systems using SIGALRM.
    On Windows, timeout is not enforced but code still runs.
    
    Parameters
    ----------
    seconds : int
        Timeout in seconds
        
    Raises
    ------
    TimeoutError
        If operation exceeds timeout
        
    Examples
    --------
    >>> with timeout_context(30):
    ...     df = pd.read_csv('large_file.csv')
    """
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation exceeded {seconds} seconds")
    
    # Set timeout (only works on Unix-like systems)
    try:
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    except (AttributeError, ValueError):
        # Windows doesn't support SIGALRM, just yield without timeout
        # In production, consider using multiprocessing or threading
        yield


def timeout_decorator(seconds: int):
    """
    Decorator for timeout protection.
    
    Parameters
    ----------
    seconds : int
        Timeout in seconds
        
    Examples
    --------
    >>> @timeout_decorator(60)
    ... def load_large_file(path):
    ...     return pd.read_csv(path)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            with timeout_context(seconds):
                return func(*args, **kwargs)
        return wrapper
    return decorator


class ThreadTimeout:
    """
    Thread-based timeout for Windows compatibility.
    
    This is a fallback for Windows systems where signal.SIGALRM is not available.
    Uses threading to implement timeout.
    
    Examples
    --------
    >>> with ThreadTimeout(30):
    ...     df = pd.read_csv('large_file.csv')
    """
    
    def __init__(self, seconds: int):
        self.seconds = seconds
        self.timer = None
        self.timed_out = False
        
    def __enter__(self):
        def timeout_handler():
            self.timed_out = True
            
        self.timer = threading.Timer(self.seconds, timeout_handler)
        self.timer.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.timer:
            self.timer.cancel()
        
        if self.timed_out:
            raise TimeoutError(f"Operation exceeded {self.seconds} seconds")
        
        return False


def get_timeout_context(seconds: int):
    """
    Get appropriate timeout context based on platform.
    
    Returns signal-based timeout on Unix, thread-based on Windows.
    
    Parameters
    ----------
    seconds : int
        Timeout in seconds
        
    Returns
    -------
    context manager
        Appropriate timeout context for the platform
    """
    try:
        # Test if SIGALRM is available
        signal.signal(signal.SIGALRM, signal.SIG_DFL)
        return timeout_context(seconds)
    except (AttributeError, ValueError):
        # Fall back to thread-based timeout
        return ThreadTimeout(seconds)


# Predefined timeout values for common operations
TIMEOUT_CSV_LOAD = 60  # 1 minute for CSV loading
TIMEOUT_DATA_MERGE = 120  # 2 minutes for data merging
TIMEOUT_QUICK_ANALYSIS = 300  # 5 minutes for quick analysis
TIMEOUT_DEEP_ANALYSIS = 900  # 15 minutes for deep analysis
TIMEOUT_MODEL_TRAINING = 600  # 10 minutes for model training
TIMEOUT_PREDICTION = 30  # 30 seconds for prediction
