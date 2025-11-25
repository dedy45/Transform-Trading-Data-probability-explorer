"""
Production Configuration
Use this for production deployment with stable data storage
"""
from config.config import *

# Override settings for production
DASH_DEBUG = False  # Disable hot-reload for stability
DATA_STORAGE_TYPE = 'session'  # Session storage for data persistence

# Production-specific settings
ENABLE_HOT_RELOAD = False
SHOW_DEBUG_INFO = False
LOG_LEVEL = 'INFO'

print("="*60)
print("PRODUCTION MODE ENABLED")
print("="*60)
print("✓ Hot-reload is DISABLED")
print("✓ Data persists during entire browser session")
print("✓ Stable performance without auto-reload")
print("="*60)
