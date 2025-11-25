"""
Development Configuration
Use this for development with hot-reload enabled
"""
from config.config import *

# Override settings for development
DASH_DEBUG = True  # Enable hot-reload for development
DATA_STORAGE_TYPE = 'session'  # Still use session to prevent data loss

# Development-specific settings
ENABLE_HOT_RELOAD = True
SHOW_DEBUG_INFO = True
LOG_LEVEL = 'DEBUG'

print("="*60)
print("DEVELOPMENT MODE ENABLED")
print("="*60)
print("⚠ Hot-reload is ENABLED")
print("⚠ Data will persist during session but may be lost on code changes")
print("⚠ For production, use: python app.py --production")
print("="*60)
