"""
Configuration settings for Trading Probability Explorer
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "dataraw"
SAMPLE_DATA_DIR = DATA_DIR / "sample"

# Ensure dataraw directory exists
if not DATA_DIR.exists():
    print(f"[WARNING] dataraw directory not found at {DATA_DIR}")
    print(f"   Please create the directory and add your CSV files")
else:
    print(f"[OK] dataraw directory found at {DATA_DIR}")

# CSV Configuration
FEATURE_CSV_SEPARATOR = ';'
TRADE_CSV_SEPARATOR = '\t'

# Data Validation
MIN_FEATURE_COLUMNS = 50
MIN_TRADE_COLUMNS = 40

# Probability Calculation Settings
DEFAULT_CONFIDENCE_LEVEL = 0.95
MIN_SAMPLES_PER_BIN = 30
DEFAULT_N_BINS = 10

# Bayesian Prior Parameters
BETA_PRIOR_ALPHA = 1.0
BETA_PRIOR_BETA = 1.0

# Monte Carlo Settings
DEFAULT_N_SIMULATIONS = 1000
MIN_N_SIMULATIONS = 100
MAX_N_SIMULATIONS = 10000

# Bootstrap Settings
DEFAULT_N_BOOTSTRAP = 1000

# Statistical Testing
SIGNIFICANCE_LEVEL = 0.05
BONFERRONI_CORRECTION = True

# Composite Score Weights
COMPOSITE_WEIGHTS = {
    'win_rate': 0.30,
    'expected_r': 0.25,
    'structure_quality': 0.15,
    'time_based': 0.10,
    'correlation': 0.10,
    'entry_quality': 0.10
}

# Score Thresholds
SCORE_THRESHOLDS = {
    'STRONG_BUY': 80,
    'BUY': 60,
    'NEUTRAL': 40,
    'AVOID': 0
}

# Trading Sessions
SESSIONS = {
    0: 'ASIA',
    1: 'EUROPE',
    2: 'US',
    3: 'OVERLAP'
}

# Dash Application Settings
DASH_HOST = '127.0.0.1'
DASH_PORT = 8050
DASH_DEBUG = False  # Set to False to prevent auto-reload and data loss
DASH_AUTO_OPEN_BROWSER = True

# Data Storage Settings
DATA_STORAGE_TYPE = 'session'  # 'memory', 'session', or 'local'
# 'memory' - data lost on page refresh
# 'session' - data persists during browser session
# 'local' - data persists even after browser close

# Visualization Settings
HEATMAP_COLORSCALE = 'RdYlGn'
DEFAULT_CHART_HEIGHT = 500
DEFAULT_CHART_WIDTH = 800

# Feature Engineering
N_PROBABILITY_FEATURES = 18

# Required Feature Columns (subset - full list in data_preprocessor.py)
REQUIRED_FEATURE_COLUMNS = [
    'timestamp', 'symbol', 'base_tf', 'trend_tf_dir', 'trend_strength_tf',
    'trend_regime', 'atr_tf_14', 'volatility_regime', 'session',
    'is_session_overlap', 'day_high', 'day_low', 'norm_close_daily'
]

# Required Trade Columns (subset - full list in data_preprocessor.py)
REQUIRED_TRADE_COLUMNS = [
    'Ticket_id', 'Symbol', 'Timestamp', 'Type', 'OpenPrice', 'Volume',
    'Timeframe', 'MagicNumber', 'StrategyType', 'MFEPips', 'MAEPips',
    'ClosePrice', 'ExitReason', 'R_multiple', 'trade_success'
]

# Target Variables
TARGET_COLUMNS = {
    'y_win': 'trade_success',
    'y_hit_1R': 'R_multiple >= 1',
    'y_hit_2R': 'R_multiple >= 2',
    'y_future_win_k': 'future_return_k > 0'
}

# Probability Features to Generate
PROBABILITY_FEATURES = [
    'prob_global_win',
    'prob_global_hit_1R',
    'prob_session_win',
    'prob_session_hit_1R',
    'prob_trend_strength_win',
    'prob_trend_strength_hit_1R',
    'prob_trend_dir_alignment_win',
    'prob_vol_regime_win',
    'prob_vol_regime_hit_1R',
    'prob_sr_zone_win',
    'prob_sr_zone_hit_1R',
    'prob_entropy_win',
    'prob_hurst_win',
    'prob_regime_cluster_win',
    'prob_streak_loss_win',
    'prob_dd_state_win',
    'prob_trend_vol_cross_win',
    'prob_session_sr_cross_win'
]
