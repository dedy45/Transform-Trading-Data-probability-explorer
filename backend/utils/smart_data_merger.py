"""
Smart Data Merger

Intelligent and flexible data merging system that:
1. Auto-detects timestamp columns
2. Auto-detects feature columns
3. Handles column name changes
4. Fuzzy matching for similar column names
5. No hardcoded column requirements
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings


class SmartDataMerger:
    """
    Smart data merger that automatically handles:
    - Timestamp detection
    - Column mapping
    - Flexible merging
    - Column name changes
    """
    
    # Common timestamp column names (priority order)
    TIMESTAMP_CANDIDATES = [
        'timestamp', 'Timestamp', 'time', 'Time', 'datetime', 'DateTime',
        'entry_time', 'EntryTime', 'open_time', 'OpenTime',
        'date', 'Date', 'dt', 'DT'
    ]
    
    # Common target column names for trading
    TARGET_CANDIDATES = [
        'trade_success', 'TradeSuccess', 'win', 'Win', 'success', 'Success',
        'is_win', 'IsWin', 'profitable', 'Profitable',
        'result', 'Result', 'outcome', 'Outcome'
    ]
    
    def __init__(self, verbose: bool = True):
        """
        Initialize SmartDataMerger
        
        Parameters
        ----------
        verbose : bool
            Print info messages during processing
        """
        self.verbose = verbose
        self.feature_timestamp_col = None
        self.trade_timestamp_col = None
        self.detected_features = []
        
    def _log(self, message: str):
        """Print log message if verbose"""
        if self.verbose:
            print(f"[SmartMerger] {message}")
    
    def detect_timestamp_column(self, df: pd.DataFrame, df_name: str = "DataFrame") -> Optional[str]:
        """
        Auto-detect timestamp column in DataFrame
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        df_name : str
            Name for logging
            
        Returns
        -------
        str or None
            Detected timestamp column name
        """
        # Check candidates in priority order
        for candidate in self.TIMESTAMP_CANDIDATES:
            if candidate in df.columns:
                # Verify it's actually a timestamp
                try:
                    pd.to_datetime(df[candidate].iloc[0])
                    self._log(f"Detected timestamp column in {df_name}: '{candidate}'")
                    return candidate
                except:
                    continue
        
        # Fallback: Find any column with datetime-like values
        for col in df.columns:
            if df[col].dtype == 'object' or 'datetime' in str(df[col].dtype):
                try:
                    # Try to parse first few values
                    pd.to_datetime(df[col].head(10))
                    self._log(f"Auto-detected timestamp column in {df_name}: '{col}'")
                    return col
                except:
                    continue
        
        self._log(f"WARNING: No timestamp column found in {df_name}")
        return None
    
    def detect_target_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Auto-detect target column (win/loss) in DataFrame
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
            
        Returns
        -------
        str or None
            Detected target column name
        """
        # Check candidates
        for candidate in self.TARGET_CANDIDATES:
            if candidate in df.columns:
                # Verify it's binary (0/1)
                unique_vals = df[candidate].dropna().unique()
                if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                    self._log(f"Detected target column: '{candidate}'")
                    return candidate
        
        # Fallback: Find any binary column
        for col in df.columns:
            if df[col].dtype in [np.int64, np.float64]:
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                    self._log(f"Auto-detected binary column as target: '{col}'")
                    return col
        
        self._log("WARNING: No target column found")
        return None
    
    def detect_feature_columns(self, df: pd.DataFrame, exclude_cols: List[str] = None) -> List[str]:
        """
        Auto-detect feature columns (numeric columns suitable for ML)
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        exclude_cols : List[str]
            Columns to exclude from features
            
        Returns
        -------
        List[str]
            List of detected feature column names
        """
        if exclude_cols is None:
            exclude_cols = []
        
        # Add common non-feature columns to exclude
        common_excludes = [
            'Ticket_id', 'ticket_id', 'id', 'ID',
            'Symbol', 'symbol',
            'MagicNumber', 'magic_number',
            'Timestamp', 'timestamp', 'time', 'datetime',
            'entry_time', 'exit_time',
            'trade_success', 'win', 'success'
        ]
        exclude_cols.extend(common_excludes)
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter out excluded columns
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Remove columns with too many missing values (>50%)
        valid_features = []
        for col in feature_cols:
            missing_pct = df[col].isnull().sum() / len(df)
            if missing_pct < 0.5:
                valid_features.append(col)
        
        self._log(f"Detected {len(valid_features)} feature columns")
        self.detected_features = valid_features
        
        return valid_features
    
    def standardize_timestamp(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """
        Standardize timestamp column to datetime format
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        timestamp_col : str
            Name of timestamp column
            
        Returns
        -------
        pd.DataFrame
            DataFrame with standardized timestamp
        """
        df = df.copy()
        
        # Convert to datetime
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
        
        # Remove timezone if present (for consistent merging)
        if df[timestamp_col].dt.tz is not None:
            df[timestamp_col] = df[timestamp_col].dt.tz_localize(None)
        
        # Round to nearest second (avoid microsecond mismatches)
        df[timestamp_col] = df[timestamp_col].dt.round('S')
        
        return df
    
    def merge_smart(
        self,
        features_df: pd.DataFrame,
        trades_df: pd.DataFrame,
        merge_tolerance: str = '1min'
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Smart merge of feature and trade datasets
        
        Parameters
        ----------
        features_df : pd.DataFrame
            Feature data
        trades_df : pd.DataFrame
            Trade data
        merge_tolerance : str
            Time tolerance for fuzzy matching (e.g., '1min', '5min')
            
        Returns
        -------
        Tuple[pd.DataFrame, Dict]
            Merged DataFrame and merge statistics
        """
        stats = {
            'total_trades': len(trades_df),
            'total_features': len(features_df),
            'matched_trades': 0,
            'unmatched_trades': 0,
            'feature_columns_used': 0,
            'merge_method': 'exact'
        }
        
        # Detect timestamp columns
        self.feature_timestamp_col = self.detect_timestamp_column(features_df, "Features")
        self.trade_timestamp_col = self.detect_timestamp_column(trades_df, "Trades")
        
        if not self.feature_timestamp_col or not self.trade_timestamp_col:
            self._log("ERROR: Cannot merge without timestamp columns")
            return trades_df, stats
        
        # Standardize timestamps
        features_df = self.standardize_timestamp(features_df, self.feature_timestamp_col)
        trades_df = self.standardize_timestamp(trades_df, self.trade_timestamp_col)
        
        # Try exact merge first
        self._log("Attempting exact timestamp merge...")
        merged_df = trades_df.merge(
            features_df,
            left_on=self.trade_timestamp_col,
            right_on=self.feature_timestamp_col,
            how='left',
            suffixes=('', '_feature')
        )
        
        # Check match rate
        feature_cols = [col for col in features_df.columns if col != self.feature_timestamp_col]
        if len(feature_cols) > 0:
            matched = merged_df[feature_cols[0]].notna().sum()
            match_rate = matched / len(merged_df)
            
            stats['matched_trades'] = matched
            stats['unmatched_trades'] = len(merged_df) - matched
            stats['feature_columns_used'] = len(feature_cols)
            
            self._log(f"Exact merge: {matched}/{len(merged_df)} trades matched ({match_rate:.1%})")
            
            # If match rate is low, try fuzzy merge
            if match_rate < 0.5:
                self._log(f"Low match rate. Attempting fuzzy merge with tolerance={merge_tolerance}...")
                merged_df = self._fuzzy_merge(
                    trades_df, features_df, merge_tolerance
                )
                stats['merge_method'] = 'fuzzy'
        
        # Detect and add target column if not present
        if 'trade_success' not in merged_df.columns:
            target_col = self.detect_target_column(merged_df)
            if target_col and target_col != 'trade_success':
                merged_df['trade_success'] = merged_df[target_col]
                self._log(f"Mapped '{target_col}' to 'trade_success'")
        
        return merged_df, stats
    
    def _fuzzy_merge(
        self,
        trades_df: pd.DataFrame,
        features_df: pd.DataFrame,
        tolerance: str
    ) -> pd.DataFrame:
        """
        Fuzzy merge with time tolerance
        
        Parameters
        ----------
        trades_df : pd.DataFrame
            Trade data
        features_df : pd.DataFrame
            Feature data
        tolerance : str
            Time tolerance (e.g., '1min', '5min')
            
        Returns
        -------
        pd.DataFrame
            Merged DataFrame
        """
        # Use merge_asof for nearest timestamp matching
        trades_sorted = trades_df.sort_values(self.trade_timestamp_col)
        features_sorted = features_df.sort_values(self.feature_timestamp_col)
        
        merged_df = pd.merge_asof(
            trades_sorted,
            features_sorted,
            left_on=self.trade_timestamp_col,
            right_on=self.feature_timestamp_col,
            direction='nearest',
            tolerance=pd.Timedelta(tolerance)
        )
        
        return merged_df
    
    def get_merge_report(self, merged_df: pd.DataFrame, stats: Dict) -> str:
        """
        Generate merge report
        
        Parameters
        ----------
        merged_df : pd.DataFrame
            Merged DataFrame
        stats : Dict
            Merge statistics
            
        Returns
        -------
        str
            Formatted report
        """
        report = []
        report.append("="*60)
        report.append("SMART MERGE REPORT")
        report.append("="*60)
        report.append(f"Total Trades: {stats['total_trades']}")
        report.append(f"Total Features: {stats['total_features']}")
        report.append(f"Matched Trades: {stats['matched_trades']}")
        report.append(f"Unmatched Trades: {stats['unmatched_trades']}")
        report.append(f"Match Rate: {stats['matched_trades']/stats['total_trades']:.1%}")
        report.append(f"Feature Columns: {stats['feature_columns_used']}")
        report.append(f"Merge Method: {stats['merge_method']}")
        report.append(f"Final Shape: {merged_df.shape}")
        report.append("="*60)
        
        return "\n".join(report)


def smart_merge_datasets(
    features_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    merge_tolerance: str = '1min',
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Smart merge function (convenience wrapper)
    
    Parameters
    ----------
    features_df : pd.DataFrame
        Feature data
    trades_df : pd.DataFrame
        Trade data
    merge_tolerance : str
        Time tolerance for fuzzy matching
    verbose : bool
        Print info messages
        
    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        Merged DataFrame and statistics
    """
    merger = SmartDataMerger(verbose=verbose)
    merged_df, stats = merger.merge_smart(features_df, trades_df, merge_tolerance)
    
    if verbose:
        print(merger.get_merge_report(merged_df, stats))
    
    return merged_df, stats
