"""
Auto Feature Selector

Advanced feature selection using multiple methods:
- Boruta: All-relevant feature selection
- SHAP: Feature contribution analysis
- RFECV: Recursive feature elimination with cross-validation
- CatBoost: Feature importance
- Random Forest + Permutation Importance
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class AutoFeatureSelector:
    """
    Comprehensive feature selection using multiple advanced methods
    """
    
    def __init__(self, n_features: int = 10, random_state: int = 42):
        """
        Initialize AutoFeatureSelector
        
        Parameters
        ----------
        n_features : int
            Target number of features to select
        random_state : int
            Random state for reproducibility
        """
        self.n_features = n_features
        self.random_state = random_state
        self.results = {}
        
    def quick_analysis(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Quick analysis using Random Forest + Permutation + SHAP
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
            
        Returns
        -------
        Dict
            Analysis results
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.inspection import permutation_importance
        from sklearn.model_selection import train_test_split
        
        results = {}
        
        # Step 1: Random Forest Importance
        print("Step 1/3: Random Forest Feature Importance...")
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_state,
            n_jobs=-1
        )
        rf.fit(X, y)
        
        # Ensure arrays have same length
        features = X.columns.tolist()
        importances = rf.feature_importances_.ravel()  # Use ravel() instead of flatten()
        
        if len(features) != len(importances):
            print(f"WARNING: Length mismatch - features: {len(features)}, importances: {len(importances)}")
            # Truncate to minimum length
            min_len = min(len(features), len(importances))
            features = features[:min_len]
            importances = importances[:min_len]
        
        rf_importance = pd.DataFrame({
            'feature': features,
            'rf_importance': importances
        }).sort_values('rf_importance', ascending=False)
        
        results['rf_importance'] = rf_importance
        
        # Step 2: Permutation Importance
        print("Step 2/3: Permutation Importance...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state
        )
        rf.fit(X_train, y_train)
        
        perm_imp = permutation_importance(
            rf, X_test, y_test,
            n_repeats=10,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Ensure arrays have same length
        features = X.columns.tolist()
        perm_mean = perm_imp.importances_mean.ravel()
        perm_std = perm_imp.importances_std.ravel()
        
        if len(features) != len(perm_mean):
            print(f"WARNING: Length mismatch - features: {len(features)}, perm_mean: {len(perm_mean)}")
            min_len = min(len(features), len(perm_mean))
            features = features[:min_len]
            perm_mean = perm_mean[:min_len]
            perm_std = perm_std[:min_len]
        
        perm_importance = pd.DataFrame({
            'feature': features,
            'perm_importance': perm_mean,
            'perm_std': perm_std
        }).sort_values('perm_importance', ascending=False)
        
        results['perm_importance'] = perm_importance
        
        # Step 3: SHAP Values
        print("Step 3/3: SHAP Analysis...")
        try:
            import shap
            
            # Use top features from permutation for SHAP (faster)
            top_features = perm_importance.head(min(20, len(X.columns)))['feature'].tolist()
            X_shap = X[top_features]
            
            rf_shap = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state
            )
            rf_shap.fit(X_shap, y)
            
            explainer = shap.TreeExplainer(rf_shap)
            shap_values = explainer.shap_values(X_shap)
            
            # Handle binary classification (shap_values might be list or 3D array)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class
            elif len(shap_values.shape) == 3:
                # For binary classification, SHAP might return (n_samples, n_features, n_classes)
                shap_values = shap_values[:, :, 1]  # Use positive class
            
            # Ensure correct shape: (n_samples, n_features)
            if len(shap_values.shape) != 2:
                print(f"WARNING: Unexpected SHAP shape: {shap_values.shape}")
                shap_values = shap_values.reshape(len(X_shap), -1)
            
            # Calculate mean absolute SHAP values per feature
            shap_mean_values = np.abs(shap_values).mean(axis=0).ravel()
            shap_direction_values = np.mean(shap_values, axis=0).ravel()
            
            # Validate lengths
            if len(top_features) != len(shap_mean_values):
                print(f"WARNING: SHAP length mismatch - features: {len(top_features)}, shap: {len(shap_mean_values)}")
                print(f"         SHAP values shape: {shap_values.shape}")
                min_len = min(len(top_features), len(shap_mean_values))
                top_features = top_features[:min_len]
                shap_mean_values = shap_mean_values[:min_len]
                shap_direction_values = shap_direction_values[:min_len]
            
            shap_importance = pd.DataFrame({
                'feature': top_features,
                'shap_mean': shap_mean_values,
                'shap_direction': shap_direction_values
            }).sort_values('shap_mean', ascending=False)
            
            results['shap_importance'] = shap_importance
            results['shap_values'] = shap_values
            results['shap_data'] = X_shap
            
        except ImportError:
            print("Warning: SHAP not installed. Skipping SHAP analysis.")
            results['shap_importance'] = None
        
        # Combine results
        combined = self._combine_rankings(results)
        results['combined_ranking'] = combined
        results['selected_features'] = combined.head(self.n_features)['feature'].tolist()
        
        return results
    
    def deep_analysis(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Deep analysis using Boruta + RFECV + SHAP
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
            
        Returns
        -------
        Dict
            Analysis results
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import RFECV
        
        results = {}
        
        # Step 1: Boruta
        print("Step 1/3: Boruta Feature Selection...")
        try:
            from boruta import BorutaPy
            
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=7,
                random_state=self.random_state,
                n_jobs=-1
            )
            
            boruta = BorutaPy(
                rf,
                n_estimators='auto',
                random_state=self.random_state,
                max_iter=100
            )
            boruta.fit(X.values, y.values)
            
            # Ensure arrays have same length
            features = X.columns.tolist()
            confirmed = boruta.support_.ravel()
            tentative = boruta.support_weak_.ravel()
            ranking = boruta.ranking_.ravel()
            
            if len(features) != len(confirmed):
                print(f"WARNING: Boruta length mismatch - features: {len(features)}, boruta: {len(confirmed)}")
                min_len = min(len(features), len(confirmed))
                features = features[:min_len]
                confirmed = confirmed[:min_len]
                tentative = tentative[:min_len]
                ranking = ranking[:min_len]
            
            boruta_results = pd.DataFrame({
                'feature': features,
                'boruta_confirmed': confirmed,
                'boruta_tentative': tentative,
                'boruta_ranking': ranking
            }).sort_values('boruta_ranking')
            
            results['boruta_results'] = boruta_results
            
            # Use confirmed + tentative features
            confirmed_features = X.columns[boruta.support_ | boruta.support_weak_].tolist()
            
            # Ensure minimum features for RFECV (need at least 2)
            if len(confirmed_features) < 2:
                print(f"WARNING: Boruta selected only {len(confirmed_features)} features. Using top 10 from ranking instead.")
                # Use top features from Boruta ranking
                top_n = min(10, len(X.columns))
                confirmed_features = boruta_results.head(top_n)['feature'].tolist()
            
        except ImportError:
            print("Warning: Boruta not installed. Using all features.")
            confirmed_features = X.columns.tolist()
            results['boruta_results'] = None
        except Exception as e:
            print(f"Warning: Boruta failed: {e}. Using all features.")
            confirmed_features = X.columns.tolist()
            results['boruta_results'] = None
        
        # Step 2: RFECV on confirmed features
        print("Step 2/3: RFECV Optimization...")
        X_confirmed = X[confirmed_features]
        
        # Ensure minimum features for RFECV
        if len(confirmed_features) < 2:
            print(f"ERROR: Not enough features ({len(confirmed_features)}) for RFECV. Need at least 2.")
            # Skip RFECV and use confirmed features directly
            results['rfecv_results'] = pd.DataFrame({
                'feature': confirmed_features,
                'rfecv_selected': [True] * len(confirmed_features),
                'rfecv_ranking': [1] * len(confirmed_features)
            })
            results['rfecv_n_features'] = len(confirmed_features)
            results['rfecv_cv_scores'] = {}
            optimal_features = confirmed_features
        else:
            rf = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            )
            
            rfecv = RFECV(
                estimator=rf,
                step=1,
                cv=5,
                scoring='roc_auc',
                n_jobs=-1
            )
            rfecv.fit(X_confirmed, y)
            
            # Ensure arrays have same length
            features = confirmed_features
            selected = rfecv.support_.ravel()
            ranking = rfecv.ranking_.ravel()
            
            if len(features) != len(selected):
                print(f"WARNING: RFECV length mismatch - features: {len(features)}, rfecv: {len(selected)}")
                min_len = min(len(features), len(selected))
                features = features[:min_len]
                selected = selected[:min_len]
                ranking = ranking[:min_len]
            
            rfecv_results = pd.DataFrame({
                'feature': features,
                'rfecv_selected': selected,
                'rfecv_ranking': ranking
            }).sort_values('rfecv_ranking')
            
            results['rfecv_results'] = rfecv_results
            results['rfecv_n_features'] = rfecv.n_features_
            results['rfecv_cv_scores'] = rfecv.cv_results_
            
            optimal_features = X_confirmed.columns[rfecv.support_].tolist()
        
        # Step 3: SHAP on optimal features
        print("Step 3/3: SHAP Analysis...")
        try:
            import shap
            
            X_optimal = X[optimal_features]
            
            rf_shap = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state
            )
            rf_shap.fit(X_optimal, y)
            
            explainer = shap.TreeExplainer(rf_shap)
            shap_values = explainer.shap_values(X_optimal)
            
            # Handle binary classification (shap_values might be list or 3D array)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class
            elif len(shap_values.shape) == 3:
                # For binary classification, SHAP might return (n_samples, n_features, n_classes)
                shap_values = shap_values[:, :, 1]  # Use positive class
            
            # Ensure correct shape: (n_samples, n_features)
            if len(shap_values.shape) != 2:
                print(f"WARNING: Unexpected SHAP shape: {shap_values.shape}")
                shap_values = shap_values.reshape(len(X_optimal), -1)
            
            # Calculate mean absolute SHAP values per feature
            shap_mean_values = np.abs(shap_values).mean(axis=0).ravel()
            shap_direction_values = np.mean(shap_values, axis=0).ravel()
            
            # Validate lengths
            if len(optimal_features) != len(shap_mean_values):
                print(f"WARNING: SHAP length mismatch - features: {len(optimal_features)}, shap: {len(shap_mean_values)}")
                print(f"         SHAP values shape: {shap_values.shape}")
                min_len = min(len(optimal_features), len(shap_mean_values))
                optimal_features = optimal_features[:min_len]
                shap_mean_values = shap_mean_values[:min_len]
                shap_direction_values = shap_direction_values[:min_len]
            
            shap_importance = pd.DataFrame({
                'feature': optimal_features,
                'shap_mean': shap_mean_values,
                'shap_direction': shap_direction_values
            }).sort_values('shap_mean', ascending=False)
            
            results['shap_importance'] = shap_importance
            results['shap_values'] = shap_values
            results['shap_data'] = X_optimal
            
        except ImportError:
            print("Warning: SHAP not installed. Skipping SHAP analysis.")
            results['shap_importance'] = None
        
        # Combine results
        combined = self._combine_deep_rankings(results)
        results['combined_ranking'] = combined
        results['selected_features'] = combined.head(self.n_features)['feature'].tolist()
        
        return results
    
    def _combine_rankings(self, results: Dict) -> pd.DataFrame:
        """Combine rankings from quick analysis"""
        
        rf_df = results['rf_importance'].copy()
        perm_df = results['perm_importance'].copy()
        
        # Merge
        combined = rf_df.merge(perm_df, on='feature', how='outer')
        
        # Add SHAP if available
        if results.get('shap_importance') is not None:
            shap_df = results['shap_importance'].copy()
            combined = combined.merge(shap_df, on='feature', how='left')
        
        # Fill NaN values before normalization
        for col in ['rf_importance', 'perm_importance', 'shap_mean', 'shap_direction']:
            if col in combined.columns:
                combined[col] = combined[col].fillna(0)
        
        # Normalize scores to 0-1 (min-max normalization)
        for col in ['rf_importance', 'perm_importance', 'shap_mean']:
            if col in combined.columns:
                col_min = combined[col].min()
                col_max = combined[col].max()
                if col_max > col_min:
                    combined[f'{col}_norm'] = (combined[col] - col_min) / (col_max - col_min)
                else:
                    # All values same, set to 0.5
                    combined[f'{col}_norm'] = 0.5
        
        # Calculate composite score
        if 'shap_mean_norm' in combined.columns and 'shap_direction' in combined.columns:
            # IMPROVED: Consider SHAP direction (positive/negative contribution)
            # Penalize negative SHAP direction
            shap_direction_factor = combined['shap_direction'].apply(
                lambda x: 1.0 if x > 0 else 0.5 if x > -0.01 else 0.2
            )
            
            combined['composite_score'] = (
                combined['rf_importance_norm'] * 0.25 +           # 25%
                combined['perm_importance_norm'] * 0.35 +         # 35%
                combined['shap_mean_norm'] * 0.30 +               # 30%
                (shap_direction_factor * 0.10)                    # 10% bonus for positive direction
            )
        elif 'shap_mean_norm' in combined.columns:
            # SHAP available but no direction
            combined['composite_score'] = (
                combined['rf_importance_norm'] * 0.3 +
                combined['perm_importance_norm'] * 0.4 +
                combined['shap_mean_norm'] * 0.3
            )
        else:
            # No SHAP
            combined['composite_score'] = (
                combined['rf_importance_norm'] * 0.5 +
                combined['perm_importance_norm'] * 0.5
            )
        
        # Ensure composite_score is between 0 and 1
        combined['composite_score'] = combined['composite_score'].clip(0, 1)
        
        combined = combined.sort_values('composite_score', ascending=False)
        combined['rank'] = range(1, len(combined) + 1)
        
        return combined
    
    def _combine_deep_rankings(self, results: Dict) -> pd.DataFrame:
        """Combine rankings from deep analysis"""
        
        # Start with Boruta results
        if results.get('boruta_results') is not None:
            combined = results['boruta_results'].copy()
        else:
            combined = pd.DataFrame({'feature': results['rfecv_results']['feature']})
        
        # Add RFECV
        rfecv_df = results['rfecv_results'].copy()
        combined = combined.merge(rfecv_df, on='feature', how='outer')
        
        # Add SHAP
        if results.get('shap_importance') is not None:
            shap_df = results['shap_importance'].copy()
            combined = combined.merge(shap_df, on='feature', how='left')
        
        # Calculate composite score
        combined['composite_score'] = 0
        
        # Boruta contribution (if available)
        if 'boruta_confirmed' in combined.columns:
            boruta_score = combined['boruta_confirmed'].fillna(0).astype(int) * 0.3
            combined['composite_score'] += boruta_score
        
        # RFECV contribution
        if 'rfecv_selected' in combined.columns:
            rfecv_score = combined['rfecv_selected'].fillna(0).astype(int) * 0.4
            combined['composite_score'] += rfecv_score
        
        # SHAP contribution
        if 'shap_mean' in combined.columns:
            shap_values = combined['shap_mean'].fillna(0)
            shap_min = shap_values.min()
            shap_max = shap_values.max()
            if shap_max > shap_min:
                shap_norm = (shap_values - shap_min) / (shap_max - shap_min + 1e-10)
            else:
                shap_norm = shap_values * 0  # All zeros if no variation
            
            # IMPROVED: Consider SHAP direction if available
            if 'shap_direction' in combined.columns:
                shap_direction = combined['shap_direction'].fillna(0)
                # Penalize negative direction
                direction_factor = shap_direction.apply(
                    lambda x: 1.0 if x > 0 else 0.5 if x > -0.01 else 0.2
                )
                combined['composite_score'] += shap_norm * direction_factor * 0.3
            else:
                combined['composite_score'] += shap_norm * 0.3
        
        # Ensure composite_score is between 0 and 1
        combined['composite_score'] = combined['composite_score'].clip(0, 1)
        
        combined = combined.sort_values('composite_score', ascending=False)
        combined['rank'] = range(1, len(combined) + 1)
        
        return combined
    
    def get_rejected_features(self, results: Dict, X: pd.DataFrame) -> pd.DataFrame:
        """Get features that should be rejected"""
        
        selected = results['selected_features']
        all_features = X.columns.tolist()
        rejected = [f for f in all_features if f not in selected]
        
        combined = results['combined_ranking']
        rejected_df = combined[combined['feature'].isin(rejected)].copy()
        
        # Add rejection reasons
        reasons = []
        for _, row in rejected_df.iterrows():
            reason_list = []
            
            if 'perm_importance' in row and row['perm_importance'] < 0.001:
                reason_list.append("Kontribusi sangat kecil")
            
            if 'boruta_confirmed' in row and not row['boruta_confirmed']:
                reason_list.append("Tidak dikonfirmasi Boruta")
            
            if 'rfecv_selected' in row and not row['rfecv_selected']:
                reason_list.append("Dieliminasi RFECV")
            
            if 'composite_score' in row and row['composite_score'] < 0.1:
                reason_list.append("Score komposit rendah")
            
            if not reason_list:
                reason_list.append("Ranking rendah")
            
            reasons.append("; ".join(reason_list))
        
        rejected_df['rejection_reason'] = reasons
        
        return rejected_df


def run_auto_feature_selection(
    df: pd.DataFrame,
    target_col: str,
    mode: str = 'quick',
    n_features: int = 10
) -> Dict:
    """
    Run automatic feature selection
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (merged trade + feature data)
    target_col : str
        Target column name (e.g., 'trade_success')
    mode : str
        'quick' or 'deep'
    n_features : int
        Number of features to select
        
    Returns
    -------
    Dict
        Analysis results
    """
    print("\n" + "="*60)
    print("AUTO FEATURE SELECTION - DATA PREPROCESSING")
    print("="*60)
    
    # ========================================
    # STEP 1: IDENTIFY AND REMOVE TRADE METADATA COLUMNS
    # ========================================
    # These columns are trade-specific metadata, NOT predictive features
    trade_metadata_cols = [
        # Trade identifiers
        'Ticket_id', 'ticket_id', 'ticket', 'trade_id',
        
        # Trade execution details (not predictive)
        'Symbol', 'symbol', 'Type', 'type', 'OpenPrice', 'open_price',
        'ClosePrice', 'close_price', 'Volume', 'volume',
        
        # EA-specific parameters (not market features)
        'Timeframe', 'timeframe', 'UseFibo50Filter', 'FiboBasePrice', 
        'FiboRange', 'MagicNumber', 'magic_number', 'StrategyType', 
        'strategy_type', 'ConsecutiveSLCount', 'TPHitsToday', 'SLHitsToday',
        
        # Session info (redundant with time features)
        'SessionHour', 'SessionMinute', 'SessionDayOfWeek', 'entry_session',
        
        # Trade results (DATA LEAKAGE - these are outcomes, not inputs!)
        'gross_profit', 'net_profit', 'R_multiple', 'ExitReason', 'exit_reason',
        'MFEPips', 'MAEPips', 'MAE_R', 'MFE_R', 'max_drawdown_k', 'max_runup_k',
        'future_return_k', 'equity_at_entry', 'equity_after_trade',
        
        # Trade timing (can cause data leakage)
        'exit_time', 'holding_bars', 'holding_minutes', 'K_bars',
        
        # Price levels (data leakage - known only after entry)
        'entry_price', 'sl_price', 'tp_price', 'sl_distance', 
        'money_risk', 'risk_percent', 'MaxSLTP',
        
        # Timestamp columns (use time-based features instead)
        'Timestamp', 'timestamp', 'entry_time', 'exit_time'
    ]
    
    # ========================================
    # STEP 2: IDENTIFY TARGET-RELATED COLUMNS
    # ========================================
    # These are potential target variables, not features
    target_related_cols = [
        'trade_success', 'y_win', 'y_hit_1R', 'y_hit_2R', 
        'y_future_win_k', 'win', 'success', 'target',
        'Morang_y_win', 'label', 'outcome'
    ]
    
    # ========================================
    # STEP 3: PREPARE TARGET VARIABLE
    # ========================================
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")
    
    y = df[target_col].copy()
    print(f"\n✓ Target variable: {target_col}")
    print(f"  - Total samples: {len(y)}")
    print(f"  - Positive class: {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"  - Negative class: {(~y.astype(bool)).sum()} ({(1-y.mean())*100:.1f}%)")
    
    # ========================================
    # STEP 4: FILTER FEATURE COLUMNS
    # ========================================
    # Start with all columns except target
    all_cols = df.columns.tolist()
    
    print(f"\n📋 Initial columns: {len(all_cols)}")
    
    # Remove target column
    feature_cols = [col for col in all_cols if col != target_col]
    print(f"  - After removing target: {len(feature_cols)}")
    
    # Remove other target-related columns
    removed_targets = [col for col in feature_cols if col in target_related_cols]
    feature_cols = [col for col in feature_cols if col not in target_related_cols]
    print(f"  - After removing target-related ({len(removed_targets)}): {len(feature_cols)}")
    
    # Remove trade metadata columns (case-insensitive)
    # Create lowercase mapping for case-insensitive comparison
    trade_metadata_lower = [c.lower() for c in trade_metadata_cols]
    
    removed_metadata = []
    filtered_cols = []
    for col in feature_cols:
        if col in trade_metadata_cols or col.lower() in trade_metadata_lower:
            removed_metadata.append(col)
        else:
            filtered_cols.append(col)
    
    feature_cols = filtered_cols
    
    print(f"\n✓ Filtered out trade metadata columns")
    print(f"  - Removed metadata: {len(removed_metadata)}")
    print(f"  - Remaining columns: {len(feature_cols)}")
    
    # Debug: Show some removed and remaining columns
    if removed_metadata:
        print(f"\n  Sample removed metadata (first 5):")
        for col in removed_metadata[:5]:
            print(f"    - {col}")
    
    if feature_cols:
        print(f"\n  Sample remaining columns (first 10):")
        for col in feature_cols[:10]:
            print(f"    - {col}")
    
    # ========================================
    # STEP 5: SELECT NUMERIC FEATURES ONLY
    # ========================================
    
    # Check if we have any columns left
    if len(feature_cols) == 0:
        raise ValueError(
            "No features remaining after filtering trade metadata!\n"
            "This usually means:\n"
            "1. Data hanya berisi trade data (tidak ada market features)\n"
            "2. Atau data belum di-merge dengan market_features CSV\n\n"
            "Solusi:\n"
            "- Pastikan Anda sudah memuat KEDUA file (trade CSV + feature CSV)\n"
            "- Klik 'Muat Data Terpilih' di panel kontrol global\n"
            "- Data harus ter-merge sebelum menjalankan Auto Feature Selection"
        )
    
    X = df[feature_cols].copy()
    
    # Keep only numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [col for col in feature_cols if col not in numeric_cols]
    
    X = X[numeric_cols]
    
    print(f"\n✓ Selected numeric features only")
    print(f"  - Numeric features: {len(numeric_cols)}")
    if non_numeric_cols:
        print(f"  - Non-numeric removed: {len(non_numeric_cols)}")
    
    # Check if we have numeric features
    if len(numeric_cols) == 0:
        raise ValueError(
            "No numeric features found!\n"
            "Semua kolom yang tersisa adalah non-numeric (string/object).\n\n"
            "Solusi:\n"
            "- Pastikan data market features sudah dimuat\n"
            "- Market features harus berisi kolom numerik (trend, volatility, dll)"
        )
    
    # ========================================
    # STEP 6: REMOVE LOW-QUALITY FEATURES
    # ========================================
    
    # 6a. Remove columns with too many missing values (>50%)
    missing_pct = X.isnull().sum() / len(X)
    high_missing_cols = missing_pct[missing_pct >= 0.5].index.tolist()
    if high_missing_cols:
        print(f"\n✓ Removed {len(high_missing_cols)} columns with >50% missing values:")
        for col in high_missing_cols[:5]:  # Show first 5
            print(f"  - {col}: {missing_pct[col]*100:.1f}% missing")
        if len(high_missing_cols) > 5:
            print(f"  ... and {len(high_missing_cols)-5} more")
    
    valid_cols = missing_pct[missing_pct < 0.5].index
    X = X[valid_cols]
    
    # 6b. Fill remaining missing values with median
    X = X.fillna(X.median())
    
    # 6c. Remove constant columns (no variation)
    nunique = X.nunique()
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        print(f"\n✓ Removed {len(constant_cols)} constant columns (no variation)")
    
    valid_cols = nunique[nunique > 1].index
    X = X[valid_cols]
    
    # 6d. Remove columns with very low variance (< 1e-10)
    variances = X.var()
    low_var_cols = variances[variances < 1e-10].index.tolist()
    if low_var_cols:
        print(f"\n✓ Removed {len(low_var_cols)} columns with very low variance")
    
    valid_cols = variances[variances >= 1e-10].index
    X = X[valid_cols]
    
    # ========================================
    # STEP 7: FINAL DATA SUMMARY
    # ========================================
    print("\n" + "="*60)
    print("FINAL PREPARED DATA")
    print("="*60)
    print(f"Features: {X.shape[1]} columns")
    print(f"Samples: {X.shape[0]} rows")
    print(f"Target: {target_col} (Win Rate: {y.mean()*100:.1f}%)")
    print("="*60 + "\n")
    
    # Show sample of feature names
    if X.shape[1] > 0:
        print("Sample features (first 10):")
        for i, col in enumerate(X.columns[:10], 1):
            print(f"  {i}. {col}")
        if X.shape[1] > 10:
            print(f"  ... and {X.shape[1]-10} more features")
        print()
    
    # Validate minimum requirements
    if X.shape[1] < 2:
        error_msg = (
            f"Not enough features after preprocessing!\n\n"
            f"📊 Summary:\n"
            f"  - Original columns: {len(all_cols)}\n"
            f"  - After filtering: {X.shape[1]} features\n"
            f"  - Required minimum: 2 features\n\n"
            f"🔍 Possible causes:\n"
            f"  1. Data hanya berisi trade data (tidak ada market features)\n"
            f"  2. Market features belum dimuat atau di-merge\n"
            f"  3. Semua fitur ter-filter karena missing values atau constant\n\n"
            f"✅ Solusi:\n"
            f"  1. Pastikan Anda memuat KEDUA file:\n"
            f"     - Trade CSV (EA_SWINGHL_BT-GOLD...)\n"
            f"     - Feature CSV (market_features_TierSA...)\n"
            f"  2. Klik 'Muat Data Terpilih' di panel kontrol global\n"
            f"  3. Tunggu hingga muncul 'Data berhasil dimuat'\n"
            f"  4. Baru jalankan Auto Feature Selection\n\n"
            f"💡 Tip: Lihat label data di atas. Jika menunjukkan '0 fitur market',\n"
            f"     berarti data belum di-merge dengan benar."
        )
        raise ValueError(error_msg)
    
    if X.shape[0] < 50:
        error_msg = (
            f"Not enough samples after preprocessing!\n\n"
            f"📊 Summary:\n"
            f"  - Original samples: {len(df)}\n"
            f"  - After preprocessing: {X.shape[0]} samples\n"
            f"  - Required minimum: 50 samples\n\n"
            f"✅ Solusi:\n"
            f"  - Gunakan dataset yang lebih besar (minimal 50 trades)\n"
            f"  - Atau kurangi filter preprocessing"
        )
        raise ValueError(error_msg)
    
    # ========================================
    # STEP 8: RUN FEATURE SELECTION
    # ========================================
    selector = AutoFeatureSelector(n_features=n_features)
    
    if mode == 'quick':
        results = selector.quick_analysis(X, y)
    else:
        results = selector.deep_analysis(X, y)
    
    # Add rejected features
    results['rejected_features'] = selector.get_rejected_features(results, X)
    
    # Add preprocessing info to results
    results['preprocessing_info'] = {
        'original_columns': len(all_cols),
        'removed_trade_metadata': len([c for c in all_cols if c in trade_metadata_cols]),
        'removed_target_related': len([c for c in all_cols if c in target_related_cols]),
        'removed_high_missing': len(high_missing_cols),
        'removed_constant': len(constant_cols),
        'removed_low_variance': len(low_var_cols),
        'final_features': X.shape[1],
        'final_samples': X.shape[0]
    }
    
    return results
