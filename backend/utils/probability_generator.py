"""
Probability Generator Utility

Auto-generate probability columns for calibration analysis when they don't exist in data.
Uses various methods: historical win rate, feature-based probability, simple models, etc.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
import warnings
warnings.filterwarnings('ignore')


def generate_global_probability(df: pd.DataFrame, target_col: str = 'trade_success') -> pd.Series:
    """
    Generate global probability based on overall win rate.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Trading data
    target_col : str
        Target column name (binary 0/1)
        
    Returns:
    --------
    pd.Series
        Probability column (same value for all rows = global win rate)
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")
    
    global_win_rate = df[target_col].mean()
    return pd.Series([global_win_rate] * len(df), index=df.index, name='prob_global_win')


def generate_session_probability(
    df: pd.DataFrame, 
    target_col: str = 'trade_success',
    session_col: str = 'session'
) -> pd.Series:
    """
    Generate probability based on trading session (Asian, European, US).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Trading data
    target_col : str
        Target column name
    session_col : str
        Session column name
        
    Returns:
    --------
    pd.Series
        Probability per session
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")
    
    # If session column doesn't exist, try to create from timestamp
    if session_col not in df.columns:
        if 'Timestamp' in df.columns:
            df = df.copy()
            df['hour'] = pd.to_datetime(df['Timestamp']).dt.hour
            # Asian: 0-8, European: 8-16, US: 16-24
            df[session_col] = pd.cut(
                df['hour'], 
                bins=[0, 8, 16, 24], 
                labels=['Asian', 'European', 'US'],
                include_lowest=True
            )
        elif 'entry_time' in df.columns:
            df = df.copy()
            df['hour'] = pd.to_datetime(df['entry_time']).dt.hour
            df[session_col] = pd.cut(
                df['hour'], 
                bins=[0, 8, 16, 24], 
                labels=['Asian', 'European', 'US'],
                include_lowest=True
            )
        else:
            # Fallback to global probability
            return generate_global_probability(df, target_col)
    
    # Calculate win rate per session
    session_probs = df.groupby(session_col)[target_col].mean()
    
    # Map to each row
    prob_series = df[session_col].map(session_probs)
    prob_series.name = 'prob_session_win'
    
    return prob_series


def generate_feature_based_probability(
    df: pd.DataFrame,
    target_col: str = 'trade_success',
    feature_cols: Optional[List[str]] = None,
    method: str = 'logistic'
) -> pd.Series:
    """
    Generate probability using machine learning model based on features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Trading data
    target_col : str
        Target column name
    feature_cols : list, optional
        List of feature columns to use. If None, auto-detect numeric columns.
    method : str
        Model method: 'logistic' (default), 'simple_avg'
        
    Returns:
    --------
    pd.Series
        Predicted probabilities
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")
    
    # Auto-detect feature columns if not provided
    if feature_cols is None:
        # Exclude target, timestamp, and ID columns
        exclude_cols = [
            target_col, 'Timestamp', 'entry_time', 'exit_time', 
            'trade_id', 'ticket', 'order_id', 'comment'
        ]
        feature_cols = [
            col for col in df.columns 
            if col not in exclude_cols 
            and pd.api.types.is_numeric_dtype(df[col])
            and not col.startswith('y_')  # Exclude other target columns
        ]
    
    if len(feature_cols) == 0:
        print("[PROB GEN] No features found, using global probability")
        return generate_global_probability(df, target_col)
    
    print(f"[PROB GEN] Using {len(feature_cols)} features: {feature_cols[:10]}...")
    
    # Prepare data
    df_clean = df[feature_cols + [target_col]].copy()
    df_clean = df_clean.dropna()
    
    if len(df_clean) < 50:
        print(f"[PROB GEN] Not enough data ({len(df_clean)} rows), using global probability")
        return generate_global_probability(df, target_col)
    
    X = df_clean[feature_cols].values
    y = df_clean[target_col].values
    
    if method == 'logistic':
        # Use Logistic Regression with cross-validation
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model = LogisticRegression(
                max_iter=1000, 
                random_state=42,
                class_weight='balanced'
            )
            
            # Use cross-validation to get out-of-fold predictions
            probs = cross_val_predict(
                model, X_scaled, y, 
                cv=5, 
                method='predict_proba'
            )[:, 1]
            
            # Create series with original index
            prob_series = pd.Series(np.nan, index=df.index, name='prob_model_win')
            prob_series.loc[df_clean.index] = probs
            
            # Fill NaN with global probability
            global_prob = y.mean()
            prob_series = prob_series.fillna(global_prob)
            
            print(f"[PROB GEN] Logistic model trained successfully")
            print(f"  - Probability range: [{prob_series.min():.3f}, {prob_series.max():.3f}]")
            print(f"  - Mean probability: {prob_series.mean():.3f}")
            
            return prob_series
            
        except Exception as e:
            print(f"[PROB GEN] Error training model: {e}")
            return generate_global_probability(df, target_col)
    
    else:  # simple_avg
        # Simple average of normalized features
        X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        feature_avg = X_norm.mean(axis=1)
        
        # Convert to probability [0, 1]
        from scipy.stats import norm
        probs = norm.cdf(feature_avg)
        
        prob_series = pd.Series(np.nan, index=df.index, name='prob_simple_win')
        prob_series.loc[df_clean.index] = probs
        prob_series = prob_series.fillna(0.5)
        
        return prob_series


def generate_r_multiple_probability(
    df: pd.DataFrame,
    r_col: str = 'R_multiple',
    bins: int = 10
) -> pd.Series:
    """
    Generate probability based on historical R-multiple distribution.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Trading data
    r_col : str
        R-multiple column name
    bins : int
        Number of bins for R-multiple
        
    Returns:
    --------
    pd.Series
        Probability based on R-multiple bin
    """
    if r_col not in df.columns:
        raise ValueError(f"R-multiple column '{r_col}' not found")
    
    df_clean = df[[r_col]].copy().dropna()
    
    if len(df_clean) < 10:
        return pd.Series([0.5] * len(df), index=df.index, name='prob_r_based_win')
    
    # Create bins
    df_clean['r_bin'] = pd.qcut(df_clean[r_col], q=bins, duplicates='drop')
    
    # Calculate win rate per bin (R > 0)
    df_clean['is_win'] = (df_clean[r_col] > 0).astype(int)
    bin_probs = df_clean.groupby('r_bin')['is_win'].mean()
    
    # Map back to original data
    prob_series = pd.Series(np.nan, index=df.index, name='prob_r_based_win')
    
    for idx in df_clean.index:
        r_bin = df_clean.loc[idx, 'r_bin']
        prob_series.loc[idx] = bin_probs[r_bin]
    
    # Fill NaN with global win rate
    global_win_rate = df_clean['is_win'].mean()
    prob_series = prob_series.fillna(global_win_rate)
    
    return prob_series


def generate_composite_probability(
    df: pd.DataFrame,
    target_col: str = 'trade_success',
    methods: List[str] = ['global', 'session', 'model']
) -> pd.Series:
    """
    Generate composite probability by averaging multiple methods.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Trading data
    target_col : str
        Target column name
    methods : list
        List of methods to combine: 'global', 'session', 'model', 'r_based'
        
    Returns:
    --------
    pd.Series
        Composite probability (average of all methods)
    """
    probs_list = []
    
    for method in methods:
        try:
            if method == 'global':
                prob = generate_global_probability(df, target_col)
            elif method == 'session':
                prob = generate_session_probability(df, target_col)
            elif method == 'model':
                prob = generate_feature_based_probability(df, target_col, method='logistic')
            elif method == 'r_based' and 'R_multiple' in df.columns:
                prob = generate_r_multiple_probability(df)
            else:
                continue
            
            probs_list.append(prob)
            print(f"[PROB GEN] Added {method} probability")
            
        except Exception as e:
            print(f"[PROB GEN] Error with {method}: {e}")
            continue
    
    if len(probs_list) == 0:
        print("[PROB GEN] No methods succeeded, using global probability")
        return generate_global_probability(df, target_col)
    
    # Average all probabilities
    prob_df = pd.concat(probs_list, axis=1)
    composite_prob = prob_df.mean(axis=1)
    composite_prob.name = 'prob_composite_win'
    
    print(f"[PROB GEN] Composite probability created from {len(probs_list)} methods")
    print(f"  - Range: [{composite_prob.min():.3f}, {composite_prob.max():.3f}]")
    print(f"  - Mean: {composite_prob.mean():.3f}")
    
    return composite_prob


def auto_generate_all_probabilities(
    df: pd.DataFrame,
    target_col: str = 'trade_success',
    include_model: bool = True
) -> pd.DataFrame:
    """
    Auto-generate all probability columns and add to DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Trading data
    target_col : str
        Target column name
    include_model : bool
        Whether to include model-based probability (slower but more accurate)
        
    Returns:
    --------
    pd.DataFrame
        Original DataFrame with added probability columns
    """
    print(f"\n{'='*80}")
    print(f"[PROB GEN] Auto-generating probability columns...")
    print(f"{'='*80}")
    
    df_result = df.copy()
    
    # 1. Global probability
    try:
        df_result['prob_global_win'] = generate_global_probability(df, target_col)
        print(f"[PROB GEN] ✅ Generated: prob_global_win")
    except Exception as e:
        print(f"[PROB GEN] ❌ Failed: prob_global_win - {e}")
    
    # 2. Session probability
    try:
        df_result['prob_session_win'] = generate_session_probability(df, target_col)
        print(f"[PROB GEN] ✅ Generated: prob_session_win")
    except Exception as e:
        print(f"[PROB GEN] ❌ Failed: prob_session_win - {e}")
    
    # 3. R-based probability
    if 'R_multiple' in df.columns:
        try:
            df_result['prob_r_based_win'] = generate_r_multiple_probability(df)
            print(f"[PROB GEN] ✅ Generated: prob_r_based_win")
        except Exception as e:
            print(f"[PROB GEN] ❌ Failed: prob_r_based_win - {e}")
    
    # 4. Model-based probability (optional, slower)
    if include_model:
        try:
            df_result['prob_model_win'] = generate_feature_based_probability(df, target_col)
            print(f"[PROB GEN] ✅ Generated: prob_model_win")
        except Exception as e:
            print(f"[PROB GEN] ❌ Failed: prob_model_win - {e}")
    
    # 5. Composite probability
    try:
        methods = ['global', 'session']
        if 'R_multiple' in df.columns:
            methods.append('r_based')
        if include_model:
            methods.append('model')
        
        df_result['prob_composite_win'] = generate_composite_probability(df, target_col, methods)
        print(f"[PROB GEN] ✅ Generated: prob_composite_win")
    except Exception as e:
        print(f"[PROB GEN] ❌ Failed: prob_composite_win - {e}")
    
    # Summary
    prob_cols = [col for col in df_result.columns if col.startswith('prob_')]
    print(f"\n[PROB GEN] Summary:")
    print(f"  - Total probability columns generated: {len(prob_cols)}")
    print(f"  - Columns: {prob_cols}")
    print(f"{'='*80}\n")
    
    return df_result


def get_probability_info(df: pd.DataFrame) -> Dict:
    """
    Get information about probability columns in DataFrame.
    
    Returns:
    --------
    dict
        Information about available probability columns
    """
    prob_cols = [col for col in df.columns if 'prob' in col.lower()]
    
    info = {
        'has_probability_columns': len(prob_cols) > 0,
        'probability_columns': prob_cols,
        'count': len(prob_cols)
    }
    
    if len(prob_cols) > 0:
        stats = {}
        for col in prob_cols:
            try:
                # Convert to numeric if needed
                col_data = pd.to_numeric(df[col], errors='coerce')
                stats[col] = {
                    'mean': float(col_data.mean()),
                    'std': float(col_data.std()),
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                    'nan_count': int(col_data.isna().sum())
                }
            except Exception as e:
                print(f"[PROB INFO] Warning: Cannot get stats for {col}: {e}")
                stats[col] = {
                    'mean': np.nan,
                    'std': np.nan,
                    'min': np.nan,
                    'max': np.nan,
                    'nan_count': int(df[col].isna().sum())
                }
        info['statistics'] = stats
    
    return info


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n = 1000
    
    df = pd.DataFrame({
        'Timestamp': pd.date_range('2023-01-01', periods=n, freq='1H'),
        'R_multiple': np.random.randn(n) * 2,
        'feature1': np.random.randn(n),
        'feature2': np.random.randn(n),
        'feature3': np.random.randn(n),
    })
    
    df['trade_success'] = (df['R_multiple'] > 0).astype(int)
    
    # Generate probabilities
    df_with_probs = auto_generate_all_probabilities(df, target_col='trade_success')
    
    # Show info
    info = get_probability_info(df_with_probs)
    print("\nProbability Info:")
    print(info)
    
    # Show sample
    print("\nSample data:")
    print(df_with_probs[['trade_success'] + info['probability_columns']].head(10))
