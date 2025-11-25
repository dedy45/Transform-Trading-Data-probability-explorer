"""
What-If Scenario Analysis Module

This module provides functionality for simulating "what-if" scenarios by modifying
trading parameters and analyzing their impact on performance metrics.

Inspired by StrategyQuant QuantAnalyzer.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from scipy import optimize


class WhatIfScenario:
    """
    What-If Scenario analyzer for trading strategy optimization.
    
    This class allows users to simulate different trading scenarios by modifying
    parameters such as position sizing, stop loss/take profit levels, filters,
    time restrictions, market conditions, and money management rules.
    
    Attributes:
        trades (pd.DataFrame): Historical trades dataframe
        baseline_metrics (dict): Baseline performance metrics
    """
    
    def __init__(self, historical_trades_df: pd.DataFrame):
        """
        Initialize What-If Scenario analyzer.
        
        Args:
            historical_trades_df: DataFrame containing historical trade data
                Required columns: R_multiple, trade_success, net_profit, 
                                 entry_time, exit_time, etc.
        """
        self.trades = historical_trades_df.copy()
        self.baseline_metrics = self._calculate_baseline()
    
    def _calculate_baseline(self) -> Dict[str, Any]:
        """
        Calculate baseline performance metrics.
        
        Returns:
            Dictionary containing baseline metrics:
                - total_trades: Total number of trades
                - win_rate: Percentage of winning trades
                - avg_r: Average R-multiple
                - expectancy: Expected profit per trade
                - total_profit: Total net profit
                - max_drawdown: Maximum drawdown percentage
                - profit_factor: Ratio of gross profit to gross loss
                - sharpe_ratio: Risk-adjusted return metric
        """
        if len(self.trades) == 0:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_r': 0.0,
                'expectancy': 0.0,
                'total_profit': 0.0,
                'max_drawdown': 0.0,
                'profit_factor': 0.0,
                'sharpe_ratio': 0.0
            }
        
        total_trades = len(self.trades)
        wins = self.trades[self.trades['trade_success'] == 1]
        losses = self.trades[self.trades['trade_success'] == 0]
        
        win_rate = len(wins) / total_trades if total_trades > 0 else 0.0
        avg_r = self.trades['R_multiple'].mean()
        expectancy = self.trades['net_profit'].mean()
        total_profit = self.trades['net_profit'].sum()
        
        # Calculate max drawdown using equity_after_trade if available
        if 'equity_after_trade' in self.trades.columns:
            # Use actual equity values from CSV
            equity_curve = self.trades['equity_after_trade']
            initial_equity = equity_curve.iloc[0] if len(equity_curve) > 0 else 10000
            lowest_equity = equity_curve.min()
            
            # Calculate drawdown: (Initial - Lowest) / Initial * 100
            if initial_equity > 0:
                max_drawdown = -((initial_equity - lowest_equity) / initial_equity * 100)
                # Cap at reasonable values (-100% to 0%)
                max_drawdown = max(min(max_drawdown, 0.0), -100.0)
            else:
                max_drawdown = 0.0
        else:
            # Fallback: Calculate from cumulative profit
            cumulative_profit = self.trades['net_profit'].cumsum()
            running_max = cumulative_profit.cummax()
            drawdown = cumulative_profit - running_max
            
            # Calculate max drawdown percentage with proper validation
            peak = running_max.max()
            worst_dd = drawdown.min()
            
            if peak > 10.0:  # Only use percentage if peak is significant (>$10)
                # Standard percentage drawdown calculation
                max_drawdown = (worst_dd / peak * 100)
                # Cap at reasonable values (-100% to 0%)
                max_drawdown = max(min(max_drawdown, 0.0), -100.0)
            elif peak > 0:
                # For small peaks, use percentage but with minimum threshold
                max_drawdown = (worst_dd / peak * 100)
                max_drawdown = max(min(max_drawdown, 0.0), -100.0)
            else:
                # No profit or negative, DD is 0
                max_drawdown = 0.0
        
        # Calculate profit factor
        gross_profit = wins['net_profit'].sum() if len(wins) > 0 else 0.0
        gross_loss = abs(losses['net_profit'].sum()) if len(losses) > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        
        # Calculate Sharpe ratio (assuming daily returns)
        returns = self.trades['net_profit']
        sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0.0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate * 100,
            'avg_r': avg_r,
            'expectancy': expectancy,
            'total_profit': total_profit,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio
        }

    def apply_ml_prediction_scenario(
        self,
        ml_predictions: pd.DataFrame,
        filter_by_quality: Optional[List[str]] = None,
        filter_by_prob_min: Optional[float] = None,
        filter_by_recommendation: bool = True
    ) -> Dict[str, Any]:
        """
        Apply ML prediction-based filtering scenario.
        
        This method filters trades based on ML predictions from the ML Prediction Engine.
        It allows filtering by quality labels (A+/A/B/C), minimum probability threshold,
        and recommendation (TRADE/SKIP).
        
        Args:
            ml_predictions: DataFrame with ML predictions containing columns:
                - prob_win_calibrated: Calibrated win probability
                - quality_label: Setup quality (A+/A/B/C)
                - recommendation: Trade recommendation (TRADE/SKIP)
                - R_P50_raw: Expected R-multiple
            filter_by_quality: List of quality labels to keep (e.g., ['A+', 'A'])
                If None, no quality filtering is applied
            filter_by_prob_min: Minimum probability threshold (0-1)
                If None, no probability filtering is applied
            filter_by_recommendation: If True, only keep trades with TRADE recommendation
        
        Returns:
            Dictionary containing recalculated metrics after ML filtering
        
        Example:
            >>> # Filter for only A+ and A quality setups
            >>> metrics = whatif.apply_ml_prediction_scenario(
            ...     ml_predictions,
            ...     filter_by_quality=['A+', 'A'],
            ...     filter_by_prob_min=0.55,
            ...     filter_by_recommendation=True
            ... )
        """
        # Make a copy of trades
        filtered_trades = self.trades.copy()
        
        # Ensure ml_predictions has same length as trades
        if len(ml_predictions) != len(filtered_trades):
            raise ValueError(
                f"ML predictions length ({len(ml_predictions)}) must match "
                f"trades length ({len(filtered_trades)})"
            )
        
        # Add ML prediction columns to trades
        filtered_trades['ml_prob_win'] = ml_predictions['prob_win_calibrated'].values
        filtered_trades['ml_quality'] = ml_predictions['quality_label'].values
        filtered_trades['ml_recommendation'] = ml_predictions['recommendation'].values
        filtered_trades['ml_expected_r'] = ml_predictions['R_P50_raw'].values
        
        # Apply filters
        mask = pd.Series([True] * len(filtered_trades))
        
        # Filter by quality
        if filter_by_quality is not None:
            mask &= filtered_trades['ml_quality'].isin(filter_by_quality)
        
        # Filter by probability
        if filter_by_prob_min is not None:
            mask &= filtered_trades['ml_prob_win'] >= filter_by_prob_min
        
        # Filter by recommendation
        if filter_by_recommendation:
            mask &= filtered_trades['ml_recommendation'] == 'TRADE'
        
        # Apply mask (use .values to avoid index alignment issues)
        filtered_trades = filtered_trades[mask.values]
        
        # Update trades temporarily
        original_trades = self.trades
        self.trades = filtered_trades
        
        # Calculate metrics
        metrics = self._calculate_baseline()
        
        # Restore original trades
        self.trades = original_trades
        
        return metrics

    def apply_position_sizing_scenario(
        self, 
        risk_percent: float = None, 
        max_position: float = None,
        fixed_lot_size: float = None
    ) -> Dict[str, Any]:
        """
        Apply position sizing changes and recalculate metrics.
        
        Args:
            risk_percent: Risk per trade as percentage (0.1 - 5.0)
            max_position: Maximum position size limit
            fixed_lot_size: Fixed lot size (if not using percentage risk)
        
        Returns:
            Dictionary containing recalculated metrics with position sizing applied
        """
        scenario_trades = self.trades.copy()
        
        if risk_percent is not None:
            # Recalculate profits based on new risk percentage
            # Assuming original risk was stored or can be inferred
            original_risk = scenario_trades['risk_percent'].mean() if 'risk_percent' in scenario_trades.columns else 1.0
            risk_multiplier = risk_percent / original_risk if original_risk > 0 else 1.0
            
            scenario_trades['net_profit'] = scenario_trades['net_profit'] * risk_multiplier
            scenario_trades['gross_profit'] = scenario_trades['gross_profit'] * risk_multiplier if 'gross_profit' in scenario_trades.columns else scenario_trades['net_profit']
        
        if fixed_lot_size is not None:
            # Apply fixed lot size
            if 'Volume' in scenario_trades.columns:
                original_volume = scenario_trades['Volume'].mean()
                volume_multiplier = fixed_lot_size / original_volume if original_volume > 0 else 1.0
                scenario_trades['net_profit'] = scenario_trades['net_profit'] * volume_multiplier
        
        if max_position is not None:
            # Cap position sizes
            if 'Volume' in scenario_trades.columns:
                scenario_trades.loc[scenario_trades['Volume'] > max_position, 'Volume'] = max_position
        
        # Recalculate metrics with modified trades
        temp_trades = self.trades
        self.trades = scenario_trades
        metrics = self._calculate_baseline()
        self.trades = temp_trades
        
        return metrics

    def apply_sl_tp_scenario(
        self, 
        sl_multiplier: float = 1.0, 
        tp_multiplier: float = 1.0
    ) -> Dict[str, Any]:
        """
        Apply SL/TP changes and recalculate metrics.
        
        This simulates what would happen if stop loss and take profit levels
        were adjusted by the given multipliers.
        
        Args:
            sl_multiplier: Multiplier for stop loss distance (0.5 - 3.0)
            tp_multiplier: Multiplier for take profit distance (0.5 - 5.0)
        
        Returns:
            Dictionary containing recalculated metrics with SL/TP adjustments
        """
        scenario_trades = self.trades.copy()
        
        # Adjust R-multiples based on SL/TP changes
        # For winners: may hit TP earlier or later
        # For losers: may hit SL earlier or later
        
        for idx, trade in scenario_trades.iterrows():
            if 'MAE_R' in scenario_trades.columns and 'MFE_R' in scenario_trades.columns:
                mae_r = trade['MAE_R']
                mfe_r = trade['MFE_R']
                original_r = trade['R_multiple']
                
                # Check if tighter SL would have stopped out
                if sl_multiplier < 1.0:
                    # Tighter SL
                    new_sl_level = sl_multiplier
                    if abs(mae_r) >= new_sl_level:
                        # Would have been stopped out
                        scenario_trades.at[idx, 'R_multiple'] = -new_sl_level
                        scenario_trades.at[idx, 'trade_success'] = 0
                
                # Check if different TP would have been hit
                if tp_multiplier != 1.0 and trade['trade_success'] == 1:
                    # Adjust TP level
                    new_tp_level = tp_multiplier * abs(original_r)
                    if mfe_r >= new_tp_level:
                        # Would have hit new TP
                        scenario_trades.at[idx, 'R_multiple'] = new_tp_level
                    elif mfe_r < new_tp_level and original_r > 0:
                        # Wouldn't have reached new TP, use MFE as exit
                        scenario_trades.at[idx, 'R_multiple'] = min(mfe_r, original_r)
        
        # Recalculate net_profit based on new R-multiples
        if 'money_risk' in scenario_trades.columns:
            scenario_trades['net_profit'] = scenario_trades['R_multiple'] * scenario_trades['money_risk']
        
        # Recalculate metrics
        temp_trades = self.trades
        self.trades = scenario_trades
        metrics = self._calculate_baseline()
        self.trades = temp_trades
        
        return metrics

    def apply_filter_scenario(self, filters_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply trade filters and recalculate metrics.
        
        Args:
            filters_dict: Dictionary of filters to apply
                Supported filters:
                - 'min_probability': Minimum probability threshold (0-1)
                - 'min_composite_score': Minimum composite score (0-100)
                - 'sessions': List of allowed sessions (e.g., ['ASIA', 'EUROPE'])
                - 'trend_alignment': Require trend alignment (True/False)
                - 'volatility_regime': List of allowed volatility regimes
                - 'min_trend_strength': Minimum trend strength
        
        Returns:
            Dictionary containing metrics for filtered trades
        """
        scenario_trades = self.trades.copy()
        
        # Apply probability filter
        if 'min_probability' in filters_dict and 'prob_global_win' in scenario_trades.columns:
            min_prob = filters_dict['min_probability']
            scenario_trades = scenario_trades[scenario_trades['prob_global_win'] >= min_prob]
        
        # Apply composite score filter
        if 'min_composite_score' in filters_dict and 'composite_score' in scenario_trades.columns:
            min_score = filters_dict['min_composite_score']
            scenario_trades = scenario_trades[scenario_trades['composite_score'] >= min_score]
        
        # Apply session filter
        if 'sessions' in filters_dict and 'session' in scenario_trades.columns:
            allowed_sessions = filters_dict['sessions']
            scenario_trades = scenario_trades[scenario_trades['session'].isin(allowed_sessions)]
        
        # Apply trend alignment filter
        if 'trend_alignment' in filters_dict and filters_dict['trend_alignment']:
            if 'trend_tf_dir' in scenario_trades.columns and 'Type' in scenario_trades.columns:
                # BUY trades should have positive trend, SELL trades negative trend
                scenario_trades = scenario_trades[
                    ((scenario_trades['Type'] == 'BUY') & (scenario_trades['trend_tf_dir'] > 0)) |
                    ((scenario_trades['Type'] == 'SELL') & (scenario_trades['trend_tf_dir'] < 0))
                ]
        
        # Apply volatility regime filter
        if 'volatility_regime' in filters_dict and 'volatility_regime' in scenario_trades.columns:
            allowed_regimes = filters_dict['volatility_regime']
            scenario_trades = scenario_trades[scenario_trades['volatility_regime'].isin(allowed_regimes)]
        
        # Apply trend strength filter
        if 'min_trend_strength' in filters_dict and 'trend_strength_tf' in scenario_trades.columns:
            min_strength = filters_dict['min_trend_strength']
            scenario_trades = scenario_trades[scenario_trades['trend_strength_tf'] >= min_strength]
        
        # Recalculate metrics
        temp_trades = self.trades
        self.trades = scenario_trades
        metrics = self._calculate_baseline()
        self.trades = temp_trades
        
        # Add filter impact info
        metrics['trades_filtered_out'] = len(self.trades) - len(scenario_trades)
        metrics['filter_percentage'] = (metrics['trades_filtered_out'] / len(self.trades) * 100) if len(self.trades) > 0 else 0
        
        return metrics

    def apply_time_scenario(self, time_restrictions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply time-based restrictions and recalculate metrics.
        
        Args:
            time_restrictions: Dictionary of time-based filters
                Supported restrictions:
                - 'trading_hours': Tuple of (start_hour, end_hour) e.g., (8, 16)
                - 'days_of_week': List of allowed days (0=Monday, 6=Sunday)
                - 'sessions_only': List of sessions to trade
                - 'avoid_first_hour': Avoid first hour of session (True/False)
                - 'avoid_last_hour': Avoid last hour of session (True/False)
                - 'news_blackout_minutes': Minutes before/after news to avoid
        
        Returns:
            Dictionary containing metrics for time-filtered trades
        """
        scenario_trades = self.trades.copy()
        
        # Apply trading hours filter
        if 'trading_hours' in time_restrictions:
            start_hour, end_hour = time_restrictions['trading_hours']
            if 'SessionHour' in scenario_trades.columns:
                scenario_trades = scenario_trades[
                    (scenario_trades['SessionHour'] >= start_hour) &
                    (scenario_trades['SessionHour'] <= end_hour)
                ]
            elif 'entry_time' in scenario_trades.columns:
                scenario_trades['hour'] = pd.to_datetime(scenario_trades['entry_time']).dt.hour
                scenario_trades = scenario_trades[
                    (scenario_trades['hour'] >= start_hour) &
                    (scenario_trades['hour'] <= end_hour)
                ]
        
        # Apply days of week filter
        if 'days_of_week' in time_restrictions:
            allowed_days = time_restrictions['days_of_week']
            if 'SessionDayOfWeek' in scenario_trades.columns:
                scenario_trades = scenario_trades[scenario_trades['SessionDayOfWeek'].isin(allowed_days)]
            elif 'entry_time' in scenario_trades.columns:
                scenario_trades['day_of_week'] = pd.to_datetime(scenario_trades['entry_time']).dt.dayofweek
                scenario_trades = scenario_trades[scenario_trades['day_of_week'].isin(allowed_days)]
        
        # Apply session filter
        if 'sessions_only' in time_restrictions and 'entry_session' in scenario_trades.columns:
            allowed_sessions = time_restrictions['sessions_only']
            scenario_trades = scenario_trades[scenario_trades['entry_session'].isin(allowed_sessions)]
        
        # Apply news blackout filter
        if 'news_blackout_minutes' in time_restrictions and 'minutes_to_next_high_impact_news' in scenario_trades.columns:
            blackout = time_restrictions['news_blackout_minutes']
            scenario_trades = scenario_trades[
                abs(scenario_trades['minutes_to_next_high_impact_news']) > blackout
            ]
        
        # Recalculate metrics
        temp_trades = self.trades
        self.trades = scenario_trades
        metrics = self._calculate_baseline()
        self.trades = temp_trades
        
        # Add time filter impact
        metrics['trades_filtered_out'] = len(self.trades) - len(scenario_trades)
        metrics['trades_per_day'] = len(scenario_trades) / max(1, (scenario_trades['entry_time'].max() - scenario_trades['entry_time'].min()).days) if 'entry_time' in scenario_trades.columns else 0
        
        return metrics

    def apply_market_condition_scenario(self, condition_filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply market condition filters and recalculate metrics.
        
        Args:
            condition_filters: Dictionary of market condition filters
                Supported filters:
                - 'trend_regime': List of allowed regimes (e.g., [1] for trending only)
                - 'volatility_regime': List of allowed volatility levels
                - 'risk_regime': List of allowed risk regimes
                - 'min_entropy': Minimum entropy threshold
                - 'max_entropy': Maximum entropy threshold
                - 'min_hurst': Minimum Hurst exponent
                - 'max_hurst': Maximum Hurst exponent
        
        Returns:
            Dictionary containing metrics for condition-filtered trades
        """
        scenario_trades = self.trades.copy()
        
        # Apply trend regime filter
        if 'trend_regime' in condition_filters and 'trend_regime' in scenario_trades.columns:
            allowed_regimes = condition_filters['trend_regime']
            scenario_trades = scenario_trades[scenario_trades['trend_regime'].isin(allowed_regimes)]
        
        # Apply volatility regime filter
        if 'volatility_regime' in condition_filters and 'volatility_regime' in scenario_trades.columns:
            allowed_vol = condition_filters['volatility_regime']
            scenario_trades = scenario_trades[scenario_trades['volatility_regime'].isin(allowed_vol)]
        
        # Apply risk regime filter
        if 'risk_regime' in condition_filters and 'risk_regime_global' in scenario_trades.columns:
            allowed_risk = condition_filters['risk_regime']
            scenario_trades = scenario_trades[scenario_trades['risk_regime_global'].isin(allowed_risk)]
        
        # Apply entropy filters
        if 'min_entropy' in condition_filters and 'ap_entropy_m1_2h' in scenario_trades.columns:
            min_entropy = condition_filters['min_entropy']
            scenario_trades = scenario_trades[scenario_trades['ap_entropy_m1_2h'] >= min_entropy]
        
        if 'max_entropy' in condition_filters and 'ap_entropy_m1_2h' in scenario_trades.columns:
            max_entropy = condition_filters['max_entropy']
            scenario_trades = scenario_trades[scenario_trades['ap_entropy_m1_2h'] <= max_entropy]
        
        # Apply Hurst exponent filters
        if 'min_hurst' in condition_filters and 'hurst_m5_2d' in scenario_trades.columns:
            min_hurst = condition_filters['min_hurst']
            scenario_trades = scenario_trades[scenario_trades['hurst_m5_2d'] >= min_hurst]
        
        if 'max_hurst' in condition_filters and 'hurst_m5_2d' in scenario_trades.columns:
            max_hurst = condition_filters['max_hurst']
            scenario_trades = scenario_trades[scenario_trades['hurst_m5_2d'] <= max_hurst]
        
        # Recalculate metrics
        temp_trades = self.trades
        self.trades = scenario_trades
        metrics = self._calculate_baseline()
        self.trades = temp_trades
        
        # Add condition filter impact
        metrics['trades_filtered_out'] = len(self.trades) - len(scenario_trades)
        
        return metrics

    def apply_money_management_scenario(self, mm_rules: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply money management rules and recalculate metrics.
        
        Args:
            mm_rules: Dictionary of money management rules
                Supported rules:
                - 'compounding': Enable compounding (True/False)
                - 'martingale_multiplier': Multiplier after loss (e.g., 2.0)
                - 'anti_martingale_multiplier': Multiplier after win (e.g., 1.5)
                - 'max_consecutive_losses': Pause trading after N losses
                - 'daily_profit_target': Stop trading after reaching target
                - 'daily_loss_limit': Stop trading after hitting limit
                - 'drawdown_reduction': Reduce size during drawdown (percentage)
        
        Returns:
            Dictionary containing metrics with money management applied
        """
        scenario_trades = self.trades.copy()
        
        # Sort by entry time to process sequentially
        if 'entry_time' in scenario_trades.columns:
            scenario_trades = scenario_trades.sort_values('entry_time')
        
        # Initialize tracking variables
        equity = 10000  # Starting equity
        consecutive_losses = 0
        daily_profit = 0
        current_date = None
        position_multiplier = 1.0
        
        modified_profits = []
        
        for idx, trade in scenario_trades.iterrows():
            # Check date change for daily limits
            if 'entry_time' in scenario_trades.columns:
                trade_date = pd.to_datetime(trade['entry_time']).date()
                if current_date != trade_date:
                    current_date = trade_date
                    daily_profit = 0
            
            # Check if should skip trade due to consecutive losses
            if 'max_consecutive_losses' in mm_rules:
                if consecutive_losses >= mm_rules['max_consecutive_losses']:
                    modified_profits.append(0)
                    continue
            
            # Check daily profit target
            if 'daily_profit_target' in mm_rules:
                if daily_profit >= mm_rules['daily_profit_target']:
                    modified_profits.append(0)
                    continue
            
            # Check daily loss limit
            if 'daily_loss_limit' in mm_rules:
                if daily_profit <= -mm_rules['daily_loss_limit']:
                    modified_profits.append(0)
                    continue
            
            # Apply position sizing based on money management rules
            base_profit = trade['net_profit']
            
            # Apply martingale/anti-martingale
            if 'martingale_multiplier' in mm_rules and consecutive_losses > 0:
                position_multiplier = mm_rules['martingale_multiplier'] ** consecutive_losses
            elif 'anti_martingale_multiplier' in mm_rules and consecutive_losses == 0:
                position_multiplier = mm_rules['anti_martingale_multiplier']
            else:
                position_multiplier = 1.0
            
            # Apply drawdown reduction
            if 'drawdown_reduction' in mm_rules:
                # Calculate current drawdown
                cumulative = sum(modified_profits) if modified_profits else 0
                peak = max(cumulative, 0)
                current_dd = (cumulative - peak) / equity if equity > 0 else 0
                
                if current_dd < -0.1:  # 10% drawdown
                    position_multiplier *= (1 - mm_rules['drawdown_reduction'])
            
            # Calculate modified profit
            modified_profit = base_profit * position_multiplier
            
            # Apply compounding
            if mm_rules.get('compounding', False):
                modified_profit = modified_profit * (equity / 10000)
            
            modified_profits.append(modified_profit)
            daily_profit += modified_profit
            equity += modified_profit
            
            # Update consecutive losses
            if trade['trade_success'] == 0:
                consecutive_losses += 1
            else:
                consecutive_losses = 0
        
        # Update scenario trades with modified profits
        scenario_trades['net_profit'] = modified_profits
        
        # Recalculate metrics
        temp_trades = self.trades
        self.trades = scenario_trades
        metrics = self._calculate_baseline()
        self.trades = temp_trades
        
        # Add money management specific metrics
        metrics['final_equity'] = equity
        metrics['equity_growth'] = ((equity - 10000) / 10000 * 100)
        
        return metrics

    def compare_scenarios(self, scenario_list: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare multiple scenarios side-by-side.
        
        Args:
            scenario_list: List of scenario dictionaries, each containing:
                - 'name': Scenario name
                - 'type': Scenario type ('position_sizing', 'sl_tp', 'filter', etc.)
                - 'params': Parameters for the scenario
        
        Returns:
            DataFrame with scenarios as rows and metrics as columns
        """
        comparison_data = []
        
        # Add baseline as first row
        baseline_row = {'scenario_name': 'Baseline', 'scenario_type': 'baseline'}
        baseline_row.update(self.baseline_metrics)
        comparison_data.append(baseline_row)
        
        # Process each scenario
        for scenario in scenario_list:
            scenario_name = scenario.get('name', 'Unnamed')
            scenario_type = scenario.get('type', 'unknown')
            params = scenario.get('params', {})
            
            # Apply appropriate scenario method
            if scenario_type == 'position_sizing':
                metrics = self.apply_position_sizing_scenario(**params)
            elif scenario_type == 'sl_tp':
                metrics = self.apply_sl_tp_scenario(**params)
            elif scenario_type == 'filter':
                metrics = self.apply_filter_scenario(params)
            elif scenario_type == 'time':
                metrics = self.apply_time_scenario(params)
            elif scenario_type == 'market_condition':
                metrics = self.apply_market_condition_scenario(params)
            elif scenario_type == 'money_management':
                metrics = self.apply_money_management_scenario(params)
            else:
                continue
            
            # Add scenario info to metrics
            scenario_row = {'scenario_name': scenario_name, 'scenario_type': scenario_type}
            scenario_row.update(metrics)
            comparison_data.append(scenario_row)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Calculate percentage changes from baseline
        if len(comparison_df) > 1:
            baseline_values = comparison_df.iloc[0]
            for col in ['win_rate', 'avg_r', 'expectancy', 'total_profit', 'max_drawdown', 'profit_factor', 'sharpe_ratio']:
                if col in comparison_df.columns:
                    comparison_df[f'{col}_change'] = (
                        (comparison_df[col] - baseline_values[col]) / baseline_values[col] * 100
                    ) if baseline_values[col] != 0 else 0
        
        return comparison_df

    def optimize_scenario(
        self, 
        scenario_type: str,
        param_ranges: Dict[str, Tuple[float, float]],
        objective: str = 'expectancy',
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Find optimal scenario parameters using optimization.
        
        Args:
            scenario_type: Type of scenario to optimize
                ('position_sizing', 'sl_tp', 'filter', etc.)
            param_ranges: Dictionary mapping parameter names to (min, max) tuples
                Example: {'risk_percent': (0.5, 3.0), 'sl_multiplier': (0.5, 2.0)}
            objective: Metric to optimize ('expectancy', 'sharpe_ratio', 
                      'profit_factor', 'total_profit')
            constraints: Optional constraints dictionary
                Example: {'min_trades': 100, 'min_win_rate': 50}
        
        Returns:
            Dictionary containing:
                - 'optimal_params': Best parameter values found
                - 'optimal_metrics': Metrics at optimal parameters
                - 'optimization_history': List of tried parameters and results
        """
        if constraints is None:
            constraints = {}
        
        # Define objective function
        def objective_function(params_array):
            # Convert array to parameter dictionary
            params_dict = {}
            param_names = list(param_ranges.keys())
            for i, param_name in enumerate(param_names):
                params_dict[param_name] = params_array[i]
            
            # Apply scenario
            if scenario_type == 'position_sizing':
                metrics = self.apply_position_sizing_scenario(**params_dict)
            elif scenario_type == 'sl_tp':
                metrics = self.apply_sl_tp_scenario(**params_dict)
            elif scenario_type == 'filter':
                metrics = self.apply_filter_scenario(params_dict)
            elif scenario_type == 'time':
                metrics = self.apply_time_scenario(params_dict)
            elif scenario_type == 'market_condition':
                metrics = self.apply_market_condition_scenario(params_dict)
            elif scenario_type == 'money_management':
                metrics = self.apply_money_management_scenario(params_dict)
            else:
                return float('inf')
            
            # Check constraints
            if 'min_trades' in constraints:
                if metrics['total_trades'] < constraints['min_trades']:
                    return float('inf')
            
            if 'min_win_rate' in constraints:
                if metrics['win_rate'] < constraints['min_win_rate']:
                    return float('inf')
            
            # Return negative of objective (since we're minimizing)
            return -metrics.get(objective, 0)
        
        # Set up bounds
        bounds = [param_ranges[param] for param in param_ranges.keys()]
        
        # Initial guess (midpoint of ranges)
        x0 = [(bounds[i][0] + bounds[i][1]) / 2 for i in range(len(bounds))]
        
        # Optimize
        result = optimize.minimize(
            objective_function,
            x0,
            method='L-BFGS-B',
            bounds=bounds
        )
        
        # Extract optimal parameters
        optimal_params = {}
        param_names = list(param_ranges.keys())
        for i, param_name in enumerate(param_names):
            optimal_params[param_name] = result.x[i]
        
        # Calculate metrics at optimal parameters
        if scenario_type == 'position_sizing':
            optimal_metrics = self.apply_position_sizing_scenario(**optimal_params)
        elif scenario_type == 'sl_tp':
            optimal_metrics = self.apply_sl_tp_scenario(**optimal_params)
        elif scenario_type == 'filter':
            optimal_metrics = self.apply_filter_scenario(optimal_params)
        elif scenario_type == 'time':
            optimal_metrics = self.apply_time_scenario(optimal_params)
        elif scenario_type == 'market_condition':
            optimal_metrics = self.apply_market_condition_scenario(optimal_params)
        elif scenario_type == 'money_management':
            optimal_metrics = self.apply_money_management_scenario(optimal_params)
        else:
            optimal_metrics = {}
        
        return {
            'optimal_params': optimal_params,
            'optimal_metrics': optimal_metrics,
            'optimization_success': result.success,
            'optimization_message': result.message
        }
