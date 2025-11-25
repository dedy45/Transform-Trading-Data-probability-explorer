"""
What-If Scenario Analysis Demo

This script demonstrates the usage of the What-If Scenario Analysis module
for trading strategy optimization.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from backend.calculators.whatif_scenario import WhatIfScenario


def generate_sample_trades(n_trades=200):
    """Generate sample trade data for demonstration"""
    np.random.seed(42)
    
    trades = pd.DataFrame({
        'Ticket_id': range(1, n_trades + 1),
        'entry_time': pd.date_range('2024-01-01', periods=n_trades, freq='6h'),
        'exit_time': pd.date_range('2024-01-01 03:00', periods=n_trades, freq='6h'),
        'Type': np.random.choice(['BUY', 'SELL'], n_trades),
        'R_multiple': np.random.normal(0.6, 1.8, n_trades),
        'trade_success': np.random.choice([0, 1], n_trades, p=[0.42, 0.58]),
        'net_profit': np.random.normal(60, 120, n_trades),
        'gross_profit': np.random.normal(120, 180, n_trades),
        'Volume': np.random.uniform(0.1, 2.0, n_trades),
        'risk_percent': np.ones(n_trades) * 1.0,
        'money_risk': np.ones(n_trades) * 100,
        'MAE_R': np.random.uniform(0, 1.5, n_trades),
        'MFE_R': np.random.uniform(0, 3.5, n_trades),
        'SessionHour': np.random.randint(0, 24, n_trades),
        'SessionDayOfWeek': np.random.randint(0, 7, n_trades),
        'session': np.random.choice([0, 1, 2, 3], n_trades),
        'entry_session': np.random.choice(['ASIA', 'EUROPE', 'US'], n_trades),
        'trend_tf_dir': np.random.choice([-1, 0, 1], n_trades),
        'trend_strength_tf': np.random.uniform(0, 1, n_trades),
        'trend_regime': np.random.choice([0, 1], n_trades),
        'volatility_regime': np.random.choice([0, 1, 2], n_trades),
        'risk_regime_global': np.random.choice([0, 1], n_trades),
        'ap_entropy_m1_2h': np.random.uniform(0, 1, n_trades),
        'hurst_m5_2d': np.random.uniform(0, 1, n_trades),
        'prob_global_win': np.random.uniform(0.4, 0.85, n_trades),
        'composite_score': np.random.uniform(35, 95, n_trades),
        'minutes_to_next_high_impact_news': np.random.randint(-120, 120, n_trades)
    })
    
    # Ensure trade_success matches R_multiple sign
    trades.loc[trades['R_multiple'] > 0, 'trade_success'] = 1
    trades.loc[trades['R_multiple'] <= 0, 'trade_success'] = 0
    
    return trades


def demo_baseline_analysis():
    """Demo 1: Baseline Analysis"""
    print("=" * 80)
    print("DEMO 1: BASELINE ANALYSIS")
    print("=" * 80)
    
    trades = generate_sample_trades()
    scenario = WhatIfScenario(trades)
    
    print("\nBaseline Performance Metrics:")
    print("-" * 80)
    for key, value in scenario.baseline_metrics.items():
        if isinstance(value, float):
            print(f"{key:20s}: {value:10.2f}")
        else:
            print(f"{key:20s}: {value:10}")
    print()


def demo_position_sizing():
    """Demo 2: Position Sizing Scenarios"""
    print("=" * 80)
    print("DEMO 2: POSITION SIZING SCENARIOS")
    print("=" * 80)
    
    trades = generate_sample_trades()
    scenario = WhatIfScenario(trades)
    
    print("\nTesting different risk levels:")
    print("-" * 80)
    
    risk_levels = [0.5, 1.0, 1.5, 2.0, 2.5]
    
    for risk in risk_levels:
        metrics = scenario.apply_position_sizing_scenario(risk_percent=risk)
        print(f"\nRisk {risk}% per trade:")
        print(f"  Total Profit: ${metrics['total_profit']:,.2f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']:.2f}%")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print()


def demo_sl_tp_optimization():
    """Demo 3: SL/TP Optimization"""
    print("=" * 80)
    print("DEMO 3: SL/TP OPTIMIZATION")
    print("=" * 80)
    
    trades = generate_sample_trades()
    scenario = WhatIfScenario(trades)
    
    print("\nTesting different SL/TP combinations:")
    print("-" * 80)
    
    sl_multipliers = [0.5, 0.75, 1.0, 1.25, 1.5]
    tp_multipliers = [1.0, 1.5, 2.0, 2.5]
    
    best_expectancy = -float('inf')
    best_combo = None
    
    for sl in sl_multipliers:
        for tp in tp_multipliers:
            metrics = scenario.apply_sl_tp_scenario(sl_multiplier=sl, tp_multiplier=tp)
            
            if metrics['expectancy'] > best_expectancy:
                best_expectancy = metrics['expectancy']
                best_combo = (sl, tp)
            
            print(f"SL {sl:.2f}x, TP {tp:.2f}x: "
                  f"Win Rate {metrics['win_rate']:.1f}%, "
                  f"Avg R {metrics['avg_r']:.2f}, "
                  f"Expectancy ${metrics['expectancy']:.2f}")
    
    print(f"\nBest Combination: SL {best_combo[0]:.2f}x, TP {best_combo[1]:.2f}x")
    print(f"Best Expectancy: ${best_expectancy:.2f}")
    print()


def demo_filtering():
    """Demo 4: Trade Filtering"""
    print("=" * 80)
    print("DEMO 4: TRADE FILTERING SCENARIOS")
    print("=" * 80)
    
    trades = generate_sample_trades()
    scenario = WhatIfScenario(trades)
    
    print("\nBaseline:")
    print(f"  Total Trades: {scenario.baseline_metrics['total_trades']}")
    print(f"  Win Rate: {scenario.baseline_metrics['win_rate']:.2f}%")
    print(f"  Expectancy: ${scenario.baseline_metrics['expectancy']:.2f}")
    
    # Test probability filter
    print("\n1. High Probability Filter (>65%):")
    metrics = scenario.apply_filter_scenario({'min_probability': 0.65})
    print(f"  Remaining Trades: {metrics['total_trades']}")
    print(f"  Win Rate: {metrics['win_rate']:.2f}%")
    print(f"  Expectancy: ${metrics['expectancy']:.2f}")
    
    # Test composite score filter
    print("\n2. High Quality Filter (Score >70):")
    metrics = scenario.apply_filter_scenario({'min_composite_score': 70})
    print(f"  Remaining Trades: {metrics['total_trades']}")
    print(f"  Win Rate: {metrics['win_rate']:.2f}%")
    print(f"  Expectancy: ${metrics['expectancy']:.2f}")
    
    # Test combined filters
    print("\n3. Combined Filters (Prob >60%, Score >65, Trend Aligned):")
    filters = {
        'min_probability': 0.60,
        'min_composite_score': 65,
        'trend_alignment': True
    }
    metrics = scenario.apply_filter_scenario(filters)
    print(f"  Remaining Trades: {metrics['total_trades']}")
    print(f"  Trades Filtered: {metrics['trades_filtered_out']} ({metrics['filter_percentage']:.1f}%)")
    print(f"  Win Rate: {metrics['win_rate']:.2f}%")
    print(f"  Expectancy: ${metrics['expectancy']:.2f}")
    print()


def demo_time_restrictions():
    """Demo 5: Time-Based Restrictions"""
    print("=" * 80)
    print("DEMO 5: TIME-BASED RESTRICTIONS")
    print("=" * 80)
    
    trades = generate_sample_trades()
    scenario = WhatIfScenario(trades)
    
    # Test trading hours
    print("\n1. Trading Hours 8 AM - 4 PM:")
    metrics = scenario.apply_time_scenario({'trading_hours': (8, 16)})
    print(f"  Remaining Trades: {metrics['total_trades']}")
    print(f"  Win Rate: {metrics['win_rate']:.2f}%")
    
    # Test weekdays only
    print("\n2. Weekdays Only:")
    metrics = scenario.apply_time_scenario({'days_of_week': [0, 1, 2, 3, 4]})
    print(f"  Remaining Trades: {metrics['total_trades']}")
    print(f"  Win Rate: {metrics['win_rate']:.2f}%")
    
    # Test session filter
    print("\n3. Europe and US Sessions Only:")
    metrics = scenario.apply_time_scenario({'sessions_only': ['EUROPE', 'US']})
    print(f"  Remaining Trades: {metrics['total_trades']}")
    print(f"  Win Rate: {metrics['win_rate']:.2f}%")
    print()


def demo_market_conditions():
    """Demo 6: Market Condition Filtering"""
    print("=" * 80)
    print("DEMO 6: MARKET CONDITION FILTERING")
    print("=" * 80)
    
    trades = generate_sample_trades()
    scenario = WhatIfScenario(trades)
    
    # Test trending markets only
    print("\n1. Trending Markets Only:")
    metrics = scenario.apply_market_condition_scenario({'trend_regime': [1]})
    print(f"  Remaining Trades: {metrics['total_trades']}")
    print(f"  Win Rate: {metrics['win_rate']:.2f}%")
    print(f"  Avg R: {metrics['avg_r']:.2f}")
    
    # Test low volatility
    print("\n2. Low Volatility Only:")
    metrics = scenario.apply_market_condition_scenario({'volatility_regime': [0]})
    print(f"  Remaining Trades: {metrics['total_trades']}")
    print(f"  Win Rate: {metrics['win_rate']:.2f}%")
    
    # Test combined conditions
    print("\n3. Trending + Low/Medium Volatility:")
    filters = {
        'trend_regime': [1],
        'volatility_regime': [0, 1],
        'min_hurst': 0.5
    }
    metrics = scenario.apply_market_condition_scenario(filters)
    print(f"  Remaining Trades: {metrics['total_trades']}")
    print(f"  Win Rate: {metrics['win_rate']:.2f}%")
    print(f"  Expectancy: ${metrics['expectancy']:.2f}")
    print()


def demo_money_management():
    """Demo 7: Money Management Strategies"""
    print("=" * 80)
    print("DEMO 7: MONEY MANAGEMENT STRATEGIES")
    print("=" * 80)
    
    trades = generate_sample_trades()
    scenario = WhatIfScenario(trades)
    
    # Test compounding
    print("\n1. Compounding:")
    metrics = scenario.apply_money_management_scenario({'compounding': True})
    print(f"  Final Equity: ${metrics['final_equity']:,.2f}")
    print(f"  Equity Growth: {metrics['equity_growth']:.2f}%")
    
    # Test martingale
    print("\n2. Martingale (1.5x after loss):")
    metrics = scenario.apply_money_management_scenario({'martingale_multiplier': 1.5})
    print(f"  Final Equity: ${metrics['final_equity']:,.2f}")
    print(f"  Max Drawdown: {metrics['max_drawdown']:.2f}%")
    
    # Test daily limits
    print("\n3. Daily Limits (Target: $500, Limit: $300):")
    mm_rules = {
        'daily_profit_target': 500,
        'daily_loss_limit': 300
    }
    metrics = scenario.apply_money_management_scenario(mm_rules)
    print(f"  Final Equity: ${metrics['final_equity']:,.2f}")
    print(f"  Total Profit: ${metrics['total_profit']:,.2f}")
    print()


def demo_scenario_comparison():
    """Demo 8: Scenario Comparison"""
    print("=" * 80)
    print("DEMO 8: SCENARIO COMPARISON")
    print("=" * 80)
    
    trades = generate_sample_trades()
    scenario = WhatIfScenario(trades)
    
    scenarios = [
        {
            'name': 'Conservative',
            'type': 'position_sizing',
            'params': {'risk_percent': 0.5}
        },
        {
            'name': 'Moderate',
            'type': 'position_sizing',
            'params': {'risk_percent': 1.5}
        },
        {
            'name': 'Aggressive',
            'type': 'position_sizing',
            'params': {'risk_percent': 2.5}
        },
        {
            'name': 'High Probability',
            'type': 'filter',
            'params': {'min_probability': 0.7}
        },
        {
            'name': 'Trending Only',
            'type': 'market_condition',
            'params': {'trend_regime': [1]}
        }
    ]
    
    comparison_df = scenario.compare_scenarios(scenarios)
    
    print("\nScenario Comparison:")
    print("-" * 80)
    print(comparison_df[['scenario_name', 'total_trades', 'win_rate', 'avg_r', 
                         'expectancy', 'total_profit', 'max_drawdown']].to_string(index=False))
    print()


def demo_optimization():
    """Demo 9: Parameter Optimization"""
    print("=" * 80)
    print("DEMO 9: PARAMETER OPTIMIZATION")
    print("=" * 80)
    
    trades = generate_sample_trades()
    scenario = WhatIfScenario(trades)
    
    print("\nOptimizing Risk Percentage for Maximum Sharpe Ratio:")
    print("-" * 80)
    
    param_ranges = {'risk_percent': (0.5, 3.0)}
    
    result = scenario.optimize_scenario(
        scenario_type='position_sizing',
        param_ranges=param_ranges,
        objective='sharpe_ratio',
        constraints={'min_trades': 100}
    )
    
    print(f"\nOptimal Risk Percentage: {result['optimal_params']['risk_percent']:.2f}%")
    print(f"Optimization Success: {result['optimization_success']}")
    print("\nOptimal Metrics:")
    for key, value in result['optimal_metrics'].items():
        if isinstance(value, float):
            print(f"  {key:20s}: {value:10.2f}")
    print()


def main():
    """Run all demos"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "WHAT-IF SCENARIO ANALYSIS DEMO" + " " * 28 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    demo_baseline_analysis()
    demo_position_sizing()
    demo_sl_tp_optimization()
    demo_filtering()
    demo_time_restrictions()
    demo_market_conditions()
    demo_money_management()
    demo_scenario_comparison()
    demo_optimization()
    
    print("=" * 80)
    print("ALL DEMOS COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print()


if __name__ == '__main__':
    main()
