"""
Interactive Filter Demo

This script demonstrates the usage of the InteractiveFilter class
for filtering trade data with various criteria.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.calculators.interactive_filter import InteractiveFilter


def generate_sample_data(n=500):
    """Generate sample trade data for demonstration"""
    np.random.seed(42)
    
    base_date = datetime(2024, 1, 1)
    dates = [base_date + timedelta(hours=i*6) for i in range(n)]
    
    df = pd.DataFrame({
        'entry_time': dates,
        'exit_time': [d + timedelta(hours=np.random.randint(1, 8)) for d in dates],
        'entry_session': np.random.choice(['ASIA', 'EUROPE', 'US', 'OVERLAP'], n, p=[0.25, 0.35, 0.30, 0.10]),
        'R_multiple': np.random.normal(0.5, 2, n),
        'trade_success': np.random.choice([0, 1], n, p=[0.42, 0.58]),
        'net_profit': np.random.normal(100, 250, n),
        'holding_minutes': np.random.randint(30, 480, n),
        'prob_global_win': np.random.uniform(0.35, 0.85, n),
        'composite_score': np.random.uniform(45, 92, n),
        'trend_regime': np.random.choice([0, 1], n, p=[0.4, 0.6]),
        'volatility_regime': np.random.choice([0, 1, 2], n, p=[0.3, 0.5, 0.2]),
        'risk_regime_global': np.random.choice([0, 1], n, p=[0.6, 0.4]),
        'trend_strength_tf': np.random.uniform(0, 1, n),
        'atr_tf_14': np.random.uniform(0.5, 2.5, n),
        'ap_entropy_m1_2h': np.random.uniform(0, 1, n),
        'hurst_m5_2d': np.random.uniform(0.3, 0.7, n),
    })
    
    return df


def print_section(title):
    """Print a section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def print_summary(filter_sys):
    """Print filter summary"""
    summary = filter_sys.get_filter_summary()
    print(f"\nFilter Summary:")
    print(f"  Original trades: {summary['original_count']}")
    print(f"  Filtered trades: {summary['filtered_count']}")
    print(f"  Removed trades: {summary['removed_count']}")
    print(f"  Removal percentage: {summary['removal_percentage']:.1f}%")
    if summary['active_filters']:
        print(f"  Active filters: {', '.join(summary['active_filters'])}")


def demo_basic_filtering():
    """Demonstrate basic filtering operations"""
    print_section("1. Basic Filtering Operations")
    
    # Generate sample data
    trades_df = generate_sample_data(500)
    print(f"\nGenerated {len(trades_df)} sample trades")
    
    # Initialize filter system
    filter_sys = InteractiveFilter(trades_df)
    print("Initialized InteractiveFilter")
    
    # Add a simple filter
    print("\n--- Adding probability filter (min_prob=0.65) ---")
    filter_sys.add_filter(
        'high_prob',
        InteractiveFilter.probability_range_filter,
        {'min_prob': 0.65}
    )
    print_summary(filter_sys)
    
    # Add another filter
    print("\n--- Adding composite score filter (min_score=70) ---")
    filter_sys.add_filter(
        'high_score',
        InteractiveFilter.composite_score_filter,
        {'min_score': 70}
    )
    print_summary(filter_sys)
    
    # Remove a filter
    print("\n--- Removing probability filter ---")
    filter_sys.remove_filter('high_prob')
    print_summary(filter_sys)
    
    # Clear all filters
    print("\n--- Clearing all filters ---")
    filter_sys.clear_all_filters()
    print_summary(filter_sys)


def demo_session_filtering():
    """Demonstrate session-based filtering"""
    print_section("2. Session-Based Filtering")
    
    trades_df = generate_sample_data(500)
    filter_sys = InteractiveFilter(trades_df)
    
    # Filter for Europe session only
    print("\n--- Filtering for EUROPE session ---")
    filter_sys.add_filter(
        'europe_only',
        InteractiveFilter.session_filter,
        {'sessions': ['EUROPE']}
    )
    print_summary(filter_sys)
    
    # Show session distribution
    filtered_data = filter_sys.get_filtered_data()
    session_counts = filtered_data['entry_session'].value_counts()
    print("\nSession distribution:")
    for session, count in session_counts.items():
        print(f"  {session}: {count}")


def demo_market_conditions():
    """Demonstrate market condition filtering"""
    print_section("3. Market Condition Filtering")
    
    trades_df = generate_sample_data(500)
    filter_sys = InteractiveFilter(trades_df)
    
    # Filter for trending markets with low volatility
    print("\n--- Filtering for trending markets with low/medium volatility ---")
    filter_sys.add_filter(
        'market_conditions',
        InteractiveFilter.market_condition_filters,
        {
            'trend_regimes': [1],
            'volatility_regimes': [0, 1]
        }
    )
    print_summary(filter_sys)
    
    # Show market condition distribution
    filtered_data = filter_sys.get_filtered_data()
    print("\nMarket condition distribution:")
    print(f"  Trend regime: {filtered_data['trend_regime'].value_counts().to_dict()}")
    print(f"  Volatility regime: {filtered_data['volatility_regime'].value_counts().to_dict()}")


def demo_time_filtering():
    """Demonstrate time-based filtering"""
    print_section("4. Time-Based Filtering")
    
    trades_df = generate_sample_data(500)
    filter_sys = InteractiveFilter(trades_df)
    
    # Filter for trading hours (8 AM to 4 PM) on weekdays
    print("\n--- Filtering for 8 AM - 4 PM, Monday-Friday ---")
    filter_sys.add_filter(
        'trading_hours',
        InteractiveFilter.time_of_day_filter,
        {
            'hour_range': (8, 16),
            'days_of_week': [0, 1, 2, 3, 4]
        }
    )
    print_summary(filter_sys)
    
    # Show hour distribution
    filtered_data = filter_sys.get_filtered_data()
    hours = filtered_data['entry_time'].dt.hour
    print(f"\nHour range: {hours.min()} - {hours.max()}")
    print(f"Days of week: {sorted(filtered_data['entry_time'].dt.dayofweek.unique())}")


def demo_performance_filtering():
    """Demonstrate performance-based filtering"""
    print_section("5. Performance-Based Filtering")
    
    trades_df = generate_sample_data(500)
    filter_sys = InteractiveFilter(trades_df)
    
    # Filter for winning trades with good R-multiple
    print("\n--- Filtering for winners with R > 1 ---")
    filter_sys.add_filter(
        'good_winners',
        InteractiveFilter.performance_filters,
        {'r_multiple_range': (1.0, 10.0)}
    )
    print_summary(filter_sys)
    
    # Show performance stats
    filtered_data = filter_sys.get_filtered_data()
    print("\nPerformance statistics:")
    print(f"  Mean R-multiple: {filtered_data['R_multiple'].mean():.2f}")
    print(f"  Median R-multiple: {filtered_data['R_multiple'].median():.2f}")
    print(f"  Min R-multiple: {filtered_data['R_multiple'].min():.2f}")
    print(f"  Max R-multiple: {filtered_data['R_multiple'].max():.2f}")


def demo_complex_filtering():
    """Demonstrate complex multi-filter scenarios"""
    print_section("6. Complex Multi-Filter Scenario")
    
    trades_df = generate_sample_data(500)
    filter_sys = InteractiveFilter(trades_df)
    
    print("\n--- Building complex filter: High quality Europe session trades ---")
    
    # Add multiple filters
    filter_sys.add_filter('session', InteractiveFilter.session_filter, 
                         {'sessions': ['EUROPE']})
    print("  ✓ Added session filter (EUROPE)")
    
    filter_sys.add_filter('probability', InteractiveFilter.probability_range_filter, 
                         {'min_prob': 0.65})
    print("  ✓ Added probability filter (>= 0.65)")
    
    filter_sys.add_filter('score', InteractiveFilter.composite_score_filter, 
                         {'min_score': 70})
    print("  ✓ Added composite score filter (>= 70)")
    
    filter_sys.add_filter('market', InteractiveFilter.market_condition_filters, 
                         {'trend_regimes': [1]})
    print("  ✓ Added market condition filter (trending)")
    
    filter_sys.add_filter('performance', InteractiveFilter.performance_filters, 
                         {'r_multiple_range': (0, 10)})
    print("  ✓ Added performance filter (R >= 0)")
    
    print_summary(filter_sys)
    
    # Show final statistics
    filtered_data = filter_sys.get_filtered_data()
    if len(filtered_data) > 0:
        print("\nFinal filtered data statistics:")
        print(f"  Win rate: {(filtered_data['trade_success'].sum() / len(filtered_data) * 100):.1f}%")
        print(f"  Mean R: {filtered_data['R_multiple'].mean():.2f}")
        print(f"  Mean probability: {filtered_data['prob_global_win'].mean():.2f}")
        print(f"  Mean score: {filtered_data['composite_score'].mean():.1f}")


def demo_presets():
    """Demonstrate filter preset functionality"""
    print_section("7. Filter Presets")
    
    trades_df = generate_sample_data(500)
    filter_sys = InteractiveFilter(trades_df)
    
    # Show available presets
    print("\nAvailable presets:")
    presets = filter_sys.get_available_presets()
    for name, description in presets.items():
        print(f"  • {name}: {description}")
    
    # Load a preset
    print("\n--- Loading 'high_probability' preset ---")
    filter_sys.load_filter_preset('high_probability')
    print_summary(filter_sys)
    
    # Create and save custom preset
    print("\n--- Creating custom preset ---")
    filter_sys.clear_all_filters()
    filter_sys.add_filter('session', InteractiveFilter.session_filter, 
                         {'sessions': ['ASIA', 'EUROPE']})
    filter_sys.add_filter('score', InteractiveFilter.composite_score_filter, 
                         {'min_score': 75})
    
    filter_sys.save_filter_preset('my_custom_preset', 'Asia/Europe sessions with score > 75')
    print("Saved custom preset: 'my_custom_preset'")
    
    # Verify it was saved
    presets = filter_sys.get_available_presets()
    if 'my_custom_preset' in presets:
        print(f"  ✓ Preset saved: {presets['my_custom_preset']}")


def demo_dynamic_updates():
    """Demonstrate dynamic filter updates"""
    print_section("8. Dynamic Filter Updates")
    
    trades_df = generate_sample_data(500)
    filter_sys = InteractiveFilter(trades_df)
    
    # Start with one threshold
    print("\n--- Initial filter: probability >= 0.6 ---")
    filter_sys.add_filter('prob', InteractiveFilter.probability_range_filter, 
                         {'min_prob': 0.6})
    print_summary(filter_sys)
    
    # Update to more restrictive
    print("\n--- Updating filter: probability >= 0.7 ---")
    filter_sys.update_filter('prob', {'min_prob': 0.7})
    print_summary(filter_sys)
    
    # Update to even more restrictive
    print("\n--- Updating filter: probability >= 0.8 ---")
    filter_sys.update_filter('prob', {'min_prob': 0.8})
    print_summary(filter_sys)


def main():
    """Run all demonstrations"""
    print("\n" + "="*70)
    print("  INTERACTIVE FILTER SYSTEM DEMONSTRATION")
    print("="*70)
    
    try:
        demo_basic_filtering()
        demo_session_filtering()
        demo_market_conditions()
        demo_time_filtering()
        demo_performance_filtering()
        demo_complex_filtering()
        demo_presets()
        demo_dynamic_updates()
        
        print("\n" + "="*70)
        print("  DEMONSTRATION COMPLETE")
        print("="*70)
        print("\nAll demonstrations completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
