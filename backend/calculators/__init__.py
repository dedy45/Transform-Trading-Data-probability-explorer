# Calculator modules for probability analysis

from .probability_calculator import (
    compute_1d_probability,
    compute_2d_probability,
    get_probability_summary
)

from .expectancy_calculator import (
    compute_expectancy_R,
    compute_expectancy_by_group,
    compute_r_percentiles,
    compute_r_threshold_probabilities
)

from .mae_mfe_analyzer import (
    analyze_mae_patterns,
    analyze_mfe_patterns,
    calculate_profit_left,
    optimize_sl_level,
    optimize_tp_level
)

from .monte_carlo_engine import (
    monte_carlo_simulation,
    kelly_criterion_calculator,
    compare_risk_scenarios
)

from .conditional_probability import (
    calculate_conditional_probability,
    calculate_lift_ratio,
    sequential_condition_builder,
    filter_by_thresholds,
    sort_by_probability_and_significance,
    find_top_combinations
)

__all__ = [
    # Probability Calculator
    'compute_1d_probability',
    'compute_2d_probability',
    'get_probability_summary',
    
    # Expectancy Calculator
    'compute_expectancy_R',
    'compute_expectancy_by_group',
    'compute_r_percentiles',
    'compute_r_threshold_probabilities',
    
    # MAE/MFE Analyzer
    'analyze_mae_patterns',
    'analyze_mfe_patterns',
    'calculate_profit_left',
    'optimize_sl_level',
    'optimize_tp_level',
    
    # Monte Carlo Engine
    'monte_carlo_simulation',
    'kelly_criterion_calculator',
    'compare_risk_scenarios',
    
    # Conditional Probability
    'calculate_conditional_probability',
    'calculate_lift_ratio',
    'sequential_condition_builder',
    'filter_by_thresholds',
    'sort_by_probability_and_significance',
    'find_top_combinations',
]
