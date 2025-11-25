"""
Frontend components for Trading Probability Explorer
"""
from .probability_heatmap import (
    create_probability_heatmap_2d,
    create_empty_heatmap
)
from .distribution_chart import (
    create_probability_distribution_1d,
    create_confidence_interval_chart,
    create_empty_distribution
)
from .top_combinations import (
    create_combination_card,
    create_top_combinations_list,
    create_combinations_summary,
    format_combination_for_export
)
from .trade_details import (
    create_trade_details_summary,
    create_trade_list_table,
    create_mae_mfe_mini_chart,
    create_empty_trade_details
)
from .transition_matrix import (
    create_transition_matrix_heatmap,
    create_empty_transition_matrix
)
from .streak_distribution import (
    create_streak_distribution_chart,
    create_empty_streak_distribution
)
from .conditional_streak_chart import (
    create_conditional_streak_chart,
    create_empty_conditional_streak_chart
)
from .markov_summary_cards import (
    create_markov_summary_cards,
    create_empty_summary_cards
)
from .reliability_diagram import (
    create_reliability_diagram,
    create_empty_reliability_diagram
)
from .calibration_histogram import (
    create_calibration_histogram,
    create_empty_calibration_histogram
)
from .calibration_metrics_cards import (
    create_calibration_metrics_cards,
    create_empty_calibration_metrics_cards
)
from .calibration_table import (
    create_calibration_table,
    create_empty_calibration_table
)
from .regime_winrate_chart import (
    create_regime_winrate_chart,
    create_empty_regime_winrate_chart
)
from .regime_r_multiple_chart import (
    create_regime_r_multiple_chart,
    create_empty_regime_r_multiple_chart
)
from .regime_comparison_table import (
    create_regime_comparison_table,
    create_empty_regime_comparison_table
)
from .regime_filter_controls import (
    create_regime_filter_controls,
    create_regime_summary_cards,
    create_empty_regime_summary_cards
)
from .scenario_builder_params import (
    get_scenario_params_panel,
    create_position_sizing_params,
    create_sl_tp_params,
    create_filter_params,
    create_time_params,
    create_market_condition_params,
    create_money_management_params
)
from .scenario_comparison_table import (
    create_comparison_table,
    create_comparison_summary_cards,
    format_scenario_type,
    calculate_percentage_changes,
    get_color_for_change
)
from .equity_curve_comparison import (
    create_equity_curve_comparison_chart,
    create_drawdown_comparison_chart,
    create_equity_growth_comparison
)
from .metrics_radar_chart import (
    create_metrics_radar_chart,
    create_metrics_heatmap,
    calculate_composite_score,
    normalize_metric
)
from .trade_distribution_comparison import (
    create_distribution_comparison_chart,
    create_histogram_comparison,
    create_box_plot_comparison,
    create_violin_plot_comparison,
    create_win_loss_distribution_comparison,
    create_r_multiple_percentile_comparison
)
from .prediction_summary_cards import (
    create_prediction_summary_cards,
    create_empty_prediction_summary_cards,
    create_prob_win_gauge,
    create_interval_bar
)
from .probability_analysis_section import (
    create_probability_analysis_section,
    create_empty_probability_analysis_section,
    create_reliability_plot,
    create_probability_histogram,
    interpret_brier_score,
    interpret_ece
)

__all__ = [
    'create_probability_heatmap_2d',
    'create_empty_heatmap',
    'create_probability_distribution_1d',
    'create_confidence_interval_chart',
    'create_empty_distribution',
    'create_combination_card',
    'create_top_combinations_list',
    'create_combinations_summary',
    'format_combination_for_export',
    'create_trade_details_summary',
    'create_trade_list_table',
    'create_mae_mfe_mini_chart',
    'create_empty_trade_details',
    'create_transition_matrix_heatmap',
    'create_empty_transition_matrix',
    'create_streak_distribution_chart',
    'create_empty_streak_distribution',
    'create_conditional_streak_chart',
    'create_empty_conditional_streak_chart',
    'create_markov_summary_cards',
    'create_empty_summary_cards',
    'create_reliability_diagram',
    'create_empty_reliability_diagram',
    'create_calibration_histogram',
    'create_empty_calibration_histogram',
    'create_calibration_metrics_cards',
    'create_empty_calibration_metrics_cards',
    'create_calibration_table',
    'create_empty_calibration_table',
    'create_regime_winrate_chart',
    'create_empty_regime_winrate_chart',
    'create_regime_r_multiple_chart',
    'create_empty_regime_r_multiple_chart',
    'create_regime_comparison_table',
    'create_empty_regime_comparison_table',
    'create_regime_filter_controls',
    'create_regime_summary_cards',
    'create_empty_regime_summary_cards',
    'get_scenario_params_panel',
    'create_position_sizing_params',
    'create_sl_tp_params',
    'create_filter_params',
    'create_time_params',
    'create_market_condition_params',
    'create_money_management_params',
    'create_comparison_table',
    'create_comparison_summary_cards',
    'format_scenario_type',
    'calculate_percentage_changes',
    'get_color_for_change',
    'create_equity_curve_comparison_chart',
    'create_drawdown_comparison_chart',
    'create_equity_growth_comparison',
    'create_metrics_radar_chart',
    'create_metrics_heatmap',
    'calculate_composite_score',
    'normalize_metric',
    'create_distribution_comparison_chart',
    'create_histogram_comparison',
    'create_box_plot_comparison',
    'create_violin_plot_comparison',
    'create_win_loss_distribution_comparison',
    'create_r_multiple_percentile_comparison',
    'create_prediction_summary_cards',
    'create_empty_prediction_summary_cards',
    'create_prob_win_gauge',
    'create_interval_bar',
    'create_probability_analysis_section',
    'create_empty_probability_analysis_section',
    'create_reliability_plot',
    'create_probability_histogram',
    'interpret_brier_score',
    'interpret_ece'
]
