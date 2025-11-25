"""
What-If Scenarios Frontend Demo

This script demonstrates the What-If Scenarios frontend layout and components
for interactive scenario comparison and analysis.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from dash import Dash, html, dcc, Input, Output, State, callback_context
import dash_bootstrap_components as dbc

# Import backend
from backend.calculators.whatif_scenario import WhatIfScenario

# Import frontend components
from frontend.layouts.whatif_scenarios_layout import create_whatif_scenarios_layout
from frontend.components.scenario_builder_params import get_scenario_params_panel
from frontend.components.scenario_comparison_table import (
    create_comparison_table,
    create_comparison_summary_cards
)
from frontend.components.equity_curve_comparison import (
    create_equity_curve_comparison_chart,
    create_equity_growth_comparison
)
from frontend.components.metrics_radar_chart import create_metrics_radar_chart
from frontend.components.trade_distribution_comparison import (
    create_distribution_comparison_chart
)


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
        'minutes_to_next_high_impact_news': np.random.randint(-120, 120, n_trades),
        'holding_minutes': np.random.randint(30, 480, n_trades)
    })
    
    # Ensure trade_success matches R_multiple sign
    trades.loc[trades['R_multiple'] > 0, 'trade_success'] = 1
    trades.loc[trades['R_multiple'] <= 0, 'trade_success'] = 0
    
    return trades


def main():
    """Run the What-If Scenarios frontend demo"""
    
    # Initialize Dash app
    app = Dash(
        __name__,
        external_stylesheets=[
            dbc.themes.BOOTSTRAP,
            'https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css'
        ],
        suppress_callback_exceptions=True
    )
    
    # Generate sample data
    trades_df = generate_sample_trades()
    scenario_engine = WhatIfScenario(trades_df)
    
    # Create layout
    app.layout = create_whatif_scenarios_layout()
    
    # Callback to update parameter panel based on scenario type
    @app.callback(
        Output('scenario-parameters-panel', 'children'),
        Input('scenario-type-dropdown', 'value')
    )
    def update_parameter_panel(scenario_type):
        return get_scenario_params_panel(scenario_type)
    
    # Callback to add scenario
    @app.callback(
        Output('scenarios-store', 'data'),
        Output('active-scenarios-list', 'children'),
        Input('add-scenario-btn', 'n_clicks'),
        State('scenario-name-input', 'value'),
        State('scenario-type-dropdown', 'value'),
        State('scenarios-store', 'data'),
        prevent_initial_call=True
    )
    def add_scenario(n_clicks, name, scenario_type, current_scenarios):
        if not name:
            name = f"Scenario {len(current_scenarios) + 1}"
        
        # Create new scenario (simplified for demo)
        new_scenario = {
            'name': name,
            'type': scenario_type,
            'params': {}  # Would collect from parameter inputs
        }
        
        current_scenarios.append(new_scenario)
        
        # Create scenario list items
        scenario_items = []
        for idx, scenario in enumerate(current_scenarios):
            item = dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Strong(scenario['name']),
                        dbc.ButtonGroup([
                            dbc.Button(
                                html.I(className="bi bi-pencil"),
                                size="sm",
                                color="link",
                                id={'type': 'edit-scenario', 'index': idx}
                            ),
                            dbc.Button(
                                html.I(className="bi bi-trash"),
                                size="sm",
                                color="link",
                                id={'type': 'delete-scenario', 'index': idx}
                            )
                        ], size="sm", className="float-end")
                    ]),
                    html.Small(scenario['type'].replace('_', ' ').title(), className="text-muted")
                ], className="py-2")
            ], className="mb-2")
            scenario_items.append(item)
        
        return current_scenarios, scenario_items
    
    # Callback to update comparison visualizations
    @app.callback(
        Output('scenario-comparison-table-container', 'children'),
        Output('equity-curve-comparison-chart', 'figure'),
        Output('metrics-radar-chart', 'figure'),
        Output('trade-distribution-comparison-chart', 'figure'),
        Input('scenarios-store', 'data'),
        Input('refresh-comparison-btn', 'n_clicks'),
        Input('distribution-type-radio', 'value'),
        prevent_initial_call=True
    )
    def update_comparisons(scenarios, n_clicks, dist_type):
        # Create sample comparison data
        scenario_list = [
            {'name': 'Conservative', 'type': 'position_sizing', 'params': {'risk_percent': 0.5}},
            {'name': 'Moderate', 'type': 'position_sizing', 'params': {'risk_percent': 1.5}},
            {'name': 'Aggressive', 'type': 'position_sizing', 'params': {'risk_percent': 2.5}}
        ]
        
        comparison_df = scenario_engine.compare_scenarios(scenario_list)
        
        # Create comparison table
        table = create_comparison_table(comparison_df)
        
        # Create scenarios data for charts
        scenarios_data = {
            'Baseline': trades_df,
            'Conservative': trades_df.copy(),
            'Moderate': trades_df.copy(),
            'Aggressive': trades_df.copy()
        }
        
        # Create charts
        equity_fig = create_equity_curve_comparison_chart(scenarios_data, show_drawdown=True)
        radar_fig = create_metrics_radar_chart(comparison_df)
        dist_fig = create_distribution_comparison_chart(scenarios_data, dist_type, 'R_multiple')
        
        return table, equity_fig, radar_fig, dist_fig
    
    print("\n" + "=" * 80)
    print("WHAT-IF SCENARIOS FRONTEND DEMO")
    print("=" * 80)
    print("\nStarting Dash application...")
    print("Open your browser and navigate to: http://127.0.0.1:8050")
    print("\nFeatures:")
    print("  - Interactive scenario builder with dynamic parameters")
    print("  - Side-by-side scenario comparison table")
    print("  - Equity curve comparison with drawdown visualization")
    print("  - Performance metrics radar chart")
    print("  - Trade distribution comparison (histogram/box/violin)")
    print("  - Scenario save/load functionality")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 80 + "\n")
    
    # Run the app
    app.run_server(debug=True, port=8050)


if __name__ == '__main__':
    main()
