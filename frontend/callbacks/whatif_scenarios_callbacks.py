"""
What-If Scenarios Callbacks
Dash callbacks for the What-If Scenarios tab

This module implements all interactive callbacks for What-If Scenarios:
- Add scenario
- Update scenario parameters
- Compare scenarios
- Optimize scenario
- Save/load scenario presets
"""
from dash import Input, Output, State, callback, no_update, html, dcc, callback_context, ALL, MATCH
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import json
from datetime import datetime

from backend.calculators.whatif_scenario import WhatIfScenario
from frontend.components.scenario_builder_params_v2 import get_scenario_params_panel
from frontend.components.scenario_comparison_table import create_comparison_table
from frontend.components.equity_curve_comparison import create_equity_curve_comparison_chart
from frontend.components.metrics_radar_chart import create_metrics_radar_chart
from frontend.components.trade_distribution_comparison import create_distribution_comparison_chart
from frontend.components.whatif_insights_panel import (
    create_whatif_insights_panel,
    create_scenario_summary_cards,
    create_recommendations_panel
)


def register_whatif_scenarios_callbacks(app):
    """
    Register all callbacks for What-If Scenarios tab
    
    Parameters:
    -----------
    app : dash.Dash
        Dash application instance
    """
    
    @app.callback(
        Output('scenario-parameters-panel', 'children'),
        Input('scenario-type-dropdown', 'value')
    )
    def update_scenario_parameters(scenario_type):
        """
        Update parameter panel based on selected scenario type
        
        Validates: Requirements 22.1
        """
        if not scenario_type:
            return html.Div("Select a scenario type", className="text-muted")
        
        return get_scenario_params_panel(scenario_type)
    
    
    @app.callback(
        [Output('scenarios-store', 'data'),
         Output('active-scenarios-list', 'children'),
         Output('scenario-name-input', 'value')],
        [Input('add-scenario-btn', 'n_clicks')],
        [State('scenario-name-input', 'value'),
         State('scenario-type-dropdown', 'value'),
         State('scenarios-store', 'data'),
         State({'type': 'scenario-param', 'name': ALL}, 'value')]
    )
    def add_scenario(n_clicks, scenario_name, scenario_type, scenarios_data, param_values):
        """
        Add a new scenario to the scenarios list
        
        Validates: Requirements 22.1, 15.4
        """
        if not n_clicks:
            raise PreventUpdate
        
        if not scenario_type:
            raise PreventUpdate
        
        if not scenario_name or not scenario_name.strip():
            scenario_name = f"Scenario {len(scenarios_data) + 1}"
        
        # Extract parameters from pattern matching inputs
        # param_values is a list of values from all {'type': 'scenario-param'} inputs
        # We need to match them with their names from ctx
        params = {}
        
        # Get the triggered context to know which params are available
        if param_values:
            # For now, use simple default params based on scenario type
            # In production, you would map param_values to param names
            if scenario_type == 'position_sizing':
                params['risk_percent'] = param_values[0] if len(param_values) > 0 and param_values[0] is not None else 1.0
            elif scenario_type == 'sl_tp':
                params['sl_multiplier'] = param_values[0] if len(param_values) > 0 and param_values[0] is not None else 1.0
                params['tp_multiplier'] = param_values[1] if len(param_values) > 1 and param_values[1] is not None else 1.0
            elif scenario_type == 'filter':
                params['min_probability'] = param_values[0] if len(param_values) > 0 and param_values[0] is not None else 0.5
            elif scenario_type == 'time':
                params['trading_hours'] = tuple(param_values[0]) if len(param_values) > 0 and param_values[0] is not None else (0, 23)
            elif scenario_type == 'market_condition':
                params['trend_regime'] = param_values[0] if len(param_values) > 0 and param_values[0] is not None else [0, 1]
            elif scenario_type == 'money_management':
                params['compounding'] = bool(param_values[0]) if len(param_values) > 0 and param_values[0] else False
            elif scenario_type == 'ml_prediction':
                # ML prediction scenario parameters
                params['filter_by_quality'] = param_values[0] if len(param_values) > 0 and param_values[0] is not None else ['A+', 'A']
                params['filter_by_prob_min'] = param_values[1] if len(param_values) > 1 and param_values[1] is not None else 0.55
                params['filter_by_recommendation'] = bool(param_values[2]) if len(param_values) > 2 and param_values[2] else True
        else:
            # No params available, use defaults
            if scenario_type == 'position_sizing':
                params['risk_percent'] = 1.0
            elif scenario_type == 'sl_tp':
                params['sl_multiplier'] = 1.0
                params['tp_multiplier'] = 1.0
            elif scenario_type == 'filter':
                params['min_probability'] = 0.5
            elif scenario_type == 'time':
                params['trading_hours'] = (0, 23)
            elif scenario_type == 'market_condition':
                params['trend_regime'] = [0, 1]
            elif scenario_type == 'money_management':
                params['compounding'] = False
            elif scenario_type == 'ml_prediction':
                params['filter_by_quality'] = ['A+', 'A']
                params['filter_by_prob_min'] = 0.55
                params['filter_by_recommendation'] = True
        
        # Create scenario object
        scenario = {
            'id': len(scenarios_data) + 1,
            'name': scenario_name.strip(),
            'type': scenario_type,
            'params': params,
            'created_at': datetime.now().isoformat()
        }
        
        # Add to scenarios list
        scenarios_data.append(scenario)
        
        # Create active scenarios list UI
        scenarios_list = create_active_scenarios_list(scenarios_data)
        
        return scenarios_data, scenarios_list, ''
    
    
    @app.callback(
        [Output('scenarios-store', 'data', allow_duplicate=True),
         Output('active-scenarios-list', 'children', allow_duplicate=True)],
        [Input({'type': 'delete-scenario-btn', 'index': ALL}, 'n_clicks')],
        [State('scenarios-store', 'data')],
        prevent_initial_call=True
    )
    def delete_scenario(n_clicks_list, scenarios_data):
        """
        Delete a scenario from the list
        
        Validates: Requirements 22.1
        """
        if not any(n_clicks_list):
            raise PreventUpdate
        
        # Find which button was clicked
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        
        triggered_id = ctx.triggered[0]['prop_id']
        # Extract index from triggered_id
        import re
        match = re.search(r'"index":(\d+)', triggered_id)
        if not match:
            raise PreventUpdate
        
        scenario_id = int(match.group(1))
        
        # Remove scenario with matching id
        scenarios_data = [s for s in scenarios_data if s['id'] != scenario_id]
        
        # Recreate scenarios list UI
        scenarios_list = create_active_scenarios_list(scenarios_data)
        
        return scenarios_data, scenarios_list
    
    
    @app.callback(
        [Output('scenario-results-store', 'data'),
         Output('scenario-comparison-table-container', 'children'),
         Output('equity-curve-comparison-chart', 'figure'),
         Output('metrics-radar-chart', 'figure'),
         Output('trade-distribution-comparison-chart', 'figure')],
        [Input('refresh-comparison-btn', 'n_clicks'),
         Input('scenarios-store', 'data')],
        [State('merged-data-store', 'data'),
         State('distribution-type-radio', 'value'),
         State('show-drawdown-toggle', 'value'),
         State('starting-equity-store', 'data'),
         State('ml-predictions-store', 'data')]
    )
    def compare_scenarios(n_clicks, scenarios_data, merged_data, 
                         distribution_type, show_drawdown, starting_equity,
                         ml_predictions_data):
        """
        Compare multiple scenarios side-by-side
        
        Validates: Requirements 22.2, 22.3, 22.4, 22.5, 22.6, 15.4
        """
        if merged_data is None or not scenarios_data:
            # Return empty visualizations
            return {}, create_empty_comparison_message(), {}, {}, {}
        
        try:
            # Load data (support both records and split JSON)
            import io
            if isinstance(merged_data, dict) and 'data' in merged_data:
                df = pd.read_json(io.StringIO(merged_data['data']), orient='split')
            else:
                df = pd.DataFrame(merged_data)
            
            # Validate required columns
            required_cols = ['R_multiple', 'trade_success', 'net_profit']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                error_msg = f"Missing required columns: {', '.join(missing_cols)}"
                return {}, dbc.Alert(error_msg, color="danger"), {}, {}, {}
            
            # Initialize What-If engine
            whatif = WhatIfScenario(df)
            
            # Calculate baseline metrics
            baseline_metrics = whatif.baseline_metrics
            
            # Calculate metrics for each scenario
            scenario_list_for_comparison = []
            
            for scenario in scenarios_data:
                scenario_list_for_comparison.append({
                    'name': scenario['name'],
                    'type': scenario['type'],
                    'params': scenario['params']
                })
            
            # Use compare_scenarios method
            comparison_df = whatif.compare_scenarios(scenario_list_for_comparison)
            
            # Store results
            scenario_results = comparison_df.to_dict('records')
            
            # Create comparison table
            comparison_table = create_comparison_table(comparison_df)
            
            # Prepare scenarios_dict for equity curve and distribution
            scenarios_dict = {'Baseline': df.copy()}
            
            for scenario in scenarios_data:
                # Create fresh instance for each scenario
                scenario_whatif = WhatIfScenario(df.copy())
                scenario_type = scenario['type']
                params = scenario['params']
                
                # Apply scenario to get filtered/modified trades
                try:
                    if scenario_type == 'position_sizing':
                        scenario_whatif.apply_position_sizing_scenario(**params)
                    elif scenario_type == 'sl_tp':
                        scenario_whatif.apply_sl_tp_scenario(**params)
                    elif scenario_type == 'filter':
                        scenario_whatif.apply_filter_scenario(params)
                    elif scenario_type == 'time':
                        scenario_whatif.apply_time_scenario(params)
                    elif scenario_type == 'market_condition':
                        scenario_whatif.apply_market_condition_scenario(params)
                    elif scenario_type == 'money_management':
                        scenario_whatif.apply_money_management_scenario(params)
                    elif scenario_type == 'ml_prediction':
                        # Handle ML prediction scenario
                        if ml_predictions_data is None:
                            print(f"Warning: ML predictions not available for scenario {scenario['name']}")
                            scenarios_dict[scenario['name']] = df.copy()
                            continue
                        
                        # Convert ML predictions to DataFrame
                        ml_predictions_df = pd.DataFrame(ml_predictions_data)
                        
                        # Apply ML prediction scenario
                        scenario_whatif.apply_ml_prediction_scenario(
                            ml_predictions=ml_predictions_df,
                            filter_by_quality=params.get('filter_by_quality'),
                            filter_by_prob_min=params.get('filter_by_prob_min'),
                            filter_by_recommendation=params.get('filter_by_recommendation', True)
                        )
                    
                    scenarios_dict[scenario['name']] = scenario_whatif.trades.copy()
                except Exception as e:
                    print(f"Error applying scenario {scenario['name']}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    scenarios_dict[scenario['name']] = df.copy()
            
            # Create equity curve comparison
            show_dd = 'show' in show_drawdown if show_drawdown else False
            equity_value = starting_equity if starting_equity else 10000
            equity_fig = create_equity_curve_comparison_chart(
                scenarios_dict, show_dd, equity_value
            )
            
            # Create metrics radar chart
            radar_fig = create_metrics_radar_chart(comparison_df)
            
            # Create trade distribution comparison
            distribution_fig = create_distribution_comparison_chart(
                scenarios_dict, distribution_type or 'histogram', 'R_multiple'
            )
            
            return scenario_results, comparison_table, equity_fig, radar_fig, distribution_fig
            
        except Exception as e:
            error_msg = f"Error comparing scenarios: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return {}, dbc.Alert(error_msg, color="danger"), {}, {}, {}
    
    
    @app.callback(
        Output('download-comparison-csv', 'data'),
        Input('export-comparison-btn', 'n_clicks'),
        State('scenario-results-store', 'data'),
        prevent_initial_call=True
    )
    def export_comparison_csv(n_clicks, scenario_results):
        """
        Export scenario comparison to CSV
        
        Validates: Requirements 22.7
        """
        if not n_clicks or not scenario_results:
            raise PreventUpdate
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame(scenario_results)
            
            # Format numeric columns for better readability
            numeric_cols = ['total_trades', 'win_rate', 'avg_r', 'expectancy', 
                           'total_profit', 'max_drawdown', 'profit_factor', 'sharpe_ratio']
            
            for col in numeric_cols:
                if col in df.columns:
                    if col in ['win_rate', 'max_drawdown']:
                        df[col] = df[col].round(2)
                    elif col in ['avg_r', 'sharpe_ratio']:
                        df[col] = df[col].round(3)
                    elif col == 'total_profit':
                        df[col] = df[col].round(2)
                    elif col == 'expectancy':
                        df[col] = df[col].round(2)
                    elif col == 'profit_factor':
                        df[col] = df[col].round(2)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'scenario_comparison_{timestamp}.csv'
            
            return dcc.send_data_frame(df.to_csv, filename, index=False)
            
        except Exception as e:
            print(f"Error exporting CSV: {str(e)}")
            raise PreventUpdate
    
    
    @app.callback(
        [Output('saved-scenario-sets-store', 'data'),
         Output('load-scenario-set-dropdown', 'options')],
        [Input('save-scenario-set-btn', 'n_clicks')],
        [State('scenario-set-name-input', 'value'),
         State('scenarios-store', 'data'),
         State('saved-scenario-sets-store', 'data')]
    )
    def save_scenario_set(n_clicks, set_name, scenarios_data, saved_sets):
        """
        Save current scenario set with a name
        
        Validates: Requirements 22.8
        """
        if not n_clicks or not set_name or not set_name.strip():
            raise PreventUpdate
        
        if saved_sets is None:
            saved_sets = {}
        
        # Save scenario set
        saved_sets[set_name.strip()] = {
            'scenarios': scenarios_data,
            'saved_at': datetime.now().isoformat()
        }
        
        # Update dropdown options
        options = [{'label': name, 'value': name} for name in saved_sets.keys()]
        
        return saved_sets, options
    
    
    @app.callback(
        [Output('scenarios-store', 'data', allow_duplicate=True),
         Output('active-scenarios-list', 'children', allow_duplicate=True)],
        [Input('load-scenario-set-btn', 'n_clicks')],
        [State('load-scenario-set-dropdown', 'value'),
         State('saved-scenario-sets-store', 'data')],
        prevent_initial_call=True
    )
    def load_scenario_set(n_clicks, set_name, saved_sets):
        """
        Load a saved scenario set
        
        Validates: Requirements 22.8
        """
        if not n_clicks or not set_name or not saved_sets:
            raise PreventUpdate
        
        if set_name not in saved_sets:
            raise PreventUpdate
        
        # Load scenarios
        scenarios_data = saved_sets[set_name]['scenarios']
        
        # Recreate scenarios list UI
        scenarios_list = create_active_scenarios_list(scenarios_data)
        
        return scenarios_data, scenarios_list
    
    
    @app.callback(
        Output('download-scenarios-json', 'data'),
        Input('export-scenarios-btn', 'n_clicks'),
        State('scenarios-store', 'data'),
        prevent_initial_call=True
    )
    def export_scenarios_json(n_clicks, scenarios_data):
        """
        Export scenarios to JSON file
        
        Validates: Requirements 22.8
        """
        if not n_clicks or not scenarios_data:
            raise PreventUpdate
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'scenarios_{timestamp}.json'
        
        # Convert to JSON string
        json_str = json.dumps(scenarios_data, indent=2)
        
        return dict(content=json_str, filename=filename)
    
    
    @app.callback(
        [Output('scenarios-store', 'data', allow_duplicate=True),
         Output('active-scenarios-list', 'children', allow_duplicate=True)],
        [Input('clear-all-scenarios-btn', 'n_clicks')],
        prevent_initial_call=True
    )
    def clear_all_scenarios(n_clicks):
        """
        Clear all scenarios
        
        Validates: Requirements 22.1
        """
        if not n_clicks:
            raise PreventUpdate
        
        # Return empty scenarios list
        empty_list = dbc.Alert(
            "No scenarios added yet. Create your first scenario above.",
            color="light",
            className="mb-0 small"
        )
        
        return [], empty_list
    
    
    @app.callback(
        [Output('whatif-insights-panel', 'children'),
         Output('whatif-summary-cards', 'children'),
         Output('whatif-recommendations-panel', 'children')],
        [Input('scenario-results-store', 'data')],
        [State('scenarios-store', 'data')]
    )
    def update_insights_and_recommendations(scenario_results, scenarios_data):
        """
        Update insights panel and recommendations based on comparison results
        """
        if not scenario_results or not scenarios_data:
            empty_msg = dbc.Alert([
                html.I(className="bi bi-info-circle me-2"),
                "Add scenarios and click Refresh to see insights"
            ], color="light", className="mb-0")
            return empty_msg, html.Div(), html.Div()
        
        try:
            # Convert results to DataFrame
            comparison_df = pd.DataFrame(scenario_results)
            
            # Generate insights
            insights_panel = create_whatif_insights_panel(comparison_df, scenarios_data)
            
            # Generate summary cards
            summary_cards = create_scenario_summary_cards(comparison_df)
            
            # Generate recommendations
            recommendations = create_recommendations_panel(comparison_df, scenarios_data)
            
            return insights_panel, summary_cards, recommendations
            
        except Exception as e:
            error_msg = f"Error generating insights: {str(e)}"
            print(error_msg)
            return dbc.Alert(error_msg, color="warning"), html.Div(), html.Div()
    
    
    @app.callback(
        Output('whatif-help-modal', 'is_open'),
        [Input('open-whatif-help-btn', 'n_clicks'),
         Input('close-whatif-help-modal', 'n_clicks')],
        [State('whatif-help-modal', 'is_open')]
    )
    def toggle_help_modal(open_clicks, close_clicks, is_open):
        """Toggle help modal visibility"""
        if open_clicks or close_clicks:
            return not is_open
        return is_open
    
    
    @app.callback(
        Output('starting-equity-store', 'data'),
        Input('starting-equity-input', 'value')
    )
    def update_starting_equity(equity_value):
        """Update starting equity store when user changes the value"""
        if equity_value is None or equity_value < 100:
            return 10000  # Default
        return equity_value


def create_active_scenarios_list(scenarios_data):
    """
    Create UI list of active scenarios
    
    Args:
        scenarios_data: List of scenario dictionaries
    
    Returns:
        Dash component with scenario cards
    """
    if not scenarios_data:
        return dbc.Alert(
            "No scenarios added yet. Create your first scenario above.",
            color="light",
            className="mb-0 small"
        )
    
    scenario_cards = []
    
    for scenario in scenarios_data:
        # Get scenario type icon
        type_icons = {
            'position_sizing': '💰',
            'sl_tp': '🎯',
            'filter': '🔍',
            'time': '⏰',
            'market_condition': '📊',
            'money_management': '💵',
            'ml_prediction': '🤖'
        }
        icon = type_icons.get(scenario['type'], '📋')
        
        # Create parameter summary
        params_summary = []
        for key, value in scenario['params'].items():
            if isinstance(value, (int, float)):
                params_summary.append(f"{key}: {value:.2f}")
            elif isinstance(value, list):
                params_summary.append(f"{key}: {len(value)} items")
            else:
                params_summary.append(f"{key}: {value}")
        
        params_text = ", ".join(params_summary[:3])  # Show first 3 params
        if len(params_summary) > 3:
            params_text += "..."
        
        card = dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.H6([
                        icon, " ", scenario['name']
                    ], className="mb-1"),
                    html.Small(
                        scenario['type'].replace('_', ' ').title(),
                        className="text-muted d-block mb-2"
                    ),
                    html.Small(
                        params_text,
                        className="text-muted d-block mb-2",
                        style={'fontSize': '0.75rem'}
                    ),
                    dbc.ButtonGroup([
                        dbc.Button(
                            [html.I(className="bi bi-trash")],
                            id={'type': 'delete-scenario-btn', 'index': scenario['id']},
                            size="sm",
                            color="danger",
                            outline=True
                        )
                    ], size="sm")
                ])
            ], className="p-2")
        ], className="mb-2")
        
        scenario_cards.append(card)
    
    return html.Div(scenario_cards)


def create_empty_comparison_message():
    """Create empty state message for comparison table"""
    return dbc.Alert([
        html.I(className="bi bi-info-circle me-2"),
        "Add scenarios and click Refresh to see comparison"
    ], color="info", className="mb-0")



    @app.callback(
        Output('mae-mfe-optimizer-container', 'children'),
        [Input('tabs', 'active_tab'),
         Input('merged-data-store', 'data')]
    )
    def render_mae_mfe_optimizer(active_tab, merged_data):
        """
        Render MAE/MFE optimizer when What-If Scenarios tab is active
        
        Only renders if:
        - Tab is active (tab-whatif-scenarios)
        - Data is available
        
        Validates: Requirements 2.2, 6.1, 6.2
        """
        # Only render when What-If Scenarios tab is active
        if active_tab != 'tab-whatif-scenarios':
            raise PreventUpdate
        
        # Check if data is available
        if merged_data is None:
            return dbc.Alert([
                html.H5("Data Tidak Tersedia", className="alert-heading"),
                html.P("Silakan muat data trading terlebih dahulu untuk menggunakan MAE/MFE Optimizer.")
            ], color="info")
        
        # Import and render MAE/MFE optimizer
        from frontend.components.mae_mfe_optimizer import create_mae_mfe_optimizer
        return create_mae_mfe_optimizer()
    
    
    @app.callback(
        Output('monte-carlo-viz-container', 'children'),
        [Input('tabs', 'active_tab'),
         Input('merged-data-store', 'data')]
    )
    def render_monte_carlo_viz(active_tab, merged_data):
        """
        Render Monte Carlo visualization when What-If Scenarios tab is active
        
        Only renders if:
        - Tab is active (tab-whatif-scenarios)
        - Data is available
        
        Validates: Requirements 3.2, 6.1, 6.2
        """
        # Only render when What-If Scenarios tab is active
        if active_tab != 'tab-whatif-scenarios':
            raise PreventUpdate
        
        # Check if data is available
        if merged_data is None:
            return dbc.Alert([
                html.H5("Data Tidak Tersedia", className="alert-heading"),
                html.P("Silakan muat data trading terlebih dahulu untuk menggunakan Monte Carlo Simulation.")
            ], color="info")
        
        # Import and render Monte Carlo visualization
        from frontend.components.monte_carlo_viz import create_monte_carlo_viz
        return create_monte_carlo_viz()
