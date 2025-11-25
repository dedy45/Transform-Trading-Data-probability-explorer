"""
Trade Analysis Dashboard Callbacks
Implements all interactive callbacks for the Trade Analysis Dashboard

Requirements: 0.1, 0.10, 0.11
"""
import base64
import io
import json
import pandas as pd
import numpy as np
from dash import callback, Input, Output, State, html, dash_table, no_update, dcc
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backend.calculators.trade_analysis_dashboard import (
    calculate_summary_metrics,
    calculate_equity_curve,
    calculate_r_distribution,
    calculate_mae_mfe_analysis,
    calculate_time_based_performance,
    calculate_trade_type_analysis,
    calculate_consecutive_analysis,
    calculate_risk_metrics
)

# Import server-side cache
from backend.utils.data_cache import (
    store_trade_data,
    store_merged_data,
    get_trade_data,
    get_merged_data,
    has_cached_data,
    clear_all_data,
    get_cache_info
)


def load_data_from_store(data_store):
    """
    Helper function to load data from store (supports both local and global formats)
    
    Returns:
        pd.DataFrame or None
    """
    # Try server cache first (most reliable)
    trades_df = get_trade_data()
    if trades_df is not None:
        return trades_df
    
    # Fallback to browser store
    if data_store and 'data' in data_store:
        try:
            return pd.read_json(io.StringIO(data_store['data']), orient='split')
        except Exception as e:
            print(f"Error loading from browser store: {e}")
            return None
    
    return None


def has_data_in_store(data_store):
    """
    Helper function to check if data exists in store
    
    Returns:
        bool
    """
    if not data_store:
        return False
    
    # Check local format
    if data_store.get('loaded'):
        return True
    
    # Check global format
    if 'data' in data_store:
        return True
    
    # Check server cache
    if has_cached_data():
        return True
    
    return False


def register_trade_analysis_callbacks(app):
    """
    Register all Trade Analysis Dashboard callbacks
    
    NOTE: Data loading is now handled by global data loader in app.py
    This module only handles visualization updates and interactions.
    
    Callbacks:
    1. Update all visualizations (from global data stores)
    2. Trade table row click (show details)
    3. Chart click (jump to trade)
    4. Export report
    5. Navigate to next page
    """
    
    # Callback removed: populate_sample_dropdown
    # Now handled by global data loader in app.py
    
    # Callback removed: load_csv_file
    # Now handled by global data loader in app.py
    
    # Callback 1: Update all visualizations
    @app.callback(
        [
            Output('summary-total-trades', 'children'),
            Output('summary-win-rate', 'children'),
            Output('summary-avg-r', 'children'),
            Output('summary-expectancy', 'children'),
            Output('summary-max-dd', 'children'),
            Output('summary-profit-factor', 'children'),
            Output('equity-curve-chart', 'figure'),
            Output('r-distribution-chart', 'figure'),
            Output('r-statistics-table', 'children'),
            Output('mae-mfe-scatter', 'figure'),
            Output('winners-stats-table', 'children'),
            Output('losers-stats-table', 'children'),
            Output('trade-count-badge', 'children'),
            # Expectancy outputs (INTEGRATED)
            Output('expectancy-results-store', 'data'),
            Output('expectancy-summary-expectancy-r', 'children'),
            Output('expectancy-summary-expectancy-status', 'children'),
            Output('expectancy-summary-win-rate', 'children'),
            Output('expectancy-summary-avg-win', 'children'),
            Output('expectancy-summary-avg-loss', 'children'),
        ],
        [
            Input('trade-data-loaded-store', 'data'),
            Input('trade-data-store', 'data'),  # Listen to global store too
            Input('merged-data-store', 'data')  # Listen to merged store too
        ],
        prevent_initial_call=False
    )
    def update_all_visualizations(local_data_store, global_trade_store, global_merged_store):
        """
        Update all visualizations when data is loaded
        
        Requirements: 0.2, 0.3, 0.4, 0.5
        """
        print(f"\n=== Update Visualizations Triggered ===")
        print(f"Local data store: {local_data_store}")
        print(f"Global trade store: {global_trade_store}")
        print(f"Global merged store: {global_merged_store}")
        
        # Use whichever store has data (priority: local > global_merged > global_trade)
        data_store = None
        if local_data_store and local_data_store.get('loaded'):
            data_store = local_data_store
            print("[INFO] Using local data store")
        elif global_merged_store and 'data' in global_merged_store:
            data_store = global_merged_store
            print("[INFO] Using global merged store")
        elif global_trade_store and 'data' in global_trade_store:
            data_store = global_trade_store
            print("[INFO] Using global trade store")
        
        print(f"Selected data store: {data_store}")
        
        # Return empty/default values if no data
        # Check if data_store has either 'loaded' flag or 'data' key
        has_data = False
        if data_store:
            if data_store.get('loaded'):  # Local store format
                has_data = True
            elif 'data' in data_store:  # Global store format
                has_data = True
        
        if not has_data:
            print("No data loaded, returning defaults")
            empty_fig = go.Figure()
            empty_fig.update_layout(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                annotations=[{
                    'text': 'Upload data to see visualization',
                    'xref': 'paper',
                    'yref': 'paper',
                    'x': 0.5,
                    'y': 0.5,
                    'showarrow': False,
                    'font': {'size': 16, 'color': 'gray'}
                }]
            )
            return (
                "0", "0%", "0.00R", "$0", "0%", "0.00",
                empty_fig, empty_fig, html.Div("No data"),
                empty_fig, html.Div("No data"), html.Div("No data"),
                "0 trades",
                # Expectancy outputs (empty state)
                {},
                "--",
                "",
                "--",
                "--",
                "--"
            )
        
        try:
            # Try to load from server cache first (more reliable)
            print("Trying to load from server cache...")
            trades_df = get_trade_data()
            
            if trades_df is None:
                # Fallback to browser store
                print("Server cache empty, loading from browser store...")
                trades_df = pd.read_json(io.StringIO(data_store['data']), orient='split')
            
            print(f"Data loaded: {len(trades_df)} rows")
            
            # Calculate summary metrics
            print("Calculating summary metrics...")
            summary = calculate_summary_metrics(trades_df)
            print(f"Summary calculated: {summary}")
        except Exception as e:
            print(f"ERROR in update_all_visualizations: {e}")
            import traceback
            traceback.print_exc()
            # Return error state
            error_fig = go.Figure()
            error_fig.update_layout(
                annotations=[{
                    'text': f'Error: {str(e)}',
                    'xref': 'paper',
                    'yref': 'paper',
                    'x': 0.5,
                    'y': 0.5,
                    'showarrow': False,
                    'font': {'size': 14, 'color': 'red'}
                }]
            )
            return (
                "Error", "Error", "Error", "Error", "Error", "Error",
                error_fig, error_fig, html.Div(f"Error: {str(e)}"),
                error_fig, html.Div(f"Error: {str(e)}"), html.Div(f"Error: {str(e)}"),
                "Error",
                # Expectancy outputs (error state)
                {},
                "--",
                html.Span(f"Error: {str(e)}", className="text-danger"),
                "--",
                "--",
                "--"
            )
        
        # Format summary values
        total_trades = f"{summary['total_trades']:,}"
        win_rate = f"{summary['win_rate']*100:.1f}%"
        avg_r = f"{summary['avg_r']:.2f}R"
        expectancy = f"${summary['expectancy']:.2f}"
        max_dd = f"{summary['max_drawdown']:.1f}%"
        profit_factor = f"{summary['profit_factor']:.2f}"
        
        # Create equity curve
        equity_data = calculate_equity_curve(trades_df)
        equity_fig = create_equity_curve_figure(equity_data)
        
        # Create R-distribution
        r_dist_data = calculate_r_distribution(trades_df)
        r_dist_fig = create_r_distribution_figure(r_dist_data)
        r_stats_table = create_r_statistics_table(r_dist_data)
        
        # Create MAE/MFE scatter
        mae_mfe_data = calculate_mae_mfe_analysis(trades_df)
        mae_mfe_fig = create_mae_mfe_figure(mae_mfe_data)
        winners_table = create_stats_table(mae_mfe_data['winners_stats'], "Winners")
        losers_table = create_stats_table(mae_mfe_data['losers_stats'], "Losers")
        
        # Trade count badge
        trade_badge = f"{len(trades_df):,} trades"
        
        # Calculate Expectancy metrics (INTEGRATED)
        from backend.calculators.expectancy_calculator import (
            compute_expectancy_R,
            compute_expectancy_by_group,
            compute_r_percentiles
        )
        try:
            expectancy_result = compute_expectancy_R(trades_df)
            
            # Format expectancy values
            exp_r = f"{expectancy_result['expectancy_R']:.3f}R"
            exp_win_rate = f"{expectancy_result['win_rate']*100:.1f}%"
            exp_avg_win = f"{expectancy_result['avg_win_R']:.3f}R" if not pd.isna(expectancy_result['avg_win_R']) else "--"
            exp_avg_loss = f"{expectancy_result['avg_loss_R']:.3f}R" if not pd.isna(expectancy_result['avg_loss_R']) else "--"
            
            # Status indicator
            if expectancy_result['expectancy_R'] > 0:
                exp_status = html.Span("Positive expectancy", className="text-success")
            elif expectancy_result['expectancy_R'] < 0:
                exp_status = html.Span("Negative expectancy - Review strategy", className="text-danger")
            else:
                exp_status = html.Span("Break-even", className="text-warning")
            
            # Calculate grouped expectancy for heatmap
            grouped_expectancy = None
            try:
                # Try to group by session if available
                if 'session' in trades_df.columns:
                    grouped_df = compute_expectancy_by_group(trades_df, 'session')
                    grouped_expectancy = grouped_df.to_json(orient='split')
                elif 'Type' in trades_df.columns:
                    # Fallback to trade type
                    grouped_df = compute_expectancy_by_group(trades_df, 'Type')
                    grouped_expectancy = grouped_df.to_json(orient='split')
            except Exception as e:
                print(f"[WARNING] Could not compute grouped expectancy: {e}")
            
            # Calculate percentiles for histogram
            percentiles = None
            try:
                percentiles = compute_r_percentiles(trades_df)
            except Exception as e:
                print(f"[WARNING] Could not compute percentiles: {e}")
            
            # Store results (include all data for visualizations)
            expectancy_store = {
                'global_expectancy': expectancy_result,
                'grouped_expectancy': grouped_expectancy,
                'percentiles': percentiles,
                'timestamp': pd.Timestamp.now().isoformat()
            }
        except Exception as e:
            print(f"Error calculating expectancy: {e}")
            exp_r = "--"
            exp_status = ""
            exp_win_rate = "--"
            exp_avg_win = "--"
            exp_avg_loss = "--"
            expectancy_store = {}
        
        return (
            total_trades,
            win_rate,
            avg_r,
            expectancy,
            max_dd,
            profit_factor,
            equity_fig,
            r_dist_fig,
            r_stats_table,
            mae_mfe_fig,
            winners_table,
            losers_table,
            trade_badge,
            # Expectancy outputs (INTEGRATED)
            expectancy_store,
            exp_r,
            exp_status,
            exp_win_rate,
            exp_avg_win,
            exp_avg_loss
        )

    
    # Callback 3: Update time-based performance
    @app.callback(
        Output('time-based-content', 'children'),
        [
            Input('time-based-tabs', 'active_tab'),
            Input('trade-data-loaded-store', 'data'),
            Input('trade-data-store', 'data'),
            Input('merged-data-store', 'data')
        ]
    )
    def update_time_based_performance(active_tab, local_store, global_trade_store, global_merged_store):
        # Use whichever store has data
        data_store = None
        if local_store and (local_store.get('loaded') or 'data' in local_store):
            data_store = local_store
        elif global_merged_store and 'data' in global_merged_store:
            data_store = global_merged_store
        elif global_trade_store and 'data' in global_trade_store:
            data_store = global_trade_store
        """
        Update time-based performance charts
        
        Requirements: 0.6
        """
        # Check if data available (support both local and global store formats)
        has_data = False
        if data_store:
            if data_store.get('loaded') or 'data' in data_store:
                has_data = True
        
        if not has_data:
            return html.Div([
                dbc.Alert("No data loaded. Please load data first.", color="info", className="text-center")
            ])
        
        try:
            # Load data using helper function
            trades_df = load_data_from_store(data_store)
            if trades_df is None:
                return html.Div([
                    dbc.Alert("No data available", color="warning", className="text-center")
                ])
            
            time_data = calculate_time_based_performance(trades_df)
            
            if active_tab == 'hourly-tab':
                return create_hourly_performance_chart(time_data['hourly'])
            elif active_tab == 'daily-tab':
                return create_daily_performance_chart(time_data['daily'])
            elif active_tab == 'weekly-tab':
                return create_weekly_performance_chart(time_data['weekly'])
            elif active_tab == 'monthly-tab':
                return create_monthly_performance_chart(time_data['monthly'])
            elif active_tab == 'session-tab':
                return create_session_performance_chart(time_data['session'])
            
            return html.Div("No data available")
        except Exception as e:
            print(f"Error in update_time_based_performance: {e}")
            import traceback
            traceback.print_exc()
            return html.Div([
                dbc.Alert(f"Error: {str(e)}", color="danger")
            ])
    
    
    # Callback 4: Update trade type analysis
    @app.callback(
        [
            Output('direction-analysis-chart', 'figure'),
            Output('exit-reason-chart', 'figure'),
        ],
        [
            Input('trade-data-loaded-store', 'data'),
            Input('trade-data-store', 'data'),
            Input('merged-data-store', 'data')
        ]
    )
    def update_trade_type_analysis(local_store, global_trade_store, global_merged_store):
        # Use whichever store has data
        data_store = None
        if local_store and (local_store.get('loaded') or 'data' in local_store):
            data_store = local_store
        elif global_merged_store and 'data' in global_merged_store:
            data_store = global_merged_store
        elif global_trade_store and 'data' in global_trade_store:
            data_store = global_trade_store
        """
        Update trade type analysis charts
        
        Requirements: 0.7
        """
        if not has_data_in_store(data_store):
            raise PreventUpdate
        
        trades_df = load_data_from_store(data_store)
        if trades_df is None:
            raise PreventUpdate
        
        trade_type_data = calculate_trade_type_analysis(trades_df)
        
        direction_fig = create_direction_analysis_figure(trade_type_data['by_direction'])
        exit_reason_fig = create_exit_reason_figure(trade_type_data['by_exit_reason'])
        
        return direction_fig, exit_reason_fig
    
    
    # Callback 5: Update consecutive analysis
    @app.callback(
        [
            Output('streak-timeline-chart', 'figure'),
            Output('cumulative-perf-chart', 'figure'),
        ],
        [
            Input('trade-data-loaded-store', 'data'),
            Input('trade-data-store', 'data'),
            Input('merged-data-store', 'data')
        ]
    )
    def update_consecutive_analysis(local_store, global_trade_store, global_merged_store):
        # Use whichever store has data
        data_store = None
        if local_store and (local_store.get('loaded') or 'data' in local_store):
            data_store = local_store
        elif global_merged_store and 'data' in global_merged_store:
            data_store = global_merged_store
        elif global_trade_store and 'data' in global_trade_store:
            data_store = global_trade_store
        """
        Update consecutive trades analysis
        
        Requirements: 0.8
        """
        if not has_data_in_store(data_store):
            raise PreventUpdate
        
        trades_df = load_data_from_store(data_store)
        if trades_df is None:
            raise PreventUpdate
        consecutive_data = calculate_consecutive_analysis(trades_df)
        
        streak_fig = create_streak_timeline_figure(consecutive_data)
        cumulative_fig = create_cumulative_performance_figure(consecutive_data)
        
        return streak_fig, cumulative_fig
    
    
    # Callback 6: Update risk metrics
    @app.callback(
        Output('risk-metrics-table', 'children'),
        [
            Input('trade-data-loaded-store', 'data'),
            Input('trade-data-store', 'data'),
            Input('merged-data-store', 'data')
        ]
    )
    def update_risk_metrics(local_store, global_trade_store, global_merged_store):
        # Use whichever store has data
        data_store = None
        if local_store and (local_store.get('loaded') or 'data' in local_store):
            data_store = local_store
        elif global_merged_store and 'data' in global_merged_store:
            data_store = global_merged_store
        elif global_trade_store and 'data' in global_trade_store:
            data_store = global_trade_store
        """
        Update risk metrics table
        
        Requirements: 0.9
        """
        if not has_data_in_store(data_store):
            raise PreventUpdate
        
        trades_df = load_data_from_store(data_store)
        if trades_df is None:
            raise PreventUpdate
        risk_metrics = calculate_risk_metrics(trades_df)
        
        return create_risk_metrics_table(risk_metrics)

    
    # Callback 7: Update trade table
    @app.callback(
        Output('trade-table-container', 'children'),
        [
            Input('trade-data-loaded-store', 'data'),
            Input('trade-data-store', 'data'),
            Input('merged-data-store', 'data')
        ]
    )
    def update_trade_table(local_store, global_trade_store, global_merged_store):
        # Use whichever store has data
        data_store = None
        if local_store and (local_store.get('loaded') or 'data' in local_store):
            data_store = local_store
        elif global_merged_store and 'data' in global_merged_store:
            data_store = global_merged_store
        elif global_trade_store and 'data' in global_trade_store:
            data_store = global_trade_store
        """
        Update trade history table
        
        Requirements: 0.10
        """
        if not has_data_in_store(data_store):
            raise PreventUpdate
        
        trades_df = load_data_from_store(data_store)
        if trades_df is None:
            raise PreventUpdate
        
        # Select columns to display
        display_cols = [
            'Ticket_id', 'Timestamp', 'Type', 'Symbol', 'OpenPrice', 'ClosePrice',
            'R_multiple', 'net_profit', 'trade_success', 'ExitReason'
        ]
        
        # Filter to available columns
        available_cols = [col for col in display_cols if col in trades_df.columns]
        display_df = trades_df[available_cols].copy()
        
        # Format columns
        if 'Timestamp' in display_df.columns:
            try:
                display_df['Timestamp'] = pd.to_datetime(display_df['Timestamp'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M')
            except:
                pass  # Keep original format if conversion fails
        if 'R_multiple' in display_df.columns:
            display_df['R_multiple'] = pd.to_numeric(display_df['R_multiple'], errors='coerce').round(2)
        if 'net_profit' in display_df.columns:
            display_df['net_profit'] = pd.to_numeric(display_df['net_profit'], errors='coerce').round(2)
        
        # Create DataTable
        table = dash_table.DataTable(
            id='trade-history-table',
            columns=[{"name": col, "id": col} for col in display_df.columns],
            data=display_df.to_dict('records'),
            page_size=20,
            page_action='native',
            sort_action='native',
            sort_mode='multi',
            filter_action='native',
            row_selectable='single',
            selected_rows=[],
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'left',
                'padding': '10px',
                'fontSize': '14px'
            },
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'filter_query': '{trade_success} = 1'},
                    'backgroundColor': 'rgba(0, 255, 0, 0.1)'
                },
                {
                    'if': {'filter_query': '{trade_success} = 0'},
                    'backgroundColor': 'rgba(255, 0, 0, 0.1)'
                }
            ]
        )
        
        return table
    
    
    # Callback 8: Trade table row click (show details)
    @app.callback(
        Output('selected-trade-store', 'data'),
        [Input('trade-history-table', 'selected_rows')],
        [State('trade-data-loaded-store', 'data')]
    )
    def handle_trade_row_click(selected_rows, data_store):
        """
        Handle trade table row click to show details
        
        Requirements: 0.10
        """
        if not selected_rows or not has_data_in_store(data_store):
            raise PreventUpdate
        
        trades_df = load_data_from_store(data_store)
        if trades_df is None:
            raise PreventUpdate
        selected_trade = trades_df.iloc[selected_rows[0]].to_dict()
        
        return {'trade': selected_trade, 'index': selected_rows[0]}

    
    # Callback 9: Chart click (jump to trade)
    @app.callback(
        Output('trade-history-table', 'selected_rows'),
        [
            Input('equity-curve-chart', 'clickData'),
            Input('r-distribution-chart', 'clickData'),
            Input('mae-mfe-scatter', 'clickData'),
        ],
        [State('trade-data-loaded-store', 'data')]
    )
    def handle_chart_click(equity_click, r_click, mae_click, data_store):
        """
        Handle chart clicks to jump to corresponding trade in table
        
        Requirements: 0.10
        """
        from dash import callback_context
        
        if not callback_context.triggered or not has_data_in_store(data_store):
            raise PreventUpdate
        
        trigger_id = callback_context.triggered[0]['prop_id'].split('.')[0]
        
        # Get the clicked point index
        if trigger_id == 'equity-curve-chart' and equity_click:
            point_index = equity_click['points'][0]['pointIndex']
            return [point_index]
        elif trigger_id == 'r-distribution-chart' and r_click:
            # For histogram, we can't directly map to a trade
            raise PreventUpdate
        elif trigger_id == 'mae-mfe-scatter' and mae_click:
            point_index = mae_click['points'][0]['pointIndex']
            return [point_index]
        
        raise PreventUpdate
    
    
    # Callback 10: Export report
    @app.callback(
        Output('download-report', 'data'),
        [Input('export-pdf-btn', 'n_clicks')],
        [State('trade-data-loaded-store', 'data')],
        prevent_initial_call=True
    )
    def export_report(n_clicks, data_store):
        """
        Export comprehensive report as PDF
        
        Requirements: 0.11
        """
        if not n_clicks or not has_data_in_store(data_store):
            raise PreventUpdate
        
        # For now, export as CSV (PDF generation requires additional libraries)
        trades_df = load_data_from_store(data_store)
        if trades_df is None:
            raise PreventUpdate
        
        # Calculate all metrics
        summary = calculate_summary_metrics(trades_df)
        risk_metrics = calculate_risk_metrics(trades_df)
        
        # Create report DataFrame
        report_data = {
            'Metric': list(summary.keys()) + list(risk_metrics.keys()),
            'Value': list(summary.values()) + list(risk_metrics.values())
        }
        report_df = pd.DataFrame(report_data)
        
        return dict(content=report_df.to_csv(index=False), filename="trade_analysis_report.csv")
    
    
    # Callback 11: Export data
    @app.callback(
        Output('download-data', 'data'),
        [Input('export-csv-btn', 'n_clicks')],
        [State('trade-data-loaded-store', 'data')],
        prevent_initial_call=True
    )
    def export_data(n_clicks, data_store):
        """
        Export trade data as CSV
        
        Requirements: 0.11
        """
        if not n_clicks or not has_data_in_store(data_store):
            raise PreventUpdate
        
        trades_df = load_data_from_store(data_store)
        if trades_df is None:
            raise PreventUpdate
        
        return dict(content=trades_df.to_csv(index=False), filename="trade_data_export.csv")
    
    
    # Callback 12: Navigate to next page
    @app.callback(
        Output('main-tabs', 'active_tab'),
        [
            Input('navigate-to-probability-btn', 'n_clicks'),
            Input('navigate-to-ml-prediction-btn', 'n_clicks')
        ],
        prevent_initial_call=True
    )
    def navigate_to_next_page(prob_clicks, ml_clicks):
        """
        Navigate to Probability Explorer or ML Prediction Engine page
        
        Requirements: 0.11, 15.1
        """
        from dash import callback_context
        
        if not callback_context.triggered:
            raise PreventUpdate
        
        trigger_id = callback_context.triggered[0]['prop_id'].split('.')[0]
        
        if trigger_id == 'navigate-to-probability-btn':
            return 'probability-explorer'
        elif trigger_id == 'navigate-to-ml-prediction-btn':
            return 'ml-prediction-engine'
        
        raise PreventUpdate
    
    
    # Callback 13: Enable navigation buttons when data is loaded
    @app.callback(
        [
            Output('navigate-to-probability-btn', 'disabled'),
            Output('navigate-to-ml-prediction-btn', 'disabled')
        ],
        [
            Input('trade-data-loaded-store', 'data'),
            Input('trade-data-store', 'data'),
            Input('merged-data-store', 'data')
        ]
    )
    def enable_navigation_buttons(local_store, global_trade_store, global_merged_store):
        """
        Enable navigation buttons when data is loaded
        
        Requirements: 0.11, 15.1
        """
        # Use whichever store has data
        data_store = None
        if local_store and (local_store.get('loaded') or 'data' in local_store):
            data_store = local_store
        elif global_merged_store and 'data' in global_merged_store:
            data_store = global_merged_store
        elif global_trade_store and 'data' in global_trade_store:
            data_store = global_trade_store
        
        # Enable buttons if data is available
        has_data = has_data_in_store(data_store)
        return not has_data, not has_data  # disabled=False when data exists
    
    
    # Callback 14: Auto-restore data status
    @app.callback(
        Output('auto-restore-status', 'children'),
        [Input('url', 'pathname')],
        prevent_initial_call=True
    )
    def show_data_status(pathname):
        """
        Show data status when page loads
        """
        # Check if data exists in server cache
        if has_cached_data():
            cache_info = get_cache_info()
            trade_info = cache_info.get('summary', {}).get('trade_data', {})
            rows = trade_info.get('rows', 0)
            
            return dbc.Alert(
                [
                    html.I(className="bi bi-info-circle me-2"),
                    f"Data loaded: {rows:,} trades available"
                ],
                color="info",
                dismissable=True,
                className="mb-2"
            )
        
        return html.Div()



# Helper functions for creating figures

def create_equity_curve_figure(equity_data):
    """Create equity curve figure with drawdown shading"""
    fig = go.Figure()
    
    if not equity_data['timestamps']:
        return fig
    
    # Add equity curve
    fig.add_trace(go.Scatter(
        x=equity_data['timestamps'],
        y=equity_data['equity_values'],
        mode='lines',
        name='Equity',
        line=dict(color='blue', width=2),
        hovertemplate='<b>Date:</b> %{x}<br><b>Equity:</b> $%{y:.2f}<extra></extra>'
    ))
    
    # Add drawdown periods as shaded regions
    for dd in equity_data['drawdown_periods']:
        fig.add_vrect(
            x0=equity_data['timestamps'][dd['start_idx']],
            x1=equity_data['timestamps'][dd['end_idx']],
            fillcolor="red",
            opacity=0.2,
            layer="below",
            line_width=0,
        )
    
    fig.update_layout(
        title="Equity Curve with Drawdown Periods",
        xaxis_title="Date",
        yaxis_title="Equity ($)",
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


def create_r_distribution_figure(r_dist_data):
    """Create R-multiple distribution histogram with KDE"""
    fig = go.Figure()
    
    if not r_dist_data['histogram_data']['bins']:
        return fig
    
    bins = r_dist_data['histogram_data']['bins']
    counts = r_dist_data['histogram_data']['counts']
    
    # Add histogram
    fig.add_trace(go.Bar(
        x=[(bins[i] + bins[i+1])/2 for i in range(len(bins)-1)],
        y=counts,
        name='Frequency',
        marker_color='lightblue',
        hovertemplate='<b>R-Multiple:</b> %{x:.2f}<br><b>Count:</b> %{y}<extra></extra>'
    ))
    
    # Add mean and median lines
    stats = r_dist_data['statistics']
    if stats:
        fig.add_vline(x=stats['mean'], line_dash="dash", line_color="red", 
                     annotation_text=f"Mean: {stats['mean']:.2f}")
        fig.add_vline(x=stats['median'], line_dash="dash", line_color="green",
                     annotation_text=f"Median: {stats['median']:.2f}")
    
    fig.update_layout(
        title="R-Multiple Distribution",
        xaxis_title="R-Multiple",
        yaxis_title="Frequency",
        template='plotly_white',
        showlegend=True
    )
    
    return fig


def create_r_statistics_table(r_dist_data):
    """Create R-multiple statistics table"""
    stats = r_dist_data['statistics']
    threshold_probs = r_dist_data['threshold_probs']
    
    if not stats:
        return html.Div("No data available")
    
    table_data = [
        html.Tr([html.Td("Mean", className="fw-bold"), html.Td(f"{stats['mean']:.2f}R")]),
        html.Tr([html.Td("Median", className="fw-bold"), html.Td(f"{stats['median']:.2f}R")]),
        html.Tr([html.Td("Std Dev", className="fw-bold"), html.Td(f"{stats['std']:.2f}R")]),
        html.Tr([html.Td("P25", className="fw-bold"), html.Td(f"{stats['p25']:.2f}R")]),
        html.Tr([html.Td("P75", className="fw-bold"), html.Td(f"{stats['p75']:.2f}R")]),
        html.Tr([html.Td("P90", className="fw-bold"), html.Td(f"{stats['p90']:.2f}R")]),
        html.Tr([html.Td("P(R>1)", className="fw-bold"), html.Td(f"{threshold_probs['p_r_gt_1']*100:.1f}%")]),
        html.Tr([html.Td("P(R>2)", className="fw-bold"), html.Td(f"{threshold_probs['p_r_gt_2']*100:.1f}%")]),
        html.Tr([html.Td("P(R>3)", className="fw-bold"), html.Td(f"{threshold_probs['p_r_gt_3']*100:.1f}%")]),
    ]
    
    return html.Table(table_data, className="table table-sm")



def create_mae_mfe_figure(mae_mfe_data):
    """Create MAE/MFE scatter plot"""
    fig = go.Figure()
    
    scatter_data = mae_mfe_data.get('scatter_data', {})
    if not scatter_data:
        return fig
    
    mae = scatter_data['mae']
    mfe = scatter_data['mfe']
    r_multiple = scatter_data['r_multiple']
    is_winner = scatter_data['is_winner']
    
    # Separate winners and losers
    winners_mask = [w == 1 for w in is_winner]
    losers_mask = [w == 0 for w in is_winner]
    
    # Add winners
    fig.add_trace(go.Scatter(
        x=[mae[i] for i in range(len(mae)) if winners_mask[i]],
        y=[mfe[i] for i in range(len(mfe)) if winners_mask[i]],
        mode='markers',
        name='Winners',
        marker=dict(color='green', size=8, opacity=0.6),
        hovertemplate='<b>MAE:</b> %{x:.2f}R<br><b>MFE:</b> %{y:.2f}R<extra></extra>'
    ))
    
    # Add losers
    fig.add_trace(go.Scatter(
        x=[mae[i] for i in range(len(mae)) if losers_mask[i]],
        y=[mfe[i] for i in range(len(mfe)) if losers_mask[i]],
        mode='markers',
        name='Losers',
        marker=dict(color='red', size=8, opacity=0.6),
        hovertemplate='<b>MAE:</b> %{x:.2f}R<br><b>MFE:</b> %{y:.2f}R<extra></extra>'
    ))
    
    fig.update_layout(
        title="MAE vs MFE Analysis",
        xaxis_title="MAE (R-Multiple)",
        yaxis_title="MFE (R-Multiple)",
        template='plotly_white',
        hovermode='closest'
    )
    
    return fig


def create_stats_table(stats_dict, title):
    """Create statistics table for winners or losers"""
    if not stats_dict:
        return html.Div("No data available")
    
    table_data = [
        html.Tr([html.Td("Avg MAE", className="fw-bold"), html.Td(f"{stats_dict.get('avg_mae', 0):.2f}R")]),
        html.Tr([html.Td("Avg MFE", className="fw-bold"), html.Td(f"{stats_dict.get('avg_mfe', 0):.2f}R")]),
    ]
    
    if 'mfe_to_r_ratio' in stats_dict:
        table_data.append(
            html.Tr([html.Td("MFE/R Ratio", className="fw-bold"), 
                    html.Td(f"{stats_dict['mfe_to_r_ratio']:.2f}")])
        )
    
    if 'profit_left' in stats_dict:
        table_data.append(
            html.Tr([html.Td("Profit Left", className="fw-bold"), 
                    html.Td(f"{stats_dict['profit_left']:.2f}R")])
        )
    
    return html.Table(table_data, className="table table-sm")


def create_hourly_performance_chart(hourly_data):
    """Create hourly performance heatmap"""
    if not hourly_data:
        return html.Div("No hourly data available")
    
    hours = sorted(hourly_data.keys())
    win_rates = [hourly_data[h]['win_rate'] * 100 for h in hours]
    counts = [hourly_data[h]['count'] for h in hours]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=hours,
        y=win_rates,
        text=[f"{wr:.1f}%<br>n={c}" for wr, c in zip(win_rates, counts)],
        textposition='auto',
        marker_color=win_rates,
        marker_colorscale='RdYlGn',
        marker_cmin=0,
        marker_cmax=100,
        hovertemplate='<b>Hour:</b> %{x}<br><b>Win Rate:</b> %{y:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title="Performance by Hour of Day",
        xaxis_title="Hour",
        yaxis_title="Win Rate (%)",
        template='plotly_white'
    )
    
    return dcc.Graph(figure=fig, config={'displayModeBar': True, 'displaylogo': False})


def create_daily_performance_chart(daily_data):
    """Create daily performance bar chart"""
    if not daily_data:
        return html.Div("No daily data available")
    
    days = list(daily_data.keys())
    win_rates = [daily_data[d]['win_rate'] * 100 for d in days]
    counts = [daily_data[d]['count'] for d in days]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=days,
        y=win_rates,
        text=[f"{wr:.1f}%<br>n={c}" for wr, c in zip(win_rates, counts)],
        textposition='auto',
        marker_color=win_rates,
        marker_colorscale='RdYlGn',
        marker_cmin=0,
        marker_cmax=100,
        hovertemplate='<b>Day:</b> %{x}<br><b>Win Rate:</b> %{y:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title="Performance by Day of Week",
        xaxis_title="Day",
        yaxis_title="Win Rate (%)",
        template='plotly_white'
    )
    
    return dcc.Graph(figure=fig, config={'displayModeBar': True, 'displaylogo': False})


def create_weekly_performance_chart(weekly_data):
    """Create weekly performance chart"""
    if not weekly_data:
        return html.Div("No weekly data available")
    
    weeks = sorted(list(weekly_data.keys()))
    win_rates = [weekly_data[w]['win_rate'] * 100 for w in weeks]
    total_profits = [weekly_data[w]['total_profit'] for w in weeks]
    counts = [weekly_data[w]['count'] for w in weeks]
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(x=weeks, y=win_rates, name="Win Rate (%)", 
               marker_color='lightblue',
               text=[f"{wr:.1f}%<br>n={c}" for wr, c in zip(win_rates, counts)],
               textposition='auto'),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=weeks, y=total_profits, name="Total Profit", 
                   mode='lines+markers',
                   line=dict(color='green', width=2),
                   marker=dict(size=8)),
        secondary_y=True
    )
    
    fig.update_layout(
        title="Performance by Week",
        template='plotly_white',
        xaxis_tickangle=-45,
        xaxis_title="Week"
    )
    fig.update_yaxes(title_text="Win Rate (%)", secondary_y=False)
    fig.update_yaxes(title_text="Total Profit ($)", secondary_y=True)
    
    return dcc.Graph(figure=fig, config={'displayModeBar': True, 'displaylogo': False})


def create_monthly_performance_chart(monthly_data):
    """Create monthly performance chart"""
    if not monthly_data:
        return html.Div("No monthly data available")
    
    months = sorted(list(monthly_data.keys()))
    win_rates = [monthly_data[m]['win_rate'] * 100 for m in months]
    total_profits = [monthly_data[m]['total_profit'] for m in months]
    counts = [monthly_data[m]['count'] for m in months]
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(x=months, y=win_rates, name="Win Rate (%)", 
               marker_color='lightcoral',
               text=[f"{wr:.1f}%<br>n={c}" for wr, c in zip(win_rates, counts)],
               textposition='auto'),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=months, y=total_profits, name="Total Profit", 
                   mode='lines+markers',
                   line=dict(color='darkgreen', width=3),
                   marker=dict(size=10)),
        secondary_y=True
    )
    
    fig.update_layout(
        title="Performance by Month",
        template='plotly_white',
        xaxis_tickangle=-45,
        xaxis_title="Month"
    )
    fig.update_yaxes(title_text="Win Rate (%)", secondary_y=False)
    fig.update_yaxes(title_text="Total Profit ($)", secondary_y=True)
    
    return dcc.Graph(figure=fig, config={'displayModeBar': True, 'displaylogo': False})


def create_session_performance_chart(session_data):
    """Create session performance grouped bar chart"""
    if not session_data:
        return html.Div("No session data available")
    
    sessions = list(session_data.keys())
    win_rates = [session_data[s]['win_rate'] * 100 for s in sessions]
    avg_r = [session_data[s]['avg_r'] for s in sessions]
    counts = [session_data[s]['count'] for s in sessions]
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(x=sessions, y=win_rates, name="Win Rate (%)", 
               marker_color='lightblue',
               text=[f"{wr:.1f}%" for wr in win_rates],
               textposition='auto'),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Bar(x=sessions, y=avg_r, name="Avg R", 
               marker_color='lightgreen',
               text=[f"{r:.2f}R" for r in avg_r],
               textposition='auto'),
        secondary_y=True
    )
    
    fig.update_layout(
        title="Performance by Trading Session",
        template='plotly_white',
        xaxis_title="Session"
    )
    fig.update_yaxes(title_text="Win Rate (%)", secondary_y=False)
    fig.update_yaxes(title_text="Avg R-Multiple", secondary_y=True)
    
    return dcc.Graph(figure=fig, config={'displayModeBar': True, 'displaylogo': False})



def create_direction_analysis_figure(direction_data):
    """Create direction analysis figure (BUY vs SELL)"""
    if not direction_data:
        return go.Figure()
    
    directions = list(direction_data.keys())
    win_rates = [direction_data[d]['win_rate'] * 100 for d in directions]
    counts = [direction_data[d]['count'] for d in directions]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=directions,
        y=win_rates,
        text=[f"{wr:.1f}%<br>n={c}" for wr, c in zip(win_rates, counts)],
        textposition='auto',
        marker_color=['green' if d == 'BUY' else 'red' for d in directions],
        hovertemplate='<b>Direction:</b> %{x}<br><b>Win Rate:</b> %{y:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title="BUY vs SELL Performance",
        xaxis_title="Direction",
        yaxis_title="Win Rate (%)",
        template='plotly_white'
    )
    
    return fig


def create_exit_reason_figure(exit_reason_data):
    """Create exit reason pie chart"""
    if not exit_reason_data:
        return go.Figure()
    
    reasons = list(exit_reason_data.keys())
    counts = [exit_reason_data[r]['count'] for r in reasons]
    
    fig = go.Figure()
    
    fig.add_trace(go.Pie(
        labels=reasons,
        values=counts,
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Exit Reason Distribution",
        template='plotly_white'
    )
    
    return fig


def create_streak_timeline_figure(consecutive_data):
    """Create streak timeline figure"""
    streaks = consecutive_data.get('streaks', {})
    
    if not streaks:
        return go.Figure()
    
    win_streaks = streaks.get('win_streaks', [])
    loss_streaks = streaks.get('loss_streaks', [])
    
    # Create alternating pattern
    all_streaks = []
    streak_types = []
    positions = []
    
    pos = 0
    for i, (w, l) in enumerate(zip(win_streaks + [0], loss_streaks + [0])):
        if w > 0:
            all_streaks.append(w)
            streak_types.append('Win')
            positions.append(pos)
            pos += w
        if l > 0:
            all_streaks.append(-l)
            streak_types.append('Loss')
            positions.append(pos)
            pos += l
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=positions,
        y=all_streaks,
        marker_color=['green' if s > 0 else 'red' for s in all_streaks],
        hovertemplate='<b>Streak:</b> %{y}<br><b>Position:</b> %{x}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Win/Loss Streak Timeline",
        xaxis_title="Trade Position",
        yaxis_title="Streak Length",
        template='plotly_white'
    )
    
    return fig


def create_cumulative_performance_figure(consecutive_data):
    """Create cumulative performance figure"""
    cumulative = consecutive_data.get('cumulative_by_streak', {})
    
    if not cumulative:
        return go.Figure()
    
    positions = cumulative.get('positions', [])
    cumulative_r = cumulative.get('cumulative_r', [])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=positions,
        y=cumulative_r,
        mode='lines',
        line=dict(color='blue', width=2),
        fill='tozeroy',
        hovertemplate='<b>Trade #:</b> %{x}<br><b>Cumulative R:</b> %{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Cumulative R-Multiple Performance",
        xaxis_title="Trade Number",
        yaxis_title="Cumulative R-Multiple",
        template='plotly_white'
    )
    
    return fig


def create_risk_metrics_table(risk_metrics):
    """Create comprehensive risk metrics table"""
    if not risk_metrics:
        return html.Div("No risk metrics available")
    
    metrics_data = [
        ["Sharpe Ratio", f"{risk_metrics['sharpe_ratio']:.2f}"],
        ["Sortino Ratio", f"{risk_metrics['sortino_ratio']:.2f}"],
        ["Calmar Ratio", f"{risk_metrics['calmar_ratio']:.2f}"],
        ["Max Drawdown", f"{risk_metrics['max_drawdown_pct']:.2f}%"],
        ["Max DD Duration", f"{risk_metrics['max_drawdown_duration']} days"],
        ["Recovery Factor", f"{risk_metrics['recovery_factor']:.2f}"],
        ["Profit/Max DD Ratio", f"{risk_metrics['profit_to_max_dd_ratio']:.2f}"],
        ["Win/Loss Ratio", f"{risk_metrics['win_loss_ratio']:.2f}"],
        ["Avg Win/Avg Loss", f"{risk_metrics['avg_win_to_avg_loss']:.2f}"],
        ["Largest Win", f"${risk_metrics['largest_win']:.2f}"],
        ["Largest Loss", f"${risk_metrics['largest_loss']:.2f}"],
        ["Max Consecutive Wins", f"{risk_metrics['consecutive_wins_max']}"],
        ["Max Consecutive Losses", f"{risk_metrics['consecutive_losses_max']}"],
        ["Percent Profitable", f"{risk_metrics['percent_profitable']:.1f}%"],
    ]
    
    table = dash_table.DataTable(
        columns=[
            {"name": "Metric", "id": "metric"},
            {"name": "Value", "id": "value"}
        ],
        data=[{"metric": m[0], "value": m[1]} for m in metrics_data],
        style_cell={
            'textAlign': 'left',
            'padding': '10px',
            'fontSize': '14px'
        },
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            }
        ]
    )
    
    return table
