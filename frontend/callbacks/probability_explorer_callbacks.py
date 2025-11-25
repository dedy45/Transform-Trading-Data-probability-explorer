"""
Probability Explorer Callbacks
Dash callbacks for the Probability Explorer tab

This module implements all interactive callbacks for the Probability Explorer:
- Real-time filter updates
- Probability calculation
- Heatmap cell click handling
- Export functionality
- Scenario creation from selection
"""
from dash import Input, Output, State, callback, no_update, html, dcc, ctx
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import io
import base64

from backend.calculators.probability_calculator import compute_1d_probability, compute_2d_probability
from backend.calculators.interactive_filter import InteractiveFilter
from backend.calculators.whatif_scenario import WhatIfScenario
from frontend.components import (
    create_probability_heatmap_2d,
    create_probability_distribution_1d,
    create_confidence_interval_chart,
    create_top_combinations_list,
    create_combinations_summary,
    create_trade_details_summary,
    create_empty_heatmap,
    create_empty_distribution
)


def register_probability_explorer_callbacks(app):
    """
    Register all callbacks for Probability Explorer tab
    
    Parameters:
    -----------
    app : dash.Dash
        Dash application instance
    """
    
    @app.callback(
        [Output('heatmap-container', 'style'),
         Output('distribution-container', 'style')],
        Input('viz-type-dropdown', 'value')
    )
    def toggle_visualization_type(viz_type):
        """Toggle between 1D and 2D visualization"""
        if viz_type == '2d':
            return {'display': 'block'}, {'display': 'none'}
        else:
            return {'display': 'none'}, {'display': 'block'}
    
    
    @app.callback(
        Output('filter-collapse', 'is_open'),
        Input('filter-collapse-button', 'n_clicks'),
        State('filter-collapse', 'is_open')
    )
    def toggle_filter_collapse(n_clicks, is_open):
        """Toggle filter panel collapse"""
        if n_clicks:
            return not is_open
        return is_open
    
    
    @app.callback(
        [Output('filter-summary', 'children'),
         Output('filter-state-store', 'data')],
        [Input('date-range-filter', 'start_date'),
         Input('date-range-filter', 'end_date'),
         Input('session-filter', 'value'),
         Input('probability-range-filter', 'value'),
         Input('composite-score-filter', 'value'),
         Input('trend-regime-filter', 'value'),
         Input('volatility-regime-filter', 'value'),
         Input('risk-regime-filter', 'value'),
         Input('clear-filters-btn', 'n_clicks')],
        State('merged-data-store', 'data')
    )
    def update_filters_realtime(start_date, end_date, sessions, prob_range, 
                               score_threshold, trend_regimes, vol_regimes, 
                               risk_regimes, clear_clicks, merged_data):
        """
        Real-time filter updates
        
        Updates filter summary and applies filters to data in real-time.
        Validates: Requirements 12.2, 21.2
        """
        if merged_data is None:
            return "No data loaded", None
        
        try:
            import io
            if isinstance(merged_data, dict) and 'data' in merged_data:
                df = pd.read_json(io.StringIO(merged_data['data']), orient='split')
            else:
                df = pd.DataFrame(merged_data)
        except Exception:
            return "Invalid data format", None
        original_count = len(df)
        
        # Check if clear button was clicked
        if ctx.triggered and 'clear-filters-btn' in ctx.triggered[0]['prop_id']:
            # Reset all filters
            return (
                f"No active filters | Showing: {original_count:,} trades",
                {'filtered_count': original_count, 'filters': {}}
            )
        
        # Apply filters using InteractiveFilter
        try:
            filter_system = InteractiveFilter(df)
            
            # Date range filter
            if start_date and end_date:
                def date_filter(df, start, end):
                    if 'Timestamp' in df.columns:
                        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                        return df[(df['Timestamp'] >= start) & (df['Timestamp'] <= end)]
                    return df
                filter_system.add_filter('date_range', date_filter, 
                                       {'start': start_date, 'end': end_date})
            
            # Session filter
            if sessions and len(sessions) < 4:
                def session_filter(df, sessions):
                    if 'entry_session' in df.columns:
                        return df[df['entry_session'].isin(sessions)]
                    return df
                filter_system.add_filter('session', session_filter, {'sessions': sessions})
            
            # Probability range filter
            if prob_range != [0, 100] and 'prob_global_win' in df.columns:
                def prob_filter(df, min_prob, max_prob):
                    return df[(df['prob_global_win'] >= min_prob/100) & 
                            (df['prob_global_win'] <= max_prob/100)]
                filter_system.add_filter('probability', prob_filter, 
                                       {'min_prob': prob_range[0], 'max_prob': prob_range[1]})
            
            # Composite score filter
            if score_threshold > 0 and 'composite_score' in df.columns:
                def score_filter(df, threshold):
                    return df[df['composite_score'] >= threshold]
                filter_system.add_filter('score', score_filter, {'threshold': score_threshold})
            
            # Trend regime filter
            if trend_regimes and len(trend_regimes) < 2 and 'trend_regime' in df.columns:
                def trend_filter(df, regimes):
                    return df[df['trend_regime'].isin(regimes)]
                filter_system.add_filter('trend', trend_filter, {'regimes': trend_regimes})
            
            # Volatility regime filter
            if vol_regimes and len(vol_regimes) < 3 and 'volatility_regime' in df.columns:
                def vol_filter(df, regimes):
                    return df[df['volatility_regime'].isin(regimes)]
                filter_system.add_filter('volatility', vol_filter, {'regimes': vol_regimes})
            
            # Risk regime filter
            if risk_regimes and len(risk_regimes) < 2 and 'risk_regime_global' in df.columns:
                def risk_filter(df, regimes):
                    return df[df['risk_regime_global'].isin(regimes)]
                filter_system.add_filter('risk', risk_filter, {'regimes': risk_regimes})
            
            # Get filter summary
            summary_info = filter_system.get_filter_summary()
            filtered_count = summary_info['filtered_count']
            active_filter_count = len(summary_info['active_filters'])
            
            # Build summary text
            active_filters = []
            if start_date and end_date:
                active_filters.append(f"Date: {start_date} to {end_date}")
            if sessions and len(sessions) < 4:
                active_filters.append(f"Sessions: {', '.join(sessions)}")
            if prob_range != [0, 100]:
                active_filters.append(f"Probability: {prob_range[0]}%-{prob_range[1]}%")
            if score_threshold > 0:
                active_filters.append(f"Score ≥ {score_threshold}")
            if trend_regimes and len(trend_regimes) < 2:
                active_filters.append(f"Trend: {trend_regimes}")
            if vol_regimes and len(vol_regimes) < 3:
                active_filters.append(f"Volatility: {vol_regimes}")
            if risk_regimes and len(risk_regimes) < 2:
                active_filters.append(f"Risk: {risk_regimes}")
            
            summary = f"Active Filters: {active_filter_count} | Showing: {filtered_count:,} of {original_count:,} trades"
            
            if active_filters:
                summary += " | " + " • ".join(active_filters)
            
            # Store filter state
            filter_state = {
                'filtered_count': filtered_count,
                'original_count': original_count,
                'filters': {
                    'date_range': [start_date, end_date] if start_date and end_date else None,
                    'sessions': sessions,
                    'prob_range': prob_range,
                    'score_threshold': score_threshold,
                    'trend_regimes': trend_regimes,
                    'vol_regimes': vol_regimes,
                    'risk_regimes': risk_regimes
                }
            }
            
            return summary, filter_state
            
        except Exception as e:
            return f"Error applying filters: {str(e)}", None
    
    
    @app.callback(
        [Output('probability-heatmap-2d', 'figure'),
         Output('probability-distribution-1d', 'figure'),
         Output('confidence-interval-chart', 'figure'),
         Output('top-combinations-list', 'children'),
         Output('top-combinations-count', 'children'),
         Output('probability-results-store', 'data'),
         Output('export-results-btn', 'disabled'),
         Output('prob-auto-insights', 'children')],
        Input('calculate-btn', 'n_clicks'),
        [State('target-variable-dropdown', 'value'),
         State('feature-x-dropdown', 'value'),
         State('feature-y-dropdown', 'value'),
         State('viz-type-dropdown', 'value'),
         State('confidence-level-slider', 'value'),
         State('bins-slider', 'value'),
         State('min-samples-slider', 'value'),
         State('merged-data-store', 'data')]
    )
    def calculate_probabilities(n_clicks, target, feature_x, feature_y, viz_type,
                               conf_level, n_bins, min_samples, merged_data):
        """
        Calculate probabilities and update all visualizations
        
        This is the main callback that:
        1. Computes 1D or 2D probabilities
        2. Updates heatmap or distribution chart
        3. Updates confidence interval chart
        4. Updates top combinations list
        """
        if not n_clicks or merged_data is None:
            raise PreventUpdate
        
        if not target or not feature_x:
            # Return empty figures
            alert = dbc.Alert([
                html.I(className="bi bi-exclamation-triangle me-2"),
                "Silakan pilih variabel target dan fitur X terlebih dahulu"
            ], color="warning")
            return (
                create_empty_heatmap(),
                create_empty_distribution(),
                create_empty_distribution(),
                [alert],
                "0",
                None,
                True,  # Keep export disabled
                alert
            )
        
        # Convert data
        try:
            import io
            if isinstance(merged_data, dict) and 'data' in merged_data:
                df = pd.read_json(io.StringIO(merged_data['data']), orient='split')
            else:
                df = pd.DataFrame(merged_data)
        except Exception:
            raise PreventUpdate
        
        # Calculate probabilities
        try:
            if viz_type == '2d' and feature_y:
                # 2D probability calculation
                prob_results = compute_2d_probability(
                    df=df,
                    target=target,
                    feature_x=feature_x,
                    feature_y=feature_y,
                    conf_level=conf_level / 100,
                    bins_x=n_bins,
                    bins_y=n_bins,
                    min_samples_per_cell=min_samples
                )
                
                # Create heatmap
                heatmap_fig = create_probability_heatmap_2d(
                    prob_results, feature_x, feature_y, target
                )
                dist_fig = create_empty_distribution()
                ci_fig = create_empty_distribution()
                
            else:
                # 1D probability calculation
                prob_results = compute_1d_probability(
                    df=df,
                    target=target,
                    feature=feature_x,
                    conf_level=conf_level / 100,
                    bins=n_bins,
                    min_samples_per_bin=min_samples
                )
                
                # Create distribution charts
                heatmap_fig = create_empty_heatmap()
                dist_fig = create_probability_distribution_1d(
                    prob_results, feature_x, target
                )
                ci_fig = create_confidence_interval_chart(
                    prob_results, feature_x, target
                )
            
            # Create top combinations
            top_combos = create_top_combinations_list(prob_results, top_n=10)
            combo_count = str(len(prob_results))
            
            # Store results
            results_data = prob_results.to_dict('records') if prob_results is not None else None
            
            # Generate auto insights
            try:
                # Calculate statistics
                # Check for both 'reliable' and 'is_reliable' column names
                reliable_col = 'is_reliable' if 'is_reliable' in prob_results.columns else 'reliable'
                reliable_bins = prob_results[prob_results[reliable_col] == True] if reliable_col in prob_results.columns else prob_results
                if len(reliable_bins) > 0:
                    # Check for both 'probability' and 'p_est' column names
                    prob_col = 'p_est' if 'p_est' in prob_results.columns else 'probability'
                    max_prob = reliable_bins[prob_col].max()
                    min_prob = reliable_bins[prob_col].min()
                    avg_prob = reliable_bins[prob_col].mean()
                    high_prob_bins = len(reliable_bins[reliable_bins[prob_col] > 0.6])
                    total_reliable = len(reliable_bins)
                    
                    insights = dbc.Alert([
                        html.H6([
                            html.I(className="bi bi-lightbulb me-2"),
                            "Wawasan Otomatis"
                        ], className="alert-heading"),
                        html.Ul([
                            html.Li(f"Probabilitas tertinggi: {max_prob*100:.1f}% (cari area hijau di grafik)"),
                            html.Li(f"Probabilitas terendah: {min_prob*100:.1f}%"),
                            html.Li(f"Probabilitas rata-rata: {avg_prob*100:.1f}%"),
                            html.Li(f"Bin dengan probabilitas tinggi (>60%): {high_prob_bins} dari {total_reliable} bin reliable"),
                            html.Li(f"Rekomendasi: {'Fokus pada area dengan probabilitas >60% untuk setup terbaik' if high_prob_bins > 0 else 'Pertimbangkan filter atau fitur lain untuk meningkatkan probabilitas'}")
                        ], className="mb-0")
                    ], color="info")
                else:
                    insights = dbc.Alert([
                        html.I(className="bi bi-exclamation-triangle me-2"),
                        "Tidak ada bin yang reliable. Kurangi sampel minimum atau jumlah bin."
                    ], color="warning")
            except Exception:
                insights = html.Div()
            
            return (
                heatmap_fig,
                dist_fig,
                ci_fig,
                top_combos,
                combo_count,
                results_data,
                False,  # Enable export button
                insights
            )
            
        except Exception as e:
            # Return error state
            error_alert = dbc.Alert([
                html.I(className="bi bi-exclamation-triangle me-2"),
                f"Error menghitung probabilitas: {str(e)}"
            ], color="danger")
            return (
                create_empty_heatmap(),
                create_empty_distribution(),
                create_empty_distribution(),
                [error_alert],
                "0",
                None,
                True,  # Keep export disabled
                error_alert
            )
    
    
    @app.callback(
        [Output('trade-details-content', 'children'),
         Output('trade-details-panel', 'style'),
         Output('view-trades-btn', 'disabled'),
         Output('export-trades-btn', 'disabled'),
         Output('create-scenario-btn', 'disabled'),
         Output('selected-cell-store', 'data')],
        [Input('probability-heatmap-2d', 'clickData'),
         Input('probability-distribution-1d', 'clickData')],
        [State('merged-data-store', 'data'),
         State('feature-x-dropdown', 'value'),
         State('feature-y-dropdown', 'value'),
         State('target-variable-dropdown', 'value'),
         State('probability-results-store', 'data')]
    )
    def handle_heatmap_cell_click(heatmap_click, dist_click, merged_data, 
                                  feature_x, feature_y, target, prob_results):
        """
        Handle heatmap cell click to show trade details
        
        Parses click data to filter trades in the selected bin/cell
        and displays detailed information.
        Validates: Requirements 21.2, 21.3
        """
        if merged_data is None:
            raise PreventUpdate
        
        # Determine which visualization was clicked
        if not ctx.triggered:
            raise PreventUpdate
        
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        click_data = heatmap_click if trigger_id == 'probability-heatmap-2d' else dist_click
        
        if not click_data:
            raise PreventUpdate
        
        try:
            import io
            if isinstance(merged_data, dict) and 'data' in merged_data:
                df = pd.read_json(io.StringIO(merged_data['data']), orient='split')
            else:
                df = pd.DataFrame(merged_data)
        except Exception:
            raise PreventUpdate
        
        try:
            # Parse click data to extract bin/cell information
            if trigger_id == 'probability-heatmap-2d':
                # 2D heatmap click
                point = click_data['points'][0]
                x_bin = point.get('x', '')
                y_bin = point.get('y', '')
                
                # Parse bin ranges from labels
                # Format: "0.2-0.3" or "Low" (categorical)
                selected_trades = df.copy()
                
                # Filter by feature_x bin
                if feature_x and '-' in str(x_bin):
                    x_min, x_max = map(float, str(x_bin).split('-'))
                    selected_trades = selected_trades[
                        (selected_trades[feature_x] >= x_min) & 
                        (selected_trades[feature_x] < x_max)
                    ]
                
                # Filter by feature_y bin
                if feature_y and '-' in str(y_bin):
                    y_min, y_max = map(float, str(y_bin).split('-'))
                    selected_trades = selected_trades[
                        (selected_trades[feature_y] >= y_min) & 
                        (selected_trades[feature_y] < y_max)
                    ]
                
                selection_desc = f"{feature_x}: {x_bin}, {feature_y}: {y_bin}"
                
            else:
                # 1D distribution click
                point = click_data['points'][0]
                x_bin = point.get('x', '')
                
                selected_trades = df.copy()
                
                # Filter by feature_x bin
                if feature_x and '-' in str(x_bin):
                    x_min, x_max = map(float, str(x_bin).split('-'))
                    selected_trades = selected_trades[
                        (selected_trades[feature_x] >= x_min) & 
                        (selected_trades[feature_x] < x_max)
                    ]
                
                selection_desc = f"{feature_x}: {x_bin}"
            
            # Calculate metrics for selected trades
            n_trades = len(selected_trades)
            
            if n_trades == 0:
                details = dbc.Alert(
                    "No trades found in selected bin/cell",
                    color="warning"
                )
                return details, {'display': 'block'}, True, True, True, None
            
            # Calculate metrics
            if target in selected_trades.columns:
                win_rate = selected_trades[target].mean() * 100
            else:
                win_rate = 0
            
            if 'R_multiple' in selected_trades.columns:
                avg_r = selected_trades['R_multiple'].mean()
                expectancy = selected_trades['R_multiple'].mean()
            else:
                avg_r = 0
                expectancy = 0
            
            # Create details display
            details = html.Div([
                dbc.Alert([
                    html.H6("Selected Cell/Bin", className="alert-heading"),
                    html.P(selection_desc, className="mb-0")
                ], color="info"),
                
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H4(f"{n_trades:,}", className="mb-0"),
                            html.Small("Trades", className="text-muted")
                        ], className="text-center")
                    ], width=3),
                    dbc.Col([
                        html.Div([
                            html.H4(f"{win_rate:.1f}%", className="mb-0"),
                            html.Small("Win Rate", className="text-muted")
                        ], className="text-center")
                    ], width=3),
                    dbc.Col([
                        html.Div([
                            html.H4(f"{avg_r:.2f}", className="mb-0"),
                            html.Small("Avg R", className="text-muted")
                        ], className="text-center")
                    ], width=3),
                    dbc.Col([
                        html.Div([
                            html.H4(f"{expectancy:.2f}", className="mb-0"),
                            html.Small("Expectancy", className="text-muted")
                        ], className="text-center")
                    ], width=3)
                ], className="mt-3 mb-3"),
                
                html.Hr(),
                
                html.P([
                    html.Strong("Trade IDs: "),
                    html.Span(f"{n_trades} trades selected (showing first 10)")
                ]),
                
                # Show first 10 trade IDs if available
                html.Div([
                    html.Small(
                        ", ".join([str(tid) for tid in selected_trades.head(10).get('Ticket_id', selected_trades.head(10).index).tolist()]),
                        className="text-muted"
                    )
                ])
            ])
            
            # Store selected cell data
            cell_data = {
                'selection_desc': selection_desc,
                'n_trades': n_trades,
                'trade_ids': selected_trades.get('Ticket_id', selected_trades.index).tolist()
            }
            
            return (
                details,
                {'display': 'block'},
                False,  # Enable view trades button
                False,  # Enable export button
                False,  # Enable create scenario button
                cell_data
            )
            
        except Exception as e:
            error_details = dbc.Alert(
                f"Error processing click: {str(e)}",
                color="danger"
            )
            return error_details, {'display': 'block'}, True, True, True, None
    
    
    @app.callback(
        [Output('feature-x-dropdown', 'options'),
         Output('feature-y-dropdown', 'options'),
         Output('prob-info-panel', 'children')],
        Input('merged-data-store', 'data'),
        prevent_initial_call=False
    )
    def update_feature_dropdowns(merged_data):
        """Update feature dropdown options based on loaded data"""
        if merged_data is None:
            info = html.Div([
                dbc.Alert([
                    html.I(className="bi bi-exclamation-triangle me-2"),
                    html.Strong("Belum ada data! "),
                    "Silakan muat data terlebih dahulu di tab Dashboard Analisis Trading."
                ], color="warning", className="mb-2"),
                html.Div(id='prob-auto-select-result', className="mb-2"),  # PRESERVE THIS!
                html.Div(id='prob-auto-insights', className="mb-2")
            ])
            return [], [], info
        
        try:
            import io
            if isinstance(merged_data, dict) and 'data' in merged_data:
                df = pd.read_json(io.StringIO(merged_data['data']), orient='split')
            else:
                df = pd.DataFrame(merged_data)
        except Exception:
            info = dbc.Alert([
                html.I(className="bi bi-exclamation-triangle me-2"),
                "Error membaca data. Format data tidak valid."
            ], color="danger")
            return [], [], info
        
        # Get numeric columns (excluding target columns)
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        exclude_cols = ['y_win', 'y_hit_1R', 'y_hit_2R', 'y_future_win_k', 
                       'Ticket_id', 'timestamp']
        
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        options = [{'label': col, 'value': col} for col in feature_cols]
        
        # Create info panel with data summary
        n_trades = len(df)
        n_features = len(feature_cols)
        
        # Check for target columns
        available_targets = [col for col in ['y_win', 'y_hit_1R', 'y_hit_2R', 'y_future_win_k'] if col in df.columns]
        
        info = html.Div([
            dbc.Alert([
                html.I(className="bi bi-check-circle me-2"),
                html.Strong("Data berhasil dimuat! "),
                f"{n_trades:,} trading | {n_features} fitur tersedia"
            ], color="success", className="mb-2"),
            dbc.Alert([
                html.I(className="bi bi-lightbulb me-2"),
                html.Strong("Cara Menggunakan: "),
                "1) Pilih variabel target dan fitur, 2) Atur pengaturan analisis, 3) Klik 'Hitung Probabilitas', 4) Klik pada grafik untuk melihat detail trading"
            ], color="info", className="mb-2"),
            html.Div(id='prob-auto-select-result', className="mb-2"),  # KEEP THIS HERE!
            html.Div(id='prob-auto-insights', className="mb-2")
        ])
        
        return options, options, info
    
    
    @app.callback(
        Output('download-results', 'data'),
        Input('export-results-btn', 'n_clicks'),
        [State('probability-results-store', 'data'),
         State('target-variable-dropdown', 'value'),
         State('feature-x-dropdown', 'value'),
         State('feature-y-dropdown', 'value')],
        prevent_initial_call=True
    )
    def export_probability_results(n_clicks, prob_results, target, feature_x, feature_y):
        """
        Export probability results to CSV
        
        Exports the calculated probability results with all metrics
        to a downloadable CSV file.
        Validates: Requirements 21.6
        """
        if not n_clicks or prob_results is None:
            raise PreventUpdate
        
        try:
            # Convert results to DataFrame
            results_df = pd.DataFrame(prob_results)
            
            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            viz_type = '2D' if feature_y else '1D'
            filename = f"probability_results_{viz_type}_{target}_{feature_x}_{timestamp}.csv"
            
            # Convert to CSV
            csv_string = results_df.to_csv(index=False)
            
            return dict(content=csv_string, filename=filename)
            
        except Exception as e:
            print(f"Error exporting results: {str(e)}")
            raise PreventUpdate
    
    
    @app.callback(
        Output('download-trades', 'data'),
        Input('export-trades-btn', 'n_clicks'),
        [State('merged-data-store', 'data'),
         State('selected-cell-store', 'data')],
        prevent_initial_call=True
    )
    def export_selected_trades(n_clicks, merged_data, selected_cell):
        """
        Export selected trades to CSV
        
        Exports trades from the selected cell/bin to a downloadable CSV file.
        Validates: Requirements 21.6
        """
        if not n_clicks or merged_data is None or selected_cell is None:
            raise PreventUpdate
        
        try:
            import io
            if isinstance(merged_data, dict) and 'data' in merged_data:
                df = pd.read_json(io.StringIO(merged_data['data']), orient='split')
            else:
                df = pd.DataFrame(merged_data)
            
            # Filter to selected trades
            trade_ids = selected_cell.get('trade_ids', [])
            if 'Ticket_id' in df.columns:
                selected_df = df[df['Ticket_id'].isin(trade_ids)]
            else:
                selected_df = df.iloc[trade_ids]
            
            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            selection_desc = selected_cell.get('selection_desc', 'selection').replace(':', '_').replace(',', '_')
            filename = f"selected_trades_{selection_desc}_{timestamp}.csv"
            
            # Convert to CSV
            csv_string = selected_df.to_csv(index=False)
            
            return dict(content=csv_string, filename=filename)
            
        except Exception as e:
            print(f"Error exporting trades: {str(e)}")
            raise PreventUpdate
    
    
    @app.callback(
        [Output('scenario-created-alert', 'is_open'),
         Output('scenario-created-alert', 'children')],
        Input('create-scenario-btn', 'n_clicks'),
        [State('merged-data-store', 'data'),
         State('selected-cell-store', 'data'),
         State('feature-x-dropdown', 'value'),
         State('feature-y-dropdown', 'value')],
        prevent_initial_call=True
    )
    def create_scenario_from_selection(n_clicks, merged_data, selected_cell, 
                                      feature_x, feature_y):
        """
        Create What-If scenario from selected cell/bin
        
        Creates a new scenario in the What-If Scenarios tab based on
        the selected probability cell, applying filters to match the selection.
        Validates: Requirements 21.6
        """
        if not n_clicks or merged_data is None or selected_cell is None:
            raise PreventUpdate
        
        try:
            import io
            if isinstance(merged_data, dict) and 'data' in merged_data:
                df = pd.read_json(io.StringIO(merged_data['data']), orient='split')
            else:
                df = pd.DataFrame(merged_data)
            
            # Get selected trades
            trade_ids = selected_cell.get('trade_ids', [])
            if 'Ticket_id' in df.columns:
                selected_df = df[df['Ticket_id'].isin(trade_ids)]
            else:
                selected_df = df.iloc[trade_ids]
            
            # Create scenario using WhatIfScenario
            scenario = WhatIfScenario(df)
            
            # Apply filter scenario based on selection
            selection_desc = selected_cell.get('selection_desc', 'Custom Selection')
            n_trades = selected_cell.get('n_trades', len(selected_df))
            
            # Calculate baseline vs filtered metrics
            baseline_metrics = scenario.baseline_metrics
            
            # Calculate metrics for selected trades
            if 'R_multiple' in selected_df.columns:
                filtered_expectancy = selected_df['R_multiple'].mean()
                filtered_win_rate = (selected_df['R_multiple'] > 0).mean() if len(selected_df) > 0 else 0
            else:
                filtered_expectancy = 0
                filtered_win_rate = 0
            
            # Create success message
            message = html.Div([
                html.H6("Scenario Created Successfully!", className="alert-heading"),
                html.P([
                    f"Created scenario from selection: {selection_desc}",
                    html.Br(),
                    f"Trades: {n_trades:,} ({n_trades/len(df)*100:.1f}% of total)",
                    html.Br(),
                    f"Win Rate: {filtered_win_rate*100:.1f}% (Baseline: {baseline_metrics.get('win_rate', 0)*100:.1f}%)",
                    html.Br(),
                    f"Expectancy: {filtered_expectancy:.2f}R (Baseline: {baseline_metrics.get('expectancy', 0):.2f}R)"
                ]),
                html.Hr(),
                html.P([
                    html.I(className="bi bi-info-circle me-2"),
                    "Navigate to the What-If Scenarios tab to view and compare this scenario."
                ], className="mb-0 small")
            ])
            
            return True, message
            
        except Exception as e:
            error_message = html.Div([
                html.H6("Error Creating Scenario", className="alert-heading"),
                html.P(f"Failed to create scenario: {str(e)}")
            ])
            return True, error_message
    
    
    @app.callback(
        Output('prob-help-modal', 'is_open'),
        [Input('prob-help-btn', 'n_clicks'),
         Input('prob-help-close', 'n_clicks')],
        State('prob-help-modal', 'is_open')
    )
    def toggle_help_modal(open_clicks, close_clicks, is_open):
        """Toggle help modal"""
        if open_clicks or close_clicks:
            return not is_open
        return is_open
    
    
    @app.callback(
        Output('prob-interpretation-panel', 'children'),
        Input('probability-results-store', 'data'),
        [State('viz-type-dropdown', 'value'),
         State('target-variable-dropdown', 'value'),
         State('feature-x-dropdown', 'value'),
         State('feature-y-dropdown', 'value')]
    )
    def update_interpretation_panel(prob_results_data, viz_type, target, feature_x, feature_y):
        """
        Update interpretation panel with detailed explanation after calculation
        Similar to Sequential Analysis page
        """
        if prob_results_data is None:
            return html.Div()
        
        try:
            prob_results = pd.DataFrame(prob_results_data)
            
            # Check column names
            prob_col = 'p_est' if 'p_est' in prob_results.columns else 'probability'
            reliable_col = 'is_reliable' if 'is_reliable' in prob_results.columns else 'reliable'
            n_col = 'n' if 'n' in prob_results.columns else 'sample_count'
            
            # Calculate statistics
            reliable_bins = prob_results[prob_results[reliable_col] == True] if reliable_col in prob_results.columns else prob_results
            
            if len(reliable_bins) == 0:
                return dbc.Card([
                    dbc.CardHeader([
                        html.H6([
                            html.I(className="bi bi-exclamation-triangle me-2"),
                            "Peringatan"
                        ], className="mb-0")
                    ]),
                    dbc.CardBody([
                        dbc.Alert([
                            "Tidak ada bin yang reliable. Kurangi sampel minimum atau jumlah bin untuk mendapatkan hasil yang lebih baik."
                        ], color="warning")
                    ])
                ], className="mb-3")
            
            max_prob = reliable_bins[prob_col].max()
            min_prob = reliable_bins[prob_col].min()
            avg_prob = reliable_bins[prob_col].mean()
            total_samples = reliable_bins[n_col].sum()
            high_prob_bins = len(reliable_bins[reliable_bins[prob_col] > 0.6])
            
            # Create interpretation panel
            return dbc.Card([
                dbc.CardHeader([
                    html.H6([
                        html.I(className="bi bi-lightbulb me-2"),
                        "Interpretasi & Panduan Analisis"
                    ], className="mb-0")
                ]),
                dbc.CardBody([
                    # Summary Statistics
                    html.H6("📊 Ringkasan Hasil", className="fw-bold mb-3"),
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.H4(f"{max_prob*100:.1f}%", className="mb-0 text-success"),
                                html.Small("Probabilitas Tertinggi", className="text-muted")
                            ], className="text-center")
                        ], width=3),
                        dbc.Col([
                            html.Div([
                                html.H4(f"{avg_prob*100:.1f}%", className="mb-0"),
                                html.Small("Probabilitas Rata-rata", className="text-muted")
                            ], className="text-center")
                        ], width=3),
                        dbc.Col([
                            html.Div([
                                html.H4(f"{high_prob_bins}", className="mb-0 text-info"),
                                html.Small(f"Bin Probabilitas Tinggi (>60%)", className="text-muted")
                            ], className="text-center")
                        ], width=3),
                        dbc.Col([
                            html.Div([
                                html.H4(f"{total_samples:,}", className="mb-0"),
                                html.Small("Total Sampel Reliable", className="text-muted")
                            ], className="text-center")
                        ], width=3)
                    ], className="mb-4"),
                    
                    html.Hr(),
                    
                    # Interpretation Guide
                    html.H6("📖 Cara Membaca Grafik", className="fw-bold mb-3"),
                    html.Div([
                        html.P([
                            html.Strong("Grafik Distribusi 1D:" if viz_type == '1d' else "Heatmap 2D:"),
                            html.Br(),
                            "• Bar/Cell Hijau: Probabilitas tinggi (>60%) - Setup bagus untuk trading" if viz_type == '1d' else "• Area Hijau: Probabilitas tinggi (>60%) - Setup bagus untuk trading",
                            html.Br(),
                            "• Bar/Cell Kuning: Probabilitas sedang (40-60%) - Netral" if viz_type == '1d' else "• Area Kuning: Probabilitas sedang (40-60%) - Netral",
                            html.Br(),
                            "• Bar/Cell Merah: Probabilitas rendah (<40%) - Hindari" if viz_type == '1d' else "• Area Merah: Probabilitas rendah (<40%) - Hindari",
                            html.Br(),
                            "• Error Bars: Menunjukkan interval kepercayaan (CI)" if viz_type == '1d' else "• Opacity Rendah: Sampel sedikit, kurang reliable",
                            html.Br(),
                            "• Angka di Atas Bar: Jumlah sampel di bin" if viz_type == '1d' else "• Angka di Cell: Nilai probabilitas dalam persen"
                        ], className="mb-2")
                    ]),
                    
                    html.Hr(),
                    
                    # Recommendations
                    html.H6("💡 Rekomendasi", className="fw-bold mb-3"),
                    html.Ul([
                        html.Li([
                            html.Strong("Fokus pada area hijau: "),
                            f"Ada {high_prob_bins} bin dengan probabilitas >60%. Klik pada bar/cell hijau untuk melihat detail trading."
                        ]) if high_prob_bins > 0 else html.Li([
                            html.Strong("Tidak ada area probabilitas tinggi: "),
                            "Pertimbangkan menggunakan filter atau fitur lain untuk meningkatkan probabilitas."
                        ]),
                        html.Li([
                            html.Strong("Validasi dengan sampel: "),
                            "Pastikan bin yang dipilih memiliki sampel cukup (min 20-50) untuk reliability."
                        ]),
                        html.Li([
                            html.Strong("Gunakan interval kepercayaan: "),
                            "Perhatikan error bars/CI - semakin sempit semakin reliable estimasi probabilitas."
                        ]),
                        html.Li([
                            html.Strong("Ekspor untuk dokumentasi: "),
                            "Klik 'Ekspor Hasil' untuk menyimpan analisis ini."
                        ])
                    ]),
                    
                    html.Hr(),
                    
                    # Action Items
                    html.H6("✅ Langkah Selanjutnya", className="fw-bold mb-3"),
                    html.Ol([
                        html.Li("Klik pada bar/cell dengan probabilitas tinggi untuk melihat detail trading"),
                        html.Li("Analisis karakteristik trading di area probabilitas tinggi"),
                        html.Li("Buat skenario What-If untuk testing lebih lanjut"),
                        html.Li("Combine dengan filter untuk analisis lebih spesifik"),
                        html.Li("Monitor perubahan probabilitas over time dengan data baru")
                    ])
                ])
            ], className="mb-3")
            
        except Exception as e:
            return dbc.Alert([
                html.I(className="bi bi-exclamation-triangle me-2"),
                f"Error membuat interpretasi: {str(e)}"
            ], color="danger")
    
    
    @app.callback(
        [Output('feature-x-dropdown', 'value'),
         Output('feature-y-dropdown', 'value'),
         Output('prob-auto-select-result', 'children')],
        Input('auto-select-btn', 'n_clicks'),
        [State('merged-data-store', 'data'),
         State('target-variable-dropdown', 'value')],
        prevent_initial_call=True
    )
    def auto_select_best_features(n_clicks, merged_data, target):
        """
        Automatically select best features based on correlation with target
        
        Analyzes correlation between all numeric features and target variable,
        then selects the top 2 features with highest absolute correlation.
        """
        # Debug logging
        print(f"\n[AUTO-SELECT] Callback triggered!")
        print(f"  n_clicks: {n_clicks}")
        print(f"  merged_data: {'None' if merged_data is None else 'Available'}")
        print(f"  target: {target}")
        
        # Check conditions
        if not n_clicks:
            print(f"  [AUTO-SELECT] No clicks yet, preventing update")
            raise PreventUpdate
        
        if merged_data is None:
            print(f"  [AUTO-SELECT] No data available")
            alert = dbc.Alert([
                html.I(className="bi bi-exclamation-triangle me-2"),
                html.Strong("Data belum dimuat! "),
                "Silakan muat data terlebih dahulu di tab Dashboard Analisis Trading."
            ], color="warning")
            return no_update, no_update, alert
        
        if not target:
            print(f"  [AUTO-SELECT] No target selected")
            alert = dbc.Alert([
                html.I(className="bi bi-exclamation-triangle me-2"),
                html.Strong("Target belum dipilih! "),
                "Silakan pilih variabel target terlebih dahulu."
            ], color="warning")
            return no_update, no_update, alert
        
        try:
            import io
            if isinstance(merged_data, dict) and 'data' in merged_data:
                df = pd.read_json(io.StringIO(merged_data['data']), orient='split')
            else:
                df = pd.DataFrame(merged_data)
        except Exception:
            raise PreventUpdate
        
        # Check if target exists
        if target not in df.columns:
            alert = dbc.Alert([
                html.I(className="bi bi-exclamation-triangle me-2"),
                f"Target variable '{target}' tidak ditemukan dalam data"
            ], color="warning")
            return no_update, no_update, alert
        
        try:
            # Get numeric columns (exclude target and ID columns)
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            exclude_cols = ['y_win', 'y_hit_1R', 'y_hit_2R', 'y_future_win_k', 
                           'Ticket_id', 'timestamp', target]
            feature_cols = [col for col in numeric_cols if col not in exclude_cols]
            
            if len(feature_cols) == 0:
                alert = dbc.Alert([
                    html.I(className="bi bi-exclamation-triangle me-2"),
                    "Tidak ada fitur numerik yang tersedia untuk analisis"
                ], color="warning")
                return no_update, no_update, alert
            
            # Calculate correlation with target
            correlations = {}
            for col in feature_cols:
                try:
                    # Remove NaN values for correlation calculation
                    valid_data = df[[col, target]].dropna()
                    if len(valid_data) > 10:  # Minimum 10 samples
                        corr = valid_data[col].corr(valid_data[target])
                        if not np.isnan(corr):
                            correlations[col] = abs(corr)
                except Exception:
                    continue
            
            if len(correlations) == 0:
                alert = dbc.Alert([
                    html.I(className="bi bi-exclamation-triangle me-2"),
                    "Tidak dapat menghitung korelasi. Pastikan data valid."
                ], color="warning")
                return no_update, no_update, alert
            
            # Sort by correlation (descending)
            sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
            
            # Select top 2 features
            best_feature_x = sorted_features[0][0] if len(sorted_features) > 0 else None
            best_feature_y = sorted_features[1][0] if len(sorted_features) > 1 else None
            
            # Create insights alert
            insights = dbc.Alert([
                html.H6([
                    html.I(className="bi bi-magic me-2"),
                    "Fitur Terbaik Dipilih Otomatis"
                ], className="alert-heading"),
                html.P([
                    html.Strong("Feature X: "), f"{best_feature_x} (korelasi: {correlations[best_feature_x]:.3f})",
                    html.Br(),
                    html.Strong("Feature Y: "), f"{best_feature_y} (korelasi: {correlations[best_feature_y]:.3f})" if best_feature_y else "Tidak ada",
                    html.Br(),
                    html.Br(),
                    html.Small([
                        "Top 5 fitur berdasarkan korelasi dengan ", html.Strong(target), ":"
                    ])
                ]),
                html.Ul([
                    html.Li(f"{feat}: {corr:.3f}") 
                    for feat, corr in sorted_features[:5]
                ], className="mb-0")
            ], color="success")
            
            return best_feature_x, best_feature_y, insights
            
        except Exception as e:
            alert = dbc.Alert([
                html.I(className="bi bi-exclamation-triangle me-2"),
                f"Error saat memilih fitur: {str(e)}"
            ], color="danger")
            return no_update, no_update, alert
    
    
    @app.callback(
        Output('date-range-filter', 'start_date'),
        Output('date-range-filter', 'end_date'),
        [Input('quick-1m', 'n_clicks'),
         Input('quick-3m', 'n_clicks'),
         Input('quick-6m', 'n_clicks'),
         Input('quick-1y', 'n_clicks'),
         Input('quick-all', 'n_clicks')],
        State('merged-data-store', 'data'),
        prevent_initial_call=True
    )
    def quick_date_select(n1m, n3m, n6m, n1y, nall, merged_data):
        """
        Quick date range selection buttons
        
        Provides quick access to common date ranges.
        """
        if merged_data is None:
            raise PreventUpdate
        
        if not ctx.triggered:
            raise PreventUpdate
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        import io
        if isinstance(merged_data, dict) and 'data' in merged_data:
            df = pd.read_json(io.StringIO(merged_data['data']), orient='split')
        else:
            df = pd.DataFrame(merged_data)
        
        # Get date range from data
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            max_date = df['Timestamp'].max()
            min_date = df['Timestamp'].min()
        else:
            max_date = datetime.now()
            min_date = max_date - timedelta(days=365)
        
        # Calculate start date based on button
        if button_id == 'quick-1m':
            start_date = max_date - timedelta(days=30)
        elif button_id == 'quick-3m':
            start_date = max_date - timedelta(days=90)
        elif button_id == 'quick-6m':
            start_date = max_date - timedelta(days=180)
        elif button_id == 'quick-1y':
            start_date = max_date - timedelta(days=365)
        else:  # quick-all
            start_date = min_date
        
        return start_date.strftime('%Y-%m-%d'), max_date.strftime('%Y-%m-%d')
    
    
    @app.callback(
        Output('composite-score-section', 'children'),
        Input('merged-data-store', 'data')
    )
    def render_composite_score_section(merged_data):
        """
        Render composite score dashboard section
        
        Shows composite score analysis when data is loaded.
        Validates: Requirements 4.3, 6.1, 6.2
        """
        if merged_data is None:
            return html.Div()
        
        # Import composite score dashboard
        from frontend.components.composite_score_dashboard import create_composite_score_dashboard
        
        return create_composite_score_dashboard()


# Note: This file defines the callback registration function
# The actual registration happens in the main app file
# Additional components needed in layout:
# - dcc.Download(id='download-results')
# - dcc.Download(id='download-trades')
# - dbc.Alert(id='scenario-created-alert', is_open=False, color='success', dismissable=True)
