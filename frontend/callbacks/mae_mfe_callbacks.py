"""
MAE/MFE Analyzer Callbacks

This module implements all interactive callbacks for MAE/MFE analysis:
- MAE/MFE analysis on data load
- SL optimization
- TP optimization
- Pattern detection
- Visualization updates

Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6
"""

from dash import Input, Output, State, html
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io

from backend.calculators.mae_mfe_analyzer import (
    analyze_mae_patterns,
    analyze_mfe_patterns,
    calculate_profit_left,
    optimize_sl_level,
    optimize_tp_level
)
from frontend.utils.error_handlers import (
    validate_required_columns,
    log_error
)


def register_mae_mfe_callbacks(app):
    """
    Register all callbacks for MAE/MFE analyzer
    
    Parameters:
    -----------
    app : dash.Dash
        Dash application instance
    """
    
    @app.callback(
        [Output('mae-mfe-results-store', 'data'),
         Output('mae-mfe-scatter-plot', 'figure'),
         Output('mae-mfe-profit-left-display', 'children')],
        [Input('merged-data-store', 'data'),
         Input('main-tabs', 'active_tab')],
        prevent_initial_call=False
    )
    def analyze_mae_mfe_on_load(merged_data, active_tab):
        """
        Analyze MAE/MFE patterns when data is loaded or tab is switched
        
        Calls backend functions:
        - analyze_mae_patterns()
        - analyze_mfe_patterns()
        - calculate_profit_left()
        
        Stores results in mae-mfe-results-store
        Updates scatter plot and profit left display
        
        Validates: Requirements 2.1
        """
        # Only run when What-If Scenarios tab is active
        if active_tab != 'what-if-scenarios':
            raise PreventUpdate
        
        if merged_data is None:
            empty_fig = create_empty_scatter_plot()
            empty_display = create_no_data_message()
            return {}, empty_fig, empty_display
        
        try:
            # Load data (support both records and split JSON)
            if isinstance(merged_data, dict) and 'data' in merged_data:
                df = pd.read_json(io.StringIO(merged_data['data']), orient='split')
            else:
                df = pd.DataFrame(merged_data)
            
            # Validate required columns
            required_cols = ['MAE_R', 'MFE_R', 'R_multiple']
            is_valid, error_alert = validate_required_columns(df, required_cols)
            
            if not is_valid:
                empty_fig = create_empty_scatter_plot()
                return {}, empty_fig, error_alert
            
            # Call backend analyzers
            mae_patterns = analyze_mae_patterns(df)
            mfe_patterns = analyze_mfe_patterns(df)
            profit_left = calculate_profit_left(df)
            
            # Store results
            results = {
                'mae_patterns': mae_patterns,
                'mfe_patterns': mfe_patterns,
                'profit_left': profit_left
            }
            
            # Create scatter plot
            scatter_fig = create_mae_mfe_scatter(df)
            
            # Create profit left display
            profit_display = create_profit_left_cards(profit_left)
            
            return results, scatter_fig, profit_display
            
        except Exception as e:
            log_error(e, "MAE/MFE analysis", include_traceback=True)
            
            empty_fig = create_empty_scatter_plot()
            error_display = dbc.Alert([
                html.I(className="bi bi-exclamation-triangle me-2"),
                html.Strong("Error analyzing MAE/MFE: "),
                str(e)
            ], color="danger")
            return {}, empty_fig, error_display
    
    
    @app.callback(
        [Output('mae-mfe-sl-results', 'children'),
         Output('mae-mfe-sl-recommendation', 'children')],
        [Input('mae-mfe-sl-slider', 'value')],
        [State('merged-data-store', 'data')],
        prevent_initial_call=True
    )
    def optimize_sl(sl_level, merged_data):
        """
        Optimize stop loss level based on slider value
        
        Calls backend function:
        - optimize_sl_level()
        
        Updates results table and recommendation display
        Uses debouncing to avoid excessive calculations
        
        Validates: Requirements 2.3
        """
        if merged_data is None or sl_level is None:
            raise PreventUpdate
        
        try:
            # Load data
            if isinstance(merged_data, dict) and 'data' in merged_data:
                df = pd.read_json(io.StringIO(merged_data['data']), orient='split')
            else:
                df = pd.DataFrame(merged_data)
            
            # Validate required columns
            is_valid, error_alert = validate_required_columns(df, ['MAE_R', 'R_multiple'])
            if not is_valid:
                return error_alert, ""
            
            # Call backend optimizer with single SL level
            results_df = optimize_sl_level(df, sl_levels=[sl_level])
            
            if results_df.empty:
                return dbc.Alert("No data available for optimization", color="warning"), ""
            
            # Get results for this SL level
            result = results_df.iloc[0]
            
            # Create results table
            results_table = dbc.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Metric"),
                        html.Th("Value")
                    ])
                ]),
                html.Tbody([
                    html.Tr([html.Td("SL Level"), html.Td(f"{result['sl_level']:.2f}R")]),
                    html.Tr([html.Td("Winners Stopped"), html.Td(f"{result['pct_winners_stopped']:.1f}%")]),
                    html.Tr([html.Td("Losers Avoided"), html.Td(f"{result['pct_losers_avoided']:.1f}%")]),
                    html.Tr([html.Td("Net Benefit"), html.Td(f"{result['net_benefit']:.1f}%")]),
                    html.Tr([html.Td("New Expectancy"), html.Td(f"{result['new_expectancy']:.3f}R")])
                ])
            ], bordered=True, striped=True, hover=True, size='sm')
            
            # Create recommendation
            recommendation = ""
            if result['net_benefit'] > 0:
                recommendation = dbc.Alert([
                    html.H5("✓ Recommendation", className="alert-heading"),
                    html.P(f"Using SL at {result['sl_level']:.2f}R shows positive net benefit of {result['net_benefit']:.1f}%"),
                    html.P(f"This would avoid {result['losers_avoided_count']} losers while stopping {result['winners_stopped_count']} winners")
                ], color="success")
            elif result['net_benefit'] < -5:
                recommendation = dbc.Alert([
                    html.H5("⚠ Warning", className="alert-heading"),
                    html.P(f"Using SL at {result['sl_level']:.2f}R shows negative net benefit of {result['net_benefit']:.1f}%"),
                    html.P("This SL level may stop too many winners")
                ], color="warning")
            
            return results_table, recommendation
            
        except Exception as e:
            log_error(e, "SL optimization")
            return dbc.Alert([
                html.I(className="bi bi-exclamation-triangle me-2"),
                html.Strong("Error optimizing SL: "),
                str(e)
            ], color="danger"), ""
    
    
    @app.callback(
        [Output('mae-mfe-tp-results', 'children'),
         Output('mae-mfe-tp-recommendation', 'children')],
        [Input('mae-mfe-tp-slider', 'value')],
        [State('merged-data-store', 'data')],
        prevent_initial_call=True
    )
    def optimize_tp(tp_level, merged_data):
        """
        Optimize take profit level based on slider value
        
        Calls backend function:
        - optimize_tp_level()
        
        Updates results table with MFE capture percentage
        Display recommendation if expectancy improves
        Uses debouncing to avoid excessive calculations
        
        Validates: Requirements 2.4
        """
        if merged_data is None or tp_level is None:
            raise PreventUpdate
        
        try:
            # Load data
            if isinstance(merged_data, dict) and 'data' in merged_data:
                df = pd.read_json(io.StringIO(merged_data['data']), orient='split')
            else:
                df = pd.DataFrame(merged_data)
            
            # Validate required columns
            is_valid, error_alert = validate_required_columns(df, ['MFE_R', 'R_multiple'])
            if not is_valid:
                return error_alert, ""
            
            # Call backend optimizer with single TP level
            results_df = optimize_tp_level(df, tp_levels=[tp_level])
            
            if results_df.empty:
                return dbc.Alert("No data available for optimization", color="warning"), ""
            
            # Get results for this TP level
            result = results_df.iloc[0]
            
            # Create results table
            results_table = dbc.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Metric"),
                        html.Th("Value")
                    ])
                ]),
                html.Tbody([
                    html.Tr([html.Td("TP Level"), html.Td(f"{result['tp_level']:.2f}R")]),
                    html.Tr([html.Td("Trades Hit TP"), html.Td(f"{result['pct_trades_hit_tp']:.1f}%")]),
                    html.Tr([html.Td("Trades Count"), html.Td(f"{result['trades_hit_tp_count']}")]),
                    html.Tr([html.Td("MFE Capture %"), html.Td(f"{result['avg_mfe_capture_pct']:.1f}%")]),
                    html.Tr([html.Td("New Expectancy"), html.Td(f"{result['new_expectancy']:.3f}R")]),
                    html.Tr([html.Td("Expectancy Change"), html.Td(f"{result['expectancy_change']:.3f}R")])
                ])
            ], bordered=True, striped=True, hover=True, size='sm')
            
            # Create recommendation
            recommendation = ""
            if result['expectancy_change'] > 0:
                recommendation = dbc.Alert([
                    html.H5("✓ Recommendation", className="alert-heading"),
                    html.P(f"Using TP at {result['tp_level']:.2f}R improves expectancy by {result['expectancy_change']:.3f}R"),
                    html.P(f"This TP level would be hit by {result['trades_hit_tp_count']} trades ({result['pct_trades_hit_tp']:.1f}%)"),
                    html.P(f"Average MFE capture: {result['avg_mfe_capture_pct']:.1f}%")
                ], color="success")
            elif result['expectancy_change'] < -0.05:
                recommendation = dbc.Alert([
                    html.H5("⚠ Warning", className="alert-heading"),
                    html.P(f"Using TP at {result['tp_level']:.2f}R decreases expectancy by {abs(result['expectancy_change']):.3f}R"),
                    html.P("This TP level may be too conservative and cut winners short")
                ], color="warning")
            else:
                recommendation = dbc.Alert([
                    html.H5("ℹ Neutral", className="alert-heading"),
                    html.P(f"Using TP at {result['tp_level']:.2f}R has minimal impact on expectancy"),
                    html.P(f"MFE capture rate: {result['avg_mfe_capture_pct']:.1f}%")
                ], color="info")
            
            return results_table, recommendation
            
        except Exception as e:
            log_error(e, "TP optimization")
            return dbc.Alert([
                html.I(className="bi bi-exclamation-triangle me-2"),
                html.Strong("Error optimizing TP: "),
                str(e)
            ], color="danger"), ""
    
    
    @app.callback(
        Output('mae-mfe-pattern-alerts', 'children'),
        [Input('mae-mfe-results-store', 'data')],
        prevent_initial_call=True
    )
    def detect_patterns(results_data):
        """
        Detect MAE/MFE patterns and display alerts
        
        Checks for:
        - High MAE winners pattern (mean_mae > 0.5R)
        - Early exit pattern (mean_profit_left > 1.0R)
        
        Displays alerts with SL/TP suggestions
        
        Validates: Requirements 2.5, 2.6, 10.2
        """
        if not results_data:
            raise PreventUpdate
        
        alerts = []
        
        try:
            # Check for high MAE winners pattern
            mae_patterns = results_data.get('mae_patterns', {})
            winners = mae_patterns.get('winners', {})
            mean_mae = winners.get('mean_mae', 0)
            
            if not np.isnan(mean_mae) and mean_mae > 0.5:
                alert = dbc.Alert([
                    html.H5("⚠ High MAE on Winners Detected", className="alert-heading"),
                    html.P(f"Your winning trades have an average MAE of {mean_mae:.2f}R"),
                    html.P("This suggests your winners are experiencing significant drawdown before closing profitably."),
                    html.Hr(),
                    html.P([
                        html.Strong("Suggestion: "),
                        f"Consider using a wider stop loss (> {mean_mae:.2f}R) to avoid getting stopped out on trades that would eventually win."
                    ], className="mb-0")
                ], color="warning", className="mb-3")
                alerts.append(alert)
            
            # Check for early exit pattern
            profit_left = results_data.get('profit_left', {})
            mean_profit_left = profit_left.get('mean_profit_left', 0)
            
            if not np.isnan(mean_profit_left) and mean_profit_left > 1.0:
                alert = dbc.Alert([
                    html.H5("💡 Early Exit Pattern Detected", className="alert-heading"),
                    html.P(f"You're leaving an average of {mean_profit_left:.2f}R on the table"),
                    html.P("Your trades are reaching higher profit levels but you're exiting too early."),
                    html.Hr(),
                    html.P([
                        html.Strong("Suggestion: "),
                        "Consider using trailing stops or higher take profit targets to capture more of the move. ",
                        f"Try setting TP at least {mean_profit_left + 1.0:.1f}R or higher."
                    ], className="mb-0")
                ], color="info", className="mb-3")
                alerts.append(alert)
            
            # If no patterns detected, show positive message
            if not alerts:
                alert = dbc.Alert([
                    html.H5("✓ No Critical Patterns Detected", className="alert-heading"),
                    html.P("Your MAE/MFE patterns look reasonable. Continue monitoring as you gather more data.")
                ], color="success", className="mb-3")
                alerts.append(alert)
            
            return html.Div(alerts)
            
        except Exception as e:
            log_error(e, "Pattern detection")
            return dbc.Alert([
                html.I(className="bi bi-exclamation-triangle me-2"),
                html.Strong("Error detecting patterns: "),
                str(e)
            ], color="danger")



def create_empty_scatter_plot():
    """Create empty scatter plot with message"""
    fig = go.Figure()
    fig.add_annotation(
        text="No data available. Please load trade data first.",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14)
    )
    fig.update_layout(
        title="MAE vs MFE Scatter Plot",
        xaxis_title="MAE (R-multiples)",
        yaxis_title="MFE (R-multiples)",
        height=500
    )
    return fig


def create_no_data_message():
    """Create no data message"""
    return dbc.Alert(
        "No data available. Please load trade data first.",
        color="info"
    )





def create_mae_mfe_scatter(df):
    """
    Create MAE vs MFE scatter plot
    Color-coded by trade outcome (winner/loser)
    """
    # Separate winners and losers
    winners = df[df['R_multiple'] > 0]
    losers = df[df['R_multiple'] <= 0]
    
    fig = go.Figure()
    
    # Add winners
    if not winners.empty:
        fig.add_trace(go.Scatter(
            x=winners['MAE_R'],
            y=winners['MFE_R'],
            mode='markers',
            name='Winners',
            marker=dict(
                color='green',
                size=8,
                opacity=0.6,
                line=dict(width=1, color='darkgreen')
            ),
            text=[f"R: {r:.2f}" for r in winners['R_multiple']],
            hovertemplate='<b>Winner</b><br>MAE: %{x:.2f}R<br>MFE: %{y:.2f}R<br>%{text}<extra></extra>'
        ))
    
    # Add losers
    if not losers.empty:
        fig.add_trace(go.Scatter(
            x=losers['MAE_R'],
            y=losers['MFE_R'],
            mode='markers',
            name='Losers',
            marker=dict(
                color='red',
                size=8,
                opacity=0.6,
                line=dict(width=1, color='darkred')
            ),
            text=[f"R: {r:.2f}" for r in losers['R_multiple']],
            hovertemplate='<b>Loser</b><br>MAE: %{x:.2f}R<br>MFE: %{y:.2f}R<br>%{text}<extra></extra>'
        ))
    
    fig.update_layout(
        title="MAE vs MFE Analysis",
        xaxis_title="MAE - Maximum Adverse Excursion (R-multiples)",
        yaxis_title="MFE - Maximum Favorable Excursion (R-multiples)",
        height=500,
        hovermode='closest',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    return fig


def create_profit_left_cards(profit_left_data):
    """
    Create cards displaying profit left statistics
    """
    if not profit_left_data or profit_left_data.get('count', 0) == 0:
        return dbc.Alert("No winning trades available for profit left analysis", color="info")
    
    cards = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Mean Profit Left", className="card-title"),
                    html.H3(f"{profit_left_data['mean_profit_left']:.2f}R", className="text-primary"),
                    html.P(f"Median: {profit_left_data['median_profit_left']:.2f}R", className="text-muted")
                ])
            ])
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Capture Ratio", className="card-title"),
                    html.H3(f"{profit_left_data['capture_ratio']*100:.1f}%", className="text-success"),
                    html.P("Of maximum profit captured", className="text-muted")
                ])
            ])
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Trades Analyzed", className="card-title"),
                    html.H3(f"{profit_left_data['count']}", className="text-info"),
                    html.P("Winning trades", className="text-muted")
                ])
            ])
        ], width=4)
    ])
    
    # Add alert if profit left is high
    alert = None
    if profit_left_data['mean_profit_left'] > 1.0:
        alert = dbc.Alert([
            html.H5("⚠ High Profit Left Detected", className="alert-heading"),
            html.P(f"Average profit left on table: {profit_left_data['mean_profit_left']:.2f}R"),
            html.P("Consider using trailing stops or higher take profit targets to capture more of the move.")
        ], color="warning", className="mt-3")
    
    if alert:
        return html.Div([cards, alert])
    else:
        return cards
